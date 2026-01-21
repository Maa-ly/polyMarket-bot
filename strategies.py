from __future__ import annotations

from dataclasses import dataclass
import json
import math
import logging
import time
from typing import Protocol

from py_clob_client.clob_types import OrderType
from py_clob_client.order_builder.constants import BUY, SELL

from polymarket_api import (
    ClobClientWrapper,
    MarketInfo,
    best_level,
    extract_order_id,
    market_asset_symbol,
    market_time_to_expiry,
    parse_order_fill,
    round_price,
    round_size,
    select_up_down_tokens,
)
from sizing import InventoryState, OrderPlan, PositionSizer

logger = logging.getLogger(__name__)


class BudgetManager(Protocol):
    def can_spend(self, amount_usdc: float) -> bool:
        ...

    def record_spend(self, amount_usdc: float) -> None:
        ...


class TradeLogger(Protocol):
    def log_order(
        self,
        market_id: str,
        market_slug: str,
        token_id: str,
        side: str,
        price: float,
        size: float,
        strategy: str,
        order_id: str | None,
        status: str,
    ) -> None:
        ...


@dataclass
class StrategyDecision:
    strategy: str
    reason: str
    placed: bool


@dataclass
class QuoteOrder:
    market_id: str
    market_slug: str
    token_id: str
    side: str
    price: float
    size: float
    label: str
    order_type: OrderType = OrderType.GTC


@dataclass
class QuotePlan:
    mode: str
    reason: str
    orders: list[QuoteOrder]


def _marketable_price(best_ask: float, tick_size: float, tick_buffer: int) -> float:
    return round_price(best_ask + tick_size * tick_buffer, tick_size, BUY)


def _marketable_price_sell(best_bid: float, tick_size: float, tick_buffer: int) -> float:
    return round_price(best_bid - tick_size * tick_buffer, tick_size, SELL)


def _min_order_size(book) -> float:
    for key in ("min_order_size", "minOrderSize", "min_size", "minSize"):
        val = getattr(book, key, None)
        if val is None:
            continue
        try:
            return float(val)
        except (TypeError, ValueError):
            continue
    return 0.0


def _tick_size(book) -> float:
    for key in ("tick_size", "tickSize", "min_tick", "minTick", "price_increment", "priceIncrement"):
        val = getattr(book, key, None)
        if val is None:
            continue
        try:
            return float(val)
        except (TypeError, ValueError):
            continue
    return 0.0


def _filled_size(order: dict | None) -> float:
    filled, size, _status = parse_order_fill(order)
    if filled is None:
        return 0.0
    if size is not None and filled > size:
        return size
    return filled


def _is_filled(order: dict | None) -> bool:
    filled, size, status = parse_order_fill(order)
    if status and str(status).upper() in {"FILLED", "CLOSED", "MATCHED", "DONE"}:
        return True
    if filled is not None and size is not None and filled >= size:
        return True
    return False


def two_leg_complete_set(
    market: MarketInfo,
    clob: ClobClientWrapper,
    cfg,
    budget: BudgetManager,
    trade_logger: TradeLogger,
    dry_run: bool,
    sizer: PositionSizer | None = None,
) -> StrategyDecision:
    arb_min_tte_seconds = getattr(cfg, "arb_min_tte_seconds", getattr(cfg, "mm_min_tte_seconds", 0))
    edge_buffer = getattr(cfg, "edge_buffer", cfg.mm_edge_buffer)
    min_liquidity_shares = getattr(cfg, "min_liquidity_shares", cfg.mm_min_liquidity_shares)
    maker_mode = getattr(cfg, "maker_mode", False)
    cross_tick_buffer = getattr(cfg, "cross_tick_buffer", 1)
    max_sum_ask = getattr(cfg, "max_sum_ask", None)

    tte = market_time_to_expiry(market)
    if tte is not None and arb_min_tte_seconds > 0 and tte <= arb_min_tte_seconds:
        return StrategyDecision("arb", "too_close_to_expiry", False)

    tokens = select_up_down_tokens(market)
    if not tokens:
        return StrategyDecision("arb", "missing_up_down_tokens", False)

    up_token = tokens["Up"]
    down_token = tokens["Down"]

    up_book = clob.get_order_book(up_token)
    down_book = clob.get_order_book(down_token)
    if up_book is None or down_book is None:
        return StrategyDecision("snipe", "no_orderbook", False)
    if up_book is None or down_book is None:
        return StrategyDecision("arb", "no_orderbook", False)

    if maker_mode:
        up_best = best_level(up_book.bids, SELL)
        down_best = best_level(down_book.bids, SELL)
    else:
        up_best = best_level(up_book.asks, BUY)
        down_best = best_level(down_book.asks, BUY)
    if not up_best or not down_best:
        return StrategyDecision("arb", "no_asks", False)

    up_price, up_size = up_best
    down_price, down_size = down_best
    sum_price = up_price + down_price

    edge = 1.0 - sum_price
    logger.debug(
        "ARB check %s up_token=%s down_token=%s price_up=%.4f price_down=%.4f sum=%.4f edge=%.4f maker=%s",
        market.slug,
        up_token,
        down_token,
        up_price,
        down_price,
        sum_price,
        edge,
        maker_mode,
    )

    if max_sum_ask is not None and sum_price > max_sum_ask:
        return StrategyDecision("arb", "sum_price_too_high", False)
    if sum_price > 1.0 - edge_buffer:
        return StrategyDecision("arb", "edge_too_small", False)

    min_liquidity = min(up_size, down_size)
    if min_liquidity < min_liquidity_shares:
        return StrategyDecision("arb", "insufficient_liquidity", False)

    min_order = max(_min_order_size(up_book), _min_order_size(down_book))
    size = None
    sizing_reason = None
    plan: OrderPlan | None = None
    inventory: InventoryState | None = None
    if sizer is not None:
        asset = market_asset_symbol(market)
        plan = sizer.get_two_leg_order_plan(market, asset, up_price, down_price, up_token=up_token, down_token=down_token)
        inventory = sizer.inventory.get_state(market.condition_id, up_token, down_token, sizer.market_costs)
        if not plan.can_trade:
            sizing_reason = plan.reason
        elif not sizer.can_trade(plan.session_id, market.condition_id, asset, plan.est_cost):
            sizing_reason = "risk_block"
        else:
            size = plan.core_shares
        if sizing_reason:
            fields = sizer.build_log_fields(plan, inventory, "skip", sizing_reason)
            logger.info("SIZER %s", json.dumps(fields, sort_keys=True))
            return StrategyDecision("arb", sizing_reason, False)
    if size is None:
        max_budget = cfg.sizing.max_market_budget_usdc if cfg.sizing.max_market_budget_usdc > 0 else float("inf")
        size = min(max_budget / sum_price, min_liquidity)
    size = min(size, min_liquidity)
    size = round_size(size)
    if size < min_order:
        return StrategyDecision("arb", "size_below_min", False)

    cost = size * sum_price
    if not budget.can_spend(cost):
        return StrategyDecision("arb", "daily_budget_exceeded", False)

    tick_up = _tick_size(up_book)
    tick_down = _tick_size(down_book)
    if maker_mode:
        price_up = round_price(up_price, tick_up, "SELL")
        price_down = round_price(down_price, tick_down, "SELL")
    else:
        price_up = _marketable_price(up_price, tick_up, cross_tick_buffer)
        price_down = _marketable_price(down_price, tick_down, cross_tick_buffer)

    logger.info(
        "ARB %s sum_price=%.4f size=%.4f price_up=%.4f price_down=%.4f",
        market.slug,
        sum_price,
        size,
        price_up,
        price_down,
    )
    if sizer is not None and plan is not None and inventory is not None:
        plan.core_shares = size
        plan.tilt_shares = 0.0
        fields = sizer.build_log_fields(plan, inventory, "trade", "orders_posted", cost)
        logger.info("SIZER %s", json.dumps(fields, sort_keys=True))

    resp_up = clob.place_limit_order(up_token, price_up, size, BUY, OrderType.GTC, dry_run)
    order_up_id = extract_order_id(resp_up)
    trade_logger.log_order(market.id, market.slug, up_token, BUY, price_up, size, "arb", order_up_id, "DRY_RUN" if dry_run else "POSTED")

    resp_down = clob.place_limit_order(down_token, price_down, size, BUY, OrderType.GTC, dry_run)
    order_down_id = extract_order_id(resp_down)
    trade_logger.log_order(market.id, market.slug, down_token, BUY, price_down, size, "arb", order_down_id, "DRY_RUN" if dry_run else "POSTED")

    if dry_run:
        return StrategyDecision("arb", "dry_run", True)
    budget.record_spend(cost)

    _hedge_partial_fill(
        clob,
        market,
        up_token,
        down_token,
        order_up_id,
        order_down_id,
        cfg,
        trade_logger,
        budget,
    )
    return StrategyDecision("arb", "orders_posted", True)


def _hedge_partial_fill(
    clob: ClobClientWrapper,
    market: MarketInfo,
    up_token: str,
    down_token: str,
    order_up_id: str | None,
    order_down_id: str | None,
    cfg,
    trade_logger: TradeLogger,
    budget: BudgetManager,
) -> None:
    if not order_up_id or not order_down_id:
        return

    deadline = time.time() + cfg.hedge_timeout_seconds
    order_up = None
    order_down = None

    while time.time() < deadline:
        order_up = clob.get_order(order_up_id)
        order_down = clob.get_order(order_down_id)
        if _is_filled(order_up) and _is_filled(order_down):
            return
        time.sleep(0.5)

    filled_up = _filled_size(order_up)
    filled_down = _filled_size(order_down)
    diff = filled_up - filled_down
    logger.info(
        "ARB hedge check %s filled_up=%.4f filled_down=%.4f diff=%.4f",
        market.slug,
        filled_up,
        filled_down,
        diff,
    )

    if abs(diff) < 1e-6:
        return

    if diff > 0:
        clob.cancel(order_down_id)
        hedged = _hedge_missing_leg(clob, market, down_token, diff, cfg, trade_logger, budget)
        if not hedged:
            _unwind_excess_leg(clob, market, up_token, diff, cfg, trade_logger)
    else:
        clob.cancel(order_up_id)
        hedged = _hedge_missing_leg(clob, market, up_token, abs(diff), cfg, trade_logger, budget)
        if not hedged:
            _unwind_excess_leg(clob, market, down_token, abs(diff), cfg, trade_logger)


def _hedge_missing_leg(
    clob: ClobClientWrapper,
    market: MarketInfo,
    token_id: str,
    size: float,
    cfg,
    trade_logger: TradeLogger,
    budget: BudgetManager,
) -> bool:
    book = clob.get_order_book(token_id)
    if book is None:
        return False
    best = best_level(book.asks, BUY)
    if not best:
        return False
    price, avail = best
    size = min(size, avail)
    size = round_size(size)
    min_order = _min_order_size(book)
    if size < min_order:
        return False

    tick = _tick_size(book)
    price = _marketable_price(price, tick, cfg.cross_tick_buffer)
    try:
        resp = clob.place_limit_order(token_id, price, size, BUY, OrderType.FOK, False)
    except Exception:
        return False
    order_id = extract_order_id(resp)
    trade_logger.log_order(market.id, market.slug, token_id, BUY, price, size, "arb_hedge", order_id, "POSTED")
    budget.record_spend(price * size)
    return True


def _unwind_excess_leg(
    clob: ClobClientWrapper,
    market: MarketInfo,
    token_id: str,
    size: float,
    cfg,
    trade_logger: TradeLogger,
) -> bool:
    book = clob.get_order_book(token_id)
    if book is None:
        return False
    best = best_level(book.bids, SELL)
    if not best:
        return False
    price, avail = best
    size = min(size, avail)
    size = round_size(size)
    min_order = _min_order_size(book)
    if size < min_order:
        return False

    tick = _tick_size(book)
    price = _marketable_price_sell(price, tick, cfg.cross_tick_buffer)
    try:
        resp = clob.place_limit_order(token_id, price, size, SELL, OrderType.FOK, False)
    except Exception:
        return False
    order_id = extract_order_id(resp)
    trade_logger.log_order(market.id, market.slug, token_id, SELL, price, size, "arb_unwind", order_id, "POSTED")
    return True


def snipe_one_sided(
    market: MarketInfo,
    clob: ClobClientWrapper,
    cfg,
    budget: BudgetManager,
    trade_logger: TradeLogger,
    dry_run: bool,
    oracle_prob_up: float | None,
    sizer: PositionSizer | None = None,
) -> StrategyDecision:
    snipe_prob_min = getattr(cfg, "snipe_prob_min", 0.98)
    snipe_price_min = getattr(cfg, "snipe_price_min", 0.95)
    snipe_min_edge = getattr(cfg, "snipe_min_edge", 0.01)
    snipe_window_seconds = getattr(cfg, "snipe_window_seconds", 900)
    snipe_budget_usdc = getattr(cfg, "snipe_budget_usdc", cfg.sizing.max_market_budget_usdc)
    min_liquidity_shares = getattr(cfg, "min_liquidity_shares", cfg.mm_min_liquidity_shares)
    cross_tick_buffer = getattr(cfg, "cross_tick_buffer", 1)
    tokens = select_up_down_tokens(market)
    if not tokens:
        return StrategyDecision("snipe", "missing_up_down_tokens", False)

    tte = market_time_to_expiry(market)
    if tte is None:
        return StrategyDecision("snipe", "missing_end_time", False)

    up_token = tokens["Up"]
    down_token = tokens["Down"]

    up_book = clob.get_order_book(up_token)
    down_book = clob.get_order_book(down_token)

    up_best = best_level(up_book.asks, BUY)
    down_best = best_level(down_book.asks, BUY)
    if not up_best or not down_best:
        return StrategyDecision("snipe", "no_asks", False)

    up_price, up_size = up_best
    down_price, down_size = down_best

    favored = None
    favored_price = None
    favored_size = None
    favored_book = None

    oracle_strong = False
    if oracle_prob_up is not None:
        if oracle_prob_up >= snipe_prob_min:
            favored = "Up"
            oracle_strong = True
        elif oracle_prob_up <= 1.0 - snipe_prob_min:
            favored = "Down"
            oracle_strong = True

    if favored is None:
        favored = "Up" if up_price >= down_price else "Down"

    if favored == "Up":
        favored_price, favored_size, favored_book = up_price, up_size, up_book
        favored_token = up_token
    else:
        favored_price, favored_size, favored_book = down_price, down_size, down_book
        favored_token = down_token

    logger.debug(
        "SNIPE check %s favored=%s price=%.4f tte=%.1f oracle=%s",
        market.slug,
        favored,
        favored_price,
        tte,
        oracle_prob_up if oracle_prob_up is not None else "none",
    )

    if favored_price < snipe_price_min:
        return StrategyDecision("snipe", "price_below_threshold", False)

    if (1.0 - favored_price) < snipe_min_edge:
        return StrategyDecision("snipe", "edge_too_small", False)

    if tte > snipe_window_seconds and not oracle_strong:
        return StrategyDecision("snipe", "window_not_reached", False)

    min_order = _min_order_size(favored_book)
    max_budget = snipe_budget_usdc
    if cfg.sizing.max_market_budget_usdc > 0:
        max_budget = min(max_budget, cfg.sizing.max_market_budget_usdc)

    if favored_size < min_liquidity_shares:
        return StrategyDecision("snipe", "insufficient_liquidity", False)

    size = min(max_budget / favored_price, favored_size)
    size = round_size(size)

    if size < min_order:
        return StrategyDecision("snipe", "size_below_min", False)

    cost = size * favored_price
    if not budget.can_spend(cost):
        return StrategyDecision("snipe", "daily_budget_exceeded", False)

    tick = _tick_size(favored_book)
    price = _marketable_price(favored_price, tick, cross_tick_buffer)

    logger.info(
        "SNIPE %s favored=%s price=%.4f size=%.4f tte=%.1f",
        market.slug,
        favored,
        price,
        size,
        tte,
    )

    resp = clob.place_limit_order(favored_token, price, size, BUY, OrderType.GTC, dry_run)
    order_id = extract_order_id(resp)
    trade_logger.log_order(market.id, market.slug, favored_token, BUY, price, size, "snipe", order_id, "DRY_RUN" if dry_run else "POSTED")
    if dry_run:
        return StrategyDecision("snipe", "dry_run", True)
    budget.record_spend(cost)

    return StrategyDecision("snipe", "order_posted", True)


def auto_route(
    market: MarketInfo,
    clob: ClobClientWrapper,
    cfg,
    budget: BudgetManager,
    trade_logger: TradeLogger,
    dry_run: bool,
    oracle_prob_up: float | None,
) -> StrategyDecision:
    tte = market_time_to_expiry(market)
    if tte is not None and tte <= cfg.snipe_window_seconds:
        decision = snipe_one_sided(market, clob, cfg, budget, trade_logger, dry_run, oracle_prob_up)
        if decision.placed:
            return decision
        return StrategyDecision("auto", "snipe_not_triggered", False)
    return two_leg_complete_set(market, clob, cfg, budget, trade_logger, dry_run)


def _maker_bid_price(
    best_bid: tuple[float, float] | None,
    best_ask: tuple[float, float] | None,
    tick_size: float,
    improve_ticks: int,
) -> float | None:
    if best_ask is None:
        return None
    ask_price, _ask_size = best_ask
    if tick_size <= 0:
        tick_size = 0.0
    if best_bid is None:
        offset = max(1, improve_ticks)
        price = ask_price - tick_size * offset
    else:
        bid_price, _bid_size = best_bid
        price = bid_price + tick_size * improve_ticks
    if tick_size > 0 and price >= ask_price:
        price = ask_price - tick_size
    if price <= 0:
        return None
    return round_price(price, tick_size, SELL)


def _apply_edge_limit(
    up_price: float,
    down_price: float,
    max_sum: float,
    reduce_up: bool,
) -> tuple[float, float]:
    if up_price + down_price <= max_sum:
        return up_price, down_price
    excess = (up_price + down_price) - max_sum
    if reduce_up:
        up_price = max(0.0, up_price - excess)
    else:
        down_price = max(0.0, down_price - excess)
    return up_price, down_price


def plan_maker_quotes(
    market: MarketInfo,
    clob: ClobClientWrapper,
    cfg,
    positions: dict[str, float],
    market_costs: dict[str, dict[str, dict[str, float]]] | None = None,
    sizer: PositionSizer | None = None,
) -> QuotePlan:
    tte = market_time_to_expiry(market)
    if tte is not None and cfg.mm_min_tte_seconds > 0 and tte <= cfg.mm_min_tte_seconds:
        return QuotePlan("maker", "too_close_to_expiry", [])

    tokens = select_up_down_tokens(market)
    if not tokens:
        return QuotePlan("maker", "missing_up_down_tokens", [])

    up_token = tokens["Up"]
    down_token = tokens["Down"]

    up_book = clob.get_order_book(up_token)
    down_book = clob.get_order_book(down_token)
    if up_book is None or down_book is None:
        return QuotePlan("maker", "no_orderbook", [])

    up_best_ask = best_level(up_book.asks, BUY)
    down_best_ask = best_level(down_book.asks, BUY)
    if not up_best_ask or not down_best_ask:
        return QuotePlan("maker", "no_asks", [])

    up_best_bid = best_level(up_book.bids, SELL)
    down_best_bid = best_level(down_book.bids, SELL)

    up_ask_price, up_ask_size = up_best_ask
    down_ask_price, down_ask_size = down_best_ask

    asset = market_asset_symbol(market)
    plan: OrderPlan | None = None
    inventory: InventoryState | None = None
    if sizer is not None:
        plan = sizer.get_two_leg_order_plan(market, asset, up_ask_price, down_ask_price, up_token=up_token, down_token=down_token)
        inventory = sizer.inventory.get_state(market.condition_id, up_token, down_token, sizer.market_costs)
        if not plan.can_trade:
            fields = sizer.build_log_fields(plan, inventory, "skip", plan.reason)
            logger.info("SIZER %s", json.dumps(fields, sort_keys=True))
            return QuotePlan("maker", plan.reason, [])

    min_liquidity = min(up_ask_size, down_ask_size)
    if min_liquidity < cfg.mm_min_liquidity_shares:
        if sizer is not None and plan is not None and inventory is not None:
            fields = sizer.build_log_fields(plan, inventory, "skip", "insufficient_liquidity")
            logger.info("SIZER %s", json.dumps(fields, sort_keys=True))
        return QuotePlan("maker", "insufficient_liquidity", [])

    tick_up = _tick_size(up_book)
    tick_down = _tick_size(down_book)

    sum_ask = up_ask_price + down_ask_price
    max_sum = 1.0 - cfg.mm_edge_buffer

    cross_tick_buffer = getattr(cfg, "cross_tick_buffer", 1)
    if sizer is not None and plan is not None and inventory is not None and plan.force_hedge_side:
        if plan.force_hedge_side == "Up":
            rehedge_token = up_token
            rehedge_book = up_book
            rehedge_price = up_ask_price
            rehedge_size = up_ask_size
            rehedge_tick = tick_up
        else:
            rehedge_token = down_token
            rehedge_book = down_book
            rehedge_price = down_ask_price
            rehedge_size = down_ask_size
            rehedge_tick = tick_down
        size = min(plan.rehedge_shares, rehedge_size)
        size = round_size(size, decimals=4)
        min_order = _min_order_size(rehedge_book)
        if size < min_order:
            fields = sizer.build_log_fields(plan, inventory, "skip", "rehedge_size_below_min")
            logger.info("SIZER %s", json.dumps(fields, sort_keys=True))
            return QuotePlan("maker", "rehedge_size_below_min", [])
        price = _marketable_price(rehedge_price, rehedge_tick, cross_tick_buffer)
        est_cost = price * size
        if not sizer.can_trade_rehedge(est_cost):
            fields = sizer.build_log_fields(plan, inventory, "skip", "risk_block", est_cost)
            logger.info("SIZER %s", json.dumps(fields, sort_keys=True))
            return QuotePlan("maker", "risk_block", [])
        fields = sizer.build_log_fields(plan, inventory, "trade", "rehedge", est_cost)
        logger.info("SIZER %s", json.dumps(fields, sort_keys=True))
        return QuotePlan(
            "taker",
            "rehedge",
            [
                QuoteOrder(
                    market.id,
                    market.slug,
                    rehedge_token,
                    BUY,
                    price,
                    size,
                    "mm_rehedge",
                    OrderType.FOK,
                )
            ],
        )

    if sum_ask <= max_sum:
        if sizer is not None and plan is not None and inventory is not None:
            size = min(plan.core_shares, up_ask_size, down_ask_size)
            size = round_size(size, decimals=4)
            min_order = max(_min_order_size(up_book), _min_order_size(down_book))
            if size < min_order:
                fields = sizer.build_log_fields(plan, inventory, "skip", "taker_size_below_min")
                logger.info("SIZER %s", json.dumps(fields, sort_keys=True))
                return QuotePlan("taker", "taker_size_below_min", [])
            est_cost = size * sum_ask
            if not sizer.can_trade(plan.session_id, plan.market_id, asset, est_cost):
                fields = sizer.build_log_fields(plan, inventory, "skip", "risk_block", est_cost)
                logger.info("SIZER %s", json.dumps(fields, sort_keys=True))
                return QuotePlan("taker", "risk_block", [])
            fields = sizer.build_log_fields(plan, inventory, "trade", "taker_discount", est_cost)
            logger.info("SIZER %s", json.dumps(fields, sort_keys=True))
            return QuotePlan(
                "taker",
                "taker_discount",
                [
                    QuoteOrder(market.id, market.slug, up_token, BUY, up_ask_price, size, "mm_taker", OrderType.FOK),
                    QuoteOrder(market.id, market.slug, down_token, BUY, down_ask_price, size, "mm_taker", OrderType.FOK),
                ],
            )
        max_total_budget = cfg.sizing.max_market_budget_usdc
        if max_total_budget <= 0:
            return QuotePlan("taker", "taker_budget_zero", [])
        min_order = max(_min_order_size(up_book), _min_order_size(down_book))
        size = min(max_total_budget / sum_ask, up_ask_size, down_ask_size)
        size = round_size(size, decimals=4)
        if size < min_order:
            return QuotePlan("taker", "taker_size_below_min", [])
        return QuotePlan(
            "taker",
            "taker_discount",
            [
                QuoteOrder(market.id, market.slug, up_token, BUY, up_ask_price, size, "mm_taker", OrderType.FOK),
                QuoteOrder(market.id, market.slug, down_token, BUY, down_ask_price, size, "mm_taker", OrderType.FOK),
            ],
        )

    up_bid = _maker_bid_price(up_best_bid, up_best_ask, tick_up, cfg.mm_quote_improve_ticks)
    down_bid = _maker_bid_price(down_best_bid, down_best_ask, tick_down, cfg.mm_quote_improve_ticks)
    if up_bid is None or down_bid is None:
        if sizer is not None and plan is not None and inventory is not None:
            fields = sizer.build_log_fields(plan, inventory, "skip", "no_bid_price")
            logger.info("SIZER %s", json.dumps(fields, sort_keys=True))
        return QuotePlan("maker", "no_bid_price", [])

    if sizer is not None and inventory is not None:
        up_pos = max(0.0, float(inventory.filled_up))
        down_pos = max(0.0, float(inventory.filled_down))
        ratio = inventory.net_ratio
    else:
        up_pos = max(0.0, float(positions.get(up_token, 0.0) or 0.0))
        down_pos = max(0.0, float(positions.get(down_token, 0.0) or 0.0))
        ratio = (up_pos + 1e-6) / (down_pos + 1e-6)

    skew_factor = 0.0
    max_skew = cfg.sizing.max_inventory_skew_ratio
    if max_skew and max_skew > 1:
        try:
            skew_factor = min(1.0, abs(math.log(ratio)) / abs(math.log(max_skew)))
        except (ValueError, ZeroDivisionError):
            skew_factor = 0.0

    shade_ticks = max(1, cfg.mm_quote_improve_ticks if cfg.mm_quote_improve_ticks > 0 else 1)
    if ratio > 1.0:
        up_bid -= tick_up * shade_ticks * skew_factor
        down_bid += tick_down * shade_ticks * skew_factor
    elif ratio < 1.0:
        up_bid += tick_up * shade_ticks * skew_factor
        down_bid -= tick_down * shade_ticks * skew_factor

    if tick_up > 0 and up_bid >= up_ask_price:
        up_bid = up_ask_price - tick_up
    if tick_down > 0 and down_bid >= down_ask_price:
        down_bid = down_ask_price - tick_down

    max_sum = 1.0 - cfg.mm_edge_buffer
    reduce_up = up_pos >= down_pos
    up_bid, down_bid = _apply_edge_limit(up_bid, down_bid, max_sum, reduce_up)

    up_bid = round_price(up_bid, tick_up, SELL)
    down_bid = round_price(down_bid, tick_down, SELL)
    if up_bid + down_bid > max_sum:
        if up_bid >= down_bid and tick_up > 0:
            up_bid = max(0.0, up_bid - tick_up)
        elif tick_down > 0:
            down_bid = max(0.0, down_bid - tick_down)

    if up_bid <= 0 or down_bid <= 0:
        if sizer is not None and plan is not None and inventory is not None:
            fields = sizer.build_log_fields(plan, inventory, "skip", "price_too_low")
            logger.info("SIZER %s", json.dumps(fields, sort_keys=True))
        return QuotePlan("maker", "price_too_low", [])

    if cfg.mm_recycle_mode.lower() == "sell":
        if up_best_bid and down_best_bid:
            bid_sum = up_best_bid[0] + down_best_bid[0]
            matched = min(up_pos, down_pos)
            if matched > cfg.mm_recycle_min_sets and bid_sum >= 1.0 - cfg.mm_recycle_edge_buffer:
                avg_ok = True
                if market_costs:
                    bucket = market_costs.get(market.condition_id, {})
                    up_cost = bucket.get(up_token, {}) if isinstance(bucket, dict) else {}
                    down_cost = bucket.get(down_token, {}) if isinstance(bucket, dict) else {}
                    up_avg = (up_cost.get("cost", 0.0) / up_cost.get("size", 1.0)) if up_cost.get("size") else None
                    down_avg = (down_cost.get("cost", 0.0) / down_cost.get("size", 1.0)) if down_cost.get("size") else None
                    if up_avg is not None and down_avg is not None:
                        avg_ok = (up_avg + down_avg) <= 1.0 - cfg.mm_edge_buffer
                if avg_ok:
                    size = min(matched, up_best_bid[1], down_best_bid[1])
                    size = round_size(size, decimals=4)
                    min_order = max(_min_order_size(up_book), _min_order_size(down_book))
                    if size >= min_order:
                        if sizer is not None and plan is not None and inventory is not None:
                            est_cost = size * (up_best_bid[0] + down_best_bid[0])
                            fields = sizer.build_log_fields(plan, inventory, "trade", "recycle_sell", est_cost)
                            logger.info("SIZER %s", json.dumps(fields, sort_keys=True))
                        return QuotePlan(
                            "taker",
                            "recycle_sell",
                            [
                                QuoteOrder(
                                    market.id,
                                    market.slug,
                                    up_token,
                                    SELL,
                                    up_best_bid[0],
                                    size,
                                    "mm_recycle",
                                    OrderType.FOK,
                                ),
                                QuoteOrder(
                                    market.id,
                                    market.slug,
                                    down_token,
                                    SELL,
                                    down_best_bid[0],
                                    size,
                                    "mm_recycle",
                                    OrderType.FOK,
                                ),
                            ],
                        )

    size_scale_up = 1.0
    size_scale_down = 1.0
    if ratio > 1.0:
        size_scale_up = max(0.0, 1.0 - 0.5 * skew_factor)
        size_scale_down = 1.0 + 0.5 * skew_factor
        if max_skew > 0 and ratio >= max_skew:
            size_scale_up = 0.0
    elif ratio < 1.0:
        size_scale_down = max(0.0, 1.0 - 0.5 * skew_factor)
        size_scale_up = 1.0 + 0.5 * skew_factor
        if max_skew > 0 and ratio <= 1.0 / max_skew:
            size_scale_down = 0.0

    orders: list[QuoteOrder] = []
    if sizer is not None and plan is not None and inventory is not None:
        base_size = plan.core_shares
        if size_scale_up > 0:
            size_up = round_size(base_size * size_scale_up, decimals=2)
            if size_up >= _min_order_size(up_book):
                orders.append(QuoteOrder(market.id, market.slug, up_token, BUY, up_bid, size_up, "mm_quote"))
        if size_scale_down > 0:
            size_down = round_size(base_size * size_scale_down, decimals=2)
            if size_down >= _min_order_size(down_book):
                orders.append(QuoteOrder(market.id, market.slug, down_token, BUY, down_bid, size_down, "mm_quote"))
    else:
        fallback_budget = cfg.sizing.max_market_budget_usdc * cfg.sizing.hedged_core_fraction
        if fallback_budget <= 0 or sum_ask <= 0:
            return QuotePlan("maker", "quote_budget_zero", [])
        base_size = fallback_budget / sum_ask
        if size_scale_up > 0:
            size_up = round_size(base_size * size_scale_up, decimals=2)
            if size_up >= _min_order_size(up_book):
                orders.append(QuoteOrder(market.id, market.slug, up_token, BUY, up_bid, size_up, "mm_quote"))
        if size_scale_down > 0:
            size_down = round_size(base_size * size_scale_down, decimals=2)
            if size_down >= _min_order_size(down_book):
                orders.append(QuoteOrder(market.id, market.slug, down_token, BUY, down_bid, size_down, "mm_quote"))

    if not orders:
        if sizer is not None and plan is not None and inventory is not None:
            fields = sizer.build_log_fields(plan, inventory, "skip", "size_below_min")
            logger.info("SIZER %s", json.dumps(fields, sort_keys=True))
        return QuotePlan("maker", "size_below_min", [])

    if sizer is not None and plan is not None and inventory is not None:
        est_cost = sum(o.price * o.size for o in orders if o.side.upper() == "BUY")
        if not sizer.can_trade(plan.session_id, plan.market_id, asset, est_cost):
            fields = sizer.build_log_fields(plan, inventory, "skip", "risk_block", est_cost)
            logger.info("SIZER %s", json.dumps(fields, sort_keys=True))
            return QuotePlan("maker", "risk_block", [])
        fields = sizer.build_log_fields(plan, inventory, "trade", "quotes_ready", est_cost)
        logger.info("SIZER %s", json.dumps(fields, sort_keys=True))

    logger.debug(
        "MM quote %s up_bid=%.4f down_bid=%.4f up_pos=%.4f down_pos=%.4f ratio=%.2f",
        market.slug,
        up_bid,
        down_bid,
        up_pos,
        down_pos,
        ratio,
    )
    return QuotePlan("maker", "quotes_ready", orders)
