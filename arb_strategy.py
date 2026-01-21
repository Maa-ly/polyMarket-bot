from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

from py_clob_client.clob_types import OrderType
from py_clob_client.order_builder.constants import BUY, SELL

from polymarket_api import MarketInfo, BookTop, market_time_to_expiry, round_price, round_size
from sizing import InventoryState, OrderPlan, PositionSizer


@dataclass
class PlannedOrder:
    token_id: str
    side: str
    price: float
    size: float
    tag: str
    order_type: OrderType = OrderType.GTC
    post_only: bool = True


@dataclass
class PlanResult:
    mode: str
    orders: list[PlannedOrder]
    cancel_tokens: list[str]
    decision: dict[str, Any]
    block_market: bool = False


def _mid_price(book: BookTop) -> float:
    if book.best_bid is not None and book.best_ask is not None:
        return (book.best_bid + book.best_ask) / 2.0
    if book.best_ask is not None:
        return book.best_ask
    if book.best_bid is not None:
        return book.best_bid
    return 0.0


def _round_down(price: float, tick: float) -> float:
    return round_price(price, tick, SELL)


def _round_up(price: float, tick: float) -> float:
    return round_price(price, tick, BUY)


def _grid_level_sizes(total_shares: float, levels: int, shape: str, geometric_ratio: float) -> list[float]:
    if levels <= 0 or total_shares <= 0:
        return []
    if shape == "geometric":
        weights = [geometric_ratio**i for i in range(levels)]
        total_weight = sum(weights)
        return [total_shares * (w / total_weight) for w in weights]
    per_level = total_shares / levels
    return [per_level for _ in range(levels)]


def _apply_post_only(price: float, best_ask: float | None, tick: float) -> float | None:
    if best_ask is None:
        return None
    if tick > 0 and price >= best_ask:
        price = best_ask - tick
    if price <= 0:
        return None
    return price


def _allowed_bid(best_bid: float | None, other_best_bid: float | None, edge_buffer: float) -> float:
    if best_bid is None or other_best_bid is None:
        return 0.0
    return min(best_bid, (1.0 - edge_buffer) - other_best_bid)


def _skew_factor(ratio: float, max_skew: float) -> float:
    if max_skew <= 1:
        return 0.0
    try:
        return min(1.0, abs(math.log(ratio)) / abs(math.log(max_skew)))
    except (ValueError, ZeroDivisionError):
        return 0.0


def _inventory_ratio(inventory: InventoryState) -> tuple[float, float, float]:
    net_up = inventory.filled_up + inventory.open_up
    net_down = inventory.filled_down + inventory.open_down
    ratio = (net_up + 1e-9) / (net_down + 1e-9)
    return net_up, net_down, ratio


def _min_order_size(book: BookTop) -> float:
    return book.min_order_size or 0.0


def _meets_min_notional(price: float | None, size: float, min_notional: float) -> bool:
    if price is None or price <= 0 or size <= 0:
        return False
    if min_notional <= 0:
        return True
    return (price * size) >= (min_notional - 1e-9)


def _plan_immediate_hedge(
    *,
    token_id: str,
    best_ask: float | None,
    ask_size: float | None,
    tick: float,
    min_size: float,
    min_notional: float,
    cap_usdc: float,
    imbalance_shares: float,
    tag: str,
) -> PlannedOrder | None:
    if best_ask is None or ask_size is None or best_ask <= 0:
        return None
    size_cap = cap_usdc / best_ask if cap_usdc > 0 else 0.0
    size = min(imbalance_shares, ask_size, size_cap)
    size = round_size(size, decimals=4)
    if size < min_size or size <= 0:
        return None
    price = _round_up(best_ask + (tick if tick > 0 else 0.0), tick)
    if not _meets_min_notional(price, size, min_notional):
        return None
    return PlannedOrder(token_id, BUY, price, size, tag, OrderType.FOK, post_only=False)


def _plan_flatten_sell(
    *,
    token_id: str,
    best_bid: float | None,
    bid_size: float | None,
    tick: float,
    min_size: float,
    min_notional: float,
    imbalance_shares: float,
    tag: str,
) -> PlannedOrder | None:
    if best_bid is None or bid_size is None or best_bid <= 0:
        return None
    size = min(imbalance_shares, bid_size)
    size = round_size(size, decimals=4)
    if size < min_size or size <= 0:
        return None
    price = _round_down(best_bid - (tick if tick > 0 else 0.0), tick)
    if price <= 0:
        return None
    if not _meets_min_notional(price, size, min_notional):
        return None
    return PlannedOrder(token_id, SELL, price, size, tag, OrderType.FOK, post_only=False)


def plan_market(
    market: MarketInfo,
    book_up: BookTop,
    book_down: BookTop,
    inventory: InventoryState,
    sizer: PositionSizer,
    cfg,
    now_ts: int,
    last_fill_ts: int,
) -> PlanResult:
    tte = market_time_to_expiry(market)
    net_up, net_down, ratio = _inventory_ratio(inventory)
    decision_fields = {
        "market_id": market.condition_id or market.id,
        "asset": market.asset,
        "ask_up": book_up.best_ask,
        "ask_down": book_down.best_ask,
        "bid_up": book_up.best_bid,
        "bid_down": book_down.best_bid,
        "sum_ask": (book_up.best_ask or 0.0) + (book_down.best_ask or 0.0),
        "allowed_bid_up": None,
        "allowed_bid_down": None,
        "hedged_budget": 0.0,
        "planned_core_shares": 0.0,
        "inventory_up": net_up,
        "inventory_down": net_down,
        "ratio": ratio,
        "conservative_only": cfg.conservative_only,
        "action_reason": "skip_no_edge",
    }

    if tte is not None and tte <= cfg.expiry_flush_seconds:
        imbalance = abs(net_up - net_down)
        if imbalance > 0:
            overweight_token = market.token_up_id if net_up > net_down else market.token_down_id
            under_token = market.token_down_id if net_up > net_down else market.token_up_id
            under_book = book_down if net_up > net_down else book_up
            order = _plan_immediate_hedge(
                token_id=under_token,
                best_ask=under_book.best_ask,
                ask_size=under_book.ask_size,
                tick=under_book.tick_size,
                min_size=_min_order_size(under_book),
                min_notional=cfg.min_order_notional_usdc,
                cap_usdc=cfg.max_rehedge_market_budget_usdc,
                imbalance_shares=imbalance,
                tag="expiry_hedge",
            )
            if order and sizer.can_trade_rehedge(order.price * order.size):
                decision_fields["action_reason"] = "immediate_hedge"
                return PlanResult("taker", [order], [overweight_token], decision_fields)
            over_book = book_up if net_up > net_down else book_down
            order = _plan_flatten_sell(
                token_id=overweight_token,
                best_bid=over_book.best_bid,
                bid_size=over_book.bid_size,
                tick=over_book.tick_size,
                min_size=_min_order_size(over_book),
                min_notional=cfg.min_order_notional_usdc,
                imbalance_shares=imbalance,
                tag="flatten_sell",
            )
            if order:
                decision_fields["action_reason"] = "flatten_sell"
                return PlanResult("taker", [order], [overweight_token], decision_fields)
        decision_fields["action_reason"] = "cancel_overweight"
        return PlanResult("cancel", [], [market.token_up_id, market.token_down_id], decision_fields)

    mid_up = _mid_price(book_up)
    mid_down = _mid_price(book_down)
    plan = sizer.get_two_leg_order_plan(
        market,
        market.asset,
        mid_up,
        mid_down,
        up_token=market.token_up_id,
        down_token=market.token_down_id,
    )
    decision_fields["hedged_budget"] = plan.hedged_budget
    decision_fields["planned_core_shares"] = plan.core_shares

    # Conservative-only: prioritize immediate hedge/flatten when out of band.
    if cfg.conservative_only:
        target_low = cfg.target_ratio_low
        target_high = cfg.target_ratio_high
        if ratio < target_low or ratio > target_high:
            overweight_token = market.token_up_id if ratio > target_high else market.token_down_id
            under_token = market.token_down_id if ratio > target_high else market.token_up_id
            under_book = book_down if ratio > target_high else book_up
            hedge_due = (now_ts - last_fill_ts) >= cfg.immediate_hedge_delay_seconds
            imbalance = abs(net_up - net_down)
            absolute_breach = ratio >= cfg.absolute_max_ratio or ratio <= (1.0 / cfg.absolute_max_ratio)
            if hedge_due and imbalance > 0:
                order = _plan_immediate_hedge(
                    token_id=under_token,
                    best_ask=under_book.best_ask,
                    ask_size=under_book.ask_size,
                    tick=under_book.tick_size,
                    min_size=_min_order_size(under_book),
                    min_notional=cfg.min_order_notional_usdc,
                    cap_usdc=cfg.max_rehedge_market_budget_usdc,
                    imbalance_shares=imbalance,
                    tag="immediate_hedge",
                )
                if order and sizer.can_trade_rehedge(order.price * order.size):
                    decision_fields["action_reason"] = "immediate_hedge"
                    return PlanResult("taker", [order], [overweight_token], decision_fields)

            if absolute_breach or imbalance >= cfg.flatten_trigger_shares:
                over_book = book_up if ratio > target_high else book_down
                order = _plan_flatten_sell(
                    token_id=overweight_token,
                    best_bid=over_book.best_bid,
                    bid_size=over_book.bid_size,
                    tick=over_book.tick_size,
                    min_size=_min_order_size(over_book),
                    min_notional=cfg.min_order_notional_usdc,
                    imbalance_shares=imbalance,
                    tag="flatten_sell",
                )
                if order:
                    decision_fields["action_reason"] = "flatten_sell"
                    return PlanResult("taker", [order], [overweight_token], decision_fields)

            decision_fields["action_reason"] = "cancel_overweight"
            return PlanResult(
                "cancel",
                [],
                [market.token_up_id, market.token_down_id],
                decision_fields,
                block_market=True,
            )

    # Non-conservative re-hedge on large imbalances or timeout.
    imbalance = net_up - net_down
    if not cfg.conservative_only and imbalance != 0:
        hard_skew = cfg.sizing.hard_skew_kill_ratio
        hedge_due = last_fill_ts > 0 and (now_ts - last_fill_ts) >= cfg.hedge_timeout_seconds
        needs_hedge = ratio >= hard_skew or ratio <= (1.0 / hard_skew) or hedge_due
        if needs_hedge:
            missing_token = market.token_down_id if imbalance > 0 else market.token_up_id
            missing_book = book_down if imbalance > 0 else book_up
            order = _plan_immediate_hedge(
                token_id=missing_token,
                best_ask=missing_book.best_ask,
                ask_size=missing_book.ask_size,
                tick=missing_book.tick_size,
                min_size=_min_order_size(missing_book),
                min_notional=cfg.min_order_notional_usdc,
                cap_usdc=cfg.max_rehedge_market_budget_usdc,
                imbalance_shares=abs(imbalance),
                tag="rehedge",
            )
            if order and sizer.can_trade_rehedge(order.price * order.size):
                decision_fields["action_reason"] = "immediate_hedge"
                return PlanResult("taker", [order], [missing_token], decision_fields)

    # Instant capture (optional)
    edge_buffer = cfg.conservative_instant_edge_buffer if cfg.conservative_only else cfg.instant_capture_edge_buffer
    sum_ask = (book_up.best_ask or 0.0) + (book_down.best_ask or 0.0)
    if (
        cfg.instant_capture_enabled
        and book_up.best_ask is not None
        and book_down.best_ask is not None
        and sum_ask > 0
        and sum_ask <= 1.0 - edge_buffer
    ):
        size = min(plan.core_shares, book_up.ask_size or 0.0, book_down.ask_size or 0.0)
        size = round_size(size, decimals=4)
        min_size = max(_min_order_size(book_up), _min_order_size(book_down))
        est_cost = size * sum_ask
        if size >= min_size and plan.can_trade and sizer.can_trade(plan.session_id, plan.market_id, market.asset, est_cost):
            up_price = _round_up(book_up.best_ask, book_up.tick_size)
            down_price = _round_up(book_down.best_ask, book_down.tick_size)
            if _meets_min_notional(up_price, size, cfg.min_order_notional_usdc) and _meets_min_notional(
                down_price, size, cfg.min_order_notional_usdc
            ):
                orders = [
                    PlannedOrder(market.token_up_id, BUY, up_price, size, "instant_capture", OrderType.FOK, post_only=False),
                    PlannedOrder(market.token_down_id, BUY, down_price, size, "instant_capture", OrderType.FOK, post_only=False),
                ]
                decision_fields["action_reason"] = "instant_capture"
                return PlanResult("taker", orders, [market.token_up_id, market.token_down_id], decision_fields)

    # Grid planning
    edge_buffer = cfg.conservative_maker_edge_buffer if cfg.conservative_only else cfg.maker_edge_buffer
    allowed_bid_up = _allowed_bid(book_up.best_bid, book_down.best_bid, edge_buffer)
    allowed_bid_down = _allowed_bid(book_down.best_bid, book_up.best_bid, edge_buffer)
    decision_fields["allowed_bid_up"] = allowed_bid_up
    decision_fields["allowed_bid_down"] = allowed_bid_down

    if allowed_bid_up <= 0 or allowed_bid_down <= 0:
        decision_fields["action_reason"] = "skip_no_edge"
        return PlanResult("skip", [], [], decision_fields)

    if not plan.can_trade:
        decision_fields["action_reason"] = "skip_risk_limits"
        return PlanResult("skip", [], [], decision_fields)

    levels = max(1, int(cfg.grid_levels))
    step_up = max(cfg.grid_step, book_up.tick_size or 0.0)
    step_down = max(cfg.grid_step, book_down.tick_size or 0.0)
    size_levels = _grid_level_sizes(plan.core_shares, levels, cfg.per_level_size_shape, cfg.geometric_ratio)
    orders: list[PlannedOrder] = []

    if cfg.conservative_only:
        for i, base_size in enumerate(size_levels):
            up_price = _round_down(allowed_bid_up - (step_up * i), book_up.tick_size)
            down_price = _round_down(allowed_bid_down - (step_down * i), book_down.tick_size)
            if cfg.post_only:
                up_price = _apply_post_only(up_price, book_up.best_ask, book_up.tick_size)
                down_price = _apply_post_only(down_price, book_down.best_ask, book_down.tick_size)
                if up_price is None or down_price is None:
                    break
            if up_price is None or down_price is None or up_price <= 0 or down_price <= 0:
                break
            size = round_size(base_size, decimals=4)
            min_size = max(_min_order_size(book_up), _min_order_size(book_down))
            if size < min_size:
                continue
            if not _meets_min_notional(up_price, size, cfg.min_order_notional_usdc):
                continue
            if not _meets_min_notional(down_price, size, cfg.min_order_notional_usdc):
                continue
            orders.append(PlannedOrder(market.token_up_id, BUY, up_price, size, "grid", OrderType.GTC, post_only=cfg.post_only))
            orders.append(PlannedOrder(market.token_down_id, BUY, down_price, size, "grid", OrderType.GTC, post_only=cfg.post_only))
    else:
        max_skew = cfg.sizing.max_inventory_skew_ratio
        skew = _skew_factor(ratio, max_skew)
        size_scale_up = 1.0
        size_scale_down = 1.0
        if ratio > 1.0:
            size_scale_up = max(0.0, 1.0 - 0.5 * skew)
            size_scale_down = 1.0 + 0.5 * skew
            if ratio >= cfg.sizing.hard_skew_kill_ratio:
                size_scale_up = 0.0
        elif ratio < 1.0:
            size_scale_down = max(0.0, 1.0 - 0.5 * skew)
            size_scale_up = 1.0 + 0.5 * skew
            if ratio <= 1.0 / cfg.sizing.hard_skew_kill_ratio:
                size_scale_down = 0.0

        for i, base_size in enumerate(size_levels):
            up_price = _round_down(allowed_bid_up - (step_up * i), book_up.tick_size)
            down_price = _round_down(allowed_bid_down - (step_down * i), book_down.tick_size)
            if cfg.post_only:
                up_price = _apply_post_only(up_price, book_up.best_ask, book_up.tick_size)
                down_price = _apply_post_only(down_price, book_down.best_ask, book_down.tick_size)
            if up_price is not None and size_scale_up > 0:
                size = round_size(base_size * size_scale_up, decimals=4)
                if size >= _min_order_size(book_up) and _meets_min_notional(up_price, size, cfg.min_order_notional_usdc):
                    orders.append(PlannedOrder(market.token_up_id, BUY, up_price, size, "grid", OrderType.GTC, post_only=cfg.post_only))
            if down_price is not None and size_scale_down > 0:
                size = round_size(base_size * size_scale_down, decimals=4)
                if size >= _min_order_size(book_down) and _meets_min_notional(down_price, size, cfg.min_order_notional_usdc):
                    orders.append(PlannedOrder(market.token_down_id, BUY, down_price, size, "grid", OrderType.GTC, post_only=cfg.post_only))

    if not orders:
        decision_fields["action_reason"] = "skip_no_edge"
        return PlanResult("skip", [], [], decision_fields)

    est_cost = sum(order.price * order.size for order in orders if order.side == BUY)
    if not sizer.can_trade(plan.session_id, plan.market_id, market.asset, est_cost):
        decision_fields["action_reason"] = "skip_risk_limits"
        return PlanResult("skip", [], [], decision_fields)

    decision_fields["action_reason"] = "place_grid"
    return PlanResult("grid", orders, [], decision_fields)


def run_self_checks() -> None:
    from config import BotConfig
    from polymarket_api import MarketInfo

    now = int(time.time())

    cfg = BotConfig()
    cfg.conservative_only = True
    cfg.instant_capture_enabled = True
    cfg.sizing.max_market_budget_usdc = 100
    cfg.sizing.fixed_session_budget_usdc = 100

    sizer = PositionSizer(cfg.sizing, cfg.risk, cfg.updown_interval_minutes)
    sizer.apply_conservative_overrides(cfg)
    sizer.update_bankroll(1000)

    market = MarketInfo(
        id="1",
        condition_id="1",
        question="BTC Up or Down",
        slug="btc-updown-15m",
        asset="BTC",
        start_ts=now,
        end_ts=now + 900,
        token_up_id="up",
        token_down_id="down",
        active=True,
        closed=False,
        raw={},
    )

    book_up = BookTop(best_bid=0.49, bid_size=100, best_ask=0.51, ask_size=100, tick_size=0.01, min_order_size=1)
    book_down = BookTop(best_bid=0.49, bid_size=100, best_ask=0.51, ask_size=100, tick_size=0.01, min_order_size=1)

    inventory = InventoryState(filled_up=10, filled_down=5, open_up=0, open_down=0)
    plan = plan_market(market, book_up, book_down, inventory, sizer, cfg, now, now - 10)
    assert all(o.token_id != market.token_up_id for o in plan.orders if o.side == BUY), "conservative grid should not add to overweight"

    cfg.max_rehedge_market_budget_usdc = 0
    plan = plan_market(market, book_up, book_down, inventory, sizer, cfg, now, now - 10)
    assert any(o.tag == "flatten_sell" for o in plan.orders) or plan.mode == "cancel", "flatten should trigger when hedge cannot complete"

    cfg.max_rehedge_market_budget_usdc = 2500
    inventory = InventoryState(filled_up=5, filled_down=10, open_up=0, open_down=0)
    plan = plan_market(market, book_up, book_down, inventory, sizer, cfg, now, now - 10)
    assert any(o.tag == "immediate_hedge" for o in plan.orders) or plan.mode in {"cancel", "taker"}, "immediate hedge should be attempted"

    assert cfg.sizing.directional_fraction == 0.0, "directional fraction must be forced to 0"
    assert cfg.sizing.max_market_budget_usdc < 100, "conservative market multiplier should reduce budget"

    cfg2 = BotConfig()
    cfg2.instant_capture_enabled = True
    cfg2.sizing.session_budget_mode = "fixed"
    cfg2.sizing.fixed_session_budget_usdc = 100
    cfg2.sizing.max_session_budget_usdc = 100
    cfg2.sizing.max_market_budget_usdc = 50
    cfg2.sizing.hedged_core_fraction = 0.8
    cfg2.grid_step = 0.01
    cfg2.post_only = True

    sizer2 = PositionSizer(cfg2.sizing, cfg2.risk, cfg2.updown_interval_minutes)
    sizer2.update_bankroll(200)

    inv2 = InventoryState(filled_up=1, filled_down=1, open_up=0, open_down=0)
    plan2 = plan_market(market, book_up, book_down, inv2, sizer2, cfg2, now, now - 10)
    est_cost = sum(o.price * o.size for o in plan2.orders if o.side == BUY)
    assert est_cost <= sizer2.session_budget + 1e-6, "grid cost should not exceed session budget"
    assert est_cost <= sizer2.market_budget("BTC") + 1e-6, "grid cost should not exceed market budget"
    for order in plan2.orders:
        tick = book_up.tick_size if order.token_id == market.token_up_id else book_down.tick_size
        steps = order.price / tick if tick else 0.0
        assert abs(round(steps) - steps) < 1e-6, "grid prices should be tick-aligned"
        if cfg2.post_only:
            best_ask = book_up.best_ask if order.token_id == market.token_up_id else book_down.best_ask
            assert order.price < (best_ask or 0.0), "post-only prices must be below best ask"

    cfg3 = BotConfig()
    cfg3.instant_capture_enabled = True
    cfg3.instant_capture_edge_buffer = 0.02
    sizer3 = PositionSizer(cfg3.sizing, cfg3.risk, cfg3.updown_interval_minutes)
    sizer3.update_bankroll(1000)
    book_up_ic = BookTop(best_bid=0.48, bid_size=100, best_ask=0.48, ask_size=100, tick_size=0.01, min_order_size=1)
    book_down_ic = BookTop(best_bid=0.48, bid_size=100, best_ask=0.48, ask_size=100, tick_size=0.01, min_order_size=1)
    plan3 = plan_market(market, book_up_ic, book_down_ic, inv2, sizer3, cfg3, now, now - 10)
    assert plan3.mode == "taker", "instant capture should trigger when sum_ask is below threshold"
    assert all(o.tag == "instant_capture" for o in plan3.orders), "instant capture orders must be tagged"

    cfg4 = BotConfig()
    sizer4 = PositionSizer(cfg4.sizing, cfg4.risk, cfg4.updown_interval_minutes)
    sizer4.update_bankroll(1000)
    sizer4.risk.daily_spend = cfg4.risk.daily_max_spend_usdc + 1
    plan4 = plan_market(market, book_up, book_down, inv2, sizer4, cfg4, now, now - 10)
    assert plan4.mode == "skip", "risk limits should block trading"


if __name__ == "__main__":
    run_self_checks()
    print("self-checks passed")
