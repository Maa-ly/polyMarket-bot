from __future__ import annotations

import argparse
import asyncio
import csv
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import logging
import os
import time
from typing import Any

from py_clob_client.clob_types import TradeParams
from py_clob_client.exceptions import PolyApiException

from arb_strategy import PlanResult, PlannedOrder, plan_market
from config import BotConfig, load_config, _parse_weights
from polymarket_api import (
    GammaClient,
    ClobClientWrapper,
    BookTop,
    book_top,
    is_updown_market,
    market_time_to_expiry,
    select_up_down_tokens,
    to_market_info,
)
from sizing import PositionSizer
from storage import Storage

logger = logging.getLogger(__name__)


class TradeLogger:
    def __init__(self, path: str) -> None:
        self.path = path
        self._ensure_header()

    def _ensure_header(self) -> None:
        expected = [
            "timestamp",
            "market_id",
            "market_slug",
            "token_id",
            "side",
            "price",
            "size",
            "tag",
            "order_id",
            "status",
        ]
        legacy = [
            "timestamp",
            "market_id",
            "token_id",
            "side",
            "price",
            "size",
            "tag",
            "order_id",
            "status",
        ]
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", newline="", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    header = next(reader, [])
                    if header == expected:
                        return
                    rows = list(reader)
            except OSError:
                header = []
                rows = []
            if header == legacy:
                with open(self.path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(expected)
                    for row in rows:
                        row = row + [""] * (len(legacy) - len(row))
                        writer.writerow(
                            [
                                row[0],
                                row[1],
                                "",
                                row[2],
                                row[3],
                                row[4],
                                row[5],
                                row[6],
                                row[7],
                                row[8],
                            ]
                        )
                return
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(expected)
            return
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(expected)

    def log_order(
        self,
        market_id: str,
        market_slug: str,
        token_id: str,
        side: str,
        price: float,
        size: float,
        tag: str,
        order_id: str | None,
        status: str,
    ) -> None:
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    datetime.now(timezone.utc).isoformat(),
                    market_id,
                    market_slug,
                    token_id,
                    side,
                    f"{price:.6f}",
                    f"{size:.6f}",
                    tag,
                    order_id or "",
                    status,
                ]
            )


class DecisionLogger:
    def __init__(self, path: str) -> None:
        self.path = path
        self._ensure_header()

    def _ensure_header(self) -> None:
        if os.path.exists(self.path):
            return
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "session_id",
                    "market_id",
                    "asset",
                    "ask_up",
                    "ask_down",
                    "bid_up",
                    "bid_down",
                    "sum_ask",
                    "allowed_bid_up",
                    "allowed_bid_down",
                    "hedged_budget",
                    "planned_core_shares",
                    "inventory_up",
                    "inventory_down",
                    "ratio",
                    "conservative_only",
                    "decision",
                    "reason",
                    "action_reason",
                ]
            )

    def log_decision(self, session_id: int | None, decision: dict[str, Any], reason: str) -> None:
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    datetime.now(timezone.utc).isoformat(),
                    session_id or "",
                    decision.get("market_id", ""),
                    decision.get("asset", ""),
                    decision.get("ask_up", ""),
                    decision.get("ask_down", ""),
                    decision.get("bid_up", ""),
                    decision.get("bid_down", ""),
                    decision.get("sum_ask", ""),
                    decision.get("allowed_bid_up", ""),
                    decision.get("allowed_bid_down", ""),
                    decision.get("hedged_budget", ""),
                    decision.get("planned_core_shares", ""),
                    decision.get("inventory_up", ""),
                    decision.get("inventory_down", ""),
                    decision.get("ratio", ""),
                    decision.get("conservative_only", False),
                    decision.get("decision", ""),
                    reason,
                    decision.get("action_reason", ""),
                ]
            )


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _get_field(order: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in order and order[key] not in (None, ""):
            return order[key]
    return None


def _parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_min_notional_error(exc: Exception) -> bool:
    if not isinstance(exc, PolyApiException):
        return False
    if getattr(exc, "status_code", None) != 400:
        return False
    msg = str(exc).lower()
    return "invalid amount" in msg and "min size" in msg


def _order_id(order: dict[str, Any]) -> str | None:
    val = _get_field(order, ("id", "orderID", "order_id"))
    return str(val) if val else None


def _order_token(order: dict[str, Any]) -> str | None:
    val = _get_field(order, ("asset_id", "assetId", "token_id", "tokenId"))
    return str(val) if val else None


def _order_side(order: dict[str, Any]) -> str | None:
    val = _get_field(order, ("side",))
    return str(val).upper() if val else None


def _order_price(order: dict[str, Any]) -> float | None:
    return _parse_float(_get_field(order, ("price", "orderPrice", "order_price")))


def _order_size(order: dict[str, Any]) -> float | None:
    return _parse_float(_get_field(order, ("size", "original_size", "originalSize", "remaining_size", "remainingSize")))


def _book_tick_size(book: Any) -> float:
    for key in ("tick_size", "tickSize", "min_tick", "minTick", "price_increment", "priceIncrement"):
        val = getattr(book, key, None)
        if val is None:
            continue
        try:
            return float(val)
        except (TypeError, ValueError):
            continue
    return 0.0


@dataclass
class SyncResult:
    placed: int
    cancelled: int


def sync_orders(
    clob: ClobClientWrapper,
    desired_orders: list[PlannedOrder],
    open_orders: list[dict[str, Any]],
    trade_logger: TradeLogger,
    storage: Storage,
    sizer: PositionSizer,
    dry_run: bool,
    market_id: str,
    market_slug: str,
    cancel_tokens: set[str] | None = None,
) -> SyncResult:
    cancel_tokens = cancel_tokens or set()
    desired = list(desired_orders)
    open_by_token_side: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for order in open_orders:
        token_id = _order_token(order)
        side = _order_side(order)
        if not token_id or not side:
            continue
        open_by_token_side.setdefault((token_id, side), []).append(order)

    matched: set[str] = set()
    placed = 0
    cancelled = 0

    # Cancel any orders in cancel_tokens outright.
    for order in open_orders:
        token_id = _order_token(order)
        if token_id and token_id in cancel_tokens:
            order_id = _order_id(order)
            if order_id and not dry_run:
                try:
                    clob.cancel(order_id)
                except Exception:
                    pass
            if order_id:
                sizer.ledger.release(order_id)
            trade_logger.log_order(
                market_id=market_id,
                market_slug=market_slug,
                token_id=token_id or "",
                side=_order_side(order) or "",
                price=_order_price(order) or 0.0,
                size=_order_size(order) or 0.0,
                tag="cancel",
                order_id=order_id,
                status="CANCELLED",
            )
            storage.record_order_action(order_id, market_id, token_id or "", _order_side(order) or "", _order_price(order) or 0.0, _order_size(order) or 0.0, "cancel", "CANCELLED")
            cancelled += 1

    # Match desired orders to open orders by exact price/size.
    remaining_desired: list[PlannedOrder] = []
    for d in desired:
        key = (d.token_id, d.side)
        candidates = open_by_token_side.get(key, [])
        keep = None
        for order in candidates:
            if _order_id(order) in matched:
                continue
            price = _order_price(order)
            size = _order_size(order)
            if price is None or size is None:
                continue
            if abs(price - d.price) <= 1e-9 and abs(size - d.size) <= 1e-6:
                keep = order
                break
        if keep is not None:
            order_id = _order_id(keep)
            if order_id:
                matched.add(order_id)
            continue
        remaining_desired.append(d)

    # Cancel unmatched open orders.
    for order in open_orders:
        order_id = _order_id(order)
        if not order_id or order_id in matched:
            continue
        token_id = _order_token(order)
        if token_id and token_id in cancel_tokens:
            continue
        if not dry_run:
            try:
                clob.cancel(order_id)
            except Exception:
                pass
        sizer.ledger.release(order_id)
        trade_logger.log_order(
            market_id=market_id,
            market_slug=market_slug,
            token_id=token_id or "",
            side=_order_side(order) or "",
            price=_order_price(order) or 0.0,
            size=_order_size(order) or 0.0,
            tag="cancel",
            order_id=order_id,
            status="CANCELLED",
        )
        storage.record_order_action(order_id, market_id, token_id or "", _order_side(order) or "", _order_price(order) or 0.0, _order_size(order) or 0.0, "cancel", "CANCELLED")
        cancelled += 1

    # Place missing orders.
    for d in remaining_desired:
        try:
            resp = clob.place_limit_order(
                d.token_id,
                d.price,
                d.size,
                d.side,
                d.order_type,
                dry_run,
            )
        except PolyApiException as exc:
            if "duplicated" in str(exc).lower():
                continue
            if _is_min_notional_error(exc):
                trade_logger.log_order(
                    market_id=market_id,
                    market_slug=market_slug,
                    token_id=d.token_id,
                    side=d.side,
                    price=d.price,
                    size=d.size,
                    tag=d.tag,
                    order_id=None,
                    status="REJECTED_MIN_NOTIONAL",
                )
                continue
            raise
        order_id = str(resp.get("orderID") or resp.get("order_id") or resp.get("id")) if isinstance(resp, dict) else None
        if order_id:
            sizer.ledger.reserve(order_id, d.price * d.size)
        trade_logger.log_order(
            market_id=market_id,
            market_slug=market_slug,
            token_id=d.token_id,
            side=d.side,
            price=d.price,
            size=d.size,
            tag=d.tag,
            order_id=order_id,
            status="DRY_RUN" if dry_run else "POSTED",
        )
        storage.record_order_action(order_id, market_id, d.token_id, d.side, d.price, d.size, d.tag, "POSTED")
        placed += 1

    return SyncResult(placed=placed, cancelled=cancelled)


def decision_label(action_reason: str, mode: str) -> str:
    if action_reason == "instant_capture":
        return "CAPTURE"
    if action_reason in {"immediate_hedge", "flatten_sell"}:
        return "REHEDGE"
    if mode == "grid":
        return "GRID"
    return "SKIP"


def fetch_new_trades(clob: ClobClientWrapper, since_ts: int) -> list[dict[str, Any]]:
    if clob.read_only:
        return []
    params = TradeParams()
    if since_ts > 0:
        params.after = since_ts
    return clob.client.get_trades(params)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Polymarket CLOB ARB bot")
    parser.add_argument("--config", help="Path to JSON/YAML config file")
    parser.add_argument("--mode", choices=["arb"], help="Strategy mode")
    parser.add_argument("--assets", nargs="*", help="Asset filters (BTC ETH SOL XRP)")
    parser.add_argument("--poll", type=int, help="Polling interval in seconds")
    parser.add_argument("--dry-run", action="store_true", help="Do not place orders")
    parser.add_argument("--simulate", action="store_true", help="Alias of --dry-run")
    parser.add_argument("--log-level", help="Log level")

    parser.add_argument("--instant-capture-enabled", action="store_true", help="Enable instant capture")
    parser.add_argument("--no-instant-capture-enabled", action="store_true", help="Disable instant capture")
    parser.add_argument("--instant-capture-edge-buffer", type=float, help="Instant capture edge buffer")
    parser.add_argument("--maker-edge-buffer", type=float, help="Maker edge buffer")
    parser.add_argument("--grid-levels", type=int, help="Grid levels per side")
    parser.add_argument("--grid-step", type=float, help="Grid price step")
    parser.add_argument("--per-level-size-shape", choices=["equal", "geometric"], help="Grid size shape")
    parser.add_argument("--geometric-ratio", type=float, help="Geometric size ratio")
    parser.add_argument("--order-refresh-seconds", type=int, help="Grid refresh cadence")
    parser.add_argument("--post-only", action="store_true", help="Post-only orders")
    parser.add_argument("--no-post-only", action="store_true", help="Disable post-only enforcement")
    parser.add_argument("--min-order-notional", type=float, help="Minimum order notional in USDC")

    parser.add_argument("--hedge-timeout-seconds", type=float, help="Hedge timeout seconds")
    parser.add_argument("--expiry-flush-seconds", type=int, help="Expiry flush window")
    parser.add_argument("--max-rehedge-market-budget", type=float, help="Max re-hedge budget per market")

    parser.add_argument("--session-budget-mode", choices=["fixed", "proportional", "ema_proportional"], help="Session budget mode")
    parser.add_argument("--fixed-session-budget", type=float, help="Fixed session budget in USDC")
    parser.add_argument("--session-fraction", type=float, help="Session fraction of bankroll")
    parser.add_argument("--max-session-budget", type=float, help="Max session budget in USDC")
    parser.add_argument("--max-market-budget", type=float, help="Max market budget in USDC")
    parser.add_argument("--asset-weights", help="Asset weights, e.g. BTC:0.55,ETH:0.24")
    parser.add_argument("--hedged-core-fraction", type=float, help="Hedged core fraction")
    parser.add_argument("--directional-fraction", type=float, help="Directional fraction")
    parser.add_argument("--max-inventory-skew", type=float, help="Soft max inventory skew ratio")
    parser.add_argument("--hard-skew-kill", type=float, help="Hard skew kill ratio")
    parser.add_argument("--reserve-buffer", type=float, help="Reserve buffer fraction")
    parser.add_argument("--max-open-orders", help="Max open orders allowed (number or 'auto')")
    parser.add_argument("--max-open-orders-mode", choices=["fixed", "auto"], help="Max open orders mode")
    parser.add_argument("--max-concurrent-sessions", type=int, help="Max concurrent sessions")
    parser.add_argument("--lock-time-seconds", type=int, help="Default lock time for fills")
    parser.add_argument("--bankroll-usdc", type=float, help="Manual bankroll override (USDC)")
    parser.add_argument("--bankroll-mode", choices=["api", "manual"], help="Bankroll source")
    parser.add_argument("--session-ema-alpha", type=float, help="EMA alpha for bankroll smoothing")

    parser.add_argument("--daily-max-spend", type=float, help="Daily max spend (USDC)")
    parser.add_argument("--daily-max-loss", type=float, help="Daily max loss (USDC)")
    parser.add_argument("--max-drawdown-fraction", type=float, help="Max drawdown fraction")
    parser.add_argument("--drawdown-risk-multiplier", type=float, help="Risk multiplier after drawdown")

    parser.add_argument("--conservative-only", action="store_true", help="Enable conservative-only mode")
    parser.add_argument("--no-conservative-only", action="store_true", help="Disable conservative-only mode")
    parser.add_argument("--target-ratio-low", type=float, help="Conservative target ratio low")
    parser.add_argument("--target-ratio-high", type=float, help="Conservative target ratio high")
    parser.add_argument("--absolute-max-ratio", type=float, help="Conservative absolute max ratio")
    parser.add_argument("--immediate-hedge-delay-seconds", type=float, help="Immediate hedge delay")
    parser.add_argument("--flatten-trigger-shares", type=float, help="Flatten trigger shares")
    parser.add_argument("--conservative-instant-edge-buffer", type=float, help="Conservative instant edge buffer")
    parser.add_argument("--conservative-maker-edge-buffer", type=float, help="Conservative maker edge buffer")
    parser.add_argument("--conservative-budget-multiplier", type=float, help="Conservative budget multiplier")
    parser.add_argument("--conservative-market-multiplier", type=float, help="Conservative market multiplier")

    return parser.parse_args()


def apply_cli_overrides(cfg: BotConfig, args: argparse.Namespace) -> BotConfig:
    if args.mode:
        cfg.mode = args.mode
    if args.assets:
        cfg.assets = [a.upper() for a in args.assets]
    if args.poll is not None:
        cfg.poll_seconds = args.poll
    if args.dry_run or args.simulate:
        cfg.dry_run = True
        cfg.simulate = True
    if args.log_level:
        cfg.log_level = args.log_level.upper()

    if args.instant_capture_enabled:
        cfg.instant_capture_enabled = True
    if args.no_instant_capture_enabled:
        cfg.instant_capture_enabled = False
    if args.instant_capture_edge_buffer is not None:
        cfg.instant_capture_edge_buffer = args.instant_capture_edge_buffer
    if args.maker_edge_buffer is not None:
        cfg.maker_edge_buffer = args.maker_edge_buffer
    if args.grid_levels is not None:
        cfg.grid_levels = args.grid_levels
    if args.grid_step is not None:
        cfg.grid_step = args.grid_step
    if args.per_level_size_shape:
        cfg.per_level_size_shape = args.per_level_size_shape
    if args.geometric_ratio is not None:
        cfg.geometric_ratio = args.geometric_ratio
    if args.order_refresh_seconds is not None:
        cfg.order_refresh_seconds = args.order_refresh_seconds
    if args.post_only:
        cfg.post_only = True
    if args.no_post_only:
        cfg.post_only = False
    if args.min_order_notional is not None:
        cfg.min_order_notional_usdc = args.min_order_notional
    if args.hedge_timeout_seconds is not None:
        cfg.hedge_timeout_seconds = args.hedge_timeout_seconds
    if args.expiry_flush_seconds is not None:
        cfg.expiry_flush_seconds = args.expiry_flush_seconds
    if args.max_rehedge_market_budget is not None:
        cfg.max_rehedge_market_budget_usdc = args.max_rehedge_market_budget

    if args.session_budget_mode:
        cfg.sizing.session_budget_mode = args.session_budget_mode
    if args.fixed_session_budget is not None:
        cfg.sizing.fixed_session_budget_usdc = args.fixed_session_budget
    if args.session_fraction is not None:
        cfg.sizing.session_fraction_of_bankroll = args.session_fraction
    if args.max_session_budget is not None:
        cfg.sizing.max_session_budget_usdc = args.max_session_budget
    if args.max_market_budget is not None:
        cfg.sizing.max_market_budget_usdc = args.max_market_budget
    if args.asset_weights:
        cfg.sizing.asset_weights = {str(k).upper(): float(v) for k, v in _parse_weights(args.asset_weights).items()}
    if args.hedged_core_fraction is not None:
        cfg.sizing.hedged_core_fraction = args.hedged_core_fraction
    if args.directional_fraction is not None:
        cfg.sizing.directional_fraction = args.directional_fraction
    if args.max_inventory_skew is not None:
        cfg.sizing.max_inventory_skew_ratio = args.max_inventory_skew
    if args.hard_skew_kill is not None:
        cfg.sizing.hard_skew_kill_ratio = args.hard_skew_kill
    if args.reserve_buffer is not None:
        cfg.sizing.reserve_buffer_fraction = args.reserve_buffer
    if args.max_open_orders_mode:
        cfg.sizing.max_open_orders_mode = args.max_open_orders_mode
    if args.max_open_orders is not None:
        raw = str(args.max_open_orders).strip().lower()
        if raw == "auto":
            cfg.sizing.max_open_orders_mode = "auto"
            cfg.sizing.max_open_orders = 0
        else:
            cfg.sizing.max_open_orders = int(float(raw))
    if args.max_concurrent_sessions is not None:
        cfg.sizing.max_concurrent_sessions = args.max_concurrent_sessions
    if args.lock_time_seconds is not None:
        cfg.sizing.lock_time_seconds_default = args.lock_time_seconds
    if args.bankroll_usdc is not None:
        cfg.sizing.bankroll_usdc = args.bankroll_usdc
    if args.bankroll_mode:
        cfg.sizing.bankroll_mode = args.bankroll_mode
    if args.session_ema_alpha is not None:
        cfg.sizing.session_ema_alpha = args.session_ema_alpha

    if args.daily_max_spend is not None:
        cfg.risk.daily_max_spend_usdc = args.daily_max_spend
    if args.daily_max_loss is not None:
        cfg.risk.daily_max_loss_usdc = args.daily_max_loss
    if args.max_drawdown_fraction is not None:
        cfg.risk.max_drawdown_fraction = args.max_drawdown_fraction
    if args.drawdown_risk_multiplier is not None:
        cfg.risk.drawdown_risk_multiplier = args.drawdown_risk_multiplier

    if args.conservative_only:
        cfg.conservative_only = True
    if args.no_conservative_only:
        cfg.conservative_only = False
    if args.target_ratio_low is not None:
        cfg.target_ratio_low = args.target_ratio_low
    if args.target_ratio_high is not None:
        cfg.target_ratio_high = args.target_ratio_high
    if args.absolute_max_ratio is not None:
        cfg.absolute_max_ratio = args.absolute_max_ratio
    if args.immediate_hedge_delay_seconds is not None:
        cfg.immediate_hedge_delay_seconds = args.immediate_hedge_delay_seconds
    if args.flatten_trigger_shares is not None:
        cfg.flatten_trigger_shares = args.flatten_trigger_shares
    if args.conservative_instant_edge_buffer is not None:
        cfg.conservative_instant_edge_buffer = args.conservative_instant_edge_buffer
    if args.conservative_maker_edge_buffer is not None:
        cfg.conservative_maker_edge_buffer = args.conservative_maker_edge_buffer
    if args.conservative_budget_multiplier is not None:
        cfg.conservative_budget_multiplier = args.conservative_budget_multiplier
    if args.conservative_market_multiplier is not None:
        cfg.conservative_market_multiplier = args.conservative_market_multiplier

    return cfg


async def run_loop(cfg: BotConfig) -> int:
    setup_logging(cfg.log_level)

    if cfg.simulate:
        cfg.dry_run = True

    if not cfg.poly_private_key:
        cfg.dry_run = True
        cfg.simulate = True
        logger.warning("POLY_PRIVATE_KEY missing; forcing dry-run")

    if cfg.mode != "arb":
        logger.warning("Unsupported mode %s; defaulting to arb", cfg.mode)
        cfg.mode = "arb"

    logger.info("Starting bot mode=%s poll=%ss assets=%s", cfg.mode, cfg.poll_seconds, ",".join(cfg.assets) or "ALL")

    gamma = GammaClient(cfg.gamma_base_url, order=cfg.gamma_order, ascending=cfg.gamma_ascending)
    clob = ClobClientWrapper(
        cfg.clob_host,
        private_key=cfg.poly_private_key,
        chain_id=cfg.poly_chain_id,
        signature_type=cfg.poly_signature_type,
        funder=cfg.poly_funder,
        read_only=cfg.dry_run,
    )

    storage = Storage(cfg.state_path)
    trade_logger = TradeLogger(cfg.trade_log_path)
    decision_logger = DecisionLogger(cfg.decision_log_path)

    sizer = PositionSizer(cfg.sizing, cfg.risk, cfg.updown_interval_minutes)
    if cfg.assets:
        sizer.set_active_assets(cfg.assets)
    else:
        sizer.set_active_assets(list(cfg.sizing.asset_weights.keys()))
    sizer.apply_conservative_overrides(cfg)
    sizer.risk.load_state(storage.load_risk_state())

    last_refresh: dict[str, int] = {}
    sem = asyncio.Semaphore(max(1, cfg.sizing.max_concurrent_sessions))

    try:
        while True:
            now = datetime.now(timezone.utc)
            now_ts = int(time.time())
            sizer.release_expired_locks(now_ts)

            if not clob.read_only:
                try:
                    balance_usdc, allowance_usdc = clob.get_collateral_balance_allowance()
                    available = None
                    if balance_usdc is not None and allowance_usdc is not None:
                        available = min(balance_usdc, allowance_usdc)
                    elif balance_usdc is not None:
                        available = balance_usdc
                    elif allowance_usdc is not None:
                        available = allowance_usdc
                    if available is not None and available <= 0:
                        if cfg.poly_signature_type != 0 and cfg.poly_funder:
                            logger.warning("USDC balance/allowance reported as 0 for proxy wallet; skipping preflight block")
                        else:
                            logger.warning("USDC balance/allowance is 0; skipping this cycle")
                            await asyncio.sleep(cfg.poll_seconds)
                            continue
                    sizer.update_bankroll(available, storage.state.get("bankroll_estimate"))
                    storage.update_bankroll_estimate(sizer.bankroll_estimate)
                except Exception as exc:
                    logger.warning("Failed to fetch balance/allowance: %s", exc)
                    await asyncio.sleep(cfg.poll_seconds)
                    continue
            else:
                sizer.update_bankroll(None, storage.state.get("bankroll_estimate"))

            end_date_max = None
            if cfg.updown_current_only:
                end_dt = now + timedelta(minutes=cfg.updown_interval_minutes)
                end_date_max = end_dt.isoformat().replace("+00:00", "Z")

            try:
                raw_markets = await gamma.iter_markets(
                    cfg.gamma_limit,
                    cfg.gamma_max_pages,
                    closed=False,
                    end_date_max=end_date_max,
                )
            except Exception as exc:
                logger.warning("Gamma API error: %s", exc)
                await asyncio.sleep(cfg.poll_seconds)
                continue

            markets: list[Any] = []
            current_window_seconds = cfg.updown_interval_minutes * 60 if cfg.updown_current_only else None
            for raw in raw_markets:
                market = to_market_info(raw)
                if not market:
                    continue
                if is_updown_market(market, cfg.assets, cfg.updown_interval_minutes):
                    if current_window_seconds is not None:
                        tte = market_time_to_expiry(market)
                        if tte is None or tte < 0 or tte > current_window_seconds:
                            continue
                    markets.append(market)
                if cfg.max_markets and len(markets) >= cfg.max_markets:
                    break

            logger.info("Found %d candidate markets", len(markets))

            open_order_list: list[dict[str, Any]] = []
            if not clob.read_only:
                try:
                    open_order_list = clob.get_open_orders()
                except Exception as exc:
                    logger.warning("Failed to fetch open orders: %s", exc)

            sizer.update_open_orders(open_order_list)
            storage.sync_open_orders(open_order_list)
            sizer.pending_reserve = 0.0
            sizer.apply_auto_max_open_orders(cfg, len(markets))

            if not clob.read_only:
                last_ts = storage.get_last_trade_ts()
                if last_ts <= 0:
                    bootstrap = now_ts - (cfg.updown_interval_minutes * 60 * 2)
                    storage.set_last_trade_ts(bootstrap)
                    last_ts = bootstrap
                try:
                    new_trades = fetch_new_trades(clob, last_ts)
                    if new_trades:
                        summary = storage.apply_trades(new_trades)
                        logger.debug("Applied %d new trades", summary.get("count", 0))
                        sizer.record_trade_summary(summary)
                except Exception as exc:
                    logger.warning("Failed to fetch trades: %s", exc)

            sizer.update_market_costs(storage.get_market_costs())

            async def process_market(market) -> None:
                async with sem:
                    await asyncio.to_thread(
                        process_market_sync,
                        market,
                        cfg,
                        clob,
                        open_order_list,
                        storage,
                        sizer,
                        trade_logger,
                        decision_logger,
                        last_refresh,
                    )

            tasks = [process_market(m) for m in markets]
            if tasks:
                await asyncio.gather(*tasks)

            storage.update_session_state(sizer.session_id, sizer.session_budget, sizer.session_spent)
            storage.update_risk_state(sizer.risk.snapshot())
            storage.save()

            await asyncio.sleep(cfg.poll_seconds)
    finally:
        await gamma.aclose()


def process_market_sync(
    market,
    cfg: BotConfig,
    clob: ClobClientWrapper,
    open_order_list: list[dict[str, Any]],
    storage: Storage,
    sizer: PositionSizer,
    trade_logger: TradeLogger,
    decision_logger: DecisionLogger,
    last_refresh: dict[str, int],
) -> None:
    market_id = market.condition_id or market.id
    market_slug = getattr(market, "slug", "") or ""
    tokens = select_up_down_tokens(market)
    if not tokens:
        return
    up_token = tokens["Up"]
    down_token = tokens["Down"]

    open_orders = [o for o in open_order_list if _order_token(o) in {up_token, down_token}]

    if storage.is_blocked(market_id, sizer.session_id):
        for order in open_orders:
            order_id = _order_id(order)
            if order_id and not cfg.dry_run:
                try:
                    clob.cancel(order_id)
                except Exception:
                    pass
            if order_id:
                sizer.ledger.release(order_id)
            trade_logger.log_order(
                market_id,
                market_slug,
                _order_token(order) or "",
                _order_side(order) or "",
                _order_price(order) or 0.0,
                _order_size(order) or 0.0,
                "cancel",
                order_id,
                "CANCELLED",
            )
            storage.record_order_action(order_id, market_id, _order_token(order) or "", _order_side(order) or "", _order_price(order) or 0.0, _order_size(order) or 0.0, "cancel", "CANCELLED")
        decision = {
            "market_id": market_id,
            "asset": market.asset,
            "ask_up": None,
            "ask_down": None,
            "bid_up": None,
            "bid_down": None,
            "sum_ask": None,
            "allowed_bid_up": None,
            "allowed_bid_down": None,
            "hedged_budget": None,
            "planned_core_shares": None,
            "inventory_up": None,
            "inventory_down": None,
            "ratio": None,
            "conservative_only": cfg.conservative_only,
            "action_reason": "skip_risk_limits",
            "decision": "SKIP",
        }
        decision_logger.log_decision(sizer.session_id, decision, "blocked_session")
        return

    def log_skip(reason: str) -> None:
        decision = {
            "market_id": market_id,
            "asset": market.asset,
            "ask_up": None,
            "ask_down": None,
            "bid_up": None,
            "bid_down": None,
            "sum_ask": None,
            "allowed_bid_up": None,
            "allowed_bid_down": None,
            "hedged_budget": None,
            "planned_core_shares": None,
            "inventory_up": None,
            "inventory_down": None,
            "ratio": None,
            "conservative_only": cfg.conservative_only,
            "action_reason": "skip_risk_limits",
            "decision": "SKIP",
        }
        decision_logger.log_decision(sizer.session_id, decision, reason)

    try:
        books = clob.get_order_books([up_token, down_token])
    except Exception as exc:
        logger.warning("Failed to fetch order books for %s: %s", market.slug, exc)
        log_skip("book_fetch_failed")
        return

    book_map: dict[str, BookTop] = {}
    for book in books or []:
        token_id = (
            getattr(book, "token_id", None)
            or getattr(book, "tokenId", None)
            or getattr(book, "asset_id", None)
            or getattr(book, "assetId", None)
        )
        if token_id:
            book_map[str(token_id)] = book_top(book)

    if up_token not in book_map or down_token not in book_map:
        log_skip("book_missing")
        return

    inventory = sizer.inventory.get_state(market.condition_id or market.id, up_token, down_token, sizer.market_costs)
    last_fill_ts = storage.get_last_fill_ts(market_id)

    now_ts = int(time.time())
    plan = plan_market(
        market,
        book_map[up_token],
        book_map[down_token],
        inventory,
        sizer,
        cfg,
        now_ts,
        last_fill_ts,
    )

    decision = plan.decision
    action_reason = decision.get("action_reason", "")
    decision["decision"] = decision_label(action_reason, plan.mode)

    if plan.mode == "grid" and cfg.order_refresh_seconds > 0:
        last = last_refresh.get(market_id, 0)
        if now_ts - last < cfg.order_refresh_seconds:
            decision["decision"] = "SKIP"
            decision["action_reason"] = "skip_risk_limits"
            decision_logger.log_decision(sizer.session_id, decision, "refresh_cooldown")
            return

    if plan.mode == "taker":
        cancel_tokens = {up_token, down_token}
        for order in open_orders:
            token_id = _order_token(order)
            if token_id and token_id not in cancel_tokens:
                continue
            order_id = _order_id(order)
            if order_id and not cfg.dry_run:
                try:
                    clob.cancel(order_id)
                except Exception:
                    pass
            if order_id:
                sizer.ledger.release(order_id)
            trade_logger.log_order(
                market_id,
                market_slug,
                token_id or "",
                _order_side(order) or "",
                _order_price(order) or 0.0,
                _order_size(order) or 0.0,
                "cancel",
                order_id,
                "CANCELLED",
            )
            storage.record_order_action(order_id, market_id, token_id or "", _order_side(order) or "", _order_price(order) or 0.0, _order_size(order) or 0.0, "cancel", "CANCELLED")
    elif plan.mode == "cancel":
        cancel_tokens = set(plan.cancel_tokens)
        for order in open_orders:
            token_id = _order_token(order)
            if cancel_tokens and token_id not in cancel_tokens:
                continue
            order_id = _order_id(order)
            if order_id and not cfg.dry_run:
                try:
                    clob.cancel(order_id)
                except Exception:
                    pass
            if order_id:
                sizer.ledger.release(order_id)
            trade_logger.log_order(
                market_id,
                market_slug,
                token_id or "",
                _order_side(order) or "",
                _order_price(order) or 0.0,
                _order_size(order) or 0.0,
                "cancel",
                order_id,
                "CANCELLED",
            )
            storage.record_order_action(order_id, market_id, token_id or "", _order_side(order) or "", _order_price(order) or 0.0, _order_size(order) or 0.0, "cancel", "CANCELLED")

    placed = 0
    if plan.mode == "taker":
        for order in plan.orders:
            try:
                resp = clob.place_limit_order(
                    order.token_id,
                    order.price,
                    order.size,
                    order.side,
                    order.order_type,
                    cfg.dry_run,
                )
            except PolyApiException as exc:
                if _is_min_notional_error(exc):
                    trade_logger.log_order(
                        market_id,
                        market_slug,
                        order.token_id,
                        order.side,
                        order.price,
                        order.size,
                        order.tag,
                        None,
                        "REJECTED_MIN_NOTIONAL",
                    )
                    continue
                raise
            order_id = str(resp.get("orderID") or resp.get("order_id") or resp.get("id")) if isinstance(resp, dict) else None
            trade_logger.log_order(
                market_id,
                market_slug,
                order.token_id,
                order.side,
                order.price,
                order.size,
                order.tag,
                order_id,
                "DRY_RUN" if cfg.dry_run else "POSTED",
            )
            storage.record_order_action(order_id, market_id, order.token_id, order.side, order.price, order.size, order.tag, "POSTED")
            placed += 1
    elif plan.mode == "grid":
        result = sync_orders(
            clob,
            plan.orders,
            open_orders,
            trade_logger,
            storage,
            sizer,
            cfg.dry_run,
            market_id,
            market_slug,
            cancel_tokens=set(plan.cancel_tokens),
        )
        placed += result.placed

    if placed and cfg.dry_run:
        total_cost = sum(o.price * o.size for o in plan.orders if str(o.side).upper() == "BUY")
        sizer.add_dry_run_lock(total_cost, now_ts)

    if plan.mode == "grid":
        last_refresh[market_id] = now_ts

    if plan.block_market:
        storage.block_market(market_id, sizer.session_id)

    decision_logger.log_decision(sizer.session_id, decision, decision.get("action_reason", ""))


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)
    asyncio.run(run_loop(cfg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
