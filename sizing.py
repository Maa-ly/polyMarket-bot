from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
import time
from typing import Any


@dataclass
class InventoryState:
    filled_up: float = 0.0
    filled_down: float = 0.0
    open_up: float = 0.0
    open_down: float = 0.0
    avg_fill_price_up: float | None = None
    avg_fill_price_down: float | None = None
    net_ratio: float = 1.0


@dataclass
class OrderPlan:
    session_id: int
    market_id: str
    asset: str
    market_budget: float
    hedged_budget: float
    directional_budget: float
    core_shares: float
    tilt_shares: float
    favored: str | None
    est_cost: float
    reason: str
    can_trade: bool
    net_ratio: float
    force_hedge_side: str | None
    rehedge_shares: float


class InventoryManager:
    def __init__(self) -> None:
        self.open_sizes_by_token: dict[str, float] = {}
        self.open_orders_count: int = 0
        self.reserved_open_usdc: float = 0.0

    def update_open_orders(self, open_orders: list[dict[str, Any]]) -> None:
        self.open_sizes_by_token = {}
        self.open_orders_count = 0
        reserved = 0.0
        for order in open_orders or []:
            token_id = _order_token(order)
            if not token_id:
                continue
            size = _order_remaining_size(order)
            price = _order_price(order)
            if size <= 0 or price <= 0:
                continue
            self.open_sizes_by_token[token_id] = self.open_sizes_by_token.get(token_id, 0.0) + size
            self.open_orders_count += 1
            reserved += price * size
        self.reserved_open_usdc = reserved

    def get_state(
        self,
        market_id: str,
        up_token: str,
        down_token: str,
        market_costs: dict[str, dict[str, dict[str, float]]],
    ) -> InventoryState:
        bucket = market_costs.get(market_id, {}) if isinstance(market_costs, dict) else {}
        up_entry = bucket.get(up_token, {}) if isinstance(bucket, dict) else {}
        down_entry = bucket.get(down_token, {}) if isinstance(bucket, dict) else {}
        filled_up = float(up_entry.get("size", 0.0) or 0.0)
        filled_down = float(down_entry.get("size", 0.0) or 0.0)
        avg_up = None
        avg_down = None
        if up_entry.get("size"):
            avg_up = float(up_entry.get("cost", 0.0) or 0.0) / float(up_entry.get("size", 1.0))
        if down_entry.get("size"):
            avg_down = float(down_entry.get("cost", 0.0) or 0.0) / float(down_entry.get("size", 1.0))
        open_up = float(self.open_sizes_by_token.get(up_token, 0.0))
        open_down = float(self.open_sizes_by_token.get(down_token, 0.0))
        ratio = (filled_up + open_up + 1e-9) / (filled_down + open_down + 1e-9)
        return InventoryState(
            filled_up=filled_up,
            filled_down=filled_down,
            open_up=open_up,
            open_down=open_down,
            avg_fill_price_up=avg_up,
            avg_fill_price_down=avg_down,
            net_ratio=ratio,
        )


class ReservationLedger:
    def __init__(self) -> None:
        self.reserved_by_order: dict[str, float] = {}
        self.open_orders_count: int = 0
        self.reserved_open_usdc: float = 0.0

    def reserve(self, order_id: str, cost: float) -> None:
        if not order_id:
            return
        self.reserved_by_order[order_id] = max(0.0, float(cost))
        self._recalc()

    def release(self, order_id: str) -> None:
        if order_id in self.reserved_by_order:
            self.reserved_by_order.pop(order_id, None)
            self._recalc()

    def sync_open_orders(self, open_orders: list[dict[str, Any]]) -> None:
        self.reserved_by_order = {}
        self.open_orders_count = 0
        total = 0.0
        for order in open_orders or []:
            order_id = str(order.get("id") or order.get("orderID") or order.get("order_id") or "")
            size = _order_remaining_size(order)
            price = _order_price(order)
            if size <= 0 or price <= 0:
                continue
            if order_id:
                self.reserved_by_order[order_id] = price * size
            self.open_orders_count += 1
            total += price * size
        self.reserved_open_usdc = total

    def _recalc(self) -> None:
        self.reserved_open_usdc = sum(self.reserved_by_order.values())
        self.open_orders_count = len(self.reserved_by_order)


class RiskManager:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.daily_spend = 0.0
        self.daily_pnl = 0.0
        self.daily_date = self._today()
        self.kill_switch = False
        self.peak_equity: float | None = None
        self.current_equity: float | None = None

    def _today(self) -> str:
        return datetime.now(timezone.utc).date().isoformat()

    def reset_if_new_day(self) -> None:
        today = self._today()
        if self.daily_date != today:
            self.daily_date = today
            self.daily_spend = 0.0
            self.daily_pnl = 0.0
            self.kill_switch = False

    def record_spend(self, amount: float) -> None:
        if amount <= 0:
            return
        self.reset_if_new_day()
        self.daily_spend += amount

    def record_realized_pnl(self, amount: float) -> None:
        if amount == 0:
            return
        self.reset_if_new_day()
        self.daily_pnl += amount

    def update_equity(self, equity: float | None) -> None:
        if equity is None:
            return
        self.current_equity = equity
        if self.peak_equity is None or equity > self.peak_equity:
            self.peak_equity = equity

    def load_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        self.daily_date = str(state.get("daily_date") or self.daily_date)
        self.daily_spend = float(state.get("daily_spend", self.daily_spend) or 0.0)
        self.daily_pnl = float(state.get("daily_pnl", self.daily_pnl) or 0.0)
        self.kill_switch = bool(state.get("kill_switch", self.kill_switch))
        peak = state.get("peak_equity")
        current = state.get("current_equity")
        if peak is not None:
            self.peak_equity = float(peak)
        if current is not None:
            self.current_equity = float(current)

    def snapshot(self) -> dict[str, Any]:
        return {
            "daily_date": self.daily_date,
            "daily_spend": self.daily_spend,
            "daily_pnl": self.daily_pnl,
            "kill_switch": self.kill_switch,
            "peak_equity": self.peak_equity,
            "current_equity": self.current_equity,
        }

    def drawdown_fraction(self) -> float:
        if self.peak_equity is None or self.current_equity is None or self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - self.current_equity) / self.peak_equity)

    def drawdown_multiplier(self) -> float:
        if self.drawdown_fraction() >= self.cfg.max_drawdown_fraction:
            return self.cfg.drawdown_risk_multiplier
        return 1.0

    def can_trade(self) -> bool:
        self.reset_if_new_day()
        if self.kill_switch:
            return False
        if self.daily_spend > self.cfg.daily_max_spend_usdc:
            self.kill_switch = True
            return False
        if self.daily_pnl <= -abs(self.cfg.daily_max_loss_usdc):
            self.kill_switch = True
            return False
        return True


class PositionSizer:
    def __init__(self, sizing_cfg, risk_cfg, interval_minutes: int) -> None:
        self.sizing = sizing_cfg
        self.risk = RiskManager(risk_cfg)
        self.inventory = InventoryManager()
        self.ledger = ReservationLedger()
        self.interval_seconds = int(interval_minutes) * 60
        self.session_id: int | None = None
        self.session_budget: float = 0.0
        self.session_spent: float = 0.0
        self.ema_bankroll: float | None = None
        self.bankroll_estimate: float | None = None
        self.pending_reserve: float = 0.0
        self.position_locks: list[tuple[int, float, int]] = []
        self.market_costs: dict[str, dict[str, dict[str, float]]] = {}
        self.active_assets: list[str] = []
        self.conservative_only: bool = False

    def set_active_assets(self, assets: list[str]) -> None:
        self.active_assets = [a.upper() for a in assets if a]

    def apply_conservative_overrides(self, cfg) -> None:
        if not getattr(cfg, "conservative_only", False):
            self.conservative_only = False
            return
        self.conservative_only = True
        cfg.sizing.directional_fraction = 0.0
        if getattr(cfg.sizing, "max_open_orders_mode", "fixed") != "auto":
            cfg.sizing.max_open_orders = max(1, int(cfg.sizing.max_open_orders * 0.5))
        cfg.sizing.max_concurrent_sessions = max(1, int(cfg.sizing.max_concurrent_sessions * 0.5))
        cfg.expiry_flush_seconds = max(cfg.expiry_flush_seconds, 60)
        cfg.order_refresh_seconds = min(cfg.order_refresh_seconds, 5)
        cfg.sizing.session_fraction_of_bankroll *= cfg.conservative_budget_multiplier
        cfg.sizing.fixed_session_budget_usdc *= cfg.conservative_budget_multiplier
        cfg.sizing.max_session_budget_usdc *= cfg.conservative_budget_multiplier
        cfg.sizing.max_market_budget_usdc *= cfg.conservative_market_multiplier

    def apply_auto_max_open_orders(self, cfg, active_markets: int) -> None:
        mode = (getattr(self.sizing, "max_open_orders_mode", "fixed") or "fixed").lower()
        if mode != "auto":
            return
        markets = max(1, int(active_markets))
        levels = max(1, int(getattr(cfg, "grid_levels", 1)))
        base = markets * levels * 2
        buffer = max(2, int(base * 0.1))
        max_orders = base + buffer
        if getattr(cfg, "conservative_only", False):
            max_orders = max(1, int(max_orders * 0.5))
        if self.ledger.open_orders_count > max_orders:
            max_orders = self.ledger.open_orders_count
        self.sizing.max_open_orders = max(1, int(max_orders))

    def update_market_costs(self, market_costs: dict[str, dict[str, dict[str, float]]]) -> None:
        if isinstance(market_costs, dict):
            self.market_costs = market_costs

    def update_open_orders(self, open_orders: list[dict[str, Any]]) -> None:
        self.inventory.update_open_orders(open_orders)
        self.ledger.sync_open_orders(open_orders)

    def update_bankroll(self, balance_usdc: float | None, fallback_estimate: float | None = None) -> None:
        if self.sizing.bankroll_mode == "manual":
            if self.sizing.bankroll_usdc > 0:
                self.bankroll_estimate = self.sizing.bankroll_usdc
        else:
            if balance_usdc is not None:
                self.bankroll_estimate = balance_usdc
        if (self.bankroll_estimate is None or self.bankroll_estimate <= 0) and fallback_estimate:
            self.bankroll_estimate = fallback_estimate
        if self.bankroll_estimate is None or self.bankroll_estimate <= 0:
            if self.sizing.bankroll_usdc > 0:
                self.bankroll_estimate = self.sizing.bankroll_usdc
            else:
                self.bankroll_estimate = self.sizing.fixed_session_budget_usdc
        self.risk.update_equity(self.bankroll_estimate)

    def record_trade_summary(self, summary: dict[str, Any]) -> None:
        spent = float(summary.get("spent_usdc", 0.0) or 0.0)
        pnl = float(summary.get("realized_pnl", 0.0) or 0.0)
        self.risk.record_spend(spent)
        self.risk.record_realized_pnl(pnl)
        for fill in summary.get("fills", []):
            match_ts = int(fill.get("match_ts", 0) or 0)
            cost = float(fill.get("cost", 0.0) or 0.0)
            session_id = self.get_session_id(match_ts)
            if match_ts and cost > 0:
                self.position_locks.append((match_ts + self.sizing.lock_time_seconds_default, cost, session_id))
            if self.session_id is not None and match_ts and session_id == self.session_id:
                self.session_spent += cost

    def release_expired_locks(self, now_ts: int) -> None:
        self.position_locks = [lock for lock in self.position_locks if lock[0] > now_ts]

    def locked_usdc(self) -> float:
        locks_total = sum(amount for _ts, amount, _sid in self.position_locks)
        if self.market_costs:
            total = 0.0
            for bucket in self.market_costs.values():
                if not isinstance(bucket, dict):
                    continue
                for entry in bucket.values():
                    if not isinstance(entry, dict):
                        continue
                    total += float(entry.get("cost", 0.0) or 0.0)
            return total + locks_total
        return locks_total

    def add_dry_run_lock(self, amount: float, now_ts: int | None = None) -> None:
        if amount <= 0:
            return
        now_ts = now_ts or int(time.time())
        session_id = self.get_session_id(now_ts)
        expiry = now_ts + int(self.sizing.lock_time_seconds_default)
        self.position_locks.append((expiry, float(amount), session_id))

    def available_bankroll(self) -> float:
        base = self.bankroll_estimate or 0.0
        reserved = self.ledger.reserved_open_usdc + self.pending_reserve + self.locked_usdc()
        return max(0.0, base - reserved)

    def get_session_id(self, now_ts: int | float) -> int:
        if self.interval_seconds <= 0:
            return int(now_ts)
        return int(now_ts // self.interval_seconds) * self.interval_seconds

    def _start_session_if_needed(self, now_ts: int) -> None:
        session_id = self.get_session_id(now_ts)
        if self.session_id == session_id:
            return
        self.session_id = session_id
        self.session_spent = 0.0
        self.session_budget = self._compute_session_budget()

    def _compute_session_budget(self) -> float:
        bankroll = self.available_bankroll()
        mode = (self.sizing.session_budget_mode or "proportional").lower()
        if mode == "fixed":
            budget = self.sizing.fixed_session_budget_usdc
        elif mode == "ema_proportional":
            alpha = self.sizing.session_ema_alpha
            if self.ema_bankroll is None:
                self.ema_bankroll = bankroll
            else:
                self.ema_bankroll = alpha * bankroll + (1 - alpha) * self.ema_bankroll
            budget = self.sizing.session_fraction_of_bankroll * (self.ema_bankroll or 0.0)
        else:
            budget = self.sizing.session_fraction_of_bankroll * bankroll
        budget = min(budget, self.sizing.max_session_budget_usdc)
        budget = max(0.0, budget)
        budget *= max(0.0, 1.0 - self.sizing.reserve_buffer_fraction)
        budget *= self.risk.drawdown_multiplier()
        if self.risk.drawdown_multiplier() < 1.0:
            self.sizing.directional_fraction = 0.0
        return min(budget, bankroll)

    def _normalized_weights(self) -> dict[str, float]:
        weights = self.sizing.asset_weights or {}
        if not self.active_assets:
            return weights
        filtered = {a: float(weights.get(a, 0.0)) for a in self.active_assets}
        total = sum(filtered.values())
        if total <= 0:
            even = 1.0 / max(1, len(self.active_assets))
            return {a: even for a in self.active_assets}
        return {a: w / total for a, w in filtered.items()}

    def market_budget(self, asset: str) -> float:
        weights = self._normalized_weights()
        weight = float(weights.get(asset.upper(), 0.0))
        budget = self.session_budget * weight
        budget = min(budget, self.sizing.max_market_budget_usdc)
        return max(0.0, budget)

    def can_trade(self, session_id: int, market_id: str, asset: str, est_cost: float) -> bool:
        if not self.risk.can_trade():
            return False
        if self.ledger.open_orders_count >= self.sizing.max_open_orders:
            return False
        if est_cost <= 0:
            return False
        remaining = max(0.0, self.session_budget - self.session_spent)
        if (self.session_spent + self.ledger.reserved_open_usdc + self.pending_reserve + est_cost) > self.session_budget:
            return False
        if est_cost > remaining:
            return False
        if est_cost > self.available_bankroll():
            return False
        if self.sizing.max_concurrent_sessions > 0:
            active_sessions = {sid for _ts, _amt, sid in self.position_locks}
            if session_id not in active_sessions and len(active_sessions) >= self.sizing.max_concurrent_sessions:
                return False
        return True

    def can_trade_rehedge(self, est_cost: float) -> bool:
        if not self.risk.can_trade():
            return False
        if est_cost <= 0:
            return False
        if est_cost > self.available_bankroll():
            return False
        return True

    def _remaining_session_budget(self) -> float:
        remaining = self.session_budget - self.session_spent - self.ledger.reserved_open_usdc - self.pending_reserve
        return max(0.0, remaining)

    def _order_budget_cap(self, orders: int) -> float:
        if orders <= 0:
            return 0.0
        remaining = self._remaining_session_budget()
        target_orders = max(1, self.sizing.max_open_orders)
        per_order = remaining / target_orders
        return per_order * orders

    def get_two_leg_order_plan(
        self,
        market,
        asset: str,
        mid_up: float,
        mid_down: float,
        signal: str | None = None,
        now_ts: int | None = None,
        up_token: str | None = None,
        down_token: str | None = None,
    ) -> OrderPlan:
        now_ts = now_ts or int(time.time())
        self._start_session_if_needed(now_ts)
        market_id = market.condition_id or market.id
        market_budget = self.market_budget(asset)
        hedged_budget = market_budget * self.sizing.hedged_core_fraction
        directional_budget = market_budget * max(0.0, self.sizing.directional_fraction)
        core_shares = 0.0
        tilt_shares = 0.0
        total_cost = 0.0
        reason = "ok"

        if mid_up <= 0 or mid_down <= 0:
            return OrderPlan(self.session_id or 0, market_id, asset, market_budget, hedged_budget, directional_budget, 0.0, 0.0, None, 0.0, "invalid_price", False, 1.0, None, 0.0)

        core_shares = hedged_budget / max(0.01, (mid_up + mid_down)) if hedged_budget > 0 else 0.0
        favored = None
        total_cost = core_shares * (mid_up + mid_down)
        # Directional tilt disabled for maker/arb flow.

        token_up = up_token or market.outcome_token_map().get("Up", "")
        token_down = down_token or market.outcome_token_map().get("Down", "")
        inventory = self.inventory.get_state(market_id, token_up, token_down, self.market_costs)
        net_ratio = inventory.net_ratio
        force_hedge_side = None
        rehedge_shares = 0.0

        if net_ratio > self.sizing.hard_skew_kill_ratio:
            force_hedge_side = "Down"
            diff = (inventory.filled_up + inventory.open_up) - (inventory.filled_down + inventory.open_down)
            rehedge_shares = max(0.0, abs(diff))
        elif net_ratio < 1.0 / self.sizing.hard_skew_kill_ratio:
            force_hedge_side = "Up"
            diff = (inventory.filled_down + inventory.open_down) - (inventory.filled_up + inventory.open_up)
            rehedge_shares = max(0.0, abs(diff))

        can_trade = self.can_trade(self.session_id or 0, market_id, asset, total_cost)
        if not can_trade:
            reason = "risk_block"

        remaining = self._remaining_session_budget()
        cap_total = self._order_budget_cap(2)
        target_cap = min(remaining, cap_total) if cap_total > 0 else remaining
        if total_cost > target_cap and total_cost > 0:
            scale = target_cap / total_cost if total_cost else 0.0
            core_shares *= scale
            tilt_shares *= scale
            total_cost *= scale
            if total_cost <= 0:
                reason = "session_budget_exhausted"
                can_trade = False

        return OrderPlan(
            self.session_id or 0,
            market_id,
            asset,
            market_budget,
            hedged_budget,
            directional_budget,
            core_shares,
            tilt_shares,
            favored,
            total_cost,
            reason,
            can_trade,
            net_ratio,
            force_hedge_side,
            rehedge_shares,
        )

    def get_one_leg_order_plan(
        self,
        market,
        asset: str,
        favored: str,
        best_ask: float,
        now_ts: int | None = None,
    ) -> OrderPlan:
        now_ts = now_ts or int(time.time())
        self._start_session_if_needed(now_ts)
        market_id = market.condition_id or market.id
        market_budget = self.market_budget(asset)
        directional_budget = 0.0
        if directional_budget <= 0:
            return OrderPlan(self.session_id or 0, market_id, asset, market_budget, 0.0, 0.0, 0.0, 0.0, favored, 0.0, "directional_disabled", False, 1.0, None, 0.0)
        if best_ask <= 0:
            return OrderPlan(self.session_id or 0, market_id, asset, market_budget, 0.0, directional_budget, 0.0, 0.0, favored, 0.0, "invalid_price", False, 1.0, None, 0.0)
        shares = directional_budget / best_ask
        total_cost = shares * best_ask
        remaining = self._remaining_session_budget()
        if total_cost > remaining and total_cost > 0:
            scale = remaining / total_cost if total_cost else 0.0
            shares *= scale
            total_cost *= scale
        can_trade = self.can_trade(self.session_id or 0, market_id, asset, total_cost)
        reason = "ok" if can_trade else "risk_block"
        return OrderPlan(
            self.session_id or 0,
            market_id,
            asset,
            market_budget,
            0.0,
            directional_budget,
            0.0,
            shares,
            favored,
            total_cost,
            reason,
            can_trade,
            1.0,
            None,
            0.0,
        )

    def build_log_fields(
        self,
        plan: OrderPlan,
        inventory: InventoryState,
        decision: str,
        reason: str,
        est_cost: float | None = None,
    ) -> dict[str, Any]:
        return {
            "session_id": plan.session_id,
            "market_id": plan.market_id,
            "asset": plan.asset,
            "bankroll_estimate": round(self.bankroll_estimate or 0.0, 4),
            "available_reserve": round(self.available_bankroll(), 4),
            "session_budget": round(self.session_budget, 4),
            "session_spent": round(self.session_spent, 4),
            "market_budget": round(plan.market_budget, 4),
            "hedged_budget": round(plan.hedged_budget, 4),
            "directional_budget": round(plan.directional_budget, 4),
            "inventory_up": round(inventory.filled_up + inventory.open_up, 6),
            "inventory_down": round(inventory.filled_down + inventory.open_down, 6),
            "net_ratio": round(inventory.net_ratio, 4),
            "planned_core_shares": round(plan.core_shares, 6),
            "planned_tilt_shares": round(plan.tilt_shares, 6),
            "est_cost": round(est_cost if est_cost is not None else plan.est_cost, 6),
            "decision": decision,
            "reason": reason,
        }


def _order_token(order: dict[str, Any]) -> str | None:
    for key in ("asset_id", "assetId", "token_id", "tokenId"):
        if key in order and order[key]:
            return str(order[key])
    return None


def _order_remaining_size(order: dict[str, Any]) -> float:
    for key in ("remaining_size", "remainingSize", "size", "original_size", "originalSize"):
        if key in order and order[key] not in (None, ""):
            try:
                return float(order[key])
            except (TypeError, ValueError):
                continue
    return 0.0


def _order_price(order: dict[str, Any]) -> float:
    for key in ("price", "orderPrice", "order_price"):
        if key in order and order[key] not in (None, ""):
            try:
                return float(order[key])
            except (TypeError, ValueError):
                continue
    return 0.0
