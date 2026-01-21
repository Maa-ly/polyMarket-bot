from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any


@dataclass
class TradeFill:
    match_ts: int
    cost: float
    market: str | None


class Storage:
    def __init__(self, path: str) -> None:
        self.path = path
        self.state = self._load_state(path)

    def _default_state(self) -> dict[str, Any]:
        return {
            "positions": {},
            "token_costs": {},
            "market_costs": {},
            "open_orders": {},
            "last_trade_ts": 0,
            "last_fill_ts": {},
            "bankroll_estimate": None,
            "risk_state": {},
            "session_state": {},
            "blocked_markets": {},
        }

    def _load_state(self, path: str) -> dict[str, Any]:
        if not os.path.exists(path):
            return self._default_state()
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                return self._default_state()
        if not isinstance(data, dict):
            return self._default_state()
        for key, value in self._default_state().items():
            data.setdefault(key, value)
        return data

    def save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.state, f)

    def get_positions(self) -> dict[str, float]:
        positions = self.state.get("positions")
        if not isinstance(positions, dict):
            positions = {}
            self.state["positions"] = positions
        return positions

    def get_token_costs(self) -> dict[str, dict[str, float]]:
        costs = self.state.get("token_costs")
        if not isinstance(costs, dict):
            costs = {}
            self.state["token_costs"] = costs
        return costs

    def get_market_costs(self) -> dict[str, dict[str, dict[str, float]]]:
        costs = self.state.get("market_costs")
        if not isinstance(costs, dict):
            costs = {}
            self.state["market_costs"] = costs
        return costs

    def get_last_trade_ts(self) -> int:
        try:
            return int(float(self.state.get("last_trade_ts", 0)))
        except (TypeError, ValueError):
            return 0

    def set_last_trade_ts(self, ts: int) -> None:
        self.state["last_trade_ts"] = int(ts)

    def get_last_fill_ts(self, market_id: str) -> int:
        last = self.state.get("last_fill_ts", {})
        if not isinstance(last, dict):
            last = {}
            self.state["last_fill_ts"] = last
        try:
            return int(float(last.get(market_id, 0)))
        except (TypeError, ValueError):
            return 0

    def set_last_fill_ts(self, market_id: str, ts: int) -> None:
        last = self.state.get("last_fill_ts")
        if not isinstance(last, dict):
            last = {}
            self.state["last_fill_ts"] = last
        last[market_id] = int(ts)

    def sync_open_orders(self, open_orders: list[dict[str, Any]]) -> None:
        mapped: dict[str, dict[str, Any]] = {}
        for order in open_orders or []:
            order_id = str(order.get("id") or order.get("orderID") or order.get("order_id") or "")
            if not order_id:
                continue
            mapped[order_id] = {
                "order_id": order_id,
                "market_id": order.get("market") or order.get("market_id") or "",
                "token_id": order.get("asset_id") or order.get("assetId") or order.get("token_id") or order.get("tokenId"),
                "side": order.get("side"),
                "price": order.get("price") or order.get("orderPrice") or order.get("order_price"),
                "size": order.get("size") or order.get("original_size") or order.get("originalSize"),
                "status": order.get("status") or order.get("state") or "open",
            }
        self.state["open_orders"] = mapped

    def record_order_action(
        self,
        order_id: str | None,
        market_id: str,
        token_id: str,
        side: str,
        price: float,
        size: float,
        tag: str,
        status: str,
    ) -> None:
        if not order_id:
            return
        open_orders = self.state.get("open_orders")
        if not isinstance(open_orders, dict):
            open_orders = {}
            self.state["open_orders"] = open_orders
        entry = {
            "order_id": order_id,
            "market_id": market_id,
            "token_id": token_id,
            "side": side,
            "price": price,
            "size": size,
            "tag": tag,
            "status": status,
        }
        if status in {"CANCELLED", "FILLED", "DONE"}:
            open_orders.pop(order_id, None)
        else:
            open_orders[order_id] = entry

    def apply_trades(self, trades: list[dict[str, Any]]) -> dict[str, Any]:
        summary = {"count": 0, "spent_usdc": 0.0, "realized_pnl": 0.0, "fills": []}
        if not trades:
            return summary
        positions = self.get_positions()
        costs = self.get_token_costs()
        market_costs = self.get_market_costs()
        last_ts = self.get_last_trade_ts()
        spent_usdc = 0.0
        realized_pnl = 0.0
        fills: list[TradeFill] = []
        count = 0
        for trade in trades:
            try:
                match_ts = int(float(trade.get("match_time") or trade.get("timestamp") or trade.get("last_update") or 0))
            except (TypeError, ValueError):
                match_ts = 0
            if match_ts <= last_ts:
                continue
            token_id = trade.get("asset_id") or trade.get("assetId")
            if not token_id:
                continue
            try:
                size = float(trade.get("size") or 0)
            except (TypeError, ValueError):
                continue
            side = str(trade.get("side") or "").upper()
            market_key = trade.get("market") or trade.get("market_id") or trade.get("marketId")
            condition_key = trade.get("condition_id") or trade.get("conditionId")
            if side in {"BUY", "SELL"}:
                entry = costs.get(token_id)
                if not isinstance(entry, dict):
                    entry = {"size": 0.0, "cost": 0.0}
                entry_size = float(entry.get("size", 0.0) or 0.0)
                entry_cost = float(entry.get("cost", 0.0) or 0.0)
                price = 0.0
                try:
                    price = float(trade.get("price") or 0)
                except (TypeError, ValueError):
                    price = 0.0

                if side == "BUY":
                    entry_size += size
                    cost = price * size
                    entry_cost += cost
                    spent_usdc += cost
                    fills.append(TradeFill(match_ts, cost, str(market_key) if market_key else None))
                else:
                    avg_cost = entry_cost / entry_size if entry_size > 0 else 0.0
                    realized_pnl += (price - avg_cost) * size
                    entry_size -= size
                    entry_cost -= avg_cost * size
                    if entry_size <= 0:
                        entry_size = 0.0
                        entry_cost = 0.0

                entry["size"] = entry_size
                entry["cost"] = entry_cost
                costs[token_id] = entry
                positions[token_id] = entry_size

                for key in {market_key, condition_key}:
                    if not key:
                        continue
                    market_bucket = market_costs.get(key)
                    if not isinstance(market_bucket, dict):
                        market_bucket = {}
                        market_costs[key] = market_bucket
                    market_entry = market_bucket.get(token_id)
                    if not isinstance(market_entry, dict):
                        market_entry = {"size": 0.0, "cost": 0.0}
                    market_size = float(market_entry.get("size", 0.0) or 0.0)
                    market_cost = float(market_entry.get("cost", 0.0) or 0.0)
                    if side == "BUY":
                        market_size += size
                        market_cost += price * size
                    else:
                        avg_cost = market_cost / market_size if market_size > 0 else 0.0
                        market_size -= size
                        market_cost -= avg_cost * size
                        if market_size <= 0:
                            market_size = 0.0
                            market_cost = 0.0
                    market_entry["size"] = market_size
                    market_entry["cost"] = market_cost
                    market_bucket[token_id] = market_entry
                    if match_ts:
                        self.set_last_fill_ts(str(key), match_ts)
            last_ts = max(last_ts, match_ts)
            count += 1

        self.state["positions"] = positions
        self.state["token_costs"] = costs
        self.state["market_costs"] = market_costs
        self.state["last_trade_ts"] = last_ts
        summary["count"] = count
        summary["spent_usdc"] = spent_usdc
        summary["realized_pnl"] = realized_pnl
        summary["fills"] = [fill.__dict__ for fill in fills]
        return summary

    def load_risk_state(self) -> dict[str, Any]:
        state = self.state.get("risk_state")
        return state if isinstance(state, dict) else {}

    def update_risk_state(self, risk_state: dict[str, Any]) -> None:
        if isinstance(risk_state, dict):
            self.state["risk_state"] = risk_state

    def update_session_state(self, session_id: int | None, budget: float, spent: float) -> None:
        self.state["session_state"] = {
            "session_id": session_id,
            "session_budget": budget,
            "session_spent": spent,
        }

    def update_bankroll_estimate(self, value: float | None) -> None:
        self.state["bankroll_estimate"] = value

    def is_blocked(self, market_id: str, session_id: int | None) -> bool:
        blocked = self.state.get("blocked_markets")
        if not isinstance(blocked, dict):
            blocked = {}
            self.state["blocked_markets"] = blocked
        if session_id is None:
            return False
        return str(blocked.get(market_id)) == str(session_id)

    def block_market(self, market_id: str, session_id: int | None) -> None:
        if session_id is None:
            return
        blocked = self.state.get("blocked_markets")
        if not isinstance(blocked, dict):
            blocked = {}
            self.state["blocked_markets"] = blocked
        blocked[market_id] = int(session_id)
