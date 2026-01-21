from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import json
import re
import time
from typing import Any, Callable

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import BalanceAllowanceParams, AssetType, BookParams, OrderArgs, OrderType, OpenOrderParams
from py_clob_client.exceptions import PolyApiException
from py_clob_client.order_builder.constants import BUY, SELL


@dataclass
class MarketInfo:
    id: str
    condition_id: str
    question: str
    slug: str
    asset: str
    start_ts: int | None
    end_ts: int | None
    token_up_id: str
    token_down_id: str
    active: bool
    closed: bool
    raw: dict[str, Any]
    tick_size: float | None = None
    min_size: float | None = None

    def outcome_token_map(self) -> dict[str, str]:
        return {"Up": self.token_up_id, "Down": self.token_down_id}


@dataclass
class BookTop:
    best_bid: float | None
    bid_size: float | None
    best_ask: float | None
    ask_size: float | None
    tick_size: float
    min_order_size: float


class GammaClient:
    def __init__(
        self,
        base_url: str,
        timeout: int = 15,
        order: str | None = None,
        ascending: bool | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.order = order
        self.ascending = ascending
        self._client: httpx.AsyncClient | None = None

    async def _client_or_create(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={"User-Agent": "polymarket-bot"},
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(httpx.HTTPError),
    )
    async def _get(self, path: str, params: dict[str, Any]) -> Any:
        url = f"{self.base_url}{path}"
        client = await self._client_or_create()
        resp = await client.get(url, params=params)
        if resp.status_code in {429} or resp.status_code >= 500:
            raise httpx.HTTPStatusError("Gamma throttled", request=resp.request, response=resp)
        resp.raise_for_status()
        return resp.json()

    async def list_markets(
        self,
        limit: int,
        offset: int,
        closed: bool,
        end_date_max: str | None = None,
    ) -> list[dict[str, Any]]:
        params = {
            "limit": limit,
            "offset": offset,
            "closed": str(closed).lower(),
        }
        if self.order:
            params["order"] = self.order
        if self.ascending is not None:
            params["ascending"] = str(self.ascending).lower()
        if end_date_max:
            params["end_date_max"] = end_date_max
        data = await self._get("/markets", params)
        if isinstance(data, dict) and "data" in data:
            return data["data"]
        if isinstance(data, list):
            return data
        return []

    async def iter_markets(
        self,
        limit: int,
        max_pages: int,
        closed: bool,
        end_date_max: str | None = None,
    ) -> list[dict[str, Any]]:
        markets: list[dict[str, Any]] = []
        for page in range(max_pages):
            batch = await self.list_markets(
                limit=limit,
                offset=page * limit,
                closed=closed,
                end_date_max=end_date_max,
            )
            if not batch:
                break
            markets.extend(batch)
            if len(batch) < limit:
                break
        return markets


class ClobClientWrapper:
    def __init__(
        self,
        host: str,
        private_key: str | None,
        chain_id: int,
        signature_type: int,
        funder: str | None,
        read_only: bool = False,
    ) -> None:
        self.host = host
        self.read_only = read_only or not private_key
        if self.read_only:
            self.client = ClobClient(host)
        else:
            self.client = ClobClient(
                host,
                key=private_key,
                chain_id=chain_id,
                signature_type=signature_type,
                funder=funder,
            )
            self.client.set_api_creds(self.client.create_or_derive_api_creds())

    def get_order_book(self, token_id: str):
        try:
            return self._with_retries(self.client.get_order_book, token_id)
        except PolyApiException as exc:
            if getattr(exc, "status_code", None) == 404:
                return None
            raise

    def get_order_books(self, token_ids: list[str]):
        params = [BookParams(token_id=token_id) for token_id in token_ids]
        return self._with_retries(self.client.get_order_books, params)

    def place_limit_order(
        self,
        token_id: str,
        price: float,
        size: float,
        side: str,
        order_type: OrderType = OrderType.GTC,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        if dry_run or self.read_only:
            return {"dry_run": True, "token_id": token_id, "price": price, "size": size, "side": side}
        order = OrderArgs(token_id=token_id, price=price, size=size, side=side)
        signed = self.client.create_order(order)
        return self._with_retries(self.client.post_order, signed, order_type)

    def get_order(self, order_id: str) -> dict[str, Any] | None:
        if self.read_only:
            return None
        return self._with_retries(self.client.get_order, order_id)

    def cancel(self, order_id: str) -> dict[str, Any] | None:
        if self.read_only:
            return None
        return self._with_retries(self.client.cancel, order_id)

    def cancel_all(self) -> dict[str, Any] | None:
        if self.read_only:
            return None
        return self._with_retries(self.client.cancel_all)

    def get_open_orders(self) -> list[dict[str, Any]]:
        if self.read_only:
            return []
        return self._with_retries(self.client.get_orders, OpenOrderParams())

    def get_collateral_balance_allowance(self) -> tuple[float | None, float | None]:
        if self.read_only:
            return None, None
        params = BalanceAllowanceParams()
        params.asset_type = AssetType.COLLATERAL
        resp = self._with_retries(self.client.get_balance_allowance, params)

        balance = None
        if isinstance(resp, dict):
            raw_balance = resp.get("balance")
            try:
                balance = float(raw_balance)
            except (TypeError, ValueError):
                balance = None

        allowance = None
        if isinstance(resp, dict):
            allowances = resp.get("allowances")
            if isinstance(allowances, dict) and allowances:
                values = []
                for val in allowances.values():
                    try:
                        values.append(float(val))
                    except (TypeError, ValueError):
                        continue
                if values:
                    allowance = max(values)
            else:
                raw_allowance = resp.get("allowance")
                try:
                    allowance = float(raw_allowance)
                except (TypeError, ValueError):
                    allowance = None
        return balance, allowance

    def _with_retries(self, fn: Callable, *args, **kwargs):
        backoff = 0.5
        for attempt in range(4):
            try:
                return fn(*args, **kwargs)
            except PolyApiException as exc:
                if getattr(exc, "status_code", None) == 404:
                    raise
                if attempt == 3:
                    raise
                time.sleep(backoff)
                backoff *= 2
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(backoff)
                backoff *= 2


def parse_list_field(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        raw = value.strip()
        if raw.startswith("[") and raw.endswith("]"):
            try:
                parsed = json.loads(raw)
                return [str(v) for v in parsed]
            except json.JSONDecodeError:
                pass
        return [v.strip() for v in raw.split(",") if v.strip()]
    return []


def _parse_ts(value: Any) -> int | None:
    if not value:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    try:
        iso = str(value).replace("Z", "+00:00")
        return int(datetime.fromisoformat(iso).timestamp())
    except ValueError:
        return None


def _extract_tokens(raw: dict[str, Any]) -> tuple[str | None, str | None]:
    outcomes = parse_list_field(raw.get("outcomes"))
    token_ids = parse_list_field(raw.get("clobTokenIds"))
    mapping = {}
    for outcome, token_id in zip(outcomes, token_ids):
        mapping[str(outcome)] = str(token_id)
    up = mapping.get("Up") or mapping.get("UP") or mapping.get("up")
    down = mapping.get("Down") or mapping.get("DOWN") or mapping.get("down")
    return up, down


def _parse_slug_ts(raw: dict[str, Any]) -> tuple[int | None, int | None]:
    slug = str(raw.get("eventSlug") or raw.get("event_slug") or "")
    if not slug:
        return None, None
    # Best-effort ISO-like timestamp extraction.
    match = re.findall(r"\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2})?", slug)
    if not match:
        return None, None
    start = _parse_ts(match[0])
    end = _parse_ts(match[-1]) if len(match) > 1 else None
    return start, end


def _market_times(raw: dict[str, Any]) -> tuple[int | None, int | None]:
    start = _parse_ts(raw.get("startDate") or raw.get("start_date"))
    end = _parse_ts(raw.get("endDate") or raw.get("end_date"))
    if start is None or end is None:
        slug_start, slug_end = _parse_slug_ts(raw)
        if start is None:
            start = slug_start
        if end is None:
            end = slug_end
    return start, end


def to_market_info(raw: dict[str, Any]) -> MarketInfo | None:
    up_token, down_token = _extract_tokens(raw)
    if not up_token or not down_token:
        return None
    asset = market_asset_symbol_raw(raw)
    start_ts, end_ts = _market_times(raw)
    return MarketInfo(
        id=str(raw.get("id", "")),
        condition_id=str(raw.get("conditionId", "")),
        question=str(raw.get("question", "")),
        slug=str(raw.get("slug", "")),
        asset=asset,
        start_ts=start_ts,
        end_ts=end_ts,
        token_up_id=up_token,
        token_down_id=down_token,
        active=bool(raw.get("active", True)),
        closed=bool(raw.get("closed", False)),
        raw=raw,
    )


def market_time_to_expiry(market: MarketInfo) -> float | None:
    if market.end_ts is None:
        return None
    now = datetime.now(timezone.utc)
    return float(market.end_ts - int(now.timestamp()))


def select_up_down_tokens(market: MarketInfo) -> dict[str, str]:
    if market.token_up_id and market.token_down_id:
        return {"Up": market.token_up_id, "Down": market.token_down_id}
    return {}


def is_updown_market(market: MarketInfo, assets: list[str], interval_minutes: int = 15) -> bool:
    question = market.question.lower()
    slug = market.slug.lower()
    interval_tag = f"updown-{interval_minutes}m"
    if "up or down" not in question and interval_tag not in slug:
        return False
    if interval_tag not in slug:
        return False
    if not market.active or market.closed:
        return False
    if assets:
        return market.asset in assets
    return True


def market_asset_symbol_raw(raw: dict[str, Any]) -> str:
    question = str(raw.get("question", "")).lower()
    slug = str(raw.get("slug", "")).lower()
    mapping = {
        "BTC": ["bitcoin", "btc", "btc-"],
        "ETH": ["ethereum", "eth", "eth-"],
        "SOL": ["solana", "sol", "sol-"],
        "XRP": ["xrp", "xrp-"],
        "DOGE": ["dogecoin", "doge", "doge-"],
    }
    for symbol, patterns in mapping.items():
        for pat in patterns:
            if question.startswith(pat) or slug.startswith(pat) or pat in slug:
                return symbol
    return "UNKNOWN"


def market_asset_symbol(market: MarketInfo) -> str:
    return market.asset


def _level_price_size(level: Any) -> tuple[float, float] | None:
    if isinstance(level, dict):
        price = level.get("price")
        size = level.get("size")
    else:
        price = getattr(level, "price", None)
        size = getattr(level, "size", None)
    try:
        return float(price), float(size)
    except (TypeError, ValueError):
        return None


def best_level(levels: list[Any], side: str) -> tuple[float, float] | None:
    if not levels:
        return None
    side_norm = str(side or "").upper()
    best: tuple[float, float] | None = None
    for level in levels:
        parsed = _level_price_size(level)
        if parsed is None:
            continue
        price, size = parsed
        if best is None:
            best = (price, size)
            continue
        if side_norm == BUY:
            if price < best[0]:
                best = (price, size)
        elif side_norm == SELL:
            if price > best[0]:
                best = (price, size)
        else:
            # Fallback to the first valid level when side is unknown.
            break
    return best


def _tick_size(book: Any) -> float:
    for key in ("tick_size", "tickSize", "min_tick", "minTick", "price_increment", "priceIncrement"):
        val = getattr(book, key, None)
        if val is None:
            continue
        try:
            return float(val)
        except (TypeError, ValueError):
            continue
    return 0.0


def _min_order_size(book: Any) -> float:
    for key in ("min_order_size", "minOrderSize", "min_size", "minSize"):
        val = getattr(book, key, None)
        if val is None:
            continue
        try:
            return float(val)
        except (TypeError, ValueError):
            continue
    return 0.0


def book_top(book: Any) -> BookTop:
    best_bid = best_level(getattr(book, "bids", []), SELL)
    best_ask = best_level(getattr(book, "asks", []), BUY)
    bid_price, bid_size = best_bid if best_bid else (None, None)
    ask_price, ask_size = best_ask if best_ask else (None, None)
    return BookTop(
        best_bid=bid_price,
        bid_size=bid_size,
        best_ask=ask_price,
        ask_size=ask_size,
        tick_size=_tick_size(book),
        min_order_size=_min_order_size(book),
    )


def round_price(price: float, tick_size: float, side: str) -> float:
    if tick_size <= 0:
        return price
    p = Decimal(str(price))
    t = Decimal(str(tick_size))
    if side.upper() == BUY:
        rounded = (p / t).to_integral_value(rounding=ROUND_CEILING) * t
    else:
        rounded = (p / t).to_integral_value(rounding=ROUND_FLOOR) * t
    return float(rounded)


def round_size(size: float, decimals: int = 4) -> float:
    if size <= 0:
        return 0.0
    fmt = Decimal("1").scaleb(-decimals)
    return float(Decimal(str(size)).quantize(fmt, rounding=ROUND_FLOOR))


def extract_order_id(response: dict[str, Any] | None) -> str | None:
    if not response:
        return None
    for key in ("orderID", "order_id", "id"):
        if key in response:
            return str(response[key])
    if isinstance(response.get("data"), dict):
        return extract_order_id(response["data"])
    return None


def parse_order_fill(order: dict[str, Any] | None) -> tuple[float | None, float | None, str | None]:
    if not order:
        return None, None, None
    status = order.get("status") or order.get("state")

    def _num(val: Any) -> float | None:
        if val is None or val == "":
            return None
        return float(val)

    size = _num(order.get("size") or order.get("original_size") or order.get("originalSize"))
    filled = _num(order.get("filled_size") or order.get("filledSize") or order.get("filled"))
    remaining = _num(order.get("remaining_size") or order.get("remainingSize"))

    if filled is None and remaining is not None and size is not None:
        filled = max(size - remaining, 0.0)

    if filled is None and status and status.upper() in {"FILLED", "CLOSED", "MATCHED", "DONE"}:
        filled = size

    return filled, size, status
