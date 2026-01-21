from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None


@dataclass
class SizingConfig:
    session_budget_mode: str = "proportional"
    fixed_session_budget_usdc: float = 13000.0
    session_fraction_of_bankroll: float = 0.15
    max_session_budget_usdc: float = 60000.0
    max_market_budget_usdc: float = 20000.0
    asset_weights: dict[str, float] = field(
        default_factory=lambda: {
            "BTC": 0.55,
            "ETH": 0.24,
            "SOL": 0.11,
            "XRP": 0.10,
        }
    )
    hedged_core_fraction: float = 0.80
    directional_fraction: float = 0.0
    max_inventory_skew_ratio: float = 1.25
    hard_skew_kill_ratio: float = 2.0
    reserve_buffer_fraction: float = 0.05
    max_open_orders: int = 50
    max_open_orders_mode: str = "fixed"
    max_concurrent_sessions: int = 8
    lock_time_seconds_default: int = 15 * 60
    bankroll_usdc: float = 0.0
    bankroll_mode: str = "api"
    session_ema_alpha: float = 0.2


@dataclass
class RiskConfig:
    daily_max_spend_usdc: float = 1_000_000.0
    daily_max_loss_usdc: float = 5_000.0
    max_drawdown_fraction: float = 0.20
    drawdown_risk_multiplier: float = 0.50


@dataclass
class BotConfig:
    poly_private_key: str | None = None
    poly_chain_id: int = 137
    poly_funder: str | None = None
    poly_signature_type: int = 0

    mode: str = "arb"
    assets: list[str] = field(default_factory=list)
    poll_seconds: int = 5

    instant_capture_enabled: bool = True
    instant_capture_edge_buffer: float = 0.015
    maker_edge_buffer: float = 0.02
    grid_levels: int = 4
    grid_step: float = 0.01
    per_level_size_shape: str = "equal"
    geometric_ratio: float = 1.4
    order_refresh_seconds: int = 8
    post_only: bool = True
    min_order_notional_usdc: float = 1.0

    hedge_timeout_seconds: float = 3.0
    expiry_flush_seconds: int = 30
    max_rehedge_market_budget_usdc: float = 2500.0

    sizing: SizingConfig = field(default_factory=SizingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)

    gamma_base_url: str = "https://gamma-api.polymarket.com"
    clob_host: str = "https://clob.polymarket.com"

    gamma_limit: int = 200
    gamma_max_pages: int = 10
    max_markets: int = 300
    gamma_order: str = "endDate"
    gamma_ascending: bool = False
    updown_interval_minutes: int = 15
    updown_current_only: bool = True

    state_path: str = "storage.json"
    trade_log_path: str = "trades.csv"
    decision_log_path: str = "decisions.csv"

    log_level: str = "INFO"
    dry_run: bool = False
    simulate: bool = False

    conservative_only: bool = False
    target_ratio_low: float = 0.98
    target_ratio_high: float = 1.02
    absolute_max_ratio: float = 1.05
    immediate_hedge_delay_seconds: float = 0.75
    flatten_trigger_shares: float = 2.0
    conservative_instant_edge_buffer: float = 0.025
    conservative_maker_edge_buffer: float = 0.03
    conservative_budget_multiplier: float = 0.5
    conservative_market_multiplier: float = 0.5


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    raw = str(value)
    parts = [p.strip() for p in raw.replace(",", " ").split()]
    return [p for p in parts if p]


def _parse_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def _parse_int(value: Any) -> int:
    if value is None or value == "":
        return 0
    return int(value)


def _maybe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _parse_weights(value: Any) -> dict[str, float]:
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return {str(k).upper(): float(v) for k, v in value.items()}
    raw = str(value).strip()
    if raw.startswith("{") and raw.endswith("}"):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return {str(k).upper(): float(v) for k, v in parsed.items()}
        except json.JSONDecodeError:
            pass
    weights: dict[str, float] = {}
    parts = [p for p in raw.replace(";", ",").split(",") if p.strip()]
    for part in parts:
        if ":" in part:
            key, val = part.split(":", 1)
        elif "=" in part:
            key, val = part.split("=", 1)
        else:
            continue
        key = key.strip().upper()
        try:
            weights[key] = float(val)
        except ValueError:
            continue
    return weights


def _load_config_file(path: str) -> dict[str, Any]:
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".json"):
            return json.load(f)
        if path.endswith((".yaml", ".yml")):
            if yaml is None:
                raise RuntimeError("PyYAML is required to load YAML config files")
            return yaml.safe_load(f) or {}
        # Default to JSON
        return json.load(f)


def _apply_config(cfg: BotConfig, data: dict[str, Any]) -> BotConfig:
    if not data:
        return cfg
    # Accept either ENV-style keys or snake_case keys.
    mapping: dict[str, str] = {
        "POLY_PRIVATE_KEY": "poly_private_key",
        "POLY_CHAIN_ID": "poly_chain_id",
        "POLY_FUNDER": "poly_funder",
        "POLY_SIGNATURE_TYPE": "poly_signature_type",
        "MODE": "mode",
        "ASSETS": "assets",
        "POLL_SECONDS": "poll_seconds",
        "LOG_LEVEL": "log_level",
        "INSTANT_CAPTURE_ENABLED": "instant_capture_enabled",
        "INSTANT_CAPTURE_EDGE_BUFFER": "instant_capture_edge_buffer",
        "MAKER_EDGE_BUFFER": "maker_edge_buffer",
        "GRID_LEVELS": "grid_levels",
        "GRID_STEP": "grid_step",
        "PER_LEVEL_SIZE_SHAPE": "per_level_size_shape",
        "GEOMETRIC_RATIO": "geometric_ratio",
        "ORDER_REFRESH_SECONDS": "order_refresh_seconds",
        "POST_ONLY": "post_only",
        "MIN_ORDER_NOTIONAL_USDC": "min_order_notional_usdc",
        "HEDGE_TIMEOUT_SECONDS": "hedge_timeout_seconds",
        "EXPIRY_FLUSH_SECONDS": "expiry_flush_seconds",
        "MAX_REHEDGE_MARKET_BUDGET_USDC": "max_rehedge_market_budget_usdc",
        "EDGE_BUFFER": "maker_edge_buffer",
        "SESSION_BUDGET_MODE": "sizing.session_budget_mode",
        "FIXED_SESSION_BUDGET_USDC": "sizing.fixed_session_budget_usdc",
        "SESSION_FRACTION_OF_BANKROLL": "sizing.session_fraction_of_bankroll",
        "MAX_SESSION_BUDGET_USDC": "sizing.max_session_budget_usdc",
        "MAX_MARKET_BUDGET_USDC": "sizing.max_market_budget_usdc",
        "ASSET_WEIGHTS": "sizing.asset_weights",
        "HEDGED_CORE_FRACTION": "sizing.hedged_core_fraction",
        "DIRECTIONAL_FRACTION": "sizing.directional_fraction",
        "MAX_INVENTORY_SKEW_RATIO": "sizing.max_inventory_skew_ratio",
        "HARD_SKEW_KILL_RATIO": "sizing.hard_skew_kill_ratio",
        "RESERVE_BUFFER_FRACTION": "sizing.reserve_buffer_fraction",
        "MAX_OPEN_ORDERS": "sizing.max_open_orders",
        "MAX_OPEN_ORDERS_MODE": "sizing.max_open_orders_mode",
        "MAX_CONCURRENT_SESSIONS": "sizing.max_concurrent_sessions",
        "LOCK_TIME_SECONDS_DEFAULT": "sizing.lock_time_seconds_default",
        "BANKROLL_USDC": "sizing.bankroll_usdc",
        "BANKROLL_MODE": "sizing.bankroll_mode",
        "SESSION_EMA_ALPHA": "sizing.session_ema_alpha",
        "DAILY_MAX_SPEND_USDC": "risk.daily_max_spend_usdc",
        "DAILY_MAX_LOSS_USDC": "risk.daily_max_loss_usdc",
        "MAX_DRAWDOWN_FRACTION": "risk.max_drawdown_fraction",
        "DRAWDOWN_RISK_MULTIPLIER": "risk.drawdown_risk_multiplier",
        "GAMMA_BASE_URL": "gamma_base_url",
        "CLOB_HOST": "clob_host",
        "GAMMA_LIMIT": "gamma_limit",
        "GAMMA_MAX_PAGES": "gamma_max_pages",
        "MAX_MARKETS": "max_markets",
        "GAMMA_ORDER": "gamma_order",
        "GAMMA_ASCENDING": "gamma_ascending",
        "UPDOWN_INTERVAL_MINUTES": "updown_interval_minutes",
        "UPDOWN_CURRENT_ONLY": "updown_current_only",
        "STATE_PATH": "state_path",
        "TRADE_LOG_PATH": "trade_log_path",
        "TRADE_LOG_CSV": "trade_log_path",
        "DECISION_LOG_PATH": "decision_log_path",
        "DECISION_LOG_CSV": "decision_log_path",
        "DRY_RUN": "dry_run",
        "SIMULATE": "simulate",
        "CONSERVATIVE_ONLY": "conservative_only",
        "TARGET_RATIO_LOW": "target_ratio_low",
        "TARGET_RATIO_HIGH": "target_ratio_high",
        "ABSOLUTE_MAX_RATIO": "absolute_max_ratio",
        "IMMEDIATE_HEDGE_DELAY_SECONDS": "immediate_hedge_delay_seconds",
        "FLATTEN_TRIGGER_SHARES": "flatten_trigger_shares",
        "CONSERVATIVE_INSTANT_EDGE_BUFFER": "conservative_instant_edge_buffer",
        "CONSERVATIVE_MAKER_EDGE_BUFFER": "conservative_maker_edge_buffer",
        "CONSERVATIVE_BUDGET_MULTIPLIER": "conservative_budget_multiplier",
        "CONSERVATIVE_MARKET_MULTIPLIER": "conservative_market_multiplier",
    }

    for key, value in data.items():
        attr = mapping.get(key, key)
        target = cfg
        if "." in attr:
            prefix, sub = attr.split(".", 1)
            target = getattr(cfg, prefix, None)
            if target is None:
                continue
            attr = sub
        if not hasattr(target, attr):
            continue
        if attr in {"assets"}:
            setattr(target, attr, _parse_list(value))
        elif attr == "max_open_orders":
            if isinstance(value, str) and value.strip().lower() == "auto":
                setattr(target, attr, 0)
                setattr(target, "max_open_orders_mode", "auto")
            else:
                parsed = _parse_int(value)
                setattr(target, attr, parsed)
                if parsed <= 0:
                    setattr(target, "max_open_orders_mode", "auto")
        elif attr in {
            "poly_chain_id",
            "poll_seconds",
            "gamma_limit",
            "gamma_max_pages",
            "max_markets",
            "poly_signature_type",
            "updown_interval_minutes",
            "max_concurrent_sessions",
            "lock_time_seconds_default",
            "grid_levels",
            "order_refresh_seconds",
            "expiry_flush_seconds",
        }:
            setattr(target, attr, _parse_int(value))
        elif attr in {
            "instant_capture_edge_buffer",
            "maker_edge_buffer",
            "grid_step",
            "geometric_ratio",
            "hedge_timeout_seconds",
            "max_rehedge_market_budget_usdc",
            "min_order_notional_usdc",
            "fixed_session_budget_usdc",
            "session_fraction_of_bankroll",
            "max_session_budget_usdc",
            "max_market_budget_usdc",
            "hedged_core_fraction",
            "directional_fraction",
            "max_inventory_skew_ratio",
            "hard_skew_kill_ratio",
            "reserve_buffer_fraction",
            "bankroll_usdc",
            "session_ema_alpha",
            "daily_max_spend_usdc",
            "daily_max_loss_usdc",
            "max_drawdown_fraction",
            "drawdown_risk_multiplier",
            "target_ratio_low",
            "target_ratio_high",
            "absolute_max_ratio",
            "immediate_hedge_delay_seconds",
            "flatten_trigger_shares",
            "conservative_instant_edge_buffer",
            "conservative_maker_edge_buffer",
            "conservative_budget_multiplier",
            "conservative_market_multiplier",
        }:
            setattr(target, attr, _parse_float(value))
        elif attr in {"asset_weights"}:
            setattr(target, attr, _parse_weights(value))
        elif attr in {
            "dry_run",
            "simulate",
            "gamma_ascending",
            "updown_current_only",
            "instant_capture_enabled",
            "post_only",
            "conservative_only",
        }:
            setattr(target, attr, _parse_bool(value))
        else:
            setattr(target, attr, value)
    return cfg


def load_config(config_path: str | None = None) -> BotConfig:
    if load_dotenv is not None:
        load_dotenv()

    cfg = BotConfig()

    file_path = config_path or os.getenv("POLY_CONFIG", "")
    if file_path:
        cfg = _apply_config(cfg, _load_config_file(file_path))

    env_data = {k: v for k, v in os.environ.items()}
    cfg = _apply_config(cfg, env_data)

    cfg.assets = [a.upper() for a in cfg.assets]
    cfg.mode = (cfg.mode or "arb").lower()
    cfg.log_level = (cfg.log_level or "INFO").upper()
    cfg.sizing.session_budget_mode = (cfg.sizing.session_budget_mode or "proportional").lower()
    cfg.sizing.bankroll_mode = (cfg.sizing.bankroll_mode or "api").lower()
    cfg.sizing.max_open_orders_mode = (cfg.sizing.max_open_orders_mode or "fixed").lower()
    cfg.per_level_size_shape = (cfg.per_level_size_shape or "equal").lower()
    cfg.sizing.asset_weights = {str(k).upper(): float(v) for k, v in (cfg.sizing.asset_weights or {}).items()}
    return cfg
