# Polymarket CLOB ARB Bot (Up/Down 15m)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` with your credentials and settings (no secrets in code):

```
POLY_PRIVATE_KEY=...
POLY_CHAIN_ID=137
POLY_SIGNATURE_TYPE=0
POLY_FUNDER=0x...
MODE=arb
ASSETS=BTC,ETH,SOL,XRP
POLL_SECONDS=5
DRY_RUN=true
LOG_LEVEL=INFO

# Strategy
INSTANT_CAPTURE_ENABLED=true
INSTANT_CAPTURE_EDGE_BUFFER=0.015
MAKER_EDGE_BUFFER=0.02
GRID_LEVELS=4
GRID_STEP=0.01
PER_LEVEL_SIZE_SHAPE=equal
GEOMETRIC_RATIO=1.4
ORDER_REFRESH_SECONDS=8
POST_ONLY=true
MIN_ORDER_NOTIONAL_USDC=1.0

# Inventory / hedge
HEDGE_TIMEOUT_SECONDS=3
EXPIRY_FLUSH_SECONDS=30
MAX_REHEDGE_MARKET_BUDGET_USDC=2500

# Sizing
SESSION_BUDGET_MODE=proportional
FIXED_SESSION_BUDGET_USDC=13000
SESSION_FRACTION_OF_BANKROLL=0.15
MAX_SESSION_BUDGET_USDC=60000
MAX_MARKET_BUDGET_USDC=20000
ASSET_WEIGHTS=BTC:0.55,ETH:0.24,SOL:0.11,XRP:0.10
HEDGED_CORE_FRACTION=0.80
DIRECTIONAL_FRACTION=0.00
RESERVE_BUFFER_FRACTION=0.05
MAX_OPEN_ORDERS=50  # or "auto"
MAX_OPEN_ORDERS_MODE=fixed  # fixed | auto
MAX_CONCURRENT_SESSIONS=8
LOCK_TIME_SECONDS_DEFAULT=900
BANKROLL_MODE=api
BANKROLL_USDC=0

# Risk
DAILY_MAX_SPEND_USDC=1000000
DAILY_MAX_LOSS_USDC=5000
MAX_DRAWDOWN_FRACTION=0.20
DRAWDOWN_RISK_MULTIPLIER=0.50

# Conservative-only mode (optional)
CONSERVATIVE_ONLY=false
TARGET_RATIO_LOW=0.98
TARGET_RATIO_HIGH=1.02
ABSOLUTE_MAX_RATIO=1.05
IMMEDIATE_HEDGE_DELAY_SECONDS=0.75
FLATTEN_TRIGGER_SHARES=2
CONSERVATIVE_INSTANT_EDGE_BUFFER=0.025
CONSERVATIVE_MAKER_EDGE_BUFFER=0.03
CONSERVATIVE_BUDGET_MULTIPLIER=0.5
CONSERVATIVE_MARKET_MULTIPLIER=0.5
```

Run (dry-run):

```bash
python bot.py --mode arb --dry-run --assets BTC ETH SOL XRP --poll 3 --bankroll-mode manual --bankroll-usdc 5000
```

## Strategy Overview

This bot profits by accumulating **full sets** (Up + Down) at a combined cost **below $1**. It does not predict direction.

### A) Instant Capture (optional, taker-like)
- Read best asks for Up and Down.
- If `sum_ask <= 1 - INSTANT_CAPTURE_EDGE_BUFFER`, buy equal shares on both sides (FOK marketable limits).
- Size uses the **hedged core budget** for the market.

### B) Grid / Mean-Reversion Maker (primary)
- Place a **post-only grid** of BUY orders on both sides.
- **Set-feasible filter** keeps bids low enough to allow an eventual cheap full set:
  - `allowed_bid_up = min(best_bid_up, (1 - MAKER_EDGE_BUFFER) - best_bid_down)`
  - `allowed_bid_down = min(best_bid_down, (1 - MAKER_EDGE_BUFFER) - best_bid_up)`
- Grid levels step down by `GRID_STEP` (rounded to tick).
- Sizes distribute across levels (equal or geometric).

### Inventory & Hedge
- Tracks filled + open inventory per market.
- Soft skew scales quotes; hard skew stops adding to the overweight side.
- Re-hedge on large imbalances or after `HEDGE_TIMEOUT_SECONDS`.
- Near expiry (`EXPIRY_FLUSH_SECONDS`), cancel grids and only hedge/flatten.

## Conservative-Only Mode
Enable with `CONSERVATIVE_ONLY=true` or `--conservative-only`.

Rules enforced:
- Directional fraction forced to **0**.
- Strict 1:1 ratio target with `TARGET_RATIO_LOW/HIGH`.
- Immediate hedge on any imbalance after `IMMEDIATE_HEDGE_DELAY_SECONDS`.
- If hedge cannot complete and imbalance persists, **flatten** via SELLs; if flatten is impossible, the bot cancels and blocks that market for the rest of the session.
- Symmetric grids only; cancel overweight-side orders when out of band.
- Stricter edge buffers + tighter budgets and order refresh cadence.

## Sizing & Risk
- Session budgets align to 15-minute windows.
- Asset weights allocate budgets across BTC/ETH/SOL/XRP.
- Reserved capital accounts for open orders and filled inventory.
- Daily max spend/loss and drawdown throttling.

## Logs
- `trades.csv`: order actions (timestamp, market_id, market_slug, token_id, side, price, size, tag, order_id, status)
- `decisions.csv`: one row per market per loop with pricing, budgets, inventory, conservative flags, action_reason, and decision label (CAPTURE/GRID/REHEDGE/SKIP).

## State
- `storage.json` persists positions, cost basis, open orders, and risk/session state.

## Known Risks
- Partial fills can create directional exposure.
- Stale order books or latency can degrade edge.
- Liquidity can disappear; FOK orders may fail.
- Fee/slippage sensitivity when edge buffers are too tight.

## Optional On-Chain Merge
`merge.py` contains a disabled stub for future full-set merges. It is OFF by default.
