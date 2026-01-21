import csv
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

import streamlit as st

ENV_PATH = ".env"
PID_PATH = "bot.pid"
RUNTIME_PATH = "bot_runtime.json"
DEFAULT_LOG_PATH = "bot.log"
DEFAULT_TRADE_LOG_PATH = "trades.csv"
DEFAULT_DECISION_LOG_PATH = "decisions.csv"
DEFAULT_STATE_PATH = "storage.json"

ENV_LINE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
SECRET_KEYS = {"POLY_PRIVATE_KEY"}
BOOL_KEYS = {
    "DRY_RUN",
    "UPDOWN_CURRENT_ONLY",
    "GAMMA_ASCENDING",
}
LOG_LEVEL_KEY = "LOG_LEVEL"
MODE_KEY = "MODE"
RECYCLE_MODE_KEY = "MM_RECYCLE_MODE"
ALLOWED_KEYS = [
    "POLY_PRIVATE_KEY",
    "POLY_CHAIN_ID",
    "POLY_SIGNATURE_TYPE",
    "POLY_FUNDER",
    "MODE",
    "ASSETS",
    "POLL_SECONDS",
    "DRY_RUN",
    "MM_EDGE_BUFFER",
    "MM_MIN_TTE_SECONDS",
    "MM_MIN_LIQUIDITY_SHARES",
    "MM_QUOTE_IMPROVE_TICKS",
    "MM_PRICE_TOLERANCE_TICKS",
    "MM_QUOTE_INTERVAL_SECONDS",
    "MM_QUOTE_JITTER_SECONDS",
    "MM_RECYCLE_MODE",
    "MM_RECYCLE_EDGE_BUFFER",
    "MM_RECYCLE_MIN_SETS",
    "SESSION_BUDGET_MODE",
    "FIXED_SESSION_BUDGET_USDC",
    "SESSION_FRACTION_OF_BANKROLL",
    "MAX_SESSION_BUDGET_USDC",
    "MAX_MARKET_BUDGET_USDC",
    "ASSET_WEIGHTS",
    "HEDGED_CORE_FRACTION",
    "MAX_INVENTORY_SKEW_RATIO",
    "HARD_SKEW_KILL_RATIO",
    "RESERVE_BUFFER_FRACTION",
    "MIN_ORDER_NOTIONAL_USDC",
    "MAX_OPEN_ORDERS",
    "MAX_OPEN_ORDERS_MODE",
    "MAX_CONCURRENT_SESSIONS",
    "LOCK_TIME_SECONDS_DEFAULT",
    "BANKROLL_USDC",
    "BANKROLL_MODE",
    "SESSION_EMA_ALPHA",
    "DAILY_MAX_SPEND_USDC",
    "DAILY_MAX_LOSS_USDC",
    "MAX_DRAWDOWN_FRACTION",
    "DRAWDOWN_RISK_MULTIPLIER",
    "GAMMA_ORDER",
    "GAMMA_ASCENDING",
    "UPDOWN_INTERVAL_MINUTES",
    "UPDOWN_CURRENT_ONLY",
    "LOG_LEVEL",
]
TRADE_LOG_HEADER = [
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
DECISION_LOG_HEADER = [
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


def load_env_file(path: str) -> tuple[dict[str, str], list[tuple[str | None, str]]]:
    env: dict[str, str] = {}
    lines: list[tuple[str | None, str]] = []
    if not os.path.exists(path):
        return env, lines
    with open(path, "r", encoding="utf-8") as f:
        for raw in f.read().splitlines():
            match = ENV_LINE_RE.match(raw)
            if match:
                key, value = match.group(1), match.group(2)
                env[key] = value
                lines.append((key, raw))
            else:
                lines.append((None, raw))
    return env, lines


def save_env_file(path: str, updated: dict[str, str], lines: list[tuple[str | None, str]]) -> None:
    rendered: list[str] = []
    seen: set[str] = set()
    for key, raw in lines:
        if key is None:
            rendered.append(raw)
            continue
        rendered.append(f"{key}={updated.get(key, '')}")
        seen.add(key)
    for key, value in updated.items():
        if key not in seen:
            rendered.append(f"{key}={value}")
    content = "\n".join(rendered)
    if content:
        content += "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def read_runtime() -> dict:
    if not os.path.exists(RUNTIME_PATH):
        return {}
    try:
        with open(RUNTIME_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def write_runtime(pid: int, log_path: str) -> None:
    payload = {
        "pid": pid,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "log_path": log_path,
    }
    with open(RUNTIME_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def clear_runtime() -> None:
    for path in (RUNTIME_PATH, PID_PATH):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def read_tail(path: str, max_lines: int = 200) -> str:
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        return "\n".join(lines[-max_lines:])
    except OSError:
        return ""


def read_trade_log(path: str, max_rows: int = 20) -> list[dict[str, str]]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        return rows[-max_rows:]
    except OSError:
        return []


def is_running(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def load_env_map() -> dict[str, str]:
    env, _lines = load_env_file(ENV_PATH)
    return env


def resolve_paths(env_map: dict[str, str]) -> dict[str, str]:
    def pick(keys: tuple[str, ...], default: str) -> str:
        for key in keys:
            value = env_map.get(key)
            if value is None:
                continue
            value = str(value).strip()
            if value:
                return value
        return default

    return {
        "log_path": pick(("LOG_PATH",), DEFAULT_LOG_PATH),
        "trade_log_path": pick(("TRADE_LOG_PATH", "TRADE_LOG_CSV"), DEFAULT_TRADE_LOG_PATH),
        "decision_log_path": pick(
            ("DECISION_LOG_PATH", "DECISION_LOG_CSV"), DEFAULT_DECISION_LOG_PATH
        ),
        "state_path": pick(("STATE_PATH",), DEFAULT_STATE_PATH),
    }


def reset_run_state(paths: dict[str, str]) -> None:
    with open(paths["trade_log_path"], "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(TRADE_LOG_HEADER)
    with open(paths["decision_log_path"], "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(DECISION_LOG_HEADER)
    with open(paths["state_path"], "w", encoding="utf-8") as f:
        f.write("{}\n")
    with open(paths["log_path"], "w", encoding="utf-8") as f:
        f.write("")


def start_bot(dry_run: bool, paths: dict[str, str]) -> tuple[bool, str]:
    runtime = read_runtime()
    pid = runtime.get("pid")
    if pid and is_running(pid):
        return False, f"Bot already running (pid {pid})."

    reset_run_state(paths)

    env = os.environ.copy()
    env.update(load_env_map())
    env["DRY_RUN"] = "true" if dry_run else "false"

    log_file = open(paths["log_path"], "a", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "bot.py"],
        stdout=log_file,
        stderr=log_file,
        env=env,
        start_new_session=True,
    )
    log_file.close()
    with open(PID_PATH, "w", encoding="utf-8") as f:
        f.write(str(proc.pid))
    write_runtime(proc.pid, paths["log_path"])
    return True, f"Started bot (pid {proc.pid})."


def stop_bot() -> tuple[bool, str]:
    runtime = read_runtime()
    pid = runtime.get("pid")
    if not pid or not is_running(pid):
        clear_runtime()
        return False, "Bot is not running."
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as exc:
        clear_runtime()
        return False, f"Failed to stop bot: {exc}"

    time.sleep(1.0)
    if is_running(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    clear_runtime()
    return True, f"Stopped bot (pid {pid})."


st.set_page_config(page_title="Polymarket Bot", layout="centered")
st.title("Polymarket Bot")

runtime = read_runtime()
pid = runtime.get("pid")
running = is_running(pid)
status = "running" if running else "stopped"
st.markdown(f"**Status:** {status}")
raw_env_map = load_env_map()
paths = resolve_paths(raw_env_map)
log_path = runtime.get("log_path", paths["log_path"])
trade_log_path = paths["trade_log_path"]
if running:
    st.caption(f"PID {pid} | Log: {log_path}")

env_map, env_lines = load_env_file(ENV_PATH)
env_map = {k: v for k, v in env_map.items() if k in ALLOWED_KEYS}
env_lines = [(key, raw) for key, raw in env_lines if key is None or key in ALLOWED_KEYS]
dry_run_default = str(env_map.get("DRY_RUN", "")).strip().lower() in {"1", "true", "yes", "y", "on"}
run_mode = st.radio(
    "Run mode",
    ["Live", "Dry-run (simulated)"],
    index=1 if dry_run_default else 0,
    horizontal=True,
)

col_start, col_stop = st.columns(2)
with col_start:
    if st.button("Start bot", use_container_width=True):
        ok, msg = start_bot(run_mode != "Live", paths)
        if ok:
            st.success(msg)
        else:
            st.warning(msg)
        st.rerun()
with col_stop:
    if st.button("Stop bot", use_container_width=True):
        ok, msg = stop_bot()
        if ok:
            st.success(msg)
        else:
            st.warning(msg)
        st.rerun()

st.divider()
st.subheader("Bot output")
log_text = read_tail(log_path, 200)
if log_text:
    log_container = st.container(height=320, border=False, key="log_box", width="stretch")
    with log_container:
        st.code(log_text)
    st.html(
        """
<script>
(() => {
  const doc = window.parent.document;
  let style = doc.getElementById("bot-log-style");
  if (!style) {
    style = doc.createElement("style");
    style.id = "bot-log-style";
    style.innerText = `
      .st-key-log_box div[data-testid="stCodeBlock"] pre,
      .st-key-log_box div[data-testid="stCodeBlock"] code {
        white-space: pre-wrap;
        word-break: break-word;
      }
      .st-key-log_box div[data-testid="stCodeBlock"] {
        overflow-x: hidden;
      }
    `;
    doc.head.appendChild(style);
  }
  const getTarget = () => {
    const box = doc.querySelector(".st-key-log_box");
    if (!box) return null;
    return { box };
  };
  const findScrollable = (root) => {
    const elements = root.querySelectorAll("*");
    for (const el of elements) {
      if (el.scrollHeight - el.clientHeight > 2) {
        return el;
      }
    }
    return root;
  };
  const scrollToBottom = () => {
    const target = getTarget();
    if (!target) return;
    const scrollEl = findScrollable(target.box);
    target.box.style.overflowY = "auto";
    scrollEl.scrollTop = scrollEl.scrollHeight;
  };
  scrollToBottom();
  setTimeout(scrollToBottom, 100);
  setTimeout(scrollToBottom, 300);

  const target = getTarget();
  if (!target) return;
  const node = target.box;
  if (!node.dataset.logObserverAttached) {
    const observer = new MutationObserver(() => scrollToBottom());
    observer.observe(node, { childList: true, subtree: true, characterData: true });
    node.dataset.logObserverAttached = "true";
  }
})();
</script>
""",
        unsafe_allow_javascript=True,
    )
else:
    st.info("No logs yet.")
col_refresh, col_clear = st.columns(2)
with col_refresh:
    if st.button("Refresh logs"):
        st.auto_refresh = True
        st.rerun()
with col_clear:
    if st.button("Clear logs"):
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("")
        st.rerun()

trade_rows = read_trade_log(trade_log_path, 20)
if trade_rows:
    st.subheader("Recent trades")
    st.dataframe(trade_rows, use_container_width=True)
else:
    st.caption("No trades yet.")

st.divider()
st.subheader("Bot settings")
st.caption("Stop the bot before changing settings.")

ordered_keys = list(ALLOWED_KEYS)

if not ordered_keys:
    st.info("No .env file found. Create one with at least POLY_PRIVATE_KEY.")

with st.form("env_form"):
    updated: dict[str, str] = {}
    for key in ordered_keys:
        value = env_map.get(key, "")
        if key in BOOL_KEYS:
            truthy = str(value).strip().lower() in {"1", "true", "yes", "y", "on"}
            new_value = st.selectbox(
                key,
                ["true", "false"],
                index=0 if truthy else 1,
            )
        elif key == MODE_KEY:
            current = str(value).strip().lower() if value else "maker"
            options = ["maker"]
            if current not in options:
                options = [current] + options
            new_value = st.selectbox(
                key,
                options,
                index=options.index(current),
            )
        elif key == RECYCLE_MODE_KEY:
            current = str(value).strip().lower() if value else "hold"
            options = ["hold", "sell"]
            if current not in options:
                options = [current] + options
            new_value = st.selectbox(
                key,
                options,
                index=options.index(current),
            )
        elif key == LOG_LEVEL_KEY:
            current = str(value).strip().upper() if value else "INFO"
            options = ["INFO", "DEBUG"]
            if current not in options:
                options = [current] + options
            new_value = st.selectbox(
                key,
                options,
                index=options.index(current),
            )
        elif key in SECRET_KEYS:
            new_value = st.text_input(key, value, type="password")
        else:
            new_value = st.text_input(key, value)
        updated[key] = new_value
    submitted = st.form_submit_button("Save")
    if submitted:
        save_env_file(ENV_PATH, updated, env_lines)
        st.success("Saved .env")
        st.rerun()

if running:
    time.sleep(2)
    st.rerun()
