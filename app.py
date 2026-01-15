# app.py
# Streamlit v1: SPY Morning Signal Terminal (Phase 1)
# - Uses SPY daily candles + VIX as IV proxy (no option chain)
# - Outputs: TRADE / NO TRADE, CALL/PUT bias, expiry window (30–45 DTE), time-stop date
# - Logs trades to trades.csv (local)
#
# Run:
#   pip install streamlit yfinance pandas numpy
#   streamlit run app.py

import math
from datetime import datetime, timedelta, date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

APP_TITLE = "SPY Morning Signal Terminal (Phase 1)"
LOG_PATH = Path("trades.csv")


# -----------------------------
# Indicator helpers
# -----------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def realized_vol(log_returns: pd.Series, window: int, annualization: int = 252) -> pd.Series:
    # Annualized realized vol using rolling std of log returns
    return log_returns.rolling(window).std() * math.sqrt(annualization)


def bollinger_bandwidth(close: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.Series:
    mid = close.rolling(window).mean()
    sd = close.rolling(window).std()
    upper = mid + n_std * sd
    lower = mid - n_std * sd
    # normalized bandwidth
    return (upper - lower) / mid


def rolling_percentile_rank(series: pd.Series, lookback: int = 252) -> pd.Series:
    """
    Percentile rank (0..100) of the latest value within a rolling window.
    Lower = "cheaper / more compressed" relative to its own history.
    """
    def _pct_rank(x):
        s = pd.Series(x)
        # rank(pct=True) gives 0..1, multiply by 100
        return 100.0 * s.rank(pct=True).iloc[-1]

    return series.rolling(lookback).apply(_pct_rank, raw=False)


def safe_last_row(df: pd.DataFrame) -> pd.Series:
    d = df.dropna()
    if len(d) == 0:
        raise ValueError("Not enough data to compute indicators yet.")
    return d.iloc[-1]


# -----------------------------
# Data loaders (cached)
# -----------------------------
@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_daily(symbol: str, period: str = "2y") -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval="1d", auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise ValueError(f"No data returned for {symbol}.")
    df = df.rename(columns=str.lower)
    # yfinance returns timezone-aware index sometimes; make it date-like
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


# -----------------------------
# Core signal logic
# -----------------------------
def compute_signal(
    spy_ohlc: pd.DataFrame,
    vix_close: pd.Series,
    *,
    vix_pct_thresh: float,
    bb_width_pct_thresh: float,
    cooldown_days: int,
    time_stop_days: int,
    direction_rule: str,
    last_trade_date: date | None
) -> dict:
    df = spy_ohlc.copy()
    df["logret"] = np.log(df["close"]).diff()

    # RV
    df["rv_5"] = realized_vol(df["logret"], 5)
    df["rv_20"] = realized_vol(df["logret"], 20)

    # Compression
    df["bb_width"] = bollinger_bandwidth(df["close"], 20, 2.0)
    df["bb_width_pct"] = rolling_percentile_rank(df["bb_width"], 252)

    # Trend / direction
    df["ema_20"] = ema(df["close"], 20)

    # VIX proxy for IV cheapness
    vix = vix_close.reindex(df.index).ffill()
    df["vix"] = vix
    df["vix_pct"] = rolling_percentile_rank(df["vix"], 252)

    latest = safe_last_row(df)

    # Cooldown gate (based on calendar days)
    if last_trade_date is not None:
        days_since = (date.today() - last_trade_date).days
        cooldown_block = days_since < cooldown_days
    else:
        days_since = None
        cooldown_block = False

    # Conditions
    vix_ok = latest["vix_pct"] < vix_pct_thresh
    compression_ok = latest["bb_width_pct"] < bb_width_pct_thresh
    rv_rising = latest["rv_5"] > latest["rv_20"]

    enter = bool(vix_ok and compression_ok and rv_rising and (not cooldown_block))

    # Direction
    if direction_rule == "EMA20":
        option_type = "CALL" if latest["close"] > latest["ema_20"] else "PUT"
    else:
        # fallback (still EMA20)
        option_type = "CALL" if latest["close"] > latest["ema_20"] else "PUT"

    # Dates you care about
    today = date.today()
    earliest_exp = today + timedelta(days=30)
    latest_exp = today + timedelta(days=45)
    time_stop_date = today + timedelta(days=time_stop_days)

    return {
        "enter": enter,
        "option_type": option_type,
        "expiry_window": (earliest_exp, latest_exp),
        "time_stop_days": time_stop_days,
        "time_stop_date": time_stop_date,
        "cooldown_block": cooldown_block,
        "days_since_last_trade": days_since,
        "metrics": {
            "spot": float(latest["close"]),
            "vix": float(latest["vix"]),
            "vix_pct": float(latest["vix_pct"]),
            "bb_width_pct": float(latest["bb_width_pct"]),
            "rv_5": float(latest["rv_5"]),
            "rv_20": float(latest["rv_20"]),
            "ema_20": float(latest["ema_20"]),
        },
        "passes": {
            "vix_ok": bool(vix_ok),
            "compression_ok": bool(compression_ok),
            "rv_rising": bool(rv_rising),
        },
        "df": df,  # for optional display
    }


# -----------------------------
# Logging
# -----------------------------
def read_log() -> pd.DataFrame:
    if LOG_PATH.exists():
        try:
            return pd.read_csv(LOG_PATH)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def append_log(row: dict) -> None:
    df = read_log()
    new = pd.DataFrame([row])
    if df is None or df.empty:
        out = new
    else:
        out = pd.concat([df, new], ignore_index=True)
    out.to_csv(LOG_PATH, index=False)


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("SPY daily + VIX (IV proxy) → cheap vol + compression + RV rising → CALL/PUT bias + rules. No option chain needed.")

# Sidebar settings
st.sidebar.header("Settings")

vix_pct_thresh = st.sidebar.slider("VIX percentile threshold (cheap vol)", min_value=5, max_value=50, value=25, step=1)
bb_width_pct_thresh = st.sidebar.slider("Compression threshold (BB width percentile)", min_value=5, max_value=50, value=20, step=1)

cooldown_days = st.sidebar.slider("Cooldown days (no new trades)", min_value=0, max_value=30, value=8, step=1)
time_stop_days = st.sidebar.slider("Time stop (exit after N days)", min_value=3, max_value=20, value=10, step=1)

direction_rule = st.sidebar.selectbox("Direction rule", options=["EMA20"], index=0)

st.sidebar.divider()
st.sidebar.subheader("Manual inputs")

log_df = read_log()
if not log_df.empty and "trade_date" in log_df.columns:
    # Parse last trade date from log
    try:
        log_df["trade_date"] = pd.to_datetime(log_df["trade_date"]).dt.date
        last_trade_date = max(log_df["trade_date"].dropna()) if len(log_df["trade_date"].dropna()) else None
    except Exception:
        last_trade_date = None
else:
    last_trade_date = None

last_trade_date_override = st.sidebar.date_input(
    "Last trade date (override)",
    value=last_trade_date if last_trade_date is not None else date.today() - timedelta(days=999),
)
use_override = st.sidebar.checkbox("Use override as last trade date", value=False)
effective_last_trade = last_trade_date_override if use_override else last_trade_date

if effective_last_trade is None:
    st.sidebar.info("No prior trade date detected. Cooldown will not block entries.")
else:
    st.sidebar.write(f"Effective last trade date: **{effective_last_trade}**")

# Load data
with st.spinner("Loading SPY + VIX data..."):
    spy = load_daily("SPY", period="2y")
    vix = load_daily("^VIX", period="2y")

vix_close = vix["close"]

# Compute signal
try:
    result = compute_signal(
        spy,
        vix_close,
        vix_pct_thresh=float(vix_pct_thresh),
        bb_width_pct_thresh=float(bb_width_pct_thresh),
        cooldown_days=int(cooldown_days),
        time_stop_days=int(time_stop_days),
        direction_rule=str(direction_rule),
        last_trade_date=effective_last_trade,
    )
except Exception as e:
    st.error(f"Signal error: {e}")
    st.stop()

# Topline badge
colA, colB, colC, colD = st.columns([1.2, 1, 1.2, 1.6])

enter = result["enter"]
badge = "✅ TRADE" if enter else "🧊 NO TRADE"
colA.markdown(f"## {badge}")

colB.metric("Bias", result["option_type"])
spot = result["metrics"]["spot"]
colC.metric("SPY Close (last)", f"{spot:.2f}")

earliest_exp, latest_exp = result["expiry_window"]
colD.markdown(
    f"**Expiry window:** {earliest_exp.isoformat()} → {latest_exp.isoformat()}  \n"
    f"**Time stop:** exit by **{result['time_stop_date'].isoformat()}** (≈ {result['time_stop_days']} days)"
)

# Explain why (pass/fail)
st.subheader("Why")
p = result["passes"]
m = result["metrics"]

why_cols = st.columns(3)
why_cols[0].markdown(
    f"**Cheap vol (VIX percentile)**: `{m['vix_pct']:.1f}` {'✅' if p['vix_ok'] else '❌'}  \n"
    f"VIX last: `{m['vix']:.2f}`"
)
why_cols[1].markdown(
    f"**Compression (BB width percentile)**: `{m['bb_width_pct']:.1f}` {'✅' if p['compression_ok'] else '❌'}"
)
why_cols[2].markdown(
    f"**RV rising**: RV5 `{m['rv_5']:.3f}` vs RV20 `{m['rv_20']:.3f}` {'✅' if p['rv_rising'] else '❌'}"
)

# Cooldown
if result["cooldown_block"]:
    st.warning(
        f"Cooldown active: last trade was {result['days_since_last_trade']} days ago "
        f"(need {cooldown_days}). Signal blocked even if conditions pass."
    )

# Trade checklist (manual strike)
st.subheader("If TRADE = ✅, do this manually (Robinhood)")
st.markdown(
    f"- Underlying: **SPY**\n"
    f"- Direction: **{result['option_type']}** (rule: close vs EMA20)\n"
    f"- Expiration: pick a date between **30–45 DTE** (≈ {earliest_exp} to {latest_exp})\n"
    f"- Strike: choose the contract closest to **~25 delta** (manual)\n"
    f"- Management: set a reminder to exit by **{result['time_stop_date']}** if it’s not working\n"
)

# Risk helper
st.subheader("Risk helper (premium budget)")
risk_col1, risk_col2, risk_col3 = st.columns([1, 1, 2])
acct_size = risk_col1.number_input("Account size ($)", min_value=0.0, value=6000.0, step=100.0)
risk_pct = risk_col2.number_input("Max premium risk %", min_value=0.0, value=0.25, step=0.05)
premium_budget = acct_size * (risk_pct / 100.0)
risk_col3.metric("Max premium budget", f"${premium_budget:,.2f}")

# Log trade section
st.subheader("Log a trade (optional but OP for learning)")
log_cols = st.columns([1, 1, 1, 1, 1, 1.2])

trade_date = log_cols[0].date_input("Trade date", value=date.today())
expiry_chosen = log_cols[1].date_input("Expiry chosen", value=earliest_exp)
call_put = log_cols[2].selectbox("Type", options=["CALL", "PUT"], index=0 if result["option_type"] == "CALL" else 1)
delta_est = log_cols[3].number_input("Delta (manual)", min_value=-1.0, max_value=1.0, value=0.25 if call_put == "CALL" else -0.25, step=0.01)
premium_paid = log_cols[4].number_input("Premium paid ($)", min_value=0.0, value=0.0, step=10.0)

notes = log_cols[5].text_input("Notes", value="")

if st.button("➕ Add to trades.csv"):
    row = {
        "trade_date": trade_date.isoformat(),
        "entered_by_signal": bool(enter),
        "bias": result["option_type"],
        "type": call_put,
        "expiry": expiry_chosen.isoformat(),
        "delta_manual": float(delta_est),
        "premium_paid": float(premium_paid),
        "spot_close": float(spot),
        "vix": float(m["vix"]),
        "vix_pct": float(m["vix_pct"]),
        "bb_width_pct": float(m["bb_width_pct"]),
        "rv5": float(m["rv_5"]),
        "rv20": float(m["rv_20"]),
        "time_stop_date": result["time_stop_date"].isoformat(),
        "notes": notes,
    }
    append_log(row)
    st.success("Logged ✅")

# Show log
st.subheader("Trade log")
log_df2 = read_log()
if log_df2.empty:
    st.info("No trades logged yet. (It’ll create trades.csv once you add one.)")
else:
    st.dataframe(log_df2.tail(50), use_container_width=True)

# Optional: show last N rows of computed dataframe
with st.expander("Debug: last 30 rows of indicators"):
    d = result["df"].copy()
    cols = ["close", "ema_20", "vix", "vix_pct", "bb_width_pct", "rv_5", "rv_20"]
    st.dataframe(d[cols].tail(30), use_container_width=True)
