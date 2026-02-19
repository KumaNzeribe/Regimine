# app.py
# Regime + Trend Strength + Chop + VIX Band + Extension Trigger (Minimal UI, System-Style)
#
# Run:
#   pip install streamlit yfinance pandas numpy
#   streamlit run app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# -------------------------------------------------
# Page setup
# -------------------------------------------------
st.set_page_config(page_title="Regime System", layout="centered")
st.title("Regime + Trend System")
st.caption("Minimal UI. EMA Regime + ADX Trend + Chop + VIX Band + ATR Extension Trigger.")

# -------------------------------------------------
# Minimal UI
# -------------------------------------------------
ticker = st.text_input("Ticker", value="SPY").upper().strip()
use_vix = st.toggle("Use VIX band filter (Index ETFs)", value=True)
show_table = st.toggle("Show table", value=False)

# -------------------------------------------------
# Fixed system parameters (no sliders)
# -------------------------------------------------
# Regime (EMA20/EMA50 + dead zone)
MA_FAST = 20
MA_SLOW = 50
DEAD_ZONE = 0.001  # 0.10% dead-zone to reduce flip-flopping

# Trend strength (ADX)
ADX_LOOKBACK = 14
ADX_MIN = 18

# Chop detection (distance band + cross count around EMA_SLOW)
DIST_BAND = 0.006         # 0.6% distance from EMA_SLOW
CROSS_LOOKBACK = 10       # days to count crosses around EMA_SLOW
CROSS_THRESHOLD = 3       # crosses >= this => chop-ish

# VIX band (income-optimized but risk-aware)
VIX_LOW = 14
VIX_HIGH = 28

# ATR / Extension trigger
ATR_LOOKBACK = 14
EXT_THRESH = 0.75         # extension in ATR units vs EMA20

# Data
PERIOD = "2y"
INTERVAL = "1d"

# -------------------------------------------------
# Data loading
# -------------------------------------------------
@st.cache_data(ttl=900, show_spinner=False)
def load_price_data(t: str) -> pd.DataFrame:
    df = yf.download(t, period=PERIOD, interval=INTERVAL, auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    return df

@st.cache_data(ttl=900, show_spinner=False)
def load_vix() -> pd.DataFrame:
    v = yf.download("^VIX", period="90d", interval="1d", auto_adjust=True, progress=False)
    if v is None or v.empty:
        return pd.DataFrame()
    if isinstance(v.columns, pd.MultiIndex):
        v.columns = v.columns.get_level_values(0)
    return v.dropna()

# -------------------------------------------------
# Helper indicators
# -------------------------------------------------
def true_range(df: pd.DataFrame) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1
    ).max(axis=1)
    return tr

def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)

    tr = true_range(df)
    atr_ = tr.rolling(n).mean()

    plus_di = 100 * plus_dm.rolling(n).mean() / atr_
    minus_di = 100 * minus_dm.rolling(n).mean() / atr_

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan)
    return dx.rolling(n).mean()

# -------------------------------------------------
# Guardrails
# -------------------------------------------------
if not ticker:
    st.info("Type a ticker to start.")
    st.stop()

df = load_price_data(ticker)
if df.empty or len(df) < 160:
    st.error("Insufficient data loaded. Try a more liquid ticker or check symbol.")
    st.stop()

vix_df = load_vix() if use_vix else pd.DataFrame()

# -------------------------------------------------
# Compute indicators
# -------------------------------------------------
df = df.copy()
df["EMA_FAST"] = df["Close"].ewm(span=MA_FAST, adjust=False).mean()
df["EMA_SLOW"] = df["Close"].ewm(span=MA_SLOW, adjust=False).mean()

# ATR
df["TR"] = true_range(df)
df["ATR"] = df["TR"].rolling(ATR_LOOKBACK).mean()

# ADX
df["ADX"] = adx(df, ADX_LOOKBACK)

# Latest values
close = float(df["Close"].iloc[-1])
ema_fast = float(df["EMA_FAST"].iloc[-1])
ema_slow = float(df["EMA_SLOW"].iloc[-1])
atr = float(df["ATR"].iloc[-1]) if not np.isnan(df["ATR"].iloc[-1]) else np.nan
adx_value = float(df["ADX"].iloc[-1]) if not np.isnan(df["ADX"].iloc[-1]) else np.nan

# -------------------------------------------------
# Regime (EMA20/EMA50 + dead zone)
# -------------------------------------------------
diff = (ema_fast - ema_slow) / ema_slow
if diff > DEAD_ZONE:
    regime = "BULLISH"
elif diff < -DEAD_ZONE:
    regime = "BEARISH"
else:
    regime = "NEUTRAL"

# -------------------------------------------------
# Trend gate (ADX)
# -------------------------------------------------
trend_ok = (not np.isnan(adx_value)) and (adx_value >= ADX_MIN)

# -------------------------------------------------
# Chop detection (distance band + cross count around EMA_SLOW)
#   - Use ADX as primary: if trend_ok is False => chop
#   - Otherwise, require BOTH cross + distance to call it chop (reduces false chop in trends)
# -------------------------------------------------
distance_pct = abs(close - ema_slow) / ema_slow
chop_distance = distance_pct < DIST_BAND

recent = df.tail(CROSS_LOOKBACK + 1).dropna(subset=["Close", "EMA_SLOW"])
above = recent["Close"] > recent["EMA_SLOW"]
crosses = int((above != above.shift()).sum() - 1) if len(above) > 1 else 0
chop_cross = crosses >= CROSS_THRESHOLD

chop = (not trend_ok) or (chop_cross and chop_distance)

# -------------------------------------------------
# VIX band filter
# -------------------------------------------------
vix_value = None
if use_vix:
    if vix_df.empty:
        st.warning("VIX data not available. VIX filter will be ignored.")
        vol_ok = True
    else:
        vix_value = float(vix_df["Close"].iloc[-1])
        vol_ok = (vix_value >= VIX_LOW) and (vix_value <= VIX_HIGH)
else:
    vol_ok = True

# -------------------------------------------------
# Extension trigger (ATR-normalized distance vs EMA20)
# -------------------------------------------------
extension = np.nan
if (not np.isnan(atr)) and atr != 0:
    extension = (close - ema_fast) / atr

pullback_ok = (regime == "BULLISH") and (not np.isnan(extension)) and (extension <= -EXT_THRESH)
rally_ok = (regime == "BEARISH") and (not np.isnan(extension)) and (extension >= EXT_THRESH)

# -------------------------------------------------
# Final decision logic (system switch)
# -------------------------------------------------
# System ON when:
# - VIX in band (or VIX off)
# - Trend is strong enough (ADX gate via chop)
# - Not chop
system_on = vol_ok and (not chop)

direction = "NO TRADE"
reason = ""

if system_on:
    if pullback_ok:
        direction = "PUT CREDIT SPREADS"
        reason = f"Bull regime + pullback (ext ≤ -{EXT_THRESH})"
    elif rally_ok:
        direction = "CALL CREDIT SPREADS"
        reason = f"Bear regime + rally (ext ≥ +{EXT_THRESH})"
    else:
        system_on = False
        direction = "NO TRADE"
        reason = "No extension trigger"
else:
    # Helpful reason string
    blockers = []
    if not vol_ok:
        blockers.append("VIX out of band")
    if chop:
        blockers.append("Chop/No-trend")
    reason = ", ".join(blockers) if blockers else "Filters not met"

# -------------------------------------------------
# UI OUTPUT
# -------------------------------------------------
st.subheader("Market Snapshot")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Close", f"{close:,.2f}")
c2.metric(f"EMA {MA_FAST}/{MA_SLOW}", f"{ema_fast:,.2f} / {ema_slow:,.2f}")
c3.metric("ADX", "—" if np.isnan(adx_value) else f"{adx_value:,.1f}")
c4.metric("Ext (ATR vs EMA20)", "—" if np.isnan(extension) else f"{extension:,.2f}")

st.write("### System Filters")
st.write(f"- Regime: **{regime}** (EMA{MA_FAST} vs EMA{MA_SLOW}, dead-zone {DEAD_ZONE*100:.2f}%)")
st.write(f"- Trend OK (ADX ≥ {ADX_MIN}): **{'YES' if trend_ok else 'NO'}**")
st.write(f"- VIX Filter: **{'ON' if use_vix else 'OFF'}**" + (
    f" (VIX {vix_value:,.2f} in [{VIX_LOW}, {VIX_HIGH}])" if use_vix and vix_value is not None else ""
))
st.write(f"- Volatility OK: **{'YES' if vol_ok else 'NO'}**")
st.write(f"- Chop Detected: **{'YES' if chop else 'NO'}**")
st.write(f"- Distance from EMA{MA_SLOW}: **{distance_pct*100:.2f}%** (band < {DIST_BAND*100:.2f}%)")
st.write(f"- Crosses around EMA{MA_SLOW} ({CROSS_LOOKBACK}d): **{crosses}** (threshold ≥ {CROSS_THRESHOLD})")
st.write(f"- ATR ({ATR_LOOKBACK}): **{'—' if np.isnan(atr) else f'{atr:,.2f}'}**")
st.write(f"- Extension trigger threshold: **±{EXT_THRESH} ATR**")

st.write("---")

if system_on:
    st.success(f"SYSTEM ON → {direction}  \n**Reason:** {reason}")
else:
    st.error(f"SYSTEM OFF → {direction}  \n**Reason:** {reason}")

# Charts (simple)
st.write("### Charts")
chart = df[["Close", "EMA_FAST", "EMA_SLOW"]].dropna().tail(260)
st.line_chart(chart)

if show_table:
    st.write("### Latest rows")
    st.dataframe(df.tail(80))

st.caption("Educational tool only. Not financial advice.")
