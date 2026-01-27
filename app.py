# app.py
# Regime + Chop + VIX + Range Location (Minimal UI, System-Style)
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
st.title("Regime + Chop System")
st.caption("Minimal UI. Uses MA Regime + Chop (band/cross) + VIX + ATR Range Location.")

# -------------------------------------------------
# Minimal UI
# -------------------------------------------------
ticker = st.text_input("Ticker", value="SPY").upper().strip()
use_vix = st.toggle("Use VIX filter (SPY/Index ETFs)", value=True)
show_table = st.toggle("Show table", value=False)

# -------------------------------------------------
# Fixed system parameters (no sliders)
# -------------------------------------------------
# Regime
MA_FAST = 20
MA_SLOW = 50
DEAD_ZONE = 0.001  # 0.10% dead-zone to reduce flip-flopping

# Chop detection
DIST_BAND = 0.006         # 0.6% distance from MA_SLOW
CROSS_LOOKBACK = 10       # days to count crosses around MA_SLOW
CROSS_THRESHOLD = 3       # crosses >= this => chop

# VIX filter
VIX_THRESHOLD = 20

# Range / ATR
RANGE_LOOKBACK = 20
ATR_LOOKBACK = 14
ATR_BUFFER = 0.25         # 25% of ATR from range edges

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
    return df.dropna()

@st.cache_data(ttl=900, show_spinner=False)
def load_vix() -> pd.DataFrame:
    v = yf.download("^VIX", period="60d", interval="1d", auto_adjust=True, progress=False)
    if v is None or v.empty:
        return pd.DataFrame()
    if isinstance(v.columns, pd.MultiIndex):
        v.columns = v.columns.get_level_values(0)
    return v.dropna()

if not ticker:
    st.info("Type a ticker to start.")
    st.stop()

df = load_price_data(ticker)
if df.empty or len(df) < 120:
    st.error("Insufficient data loaded. Try a more liquid ticker.")
    st.stop()

vix_df = load_vix() if use_vix else pd.DataFrame()

# -------------------------------------------------
# Indicators
# -------------------------------------------------
df = df.copy()
df["MA_FAST"] = df["Close"].ewm(span=MA_FAST, adjust=False).mean()
df["MA_SLOW"] = df["Close"].ewm(span=MA_SLOW, adjust=False).mean()

close = float(df["Close"].iloc[-1])
ma_fast = float(df["MA_FAST"].iloc[-1])
ma_slow = float(df["MA_SLOW"].iloc[-1])

# -------------------------------------------------
# Regime (EMA20/EMA50 + dead zone)
# -------------------------------------------------
diff = (ma_fast - ma_slow) / ma_slow
if diff > DEAD_ZONE:
    regime = "BULLISH"
elif diff < -DEAD_ZONE:
    regime = "BEARISH"
else:
    regime = "CHOP"  # neutral/transition treated as chop

# -------------------------------------------------
# Chop detection (distance band + cross count around MA_SLOW)
# -------------------------------------------------
distance_pct = abs(close - ma_slow) / ma_slow
chop_distance = distance_pct < DIST_BAND

recent = df.tail(CROSS_LOOKBACK + 1).dropna(subset=["Close", "MA_SLOW"])
above = recent["Close"] > recent["MA_SLOW"]
crosses = int((above != above.shift()).sum() - 1)
chop_cross = crosses >= CROSS_THRESHOLD

chop = bool(chop_distance or chop_cross)

# -------------------------------------------------
# VIX filter
# -------------------------------------------------
vix_value = None
if use_vix:
    if vix_df.empty:
        st.warning("VIX data not available. VIX filter will be ignored.")
        vol_ok = True
    else:
        vix_value = float(vix_df["Close"].iloc[-1])
        vol_ok = vix_value < VIX_THRESHOLD
else:
    vol_ok = True

# -------------------------------------------------
# ATR + Range Location
# -------------------------------------------------
high = df["High"]
low = df["Low"]
prev_close = df["Close"].shift()

tr = pd.concat(
    [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
    axis=1
).max(axis=1)

df["ATR"] = tr.rolling(ATR_LOOKBACK).mean()

support = float(low.rolling(RANGE_LOOKBACK).min().iloc[-1])
resistance = float(high.rolling(RANGE_LOOKBACK).max().iloc[-1])
atr = float(df["ATR"].iloc[-1])

near_support = close <= support + ATR_BUFFER * atr
near_resistance = close >= resistance - ATR_BUFFER * atr
mid_range = not (near_support or near_resistance)

location = "Mid-Range"
if near_support:
    location = "Near Support"
elif near_resistance:
    location = "Near Resistance"

# -------------------------------------------------
# Final decision logic (system switch)
# -------------------------------------------------
system_on = vol_ok and (not chop) and (not mid_range)

direction = "NO TRADE"
if system_on:
    if regime == "BULLISH" and near_support:
        direction = "PUT CREDIT SPREADS"
    elif regime == "BEARISH" and near_resistance:
        direction = "CALL CREDIT SPREADS"
    else:
        system_on = False
        direction = "NO TRADE (Location/Regime mismatch)"

# -------------------------------------------------
# UI OUTPUT
# -------------------------------------------------
st.subheader("Market Snapshot")

c1, c2, c3 = st.columns(3)
c1.metric("Close", f"{close:,.2f}")
c2.metric(f"EMA {MA_FAST}/{MA_SLOW}", f"{ma_fast:,.2f} / {ma_slow:,.2f}")
if use_vix:
    c3.metric("VIX", "—" if vix_value is None else f"{vix_value:,.2f}")
else:
    c3.metric("VIX", "OFF")

st.write("### System Filters")
st.write(f"- Regime: **{regime}** (EMA{MA_FAST} vs EMA{MA_SLOW}, dead-zone {DEAD_ZONE*100:.2f}%)")
st.write(f"- Volatility OK: **{'YES' if vol_ok else 'NO'}**" + (f" (VIX<{VIX_THRESHOLD})" if use_vix and vix_value is not None else ""))
st.write(f"- Chop Detected: **{'YES' if chop else 'NO'}**")
st.write(f"- Distance from EMA{MA_SLOW}: **{distance_pct*100:.2f}%** (band < {DIST_BAND*100:.2f}%)")
st.write(f"- Crosses around EMA{MA_SLOW} ({CROSS_LOOKBACK}d): **{crosses}** (threshold ≥ {CROSS_THRESHOLD})")

st.write("### Range Location")
st.write(f"- Support ({RANGE_LOOKBACK}d): **{support:,.2f}**")
st.write(f"- Resistance ({RANGE_LOOKBACK}d): **{resistance:,.2f}**")
st.write(f"- ATR ({ATR_LOOKBACK}): **{atr:,.2f}**")
st.write(f"- Location: **{location}**")

st.write("---")

if system_on:
    st.success(f"SYSTEM ON → {direction}")
else:
    st.error(f"SYSTEM OFF → {direction}")

# Charts (simple)
st.write("### Charts")
chart = df[["Close", "MA_FAST", "MA_SLOW"]].dropna().tail(260)
st.line_chart(chart)

if show_table:
    st.write("### Latest rows")
    st.dataframe(df.tail(60))

st.caption("Educational tool only. Not financial advice.")
