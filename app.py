# app.py
# Regime Radar (Simple UI): Bullish / Bearish / Chop
# Run:
#   pip install streamlit yfinance pandas numpy
#   streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(page_title="Regime Radar", layout="wide")
st.title("📈 Regime Radar")
st.caption("Simple UI. Smart engine. Outputs: Bullish / Bearish / Chop.")

# ----------------------------
# UI (minimal)
# ----------------------------
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    ticker = st.text_input("Ticker", value="SPY").upper().strip()
with col2:
    timeframe = st.selectbox("Timeframe", ["Daily", "4H", "1H"], index=0)
with col3:
    lookback = st.selectbox("Lookback", ["6mo", "1y", "2y", "5y"], index=1)

show_table = st.toggle("Show table", value=False)

# ----------------------------
# Fixed settings (no sliders)
# ----------------------------
EMA_FAST = 50
EMA_SLOW = 200
ADX_LEN = 14
ADX_TREND_TH = 20

CHOP_LEN = 14
CHOP_CHOPPY_TH = 61
CHOP_TREND_TH = 38

INTERVAL_MAP = {"Daily": "1d", "4H": "4h", "1H": "1h"}

# ----------------------------
# Data + indicators
# ----------------------------
@st.cache_data(show_spinner=False)
def get_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    # yfinance can return MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna()
    return df


def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """
    Wilder-style ADX (trend strength).
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    up_move = high.diff()
    down_move = -low.diff()

    up = up_move.to_numpy()
    down = down_move.to_numpy()

    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    tr1 = (high - low).to_numpy()
    tr2 = (high - close.shift()).abs().to_numpy()
    tr3 = (low - close.shift()).abs().to_numpy()

    tr = np.nanmax(np.vstack([tr1, tr2, tr3]), axis=0)
    tr = pd.Series(tr, index=df.index)

    atr = tr.ewm(alpha=1 / n, adjust=False).mean()

    plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr)

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def choppiness_index(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """
    CHOP (range/chop detector).
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr_sum = tr.rolling(n).sum()
    hh = high.rolling(n).max()
    ll = low.rolling(n).min()
    denom = (hh - ll).replace(0, np.nan)

    return 100 * np.log10(atr_sum / denom) / np.log10(n)


def classify_regime(df: pd.DataFrame) -> dict:
    df = df.copy()

    df["EMA_FAST"] = ema(df["Close"], EMA_FAST)
    df["EMA_SLOW"] = ema(df["Close"], EMA_SLOW)
    df["ADX"] = adx(df, ADX_LEN)
    df["CHOP"] = choppiness_index(df, CHOP_LEN)

    dfx = df.dropna()
    last = dfx.iloc[-1]

    is_chop = last["CHOP"] >= CHOP_CHOPPY_TH
    is_trend = (last["CHOP"] <= CHOP_TREND_TH) and (last["ADX"] >= ADX_TREND_TH)

    ema_bull = (last["Close"] > last["EMA_SLOW"]) and (last["EMA_FAST"] > last["EMA_SLOW"])
    ema_bear = (last["Close"] < last["EMA_SLOW"]) and (last["EMA_FAST"] < last["EMA_SLOW"])

    reasons = []
    reasons.append(f"EMA50 vs EMA200: {last['EMA_FAST']:.2f} vs {last['EMA_SLOW']:.2f}")
    reasons.append(f"ADX({ADX_LEN}) = {last['ADX']:.2f} (trend ≥ {ADX_TREND_TH})")
    reasons.append(f"CHOP({CHOP_LEN}) = {last['CHOP']:.2f} (chop ≥ {CHOP_CHOPPY_TH}, trend ≤ {CHOP_TREND_TH})")

    if is_chop:
        regime = "CHOP"
        allowed = ["Smaller size", "Defined-risk only", "Avoid forcing spreads"]
        reasons.insert(0, "CHOP is high → market is range/chop.")
    elif is_trend and ema_bull:
        regime = "BULLISH"
        allowed = ["Put spreads allowed", "Cash-secured puts allowed", "Buy-writes OK"]
        reasons.insert(0, "Trend confirmed + bullish EMA alignment.")
    elif is_trend and ema_bear:
        regime = "BEARISH"
        allowed = ["Call spreads allowed", "Avoid put-selling unless base/reclaim", "Hedges OK"]
        reasons.insert(0, "Trend confirmed + bearish EMA alignment.")
    else:
        regime = "CHOP"
        allowed = ["Transition zone → treat as chop", "Wait for confirmation", "Smaller size"]
        reasons.insert(0, "Mixed signals → treat as chop for safety.")

    return {"df": dfx, "last": last, "regime": regime, "allowed": allowed, "reasons": reasons}


# ----------------------------
# App run
# ----------------------------
if not ticker:
    st.info("Type a ticker to start.")
    st.stop()

interval = INTERVAL_MAP[timeframe]
df = get_data(ticker, lookback, interval)

if df.empty:
    st.error("No data returned. Check ticker or timeframe.")
    st.stop()

# Need enough rows for EMA200 and indicators
if len(df) < 250:
    st.warning("Not enough candles for EMA200 + indicators. Try a longer lookback (2y/5y) or Daily.")
    st.stop()

out = classify_regime(df)
last = out["last"]

# Big label
if out["regime"] == "BULLISH":
    st.success(f"✅ {ticker} Regime: **BULLISH**")
elif out["regime"] == "BEARISH":
    st.error(f"🟥 {ticker} Regime: **BEARISH**")
else:
    st.warning(f"🟨 {ticker} Regime: **CHOP**")

# Quick metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Close", f"{last['Close']:.2f}")
m2.metric("EMA50", f"{last['EMA_FAST']:.2f}")
m3.metric("EMA200", f"{last['EMA_SLOW']:.2f}")
m4.metric("ADX / CHOP", f"{last['ADX']:.1f} / {last['CHOP']:.1f}")

# Playbook
st.subheader("Playbook")
st.write("• " + "\n• ".join(out["allowed"]))

# Why
with st.expander("Why this regime?"):
    st.write("• " + "\n• ".join(out["reasons"]))

# Charts
st.subheader("Charts")
chart = out["df"][["Close", "EMA_FAST", "EMA_SLOW", "ADX", "CHOP"]].tail(300)

cA, cB = st.columns(2)
with cA:
    st.markdown("**Price + EMAs**")
    st.line_chart(chart[["Close", "EMA_FAST", "EMA_SLOW"]])
with cB:
    st.markdown("**ADX + CHOP**")
    st.line_chart(chart[["ADX", "CHOP"]])

if show_table:
    st.subheader("Latest rows")
    st.dataframe(chart.tail(60))
