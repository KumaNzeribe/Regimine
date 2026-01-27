# app.py
# Streamlit Regime Radar w/ CHOP (Choppiness Index) + ADX + EMA alignment
# Run:
#   pip install streamlit yfinance pandas numpy
#   streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(page_title="Regime Radar", layout="wide")
st.title("📈 Regime Radar (Bullish / Bearish / Chop)")

# ----------------------------
# Inputs
# ----------------------------
ticker = st.text_input("Ticker", value="SPY").upper().strip()

colA, colB, colC = st.columns(3)
with colA:
    period = st.selectbox("Lookback", ["6mo", "1y", "2y", "5y"], index=1)
with colB:
    interval = st.selectbox("Interval", ["1d", "1h", "4h"], index=0)
with colC:
    show_raw = st.toggle("Show raw table", value=False)

st.divider()
st.subheader("Settings")

c1, c2, c3, c4 = st.columns(4)
with c1:
    ema_fast = st.slider("Fast EMA", 10, 100, 50)
with c2:
    ema_slow = st.slider("Slow EMA", 100, 300, 200)
with c3:
    adx_len = st.slider("ADX length", 7, 30, 14)
with c4:
    adx_trend_threshold = st.slider("ADX trend threshold", 10, 40, 20)

c5, c6, c7 = st.columns(3)
with c5:
    chop_len = st.slider("CHOP length", 7, 50, 14)
with c6:
    chop_choppy_threshold = st.slider("CHOP choppy threshold", 50, 70, 61)
with c7:
    chop_trend_threshold = st.slider("CHOP trend threshold", 25, 50, 38)

# ----------------------------
# Helpers
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
    df = df.dropna()
    return df

def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()

def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """
    Wilder-style ADX.
    """
    high, low, close = df["High"], df["Low"], df["Close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / n, adjust=False).mean()

    plus_di = 100 * (
        pd.Series(plus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    )
    minus_di = 100 * (
        pd.Series(minus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    )

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).replace(
        [np.inf, -np.inf], np.nan
    )
    adx_val = dx.ewm(alpha=1 / n, adjust=False).mean()
    return adx_val

def choppiness_index(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """
    Choppiness Index (CHOP).
    High CHOP = choppy/ranging, Low CHOP = trending.
    Typical guide:
      CHOP > 61.8 -> chop/range
      CHOP < 38.2 -> trend
    """
    high, low, close = df["High"], df["Low"], df["Close"]

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr_sum = tr.rolling(n).sum()
    hh = high.rolling(n).max()
    ll = low.rolling(n).min()
    denom = (hh - ll).replace(0, np.nan)

    chop = 100 * np.log10(atr_sum / denom) / np.log10(n)
    return chop

def classify_regime(df: pd.DataFrame) -> dict:
    df = df.copy()

    # Indicators
    df["EMA_FAST"] = ema(df["Close"], ema_fast)
    df["EMA_SLOW"] = ema(df["Close"], ema_slow)
    df["ADX"] = adx(df, adx_len)
    df["CHOP"] = choppiness_index(df, chop_len)

    # Latest values
    last = df.iloc[-1]

    # State flags
    is_chop = (not np.isnan(last["CHOP"])) and (last["CHOP"] >= chop_choppy_threshold)
    is_trend = (not np.isnan(last["CHOP"])) and (last["CHOP"] <= chop_trend_threshold) and (
        not np.isnan(last["ADX"]) and last["ADX"] >= adx_trend_threshold
    )

    ema_bull = (last["Close"] > last["EMA_SLOW"]) and (last["EMA_FAST"] > last["EMA_SLOW"])
    ema_bear = (last["Close"] < last["EMA_SLOW"]) and (last["EMA_FAST"] < last["EMA_SLOW"])

    # Regime decision
    if is_chop:
        regime = "CHOP / RANGE"
        allowed = [
            "Small size only",
            "Defined-risk only",
            "Prefer buy-write on stable names or sit out",
        ]
    elif is_trend and ema_bull:
        regime = "BULLISH"
        allowed = [
            "Put credit spreads",
            "Cash-secured puts",
            "Buy-writes (covered calls)",
        ]
    elif is_trend and ema_bear:
        regime = "BEARISH"
        allowed = [
            "Call credit spreads",
            "Bearish hedges",
            "Avoid put-selling unless base/reclaim",
        ]
    else:
        regime = "UNCLEAR / TRANSITION"
        allowed = [
            "Wait for confirmation",
            "Smaller size",
            "Avoid forcing spreads",
        ]

    # Confidence score (simple)
    # Not “truth”, just a quick feel: trend signal strength vs chop signal.
    # Normalize ADX roughly (0..40) and CHOP distance from thresholds.
    adx_score = float(np.clip((last["ADX"] - adx_trend_threshold) / 20.0, 0, 1)) if not np.isnan(last["ADX"]) else 0.0
    chop_score = float(np.clip((last["CHOP"] - chop_choppy_threshold) / 10.0, 0, 1)) if not np.isnan(last["CHOP"]) else 0.0

    return {
        "df": df,
        "regime": regime,
        "last": last,
        "allowed": allowed,
        "flags": {
            "is_chop": is_chop,
            "is_trend": is_trend,
            "ema_bull": ema_bull,
            "ema_bear": ema_bear,
        },
        "scores": {
            "trend_strength": adx_score,
            "chop_strength": chop_score,
        },
    }

# ----------------------------
# Main
# ----------------------------
try:
    if not ticker:
        st.info("Type a ticker to start.")
        st.stop()

    df = get_data(ticker, period, interval)
    if df.empty:
        st.error("No data returned. Check ticker/interval.")
        st.stop()

    out = classify_regime(df)
    last = out["last"]

    # Metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Regime", out["regime"])
    m2.metric("Close", f"{last['Close']:.2f}")
    m3.metric("ADX", "—" if np.isnan(last["ADX"]) else f"{last['ADX']:.2f}")
    m4.metric("CHOP", "—" if np.isnan(last["CHOP"]) else f"{last['CHOP']:.2f}")
    m5.metric(f"EMA {ema_fast} / {ema_slow}", f"{last['EMA_FAST']:.2f} / {last['EMA_SLOW']:.2f}")

    # Flags + quick read
    st.caption(
        f"Flags → chop: **{out['flags']['is_chop']}**, trend: **{out['flags']['is_trend']}**, "
        f"EMA bull: **{out['flags']['ema_bull']}**, EMA bear: **{out['flags']['ema_bear']}**"
    )

    st.subheader("Suggested playbook")
    st.write("• " + "\n• ".join(out["allowed"]))

    # Charts
    st.subheader("Charts")
    chart_df = out["df"][["Close", "EMA_FAST", "EMA_SLOW", "ADX", "CHOP"]].dropna()

    cA, cB = st.columns(2)
    with cA:
        st.markdown("**Price + EMAs**")
        st.line_chart(chart_df[["Close", "EMA_FAST", "EMA_SLOW"]])
    with cB:
        st.markdown("**ADX + CHOP**")
        st.line_chart(chart_df[["ADX", "CHOP"]])

    # Table
    if show_raw:
        st.subheader("Latest rows")
        st.dataframe(chart_df.tail(50))

except Exception as e:
    st.error(f"Error: {e}")
