import streamlit as st
import pandas as pd
import requests
import yfinance as yf
from features import engineer_features, select_features

# Page setup
st.set_page_config(page_title="OpenGradient Signal Lab", layout="wide")

# Title
st.title("📊 OpenGradient Signal Lab")
st.write("Short-term spot forecasting model (inspired by OpenGradient research)")

# -------------------------------
# INPUT SECTION
# -------------------------------
st.subheader("Fetch Market Data")

popular_pairs = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "SUIUSDT"]
symbol = st.selectbox("Trading Pair", popular_pairs, index=0)
interval = st.selectbox("Interval", ["30m", "1h", "6h", "1d"], index=1)
limit = st.slider("Number of candles", min_value=10, max_value=500, value=100)

# Binance fetch
def get_binance_ohlcv(symbol, interval, limit=100):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    data = response.json()
    df = pd.DataFrame(data, columns=[
        "OpenTime","Open","High","Low","Close","Volume",
        "CloseTime","QuoteAssetVolume","NumberOfTrades",
        "TakerBuyBaseAssetVolume","TakerBuyQuoteAssetVolume","Ignore"
    ])
    df = df[["Open","High","Low","Close","Volume"]].apply(pd.to_numeric, errors="coerce")
    return df.dropna()

# Yahoo Finance fallback (fast: 1 day only)
def get_yahoo_ohlcv(symbol="BTC-USD", interval="1h", limit=100):
    try:
        df = yf.download(symbol, interval=interval, period="1d", progress=False)
        if df.empty:
            return pd.DataFrame()
        df = df[["Open","High","Low","Close","Volume"]].apply(pd.to_numeric, errors="coerce")
        return df.tail(limit).dropna()
    except Exception:
        return pd.DataFrame()

if st.button("Fetch Data"):
    try:
        with st.spinner("Fetching data from Binance..."):
            df = get_binance_ohlcv(symbol, interval, limit)
        st.success("Fetched data from Binance ✅")
    except Exception as e:
        st.warning(f"Binance failed ({e}), switching to Yahoo Finance...")
        with st.spinner("Fetching data from Yahoo Finance..."):
            yahoo_symbol = symbol.replace("USDT", "-USD") if "USDT" in symbol else symbol
            df = get_yahoo_ohlcv(yahoo_symbol, interval, limit)
        if df.empty:
            st.error("Yahoo Finance returned no data. Try BTCUSDT or ETHUSDT with 1h interval.")
        else:
            st.success("Fetched data from Yahoo Finance ✅")

    if df.empty:
        st.stop()

    st.write("Preview of OHLCV data:")
    st.dataframe(df.head())

    # -------------------------------
    # FEATURE ENGINEERING
    # -------------------------------
    st.subheader("🔬 Feature Engineering")
    engineered_df = engineer_features(df)
    st.dataframe(engineered_df.head())

    # -------------------------------
    # FEATURE SELECTION (Lasso)
    # -------------------------------
    st.subheader("🎯 Selected Features (Lasso)")

    if engineered_df.empty or len(engineered_df) < 10:
        st.warning("Not enough data for feature selection. Try a different symbol or interval.")
        selected_df, selected_features = engineered_df, []
    else:
        selected_df, selected_features = select_features(engineered_df)
        st.write("Selected features:", list(selected_features))
        st.dataframe(selected_df.head())

    # -------------------------------
    # MODEL LOGIC (Simple Signal)
    # -------------------------------
    st.subheader("📡 Model Output")

    if not df.empty:
        # ✅ Use tail(1).iloc[0] to guarantee a single row
        latest = df.tail(1).iloc[0]

        # Explicitly cast to float
        open_price = float(latest["Open"])
        close_price = float(latest["Close"])
        high_price = float(latest["High"])
        low_price = float(latest["Low"])

        price_change = close_price - open_price
        volatility = high_price - low_price

        if price_change > 0 and volatility > 0:
            signal = "LONG 📈"
            confidence = min((price_change / open_price) * 100 + (volatility / open_price) * 50, 100)
        else:
            signal = "SHORT 📉"
            confidence = min(abs(price_change / open_price) * 100 + (volatility / open_price) * 50, 100)

        if "LONG" in signal:
            st.success(f"Signal: {signal}")
        else:
            st.error(f"Signal: {signal}")

        st.write(f"Confidence: {confidence:.2f}%")
        st.progress(int(confidence))

        # -------------------------------
        # EXECUTION LOGIC
        # -------------------------------
        st.subheader("⚙️ Execution Logic")
        if "LONG" in signal:
            st.write("Strategy Suggestion: Consider entering a LONG position.")
        else:
            st.write("Strategy Suggestion: Consider entering a SHORT position.")

# -------------------------------
# INFO SECTION
# -------------------------------
st.markdown("---")
st.subheader("ℹ️ How it works")
st.markdown("""
- Tries to fetch OHLCV data from Binance first
- If Binance is blocked/slow, instantly switches to Yahoo Finance (fast 1-day fetch)
- Applies feature engineering + Lasso selection only if enough samples exist
- Generates short-term directional signal (LONG / SHORT)
- Inspired by OpenGradient's spot forecasting models
""")

# -------------------------------
# KEEP ALIVE TRICK
# -------------------------------
st.sidebar.markdown("### Keep Alive")
st.sidebar.checkbox("Prevent sleep", value=True, help="Keeps the app session active")
