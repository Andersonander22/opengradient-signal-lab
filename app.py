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

# -------------------------------
# DATA FETCH FUNCTIONS
# -------------------------------
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

    df = df[["Open","High","Low","Close","Volume"]]
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.dropna()

def get_yahoo_ohlcv(symbol="BTC-USD", interval="1h", limit=100):
    try:
        df = yf.download(symbol, interval=interval, period="1d", progress=False)
        if df.empty:
            return pd.DataFrame()

        df = df[["Open","High","Low","Close","Volume"]]
        df = df.apply(pd.to_numeric, errors="coerce")
        return df.tail(limit).dropna()
    except Exception:
        return pd.DataFrame()

# -------------------------------
# FETCH BUTTON
# -------------------------------
if st.button("Fetch Data"):

    try:
        with st.spinner("Fetching data from Binance..."):
            df = get_binance_ohlcv(symbol, interval, limit)
        st.success("Fetched data from Binance ✅")
    except Exception as e:
        st.warning(f"Binance failed ({e}), switching to Yahoo Finance...")
        with st.spinner("Fetching data from Yahoo Finance..."):
            yahoo_symbol = symbol.replace("USDT", "-USD")
            df = get_yahoo_ohlcv(yahoo_symbol, interval, limit)

        if df.empty:
            st.error("Yahoo Finance returned no data. Try BTCUSDT or ETHUSDT.")
            st.stop()
        else:
            st.success("Fetched data from Yahoo Finance ✅")

    # -------------------------------
    # DATA SAFETY CHECK
    # -------------------------------
    if df.empty:
        st.error("No data available.")
        st.stop()

    df = df.dropna()

    st.write("Preview of OHLCV data:")
    st.dataframe(df.head())

    # -------------------------------
    # FEATURE ENGINEERING
    # -------------------------------
    st.subheader("🔬 Feature Engineering")
    engineered_df = engineer_features(df)

    if engineered_df.empty:
        st.warning("Feature engineering failed due to insufficient data.")
        st.stop()

    st.dataframe(engineered_df.head())

    # -------------------------------
    # FEATURE SELECTION (Lasso)
    # -------------------------------
    st.subheader("🎯 Selected Features (Lasso)")

    if len(engineered_df) < 10:
        st.warning("Not enough data for feature selection.")
        selected_df, selected_features = engineered_df, []
    else:
        selected_df, selected_features = select_features(engineered_df)
        st.write("Selected features:", list(selected_features))
        st.dataframe(selected_df.head())

    # -------------------------------
    # MODEL LOGIC (SAFE VERSION)
    # -------------------------------
    st.subheader("📡 Model Output")

    try:
        latest = df.tail(1)

        open_price = float(latest["Open"].iloc[0])
        close_price = float(latest["Close"].iloc[0])
        high_price = float(latest["High"].iloc[0])
        low_price = float(latest["Low"].iloc[0])

    except Exception:
        st.error("Error reading market data. Try another pair or interval.")
        st.stop()

    price_change = close_price - open_price
    volatility = high_price - low_price

    if price_change > 0 and volatility > 0:
        signal = "LONG 📈"
        confidence = min((price_change / open_price) * 100 + (volatility / open_price) * 50, 100)
    else:
        signal = "SHORT 📉"
        confidence = min(abs(price_change / open_price) * 100 + (volatility / open_price) * 50, 100)

    # Display signal
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
- Fetches OHLCV data from Binance (fallback: Yahoo Finance)
- Applies feature engineering + Lasso selection
- Generates directional signal (LONG / SHORT)
- Inspired by OpenGradient forecasting models
""")

# -------------------------------
# KEEP ALIVE
# -------------------------------
st.sidebar.markdown("### Keep Alive")
st.sidebar.checkbox("Prevent sleep", value=True)
