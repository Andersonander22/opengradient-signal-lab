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
    try:
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

    except Exception:
        return pd.DataFrame()


def get_yahoo_ohlcv(symbol="BTC-USD", limit=100):
    try:
        df = yf.download(
            symbol,
            period="7d",     # 🔥 FIXED
            interval="1h",   # 🔥 STABLE
            progress=False
        )

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

    # Try Binance first
    with st.spinner("Fetching data from Binance..."):
        df = get_binance_ohlcv(symbol, interval, limit)

    if not df.empty:
        st.success("Fetched data from Binance ✅")
    else:
        st.warning("Binance failed or blocked → switching to Yahoo Finance")

        yahoo_symbol = symbol.replace("USDT", "-USD")

        with st.spinner("Fetching data from Yahoo Finance..."):
            df = get_yahoo_ohlcv(yahoo_symbol, limit)

        if df.empty:
            st.error("Failed to fetch data from both Binance and Yahoo.")
            st.stop()
        else:
            st.success("Fetched data from Yahoo Finance ✅")

    # -------------------------------
    # DEBUG (IMPORTANT)
    # -------------------------------
    st.write("Data shape:", df.shape)

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
        st.warning("Feature engineering failed.")
        st.stop()

    st.dataframe(engineered_df.head())

    # -------------------------------
    # FEATURE SELECTION
    # -------------------------------
    st.subheader("🎯 Selected Features (Lasso)")

    if len(engineered_df) < 10:
        st.warning("Not enough data for feature selection.")
        selected_df = engineered_df
    else:
        selected_df, selected_features = select_features(engineered_df)
        st.write("Selected features:", list(selected_features))
        st.dataframe(selected_df.head())

    # -------------------------------
    # MODEL OUTPUT
    # -------------------------------
    st.subheader("📡 Model Output")

    try:
        latest = df.tail(1)

        open_price = float(latest["Open"].iloc[0])
        close_price = float(latest["Close"].iloc[0])
        high_price = float(latest["High"].iloc[0])
        low_price = float(latest["Low"].iloc[0])

    except Exception:
        st.error("Error reading market data.")
        st.stop()

    price_change = close_price - open_price
    volatility = high_price - low_price

    if price_change > 0 and volatility > 0:
        signal = "LONG 📈"
        confidence = min((price_change / open_price) * 100 + (volatility / open_price) * 50, 100)
    else:
        signal = "SHORT 📉"
        confidence = min(abs(price_change / open_price) * 100 + (volatility / open_price) * 50, 100)

    # Display
    if "LONG" in signal:
        st.success(f"Signal: {signal}")
    else:
        st.error(f"Signal: {signal}")

    st.write(f"Confidence: {confidence:.2f}%")
    st.progress(int(confidence))

    # Execution logic
    st.subheader("⚙️ Execution Logic")

    if "LONG" in signal:
        st.write("Strategy Suggestion: Consider entering a LONG position.")
    else:
        st.write("Strategy Suggestion: Consider entering a SHORT position.")

# -------------------------------
# INFO
# -------------------------------
st.markdown("---")
st.subheader("ℹ️ How it works")

st.markdown("""
- Tries Binance first
- Falls back to Yahoo Finance (stable)
- Uses OHLCV data
- Applies feature engineering + Lasso selection
- Generates LONG / SHORT signal
""")
