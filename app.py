import streamlit as st
import pandas as pd
import time
from binance.client import Client
from binance.enums import *
from features import engineer_features, select_features

# -------------------------------
# INIT BINANCE CLIENT
# -------------------------------
client = Client()

# -------------------------------
# PAGE SETUP
# -------------------------------
st.set_page_config(page_title="OpenGradient Signal Lab", layout="wide")

st.title("📊 OpenGradient Signal Lab")
st.write("Short-term spot forecasting model (Binance-powered)")

# -------------------------------
# INPUTS
# -------------------------------
st.subheader("Fetch Market Data")

popular_pairs = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "SUIUSDT"]

interval_map = {
    "30m": Client.KLINE_INTERVAL_30MINUTE,
    "1h": Client.KLINE_INTERVAL_1HOUR,
    "6h": Client.KLINE_INTERVAL_6HOUR,
    "1d": Client.KLINE_INTERVAL_1DAY,
}

symbol = st.selectbox("Trading Pair", popular_pairs)
interval_label = st.selectbox("Interval", list(interval_map.keys()))
interval = interval_map[interval_label]

limit = st.slider("Number of candles", 10, 500, 100)

# -------------------------------
# FETCH BINANCE DATA
# -------------------------------
def get_binance_ohlcv(symbol, interval, limit=100):
    try:
        time.sleep(0.3)  # avoid rate limit

        klines = client.get_klines(
            symbol=symbol,
            interval=interval,
            limit=limit
        )

        df = pd.DataFrame(klines, columns=[
            "OpenTime","Open","High","Low","Close","Volume",
            "CloseTime","QuoteAssetVolume","NumberOfTrades",
            "TakerBuyBaseAssetVolume","TakerBuyQuoteAssetVolume","Ignore"
        ])

        df = df[["Open","High","Low","Close","Volume"]]
        df = df.apply(pd.to_numeric, errors="coerce")

        return df.dropna()

    except Exception as e:
        st.error(f"Binance API error: {e}")
        return pd.DataFrame()

# -------------------------------
# FETCH BUTTON
# -------------------------------
if st.button("Fetch Data"):

    with st.spinner("Fetching data from Binance..."):
        df = get_binance_ohlcv(symbol, interval, limit)

    if df.empty:
        st.error("Failed to fetch data from Binance. Try another pair or use VPN.")
        st.stop()

    st.success("Data fetched from Binance ✅")
    st.write("Data shape:", df.shape)

    st.subheader("OHLCV Preview")
    st.dataframe(df.head())

    # -------------------------------
    # FEATURE ENGINEERING
    # -------------------------------
    st.subheader("🔬 Feature Engineering")

    engineered_df = engineer_features(df)

    engineered_df = engineered_df.replace([float("inf"), float("-inf")], None)
    engineered_df = engineered_df.dropna()

    if engineered_df.empty:
        st.warning("Feature engineering failed.")
        st.stop()

    st.dataframe(engineered_df.head())

    # -------------------------------
    # FEATURE SELECTION (SAFE)
    # -------------------------------
    st.subheader("🎯 Feature Selection")

    try:
        if len(engineered_df) < 10:
            st.info("Not enough data for feature selection.")
            selected_df = engineered_df
        else:
            selected_df, selected_features = select_features(engineered_df)
            st.write("Selected features:", list(selected_features))
            st.dataframe(selected_df.head())

    except Exception:
        st.info("Feature selection skipped due to data issues.")
        selected_df = engineered_df

    # -------------------------------
    # MODEL OUTPUT (SAFE)
    # -------------------------------
    st.subheader("📡 Model Output")

    try:
        latest = df.tail(1)

        open_price = float(latest["Open"].iloc[0])
        close_price = float(latest["Close"].iloc[0])
        high_price = float(latest["High"].iloc[0])
        low_price = float(latest["Low"].iloc[0])

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

        # Execution logic
        st.subheader("⚙️ Execution Logic")

        if "LONG" in signal:
            st.write("Strategy Suggestion: Consider entering a LONG position.")
        else:
            st.write("Strategy Suggestion: Consider entering a SHORT position.")

    except Exception:
        st.info("Signal generation skipped due to data issues.")

# -------------------------------
# INFO
# -------------------------------
st.markdown("---")
st.subheader("ℹ️ About")

st.markdown("""
- Uses **Binance OHLCV data only**
- Applies feature engineering + Lasso selection
- Generates directional trading signal
- Built for OpenGradient experimentation
""")
