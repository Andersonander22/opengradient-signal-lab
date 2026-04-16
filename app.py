import streamlit as st
import pandas as pd
import requests
import numpy as np
import os
from dotenv import load_dotenv
import opengradient as og
from features import engineer_features, select_features

# -------------------------------
# LOAD ENV + INIT OPENGRADIENT
# -------------------------------
load_dotenv()

client = og.init(private_key=os.getenv("PRIVATE_KEY"))

# -------------------------------
# PAGE SETUP
# -------------------------------
st.set_page_config(page_title="OpenGradient Signal Lab", layout="wide")

st.title("📊 OpenGradient Signal Lab")
st.write("AI-powered short-term spot forecasting (Binance + OpenGradient)")

# -------------------------------
# INPUTS
# -------------------------------
st.subheader("Fetch Market Data")

popular_pairs = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "SUIUSDT"]

interval_map = {
    "30m": "30m",
    "1h": "1h",
    "6h": "6h",
    "1d": "1d",
}

symbol = st.selectbox("Trading Pair", popular_pairs)
interval_label = st.selectbox("Interval", list(interval_map.keys()))
interval = interval_map[interval_label]

limit = st.slider("Number of candles", 10, 500, 100)

# -------------------------------
# BINANCE API (WORKING ENDPOINT)
# -------------------------------
def get_binance_ohlcv(symbol, interval, limit=100):
    try:
        url = "https://data-api.binance.vision/api/v3/klines"

        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }

        response = requests.get(url, params=params, timeout=10)

        if response.status_code != 200:
            return pd.DataFrame()

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

# -------------------------------
# FETCH BUTTON
# -------------------------------
if st.button("Fetch Data"):

    with st.spinner("Fetching data from Binance..."):
        df = get_binance_ohlcv(symbol, interval, limit)

    if df.empty:
        st.warning("⚠️ Unable to fetch market data right now. Please try again.")
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

    engineered_df = engineered_df.replace([np.inf, -np.inf], np.nan)
    engineered_df = engineered_df.dropna()

    if engineered_df.empty:
        st.warning("Feature engineering failed.")
        st.stop()

    st.dataframe(engineered_df.head())

    # -------------------------------
    # FEATURE SELECTION
    # -------------------------------
    st.subheader("🎯 Feature Selection")

    try:
        if len(engineered_df) < 20:
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
    # AI MODEL OUTPUT (OPENGRADIENT)
    # -------------------------------
    st.subheader("📡 AI Signal (OpenGradient Model)")

    try:
        latest = df.tail(1)

        open_price = float(latest["Open"].iloc[0])
        close_price = float(latest["Close"].iloc[0])
        high_price = float(latest["High"].iloc[0])
        low_price = float(latest["Low"].iloc[0])
        volume = float(latest["Volume"].iloc[0])

        prompt = f"""
        You are a crypto trading assistant.

        Based on this market data:
        Open: {open_price}
        Close: {close_price}
        High: {high_price}
        Low: {low_price}
        Volume: {volume}

        Predict the next move: LONG or SHORT.
        Also include a confidence percentage.
        Keep it short.
        """

        response = client.llm.chat(
            model="opengradient/llama",
            messages=[{"role": "user", "content": prompt}]
        )

        result = str(response)

        st.success("AI Signal Generated ✅")
        st.write(result)

    except Exception:
        st.warning("⚠️ AI model unavailable, using fallback logic.")

        # fallback logic
        if close_price > open_price:
            st.success("Fallback Signal: LONG 📈")
        else:
            st.error("Fallback Signal: SHORT 📉")

    # -------------------------------
    # INFO
    # -------------------------------
    st.markdown("---")
    st.subheader("ℹ️ About")

    st.markdown("""
    - Uses Binance public data API
    - Applies feature engineering + selection
    - Uses OpenGradient LLM for signal generation
    - Fallback logic ensures reliability
    """)
