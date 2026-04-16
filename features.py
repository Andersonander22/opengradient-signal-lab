import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV

def engineer_features(df):
    """
    Takes OHLCV dataframe and returns engineered features.
    """
    df = df.copy()

    df['return'] = df['Close'].pct_change()
    df['volatility'] = df['High'] - df['Low']
    df['volume_change'] = df['Volume'].pct_change()

    # 🔥 Clean data
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    return df


def select_features(df, target_col='return'):
    """
    Apply Lasso feature selection safely.
    """

    # 🔥 Safety check
    if df.empty or len(df) < 20:
        return df, []

    df = df.copy()

    # Remove bad values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    if df.empty:
        return df, []

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 🔥 Ensure numeric only
    X = X.select_dtypes(include=[np.number])

    try:
        model = LassoCV(cv=3).fit(X, y)  # reduced cv for stability
        selected = X.columns[model.coef_ != 0]

        if len(selected) == 0:
            return df, []

        return df[selected], selected

    except Exception:
        # 🔥 Silent fail
        return df, []
