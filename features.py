import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV

def engineer_features(df):
    """
    Takes OHLCV dataframe and returns engineered features.
    """
    df['return'] = df['Close'].pct_change()
    df['volatility'] = df['High'] - df['Low']
    df['volume_change'] = df['Volume'].pct_change()
    df = df.dropna()
    return df

def select_features(df, target_col='return'):
    """
    Apply Lasso feature selection to keep only predictive features.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    model = LassoCV(cv=5).fit(X, y)
    selected = X.columns[model.coef_ != 0]

    return df[selected], selected
