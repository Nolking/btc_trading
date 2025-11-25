# src/ml_strategy.py
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.features import add_technical_features


def create_labels(df: pd.DataFrame, horizon: int = 3, threshold: float = 0.0) -> pd.DataFrame:
    """
    Create classification labels based on future returns.

    horizon: number of bars ahead to look (e.g. 3 candles)
    threshold: minimum future return to be considered 'up'

    Label:
      y = 1  if future_return > threshold
      y = 0  otherwise
    """
    df = df.copy()
    future_return = np.log(df["close"].shift(-horizon) / df["close"])
    df["future_return"] = future_return
    df["y"] = (df["future_return"] > threshold).astype(int)

    return df


def build_ml_dataset(df: pd.DataFrame, feature_cols: list) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Given a dataframe with features and 'y', return (X, y)
    with rows that have no NaNs.
    """
    df = df.copy()
    df = df.dropna(subset=feature_cols + ["y"])

    X = df[feature_cols]
    y = df["y"]
    return X, y


def train_test_split_time_series(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = 0.7
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Time-based split: first part is train, last part is test.
    No shuffling (important for time series).
    """
    n = len(X)
    split_idx = int(n * train_ratio)

    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    return X_train, X_test, y_train, y_test


def train_random_forest_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 200,
    max_depth: int | None = None,
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Train a RandomForestClassifier on the training data.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def apply_model_to_data(
    df: pd.DataFrame,
    model: RandomForestClassifier,
    feature_cols: list,
    proba_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Apply trained model to entire dataframe to generate trading signals.

    'signal' is 1 if model's predicted probability of 'up' > proba_threshold,
    else 0.
    """
    df = df.copy()

    # We can only predict where features are non-NaN
    mask = df[feature_cols].notna().all(axis=1)
    probs = np.full(len(df), np.nan)

    if mask.sum() > 0:
        preds_proba = model.predict_proba(df.loc[mask, feature_cols])[:, 1]
        probs[mask.values] = preds_proba

    df["proba_up"] = probs

    # Generate signals (you can play with this threshold later)
    df["signal"] = 0
    df.loc[df["proba_up"] > proba_threshold, "signal"] = 1

    return df
