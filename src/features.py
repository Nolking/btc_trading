# src/features.py
import pandas as pd
import numpy as np


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators and simple features to the dataframe.

    Features:
      - log_return_1, log_return_3, log_return_5
      - rolling_vol_10
      - rsi_14
      - macd, macd_signal, macd_hist
      - bollinger_middle_20, bollinger_upper_20, bollinger_lower_20
    """
    df = df.copy()

    # --- Returns ---
    df["log_return_1"] = np.log(df["close"] / df["close"].shift(1))
    df["log_return_3"] = np.log(df["close"] / df["close"].shift(3))
    df["log_return_5"] = np.log(df["close"] / df["close"].shift(5))

    # Rolling volatility
    df["rolling_vol_10"] = df["log_return_1"].rolling(10).std()

    # --- RSI (14) ---
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    period = 14
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    df["rsi_14"] = rsi

    # --- MACD ---
    fast_window = 12
    slow_window = 26
    signal_window = 9

    ema_fast = df["close"].ewm(span=fast_window, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow_window, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal_window, adjust=False).mean()
    macd_hist = macd - macd_signal

    df["macd"] = macd
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist

    # --- Bollinger Bands (20, 2) ---
    window = 20
    bb_middle = df["close"].rolling(window).mean()
    bb_std = df["close"].rolling(window).std(ddof=0)
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std

    df["bb_middle_20"] = bb_middle
    df["bb_upper_20"] = bb_upper
    df["bb_lower_20"] = bb_lower

    # You can add more features later (volume-based, etc.)

    return df
