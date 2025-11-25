# src/strategy.py
import pandas as pd


def moving_average_crossover(
    df: pd.DataFrame,
    short_window: int = 20,
    long_window: int = 50,
) -> pd.DataFrame:
    """
    Simple Moving Average Crossover strategy.

    - Go long (signal = 1) when short MA > long MA
    - Go flat (signal = 0) when short MA <= long MA
    """
    df = df.copy()

    df["ma_short"] = df["close"].rolling(window=short_window, min_periods=1).mean()
    df["ma_long"] = df["close"].rolling(window=long_window, min_periods=1).mean()

    df["signal"] = 0
    df.loc[df["ma_short"] > df["ma_long"], "signal"] = 1

    return df


def macd_strategy(
    df: pd.DataFrame,
    fast_window: int = 12,
    slow_window: int = 26,
    signal_window: int = 9,
) -> pd.DataFrame:
    """
    MACD (Moving Average Convergence Divergence) strategy.

    - MACD line = EMA(fast_window) - EMA(slow_window)
    - Signal line = EMA(signal_window) of MACD
    - Go long (signal = 1) when MACD > Signal line
    - Go flat (signal = 0) when MACD <= Signal line

    This is a trend-following / momentum strategy.
    """
    df = df.copy()

    # EMAs of close price
    df["ema_fast"] = df["close"].ewm(span=fast_window, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=slow_window, adjust=False).mean()

    # MACD and signal line
    df["macd"] = df["ema_fast"] - df["ema_slow"]
    df["macd_signal"] = df["macd"].ewm(span=signal_window, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Trading rule
    df["signal"] = 0
    df.loc[df["macd"] > df["macd_signal"], "signal"] = 1

    return df


def bollinger_bands_strategy(
    df: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0,
    mode: str = "mean_reversion",
) -> pd.DataFrame:
    """
    Bollinger Bands strategy.

    Creates:
      - bb_middle: rolling mean
      - bb_upper:  middle + num_std * std
      - bb_lower:  middle - num_std * std

    Modes:
      - "mean_reversion":
          Go long (signal = 1) when close < lower band
          Go flat (signal = 0) otherwise

      - "breakout":
          Go long (signal = 1) when close > upper band
          Go flat (signal = 0) otherwise
    """
    df = df.copy()

    rolling = df["close"].rolling(window=window, min_periods=1)
    df["bb_middle"] = rolling.mean()
    df["bb_std"] = rolling.std(ddof=0)
    df["bb_upper"] = df["bb_middle"] + num_std * df["bb_std"]
    df["bb_lower"] = df["bb_middle"] - num_std * df["bb_std"]

    df["signal"] = 0

    if mode == "mean_reversion":
        # Buy dips below lower band (expecting price to revert back)
        df.loc[df["close"] < df["bb_lower"], "signal"] = 1
    elif mode == "breakout":
        # Buy volatility breakouts above upper band
        df.loc[df["close"] > df["bb_upper"], "signal"] = 1
    else:
        raise ValueError("mode must be 'mean_reversion' or 'breakout'")

    return df

def rsi_strategy(
    df: pd.DataFrame,
    period: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
) -> pd.DataFrame:
    """
    RSI-based long-only strategy.

    RSI logic:
      - RSI measures momentum between 0 and 100.
      - Low RSI = oversold, high RSI = overbought.

    Trading rules (mean-reversion style):
      1) Entry (open long):
         - Currently flat
         - AND RSI < oversold

      2) Exit (close long):
         - Currently in a position
         - AND RSI > overbought

    Output columns:
      - 'rsi'
      - 'signal'     : 1 = in position, 0 = flat
      - 'entry_flag' : 1 where a long is opened
      - 'exit_flag'  : 1 where the long is closed
    """
    df = df.copy()

    # --- Compute RSI ---
    # Price changes
    delta = df["close"].diff()

    # Separate gains and losses
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Rolling average gains and losses (simple version)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    # Avoid division by zero
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))

    df["rsi"] = rsi.fillna(50.0)  # neutral for initial periods

    # --- Entry & Exit conditions ---
    entry_condition = df["rsi"] < oversold
    exit_condition = df["rsi"] > overbought

    # --- Build position (signal) over time ---
    in_position = False
    signals = []

    for idx in range(len(df)):
        enter = bool(entry_condition.iloc[idx])
        exit_ = bool(exit_condition.iloc[idx])

        if not in_position and enter:
            in_position = True
        elif in_position and exit_:
            in_position = False

        signals.append(1 if in_position else 0)

    df["signal"] = signals

    # --- Mark entries and exits explicitly ---
    df["entry_flag"] = (df["signal"].diff() == 1).astype(int)
    df["exit_flag"] = (df["signal"].diff() == -1).astype(int)

    return df