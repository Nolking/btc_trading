# src/data_loader.py
from pathlib import Path
import pandas as pd


def load_bitcoin_data(csv_path: str | Path) -> pd.DataFrame:
    """
    Load BTC price data from a Binance-style kline CSV.

    Expected original columns (case-insensitive, with optional spaces):
      - 'Open Time'
      - 'Open'
      - 'High'
      - 'Low'
      - 'Close'
      - 'Volume'
      - 'Close Time'
      - 'Quote Asset Volume'
      - 'Number of Trades'
      - 'Taker Buy Base Volume'
      - 'Taker Buy Quote Volume'
      - 'Ignore'

    Returns a DataFrame indexed by datetime (open_time) with at least:
      - 'open', 'high', 'low', 'close', 'volume'
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Normalize column names: lowercase, strip spaces, replace spaces with underscores
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # We expect 'open_time' and 'close' at minimum:
    if "open_time" not in df.columns:
        raise ValueError(
            "CSV must contain an 'Open Time' column (header may be 'Open Time')."
        )
    if "close" not in df.columns:
        raise ValueError(
            "CSV must contain a 'Close' column (header may be 'Close')."
        )

    # Parse open_time into datetime index.
    # Binance often stores open_time as milliseconds since epoch.
    if pd.api.types.is_integer_dtype(df["open_time"]) or pd.api.types.is_float_dtype(df["open_time"]):
        # interpret as milliseconds since epoch
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    else:
        # interpret as string dates
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True, errors="coerce")

    # Drop rows where the timestamp couldn't be parsed
    df = df.dropna(subset=["open_time"])

    df = df.set_index("open_time").sort_index()
    df.index.name = "date"  # optional, nice name

    # Ensure we have the main OHLCV columns, and normalize them to numeric
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise ValueError(f"CSV must contain '{col}' column (header may be '{col.capitalize()}').")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where close is NaN
    df = df.dropna(subset=["close"])

    return df
