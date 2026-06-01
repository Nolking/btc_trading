import pandas as pd
import numpy as np

# =========================================================
# CONFIG
# =========================================================
INPUT_CSV_PATH = "BTCUSDT_15m_1825d.csv"

TRAIN_OUTPUT_PATH = "bitcoin_train.csv"
TEST_OUTPUT_PATH = "bitcoin_test.csv"

TRAIN_RATIO = 0.8   # first 80% -> train, last 20% -> test


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    rename_map = {
        "Open Time": "timestamp",
        "Timestamp": "timestamp",
        "Date": "timestamp",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename_map)

    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if np.issubdtype(df["timestamp"].dtype, np.number):
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df = df.dropna(subset=["timestamp"]).copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


if __name__ == "__main__":
    df = load_data(INPUT_CSV_PATH)

    split_idx = int(len(df) * TRAIN_RATIO)

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    train_df.to_csv(TRAIN_OUTPUT_PATH, index=False)
    test_df.to_csv(TEST_OUTPUT_PATH, index=False)

    print("Saved:")
    print(TRAIN_OUTPUT_PATH, len(train_df))
    print(TEST_OUTPUT_PATH, len(test_df))

    print("\nTrain range:")
    print(train_df["timestamp"].min(), "->", train_df["timestamp"].max())

    print("\nTest range:")
    print(test_df["timestamp"].min(), "->", test_df["timestamp"].max())