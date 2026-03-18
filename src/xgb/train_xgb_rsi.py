import os
import json
import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


# =========================================================
# CONFIG
# =========================================================
TRAIN_CSV_PATH = "bitcoin_train.csv"

MODEL_OUTPUT_PATH = "xgb_divergence_model.pkl"
META_OUTPUT_PATH = "xgb_divergence_model_meta.json"
SIGNALS_OUTPUT_PATH = "xgb_divergence_training_signals.csv"

INITIAL_CAPITAL = 10000.0
LOT_SIZE = 0.05
RSI_PERIOD = 30

PIVOT_LEFT = 3
PIVOT_RIGHT = 3

REFERENCE_EQUITY = 10000.0
STOP_LOSS_PCT_OF_EQUITY = 0.01   # 1% capital = 100 USD

RANDOM_STATE = 42

# optional internal validation on train set only
USE_INTERNAL_VALIDATION = True
VALIDATION_RATIO = 0.2


def ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


# =========================================================
# DATA LOADING
# =========================================================
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

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)
    df["date"] = df["timestamp"].dt.date
    return df


# =========================================================
# RSI
# =========================================================
def compute_rsi(series: pd.Series, period: int = 30) -> pd.Series:
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


# =========================================================
# PIVOTS
# =========================================================
def pivot_low(series: pd.Series, left: int, right: int) -> pd.Series:
    out = np.full(len(series), False)
    vals = series.values
    for i in range(left, len(series) - right):
        window = vals[i - left:i + right + 1]
        if vals[i] == np.min(window):
            out[i] = True
    return pd.Series(out, index=series.index)


def pivot_high(series: pd.Series, left: int, right: int) -> pd.Series:
    out = np.full(len(series), False)
    vals = series.values
    for i in range(left, len(series) - right):
        window = vals[i - left:i + right + 1]
        if vals[i] == np.max(window):
            out[i] = True
    return pd.Series(out, index=series.index)


# =========================================================
# FEATURES
# =========================================================
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["rsi"] = compute_rsi(df["close"], RSI_PERIOD)

    df["ret_1"] = df["close"].pct_change(1)
    df["ret_3"] = df["close"].pct_change(3)
    df["ret_5"] = df["close"].pct_change(5)

    df["rsi_change_1"] = df["rsi"].diff(1)
    df["rsi_change_3"] = df["rsi"].diff(3)
    df["rsi_change_5"] = df["rsi"].diff(5)

    df["rsi_mean_5"] = df["rsi"].rolling(5).mean()
    df["rsi_mean_10"] = df["rsi"].rolling(10).mean()
    df["rsi_std_5"] = df["rsi"].rolling(5).std()
    df["rsi_std_10"] = df["rsi"].rolling(10).std()

    df["rsi_distance_30"] = df["rsi"] - 30.0
    df["rsi_distance_70"] = 70.0 - df["rsi"]

    df["price_range_pct"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)

    return df


# =========================================================
# DIVERGENCE DETECTION
# =========================================================
def detect_divergences(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["price_pivot_low"] = pivot_low(df["close"], PIVOT_LEFT, PIVOT_RIGHT)
    df["price_pivot_high"] = pivot_high(df["close"], PIVOT_LEFT, PIVOT_RIGHT)
    df["rsi_pivot_low"] = pivot_low(df["rsi"], PIVOT_LEFT, PIVOT_RIGHT)
    df["rsi_pivot_high"] = pivot_high(df["rsi"], PIVOT_LEFT, PIVOT_RIGHT)

    df["bullish_divergence"] = False
    df["bearish_divergence"] = False

    df["div_price_change"] = np.nan
    df["div_rsi_change"] = np.nan
    df["div_bars_gap"] = np.nan

    low_idxs = df.index[df["price_pivot_low"]].tolist()
    high_idxs = df.index[df["price_pivot_high"]].tolist()

    for k in range(1, len(low_idxs)):
        i1, i2 = low_idxs[k - 1], low_idxs[k]
        price_ll = df.loc[i2, "close"] < df.loc[i1, "close"]
        rsi_hl = df.loc[i2, "rsi"] > df.loc[i1, "rsi"]

        if price_ll and rsi_hl:
            df.loc[i2, "bullish_divergence"] = True
            df.loc[i2, "div_price_change"] = (
                df.loc[i2, "close"] - df.loc[i1, "close"]
            ) / df.loc[i1, "close"]
            df.loc[i2, "div_rsi_change"] = df.loc[i2, "rsi"] - df.loc[i1, "rsi"]
            df.loc[i2, "div_bars_gap"] = i2 - i1

    for k in range(1, len(high_idxs)):
        i1, i2 = high_idxs[k - 1], high_idxs[k]
        price_hh = df.loc[i2, "close"] > df.loc[i1, "close"]
        rsi_lh = df.loc[i2, "rsi"] < df.loc[i1, "rsi"]

        if price_hh and rsi_lh:
            df.loc[i2, "bearish_divergence"] = True
            df.loc[i2, "div_price_change"] = (
                df.loc[i2, "close"] - df.loc[i1, "close"]
            ) / df.loc[i1, "close"]
            df.loc[i2, "div_rsi_change"] = df.loc[i2, "rsi"] - df.loc[i1, "rsi"]
            df.loc[i2, "div_bars_gap"] = i2 - i1

    df["signal_side"] = np.where(
        df["bullish_divergence"], 1,
        np.where(df["bearish_divergence"], -1, 0)
    )

    return df


# =========================================================
# PNL
# =========================================================
def calc_pnl(side: str, entry_price: float, exit_price: float, lot_size: float = LOT_SIZE) -> float:
    if side == "long":
        return (exit_price - entry_price) * lot_size
    elif side == "short":
        return (entry_price - exit_price) * lot_size
    else:
        raise ValueError("side must be 'long' or 'short'")


# =========================================================
# LABEL CREATION
# =========================================================
def label_signal_outcome(df: pd.DataFrame, signal_idx: int, side: str, reference_equity: float) -> dict:
    entry_price = float(df.loc[signal_idx, "close"])
    # stop_loss_usd = reference_equity * STOP_LOSS_PCT_OF_EQUITY
    stop_loss_usd = 100.0  # fixed 100 USD stop loss for simplicity
    
    opposite_col = "bearish_divergence" if side == "long" else "bullish_divergence"

    exit_idx = None
    exit_reason = None
    realized_pnl = None

    for j in range(signal_idx + 1, len(df)):
        px = float(df.loc[j, "close"])
        running_pnl = calc_pnl(side, entry_price, px, LOT_SIZE)

        if running_pnl <= -stop_loss_usd:
            exit_idx = j
            exit_reason = "stop_loss"
            realized_pnl = running_pnl
            break

        if bool(df.loc[j, opposite_col]):
            exit_idx = j
            exit_reason = "opposite_divergence"
            realized_pnl = running_pnl
            break

    if exit_idx is None:
        exit_idx = len(df) - 1
        exit_reason = "end_of_data"
        realized_pnl = calc_pnl(side, entry_price, float(df.loc[exit_idx, "close"]), LOT_SIZE)

    return {
        "exit_idx": exit_idx,
        "exit_reason": exit_reason,
        "realized_pnl": realized_pnl,
        "target": int(realized_pnl > 0),
    }


def build_training_dataset(df: pd.DataFrame):
    signal_mask = (df["bullish_divergence"] | df["bearish_divergence"]).copy()
    signals = df.loc[signal_mask].copy()

    feature_columns = [
        "close",
        "rsi",
        "ret_1",
        "ret_3",
        "ret_5",
        "rsi_change_1",
        "rsi_change_3",
        "rsi_change_5",
        "rsi_mean_5",
        "rsi_mean_10",
        "rsi_std_5",
        "rsi_std_10",
        "rsi_distance_30",
        "rsi_distance_70",
        "price_range_pct",
        "div_price_change",
        "div_rsi_change",
        "div_bars_gap",
        "signal_side",
    ]

    outcomes = []
    for idx, row in signals.iterrows():
        side = "long" if row["bullish_divergence"] else "short"
        outcomes.append(label_signal_outcome(df, idx, side, REFERENCE_EQUITY))

    signals["exit_idx"] = [x["exit_idx"] for x in outcomes]
    signals["exit_reason"] = [x["exit_reason"] for x in outcomes]
    signals["realized_pnl"] = [x["realized_pnl"] for x in outcomes]
    signals["target"] = [x["target"] for x in outcomes]

    signals = signals.dropna(subset=feature_columns).copy()
    return signals, feature_columns


def train_model(signals: pd.DataFrame, feature_columns: list):
    X = signals[feature_columns].copy()
    y = signals["target"].astype(int).copy()

    if y.nunique() < 2:
        raise ValueError("Training target has only one class. Need both winning and losing examples.")

    pos = int(y.sum())
    neg = int(len(y) - pos)
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
    )

    if USE_INTERNAL_VALIDATION and len(signals) >= 50:
        split_idx = int(len(signals) * (1 - VALIDATION_RATIO))
        train_df = signals.iloc[:split_idx].copy()
        valid_df = signals.iloc[split_idx:].copy()

        X_train = train_df[feature_columns]
        y_train = train_df["target"].astype(int)
        X_valid = valid_df[feature_columns]
        y_valid = valid_df["target"].astype(int)

        model.fit(X_train, y_train)

        valid_pred = model.predict(X_valid)
        valid_proba = model.predict_proba(X_valid)[:, 1]

        print("=== Internal Validation On Train CSV Only ===")
        print(classification_report(y_valid, valid_pred, digits=4))
        print(confusion_matrix(y_valid, valid_pred))
        try:
            auc = roc_auc_score(y_valid, valid_proba)
            print(f"ROC AUC: {auc:.4f}")
        except Exception:
            print("ROC AUC: could not compute")

        # retrain on full train dataset after validation
        model.fit(X, y)
    else:
        model.fit(X, y)

    feature_importance = pd.DataFrame({
        "feature": feature_columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    print("\n=== Top Feature Importances ===")
    print(feature_importance.head(20).to_string(index=False))

    return model, feature_importance


if __name__ == "__main__":
    ensure_parent_dir(MODEL_OUTPUT_PATH)
    ensure_parent_dir(META_OUTPUT_PATH)
    ensure_parent_dir(SIGNALS_OUTPUT_PATH)

    df = load_data(TRAIN_CSV_PATH)
    df = add_features(df)
    df = detect_divergences(df)
    signals, feature_columns = build_training_dataset(df)

    print(f"Training rows: {len(df)}")
    print(f"Training divergence signals: {len(signals)}")
    print("\nTraining target distribution:")
    print(signals["target"].value_counts(dropna=False))

    model, feature_importance = train_model(signals, feature_columns)

    joblib.dump(model, MODEL_OUTPUT_PATH)

    meta = {
        "train_csv_path": TRAIN_CSV_PATH,
        "model_type": "XGBClassifier",
        "rsi_period": RSI_PERIOD,
        "initial_capital": INITIAL_CAPITAL,
        "reference_equity_for_labeling": REFERENCE_EQUITY,
        "lot_size": LOT_SIZE,
        "stop_loss_pct_of_equity": STOP_LOSS_PCT_OF_EQUITY,
        "feature_columns": feature_columns,
        "pivot_left": PIVOT_LEFT,
        "pivot_right": PIVOT_RIGHT,
    }

    with open(META_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    signals.to_csv(SIGNALS_OUTPUT_PATH, index=False)

    print("\nSaved:")
    print(MODEL_OUTPUT_PATH)
    print(META_OUTPUT_PATH)
    print(SIGNALS_OUTPUT_PATH)