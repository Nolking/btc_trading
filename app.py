import os
import json
import uuid
import time
import threading
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import joblib
import requests
import numpy as np
import pandas as pd

from flask import Flask, jsonify
from apscheduler.schedulers.background import BackgroundScheduler


# =========================================================
# CONFIG
BASE_DIR = Path(__file__).resolve().parent

SYMBOL = "BTCUSDT"
INTERVAL = "15m"
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"

MODEL_TYPE = "xgb"   # "rf" or "xgb"

MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

if MODEL_TYPE == "rf":
    MODEL_PATH = MODELS_DIR / "rf_divergence_model.pkl"
    META_PATH = MODELS_DIR / "rf_divergence_model_meta.json"
else:
    MODEL_PATH = MODELS_DIR / "xgb_divergence_model.pkl"
    META_PATH = MODELS_DIR / "xgb_divergence_model_meta.json"

LIVE_CSV_PATH = DATA_DIR / "btcusdt_15m_live.csv"

LATEST_SIGNAL_PATH = LOGS_DIR / "latest_signal.json"
LIVE_SIGNALS_CSV_PATH = LOGS_DIR / "live_signals.csv"
LIVE_TRADES_CSV_PATH = LOGS_DIR / "live_trades.csv"
OPEN_TRADE_PATH = LOGS_DIR / "open_trade.json"
PROCESSING_STATE_PATH = LOGS_DIR / "processing_state.json"

FETCH_LIMIT = 1000

INITIAL_CAPITAL = 10000.0
REFERENCE_EQUITY = 10000.0
LOT_SIZE = 0.05
STOP_LOSS_PCT_OF_EQUITY = 0.01   # 1% of 10000 = 100 USD

PROBABILITY_THRESHOLD = 0.50
REQUIRE_MODEL_CLASS_1 = True

# retry behavior when a fresh 15m candle is not yet available
FETCH_RETRY_COUNT = 3
FETCH_RETRY_DELAY_SECONDS = 15

# whether to rebuild historical signals/trades since last trained date
ENABLE_RETROACTIVE_BACKFILL = True

# if your training meta json contains one of these fields, the app will use it:
#   "train_end_timestamp"
#   "last_trained_timestamp"
#   "trained_until"
# otherwise it will backfill from the earliest candle currently present in live data
TRAIN_END_META_KEYS = ["train_end_timestamp", "last_trained_timestamp", "trained_until"]


# =========================================================
# APP
# =========================================================
app = Flask(__name__)
scheduler = BackgroundScheduler()
lock = threading.Lock()

model = None
meta = None
feature_columns = None


# =========================================================
# GENERAL HELPERS
# =========================================================
def ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

def safe_float(v):
    if pd.isna(v):
        return None
    return float(v)

def to_iso_ts(v) -> Optional[str]:
    if v is None:
        return None
    ts = pd.Timestamp(v)
    if pd.isna(ts):
        return None
    return ts.isoformat()

def parse_ts(v) -> Optional[pd.Timestamp]:
    if v is None:
        return None
    ts = pd.to_datetime(v, errors="coerce")
    if pd.isna(ts):
        return None
    return ts

def load_json_file(path: str, default_value):
    if not os.path.exists(path):
        return default_value
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default_value

def save_json_file(path: str, payload: dict):
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def append_row_to_csv(path: str, row: dict):
    ensure_parent_dir(path)
    df_new = pd.DataFrame([row])
    if os.path.exists(path):
        df_old = pd.read_csv(path)
        df_out = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.to_csv(path, index=False)

def upsert_csv_row(path: str, key_col: str, row: dict):
    ensure_parent_dir(path)
    row_df = pd.DataFrame([row])
    if not os.path.exists(path):
        row_df.to_csv(path, index=False)
        return

    df = pd.read_csv(path)
    if df.empty or key_col not in df.columns:
        row_df.to_csv(path, index=False)
        return

    key_val = str(row[key_col])
    mask = df[key_col].astype(str) == key_val
    if mask.any():
        for col, val in row.items():
            if col not in df.columns:
                df[col] = np.nan
            df.loc[mask, col] = val
    else:
        all_cols = list(dict.fromkeys(list(df.columns) + list(row_df.columns)))
        df = df.reindex(columns=all_cols)
        row_df = row_df.reindex(columns=all_cols)
        df = pd.concat([df, row_df], ignore_index=True)

    df.to_csv(path, index=False)

def file_has_rows(path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        df = pd.read_csv(path)
        return not df.empty
    except Exception:
        return False


# =========================================================
# MODEL LOADING
# =========================================================
def load_model():
    global model, meta, feature_columns

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Meta not found: {META_PATH}")

    model = joblib.load(MODEL_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    feature_columns = meta["feature_columns"]

def get_training_end_timestamp_from_meta() -> Optional[pd.Timestamp]:
    if not isinstance(meta, dict):
        return None

    for k in TRAIN_END_META_KEYS:
        if k in meta and meta[k]:
            ts = parse_ts(meta[k])
            if ts is not None:
                return ts

    return None


# =========================================================
# BINANCE DATA
# =========================================================
def fetch_binance_klines(symbol=SYMBOL, interval=INTERVAL, limit=FETCH_LIMIT) -> pd.DataFrame:
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    r = requests.get(BINANCE_KLINES_URL, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ]

    df = pd.DataFrame(data, columns=cols)
    df = df.rename(columns={"open_time": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    df = df.dropna().sort_values("timestamp").reset_index(drop=True)

    # drop incomplete current candle
    if len(df) > 1:
        df = df.iloc[:-1].copy()

    df["date"] = df["timestamp"].dt.date
    return df.reset_index(drop=True)

def load_live_data() -> pd.DataFrame:
    if not os.path.exists(LIVE_CSV_PATH):
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "date"])

    df = pd.read_csv(LIVE_CSV_PATH)
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "date"])

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"]).copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["date"] = df["timestamp"].dt.date
    return df

def save_live_data(df: pd.DataFrame):
    ensure_parent_dir(LIVE_CSV_PATH)
    df.to_csv(LIVE_CSV_PATH, index=False)

def merge_and_update_history() -> pd.DataFrame:
    old_df = load_live_data()
    new_df = fetch_binance_klines()

    df = pd.concat([old_df, new_df], ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["date"] = df["timestamp"].dt.date

    save_live_data(df)
    return df


# =========================================================
# RETRY / FRESH-CANDLE LOGIC
# =========================================================
def floor_to_15m(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.floor("15min")

def expected_latest_closed_candle(now_utc: Optional[pd.Timestamp] = None) -> pd.Timestamp:
    """
    If now is 12:15:xx, the latest fully closed candle should be 12:00.
    If now is 12:29:xx, still 12:15 is not closed yet? No:
    Binance 15m candle closes at exact 12:15 for candle 12:00-12:14:59.
    After flooring to 15m and subtracting 15m, we get latest fully closed bucket start.
    """
    if now_utc is None:
        now_utc = pd.Timestamp.utcnow()
    now_utc = pd.Timestamp(now_utc).tz_localize(None) if pd.Timestamp(now_utc).tzinfo is not None else pd.Timestamp(now_utc)
    return floor_to_15m(now_utc) - pd.Timedelta(minutes=15)

def fetch_with_retry_for_expected_candle() -> Tuple[pd.DataFrame, dict]:
    """
    Retry up to FETCH_RETRY_COUNT times, once every FETCH_RETRY_DELAY_SECONDS,
    if the newest closed candle available is older than expected_latest_closed_candle().
    """
    attempt_logs = []
    expected_ts = expected_latest_closed_candle()

    last_df = pd.DataFrame()
    for attempt in range(1, FETCH_RETRY_COUNT + 1):
        df = fetch_binance_klines(limit=FETCH_LIMIT)
        last_df = df

        if df.empty:
            attempt_logs.append({
                "attempt": attempt,
                "status": "empty",
                "expected_latest_closed_candle": expected_ts.isoformat(),
                "latest_available": None
            })
        else:
            latest_available = pd.Timestamp(df["timestamp"].max())
            ok = latest_available >= expected_ts
            attempt_logs.append({
                "attempt": attempt,
                "status": "ok" if ok else "stale",
                "expected_latest_closed_candle": expected_ts.isoformat(),
                "latest_available": latest_available.isoformat()
            })

            if ok:
                return df, {
                    "fresh_data_found": True,
                    "expected_latest_closed_candle": expected_ts.isoformat(),
                    "attempts": attempt_logs
                }

        if attempt < FETCH_RETRY_COUNT:
            time.sleep(FETCH_RETRY_DELAY_SECONDS)

    return last_df, {
        "fresh_data_found": False,
        "expected_latest_closed_candle": expected_ts.isoformat(),
        "attempts": attempt_logs
    }

def merge_and_update_history_with_retry() -> Tuple[pd.DataFrame, dict]:
    old_df = load_live_data()
    new_df, retry_info = fetch_with_retry_for_expected_candle()

    df = pd.concat([old_df, new_df], ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["date"] = df["timestamp"].dt.date

    save_live_data(df)
    return df, retry_info


# =========================================================
# FEATURE ENGINEERING
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

def add_features(df: pd.DataFrame, rsi_period: int) -> pd.DataFrame:
    df = df.copy()

    df["rsi"] = compute_rsi(df["close"], rsi_period)

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

def detect_divergences(df: pd.DataFrame, pivot_left: int, pivot_right: int) -> pd.DataFrame:
    df = df.copy()

    df["price_pivot_low"] = pivot_low(df["close"], pivot_left, pivot_right)
    df["price_pivot_high"] = pivot_high(df["close"], pivot_left, pivot_right)
    df["rsi_pivot_low"] = pivot_low(df["rsi"], pivot_left, pivot_right)
    df["rsi_pivot_high"] = pivot_high(df["rsi"], pivot_left, pivot_right)

    df["bullish_divergence"] = False
    df["bearish_divergence"] = False

    df["div_price_change"] = np.nan
    df["div_rsi_change"] = np.nan
    df["div_bars_gap"] = np.nan

    low_idxs = df.index[df["price_pivot_low"]].tolist()
    high_idxs = df.index[df["price_pivot_high"]].tolist()

    # bullish divergence = price lower low, RSI higher low
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

    # bearish divergence = price higher high, RSI lower high
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
# STATE / FILE HELPERS
# =========================================================
def read_latest_signal():
    return load_json_file(LATEST_SIGNAL_PATH, None)

def save_latest_signal(payload: dict):
    save_json_file(LATEST_SIGNAL_PATH, payload)

def read_open_trade() -> Optional[Dict[str, Any]]:
    return load_json_file(OPEN_TRADE_PATH, None)

def save_open_trade(payload: dict):
    save_json_file(OPEN_TRADE_PATH, payload)

def clear_open_trade():
    if os.path.exists(OPEN_TRADE_PATH):
        os.remove(OPEN_TRADE_PATH)

def read_processing_state() -> dict:
    return load_json_file(
        PROCESSING_STATE_PATH,
        {
            "initialized": False,
            "backfill_completed": False,
            "last_processed_timestamp": None,
            "last_signal_timestamp": None,
            "last_backfill_timestamp": None,
            "last_retry_info": None
        }
    )

def save_processing_state(payload: dict):
    save_json_file(PROCESSING_STATE_PATH, payload)


# =========================================================
# MODEL / SIGNAL HELPERS
# =========================================================
def calc_pnl(side: str, entry_price: float, exit_price: float, lot_size: float = LOT_SIZE) -> float:
    if side == "long":
        return (exit_price - entry_price) * lot_size
    elif side == "short":
        return (entry_price - exit_price) * lot_size
    else:
        raise ValueError("side must be 'long' or 'short'")

def estimate_signal_probability(row: pd.Series) -> Tuple[Optional[float], Optional[int]]:
    if row[feature_columns].isna().any():
        return None, None

    X = pd.DataFrame([row[feature_columns].values], columns=feature_columns)
    prob = float(model.predict_proba(X)[0, 1])
    pred = int(model.predict(X)[0])
    return prob, pred

def should_enter_trade(prob: Optional[float], pred: Optional[int]) -> bool:
    if prob is None or pred is None:
        return False

    if REQUIRE_MODEL_CLASS_1:
        return (pred == 1) and (prob >= PROBABILITY_THRESHOLD)
    return prob >= PROBABILITY_THRESHOLD

def build_signal_payload(row: pd.Series, prob: Optional[float], pred: Optional[int], trade_action: str) -> dict:
    side = "long" if bool(row["bullish_divergence"]) else "short"

    return {
        "signal_timestamp": to_iso_ts(row["timestamp"]),
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "side": side,
        "close": float(row["close"]),
        "rsi": float(row["rsi"]),
        "probability_profitable": None if prob is None else float(prob),
        "prediction": None if pred is None else int(pred),
        "bullish_divergence": bool(row["bullish_divergence"]),
        "bearish_divergence": bool(row["bearish_divergence"]),
        "div_price_change": safe_float(row["div_price_change"]),
        "div_rsi_change": safe_float(row["div_rsi_change"]),
        "div_bars_gap": safe_float(row["div_bars_gap"]),
        "trade_action": trade_action
    }

def signal_exists(signal_timestamp: str) -> bool:
    if not os.path.exists(LIVE_SIGNALS_CSV_PATH):
        return False
    try:
        df = pd.read_csv(LIVE_SIGNALS_CSV_PATH)
        if df.empty or "signal_timestamp" not in df.columns:
            return False
        return (df["signal_timestamp"].astype(str) == str(signal_timestamp)).any()
    except Exception:
        return False

def log_detected_signal_once(row: pd.Series, prob: Optional[float], pred: Optional[int], trade_action: str):
    payload = build_signal_payload(row, prob, pred, trade_action)
    if not signal_exists(payload["signal_timestamp"]):
        append_row_to_csv(LIVE_SIGNALS_CSV_PATH, payload)
    save_latest_signal({
        "status": "ok",
        "timestamp": payload["signal_timestamp"],
        "symbol": payload["symbol"],
        "interval": payload["interval"],
        "side": payload["side"],
        "close": payload["close"],
        "rsi": payload["rsi"],
        "probability_profitable": payload["probability_profitable"],
        "prediction": payload["prediction"],
        "bullish_divergence": payload["bullish_divergence"],
        "bearish_divergence": payload["bearish_divergence"],
        "div_price_change": payload["div_price_change"],
        "div_rsi_change": payload["div_rsi_change"],
        "div_bars_gap": payload["div_bars_gap"],
        "trade_action": trade_action
    })

def trade_exists_by_id(trade_id: str) -> bool:
    if not os.path.exists(LIVE_TRADES_CSV_PATH):
        return False
    try:
        df = pd.read_csv(LIVE_TRADES_CSV_PATH)
        if df.empty or "trade_id" not in df.columns:
            return False
        return (df["trade_id"].astype(str) == str(trade_id)).any()
    except Exception:
        return False

def create_trade_row(open_trade: dict) -> dict:
    return {
        "trade_id": open_trade["trade_id"],
        "symbol": open_trade["symbol"],
        "interval": open_trade["interval"],
        "entry_time": open_trade["entry_time"],
        "entry_side": open_trade["side"],
        "entry_price": open_trade["entry_price"],
        "entry_probability": open_trade["entry_probability"],
        "entry_prediction": open_trade["entry_prediction"],
        "entry_signal_type": open_trade["entry_signal_type"],
        "stop_loss_usd": open_trade["stop_loss_usd"],
        "status": "open",
        "exit_time": None,
        "exit_price": None,
        "exit_reason": None,
        "pnl_usd": None,
        "bars_held": 0,
        "source": open_trade.get("source", "live")
    }

def append_trade_entry_once(open_trade: dict):
    if not trade_exists_by_id(open_trade["trade_id"]):
        append_row_to_csv(LIVE_TRADES_CSV_PATH, create_trade_row(open_trade))

def update_trade_exit(trade_id: str, exit_time: str, exit_price: float, exit_reason: str, pnl_usd: float, bars_held: int):
    row = {
        "trade_id": trade_id,
        "status": "closed",
        "exit_time": exit_time,
        "exit_price": float(exit_price),
        "exit_reason": exit_reason,
        "pnl_usd": float(pnl_usd),
        "bars_held": int(bars_held),
    }
    upsert_csv_row(LIVE_TRADES_CSV_PATH, "trade_id", row)


# =========================================================
# LIVE TRADE ENGINE
# =========================================================
def create_open_trade_from_row(row: pd.Series, prob: float, pred: int, source: str = "live") -> dict:
    side = "long" if bool(row["bullish_divergence"]) else "short"
    signal_type = "bullish_divergence" if side == "long" else "bearish_divergence"

    return {
        "trade_id": str(uuid.uuid4()),
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "entry_time": to_iso_ts(row["timestamp"]),
        "side": side,
        "entry_price": float(row["close"]),
        "entry_probability": float(prob),
        "entry_prediction": int(pred),
        "entry_signal_type": signal_type,
        "stop_loss_usd": REFERENCE_EQUITY * STOP_LOSS_PCT_OF_EQUITY,
        "source": source
    }

def maybe_close_trade(open_trade: dict, row: pd.Series) -> Optional[dict]:
    side = open_trade["side"]
    entry_price = float(open_trade["entry_price"])
    stop_loss_usd = float(open_trade["stop_loss_usd"])
    close_i = float(row["close"])

    running_pnl = calc_pnl(side, entry_price, close_i, LOT_SIZE)

    stop_hit = running_pnl <= -stop_loss_usd
    opposite_hit = (
        (side == "long" and bool(row["bearish_divergence"])) or
        (side == "short" and bool(row["bullish_divergence"]))
    )

    if not stop_hit and not opposite_hit:
        return None

    exit_reason = "stop_loss" if stop_hit else "opposite_divergence"

    entry_ts = parse_ts(open_trade["entry_time"])
    exit_ts = parse_ts(row["timestamp"])
    bars_held = 0
    if entry_ts is not None and exit_ts is not None:
        delta_minutes = (exit_ts - entry_ts).total_seconds() / 60.0
        bars_held = max(0, int(round(delta_minutes / 15.0)))

    exit_payload = {
        "trade_id": open_trade["trade_id"],
        "entry_time": open_trade["entry_time"],
        "exit_time": to_iso_ts(row["timestamp"]),
        "side": side,
        "entry_price": entry_price,
        "exit_price": close_i,
        "pnl_usd": float(running_pnl),
        "exit_reason": exit_reason,
        "bars_held": bars_held
    }

    update_trade_exit(
        trade_id=open_trade["trade_id"],
        exit_time=exit_payload["exit_time"],
        exit_price=close_i,
        exit_reason=exit_reason,
        pnl_usd=float(running_pnl),
        bars_held=bars_held
    )
    clear_open_trade()
    return exit_payload


# =========================================================
# RETROACTIVE BACKFILL
# =========================================================
def build_backfill_start_timestamp(df: pd.DataFrame, state: dict) -> Optional[pd.Timestamp]:
    # priority 1: meta-defined train end timestamp
    meta_train_end = get_training_end_timestamp_from_meta()
    if meta_train_end is not None:
        return meta_train_end

    # priority 2: last_backfill_timestamp from state
    last_backfill = parse_ts(state.get("last_backfill_timestamp"))
    if last_backfill is not None:
        return last_backfill

    # priority 3: earliest available df timestamp
    if df.empty:
        return None
    return pd.Timestamp(df["timestamp"].min())

def run_retroactive_backfill(df: pd.DataFrame, state: dict) -> dict:
    """
    Reconstruct historical signals and completed trades from the backfill start timestamp
    up to the latest available closed candle, using the same exit semantics as training/backtest.
    This writes historical rows to CSV.
    """
    if df.empty:
        return {"backfill_completed": False, "message": "No data available for backfill"}

    start_ts = build_backfill_start_timestamp(df, state)
    if start_ts is None:
        return {"backfill_completed": False, "message": "No valid backfill start timestamp"}

    work_df = df.loc[df["timestamp"] >= start_ts].copy().reset_index(drop=True)
    if work_df.empty:
        return {"backfill_completed": True, "signals_logged": 0, "trades_logged": 0}

    signals_logged = 0
    trades_logged = 0

    i = 0
    while i < len(work_df):
        row = work_df.iloc[i]

        has_signal = bool(row["bullish_divergence"]) or bool(row["bearish_divergence"])
        if not has_signal:
            i += 1
            continue

        signal_ts = to_iso_ts(row["timestamp"])

        prob, pred = estimate_signal_probability(row)
        if prob is None or pred is None:
            if not signal_exists(signal_ts):
                log_detected_signal_once(row, prob, pred, "historical_signal_logged_incomplete_features")
                signals_logged += 1
            i += 1
            continue

        if not signal_exists(signal_ts):
            action = "historical_trade_opened" if should_enter_trade(prob, pred) else "historical_signal_logged_no_trade"
            log_detected_signal_once(row, prob, pred, action)
            signals_logged += 1

        if not should_enter_trade(prob, pred):
            i += 1
            continue

        open_trade = create_open_trade_from_row(row, prob, pred, source="retroactive")
        append_trade_entry_once(open_trade)

        side = open_trade["side"]
        entry_price = float(open_trade["entry_price"])
        stop_loss_usd = float(open_trade["stop_loss_usd"])

        exit_found = False
        for j in range(i + 1, len(work_df)):
            future_row = work_df.iloc[j]
            px = float(future_row["close"])
            running_pnl = calc_pnl(side, entry_price, px, LOT_SIZE)

            stop_hit = running_pnl <= -stop_loss_usd
            opposite_hit = (
                (side == "long" and bool(future_row["bearish_divergence"])) or
                (side == "short" and bool(future_row["bullish_divergence"]))
            )

            if stop_hit or opposite_hit:
                exit_reason = "stop_loss" if stop_hit else "opposite_divergence"
                entry_ts = parse_ts(open_trade["entry_time"])
                exit_ts = parse_ts(future_row["timestamp"])
                bars_held = 0
                if entry_ts is not None and exit_ts is not None:
                    delta_minutes = (exit_ts - entry_ts).total_seconds() / 60.0
                    bars_held = max(0, int(round(delta_minutes / 15.0)))

                update_trade_exit(
                    trade_id=open_trade["trade_id"],
                    exit_time=to_iso_ts(future_row["timestamp"]),
                    exit_price=float(future_row["close"]),
                    exit_reason=exit_reason,
                    pnl_usd=float(running_pnl),
                    bars_held=bars_held
                )
                trades_logged += 1
                i = j + 1
                exit_found = True
                break

        if not exit_found:
            # keep the final historical trade open into live trading only if it is the latest one
            # and only if there is not already an active open trade file.
            if read_open_trade() is None:
                save_open_trade(open_trade)
            trades_logged += 1
            i = len(work_df)

    state["backfill_completed"] = True
    state["last_backfill_timestamp"] = to_iso_ts(work_df["timestamp"].max())
    save_processing_state(state)

    return {
        "backfill_completed": True,
        "signals_logged": signals_logged,
        "trades_logged": trades_logged,
        "start_timestamp": to_iso_ts(start_ts),
        "end_timestamp": to_iso_ts(work_df["timestamp"].max())
    }


# =========================================================
# LIVE PROCESSING
# =========================================================
def build_no_signal_result(message: str, timestamp_value=None, extra: Optional[dict] = None) -> dict:
    out = {
        "status": "ok",
        "message": message,
        "timestamp": to_iso_ts(timestamp_value) if timestamp_value is not None else pd.Timestamp.utcnow().isoformat(),
        "signal": None
    }
    if extra:
        out.update(extra)
    return out

def process_new_rows_live(df: pd.DataFrame, retry_info: Optional[dict] = None) -> dict:
    state = read_processing_state()
    open_trade = read_open_trade()

    if df.empty:
        result = build_no_signal_result("No market data available", extra={"retry_info": retry_info})
        save_latest_signal(result)
        return result

    latest_ts = df["timestamp"].max()
    last_processed_ts = parse_ts(state.get("last_processed_timestamp"))

    if last_processed_ts is None:
        new_rows = df.copy()
    else:
        new_rows = df.loc[df["timestamp"] > last_processed_ts].copy()

    if new_rows.empty:
        last_sig = read_latest_signal()
        if last_sig is not None:
            if retry_info is not None:
                last_sig["retry_info"] = retry_info
            return last_sig

        result = build_no_signal_result("No new closed candle to process", latest_ts, extra={"retry_info": retry_info})
        save_latest_signal(result)
        return result

    latest_result = None

    for _, row in new_rows.iterrows():
        if open_trade is not None:
            exit_payload = maybe_close_trade(open_trade, row)
            if exit_payload is not None:
                latest_result = {
                    "status": "ok",
                    "message": "Trade closed",
                    "event": "trade_closed",
                    "trade": exit_payload,
                    "retry_info": retry_info
                }
                open_trade = None

        has_signal = bool(row["bullish_divergence"]) or bool(row["bearish_divergence"])
        if has_signal:
            signal_ts = to_iso_ts(row["timestamp"])
            last_signal_ts = state.get("last_signal_timestamp")

            if signal_ts != last_signal_ts:
                prob, pred = estimate_signal_probability(row)

                if open_trade is None and should_enter_trade(prob, pred):
                    new_trade = create_open_trade_from_row(row, prob, pred, source="live")
                    append_trade_entry_once(new_trade)
                    save_open_trade(new_trade)
                    log_detected_signal_once(row, prob, pred, "trade_opened")
                    latest_result = {
                        "status": "ok",
                        "message": "New trade opened",
                        "event": "trade_opened",
                        "trade": new_trade,
                        "retry_info": retry_info
                    }
                    open_trade = new_trade
                else:
                    action = "signal_logged_trade_already_open" if open_trade is not None else "signal_logged_no_trade"
                    log_detected_signal_once(row, prob, pred, action)
                    latest_result = {
                        "status": "ok",
                        "message": "Signal processed",
                        "event": action,
                        "timestamp": signal_ts,
                        "retry_info": retry_info
                    }

                state["last_signal_timestamp"] = signal_ts

        state["last_processed_timestamp"] = to_iso_ts(row["timestamp"])

    save_processing_state(state)

    if latest_result is None:
        latest_result = build_no_signal_result(
            "New closed candles processed. No trade action.",
            latest_ts,
            extra={"retry_info": retry_info}
        )

    if "signal" not in latest_result and "trade" not in latest_result and latest_result.get("event") is None:
        save_latest_signal(latest_result)

    return latest_result


# =========================================================
# CORE ENGINE
# =========================================================
def run_inference():
    if model is None or meta is None or feature_columns is None:
        raise RuntimeError("Model not loaded")

    with lock:
        state = read_processing_state()

        df, retry_info = merge_and_update_history_with_retry()
        df = add_features(df, rsi_period=meta["rsi_period"])
        df = detect_divergences(
            df,
            pivot_left=meta["pivot_left"],
            pivot_right=meta["pivot_right"]
        )

        if ENABLE_RETROACTIVE_BACKFILL and not state.get("backfill_completed", False):
            backfill_result = run_retroactive_backfill(df, state)
            state = read_processing_state()
        else:
            backfill_result = {
                "backfill_completed": state.get("backfill_completed", False),
                "signals_logged": 0,
                "trades_logged": 0
            }

        # initialize last_processed_timestamp after backfill, but do not skip future new rows
        if not state.get("initialized", False):
            state["initialized"] = True
            if state.get("last_processed_timestamp") is None:
                # set to current latest only after backfill has consumed history
                state["last_processed_timestamp"] = to_iso_ts(df["timestamp"].max()) if not df.empty else None
            state["last_retry_info"] = retry_info
            save_processing_state(state)

            result = build_no_signal_result(
                "Bootstrap completed",
                df["timestamp"].max() if not df.empty else None,
                extra={
                    "retry_info": retry_info,
                    "backfill_result": backfill_result
                }
            )
            save_latest_signal(result)
            return result

        result = process_new_rows_live(df, retry_info=retry_info)

        # persist retry info
        state = read_processing_state()
        state["last_retry_info"] = retry_info
        save_processing_state(state)

        # attach backfill summary for visibility
        if isinstance(result, dict):
            result["backfill_result"] = backfill_result

        return result


# =========================================================
# SCHEDULER JOB
# =========================================================
def scheduled_job():
    try:
        result = run_inference()
        print(f"[OK] Scheduled inference done: {json.dumps(result, default=str)}")
    except Exception as e:
        print(f"[ERROR] Scheduled inference failed: {e}")


# =========================================================
# FLASK ROUTES
# =========================================================
@app.route("/btc_15m/status", methods=["GET"])
def status():
    open_trade = read_open_trade()
    state = read_processing_state()

    return jsonify({
        "status": "running",
        "model_type": MODEL_TYPE,
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "has_open_trade": open_trade is not None,
        "open_trade": open_trade,
        "processing_state": state,
        "training_end_from_meta": to_iso_ts(get_training_end_timestamp_from_meta())
    })

@app.route("/btc_15m/predict", methods=["GET"])
def predict_now():
    try:
        result = run_inference()
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/btc_15m/latest-signal", methods=["GET"])
def latest_signal():
    payload = read_latest_signal()
    if payload is None:
        return jsonify({"status": "empty", "message": "No signal yet"})
    return jsonify(payload)

@app.route("/btc_15m/open-trade", methods=["GET"])
def get_open_trade():
    payload = read_open_trade()
    if payload is None:
        return jsonify({"status": "empty", "message": "No open trade"})
    return jsonify({"status": "ok", "trade": payload})

@app.route("/btc_15m/trades", methods=["GET"])
def get_trades():
    if not os.path.exists(LIVE_TRADES_CSV_PATH):
        return jsonify({"status": "empty", "message": "No trades logged yet", "rows": []})

    df = pd.read_csv(LIVE_TRADES_CSV_PATH)
    return jsonify({
        "status": "ok",
        "count": int(len(df)),
        "rows": df.tail(200).fillna("").to_dict(orient="records")
    })

@app.route("/btc_15m/signals", methods=["GET"])
def get_signals():
    if not os.path.exists(LIVE_SIGNALS_CSV_PATH):
        return jsonify({"status": "empty", "message": "No signals logged yet", "rows": []})

    df = pd.read_csv(LIVE_SIGNALS_CSV_PATH)
    return jsonify({
        "status": "ok",
        "count": int(len(df)),
        "rows": df.tail(200).fillna("").to_dict(orient="records")
    })

@app.route("/btc_15m/rebuild-history", methods=["POST", "GET"])
def rebuild_history():
    try:
        with lock:
            df, retry_info = merge_and_update_history_with_retry()
            df = add_features(df, rsi_period=meta["rsi_period"])
            df = detect_divergences(
                df,
                pivot_left=meta["pivot_left"],
                pivot_right=meta["pivot_right"]
            )

            state = read_processing_state()
            state["backfill_completed"] = False
            save_processing_state(state)

            result = run_retroactive_backfill(df, state)
            result["retry_info"] = retry_info
            return jsonify({"status": "ok", "result": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# =========================================================
# STARTUP
# =========================================================
def bootstrap():
    ensure_parent_dir(LIVE_CSV_PATH)
    ensure_parent_dir(LATEST_SIGNAL_PATH)
    ensure_parent_dir(LIVE_SIGNALS_CSV_PATH)
    ensure_parent_dir(LIVE_TRADES_CSV_PATH)
    ensure_parent_dir(OPEN_TRADE_PATH)
    ensure_parent_dir(PROCESSING_STATE_PATH)

    load_model()

    try:
        scheduled_job()
    except Exception as e:
        print(f"[BOOTSTRAP ERROR] {e}")

    scheduler.add_job(
        scheduled_job,
        trigger="cron",
        minute="*/15",
        id="btc_15m_job",
        replace_existing=True,
        max_instances=1
    )
    scheduler.start()

bootstrap()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)