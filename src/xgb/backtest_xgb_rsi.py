import os
import json
import joblib
import numpy as np
import pandas as pd


# =========================================================
# CONFIG
# =========================================================
TEST_CSV_PATH = "bitcoin_test.csv"

MODEL_PATH = "xgb_divergence_model.pkl"
META_PATH = "xgb_divergence_model_meta.json"

TRADES_OUTPUT_PATH = "xgb_backtest_divergence_trades.csv"
SIGNALS_OUTPUT_PATH = "xgb_backtest_divergence_signals.csv"
EQUITY_OUTPUT_PATH = "xgb_backtest_equity_curve.csv"

INITIAL_CAPITAL = 10000.0
LOT_SIZE = 0.05
MAX_DAILY_TRADES = 2
MAX_TOTAL_LOSS_PCT = 0.05
STOP_LOSS_PCT_OF_EQUITY = 0.01
RSI_PERIOD = 30

PIVOT_LEFT = 3
PIVOT_RIGHT = 3

USE_MODEL = True
PROBABILITY_THRESHOLD = 0.58
WAIT_FOR_NEXT_DIVERGENCE_IF_PROB_BELOW_THRESHOLD = True
WAIT_FOR_NEXT_DIVERGENCE_IF_NO_MODEL = True


def ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


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


def load_model_and_features(model_path: str, meta_path: str):
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        return None, None

    model = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return model, meta["feature_columns"]


def estimate_reversal_probability(row: pd.Series, model, feature_columns: list):
    if model is None or feature_columns is None:
        return None

    try:
        X = pd.DataFrame([row[feature_columns].values], columns=feature_columns)
        prob = model.predict_proba(X)[0, 1]
        return float(prob)
    except Exception:
        return None


def calc_pnl(side: str, entry_price: float, exit_price: float, lot_size: float = LOT_SIZE) -> float:
    if side == "long":
        return (exit_price - entry_price) * lot_size
    elif side == "short":
        return (entry_price - exit_price) * lot_size
    else:
        raise ValueError("side must be 'long' or 'short'")


def run_backtest(df: pd.DataFrame, model=None, feature_columns=None):
    equity = INITIAL_CAPITAL
    lockout_equity = INITIAL_CAPITAL * (1 - MAX_TOTAL_LOSS_PCT)

    open_trade = None
    trades = []
    equity_curve = []

    daily_trade_count = {}
    bullish_waiting = False
    bearish_waiting = False

    for i, row in df.iterrows():
        timestamp = row["timestamp"]
        date_i = row["date"]
        close_i = float(row["close"])

        if date_i not in daily_trade_count:
            daily_trade_count[date_i] = 0

        floating_pnl = 0.0
        if open_trade is not None:
            floating_pnl = calc_pnl(open_trade["side"], open_trade["entry_price"], close_i, LOT_SIZE)

        equity_curve.append({
            "timestamp": timestamp,
            "close": close_i,
            "realized_equity": equity,
            "floating_pnl": floating_pnl,
            "equity_with_floating": equity + floating_pnl,
            "has_open_trade": int(open_trade is not None),
        })

        if open_trade is not None:
            side = open_trade["side"]
            entry_price = open_trade["entry_price"]
            stop_loss_usd = open_trade["stop_loss_usd"]

            running_pnl = calc_pnl(side, entry_price, close_i, LOT_SIZE)

            stop_hit = running_pnl <= -stop_loss_usd
            opposite_hit = (
                (side == "long" and bool(row["bearish_divergence"])) or
                (side == "short" and bool(row["bullish_divergence"]))
            )

            if stop_hit or opposite_hit:
                exit_reason = "stop_loss" if stop_hit else "reversal"
                equity += running_pnl

                trades.append({
                    "entry_exit_time": f"Entry: {open_trade['entry_time']}\nExit: {timestamp}",
                    "side": side,
                    "entry_price": entry_price,
                    "exit_price": close_i,
                    "pnl_usd": running_pnl,
                    "equity_after": equity,
                    "exit_reason": exit_reason,
                    "entry_probability": open_trade["entry_probability"],
                    "trade_day": open_trade["trade_day"],
                    "stop_loss_usd": stop_loss_usd,
                })

                open_trade = None
                continue

        if equity <= lockout_equity:
            continue

        if open_trade is not None:
            continue

        if daily_trade_count[date_i] >= MAX_DAILY_TRADES:
            continue

        enter_side = None
        entry_probability = None

        if bool(row["bullish_divergence"]):
            prob = estimate_reversal_probability(row, model, feature_columns) if USE_MODEL else None
            entry_probability = prob

            if prob is not None:
                if prob >= PROBABILITY_THRESHOLD:
                    enter_side = "long"
                    bullish_waiting = False
                elif WAIT_FOR_NEXT_DIVERGENCE_IF_PROB_BELOW_THRESHOLD:
                    if bullish_waiting:
                        enter_side = "long"
                        bullish_waiting = False
                    else:
                        bullish_waiting = True
            else:
                if WAIT_FOR_NEXT_DIVERGENCE_IF_NO_MODEL:
                    if bullish_waiting:
                        enter_side = "long"
                        bullish_waiting = False
                    else:
                        bullish_waiting = True
                else:
                    enter_side = "long"

        elif bool(row["bearish_divergence"]):
            prob = estimate_reversal_probability(row, model, feature_columns) if USE_MODEL else None
            entry_probability = prob

            if prob is not None:
                if prob >= PROBABILITY_THRESHOLD:
                    enter_side = "short"
                    bearish_waiting = False
                elif WAIT_FOR_NEXT_DIVERGENCE_IF_PROB_BELOW_THRESHOLD:
                    if bearish_waiting:
                        enter_side = "short"
                        bearish_waiting = False
                    else:
                        bearish_waiting = True
            else:
                if WAIT_FOR_NEXT_DIVERGENCE_IF_NO_MODEL:
                    if bearish_waiting:
                        enter_side = "short"
                        bearish_waiting = False
                    else:
                        bearish_waiting = True
                else:
                    enter_side = "short"

        if enter_side is not None:
            open_trade = {
                "entry_time": timestamp,
                "entry_index": i,
                "trade_day": date_i,
                "side": enter_side,
                "entry_price": close_i,
                "entry_probability": entry_probability,
                # "stop_loss_usd": equity * STOP_LOSS_PCT_OF_EQUITY,
                "stop_loss_usd": 100.0
            }
            daily_trade_count[date_i] += 1

    if open_trade is not None:
        final_row = df.iloc[-1]
        final_price = float(final_row["close"])
        realized_pnl = calc_pnl(open_trade["side"], open_trade["entry_price"], final_price, LOT_SIZE)
        equity += realized_pnl

        trades.append({
            "entry_exit_time": f"Entry: {open_trade['entry_time']}\nExit: {final_row["timestamp"]}",
            "side": open_trade["side"],
            "entry_price": open_trade["entry_price"],
            "exit_price": final_price,
            "pnl_usd": realized_pnl,
            "equity_after": equity,
            "exit_reason": "end_of_data",
            "entry_probability": open_trade["entry_probability"],
            "trade_day": open_trade["trade_day"],
            "stop_loss_usd": open_trade["stop_loss_usd"],
        })

    trades_df = pd.DataFrame(trades)
    equity_curve_df = pd.DataFrame(equity_curve)
    if len(trades_df):
        winning_trades = trades_df.loc[trades_df["pnl_usd"] > 0, "pnl_usd"]
        losing_trades = trades_df.loc[trades_df["pnl_usd"] <= 0, "pnl_usd"]
    else:
        winning_trades = pd.Series(dtype=float)
        losing_trades = pd.Series(dtype=float)

    summary = {
        "initial_capital": INITIAL_CAPITAL,
        "final_equity": equity,
        "net_profit": equity - INITIAL_CAPITAL,
        "max_allowed_loss_pct": MAX_TOTAL_LOSS_PCT,
        "lockout_equity_level": lockout_equity,
        "num_trades": int(len(trades_df)),
        "wins": int(len(winning_trades)),
        "losses": int(len(losing_trades)),
        "win_rate": float((trades_df["pnl_usd"] > 0).mean()) if len(trades_df) else 0.0,
        "average_win": float(winning_trades.mean()) if len(winning_trades) else 0.0,
        "average_loss": float(losing_trades.mean()) if len(losing_trades) else 0.0,
        "highest_win": float(winning_trades.max()) if len(winning_trades) else 0.0,
        "lowest_win": float(winning_trades.min()) if len(winning_trades) else 0.0,
    }

    return trades_df, equity_curve_df, summary


if __name__ == "__main__":
    ensure_parent_dir(TRADES_OUTPUT_PATH)
    ensure_parent_dir(SIGNALS_OUTPUT_PATH)
    ensure_parent_dir(EQUITY_OUTPUT_PATH)

    df = load_data(TEST_CSV_PATH)
    df = add_features(df)
    df = detect_divergences(df)

    model, feature_columns = load_model_and_features(MODEL_PATH, META_PATH)

    if USE_MODEL and model is None:
        print("Warning: model/meta not found. Backtest will use divergence logic only.")

    trades_df, equity_curve_df, summary = run_backtest(df, model=model, feature_columns=feature_columns)

    df.to_csv(SIGNALS_OUTPUT_PATH, index=False)
    trades_df.to_csv(TRADES_OUTPUT_PATH, index=False)
    equity_curve_df.to_csv(EQUITY_OUTPUT_PATH, index=False)

    print("=== Backtest Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\nSaved:")
    print(SIGNALS_OUTPUT_PATH)
    print(TRADES_OUTPUT_PATH)
    print(EQUITY_OUTPUT_PATH)