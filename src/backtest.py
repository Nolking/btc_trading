# src/backtest.py
import pandas as pd

def backtest_strategy(
    df: pd.DataFrame,
    signal_col: str = "signal",
    price_col: str = "close",
    trading_cost_bps: float = 0.0
) -> dict:
    df = df.copy()

    df["returns"] = df[price_col].pct_change()
    df["position"] = df[signal_col].shift(1).fillna(0)
    df["strategy_returns"] = df["position"] * df["returns"]

    if trading_cost_bps > 0:
        df["pos_change"] = df["position"].diff().abs().fillna(0)
        cost_per_trade = trading_cost_bps / 10_000.0
        df["cost"] = df["pos_change"] * cost_per_trade
        df["strategy_returns"] -= df["cost"]
    else:
        df["cost"] = 0.0

    df["equity_curve"] = (1 + df["strategy_returns"]).cumprod()

    total_return = df["equity_curve"].iloc[-1] - 1
    num_days = (df.index[-1] - df.index[0]).days
    if num_days > 0:
        ann_factor = 365 / num_days
        ann_return = (1 + total_return) ** ann_factor - 1
    else:
        ann_return = 0.0

    max_drawdown = compute_max_drawdown(df["equity_curve"])

    return {
        "df": df,
        "total_return": total_return,
        "annualized_return": ann_return,
        "max_drawdown": max_drawdown,
    }


def compute_max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return drawdown.min()
