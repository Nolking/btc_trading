# src/main.py
from pathlib import Path

import matplotlib.pyplot as plt

from src.data_loader import load_bitcoin_data
from src.features import add_technical_features
from src.ml_strategy import (
    create_labels,
    build_ml_dataset,
    train_test_split_time_series,
    train_random_forest_classifier,
    apply_model_to_data,
)
from src.backtest import backtest_strategy


def main():
    data_path = Path("data/bitcoin_prices.csv")
    df = load_bitcoin_data(data_path)

    # 1) Add features
    df_feat = add_technical_features(df)

    # Choose which features to use for ML
    feature_cols = [
        "log_return_1",
        "log_return_3",
        "log_return_5",
        "rolling_vol_10",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_middle_20",
        "bb_upper_20",
        "bb_lower_20",
    ]

    # 2) Create labels (future direction)
    df_labeled = create_labels(df_feat, horizon=3, threshold=0.0)

    # 3) Build dataset
    X, y = build_ml_dataset(df_labeled, feature_cols)

    # 4) Time-based train/test split
    X_train, X_test, y_train, y_test = train_test_split_time_series(X, y, train_ratio=0.7)

    # 5) Train model
    model = train_random_forest_classifier(X_train, y_train)

    # (Optional) Quick check of accuracy on test set
    test_accuracy = model.score(X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.3f}")

    # 6) Apply model to full dataframe and generate 'signal'
    df_with_signals = apply_model_to_data(df_labeled, model, feature_cols, proba_threshold=0.55)

    # 7) Backtest the ML-based signals
    results = backtest_strategy(
        df_with_signals,
        signal_col="signal",
        price_col="close",
        trading_cost_bps=10,  # 0.1% per trade
    )

    bt_df = results["df"]
    print(bt_df)
    print("=== ML Strategy Backtest Results (Random Forest) ===")
    print(f"Start date:        {bt_df.index[0]}")
    print(f"End date:          {bt_df.index[-1]}")
    print(f"Total return:      {results['total_return']:.2f}%")
    print(f"Annualized return: {results['annualized_return']:.2f}%")
    print(f"Max drawdown:      {results['max_drawdown']:.2f}%")

    # 8) Compare equity vs Buy & Hold
    plt.figure(figsize=(10, 5))
    bt_df["equity_curve"].plot(label="ML Strategy Equity")
    (bt_df["close"] / bt_df["close"].iloc[0]).plot(alpha=0.7, label="Buy & Hold BTC")
    plt.title("BTC Backtest - ML Strategy (Random Forest)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
