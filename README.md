## Descriptions
- The project retrieve data of Bitcoin from Binance and train the model using Random Forest or XGBoost algorithm. The Flask live server can retrieve data every 15 minutes to check for live signals

## Requirements
- Python: >=3.10

## Retrieve 5 years data from Binance
- python src/binance.py

## Train random forest model
- python3 src/rf/split_train_test.py
- python3 src/rf/train_rf_rsi.py
- python3 src/rf/backtest_rf_rsi.py

## Train xgb model
- python3 src/xgb/split_train_test.py
- python3 src/xgb/train_xgb_rsi.py
- python3 src/xgbf/backtest_xgb_rsi.py

## Copy the models for live server
- mkdir models
- cp src/rf/rf_divergence_model.pkl models/rf_divergence_model.pkl
- cp src/xgb/xgb_divergence_model.pkl models/xgb_divergence_model.pkl

## How to run Flask server
- pip install -r requirements.txt
- python app.py

