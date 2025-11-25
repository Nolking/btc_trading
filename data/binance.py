import requests
import pandas as pd
import time
from datetime import datetime, timedelta

def get_binance_klines(symbol, interval, start_time=None, end_time=None, limit=1000):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }
    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time

    response = requests.get(url, params=params)
    data = response.json()

    df = pd.DataFrame(data, columns=[
        "Open Time", "Open", "High", "Low", "Close", "Volume",
        "Close Time", "Quote Asset Volume", "Number of Trades",
        "Taker Buy Base Volume", "Taker Buy Quote Volume", "Ignore"
    ])
    df["Open Time"] = pd.to_datetime(df["Open Time"], unit='ms')
    df["Close Time"] = pd.to_datetime(df["Close Time"], unit='ms')
    return df

def get_full_klines(symbol, interval, days=730):
    # Compute how many candles we need and how many per request
    interval_in_minutes = {
        "1m": 1, "3m": 3, "5m": 5, "15m": 15,
        "30m": 30, "1h": 60, "2h": 120, "4h": 240,
        "6h": 360, "8h": 480, "12h": 720,
        "1d": 1440, "3d": 4320, "1w": 10080
    }[interval]

    candles_needed = int((days * 1440) / interval_in_minutes)
    limit = 1000
    loops = (candles_needed // limit) + 1

    print(f"Fetching approx {candles_needed} candles in {loops} request(s)...")

    df_list = []
    end_time = int(time.time() * 1000)  # current time in ms

    for i in range(loops):
        start_time = end_time - (limit * interval_in_minutes * 60 * 1000)
        print(f"Fetching {symbol} {interval} data from {datetime.utcfromtimestamp(start_time / 1000)} to {datetime.utcfromtimestamp(end_time / 1000)}")
        df = get_binance_klines(symbol, interval, start_time=start_time, end_time=end_time)
        if df.empty:
            break
        df_list.insert(0, df)
        end_time = int(df["Open Time"].iloc[0].timestamp() * 1000) - 1
        time.sleep(0.5)  # be nice to the API

    full_df = pd.concat(df_list).reset_index(drop=True)
    return full_df

# --- Parameters ---
symbol = "BTCUSDT"
interval = "15m"   # Change to "1d" for daily chart
days_back = 730   # 2 years

# --- Run the data fetch and save ---
df = get_full_klines(symbol, interval, days=days_back)
csv_filename = f"{symbol}_{interval}_{days_back}d.csv"
df.to_csv(csv_filename, index=False)
print(f"\n✅ Data saved to: {csv_filename}")
