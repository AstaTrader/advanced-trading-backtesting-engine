import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Optional
import numpy as np


class BinanceDataFetcher:
    BASE_URL = "https://api.binance.com/api/v3/klines"
    
    def __init__(self):
        self.session = requests.Session()
    
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        params = {
            "symbol": symbol.upper(),
            "interval": timeframe,
            "limit": limit
        }
        
        if start_date:
            start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
            params["startTime"] = start_ts
        if end_date:
            end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
            params["endTime"] = end_ts
        
        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                raise ValueError(f"No data returned for {symbol}")
            
            df = pd.DataFrame(data, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades", "taker_buy_volume",
                "taker_buy_quote_volume", "ignore"
            ])
            
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            numeric_cols = ["open", "high", "low", "close", "volume"]
            df[numeric_cols] = df[numeric_cols].astype(float)
            
            df.set_index("timestamp", inplace=True)
            df = df[["open", "high", "low", "close", "volume"]]
            
            return df
            
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to fetch data: {e}")


def generate_sample_data(
    symbol: str = "BTCUSDT",
    periods: int = 5000,
    start_date: str = "2023-01-01"
) -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range(start=start_date, periods=periods, freq="1h")
    
    returns = np.random.normal(0.0001, 0.02, periods)
    price = 30000 * np.exp(np.cumsum(returns))
    
    volatility = np.abs(returns) * price
    high = price + np.abs(np.random.normal(0, volatility))
    low = price - np.abs(np.random.normal(0, volatility))
    open_price = price + np.random.normal(0, volatility * 0.3)
    
    high = np.maximum(high, np.maximum(open_price, price))
    low = np.minimum(low, np.minimum(open_price, price))
    
    volume = np.random.lognormal(15, 1, periods)
    
    df = pd.DataFrame({
        "open": open_price,
        "high": high,
        "low": low,
        "close": price,
        "volume": volume
    }, index=dates)
    
    return df