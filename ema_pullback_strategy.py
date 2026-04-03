"""
9 & 20 EMA Pullback with Wick Rejection Strategy
Professional Backtesting Implementation

Author: Quantitative Trading Developer
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from enum import Enum
import requests
import time


class TradeDirection(Enum):
    LONG = "long"
    SHORT = "short"


class TradeStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"


@dataclass
class StrategyParameters:
    """Strategy parameters with default values"""
    # EMA Settings
    ema_fast: int = 9
    ema_slow: int = 20
    
    # Slope Analysis
    slope_lookback: int = 3
    min_slope_pct: float = 0.01
    
    # Wick Rejection
    wick_ratio_threshold: float = 1.5
    min_wick_pct: float = 0.002
    
    # Risk Management
    risk_reward: float = 2.0
    use_breakeven: bool = True
    breakeven_trigger_rr: float = 1.0
    partial_close_rr: float = 2.0
    partial_close_pct: float = 0.5
    use_trailing_stop: bool = True
    
    # Filters
    use_atr_filter: bool = True
    atr_period: int = 14
    min_atr_pct: float = 0.005
    min_ema_distance_pct: float = 0.003


@dataclass
class Signal:
    """Trading signal data structure"""
    direction: TradeDirection
    timestamp: pd.Timestamp
    entry_price: float
    stop_loss: float
    take_profit: float
    ema_touched: int
    wick_ratio: float
    confidence: float = 1.0


@dataclass
class Trade:
    """Trade execution data structure"""
    trade_id: int
    direction: TradeDirection
    entry_time: datetime
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float = 1.0
    
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    status: TradeStatus = TradeStatus.OPEN
    
    pnl: float = 0.0
    pnl_pct: float = 0.0
    rr_achieved: float = 0.0
    
    def calculate_pnl(self, exit_price: float, size: float = 1.0) -> float:
        """Calculate profit/loss for the trade"""
        if self.direction == TradeDirection.LONG:
            return (exit_price - self.entry_price) * size
        else:
            return (self.entry_price - exit_price) * size


@dataclass
class BacktestResult:
    """Backtest performance metrics"""
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # Performance Metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_profit: float = 0.0
    total_loss: float = 0.0
    net_profit: float = 0.0
    avg_trade: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    
    # Risk Metrics
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    
    # R-Multiple Metrics
    total_r: float = 0.0
    avg_r: float = 0.0


class DataFetcher:
    """Data fetching from Binance API"""
    
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
        """Fetch OHLCV data from Binance"""
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
        self,
        symbol: str = "BTCUSDT",
        periods: int = 5000,
        start_date: str = "2023-01-01"
    ) -> pd.DataFrame:
        """Generate sample data for testing"""
        np.random.seed(42)
        dates = pd.date_range(start=start_date, periods=periods, freq="1h")
        
        # Generate realistic price movement
        returns = np.random.normal(0.0001, 0.02, periods)
        price = 30000 * np.exp(np.cumsum(returns))
        
        # Generate OHLC
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


class EMAPullbackStrategy:
    """9 & 20 EMA Pullback with Wick Rejection Strategy"""
    
    def __init__(self, params: Optional[StrategyParameters] = None):
        self.params = params or StrategyParameters()
        self.signals: List[Signal] = []
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all required indicators"""
        df = df.copy()
        
        # EMAs
        df["ema_fast"] = df["close"].ewm(span=self.params.ema_fast, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=self.params.ema_slow, adjust=False).mean()
        
        # EMA Slopes
        df["ema_fast_slope"] = df["ema_fast"].pct_change(self.params.slope_lookback) * 100
        df["ema_slow_slope"] = df["ema_slow"].pct_change(self.params.slope_lookback) * 100
        
        # ATR
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(self.params.atr_period).mean()
        df["atr_pct"] = df["atr"] / df["close"]
        
        # Candlestick Analysis
        df["body"] = np.abs(df["close"] - df["open"])
        df["upper_wick"] = df["high"] - np.maximum(df["open"], df["close"])
        df["lower_wick"] = np.minimum(df["open"], df["close"]) - df["low"]
        
        df["upper_wick_ratio"] = df["upper_wick"] / (df["body"] + 1e-10)
        df["lower_wick_ratio"] = df["lower_wick"] / (df["body"] + 1e-10)
        
        df["is_bullish"] = df["close"] > df["open"]
        df["is_bearish"] = df["close"] < df["open"]
        
        # EMA Distance Filter
        df["ema_distance"] = np.abs(df["ema_fast"] - df["ema_slow"]) / df["close"]
        
        return df
    
    def find_swing_low(self, df: pd.DataFrame, idx: int, lookback: int = 10) -> float:
        """Find nearest swing low"""
        start = max(0, idx - lookback)
        return df.iloc[start:idx]["low"].min()
    
    def find_swing_high(self, df: pd.DataFrame, idx: int, lookback: int = 10) -> float:
        """Find nearest swing high"""
        start = max(0, idx - lookback)
        return df.iloc[start:idx]["high"].max()
    
    def check_long_conditions(self, df: pd.DataFrame, idx: int) -> Optional[Dict]:
        """Check long entry conditions"""
        if idx < max(self.params.ema_slow, self.params.atr_period) + 10:
            return None
        
        row = df.iloc[idx]
        prev = df.iloc[idx - 1]
        
        # 1. Price above both EMAs
        if not (row["close"] > row["ema_fast"] and row["close"] > row["ema_slow"]):
            return None
        
        # 2. EMA 9 above EMA 20
        if not (row["ema_fast"] > row["ema_slow"]):
            return None
        
        # 3. EMAs sloping upward
        if not (row["ema_fast_slope"] > self.params.min_slope_pct and 
                row["ema_slow_slope"] > self.params.min_slope_pct):
            return None
        
        # 4. Pullback to EMA
        touched_9 = prev["low"] <= prev["ema_fast"] or row["low"] <= row["ema_fast"]
        touched_20 = prev["low"] <= prev["ema_slow"] or row["low"] <= row["ema_slow"]
        
        if not (touched_9 or touched_20):
            return None
        
        ema_touched = 9 if touched_9 else 20
        
        # 5. Lower wick rejection
        wick_ratio = row["lower_wick_ratio"]
        wick_size_pct = row["lower_wick"] / row["close"]
        
        if not (wick_ratio >= self.params.wick_ratio_threshold and 
                wick_size_pct >= self.params.min_wick_pct):
            return None
        
        # 6. Bullish close
        if not row["is_bullish"]:
            return None
        
        # Filters
        if self.params.use_atr_filter and row["atr_pct"] < self.params.min_atr_pct:
            return None
        
        if row["ema_distance"] < self.params.min_ema_distance_pct:
            return None
        
        return {
            "ema_touched": ema_touched,
            "wick_ratio": wick_ratio,
            "entry_price": row["close"]
        }
    
    def check_short_conditions(self, df: pd.DataFrame, idx: int) -> Optional[Dict]:
        """Check short entry conditions"""
        if idx < max(self.params.ema_slow, self.params.atr_period) + 10:
            return None
        
        row = df.iloc[idx]
        prev = df.iloc[idx - 1]
        
        # 1. Price below both EMAs
        if not (row["close"] < row["ema_fast"] and row["close"] < row["ema_slow"]):
            return None
        
        # 2. EMA 9 below EMA 20
        if not (row["ema_fast"] < row["ema_slow"]):
            return None
        
        # 3. EMAs sloping downward
        if not (row["ema_fast_slope"] < -self.params.min_slope_pct and 
                row["ema_slow_slope"] < -self.params.min_slope_pct):
            return None
        
        # 4. Pullback to EMA
        touched_9 = prev["high"] >= prev["ema_fast"] or row["high"] >= row["ema_fast"]
        touched_20 = prev["high"] >= prev["ema_slow"] or row["high"] >= row["ema_slow"]
        
        if not (touched_9 or touched_20):
            return None
        
        ema_touched = 9 if touched_9 else 20
        
        # 5. Upper wick rejection
        wick_ratio = row["upper_wick_ratio"]
        wick_size_pct = row["upper_wick"] / row["close"]
        
        if not (wick_ratio >= self.params.wick_ratio_threshold and 
                wick_size_pct >= self.params.min_wick_pct):
            return None
        
        # 6. Bearish close
        if not row["is_bearish"]:
            return None
        
        # Filters
        if self.params.use_atr_filter and row["atr_pct"] < self.params.min_atr_pct:
            return None
        
        if row["ema_distance"] < self.params.min_ema_distance_pct:
            return None
        
        return {
            "ema_touched": ema_touched,
            "wick_ratio": wick_ratio,
            "entry_price": row["close"]
        }
    
    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """Generate trading signals"""
        df = self.calculate_indicators(df)
        self.signals = []
        
        for i in range(len(df)):
            row = df.iloc[i]
            timestamp = df.index[i]
            
            # Check long conditions
            long_result = self.check_long_conditions(df, i)
            if long_result:
                entry = long_result["entry_price"]
                
                # Stop loss below wick low or swing low
                stop_below_wick = row["low"] - (row["lower_wick"] * 0.2)
                swing_low = self.find_swing_low(df, i)
                stop = min(stop_below_wick, swing_low * 0.998)
                
                # Take profit with minimum 1:2 RR
                risk = entry - stop
                target = entry + (risk * self.params.risk_reward)
                
                signal = Signal(
                    direction=TradeDirection.LONG,
                    timestamp=timestamp,
                    entry_price=entry,
                    stop_loss=stop,
                    take_profit=target,
                    ema_touched=long_result["ema_touched"],
                    wick_ratio=long_result["wick_ratio"]
                )
                self.signals.append(signal)
            
            # Check short conditions
            short_result = self.check_short_conditions(df, i)
            if short_result:
                entry = short_result["entry_price"]
                
                # Stop loss above wick high or swing high
                stop_above_wick = row["high"] + (row["upper_wick"] * 0.2)
                swing_high = self.find_swing_high(df, i)
                stop = max(stop_above_wick, swing_high * 1.002)
                
                # Take profit with minimum 1:2 RR
                risk = stop - entry
                target = entry - (risk * self.params.risk_reward)
                
                signal = Signal(
                    direction=TradeDirection.SHORT,
                    timestamp=timestamp,
                    entry_price=entry,
                    stop_loss=stop,
                    take_profit=target,
                    ema_touched=short_result["ema_touched"],
                    wick_ratio=short_result["wick_ratio"]
                )
                self.signals.append(signal)
        
        self.signals.sort(key=lambda x: x.timestamp)
        return self.signals
