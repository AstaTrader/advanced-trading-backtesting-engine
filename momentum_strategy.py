import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum

from strategy import TradeDirection, Signal, StrategyParameters


class MomentumStrategy:
    def __init__(self, params: Optional[StrategyParameters] = None):
        self.params = params or StrategyParameters()
        self.signals: List[Signal] = []
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Moving Averages
        df["sma_20"] = df["close"].rolling(20).mean()
        df["sma_50"] = df["close"].rolling(50).mean()
        df["sma_200"] = df["close"].rolling(200).mean()
        
        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df["close"].ewm(span=12).mean()
        ema_26 = df["close"].ewm(span=26).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        
        # ATR
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()
        
        # Volume
        df["volume_sma"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma"]
        
        # Price Rate of Change
        df["roc_5"] = df["close"].pct_change(5) * 100
        df["roc_10"] = df["close"].pct_change(10) * 100
        
        # ADX (simplified)
        df["plus_dm"] = np.where((df["high"] - df["high"].shift()) > (df["low"].shift() - df["low"]), 
                                 df["high"] - df["high"].shift(), 0)
        df["minus_dm"] = np.where((df["low"].shift() - df["low"]) > (df["high"] - df["high"].shift()), 
                                  df["low"].shift() - df["low"], 0)
        df["plus_di"] = 100 * (df["plus_dm"].rolling(14).mean() / df["atr"])
        df["minus_di"] = 100 * (df["minus_dm"].rolling(14).mean() / df["atr"])
        df["adx"] = 100 * (np.abs(df["plus_di"] - df["minus_di"]) / (df["plus_di"] + df["minus_di"])).rolling(14).mean()
        
        return df
    
    def check_momentum_long(self, df: pd.DataFrame, idx: int) -> Optional[Dict]:
        if idx < 200:  # Need enough data
            return None
        
        row = df.iloc[idx]
        prev = df.iloc[idx - 1]
        
        confirmations = 0
        
        # 1. Price above key moving averages
        if row["close"] > row["sma_20"] > row["sma_50"] > row["sma_200"]:
            confirmations += 1
        
        # 2. Strong momentum (ROC)
        if row["roc_5"] > 0.5 and row["roc_10"] > 1.0:
            confirmations += 1
        
        # 3. RSI not overbought but strong
        if 50 < row["rsi"] < 75:
            confirmations += 1
        
        # 4. MACD bullish
        if row["macd"] > row["macd_signal"] and row["macd"] > 0:
            confirmations += 1
        
        # 5. Volume confirmation
        if row["volume_ratio"] > 1.2:
            confirmations += 1
        
        # 6. ADX showing strength
        if row["adx"] > 25 and row["plus_di"] > row["minus_di"]:
            confirmations += 1
        
        # 7. Pullback opportunity
        if (row["low"] <= row["sma_20"] * 1.02 and 
            row["close"] > row["sma_20"] and
            prev["close"] < row["sma_20"]):
            confirmations += 1
        
        # Need at least 5 confirmations - sweet spot
        if confirmations >= 5:
            return {
                "confirmations": confirmations,
                "entry_price": row["close"],
                "stop_loss": row["low"] - (row["atr"] * 1.8),  # Balanced stops
                "take_profit": row["close"] + (row["atr"] * 3.5)  # Balanced targets
            }
        
        return None
    
    def check_momentum_short(self, df: pd.DataFrame, idx: int) -> Optional[Dict]:
        if idx < 200:
            return None
        
        row = df.iloc[idx]
        prev = df.iloc[idx - 1]
        
        confirmations = 0
        
        # 1. Price below key moving averages
        if row["close"] < row["sma_20"] < row["sma_50"] < row["sma_200"]:
            confirmations += 1
        
        # 2. Strong downward momentum
        if row["roc_5"] < -0.5 and row["roc_10"] < -1.0:
            confirmations += 1
        
        # 3. RSI not oversold but bearish
        if 25 < row["rsi"] < 50:
            confirmations += 1
        
        # 4. MACD bearish
        if row["macd"] < row["macd_signal"] and row["macd"] < 0:
            confirmations += 1
        
        # 5. Volume confirmation
        if row["volume_ratio"] > 1.2:
            confirmations += 1
        
        # 6. ADX showing strength
        if row["adx"] > 25 and row["minus_di"] > row["plus_di"]:
            confirmations += 1
        
        # 7. Pullback opportunity
        if (row["high"] >= row["sma_20"] * 0.98 and 
            row["close"] < row["sma_20"] and
            prev["close"] > row["sma_20"]):
            confirmations += 1
        
        # Need at least 5 confirmations - sweet spot
        if confirmations >= 5:
            return {
                "confirmations": confirmations,
                "entry_price": row["close"],
                "stop_loss": row["high"] + (row["atr"] * 1.8),  # Balanced stops
                "take_profit": row["close"] - (row["atr"] * 3.5)  # Balanced targets
            }
        
        return None
    
    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        df = self.calculate_indicators(df)
        self.signals = []
        
        for i in range(len(df)):
            timestamp = df.index[i]
            
            long_signal = self.check_momentum_long(df, i)
            if long_signal:
                signal = Signal(
                    direction=TradeDirection.LONG,
                    timestamp=timestamp,
                    entry_price=long_signal["entry_price"],
                    stop_loss=long_signal["stop_loss"],
                    take_profit=long_signal["take_profit"],
                    ema_touched=20,
                    wick_ratio=long_signal["confirmations"]
                )
                self.signals.append(signal)
            
            short_signal = self.check_momentum_short(df, i)
            if short_signal:
                signal = Signal(
                    direction=TradeDirection.SHORT,
                    timestamp=timestamp,
                    entry_price=short_signal["entry_price"],
                    stop_loss=short_signal["stop_loss"],
                    take_profit=short_signal["take_profit"],
                    ema_touched=20,
                    wick_ratio=short_signal["confirmations"]
                )
                self.signals.append(signal)
        
        self.signals.sort(key=lambda x: x.timestamp)
        return self.signals
