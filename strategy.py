import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum


class TradeDirection(Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class Signal:
    direction: TradeDirection
    timestamp: pd.Timestamp
    entry_price: float
    stop_loss: float
    take_profit: float
    ema_touched: int
    wick_ratio: float


@dataclass 
class StrategyParameters:
    # EMA Settings
    ema_fast: int = 9
    ema_slow: int = 20
    ema_trend: int = 50  # Trend confirmation EMA
    slope_lookback: int = 3
    min_slope_pct: float = 0.01
    
    # Wick Rejection Settings
    wick_ratio_threshold: float = 1.5
    min_wick_pct: float = 0.002
    
    # RSI Settings
    rsi_period: int = 14
    rsi_oversold: float = 30
    rsi_overbought: float = 70
    rsi_pullback_low: float = 40
    rsi_pullback_high: float = 60
    
    # MACD Settings
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Bollinger Bands Settings
    bb_period: int = 20
    bb_std: float = 2.0
    bb_squeeze_threshold: float = 0.1
    
    # Volume Settings
    volume_ma_period: int = 20
    volume_spike_threshold: float = 1.5
    min_volume_ratio: float = 0.8
    
    # Stochastic Settings
    stoch_k: int = 14
    stoch_d: int = 3
    stoch_oversold: float = 20
    stoch_overbought: float = 80
    
    # ATR Settings
    atr_period: int = 14
    min_atr_pct: float = 0.005
    atr_multiplier: float = 2.0
    
    # Risk Management
    risk_reward: float = 2.5  # Increased RR
    use_breakeven: bool = True
    breakeven_trigger_rr: float = 1.0
    partial_close_rr: float = 2.0
    partial_close_pct: float = 0.5
    
    # Confirmation Filters
    use_rsi_filter: bool = True
    use_macd_filter: bool = True
    use_bb_filter: bool = True
    use_volume_filter: bool = True
    use_stoch_filter: bool = True
    use_trend_filter: bool = True
    
    # Advanced Filters
    min_confirmations: int = 3  # Minimum confirmations required
    use_confluence: bool = True  # Require multiple indicators alignment
    use_volatility_filter: bool = True


class EMAPullbackStrategy:
    def __init__(self, params: Optional[StrategyParameters] = None):
        self.params = params or StrategyParameters()
        self.signals: List[Signal] = []
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # EMAs
        df["ema_fast"] = df["close"].ewm(span=self.params.ema_fast, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=self.params.ema_slow, adjust=False).mean()
        df["ema_trend"] = df["close"].ewm(span=self.params.ema_trend, adjust=False).mean()
        
        # EMA Slopes
        df["ema_fast_slope"] = df["ema_fast"].pct_change(self.params.slope_lookback) * 100
        df["ema_slow_slope"] = df["ema_slow"].pct_change(self.params.slope_lookback) * 100
        df["ema_trend_slope"] = df["ema_trend"].pct_change(self.params.slope_lookback) * 100
        
        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.params.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.params.rsi_period).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df["close"].ewm(span=self.params.macd_fast, adjust=False).mean()
        ema_26 = df["close"].ewm(span=self.params.macd_slow, adjust=False).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=self.params.macd_signal, adjust=False).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]
        
        # Bollinger Bands
        df["bb_middle"] = df["close"].rolling(window=self.params.bb_period).mean()
        bb_std = df["close"].rolling(window=self.params.bb_period).std()
        df["bb_upper"] = df["bb_middle"] + (bb_std * self.params.bb_std)
        df["bb_lower"] = df["bb_middle"] - (bb_std * self.params.bb_std)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        df["bb_squeeze"] = df["bb_width"] < df["bb_width"].rolling(50).quantile(self.params.bb_squeeze_threshold)
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
        
        # Volume Analysis
        df["volume_ma"] = df["volume"].rolling(window=self.params.volume_ma_period).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma"]
        df["volume_spike"] = df["volume_ratio"] > self.params.volume_spike_threshold
        
        # Stochastic Oscillator
        low_min = df["low"].rolling(window=self.params.stoch_k).min()
        high_max = df["high"].rolling(window=self.params.stoch_k).max()
        df["stoch_k"] = 100 * (df["close"] - low_min) / (high_max - low_min)
        df["stoch_d"] = df["stoch_k"].rolling(window=self.params.stoch_d).mean()
        
        # ATR and Volatility
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
        df["total_range"] = df["high"] - df["low"]
        
        df["upper_wick_ratio"] = df["upper_wick"] / (df["body"] + 1e-10)
        df["lower_wick_ratio"] = df["lower_wick"] / (df["body"] + 1e-10)
        df["body_ratio"] = df["body"] / df["total_range"]
        
        df["is_bullish"] = df["close"] > df["open"]
        df["is_bearish"] = df["close"] < df["open"]
        df["is_doji"] = df["body_ratio"] < 0.1
        
        # Price Action Patterns
        df["higher_high"] = df["high"] > df["high"].shift(1)
        df["higher_low"] = df["low"] > df["low"].shift(1)
        df["lower_high"] = df["high"] < df["high"].shift(1)
        df["lower_low"] = df["low"] < df["low"].shift(1)
        
        # Support/Resistance Levels
        df["resistance"] = df["high"].rolling(window=20).max()
        df["support"] = df["low"].rolling(window=20).min()
        df["near_resistance"] = (df["resistance"] - df["close"]) / df["close"] < 0.02
        df["near_support"] = (df["close"] - df["support"]) / df["close"] < 0.02
        
        return df
    
    def check_long_conditions(self, df: pd.DataFrame, idx: int) -> Optional[Dict]:
        if idx < max(self.params.ema_trend, self.params.atr_period, self.params.rsi_period) + 20:
            return None
        
        row = df.iloc[idx]
        prev = df.iloc[idx - 1]
        
        confirmations = 0
        confirmation_details = {}
        
        # 1. Basic EMA Setup (Required)
        if not (row["close"] > row["ema_fast"] and row["close"] > row["ema_slow"] and row["close"] > row["ema_trend"]):
            return None
        confirmations += 1
        confirmation_details["ema_setup"] = True
        
        # 2. EMA Alignment (Required)
        if not (row["ema_fast"] > row["ema_slow"] > row["ema_trend"]):
            return None
        confirmations += 1
        confirmation_details["ema_alignment"] = True
        
        # 3. EMA Slopes (Required)
        if not (row["ema_fast_slope"] > self.params.min_slope_pct and 
                row["ema_slow_slope"] > self.params.min_slope_pct and
                row["ema_trend_slope"] > self.params.min_slope_pct/2):
            return None
        confirmations += 1
        confirmation_details["ema_slopes"] = True
        
        # 4. Pullback Detection (Required)
        touched_9 = prev["low"] <= prev["ema_fast"] or row["low"] <= row["ema_fast"]
        touched_20 = prev["low"] <= prev["ema_slow"] or row["low"] <= row["ema_slow"]
        
        if not (touched_9 or touched_20):
            return None
        
        ema_touched = 9 if touched_9 else 20
        ema_level = row["ema_fast"] if ema_touched == 9 else row["ema_slow"]
        
        if row["close"] <= ema_level:
            return None
        confirmations += 1
        confirmation_details["pullback"] = True
        
        # 5. Wick Rejection (Required)
        wick_ratio = row["lower_wick_ratio"]
        wick_size_pct = row["lower_wick"] / row["close"]
        
        if not (wick_ratio >= self.params.wick_ratio_threshold and 
                wick_size_pct >= self.params.min_wick_pct):
            return None
        confirmations += 1
        confirmation_details["wick_rejection"] = True
        
        # 6. Bullish Candle (Required)
        if not row["is_bullish"]:
            return None
        confirmations += 1
        confirmation_details["bullish_candle"] = True
        
        # 7. RSI Confirmation - Ultra-Strict
        if self.params.use_rsi_filter:
            rsi_bullish = (row["rsi"] > self.params.rsi_pullback_low and 
                          row["rsi"] < self.params.rsi_overbought and
                          row["rsi"] > prev["rsi"] and
                          prev["rsi"] < self.params.rsi_pullback_low and
                          row["rsi"] < 50)  # Still in lower half
            if rsi_bullish:
                confirmations += 1
                confirmation_details["rsi"] = True
        
        # 8. MACD Confirmation - Ultra-Strict
        if self.params.use_macd_filter:
            macd_bullish = (row["macd"] > row["macd_signal"] and 
                           row["macd_histogram"] > prev["macd_histogram"] and
                           row["macd"] > 0 and
                           prev["macd"] <= prev["macd_signal"] and
                           row["macd_histogram"] > 0)  # Positive momentum
            if macd_bullish:
                confirmations += 1
                confirmation_details["macd"] = True
        
        # 9. Bollinger Bands Confirmation
        if self.params.use_bb_filter:
            bb_bullish = (row["bb_position"] > 0.2 and row["bb_position"] < 0.8 and
                         not row["bb_squeeze"])
            if bb_bullish:
                confirmations += 1
                confirmation_details["bollinger"] = True
        
        # 10. Volume Confirmation - Enhanced
        if self.params.use_volume_filter:
            volume_10ma = df.iloc[max(0, idx-10):idx]["volume"].mean()
            volume_bullish = (row["volume_ratio"] > self.params.min_volume_ratio and
                             row["volume_spike"] and
                             row["volume"] > volume_10ma * 1.5)  # Strong volume
            if volume_bullish:
                confirmations += 1
                confirmation_details["volume"] = True
        
        # 11. Stochastic Confirmation - Enhanced
        if self.params.use_stoch_filter:
            stoch_bullish = (row["stoch_k"] > self.params.stoch_oversold and 
                           row["stoch_k"] < self.params.stoch_overbought and
                           row["stoch_k"] > row["stoch_d"] and
                           prev["stoch_k"] < self.params.stoch_oversold)  # Was oversold
            if stoch_bullish:
                confirmations += 1
                confirmation_details["stochastic"] = True
        
        # 12. Price Action Confirmation - Enhanced
        if self.params.use_confluence:
            pa_bullish = (row["higher_high"] and row["higher_low"] and 
                         not row["near_resistance"] and
                         row["body_ratio"] > 0.6)  # Strong candle body
            if pa_bullish:
                confirmations += 1
                confirmation_details["price_action"] = True
        
        # 13. Volatility Filter
        if self.params.use_volatility_filter and row["atr_pct"] < self.params.min_atr_pct:
            return None
        
        # Minimum confirmations check
        if confirmations < self.params.min_confirmations:
            return None
        
        return {
            "ema_touched": ema_touched,
            "wick_ratio": wick_ratio,
            "confirmations": confirmations,
            "details": confirmation_details
        }
    
    def check_short_conditions(self, df: pd.DataFrame, idx: int) -> Optional[Dict]:
        if idx < max(self.params.ema_trend, self.params.atr_period, self.params.rsi_period) + 20:
            return None
        
        row = df.iloc[idx]
        prev = df.iloc[idx - 1]
        
        confirmations = 0
        confirmation_details = {}
        
        # 1. Basic EMA Setup (Required)
        if not (row["close"] < row["ema_fast"] and row["close"] < row["ema_slow"] and row["close"] < row["ema_trend"]):
            return None
        confirmations += 1
        confirmation_details["ema_setup"] = True
        
        # 2. EMA Alignment (Required)
        if not (row["ema_fast"] < row["ema_slow"] < row["ema_trend"]):
            return None
        confirmations += 1
        confirmation_details["ema_alignment"] = True
        
        # 3. EMA Slopes (Required)
        if not (row["ema_fast_slope"] < -self.params.min_slope_pct and 
                row["ema_slow_slope"] < -self.params.min_slope_pct and
                row["ema_trend_slope"] < -self.params.min_slope_pct/2):
            return None
        confirmations += 1
        confirmation_details["ema_slopes"] = True
        
        # 4. Pullback Detection (Required)
        touched_9 = prev["high"] >= prev["ema_fast"] or row["high"] >= row["ema_fast"]
        touched_20 = prev["high"] >= prev["ema_slow"] or row["high"] >= row["ema_slow"]
        
        if not (touched_9 or touched_20):
            return None
        
        ema_touched = 9 if touched_9 else 20
        ema_level = row["ema_fast"] if ema_touched == 9 else row["ema_slow"]
        
        if row["close"] >= ema_level:
            return None
        confirmations += 1
        confirmation_details["pullback"] = True
        
        # 5. Wick Rejection (Required)
        wick_ratio = row["upper_wick_ratio"]
        wick_size_pct = row["upper_wick"] / row["close"]
        
        if not (wick_ratio >= self.params.wick_ratio_threshold and 
                wick_size_pct >= self.params.min_wick_pct):
            return None
        confirmations += 1
        confirmation_details["wick_rejection"] = True
        
        # 6. Bearish Candle (Required)
        if not row["is_bearish"]:
            return None
        confirmations += 1
        confirmation_details["bearish_candle"] = True
        
        # 7. RSI Confirmation - Enhanced
        if self.params.use_rsi_filter:
            rsi_bearish = (row["rsi"] < self.params.rsi_pullback_high and 
                          row["rsi"] > self.params.rsi_oversold and
                          row["rsi"] < prev["rsi"] and
                          prev["rsi"] > self.params.rsi_pullback_high)  # RSI was overbought
            if rsi_bearish:
                confirmations += 1
                confirmation_details["rsi"] = True
        
        # 8. MACD Confirmation - Enhanced
        if self.params.use_macd_filter:
            macd_bearish = (row["macd"] < row["macd_signal"] and 
                           row["macd_histogram"] < prev["macd_histogram"] and
                           row["macd"] < 0 and
                           prev["macd"] >= prev["macd_signal"])  # MACD cross down
            if macd_bearish:
                confirmations += 1
                confirmation_details["macd"] = True
        
        # 9. Bollinger Bands Confirmation
        if self.params.use_bb_filter:
            bb_bearish = (row["bb_position"] > 0.2 and row["bb_position"] < 0.8 and
                         not row["bb_squeeze"])
            if bb_bearish:
                confirmations += 1
                confirmation_details["bollinger"] = True
        
        # 10. Volume Confirmation
        if self.params.use_volume_filter:
            volume_bearish = (row["volume_ratio"] > self.params.min_volume_ratio and
                             row["volume_spike"])
            if volume_bearish:
                confirmations += 1
                confirmation_details["volume"] = True
        
        # 11. Stochastic Confirmation
        if self.params.use_stoch_filter:
            stoch_bearish = (row["stoch_k"] < self.params.stoch_overbought and 
                           row["stoch_k"] > self.params.stoch_oversold and
                           row["stoch_k"] < row["stoch_d"])
            if stoch_bearish:
                confirmations += 1
                confirmation_details["stochastic"] = True
        
        # 12. Price Action Confirmation
        if self.params.use_confluence:
            pa_bearish = (row["lower_high"] and row["lower_low"] and 
                         not row["near_support"])
            if pa_bearish:
                confirmations += 1
                confirmation_details["price_action"] = True
        
        # 13. Volatility Filter
        if self.params.use_volatility_filter and row["atr_pct"] < self.params.min_atr_pct:
            return None
        
        # Minimum confirmations check
        if confirmations < self.params.min_confirmations:
            return None
        
        return {
            "ema_touched": ema_touched,
            "wick_ratio": wick_ratio,
            "confirmations": confirmations,
            "details": confirmation_details
        }
    
    def find_swing_low(self, df: pd.DataFrame, idx: int, lookback: int = 10) -> float:
        start = max(0, idx - lookback)
        return df.iloc[start:idx]["low"].min()
    
    def find_swing_high(self, df: pd.DataFrame, idx: int, lookback: int = 10) -> float:
        start = max(0, idx - lookback)
        return df.iloc[start:idx]["high"].max()
    
    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        df = self.calculate_indicators(df)
        self.signals = []
        
        for i in range(len(df)):
            row = df.iloc[i]
            timestamp = df.index[i]
            
            long_result = self.check_long_conditions(df, i)
            if long_result:
                ema_touched = long_result["ema_touched"]
                wick_ratio = long_result["wick_ratio"]
                confirmations = long_result["confirmations"]
                
                entry = row["close"]
                
                # Enhanced stop loss with ATR
                atr_stop = row["low"] - (row["atr"] * self.params.atr_multiplier)
                stop_below_wick = row["low"] - (row["lower_wick"] * 0.2)
                swing_low = self.find_swing_low(df, i, lookback=15)
                support_level = row["support"] * 0.995
                
                stop = min(atr_stop, stop_below_wick, swing_low, support_level)
                
                risk = entry - stop
                target = entry + (risk * self.params.risk_reward)
                
                signal = Signal(
                    direction=TradeDirection.LONG,
                    timestamp=timestamp,
                    entry_price=entry,
                    stop_loss=stop,
                    take_profit=target,
                    ema_touched=ema_touched,
                    wick_ratio=wick_ratio
                )
                self.signals.append(signal)
            
            short_result = self.check_short_conditions(df, i)
            if short_result:
                ema_touched = short_result["ema_touched"]
                wick_ratio = short_result["wick_ratio"]
                confirmations = short_result["confirmations"]
                
                entry = row["close"]
                
                # Enhanced stop loss with ATR
                atr_stop = row["high"] + (row["atr"] * self.params.atr_multiplier)
                stop_above_wick = row["high"] + (row["upper_wick"] * 0.2)
                swing_high = self.find_swing_high(df, i, lookback=15)
                resistance_level = row["resistance"] * 1.005
                
                stop = max(atr_stop, stop_above_wick, swing_high, resistance_level)
                
                risk = stop - entry
                target = entry - (risk * self.params.risk_reward)
                
                signal = Signal(
                    direction=TradeDirection.SHORT,
                    timestamp=timestamp,
                    entry_price=entry,
                    stop_loss=stop,
                    take_profit=target,
                    ema_touched=ema_touched,
                    wick_ratio=wick_ratio
                )
                self.signals.append(signal)
        
        self.signals.sort(key=lambda x: x.timestamp)
        return self.signals