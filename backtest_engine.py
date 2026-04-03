"""
Professional Backtesting Engine
Handles trade execution, management, and performance analysis
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
from enum import Enum

from ema_pullback_strategy import (
    EMAPullbackStrategy, Signal, Trade, TradeDirection, 
    TradeStatus, StrategyParameters, BacktestResult
)


class BacktestEngine:
    """Professional backtesting engine with advanced trade management"""
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        risk_per_trade: float = 0.01,
        commission: float = 0.001,
        slippage: float = 0.0005
    ):
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.commission = commission
        self.slippage = slippage
        
        self.capital = initial_capital
        self.equity_history: List[Dict] = []
        self.open_trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.trade_counter = 0
    
    def reset(self):
        """Reset engine state"""
        self.capital = self.initial_capital
        self.equity_history = []
        self.open_trades = []
        self.closed_trades = []
        self.trade_counter = 0
    
    def calculate_position_size(self, entry: float, stop: float) -> float:
        """Calculate position size based on risk"""
        risk_amount = self.capital * self.risk_per_trade
        risk_per_unit = abs(entry - stop)
        
        if risk_per_unit == 0:
            return 0
        
        # Account for commission in position sizing
        total_commission_factor = 1 + (2 * self.commission)
        position_size = (risk_amount / risk_per_unit) * total_commission_factor
        
        return position_size
    
    def apply_slippage(self, price: float, direction: TradeDirection, is_entry: bool = True) -> float:
        """Apply realistic slippage to prices"""
        slippage_factor = 1 + self.slippage
        
        if is_entry:
            if direction == TradeDirection.LONG:
                return price * slippage_factor
            else:
                return price / slippage_factor
        else:
            if direction == TradeDirection.LONG:
                return price / slippage_factor
            else:
                return price * slippage_factor
    
    def update_trailing_stops(self, current_price: float, df: pd.DataFrame, current_idx: int):
        """Update trailing stops for open trades"""
        for trade in self.open_trades:
            if trade.direction == TradeDirection.LONG:
                # Trail using previous swing lows or EMA 20
                if current_idx > 10:
                    recent_low = df.iloc[current_idx-10:current_idx]["low"].min()
                    ema_20 = df.iloc[current_idx]["ema_slow"]
                    
                    new_stop = max(recent_low * 0.995, ema_20)
                    if new_stop > trade.stop_loss:
                        trade.stop_loss = new_stop
                        trade.exit_reason = "trailing_stop"
            
            else:  # SHORT
                # Trail using previous swing highs or EMA 20
                if current_idx > 10:
                    recent_high = df.iloc[current_idx-10:current_idx]["high"].max()
                    ema_20 = df.iloc[current_idx]["ema_slow"]
                    
                    new_stop = min(recent_high * 1.005, ema_20)
                    if new_stop < trade.stop_loss:
                        trade.stop_loss = new_stop
                        trade.exit_reason = "trailing_stop"
    
    def execute_exit(
        self,
        trade: Trade,
        exit_price: float,
        exit_time: datetime,
        reason: str
    ) -> float:
        """Execute trade exit and calculate P&L"""
        filled_price = self.apply_slippage(exit_price, trade.direction, is_entry=False)
        pnl = trade.calculate_pnl(filled_price, trade.position_size)
        
        # Subtract commission
        commission_cost = filled_price * trade.position_size * self.commission
        pnl -= commission_cost
        
        # Update trade
        trade.exit_price = filled_price
        trade.exit_time = exit_time
        trade.exit_reason = reason
        trade.pnl = pnl
        trade.pnl_pct = (pnl / self.initial_capital) * 100
        trade.status = TradeStatus.CLOSED
        
        # Calculate R-multiple
        risk = abs(trade.entry_price - trade.stop_loss) * trade.position_size
        trade.rr_achieved = pnl / risk if risk != 0 else 0
        
        # Move to closed trades
        self.closed_trades.append(trade)
        self.capital += pnl
        
        return pnl
    
    def check_breakeven_and_partial_close(self, trade: Trade, current_price: float):
        """Handle breakeven moves and partial closes"""
        if trade.direction == TradeDirection.LONG:
            current_rr = (current_price - trade.entry_price) / abs(trade.entry_price - trade.stop_loss)
        else:
            current_rr = (trade.entry_price - current_price) / abs(trade.entry_price - trade.stop_loss)
        
        # Move to breakeven at 1:1
        if (self.params.use_breakeven and 
            current_rr >= self.params.breakeven_trigger_rr and 
            trade.stop_loss != trade.entry_price):
            
            trade.stop_loss = trade.entry_price
            trade.exit_reason = "breakeven"
        
        # Partial close at 2:1
        if current_rr >= self.params.partial_close_rr and trade.position_size == trade.original_size:
            partial_size = trade.position_size * self.params.partial_close_pct
            pnl = trade.calculate_pnl(current_price, partial_size)
            
            commission_cost = current_price * partial_size * self.commission
            pnl -= commission_cost
            
            self.capital += pnl
            trade.position_size *= (1 - self.params.partial_close_pct)
            trade.exit_reason = "partial_close"
    
    def run_backtest(
        self,
        df: pd.DataFrame,
        strategy: EMAPullbackStrategy,
        params: StrategyParameters
    ) -> BacktestResult:
        """Run complete backtest"""
        self.reset()
        self.params = params
        
        # Generate signals
        signals = strategy.generate_signals(df)
        signal_idx = 0
        
        # Calculate indicators for the whole dataset
        df = strategy.calculate_indicators(df)
        
        for i in range(len(df)):
            timestamp = df.index[i]
            row = df.iloc[i]
            
            # Update equity
            self.equity_history.append({
                "timestamp": timestamp,
                "equity": self.capital
            })
            
            # Update trailing stops
            if params.use_trailing_stop:
                self.update_trailing_stops(row["close"], df, i)
            
            # Check exits for open trades
            trades_to_close = []
            for trade in self.open_trades[:]:
                exit_reason = None
                
                if trade.direction == TradeDirection.LONG:
                    if row["low"] <= trade.stop_loss:
                        exit_reason = "stop_loss"
                    elif row["high"] >= trade.take_profit:
                        exit_reason = "take_profit"
                else:  # SHORT
                    if row["high"] >= trade.stop_loss:
                        exit_reason = "stop_loss"
                    elif row["low"] <= trade.take_profit:
                        exit_reason = "take_profit"
                
                if exit_reason:
                    exit_price = trade.stop_loss if exit_reason == "stop_loss" else trade.take_profit
                    self.execute_exit(trade, exit_price, timestamp, exit_reason)
                    trades_to_close.append(trade)
                else:
                    # Check breakeven and partial close
                    self.check_breakeven_and_partial_close(trade, row["close"])
            
            # Remove closed trades
            for trade in trades_to_close:
                if trade in self.open_trades:
                    self.open_trades.remove(trade)
            
            # Check for new signals
            while (signal_idx < len(signals) and 
                   signals[signal_idx].timestamp == timestamp):
                
                signal = signals[signal_idx]
                
                # Limit concurrent trades
                if len(self.open_trades) < 5:
                    self.trade_counter += 1
                    
                    # Calculate position size
                    position_size = self.calculate_position_size(
                        signal.entry_price, 
                        signal.stop_loss
                    )
                    
                    # Apply slippage
                    entry_price = self.apply_slippage(
                        signal.entry_price, 
                        signal.direction
                    )
                    
                    # Pay commission
                    commission_cost = entry_price * position_size * self.commission
                    self.capital -= commission_cost
                    
                    # Create trade
                    trade = Trade(
                        trade_id=self.trade_counter,
                        direction=signal.direction,
                        entry_time=timestamp,
                        entry_price=entry_price,
                        stop_loss=signal.stop_loss,
                        take_profit=signal.take_profit,
                        position_size=position_size
                    )
                    # Store original size for partial closes
                    trade.original_size = position_size
                    
                    self.open_trades.append(trade)
                
                signal_idx += 1
        
        # Close remaining trades at end of data
        if self.open_trades:
            last_row = df.iloc[-1]
            last_time = df.index[-1]
            for trade in self.open_trades[:]:
                self.execute_exit(trade, last_row["close"], last_time, "end_of_data")
        
        return self._calculate_results()
    
    def _calculate_results(self) -> BacktestResult:
        """Calculate performance metrics"""
        result = BacktestResult()
        result.trades = self.closed_trades
        
        if not self.closed_trades:
            return result
        
        # Basic metrics
        result.total_trades = len(self.closed_trades)
        
        for trade in self.closed_trades:
            if trade.pnl > 0:
                result.winning_trades += 1
                result.total_profit += trade.pnl
            else:
                result.losing_trades += 1
                result.total_loss += abs(trade.pnl)
            
            result.total_r += trade.rr_achieved
        
        # Performance calculations
        result.net_profit = result.total_profit - result.total_loss
        result.win_rate = result.winning_trades / result.total_trades if result.total_trades > 0 else 0
        result.profit_factor = result.total_profit / result.total_loss if result.total_loss > 0 else float('inf')
        result.avg_trade = result.net_profit / result.total_trades if result.total_trades > 0 else 0
        result.avg_win = result.total_profit / result.winning_trades if result.winning_trades > 0 else 0
        result.avg_loss = result.total_loss / result.losing_trades if result.losing_trades > 0 else 0
        result.avg_r = result.total_r / result.total_trades if result.total_trades > 0 else 0
        
        # Equity curve and drawdown
        result.equity_curve = pd.DataFrame(self.equity_history)
        if not result.equity_curve.empty:
            result.equity_curve.set_index("timestamp", inplace=True)
            
            cummax = result.equity_curve["equity"].cummax()
            drawdown = result.equity_curve["equity"] - cummax
            result.max_drawdown = drawdown.min()
            result.max_drawdown_pct = (result.max_drawdown / cummax.max()) * 100 if cummax.max() > 0 else 0
            
            # Sharpe ratio (simplified)
            returns = result.equity_curve["equity"].pct_change().dropna()
            if len(returns) > 1 and returns.std() != 0:
                result.sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)  # Annualized
        
        return result
