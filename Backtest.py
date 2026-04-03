import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
from enum import Enum

from strategy import EMAPullbackStrategy, Signal, TradeDirection, StrategyParameters


class TradeStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"


@dataclass
class Trade:
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
        if self.direction == TradeDirection.LONG:
            return (exit_price - self.entry_price) * size
        else:
            return (self.entry_price - exit_price) * size


@dataclass
class BacktestResult:
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    
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
    
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    
    total_r: float = 0.0
    avg_r: float = 0.0


class BacktestEngine:
    def __init__(
        self,
        initial_capital: float = 10000.0,
        risk_per_trade: float = 0.015,  # 1.5% risk - balanced
        commission: float = 0.0007,  # Realistic commission
        slippage: float = 0.0003  # Realistic slippage
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
        self.capital = self.initial_capital
        self.equity_history = []
        self.open_trades = []
        self.closed_trades = []
        self.trade_counter = 0
    
    def calculate_position_size(self, entry: float, stop: float) -> float:
        risk_amount = self.capital * self.risk_per_trade
        risk_per_unit = abs(entry - stop)
        
        if risk_per_unit == 0:
            return 0
        
        total_commission_factor = 1 + (2 * self.commission)
        position_size = (risk_amount / risk_per_unit) * total_commission_factor
        
        return position_size
    
    def apply_slippage(self, price: float, direction: TradeDirection, is_entry: bool = True) -> float:
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
    
    def execute_exit(
        self,
        trade: Trade,
        exit_price: float,
        exit_time: datetime,
        reason: str
    ) -> float:
        filled_price = self.apply_slippage(exit_price, trade.direction, is_entry=False)
        pnl = trade.calculate_pnl(filled_price, trade.position_size)
        
        commission_cost = filled_price * trade.position_size * self.commission
        pnl -= commission_cost
        
        trade.exit_price = filled_price
        trade.exit_time = exit_time
        trade.exit_reason = reason
        trade.pnl = pnl
        trade.pnl_pct = (pnl / self.initial_capital) * 100
        trade.status = TradeStatus.CLOSED
        
        risk = abs(trade.entry_price - trade.stop_loss) * trade.position_size
        trade.rr_achieved = pnl / risk if risk != 0 else 0
        
        self.closed_trades.append(trade)
        self.capital += pnl
        
        return pnl
    
    def run_backtest(
        self,
        df: pd.DataFrame,
        strategy: EMAPullbackStrategy,
        params: StrategyParameters
    ) -> BacktestResult:
        self.reset()
        
        signals = strategy.generate_signals(df)
        signal_idx = 0
        df = strategy.calculate_indicators(df)
        
        for i in range(len(df)):
            timestamp = df.index[i]
            row = df.iloc[i]
            
            self.equity_history.append({
                "timestamp": timestamp,
                "equity": self.capital
            })
            
            # Check exits for open trades
            trades_to_close = []
            for trade in self.open_trades[:]:
                exit_reason = None
                
                if trade.direction == TradeDirection.LONG:
                    if row["low"] <= trade.stop_loss:
                        exit_reason = "stop_loss"
                    elif row["high"] >= trade.take_profit:
                        exit_reason = "take_profit"
                else:
                    if row["high"] >= trade.stop_loss:
                        exit_reason = "stop_loss"
                    elif row["low"] <= trade.take_profit:
                        exit_reason = "take_profit"
                
                if exit_reason:
                    exit_price = trade.stop_loss if exit_reason == "stop_loss" else trade.take_profit
                    self.execute_exit(trade, exit_price, timestamp, exit_reason)
                    trades_to_close.append(trade)
            
            for trade in trades_to_close:
                if trade in self.open_trades:
                    self.open_trades.remove(trade)
            
            # Check for new signals
            while (signal_idx < len(signals) and 
                   signals[signal_idx].timestamp == timestamp):
                signal = signals[signal_idx]
                
                # Dynamic position sizing based on recent performance
                max_concurrent = 8 if self.capital > self.initial_capital * 1.1 else 5
                
                if len(self.open_trades) < max_concurrent:
                    self.trade_counter += 1
                    
                    # Adaptive position sizing
                    recent_trades = self.closed_trades[-10:] if len(self.closed_trades) >= 10 else self.closed_trades
                    if recent_trades:
                        recent_win_rate = sum(1 for t in recent_trades if t.pnl > 0) / len(recent_trades)
                        risk_multiplier = 1.2 if recent_win_rate > 0.6 else 0.8
                    else:
                        risk_multiplier = 1.0
                    
                    original_risk = self.risk_per_trade
                    self.risk_per_trade = min(original_risk * risk_multiplier, 0.015)  # Cap at 1.5%
                    
                    position_size = self.calculate_position_size(
                        signal.entry_price, 
                        signal.stop_loss
                    )
                    
                    # Restore original risk
                    self.risk_per_trade = original_risk
                    
                    entry_price = self.apply_slippage(
                        signal.entry_price, 
                        signal.direction
                    )
                    
                    commission_cost = entry_price * position_size * self.commission
                    self.capital -= commission_cost
                    
                    trade = Trade(
                        trade_id=self.trade_counter,
                        direction=signal.direction,
                        entry_time=timestamp,
                        entry_price=entry_price,
                        stop_loss=signal.stop_loss,
                        take_profit=signal.take_profit,
                        position_size=position_size
                    )
                    
                    self.open_trades.append(trade)
                
                signal_idx += 1
        
        # Close remaining trades
        if self.open_trades:
            last_row = df.iloc[-1]
            last_time = df.index[-1]
            for trade in self.open_trades[:]:
                self.execute_exit(trade, last_row["close"], last_time, "end_of_data")
        
        return self._calculate_results()
    
    def _calculate_results(self) -> BacktestResult:
        result = BacktestResult()
        result.trades = self.closed_trades
        
        if not self.closed_trades:
            return result
        
        result.total_trades = len(self.closed_trades)
        
        for trade in self.closed_trades:
            if trade.pnl > 0:
                result.winning_trades += 1
                result.total_profit += trade.pnl
            else:
                result.losing_trades += 1
                result.total_loss += abs(trade.pnl)
            
            result.total_r += trade.rr_achieved
        
        result.net_profit = result.total_profit - result.total_loss
        result.win_rate = result.winning_trades / result.total_trades if result.total_trades > 0 else 0
        result.profit_factor = result.total_profit / result.total_loss if result.total_loss > 0 else float('inf')
        result.avg_trade = result.net_profit / result.total_trades if result.total_trades > 0 else 0
        result.avg_win = result.total_profit / result.winning_trades if result.winning_trades > 0 else 0
        result.avg_loss = result.total_loss / result.losing_trades if result.losing_trades > 0 else 0
        result.avg_r = result.total_r / result.total_trades if result.total_trades > 0 else 0
        
        result.equity_curve = pd.DataFrame(self.equity_history)
        if not result.equity_curve.empty:
            result.equity_curve.set_index("timestamp", inplace=True)
            
            cummax = result.equity_curve["equity"].cummax()
            drawdown = result.equity_curve["equity"] - cummax
            result.max_drawdown = drawdown.min()
            result.max_drawdown_pct = (result.max_drawdown / cummax.max()) * 100 if cummax.max() > 0 else 0
        
        return result