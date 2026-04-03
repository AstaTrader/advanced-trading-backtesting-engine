"""
Visualization and Analysis Tools
Creates charts, plots, and performance reports
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from typing import List, Dict, Optional

from ema_pullback_strategy import Trade, BacktestResult, StrategyParameters, TradeDirection


class Visualizer:
    """Professional visualization tools for backtesting results"""
    
    def __init__(self):
        plt.style.use('default')
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['grid.alpha'] = 0.3
    
    def format_currency(self, value: float) -> str:
        """Format currency values"""
        return f"${value:,.2f}"
    
    def plot_results(
        self,
        df: pd.DataFrame,
        result: BacktestResult,
        symbol: str,
        params: StrategyParameters,
        save_plots: bool = False
    ):
        """Create comprehensive visualization dashboard"""
        fig = plt.figure(figsize=(20, 16))
        
        # Create subplots
        gs = fig.add_gridspec(4, 2, height_ratios=[3, 1.5, 1, 1], width_ratios=[3, 1])
        
        # Main price chart
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_price_chart(ax1, df, result, symbol, params)
        
        # Equity curve
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_equity_curve(ax2, result)
        
        # Drawdown chart
        ax3 = fig.add_subplot(gs[2, 0])
        self._plot_drawdown(ax3, result)
        
        # R-multiple distribution
        ax4 = fig.add_subplot(gs[3, 0])
        self._plot_r_multiples(ax4, result)
        
        # Performance metrics
        ax5 = fig.add_subplot(gs[1:, 1])
        self._plot_metrics(ax5, result, symbol, params)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'{symbol}_backtest_results.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _plot_price_chart(self, ax, df: pd.DataFrame, result: BacktestResult, symbol: str, params: StrategyParameters):
        """Plot price chart with trades"""
        # Plot price
        ax.plot(df.index, df["close"], color="gray", alpha=0.7, linewidth=1, label="Close Price")
        
        # Plot EMAs
        ax.plot(df.index, df["ema_fast"], color="blue", linewidth=1.5, 
                label=f"EMA {params.ema_fast}", alpha=0.8)
        ax.plot(df.index, df["ema_slow"], color="orange", linewidth=1.5, 
                label=f"EMA {params.ema_slow}", alpha=0.8)
        
        # Separate long and short trades
        long_trades = [t for t in result.trades if t.direction == TradeDirection.LONG]
        short_trades = [t for t in result.trades if t.direction == TradeDirection.SHORT]
        
        # Plot long trades
        if long_trades:
            long_entries = [(t.entry_time, t.entry_price) for t in long_trades]
            long_exits = [(t.exit_time, t.exit_price) for t in long_trades]
            
            entry_times, entry_prices = zip(*long_entries)
            exit_times, exit_prices = zip(*long_exits)
            
            ax.scatter(entry_times, entry_prices, marker="^", color="green", s=80, 
                      zorder=5, label=f"Long Entries ({len(long_trades)})")
            ax.scatter(exit_times, exit_prices, marker="x", color="red", s=60, 
                      zorder=5, label=f"Long Exits ({len(long_trades)})")
            
            # Connect entries to exits
            for trade in long_trades:
                color = "green" if trade.pnl > 0 else "red"
                alpha = 0.3 if trade.pnl > 0 else 0.2
                ax.plot([trade.entry_time, trade.exit_time], 
                       [trade.entry_price, trade.exit_price], 
                       color=color, alpha=alpha, linewidth=0.8)
        
        # Plot short trades
        if short_trades:
            short_entries = [(t.entry_time, t.entry_price) for t in short_trades]
            short_exits = [(t.exit_time, t.exit_price) for t in short_trades]
            
            entry_times, entry_prices = zip(*short_entries)
            exit_times, exit_prices = zip(*short_exits)
            
            ax.scatter(entry_times, entry_prices, marker="v", color="purple", s=80, 
                      zorder=5, label=f"Short Entries ({len(short_trades)})")
            ax.scatter(exit_times, exit_prices, marker="x", color="red", s=60, 
                      zorder=5, label=f"Short Exits ({len(short_trades)})")
            
            # Connect entries to exits
            for trade in short_trades:
                color = "green" if trade.pnl > 0 else "red"
                alpha = 0.3 if trade.pnl > 0 else 0.2
                ax.plot([trade.entry_time, trade.exit_time], 
                       [trade.entry_price, trade.exit_price], 
                       color=color, alpha=alpha, linewidth=0.8)
        
        ax.set_title(f"{symbol} - EMA {params.ema_fast}/{params.ema_slow} Pullback Strategy\n"
                    f"Win Rate: {result.win_rate*100:.1f}% | Net P&L: {self.format_currency(result.net_profit)}",
                    fontsize=14, fontweight='bold')
        ax.set_ylabel("Price ($)", fontsize=11)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _plot_equity_curve(self, ax, result: BacktestResult):
        """Plot equity curve"""
        if result.equity_curve.empty:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            return
        
        ax.plot(result.equity_curve.index, result.equity_curve["equity"], 
                color="green", linewidth=2, label="Equity")
        
        # Add initial capital line
        ax.axhline(y=result.equity_curve["equity"].iloc[0], color="blue", 
                  linestyle="--", alpha=0.5, label="Initial Capital")
        
        # Add peak line
        peak = result.equity_curve["equity"].max()
        ax.axhline(y=peak, color="red", linestyle="--", alpha=0.5, 
                  label=f"Peak: {self.format_currency(peak)}")
        
        ax.set_title("Equity Curve", fontsize=12, fontweight='bold')
        ax.set_ylabel("Equity ($)", fontsize=10)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _plot_drawdown(self, ax, result: BacktestResult):
        """Plot drawdown chart"""
        if result.equity_curve.empty:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            return
        
        equity = result.equity_curve["equity"]
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax * 100
        
        ax.fill_between(result.equity_curve.index, drawdown, 0, 
                       color="red", alpha=0.3, label="Drawdown")
        ax.plot(result.equity_curve.index, drawdown, color="red", linewidth=1)
        
        ax.set_title(f"Drawdown (Max: {result.max_drawdown_pct:.1f}%)", 
                    fontsize=12, fontweight='bold')
        ax.set_ylabel("Drawdown (%)", fontsize=10)
        ax.set_xlabel("Date", fontsize=10)
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _plot_r_multiples(self, ax, result: BacktestResult):
        """Plot R-multiple distribution"""
        if not result.trades:
            ax.text(0.5, 0.5, 'No Trades', ha='center', va='center', transform=ax.transAxes)
            return
        
        r_values = [t.rr_achieved for t in result.trades]
        colors = ["green" if r > 0 else "red" for r in r_values]
        
        bars = ax.bar(range(len(r_values)), r_values, color=colors, alpha=0.7)
        
        # Add reference lines
        ax.axhline(y=0, color="black", linestyle="-", linewidth=1)
        ax.axhline(y=1, color="blue", linestyle="--", linewidth=1, alpha=0.5, label="1R")
        ax.axhline(y=2, color="purple", linestyle="--", linewidth=1, alpha=0.5, label="2R")
        ax.axhline(y=-1, color="orange", linestyle="--", linewidth=1, alpha=0.5, label="-1R")
        
        ax.set_title(f"Trade R-Multiples (Avg: {result.avg_r:.2f}R)", 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel("Trade Number", fontsize=10)
        ax.set_ylabel("R-Multiple", fontsize=10)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
    
    def _plot_metrics(self, ax, result: BacktestResult, symbol: str, params: StrategyParameters):
        """Plot performance metrics dashboard"""
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.95, f"Performance Metrics", 
                ha='center', va='top', fontsize=14, fontweight='bold', transform=ax.transAxes)
        
        # Create metrics text
        metrics_text = f"""
        ╔════════════════════════════════════╗
        ║         TRADING STATISTICS          ║
        ╠════════════════════════════════════╣
        ║ Total Trades:     {result.total_trades:>15} ║
        ║ Winning Trades:   {result.winning_trades:>15} ║
        ║ Losing Trades:    {result.losing_trades:>15} ║
        ║ Win Rate:         {result.win_rate*100:>13.1f}% ║
        ╠════════════════════════════════════╣
        ║           P&L METRICS               ║
        ╠════════════════════════════════════╣
        ║ Net Profit:       {self.format_currency(result.net_profit):>15} ║
        ║ Total Profit:     {self.format_currency(result.total_profit):>15} ║
        ║ Total Loss:       {self.format_currency(result.total_loss):>15} ║
        ║ Profit Factor:    {result.profit_factor:>13.2f} ║
        ║ Average Trade:    {self.format_currency(result.avg_trade):>15} ║
        ║ Average Win:      {self.format_currency(result.avg_win):>15} ║
        ║ Average Loss:     {self.format_currency(result.avg_loss):>15} ║
        ╠════════════════════════════════════╣
        ║          RISK METRICS               ║
        ╠════════════════════════════════════╣
        ║ Max Drawdown:     {self.format_currency(result.max_drawdown):>15} ║
        ║ Max Drawdown %:   {result.max_drawdown_pct:>13.1f}% ║
        ║ Sharpe Ratio:     {result.sharpe_ratio:>13.2f} ║
        ║ Total R-Multiple: {result.total_r:>13.2f}R ║
        ║ Average R:        {result.avg_r:>13.2f}R ║
        ╚════════════════════════════════════╝

        Strategy Parameters:
        • EMA Fast: {params.ema_fast}
        • EMA Slow: {params.ema_slow}
        • Risk/Reward: {params.risk_reward}:1
        """
        
        ax.text(0.5, 0.85, metrics_text, ha='center', va='top', 
                fontfamily='monospace', fontsize=8, transform=ax.transAxes)
    
    def print_results(self, result: BacktestResult, symbol: str, params: StrategyParameters):
        """Print detailed results to console"""
        print("\n" + "="*80)
        print(f"🚀 BACKTEST RESULTS: {symbol}")
        print(f"📈 Strategy: EMA {params.ema_fast} & {params.ema_slow} Pullback with Wick Rejection")
        print("="*80)
        
        print(f"\n📊 TRADE STATISTICS:")
        print(f"  Total Trades:     {result.total_trades}")
        print(f"  Winning Trades:   {result.winning_trades}")
        print(f"  Losing Trades:    {result.losing_trades}")
        print(f"  Win Rate:         {result.win_rate*100:.1f}%")
        
        print(f"\n💰 P&L METRICS:")
        print(f"  Net Profit:       {self.format_currency(result.net_profit)}")
        print(f"  Total Profit:     {self.format_currency(result.total_profit)}")
        print(f"  Total Loss:       {self.format_currency(result.total_loss)}")
        print(f"  Profit Factor:    {result.profit_factor:.2f}")
        print(f"  Average Trade:    {self.format_currency(result.avg_trade)}")
        print(f"  Average Win:      {self.format_currency(result.avg_win)}")
        print(f"  Average Loss:     {self.format_currency(result.avg_loss)}")
        
        print(f"\n📉 RISK METRICS:")
        print(f"  Max Drawdown:     {self.format_currency(result.max_drawdown)}")
        print(f"  Max Drawdown %:   {result.max_drawdown_pct:.1f}%")
        print(f"  Sharpe Ratio:     {result.sharpe_ratio:.2f}")
        print(f"  Total R-Multiple: {result.total_r:.2f}R")
        print(f"  Average R:        {result.avg_r:.2f}R")
        
        if result.trades:
            # Exit reasons
            exits = {}
            for t in result.trades:
                exits[t.exit_reason] = exits.get(t.exit_reason, 0) + 1
            
            print(f"\n🚪 EXIT REASONS:")
            for reason, count in sorted(exits.items(), key=lambda x: -x[1]):
                pct = (count / result.total_trades) * 100
                print(f"  {reason}: {count} ({pct:.1f}%)")
    
    def export_trade_log(self, result: BacktestResult, symbol: str, filename: Optional[str] = None):
        """Export detailed trade log to CSV"""
        if not result.trades:
            print("No trades to export")
            return
        
        if filename is None:
            filename = f"{symbol}_trade_log.csv"
        
        trade_data = []
        for trade in result.trades:
            trade_data.append({
                'Trade_ID': trade.trade_id,
                'Direction': trade.direction.value,
                'Entry_Time': trade.entry_time,
                'Entry_Price': trade.entry_price,
                'Stop_Loss': trade.stop_loss,
                'Take_Profit': trade.take_profit,
                'Exit_Time': trade.exit_time,
                'Exit_Price': trade.exit_price,
                'Exit_Reason': trade.exit_reason,
                'Position_Size': trade.position_size,
                'P&L': trade.pnl,
                'P&L_%': trade.pnl_pct,
                'R_Multiple': trade.rr_achieved
            })
        
        df = pd.DataFrame(trade_data)
        df.to_csv(filename, index=False)
        print(f"Trade log exported to {filename}")
