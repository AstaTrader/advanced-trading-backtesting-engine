import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Optional, List, Dict

from data import BinanceDataFetcher, generate_sample_data
from strategy import EMAPullbackStrategy, StrategyParameters, TradeDirection
from Backtest import BacktestEngine, BacktestResult
from momentum_strategy import MomentumStrategy


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


def print_results(result: BacktestResult, symbol: str, params: StrategyParameters):
    print("\n" + "="*60)
    print(f" BACKTEST RESULTS: {symbol}")
    print(f" Strategy: EMA {params.ema_fast} & {params.ema_slow} Pullback")
    print("="*60)
    
    print(f"\n📊 TRADE STATISTICS:")
    print(f"  Total Trades:     {result.total_trades}")
    print(f"  Winning Trades:   {result.winning_trades}")
    print(f"  Losing Trades:    {result.losing_trades}")
    print(f"  Win Rate:         {result.win_rate*100:.1f}%")
    
    print(f"\n💰 P&L METRICS:")
    print(f"  Net Profit:       {format_currency(result.net_profit)}")
    print(f"  Total Profit:     {format_currency(result.total_profit)}")
    print(f"  Total Loss:       {format_currency(result.total_loss)}")
    print(f"  Profit Factor:    {result.profit_factor:.2f}")
    print(f"  Average Trade:    {format_currency(result.avg_trade)}")
    print(f"  Average Win:      {format_currency(result.avg_win)}")
    print(f"  Average Loss:     {format_currency(result.avg_loss)}")
    
    print(f"\n📉 RISK METRICS:")
    print(f"  Max Drawdown:     {format_currency(result.max_drawdown)}")
    print(f"  Max Drawdown %:   {result.max_drawdown_pct:.1f}%")
    print(f"  Total R-Multiple: {result.total_r:.2f}R")
    print(f"  Average R:        {result.avg_r:.2f}R")
    
    if result.trades:
        exits = {}
        for t in result.trades:
            exits[t.exit_reason] = exits.get(t.exit_reason, 0) + 1
        print(f"\n🚪 EXIT REASONS:")
        for reason, count in sorted(exits.items(), key=lambda x: -x[1]):
            pct = (count / result.total_trades) * 100
            print(f"  {reason}: {count} ({pct:.1f}%)")


def plot_results(
    df: pd.DataFrame, 
    result: BacktestResult, 
    symbol: str,
    params: StrategyParameters
):
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), 
                             gridspec_kw={'height_ratios': [3, 1, 1]})
    
    strategy = EMAPullbackStrategy(params)
    df_ind = strategy.calculate_indicators(df)
    
    # Plot 1: Price chart with trades
    ax1 = axes[0]
    ax1.plot(df_ind.index, df_ind["close"], color="gray", alpha=0.7, linewidth=0.8, label="Close")
    ax1.plot(df_ind.index, df_ind["ema_fast"], color="blue", linewidth=1.5, 
             label=f"EMA {params.ema_fast}")
    ax1.plot(df_ind.index, df_ind["ema_slow"], color="orange", linewidth=1.5, 
             label=f"EMA {params.ema_slow}")
    
    # Plot trades
    longs = [t for t in result.trades if t.direction == TradeDirection.LONG]
    shorts = [t for t in result.trades if t.direction == TradeDirection.SHORT]
    
    ax1.scatter([t.entry_time for t in longs], 
                [t.entry_price for t in longs], 
                marker="^", color="green", s=100, zorder=5, label="Long Entry")
    ax1.scatter([t.entry_time for t in shorts], 
                [t.entry_price for t in shorts], 
                marker="v", color="red", s=100, zorder=5, label="Short Entry")
    
    # Connect entry to exit
    for t in result.trades:
        color = "green" if t.pnl > 0 else "red"
        alpha = 0.3 if t.pnl > 0 else 0.2
        ax1.plot([t.entry_time, t.exit_time], 
                 [t.entry_price, t.exit_price], 
                 color=color, alpha=alpha, linewidth=1)
    
    ax1.set_title(f"{symbol} - EMA Pullback Strategy | "
                  f"Win Rate: {result.win_rate*100:.1f}% | "
                  f"Net Profit: {format_currency(result.net_profit)}",
                  fontsize=12, fontweight='bold')
    ax1.set_ylabel("Price")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Equity curve
    ax2 = axes[1]
    ax2.plot(result.equity_curve.index, result.equity_curve["equity"], 
             color="purple", linewidth=1.5, label="Equity")
    
    cummax = result.equity_curve["equity"].cummax()
    ax2.fill_between(result.equity_curve.index, result.equity_curve["equity"], cummax,
                     color="red", alpha=0.3, label="Drawdown")
    
    ax2.set_ylabel("Equity ($)")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: R-multiples
    ax3 = axes[2]
    if result.trades:
        r_values = [t.rr_achieved for t in result.trades]
        colors = ["green" if r > 0 else "red" for r in r_values]
        ax3.bar(range(len(r_values)), r_values, color=colors, alpha=0.7)
        ax3.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax3.axhline(y=2, color="blue", linestyle="--", linewidth=1, alpha=0.5, label="2R Target")
        ax3.set_xlabel("Trade Number")
        ax3.set_ylabel("R-Multiple")
        ax3.set_title("Trade R-Multiples")
        ax3.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.show()


def run_backtest(symbol: str = "BTCUSDT", use_sample_data: bool = True):
    # Get data
    if use_sample_data:
        print(f"Using sample data for {symbol}...")
        df = generate_sample_data(symbol, periods=5000)
    else:
        fetcher = BinanceDataFetcher()
        print(f"Fetching data for {symbol}...")
        df = fetcher.fetch_ohlcv(symbol, "1h", "2023-01-01", "2024-01-01")
    
    print(f"Loaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    
    # BALANCED HIGH-PRECISION STRATEGY
    params = StrategyParameters(
        # EMA Settings - Optimized
        ema_fast=9,
        ema_slow=20,
        ema_trend=50,
        slope_lookback=3,
        min_slope_pct=0.008,  # Moderate threshold
        
        # Wick Rejection - Balanced
        wick_ratio_threshold=1.6,  # Moderate requirement
        min_wick_pct=0.002,
        
        # RSI Settings - Effective
        rsi_period=14,
        rsi_oversold=25,
        rsi_overbought=75,
        rsi_pullback_low=35,
        rsi_pullback_high=65,
        
        # MACD Settings - Standard
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        
        # Bollinger Bands
        bb_period=20,
        bb_std=2.0,
        bb_squeeze_threshold=0.15,
        
        # Volume Settings - Effective
        volume_ma_period=20,
        volume_spike_threshold=1.4,
        min_volume_ratio=0.9,
        
        # Stochastic Settings
        stoch_k=14,
        stoch_d=3,
        stoch_oversold=20,
        stoch_overbought=80,
        
        # ATR Settings - Balanced
        atr_period=14,
        min_atr_pct=0.005,
        atr_multiplier=2.0,
        
        # Risk Management - Optimized
        risk_reward=2.5,  # Balanced RR
        use_breakeven=True,
        breakeven_trigger_rr=1.0,
        partial_close_rr=2.0,
        partial_close_pct=0.5,
        
        # Confirmation Filters - Selective but not too strict
        use_rsi_filter=True,
        use_macd_filter=True,
        use_bb_filter=False,  # Disabled for more opportunities
        use_volume_filter=True,
        use_stoch_filter=False,  # Disabled for simplicity
        use_trend_filter=True,
        
        # Advanced Filters - High Precision
        min_confirmations=4,  # Require 4 confirmations for quality
        use_confluence=True,
        use_volatility_filter=True  # Enable for better filtering
    )
    
    # Test both strategies
    print("\n" + "="*60)
    print(" TESTING MOMENTUM STRATEGY")
    print("="*60)
    
    momentum_strategy = MomentumStrategy()
    engine = BacktestEngine(initial_capital=10000.0)
    
    print("Running momentum backtest...")
    momentum_result = engine.run_backtest(df, momentum_strategy, params)
    
    print_results(momentum_result, symbol, params)
    
    print("\n" + "="*60)
    print(" TESTING EMA PULLBACK STRATEGY")
    print("="*60)
    
    # Initialize and run
    strategy = EMAPullbackStrategy(params)
    
    print("Running EMA pullback backtest...")
    result = engine.run_backtest(df, strategy, params)
    
    # Results
    print_results(result, symbol, params)
    
    # Plot
    if not result.equity_curve.empty:
        plot_results(df, result, symbol, params)
    
    return result


if __name__ == "__main__":
    # Run the backtest
    result = run_backtest("BTCUSDT", use_sample_data=True)
