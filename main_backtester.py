"""
Main Backtesting Script
9 & 20 EMA Pullback with Wick Rejection Strategy

Usage:
    python main_backtester.py

Features:
- Multi-coin testing
- Parameter optimization
- Comprehensive reporting
- Professional visualization
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import argparse
import warnings
warnings.filterwarnings('ignore')

from ema_pullback_strategy import EMAPullbackStrategy, StrategyParameters
from backtest_engine import BacktestEngine
from visualization import Visualizer
from ema_pullback_strategy import DataFetcher


def run_single_backtest(
    symbol: str = "BTCUSDT",
    use_sample_data: bool = True,
    params: Optional[StrategyParameters] = None
):
    """Run single backtest for one symbol"""
    
    print(f"\n🚀 Running backtest for {symbol}")
    print("="*60)
    
    # Initialize data fetcher
    fetcher = DataFetcher()
    
    # Get data
    if use_sample_data:
        print("📊 Using sample data...")
        df = fetcher.generate_sample_data(symbol, periods=5000)
    else:
        print("📊 Fetching real data from Binance...")
        df = fetcher.fetch_ohlcv(symbol, "1h", "2023-01-01", "2024-01-01")
    
    print(f"📈 Loaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    
    # Use default or custom parameters
    if params is None:
        params = StrategyParameters()
    
    # Initialize strategy and engine
    strategy = EMAPullbackStrategy(params)
    engine = BacktestEngine(
        initial_capital=10000.0,
        risk_per_trade=0.01,
        commission=0.001,
        slippage=0.0005
    )
    
    # Run backtest
    print("⚡ Running backtest...")
    result = engine.run_backtest(df, strategy, params)
    
    # Calculate indicators for visualization
    df = strategy.calculate_indicators(df)
    
    # Print results
    visualizer = Visualizer()
    visualizer.print_results(result, symbol, params)
    
    # Create visualizations
    visualizer.plot_results(df, result, symbol, params, save_plots=True)
    
    # Export trade log
    visualizer.export_trade_log(result, symbol)
    
    return result


def run_multi_coin_backtest(symbols: List[str], use_sample_data: bool = True):
    """Run backtest across multiple coins"""
    
    print(f"\n🚀 Running multi-coin backtest")
    print(f"Symbols: {', '.join(symbols)}")
    print("="*60)
    
    results = {}
    
    for symbol in symbols:
        try:
            print(f"\n📊 Testing {symbol}...")
            result = run_single_backtest(symbol, use_sample_data)
            results[symbol] = result
        except Exception as e:
            print(f"❌ Error testing {symbol}: {e}")
            continue
    
    # Compare results
    print("\n" + "="*80)
    print("📊 MULTI-COIN COMPARISON")
    print("="*80)
    
    comparison_data = []
    for symbol, result in results.items():
        comparison_data.append({
            'Symbol': symbol,
            'Total Trades': result.total_trades,
            'Win Rate %': result.win_rate * 100,
            'Net Profit $': result.net_profit,
            'Profit Factor': result.profit_factor,
            'Max DD %': result.max_drawdown_pct,
            'Sharpe Ratio': result.sharpe_ratio,
            'Avg R': result.avg_r
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    print(df_comparison.to_string(index=False, float_format='%.2f'))
    
    return results


def run_parameter_optimization(symbol: str = "BTCUSDT", use_sample_data: bool = True):
    """Run parameter optimization"""
    
    print(f"\n🔬 Running parameter optimization for {symbol}")
    print("="*60)
    
    # Define parameter ranges
    ema_fast_range = [8, 9, 10, 11, 12]
    ema_slow_range = [18, 20, 21, 22, 25]
    rr_range = [1.5, 2.0, 2.5, 3.0]
    
    results = []
    
    for ema_fast in ema_fast_range:
        for ema_slow in ema_slow_range:
            if ema_fast >= ema_slow:
                continue
                
            for rr in rr_range:
                print(f"📊 Testing EMA {ema_fast}/{ema_slow}, RR {rr}:1")
                
                # Create parameters
                params = StrategyParameters(
                    ema_fast=ema_fast,
                    ema_slow=ema_slow,
                    risk_reward=rr
                )
                
                try:
                    # Run backtest
                    result = run_single_backtest(symbol, use_sample_data, params)
                    
                    results.append({
                        'EMA_Fast': ema_fast,
                        'EMA_Slow': ema_slow,
                        'Risk_Reward': rr,
                        'Total_Trades': result.total_trades,
                        'Win_Rate': result.win_rate * 100,
                        'Net_Profit': result.net_profit,
                        'Profit_Factor': result.profit_factor,
                        'Max_Drawdown': result.max_drawdown_pct,
                        'Sharpe_Ratio': result.sharpe_ratio
                    })
                    
                except Exception as e:
                    print(f"❌ Error: {e}")
                    continue
    
    # Create results DataFrame
    df_results = pd.DataFrame(results)
    
    if not df_results.empty:
        # Sort by net profit
        df_results = df_results.sort_values('Net_Profit', ascending=False)
        
        print("\n" + "="*80)
        print("🏆 OPTIMIZATION RESULTS (Top 10)")
        print("="*80)
        print(df_results.head(10).to_string(index=False, float_format='%.2f'))
        
        # Save results
        df_results.to_csv(f"{symbol}_optimization_results.csv", index=False)
        print(f"\n💾 Results saved to {symbol}_optimization_results.csv")
    
    return df_results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='9 & 20 EMA Pullback Strategy Backtester')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading symbol')
    parser.add_argument('--real-data', action='store_true', help='Use real data instead of sample')
    parser.add_argument('--multi-coin', action='store_true', help='Test multiple coins')
    parser.add_argument('--optimize', action='store_true', help='Run parameter optimization')
    parser.add_argument('--symbols', nargs='+', default=['BTCUSDT', 'ETHUSDT', 'SOLUSDT'], 
                       help='Symbols for multi-coin testing')
    
    args = parser.parse_args()
    
    print("🎯 9 & 20 EMA Pullback with Wick Rejection Strategy")
    print("="*80)
    
    if args.optimize:
        run_parameter_optimization(args.symbol, not args.real_data)
    elif args.multi_coin:
        run_multi_coin_backtest(args.symbols, not args.real_data)
    else:
        run_single_backtest(args.symbol, not args.real_data)
    
    print("\n✅ Backtesting completed!")


if __name__ == "__main__":
    main()
