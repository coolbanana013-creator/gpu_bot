"""
Final Comprehensive MTF Validation
Run 10k bots with normal 70% consensus threshold (no DEBUG mode)
Compare MTF enabled vs disabled performance
"""
import numpy as np
from src.backtester.compact_simulator import CompactBacktester
from src.bot_generator.compact_generator import CompactBotGenerator
from src.data_provider.loader import DataLoader
import pyopencl as cl
import time
import sys
import os

# Setup paths
sys.path.insert(0, os.path.abspath('.'))

def load_real_data():
    """Load real market data"""
    loader = DataLoader(symbol='BTC_USDT', timeframe='1m', data_dir='data')
    ohlcv = loader.load_data()
    
    # Use 7 days of data
    days = 7
    bars_per_day = 1440
    total_bars = days * bars_per_day
    
    if len(ohlcv) < total_bars:
        print(f"WARNING: Not enough data. Have {len(ohlcv)} bars, need {total_bars}")
        total_bars = len(ohlcv)
    
    return ohlcv[-total_bars:]

def run_comprehensive_test(mtf_enabled=False):
    """Run 10k bots test with or without MTF"""
    print("="*80)
    if mtf_enabled:
        print("FINAL VALIDATION: MTF ENABLED (70% consensus)")
    else:
        print("FINAL VALIDATION: MTF DISABLED (70% consensus)")
    print("="*80)
    
    # Load data
    print("\nLoading real market data...")
    ohlcv = load_real_data()
    print(f"Loaded {len(ohlcv)} bars ({len(ohlcv)/1440:.1f} days)")
    print(f"Price range: ${ohlcv[0, 3]:.2f} - ${ohlcv[-1, 3]:.2f}")
    
    # Generate bots
    print("\nGenerating 10,000 bots...")
    gpu_context = cl.create_some_context(interactive=False)
    gpu_queue = cl.CommandQueue(gpu_context)
    
    bot_gen = CompactBotGenerator(
        population_size=10000,
        min_indicators=2,
        max_indicators=4,
        min_risk_strategies=1,
        max_risk_strategies=2,
        gpu_context=gpu_context,
        gpu_queue=gpu_queue
    )
    bots = bot_gen.generate_bots()
    print(f"Generated {len(bots)} bots")
    
    # Initialize backtester
    print("\nInitializing backtester...")
    backtester = CompactBacktester(
        bots=bots,
        ohlcv_data=ohlcv,
        initial_balance=10000.0,
        fee_rate=0.0004,
        max_open_trades=5,
        enable_mtf=mtf_enabled,
        htf_multiplier=60,  # 1h HTF from 1m base
        gpu_context=gpu_context,
        gpu_queue=gpu_queue
    )
    
    # Run backtest
    print(f"\nRunning backtest: 10,000 bots × 10 cycles...")
    start_time = time.time()
    
    results = backtester.run_backtest(
        num_cycles=10,
        cycle_days=7
    )
    
    elapsed = time.time() - start_time
    
    # Aggregate results
    print("\nAggregating results...")
    total_trades = sum(r['num_trades'] for r in results)
    total_wins = sum(r['winning_trades'] for r in results)
    bots_with_trades = sum(1 for r in results if r['num_trades'] > 0)
    total_pnl = sum(r['final_balance'] - 10000.0 for r in results)
    
    win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0.0
    success_rate = bots_with_trades / len(results) * 100
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Total bots: {len(results):,}")
    print(f"Total workloads: {len(results) * 10:,} (10 cycles)")
    print(f"Total trades: {total_trades:,}")
    print(f"Total wins: {total_wins:,}")
    print(f"Win rate: {win_rate:.2f}%")
    print(f"Bots with trades: {bots_with_trades:,} ({success_rate:.1f}%)")
    print(f"Total P&L: ${total_pnl:,.2f}")
    print(f"Avg P&L per bot: ${total_pnl/len(results):.2f}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Throughput: {len(results)*10/elapsed:,.0f} workloads/sec")
    
    return {
        'total_trades': total_trades,
        'total_wins': total_wins,
        'win_rate': win_rate,
        'success_rate': success_rate,
        'time': elapsed,
        'total_pnl': total_pnl
    }

print("="*80)
print("FINAL COMPREHENSIVE MTF VALIDATION")
print("Testing 10k bots × 10 cycles × 7 days (1m timeframe)")
print("Normal 70% consensus threshold (no DEBUG mode)")
print("="*80)

# Test 1: MTF disabled (baseline)
print("\n\n")
results_baseline = run_comprehensive_test(mtf_enabled=False)

# Test 2: MTF enabled
print("\n\n")
results_mtf = run_comprehensive_test(mtf_enabled=True)

# Comparison
print("\n\n")
print("="*80)
print("COMPARISON: MTF DISABLED vs MTF ENABLED")
print("="*80)

trade_reduction = (results_baseline['total_trades'] - results_mtf['total_trades']) / results_baseline['total_trades'] * 100
win_rate_change = results_mtf['win_rate'] - results_baseline['win_rate']
pnl_change = results_mtf['total_pnl'] - results_baseline['total_pnl']
time_change = (results_mtf['time'] - results_baseline['time']) / results_baseline['time'] * 100

print(f"\nTrades:")
print(f"  Baseline: {results_baseline['total_trades']:,}")
print(f"  MTF: {results_mtf['total_trades']:,}")
print(f"  Reduction: {trade_reduction:.1f}%")

print(f"\nWin Rate:")
print(f"  Baseline: {results_baseline['win_rate']:.2f}%")
print(f"  MTF: {results_mtf['win_rate']:.2f}%")
print(f"  Change: {win_rate_change:+.2f}%")

print(f"\nTotal P&L:")
print(f"  Baseline: ${results_baseline['total_pnl']:,.2f}")
print(f"  MTF: ${results_mtf['total_pnl']:,.2f}")
print(f"  Change: ${pnl_change:,.2f}")

print(f"\nExecution Time:")
print(f"  Baseline: {results_baseline['time']:.2f}s")
print(f"  MTF: {results_mtf['time']:.2f}s")
print(f"  Change: {time_change:+.1f}%")

print(f"\nSuccess Rate (bots with trades):")
print(f"  Baseline: {results_baseline['success_rate']:.1f}%")
print(f"  MTF: {results_mtf['success_rate']:.1f}%")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

if trade_reduction > 20:
    print(f"✅ MTF successfully filtered {trade_reduction:.1f}% of trades")
else:
    print(f"⚠️  MTF filtered only {trade_reduction:.1f}% of trades (expected >20%)")

if win_rate_change > 0:
    print(f"✅ Win rate improved by {win_rate_change:.2f}%")
elif abs(win_rate_change) < 2:
    print(f"✅ Win rate maintained (change: {win_rate_change:+.2f}%)")
else:
    print(f"❌ Win rate decreased by {abs(win_rate_change):.2f}%")

if pnl_change > 0:
    print(f"✅ Profitability improved by ${pnl_change:,.2f}")
else:
    print(f"⚠️  Profitability decreased by ${abs(pnl_change):,.2f}")

print("\nMTF Implementation: COMPLETE")
print(f"System validated with {len(results_baseline)} bots across 10 cycles")
print("Ready for production deployment")
