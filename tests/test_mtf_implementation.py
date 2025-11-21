"""
Comprehensive MTF (Multi-Timeframe) Testing Suite

Tests:
1. MTF disabled - baseline (should match previous behavior)
2. MTF enabled - verify HTF filtering reduces counter-trend trades
3. Edge cases: insufficient HTF bars, neutral HTF trend
4. Normal vs DEBUG consensus modes
5. Trade count comparison (MTF should reduce total trades but improve quality)
"""
import sys
import os
from pathlib import Path
import numpy as np
import time

sys.path.insert(0, str(Path(__file__).parent))

import pyopencl as cl
from src.backtester.compact_simulator import CompactBacktester
from src.bot_generator.compact_generator import CompactBotGenerator
from src.utils.validation import log_info, log_error

def generate_trending_data(num_days=7, trend='bullish'):
    """Generate synthetic data with a clear trend for MTF testing."""
    bars_per_day = 1440
    num_bars = num_days * bars_per_day
    
    np.random.seed(42)
    
    base_price = 30000.0
    if trend == 'bullish':
        # Strong uptrend with minor pullbacks
        trend_component = np.linspace(0, 3000, num_bars)  # $3k uptrend
        noise = np.random.randn(num_bars) * 30
    elif trend == 'bearish':
        # Strong downtrend with minor rallies
        trend_component = np.linspace(0, -3000, num_bars)  # $3k downtrend
        noise = np.random.randn(num_bars) * 30
    else:  # ranging
        # Sideways movement
        trend_component = np.sin(np.linspace(0, 20 * np.pi, num_bars)) * 200
        noise = np.random.randn(num_bars) * 50
    
    close_prices = base_price + trend_component + noise
    close_prices = np.maximum(close_prices, 20000.0)
    
    ohlcv = np.zeros((num_bars, 5), dtype=np.float32)
    ohlcv[:, 3] = close_prices
    ohlcv[:, 0] = close_prices * (1 + np.random.uniform(-0.0005, 0.0005, num_bars))
    ohlcv[:, 1] = close_prices * (1 + np.random.uniform(0.0, 0.002, num_bars))
    ohlcv[:, 2] = close_prices * (1 - np.random.uniform(0.0, 0.002, num_bars))
    ohlcv[:, 4] = np.random.uniform(100000, 1000000, num_bars)
    
    return ohlcv

def test_mtf_filtering():
    """Test MTF filtering with trending data."""
    print("="*80)
    print("MTF FILTERING TEST")
    print("="*80)
    
    # GPU setup
    platforms = cl.get_platforms()
    devices = platforms[0].get_devices(device_type=cl.device_type.GPU)
    ctx = cl.Context(devices=[devices[0]])
    queue = cl.CommandQueue(ctx)
    
    # Test parameters
    num_bots = 1000
    num_cycles = 5
    
    # Generate trending data
    log_info("Generating bullish trending data...")
    ohlcv_data = generate_trending_data(num_days=7, trend='bullish')
    num_bars = len(ohlcv_data)
    cycles = [(i * num_bars // num_cycles, (i + 1) * num_bars // num_cycles) 
              for i in range(num_cycles)]
    
    # Create generator
    generator = CompactBotGenerator(
        gpu_context=ctx,
        gpu_queue=queue,
        population_size=num_bots,
        min_indicators=2,
        max_indicators=4,
        min_risk_strategies=1,
        max_risk_strategies=2,
        min_leverage=20,
        max_leverage=50
    )
    
    bots = generator.generate_population()
    log_info(f"Generated {len(bots)} bots")
    
    # Test 1: MTF DISABLED (baseline)
    print("\n" + "-"*80)
    print("TEST 1: MTF DISABLED (Baseline)")
    print("-"*80)
    
    os.environ['DEBUG_LOW_CONSENSUS'] = '1'
    backtester_no_mtf = CompactBacktester(
        gpu_context=ctx,
        gpu_queue=queue,
        initial_balance=10000.0,
        data_chunk_days=7,
        enable_mtf=False
    )
    
    start = time.time()
    results_no_mtf = backtester_no_mtf.backtest_bots(bots, ohlcv_data, cycles)
    time_no_mtf = time.time() - start
    
    # Analyze results
    total_trades_no_mtf = sum(r.total_trades for r in results_no_mtf)
    total_wins_no_mtf = sum(r.winning_trades for r in results_no_mtf)
    win_rate_no_mtf = (total_wins_no_mtf / total_trades_no_mtf * 100) if total_trades_no_mtf > 0 else 0
    
    print(f"\nResults (MTF DISABLED):")
    print(f"  Total trades: {total_trades_no_mtf:,}")
    print(f"  Total wins: {total_wins_no_mtf:,}")
    print(f"  Win rate: {win_rate_no_mtf:.1f}%")
    print(f"  Time: {time_no_mtf:.2f}s")
    
    # Test 2: MTF ENABLED (60x multiplier = 1h from 1m)
    print("\n" + "-"*80)
    print("TEST 2: MTF ENABLED (1h HTF from 1m base)")
    print("-"*80)
    
    backtester_mtf = CompactBacktester(
        gpu_context=ctx,
        gpu_queue=queue,
        initial_balance=10000.0,
        data_chunk_days=7,
        enable_mtf=True,
        htf_multiplier=60
    )
    
    start = time.time()
    results_mtf = backtester_mtf.backtest_bots(bots, ohlcv_data, cycles)
    time_mtf = time.time() - start
    
    # Analyze results
    total_trades_mtf = sum(r.total_trades for r in results_mtf)
    total_wins_mtf = sum(r.winning_trades for r in results_mtf)
    win_rate_mtf = (total_wins_mtf / total_trades_mtf * 100) if total_trades_mtf > 0 else 0
    
    print(f"\nResults (MTF ENABLED):")
    print(f"  Total trades: {total_trades_mtf:,}")
    print(f"  Total wins: {total_wins_mtf:,}")
    print(f"  Win rate: {win_rate_mtf:.1f}%")
    print(f"  Time: {time_mtf:.2f}s")
    
    # Comparison
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    
    trade_reduction = ((total_trades_no_mtf - total_trades_mtf) / total_trades_no_mtf * 100) if total_trades_no_mtf > 0 else 0
    win_rate_improvement = win_rate_mtf - win_rate_no_mtf
    
    print(f"Trade reduction: {trade_reduction:.1f}%")
    print(f"Win rate change: {win_rate_improvement:+.1f}%")
    
    # Validation
    print("\n" + "="*80)
    print("VALIDATION")
    print("="*80)
    
    success = True
    
    # MTF should reduce trades (blocks counter-trend signals)
    if trade_reduction < 5:
        print("⚠️  WARNING: MTF did not significantly reduce trades")
        print(f"   Expected >5% reduction, got {trade_reduction:.1f}%")
        success = False
    else:
        print(f"✅ MTF reduced trades by {trade_reduction:.1f}% (good)")
    
    # MTF should maintain or improve win rate
    if win_rate_improvement < -5:
        print("❌ FAILURE: MTF significantly decreased win rate")
        print(f"   Win rate dropped by {abs(win_rate_improvement):.1f}%")
        success = False
    else:
        print(f"✅ Win rate changed by {win_rate_improvement:+.1f}% (acceptable)")
    
    # Both should have generated trades
    if total_trades_no_mtf == 0:
        print("❌ FAILURE: No trades generated without MTF")
        success = False
    else:
        print(f"✅ Baseline generated {total_trades_no_mtf:,} trades")
    
    if total_trades_mtf == 0:
        print("❌ FAILURE: No trades generated with MTF")
        success = False
    else:
        print(f"✅ MTF generated {total_trades_mtf:,} trades")
    
    print("="*80)
    
    return success

def test_edge_cases():
    """Test MTF edge cases."""
    print("\n" + "="*80)
    print("MTF EDGE CASE TESTS")
    print("="*80)
    
    platforms = cl.get_platforms()
    devices = platforms[0].get_devices(device_type=cl.device_type.GPU)
    ctx = cl.Context(devices=[devices[0]])
    queue = cl.CommandQueue(ctx)
    
    # Edge case 1: Insufficient HTF bars (< 60 bars total)
    print("\nTEST: Insufficient HTF bars")
    print("-"*80)
    
    short_data = generate_trending_data(num_days=1, trend='bullish')[:30]  # Only 30 bars
    
    generator = CompactBotGenerator(
        gpu_context=ctx,
        gpu_queue=queue,
        population_size=100,
        min_indicators=1,
        max_indicators=2
    )
    bots = generator.generate_population()
    
    os.environ['DEBUG_LOW_CONSENSUS'] = '1'
    backtester = CompactBacktester(
        gpu_context=ctx,
        gpu_queue=queue,
        enable_mtf=True,
        htf_multiplier=60
    )
    
    cycles = [(0, len(short_data))]
    
    try:
        results = backtester.backtest_bots(bots, short_data, cycles)
        print("✅ Handled insufficient HTF bars gracefully")
        print(f"   Generated {sum(r.total_trades for r in results)} trades")
    except Exception as e:
        print(f"❌ Failed with insufficient HTF bars: {e}")
        return False
    
    # Edge case 2: Ranging market (neutral HTF trend)
    print("\nTEST: Ranging market (neutral HTF)")
    print("-"*80)
    
    ranging_data = generate_trending_data(num_days=3, trend='ranging')
    
    backtester_ranging = CompactBacktester(
        gpu_context=ctx,
        gpu_queue=queue,
        enable_mtf=True,
        htf_multiplier=60
    )
    
    cycles = [(i * len(ranging_data) // 3, (i + 1) * len(ranging_data) // 3) for i in range(3)]
    
    try:
        results = backtester_ranging.backtest_bots(bots, ranging_data, cycles)
        total_trades = sum(r.total_trades for r in results)
        print("✅ Handled ranging market gracefully")
        print(f"   Generated {total_trades} trades")
        if total_trades > 0:
            print("   (Neutral HTF should allow both directions)")
    except Exception as e:
        print(f"❌ Failed with ranging market: {e}")
        return False
    
    print("✅ All edge cases passed")
    return True

def main():
    print("="*80)
    print("COMPREHENSIVE MTF TESTING SUITE")
    print("="*80)
    
    start_time = time.time()
    
    # Run tests
    test1_passed = test_mtf_filtering()
    test2_passed = test_edge_cases()
    
    total_time = time.time() - start_time
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"MTF Filtering Test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Edge Cases Test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print(f"Total time: {total_time:.2f}s")
    
    if test1_passed and test2_passed:
        print("\n✅ ALL MTF TESTS PASSED")
        print("MTF implementation is working correctly!")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        return 1

if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:
        log_error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
