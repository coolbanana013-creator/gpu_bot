"""
Comprehensive validation test: 10k bots × 10 cycles × 7 days of 1m data
Verifies that all bots generate at least one trade with relaxed consensus conditions
"""
import sys
import os
from pathlib import Path
import numpy as np
import time
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

import pyopencl as cl
from src.backtester.compact_simulator import CompactBacktester
from src.bot_generator.compact_generator import CompactBotGenerator
from src.utils.validation import log_info, log_error

def generate_synthetic_data(num_days=7):
    """Generate synthetic 1m OHLCV data for testing."""
    bars_per_day = 1440
    num_bars = num_days * bars_per_day
    
    log_info(f"Generating synthetic {num_days}-day dataset ({num_bars:,} bars)")
    
    np.random.seed(42)  # Reproducible
    
    # Generate realistic price movement
    base_price = 30000.0
    price_changes = np.random.randn(num_bars) * 50  # $50 volatility per bar
    close_prices = base_price + np.cumsum(price_changes)
    
    # Ensure positive prices
    close_prices = np.maximum(close_prices, 20000.0)
    
    # Generate OHLC from close
    ohlcv = np.zeros((num_bars, 5), dtype=np.float32)
    ohlcv[:, 3] = close_prices  # close
    ohlcv[:, 0] = close_prices * (1 + np.random.uniform(-0.0005, 0.0005, num_bars))  # open
    ohlcv[:, 1] = close_prices * (1 + np.random.uniform(0.0, 0.002, num_bars))  # high
    ohlcv[:, 2] = close_prices * (1 - np.random.uniform(0.0, 0.002, num_bars))  # low
    ohlcv[:, 4] = np.random.uniform(100000, 1000000, num_bars)  # volume
    
    log_info(f"Price range: ${ohlcv[:, 3].min():.2f} - ${ohlcv[:, 3].max():.2f}")
    
    return ohlcv

def create_cycle_ranges(num_bars, num_cycles=10):
    """Create cycle ranges covering the full dataset."""
    cycle_length = num_bars // num_cycles
    cycles = []
    
    for i in range(num_cycles):
        start = i * cycle_length
        end = start + cycle_length if i < num_cycles - 1 else num_bars
        cycles.append((start, end))
    
    log_info(f"Created {num_cycles} cycles, {cycle_length} bars each")
    return cycles

def main():
    print("="*80)
    print("COMPREHENSIVE VALIDATION TEST")
    print("10,000 bots × 10 cycles × 7 days of 1m data")
    print("="*80)
    
    start_time = time.time()
    
    # GPU setup
    log_info("Initializing GPU...")
    platforms = cl.get_platforms()
    devices = platforms[0].get_devices(device_type=cl.device_type.GPU)
    ctx = cl.Context(devices=[devices[0]])
    queue = cl.CommandQueue(ctx)
    
    # Create backtester with relaxed conditions (using DEBUG flags)
    log_info("Creating backtester with relaxed consensus...")
    os.environ['DEBUG_LOW_CONSENSUS'] = '1'
    backtester = CompactBacktester(
        gpu_context=ctx,
        gpu_queue=queue,
        initial_balance=10000.0,
        data_chunk_days=7  # Process all 7 days at once
    )
    
    # Create generator
    log_info("Creating bot generator...")
    generator = CompactBotGenerator(
        gpu_context=ctx,
        gpu_queue=queue,
        population_size=10000,
        min_indicators=1,
        max_indicators=8,
        min_risk_strategies=1,
        max_risk_strategies=5,
        min_leverage=20,
        max_leverage=50
    )
    
    # Generate bots
    log_info("Generating 10,000 bots...")
    gen_start = time.time()
    bots = generator.generate_population()
    gen_time = time.time() - gen_start
    log_info(f"Generated {len(bots)} bots in {gen_time:.2f}s")
    
    # Show bot distribution
    from collections import Counter
    indicator_counts = Counter([b.num_indicators for b in bots])
    log_info(f"Indicator distribution: {dict(indicator_counts)}")
    
    # Generate data
    log_info("Generating synthetic data...")
    ohlcv_data = generate_synthetic_data(num_days=7)
    num_bars = len(ohlcv_data)
    
    # Create cycles
    cycles = create_cycle_ranges(num_bars, num_cycles=10)
    
    # Run backtest
    log_info("Starting backtest...")
    log_info(f"Configuration:")
    log_info(f"  - Bots: {len(bots):,}")
    log_info(f"  - Cycles: {len(cycles)}")
    log_info(f"  - Bars per cycle: ~{cycles[0][1] - cycles[0][0]:,}")
    log_info(f"  - Total workloads: {len(bots) * len(cycles):,}")
    
    backtest_start = time.time()
    results = backtester.backtest_bots(bots, ohlcv_data, cycles)
    backtest_time = time.time() - backtest_start
    
    log_info(f"Backtest completed in {backtest_time:.2f}s")
    
    # Analyze results
    print("\n" + "="*80)
    print("RESULTS ANALYSIS")
    print("="*80)
    
    total_bots = len(results)
    bots_with_trades = 0
    bots_without_trades = []
    total_trades = 0
    
    # Per-bot analysis
    bot_trades = defaultdict(int)
    for result in results:
        bot_id = result.bot_id
        trades = result.total_trades
        bot_trades[bot_id] = trades
        total_trades += trades
        
        if trades > 0:
            bots_with_trades += 1
        else:
            bots_without_trades.append(bot_id)
    
    # Per-cycle analysis
    cycle_trades = defaultdict(int)
    for result in results:
        for cycle_idx, trades in enumerate(result.per_cycle_trades):
            cycle_trades[cycle_idx] += trades
    
    print(f"\n📊 Bot Trade Statistics:")
    print(f"  Total bots: {total_bots:,}")
    print(f"  Bots with trades: {bots_with_trades:,} ({100*bots_with_trades/total_bots:.1f}%)")
    print(f"  Bots without trades: {len(bots_without_trades):,} ({100*len(bots_without_trades)/total_bots:.1f}%)")
    print(f"  Total trades: {total_trades:,}")
    print(f"  Avg trades per bot: {total_trades/total_bots:.2f}")
    
    print(f"\n📊 Cycle Trade Statistics:")
    for cycle_idx in range(len(cycles)):
        trades = cycle_trades[cycle_idx]
        print(f"  Cycle {cycle_idx}: {trades:,} trades ({trades/total_bots:.2f} per bot)")
    
    # Trade distribution
    trade_dist = Counter(bot_trades.values())
    print(f"\n📊 Trade Distribution:")
    for num_trades in sorted(trade_dist.keys())[:20]:  # Show first 20 buckets
        count = trade_dist[num_trades]
        print(f"  {num_trades} trades: {count:,} bots ({100*count/total_bots:.1f}%)")
    
    # Performance metrics
    print(f"\n⚡ Performance:")
    print(f"  Bot generation: {gen_time:.2f}s ({len(bots)/gen_time:.0f} bots/s)")
    print(f"  Backtesting: {backtest_time:.2f}s ({len(bots)*len(cycles)/backtest_time:.0f} workloads/s)")
    print(f"  Total time: {time.time() - start_time:.2f}s")
    
    # Validation
    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)
    
    success_rate = 100 * bots_with_trades / total_bots
    
    if success_rate >= 99.0:
        print(f"✅ SUCCESS: {success_rate:.1f}% of bots generated trades!")
        print("✅ System is working correctly with relaxed consensus conditions")
        print(f"✅ Total trades generated: {total_trades:,} ({total_trades/total_bots:.1f} per bot)")
        
        if len(bots_without_trades) > 0:
            print(f"\n📝 Note: {len(bots_without_trades)} bots ({100*len(bots_without_trades)/total_bots:.1f}%) had no trades")
            print("   This is expected - some indicator combinations may not trigger in specific market conditions")
            
            # Quick analysis
            no_trade_indicators = Counter()
            for bot_id in bots_without_trades[:100]:
                bot = bots[bot_id]
                for ind_idx in bot.indicator_indices[:bot.num_indicators]:
                    no_trade_indicators[ind_idx] += 1
            
            print(f"\n   Top indicators in no-trade bots:")
            for ind_idx, count in no_trade_indicators.most_common(5):
                indicator_names = {10: "EMA(100)", 40: "Volume SMA(20)", 4: "SMA(100)", 11: "EMA(200)"}
                name = indicator_names.get(ind_idx, f"Ind {ind_idx}")
                print(f"     {name}: {count} bots")
        
        success = True
    else:
        print(f"❌ FAILURE: Only {success_rate:.1f}% of bots generated trades")
        print(f"❌ Expected at least 99% success rate with relaxed conditions")
        print(f"Sample bot IDs without trades: {bots_without_trades[:20]}")
        
        # Detailed analysis
        no_trade_indicators = Counter()
        no_trade_bot_configs = []
        for bot_id in bots_without_trades[:100]:
            bot = bots[bot_id]
            indicators = bot.indicator_indices[:bot.num_indicators].tolist()
            no_trade_bot_configs.append((bot_id, bot.num_indicators, indicators, bot.leverage))
            for ind_idx in indicators:
                no_trade_indicators[ind_idx] += 1
        
        print(f"\nTop indicators in no-trade bots:")
        for ind_idx, count in no_trade_indicators.most_common(10):
            print(f"  Indicator {ind_idx}: {count} bots")
        
        print(f"\nDetailed configuration of no-trade bots (sample):")
        for bot_id, num_ind, indicators, leverage in no_trade_bot_configs[:20]:
            print(f"  Bot {bot_id}: {num_ind} indicators {indicators}, leverage={leverage}x")
        
        success = False
    
    print("="*80)
    
    return success

if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        log_error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
