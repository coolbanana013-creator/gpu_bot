"""
Automated test for multi-cycle optimization.
Tests the new parallel cycle processing without user input.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.bot_generator.compact_generator import CompactBotGenerator, CompactBotConfig
from src.backtester.compact_simulator import CompactBacktester
from src.data_provider.loader import DataLoader
from src.utils.validation import log_info, log_error
import pyopencl as cl

def test_multicycle_processing():
    """Test the new multi-cycle parallel processing."""
    
    print("\n" + "="*80)
    print("AUTOMATED MULTI-CYCLE TEST")
    print("="*80 + "\n")
    
    # Initialize GPU
    log_info("Initializing GPU...")
    try:
        platforms = cl.get_platforms()
        if not platforms:
            log_error("No OpenCL platforms found!")
            return False
        
        devices = platforms[0].get_devices(device_type=cl.device_type.GPU)
        if not devices:
            log_error("No GPU devices found!")
            return False
        
        gpu_context = cl.Context([devices[0]])
        gpu_queue = cl.CommandQueue(gpu_context)
        
        log_info(f"Using GPU: {devices[0].name}")
        log_info(f"Global memory: {devices[0].global_mem_size / (1024**3):.2f} GB")
        
    except Exception as e:
        log_error(f"GPU initialization failed: {e}")
        return False
    
    # Load small dataset for testing
    log_info("\nLoading test data...")
    try:
        from pathlib import Path
        
        # Find available data files
        data_dir = Path(__file__).parent.parent / "data" / "BTC_USDT" / "15m"
        parquet_files = sorted(list(data_dir.glob("*.parquet")))
        
        if not parquet_files:
            log_error(f"No parquet files found in {data_dir}")
            return False
        
        # Use all available files (already limited to recent data)
        log_info(f"Using {len(parquet_files)} data files from {data_dir}")
        
        loader = DataLoader(
            file_paths=parquet_files,
            timeframe="15m",
            random_seed=42,
            gpu_context=gpu_context,
            gpu_queue=gpu_queue,
            use_gpu_processing=True
        )
        ohlcv_df = loader.load_all_data()
        ohlcv_data = ohlcv_df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].values.astype(np.float32)
        
        log_info(f"Loaded {len(ohlcv_data)} bars ({len(ohlcv_data)/96:.1f} days)")
        
    except Exception as e:
        log_error(f"Data loading failed: {e}")
        return False
    
    # Generate small population for testing
    log_info("\nGenerating test bots...")
    try:
        bot_generator = CompactBotGenerator(
            gpu_context=gpu_context,
            gpu_queue=gpu_queue,
            population_size=100,  # Small population for fast testing
            min_indicators=3,
            max_indicators=5,
            min_risk_strategies=2,
            max_risk_strategies=5,
            min_leverage=1,
            max_leverage=10,
            random_seed=42
        )
        
        bots = bot_generator.generate_population()
        log_info(f"Generated {len(bots)} bots")
        
    except Exception as e:
        log_error(f"Bot generation failed: {e}")
        return False
    
    # Create test cycles (5 cycles of 7 days each)
    log_info("\nCreating test cycles...")
    bars_per_day = 96  # 15m timeframe
    days_per_cycle = 5
    bars_per_cycle = days_per_cycle * bars_per_day
    
    cycles = []
    total_bars = len(ohlcv_data)
    
    for i in range(5):  # 5 cycles
        start = i * bars_per_cycle
        end = min(start + bars_per_cycle, total_bars)
        if end > start:
            cycles.append((start, end))
    
    log_info(f"Created {len(cycles)} cycles:")
    for i, (start, end) in enumerate(cycles):
        log_info(f"  Cycle {i+1}: bars {start:,} to {end:,} ({(end-start)/bars_per_day:.1f} days)")
    
    # Run backtest with multi-cycle processing
    log_info("\nRunning multi-cycle backtest...")
    try:
        backtester = CompactBacktester(
            gpu_context=gpu_context,
            gpu_queue=gpu_queue,
            initial_balance=100.0,
            target_chunk_seconds=1.0
        )
        
        import time
        start_time = time.time()
        
        results = backtester.backtest_bots(
            bots=bots,
            ohlcv_data=ohlcv_data,
            cycles=cycles
        )
        
        elapsed = time.time() - start_time
        
        log_info(f"\n✅ Backtest completed in {elapsed:.2f}s")
        log_info(f"   Processed {len(bots)} bots × {len(cycles)} cycles = {len(bots) * len(cycles)} workloads")
        log_info(f"   Throughput: {(len(bots) * len(cycles)) / elapsed:.0f} workloads/second")
        
        # Validate results
        if len(results) != len(bots):
            log_error(f"Expected {len(bots)} results, got {len(results)}")
            return False
        
        # Check result structure
        profitable_count = sum(1 for r in results if r.total_pnl > 0)
        log_info(f"\n   Results: {profitable_count}/{len(results)} bots profitable")
        
        # Show top 3 bots
        sorted_results = sorted(results, key=lambda r: r.total_pnl, reverse=True)[:3]
        log_info("\n   Top 3 bots:")
        for i, result in enumerate(sorted_results, 1):
            log_info(f"     {i}. Bot {result.bot_id}: PnL ${result.total_pnl:.2f}, "
                    f"Trades: {result.total_trades}, Win Rate: {result.win_rate:.1%}")
        
        return True
        
    except Exception as e:
        log_error(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_multicycle_processing()
    
    if success:
        print("\n" + "="*80)
        print("✅ MULTI-CYCLE TEST PASSED")
        print("="*80 + "\n")
        sys.exit(0)
    else:
        print("\n" + "="*80)
        print("❌ MULTI-CYCLE TEST FAILED")
        print("="*80 + "\n")
        sys.exit(1)
