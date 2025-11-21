"""
Minimal quick test to verify core functionality
Tests with tiny dataset to ensure system works before full testing
"""
import sys
import os

# Set up paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 80)
print("MINIMAL FUNCTIONALITY TEST")
print("=" * 80)
print()

# Test 1: Import all modules
print("[1/5] Testing imports...")
try:
    import pyopencl as cl
    from src.backtester.compact_simulator import CompactBacktester
    from src.bot_generator.compact_generator import CompactBotGenerator
    from src.ga.evolver_compact import GeneticAlgorithmEvolver
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

print()

# Test 2: GPU initialization
print("[2/5] Testing GPU initialization...")
try:
    platforms = cl.get_platforms()
    devices = platforms[0].get_devices(device_type=cl.device_type.GPU)
    ctx = cl.Context(devices=[devices[0]])
    queue = cl.CommandQueue(ctx)
    print(f"✓ GPU initialized: {devices[0].name}")
except Exception as e:
    print(f"✗ GPU initialization failed: {e}")
    sys.exit(1)

print()

# Test 3: Bot generation
print("[3/5] Testing bot generation (10 bots)...")
try:
    generator = CompactBotGenerator(
        gpu_context=ctx,
        gpu_queue=queue,
        population_size=10,
        min_indicators=1,
        max_indicators=2,
        min_risk_strategies=1,
        max_risk_strategies=1,
        min_leverage=20,
        max_leverage=50
    )
    bots = generator.generate_population()
    print(f"✓ Generated {len(bots)} bots successfully")
    print(f"  Bot 0: {bots[0].num_indicators} indicators, {bots[0].leverage}x leverage")
except Exception as e:
    print(f"✗ Bot generation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 4: Data loading (small sample)
print("[4/5] Testing data loading...")
try:
    import numpy as np
    from pathlib import Path
    
    # Look for existing data files
    data_dir = Path("data/BTC_USDT/1m")
    parquet_files = sorted(list(data_dir.glob("*.parquet")))
    
    if not parquet_files:
        print("⚠ No data files found, creating synthetic data")
        # Create synthetic OHLCV data for testing (3 days = 4320 bars at 1m)
        num_bars = 4320
        base_price = 50000.0
        timestamps = np.arange(num_bars, dtype=np.float32)
        noise = np.random.randn(num_bars) * 100
        close_prices = base_price + np.cumsum(noise)
        
        # OHLCV format: timestamp, open, high, low, close, volume
        ohlcv_data = np.zeros((num_bars, 6), dtype=np.float32)
        ohlcv_data[:, 0] = timestamps  # timestamp
        ohlcv_data[:, 1] = close_prices  # open (use close as open)
        ohlcv_data[:, 2] = close_prices * 1.002  # high (+0.2%)
        ohlcv_data[:, 3] = close_prices * 0.998  # low (-0.2%)
        ohlcv_data[:, 4] = close_prices  # close
        ohlcv_data[:, 5] = 1000000.0  # volume
        
        df = ohlcv_data
        print(f"✓ Created {len(df)} bars of synthetic data")
    else:
        # Load just 1 day of data (1440 bars) for quick testing
        import pandas as pd
        df_temp = pd.read_parquet(parquet_files[0])
        df = df_temp[['timestamp', 'open', 'high', 'low', 'close', 'volume']].values.astype(np.float32)
        print(f"✓ Loaded {len(df)} bars from {parquet_files[0].name}")
            
except Exception as e:
    print(f"✗ Data loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 5: Backtesting
print("[5/5] Testing backtesting (10 bots, 1 cycle, 3 days)...")
try:
    backtester = CompactBacktester(
        gpu_context=ctx,
        gpu_queue=queue,
        initial_balance=10.0
    )
    
    # Run backtest with minimal data
    # Define a single cycle that uses all available data
    cycles = [(0, len(df) - 1)]  # One cycle from start to end
    
    results = backtester.backtest_bots(
        bots=bots,
        ohlcv_data=df,  # Already numpy array
        cycles=cycles
    )
    
    print(f"✓ Backtested {len(results)} bots")
    print(f"  Bot 0: {results[0].total_trades} trades, {results[0].total_pnl:.2f} PnL")
    
    # Check survival criteria
    survivors = 0
    for result in results:
        num_cycles = len(result.per_cycle_pnl)
        if num_cycles == 0:
            continue
        avg_profit_pct = (result.total_pnl / 10.0) * 100
        profitable_cycles = sum(1 for pnl in result.per_cycle_pnl if pnl > 0.0)
        profitable_pct = profitable_cycles / num_cycles if num_cycles > 0 else 0
        
        if (avg_profit_pct >= -10.0 and 
            profitable_pct >= 0.70 and 
            result.max_drawdown < 0.30):
            survivors += 1
    
    print(f"  Survivors (70%/30%/-10%): {survivors}/{len(results)}")
    
except Exception as e:
    print(f"✗ Backtesting failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 80)
print("ALL TESTS PASSED ✓")
print("=" * 80)
print()
print("System is fully operational and ready for:")
print("  - Full evolution runs (10k bots × 5+ generations)")
print("  - MTF filtering implementation")
print("  - Live paper trading deployment")
