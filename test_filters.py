import os
os.environ['ENABLE_TRADE_LOGS'] = '1'
os.environ['TRADE_LOG_MAX'] = '50000'

from pathlib import Path
import pyopencl as cl
from src.data_provider.loader import DataLoader
from src.bot_generator.compact_generator import CompactBotGenerator
from src.backtester.compact_simulator import CompactBacktester

# Setup GPU
ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)

# Load data
data_dir = Path('data/BTC_USDT/1m')
file_paths = sorted(data_dir.glob('*.parquet'))
loader = DataLoader(file_paths=file_paths, timeframe='1m', random_seed=42, gpu_context=ctx, gpu_queue=queue, use_gpu_processing=False)
df = loader.load_all_data()
cycles = loader.generate_cycle_ranges(5, 7)

print(f"Loaded {len(df)} bars")
print(f"Cycle ranges: {cycles}")

# Generate test bots
generator = CompactBotGenerator(
    gpu_context=ctx,
    gpu_queue=queue,
    population_size=10,
    min_indicators=2,
    max_indicators=5
)
bots = generator.generate_population()

print(f"\nGenerated {len(bots)} bots")
for i, bot in enumerate(bots[:3]):
    from src.indicators.gpu_indicators import get_gpu_indicator_name
    ind_names = [get_gpu_indicator_name(bot.indicator_indices[j]) for j in range(bot.num_indicators)]
    print(f"  Bot {i}: {', '.join(ind_names)}")

# Run backtest
backtester = CompactBacktester(gpu_context=ctx, gpu_queue=queue, initial_balance=10.0)
results = backtester.backtest_bots(bots, df, cycles)

# Analyze results
print(f"\n{'='*60}")
print("BACKTEST RESULTS AFTER FILTER ADJUSTMENT")
print(f"{'='*60}")

bots_with_trades = 0
total_trades = 0

for i, result in enumerate(results):
    trades_per_cycle = result.per_cycle_trades
    total = sum(trades_per_cycle)
    all_cycles_traded = all(t > 0 for t in trades_per_cycle)
    
    if total > 0:
        bots_with_trades += 1
    total_trades += total
    
    if i < 10:  # Show first 10
        status = "✓ ALL CYCLES" if all_cycles_traded else "✗ MISSING"
        print(f"Bot {i:2d}: {trades_per_cycle} total={total:4d} {status}")

print(f"\n{'-'*60}")
print(f"Bots with trades: {bots_with_trades}/{len(bots)} ({100*bots_with_trades/len(bots):.1f}%)")
print(f"Total trades: {total_trades}")
print(f"Average trades per bot: {total_trades/len(bots):.1f}")

# Check if all bots trade in all cycles
all_pass = all(all(t > 0 for t in r.per_cycle_trades) for r in results)
if all_pass:
    print("\n✅ SUCCESS: All bots trade in all cycles!")
else:
    bots_all_cycles = sum(1 for r in results if all(t > 0 for t in r.per_cycle_trades))
    print(f"\n⚠️  {bots_all_cycles}/{len(bots)} bots trade in all cycles")
    print("Need to relax filters more...")
