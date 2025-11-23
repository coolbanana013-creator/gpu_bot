import os
os.environ['ENABLE_TRADE_LOGS'] = '0'  # Disable for speed

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

# Generate 100 test bots
generator = CompactBotGenerator(
    gpu_context=ctx,
    gpu_queue=queue,
    population_size=100,
    min_indicators=2,
    max_indicators=5
)
bots = generator.generate_population()
print(f"Generated {len(bots)} bots")

# Run backtest
backtester = CompactBacktester(gpu_context=ctx, gpu_queue=queue, initial_balance=10.0)
results = backtester.backtest_bots(bots, df, cycles)

# Analyze results
print(f"\n{'='*70}")
print("STATISTICAL VALIDATION: 100 BOTS × 5 CYCLES")
print(f"{'='*70}")

bots_all_cycles = 0
bots_with_trades = 0
total_trades = 0
trades_per_cycle = [0] * 5

for i, result in enumerate(results):
    cycle_trades = result.per_cycle_trades
    total = sum(cycle_trades)
    all_cycles_traded = all(t > 0 for t in cycle_trades)
    
    if all_cycles_traded:
        bots_all_cycles += 1
    if total > 0:
        bots_with_trades += 1
    
    total_trades += total
    for j, t in enumerate(cycle_trades):
        trades_per_cycle[j] += t

# Calculate statistics
avg_trades_per_bot = total_trades / len(bots)
avg_trades_per_cycle = [t / len(bots) for t in trades_per_cycle]

print(f"\nBots trading in ALL cycles: {bots_all_cycles}/{len(bots)} ({100*bots_all_cycles/len(bots):.1f}%)")
print(f"Bots with any trades: {bots_with_trades}/{len(bots)} ({100*bots_with_trades/len(bots):.1f}%)")
print(f"\nTotal trades: {total_trades:,}")
print(f"Average trades per bot: {avg_trades_per_bot:.1f}")
print(f"\nTrades per cycle (average across all bots):")
for i, avg in enumerate(avg_trades_per_cycle):
    print(f"  Cycle {i}: {trades_per_cycle[i]:,} total ({avg:.1f} per bot)")

# Statistical significance check
min_trades_per_cycle = 30  # Reasonable minimum for statistical significance
statistically_significant = all(avg >= min_trades_per_cycle for avg in avg_trades_per_cycle)

print(f"\n{'='*70}")
if bots_all_cycles == len(bots) and statistically_significant:
    print("✅ SUCCESS: All criteria met!")
    print(f"   - 100% of bots trade in all cycles")
    print(f"   - All cycles have ≥{min_trades_per_cycle} trades/bot (statistically significant)")
    print("   - Filters are properly calibrated for 1m timeframe")
else:
    if bots_all_cycles < len(bots):
        print(f"⚠️  {len(bots) - bots_all_cycles} bots missing trades in some cycles")
    if not statistically_significant:
        cycles_low = [i for i, avg in enumerate(avg_trades_per_cycle) if avg < min_trades_per_cycle]
        print(f"⚠️  Cycles {cycles_low} have <{min_trades_per_cycle} trades/bot")
    print("   Need further filter adjustment...")
print(f"{'='*70}")
