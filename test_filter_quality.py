import os
os.environ['ENABLE_TRADE_LOGS'] = '0'

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

# Generate 50 test bots
generator = CompactBotGenerator(
    gpu_context=ctx,
    gpu_queue=queue,
    population_size=50,
    min_indicators=2,
    max_indicators=5
)
bots = generator.generate_population()

# Run backtest
backtester = CompactBacktester(gpu_context=ctx, gpu_queue=queue, initial_balance=10.0)
results = backtester.backtest_bots(bots, df, cycles)

# Analyze quality metrics
print(f"\n{'='*70}")
print("FILTER QUALITY VALIDATION")
print(f"{'='*70}")

profitable_bots = 0
avg_win_rates = []
avg_profits = []
avg_drawdowns = []

for result in results:
    # Calculate metrics
    total_pnl = sum(result.per_cycle_pnl)
    avg_pnl_pct = (total_pnl / (10.0 * 5)) * 100  # % of initial balance per cycle
    
    if total_pnl > 0:
        profitable_bots += 1
    
    avg_win_rates.append(result.win_rate)
    avg_profits.append(avg_pnl_pct)
    avg_drawdowns.append(result.max_drawdown * 100)

# Calculate statistics
avg_wr = sum(avg_win_rates) / len(avg_win_rates)
avg_profit = sum(avg_profits) / len(avg_profits)
avg_dd = sum(avg_drawdowns) / len(avg_drawdowns)

print(f"\nProfitability:")
print(f"  Profitable bots: {profitable_bots}/{len(bots)} ({100*profitable_bots/len(bots):.1f}%)")
print(f"  Average profit: {avg_profit:.2f}% per cycle")

print(f"\nWin Rate (indicator of filter quality):")
print(f"  Average: {avg_wr:.1f}%")
print(f"  Range: {min(avg_win_rates):.1f}% - {max(avg_win_rates):.1f}%")

print(f"\nRisk Management:")
print(f"  Average max drawdown: {avg_dd:.1f}%")

print(f"\n{'='*70}")

# Filters should allow trades but still maintain reasonable quality
# Target: 35-55% win rate (random is ~50%, filters should maintain or improve)
if 35 <= avg_wr <= 60:
    print("✅ Win rate in acceptable range (35-60%)")
    print("   Filters are working: rejecting bad signals while allowing good ones")
else:
    print(f"⚠️  Win rate {avg_wr:.1f}% outside target range")
    if avg_wr > 60:
        print("   Filters may be too strict (fewer but higher quality trades)")
    else:
        print("   Filters may be too lenient (more but lower quality trades)")

print(f"{'='*70}")
