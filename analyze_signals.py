"""
Analyze signal generation patterns from backtest logs
"""
import json
from pathlib import Path
from collections import Counter, defaultdict

# Find the most recent run
runs_dir = Path("bots/BTC_USDT/1m")
if not runs_dir.exists():
    print("No runs found")
    exit(1)

latest_run = sorted(runs_dir.glob("run_*"))[-1]
print(f"Analyzing run: {latest_run.name}")
print()

# Load generation 0 logs
gen0_file = latest_run / "logs" / "generation_0.json"
if not gen0_file.exists():
    print(f"No generation 0 log found at {gen0_file}")
    exit(1)

with open(gen0_file) as f:
    data = json.load(f)

print(f"Total bots: {len(data)}")
print()

# Analyze trade patterns
no_trade_count = 0
trade_counts = []
indicator_usage = Counter()
indicator_counts = Counter()

for bot_id, bot_data in data.items():
    results = bot_data.get("results", [])
    
    # Count cycles with no trades
    cycles_with_trades = sum(1 for cycle in results if cycle.get("total_trades", 0) > 0)
    cycles_without_trades = len(results) - cycles_with_trades
    
    if cycles_without_trades == len(results):
        no_trade_count += 1
    
    # Track total trades across all cycles
    total_trades = sum(cycle.get("total_trades", 0) for cycle in results)
    trade_counts.append(total_trades)
    
    # Track indicator usage
    num_indicators = bot_data.get("num_indicators", 0)
    indicator_counts[num_indicators] += 1
    
    for ind_idx in bot_data.get("indicators", []):
        indicator_usage[ind_idx] += 1

print(f"=== TRADE STATISTICS ===")
print(f"Bots with 0 trades across ALL cycles: {no_trade_count}/{len(data)} ({100*no_trade_count/len(data):.1f}%)")
print(f"Total trades across all bots: {sum(trade_counts)}")
print(f"Average trades per bot: {sum(trade_counts)/len(data):.1f}")
print(f"Max trades by single bot: {max(trade_counts)}")
print()

print(f"=== INDICATOR COUNT DISTRIBUTION ===")
for num_ind in sorted(indicator_counts.keys()):
    count = indicator_counts[num_ind]
    pct = 100 * count / len(data)
    print(f"{num_ind} indicators: {count} bots ({pct:.1f}%)")
print()

print(f"=== MOST COMMON INDICATORS ===")
for ind_idx, count in indicator_usage.most_common(10):
    print(f"Indicator {ind_idx}: used by {count} bots")
print()

# Analyze bots with trades vs without
bots_with_trades = [bot_id for bot_id, bot_data in data.items() 
                    if sum(cycle.get("total_trades", 0) for cycle in bot_data.get("results", [])) > 0]
bots_without_trades = [bot_id for bot_id, bot_data in data.items() 
                       if sum(cycle.get("total_trades", 0) for cycle in bot_data.get("results", [])) == 0]

print(f"=== COMPARISON ===")
print(f"Bots WITH trades: {len(bots_with_trades)}/{len(data)} ({100*len(bots_with_trades)/len(data):.1f}%)")
print(f"Bots WITHOUT trades: {len(bots_without_trades)}/{len(data)} ({100*len(bots_without_trades)/len(data):.1f}%)")

# Compare indicator patterns
if bots_with_trades:
    trade_indicators = defaultdict(int)
    for bot_id in bots_with_trades[:100]:  # Sample first 100
        bot_data = data[bot_id]
        for ind_idx in bot_data.get("indicators", []):
            trade_indicators[ind_idx] += 1
    
    print()
    print("Top indicators in TRADING bots:")
    for ind_idx, count in sorted(trade_indicators.items(), key=lambda x: -x[1])[:10]:
        print(f"  Indicator {ind_idx}: {count} occurrences")

if bots_without_trades:
    notrade_indicators = defaultdict(int)
    for bot_id in bots_without_trades[:100]:  # Sample first 100
        bot_data = data[bot_id]
        for ind_idx in bot_data.get("indicators", []):
            notrade_indicators[ind_idx] += 1
    
    print()
    print("Top indicators in NON-TRADING bots:")
    for ind_idx, count in sorted(notrade_indicators.items(), key=lambda x: -x[1])[:10]:
        print(f"  Indicator {ind_idx}: {count} occurrences")
