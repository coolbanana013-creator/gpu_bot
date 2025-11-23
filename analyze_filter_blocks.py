import csv
from pathlib import Path
from collections import defaultdict

# Filter bit definitions from OpenCL kernel
FILTER_NAMES = {
    1: "ADX",
    2: "ATR", 
    4: "Volume",
    8: "S/R",
    16: "RSI",
    32: "NaN"
}

# Read filter debug
filter_blocks = defaultdict(lambda: defaultdict(int))
total_blocks = defaultdict(int)

with open('logs/filter_debug.csv', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter=';')
    for row in reader:
        bot_id = int(row['BotID'])
        cycle = int(row['Cycle'])
        bits = int(row['FilterDebugBits'])
        
        # Decode which filters blocked
        for bit_val, name in FILTER_NAMES.items():
            if bits & bit_val:
                filter_blocks[name][bot_id] += 1
                total_blocks[name] += 1

print("Filter Blocking Analysis")
print("=" * 60)
print(f"\nTotal filter blocks by type:")
for filter_name, count in sorted(total_blocks.items(), key=lambda x: x[1], reverse=True):
    print(f"  {filter_name:10s}: {count:6d} blocks")

print(f"\nBots blocked by each filter:")
for filter_name in sorted(FILTER_NAMES.values()):
    if filter_name in filter_blocks:
        bots_affected = len(filter_blocks[filter_name])
        print(f"  {filter_name:10s}: {bots_affected:4d} bots affected")

# Find bots blocked by all filters
with open('logs/filter_debug.csv', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter=';')
    bot_cycles = defaultdict(list)
    for row in reader:
        bot_id = int(row['BotID'])
        cycle = int(row['Cycle'])
        bits = int(row['FilterDebugBits'])
        bot_cycles[bot_id].append((cycle, bits))

# Check if any bot has NO filter blocks in any cycle
print(f"\nBots with at least one cycle having NO filter blocks:")
no_block_bots = []
for bot_id, cycles in bot_cycles.items():
    for cycle, bits in cycles:
        if bits == 0:
            no_block_bots.append(bot_id)
            break

if no_block_bots:
    print(f"  Found {len(no_block_bots)} bots with unblocked cycles")
    print(f"  Examples: {no_block_bots[:10]}")
else:
    print(f"  NONE - All bots are blocked by filters in ALL cycles!")

# Most common filter combinations
print(f"\nMost common filter combinations:")
combo_counts = defaultdict(int)
for bot_id, cycles in bot_cycles.items():
    for cycle, bits in cycles:
        combo_counts[bits] += 1

for bits, count in sorted(combo_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
    filters = [name for bit_val, name in FILTER_NAMES.items() if bits & bit_val]
    print(f"  Bits {bits:3d} ({', '.join(filters) if filters else 'NONE'}): {count:5d} occurrences")
