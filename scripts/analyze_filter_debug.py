"""Analyze filter debug bits from filter_reenable_summary.csv to identify top blockers.

This script:
- Reads logs/filter_reenable_summary.csv
- Decodes filter bit masks for each bot/config
- Aggregates frequency of each filter and filter combinations
- Produces a summary report showing which filters block most frequently
"""
import csv
from pathlib import Path
from collections import Counter, defaultdict

# Filter bit definitions
FILTER_BIT_ADX = 1 << 0      # 0x01
FILTER_BIT_ATR = 1 << 1      # 0x02
FILTER_BIT_VOLUME = 1 << 2   # 0x04
FILTER_BIT_SR = 1 << 3       # 0x08
FILTER_BIT_RSI = 1 << 4      # 0x10
FILTER_BIT_NAN = 1 << 5      # 0x20

FILTER_NAMES = {
    FILTER_BIT_ADX: 'ADX',
    FILTER_BIT_ATR: 'ATR',
    FILTER_BIT_VOLUME: 'Volume',
    FILTER_BIT_SR: 'S/R',
    FILTER_BIT_RSI: 'RSI',
    FILTER_BIT_NAN: 'NaN',
}

def decode_bits(bits_int):
    """Return list of filter names that are set in the bitmask."""
    filters = []
    for bit, name in FILTER_NAMES.items():
        if bits_int & bit:
            filters.append(name)
    return filters

def main():
    csv_path = Path('logs') / 'filter_reenable_summary.csv'
    if not csv_path.exists():
        print(f'[ERROR] {csv_path} not found')
        return
    
    # Per-config statistics
    configs = ['all_bypass', 'quality_on_srvol_bypass', 'quality_on_volume_on', 'all_on']
    
    # Per-config: counter of individual filter bits
    config_filter_counts = {cfg: Counter() for cfg in configs}
    # Per-config: counter of bit combinations (full mask)
    config_combo_counts = {cfg: Counter() for cfg in configs}
    # Per-config: how many bots had zero trades
    config_zero_trade_bots = {cfg: 0 for cfg in configs}
    
    total_bots = 0
    
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            bot_id = row['bot_id']
            total_bots += 1
            
            for cfg in configs:
                # Check if bot had zero trades for this config
                trades_str = row.get(cfg, '0|0|0|0|0')
                trades = [int(x) for x in trades_str.split('|')]
                if all(t == 0 for t in trades):
                    config_zero_trade_bots[cfg] += 1
                
                # Parse filter bits for this config
                bits_col = cfg + '_filter_bits'
                bits_str = row.get(bits_col, '0x0|0x0|0x0|0x0|0x0')
                bits_list = bits_str.split('|')
                
                for bits_hex in bits_list:
                    try:
                        bits_int = int(bits_hex, 16)
                    except ValueError:
                        continue
                    
                    if bits_int == 0:
                        continue
                    
                    # Count full combination
                    config_combo_counts[cfg][bits_int] += 1
                    
                    # Count individual filters
                    for bit, name in FILTER_NAMES.items():
                        if bits_int & bit:
                            config_filter_counts[cfg][name] += 1
    
    print(f'\n{"="*80}')
    print(f'FILTER DEBUG ANALYSIS - {total_bots} Bots Tested')
    print(f'{"="*80}\n')
    
    for cfg in configs:
        print(f'\n{"-"*80}')
        print(f'CONFIG: {cfg}')
        print(f'{"-"*80}')
        print(f'Bots with zero trades: {config_zero_trade_bots[cfg]} / {total_bots} ({100*config_zero_trade_bots[cfg]/total_bots:.1f}%)')
        
        print(f'\nIndividual Filter Frequencies (how many times each filter blocked):')
        if config_filter_counts[cfg]:
            for name, count in config_filter_counts[cfg].most_common():
                print(f'  {name:12s}: {count:5d} occurrences')
        else:
            print('  (No filters triggered)')
        
        print(f'\nTop Filter Combinations (bitmasks):')
        if config_combo_counts[cfg]:
            for bits_int, count in config_combo_counts[cfg].most_common(10):
                filters = decode_bits(bits_int)
                print(f'  0x{bits_int:02x} ({"+".join(filters):30s}): {count:5d} occurrences')
        else:
            print('  (No filter combinations)')
    
    print(f'\n{"="*80}')
    print('SUMMARY')
    print(f'{"="*80}')
    print(f'Total bots: {total_bots}')
    print(f'\nZero-trade rates per config:')
    for cfg in configs:
        rate = 100 * config_zero_trade_bots[cfg] / total_bots if total_bots > 0 else 0
        print(f'  {cfg:30s}: {config_zero_trade_bots[cfg]:3d} / {total_bots} ({rate:5.1f}%)')
    
    print(f'\nMost frequent blockers across all configs:')
    all_filters = Counter()
    for cfg in configs:
        all_filters.update(config_filter_counts[cfg])
    for name, count in all_filters.most_common():
        print(f'  {name:12s}: {count:6d} total occurrences')
    
    print()
    # Read aggregated per-bot per-filter counts (if available)
    fc_path = Path('logs') / 'filter_debug_counts.csv'
    if fc_path.exists():
        fc_counts = Counter()
        with open(fc_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                for filter_name in ['ADX', 'ATR', 'VOLUME', 'SR', 'RSI', 'NAN']:
                    try:
                        cnt = int(row.get(filter_name, '0'))
                    except ValueError:
                        cnt = 0
                    fc_counts[filter_name] += cnt
        print(f'Aggregated per-bot filter counts (from filter_debug_counts.csv):')
        for name, count in fc_counts.most_common():
            print(f'  {name:12s}: {count:6d} occurrences (sum across all bots)')
        print()

if __name__ == '__main__':
    main()
