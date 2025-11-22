"""Quick script to analyze why filters still block 100% despite threshold loosening."""
import pandas as pd
import numpy as np

# Read summary to see if there's improvement
df = pd.read_csv('logs/filter_reenable_summary.csv', delimiter=';')

print(f"Bots tested: {len(df)}")
print("\nTrade counts by config:")

for config in ['all_bypass', 'quality_on_srvol_bypass', 'quality_on_volume_on', 'all_on']:
    values = df[config].apply(lambda x: [int(v) for v in str(x).split('|')] if pd.notna(x) and x != '0|0|0|0|0' else [])
    total_trades = sum([sum(v) for v in values if v])
    zero_trade_bots = sum([1 for v in values if not v or sum(v) == 0])
    
    print(f"\n{config}:")
    print(f"  Total trades across all bots: {total_trades}")
    print(f"  Bots with zero trades: {zero_trade_bots}/{len(df)}")
    if total_trades > 0:
        avg_per_bot = total_trades / (len(df) - zero_trade_bots)
        print(f"  Avg trades per active bot: {avg_per_bot:.1f}")

# Parse filter bit patterns
print("\n" + "="*80)
print("FILTER BIT ANALYSIS")
print("="*80)

def parse_bits(bits_str):
    """Parse hex bitmask string like '0x1f|0x1f|0xf|0xf|0xf'"""
    if pd.isna(bits_str) or bits_str == '':
        return []
    return [int(b, 16) for b in str(bits_str).split('|')]

for config in ['quality_on_srvol_bypass', 'quality_on_volume_on', 'all_on']:
    bit_col = f"{config}_filter_bits"
    if bit_col not in df.columns:
        continue
    
    all_bits = []
    for bits_str in df[bit_col]:
        all_bits.extend(parse_bits(bits_str))
    
    unique_patterns = set(all_bits)
    pattern_counts = {p: all_bits.count(p) for p in unique_patterns}
    
    print(f"\n{config}:")
    print(f"  Unique bitmask patterns: {len(unique_patterns)}")
    print(f"  Total cycle measurements: {len(all_bits)}")
    
    # Decode top patterns
    filter_names = {
        0x01: 'ADX',
        0x02: 'ATR',
        0x04: 'Volume',
        0x08: 'S/R',
        0x10: 'RSI',
        0x20: 'NaN'
    }
    
    def decode(mask):
        parts = []
        for bit, name in filter_names.items():
            if mask & bit:
                parts.append(name)
        return '+'.join(parts) if parts else 'No filters'
    
    sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  Top 5 patterns:")
    for pattern, count in sorted_patterns[:5]:
        pct = 100 * count / len(all_bits)
        print(f"    0x{pattern:02x} ({decode(pattern):30s}): {count:4d} ({pct:5.1f}%)")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("If ADX+ATR still appear in 100% of blocked cases, the threshold values")
print("themselves may not be the issue. The problem could be:")
print("  1. check_signal_quality() being called at wrong point (before signals exist)")
print("  2. Per-cycle bitmask accumulation logic (OR-ing all rejections)")
print("  3. Filters checking indicators that aren't part of bot's selected set")
print("  4. MTF/HTF interaction causing premature rejection")
