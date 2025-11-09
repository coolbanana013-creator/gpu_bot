"""
Comprehensive duplicate checker for all generation CSV files.
Checks each generation for duplicate indicator combinations.
"""
import pandas as pd
import os
import glob

print("\n" + "="*70)
print("COMPREHENSIVE GENERATION DUPLICATE CHECK")
print("="*70)

# Find all generation CSV files
csv_files = sorted(glob.glob('logs/generation_*.csv'))

if not csv_files:
    print("\n❌ No generation CSV files found in logs/")
    exit(1)

print(f"\nFound {len(csv_files)} generation files\n")

all_results = []

for csv_file in csv_files:
    gen_num = os.path.basename(csv_file).replace('generation_', '').replace('.csv', '')
    
    try:
        df = pd.read_csv(csv_file, sep=';')
        indicators = df['IndicatorsUsed']
        
        total = len(indicators)
        unique = indicators.nunique()
        diversity_pct = (unique / total) * 100
        
        result = {
            'generation': gen_num,
            'total_bots': total,
            'unique_combos': unique,
            'diversity_pct': diversity_pct,
            'duplicates': total - unique
        }
        all_results.append(result)
        
        status = "✅" if diversity_pct >= 99.9 else "❌"
        print(f"{status} Generation {gen_num:>2}: {unique:>5}/{total:>5} unique ({diversity_pct:>6.2f}%) | {result['duplicates']:>4} duplicates")
        
        # Show top duplicates if any
        if result['duplicates'] > 0:
            dups = indicators[indicators.duplicated(keep=False)]
            dup_counts = dups.value_counts().head(5)
            print(f"    Top duplicates:")
            for combo, count in dup_counts.items():
                print(f"      - '{combo}' × {count}")
    
    except Exception as e:
        print(f"❌ Error reading {csv_file}: {e}")

print("\n" + "-"*70)
print("SUMMARY")
print("-"*70)

if all_results:
    avg_diversity = sum(r['diversity_pct'] for r in all_results) / len(all_results)
    total_duplicates = sum(r['duplicates'] for r in all_results)
    
    print(f"Average diversity: {avg_diversity:.2f}%")
    print(f"Total duplicates across all generations: {total_duplicates}")
    
    if avg_diversity >= 99.9:
        print("\n✅ EXCELLENT - Near 100% diversity achieved!")
    elif avg_diversity >= 95:
        print("\n⚠️  GOOD - High diversity but room for improvement")
    elif avg_diversity >= 80:
        print("\n⚠️  FAIR - Moderate diversity, optimization recommended")
    else:
        print("\n❌ POOR - Low diversity, major issues detected")

print("="*70 + "\n")
