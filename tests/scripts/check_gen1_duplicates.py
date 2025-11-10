"""Check generation 1 for duplicates after duplicate fix."""
import pandas as pd
import os

if not os.path.exists('logs/generation_1.csv'):
    print("❌ generation_1.csv not found - run main.py first")
    exit(1)

df = pd.read_csv('logs/generation_1.csv', sep=';')
indicators = df['IndicatorsUsed']

total = len(indicators)
unique = indicators.nunique()
diversity_pct = (unique / total) * 100

print(f'\n=== GENERATION 1 DUPLICATE CHECK ===')
print(f'Total bots: {total}')
print(f'Unique combinations: {unique}')
print(f'Diversity: {diversity_pct:.2f}%')

dups = indicators[indicators.duplicated(keep=False)]
if len(dups) > 0:
    print(f'\n❌ DUPLICATES FOUND: {len(dups)} bots with duplicate combinations')
    print('\nTop 10 duplicate combinations:')
    dup_counts = indicators[indicators.duplicated(keep=False)].value_counts().head(10)
    for combo, count in dup_counts.items():
        print(f'  "{combo}" appears {count} times')
else:
    print('\n✅ NO DUPLICATES - All bots in generation 1 have unique indicator combinations!')
