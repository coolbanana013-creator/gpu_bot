"""Check for duplicate indicator combinations in generation 4."""
import pandas as pd

df = pd.read_csv('logs/generation_4.csv', sep=';')
indicators = df['IndicatorsUsed']

print(f'Total bots in generation 4: {len(indicators)}')
print(f'Unique indicator combinations: {indicators.nunique()}')

dups = indicators[indicators.duplicated(keep=False)]
if len(dups) > 0:
    print(f'\n❌ DUPLICATES FOUND: {len(dups)} bots with duplicate combinations')
    print('\nDuplicate combinations:')
    for combo in dups.unique():
        count = (indicators == combo).sum()
        print(f'  "{combo}" appears {count} times')
else:
    print('\n✅ NO DUPLICATES - All bots have unique indicator combinations!')

# Check diversity percentage
diversity_pct = (indicators.nunique() / len(indicators)) * 100
print(f'\nDiversity: {diversity_pct:.2f}%')
