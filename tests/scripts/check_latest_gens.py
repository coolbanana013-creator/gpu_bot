"""Check only the most recently generated files."""
import pandas as pd

gens_to_check = [0, 1, 2]  # Latest run

print("\n" + "="*60)
print("CHECKING LATEST GENERATIONS (0-2)")
print("="*60 + "\n")

for gen in gens_to_check:
    csv_file = f'logs/generation_{gen}.csv'
    df = pd.read_csv(csv_file, sep=';')
    indicators = df['IndicatorsUsed']
    
    total = len(indicators)
    unique = indicators.nunique()
    diversity = (unique / total) * 100
    dups = total - unique
    
    status = "✅" if diversity >= 99.9 else "❌"
    print(f"{status} Gen {gen}: {unique}/{total} unique ({diversity:.2f}%) - {dups} dups")
    
    if dups > 0:
        top_dups = indicators[indicators.duplicated(keep=False)].value_counts().head(3)
        print(f"   Top: {dict(top_dups)}")

print("\n" + "="*60)
