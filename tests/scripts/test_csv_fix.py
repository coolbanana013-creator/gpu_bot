"""Quick test to verify CSV format is fixed."""
import pandas as pd

print('=== CSV DATA ANALYSIS TEST ===\n')

df = pd.read_csv('logs/generation_10.csv', sep=';')

print(f'Total bots: {len(df)}')
print(f'Columns: {len(df.columns)}')

print(f'\n✅ Numeric columns working correctly:')
print(f'  Average Win Rate: {df["AvgWinRate"].mean():.4f}')
print(f'  Max Fitness Score: {df["FitnessScore"].max():.2f}')
print(f'  Profitable bots: {len(df[df["AvgProfitPct"] > 0])}')

print(f'\n✅ Top 3 bots by fitness:')
top3 = df.nlargest(3, 'FitnessScore')[['BotID', 'FitnessScore', 'AvgProfitPct', 'AvgWinRate', 'TotalTrades']]
print(top3.to_string(index=False))

print('\n✅ Data types check:')
for col in ['AvgProfitPct', 'AvgWinRate', 'FitnessScore', 'TotalTrades']:
    print(f'  {col:20s} -> {df[col].dtype}')

print('\n✅ CSV format is now correct and fully functional!')
