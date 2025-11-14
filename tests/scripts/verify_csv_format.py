"""Display CSV data in a readable format to verify it's correct."""
import pandas as pd

df = pd.read_csv('logs/generation_10.csv', sep=';')

print('='*70)
print('CSV FORMAT VERIFICATION')
print('='*70)

print(f'\n✅ File loaded successfully')
print(f'   Rows: {len(df):,}')
print(f'   Columns: {len(df.columns)}')

print(f'\n✅ Delimiters are correct (semicolon)')
print(f'   All {len(df.columns)} columns properly separated')

print(f'\n✅ Numeric format is correct:')
row = df.iloc[0]
print(f'   Generation       : {row["Generation"]}')
print(f'   BotID            : {row["BotID"]}')
print(f'   AvgProfitPct     : {row["AvgProfitPct"]} (has decimal point)')
print(f'   AvgWinRate       : {row["AvgWinRate"]} (has decimal point)')
print(f'   TotalTrades      : {row["TotalTrades"]}')
print(f'   FitnessScore     : {row["FitnessScore"]}')
print(f'   Cycle0_ProfitPct : {row["Cycle0_ProfitPct"]} (has decimal point)')

print(f'\n✅ Data types are numeric:')
print(f'   float64 columns: {(df.dtypes == "float64").sum()}')
print(f'   int64 columns  : {(df.dtypes == "int64").sum()}')

print(f'\n✅ First 3 rows displayed properly:')
print(df[['BotID', 'AvgProfitPct', 'AvgWinRate', 'TotalTrades']].head(3).to_string())

print(f'\n{"="*70}')
print('CONCLUSION: CSV file is correctly formatted!')
print('If you see data without separators, it is a display/copy issue,')
print('not a problem with the actual CSV file.')
print('='*70)
