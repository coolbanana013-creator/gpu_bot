import pandas as pd

# Read CSV with semicolon separator
df = pd.read_csv('logs/generation_0.csv', sep=';')

print('=== GENERATION 0 STATISTICS ===')
print(f'Total bots: {len(df)}\n')

# Convert comma decimal separator to dot for numeric columns
df_numeric = df.copy()
for col in ['AvgProfitPctPerCycle', 'MaxDrawdown', 'SharpeRatio', 'TotalWinRate']:
    df_numeric[col] = df[col].astype(str).str.replace(',', '.').astype(float)

print('Fitness Score:')
print(df['FitnessScore'].describe())

print('\nAvg Profit % Per Cycle:')
print(df_numeric['AvgProfitPctPerCycle'].describe())

print('\nMax Drawdown %:')
print(df_numeric['MaxDrawdown'].describe())

print('\nSharpe Ratio:')
print(df_numeric['SharpeRatio'].describe())

print('\nTotal Win Rate %:')
print(df_numeric['TotalWinRate'].describe())

print('\nTotal Trades:')
print(df['TotalTrades'].describe())

print(f'\n\n=== PROFIT ANALYSIS ===')
print(f'Bots with positive avg profit: {(df_numeric["AvgProfitPctPerCycle"] > 0).sum()}')
print(f'Bots with negative avg profit: {(df_numeric["AvgProfitPctPerCycle"] <= 0).sum()}')
print(f'Bots with 0 trades: {(df["TotalTrades"] == 0).sum()}')

print(f'\n\n=== LEAST WORST BOTS (lowest loss) ===')
df_numeric['FitnessScore_num'] = df['FitnessScore'].astype(str).str.replace(',', '.').astype(float)
df_numeric['AvgProfitPctPerCycle_num'] = df_numeric['AvgProfitPctPerCycle']
least_worst = df_numeric.nlargest(10, 'AvgProfitPctPerCycle_num')
print(least_worst[['BotID', 'FitnessScore_num', 'AvgProfitPctPerCycle_num', 'MaxDrawdown', 'SharpeRatio', 'TotalWinRate', 'TotalTrades', 'NumIndicators', 'RiskStrategies']].to_string())

print(f'\n\n=== SURVIVAL CRITERIA CHECK ===')
print(f'Bots meeting "All cycles profitable": {df["AllCyclesPositive"].sum()}')
print(f'Bots with MaxDrawdown < 15%: {(df_numeric["MaxDrawdown"] < 15).sum()}')
print(f'Bots with Avg Profit > 0: {(df_numeric["AvgProfitPctPerCycle"] > 0).sum()}')

# Check how many meet ALL criteria
meets_all = (
    (df_numeric['AvgProfitPctPerCycle'] > 0) & 
    (df['AllCyclesPositive'] == 'true') & 
    (df_numeric['MaxDrawdown'] < 15)
)
print(f'\nBots meeting ALL 3 survival criteria: {meets_all.sum()}')

print(f'\n\n=== TRADE FREQUENCY ANALYSIS ===')
print(f'Avg trades per bot: {df["TotalTrades"].mean():.1f}')
print(f'Avg trades per cycle: {df["TotalTrades"].mean() / 20:.1f}')
print(f'Bots with >0 trades: {(df["TotalTrades"] > 0).sum()}')
