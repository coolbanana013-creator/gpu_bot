import pandas as pd

df = pd.read_csv('logs/generation_0.csv', sep=';')

print('Bot stats:')
print(f'Total bots: {len(df)}')
print(f'Bots with trades in Cycle 2: {(df["Cycle2_ProfitPct"] != 0.0).sum()}')
print(f'Avg cycle 2 profit: {df["Cycle2_ProfitPct"].mean():.2f}%')
print(f'Avg max DD: {df["MaxDrawdown"].mean():.4f}')
print(f'Max DD > 40%: {(df["MaxDrawdown"] > 0.4).sum()}')
print(f'Loss < -20%: {(df["Cycle2_ProfitPct"] < -20).sum()}')

print('\nTop 5 bots by Cycle2 profit:')
print(df.nlargest(5, 'Cycle2_ProfitPct')[['BotID', 'Cycle2_ProfitPct', 'MaxDrawdown', 'SharpeRatio', 'Leverage']])

print('\nBots failing survival criteria:')
all_cycles_cols = [f'Cycle{i}_ProfitPct' for i in range(5)]
df['avg_profit'] = df[all_cycles_cols].mean(axis=1)
df['cycles_with_trades'] = (df[[f'Cycle{i}_ProfitPct' for i in range(5)]] != 0).sum(axis=1)
df['profitable_cycles'] = (df[[f'Cycle{i}_ProfitPct' for i in range(5)]] > 0).sum(axis=1)
df['pct_profitable'] = df['profitable_cycles'] / df['cycles_with_trades'].replace(0, 1)

print(f'Avg profit < -20%: {(df["avg_profit"] < -20).sum()}')
print(f'Max DD > 40%: {(df["MaxDrawdown"] > 0.4).sum()}')
print(f'Pct profitable < 40%: {(df["pct_profitable"] < 0.4).sum()}')
print(f'No trades in any cycle: {(df["cycles_with_trades"] == 0).sum()}')
