import csv
from pathlib import Path

path = Path('logs/generation_5.csv')

with path.open('r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter=';')
    rows = list(reader)

# Sort by TotalWinRate
rows_sorted = sorted(rows, key=lambda r: float(r['TotalWinRate'].replace(',','.')), reverse=True)

print(f"Total bots in generation 5: {len(rows)}")
print(f"\nTop 20 bots by TotalWinRate:\n")
print(f"{'Rank':<6}{'BotID':<10}{'WinRate':<12}{'AvgProfit%':<15}{'MaxDD':<10}{'TotalTrades':<12}{'Sharpe':<10}")
print("="*85)

for i, r in enumerate(rows_sorted[:20], 1):
    bot_id = r['BotID']
    win_rate = r['TotalWinRate']
    avg_profit = r['AvgProfitPctPerCycle']
    max_dd = r['MaxDrawdown']
    total_trades = r['TotalTrades']
    sharpe = r['SharpeRatio']
    
    print(f"{i:<6}{bot_id:<10}{win_rate:<12}{avg_profit:<15}{max_dd:<10}{total_trades:<12}{sharpe:<10}")

# Count bots with 80%+ win rate
high_wr_bots = [r for r in rows if float(r['TotalWinRate'].replace(',','.')) >= 80.0]
print(f"\n{'='*85}")
print(f"Bots with 80%+ win rate: {len(high_wr_bots)}")

# Count bots with 70%+ win rate
very_high_wr_bots = [r for r in rows if float(r['TotalWinRate'].replace(',','.')) >= 70.0]
print(f"Bots with 70%+ win rate: {len(very_high_wr_bots)}")

# Count bots with 60%+ win rate
high_wr_bots_60 = [r for r in rows if float(r['TotalWinRate'].replace(',','.')) >= 60.0]
print(f"Bots with 60%+ win rate: {len(high_wr_bots_60)}")

# Check highest win rate achieved
if rows_sorted:
    max_wr = float(rows_sorted[0]['TotalWinRate'].replace(',','.'))
    print(f"\nHighest win rate achieved: {max_wr:.2f}%")
