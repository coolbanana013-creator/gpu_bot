import csv
from pathlib import Path
path = Path('c:/Users/Standard/Desktop/gpu_bot/logs/generation_0.csv')
with path.open('r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter=';')
    rows = list(reader)

rows_sorted = sorted(rows, key=lambda r: float(r['TotalWinRate'].replace(',','.')), reverse=True)
print('Top 10 by TotalWinRate:')
for r in rows_sorted[:10]:
    print(f"BotID: {r['BotID']}, TotalWinRate: {r['TotalWinRate']}, AvgProfitPctPerCycle: {r['AvgProfitPctPerCycle']}, TotalTrades: {r['TotalTrades']}, AllCyclesHaveTrades: {r['AllCyclesHaveTrades']}, AllCyclesPositive: {r['AllCyclesPositive']}")

# Also list top 10 by AvgProfitPctPerCycle
rows_sorted2 = sorted(rows, key=lambda r: float(r['AvgProfitPctPerCycle'].replace(',','.')), reverse=True)
print('\nTop 10 by AvgProfitPctPerCycle:')
for r in rows_sorted2[:10]:
    print(f"BotID: {r['BotID']}, AvgProfitPctPerCycle: {r['AvgProfitPctPerCycle']}, TotalWinRate: {r['TotalWinRate']}, TotalTrades: {r['TotalTrades']}")
