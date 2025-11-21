import csv
from pathlib import Path
path = Path('c:/Users/Standard/Desktop/gpu_bot/logs/generation_0.csv')
with path.open('r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter=';')
    rows = list(reader)

survivors = [r for r in rows if r['AllCyclesPositive'].lower() == 'true' and r['AllCyclesHaveTrades'].lower() == 'true' and abs(float(r['MaxDrawdown'].replace(',','.'))) < 15]
print('Survivors count:', len(survivors))
survivors_sorted = sorted(survivors, key=lambda r: float(r['TotalWinRate'].replace(',','.')), reverse=True)
print('\nTop 10 survivors by TotalWinRate:')
for r in survivors_sorted[:10]:
    print(f"BotID: {r['BotID']}, TotalWinRate: {r['TotalWinRate']}, AvgProfitPctPerCycle: {r['AvgProfitPctPerCycle']}, MaxDrawdown: {r['MaxDrawdown']}, TotalTrades: {r['TotalTrades']}")

# Print top 10 survivors by AvgProfitPctPerCycle
surv_sorted_profit = sorted(survivors, key=lambda r: float(r['AvgProfitPctPerCycle'].replace(',','.')), reverse=True)
print('\nTop 10 survivors by AvgProfitPctPerCycle:')
for r in surv_sorted_profit[:10]:
    print(f"BotID: {r['BotID']}, AvgProfitPctPerCycle: {r['AvgProfitPctPerCycle']}, TotalWinRate: {r['TotalWinRate']}, MaxDrawdown: {r['MaxDrawdown']}, TotalTrades: {r['TotalTrades']}")
