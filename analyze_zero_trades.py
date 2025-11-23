import csv
from pathlib import Path

# Find bots with zero trades in at least one cycle
rows = list(csv.DictReader(open('logs/generation_0.csv', encoding='utf-8'), delimiter=';'))
zero_trades = []

for r in rows:
    trades_str = r.get('Trades(perCycle)', '0')
    trades = [int(x) for x in trades_str.split('|')]
    if any(t == 0 for t in trades):
        zero_trades.append((r['BotID'], trades_str, r.get('IndicatorsUsed', '')))

print(f'Bots with zero trades in at least one cycle: {len(zero_trades)} / {len(rows)}')
print('\nFirst 10 examples:')
for bot_id, trades, indicators in zero_trades[:10]:
    print(f"  Bot {bot_id}: trades={trades}, indicators={indicators}")
