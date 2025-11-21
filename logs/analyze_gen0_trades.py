import csv
from pathlib import Path
path = Path('c:/Users/Standard/Desktop/gpu_bot/logs/generation_0.csv')
with path.open('r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter=';')
    rows = list(reader)

no_trade_count = sum(1 for r in rows if r['AllCyclesHaveTrades'].lower()!='true')
print('Bots without trades in all cycles:', no_trade_count, 'out of', len(rows))

trade_count = sum(1 for r in rows if r['AllCyclesHaveTrades'].lower()=='true')
print('Bots with trades in all cycles:', trade_count)

sample = [r['BotID'] for r in rows if r['AllCyclesHaveTrades'].lower()!='true'][:10]
print('Sample (up to 10) botIDs missing trades in at least one cycle:', sample)
