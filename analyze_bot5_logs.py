import csv

rows = [r for r in csv.DictReader(open('logs/trade_logs.csv', encoding='utf-8'), delimiter=';') if int(r['BotID']) == 5]

print(f'Total logged trades for Bot 5: {len(rows)}')
print('\nAll trades:')
for i, r in enumerate(rows):
    print(f"  {i+1}. Cycle {r['Cycle']}: Entry {r['EntryBar']} Exit {r['ExitBar']} PnL={r['PnL']} OutOfCycle={r.get('OutOfCycle', '0')} ChunkID={r.get('ChunkID', 'N/A')}")
