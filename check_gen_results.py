import csv

data = list(csv.DictReader(open('logs/generation_0.csv', encoding='utf-8'), delimiter=';'))
print(f'Total bots: {len(data)}')
print('\nSample bots with trades:')
count = 0
for r in data:
    if int(r.get('TotalTrades', 0)) > 0:
        print(f"  Bot {r['BotID']}: {r['TotalTrades']} trades, PnL={r['TotalPnL']}, " +
              f"Cycle0={r.get('Cycle0_Trades', 'N/A')} trades, Cycle1={r.get('Cycle1_Trades', 'N/A')} trades")
        count += 1
        if count >= 10:
            break

print(f"\nTotal bots with trades: {sum(1 for r in data if int(r.get('TotalTrades', 0)) > 0)}")
