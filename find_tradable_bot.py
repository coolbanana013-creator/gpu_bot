import csv

with open('logs/generation_0.csv', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter=';')
    for r in reader:
        trades = int(r.get('TotalTrades', 0))
        if trades > 5:
            print(f"Bot {r['BotID']}: {trades} trades, PnL={r['TotalPnL']}, WinRate={r['TotalWinRate']}")
            if trades > 10:
                print(f"  -> Selected Bot {r['BotID']} for tracing")
                break
