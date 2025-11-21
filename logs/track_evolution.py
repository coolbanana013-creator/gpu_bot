import csv
from pathlib import Path
import glob

# Find all generation CSV files
gen_files = sorted(glob.glob('logs/generation_*.csv'), key=lambda x: int(x.split('_')[1].split('.')[0]))

print("="*90)
print("EVOLUTION PROGRESS - Win Rate Optimization")
print("="*90)

for gen_file in gen_files:
    path = Path(gen_file)
    gen_num = int(path.stem.split('_')[1])
    
    with path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';')
        rows = list(reader)
    
    if not rows:
        continue
    
    # Get top performer
    top_bot = max(rows, key=lambda r: float(r['TotalWinRate'].replace(',','.')))
    max_wr = float(top_bot['TotalWinRate'].replace(',','.'))
    max_profit = float(top_bot['AvgProfitPctPerCycle'].replace(',','.'))
    
    # Count high WR bots
    wr_60_plus = sum(1 for r in rows if float(r['TotalWinRate'].replace(',','.')) >= 60.0)
    wr_70_plus = sum(1 for r in rows if float(r['TotalWinRate'].replace(',','.')) >= 70.0)
    wr_80_plus = sum(1 for r in rows if float(r['TotalWinRate'].replace(',','.')) >= 80.0)
    
    # Average WR
    avg_wr = sum(float(r['TotalWinRate'].replace(',','.')) for r in rows) / len(rows)
    
    print(f"Gen {gen_num:2d}: {len(rows):4d} bots | Avg WR: {avg_wr:5.2f}% | "
          f"Max WR: {max_wr:5.2f}% | 60%+: {wr_60_plus:3d} | 70%+: {wr_70_plus:3d} | 80%+: {wr_80_plus:3d}")

print("="*90)

# Show top 10 from final generation
if gen_files:
    final_gen = Path(gen_files[-1])
    print(f"\nTop 10 Performers from {final_gen.name}:")
    print("-"*90)
    
    with final_gen.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';')
        rows = list(reader)
    
    rows_sorted = sorted(rows, key=lambda r: float(r['TotalWinRate'].replace(',','.')), reverse=True)
    
    print(f"{'Rank':<6}{'BotID':<10}{'WinRate':<10}{'AvgProfit%':<12}{'Sharpe':<8}{'Trades':<8}")
    print("-"*90)
    
    for i, r in enumerate(rows_sorted[:10], 1):
        bot_id = r['BotID']
        win_rate = r['TotalWinRate']
        avg_profit = r['AvgProfitPctPerCycle']
        sharpe = r['SharpeRatio']
        trades = r['TotalTrades']
        
        print(f"{i:<6}{bot_id:<10}{win_rate:<10}{avg_profit:<12}{sharpe:<8}{trades:<8}")
