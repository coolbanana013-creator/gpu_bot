import csv
from pathlib import Path
path = Path('c:/Users/Standard/Desktop/gpu_bot/logs/generation_0.csv')
with path.open('r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter=';')
    rows = list(reader)

count_all_iters = sum(1 for r in rows if r['AllCyclesPositive'].lower()=='true')
print('AllCyclesPositive count:', count_all_iters)

count_both = sum(1 for r in rows if r['AllCyclesPositive'].lower()=='true' and r['AllCyclesHaveTrades'].lower()=='true')
print('AllCyclesPositive and AllCyclesHaveTrades true count:', count_both)

count_dd = sum(1 for r in rows if float(r['MaxDrawdown'].replace(',','.')) < 15)
print('MaxDrawdown < 15 count:', count_dd)

count_all_three = sum(1 for r in rows if r['AllCyclesPositive'].lower()=='true' and r['AllCyclesHaveTrades'].lower()=='true' and float(r['MaxDrawdown'].replace(',','.')) < 15)
print('All 3 criteria count:', count_all_three)
