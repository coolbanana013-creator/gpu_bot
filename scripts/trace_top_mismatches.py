import csv
import argparse
import subprocess
from collections import defaultdict

parser = argparse.ArgumentParser(description='Trace top N mismatched bots using existing trace_mismatch.py')
parser.add_argument('--top', type=int, default=5, help='Number of top mismatched bots to trace')
parser.add_argument('--generation', type=int, default=0)
parser.add_argument('--fuzzy', action='store_true', help='Use fuzzy dedup from analyzer')
args = parser.parse_args()

gen_csv = f'logs/generation_{args.generation}.csv'
trade_csv = 'logs/trade_logs.csv'

bots_per_cycle_pnl = {}
with open(gen_csv, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter=';')
    for row in reader:
        bot_id = int(row['BotID'])
        cycle_pnls = []
        for key in row.keys():
            if key.endswith('_TotalPnL'):
                val = row[key].replace('.', '').replace(',', '.') if row[key] else '0'
                try:
                    cycle_pnls.append(float(val))
                except:
                    cycle_pnls.append(0.0)
        bots_per_cycle_pnl[bot_id] = cycle_pnls

# accumulate per-trade sums
trade_accum = defaultdict(lambda: defaultdict(float))
trade_rows = []
with open(trade_csv, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter=';')
    for row in reader:
        try:
            bot = int(row['BotID'])
            cycle = int(row['Cycle'])
            pnl = float(row['PnL'].replace(',', '.'))
        except Exception:
            continue
        trade_accum[bot][cycle] += pnl
        trade_rows.append(row)

# Compute mismatches list
mismatches = []
for bot, cycles in trade_accum.items():
    for cycle, pnl_sum in cycles.items():
        if bot in bots_per_cycle_pnl and cycle < len(bots_per_cycle_pnl[bot]):
            expected = bots_per_cycle_pnl[bot][cycle]
            if abs(pnl_sum - expected) > 0.01:
                mismatches.append((bot, cycle, expected, pnl_sum, abs(pnl_sum-expected)))

if not mismatches:
    print('No mismatches found')
    raise SystemExit(0)

# Rank by absolute difference
mismatches.sort(key=lambda x: x[4], reverse=True)
top = mismatches[:args.top]
print('Top mismatches (bot, cycle, expected, actual, diff):')
for t in top:
    print(t)

# Trace each top bot
for bot, cycle, exp, actual, diff in top:
    print('\nTracing bot', bot)
    cmd = ['python', 'scripts/trace_mismatch.py', '--bot', str(bot), '--generation', str(args.generation)]
    subprocess.run(cmd)

print('\nTraced top bots; check analysis/mismatch_trace_outputs for trace files.')
