import csv
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser(description='Analyze trade logs mismatches between per-trade logs and generation per-cycle PnL')
parser.add_argument('--fuzzy', action='store_true', help='Enable fuzzy deduplication (merge near-duplicate trades)')
parser.add_argument('--top', type=int, default=0, help='If >0, print only top N mismatches by abs diff')
args = parser.parse_args()

# Read generation CSV to get per-cycle PnL values
gen_csv = 'logs/generation_0.csv'
trade_csv = 'logs/trade_logs.csv'

bots_per_cycle_pnl = {}

with open(gen_csv, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter=';')
    for row in reader:
        bot_id = int(row['BotID']) if row.get('BotID') else int(row['BotID'])
        # Parse per-cycle PnL columns: `Cycle{i}_TotalPnL` or the pattern used in header
        # Header structure: Cycle0_TotalPnL, Cycle0_Trades, ... but we used custom header. We will parse columns that match `Cycle` and `_TotalPnL`
        cycle_pnls = []
        for key in row.keys():
            if key.endswith('_TotalPnL'):
                # Convert comma as decimal to float
                val = row[key].replace('.', '').replace(',', '.') if row[key] else '0'
                try:
                    cycle_pnls.append(float(val))
                except:
                    cycle_pnls.append(0.0)
        bots_per_cycle_pnl[bot_id] = cycle_pnls

# Read trade logs and accumulate per bot/cycle
trade_accum = defaultdict(lambda: defaultdict(float))
trade_counts = defaultdict(lambda: defaultdict(int))
out_of_cycle_counts = defaultdict(lambda: defaultdict(int))
unique_trades = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

with open(trade_csv, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter=';')
    for row in reader:
        try:
            bot = int(row['BotID'])
            cycle = int(row['Cycle'])
            pnl = float(row['PnL'].replace(',', '.'))
            # Optional diagnostics columns added in newer runs
            out_of_cycle = int(row['OutOfCycle']) if 'OutOfCycle' in row and row['OutOfCycle']!='' else 0
            chunk_id = int(row['ChunkID']) if 'ChunkID' in row and row['ChunkID']!='' else -1
        except Exception as e:
            continue
        # Exclude trades that are explicitly flagged as outside the cycle range
        if out_of_cycle:
            out_of_cycle_counts[bot][cycle] += 1
        else:
            trade_accum[bot][cycle] += pnl
            # Track unique trade signature for duplication detection
            sig = (int(row.get('EntryBar') or 0), int(row.get('ExitBar') or 0), row.get('Direction'))
            unique_trades[bot][cycle][sig].append(chunk_id)
        trade_counts[bot][cycle] += 1

# Compare bots that we have logs for
mismatches = []
# Compute deduplicated trade sums per bot/cycle by unique signature (EntryBar, ExitBar, Direction)
deduped_trade_accum = defaultdict(lambda: defaultdict(float))
deduped_counts = defaultdict(lambda: defaultdict(int))
duplicates_removed = 0
unique_seen = set()
# Single-pass dedupe: prefer first occurrence by signature
unique_map = {}
with open(trade_csv, newline='', encoding='utf-8') as f2:
    reader2 = csv.DictReader(f2, delimiter=';')
    for r in reader2:
        try:
            b = int(r['BotID'])
            c = int(r['Cycle'])
            entry = int(r.get('EntryBar') or 0)
            exitb = int(r.get('ExitBar') or 0)
            direction = r.get('Direction')
        except Exception:
            continue
        sig = (b, c, entry, exitb, direction)
        if sig in unique_map:
            duplicates_removed += 1
            continue
        try:
            pnl = float(r['PnL'].replace(',', '.'))
        except Exception:
            pnl = 0.0
        unique_map[sig] = pnl
        deduped_trade_accum[b][c] += pnl
        deduped_counts[b][c] += 1
        unique_seen.add(sig)

    # Fuzzy deduplication (optional): merge trades whose entry/exit bars are +/-1 and pnl similar
    def fuzzy_merge_trades(trade_rows, bar_tolerance=1, pnl_tol=0.5):
        # trade_rows: list of dicts containing 'BotID', 'Cycle', 'EntryBar','ExitBar','Direction','PnL','ChunkID'
        merged = []
        taken = [False] * len(trade_rows)
        for i, a in enumerate(trade_rows):
            if taken[i]:
                continue
            # Base group
            group = [a]
            taken[i] = True
            a_entry = int(a.get('EntryBar') or 0)
            a_exit = int(a.get('ExitBar') or 0)
            try:
                a_pnl = float(str(a.get('PnL') or '0').replace(',', '.'))
            except Exception:
                a_pnl = 0.0
            for j in range(i+1, len(trade_rows)):
                if taken[j]:
                    continue
                b = trade_rows[j]
                if b.get('Direction') != a.get('Direction'):
                    continue
                try:
                    b_entry = int(b.get('EntryBar') or 0)
                    b_exit = int(b.get('ExitBar') or 0)
                    b_pnl = float(str(b.get('PnL') or '0').replace(',', '.'))
                except Exception:
                    continue
                if abs(a_entry - b_entry) <= bar_tolerance and abs(a_exit - b_exit) <= bar_tolerance and abs(a_pnl - b_pnl) <= pnl_tol:
                    group.append(b)
                    taken[j] = True
            # For group, pick a as canonical and sum pnl
            if len(group) > 1:
                merged_pnl = sum([float(str(t['PnL']).replace(',', '.')) for t in group])
                a['PnL'] = merged_pnl
            merged.append(a)
        return merged
for bot, cycles in trade_accum.items():
    for cycle, pnl_sum in cycles.items():
        expected = None
        if bot in bots_per_cycle_pnl and cycle < len(bots_per_cycle_pnl[bot]):
            expected = bots_per_cycle_pnl[bot][cycle]
        else:
            # generation csv might not include this bot; skip
            continue
        if abs(pnl_sum - expected) > 0.01:  # 1 cent tolerance
            mismatches.append((bot, cycle, expected, pnl_sum, trade_counts[bot][cycle]))

print('Logged trades for %d bots' % len(trade_accum))
print('Mismatches found: %d' % len(mismatches))
for bot, cycle, exp, actual, cnt in mismatches[:200]:
    print(f'Bot {bot} cycle {cycle}: Expected {exp:.2f}, SumTrades {actual:.2f} from {cnt} trades')

print('\nOut-of-cycle trade counts (excluded from sums):')
for bot, cycles in out_of_cycle_counts.items():
    for cycle, count in cycles.items():
        if count>0:
            print(f'Bot {bot} cycle {cycle}: {count} out-of-cycle trades')

print('\nDuplicate trades detected in per-trade logs:')
dup_count = 0
for bot, cycles in unique_trades.items():
    for cycle, sigs in cycles.items():
        for sig, chunks in sigs.items():
            if len(chunks) > 1:
                dup_count += 1
                print(f'Bot {bot} cycle {cycle} duplicate trade {sig}: logged in chunks {sorted(set(chunks))}')
print(f'Found {dup_count} duplicate trade signatures across chunks')
print(f'Unique trades (deduped): {len(unique_seen)} duplicates removed: {duplicates_removed}')

# Recompute mismatches using deduped sums
dedup_mismatches = []
for bot, cycles in deduped_trade_accum.items():
    for cycle, pnl_sum in cycles.items():
        expected = None
        if bot in bots_per_cycle_pnl and cycle < len(bots_per_cycle_pnl[bot]):
            expected = bots_per_cycle_pnl[bot][cycle]
        else:
            continue
        if abs(pnl_sum - expected) > 0.01:
            dedup_mismatches.append((bot, cycle, expected, pnl_sum, deduped_counts[bot][cycle]))

print('\nMismatches after deduplication: %d' % len(dedup_mismatches))
for bot, cycle, exp, actual, cnt in dedup_mismatches[:200]:
    print(f'[DEDUPED] Bot {bot} cycle {cycle}: Expected {exp:.2f}, SumTrades {actual:.2f} from {cnt} trades')

if args.fuzzy:
    # Collect all trade rows for fuzzy merging
    with open(trade_csv, newline='', encoding='utf-8') as f:
        reader = list(csv.DictReader(f, delimiter=';'))
    # Filter only non-out_of_cycle
    filtered = [r for r in reader if not (r.get('OutOfCycle') and int(r.get('OutOfCycle')))]
    merged_rows = fuzzy_merge_trades(filtered)
    print('\nFuzzy merged groups: original trades %d -> merged %d' % (len(filtered), len(merged_rows)))
    # Re-sum by bot/cycle using merged rows
    fuzzy_acc = defaultdict(lambda: defaultdict(float))
    fuzzy_counts = defaultdict(lambda: defaultdict(int))
    for r in merged_rows:
        try:
            b = int(r['BotID'])
            c = int(r['Cycle'])
            pnl = float(str(r['PnL']).replace(',', '.'))
        except Exception:
            continue
        fuzzy_acc[b][c] += pnl
        fuzzy_counts[b][c] += 1

    fuzzy_mism = []
    for bot, cycles in fuzzy_acc.items():
        for cycle, pnl_sum in cycles.items():
            expected = None
            if bot in bots_per_cycle_pnl and cycle < len(bots_per_cycle_pnl[bot]):
                expected = bots_per_cycle_pnl[bot][cycle]
            else:
                continue
            if abs(pnl_sum - expected) > 0.01:
                fuzzy_mism.append((bot, cycle, expected, pnl_sum, fuzzy_counts[bot][cycle]))

    print('\nMismatches after fuzzy deduplication: %d' % len(fuzzy_mism))
    for bot, cycle, exp, actual, cnt in fuzzy_mism[:args.top if args.top>0 else 200]:
        print(f'[FUZZY-DEDUPE] Bot {bot} cycle {cycle}: Expected {exp:.2f}, SumTrades {actual:.2f} from {cnt} trades')

# If no mismatches, output aggregate counts
if len(mismatches) == 0:
    print('No mismatches between per-trade sums and logged per-cycle PnL')

