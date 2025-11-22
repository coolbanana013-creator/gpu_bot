"""Run filter re-enable sequence to locate which filters cause no-trade cycles.

Sequence tested (per-bot):
  1) All filters bypassed (DEBUG_DISABLE_FILTERS=1) [baseline]
  2) Disable all -> Enable quality filters, bypass SR & volume (DEBUG_DISABLE_FILTERS=0, BYPASS_SR=1, BYPASS_VOLUME=1)
  3) Re-enable volume (BYPASS_VOLUME=0)
  4) Re-enable SR (BYPASS_SR=0) -> final state

This script tests a set of bots and returns which config causes zero trades per cycle.

Usage:
  $env:PYTHONPATH='.'; & .venv\Scripts\python.exe scripts/run_filter_reenable.py --generation 0 --limit 100 --cycles 5
"""
import os
import csv
import argparse
import sys
from pathlib import Path
import numpy as np
import pyopencl as cl
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bot_generator.compact_generator import CompactBotConfig
from src.data_provider.loader import DataLoader
from src.backtester.compact_simulator import CompactBacktester
from src.utils.indicator_parser import parse_indices, parse_indicator_params

parser = argparse.ArgumentParser()
parser.add_argument('--generation', type=int, default=0)
parser.add_argument('--limit', type=int, default=50, help='Limit number of bots to test (0=all)')
parser.add_argument('--pair', type=str, default='BTC_USDT')
parser.add_argument('--timeframe', type=str, default='1m')
parser.add_argument('--cycles', type=int, default=5)
args = parser.parse_args()

# Prepare generation CSV
gen_csv = Path('logs') / f'generation_{args.generation}.csv'
if not gen_csv.exists():
    raise FileNotFoundError(gen_csv)

# Build bot configs list
bots = []
with open(gen_csv, newline='', encoding='utf-8') as f:
    rdr = csv.DictReader(f, delimiter=';')
    for r in rdr:
        if not r.get('BotID'):
            continue
        bot_id = int(r['BotID'])
        inds = parse_indices(r.get('IndicatorsUsed') or r.get('IndicatorIndices') or '')
        iparams = parse_indicator_params(r.get('IndicatorParams') or '')
        while len(inds) < 8:
            inds.append(0)
        while len(iparams) < 8:
            iparams.append([0.0, 0.0, 0.0])
        cfg = CompactBotConfig(
            bot_id=bot_id,
            num_indicators=sum(1 for i in inds if i != 0),
            indicator_indices=np.array(inds, dtype=np.uint8),
            indicator_params=np.array(iparams, dtype=np.float32),
            indicator_risk_strategies=np.array([0] * 8, dtype=np.uint8),
            risk_param=float(r.get('RiskStrategies', '0').split('(')[1].split(')')[0]) if r.get('RiskStrategies') and '(' in r.get('RiskStrategies') else 0.05,
            tp_multiplier=float(str(r.get('TPMultiplier') or '1.0').replace(',', '.')), 
            sl_multiplier=float(str(r.get('SLMultiplier') or '1.0').replace(',', '.')), 
            leverage=int(float(str(r.get('Leverage') or '1').replace(',', '.')))
        )
        bots.append(cfg)
        if args.limit and len(bots) >= args.limit:
            break

if not bots:
    print('No bots found in generation file, abort')
    raise SystemExit(1)

# Setup GPU and data
ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)
file_paths = sorted((Path('data')/args.pair/args.timeframe).glob('*.parquet'))
loader = DataLoader(file_paths=file_paths, timeframe=args.timeframe, random_seed=42, gpu_context=ctx, gpu_queue=queue, use_gpu_processing=False)
df = loader.load_all_data()
cycles = loader.generate_cycle_ranges(args.cycles, 7)

# Sequence configs to test
configs = [
    {'name': 'all_bypass', 'env': {'DEBUG_DISABLE_FILTERS': '1', 'DEBUG_BYPASS_SR':'1', 'DEBUG_BYPASS_VOLUME':'1', 'DEBUG_FORCE_SIGNALS':'0'}},
    {'name': 'quality_on_srvol_bypass', 'env': {'DEBUG_DISABLE_FILTERS': '0', 'DEBUG_BYPASS_SR':'1', 'DEBUG_BYPASS_VOLUME':'1', 'DEBUG_FORCE_SIGNALS':'0'}},
    {'name': 'quality_on_volume_on', 'env': {'DEBUG_DISABLE_FILTERS': '0', 'DEBUG_BYPASS_SR':'1', 'DEBUG_BYPASS_VOLUME':'0', 'DEBUG_FORCE_SIGNALS':'0'}},
    {'name': 'all_on', 'env': {'DEBUG_DISABLE_FILTERS': '0', 'DEBUG_BYPASS_SR':'0', 'DEBUG_BYPASS_VOLUME':'0', 'DEBUG_FORCE_SIGNALS':'0'}},
]

output_rows = []
# For each bot, test the configs and collect per-cycle trades
for bot_cfg in bots:
    result_per_config = {}
    for cfg in configs:
        # Apply env vars for this run
        os.environ.update(cfg['env'])
        os.environ['ENABLE_TRADE_LOGS'] = '1'
        # Force non-interactive context
        backtester = CompactBacktester(gpu_context=ctx, gpu_queue=queue, initial_balance=10.0)
        # Remove old logs
        tpath = Path('logs') / 'trade_logs.csv'
        if tpath.exists():
            tpath.unlink()
        # Remove old filter debug logs to avoid stale data
        fbpath = Path('logs') / 'filter_debug.csv'
        if fbpath.exists():
            fbpath.unlink()
        results = backtester.backtest_bots([bot_cfg], df, cycles)
        res = results[0]
        # Read filter debug per-cycle bits from logs (if present)
        filter_bits_per_cycle = [0] * args.cycles
        fb_path = Path('logs') / 'filter_debug.csv'
        if fb_path.exists():
            with open(fb_path, newline='') as f:
                rdr = csv.DictReader(f, delimiter=';')
                for r2 in rdr:
                    try:
                        b_id = int(r2['BotID'])
                        cycle_idx = int(r2['Cycle'])
                        bits = int(r2['FilterDebugBits'])
                    except Exception:
                        continue
                    if b_id == bot_cfg.bot_id and 0 <= cycle_idx < args.cycles:
                        filter_bits_per_cycle[cycle_idx] |= bits
        # Save results per cycle
        result_per_config[cfg['name']] = res.per_cycle_trades
        result_per_config[cfg['name'] + '_filter_bits'] = filter_bits_per_cycle
    # Evaluate which config first gives all cycles > 0
    row = {
        'bot_id': bot_cfg.bot_id,
    }
    for cfg in configs:
        row[cfg['name']] = '|'.join([str(int(x)) for x in result_per_config[cfg['name']]])
        row[cfg['name'] + '_filter_bits'] = '|'.join([hex(int(x)) for x in result_per_config[cfg['name'] + '_filter_bits']])
    output_rows.append(row)

# Write summary CSV
out_path = Path('logs') / 'filter_reenable_summary.csv'
with open(out_path, 'w', newline='', encoding='utf-8') as f:
    fieldnames = ['bot_id'] + [c['name'] for c in configs] + [c['name'] + '_filter_bits' for c in configs]
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
    writer.writeheader()
    for r in output_rows:
        writer.writerow(r)

print('Saved summary to', out_path)

# Print problematic bots where 'all_on' config still had 0s
fails = [r for r in output_rows if '0' in r['all_on'].split('|')]
print('Total tested:', len(output_rows))
print('Failures with all filters enabled:', len(fails))
if fails:
    print('Sample failures:')
    for f in fails[:20]:
        print(f"  - Bot {f['bot_id']}: all_on={f['all_on']}")

print('Done')
# Summarize filter bit frequency per config
from collections import Counter
filter_counters = {c['name']: Counter() for c in configs}
fb_path = Path('logs') / 'filter_debug.csv'
if fb_path.exists():
    with open(fb_path, newline='') as f:
        rdr = csv.DictReader(f, delimiter=';')
        for row in rdr:
            try:
                b_id = int(row['BotID'])
                cycle_idx = int(row['Cycle'])
                bits = int(row['FilterDebugBits'])
            except Exception:
                continue
            # Determine which config run this corresponds to: last run appended to file will be the last config
            # We cannot map per-config easily here. Instead, count total occurrences of each bit across all runs
            for cfg in configs:
                filter_counters[cfg['name']][bits] += 1

print('\nFilter debug bit summary across runs:')
for cfg in configs:
    print(f"  - {cfg['name']}: {len(filter_counters[cfg['name']])} distinct bitmasks (sample: {filter_counters[cfg['name']].most_common(5)})")
