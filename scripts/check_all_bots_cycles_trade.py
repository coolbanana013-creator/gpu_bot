"""Check that all cycles have at least one trade for a selection of bots using debug bypass flags.

This script loads a generation CSV (by default generation_0.csv), constructs bot configs,
backtests them with DEBUG_DISABLE_FILTERS=1 (skip filters) and verifies each cycle has at least one trade.

Usage:
  $env:DEBUG_DISABLE_FILTERS='1'; $env:DEBUG_BYPASS_SR='1'; $env:DEBUG_BYPASS_VOLUME='1'; $env:ENABLE_TRADE_LOGS='1'; & .venv\Scripts\python.exe scripts/check_all_bots_cycles_trade.py --generation 0 --limit 50

"""
import os
import csv
import argparse
import sys
from pathlib import Path
import numpy as np
import pyopencl as cl

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

gen_csv = Path('logs') / f'generation_{args.generation}.csv'
if not gen_csv.exists():
    raise FileNotFoundError(gen_csv)

# Create a list of bot cfgs
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

# Ensure debug bypass flags are set
os.environ['DEBUG_DISABLE_FILTERS'] = '1'
os.environ['DEBUG_BYPASS_SR'] = '1'
os.environ['DEBUG_BYPASS_VOLUME'] = '1'
os.environ['ENABLE_TRADE_LOGS'] = '1'

backtester = CompactBacktester(gpu_context=ctx, gpu_queue=queue, initial_balance=10.0)

# Run in batches of a few bots to reduce memory usage
batch_size = 10
failures = []
for i in range(0, len(bots), batch_size):
    batch = bots[i:i+batch_size]
    print(f'Running batch {i // batch_size + 1} - bots {len(batch)}')
    results = backtester.backtest_bots(batch, df, cycles)
    for res in results:
        all_traded = all(n > 0 for n in res.per_cycle_trades)
        if not all_traded:
            failures.append((res.bot_id, res.per_cycle_trades))

print('=== Summary ===')
print(f'Tested {len(bots)} bots')
print(f'Failures (not all cycles traded): {len(failures)}')
if failures:
    for bot_id, per_cycle_trades in failures[:20]:
        print(f'  - Bot {bot_id}: per_cycle_trades={per_cycle_trades}')
    if len(failures) > 20:
        print(f'  and {len(failures)-20} more...')
else:
    print('✅ All tested bots traded in all cycles with debug bypass flags.')

# Print quick guidance
if failures:
    print('\nTip: Re-run failing bots individually with trace_mismatch.py or inspect generated logs for those BotIDs.')

print('Done')
