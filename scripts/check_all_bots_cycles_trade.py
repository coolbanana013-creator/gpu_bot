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
parser.add_argument('--disable-filters', action='store_true', help='Disable quality filters for baseline testing (sets DEBUG_DISABLE_FILTERS=1)')
parser.add_argument('--bypass-sr', action='store_true', help='Bypass S/R checks during baseline testing (sets DEBUG_BYPASS_SR=1)')
parser.add_argument('--bypass-volume', action='store_true', help='Bypass volume checks during baseline testing (sets DEBUG_BYPASS_VOLUME=1)')
parser.add_argument('--force-signals', action='store_true', help='Force signals to evaluate trade pipeline (sets DEBUG_FORCE_SIGNALS=1)')
parser.add_argument('--mutate', action='store_true', help='Mutate CSV bots to be directional and unique using generator heuristics')
parser.add_argument('--generate', action='store_true', help='Generate a population with CompactBotGenerator instead of loading from CSV')
args = parser.parse_args()

gen_csv = Path('logs') / f'generation_{args.generation}.csv'

# Prepare GPU context early so we can generate bots on GPU when requested
ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)

# Create a list of bot cfgs either by loading the generation CSV or generating new ones
bots = []
if args.generate:
    # Lazy import to avoid extra compile costs if not generating
    from src.bot_generator.compact_generator import CompactBotGenerator
    pop = int(args.limit) if args.limit and args.limit > 0 else 100
    gen = CompactBotGenerator(gpu_context=ctx, gpu_queue=queue, population_size=pop, min_indicators=3, max_indicators=6, min_leverage=1, max_leverage=10, random_seed=42)
    bots = gen.generate_population()
else:
    if not gen_csv.exists():
        raise FileNotFoundError(gen_csv)
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

# Ensure debug bypass flags are set (if requested)
if args.disable_filters:
    os.environ['DEBUG_DISABLE_FILTERS'] = '1'
else:
    if 'DEBUG_DISABLE_FILTERS' in os.environ:
        del os.environ['DEBUG_DISABLE_FILTERS']
if args.bypass_sr:
    os.environ['DEBUG_BYPASS_SR'] = '1'
else:
    if 'DEBUG_BYPASS_SR' in os.environ:
        del os.environ['DEBUG_BYPASS_SR']
if args.bypass_volume:
    os.environ['DEBUG_BYPASS_VOLUME'] = '1'
else:
    if 'DEBUG_BYPASS_VOLUME' in os.environ:
        del os.environ['DEBUG_BYPASS_VOLUME']
if args.force_signals:
    os.environ['DEBUG_FORCE_SIGNALS'] = '1'
else:
    if 'DEBUG_FORCE_SIGNALS' in os.environ:
        del os.environ['DEBUG_FORCE_SIGNALS']
os.environ['ENABLE_TRADE_LOGS'] = '1'

backtester = CompactBacktester(gpu_context=ctx, gpu_queue=queue, initial_balance=10.0)

# Optionally mutate CSV-loaded bots to be directional and unique
if args.mutate and not args.generate:
    from src.bot_generator.compact_generator import CompactBotGenerator
    # Create a small generator for mutation operations
    gen = CompactBotGenerator(gpu_context=ctx, gpu_queue=queue, population_size=len(bots), min_indicators=3, max_indicators=8, min_leverage=1, max_leverage=10, random_seed=42)
    gen.clear_used_combinations()
    gen._enforce_directional_and_unique(bots)

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
    # If mutate option enabled, try to replace failing bots using generator heuristics
    if args.mutate:
        print('\nAttempting to refill failing bots with generated directional replacements...')
        from src.bot_generator.compact_generator import CompactBotGenerator
        gen = CompactBotGenerator(gpu_context=ctx, gpu_queue=queue, population_size=args.limit or 50, min_indicators=3, max_indicators=8, min_leverage=1, max_leverage=10, random_seed=42)
        gen.clear_used_combinations()
        attempts = 0
        max_attempts = 3
        while failures and attempts < max_attempts:
            n_fail = len(failures)
            print(f'Refill attempt {attempts + 1}: generating {n_fail} replacements')
            old_pop = gen.population_size
            gen.population_size = n_fail
            replacements = gen.generate_population()
            gen.population_size = old_pop
            # Backtest replacements only
            rep_results = backtester.backtest_bots(replacements, df, cycles)
            new_failures = []
            replace_map = {}
            for idx, r in enumerate(rep_results):
                if not all(n > 0 for n in r.per_cycle_trades):
                    new_failures.append((r.bot_id, r.per_cycle_trades))
                else:
                    # Find a failing bot id to replace
                    if idx < len(failures):
                        orig_bot_id = failures[idx][0]
                        # Replace in `bots` list
                        for bi, b in enumerate(bots):
                            if b.bot_id == orig_bot_id:
                                bots[bi] = replacements[idx]
                                replace_map[orig_bot_id] = replacements[idx].bot_id
                                break
            failures = new_failures
            attempts += 1
        if not failures:
            print('Refill successful - previously failing bots replaced with new directional unique bots.')
        else:
            print('Refill incomplete. Some bots still failed:', len(failures))
    if len(failures) > 20:
        print(f'  and {len(failures)-20} more...')
else:
    print('✅ All tested bots traded in all cycles with debug bypass flags.')

# Exit code: 0 if all bots have trades across cycles, 1 otherwise
if failures:
    sys.exit(1)
else:
    sys.exit(0)

# Print quick guidance
if failures:
    print('\nTip: Re-run failing bots individually with trace_mismatch.py or inspect generated logs for those BotIDs.')

print('Done')
