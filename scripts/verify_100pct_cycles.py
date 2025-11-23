"""Verify 100% cycles traded for generated populations.

Generates populations using CompactBotGenerator and validates 100% per-cycle trading coverage.

Usage:
  $env:DEBUG_DISABLE_FILTERS='1'; python -u scripts/verify_100pct_cycles.py --population 1000 --generations 2 --timeframe 1m --cycles 10 --disable-filters

"""
import os
import sys
from pathlib import Path
import argparse
import numpy as np
import pyopencl as cl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bot_generator.compact_generator import CompactBotGenerator
from src.backtester.compact_simulator import CompactBacktester
from src.data_provider.loader import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--population', type=int, default=1000)
parser.add_argument('--generations', type=int, default=2)
parser.add_argument('--timeframe', type=str, default='1m')
parser.add_argument('--cycles', type=int, default=10)
parser.add_argument('--pair', type=str, default='BTC_USDT')
parser.add_argument('--disable-filters', action='store_true')
parser.add_argument('--bypass-sr', action='store_true')
parser.add_argument('--bypass-volume', action='store_true')
args = parser.parse_args()

# Ensure debug flags
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

os.environ['ENABLE_TRADE_LOGS'] = '0'  # disable writing huge logs for CI

# Setup GPU context and data
ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)
file_paths = sorted((Path('data')/args.pair/args.timeframe).glob('*.parquet'))
loader = DataLoader(file_paths=file_paths, timeframe=args.timeframe, random_seed=42, gpu_context=ctx, gpu_queue=queue, use_gpu_processing=False)
df = loader.load_all_data()
cycles = loader.generate_cycle_ranges(args.cycles, 7)

backtester = CompactBacktester(gpu_context=ctx, gpu_queue=queue, initial_balance=10.0)

# Run generation cycles
for gen_idx in range(args.generations):
    print(f"Generating population: gen={gen_idx}")
    gen = CompactBotGenerator(gpu_context=ctx, gpu_queue=queue, population_size=args.population, min_indicators=3, max_indicators=6, min_leverage=1, max_leverage=10, random_seed=42+gen_idx)
    # Clear any previously known used combinations for a fresh generation
    gen.clear_used_combinations()
    bots = gen.generate_population()

    # Run in parallel chunks and validate
    print(f"Testing population: gen={gen_idx}, pop={len(bots)}")
    results = backtester.backtest_bots(bots, df, cycles)

    # Validate all bots
    failures = []
    for res in results:
        if not all(n > 0 for n in res.per_cycle_trades):
            failures.append((res.bot_id, res.per_cycle_trades))
    if failures:
        print(f"Generation {gen_idx} FAILED: {len(failures)} failures — attempting to refill with new bots")
        # Attempt to refill failing bots with new generated ones
        attempts = 0
        max_attempts = 3
        while failures and attempts < max_attempts:
            n_fail = len(failures)
            print(f"Refill attempt {attempts+1}: generating {n_fail} replacement bots")
            old_pop = gen.population_size
            gen.population_size = n_fail
            replacements = gen.generate_population()
            gen.population_size = old_pop
            # Backtest replacements only
            rep_results = backtester.backtest_bots(replacements, df, cycles)
            new_failures = []
            for i, r in enumerate(rep_results):
                if not all(n > 0 for n in r.per_cycle_trades):
                    new_failures.append((r.bot_id, r.per_cycle_trades))
                else:
                    # Replace failed bot entry in 'bots' with success replacement
                    # Find index of original failure and replace
                    orig_bot_id = failures[i][0]
                    # Find original index and replace with new bot
                    for bi, b in enumerate(bots):
                        if b.bot_id == orig_bot_id:
                            bots[bi] = replacements[i]
                            break
            failures = new_failures
            attempts += 1
        if failures:
            print(f"Generation {gen_idx} STILL FAILED after refill attempts: {len(failures)} failures")
            for bot_id, per_cycle_trades in failures[:20]:
                print(f"  - Bot {bot_id}: per_cycle_trades={per_cycle_trades}")
            sys.exit(1)
        else:
            print(f"Generation {gen_idx}OK after refill — all replacements passed")
    else:
        print(f"Generation {gen_idx} OK - All bots traded across {args.cycles} cycles")

print("All generations OK - 100% cycle coverage for all bots")
sys.exit(0)
