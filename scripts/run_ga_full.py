import os, argparse, glob
from pathlib import Path
import pyopencl as cl
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.data_provider.loader import DataLoader
from src.bot_generator.compact_generator import CompactBotGenerator
from src.backtester.compact_simulator import CompactBacktester
from src.ga.evolver_compact import GeneticAlgorithmEvolver

parser = argparse.ArgumentParser(description='Run full GA (compact) with GPU acceleration')
parser.add_argument('--population', type=int, default=1000, help='Population size (e.g., 1000)')
parser.add_argument('--generations', type=int, default=2, help='Number of generations to run')
parser.add_argument('--cycles', type=int, default=5, help='Cycles per generation')
parser.add_argument('--chunk-days', type=int, default=20, help='Data chunk days to reduce GPU memory per chunk (default 20)')
parser.add_argument('--timeframe', type=str, default='1m')
parser.add_argument('--pair', type=str, default='BTC_USDT')
parser.add_argument('--data-dir', type=str, default='data')
parser.add_argument('--trade-logs', action='store_true', help='Enable per-trade logging')
parser.add_argument('--trade-log-max', type=int, default=200000)
parser.add_argument('--prefer-winrate', action='store_true', help='Prefer survivors with higher win rate (favor WR in selection)')
args = parser.parse_args()

if args.trade_logs:
    os.environ['ENABLE_TRADE_LOGS'] = '1'
    os.environ['TRADE_LOG_MAX'] = str(args.trade_log_max)

# Create GPU context
ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)

# Build file list
file_list = sorted(list(Path(args.data_dir).glob(f"{args.pair}/{args.timeframe}/*.parquet")))
if not file_list:
    raise RuntimeError(f"No data found in {args.data_dir}/{args.pair}/{args.timeframe}/")

loader = DataLoader(file_paths=file_list, timeframe=args.timeframe, random_seed=42, gpu_context=ctx, gpu_queue=queue, use_gpu_processing=False)
print('Loading data...')
df = loader.load_all_data()
cycles = loader.generate_cycle_ranges(args.cycles, 7)
print('Cycles', cycles)

# Create bot generator and backtester
bot_gen = CompactBotGenerator(gpu_context=ctx, gpu_queue=queue, population_size=args.population, min_indicators=2, max_indicators=5, min_risk_strategies=1, max_risk_strategies=3, min_leverage=1, max_leverage=25)
backtester = CompactBacktester(gpu_context=ctx, gpu_queue=queue, initial_balance=10.0, data_chunk_days=args.chunk_days)

evolver = GeneticAlgorithmEvolver(bot_generator=bot_gen, backtester=backtester, pair=args.pair, timeframe=args.timeframe, gpu_context=ctx, gpu_queue=queue)

print('Starting evolution...')
evolver.run_evolution(
            num_generations=args.generations,
            ohlcv_data=df,
            cycles=cycles,
            initial_balance=10.0,
            prefer_win_rate=args.prefer_winrate
        )
print('Evolution complete; logs in logs/')
