import csv, argparse, os
from pathlib import Path
import numpy as np
import pyopencl as cl

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.bot_generator.compact_generator import CompactBotConfig
from src.utils.indicator_parser import parse_indices, parse_indicator_params
from src.backtester.compact_simulator import CompactBacktester
from src.data_provider.loader import DataLoader

parser = argparse.ArgumentParser(description='Trace mismatch for a single bot from generation logs')
parser.add_argument('--bot', type=int, required=True, help='BotID to trace')
parser.add_argument('--generation', type=int, default=0)
parser.add_argument('--pair', type=str, default='BTC_USDT')
parser.add_argument('--timeframe', type=str, default='1m')
parser.add_argument('--cycles', type=int, default=5)
args = parser.parse_args()

# Find the bot row in generation CSV
gen_csv = Path('logs') / f'generation_{args.generation}.csv'
if not gen_csv.exists():
    raise FileNotFoundError(gen_csv)
row=None
with open(gen_csv, newline='', encoding='utf-8') as f:
    reader=csv.DictReader(f, delimiter=';')
    for r in reader:
        if 'BotID' in r and int(r['BotID'])==args.bot:
            row=r; break
if row is None:
    print('Bot not found in generation', args.generation); raise SystemExit(1)

# Reconstruct bot
# WARNING: fields vary; use CompactBotConfig.from_dict if possible
from src.bot_generator.compact_generator import CompactBotGenerator

# compact format in generation CSV may differ; try to parse indicator fields
inds_str = row.get('IndicatorsUsed') or row.get('IndicatorsUsed', '')
params_str = row.get('IndicatorParams') or row.get('IndicatorParams', '')

# Minimal bot config for backtest use
from src.indicators.gpu_indicators import get_gpu_indicator_name

# fallback: generate random bot with saved config
try:
    # The generation log often includes a 'config' column - try to load
    if 'config' in row and row['config']:
        cfg = ast.literal_eval(row['config'])
        bot = CompactBotConfig.from_dict({'config':cfg, 'bot_id':int(row['BotID'])})
    else:
        # Best-effort parse of indicator indices and params
        ind_idxs = parse_indices(row.get('IndicatorIndices') or row.get('IndicatorsUsed') or '')
        ind_params = parse_indicator_params(row.get('IndicatorParams') or '')
        while len(ind_idxs) < 8:
            ind_idxs.append(0)
        while len(ind_params) < 8:
            ind_params.append([0.0, 0.0, 0.0])
        # risk strategies default
        risk_strat = [0]*8
        print('Parsed indicator indexes (first 8):', ind_idxs[:8])
        print('Parsed indicator params (first 8):', ind_params[:8])
        def parse_float_field(v, default=0.0):
            if not v:
                return default
            try:
                return float(str(v).replace(',', '.'))
            except Exception:
                return default

        risk_val = 0.05
        if row.get('RiskStrategies'):
            try:
                # Often like 'RISK(0,00)'
                risk_val = parse_float_field(row.get('RiskStrategies').split('(')[-1].strip(')'))
            except Exception:
                risk_val = parse_float_field(row.get('RiskStrategies'))

        bot = CompactBotConfig(bot_id=int(row['BotID']), num_indicators=sum(1 for i in ind_idxs if i!=0), indicator_indices=np.array(ind_idxs, dtype=np.uint8), indicator_params=np.array(ind_params, dtype=np.float32), indicator_risk_strategies=np.array(risk_strat, dtype=np.uint8), risk_param=risk_val, tp_multiplier=parse_float_field(row.get('TPMultiplier') or 1.0), sl_multiplier=parse_float_field(row.get('SLMultiplier') or 1.0), leverage=int(parse_float_field(row.get('Leverage', 1))))
except Exception as e:
    print('Failed to reconstruct bot config from CSV: ', e)
    raise

print('Trace bot:', bot.bot_id, 'ind count', bot.num_indicators, 'leverage', bot.leverage)

# Prepare data
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
data_dir = Path('data')/args.pair/args.timeframe
file_paths = sorted(data_dir.glob('*.parquet'))
loader = DataLoader(file_paths=file_paths, timeframe=args.timeframe, random_seed=42, gpu_context=ctx, gpu_queue=queue, use_gpu_processing=False)
df = loader.load_all_data()
cycles = loader.generate_cycle_ranges(args.cycles, 7)
print('Using cycles', cycles)

# Run single bot *with* trade logs (env var)
os.environ['ENABLE_TRADE_LOGS'] = '1'
os.environ['TRADE_LOG_MAX'] = '200000'
backtester = CompactBacktester(gpu_context=ctx, gpu_queue=queue, initial_balance=parse_float_field(row.get('FinalBalance', 10.0)))
# Remove old logs so analyzer compares only this run (optional for isolated reproduction)
trade_logs_path = Path('logs') / 'trade_logs.csv'
if trade_logs_path.exists():
    trade_logs_path.unlink()
results = backtester.backtest_bots([bot], df, cycles)
res = results[0]
print('Backtest per-cycle PnL:', res.per_cycle_pnl)

# Read trade logs and filter for this bot
trade_file = Path('logs')/'trade_logs.csv'
if trade_file.exists():
    with open(trade_file, newline='', encoding='utf-8') as f:
        reader=csv.DictReader(f, delimiter=';')
        trade_acc={}
        trades=[]
        for r in reader:
            if int(r['BotID'])==bot.bot_id:
                trades.append(r)
                c=int(r['Cycle']); pnl=float(r['PnL'].replace(',','.'))
                trade_acc[c]=trade_acc.get(c,0.0)+pnl
    print('Per-trade sums by cycle from logs:', trade_acc)
    print('Number of trades logged for this bot:', len(trades))
    # Count out-of-cycle flags per chunk
    chunk_counts = {}
    for t in trades:
        cid = int(t.get('ChunkID') or -1)
        oc = int(t.get('OutOfCycle') or 0)
        chunk_counts.setdefault(cid, {'total':0, 'out_of_cycle':0})
        chunk_counts[cid]['total'] += 1
        chunk_counts[cid]['out_of_cycle'] += oc
    print('Chunk counts (total,out_of_cycle):', chunk_counts)
    # Detect duplicate trade signatures across chunks for this bot
    sig_map = {}
    for t in trades:
        if int(t.get('OutOfCycle') or 0):
            continue
        sig = (int(t.get('EntryBar') or 0), int(t.get('ExitBar') or 0), t.get('Direction'))
        sig_map.setdefault(sig, set()).add(int(t.get('ChunkID') or -1))
    duplicates = {sig: ch for sig, ch in sig_map.items() if len(ch) > 1}
    print('Duplicate trade signatures across chunks (if any):', duplicates)
    # Save a detailed trace output for inspection
    out_dir = Path('analysis') / 'mismatch_trace_outputs'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f'trace_bot_{bot.bot_id}_gen_{args.generation}.csv'
    with open(out_file, 'w', newline='', encoding='utf-8') as of:
        w = csv.DictWriter(of, fieldnames=list(trades[0].keys()) if trades else ['BotID','Cycle','EntryBar','ExitBar','PnL'])
        w.writeheader()
        for t in trades:
            w.writerow(t)
    print('Saved trace output to', out_file)
    # Print trades out of cycle ranges
    for t in trades:
        c=int(t['Cycle']); eb=int(t['EntryBar']); xb=int(t['ExitBar'])
        start,end = cycles[c]
        if eb < start or xb > end:
            print('Trade outside expected cycle boundary:', t)
else:
    print('No trade log found')
