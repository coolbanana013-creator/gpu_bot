import os, csv, glob, sys
from pathlib import Path
repo_root = str(Path(__file__).resolve().parents[1])
sys.path.append(repo_root)
from pathlib import Path
import numpy as np
import pyopencl as cl
from src.bot_generator.compact_generator import CompactBotConfig
from src.data_provider.loader import DataLoader
from src.backtester.compact_simulator import CompactBacktester

# enable local env vars
os.environ['ENABLE_TRADE_LOGS'] = '1'
os.environ['TRADE_LOG_MAX'] = '50000'

row=None
with open('logs/generation_0.csv', newline='', encoding='utf-8') as f:
    for r in csv.DictReader(f, delimiter=';'):
        if r['BotID']=='98': row=r; break
if not row:
    print('Bot not found'); raise SystemExit(1)

# Reconstruct indicators and params
params=[p.strip() for p in row['IndicatorParams'].split('|')]
inds=[]
iparams=[]
from src.indicators.gpu_indicators import get_gpu_indicator_name
from src.utils.indicator_parser import parse_indicator_params

for p in params:
    name=p.split('(')[0].strip(); idx=None
    for i in range(50):
        if get_gpu_indicator_name(i).split('(')[0].lower().startswith(name.lower().split('(')[0]): idx=i; break
    if idx is None: continue
    inds.append(idx)
    if '(' in p and ')' in p:
        inside=p[p.rfind('(')+1:p.rfind(')')]
        try:
            pars=[float(x) for x in inside.split(',')]
        except Exception:
            # Fallback to robust parser
            parsed = parse_indicator_params(p)
            pars = parsed[0] if parsed else [0.0, 0.0, 0.0]
    else: pars=[0.0,0.0,0.0]
    while len(pars)<3: pars.append(0.0)
    iparams.append(pars[:3])
while len(inds)<8: inds.append(0)
while len(iparams)<8: iparams.append([0.0,0.0,0.0])
try: risk=float(row['RiskStrategies'].split('(')[1].split(')')[0])
except Exception: risk=0.05

bot=CompactBotConfig(bot_id=98, num_indicators=sum(1 for i in inds if i!=0), indicator_indices=np.array(inds,dtype=np.uint8), indicator_params=np.array(iparams,dtype=np.float32), indicator_risk_strategies=np.array([0]*8,dtype=np.uint8), risk_param=risk, tp_multiplier=float(row['TPMultiplier'].replace(',','.')) if row.get('TPMultiplier') else 1.0, sl_multiplier=float(row['SLMultiplier'].replace(',','.')) if row.get('SLMultiplier') else 1.0, leverage=int(float(row['Leverage'])), survival_generations=0)

print('Bot loaded', bot.bot_id, 'ind_count', bot.num_indicators, 'leverage', bot.leverage)

# Load local parquet files
file_paths = sorted(Path(p) for p in glob.glob(str(Path('data')/'BTC_USDT'/'1m'/'*.parquet')))
ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)
loader = DataLoader(file_paths=file_paths, timeframe='1m', random_seed=42, gpu_context=ctx, gpu_queue=queue, use_gpu_processing=False)

ohlcv = loader.load_all_data()
cycles = loader.generate_cycle_ranges(5, 7)
print('Cycles', cycles)

# Match initial balance used by GA (very low) to reproduce generation logs
try:
    backtester = CompactBacktester(gpu_context=ctx, gpu_queue=queue, initial_balance=10.0)
except Exception:
    import traceback; traceback.print_exc(); raise

# Remove old logs so analyzer compares only this run (optional for isolated reproduction)
trade_logs_path = Path('logs') / 'trade_logs.csv'
if trade_logs_path.exists():
    trade_logs_path.unlink()
try:
    results = backtester.backtest_bots([bot], ohlcv, cycles)
    res = results[0]
    print('Bot per_cycle_pnl', res.per_cycle_pnl)
    print('Total trades', res.total_trades)
    print('Final balance', res.final_balance)
except Exception:
    import traceback; traceback.print_exc(); raise
