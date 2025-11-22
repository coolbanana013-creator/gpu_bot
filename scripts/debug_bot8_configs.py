from pathlib import Path
import csv
import os
from src.bot_generator.compact_generator import CompactBotConfig
from src.data_provider.loader import DataLoader
from src.backtester.compact_simulator import CompactBacktester
import pyopencl as cl
import numpy as np

# load bot config for bot id 8
bot_id = 8
# load gen csv
with open('logs/generation_0.csv', newline='', encoding='utf-8') as f:
    rdr = csv.DictReader(f, delimiter=';')
    bot_row = None
    for r in rdr:
        if int(r['BotID']) == bot_id:
            bot_row = r
            break
if not bot_row:
    print('Bot not found')
    raise SystemExit(1)

# create CompactBotConfig from row
from src.utils.indicator_parser import parse_indices, parse_indicator_params
inds = parse_indices(bot_row.get('IndicatorsUsed') or '')
params = parse_indicator_params(bot_row.get('IndicatorParams') or '')
while len(inds) < 8: inds.append(0)
while len(params) < 8: params.append([0.0,0.0,0.0])

cfg = CompactBotConfig(
    bot_id=bot_id,
    num_indicators=sum(1 for i in inds if i!=0),
    indicator_indices=np.array(inds, dtype=np.uint8),
    indicator_params=np.array(params, dtype=np.float32),
    indicator_risk_strategies=np.array([0]*8, dtype=np.uint8),
    risk_param=0.05,
    tp_multiplier=1.0, sl_multiplier=1.0,
    leverage=int(float(bot_row.get('Leverage') or 1))
)

configs = [
    {'name': 'all_bypass', 'env': {'DEBUG_DISABLE_FILTERS': '1', 'DEBUG_BYPASS_SR':'1', 'DEBUG_BYPASS_VOLUME':'1', 'DEBUG_FORCE_SIGNALS':'0'}},
    {'name': 'quality_on_srvol_bypass', 'env': {'DEBUG_DISABLE_FILTERS': '0', 'DEBUG_BYPASS_SR':'1', 'DEBUG_BYPASS_VOLUME':'1', 'DEBUG_FORCE_SIGNALS':'0'}},
    {'name': 'quality_on_volume_on', 'env': {'DEBUG_DISABLE_FILTERS': '0', 'DEBUG_BYPASS_SR':'1', 'DEBUG_BYPASS_VOLUME':'0', 'DEBUG_FORCE_SIGNALS':'0'}},
    {'name': 'all_on', 'env': {'DEBUG_DISABLE_FILTERS': '0', 'DEBUG_BYPASS_SR':'0', 'DEBUG_BYPASS_VOLUME':'0', 'DEBUG_FORCE_SIGNALS':'0'}},
]

ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)
file_paths = sorted((Path('data') / 'BTC_USDT' / '1m').glob('*.parquet'))
loader = DataLoader(file_paths=file_paths, timeframe='1m', random_seed=42, gpu_context=ctx, gpu_queue=queue, use_gpu_processing=False)
df = loader.load_all_data()
cycles = loader.generate_cycle_ranges(5, 7)

for cfg_def in configs:
    os.environ.update(cfg_def['env'])
    # clear prior logs
    fb = Path('logs') / 'filter_debug.csv'
    if fb.exists():
        fb.unlink()
    tlog = Path('logs') / 'trade_logs.csv'
    if tlog.exists():
        tlog.unlink()
    backtester = CompactBacktester(gpu_context=ctx, gpu_queue=queue, initial_balance=10.0)
    res = backtester.backtest_bots([cfg], df, cycles)[0]
    print('Config', cfg_def['name'], 'trades:', res.per_cycle_trades)
    # read bits
    fbits = [0]*len(cycles)
    fb = Path('logs') / 'filter_debug.csv'
    if fb.exists():
        with open(fb,newline='') as f:
            rdr = csv.DictReader(f, delimiter=';')
            for row in rdr:
                if int(row['BotID'])==bot_id:
                    fbits[int(row['Cycle'])] |= int(row['FilterDebugBits'])
    print('Filter bits:', [hex(b) for b in fbits])

# Also try forcing signals
os.environ.update({'DEBUG_FORCE_SIGNALS':'1','DEBUG_DISABLE_FILTERS':'0','DEBUG_BYPASS_SR':'0','DEBUG_BYPASS_VOLUME':'0'})
fb = Path('logs') / 'filter_debug.csv'
if fb.exists():
    fb.unlink()
if tlog.exists():
    tlog.unlink()
backtester = CompactBacktester(gpu_context=ctx, gpu_queue=queue, initial_balance=10.0)
res = backtester.backtest_bots([cfg], df, cycles)[0]
print('Forced signals trades:', res.per_cycle_trades)
fbits = []
if fb.exists():
    with open(fb,newline='') as f:
        rdr = csv.DictReader(f, delimiter=';')
        for row in rdr:
            if int(row['BotID'])==bot_id:
                fbits.append((int(row['Cycle']), int(row['FilterDebugBits'])))
print('Force bits entries:', fbits)
