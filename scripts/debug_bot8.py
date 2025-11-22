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

# set force signals env var
os.environ['DEBUG_FORCE_SIGNALS'] = '1'

ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)
file_paths = sorted((Path('data') / 'BTC_USDT' / '1m').glob('*.parquet'))
loader = DataLoader(file_paths=file_paths, timeframe='1m', random_seed=42, gpu_context=ctx, gpu_queue=queue, use_gpu_processing=False)
df = loader.load_all_data()
cycles = loader.generate_cycle_ranges(5, 7)

backtester = CompactBacktester(gpu_context=ctx, gpu_queue=queue, initial_balance=10.0)
res = backtester.backtest_bots([cfg], df, cycles)[0]
print('Per cycle trades:', res.per_cycle_trades)
# print saved filter bits
from pathlib import Path
fb = Path('logs') / 'filter_debug.csv'
if fb.exists():
    import csv
    with open(fb,newline='') as f:
        r=csv.DictReader(f, delimiter=';')
        for row in r:
            if int(row['BotID'])==bot_id:
                print('Filter debug:', row)
