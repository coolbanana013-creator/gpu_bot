import numpy as np
import pyopencl as cl
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data_provider.fetcher import DataFetcher
from src.data_provider.loader import DataLoader
from src.backtester.compact_simulator import CompactBacktester

ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)
fetcher = DataFetcher(exchange_type='futures')
file_paths = fetcher.fetch_data_range(pair='BTC/USDT', timeframe='1m', total_days=1)
loader = DataLoader(file_paths=file_paths, timeframe='1m', gpu_context=ctx, gpu_queue=queue)
ohlcv = loader.load_all_data()
num_bars = len(ohlcv)

highs = ohlcv['high'].values
lows = ohlcv['low'].values
closes = ohlcv['close'].values
volumes = ohlcv['volume'].values

# compute ad using kernel's formula (GPU formula)
ad_custom = np.zeros_like(closes, dtype=np.float64)
ad = 0.0
for i in range(len(closes)):
    hl = highs[i] - lows[i]
    if hl < 1e-10:
        ad_custom[i] = ad
    else:
        clv = ((closes[i] - lows[i]) - (highs[i] - closes[i])) / hl
        ad += clv * volumes[i]
        ad_custom[i] = ad

# Precompute GPU
backtester = CompactBacktester(gpu_context=ctx, gpu_queue=queue)
indicators_buf = backtester._precompute_indicators(ohlcv[['open','high','low','close','volume']].values.astype(np.float32))
indicators_flat = np.empty(backtester.NUM_INDICATORS * num_bars, dtype=np.float32)
cl.enqueue_copy(queue, indicators_flat, indicators_buf)
queue.finish()
indicators_gpu = indicators_flat.reshape((backtester.NUM_INDICATORS, num_bars))

# Compare first 20
for i in range(20):
    print(i, 'GPU:', indicators_gpu[39,i], 'custom:', ad_custom[i], 'diff:', indicators_gpu[39,i]-ad_custom[i])
