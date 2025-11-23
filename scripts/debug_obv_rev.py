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

# compute reversed obv using kernel's logic (accumulating from last to first)
obv_rev = np.zeros_like(volumes, dtype=np.float64)
obv_r = 0.0
for idx in range(num_bars-1, -1, -1):
    if idx == num_bars-1:
        obv_rev[idx] = 0.0
    else:
        if closes[idx] > closes[idx+1]:
            obv_r += volumes[idx]
        elif closes[idx] < closes[idx+1]:
            obv_r -= volumes[idx]
        obv_rev[idx] = obv_r

backtester = CompactBacktester(gpu_context=ctx, gpu_queue=queue)
indicators_buf = backtester._precompute_indicators(ohlcv[['open','high','low','close','volume']].values.astype(np.float32))
indicators_flat = np.empty(backtester.NUM_INDICATORS * num_bars, dtype=np.float32)
cl.enqueue_copy(queue, indicators_flat, indicators_buf)
queue.finish()
indicators_gpu = indicators_flat.reshape((backtester.NUM_INDICATORS, num_bars))

for i in range(20):
    print(i, 'GPU OBV:', indicators_gpu[36,i], 'custom rev:', obv_rev[i], 'diff GPU-custom rev:', indicators_gpu[36,i]-obv_rev[i])
