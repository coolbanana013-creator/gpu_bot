import numpy as np
import pandas as pd
import pyopencl as cl
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import talib
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
backtester = CompactBacktester(gpu_context=ctx, gpu_queue=queue)
indicators_buf = backtester._precompute_indicators(ohlcv[['open','high','low','close','volume']].values.astype(np.float32))
indicators_flat = np.empty(backtester.NUM_INDICATORS * num_bars, dtype=np.float32)
cl.enqueue_copy(queue, indicators_flat, indicators_buf)
queue.finish()
indicators_gpu = indicators_flat.reshape((backtester.NUM_INDICATORS, num_bars))

highs = ohlcv['high'].values
lows = ohlcv['low'].values
closes = ohlcv['close'].values
volumes = ohlcv['volume'].values

ad_cpu = talib.AD(highs, lows, closes, volumes)

print('GPU AD first 10:', indicators_gpu[39,:10])
print('CPU AD first 10 :', ad_cpu[:10])
