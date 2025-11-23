import numpy as np
import pyopencl as cl
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data_provider.fetcher import DataFetcher
from src.data_provider.loader import DataLoader
from src.backtester.compact_simulator import CompactBacktester
import talib

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

open_sma = talib.SMA(ohlcv['open'].values, timeperiod=5)
high_sma = talib.SMA(ohlcv['high'].values, timeperiod=5)
low_sma = talib.SMA(ohlcv['low'].values, timeperiod=5)
close_sma = talib.SMA(ohlcv['close'].values, timeperiod=5)

print('GPU SMA(5) first 10 ', indicators_gpu[0,:10])
print('open_sma first 10', open_sma[:10])
print('high_sma first 10', high_sma[:10])
print('low_sma first 10', low_sma[:10])
print('close_sma first 10', close_sma[:10])
