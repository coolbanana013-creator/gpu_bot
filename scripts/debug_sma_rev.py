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

closes = ohlcv['close'].values
close_sma = talib.SMA(closes, timeperiod=5)
rev_close_sma = talib.SMA(closes[::-1], timeperiod=5)[::-1]

print('close_sma[:10]', close_sma[:10])
print('rev_close_sma[:10]', rev_close_sma[:10])
print('gpu sma[:10]', indicators_gpu[0,:10])
print('diff close vs gpu', close_sma[:10] - indicators_gpu[0,:10])
print('diff rev_close vs gpu', rev_close_sma[:10] - indicators_gpu[0,:10])
