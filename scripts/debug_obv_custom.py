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

# compute obv using kernel's logic
obv_custom = np.zeros_like(volumes, dtype=np.float64)
obv = 0.0
for i in range(len(closes)):
    if i == 0:
        obv_custom[i] = 0.0
    else:
        if closes[i] > closes[i-1]:
            obv += volumes[i]
        elif closes[i] < closes[i-1]:
            obv -= volumes[i]
        obv_custom[i] = obv

# Precompute GPU
backtester = CompactBacktester(gpu_context=ctx, gpu_queue=queue)
indicators_buf = backtester._precompute_indicators(ohlcv[['open','high','low','close','volume']].values.astype(np.float32))
indicators_flat = np.empty(backtester.NUM_INDICATORS * num_bars, dtype=np.float32)
cl.enqueue_copy(queue, indicators_flat, indicators_buf)
queue.finish()
indicators_gpu = indicators_flat.reshape((backtester.NUM_INDICATORS, num_bars))

# Compute TA-Lib OBV
import talib
obv_ta = talib.OBV(closes, volumes)

for i in range(20):
    print(i, 'GPU OBV:', indicators_gpu[36,i], 'custom:', obv_custom[i], 'ta:', obv_ta[i], 'diff GPU-custom:', indicators_gpu[36,i]-obv_custom[i])
