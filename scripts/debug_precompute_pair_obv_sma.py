import numpy as np
import pyopencl as cl
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data_provider.fetcher import DataFetcher
from src.data_provider.loader import DataLoader

kernel_src = r"""
typedef struct { float open; float high; float low; float close; float volume; } OHLCVBar;

float compute_sma_helper(__global OHLCVBar *ohlcv, int bar, int period) {
    if (bar < period - 1) return 0.0f;
    float sum = 0.0f;
    for (int i = 0; i < period; i++) {
        sum += ohlcv[bar - i].close;
    }
    return sum / (float)period;
}

void compute_obv(__global OHLCVBar *ohlcv, int num_bars, __global float *out) {
    float obv = 0.0f;
    for (int bar = 0; bar < num_bars; bar++) {
        if (bar == 0) { out[bar] = 0.0f; }
        else {
            if (ohlcv[bar].close > ohlcv[bar-1].close) obv += ohlcv[bar].volume;
            else if (ohlcv[bar].close < ohlcv[bar-1].close) obv -= ohlcv[bar].volume;
            out[bar] = obv;
        }
    }
}

__kernel void pair_kernel(__global const float* ohlcv_flat, int num_bars, __global float* out_sma, __global float* out_obv) {
    __global OHLCVBar *ohlcv = (__global OHLCVBar *) ohlcv_flat;
    int work_item_id = get_global_id(0);
    int work_items = get_global_size(0);
    int bars_per_item = (num_bars + work_items - 1) / work_items;
    int start_bar = work_item_id * bars_per_item;
    int end_bar = min(start_bar + bars_per_item, num_bars);

    // Compute SMA(5) in parallel blocks
    for (int bar = start_bar; bar < end_bar; bar++) {
        out_sma[bar] = compute_sma_helper(ohlcv, bar, 5);
    }

    // Compute OBV sequentially using work_item_id==0
    if (work_item_id == 0) {
        compute_obv(ohlcv, num_bars, out_obv);
    }
}
"""

ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)

fetcher = DataFetcher(exchange_type='futures')
file_paths = fetcher.fetch_data_range(pair='BTC/USDT', timeframe='1m', total_days=1)
loader = DataLoader(file_paths=file_paths, timeframe='1m', gpu_context=ctx, gpu_queue=queue)
ohlcv = loader.load_all_data()
num_bars = len(ohlcv)

ohlcv_flat = ohlcv[['open','high','low','close','volume']].values.astype(np.float32).flatten()

mf = cl.mem_flags
prg = cl.Program(ctx, kernel_src).build()

# outputs
sma_out = np.zeros(num_bars, dtype=np.float32)
obv_out = np.zeros(num_bars, dtype=np.float32)

buf_ohlcv = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ohlcv_flat)
buf_sma = cl.Buffer(ctx, mf.WRITE_ONLY, size=sma_out.nbytes)
buf_obv = cl.Buffer(ctx, mf.WRITE_ONLY, size=obv_out.nbytes)

work_size = (64,) # 64 work-items
prg.pair_kernel(queue, work_size, None, buf_ohlcv, np.int32(num_bars), buf_sma, buf_obv)
queue.finish()

cl.enqueue_copy(queue, sma_out, buf_sma)
cl.enqueue_copy(queue, obv_out, buf_obv)
queue.finish()

# Compare to CPU
open_host = ohlcv['open'].values
close_host = ohlcv['close'].values
closes = ohlcv['close'].values
import talib
sma_cpu = talib.SMA(closes, timeperiod=5)

# CPU obv
obv_cpu = np.zeros_like(closes)
cur = 0.0
for i in range(num_bars):
    if i == 0:
        obv_cpu[i] = 0.0
    else:
        if closes[i] > closes[i-1]: cur += ohlcv.iloc[i]['volume']
        elif closes[i] < closes[i-1]: cur -= ohlcv.iloc[i]['volume']
        obv_cpu[i] = cur

# Print first 20 compar
for i in range(20):
    print(i, 'GPU pair sma', sma_out[i], 'cpu sma', sma_cpu[i], 'GPU obv', obv_out[i], 'cpu obv', obv_cpu[i])

# Save mismatches
import csv
p = Path('logs') / 'debug_pair_obv_sma.csv'
with open(p, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['idx','gpu_sma','cpu_sma','gpu_obv','cpu_obv','close','vol'])
    for i in range(num_bars):
        w.writerow([i, float(sma_out[i]), float(sma_cpu[i]), float(obv_out[i]), float(obv_cpu[i]), float(ohlcv.iloc[i]['close']), float(ohlcv.iloc[i]['volume'])])
print('Dump written to', p)
