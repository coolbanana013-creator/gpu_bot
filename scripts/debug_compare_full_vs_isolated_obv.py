import numpy as np
import pyopencl as cl
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_provider.fetcher import DataFetcher
from src.data_provider.loader import DataLoader
from src.backtester.compact_simulator import CompactBacktester

# Kernel with compute_obv code (same as debug_compute_obv_kernel)
kernel_src = r"""
typedef struct { float open; float high; float low; float close; float volume; } OHLCVBar;

void compute_obv(__global OHLCVBar *ohlcv, int num_bars, __global float *out) {
    float obv = 0.0f;
    for (int bar = 0; bar < num_bars; bar++) {
        if (bar == 0) {
            out[bar] = 0.0f;
        } else {
            if (ohlcv[bar].close > ohlcv[bar-1].close) {
                obv += ohlcv[bar].volume;
            } else if (ohlcv[bar].close < ohlcv[bar-1].close) {
                obv -= ohlcv[bar].volume;
            }
            out[bar] = obv;
        }
    }
}

__kernel void obv_runner(__global const float* ohlcv_flat, int num_bars, __global float* out) {
    __global OHLCVBar *ohlcv = (__global OHLCVBar *) ohlcv_flat;
    if (get_global_id(0) == 0) {
        compute_obv(ohlcv, num_bars, out);
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

# Full precompute indicators using backtester
backtester = CompactBacktester(gpu_context=ctx, gpu_queue=queue)
indicators_buf = backtester._precompute_indicators(ohlcv[['open','high','low','close','volume']].values.astype(np.float32))
indicators_flat = np.empty(backtester.NUM_INDICATORS * num_bars, dtype=np.float32)
cl.enqueue_copy(queue, indicators_flat, indicators_buf)
queue.finish()
indicators_gpu_full = indicators_flat.reshape((backtester.NUM_INDICATORS, num_bars))

# Run isolated OBV kernel
mf = cl.mem_flags
prg = cl.Program(ctx, kernel_src).build()
obv_kernel = prg.obv_runner

ohlcv_flat = ohlcv[['open','high','low','close','volume']].values.astype(np.float32).flatten()
buf_ohlcv = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ohlcv_flat)
buf_out = cl.Buffer(ctx, mf.WRITE_ONLY, size=num_bars * 4)
obv_kernel(queue, (1,), None, buf_ohlcv, np.int32(num_bars), buf_out)
queue.finish()
obv_gpu_isolated = np.empty(num_bars, dtype=np.float32)
cl.enqueue_copy(queue, obv_gpu_isolated, buf_out)
queue.finish()

# Compare isolated OBV to full precompute OBV (indicator 36)
full = indicators_gpu_full[36].astype(np.float64)
iso = obv_gpu_isolated.astype(np.float64)

diffs = np.abs(full - iso)

print('Max difference between full precompute OBV and isolated OBV:', diffs.max())
print('Sample first 20 differences:')
for i in range(20):
    print(i, 'full:', full[i], 'iso:', iso[i], 'diff:', diffs[i])

# If differences exist, write full vs iso to CSV for the first 200 rows
import csv
p = Path('logs') / 'debug_compare_full_vs_isolated_obv.csv'
with open(p, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['idx', 'full_obv', 'isolated_obv', 'open', 'high', 'low', 'close', 'vol'])
    for i in range(200):
        writer.writerow([i, float(full[i]), float(iso[i]), float(ohlcv.iloc[i]['open']), float(ohlcv.iloc[i]['high']), float(ohlcv.iloc[i]['low']), float(ohlcv.iloc[i]['close']), float(ohlcv.iloc[i]['volume'])])
print('Dump written to', p)
