import numpy as np
import pyopencl as cl
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data_provider.fetcher import DataFetcher
from src.data_provider.loader import DataLoader

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
    // Map flat to struct-like array using 5 floats per bar
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

ohlcv_flat = ohlcv[['open','high','low','close','volume']].values.astype(np.float32).flatten()

mf = cl.mem_flags
prg = cl.Program(ctx, kernel_src).build()
obv_kernel = prg.obv_runner

buf_ohlcv = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ohlcv_flat)
buf_out = cl.Buffer(ctx, mf.WRITE_ONLY, size=num_bars * 4)

obv_kernel(queue, (1,), None, buf_ohlcv, np.int32(num_bars), buf_out)
queue.finish()

obv_gpu = np.empty(num_bars, dtype=np.float32)
cl.enqueue_copy(queue, obv_gpu, buf_out)
queue.finish()

# Compute OBV on CPU using same logic
highs = ohlcv['high'].values
lows = ohlcv['low'].values
closes = ohlcv['close'].values
volumes = ohlcv['volume'].values

obv_cpu = np.zeros_like(volumes, dtype=np.float64)
cur = 0.0
for bar in range(num_bars):
    if bar == 0:
        obv_cpu[bar] = 0.0
    else:
        if closes[bar] > closes[bar-1]:
            cur += volumes[bar]
        elif closes[bar] < closes[bar-1]:
            cur -= volumes[bar]
        obv_cpu[bar] = cur

# Compare first 20
for i in range(20):
    print(i, 'GPU OBV:', obv_gpu[i], 'CPU OBV:', obv_cpu[i], 'volume:', volumes[i], 'close', closes[i])

# Show any mismatches
mism = np.where(~np.isclose(obv_gpu.astype(np.float64), obv_cpu, atol=1e-6))[0]
print('Num mismatches:', len(mism))
if len(mism) > 0:
    for mi in mism[:20]:
        print('idx', mi, 'gpu', obv_gpu[mi], 'cpu', obv_cpu[mi], 'vol', volumes[mi])

# Save to CSV for analysis
import csv
p = Path('logs') / 'debug_obv_kernel.csv'
p.parent.mkdir(parents=True, exist_ok=True)
with open(p, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['idx', 'gpu_obv', 'cpu_obv', 'vol', 'close'])
    for i in range(num_bars):
        writer.writerow([i, float(obv_gpu[i]), float(obv_cpu[i]), float(volumes[i]), float(closes[i])])
print('Dump written to', p)
