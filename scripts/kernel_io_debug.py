import numpy as np
import pyopencl as cl
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_provider.fetcher import DataFetcher
from src.data_provider.loader import DataLoader

# Simple debug kernel that copies flattened OHLCV (rows of 5 floats) into separate output arrays
kernel_src = r"""
__kernel void dump_ohlcv(__global const float* ohlcv_flat, int num_bars, __global float* open_out, __global float* high_out, __global float* low_out, __global float* close_out, __global float* vol_out) {
    int gid = get_global_id(0);
    if (gid >= num_bars) return;
    int idx = gid * 5;
    open_out[gid] = ohlcv_flat[idx + 0];
    high_out[gid] = ohlcv_flat[idx + 1];
    low_out[gid] = ohlcv_flat[idx + 2];
    close_out[gid] = ohlcv_flat[idx + 3];
    vol_out[gid] = ohlcv_flat[idx + 4];
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
dump_kernel = prg.dump_ohlcv

open_out = np.empty(num_bars, dtype=np.float32)
high_out = np.empty(num_bars, dtype=np.float32)
low_out = np.empty(num_bars, dtype=np.float32)
close_out = np.empty(num_bars, dtype=np.float32)
vol_out = np.empty(num_bars, dtype=np.float32)

# GPU bufs
buf_ohlcv = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ohlcv_flat)
buf_open = cl.Buffer(ctx, mf.WRITE_ONLY, size=open_out.nbytes)
buf_high = cl.Buffer(ctx, mf.WRITE_ONLY, size=high_out.nbytes)
buf_low = cl.Buffer(ctx, mf.WRITE_ONLY, size=low_out.nbytes)
buf_close = cl.Buffer(ctx, mf.WRITE_ONLY, size=close_out.nbytes)
buf_vol = cl.Buffer(ctx, mf.WRITE_ONLY, size=vol_out.nbytes)

# Run kernel
work_size = (num_bars,)
dump_kernel(queue, work_size, None, buf_ohlcv, np.int32(num_bars), buf_open, buf_high, buf_low, buf_close, buf_vol)
queue.finish()

cl.enqueue_copy(queue, open_out, buf_open)
cl.enqueue_copy(queue, high_out, buf_high)
cl.enqueue_copy(queue, low_out, buf_low)
cl.enqueue_copy(queue, close_out, buf_close)
cl.enqueue_copy(queue, vol_out, buf_vol)
queue.finish()

# Compare with host arrays
open_host = ohlcv['open'].values.astype(np.float32)
high_host = ohlcv['high'].values.astype(np.float32)
low_host = ohlcv['low'].values.astype(np.float32)
close_host = ohlcv['close'].values.astype(np.float32)
vol_host = ohlcv['volume'].values.astype(np.float32)

mismatches = []
for i in range(num_bars):
    if not (np.isclose(open_out[i], open_host[i], atol=1e-6) and np.isclose(high_out[i], high_host[i], atol=1e-6) and np.isclose(low_out[i], low_host[i], atol=1e-6) and np.isclose(close_out[i], close_host[i], atol=1e-6) and np.isclose(vol_out[i], vol_host[i], atol=1e-6)):
        mismatches.append((i, open_out[i], open_host[i], high_out[i], high_host[i], low_out[i], low_host[i], close_out[i], close_host[i], vol_out[i], vol_host[i]))

print('Total mismatches:', len(mismatches))
if len(mismatches) > 0:
    print('Example mismatch at index 0:', mismatches[0])
    # Output first 10 mismatches
    for m in mismatches[:10]:
        i, o_out, o_host, h_out, h_host, l_out, l_host, cl_out, cl_host, v_out, v_host = m
        print(f'idx={i}: open_gpu={o_out} open_host={o_host} | high_gpu={h_out} high_host={h_host} | low_gpu={l_out} low_host={l_host} | close_gpu={cl_out} close_host={cl_host} | vol_gpu={v_out} vol_host={v_host}')
else:
    print('No mismatches found — kernel sees OHLCV the same as host')

# Save mismatches if any to CSV
import csv
p = Path('logs') / 'kernel_io_debug.csv'
p.parent.mkdir(parents=True, exist_ok=True)
with open(p, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['idx', 'open_gpu', 'open_host', 'high_gpu', 'high_host', 'low_gpu', 'low_host', 'close_gpu', 'close_host', 'vol_gpu', 'vol_host'])
    for m in mismatches:
        writer.writerow(m)

print('Dump written to', p)
