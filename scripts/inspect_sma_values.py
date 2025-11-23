#!/usr/bin/env python3
import numpy as np
from pathlib import Path
import pyopencl as cl
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_provider.fetcher import DataFetcher
from src.data_provider.loader import DataLoader

def kernel_sma(arr, period):
    n = len(arr)
    out = np.zeros(n, dtype=np.float32)
    if period <= 0:
        return out
    for i in range(n):
        if i < period - 1:
            out[i] = np.float32(0.0)
        else:
            s = np.float32(0.0)
            for j in range(i - period + 1, i + 1):
                s = np.float32(s + np.float32(arr[j]))
            out[i] = np.float32(s / np.float32(period))
    return out

if __name__ == '__main__':
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    fetcher = DataFetcher(exchange_type='futures')
    file_paths = fetcher.fetch_data_range(pair='BTC/USDT', timeframe='1m', total_days=1)
    loader = DataLoader(file_paths=file_paths, timeframe='1m', gpu_context=ctx, gpu_queue=queue)
    ohlcv_df = loader.load_all_data()
    closes = ohlcv_df['close'].values.astype(np.float32)

    for idx, period in [(3, 50), (4, 100), (5, 200)]:
        print(f'\n=== SMA period={period} (idx {idx}) ===')
        sma_cpu = kernel_sma(closes, period)
        
        # Compute GPU SMA using isolated kernel
        kernel_src = (ROOT / 'src' / 'gpu_kernels' / 'precompute_all_indicators.cl').read_text()
        wrapper = f'\n__kernel void iso(__global OHLCVBar *ohlcv, const int num_bars, __global float *out) {{ if (get_global_id(0) == 0) compute_sma(ohlcv, num_bars, {period}, &out[{idx} * num_bars]); }}'
        prg = cl.Program(ctx, kernel_src + '\n' + wrapper).build()
        ohlcv_flat = ohlcv_df[['open','high','low','close','volume']].values.astype(np.float32).flatten()
        mf = cl.mem_flags
        ohlcv_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ohlcv_flat)
        n_out_indicators = 110
        out_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=n_out_indicators * len(closes) * 4)
        prg.iso(queue, (1,), None, ohlcv_buf, np.int32(len(closes)), out_buf)
        queue.finish()
        out_flat = np.empty(n_out_indicators * len(closes), dtype=np.float32)
        cl.enqueue_copy(queue, out_flat, out_buf)
        out_2d = out_flat.reshape((n_out_indicators, len(closes)))
        gpu_sma = out_2d[idx]
        
        print(f'CPU SMA sample at {period}: {sma_cpu[period]}')
        print(f'GPU SMA sample at {period}: {gpu_sma[period]}')
        
        # Print comparison
        print(f'\nbar | close | cpu_sma | gpu_sma | diff')
        for i in range(period-2, min(period+10, len(closes))):
            diff = abs(gpu_sma[i] - sma_cpu[i])
            print(f'{i:3d} | {closes[i]:8.2f} | {sma_cpu[i]:10.4f} | {gpu_sma[i]:10.4f} | {diff:8.4f}')
