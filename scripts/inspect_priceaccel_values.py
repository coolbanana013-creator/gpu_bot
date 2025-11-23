#!/usr/bin/env python3
import numpy as np
from pathlib import Path
import pyopencl as cl
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_provider.fetcher import DataFetcher
from src.data_provider.loader import DataLoader

if __name__ == '__main__':
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    fetcher = DataFetcher(exchange_type='futures')
    file_paths = fetcher.fetch_data_range(pair='BTC/USDT', timeframe='1m', total_days=1)
    loader = DataLoader(file_paths=file_paths, timeframe='1m', gpu_context=ctx, gpu_queue=queue)
    ohlcv_df = loader.load_all_data()
    closes = ohlcv_df['close'].values.astype(np.float32)

    period = 0  # Default from gpu_default_params
    print(f'Testing PriceAccel with period={period}')
    
    # CPU implementation matching GPU
    def cpu_price_accel(closes, period):
        n = len(closes)
        out = np.zeros(n, dtype=np.float32)
        for bar in range(n):
            if bar < period + 1:
                out[bar] = 0.0
            else:
                velocity_now = closes[bar] - closes[bar - period]
                velocity_prev = closes[bar - 1] - closes[bar - period - 1]
                out[bar] = velocity_now - velocity_prev
        return out
    
    accel_cpu = cpu_price_accel(closes, period)
    
    # Compute GPU PriceAccel using isolated kernel
    kernel_src = (ROOT / 'src' / 'gpu_kernels' / 'precompute_all_indicators.cl').read_text()
    wrapper = f'\n__kernel void iso(__global OHLCVBar *ohlcv, const int num_bars, __global float *out) {{ if (get_global_id(0) == 0) compute_price_acceleration(ohlcv, num_bars, {period}, &out[48 * num_bars]); }}'
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
    gpu_accel = out_2d[48]
    
    print(f'CPU PriceAccel sample at 10: {accel_cpu[10]}')
    print(f'GPU PriceAccel sample at 10: {gpu_accel[10]}')
    
    # Print comparison
    print(f'\nbar | close | cpu_accel | gpu_accel | diff')
    for i in range(0, 20):
        diff = abs(gpu_accel[i] - accel_cpu[i])
        print(f'{i:3d} | {closes[i]:8.2f} | {accel_cpu[i]:10.4f} | {gpu_accel[i]:10.4f} | {diff:8.4f}')
