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

def kernel_dpo(closes, period):
    n = len(closes)
    out = np.zeros(n, dtype=np.float32)
    offset = (period // 2) + 1
    
    for bar in range(n):
        if bar < period - 1 + offset:
            out[bar] = np.float32(0.0)
            continue
        
        # Compute SMA at bar - offset
        idx = bar - offset
        if idx < period - 1:
            out[bar] = np.float32(0.0)
            continue
            
        s = np.float32(0.0)
        for j in range(idx - period + 1, idx + 1):
            s = np.float32(s + np.float32(closes[j]))
        sma = np.float32(s / np.float32(period))
        
        out[bar] = np.float32(closes[idx] - sma)
    
    return out

if __name__ == '__main__':
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    fetcher = DataFetcher(exchange_type='futures')
    file_paths = fetcher.fetch_data_range(pair='BTC/USDT', timeframe='1m', total_days=1)
    loader = DataLoader(file_paths=file_paths, timeframe='1m', gpu_context=ctx, gpu_queue=queue)
    ohlcv_df = loader.load_all_data()
    closes = ohlcv_df['close'].values.astype(np.float32)

    period = 20
    dpo_cpu = kernel_dpo(closes, period)
    sma_cpu = kernel_sma(closes, period)
    
    print(f'CPU DPO sample at 30: {dpo_cpu[30]}')
    
    # Compute GPU DPO using isolated kernel
    kernel_src = (ROOT / 'src' / 'gpu_kernels' / 'precompute_all_indicators.cl').read_text()
    wrapper = f'\n__kernel void iso(__global OHLCVBar *ohlcv, const int num_bars, __global float *out) {{ if (get_global_id(0) == 0) compute_dpo(ohlcv, num_bars, {period}, &out[30 * num_bars]); }}'
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
    gpu_dpo = out_2d[30]
    
    print(f'GPU DPO sample at 30: {gpu_dpo[30]}')
    
    # Print comparison for bars around the offset threshold
    offset = (period // 2) + 1
    print(f'\nDPO period={period}, offset={offset}')
    print('bar | close | cpu_dpo | gpu_dpo | diff')
    for i in range(25, 45):
        diff = abs(gpu_dpo[i] - dpo_cpu[i])
        print(f'{i:3d} | {closes[i]:8.2f} | {dpo_cpu[i]:8.4f} | {gpu_dpo[i]:8.4f} | {diff:8.4f}')
