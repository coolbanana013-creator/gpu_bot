#!/usr/bin/env python3
import numpy as np
from pathlib import Path
import pyopencl as cl
import talib
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_provider.fetcher import DataFetcher
from src.data_provider.loader import DataLoader

# Helper: CPU kernel SMA/EMA implementations (float32)

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


def kernel_ema(arr, period):
    n = len(arr)
    out = np.zeros(n, dtype=np.float32)
    if period <= 0:
        return out
    prev = np.float32(0.0)
    k = np.float32(2.0) / np.float32(period + 1)
    for i in range(n):
        if i < period - 1:
            out[i] = np.float32(0.0)
            continue
        if i == period - 1:
            s = np.float32(0.0)
            for j in range(i - period + 1, i + 1):
                s = np.float32(s + np.float32(arr[j]))
            prev = np.float32(s / np.float32(period))
            out[i] = prev
            continue
        prev = np.float32((np.float32(arr[i]) - prev) * k + prev)
        out[i] = prev
    return out


def kernel_macd(closes, fast, slow, signal_period):
    fast_arr = kernel_ema(closes, fast)
    slow_arr = kernel_ema(closes, slow)
    n = len(closes)
    macd = np.zeros(n, dtype=np.float32)
    signal = np.float32(0.0)
    for i in range(n):
        if i < slow:
            macd[i] = 0.0
            continue
        macd_line = np.float32(fast_arr[i] - slow_arr[i])
        if i >= slow - 1:
            if i == slow - 1:
                signal = macd_line
            else:
                k = np.float32(2.0) / np.float32(signal_period + 1)
                signal = np.float32((macd_line - signal) * k + signal)
        macd[i] = macd_line
    return macd, signal


if __name__ == '__main__':
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    fetcher = DataFetcher(exchange_type='futures')
    file_paths = fetcher.fetch_data_range(pair='BTC/USDT', timeframe='1m', total_days=1)
    loader = DataLoader(file_paths=file_paths, timeframe='1m', gpu_context=ctx, gpu_queue=queue)
    ohlcv_df = loader.load_all_data()
    closes = ohlcv_df['close'].values.astype(np.float32)

    fast = 12
    slow = 26
    signal = 9

    macd_cpu, sig_cpu = kernel_macd(closes, fast, slow, signal)
    fast_arr = kernel_ema(closes, fast)
    slow_arr = kernel_ema(closes, slow)
    print('CPU MACD sample at 26:', macd_cpu[26])
    print('CPU fast_ema[26]:', fast_arr[26], 'CPU slow_ema[26]:', slow_arr[26])

    # Compute GPU MACD using isolated kernel
    kernel_src = (ROOT / 'src' / 'gpu_kernels' / 'precompute_all_indicators.cl').read_text()
    wrapper = '\n__kernel void iso(__global OHLCVBar *ohlcv, const int num_bars, __global float *out) { if (get_global_id(0) == 0) { compute_macd(ohlcv, num_bars, 12, 26, 9, &out[26 * num_bars]); compute_ema(ohlcv, num_bars, 12, &out[100 * num_bars]); compute_ema(ohlcv, num_bars, 26, &out[101 * num_bars]); } }'
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
    gpu_macd = out_2d[26]
    gpu_fast = out_2d[100]
    gpu_slow = out_2d[101]

    # Also test with fast==slow (should give 0 macd_line always)
    wrapper_equal = '\n__kernel void iso_equal(__global OHLCVBar *ohlcv, const int num_bars, __global float *out) { if (get_global_id(0) == 0) compute_macd(ohlcv, num_bars, 12, 12, 9, &out[30 * num_bars]); }'
    prg2 = cl.Program(ctx, kernel_src + '\n' + wrapper_equal).build()
    out_buf2 = cl.Buffer(ctx, mf.WRITE_ONLY, size=n_out_indicators * len(closes) * 4)
    prg2.iso_equal(queue, (1,), None, ohlcv_buf, np.int32(len(closes)), out_buf2)
    queue.finish()
    out_flat2 = np.empty(n_out_indicators * len(closes), dtype=np.float32)
    cl.enqueue_copy(queue, out_flat2, out_buf2)
    out_2d_2 = out_flat2.reshape((n_out_indicators, len(closes)))
    gpu_macd_equal = out_2d_2[30]
    print('\nGPU MACD (fast==slow) sample at 26:', gpu_macd_equal[26])

    print('GPU MACD sample at 26:', gpu_macd[26])
    print('GPU fast_ema[26]:', gpu_fast[26], 'GPU slow_ema[26]:', gpu_slow[26])
    # Print a few bars for comparison
    for i in range(24, 36):
        diff = gpu_macd[i] - (gpu_fast[i] - gpu_slow[i])
        print(i, 'close', closes[i], 'gpu_macd', gpu_macd[i], 'gpu_fast', gpu_fast[i], 'gpu_slow', gpu_slow[i], 'gpu_diff', diff, 'cpu_macd', macd_cpu[i], 'cpu_fast', fast_arr[i], 'cpu_slow', slow_arr[i])

    # Print first 50 indicators for a sample bar to inspect potential writes / overlaps
    print('\nSample all first 50 indicators for bar 26:')
    for idx in range(0, 50):
        print(idx, '=>', out_2d[idx, 26])

    # Compare to talib MACD
    macd, macdsignal, macdhist = talib.MACD(closes, fastperiod=fast, slowperiod=slow, signalperiod=signal)
    print('TA-LIB MACD sample at 26:', macd[26])
    for i in range(24, 36):
        print(i, 'talib', macd[i])
