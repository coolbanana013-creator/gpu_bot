#!/usr/bin/env python3
import numpy as np
from pathlib import Path
import pyopencl as cl
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_provider.fetcher import DataFetcher
from src.data_provider.loader import DataLoader
from src.live_trading.indicator_calculator import RealTimeIndicatorCalculator


def compare_rsi(period):
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    fetcher = DataFetcher(exchange_type='futures')
    file_paths = fetcher.fetch_data_range(pair='BTC/USDT', timeframe='1m', total_days=1)
    loader = DataLoader(file_paths=file_paths, timeframe='1m', gpu_context=ctx, gpu_queue=queue)
    ohlcv_df = loader.load_all_data()

    # GPU
    kernel_src = (ROOT / 'src' / 'gpu_kernels' / 'precompute_all_indicators.cl').read_text()
    wrapper = f"\n__kernel void iso(__global OHLCVBar *o, const int n, __global float *out) {{ if (get_global_id(0) == 0) compute_rsi(o, n, {period}, out); }}"
    prg = cl.Program(ctx, kernel_src + '\n' + wrapper).build()
    ohlcv_flat = ohlcv_df[['open','high','low','close','volume']].values.astype(np.float32).flatten()
    num_bars = len(ohlcv_df)
    mf = cl.mem_flags
    ohlcv_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ohlcv_flat)
    out_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=num_bars * 4)
    prg.iso(queue, (1,), None, ohlcv_buf, np.int32(num_bars), out_buf)
    queue.finish()
    gpu_res = np.empty(num_bars, dtype=np.float32)
    cl.enqueue_copy(queue, gpu_res, out_buf)

    # CPU
    calc = RealTimeIndicatorCalculator()
    # Use internal _kernel_rsi via calculate_indicator or directly by regenerating the arr
    rt = RealTimeIndicatorCalculator(lookback_bars=num_bars)
    cpu_res = np.zeros(num_bars, dtype=np.float32)
    for i in range(num_bars):
        r = ohlcv_df.iloc[i]
        rt.update_price_data(r['open'], r['high'], r['low'], r['close'], r['volume'])
        cpu_res[i] = float(rt.calculate_indicator(12 if period == 7 else 14, float(period), 0.0, 0.0))

    # Compare
    mask = ~np.isnan(cpu_res) & ~np.isnan(gpu_res)
    diffs = np.abs(gpu_res - cpu_res)
    mismatches = np.where((diffs > 0.0009) & mask)[0]
    print(f'RSI period={period} - mismatches: {len(mismatches)}')
    if len(mismatches) > 0:
        print('first mismatches:')
        for m in mismatches[:20]:
            print(m, gpu_res[m], cpu_res[m], diffs[m])

    # Print sample around first mismatch if exists
    if len(mismatches) > 0:
        m = mismatches[0]
        print('\nDetailed differences around first mismatch:')
        for i in range(max(0, m - 5), min(num_bars, m + 6)):
            print(i, 'gpu', gpu_res[i], 'cpu', cpu_res[i], 'diff', diffs[i])

    # Manual GPU-emulation in Python using float32 to verify algorithm parity
    closes = ohlcv_df['close'].values.astype(np.float32)
    calc_em = np.full(num_bars, np.float32(50.0), dtype=np.float32)
    if num_bars >= period + 1:
        avg_gain = np.float32(0.0)
        avg_loss = np.float32(0.0)
        for i in range(1, period + 1):
            change = np.float32(closes[i] - closes[i - 1])
            if change > 0:
                avg_gain = np.float32(avg_gain + change)
            else:
                avg_loss = np.float32(avg_loss + np.abs(change))
        avg_gain = np.float32(avg_gain / np.float32(period))
        avg_loss = np.float32(avg_loss / np.float32(period))
        for bar in range(num_bars):
            if bar < period:
                calc_em[bar] = np.float32(50.0)
                continue
            change = np.float32(closes[bar] - closes[bar - 1])
            gain = np.float32(change if change > 0 else 0.0)
            loss = np.float32(-change if change < 0 else 0.0)
            avg_gain = np.float32((avg_gain * np.float32(period - 1) + gain) / np.float32(period))
            avg_loss = np.float32((avg_loss * np.float32(period - 1) + loss) / np.float32(period))
            if avg_loss < np.float32(1e-10):
                calc_em[bar] = np.float32(100.0)
            else:
                rs = np.float32(avg_gain / avg_loss)
                calc_em[bar] = np.float32(100.0 - (100.0 / (np.float32(1.0) + rs)))

    # Compare GPU to manual emulation
    diffs_em = np.abs(gpu_res - calc_em)
    diffs_cpu_em = np.abs(cpu_res - calc_em)
    em_mismatches = np.where((diffs_em > 0.0009) & mask)[0]
    print('\nEmulation mismatches (GPU vs manual):', len(em_mismatches))
    if len(em_mismatches) > 0:
        for i in em_mismatches[:10]:
            print('bar', i, 'gpu', gpu_res[i], 'manual', calc_em[i], 'cpu', cpu_res[i], 'diff_gpu_manual', diffs_em[i], 'diff_cpu_manual', diffs_cpu_em[i])


if __name__ == '__main__':
    for p in [7, 21]:
        compare_rsi(p)
