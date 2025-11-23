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

if __name__ == '__main__':
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    fetcher = DataFetcher(exchange_type='futures')
    file_paths = fetcher.fetch_data_range(pair='BTC/USDT', timeframe='1m', total_days=1)
    loader = DataLoader(file_paths=file_paths, timeframe='1m', gpu_context=ctx, gpu_queue=queue)
    ohlcv_df = loader.load_all_data()
    
    period = 25
    print(f'Testing Aroon Up with period={period}')
    
    # Compute GPU Aroon
    kernel_src = (ROOT / 'src' / 'gpu_kernels' / 'precompute_all_indicators.cl').read_text()
    wrapper = f'\n__kernel void iso(__global OHLCVBar *o, const int n, __global float *out) {{ if (get_global_id(0) == 0) compute_aroon_up(o, n, {period}, out); }}'
    prg = cl.Program(ctx, kernel_src + '\n' + wrapper).build()
    ohlcv_flat = ohlcv_df[['open','high','low','close','volume']].values.astype(np.float32).flatten()
    num_bars = len(ohlcv_df)
    mf = cl.mem_flags
    ohlcv_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ohlcv_flat)
    out_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=num_bars * 4)
    prg.iso(queue, (1,), None, ohlcv_buf, np.int32(num_bars), out_buf)
    queue.finish()
    gpu_aroon = np.empty(num_bars, dtype=np.float32)
    cl.enqueue_copy(queue, gpu_aroon, out_buf)
    
    # Compute CPU Aroon
    calc = RealTimeIndicatorCalculator()
    cpu_aroon = np.full(num_bars, np.nan, dtype=np.float32)
    for i in range(num_bars):
        bar = ohlcv_df.iloc[i]
        calc.update_price_data(bar['open'], bar['high'], bar['low'], bar['close'], bar['volume'])
        cpu_aroon[i] = calc.calculate_indicator(28, 0.0, 0.0, 0.0)
    
    print(f'GPU Aroon sample at period={period}: {gpu_aroon[period]}')
    print(f'CPU Aroon sample at period={period}: {cpu_aroon[period]}')
    
    # Find bars with any differences (mask positions align with full arrays)
    valid_mask = ~np.isnan(gpu_aroon) & ~np.isnan(cpu_aroon)
    diffs = np.abs(gpu_aroon - cpu_aroon)
    diff_indices = np.where((diffs > 0.0001) & valid_mask)[0]
    
    print(f'\nBars with any diff: {len(diff_indices)}')
    print(f'\nFirst 20 mismatches (bar | gpu_val | cpu_val | diff):')
    for idx in diff_indices[:20]:
        gpu_val = gpu_aroon[idx]
        cpu_val = cpu_aroon[idx]
        diff = abs(gpu_val - cpu_val)
        print(f'{idx:3d} | {gpu_val:10.4f} | {cpu_val:10.4f} | {diff:10.4f}')
    
    # Helper functions for variant matching
    def rightmost_idx(arr):
        highest = arr[0]
        idx = 0
        for i in range(len(arr)):
            if arr[i] >= highest:
                highest = arr[i]
                idx = i
        return idx

    def leftmost_idx(arr):
        highest = arr[0]
        idx = 0
        for i in range(len(arr)):
            if arr[i] > highest:
                highest = arr[i]
                idx = i
        return idx

    def compute_aroon_by_high_idx(bar, period, high_idx):
        bars_since_high = bar - high_idx
        return ((period - bars_since_high) / period) * 100.0

    # For all mismatched bars: classify how TA-Lib matches variants and whether current bar equals max
    variants_count = {'inc_l':0, 'inc_r':0, 'exc_l':0, 'exc_r':0, 'none':0}
    current_is_max_count = 0
    inspect_list = diff_indices[:200]  # cap for sanity
    for m in inspect_list:
        high_arr = ohlcv_df['high'].values.astype(np.float64)
        start_inclusive = max(0, m - period + 1)
        start_exclusive = max(0, m - period)
        win_inclusive = high_arr[start_inclusive:m+1]
        win_exclusive = high_arr[start_exclusive:m]
        # get candidate right/left indices
        r_idx_inc = rightmost_idx(win_inclusive) + start_inclusive
        l_idx_inc = leftmost_idx(win_inclusive) + start_inclusive
        r_idx_exc = rightmost_idx(win_exclusive) + start_exclusive
        l_idx_exc = leftmost_idx(win_exclusive) + start_exclusive
        inc_l = compute_aroon_by_high_idx(m, period, l_idx_inc)
        inc_r = compute_aroon_by_high_idx(m, period, r_idx_inc)
        exc_l = compute_aroon_by_high_idx(m, period, l_idx_exc)
        exc_r = compute_aroon_by_high_idx(m, period, r_idx_exc)
        talib_val = float(cpu_aroon[m])
        matched = False
        if abs(talib_val - inc_l) < 1e-6:
            variants_count['inc_l'] += 1
            matched = True
        if abs(talib_val - inc_r) < 1e-6:
            variants_count['inc_r'] += 1
            matched = True
        if abs(talib_val - exc_l) < 1e-6:
            variants_count['exc_l'] += 1
            matched = True
        if abs(talib_val - exc_r) < 1e-6:
            variants_count['exc_r'] += 1
            matched = True
        if not matched:
            variants_count['none'] += 1
        if abs(high_arr[m] - max(win_inclusive)) < 1e-12:
            current_is_max_count += 1
    print('\nVariant counts among first mismatches:', variants_count)
    print('current_is_max_count among first mismatches:', current_is_max_count)

    # For the first mismatched bar, dump the search window and candidate highs
    if len(diff_indices) > 0:
        m = diff_indices[0]
        print('\nDetailed mismatch info for the first mismatched bar:')
        print('bar | gpu | cpu | diff')
        print(f'{m:3d} | {gpu_aroon[m]:10.4f} | {cpu_aroon[m]:10.4f} | {abs(gpu_aroon[m]-cpu_aroon[m]):10.4f}')
        
        # Dump search windows and highest value positions - test different interpretations
        high_arr = ohlcv_df['high'].values.astype(np.float64)
        start_inclusive = m - period + 1
        start_exclusive = m - period
        if start_inclusive < 0:
            start_inclusive = 0
        if start_exclusive < 0:
            start_exclusive = 0
        win_inclusive = high_arr[start_inclusive:m+1]
        win_exclusive = high_arr[start_exclusive:m]
        print(f'\nWindow inclusive (bars {start_inclusive}..{m}):')
        print(win_inclusive)
        print(f'\nWindow exclusive (bars {start_exclusive}..{m-1}):')
        print(win_exclusive)
        
        # Find indices of highest in both windows
        # Rightmost and leftmost occurrences
        def rightmost_idx(arr, base_index):
            highest = arr[0]
            idx = 0
            for i in range(len(arr)):
                if arr[i] >= highest:
                    highest = arr[i]
                    idx = i
            return idx

        def leftmost_idx(arr, base_index):
            highest = arr[0]
            idx = 0
            for i in range(len(arr)):
                if arr[i] > highest:
                    highest = arr[i]
                    idx = i
            return idx

        # Inclusive window computations
        r_idx_inc = rightmost_idx(win_inclusive, start_inclusive) + start_inclusive
        l_idx_inc = leftmost_idx(win_inclusive, start_inclusive) + start_inclusive
        # Exclusive window
        r_idx_exc = rightmost_idx(win_exclusive, start_exclusive) + start_exclusive
        l_idx_exc = leftmost_idx(win_exclusive, start_exclusive) + start_exclusive

        print(f'\nHighest indices (inclusive window): leftmost {l_idx_inc}, rightmost {r_idx_inc}, highest val {high_arr[r_idx_inc]:.6f}')
        print(f'Highest indices (exclusive window): leftmost {l_idx_exc}, rightmost {r_idx_exc}, highest val {high_arr[r_idx_exc]:.6f}')
        
        # Manual Aroon calculations for each behavior
        def compute_aroon_by_high_idx(bar, period, high_idx):
            bars_since_high = bar - high_idx
            return ((period - bars_since_high) / period) * 100.0

        print('\nComputed Aroon for each candidate:')
        inc_l = compute_aroon_by_high_idx(m, period, l_idx_inc)
        inc_r = compute_aroon_by_high_idx(m, period, r_idx_inc)
        exc_l = compute_aroon_by_high_idx(m, period, l_idx_exc)
        exc_r = compute_aroon_by_high_idx(m, period, r_idx_exc)
        print('inclusive_leftmost:', inc_l)
        print('inclusive_rightmost:', inc_r)
        print('exclusive_leftmost:', exc_l)
        print('exclusive_rightmost:', exc_r)
        print('\nTA-Lib (CPU) reported:', cpu_aroon[m])
        print('GPU reported:', gpu_aroon[m])

        # Which variant matches TA-Lib value?
        talib_val = float(cpu_aroon[m])
        matches = []
        if abs(talib_val - inc_l) < 1e-6:
            matches.append('inclusive_leftmost')
        if abs(talib_val - inc_r) < 1e-6:
            matches.append('inclusive_rightmost')
        if abs(talib_val - exc_l) < 1e-6:
            matches.append('exclusive_leftmost')
        if abs(talib_val - exc_r) < 1e-6:
            matches.append('exclusive_rightmost')
        print('\nTA-Lib matches variants:', matches)
    else:
        print('\nNo mismatches found - all matched')

    # Show some normal bars for comparison
    print(f'\nSample bars (25-45):')
    print(f'bar | gpu_aroon | gpu_nan | cpu_aroon | cpu_nan | diff')
    for i in range(25, 45):
        gpu_val = gpu_aroon[i]
        cpu_val = cpu_aroon[i]
        gpu_nan = np.isnan(gpu_val)
        cpu_nan = np.isnan(cpu_val)
        if not gpu_nan and not cpu_nan:
            diff = abs(gpu_val - cpu_val)
            print(f'{i:3d} | {gpu_val:10.4f} | {gpu_nan} | {cpu_val:10.4f} | {cpu_nan} | {diff:10.4f}')
        else:
            print(f'{i:3d} | {gpu_val:10.4f} | {gpu_nan} | {cpu_val:10.4f} | {cpu_nan} | N/A')
