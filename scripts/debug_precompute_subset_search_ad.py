#!/usr/bin/env python3
"""
Find minimal batch composition that causes AD (indicator 39) to differ between batched precompute and an isolated compute_ad kernel.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pyopencl as cl
from src.data_provider.fetcher import DataFetcher
from src.data_provider.loader import DataLoader
from src.backtester.compact_simulator import CompactBacktester
import csv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

kernel_src_base = (ROOT / 'src' / 'gpu_kernels' / 'precompute_all_indicators.cl').read_text()

ID_TO_CALL = {
    0: 'compute_sma(ohlcv, num_bars, 5, &indicators_out[0 * num_bars]);',
    1: 'compute_sma(ohlcv, num_bars, 10, &indicators_out[1 * num_bars]);',
    2: 'compute_sma(ohlcv, num_bars, 20, &indicators_out[2 * num_bars]);',
    3: 'compute_sma(ohlcv, num_bars, 50, &indicators_out[3 * num_bars]);',
    4: 'compute_sma(ohlcv, num_bars, 100, &indicators_out[4 * num_bars]);',
    5: 'compute_sma(ohlcv, num_bars, 200, &indicators_out[5 * num_bars]);',
    6: 'compute_ema(ohlcv, num_bars, 5, &indicators_out[6 * num_bars]);',
    7: 'compute_ema(ohlcv, num_bars, 10, &indicators_out[7 * num_bars]);',
    8: 'compute_ema(ohlcv, num_bars, 20, &indicators_out[8 * num_bars]);',
    9: 'compute_ema(ohlcv, num_bars, 50, &indicators_out[9 * num_bars]);',
    10: 'compute_ema(ohlcv, num_bars, 100, &indicators_out[10 * num_bars]);',
    11: 'compute_ema(ohlcv, num_bars, 200, &indicators_out[11 * num_bars]);',
    12: 'compute_rsi(ohlcv, num_bars, 7, &indicators_out[12 * num_bars]);',
    13: 'compute_rsi(ohlcv, num_bars, 14, &indicators_out[13 * num_bars]);',
    14: 'compute_rsi(ohlcv, num_bars, 21, &indicators_out[14 * num_bars]);',
    15: 'compute_stochastic(ohlcv, num_bars, 14, 3, &indicators_out[15 * num_bars]);',
    16: 'compute_stochrsi(ohlcv, num_bars, 14, &indicators_out[16 * num_bars], &indicators_out[13 * num_bars]);',
    17: 'compute_momentum(ohlcv, num_bars, 10, &indicators_out[17 * num_bars]);',
    18: 'compute_roc(ohlcv, num_bars, 10, &indicators_out[18 * num_bars]);',
    19: 'compute_willr(ohlcv, num_bars, 14, &indicators_out[19 * num_bars]);',
    20: 'compute_atr(ohlcv, num_bars, 14, &indicators_out[20 * num_bars]);',
    21: 'compute_atr(ohlcv, num_bars, 20, &indicators_out[21 * num_bars]);',
    22: 'compute_natr(ohlcv, num_bars, 14, &indicators_out[22 * num_bars], &indicators_out[20 * num_bars]);',
    23: 'compute_bollinger_bands(ohlcv, num_bars, 20, 2.0f, &indicators_out[23 * num_bars], &indicators_out[24 * num_bars]);',
    24: 'compute_bollinger_bands(ohlcv, num_bars, 20, 2.0f, &indicators_out[23 * num_bars], &indicators_out[24 * num_bars]);',
    25: 'compute_keltner(ohlcv, num_bars, 20, &indicators_out[25 * num_bars], &indicators_out[21 * num_bars]);',
    26: 'compute_macd(ohlcv, num_bars, 12, 26, 9, &indicators_out[26 * num_bars]);',
    27: 'compute_adx(ohlcv, num_bars, 14, &indicators_out[27 * num_bars]);',
    28: 'compute_aroon_up(ohlcv, num_bars, 25, &indicators_out[28 * num_bars]);',
    29: 'compute_cci(ohlcv, num_bars, 20, &indicators_out[29 * num_bars]);',
    30: 'compute_dpo(ohlcv, num_bars, 20, &indicators_out[30 * num_bars]);',
    31: 'compute_psar(ohlcv, num_bars, 0.02f, 0.2f, &indicators_out[31 * num_bars]);',
    32: 'compute_supertrend(ohlcv, num_bars, 10, 3.0f, &indicators_out[32 * num_bars], &indicators_out[20 * num_bars]);',
    33: 'compute_trend_strength(ohlcv, num_bars, 20, &indicators_out[33 * num_bars]);',
    34: 'compute_trend_strength(ohlcv, num_bars, 50, &indicators_out[34 * num_bars]);',
    35: 'compute_trend_strength(ohlcv, num_bars, 100, &indicators_out[35 * num_bars]);',
    36: 'compute_obv(ohlcv, num_bars, &indicators_out[36 * num_bars]);',
    37: 'compute_vwap(ohlcv, num_bars, &indicators_out[37 * num_bars]);',
    38: 'compute_mfi(ohlcv, num_bars, 14, &indicators_out[38 * num_bars]);',
    39: 'compute_ad(ohlcv, num_bars, &indicators_out[39 * num_bars]);',
    40: 'compute_volume_sma(ohlcv, num_bars, 20, &indicators_out[40 * num_bars]);',
    41: 'compute_pivot_points(ohlcv, num_bars, &indicators_out[41 * num_bars]);',
    42: 'compute_fractal_high(ohlcv, num_bars, 5, &indicators_out[42 * num_bars]);',
    43: 'compute_fractal_low(ohlcv, num_bars, 5, &indicators_out[43 * num_bars]);',
    44: 'compute_support_resistance(ohlcv, num_bars, 20, &indicators_out[44 * num_bars]);',
    45: 'compute_price_channel(ohlcv, num_bars, 20, &indicators_out[45 * num_bars]);',
    46: 'compute_hl_range(ohlcv, num_bars, &indicators_out[46 * num_bars]);',
    47: 'compute_close_position(ohlcv, num_bars, &indicators_out[47 * num_bars]);',
    48: 'compute_price_acceleration(ohlcv, num_bars, 10, &indicators_out[48 * num_bars]);',
    49: 'compute_volume_roc(ohlcv, num_bars, 10, &indicators_out[49 * num_bars]);',
}


def run_subset(ctx, queue, ohlcv_flat, ohlcv_df, subset_ids, num_bars, backtester):
    """Build a kernel with subset_ids calls, run it and return the AD (39) output array."""
    mf = cl.mem_flags
    ohlcv_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ohlcv_flat)
    indicators_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=backtester.NUM_INDICATORS * num_bars * 4)
    call_lines = [ID_TO_CALL[i] for i in subset_ids]
    wrapper = '\n__kernel void precompute_subset(__global OHLCVBar *ohlcv, const int num_bars, __global float *indicators_out) {\n    if (get_global_id(0) == 0) {\n'
    for l in call_lines:
        wrapper += '        ' + l + '\n'
    wrapper += '    }\n}\n'
    kernel_src = kernel_src_base + '\n' + wrapper
    try:
        prg = cl.Program(ctx, kernel_src).build()
    except Exception as e:
        print('Build failed:', e)
        return None
    subset_kernel = prg.precompute_subset
    subset_kernel(queue, (1,), None, ohlcv_buf, np.int32(num_bars), indicators_buf)
    queue.finish()
    out = np.empty(backtester.NUM_INDICATORS * num_bars, dtype=np.float32)
    cl.enqueue_copy(queue, out, indicators_buf)
    queue.finish()
    arr = out.reshape((backtester.NUM_INDICATORS, num_bars))
    return arr[39]


def run_isolated(ctx, queue, ohlcv_flat, ohlcv_df, num_bars, backtester):
    """Run isolated compute_ad kernel to retrieve AD output."""
    return run_subset(ctx, queue, ohlcv_flat, ohlcv_df, [39], num_bars, backtester)


def main():
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    fetcher = DataFetcher(exchange_type='futures')
    file_paths = fetcher.fetch_data_range(pair='BTC/USDT', timeframe='1m', total_days=1)
    loader = DataLoader(file_paths=file_paths, timeframe='1m', gpu_context=ctx, gpu_queue=queue)
    ohlcv_df = loader.load_all_data()
    num_bars = len(ohlcv_df)

    backtester = CompactBacktester(gpu_context=ctx, gpu_queue=queue, initial_balance=100.0)
    ohlcv_flat = ohlcv_df[['open','high','low','close','volume']].values.astype(np.float32).flatten()

    # Baseline: isolated AD
    ad_isolated = run_isolated(ctx, queue, ohlcv_flat, ohlcv_df, num_bars, backtester)
    # Baseline: batched full run
    buf_batched = backtester._precompute_indicators(ohlcv_df[['open','high','low','close','volume']].values.astype(np.float32))
    out_batched = np.empty(backtester.NUM_INDICATORS * num_bars, dtype=np.float32)
    cl.enqueue_copy(queue, out_batched, buf_batched)
    queue.finish()
    ad_batched = out_batched.reshape((backtester.NUM_INDICATORS, num_bars))[39]

    # If batched differs from isolated, search for the culprit
    diffs = np.abs(ad_batched - ad_isolated)
    base_diff = diffs.max()
    print('Full batched vs isolated AD max diff:', base_diff)
    if base_diff == 0:
        print('No corruption with full batched kernel.')
        return

    # We'll test adding groups of indicators to an ad-only kernel to see what breaks.
    groups = [list(range(0,12)), list(range(12,26)), list(range(26,40)), list(range(40,50))]
    # Add AD into each group and run
    results = []
    for group in groups:
        test_ids = group + [39]
        print('Testing group', group)
        arr = run_subset(ctx, queue, ohlcv_flat, ohlcv_df, test_ids, num_bars, backtester)
        if arr is None:
            print('Build failure for group', group)
            continue
        diffs = np.abs(arr - ad_isolated)
        maxd = diffs.max()
        print('Group max diff:', maxd)
        results.append((group, maxd))

    # Now try cumulative addition to find combined-group corruption
    print('\nCumulative group check:')
    cum_ids = [39]
    cum_results = []
    for g in groups:
        cum_ids = cum_ids + g
        print('Testing cumulative subset:', cum_ids)
        arr = run_subset(ctx, queue, ohlcv_flat, ohlcv_df, cum_ids, num_bars, backtester)
        if arr is None:
            print('Build failure for cumulative subset')
            break
        maxd = np.abs(arr - ad_isolated).max()
        print('Cumulative maxd:', maxd)
        cum_results.append((list(cum_ids), maxd))
        if maxd > 1e-6:
            print('Corruption found when adding group', g)
            # bisect within the latest added group
            # Find minimal subset inside the union that causes corruption
            candidate = list(cum_ids)
            # We'll try removing each of the newly added group's indicators one-by-one to find culprits
            newly_added = g
            suspects = []
            for ind in newly_added:
                test_ids = [x for x in cum_ids if x != ind]
                arr_tmp = run_subset(ctx, queue, ohlcv_flat, ohlcv_df, test_ids, num_bars, backtester)
                if arr_tmp is None:
                    continue
                maxd_tmp = np.abs(arr_tmp - ad_isolated).max()
                if maxd_tmp > 1e-6:
                    suspects.append(ind)
            print('Suspected culprits in group:', suspects)
            break

    # If a group corrupts it, bisect that group
    for group, maxd in results:
        if maxd > 1e-6:
            print('Corrupting group found:', group)
            # Bisection search inside the group
            candidates = group.copy()
            while len(candidates) > 1:
                mid = len(candidates) // 2
                left = candidates[:mid]
                right = candidates[mid:]
                # left + ad
                arr_left = run_subset(ctx, queue, ohlcv_flat, ohlcv_df, left + [39], num_bars, backtester)
                if arr_left is None:
                    print('left build fail; assume break')
                    candidates = left
                    continue
                maxd_left = np.abs(arr_left - ad_isolated).max()
                if maxd_left > 1e-6:
                    candidates = left
                    continue
                # right side
                arr_right = run_subset(ctx, queue, ohlcv_flat, ohlcv_df, right + [39], num_bars, backtester)
                if arr_right is None:
                    print('right build fail; assume break')
                    candidates = right
                    continue
                maxd_right = np.abs(arr_right - ad_isolated).max()
                if maxd_right > 1e-6:
                    candidates = right
                    continue
                # Not found in either half — try combining halves with ad
                print('Not found in halves - breaking out')
                break

            print('Likely culprit indicators:', candidates)

    # Finally, try adding indicators one-by-one together with AD to find precise culprit
    for i in range(50):
        if i == 39:
            continue
        arr = run_subset(ctx, queue, ohlcv_flat, ohlcv_df, [39, i], num_bars, backtester)
        if arr is None:
            print('Build failed for pair (39,', i, ')')
            continue
        maxd = np.abs(arr - ad_isolated).max()
        if maxd > 1e-6:
            print('Indicator', i, 'breaks AD: max diff', maxd)

    print('Done.')

if __name__ == '__main__':
    main()
