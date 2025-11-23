#!/usr/bin/env python3
"""
Compare each indicator's output from the batched precompute vs an isolated single-indicator kernel.
This helps identify batched precompute wrappers that corrupt outputs for certain indicators.
"""
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from pathlib import Path
import numpy as np
import pyopencl as cl
from src.data_provider.fetcher import DataFetcher
from src.data_provider.loader import DataLoader
from src.backtester.compact_simulator import CompactBacktester
import os
import csv


def main():
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    fetcher = DataFetcher(exchange_type='futures')
    file_paths = fetcher.fetch_data_range(pair='BTC/USDT', timeframe='1m', total_days=1)
    loader = DataLoader(file_paths=file_paths, timeframe='1m', gpu_context=ctx, gpu_queue=queue)
    ohlcv = loader.load_all_data()
    num_bars = len(ohlcv)
    print('Loaded bars:', num_bars)

    backtester = CompactBacktester(gpu_context=ctx, gpu_queue=queue, initial_balance=100.0)

    # Batched (official) precompute
    print('Running batched precompute...')
    buf_batched = backtester._precompute_indicators(ohlcv[['open','high','low','close','volume']].values.astype(np.float32))
    out_batched = np.empty(backtester.NUM_INDICATORS * num_bars, dtype=np.float32)
    cl.enqueue_copy(queue, out_batched, buf_batched)
    queue.finish()
    indicators_batched = out_batched.reshape((backtester.NUM_INDICATORS, num_bars))

    # Compare per-indicator with isolated kernels
    results = []

    kernel_src_base = (Path(__file__).resolve().parents[1] / 'src' / 'gpu_kernels' / 'precompute_all_indicators.cl').read_text()

    for idx in range(backtester.NUM_INDICATORS):
        print(f'Checking indicator {idx}...')
        # Build wrapper for isolated indicator idx
        call_line = {
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
        }.get(idx, '')

        wrapper = '\n__kernel void precompute_subset(__global OHLCVBar *ohlcv, const int num_bars, __global float *indicators_out) {\n    if (get_global_id(0) == 0) {\n'
        wrapper += '        ' + call_line + '\n'
        wrapper += '    }\n}\n'

        # Build program for this isolated indicator
        kernel_src = kernel_src_base + '\n' + wrapper
        prg = cl.Program(ctx, kernel_src).build()
        subset_kernel = prg.precompute_subset

        # Prepare buffers
        ohlcv_flat = ohlcv[['open','high','low','close','volume']].values.astype(np.float32)
        ohlcv_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=ohlcv_flat)
        indicators_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=backtester.NUM_INDICATORS * num_bars * 4)
        # Run kernel
        subset_kernel(queue, (1,), None, ohlcv_buf, np.int32(num_bars), indicators_buf)
        queue.finish()

        # Read back isolated output
        out_isolated = np.empty(backtester.NUM_INDICATORS * num_bars, dtype=np.float32)
        cl.enqueue_copy(queue, out_isolated, indicators_buf)
        queue.finish()
        indicators_isolated = out_isolated.reshape((backtester.NUM_INDICATORS, num_bars))

        # Compare batched vs isolated
        batched = indicators_batched[idx]
        isolated = indicators_isolated[idx]
        diffs = np.abs(batched - isolated)
        max_diff = diffs.max()
        mean_diff = diffs.mean()
        results.append((idx, mean_diff, max_diff))

    # Write results
    out_dir = Path('logs') / 'compare_batched_isolated'
    os.makedirs(out_dir, exist_ok=True)
    with open(out_dir / 'batched_isolated_summary.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['indicator', 'mean_diff', 'max_diff'])
        for r in results:
            writer.writerow(r)
    print('Done')


if __name__ == '__main__':
    main()
