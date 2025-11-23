#!/usr/bin/env python3
"""
Simulate `CompactBacktester._precompute_indicators` batched kernel execution step-by-step and monitor AD (index 39) after each batch, to find which batch writes corruption.
"""
import sys
from pathlib import Path
import numpy as np
import pyopencl as cl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtester.compact_simulator import CompactBacktester
from src.data_provider.fetcher import DataFetcher
from src.data_provider.loader import DataLoader

kernel_src_base = (ROOT / 'src' / 'gpu_kernels' / 'precompute_all_indicators.cl').read_text()

# The exact map used by CompactBacktester._precompute_indicators
id_to_call = {
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

def generate_batches(num_indicators=50, batch_size=8):
    ids = list(range(num_indicators))
    for i in range(0, len(ids), batch_size):
        yield ids[i:i+batch_size]


def build_and_run_wrapper(ctx, queue, ohlcv_flat, ohlcv_df, call_ids, num_bars, indicators_buf):
    call_lines = [id_to_call[i] for i in call_ids]
    wrapper = '\n__kernel void precompute_subset(__global OHLCVBar *ohlcv, const int num_bars, __global float *indicators_out) {\n    if (get_global_id(0) == 0) {\n'
    for l in call_lines:
        wrapper += '        ' + l + '\n'
    wrapper += '    }\n}\n'
    kernel_src = kernel_src_base + '\n' + wrapper
    prg = cl.Program(ctx, kernel_src).build()
    kernel = prg.precompute_subset
    # buffers
    mf = cl.mem_flags
    ohlcv_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ohlcv_flat)
    kernel(queue, (1,), None, ohlcv_buf, np.int32(num_bars), indicators_buf)
    queue.finish()


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

    # allocate a single buffer for batched run equivalent to COMPACT backtester
    mf = cl.mem_flags
    indicator_bytes = backtester.NUM_INDICATORS * num_bars * 4
    indicators_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=indicator_bytes)

    # Baseline isolated AD
    isolated_prg = cl.Program(ctx, kernel_src_base + '\n__kernel void precompute_subset(__global OHLCVBar *ohlcv, const int num_bars, __global float *indicators_out) { if (get_global_id(0) == 0) { compute_ad(ohlcv, num_bars, &indicators_out[39 * num_bars]); } }').build()
    iso_kernel = isolated_prg.precompute_subset
    ohlcv_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ohlcv_flat)
    iso_out_b = cl.Buffer(ctx, mf.WRITE_ONLY, size=backtester.NUM_INDICATORS * num_bars * 4)
    iso_kernel(queue, (1,), None, ohlcv_b, np.int32(num_bars), iso_out_b)
    queue.finish()
    isolated_out = np.empty(backtester.NUM_INDICATORS * num_bars, dtype=np.float32)
    cl.enqueue_copy(queue, isolated_out, iso_out_b)
    queue.finish()
    ad_isolated = isolated_out.reshape((backtester.NUM_INDICATORS, num_bars))[39]

    # Now execute batches sequentially like CompactBacktester (batch_size=8)
    batch_size = 8
    batches = list(generate_batches(backtester.NUM_INDICATORS, batch_size))
    ad_after_batch = None
    for i, batch in enumerate(batches):
        print('\nExecuting batch', i, 'indicators', batch)
        build_and_run_wrapper(ctx, queue, ohlcv_flat, ohlcv_df, batch, num_bars, indicators_buf)
        # read back AD
        tmp = np.empty(backtester.NUM_INDICATORS * num_bars, dtype=np.float32)
        cl.enqueue_copy(queue, tmp, indicators_buf)
        queue.finish()
        ad_after_batch = tmp.reshape((backtester.NUM_INDICATORS, num_bars))[39]
        # Only check `ad` validity after we've executed the batch that includes indicator 39
        if 39 in batch:
            maxd = np.abs(ad_after_batch - ad_isolated).max()
            print('maxd after batch', i, maxd)
            if maxd > 1e-6:
                print('Corruption seen after batch', i, 'breaking for further analysis')
                # Perform per-function isolation inside the batch
                print('Running per-function isolation inside corrupted batch...')
                for idx in batch:
                    if idx == 39:
                        continue
                    print('Testing pair (39,', idx, ')')
                    build_and_run_wrapper(ctx, queue, ohlcv_flat, ohlcv_df, [39, idx], num_bars, indicators_buf)
                    tmp2 = np.empty(backtester.NUM_INDICATORS * num_bars, dtype=np.float32)
                    cl.enqueue_copy(queue, tmp2, indicators_buf)
                    queue.finish()
                    ad_tmp = tmp2.reshape((backtester.NUM_INDICATORS, num_bars))[39]
                    d = np.abs(ad_tmp - ad_isolated).max()
                    print('pair', idx, 'max diff', d)
                    # reset the region for AD by re-running isolated AD to clear buffer
                    build_and_run_wrapper(ctx, queue, ohlcv_flat, ohlcv_df, [39], num_bars, indicators_buf)
                break

    print('\nDone. If corruption appeared above, inspect the batch and adjacent batches for possible causes.')

if __name__ == '__main__':
    main()
