#!/usr/bin/env python3
"""
Compare isolated GPU kernel vs CPU RealTimeIndicatorCalculator per bar for a chosen indicator(s).
Usage: python scripts/debug_indicator_parity.py <indicator_idx>
"""
import sys
from pathlib import Path
import numpy as np
import pyopencl as cl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_provider.fetcher import DataFetcher
from src.data_provider.loader import DataLoader
import importlib.util
from pathlib import Path as _Path
_module_path = _Path(__file__).resolve().parents[1] / 'src' / 'live_trading' / 'indicator_calculator.py'
spec = importlib.util.spec_from_file_location('src.live_trading.indicator_calculator', str(_module_path))
ic_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ic_mod)
RealTimeIndicatorCalculator = ic_mod.RealTimeIndicatorCalculator
from src.backtester.compact_simulator import CompactBacktester

import csv


def run_indicator_debug(ind_idx: int):
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    fetcher = DataFetcher(exchange_type='futures')
    file_paths = fetcher.fetch_data_range(pair='BTC/USDT', timeframe='1m', total_days=1)
    loader = DataLoader(file_paths=file_paths, timeframe='1m', gpu_context=ctx, gpu_queue=queue)
    ohlcv_df = loader.load_all_data()

    # Build isolated kernel program to only compute the requested indicator
    kernel_src = (ROOT / 'src' / 'gpu_kernels' / 'precompute_all_indicators.cl').read_text()
    wrapper = f"\n__kernel void isolated_ind(__global OHLCVBar *ohlcv, const int num_bars, __global float *out) {{ if (get_global_id(0) == 0) {{ compute_indicator_{ind_idx}(ohlcv, num_bars, &out[{ind_idx} * num_bars]); }} }}"
    # Note: compute_indicator_N must exist; older kernels may expose functions by name like compute_rsi
    # For now use the per-indicator function names from kernel (e.g., compute_rsi for indicator 13)
    # Use a mapping for indicator functions
    idx_to_fn = {
        13: 'compute_rsi',
        12: 'compute_rsi',
        14: 'compute_rsi',
        15: 'compute_stochastic',
        16: 'compute_stochrsi',
        23: 'compute_bollinger_bands',
        24: 'compute_bollinger_bands',
        26: 'compute_macd',
    }
    if ind_idx not in idx_to_fn:
        print('Indicator function mapping not defined for index', ind_idx)
        return
    fn_name = idx_to_fn[ind_idx]
    # Build a wrapper that calls the correct kernel function; pass dummy params accordingly
    if fn_name == 'compute_rsi':
        wrapper = '\n__kernel void iso(__global OHLCVBar *ohlcv, const int num_bars, __global float *out) { if (get_global_id(0) == 0) { compute_rsi(ohlcv, num_bars, 14, &out[13 * num_bars]); } }'
    elif fn_name == 'compute_stochastic':
        wrapper = '\n__kernel void iso(__global OHLCVBar *ohlcv, const int num_bars, __global float *out) { if (get_global_id(0) == 0) { compute_stochastic(ohlcv, num_bars, 14, 3, &out[15 * num_bars]); } }'
    elif fn_name == 'compute_stochrsi':
        wrapper = '\n__kernel void iso(__global OHLCVBar *ohlcv, const int num_bars, __global float *out, __global float *rsi_buf) { if (get_global_id(0) == 0) { compute_rsi(ohlcv, num_bars, 14, rsi_buf); compute_stochrsi(ohlcv, num_bars, 14, &out[16 * num_bars], rsi_buf); } }'
    elif fn_name == 'compute_bollinger_bands':
        wrapper = '\n__kernel void iso(__global OHLCVBar *ohlcv, const int num_bars, __global float *upper, __global float *lower) { if (get_global_id(0) == 0) { compute_bollinger_bands(ohlcv, num_bars, 20, 2.0f, upper, lower); } }'
    elif fn_name == 'compute_macd':
        wrapper = '\n__kernel void iso(__global OHLCVBar *ohlcv, const int num_bars, __global float *out) { if (get_global_id(0) == 0) { compute_macd(ohlcv, num_bars, 12, 26, 9, &out[26 * num_bars]); } }'

    prg = cl.Program(ctx, kernel_src + '\n' + wrapper).build()
    num_bars = len(ohlcv_df)

    mf = cl.mem_flags
    if fn_name == 'compute_bollinger_bands':
        ohlcv_flat = ohlcv_df[['open','high','low','close','volume']].values.astype(np.float32).flatten()
        ohlcv_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ohlcv_flat)
        upper_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=num_bars * 4)
        lower_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=num_bars * 4)
        prg.iso(queue, (1,), None, ohlcv_buf, np.int32(num_bars), upper_buf, lower_buf)
        queue.finish()
        upper = np.empty(num_bars, dtype=np.float32)
        lower = np.empty(num_bars, dtype=np.float32)
        cl.enqueue_copy(queue, upper, upper_buf)
        cl.enqueue_copy(queue, lower, lower_buf)
        gpu_arr = lower if ind_idx == 24 else upper
    elif fn_name == 'compute_stochrsi':
        ohlcv_flat = ohlcv_df[['open','high','low','close','volume']].values.astype(np.float32).flatten()
        ohlcv_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ohlcv_flat)
        out_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=50 * num_bars * 4)
        rsi_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=num_bars * 4)
        prg.iso(queue, (1,), None, ohlcv_buf, np.int32(num_bars), out_buf, rsi_buf)
        queue.finish()
        out_flat = np.empty(50 * num_bars, dtype=np.float32)
        cl.enqueue_copy(queue, out_flat, out_buf)
        cl.enqueue_copy(queue, np.empty(num_bars, dtype=np.float32), rsi_buf)
        gpu_arr = out_flat.reshape((50, num_bars))[16]
    else:
        ohlcv_flat = ohlcv_df[['open','high','low','close','volume']].values.astype(np.float32).flatten()
        ohlcv_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ohlcv_flat)
        out_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=50 * num_bars * 4)
        # Construct call: default params are used
        if fn_name == 'compute_rsi':
            prg.iso(queue, (1,), None, ohlcv_buf, np.int32(num_bars), out_buf)
        else:
            prg.iso(queue, (1,), None, ohlcv_buf, np.int32(num_bars), out_buf)
        queue.finish()
        out_flat = np.empty(50 * num_bars, dtype=np.float32)
        cl.enqueue_copy(queue, out_flat, out_buf)
        gpu_arr = out_flat.reshape((50, num_bars))[ind_idx]

    # CPU compute via RealTimeIndicatorCalculator
    calc = RealTimeIndicatorCalculator(lookback_bars=max(500, num_bars))
    cpu_arr = np.zeros(num_bars, dtype=np.float32)
    for i in range(num_bars):
        r = ohlcv_df.iloc[i]
        calc.update_price_data(r['open'], r['high'], r['low'], r['close'], r['volume'])
        cpu_arr[i] = float(calc.calculate_indicator(ind_idx, 0.0, 0.0, 0.0))

    # Compare
    mask = ~np.isnan(cpu_arr) & ~np.isnan(gpu_arr)
    diffs = np.abs(gpu_arr[mask] - cpu_arr[mask])
    print(f'Indicator {ind_idx} parity: mean diff {diffs.mean()}, max diff {diffs.max()}, equal_count {np.sum(diffs < 1e-3)} / {mask.sum()}')

    out_dir = Path('logs') / 'indicator_debug'
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f'ind_{ind_idx}_parity.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['bar', 'open', 'high', 'low', 'close', 'volume', 'gpu', 'cpu', 'diff'])
        for b in range(num_bars):
            row = ohlcv_df.iloc[b]
            gpu_v = float(gpu_arr[b])
            cpu_v = float(cpu_arr[b])
            w.writerow([b, row['open'], row['high'], row['low'], row['close'], row['volume'], gpu_v, cpu_v, float(abs(gpu_v - cpu_v))])
    print('Wrote CSV to', out_dir)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python scripts/debug_indicator_parity.py <indicator_idx>')
        sys.exit(1)
    idx = int(sys.argv[1])
    run_indicator_debug(idx)
