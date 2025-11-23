#!/usr/bin/env python3
"""
Check per-bar parity for A/D (indicator 39) between isolated GPU kernel and CPU RealTimeIndicatorCalculator.
Print summary and top mismatches for investigation.
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
import importlib.util
from pathlib import Path as _Path

# Load CPU real-time indicator calculator w/o package-level imports
_indicator_path = _Path(__file__).resolve().parents[1] / 'src' / 'live_trading' / 'indicator_calculator.py'
spec = importlib.util.spec_from_file_location('src.live_trading.indicator_calculator', str(_indicator_path))
ic_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ic_mod)
RealTimeIndicatorCalculator = ic_mod.RealTimeIndicatorCalculator


def main():
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    fetcher = DataFetcher(exchange_type='futures')
    file_paths = fetcher.fetch_data_range(pair='BTC/USDT', timeframe='1m', total_days=1)
    loader = DataLoader(file_paths=file_paths, timeframe='1m', gpu_context=ctx, gpu_queue=queue)
    ohlcv_df = loader.load_all_data()
    num_bars = len(ohlcv_df)

    # Run isolated GPU AD kernel
    kernel_src = (ROOT / 'src' / 'gpu_kernels' / 'precompute_all_indicators.cl').read_text()
    wrapper = '\n__kernel void ad_isolated(__global OHLCVBar *ohlcv, const int num_bars, __global float *indicators_out) { if (get_global_id(0) == 0) { compute_ad(ohlcv, num_bars, &indicators_out[39 * num_bars]); } }'
    prg = cl.Program(ctx, kernel_src + '\n' + wrapper).build()
    ad_kernel = prg.ad_isolated

    ohlcv_flat = ohlcv_df[['open','high','low','close','volume']].values.astype(np.float32).flatten()
    mf = cl.mem_flags
    ohlcv_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ohlcv_flat)
    out_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=50 * num_bars * 4)
    ad_kernel(queue, (1,), None, ohlcv_buf, np.int32(num_bars), out_buf)
    queue.finish()
    out_flat = np.empty(50 * num_bars, dtype=np.float32)
    cl.enqueue_copy(queue, out_flat, out_buf)
    queue.finish()
    ad_isolated = out_flat.reshape((50, num_bars))[39]

    # CPU compute via RealTimeIndicatorCalculator
    calc = RealTimeIndicatorCalculator(lookback_bars=max(500, num_bars))
    ad_cpu = np.zeros(num_bars, dtype=np.float32)
    for i in range(num_bars):
        r = ohlcv_df.iloc[i]
        calc.update_price_data(r['open'], r['high'], r['low'], r['close'], r['volume'])
        ad_cpu[i] = float(calc.calculate_indicator(39, 0.0, 0.0, 0.0))

    # Compare
    mask = ~np.isnan(ad_cpu) & ~np.isnan(ad_isolated)
    if not mask.any():
        print('No valid bars to compare')
        return
    diffs = np.abs(ad_isolated[mask] - ad_cpu[mask])
    print('AD parity: mean diff', diffs.mean(), 'max diff', diffs.max(), 'same count', (diffs < 1e-3).sum(), 'out of', mask.sum())

    # Print top mismatches
    mism_idx = np.where(mask)[0][np.argsort(diffs)[-10:]]
    print('Top mismatches (bar_idx, isolated, cpu, diff):')
    for mi in mism_idx:
        print(mi, ad_isolated[mi], ad_cpu[mi], abs(ad_isolated[mi] - ad_cpu[mi]))

    # Save CSV
    out_dir = Path('logs') / 'ad_debug'
    out_dir.mkdir(parents=True, exist_ok=True)
    import csv
    with open(out_dir / 'ad_parity.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['bar', 'open', 'high', 'low', 'close', 'volume', 'isolated', 'cpu', 'diff'])
        for b in range(num_bars):
            r = ohlcv_df.iloc[b]
            w.writerow([b, r['open'], r['high'], r['low'], r['close'], r['volume'], float(ad_isolated[b]), float(ad_cpu[b]), float(abs(ad_isolated[b] - ad_cpu[b]))])
    print('Wrote logs to', out_dir)

if __name__ == '__main__':
    main()
