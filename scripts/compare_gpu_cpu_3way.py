#!/usr/bin/env python3
"""
Three-way comparison: batched GPU precompute, isolated GPU kernel per indicator, and CPU RealTimeIndicatorCalculator.
Outputs per-indicator percentage equality and writes CSVs for mismatches.
"""
import sys
from pathlib import Path
import numpy as np
import pyopencl as cl
import csv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtester.compact_simulator import CompactBacktester
from src.data_provider.fetcher import DataFetcher
from src.data_provider.loader import DataLoader
from src.indicators.gpu_indicators import get_gpu_indicator_name
import importlib.util
from pathlib import Path as _Path

# Load the CPU real-time indicator calculator w/o importing package-level heavy deps
_indicator_path = _Path(__file__).resolve().parents[1] / 'src' / 'live_trading' / 'indicator_calculator.py'
spec = importlib.util.spec_from_file_location('src.live_trading.indicator_calculator', str(_indicator_path))
ic_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ic_mod)
RealTimeIndicatorCalculator = ic_mod.RealTimeIndicatorCalculator

# Helper: wrapper lines for call mapping (matches CompactBacktester id_to_call)
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


def is_close(a, b, tol=1e-3):
    return np.isclose(a, b, atol=tol, rtol=1e-6)


def main():
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    fetcher = DataFetcher(exchange_type='futures')
    file_paths = fetcher.fetch_data_range(pair='BTC/USDT', timeframe='1m', total_days=1)
    loader = DataLoader(file_paths=file_paths, timeframe='1m', gpu_context=ctx, gpu_queue=queue)
    ohlcv_df = loader.load_all_data()
    num_bars = len(ohlcv_df)
    print('Loaded bars:', num_bars)

    backtester = CompactBacktester(gpu_context=ctx, gpu_queue=queue, initial_balance=100.0)

    # Batched precompute (standard flow)
    buf_batched = backtester._precompute_indicators(ohlcv_df[['open','high','low','close','volume']].values.astype(np.float32))
    out_batched = np.empty(backtester.NUM_INDICATORS * num_bars, dtype=np.float32)
    cl.enqueue_copy(queue, out_batched, buf_batched)
    queue.finish()
    indicators_batched = out_batched.reshape((backtester.NUM_INDICATORS, num_bars))

    # CPU calculator values
    calc = RealTimeIndicatorCalculator(lookback_bars=max(500, num_bars))
    indicators_cpu = np.zeros_like(indicators_batched)
    for bar in range(num_bars):
        row = ohlcv_df.iloc[bar]
        calc.update_price_data(row['open'], row['high'], row['low'], row['close'], row['volume'])
        for idx in range(backtester.NUM_INDICATORS):
            indicators_cpu[idx, bar] = float(calc.calculate_indicator(idx, 0.0, 0.0, 0.0))

    # Build per-indicator isolated kernels and compare
    kernel_src_base = (ROOT / 'src' / 'gpu_kernels' / 'precompute_all_indicators.cl').read_text()

    results = []

    out_dir = Path('logs') / 'compare_3way'
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(backtester.NUM_INDICATORS):
        print(f'Processing indicator {idx}: {get_gpu_indicator_name(idx)}')
        call_line = ID_TO_CALL.get(idx, '')
        # Build isolated wrapper
        wrapper = '\n__kernel void precompute_subset(__global OHLCVBar *ohlcv, const int num_bars, __global float *indicators_out) {\n    if (get_global_id(0) == 0) {\n'
        wrapper += '        ' + call_line + '\n'
        wrapper += '    }\n}\n'
        kernel_src = kernel_src_base + '\n' + wrapper
        prg = cl.Program(ctx, kernel_src).build()
        subset_kernel = prg.precompute_subset

        ohlcv_flat = ohlcv_df[['open','high','low','close','volume']].values.astype(np.float32).flatten()
        mf = cl.mem_flags
        ohlcv_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ohlcv_flat)
        indicators_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=backtester.NUM_INDICATORS * num_bars * 4)

        subset_kernel(queue, (1,), None, ohlcv_buf, np.int32(num_bars), indicators_buf)
        queue.finish()

        out_isolated = np.empty(backtester.NUM_INDICATORS * num_bars, dtype=np.float32)
        cl.enqueue_copy(queue, out_isolated, indicators_buf)
        queue.finish()
        indicators_isolated = out_isolated.reshape((backtester.NUM_INDICATORS, num_bars))

        batched = indicators_batched[idx]
        isolated = indicators_isolated[idx]
        cpu = indicators_cpu[idx]

        # valid mask where neither side NaN
        mask_bi = ~np.isnan(batched) & ~np.isnan(isolated)
        mask_ic = ~np.isnan(isolated) & ~np.isnan(cpu)
        mask_bc = ~np.isnan(batched) & ~np.isnan(cpu)

        def pct_equal(a, b, mask):
            if not mask.any():
                return 100.0
            eq = is_close(a[mask], b[mask])
            return float((eq.sum() / mask.sum()) * 100.0)

        pct_bi = pct_equal(batched, isolated, mask_bi)
        pct_ic = pct_equal(isolated, cpu, mask_ic)
        pct_bc = pct_equal(batched, cpu, mask_bc)

        # Also produce per-indicator CSV of per-bar differences for deeper analysis
        csv_path = out_dir / f'3way_indicator_{idx}.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['bar', 'open', 'high', 'low', 'close', 'volume', 'batched', 'isolated', 'cpu', 'abs_bi', 'abs_ic', 'abs_bc'])
            for b in range(num_bars):
                ohlcv_row = ohlcv_df.iloc[b]
                bval = float(batched[b])
                ival = float(isolated[b])
                cval = float(cpu[b])
                writer.writerow([b, float(ohlcv_row['open']), float(ohlcv_row['high']), float(ohlcv_row['low']), float(ohlcv_row['close']), float(ohlcv_row['volume']), bval, ival, cval, abs(bval-ival), abs(ival-cval), abs(bval-cval)])

        results.append((idx, get_gpu_indicator_name(idx), pct_bi, pct_ic, pct_bc))

    # Write summary
    summary_path = out_dir / '3way_summary.csv'
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['idx', 'name', 'pct_batched_vs_isolated', 'pct_isolated_vs_cpu', 'pct_batched_vs_cpu'])
        for r in results:
            writer.writerow(r)

    # Sort and display low parity indicators per CPU vs GPU
    results_sorted = sorted(results, key=lambda x: x[4])  # by batched vs cpu
    print('\nTop indicators where batch-vs-cpu differs the most:')
    for idx, name, pbi, pic, pbc in results_sorted[:10]:
        print(f'{idx:02d} {name:30s} b_vs_i={pbi:6.2f}% i_vs_c={pic:6.2f}% b_vs_c={pbc:6.2f}%')

    print('\nDone. CSV outputs in', out_dir)

    # Also compute global parity metrics across all indicators and bars and append to a running CSV
    total_bi_valid = 0
    total_bi_eq = 0
    total_ic_valid = 0
    total_ic_eq = 0
    total_bc_valid = 0
    total_bc_eq = 0

    for idx in range(backtester.NUM_INDICATORS):
        batched = indicators_batched[idx]
        isolated = indicators_isolated[idx] if 'indicators_isolated' in locals() else None
        cpu = indicators_cpu[idx]
        mask_bi = ~np.isnan(batched) & ~np.isnan(isolated)
        mask_ic = ~np.isnan(isolated) & ~np.isnan(cpu)
        mask_bc = ~np.isnan(batched) & ~np.isnan(cpu)
        if mask_bi.any():
            total_bi_valid += int(mask_bi.sum())
            total_bi_eq += int(is_close(batched[mask_bi], isolated[mask_bi]).sum())
        if mask_ic.any():
            total_ic_valid += int(mask_ic.sum())
            total_ic_eq += int(is_close(isolated[mask_ic], cpu[mask_ic]).sum())
        if mask_bc.any():
            total_bc_valid += int(mask_bc.sum())
            total_bc_eq += int(is_close(batched[mask_bc], cpu[mask_bc]).sum())

    pct_bi_global = (total_bi_eq / total_bi_valid * 100.0) if total_bi_valid > 0 else 100.0
    pct_ic_global = (total_ic_eq / total_ic_valid * 100.0) if total_ic_valid > 0 else 100.0
    pct_bc_global = (total_bc_eq / total_bc_valid * 100.0) if total_bc_valid > 0 else 100.0

    stats_path = out_dir / 'stats.csv'
    import datetime
    now = datetime.datetime.utcnow().isoformat()
    header_needed = not stats_path.exists()
    with open(stats_path, 'a', newline='') as sf:
        sw = csv.writer(sf)
        if header_needed:
            sw.writerow(['timestamp', 'pct_batched_vs_isolated', 'pct_isolated_vs_cpu', 'pct_batched_vs_cpu'])
        sw.writerow([now, f"{pct_bi_global:.6f}", f"{pct_ic_global:.6f}", f"{pct_bc_global:.6f}"])

    print(f'Global parity: batched_vs_isolated={pct_bi_global:.6f}%, isolated_vs_cpu={pct_ic_global:.6f}%, batched_vs_cpu={pct_bc_global:.6f}%')


if __name__ == '__main__':
    main()
