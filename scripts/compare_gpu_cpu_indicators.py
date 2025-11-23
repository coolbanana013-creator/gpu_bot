#!/usr/bin/env python3
"""
Compare GPU precomputed indicators (precompute_all_indicators.cl) to CPU RealTimeIndicatorCalculator
for a short dataset and one sample bot. Useful to verify parity across all 50 indicators per bar.
"""
import sys
from pathlib import Path
import numpy as np
import pyopencl as cl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bot_generator.compact_generator import CompactBotGenerator
from src.backtester.compact_simulator import CompactBacktester
import importlib.util
from pathlib import Path as _Path

# Avoid importing the entire package (which may pull in network SDKs) by loading the
# indicator_calculator module directly from its file path. This prevents package
# level imports such as KuCoin SDKs from blocking local diagnostics.
_indicator_path = _Path(__file__).resolve().parents[1] / 'src' / 'live_trading' / 'indicator_calculator.py'
spec = importlib.util.spec_from_file_location('src.live_trading.indicator_calculator', str(_indicator_path))
ic_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ic_mod)
RealTimeIndicatorCalculator = ic_mod.RealTimeIndicatorCalculator
from src.data_provider.fetcher import DataFetcher
from src.data_provider.loader import DataLoader
import os
import csv


def main():
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    # Create a generator and get a single bot with filled params
    gen = CompactBotGenerator(gpu_context=ctx, gpu_queue=queue, population_size=1, min_indicators=8, max_indicators=8)
    bot = gen.generate_population()[0]
    print('Bot id', bot.bot_id)
    print('Indicators', bot.indicator_indices[:bot.num_indicators])
    print('Params', bot.indicator_params[:bot.num_indicators])

    # Fetch small dataset
    fetcher = DataFetcher(exchange_type='futures')
    file_paths = fetcher.fetch_data_range(pair='BTC/USDT', timeframe='1m', total_days=1)
    loader = DataLoader(file_paths=file_paths, timeframe='1m', gpu_context=ctx, gpu_queue=queue)
    ohlcv = loader.load_all_data()
    num_bars = len(ohlcv)
    print('Loaded bars:', num_bars)

    backtester = CompactBacktester(gpu_context=ctx, gpu_queue=queue, initial_balance=100.0)
    indicators_buf = backtester._precompute_indicators(ohlcv[['open','high','low','close','volume']].values.astype(np.float32))

    # Read back indicators into numpy array
    indicators_flat = np.empty(backtester.NUM_INDICATORS * num_bars, dtype=np.float32)
    cl.enqueue_copy(queue, indicators_flat, indicators_buf)
    queue.finish()
    indicators_gpu = indicators_flat.reshape((backtester.NUM_INDICATORS, num_bars))

    # CPU calculator: compute each indicator bar-by-bar using the bot's parameters
    calc = RealTimeIndicatorCalculator(lookback_bars=max(500, num_bars))
    indicators_cpu = np.zeros_like(indicators_gpu)

    # Populate calculator sequentially and compute all indicators per bar
    for bar in range(num_bars):
        row = ohlcv.iloc[bar]
        calc.update_price_data(row['open'], row['high'], row['low'], row['close'], row['volume'])

        for idx in range(backtester.NUM_INDICATORS):
            # Find if the bot uses this indicator, else skip CPU compute for unused indicators
            # We'll compute CPU indicator values for all indices for comparison though.
            # Use bot params for this index when available; fallback to default mapping
            params = (0.0, 0.0, 0.0)
            found_slot = None
            for i in range(bot.num_indicators):
                if int(bot.indicator_indices[i]) == idx:
                    found_slot = i
                    params = tuple(float(x) for x in bot.indicator_params[i])
                    break

            p0, p1, p2 = params
            try:
                v = calc.calculate_indicator(idx, p0, p1, p2)
            except Exception as e:
                v = float('nan')
            indicators_cpu[idx, bar] = v

    # Compare GPU vs CPU
    mismatches = []
    for idx in range(backtester.NUM_INDICATORS):
        gpu_vals = indicators_gpu[idx]
        cpu_vals = indicators_cpu[idx]
        # Flatten NaNs and compare with tolerance
        valid_mask = ~np.isnan(cpu_vals) & ~np.isnan(gpu_vals)
        if not valid_mask.any():
            continue
        diffs = np.abs(gpu_vals[valid_mask] - cpu_vals[valid_mask])
        max_diff = diffs.max() if diffs.size > 0 else 0.0
        mean_diff = diffs.mean() if diffs.size > 0 else 0.0
        if max_diff > 1e-6:  # tolerance - can be tuned
            mismatches.append((idx, mean_diff, max_diff, int(valid_mask.sum())))

    # Sort mismatches by max_diff desc
    mismatches.sort(key=lambda x: x[2], reverse=True)

    # Create logs dir
    out_dir = Path('logs') / 'compare_mismatches'
    os.makedirs(out_dir, exist_ok=True)

    # Write a CSV summary of mismatches
    summary_path = out_dir / 'mismatch_summary.csv'
    with open(summary_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['indicator', 'mean_diff', 'max_diff', 'samples'])
        for idx, mean_diff, max_diff, samples in mismatches:
            writer.writerow([idx, mean_diff, max_diff, samples])

    if not mismatches:
        print('OK: GPU and CPU indicators match within tolerance for all indicators')
    else:
        print('MISMATCHES FOUND:')
        for m in mismatches:
            idx, mean_diff, max_diff, samples = m
            print(f'Indicator {idx}: mean_diff={mean_diff:.6f}, max_diff={max_diff:.6f}, samples={samples}')
    # Dump per-bar diffs for top N mismatches
    top_n = 5
    for idx, mean_diff, max_diff, samples in mismatches[:top_n]:
        gpu_vals = indicators_gpu[idx]
        cpu_vals = indicators_cpu[idx]
        valid_mask = ~np.isnan(cpu_vals) & ~np.isnan(gpu_vals)
        out_path = out_dir / f'mismatch_indicator_{idx}.csv'
        with open(out_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['bar', 'open', 'high', 'low', 'close', 'volume', 'gpu_value', 'cpu_value', 'diff'])
            for bar_index, valid in enumerate(valid_mask):
                if not valid:
                    continue
                g = gpu_vals[bar_index]
                c = cpu_vals[bar_index]
                o = float(ohlcv.iloc[bar_index]['open'])
                h = float(ohlcv.iloc[bar_index]['high'])
                l = float(ohlcv.iloc[bar_index]['low'])
                close_val = float(ohlcv.iloc[bar_index]['close'])
                v = float(ohlcv.iloc[bar_index]['volume'])
                writer.writerow([bar_index, o, h, l, close_val, v, g, c, abs(g - c)])
        print(f'Wrote per-bar CSV for indicator {idx} to {out_path}')

    print('Done')


if __name__ == '__main__':
    main()
