#!/usr/bin/env python3
"""
Compare batched GPU OBV (indicator 36) to CPU RealTimeIndicatorCalculator OBV and isolated kernel OBV for a dataset.
Print per-bar values for inspection.
"""
import sys
from pathlib import Path
import numpy as np
import pyopencl as cl
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.backtester.compact_simulator import CompactBacktester
from src.data_provider.fetcher import DataFetcher
from src.data_provider.loader import DataLoader
import importlib.util
from pathlib import Path as _Path

# Load RealTimeIndicatorCalculator without package imports
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
    ohlcv = loader.load_all_data()
    num_bars = len(ohlcv)
    print('Loaded bars:', num_bars)

    backtester = CompactBacktester(gpu_context=ctx, gpu_queue=queue, initial_balance=100.0)
    buf = backtester._precompute_indicators(ohlcv[['open','high','low','close','volume']].values.astype(np.float32))
    out = np.empty(backtester.NUM_INDICATORS * num_bars, dtype=np.float32)
    cl.enqueue_copy(queue, out, buf)
    queue.finish()
    indicators = out.reshape((backtester.NUM_INDICATORS, num_bars))

    # CPU calc
    calc = RealTimeIndicatorCalculator(lookback_bars=max(500, num_bars))

    for bar in range(num_bars):
        row = ohlcv.iloc[bar]
        calc.update_price_data(row['open'], row['high'], row['low'], row['close'], row['volume'])
        gpu_obv = float(indicators[36, bar])
        cpu_obv = float(calc.calculate_indicator(36, 0.0, 0.0, 0.0))
        print(f'Bar {bar:3d}: open {row["open"]} close {row["close"]} vol {row["volume"]} GPU_OBV: {gpu_obv} CPU_OBV: {cpu_obv} diff: {abs(gpu_obv-cpu_obv)}')


if __name__ == '__main__':
    main()
