#!/usr/bin/env python3
# Moved debug script: was at scripts/inspect_adx_values.py
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data_provider.fetcher import DataFetcher
from src.data_provider.loader import DataLoader
from src.live_trading.indicator_calculator import RealTimeIndicatorCalculator
import numpy as np
import pyopencl as cl

if __name__ == '__main__':
    # existing code starts here (kept intact)
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    fetcher = DataFetcher(exchange_type='futures')
    file_paths = fetcher.fetch_data_range(pair='BTC/USDT', timeframe='1m', total_days=1)
    loader = DataLoader(file_paths=file_paths, timeframe='1m', gpu_context=ctx, gpu_queue=queue)
    ohlcv_df = loader.load_all_data()
    period = 14
    print(f'Testing ADX with period={period}')
    kernel_src = (ROOT / 'src' / 'gpu_kernels' / 'precompute_all_indicators.cl').read_text()
    wrapper = f'\n__kernel void iso(__global OHLCVBar *o, const int n, __global float *out) {{ if (get_global_id(0) == 0) compute_adx(o, n, {period}, out); }}'
    prg = cl.Program(ctx, kernel_src + '\n' + wrapper).build()
    ohlcv_flat = ohlcv_df[['open','high','low','close','volume']].values.astype(np.float32).flatten()
    mf = cl.mem_flags
    ohlcv_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ohlcv_flat)
    num_bars = len(ohlcv_df)
    out_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=num_bars * 4)
    prg.iso(queue, (1,), None, ohlcv_buf, np.int32(num_bars), out_buf)
    queue.finish()
    gpu_adx = np.empty(num_bars, dtype=np.float32)
    cl.enqueue_copy(queue, gpu_adx, out_buf)
    from src.live_trading.indicator_calculator import RealTimeIndicatorCalculator
    calc = RealTimeIndicatorCalculator()
    cpu_adx = np.full(num_bars, np.nan, dtype=np.float32)
    for i in range(num_bars):
        bar = ohlcv_df.iloc[i]
        calc.update_price_data(bar['open'], bar['high'], bar['low'], bar['close'], bar['volume'])
        cpu_adx[i] = calc.calculate_indicator(27, 0.0, 0.0, 0.0)
    print('GPU sample', gpu_adx[period*2-1], 'CPU sample', cpu_adx[period*2-1])
