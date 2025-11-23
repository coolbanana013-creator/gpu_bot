#!/usr/bin/env python3
# Moved VWAP inspector to scripts/debug; ROOT adjusted
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pyopencl as cl
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
    num_bars = len(ohlcv_df)
    # GPU VWAP
    kernel_src = (ROOT / 'src' / 'gpu_kernels' / 'precompute_all_indicators.cl').read_text()
    wrapper = f"\n__kernel void iso(__global OHLCVBar *o, const int n, __global float *out) {{ if (get_global_id(0) == 0) compute_vwap(o, n, out); }}"
    prg = cl.Program(ctx, kernel_src + '\n' + wrapper).build()
    ohlcv_flat = ohlcv_df[['open','high','low','close','volume']].values.astype(np.float32).flatten()
    mf = cl.mem_flags
    ohlcv_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ohlcv_flat)
    out_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=num_bars * 4)
    prg.iso(queue, (1,), None, ohlcv_buf, np.int32(num_bars), out_buf)
    queue.finish()
    gpu_res = np.empty(num_bars, dtype=np.float32)
    cl.enqueue_copy(queue, gpu_res, out_buf)
    # CPU
    calc = RealTimeIndicatorCalculator()
    cpu_res = np.full(num_bars, np.nan, dtype=np.float32)
    for i in range(num_bars):
        bar = ohlcv_df.iloc[i]
        calc.update_price_data(bar['open'], bar['high'], bar['low'], bar['close'], bar['volume'])
        cpu_res[i] = calc.calculate_indicator(37, 0.0, 0.0, 0.0)
    mask = ~np.isnan(cpu_res) & ~np.isnan(gpu_res)
    diffs = np.abs(gpu_res - cpu_res)
    print('Num bars:', num_bars, 'Mismatches > 0.0009:', np.sum((diffs > 0.0009) & mask))
    m_idx = np.where((diffs > 0.0009) & mask)[0]
    if len(m_idx) > 0:
        for idx in m_idx[:20]:
            print(idx, 'gpu', gpu_res[idx], 'cpu', cpu_res[idx], 'diff', diffs[idx])
    else:
        print('No mismatches under threshold')
    for i in range(10, 40):
        print(i, 'gpu', gpu_res[i], 'cpu', cpu_res[i], 'diff', abs(gpu_res[i] - cpu_res[i]))
