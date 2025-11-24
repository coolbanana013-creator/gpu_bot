#!/usr/bin/env python3
import numpy as np
from pathlib import Path
import pyopencl as cl
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data_provider.fetcher import DataFetcher
from src.data_provider.loader import DataLoader

if __name__ == '__main__':
	ctx = cl.create_some_context(interactive=False)
	queue = cl.CommandQueue(ctx)
	fetcher = DataFetcher(exchange_type='futures')
	file_paths = fetcher.fetch_data_range(pair='BTC/USDT', timeframe='1m', total_days=1)
	loader = DataLoader(file_paths=file_paths, timeframe='1m', gpu_context=ctx, gpu_queue=queue)
	ohlcv_df = loader.load_all_data()
    
	period = 14
	print(f'Testing ADX with period={period}')
    
	# Compute GPU ADX using isolated kernel
	kernel_src = (ROOT / 'src' / 'gpu_kernels' / 'precompute_all_indicators.cl').read_text()
	wrapper = f'\n__kernel void iso(__global OHLCVBar *o, const int n, __global float *out) {{ if (get_global_id(0) == 0) compute_adx(o, n, {period}, out); }}'
	prg = cl.Program(ctx, kernel_src + '\n' + wrapper).build()
	ohlcv_flat = ohlcv_df[['open','high','low','close','volume']].values.astype(np.float32).flatten()
	num_bars = len(ohlcv_df)
	mf = cl.mem_flags
	ohlcv_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ohlcv_flat)
	out_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=num_bars * 4)
	prg.iso(queue, (1,), None, ohlcv_buf, np.int32(num_bars), out_buf)
	queue.finish()
	gpu_adx = np.empty(num_bars, dtype=np.float32)
	cl.enqueue_copy(queue, gpu_adx, out_buf)
    
	# Compute CPU ADX
	from src.live_trading.indicator_calculator import RealTimeIndicatorCalculator
	calc = RealTimeIndicatorCalculator()
	cpu_adx = np.full(num_bars, np.nan, dtype=np.float32)
	for i in range(num_bars):
		bar = ohlcv_df.iloc[i]
		calc.update_price_data(bar['open'], bar['high'], bar['low'], bar['close'], bar['volume'])
		cpu_adx[i] = calc.calculate_indicator(27, 0.0, 0.0, 0.0)
    
	print(f'GPU ADX sample at period*2={period*2}: {gpu_adx[period*2]}')
	print(f'CPU ADX sample at period*2={period*2}: {cpu_adx[period*2]}')
    
	# Print some values
	print(f'\nbar | gpu_adx | gpu_nan | cpu_adx | cpu_nan | diff')
	for i in range(20, 50):
		gpu_nan = np.isnan(gpu_adx[i])
		cpu_nan = np.isnan(cpu_adx[i])
		if not gpu_nan and not cpu_nan:
			diff = abs(gpu_adx[i] - cpu_adx[i])
			print(f'{i:3d} | {gpu_adx[i]:10.4f} | {gpu_nan} | {cpu_adx[i]:10.4f} | {cpu_nan} | {diff:10.4f}')
		else:
			print(f'{i:3d} | {gpu_adx[i]:10.4f} | {gpu_nan} | {cpu_adx[i]:10.4f} | {cpu_nan} | N/A')

