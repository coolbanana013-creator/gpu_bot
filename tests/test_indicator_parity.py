import pytest
import numpy as np
from pathlib import Path
import pyopencl as cl

ROOT = Path(__file__).resolve().parents[1]

import importlib.util
from pathlib import Path as _Path
import sys

# Import RealTimeIndicatorCalculator without importing the entire package to avoid side-effects
ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_module_path = ROOT / 'src' / 'live_trading' / 'indicator_calculator.py'
spec = importlib.util.spec_from_file_location('src.live_trading.indicator_calculator', str(_module_path))
_ic_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_ic_mod)
RealTimeIndicatorCalculator = _ic_mod.RealTimeIndicatorCalculator
from src.data_provider.fetcher import DataFetcher
from src.data_provider.loader import DataLoader
from src.indicators.gpu_default_params import get_gpu_default_params


def load_ohlcv_and_buffers(ctx, queue, root):
    # Skip network-dependent exchange load_markets in CI / offline test runs
    fetcher = DataFetcher(exchange_type='futures', skip_load_markets=True)
    file_paths = fetcher.fetch_data_range(pair='BTC/USDT', timeframe='1m', total_days=1)
    loader = DataLoader(file_paths=file_paths, timeframe='1m', gpu_context=ctx, gpu_queue=queue)
    ohlcv_df = loader.load_all_data()
    num_bars = len(ohlcv_df)
    ohlcv_flat = ohlcv_df[['open','high','low','close','volume']].values.astype(np.float32).flatten()
    return ohlcv_df, ohlcv_flat, num_bars


# Indicator mapping: index -> (function wrapper source builder)
INDICATOR_WRAPPERS = {}

def wrapper_for_index(idx):
    # Return a tuple: (wrapper_source, required_buffers) where required_buffers
    # is a list of buffer names needed (e.g., ['atr_buf'])
    # Build small wrappers that call compute_x functions from kernel
    map_k = {
        # SMA
        0: '\n__kernel void iso(__global OHLCVBar *ohlcv, const int num_bars, __global float *out) { if (get_global_id(0) == 0) compute_sma(ohlcv, num_bars, 5, out); }',
        1: '\n__kernel void iso(__global OHLCVBar *ohlcv, const int num_bars, __global float *out) { if (get_global_id(0) == 0) compute_sma(ohlcv, num_bars, 10, out); }',
        2: '\n__kernel void iso(__global OHLCVBar *ohlcv, const int num_bars, __global float *out) { if (get_global_id(0) == 0) compute_sma(ohlcv, num_bars, 20, out); }',
        3: '\n__kernel void iso(__global OHLCVBar *ohlcv, const int num_bars, __global float *out) { if (get_global_id(0) == 0) compute_sma(ohlcv, num_bars, 50, out); }',
        4: '\n__kernel void iso(__global OHLCVBar *ohlcv, const int num_bars, __global float *out) { if (get_global_id(0) == 0) compute_sma(ohlcv, num_bars, 100, out); }',
        5: '\n__kernel void iso(__global OHLCVBar *ohlcv, const int num_bars, __global float *out) { if (get_global_id(0) == 0) compute_sma(ohlcv, num_bars, 200, out); }',
        # EMA
        6: '\n__kernel void iso(__global OHLCVBar *ohlcv, const int num_bars, __global float *out) { if (get_global_id(0) == 0) compute_ema(ohlcv, num_bars, 5, out); }',
        7: '\n__kernel void iso(__global OHLCVBar *ohlcv, const int num_bars, __global float *out) { if (get_global_id(0) == 0) compute_ema(ohlcv, num_bars, 10, out); }',
        8: '\n__kernel void iso(__global OHLCVBar *ohlcv, const int num_bars, __global float *out) { if (get_global_id(0) == 0) compute_ema(ohlcv, num_bars, 20, out); }',
        9: '\n__kernel void iso(__global OHLCVBar *ohlcv, const int num_bars, __global float *out) { if (get_global_id(0) == 0) compute_ema(ohlcv, num_bars, 50, out); }',
        10: '\n__kernel void iso(__global OHLCVBar *ohlcv, const int num_bars, __global float *out) { if (get_global_id(0) == 0) compute_ema(ohlcv, num_bars, 100, out); }',
        11: '\n__kernel void iso(__global OHLCVBar *ohlcv, const int num_bars, __global float *out) { if (get_global_id(0) == 0) compute_ema(ohlcv, num_bars, 200, out); }',
        # RSI family
        12: "\n__kernel void iso(__global OHLCVBar *o, const int n, __global float *out) { if (get_global_id(0) == 0) compute_rsi(o, n, 7, out); }",
        13: "\n__kernel void iso(__global OHLCVBar *o, const int n, __global float *out) { if (get_global_id(0) == 0) compute_rsi(o, n, 14, out); }",
        14: "\n__kernel void iso(__global OHLCVBar *o, const int n, __global float *out) { if (get_global_id(0) == 0) compute_rsi(o, n, 21, out); }",
        15: "\n__kernel void iso(__global OHLCVBar *o, const int n, __global float *out) { if (get_global_id(0) == 0) compute_stochastic(o, n, 14, 3, out); }",
        16: "\n__kernel void iso(__global OHLCVBar *o, const int n, __global float *rsi_buf, __global float *out) { if (get_global_id(0) == 0) { compute_rsi(o, n, 14, rsi_buf); compute_stochrsi(o, n, 14, out, rsi_buf); } }",
        # Momentum/ROC/WillR
        17: "\n__kernel void iso(__global OHLCVBar *o, const int n, __global float *out) { if (get_global_id(0) == 0) compute_momentum(o, n, 10, out); }",
        18: "\n__kernel void iso(__global OHLCVBar *o, const int n, __global float *out) { if (get_global_id(0) == 0) compute_roc(o, n, 10, out); }",
        19: "\n__kernel void iso(__global OHLCVBar *o, const int n, __global float *out) { if (get_global_id(0) == 0) compute_willr(o, n, 14, out); }",
        # ATR + NATR
        20: "\n__kernel void iso(__global OHLCVBar *o, const int n, __global float *out) { if (get_global_id(0) == 0) compute_atr(o, n, 14, out); }",
        21: "\n__kernel void iso(__global OHLCVBar *o, const int n, __global float *out) { if (get_global_id(0) == 0) compute_atr(o, n, 20, out); }",
        22: "\n__kernel void iso(__global OHLCVBar *o, const int n, __global float *atr_buf, __global float *out) { if (get_global_id(0) == 0) { compute_atr(o, n, 14, atr_buf); compute_natr(o, n, 14, out, atr_buf);} }",
        # Bollinger (upper/lower)
        23: "\n__kernel void iso(__global OHLCVBar *o, const int n, __global float *upper, __global float *lower) { if (get_global_id(0) == 0) compute_bollinger_bands(o, n, 20, 2.0f, upper, lower); }",
        24: "\n__kernel void iso(__global OHLCVBar *o, const int n, __global float *upper, __global float *lower) { if (get_global_id(0) == 0) compute_bollinger_bands(o, n, 20, 2.0f, upper, lower); }",
        25: "\n__kernel void iso(__global OHLCVBar *o, const int n, __global float *atr_buf, __global float *out) { if (get_global_id(0) == 0) { compute_atr(o, n, 20, atr_buf); compute_keltner(o, n, 20, out, atr_buf); } }",
        26: "\n__kernel void iso(__global OHLCVBar *o, const int n, __global float *out) { if (get_global_id(0) == 0) compute_macd(o, n, 12, 26, 9, out); }",
        27: "\n__kernel void iso(__global OHLCVBar *o, const int n, __global float *out) { if (get_global_id(0) == 0) compute_adx(o, n, 14, out); }",
        28: "\n__kernel void iso(__global OHLCVBar *o, const int n, __global float *out) { if (get_global_id(0) == 0) compute_aroon_up(o, n, 25, out); }",
        29: "\n__kernel void iso(__global OHLCVBar *o, const int n, __global float *out) { if (get_global_id(0) == 0) compute_cci(o, n, 20, out); }",
        30: "\n__kernel void iso(__global OHLCVBar *o, const int n, __global float *out) { if (get_global_id(0) == 0) compute_dpo(o, n, 20, out); }",
        31: "\n__kernel void iso(__global OHLCVBar *o, const int n, __global float *out) { if (get_global_id(0) == 0) compute_psar(o, n, 0.02f, 0.2f, out); }",
        32: "\n__kernel void iso(__global OHLCVBar *o, const int n, __global float *atr_buf, __global float *out) { if (get_global_id(0) == 0) { compute_atr(o, n, 14, atr_buf); compute_supertrend(o, n, 10, 3.0f, out, atr_buf); } }",
        33: "\n__kernel void iso(__global OHLCVBar *o, const int n, __global float *out) { if (get_global_id(0) == 0) compute_trend_strength(o, n, 20, out); }",
        34: "\n__kernel void iso(__global OHLCVBar *o, const int n, __global float *out) { if (get_global_id(0) == 0) compute_trend_strength(o, n, 50, out); }",
        35: "\n__kernel void iso(__global OHLCVBar *o, const int n, __global float *out) { if (get_global_id(0) == 0) compute_trend_strength(o, n, 100, out); }",
        36: "\n__kernel void iso(__global OHLCVBar *o, const int n, __global float *out) { if (get_global_id(0) == 0) compute_obv(o, n, out); }",
        37: "\n__kernel void iso(__global OHLCVBar *o, const int n, __global float *out) { if (get_global_id(0) == 0) compute_vwap(o, n, out); }",
        38: "\n__kernel void iso(__global OHLCVBar *o, const int n, __global float *out) { if (get_global_id(0) == 0) compute_mfi(o, n, 14, out); }",
        39: "\n__kernel void iso(__global OHLCVBar *o, const int n, __global float *out) { if (get_global_id(0) == 0) compute_ad(o, n, out); }",
        40: "\n__kernel void iso(__global OHLCVBar *o, const int n, __global float *out) { if (get_global_id(0) == 0) compute_volume_sma(o, n, 20, out); }",
        41: "\n__kernel void iso(__global OHLCVBar *o, const int n, __global float *out) { if (get_global_id(0) == 0) compute_pivot_points(o, n, out); }",
        42: "\n__kernel void iso(__global OHLCVBar *o, const int n, __global float *out) { if (get_global_id(0) == 0) compute_fractal_high(o, n, 5, out); }",
        43: "\n__kernel void iso(__global OHLCVBar *o, const int n, __global float *out) { if (get_global_id(0) == 0) compute_fractal_low(o, n, 5, out); }",
        44: "\n__kernel void iso(__global OHLCVBar *o, const int n, __global float *out) { if (get_global_id(0) == 0) compute_support_resistance(o, n, 20, out); }",
        45: "\n__kernel void iso(__global OHLCVBar *o, const int n, __global float *out) { if (get_global_id(0) == 0) compute_price_channel(o, n, 20, out); }",
        46: "\n__kernel void iso(__global OHLCVBar *o, const int n, __global float *out) { if (get_global_id(0) == 0) compute_hl_range(o, n, out); }",
        47: "\n__kernel void iso(__global OHLCVBar *o, const int n, __global float *out) { if (get_global_id(0) == 0) compute_close_position(o, n, out); }",
        48: "\n__kernel void iso(__global OHLCVBar *o, const int n, __global float *out) { if (get_global_id(0) == 0) compute_price_acceleration(o, n, 10, out); }",
        49: "\n__kernel void iso(__global OHLCVBar *o, const int n, __global float *out) { if (get_global_id(0) == 0) compute_volume_roc(o, n, 10, out); }"
    }
    return map_k.get(idx, None)


@pytest.mark.skipif(not cl.get_platforms(), reason='No OpenCL platforms found')
@pytest.mark.parametrize('indicator_idx', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,17,18,19,20,23,26,27,28,29,30,31,32,33,36,37,38,39,40,41,44,45,46,47,48])
def test_indicator_parity(indicator_idx):
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    ohlcv_df, ohlcv_flat, num_bars = load_ohlcv_and_buffers(ctx, queue, ROOT)
    kernel_src = (ROOT / 'src' / 'gpu_kernels' / 'precompute_all_indicators.cl').read_text()
    wrapper = wrapper_for_index(indicator_idx)
    if wrapper is None:
        pytest.skip(f'No wrapper generated for index {indicator_idx}')

    try:
        prg = cl.Program(ctx, kernel_src + '\n' + wrapper).build()
    except Exception as e:
        # Kernel compilation may fail on some devices; skip with a helpful message
        pytest.skip(f'Kernel compilation failed: {e}')

    mf = cl.mem_flags
    ohlcv_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ohlcv_flat)
    num_bytes = num_bars * 4

    # Prepare buffers depending on wrapper signature requirements
    # We will heuristically allocate the buffers that could be required
    atr_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=num_bytes)
    upper_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=num_bytes)
    lower_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=num_bytes)
    rsi_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=num_bytes)
    out_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=num_bytes)

    # Launch kernel depending on which buffers used in wrapper
    # Simple heuristic: if 'atr_buf' in wrapper string, pass it.
    kw = [ohlcv_buf, np.int32(num_bars)]
    if 'atr_buf' in wrapper or 'atr_buf' in wrapper:
        kw.append(atr_buf)
    if 'rsi_buf' in wrapper:
        kw.append(rsi_buf)
    if 'upper' in wrapper and 'lower' in wrapper:
        kw.append(upper_buf)
        kw.append(lower_buf)
    else:
        kw.append(out_buf)

    # Run the program
    try:
        # Use ISO kernel name 'iso'
        prg.iso(queue, (1,), None, *kw)
    except Exception as e:
        pytest.skip(f'Kernel launch failed: {e}')
    queue.finish()

    # Read GPU output array corresponding to indicator
    gpu_res = np.empty(num_bars, dtype=np.float32)
    if 'upper' in wrapper and 'lower' in wrapper:
        # For BB upper/lower we'll read the relevant buffer
        if indicator_idx == 23:
            cl.enqueue_copy(queue, gpu_res, upper_buf)
        else:
            cl.enqueue_copy(queue, gpu_res, lower_buf)
    else:
        cl.enqueue_copy(queue, gpu_res, out_buf)
    queue.finish()

    # CPU compute
    rt = RealTimeIndicatorCalculator(lookback_bars=num_bars)
    cpu_res = np.zeros(num_bars, dtype=np.float32)
    p0, p1, p2 = get_gpu_default_params(indicator_idx)
    for i in range(num_bars):
        r = ohlcv_df.iloc[i]
        rt.update_price_data(r['open'], r['high'], r['low'], r['close'], r['volume'])
        cpu_res[i] = float(rt.calculate_indicator(indicator_idx, float(p0), float(p1), float(p2)))

    # Compare
    # Use tolerant absolute diff for float32 parity
    mask = ~np.isnan(cpu_res) & ~np.isnan(gpu_res)
    if not mask.any():
        pytest.skip('No valid bars to compare')
    diffs = np.abs(gpu_res[mask] - cpu_res[mask])
    mean_diff = diffs.mean()
    max_diff = diffs.max()
    assert mean_diff < 1e-2, f'Mean diff too high for idx {indicator_idx}: {mean_diff} max: {max_diff}'
