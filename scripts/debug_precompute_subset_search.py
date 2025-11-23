import re
import textwrap
import numpy as np
import pyopencl as cl
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data_provider.fetcher import DataFetcher
from src.data_provider.loader import DataLoader
from src.backtester.compact_simulator import CompactBacktester

KERNEL_PATH = Path('src') / 'gpu_kernels' / 'precompute_all_indicators.cl'
KERNEL_SRC = KERNEL_PATH.read_text()

# Extract the switch block
start = KERNEL_SRC.find('switch(indicator_id)')
if start == -1:
    raise RuntimeError('Cannot find switch block in kernel')
brace_index = KERNEL_SRC.find('{', start)
if brace_index == -1:
    raise RuntimeError('Cannot find opening brace for switch block')

# Find matching closing brace by counting braces
pos = brace_index + 1
level = 1
while pos < len(KERNEL_SRC) and level > 0:
    if KERNEL_SRC[pos] == '{':
        level += 1
    elif KERNEL_SRC[pos] == '}':
        level -= 1
    pos += 1
if level != 0:
    raise RuntimeError('Unbalanced braces in kernel switch parsing')

switch_start = KERNEL_SRC.rfind('\n', 0, start) + 1
switch_end = pos
original_switch = KERNEL_SRC[switch_start:switch_end]
cases_block = KERNEL_SRC[brace_index + 1:pos - 1]

# Helper function to build new kernel for a subset

def build_subset_kernel(selected_cases):
    # Build a new kernel source by reusing all function definitions and adding
    # a custom kernel that calls only the selected indicators directly.
    # Construct a kernel that invokes specific compute_* functions for each index;
    call_lines = []
    # Map cases to their direct calling forms (hardcode common calls from original file)
    case_call_map = {
        # Moving averages coverage
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
    for c in selected_cases:
        if c in case_call_map:
            call_lines.append('    ' + case_call_map[c])
    # Build the new kernel wrapper that runs all selected calls sequentially in a single work item
    wrapper = '\n__kernel void precompute_subset(__global OHLCVBar *ohlcv, const int num_bars, __global float *indicators_out) {\n    if (get_global_id(0) == 0) {\n'
    for l in call_lines:
        wrapper += l + '\n'
    wrapper += '    }\n}\n'
    new_src = KERNEL_SRC + '\n' + wrapper
    return new_src


# We'll create a kernel with a few different subsets and test OBV for corruption

def run_kernel_with_src(kernel_src, subset_ids, ohlcv):
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    prg = cl.Program(ctx, kernel_src).build()
    # Our appended kernel is precompute_subset
    kernel = prg.precompute_subset

    num_bars = len(ohlcv)
    ohlcv_flat = ohlcv[['open','high','low','close','volume']].values.astype(np.float32).flatten()
    mf = cl.mem_flags
    buf_ohlcv = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ohlcv_flat)
    indicators_size = len(subset_ids) * num_bars
    # To avoid confusion, create a small indicator buffer of size 50 indicators as before
    full_indicators_size = 50 * num_bars
    buf_ind = cl.Buffer(ctx, mf.WRITE_ONLY, size=full_indicators_size * 4)

    # global_size as before
    global_size = (1,)
    local_size = None

    kernel(queue, global_size, local_size, buf_ohlcv, np.int32(num_bars), buf_ind)
    queue.finish()

    indicators_flat = np.empty(50 * num_bars, dtype=np.float32)
    cl.enqueue_copy(queue, indicators_flat, buf_ind)
    queue.finish()
    indicators_out = indicators_flat.reshape((50, num_bars))
    return indicators_out


# Start testing different subsets - but to keep it fast, we'll test groups of indicators in halves
if __name__ == '__main__':
    # Load data
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    fetcher = DataFetcher(exchange_type='futures')
    file_paths = fetcher.fetch_data_range(pair='BTC/USDT', timeframe='1m', total_days=1)
    loader = DataLoader(file_paths=file_paths, timeframe='1m', gpu_context=ctx, gpu_queue=queue)
    ohlcv = loader.load_all_data()

    # Candidate initial subset include 0 & 36 (works), and other sets to add
    # We'll test adding indicators in ranges until OBV corruption appears
    half = list(range(0, 25))
    other_half = list(range(25, 50))

    print('Testing base subset [0,36] to ensure baseline')
    new_src = build_subset_kernel([0, 36])
    ind = run_kernel_with_src(new_src, [0, 36], ohlcv)
    print('OBV (36) first 10', ind[36,:10])

    # Now test adding half-range of indicators
    print('Testing adding indicators 1-12 with OBV')
    subset = [0,36] + list(range(1,13))
    new_src = build_subset_kernel(subset)
    ind = run_kernel_with_src(new_src, subset, ohlcv)
    print('OBV (36) first 10', ind[36,:10])

    print('Testing adding indicators 13-25 with OBV')
    subset = [0,36] + list(range(13,26))
    new_src = build_subset_kernel(subset)
    ind = run_kernel_with_src(new_src, subset, ohlcv)
    print('OBV (36) first 10', ind[36,:10])

    print('Testing adding indicators 26-39 with OBV')
    subset = [0,36] + list(range(26,40))
    new_src = build_subset_kernel(subset)
    ind = run_kernel_with_src(new_src, subset, ohlcv)
    print('OBV (36) first 10', ind[36,:10])

    print('Testing adding indicators 40-49 with OBV')
    subset = [0,36] + list(range(40,50))
    new_src = build_subset_kernel(subset)
    ind = run_kernel_with_src(new_src, subset, ohlcv)
    print('OBV (36) first 10', ind[36,:10])

    print('\nNow try narrowing down the block causing corruption (if any), run repeated splits as needed (manual analysis suggested)')
