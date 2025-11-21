"""
Verify that precomputed indicators (GPU) match a Python reference computation for VWAP and OBV on a small synthetic dataset.
"""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pyopencl as cl
from src.backtester.compact_simulator import CompactBacktester


def py_vwap(ohlcv_no_ts, bar):
    cum_tp_vol = 0.0
    cum_vol = 0.0
    for b in range(0, bar + 1):
        tp_b = (ohlcv_no_ts[b, 1] + ohlcv_no_ts[b, 2] + ohlcv_no_ts[b, 3]) / 3.0
        v_b = float(ohlcv_no_ts[b, 4])
        cum_tp_vol += tp_b * v_b
        cum_vol += v_b
    return cum_tp_vol / cum_vol if cum_vol > 0 else float(ohlcv_no_ts[bar, 3])


def py_obv(ohlcv_no_ts, bar):
    obv = 0.0
    for b in range(1, bar + 1):
        if ohlcv_no_ts[b, 3] > ohlcv_no_ts[b - 1, 3]:
            obv += float(ohlcv_no_ts[b, 4])
        elif ohlcv_no_ts[b, 3] < ohlcv_no_ts[b - 1, 3]:
            obv -= float(ohlcv_no_ts[b, 4])
    return obv


def test_gpu_precompute_vwap_obv():
    # Skip test if no GPU found to avoid false negatives in CPU-only environments
    platforms = cl.get_platforms()
    if len(platforms) == 0:
        pytest.skip("No OpenCL platforms found")
    # Find a GPU device if available, otherwise skip
    devices = None
    for p in platforms:
        devs = [d for d in p.get_devices() if d.type == cl.device_type.GPU]
        if devs:
            devices = devs
            break
    if not devices:
        pytest.skip("No OpenCL GPU devices found on this machine")
    ctx = cl.Context(devices=[devices[0]])
    queue = cl.CommandQueue(ctx)

    backtester = CompactBacktester(gpu_context=ctx, gpu_queue=queue, initial_balance=10.0)

    np.random.seed(123)
    # synthetic OHLCV: open, high, low, close, volume
    num_bars = 1440
    price = 50000.0 + np.cumsum(np.random.randn(num_bars).astype(np.float32))
    ohlcv_no_ts = np.zeros((num_bars, 5), dtype=np.float32)
    ohlcv_no_ts[:, 0] = price
    ohlcv_no_ts[:, 1] = price * 1.001
    ohlcv_no_ts[:, 2] = price * 0.999
    ohlcv_no_ts[:, 3] = price
    ohlcv_no_ts[:, 4] = np.random.randint(1000, 1000000, size=num_bars).astype(np.float32)

    indicators_buf = backtester._precompute_indicators(ohlcv_no_ts)

    # Read back indicators
    num_indicators = backtester.NUM_INDICATORS
    indicators_flat = np.empty(num_indicators * num_bars, dtype=np.float32)
    cl.enqueue_copy(queue, indicators_flat, indicators_buf)
    indicators_flat = indicators_flat.reshape((num_indicators, num_bars))
    queue.finish()

    # Check VWAP (index 37) and OBV (index 36) at random bars
    for bar in [10, 100, 500, 1000, 1439]:
        gpu_vwap = indicators_flat[37, bar]
        gpu_obv = indicators_flat[36, bar]
        py_v = py_vwap(ohlcv_no_ts, bar)
        py_o = py_obv(ohlcv_no_ts, bar)

        # Ensure VWAP close to python (tolerance due to float rounding)
        assert np.isfinite(gpu_vwap), f"GPU VWAP NaN at bar {bar}"
        assert abs(gpu_vwap - py_v) < 1e-2 * max(1.0, abs(py_v)), f"VWAP mismatch at bar {bar}: GPU={gpu_vwap} vs PY={py_v}"

        # OBV is cumulative; tolerances larger; use relative tolerance
        assert np.isfinite(gpu_obv), f"GPU OBV NaN at bar {bar}"
        assert abs(gpu_obv - py_o) < 1e-3 * max(1.0, abs(py_o)), f"OBV mismatch at bar {bar}: GPU={gpu_obv} vs PY={py_o}"


if __name__ == '__main__':
    test_gpu_precompute_vwap_obv()
    print('Vwap/OBV precompute checks passed')
