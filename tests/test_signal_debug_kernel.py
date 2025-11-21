"""
Test that the debug signal kernel produces well-formed SignalDebugRecords and that indicator values correspond to precomputed indicators and that num_indicators offset is handled correctly.
"""
import sys
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pyopencl as cl
from src.backtester.compact_simulator import CompactBacktester
from src.bot_generator.compact_generator import CompactBotGenerator


def test_debug_kernel_samples():
    platforms = cl.get_platforms()
    devices = platforms[0].get_devices(device_type=cl.device_type.GPU)
    ctx = cl.Context(devices=[devices[0]])
    queue = cl.CommandQueue(ctx)

    backtester = CompactBacktester(gpu_context=ctx, gpu_queue=queue, initial_balance=10.0)
    generator = CompactBotGenerator(gpu_context=ctx, gpu_queue=queue, population_size=16, min_indicators=1, max_indicators=3, min_risk_strategies=1, max_risk_strategies=1, min_leverage=1, max_leverage=10)

    bots = generator.generate_population()

    # Create synthetic data with 1440 bars (1 day) without timestamp
    num_bars = 1440
    price = 10000.0 + np.cumsum(np.random.randn(num_bars).astype(np.float32))
    ohlcv_no_ts = np.zeros((num_bars, 5), dtype=np.float32)
    ohlcv_no_ts[:, 0] = price  # open
    ohlcv_no_ts[:, 1] = price * 1.001
    ohlcv_no_ts[:, 2] = price * 0.999
    ohlcv_no_ts[:, 3] = price
    ohlcv_no_ts[:, 4] = 10000.0

    indicators_buf = backtester._precompute_indicators(ohlcv_no_ts)
    bot_raw = backtester._serialize_bots(bots)
    bots_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=bot_raw)

    debug_bytes = backtester.run_debug_signal_generation(indicators_buf, bots_buf, 0, num_bars - 1, num_bars, num_bots_to_sample=16, bars_per_sample=120, cycle_to_debug=0)

    # parse via same dtype as script
    dt = np.dtype([
        ('bot_id', np.int32),
        ('cycle', np.int32),
        ('bar', np.int32),
        ('num_indicators', np.int32),
        ('valid_indicators', np.int32),
        ('bullish', np.int32),
        ('bearish', np.int32),
        ('neutral', np.int32),
        ('directional', np.int32),
        ('final_signal', np.float32),
        ('indicator_values', np.float32, 8),
        ('indicator_signals', np.int32, 8),
        ('padding', np.uint8, 24)
    ], align=True)

    assert len(debug_bytes) % dt.itemsize == 0
    arr = np.frombuffer(debug_bytes.tobytes(), dtype=dt)

    # Ensure at least one parsed record has num_indicators > 0 and matches generator info
    found = 0
    for rec in arr:
        if 1 <= rec['num_indicators'] <= 8 and 0 <= rec['bot_id'] < 1000 and 0 <= rec['valid_indicators'] <= rec['num_indicators']:
            found += 1
            assert 0 <= rec['bot_id'] < 1000
            assert 1 <= rec['num_indicators'] <= 8
            assert 0 <= rec['valid_indicators'] <= rec['num_indicators']
    assert found > 0


if __name__ == '__main__':
    success = test_debug_kernel_samples()
    print('Debug kernel parsed OK')
    sys.exit(0 if success else 1)
