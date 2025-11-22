"""
Run the GPU signal debug kernel for a small sample of bots and print diagnostic summaries
"""
import sys
import os
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# Force UTF-8
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pyopencl as cl
from src.backtester.compact_simulator import CompactBacktester
from src.bot_generator.compact_generator import CompactBotGenerator


def parse_debug_bytes(debug_bytes):
    # Define numpy dtype matching SignalDebugRecord in kernel (128 bytes per record)
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

    assert (len(debug_bytes) % dt.itemsize) == 0, "Debug bytes length not multiple of struct size"
    num_records = len(debug_bytes) // dt.itemsize
    arr = np.frombuffer(debug_bytes.tobytes(), dtype=dt, count=num_records)

    records = []
    for r in arr:
        records.append({
            'bot_id': int(r['bot_id']),
            'cycle': int(r['cycle']),
            'bar': int(r['bar']),
            'num_indicators': int(r['num_indicators']),
            'valid_indicators': int(r['valid_indicators']),
            'bullish': int(r['bullish']),
            'bearish': int(r['bearish']),
            'neutral': int(r['neutral']),
            'directional': int(r['directional']),
            'final_signal': float(r['final_signal']),
            'indicator_values': [float(x) for x in r['indicator_values']],
            'indicator_signals': [int(x) for x in r['indicator_signals']]
        })
    return records


if __name__ == '__main__':
    # Basic GPU init
    platforms = cl.get_platforms()
    devices = platforms[0].get_devices(device_type=cl.device_type.GPU)
    ctx = cl.Context(devices=[devices[0]])
    queue = cl.CommandQueue(ctx)

    # Create backtester
    backtester = CompactBacktester(gpu_context=ctx, gpu_queue=queue, initial_balance=100.0)

    # Create generator and generate a small population of bots
    generator = CompactBotGenerator(
        gpu_context=ctx,
        gpu_queue=queue,
        population_size=64,
        min_indicators=1,
        max_indicators=3,
        min_risk_strategies=1,
        max_risk_strategies=1,
        min_leverage=20,
        max_leverage=50
    )

    bots = generator.generate_population()
    # Quick diagnostic: distribution of number of indicators per bot
    from collections import Counter
    indicator_counts = Counter([b.num_indicators for b in bots])
    print(f"Bot indicator count distribution: {indicator_counts}")
    # Show a few sample bots
    print("Sample bot configs:")
    for i, b in enumerate(bots[:5]):
        print(f"  Bot {i}: num_indicators={b.num_indicators}, indices={b.indicator_indices[:b.num_indicators].tolist()}, leverage={b.leverage}")

    # Load small data sample (1 day) or create synthetic
    data_dir = Path('data/BTC_USDT/1m')
    if list(data_dir.glob('*.parquet')):
        import pandas as pd
        df = pd.read_parquet(sorted(list(data_dir.glob('*.parquet')))[0])
        ohlcv = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].values.astype(np.float32)
    else:
        # Synthetic: 1 day = 1440 bars
        num_bars = 1440
        ts = np.arange(num_bars, dtype=np.float32)
        price = 50000 + np.cumsum(np.random.randn(num_bars).astype(np.float32))
        ohlcv = np.zeros((num_bars, 6), dtype=np.float32)
        ohlcv[:, 0] = ts
        ohlcv[:, 1] = price
        ohlcv[:, 2] = price * 1.001
        ohlcv[:, 3] = price * 0.999
        ohlcv[:, 4] = price
        ohlcv[:, 5] = 1000000.0

    # Remove timestamp column if present: ensure OHLCV format: open, high, low, close, volume
    if ohlcv.shape[1] == 6:
        ohlcv_no_ts = ohlcv[:, 1:]
    else:
        ohlcv_no_ts = ohlcv
    num_bars = len(ohlcv_no_ts)
    print(f"Sample ohlcv row at {min(540, num_bars-1)}: {ohlcv_no_ts[min(540, num_bars-1)].tolist()}")

    # Precompute indicators
    indicators_buf = backtester._precompute_indicators(ohlcv_no_ts)

    # Diagnostic: read back some indicator values and check for NaNs
    num_indicators = backtester.NUM_INDICATORS
    indicators_flat = np.empty(num_indicators * num_bars, dtype=np.float32)
    cl.enqueue_copy(queue, indicators_flat, indicators_buf)
    indicators_flat = indicators_flat.reshape((num_indicators, num_bars))
    # Count NaNs per indicator
    nan_counts = np.sum(np.isnan(indicators_flat), axis=1)
    print("Indicator NaN counts (first 10):", nan_counts[:10].tolist())
    print("Total NaNs across all indicators:", int(np.sum(nan_counts)))
    # Print indicator values at sample bar 540 to inspect suspicious large values
    sample_bar = min(540, num_bars - 1)
    print(f"Indicator values at bar {sample_bar}: ")
    for i in range(50):
        print(f"  Ind {i}: {indicators_flat[i, sample_bar]}")
    # Compute max and min per indicator
    for i in range(50):
        maxi = float(np.nanmax(np.abs(indicators_flat[i, :])))
        if maxi > 1e8 or np.isnan(maxi):
            print(f"  [ANOMALY] Indicator {i} max abs value: {maxi}")

    # Serialize bots and create buffer
    bot_raw = backtester._serialize_bots(bots)
    bots_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=bot_raw)

    # Determine a cycle range: entire dataset
    cycle_start = 0
    cycle_end = num_bars - 1

    print("Running debug signal kernel for 64 bots ...")
    debug_bytes = backtester.run_debug_signal_generation(indicators_buf, bots_buf, cycle_start, cycle_end, num_bars, num_bots_to_sample=64, bars_per_sample=60, cycle_to_debug=0)

    # Parse records and show summary
    recs = parse_debug_bytes(debug_bytes)

    # Basic summarization: filter records
    neutral_counts = []
    all_neutral_bots = set()
    bullish_counts = []
    bearish_counts = []

    # Filter out invalid/uninitialized records: num_indicators > 0 and valid_indicators > 0
    good_recs = [r for r in recs if r['num_indicators'] > 0 and r['num_indicators'] <= 8 and r['valid_indicators'] > 0 and r['valid_indicators'] <= r['num_indicators'] and 0 <= r['bot_id'] < 100000 and (r['bullish'] + r['bearish'] + r['neutral'] == r['valid_indicators'])]
    for r in good_recs:
        neutral_counts.append(r['neutral'])
        bullish_counts.append(r['bullish'])
        bearish_counts.append(r['bearish'])
        if r['directional'] == 0 and r['valid_indicators'] > 0:
            all_neutral_bots.add(r['bot_id'])

    print(f"Total records (raw): {len(recs)}")
    print(f"Total good records (num_indicators>0 & valid>0): {len(good_recs)}")
    # Show first 3 raw record bytes for diagnostics
    print("First record raw bytes (hex):", debug_bytes[:128].tobytes().hex())
    print("First 3 parsed records:")
    for i, r in enumerate(recs[:3]):
        print(f"  Record {i}: {r}")
    print(f"Bots with at least one sample with all neutral/directional==0: {len(all_neutral_bots)} (IDs sample) {sorted(list(all_neutral_bots))[:10]}")
    print(f"Avg neutral signals per record: {np.mean(neutral_counts):.2f}")
    print(f"Avg bullish signals per record: {np.mean(bullish_counts):.2f}")
    print(f"Avg bearish signals per record: {np.mean(bearish_counts):.2f}")

    # Print top offenders (bots with many neutral records)
    neutral_per_bot = {}
    for r in recs:
        neutral_per_bot.setdefault(r['bot_id'], 0)
        neutral_per_bot[r['bot_id']] += r['neutral']

    ranked = sorted(neutral_per_bot.items(), key=lambda x: x[1], reverse=True)
    print("Top 10 bots by total neutral counts (sample):")
    for bot_id, count in ranked[:10]:
        print(f"  Bot {bot_id}: total neutral count (samples × neutral_count) = {count}")

    print("Done.")

    # Some diagnostics: find records with anomalous field values
    anomalies = []
    for i, r in enumerate(recs):
        if r['neutral'] < 0 or r['neutral'] > 1000 or r['bullish'] < 0 or r['bullish'] > 1000 or r['bearish'] < 0 or r['bearish'] > 1000:
            anomalies.append((i, r))
    if anomalies:
        print(f"Found {len(anomalies)} anomaly records (neutral/bullish/bearish outside 0-1000)")
        # Print first 5 anomalies with their raw bytes
        for idx, r in anomalies[:5]:
            start = idx * 128
            print(f"Anomaly record idx {idx}, bot {r['bot_id']} raw bytes: {debug_bytes[start:start+128].tobytes().hex()}")

    # Per-bot summary for good records
    from collections import defaultdict
    per_bot = defaultdict(lambda: {'samples': 0, 'avg_neutral': 0, 'avg_directional': 0, 'total_directional': 0})
    for r in good_recs:
        b = per_bot[r['bot_id']]
        b['samples'] += 1
        b['avg_neutral'] += r['neutral']
        b['total_directional'] += r['directional']
    # finalize averages
    bot_summary = []
    for k, v in per_bot.items():
        v['avg_neutral'] = v['avg_neutral'] / v['samples'] if v['samples'] else 0
        bot_summary.append((k, v['samples'], v['avg_neutral'], v['total_directional']))
    bot_summary.sort(key=lambda x: (x[2], -x[3]), reverse=True)
    print("Top 10 bots by avg neutral (good records):")
    for k, s, an, td in bot_summary[:10]:
        print(f"  Bot {k}: samples={s}, avg_neutral={an:.2f}, total_directional={td}")

    print("\nDetailed neutral-only samples:")
    for r in good_recs:
        if r['directional'] == 0 and r['valid_indicators'] > 0:
            bcfg = bots[r['bot_id']]
            inds = bcfg.indicator_indices[:bcfg.num_indicators].tolist()
            print(f"Bot {r['bot_id']} @ bar {r['bar']}: indicators={inds}, values={r['indicator_values'][:bcfg.num_indicators]}, signals={r['indicator_signals'][:bcfg.num_indicators]}")
            # Print actual precomputed indicator values for comparison
            for idx in inds:
                if 0 <= idx < indicators_flat.shape[0] and 0 <= r['bar'] < indicators_flat.shape[1]:
                    print(f"   Precomputed indicator {idx} at bar {r['bar']}: {indicators_flat[idx, r['bar']]} (from precomputed buffer)")
            # Also compute VWAP and OBV in Python to compare with precomputed values
            if 37 in inds:
                # Compute VWAP up to this bar
                cum_tp_vol = 0.0
                cum_vol = 0.0
                for b in range(0, r['bar'] + 1):
                    tp_b = (ohlcv_no_ts[b, 1] + ohlcv_no_ts[b, 2] + ohlcv_no_ts[b, 3]) / 3.0
                    v_b = float(ohlcv_no_ts[b, 4])
                    cum_tp_vol += tp_b * v_b
                    cum_vol += v_b
                py_vwap = cum_tp_vol / cum_vol if cum_vol > 0 else float(ohlcv_no_ts[r['bar'], 3])
                print(f"   Python VWAP at bar {r['bar']}: {py_vwap}")
            if 36 in inds:
                # Compute OBV up to bar
                obv = 0.0
                for b in range(1, r['bar'] + 1):
                    if ohlcv_no_ts[b, 3] > ohlcv_no_ts[b-1, 3]:
                        obv += float(ohlcv_no_ts[b, 4])
                    elif ohlcv_no_ts[b, 3] < ohlcv_no_ts[b-1, 3]:
                        obv -= float(ohlcv_no_ts[b, 4])
                print(f"   Python OBV at bar {r['bar']}: {obv}")
