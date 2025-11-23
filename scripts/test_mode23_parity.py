#!/usr/bin/env python3
"""
Test parity between GPU backtest (Mode 1/4) and RealTimeTradingEngine (Mode 2/3) for a single bot.
"""
import csv
import json
import sys
from pathlib import Path
import pyopencl as cl
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bot_generator.compact_generator import CompactBotConfig, CompactBotGenerator
from src.backtester.compact_simulator import CompactBacktester
from src.live_trading.engine import RealTimeTradingEngine
from src.live_trading.kucoin_universal_client import KucoinUniversalClient
from src.data_provider.loader import DataLoader
from src.data_provider.fetcher import DataFetcher

# Setup GPU context
ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)

# Generate a test bot
generator = CompactBotGenerator(gpu_context=ctx, gpu_queue=queue, population_size=1, min_indicators=2, max_indicators=5)
bot = generator.generate_population()[0]
print('Test bot indicator indices:', bot.indicator_indices[:bot.num_indicators])
print('Test bot indicator params:', bot.indicator_params[:bot.num_indicators])

# Backtest with GPU
fetcher = DataFetcher(exchange_type='futures')
file_paths = fetcher.fetch_data_range(pair='BTC/USDT', timeframe='1m', total_days=2)
loader = DataLoader(file_paths=file_paths, timeframe='1m', gpu_context=ctx, gpu_queue=queue)
ohlcv_data = loader.load_all_data()
# For parity test, use a single 1-day cycle (matches downloaded data and lookback)
cycles = loader.generate_cycle_ranges(1, 1)

gpu_backtester = CompactBacktester(gpu_context=ctx, gpu_queue=queue, initial_balance=10.0)
results = gpu_backtester.backtest_bots([bot], ohlcv_data, cycles)
res = results[0]
print('GPU backtest result per-cycle trades:', res.per_cycle_trades)
print('GPU backtest per-cycle pnl:', res.per_cycle_pnl)

# Now run Engine in paper trading mode with the same data
# Use the kucoin client in test_mode; we will not place real orders
client = KucoinUniversalClient(api_key='xxx', api_secret='xxx', api_passphrase='xxx', test_mode=True)
engine = RealTimeTradingEngine(bot, initial_balance=10.0, kucoin_client=client, pair='XBTUSDTM', timeframe='1m', test_mode=True)

ohlcv_array = ohlcv_data[['timestamp','open','high','low','close','volume']].values.astype(np.float32)
for row in ohlcv_array:
    engine.process_candle(row[1], row[2], row[3], row[4], row[5], row[0]/1000.0)

print('\nEngine signals summary:')
print('  Total signals:', engine.total_signals)
print('  Buy signals:', engine.buy_signals)
print('  Sell signals:', engine.sell_signals)
print('\nEngine indicator values (sample):')
for ind_idx, val in engine.indicator_values.items():
    print(f'  Ind {ind_idx}: {val}')
print('Engine bars_count:', engine.indicator_calculator.bars_count)
print('Sample indicator history lengths:')
for k, v in engine.indicator_history.items():
    print(f'  Ind {k}: {len(v)}')

# Recompute final signal using consensus
from src.live_trading.gpu_kernel_port import generate_signal_consensus
signal, breakdown = generate_signal_consensus(
    engine.indicator_values,
    engine.indicator_params,
    engine.indicator_history,
    engine.candles_processed,
    engine.current_price,
    engine.force_signals
)
print('Final computed signal:', signal)
print('Breakdown:', breakdown)
print('\nDirect indicator calculation check:')
for i in range(bot.num_indicators):
    ind_idx = int(bot.indicator_indices[i])
    p0 = float(bot.indicator_params[i][0])
    p1 = float(bot.indicator_params[i][1])
    p2 = float(bot.indicator_params[i][2])
    try:
        v = engine.indicator_calculator.calculate_indicator(ind_idx, p0, p1, p2)
        print(f'  Ind {ind_idx} via calculate_indicator: {v}')
    except Exception as e:
        print(f'  Ind {ind_idx} calc error: {e}')

# Convert engine closed_positions summary to per-cycle pnl/trade counts
# In the engine we track closed_positions as history; but since we didn't chunk by cycle here, we'll aggregate by timestamps

from collections import defaultdict
cycle_pnl_engine = defaultdict(float)
cycle_trades_engine = defaultdict(int)

# Map timestamp to cycle index
cycle_ranges = cycles

def find_cycle(ts_ms):
    for i, (start, end) in enumerate(cycle_ranges):
        if start <= ts_ms < end:
            return i
    return None

for cp in engine.closed_positions:
    ts = int(cp['timestamp']*1000)
    idx = find_cycle(ts)
    if idx is not None:
        cycle_pnl_engine[idx] += float(cp['net_pnl'])
        cycle_trades_engine[idx] += 1

# Compare
print('Engine per-cycle trades:', [cycle_trades_engine.get(i, 0) for i in range(5)])
print('Engine per-cycle pnl:', [cycle_pnl_engine.get(i, 0.0) for i in range(5)])

# Compare per cycle values
for i in range(5):
    gpu_pnl = res.per_cycle_pnl[i]
    engine_pnl = cycle_pnl_engine.get(i, 0.0)
    if abs(gpu_pnl - engine_pnl) > 0.01:
        print(f'Cycle {i} PnL mismatch: GPU={gpu_pnl} vs Engine={engine_pnl}')
    else:
        print(f'Cycle {i} PnL match within tolerance: {gpu_pnl} == {engine_pnl}')

print('Done')
