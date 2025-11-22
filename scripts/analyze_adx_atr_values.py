#!/usr/bin/env python
"""
Analyze actual ADX and ATR values from a sample bot to understand
why ADX=14 and ATR=4x still block 100% of trades.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from backtester.compact_simulator import CompactBacktester
from config.config_manager import ConfigManager

# Load bot 0 (high-trade bot)
config_mgr = ConfigManager()
bots = config_mgr.load_bots('bots', generation=0)
bot = bots[0]  # Use first bot

# Load dataset
df = pd.read_csv('data/BTC_USDT/BTC_USDT_ohlcv_1m.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.tail(144000).reset_index(drop=True)  # One chunk

print(f"Bot 0: {len(bot['indicator_params'])} indicators")
print(f"Dataset: {len(df)} bars")

# Initialize backtester
backtester = CompactBacktester()
cycles = [
    {'bot_index': 0, 'max_open': 1, 'max_gap': 60, 'trade_against_trend': False}
]

# Backtest with filters bypassed to get indicator values
import os
os.environ['DEBUG_DISABLE_FILTERS'] = '1'

results = backtester.backtest_bots([bot], df, cycles)

# Get precomputed indicators
print("\nReading precomputed ADX and ATR values...")
num_bars = len(df)

# Indicator indices: ADX=24, ATR=19, ATR_20=21
adx_idx = 24
atr_idx = 19
atr20_idx = 21

# Read from precomputed buffer (host copy)
precomputed = backtester.precomputed_host_copy
if precomputed is None:
    print("ERROR: No precomputed data available")
    sys.exit(1)

adx_values = precomputed[adx_idx * num_bars:(adx_idx + 1) * num_bars]
atr_values = precomputed[atr_idx * num_bars:(atr_idx + 1) * num_bars]
atr20_values = precomputed[atr20_idx * num_bars:(atr20_idx + 1) * num_bars]

# Remove NaN values
valid_mask = ~(np.isnan(adx_values) | np.isnan(atr_values) | np.isnan(atr20_values))
adx_valid = adx_values[valid_mask]
atr_valid = atr_values[valid_mask]
atr20_valid = atr20_values[valid_mask]
atr_ratios = atr_valid / atr20_valid

print(f"\n{'='*80}")
print("ADX ANALYSIS")
print(f"{'='*80}")
print(f"Valid samples: {len(adx_valid)}")
print(f"ADX percentiles:")
print(f"  Min:    {adx_valid.min():.2f}")
print(f"  5th:    {np.percentile(adx_valid, 5):.2f}")
print(f"  10th:   {np.percentile(adx_valid, 10):.2f}")
print(f"  25th:   {np.percentile(adx_valid, 25):.2f}")
print(f"  50th:   {np.percentile(adx_valid, 50):.2f}")
print(f"  75th:   {np.percentile(adx_valid, 75):.2f}")
print(f"  90th:   {np.percentile(adx_valid, 90):.2f}")
print(f"  95th:   {np.percentile(adx_valid, 95):.2f}")
print(f"  Max:    {adx_valid.max():.2f}")
print(f"\nADX < 14 (current threshold): {(adx_valid < 14).sum()} / {len(adx_valid)} ({100*(adx_valid < 14).mean():.1f}%)")
print(f"ADX < 10: {(adx_valid < 10).sum()} / {len(adx_valid)} ({100*(adx_valid < 10).mean():.1f}%)")
print(f"ADX 14-50 (acceptable range): {((adx_valid >= 14) & (adx_valid <= 50)).sum()} / {len(adx_valid)} ({100*((adx_valid >= 14) & (adx_valid <= 50)).mean():.1f}%)")
print(f"ADX > 50 (overextended): {(adx_valid > 50).sum()} / {len(adx_valid)} ({100*(adx_valid > 50).mean():.1f}%)")

print(f"\n{'='*80}")
print("ATR RATIO ANALYSIS (ATR / ATR_20)")
print(f"{'='*80}")
print(f"Valid samples: {len(atr_ratios)}")
print(f"ATR ratio percentiles:")
print(f"  Min:    {atr_ratios.min():.2f}")
print(f"  5th:    {np.percentile(atr_ratios, 5):.2f}")
print(f"  10th:   {np.percentile(atr_ratios, 10):.2f}")
print(f"  25th:   {np.percentile(atr_ratios, 25):.2f}")
print(f"  50th:   {np.percentile(atr_ratios, 50):.2f}")
print(f"  75th:   {np.percentile(atr_ratios, 75):.2f}")
print(f"  90th:   {np.percentile(atr_ratios, 90):.2f}")
print(f"  95th:   {np.percentile(atr_ratios, 95):.2f}")
print(f"  Max:    {atr_ratios.max():.2f}")
print(f"\nATR ratio > 4.0 (current threshold): {(atr_ratios > 4.0).sum()} / {len(atr_ratios)} ({100*(atr_ratios > 4.0).mean():.1f}%)")
print(f"ATR ratio > 3.0: {(atr_ratios > 3.0).sum()} / {len(atr_ratios)} ({100*(atr_ratios > 3.0).mean():.1f}%)")
print(f"ATR ratio > 2.0: {(atr_ratios > 2.0).sum()} / {len(atr_ratios)} ({100*(atr_ratios > 2.0).mean():.1f}%)")
print(f"ATR ratio ≤ 4.0 (acceptable): {(atr_ratios <= 4.0).sum()} / {len(atr_ratios)} ({100*(atr_ratios <= 4.0).mean():.1f}%)")

print(f"\n{'='*80}")
print("COMBINED FILTER PASS RATE")
print(f"{'='*80}")
combined_pass = (adx_valid >= 14) & (adx_valid <= 50) & (atr_ratios <= 4.0)
print(f"Bars passing ADX [14-50] AND ATR ≤ 4x: {combined_pass.sum()} / {len(combined_pass)} ({100*combined_pass.mean():.1f}%)")

# Clean up
del os.environ['DEBUG_DISABLE_FILTERS']
backtester.cleanup()
