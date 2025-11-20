"""
Debug signal generation in detail
Tracks exactly why signals are/aren't being generated
"""
import numpy as np
import pyopencl as cl
from pathlib import Path
import json
from collections import Counter

# Initialize OpenCL
ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)

print("Loading data and components...")

# Load the most recent config
with open("config/last_run_config.json") as f:
    config = json.load(f)

# Load saved bot data from most recent run
runs_dir = Path("bots/BTC_USDT/1m")
latest_run = sorted(runs_dir.glob("run_*"))[-1]
print(f"Analyzing run: {latest_run.name}\n")

# For this debug, we'll simulate loading a small dataset
# and tracking signal generation for first 10 bots

print("=" * 80)
print("SIGNAL GENERATION DEBUG ANALYSIS")
print("=" * 80)
print()

# Since we don't have the full GPU pipeline here, let's analyze the backtest results
# to understand signal patterns

print("Loading OHLCV data for manual analysis...")

# Load one day of data to analyze
from datetime import datetime, timedelta
import pandas as pd

data_dir = Path("data/BTC_USDT/1m")
data_files = sorted(data_dir.glob("*.parquet"))

if not data_files:
    print("No data files found")
    exit(1)

# Load first file
df = pd.read_parquet(data_files[0])
print(f"Loaded {len(df)} candles from {data_files[0].name}")
print(f"Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
print()

# Now let's manually calculate some indicators and see their signal patterns
print("=" * 80)
print("MANUAL INDICATOR ANALYSIS")
print("=" * 80)
print()

close = df['close'].values
high = df['high'].values
low = df['low'].values
volume = df['volume'].values

# Calculate a simple SMA(20)
sma_period = 20
sma = np.zeros(len(close))
for i in range(sma_period, len(close)):
    sma[i] = np.mean(close[i-sma_period:i])

# Calculate RSI(14)
rsi_period = 14
rsi = np.zeros(len(close))
for i in range(rsi_period + 1, len(close)):
    gains = []
    losses = []
    for j in range(i - rsi_period, i):
        change = close[j] - close[j-1]
        if change > 0:
            gains.append(change)
        else:
            losses.append(abs(change))
    
    if not losses:
        rsi[i] = 100
    elif not gains:
        rsi[i] = 0
    else:
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            rsi[i] = 100
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))

# Calculate Momentum
momentum_period = 10
momentum = np.zeros(len(close))
for i in range(momentum_period, len(close)):
    momentum[i] = close[i] - close[i - momentum_period]

print("Indicator signal statistics (sample of 1000 bars):")
print()

# Analyze SMA signals with FIXED threshold (0.001% instead of 0.1%)
sma_signals = []
for i in range(100, 1100):
    if i >= len(sma) or sma[i] == 0 or sma[i-1] == 0:
        continue
    if sma[i] > sma[i-1] * 1.00001:  # 0.001% threshold
        sma_signals.append(1)
    elif sma[i] < sma[i-1] * 0.99999:  # 0.001% threshold
        sma_signals.append(-1)
    else:
        sma_signals.append(0)

sma_counter = Counter(sma_signals)
print(f"SMA(20) signals (1000 bars):")
print(f"  Bullish: {sma_counter[1]} ({100*sma_counter[1]/len(sma_signals):.1f}%)")
print(f"  Bearish: {sma_counter[-1]} ({100*sma_counter[-1]/len(sma_signals):.1f}%)")
print(f"  Neutral: {sma_counter[0]} ({100*sma_counter[0]/len(sma_signals):.1f}%)")
print()

# Analyze RSI signals
rsi_signals = []
for i in range(100, 1100):
    if i >= len(rsi) or rsi[i] == 0:
        continue
    if rsi[i] < 30:
        rsi_signals.append(1)
    elif rsi[i] > 70:
        rsi_signals.append(-1)
    else:
        rsi_signals.append(0)

rsi_counter = Counter(rsi_signals)
print(f"RSI(14) signals (1000 bars):")
print(f"  Bullish: {rsi_counter[1]} ({100*rsi_counter[1]/len(rsi_signals):.1f}%)")
print(f"  Bearish: {rsi_counter[-1]} ({100*rsi_counter[-1]/len(rsi_signals):.1f}%)")
print(f"  Neutral: {rsi_counter[0]} ({100*rsi_counter[0]/len(rsi_signals):.1f}%)")
print()

# Analyze Momentum signals
momentum_signals = []
for i in range(100, 1100):
    if i >= len(momentum):
        continue
    if momentum[i] > 0:
        momentum_signals.append(1)
    elif momentum[i] < 0:
        momentum_signals.append(-1)
    else:
        momentum_signals.append(0)

momentum_counter = Counter(momentum_signals)
print(f"Momentum(10) signals (1000 bars):")
print(f"  Bullish: {momentum_counter[1]} ({100*momentum_counter[1]/len(momentum_signals):.1f}%)")
print(f"  Bearish: {momentum_counter[-1]} ({100*momentum_counter[-1]/len(momentum_signals):.1f}%)")
print(f"  Neutral: {momentum_counter[0]} ({100*momentum_counter[0]/len(momentum_signals):.1f}%)")
print()

# Now simulate consensus for different combinations
print("=" * 80)
print("CONSENSUS SIMULATION")
print("=" * 80)
print()

# Test 100% consensus with all 3 indicators
consensus_signals = []
for i in range(len(sma_signals)):
    if i >= len(rsi_signals) or i >= len(momentum_signals):
        break
    
    bull = 0
    bear = 0
    
    if sma_signals[i] == 1: bull += 1
    elif sma_signals[i] == -1: bear += 1
    
    if rsi_signals[i] == 1: bull += 1
    elif rsi_signals[i] == -1: bear += 1
    
    if momentum_signals[i] == 1: bull += 1
    elif momentum_signals[i] == -1: bear += 1
    
    directional = bull + bear
    
    if directional == 0:
        consensus_signals.append(0)  # All neutral
    elif bear == 0:
        consensus_signals.append(1)  # 100% bull
    elif bull == 0:
        consensus_signals.append(-1)  # 100% bear
    else:
        consensus_signals.append(0)  # Mixed

consensus_counter = Counter(consensus_signals)
print(f"100% Consensus (SMA+RSI+Momentum, neutrals ignored):")
print(f"  Bullish: {consensus_counter[1]} ({100*consensus_counter[1]/len(consensus_signals):.1f}%)")
print(f"  Bearish: {consensus_counter[-1]} ({100*consensus_counter[-1]/len(consensus_signals):.1f}%)")
print(f"  No signal: {consensus_counter[0]} ({100*consensus_counter[0]/len(consensus_signals):.1f}%)")
print()

# Calculate how many bars would generate trades
trades_possible = consensus_counter[1] + consensus_counter[-1]
print(f"Bars with potential trades: {trades_possible}/{len(consensus_signals)} ({100*trades_possible/len(consensus_signals):.1f}%)")
print()

# For a 7-day cycle (10,080 minutes)
cycle_bars = 10080
expected_signals = int(cycle_bars * trades_possible / len(consensus_signals))
print(f"Expected signals per 7-day cycle: ~{expected_signals}")
print()

print("=" * 80)
print("ANALYSIS SUMMARY")
print("=" * 80)
print()

if trades_possible / len(consensus_signals) < 0.05:
    print("⚠️  PROBLEM IDENTIFIED:")
    print("  - Indicators generate signals <5% of the time")
    print("  - 100% consensus makes it even more rare")
    print("  - This explains why 40% of cycles have 0 trades")
    print()
    print("SOLUTIONS:")
    print("  1. Lower consensus threshold (e.g., 75% or 67%)")
    print("  2. Relax indicator thresholds (make them more sensitive)")
    print("  3. Use majority voting instead of unanimous agreement")
else:
    print("✓ Signal generation seems reasonable")
    print("  - Issue may be elsewhere (position sizing, balance checks, etc.)")
