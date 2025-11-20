"""
Deep dive into why SMA isn't generating signals
"""
import pandas as pd
import numpy as np
from pathlib import Path

data_dir = Path("data/BTC_USDT/1m")
df = pd.read_parquet(sorted(data_dir.glob("*.parquet"))[0])

close = df['close'].values

# Calculate SMA(20)
sma_period = 20
sma = np.zeros(len(close))
for i in range(sma_period, len(close)):
    sma[i] = np.mean(close[i-sma_period:i])

print("Analyzing SMA(20) changes...")
print()

changes = []
for i in range(100, 1100):
    if sma[i] == 0 or sma[i-1] == 0:
        continue
    
    pct_change = (sma[i] - sma[i-1]) / sma[i-1]
    changes.append(pct_change)

changes = np.array(changes)

print(f"SMA change statistics (1000 bars):")
print(f"  Min: {np.min(changes)*100:.4f}%")
print(f"  Max: {np.max(changes)*100:.4f}%")
print(f"  Mean: {np.mean(changes)*100:.4f}%")
print(f"  Median: {np.median(changes)*100:.4f}%")
print(f"  Std: {np.std(changes)*100:.4f}%")
print()

# Count how many meet the 0.1% threshold
above_threshold = np.sum(changes > 0.001)
below_threshold = np.sum(changes < -0.001)
neutral = len(changes) - above_threshold - below_threshold

print(f"With 0.1% threshold:")
print(f"  Bullish (>+0.1%): {above_threshold} ({100*above_threshold/len(changes):.1f}%)")
print(f"  Bearish (<-0.1%): {below_threshold} ({100*below_threshold/len(changes):.1f}%)")
print(f"  Neutral: {neutral} ({100*neutral/len(changes):.1f}%)")
print()

# Try different thresholds
thresholds = [0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00001]

print("Testing different thresholds:")
for thresh in thresholds:
    above = np.sum(changes > thresh)
    below = np.sum(changes < -thresh)
    neut = len(changes) - above - below
    
    print(f"  {thresh*100:.3f}% threshold: {above} bull ({100*above/len(changes):.1f}%), {below} bear ({100*below/len(changes):.1f}%), {neut} neutral ({100*neut/len(changes):.1f}%)")

print()
print("CONCLUSION:")
print("  SMA on 1-minute timeframe changes VERY slowly")
print("  0.1% change per minute is unrealistic for a 20-period MA")
print("  Need much lower threshold (0.01% or 0.001%) OR longer lookback")
