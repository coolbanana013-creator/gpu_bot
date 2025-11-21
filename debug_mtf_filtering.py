"""
Debug MTF Filtering Logic
Analyze why MTF didn't reduce trades as expected
"""
import numpy as np
from src.backtester.compact_simulator import CompactBacktester
from src.bot_generator.compact_generator import CompactBotGenerator
import sys
import os

# Setup paths
sys.path.insert(0, os.path.abspath('.'))

def generate_strong_trend_data(num_bars=10080, trend='bullish'):
    """Generate strongly trending data with 50% trend + noise"""
    np.random.seed(42)
    base_price = 50000.0
    
    if trend == 'bullish':
        # Strong uptrend: 5% gain over period
        trend_component = np.linspace(0, 2500, num_bars)  # +5% trend
    else:
        # Strong downtrend: 5% loss over period
        trend_component = np.linspace(0, -2500, num_bars)  # -5% trend
    
    # Add smaller noise
    noise = np.random.randn(num_bars) * 50  # Reduced noise
    
    close = base_price + trend_component + noise
    
    # Generate realistic OHLCV
    high = close + np.abs(np.random.randn(num_bars) * 30)
    low = close - np.abs(np.random.randn(num_bars) * 30)
    open_price = close + np.random.randn(num_bars) * 20
    volume = 1000000 + np.random.randn(num_bars) * 100000
    
    return np.column_stack([open_price, high, low, close, volume])

def analyze_htf_indicators(ohlcv, htf_multiplier=60):
    """Analyze HTF indicator values to see if trends are detected"""
    print("\n" + "="*80)
    print("HTF INDICATOR ANALYSIS")
    print("="*80)
    
    # Initialize backtester to get HTF indicators
    bot_gen = CompactBotGenerator(
        population_size=1,
        min_indicators=1,
        max_indicators=1,
        min_risk_strategies=1,
        max_risk_strategies=1
    )
    bots = bot_gen.generate_bots()
    
    backtester = CompactBacktester(
        bots=bots,
        ohlcv_data=ohlcv,
        initial_balance=10000.0,
        fee_rate=0.0004,
        max_open_trades=5,
        enable_mtf=True,
        htf_multiplier=htf_multiplier,
        verbose=False
    )
    
    # Get HTF indicators (this triggers computation)
    htf_indicators = backtester._compute_htf_indicators()
    
    if htf_indicators is None:
        print("ERROR: HTF indicators not computed")
        return
    
    num_htf_bars = htf_indicators.shape[1]
    print(f"\nHTF multiplier: {htf_multiplier}x")
    print(f"Base bars: {len(ohlcv)}")
    print(f"HTF bars: {num_htf_bars}")
    print(f"Expected HTF bars: {len(ohlcv) // htf_multiplier}")
    
    # Analyze SMA(20) (index 2) which is used for trend detection
    sma20_idx = 2
    sma20_htf = htf_indicators[sma20_idx]
    
    print(f"\nSMA(20) HTF values (first 20 bars):")
    for i in range(min(20, num_htf_bars)):
        if i > 0:
            change_pct = ((sma20_htf[i] - sma20_htf[i-1]) / sma20_htf[i-1]) * 100
            trend = "BULLISH" if change_pct > 0.2 else ("BEARISH" if change_pct < -0.2 else "NEUTRAL")
            print(f"  Bar {i:3d}: {sma20_htf[i]:10.2f} ({change_pct:+6.3f}%) - {trend}")
        else:
            print(f"  Bar {i:3d}: {sma20_htf[i]:10.2f}")
    
    # Count trends
    bullish_bars = 0
    bearish_bars = 0
    neutral_bars = 0
    
    for i in range(1, num_htf_bars):
        if not np.isnan(sma20_htf[i]) and not np.isnan(sma20_htf[i-1]):
            change_pct = ((sma20_htf[i] - sma20_htf[i-1]) / sma20_htf[i-1]) * 100
            if change_pct > 0.2:
                bullish_bars += 1
            elif change_pct < -0.2:
                bearish_bars += 1
            else:
                neutral_bars += 1
    
    total = bullish_bars + bearish_bars + neutral_bars
    print(f"\nTrend distribution:")
    print(f"  Bullish: {bullish_bars}/{total} ({bullish_bars*100/total:.1f}%)")
    print(f"  Bearish: {bearish_bars}/{total} ({bearish_bars*100/total:.1f}%)")
    print(f"  Neutral: {neutral_bars}/{total} ({neutral_bars*100/total:.1f}%)")
    
    if neutral_bars > total * 0.7:
        print("\n⚠️  WARNING: >70% neutral bars - threshold too strict!")
        print("   This explains why MTF filtering had no effect")
        print("   Recommendation: Lower threshold from 0.2% to 0.05% or use different indicator")

print("="*80)
print("MTF FILTERING DEBUG ANALYSIS")
print("="*80)

# Test 1: Bullish trend
print("\n" + "="*80)
print("TEST 1: BULLISH TRENDING DATA")
print("="*80)
ohlcv_bull = generate_strong_trend_data(num_bars=10080, trend='bullish')
print(f"\nGenerated bullish trend data:")
print(f"  Start price: ${ohlcv_bull[0, 3]:.2f}")
print(f"  End price: ${ohlcv_bull[-1, 3]:.2f}")
print(f"  Total gain: {((ohlcv_bull[-1, 3] - ohlcv_bull[0, 3]) / ohlcv_bull[0, 3] * 100):.2f}%")

analyze_htf_indicators(ohlcv_bull, htf_multiplier=60)

# Test 2: Bearish trend
print("\n" + "="*80)
print("TEST 2: BEARISH TRENDING DATA")
print("="*80)
ohlcv_bear = generate_strong_trend_data(num_bars=10080, trend='bearish')
print(f"\nGenerated bearish trend data:")
print(f"  Start price: ${ohlcv_bear[0, 3]:.2f}")
print(f"  End price: ${ohlcv_bear[-1, 3]:.2f}")
print(f"  Total loss: {((ohlcv_bear[-1, 3] - ohlcv_bear[0, 3]) / ohlcv_bear[0, 3] * 100):.2f}%")

analyze_htf_indicators(ohlcv_bear, htf_multiplier=60)

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)
print("If most HTF bars show NEUTRAL trend, the 0.2% threshold is too strict.")
print("Solution: Reduce threshold to 0.05% or use trend strength indicator like ADX")
