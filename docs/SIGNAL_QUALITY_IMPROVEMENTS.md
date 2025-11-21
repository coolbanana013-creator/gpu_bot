# Signal Quality & Win Rate Optimization Summary

## Research-Based Improvements Implemented

### 1. ADX Trend Strength Filter
- **Purpose**: Filter out choppy, ranging markets where win rates are lower
- **Implementation**: Require ADX > 15 before taking any trades
- **Benefit**: Ensures trades only happen in markets with directional movement
- **Research Source**: Investopedia - ADX values 15-25 indicate developing trends

### 2. ATR Volatility Filter  
- **Purpose**: Avoid trading during volatility spikes (unpredictable price action)
- **Implementation**: Filter out trades when ATR_14 > 2× ATR_20
- **Benefit**: Reduces losses from whipsaw price movements
- **Research Source**: TradingView - ATR spikes indicate increased risk

### 3. Wider Profit Targets in Strong Trends
- **Purpose**: Let winners run further in trending markets
- **Implementation**: 1.5× wider TP targets when ADX confirms trend
- **Benefit**: Improves profit per trade while maintaining high win rate
- **Formula**: TP_multiplier = base_tp × 1.5 (in trending markets)

### 4. Intelligent Breeding & Mutation
- **Crossover**: Combine indicators from top 10% performers by win rate
- **Mutation**: Adjust parameters (leverage, SL/TP, indicator swaps) on 20% of offspring
- **Min WR Filter**: 60% minimum win rate required for survival (when prefer_winrate enabled)
- **Breeding Rate**: 92% breeding, 5% mutation, 3% random (ultra-aggressive evolution)

### 5. Extreme Evolution Weights
- **Win Rate Weight**: 200× multiplier in survivor selection
- **Trade Volume Requirement**: 200+ trades minimum to ensure statistical significance
- **Consensus Threshold**: 70% indicator agreement required for signals

## Expected Outcomes

### High Win Rate (Target: 90-99%)
- ADX filter ensures only high-probability setups
- MTF (Multi-Timeframe) filtering aligns signals with higher timeframe trends
- Consensus signals require 70% of indicators to agree

### Improved Returns
- 1.5× wider profit targets in trending markets
- Better risk/reward ratios (4.5:1 instead of 3:1)
- Fewer but higher quality trades

### Trade Quality Metrics
```
Before Filters:
- Avg Win Rate: 56-70%
- Avg Profit: Low (tight TP)
- Trade Frequency: High (any market condition)

After Filters:
- Avg Win Rate: 70-90%+ (target)
- Avg Profit: Higher (wider TP in trends)
- Trade Frequency: Lower (filtered to trends only)
```

## Files Modified

1. `src/gpu_kernels/backtest_with_precomputed.cl`
   - Added `check_signal_quality()` function
   - Modified `calculate_dynamic_tp_sl()` with trend multiplier
   - Integrated ADX/ATR filters before consensus

2. `src/ga/evolver_compact.py`
   - Added `mutate_bot_parameters()` function
   - Added `breed_top_performers()` function
   - Modified survival filters (min WR, trade count)
   - Adjusted scoring weights (200× WR multiplier)

## Next Steps to Reach 90%+ Win Rate

1. **Volume Profile Integration**
   - Add volume-based confirmation (high volume = stronger signals)
   - Filter out low-volume breakouts (often fail)

2. **Support/Resistance Filtering**
   - Avoid entries near major S/R levels (prone to rejection)
   - Wait for S/R breakouts with confirmation

3. **Time-Based Filters**
   - Avoid first/last hour of trading (high volatility, low predictability)
   - Focus on high-liquidity periods

4. **Correlation Analysis**
   - Track indicator correlation to find optimal combinations
   - Prefer uncorrelated indicators for better consensus quality

## Performance Monitoring

Track these metrics per generation:
- Max Win Rate achieved
- Number of bots with 80%+ WR
- Average profit per trade
- Total trades (ensure sufficient volume)
- Sharpe Ratio (risk-adjusted returns)

Run `python logs/track_evolution.py` to monitor progress.
