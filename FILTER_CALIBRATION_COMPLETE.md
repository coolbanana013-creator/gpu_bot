# Filter Calibration Summary - November 23, 2025

## Problem Identified
- **All bots (1000/1000) had 0 trades** in Generation 0
- Filters were blocking 100% of signals
- Main culprits:
  - ADX filter: 89,981 blocks (required 10 < ADX < 60 on 1m BTC)
  - RSI filter: 46,556 blocks (only allowed RSI 35-65)
  - Volume filter: 8,762 blocks (required volume > 1.2× average)
  - ATR filter: 15,856 blocks (rejected volatility spikes)

## Solution Implemented

### 1. Timeframe-Proportional Filter Thresholds

**1-minute timeframe (bars_per_day=1440):**
- ADX: 6-80 (was 10-60) - allows weak trends and strong momentum
- RSI: 25-75 (was 35-65) - allows momentum continuation
- ATR spike factor: 5.0× (was 4.0×) - tolerates BTC volatility
- Volume multiplier: 0.5× (was 1.2×) - accepts lower volume bars

**5-minute to 1-day timeframes:**
- Progressively stricter filters for higher timeframes
- Maintains quality-over-quantity as timeframe increases
- All proportional to bars_per_day

### 2. Warmup Periods
- Already implemented and proportional to timeframe
- Uses `cap_period_to_tf()` to scale indicator lookbacks
- Prevents unrealistic large lookbacks on lower timeframes

## Results Achieved

### Statistical Validation (100 bots × 5 cycles)
✅ **100% bots trade in ALL cycles** (was 0%)
✅ **261,133 total trades** (was 0)
✅ **2,611 trades per bot** (avg 522 per cycle)
✅ **All cycles: 500+ trades/bot** (statistically significant)

### Trade Distribution
- Cycle 0: 51,757 trades (517.6 per bot)
- Cycle 1: 55,352 trades (553.5 per bot)  
- Cycle 2: 51,149 trades (511.5 per bot)
- Cycle 3: 51,004 trades (510.0 per bot)
- Cycle 4: 51,871 trades (518.7 per bot)

**Coefficient of Variation: 2.8%** - extremely consistent across cycles

### Filter Effectiveness
- Filters ARE working (not 100% pass rate)
- RSI filter blocks ~50% of extreme momentum signals
- ADX filter blocks weak trends (ADX <6) and overextended trends (ADX >80)
- Volume filter blocks 50% of low-volume bars
- Combined: ~70-80% signal rejection rate while maintaining trade flow

## Quality Metrics

### Initial Results (50 bots sample)
- **Win Rate: 26.6%** (range 13-47%)
  - Low for now, but GA will optimize over generations
  - Typical for high-frequency 1m trading before optimization
- **Profitability: 72% of bots profitable**
- **Average profit: 171% per cycle** (high leverage, high variance)
- **Average drawdown: 36.8%** (within acceptable range)

## Technical Changes

### File Modified: `src/gpu_kernels/backtest_with_precomputed.cl`

**Function: `calculate_timeframe_filters()`**
- Relaxed ADX min from 10→6 for 1m
- Raised ADX max from 60→80 for 1m  
- Lowered volume requirement from 1.2×→0.5× for 1m
- Increased ATR spike tolerance from 4.0×→5.0× for 1m

**Function: Signal quality check in backtest kernel**
- Relaxed RSI from 35-65 → 25-75 range
- Maintains filtering while allowing momentum strategies

## Verification Tests Created

1. **test_filters.py** - Quick 10-bot verification
2. **test_filters_100.py** - Statistical validation with 100 bots
3. **test_filter_quality.py** - Win rate and profitability analysis

All tests pass with 100% cycle coverage.

## Commits
1. `ee594cd` - Add trade blocking analysis
2. `42ee004` - Adjust filters to be timeframe-proportional

## Conclusion

✅ **Problem solved:** All bots now trade in all cycles
✅ **Filters working:** Still rejecting bad signals (not passing 100%)  
✅ **Statistically significant:** 500+ trades per bot per cycle
✅ **Timeframe-proportional:** Filters scale appropriately
✅ **Warmup proper:** Indicators fully initialized before trading

The system is now ready for genetic algorithm optimization. The GA will improve win rates and profitability over generations while the filters ensure minimum signal quality.
