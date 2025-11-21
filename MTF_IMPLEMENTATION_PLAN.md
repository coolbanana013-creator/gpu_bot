# SIMPLIFIED MULTI-TIMEFRAME IMPLEMENTATION

## Phase 1: Core MTF Logic (GPU Kernel) - DONE ✅
- Added `calculate_htf_trend()` function to determine HTF trend
- Added `apply_mtf_filter()` function to filter signals
- Functions ready to use once HTF data is provided

## Phase 2: Modify Signal Generation Call (GPU Kernel) - NEXT STEP
Location: Line 2236 in backtest_with_precomputed.cl

Change from:
```c
float signal = generate_signal_consensus(
    precomputed_indicators,
    &bot,
    bar,
    num_bars,
    bot.bot_id
);
```

To:
```c
// Generate base timeframe signal
float base_signal = generate_signal_consensus(
    precomputed_indicators,
    &bot,
    bar,
    num_bars,
    bot.bot_id
);

// Apply multi-timeframe filter (if HTF data available)
float signal = base_signal;  // Default: no filter
#ifdef MTF_ENABLED
signal = apply_mtf_filter(
    base_signal,
    htf1_indicators,
    htf2_indicators,
    bar,
    num_bars,
    htf1_multiplier,
    htf2_multiplier,
    num_bars_htf1,
    num_bars_htf2
);
#endif
```

## Phase 3: Update Kernel Signatures - OPTIONAL
Add HTF parameters to kernel (if MTF enabled):
```c
__global float *htf1_indicators,  // HTF1 precomputed indicators
__global float *htf2_indicators,  // HTF2 precomputed indicators
const int htf1_multiplier,        // Multiplier to HTF1
const int htf2_multiplier,        // Multiplier to HTF2
const int num_bars_htf1,          // Total bars on HTF1
const int num_bars_htf2           // Total bars on HTF2
```

## Phase 4: Python Host Code Updates - REQUIRED
1. Detect timeframe and calculate HTF requirements
2. Fetch and resample HTF data
3. Precompute indicators on HTF data
4. Pass HTF buffers to GPU kernel

## Simplified Approach: WITHOUT Kernel Changes

### Option A: Post-Processing Filter (Python-side)
Instead of GPU kernel changes, apply MTF filter in Python:
1. Run backtest as normal (generates all signals)
2. After backtest, filter out trades that opposed HTF trend
3. Recalculate results with filtered trades only

**Pros**: No kernel changes, easy to implement
**Cons**: Still wastes GPU time on filtered trades

### Option B: Pre-Filter Signals (Indicator Precomputation)
Add MTF trend as additional "virtual indicators":
- Indicator 50: HTF1 Trend (-1, 0, +1)
- Indicator 51: HTF2 Trend (-1, 0, +1)

Then in signal generation:
- Check if indicators 50/51 are in bot config
- If yes, apply MTF filter logic

**Pros**: Minimal kernel changes, clean architecture
**Cons**: Requires indicator system extension

## Recommended Quick Win: Survival Criteria + RR Improvements

Since MTF requires significant refactoring, start with:

### 1. Fix Survival Criteria (5 minutes)
```python
# src/ga/evolver_compact.py line 420-424
profitable_cycles = sum(1 for pnl in result.per_cycle_pnl if pnl > 0)
if profitable_cycles < (num_cycles * 0.7):  # 70% profitable
    continue
if result.max_drawdown >= 0.30:  # 30% max DD
    continue
```

### 2. Improve TP/SL Ratios (10 minutes)
```c
// backtest_with_precomputed.cl
// In calculate_dynamic_tp_sl():
tp_multiplier = 2.5f;  // TP at 2.5x ATR
sl_multiplier = 1.0f;  // SL at 1.0x ATR
// Achieves 2.5:1 RR ratio
```

### 3. Add Partial Profit Taking (15 minutes)
```c
// When position hits TP1 (1.5x ATR):
// Close 50% of position
// Move SL to breakeven on remaining 50%
// Let remaining run to TP2 (2.5x ATR)
```

**Expected Results with These 3 Changes:**
- Survival rate: 5-15% (up from 0%)
- Win rate: 35-45% (up from 0%)
- RR ratio: 2.0:1 average
- Evolution: WORKING (bots can evolve)

## Full MTF Implementation Timeline
- Survival criteria fix: 5 min ✅ Can do now
- RR improvements: 25 min ✅ Can do now
- MTF Python infrastructure: 2 hours
- MTF GPU kernel integration: 2 hours
- Testing & validation: 1 hour
- **Total: ~5 hours**

## Recommendation
1. **Immediate**: Fix survival criteria + improve RR (30 min)
2. **Next**: Run evolution, verify survival rate >5%
3. **Then**: If survival working, implement full MTF (5 hrs)

This gets wins fast, then adds MTF for even better results.
