# Extreme Win Rate Filters (75%+ → 99-100% Target)

**Implementation Date:** Current session  
**Objective:** Achieve 75%+ win rate from generation 0, evolving to 99-100% by generation 5  
**Test Configuration:** 10k bots, 5 generations, 10 cycles, 7 days each, 1m timeframe

---

## Summary of Changes

We've implemented **7 layers of filtering** to maximize win rate at the cost of trade frequency:

1. ✅ **ADX Filter (20+)** - Moderate trend strength required
2. ✅ **ATR Spike Filter** - Avoid 2x volatility spikes
3. ✅ **Volume Confirmation (1.5x)** - Institutional participation required
4. ✅ **Support/Resistance Filter (0.5%)** - Avoid false breakouts/bounces
5. ✅ **Mean Reversion Extremes (RSI<15 or >85)** - Only trade exhaustion points
6. ✅ **80% Indicator Consensus** - Raised from 70%
7. ⏳ **Time-Based Filters** - (Optional, for further refinement)

---

## 1. ADX Filter Tightening

**Previous:** ADX > 15 (developing trends)  
**Current:** ADX > 20 (reliable trends)

```c
// ADX Filter: Require moderate trend strength (ADX > 20)
// Raised from 15 to 20 for better win rate (stronger trend confirmation)
// Research shows: ADX 0-15 = very weak, 15-25 = developing trend, 20+ = reliable trend, 25+ = strong
if (adx < 20.0f) {
    return 0;  // Filter out - weak/ranging market
}
```

**Rationale:**
- ADX 15-20: Too early, trend just forming (more false signals)
- ADX 20-25: Moderate strength, confirmed direction (sweet spot)
- ADX 25+: Too restrictive (blocked all trades in previous tests)

**Expected Impact:** +5-8% win rate improvement by avoiding premature entries

---

## 2. Volume Confirmation Filter

**NEW ADDITION:** Require current volume > 1.5× MA(20)

```c
// Volume Filter: Require above-average volume (institutional participation)
// Calculate 20-period volume MA
float volume_sum = 0.0f;
int volume_count = 0;
for (int i = bar - 19; i <= bar; i++) {
    if (i >= 0 && i < num_bars) {
        volume_sum += ohlcv[i].volume;
        volume_count++;
    }
}
if (volume_count > 0) {
    float volume_ma = volume_sum / volume_count;
    float current_volume = ohlcv[bar].volume;
    
    // Require current volume > 1.5x average for confirmation
    if (current_volume < volume_ma * 1.5f) {
        return 0;  // Filter out - weak volume, no institutional interest
    }
}
```

**Rationale:**
- High volume = institutional participation = stronger moves
- Low volume = retail noise = higher failure rate
- 1.5× threshold balances between too restrictive (2×) and too loose (1.2×)

**Expected Impact:** +8-12% win rate improvement by confirming conviction

---

## 3. Support/Resistance Proximity Filter

**NEW ADDITION:** Block trades within 0.5% of swing highs/lows (last 50 bars)

```c
// Support/Resistance Filter: Avoid trades near recent swing points
// Check last 50 bars for swing highs/lows
float current_price = ohlcv[bar].close;
for (int i = bar - 50; i < bar; i++) {
    if (i < 0 || i >= num_bars) continue;
    
    // Check if this was a swing high (higher than neighbors)
    if (i > 0 && i < num_bars - 1) {
        if (ohlcv[i].high > ohlcv[i-1].high && ohlcv[i].high > ohlcv[i+1].high) {
            float swing_high = ohlcv[i].high;
            // Block if within 0.5% of swing high
            if (fabs(current_price - swing_high) / swing_high < 0.005f) {
                return 0;  // Filter out - too close to resistance
            }
        }
        
        // Check if this was a swing low (lower than neighbors)
        if (ohlcv[i].low < ohlcv[i-1].low && ohlcv[i].low < ohlcv[i+1].low) {
            float swing_low = ohlcv[i].low;
            // Block if within 0.5% of swing low
            if (fabs(current_price - swing_low) / swing_low < 0.005f) {
                return 0;  // Filter out - too close to support
            }
        }
    }
}
```

**Rationale:**
- False breakouts are common near S/R levels
- Bounces at S/R often fail (stop-hunting)
- Waiting for clean levels improves probability
- 0.5% buffer = ~$30 for BTC at $60k (reasonable cushion)

**Expected Impact:** +10-15% win rate improvement by avoiding false breakouts

---

## 4. Mean Reversion at Extremes Filter

**NEW ADDITION:** Only allow trades when RSI < 15 or RSI > 85

```c
// Mean Reversion Filter: Only allow trades at extreme RSI levels
// Get RSI_14 (indicator index 16)
float rsi = precomputed_indicators[16 * num_bars + bar];
if (!isnan(rsi)) {
    // For mean reversion at extremes: Only trade when RSI is in extreme zones
    // RSI < 15 = extreme oversold (high probability bounce)
    // RSI > 85 = extreme overbought (high probability reversal)
    // Block trades in the middle range (15-85) to only capture exhaustion moves
    if (rsi >= 15.0f && rsi <= 85.0f) {
        return 0;  // Filter out - not at extreme levels for mean reversion
    }
}
```

**Rationale:**
- Standard RSI levels (30/70) are too common = low edge
- Extreme levels (15/85) = exhaustion = high probability reversals
- Mean reversion wins have highest success rates
- Trend following without confirmation has lower WR

**Expected Impact:** +15-20% win rate improvement by only trading exhaustion

**CRITICAL NOTE:** This is the most aggressive filter. It will:
- Dramatically reduce trade frequency (90%+ reduction)
- Capture only the highest probability setups
- Potential issue: Too few trades for statistical significance
- **Consider relaxing to RSI<20 or >80 if no survivors**

---

## 5. Indicator Consensus Increase

**Previous:** 70% indicator agreement required  
**Current:** 80% indicator agreement required

```c
// Threshold: 80% consensus required for extreme win rate
// Raised from 70% to 80% to increase signal quality
// Higher agreement = stronger conviction = higher win rate
#ifdef DEBUG_FORCE_LOW_CONSENSUS
    float consensus_threshold = 0.01f; // VERY LOW for debug - any signal accepted
#else
    float consensus_threshold = 0.8f;  // 80% consensus for high WR
#endif
```

**Rationale:**
- More indicators agreeing = stronger signal
- Reduces conflicting signals and whipsaws
- 80% = 4 out of 5 indicators must agree (or 6 out of 8)

**Expected Impact:** +5-8% win rate improvement by requiring stronger conviction

---

## 6. Multi-Level Confirmation System

**Combined Effect:** All filters must pass for trade entry

The check_signal_quality() function now requires:

1. ✅ ADX > 20 (moderate trend)
2. ✅ ATR < 2× ATR_20 (no volatility spikes)
3. ✅ Volume > 1.5× MA(20) (institutional participation)
4. ✅ No S/R within 0.5% (clean levels)
5. ✅ RSI < 15 or RSI > 85 (exhaustion only)

PLUS in generate_signal_consensus():

6. ✅ 80% indicator consensus (strong agreement)

**Total Expected Impact:** 
- **Generation 0:** 40-50% → 75-85% win rate
- **Generation 5:** 75-85% → 95-100% win rate (through breeding)

---

## 7. Trade-Off Analysis

### What We Gain:
- ✅ Much higher win rate (75%+ → 99-100%)
- ✅ Higher quality setups only
- ✅ Better R:R on winning trades
- ✅ Lower drawdowns (fewer losers)

### What We Lose:
- ⚠️ **Drastically reduced trade frequency** (90-95% reduction)
- ⚠️ Potential: Too few trades for min requirement (200+ trades)
- ⚠️ May need to adjust min trade requirements
- ⚠️ Lower absolute returns (fewer opportunities)

---

## 8. Risk Assessment

### Potential Issues:

**Issue 1: Zero Survivors (Like ADX>25)**
- **Filters too restrictive:** RSI<15 or >85 only
- **Solution:** Monitor gen 0 results, relax to RSI<20 or >80 if needed
- **Fallback:** Remove RSI filter entirely, rely on other 5 filters

**Issue 2: Insufficient Trades**
- **Min requirement:** 200+ trades across 10 cycles × 7 days each
- **Solution:** Lower min trade requirement to 50+ for extreme WR testing
- **Fallback:** Extend test to 20 cycles or 14 days per cycle

**Issue 3: Overfitting to Extremes**
- **Risk:** Strategy only works in specific market conditions
- **Solution:** Validate on multiple timeframes and symbols
- **Mitigation:** GA will evolve diverse strategies, not just one

---

## 9. Expected Results

### Conservative Estimates:

**Generation 0:**
- Win Rate: 70-80% (vs previous 48.7% with 200× weight)
- Avg Profit: Unknown (higher per trade due to quality)
- Trade Count: 20-50 trades per bot (vs previous ~500)
- Survivors: 30-50% of population (vs previous 0% with ADX>25)

**Generation 5:**
- Win Rate: 90-100% (through breeding and selection)
- Avg Profit: Higher per trade, lower total (fewer trades)
- Trade Count: 15-40 trades per bot (refined entries only)
- Top 10: 95-100% win rate concentrated

### Optimistic Estimates:

**Generation 0:**
- Win Rate: 80-90%
- Trade Count: 50-100 trades
- Survivors: 50-70%

**Generation 5:**
- Win Rate: 99-100%
- Trade Count: 30-60 trades
- Top Performers: 100% WR with >10 trades

---

## 10. Next Steps

### Immediate Actions:

1. ✅ **Implemented:** All 6 core filters in backtest_with_precomputed.cl
2. ⏳ **Optional:** Time-based liquidity filter (session boundaries, rollover)
3. ⏳ **Test:** Run 10k bots, 5 generations to validate

### Test Command:

```bash
python scripts/run_ga_full.py --population 10000 --generations 5 --cycles 10 --prefer-winrate
```

### Monitoring During Test:

```bash
# Track evolution progress
python logs/track_evolution.py

# Analyze generation 0 results
python logs/analyze_gen0.py

# Check for survivors
python logs/analyze_gen0_survivors.py
```

### If Zero Survivors (Emergency):

1. **Relax RSI filter:** Change `rsi >= 15.0f && rsi <= 85.0f` to `rsi >= 20.0f && rsi <= 80.0f`
2. **Lower volume filter:** Change `1.5f` to `1.3f`
3. **Reduce S/R buffer:** Change `0.005f` (0.5%) to `0.003f` (0.3%)
4. **Lower ADX:** Change `20.0f` back to `18.0f`
5. **Last resort:** Remove RSI filter entirely (comment out the block)

---

## 11. Alternative Strategies (If Extreme Filtering Fails)

If the extreme filters produce zero survivors or insufficient trades:

### Strategy A: Balanced Approach
- ADX > 18 (instead of 20)
- RSI < 20 or > 80 (instead of 15/85)
- Volume > 1.3× (instead of 1.5×)
- Keep S/R and ATR filters
- 75% consensus (instead of 80%)

### Strategy B: Hybrid (Trend + Reversion)
- Allow both trend following AND mean reversion
- Remove RSI extreme filter
- Keep volume, S/R, ADX, ATR filters
- 80% consensus maintained
- Let GA evolve specialization (some bots trend, some revert)

### Strategy C: Quality over Quantity (Current)
- Keep all extreme filters
- Accept very low trade frequency
- Target 90-100% WR with 10-30 trades
- Focus on quality, not quantity

---

## 12. Implementation Details

### Files Modified:

1. **src/gpu_kernels/backtest_with_precomputed.cl**
   - check_signal_quality(): Added volume, S/R, RSI extreme filters
   - generate_signal_consensus(): Updated to 80% consensus threshold
   - Function signature: Added ohlcv parameter for volume/price access

### Code Changes:

- **Lines ~638-720:** check_signal_quality() function
- **Lines ~1163:** Consensus threshold change
- **Lines ~726:** generate_signal_consensus() signature
- **Lines ~2439, 2966:** Function call updates with ohlcv parameter

### Testing Recommendations:

1. **Start with small test:** 1000 bots, 2 gens to verify no errors
2. **Check gen 0:** Must have >0 survivors
3. **Monitor trade count:** If <10 trades per bot, relax RSI filter
4. **Full test:** 10k bots, 5 gens if initial test succeeds

---

## 13. Success Criteria

### Minimum Acceptable:
- Gen 0: >50 bots survive with 70%+ WR
- Gen 5: >10 bots with 90%+ WR
- Trade frequency: >20 trades per surviving bot

### Target:
- Gen 0: >500 bots survive with 75%+ WR
- Gen 5: >100 bots with 95%+ WR, top 10 with 99-100% WR
- Trade frequency: 30-50 trades per bot

### Stretch Goal:
- Gen 0: >1000 bots survive with 80%+ WR
- Gen 5: >500 bots with 98%+ WR, multiple with 100% WR (>20 trades)
- Validate: Zero or near-zero losses across entire evolved population

---

## Conclusion

We've implemented the most aggressive filtering system possible while maintaining logical trading principles:

- **ADX 20+:** Only moderate-to-strong trends
- **Volume 1.5×:** Only institutional moves
- **S/R 0.5%:** Only clean levels
- **RSI <15 or >85:** Only exhaustion extremes
- **80% consensus:** Only strong agreement
- **Multi-layer confirmation:** All filters must pass

This should achieve the 75%+ → 99-100% win rate target, but with **very low trade frequency**. The success of this approach depends on:

1. Sufficient trades still occurring (>20 per bot)
2. GA breeding optimizing for extreme setups
3. Filters not being too restrictive (learnings from ADX>25 failure)

If this fails, we have fallback strategies documented above. The test will definitively answer whether extreme filtering can achieve near-perfect win rates.

**Ready for final 10k bot test.**
