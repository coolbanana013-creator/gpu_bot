# MULTI-TIMEFRAME SIGNAL IMPLEMENTATION PLAN

## Objective
Implement multi-timeframe analysis to dramatically improve win rates by ensuring trades align with higher timeframe trends. This prevents trading against the dominant trend and significantly reduces false signals.

## Core Concept: Timeframe Alignment

### Rule: Don't Fight Higher Timeframes
- **1m trades** must align with 5m and 15m trends
- **5m trades** must align with 15m and 1h trends  
- **15m trades** must align with 1h and 4h trends
- **30m trades** must align with 1h and 4h trends
- **1h trades** must align with 4h and 1d trends
- **4h trades** must align with 1d trend
- **1d trades** (no higher TF check needed)

### Implementation Strategy

#### 1. Multi-Timeframe Data Preparation
- Calculate indicators on multiple timeframes simultaneously
- Store higher TF indicators alongside base TF
- Minimal memory overhead (just 2-3x indicator count)

#### 2. Trend Determination Algorithm
For each higher timeframe, determine trend using:
- **Fast EMA vs Slow EMA** (EMA20 vs EMA50)
- **MACD sign** (above/below zero line)
- **ADX strength** (>25 = strong trend, use direction)
- **Price momentum** (ROC or Momentum indicator)

**Consensus**: All 3-4 methods must agree for strong trend signal

#### 3. Signal Filter Logic
```
Base TF Signal = generate_signal_consensus() // existing 70% consensus
Higher TF1 Trend = calculate_trend(indicators_HTF1) 
Higher TF2 Trend = calculate_trend(indicators_HTF2)

Final Signal = Base TF Signal if:
  - (Base TF Signal == LONG) AND (HTF1 Trend >= 0) AND (HTF2 Trend >= 0)
  - (Base TF Signal == SHORT) AND (HTF1 Trend <= 0) AND (HTF2 Trend <= 0)
  
Otherwise: Final Signal = NEUTRAL (skip trade)
```

#### 4. Expected Impact on Metrics

**Win Rate**: 
- Current: ~0-5% (random signals)
- Expected: 45-60% (trend-aligned signals)
- Improvement: +40-55 percentage points

**Trade Frequency**:
- Current: 4.3 trades/cycle (70% consensus)
- Expected: 1-2 trades/cycle (multi-TF filter)
- Change: -50-75% trades, but MUCH higher quality

**Risk/Reward**:
- Current: Negative (losing on every trade)
- Expected: 1.5:1 to 2.5:1 (trend trades have momentum)
- Improvement: Positive expectancy

**Survival Rate**:
- Current: 0% (all random strategies fail)
- Expected: 5-15% (trend-aligned strategies survive)
- Improvement: Evolution can finally work!

## Implementation Steps

### Step 1: Modify Indicator Precomputation
- Add 2 higher timeframes per base timeframe
- Resample OHLCV data to higher TF
- Calculate key trend indicators (EMA20/50, MACD, ADX)

### Step 2: Add Trend Calculation Function
- New GPU kernel function: `calculate_trend_htf()`
- Fast EMA cross check
- MACD sign check
- Return: -1 (bearish), 0 (neutral), +1 (bullish)

### Step 3: Modify Signal Generation
- Add HTF trend checks to `generate_signal_consensus()`
- Filter signals that oppose higher TF trend
- Maintain 70% consensus on base TF, add HTF filter

### Step 4: Update Bot Config
- Add HTF settings to CompactBotConfig (optional)
- Default: auto-detect based on base timeframe
- Allow disabling for testing (compare w/ vs w/o MTF)

### Step 5: Testing & Validation
- Run same 10k bot population with MTF enabled
- Compare win rates, survival rates, trade quality
- Expect: Fewer trades, much higher win rate

## GPU Memory Optimization

**Additional Memory Required**:
- 2 higher timeframes × 3 indicators × 4 bytes/float = 24 bytes per bar
- For 288k bars: 6.9 MB (minimal overhead)
- Total precomputed buffer: ~60 MB → ~67 MB (12% increase)

**Performance Impact**:
- Trend calculation: <1ms per bot (simple EMA/MACD checks)
- Overall slowdown: <5% (negligible)
- Benefit: 10x+ improvement in strategy quality

## Risk/Reward Enhancement

### Additional Improvements for RR Ratio

#### 1. ATR-Based TP/SL
- TP = entry + (2.5 × ATR) for longs
- SL = entry - (1.0 × ATR) for longs
- Maintains 2.5:1 RR ratio

#### 2. Partial Profit Taking
- Close 50% at 1.5× ATR (lock in 1.5:1)
- Let 50% run to 2.5× ATR (for 2.5:1)
- Average RR: 2.0:1

#### 3. Trailing Stop After TP1
- After hitting TP1, trail stop to break-even
- Reduces risk to zero, maximizes winners
- Asymmetric payoff profile

## Expected Final Results

**With Multi-Timeframe + RR Enhancement**:
```
Win Rate:           50-60% (up from 0-5%)
Average RR:         2.0:1 (up from negative)
Survival Rate:      10-20% Gen0 (up from 0%)
Fitness Score:      Positive (many bots profitable)
Trade Quality:      High (trend-aligned only)
Evolution:          Working (survivors reproduce)
```

## Timeline
- Indicator precomputation: 2 hours
- Trend calculation function: 1 hour  
- Signal filter integration: 1 hour
- Testing & validation: 1 hour
- Total: ~5 hours implementation
