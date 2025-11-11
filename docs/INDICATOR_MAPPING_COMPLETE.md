# 📋 COMPLETE GPU INDICATOR MAPPING (Indices 0-49)

## **DEFINITIVE REFERENCE: GPU Kernel vs CPU Implementation**

This document maps ALL 50 indicators from the GPU kernel to their correct implementations.

---

## ✅ **CATEGORY 1: TREND INDICATORS (0-16)**

| Index | GPU Name | GPU Logic | CPU Implementation | Status |
|-------|----------|-----------|-------------------|--------|
| 0-2 | SMA | Simple Moving Average | `talib.SMA()` | ✅ OK |
| 3-5 | EMA | Exponential Moving Average | `talib.EMA()` | ✅ OK |
| 6-8 | WMA | Weighted Moving Average | `talib.WMA()` | ✅ OK |
| 9-11 | DEMA | Double EMA | `talib.DEMA()` | ✅ OK |
| 12-14 | TEMA | Triple EMA | `talib.TEMA()` | ✅ OK |
| 15-16 | HMA | Hull Moving Average | Custom formula | ⚠️ Verify |

---

## ❌ **CATEGORY 2: MOMENTUM INDICATORS (17-22)** - **MAJOR BUGS**

| Index | GPU Name | GPU Logic | CPU Implementation | Status |
|-------|----------|-----------|-------------------|--------|
| **17** | **Momentum** | `> 0 = bullish` | ✅ `talib.MOM()` | ✅ OK |
| **18** | **ROC** | `> 2% = bullish, < -2% = bearish` | ❌ Returns `MOM` instead of `ROC` | 🔴 **BUG** |
| **19** | **Williams %R** | `< -80 = oversold, > -20 = overbought` | ❌ Returns `MOM` instead of `WILLR` | 🔴 **BUG** |
| **20** | **ATR (20)** | Volatility expansion | ❌ Returns `ROC` instead of `ATR` | 🔴 **BUG** |
| **21** | **ATR (21)** | Volatility expansion | ❌ Returns `ROC` instead of `ATR` | 🔴 **BUG** |
| **22** | **NATR** | Normalized ATR (% of price) | ❌ Returns `ROC` instead of `NATR` | 🔴 **BUG** |

---

## ❌ **CATEGORY 3: VOLATILITY/BANDS (23-25)** - **MODERATE BUGS**

| Index | GPU Name | GPU Logic | CPU Implementation | Status |
|-------|----------|-----------|-------------------|--------|
| **23** | **Bollinger Upper** | Price near upper band | ✅ `talib.BBANDS()[0]` | ✅ OK |
| **24** | **Bollinger Lower** | Price near lower band | ❌ Returns **middle band** | 🔴 **BUG** |
| **25** | **Keltner Channel** | EMA ± ATR * multiplier | ❌ Returns **BB lower** | 🔴 **BUG** |

---

## ✅ **CATEGORY 4: OSCILLATORS (26-30)**

| Index | GPU Name | GPU Logic | CPU Implementation | Status |
|-------|----------|-----------|-------------------|--------|
| **26** | **MACD** | `> signal = bullish` | ✅ `talib.MACD()` | ✅ OK |
| **27** | **ADX** | Trend strength (no direction) | ✅ `talib.ADX()` | ✅ OK |
| **28** | **Aroon Up** | `> 70 = bullish, < 30 = bearish` | ❌ Returns `ADX` | 🔴 **BUG** |
| **29** | **CCI** | `< -100 = oversold, > 100 = overbought` | ✅ `talib.CCI()` | ✅ OK |
| **30** | **DPO** | Detrended Price Oscillator | ❌ Returns `Williams %R` | 🔴 **BUG** |

---

## ❌ **CATEGORY 5: TREND STRENGTH (31-35)** - **CRITICAL BUGS**

| Index | GPU Name | GPU Logic | CPU Implementation | Status |
|-------|----------|-----------|-------------------|--------|
| **31** | **Parabolic SAR** | Trailing stop, trend detection | ❌ Returns `ATR` | 🔴 **BUG** |
| **32** | **SuperTrend** | ATR-based trend indicator | ❌ Returns `ATR` | 🔴 **BUG** |
| **33** | **Linear Reg Slope (33)** | Positive = uptrend | ❌ Returns `ATR` | 🔴 **BUG** |
| **34** | **Linear Reg Slope (34)** | Positive = uptrend | ❌ Returns `SAR` | 🔴 **BUG** |
| **35** | **Linear Reg Slope (35)** | Positive = uptrend | ❌ Returns `SAR` | 🔴 **BUG** |

---

## ❌ **CATEGORY 6: VOLUME INDICATORS (36-40)** - **MAJOR BUGS**

| Index | GPU Name | GPU Logic | CPU Implementation | Status |
|-------|----------|-----------|-------------------|--------|
| **36** | **OBV** | On-Balance Volume (trend) | ✅ `talib.OBV()` | ✅ OK |
| **37** | **VWAP** | Volume-Weighted Average Price | ❌ Returns `AD` (Accumulation/Distribution) | 🔴 **BUG** |
| **38** | **MFI** | Money Flow Index (volume-weighted RSI) | ❌ Returns `ADOSC` | 🔴 **BUG** |
| **39** | **A/D** | Accumulation/Distribution | ❌ Returns `Volume SMA` | 🔴 **BUG** |
| **40** | **Volume SMA** | Volume trend (> 1.2x = breakout) | ✅ `talib.SMA(volumes)` (but indices 39-40 swapped) | ⚠️ **SWAPPED** |

---

## ❌ **CATEGORY 7: PATTERN INDICATORS (41-45)** - **COMPLETE MISMATCH**

| Index | GPU Name | GPU Logic | CPU Implementation | Status |
|-------|----------|-----------|-------------------|--------|
| **41** | **Pivot Points** | Support/resistance levels | ❌ Returns `Typical Price` | 🔴 **BUG** |
| **42** | **Fractal High** | Local maximum (resistance) | ❌ Returns `Median Price` | 🔴 **BUG** |
| **43** | **Fractal Low** | Local minimum (support) | ❌ Returns `Weighted Close` | 🔴 **BUG** |
| **44** | **Support/Resistance** | Dynamic S/R levels | ❌ Returns `Price change` | 🔴 **BUG** |
| **45** | **Price Channel** | Highest high / lowest low | ❌ Returns `Price change %` | 🔴 **BUG** |

---

## ❌ **CATEGORY 8: SIMPLE INDICATORS (46-49)** - **PARTIAL MISMATCH**

| Index | GPU Name | GPU Logic | CPU Implementation | Status |
|-------|----------|-----------|-------------------|--------|
| **46** | **High-Low Range** | `highs[-1] - lows[-1]` | ✅ Same formula | ✅ OK |
| **47** | **Close Position in Range** | `(close - low) / (high - low)` (0.7 = bullish, 0.3 = bearish) | ❌ Returns `(high - low) / low * 100` | 🔴 **BUG** |
| **48** | **Price Acceleration** | Second derivative (rate of change of ROC) | ❌ Returns `close - open` | 🔴 **BUG** |
| **49** | **Volume ROC** | Volume rate of change (> 10% = bullish) | ❌ Returns simple `volume[-1] - volume[-2]` | 🔴 **BUG** |

---

## 📊 **BUG SUMMARY**

### **Critical Issues Found**: **26 out of 50 indicators (52%)**

| Category | Total | Bugs | Status |
|----------|-------|------|--------|
| Trend (0-16) | 17 | 1 | 94% OK |
| Momentum (17-22) | 6 | 5 | **17% OK** 🔴 |
| Volatility (23-25) | 3 | 2 | **33% OK** 🔴 |
| Oscillators (26-30) | 5 | 2 | **60% OK** 🟡 |
| Trend Strength (31-35) | 5 | 5 | **0% OK** 🔴 |
| Volume (36-40) | 5 | 3 | **40% OK** 🔴 |
| Patterns (41-45) | 5 | 5 | **0% OK** 🔴 |
| Simple (46-49) | 4 | 3 | **25% OK** 🔴 |

### **Overall Accuracy**: **48% (24 out of 50 correct)**

---

## 🔧 **REQUIRED FIXES**

### **1. Fix Momentum Indicators (18-22)**
```python
# ROC (18)
elif indicator_index == 18:
    period = int(param0)
    return talib.ROC(closes, timeperiod=period)[-1]

# Williams %R (19)
elif indicator_index == 19:
    period = int(param0)
    return talib.WILLR(highs, lows, closes, timeperiod=period)[-1]

# ATR (20-21)
elif indicator_index in [20, 21]:
    period = int(param0)
    return talib.ATR(highs, lows, closes, timeperiod=period)[-1]

# NATR (22)
elif indicator_index == 22:
    period = int(param0)
    return talib.NATR(highs, lows, closes, timeperiod=period)[-1]
```

### **2. Fix Bollinger/Keltner (24-25)**
```python
# Bollinger Lower (24)
elif indicator_index == 24:
    period = int(param0)
    stddev = param1 if param1 > 0 else 2.0
    upper, middle, lower = talib.BBANDS(closes, timeperiod=period, 
                                        nbdevup=stddev, nbdevdn=stddev)
    return lower[-1]  # NOT MIDDLE!

# Keltner Channel (25)
elif indicator_index == 25:
    period = int(param0) if param0 > 0 else 20
    atr_period = int(param1) if param1 > 0 else 10
    multiplier = param2 if param2 > 0 else 2.0
    
    ema = talib.EMA(closes, timeperiod=period)
    atr = talib.ATR(highs, lows, closes, timeperiod=atr_period)
    
    keltner_upper = ema[-1] + (atr[-1] * multiplier)
    keltner_lower = ema[-1] - (atr[-1] * multiplier)
    
    # Return upper band (or implement full channel logic)
    return keltner_upper
```

### **3. Fix Aroon/DPO (28, 30)**
```python
# Aroon Up (28)
elif indicator_index == 28:
    period = int(param0)
    aroon_down, aroon_up = talib.AROON(highs, lows, timeperiod=period)
    return aroon_up[-1]

# DPO (30)
elif indicator_index == 30:
    period = int(param0) if param0 > 0 else 20
    shift = int(period / 2) + 1
    
    # DPO = Close - SMA(shifted back)
    sma = talib.SMA(closes, timeperiod=period)
    if len(sma) >= shift:
        dpo = closes[-1] - sma[-shift]
    else:
        dpo = 0.0
    return dpo
```

### **4. Fix SAR/SuperTrend/Linear Reg (31-35)**
```python
# Parabolic SAR (31)
elif indicator_index == 31:
    acceleration = param0 if param0 > 0 else 0.02
    maximum = param1 if param1 > 0 else 0.2
    return talib.SAR(highs, lows, acceleration=acceleration, maximum=maximum)[-1]

# SuperTrend (32)
elif indicator_index == 32:
    period = int(param0) if param0 > 0 else 10
    multiplier = param1 if param1 > 0 else 3.0
    
    # SuperTrend = EMA ± ATR * multiplier
    atr = talib.ATR(highs, lows, closes, timeperiod=period)
    hl_avg = (highs + lows) / 2
    
    basic_upper = hl_avg[-1] + (multiplier * atr[-1])
    basic_lower = hl_avg[-1] - (multiplier * atr[-1])
    
    # Simplified: return upper band if price below it, else lower
    return basic_upper if closes[-1] < hl_avg[-1] else basic_lower

# Linear Regression Slope (33-35)
elif indicator_index in [33, 34, 35]:
    period = int(param0) if param0 > 0 else 20
    slope = talib.LINEARREG_SLOPE(closes, timeperiod=period)[-1]
    return slope
```

### **5. Fix Volume Indicators (37-40)**
```python
# VWAP (37) - NOT AVAILABLE IN TALIB, NEEDS CUSTOM CALCULATION
elif indicator_index == 37:
    # VWAP = SUM(price * volume) / SUM(volume)
    typical_price = (highs + lows + closes) / 3
    vwap = np.sum(typical_price * volumes) / np.sum(volumes)
    return vwap

# MFI (38)
elif indicator_index == 38:
    period = int(param0) if param0 > 0 else 14
    return talib.MFI(highs, lows, closes, volumes, timeperiod=period)[-1]

# A/D (39)
elif indicator_index == 39:
    return talib.AD(highs, lows, closes, volumes)[-1]

# Volume SMA (40)
elif indicator_index == 40:
    period = int(param0) if param0 > 0 else 20
    return talib.SMA(volumes, timeperiod=period)[-1]
```

### **6. Fix Pattern Indicators (41-45)**
```python
# Pivot Points (41)
elif indicator_index == 41:
    # Classic pivot = (High + Low + Close) / 3
    # But GPU uses trending pivot, check previous bar
    pivot = (highs[-2] + lows[-2] + closes[-2]) / 3 if len(closes) >= 2 else closes[-1]
    return pivot

# Fractal High (42) - REQUIRES CUSTOM IMPLEMENTATION
elif indicator_index == 42:
    # Fractal high: middle bar higher than 2 bars on each side
    if len(highs) >= 5:
        if highs[-3] > highs[-5] and highs[-3] > highs[-4] and \
           highs[-3] > highs[-2] and highs[-3] > highs[-1]:
            return 1.0  # Fractal detected
    return 0.0

# Fractal Low (43) - REQUIRES CUSTOM IMPLEMENTATION
elif indicator_index == 43:
    # Fractal low: middle bar lower than 2 bars on each side
    if len(lows) >= 5:
        if lows[-3] < lows[-5] and lows[-3] < lows[-4] and \
           lows[-3] < lows[-2] and lows[-3] < lows[-1]:
            return 1.0  # Fractal detected
    return 0.0

# Support/Resistance (44) - REQUIRES CUSTOM IMPLEMENTATION
elif indicator_index == 44:
    # Dynamic S/R using recent highs/lows
    lookback = int(param0) if param0 > 0 else 20
    if len(highs) >= lookback:
        resistance = np.max(highs[-lookback:])
        support = np.min(lows[-lookback:])
        # Return which level is closer
        dist_to_resistance = resistance - closes[-1]
        dist_to_support = closes[-1] - support
        return resistance if dist_to_resistance < dist_to_support else support
    return closes[-1]

# Price Channel (45)
elif indicator_index == 45:
    period = int(param0) if param0 > 0 else 20
    if len(highs) >= period:
        highest = np.max(highs[-period:])
        lowest = np.min(lows[-period:])
        # Return midpoint or upper channel
        return (highest + lowest) / 2
    return closes[-1]
```

### **7. Fix Simple Indicators (47-49)**
```python
# Close Position in Range (47)
elif indicator_index == 47:
    # (Close - Low) / (High - Low)
    range_size = highs[-1] - lows[-1]
    if range_size > 0:
        return (closes[-1] - lows[-1]) / range_size
    return 0.5

# Price Acceleration (48)
elif indicator_index == 48:
    # Second derivative: rate of change of momentum
    if len(closes) >= 3:
        mom1 = closes[-1] - closes[-2]
        mom2 = closes[-2] - closes[-3]
        acceleration = mom1 - mom2
        return acceleration
    return 0.0

# Volume ROC (49)
elif indicator_index == 49:
    period = int(param0) if param0 > 0 else 1
    return talib.ROC(volumes, timeperiod=period)[-1]
```

---

## ⚠️ **CRITICAL NOTES**

1. **VWAP (37)** requires cumulative calculation from session start (not available in TALib)
2. **SuperTrend (32)** needs state tracking for proper trend flips
3. **Fractals (42-43)** require lookback pattern matching
4. **Support/Resistance (44)** needs custom level detection
5. **All fixes must match GPU signal generation logic exactly**

---

## ✅ **VALIDATION STEPS**

1. Fix all 26 buggy indicators
2. Run test comparing CPU vs GPU on same data
3. Verify signal generation matches
4. Test 100% consensus with corrected indicators
5. Validate paper trading matches backtest

---

*Last Updated: November 11, 2025*  
*Total Bugs Found: 26/50 (52%)*  
*Priority: CRITICAL - DO NOT DEPLOY*
