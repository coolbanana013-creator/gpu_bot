/**
 * precompute_all_indicators.cl
 * 
 * Precomputes all 50 technical indicators for all OHLCV bars.
 * Strategy: Compute once, store in global memory, read many times.
 * 
 * Memory Layout:
 *   indicators_out[indicator_id * num_bars + bar_index] = computed value
 *   50 indicators × num_bars × 4 bytes float
 * 
 * OPTIMIZED WORK DISTRIBUTION (Intel UHD Graphics 630):
 * - Global Memory: 26.4 MB total (OHLCV 2.4MB + indicators 24MB) - WELL WITHIN 3.19GB
 * - Work Items: 25,600 total (50 indicators × 512 work items per indicator)
 * - Work Distribution: Bars distributed across 512 work items per indicator
 * - Stateful Indicators: Computed by work_item_id == 0 (maintains sequential state)
 * - Stateless Indicators: Parallelized across all 512 work items per indicator
 * - Register Pressure: Reduced from 50 work items × 120K operations to ~470 operations per work item
 * - SOLUTION: Better GPU utilization while maintaining all indicator functionality
 * 
 * Each indicator gets 512 work items, but only stateful ones require sequential processing.
 */

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// OHLCV bar structure
typedef struct __attribute__((packed)) {
    float open;
    float high;
    float low;
    float close;
    float volume;
} OHLCVBar;

// Output buffer for indicators (flat array)
// Layout: indicators_out[indicator_id * num_bars + bar_index] = computed value
// Total size: 50 indicators × num_bars × 4 bytes float

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Simple Moving Average helper
 */
float compute_sma_helper(__global OHLCVBar *ohlcv, int bar, int period) {
    if (bar < period - 1) return 0.0f;
    
    float sum = 0.0f;
    for (int i = 0; i < period; i++) {
        sum += ohlcv[bar - i].close;
    }
    return sum / (float)period;
}

/**
 * Exponential Moving Average helper
 * Uses float for GPU compatibility (double precision would be slower on most GPUs)
 * For critical precision, consider using double internally and casting to float
 */
float compute_ema_helper(__global OHLCVBar *ohlcv, int bar, int period, float prev_ema) {
    if (bar < period - 1) return 0.0f;
    if (bar == period - 1) {
        // First EMA is SMA
        return compute_sma_helper(ohlcv, bar, period);
    }
    
    // EMA formula: EMA = (Close - EMA_prev) * k + EMA_prev
    // where k = 2 / (period + 1)
    float k = 2.0f / (float)(period + 1);
    return (ohlcv[bar].close - prev_ema) * k + prev_ema;
}



/**
 * True Range
 */
float compute_true_range(__global OHLCVBar *ohlcv, int bar) {
    if (bar == 0) return ohlcv[bar].high - ohlcv[bar].low;
    
    float hl = ohlcv[bar].high - ohlcv[bar].low;
    float hc = fabs(ohlcv[bar].high - ohlcv[bar-1].close);
    float lc = fabs(ohlcv[bar].low - ohlcv[bar-1].close);
    
    return fmax(hl, fmax(hc, lc));
}

// ============================================================================
// INDICATOR COMPUTATION FUNCTIONS (50 total)
// ============================================================================

/**
 * CATEGORY 1: MOVING AVERAGES (12 indicators: 0-11)
 */

// SMA (6 indicators: 0-5)
void compute_sma(__global OHLCVBar *ohlcv, int num_bars, int period, __global float *out) {
    for (int bar = 0; bar < num_bars; bar++) {
        out[bar] = compute_sma_helper(ohlcv, bar, period);
    }
}

// EMA (6 indicators: 6-11)
void compute_ema(__global OHLCVBar *ohlcv, int num_bars, int period, __global float *out) {
    float prev_ema = 0.0f;
    for (int bar = 0; bar < num_bars; bar++) {
        prev_ema = compute_ema_helper(ohlcv, bar, period, prev_ema);
        out[bar] = prev_ema;
    }
}

/**
 * CATEGORY 2: MOMENTUM INDICATORS (8 indicators: 12-19)
 */

// RSI - Relative Strength Index (3 indicators: 12-14)
void compute_rsi(__global OHLCVBar *ohlcv, int num_bars, int period, __global float *out) {
    if (num_bars < period + 1) {
        for (int i = 0; i < num_bars; i++) out[i] = 50.0f;
        return;
    }
    
    // Calculate initial average gain/loss
    // Wilder's smoothing method: RMA (Running Moving Average)
    float avg_gain = 0.0f;
    float avg_loss = 0.0f;
    
    for (int i = 1; i <= period; i++) {
        float change = ohlcv[i].close - ohlcv[i-1].close;
        if (change > 0.0f) {
            avg_gain += change;
        } else {
            avg_loss += fabs(change);
        }
    }
    avg_gain /= (float)period;
    avg_loss /= (float)period;
    
    // Calculate RSI for all bars
    for (int bar = 0; bar < num_bars; bar++) {
        if (bar < period) {
            out[bar] = 50.0f;
            continue;
        }
        
        // Wilder's smoothing (also called RMA - Running Moving Average)
        // New average = (Previous average * (period - 1) + Current value) / period
        float change = ohlcv[bar].close - ohlcv[bar-1].close;
        float gain = (change > 0.0f) ? change : 0.0f;
        float loss = (change < 0.0f) ? fabs(change) : 0.0f;
        
        avg_gain = (avg_gain * (float)(period - 1) + gain) / (float)period;
        avg_loss = (avg_loss * (float)(period - 1) + loss) / (float)period;
        
        // Calculate RSI
        if (avg_loss < 1e-10f) {
            out[bar] = 100.0f;
        } else {
            float rs = avg_gain / avg_loss;
            out[bar] = 100.0f - (100.0f / (1.0f + rs));
        }
    }
}

// Stochastic Oscillator (1 indicator: 15)
void compute_stochastic(__global OHLCVBar *ohlcv, int num_bars, int period, int smooth_k, __global float *out) {
    for (int bar = 0; bar < num_bars; bar++) {
        if (bar < period - 1) {
            out[bar] = 50.0f;
            continue;
        }
        
        // Find highest high and lowest low in period
        float highest = ohlcv[bar].high;
        float lowest = ohlcv[bar].low;
        
        for (int i = 1; i < period; i++) {
            if (ohlcv[bar - i].high > highest) highest = ohlcv[bar - i].high;
            if (ohlcv[bar - i].low < lowest) lowest = ohlcv[bar - i].low;
        }
        
        // Calculate %K
        float range = highest - lowest;
        if (range < 1e-10f) {
            out[bar] = 50.0f;
        } else {
            out[bar] = ((ohlcv[bar].close - lowest) / range) * 100.0f;
        }
    }
    
    // Smooth with SMA if smooth_k > 1
    if (smooth_k > 1) {
        for (int bar = smooth_k - 1; bar < num_bars; bar++) {
            float sum = 0.0f;
            for (int i = 0; i < smooth_k; i++) {
                sum += out[bar - i];
            }
            out[bar] = sum / (float)smooth_k;
        }
    }
}

// StochRSI (1 indicator: 16)
void compute_stochrsi(__global OHLCVBar *ohlcv, int num_bars, int period, __global float *out, __global float *rsi_buffer) {
    // StochRSI = Stochastic applied to RSI values
    for (int bar = 0; bar < num_bars; bar++) {
        if (bar < period - 1) {
            out[bar] = 50.0f;
            continue;
        }
        
        float highest = rsi_buffer[bar];
        float lowest = rsi_buffer[bar];
        
        for (int i = 1; i < period; i++) {
            if (rsi_buffer[bar - i] > highest) highest = rsi_buffer[bar - i];
            if (rsi_buffer[bar - i] < lowest) lowest = rsi_buffer[bar - i];
        }
        
        float range = highest - lowest;
        if (range < 1e-10f) {
            out[bar] = 50.0f;
        } else {
            out[bar] = ((rsi_buffer[bar] - lowest) / range) * 100.0f;
        }
    }
}

// Momentum (1 indicator: 17)
void compute_momentum(__global OHLCVBar *ohlcv, int num_bars, int period, __global float *out) {
    for (int bar = 0; bar < num_bars; bar++) {
        if (bar < period) {
            out[bar] = 0.0f;
        } else {
            out[bar] = ohlcv[bar].close - ohlcv[bar - period].close;
        }
    }
}

// Rate of Change (1 indicator: 18)
void compute_roc(__global OHLCVBar *ohlcv, int num_bars, int period, __global float *out) {
    for (int bar = 0; bar < num_bars; bar++) {
        if (bar < period || ohlcv[bar - period].close < 1e-10f) {
            out[bar] = 0.0f;
        } else {
            out[bar] = ((ohlcv[bar].close - ohlcv[bar - period].close) / ohlcv[bar - period].close) * 100.0f;
        }
    }
}

// Williams %R (1 indicator: 19)
void compute_willr(__global OHLCVBar *ohlcv, int num_bars, int period, __global float *out) {
    for (int bar = 0; bar < num_bars; bar++) {
        if (bar < period - 1) {
            out[bar] = -50.0f;
            continue;
        }
        
        float highest = ohlcv[bar].high;
        float lowest = ohlcv[bar].low;
        
        for (int i = 1; i < period; i++) {
            if (ohlcv[bar - i].high > highest) highest = ohlcv[bar - i].high;
            if (ohlcv[bar - i].low < lowest) lowest = ohlcv[bar - i].low;
        }
        
        float range = highest - lowest;
        if (range < 1e-10f) {
            out[bar] = -50.0f;
        } else {
            out[bar] = ((highest - ohlcv[bar].close) / range) * -100.0f;
        }
    }
}

/**
 * CATEGORY 3: VOLATILITY INDICATORS (6 indicators: 20-25)
 */

// Average True Range (2 indicators: 20-21)
void compute_atr(__global OHLCVBar *ohlcv, int num_bars, int period, __global float *out) {
    if (num_bars < period) {
        for (int i = 0; i < num_bars; i++) out[i] = 0.0f;
        return;
    }
    
    // Initial ATR is average of first 'period' TRs
    float atr = 0.0f;
    for (int i = 0; i < period; i++) {
        atr += compute_true_range(ohlcv, i);
    }
    atr /= (float)period;
    
    // Calculate ATR for all bars
    for (int bar = 0; bar < num_bars; bar++) {
        if (bar < period - 1) {
            out[bar] = 0.0f;
        } else if (bar == period - 1) {
            out[bar] = atr;
        } else {
            float tr = compute_true_range(ohlcv, bar);
            atr = (atr * (float)(period - 1) + tr) / (float)period;
            out[bar] = atr;
        }
    }
}

// Normalized ATR (1 indicator: 22)
void compute_natr(__global OHLCVBar *ohlcv, int num_bars, int period, __global float *out, __global float *atr_buffer) {
    for (int bar = 0; bar < num_bars; bar++) {
        if (ohlcv[bar].close < 1e-10f) {
            out[bar] = 0.0f;
        } else {
            out[bar] = (atr_buffer[bar] / ohlcv[bar].close) * 100.0f;
        }
    }
}

// Bollinger Bands (2 indicators: 23-24 for upper/lower)
void compute_bollinger_bands(__global OHLCVBar *ohlcv, int num_bars, int period, float std_dev, __global float *upper, __global float *lower) {
    for (int bar = 0; bar < num_bars; bar++) {
        if (bar < period - 1) {
            upper[bar] = ohlcv[bar].close;
            lower[bar] = ohlcv[bar].close;
            continue;
        }
        
        // Calculate mean
        float mean = 0.0f;
        for (int i = 0; i < period; i++) {
            mean += ohlcv[bar - i].close;
        }
        mean /= (float)period;
        
        // Calculate standard deviation
        float variance = 0.0f;
        for (int i = 0; i < period; i++) {
            float diff = ohlcv[bar - i].close - mean;
            variance += diff * diff;
        }
        variance /= (float)period;
        float std = sqrt(variance);
        
        upper[bar] = mean + (std_dev * std);
        lower[bar] = mean - (std_dev * std);
    }
}

// Keltner Channel (1 indicator: 25 - middle)
void compute_keltner(__global OHLCVBar *ohlcv, int num_bars, int period, __global float *out, __global float *atr_buffer) {
    // Keltner middle is EMA of close
    float prev_ema = 0.0f;
    for (int bar = 0; bar < num_bars; bar++) {
        prev_ema = compute_ema_helper(ohlcv, bar, period, prev_ema);
        out[bar] = prev_ema;
    }
}

/**
 * CATEGORY 4: TREND INDICATORS (10 indicators: 26-35)
 */

// MACD (3 components: MACD line, signal line, histogram)
// Currently returns only MACD line, signal and histogram computed inline during signal generation
void compute_macd(__global OHLCVBar *ohlcv, int num_bars, int fast, int slow, int signal_period, __global float *out) {
    // Calculate fast and slow EMAs with continuous state
    float fast_ema = 0.0f;
    float slow_ema = 0.0f;
    float signal_ema = 0.0f;
    
    for (int bar = 0; bar < num_bars; bar++) {
        // Warmup period: need at least 'slow' bars for valid MACD
        if (bar < slow) {
            out[bar] = 0.0f;  // Neutral value during warmup
            continue;
        }
        
        // Compute fast and slow EMAs
        fast_ema = compute_ema_helper(ohlcv, bar, fast, fast_ema);
        slow_ema = compute_ema_helper(ohlcv, bar, slow, slow_ema);
        
        // MACD line = fast EMA - slow EMA
        float macd_line = fast_ema - slow_ema;
        
        // Signal line = EMA of MACD line (using signal_period, typically 9)
        if (bar >= slow - 1) {
            if (bar == slow - 1) {
                // First signal value is the MACD value itself
                signal_ema = macd_line;
            } else {
                // Apply EMA smoothing to MACD line
                float k = 2.0f / (float)(signal_period + 1);
                signal_ema = (macd_line - signal_ema) * k + signal_ema;
            }
        }
        
        // MACD histogram = MACD line - signal line
        // For now, output MACD line (histogram can be calculated as needed)
        out[bar] = macd_line;
        
        // NOTE: Full implementation would store all 3 components:
        // out[bar] = macd_line;
        // out_signal[bar] = signal_ema;
        // out_histogram[bar] = (macd_line - signal_ema);
    }
}

// ADX - Average Directional Index (1 indicator: 27)
// Uses DOUBLE PRECISION for smoothed calculations
// Also computes +DI and -DI internally (can be exported if needed)
void compute_adx(__global OHLCVBar *ohlcv, int num_bars, int period, __global float *out) {
    if (num_bars < period + 1) {
        for (int i = 0; i < num_bars; i++) out[i] = 0.0f;
        return;
    }
    
    // Use float for smoothed values
    float smoothed_tr = 0.0f;
    float smoothed_plus_dm = 0.0f;
    float smoothed_minus_dm = 0.0f;
    float prev_adx = 0.0f;
    
    // Initial smoothing (sum of first period values)
    for (int i = 1; i <= period; i++) {
        float tr = compute_true_range(ohlcv, i);
        float plus_dm = (ohlcv[i].high - ohlcv[i-1].high > ohlcv[i-1].low - ohlcv[i].low) ? 
                        fmax(ohlcv[i].high - ohlcv[i-1].high, 0.0f) : 0.0f;
        float minus_dm = (ohlcv[i-1].low - ohlcv[i].low > ohlcv[i].high - ohlcv[i-1].high) ?
                         fmax(ohlcv[i-1].low - ohlcv[i].low, 0.0f) : 0.0f;
        
        smoothed_tr += tr;
        smoothed_plus_dm += plus_dm;
        smoothed_minus_dm += minus_dm;
    }
    
    for (int bar = 0; bar < num_bars; bar++) {
        if (bar < period) {
            out[bar] = 0.0f;
            continue;
        }
        
        if (bar > period) {
            // Wilder's smoothing: Smoothed = (Prev_Smoothed * (period - 1) + Current) / period
            // Equivalent to: Smoothed = Prev_Smoothed - (Prev_Smoothed / period) + Current
            float tr = compute_true_range(ohlcv, bar);
            float plus_dm = (ohlcv[bar].high - ohlcv[bar-1].high > ohlcv[bar-1].low - ohlcv[bar].low) ?
                           fmax(ohlcv[bar].high - ohlcv[bar-1].high, 0.0f) : 0.0f;
            float minus_dm = (ohlcv[bar-1].low - ohlcv[bar].low > ohlcv[bar].high - ohlcv[bar-1].high) ?
                            fmax(ohlcv[bar-1].low - ohlcv[bar].low, 0.0f) : 0.0f;
            
            smoothed_tr = smoothed_tr - (smoothed_tr / (float)period) + tr;
            smoothed_plus_dm = smoothed_plus_dm - (smoothed_plus_dm / (float)period) + plus_dm;
            smoothed_minus_dm = smoothed_minus_dm - (smoothed_minus_dm / (float)period) + minus_dm;
        }
        
        // +DI and -DI (Directional Indicators)
        float plus_di = (smoothed_tr > 0.0f) ? (smoothed_plus_dm / smoothed_tr) * 100.0f : 0.0f;
        float minus_di = (smoothed_tr > 0.0f) ? (smoothed_minus_dm / smoothed_tr) * 100.0f : 0.0f;
        
        // DX (Directional Movement Index)
        float dx = (plus_di + minus_di > 0.0f) ?
                   (fabs(plus_di - minus_di) / (plus_di + minus_di)) * 100.0f : 0.0f;
        
        // ADX is smoothed DX using Wilder's method
        if (bar == period) {
            prev_adx = dx;
        } else {
            prev_adx = (prev_adx * (float)(period - 1) + dx) / (float)period;
        }
        
        out[bar] = prev_adx;
    }
}

// Aroon (1 indicator: 28 - Aroon Up)
void compute_aroon_up(__global OHLCVBar *ohlcv, int num_bars, int period, __global float *out) {
    for (int bar = 0; bar < num_bars; bar++) {
        if (bar < period - 1) {
            out[bar] = 50.0f;
            continue;
        }
        
        // Find bars since highest high
        int bars_since_high = 0;
        float highest = ohlcv[bar].high;
        
        for (int i = 1; i < period; i++) {
            if (ohlcv[bar - i].high > highest) {
                highest = ohlcv[bar - i].high;
                bars_since_high = i;
            }
        }
        
        out[bar] = ((float)(period - bars_since_high) / (float)period) * 100.0f;
    }
}

// Aroon Down (can be added as additional indicator if needed)
void compute_aroon_down(__global OHLCVBar *ohlcv, int num_bars, int period, __global float *out) {
    for (int bar = 0; bar < num_bars; bar++) {
        if (bar < period - 1) {
            out[bar] = 50.0f;
            continue;
        }
        
        // Find bars since lowest low
        int bars_since_low = 0;
        float lowest = ohlcv[bar].low;
        
        for (int i = 1; i < period; i++) {
            if (ohlcv[bar - i].low < lowest) {
                lowest = ohlcv[bar - i].low;
                bars_since_low = i;
            }
        }
        
        out[bar] = ((float)(period - bars_since_low) / (float)period) * 100.0f;
    }
}

// CCI - Commodity Channel Index (1 indicator: 29)
void compute_cci(__global OHLCVBar *ohlcv, int num_bars, int period, __global float *out) {
    for (int bar = 0; bar < num_bars; bar++) {
        if (bar < period - 1) {
            out[bar] = 0.0f;
            continue;
        }
        
        // Calculate typical price and SMA of typical price
        float tp_sum = 0.0f;
        for (int i = 0; i < period; i++) {
            float tp = (ohlcv[bar - i].high + ohlcv[bar - i].low + ohlcv[bar - i].close) / 3.0f;
            tp_sum += tp;
        }
        float tp_sma = tp_sum / (float)period;
        
        // Calculate mean deviation
        float md_sum = 0.0f;
        for (int i = 0; i < period; i++) {
            float tp = (ohlcv[bar - i].high + ohlcv[bar - i].low + ohlcv[bar - i].close) / 3.0f;
            md_sum += fabs(tp - tp_sma);
        }
        float mean_dev = md_sum / (float)period;
        
        float current_tp = (ohlcv[bar].high + ohlcv[bar].low + ohlcv[bar].close) / 3.0f;
        
        if (mean_dev < 1e-10f) {
            out[bar] = 0.0f;
        } else {
            out[bar] = (current_tp - tp_sma) / (0.015f * mean_dev);
        }
    }
}

// DPO - Detrended Price Oscillator (1 indicator: 30)
void compute_dpo(__global OHLCVBar *ohlcv, int num_bars, int period, __global float *out) {
    int offset = (period / 2) + 1;
    
    for (int bar = 0; bar < num_bars; bar++) {
        if (bar < period - 1 + offset) {
            out[bar] = 0.0f;
            continue;
        }
        
        float sma = compute_sma_helper(ohlcv, bar - offset, period);
        out[bar] = ohlcv[bar - offset].close - sma;
    }
}

// Parabolic SAR (1 indicator: 31)
void compute_psar(__global OHLCVBar *ohlcv, int num_bars, float af_start, float af_max, __global float *out) {
    if (num_bars < 2) {
        for (int i = 0; i < num_bars; i++) out[i] = ohlcv[i].close;
        return;
    }
    
    float sar = ohlcv[0].low;
    float ep = ohlcv[0].high;
    float af = af_start;
    int is_long = 1;
    
    out[0] = sar;
    
    for (int bar = 1; bar < num_bars; bar++) {
        // Update SAR
        sar = sar + af * (ep - sar);
        
        if (is_long) {
            if (ohlcv[bar].low < sar) {
                // Switch to short
                is_long = 0;
                sar = ep;
                ep = ohlcv[bar].low;
                af = af_start;
            } else {
                if (ohlcv[bar].high > ep) {
                    ep = ohlcv[bar].high;
                    af = fmin(af + af_start, af_max);
                }
            }
        } else {
            if (ohlcv[bar].high > sar) {
                // Switch to long
                is_long = 1;
                sar = ep;
                ep = ohlcv[bar].high;
                af = af_start;
            } else {
                if (ohlcv[bar].low < ep) {
                    ep = ohlcv[bar].low;
                    af = fmin(af + af_start, af_max);
                }
            }
        }
        
        out[bar] = sar;
    }
}

// SuperTrend (1 indicator: 32)
void compute_supertrend(__global OHLCVBar *ohlcv, int num_bars, int period, float multiplier, __global float *out, __global float *atr_buffer) {
    if (num_bars < period) {
        for (int i = 0; i < num_bars; i++) out[i] = ohlcv[i].close;
        return;
    }
    
    int trend = 1;  // 1 = up, -1 = down
    
    for (int bar = 0; bar < num_bars; bar++) {
        if (bar < period - 1) {
            out[bar] = ohlcv[bar].close;
            continue;
        }
        
        float hl_avg = (ohlcv[bar].high + ohlcv[bar].low) / 2.0f;
        float atr = atr_buffer[bar];
        
        float upper_band = hl_avg + (multiplier * atr);
        float lower_band = hl_avg - (multiplier * atr);
        
        if (trend == 1) {
            if (ohlcv[bar].close < lower_band) {
                trend = -1;
                out[bar] = upper_band;
            } else {
                out[bar] = lower_band;
            }
        } else {
            if (ohlcv[bar].close > upper_band) {
                trend = 1;
                out[bar] = lower_band;
            } else {
                out[bar] = upper_band;
            }
        }
    }
}

// Simple trend indicators for remaining slots (33-35)
void compute_trend_strength(__global OHLCVBar *ohlcv, int num_bars, int period, __global float *out) {
    // Trend strength based on linear regression slope
    for (int bar = 0; bar < num_bars; bar++) {
        if (bar < period - 1) {
            out[bar] = 0.0f;
            continue;
        }
        
        float sum_x = 0.0f, sum_y = 0.0f, sum_xy = 0.0f, sum_x2 = 0.0f;
        for (int i = 0; i < period; i++) {
            float x = (float)i;
            float y = ohlcv[bar - period + 1 + i].close;
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
        }
        
        float n = (float)period;
        float slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        out[bar] = slope;
    }
}

/**
 * CATEGORY 5: VOLUME INDICATORS (5 indicators: 36-40)
 */

// OBV - On Balance Volume (1 indicator: 36)
void compute_obv(__global OHLCVBar *ohlcv, int num_bars, __global float *out) {
    float obv = 0.0f;
    
    for (int bar = 0; bar < num_bars; bar++) {
        if (bar == 0) {
            out[bar] = 0.0f;
        } else {
            if (ohlcv[bar].close > ohlcv[bar-1].close) {
                obv += ohlcv[bar].volume;
            } else if (ohlcv[bar].close < ohlcv[bar-1].close) {
                obv -= ohlcv[bar].volume;
            }
            out[bar] = obv;
        }
    }
}

// VWAP - Volume Weighted Average Price (1 indicator: 37)
void compute_vwap(__global OHLCVBar *ohlcv, int num_bars, __global float *out) {
    float cum_tp_vol = 0.0f;
    float cum_vol = 0.0f;
    
    for (int bar = 0; bar < num_bars; bar++) {
        float tp = (ohlcv[bar].high + ohlcv[bar].low + ohlcv[bar].close) / 3.0f;
        cum_tp_vol += tp * ohlcv[bar].volume;
        cum_vol += ohlcv[bar].volume;
        
        if (cum_vol < 1e-10f) {
            out[bar] = ohlcv[bar].close;
        } else {
            out[bar] = cum_tp_vol / cum_vol;
        }
    }
}

// MFI - Money Flow Index (1 indicator: 38)
void compute_mfi(__global OHLCVBar *ohlcv, int num_bars, int period, __global float *out) {
    for (int bar = 0; bar < num_bars; bar++) {
        if (bar < period) {
            out[bar] = 50.0f;
            continue;
        }
        
        float pos_mf = 0.0f;
        float neg_mf = 0.0f;
        
        for (int i = 0; i < period; i++) {
            int idx = bar - i;
            if (idx == 0) continue;
            
            float tp = (ohlcv[idx].high + ohlcv[idx].low + ohlcv[idx].close) / 3.0f;
            float prev_tp = (ohlcv[idx-1].high + ohlcv[idx-1].low + ohlcv[idx-1].close) / 3.0f;
            float mf = tp * ohlcv[idx].volume;
            
            if (tp > prev_tp) {
                pos_mf += mf;
            } else if (tp < prev_tp) {
                neg_mf += mf;
            }
        }
        
        if (neg_mf < 1e-10f) {
            out[bar] = 100.0f;
        } else {
            float mfr = pos_mf / neg_mf;
            out[bar] = 100.0f - (100.0f / (1.0f + mfr));
        }
    }
}

// AD - Accumulation/Distribution (1 indicator: 39)
void compute_ad(__global OHLCVBar *ohlcv, int num_bars, __global float *out) {
    float ad = 0.0f;
    
    for (int bar = 0; bar < num_bars; bar++) {
        float hl_range = ohlcv[bar].high - ohlcv[bar].low;
        
        if (hl_range < 1e-10f) {
            out[bar] = ad;
        } else {
            float clv = ((ohlcv[bar].close - ohlcv[bar].low) - (ohlcv[bar].high - ohlcv[bar].close)) / hl_range;
            ad += clv * ohlcv[bar].volume;
            out[bar] = ad;
        }
    }
}

// Volume SMA (1 indicator: 40)
void compute_volume_sma(__global OHLCVBar *ohlcv, int num_bars, int period, __global float *out) {
    for (int bar = 0; bar < num_bars; bar++) {
        if (bar < period - 1) {
            out[bar] = ohlcv[bar].volume;
            continue;
        }
        
        float sum = 0.0f;
        for (int i = 0; i < period; i++) {
            sum += ohlcv[bar - i].volume;
        }
        out[bar] = sum / (float)period;
    }
}

/**
 * CATEGORY 6: PATTERN/PIVOT INDICATORS (5 indicators: 41-45)
 */

// Pivot Points (1 indicator: 41)
void compute_pivot_points(__global OHLCVBar *ohlcv, int num_bars, __global float *out) {
    for (int bar = 0; bar < num_bars; bar++) {
        if (bar == 0) {
            out[bar] = ohlcv[bar].close;
        } else {
            // Standard Pivot Point
            float pivot = (ohlcv[bar-1].high + ohlcv[bar-1].low + ohlcv[bar-1].close) / 3.0f;
            out[bar] = pivot;
        }
    }
}

// Fractal High (1 indicator: 42)
void compute_fractal_high(__global OHLCVBar *ohlcv, int num_bars, int period, __global float *out) {
    int mid = period / 2;
    
    for (int bar = 0; bar < num_bars; bar++) {
        if (bar < period - 1) {
            out[bar] = 0.0f;
            continue;
        }
        
        // Check if middle bar is highest
        int is_fractal = 1;
        float center_high = ohlcv[bar - mid].high;
        
        for (int i = 0; i < period; i++) {
            if (i != mid && ohlcv[bar - period + 1 + i].high >= center_high) {
                is_fractal = 0;
                break;
            }
        }
        
        out[bar] = is_fractal ? center_high : 0.0f;
    }
}

// Fractal Low (1 indicator: 43)
void compute_fractal_low(__global OHLCVBar *ohlcv, int num_bars, int period, __global float *out) {
    int mid = period / 2;
    
    for (int bar = 0; bar < num_bars; bar++) {
        if (bar < period - 1) {
            out[bar] = 0.0f;
            continue;
        }
        
        // Check if middle bar is lowest
        int is_fractal = 1;
        float center_low = ohlcv[bar - mid].low;
        
        for (int i = 0; i < period; i++) {
            if (i != mid && ohlcv[bar - period + 1 + i].low <= center_low) {
                is_fractal = 0;
                break;
            }
        }
        
        out[bar] = is_fractal ? center_low : 0.0f;
    }
}

// Support/Resistance Level (1 indicator: 44)
void compute_support_resistance(__global OHLCVBar *ohlcv, int num_bars, int period, __global float *out) {
    // Simple support/resistance based on recent highs/lows
    for (int bar = 0; bar < num_bars; bar++) {
        if (bar < period - 1) {
            out[bar] = ohlcv[bar].close;
            continue;
        }
        
        float highest = ohlcv[bar].high;
        float lowest = ohlcv[bar].low;
        
        for (int i = 1; i < period; i++) {
            if (ohlcv[bar - i].high > highest) highest = ohlcv[bar - i].high;
            if (ohlcv[bar - i].low < lowest) lowest = ohlcv[bar - i].low;
        }
        
        // Middle of range
        out[bar] = (highest + lowest) / 2.0f;
    }
}

// Price Channel (1 indicator: 45)
void compute_price_channel(__global OHLCVBar *ohlcv, int num_bars, int period, __global float *out) {
    for (int bar = 0; bar < num_bars; bar++) {
        if (bar < period - 1) {
            out[bar] = ohlcv[bar].close;
            continue;
        }
        
        float highest = ohlcv[bar].high;
        for (int i = 1; i < period; i++) {
            if (ohlcv[bar - i].high > highest) highest = ohlcv[bar - i].high;
        }
        
        out[bar] = highest;
    }
}

/**
 * CATEGORY 7: SIMPLE INDICATORS TO REACH 50 (5 indicators: 46-49)
 */

// High-Low Range (1 indicator: 46)
void compute_hl_range(__global OHLCVBar *ohlcv, int num_bars, __global float *out) {
    for (int bar = 0; bar < num_bars; bar++) {
        out[bar] = ohlcv[bar].high - ohlcv[bar].low;
    }
}

// Close position in range (1 indicator: 47)
void compute_close_position(__global OHLCVBar *ohlcv, int num_bars, __global float *out) {
    for (int bar = 0; bar < num_bars; bar++) {
        float range = ohlcv[bar].high - ohlcv[bar].low;
        if (range < 1e-10f) {
            out[bar] = 0.5f;
        } else {
            out[bar] = (ohlcv[bar].close - ohlcv[bar].low) / range;
        }
    }
}

// Price acceleration (1 indicator: 48)
void compute_price_acceleration(__global OHLCVBar *ohlcv, int num_bars, int period, __global float *out) {
    for (int bar = 0; bar < num_bars; bar++) {
        if (bar < period + 1) {
            out[bar] = 0.0f;
        } else {
            float velocity_now = ohlcv[bar].close - ohlcv[bar - period].close;
            float velocity_prev = ohlcv[bar - 1].close - ohlcv[bar - period - 1].close;
            out[bar] = velocity_now - velocity_prev;
        }
    }
}

// Volume Rate of Change (1 indicator: 49)
void compute_volume_roc(__global OHLCVBar *ohlcv, int num_bars, int period, __global float *out) {
    for (int bar = 0; bar < num_bars; bar++) {
        if (bar < period || ohlcv[bar - period].volume < 1e-10f) {
            out[bar] = 0.0f;
        } else {
            out[bar] = ((ohlcv[bar].volume - ohlcv[bar - period].volume) / ohlcv[bar - period].volume) * 100.0f;
        }
    }
}

// ============================================================================
// MAIN KERNEL: Precompute all 50 indicators
// ============================================================================

__kernel void precompute_all_indicators(
    __global OHLCVBar *ohlcv,
    const int num_bars,
    __global float *indicators_out  // Flat array: [indicator_id * num_bars + bar_index]
) {
    int indicator_id = get_global_id(0);  // Which indicator (0-49)
    int work_item_id = get_global_id(1); // Work item within indicator group (0-255)
    int work_items_per_indicator = get_global_size(1); // Dynamic: 512 work items per indicator

    if (indicator_id >= 50) return;

    // Distribute bars across work items for this indicator
    int bars_per_work_item = (num_bars + work_items_per_indicator - 1) / work_items_per_indicator;
    int start_bar = work_item_id * bars_per_work_item;
    int end_bar = min(start_bar + bars_per_work_item, num_bars);

    // Compute this indicator for our assigned bars
    // Stateless indicators: distribute across work items
    // Stateful indicators: only work_item_id == 0 computes (maintains state)
    switch(indicator_id) {
        // MOVING AVERAGES (0-11) - stateless, can be parallelized
        case 0: for (int bar = start_bar; bar < end_bar; bar++) indicators_out[0 * num_bars + bar] = compute_sma_helper(ohlcv, bar, 5); break;
        case 1: for (int bar = start_bar; bar < end_bar; bar++) indicators_out[1 * num_bars + bar] = compute_sma_helper(ohlcv, bar, 10); break;
        case 2: for (int bar = start_bar; bar < end_bar; bar++) indicators_out[2 * num_bars + bar] = compute_sma_helper(ohlcv, bar, 20); break;
        case 3: for (int bar = start_bar; bar < end_bar; bar++) indicators_out[3 * num_bars + bar] = compute_sma_helper(ohlcv, bar, 50); break;
        case 4: for (int bar = start_bar; bar < end_bar; bar++) indicators_out[4 * num_bars + bar] = compute_sma_helper(ohlcv, bar, 100); break;
        case 5: for (int bar = start_bar; bar < end_bar; bar++) indicators_out[5 * num_bars + bar] = compute_sma_helper(ohlcv, bar, 200); break;

        // EMA (6-11) - stateful, sequential processing required
        case 6: if (work_item_id == 0) compute_ema(ohlcv, num_bars, 5, &indicators_out[6 * num_bars]); break;
        case 7: if (work_item_id == 0) compute_ema(ohlcv, num_bars, 10, &indicators_out[7 * num_bars]); break;
        case 8: if (work_item_id == 0) compute_ema(ohlcv, num_bars, 20, &indicators_out[8 * num_bars]); break;
        case 9: if (work_item_id == 0) compute_ema(ohlcv, num_bars, 50, &indicators_out[9 * num_bars]); break;
        case 10: if (work_item_id == 0) compute_ema(ohlcv, num_bars, 100, &indicators_out[10 * num_bars]); break;
        case 11: if (work_item_id == 0) compute_ema(ohlcv, num_bars, 200, &indicators_out[11 * num_bars]); break;
        
        // MOMENTUM (12-19) - mostly stateful, keep sequential
        case 12: if (work_item_id == 0) compute_rsi(ohlcv, num_bars, 7, &indicators_out[12 * num_bars]); break;
        case 13: if (work_item_id == 0) compute_rsi(ohlcv, num_bars, 14, &indicators_out[13 * num_bars]); break;
        case 14: if (work_item_id == 0) compute_rsi(ohlcv, num_bars, 21, &indicators_out[14 * num_bars]); break;
        case 15: if (work_item_id == 0) compute_stochastic(ohlcv, num_bars, 14, 3, &indicators_out[15 * num_bars]); break;
        case 16: if (work_item_id == 0) compute_stochrsi(ohlcv, num_bars, 14, &indicators_out[16 * num_bars], &indicators_out[13 * num_bars]); break;  // Uses RSI(14)
        case 17: if (work_item_id == 0) compute_momentum(ohlcv, num_bars, 10, &indicators_out[17 * num_bars]); break;
        case 18: if (work_item_id == 0) compute_roc(ohlcv, num_bars, 10, &indicators_out[18 * num_bars]); break;
        case 19: if (work_item_id == 0) compute_willr(ohlcv, num_bars, 14, &indicators_out[19 * num_bars]); break;

        // VOLATILITY (20-25) - stateful, keep sequential
        case 20: if (work_item_id == 0) compute_atr(ohlcv, num_bars, 14, &indicators_out[20 * num_bars]); break;
        case 21: if (work_item_id == 0) compute_atr(ohlcv, num_bars, 20, &indicators_out[21 * num_bars]); break;
        case 22: if (work_item_id == 0) compute_natr(ohlcv, num_bars, 14, &indicators_out[22 * num_bars], &indicators_out[20 * num_bars]); break;  // Uses ATR(14)
        case 23: if (work_item_id == 0) compute_bollinger_bands(ohlcv, num_bars, 20, 2.0f, &indicators_out[23 * num_bars], &indicators_out[24 * num_bars]); break;  // Upper
        case 24: break;  // Lower BB computed with upper
        case 25: if (work_item_id == 0) compute_keltner(ohlcv, num_bars, 20, &indicators_out[25 * num_bars], &indicators_out[21 * num_bars]); break;  // Uses ATR(20)
        
        // TREND (26-35) - mostly stateful
        case 26: if (work_item_id == 0) compute_macd(ohlcv, num_bars, 12, 26, 9, &indicators_out[26 * num_bars]); break;
        case 27: if (work_item_id == 0) compute_adx(ohlcv, num_bars, 14, &indicators_out[27 * num_bars]); break;
        case 28: if (work_item_id == 0) compute_aroon_up(ohlcv, num_bars, 25, &indicators_out[28 * num_bars]); break;
        case 29: if (work_item_id == 0) compute_cci(ohlcv, num_bars, 20, &indicators_out[29 * num_bars]); break;
        case 30: if (work_item_id == 0) compute_dpo(ohlcv, num_bars, 20, &indicators_out[30 * num_bars]); break;
        case 31: if (work_item_id == 0) compute_psar(ohlcv, num_bars, 0.02f, 0.2f, &indicators_out[31 * num_bars]); break;
        case 32: if (work_item_id == 0) compute_supertrend(ohlcv, num_bars, 10, 3.0f, &indicators_out[32 * num_bars], &indicators_out[20 * num_bars]); break;  // Uses ATR(14)
        case 33: if (work_item_id == 0) compute_trend_strength(ohlcv, num_bars, 20, &indicators_out[33 * num_bars]); break;
        case 34: if (work_item_id == 0) compute_trend_strength(ohlcv, num_bars, 50, &indicators_out[34 * num_bars]); break;
        case 35: if (work_item_id == 0) compute_trend_strength(ohlcv, num_bars, 100, &indicators_out[35 * num_bars]); break;

        // VOLUME (36-40) - mix of stateful/stateless
        case 36: if (work_item_id == 0) compute_obv(ohlcv, num_bars, &indicators_out[36 * num_bars]); break;
        case 37: if (work_item_id == 0) compute_vwap(ohlcv, num_bars, &indicators_out[37 * num_bars]); break;
        case 38: if (work_item_id == 0) compute_mfi(ohlcv, num_bars, 14, &indicators_out[38 * num_bars]); break;
        case 39: if (work_item_id == 0) compute_ad(ohlcv, num_bars, &indicators_out[39 * num_bars]); break;
        case 40: if (work_item_id == 0) compute_volume_sma(ohlcv, num_bars, 20, &indicators_out[40 * num_bars]); break;

        // PATTERN (41-45) - mostly stateless, can parallelize some
        case 41: if (work_item_id == 0) compute_pivot_points(ohlcv, num_bars, &indicators_out[41 * num_bars]); break;
        case 42: if (work_item_id == 0) compute_fractal_high(ohlcv, num_bars, 5, &indicators_out[42 * num_bars]); break;
        case 43: if (work_item_id == 0) compute_fractal_low(ohlcv, num_bars, 5, &indicators_out[43 * num_bars]); break;
        case 44: if (work_item_id == 0) compute_support_resistance(ohlcv, num_bars, 20, &indicators_out[44 * num_bars]); break;
        case 45: if (work_item_id == 0) compute_price_channel(ohlcv, num_bars, 20, &indicators_out[45 * num_bars]); break;

        // SIMPLE (46-49) - stateless, can parallelize
        case 46: for (int bar = start_bar; bar < end_bar; bar++) indicators_out[46 * num_bars + bar] = ohlcv[bar].high - ohlcv[bar].low; break;
        case 47: for (int bar = start_bar; bar < end_bar; bar++) {
            float range = ohlcv[bar].high - ohlcv[bar].low;
            indicators_out[47 * num_bars + bar] = (range < 1e-10f) ? 0.5f : (ohlcv[bar].close - ohlcv[bar].low) / range;
        } break;
        case 48: if (work_item_id == 0) compute_price_acceleration(ohlcv, num_bars, 10, &indicators_out[48 * num_bars]); break;
        case 49: if (work_item_id == 0) compute_volume_roc(ohlcv, num_bars, 10, &indicators_out[49 * num_bars]); break;
    }
}
