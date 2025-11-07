/*
 * Indicator Precomputation OpenCL Kernel
 * 
 * Precomputes all indicator values for entire dataset before backtesting.
 * This avoids redundant computation across thousands of bots.
 * 
 * CRITICAL: No CPU fallbacks. Must execute on GPU.
 */

// Configuration constants
#define MAX_PARAMS 10
#define MAX_LOOKBACK 200  // Maximum bars needed for any indicator

// OHLCV bar structure
typedef struct {
    float open;
    float high;
    float low;
    float close;
    float volume;
} OHLCVBar;

// ============================================================================
// TECHNICAL INDICATOR IMPLEMENTATIONS
// ============================================================================

// Simple Moving Average
float compute_sma(
    __global OHLCVBar *ohlcv,
    int current_bar,
    int period
) {
    if (current_bar < period - 1) {
        return ohlcv[current_bar].close;  // Not enough data
    }
    
    float sum = 0.0f;
    for (int i = 0; i < period; i++) {
        sum += ohlcv[current_bar - i].close;
    }
    return sum / (float)period;
}

// Exponential Moving Average (simplified, single-pass)
float compute_ema(
    __global OHLCVBar *ohlcv,
    int current_bar,
    int period
) {
    if (current_bar < period - 1) {
        return ohlcv[current_bar].close;
    }
    
    float multiplier = 2.0f / (float)(period + 1);
    
    // Start with SMA
    float ema = compute_sma(ohlcv, current_bar - (period - 1) + period - 1, period);
    
    // Calculate EMA from SMA point forward
    for (int i = current_bar - period + 1; i <= current_bar; i++) {
        ema = (ohlcv[i].close - ema) * multiplier + ema;
    }
    
    return ema;
}

// Relative Strength Index
float compute_rsi(
    __global OHLCVBar *ohlcv,
    int current_bar,
    int period
) {
    if (current_bar < period) {
        return 50.0f;  // Neutral
    }
    
    float gains = 0.0f;
    float losses = 0.0f;
    
    for (int i = 1; i <= period; i++) {
        float change = ohlcv[current_bar - i + 1].close - ohlcv[current_bar - i].close;
        if (change > 0) {
            gains += change;
        } else {
            losses += fabs(change);
        }
    }
    
    float avg_gain = gains / (float)period;
    float avg_loss = losses / (float)period;
    
    if (avg_loss == 0.0f) {
        return 100.0f;
    }
    
    float rs = avg_gain / avg_loss;
    float rsi = 100.0f - (100.0f / (1.0f + rs));
    
    return rsi;
}

// Average True Range
float compute_atr(
    __global OHLCVBar *ohlcv,
    int current_bar,
    int period
) {
    if (current_bar < period) {
        return ohlcv[current_bar].high - ohlcv[current_bar].low;
    }
    
    float sum_tr = 0.0f;
    
    for (int i = 0; i < period; i++) {
        int idx = current_bar - i;
        float high = ohlcv[idx].high;
        float low = ohlcv[idx].low;
        float prev_close = (idx > 0) ? ohlcv[idx - 1].close : ohlcv[idx].close;
        
        float tr = fmax(high - low, fmax(fabs(high - prev_close), fabs(low - prev_close)));
        sum_tr += tr;
    }
    
    return sum_tr / (float)period;
}

// Bollinger Bands (returns middle, upper, lower in output array)
void compute_bollinger(
    __global OHLCVBar *ohlcv,
    int current_bar,
    int period,
    float num_std,
    float *output  // [0]=middle, [1]=upper, [2]=lower
) {
    float sma = compute_sma(ohlcv, current_bar, period);
    
    if (current_bar < period - 1) {
        output[0] = sma;
        output[1] = sma;
        output[2] = sma;
        return;
    }
    
    // Calculate standard deviation
    float sum_sq_diff = 0.0f;
    for (int i = 0; i < period; i++) {
        float diff = ohlcv[current_bar - i].close - sma;
        sum_sq_diff += diff * diff;
    }
    float std_dev = sqrt(sum_sq_diff / (float)period);
    
    output[0] = sma;
    output[1] = sma + (num_std * std_dev);
    output[2] = sma - (num_std * std_dev);
}

// MACD (returns macd, signal, histogram in output array)
void compute_macd(
    __global OHLCVBar *ohlcv,
    int current_bar,
    int fast_period,
    int slow_period,
    int signal_period,
    float *output  // [0]=macd, [1]=signal, [2]=histogram
) {
    float fast_ema = compute_ema(ohlcv, current_bar, fast_period);
    float slow_ema = compute_ema(ohlcv, current_bar, slow_period);
    
    float macd = fast_ema - slow_ema;
    
    // Simplified signal line (should be EMA of MACD, but approximating)
    float signal = macd * 0.9f;  // Placeholder
    
    output[0] = macd;
    output[1] = signal;
    output[2] = macd - signal;
}

// Stochastic Oscillator
void compute_stochastic(
    __global OHLCVBar *ohlcv,
    int current_bar,
    int k_period,
    int d_period,
    float *output  // [0]=%K, [1]=%D
) {
    if (current_bar < k_period - 1) {
        output[0] = 50.0f;
        output[1] = 50.0f;
        return;
    }
    
    // Find highest high and lowest low in period
    float highest = ohlcv[current_bar].high;
    float lowest = ohlcv[current_bar].low;
    
    for (int i = 1; i < k_period; i++) {
        float high = ohlcv[current_bar - i].high;
        float low = ohlcv[current_bar - i].low;
        if (high > highest) highest = high;
        if (low < lowest) lowest = low;
    }
    
    float close = ohlcv[current_bar].close;
    float k = 0.0f;
    
    if (highest != lowest) {
        k = 100.0f * (close - lowest) / (highest - lowest);
    } else {
        k = 50.0f;
    }
    
    // %D is SMA of %K (simplified)
    float d = k;
    
    output[0] = k;
    output[1] = d;
}

// CCI - Commodity Channel Index
float compute_cci(
    __global OHLCVBar *ohlcv,
    int current_bar,
    int period
) {
    if (current_bar < period - 1) {
        return 0.0f;
    }
    
    // Calculate typical price SMA
    float sum_tp = 0.0f;
    for (int i = 0; i < period; i++) {
        int idx = current_bar - i;
        float tp = (ohlcv[idx].high + ohlcv[idx].low + ohlcv[idx].close) / 3.0f;
        sum_tp += tp;
    }
    float sma_tp = sum_tp / (float)period;
    
    // Calculate mean deviation
    float sum_dev = 0.0f;
    for (int i = 0; i < period; i++) {
        int idx = current_bar - i;
        float tp = (ohlcv[idx].high + ohlcv[idx].low + ohlcv[idx].close) / 3.0f;
        sum_dev += fabs(tp - sma_tp);
    }
    float mean_dev = sum_dev / (float)period;
    
    float current_tp = (ohlcv[current_bar].high + ohlcv[current_bar].low + ohlcv[current_bar].close) / 3.0f;
    
    if (mean_dev == 0.0f) {
        return 0.0f;
    }
    
    float cci = (current_tp - sma_tp) / (0.015f * mean_dev);
    
    return cci;
}

// Add more indicators: ADX, Aroon, Williams %R, etc.
// (Simplified versions for performance)

// ============================================================================
// MAIN PRECOMPUTE KERNEL
// ============================================================================

__kernel void precompute_indicators(
    __global OHLCVBar *ohlcv_data,           // Input: OHLCV bars [total_bars]
    __global float *indicator_values,         // Output: [total_bars, num_types, MAX_PARAMS]
    __constant float *indicator_params,       // Parameters for each indicator type
    const int total_bars,
    const int num_indicator_types
) {
    int bar = get_global_id(0);
    
    if (bar >= total_bars) {
        return;
    }
    
    // For each indicator type, compute and store values
    for (int ind_type = 0; ind_type < num_indicator_types; ind_type++) {
        int output_idx = (bar * num_indicator_types + ind_type) * MAX_PARAMS;
        
        // Get parameters for this indicator type
        int param_idx = ind_type * MAX_PARAMS;
        int period = (int)indicator_params[param_idx];
        float param1 = indicator_params[param_idx + 1];
        float param2 = indicator_params[param_idx + 2];
        
        // Compute based on indicator type
        switch (ind_type) {
            case 0:  // RSI
                indicator_values[output_idx] = compute_rsi(ohlcv_data, bar, period);
                indicator_values[output_idx + 1] = param1;  // Overbought
                indicator_values[output_idx + 2] = param2;  // Oversold
                break;
            
            case 1:  // MACD
                {
                    float macd_output[3];
                    compute_macd(ohlcv_data, bar, period, (int)param1, (int)param2, macd_output);
                    indicator_values[output_idx] = macd_output[0];
                    indicator_values[output_idx + 1] = macd_output[1];
                    indicator_values[output_idx + 2] = macd_output[2];
                }
                break;
            
            case 2:  // Stochastic
                {
                    float stoch_output[2];
                    compute_stochastic(ohlcv_data, bar, period, (int)param1, stoch_output);
                    indicator_values[output_idx] = stoch_output[0];
                    indicator_values[output_idx + 1] = stoch_output[1];
                }
                break;
            
            case 3:  // Bollinger Bands
                {
                    float bb_output[3];
                    compute_bollinger(ohlcv_data, bar, period, param1, bb_output);
                    indicator_values[output_idx] = bb_output[0];
                    indicator_values[output_idx + 1] = bb_output[1];
                    indicator_values[output_idx + 2] = bb_output[2];
                }
                break;
            
            case 4:  // EMA
                indicator_values[output_idx] = compute_ema(ohlcv_data, bar, period);
                break;
            
            case 5:  // SMA
                indicator_values[output_idx] = compute_sma(ohlcv_data, bar, period);
                break;
            
            case 6:  // ATR
                indicator_values[output_idx] = compute_atr(ohlcv_data, bar, period);
                break;
            
            case 7:  // CCI
                indicator_values[output_idx] = compute_cci(ohlcv_data, bar, period);
                break;
            
            // Add more cases for 50+ indicators
            // Cases 8-50: Additional indicators (HT_TRENDLINE, KAMA, etc.)
            
            default:
                // Unknown indicator type - store zeros
                for (int p = 0; p < MAX_PARAMS; p++) {
                    indicator_values[output_idx + p] = 0.0f;
                }
                break;
        }
    }
}

// ============================================================================
// HELPER KERNEL: Validate Precomputed Data
// ============================================================================

__kernel void validate_indicators(
    __global float *indicator_values,
    __global int *error_flags,
    const int total_bars,
    const int num_indicator_types
) {
    int bar = get_global_id(0);
    
    if (bar >= total_bars) {
        return;
    }
    
    // Check for NaN or Inf values
    for (int ind_type = 0; ind_type < num_indicator_types; ind_type++) {
        int output_idx = (bar * num_indicator_types + ind_type) * MAX_PARAMS;
        
        for (int p = 0; p < MAX_PARAMS; p++) {
            float val = indicator_values[output_idx + p];
            if (isnan(val) || isinf(val)) {
                atomic_inc(&error_flags[0]);  // Increment error counter
            }
        }
    }
}
