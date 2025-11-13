/**
 * Signal Generation Debug Kernel
 * Tracks detailed signal statistics for analysis
 */

typedef struct {
    int bot_id;
    int cycle;
    int bar;
    int num_indicators;
    int valid_indicators;
    int bullish_signals;
    int bearish_signals;
    int neutral_signals;
    int directional_signals;
    float final_signal;
    float indicator_values[8];  // Store up to 8 indicator values
    int indicator_signals[8];   // Store individual signals
} SignalDebugRecord;

/**
 * Sample signal generation for first N bots at regular intervals
 * This helps understand why signals are/aren't being generated
 */
__kernel void debug_signal_generation(
    __global float *precomputed_indicators,
    __global uchar *compact_bots_data,
    __global int *cycle_ranges,
    __global SignalDebugRecord *debug_records,
    int num_bars,
    int num_bots_to_sample,
    int bars_per_sample,
    int cycle_to_debug
) {
    int bot_id = get_global_id(0);
    
    // Only sample first N bots
    if (bot_id >= num_bots_to_sample) return;
    
    // Load compact bot
    __global uchar *bot_ptr = compact_bots_data + (bot_id * 128);
    
    // Parse compact bot (first 32 bytes)
    int num_indicators = bot_ptr[0];
    int num_risk_strategies = bot_ptr[1];
    uchar indicators[8];
    float indicator_params[8][3];
    
    // Load indicators
    for (int i = 0; i < num_indicators && i < 8; i++) {
        indicators[i] = bot_ptr[2 + i];
    }
    
    // Load indicator params (skip for now, focus on signal counts)
    
    // Get cycle range
    int cycle_start = cycle_ranges[cycle_to_debug * 2];
    int cycle_end = cycle_ranges[cycle_to_debug * 2 + 1];
    int cycle_bars = cycle_end - cycle_start;
    
    // Sample every N bars
    int num_samples = cycle_bars / bars_per_sample;
    
    for (int sample = 0; sample < num_samples && sample < 100; sample++) {
        int bar = cycle_start + (sample * bars_per_sample);
        
        // Generate signal for this bar
        int bullish_count = 0;
        int bearish_count = 0;
        int valid_count = 0;
        
        float ind_vals[8] = {0};
        int ind_sigs[8] = {0};
        
        for (int i = 0; i < num_indicators && i < 8; i++) {
            int ind_idx = indicators[i];
            float ind_value = precomputed_indicators[ind_idx * num_bars + bar];
            
            // Skip invalid
            if (isnan(ind_value) || isinf(ind_value)) {
                ind_vals[i] = NAN;
                ind_sigs[i] = 0;
                continue;
            }
            
            valid_count++;
            ind_vals[i] = ind_value;
            
            // Simplified signal logic (just for debugging)
            int signal = 0;
            
            // Moving averages (0-11)
            if (ind_idx >= 0 && ind_idx <= 11) {
                if (bar > cycle_start) {
                    float prev = precomputed_indicators[ind_idx * num_bars + (bar-1)];
                    if (ind_value > prev * 1.001f) signal = 1;
                    else if (ind_value < prev * 0.999f) signal = -1;
                }
            }
            // RSI (12-14)
            else if (ind_idx >= 12 && ind_idx <= 14) {
                if (ind_value < 30.0f) signal = 1;
                else if (ind_value > 70.0f) signal = -1;
            }
            // Momentum (17)
            else if (ind_idx == 17) {
                if (ind_value > 0.0f) signal = 1;
                else if (ind_value < 0.0f) signal = -1;
            }
            // Add more as needed...
            
            ind_sigs[i] = signal;
            if (signal == 1) bullish_count++;
            else if (signal == -1) bearish_count++;
        }
        
        int neutral_count = valid_count - bullish_count - bearish_count;
        int directional_signals = bullish_count + bearish_count;
        
        float final_signal = 0.0f;
        if (directional_signals > 0) {
            float bullish_pct = (float)bullish_count / (float)directional_signals;
            if (bullish_pct >= 1.0f) final_signal = 1.0f;
            else if (bullish_pct <= 0.0f) final_signal = -1.0f;
        }
        
        // Store debug record
        int record_idx = bot_id * num_samples + sample;
        debug_records[record_idx].bot_id = bot_id;
        debug_records[record_idx].cycle = cycle_to_debug;
        debug_records[record_idx].bar = bar;
        debug_records[record_idx].num_indicators = num_indicators;
        debug_records[record_idx].valid_indicators = valid_count;
        debug_records[record_idx].bullish_signals = bullish_count;
        debug_records[record_idx].bearish_signals = bearish_count;
        debug_records[record_idx].neutral_signals = neutral_count;
        debug_records[record_idx].directional_signals = directional_signals;
        debug_records[record_idx].final_signal = final_signal;
        
        // Copy indicator values and signals
        for (int i = 0; i < 8; i++) {
            debug_records[record_idx].indicator_values[i] = ind_vals[i];
            debug_records[record_idx].indicator_signals[i] = ind_sigs[i];
        }
    }
}
