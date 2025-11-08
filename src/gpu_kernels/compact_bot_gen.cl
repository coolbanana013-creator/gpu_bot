/*
 * COMPACT BOT GENERATION KERNEL
 * 
 * Generates compact bot configurations (128 bytes each).
 * Stores only indicator indices (0-49) instead of full parameters.
 * 
 * Memory: 128 bytes per bot (vs 1344 bytes old)
 * 1M bots = 122MB (vs 1.25GB old) - 90.7% savings
 * 
 * UPDATED: Supports leverage 1-125x with proper TP/SL validation
 */

// ============================================================================
// CONSTANTS
// ============================================================================

#define MAX_INDICATORS_PER_BOT 8
#define MAX_PARAMS_PER_INDICATOR 3
#define NUM_TOTAL_INDICATORS 50
#define NUM_RISK_STRATEGIES 15

// Kucoin fees
#define MAKER_FEE 0.0002f      // 0.02%
#define TAKER_FEE 0.0006f      // 0.06%

// ============================================================================
// DATA STRUCTURES
// ============================================================================

// Compact bot configuration (128 bytes total)
typedef struct __attribute__((packed)) {
    int bot_id;                                          // 4 bytes
    unsigned char num_indicators;                        // 1 byte (max 8)
    unsigned char indicator_indices[MAX_INDICATORS_PER_BOT];  // 8 bytes (0-49)
    float indicator_params[MAX_INDICATORS_PER_BOT][MAX_PARAMS_PER_INDICATOR]; // 96 bytes
    unsigned int risk_strategy_bitmap;                   // 4 bytes (15 bits for strategies)
    float tp_multiplier;                                 // 4 bytes
    float sl_multiplier;                                 // 4 bytes
    unsigned char leverage;                              // 1 byte
    unsigned char padding[6];                            // 6 bytes (align to 128)
} CompactBotConfig;

// Parameter ranges for each indicator type
typedef struct {
    float param_mins[MAX_PARAMS_PER_INDICATOR];
    float param_maxs[MAX_PARAMS_PER_INDICATOR];
} IndicatorParamRange;

// ============================================================================
// TP/SL VALIDATION FUNCTIONS
// ============================================================================

/**
 * Validate and fix TP/SL to ensure:
 * 1. TP covers fees + minimum profit
 * 2. SL is reasonable (< TP/2)
 * 3. SL doesn't trigger liquidation
 * 4. Both avoid noise (minimum thresholds)
 */
void validate_and_fix_tp_sl(float *tp_multiplier, float *sl_multiplier, int leverage) {
    // Total fee cost (entry + exit) with leverage
    float total_fee_pct = (MAKER_FEE + TAKER_FEE) * (float)leverage;
    
    // Minimum TP must cover fees + 0.5% profit
    float min_tp = total_fee_pct + 0.005f;
    if (*tp_multiplier < min_tp) {
        *tp_multiplier = min_tp;
    }
    
    // Maximum TP is 25% (reasonable cap)
    if (*tp_multiplier > 0.25f) {
        *tp_multiplier = 0.25f;
    }
    
    // SL must be at most TP/2 (risk/reward ratio)
    float max_sl = *tp_multiplier / 2.0f;
    if (*sl_multiplier > max_sl) {
        *sl_multiplier = max_sl;
    }
    
    // SL must not trigger liquidation
    // Liquidation threshold: ~(1/leverage - 1% buffer)
    float liq_threshold = (1.0f / (float)leverage) - 0.01f;
    if (*sl_multiplier > liq_threshold) {
        *sl_multiplier = liq_threshold * 0.9f;  // 90% of liquidation
    }
    
    // Minimum SL is 0.2% (avoid market noise)
    if (*sl_multiplier < 0.002f) {
        *sl_multiplier = 0.002f;
    }
}

// ============================================================================
// RANDOM NUMBER GENERATOR - XorShift32
// ============================================================================

inline uint xorshift32(uint *state) {
    uint x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

inline float rand_float(uint *state, float min_val, float max_val) {
    uint r = xorshift32(state);
    float normalized = (float)r / (float)0xFFFFFFFF;
    return min_val + normalized * (max_val - min_val);
}

inline int rand_int(uint *state, int min_val, int max_val) {
    uint r = xorshift32(state);
    return min_val + (r % (max_val - min_val));
}

// ============================================================================
// BOT GENERATION KERNEL
// ============================================================================

__kernel void generate_compact_bots(
    __global CompactBotConfig *bots,          // Output: generated bots
    __global IndicatorParamRange *param_ranges,  // Input: parameter ranges for each indicator
    __global uint *seeds,                     // Input: random seeds per bot
    const int min_indicators,                 // Input: min indicators per bot
    const int max_indicators,                 // Input: max indicators per bot
    const int min_risk_strategies,            // Input: min risk strategies per bot
    const int max_risk_strategies,            // Input: max risk strategies per bot
    const int min_leverage,                   // Input: minimum leverage (1-125)
    const int max_leverage                    // Input: maximum leverage (1-125)
) {
    int bot_idx = get_global_id(0);
    
    // Initialize RNG with unique seed
    uint rng_state = seeds[bot_idx];
    
    // Initialize bot
    CompactBotConfig bot;
    bot.bot_id = bot_idx;
    
    // Generate number of indicators (3-8)
    bot.num_indicators = (unsigned char)rand_int(&rng_state, min_indicators, max_indicators + 1);
    if (bot.num_indicators > MAX_INDICATORS_PER_BOT) {
        bot.num_indicators = MAX_INDICATORS_PER_BOT;
    }
    
    // Select unique random indicators (0-49)
    unsigned char selected[MAX_INDICATORS_PER_BOT];
    int selected_count = 0;
    
    for (int i = 0; i < bot.num_indicators; i++) {
        int attempts = 0;
        unsigned char ind_idx;
        int is_unique;
        
        do {
            ind_idx = (unsigned char)rand_int(&rng_state, 0, NUM_TOTAL_INDICATORS);
            
            // Check uniqueness
            is_unique = 1;
            for (int j = 0; j < selected_count; j++) {
                if (selected[j] == ind_idx) {
                    is_unique = 0;
                    break;
                }
            }
            
            attempts++;
            if (attempts > 100) {
                // Fallback: use sequential
                ind_idx = (unsigned char)((bot_idx * 7 + i * 13) % NUM_TOTAL_INDICATORS);
                break;
            }
        } while (!is_unique);
        
        selected[selected_count++] = ind_idx;
        bot.indicator_indices[i] = ind_idx;
        
        // Generate parameters for this indicator
        for (int p = 0; p < MAX_PARAMS_PER_INDICATOR; p++) {
            bot.indicator_params[i][p] = rand_float(
                &rng_state,
                param_ranges[ind_idx].param_mins[p],
                param_ranges[ind_idx].param_maxs[p]
            );
        }
    }
    
    // Fill unused indicator slots with zeros
    for (int i = bot.num_indicators; i < MAX_INDICATORS_PER_BOT; i++) {
        bot.indicator_indices[i] = 0;
        for (int p = 0; p < MAX_PARAMS_PER_INDICATOR; p++) {
            bot.indicator_params[i][p] = 0.0f;
        }
    }
    
    // Generate risk strategy bitmap (15 strategies)
    int num_risk_strategies = rand_int(&rng_state, min_risk_strategies, max_risk_strategies + 1);
    bot.risk_strategy_bitmap = 0;
    
    for (int i = 0; i < num_risk_strategies; i++) {
        int strategy_id = rand_int(&rng_state, 0, NUM_RISK_STRATEGIES);
        bot.risk_strategy_bitmap |= (1 << strategy_id);  // Set bit
    }
    
    // Ensure at least one strategy is active
    if (bot.risk_strategy_bitmap == 0) {
        bot.risk_strategy_bitmap = 1;  // Default to Fixed% (bit 0)
    }
    
    // Generate leverage (min_leverage to max_leverage, e.g., 1-125)
    bot.leverage = (unsigned char)rand_int(&rng_state, min_leverage, max_leverage + 1);
    if (bot.leverage < 1) bot.leverage = 1;
    if (bot.leverage > 125) bot.leverage = 125;
    
    // Generate TP/SL multipliers (percentage of price)
    // Initial ranges: TP 0.5-25%, SL 0.2-10%
    bot.tp_multiplier = rand_float(&rng_state, 0.005f, 0.25f);  // 0.5% - 25%
    bot.sl_multiplier = rand_float(&rng_state, 0.002f, 0.10f);  // 0.2% - 10%
    
    // VALIDATE AND FIX TP/SL based on leverage and fees
    validate_and_fix_tp_sl(&bot.tp_multiplier, &bot.sl_multiplier, bot.leverage);
    
    // Zero padding
    for (int i = 0; i < 6; i++) {
        bot.padding[i] = 0;
    }
    
    // Write bot to global memory
    bots[bot_idx] = bot;
}
