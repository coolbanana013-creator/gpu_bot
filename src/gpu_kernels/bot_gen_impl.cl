/*
 * Bot Generation OpenCL Kernel - FULL IMPLEMENTATION
 * 
 * Generates trading bot configurations in parallel on GPU.
 * Each work-item creates one complete bot with unique indicator combinations.
 * 
 * CRITICAL: No CPU fallbacks. This kernel MUST execute successfully or application crashes.
 */

// Configuration constants
#define MAX_INDICATORS 20
#define MAX_RISK_STRATEGIES 10
#define MAX_PARAMS 10
#define NUM_INDICATOR_TYPES 30
#define NUM_RISK_STRATEGY_TYPES 12

// Bot configuration structure
typedef struct {
    int bot_id;
    int num_indicators;
    int indicator_types[MAX_INDICATORS];
    float indicator_params[MAX_INDICATORS * MAX_PARAMS];  // Flattened array
    int num_risk_strategies;
    int risk_strategy_types[MAX_RISK_STRATEGIES];
    float risk_strategy_params[MAX_RISK_STRATEGIES * MAX_PARAMS];  // Flattened
    float take_profit_pct;
    float stop_loss_pct;
    int leverage;
} BotConfig;

// Parameter range structure for indicators
typedef struct {
    int indicator_type;
    int num_params;
    float param_mins[MAX_PARAMS];
    float param_maxs[MAX_PARAMS];
} IndicatorParamRange;

// Parameter range structure for risk strategies
typedef struct {
    int strategy_type;
    int num_params;
    float param_mins[MAX_PARAMS];
    float param_maxs[MAX_PARAMS];
} RiskStrategyParamRange;

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

// Generate random float in range [min, max)
inline float rand_float(uint *state, float min_val, float max_val) {
    uint r = xorshift32(state);
    float normalized = (float)r / (float)0xFFFFFFFF;  // [0, 1]
    return min_val + normalized * (max_val - min_val);
}

// Generate random int in range [min, max)
inline int rand_int(uint *state, int min_val, int max_val) {
    uint r = xorshift32(state);
    return min_val + (r % (max_val - min_val));
}

// ============================================================================
// INDICATOR PARAMETER GENERATION
// ============================================================================

void generate_indicator_params(
    uint *rng_state,
    int indicator_type,
    __constant IndicatorParamRange *param_ranges,
    int num_param_ranges,
    float *out_params
) {
    // Find parameter range for this indicator type
    for (int i = 0; i < num_param_ranges; i++) {
        if (param_ranges[i].indicator_type == indicator_type) {
            // Generate random parameters within ranges
            for (int p = 0; p < param_ranges[i].num_params; p++) {
                out_params[p] = rand_float(
                    rng_state,
                    param_ranges[i].param_mins[p],
                    param_ranges[i].param_maxs[p]
                );
            }
            return;
        }
    }
    
    // Default: no parameters
    for (int p = 0; p < MAX_PARAMS; p++) {
        out_params[p] = 0.0f;
    }
}

// ============================================================================
// RISK STRATEGY PARAMETER GENERATION
// ============================================================================

void generate_risk_strategy_params(
    uint *rng_state,
    int strategy_type,
    __constant RiskStrategyParamRange *param_ranges,
    int num_param_ranges,
    float *out_params
) {
    // Find parameter range for this strategy type
    for (int i = 0; i < num_param_ranges; i++) {
        if (param_ranges[i].strategy_type == strategy_type) {
            // Generate random parameters within ranges
            for (int p = 0; p < param_ranges[i].num_params; p++) {
                out_params[p] = rand_float(
                    rng_state,
                    param_ranges[i].param_mins[p],
                    param_ranges[i].param_maxs[p]
                );
            }
            return;
        }
    }
    
    // Default: no parameters
    for (int p = 0; p < MAX_PARAMS; p++) {
        out_params[p] = 0.0f;
    }
}

// ============================================================================
// UNIQUE SELECTION WITHIN BOT
// ============================================================================

// Select unique indicator type (not already in bot's selections)
int select_unique_indicator(
    uint *rng_state,
    __constant int *available_types,
    int num_available,
    int *already_selected,
    int num_selected
) {
    int attempts = 0;
    while (attempts < 1000) {  // Prevent infinite loop
        int idx = rand_int(rng_state, 0, num_available);
        int candidate = available_types[idx];
        
        // Check if already selected
        int is_duplicate = 0;
        for (int i = 0; i < num_selected; i++) {
            if (already_selected[i] == candidate) {
                is_duplicate = 1;
                break;
            }
        }
        
        if (!is_duplicate) {
            return candidate;
        }
        
        attempts++;
    }
    
    // Fallback: return first available not selected
    for (int i = 0; i < num_available; i++) {
        int candidate = available_types[i];
        int is_duplicate = 0;
        for (int j = 0; j < num_selected; j++) {
            if (already_selected[j] == candidate) {
                is_duplicate = 1;
                break;
            }
        }
        if (!is_duplicate) {
            return candidate;
        }
    }
    
    return available_types[0];  // Last resort
}

// ============================================================================
// MAIN BOT GENERATION KERNEL
// ============================================================================

__kernel void generate_bots(
    __global BotConfig *bot_configs,                          // Output: bot configurations
    __constant int *indicator_types,                          // Available indicator type IDs
    const int num_indicator_types,                            // Number of available indicator types
    __constant int *risk_strategy_types,                      // Available risk strategy type IDs
    const int num_risk_strategy_types,                        // Number of available risk strategies
    __constant IndicatorParamRange *indicator_param_ranges,   // Parameter ranges for indicators
    const int num_indicator_param_ranges,                     // Number of indicator param ranges
    __constant RiskStrategyParamRange *risk_param_ranges,     // Parameter ranges for risk strategies
    const int num_risk_param_ranges,                          // Number of risk param ranges
    __global uint *random_seeds,                              // Random seeds (one per bot)
    const int population_size,                                // Total bots to generate
    const int min_indicators,                                 // Min indicators per bot
    const int max_indicators,                                 // Max indicators per bot
    const int min_risk_strategies,                            // Min risk strategies per bot
    const int max_risk_strategies,                            // Max risk strategies per bot
    const int leverage,                                       // Leverage multiplier
    const float tp_min,                                       // Min take profit %
    const float tp_max,                                       // Max take profit %
    const float sl_min,                                       // Min stop loss %
    const float sl_max                                        // Max stop loss %
) {
    int bot_id = get_global_id(0);
    
    if (bot_id >= population_size) {
        return;
    }
    
    // Initialize RNG state
    uint rng_state = random_seeds[bot_id];
    
    // Initialize bot struct
    BotConfig bot;
    bot.bot_id = bot_id;
    bot.leverage = leverage;
    
    // ========================================================================
    // GENERATE INDICATORS
    // ========================================================================
    
    // Random number of indicators
    bot.num_indicators = rand_int(&rng_state, min_indicators, max_indicators + 1);
    
    // Local array to track selected indicators
    int selected_indicators[MAX_INDICATORS];
    
    for (int i = 0; i < bot.num_indicators; i++) {
        // Select unique indicator type
        int ind_type = select_unique_indicator(
            &rng_state,
            indicator_types,
            num_indicator_types,
            selected_indicators,
            i  // number already selected
        );
        
        bot.indicator_types[i] = ind_type;
        selected_indicators[i] = ind_type;
        
        // Generate parameters for this indicator
        float temp_params[MAX_PARAMS];
        generate_indicator_params(
            &rng_state,
            ind_type,
            indicator_param_ranges,
            num_indicator_param_ranges,
            temp_params
        );
        
        // Copy to flattened array
        for (int p = 0; p < MAX_PARAMS; p++) {
            bot.indicator_params[i * MAX_PARAMS + p] = temp_params[p];
        }
    }
    
    // Initialize unused indicator slots
    for (int i = bot.num_indicators; i < MAX_INDICATORS; i++) {
        bot.indicator_types[i] = -1;
        for (int p = 0; p < MAX_PARAMS; p++) {
            bot.indicator_params[i * MAX_PARAMS + p] = 0.0f;
        }
    }
    
    // ========================================================================
    // GENERATE RISK STRATEGIES
    // ========================================================================
    
    // Random number of risk strategies
    bot.num_risk_strategies = rand_int(&rng_state, min_risk_strategies, max_risk_strategies + 1);
    
    // Local array to track selected strategies
    int selected_strategies[MAX_RISK_STRATEGIES];
    
    for (int i = 0; i < bot.num_risk_strategies; i++) {
        // Select unique strategy type (reuse indicator selection logic)
        int strat_type = select_unique_indicator(  // Same logic works for strategies
            &rng_state,
            risk_strategy_types,
            num_risk_strategy_types,
            selected_strategies,
            i  // number already selected
        );
        
        bot.risk_strategy_types[i] = strat_type;
        selected_strategies[i] = strat_type;
        
        // Generate parameters for this strategy
        float temp_params[MAX_PARAMS];
        generate_risk_strategy_params(
            &rng_state,
            strat_type,
            risk_param_ranges,
            num_risk_param_ranges,
            temp_params
        );
        
        // Copy to flattened array
        for (int p = 0; p < MAX_PARAMS; p++) {
            bot.risk_strategy_params[i * MAX_PARAMS + p] = temp_params[p];
        }
    }
    
    // Initialize unused strategy slots
    for (int i = bot.num_risk_strategies; i < MAX_RISK_STRATEGIES; i++) {
        bot.risk_strategy_types[i] = -1;
        for (int p = 0; p < MAX_PARAMS; p++) {
            bot.risk_strategy_params[i * MAX_PARAMS + p] = 0.0f;
        }
    }
    
    // ========================================================================
    // GENERATE TP/SL WITH LEVERAGE ADJUSTMENT
    // ========================================================================
    
    // Adjust ranges based on leverage (higher leverage = tighter TP/SL)
    float lev_factor = 1.0f / sqrt((float)leverage);
    float adjusted_tp_min = tp_min * lev_factor;
    float adjusted_tp_max = tp_max * lev_factor;
    float adjusted_sl_min = sl_min * lev_factor;
    float adjusted_sl_max = sl_max * lev_factor;
    
    bot.take_profit_pct = rand_float(&rng_state, adjusted_tp_min, adjusted_tp_max);
    bot.stop_loss_pct = rand_float(&rng_state, adjusted_sl_min, adjusted_sl_max);
    
    // Ensure TP > SL for sanity
    if (bot.take_profit_pct <= bot.stop_loss_pct) {
        bot.take_profit_pct = bot.stop_loss_pct * 1.5f;
    }
    
    // ========================================================================
    // WRITE TO GLOBAL MEMORY
    // ========================================================================
    
    bot_configs[bot_id] = bot;
}

// ============================================================================
// HELPER KERNEL: Initialize Random Seeds
// ============================================================================

__kernel void initialize_random_seeds(
    __global uint *random_seeds,
    const int population_size,
    const uint base_seed
) {
    int id = get_global_id(0);
    
    if (id >= population_size) {
        return;
    }
    
    // Generate unique seed per bot using hash function
    uint seed = base_seed;
    seed ^= (uint)id * 0x9E3779B9u;  // Golden ratio
    seed ^= seed << 13;
    seed ^= seed >> 7;
    seed ^= seed << 17;
    
    random_seeds[id] = seed;
}
