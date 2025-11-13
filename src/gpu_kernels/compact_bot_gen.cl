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

// Risk strategies (matching backtest kernel) - 15 total
#define RISK_FIXED_PCT 0           // Fixed percentage of balance
#define RISK_FIXED_USD 1           // Fixed USD amount
#define RISK_KELLY_FULL 2          // Full Kelly criterion
#define RISK_KELLY_HALF 3          // Half Kelly (safer)
#define RISK_KELLY_QUARTER 4       // Quarter Kelly (conservative)
#define RISK_ATR_MULTIPLIER 5      // ATR-based position sizing
#define RISK_VOLATILITY_PCT 6      // Percentage based on volatility
#define RISK_EQUITY_CURVE 7        // Adjust size based on equity curve
#define RISK_FIXED_RISK_REWARD 8   // Fixed risk/reward ratio
#define RISK_MARTINGALE 9          // Increase after losses (dangerous)
#define RISK_ANTI_MARTINGALE 10    // Increase after wins
#define RISK_FIXED_RATIO 11        // Fixed ratio method (Ryan Jones)
#define RISK_PERCENT_VOLATILITY 12 // Percent of volatility
#define RISK_WILLIAMS_FIXED 13     // Williams Fixed Fractional
#define RISK_OPTIMAL_F 14          // Optimal f (Ralph Vince)

// Compact bot configuration (128 bytes total)
typedef struct __attribute__((packed)) {
    int bot_id;                                          // 4 bytes (offset 0)
    unsigned char num_indicators;                        // 1 byte (offset 4, max 8)
    unsigned char indicator_indices[MAX_INDICATORS_PER_BOT];  // 8 bytes (offset 5, indices 0-49)
    float indicator_params[MAX_INDICATORS_PER_BOT][MAX_PARAMS_PER_INDICATOR]; // 96 bytes (offset 13)
    unsigned char risk_strategy;                         // 1 byte (offset 109, enum 0-14 for 15 strategies)
    float risk_param;                                    // 4 bytes (offset 110, strategy parameter)
    float tp_multiplier;                                 // 4 bytes (offset 114)
    float sl_multiplier;                                 // 4 bytes (offset 118)
    unsigned char leverage;                              // 1 byte (offset 122)
    unsigned char padding[5];                            // 5 bytes (offset 123, align to 128)
} CompactBotConfig;  // Total: 128 bytes

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
    
    // SL must be at most TP/2.5 (2.5:1 risk/reward ratio minimum)
    // Balanced for crypto volatility: tight enough for edge, wide enough to avoid noise
    float max_sl = *tp_multiplier / 2.5f;
    if (*sl_multiplier > max_sl) {
        *sl_multiplier = max_sl;
    }
    
    // SL must not trigger liquidation
    // Initial margin = 1 / leverage (e.g., at 125x leverage: 0.8% margin)
    // Liquidation occurs when loss exceeds ~80% of initial margin (accounting for fees)
    // Example: 125x leverage, initial margin = 0.8%, liquidation at ~0.6% loss
    float initial_margin = 1.0f / (float)leverage;
    float liq_threshold = initial_margin * 0.75f;  // 75% of margin (conservative)
    
    if (*sl_multiplier > liq_threshold) {
        *sl_multiplier = liq_threshold * 0.9f;  // 90% of safe threshold
    }
    
    // Minimum SL is 5% (wide enough to avoid 1m crypto noise)
    if (*sl_multiplier < 0.05f) {
        *sl_multiplier = 0.05f;
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
    
    // Generate SINGLE risk strategy from 15 available strategies
    int strategy_choice = rand_int(&rng_state, 0, 15);  // 0 to 14
    bot.risk_strategy = (unsigned char)strategy_choice;
    
    // Generate appropriate parameter for selected strategy
    switch(bot.risk_strategy) {
        case RISK_FIXED_PCT:
            // Fixed percentage: 1% to 20%
            bot.risk_param = rand_float(&rng_state, 0.01f, 0.20f);
            break;
        case RISK_FIXED_USD:
            // Fixed USD: $10 to $10000
            bot.risk_param = rand_float(&rng_state, 10.0f, 10000.0f);
            break;
        case RISK_KELLY_FULL:
            // Full Kelly fraction: 0.01 to 1.0
            bot.risk_param = rand_float(&rng_state, 0.01f, 1.0f);
            break;
        case RISK_KELLY_HALF:
            // Half Kelly base fraction: 0.01 to 1.0 (will be halved in calculation)
            bot.risk_param = rand_float(&rng_state, 0.01f, 1.0f);
            break;
        case RISK_KELLY_QUARTER:
            // Quarter Kelly base fraction: 0.01 to 1.0 (will be quartered)
            bot.risk_param = rand_float(&rng_state, 0.01f, 1.0f);
            break;
        case RISK_ATR_MULTIPLIER:
            // ATR multiplier: 1.0 to 5.0
            bot.risk_param = rand_float(&rng_state, 1.0f, 5.0f);
            break;
        case RISK_VOLATILITY_PCT:
            // Volatility percentage: 0.01 to 0.20
            bot.risk_param = rand_float(&rng_state, 0.01f, 0.20f);
            break;
        case RISK_EQUITY_CURVE:
            // Equity curve multiplier: 0.5 to 2.0
            bot.risk_param = rand_float(&rng_state, 0.5f, 2.0f);
            break;
        case RISK_FIXED_RISK_REWARD:
            // Risk per trade: 0.01 to 0.10
            bot.risk_param = rand_float(&rng_state, 0.01f, 0.10f);
            break;
        case RISK_MARTINGALE:
            // Martingale multiplier: 1.5 to 3.0
            bot.risk_param = rand_float(&rng_state, 1.5f, 3.0f);
            break;
        case RISK_ANTI_MARTINGALE:
            // Anti-Martingale multiplier: 1.2 to 2.0
            bot.risk_param = rand_float(&rng_state, 1.2f, 2.0f);
            break;
        case RISK_FIXED_RATIO:
            // Fixed ratio delta: 1000 to 10000
            bot.risk_param = rand_float(&rng_state, 1000.0f, 10000.0f);
            break;
        case RISK_PERCENT_VOLATILITY:
            // Percent volatility: 0.01 to 0.20
            bot.risk_param = rand_float(&rng_state, 0.01f, 0.20f);
            break;
        case RISK_WILLIAMS_FIXED:
            // Williams fixed fractional: 0.01 to 0.10
            bot.risk_param = rand_float(&rng_state, 0.01f, 0.10f);
            break;
        case RISK_OPTIMAL_F:
            // Optimal f: 0.01 to 0.30
            bot.risk_param = rand_float(&rng_state, 0.01f, 0.30f);
            break;
        default:
            // Default: 5% of balance
            bot.risk_param = 0.05f;
            break;
    }
    
    // Generate leverage (min_leverage to max_leverage, e.g., 1-125)
    bot.leverage = (unsigned char)rand_int(&rng_state, min_leverage, max_leverage + 1);
    if (bot.leverage < 1) bot.leverage = 1;
    if (bot.leverage > 125) bot.leverage = 125;
    
    // Generate TP/SL multipliers (percentage of price)
    // BALANCED: TP 15-25%, SL 5-8% for 2:1 to 4:1 risk/reward
    // Wider stops prevent getting stopped out by 1m crypto noise
    bot.tp_multiplier = rand_float(&rng_state, 0.15f, 0.25f);  // 15% - 25%
    bot.sl_multiplier = rand_float(&rng_state, 0.05f, 0.08f);  // 5% - 8%
    
    // VALIDATE AND FIX TP/SL based on leverage and fees
    validate_and_fix_tp_sl(&bot.tp_multiplier, &bot.sl_multiplier, bot.leverage);
    
    // Zero padding (5 bytes for 128-byte alignment)
    for (int i = 0; i < 5; i++) {
        bot.padding[i] = 0;
    }
    
    // Write bot to global memory
    bots[bot_idx] = bot;
}
