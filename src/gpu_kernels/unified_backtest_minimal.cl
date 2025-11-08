/*
 * MINIMAL TEST KERNEL - Validate Infrastructure
 * Stripped down version to test memory transfer and basic execution
 */

#define MAX_INDICATORS_PER_BOT 8
#define MAX_PARAMS_PER_INDICATOR 3

// Compact bot configuration (128 bytes)
typedef struct __attribute__((packed)) {
    int bot_id;
    unsigned char num_indicators;
    unsigned char indicator_indices[MAX_INDICATORS_PER_BOT];
    float indicator_params[MAX_INDICATORS_PER_BOT][MAX_PARAMS_PER_INDICATOR];
    unsigned int risk_strategy_bitmap;
    float tp_multiplier;
    float sl_multiplier;
    unsigned char leverage;
    unsigned char padding[6];
} CompactBotConfig;

// OHLCV bar
typedef struct {
    float open;
    float high;
    float low;
    float close;
    float volume;
} OHLCVBar;

// Backtest result
typedef struct {
    int bot_id;
    int total_trades;
    int winning_trades;
    int losing_trades;
    float total_return_pct;
    float sharpe_ratio;
    float max_drawdown_pct;
    float final_balance;
} BacktestResult;

// Simple SMA (for testing)
float compute_sma(__global OHLCVBar *ohlcv, int bar_idx, int period) {
    if (bar_idx < period - 1) return 0.0f;
    
    float sum = 0.0f;
    for (int i = 0; i < period; i++) {
        sum += ohlcv[bar_idx - i].close;
    }
    return sum / (float)period;
}

// MINIMAL kernel - just test data flow
__kernel void unified_backtest(
    __global CompactBotConfig *bots,
    __global OHLCVBar *ohlcv_data,
    __global int *cycle_starts,
    __global int *cycle_ends,
    const int num_cycles,
    const int total_bars,
    const float initial_balance,
    __global BacktestResult *results
) {
    int bot_idx = get_global_id(0);
    
    // Load bot
    CompactBotConfig bot = bots[bot_idx];
    
    // Initialize result
    BacktestResult result;
    result.bot_id = bot.bot_id;
    result.total_trades = 0;
    result.winning_trades = 0;
    result.losing_trades = 0;
    result.total_return_pct = 0.0f;
    result.sharpe_ratio = 0.0f;
    result.max_drawdown_pct = 0.0f;
    result.final_balance = initial_balance;
    
    // Simple test: compute one SMA
    if (total_bars > 20) {
        float sma = compute_sma(ohlcv_data, 20, 10);
        result.final_balance = initial_balance + sma * 0.01f;  // Tiny effect
    }
    
    // Simple backtest simulation (very simplified)
    for (int cycle = 0; cycle < num_cycles; cycle++) {
        int start_bar = cycle_starts[cycle];
        int end_bar = cycle_ends[cycle];
        
        // Count bars processed (as "trades")
        result.total_trades += (end_bar - start_bar + 1) / 100;
        result.winning_trades = result.total_trades / 2;
        result.losing_trades = result.total_trades - result.winning_trades;
    }
    
    result.total_return_pct = ((result.final_balance - initial_balance) / initial_balance) * 100.0f;
    
    // Write result
    results[bot_idx] = result;
}
