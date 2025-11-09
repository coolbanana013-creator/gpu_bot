/**
 * GPU-Accelerated Data Logging and Serialization
 * High-performance kernels for converting bot data to binary format.
 * CPU handles the final CSV formatting for reliability.
 */

// Structure to hold bot data in binary format
typedef struct {
    int generation;
    int bot_id;
    int num_indicators;
    int leverage;
    // NOTE: survival_generations removed - will be read directly from Python bot objects
    // This avoids binary serialization corruption issues
    int total_trades;
    float total_pnl;
    float win_rate;
    float final_balance;
    float fitness_score;
    float sharpe_ratio;
    float max_drawdown;
    unsigned char indicator_indices[8];
} BotData;

// Kernel to serialize bot data to binary format
__kernel void serialize_bot_data_binary(
    __global const int* bot_ids,                    // Bot IDs
    __global const int* num_indicators,             // Number of indicators per bot
    __global const unsigned char* indicator_indices, // Indicator indices (8 per bot)
    __global const int* leverages,                  // Leverage values
    // NOTE: survival_generations parameter removed - not written to binary buffer
    __global const float* total_pnls,               // Total PnL values
    __global const float* win_rates,                // Win rates
    __global const int* total_trades,               // Total trades
    __global const float* final_balances,           // Final balances
    __global const float* fitness_scores,           // Fitness scores
    __global const float* sharpe_ratios,            // Sharpe ratios
    __global const float* max_drawdowns,            // Max drawdowns
    __global const float* per_cycle_trades,         // Per-cycle trades (num_bots * num_cycles)
    __global const float* per_cycle_pnls,           // Per-cycle PnLs (num_bots * num_cycles)
    __global const float* per_cycle_winrates,       // Per-cycle winrates (num_bots * num_cycles)

    __global char* output_buffer,                   // Output binary buffer
    const int num_bots,
    const int num_cycles,
    const int generation,
    const float initial_balance
) {
    int bot_idx = get_global_id(0);

    if (bot_idx >= num_bots) return;

    // Calculate output position (each bot gets a fixed-size record)
    int record_size = sizeof(BotData) + num_cycles * 3 * sizeof(float);  // BotData + per-cycle data
    int start_offset = bot_idx * record_size;
    __global char* output = output_buffer + start_offset;

    // Create bot data structure
    BotData bot_data;
    bot_data.generation = generation;
    bot_data.bot_id = bot_ids[bot_idx];
    bot_data.num_indicators = num_indicators[bot_idx];
    bot_data.leverage = leverages[bot_idx];
    // survival_generations NOT written to buffer - read from Python objects instead
    bot_data.total_trades = total_trades[bot_idx];
    bot_data.total_pnl = total_pnls[bot_idx];
    bot_data.win_rate = win_rates[bot_idx];
    bot_data.final_balance = final_balances[bot_idx];
    bot_data.fitness_score = fitness_scores[bot_idx];
    bot_data.sharpe_ratio = sharpe_ratios[bot_idx];
    bot_data.max_drawdown = max_drawdowns[bot_idx];

    // Copy indicator indices
    for (int i = 0; i < 8; i++) {
        bot_data.indicator_indices[i] = indicator_indices[bot_idx * 8 + i];
    }

    // Copy bot data to output buffer
    __global BotData* output_bot = (__global BotData*)output;
    *output_bot = bot_data;

    // Copy per-cycle data
    __global float* output_cycles = (__global float*)(output + sizeof(BotData));
    for (int c = 0; c < num_cycles; c++) {
        int cycle_idx = bot_idx * num_cycles + c;
        output_cycles[c * 3] = per_cycle_trades[cycle_idx];      // trades
        output_cycles[c * 3 + 1] = per_cycle_pnls[cycle_idx];    // pnl
        output_cycles[c * 3 + 2] = per_cycle_winrates[cycle_idx]; // winrate
    }
}