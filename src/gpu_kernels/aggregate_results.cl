__kernel void aggregate_bot_results(
    __global const float* chunk_results,
    __global float* aggregated_results,
    const int num_chunks,
    const int num_bots
) {
    const int bot_id = get_global_id(0);
    if (bot_id >= num_bots) return;

    // Result structure size in floats
    const int RESULT_SIZE_FLOATS = (64 + (100 * 12)) / 4;
    // Field offsets in floats (byte offset / 4)
    const int TOTAL_TRADES_OFFSET = 1;      // 4 / 4
    const int WINNING_TRADES_OFFSET = 2;    // 8 / 4
    const int LOSING_TRADES_OFFSET = 3;     // 12 / 4
    const int TOTAL_PNL_OFFSET = 4;         // 16 / 4
    const int MAX_DRAWDOWN_OFFSET = 5;      // 20 / 4

    float total_trades = 0.0f;
    float winning_trades = 0.0f;
    float losing_trades = 0.0f;
    float total_pnl = 0.0f;
    float max_drawdown = 0.0f;

    for (int chunk = 0; chunk < num_chunks; chunk++) {
        int chunk_offset = chunk * num_bots * RESULT_SIZE_FLOATS + bot_id * RESULT_SIZE_FLOATS;

        total_trades += chunk_results[chunk_offset + TOTAL_TRADES_OFFSET];
        winning_trades += chunk_results[chunk_offset + WINNING_TRADES_OFFSET];
        losing_trades += chunk_results[chunk_offset + LOSING_TRADES_OFFSET];
        total_pnl += chunk_results[chunk_offset + TOTAL_PNL_OFFSET];

        float chunk_drawdown = chunk_results[chunk_offset + MAX_DRAWDOWN_OFFSET];
        if (chunk_drawdown > max_drawdown) {
            max_drawdown = chunk_drawdown;
        }
    }

    int agg_offset = bot_id * 5;
    aggregated_results[agg_offset] = total_trades;
    aggregated_results[agg_offset + 1] = winning_trades;
    aggregated_results[agg_offset + 2] = losing_trades;
    aggregated_results[agg_offset + 3] = total_pnl;
    aggregated_results[agg_offset + 4] = max_drawdown;
}

/**
 * Ultra-fast aggregation kernel for flat bot-cycle results.
 * 
 * Each work item processes ONE bot and sums all its data points across
 * all cycles and chunks in parallel.
 * 
 * Input: Flat arrays of [bot_id, cycle_id, trades, wins, pnl] × N data points
 * Output: Aggregated [trades, wins, pnl] per bot
 */
__kernel void aggregate_flat_results(
    __global const int* bot_id_lookup,    // Bot IDs in order [bot0_id, bot1_id, ...]
    __global const int* data_bot_ids,     // Bot ID for each data point
    __global const int* data_cycle_ids,   // Cycle ID for each data point
    __global const int* data_trades,      // Trades for each data point
    __global const int* data_wins,        // Wins for each data point
    __global const float* data_pnls,      // PnL for each data point
    const int num_bots,
    const int num_data_points,
    __global float* output                // Output: [trades, wins, pnl] × num_bots
) {
    const int bot_idx = get_global_id(0);
    if (bot_idx >= num_bots) return;
    
    const int my_bot_id = bot_id_lookup[bot_idx];
    
    // Accumulate all data points belonging to this bot
    float total_trades = 0.0f;
    float total_wins = 0.0f;
    float total_pnl = 0.0f;
    
    for (int i = 0; i < num_data_points; i++) {
        if (data_bot_ids[i] == my_bot_id) {
            total_trades += (float)data_trades[i];
            total_wins += (float)data_wins[i];
            total_pnl += data_pnls[i];
        }
    }
    
    // Write aggregated results
    int out_idx = bot_idx * 3;
    output[out_idx] = total_trades;
    output[out_idx + 1] = total_wins;
    output[out_idx + 2] = total_pnl;
}