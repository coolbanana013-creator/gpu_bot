/**
 * GPU-Accelerated Data Logging and Serialization
 * High-performance kernels for converting bot data to CSV format.
 * Simplified version without sprintf for better OpenCL compatibility.
 */

// Simple integer to string conversion (up to 10 digits)
void int_to_string(int value, __global char* buffer, int* pos) {
    if (value == 0) {
        buffer[(*pos)++] = '0';
        return;
    }

    if (value < 0) {
        buffer[(*pos)++] = '-';
        value = -value;
    }

    // Convert to string in reverse
    char temp[12];
    int len = 0;
    while (value > 0 && len < 11) {
        temp[len++] = '0' + (value % 10);
        value /= 10;
    }

    // Reverse into buffer
    for (int i = len - 1; i >= 0; i--) {
        buffer[(*pos)++] = temp[i];
    }
}

// Simple float to string conversion (basic implementation)
void float_to_string(float value, int decimals, __global char* buffer, int* pos) {
    if (value < 0) {
        buffer[(*pos)++] = '-';
        value = -value;
    }

    // Integer part
    int int_part = (int)value;
    int_to_string(int_part, buffer, pos);

    if (decimals > 0) {
        buffer[(*pos)++] = '.';

        // Decimal part
        float decimal_part = value - int_part;
        for (int i = 0; i < decimals; i++) {
            decimal_part *= 10;
            int digit = (int)decimal_part;
            buffer[(*pos)++] = '0' + digit;
            decimal_part -= digit;
        }
    }
}

// Kernel to calculate CSV row offsets and lengths
__kernel void calculate_csv_row_offsets(
    __global const int* num_indicators,    // Number of indicators per bot
    __global int* output_offsets,          // Output: starting offset for each bot
    __global int* row_lengths,             // Output: length of each row
    const int num_bots,
    const int num_cycles,
    const int base_row_length
) {
    int bot_idx = get_global_id(0);

    if (bot_idx >= num_bots) return;

    // Calculate row length based on indicators and cycles
    int indicators = num_indicators[bot_idx];
    int cycle_data_length = num_cycles * 50;  // Rough estimate per cycle
    int indicator_data_length = indicators * 10;  // Rough estimate per indicator

    int total_length = base_row_length + cycle_data_length + indicator_data_length;

    // Calculate cumulative offset
    if (bot_idx == 0) {
        output_offsets[0] = 0;
    } else {
        output_offsets[bot_idx] = output_offsets[bot_idx - 1] + row_lengths[bot_idx - 1];
    }

    row_lengths[bot_idx] = total_length;
}

// Kernel to write CSV header
__kernel void write_csv_header(
    __global char* header_buffer,          // Output buffer for header
    const int num_cycles
) {
    int pos = 0;

    // Helper macro for writing strings
    #define WRITE_STR(str) \
        for (int i = 0; str[i] != '\0'; i++) { \
            header_buffer[pos++] = str[i]; \
        }

    #define WRITE_SEMICOLON() header_buffer[pos++] = ';'

    // Write header fields
    WRITE_STR("Generation");
    WRITE_SEMICOLON();
    WRITE_STR("BotID");
    WRITE_SEMICOLON();
    WRITE_STR("ProfitPct");
    WRITE_SEMICOLON();
    WRITE_STR("WinRate");
    WRITE_SEMICOLON();
    WRITE_STR("TotalTrades");
    WRITE_SEMICOLON();
    WRITE_STR("FinalBalance");
    WRITE_SEMICOLON();
    WRITE_STR("FitnessScore");
    WRITE_SEMICOLON();
    WRITE_STR("SharpeRatio");
    WRITE_SEMICOLON();
    WRITE_STR("MaxDrawdown");
    WRITE_SEMICOLON();
    WRITE_STR("SurvivedGenerations");
    WRITE_SEMICOLON();
    WRITE_STR("NumIndicators");
    WRITE_SEMICOLON();
    WRITE_STR("Leverage");
    WRITE_SEMICOLON();
    WRITE_STR("TotalPnL");
    WRITE_SEMICOLON();
    WRITE_STR("NumCycles");
    WRITE_SEMICOLON();
    WRITE_STR("IndicatorsUsed");
    WRITE_SEMICOLON();

    // Write per-cycle headers
    for (int c = 0; c < num_cycles; c++) {
        // Cycle trades
        WRITE_STR("Cycle");
        int_to_string(c, header_buffer, &pos);
        WRITE_STR("_Trades");
        WRITE_SEMICOLON();

        // Cycle profit percentage
        WRITE_STR("Cycle");
        int_to_string(c, header_buffer, &pos);
        WRITE_STR("_ProfitPct");
        WRITE_SEMICOLON();

        // Cycle win rate
        WRITE_STR("Cycle");
        int_to_string(c, header_buffer, &pos);
        WRITE_STR("_WinRate");
        if (c < num_cycles - 1) WRITE_SEMICOLON();
    }

    // End with newline
    header_buffer[pos++] = '\n';
    header_buffer[pos] = '\0';  // Null terminate
}

// Kernel to serialize bot performance data to CSV format
__kernel void serialize_bot_data_to_csv(
    __global const int* bot_ids,                    // Bot IDs
    __global const int* num_indicators,             // Number of indicators per bot
    __global const unsigned char* indicator_indices, // Indicator indices (8 per bot)
    __global const int* leverages,                  // Leverage values
    __global const int* survival_generations,       // Survival generations
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

    __global char* output_buffer,                   // Output CSV buffer
    __global int* output_offsets,                   // Starting offset for each bot's CSV row
    const int num_bots,
    const int num_cycles,
    const int generation,
    const float initial_balance
) {
    int bot_idx = get_global_id(0);

    if (bot_idx >= num_bots) return;

    // Calculate output position
    int start_offset = output_offsets[bot_idx];
    __global char* output = output_buffer + start_offset;
    int pos = 0;

    // Helper macro for writing strings
    #define WRITE_STR(str) \
        for (int i = 0; str[i] != '\0'; i++) { \
            output[pos++] = str[i]; \
        }

    #define WRITE_SEMICOLON() output[pos++] = ';'

    // Write generation
    int_to_string(generation, output, &pos);
    WRITE_SEMICOLON();

    // Write bot ID
    int_to_string(bot_ids[bot_idx], output, &pos);
    WRITE_SEMICOLON();

    // Write profit percentage (replace . with ,)
    float profit_pct = (total_pnls[bot_idx] / initial_balance) * 100.0f;
    float_to_string(profit_pct, 2, output, &pos);
    // Replace decimal point with comma (find and replace last '.')
    for (int i = pos - 1; i >= start_offset; i--) {
        if (output[i] == '.') {
            output[i] = ',';
            break;
        }
    }
    WRITE_SEMICOLON();

    // Write win rate
    float_to_string(win_rates[bot_idx], 4, output, &pos);
    // Replace decimal point with comma
    for (int i = pos - 1; i >= start_offset; i--) {
        if (output[i] == '.') {
            output[i] = ',';
            break;
        }
    }
    WRITE_SEMICOLON();

    // Write total trades
    int_to_string(total_trades[bot_idx], output, &pos);
    WRITE_SEMICOLON();

    // Write final balance
    float_to_string(final_balances[bot_idx], 2, output, &pos);
    // Replace decimal point with comma
    for (int i = pos - 1; i >= start_offset; i--) {
        if (output[i] == '.') {
            output[i] = ',';
            break;
        }
    }
    WRITE_SEMICOLON();

    // Write fitness score
    float_to_string(fitness_scores[bot_idx], 2, output, &pos);
    // Replace decimal point with comma
    for (int i = pos - 1; i >= start_offset; i--) {
        if (output[i] == '.') {
            output[i] = ',';
            break;
        }
    }
    WRITE_SEMICOLON();

    // Write Sharpe ratio
    float_to_string(sharpe_ratios[bot_idx], 2, output, &pos);
    // Replace decimal point with comma
    for (int i = pos - 1; i >= start_offset; i--) {
        if (output[i] == '.') {
            output[i] = ',';
            break;
        }
    }
    WRITE_SEMICOLON();

    // Write max drawdown
    float_to_string(max_drawdowns[bot_idx], 4, output, &pos);
    // Replace decimal point with comma
    for (int i = pos - 1; i >= start_offset; i--) {
        if (output[i] == '.') {
            output[i] = ',';
            break;
        }
    }
    WRITE_SEMICOLON();

    // Write survival generations
    int_to_string(survival_generations[bot_idx], output, &pos);
    WRITE_SEMICOLON();

    // Write num indicators
    int_to_string(num_indicators[bot_idx], output, &pos);
    WRITE_SEMICOLON();

    // Write leverage
    int_to_string(leverages[bot_idx], output, &pos);
    WRITE_SEMICOLON();

    // Write total PnL
    float_to_string(total_pnls[bot_idx], 2, output, &pos);
    // Replace decimal point with comma
    for (int i = pos - 1; i >= start_offset; i--) {
        if (output[i] == '.') {
            output[i] = ',';
            break;
        }
    }
    WRITE_SEMICOLON();

    // Write num cycles
    int_to_string(num_cycles, output, &pos);
    WRITE_SEMICOLON();

    // Write indicators used (simplified - just first few)
    int num_inds = num_indicators[bot_idx];
    for (int i = 0; i < min(num_inds, 3); i++) {
        if (i > 0) {
            WRITE_STR(", ");
        }
        int_to_string((int)indicator_indices[bot_idx * 8 + i], output, &pos);
    }
    if (num_inds > 3) {
        WRITE_STR("...");
    }
    WRITE_SEMICOLON();

    // Write per-cycle data
    for (int c = 0; c < num_cycles; c++) {
        int cycle_idx = bot_idx * num_cycles + c;

        // Cycle trades
        int c_trades = (int)per_cycle_trades[cycle_idx];
        int_to_string(c_trades, output, &pos);
        WRITE_SEMICOLON();

        // Cycle profit percentage
        float c_pnl = per_cycle_pnls[cycle_idx];
        float c_profit_pct = (c_pnl / initial_balance) * 100.0f;
        float_to_string(c_profit_pct, 2, output, &pos);
        // Replace decimal point with comma
        for (int i = pos - 1; i >= start_offset; i--) {
            if (output[i] == '.') {
                output[i] = ',';
                break;
            }
        }
        WRITE_SEMICOLON();

        // Cycle win rate
        float c_winrate = per_cycle_winrates[cycle_idx];
        float_to_string(c_winrate, 4, output, &pos);
        // Replace decimal point with comma
        for (int i = pos - 1; i >= start_offset; i--) {
            if (output[i] == '.') {
                output[i] = ',';
                break;
            }
        }
        if (c < num_cycles - 1) WRITE_SEMICOLON();
    }

    // End with newline
    output[pos++] = '\n';
    output[pos] = '\0';  // Null terminate
}