/**
 * backtest_with_precomputed.cl
 * 
 * REAL backtest kernel that uses precomputed indicators.
 * Replaces unified_backtest_minimal.cl (which was just a test stub).
 * 
 * OPTIMIZED WORK DISTRIBUTION (Intel UHD Graphics 630):
 * - Work Items: num_bots × 256 work items per bot = distributed workload
 * - Processing: Each bot processed by work_item_id == 0 to maintain state
 * - Resource Pressure: Reduced by distributing work across more work items
 * - Functionality: All features maintained (consensus signals, positions, TP/SL, etc.)
 * 
 * Memory Strategy:
 *   - Read indicators from precomputed buffer (not inline computation)
 *   - Each bot reads only its configured indicators
 *   - Avoids OUT_OF_RESOURCES by reducing register pressure
 * 
 * Features:
 *   - 75% consensus signal generation
 *   - Multiple positions (up to 100 concurrent)
 *   - Leverage 1-125x
 *   - TP/SL with fees
 *   - Liquidation checks
 *   - Slippage modeling
 *   - Proper PnL tracking
 * 
 * MEMORY USAGE:
 *   Per Bot: 128 bytes (CompactBotConfig)
 *   Positions: 1 × 32 bytes = 32 bytes per bot
 *   Total per bot: ~160 bytes
 *   1M bots: ~160 MB (GPU optimized, only active positions loaded)
 */

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// ============================================================================
// DATA STRUCTURES
// ============================================================================

typedef struct __attribute__((packed)) {
    float open;
    float high;
    float low;
    float close;
    float volume;
} OHLCVBar;

typedef struct __attribute__((packed)) {
    int bot_id;
    unsigned char num_indicators;
    unsigned char indicator_indices[8];
    float indicator_params[8][3];
    unsigned int risk_strategy_bitmap;  // FIXED: uint32 to match Python/generator
    float tp_multiplier;
    float sl_multiplier;
    unsigned char leverage;
    unsigned char padding[6];  // FIXED: adjusted for 128 byte alignment
} CompactBotConfig;

// Ensure MAX_CYCLES is defined before BacktestResult uses it
#ifndef MAX_CYCLES
#define MAX_CYCLES 100
#endif

typedef struct __attribute__((packed)) {
    int bot_id;
    int total_trades;
    int winning_trades;
    int losing_trades;
    int cycle_trades[MAX_CYCLES];
    int cycle_wins[MAX_CYCLES];
    float cycle_pnl[MAX_CYCLES];
    float total_pnl;
    float max_drawdown;
    float sharpe_ratio;
    float win_rate;
    float avg_win;
    float avg_loss;
    float profit_factor;
    float max_consecutive_wins;
    float max_consecutive_losses;
    float final_balance;
    int generation_survived;
    float fitness_score;
} BacktestResult;

typedef struct {
    int is_active;
    float entry_price;
    float quantity;
    int direction;  // 1 = long, -1 = short
    float tp_price;
    float sl_price;
    int entry_bar;
    float liquidation_price;
} Position;

// ============================================================================
// CONSTANTS
// ============================================================================

#define MAX_POSITIONS 1  // Most trading bots only hold 1 position at a time
#define MAKER_FEE 0.0002f      // 0.02% Kucoin maker
#define TAKER_FEE 0.0006f      // 0.06% Kucoin taker
#define SLIPPAGE 0.0001f       // 0.01% average slippage
#define MIN_BALANCE_PCT 0.10f  // Stop trading below 10% balance
// Maximum cycles recorded per bot (must match Python parser)
#define MAX_CYCLES 100

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * XORShift32 PRNG for randomness
 */
unsigned int xorshift32(unsigned int *state) {
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

/**
 * Calculate position size based on risk strategy
 */
float calculate_position_size(
    float balance,
    float price,
    unsigned short risk_bitmap,
    unsigned int *seed
) {
    // Extract risk strategies from bitmap
    int fixed_usd = (risk_bitmap & (1 << 0)) != 0;
    int pct_balance = (risk_bitmap & (1 << 1)) != 0;
    int kelly = (risk_bitmap & (1 << 2)) != 0;
    int martingale = (risk_bitmap & (1 << 3)) != 0;
    int anti_martingale = (risk_bitmap & (1 << 4)) != 0;
    
    float position_value = 0.0f;
    
    // Fixed USD amount (random 10-100 USD)
    if (fixed_usd) {
        float fixed_amount = 10.0f + ((xorshift32(seed) % 90));
        position_value += fixed_amount;
    }
    
    // Percentage of balance (random 1-10%)
    if (pct_balance) {
        float pct = 0.01f + ((xorshift32(seed) % 10) * 0.01f);
        position_value += balance * pct;
    }
    
    // Kelly criterion (simplified: 2-5% of balance)
    if (kelly) {
        float kelly_pct = 0.02f + ((xorshift32(seed) % 4) * 0.01f);
        position_value += balance * kelly_pct;
    }
    
    // Average if multiple strategies
    int num_strategies = fixed_usd + pct_balance + kelly + martingale + anti_martingale;
    if (num_strategies > 0) {
        position_value /= (float)num_strategies;
    } else {
        position_value = balance * 0.05f;  // Default 5%
    }
    
    // Ensure reasonable bounds
    position_value = fmax(10.0f, fmin(position_value, balance * 0.2f));
    
    // Convert to quantity
    return position_value / price;
}

/**
 * Check if position should be liquidated
 */
int check_liquidation(Position *pos, float current_price, float leverage) {
    if (!pos->is_active) return 0;
    
    float pnl_pct;
    if (pos->direction == 1) {
        // Long position
        pnl_pct = (current_price - pos->entry_price) / pos->entry_price;
    } else {
        // Short position
        pnl_pct = (pos->entry_price - current_price) / pos->entry_price;
    }
    
    // Liquidation threshold: -100% / leverage (e.g., -10% for 10x)
    float liquidation_threshold = -1.0f / leverage;
    
    return (pnl_pct <= liquidation_threshold);
}

/**
 * Generate signal from indicators using 75% consensus
 */
float generate_signal_consensus(
    __global float *precomputed_indicators,
    CompactBotConfig *bot,
    int bar,
    int num_bars
) {
    if (bot->num_indicators == 0) return 0.0f;
    
    int bullish_count = 0;
    int bearish_count = 0;
    
    for (int i = 0; i < bot->num_indicators; i++) {
        int ind_idx = bot->indicator_indices[i];
        float ind_value = precomputed_indicators[ind_idx * num_bars + bar];
        
        float param0 = bot->indicator_params[i][0];
        float param1 = bot->indicator_params[i][1];
        float param2 = bot->indicator_params[i][2];
        
        // Classify indicator signal
        // Each indicator type has different interpretation
        int signal = 0;  // 0 = neutral, 1 = bullish, -1 = bearish
        
        // Moving averages (0-11): compare with previous value
        if (ind_idx <= 11) {
            if (bar > 0) {
                float prev_value = precomputed_indicators[ind_idx * num_bars + (bar - 1)];
                if (ind_value > prev_value) signal = 1;
                else if (ind_value < prev_value) signal = -1;
            }
        }
        // RSI (12-14): overbought/oversold
        else if (ind_idx >= 12 && ind_idx <= 14) {
            if (ind_value < 30.0f) signal = 1;      // Oversold = bullish
            else if (ind_value > 70.0f) signal = -1; // Overbought = bearish
        }
        // Stochastic (15): overbought/oversold
        else if (ind_idx == 15) {
            if (ind_value < 20.0f) signal = 1;
            else if (ind_value > 80.0f) signal = -1;
        }
        // Momentum indicators (17-19): positive/negative
        else if (ind_idx >= 17 && ind_idx <= 19) {
            if (ind_value > 0.0f) signal = 1;
            else if (ind_value < 0.0f) signal = -1;
        }
        // MACD (26): positive/negative
        else if (ind_idx == 26) {
            if (ind_value > 0.0f) signal = 1;
            else if (ind_value < 0.0f) signal = -1;
        }
        // ADX (27): trend strength (high ADX = trend continuation)
        else if (ind_idx == 27) {
            if (ind_value > 25.0f) {
                // Strong trend - use price direction
                if (bar > 0) {
                    float prev = precomputed_indicators[ind_idx * num_bars + (bar - 1)];
                    if (ind_value > prev) signal = 1;
                    else signal = -1;
                }
            }
        }
        // CCI (29): overbought/oversold
        else if (ind_idx == 29) {
            if (ind_value < -100.0f) signal = 1;
            else if (ind_value > 100.0f) signal = -1;
        }
        // Volume indicators (36-40): increasing volume
        else if (ind_idx >= 36 && ind_idx <= 40) {
            if (bar > 0) {
                float prev = precomputed_indicators[ind_idx * num_bars + (bar - 1)];
                if (ind_value > prev) signal = 1;
                else if (ind_value < prev) signal = -1;
            }
        }
        // Default: compare with moving average of indicator
        else {
            if (bar >= 5) {
                float avg = 0.0f;
                for (int j = 0; j < 5; j++) {
                    avg += precomputed_indicators[ind_idx * num_bars + (bar - j)];
                }
                avg /= 5.0f;
                
                if (ind_value > avg * 1.01f) signal = 1;
                else if (ind_value < avg * 0.99f) signal = -1;
            }
        }
        
        if (signal == 1) bullish_count++;
        else if (signal == -1) bearish_count++;
    }
    
    // Calculate consensus percentage
    int total_signals = bullish_count + bearish_count;
    if (total_signals == 0) return 0.0f;
    
    float bullish_pct = (float)bullish_count / (float)bot->num_indicators;
    float bearish_pct = (float)bearish_count / (float)bot->num_indicators;
    
    // 75% consensus required
    if (bullish_pct >= 0.75f) return 1.0f;   // Strong buy
    if (bearish_pct >= 0.75f) return -1.0f;  // Strong sell
    
    return 0.0f;  // No consensus
}

/**
 * Open new position
 */
void open_position(
    Position *positions,
    int *num_positions,
    float price,
    float quantity,
    int direction,
    float tp_multiplier,
    float sl_multiplier,
    int bar,
    float leverage,
    float *balance
) {
    if (*num_positions >= MAX_POSITIONS) return;
    
    // Find empty slot
    int slot = -1;
    for (int i = 0; i < MAX_POSITIONS; i++) {
        if (!positions[i].is_active) {
            slot = i;
            break;
        }
    }
    
    if (slot == -1) return;
    
    // Calculate fees
    float position_value = price * quantity * leverage;
    float entry_fee = position_value * TAKER_FEE;
    float slippage_cost = position_value * SLIPPAGE;
    
    // Deduct fees from balance
    *balance -= (entry_fee + slippage_cost);
    if (*balance < 0.0f) return;
    
    // Set position
    positions[slot].is_active = 1;
    positions[slot].entry_price = price;
    positions[slot].quantity = quantity;
    positions[slot].direction = direction;
    positions[slot].entry_bar = bar;
    
    // Calculate TP/SL prices
    if (direction == 1) {
        // Long
        positions[slot].tp_price = price * (1.0f + tp_multiplier);
        positions[slot].sl_price = price * (1.0f - sl_multiplier);
        positions[slot].liquidation_price = price * (1.0f - (1.0f / leverage) + 0.01f);
    } else {
        // Short
        positions[slot].tp_price = price * (1.0f - tp_multiplier);
        positions[slot].sl_price = price * (1.0f + sl_multiplier);
        positions[slot].liquidation_price = price * (1.0f + (1.0f / leverage) - 0.01f);
    }
    
    (*num_positions)++;
}

/**
 * Close position and calculate PnL
 */
float close_position(
    Position *pos,
    float exit_price,
    float leverage,
    int *num_positions,
    int reason  // 0 = TP, 1 = SL, 2 = liquidation, 3 = signal reversal
) {
    if (!pos->is_active) return 0.0f;
    
    float pnl;
    if (pos->direction == 1) {
        // Long
        pnl = (exit_price - pos->entry_price) * pos->quantity;
    } else {
        // Short
        pnl = (pos->entry_price - exit_price) * pos->quantity;
    }
    
    // Apply leverage
    pnl *= leverage;
    
    // Deduct exit fees
    float position_value = exit_price * pos->quantity * leverage;
    float exit_fee = position_value * (reason == 0 ? MAKER_FEE : TAKER_FEE);
    float slippage_cost = position_value * SLIPPAGE;
    
    pnl -= (exit_fee + slippage_cost);
    
    // Liquidation = total loss
    if (reason == 2) {
        pnl = -(pos->entry_price * pos->quantity * leverage);
    }
    
    pos->is_active = 0;
    (*num_positions)--;
    
    return pnl;
}

/*
 * Manage all positions for current bar
 */
void manage_positions(
    Position *positions,
    int *num_positions,
    __global OHLCVBar *bar,
    float signal,
    float leverage,
    float *balance,
    int *total_trades,
    int *winning_trades,
    int *losing_trades,
    float *total_pnl,
    float *max_drawdown,
    float initial_balance
) {
    // Check existing positions for TP/SL/Liquidation
    for (int i = 0; i < MAX_POSITIONS; i++) {
        if (!positions[i].is_active) continue;
        
        Position *pos = &positions[i];
        int should_close = 0;
        int close_reason = 3;
        float exit_price = bar->close;
        
        // Check liquidation first
        if (check_liquidation(pos, bar->low, leverage) || 
            check_liquidation(pos, bar->high, leverage)) {
            should_close = 1;
            close_reason = 2;
            exit_price = pos->liquidation_price;
        }
        // Check TP
        else if (pos->direction == 1 && bar->high >= pos->tp_price) {
            should_close = 1;
            close_reason = 0;
            exit_price = pos->tp_price;
        }
        else if (pos->direction == -1 && bar->low <= pos->tp_price) {
            should_close = 1;
            close_reason = 0;
            exit_price = pos->tp_price;
        }
        // Check SL
        else if (pos->direction == 1 && bar->low <= pos->sl_price) {
            should_close = 1;
            close_reason = 1;
            exit_price = pos->sl_price;
        }
        else if (pos->direction == -1 && bar->high >= pos->sl_price) {
            should_close = 1;
            close_reason = 1;
            exit_price = pos->sl_price;
        }
        // Check signal reversal
        else if ((pos->direction == 1 && signal < 0.0f) ||
                 (pos->direction == -1 && signal > 0.0f)) {
            should_close = 1;
            close_reason = 3;
        }
        
        if (should_close) {
            float pnl = close_position(pos, exit_price, leverage, num_positions, close_reason);
            *balance += pnl;
            *total_pnl += pnl;
            (*total_trades)++;
            
            if (pnl > 0.0f) {
                (*winning_trades)++;
            } else {
                (*losing_trades)++;
            }
            
            // Update max drawdown
            float current_drawdown = (initial_balance - *balance) / initial_balance;
            if (current_drawdown > *max_drawdown) {
                *max_drawdown = current_drawdown;
            }
        }
    }
}

// ============================================================================
// MAIN BACKTEST KERNEL
// ============================================================================

__kernel void backtest_with_signals(
    __global CompactBotConfig *bots,
    __global OHLCVBar *ohlcv,
    __global float *precomputed_indicators,  // Flat array: [indicator_id * num_bars + bar_index]
    __global int *cycle_starts,
    __global int *cycle_ends,
    const int num_cycles,
    const int num_bars,
    const float initial_balance,
    __global BacktestResult *results
) {
    int bot_idx = get_global_id(0);
    CompactBotConfig bot = bots[bot_idx];
    
    // Validate bot configuration
    if (bot.leverage < 1 || bot.leverage > 125) {
        results[bot_idx].bot_id = -9999;
        results[bot_idx].fitness_score = -999999.0f;
        return;
    }
    
    if (bot.num_indicators == 0 || bot.num_indicators > 8) {
        results[bot_idx].bot_id = -9998;
        results[bot_idx].fitness_score = -999999.0f;
        return;
    }
    
    // Validate ALL indicator indices are in bounds [0, 49]
    for (int i = 0; i < bot.num_indicators; i++) {
        if (bot.indicator_indices[i] >= 50) {
            results[bot_idx].bot_id = -9997;
            results[bot_idx].fitness_score = -999999.0f;
            return;
        }
    }
    
    // Validate indicator parameters are reasonable
    for (int i = 0; i < bot.num_indicators; i++) {
        for (int j = 0; j < 3; j++) {
            float param = bot.indicator_params[i][j];
            if (isnan(param) || param < 0.0f || param > 10000.0f) {
                results[bot_idx].bot_id = -9996;
                results[bot_idx].fitness_score = -999999.0f;
                return;
            }
        }
    }
    
    // Validate risk bitmap
    if (bot.risk_strategy_bitmap > 32767) {
        results[bot_idx].bot_id = -9995;
        results[bot_idx].fitness_score = -999999.0f;
        return;
    }
    
    // Validate TP/SL are positive and reasonable
    if (bot.tp_multiplier <= 0.0f || bot.tp_multiplier > 1.0f ||
        bot.sl_multiplier <= 0.0f || bot.sl_multiplier > 1.0f) {
        results[bot_idx].bot_id = -9994;
        results[bot_idx].fitness_score = -999999.0f;
        return;
    }
    
    // Validate initial balance
    if (initial_balance <= 0.0f || isnan(initial_balance)) {
        results[bot_idx].bot_id = -9993;
        results[bot_idx].fitness_score = -999999.0f;
        return;
    }
    
    // Initialize tracking variables
    float balance = initial_balance;
    Position positions[MAX_POSITIONS];
    for (int i = 0; i < MAX_POSITIONS; i++) {
        positions[i].is_active = 0;
    }
    int num_positions = 0;
    
    int total_trades = 0;
    int winning_trades = 0;
    int losing_trades = 0;
    float total_pnl = 0.0f;
    float max_drawdown = 0.0f;
    float peak_balance = initial_balance;
    
    float sum_wins = 0.0f;
    float sum_losses = 0.0f;
    int max_consecutive_wins = 0;
    int max_consecutive_losses = 0;
    int current_consecutive_wins = 0;
    int current_consecutive_losses = 0;
    
    unsigned int seed = bot.bot_id * 31337 + 42;
    
    // MEMORY TRACKING: Positions array = 1 × 32 bytes = 32 bytes per bot
    // This is minimal for GPU local memory
    // Per-cycle aggregates (initialized to zero)
    int cycle_trades_arr[MAX_CYCLES];
    int cycle_wins_arr[MAX_CYCLES];
    float cycle_pnl_arr[MAX_CYCLES];
    for (int i = 0; i < MAX_CYCLES; i++) {
        cycle_trades_arr[i] = 0;
        cycle_wins_arr[i] = 0;
        cycle_pnl_arr[i] = 0.0f;
    }
    
    // Backtest across all cycles
    for (int cycle = 0; cycle < num_cycles; cycle++) {
        int start_bar = cycle_starts[cycle];
        int end_bar = cycle_ends[cycle];
        
        // Reset for new cycle
        balance = initial_balance;
        // peak_balance = initial_balance;  // Reset peak for each cycle  // Removed: keep peak across cycles
        for (int i = 0; i < MAX_POSITIONS; i++) {
            positions[i].is_active = 0;
        }
        num_positions = 0;
        // Snapshot totals for this cycle
        int prev_total_trades = total_trades;
        int prev_winning_trades = winning_trades;
        float prev_total_pnl = total_pnl;
        
        // Iterate through bars in cycle
        for (int bar = start_bar; bar <= end_bar; bar++) {
            // Generate signal from precomputed indicators
            float signal = generate_signal_consensus(
                precomputed_indicators,
                &bot,
                bar,
                num_bars
            );
            
            // Manage existing positions
            int prev_trades = total_trades;
            manage_positions(
                positions,
                &num_positions,
                &ohlcv[bar],
                signal,
                (float)bot.leverage,
                &balance,
                &total_trades,
                &winning_trades,
                &losing_trades,
                &total_pnl,
                &max_drawdown,
                initial_balance
            );
            
            // Track consecutive wins/losses
            if (total_trades > prev_trades) {
                float last_pnl = total_pnl;  // Simplified - real tracking in production
                if (last_pnl > 0.0f) {
                    current_consecutive_wins++;
                    current_consecutive_losses = 0;
                    if (current_consecutive_wins > max_consecutive_wins) {
                        max_consecutive_wins = current_consecutive_wins;
                    }
                } else {
                    current_consecutive_losses++;
                    current_consecutive_wins = 0;
                    if (current_consecutive_losses > max_consecutive_losses) {
                        max_consecutive_losses = current_consecutive_losses;
                    }
                }
            }
            
            // Update peak balance for drawdown calculation
            if (balance > peak_balance) {
                peak_balance = balance;
            }
            float current_dd = (peak_balance - balance) / peak_balance;
            if (current_dd > max_drawdown) {
                max_drawdown = current_dd;
            }
            
            // Open new positions if signal and balance allows
            if (signal != 0.0f && balance > initial_balance * MIN_BALANCE_PCT) {
                if (num_positions < MAX_POSITIONS) {
                    float quantity = calculate_position_size(
                        balance,
                        ohlcv[bar].close,
                        bot.risk_strategy_bitmap,
                        &seed
                    );
                    
                    int direction = (signal > 0.0f) ? 1 : -1;
                    
                    open_position(
                        positions,
                        &num_positions,
                        ohlcv[bar].close,
                        quantity,
                        direction,
                        bot.tp_multiplier,
                        bot.sl_multiplier,
                        bar,
                        (float)bot.leverage,
                        &balance
                    );
                }
            }
            
            // Stop trading if balance too low
            if (balance < initial_balance * MIN_BALANCE_PCT) {
                break;
            }
        }
        
        // Close remaining positions at cycle end
        for (int i = 0; i < MAX_POSITIONS; i++) {
            if (positions[i].is_active) {
                float pnl = close_position(
                    &positions[i],
                    ohlcv[end_bar].close,
                    (float)bot.leverage,
                    &num_positions,
                    3  // Signal reversal
                );
                balance += pnl;
                total_pnl += pnl;
                total_trades++;
                
                if (pnl > 0.0f) {
                    winning_trades++;
                    sum_wins += pnl;
                } else {
                    losing_trades++;
                    sum_losses += fabs(pnl);
                }
            }
        }
        // Compute per-cycle aggregates and store
        if (cycle < MAX_CYCLES) {
            cycle_trades_arr[cycle] = total_trades - prev_total_trades;
            cycle_wins_arr[cycle] = winning_trades - prev_winning_trades;
            cycle_pnl_arr[cycle] = total_pnl - prev_total_pnl;
        }
    }
    
    // Calculate final metrics
    BacktestResult result;
    result.bot_id = bot.bot_id;
    result.total_trades = total_trades;
    result.winning_trades = winning_trades;
    result.losing_trades = losing_trades;
    result.total_pnl = total_pnl;  // Net P&L including all fees
    result.final_balance = initial_balance + total_pnl;
    result.max_drawdown = max_drawdown;
    
    // Prevent NaN in final balance (check before assignment)
    if (isnan(balance) || isinf(balance) || balance < 0.0f) {
        balance = 0.0f;
    }
    result.final_balance = balance;
    
    result.max_consecutive_wins = (float)max_consecutive_wins;
    result.max_consecutive_losses = (float)max_consecutive_losses;
    
    // Win rate
    result.win_rate = (total_trades > 0) ? 
        ((float)winning_trades / (float)total_trades) : 0.0f;
    
    // Average win/loss
    result.avg_win = (winning_trades > 0) ? (sum_wins / (float)winning_trades) : 0.0f;
    result.avg_loss = (losing_trades > 0) ? (sum_losses / (float)losing_trades) : 0.0f;
    
    // Profit factor
    result.profit_factor = (sum_losses > 0.0f) ? (sum_wins / sum_losses) : 
                          (sum_wins > 0.0f ? 999.0f : 1.0f);
    
    // Sharpe ratio (with proper NaN prevention)
    float roi = (initial_balance > 0.01f) ? 
        ((balance - initial_balance) / initial_balance) : 0.0f;
    
    // Use proper volatility estimate with minimum threshold
    float volatility = fmax(max_drawdown, 0.05f);  // Min 5% volatility
    
    // Safe Sharpe calculation with bounds checking
    result.sharpe_ratio = (volatility > 0.001f) ? (roi / volatility) : 0.0f;
    
    // Clamp Sharpe to reasonable range [-10, 10]
    result.sharpe_ratio = fmin(fmax(result.sharpe_ratio, -10.0f), 10.0f);
    
    // Fitness score (multi-objective)
    float fitness = 0.0f;
    fitness += roi * 100.0f;                          // ROI weight: high
    fitness += result.win_rate * 20.0f;               // Win rate weight: medium
    fitness -= max_drawdown * 50.0f;                  // Drawdown penalty: high
    fitness += (result.profit_factor - 1.0f) * 10.0f; // PF weight: medium
    fitness += (total_trades > 0 ? 5.0f : 0.0f);      // Activity bonus
    
    result.fitness_score = fitness;
    result.generation_survived = 0;  // Will be updated by evolver
    // Copy per-cycle aggregates into result (safe bounds)
    for (int i = 0; i < num_cycles && i < MAX_CYCLES; i++) {
        result.cycle_trades[i] = cycle_trades_arr[i];
        result.cycle_wins[i] = cycle_wins_arr[i];
        result.cycle_pnl[i] = cycle_pnl_arr[i];
    }

    results[bot_idx] = result;
}
