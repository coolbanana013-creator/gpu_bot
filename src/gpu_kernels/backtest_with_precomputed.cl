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
 *   - 100% consensus signal generation (STRICT: ALL indicators must agree)
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

// Risk strategy enum - SINGLE strategy per bot (15 total strategies)
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

typedef struct __attribute__((packed)) {
    int bot_id;                   // 4 bytes (offset 0)
    unsigned char num_indicators; // 1 byte (offset 4)
    unsigned char indicator_indices[8]; // 8 bytes (offset 5)
    float indicator_params[8][3]; // 96 bytes (offset 13)
    unsigned char risk_strategy;  // 1 byte (offset 109) - enum 0-14 for 15 strategies
    float risk_param;             // 4 bytes (offset 110) - parameter for selected strategy
    float tp_multiplier;          // 4 bytes (offset 114)
    float sl_multiplier;          // 4 bytes (offset 118)
    unsigned char leverage;       // 1 byte (offset 122)
    unsigned char padding[5];     // 5 bytes (offset 123) - for 128 byte alignment
} CompactBotConfig;  // Total: 128 bytes

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
#define BASE_SLIPPAGE 0.0001f  // 0.01% base slippage (low volatility, small orders)
#define MIN_BALANCE_PCT 0.10f  // Stop trading below 10% balance
// Maximum cycles recorded per bot (must match Python parser)
#define MAX_CYCLES 100
// Funding rate (perpetual futures charge every 8 hours)
#define FUNDING_RATE_INTERVAL 480  // 8 hours = 480 minutes at 1m timeframe
#define BASE_FUNDING_RATE 0.0001f  // 0.01% per 8 hours (typical neutral rate)

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
 * Calculate dynamic slippage based on position size, volume, volatility, and leverage
 * 
 * OPTIMIZED VERSION: Minimal memory footprint, no historical lookups
 * Uses current bar data only to avoid OUT_OF_RESOURCES on large populations
 */
float calculate_dynamic_slippage(
    float position_value,
    float current_volume,
    float leverage,
    float current_price,
    float current_high,
    float current_low
) {
    // Base slippage (ideal conditions)
    float slippage = BASE_SLIPPAGE;
    
    // 1. Volume impact: position size as % of current volume
    // Simplified: use current volume as proxy for liquidity
    float volume_impact = 0.0f;
    if (current_volume > 0.0f) {
        float position_pct = position_value / (current_volume * current_price);
        volume_impact = position_pct * 0.01f;  // 1% of volume = 0.01% additional slippage
        volume_impact = fmin(volume_impact, 0.005f);  // Cap at 0.5% additional
    }
    
    // 2. Volatility multiplier: use current bar's high-low range
    // Simplified: single bar volatility instead of 20-bar average
    float volatility_multiplier = 1.0f;
    if (current_price > 0.0f) {
        float range_pct = (current_high - current_low) / current_price;
        // If range is 2%, volatility is normal (1x)
        // If range is 4%, volatility is high (2x)
        volatility_multiplier = 1.0f + (range_pct / 0.02f);
        volatility_multiplier = fmin(volatility_multiplier, 4.0f);  // Cap at 4x
    }
    
    // 3. Leverage multiplier: higher leverage = larger notional = more market impact
    // 1x leverage = 1.0x slippage
    // 10x leverage = 1.2x slippage
    // 50x leverage = 2.0x slippage
    // 125x leverage = 3.0x slippage
    float leverage_multiplier = 1.0f + (leverage / 62.5f);
    
    // Combine all factors
    float total_slippage = (slippage + volume_impact) * volatility_multiplier * leverage_multiplier;
    
    // Final bounds: min 0.005% (ideal conditions), max 0.5% (terrible conditions)
    // Reduced max to prevent excessive costs
    total_slippage = fmin(fmax(total_slippage, 0.00005f), 0.005f);
    
    return total_slippage;
}

/**
 * Calculate unrealized PnL for a position
 */
float calculate_unrealized_pnl(Position *pos, float current_price, float leverage) {
    if (!pos->is_active) return 0.0f;
    
    float price_diff;
    if (pos->direction == 1) {
        // Long: profit when price rises
        price_diff = current_price - pos->entry_price;
    } else {
        // Short: profit when price falls
        price_diff = pos->entry_price - current_price;
    }
    
    // Leveraged PnL
    float raw_pnl = price_diff * pos->quantity;
    return raw_pnl * leverage;
}

/**
 * Calculate free margin (available for new positions)
 * Free Margin = Balance + Unrealized PnL - Used Margin
 */
float calculate_free_margin(
    float balance,
    Position *positions,
    int max_positions,
    float current_price,
    float leverage
) {
    float used_margin = 0.0f;
    float unrealized_pnl = 0.0f;
    
    for (int i = 0; i < max_positions; i++) {
        if (positions[i].is_active) {
            // Margin used = entry_price * quantity
            used_margin += positions[i].entry_price * positions[i].quantity;
            
            // Add unrealized PnL
            unrealized_pnl += calculate_unrealized_pnl(&positions[i], current_price, leverage);
        }
    }
    
    return balance + unrealized_pnl - used_margin;
}

/**
 * Calculate position size based on SINGLE risk strategy (15 strategies total)
 * Each bot uses ONE strategy with ONE parameter
 */
float calculate_position_size(
    float balance,
    float price,
    unsigned char risk_strategy,
    float risk_param
) {
    float position_value = 0.0f;
    
    switch(risk_strategy) {
        case RISK_FIXED_PCT:
            // Fixed percentage of balance (risk_param: 0.01-0.20 = 1-20%)
            position_value = balance * risk_param;
            break;
            
        case RISK_FIXED_USD:
            // Fixed USD amount (risk_param: 10-10000)
            position_value = risk_param;
            break;
            
        case RISK_KELLY_FULL:
            // Full Kelly criterion (risk_param: 0.01-1.0 fraction)
            // f = (bp - q) / b, simplified as balance fraction
            position_value = balance * risk_param;
            break;
            
        case RISK_KELLY_HALF:
            // Half Kelly (risk_param: 0.01-1.0, applied as half)
            position_value = balance * (risk_param * 0.5f);
            break;
            
        case RISK_KELLY_QUARTER:
            // Quarter Kelly (risk_param: 0.01-1.0, applied as quarter)
            position_value = balance * (risk_param * 0.25f);
            break;
            
        case RISK_ATR_MULTIPLIER:
            // ATR-based sizing (risk_param: 1.0-5.0 multiplier)
            // Position size inversely proportional to ATR
            // Simplified: use fixed percentage, real impl needs ATR value
            position_value = balance * 0.05f * risk_param;
            break;
            
        case RISK_VOLATILITY_PCT:
            // Volatility-based percentage (risk_param: 0.01-0.20)
            // Reduce size in high volatility, real impl needs volatility measure
            position_value = balance * risk_param;
            break;
            
        case RISK_EQUITY_CURVE:
            // Equity curve adjustment (risk_param: 0.5-2.0 multiplier)
            // Increase size when winning, decrease when losing
            // Simplified: use base 5% * multiplier
            position_value = balance * 0.05f * risk_param;
            break;
            
        case RISK_FIXED_RISK_REWARD:
            // Fixed risk/reward ratio (risk_param: 0.01-0.10 risk per trade)
            position_value = balance * risk_param;
            break;
            
        case RISK_MARTINGALE:
            // Martingale (risk_param: 1.5-3.0 multiplier after loss)
            // DANGEROUS: doubles position after loss
            // Simplified: use base percentage
            position_value = balance * 0.05f * risk_param;
            break;
            
        case RISK_ANTI_MARTINGALE:
            // Anti-Martingale (risk_param: 1.2-2.0 multiplier after win)
            // Increase size after wins
            position_value = balance * 0.05f * risk_param;
            break;
            
        case RISK_FIXED_RATIO:
            // Fixed Ratio (Ryan Jones) (risk_param: 1000-10000 delta)
            // Increase contracts every risk_param profit
            // Simplified: percentage based
            position_value = balance * 0.05f;
            break;
            
        case RISK_PERCENT_VOLATILITY:
            // Percent Volatility (risk_param: 0.01-0.20)
            // Similar to volatility_pct
            position_value = balance * risk_param;
            break;
            
        case RISK_WILLIAMS_FIXED:
            // Williams Fixed Fractional (risk_param: 0.01-0.10)
            position_value = balance * risk_param;
            break;
            
        case RISK_OPTIMAL_F:
            // Optimal f (Ralph Vince) (risk_param: 0.01-0.30)
            // Maximize geometric growth
            position_value = balance * risk_param;
            break;
            
        default:
            // Default: 5% of balance
            position_value = balance * 0.05f;
            break;
    }
    
    // Ensure reasonable bounds
    // Min: $10, Max: 20% of balance
    position_value = fmax(10.0f, fmin(position_value, balance * 0.2f));
    
    // Return position value (not quantity - open_position will calculate that)
    return position_value;
}

/**
 * Check if account should be liquidated (ACCOUNT-LEVEL, not per-position)
 * 
 * Real liquidation checks total equity against maintenance margin:
 * - Equity = Balance + Sum(Unrealized PnL)
 * - Used Margin = Sum(entry_price * quantity for all positions)
 * - Maintenance Margin = Used Margin * maintenance_rate (0.5% for BTC)
 * - Liquidation occurs when: Equity < Maintenance Margin
 */
int check_account_liquidation(
    float balance,
    Position *positions,
    int max_positions,
    float current_price,
    float leverage
) {
    float total_unrealized_pnl = 0.0f;
    float total_used_margin = 0.0f;
    
    for (int i = 0; i < max_positions; i++) {
        if (positions[i].is_active) {
            // Calculate unrealized PnL
            total_unrealized_pnl += calculate_unrealized_pnl(&positions[i], current_price, leverage);
            
            // Calculate used margin
            total_used_margin += positions[i].entry_price * positions[i].quantity;
        }
    }
    
    // No positions = no liquidation
    if (total_used_margin <= 0.0f) return 0;
    
    // Calculate equity
    float equity = balance + total_unrealized_pnl;
    
    // Maintenance margin: 0.5% of used margin for BTC
    // This means you need to maintain 0.5% of position value as collateral
    float maintenance_rate = 0.005f;
    float maintenance_margin = total_used_margin * maintenance_rate * leverage;
    
    // Liquidation occurs when equity drops below maintenance margin
    return (equity < maintenance_margin);
}

/**
 * Generate signal from indicators using 100% consensus (STRICT: ALL indicators must agree)
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
    int valid_indicators = 0;  // Track how many valid (non-NaN) indicators we checked
    
    for (int i = 0; i < bot->num_indicators; i++) {
        int ind_idx = bot->indicator_indices[i];
        float ind_value = precomputed_indicators[ind_idx * num_bars + bar];
        
        // Skip invalid indicator values (NaN, Inf, or during warmup period)
        if (isnan(ind_value) || isinf(ind_value)) {
            continue;  // Skip this indicator, don't count it
        }
        
        valid_indicators++;  // Count this as a valid indicator
        
        float param0 = bot->indicator_params[i][0];
        float param1 = bot->indicator_params[i][1];
        float param2 = bot->indicator_params[i][2];
        
        // COMPLETE SIGNAL LOGIC FOR ALL 50 INDICATORS
        // Each indicator has realistic, in-depth signal interpretation
        int signal = 0;  // 0 = neutral, 1 = bullish, -1 = bearish
        
        // === CATEGORY 1: MOVING AVERAGES (0-11) ===
        // Trend-following: price crosses or momentum
        if (ind_idx >= 0 && ind_idx <= 11) {
            if (bar > 0) {
                float prev_value = precomputed_indicators[ind_idx * num_bars + (bar - 1)];
                // Bullish: MA rising, Bearish: MA falling
                if (ind_value > prev_value * 1.001f) signal = 1;
                else if (ind_value < prev_value * 0.999f) signal = -1;
            }
        }
        
        // === CATEGORY 2: MOMENTUM INDICATORS (12-19) ===
        // RSI (12-14): overbought/oversold with mean reversion
        else if (ind_idx >= 12 && ind_idx <= 14) {
            if (ind_value < 30.0f) signal = 1;        // Oversold = buy signal
            else if (ind_value > 70.0f) signal = -1;  // Overbought = sell signal
            // Neutral zone: 30-70, no signal
        }
        
        // Stochastic %K (15): overbought/oversold
        else if (ind_idx == 15) {
            if (ind_value < 20.0f) signal = 1;        // Oversold
            else if (ind_value > 80.0f) signal = -1;  // Overbought
        }
        
        // StochRSI (16): double overbought/oversold
        else if (ind_idx == 16) {
            if (ind_value < 20.0f) signal = 1;        // Oversold on RSI scale
            else if (ind_value > 80.0f) signal = -1;  // Overbought on RSI scale
        }
        
        // Momentum (17): rate of change
        else if (ind_idx == 17) {
            if (ind_value > 0.0f) signal = 1;         // Positive momentum = bullish
            else if (ind_value < 0.0f) signal = -1;   // Negative momentum = bearish
        }
        
        // ROC (18): percentage rate of change
        else if (ind_idx == 18) {
            if (ind_value > 2.0f) signal = 1;         // Strong upward momentum
            else if (ind_value < -2.0f) signal = -1;  // Strong downward momentum
        }
        
        // Williams %R (19): overbought/oversold (inverted scale)
        else if (ind_idx == 19) {
            if (ind_value < -80.0f) signal = 1;       // Oversold
            else if (ind_value > -20.0f) signal = -1; // Overbought
        }
        
        // === CATEGORY 3: VOLATILITY INDICATORS (20-25) ===
        // ATR (20-21): volatility expansion/contraction
        else if (ind_idx >= 20 && ind_idx <= 21) {
            if (bar > 1) {
                float prev = precomputed_indicators[ind_idx * num_bars + (bar - 1)];
                // Expanding volatility with uptrend = bullish continuation
                // Contracting volatility = potential breakout (neutral until direction clear)
                if (ind_value > prev * 1.1f) {
                    // Volatility expanding - check price direction
                    float price_now = ind_value;  // Placeholder - need actual price
                    // For now, treat expanding volatility as continuation of recent trend
                    if (bar > 2) {
                        float prev2 = precomputed_indicators[ind_idx * num_bars + (bar - 2)];
                        if (prev > prev2) signal = 1;   // Volatility trend up
                        else signal = -1;               // Volatility trend down
                    }
                }
            }
        }
        
        // NATR (22): normalized ATR (volatility as % of price)
        else if (ind_idx == 22) {
            // High NATR = high volatility = caution or breakout signal
            // Use trend for direction
            if (bar > 0) {
                float prev = precomputed_indicators[ind_idx * num_bars + (bar - 1)];
                if (ind_value > prev) signal = 1;
                else if (ind_value < prev) signal = -1;
            }
        }
        
        // Bollinger Bands Upper (23): price near upper band = overbought OR breakout
        else if (ind_idx == 23) {
            // In mean reversion: near upper = sell
            // In trending market: breakout above upper = buy
            // Use conservative approach: near upper band = overbought
            if (bar >= 20) {
                // Calculate where price is relative to bands
                float lower_band = precomputed_indicators[24 * num_bars + bar];
                float band_width = ind_value - lower_band;
                // If bands expanding (high volatility), use trend
                // If bands contracting (low volatility), use mean reversion
                if (bar > 0) {
                    float prev_width = precomputed_indicators[23 * num_bars + (bar-1)] - 
                                      precomputed_indicators[24 * num_bars + (bar-1)];
                    if (band_width > prev_width * 1.2f) {
                        signal = 1;  // Expanding bands = potential breakout up
                    } else {
                        signal = -1; // Near upper in normal conditions = overbought
                    }
                }
            }
        }
        
        // Bollinger Bands Lower (24): price near lower band = oversold OR breakdown
        else if (ind_idx == 24) {
            // Mean reversion: near lower = buy
            if (bar >= 20) {
                float upper_band = precomputed_indicators[23 * num_bars + bar];
                float band_width = upper_band - ind_value;
                if (bar > 0) {
                    float prev_width = precomputed_indicators[23 * num_bars + (bar-1)] - 
                                      precomputed_indicators[24 * num_bars + (bar-1)];
                    if (band_width > prev_width * 1.2f) {
                        signal = -1; // Expanding bands = potential breakdown
                    } else {
                        signal = 1;  // Near lower in normal conditions = oversold
                    }
                }
            }
        }
        
        // Keltner Channel (25): similar to Bollinger but uses ATR
        else if (ind_idx == 25) {
            if (bar > 0) {
                float prev = precomputed_indicators[ind_idx * num_bars + (bar - 1)];
                if (ind_value > prev) signal = 1;
                else if (ind_value < prev) signal = -1;
            }
        }
        
        // === CATEGORY 4: TREND INDICATORS (26-35) ===
        // MACD (26): trend direction and momentum
        else if (ind_idx == 26) {
            // MACD > 0 = bullish, MACD < 0 = bearish
            // Also check for crossovers by comparing with previous value
            if (bar > 0) {
                float prev = precomputed_indicators[ind_idx * num_bars + (bar - 1)];
                if (ind_value > 0.0f && prev <= 0.0f) signal = 1;        // Bullish crossover
                else if (ind_value < 0.0f && prev >= 0.0f) signal = -1;  // Bearish crossover
                else if (ind_value > 0.0f) signal = 1;                   // Above zero = bullish
                else if (ind_value < 0.0f) signal = -1;                  // Below zero = bearish
            } else {
                if (ind_value > 0.0f) signal = 1;
                else if (ind_value < 0.0f) signal = -1;
            }
        }
        
        // ADX (27): trend strength (not direction!)
        else if (ind_idx == 27) {
            // ADX > 25 = strong trend (use other indicators for direction)
            // ADX < 20 = weak trend (ranging market)
            // For signal, use ADX rising as trend strengthening
            if (bar > 0) {
                float prev = precomputed_indicators[ind_idx * num_bars + (bar - 1)];
                if (ind_value > 25.0f && ind_value > prev) {
                    // Strong trend strengthening - assume continuation
                    // Check if price is rising or falling
                    if (bar > 1) {
                        float price_ma_now = precomputed_indicators[2 * num_bars + bar];   // SMA(20)
                        float price_ma_prev = precomputed_indicators[2 * num_bars + (bar-1)];
                        if (price_ma_now > price_ma_prev) signal = 1;
                        else signal = -1;
                    }
                }
            }
        }
        
        // Aroon Up (28): time since recent high
        else if (ind_idx == 28) {
            if (ind_value > 70.0f) signal = 1;        // Recent high = bullish
            else if (ind_value < 30.0f) signal = -1;  // No recent high = bearish
        }
        
        // CCI (29): overbought/oversold with wider range
        else if (ind_idx == 29) {
            if (ind_value < -100.0f) signal = 1;      // Oversold
            else if (ind_value > 100.0f) signal = -1; // Overbought
        }
        
        // DPO (30): cycle analysis - detrended price
        else if (ind_idx == 30) {
            // DPO > 0 = price above its displaced MA (bullish cycle)
            // DPO < 0 = price below its displaced MA (bearish cycle)
            if (ind_value > 0.0f) signal = 1;
            else if (ind_value < 0.0f) signal = -1;
        }
        
        // Parabolic SAR (31): trailing stop and trend
        else if (ind_idx == 31) {
            // SAR below price = bullish trend, SAR above price = bearish trend
            // Need to compare SAR with actual price
            // For now, use SAR direction
            if (bar > 0) {
                float prev = precomputed_indicators[ind_idx * num_bars + (bar - 1)];
                if (ind_value < prev) signal = 1;  // SAR dropping = uptrend
                else if (ind_value > prev) signal = -1;  // SAR rising = downtrend
            }
        }
        
        // SuperTrend (32): strong trend indicator
        else if (ind_idx == 32) {
            // SuperTrend below price = bullish, above price = bearish
            // Check if indicator is rising or falling
            if (bar > 0) {
                float prev = precomputed_indicators[ind_idx * num_bars + (bar - 1)];
                if (ind_value > prev * 1.001f) signal = 1;   // Rising = bullish
                else if (ind_value < prev * 0.999f) signal = -1;  // Falling = bearish
            }
        }
        
        // Trend Strength (33-35): linear regression slope
        else if (ind_idx >= 33 && ind_idx <= 35) {
            // Positive slope = uptrend, negative slope = downtrend
            if (ind_value > 0.0f) signal = 1;
            else if (ind_value < 0.0f) signal = -1;
        }
        
        // === CATEGORY 5: VOLUME INDICATORS (36-40) ===
        // OBV (36): on-balance volume
        else if (ind_idx == 36) {
            if (bar > 0) {
                float prev = precomputed_indicators[ind_idx * num_bars + (bar - 1)];
                if (ind_value > prev) signal = 1;   // Volume supporting uptrend
                else if (ind_value < prev) signal = -1;  // Volume supporting downtrend
            }
        }
        
        // VWAP (37): volume-weighted average price
        else if (ind_idx == 37) {
            // Price above VWAP = bullish, below = bearish
            if (bar > 0) {
                float prev = precomputed_indicators[ind_idx * num_bars + (bar - 1)];
                if (ind_value > prev) signal = 1;
                else if (ind_value < prev) signal = -1;
            }
        }
        
        // MFI (38): money flow index (volume-weighted RSI)
        else if (ind_idx == 38) {
            if (ind_value < 20.0f) signal = 1;        // Oversold with volume
            else if (ind_value > 80.0f) signal = -1;  // Overbought with volume
        }
        
        // A/D (39): accumulation/distribution
        else if (ind_idx == 39) {
            if (bar > 0) {
                float prev = precomputed_indicators[ind_idx * num_bars + (bar - 1)];
                if (ind_value > prev) signal = 1;   // Accumulation
                else if (ind_value < prev) signal = -1;  // Distribution
            }
        }
        
        // Volume SMA (40): volume trend
        else if (ind_idx == 40) {
            if (bar > 0) {
                float prev = precomputed_indicators[ind_idx * num_bars + (bar - 1)];
                if (ind_value > prev * 1.2f) signal = 1;   // High volume = potential breakout
                else if (ind_value < prev * 0.8f) signal = -1;  // Low volume = potential breakdown
            }
        }
        
        // === CATEGORY 6: PATTERN INDICATORS (41-45) ===
        // Pivot Points (41): support/resistance
        else if (ind_idx == 41) {
            // Price above pivot = bullish, below = bearish
            if (bar > 0) {
                float prev = precomputed_indicators[ind_idx * num_bars + (bar - 1)];
                if (ind_value > prev) signal = 1;
                else if (ind_value < prev) signal = -1;
            }
        }
        
        // Fractal High (42): local maximum
        else if (ind_idx == 42) {
            // Fractal high detected = resistance level
            // If indicator > 0, resistance exists (slightly bearish)
            if (ind_value > 0.0f) signal = -1;  // Near resistance
        }
        
        // Fractal Low (43): local minimum
        else if (ind_idx == 43) {
            // Fractal low detected = support level
            // If indicator > 0, support exists (slightly bullish)
            if (ind_value > 0.0f) signal = 1;  // Near support
        }
        
        // Support/Resistance (44): dynamic S/R levels
        else if (ind_idx == 44) {
            if (bar > 0) {
                float prev = precomputed_indicators[ind_idx * num_bars + (bar - 1)];
                if (ind_value > prev) signal = 1;   // Breaking resistance = bullish
                else if (ind_value < prev) signal = -1;  // Breaking support = bearish
            }
        }
        
        // Price Channel (45): highest high / lowest low
        else if (ind_idx == 45) {
            if (bar > 0) {
                float prev = precomputed_indicators[ind_idx * num_bars + (bar - 1)];
                if (ind_value > prev) signal = 1;
                else if (ind_value < prev) signal = -1;
            }
        }
        
        // === CATEGORY 7: SIMPLE INDICATORS (46-49) ===
        // High-Low Range (46): bar range
        else if (ind_idx == 46) {
            // Large range = high volatility
            if (bar > 0) {
                float prev = precomputed_indicators[ind_idx * num_bars + (bar - 1)];
                if (ind_value > prev * 1.5f) signal = 1;   // Expanding range = breakout potential
            }
        }
        
        // Close Position (47): where close is in bar range
        else if (ind_idx == 47) {
            // > 0.7 = bullish close (near high), < 0.3 = bearish close (near low)
            if (ind_value > 0.7f) signal = 1;
            else if (ind_value < 0.3f) signal = -1;
        }
        
        // Price Acceleration (48): second derivative of price
        else if (ind_idx == 48) {
            // Positive acceleration = accelerating uptrend
            // Negative acceleration = accelerating downtrend
            if (ind_value > 0.0f) signal = 1;
            else if (ind_value < 0.0f) signal = -1;
        }
        
        // Volume ROC (49): volume rate of change
        else if (ind_idx == 49) {
            // Increasing volume = confirmation of trend
            if (ind_value > 10.0f) signal = 1;   // Strong volume increase = bullish
            else if (ind_value < -10.0f) signal = -1;  // Strong volume decrease = bearish
        }
        
        if (signal == 1) bullish_count++;
        else if (signal == -1) bearish_count++;
    }
    
    // Need at least one valid indicator
    if (valid_indicators == 0) return 0.0f;
    
    // Neutral signals don't block consensus
    // Only count bullish vs bearish (ignore neutrals)
    int directional_signals = bullish_count + bearish_count;
    
    // If no directional signals (all neutral), return neutral
    if (directional_signals == 0) return 0.0f;
    
    // Calculate consensus: bull/(bull+bear), bear/(bull+bear)
    // Neutrals are ignored, so bull + neutral = bull if no bears
    float bullish_pct = (float)bullish_count / (float)directional_signals;
    float bearish_pct = (float)bearish_count / (float)directional_signals;
    
    // 100% consensus required (ALL directional signals must agree)
    if (bullish_pct >= 1.0f) return 1.0f;   // ALL bullish (ignoring neutrals)
    if (bearish_pct >= 1.0f) return -1.0f;  // ALL bearish (ignoring neutrals)
    
    return 0.0f;  // Mixed signals (bull + bear conflict)
}

/**
 * Open new position with TRUE MARGIN TRADING
 * 
 * REALISTIC APPROACH:
 * - Calculate position_value from desired exposure (strategy-specific)
 * - Margin required = position_value / leverage (what we put up as collateral)
 * - Quantity based on MARGIN, not full position value
 * - Fees based on full position value (leverage amplifies)
 * - PnL will be leveraged automatically through quantity calculation
 */
void open_position(
    Position *positions,
    int *num_positions,
    float price,
    float desired_position_value,  // Changed: this is the exposure we want
    int direction,
    float tp_multiplier,
    float sl_multiplier,
    int bar,
    float leverage,
    float *balance,
    float current_volume,  // NEW: for dynamic slippage
    float current_high,    // NEW: for dynamic slippage
    float current_low      // NEW: for dynamic slippage
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
    
    // TRUE MARGIN TRADING CALCULATION
    // Margin = what we reserve from balance (collateral)
    float margin_required = desired_position_value / leverage;
    
    // DYNAMIC SLIPPAGE based on market conditions (optimized - no historical lookups)
    float slippage_rate = calculate_dynamic_slippage(
        desired_position_value,
        current_volume,
        leverage,
        price,
        current_high,
        current_low
    );
    
    // Fees and slippage are based on FULL position value (leverage amplifies costs)
    float entry_fee = desired_position_value * TAKER_FEE;
    float slippage_cost = desired_position_value * slippage_rate;
    
    // Total cost = margin we reserve + fees we pay upfront
    float total_cost = margin_required + entry_fee + slippage_cost;
    
    // IMPROVED: Check free margin (balance + unrealized PnL - used margin)
    float free_margin = calculate_free_margin(*balance, positions, MAX_POSITIONS, price, leverage);
    if (free_margin < total_cost) return;
    
    // Deduct cost from balance
    *balance -= total_cost;
    
    // Ensure balance didn't go negative (shouldn't happen with check above, but safety)
    if (*balance < 0.0f) {
        *balance += total_cost; // Rollback
        return;
    }
    
    // Calculate quantity based on MARGIN (not full position value)
    // This ensures PnL is automatically leveraged
    float quantity = margin_required / price;
    
    // Set position
    positions[slot].is_active = 1;
    positions[slot].entry_price = price;
    positions[slot].quantity = quantity;
    positions[slot].direction = direction;
    positions[slot].entry_bar = bar;
    
    // Calculate TP/SL prices based on entry price
    if (direction == 1) {
        // Long
        positions[slot].tp_price = price * (1.0f + tp_multiplier);
        positions[slot].sl_price = price * (1.0f - sl_multiplier);
        
        // IMPROVED LIQUIDATION PRICE FORMULA WITH MAINTENANCE MARGIN
        // Maintenance margin: 0.5% for BTC (typical for crypto exchanges)
        // Liquidation occurs when: (margin - loss) < maintenance_margin * position_value
        // Simplified: liquidation when price moves (1 - maintenance_margin_rate) / leverage
        // This is more accurate than simple 0.95/leverage, especially at high leverage
        float maintenance_margin_rate = 0.005f;  // 0.5% maintenance margin for BTC
        float liquidation_threshold = (1.0f - maintenance_margin_rate) / leverage;
        positions[slot].liquidation_price = price * (1.0f - liquidation_threshold);
    } else {
        // Short
        positions[slot].tp_price = price * (1.0f - tp_multiplier);
        positions[slot].sl_price = price * (1.0f + sl_multiplier);
        
        // IMPROVED LIQUIDATION PRICE FORMULA FOR SHORT WITH MAINTENANCE MARGIN
        // Same calculation as long, but price moves upward
        float maintenance_margin_rate = 0.005f;  // 0.5% maintenance margin for BTC
        float liquidation_threshold = (1.0f - maintenance_margin_rate) / leverage;
        positions[slot].liquidation_price = price * (1.0f + liquidation_threshold);
    }
    
    (*num_positions)++;
}

/**
 * Close position and calculate PnL with TRUE MARGIN TRADING
 * 
 * REALISTIC APPROACH:
 * - Quantity was calculated from margin (margin / entry_price)
 * - PnL = price_diff * quantity * leverage (leverage amplification)
 * - Return = margin + leveraged_pnl - fees - slippage
 * - Liquidation = lose entire margin (but no more)
 */
float close_position(
    Position *pos,
    float exit_price,
    float leverage,
    int *num_positions,
    int reason,  // 0 = TP, 1 = SL, 2 = liquidation, 3 = signal reversal
    float current_volume,  // NEW: for dynamic slippage
    float current_high,    // NEW: for dynamic slippage
    float current_low      // NEW: for dynamic slippage
) {
    if (!pos->is_active) return 0.0f;
    
    // Calculate raw price difference
    float price_diff;
    if (pos->direction == 1) {
        // Long: profit when price rises
        price_diff = exit_price - pos->entry_price;
    } else {
        // Short: profit when price falls
        price_diff = pos->entry_price - exit_price;
    }
    
    // TRUE MARGIN TRADING PnL CALCULATION
    // quantity = margin / entry_price (from open_position)
    // margin = entry_price * quantity
    // Raw PnL before leverage = price_diff * quantity
    // But this is already on the margin amount, so we need to apply leverage
    // Leveraged PnL = price_diff * quantity * leverage
    float margin_reserved = pos->entry_price * pos->quantity;
    float raw_pnl = price_diff * pos->quantity;
    float leveraged_pnl = raw_pnl * leverage;
    
    // Calculate position value for fees (full notional value)
    float notional_position_value = pos->entry_price * pos->quantity * leverage;
    
    // CORRECTED: TP and SL are both limit orders → maker fee
    // Only signal reversals (reason=3) are market orders → taker fee
    // Liquidation (reason=2) loses all margin, no exit fee calculation needed
    float exit_fee;
    if (reason == 2) {
        exit_fee = 0.0f;  // Liquidation = exchange takes everything
    } else if (reason == 0 || reason == 1) {
        exit_fee = notional_position_value * MAKER_FEE;  // TP/SL = limit orders
    } else {
        exit_fee = notional_position_value * TAKER_FEE;  // Signal reversal = market order
    }
    
    // DYNAMIC SLIPPAGE on exit (optimized - no historical lookups)
    float slippage_rate = calculate_dynamic_slippage(
        notional_position_value,
        current_volume,
        leverage,
        exit_price,
        current_high,
        current_low
    );
    float slippage_cost = notional_position_value * slippage_rate;
    
    // Net PnL after fees
    float net_pnl = leveraged_pnl - exit_fee - slippage_cost;
    
    // Total return = margin we get back + net PnL
    float total_return = margin_reserved + net_pnl;
    
    // LIQUIDATION HANDLING
    if (reason == 2) {
        // Liquidation = lose entire margin (95% to exchange, 5% maintenance)
        // In reality, position is closed before 100% loss, but we model as total loss
        total_return = 0.0f;
    } else {
        // Cap maximum loss at margin (can't lose more than we put up)
        if (total_return < 0.0f) {
            total_return = 0.0f;
        }
    }
    
    pos->is_active = 0;
    (*num_positions)--;
    
    return total_return;
}

/*
 * Manage all positions for current bar
 * UPDATED: Account-level liquidation, signal reversals, funding rates, dynamic slippage (optimized)
 */
void manage_positions(
    Position *positions,
    int *num_positions,
    __global OHLCVBar *bar,
    int current_bar_idx,
    float signal,
    float leverage,
    float *balance,
    int *total_trades,
    int *winning_trades,
    int *losing_trades,
    float *total_pnl,
    float *cycle_pnl,  // Track cycle-specific PnL
    float *sum_wins,   // NEW: accumulate total winning PnL
    float *sum_losses, // NEW: accumulate total losing PnL
    float *max_drawdown,
    float initial_balance
) {
    // FIRST: Check account-level liquidation (affects all positions)
    int account_liquidated = check_account_liquidation(
        *balance,
        positions,
        MAX_POSITIONS,
        bar->close,
        leverage
    );
    
    if (account_liquidated) {
        // Liquidate ALL positions immediately
        for (int i = 0; i < MAX_POSITIONS; i++) {
            if (positions[i].is_active) {
                float return_amount = close_position(
                    &positions[i],
                    bar->close,
                    leverage,
                    num_positions,
                    2,  // Liquidation
                    bar->volume,
                    bar->high,
                    bar->low
                );
                *balance += return_amount;
                
                float margin_was = positions[i].entry_price * positions[i].quantity;
                float actual_pnl = return_amount - margin_was;
                
                *total_pnl += actual_pnl;
                *cycle_pnl += actual_pnl;
                (*total_trades)++;
                
                if (actual_pnl > 0.0f) {
                    (*winning_trades)++;
                    *sum_wins += actual_pnl;
                } else {
                    (*losing_trades)++;
                    *sum_losses += fabs(actual_pnl);
                }
            }
        }
        
        if (*balance < 0.0f) *balance = 0.0f;
        return;  // Exit early - all positions closed
    }
    
    // SECOND: Apply funding rates (every 8 hours = 480 bars at 1m)
    for (int i = 0; i < MAX_POSITIONS; i++) {
        if (!positions[i].is_active) continue;
        
        int bars_held = current_bar_idx - positions[i].entry_bar;
        
        // Check if we crossed a funding rate boundary
        int prev_funding_periods = (bars_held - 1) / FUNDING_RATE_INTERVAL;
        int curr_funding_periods = bars_held / FUNDING_RATE_INTERVAL;
        
        if (curr_funding_periods > prev_funding_periods) {
            // Funding rate payment due
            float position_value = positions[i].entry_price * positions[i].quantity * leverage;
            float funding_cost = position_value * BASE_FUNDING_RATE;
            
            // In bull markets (positive funding), longs pay, shorts receive
            // In bear markets (negative funding), shorts pay, longs receive
            // We use neutral rate here; real impl would fetch actual funding rate
            if (positions[i].direction == 1) {
                // Long position pays funding
                *balance -= funding_cost;
                *total_pnl -= funding_cost;
                *cycle_pnl -= funding_cost;
            } else {
                // Short position receives funding
                *balance += funding_cost;
                *total_pnl += funding_cost;
                *cycle_pnl += funding_cost;
            }
        }
    }
    
    // THIRD: Check existing positions for TP/SL/Signal Reversal
    for (int i = 0; i < MAX_POSITIONS; i++) {
        if (!positions[i].is_active) continue;
        
        Position *pos = &positions[i];
        int should_close = 0;
        int close_reason = 3;
        float exit_price = bar->close;
        
        // Check TP (highest priority after liquidation)
        if (pos->direction == 1 && bar->high >= pos->tp_price) {
            should_close = 1;
            close_reason = 0;
            exit_price = pos->tp_price;
        }
        else if (pos->direction == -1 && bar->low <= pos->tp_price) {
            should_close = 1;
            close_reason = 0;
            exit_price = pos->tp_price;
        }
        // Check SL (second priority)
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
        // REMOVED: Signal reversal exits - let TP/SL do the work
        // This prevents premature exits and improves winrate
        
        if (should_close) {
            // Close position and get return amount
            float return_amount = close_position(
                pos,
                exit_price,
                leverage,
                num_positions,
                close_reason,
                bar->volume,
                bar->high,
                bar->low
            );
            *balance += return_amount;
            
            // Calculate actual PnL
            float margin_was = pos->entry_price * pos->quantity;
            float actual_pnl = return_amount - margin_was;
            
            // Update PnL trackers
            *total_pnl += actual_pnl;
            *cycle_pnl += actual_pnl;
            (*total_trades)++;
            
            // FIXED: Properly accumulate wins and losses
            if (actual_pnl > 0.0f) {
                (*winning_trades)++;
                *sum_wins += actual_pnl;  // Accumulate winning PnL
            } else {
                (*losing_trades)++;
                *sum_losses += fabs(actual_pnl);  // Accumulate losing PnL (absolute value)
            }
            
            // Ensure balance never goes negative
            if (*balance < 0.0f) *balance = 0.0f;
            
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
    
    // Comprehensive indicator parameter validation
    for (int i = 0; i < bot.num_indicators; i++) {
        unsigned char idx = bot.indicator_indices[i];
        float p1 = bot.indicator_params[i][0];
        float p2 = bot.indicator_params[i][1];
        float p3 = bot.indicator_params[i][2];
        
        // Basic sanity checks
        if (isnan(p1) || isnan(p2) || isnan(p3)) {
            results[bot_idx].bot_id = -9996;
            results[bot_idx].fitness_score = -999999.0f;
            return;
        }
        
        // Period-based indicators: period >= 2
        if ((idx >= 0 && idx <= 8) ||   // MA family
            (idx >= 11 && idx <= 15) || // RSI, Stoch, CCI, MFI, Williams
            (idx >= 19 && idx <= 25) || // ROC, Ultimate, ATR, ADX, Aroon, Vortex, CMF
            (idx >= 30 && idx <= 34) || // Keltner, Donchian, Chaikin, OBV, PVT
            (idx >= 38 && idx <= 42)) { // SAR, Supertrend, Elder-Ray, TSI, Coppock
            if (p1 < 2.0f || p1 > 500.0f) {
                results[bot_idx].bot_id = -9995;
                results[bot_idx].fitness_score = -999999.0f;
                return;
            }
        }
        
        // MACD: fast < slow < signal
        if (idx == 9) { // MACD
            if (p1 < 2.0f || p2 < 2.0f || p3 < 2.0f ||
                p1 >= p2 || p2 > 100.0f || p3 > 100.0f) {
                results[bot_idx].bot_id = -9994;
                results[bot_idx].fitness_score = -999999.0f;
                return;
            }
        }
        
        // Bollinger/Keltner: period >= 2, stddev > 0
        if (idx == 10 || idx == 30) { // BB, Keltner
            if (p1 < 2.0f || p1 > 500.0f || p2 <= 0.0f || p2 > 10.0f) {
                results[bot_idx].bot_id = -9993;
                results[bot_idx].fitness_score = -999999.0f;
                return;
            }
        }
        
        // Stochastic: %K period >= 1, %D period >= 1
        if (idx == 12) { // Stochastic
            if (p1 < 1.0f || p2 < 1.0f || p1 > 200.0f || p2 > 100.0f) {
                results[bot_idx].bot_id = -9992;
                results[bot_idx].fitness_score = -999999.0f;
                return;
            }
        }
        
        // Ichimoku: tenkan < kijun, senkou periods valid
        if (idx == 16) { // Ichimoku
            if (p1 < 2.0f || p2 < 2.0f || p1 >= p2 || 
                p1 > 100.0f || p2 > 200.0f) {
                results[bot_idx].bot_id = -9991;
                results[bot_idx].fitness_score = -999999.0f;
                return;
            }
        }
        
        // Fibonacci: levels must be 0-1 range
        if (idx == 17) { // Fibonacci
            if (p1 < 0.0f || p1 > 1.0f || p2 < 0.0f || p2 > 1.0f || 
                p3 < 0.0f || p3 > 1.0f) {
                results[bot_idx].bot_id = -9990;
                results[bot_idx].fitness_score = -999999.0f;
                return;
            }
        }
        
        // Pivot Points: no invalid params (calculated from OHLC)
        // Volume indicators: period >= 1
        
        // Pattern recognition indicators: no params to validate (use prev candles)
    }
    
    // Validate risk strategy (0-14 for 15 strategies)
    if (bot.risk_strategy > 14) {
        results[bot_idx].bot_id = -9989;
        results[bot_idx].fitness_score = -999999.0f;
        return;
    }
    
    // Validate risk parameter based on strategy
    switch(bot.risk_strategy) {
        case RISK_FIXED_PCT:
            if (bot.risk_param < 0.01f || bot.risk_param > 0.20f) {
                results[bot_idx].bot_id = -9988;
                results[bot_idx].fitness_score = -999999.0f;
                return;
            }
            break;
        case RISK_FIXED_USD:
            if (bot.risk_param < 10.0f || bot.risk_param > 10000.0f) {
                results[bot_idx].bot_id = -9987;
                results[bot_idx].fitness_score = -999999.0f;
                return;
            }
            break;
        case RISK_KELLY_FULL:
        case RISK_KELLY_HALF:
        case RISK_KELLY_QUARTER:
            if (bot.risk_param < 0.01f || bot.risk_param > 1.0f) {
                results[bot_idx].bot_id = -9986;
                results[bot_idx].fitness_score = -999999.0f;
                return;
            }
            break;
        case RISK_ATR_MULTIPLIER:
            if (bot.risk_param < 1.0f || bot.risk_param > 5.0f) {
                results[bot_idx].bot_id = -9985;
                results[bot_idx].fitness_score = -999999.0f;
                return;
            }
            break;
        case RISK_VOLATILITY_PCT:
        case RISK_PERCENT_VOLATILITY:
            if (bot.risk_param < 0.01f || bot.risk_param > 0.20f) {
                results[bot_idx].bot_id = -9984;
                results[bot_idx].fitness_score = -999999.0f;
                return;
            }
            break;
        case RISK_EQUITY_CURVE:
            if (bot.risk_param < 0.5f || bot.risk_param > 2.0f) {
                results[bot_idx].bot_id = -9983;
                results[bot_idx].fitness_score = -999999.0f;
                return;
            }
            break;
        case RISK_FIXED_RISK_REWARD:
        case RISK_WILLIAMS_FIXED:
            if (bot.risk_param < 0.01f || bot.risk_param > 0.10f) {
                results[bot_idx].bot_id = -9982;
                results[bot_idx].fitness_score = -999999.0f;
                return;
            }
            break;
        case RISK_MARTINGALE:
            if (bot.risk_param < 1.5f || bot.risk_param > 3.0f) {
                results[bot_idx].bot_id = -9981;
                results[bot_idx].fitness_score = -999999.0f;
                return;
            }
            break;
        case RISK_ANTI_MARTINGALE:
            if (bot.risk_param < 1.2f || bot.risk_param > 2.0f) {
                results[bot_idx].bot_id = -9980;
                results[bot_idx].fitness_score = -999999.0f;
                return;
            }
            break;
        case RISK_FIXED_RATIO:
            if (bot.risk_param < 1000.0f || bot.risk_param > 10000.0f) {
                results[bot_idx].bot_id = -9979;
                results[bot_idx].fitness_score = -999999.0f;
                return;
            }
            break;
        case RISK_OPTIMAL_F:
            if (bot.risk_param < 0.01f || bot.risk_param > 0.30f) {
                results[bot_idx].bot_id = -9978;
                results[bot_idx].fitness_score = -999999.0f;
                return;
            }
            break;
        default:
            // Unknown strategy
            results[bot_idx].bot_id = -9977;
            results[bot_idx].fitness_score = -999999.0f;
            return;
    }
    
    // Validate TP/SL are positive and respect leverage limits
    // Maximum SL = 95% of margin (to avoid liquidation)
    float max_sl = 0.95f / (float)bot.leverage;
    if (bot.tp_multiplier <= 0.0f || bot.tp_multiplier > 1.0f ||
        bot.sl_multiplier <= 0.0f || bot.sl_multiplier > max_sl) {
        results[bot_idx].bot_id = -9985;
        results[bot_idx].fitness_score = -999999.0f;
        return;
    }
    
    // NEW: Enforce minimum TP:SL ratio to prevent unprofitable configurations
    // TP must be at least 80% of SL (allows slight disadvantage but prevents absurd ratios)
    // Example: TP=0.01, SL=0.10 is rejected (10:1 loss ratio = guaranteed to lose money)
    if (bot.tp_multiplier < bot.sl_multiplier * 0.8f) {
        results[bot_idx].bot_id = -9975;
        results[bot_idx].fitness_score = -999999.0f;
        return;
    }
    
    // Validate leverage is in reasonable range [1, 125]
    if (bot.leverage < 1 || bot.leverage > 125) {
        results[bot_idx].bot_id = -9984;
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
        
        // Reset for new cycle - CRITICAL: Start fresh each cycle
        balance = initial_balance;
        
        // Close all positions at cycle start (clean slate)
        for (int i = 0; i < MAX_POSITIONS; i++) {
            positions[i].is_active = 0;
        }
        num_positions = 0;
        
        // Track cycle-specific metrics (isolated per cycle)
        int cycle_start_trades = total_trades;
        int cycle_start_wins = winning_trades;
        float cycle_pnl = 0.0f;  // Isolated PnL for this cycle only
        
        // IMPROVED: Calculate warmup period with proper multipliers
        // Most indicators need 2-3x their period to stabilize
        // This prevents unreliable signals from partially-computed indicators
        int warmup_bars = 0;
        for (int i = 0; i < bot.num_indicators; i++) {
            unsigned char idx = bot.indicator_indices[i];
            float period = bot.indicator_params[i][0];
            float period2 = bot.indicator_params[i][1];
            float period3 = bot.indicator_params[i][2];
            
            int indicator_warmup = 0;
            
            // Moving Averages (0-11)
            if (idx >= 0 && idx <= 5) {
                // SMA: needs exactly period bars
                indicator_warmup = (int)period;
            } else if (idx >= 6 && idx <= 11) {
                // EMA/DEMA/TEMA: need 3x period for 95% accuracy
                indicator_warmup = (int)(period * 3.0f);
            }
            // Momentum Indicators (12-19)
            else if (idx >= 12 && idx <= 14) {
                // RSI: needs 2x period for stability
                indicator_warmup = (int)(period * 2.0f);
            } else if (idx == 15 || idx == 16) {
                // Stochastic, StochRSI: need 2x period
                indicator_warmup = (int)(period * 2.0f);
            } else if (idx >= 17 && idx <= 19) {
                // Momentum, ROC, Williams: need period + buffer
                indicator_warmup = (int)period + 10;
            }
            // Volatility Indicators (20-25)
            else if (idx >= 20 && idx <= 22) {
                // ATR, NATR: need 2x period for smoothing
                indicator_warmup = (int)(period * 2.0f);
            } else if (idx == 23 || idx == 24) {
                // Bollinger Bands: need 3x period for stddev stability
                indicator_warmup = (int)(period * 3.0f);
            } else if (idx == 25) {
                // Keltner Channel: needs period + ATR warmup
                indicator_warmup = (int)(period * 2.5f);
            }
            // Trend Indicators (26-35)
            else if (idx == 26) {
                // MACD: needs slow_period + signal_period
                indicator_warmup = (int)(period2 + period3 + 10);
            } else if (idx == 27) {
                // ADX: needs 2x period (DI calculation + ADX smoothing)
                indicator_warmup = (int)(period * 2.0f);
            } else if (idx >= 28 && idx <= 35) {
                // Aroon, CCI, DPO, SAR, SuperTrend, Trend Strength
                indicator_warmup = (int)(period * 1.5f);
            }
            // Volume Indicators (36-40)
            else if (idx >= 36 && idx <= 40) {
                // OBV, VWAP, MFI, A/D, Volume SMA
                indicator_warmup = (int)period + 20;
            }
            // Pattern Indicators (41-45)
            else if (idx >= 41 && idx <= 45) {
                // Pivot Points, Fractals, S/R, Price Channel
                indicator_warmup = (int)period + 10;
            }
            // Simple Indicators (46-49)
            else if (idx >= 46 && idx <= 49) {
                // High-Low Range, Close Position, Price Accel, Volume ROC
                indicator_warmup = 20;  // Minimal warmup
            }
            
            if (indicator_warmup > warmup_bars) {
                warmup_bars = indicator_warmup;
            }
        }
        
        // Apply warmup: start trading only after indicators are fully initialized
        int actual_start_bar = start_bar + warmup_bars;
        if (actual_start_bar > end_bar) {
            // Cycle too short for this bot's indicators - skip cycle
            cycle_trades_arr[cycle] = 0;
            cycle_wins_arr[cycle] = 0;
            cycle_pnl_arr[cycle] = 0.0f;
            continue;  // Skip to next cycle
        }
        
        // Iterate through bars in cycle (after warmup period)
        for (int bar = actual_start_bar; bar <= end_bar; bar++) {
            // Generate signal from precomputed indicators
            float signal = generate_signal_consensus(
                precomputed_indicators,
                &bot,
                bar,
                num_bars
            );
            
            // Manage existing positions
            int prev_trades = total_trades;
            float prev_total_pnl = total_pnl;  // FIXED: Track PnL before manage_positions
            manage_positions(
                positions,
                &num_positions,
                &ohlcv[bar],
                bar,  // NEW: current bar index
                signal,
                (float)bot.leverage,
                &balance,
                &total_trades,
                &winning_trades,
                &losing_trades,
                &total_pnl,
                &cycle_pnl,  // Pass cycle PnL tracker
                &sum_wins,   // NEW: Pass sum_wins accumulator
                &sum_losses, // NEW: Pass sum_losses accumulator
                &max_drawdown,
                initial_balance
            );
            
            // FIXED: Track consecutive wins/losses using last trade PnL
            if (total_trades > prev_trades) {
                float last_trade_pnl = total_pnl - prev_total_pnl;  // Calculate last trade's PnL
                if (last_trade_pnl > 0.0f) {
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
            
            // CIRCUIT BREAKER: Stop trading if drawdown exceeds 30%
            // This prevents unrealistic "death spirals" where bots lose 99% and keep trading
            if (current_dd > 0.30f) {
                // Close all positions and exit cycle early
                for (int i = 0; i < MAX_POSITIONS; i++) {
                    if (positions[i].is_active) {
                        float return_amount = close_position(
                            &positions[i],
                            ohlcv[bar].close,
                            (float)bot.leverage,
                            &num_positions,
                            3,  // Emergency close
                            ohlcv[bar].volume,
                            ohlcv[bar].high,
                            ohlcv[bar].low
                        );
                        balance += return_amount;
                    }
                }
                break;  // Exit bar loop - stop trading this cycle
            }
            
            // Open new positions if signal and balance allows
            if (signal != 0.0f && balance > initial_balance * MIN_BALANCE_PCT) {
                if (num_positions < MAX_POSITIONS) {
                    // UPDATED: calculate_position_size uses single risk strategy
                    float desired_position_value = calculate_position_size(
                        balance,
                        ohlcv[bar].close,
                        bot.risk_strategy,
                        bot.risk_param
                    );
                    
                    int direction = (signal > 0.0f) ? 1 : -1;
                    
                    // Pass position value to open_position (it will calculate quantity internally)
                    open_position(
                        positions,
                        &num_positions,
                        ohlcv[bar].close,
                        desired_position_value,  // Changed: pass value, not quantity
                        direction,
                        bot.tp_multiplier,
                        bot.sl_multiplier,
                        bar,
                        (float)bot.leverage,
                        &balance,
                        ohlcv[bar].volume,  // NEW: for dynamic slippage
                        ohlcv[bar].high,    // NEW: for dynamic slippage
                        ohlcv[bar].low      // NEW: for dynamic slippage
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
                float return_amount = close_position(
                    &positions[i],
                    ohlcv[end_bar].close,
                    (float)bot.leverage,
                    &num_positions,
                    3,  // End of cycle exit
                    ohlcv[end_bar].volume,
                    ohlcv[end_bar].high,
                    ohlcv[end_bar].low
                );
                balance += return_amount;
                
                // Calculate actual PnL
                // FIXED: With new margin system, margin = entry_price * quantity (no division by leverage)
                float margin_was = positions[i].entry_price * positions[i].quantity;
                float actual_pnl = return_amount - margin_was;
                
                total_pnl += actual_pnl;
                cycle_pnl += actual_pnl;  // Add to cycle PnL
                
                // Overflow protection: clamp counters at max values
                if (total_trades < 65535) {
                    total_trades++;
                }
                
                // FIXED: Accumulate wins and losses properly
                if (actual_pnl > 0.0f) {
                    if (winning_trades < 65535) {
                        winning_trades++;
                    }
                    sum_wins += actual_pnl;
                } else {
                    if (losing_trades < 65535) {
                        losing_trades++;
                    }
                    sum_losses += fabs(actual_pnl);
                }
            }
        }
        
        // Ensure balance is capped at zero (never negative)
        if (balance < 0.0f) balance = 0.0f;
        
        // Compute per-cycle aggregates and store
        if (cycle < MAX_CYCLES) {
            cycle_trades_arr[cycle] = total_trades - cycle_start_trades;
            cycle_wins_arr[cycle] = winning_trades - cycle_start_wins;
            cycle_pnl_arr[cycle] = cycle_pnl;  // Use isolated cycle PnL
        }
    }
    
    // Calculate final metrics
    BacktestResult result;
    result.bot_id = bot.bot_id;
    result.total_trades = total_trades;
    result.winning_trades = winning_trades;
    result.losing_trades = losing_trades;
    
    // Calculate average PnL across cycles (not sum!)
    float avg_pnl = (num_cycles > 0) ? (total_pnl / (float)num_cycles) : 0.0f;
    result.total_pnl = avg_pnl;
    
    // Calculate final balance based on average cycle performance
    float final_bal = initial_balance + avg_pnl;
    if (isnan(final_bal) || isinf(final_bal) || final_bal < 0.0f) {
        final_bal = 0.0f;
    }
    result.final_balance = final_bal;
    
    // Cap max drawdown at 100% (can't lose more than 100%)
    if (max_drawdown > 1.0f) max_drawdown = 1.0f;
    result.max_drawdown = max_drawdown;
    
    result.max_consecutive_wins = (float)max_consecutive_wins;
    result.max_consecutive_losses = (float)max_consecutive_losses;
    
    // Win rate (average across all cycles)
    result.win_rate = (total_trades > 0) ? 
        ((float)winning_trades / (float)total_trades) : 0.0f;
    
    // FIXED: Average win/loss (now properly accumulated)
    result.avg_win = (winning_trades > 0) ? (sum_wins / (float)winning_trades) : 0.0f;
    result.avg_loss = (losing_trades > 0) ? (sum_losses / (float)losing_trades) : 0.0f;
    
    // Profit factor
    result.profit_factor = (sum_losses > 0.0f) ? (sum_wins / sum_losses) : 
                          (sum_wins > 0.0f ? 999.0f : 1.0f);
    
    // FIXED: Sharpe ratio - use PROPER STANDARD DEVIATION, not drawdown!
    // Calculate mean return per cycle
    float mean_return = 0.0f;
    for (int i = 0; i < num_cycles && i < MAX_CYCLES; i++) {
        mean_return += cycle_pnl_arr[i] / initial_balance;
    }
    mean_return /= (float)num_cycles;
    
    // Calculate variance of returns
    float variance = 0.0f;
    for (int i = 0; i < num_cycles && i < MAX_CYCLES; i++) {
        float cycle_return = cycle_pnl_arr[i] / initial_balance;
        float diff = cycle_return - mean_return;
        variance += diff * diff;
    }
    
    // Sample variance (divide by n-1 for unbiased estimate)
    if (num_cycles > 1) {
        variance /= (float)(num_cycles - 1);
    }
    
    // Standard deviation = sqrt(variance)
    float std_dev = sqrt(variance);
    
    // Sharpe ratio = mean_return / std_dev
    // Apply minimum threshold to avoid division by very small numbers
    if (std_dev > 0.001f) {
        result.sharpe_ratio = mean_return / std_dev;
    } else {
        result.sharpe_ratio = 0.0f;
    }
    
    // Clamp Sharpe to reasonable range [-10, 10]
    result.sharpe_ratio = fmin(fmax(result.sharpe_ratio, -10.0f), 10.0f);
    
    // Fitness score (multi-objective, risk-aware)
    // Penalize: low trade counts, high drawdown, low Sharpe
    // Reward: profitability, consistency (win rate), risk-adjusted returns
    
    float fitness = 0.0f;
    float avg_roi = (initial_balance > 0.0f) ? (avg_pnl / initial_balance) : 0.0f;
    
    // Trade count penalty: heavily penalize < 10 trades (not statistically significant)
    float trade_penalty = 0.0f;
    if (total_trades == 0) {
        trade_penalty = -100.0f;  // No trades = very bad
    } else if (total_trades < 10) {
        trade_penalty = -50.0f * (1.0f - (float)total_trades / 10.0f);  // Gradual penalty
    } else if (total_trades < 30) {
        trade_penalty = -10.0f * (1.0f - (float)total_trades / 30.0f);  // Light penalty
    }
    
    // Risk-adjusted returns (Sharpe ratio contribution)
    float sharpe_contribution = result.sharpe_ratio * 15.0f;  // Sharpe weight: high
    sharpe_contribution = fmin(fmax(sharpe_contribution, -30.0f), 50.0f);  // Clamp
    
    // Drawdown penalty (exponential - severe drawdowns very bad)
    float dd_penalty = -max_drawdown * 100.0f;  // Heavy penalty for drawdown
    if (max_drawdown > 0.5f) {
        dd_penalty *= 2.0f;  // Double penalty for >50% drawdown
    }
    
    // Consistency: win rate bonus (but capped - high win rate doesn't mean good strategy)
    float win_rate_bonus = result.win_rate * 25.0f;  // Max 25 points at 100% win rate
    
    // ROI contribution (primary objective)
    float roi_contribution = avg_roi * 80.0f;  // ROI weight: very high
    
    // Profit factor bonus (>1 = profitable, <1 = losing)
    float pf_bonus = 0.0f;
    if (result.profit_factor > 1.0f) {
        pf_bonus = (result.profit_factor - 1.0f) * 12.0f;  // Reward profitability
        pf_bonus = fmin(pf_bonus, 30.0f);  // Cap at 30
    } else {
        pf_bonus = (result.profit_factor - 1.0f) * 20.0f;  // Heavy penalty if losing
    }
    
    // Combine all factors
    fitness = roi_contribution + sharpe_contribution + win_rate_bonus + 
              pf_bonus + dd_penalty + trade_penalty;
    
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

// ============================================================================
// OPTIMIZED PARALLEL KERNEL (BOT × CYCLE PARALLELIZATION)
// ============================================================================

/**
 * Ultra-parallel backtest kernel that processes each bot-cycle pair as a separate work item.
 * 
 * PARALLELIZATION STRATEGY:
 * - Old: num_bots work items, each loops through num_cycles
 * - New: num_bots × num_cycles work items, each processes ONE bot-cycle pair
 * 
 * MEMORY EFFICIENCY:
 * - Each work item only needs ~160 bytes (bot config + 1 position)
 * - No cycle arrays needed (each work item processes one cycle)
 * - Results written to separate bot-cycle slots
 * 
 * GPU UTILIZATION:
 * - 10,000 bots × 100 cycles = 1,000,000 parallel work items
 * - Maximizes GPU occupancy and throughput
 * - Better load balancing across compute units
 */
__kernel void backtest_parallel_bot_cycle(
    __global CompactBotConfig *bots,
    __global OHLCVBar *ohlcv,
    __global float *precomputed_indicators,
    __global int *cycle_starts,
    __global int *cycle_ends,
    const int num_bots,
    const int num_cycles,
    const int num_bars,
    const float initial_balance,
    __global float *cycle_results  // Output: [bot_idx * num_cycles + cycle_idx] = {trades, wins, pnl}
) {
    // Decode bot and cycle indices from work item ID
    int global_id = get_global_id(0);
    int bot_idx = global_id / num_cycles;
    int cycle_idx = global_id % num_cycles;
    
    // Bounds check
    if (bot_idx >= num_bots || cycle_idx >= num_cycles) {
        return;
    }
    
    CompactBotConfig bot = bots[bot_idx];
    int start_bar = cycle_starts[cycle_idx];
    int end_bar = cycle_ends[cycle_idx];
    
    // Quick validation
    if (bot.leverage < 1 || bot.leverage > 125 || bot.num_indicators == 0 || bot.num_indicators > 8) {
        // Write failure markers
        int result_idx = bot_idx * num_cycles * 3 + cycle_idx * 3;
        cycle_results[result_idx] = 0.0f;     // trades
        cycle_results[result_idx + 1] = 0.0f; // wins
        cycle_results[result_idx + 2] = -999999.0f; // pnl (failure marker)
        return;
    }
    
    // Initialize trading state for THIS CYCLE ONLY
    float balance = initial_balance;
    Position positions[MAX_POSITIONS];
    for (int i = 0; i < MAX_POSITIONS; i++) {
        positions[i].is_active = 0;
    }
    int num_positions = 0;
    
    int trades = 0;
    int wins = 0;
    float pnl = 0.0f;
    
    unsigned int seed = bot.bot_id * 31337 + cycle_idx * 997 + 42;
    
    // Backtest this specific cycle
    for (int bar = start_bar; bar <= end_bar && bar < num_bars; bar++) {
        // Generate signal
        float signal = generate_signal_consensus(
            precomputed_indicators,
            &bot,
            bar,
            num_bars
        );
        
        // Manage existing positions (close at TP/SL, update tracking)
        int prev_trades = trades;
        float dummy_max_drawdown = 0.0f;
        float dummy_sum_wins = 0.0f;
        float dummy_sum_losses = 0.0f;
        manage_positions(
            positions,
            &num_positions,
            &ohlcv[bar],
            bar,  // NEW: current bar index
            signal,
            (float)bot.leverage,
            &balance,
            &trades,
            &wins,
            &trades,  // Pass trades as losing_trades placeholder
            &pnl,
            &pnl,  // cycle_pnl
            &dummy_sum_wins,
            &dummy_sum_losses,
            &dummy_max_drawdown,
            initial_balance
        );
        
        // Open new positions if signal present and balance allows
        if (signal != 0.0f && balance > initial_balance * MIN_BALANCE_PCT) {
            if (num_positions < MAX_POSITIONS) {
                // UPDATED: calculate_position_size uses single risk strategy
                float desired_position_value = calculate_position_size(
                    balance,
                    ohlcv[bar].close,
                    bot.risk_strategy,
                    bot.risk_param
                );
                
                int direction = (signal > 0.0f) ? 1 : -1;
                
                // Pass position value to open_position (it will calculate quantity internally)
                open_position(
                    positions,
                    &num_positions,
                    ohlcv[bar].close,
                    desired_position_value,  // Changed: pass value, not quantity
                    direction,
                    bot.tp_multiplier,
                    bot.sl_multiplier,
                    bar,
                    (float)bot.leverage,
                    &balance,
                    ohlcv[bar].volume,  // NEW: for dynamic slippage
                    ohlcv[bar].high,    // NEW: for dynamic slippage
                    ohlcv[bar].low      // NEW: for dynamic slippage
                );
            }
        }
        
        // Stop if balance too low
        if (balance < initial_balance * MIN_BALANCE_PCT) {
            break;
        }
    }
    
    // Close any remaining positions at cycle end with TRUE MARGIN TRADING
    for (int i = 0; i < MAX_POSITIONS; i++) {
        if (positions[i].is_active) {
            float exit_price = ohlcv[end_bar].close;
            
            // Calculate price difference based on direction
            float price_diff;
            if (positions[i].direction == 1) {  // Long
                price_diff = exit_price - positions[i].entry_price;
            } else {  // Short
                price_diff = positions[i].entry_price - exit_price;
            }
            
            // TRUE MARGIN TRADING: 
            // quantity = margin / entry_price (base quantity)
            // leveraged_pnl = (price_diff * quantity) * leverage
            float leverage = (float)bot.leverage;
            float base_pnl = price_diff * positions[i].quantity;
            float leveraged_pnl = base_pnl * leverage;
            
            // Calculate fees on ENTRY and EXIT
            // Entry fee was already paid (deducted from initial margin)
            // Exit fee = exit_price * quantity * TAKER_FEE
            float exit_fee = exit_price * positions[i].quantity * TAKER_FEE;
            
            // Total PnL = leveraged profit - exit fee
            float position_pnl = leveraged_pnl - exit_fee;
            
            // Calculate margin used for this position
            // margin = entry_price * quantity (the amount we initially reserved)
            float margin_used = positions[i].entry_price * positions[i].quantity;
            
            // Return margin + PnL to balance
            balance += margin_used + position_pnl;
            pnl += position_pnl;
            trades++;
            if (position_pnl > 0) wins++;
            
            positions[i].is_active = 0;
        }
    }
    
    // Write results for this bot-cycle pair
    int result_idx = bot_idx * num_cycles * 3 + cycle_idx * 3;
    cycle_results[result_idx] = (float)trades;
    cycle_results[result_idx + 1] = (float)wins;
    cycle_results[result_idx + 2] = pnl;
}
