/**
 * GPU-Accelerated Genetic Algorithm Operations
 * Kernels for selection, crossover, and mutation operations
 */

// Selection kernel - sorts population by fitness and selects survivors
__kernel void select_survivors_gpu(
    __global const float* fitness_scores,  // Input: fitness scores for each bot
    __global const int* total_pnl,         // Input: total PnL for each bot
    __global int* survivor_indices,        // Output: indices of selected survivors
    const int population_size,
    const int max_survivors,
    const float survival_threshold       // Minimum total_pnl to survive
) {
    int gid = get_global_id(0);

    if (gid >= population_size) return;

    // First pass: mark potential survivors (total_pnl >= threshold)
    __local int local_survivors[256];
    __local int local_count;

    if (get_local_id(0) == 0) {
        local_count = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (total_pnl[gid] >= survival_threshold) {
        int idx = atomic_inc(&local_count);
        if (idx < 256) {
            local_survivors[idx] = gid;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // If no survivors found, keep the best performer
    if (local_count == 0 && gid == 0) {
        // Find bot with highest total_pnl
        int best_idx = 0;
        int best_pnl = total_pnl[0];
        for (int i = 1; i < population_size; i++) {
            if (total_pnl[i] > best_pnl) {
                best_pnl = total_pnl[i];
                best_idx = i;
            }
        }
        survivor_indices[0] = best_idx;
        return;
    }

    // Sort survivors by fitness score (descending)
    if (gid < local_count && gid < max_survivors) {
        int my_idx = local_survivors[gid];
        float my_fitness = fitness_scores[my_idx];

        // Simple bubble sort for survivors
        for (int i = 0; i < local_count - 1; i++) {
            for (int j = i + 1; j < local_count; j++) {
                int idx_i = local_survivors[i];
                int idx_j = local_survivors[j];
                if (fitness_scores[idx_i] < fitness_scores[idx_j]) {
                    // Swap
                    int temp = local_survivors[i];
                    local_survivors[i] = local_survivors[j];
                    local_survivors[j] = temp;
                }
            }
        }

        survivor_indices[gid] = local_survivors[gid];
    }
}

// Crossover kernel - performs uniform crossover between parent pairs
__kernel void uniform_crossover_gpu(
    __global const unsigned char* parent1_indicators,  // Parent 1 indicator indices
    __global const float* parent1_params,              // Parent 1 indicator params (8x3)
    __global const int* parent1_metadata,              // [num_indicators, risk_bitmap, leverage]
    __global const float* parent1_multipliers,         // [tp_multiplier, sl_multiplier]

    __global const unsigned char* parent2_indicators,  // Parent 2 indicator indices
    __global const float* parent2_params,              // Parent 2 indicator params (8x3)
    __global const int* parent2_metadata,              // [num_indicators, risk_bitmap, leverage]
    __global const float* parent2_multipliers,         // [tp_multiplier, sl_multiplier]

    __global unsigned char* child_indicators,          // Output child indicator indices
    __global float* child_params,                      // Output child indicator params (8x3)
    __global int* child_metadata,                      // Output [num_indicators, risk_bitmap, leverage]
    __global float* child_multipliers,                 // Output [tp_multiplier, sl_multiplier]

    __global const unsigned int* random_seeds,         // Random seeds for crossover decisions
    const int num_pairs
) {
    int pair_idx = get_global_id(0);

    if (pair_idx >= num_pairs) return;

    unsigned int seed = random_seeds[pair_idx];

    // Simple LCG random number generator
    unsigned int rand_val = seed;
    #define RAND_NEXT() (rand_val = (rand_val * 1664525u + 1013904223u) & 0xFFFFFFFFu)
    #define RAND_FLOAT() ((float)RAND_NEXT() / (float)0xFFFFFFFFu)

    // Determine number of indicators for child
    int p1_num_ind = parent1_metadata[pair_idx * 3];
    int p2_num_ind = parent2_metadata[pair_idx * 3 + 3];  // Offset for parent2
    int child_num_ind = (RAND_FLOAT() < 0.5f) ? p1_num_ind : p2_num_ind;

    child_metadata[pair_idx * 3] = child_num_ind;

    // Uniform crossover for indicators and parameters
    for (int i = 0; i < child_num_ind; i++) {
        if (RAND_FLOAT() < 0.5f) {
            // Take from parent 1
            if (i < p1_num_ind) {
                child_indicators[pair_idx * 8 + i] = parent1_indicators[pair_idx * 8 + i];
                child_params[pair_idx * 24 + i * 3] = parent1_params[pair_idx * 24 + i * 3];
                child_params[pair_idx * 24 + i * 3 + 1] = parent1_params[pair_idx * 24 + i * 3 + 1];
                child_params[pair_idx * 24 + i * 3 + 2] = parent1_params[pair_idx * 24 + i * 3 + 2];
            } else {
                // Fallback to parent 2
                child_indicators[pair_idx * 8 + i] = parent2_indicators[pair_idx * 8 + i];
                child_params[pair_idx * 24 + i * 3] = parent2_params[pair_idx * 24 + i * 3];
                child_params[pair_idx * 24 + i * 3 + 1] = parent2_params[pair_idx * 24 + i * 3 + 1];
                child_params[pair_idx * 24 + i * 3 + 2] = parent2_params[pair_idx * 24 + i * 3 + 2];
            }
        } else {
            // Take from parent 2
            if (i < p2_num_ind) {
                child_indicators[pair_idx * 8 + i] = parent2_indicators[pair_idx * 8 + i];
                child_params[pair_idx * 24 + i * 3] = parent2_params[pair_idx * 24 + i * 3];
                child_params[pair_idx * 24 + i * 3 + 1] = parent2_params[pair_idx * 24 + i * 3 + 1];
                child_params[pair_idx * 24 + i * 3 + 2] = parent2_params[pair_idx * 24 + i * 3 + 2];
            } else {
                // Fallback to parent 1
                child_indicators[pair_idx * 8 + i] = parent1_indicators[pair_idx * 8 + i];
                child_params[pair_idx * 24 + i * 3] = parent1_params[pair_idx * 24 + i * 3];
                child_params[pair_idx * 24 + i * 3 + 1] = parent1_params[pair_idx * 24 + i * 3 + 1];
                child_params[pair_idx * 24 + i * 3 + 2] = parent1_params[pair_idx * 24 + i * 3 + 2];
            }
        }
    }

    // Risk strategy: randomly select from parent1 or parent2
    if (RAND_FLOAT() < 0.5f) {
        child_metadata[pair_idx * 3 + 1] = parent1_metadata[pair_idx * 3 + 1];
    } else {
        child_metadata[pair_idx * 3 + 1] = parent2_metadata[pair_idx * 3 + 3 + 1];
    }

    // TP/SL multipliers: randomly select
    if (RAND_FLOAT() < 0.5f) {
        child_multipliers[pair_idx * 2] = parent1_multipliers[pair_idx * 2];
        child_multipliers[pair_idx * 2 + 1] = parent1_multipliers[pair_idx * 2 + 1];
    } else {
        child_multipliers[pair_idx * 2] = parent2_multipliers[pair_idx * 2];
        child_multipliers[pair_idx * 2 + 1] = parent2_multipliers[pair_idx * 2 + 1];
    }

    // Leverage: randomly select
    if (RAND_FLOAT() < 0.5f) {
        child_metadata[pair_idx * 3 + 2] = parent1_metadata[pair_idx * 3 + 2];
    } else {
        child_metadata[pair_idx * 3 + 2] = parent2_metadata[pair_idx * 3 + 3 + 2];
    }
}

// Mutation kernel - applies random mutations to bot configurations
__kernel void mutate_population_gpu(
    __global unsigned char* indicators,           // Bot indicator indices (mutable)
    __global float* params,                       // Bot indicator params (mutable) (Nx8x3)
    __global int* metadata,                       // Bot metadata [num_indicators, risk_bitmap, leverage] (mutable)
    __global float* multipliers,                  // Bot multipliers [tp_multiplier, sl_multiplier] (mutable)

    __global const unsigned int* random_seeds,    // Random seeds for mutation decisions
    const int population_size,
    const float mutation_rate,                    // Probability of mutation per bot
    const int num_mutation_types                   // Number of available mutation types
) {
    int bot_idx = get_global_id(0);

    if (bot_idx >= population_size) return;

    unsigned int seed = random_seeds[bot_idx];

    // Simple LCG random number generator
    unsigned int rand_val = seed;
    #define RAND_NEXT() (rand_val = (rand_val * 1664525u + 1013904223u) & 0xFFFFFFFFu)
    #define RAND_FLOAT() ((float)RAND_NEXT() / (float)0xFFFFFFFFu)
    #define RAND_INT(max_val) ((int)(RAND_FLOAT() * (max_val)))

    // Check if this bot should mutate
    if (RAND_FLOAT() >= mutation_rate) return;

    // Select mutation type
    int mutation_type = RAND_INT(num_mutation_types);

    int num_indicators = metadata[bot_idx * 3];

    if (mutation_type == 0) {
        // Mutation 1: Change indicator
        if (num_indicators > 0) {
            int idx = RAND_INT(num_indicators);
            int old_ind = indicators[bot_idx * 8 + idx];
            int new_ind = RAND_INT(256);  // 0-255 indicator range
            indicators[bot_idx * 8 + idx] = (unsigned char)new_ind;
        }

    } else if (mutation_type == 1) {
        // Mutation 2: Adjust parameter
        if (num_indicators > 0) {
            int idx = RAND_INT(num_indicators);
            int param_idx = RAND_INT(3);
            float old_val = params[bot_idx * 24 + idx * 3 + param_idx];
            // Adjust by ±20%
            float factor = 0.8f + RAND_FLOAT() * 0.4f;  // 0.8 to 1.2
            params[bot_idx * 24 + idx * 3 + param_idx] = old_val * factor;
        }

    } else if (mutation_type == 2) {
        // Mutation 3: Flip risk strategy bit
        int bit = RAND_INT(15);  // 0-14 bits
        int old_bitmap = metadata[bot_idx * 3 + 1];
        metadata[bot_idx * 3 + 1] = old_bitmap ^ (1 << bit);

    } else if (mutation_type == 3) {
        // Mutation 4: Adjust TP
        float old_tp = multipliers[bot_idx * 2];
        float factor = 0.9f + RAND_FLOAT() * 0.2f;  // 0.9 to 1.1
        float new_tp = old_tp * factor;
        new_tp = max(0.005f, min(0.25f, new_tp));
        multipliers[bot_idx * 2] = new_tp;

    } else if (mutation_type == 4) {
        // Mutation 5: Adjust SL
        float old_sl = multipliers[bot_idx * 2 + 1];
        float tp = multipliers[bot_idx * 2];
        float factor = 0.9f + RAND_FLOAT() * 0.2f;  // 0.9 to 1.1
        float new_sl = old_sl * factor;
        new_sl = max(0.002f, min(tp/2.0f, new_sl));
        multipliers[bot_idx * 2 + 1] = new_sl;

    } else {  // mutation_type == 5
        // Mutation 6: Adjust leverage
        int old_lev = metadata[bot_idx * 3 + 2];
        int delta = (RAND_INT(2) == 0) ? -1 : 1;  // -1 or +1
        delta *= (1 + RAND_INT(5));  // 1-5
        int new_lev = old_lev + delta;
        new_lev = max(1, min(25, new_lev));
        metadata[bot_idx * 3 + 2] = new_lev;
    }
}

// Fitness evaluation kernel - calculates fitness scores for population
__kernel void calculate_fitness_gpu(
    __global const float* total_pnl,           // Total PnL for each bot
    __global const float* win_rate,            // Win rate for each bot
    __global const float* sharpe_ratio,        // Sharpe ratio for each bot
    __global const float* max_drawdown,        // Max drawdown for each bot
    __global const int* total_trades,          // Total trades for each bot
    __global float* fitness_scores,            // Output fitness scores

    const int population_size,
    const float pnl_weight,
    const float win_rate_weight,
    const float sharpe_weight,
    const float drawdown_weight,
    const float trade_weight
) {
    int gid = get_global_id(0);

    if (gid >= population_size) return;

    // Normalize values (simple min-max scaling per generation)
    float pnl = total_pnl[gid];
    float wr = win_rate[gid];
    float sharpe = sharpe_ratio[gid];
    float dd = max_drawdown[gid];  // Lower is better
    float trades = (float)total_trades[gid];

    // Calculate fitness as weighted sum
    float fitness = pnl_weight * pnl +
                   win_rate_weight * wr +
                   sharpe_weight * sharpe -
                   drawdown_weight * dd +  // Negative because lower drawdown is better
                   trade_weight * trades;

    fitness_scores[gid] = fitness;
}