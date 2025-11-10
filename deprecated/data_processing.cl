/**
 * GPU-Accelerated Data Processing Kernel
 *
 * Performs fast data validation, cleaning, and preprocessing on GPU.
 * Handles large datasets efficiently with parallel processing.
 */

#define WORK_GROUP_SIZE 256

/**
 * Validate OHLCV data integrity on GPU
 * Checks for invalid prices, volumes, and data consistency
 */
__kernel void validate_ohlcv_data(
    __global float *ohlcv_data,    // [N, 5] array: timestamp, open, high, low, close, volume
    __global int *valid_flags,     // [N] output: 1=valid, 0=invalid
    const int num_candles,
    const float min_price,
    const float max_price,
    const float min_volume
) {
    int gid = get_global_id(0);
    if (gid >= num_candles) return;

    // Get OHLCV values for this candle
    int idx = gid * 5;
    float timestamp = ohlcv_data[idx];
    float open = ohlcv_data[idx + 1];
    float high = ohlcv_data[idx + 2];
    float low = ohlcv_data[idx + 3];
    float close = ohlcv_data[idx + 4];
    float volume = ohlcv_data[idx + 5];

    // Validate data
    bool is_valid = true;

    // Check for NaN or infinite values
    if (isnan(open) || isnan(high) || isnan(low) || isnan(close) || isnan(volume) ||
        isinf(open) || isinf(high) || isinf(low) || isinf(close) || isinf(volume)) {
        is_valid = false;
    }

    // Check price ranges
    if (open <= 0 || high <= 0 || low <= 0 || close <= 0 ||
        open < min_price || high < min_price || low < min_price || close < min_price ||
        open > max_price || high > max_price || low > max_price || close > max_price) {
        is_valid = false;
    }

    // Check OHLC logic: high >= max(open, close), low <= min(open, close)
    if (high < fmax(open, close) || low > fmin(open, close)) {
        is_valid = false;
    }

    // Check volume
    if (volume < min_volume) {
        is_valid = false;
    }

    valid_flags[gid] = is_valid ? 1 : 0;
}

/**
 * Clean and normalize OHLCV data
 * Fixes common data issues and normalizes values
 */
__kernel void clean_ohlcv_data(
    __global float *ohlcv_data,        // [N, 5] input/output
    __global int *valid_flags,         // [N] validity flags
    const int num_candles,
    const float price_epsilon,         // Small value for floating point comparisons
    const float volume_epsilon
) {
    int gid = get_global_id(0);
    if (gid >= num_candles) return;

    // Skip invalid candles
    if (valid_flags[gid] == 0) return;

    int idx = gid * 5;
    float open = ohlcv_data[idx + 1];
    float high = ohlcv_data[idx + 2];
    float low = ohlcv_data[idx + 3];
    float close = ohlcv_data[idx + 4];
    float volume = ohlcv_data[idx + 5];

    // Fix common OHLC issues
    // Ensure high >= max(open, close)
    high = fmax(high, fmax(open, close));

    // Ensure low <= min(open, close)
    low = fmin(low, fmin(open, close));

    // Ensure small positive volume
    volume = fmax(volume, volume_epsilon);

    // Write back cleaned data
    ohlcv_data[idx + 1] = open;
    ohlcv_data[idx + 2] = high;
    ohlcv_data[idx + 3] = low;
    ohlcv_data[idx + 4] = close;
    ohlcv_data[idx + 5] = volume;
}

/**
 * Detect and mark data gaps
 * Identifies missing candles based on timestamp differences
 */
__kernel void detect_data_gaps(
    __global float *timestamps,        // [N] timestamp array
    __global int *gap_flags,          // [N] output: 1=gap detected, 0=no gap
    const int num_candles,
    const float expected_interval_ms, // Expected time between candles
    const float tolerance_multiplier  // How much deviation to allow
) {
    int gid = get_global_id(0);
    if (gid >= num_candles - 1) return; // Skip last candle

    float current_time = timestamps[gid];
    float next_time = timestamps[gid + 1];
    float actual_interval = next_time - current_time;
    float expected_interval = expected_interval_ms;

    // Check if gap is larger than expected (with tolerance)
    float max_allowed_interval = expected_interval * tolerance_multiplier;

    gap_flags[gid] = (actual_interval > max_allowed_interval) ? 1 : 0;
}

/**
 * Interpolate missing data points
 * Fills small gaps using linear interpolation
 */
__kernel void interpolate_gaps(
    __global float *ohlcv_data,        // [N, 5] input/output
    __global int *gap_flags,          // [N] gap flags
    const int num_candles,
    const int max_gap_size            // Maximum gap size to interpolate
) {
    int gid = get_global_id(0);
    if (gid >= num_candles) return;

    // Only process gap start points
    if (gap_flags[gid] == 0) return;

    // Find gap end
    int gap_end = gid + 1;
    int gap_size = 1;

    while (gap_end < num_candles && gap_flags[gap_end - 1] == 1 && gap_size < max_gap_size) {
        gap_end++;
        gap_size++;
    }

    // Only interpolate small gaps
    if (gap_size >= max_gap_size) return;

    // Get values before and after gap
    int before_idx = gid * 5;
    int after_idx = gap_end * 5;

    float before_open = ohlcv_data[before_idx + 1];
    float before_high = ohlcv_data[before_idx + 2];
    float before_low = ohlcv_data[before_idx + 3];
    float before_close = ohlcv_data[before_idx + 4];
    float before_volume = ohlcv_data[before_idx + 5];

    float after_open = ohlcv_data[after_idx + 1];
    float after_high = ohlcv_data[after_idx + 2];
    float after_low = ohlcv_data[after_idx + 3];
    float after_close = ohlcv_data[after_idx + 4];
    float after_volume = ohlcv_data[after_idx + 5];

    // Interpolate values across the gap
    for (int i = 1; i < gap_size; i++) {
        float ratio = (float)i / (float)gap_size;

        int interp_idx = (gid + i) * 5;
        ohlcv_data[interp_idx + 1] = before_open + ratio * (after_open - before_open);
        ohlcv_data[interp_idx + 2] = before_high + ratio * (after_high - before_high);
        ohlcv_data[interp_idx + 3] = before_low + ratio * (after_low - before_low);
        ohlcv_data[interp_idx + 4] = before_close + ratio * (after_close - before_close);
        ohlcv_data[interp_idx + 5] = before_volume + ratio * (after_volume - before_volume);
    }
}

/**
 * Calculate basic statistics for data quality assessment
 */
__kernel void calculate_data_stats(
    __global float *ohlcv_data,        // [N, 5] input
    __global float *stats_output,      // [6] output: mean_return, volatility, volume_mean, gaps, zeros, invalids
    __global int *valid_flags,         // [N] validity flags
    __global int *gap_flags,          // [N] gap flags
    const int num_candles
) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int group_size = get_local_size(0);

    // Shared memory for reduction
    __local float local_returns[WORK_GROUP_SIZE];
    __local float local_volumes[WORK_GROUP_SIZE];
    __local int local_gaps[WORK_GROUP_SIZE];
    __local int local_zeros[WORK_GROUP_SIZE];
    __local int local_invalids[WORK_GROUP_SIZE];

    // Initialize local memory
    local_returns[lid] = 0.0f;
    local_volumes[lid] = 0.0f;
    local_gaps[lid] = 0;
    local_zeros[lid] = 0;
    local_invalids[lid] = 0;

    // Calculate per-thread statistics
    if (gid < num_candles) {
        int idx = gid * 5;
        float close = ohlcv_data[idx + 4];
        float volume = ohlcv_data[idx + 5];

        // Calculate return (skip first candle)
        if (gid > 0) {
            int prev_idx = (gid - 1) * 5;
            float prev_close = ohlcv_data[prev_idx + 4];
            if (prev_close > 0) {
                local_returns[lid] = (close - prev_close) / prev_close;
            }
        }

        local_volumes[lid] = volume;
        local_gaps[lid] = (gid < num_candles - 1) ? gap_flags[gid] : 0;
        local_zeros[lid] = (close == 0.0f || volume == 0.0f) ? 1 : 0;
        local_invalids[lid] = (valid_flags[gid] == 0) ? 1 : 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Parallel reduction
    for (int stride = group_size / 2; stride > 0; stride /= 2) {
        if (lid < stride) {
            local_returns[lid] += local_returns[lid + stride];
            local_volumes[lid] += local_volumes[lid + stride];
            local_gaps[lid] += local_gaps[lid + stride];
            local_zeros[lid] += local_zeros[lid + stride];
            local_invalids[lid] += local_invalids[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write results (only first thread in each work group)
    if (lid == 0) {
        int group_id = get_group_id(0);
        int output_offset = group_id * 6;

        stats_output[output_offset] = local_returns[0] / group_size;     // mean return
        stats_output[output_offset + 1] = 0.0f;                         // volatility (calculated separately)
        stats_output[output_offset + 2] = local_volumes[0] / group_size; // mean volume
        stats_output[output_offset + 3] = (float)local_gaps[0];          // gap count
        stats_output[output_offset + 4] = (float)local_zeros[0];         // zero count
        stats_output[output_offset + 5] = (float)local_invalids[0];      // invalid count
    }
}