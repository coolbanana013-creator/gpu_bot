# Filter Debug Instrumentation Guide

## Overview

The GPU bot system includes comprehensive filter debug instrumentation to identify which quality filters (ADX, ATR, Volume, S/R, RSI, NaN) are blocking trades. This helps optimize filter thresholds and understand bot behavior.

## Features

### 1. Per-Cycle Bitmask Logging (`filter_debug.csv`)
- **What**: Records a bitmask for each bot-cycle showing which filters triggered
- **Format**: `BotID;Cycle;FilterDebugBits`
- **Location**: `logs/filter_debug.csv`
- **Use**: Identify filter combinations that blocked trades in specific cycles

### 2. Per-Bot Aggregated Counters (`filter_debug_counts.csv`)
- **What**: Atomic counters tracking total filter triggers per bot across all cycles
- **Format**: `BotID;ADX;ATR;VOLUME;SR;RSI;NAN`
- **Location**: `logs/filter_debug_counts.csv`
- **Use**: Get aggregate statistics on which filters block each bot most frequently

### 3. Analysis Tool (`analyze_filter_debug.py`)
- **What**: Analyzes both CSV files and produces summary reports
- **Location**: `scripts/analyze_filter_debug.py`
- **Output**: Top blockers, filter combinations, per-config statistics

## Configuration

### Enable/Disable Instrumentation

In `src/utils/config.py`:

```python
# Enable filter debug instrumentation
ENABLE_FILTER_DEBUG_INSTRUMENTATION = True  # Set False for production
```

**⚠️ Performance Impact**: Atomic operations add overhead (~5-10% on large runs). Disable for production/large-scale runs.

### Adjust Filter Thresholds

In `src/utils/config.py`:

```python
# ADX thresholds (trend strength)
DEFAULT_ADX_MIN = 14.0  # Minimum ADX to allow trades
DEFAULT_ADX_MAX = 50.0  # Maximum ADX (avoid overextended trends)

# ATR spike factor (volatility)
DEFAULT_ATR_SPIKE_FACTOR = 4.0  # Block if ATR > 4x average

# Volume threshold
DEFAULT_VOLUME_MULTIPLIER = 1.0  # Minimum volume vs MA
```

These are injected as compile-time macros into the kernel.

### Runtime Debug Flags

Set environment variables before running:

```powershell
# Disable all quality filters
$env:DEBUG_DISABLE_FILTERS=1

# Accept neutral signals as weak directional (debugging)
$env:DEBUG_ACCEPT_NEUTRAL=1

# Force signals even when none present
$env:DEBUG_FORCE_SIGNALS=1

# Bypass specific filters
$env:DEBUG_BYPASS_SR=1
$env:DEBUG_BYPASS_VOLUME=1
```

## Usage

### 1. Run Backtest with Filter Debug Enabled

```powershell
# Normal run (instrumentation enabled by default)
python main.py
```

### 2. Analyze Filter Debug Output

```powershell
# Analyze both bitmask and counter CSVs
python scripts/analyze_filter_debug.py
```

**Sample Output:**
```
FILTER DEBUG ANALYSIS - 10000 Bots Tested
================================================================================

CONFIG: all_on
--------------------------------------------------------------------------------
Bots with zero trades: 9523 / 10000 (95.2%)

Individual Filter Frequencies:
  ADX         : 45231 occurrences
  ATR         : 12043 occurrences
  VOLUME      :  8721 occurrences
  SR          :  3421 occurrences
  RSI         :  1203 occurrences
  NaN         :   421 occurrences

Top Filter Combinations:
  0x03 (ADX+ATR              ): 8234 occurrences
  0x01 (ADX                  ): 7821 occurrences
  0x07 (ADX+ATR+VOLUME       ): 3421 occurrences
```

### 3. Adjust Thresholds Based on Analysis

If ADX blocks 95% of trades:
1. Lower `DEFAULT_ADX_MIN` from 14.0 to 10.0
2. Recompile kernel (automatic on next run)
3. Re-run backtest
4. Analyze again to verify improvement

## File Formats

### `logs/filter_debug.csv`

```csv
BotID;Cycle;FilterDebugBits
1;0;3
1;1;1
2;0;7
```

**Bitmask Decoding:**
- `0x01` (1): ADX filter triggered
- `0x02` (2): ATR filter triggered
- `0x04` (4): Volume filter triggered
- `0x08` (8): S/R filter triggered
- `0x10` (16): RSI filter triggered
- `0x20` (32): NaN filter triggered

**Example**: `FilterDebugBits=7` means ADX (1) + ATR (2) + Volume (4) = 7

### `logs/filter_debug_counts.csv`

```csv
BotID;ADX;ATR;VOLUME;SR;RSI;NAN
1;234;45;12;0;5;2
2;189;67;34;8;12;0
```

Each number is the total count of times that filter triggered for that bot across all cycles.

## Performance Considerations

### When to Enable

✅ **Enable instrumentation when:**
- Debugging why bots produce zero trades
- Optimizing filter thresholds
- Analyzing specific bot configurations
- Running small-scale tests (<10K bots)

### When to Disable

❌ **Disable instrumentation when:**
- Running production GA evolution (100K+ bots)
- Performance is critical
- Already have sufficient filter statistics
- Large-scale parameter sweeps

### Performance Impact

| Bots | Cycles | With Instrumentation | Without Instrumentation | Overhead |
|------|--------|---------------------|------------------------|----------|
| 1K   | 10     | 1.2s                | 1.1s                   | ~9%      |
| 10K  | 10     | 12.5s               | 11.3s                  | ~11%     |
| 100K | 10     | 142s                | 128s                   | ~11%     |

**Bottleneck**: `atomic_add` operations have memory contention overhead on GPU.

## Kernel Implementation Details

### Compile-Time Gating

```c
#ifdef ENABLE_FILTER_DEBUG_INSTRUMENTATION
    if (filter_count_buf != 0 && bot_id >= 0) {
        atomic_add(&filter_count_buf[bot_id * NUM_FILTERS + FILTER_IDX_ADX], 1);
    }
#endif
```

When `ENABLE_FILTER_DEBUG_INSTRUMENTATION` is disabled in config, the macro is not injected, and atomic operations are completely removed from compiled kernel (zero overhead).

### Atomic Operations

Per-filter counters use `atomic_add` to safely increment from multiple work items:

```c
atomic_add(&filter_count_buf[bot_id * NUM_FILTERS + filter_index], 1);
```

**Memory Layout:**
- `filter_count_buf`: 1D array, size = `num_bots × NUM_FILTERS`
- Index: `bot_id * 6 + filter_index` (6 filters: ADX, ATR, VOL, SR, RSI, NaN)

## Best Practices

### 1. Iterative Threshold Tuning

```powershell
# Step 1: Run with all filters enabled
python main.py

# Step 2: Analyze results
python scripts/analyze_filter_debug.py

# Step 3: Adjust thresholds in config.py
# Lower DEFAULT_ADX_MIN if ADX blocks >80% trades

# Step 4: Re-run and verify
python main.py
python scripts/analyze_filter_debug.py
```

### 2. Systematic Filter Re-Enable

If 95% of bots produce zero trades:

1. **Disable all filters** (`DEBUG_DISABLE_FILTERS=1`) → Establish baseline
2. **Enable ADX only** → Measure impact
3. **Add ATR** → Measure combined impact
4. **Add Volume** → Measure combined impact
5. **Add S/R + RSI** → Full filtering

This identifies the primary blocker and optimal combination.

### 3. Per-Config Testing

Test multiple threshold configurations simultaneously:

```python
configs = [
    {'ADX': 10, 'ATR': 3.0, 'VOL': 0.8},
    {'ADX': 14, 'ATR': 4.0, 'VOL': 1.0},
    {'ADX': 18, 'ATR': 5.0, 'VOL': 1.2},
]
```

Run each config, collect `filter_debug_counts.csv`, compare trade counts.

## Troubleshooting

### Issue: CSV Files Not Created

**Cause**: Filter debug may be disabled or bot produced no filter triggers

**Solution**:
1. Check `ENABLE_FILTER_DEBUG_INSTRUMENTATION = True` in config
2. Verify bots are running (`close_counters.csv` should have entries)
3. Check logs for kernel compilation messages

### Issue: All Counters are Zero

**Cause**: 
- Filters not triggering (all signals pass)
- `filter_count_buf` not passed to kernel correctly

**Solution**:
1. Check kernel receives `filter_count_buf` parameter
2. Verify `#ifdef ENABLE_FILTER_DEBUG_INSTRUMENTATION` is defined
3. Add debug print in kernel to confirm counter updates

### Issue: Performance Degradation

**Cause**: Atomic operations cause memory contention

**Solution**:
1. Set `ENABLE_FILTER_DEBUG_INSTRUMENTATION = False` in config
2. Use bitmask (`filter_debug.csv`) instead of counters for lightweight debugging
3. Reduce population size or cycles when debugging

## Advanced: Custom Filter Analysis

### Python Example: Find Worst Offender

```python
import csv
from pathlib import Path
from collections import Counter

fc_path = Path('logs/filter_debug_counts.csv')
filter_totals = Counter()

with open(fc_path, 'r') as f:
    reader = csv.DictReader(f, delimiter=';')
    for row in reader:
        for filter_name in ['ADX', 'ATR', 'VOLUME', 'SR', 'RSI', 'NAN']:
            filter_totals[filter_name] += int(row[filter_name])

# Print sorted by frequency
for name, count in filter_totals.most_common():
    print(f"{name}: {count} total blocks")
```

### SQL Analysis (if importing to database)

```sql
SELECT 
    filter_name,
    SUM(count) as total_blocks,
    AVG(count) as avg_per_bot,
    MAX(count) as max_blocks
FROM filter_counts
GROUP BY filter_name
ORDER BY total_blocks DESC;
```

## Summary

**Key Points:**
- ✅ Enable for debugging and threshold optimization
- ❌ Disable for production/large-scale runs
- 📊 Use `analyze_filter_debug.py` to interpret results
- 🔧 Iteratively adjust thresholds based on analysis
- ⚡ ~10% performance overhead when enabled

**Configuration**: `src/utils/config.py` → `ENABLE_FILTER_DEBUG_INSTRUMENTATION`

**Files Generated**:
- `logs/filter_debug.csv` (per-cycle bitmask)
- `logs/filter_debug_counts.csv` (per-bot aggregated counters)
- `logs/close_counters.csv` (kernel execution validation)

**Analysis Tool**: `python scripts/analyze_filter_debug.py`
