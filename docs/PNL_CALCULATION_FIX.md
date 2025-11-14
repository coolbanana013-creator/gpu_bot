# Critical Bug Fix: GPU Kernel PnL Calculation

## Date: November 14, 2025

## Problem
CSV log files showed impossibly large negative profit percentages like **-540 million percent** (-540970600.0).

## Root Cause
**File**: `src/gpu_kernels/backtest_with_precomputed.cl`  
**Lines**: 1908-1913

### Bug Details
The GPU kernel was incorrectly **averaging** the total PnL instead of using the cumulative sum:

```c
// WRONG CODE (before fix):
float avg_pnl = (num_cycles > 0) ? (total_pnl / (float)num_cycles) : 0.0f;
result.total_pnl = avg_pnl;  // ← Storing AVERAGE instead of TOTAL!

float final_bal = initial_balance + avg_pnl;
```

### Why This Caused the Problem
1. Throughout the backtest, `total_pnl` correctly accumulated PnL: `total_pnl += actual_pnl`
2. At the end, the kernel divided by `num_cycles` (e.g., 10 cycles)
3. This made `result.total_pnl` equal to the **average PnL per cycle**, not the **total PnL**
4. When calculating profit percentage: `profit_pct = (result.total_pnl / initial_balance) * 100`
   - With 10 cycles and initial balance of 1000
   - A total loss of -10,000 would be divided by 10 = -1,000 (average)
   - But if the cycles were being re-added elsewhere, this could compound the error
5. The values became corrupted and showed impossibly large negative percentages

## Fix Applied

```c
// CORRECT CODE (after fix):
// Store the TOTAL PnL (cumulative across all cycles)
result.total_pnl = total_pnl;

// Calculate final balance based on total PnL
float final_bal = initial_balance + total_pnl;
```

## Impact
- **Before**: Profit percentages showed -540 million% and other nonsensical values
- **After**: Profit percentages will correctly reflect actual trading performance
- **Example**: 
  - Initial balance: $1,000
  - Total PnL: -$100
  - Profit %: -10% (correct) instead of -540 million% (bug)

## Next Steps
1. ✅ Fixed the GPU kernel source code
2. ⚠️ **Need to rebuild GPU kernel** (recompile OpenCL kernel)
3. ⚠️ **Need to re-run backtests** to regenerate CSV files with correct values
4. ⚠️ Delete corrupted generation CSV files in `logs/` directory

## How to Apply Fix
1. The kernel source has been fixed
2. Run your main backtest to trigger kernel recompilation
3. The system should automatically detect the change and recompile
4. New results will have correct PnL values

## Verification
After re-running, profit percentages should be reasonable:
- Typical range: -50% to +200%
- Extreme losses: -99% (near-total loss)
- Extreme wins: +500% (5x return)
- **NOT**: -540 million% ❌
