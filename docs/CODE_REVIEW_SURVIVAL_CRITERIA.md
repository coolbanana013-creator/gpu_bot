# Code Review: Survival Criteria & Consensus Changes

Summary of changes:

- Moved the `check_signal_quality` call behind consensus computation in `src/gpu_kernels/backtest_with_precomputed.cl`. Quality filters are applied only if a directional consensus exists, preventing neutral-only indicator sets from being filtered out premptively.
- Adjusted survival selection thresholds in `src/ga/evolver_compact.py` to require:
  - Average profit per cycle > 0% (strict)
  - Max drawdown across cycles <= 15% (0.15)
  - The per-cycle profitability threshold remains generation-aware (0.40 early / 0.70 later).
- Added additional CSV columns and summary outputs:
  - `TotalProfitPct`, `AvgProfitPctPerCycle`, `TotalTrades`, `AvgTradesPerCycle`, `AvgWinRate`, plus median & mean display.
- Scoring bonuses added:
  - `winrate_bonus`: Scaled bonus for win_rate > 50% (0.5 * (win_rate - 50)).
  - `all_positive_bonus`: +25 points if all cycles had positive P&L.

Files modified:
- `src/gpu_kernels/backtest_with_precomputed.cl` (consensus / filter ordering)
- `src/ga/evolver_compact.py` (selection thresholds, scoring bonuses, CSV & summary changes)
- `src/ga/gpu_logging_processor.py` (adjusted header formatting; ensured averages included)

Review checklist:
1. Runtime correctness
   - Verify the new order of `check_signal_quality` in the kernel does not allow low-quality signals through in any edge cases.
   - Validate that neutral consensus modes and debug flags behave as expected.
2. Selection criteria and fairness
   - Confirm the thresholds (avg > 0% and max DD <= 15%) are correct and conservative enough for your strategy/timeframe.
   - Test in a staging environment: run GA with and without filters, compare survival size & metrics.
3. Scoring and bonuses
   - Ensure the added winrate and all-positive bonuses do not overweight outliers.
   - Validate that the scoring normalization is appropriate across generations.
4. Logging and CSV format consistency
   - Ensure CSV headers align between GPU and CPU logging formats.
   - Confirm CSV imports and downstream analytics (if any) continue to work with new columns.
5. Performance
   - Ensure the consensus and filter ordering change does not cause significant performance regressions in the kernel.
6. Security/Robustness
   - Confirm environment flags and runtime debug toggles behave safely in production.

Suggested follow-up tasks:
- Add per-filter counters into kernels to more accurately track how often filters block signals (not just bitmasks per cycle).
- Add unit tests for the `select_survivors()` method to verify behavior across edge cases.
- Add configuration flags for switching between mean/median selection scoring.

Notes:
- The updated selection logic uses `result.max_drawdown` (global across cycles) to enforce the 15% threshold. Per-cycle drawdowns cannot be derived from the current aggregated result; if per-cycle intra-cycle drawdown is needed, the kernel must store per-bar equity per cycle or compute per-cycle drawdown separately.
