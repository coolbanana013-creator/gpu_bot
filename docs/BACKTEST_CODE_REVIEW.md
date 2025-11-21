# Backtest & GA Code Review: Findings and Fixes

## Summary
This review focused on the backtesting pipeline, GPU kernels, logging, and the GA selection logic — especially around unrealistic profit outputs (e.g., 1k+% profits) and trade/win metrics in the log `logs/generation_0.csv`.

Key findings:
- Per-cycle isolation not properly represented in summary/selection: cumulative PnL across independent cycles was being converted to percent by dividing by the initial balance. This exaggerates returns (e.g., 10 cycles with 10% PnL each appears as 100% / cycle × 10 cycles = 1000% cumulative percent), and misleads selection and logging.
- CSV logging reported `TotalProfitPct` which actually represented sum-of-per-cycle profits as a percent of initial balance (cumulative), not average per cycle.
- `select_survivors()` used cumulative PnL percent to decide if bots pass survival thresholds rather than per-cycle averages (even though comments implied average per-cycle evaluation) — we fixed that to use per-cycle average profit %.
- Introduced a new selection option `prefer_win_rate` to prioritize win rate in survivor ranking, enabling the GA to retain high win-rate bots even if their cumulative profit is smaller.
- GPU logging header had ambiguous fields. We've renamed the header fields to clarify "average per-cycle profit percentage" and added clarity for per-cycle averages.
- Added new tests that validate these changes.

## Fixes Implemented
1. Selection updated to use average per-cycle profit percent:
   - File: `src/ga/evolver_compact.py`
   - The survivor selection now computes `avg_profit_pct` as average per-cycle profit percent:
     `avg_profit_pct = (sum(result.per_cycle_pnl) / num_cycles) / initial_balance * 100`.
   - This prevents inflated cumulative percent metrics and makes survival filtering fairer across isolated cycles.

2. Generation summary printing updated to use per-cycle average:
   - File: `src/ga/evolver_compact.py`
   - `avg_pnl_pct` is now per-cycle average percent in the generation summary output.

3. GPU logging CSV changed to report per-cycle average profit percent:
   - File: `src/ga/gpu_logging_processor.py`
   - `TotalProfitPct` has been renamed to `AvgProfitPctPerCycle`. The per-active cycle average has a clarified header `AvgProfitPctPerActiveCycle`.
   - CSV formatting uses per-cycle averages now to reduce confusion.

4. Add optional high-win-rate selection:
   - `select_survivors` now accepts `prefer_win_rate` boolean parameter to guide ranking and prioritization of candidates.
   - The selection scoring method uses both `avg_profit_pct`, `win_rate`, and `max_drawdown` in the score to select better high-win-rate bots.
   - Also added `prefer_win_rate` parameter to `run_evolution` so the GA can propagate preferred selection during runs.

5. Tests
   - New tests added to verify the selection logic and generation summary formatting:
     - `tests/test_survivor_selection_per_cycle_avg.py` tests per-cycle average selection and `prefer_win_rate` selection behavior.
     - `tests/test_generation_summary_print.py` asserts the generation summary prints per-cycle average percent rather than cumulative percent.

## Next Steps & Recommendations (High Level)
- Validate & tune selection weights: the `prefer_win_rate` scoring uses a specific weight for `win_rate` (2.0×) and drawdown penalty (100×). These weights may need tuning based on the user's strategy and data.
- Consider adding a separate scorer for top-bot ranking used by `save_top_bots`: e.g., a configurable scoring formula combining PnL, Win Rate, Drawdown, and Sharpe.
- Add a `--prefer-winrate` CLI flag in `main.py` or environment variable to toggle `prefer_win_rate` during runs.
- Additional kernel/CPU checks for PnL correctness:
  - Check sign and unit consistency for `position.quantity` vs `position.notional` in `open_position` and `close_position`.
  - Audit `sum_wins`/`sum_losses` accumulation. Ensure per-cycle PnL uses `actual_pnl` (not margin) and we don't accidentally include margin or reserved funds.
- Add guardrails in the GA to prevent runaway leverage strategies by strict constraints (e.g., `max_leverage=50` by default) and per-cycle PnL clamping as a safety threshold if necessary.

## Targets for Top Bots (80–100% win rate)
Achieving a win rate of 80–100% for top bots is ambitious, but possible if:
1. We bias selection toward high `win_rate` using the new `prefer_win_rate` flag.
2. We enforce strict per-cycle profitability and low drawdown thresholds (e.g., `max_drawdown < 15%`, `profitable_pct > 80%`).
3. We tune risk settings: favor strategies with conservative position sizing and stop-loss management (e.g., `FixedPct`, `ATRMultiplier` with tight stops).
4. Consider filtering indicators used in top bots to rely on mean reversion and trend confirmation approaches that historically yield high win rates (though PnL may be small per trade).

## Summary of Expected Impacts
- The reported `Avg profit %` numbers will be less inflated (no longer summing across cycles) and will align with intuitive single-cycle interpretations.
- Survivor selection should now avoid overvaluing bots with huge cumulative profit across many isolated cycles; instead we measure consistent single-cycle performance.
- With `prefer_win_rate`, you can tune the GA to identify bots with higher win rate — a necessary step to reach the 80–100% WR target.

## Follow-up Work Items
1. Add a CLI toggle for `prefer_win_rate` and adjustable selection weights.
2. Add additional unit tests for kernel PnL correctness and per-trade log vs aggregated PnL checks.
3. Implement a `scoring_strategy` module that exposes several ranking strategies (profit-first, WR-first, balanced), and a config to choose one.
4. Add monitoring/test harness for top-bot stability over additional real datasets.

---

If you'd like, I can:
- Add a `--prefer-winrate` CLI option to `main.py` and update `scripts/run_ga_full.py`.
- Tune scoring weights and re-run a small test to find top 10 bots with high WR and review their logs.
- Add a new test that asserts top N bots have WR >= 80% in a small run.

Which of those would you prefer next? 
