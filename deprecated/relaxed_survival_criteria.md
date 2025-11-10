# Relaxed Survival Criteria - REMOVED

**Date Removed**: 2024  
**Reason**: Enforcing strict quality standards - only bots where ALL cycles are profitable should survive

## Problem

The original implementation had fallback criteria that would relax standards when no bots met strict requirements:

1. **Strict Criteria** (PRIMARY): ALL cycles must have trades AND positive profit percentage
2. **Relaxed Criteria** (FALLBACK): Overall positive total PnL across all cycles
3. **Last Resort** (FALLBACK): Keep the most profitable bot even if negative

This led to contradictory logging:
- "No bots passed all-cycles-profitable criteria" 
- Then: "1953 bots passed (ALL cycles profitable)" ← MISLEADING

The second message was incorrect - those bots passed relaxed criteria, not strict criteria.

## Original Code (Lines 418-433)

```python
# If no bots passed criteria, relax to just overall positive total PnL
if not profitable_pairs:
    log_warning(f"SURVIVAL FILTER: {eliminated_no_trades} cycles with no trades, {eliminated_negative_cycle} cycles with negative profit")
    log_warning("No bots passed all-cycles-profitable criteria, relaxing to overall positive PnL")
    for bot, result in zip(population, results):
        profit_pct = (result.total_pnl / initial_balance) * 100
        if profit_pct > 0:
            profitable_pairs.append((bot, result))

# If still no profitable bots, keep the most profitable
if not profitable_pairs:
    all_pairs = list(zip(population, results))
    all_pairs.sort(key=lambda x: x[1].total_pnl, reverse=True)
    profitable_pairs = [all_pairs[0]]
    log_warning("No bots with positive profit, keeping most profitable")
else:
    if used_strict_criteria:
        log_info(f"SURVIVAL FILTER: {eliminated_no_trades} cycles with no trades, {eliminated_negative_cycle} cycles with negative profit, {len(profitable_pairs)} bots passed (ALL cycles profitable)")
    else:
        log_info(f"SURVIVAL FILTER: {len(profitable_pairs)} bots passed with relaxed criteria (overall positive PnL only)")
```

## New Approach (Strict Only)

```python
# STRICT CRITERIA ONLY - No relaxation
if not profitable_pairs:
    log_error(f"SURVIVAL FILTER: {eliminated_no_trades} cycles with no trades, {eliminated_negative_cycle} cycles with negative profit, 0 bots passed")
    log_error("No bots met strict criteria: ALL cycles must have trades AND positive profit")
    log_error("Generating completely new population for next generation")
    # Return empty survivors - refill_population will generate all new bots
    return [], []

log_info(f"SURVIVAL FILTER: {eliminated_no_trades} cycles with no trades, {eliminated_negative_cycle} cycles with negative profit, {len(profitable_pairs)} bots passed (ALL cycles profitable)")
```

## Benefits of Strict-Only Approach

1. **Clear Quality Standards**: No ambiguity - either ALL cycles are profitable or bot is eliminated
2. **Honest Logging**: No misleading messages about which criteria was used
3. **Fresh Gene Pool**: Generating new population when none pass encourages exploration
4. **Prevents Degradation**: Avoids keeping marginally profitable bots that would pollute gene pool

## Edge Case Handling

When no bots pass strict criteria:
- Log detailed statistics (how many failed each check)
- Return empty survivor list
- `refill_population()` generates completely new unique combinations
- Evolution continues with fresh gene pool rather than degraded solutions

## Rationale

The goal is evolution of consistently profitable bots across ALL market cycles, not bots that are "mostly profitable" or "profitable overall but lose money in some cycles". By enforcing strict criteria:
- Only truly robust strategies survive
- Bad strategies are eliminated quickly
- Fresh combinations get tested regularly
- Final results are higher quality

If the indicator pool or parameters aren't capable of producing profitable strategies, better to fail cleanly than silently accept mediocre results.
