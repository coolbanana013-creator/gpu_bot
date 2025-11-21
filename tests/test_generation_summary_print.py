"""Tests for generation summary printing
Ensure the summary prints average per-cycle profit percentage rather than cumulative.
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))
from io import StringIO
import contextlib
import types
from src.ga.evolver_compact import GeneticAlgorithmEvolver
from src.backtester.compact_simulator import BacktestResult


def make_bot_and_result(bot_id=1, per_cycle_pnl=None, total_pnl=None, final_balance=None, win_rate=50.0):
    import types
    bot = types.SimpleNamespace(
        bot_id=bot_id,
        num_indicators=1,
        leverage=1,
        indicator_indices=[0],
        indicator_risk_strategies=[0],
        indicator_params=[[1.0, 1.0, 0.0]],
        survival_generations=0,
        tp_multiplier=0.05,
        sl_multiplier=0.02
    )
    if per_cycle_pnl is None:
        per_cycle_pnl = [10.0] * 10
    if total_pnl is None:
        total_pnl = sum(per_cycle_pnl)
    if final_balance is None:
        final_balance = 1000.0 + total_pnl

    result = BacktestResult(
        bot_id=bot_id,
        total_trades=sum(1 for _ in per_cycle_pnl),
        winning_trades=int(len(per_cycle_pnl)*0.6),
        losing_trades=int(len(per_cycle_pnl)*0.4),
        per_cycle_trades=[1]*len(per_cycle_pnl),
        per_cycle_wins=[1]*int(len(per_cycle_pnl)*0.6) + [0]*int(len(per_cycle_pnl)*0.4),
        per_cycle_pnl=per_cycle_pnl,
        per_cycle_signals=[1]*len(per_cycle_pnl),
        total_pnl=total_pnl,
        max_drawdown=0.1,
        sharpe_ratio=0.5,
        win_rate=win_rate,
        avg_win=10.0,
        avg_loss=5.0,
        profit_factor=2.0,
        max_consecutive_wins=2,
        max_consecutive_losses=2,
        final_balance=final_balance
    )
    return bot, result


def test_generation_summary_print_avg_per_cycle():
    evol = object.__new__(GeneticAlgorithmEvolver)
    evol.initial_balance = 1000.0
    evol.used_combinations = set()
    evol.all_time_best = []

    # Create a result with highly inflated total_pnl but small per-cycle average
    per_cycle_pnl = [1.0] * 10
    bot, result = make_bot_and_result(1, per_cycle_pnl, total_pnl=100000.0)

    # Captured stdout
    buf = StringIO()
    with contextlib.redirect_stdout(buf):
        evol._print_generation_summary(0, [result], initial_balance=1000.0, survivor_count=1)
    out = buf.getvalue()
    # Check we printed percent roughly equal to per-cycle average = 1/1000 * 100 = 0.1%
    assert "+0.1%" in out or "+0.0%" in out
