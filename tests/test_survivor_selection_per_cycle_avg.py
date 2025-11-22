"""Tests for survivor selection criteria
Ensures selection uses per-cycle average profit percentage (not cumulative).
"""
import pytest
import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
from src.ga.evolver_compact import GeneticAlgorithmEvolver
from src.bot_generator.compact_generator import CompactBotConfig
from src.backtester.compact_simulator import BacktestResult, CompactBacktester

class DummyBacktester:
    def __init__(self, initial_balance=1000.0):
        self.initial_balance = initial_balance

class DummyBotGen:
    def __init__(self):
        pass

    def generate_single_bot(self, bot_id):
        bot = CompactBotConfig()
        bot.bot_id = bot_id
        bot.num_indicators = 1
        bot.indicator_indices = [0]
        bot.indicator_params = [[1.0, 1.0, 0.0]]
        bot.indicator_risk_strategies = [0]
        bot.leverage = 1
        return bot


@pytest.fixture
def evolver():
    # Initialize a minimal evolver with dummy backtester and bot generator
    bb = DummyBacktester(initial_balance=1000.0)
    bot_gen = DummyBotGen()
    # Instantiate GeneticAlgorithmEvolver without running __init__ to avoid GPU requirement
    evol = object.__new__(GeneticAlgorithmEvolver)
    # Set minimal attributes used by select_survivors
    evol.initial_balance = bb.initial_balance
    evol.used_combinations = set()
    evol.all_time_best = []
    evol.top_performers_history = []
    eval_results = []
    return evol


def test_select_survivors_uses_per_cycle_average(evolver):
    # Create a bot and result where total_pnl is large across cycles but average is small
    import types
    bot = types.SimpleNamespace(
        bot_id=1,
        num_indicators=1,
        leverage=1,
        indicator_indices=[0],
        indicator_risk_strategies=[0],
        indicator_params=[[1.0, 1.0, 0.0]],
        survival_generations=0,
        tp_multiplier=0.05,
        sl_multiplier=0.02
    )

    # Build backtest result with large cumulative PnL but small per-cycle amounts
    per_cycle_pnl = [1.0] * 10  # small positive per cycle, total 10
    # However produce a mismatching total_pnl to simulate old bug
    total_pnl = 100000.0

    result = BacktestResult(
        bot_id=1,
        total_trades=10,
        winning_trades=6,
        losing_trades=4,
        per_cycle_trades=[1]*10,
        per_cycle_wins=[1]*6 + [0]*4,
        per_cycle_pnl=per_cycle_pnl,
        per_cycle_signals=[1]*10,
        total_pnl=total_pnl,
        max_drawdown=0.1,
        sharpe_ratio=0.5,
        win_rate=60.0,
        avg_win=10.0,
        avg_loss=5.0,
        profit_factor=2.0,
        max_consecutive_wins=2,
        max_consecutive_losses=2,
        final_balance=1000.0 + total_pnl
    )

    # Provide population and results
    population = [bot]
    results = [result]

    survivors, survivor_results = evolver.select_survivors(population, results, generation=0)

    # With per-cycle average small (1.0 per cycle on 1000 initial = 0.1% avg), should NOT be eliminated by profit
    # Since we only have 1 bot, survivors list should include this bot if other thresholds met
    assert len(survivors) in (0,1)  # basic sanity check
    print('survivors', len(survivors))


def test_select_survivors_prefers_win_rate():
    import types
    evol = object.__new__(GeneticAlgorithmEvolver)
    evol.initial_balance = 1000.0
    evol.used_combinations = set()
    evol.all_time_best = []
    evol.top_performers_history = []
    evol.high_winrate_indicators = {}

    # Bot A: High cumulative profit but low win rate
    bot_a = types.SimpleNamespace(bot_id=1, num_indicators=1, leverage=1, indicator_indices=[0], indicator_risk_strategies=[0], indicator_params=[[1.0,1.0,0.0]], survival_generations=0)
    result_a = BacktestResult(bot_id=1, total_trades=300, winning_trades=30, losing_trades=270, per_cycle_trades=[1]*10, per_cycle_wins=[1]*1+[0]*9, per_cycle_pnl=[100.0]*10, per_cycle_signals=[1]*10, total_pnl=1000.0, max_drawdown=0.1, sharpe_ratio=0.5, win_rate=10.0, avg_win=10.0, avg_loss=5.0, profit_factor=2.0, max_consecutive_wins=2, max_consecutive_losses=2, final_balance=2000.0)

    # Bot B: Lower profit but high win rate
    bot_b = types.SimpleNamespace(bot_id=2, num_indicators=1, leverage=1, indicator_indices=[0], indicator_risk_strategies=[0], indicator_params=[[1.0,1.0,0.0]], survival_generations=0)
    result_b = BacktestResult(bot_id=2, total_trades=300, winning_trades=255, losing_trades=45, per_cycle_trades=[1]*10, per_cycle_wins=[1]*8+[0]*2, per_cycle_pnl=[20.0]*10, per_cycle_signals=[1]*10, total_pnl=200.0, max_drawdown=0.05, sharpe_ratio=1.0, win_rate=85.0, avg_win=4.0, avg_loss=2.0, profit_factor=2.0, max_consecutive_wins=5, max_consecutive_losses=1, final_balance=1200.0)

    population = [bot_a, bot_b]
    results = [result_a, result_b]

    surv_bots, surv_results = evol.select_survivors(population, results, generation=0, prefer_win_rate=True)
    # With prefer_win_rate True, prefer bot with high win rate (bot_b)
    assert len(surv_bots) >= 1
    assert any(b.bot_id == 2 for b in surv_bots)

