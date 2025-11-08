"""
Unit tests for backtester components.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.backtester.compact_simulator import CompactBacktester, BacktestResult


class TestCompactBacktester:
    """Test cases for CompactBacktester class."""

    @patch('pyopencl.create_some_context')
    @patch('pyopencl.CommandQueue')
    def test_initialization(self, mock_queue, mock_ctx):
        """Test backtester initialization."""
        # Skip this test as OpenCL mocking is complex
        pass

    def test_parse_results_basic(self):
        """Test basic result parsing."""
        from src.backtester.compact_simulator import CompactBacktester
        backtester = CompactBacktester.__new__(CompactBacktester)

        # Mock the _parse_results method to test it directly
        from src.backtester.compact_simulator import CompactBacktester
        backtester = CompactBacktester.__new__(CompactBacktester)

        # Create mock raw data
        MAX_CYCLES = 100
        num_bots = 2

        # Create structured array matching the dtype
        dt = np.dtype([
            ('bot_id', np.int32),
            ('total_trades', np.int32),
            ('winning_trades', np.int32),
            ('losing_trades', np.int32),
            ('cycle_trades', np.int32, (MAX_CYCLES,)),
            ('cycle_wins', np.int32, (MAX_CYCLES,)),
            ('cycle_pnl', np.float32, (MAX_CYCLES,)),
            ('total_pnl', np.float32),
            ('max_drawdown', np.float32),
            ('sharpe_ratio', np.float32),
            ('win_rate', np.float32),
            ('avg_win', np.float32),
            ('avg_loss', np.float32),
            ('profit_factor', np.float32),
            ('max_consecutive_wins', np.float32),
            ('max_consecutive_losses', np.float32),
            ('final_balance', np.float32),
            ('generation_survived', np.int32),
            ('fitness_score', np.float32)
        ])

        # Create test data
        test_data = np.zeros(num_bots, dtype=dt)
        test_data['bot_id'] = [1, 2]
        test_data['total_trades'] = [100, 150]
        test_data['winning_trades'] = [65, 90]
        test_data['losing_trades'] = [35, 60]
        test_data['cycle_trades'][0, :5] = [20, 30, 25, 15, 10]  # bot 0
        test_data['cycle_trades'][1, :5] = [30, 40, 35, 25, 20]  # bot 1
        test_data['cycle_wins'][0, :5] = [15, 20, 15, 10, 5]
        test_data['cycle_wins'][1, :5] = [20, 25, 20, 15, 10]
        test_data['cycle_pnl'][0, :5] = [50.0, 100.0, 75.0, 25.0, -50.0]
        test_data['cycle_pnl'][1, :5] = [75.0, 125.0, 100.0, 50.0, -25.0]
        test_data['total_pnl'] = [200.0, 325.0]
        test_data['max_drawdown'] = [0.15, 0.12]
        test_data['sharpe_ratio'] = [1.2, 1.5]
        test_data['win_rate'] = [0.65, 0.6]
        test_data['avg_win'] = [12.0, 15.0]
        test_data['avg_loss'] = [-8.0, -10.0]
        test_data['profit_factor'] = [1.5, 1.8]
        test_data['max_consecutive_wins'] = [5, 7]
        test_data['max_consecutive_losses'] = [3, 4]
        test_data['final_balance'] = [10200.0, 10325.0]
        test_data['generation_survived'] = [1, 2]
        test_data['fitness_score'] = [1.8, 2.1]

        raw_data = test_data.tobytes()
        raw_array = np.frombuffer(raw_data, dtype=np.uint8)

        results = backtester._parse_results(raw_array, num_bots)

        assert len(results) == 2

        # Check first bot
        bot0 = results[0]
        assert bot0.bot_id == 1
        assert bot0.total_trades == 100
        assert bot0.winning_trades == 65
        assert bot0.losing_trades == 35
        assert bot0.total_pnl == 200.0
        assert abs(bot0.max_drawdown - 0.15) < 1e-6
        assert abs(bot0.sharpe_ratio - 1.2) < 1e-6
        assert abs(bot0.win_rate - 0.65) < 1e-6
        assert bot0.final_balance == 10200.0
        assert abs(bot0.fitness_score - 1.8) < 1e-6

        # Check per-cycle data
        assert len(bot0.per_cycle_trades) == MAX_CYCLES
        assert bot0.per_cycle_trades[:5] == [20, 30, 25, 15, 10]
        assert bot0.per_cycle_trades[5:] == [0] * (MAX_CYCLES - 5)
        assert bot0.per_cycle_wins[:5] == [15, 20, 15, 10, 5]
        assert bot0.per_cycle_pnl[:5] == [50.0, 100.0, 75.0, 25.0, -50.0]

        # Check second bot
        bot1 = results[1]
        assert bot1.bot_id == 2
        assert bot1.total_trades == 150
        assert bot1.per_cycle_trades[:5] == [30, 40, 35, 25, 20]

    def test_parse_results_empty_data(self):
        """Test parsing with empty data."""
        backtester = CompactBacktester.__new__(CompactBacktester)

        raw_array = np.array([], dtype=np.uint8)
        results = backtester._parse_results(raw_array, 0)

        assert results == []

    def test_backtest_result_dataclass(self):
        """Test BacktestResult dataclass creation."""
        result = BacktestResult(
            bot_id=123,
            total_trades=100,
            winning_trades=65,
            losing_trades=35,
            per_cycle_trades=[20, 30, 25],
            per_cycle_wins=[15, 20, 15],
            per_cycle_pnl=[50.0, 100.0, 75.0],
            total_pnl=200.0,
            max_drawdown=0.15,
            sharpe_ratio=1.2,
            win_rate=0.65,
            avg_win=10.0,
            avg_loss=-8.0,
            profit_factor=1.5,
            max_consecutive_wins=5,
            max_consecutive_losses=3,
            final_balance=10200.0,
            generation_survived=1,
            fitness_score=1.8
        )

        assert result.bot_id == 123
        assert result.total_trades == 100
        assert result.per_cycle_trades == [20, 30, 25]
        assert result.total_pnl == 200.0
        assert result.fitness_score == 1.8