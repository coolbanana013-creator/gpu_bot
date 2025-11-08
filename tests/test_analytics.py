"""
Unit tests for analytics module.
"""
import pytest
import numpy as np
from unittest.mock import Mock

from src.analytics.analytics import BacktestAnalytics
from src.backtester.compact_simulator import BacktestResult


class TestBacktestAnalytics:
    """Test cases for BacktestAnalytics class."""

    def test_compute_cycle_analytics_empty_results(self):
        """Test cycle analytics with empty results."""
        result = BacktestAnalytics.compute_cycle_analytics([], 5)
        assert result == {}

    def test_compute_cycle_analytics_single_bot(self):
        """Test cycle analytics with single bot result."""
        # Create mock BacktestResult
        mock_result = Mock(spec=BacktestResult)
        mock_result.per_cycle_trades = [10, 5, 0, 0, 0]
        mock_result.per_cycle_wins = [7, 3, 0, 0, 0]
        mock_result.per_cycle_pnl = [100.0, 50.0, 0.0, 0.0, 0.0]

        results = [mock_result]
        analytics = BacktestAnalytics.compute_cycle_analytics(results, 5)

        assert analytics['total_bots'] == 1
        assert len(analytics['cycle_analytics']) == 5

        # Check first cycle
        cycle_0 = analytics['cycle_analytics'][0]
        assert cycle_0['total_trades'] == 10
        assert cycle_0['total_wins'] == 7
        assert cycle_0['total_pnl'] == 100.0
        assert cycle_0['win_rate'] == 0.7
        assert cycle_0['avg_pnl_per_trade'] == 10.0
        assert cycle_0['bots_with_trades'] == 1

        # Check aggregate
        agg = analytics['aggregate']
        assert agg['total_trades_all_cycles'] == 15
        assert agg['total_wins_all_cycles'] == 10
        assert agg['total_pnl_all_cycles'] == 150.0
        assert agg['overall_win_rate'] == 10/15
        assert agg['cycles_with_activity'] == 2

    def test_compute_cycle_analytics_multiple_bots(self):
        """Test cycle analytics with multiple bots."""
        # Create mock results
        results = []
        for i in range(3):
            mock_result = Mock(spec=BacktestResult)
            mock_result.per_cycle_trades = [10 + i, 5 + i, 0, 0, 0]
            mock_result.per_cycle_wins = [7 + i, 3 + i, 0, 0, 0]
            mock_result.per_cycle_pnl = [100.0 + i*10, 50.0 + i*5, 0.0, 0.0, 0.0]
            results.append(mock_result)

        analytics = BacktestAnalytics.compute_cycle_analytics(results, 5)

        assert analytics['total_bots'] == 3
        assert len(analytics['cycle_analytics']) == 5

        # Check first cycle aggregates
        cycle_0 = analytics['cycle_analytics'][0]
        assert cycle_0['total_trades'] == 10 + 11 + 12  # 33
        assert cycle_0['total_wins'] == 7 + 8 + 9  # 24
        assert cycle_0['total_pnl'] == 100 + 110 + 120  # 330
        assert cycle_0['win_rate'] == 24/33
        assert cycle_0['bots_with_trades'] == 3

    def test_compute_bot_analytics_empty(self):
        """Test bot analytics with empty results."""
        result = BacktestAnalytics.compute_bot_analytics([])
        assert result == {}

    def test_compute_bot_analytics_single_bot(self):
        """Test bot analytics with single bot."""
        mock_result = Mock(spec=BacktestResult)
        mock_result.bot_id = 123
        mock_result.total_trades = 100
        mock_result.win_rate = 0.65
        mock_result.total_pnl = 500.0
        mock_result.sharpe_ratio = 1.2
        mock_result.max_drawdown = 0.15
        mock_result.final_balance = 10500.0
        mock_result.per_cycle_trades = [20, 30, 25, 15, 10]
        mock_result.per_cycle_wins = [15, 20, 15, 10, 5]
        mock_result.per_cycle_pnl = [50.0, 100.0, 75.0, 25.0, -50.0]

        results = [mock_result]
        analytics = BacktestAnalytics.compute_bot_analytics(results)

        assert len(analytics['bot_stats']) == 1
        bot_stat = analytics['bot_stats'][0]
        assert bot_stat['bot_id'] == 123
        assert bot_stat['total_trades'] == 100
        assert bot_stat['win_rate'] == 0.65
        assert bot_stat['cycles_active'] == 5
        assert bot_stat['best_cycle']['cycle'] == 1  # cycle with 100.0 pnl
        assert bot_stat['worst_cycle']['cycle'] == 4  # cycle with -50.0 pnl

        summary = analytics['summary']
        assert summary['avg_win_rate'] == 0.65
        assert summary['best_win_rate'] == 0.65
        assert summary['avg_total_pnl'] == 500.0

    def test_generate_analytics_report(self):
        """Test analytics report generation."""
        mock_result = Mock(spec=BacktestResult)
        mock_result.bot_id = 123
        mock_result.total_trades = 100
        mock_result.win_rate = 0.65
        mock_result.total_pnl = 500.0
        mock_result.sharpe_ratio = 1.2
        mock_result.max_drawdown = 0.15
        mock_result.final_balance = 10500.0
        mock_result.per_cycle_trades = [20, 30, 25, 15, 10]
        mock_result.per_cycle_wins = [15, 20, 15, 10, 5]
        mock_result.per_cycle_pnl = [50.0, 100.0, 75.0, 25.0, -50.0]

        results = [mock_result]
        report = BacktestAnalytics.generate_analytics_report(results, 5)

        assert "BACKTEST ANALYTICS REPORT" in report
        assert "Total Bots Analyzed: 1" in report
        assert "Cycles Analyzed: 5" in report
        assert "AGGREGATE STATISTICS:" in report
        assert "BOT PERFORMANCE SUMMARY:" in report

    def test_cycle_analytics_bounds_checking(self):
        """Test that cycle analytics handles bounds correctly."""
        mock_result = Mock(spec=BacktestResult)
        # Only 3 cycles of data but requesting 5 cycles analysis
        mock_result.per_cycle_trades = [10, 5, 2]
        mock_result.per_cycle_wins = [7, 3, 1]
        mock_result.per_cycle_pnl = [100.0, 50.0, 20.0]

        results = [mock_result]
        analytics = BacktestAnalytics.compute_cycle_analytics(results, 5)

        assert len(analytics['cycle_analytics']) == 5
        # Cycles 3 and 4 should have zero activity
        assert analytics['cycle_analytics'][3]['total_trades'] == 0
        assert analytics['cycle_analytics'][4]['total_trades'] == 0