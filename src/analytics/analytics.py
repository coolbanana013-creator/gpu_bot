"""
Analytics module for computing summary statistics from backtest results.
"""
from typing import List, Dict, Any
import numpy as np
from ..backtester.compact_simulator import BacktestResult


class BacktestAnalytics:
    """
    Analytics class for computing summary statistics from backtest results.
    """

    @staticmethod
    def compute_cycle_analytics(results: List[BacktestResult], num_cycles: int) -> Dict[str, Any]:
        """
        Compute analytics across all cycles for a list of backtest results.

        Args:
            results: List of BacktestResult objects
            num_cycles: Number of cycles to analyze

        Returns:
            Dictionary with cycle-by-cycle and aggregate analytics
        """
        if not results:
            return {}

        analytics = {
            'total_bots': len(results),
            'cycle_analytics': [],
            'aggregate': {}
        }

        # Per-cycle analytics
        for cycle in range(num_cycles):
            cycle_data = {
                'cycle': cycle,
                'total_trades': 0,
                'total_wins': 0,
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'avg_pnl_per_trade': 0.0,
                'profit_pct_distribution': [],
                'bots_with_trades': 0
            }

            for result in results:
                if cycle < len(result.per_cycle_trades):
                    trades = result.per_cycle_trades[cycle]
                    wins = result.per_cycle_wins[cycle]
                    pnl = result.per_cycle_pnl[cycle]

                    cycle_data['total_trades'] += trades
                    cycle_data['total_wins'] += wins
                    cycle_data['total_pnl'] += pnl

                    if trades > 0:
                        cycle_data['bots_with_trades'] += 1
                        win_rate = wins / trades if trades > 0 else 0.0
                        cycle_data['profit_pct_distribution'].append(win_rate * 100)

            # Compute averages
            if cycle_data['total_trades'] > 0:
                cycle_data['win_rate'] = cycle_data['total_wins'] / cycle_data['total_trades']
                cycle_data['avg_pnl_per_trade'] = cycle_data['total_pnl'] / cycle_data['total_trades']

            analytics['cycle_analytics'].append(cycle_data)

        # Aggregate analytics
        total_trades_all = sum(c['total_trades'] for c in analytics['cycle_analytics'])
        total_wins_all = sum(c['total_wins'] for c in analytics['cycle_analytics'])
        total_pnl_all = sum(c['total_pnl'] for c in analytics['cycle_analytics'])

        analytics['aggregate'] = {
            'total_trades_all_cycles': total_trades_all,
            'total_wins_all_cycles': total_wins_all,
            'total_pnl_all_cycles': total_pnl_all,
            'overall_win_rate': total_wins_all / total_trades_all if total_trades_all > 0 else 0.0,
            'avg_pnl_per_trade_all': total_pnl_all / total_trades_all if total_trades_all > 0 else 0.0,
            'cycles_with_activity': sum(1 for c in analytics['cycle_analytics'] if c['total_trades'] > 0),
            'avg_trades_per_cycle': total_trades_all / num_cycles if num_cycles > 0 else 0,
            'avg_win_rate_per_cycle': np.mean([c['win_rate'] for c in analytics['cycle_analytics']]),
            'std_win_rate_per_cycle': np.std([c['win_rate'] for c in analytics['cycle_analytics']])
        }

        return analytics

    @staticmethod
    def compute_bot_analytics(results: List[BacktestResult]) -> Dict[str, Any]:
        """
        Compute analytics for individual bots.

        Args:
            results: List of BacktestResult objects

        Returns:
            Dictionary with bot-level analytics
        """
        if not results:
            return {}

        analytics = {
            'bot_stats': [],
            'summary': {}
        }

        for result in results:
            bot_stat = {
                'bot_id': result.bot_id,
                'total_trades': result.total_trades,
                'win_rate': result.win_rate,
                'total_pnl': result.total_pnl,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'final_balance': result.final_balance,
                'cycles_active': sum(1 for t in result.per_cycle_trades if t > 0),
                'best_cycle': None,
                'worst_cycle': None
            }

            # Find best and worst cycles
            if result.per_cycle_trades:
                cycle_pnls = [(i, pnl) for i, pnl in enumerate(result.per_cycle_pnl) if result.per_cycle_trades[i] > 0]
                if cycle_pnls:
                    best_cycle = max(cycle_pnls, key=lambda x: x[1])
                    worst_cycle = min(cycle_pnls, key=lambda x: x[1])
                    bot_stat['best_cycle'] = {'cycle': best_cycle[0], 'pnl': best_cycle[1]}
                    bot_stat['worst_cycle'] = {'cycle': worst_cycle[0], 'pnl': worst_cycle[1]}

            analytics['bot_stats'].append(bot_stat)

        # Summary statistics
        if analytics['bot_stats']:
            win_rates = [b['win_rate'] for b in analytics['bot_stats']]
            pnls = [b['total_pnl'] for b in analytics['bot_stats']]
            sharpes = [b['sharpe_ratio'] for b in analytics['bot_stats']]
            drawdowns = [b['max_drawdown'] for b in analytics['bot_stats']]

            analytics['summary'] = {
                'avg_win_rate': np.mean(win_rates),
                'std_win_rate': np.std(win_rates),
                'best_win_rate': max(win_rates),
                'worst_win_rate': min(win_rates),
                'avg_total_pnl': np.mean(pnls),
                'std_total_pnl': np.std(pnls),
                'best_total_pnl': max(pnls),
                'worst_total_pnl': min(pnls),
                'avg_sharpe': np.mean(sharpes),
                'std_sharpe': np.std(sharpes),
                'avg_max_drawdown': np.mean(drawdowns),
                'std_max_drawdown': np.std(drawdowns)
            }

        return analytics

    @staticmethod
    def generate_analytics_report(results: List[BacktestResult], num_cycles: int) -> str:
        """
        Generate a formatted analytics report.

        Args:
            results: List of BacktestResult objects
            num_cycles: Number of cycles analyzed

        Returns:
            Formatted string report
        """
        cycle_analytics = BacktestAnalytics.compute_cycle_analytics(results, num_cycles)
        bot_analytics = BacktestAnalytics.compute_bot_analytics(results)

        report = []
        report.append("=" * 80)
        report.append("BACKTEST ANALYTICS REPORT")
        report.append("=" * 80)
        report.append(f"Total Bots Analyzed: {len(results)}")
        report.append(f"Cycles Analyzed: {num_cycles}")
        report.append("")

        # Aggregate stats
        agg = cycle_analytics.get('aggregate', {})
        report.append("AGGREGATE STATISTICS:")
        report.append(f"  Total Trades (All Cycles): {agg.get('total_trades_all_cycles', 0)}")
        report.append(f"  Overall Win Rate: {agg.get('overall_win_rate', 0):.2%}")
        report.append(f"  Average P&L per Trade: ${agg.get('avg_pnl_per_trade_all', 0):.2f}")
        report.append(f"  Cycles with Activity: {agg.get('cycles_with_activity', 0)}")
        report.append(f"  Average Trades per Cycle: {agg.get('avg_trades_per_cycle', 0):.1f}")
        report.append("")

        # Bot summary
        summary = bot_analytics.get('summary', {})
        report.append("BOT PERFORMANCE SUMMARY:")
        report.append(f"  Average Win Rate: {summary.get('avg_win_rate', 0):.2%}")
        report.append(f"  Best Win Rate: {summary.get('best_win_rate', 0):.2%}")
        report.append(f"  Average Total P&L: ${summary.get('avg_total_pnl', 0):.2f}")
        report.append(f"  Best Total P&L: ${summary.get('best_total_pnl', 0):.2f}")
        report.append(f"  Average Sharpe Ratio: {summary.get('avg_sharpe', 0):.2f}")
        report.append("")

        # Top 5 cycles by activity
        cycle_activity = [(i, c['total_trades']) for i, c in enumerate(cycle_analytics.get('cycle_analytics', []))]
        cycle_activity.sort(key=lambda x: x[1], reverse=True)
        report.append("TOP 5 MOST ACTIVE CYCLES:")
        for i, (cycle_idx, trades) in enumerate(cycle_activity[:5]):
            if trades > 0:
                cycle_data = cycle_analytics['cycle_analytics'][cycle_idx]
                report.append(f"  Cycle {cycle_idx}: {trades} trades, {cycle_data['win_rate']:.2%} win rate")
        report.append("")

        report.append("=" * 80)

        return "\n".join(report)