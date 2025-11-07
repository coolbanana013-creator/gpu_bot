"""
Genetic algorithm evolver - selection, elimination, and evolution logic.
"""
import numpy as np
from typing import List, Dict, Tuple
import json

from ..bot_generator.generator import BotConfig, BotGenerator
from ..backtester.simulator import BacktestResult, Backtester
from ..utils.validation import log_info, log_debug
from ..utils.config import TOP_BOTS_COUNT, RESULTS_FILE


class BotPerformance:
    """Track bot performance across generations."""
    
    def __init__(self, bot: BotConfig):
        self.bot = bot
        self.generation_results: Dict[int, Dict[str, float]] = {}
        self.generations_survived = 0
        self.first_generation = None
        self.last_generation = None
    
    def add_generation_result(self, generation: int, metrics: Dict[str, float]) -> None:
        """Add results for a generation."""
        self.generation_results[generation] = metrics
        self.generations_survived = len(self.generation_results)
        
        if self.first_generation is None:
            self.first_generation = generation
        self.last_generation = generation
    
    def get_average_metrics(self) -> Dict[str, float]:
        """Get average metrics across all generations."""
        if not self.generation_results:
            return {
                'avg_profit_pct': 0.0,
                'avg_winrate': 0.0,
                'avg_trades': 0.0
            }
        
        all_profits = [m['avg_profit_pct'] for m in self.generation_results.values()]
        all_winrates = [m['avg_winrate'] for m in self.generation_results.values()]
        all_trades = [m['avg_trades'] for m in self.generation_results.values()]
        
        return {
            'avg_profit_pct': np.mean(all_profits),
            'avg_winrate': np.mean(all_winrates),
            'avg_trades': np.mean(all_trades)
        }
    
    def is_profitable(self) -> bool:
        """Check if bot is profitable on average."""
        avg_metrics = self.get_average_metrics()
        return avg_metrics['avg_profit_pct'] > 0.0


class GeneticAlgorithmEvolver:
    """
    Manages the genetic algorithm evolution process.
    Handles selection, elimination, and population refill.
    """
    
    def __init__(
        self,
        bot_generator: BotGenerator,
        backtester: Backtester
    ):
        """
        Initialize GA evolver.
        
        Args:
            bot_generator: Bot generator instance
            backtester: Backtester instance
        """
        self.bot_generator = bot_generator
        self.backtester = backtester
        self.current_generation = 0
        
        # Track all bot performances
        self.bot_performances: Dict[int, BotPerformance] = {}
        
        log_info("Genetic algorithm evolver initialized")
    
    def initialize_population(self) -> List[BotConfig]:
        """
        Initialize the population for generation 0.
        
        Returns:
            Initial population
        """
        log_info("Initializing population for generation 0...")
        population = self.bot_generator.generate_initial_population()
        
        # Create performance trackers
        for bot in population:
            self.bot_performances[bot.bot_id] = BotPerformance(bot)
        
        self.current_generation = 0
        log_info(f"Population initialized: {len(population)} bots")
        return population
    
    def evaluate_population(
        self,
        population: List[BotConfig],
        ohlcv_data,
        cycle_ranges: List[Tuple[int, int]]
    ) -> Dict[int, Dict[str, float]]:
        """
        Evaluate population performance.
        
        Args:
            population: List of bots to evaluate
            ohlcv_data: OHLCV DataFrame
            cycle_ranges: List of cycle ranges
            
        Returns:
            Dictionary mapping bot_id to average metrics
        """
        log_info(f"Evaluating generation {self.current_generation} ({len(population)} bots)...")
        
        # Backtest all bots
        results = self.backtester.backtest_population(population, ohlcv_data, cycle_ranges)
        
        # Calculate average metrics for each bot
        bot_metrics = {}
        for bot_id, bot_results in results.items():
            avg_metrics = self.backtester.calculate_average_metrics(bot_results)
            bot_metrics[bot_id] = avg_metrics
            
            # Update performance tracker
            if bot_id in self.bot_performances:
                self.bot_performances[bot_id].add_generation_result(
                    self.current_generation,
                    avg_metrics
                )
        
        log_info(f"Evaluation complete for generation {self.current_generation}")
        return bot_metrics
    
    def select_survivors(
        self,
        population: List[BotConfig],
        bot_metrics: Dict[int, Dict[str, float]]
    ) -> List[BotConfig]:
        """
        Select bots that survive to next generation.
        Selection criterion: average profit > 0%
        
        Args:
            population: Current population
            bot_metrics: Performance metrics for each bot
            
        Returns:
            List of surviving bots
        """
        survivors = []
        
        for bot in population:
            metrics = bot_metrics.get(bot.bot_id)
            if metrics and metrics['avg_profit_pct'] > 0.0:
                survivors.append(bot)
        
        survival_rate = (len(survivors) / len(population)) * 100.0
        log_info(
            f"Selection complete: {len(survivors)}/{len(population)} survived "
            f"({survival_rate:.1f}%)"
        )
        
        return survivors
    
    def evolve_generation(
        self,
        ohlcv_data,
        cycle_ranges: List[Tuple[int, int]]
    ) -> Tuple[List[BotConfig], Dict[int, Dict[str, float]]]:
        """
        Run one generation of evolution.
        
        Args:
            ohlcv_data: OHLCV DataFrame
            cycle_ranges: List of cycle ranges
            
        Returns:
            Tuple of (new_population, bot_metrics)
        """
        log_info(f"\n{'='*60}")
        log_info(f"GENERATION {self.current_generation}")
        log_info(f"{'='*60}\n")
        
        # Get current population
        population = self.bot_generator.get_population()
        
        # Evaluate
        bot_metrics = self.evaluate_population(population, ohlcv_data, cycle_ranges)
        
        # Select survivors
        survivors = self.select_survivors(population, bot_metrics)
        
        # Refill population
        new_population = self.bot_generator.refill_population(survivors)
        
        # Add new bots to performance tracker
        for bot in new_population:
            if bot.bot_id not in self.bot_performances:
                self.bot_performances[bot.bot_id] = BotPerformance(bot)
        
        # Increment generation counter
        self.current_generation += 1
        
        return new_population, bot_metrics
    
    def run_evolution(
        self,
        num_generations: int,
        ohlcv_data,
        cycle_ranges: List[Tuple[int, int]]
    ) -> None:
        """
        Run complete evolution process.
        
        Args:
            num_generations: Number of generations to run
            ohlcv_data: OHLCV DataFrame
            cycle_ranges: List of cycle ranges
        """
        log_info(f"\n{'#'*60}")
        log_info(f"STARTING EVOLUTION: {num_generations} GENERATIONS")
        log_info(f"{'#'*60}\n")
        
        # Initialize if needed
        if self.current_generation == 0:
            self.initialize_population()
        
        # Run generations
        for gen in range(num_generations):
            self.evolve_generation(ohlcv_data, cycle_ranges)
            
            # Print summary
            self.print_generation_summary()
        
        log_info(f"\n{'#'*60}")
        log_info(f"EVOLUTION COMPLETE")
        log_info(f"{'#'*60}\n")
    
    def print_generation_summary(self) -> None:
        """Print summary of current generation."""
        # Get current population metrics
        population = self.bot_generator.get_population()
        
        profitable_bots = sum(
            1 for bot_id in [b.bot_id for b in population]
            if bot_id in self.bot_performances and self.bot_performances[bot_id].is_profitable()
        )
        
        log_info(f"\nGeneration {self.current_generation - 1} Summary:")
        log_info(f"  Population: {len(population)} bots")
        log_info(f"  Profitable: {profitable_bots} bots")
        log_info(f"  Total bots tracked: {len(self.bot_performances)}")
    
    def get_top_bots(self, count: int = TOP_BOTS_COUNT) -> List[Dict]:
        """
        Get top performing bots ranked by:
        1. Generations survived (descending)
        2. Average profit % (descending)
        3. Average winrate (descending)
        
        Args:
            count: Number of top bots to return
            
        Returns:
            List of dictionaries with bot info and performance
        """
        # Filter to only profitable bots
        profitable = [
            perf for perf in self.bot_performances.values()
            if perf.is_profitable()
        ]
        
        if not profitable:
            log_info("No profitable bots found")
            return []
        
        # Sort by criteria
        sorted_bots = sorted(
            profitable,
            key=lambda p: (
                -p.generations_survived,  # More generations = better
                -p.get_average_metrics()['avg_profit_pct'],  # Higher profit = better
                -p.get_average_metrics()['avg_winrate']  # Higher winrate = better
            )
        )
        
        # Get top N
        top_bots = sorted_bots[:count]
        
        # Format results
        results = []
        for rank, perf in enumerate(top_bots, 1):
            avg_metrics = perf.get_average_metrics()
            
            results.append({
                'rank': rank,
                'bot_id': perf.bot.bot_id,
                'generations_survived': perf.generations_survived,
                'first_generation': perf.first_generation,
                'last_generation': perf.last_generation,
                'avg_profit_pct': round(avg_metrics['avg_profit_pct'], 2),
                'avg_winrate': round(avg_metrics['avg_winrate'], 4),
                'avg_trades': round(avg_metrics['avg_trades'], 1),
                'config': perf.bot.to_dict()
            })
        
        return results
    
    def save_results(self, filepath: str = RESULTS_FILE) -> None:
        """
        Save top bots to JSON file.
        
        Args:
            filepath: Path to save results
        """
        top_bots = self.get_top_bots()
        
        results = {
            'total_generations': self.current_generation,
            'total_bots_evaluated': len(self.bot_performances),
            'top_bots': top_bots
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        log_info(f"\nResults saved to {filepath}")
        log_info(f"Top {len(top_bots)} bots recorded")
    
    def print_top_bots(self, count: int = 10) -> None:
        """
        Print top bots to console.
        
        Args:
            count: Number of bots to print
        """
        top_bots = self.get_top_bots(count)
        
        if not top_bots:
            log_info("No profitable bots to display")
            return
        
        log_info(f"\n{'='*80}")
        log_info(f"TOP {len(top_bots)} BOTS")
        log_info(f"{'='*80}\n")
        
        for bot_info in top_bots:
            log_info(f"Rank #{bot_info['rank']}")
            log_info(f"  Bot ID: {bot_info['bot_id']}")
            log_info(f"  Generations Survived: {bot_info['generations_survived']}")
            log_info(f"  Average Profit: {bot_info['avg_profit_pct']:.2f}%")
            log_info(f"  Average Winrate: {bot_info['avg_winrate']:.2%}")
            log_info(f"  Average Trades: {bot_info['avg_trades']:.1f}")
            log_info(f"  Indicators: {len(bot_info['config']['indicators'])}")
            log_info(f"  Risk Strategies: {len(bot_info['config']['risk_strategies'])}")
            log_info("")
