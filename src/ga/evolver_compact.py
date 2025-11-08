"""
Genetic Algorithm Evolver - COMPACT BOT VERSION

Handles selection, crossover, mutation for compact bot architecture.
"""
import numpy as np
import copy
import random
from typing import List, Dict, Tuple, Optional, Set
import json

from ..bot_generator.compact_generator import CompactBotConfig, CompactBotGenerator
from ..backtester.compact_simulator import BacktestResult, CompactBacktester
from ..utils.validation import log_info, log_debug, log_warning
from ..utils.config import TOP_BOTS_COUNT, RESULTS_FILE


class GeneticAlgorithmEvolver:
    """
    Manages genetic algorithm evolution for compact bots.
    """
    
    def __init__(
        self,
        bot_generator: CompactBotGenerator,
        backtester: CompactBacktester,
        mutation_rate: float = 0.15,
        elite_pct: float = 0.1
    ):
        """
        Initialize GA evolver.
        
        Args:
            bot_generator: Bot generator instance
            backtester: Backtester instance
            mutation_rate: Probability of mutation (default 15%)
            elite_pct: Percentage of top bots to keep unchanged (default 10%)
        """
        self.bot_generator = bot_generator
        self.backtester = backtester
        self.mutation_rate = mutation_rate
        self.elite_pct = elite_pct
        
        self.current_generation = 0
        self.population: List[CompactBotConfig] = []
        self.population_results: List[BacktestResult] = []
        
        # Track combination usage for diversity
        self.used_combinations: Set[frozenset] = set()
        
        # Track best bots across generations
        self.all_time_best: List[Tuple[CompactBotConfig, BacktestResult]] = []
        
        log_info("GeneticAlgorithmEvolver initialized (compact architecture)")
        log_info(f"  Mutation rate: {mutation_rate:.1%}")
        log_info(f"  Elite percentage: {elite_pct:.1%}")
    
    def initialize_population(self) -> List[CompactBotConfig]:
        """
        Generate initial population.
        
        Returns:
            Initial population of bots
        """
        log_info("Generating initial population...")
        self.population = self.bot_generator.generate_population()
        self.current_generation = 0
        
        # Track combinations
        for bot in self.population:
            combo = frozenset(bot.indicator_indices[:bot.num_indicators])
            self.used_combinations.add(combo)
        
        log_info(f"Initial population: {len(self.population)} bots")
        log_info(f"Unique indicator combinations: {len(self.used_combinations)}")
        
        return self.population
    
    def evaluate_population(
        self,
        population: List[CompactBotConfig],
        ohlcv_data: np.ndarray,
        cycles: List[Tuple[int, int]]
    ) -> List[BacktestResult]:
        """
        Evaluate population performance.
        
        Args:
            population: List of bots
            ohlcv_data: OHLCV data array
            cycles: List of (start, end) cycle ranges
            
        Returns:
            List of backtest results
        """
        results = self.backtester.backtest_bots(population, ohlcv_data, cycles)
        
        # Filter out error results
        valid_results = [r for r in results if r.bot_id >= 0]
        
        return valid_results
    
    def select_survivors(
        self,
        population: List[CompactBotConfig],
        results: List[BacktestResult],
        survival_rate: float = 0.5
    ) -> Tuple[List[CompactBotConfig], List[BacktestResult]]:
        """
        Select top performing bots.
        Preserves ALL profitable bots (total_pnl > 0) unchanged.
        
        Args:
            population: Current population
            results: Backtest results
            survival_rate: Fraction of population to keep
            
        Returns:
            Tuple of (surviving_bots, surviving_results)
        """
        # Separate profitable and unprofitable bots
        profitable_pairs = []
        unprofitable_pairs = []
        
        for bot, result in zip(population, results):
            if result.total_pnl > 0:
                profitable_pairs.append((bot, result))
            else:
                unprofitable_pairs.append((bot, result))
        
        # Sort both groups by fitness
        profitable_pairs.sort(key=lambda x: x[1].fitness_score, reverse=True)
        unprofitable_pairs.sort(key=lambda x: x[1].fitness_score, reverse=True)
        
        # Calculate target number of survivors
        num_survivors = max(1, int(len(population) * survival_rate))
        
        # Preserve ALL profitable bots (up to num_survivors)
        if len(profitable_pairs) >= num_survivors:
            survivors = profitable_pairs[:num_survivors]
        else:
            survivors = profitable_pairs.copy()
            remaining_slots = num_survivors - len(profitable_pairs)
            survivors.extend(unprofitable_pairs[:remaining_slots])
        
        survivor_bots = [bot for bot, _ in survivors]
        survivor_results = [res for _, res in survivors]
        
        # Increment survival generations for survivors
        for bot in survivor_bots:
            bot.survival_generations += 1
        
        # Update all-time best
        self._update_all_time_best(survivors)
        
        return survivor_bots, survivor_results
    
    def _update_all_time_best(self, current_bots: List[Tuple[CompactBotConfig, BacktestResult]]):
        """Update list of all-time best bots."""
        # Add current generation's bots
        self.all_time_best.extend(current_bots)
        
        # Sort and keep top 100
        self.all_time_best.sort(key=lambda x: x[1].fitness_score, reverse=True)
        self.all_time_best = self.all_time_best[:100]
    
    def mutate_bot(self, bot: CompactBotConfig) -> CompactBotConfig:
        """
        Mutate a bot's configuration.
        Uses per-bot probability: mutation_rate% chance to apply ONE random mutation.
        
        Mutations:
        - Change indicator type
        - Adjust indicator parameter
        - Flip risk strategy bit
        - Adjust TP/SL
        - Adjust leverage
        """
        # Per-bot mutation probability (15% chance)
        if random.random() >= self.mutation_rate:
            return bot  # No mutation
        
        mutated = copy.deepcopy(bot)
        
        # Select ONE random mutation type
        mutation_type = random.randint(0, 5)
        
        if mutation_type == 0:
            # Mutation 1: Change indicator
            idx = random.randint(0, mutated.num_indicators - 1)
            old_ind = mutated.indicator_indices[idx]
            new_ind = random.randint(0, 49)
            mutated.indicator_indices[idx] = new_ind
            log_debug(f"Bot {bot.bot_id}: Mutated indicator {idx}: {old_ind} -> {new_ind}")
        
        elif mutation_type == 1:
            # Mutation 2: Adjust parameter
            idx = random.randint(0, mutated.num_indicators - 1)
            param_idx = random.randint(0, 2)
            old_val = mutated.indicator_params[idx][param_idx]
            # Adjust by ±20%
            mutated.indicator_params[idx][param_idx] *= random.uniform(0.8, 1.2)
            log_debug(f"Bot {bot.bot_id}: Mutated param [{idx}][{param_idx}]: {old_val:.2f} -> {mutated.indicator_params[idx][param_idx]:.2f}")
        
        elif mutation_type == 2:
            # Mutation 3: Flip risk strategy bit
            bit = random.randint(0, 14)
            old_bitmap = mutated.risk_strategy_bitmap
            mutated.risk_strategy_bitmap ^= (1 << bit)
            log_debug(f"Bot {bot.bot_id}: Flipped risk bit {bit}: {bin(old_bitmap)} -> {bin(mutated.risk_strategy_bitmap)}")
        
        elif mutation_type == 3:
            # Mutation 4: Adjust TP
            old_tp = mutated.tp_multiplier
            mutated.tp_multiplier *= random.uniform(0.9, 1.1)
            mutated.tp_multiplier = max(0.005, min(0.25, mutated.tp_multiplier))
            log_debug(f"Bot {bot.bot_id}: Mutated TP: {old_tp:.4f} -> {mutated.tp_multiplier:.4f}")
        
        elif mutation_type == 4:
            # Mutation 5: Adjust SL
            old_sl = mutated.sl_multiplier
            mutated.sl_multiplier *= random.uniform(0.9, 1.1)
            # Ensure SL < TP/2
            mutated.sl_multiplier = max(0.002, min(mutated.tp_multiplier/2, mutated.sl_multiplier))
            log_debug(f"Bot {bot.bot_id}: Mutated SL: {old_sl:.4f} -> {mutated.sl_multiplier:.4f}")
        
        else:  # mutation_type == 5
            # Mutation 6: Adjust leverage
            old_lev = mutated.leverage
            delta = random.choice([-1, 1]) * random.randint(1, 5)
            mutated.leverage = max(1, min(125, mutated.leverage + delta))
            log_debug(f"Bot {bot.bot_id}: Mutated leverage: {old_lev} -> {mutated.leverage}")
        
        return mutated
    
    def crossover(self, parent1: CompactBotConfig, parent2: CompactBotConfig) -> CompactBotConfig:
        """
        Perform crossover between two parents using uniform selection.
        For each gene, randomly select from parent1 or parent2.
        This preserves diversity (no averaging).
        """
        child = CompactBotConfig(
            bot_id=-1,  # Will be assigned
            num_indicators=0,
            indicator_indices=np.zeros(8, dtype=np.uint8),
            indicator_params=np.zeros((8, 3), dtype=np.float32),
            risk_strategy_bitmap=0,
            tp_multiplier=0.0,
            sl_multiplier=0.0,
            leverage=1
        )
        
        # Determine number of indicators (use min or randomly choose)
        min_ind = min(parent1.num_indicators, parent2.num_indicators)
        max_ind = max(parent1.num_indicators, parent2.num_indicators)
        child.num_indicators = random.randint(min_ind, max_ind) if min_ind < max_ind else min_ind
        
        # For each indicator slot, uniformly select from parent1 or parent2
        for i in range(child.num_indicators):
            if random.random() < 0.5:
                # Take from parent1
                if i < parent1.num_indicators:
                    child.indicator_indices[i] = parent1.indicator_indices[i]
                    child.indicator_params[i] = parent1.indicator_params[i]
                else:
                    # Fallback to parent2 if parent1 doesn't have this index
                    child.indicator_indices[i] = parent2.indicator_indices[i]
                    child.indicator_params[i] = parent2.indicator_params[i]
            else:
                # Take from parent2
                if i < parent2.num_indicators:
                    child.indicator_indices[i] = parent2.indicator_indices[i]
                    child.indicator_params[i] = parent2.indicator_params[i]
                else:
                    # Fallback to parent1 if parent2 doesn't have this index
                    child.indicator_indices[i] = parent1.indicator_indices[i]
                    child.indicator_params[i] = parent1.indicator_params[i]
        
        # Risk strategy: randomly select from parent1 or parent2 (not OR)
        child.risk_strategy_bitmap = random.choice([parent1.risk_strategy_bitmap, parent2.risk_strategy_bitmap])
        
        # TP/SL: uniformly select from parent1 or parent2 (not average)
        child.tp_multiplier = random.choice([parent1.tp_multiplier, parent2.tp_multiplier])
        child.sl_multiplier = random.choice([parent1.sl_multiplier, parent2.sl_multiplier])
        
        # Leverage: uniformly select from parent1 or parent2 (not average)
        child.leverage = random.choice([parent1.leverage, parent2.leverage])
        
        log_debug(f"Crossover: P1({parent1.bot_id}) × P2({parent2.bot_id}) -> Child (uniform selection)")
        
        return child
    
    def release_combinations(self, dead_bots: List[CompactBotConfig]):
        """Release indicator combinations from eliminated bots."""
        for bot in dead_bots:
            combo = frozenset(bot.indicator_indices[:bot.num_indicators])
            self.used_combinations.discard(combo)
        log_debug(f"Released {len(dead_bots)} combinations")
    
    def generate_unique_bot(self, bot_id: int) -> CompactBotConfig:
        """
        Generate a bot with a unique indicator combination.
        Ensures no duplicate combinations in population.
        
        Args:
            bot_id: ID to assign to new bot
            
        Returns:
            New bot with unique indicator combination
        """
        max_attempts = 50
        
        for attempt in range(max_attempts):
            # Generate new bot
            bot = self.bot_generator.generate_single_bot(bot_id)
            
            # Check if combination is unique
            combo = frozenset(bot.indicator_indices[:bot.num_indicators])
            
            if combo not in self.used_combinations:
                # Mark as used
                self.used_combinations.add(combo)
                return bot
        
        # Fallback: force unique by clearing old combinations
        log_warning(f"Could not find unique combination after {max_attempts} attempts. Clearing used_combinations.")
        self.used_combinations.clear()
        
        # Generate and mark new bot
        bot = self.bot_generator.generate_single_bot(bot_id)
        combo = frozenset(bot.indicator_indices[:bot.num_indicators])
        self.used_combinations.add(combo)
        
        return bot
    
    def refill_population(
        self,
        survivors: List[CompactBotConfig],
        target_size: int
    ) -> List[CompactBotConfig]:
        """
        Refill population to target size.
        Preserves ALL survivors unchanged (they're already profitable/best).
        Fills remaining slots with NEW unique indicator combinations.
        
        Args:
            survivors: Surviving bots from selection (profitable + best unprofitable)
            target_size: Target population size
            
        Returns:
            New full population
        """
        if len(survivors) == 0:
            log_warning("No survivors! Generating completely new population")
            return self.bot_generator.generate_population()
        
        new_population = []
        
        # Keep ALL survivors unchanged (they're already profitable or best performers)
        new_population.extend(copy.deepcopy(survivors))
        
        # Track combinations from survivors
        for bot in survivors:
            combo = frozenset(bot.indicator_indices[:bot.num_indicators])
            self.used_combinations.add(combo)
        
        # Fill remaining slots with NEW unique indicator combinations
        next_bot_id = max(bot.bot_id for bot in survivors) + 1
        num_new_bots = target_size - len(survivors)
        
        for _ in range(num_new_bots):
            # Generate bot with unique indicator combination
            new_bot = self.generate_unique_bot(next_bot_id)
            next_bot_id += 1
            new_population.append(new_bot)
        
        return new_population
    
    def run_evolution(
        self,
        num_generations: int,
        ohlcv_data: np.ndarray,
        cycles: List[Tuple[int, int]]
    ) -> None:
        """
        Run complete evolution process.
        
        Args:
            num_generations: Number of generations
            ohlcv_data: OHLCV data
            cycles: Cycle ranges
        """
        log_info(f"\nStarting Evolution: {num_generations} generations, {len(cycles)} cycles")
        
        # Initialize population if needed
        if not self.population:
            self.population = self.initialize_population()
        
        target_size = len(self.population)
        
        # Run generations
        for gen in range(num_generations):
            # Evaluate
            self.population_results = self.evaluate_population(
                self.population,
                ohlcv_data,
                cycles
            )
            
            # Select survivors
            survivors, survivor_results = self.select_survivors(
                self.population,
                self.population_results,
                survival_rate=0.5
            )
            
            # Print generation summary
            self._print_generation_summary(gen, survivor_results)
            
            # Release combinations from dead bots
            dead_bots = [bot for bot in self.population if bot not in survivors]
            self.release_combinations(dead_bots)
            
            # Refill for next generation
            self.population = self.refill_population(survivors, target_size)
            
            self.current_generation += 1
        
        # Final evaluation
        self.population_results = self.evaluate_population(
            self.population,
            ohlcv_data,
            cycles
        )
        
        log_info(f"\nEvolution complete!\n")
    
    def _print_generation_summary(self, gen: int, results: List[BacktestResult]):
        """Print concise generation summary with key metrics."""
        if not results:
            return
        
        profitable = [r for r in results if r.total_pnl > 0]
        
        if not profitable:
            print(f"Gen {gen}: {len(results)} bots, 0 profitable")
            return
        
        # Calculate averages across profitable bots
        avg_pnl = np.mean([r.total_pnl for r in profitable])
        avg_pnl_pct = (avg_pnl / 100.0) * 100  # Assuming $100 initial balance
        avg_winrate = np.mean([r.win_rate for r in profitable])
        avg_trades = np.mean([r.total_trades for r in profitable])
        avg_sharpe = np.mean([r.sharpe_ratio for r in profitable])
        max_pnl = max([r.total_pnl for r in profitable])
        
        # Print compact summary
        print(f"Gen {gen}: {len(profitable)}/{len(results)} profitable | "
              f"Avg: {avg_pnl_pct:+.1f}% profit, {avg_winrate:.1%} WR, "
              f"{avg_trades:.0f} trades, {avg_sharpe:.2f} Sharpe | "
              f"Best: ${max_pnl:.2f}")
    
    def get_top_bots(self, count: int = 10) -> List[Tuple[CompactBotConfig, BacktestResult]]:
        """
        Get top N bots from all-time best, ensuring the longest-surviving bot is included.
        
        Returns:
            List of (bot, result) tuples
        """
        if not self.all_time_best:
            return []
        
        # Get the bot that survived the most generations
        longest_surviving = max(self.all_time_best, key=lambda x: x[0].survival_generations)
        
        # Get top bots by fitness score
        top_by_fitness = sorted(self.all_time_best, key=lambda x: x[1].fitness_score, reverse=True)
        
        # Ensure longest-surviving bot is in the top 10
        top_bots = top_by_fitness[:count]
        
        # If longest-surviving bot is not in top 10, replace the lowest-ranked one
        if longest_surviving not in [bot for bot, _ in top_bots]:
            # Find the bot with lowest fitness in top 10
            lowest_in_top = min(top_bots, key=lambda x: x[1].fitness_score)
            # Replace it with longest-surviving bot
            top_bots.remove(lowest_in_top)
            top_bots.append(longest_surviving)
            # Re-sort by fitness
            top_bots.sort(key=lambda x: x[1].fitness_score, reverse=True)
        
        return top_bots
    
    def save_top_bots(self, filepath: str = RESULTS_FILE, count: int = TOP_BOTS_COUNT):
        """
        Save top bots to file and individual bot files.
        
        Args:
            filepath: Output file path
            count: Number of top bots to save
        """
        top_bots = self.get_top_bots(count)
        
        results_data = {
            'total_generations': self.current_generation,
            'total_bots_evaluated': len(self.all_time_best),
            'top_bots': []
        }
        
        for rank, (bot, result) in enumerate(top_bots, 1):
            bot_data = {
                'rank': rank,
                'bot_id': bot.bot_id,
                'fitness_score': float(result.fitness_score),
                'total_pnl': float(result.total_pnl),
                'win_rate': float(result.win_rate),
                'total_trades': int(result.total_trades),
                'sharpe_ratio': float(result.sharpe_ratio),
                'max_drawdown': float(result.max_drawdown),
                'survival_generations': bot.survival_generations,
                'config': {
                    'num_indicators': int(bot.num_indicators),
                    'indicator_indices': bot.indicator_indices[:bot.num_indicators].tolist(),
                    'risk_strategy_bitmap': int(bot.risk_strategy_bitmap),
                    'tp_multiplier': float(bot.tp_multiplier),
                    'sl_multiplier': float(bot.sl_multiplier),
                    'leverage': int(bot.leverage)
                }
            }
            results_data['top_bots'].append(bot_data)
            
            # Save individual bot file
            individual_filepath = f"bot_{bot.bot_id}.json"
            with open(individual_filepath, 'w') as f:
                json.dump(bot_data, f, indent=2)
            log_info(f"Saved bot {bot.bot_id} to {individual_filepath}")
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        log_info(f"\nSaved top {len(top_bots)} bots to {filepath} and individual files")
    
    def print_top_bots(self, count: int = 10):
        """Print top bots to console."""
        top_bots = self.get_top_bots(count)
        
        if not top_bots:
            log_info("No bots to display")
            return
        
        log_info(f"\n{'='*80}")
        log_info(f"TOP {len(top_bots)} BOTS (All-Time Best)")
        log_info(f"{'='*80}\n")
        
        for rank, (bot, result) in enumerate(top_bots, 1):
            log_info(f"Rank #{rank}")
            log_info(f"  Bot ID: {bot.bot_id}")
            log_info(f"  Fitness Score: {result.fitness_score:.2f}")
            log_info(f"  Total PnL: ${result.total_pnl:.2f}")
            log_info(f"  Win Rate: {result.win_rate:.2%}")
            log_info(f"  Total Trades: {result.total_trades}")
            log_info(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
            log_info(f"  Max Drawdown: {result.max_drawdown:.2%}")
            log_info(f"  Indicators: {bot.num_indicators}")
            log_info(f"  Leverage: {bot.leverage}x")
            log_info("")
