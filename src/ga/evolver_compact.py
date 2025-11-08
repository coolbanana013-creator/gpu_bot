"""
Genetic Algorithm Evolver - COMPACT BOT VERSION

Handles selection, crossover, mutation for compact bot architecture.
"""
import numpy as np
import copy
import random
from typing import List, Dict, Tuple, Optional, Set
import json
import time

from ..bot_generator.compact_generator import CompactBotConfig, CompactBotGenerator
from ..backtester.compact_simulator import BacktestResult, CompactBacktester
from ..utils.validation import log_info, log_debug, log_warning
from ..utils.config import TOP_BOTS_COUNT, RESULTS_FILE
from ..indicators.factory import IndicatorFactory
from .gpu_ga_processor import GPUGAProcessor
from .gpu_logging_processor import GPULoggingProcessor


class EvolutionProfiler:
    """Tracks timing and performance metrics for evolution phases."""

    def __init__(self):
        self.phase_times = {}
        self.generation_times = []
        self.start_time = None
        self.phase_start = None

    def start_evolution(self):
        """Start overall evolution timing."""
        self.start_time = time.time()
        log_info("Evolution profiler started")

    def start_phase(self, phase_name: str):
        """Start timing a specific phase."""
        self.phase_start = time.time()
        if phase_name not in self.phase_times:
            self.phase_times[phase_name] = []
        log_debug(f"Started phase: {phase_name}")

    def end_phase(self, phase_name: str):
        """End timing a specific phase."""
        if self.phase_start is None:
            log_warning(f"Phase '{phase_name}' ended without starting")
            return

        duration = time.time() - self.phase_start
        self.phase_times[phase_name].append(duration)
        log_debug(f"Phase '{phase_name}' completed in {duration:.3f}s")
        self.phase_start = None

    def start_generation(self):
        """Start timing a generation."""
        self.gen_start = time.time()

    def end_generation(self, gen_num: int):
        """End timing a generation."""
        gen_duration = time.time() - self.gen_start
        self.generation_times.append(gen_duration)
        log_info(f"Generation {gen_num} completed in {gen_duration:.3f}s")

    def print_summary(self):
        """Print comprehensive performance summary."""
        total_time = time.time() - self.start_time

        print("\n" + "="*80)
        print("EVOLUTION PERFORMANCE PROFILE")
        print("="*80)

        print(f"Total evolution time: {total_time:.3f}s")
        print(f"Total generations: {len(self.generation_times)}")

        if self.generation_times:
            avg_gen_time = np.mean(self.generation_times)
            min_gen_time = np.min(self.generation_times)
            max_gen_time = np.max(self.generation_times)
            print(f"Average generation time: {avg_gen_time:.3f}s")
            print(f"Fastest generation: {min_gen_time:.3f}s")
            print(f"Slowest generation: {max_gen_time:.3f}s")

        print("\nPhase Breakdown:")
        print("-" * 40)

        for phase, times in self.phase_times.items():
            if times:
                avg_time = np.mean(times)
                total_phase_time = np.sum(times)
                pct_of_total = (total_phase_time / total_time) * 100
                print("25")

        # Identify bottlenecks
        print("\nPotential Bottlenecks:")
        print("-" * 40)

        if self.phase_times:
            # Sort phases by average time
            sorted_phases = sorted(self.phase_times.items(),
                                 key=lambda x: np.mean(x[1]) if x[1] else 0,
                                 reverse=True)

            for phase, times in sorted_phases[:3]:  # Top 3 bottlenecks
                if times:
                    avg_time = np.mean(times)
                    pct_of_total = (np.sum(times) / total_time) * 100
                    print("15")

        print("="*80)


class GeneticAlgorithmEvolver:
    """
    Manages genetic algorithm evolution for compact bots.
    """

    def __init__(
        self,
        bot_generator: CompactBotGenerator,
        backtester: CompactBacktester,
        mutation_rate: float = 0.15,
        elite_pct: float = 0.1,
        pair: str = "xbtusdtm",
        timeframe: str = "1m",
        gpu_context=None,
        gpu_queue=None
    ):
        """
        Initialize GA evolver.

        Args:
            bot_generator: Bot generator instance
            backtester: Backtester instance
            mutation_rate: Probability of mutation (default 15%)
            elite_pct: Percentage of top bots to keep unchanged (default 10%)
            pair: Trading pair symbol (default "xbtusdtm")
            timeframe: Timeframe string (default "1m")
            gpu_context: OpenCL GPU context (optional)
            gpu_queue: OpenCL GPU command queue (optional)
        """
        self.bot_generator = bot_generator
        self.backtester = backtester
        self.mutation_rate = mutation_rate
        self.elite_pct = elite_pct
        self.pair = pair
        self.timeframe = timeframe

        # GPU acceleration
        self.gpu_processor = None
        if gpu_context is not None and gpu_queue is not None:
            try:
                self.gpu_processor = GPUGAProcessor(gpu_context, gpu_queue)
                log_info("GPU GA acceleration enabled")
            except Exception as e:
                log_warning(f"Failed to initialize GPU GA processor: {e}")
                log_info("Falling back to CPU-only GA operations")
        
        # GPU logging acceleration
        self.gpu_logger = None
        if gpu_context is not None and gpu_queue is not None:
            try:
                self.gpu_logger = GPULoggingProcessor(gpu_context, gpu_queue)
                log_info("GPU logging acceleration enabled")
            except Exception as e:
                log_warning(f"Failed to initialize GPU logging processor: {e}")
                log_info("Falling back to CPU logging")
        else:
            log_info("GPU context not provided, using CPU-only GA operations")
        
        self.current_generation = 0
        self.population: List[CompactBotConfig] = []
        self.population_results: List[BacktestResult] = []
        
        # Track combination usage for diversity
        self.used_combinations: Set[frozenset] = set()
        
        # Track best bots across generations
        self.all_time_best: List[Tuple[CompactBotConfig, BacktestResult]] = []
        
        # Performance profiler
        self.profiler = EvolutionProfiler()
        
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
        Select bots that survived with positive or null profit/loss.
        Only bots with total_pnl >= 0 are passed to the next generation.
        
        Args:
            population: Current population
            results: Backtest results
            survival_rate: Fraction of population to keep (used if all bots are profitable)
            
        Returns:
            Tuple of (surviving_bots, surviving_results)
        """
        # Use GPU acceleration if available
        if self.gpu_processor is not None:
            try:
                survivor_bots, survivor_results = self.gpu_processor.select_survivors_gpu(
                    population, results, survival_rate, survival_threshold=0.0
                )
                
                # Increment survival generations for survivors
                for bot in survivor_bots:
                    bot.survival_generations += 1
                
                # Update all-time best
                surviving_pairs = list(zip(survivor_bots, survivor_results))
                self._update_all_time_best(surviving_pairs)
                
                log_info(f"GPU-selected {len(surviving_pairs)} survivors with total_pnl >= 0")
                return survivor_bots, survivor_results
                
            except Exception as e:
                log_warning(f"GPU survivor selection failed: {e}")
                log_info("Falling back to CPU survivor selection")

        # CPU fallback implementation
        # Only keep bots with positive or null profit/loss (total_pnl >= 0)
        surviving_pairs = []
        
        for bot, result in zip(population, results):
            if result.total_pnl >= 0:  # Changed from > 0 to >= 0
                surviving_pairs.append((bot, result))
        
        # If no bots survived, keep the least unprofitable one to maintain population
        if not surviving_pairs:
            # Sort by least negative profit (closest to zero)
            all_pairs = list(zip(population, results))
            all_pairs.sort(key=lambda x: x[1].total_pnl, reverse=True)  # Best to worst
            surviving_pairs = [all_pairs[0]]  # Keep the least unprofitable
            log_warning("No bots with non-negative profit found, keeping the least unprofitable bot")
        
        # Sort survivors by fitness score (best first)
        surviving_pairs.sort(key=lambda x: x[1].fitness_score, reverse=True)
        
        # If we have more survivors than needed, keep only the best ones
        num_survivors = max(1, int(len(population) * survival_rate))
        if len(surviving_pairs) > num_survivors:
            surviving_pairs = surviving_pairs[:num_survivors]
        
        survivor_bots = [bot for bot, _ in surviving_pairs]
        survivor_results = [res for _, res in surviving_pairs]
        
        # Increment survival generations for survivors
        for bot in survivor_bots:
            bot.survival_generations += 1
        
        # Update all-time best
        self._update_all_time_best(surviving_pairs)
        
        log_info(f"Selected {len(surviving_pairs)} survivors with total_pnl >= 0")
        
        return survivor_bots, survivor_results
    
    def _update_all_time_best(self, current_bots: List[Tuple[CompactBotConfig, BacktestResult]]):
        """Update list of all-time best bots."""
        # Add current generation's bots
        self.all_time_best.extend(current_bots)
        
        # Sort and keep top 100
        self.all_time_best.sort(key=lambda x: x[1].fitness_score, reverse=True)
        self.all_time_best = self.all_time_best[:100]
    
    def _cpu_mutate_bot(self, bot: CompactBotConfig) -> CompactBotConfig:
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
            mutated.leverage = max(1, min(25, mutated.leverage + delta))
            log_debug(f"Bot {bot.bot_id}: Mutated leverage: {old_lev} -> {mutated.leverage}")
        
        return mutated
    
    def _cpu_crossover(self, parent1: CompactBotConfig, parent2: CompactBotConfig) -> CompactBotConfig:
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
        Fills remaining slots with crossover + mutation of survivors.
        
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
        
        # Fill remaining slots with crossover + mutation of survivors
        next_bot_id = max(bot.bot_id for bot in survivors) + 1
        num_new_bots = target_size - len(survivors)
        
        if num_new_bots > 0:
            # Generate children through crossover and mutation
            children = self._generate_children(survivors, num_new_bots, next_bot_id)
            new_population.extend(children)
        
        return new_population
    
    def _generate_children(
        self,
        parents: List[CompactBotConfig],
        num_children: int,
        start_bot_id: int
    ) -> List[CompactBotConfig]:
        """
        Generate children through crossover and mutation.
        
        Args:
            parents: Parent bots for breeding
            num_children: Number of children to generate
            start_bot_id: Starting bot ID for new children
            
        Returns:
            List of child bots
        """
        if len(parents) < 2:
            # Not enough parents for crossover, generate new unique bots
            children = []
            for i in range(num_children):
                new_bot = self.generate_unique_bot(start_bot_id + i)
                children.append(new_bot)
            return children
        
        # Use GPU acceleration if available
        if self.gpu_processor is not None:
            try:
                return self._gpu_generate_children(parents, num_children, start_bot_id)
            except Exception as e:
                log_warning(f"GPU child generation failed: {e}")
                log_info("Falling back to CPU child generation")
        
        # CPU fallback
        return self._cpu_generate_children(parents, num_children, start_bot_id)
    
    def _gpu_generate_children(
        self,
        parents: List[CompactBotConfig],
        num_children: int,
        start_bot_id: int
    ) -> List[CompactBotConfig]:
        """GPU-accelerated child generation."""
        # Create parent pairs for crossover
        parent_pairs = []
        for i in range(num_children):
            p1_idx = i % len(parents)
            p2_idx = (i + 1) % len(parents)  # Simple pairing strategy
            parent_pairs.append((parents[p1_idx], parents[p2_idx]))
        
        # GPU crossover
        children = self.gpu_processor.crossover_population_gpu(parent_pairs)
        
        # Assign bot IDs
        for i, child in enumerate(children):
            child.bot_id = start_bot_id + i
        
        # GPU mutation
        self.gpu_processor.mutate_population_gpu(children, self.mutation_rate)
        
        log_debug(f"GPU-generated {len(children)} children via crossover + mutation")
        return children
    
    def _cpu_generate_children(
        self,
        parents: List[CompactBotConfig],
        num_children: int,
        start_bot_id: int
    ) -> List[CompactBotConfig]:
        """CPU fallback for child generation."""
        children = []
        
        for i in range(num_children):
            # Select two random parents
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            
            # Crossover
            child = self._cpu_crossover(parent1, parent2)
            child.bot_id = start_bot_id + i
            
            # Mutation
            if random.random() < self.mutation_rate:
                self._cpu_mutate_bot(child)
            
            children.append(child)
        
        log_debug(f"CPU-generated {len(children)} children via crossover + mutation")
        return children
    
    def run_evolution(
        self,
        num_generations: int,
        ohlcv_data: np.ndarray,
        cycles: List[Tuple[int, int]],
        initial_balance: float = 100.0
    ) -> None:
        """
        Run complete evolution process.
        
        Args:
            num_generations: Number of generations
            ohlcv_data: OHLCV data
            cycles: Cycle ranges
        """
        log_info(f"\nStarting Evolution: {num_generations} generations, {len(cycles)} cycles")
        
        # Start profiling
        self.profiler.start_evolution()
        
        # Initialize population if needed
        self.profiler.start_phase("population_initialization")
        if not self.population:
            self.population = self.initialize_population()
        self.profiler.end_phase("population_initialization")
        
        target_size = len(self.population)
        
        # Run generations
        for gen in range(num_generations):
            self.profiler.start_generation()
            log_info(f"\n--- GENERATION {gen} ---")
            
            # Phase 1: Population evaluation (backtesting)
            self.profiler.start_phase("population_evaluation")
            log_info(f"Evaluating {len(self.population)} bots...")
            self.population_results = self.evaluate_population(
                self.population,
                ohlcv_data,
                cycles
            )
            self.profiler.end_phase("population_evaluation")
            
            # Phase 2: Logging generation results
            self.profiler.start_phase("logging_generation")
            log_info(f"Logging generation {gen} results...")
            self.log_generation_bots(gen, self.population, self.population_results, initial_balance, len(cycles))
            self.profiler.end_phase("logging_generation")
            
            # Phase 3: Survivor selection
            self.profiler.start_phase("survivor_selection")
            log_info("Selecting survivors...")
            survivors, survivor_results = self.select_survivors(
                self.population,
                self.population_results,
                survival_rate=0.5
            )
            self.profiler.end_phase("survivor_selection")
            
            # Phase 4: Print generation summary
            self.profiler.start_phase("generation_summary")
            self._print_generation_summary(gen, survivor_results, initial_balance)
            self.profiler.end_phase("generation_summary")
            
            # Phase 5: Release combinations from dead bots
            self.profiler.start_phase("combination_cleanup")
            dead_bots = [bot for bot in self.population if bot not in survivors]
            self.release_combinations(dead_bots)
            self.profiler.end_phase("combination_cleanup")
            
            # Phase 6: Refill population (crossover/mutation)
            self.profiler.start_phase("population_refill")
            log_info("Refilling population...")
            self.population = self.refill_population(survivors, target_size)
            self.profiler.end_phase("population_refill")
            
            self.current_generation += 1
            self.profiler.end_generation(gen)
        
        # Final evaluation
        log_info(f"\n--- FINAL EVALUATION ---")
        self.profiler.start_phase("final_evaluation")
        self.population_results = self.evaluate_population(
            self.population,
            ohlcv_data,
            cycles
        )
        self.profiler.end_phase("final_evaluation")
        
        # Final logging
        self.profiler.start_phase("final_logging")
        self.log_generation_bots(num_generations, self.population, self.population_results, initial_balance, len(cycles))
        self.profiler.end_phase("final_logging")
        
        # Print performance summary
        self.profiler.print_summary()
        
        log_info(f"\nEvolution complete!\n")
    
    def _print_generation_summary(self, gen: int, results: List[BacktestResult], initial_balance: float = 100.0):
        """Print concise generation summary with key metrics."""
        if not results:
            return
        
        profitable = [r for r in results if r.total_pnl > 0]
        
        if not profitable:
            print(f"Gen {gen}: {len(results)} bots, 0 profitable")
            return
        
        # Calculate averages across profitable bots
        avg_pnl = np.mean([r.total_pnl for r in profitable])
        avg_pnl_pct = (avg_pnl / initial_balance) * 100  # Use actual initial balance
        avg_winrate = np.mean([r.win_rate for r in profitable])
        avg_trades = np.mean([r.total_trades for r in profitable])
        avg_sharpe = np.mean([r.sharpe_ratio for r in profitable])
        max_pnl = max([r.total_pnl for r in profitable])
        
        # Print compact summary
        print(f"Gen {gen}: {len(profitable)}/{len(results)} profitable | "
              f"Avg: {avg_pnl_pct:+.1f}% profit, {avg_winrate:.1%} WR, "
              f"{avg_trades:.0f} trades, {avg_sharpe:.2f} Sharpe | "
              f"Best: ${max_pnl:.2f}")
    
    def log_generation_bots(self, gen: int, bots: List[CompactBotConfig], results: List[BacktestResult], initial_balance: float = 100.0, num_cycles: int = 10):
        """Log individual bot performance for this generation to a CSV file."""
        # Use GPU acceleration if available
        if self.gpu_logger is not None:
            try:
                self.gpu_logger.log_generation_bots_gpu(
                    gen, bots, results, initial_balance, num_cycles
                )
                log_debug(f"GPU logging completed for generation {gen}")
                return
            except Exception as e:
                log_warning(f"GPU logging failed: {e}")
                log_info("Falling back to CPU logging")

        # CPU fallback implementation
        self._cpu_log_generation_bots(gen, bots, results, initial_balance, num_cycles)
    
    def _cpu_log_generation_bots(self, gen: int, bots: List[CompactBotConfig], results: List[BacktestResult], initial_balance: float = 100.0, num_cycles: int = 10):
        """CPU fallback implementation of generation logging."""
        import os
        import csv
        
        # Create logs directory if it doesn't exist
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # Get all indicator names for mapping
        all_indicator_types = IndicatorFactory.get_all_indicator_types()
        indicator_names = [indicator_type.value for indicator_type in all_indicator_types]
        
        # CSV file for this generation
        csv_file = os.path.join(log_dir, f"generation_{gen}.csv")
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=';')  # Use semicolon as delimiter
            
            # Write header (include per-cycle columns)
            header = [
                'Generation', 'BotID', 'ProfitPct', 'WinRate', 'TotalTrades', 'FinalBalance', 
                'FitnessScore', 'SharpeRatio', 'MaxDrawdown', 'SurvivedGenerations',
                'NumIndicators', 'Leverage', 'TotalPnL', 'NumCycles', 'IndicatorsUsed'
            ]
            # Add dynamic per-cycle columns
            for i in range(num_cycles):
                header.extend([
                    f'Cycle{i}_Trades',
                    f'Cycle{i}_ProfitPct',
                    f'Cycle{i}_WinRate'
                ])
            writer.writerow(header)
            
            # Write data rows
            for bot, result in zip(bots, results):
                profit_pct = (result.total_pnl / initial_balance) * 100
                # Map indicator indices to names
                indicators_used = [indicator_names[idx] for idx in bot.indicator_indices if idx < len(indicator_names)]
                indicators_str = ', '.join(indicators_used)
                
                row = [
                    gen,
                    bot.bot_id,
                    f"{profit_pct:.2f}".replace('.', ','),
                    f"{result.win_rate:.4f}".replace('.', ','),
                    result.total_trades,
                    f"{result.final_balance:.2f}".replace('.', ','),
                    f"{result.fitness_score:.2f}".replace('.', ','),
                    f"{result.sharpe_ratio:.2f}".replace('.', ','),
                    f"{result.max_drawdown:.4f}".replace('.', ','),
                    bot.survival_generations,
                    bot.num_indicators,
                    bot.leverage,
                    f"{result.total_pnl:.2f}".replace('.', ','),
                    num_cycles,
                    indicators_str
                ]

                # Append per-cycle stats (trades, profit% relative to initial balance, winrate)
                for i in range(num_cycles):
                    # Safe extraction from result.per_cycle_trades/wins/pnl
                    try:
                        c_trades = result.per_cycle_trades[i]
                        c_wins = result.per_cycle_wins[i]
                        c_pnl = result.per_cycle_pnl[i]
                    except Exception:
                        c_trades = 0
                        c_wins = 0
                        c_pnl = 0.0

                    c_profit_pct = (c_pnl / initial_balance) * 100 if initial_balance != 0 else 0.0
                    c_winrate = (c_wins / c_trades) if c_trades > 0 else 0.0

                    row.extend([
                        c_trades,
                        f"{c_profit_pct:.2f}".replace('.', ','),
                        f"{c_winrate:.4f}".replace('.', ',')
                    ])

                writer.writerow(row)
        
        log_info(f"Logged {len(bots)} bots to {csv_file}")
    
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
    
    def save_top_bots(self, filepath: str = None, count: int = TOP_BOTS_COUNT):
        """
        Save top bots to file and individual bot files.
        
        Args:
            filepath: Output file path (if None, uses bot directory)
            count: Number of top bots to save
        """
        if filepath is None:
            import os
            bot_dir = f"bots/{self.pair}/{self.timeframe}"
            os.makedirs(bot_dir, exist_ok=True)
            filepath = f"{bot_dir}/best_bots.json"
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
            import os
            bot_dir = f"bots/{self.pair}/{self.timeframe}"
            os.makedirs(bot_dir, exist_ok=True)
            individual_filepath = f"{bot_dir}/bot_{bot.bot_id}.json"
            with open(individual_filepath, 'w') as f:
                json.dump(bot_data, f, indent=2)
            log_info(f"Saved bot {bot.bot_id} to {individual_filepath}")
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        log_info(f"\nSaved top {len(top_bots)} bots to {filepath} and individual files")
    
    def print_top_bots(self, count: int = 10, initial_balance: float = 10000.0):
        """Print top bots to console."""
        top_bots = self.get_top_bots(count)
        
        if not top_bots:
            log_info("No bots to display")
            return
        
        log_info(f"\n{'='*80}")
        log_info(f"TOP {len(top_bots)} BOTS (All-Time Best)")
        log_info(f"{'='*80}\n")
        
        for rank, (bot, result) in enumerate(top_bots, 1):
            profit_pct = (result.total_pnl / initial_balance) * 100
            log_info(f"Rank #{rank}")
            log_info(f"  Bot ID: {bot.bot_id}")
            log_info(f"  Fitness Score: {result.fitness_score:.2f}")
            log_info(f"  Total PnL: ${result.total_pnl:.2f}")
            log_info(f"  Profit %: {profit_pct:+.2f}%")
            log_info(f"  Final Balance: ${result.final_balance:.2f}")
            log_info(f"  Win Rate: {result.win_rate:.2%}")
            log_info(f"  Total Trades: {result.total_trades}")
            log_info(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
            log_info(f"  Max Drawdown: {result.max_drawdown:.2%}")
            log_info(f"  Indicators: {bot.num_indicators}")
            log_info(f"  Leverage: {bot.leverage}x")
            log_info("")
    
    def print_current_generation(self, initial_balance: float = 10000.0):
        """Print all bots from the current generation with their results."""
        if not self.population or not self.population_results:
            log_info("No current generation data to display")
            return
        
        log_info(f"\n{'='*80}")
        log_info(f"CURRENT GENERATION #{self.current_generation} - ALL {len(self.population)} BOTS")
        log_info(f"{'='*80}\n")
        
        for i, (bot, result) in enumerate(zip(self.population, self.population_results)):
            profit_pct = (result.total_pnl / initial_balance) * 100
            log_info(f"Bot #{i+1} (ID: {bot.bot_id})")
            log_info(f"  Fitness Score: {result.fitness_score:.2f}")
            log_info(f"  Total PnL: ${result.total_pnl:.2f}")
            log_info(f"  Profit %: {profit_pct:+.2f}%")
            log_info(f"  Final Balance: ${result.final_balance:.2f}")
            log_info(f"  Win Rate: {result.win_rate:.2%}")
            log_info(f"  Total Trades: {result.total_trades}")
            log_info(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
            log_info(f"  Max Drawdown: {result.max_drawdown:.2%}")
            log_info(f"  Indicators: {bot.num_indicators}")
            log_info(f"  Leverage: {bot.leverage}x")
            log_info("")
    
    def shutdown(self):
        """Shutdown GPU processors."""
        if self.gpu_processor is not None:
            log_info("Shutting down GPU GA processor")
        
        if self.gpu_logger is not None:
            self.gpu_logger.shutdown()
            log_info("Shutting down GPU logging processor")
