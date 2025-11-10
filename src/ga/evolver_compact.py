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
from ..utils.validation import log_info, log_debug, log_warning, log_error
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
        
        # NO global tracking - combinations can be reused across generations
        # Only enforce uniqueness WITHIN each generation (not across all history)
        
        # Global pre-computed combination pools for O(1) selection
        self._precompute_combination_pools()
        
        # Track best bots across generations
        self.all_time_best: List[Tuple[CompactBotConfig, BacktestResult]] = []
        
        # Performance profiler
        self.profiler = EvolutionProfiler()
        
        # Interactive mode for debugging
        self.interactive_mode = False
        
        log_info("GeneticAlgorithmEvolver initialized (compact architecture)")
        log_info(f"  Mutation rate: {mutation_rate:.1%}")
        log_info(f"  Elite percentage: {elite_pct:.1%}")
        log_info(f"  Pre-computed combination pools: {sum(len(p) for p in self.unused_combinations.values())} total combinations")
    
    def _precompute_combination_pools(self):
        """
        Pre-compute all possible indicator combinations for fast O(1) selection.
        
        INCREMENTAL CACHING: Each combination size (1, 2, 3, etc.) is saved separately.
        This allows reusing existing cache files when increasing max_indicators.
        
        Example:
        - First run with max=5: Generates and saves 1.pkl, 2.pkl, 3.pkl, 4.pkl, 5.pkl
        - Next run with max=6: Reuses 1-5.pkl files, only generates 6.pkl
        - Next run with max=8: Reuses 1-6.pkl files, only generates 7.pkl and 8.pkl
        
        Combinations by size:
        - 1 indicator: C(50,1) = 50
        - 2 indicators: C(50,2) = 1,225
        - 3 indicators: C(50,3) = 19,600
        - 4 indicators: C(50,4) = 230,300
        - 5 indicators: C(50,5) = 2,118,760
        - 6 indicators: C(50,6) = 15,890,700
        - 7 indicators: C(50,7) = 99,884,400
        - 8 indicators: C(50,8) = 536,878,650
        Total: ~652 million combinations (1-8 indicators)
        
        Note: Pools are shared across all generations. Combinations can be reused.
        """
        import itertools
        import pickle
        import os
        
        self.unused_combinations = {}
        available_indicators = list(range(50))
        
        # Only pre-compute up to max_indicators to avoid excessive memory/time
        min_ind = self.bot_generator.min_indicators
        max_ind = self.bot_generator.max_indicators
        
        os.makedirs('cache', exist_ok=True)
        
        # Load or generate each combination size separately
        for num_indicators in range(min_ind, max_ind + 1):
            cache_file = f"cache/indicator_combinations_{num_indicators}.pkl"
            
            # Try to load from cache first
            if os.path.exists(cache_file):
                try:
                    log_info(f"  Loading {num_indicators}-indicator combinations from cache...")
                    with open(cache_file, 'rb') as f:
                        combos = pickle.load(f)
                    self.unused_combinations[num_indicators] = combos
                    log_info(f"    Loaded {len(combos):,} combinations from cache")
                    continue
                except Exception as e:
                    log_warning(f"  Failed to load cache for {num_indicators}-indicator combinations: {e}")
            
            # Generate if not cached
            log_info(f"  Generating {num_indicators}-indicator combinations...")
            combos = set(
                frozenset(combo) 
                for combo in itertools.combinations(available_indicators, num_indicators)
            )
            self.unused_combinations[num_indicators] = combos
            log_info(f"    Generated {len(combos):,} combinations")
            
            # Save to individual cache file
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(combos, f)
                log_info(f"    Saved to cache: {cache_file}")
            except Exception as e:
                log_warning(f"  Failed to save cache for {num_indicators}-indicator combinations: {e}")
        
        total_combos = sum(len(v) for v in self.unused_combinations.values())
        log_info(f"  Total combinations ready: {total_combos:,}")
    
    def initialize_population(self) -> List[CompactBotConfig]:
        """
        Generate initial population with 100% unique indicator combinations.
        
        Returns:
            Initial population of bots with guaranteed unique combinations
        """
        log_info("Generating initial population...")
        self.population = self.bot_generator.generate_population()
        self.current_generation = 0
        
        # GPU generates random combinations with natural diversity (~70-98% for large populations)
        # No global tracking - combinations can be reused across generations
        seen_combinations = set()
        
        for bot in self.population:
            combo = frozenset(bot.indicator_indices[:bot.num_indicators])
            seen_combinations.add(combo)
        
        # DEBUG: Log diversity metrics
        unique_combos = len(seen_combinations)
        total_bots = len(self.population)
        diversity_pct = 100 * unique_combos / total_bots if total_bots > 0 else 0
        
        log_info(f"Initial population: {total_bots} bots, {unique_combos} unique ({diversity_pct:.1f}% natural diversity)")
        log_info(f"Note: Combinations can be reused across generations (no global tracking)")
        
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
        Select bots with UNIQUE indicator combinations that meet survival criteria:
        1) Positive PnL percentage
        2) At least 1 trade per cycle
        
        Enforces 100% diversity among survivors by keeping only the BEST bot per unique indicator combination.
        
        Args:
            population: Current population
            results: Backtest results
            survival_rate: Fraction of population to keep (target survivor count)
            
        Returns:
            Tuple of (surviving_bots, surviving_results) with 100% unique indicator combinations
        """
        # Step 1: Filter bots with:
        # a) Positive PnL percentage (profit_pct > 0)
        # b) At least 1 trade per cycle (all cycles must have >= 1 trade)
        profitable_pairs = []
        eliminated_no_trades = 0
        eliminated_negative_pnl = 0
        
        for bot, result in zip(population, results):
            # Check 1: Positive profit percentage
            initial_balance = 1000.0  # Standard initial balance
            profit_pct = (result.total_pnl / initial_balance) * 100 if initial_balance != 0 else 0.0
            if profit_pct <= 0:
                eliminated_negative_pnl += 1
                continue
            
            # Check 2: All cycles must have at least 1 trade
            has_empty_cycle = False
            for cycle_trades in result.per_cycle_trades:
                if cycle_trades < 1:
                    has_empty_cycle = True
                    eliminated_no_trades += 1
                    break
            
            # Only keep bots that pass BOTH criteria
            if not has_empty_cycle:
                profitable_pairs.append((bot, result))
        
        # If no bots passed criteria, relax to just positive PnL
        if not profitable_pairs:
            log_warning(f"SURVIVAL FILTER: {eliminated_no_trades} bots eliminated for missing trades, {eliminated_negative_pnl} for non-positive PnL")
            log_warning("No bots passed both criteria, relaxing to positive PnL only")
            for bot, result in zip(population, results):
                initial_balance = 1000.0
                profit_pct = (result.total_pnl / initial_balance) * 100 if initial_balance != 0 else 0.0
                if profit_pct > 0:
                    profitable_pairs.append((bot, result))
        
        # If still no profitable bots, keep the most profitable
        if not profitable_pairs:
            all_pairs = list(zip(population, results))
            all_pairs.sort(key=lambda x: x[1].total_pnl, reverse=True)
            profitable_pairs = [all_pairs[0]]
            log_warning("No bots with positive profit, keeping most profitable")
        else:
            log_info(f"SURVIVAL FILTER: {eliminated_no_trades} bots eliminated for missing trades, {eliminated_negative_pnl} for non-positive PnL, {len(profitable_pairs)} passed")
        
        # Step 2: Sort by fitness score (best first)
        profitable_pairs.sort(key=lambda x: x[1].fitness_score, reverse=True)
        
        # Step 3: ENFORCE DIVERSITY - Keep only BEST bot per unique indicator combination
        unique_survivors = {}  # {combo: (bot, result)}
        seen_combos = set()
        
        for bot, result in profitable_pairs:
            combo = frozenset(bot.indicator_indices[:bot.num_indicators])
            
            # Keep FIRST occurrence (already sorted by fitness, so this is the best)
            if combo not in seen_combos:
                unique_survivors[combo] = (bot, result)
                seen_combos.add(combo)
        
        # Convert to lists
        surviving_pairs = list(unique_survivors.values())
        
        # Step 4: Keep ALL unique survivors (no artificial cap)
        # Previously limited to survival_rate * population, but now we keep all that passed filters
        log_info(f"Survivors: {len(surviving_pairs)} unique bots kept (no cap applied)")
        
        # Extract bots and results
        survivor_bots = [bot for bot, _ in surviving_pairs]
        survivor_results = [result for _, result in surviving_pairs]
        
        # Increment survival generations
        for bot in survivor_bots:
            if hasattr(bot, 'survival_generations'):
                try:
                    current_sg = int(bot.survival_generations)
                    if not isinstance(current_sg, int) or current_sg < 0 or current_sg > 1000:
                        current_sg = 0
                except (ValueError, TypeError):
                    current_sg = 0
                bot.survival_generations = current_sg + 1
            else:
                bot.survival_generations = 1
        
        # Update all-time best
        self._update_all_time_best(surviving_pairs)
        
        # Log diversity and filtering stats
        total_profitable = len(profitable_pairs)
        unique_count = len(surviving_pairs)
        eliminated_total = len(population) - len(profitable_pairs)
        log_info(f"SURVIVAL: {unique_count} survivors (from {total_profitable} bots with positive PnL and all cycles with trades)")
        log_info(f"ELIMINATED: {eliminated_total} bots total")
        
        return survivor_bots, survivor_results
    
    def _update_all_time_best(self, current_bots: List[Tuple[CompactBotConfig, BacktestResult]]):
        """Update list of all-time best bots."""
        # Add current generation's bots
        self.all_time_best.extend(current_bots)
        
        # Sort and keep top 100
        self.all_time_best.sort(key=lambda x: x[1].fitness_score, reverse=True)
        self.all_time_best = self.all_time_best[:100]
    
    # LEGACY METHODS REMOVED: _cpu_mutate_bot, _cpu_crossover
    # These are no longer used as we generate only new unique bots instead of offspring
    
    def release_combinations(self, dead_bots: List[CompactBotConfig]):
        """
        DEPRECATED: No longer used (per-generation uniqueness only).
        Release indicator combinations from eliminated bots.
        Returns combinations to the unused pool for recycling.
        """
        # NOTE: This method is no longer called.
        # We use per-generation uniqueness, not global tracking.
        # Combinations are automatically available for reuse in next generation.
        pass
    
    def generate_unique_bot(self, bot_id: int, excluded_combinations: set = None) -> CompactBotConfig:
        """
        Generate a bot with a GUARANTEED unique indicator combination.
        Unique within current batch only - allows reuse from dead bots in previous generations.
        
        Args:
            bot_id: ID to assign to new bot
            excluded_combinations: Combinations to exclude in THIS batch (survivors + already generated)
            
        Returns:
            New bot with unique indicator combination within current batch
        """
        # Generate base bot with random parameters
        bot = self.bot_generator.generate_single_bot(bot_id)
        
        # Only exclude combinations in THIS batch (not global history)
        batch_excluded = excluded_combinations if excluded_combinations else set()
        
        # Check if combination is already used in this batch
        combo = frozenset(bot.indicator_indices[:bot.num_indicators])
        
        if combo not in batch_excluded:
            # Lucky! The random combo is unique in this batch
            return bot
        
        # Need to select from pre-computed unused pool
        num_indicators = bot.num_indicators
        
        # Get available combinations for this size (exclude only current batch)
        available = self.unused_combinations[num_indicators] - batch_excluded
        
        if available:
            # Pop one combination from the available set
            combo = available.pop()
            
            # Update bot with this combination
            for i, idx in enumerate(sorted(combo)):
                bot.indicator_indices[i] = idx
            bot.num_indicators = len(combo)
            return bot
        
        # Try alternative sizes if current size is exhausted in this batch
        # Prefer LARGER sizes first (4->5->3->2->1 for more combinations)
        alternatives = []
        for delta in [1, 2, -1, -2]:
            alt_size = num_indicators + delta
            if 1 <= alt_size <= 8:  # Support up to 8 indicators
                alternatives.append(alt_size)
        
        for alternative_size in alternatives:
            available = self.unused_combinations[alternative_size] - batch_excluded
            if available:
                combo = available.pop()
                
                # Update bot with alternative size
                bot.num_indicators = alternative_size
                for i, idx in enumerate(sorted(combo)):
                    bot.indicator_indices[i] = idx
                log_debug(f"Changed from {num_indicators} to {alternative_size} indicators (batch exhausted)")
                return bot
        
        # Critical: all pools exhausted for this batch
        # This is unlikely unless batch size > 2M combinations
        total_available = sum(len(pool - batch_excluded) for pool in self.unused_combinations.values())
        log_error(f"CRITICAL: All pools exhausted for batch! {total_available} globally available")
        log_error(f"  Batch size: {len(batch_excluded)}, Pool availability: {[len(self.unused_combinations[i] - batch_excluded) for i in range(1, 9)]}")
        
        # Last resort: return duplicate (will be logged in refill_population)
        return bot
    
    def refill_population(
        self,
        survivors: List[CompactBotConfig],
        target_size: int
    ) -> List[CompactBotConfig]:
        """
        Refill population to target size.
        Preserves ALL survivors unchanged (they're already profitable/best).
        Fills remaining slots with NEW UNIQUE BOTS (no crossover, no mutation).
        
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
        
        # Track unique indicator combinations in THIS generation only
        batch_combinations = set()  # Only track within THIS batch/generation
        survivor_combos = []  # For debugging
        
        # Add all survivors to new population (no deduplication)
        for survivor in survivors:
            # Validate survival_generations before processing
            if hasattr(survivor, 'survival_generations'):
                try:
                    sg = int(survivor.survival_generations)
                    if not isinstance(sg, int) or sg < 0 or sg > 1000:
                        survivor.survival_generations = 0
                except (ValueError, TypeError):
                    survivor.survival_generations = 0
            
            combo = frozenset(survivor.indicator_indices[:survivor.num_indicators])
            
            # Track this combination ONLY in batch (no global tracking)
            batch_combinations.add(combo)
            survivor_combos.append(combo)
            
            # Add survivor to population
            copied_survivor = copy.deepcopy(survivor)
            new_population.append(copied_survivor)
        
        # DEBUG: Log survivor diversity (should be 100% after select_survivors enforces uniqueness)
        unique_survivor_combos = len(batch_combinations)
        total_survivors = len(survivors)
        diversity_pct = 100 * unique_survivor_combos / total_survivors if total_survivors > 0 else 0
        log_info(f"Survivors for refill: {total_survivors} total, {unique_survivor_combos} unique indicator combinations ({diversity_pct:.1f}% diversity)")
        
        # Fill remaining slots with NEW UNIQUE BOTS (no offspring/mutation)
        next_bot_id = max(bot.bot_id for bot in survivors) + 1
        num_new_bots = target_size - len(survivors)  # Use actual survivor count
        
        if num_new_bots > 0:
            # Generate completely new unique bots with NO duplicates within this batch
            log_info(f"Generating {num_new_bots} new unique bots (no crossover/mutation)")
            
            # DEBUG: Log pool sizes before generation
            pool_sizes = {size: len(pool) for size, pool in self.unused_combinations.items()}
            log_info(f"Unused pool sizes: {pool_sizes}")
            log_info(f"batch_combinations size: {len(batch_combinations)} (tracking THIS generation only)")
            
            for i in range(num_new_bots):
                # CRITICAL: Pass batch_combinations to exclude combos already in THIS batch
                # This ensures uniqueness WITHIN generation, but allows reuse across generations
                new_bot = self.generate_unique_bot(next_bot_id + i, batch_combinations)
                combo = frozenset(new_bot.indicator_indices[:new_bot.num_indicators])
                
                # DEBUG: Check if combo was already in batch (should be rare)
                if combo in batch_combinations:
                    all_indicator_types = IndicatorFactory.get_all_indicator_types()
                    indicator_names = [indicator_type.value for indicator_type in all_indicator_types]
                    indicators_str = ', '.join([indicator_names[idx] for idx in sorted(combo)])
                    log_warning(f"Bot {i}: DUPLICATE DETECTED! combo = [{indicators_str}]")
                    log_warning(f"  batch_combinations size: {len(batch_combinations)}, unused pool sizes: {pool_sizes}")
                
                # CRITICAL: Track in batch (no global tracking - allows reuse across generations)
                batch_combinations.add(combo)
                
                new_population.append(new_bot)
            
            # DEBUG: Log diversity metrics after refill
            all_combos = [frozenset(bot.indicator_indices[:bot.num_indicators]) for bot in new_population]
            unique_after_refill = len(set(all_combos))
            total_after_refill = len(new_population)
            diversity_pct = 100 * unique_after_refill / total_after_refill if total_after_refill > 0 else 0
            
            log_info(f"After refill: {total_after_refill} total bots, {unique_after_refill} unique indicator combinations ({diversity_pct:.1f}% diversity)")
            log_info(f"Combinations reused across generations (allowed): {total_after_refill - unique_after_refill} potential duplicates")
        
        return new_population
    
    # LEGACY METHODS REMOVED: _generate_children, _gpu_generate_children, _cpu_generate_children
    # These are no longer used - we generate only new unique bots in refill_population()
    
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
            
            # Phase 3: Survivor selection (MOVED BEFORE LOGGING)
            self.profiler.start_phase("survivor_selection")
            log_info("Selecting survivors...")
            survivors, survivor_results = self.select_survivors(
                self.population,
                self.population_results,
                survival_rate=0.5
            )
            self.profiler.end_phase("survivor_selection")
            
            # Phase 2: Logging generation results (AFTER selection so survival_generations is incremented)
            self.profiler.start_phase("logging_generation")
            log_info(f"Logging generation {gen} results...")
            self.log_generation_bots(gen, self.population, self.population_results, initial_balance, len(cycles))
            self.profiler.end_phase("logging_generation")
            
            # Phase 4: Print generation summary (use FULL population results, not just survivors)
            self.profiler.start_phase("generation_summary")
            self._print_generation_summary(gen, self.population_results, initial_balance)
            self.profiler.end_phase("generation_summary")
            
            # Phase 5: DON'T Release combinations - keep them reserved for maximum diversity!
            # NOTE: We used to release dead bot combinations back to unused pool, but that caused
            # massive duplicates because new bots would immediately reuse the same combinations.
            # By keeping combinations reserved (in self.used_combinations), we ensure continuous
            # exploration of the 2.37M combination space.
            self.profiler.start_phase("combination_cleanup")
            # survivor_set = set(survivors)
            # dead_bots = [bot for bot in self.population if bot not in survivor_set]
            # self.release_combinations(dead_bots)  # DISABLED for maximum diversity
            self.profiler.end_phase("combination_cleanup")
            
            # Phase 6: Refill population (crossover/mutation)
            self.profiler.start_phase("population_refill")
            log_info("Refilling population...")
            self.population = self.refill_population(survivors, target_size)
            log_info(f"Population refilled: {len(self.population)} bots")
            # Note: No global tracking - combinations reused across generations
            self.profiler.end_phase("population_refill")
            
            self.current_generation += 1
            self.profiler.end_generation(gen)
            
            # INTERACTIVE DEBUG: Wait for user input after each generation
            if hasattr(self, 'interactive_mode') and self.interactive_mode:
                print(f"\n{'='*60}")
                print(f"Generation {gen} complete. Press Enter to continue...")
                print(f"{'='*60}")
                input()
        
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
    
    def save_top_bots(self, filepath: str = None, count: int = TOP_BOTS_COUNT, filter_all_profitable: bool = True):
        """
        Save top bots to file and individual bot files.
        Only saves bots where all cycles are profitable.
        
        Args:
            filepath: Output file path (if None, uses bot directory with timestamp)
            count: Number of top bots to save
            filter_all_profitable: If True, only save bots where all cycles are profitable
        """
        import os
        from datetime import datetime
        
        # Create run-specific directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_pair = self.pair.replace('/', '_').replace(':', '_')
        bot_dir = f"bots/{safe_pair}/{self.timeframe}/run_{timestamp}"
        os.makedirs(bot_dir, exist_ok=True)
        
        if filepath is None:
            filepath = f"{bot_dir}/best_bots.json"
            
        top_bots = self.get_top_bots(count)
        
        # Filter bots: only keep those where ALL cycles are profitable
        if filter_all_profitable:
            filtered_bots = []
            for bot, result in top_bots:
                # Check if bot has per-cycle results
                if hasattr(result, 'per_cycle_results') and result.per_cycle_results:
                    all_profitable = all(
                        cycle_result.get('pnl', 0) > 0 
                        for cycle_result in result.per_cycle_results
                    )
                    if all_profitable:
                        filtered_bots.append((bot, result))
                else:
                    # If no per-cycle data, use overall profitability
                    if result.total_pnl > 0:
                        filtered_bots.append((bot, result))
            
            top_bots = filtered_bots
            log_info(f"Filtered to {len(top_bots)} bots where all cycles are profitable")
        
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
                    'indicator_params': bot.indicator_params[:bot.num_indicators].tolist(),  # CRITICAL: Save params!
                    'risk_strategy_bitmap': int(bot.risk_strategy_bitmap),
                    'tp_multiplier': float(bot.tp_multiplier),
                    'sl_multiplier': float(bot.sl_multiplier),
                    'leverage': int(bot.leverage)
                }
            }
            results_data['top_bots'].append(bot_data)
            
            # Save individual bot file
            import os
            # Sanitize pair name for file path (replace / and : with _)
            safe_pair = self.pair.replace('/', '_').replace(':', '_')
            bot_dir = f"bots/{safe_pair}/{self.timeframe}"
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
        """DEPRECATED: Detailed per-bot printing removed to reduce output noise."""
        # This function intentionally does nothing - detailed logging removed
        pass
    
    def shutdown(self):
        """Shutdown GPU processors."""
        if self.gpu_processor is not None:
            log_info("Shutting down GPU GA processor")
        
        if self.gpu_logger is not None:
            self.gpu_logger.shutdown()
            log_info("Shutting down GPU logging processor")
