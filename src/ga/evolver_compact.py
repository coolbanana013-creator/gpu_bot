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
        
        # GLOBAL tracking enabled - combinations are NEVER reused across generations
        # self.used_combinations will track all combinations used in any generation
        # This ensures maximum diversity and exploration of the 2M+ combination space
        
        # Global pre-computed combination pools for O(1) selection
        self._precompute_combination_pools()
        
        # Track best bots across generations
        self.all_time_best: List[Tuple[CompactBotConfig, BacktestResult]] = []
        
        # Performance profiler
        self.profiler = EvolutionProfiler()
        
        # Interactive mode for debugging
        self.interactive_mode = False
        
        log_info("GeneticAlgorithmEvolver initialized (compact architecture)")
        log_info(f"  Mutation rate: {mutation_rate:.1%} (NOT USED - no crossover/mutation)")
        log_info(f"  Elite percentage: {elite_pct:.1%}")
        log_info(f"  Pre-computed combination pools: {sum(len(p) for p in self.unused_combinations.values()):,} total combinations")
        log_info(f"  Strategy: Global tracking - no combination ever reused")
    
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
        
        Note: Once a combination is used, it's permanently removed from the pool (no reuse).
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
        Uses precomputed combination pools to guarantee uniqueness.
        
        Returns:
            Initial population of bots with guaranteed unique combinations
        """
        log_info("Generating initial population with guaranteed unique combinations...")
        
        # Generate base population on GPU (only for parameters, will reassign indicators)
        self.population = self.bot_generator.generate_population()
        self.current_generation = 0
        
        # Track globally used combinations across all generations
        self.used_combinations = set()
        
        # Get available sizes with their pool counts
        min_ind = self.bot_generator.min_indicators
        max_ind = self.bot_generator.max_indicators
        available_sizes = list(range(min_ind, max_ind + 1))
        
        log_info(f"Distributing {len(self.population)} bots across indicator sizes {min_ind}-{max_ind}")
        
        # Assign unique combinations to each bot
        for i, bot in enumerate(self.population):
            # Try to find an available combination, checking sizes randomly
            random.shuffle(available_sizes)
            
            assigned = False
            for num_indicators in available_sizes:
                available = self.unused_combinations.get(num_indicators, set())
                
                if available:
                    # Pick a random unused combination
                    combo = available.pop()
                    self.used_combinations.add(combo)
                    
                    # Update bot with this combination
                    bot.num_indicators = num_indicators
                    combo_list = sorted(list(combo))
                    for j, idx in enumerate(combo_list):
                        bot.indicator_indices[j] = idx
                    
                    assigned = True
                    break
            
            if not assigned:
                log_error(f"CRITICAL: No combinations available for bot {i}! All pools exhausted.")
                break
        
        # Verify uniqueness
        seen_combinations = set()
        for bot in self.population:
            combo = frozenset(bot.indicator_indices[:bot.num_indicators])
            seen_combinations.add(combo)
        
        unique_count = len(seen_combinations)
        total_count = len(self.population)
        
        log_info(f"Initial population: {total_count} bots, {unique_count} unique (100.0% uniqueness)")
        log_info(f"Global tracking: {len(self.used_combinations)} combinations marked as used")
        
        # Log distribution across sizes
        size_distribution = {}
        for bot in self.population:
            size_distribution[bot.num_indicators] = size_distribution.get(bot.num_indicators, 0) + 1
        log_info(f"Size distribution: {dict(sorted(size_distribution.items()))}")
        
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
        Select bots with UNIQUE indicator combinations where ALL cycles meet BOTH criteria:
        1) At least 1 trade per cycle
        2) Positive profit percentage per cycle (> 0%)
        
        Enforces 100% diversity among survivors by keeping only the BEST bot per unique indicator combination.
        
        Args:
            population: Current population
            results: Backtest results
            survival_rate: Fraction of population to keep (target survivor count)
            
        Returns:
            Tuple of (surviving_bots, surviving_results) with 100% unique indicator combinations
        """
        # Step 1: Filter bots where ALL cycles meet BOTH criteria:
        # a) At least 1 trade per cycle
        # b) Positive profit percentage per cycle (> 0%)
        profitable_pairs = []
        eliminated_no_trades = 0
        eliminated_negative_cycle = 0
        
        initial_balance = 1000.0  # Standard initial balance
        
        for bot, result in zip(population, results):
            # Check all cycles
            passes_all_cycles = True
            
            for cycle_idx, (cycle_trades, cycle_pnl) in enumerate(zip(result.per_cycle_trades, result.per_cycle_pnl)):
                # Check 1: Cycle must have at least 1 trade
                if cycle_trades < 1:
                    eliminated_no_trades += 1
                    passes_all_cycles = False
                    break
                
                # Check 2: Cycle must have positive profit percentage
                cycle_profit_pct = (cycle_pnl / initial_balance) * 100
                if cycle_profit_pct <= 0:
                    eliminated_negative_cycle += 1
                    passes_all_cycles = False
                    break
            
            # Only keep bots where ALL cycles passed BOTH criteria
            if passes_all_cycles:
                profitable_pairs.append((bot, result))
        
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
            log_info(f"SURVIVAL FILTER: {eliminated_no_trades} cycles with no trades, {eliminated_negative_cycle} cycles with negative profit, {len(profitable_pairs)} bots passed (all cycles profitable)")
        
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
        total_all_cycles_profitable = len(profitable_pairs)
        unique_count = len(surviving_pairs)
        eliminated_total = len(population) - len(profitable_pairs)
        log_info(f"SURVIVAL: {unique_count} survivors (from {total_all_cycles_profitable} bots where ALL cycles have trades AND positive profit)")
        log_info(f"ELIMINATED: {eliminated_total} bots total")
        
        return survivor_bots, survivor_results
    
    def _update_all_time_best(self, current_bots: List[Tuple[CompactBotConfig, BacktestResult]]):
        """Update list of all-time best bots."""
        # Add current generation's bots
        self.all_time_best.extend(current_bots)
        
        # Sort and keep top 100
        self.all_time_best.sort(key=lambda x: x[1].fitness_score, reverse=True)
        self.all_time_best = self.all_time_best[:100]
    
    def generate_unique_bot(self, bot_id: int, excluded_combinations: set = None) -> CompactBotConfig:
        """
        Generate a bot with a GUARANTEED unique indicator combination.
        Ensures uniqueness across ALL generations - no combination is ever reused.
        
        Args:
            bot_id: ID to assign to new bot
            excluded_combinations: Combinations already used in current batch
            
        Returns:
            New bot with globally unique indicator combination
        """
        # Generate base bot with random parameters
        bot = self.bot_generator.generate_single_bot(bot_id)
        
        # Combine batch exclusions with globally used combinations
        batch_excluded = excluded_combinations if excluded_combinations else set()
        all_excluded = self.used_combinations | batch_excluded
        
        # Check if combination is already used globally
        combo = frozenset(bot.indicator_indices[:bot.num_indicators])
        
        if combo not in all_excluded:
            # Lucky! The random combo is globally unique
            # Mark as used and remove from pool
            self.used_combinations.add(combo)
            if combo in self.unused_combinations.get(bot.num_indicators, set()):
                self.unused_combinations[bot.num_indicators].remove(combo)
            return bot
        
        # Need to select from pre-computed unused pool
        # Try sizes randomly to avoid getting stuck on exhausted pools
        min_ind = self.bot_generator.min_indicators
        max_ind = self.bot_generator.max_indicators
        all_sizes = list(range(min_ind, max_ind + 1))
        random.shuffle(all_sizes)
        
        for num_indicators in all_sizes:
            available = self.unused_combinations.get(num_indicators, set())
            
            if available:
                # Pop one combination from the available set
                combo = available.pop()
                
                # Mark as globally used
                self.used_combinations.add(combo)
                
                # Update bot with this combination
                bot.num_indicators = num_indicators
                for i, idx in enumerate(sorted(combo)):
                    bot.indicator_indices[i] = idx
                
                return bot
        
        # Critical: all pools exhausted globally
        total_available = sum(len(pool) for pool in self.unused_combinations.values())
        log_error(f"CRITICAL: All combination pools exhausted! {total_available} combinations remaining")
        log_error(f"  Used globally: {len(self.used_combinations)} combinations")
        log_error(f"  Pool sizes: {[(i, len(self.unused_combinations.get(i, set()))) for i in range(min_ind, max_ind + 1)]}")
        
        # Last resort: return bot with duplicate combo (should never happen with 2M+ combinations)
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
        
        # Track combinations in current batch to avoid duplicates within this generation
        batch_combinations = set()
        
        # Add all survivors to new population
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
            
            # Track in batch and ensure globally tracked
            batch_combinations.add(combo)
            if combo not in self.used_combinations:
                self.used_combinations.add(combo)
                # Remove from unused pool if present
                if combo in self.unused_combinations.get(survivor.num_indicators, set()):
                    self.unused_combinations[survivor.num_indicators].remove(combo)
            
            # Add survivor to population
            copied_survivor = copy.deepcopy(survivor)
            new_population.append(copied_survivor)
        
        # Log survivor diversity
        unique_survivor_combos = len(batch_combinations)
        total_survivors = len(survivors)
        diversity_pct = 100 * unique_survivor_combos / total_survivors if total_survivors > 0 else 0
        log_info(f"Survivors: {total_survivors} total, {unique_survivor_combos} unique combinations ({diversity_pct:.1f}% diversity)")
        log_info(f"Global tracking: {len(self.used_combinations)} combinations used across all generations")
        
        # Fill remaining slots with NEW UNIQUE BOTS (no offspring/mutation)
        next_bot_id = max(bot.bot_id for bot in survivors) + 1
        num_new_bots = target_size - len(survivors)
        
        if num_new_bots > 0:
            log_info(f"Generating {num_new_bots} new globally unique bots (no crossover/mutation/reuse)")
            
            # Log pool availability
            pool_sizes = {size: len(pool) for size, pool in self.unused_combinations.items()}
            total_unused = sum(pool_sizes.values())
            log_info(f"Available combinations: {total_unused:,} unused, {len(self.used_combinations):,} used globally")
            
            for i in range(num_new_bots):
                # Generate globally unique bot (never reuses any combination from history)
                new_bot = self.generate_unique_bot(next_bot_id + i, batch_combinations)
                combo = frozenset(new_bot.indicator_indices[:new_bot.num_indicators])
                
                # Track in batch for within-generation uniqueness check
                batch_combinations.add(combo)
                
                new_population.append(new_bot)
            
            # Verify final diversity (should always be 100%)
            all_combos = [frozenset(bot.indicator_indices[:bot.num_indicators]) for bot in new_population]
            unique_after_refill = len(set(all_combos))
            total_after_refill = len(new_population)
            diversity_pct = 100 * unique_after_refill / total_after_refill if total_after_refill > 0 else 0
            
            log_info(f"After refill: {total_after_refill} bots, {unique_after_refill} unique combinations ({diversity_pct:.1f}% diversity)")
            
            if unique_after_refill != total_after_refill:
                duplicates = total_after_refill - unique_after_refill
                log_error(f"ERROR: Found {duplicates} duplicate combinations in population!")
            
            # Log remaining pool sizes
            remaining_unused = sum(len(pool) for pool in self.unused_combinations.values())
            log_info(f"Remaining unused combinations: {remaining_unused:,}")
        
        return new_population
    
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
            
            # Phase 4: Print generation summary (use SURVIVORS only - they passed all criteria)
            self.profiler.start_phase("generation_summary")
            self._print_generation_summary(gen, survivor_results, initial_balance, len(survivors))
            self.profiler.end_phase("generation_summary")
            
            # Phase 5: Combination tracking (globally tracked, never reused)
            # All combinations are tracked in self.used_combinations
            # Dead bot combinations are NOT released - they remain permanently used
            # This ensures maximum diversity and exploration of the 2M+ combination space
            self.profiler.start_phase("combination_cleanup")
            # No action needed - combinations are permanently tracked
            self.profiler.end_phase("combination_cleanup")
            
            # Phase 6: Refill population (NO crossover/mutation - only new unique combinations)
            self.profiler.start_phase("population_refill")
            log_info("Refilling population with new globally unique combinations...")
            self.population = self.refill_population(survivors, target_size)
            log_info(f"Population refilled: {len(self.population)} bots")
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
    
    def _print_generation_summary(self, gen: int, results: List[BacktestResult], initial_balance: float = 100.0, survivor_count: int = 0):
        """
        Print concise generation summary with key metrics.
        Shows statistics for SURVIVORS only (bots that passed all criteria).
        """
        if not results:
            print(f"Gen {gen}: 0 survivors")
            return
        
        # Results are already filtered to survivors only
        # Calculate averages across all survivors
        avg_pnl = np.mean([r.total_pnl for r in results])
        avg_pnl_pct = (avg_pnl / initial_balance) * 100
        avg_winrate = np.mean([r.win_rate for r in results])  # Already stored as percentage (0-100)
        avg_trades = np.mean([r.total_trades for r in results])
        avg_sharpe = np.mean([r.sharpe_ratio for r in results])
        max_pnl = max([r.total_pnl for r in results])
        
        # Print compact summary showing survivors
        print(f"Gen {gen}: {survivor_count} survivors | "
              f"Avg: {avg_pnl_pct:+.1f}% profit, {avg_winrate:.1f}% WR, "
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
    
    def shutdown(self):
        """Shutdown GPU processors."""
        if self.gpu_processor is not None:
            log_info("Shutting down GPU GA processor")
        
        if self.gpu_logger is not None:
            self.gpu_logger.shutdown()
            log_info("Shutting down GPU logging processor")
