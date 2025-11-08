"""
GPU-Accelerated Genetic Algorithm Operations
Provides GPU acceleration for selection, crossover, and mutation operations.
"""

import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
from typing import List, Tuple, Optional
import logging
from ..utils.validation import log_info, log_warning, log_error, log_debug

from ..bot_generator.compact_generator import CompactBotConfig
from ..backtester.compact_simulator import BacktestResult


class GPUGAProcessor:
    """
    GPU-accelerated genetic algorithm operations using OpenCL.
    Provides parallel processing for selection, crossover, and mutation.
    """

    def __init__(self, gpu_context: cl.Context, gpu_queue: cl.CommandQueue):
        """
        Initialize GPU GA processor.

        Args:
            gpu_context: OpenCL context
            gpu_queue: OpenCL command queue
        """
        self.context = gpu_context
        self.queue = gpu_queue
        self.kernels = {}

        # Load and compile kernels
        self._load_kernels()

        log_info("GPU GA Processor initialized")

    def _load_kernels(self):
        """Load and compile OpenCL kernels for GA operations."""
        try:
            with open('src/ga/ga_operations.cl', 'r') as f:
                kernel_source = f.read()

            program = cl.Program(self.context, kernel_source).build()

            # Load all kernels
            self.kernels['select_survivors'] = program.select_survivors_gpu
            self.kernels['uniform_crossover'] = program.uniform_crossover_gpu
            self.kernels['mutate_population'] = program.mutate_population_gpu
            self.kernels['calculate_fitness'] = program.calculate_fitness_gpu

            log_info("GA kernels compiled successfully")

        except Exception as e:
            log_error(f"Failed to load GA kernels: {e}")
            raise

    def select_survivors_gpu(
        self,
        population: List[CompactBotConfig],
        results: List[BacktestResult],
        survival_rate: float = 0.5,
        survival_threshold: float = 0.0
    ) -> Tuple[List[CompactBotConfig], List[BacktestResult]]:
        """
        GPU-accelerated survivor selection.

        Args:
            population: Current population
            results: Backtest results
            survival_rate: Fraction of population to keep
            survival_threshold: Minimum total_pnl to survive

        Returns:
            Tuple of (surviving_bots, surviving_results)
        """
        try:
            population_size = len(population)
            max_survivors = max(1, int(population_size * survival_rate))

            # Prepare input data
            fitness_scores = np.array([r.fitness_score for r in results], dtype=np.float32)
            total_pnl = np.array([int(r.total_pnl) for r in results], dtype=np.int32)

            # GPU buffers
            fitness_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=fitness_scores)
            pnl_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=total_pnl)

            survivor_indices = np.zeros(max_survivors, dtype=np.int32)
            survivor_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, survivor_indices.nbytes)

            # Execute kernel
            kernel = self.kernels['select_survivors']
            kernel.set_args(
                fitness_buf,
                pnl_buf,
                survivor_buf,
                np.int32(population_size),
                np.int32(max_survivors),
                np.float32(survival_threshold)
            )

            # Run kernel
            global_size = (population_size,)
            local_size = None
            cl.enqueue_nd_range_kernel(self.queue, kernel, global_size, local_size)

            # Read results
            cl.enqueue_copy(self.queue, survivor_indices, survivor_buf)

            # Extract survivors
            surviving_bots = []
            surviving_results = []

            for idx in survivor_indices:
                if idx >= 0 and idx < population_size:
                    surviving_bots.append(population[idx])
                    surviving_results.append(results[idx])

            # If no survivors, keep the best performer (fallback)
            if not surviving_bots:
                best_idx = np.argmax(total_pnl)
                surviving_bots = [population[best_idx]]
                surviving_results = [results[best_idx]]
                log_warning("GPU selection: No survivors found, keeping best performer")

            log_debug(f"GPU selection: {len(surviving_bots)} survivors from {population_size} bots")
            return surviving_bots, surviving_results

        except Exception as e:
            log_warning(f"GPU selection failed, falling back to CPU: {e}")
            return self._cpu_select_survivors(population, results, survival_rate, survival_threshold)

    def _cpu_select_survivors(
        self,
        population: List[CompactBotConfig],
        results: List[BacktestResult],
        survival_rate: float,
        survival_threshold: float
    ) -> Tuple[List[CompactBotConfig], List[BacktestResult]]:
        """CPU fallback for survivor selection."""
        surviving_pairs = []

        for bot, result in zip(population, results):
            if result.total_pnl >= survival_threshold:
                surviving_pairs.append((bot, result))

        if not surviving_pairs:
            all_pairs = list(zip(population, results))
            all_pairs.sort(key=lambda x: x[1].total_pnl, reverse=True)
            surviving_pairs = [all_pairs[0]]
            log_warning("CPU selection: No survivors found, keeping best performer")

        surviving_pairs.sort(key=lambda x: x[1].fitness_score, reverse=True)

        num_survivors = max(1, int(len(population) * survival_rate))
        if len(surviving_pairs) > num_survivors:
            surviving_pairs = surviving_pairs[:num_survivors]

        surviving_bots = [pair[0] for pair in surviving_pairs]
        surviving_results = [pair[1] for pair in surviving_pairs]

        return surviving_bots, surviving_results

    def crossover_population_gpu(
        self,
        parent_pairs: List[Tuple[CompactBotConfig, CompactBotConfig]]
    ) -> List[CompactBotConfig]:
        """
        GPU-accelerated uniform crossover for multiple parent pairs.

        Args:
            parent_pairs: List of (parent1, parent2) tuples

        Returns:
            List of child bots
        """
        try:
            num_pairs = len(parent_pairs)

            # Prepare input data arrays
            p1_indicators = np.zeros((num_pairs, 8), dtype=np.uint8)
            p1_params = np.zeros((num_pairs, 8, 3), dtype=np.float32)
            p1_metadata = np.zeros((num_pairs, 3), dtype=np.int32)  # [num_indicators, risk_bitmap, leverage]
            p1_multipliers = np.zeros((num_pairs, 2), dtype=np.float32)  # [tp_multiplier, sl_multiplier]

            p2_indicators = np.zeros((num_pairs, 8), dtype=np.uint8)
            p2_params = np.zeros((num_pairs, 8, 3), dtype=np.float32)
            p2_metadata = np.zeros((num_pairs, 3), dtype=np.int32)
            p2_multipliers = np.zeros((num_pairs, 2), dtype=np.float32)

            # Fill input arrays
            for i, (p1, p2) in enumerate(parent_pairs):
                # Parent 1
                p1_metadata[i, 0] = p1.num_indicators
                p1_metadata[i, 1] = p1.risk_strategy_bitmap
                p1_metadata[i, 2] = p1.leverage
                p1_multipliers[i, 0] = p1.tp_multiplier
                p1_multipliers[i, 1] = p1.sl_multiplier
                p1_indicators[i, :p1.num_indicators] = p1.indicator_indices[:p1.num_indicators]
                p1_params[i, :p1.num_indicators] = p1.indicator_params[:p1.num_indicators]

                # Parent 2
                p2_metadata[i, 0] = p2.num_indicators
                p2_metadata[i, 1] = p2.risk_strategy_bitmap
                p2_metadata[i, 2] = p2.leverage
                p2_multipliers[i, 0] = p2.tp_multiplier
                p2_multipliers[i, 1] = p2.sl_multiplier
                p2_indicators[i, :p2.num_indicators] = p2.indicator_indices[:p2.num_indicators]
                p2_params[i, :p2.num_indicators] = p2.indicator_params[:p2.num_indicators]

            # Flatten arrays for GPU
            p1_indicators_flat = p1_indicators.flatten()
            p1_params_flat = p1_params.flatten()
            p1_metadata_flat = p1_metadata.flatten()
            p1_multipliers_flat = p1_multipliers.flatten()

            p2_indicators_flat = p2_indicators.flatten()
            p2_params_flat = p2_params.flatten()
            p2_metadata_flat = p2_metadata.flatten()
            p2_multipliers_flat = p2_multipliers.flatten()

            # Output arrays
            child_indicators_flat = np.zeros(num_pairs * 8, dtype=np.uint8)
            child_params_flat = np.zeros(num_pairs * 24, dtype=np.float32)  # 8*3 = 24
            child_metadata_flat = np.zeros(num_pairs * 3, dtype=np.int32)
            child_multipliers_flat = np.zeros(num_pairs * 2, dtype=np.float32)

            # Random seeds
            random_seeds = np.random.randint(0, 2**32, num_pairs, dtype=np.uint32)

            # Create GPU buffers
            buffers = {
                'p1_indicators': cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=p1_indicators_flat),
                'p1_params': cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=p1_params_flat),
                'p1_metadata': cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=p1_metadata_flat),
                'p1_multipliers': cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=p1_multipliers_flat),

                'p2_indicators': cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=p2_indicators_flat),
                'p2_params': cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=p2_params_flat),
                'p2_metadata': cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=p2_metadata_flat),
                'p2_multipliers': cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=p2_multipliers_flat),

                'child_indicators': cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, child_indicators_flat.nbytes),
                'child_params': cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, child_params_flat.nbytes),
                'child_metadata': cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, child_metadata_flat.nbytes),
                'child_multipliers': cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, child_multipliers_flat.nbytes),

                'random_seeds': cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=random_seeds)
            }

            # Execute kernel
            kernel = self.kernels['uniform_crossover']
            kernel.set_args(
                buffers['p1_indicators'], buffers['p1_params'], buffers['p1_metadata'], buffers['p1_multipliers'],
                buffers['p2_indicators'], buffers['p2_params'], buffers['p2_metadata'], buffers['p2_multipliers'],
                buffers['child_indicators'], buffers['child_params'], buffers['child_metadata'], buffers['child_multipliers'],
                buffers['random_seeds'], np.int32(num_pairs)
            )

            # Run kernel
            global_size = (num_pairs,)
            cl.enqueue_nd_range_kernel(self.queue, kernel, global_size, None)

            # Read results
            cl.enqueue_copy(self.queue, child_indicators_flat, buffers['child_indicators'])
            cl.enqueue_copy(self.queue, child_params_flat, buffers['child_params'])
            cl.enqueue_copy(self.queue, child_metadata_flat, buffers['child_metadata'])
            cl.enqueue_copy(self.queue, child_multipliers_flat, buffers['child_multipliers'])

            # Reconstruct child bots
            children = []
            for i in range(num_pairs):
                child_indicators = child_indicators_flat[i*8:(i+1)*8]
                child_params = child_params_flat[i*24:(i+1)*24].reshape(8, 3)
                child_metadata = child_metadata_flat[i*3:(i+1)*3]
                child_multipliers = child_multipliers_flat[i*2:(i+1)*2]

                child = CompactBotConfig(
                    bot_id=-1,  # Will be assigned later
                    num_indicators=child_metadata[0],
                    indicator_indices=child_indicators,
                    indicator_params=child_params,
                    risk_strategy_bitmap=child_metadata[1],
                    tp_multiplier=child_multipliers[0],
                    sl_multiplier=child_multipliers[1],
                    leverage=child_metadata[2]
                )
                children.append(child)

            log_debug(f"GPU crossover: Generated {len(children)} children from {num_pairs} parent pairs")
            return children

        except Exception as e:
            log_warning(f"GPU crossover failed, falling back to CPU: {e}")
            return self._cpu_crossover_population(parent_pairs)

    def _cpu_crossover_population(self, parent_pairs: List[Tuple[CompactBotConfig, CompactBotConfig]]) -> List[CompactBotConfig]:
        """CPU fallback for population crossover."""
        children = []
        for parent1, parent2 in parent_pairs:
            child = self._cpu_crossover(parent1, parent2)
            children.append(child)
        return children

    def _cpu_crossover(self, parent1: CompactBotConfig, parent2: CompactBotConfig) -> CompactBotConfig:
        """CPU uniform crossover implementation."""
        import random

        child = CompactBotConfig(
            bot_id=-1,
            num_indicators=0,
            indicator_indices=np.zeros(8, dtype=np.uint8),
            indicator_params=np.zeros((8, 3), dtype=np.float32),
            risk_strategy_bitmap=0,
            tp_multiplier=0.0,
            sl_multiplier=0.0,
            leverage=1
        )

        min_ind = min(parent1.num_indicators, parent2.num_indicators)
        max_ind = max(parent1.num_indicators, parent2.num_indicators)
        child.num_indicators = random.randint(min_ind, max_ind) if min_ind < max_ind else min_ind

        for i in range(child.num_indicators):
            if random.random() < 0.5:
                if i < parent1.num_indicators:
                    child.indicator_indices[i] = parent1.indicator_indices[i]
                    child.indicator_params[i] = parent1.indicator_params[i]
                else:
                    child.indicator_indices[i] = parent2.indicator_indices[i]
                    child.indicator_params[i] = parent2.indicator_params[i]
            else:
                if i < parent2.num_indicators:
                    child.indicator_indices[i] = parent2.indicator_indices[i]
                    child.indicator_params[i] = parent2.indicator_params[i]
                else:
                    child.indicator_indices[i] = parent1.indicator_indices[i]
                    child.indicator_params[i] = parent1.indicator_params[i]

        child.risk_strategy_bitmap = random.choice([parent1.risk_strategy_bitmap, parent2.risk_strategy_bitmap])
        child.tp_multiplier = random.choice([parent1.tp_multiplier, parent2.tp_multiplier])
        child.sl_multiplier = random.choice([parent1.sl_multiplier, parent2.sl_multiplier])
        child.leverage = random.choice([parent1.leverage, parent2.leverage])

        return child

    def mutate_population_gpu(
        self,
        population: List[CompactBotConfig],
        mutation_rate: float = 0.1
    ) -> List[CompactBotConfig]:
        """
        GPU-accelerated population mutation.

        Args:
            population: Population to mutate
            mutation_rate: Probability of mutation per bot

        Returns:
            Mutated population (in-place modification)
        """
        try:
            population_size = len(population)
            num_mutation_types = 6  # 6 different mutation types

            # Prepare data arrays
            indicators = np.zeros((population_size, 8), dtype=np.uint8)
            params = np.zeros((population_size, 8, 3), dtype=np.float32)
            metadata = np.zeros((population_size, 3), dtype=np.int32)  # [num_indicators, risk_bitmap, leverage]
            multipliers = np.zeros((population_size, 2), dtype=np.float32)  # [tp_multiplier, sl_multiplier]

            # Fill arrays
            for i, bot in enumerate(population):
                metadata[i, 0] = bot.num_indicators
                metadata[i, 1] = bot.risk_strategy_bitmap
                metadata[i, 2] = bot.leverage
                multipliers[i, 0] = bot.tp_multiplier
                multipliers[i, 1] = bot.sl_multiplier
                indicators[i, :bot.num_indicators] = bot.indicator_indices[:bot.num_indicators]
                params[i, :bot.num_indicators] = bot.indicator_params[:bot.num_indicators]

            # Flatten for GPU
            indicators_flat = indicators.flatten()
            params_flat = params.flatten()
            metadata_flat = metadata.flatten()
            multipliers_flat = multipliers.flatten()

            # Random seeds
            random_seeds = np.random.randint(0, 2**32, population_size, dtype=np.uint32)

            # Create GPU buffers
            buffers = {
                'indicators': cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=indicators_flat),
                'params': cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=params_flat),
                'metadata': cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=metadata_flat),
                'multipliers': cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=multipliers_flat),
                'random_seeds': cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=random_seeds)
            }

            # Execute kernel
            kernel = self.kernels['mutate_population']
            kernel.set_args(
                buffers['indicators'], buffers['params'], buffers['metadata'], buffers['multipliers'],
                buffers['random_seeds'], np.int32(population_size),
                np.float32(mutation_rate), np.int32(num_mutation_types)
            )

            # Run kernel
            global_size = (population_size,)
            cl.enqueue_nd_range_kernel(self.queue, kernel, global_size, None)

            # Read results
            cl.enqueue_copy(self.queue, indicators_flat, buffers['indicators'])
            cl.enqueue_copy(self.queue, params_flat, buffers['params'])
            cl.enqueue_copy(self.queue, metadata_flat, buffers['metadata'])
            cl.enqueue_copy(self.queue, multipliers_flat, buffers['multipliers'])

            # Update population in-place
            for i in range(population_size):
                bot = population[i]
                bot.num_indicators = metadata_flat[i*3]
                bot.risk_strategy_bitmap = metadata_flat[i*3 + 1]
                bot.leverage = metadata_flat[i*3 + 2]
                bot.tp_multiplier = multipliers_flat[i*2]
                bot.sl_multiplier = multipliers_flat[i*2 + 1]
                bot.indicator_indices[:bot.num_indicators] = indicators_flat[i*8:i*8 + bot.num_indicators]
                bot.indicator_params[:bot.num_indicators] = params_flat[i*24:i*24 + bot.num_indicators*3].reshape(bot.num_indicators, 3)

            log_debug(f"GPU mutation: Applied mutations to {population_size} bots (rate: {mutation_rate})")
            return population

        except Exception as e:
            log_warning(f"GPU mutation failed, falling back to CPU: {e}")
            return self._cpu_mutate_population(population, mutation_rate)

    def _cpu_mutate_population(self, population: List[CompactBotConfig], mutation_rate: float) -> List[CompactBotConfig]:
        """CPU fallback for population mutation."""
        import random

        for bot in population:
            if random.random() < mutation_rate:
                self._cpu_mutate_bot(bot)
        return population

    def _cpu_mutate_bot(self, bot: CompactBotConfig):
        """CPU single bot mutation implementation."""
        import random

        mutation_type = random.randint(0, 5)

        if mutation_type == 0:
            # Change indicator
            if bot.num_indicators > 0:
                idx = random.randint(0, bot.num_indicators - 1)
                bot.indicator_indices[idx] = random.randint(0, 255)

        elif mutation_type == 1:
            # Adjust parameter
            if bot.num_indicators > 0:
                idx = random.randint(0, bot.num_indicators - 1)
                param_idx = random.randint(0, 2)
                bot.indicator_params[idx][param_idx] *= random.uniform(0.8, 1.2)

        elif mutation_type == 2:
            # Flip risk strategy bit
            bit = random.randint(0, 14)
            bot.risk_strategy_bitmap ^= (1 << bit)

        elif mutation_type == 3:
            # Adjust TP
            bot.tp_multiplier *= random.uniform(0.9, 1.1)
            bot.tp_multiplier = max(0.005, min(0.25, bot.tp_multiplier))

        elif mutation_type == 4:
            # Adjust SL
            bot.sl_multiplier *= random.uniform(0.9, 1.1)
            bot.sl_multiplier = max(0.002, min(bot.tp_multiplier/2, bot.sl_multiplier))

        else:  # mutation_type == 5
            # Adjust leverage
            delta = random.choice([-1, 1]) * random.randint(1, 5)
            bot.leverage = max(1, min(25, bot.leverage + delta))

    def calculate_fitness_gpu(
        self,
        results: List[BacktestResult],
        weights: Optional[dict] = None
    ) -> List[float]:
        """
        GPU-accelerated fitness calculation.

        Args:
            results: Backtest results
            weights: Fitness weights for different metrics

        Returns:
            Fitness scores for each bot
        """
        if weights is None:
            weights = {
                'pnl': 1.0,
                'win_rate': 0.3,
                'sharpe': 0.2,
                'drawdown': 0.1,
                'trades': 0.05
            }

        try:
            population_size = len(results)

            # Prepare input arrays
            total_pnl = np.array([r.total_pnl for r in results], dtype=np.float32)
            win_rate = np.array([r.win_rate for r in results], dtype=np.float32)
            sharpe_ratio = np.array([r.sharpe_ratio for r in results], dtype=np.float32)
            max_drawdown = np.array([r.max_drawdown for r in results], dtype=np.float32)
            total_trades = np.array([r.total_trades for r in results], dtype=np.int32)

            fitness_scores = np.zeros(population_size, dtype=np.float32)

            # GPU buffers
            buffers = {
                'total_pnl': cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=total_pnl),
                'win_rate': cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=win_rate),
                'sharpe_ratio': cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=sharpe_ratio),
                'max_drawdown': cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=max_drawdown),
                'total_trades': cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=total_trades),
                'fitness_scores': cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, fitness_scores.nbytes)
            }

            # Execute kernel
            kernel = self.kernels['calculate_fitness']
            kernel.set_args(
                buffers['total_pnl'], buffers['win_rate'], buffers['sharpe_ratio'],
                buffers['max_drawdown'], buffers['total_trades'], buffers['fitness_scores'],
                np.int32(population_size),
                np.float32(weights['pnl']), np.float32(weights['win_rate']),
                np.float32(weights['sharpe']), np.float32(weights['drawdown']),
                np.float32(weights['trades'])
            )

            # Run kernel
            global_size = (population_size,)
            cl.enqueue_nd_range_kernel(self.queue, kernel, global_size, None)

            # Read results
            cl.enqueue_copy(self.queue, fitness_scores, buffers['fitness_scores'])

            # Update results with fitness scores
            for i, result in enumerate(results):
                result.fitness_score = fitness_scores[i]

            log_debug(f"GPU fitness: Calculated fitness for {population_size} bots")
            return fitness_scores.tolist()

        except Exception as e:
            log_warning(f"GPU fitness calculation failed, falling back to CPU: {e}")
            return self._cpu_calculate_fitness(results, weights)

    def _cpu_calculate_fitness(self, results: List[BacktestResult], weights: dict) -> List[float]:
        """CPU fallback for fitness calculation."""
        fitness_scores = []

        for result in results:
            fitness = (weights['pnl'] * result.total_pnl +
                      weights['win_rate'] * result.win_rate +
                      weights['sharpe'] * result.sharpe_ratio -
                      weights['drawdown'] * result.max_drawdown +
                      weights['trades'] * result.total_trades)
            result.fitness_score = fitness
            fitness_scores.append(fitness)

        return fitness_scores