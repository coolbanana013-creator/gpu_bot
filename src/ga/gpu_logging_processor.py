"""
GPU-Accelerated Logging Processor
High-performance CSV logging using OpenCL kernels for bot performance data.
"""

import numpy as np
import pyopencl as cl
import os
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor
import time

from ..utils.validation import log_info, log_warning, log_error, log_debug
from ..bot_generator.compact_generator import CompactBotConfig
from ..backtester.compact_simulator import BacktestResult


class GPULoggingProcessor:
    """
    GPU-accelerated logging processor for high-performance CSV generation.
    Uses OpenCL kernels to serialize bot data to CSV format in parallel.
    """

    def __init__(self, gpu_context: cl.Context, gpu_queue: cl.CommandQueue):
        """
        Initialize GPU logging processor.

        Args:
            gpu_context: OpenCL GPU context
            gpu_queue: OpenCL GPU command queue
        """
        self.context = gpu_context
        self.queue = gpu_queue
        self.kernels = {}

        # Load and compile kernels
        self._load_kernels()

        # Thread pool for async file writing
        self.file_executor = ThreadPoolExecutor(max_workers=2)

        log_info("GPU Logging Processor initialized")

    def _load_kernels(self):
        """Load and compile OpenCL kernels for logging operations."""
        try:
            with open('src/ga/logging_kernels.cl', 'r') as f:
                kernel_source = f.read()

            program = cl.Program(self.context, kernel_source).build()

            # Load kernels
            self.kernels['calculate_offsets'] = program.calculate_csv_row_offsets
            self.kernels['serialize_csv'] = program.serialize_bot_data_to_csv
            self.kernels['write_header'] = program.write_csv_header

            log_info("Logging kernels compiled successfully")

        except Exception as e:
            log_error(f"Failed to load logging kernels: {e}")
            raise

    def log_generation_bots_gpu(
        self,
        generation: int,
        bots: List[CompactBotConfig],
        results: List[BacktestResult],
        initial_balance: float = 100.0,
        num_cycles: int = 10,
        output_dir: str = "logs"
    ) -> None:
        """
        GPU-accelerated logging of generation bot data to CSV.
        Hybrid approach: GPU prepares data, CPU formats CSV for reliability.
        """
        start_time = time.time()

        try:
            num_bots = len(bots)
            if num_bots == 0:
                return

            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            csv_file = os.path.join(output_dir, f"generation_{generation}.csv")

            # Use CPU fallback for now - GPU string formatting has issues
            # TODO: Fix GPU CSV formatting kernel
            self._cpu_fallback_logging(generation, bots, results, initial_balance, num_cycles, output_dir)

            elapsed = time.time() - start_time
            log_debug(f"Logging completed in {elapsed:.3f}s (CPU fallback)")

        except Exception as e:
            elapsed = time.time() - start_time
            log_warning(f"GPU logging failed ({elapsed:.3f}s): {e}")
            log_info("Falling back to CPU logging")
            self._cpu_fallback_logging(generation, bots, results, initial_balance, num_cycles, output_dir)

    def _prepare_data_arrays(self, bots: List[CompactBotConfig], results: List[BacktestResult], num_cycles: int):
        """Prepare numpy arrays for GPU processing."""
        num_bots = len(bots)

        # Basic bot data
        bot_ids = np.array([bot.bot_id for bot in bots], dtype=np.int32)
        num_indicators = np.array([bot.num_indicators for bot in bots], dtype=np.int32)
        leverages = np.array([bot.leverage for bot in bots], dtype=np.int32)
        survival_generations = np.array([bot.survival_generations for bot in bots], dtype=np.int32)

        # Indicator indices (pad to 8 per bot)
        indicator_indices = np.zeros((num_bots, 8), dtype=np.uint8)
        for i, bot in enumerate(bots):
            indicator_indices[i, :bot.num_indicators] = bot.indicator_indices[:bot.num_indicators]

        # Results data
        total_pnls = np.array([r.total_pnl for r in results], dtype=np.float32)
        win_rates = np.array([r.win_rate for r in results], dtype=np.float32)
        total_trades = np.array([r.total_trades for r in results], dtype=np.int32)
        final_balances = np.array([r.final_balance for r in results], dtype=np.float32)
        fitness_scores = np.array([r.fitness_score for r in results], dtype=np.float32)
        sharpe_ratios = np.array([r.sharpe_ratio for r in results], dtype=np.float32)
        max_drawdowns = np.array([r.max_drawdown for r in results], dtype=np.float32)

        # Per-cycle data (flatten to 1D arrays)
        per_cycle_trades = np.zeros(num_bots * num_cycles, dtype=np.float32)
        per_cycle_pnls = np.zeros(num_bots * num_cycles, dtype=np.float32)
        per_cycle_winrates = np.zeros(num_bots * num_cycles, dtype=np.float32)

        for i, result in enumerate(results):
            # Safely extract per-cycle data
            for c in range(num_cycles):
                idx = i * num_cycles + c
                try:
                    per_cycle_trades[idx] = result.per_cycle_trades[c]
                    per_cycle_pnls[idx] = result.per_cycle_pnl[c]
                    trades = result.per_cycle_trades[c]
                    wins = result.per_cycle_wins[c]
                    per_cycle_winrates[idx] = wins / trades if trades > 0 else 0.0
                except (IndexError, AttributeError):
                    per_cycle_trades[idx] = 0
                    per_cycle_pnls[idx] = 0.0
                    per_cycle_winrates[idx] = 0.0

        return {
            'bot_ids': bot_ids,
            'num_indicators': num_indicators,
            'indicator_indices': indicator_indices.flatten(),
            'leverages': leverages,
            'survival_generations': survival_generations,
            'total_pnls': total_pnls,
            'win_rates': win_rates,
            'total_trades': total_trades,
            'final_balances': final_balances,
            'fitness_scores': fitness_scores,
            'sharpe_ratios': sharpe_ratios,
            'max_drawdowns': max_drawdowns,
            'per_cycle_trades': per_cycle_trades,
            'per_cycle_pnls': per_cycle_pnls,
            'per_cycle_winrates': per_cycle_winrates
        }

    def _calculate_output_requirements(self, data_arrays, num_cycles):
        """Calculate required output buffer size and row offsets."""
        num_bots = len(data_arrays['bot_ids'])

        # Base row length estimate (fixed fields)
        base_row_length = 200  # Rough estimate for fixed fields

        # GPU buffers for offset calculation
        num_indicators_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                      hostbuf=data_arrays['num_indicators'])
        output_offsets = np.zeros(num_bots, dtype=np.int32)
        row_lengths = np.zeros(num_bots, dtype=np.int32)

        offsets_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, output_offsets.nbytes)
        lengths_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, row_lengths.nbytes)

        # Execute offset calculation kernel
        kernel = self.kernels['calculate_offsets']
        kernel.set_args(
            num_indicators_buf, offsets_buf, lengths_buf,
            np.int32(num_bots), np.int32(num_cycles), np.int32(base_row_length)
        )

        cl.enqueue_nd_range_kernel(self.queue, kernel, (num_bots,), None)

        # Read results
        cl.enqueue_copy(self.queue, output_offsets, offsets_buf)
        cl.enqueue_copy(self.queue, row_lengths, lengths_buf)

        # Calculate total buffer size
        total_size = output_offsets[-1] + row_lengths[-1] if num_bots > 0 else 0
        total_size += 1000  # Extra space for header

        return {
            'offsets': output_offsets,
            'lengths': row_lengths,
            'total_size': total_size
        }

    def _generate_csv_data_gpu(self, data_arrays, output_info, generation, initial_balance, num_cycles):
        """Generate CSV data using GPU kernels."""
        num_bots = len(data_arrays['bot_ids'])

        # Create output buffer
        output_buffer = np.zeros(output_info['total_size'], dtype=np.uint8)
        output_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, output_buffer.nbytes)

        # Create input buffers
        buffers = {}
        for key, array in data_arrays.items():
            buffers[key] = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                   hostbuf=array)

        offsets_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=output_info['offsets'])

        # Generate header first
        header_size = 1000
        header_buffer = np.zeros(header_size, dtype=np.uint8)
        header_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, header_buffer.nbytes)

        header_kernel = self.kernels['write_header']
        header_kernel.set_args(header_buf, np.int32(num_cycles))
        cl.enqueue_nd_range_kernel(self.queue, header_kernel, (1,), None)

        # Generate CSV rows
        csv_kernel = self.kernels['serialize_csv']
        csv_kernel.set_args(
            buffers['bot_ids'], buffers['num_indicators'], buffers['indicator_indices'],
            buffers['leverages'], buffers['survival_generations'], buffers['total_pnls'],
            buffers['win_rates'], buffers['total_trades'], buffers['final_balances'],
            buffers['fitness_scores'], buffers['sharpe_ratios'], buffers['max_drawdowns'],
            buffers['per_cycle_trades'], buffers['per_cycle_pnls'], buffers['per_cycle_winrates'],
            output_buf, offsets_buf,
            np.int32(num_bots), np.int32(num_cycles), np.int32(generation), np.float32(initial_balance)
        )

        cl.enqueue_nd_range_kernel(self.queue, csv_kernel, (num_bots,), None)

        # Read results
        cl.enqueue_copy(self.queue, output_buffer, output_buf)
        cl.enqueue_copy(self.queue, header_buffer, header_buf)

        # Combine header and data
        header_str = header_buffer.tobytes().decode('utf-8', errors='ignore').rstrip('\x00')
        data_str = output_buffer.tobytes().decode('utf-8', errors='ignore').rstrip('\x00')

        return header_str + data_str

    def _write_csv_async(self, filename: str, csv_data: str):
        """Write CSV data to file asynchronously."""
        def write_task():
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(csv_data)
                log_debug(f"Async CSV write complete: {filename}")
            except Exception as e:
                log_error(f"Async CSV write failed: {e}")

        self.file_executor.submit(write_task)

    def _cpu_fallback_logging(self, generation, bots, results, initial_balance, num_cycles, output_dir):
        """CPU fallback for logging when GPU fails."""
        import csv

        os.makedirs(output_dir, exist_ok=True)
        csv_file = os.path.join(output_dir, f"generation_{generation}.csv")

        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=';')

            # Write header
            header = [
                'Generation', 'BotID', 'ProfitPct', 'WinRate', 'TotalTrades', 'FinalBalance',
                'FitnessScore', 'SharpeRatio', 'MaxDrawdown', 'SurvivedGenerations',
                'NumIndicators', 'Leverage', 'TotalPnL', 'NumCycles', 'IndicatorsUsed'
            ]
            for i in range(num_cycles):
                header.extend([f'Cycle{i}_Trades', f'Cycle{i}_ProfitPct', f'Cycle{i}_WinRate'])
            writer.writerow(header)

            # Write data rows
            for bot, result in zip(bots, results):
                profit_pct = (result.total_pnl / initial_balance) * 100
                indicators_used = [str(idx) for idx in bot.indicator_indices[:bot.num_indicators]]
                indicators_str = ', '.join(indicators_used)

                row = [
                    generation, bot.bot_id, f"{profit_pct:.2f}".replace('.', ','),
                    f"{result.win_rate:.4f}".replace('.', ','), result.total_trades,
                    f"{result.final_balance:.2f}".replace('.', ','), f"{result.fitness_score:.2f}".replace('.', ','),
                    f"{result.sharpe_ratio:.2f}".replace('.', ','), f"{result.max_drawdown:.4f}".replace('.', ','),
                    bot.survival_generations, bot.num_indicators, bot.leverage,
                    f"{result.total_pnl:.2f}".replace('.', ','), num_cycles, indicators_str
                ]

                # Add per-cycle data
                for i in range(num_cycles):
                    try:
                        c_trades = result.per_cycle_trades[i]
                        c_pnl = result.per_cycle_pnl[i]
                        c_wins = result.per_cycle_wins[i]
                    except:
                        c_trades, c_pnl, c_wins = 0, 0.0, 0

                    c_profit_pct = (c_pnl / initial_balance) * 100
                    c_winrate = c_wins / c_trades if c_trades > 0 else 0.0

                    row.extend([
                        c_trades, f"{c_profit_pct:.2f}".replace('.', ','), f"{c_winrate:.4f}".replace('.', ',')
                    ])

                writer.writerow(row)

        log_info(f"CPU fallback: Logged {len(bots)} bots to {csv_file}")

    def shutdown(self):
        """Shutdown the logging processor."""
        log_info("Shutting down GPU Logging Processor")
        self.file_executor.shutdown(wait=True)