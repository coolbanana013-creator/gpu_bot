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
import csv

from ..utils.validation import log_info, log_warning, log_error, log_debug
from ..bot_generator.compact_generator import CompactBotConfig
from ..backtester.compact_simulator import BacktestResult
from ..indicators.factory import IndicatorFactory
from ..indicators.gpu_indicators import get_all_gpu_indicators, get_gpu_indicator_name, GPU_INDICATOR_COUNT


# Risk strategy names corresponding to enum values (0-14)
RISK_STRATEGY_NAMES = [
    'FixedPct',         # 0
    'FixedUSD',         # 1
    'KellyFull',        # 2
    'KellyHalf',        # 3
    'KellyQuarter',     # 4
    'ATRMultiplier',    # 5
    'VolatilityPct',    # 6
    'EquityCurve',      # 7
    'FixedRiskReward',  # 8
    'Martingale',       # 9
    'AntiMartingale',   # 10
    'FixedRatio',       # 11
    'PercentVolatility',# 12
    'WilliamsFixed',    # 13
    'OptimalF'          # 14
    'Reserved9',     # bit 9
    'Reserved10',    # bit 10
    'Reserved11',    # bit 11
    'Reserved12',    # bit 12
    'Reserved13',    # bit 13
    'Reserved14'     # bit 14
]


def decode_risk_strategy(strategy_enum: int, risk_param: float) -> str:
    """
    Decode risk strategy enum to human-readable string with parameter.
    
    Args:
        strategy_enum: Risk strategy enum value (0-14)
        risk_param: Risk parameter value
        
    Returns:
        Strategy name with parameter
    """
    if 0 <= strategy_enum < len(RISK_STRATEGY_NAMES):
        strategy_name = RISK_STRATEGY_NAMES[strategy_enum]
        return f"{strategy_name}({risk_param:.4f})"
    return f"Unknown({strategy_enum})"


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
            self.kernels['serialize_binary'] = program.serialize_bot_data_binary

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
        """
        start_time = time.time()

        num_bots = len(bots)
        if num_bots == 0:
            return

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Use GPU logging exclusively
        self._gpu_logging_attempt(generation, bots, results, initial_balance, num_cycles, output_dir)

        elapsed = time.time() - start_time
        log_info(f"GPU logging completed for generation {generation} in {elapsed:.3f}s")

    def _gpu_logging_attempt(
        self,
        generation: int,
        bots: List[CompactBotConfig],
        results: List[BacktestResult],
        initial_balance: float,
        num_cycles: int,
        output_dir: str
    ) -> None:
        """
        GPU-accelerated logging using binary serialization.
        """
        data_arrays = self._prepare_data_arrays(bots, results, num_cycles)
        binary_data = self._serialize_bot_data_gpu(data_arrays, generation, initial_balance, num_cycles)
        csv_data = self._format_csv_from_binary(binary_data, bots, results, generation, initial_balance, num_cycles)

        csv_file = os.path.join(output_dir, f"generation_{generation}.csv")
        self._write_csv_async(csv_file, csv_data)

    def _serialize_bot_data_gpu(self, data_arrays, generation, initial_balance, num_cycles):
        """Serialize bot data to binary format using GPU."""
        num_bots = len(data_arrays['bot_ids'])

        # Calculate buffer size (BotData struct + per-cycle data)
        # BotData struct: generation(4) + bot_id(4) + num_indicators(4) + leverage(4) + total_trades(4) + 
        #                total_pnl(4) + win_rate(4) + final_balance(4) + fitness_score(4) + sharpe_ratio(4) + 
        #                max_drawdown(4) + indicator_indices[8](8) = 52 bytes
        bot_data_size = 4 * 11 + 8  # 11 ints (44 bytes) + 8 indicator indices = 52 bytes
        per_cycle_size = num_cycles * 3 * 4  # 3 floats per cycle
        record_size = bot_data_size + per_cycle_size
        total_size = num_bots * record_size

        # Create output buffer
        output_buffer = np.zeros(total_size, dtype=np.uint8)
        output_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, output_buffer.nbytes)

        # Create input buffers
        buffers = {}
        for key, array in data_arrays.items():
            buffers[key] = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                   hostbuf=array)

        # Execute serialization kernel
        kernel = self.kernels['serialize_binary']
        kernel.set_args(
            buffers['bot_ids'], buffers['num_indicators'], buffers['indicator_indices'],
            buffers['leverages'], buffers['total_pnls'],
            buffers['win_rates'], buffers['total_trades'], buffers['final_balances'],
            buffers['fitness_scores'], buffers['sharpe_ratios'], buffers['max_drawdowns'],
            buffers['per_cycle_trades'], buffers['per_cycle_pnls'], buffers['per_cycle_winrates'],
            output_buf,
            np.int32(num_bots), np.int32(num_cycles), np.int32(generation), np.float32(initial_balance)
        )

        cl.enqueue_nd_range_kernel(self.queue, kernel, (num_bots,), None)

        # Read results
        cl.enqueue_copy(self.queue, output_buffer, output_buf)

        return output_buffer, record_size

    def _format_csv_from_binary(self, binary_data, bots, results, generation, initial_balance, num_cycles):
        """Format binary data to CSV string on CPU."""
        output_buffer, record_size = binary_data
        num_bots = len(bots)
        
        # Get indicator names for mapping (GPU indicators 0-49)
        indicator_names = [get_gpu_indicator_name(i) for i in range(GPU_INDICATOR_COUNT)]

        # Build CSV header - Reorganized: Averages, Totals, Config, Cycle Details
        header = [
            # Filters
            'AllCyclesPositive', 'AllCyclesHaveTrades',
            # IDs
            'Generation', 'BotID', 'SurvivedGenerations',
            # Totals (aggregate across all cycles)
            'TotalProfitPct', 'TotalWinRate', 'TotalTrades',
            # Per-cycle averages (excluding zero-trade cycles)
            'AvgProfitPctPerCycle', 'AvgWinRatePerCycle', 'AvgTradesPerCycle',
            # Scoring metrics
            'FitnessScore', 'SharpeRatio', 'MaxDrawdown',
            # Final stats
            'TotalPnL', 'FinalBalance',
            # Configuration
            'NumIndicators', 'Leverage', 'TPMultiplier', 'SLMultiplier', 'RiskStrategies',
            'NumCycles', 'IndicatorsUsed', 'IndicatorParams'
        ]
        for i in range(num_cycles):
            header.extend([f'Cycle{i}_Trades', f'Cycle{i}_ProfitPct', f'Cycle{i}_TotalPnL', f'Cycle{i}_WinRate'])
        csv_lines = [';'.join(header)]

        # Process each bot's binary data
        for bot_idx in range(num_bots):
            bot = bots[bot_idx]
            result = results[bot_idx]

            # Parse binary data (little-endian)
            offset = bot_idx * record_size
            data_view = output_buffer[offset:offset + record_size].view(dtype=np.uint8)

            # Extract bot data (struct layout)
            bot_data_offset = 0
            generation_val = np.frombuffer(data_view[bot_data_offset:bot_data_offset+4], dtype=np.int32)[0]
            bot_data_offset += 4
            bot_id_from_binary = np.frombuffer(data_view[bot_data_offset:bot_data_offset+4], dtype=np.int32)[0]
            bot_data_offset += 4
            num_indicators = np.frombuffer(data_view[bot_data_offset:bot_data_offset+4], dtype=np.int32)[0]
            bot_data_offset += 4
            leverage = np.frombuffer(data_view[bot_data_offset:bot_data_offset+4], dtype=np.int32)[0]
            bot_data_offset += 4
            
            # NOTE: survival_generations NOT in binary buffer - read directly from bot object
            # This ensures it's always accurate and not corrupted by binary serialization
            survival_generations = max(0, getattr(bot, 'survival_generations', 0))
            bot_id = bot.bot_id  # Use the correct bot_id from bot object, not binary
            
            # Extract total_trades (int) - this was being skipped, causing offset misalignment!
            total_trades_val = np.frombuffer(data_view[bot_data_offset:bot_data_offset+4], dtype=np.int32)[0]
            bot_data_offset += 4
            
            total_pnl = np.frombuffer(data_view[bot_data_offset:bot_data_offset+4], dtype=np.float32)[0]
            bot_data_offset += 4
            win_rate = np.frombuffer(data_view[bot_data_offset:bot_data_offset+4], dtype=np.float32)[0]
            bot_data_offset += 4
            final_balance = np.frombuffer(data_view[bot_data_offset:bot_data_offset+4], dtype=np.float32)[0]
            bot_data_offset += 4
            fitness_score = np.frombuffer(data_view[bot_data_offset:bot_data_offset+4], dtype=np.float32)[0]
            bot_data_offset += 4
            sharpe_ratio = np.frombuffer(data_view[bot_data_offset:bot_data_offset+4], dtype=np.float32)[0]
            bot_data_offset += 4
            max_drawdown = np.frombuffer(data_view[bot_data_offset:bot_data_offset+4], dtype=np.float32)[0]
            bot_data_offset += 4

            # Extract indicator indices
            indicator_indices = data_view[bot_data_offset:bot_data_offset+8]
            bot_data_offset += 8

            # Extract per-cycle data
            per_cycle_data = []
            for c in range(num_cycles):
                trades = np.frombuffer(data_view[bot_data_offset:bot_data_offset+4], dtype=np.float32)[0]
                bot_data_offset += 4
                pnl = np.frombuffer(data_view[bot_data_offset:bot_data_offset+4], dtype=np.float32)[0]
                bot_data_offset += 4
                wins_count = np.frombuffer(data_view[bot_data_offset:bot_data_offset+4], dtype=np.float32)[0]  # WINS COUNT not ratio
                bot_data_offset += 4
                per_cycle_data.extend([trades, pnl, wins_count])

            # Use values directly from result object (already calculated correctly)
            total_profit_pct = (result.total_pnl / initial_balance) * 100 if initial_balance != 0 else 0.0
            total_win_rate = result.win_rate  # Already percentage 0-100 from GPU
            total_trades_val = result.total_trades
            total_pnl_val = result.total_pnl
            sharpe_ratio_val = result.sharpe_ratio
            max_drawdown_val = result.max_drawdown * 100  # Convert from decimal to percentage
            
            # Extract cycle data from per_cycle_data for cycle-specific checks
            cycle_pnls = []
            cycle_trades = []
            cycle_wins = []
            
            for c in range(num_cycles):
                idx = c * 3
                c_trades = per_cycle_data[idx]
                c_pnl = per_cycle_data[idx + 1]
                c_wins = per_cycle_data[idx + 2]
                cycle_trades.append(c_trades)
                cycle_pnls.append(c_pnl)
                cycle_wins.append(c_wins)
            
            # Check if ALL cycles have positive profit percentage
            all_cycles_positive = all((pnl / initial_balance) * 100 > 0 for pnl in cycle_pnls) if cycle_pnls else False
            
            # Check if ALL cycles have at least 1 trade
            all_cycles_have_trades = all(trades > 0 for trades in cycle_trades) if cycle_trades else False
            
            # Calculate TRUE per-cycle averages (only cycles with trades)
            active_cycles = [i for i in range(num_cycles) if cycle_trades[i] > 0]
            num_active_cycles = len(active_cycles)
            
            if num_active_cycles > 0:
                # Average profit % per active cycle
                avg_profit_pct_per_cycle = sum((cycle_pnls[i] / initial_balance) * 100 for i in active_cycles) / num_active_cycles
                
                # Average win rate per active cycle
                cycle_winrates = [(cycle_wins[i] / cycle_trades[i] * 100.0) if cycle_trades[i] > 0 else 0.0 for i in active_cycles]
                avg_winrate_per_cycle = sum(cycle_winrates) / num_active_cycles
                
                # Average trades per active cycle
                avg_trades_per_cycle = sum(cycle_trades[i] for i in active_cycles) / num_active_cycles
            else:
                avg_profit_pct_per_cycle = 0.0
                avg_winrate_per_cycle = 0.0
                avg_trades_per_cycle = 0.0

            # Format as CSV row
            # Safety: Ensure num_indicators doesn't exceed array bounds
            safe_num_indicators = min(bot.num_indicators, len(bot.indicator_indices), len(bot.indicator_params))
            
            indicators_used = [indicator_names[idx] for idx in bot.indicator_indices[:safe_num_indicators] if idx < len(indicator_names)]
            indicators_str = ', '.join(indicators_used)
            
            # Format indicator parameters
            params_list = []
            for i in range(safe_num_indicators):
                ind_name = indicator_names[bot.indicator_indices[i]] if bot.indicator_indices[i] < len(indicator_names) else f"Ind{bot.indicator_indices[i]}"
                params = bot.indicator_params[i]
                # Format as: IndicatorName(param0, param1, param2)
                params_str = f"{ind_name}({params[0]:.1f},{params[1]:.1f},{params[2]:.1f})"
                params_list.append(params_str)
            indicator_params_str = ' | '.join(params_list)
            
            # Decode risk strategy from enum (use first indicator's strategy)
            risk_strategies_str = decode_risk_strategy(bot.indicator_risk_strategies[0], bot.risk_param)

            row = [
                # Filters
                str(all_cycles_positive).lower(),
                str(all_cycles_have_trades).lower(),
                # IDs
                str(generation),
                str(bot_id),
                str(survival_generations),
                # TOTALS (aggregate across ALL cycles)
                f"{total_profit_pct:.2f}".replace('.', ','),
                f"{total_win_rate:.2f}".replace('.', ','),  # Total win rate percentage
                str(int(total_trades_val)),
                # PER-CYCLE AVERAGES (only active cycles with trades)
                f"{avg_profit_pct_per_cycle:.2f}".replace('.', ','),
                f"{avg_winrate_per_cycle:.2f}".replace('.', ','),
                f"{avg_trades_per_cycle:.2f}".replace('.', ','),
                # Scoring metrics
                f"{fitness_score:.2f}".replace('.', ','),
                f"{sharpe_ratio_val:.2f}".replace('.', ','),
                f"{max_drawdown_val:.2f}".replace('.', ','),
                # Final stats
                f"{total_pnl_val:.2f}".replace('.', ','),
                f"{final_balance:.2f}".replace('.', ','),
                # Configuration
                str(num_indicators),
                str(leverage),
                f"{bot.tp_multiplier:.2f}".replace('.', ','),
                f"{bot.sl_multiplier:.2f}".replace('.', ','),
                risk_strategies_str,
                str(num_cycles),
                indicators_str,
                indicator_params_str
            ]

            # Add per-cycle data
            for c in range(num_cycles):
                base_idx = c * 3
                c_trades = int(per_cycle_data[base_idx])
                c_pnl = per_cycle_data[base_idx + 1]
                c_wins = per_cycle_data[base_idx + 2]  # This is wins count, not winrate
                
                # Calculate cycle-specific metrics
                c_profit_pct = (c_pnl / initial_balance) * 100 if initial_balance != 0 else 0.0
                c_winrate = (c_wins / c_trades * 100.0) if c_trades > 0 else 0.0  # Calculate actual winrate percentage
                
                # Show actual profit percentage (no artificial limits)

                row.extend([
                    str(c_trades),
                    f"{c_profit_pct:.2f}".replace('.', ','),
                    f"{c_pnl:.2f}".replace('.', ','),  # Add TotalPnL for each cycle
                    f"{c_winrate:.2f}".replace('.', ',')
                ])

            csv_lines.append(';'.join(row))

        return '\n'.join(csv_lines) + '\n'

    def _prepare_data_arrays(self, bots: List[CompactBotConfig], results: List[BacktestResult], num_cycles: int):
        """Prepare numpy arrays for GPU processing."""
        num_bots = len(bots)

        # Basic bot data
        bot_ids = np.array([bot.bot_id for bot in bots], dtype=np.int32)
        num_indicators = np.array([bot.num_indicators for bot in bots], dtype=np.int32)
        leverages = np.array([bot.leverage for bot in bots], dtype=np.int32)
        # NOTE: survival_generations NOT sent to GPU - read from bot objects during CSV formatting
        
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
        per_cycle_winrates = np.zeros(num_bots * num_cycles, dtype=np.float32)  # Actually stores wins COUNT

        # Indicator indices (8 bytes per bot)
        indicator_indices = np.zeros((num_bots, 8), dtype=np.uint8)
        for i, bot in enumerate(bots):
            for j in range(min(8, len(bot.indicator_indices))):
                indicator_indices[i, j] = bot.indicator_indices[j]

        for i, result in enumerate(results):
            # Safely extract per-cycle data
            for c in range(num_cycles):
                idx = i * num_cycles + c
                try:
                    per_cycle_trades[idx] = result.per_cycle_trades[c]
                    per_cycle_pnls[idx] = result.per_cycle_pnl[c]
                    # CRITICAL FIX: Store WINS COUNT not winrate ratio (matches GPU kernel output)
                    per_cycle_winrates[idx] = result.per_cycle_wins[c]
                except (IndexError, AttributeError):
                    per_cycle_trades[idx] = 0
                    per_cycle_pnls[idx] = 0.0
                    per_cycle_winrates[idx] = 0.0

        return {
            'bot_ids': bot_ids,
            'num_indicators': num_indicators,
            'indicator_indices': indicator_indices.flatten(),
            'leverages': leverages,
            # NOTE: survival_generations NOT included - read from bot objects during formatting
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

        # Use the binary serialization kernel (CSV formatting is done on CPU from binary data)
        binary_kernel = self.kernels['serialize_binary']
        binary_kernel.set_args(
            buffers['bot_ids'], buffers['num_indicators'], buffers['indicator_indices'],
            buffers['leverages'], buffers['total_pnls'],
            buffers['win_rates'], buffers['total_trades'], buffers['final_balances'],
            buffers['fitness_scores'], buffers['sharpe_ratios'], buffers['max_drawdowns'],
            buffers['per_cycle_trades'], buffers['per_cycle_pnls'], buffers['per_cycle_winrates'],
            output_buf,
            np.int32(num_bots), np.int32(num_cycles), np.int32(generation), np.float32(initial_balance)
        )

        cl.enqueue_nd_range_kernel(self.queue, binary_kernel, (num_bots,), None)

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



    def shutdown(self):
        """Shutdown the logging processor."""
        log_info("Shutting down GPU Logging Processor")
        self.file_executor.shutdown(wait=True)