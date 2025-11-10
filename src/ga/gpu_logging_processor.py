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


# Risk strategy names corresponding to bitmap bits
RISK_STRATEGY_NAMES = [
    'FixedUSD',      # bit 0
    'PctBalance',    # bit 1
    'Kelly',         # bit 2
    'Martingale',    # bit 3
    'AntiMartingale',# bit 4
    'Volatility',    # bit 5
    'ATR',           # bit 6
    'Momentum',      # bit 7
    'Reserved8',     # bit 8
    'Reserved9',     # bit 9
    'Reserved10',    # bit 10
    'Reserved11',    # bit 11
    'Reserved12',    # bit 12
    'Reserved13',    # bit 13
    'Reserved14'     # bit 14
]


def decode_risk_strategies(bitmap: int) -> str:
    """
    Decode risk strategy bitmap to human-readable string.
    
    Args:
        bitmap: Risk strategy bitmap (15-bit)
        
    Returns:
        Comma-separated list of active strategy names
    """
    active_strategies = []
    for bit_idx, strategy_name in enumerate(RISK_STRATEGY_NAMES):
        if bitmap & (1 << bit_idx):
            active_strategies.append(strategy_name)
    return ', '.join(active_strategies) if active_strategies else 'None'


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
        
        # Get indicator names for mapping
        all_indicator_types = IndicatorFactory.get_all_indicator_types()
        indicator_names = [indicator_type.value for indicator_type in all_indicator_types]

        # Build CSV header - Reorganized: Averages, Totals, Config, Cycle Details
        header = [
            # Filters
            'AllCyclesPositive', 'AllCyclesHaveTrades',
            # IDs
            'Generation', 'BotID', 'SurvivedGenerations',
            # Averages (per cycle)
            'AvgProfitPct', 'AvgWinRate', 'AvgTradesPerCycle',
            'FitnessScore', 'SharpeRatio', 'AvgDrawdown',
            # Totals (across all cycles)
            'TotalPnL', 'TotalTrades', 'FinalBalance',
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
                winrate = np.frombuffer(data_view[bot_data_offset:bot_data_offset+4], dtype=np.float32)[0]
                bot_data_offset += 4
                per_cycle_data.extend([trades, pnl, winrate])

            # Calculate AVERAGES across cycles for main columns
            avg_profit_pct = 0.0
            avg_win_rate = 0.0
            avg_total_trades = 0
            avg_total_pnl = 0.0
            avg_sharpe_ratio = 0.0
            avg_max_drawdown = 0.0
            all_cycles_positive = False
            all_cycles_have_trades = False
            
            if num_cycles > 0:
                # Average across all cycles
                cycle_pnls = result.per_cycle_pnl if hasattr(result, 'per_cycle_pnl') and result.per_cycle_pnl else []
                cycle_trades = result.per_cycle_trades if hasattr(result, 'per_cycle_trades') and result.per_cycle_trades else []
                cycle_wins = result.per_cycle_wins if hasattr(result, 'per_cycle_wins') and result.per_cycle_wins else []
                
                # Calculate averages
                avg_total_pnl = sum(cycle_pnls) / num_cycles if cycle_pnls else 0.0
                avg_total_trades = int(sum(cycle_trades) / num_cycles) if cycle_trades else 0
                total_wins = sum(cycle_wins)
                total_trades_sum = sum(cycle_trades)
                
                avg_profit_pct = (avg_total_pnl / initial_balance) * 100 if initial_balance != 0 else 0.0
                avg_win_rate = total_wins / total_trades_sum if total_trades_sum > 0 else 0.0
                avg_sharpe_ratio = result.sharpe_ratio  # Already an aggregate metric
                avg_max_drawdown = result.max_drawdown  # Worst case across all cycles
                
                # Check if ALL cycles have positive profit percentage
                if cycle_pnls:
                    all_cycles_positive = all((pnl / initial_balance) * 100 > 0 for pnl in cycle_pnls)
                
                # Check if ALL cycles have at least 1 trade
                if cycle_trades:
                    all_cycles_have_trades = all(trades > 0 for trades in cycle_trades)

            # Format as CSV row
            indicators_used = [indicator_names[idx] for idx in bot.indicator_indices[:bot.num_indicators] if idx < len(indicator_names)]
            indicators_str = ', '.join(indicators_used)
            
            # Format indicator parameters
            params_list = []
            for i in range(bot.num_indicators):
                ind_name = indicator_names[bot.indicator_indices[i]] if bot.indicator_indices[i] < len(indicator_names) else f"Ind{bot.indicator_indices[i]}"
                params = bot.indicator_params[i]
                # Format as: IndicatorName(param0, param1, param2)
                params_str = f"{ind_name}({params[0]:.1f},{params[1]:.1f},{params[2]:.1f})"
                params_list.append(params_str)
            indicator_params_str = ' | '.join(params_list)
            
            # Decode risk strategies from bitmap
            risk_strategies_str = decode_risk_strategies(bot.risk_strategy_bitmap)

            row = [
                # Filters
                str(all_cycles_positive).lower(),
                str(all_cycles_have_trades).lower(),
                # IDs
                str(generation),
                str(bot_id),
                str(survival_generations),
                # Averages
                f"{avg_profit_pct:.2f}".replace('.', ','),
                f"{avg_win_rate:.4f}".replace('.', ','),
                str(avg_total_trades),
                f"{fitness_score:.2f}".replace('.', ','),
                f"{avg_sharpe_ratio:.2f}".replace('.', ','),
                f"{avg_max_drawdown:.4f}".replace('.', ','),
                # Totals
                f"{avg_total_pnl:.2f}".replace('.', ','),
                str(total_trades_sum),
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
                c_winrate = per_cycle_data[base_idx + 2]
                c_profit_pct = (c_pnl / initial_balance) * 100 if initial_balance != 0 else 0.0

                row.extend([
                    str(c_trades),
                    f"{c_profit_pct:.2f}".replace('.', ','),
                    f"{c_pnl:.2f}".replace('.', ','),  # Add TotalPnL for each cycle
                    f"{c_winrate:.4f}".replace('.', ',')
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
        per_cycle_winrates = np.zeros(num_bots * num_cycles, dtype=np.float32)

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

    def _log_first_bot_details(self, bot: CompactBotConfig, result: BacktestResult, 
                                 initial_balance: float, num_cycles: int, 
                                 output_dir: str, generation: int):
        """
        DEPRECATED: No longer used. First bot details are not logged separately.
        All bot information is available in the generation CSV files.
        """
        return
        
        # OLD CODE (disabled to remove extra log files)
        log_file = os.path.join(output_dir, f"generation_{generation}_bot_{bot.bot_id}_details.txt")
        
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(f"=== BOT {bot.bot_id} DETAILED VERIFICATION LOG ===\n")
                f.write(f"Generation: {generation}\n")
                f.write(f"Leverage: {bot.leverage}x\n")
                f.write(f"Number of Indicators: {bot.num_indicators}\n")
                f.write(f"Initial Balance: ${initial_balance:.2f}\n\n")
                
                f.write(f"=== AGGREGATE RESULTS ===\n")
                f.write(f"Total Trades (all cycles): {result.total_trades}\n")
                f.write(f"Winning Trades: {result.winning_trades}\n")
                f.write(f"Losing Trades: {result.losing_trades}\n")
                f.write(f"Total PnL: ${result.total_pnl:.2f}\n")
                f.write(f"Win Rate: {result.win_rate:.4f}\n")
                f.write(f"Final Balance: ${result.final_balance:.2f}\n")
                f.write(f"Fitness Score: {result.fitness_score:.2f}\n")
                f.write(f"Sharpe Ratio: {result.sharpe_ratio:.2f}\n")
                f.write(f"Max Drawdown: {result.max_drawdown:.4f}\n\n")
                
                f.write(f"=== PER-CYCLE BREAKDOWN ===\n")
                cycle_pnls = result.per_cycle_pnl if hasattr(result, 'per_cycle_pnl') and result.per_cycle_pnl else []
                cycle_trades = result.per_cycle_trades if hasattr(result, 'per_cycle_trades') and result.per_cycle_trades else []
                cycle_wins = result.per_cycle_wins if hasattr(result, 'per_cycle_wins') and result.per_cycle_wins else []
                
                for i in range(min(num_cycles, len(cycle_pnls))):
                    trades = cycle_trades[i] if i < len(cycle_trades) else 0
                    pnl = cycle_pnls[i] if i < len(cycle_pnls) else 0.0
                    wins = cycle_wins[i] if i < len(cycle_wins) else 0
                    profit_pct = (pnl / initial_balance) * 100 if initial_balance > 0 else 0.0
                    win_rate = wins / trades if trades > 0 else 0.0
                    
                    f.write(f"\nCycle {i}:\n")
                    f.write(f"  Trades: {trades}\n")
                    f.write(f"  Wins: {wins}\n")
                    f.write(f"  PnL: ${pnl:.2f}\n")
                    f.write(f"  Profit %: {profit_pct:.2f}%\n")
                    f.write(f"  Win Rate: {win_rate:.4f}\n")
                
                # Calculate verification metrics
                f.write(f"\n=== VERIFICATION CHECKS ===\n")
                total_cycle_trades = sum(cycle_trades) if cycle_trades else 0
                f.write(f"Sum of per-cycle trades: {total_cycle_trades}\n")
                f.write(f"Aggregate total trades: {result.total_trades}\n")
                f.write(f"Match: {total_cycle_trades == result.total_trades}\n\n")
                
                avg_pnl = sum(cycle_pnls) / num_cycles if cycle_pnls else 0.0
                f.write(f"Average per-cycle PnL: ${avg_pnl:.2f}\n")
                f.write(f"Aggregate total PnL: ${result.total_pnl:.2f}\n")
                f.write(f"Match: {abs(avg_pnl - result.total_pnl) < 0.01}\n")
                
            log_info(f"First bot details logged to {log_file}")
        except Exception as e:
            log_error(f"Failed to log first bot details: {e}")

    def shutdown(self):
        """Shutdown the logging processor."""
        log_info("Shutting down GPU Logging Processor")
        self.file_executor.shutdown(wait=True)