"""
COMPACT GPU Backtester - Precomputed Indicator Architecture

NEW APPROACH: Two-kernel strategy for memory efficiency
1. Precompute all 50 indicators ONCE for all bars
2. Bots read only their configured indicators

Memory Analysis:
- OLD (inline): 10K bots × 50 indicators inline = OUT_OF_RESOURCES
- NEW (precomputed): 50 indicators × 5000 bars × 4 bytes = 1 MB
                     + 10K bots × 128 bytes = 1.28 MB
                     + Results 10K × 64 bytes = 640 KB
                     = ~3 MB total for 10K bots (vs OUT_OF_RESOURCES)

Scalability: Can handle 1M+ bots with same 1 MB indicator buffer!
"""
import numpy as np
import pyopencl as cl
from typing import List, Tuple, Dict
from dataclasses import dataclass
from pathlib import Path
import time
from tqdm import tqdm
import time
import threading
import concurrent.futures

from ..bot_generator.compact_generator import CompactBotConfig, COMPACT_BOT_SIZE
from ..utils.validation import log_info, log_error, log_debug, log_warning


@dataclass
class BacktestResult:
    """Result from backtesting a bot."""
    bot_id: int
    total_trades: int
    winning_trades: int
    losing_trades: int
    per_cycle_trades: List[int]
    per_cycle_wins: List[int]
    per_cycle_pnl: List[float]
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_consecutive_wins: float
    max_consecutive_losses: float
    final_balance: float = 0.0
    generation_survived: int = 0
    fitness_score: float = 0.0
    # Data chunking metadata
    chunk_id: int = 0
    chunk_bars: int = 0


class CompactBacktester:
    """
    Two-kernel GPU backtester with precomputed indicators.
    
    Kernel 1: Precompute all 50 indicators once
    Kernel 2: Backtest all bots reading from precomputed buffer
    
    Memory efficiency: O(indicators × bars) instead of O(bots × indicators × bars)
    """
    
    NUM_INDICATORS = 50
    
    def __init__(
        self,
        gpu_context: cl.Context,
        gpu_queue: cl.CommandQueue,
        initial_balance: float = 10000.0,
        target_chunk_seconds: float = 1.0
    ):
        """Initialize two-kernel backtester."""
        if gpu_context is None or gpu_queue is None:
            raise RuntimeError("GPU context and queue required")
        
        self.ctx = gpu_context
        self.queue = gpu_queue
        self.initial_balance = initial_balance
        self.target_chunk_seconds = target_chunk_seconds  # Target processing time per chunk
        
        # Memory tracking
        self.memory_usage: Dict[str, int] = {}
        
        # Track active buffers for cleanup
        self._active_buffers: List[cl.Buffer] = []
        
        # Compile both kernels
        self._compile_kernels()
        
        log_info("CompactBacktester initialized (two-kernel precomputed strategy)")
        log_info(f"  - Kernel 1: Precompute {self.NUM_INDICATORS} indicators")
        log_info(f"  - Kernel 2: Backtest with signal generation")
    
    def __del__(self):
        """Cleanup OpenCL resources."""
        self.cleanup()
    
    def cleanup(self):
        """Release all OpenCL buffers."""
        if hasattr(self, '_active_buffers'):
            for buf in self._active_buffers:
                try:
                    buf.release()
                except:
                    pass
            self._active_buffers.clear()
            log_debug("Released all OpenCL buffers")
    
    def _compile_kernels(self):
        """Compile both precompute and backtest kernels."""
        kernel_dir = Path(__file__).parent.parent / "gpu_kernels"
        
        # Kernel 1: Precompute indicators
        precompute_path = kernel_dir / "precompute_all_indicators.cl"
        if not precompute_path.exists():
            raise FileNotFoundError(f"Kernel not found: {precompute_path}")
        
        precompute_src = precompute_path.read_text()
        
        try:
            self.precompute_program = cl.Program(self.ctx, precompute_src).build()
            self._precompute_kernel = cl.Kernel(self.precompute_program, "precompute_all_indicators")
            log_info("[OK] Compiled precompute_all_indicators.cl (50 indicators)")
        except cl.RuntimeError as e:
            log_error(f"Precompute kernel compilation failed: {e}")
            raise
        
        # Kernel 2: Backtest with precomputed indicators
        backtest_path = kernel_dir / "backtest_with_precomputed.cl"
        if not backtest_path.exists():
            raise FileNotFoundError(f"Kernel not found: {backtest_path}")
        
        backtest_src = backtest_path.read_text()
        
        try:
            self.backtest_program = cl.Program(self.ctx, backtest_src).build()
            self._backtest_kernel = cl.Kernel(self.backtest_program, "backtest_with_signals")
            log_info("[OK] Compiled backtest_with_precomputed.cl (real trading logic)")
        except cl.RuntimeError as e:
            log_error(f"Backtest kernel compilation failed: {e}")
            raise
    
    def backtest_bots(
        self,
        bots: List[CompactBotConfig],
        ohlcv_data: np.ndarray,
        cycles: List[Tuple[int, int]]
    ) -> List[BacktestResult]:
        """
        Backtest bots using data chunking approach for large datasets.
        
        NEW APPROACH: Data chunking instead of bot batching
        - Keep all bots in memory (small footprint)
        - Slice data into manageable chunks (10-30 days each)
        - Process all bots against each data chunk in parallel
        - Aggregate results across all chunks
        
        Args:
            bots: List of compact bot configs
            ohlcv_data: OHLCV array [N x 5] (open, high, low, close, volume)
            cycles: List of (start_idx, end_idx) tuples
        
        Returns:
            List of backtest results with detailed metrics
        """
        num_bots = len(bots)
        num_bars = len(ohlcv_data)
        num_cycles = len(cycles)
        
        # Memory tracking
        self.memory_usage.clear()
        
        # === DATA CHUNKING STRATEGY ===
        # Calculate optimal chunk size based on GPU memory
        chunk_days = self._calculate_optimal_data_chunk_size(ohlcv_data, cycles)
        bars_per_day = 1440  # Assuming 1m timeframe
        chunk_size_bars = chunk_days * bars_per_day
        
        log_info(f"Data chunking strategy: {chunk_days} days per chunk ({chunk_size_bars:,} bars)")
        log_info(f"Total data: {num_bars:,} bars across {len(cycles)} cycles")
        
        # Create data chunks
        data_chunks = self._create_data_chunks(ohlcv_data, cycles, chunk_size_bars)
        log_info(f"Created {len(data_chunks)} data chunks for processing")
        
        # Process all bots against all data chunks
        all_results = self._backtest_bots_against_data_chunks(bots, data_chunks)
        
        log_info(f"Completed data-chunked backtesting: {len(all_results)} bots processed across {len(data_chunks)} chunks")
        return all_results
    
    def _calculate_optimal_data_chunk_size(
        self,
        ohlcv_data: np.ndarray,
        cycles: List[Tuple[int, int]]
    ) -> int:
        """
        Calculate optimal data chunk size in days based on GPU memory constraints and target processing time.
        
        Strategy: Balance memory usage with smooth processing (target ~1 second per chunk).
        This allows processing 10k bots against large datasets with responsive progress updates.
        """
        bars_per_day = 1440  # 1m timeframe
        
        # === MEMORY-BASED MAX CHUNK SIZE ===
        memory_per_day_mb = (bars_per_day * 50 * 4) / (1024 * 1024)  # 50 indicators × 4 bytes
        available_memory_mb = 2500  # Conservative estimate for Intel UHD 630
        max_chunk_days_memory = int(available_memory_mb / memory_per_day_mb)
        
        # === PERFORMANCE-BASED TARGET CHUNK SIZE ===
        # Target: configurable seconds per chunk for smooth processing
        # Based on observed performance: ~5,200 bars/second
        target_bars_per_second = 5200  # Conservative estimate
        target_chunk_bars = int(target_bars_per_second * self.target_chunk_seconds)
        target_chunk_days = max(1, target_chunk_bars // bars_per_day)  # At least 1 day
        
        # Use the smaller of memory limit and performance target
        optimal_chunk_days = min(target_chunk_days, max_chunk_days_memory)
        
        # Ensure minimum of 1 day for indicator accuracy
        optimal_chunk_days = max(1, optimal_chunk_days)
        
        log_debug(f"Chunk sizing: Memory allows {max_chunk_days_memory} days, performance targets {target_chunk_days} days")
        log_debug(f"Selected chunk size: {optimal_chunk_days} days ({optimal_chunk_days * bars_per_day:,} bars)")
        
        return optimal_chunk_days
    
    def _create_data_chunks(
        self,
        ohlcv_data: np.ndarray,
        cycles: List[Tuple[int, int]],
        chunk_size_bars: int
    ) -> List[Dict]:
        """
        Create data chunks for processing.
        
        Each chunk contains:
        - OHLCV data slice
        - Adjusted cycles for the chunk
        - Metadata for result aggregation
        """
        chunks = []
        total_bars = len(ohlcv_data)
        
        # Create overlapping chunks to ensure complete coverage
        # Dynamic overlap: minimum 200 bars for indicator lookback, or 20% of chunk size (whichever is larger)
        min_overlap_bars = 200  # For SMA(200) and other indicators requiring lookback
        overlap_bars = max(min_overlap_bars, int(chunk_size_bars * 0.2))
        effective_chunk_size = chunk_size_bars - overlap_bars
        
        start_idx = 0
        chunk_id = 0
        
        while start_idx < total_bars:
            end_idx = min(start_idx + chunk_size_bars, total_bars)
            
            # Extract data chunk
            chunk_data = ohlcv_data[start_idx:end_idx]
            
            # Adjust cycles for this chunk
            chunk_cycles = []
            for cycle_start, cycle_end in cycles:
                # Adjust cycle indices relative to chunk
                adjusted_start = max(0, cycle_start - start_idx)
                adjusted_end = min(len(chunk_data), cycle_end - start_idx)
                
                # Only include cycles that overlap with this chunk
                if adjusted_start < adjusted_end:
                    chunk_cycles.append((adjusted_start, adjusted_end))
            
            if chunk_cycles:  # Only add chunks that have cycles
                chunks.append({
                    'id': chunk_id,
                    'data': chunk_data,
                    'cycles': chunk_cycles,
                    'global_start_idx': start_idx,
                    'global_end_idx': end_idx
                })
                chunk_id += 1
            
            # Move to next chunk with overlap
            start_idx += effective_chunk_size
        
        return chunks
    
    def _backtest_bots_against_data_chunks(
        self,
        bots: List[CompactBotConfig],
        data_chunks: List[Dict]
    ) -> List[BacktestResult]:
        """
        Backtest all bots against all data chunks and aggregate results.
        
        Strategy: Process multiple chunks in parallel for maximum GPU utilization,
        keeping all bots in memory for efficiency.
        """
        num_bots = len(bots)
        all_chunk_results = []
        
        log_info(f"Processing {len(data_chunks)} data chunks with {num_bots} bots each")
        
        # Determine optimal parallel processing level based on GPU capabilities
        # Intel UHD 630 has 80 compute units - we can run multiple kernels simultaneously
        max_parallel_chunks = min(4, len(data_chunks))  # Conservative: 4 parallel chunks max
        
        log_info(f"Using parallel processing: {max_parallel_chunks} chunks simultaneously for max GPU utilization")
        
        # Use progress bar for overall progress
        with tqdm(total=len(data_chunks), desc="Processing data chunks", unit="chunk") as pbar:
            
            def process_chunk_with_progress(chunk):
                """Process a single chunk and update progress."""
                chunk_id = chunk['id']
                chunk_data = chunk['data']
                chunk_cycles = chunk['cycles']
                
                # Process the chunk
                chunk_results = self._backtest_bots_against_single_chunk(bots, chunk_data, chunk_cycles)
                
                # Store results with chunk metadata
                for result in chunk_results:
                    result.chunk_id = chunk_id
                    result.chunk_bars = len(chunk_data)
                
                # Update progress
                pbar.update(1)
                pbar.set_postfix({
                    'chunk': f"{chunk_id + 1}/{len(data_chunks)}",
                    'bars': f"{len(chunk_data):,}",
                    'cycles': len(chunk_cycles),
                    'parallel': f"{max_parallel_chunks} threads"
                })
                
                return chunk_results
            
            # Process chunks in parallel batches to maintain memory efficiency
            for i in range(0, len(data_chunks), max_parallel_chunks):
                batch_chunks = data_chunks[i:i + max_parallel_chunks]
                
                # Process this batch in parallel
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel_chunks) as executor:
                    batch_results = list(executor.map(process_chunk_with_progress, batch_chunks))
                
                # Collect results from this batch
                for chunk_results in batch_results:
                    all_chunk_results.extend(chunk_results)
        
        # Aggregate results across all chunks
        aggregated_results = self._aggregate_chunk_results(all_chunk_results, num_bots)
        
        return aggregated_results
    
    def _backtest_bots_against_single_chunk(
        self,
        bots: List[CompactBotConfig],
        chunk_data: np.ndarray,
        chunk_cycles: List[Tuple[int, int]]
    ) -> List[BacktestResult]:
        """
        Backtest all bots against a single data chunk.
        
        Uses direct kernel execution without bot batching since we have all bots in memory.
        """
        num_bots = len(bots)
        num_bars = len(chunk_data)
        num_cycles = len(chunk_cycles)
        
        # === STEP 1: Precompute indicators for this chunk ===
        indicators_buffer = self._precompute_indicators(chunk_data)
        
        # === STEP 2: Backtest all bots against this chunk ===
        # Direct kernel execution for all bots at once
        results = self._run_backtest_kernel_direct(bots, chunk_data, indicators_buffer, chunk_cycles)
        
        # Cleanup
        indicators_buffer.release()
        
        return results
    
    def _run_backtest_kernel_direct(
        self,
        bots: List[CompactBotConfig],
        ohlcv_data: np.ndarray,
        indicators_buffer: cl.Buffer,
        cycles: List[Tuple[int, int]]
    ) -> List[BacktestResult]:
        """
        Run backtest kernel directly for all bots without batching.
        
        Optimized for data chunking approach where we process all bots against one chunk.
        """
        num_bots = len(bots)
        num_bars = len(ohlcv_data)
        num_cycles = len(cycles)
        
        # Serialize bot configs
        bot_configs_raw = self._serialize_bots(bots)
        
        bots_buf = cl.Buffer(
            self.ctx,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=bot_configs_raw
        )
        
        # OHLCV buffer
        ohlcv_flat = ohlcv_data.astype(np.float32)
        
        ohlcv_buf = cl.Buffer(
            self.ctx,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=ohlcv_flat
        )
        
        # Cycles buffers
        cycle_starts = np.array([c[0] for c in cycles], dtype=np.int32)
        cycle_ends = np.array([c[1] for c in cycles], dtype=np.int32)
        
        cycle_starts_buf = cl.Buffer(
            self.ctx,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=cycle_starts
        )
        
        cycle_ends_buf = cl.Buffer(
            self.ctx,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=cycle_ends
        )
        
        # Results buffer: base 64 bytes + per-cycle arrays (MAX_CYCLES × 12 bytes)
        MAX_CYCLES = 100
        RESULT_BASE = 64
        results_bytes = (RESULT_BASE + (MAX_CYCLES * 12)) * num_bots
        
        results_buf = cl.Buffer(
            self.ctx,
            cl.mem_flags.WRITE_ONLY,
            size=results_bytes
        )
        
        # Execute backtest kernel for all bots
        kernel = self._backtest_kernel
        global_size = (num_bots,)
        local_size = None  # Let OpenCL choose optimal work group size
        
        try:
            kernel(
                self.queue,
                global_size,
                local_size,
                bots_buf,                          # Bot configs
                ohlcv_buf,                         # OHLCV data
                indicators_buffer,                 # Precomputed indicators
                cycle_starts_buf,                  # Cycle starts
                cycle_ends_buf,                    # Cycle ends
                np.int32(num_cycles),              # Number of cycles
                np.int32(num_bars),                # Number of bars
                np.float32(self.initial_balance),  # Initial balance
                results_buf                        # Results output
            )
            
            self.queue.finish()
            
        except cl.RuntimeError as e:
            log_error(f"Backtest kernel execution failed for {num_bots} bots: {e}")
            # Cleanup on error
            bots_buf.release()
            ohlcv_buf.release()
            cycle_starts_buf.release()
            cycle_ends_buf.release()
            results_buf.release()
            raise
        
        # Read results
        results_raw = np.empty(results_bytes, dtype=np.uint8)
        cl.enqueue_copy(self.queue, results_raw, results_buf)
        self.queue.finish()
        
        # Parse results
        results = self._parse_results(results_raw, num_bots)
        
        # Adjust bot IDs to match input bots
        for i, result in enumerate(results):
            result.bot_id = bots[i].bot_id
        
        # Cleanup
        bots_buf.release()
        ohlcv_buf.release()
        cycle_starts_buf.release()
        cycle_ends_buf.release()
        results_buf.release()
        
        return results
    
    def _aggregate_chunk_results(
        self,
        all_chunk_results: List[BacktestResult],
        num_bots: int
    ) -> List[BacktestResult]:
        """
        Aggregate backtest results from all data chunks.
        
        For each bot, combine results across all chunks.
        """
        # Group results by bot_id
        bot_results = {}
        
        for result in all_chunk_results:
            bot_id = result.bot_id
            
            if bot_id not in bot_results:
                # Initialize aggregated result
                bot_results[bot_id] = BacktestResult(
                    bot_id=bot_id,
                    total_trades=0,
                    winning_trades=0,
                    losing_trades=0,
                    per_cycle_trades=[],
                    per_cycle_wins=[],
                    per_cycle_pnl=[],
                    total_pnl=0.0,
                    max_drawdown=0.0,
                    sharpe_ratio=0.0,
                    win_rate=0.0,
                    avg_win=0.0,
                    avg_loss=0.0,
                    profit_factor=0.0,
                    max_consecutive_wins=0,
                    max_consecutive_losses=0
                )
            
            # Aggregate metrics
            aggregated = bot_results[bot_id]
            aggregated.total_trades += result.total_trades
            aggregated.winning_trades += result.winning_trades
            aggregated.losing_trades += result.losing_trades
            aggregated.total_pnl += result.total_pnl
            
            # Track worst drawdown across chunks
            aggregated.max_drawdown = max(aggregated.max_drawdown, result.max_drawdown)
            
            # Extend per-cycle arrays
            aggregated.per_cycle_trades.extend(result.per_cycle_trades)
            aggregated.per_cycle_wins.extend(result.per_cycle_wins)
            aggregated.per_cycle_pnl.extend(result.per_cycle_pnl)
        
        # Calculate derived metrics for each bot
        final_results = []
        for bot_result in bot_results.values():
            # Calculate win rate
            if bot_result.total_trades > 0:
                bot_result.win_rate = bot_result.winning_trades / bot_result.total_trades
            
            # Calculate average win/loss
            if bot_result.winning_trades > 0:
                bot_result.avg_win = bot_result.total_pnl / bot_result.winning_trades
            if bot_result.losing_trades > 0:
                bot_result.avg_loss = abs(bot_result.total_pnl) / bot_result.losing_trades
            
            # Calculate profit factor
            total_wins = bot_result.winning_trades * bot_result.avg_win if bot_result.winning_trades > 0 else 0
            total_losses = bot_result.losing_trades * bot_result.avg_loss if bot_result.losing_trades > 0 else 0
            if total_losses > 0:
                bot_result.profit_factor = total_wins / total_losses
            
            # Calculate Sharpe ratio (simplified)
            if len(bot_result.per_cycle_pnl) > 1:
                pnl_std = np.std(bot_result.per_cycle_pnl)
                if pnl_std > 0:
                    bot_result.sharpe_ratio = np.mean(bot_result.per_cycle_pnl) / pnl_std
            
            final_results.append(bot_result)
        
        return final_results

    def _precompute_indicators(self, ohlcv_data: np.ndarray) -> cl.Buffer:
        """
        Precompute all 50 indicators for all bars.
        
        Returns OpenCL buffer containing indicator values.
        Buffer layout: [50 indicators][num_bars values]
        """
        num_bars = len(ohlcv_data)
        
        # Prepare OHLCV buffer
        ohlcv_flat = ohlcv_data.astype(np.float32)
        
        ohlcv_buf = cl.Buffer(
            self.ctx,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=ohlcv_flat
        )
        
        ohlcv_bytes = ohlcv_flat.nbytes
        self.memory_usage['ohlcv_mb'] = ohlcv_bytes / (1024 * 1024)
        
        # Allocate output buffer for indicators (dynamic size based on actual bars)
        # 50 indicators × num_bars × 4 bytes (float32)
        indicator_bytes = self.NUM_INDICATORS * num_bars * 4
        
        indicators_buf = cl.Buffer(
            self.ctx,
            cl.mem_flags.WRITE_ONLY,
            size=indicator_bytes
        )
        
        self.memory_usage['indicators_mb'] = indicator_bytes / (1024 * 1024)
        
        # === MEMORY USAGE ANALYSIS === (removed verbose logging for cleaner output)
        
        # Execute precompute kernel
        # Increased work items per indicator for better GPU utilization
        # 50 indicators × 512 work items per indicator = 25,600 total work items
        # Each work item processes ~235 bars (for 4,320 bar chunks)
        # Better utilizes 80 compute units on Intel UHD 630
        kernel = self._precompute_kernel
        global_size = (self.NUM_INDICATORS, 512)  # 2D: (indicators, work_items_per_indicator)
        local_size = (1, 512)  # Work group size
        
        try:
            kernel(
                self.queue,
                global_size,
                local_size,
                ohlcv_buf,               # Input: OHLCV data
                np.int32(num_bars),      # Number of bars
                indicators_buf           # Output: Indicator values
            )
            
            self.queue.finish()
            
        except cl.RuntimeError as e:
            log_error(f"Precompute kernel execution failed: {e}")
            # Cleanup on error
            ohlcv_buf.release()
            indicators_buf.release()
            raise
        except Exception as e:
            log_error(f"Unexpected error in precompute kernel: {e}")
            ohlcv_buf.release()
            indicators_buf.release()
            raise
        
        # Clean up OHLCV buffer (not needed anymore)
        ohlcv_buf.release()
        
        return indicators_buf
    
    def _run_backtest_kernel(
        self,
        bots: List[CompactBotConfig],
        ohlcv_data: np.ndarray,
        indicators_buffer: cl.Buffer,
        cycles: List[Tuple[int, int]]
    ) -> List[BacktestResult]:
        """
        Run backtest kernel with precomputed indicators.
        
        Each bot reads only its configured indicators from the buffer.
        """
        num_bots = len(bots)
        num_bars = len(ohlcv_data)
        num_cycles = len(cycles)
        
        # Try GPU backtesting with reduced batch size
        try:
            log_info("Attempting GPU backtesting with minimal batch size...")
            results = self._run_backtest_gpu_minimal(bots, ohlcv_data, indicators_buffer, cycles)
            log_info(f"GPU backtesting successful: {len(results)} bots processed")
            return results
        except cl.RuntimeError as e:
            if "OUT_OF_RESOURCES" in str(e):
                log_warning(f"GPU backtesting failed with OUT_OF_RESOURCES: {e}")
                log_warning("Attempting automatic data size reduction to enable GPU processing...")
                
                # Try with reduced cycles to make data size manageable
                original_cycles = len(cycles)
                reduced_cycles = max(1, original_cycles // 2)
                
                if reduced_cycles < original_cycles:
                    log_info(f"Reducing cycles from {original_cycles} to {reduced_cycles} for GPU compatibility")
                    
                    # Generate reduced cycle ranges
                    # For simplicity, take the first N cycles
                    reduced_cycle_ranges = cycles[:reduced_cycles]
                    
                    # Retry with reduced cycles
                    try:
                        results = self._run_backtest_gpu_minimal(bots, ohlcv_data, indicators_buffer, reduced_cycle_ranges)
                        log_warning(f"GPU backtesting successful with reduced cycles: {len(results)} bots processed")
                        log_warning("Note: Results are based on fewer cycles - consider reducing data size in configuration")
                        return results
                    except cl.RuntimeError as e2:
                        log_error(f"GPU backtesting still failed even with reduced cycles: {e2}")
                
                # If still failing, raise the original error
                log_error("The backtest kernel is too complex for Intel UHD Graphics GPU")
                log_error(f"Current setup: {len(ohlcv_data)} bars × {len(cycles)} cycles = {len(ohlcv_data) * len(cycles):,} operations per bot")
                log_error(f"Theoretical maximum batch size: {self._calculate_theoretical_max_batch_size(ohlcv_data, cycles)} bots")
                log_error("Solutions:")
                log_error("  1. Reduce days per cycle (current configuration)")
                log_error("  2. Reduce cycles per generation")
                log_error("  3. Reduce population size")
                log_error("  4. Use a GPU with more resources")
                raise RuntimeError("GPU backtest kernel too complex for available hardware")
            else:
                raise
    
    def _calculate_theoretical_max_batch_size(
        self,
        ohlcv_data: np.ndarray,
        cycles: List[Tuple[int, int]]
    ) -> int:
        """
        Calculate theoretical maximum batch size based on memory usage and data size.

        The key insight: Memory usage scales with data size, not operations per bot.
        Large datasets require more memory for OHLCV/indicators buffers.
        """
        num_bars = len(ohlcv_data)
        num_cycles = len(cycles)

        # Memory-based complexity: Total data size affects buffer requirements
        total_data_complexity = num_bars * num_cycles

        # Intel UHD Graphics 630 specifications
        compute_units = 80
        registers_per_cu = 256  # Estimated total registers per compute unit
        max_workgroup_size = 512
        total_gpu_memory_mb = 3190  # 3.19 GB

        # CALIBRATED: Based on actual test results with conservative estimates
        # With large datasets, GPU resources are heavily constrained
        registers_per_bot = 10  # Increased from 5 - more conservative
        memory_per_bot_kb = 0.5  # Increased from 0.1 - more conservative

        # Local memory limit per compute unit
        local_memory_per_cu_kb = 64

        # Calculate memory-based limits
        max_bots_by_registers = (registers_per_cu // registers_per_bot) * compute_units
        max_bots_by_memory = int((local_memory_per_cu_kb / memory_per_bot_kb) * compute_units)
        max_bots_by_workgroup = max_workgroup_size

        # DATA SIZE COMPLEXITY: Much more restrictive for large datasets
        # Based on testing: large datasets severely limit batch sizes
        if total_data_complexity > 1000000:  # > 1M total operations (very large datasets)
            data_complexity_factor = 0.5  # Very restrictive
        elif total_data_complexity > 500000:  # > 500K total operations (large datasets)
            data_complexity_factor = 0.6  # Restrictive
        elif total_data_complexity > 100000:  # > 100K total operations
            data_complexity_factor = 0.7  # Moderate restriction
        else:
            data_complexity_factor = 0.8  # Light restriction

        # Apply data complexity factor
        theoretical_max = min(
            max_bots_by_registers,
            max_bots_by_memory
        )

        theoretical_max = int(theoretical_max * data_complexity_factor)

        # Ensure minimum of 1
        theoretical_max = max(1, theoretical_max)        # Log detailed analysis
        log_debug("Theoretical batch size calculation (MEMORY-BASED data complexity):")
        log_debug(f"  Data size: {num_bars:,} bars × {num_cycles} cycles = {total_data_complexity:,} total operations")
        log_debug(f"  GPU: {compute_units} CU, {registers_per_cu} registers/CU, {local_memory_per_cu_kb} KB local/CU")
        log_debug(f"  Per-bot resources: {registers_per_bot} registers, {memory_per_bot_kb:.1f} KB memory")
        log_debug(f"  Limits: registers={max_bots_by_registers}, memory={max_bots_by_memory}")
        log_debug(f"  Data complexity factor: {data_complexity_factor:.1f} (based on total data size)")
        log_debug(f"  Theoretical maximum: {theoretical_max} bots (accounts for OHLCV/indicators buffer memory)")
        
        return theoretical_max
    
    def _run_backtest_gpu_minimal(
        self,
        bots: List[CompactBotConfig],
        ohlcv_data: np.ndarray,
        indicators_buffer: cl.Buffer,
        cycles: List[Tuple[int, int]]
    ) -> List[BacktestResult]:
        """
        Run backtest with adaptive batch sizing based on GPU capabilities.
        Uses parallel processing for better performance with large populations.
        """
        num_bots = len(bots)
        all_results = []
        
        # Calculate theoretical maximum batch size
        theoretical_max = self._calculate_theoretical_max_batch_size(ohlcv_data, cycles)
        log_info(f"Theoretical maximum batch size: {theoretical_max} bots")
        
        # Find optimal batch size by testing progressively larger batches
        optimal_size = self._find_optimal_batch_size(bots, ohlcv_data, indicators_buffer, cycles, theoretical_max)
        
        log_info(f"Using optimal batch size of {optimal_size} bots for GPU backtesting")
        
        # CRITICAL SAFETY CHECK: If optimal batch size is too small, data is too large for GPU
        if optimal_size <= 10:
            log_error(f"CRITICAL: Optimal batch size is too small ({optimal_size} bots) - data size is too large for GPU")
            log_error(f"Current setup: {len(ohlcv_data)} bars × {len(cycles)} cycles = {len(ohlcv_data) * len(cycles):,} operations per bot")
            log_error("This will cause infinite kernel failures and system instability")
            log_error("SOLUTION: Reduce data size by:")
            log_error("  1. Reduce days per cycle (recommended: 1-7 days)")
            log_error("  2. Reduce cycles per generation (recommended: 5-10 cycles)")
            log_error("  3. Use smaller population size")
            raise RuntimeError(f"Data size too large for GPU processing (optimal batch size: {optimal_size})")
        
        # Process all bots using parallel batch processing for better throughput
        log_info(f"Processing {num_bots} bots in parallel batches of {optimal_size}")
        
        # Create batches
        batches = []
        for batch_start in range(0, num_bots, optimal_size):
            batch_end = min(batch_start + optimal_size, num_bots)
            batch_bots = bots[batch_start:batch_end]
            batches.append((batch_bots, batch_start))
        
        # Use thread pool for parallel processing
        # Limit concurrent batches to avoid overwhelming the GPU
        max_concurrent_batches = min(4, len(batches))  # Max 4 concurrent batches
        
        log_info(f"Processing {len(batches)} batches with up to {max_concurrent_batches} concurrent batches")
        
        # Use a lock to ensure GPU operations are serialized
        gpu_lock = threading.Lock()
        
        def process_batch_with_lock(batch_bots, ohlcv_data, indicators_buffer, cycles, batch_start):
            with gpu_lock:
                return self._run_backtest_batch_adaptive(batch_bots, ohlcv_data, indicators_buffer, cycles, batch_start)
        
        with tqdm(total=num_bots, desc="Backtesting bots", unit="bot") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_batches) as executor:
                # Submit all batches for parallel execution
                future_to_batch = {
                    executor.submit(process_batch_with_lock, batch_bots, ohlcv_data, indicators_buffer, cycles, batch_start): (batch_bots, batch_start)
                    for batch_bots, batch_start in batches
                }
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_batch):
                    batch_bots, batch_start = future_to_batch[future]
                    try:
                        batch_results = future.result()
                        all_results.extend(batch_results)
                        pbar.update(len(batch_results))
                    except Exception as e:
                        log_error(f"Batch starting at bot {batch_start} failed: {e}")
                        raise
        
        log_info(f"Completed GPU backtesting: {len(all_results)} bots processed in {len(batches)} parallel batches")
        return all_results
    
    def _find_optimal_batch_size(
        self,
        bots: List[CompactBotConfig],
        ohlcv_data: np.ndarray,
        indicators_buffer: cl.Buffer,
        cycles: List[Tuple[int, int]],
        theoretical_max: int
    ) -> int:
        """
        Find the optimal batch size by testing progressively larger batches.
        Starts from small sizes and works up to find the maximum that works.
        """
        # Generate test sizes starting from small and working up to theoretical max and beyond
        test_sizes = []
        max_test_size = min(len(bots), 10000)  # Test up to 10,000 bots maximum

        # Start from small sizes and work up
        current = 50  # Start with small batch
        while current <= max_test_size and len(test_sizes) < 15:  # Limit to 15 test sizes max
            test_sizes.append(current)
            if current >= max_test_size:
                break
            # Increase by 25-50% each time
            increment = max(1, current // 3)
            current += increment

        # Ensure we include theoretical_max if it's not already in the list
        if theoretical_max not in test_sizes and theoretical_max <= max_test_size:
            test_sizes.append(theoretical_max)

        # Ensure we test at least some sizes above theoretical max
        if max_test_size > theoretical_max and theoretical_max * 2 not in test_sizes:
            test_sizes.append(min(theoretical_max * 2, max_test_size))

        # Remove duplicates and sort
        test_sizes = sorted(list(set(test_sizes)))

        # Allow more tests for wider range (increased from 12)
        if len(test_sizes) > 20:
            # Keep more samples, especially higher ones
            indices = [0, len(test_sizes)-1]
            step = (len(test_sizes) - 1) // 18
            for i in range(step, len(test_sizes)-1, step):
                indices.append(i)
            test_sizes = [test_sizes[i] for i in sorted(set(indices))]

        log_debug(f"Comprehensive stress testing batch sizes: {test_sizes} (theoretical max: {theoretical_max}, max test: {max_test_size})")

        # Don't test sizes larger than total bots
        test_sizes = [s for s in test_sizes if s <= len(bots)]

        if not test_sizes:
            return 1

        optimal_size = 1
        successful_sizes = []

        with tqdm(total=len(test_sizes), desc="Finding optimal batch size", unit="test") as pbar:
            for test_size in test_sizes:
                test_bots = bots[:test_size]

                # MULTIPLE TEST RUNS: Run the same batch size multiple times to catch
                # intermittent failures and memory fragmentation issues
                test_runs = 3  # Run each size 3 times
                all_runs_passed = True

                for run in range(test_runs):
                    try:
                        log_debug(f"Stress testing batch size {test_size} run {run+1}/{test_runs}...")
                        start_time = time.time()

                        # Use threading with timeout to prevent hanging
                        import threading
                        result = [None]
                        exception = [None]
                        
                        def run_test():
                            try:
                                result[0] = self._run_backtest_batch_adaptive(
                                    test_bots,
                                    ohlcv_data,
                                    indicators_buffer,
                                    cycles,
                                    0  # Test batch
                                )
                            except Exception as e:
                                exception[0] = e
                        
                        test_thread = threading.Thread(target=run_test)
                        test_thread.start()
                        test_thread.join(timeout=30)  # 30 second timeout
                        
                        if test_thread.is_alive():
                            log_debug(f"Batch size {test_size} run {run+1} timed out after 30 seconds")
                            all_runs_passed = False
                            break
                        elif exception[0]:
                            raise exception[0]
                        
                        elapsed = time.time() - start_time
                        log_debug(f"Batch size {test_size} run {run+1} passed ({elapsed:.2f}s)")

                    except cl.RuntimeError as e:
                        if "OUT_OF_RESOURCES" in str(e):
                            log_debug(f"Batch size {test_size} run {run+1} failed with OUT_OF_RESOURCES")
                            all_runs_passed = False
                            break  # Stop testing this size
                        else:
                            log_debug(f"Batch size {test_size} run {run+1} failed with unexpected error: {e}")
                            all_runs_passed = False
                            break

                if all_runs_passed:
                    optimal_size = test_size
                    successful_sizes.append(test_size)
                    log_debug(f"Batch size {test_size} passed all {test_runs} test runs")
                else:
                    log_debug(f"Batch size {test_size} failed - stopping at this size")
                    break  # Stop at first size that fails

                pbar.update(1)

        # SAFETY CHECK: If no sizes worked, fail fast
        if not successful_sizes:
            log_error("CRITICAL: No batch sizes passed GPU testing - data is too large for this GPU")
            log_error(f"Tested sizes: {test_sizes}")
            log_error("This configuration will cause infinite kernel failures")
            return 1  # Return 1 to trigger the safety check above

        # APPLY MODERATE SAFETY MARGIN: Use 70% of the maximum successful size to account for
        # real-world complexity factors not captured in testing (reduced from 50%)
        if successful_sizes:
            max_successful = max(successful_sizes)
            safety_margin = int(max_successful * 0.7)  # 30% safety margin
            final_optimal = max(1, safety_margin)

            log_info(f"Optimal batch size found: {final_optimal} bots (with 30% safety margin from max successful: {max_successful})")
            log_info(f"Tested up to {max_test_size} bots, successful sizes: {successful_sizes}")
        else:
            final_optimal = 1
            log_warning("No batch sizes passed testing, using minimum size of 1")

        return final_optimal
    
    def _run_backtest_batch_adaptive(
        self,
        batch_bots: List[CompactBotConfig],
        ohlcv_data: np.ndarray,
        indicators_buffer: cl.Buffer,
        cycles: List[Tuple[int, int]],
        bot_id_offset: int
    ) -> List[BacktestResult]:
        """
        Run backtest kernel for a batch of bots with adaptive sizing.
        """
        num_bots = len(batch_bots)
        num_bars = len(ohlcv_data)
        num_cycles = len(cycles)
        
        # Serialize bot configs for this batch
        bot_configs_raw = self._serialize_bots(batch_bots)
        
        bots_buf = cl.Buffer(
            self.ctx,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=bot_configs_raw
        )
        
        # OHLCV buffer
        ohlcv_flat = ohlcv_data.astype(np.float32)
        
        ohlcv_buf = cl.Buffer(
            self.ctx,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=ohlcv_flat
        )
        
        # Cycles buffers
        cycle_starts = np.array([c[0] for c in cycles], dtype=np.int32)
        cycle_ends = np.array([c[1] for c in cycles], dtype=np.int32)
        
        cycle_starts_buf = cl.Buffer(
            self.ctx,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=cycle_starts
        )
        
        cycle_ends_buf = cl.Buffer(
            self.ctx,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=cycle_ends
        )
        
        # Results buffer: base 64 bytes + per-cycle arrays (MAX_CYCLES × 12 bytes)
        MAX_CYCLES = 100
        RESULT_BASE = 64
        results_bytes = (RESULT_BASE + (MAX_CYCLES * 12)) * num_bots
        
        results_buf = cl.Buffer(
            self.ctx,
            cl.mem_flags.WRITE_ONLY,
            size=results_bytes
        )
        
        # Execute backtest kernel for this batch
        kernel = self._backtest_kernel
        global_size = (num_bots,)
        local_size = None  # Let OpenCL choose optimal work group size
        
        try:
            kernel(
                self.queue,
                global_size,
                local_size,
                bots_buf,                          # Bot configs
                ohlcv_buf,                         # OHLCV data
                indicators_buffer,                 # Precomputed indicators
                cycle_starts_buf,                  # Cycle starts
                cycle_ends_buf,                    # Cycle ends
                np.int32(num_cycles),              # Number of cycles
                np.int32(num_bars),                # Number of bars
                np.float32(self.initial_balance),  # Initial balance
                results_buf                        # Results output
            )
            
            self.queue.finish()
            
        except cl.RuntimeError as e:
            log_error(f"Backtest kernel execution failed for batch of {num_bots} bots: {e}")
            # Cleanup on error
            bots_buf.release()
            ohlcv_buf.release()
            cycle_starts_buf.release()
            cycle_ends_buf.release()
            results_buf.release()
            raise
        except Exception as e:
            log_error(f"Unexpected error in backtest kernel for batch of {num_bots} bots: {e}")
            # Cleanup on error
            bots_buf.release()
            ohlcv_buf.release()
            cycle_starts_buf.release()
            cycle_ends_buf.release()
            results_buf.release()
            raise
        
        # Read results
        results_raw = np.empty(results_bytes, dtype=np.uint8)
        cl.enqueue_copy(self.queue, results_raw, results_buf)
        
        # Parse results and adjust bot IDs
        results = self._parse_results(results_raw, num_bots)
        
        # Adjust bot IDs to account for batch offset
        for i, result in enumerate(results):
            result.bot_id = batch_bots[i].bot_id
        
        # Clean up buffers
        bots_buf.release()
        ohlcv_buf.release()
        cycle_starts_buf.release()
        cycle_ends_buf.release()
        results_buf.release()
        
        return results
    
    def _serialize_bots(self, bots: List[CompactBotConfig]) -> np.ndarray:
        """Serialize bots to raw bytes matching OpenCL struct."""
        raw_data = np.empty(len(bots) * COMPACT_BOT_SIZE, dtype=np.uint8)
        
        # Define struct layout (128 bytes total) - MUST MATCH OpenCL struct!
        dt = np.dtype([
            ('bot_id', np.int32),
            ('num_indicators', np.uint8),
            ('indicator_indices', np.uint8, 8),
            ('indicator_params', np.float32, (8, 3)),
            ('risk_strategy_bitmap', np.uint32),  # FIXED: uint32 to match OpenCL
            ('tp_multiplier', np.float32),
            ('sl_multiplier', np.float32),
            ('leverage', np.uint8),
            ('padding', np.uint8, 6)  # FIXED: 6 bytes padding for 128-byte alignment
        ])
        
        structured = np.zeros(len(bots), dtype=dt)
        
        for i, bot in enumerate(bots):
            structured[i]['bot_id'] = bot.bot_id
            structured[i]['num_indicators'] = bot.num_indicators
            structured[i]['indicator_indices'] = bot.indicator_indices
            structured[i]['indicator_params'] = bot.indicator_params
            structured[i]['risk_strategy_bitmap'] = bot.risk_strategy_bitmap
            structured[i]['tp_multiplier'] = bot.tp_multiplier
            structured[i]['sl_multiplier'] = bot.sl_multiplier
            structured[i]['leverage'] = bot.leverage
        
        raw_data = structured.tobytes()
        return np.frombuffer(raw_data, dtype=np.uint8)
    
    def _parse_results(self, raw_data: np.ndarray, num_bots: int) -> List[BacktestResult]:
        """Parse backtest results from raw bytes."""
        # Define result struct (base fields + per-cycle arrays)
        MAX_CYCLES = 100
        dt = np.dtype([
            ('bot_id', np.int32),
            ('total_trades', np.int32),
            ('winning_trades', np.int32),
            ('losing_trades', np.int32),
            ('cycle_trades', np.int32, (MAX_CYCLES,)),
            ('cycle_wins', np.int32, (MAX_CYCLES,)),
            ('cycle_pnl', np.float32, (MAX_CYCLES,)),
            ('total_pnl', np.float32),
            ('max_drawdown', np.float32),
            ('sharpe_ratio', np.float32),
            ('win_rate', np.float32),
            ('avg_win', np.float32),
            ('avg_loss', np.float32),
            ('profit_factor', np.float32),
            ('max_consecutive_wins', np.float32),
            ('max_consecutive_losses', np.float32),
            ('final_balance', np.float32),
            ('generation_survived', np.int32),
            ('fitness_score', np.float32)
        ])
        
        structured = np.frombuffer(raw_data, dtype=dt)
        
        results = []
        for res in structured[:num_bots]:
            # Extract per-cycle arrays
            per_cycle_trades = [int(x) for x in res['cycle_trades'].tolist()]
            per_cycle_wins = [int(x) for x in res['cycle_wins'].tolist()]
            per_cycle_pnl = [float(x) for x in res['cycle_pnl'].tolist()]

            result = BacktestResult(
                bot_id=int(res['bot_id']),
                total_trades=int(res['total_trades']),
                winning_trades=int(res['winning_trades']),
                losing_trades=int(res['losing_trades']),
                per_cycle_trades=per_cycle_trades,
                per_cycle_wins=per_cycle_wins,
                per_cycle_pnl=per_cycle_pnl,
                total_pnl=float(res['total_pnl']),
                max_drawdown=float(res['max_drawdown']),
                sharpe_ratio=float(res['sharpe_ratio']),
                win_rate=float(res['win_rate']),
                avg_win=float(res['avg_win']),
                avg_loss=float(res['avg_loss']),
                profit_factor=float(res['profit_factor']),
                max_consecutive_wins=float(res['max_consecutive_wins']),
                max_consecutive_losses=float(res['max_consecutive_losses']),
                final_balance=float(res['final_balance']),
                generation_survived=int(res['generation_survived']),
                fitness_score=float(res['fitness_score'])
            )
            results.append(result)
        
        return results
    
    def estimate_vram(self, num_bots: int, num_bars: int, num_cycles: int) -> dict:
        """
        Estimate VRAM usage for two-kernel backtest.
        
        NEW: Precomputed indicator approach uses fixed indicator buffer size
        regardless of bot count. Scales much better!
        """
        # Dynamic indicator buffer (50 × num_bars × 4 bytes)
        indicators_mb = (self.NUM_INDICATORS * num_bars * 4) / (1024 * 1024)
        
        # Bot configs scale with num_bots
        bot_configs_mb = (num_bots * COMPACT_BOT_SIZE) / (1024 * 1024)
        
        # OHLCV data (fixed per dataset)
        ohlcv_mb = (num_bars * 5 * 4) / (1024 * 1024)
        
        # Cycles (minimal)
        cycles_mb = (num_cycles * 2 * 4) / (1024 * 1024)
        
        # Results scale with num_bots
        results_mb = (num_bots * 64) / (1024 * 1024)
        
        total_mb = indicators_mb + bot_configs_mb + ohlcv_mb + cycles_mb + results_mb
        
        return {
            'indicators_mb': indicators_mb,  # NEW: Fixed size
            'bot_configs_mb': bot_configs_mb,
            'ohlcv_mb': ohlcv_mb,
            'cycles_mb': cycles_mb,
            'results_mb': results_mb,
            'total_mb': total_mb,
            'total_bytes': int(total_mb * 1024 * 1024),
            'scalability': f"Fixed {indicators_mb:.1f}MB indicator buffer, scales O(N) with bots"
        }
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get actual memory usage from last backtest."""
        return self.memory_usage.copy()

