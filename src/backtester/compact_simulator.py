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

from ..bot_generator.compact_generator import CompactBotConfig, COMPACT_BOT_SIZE
from ..utils.validation import log_info, log_error, log_debug


@dataclass
class BacktestResult:
    """Result from backtesting a bot."""
    bot_id: int
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_consecutive_wins: float
    max_consecutive_losses: float
    final_balance: float
    generation_survived: int
    fitness_score: float


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
        initial_balance: float = 10000.0
    ):
        """Initialize two-kernel backtester."""
        if gpu_context is None or gpu_queue is None:
            raise RuntimeError("GPU context and queue required")
        
        self.ctx = gpu_context
        self.queue = gpu_queue
        self.initial_balance = initial_balance
        
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
        Backtest bots using two-kernel approach.
        
        Step 1: Precompute all 50 indicators for all bars (ONCE)
        Step 2: Backtest all bots reading from precomputed buffer
        
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
        
        # === STEP 1: Precompute all indicators ===
        indicators_buffer = self._precompute_indicators(ohlcv_data)
        
        # === STEP 2: Backtest all bots ===
        results = self._run_backtest_kernel(
            bots,
            ohlcv_data,
            indicators_buffer,
            cycles
        )
        
        return results
    
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
        
        # === MEMORY USAGE ANALYSIS ===
        # Why OUT_OF_RESOURCES occurs:
        # 
        # 1. Global Memory Usage (NOT the issue):
        #    - OHLCV input: 120,206 bars × 5 floats × 4 bytes = 2.4 MB
        #    - Indicators output: 50 indicators × 120,206 bars × 4 bytes = 24.0 MB
        #    - Total global memory: ~26.4 MB (well within 3.19 GB limit)
        #
        # 2. Per-Work-Item Resource Pressure (THE REAL ISSUE):
        #    - 50 concurrent work items, each processing 120,206 bars sequentially
        #    - Each work item needs: ~20-30 registers for loop variables, function calls, local arrays
        #    - Intel UHD Graphics: Limited registers per work item (~128-256 total registers per compute unit)
        #    - With 80 compute units, each might support 2-4 work items before register exhaustion
        #
        # 3. Sequential Processing Bottleneck:
        #    - Each work item executes 120,206 iterations in nested loops (compute_* functions)
        #    - Register spilling: When registers exceed available, data spills to slower global memory
        #    - Local memory pressure: 64 KB limit per compute unit, shared across work items
        #    - Function call overhead: Each bar calls compute_sma_helper(), creating stack pressure
        #
        # 4. Work Distribution Inefficiency:
        #    - Current: 50 work items × 120K operations each = 6M total operations
        #    - Optimal: 12,800 work items × ~470 operations each (better GPU utilization)
        #    - Current approach: Few work items doing massive sequential work = resource contention
        #
        # 5. Indicator Computation Complexity:
        #    - SMA(200): Each bar needs 200 OHLCV lookups in a loop
        #    - RSI: Complex averaging with multiple loops per bar
        #    - MACD: Multiple EMA computations with interdependencies
        #    - Total: Millions of memory accesses per work item
        
        log_info(f"Precompute kernel memory analysis:")
        log_info(f"  - OHLCV buffer: {ohlcv_bytes / (1024*1024):.1f} MB ({num_bars} bars × 5 floats × 4 bytes)")
        log_info(f"  - Indicators buffer: {indicator_bytes / (1024*1024):.1f} MB (50 indicators × {num_bars} bars × 4 bytes)")
        log_info(f"  - Total global memory: {(ohlcv_bytes + indicator_bytes) / (1024*1024):.1f} MB")
        log_info(f"  - Work items: {self.NUM_INDICATORS * 256} total ({self.NUM_INDICATORS} indicators × 256 work items each)")
        log_info(f"  - Operations per work item: ~{num_bars // 256} bars (distributed among 256 work items per indicator)")
        log_info(f"  - Stateful indicators: Computed by 1 work item per indicator (maintains state)")
        log_info(f"  - Stateless indicators: Distributed across all 256 work items per indicator")
        log_info(f"  - GPU limits: 80 compute units, 64 KB local memory, ~128-256 registers per compute unit")
        log_info(f"  - SOLUTION: Better work distribution reduces per-work-item resource pressure")
        
        # Execute precompute kernel
        # 50 indicators × 256 work items per indicator = 12,800 total work items
        # Each work item processes ~470 bars (120,206 / 256)
        # Stateful indicators use only work_item_id == 0, stateless ones distribute work
        kernel = self._precompute_kernel
        global_size = (self.NUM_INDICATORS, 256)  # 2D: (indicators, work_items_per_indicator)
        local_size = (1, 256)  # Work group size
        
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
                log_error(f"GPU backtesting failed with OUT_OF_RESOURCES even with minimal kernel: {e}")
                log_error("The backtest kernel is too complex for Intel UHD Graphics GPU")
                log_error("Root causes:")
                log_error("  1. Position array (1 × 32 bytes = 32 bytes per work item)")
                log_error("  2. Deep nested loops (cycles → bars → position management)")
                log_error("  3. Complex signal generation with many conditionals")
                log_error("  4. Limited registers per compute unit on Intel UHD Graphics")
                raise RuntimeError("GPU backtest kernel too complex for available hardware")
            else:
                raise
    
    def _run_backtest_gpu_minimal(
        self,
        bots: List[CompactBotConfig],
        ohlcv_data: np.ndarray,
        indicators_buffer: cl.Buffer,
        cycles: List[Tuple[int, int]]
    ) -> List[BacktestResult]:
        """
        Run backtest with adaptive batch sizing based on GPU capabilities.
        Start small, increase batch size until failure, then use optimal size.
        """
        num_bots = len(bots)
        all_results = []
        
        # Find optimal batch size by testing progressively larger batches
        optimal_batch_size = self._find_optimal_batch_size(bots, ohlcv_data, indicators_buffer, cycles)
        
        log_info(f"Using optimal batch size of {optimal_batch_size} bots for GPU backtesting")
        
        # Process all bots using the optimal batch size
        log_info(f"Processing {num_bots} bots in batches of {optimal_batch_size}")
        
        # Calculate total batches for progress bar
        total_batches = (num_bots + optimal_batch_size - 1) // optimal_batch_size
        
        with tqdm(total=num_bots, desc="Backtesting bots", unit="bot") as pbar:
            for batch_start in range(0, num_bots, optimal_batch_size):
                batch_end = min(batch_start + optimal_batch_size, num_bots)
                batch_bots = bots[batch_start:batch_end]
                
                batch_results = self._run_backtest_batch_adaptive(
                    batch_bots,
                    ohlcv_data,
                    indicators_buffer,
                    cycles,
                    batch_start
                )
                all_results.extend(batch_results)
                
                # Update progress bar
                pbar.update(len(batch_results))
        
        log_info(f"Completed GPU backtesting: {len(all_results)} bots processed in {total_batches} batches")
        return all_results
    
    def _find_optimal_batch_size(
        self,
        bots: List[CompactBotConfig],
        ohlcv_data: np.ndarray,
        indicators_buffer: cl.Buffer,
        cycles: List[Tuple[int, int]]
    ) -> int:
        """
        Find the optimal batch size by testing progressively larger batches.
        Uses a more efficient approach with fewer test points.
        """
        # Test sizes that are conservative for Intel UHD Graphics
        test_sizes = [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000]
        
        # Don't test sizes larger than total bots
        test_sizes = [s for s in test_sizes if s <= len(bots)]
        
        if not test_sizes:
            return 1
            
        optimal_size = 1
        
        log_debug(f"Testing batch sizes: {test_sizes}")
        
        with tqdm(total=len(test_sizes), desc="Finding optimal batch size", unit="test") as pbar:
            for test_size in test_sizes:
                test_bots = bots[:test_size]
                try:
                    log_debug(f"Testing batch size {test_size}...")
                    start_time = time.time()
                    
                    self._run_backtest_batch_adaptive(
                        test_bots,
                        ohlcv_data,
                        indicators_buffer,
                        cycles,
                        0  # Test batch
                    )
                    
                    elapsed = time.time() - start_time
                    optimal_size = test_size
                    log_debug(f"Batch size {test_size} works! ({elapsed:.2f}s)")
                    
                except cl.RuntimeError as e:
                    if "OUT_OF_RESOURCES" in str(e):
                        log_debug(f"Batch size {test_size} failed with OUT_OF_RESOURCES")
                        pbar.update(1)
                        break  # Stop at first failure
                    else:
                        pbar.update(1)
                        raise  # Re-raise non-resource errors
                
                pbar.update(1)
        
        log_info(f"Optimal batch size found: {optimal_size} bots")
        return optimal_size
    
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
        
        # Results buffer (64 bytes per bot)
        RESULT_SIZE = 64
        results_bytes = RESULT_SIZE * num_bots
        
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
        # Define result struct (64 bytes)
        dt = np.dtype([
            ('bot_id', np.int32),
            ('total_trades', np.int32),
            ('winning_trades', np.int32),
            ('losing_trades', np.int32),
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
            result = BacktestResult(
                bot_id=int(res['bot_id']),
                total_trades=int(res['total_trades']),
                winning_trades=int(res['winning_trades']),
                losing_trades=int(res['losing_trades']),
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

