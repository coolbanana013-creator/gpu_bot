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
import os
import pyopencl as cl
from typing import List, Tuple, Dict
import math
from dataclasses import dataclass
from pathlib import Path
import time
import threading
import concurrent.futures
from tqdm import tqdm
import csv
import queue
import atexit

from ..bot_generator.compact_generator import CompactBotConfig, COMPACT_BOT_SIZE
from ..utils.validation import log_info, log_error, log_debug, log_warning

# Trade log structure for detailed trade recording
TRADE_LOG_DTYPE = np.dtype([
    ('bot_id', np.int32),
    ('cycle', np.int32),
    ('entry_price', np.float32),
    ('exit_price', np.float32),
    ('entry_bar', np.int32),
    ('exit_bar', np.int32),
    ('leverage', np.float32),
    ('pnl', np.float32),
    ('direction', np.int32)  # 1 = long, -1 = short
    ,('chunk_id', np.int32)
    ,('out_of_cycle', np.int32)
])


class TradeLogStreamWriter:
    """Production streaming writer for trade logs using queue-based worker thread."""
    
    def __init__(self, csv_path: Path, batch_size: int = 100, max_queue_size: int = 1000):
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.write_queue = queue.Queue(maxsize=max_queue_size)
        self.worker_thread = None
        self.shutdown_event = threading.Event()
        self.total_written = 0
        self.total_enqueued = 0
        self._lock = threading.Lock()
        self._started = False
        
        # Register cleanup on exit
        atexit.register(self.shutdown)
    
    def start(self):
        """Start the background writer thread."""
        if self._started:
            return
        
        self._started = True
        self.worker_thread = threading.Thread(target=self._writer_loop, daemon=False)
        self.worker_thread.start()
        log_debug(f"TradeLogStreamWriter started: {self.csv_path}")
    
    def _writer_loop(self):
        """Worker thread that processes write queue with buffering."""
        buffer = []
        csv_file = None
        csv_writer = None
        
        try:
            # Open CSV file once
            csv_file = open(self.csv_path, 'a', newline='')
            csv_writer = csv.writer(csv_file, delimiter=';')
            
            while not self.shutdown_event.is_set() or not self.write_queue.empty():
                try:
                    # Get batch from queue with timeout
                    logs = self.write_queue.get(timeout=0.5)
                    buffer.extend(logs)
                    
                    # Flush when batch size reached or queue empty
                    if len(buffer) >= self.batch_size or self.write_queue.empty():
                        self._flush_buffer(csv_writer, buffer)
                        buffer.clear()
                        csv_file.flush()
                    
                    self.write_queue.task_done()
                    
                except queue.Empty:
                    # Flush any remaining buffer on timeout
                    if buffer:
                        self._flush_buffer(csv_writer, buffer)
                        buffer.clear()
                        csv_file.flush()
                    continue
                    
        except Exception as e:
            log_error(f"TradeLogStreamWriter worker error: {e}")
        finally:
            # Final flush
            if buffer and csv_writer:
                try:
                    self._flush_buffer(csv_writer, buffer)
                except Exception as e:
                    log_error(f"TradeLogStreamWriter final flush error: {e}")
            
            if csv_file:
                csv_file.close()
            
            log_debug(f"TradeLogStreamWriter worker stopped. Total written: {self.total_written}")
    
    def _flush_buffer(self, csv_writer, buffer):
        """Write buffered logs to CSV."""
        if not buffer:
            return
        
        rows = []
        def safe_int(v, default=0):
            try:
                if isinstance(v, float) and not math.isfinite(v):
                    return default
                return int(v)
            except Exception:
                return default

        for log_entry in buffer:
            # Handle both numpy structured array and tuple formats
            if isinstance(log_entry, tuple):
                # Tuple format from .tolist()
                bot_id, cycle, entry_price, exit_price, entry_bar, exit_bar, leverage, pnl, direction, chunk_id, out_of_cycle = log_entry
            else:
                # Numpy structured array format
                bot_id = safe_int(log_entry['bot_id'])
                cycle = safe_int(log_entry['cycle'])
                entry_price = float(log_entry['entry_price'])
                exit_price = float(log_entry['exit_price'])
                entry_bar = safe_int(log_entry['entry_bar'])
                exit_bar = safe_int(log_entry['exit_bar'])
                leverage = float(log_entry['leverage'])
                pnl = float(log_entry['pnl'])
                direction = safe_int(log_entry['direction'], default=0)
                chunk_id = safe_int(log_entry['chunk_id'])
                out_of_cycle = safe_int(log_entry['out_of_cycle'])
            
            # Compute signature for deduplication
            # Ensure we guard against NaN/Inf values which would break int conversions
            direction_str = 'LONG' if direction == 1 else ('SHORT' if direction == -1 else 'UNKNOWN')
            if not isinstance(pnl, float) or not math.isfinite(pnl):
                pnl_cents = 'nan'
            else:
                try:
                    pnl_cents = int(round(pnl * 100))
                except OverflowError:
                    pnl_cents = 'inf'

            signature = f"{entry_bar}:{exit_bar}:{direction_str}:{pnl_cents}"
            
            rows.append([
                bot_id,
                cycle,
                round(entry_price, 4),
                round(exit_price, 4),
                entry_bar,
                exit_bar,
                round(leverage, 1),
                round(pnl, 2),
                direction_str,
                chunk_id,
                out_of_cycle,
                signature
            ])
        
        csv_writer.writerows(rows)
        
        with self._lock:
            self.total_written += len(buffer)
    
    def enqueue(self, logs: np.ndarray):
        """Enqueue trade logs for async writing."""
        if not self._started:
            self.start()
        
        try:
            # Convert to list for queue
            log_list = logs.tolist()
            self.write_queue.put(log_list, block=True, timeout=5.0)
            
            with self._lock:
                self.total_enqueued += len(log_list)
                
        except queue.Full:
            log_warning(f"TradeLogStreamWriter queue full, dropping {len(logs)} logs")
    
    def shutdown(self, timeout: float = 30.0):
        """Gracefully shutdown writer and wait for queue to drain."""
        if not self._started:
            return
        
        log_info(f"TradeLogStreamWriter shutting down (queued: {self.write_queue.qsize()}, written: {self.total_written})")
        
        # Signal shutdown and wait for queue to drain
        self.shutdown_event.set()
        
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=timeout)
            
            if self.worker_thread.is_alive():
                log_warning(f"TradeLogStreamWriter worker did not finish within {timeout}s")
            else:
                log_info(f"TradeLogStreamWriter shutdown complete. Total written: {self.total_written}")
    
    def get_stats(self) -> Dict[str, int]:
        """Get writer statistics."""
        with self._lock:
            return {
                'enqueued': self.total_enqueued,
                'written': self.total_written,
                'queue_size': self.write_queue.qsize()
            }


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
    per_cycle_signals: List[int]  # NEW: signals per cycle
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
        target_chunk_seconds: float = 1.0,
        data_chunk_days: int = 100,
        enable_mtf: bool = True,
        htf_multiplier: int = 60  # 1h HTF from 1m base (60x multiplier)
    ):
        """Initialize two-kernel backtester with thread-safety and MTF support."""
        if gpu_context is None or gpu_queue is None:
            raise RuntimeError("GPU context and queue required")
        
        self.ctx = gpu_context
        self.queue = gpu_queue
        self.initial_balance = initial_balance
        self.target_chunk_seconds = target_chunk_seconds  # Target processing time per chunk
        self.user_data_chunk_days = data_chunk_days  # User-specified chunk size
        self.enable_mtf = enable_mtf  # Enable multi-timeframe filtering
        self.htf_multiplier = htf_multiplier  # HTF bar multiplier (e.g., 60 for 1h from 1m)
        
        # Get GPU device info for memory calculations
        self.device = self.ctx.devices[0]
        self.global_mem_size = self.device.global_mem_size
        self.local_mem_size = self.device.local_mem_size
        self.max_work_group_size = self.device.max_work_group_size
        self.compute_units = self.device.max_compute_units
        
        # Memory tracking (thread-safe)
        self.memory_usage: Dict[str, int] = {}
        self._memory_lock = threading.Lock()
        
        # Track active buffers for cleanup (thread-safe)
        self._active_buffers: List[cl.Buffer] = []
        self._buffer_lock = threading.Lock()
        
        # GPU queue lock (OpenCL queues are NOT thread-safe)
        self._queue_lock = threading.Lock()
        
        # Compile both kernels
        self._compile_kernels()
        # Optional GPU trade logging (toggle via env var ENABLE_TRADE_LOGS=1)
        import os
        self.trade_log_enabled = os.getenv('ENABLE_TRADE_LOGS', '0') == '1'
        # Debugging: allow disabling trade logs via env var DEBUG_DISABLE_TRADE_LOGS
        if os.getenv('DEBUG_DISABLE_TRADE_LOGS', '0') == '1':
            log_info('[DEBUG] DEBUG_DISABLE_TRADE_LOGS set - disabling trade logging for debugging')
            self.trade_log_enabled = False
        self.trade_log_max = int(os.getenv('TRADE_LOG_MAX', '20000'))
        
        # Streaming trade log writer
        self.trade_log_writer = None
        
        # Calculate optimal chunk size based on GPU memory
        self.optimal_chunk_days = None  # Will be set during backtest_bots
        
        log_info("CompactBacktester initialized (two-kernel precomputed strategy)")
        log_info(f"  - Kernel 1: Precompute {self.NUM_INDICATORS} indicators")
        log_info(f"  - Kernel 2: Backtest with signal generation")
        log_info(f"  - GPU Memory: {self.global_mem_size / (1024**3):.2f} GB")
        log_info(f"  - Compute Units: {self.compute_units}")
        if self.enable_mtf:
            log_info(f"  - MTF Enabled: HTF multiplier = {self.htf_multiplier}x (higher timeframe filtering)")
        # Allow disabling many verbose debug logs to reduce output and memory
        import os
        self.disable_debug_logging = os.getenv('DISABLE_GPU_LOGGING', '0') == '1'
        # Debug bypass / filters toggles at runtime
        self.debug_disable_filters = int(os.getenv('DEBUG_DISABLE_FILTERS', '0') == '1')
        self.debug_bypass_sr = int(os.getenv('DEBUG_BYPASS_SR', '0') == '1')
        self.debug_bypass_volume = int(os.getenv('DEBUG_BYPASS_VOLUME', '0') == '1')
        self.debug_force_signals = int(os.getenv('DEBUG_FORCE_SIGNALS', '0') == '1')

    def cleanup(self):
        """HIGH PRIORITY FIX: Explicitly release GPU resources to prevent memory leaks."""
        with self._buffer_lock:
            for buf in self._active_buffers:
                try:
                    buf.release()
                except:
                    pass  # Already released
            self._active_buffers.clear()
        
        # Clear memory tracking
        with self._memory_lock:
            self.memory_usage.clear()
        
        log_info("CompactBacktester: GPU resources cleaned up")
    
    def __del__(self):
        """HIGH PRIORITY FIX: Release GPU resources when object destroyed."""
        try:
            self.cleanup()
        except:
            pass  # Cleanup already done or failed
    
    def _maybe_debug_log(self, message: str) -> None:
        if not self.disable_debug_logging:
            log_info(message)
    
    def __del__(self):
        """Cleanup OpenCL resources."""
        self.cleanup()
    
    def cleanup(self):
        """Release all OpenCL buffers (thread-safe)."""
        # Shutdown trade log writer first
        if hasattr(self, 'trade_log_writer') and self.trade_log_writer:
            self.trade_log_writer.shutdown()
            self.trade_log_writer = None
        
        if hasattr(self, '_active_buffers'):
            with self._buffer_lock:
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
        # Inject compile-time filter threshold macros from configuration
        from ..utils.config import (
            DEFAULT_ADX_MIN, DEFAULT_ADX_MAX, DEFAULT_ATR_SPIKE_FACTOR, DEFAULT_VOLUME_MULTIPLIER,
            ENABLE_FILTER_DEBUG_INSTRUMENTATION
        )
        backtest_src = f"#define ADX_MIN_THRESHOLD {DEFAULT_ADX_MIN}f\n" + backtest_src
        backtest_src = f"#define ADX_MAX_THRESHOLD {DEFAULT_ADX_MAX}f\n" + backtest_src
        backtest_src = f"#define ATR_SPIKE_FACTOR {DEFAULT_ATR_SPIKE_FACTOR}f\n" + backtest_src
        backtest_src = f"#define VOLUME_MULTIPLIER_THRESHOLD {DEFAULT_VOLUME_MULTIPLIER}f\n" + backtest_src
        
        # Conditionally enable filter debug instrumentation
        if ENABLE_FILTER_DEBUG_INSTRUMENTATION:
            backtest_src = f"#define ENABLE_FILTER_DEBUG_INSTRUMENTATION\n" + backtest_src
            log_info('[DEBUG] Filter debug instrumentation ENABLED (atomic counters active)')
        else:
            log_info('[DEBUG] Filter debug instrumentation DISABLED (no atomic counter overhead)')
        
        # Allow optional debug compile-time flags via environment variables
        if os.getenv('DEBUG_LOW_CONSENSUS', '0') == '1':
            log_info('[DEBUG] DEBUG_LOW_CONSENSUS set - compiling kernel with low consensus threshold')
            backtest_src = '#define DEBUG_FORCE_LOW_CONSENSUS\n' + backtest_src
        if os.getenv('DEBUG_ACCEPT_NEUTRAL', '0') == '1':
            log_info('[DEBUG] DEBUG_ACCEPT_NEUTRAL set - compiling kernel to accept neutral signals as directional for debug')
            backtest_src = '#define DEBUG_ACCEPT_NEUTRAL_AS_SIGNAL\n' + backtest_src
        if os.getenv('DEBUG_BYPASS_QUALITY_FILTERS', '0') == '1':
            log_info('[DEBUG] DEBUG_BYPASS_QUALITY_FILTERS set - compiling kernel with bypass for quality filters')
            backtest_src = '#define DEBUG_BYPASS_QUALITY_FILTERS\n' + backtest_src
        
        try:
            self.backtest_program = cl.Program(self.ctx, backtest_src).build()
            self._backtest_kernel = cl.Kernel(self.backtest_program, "backtest_with_signals")
            self._backtest_parallel_kernel = cl.Kernel(self.backtest_program, "backtest_parallel_bot_cycle")
            log_info("[OK] Compiled backtest_with_precomputed.cl (real trading logic + parallel bot-cycle kernel)")
        except cl.RuntimeError as e:
            log_error(f"Backtest kernel compilation failed: {e}")
            raise
        
        # Kernel 3: GPU-accelerated result aggregation
        aggregate_path = kernel_dir / "aggregate_results.cl"
        if not aggregate_path.exists():
            raise FileNotFoundError(f"Kernel not found: {aggregate_path}")
        
        aggregate_src = aggregate_path.read_text()
        
        try:
            self.aggregate_program = cl.Program(self.ctx, aggregate_src).build()
            self._aggregate_kernel = cl.Kernel(self.aggregate_program, "aggregate_bot_results")
            self._aggregate_flat_kernel = cl.Kernel(self.aggregate_program, "aggregate_flat_results")
            log_info("[OK] Compiled aggregate_results.cl (GPU result aggregation)")
        except cl.RuntimeError as e:
            log_error(f"Aggregation kernel compilation failed: {e}")
            raise

        # Optional: compile the signal debug kernel if present (separate from aggregate build)
        debug_path = kernel_dir / "signal_debug.cl"
        if debug_path.exists():
            try:
                debug_src = debug_path.read_text()
                self.debug_program = cl.Program(self.ctx, debug_src).build()
                self._debug_signal_kernel = cl.Kernel(self.debug_program, "debug_signal_generation")
                log_info("[OK] Compiled signal_debug.cl (debug helper kernel)")
            except cl.RuntimeError:
                log_warning("Failed to compile signal_debug.cl - skipping debug kernel")

    def run_debug_signal_generation(self, precomputed_indicators_buf, compact_bots_buf, cycle_start: int, cycle_end: int, num_bars: int, num_bots_to_sample: int = 64, bars_per_sample: int = 60, cycle_to_debug: int = 0):
        """Run the signal debug kernel to gather signal stats for a subset of bots.
        Returns a list of SignalDebugRecord serialized entries.
        """
        if not hasattr(self, '_debug_signal_kernel'):
            raise RuntimeError('Debug kernel not compiled')

        # Estimate num samples per bot across the cycle range (we keep the sample count reasonable)
        # For debug, we limit to 128 samples per bot (to avoid huge allocations)
        # cycle_start and cycle_end are passed by host
        cycle_bars = max(1, cycle_end - cycle_start)
        num_samples = min(128, max(1, cycle_bars // max(1, bars_per_sample)))

        debug_record_struct_size = 128  # Rough estimate (depends on kernel struct)
        total_records = num_bots_to_sample * num_samples

        # Allocate and zero-initialize buffer for records to avoid reading uninitialized memory
        debug_records_host = np.zeros(total_records * debug_record_struct_size, dtype=np.uint8)
        debug_records_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=debug_records_host)

        # Create cycle ranges buffer for this single cycle
        cycle_ranges_array = np.array([cycle_start, cycle_end], dtype=np.int32)
        cycle_ranges_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=cycle_ranges_array)

        # Launch kernel
        kernel = self._debug_signal_kernel
        kernel(self.queue, (num_bots_to_sample,), None,
               precomputed_indicators_buf,
               compact_bots_buf,
               cycle_ranges_buf,
               debug_records_buf,
               np.int32(num_bars),
               np.int32(num_bots_to_sample),
               np.int32(bars_per_sample),
               np.int32(cycle_to_debug))
        self.queue.finish()

        # Read back records as raw bytes (for robust parsing of int/float fields)
        debug_flat = np.empty(total_records * debug_record_struct_size, dtype=np.uint8)
        cl.enqueue_copy(self.queue, debug_flat, debug_records_buf)
        self.queue.finish()
        # Return raw byte array so host can interpret struct with numpy dtype
        return debug_flat
    
    def backtest_bots(
        self,
        bots: List[CompactBotConfig],
        ohlcv_data: np.ndarray,
        cycles: List[Tuple[int, int]]
    ) -> List[BacktestResult]:
        """
        Backtest bots using multi-cycle parallel processing with data chunking.

        NEW STRATEGY (Optimized):
        - Process ALL cycles together in each data chunk (up to memory limit)
        - Chunk the full dataset into pieces that fit GPU memory
        - For each chunk, run ALL cycles that overlap with that chunk
        - Each cycle uses its own slice of the data within the chunk
        - Aggregate results across chunks and cycles

        Benefits:
        - Massive GPU utilization: Process 100 cycles × 10k bots = 1M workloads in parallel
        - Fewer kernel launches: ~170 chunks instead of 100 cycles × ~170 chunks
        - Better memory efficiency: Load data once, test all cycles
        """
        num_bots = len(bots)
        # Allow pandas DataFrame to be passed in (convenience for other modules)
        try:
            import pandas as _pd
        except Exception:
            _pd = None

        from ..utils.config import OHLCV_COLUMNS

        if _pd is not None and isinstance(ohlcv_data, _pd.DataFrame):
            # Convert to contiguous numpy array (float32). Kernel expects 5 floats per bar:
            # open, high, low, close, volume (timestamp is excluded)
            ohlcv_no_ts = OHLCV_COLUMNS[1:]  # drop timestamp
            ohlcv_data = ohlcv_data[ohlcv_no_ts].to_numpy(dtype=np.float32)

        num_bars = len(ohlcv_data)
        num_cycles = len(cycles)

        # Use user-specified chunk size, halve on OUT_OF_RESOURCES
        if not hasattr(self, '_optimal_data_chunk_bars'):
            bars_per_day = 1440
            user_chunk_bars = self.user_data_chunk_days * bars_per_day
            self._optimal_data_chunk_bars = min(num_bars, user_chunk_bars)

        print(f"\n[BACKTEST] Testing {num_bots:,} bots × {num_cycles} cycles = {num_bots * num_cycles:,} workloads")
        print(f"[BACKTEST] Dataset: {num_bars:,} bars ({num_bars/1440:.1f} days)")
        print(f"[BACKTEST] Chunk size: {self._optimal_data_chunk_bars/1440:.1f} days ({self._optimal_data_chunk_bars:,} bars)")
        print(f"[BACKTEST] Strategy: Dynamically adjust chunk size if OUT_OF_RESOURCES")
        
        # Initialize streaming trade log writer if enabled
        if self.trade_log_enabled:
            trade_log_path = Path('logs') / 'trade_logs.csv'
            trade_log_path.parent.mkdir(exist_ok=True)
            
            # Clear existing file and write header
            with open(trade_log_path, 'w', newline='') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow([
                    'BotID', 'Cycle', 'EntryPrice', 'ExitPrice', 'EntryBar', 'ExitBar',
                    'Leverage', 'PnL', 'Direction', 'ChunkID', 'OutOfCycle', 'Signature'
                ])
            
            # Initialize streaming writer
            batch_size = int(os.getenv('TRADE_LOG_BATCH_SIZE', '100'))
            max_queue = int(os.getenv('TRADE_LOG_MAX_QUEUE', '1000'))
            self.trade_log_writer = TradeLogStreamWriter(
                trade_log_path, 
                batch_size=batch_size, 
                max_queue_size=max_queue
            )
            self.trade_log_writer.start()
            log_info(f"Trade log streaming writer initialized (batch={batch_size}, queue={max_queue})")

        # Try processing with current chunk size, halve on failure

        while True:
            try:
                # Create data chunks covering the full dataset
                data_chunks = self._chunk_full_data(ohlcv_data, self._optimal_data_chunk_bars)
                num_chunks = len(data_chunks)
                
                print(f"[BACKTEST] Split into {num_chunks} chunks (maximum GPU parallelization per chunk)")

                # Process each chunk with ALL cycles that overlap
                all_cycle_results = [[] for _ in range(num_cycles)]  # results[cycle_idx][bot_idx]

                with tqdm(total=num_chunks, desc="Processing chunks", unit="chunk") as pbar:
                    for chunk_idx, chunk_info in enumerate(data_chunks):
                        chunk_data = chunk_info['data']
                        chunk_start = chunk_info['global_start']
                        chunk_end = chunk_info['global_end']
                        
                        # Find all cycles that overlap with this chunk
                        active_cycles = []
                        cycle_indices = []
                        
                        for cycle_idx, (cycle_start, cycle_end) in enumerate(cycles):
                            # Check if cycle overlaps with this chunk
                            if cycle_start < chunk_end and cycle_end > chunk_start:
                                # Calculate the cycle's range within this chunk
                                cycle_chunk_start = max(0, cycle_start - chunk_start)
                                cycle_chunk_end = min(len(chunk_data), cycle_end - chunk_start)
                                
                                if cycle_chunk_end > cycle_chunk_start:
                                    active_cycles.append((cycle_chunk_start, cycle_chunk_end))
                                    cycle_indices.append(cycle_idx)
                        
                        if not active_cycles:
                            pbar.update(1)
                            continue
                        
                        num_active_cycles = len(active_cycles)
                        pbar.set_description(f"Chunk {chunk_idx+1}/{num_chunks} ({num_active_cycles} cycles, {num_bots * num_active_cycles:,} parallel tasks)")
                        
                        # Precompute indicators for this chunk ONCE
                        indicators_buffer = self._precompute_indicators(chunk_data)
                        
                        # Compute HTF indicators if MTF enabled
                        htf_indicators_buffer = None
                        num_htf_bars = 0
                        if self.enable_mtf:
                            htf_indicators_buffer = self._compute_htf_indicators(indicators_buffer, len(chunk_data))
                            if htf_indicators_buffer is not None:
                                num_htf_bars = len(chunk_data) // self.htf_multiplier
                        
                        # Process all cycles at once (chunk size was optimized for this)
                        chunk_results = self._run_parallel_bot_cycle_kernel(
                            bots,
                            chunk_data,
                            indicators_buffer,
                            active_cycles,
                            cycle_indices,
                            num_active_cycles,
                            chunk_idx,
                            chunk_start,
                            chunk_end,
                            htf_indicators_buffer,
                            num_htf_bars
                        )
                        
                        # Cleanup
                        indicators_buffer.release()
                        if htf_indicators_buffer is not None:
                            htf_indicators_buffer.release()
                        
                        # Distribute results to appropriate cycles
                        for local_cycle_idx, global_cycle_idx in enumerate(cycle_indices):
                            # Extract results for this cycle (one per bot)
                            # Convert dict results to simple tuples for aggregation
                            cycle_data = []
                            for bot_idx in range(num_bots):
                                result_dict = chunk_results[bot_idx][local_cycle_idx]
                                # Store as tuple: (bot_idx, trades, wins, pnl, signals)
                                cycle_data.append((
                                    bots[bot_idx].bot_id,
                                    result_dict['trades'],
                                    result_dict['wins'],
                                    result_dict['pnl'],
                                    result_dict['signals']  # NEW
                                ))
                            all_cycle_results[global_cycle_idx].append(cycle_data)
                        
                        # Explicitly delete large chunk data to free memory immediately
                        del chunk_results
                        del chunk_data
                        
                        pbar.update(1)
                
                # Explicit cleanup after chunk loop to release accumulated memory
                    import gc
                    gc.collect()
                # Success! Break out of retry loop
                break
                
            except cl.RuntimeError as e:
                if "OUT_OF_RESOURCES" in str(e):
                    # Halve the chunk size and retry
                    old_chunk_size = self._optimal_data_chunk_bars
                    self._optimal_data_chunk_bars = max(1440, self._optimal_data_chunk_bars // 2)  # Minimum 1 day
                    
                    print(f"\n[BACKTEST] OUT_OF_RESOURCES - Halving chunk size: {old_chunk_size/1440:.1f} → {self._optimal_data_chunk_bars/1440:.1f} days")
                    print(f"[BACKTEST] Retrying with {self._optimal_data_chunk_bars:,} bars ({self._optimal_data_chunk_bars/1440:.1f} days)...")
                    
                    if self._optimal_data_chunk_bars < 1440:
                        raise RuntimeError("Cannot reduce chunk size below 1 day - GPU resources insufficient")
                    
                    # Retry with smaller chunks
                    continue
                else:
                    # Different error, re-raise
                    raise

        # Force GPU synchronization and memory cleanup before aggregation
        # This ensures all previous buffers are fully released and memory is defragmented
        print("[BACKTEST] Finalizing GPU memory cleanup before aggregation...")
        self.queue.finish()
        
        # Trigger Python garbage collection to release host-side buffer references
        import gc
        gc.collect()
        
        # Check available VRAM before aggregation
        try:
            # Try a small test allocation to verify GPU is responsive
            test_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=1024)
            test_buf.release()
        except cl.RuntimeError:
            print("[BACKTEST] WARNING: GPU memory critically low, forcing additional cleanup...")
            gc.collect()
            import time
            time.sleep(0.1)  # Brief pause for driver cleanup
        
        # GPU-accelerated aggregation with batching to manage memory
        print("[BACKTEST] GPU-accelerated aggregation of multi-cycle results...")
        
        # Flatten all chunk results into single array for GPU processing
        flat_bot_ids = []
        flat_cycle_ids = []
        flat_trades = []
        flat_wins = []
        flat_pnls = []
        flat_signals = []  # NEW
        
        # DEBUG: Track what we're aggregating
        debug_cycle_counts = {}
        
        for cycle_idx in range(num_cycles):
            for chunk_data in all_cycle_results[cycle_idx]:
                for bot_id, trades, wins, pnl, signals in chunk_data:  # UPDATED
                    flat_bot_ids.append(bot_id)
                    flat_cycle_ids.append(cycle_idx)
                    flat_trades.append(trades)
            
                    flat_wins.append(wins)
                    flat_pnls.append(pnl)
                    flat_signals.append(signals)  # NEW
                    
                    # DEBUG
                    key = f"Bot{bot_id}_Cycle{cycle_idx}"
                    if key not in debug_cycle_counts:
                        debug_cycle_counts[key] = {'chunks': 0, 'total_trades': 0}
                    debug_cycle_counts[key]['chunks'] += 1
                    debug_cycle_counts[key]['total_trades'] += trades
        
        # DEBUG: Print aggregation input for first bot
        if len(debug_cycle_counts) > 0 and num_bots <= 10:
            print(f"  [DEBUG] Aggregation input data points:")
            for key, data in sorted(debug_cycle_counts.items())[:10]:
                print(f"    {key}: {data['chunks']} chunk(s), {data['total_trades']} trades total")
        
        # Convert to numpy arrays
        flat_bot_ids = np.array(flat_bot_ids, dtype=np.int32)
        flat_cycle_ids = np.array(flat_cycle_ids, dtype=np.int32)
        flat_trades = np.array(flat_trades, dtype=np.int32)
        flat_wins = np.array(flat_wins, dtype=np.int32)
        flat_pnls = np.array(flat_pnls, dtype=np.float32)
        flat_signals = np.array(flat_signals, dtype=np.int32)  # NEW
        
        # Free memory from original chunk data
        del all_cycle_results
        import gc
        gc.collect()
        
        # Run GPU aggregation in batches to avoid OUT_OF_RESOURCES
        final_results = self._gpu_aggregate_results_batched(
            bots,
            flat_bot_ids,
            flat_cycle_ids,
            flat_trades,
            flat_wins,
            flat_pnls,
            flat_signals,
            num_cycles
        )
        
        # Ensure all trade logs are written before returning
        if self.trade_log_writer:
            stats = self.trade_log_writer.get_stats()
            log_info(f"Shutting down trade log writer (enqueued: {stats['enqueued']}, written: {stats['written']}, pending: {stats['queue_size']})")
            self.trade_log_writer.shutdown(timeout=60.0)
            final_stats = self.trade_log_writer.get_stats()
            log_info(f"Trade log writer shutdown complete (total written: {final_stats['written']})")

        print("[OK] Ultra-parallel backtesting completed successfully!")
        return final_results

    def _chunk_full_data(
        self,
        ohlcv_data: np.ndarray,
        chunk_size_bars: int
    ) -> List[Dict]:
        """
        Split FULL dataset into chunks that will be tested with ALL cycles.
        Each chunk contains a window of data, and we'll run all relevant cycles on it.
        
        Returns:
            List of dicts with 'data', 'global_start', 'global_end'
        """
        chunks = []
        total_bars = len(ohlcv_data)
        
        # Add overlap for indicator lookback
        overlap_bars = 200  # For SMA(200) and other indicators
        
        chunk_start = 0
        chunk_id = 0
        
        while chunk_start < total_bars:
            # Calculate chunk end
            chunk_end = min(chunk_start + chunk_size_bars, total_bars)
            
            # Extract chunk with lookback
            data_start = max(0, chunk_start - overlap_bars)
            chunk_data_slice = ohlcv_data[data_start:chunk_end]
            
            chunks.append({
                'id': chunk_id,
                'data': chunk_data_slice,
                'global_start': chunk_start,
                'global_end': chunk_end,
                'lookback_offset': chunk_start - data_start
            })
            
            chunk_id += 1
            chunk_start = chunk_end  # No overlap in processing, only for indicators
        
        return chunks
    
    def _aggregate_data_chunks_for_cycle(
        self,
        bot_chunks: List[List[BacktestResult]]
    ) -> List[BacktestResult]:
        """
        Aggregate results from multiple data chunks for each bot within a cycle.
        
        Args:
            bot_chunks: List where bot_chunks[bot_idx] = [result1, result2, ...] for that bot
        
        Returns:
            List of aggregated BacktestResult, one per bot
        """
        aggregated = []
        
        for bot_chunk_results in bot_chunks:
            if not bot_chunk_results:
                continue
            
            # Sum trades across chunks
            total_trades = sum(r.total_trades for r in bot_chunk_results)
            total_wins = sum(r.winning_trades for r in bot_chunk_results)
            total_losses = sum(r.losing_trades for r in bot_chunk_results)
            
            # Average metrics (weighted by number of trades if needed)
            if total_trades > 0:
                # Weight by trades in each chunk
                avg_win = sum(
                    r.avg_win * r.winning_trades for r in bot_chunk_results if r.winning_trades > 0
                ) / total_wins if total_wins > 0 else 0.0
                
                avg_loss = sum(
                    r.avg_loss * r.losing_trades for r in bot_chunk_results if r.losing_trades > 0
                ) / total_losses if total_losses > 0 else 0.0
                
                win_rate = total_wins / total_trades if total_trades > 0 else 0.0
            else:
                avg_win = 0.0
                avg_loss = 0.0
                win_rate = 0.0
            
            # Take worst drawdown and average other metrics
            max_drawdown = max(r.max_drawdown for r in bot_chunk_results)
            avg_sharpe = sum(r.sharpe_ratio for r in bot_chunk_results) / len(bot_chunk_results)
            avg_profit_factor = sum(r.profit_factor for r in bot_chunk_results) / len(bot_chunk_results)
            
            # Use last chunk's final balance
            final_balance = bot_chunk_results[-1].final_balance
            
            result = BacktestResult(
                bot_id=bot_chunk_results[0].bot_id,
                total_trades=total_trades,
                winning_trades=total_wins,
                losing_trades=total_losses,
                per_cycle_trades=[],
                per_cycle_wins=[],
                per_cycle_pnl=[],
                total_pnl=sum(r.total_pnl for r in bot_chunk_results),
                max_drawdown=max_drawdown,
                sharpe_ratio=avg_sharpe,
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                profit_factor=avg_profit_factor,
                max_consecutive_wins=max(r.max_consecutive_wins for r in bot_chunk_results),
                max_consecutive_losses=max(r.max_consecutive_losses for r in bot_chunk_results),
                final_balance=final_balance
            )
            
            aggregated.append(result)
        
        return aggregated

    def _aggregate_cycle_results(
        self,
        all_cycle_results: List[List[BacktestResult]],
        num_bots: int,
        num_cycles: int
    ) -> List[BacktestResult]:
        """
        Aggregate results from multiple cycles for each bot.
        
        all_cycle_results[cycle_idx][bot_idx] = result for bot_idx in cycle_idx
        Returns: List of BacktestResult with combined metrics across all cycles
        """
        log_info(f"Aggregating results from {num_cycles} cycles for {num_bots} bots")
        
        final_results = []
        
        for bot_idx in range(num_bots):
            # Collect this bot's results from all cycles
            bot_cycle_results = [cycle_results[bot_idx] for cycle_results in all_cycle_results]
            
            # Combine metrics across cycles
            total_trades = sum(r.total_trades for r in bot_cycle_results)
            total_wins = sum(r.winning_trades for r in bot_cycle_results)
            total_losses = sum(r.losing_trades for r in bot_cycle_results)
            
            # Average metrics across cycles
            avg_win = sum(r.avg_win for r in bot_cycle_results) / num_cycles
            avg_loss = sum(r.avg_loss for r in bot_cycle_results) / num_cycles
            avg_win_rate = sum(r.win_rate for r in bot_cycle_results) / num_cycles
            avg_profit_factor = sum(r.profit_factor for r in bot_cycle_results) / num_cycles
            avg_max_drawdown = sum(r.max_drawdown for r in bot_cycle_results) / num_cycles
            avg_sharpe = sum(r.sharpe_ratio for r in bot_cycle_results) / num_cycles
            
            # Create aggregated result
            aggregated = BacktestResult(
                bot_id=bot_cycle_results[0].bot_id,
                total_trades=total_trades,
                winning_trades=total_wins,
                losing_trades=total_losses,
                per_cycle_trades=[r.total_trades for r in bot_cycle_results],
                per_cycle_wins=[r.winning_trades for r in bot_cycle_results],
                per_cycle_pnl=[r.total_pnl for r in bot_cycle_results],
                total_pnl=sum(r.total_pnl for r in bot_cycle_results),
                max_drawdown=avg_max_drawdown,
                sharpe_ratio=avg_sharpe,
                win_rate=avg_win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                profit_factor=avg_profit_factor,
                max_consecutive_wins=max(r.max_consecutive_wins for r in bot_cycle_results),
                max_consecutive_losses=max(r.max_consecutive_losses for r in bot_cycle_results),
                final_balance=bot_cycle_results[-1].final_balance  # Use last cycle's final balance
            )
            
            final_results.append(aggregated)
        
        return final_results

    def _verify_trade_log_consistency(self, results: List[BacktestResult]) -> None:
        """
        Optional verification: read `logs/trade_logs.csv` and ensure per-cycle
        PnL sums match `results.per_cycle_pnl`. Only run if env var
        `ENFORCE_LOG_CHECK` is set to '1'.
        """
        import os
        if os.getenv('ENFORCE_LOG_CHECK', '0') != '1':
            return

        trade_csv = Path('logs') / 'trade_logs.csv'
        if not trade_csv.exists():
            print('[VERIFY] No trade_logs.csv found, skipping consistency check')
            return

        # Accumulate per-bot, per-cycle sums (ignore OutOfCycle and duplicates)
        from collections import defaultdict
        trade_acc = defaultdict(lambda: defaultdict(float))
        sig_map = defaultdict(lambda: defaultdict(set))

        with open(trade_csv, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=';')
            for r in reader:
                try:
                    b = int(r['BotID'])
                    c = int(r['Cycle'])
                    pnl = float(str(r['PnL']).replace(',', '.'))
                    oc = int(r.get('OutOfCycle') or 0)
                except Exception:
                    continue
                sig = (int(r.get('EntryBar') or 0), int(r.get('ExitBar') or 0), r.get('Direction'))
                if sig in sig_map[b][c]:
                    continue
                sig_map[b][c].add(sig)
                if not oc:
                    trade_acc[b][c] += pnl

        # Compare
        mismatches = []
        for res in results:
            bot_id = res.bot_id
            for cycle_idx, expected in enumerate(res.per_cycle_pnl):
                actual = trade_acc.get(bot_id, {}).get(cycle_idx, 0.0)
                if abs(float(actual) - float(expected)) > 0.01:
                    mismatches.append((bot_id, cycle_idx, expected, actual))

        if mismatches:
            print('[VERIFY] Found per-cycle mismatches between trade logs and kernel results:')
            for bot, cycle, expected, actual in mismatches:
                print(f'  Bot {bot} cycle {cycle}: kernel {expected} != logs {actual}')
            # Optionally reconcile results with logs if requested via env var
            if os.getenv('RECONCILE_WITH_LOGS', '0') == '1':
                print('[VERIFY] Reconciling kernel per-cycle PnL with per-trade logs (RECONCILE_WITH_LOGS=1)')
                # Create mapping per bot/cycle
                trade_acc = trade_acc
                for res in results:
                    bot_id = res.bot_id
                    # Replace per-cycle pnl if available
                    for c_idx in range(len(res.per_cycle_pnl)):
                        if bot_id in trade_acc and c_idx in trade_acc[bot_id]:
                            res.per_cycle_pnl[c_idx] = float(trade_acc[bot_id][c_idx])

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
        Backtest all bots against all data chunks with parallel multi-workload processing.

        NEW APPROACH: Process multiple cycles/batches in parallel simultaneously
        to maximize GPU utilization by running independent workloads concurrently.
        """
        num_bots = len(bots)
        all_chunk_results = []

        log_info(f"Processing {len(data_chunks)} data chunks with {num_bots} bots each")

        # === PHASE 1: PRELOAD ALL CHUNKS INTO MEMORY ===
        log_info("Preloading all data chunks into memory for parallel processing...")
        self._maybe_debug_log(f"[DEBUG] About to preload {len(data_chunks)} chunks")
        preloaded_chunks = []

        for chunk in data_chunks:
            chunk_id = chunk['id']
            chunk_data = chunk['data']
            chunk_cycles = chunk['cycles']
            
            self._maybe_debug_log(f"[DEBUG] Preloading chunk {chunk_id}: {len(chunk_data)} bars, {len(chunk_cycles)} cycles")

            # Precompute indicators for this chunk (done once upfront)
            self._maybe_debug_log(f"[DEBUG] Starting indicator precomputation for chunk {chunk_id}...")
            indicators_buffer = self._precompute_indicators(chunk_data)
            self._maybe_debug_log(f"[DEBUG] ✓ Indicators precomputed for chunk {chunk_id}")

            preloaded_chunks.append({
                'id': chunk_id,
                'data': chunk_data,
                'cycles': chunk_cycles,
                'indicators_buffer': indicators_buffer
            })

        log_info(f"Successfully preloaded {len(preloaded_chunks)} chunks into memory")

        # === PHASE 2: PARALLEL MULTI-WORKLOAD PROCESSING ===
        # Process ALL chunks simultaneously for maximum GPU utilization
        # Each chunk becomes a separate workload running concurrently
        max_parallel_workloads = len(preloaded_chunks)  # Process ALL chunks in parallel
        log_info(f"Processing ALL {max_parallel_workloads} chunks simultaneously for maximum GPU utilization")

        start_time = time.time()

        # Process ALL chunks simultaneously for maximum parallelism
        # No batching - all workloads run at once
        batch_chunks = preloaded_chunks  # All chunks in one batch

        # Process this batch with multiple parallel workloads
        batch_results = self._process_parallel_workloads_all_at_once(bots, batch_chunks)

        # Collect results from this batch
        all_chunk_results.extend(batch_results)

        elapsed = time.time() - start_time
        log_info(f"Parallel GPU processing: {len(preloaded_chunks)} chunks completed in {elapsed:.3f}s")

        # Cleanup all preloaded indicator buffers
        for chunk in preloaded_chunks:
            chunk['indicators_buffer'].release()

        # Aggregate results across all chunks
        aggregated_results = self._aggregate_chunk_results(all_chunk_results, num_bots)

        return aggregated_results
    
    def _process_parallel_workloads(
        self,
        bots: List[CompactBotConfig],
        batch_chunks: List[Dict]
    ) -> List[BacktestResult]:
        """
        Process multiple chunks in parallel workloads to maximize GPU utilization.
        
        Each workload runs independently on the GPU, processing different chunks simultaneously.
        This keeps all 80 compute units busy with multiple concurrent kernels.
        """
        batch_results = []
        
        # For maximum parallelism, process all chunks in this batch simultaneously
        # Each chunk becomes a separate workload running on the GPU
        max_concurrent = len(batch_chunks)  # Process all chunks in parallel
        
        log_debug(f"Running {max_concurrent} parallel workloads for {len(batch_chunks)} chunks")
        
        def process_single_workload(chunk):
            """Process one chunk as an independent workload."""
            chunk_id = chunk['id']
            chunk_data = chunk['data']
            chunk_cycles = chunk['cycles']
            indicators_buffer = chunk['indicators_buffer']
            
            # Run the backtest kernel for this workload
            chunk_results = self._run_backtest_kernel_direct(bots, chunk_data, indicators_buffer, chunk_cycles, chunk['global_start'], chunk['global_end'])
            
            # Add chunk metadata to results
            for result in chunk_results:
                result.chunk_id = chunk_id
                result.chunk_bars = len(chunk_data)
            
            return chunk_results
        
        # Process all chunks in parallel using threading
        # Each thread runs a separate GPU kernel workload
        # Collect ALL results at the end to avoid interruptions during processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            # Submit all workloads - they will run simultaneously
            workload_futures = [executor.submit(process_single_workload, chunk) for chunk in batch_chunks]
            
            # Wait for ALL workloads to complete, then collect results
            # This eliminates interruptions during processing and maximizes GPU utilization
            concurrent.futures.wait(workload_futures, return_when=concurrent.futures.ALL_COMPLETED)
            
            # Now collect all results at once
            for future in workload_futures:
                workload_results = future.result()
                batch_results.extend(workload_results)
        
        return batch_results
    
    def _process_parallel_workloads_all_at_once(
        self,
        bots: List[CompactBotConfig],
        batch_chunks: List[Dict]
    ) -> List[BacktestResult]:
        """
        Process chunks sequentially for GPU stability.
        
        GPU operations must be serialized to avoid resource conflicts.
        """
        batch_results = []
        
        self._maybe_debug_log(f"[DEBUG] About to process {len(batch_chunks)} chunks")
        
        # Process chunks sequentially with progress tracking
        with tqdm(total=len(batch_chunks), desc="Processing chunks", unit="chunk") as pbar:
            self._maybe_debug_log(f"[DEBUG] Entered tqdm context, starting loop over {len(batch_chunks)} chunks")
            for chunk in batch_chunks:
                self._maybe_debug_log(f"[DEBUG] Loop iteration started for chunk {chunk.get('id', '?')}")
                chunk_id = chunk['id']
                chunk_data = chunk['data']
                chunk_cycles = chunk['cycles']
                indicators_buffer = chunk['indicators_buffer']
                
                # Run the backtest kernel for this chunk
                chunk_results = self._run_backtest_kernel_direct(bots, chunk_data, indicators_buffer, chunk_cycles, chunk['global_start'], chunk['global_end'])
                
                # Add chunk metadata to results
                for result in chunk_results:
                    result.chunk_id = chunk_id
                    result.chunk_bars = len(chunk_data)
                
                batch_results.extend(chunk_results)
                pbar.update(1)
        
        return batch_results
    
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
        results = self._run_backtest_kernel_direct(bots, chunk_data, indicators_buffer, chunk_cycles, 0, num_bars)
        
        # Cleanup
        indicators_buffer.release()
        
        return results
    
    def _run_backtest_kernel_direct(
        self,
        bots: List[CompactBotConfig],
        ohlcv_data: np.ndarray,
        indicators_buffer: cl.Buffer,
        cycles: List[Tuple[int, int]]
        , chunk_global_start: int = 0
        , chunk_global_end: int = 0
    ) -> List[BacktestResult]:
        """
        Run backtest kernel directly for all bots without batching.
        
        Optimized for data chunking approach where we process all bots against one chunk.
        """
        self._maybe_debug_log(f"[DEBUG] === _run_backtest_kernel_direct ENTERED ===")
        num_bots = len(bots)
        num_bars = len(ohlcv_data)
        num_cycles = len(cycles)
        self._maybe_debug_log(f"[DEBUG] Params: {num_bots} bots, {num_bars} bars, {num_cycles} cycles")
        
        # Serialize bot configs
        self._maybe_debug_log(f"[DEBUG] Serializing bots...")
        bot_configs_raw = self._serialize_bots(bots)
        self._maybe_debug_log(f"[DEBUG] ✓ Bots serialized ({bot_configs_raw.nbytes} bytes)")
        
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
        # Optional GPU trade logging buffers
        trade_logs_buf = None
        trade_log_index_buf = None
        if self.trade_log_enabled:
            trade_logs_buf = cl.Buffer(
                self.ctx,
                cl.mem_flags.WRITE_ONLY,
                size=self.trade_log_max * TRADE_LOG_DTYPE.itemsize
            )

            trade_log_index_host = np.array([0], dtype=np.int32)
            trade_log_index_buf = cl.Buffer(
                self.ctx,
                cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=trade_log_index_host
            )
        # Close counters for diagnostics (always allocate so kernel can write)
        # CRITICAL: Check if bot 0 writes 99999 to [0] - proves kernel entered
        close_counters_host = np.zeros(num_bots * num_cycles, dtype=np.int32)
        close_counters_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=close_counters_host)
        # Allocate per-bot per-cycle filter debug buffer (bitmask of failing filters)
        filter_debug_host = np.zeros(num_bots * num_cycles, dtype=np.int32)
        filter_debug_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=filter_debug_host)
        # Allocate per-bot per-filter counters (atomic counts for each filter triggered across all cycles)
        NUM_FILTERS = 6  # ADX, ATR, VOLUME, SR, RSI, NAN
        filter_count_host = np.zeros(num_bots * NUM_FILTERS, dtype=np.int32)
        filter_count_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=filter_count_host)
        
        # Respect env var to disable verbose GPU debug logging (reduces memory and logs)
        disable_gpu_logs = os.getenv("DISABLE_GPU_LOGGING", "0") == "1"
        if not disable_gpu_logs:
            self._maybe_debug_log(f"[DEBUG] close_counters[0] BEFORE kernel: {close_counters_host[0]}")
        
        # Execute backtest kernel for all bots
        kernel = self._backtest_kernel
        global_size = (num_bots,)
        local_size = None  # Let OpenCL choose optimal work group size
        
        # DEBUG: Print kernel invocation parameters
        self._maybe_debug_log(f"[DEBUG] Launching backtest kernel:")
        log_info(f"  - num_bots: {num_bots}")
        log_info(f"  - num_bars: {num_bars}")
        log_info(f"  - num_cycles: {num_cycles}")
        log_info(f"  - global_size: {global_size}")
        log_info(f"  - local_size: {local_size}")
        log_info(f"  - initial_balance: {self.initial_balance}")
        log_info(f"  - chunk_global_start: {chunk_global_start}")
        log_info(f"  - chunk_global_end: {chunk_global_end if chunk_global_end else num_bars}")
        self._maybe_debug_log(f"[DEBUG] Kernel enqueued, waiting for execution...")
        # Local-style fallback (kept for backward compatibility), main values come from self.debug_*
        debug_disable_filters = self.debug_disable_filters
        debug_bypass_sr = self.debug_bypass_sr
        debug_bypass_volume = self.debug_bypass_volume
        
        import time
        start_time = time.time()
        
        try:
            # Enqueue the kernel
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
                results_buf,                       # Results output
                trade_logs_buf if trade_logs_buf is not None else cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, 1),
                trade_log_index_buf if trade_log_index_buf is not None else cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, 4),
                cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.arange(num_cycles, dtype=np.int32))
                , np.int32(0)
                , np.int32(chunk_global_start)
                , np.int32(chunk_global_end if chunk_global_end else num_bars)
                , close_counters_buf
                , np.int32(self.debug_disable_filters)
                , np.int32(self.debug_bypass_sr)
                , np.int32(self.debug_bypass_volume)
                , np.int32(self.debug_force_signals)
                , filter_debug_buf
                , filter_count_buf
            )
            
            self._maybe_debug_log(f"[DEBUG] Kernel enqueued successfully, waiting up to 60s...")
            
            # Simple timeout approach - wait with timeout
            import threading
            
            timeout_seconds = 60
            finish_success = [False]
            finish_error = [None]
            
            def finish_kernel():
                try:
                    self.queue.finish()
                    finish_success[0] = True
                except Exception as e:
                    finish_error[0] = e
            
            finish_thread = threading.Thread(target=finish_kernel)
            finish_thread.daemon = True
            finish_thread.start()
            
            # Wait with progress updates and check for kernel entry
            for i in range(timeout_seconds):
                time.sleep(1)
                if finish_success[0]:
                    if not disable_gpu_logs:
                        self._maybe_debug_log(f"[DEBUG] ✓ Kernel completed after {time.time() - start_time:.1f}s")
                    break
                if finish_error[0]:
                    if not disable_gpu_logs:
                        self._maybe_debug_log(f"[DEBUG] ✗ Kernel error: {finish_error[0]}")
                    raise finish_error[0]
                
                # Check if kernel started by reading close_counters
                if i == 2 or i == 5:
                    check_counters = np.zeros(num_bots * num_cycles, dtype=np.int32)
                    cl.enqueue_copy(self.queue, check_counters, close_counters_buf, is_blocking=True)
                    if not disable_gpu_logs:
                        self._maybe_debug_log(f"[DEBUG] At {i}s: close_counters[0] = {check_counters[0]} (should be 99999 if kernel entered)")
                
                if i % 5 == 0:
                    if not disable_gpu_logs:
                        self._maybe_debug_log(f"[DEBUG] Still waiting... {i}s elapsed (Check GPU usage in Task Manager)")
            else:
                # Timeout reached
                self._maybe_debug_log(f"[DEBUG] ✗ TIMEOUT after {timeout_seconds}s - kernel hung!")
                self._maybe_debug_log(f"[DEBUG] The kernel was enqueued but never completed")
                self._maybe_debug_log(f"[DEBUG] GPU usage should be 60-100% but is likely 1-10%")
                self._maybe_debug_log(f"[DEBUG] This indicates kernel infinite loop or invalid memory access")
                raise RuntimeError(f"Kernel execution timeout after {timeout_seconds}s")
            
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
        # Optional: grab GPU trade logs
        if self.trade_log_enabled:
            idx_host = np.empty(1, dtype=np.int32)
            cl.enqueue_copy(self.queue, idx_host, trade_log_index_buf)
            self.queue.finish()
            count = int(idx_host[0])
            if count > 0:
                count = min(count, self.trade_log_max)
                trade_logs = np.empty(self.trade_log_max, dtype=TRADE_LOG_DTYPE)
                cl.enqueue_copy(self.queue, trade_logs, trade_logs_buf)
                self.queue.finish()
                trade_logs = trade_logs[:count]
                self._write_trade_logs_csv(trade_logs)

        # Read close counters for diagnostics and write CSV
            cl.enqueue_copy(self.queue, close_counters_host, close_counters_buf)
            # Read back per-filter debug bitmap and per-filter counters
            cl.enqueue_copy(self.queue, filter_debug_host, filter_debug_buf)
            cl.enqueue_copy(self.queue, filter_count_host, filter_count_buf)
        self.queue.finish()
        cc_path = Path('logs') / 'close_counters.csv'
        write_header = not cc_path.exists()
        with open(cc_path, 'a', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            if write_header:
                writer.writerow(['BotID', 'Cycle', 'KernelCloseCount'])
            for b_idx in range(num_bots):
                for c_idx in range(num_cycles):
                    kc = int(close_counters_host[b_idx * num_cycles + c_idx])
                    if kc > 0:
                        writer.writerow([bots[b_idx].bot_id, c_idx, kc])
        
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
    
    def _run_parallel_bot_cycle_kernel(
        self,
        bots: List[CompactBotConfig],
        ohlcv_data: np.ndarray,
        indicators_buffer: cl.Buffer,
        cycles: List[Tuple[int, int]],
        cycle_indices: List[int],
        num_active_cycles: int,
        chunk_id: int = 0,
        chunk_global_start: int = 0,
        chunk_global_end: int = 0,
        htf_indicators_buffer: cl.Buffer = None,
        num_htf_bars: int = 0
    ) -> List[List[Dict]]:
        """
        Run ultra-parallel kernel that processes each bot-cycle pair as a separate work item.
        
        Returns:
            results[bot_idx][cycle_idx] = {'trades': int, 'wins': int, 'pnl': float}
        """
        num_bots = len(bots)
        num_bars = len(ohlcv_data)
        num_cycles = num_active_cycles
        
        # Serialize bot configs
        bot_configs_raw = self._serialize_bots(bots)
        # Serialization and buffer creation (debug logs disabled for performance)
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

        # Map local cycle index -> global cycle index for correct logging
        cycle_global_idx = np.array(cycle_indices, dtype=np.int32)
        cycle_global_idx_buf = cl.Buffer(
            self.ctx,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=cycle_global_idx
        )
        
        # Results buffer: [bot_idx * num_cycles * 4 + cycle_idx * 4] = {trades, wins, pnl, signals}
        results_size = num_bots * num_cycles * 4  # 4 floats per bot-cycle pair
        results_buf = cl.Buffer(
            self.ctx,
            cl.mem_flags.WRITE_ONLY,
            size=results_size * 4  # 4 bytes per float
        )
        # Optional GPU trade logging buffers
        trade_logs_buf = None
        trade_log_index_buf = None
        if self.trade_log_enabled:
            trade_logs_buf = cl.Buffer(
                self.ctx,
                cl.mem_flags.WRITE_ONLY,
                size=self.trade_log_max * TRADE_LOG_DTYPE.itemsize
            )

            trade_log_index_host = np.array([0], dtype=np.int32)
            trade_log_index_buf = cl.Buffer(
                self.ctx,
                cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=trade_log_index_host
            )
        # Close counters buffer for kernel-level close diagnostics
        close_counters_host = np.zeros(num_bots * num_cycles, dtype=np.int32)
        close_counters_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=close_counters_host)
        # filter debug buffer for per-bot per-cycle filter bits
        filter_debug_host = np.zeros(num_bots * num_cycles, dtype=np.int32)
        filter_debug_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=filter_debug_host)
        # Per-bot per-filter counters for aggregated debug statistics (ADX/ATR/VOL/SR/RSI/NAN)
        NUM_FILTERS = 6
        filter_count_host = np.zeros(num_bots * NUM_FILTERS, dtype=np.int32)
        filter_count_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=filter_count_host)
        
        # Execute kernel for all bot-cycle pairs in parallel
        global_size = (num_bots * num_cycles,)
        # Determine bars per day based on timeframe - default 1m = 1440
        bars_per_day = 1440
        
        try:
            kernel = self._backtest_parallel_kernel
            debug_disable_filters = self.debug_disable_filters
            debug_bypass_sr = self.debug_bypass_sr
            debug_bypass_volume = self.debug_bypass_volume
            # Kernel invocation (debug logs disabled for performance)
            kernel(
                self.queue,
                global_size,
                None,
                bots_buf,
                ohlcv_buf,
                indicators_buffer,
                cycle_starts_buf,
                cycle_ends_buf,
                np.int32(num_bots),
                np.int32(num_cycles),
                np.int32(num_bars),
                np.float32(self.initial_balance),
                results_buf,
                trade_logs_buf if trade_logs_buf is not None else cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, 1),
                trade_log_index_buf if trade_log_index_buf is not None else cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, 4)
                ,cycle_global_idx_buf,
                np.int32(chunk_id),
                np.int32(chunk_global_start),
                np.int32(chunk_global_end)
                , close_counters_buf
                , htf_indicators_buffer if htf_indicators_buffer is not None else cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY, 4)
                , np.int32(num_htf_bars)
                , np.int32(self.htf_multiplier)
                , np.int32(1 if self.enable_mtf else 0)
                , np.int32(bars_per_day)
                , np.int32(self.debug_disable_filters)
                , np.int32(self.debug_bypass_sr)
                , np.int32(self.debug_bypass_volume)
                , np.int32(self.debug_force_signals)
                , filter_debug_buf
                , filter_count_buf
            )
            
            # Use a timeout-enabled finish to avoid hanging the process
            import threading, time as _time
            finish_success = [False]
            finish_error = [None]
            def finish_queue():
                try:
                    self.queue.finish()
                    finish_success[0] = True
                except Exception as _e:
                    finish_error[0] = _e
            finish_thread = threading.Thread(target=finish_queue)
            finish_thread.daemon = True
            finish_thread.start()
            # Poll for up to timeout_seconds
            timeout_seconds = 30
            poll_interval = 0.5
            start_t = _time.time()
            while _time.time() - start_t < timeout_seconds:
                if finish_success[0]:
                    self._maybe_debug_log(f"[DEBUG] Kernel finished within {(_time.time()-start_t):.2f}s")
                    break
                if finish_error[0]:
                    self._maybe_debug_log(f"[DEBUG] Kernel finish error: {finish_error[0]}")
                    raise finish_error[0]
                _time.sleep(poll_interval)
            else:
                # Timeout reached - try to fetch current GPU usage and record
                self._maybe_debug_log(f"[DEBUG] Kernel finish timed out after {timeout_seconds}s. Kernel might be hung.")
                # Read close counters buffer as a diagnostic attempt (non-blocking may also fail)
                try:
                    cl.enqueue_copy(self.queue, close_counters_host, close_counters_buf)
                    self._maybe_debug_log(f"[DEBUG] close_counters_sample: {close_counters_host[:min(10, len(close_counters_host))]}")
                except Exception as _e:
                    self._maybe_debug_log(f"[DEBUG] Failed to read close_counters during timeout handling: {_e}")
                # Attempt to cancel or raise error
                raise RuntimeError(f"Kernel timeout after {timeout_seconds}s - likely hang")
            
            self._maybe_debug_log("[DEBUG] Kernel finished, reading results...")
            # Read results
            self._maybe_debug_log(f"[DEBUG] Preparing to read results: results_size={results_size}")
            results_flat = np.empty(results_size, dtype=np.float32)
            self._maybe_debug_log("[DEBUG] Enqueuing copy to read results buffer...")
            cl.enqueue_copy(self.queue, results_flat, results_buf)
            self._maybe_debug_log("[DEBUG] Results copy enqueued (results_flat size %d)" % results_flat.size)
            # Optional: read trade logs if logging enabled
            trade_logs = None
            if self.trade_log_enabled:
                # Get count
                idx_host = np.empty(1, dtype=np.int32)
                self._maybe_debug_log("[DEBUG] Enqueuing copy to read trade_log_index_buf")
                # Try a blocking copy to force immediate result transfer; this may avoid hanging queue.finish
                try:
                    cl.enqueue_copy(self.queue, idx_host, trade_log_index_buf, is_blocking=True)
                except TypeError:
                    # Fallback (some pyopencl versions accept named param differently)
                    cl.enqueue_copy(self.queue, idx_host, trade_log_index_buf, True)
                
                # MEDIUM PRIORITY FIX: Trade log overflow detection
                trades_attempted = idx_host[0]
                if trades_attempted >= self.trade_log_max:
                    log_warning(f"⚠️ Trade log OVERFLOW: {trades_attempted} trades attempted but limit is {self.trade_log_max}")
                    log_warning(f"   Some trade data was lost. Consider increasing TRADE_LOG_MAX env var.")
                
                # Use timeout finish here in case GPU copy blocks
                self._maybe_debug_log("[DEBUG] Waiting for trade log index copy to finish (with timeout)")
                import time as _time, threading as _threading
                finish_success = [False]
                finish_error = [None]
                def _fin():
                    try:
                        self.queue.finish()
                        finish_success[0] = True
                    except Exception as _e:
                        finish_error[0] = _e
                _t = _threading.Thread(target=_fin)
                _t.daemon = True
                _t.start()
                _start = _time.time()
                _timeout = 30
                while _time.time() - _start < _timeout:
                    if finish_success[0]:
                        break
                    if finish_error[0]:
                        self._maybe_debug_log(f"[DEBUG] Trade log index copy finish error: {finish_error[0]}")
                        raise finish_error[0]
                    _time.sleep(0.1)
                else:
                    self._maybe_debug_log(f"[DEBUG] Trade log index copy finish timed out after {_timeout}s")
                    # Attempt to read close_counters buffer for diagnostics
                    try:
                        tmp = np.empty_like(close_counters_host)
                        cl.enqueue_copy(self.queue, tmp, close_counters_buf)
                        self._maybe_debug_log(f"[DEBUG] close_counters_sample (on timeout): {tmp[:min(10, len(tmp))]}")
                    except Exception as _e:
                        self._maybe_debug_log(f"[DEBUG] Failed reading close counters during trade_log_index timeout: {_e}")
                    raise RuntimeError("Trade log index copy finish timeout")
                count = int(idx_host[0])
                if count > 0:
                    count = min(count, self.trade_log_max)
                    trade_logs = np.empty(self.trade_log_max, dtype=TRADE_LOG_DTYPE)
                    self._maybe_debug_log(f"[DEBUG] Reading {count} trade logs for streaming write")
                    cl.enqueue_copy(self.queue, trade_logs, trade_logs_buf)
                    self.queue.finish()
                    trade_logs = trade_logs[:count]
                    
                    # Use streaming writer for async, buffered writes
                    if self.trade_log_writer:
                        self.trade_log_writer.enqueue(trade_logs)
                        stats = self.trade_log_writer.get_stats()
                        log_debug(f"Trade logs enqueued: {count} (total: {stats['enqueued']}, written: {stats['written']}, queued: {stats['queue_size']})")
                    else:
                        log_warning("Trade log writer not initialized, logs will be dropped")

            # Read close counters for diagnostics
            self._maybe_debug_log("[DEBUG] Enqueuing copy to read close_counters_buf (for diagnostics)")
            # Attempt blocking copy for diagnostics (should be fast)
            try:
                cl.enqueue_copy(self.queue, close_counters_host, close_counters_buf, is_blocking=True)
            except TypeError:
                cl.enqueue_copy(self.queue, close_counters_host, close_counters_buf, True)
            # Read per-filter debug bits from GPU
            try:
                cl.enqueue_copy(self.queue, filter_debug_host, filter_debug_buf, is_blocking=True)
            except TypeError:
                cl.enqueue_copy(self.queue, filter_debug_host, filter_debug_buf, True)
            # Read per-filter counter buffer
            try:
                cl.enqueue_copy(self.queue, filter_count_host, filter_count_buf, is_blocking=True)
            except TypeError:
                cl.enqueue_copy(self.queue, filter_count_host, filter_count_buf, True)
            # Timeout-safe finish to avoid hanging
            self._maybe_debug_log("[DEBUG] Waiting for close counters copy to finish (with timeout)")
            import time as _time, threading as _threading
            finish_success2 = [False]
            finish_error2 = [None]
            def _fin2():
                try:
                    self.queue.finish()
                    finish_success2[0] = True
                except Exception as _e:
                    finish_error2[0] = _e
            _t2 = _threading.Thread(target=_fin2)
            _t2.daemon = True
            _t2.start()
            _start2 = _time.time()
            _timeout2 = 30
            while _time.time() - _start2 < _timeout2:
                if finish_success2[0]:
                    break
                if finish_error2[0]:
                    self._maybe_debug_log(f"[DEBUG] Close counters copy finish error: {finish_error2[0]}")
                    raise finish_error2[0]
                _time.sleep(0.1)
            else:
                self._maybe_debug_log(f"[DEBUG] Close counters copy finish timed out after {_timeout2}s")
                raise RuntimeError("Close counters copy finish timed out")
            # Write close counters to CSV for later debug (append)
            cc_path = Path('logs') / 'close_counters.csv'
            write_header = not cc_path.exists()
            with open(cc_path, 'a', newline='') as f:
                writer = csv.writer(f, delimiter=';')
                if write_header:
                    writer.writerow(['BotID', 'Cycle', 'KernelCloseCount'])
                for b_idx in range(num_bots):
                    for c_idx in range(num_cycles):
                        kc = int(close_counters_host[b_idx * num_cycles + c_idx])
                        if kc > 0:
                            # Use global cycle index for CSV
                            global_cycle = cycle_global_idx[c_idx]
                            writer.writerow([bots[b_idx].bot_id, global_cycle, kc])
                # Append filter debug summary CSV for each bot-cycle
            fb_path = Path('logs') / 'filter_debug.csv'
            write_header2 = not fb_path.exists()
            with open(fb_path, 'a', newline='') as f:
                writer2 = csv.writer(f, delimiter=';')
                if write_header2:
                    writer2.writerow(['BotID', 'Cycle', 'FilterDebugBits'])
                for b_idx in range(num_bots):
                    for c_idx in range(num_cycles):
                        bits = int(filter_debug_host[b_idx * num_cycles + c_idx])
                        if bits != 0:
                            global_cycle = cycle_global_idx[c_idx]
                            writer2.writerow([bots[b_idx].bot_id, global_cycle, bits])
                # Write aggregated per-bot filter counts
                fc_path = Path('logs') / 'filter_debug_counts.csv'
                write_header_fc = not fc_path.exists()
                with open(fc_path, 'a', newline='') as fcnt:
                    writer_fc = csv.writer(fcnt, delimiter=';')
                    if write_header_fc:
                        writer_fc.writerow(['BotID', 'ADX', 'ATR', 'VOLUME', 'SR', 'RSI', 'NAN'])
                    for b_idx in range(num_bots):
                        row_counts = [int(filter_count_host[b_idx * NUM_FILTERS + i]) for i in range(NUM_FILTERS)]
                        if sum(row_counts) > 0:
                            writer_fc.writerow([bots[b_idx].bot_id] + row_counts)
            # (filter_debug_counts.csv already written above)
            
        except cl.RuntimeError as e:
            log_error(f"GPU kernel execution failed: {e}")
            if "OUT_OF_RESOURCES" in str(e):
                log_error(f"Workload too large: {num_bots} bots × {num_cycles} cycles = {num_bots * num_cycles:,} work items")
                log_error(f"Data chunk: {num_bars:,} bars")
                log_error("Consider reducing data chunk size or number of overlapping cycles")
            raise
        
        # Parse results into structured format
        results = [[{} for _ in range(num_cycles)] for _ in range(num_bots)]
        
        for bot_idx in range(num_bots):
            for cycle_idx in range(num_cycles):
                idx = bot_idx * num_cycles * 4 + cycle_idx * 4
                trades_val = int(results_flat[idx])
                wins_val = int(results_flat[idx + 1])
                pnl_val = results_flat[idx + 2]
                signals_val = int(results_flat[idx + 3])
                
                # DEBUG: Log for first bot
                if bot_idx == 0 and num_bots <= 10:
                    close_counter = int(close_counters_host[bot_idx * num_cycles + cycle_idx])
                    print(f"  [DEBUG] Bot {bots[bot_idx].bot_id} Cycle {cycle_idx}: kernel reports {trades_val} trades, close_counter={close_counter}")
                
                filter_bits = int(filter_debug_host[bot_idx * num_cycles + cycle_idx])
                results[bot_idx][cycle_idx] = {
                    'trades': trades_val,
                    'wins': wins_val,
                    'pnl': pnl_val,
                    'signals': signals_val
                }
                if filter_bits != 0:
                    results[bot_idx][cycle_idx]['filter_debug_bits'] = filter_bits
        
        # Cleanup
        bots_buf.release()
        ohlcv_buf.release()
        cycle_starts_buf.release()
        cycle_ends_buf.release()
        cycle_global_idx_buf.release()
        results_buf.release()
        
        return results
    
    def _calculate_max_drawdown(self, per_cycle_pnl: List[float]) -> float:
        """
        Calculate maximum drawdown from per-cycle PnL.
        Returns decimal format (0.0-1.0) to match GPU kernel output.
        """
        if not per_cycle_pnl:
            return 0.0
        
        cumulative = self.initial_balance
        peak = cumulative
        max_dd = 0.0
        
        for pnl in per_cycle_pnl:
            cumulative += pnl
            if cumulative > peak:
                peak = cumulative
            # Calculate as decimal (0.0-1.0), not percentage
            dd = ((peak - cumulative) / peak) if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
        
        # Cap at 1.0 (100% loss maximum)
        return min(max_dd, 1.0)
    
    def _calculate_sharpe_ratio(self, per_cycle_pnl: List[float]) -> float:
        """Calculate Sharpe ratio from per-cycle PnL."""
        if not per_cycle_pnl or len(per_cycle_pnl) < 2:
            return 0.0
        
        # Convert PnL to returns (percentage)
        returns = [(pnl / self.initial_balance * 100) for pnl in per_cycle_pnl]
        
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        if std_return == 0:
            return 0.0
        
        # Annualized Sharpe (assuming ~252 trading days, cycles are portions of that)
        sharpe = mean_return / std_return
        return sharpe
    
    def _gpu_aggregate_results_batched(
        self,
        bots: List[CompactBotConfig],
        bot_ids: np.ndarray,
        cycle_ids: np.ndarray,
        trades: np.ndarray,
        wins: np.ndarray,
        pnls: np.ndarray,
        signals: np.ndarray,  # NEW
        num_cycles: int
    ) -> List[BacktestResult]:
        """
        GPU aggregation with batching to manage memory.
        Processes bots in batches to avoid OUT_OF_RESOURCES errors.
        """
        num_bots = len(bots)
        num_data_points = len(bot_ids)
        
        # Determine batch size based on available memory
        # With 3.19GB VRAM and ~107MB for 100K bots, use 20K bot batches (~21MB each)
        batch_size = 20000
        num_batches = (num_bots + batch_size - 1) // batch_size
        
        print(f"  GPU aggregating {num_data_points:,} data points for {num_bots:,} bots in {num_batches} batches...")
        
        all_outputs = []
        
        # Process bots in batches
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, num_bots)
            batch_bots = bots[batch_start:batch_end]
            batch_num_bots = len(batch_bots)
            
            # Create bot_id lookup for this batch
            batch_bot_ids = set(b.bot_id for b in batch_bots)
            
            # Filter data points for this batch
            mask = np.isin(bot_ids, list(batch_bot_ids))
            batch_bot_ids_arr = bot_ids[mask]
            batch_cycle_ids_arr = cycle_ids[mask]
            batch_trades_arr = trades[mask]
            batch_wins_arr = wins[mask]
            batch_pnls_arr = pnls[mask]
            batch_signals_arr = signals[mask]  # NEW
            batch_data_points = len(batch_bot_ids_arr)
            
            if batch_data_points == 0:
                # No data for this batch, skip
                all_outputs.extend([[0, 0, 0.0] for _ in range(batch_num_bots)])
                continue
            
            # Upload batch data to GPU
            bot_ids_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=batch_bot_ids_arr)
            cycle_ids_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=batch_cycle_ids_arr)
            trades_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=batch_trades_arr)
            wins_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=batch_wins_arr)
            pnls_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=batch_pnls_arr)
            signals_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=batch_signals_arr)  # NEW
            
            # Create bot_id lookup array for this batch
            bot_id_lookup = np.array([b.bot_id for b in batch_bots], dtype=np.int32)
            bot_id_lookup_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=bot_id_lookup)
            
            # Output: aggregated results per bot [trades, wins, pnl, signals] × batch_num_bots
            output_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, size=batch_num_bots * 4 * 4)
            
            # Run aggregation kernel
            kernel = self._aggregate_flat_kernel
            global_size = (batch_num_bots,)
            
            kernel(
                self.queue,
                global_size,
                None,
                bot_id_lookup_buf,
                bot_ids_buf,
                cycle_ids_buf,
                trades_buf,
                wins_buf,
                pnls_buf,
                signals_buf,  # NEW
                np.int32(batch_num_bots),
                np.int32(batch_data_points),
                output_buf
            )
            
            self.queue.finish()
            
            # Read results
            output = np.empty(batch_num_bots * 4, dtype=np.float32)
            cl.enqueue_copy(self.queue, output, output_buf)
            self.queue.finish()
            
            # Store results
            for i in range(batch_num_bots):
                idx = i * 4
                all_outputs.append([
                    int(output[idx]),      # total_trades
                    int(output[idx + 1]),  # total_wins
                    output[idx + 2],       # total_pnl
                    int(output[idx + 3])   # total_signals  # NEW
                ])
            
            # Cleanup batch buffers
            bot_ids_buf.release()
            cycle_ids_buf.release()
            trades_buf.release()
            wins_buf.release()
            pnls_buf.release()
            bot_id_lookup_buf.release()
            output_buf.release()
            
            # Force memory cleanup between batches
            if batch_idx < num_batches - 1:
                import gc
                gc.collect()
        
        # Now build per-cycle results (CPU aggregation, already have the data)
        return self._build_final_results(bots, bot_ids, cycle_ids, trades, wins, pnls, signals, num_cycles, all_outputs)
    
    def _build_final_results(
        self,
        bots: List[CompactBotConfig],
        bot_ids: np.ndarray,
        cycle_ids: np.ndarray,
        trades: np.ndarray,
        wins: np.ndarray,
        pnls: np.ndarray,
        signals: np.ndarray,  # NEW
        num_cycles: int,
        all_outputs: List[List]
    ) -> List[BacktestResult]:
        """Build final BacktestResult objects with per-cycle data."""
        num_bots = len(bots)
        num_data_points = len(bot_ids)
        
        # Build per-cycle arrays by aggregating data points per cycle
        bot_cycle_map = {}  # {bot_id: {cycle_id: [trades, wins, pnl, signals]}}
        
        for i in range(num_data_points):
            b_id = bot_ids[i]
            c_id = cycle_ids[i]
            if b_id not in bot_cycle_map:
                bot_cycle_map[b_id] = {}
            if c_id not in bot_cycle_map[b_id]:
                bot_cycle_map[b_id][c_id] = [0, 0, 0.0, 0]  # Added signals
            bot_cycle_map[b_id][c_id][0] += trades[i]
            bot_cycle_map[b_id][c_id][1] += wins[i]
            bot_cycle_map[b_id][c_id][2] += pnls[i]
            bot_cycle_map[b_id][c_id][3] += signals[i]  # NEW
        
        # Convert to BacktestResult objects
        final_results = []
        for bot_idx in range(num_bots):
            bot = bots[bot_idx]
            
            total_trades = all_outputs[bot_idx][0]
            total_wins = all_outputs[bot_idx][1]
            total_pnl = all_outputs[bot_idx][2]
            total_signals = all_outputs[bot_idx][3]  # NEW
            
            # Build per-cycle arrays
            per_cycle_trades = []
            per_cycle_wins = []
            per_cycle_pnl = []
            per_cycle_signals = []  # NEW
            
            bot_data = bot_cycle_map.get(bot.bot_id, {})
            for cycle_idx in range(num_cycles):
                cycle_data = bot_data.get(cycle_idx, [0, 0, 0.0, 0])
                per_cycle_trades.append(cycle_data[0])
                per_cycle_wins.append(cycle_data[1])
                per_cycle_pnl.append(cycle_data[2])
                per_cycle_signals.append(cycle_data[3])  # NEW
            
            # Calculate metrics
            win_rate = (total_wins / total_trades * 100.0) if total_trades > 0 else 0.0
            sharpe = self._calculate_sharpe_ratio(per_cycle_pnl)
            max_dd = self._calculate_max_drawdown(per_cycle_pnl)
            
            losing_trades = total_trades - total_wins
            final_balance = self.initial_balance + total_pnl
            
            final_results.append(BacktestResult(
                bot_id=bot.bot_id,
                total_trades=total_trades,
                winning_trades=total_wins,
                losing_trades=losing_trades,
                per_cycle_trades=per_cycle_trades,
                per_cycle_wins=per_cycle_wins,
                per_cycle_pnl=per_cycle_pnl,
                per_cycle_signals=per_cycle_signals,  # NEW
                total_pnl=total_pnl,
                max_drawdown=max_dd,
                sharpe_ratio=sharpe,
                win_rate=win_rate,
                avg_win=0.0,
                avg_loss=0.0,
                profit_factor=0.0,
                max_consecutive_wins=0.0,
                max_consecutive_losses=0.0,
                final_balance=final_balance
            ))
        
        return final_results
    
    def _gpu_aggregate_results(
        self,
        bots: List[CompactBotConfig],
        bot_ids: np.ndarray,
        cycle_ids: np.ndarray,
        trades: np.ndarray,
        wins: np.ndarray,
        pnls: np.ndarray,
        num_cycles: int
    ) -> List[BacktestResult]:
        """
        Ultra-fast GPU aggregation of bot-cycle results.
        
        Sums trades/wins/pnl across all cycles and chunks for each bot in parallel.
        Each work item processes ONE bot and sums all its data points.
        """
        num_bots = len(bots)
        num_data_points = len(bot_ids)
        
        print(f"  GPU aggregating {num_data_points:,} data points for {num_bots:,} bots...")
        
        # Upload data to GPU
        bot_ids_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=bot_ids)
        cycle_ids_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=cycle_ids)
        trades_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=trades)
        wins_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=wins)
        pnls_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=pnls)
        
        # Create bot_id lookup array
        bot_id_lookup = np.array([b.bot_id for b in bots], dtype=np.int32)
        bot_id_lookup_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=bot_id_lookup)
        
        # Output: aggregated results per bot [trades, wins, pnl] × num_bots
        output_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, size=num_bots * 3 * 4)  # 3 floats per bot
        
        # Run aggregation kernel: one work item per bot
        kernel = self._aggregate_flat_kernel
        global_size = (num_bots,)
        
        kernel(
            self.queue,
            global_size,
            None,
            bot_id_lookup_buf,
            bot_ids_buf,
            cycle_ids_buf,
            trades_buf,
            wins_buf,
            pnls_buf,
            np.int32(num_bots),
            np.int32(num_data_points),
            output_buf
        )
        
        self.queue.finish()
        
        # Read results
        output = np.empty(num_bots * 3, dtype=np.float32)
        cl.enqueue_copy(self.queue, output, output_buf)
        self.queue.finish()
        
        # Cleanup
        bot_ids_buf.release()
        cycle_ids_buf.release()
        trades_buf.release()
        wins_buf.release()
        pnls_buf.release()
        bot_id_lookup_buf.release()
        output_buf.release()
        
        # Build per-cycle arrays by aggregating data points per cycle
        # Group data points by (bot_id, cycle_id) for per-cycle results
        bot_cycle_map = {}  # {bot_id: {cycle_id: [trades, wins, pnl]}}
        
        for i in range(num_data_points):
            b_id = bot_ids[i]
            c_id = cycle_ids[i]
            if b_id not in bot_cycle_map:
                bot_cycle_map[b_id] = {}
            if c_id not in bot_cycle_map[b_id]:
                bot_cycle_map[b_id][c_id] = [0, 0, 0.0]
            bot_cycle_map[b_id][c_id][0] += trades[i]
            bot_cycle_map[b_id][c_id][1] += wins[i]
            bot_cycle_map[b_id][c_id][2] += pnls[i]
        
        # Convert to BacktestResult objects
        final_results = []
        for bot_idx in range(num_bots):
            bot = bots[bot_idx]
            idx = bot_idx * 3
            
            total_trades = int(output[idx])
            total_wins = int(output[idx + 1])
            total_pnl = output[idx + 2]
            
            # Build per-cycle arrays
            per_cycle_trades = []
            per_cycle_wins = []
            per_cycle_pnl = []
            
            bot_data = bot_cycle_map.get(bot.bot_id, {})
            for cycle_idx in range(num_cycles):
                cycle_data = bot_data.get(cycle_idx, [0, 0, 0.0])
                per_cycle_trades.append(cycle_data[0])
                per_cycle_wins.append(cycle_data[1])
                per_cycle_pnl.append(cycle_data[2])
            
            win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0.0
            final_balance = self.initial_balance + total_pnl
            roi = (total_pnl / self.initial_balance * 100) if self.initial_balance > 0 else 0.0
            
            # Calculate risk metrics
            max_drawdown = self._calculate_max_drawdown(per_cycle_pnl)
            sharpe_ratio = self._calculate_sharpe_ratio(per_cycle_pnl)
            
            result = BacktestResult(
                bot_id=bot.bot_id,
                total_trades=total_trades,
                winning_trades=total_wins,
                losing_trades=total_trades - total_wins,
                per_cycle_trades=per_cycle_trades,
                per_cycle_wins=per_cycle_wins,
                per_cycle_pnl=per_cycle_pnl,
                total_pnl=total_pnl,
                final_balance=final_balance,
                win_rate=win_rate,
                fitness_score=roi,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                avg_win=0.0,
                avg_loss=0.0,
                profit_factor=0.0,
                max_consecutive_wins=0,
                max_consecutive_losses=0,
                generation_survived=0
            )
            
            final_results.append(result)
        # Optional: verify results with trade logs on host
        try:
            self._verify_trade_log_consistency(final_results)
        except Exception as e:
            log_error(f"Trade log consistency check failed: {e}")

        return final_results
    
    def _aggregate_chunk_results(
        self,
        all_chunk_results: List[BacktestResult],
        num_bots: int
    ) -> List[BacktestResult]:
        """
        GPU-accelerated aggregation of backtest results from all data chunks.

        NEW APPROACH: Use GPU to aggregate core metrics, CPU for complex operations.
        """
        if not all_chunk_results:
            return []

        num_chunks = len(all_chunk_results) // num_bots
        log_info(f"GPU aggregating results for {num_bots} bots across {num_chunks} chunks")

        # === PHASE 1: GPU AGGREGATION OF CORE METRICS ===
        start_time = time.time()

        # Serialize all chunk results for GPU processing
        gpu_results = self._serialize_chunk_results_for_gpu(all_chunk_results, num_bots, num_chunks)

        # Run GPU aggregation kernel
        aggregated_gpu_data = self._run_gpu_aggregation(gpu_results, num_bots, num_chunks)

        gpu_time = time.time() - start_time
        log_debug(f"GPU aggregation completed in {gpu_time:.4f}s")

        # === PHASE 2: CPU PROCESSING FOR COMPLEX OPERATIONS ===
        # Group per-cycle arrays by bot_id (CPU operation)
        bot_cycle_data = self._group_per_cycle_data(all_chunk_results, num_bots)

        # Combine GPU results with CPU cycle data and calculate derived metrics
        final_results = self._combine_gpu_cpu_results(aggregated_gpu_data, bot_cycle_data, num_bots)

        return final_results

    def _serialize_chunk_results_for_gpu(
        self,
        all_chunk_results: List[BacktestResult],
        num_bots: int,
        num_chunks: int
    ) -> np.ndarray:
        """
        Serialize chunk results into format expected by GPU aggregation kernel.

        Layout: [chunk][bot][result_fields]
        """
        # Result structure size in floats (GPU kernel expects float32)
        RESULT_SIZE_BYTES = 64 + (100 * 12)  # Base fields + max cycles
        RESULT_SIZE_FLOATS = RESULT_SIZE_BYTES // 4  # Convert to float32 count
        total_floats = num_chunks * num_bots * RESULT_SIZE_FLOATS

        # Create flat array for GPU (all float32)
        gpu_data = np.zeros(total_floats, dtype=np.float32)

        for i, result in enumerate(all_chunk_results):
            chunk_idx = i // num_bots
            bot_idx = i % num_bots
            offset = chunk_idx * num_bots * RESULT_SIZE_FLOATS + bot_idx * RESULT_SIZE_FLOATS

            # Pack core metrics that GPU will aggregate (float32 indices)
            gpu_data[offset + 1] = float(result.total_trades)        # TOTAL_TRADES_OFFSET // 4
            gpu_data[offset + 2] = float(result.winning_trades)      # WINNING_TRADES_OFFSET // 4
            gpu_data[offset + 3] = float(result.losing_trades)       # LOSING_TRADES_OFFSET // 4
            gpu_data[offset + 4] = result.total_pnl                  # TOTAL_PNL_OFFSET // 4
            gpu_data[offset + 5] = result.max_drawdown               # MAX_DRAWDOWN_OFFSET // 4

        return gpu_data

    def _run_gpu_aggregation(
        self,
        gpu_results: np.ndarray,
        num_bots: int,
        num_chunks: int
    ) -> np.ndarray:
        """
        Run GPU kernel to aggregate results across chunks.
        """
        # Input buffer: all chunk results
        input_buf = cl.Buffer(
            self.ctx,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=gpu_results
        )

        # Output buffer: aggregated results per bot (5 floats: trades, wins, losses, pnl, max_dd)
        output_size = num_bots * 5
        output_buf = cl.Buffer(
            self.ctx,
            cl.mem_flags.WRITE_ONLY,
            size=output_size * 4  # float32 = 4 bytes
        )

        # Execute aggregation kernel
        kernel = self._aggregate_kernel
        global_size = (num_bots,)
        local_size = None

        try:
            kernel(
                self.queue,
                global_size,
                local_size,
                input_buf,              # All chunk results
                output_buf,             # Aggregated output
                np.int32(num_chunks),   # Number of chunks
                np.int32(num_bots)      # Number of bots
            )

            self.queue.finish()

        except cl.RuntimeError as e:
            log_error(f"GPU aggregation kernel failed: {e}")
            input_buf.release()
            output_buf.release()
            raise

        # Read results
        aggregated_data = np.empty(output_size, dtype=np.float32)
        cl.enqueue_copy(self.queue, aggregated_data, output_buf)

        # Cleanup
        input_buf.release()
        output_buf.release()

        return aggregated_data

    def _group_per_cycle_data(
        self,
        all_chunk_results: List[BacktestResult],
        num_bots: int
    ) -> Dict[int, Dict]:
        """
        Group and aggregate per-cycle arrays by bot_id across all chunks.
        """
        bot_cycle_data = {}

        for result in all_chunk_results:
            bot_id = result.bot_id

            if bot_id not in bot_cycle_data:
                bot_cycle_data[bot_id] = {
                    'per_cycle_trades': [],
                    'per_cycle_wins': [],
                    'per_cycle_pnl': []
                }

            # For each cycle, add the chunk's contribution
            for i in range(len(result.per_cycle_trades)):
                # Extend arrays to accommodate this cycle if needed
                while len(bot_cycle_data[bot_id]['per_cycle_trades']) <= i:
                    bot_cycle_data[bot_id]['per_cycle_trades'].append(0)
                    bot_cycle_data[bot_id]['per_cycle_wins'].append(0)
                    bot_cycle_data[bot_id]['per_cycle_pnl'].append(0.0)

                # Aggregate this chunk's contribution to this cycle
                bot_cycle_data[bot_id]['per_cycle_trades'][i] += result.per_cycle_trades[i]
                bot_cycle_data[bot_id]['per_cycle_wins'][i] += result.per_cycle_wins[i]
                bot_cycle_data[bot_id]['per_cycle_pnl'][i] += result.per_cycle_pnl[i]

        return bot_cycle_data

    def _combine_gpu_cpu_results(
        self,
        aggregated_gpu_data: np.ndarray,
        bot_cycle_data: Dict[int, Dict],
        num_bots: int
    ) -> List[BacktestResult]:
        """
        Combine GPU-aggregated metrics with CPU cycle data and calculate derived metrics.
        """
        final_results = []

        for bot_id in range(num_bots):
            # Extract GPU-aggregated data
            offset = bot_id * 5
            total_trades = int(aggregated_gpu_data[offset])
            winning_trades = int(aggregated_gpu_data[offset + 1])
            losing_trades = int(aggregated_gpu_data[offset + 2])
            total_pnl = aggregated_gpu_data[offset + 3]
            max_drawdown = aggregated_gpu_data[offset + 4]

            # Get CPU cycle data
            cycle_data = bot_cycle_data.get(bot_id, {
                'per_cycle_trades': [],
                'per_cycle_wins': [],
                'per_cycle_pnl': []
            })

            # Calculate final balance
            final_balance = self.initial_balance + total_pnl

            # Create result object
            bot_result = BacktestResult(
                bot_id=bot_id,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                per_cycle_trades=cycle_data['per_cycle_trades'],
                per_cycle_wins=cycle_data['per_cycle_wins'],
                per_cycle_pnl=cycle_data['per_cycle_pnl'],
                total_pnl=total_pnl,
                max_drawdown=max_drawdown,
                sharpe_ratio=0.0,
                win_rate=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                profit_factor=0.0,
                max_consecutive_wins=0,
                max_consecutive_losses=0,
                final_balance=final_balance
            )

            # Calculate derived metrics
            self._calculate_derived_metrics(bot_result)
            final_results.append(bot_result)

        return final_results

    def _calculate_derived_metrics(self, bot_result: BacktestResult):
        """
        Calculate win rate, profit factor, Sharpe ratio, etc.
        """
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
    
    def _compute_htf_indicators(self, base_indicators_buf: cl.Buffer, num_bars: int) -> cl.Buffer:
        """
        Compute HTF indicators by subsampling base timeframe indicators.
        
        For example, if HTF multiplier is 60 (1h from 1m):
        - HTF bar 0 = base bar 59 (first complete hour)
        - HTF bar 1 = base bar 119 (second complete hour)
        - etc.
        
        Args:
            base_indicators_buf: Base timeframe precomputed indicators
            num_bars: Number of base timeframe bars
        
        Returns:
            HTF indicators buffer (same layout, but fewer bars)
        """
        if not self.enable_mtf:
            return None
        
        # Calculate number of complete HTF bars
        num_htf_bars = num_bars // self.htf_multiplier
        
        if num_htf_bars == 0:
            log_warning(f"Not enough bars for HTF calculation: {num_bars} bars, need {self.htf_multiplier}")
            return None
        
        # Read base timeframe indicators from GPU
        indicators_flat = np.empty(self.NUM_INDICATORS * num_bars, dtype=np.float32)
        cl.enqueue_copy(self.queue, indicators_flat, base_indicators_buf)
        self.queue.finish()
        
        # Reshape to (indicators, bars)
        indicators_array = indicators_flat.reshape((self.NUM_INDICATORS, num_bars))
        
        # Subsample at HTF intervals (take every Nth bar where N = htf_multiplier)
        # HTF bar i corresponds to base bar (i+1)*htf_multiplier - 1
        htf_indices = [(i + 1) * self.htf_multiplier - 1 for i in range(num_htf_bars)]
        htf_indicators = indicators_array[:, htf_indices]
        
        # Create GPU buffer for HTF indicators
        htf_indicator_bytes = self.NUM_INDICATORS * num_htf_bars * 4
        htf_flat = htf_indicators.flatten().astype(np.float32)
        
        htf_indicators_buf = cl.Buffer(
            self.ctx,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=htf_flat
        )
        
        log_info(f"[MTF] Computed {num_htf_bars} HTF bars from {num_bars} base bars (1:{self.htf_multiplier})")
        
        return htf_indicators_buf
    
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
    
    def _serialize_bots(self, bots: List[CompactBotConfig]) -> np.ndarray:
        """Serialize bots to raw bytes matching OpenCL struct."""
        raw_data = np.empty(len(bots) * COMPACT_BOT_SIZE, dtype=np.uint8)
        
        # Define struct layout (132 bytes total) - MUST MATCH OpenCL struct!
        dt = np.dtype([
            ('bot_id', np.int32),
            ('num_indicators', np.uint8),
            ('indicator_indices', np.uint8, 8),
            ('indicator_params', np.float32, (8, 3)),
            ('indicator_risk_strategies', np.uint8, 8),  # Risk strategy per indicator (0-14)
            ('risk_param', np.float32),                  # Global risk parameter
            ('tp_multiplier', np.float32),
            ('sl_multiplier', np.float32),
            ('leverage', np.uint8),
            ('padding', np.uint8, 2)  # 2 bytes padding for 132-byte alignment
        ])
        
        structured = np.zeros(len(bots), dtype=dt)
        
        for i, bot in enumerate(bots):
            structured[i]['bot_id'] = bot.bot_id
            # Ensure num_indicators is consistent with the indicator_indices provided
            num_inds = int(bot.num_indicators)
            if num_inds > 8:
                log_warning(f"Bot {bot.bot_id} num_indicators={num_inds} > 8; truncating to 8")
                num_inds = 8
            structured[i]['num_indicators'] = num_inds
            # Ensure indicator arrays match expected shapes (8, ) and (8,3)
            try:
                inds = np.array(bot.indicator_indices, dtype=np.uint8).flatten()
            except Exception:
                inds = np.zeros(8, dtype=np.uint8)
            if inds.size > 8:
                log_warning(f"Bot {bot.bot_id} has {inds.size} indicator_indices; truncating to 8")
                inds = inds[:8]
            elif inds.size < 8:
                # Pad
                inds = np.concatenate([inds, np.zeros(8 - inds.size, dtype=np.uint8)])
            structured[i]['indicator_indices'] = inds

            try:
                ipar = np.asarray(bot.indicator_params, dtype=np.float32)
                # Flatten rows if necessary
                if ipar.ndim == 1:
                    # e.g., serialized as flat list of 24 floats
                    ipar = ipar.reshape((-1, 3)) if ipar.size >= 3 else np.zeros((0, 3), dtype=np.float32)
            except Exception:
                ipar = np.zeros((0, 3), dtype=np.float32)
            if ipar.shape[0] > 8:
                log_warning(f"Bot {bot.bot_id} has {ipar.shape[0]} indicator_params; truncating to 8")
                ipar = ipar[:8, :]
            elif ipar.shape[0] < 8:
                # pad to 8 rows
                pad_rows = np.zeros((8 - ipar.shape[0], 3), dtype=np.float32)
                ipar = np.vstack([ipar, pad_rows]) if ipar.size else pad_rows
            structured[i]['indicator_params'] = ipar

            try:
                irs = np.array(bot.indicator_risk_strategies, dtype=np.uint8).flatten()
            except Exception:
                irs = np.zeros(8, dtype=np.uint8)
            if irs.size > 8:
                irs = irs[:8]
            elif irs.size < 8:
                irs = np.concatenate([irs, np.zeros(8 - irs.size, dtype=np.uint8)])
            structured[i]['indicator_risk_strategies'] = irs
            structured[i]['risk_param'] = bot.risk_param
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
    
    def _write_trade_logs_csv(self, trade_logs: np.ndarray):
        """Write detailed trade logs to CSV for verification."""
        import os
        import csv
        
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        csv_file = os.path.join(log_dir, "trade_logs.csv")
        
        # Append to CSV so we can collect logs across multiple chunks
        write_header = not os.path.exists(csv_file)
        with open(csv_file, 'a' if not write_header else 'w', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            # Header includes optional Signature column for dedup helpers
            if write_header:
                writer.writerow([
                    'BotID', 'Cycle', 'EntryPrice', 'ExitPrice', 'EntryBar', 'ExitBar',
                    'Leverage', 'PnL', 'Direction', 'ChunkID', 'OutOfCycle', 'Signature'
                ])

            for trade in trade_logs:
                # Compute signature: EntryBar:ExitBar:Direction:PnLCents
                entry = int(trade['entry_bar']) if 'entry_bar' in trade.dtype.names else 0
                exitb = int(trade['exit_bar']) if 'exit_bar' in trade.dtype.names else 0
                direction = 'LONG' if trade['direction'] == 1 else 'SHORT'
                pnl = float(trade['pnl']) if 'pnl' in trade.dtype.names else 0.0
                signature = f"{entry}:{exitb}:{direction}:{int(round(pnl*100))}"
                writer.writerow([
                    trade['bot_id'],
                    trade['cycle'],
                    round(trade['entry_price'], 4),
                    round(trade['exit_price'], 4),
                    trade['entry_bar'],
                    trade['exit_bar'],
                    round(trade['leverage'], 1),
                    round(trade['pnl'], 2),
                    'LONG' if trade['direction'] == 1 else 'SHORT'
                    , int(trade['chunk_id']) if 'chunk_id' in trade.dtype.names else -1,
                    int(trade['out_of_cycle']) if 'out_of_cycle' in trade.dtype.names else 0,
                    signature,
                ])
                # Add an optional signature column for deduplication helpers (EntryBar:ExitBar:Direction:rounded PnL cents)
                with open(csv_file, 'r', encoding='utf-8') as f_re:
                    content = f_re.read()
                # If signature header missing, rewrite file with signature header and value
                # protect against empty file
                if not content.strip():
                    # nothing to rewrite
                    pass
                elif 'Signature' not in content.splitlines()[0]:
                    # Reopen for read and rewrite including signature
                    import tempfile
                    temp = tempfile.NamedTemporaryFile('w', delete=False, newline='', encoding='utf-8')
                    tmp_writer = csv.writer(temp, delimiter=';')
                    with open(csv_file, 'r', encoding='utf-8') as f2:
                        rdr = csv.reader(f2, delimiter=';')
                        header = next(rdr)
                        header.append('Signature')
                        tmp_writer.writerow(header)
                        for row in rdr:
                            # Compute signature from row values (EntryBar, ExitBar, Direction, PnL)
                            try:
                                entry = int(row[4])
                                exitb = int(row[5])
                                direction = row[8]
                                pnl = float(row[7].replace(',', '.'))
                            except Exception:
                                entry = 0; exitb = 0; direction = 'LONG'; pnl = 0.0
                            signature = f"{entry}:{exitb}:{direction}:{int(round(pnl*100))}"
                            row.append(signature)
                            tmp_writer.writerow(row)
                    temp.close()
                    import shutil
                    shutil.move(temp.name, csv_file)
        
        log_info(f"Logged {len(trade_logs)} trades to {csv_file}")
    
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

