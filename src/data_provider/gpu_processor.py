"""
GPU-Accelerated Data Processing
Fast validation, cleaning, and preprocessing of market data using OpenCL.
"""

import numpy as np
import pyopencl as cl
from typing import Optional, Tuple
from pathlib import Path

from ..utils.validation import log_info, log_error, log_debug
from ..utils.config import OHLCV_COLUMNS


class GPUDataProcessor:
    """
    GPU-accelerated data processing for market data validation and cleaning.
    Processes large datasets efficiently using parallel GPU kernels.
    """

    def __init__(self, gpu_context: cl.Context, gpu_queue: cl.CommandQueue):
        """
        Initialize GPU data processor.

        Args:
            gpu_context: OpenCL context
            gpu_queue: OpenCL command queue
        """
        self.ctx = gpu_context
        self.queue = gpu_queue

        # Load and compile the data processing kernel
        kernel_path = Path(__file__).parent.parent / "gpu_kernels" / "data_processing.cl"
        with open(kernel_path, 'r') as f:
            kernel_source = f.read()

        self.program = cl.Program(self.ctx, kernel_source).build()
        log_info("Compiled GPU data processing kernels")

    def validate_and_clean_data(
        self,
        ohlcv_data: np.ndarray,
        min_price: float = 1e-8,
        max_price: float = 1e6,
        min_volume: float = 1e-8,
        max_gap_size: int = 5
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Validate and clean OHLCV data using GPU acceleration.

        Args:
            ohlcv_data: OHLCV array [N, 5] or [N, 6] (timestamp, open, high, low, close, volume)
            min_price: Minimum valid price
            max_price: Maximum valid price
            min_volume: Minimum valid volume
            max_gap_size: Maximum gap size to interpolate

        Returns:
            Tuple of (cleaned_data, valid_flags, stats_dict)
        """
        num_candles = len(ohlcv_data)

        # Ensure data is in float32 format and contiguous
        data = np.ascontiguousarray(ohlcv_data.astype(np.float32))
        if data.shape[1] == 5:  # Add timestamp column if missing
            timestamps = np.arange(num_candles, dtype=np.float32) * 60000  # 1min intervals
            data = np.column_stack([timestamps, data])

        # Create GPU buffers
        data_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=data)
        valid_flags = np.ascontiguousarray(np.zeros(num_candles, dtype=np.int32))
        valid_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, valid_flags.nbytes)
        gap_flags = np.ascontiguousarray(np.zeros(num_candles, dtype=np.int32))
        gap_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, gap_flags.nbytes)

        # Get device max work group size
        max_work_group_size = self.queue.device.max_work_group_size
        work_group_size = min(256, max_work_group_size)
        
        log_debug(f"Using work group size: {work_group_size} (device max: {max_work_group_size})")
        
        # For simplicity, use None to let OpenCL choose optimal work group size
        wg_size = None

        # Step 1: Validate data
        self.program.validate_ohlcv_data(
            self.queue, (num_candles,), wg_size,
            data_buf, valid_buf, np.int32(num_candles),
            np.float32(min_price), np.float32(max_price), np.float32(min_volume)
        )

        # Step 2: Detect gaps
        expected_interval = 60000.0  # 1 minute in milliseconds
        tolerance = 1.5  # Allow 50% deviation

        timestamps_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                 hostbuf=np.ascontiguousarray(data[:, 0]))

        self.program.detect_data_gaps(
            self.queue, (num_candles,), wg_size,
            timestamps_buf, gap_buf, np.int32(num_candles),
            np.float32(expected_interval), np.float32(tolerance)
        )

        # Step 3: Clean data
        self.program.clean_ohlcv_data(
            self.queue, (num_candles,), wg_size,
            data_buf, valid_buf, np.int32(num_candles),
            np.float32(1e-6), np.float32(1e-6)
        )

        # Step 4: Interpolate small gaps
        self.program.interpolate_gaps(
            self.queue, (num_candles,), wg_size,
            data_buf, gap_buf, np.int32(num_candles), np.int32(max_gap_size)
        )

        # Step 5: Calculate statistics
        num_work_groups = (num_candles + work_group_size - 1) // work_group_size
        stats_output = np.ascontiguousarray(np.zeros(num_work_groups * 6, dtype=np.float32))
        stats_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, stats_output.nbytes)

        self.program.calculate_data_stats(
            self.queue, (num_candles,), wg_size,
            data_buf, stats_buf, valid_buf, gap_buf, np.int32(num_candles)
        )

        # Read results back to host
        cl.enqueue_copy(self.queue, valid_flags, valid_buf)
        cl.enqueue_copy(self.queue, gap_flags, gap_buf)
        cl.enqueue_copy(self.queue, data, data_buf)
        cl.enqueue_copy(self.queue, stats_output, stats_buf)

        self.queue.finish()

        # Aggregate statistics
        stats = {
            'mean_return': np.mean(stats_output[::6]),
            'mean_volume': np.mean(stats_output[2::6]),
            'total_gaps': int(np.sum(stats_output[3::6])),
            'total_zeros': int(np.sum(stats_output[4::6])),
            'total_invalids': int(np.sum(stats_output[5::6])),
            'data_quality_score': 1.0 - (np.sum(stats_output[3::6] + stats_output[4::6] + stats_output[5::6]) / num_candles)
        }

        log_info(f"GPU data processing complete: {stats['data_quality_score']:.1%} quality score")
        log_debug(f"Found {stats['total_gaps']} gaps, {stats['total_invalids']} invalid candles")

        return data, valid_flags, stats

    def process_pandas_dataframe(
        self,
        df: 'pd.DataFrame',
        timestamp_col: str = 'timestamp',
        columns: list = None
    ) -> Tuple['pd.DataFrame', dict]:
        """
        Process a pandas DataFrame using GPU acceleration.

        Args:
            df: Input DataFrame
            timestamp_col: Name of timestamp column
            columns: OHLCV column names [open, high, low, close, volume]

        Returns:
            Tuple of (processed_df, stats_dict)
        """
        if columns is None:
            columns = ['open', 'high', 'low', 'close', 'volume']

        # Extract data as numpy array
        data_cols = [timestamp_col] + columns
        data_array = df[data_cols].values.astype(np.float32)

        # Process on GPU
        processed_data, valid_flags, stats = self.validate_and_clean_data(data_array)

        # Create new DataFrame with processed data
        processed_df = df.copy()
        processed_df[timestamp_col] = processed_data[:, 0]
        for i, col in enumerate(columns):
            processed_df[col] = processed_data[:, i + 1]

        # Add validity column
        processed_df['valid'] = valid_flags.astype(bool)

        return processed_df, stats