"""
Data loader for market data with validation and cycle generation.
Loads Parquet files, validates data integrity, generates non-overlapping cycle ranges.
GPU-ACCELERATED: Uses GPU for fast data validation and cleaning.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime, timedelta
import pyopencl as cl

from ..utils.validation import (
    validate_int, validate_float, log_info, log_error, log_warning, log_debug
)
from ..utils.config import (
    OHLCV_COLUMNS, TIMEFRAME_TO_MS, DATA_GAP_TOLERANCE,
    MIN_DATA_POINTS_PER_DAY, MAX_INDICATOR_LOOKBACK
)
from .gpu_processor import GPUDataProcessor


class DataLoader:
    """
    Loads and validates market data, generates non-overlapping cycle ranges.
    GPU-ACCELERATED: Optional GPU processing for fast validation and cleaning.
    """
    
    def __init__(
        self,
        file_paths: List[Path],
        timeframe: str,
        random_seed: Optional[int] = None,
        gpu_context: Optional[cl.Context] = None,
        gpu_queue: Optional[cl.CommandQueue] = None,
        use_gpu_processing: bool = True
    ):
        """
        Initialize DataLoader.
        
        Args:
            file_paths: List of Parquet file paths to load
            timeframe: Timeframe string (e.g., "1m")
            random_seed: Random seed for reproducibility
            gpu_context: Optional GPU context for accelerated processing
            gpu_queue: Optional GPU queue for accelerated processing
            use_gpu_processing: Whether to use GPU for data validation/cleaning
        """
        self.file_paths = sorted(file_paths)
        self.timeframe = timeframe
        self.timeframe_ms = TIMEFRAME_TO_MS[timeframe]
        self.random_seed = random_seed
        self.use_gpu_processing = use_gpu_processing
        
        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)
            log_info(f"Set random seed to {random_seed}")
        
        self.data: Optional[pd.DataFrame] = None
        self.cycle_ranges: List[Tuple[int, int]] = []
        
        # Initialize GPU processor if requested
        self.gpu_processor = None
        if use_gpu_processing and gpu_context is not None and gpu_queue is not None:
            try:
                self.gpu_processor = GPUDataProcessor(gpu_context, gpu_queue)
                log_info("Initialized GPU-accelerated data processing")
            except Exception as e:
                log_warning(f"Failed to initialize GPU data processor: {e}")
                log_warning("Falling back to CPU processing")
                self.gpu_processor = None
    
    def load_all_data(self) -> pd.DataFrame:
        """
        Load all data from file paths into a single DataFrame.
        
        Returns:
            Combined DataFrame with all OHLCV data
            
        Raises:
            RuntimeError: If data loading or validation fails
        """
        if not self.file_paths:
            raise RuntimeError("No file paths provided")
        
        log_info(f"Loading {len(self.file_paths)} data files")
        
        dataframes = []
        
        for file_path in self.file_paths:
            try:
                df = pd.read_parquet(file_path)
                
                # Validate columns
                if not all(col in df.columns for col in OHLCV_COLUMNS):
                    raise RuntimeError(
                        f"File {file_path.name} missing required columns. "
                        f"Expected: {OHLCV_COLUMNS}, Got: {list(df.columns)}"
                    )
                
                dataframes.append(df[OHLCV_COLUMNS])
                log_debug(f"Loaded {len(df)} rows from {file_path.name}")
                
            except Exception as e:
                log_error(f"Failed to load {file_path.name}: {e}")
                raise RuntimeError(f"Cannot load data file {file_path.name}: {e}")
        
        # Combine all dataframes
        self.data = pd.concat(dataframes, ignore_index=True)
        
        # Sort by timestamp
        self.data.sort_values('timestamp', inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        
        log_info(f"Loaded total of {len(self.data)} candles")
        
        # Validate and clean data (GPU-accelerated if available)
        if self.gpu_processor is not None:
            self._validate_and_clean_data_gpu()
        else:
            self._validate_data()
        
        return self.data
    
    def _validate_data(self) -> None:
        """
        Validate data integrity: check for gaps, duplicates, invalid values.
        
        Raises:
            RuntimeError: If validation fails
        """
        if self.data is None or self.data.empty:
            raise RuntimeError("No data to validate")
        
        log_info("Validating data integrity")
        
        # Check for NaN values
        nan_counts = self.data[OHLCV_COLUMNS].isna().sum()
        if nan_counts.any():
            log_error(f"Found NaN values: {nan_counts[nan_counts > 0].to_dict()}")
            raise RuntimeError("Data contains NaN values")
        
        # Check for duplicate timestamps
        duplicates = self.data['timestamp'].duplicated().sum()
        if duplicates > 0:
            log_warning(f"Found {duplicates} duplicate timestamps, removing...")
            self.data.drop_duplicates(subset='timestamp', keep='first', inplace=True)
            self.data.reset_index(drop=True, inplace=True)
        
        # Check for gaps in data
        timestamps = self.data['timestamp'].values
        time_diffs = np.diff(timestamps)
        
        expected_diff = self.timeframe_ms
        max_allowed_diff = expected_diff * DATA_GAP_TOLERANCE
        
        gaps = np.where(time_diffs > max_allowed_diff)[0]
        
        if len(gaps) > 0:
            gap_count = len(gaps)
            gap_pct = (gap_count / len(time_diffs)) * 100
            
            log_warning(
                f"Found {gap_count} gaps in data ({gap_pct:.2f}% of intervals). "
                f"Max gap: {time_diffs[gaps].max() / expected_diff:.1f}x expected interval"
            )
            
            # Allow some gaps but not too many
            if gap_pct > 5.0:
                raise RuntimeError(
                    f"Too many gaps in data ({gap_pct:.2f}%). Data quality insufficient."
                )
        
        # Validate OHLC relationships
        invalid_ohlc = (
            (self.data['high'] < self.data['low']) |
            (self.data['high'] < self.data['open']) |
            (self.data['high'] < self.data['close']) |
            (self.data['low'] > self.data['open']) |
            (self.data['low'] > self.data['close'])
        )
        
        if invalid_ohlc.any():
            invalid_count = invalid_ohlc.sum()
            log_error(f"Found {invalid_count} rows with invalid OHLC relationships")
            raise RuntimeError(f"Data contains {invalid_count} invalid OHLC rows")
        
        # Check for zero or negative prices
        price_cols = ['open', 'high', 'low', 'close']
        invalid_prices = (self.data[price_cols] <= 0).any(axis=1)
        
        if invalid_prices.any():
            invalid_count = invalid_prices.sum()
            log_error(f"Found {invalid_count} rows with zero or negative prices")
            raise RuntimeError(f"Data contains {invalid_count} invalid price rows")
        
        log_info("Data validation passed")
    
    def _validate_and_clean_data_gpu(self) -> None:
        """
        GPU-accelerated data validation and cleaning.
        Much faster than CPU validation for large datasets.
        """
        log_info("Performing GPU-accelerated data validation and cleaning")
        
        try:
            # Convert DataFrame to numpy array for GPU processing
            data_array = self.data[OHLCV_COLUMNS].values
            
            # Process data on GPU
            processed_data, valid_flags, stats = self.gpu_processor.validate_and_clean_data(
                data_array,
                min_price=1e-8,    # Minimum valid price
                max_price=1e6,     # Maximum valid price  
                min_volume=1e-8,   # Minimum valid volume
                max_gap_size=5     # Interpolate gaps of 5 candles or less
            )
            
            # Update DataFrame with processed data
            self.data[OHLCV_COLUMNS] = processed_data[:, 1:]  # Skip timestamp column
            self.data['valid'] = valid_flags.astype(bool)
            
            # Log statistics
            log_info(f"GPU processing complete:")
            log_info(f"  Data quality score: {stats['data_quality_score']:.1%}")
            log_info(f"  Mean return: {stats['mean_return']:.4f}")
            log_info(f"  Mean volume: {stats['mean_volume']:.2f}")
            log_info(f"  Gaps found: {stats['total_gaps']}")
            log_info(f"  Invalid candles: {stats['total_invalids']}")
            
            # Check if data quality is acceptable
            if stats['data_quality_score'] < 0.95:
                log_warning(f"Data quality score {stats['data_quality_score']:.1%} is below 95% threshold")
            
            # Remove invalid candles if any
            invalid_count = (valid_flags == 0).sum()
            if invalid_count > 0:
                log_warning(f"Removing {invalid_count} invalid candles")
                self.data = self.data[valid_flags == 1].reset_index(drop=True)
            
            log_info("GPU data validation and cleaning completed successfully")
            
        except Exception as e:
            log_error(f"GPU data processing failed: {e}")
            log_warning("Falling back to CPU validation")
            self._validate_data()
    
    def generate_cycle_ranges(
        self,
        num_cycles: int,
        backtest_days: int,
        lookback_buffer: int = MAX_INDICATOR_LOOKBACK
    ) -> List[Tuple[int, int]]:
        """
        Generate non-overlapping random ranges for backtest cycles.
        
        Args:
            num_cycles: Number of cycles to generate
            backtest_days: Number of days per cycle
            lookback_buffer: Additional bars needed for indicator lookback
            
        Returns:
            List of (start_idx, end_idx) tuples for each cycle
            
        Raises:
            ValueError: If not enough data for requested cycles
        """
        if self.data is None:
            raise RuntimeError("Must load data before generating cycle ranges")
        
        num_cycles = validate_int(num_cycles, "num_cycles", min_val=1)
        backtest_days = validate_int(backtest_days, "backtest_days", min_val=1)
        lookback_buffer = validate_int(lookback_buffer, "lookback_buffer", min_val=0)
        
        log_info(f"Generating {num_cycles} non-overlapping cycle ranges ({backtest_days} days each)")
        
        # Calculate bars per day for this timeframe
        bars_per_day = self._estimate_bars_per_day()
        
        # Calculate bars per cycle
        bars_per_cycle = int(backtest_days * bars_per_day)
        total_bars_needed = bars_per_cycle * num_cycles + lookback_buffer
        
        available_bars = len(self.data)
        
        if total_bars_needed > available_bars:
            raise ValueError(
                f"Not enough data for {num_cycles} cycles of {backtest_days} days. "
                f"Need {total_bars_needed} bars, have {available_bars} bars. "
                f"Reduce cycles or backtest days, or fetch more data."
            )
        
        # Create list of possible starting positions (with lookback buffer)
        # Each cycle needs: lookback_buffer + bars_per_cycle
        cycle_total_bars = lookback_buffer + bars_per_cycle
        
        # Generate non-overlapping ranges by placing them sequentially
        self.cycle_ranges = []
        
        max_start_idx = available_bars - cycle_total_bars
        if max_start_idx < 0:
            raise ValueError(f"Data too short for even one cycle. Need at least {cycle_total_bars} bars, have {available_bars}")
        
        # Place cycles sequentially with no gaps between them
        current_start = 0
        
        for i in range(num_cycles):
            if current_start > max_start_idx:
                break
                
            # Start of actual backtest data (after lookback)
            backtest_start = current_start + lookback_buffer
            backtest_end = backtest_start + bars_per_cycle
            
            # Ensure we don't go beyond available data
            if backtest_end >= available_bars:
                break
            
            self.cycle_ranges.append((backtest_start, backtest_end))
            
            log_debug(
                f"Cycle {len(self.cycle_ranges)}: "
                f"indices [{backtest_start}:{backtest_end}] "
                f"(lookback from {current_start})"
            )
            
            # Move to next non-overlapping position
            current_start += bars_per_cycle
        
        if len(self.cycle_ranges) < num_cycles:
            raise ValueError(
                f"Could only generate {len(self.cycle_ranges)} non-overlapping cycles, "
                f"requested {num_cycles}. This suggests data gaps are preventing proper cycle placement. "
                f"Try reducing the number of cycles or backtest days per cycle."
            )
        
        log_info(f"Successfully generated {len(self.cycle_ranges)} non-overlapping cycle ranges")
        return self.cycle_ranges
    
    def _estimate_bars_per_day(self) -> float:
        """
        Estimate average bars per day based on loaded data.
        
        Returns:
            Average bars per day
        """
        if self.data is None or len(self.data) < 2:
            # Fallback to theoretical calculation
            bars_per_day = (24 * 60 * 60 * 1000) / self.timeframe_ms
            log_warning(f"Using theoretical bars per day: {bars_per_day}")
            return bars_per_day
        
        # Calculate from actual data
        first_ts = self.data['timestamp'].iloc[0]
        last_ts = self.data['timestamp'].iloc[-1]
        
        total_days = (last_ts - first_ts) / (24 * 60 * 60 * 1000)
        
        if total_days > 0:
            bars_per_day = len(self.data) / total_days
            log_debug(f"Estimated bars per day from data: {bars_per_day:.2f}")
            return bars_per_day
        else:
            bars_per_day = (24 * 60 * 60 * 1000) / self.timeframe_ms
            return bars_per_day
    
    def get_cycle_data(
        self,
        cycle_idx: int,
        include_lookback: bool = True,
        lookback_bars: int = MAX_INDICATOR_LOOKBACK
    ) -> pd.DataFrame:
        """
        Get data for a specific cycle.
        
        Args:
            cycle_idx: Index of the cycle (0-based)
            include_lookback: Whether to include lookback period
            lookback_bars: Number of lookback bars to include
            
        Returns:
            DataFrame for the cycle
        """
        if not self.cycle_ranges:
            raise RuntimeError("Must generate cycle ranges first")
        
        if cycle_idx < 0 or cycle_idx >= len(self.cycle_ranges):
            raise ValueError(f"Invalid cycle index {cycle_idx}, must be 0-{len(self.cycle_ranges)-1}")
        
        start_idx, end_idx = self.cycle_ranges[cycle_idx]
        
        if include_lookback:
            # Include lookback period
            actual_start = max(0, start_idx - lookback_bars)
        else:
            actual_start = start_idx
        
        cycle_data = self.data.iloc[actual_start:end_idx].copy()
        cycle_data.reset_index(drop=True, inplace=True)
        
        log_debug(
            f"Retrieved cycle {cycle_idx} data: "
            f"{len(cycle_data)} bars (indices {actual_start}:{end_idx})"
        )
        
        return cycle_data
    
    def get_all_data(self) -> pd.DataFrame:
        """
        Get all loaded data.
        
        Returns:
            Complete DataFrame
        """
        if self.data is None:
            raise RuntimeError("No data loaded")
        return self.data.copy()
    
    def get_data_summary(self) -> dict:
        """
        Get summary statistics about loaded data.
        
        Returns:
            Dictionary with data summary
        """
        if self.data is None:
            return {"status": "No data loaded"}
        
        first_ts = pd.to_datetime(self.data['timestamp'].iloc[0], unit='ms')
        last_ts = pd.to_datetime(self.data['timestamp'].iloc[-1], unit='ms')
        
        return {
            "total_bars": len(self.data),
            "timeframe": self.timeframe,
            "start_date": first_ts.strftime('%Y-%m-%d %H:%M:%S'),
            "end_date": last_ts.strftime('%Y-%m-%d %H:%M:%S'),
            "days_covered": (last_ts - first_ts).days,
            "num_cycles": len(self.cycle_ranges),
            "files_loaded": len(self.file_paths)
        }
