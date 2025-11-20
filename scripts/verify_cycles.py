#!/usr/bin/env python3
"""
Script to verify cycle lengths and profit calculations.
"""

import sys
import os
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_provider.loader import DataLoader
from src.utils.validation import log_info, log_error
from src.utils.config import DEFAULT_TRADING_PAIR, DEFAULT_BACKTEST_DAYS, DEFAULT_CYCLES

def main():
    # Get file paths
    data_dir = Path('data') / 'BTC_USDT' / '1m'
    file_paths = list(data_dir.glob('*.parquet'))
    file_paths.sort()

    if not file_paths:
        log_error("No data files found")
        return

    log_info(f"Found {len(file_paths)} data files")

    # Load data
    loader = DataLoader(
        file_paths=file_paths,
        timeframe='1m'
    )

    try:
        loader.load_all_data()
        log_info(f"Loaded {len(loader.data)} bars of data")
    except Exception as e:
        log_error(f"Failed to load data: {e}")
        return

    # Generate cycle ranges
    num_cycles = 20  # As per user requirement
    backtest_days = 7  # As per user requirement

    try:
        cycle_ranges = loader.generate_cycle_ranges(
            num_cycles=num_cycles,
            backtest_days=backtest_days
        )
        log_info(f"Generated {len(cycle_ranges)} cycles")
    except Exception as e:
        log_error(f"Failed to generate cycles: {e}")
        return

    # Check each cycle length
    bars_per_day = loader._estimate_bars_per_day()
    log_info(f"Estimated bars per day: {bars_per_day:.2f}")

    for i, (start_idx, end_idx) in enumerate(cycle_ranges):
        cycle_length = end_idx - start_idx
        expected_length = int(backtest_days * bars_per_day)

        # Get timestamps
        start_ts = loader.data.iloc[start_idx]['timestamp']
        end_ts = loader.data.iloc[end_idx-1]['timestamp']  # end_idx is exclusive

        start_dt = datetime.fromtimestamp(start_ts / 1000)
        end_dt = datetime.fromtimestamp(end_ts / 1000)

        actual_days = (end_ts - start_ts) / (24 * 60 * 60 * 1000)

        log_info(f"Cycle {i}: indices [{start_idx}:{end_idx}] ({cycle_length} bars)")
        log_info(f"  Time: {start_dt} to {end_dt} ({actual_days:.2f} days)")
        log_info(f"  Expected bars: {expected_length}, actual: {cycle_length}")

        if abs(cycle_length - expected_length) > 1:
            log_error(f"  Cycle {i} length mismatch!")

if __name__ == "__main__":
    main()