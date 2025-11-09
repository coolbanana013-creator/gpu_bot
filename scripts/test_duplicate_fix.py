"""Quick test to verify duplicate fix - runs mode 1 with minimal settings."""
import sys
import os
import pyopencl as cl

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.ga.evolver_compact import GeneticAlgorithmEvolver
from src.data_provider.fetcher import DataFetcher
from src.data_provider.loader import DataLoader
from src.utils.validation import validate_pair, log_info, log_error
import numpy as np
from datetime import datetime, timedelta

# Minimal test settings
PAIR = "BTC/USDT:USDT"
NUM_BOTS = 1000  # Smaller for quick test
NUM_GENERATIONS = 2  # Just 2 generations
NUM_CYCLES = 1  # Single cycle
DAYS_PER_CYCLE = 30

def main():
    """Run quick duplicate check test."""
    log_info("="*60)
    log_info("DUPLICATE FIX TEST - Mode 1 (Minimal Settings)")
    log_info("="*60)
    
    # Initialize GPU
    log_info("Initializing GPU...")
    platforms = cl.get_platforms()
    gpu_device = platforms[0].get_devices(device_type=cl.device_type.GPU)[0]
    ctx = cl.Context([gpu_device])
    queue = cl.CommandQueue(ctx)
    log_info(f"Using device: {gpu_device.name}")
    
    # Validate and normalize pair
    normalized_pair = validate_pair(PAIR)
    log_info(f"Using trading pair: {normalized_pair}")
    
    # Calculate total days and fetch data
    fetcher = DataFetcher()
    total_days = fetcher.calculate_required_days(DAYS_PER_CYCLE, NUM_CYCLES)
    
    log_info(f"Fetching {total_days} days of data ({NUM_CYCLES} cycles x {DAYS_PER_CYCLE} days)")
    
    # Fetch data files
    file_paths = fetcher.fetch_data_range(
        pair=normalized_pair,
        timeframe='1h',
        total_days=total_days
    )
    
    # Load data
    log_info("Loading data...")
    loader = DataLoader(
        file_paths=file_paths,
        timeframe='1h',
        random_seed=42,
        gpu_context=ctx,
        gpu_queue=queue,
        use_gpu_processing=True
    )
    
    ohlcv_data = loader.load_and_combine()
    
    if ohlcv_data is None or len(ohlcv_data) == 0:
        log_error("Failed to load data")
        return
    
    log_info(f"Loaded {len(ohlcv_data)} candles")
    
    # Define cycles
    candles_per_cycle = DAYS_PER_CYCLE * 24
    cycles = []
    for i in range(NUM_CYCLES):
        start_idx = i * candles_per_cycle
        end_idx = min((i + 1) * candles_per_cycle, len(ohlcv_data))
        cycles.append((start_idx, end_idx))
        log_info(f"  Cycle {i}: candles {start_idx} to {end_idx} ({end_idx - start_idx} candles)")
    
    # Initialize evolver
    log_info(f"\nInitializing evolver with {NUM_BOTS} bots...")
    evolver = GeneticAlgorithmEvolver(
        population_size=NUM_BOTS,
        min_indicators=1,
        max_indicators=5,
        min_risk_strategies=1,
        max_risk_strategies=3,
        min_leverage=1,
        max_leverage=5,
        mutation_rate=0.1,
        elite_pct=0.1,
        random_seed=42,
        use_gpu=True,
        ctx=ctx,
        queue=queue
    )
    
    # Run evolution
    log_info(f"\nRunning evolution: {NUM_GENERATIONS} generations")
    evolver.run_evolution(
        num_generations=NUM_GENERATIONS,
        ohlcv_data=ohlcv_data,
        cycles=cycles,
        initial_balance=100.0
    )
    
    log_info("\nTest complete! Check logs/generation_1.csv for duplicates")

if __name__ == "__main__":
    main()
