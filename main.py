#!/usr/bin/env python3
"""
GPU-Accelerated Genetic Algorithm Crypto Trading Bot
Main entry point for the application.

CRITICAL: This application REQUIRES OpenCL-capable GPU.
No CPU fallbacks are provided. Application will crash if GPU unavailable.
"""
import sys
import numpy as np
from typing import Optional
import pyopencl as cl

from src.utils.validation import (
    validate_int, validate_float, validate_pair, validate_timeframe,
    validate_leverage, log_info, log_error, log_warning
)
from src.utils.config import (
    DEFAULT_TRADING_PAIR, DEFAULT_INITIAL_BALANCE, DEFAULT_POPULATION,
    DEFAULT_GENERATIONS, DEFAULT_CYCLES, DEFAULT_BACKTEST_DAYS,
    DEFAULT_MIN_INDICATORS, DEFAULT_MAX_INDICATORS,
    DEFAULT_MIN_RISK_STRATEGIES, DEFAULT_MAX_RISK_STRATEGIES,
    DEFAULT_RANDOM_SEED, IMPLEMENTED_MODES, MODE_DESCRIPTIONS
)
from src.data_provider.fetcher import DataFetcher
from src.data_provider.loader import DataLoader
from src.bot_generator.generator import BotGenerator
from src.backtester.simulator import GPUBacktester
from src.ga.evolver import GeneticAlgorithmEvolver


def initialize_gpu() -> tuple:
    """
    Initialize OpenCL GPU context and command queue.
    
    CRITICAL: This function MUST succeed or the application crashes.
    No CPU fallbacks are allowed per specification.
    
    Returns:
        Tuple of (context, queue, device_info)
        
    Raises:
        RuntimeError: If no OpenCL platforms/devices found or initialization fails
    """
    try:
        # Get all platforms
        platforms = cl.get_platforms()
        if not platforms:
            raise RuntimeError(
                "No OpenCL platforms found. This application requires GPU with OpenCL support.\n"
                "Install GPU drivers: NVIDIA CUDA Toolkit, AMD ROCm, or Intel OpenCL Runtime."
            )
        
        # Find GPU device (prefer dedicated GPU over integrated)
        gpu_device = None
        for platform in platforms:
            try:
                devices = platform.get_devices(device_type=cl.device_type.GPU)
                if devices:
                    gpu_device = devices[0]  # Use first GPU found
                    break
            except cl.RuntimeError:
                continue
        
        if gpu_device is None:
            raise RuntimeError(
                "No OpenCL GPU device found. This application requires GPU.\n"
                "Available device types: " + str([p.get_devices() for p in platforms])
            )
        
        # Create context and queue
        context = cl.Context([gpu_device])
        queue = cl.CommandQueue(context)
        
        # Get device info for logging
        device_info = {
            'name': gpu_device.name,
            'vendor': gpu_device.vendor,
            'version': gpu_device.version,
            'max_compute_units': gpu_device.max_compute_units,
            'max_work_group_size': gpu_device.max_work_group_size,
            'global_mem_size': gpu_device.global_mem_size / (1024**3),  # GB
            'local_mem_size': gpu_device.local_mem_size / 1024,  # KB
        }
        
        log_info("="*60)
        log_info("GPU INITIALIZATION SUCCESSFUL")
        log_info("="*60)
        log_info(f"Device: {device_info['name']}")
        log_info(f"Vendor: {device_info['vendor']}")
        log_info(f"Version: {device_info['version']}")
        log_info(f"Compute Units: {device_info['max_compute_units']}")
        log_info(f"Max Work Group Size: {device_info['max_work_group_size']}")
        log_info(f"Global Memory: {device_info['global_mem_size']:.2f} GB")
        log_info(f"Local Memory: {device_info['local_mem_size']:.2f} KB")
        log_info("="*60 + "\n")
        
        return context, queue, device_info
        
    except Exception as e:
        log_error("="*60)
        log_error("FATAL: GPU INITIALIZATION FAILED")
        log_error("="*60)
        log_error(str(e))
        log_error("\nThis application REQUIRES OpenCL-capable GPU.")
        log_error("No CPU fallbacks available per specification.")
        log_error("="*60)
        raise RuntimeError(f"GPU initialization failed: {e}")


def get_user_input(prompt: str, default: any, validator=None) -> any:
    """
    Get validated user input with default value.
    
    Args:
        prompt: Input prompt to display
        default: Default value if user presses Enter
        validator: Optional validation function
        
    Returns:
        Validated user input or default
    """
    while True:
        user_input = input(f"{prompt} [{default}]: ").strip()
        
        if not user_input:
            return default
        
        if validator:
            try:
                return validator(user_input)
            except ValueError as e:
                log_error(f"Invalid input: {e}")
                continue
        
        return user_input


def get_mode_selection() -> int:
    """
    Get mode selection from user.
    
    Returns:
        Selected mode number
    """
    print("\n" + "="*60)
    print("GPU-ACCELERATED GENETIC ALGORITHM CRYPTO TRADING BOT")
    print("="*60 + "\n")
    
    print("Available Modes:")
    for mode_num, description in MODE_DESCRIPTIONS.items():
        status = "✓" if mode_num in IMPLEMENTED_MODES else "✗"
        print(f"  {status} Mode {mode_num}: {description}")
    
    print()
    
    while True:
        try:
            mode = int(input("Select mode (1-4): ").strip())
            if mode not in MODE_DESCRIPTIONS:
                log_error(f"Invalid mode. Please select 1-4.")
                continue
            if mode not in IMPLEMENTED_MODES:
                log_error(f"Mode {mode} is not implemented yet. Please select Mode 1.")
                continue
            return mode
        except ValueError:
            log_error("Please enter a number.")


def get_mode1_parameters() -> dict:
    """
    Get parameters for Mode 1 (Genetic Algorithm).
    
    Returns:
        Dictionary of parameters
    """
    print("\n" + "-"*60)
    print("MODE 1: GENETIC ALGORITHM CONFIGURATION")
    print("-"*60 + "\n")
    
    params = {}
    
    # Trading pair
    params['pair'] = get_user_input(
        "Trading pair (e.g., BTC/USDT)",
        DEFAULT_TRADING_PAIR,
        lambda x: validate_pair(x)
    )
    
    # Initial balance
    params['initial_balance'] = get_user_input(
        "Initial balance (USDT)",
        DEFAULT_INITIAL_BALANCE,
        lambda x: validate_float(float(x), "initial_balance", strict_positive=True)
    )
    
    # Population size
    params['population'] = get_user_input(
        "Population size (1,000-1,000,000)",
        DEFAULT_POPULATION,
        lambda x: validate_int(int(x), "population", min_val=1000, max_val=1000000)
    )
    
    # Generations
    params['generations'] = get_user_input(
        "Number of generations (1-100)",
        DEFAULT_GENERATIONS,
        lambda x: validate_int(int(x), "generations", min_val=1, max_val=100)
    )
    
    # Cycles
    params['cycles'] = get_user_input(
        "Cycles per generation (1-100)",
        DEFAULT_CYCLES,
        lambda x: validate_int(int(x), "cycles", min_val=1, max_val=100)
    )
    
    # Backtest days
    params['backtest_days'] = get_user_input(
        "Days per backtest cycle (1-365)",
        DEFAULT_BACKTEST_DAYS,
        lambda x: validate_int(int(x), "backtest_days", min_val=1, max_val=365)
    )
    
    # Timeframe
    params['timeframe'] = get_user_input(
        "Timeframe (1m/5m/15m/30m/1h/4h/1d)",
        "1m",
        lambda x: validate_timeframe(x)
    )
    
    # Leverage
    params['leverage'] = get_user_input(
        "Leverage (1-125x)",
        10,
        lambda x: validate_leverage(int(x))
    )
    
    # Indicators per bot
    params['min_indicators'] = get_user_input(
        "Min indicators per bot (1-10)",
        DEFAULT_MIN_INDICATORS,
        lambda x: validate_int(int(x), "min_indicators", min_val=1, max_val=10)
    )
    
    params['max_indicators'] = get_user_input(
        f"Max indicators per bot ({params['min_indicators']}-10)",
        DEFAULT_MAX_INDICATORS,
        lambda x: validate_int(int(x), "max_indicators", 
                              min_val=params['min_indicators'], max_val=10)
    )
    
    # Risk strategies per bot
    params['min_risk_strategies'] = get_user_input(
        "Min risk strategies per bot (1-10)",
        DEFAULT_MIN_RISK_STRATEGIES,
        lambda x: validate_int(int(x), "min_risk_strategies", min_val=1, max_val=10)
    )
    
    params['max_risk_strategies'] = get_user_input(
        f"Max risk strategies per bot ({params['min_risk_strategies']}-10)",
        DEFAULT_MAX_RISK_STRATEGIES,
        lambda x: validate_int(int(x), "max_risk_strategies",
                              min_val=params['min_risk_strategies'], max_val=10)
    )
    
    # Random seed
    use_seed = get_user_input(
        "Use random seed for reproducibility? (y/n)",
        "y",
        lambda x: x.lower() in ['y', 'n']
    )
    
    if use_seed.lower() == 'y':
        params['random_seed'] = get_user_input(
            "Random seed",
            DEFAULT_RANDOM_SEED,
            lambda x: validate_int(int(x), "random_seed", min_val=0)
        )
    else:
        params['random_seed'] = None
    
    return params


def run_mode1(params: dict, gpu_context, gpu_queue, gpu_info: dict) -> None:
    """
    Run Mode 1: Genetic Algorithm Evolution.
    
    Args:
        params: Parameter dictionary
        gpu_context: PyOpenCL context
        gpu_queue: PyOpenCL command queue
        gpu_info: GPU device information dict
    """
    print("\n" + "="*60)
    print("STARTING MODE 1: GENETIC ALGORITHM")
    print("="*60 + "\n")
    
    try:
        # Step 1: Fetch data
        log_info("Step 1/5: Fetching market data...")
        fetcher = DataFetcher()
        
        # Calculate required days
        total_days = fetcher.calculate_required_days(
            params['backtest_days'],
            params['cycles']
        )
        
        log_info(f"Fetching {total_days} days of data for {params['pair']} {params['timeframe']}")
        
        file_paths = fetcher.fetch_data_range(
            pair=params['pair'],
            timeframe=params['timeframe'],
            total_days=total_days
        )
        
        # Step 2: Load and validate data
        log_info("\nStep 2/5: Loading and validating data...")
        loader = DataLoader(
            file_paths=file_paths,
            timeframe=params['timeframe'],
            random_seed=params['random_seed']
        )
        
        ohlcv_data = loader.load_all_data()
        
        # Generate cycle ranges
        cycle_ranges = loader.generate_cycle_ranges(
            num_cycles=params['cycles'],
            backtest_days=params['backtest_days']
        )
        
        data_summary = loader.get_data_summary()
        log_info(f"Data loaded: {data_summary}")
        
        # Step 3: Initialize components with GPU context
        log_info("\nStep 3/5: Initializing GA components with GPU...")
        
        bot_generator = BotGenerator(
            population_size=params['population'],
            min_indicators=params['min_indicators'],
            max_indicators=params['max_indicators'],
            min_risk_strategies=params['min_risk_strategies'],
            max_risk_strategies=params['max_risk_strategies'],
            leverage=params['leverage'],
            random_seed=params['random_seed'],
            gpu_context=gpu_context,
            gpu_queue=gpu_queue
        )
        
        backtester = GPUBacktester(
            gpu_context=gpu_context,
            gpu_queue=gpu_queue,
            initial_balance=params['initial_balance']
        )
        
        evolver = GeneticAlgorithmEvolver(
            bot_generator=bot_generator,
            backtester=backtester
        )
        
        # Step 4: Run evolution
        log_info("\nStep 4/5: Running genetic algorithm evolution...")
        
        evolver.run_evolution(
            num_generations=params['generations'],
            ohlcv_data=ohlcv_data,
            cycle_ranges=cycle_ranges
        )
        
        # Step 5: Save and display results
        log_info("\nStep 5/5: Saving results...")
        
        evolver.save_results()
        evolver.print_top_bots(count=10)
        
        print("\n" + "="*60)
        print("MODE 1 COMPLETE")
        print("="*60 + "\n")
        
        log_info(f"Results saved to {evolver.backtester}")
        log_info("Thank you for using the GPU Trading Bot!")
        
    except KeyboardInterrupt:
        log_warning("\n\nExecution interrupted by user")
        sys.exit(1)
    except Exception as e:
        log_error(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def get_mode4_parameters() -> dict:
    """
    Get parameters for Mode 4: Single Bot Detailed Backtest.
    
    Returns:
        Dictionary of parameters
    """
    params = {}
    
    print("\nMode 4: Single Bot Detailed Backtest")
    print("This mode backtests a single bot and shows detailed trade log.\n")
    
    # Date range
    import datetime
    today = datetime.date.today()
    default_end = today.strftime("%Y-%m-%d")
    default_start = (today - datetime.timedelta(days=90)).strftime("%Y-%m-%d")
    
    params['start_date'] = get_user_input(
        "Start date (YYYY-MM-DD)",
        default_start
    )
    
    params['end_date'] = get_user_input(
        "End date (YYYY-MM-DD)",
        default_end
    )
    
    # Trading pair and timeframe
    params['pair'] = get_user_input(
        "Trading pair",
        DEFAULT_TRADING_PAIR,
        validate_pair
    )
    
    params['timeframe'] = get_user_input(
        "Timeframe (1m/5m/15m/1h/4h/1d)",
        "1h",
        validate_timeframe
    )
    
    # Bot source
    bot_choice = get_user_input(
        "Load existing bot ID or generate new? (id/new)",
        "new"
    )
    
    if bot_choice.lower() == 'id':
        params['bot_id'] = get_user_input(
            "Bot ID to load",
            1,
            lambda x: validate_int(int(x), "bot_id", min_val=1)
        )
    else:
        params['bot_id'] = None  # Generate new random bot
    
    # Initial balance
    params['initial_balance'] = get_user_input(
        "Initial balance",
        DEFAULT_INITIAL_BALANCE,
        lambda x: validate_float(float(x), "initial_balance", min_val=100.0)
    )
    
    # Leverage
    params['leverage'] = get_user_input(
        "Leverage (1-125)",
        10,
        lambda x: validate_leverage(int(x))
    )
    
    return params


def run_mode4(params: dict, gpu_context, gpu_queue, gpu_info: dict) -> None:
    """
    Run Mode 4: Single Bot Detailed Backtest.
    
    Args:
        params: Parameter dictionary
        gpu_context: PyOpenCL context
        gpu_queue: PyOpenCL command queue
        gpu_info: GPU device information dict
    """
    import pandas as pd
    from datetime import datetime
    
    print("\n" + "="*60)
    print("STARTING MODE 4: SINGLE BOT BACKTEST")
    print("="*60 + "\n")
    
    try:
        # Step 1: Fetch data for date range
        log_info("Step 1/4: Fetching market data...")
        fetcher = DataFetcher()
        
        # Calculate days between dates
        start = datetime.strptime(params['start_date'], "%Y-%m-%d")
        end = datetime.strptime(params['end_date'], "%Y-%m-%d")
        total_days = (end - start).days + 1
        
        log_info(f"Fetching {total_days} days of data for {params['pair']} {params['timeframe']}")
        
        file_paths = fetcher.fetch_data_range(
            pair=params['pair'],
            timeframe=params['timeframe'],
            total_days=total_days,
            end_date=params['end_date']
        )
        
        # Step 2: Load data
        log_info("\nStep 2/4: Loading data...")
        loader = DataLoader(
            file_paths=file_paths,
            timeframe=params['timeframe'],
            random_seed=42
        )
        
        ohlcv_data = loader.load_all_data()
        
        # Single cycle covering entire range
        cycle_ranges = [(0, len(ohlcv_data) - 1)]
        
        log_info(f"Loaded {len(ohlcv_data)} bars from {params['start_date']} to {params['end_date']}")
        
        # Step 3: Get or generate bot
        log_info("\nStep 3/4: Preparing bot...")
        
        if params['bot_id'] is not None:
            # TODO: Load bot from saved results
            log_warning("Bot loading not yet implemented, generating random bot instead")
            params['bot_id'] = None
        
        if params['bot_id'] is None:
            # Generate single random bot
            bot_generator = BotGenerator(
                population_size=1,
                min_indicators=3,
                max_indicators=8,
                min_risk_strategies=2,
                max_risk_strategies=5,
                leverage=params['leverage'],
                random_seed=42,
                gpu_context=gpu_context,
                gpu_queue=gpu_queue
            )
            
            bots = bot_generator.generate_population()
            bot = bots[0]
            log_info(f"Generated random bot with ID {bot.bot_id}")
        
        # Step 4: Run backtest
        log_info("\nStep 4/4: Running backtest...")
        
        from src.backtester.simulator import GPUBacktester
        backtester = GPUBacktester(
            gpu_context=gpu_context,
            gpu_queue=gpu_queue,
            initial_balance=params['initial_balance']
        )
        
        results = backtester.backtest_bots(
            bots=[bot],
            ohlcv_data=ohlcv_data,
            cycle_starts=np.array([cycle_ranges[0][0]], dtype=np.int32),
            cycle_ends=np.array([cycle_ranges[0][1]], dtype=np.int32)
        )
        
        result = results[0]
        
        # Display results
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        print(f"Bot ID: {result.bot_id}")
        print(f"Total Trades: {result.total_trades}")
        print(f"Winning Trades: {result.winning_trades}")
        print(f"Losing Trades: {result.losing_trades}")
        print(f"Win Rate: {result.win_rate:.2f}%")
        print(f"Total Return: {result.total_return_pct:.2f}%")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.3f}")
        print(f"Max Drawdown: {result.max_drawdown_pct:.2f}%")
        print(f"Final Balance: ${result.final_balance:.2f}")
        print("="*60 + "\n")
        
        # Display bot configuration
        print("BOT CONFIGURATION:")
        print(f"  Indicators ({len(bot.indicators)}):")
        for i, ind in enumerate(bot.indicators, 1):
            print(f"    {i}. {ind.indicator_type.value}: {ind.params}")
        
        print(f"\n  Risk Strategies ({len(bot.risk_strategies)}):")
        for i, risk in enumerate(bot.risk_strategies, 1):
            print(f"    {i}. {risk.strategy_type.value}: {risk.params}")
        
        print(f"\n  Take Profit: {bot.take_profit_pct:.2f}%")
        print(f"  Stop Loss: {bot.stop_loss_pct:.2f}%")
        print(f"  Leverage: {bot.leverage}x")
        
        print("\n" + "="*60)
        print("MODE 4 COMPLETE")
        print("="*60 + "\n")
        
        log_info("Note: Detailed trade log not yet implemented in GPU kernel")
        
    except KeyboardInterrupt:
        log_warning("\n\nExecution interrupted by user")
        sys.exit(1)
    except Exception as e:
        log_error(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point."""
    try:
        # CRITICAL: Initialize GPU first - crashes if unavailable
        log_info("Initializing GPU (OpenCL)...\n")
        gpu_context, gpu_queue, gpu_info = initialize_gpu()
        
        # Get mode selection
        mode = get_mode_selection()
        
        if mode == 1:
            # Mode 1: Genetic Algorithm
            params = get_mode1_parameters()
            run_mode1(params, gpu_context, gpu_queue, gpu_info)
        elif mode == 4:
            # Mode 4: Single Bot Backtest
            params = get_mode4_parameters()
            run_mode4(params, gpu_context, gpu_queue, gpu_info)
        else:
            # Other modes not implemented
            log_error(f"Mode {mode} is not implemented yet.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        log_warning("\n\nExecution cancelled by user")
        sys.exit(0)
    except Exception as e:
        log_error(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
