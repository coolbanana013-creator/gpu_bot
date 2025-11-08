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
from src.bot_generator.compact_generator import CompactBotGenerator
from src.backtester.compact_simulator import CompactBacktester
from src.ga.evolver_compact import GeneticAlgorithmEvolver  # NEW: Use compact evolver


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
        status = "[OK]" if mode_num in IMPLEMENTED_MODES else "[X]"
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
    
    # Leverage range (NEW: min and max)
    params['min_leverage'] = get_user_input(
        "Min leverage (1-125x)",
        1,
        lambda x: validate_int(int(x), "min_leverage", min_val=1, max_val=125)
    )
    
    params['max_leverage'] = get_user_input(
        f"Max leverage ({params['min_leverage']}-125x)",
        10,
        lambda x: validate_int(int(x), "max_leverage", 
                              min_val=params['min_leverage'], max_val=125)
    )
    
    # Indicators per bot
    params['min_indicators'] = get_user_input(
        "Min indicators per bot (1-8)",
        DEFAULT_MIN_INDICATORS,
        lambda x: validate_int(int(x), "min_indicators", min_val=1, max_val=8)
    )
    
    params['max_indicators'] = get_user_input(
        f"Max indicators per bot ({params['min_indicators']}-8)",
        DEFAULT_MAX_INDICATORS,
        lambda x: validate_int(int(x), "max_indicators", 
                              min_val=params['min_indicators'], max_val=8)
    )
    
    # Risk strategies per bot
    params['min_risk_strategies'] = get_user_input(
        "Min risk strategies per bot (1-15)",
        DEFAULT_MIN_RISK_STRATEGIES,
        lambda x: validate_int(int(x), "min_risk_strategies", min_val=1, max_val=15)
    )
    
    params['max_risk_strategies'] = get_user_input(
        f"Max risk strategies per bot ({params['min_risk_strategies']}-15)",
        DEFAULT_MAX_RISK_STRATEGIES,
        lambda x: validate_int(int(x), "max_risk_strategies",
                              min_val=params['min_risk_strategies'], max_val=15)
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
    print("GENETIC ALGORITHM - STARTING")
    print("="*60)
    
    try:
        # Fetch data
        fetcher = DataFetcher()
        total_days = fetcher.calculate_required_days(
            params['backtest_days'],
            params['cycles']
        )
        
        print(f"\nFetching {total_days} days of {params['pair']} {params['timeframe']} data...")
        
        file_paths = fetcher.fetch_data_range(
            pair=params['pair'],
            timeframe=params['timeframe'],
            total_days=total_days
        )
        
        # Load and validate data
        print("Loading data...")
        loader = DataLoader(
            file_paths=file_paths,
            timeframe=params['timeframe'],
            random_seed=params['random_seed']
        )
        
        ohlcv_data = loader.load_all_data()
        ohlcv_array = ohlcv_data[['timestamp', 'open', 'high', 'low', 'close', 'volume']].values.astype(np.float32)
        
        cycle_ranges = loader.generate_cycle_ranges(
            num_cycles=params['cycles'],
            backtest_days=params['backtest_days']
        )
        
        print(f"Loaded {len(ohlcv_array)} bars, {len(cycle_ranges)} cycles\n")
        
        # Initialize components
        bot_generator = CompactBotGenerator(
            population_size=params['population'],
            min_indicators=params['min_indicators'],
            max_indicators=params['max_indicators'],
            min_risk_strategies=params['min_risk_strategies'],
            max_risk_strategies=params['max_risk_strategies'],
            min_leverage=params['min_leverage'],
            max_leverage=params['max_leverage'],
            random_seed=params['random_seed'],
            gpu_context=gpu_context,
            gpu_queue=gpu_queue
        )
        
        backtester = CompactBacktester(
            gpu_context=gpu_context,
            gpu_queue=gpu_queue,
            initial_balance=params['initial_balance']
        )
        
        evolver = GeneticAlgorithmEvolver(
            bot_generator=bot_generator,
            backtester=backtester
        )
        
        # Run evolution
        print("Running evolution...")
        
        evolver.run_evolution(
            num_generations=params['generations'],
            ohlcv_data=ohlcv_array,
            cycles=cycle_ranges,
            initial_balance=params['initial_balance']
        )
        
        # Save and display results
        print("\nSaving results...")
        
        evolver.save_top_bots(count=100)
        evolver.print_top_bots(count=10, initial_balance=params['initial_balance'])
        evolver.print_current_generation(initial_balance=params['initial_balance'])
        
        print("\n" + "="*60)
        print("GENETIC ALGORITHM - COMPLETE")
        print("="*60 + "\n")
        
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
    import os
    import glob
    
    # List available saved bots
    bot_files = glob.glob("bot_*.json")
    if bot_files:
        print(f"\nAvailable saved bots: {len(bot_files)}")
        for i, filename in enumerate(sorted(bot_files), 1):
            try:
                with open(filename, 'r') as f:
                    bot_data = json.load(f)
                print(f"  {i}. {filename} - ID:{bot_data['bot_id']}, Fitness:{bot_data['fitness_score']:.2f}, Survived:{bot_data.get('survival_generations', 0)} gens")
            except:
                print(f"  {i}. {filename} - (invalid file)")
    
    bot_choice = get_user_input(
        "Load existing bot ID, filename, or generate new? (id/filename/new)",
        "new"
    )
    
    if bot_choice.lower() == 'id':
        params['bot_id'] = get_user_input(
            "Bot ID to load",
            1,
            lambda x: validate_int(int(x), "bot_id", min_val=1)
        )
        params['bot_source'] = 'id'
    elif bot_choice.lower() == 'filename':
        filename = get_user_input(
            "Bot filename to load (e.g., bot_123.json)",
            "bot_1.json"
        )
        if not os.path.exists(filename):
            log_error(f"Bot file {filename} not found, generating new bot instead")
            params['bot_id'] = None
            params['bot_source'] = 'new'
        else:
            params['bot_filename'] = filename
            params['bot_source'] = 'file'
    else:
        params['bot_id'] = None  # Generate new random bot
        params['bot_source'] = 'new'
    
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
        
        if params['bot_source'] == 'file':
            # Load bot from file
            try:
                with open(params['bot_filename'], 'r') as f:
                    bot_data = json.load(f)
                
                # Reconstruct bot from saved data
                from src.bot_generator.compact_generator import CompactBotConfig
                import numpy as np
                
                config = bot_data['config']
                bot = CompactBotConfig(
                    bot_id=config['bot_id'] if 'bot_id' in config else bot_data['bot_id'],
                    num_indicators=config['num_indicators'],
                    indicator_indices=np.array(config['indicator_indices'] + [0] * (8 - len(config['indicator_indices'])), dtype=np.uint8),
                    indicator_params=np.array(config['indicator_params'] + [[0.0, 0.0, 0.0]] * (8 - len(config['indicator_params'])), dtype=np.float32),
                    risk_strategy_bitmap=config['risk_strategy_bitmap'],
                    tp_multiplier=config['tp_multiplier'],
                    sl_multiplier=config['sl_multiplier'],
                    leverage=config['leverage'],
                    survival_generations=bot_data.get('survival_generations', 0)
                )
                log_info(f"Loaded bot from {params['bot_filename']}")
                log_info(f"Bot ID: {bot.bot_id}, Survival generations: {bot.survival_generations}")
                
            except Exception as e:
                log_error(f"Failed to load bot from {params['bot_filename']}: {e}")
                log_warning("Generating random bot instead")
                params['bot_source'] = 'new'
        
        if params['bot_source'] == 'id':
            # TODO: Load bot from saved results by ID
            log_warning("Bot loading by ID not yet implemented, generating random bot instead")
            params['bot_source'] = 'new'
        
        if params['bot_source'] == 'new':
            # Generate single random bot with default params
            bot_generator = CompactBotGenerator(
                population_size=1,
                min_indicators=3,
                max_indicators=8,
                min_risk_strategies=2,
                max_risk_strategies=5,
                min_leverage=1,
                max_leverage=10,
                random_seed=42,
                gpu_context=gpu_context,
                gpu_queue=gpu_queue
            )
            
            bots = bot_generator.generate_population()
            bot = bots[0]
            log_info(f"Generated random bot with ID {bot.bot_id}")
        
        # Step 4: Run backtest
        log_info("\nStep 4/4: Running backtest...")
        
        backtester = CompactBacktester(
            gpu_context=gpu_context,
            gpu_queue=gpu_queue,
            initial_balance=params['initial_balance']
        )
        
        results = backtester.backtest_bots(
            bots=[bot],
            ohlcv_data=ohlcv_data,
            cycles=cycle_ranges
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
        print(f"  Indicators: {bot.num_indicators} indicators")
        for i in range(bot.num_indicators):
            print(f"    {i+1}. Indicator {bot.indicator_indices[i]} with params {bot.indicator_params[i]}")
        
        print(f"\n  Risk Strategy Bitmap: {bin(bot.risk_strategy_bitmap)}")
        print(f"  TP Multiplier: {bot.tp_multiplier:.2f}")
        print(f"  SL Multiplier: {bot.sl_multiplier:.2f}")
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
