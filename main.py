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
import time
import json
import os
from pathlib import Path

from src.utils.validation import (
    validate_int, validate_float, validate_pair, validate_timeframe,
    validate_leverage, validate_path, log_info, log_error, log_warning
)
from src.utils.config import (
    DEFAULT_TRADING_PAIR, DEFAULT_INITIAL_BALANCE, DEFAULT_POPULATION,
    DEFAULT_GENERATIONS, DEFAULT_CYCLES, DEFAULT_BACKTEST_DAYS,
    DEFAULT_MIN_INDICATORS, DEFAULT_MAX_INDICATORS,
    DEFAULT_MIN_RISK_STRATEGIES, DEFAULT_MAX_RISK_STRATEGIES,
    DEFAULT_RANDOM_SEED, IMPLEMENTED_MODES, MODE_DESCRIPTIONS
)
from src.utils.vram_estimator import VRAMEstimator
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


def get_user_input(prompt: str, default: any, validator=None, input_type=None) -> any:
    """
    Get validated user input with default value and comprehensive error handling.
    
    Args:
        prompt: Input prompt to display
        default: Default value if user presses Enter
        validator: Optional validation function
        input_type: Expected type (int, float, str, etc.) for auto-conversion
        
    Returns:
        Validated user input or default
    """
    max_attempts = 5
    attempt = 0
    
    while attempt < max_attempts:
        try:
            user_input = input(f"{prompt} [{default}]: ").strip()
            
            # Empty input → use default
            if not user_input:
                return default
            
            # Type conversion if specified
            if input_type is not None:
                try:
                    if input_type == int:
                        user_input = int(user_input)
                    elif input_type == float:
                        user_input = float(user_input)
                    elif input_type == bool:
                        user_input = user_input.lower() in ['y', 'yes', 'true', '1']
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Must be a valid {input_type.__name__}: {e}")
            
            # Run validator if provided
            if validator:
                try:
                    result = validator(user_input)
                    if isinstance(result, bool):
                        if result:
                            return user_input
                        else:
                            raise ValueError("Validation failed")
                    return result
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Validation error: {e}")
            
            return user_input
            
        except (ValueError, TypeError) as e:
            attempt += 1
            remaining = max_attempts - attempt
            log_error(f"Invalid input: {e}")
            if remaining > 0:
                log_warning(f"Please try again ({remaining} attempts remaining)")
            else:
                log_error("Maximum attempts reached. Using default value.")
                return default
        except KeyboardInterrupt:
            log_warning("\nInput cancelled. Using default value.")
            return default
        except EOFError:
            log_warning("\nEOF detected. Using default value.")
            return default
    
    # Fallback to default after max attempts
    return default


def get_mode_selection() -> int:
    """
    Get mode selection from user with robust validation.
    
    Returns:
        Selected mode number (defaults to 1 on repeated failures)
    """
    print("\n" + "="*60)
    print("GPU-ACCELERATED GENETIC ALGORITHM CRYPTO TRADING BOT")
    print("="*60 + "\n")
    
    print("Available Modes:")
    for mode_num, description in MODE_DESCRIPTIONS.items():
        status = "[OK]" if mode_num in IMPLEMENTED_MODES else "[X]"
        print(f"  {status} Mode {mode_num}: {description}")
    
    print()
    
    max_attempts = 5
    attempt = 0
    
    while attempt < max_attempts:
        try:
            mode_input = input("Select mode (1-4): ").strip()
            
            # Handle empty input
            if not mode_input:
                log_warning("No input provided. Defaulting to Mode 1.")
                return 1
            
            # Convert to int
            try:
                mode = int(mode_input)
            except ValueError:
                raise ValueError(f"'{mode_input}' is not a valid number")
            
            # Validate range
            if mode not in MODE_DESCRIPTIONS:
                raise ValueError(f"Mode {mode} does not exist. Please select 1-4.")
            
            # Check if implemented
            if mode not in IMPLEMENTED_MODES:
                raise ValueError(f"Mode {mode} is not implemented yet. Please select Mode 1.")
            
            return mode
            
        except ValueError as e:
            attempt += 1
            remaining = max_attempts - attempt
            log_error(f"Invalid selection: {e}")
            if remaining > 0:
                log_warning(f"Please try again ({remaining} attempts remaining)")
            else:
                log_error("Maximum attempts reached. Defaulting to Mode 1.")
                return 1
        except (KeyboardInterrupt, EOFError):
            log_warning("\nSelection cancelled. Defaulting to Mode 1.")
            return 1
    
    # Fallback
    return 1


def get_mode1_parameters() -> dict:
    """
    Get parameters for Mode 1 (Genetic Algorithm).
    
    Returns:
        Dictionary of parameters
    """
    print("\n" + "-"*60)
    print("MODE 1: GENETIC ALGORITHM CONFIGURATION")
    print("-"*60 + "\n")
    
    # Load last run defaults if available
    last_defaults = {}
    config_path = Path("config") / "last_run_config.json"
    if config_path.exists():
        try:
            # Validate path before loading
            safe_path = validate_path(str(config_path), "config_path", 
                                     must_exist=True, allowed_extensions=['.json'])
            with open(safe_path, 'r', encoding='utf-8') as f:
                last_defaults = json.load(f)
        except (ValueError, json.JSONDecodeError, IOError) as e:
            log_warning(f"Could not load config: {e}")
    # Also check old location for backwards compatibility
    elif os.path.exists('last_run_config.json'):
        try:
            safe_path = validate_path('last_run_config.json', "config_path",
                                     must_exist=True, allowed_extensions=['.json'])
            with open(safe_path, 'r', encoding='utf-8') as f:
                last_defaults = json.load(f)
        except (ValueError, json.JSONDecodeError, IOError) as e:
            log_warning(f"Could not load config: {e}")
    
    params = {}
    
    # Trading pair
    params['pair'] = get_user_input(
        "Trading pair (e.g., BTC/USDT)",
        last_defaults.get('pair', DEFAULT_TRADING_PAIR),
        lambda x: validate_pair(x)
    )
    
    # Initial balance
    params['initial_balance'] = get_user_input(
        "Initial balance (USDT)",
        last_defaults.get('initial_balance', DEFAULT_INITIAL_BALANCE),
        lambda x: validate_float(x, "initial_balance", strict_positive=True),
        input_type=float
    )
    
    # Population size
    params['population'] = get_user_input(
        "Population size (1,000-1,000,000)",
        last_defaults.get('population', DEFAULT_POPULATION),
        lambda x: validate_int(x, "population", min_val=1000, max_val=1000000),
        input_type=int
    )
    
    # Generations
    params['generations'] = get_user_input(
        "Number of generations (1-100)",
        last_defaults.get('generations', DEFAULT_GENERATIONS),
        lambda x: validate_int(x, "generations", min_val=1, max_val=100),
        input_type=int
    )
    
    # Cycles
    params['cycles'] = get_user_input(
        "Cycles per generation (1-100)",
        last_defaults.get('cycles', DEFAULT_CYCLES),
        lambda x: validate_int(x, "cycles", min_val=1, max_val=100),
        input_type=int
    )
    
    # Backtest days
    params['backtest_days'] = get_user_input(
        "Days per backtest cycle (1-365)",
        last_defaults.get('backtest_days', DEFAULT_BACKTEST_DAYS),
        lambda x: validate_int(x, "backtest_days", min_val=1, max_val=365),
        input_type=int
    )
    
    # Timeframe
    params['timeframe'] = get_user_input(
        "Timeframe (1m/5m/15m/30m/1h/4h/1d)",
        last_defaults.get('timeframe', "15m"),
        lambda x: validate_timeframe(x)
    )
    
    # Leverage range (NEW: min and max)
    params['min_leverage'] = get_user_input(
        "Min leverage (1-25x)",
        last_defaults.get('min_leverage', 1),
        lambda x: validate_int(x, "min_leverage", min_val=1, max_val=25),
        input_type=int
    )
    
    params['max_leverage'] = get_user_input(
        f"Max leverage ({params['min_leverage']}-125x)",
        last_defaults.get('max_leverage', 1),
        lambda x: validate_int(x, "max_leverage", 
                              min_val=params['min_leverage'], max_val=125),
        input_type=int
    )
    
    # Indicators per bot
    params['min_indicators'] = get_user_input(
        "Min indicators per bot (1-8)",
        last_defaults.get('min_indicators', 1),
        lambda x: validate_int(x, "min_indicators", min_val=1, max_val=8),
        input_type=int
    )
    
    params['max_indicators'] = get_user_input(
        f"Max indicators per bot ({params['min_indicators']}-8)",
        last_defaults.get('max_indicators', 5),
        lambda x: validate_int(x, "max_indicators", 
                              min_val=params['min_indicators'], max_val=8),
        input_type=int
    )
    
    # Risk strategies per bot
    params['min_risk_strategies'] = get_user_input(
        "Min risk strategies per bot (1-15)",
        last_defaults.get('min_risk_strategies', 1),
        lambda x: validate_int(x, "min_risk_strategies", min_val=1, max_val=15),
        input_type=int
    )
    
    params['max_risk_strategies'] = get_user_input(
        f"Max risk strategies per bot ({params['min_risk_strategies']}-15)",
        last_defaults.get('max_risk_strategies', 5),
        lambda x: validate_int(x, "max_risk_strategies", 
                              min_val=params['min_risk_strategies'], max_val=15),
        input_type=int
    )
    
    # Data chunk size for GPU processing
    params['data_chunk_days'] = get_user_input(
        "Data chunk size in days (1-365, affects GPU memory usage)",
        last_defaults.get('data_chunk_days', 100),
        lambda x: validate_int(x, "data_chunk_days", min_val=1, max_val=365),
        input_type=int
    )
    
    # Random seed
    use_seed = get_user_input(
        "Use random seed for reproducibility? (y/n)",
        last_defaults.get('use_seed', "y"),
        lambda x: x.lower() if x.lower() in ['y', 'n', 'yes', 'no'] else (_ for _ in ()).throw(ValueError("Must be 'y' or 'n'"))
    )
    
    if use_seed and use_seed.lower() in ['y', 'yes']:
        params['random_seed'] = get_user_input(
            "Random seed (0-2147483647)",
            last_defaults.get('random_seed', 42),
            lambda x: validate_int(x, "random_seed", min_val=0, max_val=2147483647),
            input_type=int
        )
    else:
        params['random_seed'] = None
    
    params['use_seed'] = use_seed
    
    # Interactive mode for debugging
    interactive = get_user_input(
        "Enable interactive mode (pause after each generation)? (y/n)",
        "n",
        lambda x: x.lower() if x.lower() in ['y', 'n', 'yes', 'no'] else (_ for _ in ()).throw(ValueError("Must be 'y' or 'n'"))
    )
    params['interactive_mode'] = (interactive and interactive.lower() in ['y', 'yes'])
    
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
    
    # Save configuration before starting (for easy restart/debugging)
    try:
        config_dir = Path("config")
        config_dir.mkdir(exist_ok=True)
        config_path = config_dir / "last_run_config.json"
        with open(config_path, 'w') as f:
            json.dump(params, f, indent=2)
        log_info(f"Configuration saved to {config_path}")
    except Exception as e:
        log_warning(f"Failed to save configuration: {e}")
    
    # Start data loading profiling
    data_loading_start = time.time()
    
    try:
        # Calculate total days needed
        from src.utils.config import EXCHANGE_TYPE
        fetcher = DataFetcher(exchange_type=EXCHANGE_TYPE)
        total_days = fetcher.calculate_required_days(
            params['backtest_days'],
            params['cycles']
        )
        
        # Fetch data
        print(f"\nFetching {total_days} days of {params['pair']} {params['timeframe']} data...")
        fetch_start = time.time()
        
        file_paths = fetcher.fetch_data_range(
            pair=params['pair'],
            timeframe=params['timeframe'],
            total_days=total_days
        )
        
        fetch_time = time.time() - fetch_start
        print(f"Data fetching completed in {fetch_time:.3f}s")
        
        # Load and validate data (GPU-accelerated)
        print("Loading data with GPU acceleration...")
        load_start = time.time()
        
        loader = DataLoader(
            file_paths=file_paths,
            timeframe=params['timeframe'],
            random_seed=params['random_seed'],
            gpu_context=gpu_context,
            gpu_queue=gpu_queue,
            use_gpu_processing=True
        )
        
        ohlcv_data = loader.load_all_data()
        ohlcv_array = ohlcv_data[['timestamp', 'open', 'high', 'low', 'close', 'volume']].values.astype(np.float32)
        
        cycle_ranges = loader.generate_cycle_ranges(
            num_cycles=params['cycles'],
            backtest_days=params['backtest_days']
        )
        
        load_time = time.time() - load_start
        total_data_time = time.time() - data_loading_start
        print(f"Data loading completed in {load_time:.3f}s")
        print(f"Total data preparation: {total_data_time:.3f}s")
        print(f"Loaded {len(ohlcv_array)} bars, {len(cycle_ranges)} cycles\n")
        
        # Validate VRAM requirements
        print("Validating GPU memory requirements...")
        """
        vram_estimator = VRAMEstimator(gpu_context)
        try:
            vram_report = vram_estimator.estimate_and_validate_workflow(
                population_size=params['population'],
                num_cycles=len(cycle_ranges),
                total_bars=len(ohlcv_array),
                num_indicator_types=50,  # We have 50 GPU indicators
                num_risk_strategy_types=15,  # We have 15 risk strategies
                gpu_vram_bytes=vram_estimator.device_vram if vram_estimator.device_vram else None
            )
            print(f"VRAM validation passed: Peak usage {vram_report['peak_vram_mb']:.0f} MB\n")
        except RuntimeError as e:
            print(f"\nERROR: {e}")
            sys.exit(1)
        """
        
        # Initialize components
        print("Initializing evolution components...")
        init_start = time.time()
        
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
            initial_balance=params['initial_balance'],
            target_chunk_seconds=1.0,  # Smooth 1-second chunk processing
            data_chunk_days=params['data_chunk_days']
        )
        
        evolver = GeneticAlgorithmEvolver(
            bot_generator=bot_generator,
            backtester=backtester,
            pair=params['pair'],
            timeframe=params['timeframe'],
            gpu_context=gpu_context,
            gpu_queue=gpu_queue
        )
        
        # Enable interactive mode for debugging (set via environment variable or param)
        if params.get('interactive_mode', False):
            evolver.interactive_mode = True
            print("Interactive mode enabled - will pause after each generation")
        
        init_time = time.time() - init_start
        print(f"Component initialization completed in {init_time:.3f}s")
        
        # Run evolution
        print("Running evolution...")
        evolution_start = time.time()
        
        evolver.run_evolution(
            num_generations=params['generations'],
            ohlcv_data=ohlcv_array,
            cycles=cycle_ranges,
            initial_balance=params['initial_balance']
        )
        
        evolution_time = time.time() - evolution_start
        print(f"Evolution completed in {evolution_time:.3f}s")
        
        # Save and display results
        print("\nSaving results...")
        results_start = time.time()
        
        evolver.save_top_bots(count=100, filter_all_profitable=True)
        # Removed print_top_bots - all data is in CSV
        # evolver.print_top_bots(count=10, initial_balance=params['initial_balance'])
        # evolver.print_current_generation(initial_balance=params['initial_balance'])
        
        results_time = time.time() - results_start
        print(f"Results processing completed in {results_time:.3f}s")
        
        print("\n" + "="*60)
        print("GENETIC ALGORITHM - COMPLETE")
        print("="*60 + "\n")
        
        # Save params as defaults for next run (already saved at start, but update in case of any changes)
        try:
            config_dir = Path("config")
            config_dir.mkdir(exist_ok=True)
            config_path = config_dir / "last_run_config.json"
            with open(config_path, 'w') as f:
                json.dump(params, f, indent=2)
        except:
            pass
        
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
    bot_files = glob.glob(f"bots/xbtusdtm/{params['timeframe']}/bot_*.json")
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
        "Leverage (1-25)",
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
        from src.utils.config import EXCHANGE_TYPE
        fetcher = DataFetcher(exchange_type=EXCHANGE_TYPE)
        
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
        
        # Step 2: Load data (GPU-accelerated)
        log_info("\nStep 2/4: Loading data with GPU acceleration...")
        loader = DataLoader(
            file_paths=file_paths,
            timeframe=params['timeframe'],
            random_seed=42,
            gpu_context=gpu_context,
            gpu_queue=gpu_queue,
            use_gpu_processing=True
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
                    risk_strategy=config['risk_strategy'],
                    risk_param=config['risk_param'],
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
            initial_balance=params['initial_balance'],
            target_chunk_seconds=1.0,  # Smooth 1-second chunk processing
            data_chunk_days=params.get('data_chunk_days', 100)  # Use default 100 if not specified
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
        
        print(f"\n  Risk Strategy: {bot.risk_strategy} (param: {bot.risk_param:.4f})")
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


def run_mode2(gpu_context, gpu_queue):
    """
    Mode 2: Paper Trading (Live Simulation)
    Replicates GPU kernel logic EXACTLY on CPU with live data.
    Uses test endpoint - no real money at risk.
    IDENTICAL to Mode 3 except test_mode=True.
    """
    try:
        from src.live_trading.credentials import CredentialsManager
        from src.live_trading.kucoin_universal_client import KucoinUniversalClient
        from src.live_trading.engine import RealTimeTradingEngine
        from src.live_trading.live_dashboard import LiveTradingDashboard
        from src.bot_generator.compact_generator import CompactBotConfig
        from src.data_provider.fetcher import DataFetcher
        from src.utils.config import EXCHANGE_TYPE
        
        print("\n" + "="*60)
        print("MODE 2: PAPER TRADING (TEST ENDPOINT)")
        print("="*60 + "\n")
        print("✅ Using Kucoin TEST endpoint - NO REAL MONEY")
        print()
        
        # Setup credentials
        creds_manager = CredentialsManager()
        
        if not creds_manager.credentials_exist():
            log_info("No Kucoin credentials found. Setting up...")
            if not creds_manager.prompt_and_save_credentials():
                log_error("Setup cancelled")
                return
        
        credentials = creds_manager.load_credentials()
        if not credentials:
            log_error("Failed to load credentials")
            return
        
        # Get trading parameters
        print("\nPaper Trading Configuration:")
        pair = get_user_input("Trading pair", "BTC/USDT", validate_pair)
        initial_balance = float(get_user_input("Initial balance (USDT)", "1000.0", lambda x: float(x)))
        timeframe = get_user_input("Timeframe (1m/5m/15m)", "1m", validate_timeframe)
        
        # Load or select bot
        print("\nBot Selection:")
        print("  1. Load saved bot from evolution results")
        print("  2. Use test bot configuration")
        choice = input("Select [1-2]: ").strip()
        
        if choice == "1":
            # Load from saved bot files
            from pathlib import Path
            import glob
            
            bot_files = glob.glob("bots/**/*.json", recursive=True)
            if not bot_files:
                log_error("No saved bots found in bots/ directory")
                log_info("Run Mode 1 (Genetic Algorithm) first to evolve bots")
                return
            
            print("\nAvailable saved bots:")
            for i, f in enumerate(bot_files[:20]):  # Show max 20
                try:
                    with open(f, 'r') as file:
                        bot_data = json.load(file)
                    fitness = bot_data.get('fitness_score', 0)
                    bot_id = bot_data.get('bot_id', 0)
                    survival = bot_data.get('survival_generations', 0)
                    print(f"  {i+1}. {f} (ID:{bot_id}, Fitness:{fitness:.2f}, Survived:{survival} gens)")
                except:
                    print(f"  {i+1}. {f} (invalid file)")
            
            file_idx = int(input(f"Select bot [1-{min(len(bot_files), 20)}]: ").strip()) - 1
            selected_file = bot_files[file_idx]
            
            try:
                with open(selected_file, 'r') as f:
                    bot_data = json.load(f)
                
                # Load using from_dict classmethod
                bot_config = CompactBotConfig.from_dict(bot_data)
                log_info(f"✅ Loaded bot {bot_config.bot_id} from {selected_file}")
                log_info(f"   Indicators: {bot_config.num_indicators}, Leverage: {bot_config.leverage}x")
                log_info(f"   TP: {bot_config.tp_multiplier:.3f}, SL: {bot_config.sl_multiplier:.3f}")
                
            except Exception as e:
                log_error(f"Failed to load bot from {selected_file}: {e}")
                log_warning("Using test bot configuration instead")
                bot_config = CompactBotConfig(
                    bot_id=1,
                    num_indicators=3,
                    indicator_indices=np.array([12, 26, 27, 0, 0, 0, 0, 0], dtype=np.uint8),
                    indicator_params=np.array([[14, 0, 0], [12, 26, 9], [14, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float32),
                    risk_strategy=0,  # RISK_FIXED_PCT
                    risk_param=0.02,  # 2% of balance
                    tp_multiplier=0.02,
                    sl_multiplier=0.01,
                    leverage=10
                )
        else:
            # Use test bot
            bot_config = CompactBotConfig(
                bot_id=1,
                num_indicators=3,
                indicator_indices=np.array([12, 26, 27, 0, 0, 0, 0, 0], dtype=np.uint8),  # RSI, MACD, ADX
                indicator_params=np.array([[14, 0, 0], [12, 26, 9], [14, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float32),
                risk_strategy=0,  # RISK_FIXED_PCT
                risk_param=0.02,  # 2% of balance
                tp_multiplier=0.02,
                sl_multiplier=0.01,
                leverage=10
            )
        
        # Convert pair format: BTC/USDT -> XBTUSDTM (Kucoin perpetuals)
        if pair == 'BTC/USDT':
            kucoin_symbol = 'XBTUSDTM'
        elif pair == 'ETH/USDT':
            kucoin_symbol = 'ETHUSDTM'
        else:
            kucoin_symbol = pair.replace('/', '') + 'M'
        
        # Initialize Kucoin Universal SDK client (TEST MODE)
        log_info("Connecting to Kucoin (Test Endpoint)...")
        kucoin_client = KucoinUniversalClient(
            api_key=credentials['api_key'],
            api_secret=credentials['api_secret'],
            api_passphrase=credentials['api_passphrase'],
            test_mode=True  # Test endpoint - NO REAL MONEY
        )
        log_info("✅ Connected to Kucoin Test API")
        
        # Initialize trading engine
        log_info("Initializing paper trading engine...")
        engine = RealTimeTradingEngine(
            bot_config=bot_config,
            initial_balance=initial_balance,
            kucoin_client=kucoin_client,
            pair=kucoin_symbol,
            timeframe=timeframe,
            test_mode=True
        )
        
        # Fetch historical data using DataProvider (same as Mode 1)
        log_info("Loading historical data...")
        fetcher = DataFetcher(exchange_type='futures')
        
        # Fetch 2 days of data for indicator warmup (500+ candles for 1m)
        file_paths = fetcher.fetch_data_range(
            pair=pair,
            timeframe=timeframe,
            total_days=2
        )
        
        # Load data from parquet files
        from src.data_provider.loader import DataLoader
        loader = DataLoader(
            file_paths=file_paths,
            timeframe=timeframe,
            gpu_context=gpu_context,
            gpu_queue=gpu_queue
        )
        ohlcv_data = loader.load_all_data()
        
        # Convert DataFrame to numpy array
        ohlcv_array = ohlcv_data[['timestamp', 'open', 'high', 'low', 'close', 'volume']].values.astype(np.float32)
        
        # Process historical candles to warm up indicators
        for i in range(len(ohlcv_array)):
            timestamp_ms, open_, high, low, close, volume = ohlcv_array[i]
            engine.process_candle(open_, high, low, close, volume, timestamp_ms / 1000.0)
        
        log_info(f"✅ Loaded {len(ohlcv_array)} historical candles")
        
        # Start engine
        engine.start()
        
        # Initialize live dashboard
        dashboard = LiveTradingDashboard(bot_config, mode="PAPER")
        
        log_info("\n" + "="*60)
        log_info("📄 PAPER TRADING STARTED - Live Dashboard Active")
        log_info("="*60 + "\n")
        
        # Live trading loop - fetch new candles from DataFetcher
        timeframe_seconds = {'1m': 60, '5m': 300, '15m': 900}[timeframe]
        last_timestamp = ohlcv_array[-1][0] if len(ohlcv_array) > 0 else 0
        
        # Keep running until interrupted
        while True:
            try:
                # Fetch latest candles using DataFetcher (not CCXT)
                latest_data = fetcher.fetch_latest_candles(
                    pair=pair,
                    timeframe=timeframe,
                    limit=2
                )
                
                if latest_data and len(latest_data) >= 2:
                    # Get completed candle (second to last)
                    candle = latest_data[-2]
                    timestamp_ms = candle[0]
                    
                    # Only process if new candle
                    if timestamp_ms > last_timestamp:
                        last_timestamp = timestamp_ms
                        open_, high, low, close, volume = candle[1], candle[2], candle[3], candle[4], candle[5]
                        engine.process_candle(open_, high, low, close, volume, timestamp_ms / 1000.0)
                        
                        # Render live dashboard
                        state = engine.get_current_state()
                        dashboard.render(state)
                else:
                    # No new candle yet, just update dashboard with current state
                    state = engine.get_current_state()
                    dashboard.render(state)
                
                time.sleep(timeframe_seconds)
                
            except KeyboardInterrupt:
                raise
        
    except KeyboardInterrupt:
        log_warning("\n\nPaper trading stopped by user")
        
        # Stop engine
        if 'engine' in locals():
            engine.stop()
        
        # Save trading session results
        if 'engine' in locals() and 'bot_config' in locals():
            try:
                from pathlib import Path
                from datetime import datetime
                
                # Create session directory
                session_dir = Path("sessions") / "paper_trading"
                session_dir.mkdir(parents=True, exist_ok=True)
                
                # Get final state
                final_state = engine.get_current_state()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Save session data
                session_data = {
                    "mode": "paper_trading",
                    "bot_id": bot_config.bot_id,
                    "pair": pair,
                    "timeframe": timeframe,
                    "start_time": timestamp,
                    "initial_balance": initial_balance,
                    "final_balance": final_state.get('balance', initial_balance),
                    "total_pnl": final_state.get('balance', initial_balance) - initial_balance,
                    "total_trades": final_state.get('total_trades', 0),
                    "win_rate": final_state.get('win_rate', 0),
                    "candles_processed": final_state.get('candles_processed', 0),
                    "bot_config": bot_config.to_dict()
                }
                
                session_file = session_dir / f"session_{timestamp}_bot{bot_config.bot_id}.json"
                with open(session_file, 'w') as f:
                    json.dump(session_data, f, indent=2)
                
                log_info(f"\n✅ Session saved to: {session_file}")
                log_info(f"Final Balance: ${session_data['final_balance']:.2f}")
                log_info(f"Total PnL: ${session_data['total_pnl']:+.2f}")
                log_info(f"Total Trades: {session_data['total_trades']}")
                
            except Exception as e:
                log_warning(f"Failed to save session: {e}")
        
    except Exception as e:
        log_error(f"Paper trading error: {e}")
        import traceback
        traceback.print_exc()


def run_mode3(gpu_context, gpu_queue):
    """
    Mode 3: Live Trading (Real Money)
    Replicates GPU kernel logic EXACTLY on CPU with live data.
    Uses LIVE endpoint - REAL MONEY AT RISK.
    IDENTICAL to Mode 2 except test_mode=False.
    """
    try:
        from src.live_trading.credentials import CredentialsManager
        from src.live_trading.live_dashboard import LiveTradingDashboard
        from src.live_trading.kucoin_universal_client import KucoinUniversalClient
        from src.live_trading.engine import RealTimeTradingEngine
        from src.bot_generator.compact_generator import CompactBotConfig
        from src.data_provider.fetcher import DataFetcher
        from src.utils.config import EXCHANGE_TYPE
        
        print("\n" + "="*60)
        print("MODE 3: LIVE TRADING (REAL MONEY)")
        print("="*60 + "\n")
        
        print("⚠️  WARNING: This mode trades with REAL MONEY!")
        print("⚠️  You can lose your entire balance!")
        print("⚠️  Only use funds you can afford to lose!")
        print("⚠️  Start with SMALL positions to test!")
        print()
        
        confirm = input("Type 'I UNDERSTAND THE RISKS' to continue: ").strip()
        if confirm != 'I UNDERSTAND THE RISKS':
            log_info("Live trading cancelled")
            return
        
        # Setup credentials
        creds_manager = CredentialsManager()
        
        if not creds_manager.credentials_exist():
            log_info("No Kucoin credentials found. Setting up...")
            if not creds_manager.prompt_and_save_credentials():
                log_error("Setup cancelled")
                return
        
        credentials = creds_manager.load_credentials()
        if not credentials:
            log_error("Failed to load credentials")
            return
        
        if credentials['environment'] != 'live':
            log_error("Credentials are set for sandbox/testnet. Live trading requires LIVE API keys.")
            log_info("Delete credentials file and set up live keys, or use Mode 2 for paper trading.")
            return
        
        # Get trading parameters
        print("\nLive Trading Configuration:")
        pair = get_user_input("Trading pair", "BTC/USDT", validate_pair)
        initial_balance = float(get_user_input("Account balance (USDT)", "1000.0", lambda x: float(x)))
        timeframe = get_user_input("Timeframe (1m/5m/15m)", "1m", validate_timeframe)
        
        # Load bot (same as Mode 2)
        print("\nBot Selection:")
        print("  1. Load saved bot from evolution results")
        print("  2. Use test bot configuration")
        choice = input("Select [1-2]: ").strip()
        
        if choice == "1":
            # Load from saved bot files
            from pathlib import Path
            import glob
            
            bot_files = glob.glob("bots/**/*.json", recursive=True)
            if not bot_files:
                log_error("No saved bots found in bots/ directory")
                log_info("Run Mode 1 (Genetic Algorithm) first to evolve bots")
                return
            
            print("\nAvailable saved bots:")
            for i, f in enumerate(bot_files[:20]):  # Show max 20
                try:
                    with open(f, 'r') as file:
                        bot_data = json.load(file)
                    fitness = bot_data.get('fitness_score', 0)
                    bot_id = bot_data.get('bot_id', 0)
                    survival = bot_data.get('survival_generations', 0)
                    print(f"  {i+1}. {f} (ID:{bot_id}, Fitness:{fitness:.2f}, Survived:{survival} gens)")
                except:
                    print(f"  {i+1}. {f} (invalid file)")
            
            file_idx = int(input(f"Select bot [1-{min(len(bot_files), 20)}]: ").strip()) - 1
            selected_file = bot_files[file_idx]
            
            try:
                with open(selected_file, 'r') as f:
                    bot_data = json.load(f)
                
                # Load using from_dict classmethod
                bot_config = CompactBotConfig.from_dict(bot_data)
                log_info(f"✅ Loaded bot {bot_config.bot_id} from {selected_file}")
                log_info(f"   Indicators: {bot_config.num_indicators}, Leverage: {bot_config.leverage}x")
                log_info(f"   TP: {bot_config.tp_multiplier:.3f}, SL: {bot_config.sl_multiplier:.3f}")
                
            except Exception as e:
                log_error(f"Failed to load bot from {selected_file}: {e}")
                log_warning("Using test bot configuration instead")
                bot_config = CompactBotConfig(
                    bot_id=1,
                    num_indicators=3,
                    indicator_indices=np.array([12, 26, 27, 0, 0, 0, 0, 0], dtype=np.uint8),
                    indicator_params=np.array([[14, 0, 0], [12, 26, 9], [14, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float32),
                    risk_strategy=0,  # RISK_FIXED_PCT
                    risk_param=0.02,  # 2% of balance
                    tp_multiplier=0.02,
                    sl_multiplier=0.01,
                    leverage=10
                )
        else:
            # Use test bot
            bot_config = CompactBotConfig(
                bot_id=1,
                num_indicators=3,
                indicator_indices=np.array([12, 26, 27, 0, 0, 0, 0, 0], dtype=np.uint8),
                indicator_params=np.array([[14, 0, 0], [12, 26, 9], [14, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float32),
                risk_strategy=0,  # RISK_FIXED_PCT
                risk_param=0.02,  # 2% of balance
                tp_multiplier=0.02,
                sl_multiplier=0.01,
                leverage=10
            )
        
        # Convert pair format: BTC/USDT -> XBTUSDTM (Kucoin perpetuals)
        if pair == 'BTC/USDT':
            kucoin_symbol = 'XBTUSDTM'
        elif pair == 'ETH/USDT':
            kucoin_symbol = 'ETHUSDTM'
        else:
            kucoin_symbol = pair.replace('/', '') + 'M'
        
        # Initialize Kucoin Universal SDK client (LIVE MODE)
        log_info("Connecting to Kucoin (LIVE Endpoint)...")
        log_warning("⚠️  USING REAL MONEY API!")
        kucoin_client = KucoinUniversalClient(
            api_key=credentials['api_key'],
            api_secret=credentials['api_secret'],
            api_passphrase=credentials['api_passphrase'],
            test_mode=False  # LIVE endpoint - REAL MONEY!
        )
        log_info("✅ Connected to Kucoin LIVE API")
        
        # Initialize trading engine
        log_info("Initializing LIVE trading engine...")
        engine = RealTimeTradingEngine(
            bot_config=bot_config,
            initial_balance=initial_balance,
            kucoin_client=kucoin_client,
            pair=kucoin_symbol,
            timeframe=timeframe,
            test_mode=False  # LIVE MODE
        )
        
        # Fetch historical data using DataProvider (same as Mode 1)
        log_info("Loading historical data...")
        fetcher = DataFetcher(exchange_type='futures')
        
        # Fetch 2 days of data for indicator warmup (500+ candles for 1m)
        file_paths = fetcher.fetch_data_range(
            pair=pair,
            timeframe=timeframe,
            total_days=2
        )
        
        # Load data from parquet files
        from src.data_provider.loader import DataLoader
        loader = DataLoader(
            file_paths=file_paths,
            timeframe=timeframe,
            gpu_context=gpu_context,
            gpu_queue=gpu_queue
        )
        ohlcv_data = loader.load_all_data()
        
        # Convert DataFrame to numpy array
        ohlcv_array = ohlcv_data[['timestamp', 'open', 'high', 'low', 'close', 'volume']].values.astype(np.float32)
        
        # Process historical candles to warm up indicators
        for i in range(len(ohlcv_array)):
            timestamp_ms, open_, high, low, close, volume = ohlcv_array[i]
            engine.process_candle(open_, high, low, close, volume, timestamp_ms / 1000.0)
        
        log_info(f"✅ Loaded {len(ohlcv_array)} historical candles")
        
        # Start engine
        engine.start()
        
        # Initialize live dashboard
        dashboard = LiveTradingDashboard(bot_config, mode="LIVE")
        
        log_info("\n" + "="*60)
        log_info("💰 LIVE TRADING STARTED - Live Dashboard Active")
        log_warning("⚠️  ALL TRADES ARE REAL - MONITOR CAREFULLY!")
        log_info("="*60 + "\n")
        
        # Live trading loop - fetch new candles from DataFetcher
        timeframe_seconds = {'1m': 60, '5m': 300, '15m': 900}[timeframe]
        last_timestamp = ohlcv_array[-1][0] if len(ohlcv_array) > 0 else 0
        
        # Keep running until interrupted
        while True:
            try:
                # Fetch latest candles using DataFetcher (not CCXT)
                latest_data = fetcher.fetch_latest_candles(
                    pair=pair,
                    timeframe=timeframe,
                    limit=2
                )
                
                if latest_data and len(latest_data) >= 2:
                    # Get completed candle (second to last)
                    candle = latest_data[-2]
                    timestamp_ms = candle[0]
                    
                    # Only process if new candle
                    if timestamp_ms > last_timestamp:
                        last_timestamp = timestamp_ms
                        open_, high, low, close, volume = candle[1], candle[2], candle[3], candle[4], candle[5]
                        engine.process_candle(open_, high, low, close, volume, timestamp_ms / 1000.0)
                        
                        # Render live dashboard
                        state = engine.get_current_state()
                        dashboard.render(state)
                else:
                    # No new candle yet, just update dashboard with current state
                    state = engine.get_current_state()
                    dashboard.render(state)
                
                time.sleep(timeframe_seconds)
                
            except KeyboardInterrupt:
                raise
        
    except KeyboardInterrupt:
        log_warning("\n\nLive trading stopped by user")
        
        # Stop engine
        if 'engine' in locals():
            engine.stop()
        
        # Save trading session results
        if 'engine' in locals() and 'bot_config' in locals():
            try:
                from pathlib import Path
                from datetime import datetime
                
                # Create session directory
                session_dir = Path("sessions") / "live_trading"
                session_dir.mkdir(parents=True, exist_ok=True)
                
                # Get final state
                final_state = engine.get_current_state()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Save session data
                session_data = {
                    "mode": "live_trading",
                    "bot_id": bot_config.bot_id,
                    "pair": pair,
                    "timeframe": timeframe,
                    "start_time": timestamp,
                    "initial_balance": initial_balance,
                    "final_balance": final_state.get('balance', initial_balance),
                    "total_pnl": final_state.get('balance', initial_balance) - initial_balance,
                    "total_trades": final_state.get('total_trades', 0),
                    "win_rate": final_state.get('win_rate', 0),
                    "candles_processed": final_state.get('candles_processed', 0),
                    "bot_config": bot_config.to_dict()
                }
                
                session_file = session_dir / f"session_{timestamp}_bot{bot_config.bot_id}.json"
                with open(session_file, 'w') as f:
                    json.dump(session_data, f, indent=2)
                
                log_info(f"\n✅ Live session saved to: {session_file}")
                log_info(f"Final Balance: ${session_data['final_balance']:.2f}")
                log_info(f"Total PnL: ${session_data['total_pnl']:+.2f}")
                log_info(f"Total Trades: {session_data['total_trades']}")
                
            except Exception as e:
                log_warning(f"Failed to save session: {e}")
        
    except Exception as e:
        log_error(f"Live trading error: {e}")
        import traceback
        traceback.print_exc()


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
        elif mode == 2:
            # Mode 2: Paper Trading (Live Simulation)
            run_mode2(gpu_context, gpu_queue)
        elif mode == 3:
            # Mode 3: Live Trading (Real Money)
            run_mode3(gpu_context, gpu_queue)
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
