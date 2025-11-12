#!/usr/bin/env python3
"""
Quick test script for Mode 2 (Paper Trading)
Tests the complete trading loop with DataProvider
"""

import sys
import os
import time
import numpy as np
from datetime import datetime

# Import what we need from main.py
sys.path.insert(0, os.path.dirname(__file__))

from src.utils.validation import log_info, log_error
from src.live_trading.credentials import CredentialsManager
from src.live_trading.kucoin_universal_client import KucoinUniversalClient
from src.live_trading.engine import RealTimeTradingEngine
from src.bot_generator.compact_generator import CompactBotConfig
from src.data_provider.fetcher import DataFetcher
from src.data_provider.loader import DataLoader
from src.utils.config import EXCHANGE_TYPE
import pyopencl as cl
import json

def initialize_gpu():
    """Initialize OpenCL GPU - from main.py"""
    try:
        platforms = cl.get_platforms()
        if not platforms:
            raise RuntimeError("No OpenCL platforms found")
        
        gpu_device = None
        for platform in platforms:
            try:
                devices = platform.get_devices(device_type=cl.device_type.GPU)
                if devices:
                    gpu_device = devices[0]
                    break
            except cl.RuntimeError:
                continue
        
        if gpu_device is None:
            raise RuntimeError("No OpenCL GPU device found")
        
        context = cl.Context([gpu_device])
        queue = cl.CommandQueue(context)
        
        log_info("✅ GPU Initialized: " + gpu_device.name)
        return context, queue, gpu_device
        
    except Exception as e:
        log_error(f"GPU initialization failed: {e}")
        raise

def cleanup_gpu(context, queue):
    """Clean up GPU resources"""
    if queue is not None:
        queue.finish()
    # Context cleanup is automatic in pyopencl

def test_mode2():
    """Test Mode 2 - Paper Trading with DataProvider"""
    
    log_info("=" * 60)
    log_info("TESTING MODE 2: PAPER TRADING")
    log_info("=" * 60)
    
    # Initialize GPU
    log_info("Initializing GPU...")
    gpu_context, gpu_queue, gpu_device = initialize_gpu()
    
    # Configuration
    pair = "BTC/USDT"
    timeframe = "1m"
    timeframe_seconds = 60
    initial_balance = 1000.0
    test_mode = True  # Paper trading
    
    log_info(f"Pair: {pair}")
    log_info(f"Timeframe: {timeframe}")
    log_info(f"Initial Balance: ${initial_balance}")
    log_info(f"Test Mode: {test_mode}")
    
    # Load Kucoin credentials
    log_info("Loading Kucoin credentials...")
    creds_manager = CredentialsManager()
    if not creds_manager.credentials_exist():
        log_error("Kucoin credentials not configured. Run main.py first to set up credentials.")
        cleanup_gpu(gpu_context, gpu_queue)
        return False
    
    credentials = creds_manager.load_credentials()
    
    # Initialize Kucoin Universal Client (test mode)
    log_info("Initializing Kucoin Universal Client (Test Mode)...")
    kucoin_client = KucoinUniversalClient(
        api_key=credentials['api_key'],
        api_secret=credentials['api_secret'],
        api_passphrase=credentials['api_passphrase'],
        test_mode=test_mode
    )
    
    # Load bot configuration (use first bot found or test bot)
    log_info("Loading bot configuration...")
    import glob
    bot_files = glob.glob("bots/**/*.json", recursive=True)
    
    if bot_files:
        bot_file = bot_files[0]
        log_info(f"Loading bot from: {bot_file}")
        with open(bot_file, 'r') as f:
            bot_data = json.load(f)
        bot_config = CompactBotConfig.from_dict(bot_data)
    else:
        log_info("No saved bots found, creating test bot...")
        bot_config = CompactBotConfig(
            bot_id=1,
            num_indicators=3,
            indicator_indices=np.array([12, 26, 27, 0, 0, 0, 0, 0], dtype=np.uint8),
            indicator_params=np.array([[14, 0, 0], [12, 26, 9], [14, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float32),
            risk_strategy=0,
            risk_param=0.02,
            tp_multiplier=0.02,
            sl_multiplier=0.01,
            leverage=1
        )
    
    log_info(f"Bot loaded: {bot_config.num_indicators} indicators, leverage: {bot_config.leverage}x")
    
    # Initialize trading engine
    log_info("Initializing Real-Time Trading Engine...")
    # Note: RealTimeTradingEngine uses Kucoin symbol format (XBTUSDTM)
    kucoin_symbol = "XBTUSDTM"  # BTC perpetual futures
    engine = RealTimeTradingEngine(
        bot_config=bot_config,
        initial_balance=initial_balance,
        kucoin_client=kucoin_client,
        pair=kucoin_symbol,
        timeframe=timeframe,
        test_mode=test_mode
    )
    
    # Initialize DataProvider
    log_info("Initializing DataProvider...")
    fetcher = DataFetcher(exchange_type=EXCHANGE_TYPE)
    
    # Fetch and load historical data for warmup
    log_info(f"Fetching 2 days of historical data for {pair} {timeframe}...")
    file_paths = fetcher.fetch_data_range(
        pair=pair,
        timeframe=timeframe,
        total_days=2
    )
    
    if not file_paths:
        log_error("Failed to fetch historical data")
        cleanup_gpu(gpu_context, gpu_queue)
        return False
    
    log_info(f"Loading historical data from {len(file_paths)} files...")
    loader = DataLoader(
        file_paths=file_paths,
        timeframe=timeframe,
        gpu_context=gpu_context,
        gpu_queue=gpu_queue
    )
    ohlcv_data = loader.load_all_data()
    
    if ohlcv_data is None or ohlcv_data.empty:
        log_error("Failed to load historical data")
        cleanup_gpu(gpu_context, gpu_queue)
        return False
    
    # Convert DataFrame to numpy array
    ohlcv_array = ohlcv_data[['timestamp', 'open', 'high', 'low', 'close', 'volume']].values.astype(np.float32)
    log_info(f"✅ Loaded {len(ohlcv_array)} historical candles")
    
    # Process historical candles to warm up indicators
    log_info("Processing historical candles to warm up indicators...")
    for i in range(len(ohlcv_array)):
        timestamp_ms, open_, high, low, close, volume = ohlcv_array[i]
        engine.process_candle(open_, high, low, close, volume, timestamp_ms / 1000.0)
    
    log_info("✅ Historical warmup complete")
    
    # Start the engine
    log_info("Starting trading engine...")
    engine.start()
    
    # Live trading loop (test for 5 minutes)
    log_info("=" * 60)
    log_info("STARTING LIVE TRADING LOOP (5 minute test)")
    log_info("=" * 60)
    
    last_timestamp = 0
    test_duration = 300  # 5 minutes
    start_time = time.time()
    candle_count = 0
    
    try:
        while (time.time() - start_time) < test_duration:
            # Fetch latest candles using DataFetcher
            latest_data = fetcher.fetch_latest_candles(
                pair=pair,
                timeframe=timeframe,
                limit=2
            )
            
            if latest_data and len(latest_data) >= 2:
                candle = latest_data[-2]  # Get the completed candle
                timestamp_ms = candle[0]
                
                if timestamp_ms > last_timestamp:
                    last_timestamp = timestamp_ms
                    open_ = candle[1]
                    high = candle[2]
                    low = candle[3]
                    close = candle[4]
                    volume = candle[5]
                    
                    # Process the new candle
                    engine.process_candle(
                        open_, high, low, close, volume,
                        timestamp_ms / 1000.0
                    )
                    
                    candle_count += 1
                    
                    # Display real-time state
                    state = engine.get_current_state()
                    balance = state.get('balance', initial_balance)
                    pnl = balance - initial_balance
                    positions = state.get('positions', [])
                    timestamp = datetime.fromtimestamp(timestamp_ms / 1000.0).strftime('%H:%M:%S')
                    
                    log_info(
                        f"[{timestamp}] Price: ${close:.2f} | "
                        f"Balance: ${balance:.2f} | "
                        f"PnL: ${pnl:+.2f} | "
                        f"Positions: {len(positions)}"
                    )
            
            time.sleep(timeframe_seconds)
    
    except KeyboardInterrupt:
        log_info("Test interrupted by user")
    
    # Test results
    log_info("=" * 60)
    log_info("TEST RESULTS")
    log_info("=" * 60)
    log_info(f"Total candles processed: {candle_count}")
    log_info(f"Test duration: {time.time() - start_time:.1f} seconds")
    
    final_state = engine.get_current_state()
    final_balance = final_state.get('balance', initial_balance)
    final_pnl = final_balance - initial_balance
    final_positions = final_state.get('positions', [])
    
    log_info(f"Final balance: ${final_balance:.2f}")
    log_info(f"Final PnL: ${final_pnl:+.2f}")
    log_info(f"Final positions: {len(final_positions)}")
    
    # Cleanup
    cleanup_gpu(gpu_context, gpu_queue)
    
    log_info("✅ Mode 2 test complete!")
    return True

if __name__ == "__main__":
    try:
        success = test_mode2()
        sys.exit(0 if success else 1)
    except Exception as e:
        log_error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
