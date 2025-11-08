#!/usr/bin/env python3
"""
Quick test script to identify generation bottleneck
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from main import run_mode1, initialize_gpu
import numpy as np

# Initialize GPU
gpu_context, gpu_queue, gpu_info = initialize_gpu()

# Run with minimal parameters to identify bottleneck
params = {
    'pair': 'XBTUSDTM',
    'population_size': 100,
    'generations': 2,
    'cycles': 1,
    'backtest_days': 1,  # Reduced to work with available data
    'timeframe': '1m',
    'min_leverage': 1,
    'max_leverage': 3,
    'min_indicators': 1,
    'max_indicators': 2,
    'min_risk_strategies': 1,
    'max_risk_strategies': 2,
    'random_seed': 42
}

print("Starting minimal GA test to identify generation bottleneck...")
run_mode1(params, gpu_context, gpu_queue, gpu_info)