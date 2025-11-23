#!/usr/bin/env python3
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_provider.fetcher import DataFetcher
from src.data_provider.loader import DataLoader
import talib

if __name__ == '__main__':
    fetcher = DataFetcher(exchange_type='futures')
    file_paths = fetcher.fetch_data_range(pair='BTC/USDT', timeframe='1m', total_days=1)
    loader = DataLoader(file_paths=file_paths, timeframe='1m', gpu_context=None, gpu_queue=None)
    ohlcv_df = loader.load_all_data()
    
    period = 14
    num_bars = len(ohlcv_df)
    
    # Get numpy arrays
    highs = ohlcv_df['high'].values.astype(np.float64)
    lows = ohlcv_df['low'].values.astype(np.float64)
    closes = ohlcv_df['close'].values.astype(np.float64)
    
    # Manually compute ADX using Wilder's method to match GPU kernel logic
    def true_range(high, low, prev_close):
        return max(high - low, abs(high - prev_close), abs(low - prev_close))
    
    def directional_movement(high, low, prev_high, prev_low):
        up_move = high - prev_high
        down_move = prev_low - low
        
        if up_move > down_move and up_move > 0:
            plus_dm = up_move
        else:
            plus_dm = 0.0
            
        if down_move > up_move and down_move > 0:
            minus_dm = down_move
        else:
            minus_dm = 0.0
            
        return plus_dm, minus_dm
    
    # Test 1: Initial smoothing (sum of first period values) - starting from bar 1
    smoothed_tr = 0.0
    smoothed_plus_dm = 0.0
    smoothed_minus_dm = 0.0
    
    for i in range(1, period + 1):
        tr = true_range(highs[i], lows[i], closes[i-1])
        plus_dm, minus_dm = directional_movement(highs[i], lows[i], highs[i-1], lows[i-1])
        smoothed_tr += tr
        smoothed_plus_dm += plus_dm
        smoothed_minus_dm += minus_dm
    
    print('=== METHOD 1: Sum bars 1 to period (1-14) ===')
    print(f'Initial smoothed values:')
    print(f'  smoothed_tr = {smoothed_tr:.6f}')
    print(f'  smoothed_plus_dm = {smoothed_plus_dm:.6f}')
    print(f'  smoothed_minus_dm = {smoothed_minus_dm:.6f}')
    
    # Try alternative: Skip first bar, take average
    smoothed_tr_v2 = 0.0
    smoothed_plus_dm_v2 = 0.0
    smoothed_minus_dm_v2 = 0.0
    
    for i in range(1, period + 1):
        tr = true_range(highs[i], lows[i], closes[i-1])
        plus_dm, minus_dm = directional_movement(highs[i], lows[i], highs[i-1], lows[i-1])
        smoothed_tr_v2 += tr
        smoothed_plus_dm_v2 += plus_dm
        smoothed_minus_dm_v2 += minus_dm
    
    # Divide by period for initial average
    smoothed_tr_v2 /= period
    smoothed_plus_dm_v2 /= period  
    smoothed_minus_dm_v2 /= period
    
    print('\n=== METHOD 2: Average bars 1 to period (1-14) ===')
    print(f'Initial smoothed values (averaged):')
    print(f'  smoothed_tr = {smoothed_tr_v2:.6f}')
    print(f'  smoothed_plus_dm = {smoothed_plus_dm_v2:.6f}')
    print(f'  smoothed_minus_dm = {smoothed_minus_dm_v2:.6f}')
    
    # Now compute using Method 2 (averaged init)
    # Reset to use averaged version
    smoothed_tr = smoothed_tr_v2
    smoothed_plus_dm = smoothed_plus_dm_v2
    smoothed_minus_dm = smoothed_minus_dm_v2
    
    # At bar 14, compute first DX
    plus_di = (smoothed_plus_dm / smoothed_tr) * 100.0 if smoothed_tr > 0 else 0.0
    minus_di = (smoothed_minus_dm / smoothed_tr) * 100.0 if smoothed_tr > 0 else 0.0
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100.0 if (plus_di + minus_di) > 0 else 0.0
    
    print(f'\nAt bar {period} (using averaged init):')
    print(f'  +DI = {plus_di:.6f}')
    print(f'  -DI = {minus_di:.6f}')
    print(f'  DX = {dx:.6f}')
    
    # Initialize ADX
    prev_adx = dx
    print(f'  Initial ADX = {prev_adx:.6f}')
    
    # Continue for bars period+1 to period*2-1
    print(f'\nBar-by-bar ADX computation (bars {period+1} to {period*2-1}):')
    for bar in range(period + 1, period * 2):
        # Wilder's smoothing
        tr = true_range(highs[bar], lows[bar], closes[bar-1])
        plus_dm, minus_dm = directional_movement(highs[bar], lows[bar], highs[bar-1], lows[bar-1])
        
        smoothed_tr = smoothed_tr - (smoothed_tr / period) + tr
        smoothed_plus_dm = smoothed_plus_dm - (smoothed_plus_dm / period) + plus_dm
        smoothed_minus_dm = smoothed_minus_dm - (smoothed_minus_dm / period) + minus_dm
        
        plus_di = (smoothed_plus_dm / smoothed_tr) * 100.0 if smoothed_tr > 0 else 0.0
        minus_di = (smoothed_minus_dm / smoothed_tr) * 100.0 if smoothed_tr > 0 else 0.0
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100.0 if (plus_di + minus_di) > 0 else 0.0
        
        prev_adx = (prev_adx * (period - 1) + dx) / period
        
        print(f'  Bar {bar}: DX={dx:.4f}, ADX={prev_adx:.4f}')
    
    # Output at bar period*2-1
    manual_adx = prev_adx
    print(f'\n=== Manual ADX (averaged init) at bar {period*2-1}: {manual_adx:.6f}')
    
    # Compare with TA-Lib
    talib_adx = talib.ADX(highs, lows, closes, timeperiod=period)
    print(f'=== TA-Lib ADX at bar {period*2-1}: {talib_adx[period*2-1]:.6f}')
    print(f'=== Difference: {abs(manual_adx - talib_adx[period*2-1]):.6f}')
