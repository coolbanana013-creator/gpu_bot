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
    
    # TA-Lib ADX Algorithm (correct):
    # 1. Compute TR, +DM, -DM for each bar
    # 2. Smooth TR, +DM, -DM using Wilder's method
    # 3. Compute +DI, -DI from smoothed values
    # 4. Compute DX from +DI, -DI
    # 5. Take SIMPLE AVERAGE of first 'period' DX values to initialize ADX
    # 6. Then smooth ADX using Wilder's method
    
    # Step 1 & 2: Initial smoothing (sum of first period values)
    smoothed_tr = 0.0
    smoothed_plus_dm = 0.0
    smoothed_minus_dm = 0.0
    
    for i in range(1, period + 1):
        tr = true_range(highs[i], lows[i], closes[i-1])
        plus_dm, minus_dm = directional_movement(highs[i], lows[i], highs[i-1], lows[i-1])
        smoothed_tr += tr
        smoothed_plus_dm += plus_dm
        smoothed_minus_dm += minus_dm
    
    # Step 3 & 4: Compute DX values for period bars (bar period to period*2-1)
    dx_values = []
    
    for bar in range(period, period * 2):
        if bar > period:
            # Update smoothed values
            tr = true_range(highs[bar], lows[bar], closes[bar-1])
            plus_dm, minus_dm = directional_movement(highs[bar], lows[bar], highs[bar-1], lows[bar-1])
            
            smoothed_tr = smoothed_tr - (smoothed_tr / period) + tr
            smoothed_plus_dm = smoothed_plus_dm - (smoothed_plus_dm / period) + plus_dm
            smoothed_minus_dm = smoothed_minus_dm - (smoothed_minus_dm / period) + minus_dm
        
        # Compute DI and DX
        plus_di = (smoothed_plus_dm / smoothed_tr) * 100.0 if smoothed_tr > 0 else 0.0
        minus_di = (smoothed_minus_dm / smoothed_tr) * 100.0 if smoothed_tr > 0 else 0.0
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100.0 if (plus_di + minus_di) > 0 else 0.0
        
        dx_values.append(dx)
        print(f'Bar {bar}: DX={dx:.4f}')
    
    # Step 5: Initialize ADX as simple average of first 'period' DX values
    adx = sum(dx_values) / period
    print(f'\nInitial ADX (SMA of {period} DX values): {adx:.6f}')
    print(f'TA-Lib ADX at bar {period*2-1}: {talib.ADX(highs, lows, closes, timeperiod=period)[period*2-1]:.6f}')
    print(f'Difference: {abs(adx - talib.ADX(highs, lows, closes, timeperiod=period)[period*2-1]):.6f}')
