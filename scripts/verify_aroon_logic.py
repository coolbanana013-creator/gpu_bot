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
    
    period = 25
    highs = ohlcv_df['high'].values.astype(np.float64)
    lows = ohlcv_df['low'].values.astype(np.float64)
    
    # Manual computation for bar 25
    bar = 25
    search_start = bar - period + 1  # 25 - 25 + 1 = 1
    search_end = bar  # 25
    
    print(f'Bar {bar}, period {period}')
    print(f'Search window: bars [{search_start}, {search_end}]')
    print(f'Highs in window: {highs[search_start:search_end+1]}')
    
    highest = highs[search_start]
    highest_idx = search_start
    for i in range(search_start + 1, search_end + 1):
        if highs[i] >= highest:
            highest = highs[i]
            highest_idx = i
    
    bars_since_high = bar - highest_idx
    aroon_up = ((period - bars_since_high) / period) * 100.0
    
    print(f'Highest: {highest} at bar {highest_idx}')
    print(f'Bars since highest: {bars_since_high}')
    print(f'Manual Aroon Up: {aroon_up}')
    
    # TA-Lib
    aroon_down, aroon_up_talib = talib.AROON(highs, lows, timeperiod=period)
    print(f'TA-Lib Aroon Up: {aroon_up_talib[bar]}')
    
    # Check several bars
    print(f'\nBar-by-bar comparison (25-35):')
    print(f'bar | manual | talib | diff')
    for b in range(25, 36):
        search_start = b - period + 1
        search_end = b
        
        highest = highs[search_start]
        highest_idx = search_start
        for i in range(search_start + 1, search_end + 1):
            if highs[i] >= highest:
                highest = highs[i]
                highest_idx = i
        
        bars_since_high = b - highest_idx
        manual_aroon = ((period - bars_since_high) / period) * 100.0
        talib_aroon = aroon_up_talib[b]
        
        diff = abs(manual_aroon - talib_aroon) if not np.isnan(talib_aroon) else np.nan
        print(f'{b:3d} | {manual_aroon:6.2f} | {talib_aroon:6.2f} | {diff:6.2f}')
