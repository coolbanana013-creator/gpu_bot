#!/usr/bin/env python3
# Moved verify_aroon_logic.py to scripts/debug
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
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
    # Manual comparison & TA-Lib
    aroon_down, aroon_up_talib = talib.AROON(highs, lows, timeperiod=period)
    print('TA-Lib Aroon sample:', aroon_up_talib[25])
    print('Manual checks...')
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
        print(b, 'manual', manual_aroon, 'talib', aroon_up_talib[b])
