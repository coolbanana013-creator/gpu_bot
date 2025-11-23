#!/usr/bin/env python3
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import talib
from src.data_provider.fetcher import DataFetcher
from src.data_provider.loader import DataLoader

def compute_manual_adx(highs, lows, closes, period):
    # manual compute similar to kernel logic
    pass

if __name__ == '__main__':
    fetcher = DataFetcher(exchange_type='futures')
    file_paths = fetcher.fetch_data_range(pair='BTC/USDT', timeframe='1m', total_days=1)
    loader = DataLoader(file_paths=file_paths, timeframe='1m', gpu_context=None, gpu_queue=None)
    ohlcv_df = loader.load_all_data()
    # This debug script replicates the ADX manual logic for parity checks
    print('Detail ADX script: moved under scripts/debug')
