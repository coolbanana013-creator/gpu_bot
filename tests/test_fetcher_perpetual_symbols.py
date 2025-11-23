import pytest
import os
import pyopencl as cl
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_provider.fetcher import DataFetcher


@pytest.mark.skipif(os.environ.get('RUN_INTEGRATION_TESTS') != '1', reason='Online integration tests disabled')
def test_fetcher_prefers_perpetual_symbol():
    # This test requires network access and will try to load kucoinfutures markets
    df = DataFetcher(exchange_type='futures', skip_load_markets=False)
    # Normalize a common BTC/USDT pair and ensure we return a perpetual swap market
    symbol = df._normalize_symbol('BTC/USDT')
    assert symbol is not None
    assert ':' in symbol or symbol.upper().endswith('M')
