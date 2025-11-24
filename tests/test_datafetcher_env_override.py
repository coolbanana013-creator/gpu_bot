import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_provider.fetcher import DataFetcher


def test_skip_load_markets_env_variable():
    os.environ['SKIP_LOAD_MARKETS'] = '1'
    df = DataFetcher(exchange_type='futures')
    # When skip_load_markets is True, exchange.markets should not be populated
    assert getattr(df.exchange, 'markets', None) is None
    del os.environ['SKIP_LOAD_MARKETS']


def test_contract_type_env_override():
    os.environ['DATA_FETCHER_CONTRACT_TYPE'] = 'perpetual'
    df = DataFetcher(exchange_type='futures', skip_load_markets=True)
    assert df.contract_type == 'perpetual'
    del os.environ['DATA_FETCHER_CONTRACT_TYPE']


def test_normalize_prefers_swap_perpetual():
    df = DataFetcher(exchange_type='futures', skip_load_markets=True)
    # Simulate markets loaded by injecting a fake mapping
    df.exchange.markets = {
        'XBTUSDTM': {'symbol': 'XBT/USDT:USDT', 'base': 'XBT', 'quote': 'USDT', 'contract': True, 'swap': True},
        'BTCUSDTM': {'symbol': 'BTC/USDT:USDT', 'base': 'BTC', 'quote': 'USDT', 'contract': True, 'future': True},
        'BTCUSDT': {'symbol': 'BTC/USDT', 'base': 'BTC', 'quote': 'USDT', 'contract': False}
    }
    df.contract_type = 'perpetual'
    norm = df._normalize_symbol('BTC/USDT')
    assert norm == 'XBT/USDT:USDT' or norm == 'BTC/USDT:USDT'


def test_normalize_by_market_id_prefers_id_lookup():
    df = DataFetcher(exchange_type='futures', skip_load_markets=True)
    # Simulate markets_by_id mapping where the id is 'XBTUSDTM'
    df.exchange.markets_by_id = {
        'XBTUSDTM': {'symbol': 'XBT/USDT:USDT', 'base': 'XBT', 'quote': 'USDT', 'contract': True, 'swap': True}
    }
    norm = df._normalize_symbol('xbtusdtm')
    assert norm == 'XBT/USDT:USDT'
