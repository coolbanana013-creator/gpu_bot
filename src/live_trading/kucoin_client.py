"""
Kucoin Futures Integration

Handles connection to Kucoin Futures API for live data and trading.
"""

import ccxt
from typing import Optional, Dict, Callable
import time
import threading
from datetime import datetime
import requests

from ..utils.validation import log_info, log_error, log_warning
from .credentials import CredentialsManager


class KucoinFuturesClient:
    """Kucoin Futures API client wrapper with time synchronization."""
    
    def __init__(self, credentials: Dict[str, str], testnet: bool = False):
        """
        Initialize Kucoin client.
        
        Args:
            credentials: Dict with api_key, api_secret, api_passphrase
            testnet: Use testnet/sandbox
        """
        self.credentials = credentials
        self.testnet = testnet
        self.time_offset_ms = 0
        
        # Sync time with Kucoin server BEFORE creating exchange
        self._sync_server_time()
        
        # Initialize CCXT
        self.exchange = ccxt.kucoinfutures({
            'apiKey': credentials['api_key'],
            'secret': credentials['api_secret'],
            'password': credentials['api_passphrase'],
            'enableRateLimit': True
        })
        
        # Override the nonce function to use synchronized time
        original_nonce = self.exchange.nonce
        def synced_nonce():
            return int((time.time() * 1000) + self.time_offset_ms)
        self.exchange.nonce = synced_nonce
        
        if testnet:
            self.exchange.set_sandbox_mode(True)
            log_info("🧪 Kucoin TESTNET mode enabled")
        else:
            log_warning("⚠️  Kucoin LIVE mode - real money!")
        
        # Test connection
        try:
            self.exchange.load_markets()
            balance = self.exchange.fetch_balance()
            log_info(f"✅ Connected to Kucoin Futures")
            log_info(f"   Account Balance: ${balance['USDT']['total']:.2f} USDT")
        except Exception as e:
            log_error(f"❌ Failed to connect to Kucoin: {e}")
            raise
    
    def _sync_server_time(self):
        """Synchronize with Kucoin server time."""
        try:
            url = 'https://api-futures.kucoin.com/api/v1/timestamp'
            local_before = int(time.time() * 1000)
            response = requests.get(url, timeout=5)
            local_after = int(time.time() * 1000)
            
            if response.status_code == 200:
                server_time = response.json().get('data', 0)
                local_time = (local_before + local_after) // 2
                self.time_offset_ms = server_time - local_time
                log_info(f"⏰ Time synchronized: offset = {self.time_offset_ms} ms")
            else:
                log_warning(f"⚠️  Time sync failed: HTTP {response.status_code}")
                self.time_offset_ms = 0
        except Exception as e:
            log_warning(f"⚠️  Time sync error: {e}")
            self.time_offset_ms = 0
    
    def fetch_ticker(self, symbol: str = 'BTC/USDT:USDT') -> Dict:
        """
        Fetch current ticker data.
        
        Args:
            symbol: Trading pair symbol
        
        Returns:
            Dict with ticker data
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker
        except Exception as e:
            log_error(f"Failed to fetch ticker: {e}")
            return {}
    
    def fetch_ohlcv(self, symbol: str = 'BTC/USDT:USDT', timeframe: str = '1m', limit: int = 500) -> list:
        """
        Fetch OHLCV candle data.
        
        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            limit: Number of candles
        
        Returns:
            List of [timestamp, open, high, low, close, volume]
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            return ohlcv
        except Exception as e:
            log_error(f"Failed to fetch OHLCV: {e}")
            return []
    
    def create_market_order(self, symbol: str, side: str, size: float, leverage: int = 1) -> Dict:
        """
        Create market order (LIVE TRADING ONLY).
        
        Args:
            symbol: Trading pair
            side: 'buy' or 'sell'
            size: Order size in contracts
            leverage: Leverage
        
        Returns:
            Order info dict
        """
        try:
            # Set leverage first
            self.exchange.set_leverage(leverage, symbol)
            
            # Create order
            order = self.exchange.create_market_order(
                symbol=symbol,
                side=side,
                amount=size
            )
            
            log_info(f"✅ Order placed: {side.upper()} {size} {symbol} @ Market [Leverage: {leverage}x]")
            return order
            
        except Exception as e:
            log_error(f"❌ Failed to place order: {e}")
            return {}
    
    def get_position(self, symbol: str = 'BTC/USDT:USDT') -> Optional[Dict]:
        """Get current position for symbol."""
        try:
            positions = self.exchange.fetch_positions([symbol])
            if positions:
                return positions[0]
            return None
        except Exception as e:
            log_error(f"Failed to fetch position: {e}")
            return None
    
    def close_position(self, symbol: str, side: str, size: float) -> Dict:
        """Close position (opposite order with reduce_only)."""
        try:
            order = self.exchange.create_market_order(
                symbol=symbol,
                side=side,  # Opposite of position side
                amount=size,
                params={'reduceOnly': True}
            )
            
            log_info(f"✅ Position closed: {side.upper()} {size} {symbol}")
            return order
            
        except Exception as e:
            log_error(f"❌ Failed to close position: {e}")
            return {}


class LiveDataStreamer:
    """Stream live price data from Kucoin."""
    
    def __init__(self, client: KucoinFuturesClient, symbol: str = 'BTC/USDT:USDT', timeframe: str = '1m'):
        """
        Initialize data streamer.
        
        Args:
            client: Kucoin client
            symbol: Trading pair
            timeframe: Candle timeframe
        """
        self.client = client
        self.symbol = symbol
        self.timeframe = timeframe
        
        self.is_running = False
        self.thread = None
        self.callback = None
        
        self.last_candle_timestamp = 0
    
    def start(self, callback: Callable):
        """
        Start streaming live data.
        
        Args:
            callback: Function to call with new candle data
                     Signature: callback(open, high, low, close, volume, timestamp)
        """
        self.callback = callback
        self.is_running = True
        
        # Start streaming thread
        self.thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.thread.start()
        
        log_info(f"📡 Live data stream started: {self.symbol} {self.timeframe}")
    
    def stop(self):
        """Stop streaming."""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=5)
        log_info("📡 Live data stream stopped")
    
    def _stream_loop(self):
        """Main streaming loop."""
        while self.is_running:
            try:
                # Fetch latest candles
                candles = self.client.fetch_ohlcv(self.symbol, self.timeframe, limit=2)
                
                if candles and len(candles) >= 2:
                    # Get completed candle (second to last)
                    candle = candles[-2]
                    timestamp = candle[0]
                    
                    # Only process if it's a new candle
                    if timestamp > self.last_candle_timestamp:
                        self.last_candle_timestamp = timestamp
                        
                        open_ = candle[1]
                        high = candle[2]
                        low = candle[3]
                        close = candle[4]
                        volume = candle[5]
                        
                        # Call callback
                        if self.callback:
                            self.callback(open_, high, low, close, volume, timestamp / 1000.0)
                
                # Sleep until next candle (for 1m, sleep 60s)
                sleep_time = self._get_sleep_time()
                time.sleep(sleep_time)
                
            except Exception as e:
                log_error(f"Stream error: {e}")
                time.sleep(5)  # Wait before retry
    
    def _get_sleep_time(self) -> float:
        """Calculate sleep time based on timeframe."""
        timeframe_seconds = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '30m': 1800,
            '1h': 3600,
            '4h': 14400,
            '1d': 86400
        }
        return timeframe_seconds.get(self.timeframe, 60)
