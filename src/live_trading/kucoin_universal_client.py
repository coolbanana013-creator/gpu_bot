"""
Kucoin Universal SDK Client Wrapper

Replaces CCXT with official Kucoin Universal SDK.
Supports both test orders (paper trading) and real orders (live trading).
"""

import logging
import os
import uuid
from typing import Optional, Dict, List
from datetime import datetime

from kucoin_universal_sdk.api import DefaultClient
from kucoin_universal_sdk.model import ClientOptionBuilder
from kucoin_universal_sdk.model import GLOBAL_API_ENDPOINT, GLOBAL_FUTURES_API_ENDPOINT
from kucoin_universal_sdk.model import TransportOptionBuilder
from kucoin_universal_sdk.generate.futures.order import (
    AddOrderReqBuilder,
    AddOrderReq,
    AddOrderTestReqBuilder,
    CancelOrderByIdReqBuilder,
    GetOrderByOrderIdReqBuilder
)
from kucoin_universal_sdk.generate.futures.market import (
    GetKlinesReqBuilder,
    GetKlinesReq,
    GetTickerReqBuilder
)
from kucoin_universal_sdk.generate.futures.positions import (
    GetPositionListReqBuilder,
    GetPositionDetailsReqBuilder
)

from ..utils.validation import log_info, log_error, log_warning
from .direct_futures_client import DirectKucoinFuturesClient
from .exceptions import (
    TradingError, OrderError, OrderCreationError, OrderCancellationError,
    NetworkError, RateLimitError, CircuitBreakerError
)
from .rate_limiter import rate_limit_order, rate_limit_general
from .circuit_breaker import order_circuit_breaker
from .enhanced_risk_manager import EnhancedRiskManager, RiskConfig


class KucoinUniversalClient:
    """
    Kucoin Universal SDK client for futures trading.
    
    Supports two modes:
    - Paper Trading: Uses /api/v1/orders/test endpoint (no real execution)
    - Live Trading: Uses /api/v1/orders endpoint (real money)
    """
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        api_passphrase: str,
        test_mode: bool = True
    ):
        """
        Initialize Kucoin client.
        
        Args:
            api_key: Kucoin API key
            api_secret: Kucoin API secret
            api_passphrase: Kucoin API passphrase
            test_mode: If True, uses test endpoint (paper trading)
        """
        self.test_mode = test_mode
        
        # Build HTTP transport options
        http_transport_option = (
            TransportOptionBuilder()
            .set_keep_alive(True)
            .set_max_pool_size(10)
            .set_max_connection_per_pool(10)
            .set_connect_timeout(10)
            .set_read_timeout(30)
            .set_max_retries(3)
            .build()
        )
        
        # Build client options
        client_option = (
            ClientOptionBuilder()
            .set_key(api_key)
            .set_secret(api_secret)
            .set_passphrase(api_passphrase)
            .set_spot_endpoint(GLOBAL_API_ENDPOINT)
            .set_futures_endpoint(GLOBAL_FUTURES_API_ENDPOINT)
            .set_transport_option(http_transport_option)
            .build()
        )
        
        # Create client
        self.client = DefaultClient(client_option)
        self.rest_service = self.client.rest_service()
        self.futures_service = self.rest_service.get_futures_service()
        
        # Get API instances (SDK - for public endpoints)
        self.order_api = self.futures_service.get_order_api()
        self.market_api = self.futures_service.get_market_api()
        self.position_api = self.futures_service.get_positions_api()
        
        # Direct client (for private endpoints with timestamp sync)
        self.direct_client = DirectKucoinFuturesClient(
            api_key=api_key,
            api_secret=api_secret,
            api_passphrase=api_passphrase,
            test_mode=test_mode
        )
        
        # Initialize risk manager with appropriate limits
        # Use more conservative limits for live trading, relaxed for paper trading
        risk_config = RiskConfig()
        if not test_mode:
            # Live trading: strict limits
            risk_config.max_position_size_btc = 10.0
            risk_config.max_position_size_eth = 100.0
            risk_config.max_leverage = 3
            risk_config.daily_loss_limit_usd = 500.0
            risk_config.max_daily_trades = 50
        else:
            # Paper trading: relaxed limits
            risk_config.max_position_size_btc = 100.0
            risk_config.max_position_size_eth = 1000.0
            risk_config.max_leverage = 10
            risk_config.daily_loss_limit_usd = 10000.0
            risk_config.max_daily_trades = 500
        
        self.risk_manager = EnhancedRiskManager(config=risk_config)
        
        mode_str = "🧪 TEST MODE (Paper Trading)" if test_mode else "💰 LIVE MODE (Real Money)"
        log_info(f"✅ Kucoin client initialized - {mode_str}")
        log_info(f"🛡️ Risk manager initialized - Max leverage: {risk_config.max_leverage}x, Daily loss limit: ${risk_config.daily_loss_limit_usd}")
    
    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """
        Set leverage for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'XBTUSDTM')
            leverage: Leverage (1-125x)
        
        Returns:
            True if successful
        """
        try:
            # Kucoin Universal SDK leverage setting
            # Note: Leverage is set per-position, not globally
            # We'll set it when opening positions
            return True
        except Exception as e:
            log_error(f"Failed to set leverage: {e}")
            return False
    
    @rate_limit_order
    def create_market_order(
        self,
        symbol: str,
        side: str,
        size: int,
        leverage: int = 1,
        margin_mode: str = "ISOLATED",
        reduce_only: bool = False
    ) -> Optional[Dict]:
        """
        Create market order (or test order in paper trading).
        
        Args:
            symbol: Trading symbol (e.g., 'XBTUSDTM')
            side: 'buy' or 'sell'
            size: Order size in contracts
            leverage: Leverage (1-125x)
            margin_mode: 'ISOLATED' or 'CROSS'
            reduce_only: Only reduce position, don't open new one
        
        Returns:
            Order info dict or None if failed
        """
        try:
            # Pre-order risk check (comprehensive validation)
            current_position = self.get_position(symbol)
            self.risk_manager.pre_order_check(
                symbol=symbol,
                side=side,
                size=size,
                leverage=leverage,
                position=current_position
            )
            
            # Use circuit breaker to protect against cascading failures
            def place_order():
                return self.direct_client.create_market_order(
                    symbol=symbol,
                    side=side,
                    size=size,
                    leverage=leverage
                )
            
            response = order_circuit_breaker.call(place_order)
            
            if response:
                # Record trade for daily limits tracking
                self.risk_manager.record_trade()
                
                # Update position tracking
                updated_position = self.get_position(symbol)
                if updated_position:
                    self.risk_manager.update_position(symbol, updated_position)
                
                log_info(f"🧪 TEST ORDER: {side.upper()} {size} {symbol} @ Market [Leverage: {leverage}x]" if self.test_mode else f"💰 LIVE ORDER: {side.upper()} {size} {symbol} @ Market [Leverage: {leverage}x]")
                
                return {
                    'orderId': response.get('orderId'),
                    'clientOid': response.get('clientOid'),
                    'symbol': symbol,
                    'side': side,
                    'size': size,
                    'leverage': leverage
                }
            else:
                raise OrderCreationError("Order placement returned no response")
                
        except CircuitBreakerError as e:
            log_error(f"🚫 Circuit breaker OPEN - order blocked: {e}")
            raise
        except RateLimitError as e:
            log_error(f"⏱️ Rate limit exceeded: {e}")
            raise
        except OrderCreationError as e:
            log_error(f"❌ Order creation failed: {e}")
            raise
        except Exception as e:
            log_error(f"❌ Failed to place order: {e}")
            raise OrderError(f"Unexpected error placing market order: {e}")
    
    @rate_limit_order
    def create_limit_order(
        self,
        symbol: str,
        side: str,
        price: float,
        size: int,
        leverage: int = 1,
        margin_mode: str = "ISOLATED",
        reduce_only: bool = False,
        time_in_force: str = "GTC"
    ) -> Optional[Dict]:
        """
        Create limit order (or test order in paper trading).
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            price: Limit price
            size: Order size in contracts
            leverage: Leverage (1-125x)
            margin_mode: 'ISOLATED' or 'CROSS'
            reduce_only: Only reduce position
            time_in_force: 'GTC', 'IOC', 'FOK'
        
        Returns:
            Order info dict or None if failed
        """
        try:
            # Round price to 0.1 (Kucoin requirement)
            price = round(price, 1)
            
            # Pre-order risk check (comprehensive validation)
            current_position = self.get_position(symbol)
            self.risk_manager.pre_order_check(
                symbol=symbol,
                side=side,
                size=size,
                leverage=leverage,
                position=current_position
            )
            
            # Use circuit breaker to protect against cascading failures
            def place_order():
                return self.direct_client.create_limit_order(
                    symbol=symbol,
                    side=side,
                    price=price,
                    size=size,
                    leverage=leverage
                )
            
            response = order_circuit_breaker.call(place_order)
            
            if response:
                # Record trade for daily limits tracking
                self.risk_manager.record_trade()
                
                # Update position tracking
                updated_position = self.get_position(symbol)
                if updated_position:
                    self.risk_manager.update_position(symbol, updated_position)
                
                log_info(f"🧪 TEST ORDER: {side.upper()} {size} {symbol} @ ${price} [Leverage: {leverage}x]" if self.test_mode else f"💰 LIVE ORDER: {side.upper()} {size} {symbol} @ ${price} [Leverage: {leverage}x]")
                
                return {
                    'orderId': response.get('orderId'),
                    'clientOid': response.get('clientOid'),
                    'symbol': symbol,
                    'side': side,
                    'price': price,
                    'size': size,
                    'leverage': leverage
                }
            else:
                raise OrderCreationError("Order placement returned no response")
            
        except CircuitBreakerError as e:
            log_error(f"🚫 Circuit breaker OPEN - order blocked: {e}")
            raise
        except RateLimitError as e:
            log_error(f"⏱️ Rate limit exceeded: {e}")
            raise
        except OrderCreationError as e:
            log_error(f"❌ Order creation failed: {e}")
            raise
        except Exception as e:
            log_error(f"❌ Failed to place limit order: {e}")
            raise OrderError(f"Unexpected error placing limit order: {e}")
    
    @rate_limit_general
    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an order by order ID."""
        try:
            # Use direct client to avoid SDK method signature issues
            success = self.direct_client.cancel_order(order_id)
            
            if success:
                log_info(f"✅ Order cancelled: {order_id}")
                return True
            else:
                raise OrderCancellationError(f"Failed to cancel order: {order_id}", order_id=order_id)
            
        except OrderCancellationError as e:
            log_error(f"❌ {e}")
            raise
        except Exception as e:
            log_error(f"❌ Failed to cancel order: {e}")
            raise OrderCancellationError(f"Unexpected error cancelling order: {e}", order_id=order_id)
    
    @rate_limit_general
    def get_order(self, symbol: str, order_id: str) -> Optional[Dict]:
        """Get order details by order ID."""
        try:
            # Use direct client to avoid SDK method signature issues
            response = self.direct_client.get_order(order_id)
            
            if not response:
                return None
            
            return {
                'order_id': response.get('id'),
                'symbol': response.get('symbol'),
                'side': response.get('side'),
                'price': float(response.get('price', 0)) if response.get('price') else None,
                'size': response.get('size'),
                'filled_size': response.get('filledSize', 0),
                'status': response.get('status')
            }
            
        except Exception as e:
            log_error(f"❌ Failed to get order: {e}")
            raise NetworkError(f"Failed to retrieve order details: {e}")
    
    @rate_limit_general
    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get current position for symbol using direct client."""
        try:
            # Use direct client to avoid timestamp issues
            position_data = self.direct_client.get_position(symbol)
            
            if not position_data:
                # No position found - return empty dict with symbol
                return {'symbol': symbol, 'currentQty': 0}
            
            position = {
                'symbol': position_data.get('symbol'),
                'side': 'long' if position_data.get('currentQty', 0) > 0 else 'short',
                'size': abs(position_data.get('currentQty', 0)),
                'currentQty': position_data.get('currentQty', 0),
                'entry_price': float(position_data.get('avgEntryPrice', 0)),
                'leverage': position_data.get('realLeverage', 1),
                'unrealized_pnl': float(position_data.get('unrealisedPnl', 0)),
                'margin': float(position_data.get('positionMargin', 0)),
                'liquidation_price': float(position_data.get('liquidationPrice', 0)) if position_data.get('liquidationPrice') else None
            }
            
            # Update risk manager position tracking
            self.risk_manager.update_position(symbol, position)
            
            return position
            
        except Exception as e:
            log_error(f"❌ Failed to get position: {e}")
            return {'symbol': symbol, 'currentQty': 0, 'error': str(e)}
    
    @rate_limit_general
    def fetch_ticker(self, symbol: str) -> Optional[Dict]:
        """Fetch current ticker data."""
        try:
            # Build request
            req = GetTickerReqBuilder().set_symbol(symbol).build()
            ticker = self.market_api.get_ticker(req)
            
            return {
                'symbol': symbol,
                'last': float(ticker.price) if ticker.price else None,
                'bid': float(ticker.best_bid_price) if hasattr(ticker, 'best_bid_price') and ticker.best_bid_price else None,
                'ask': float(ticker.best_ask_price) if hasattr(ticker, 'best_ask_price') and ticker.best_ask_price else None,
                'volume': float(ticker.volume_24h) if hasattr(ticker, 'volume_24h') and ticker.volume_24h else None
            }
        except Exception as e:
            log_error(f"❌ Failed to fetch ticker: {e}")
            raise NetworkError(f"Failed to fetch ticker data: {e}")
    
    @rate_limit_general
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = '1m',
        limit: int = 500
    ) -> List[List]:
        """
        Fetch OHLCV candle data.
        
        Args:
            symbol: Trading symbol
            timeframe: '1m', '5m', '15m', '30m', '1h', '4h', '1d'
            limit: Number of candles
        
        Returns:
            List of [timestamp, open, high, low, close, volume]
        """
        try:
            # Map timeframe to Kucoin granularity
            granularity_map = {
                '1m': GetKlinesReq.GranularityEnum.T_1,
                '5m': GetKlinesReq.GranularityEnum.T_5,
                '15m': GetKlinesReq.GranularityEnum.T_15,
                '30m': GetKlinesReq.GranularityEnum.T_30,
                '1h': GetKlinesReq.GranularityEnum.T_60,
                '4h': GetKlinesReq.GranularityEnum.T_240,
                '1d': GetKlinesReq.GranularityEnum.T_1440
            }
            
            granularity = granularity_map.get(timeframe, GetKlinesReq.GranularityEnum.T_1)
            
            klines_req = (
                GetKlinesReqBuilder()
                .set_symbol(symbol)
                .set_granularity(granularity)
                .build()
            )
            
            response = self.market_api.get_klines(klines_req)
            
            # Convert to standard format: [timestamp, open, high, low, close, volume]
            candles = []
            for row in response.data[:limit]:
                candles.append([
                    row[0],  # timestamp (ms)
                    float(row[1]),  # open
                    float(row[2]),  # high
                    float(row[3]),  # low
                    float(row[4]),  # close
                    float(row[5])   # volume
                ])
            
            return candles
            
        except Exception as e:
            log_error(f"❌ Failed to fetch OHLCV: {e}")
            raise NetworkError(f"Failed to fetch OHLCV data: {e}")
