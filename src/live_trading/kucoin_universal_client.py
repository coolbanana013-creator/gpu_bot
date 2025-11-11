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
    CancelOrderByOrderIdReqBuilder,
    GetOrderByOrderIdReqBuilder
)
from kucoin_universal_sdk.generate.futures.market import (
    GetKlinesReqBuilder,
    GetKlinesReq,
    GetAllSymbolsReq
)
from kucoin_universal_sdk.generate.futures.position import (
    GetPositionListReqBuilder,
    GetPositionDetailsReqBuilder
)

from ..utils.validation import log_info, log_error, log_warning


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
        
        # Get API instances
        self.order_api = self.futures_service.get_order_api()
        self.market_api = self.futures_service.get_market_api()
        self.position_api = self.futures_service.get_position_api()
        
        mode_str = "🧪 TEST MODE (Paper Trading)" if test_mode else "💰 LIVE MODE (Real Money)"
        log_info(f"✅ Kucoin Universal SDK initialized - {mode_str}")
        
        # Test connection
        try:
            symbols = self.market_api.get_all_symbols(GetAllSymbolsReq())
            log_info(f"   Connected to Kucoin Futures - {len(symbols.data)} contracts available")
        except Exception as e:
            log_error(f"❌ Failed to connect to Kucoin: {e}")
            raise
    
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
            client_oid = str(uuid.uuid4())
            
            # Build order request
            if self.test_mode:
                # PAPER TRADING: Use test endpoint
                order_req = (
                    AddOrderTestReqBuilder()
                    .set_client_oid(client_oid)
                    .set_symbol(symbol)
                    .set_side(AddOrderReq.SideEnum.BUY if side == 'buy' else AddOrderReq.SideEnum.SELL)
                    .set_type(AddOrderReq.TypeEnum.MARKET)
                    .set_size(size)
                    .set_leverage(str(leverage))
                    .set_margin_mode(margin_mode)
                    .set_reduce_only(reduce_only)
                    .build()
                )
                
                response = self.order_api.add_order_test(order_req)
                log_info(f"🧪 TEST ORDER: {side.upper()} {size} {symbol} @ Market [Leverage: {leverage}x]")
                
            else:
                # LIVE TRADING: Use real endpoint
                order_req = (
                    AddOrderReqBuilder()
                    .set_client_oid(client_oid)
                    .set_symbol(symbol)
                    .set_side(AddOrderReq.SideEnum.BUY if side == 'buy' else AddOrderReq.SideEnum.SELL)
                    .set_type(AddOrderReq.TypeEnum.MARKET)
                    .set_size(size)
                    .set_leverage(str(leverage))
                    .set_margin_mode(margin_mode)
                    .set_reduce_only(reduce_only)
                    .build()
                )
                
                response = self.order_api.add_order(order_req)
                log_info(f"✅ REAL ORDER: {side.upper()} {size} {symbol} @ Market [Leverage: {leverage}x]")
            
            return {
                'order_id': response.order_id,
                'client_oid': response.client_oid,
                'symbol': symbol,
                'side': side,
                'size': size,
                'leverage': leverage
            }
            
        except Exception as e:
            log_error(f"❌ Failed to place order: {e}")
            return None
    
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
            client_oid = str(uuid.uuid4())
            
            if self.test_mode:
                # PAPER TRADING
                order_req = (
                    AddOrderTestReqBuilder()
                    .set_client_oid(client_oid)
                    .set_symbol(symbol)
                    .set_side(AddOrderReq.SideEnum.BUY if side == 'buy' else AddOrderReq.SideEnum.SELL)
                    .set_type(AddOrderReq.TypeEnum.LIMIT)
                    .set_price(str(price))
                    .set_size(size)
                    .set_leverage(str(leverage))
                    .set_margin_mode(margin_mode)
                    .set_reduce_only(reduce_only)
                    .set_time_in_force(time_in_force)
                    .build()
                )
                
                response = self.order_api.add_order_test(order_req)
                log_info(f"🧪 TEST ORDER: {side.upper()} {size} {symbol} @ ${price} [Leverage: {leverage}x]")
                
            else:
                # LIVE TRADING
                order_req = (
                    AddOrderReqBuilder()
                    .set_client_oid(client_oid)
                    .set_symbol(symbol)
                    .set_side(AddOrderReq.SideEnum.BUY if side == 'buy' else AddOrderReq.SideEnum.SELL)
                    .set_type(AddOrderReq.TypeEnum.LIMIT)
                    .set_price(str(price))
                    .set_size(size)
                    .set_leverage(str(leverage))
                    .set_margin_mode(margin_mode)
                    .set_reduce_only(reduce_only)
                    .set_time_in_force(time_in_force)
                    .build()
                )
                
                response = self.order_api.add_order(order_req)
                log_info(f"✅ REAL ORDER: {side.upper()} {size} {symbol} @ ${price} [Leverage: {leverage}x]")
            
            return {
                'order_id': response.order_id,
                'client_oid': response.client_oid,
                'symbol': symbol,
                'side': side,
                'price': price,
                'size': size,
                'leverage': leverage
            }
            
        except Exception as e:
            log_error(f"❌ Failed to place limit order: {e}")
            return None
    
    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an order by order ID."""
        try:
            cancel_req = (
                CancelOrderByOrderIdReqBuilder()
                .set_order_id(order_id)
                .set_symbol(symbol)
                .build()
            )
            
            response = self.order_api.cancel_order_by_order_id(cancel_req)
            log_info(f"✅ Order cancelled: {order_id}")
            return True
            
        except Exception as e:
            log_error(f"❌ Failed to cancel order: {e}")
            return False
    
    def get_order(self, symbol: str, order_id: str) -> Optional[Dict]:
        """Get order details by order ID."""
        try:
            order_req = (
                GetOrderByOrderIdReqBuilder()
                .set_order_id(order_id)
                .set_symbol(symbol)
                .build()
            )
            
            response = self.order_api.get_order_by_order_id(order_req)
            return {
                'order_id': response.order_id,
                'symbol': response.symbol,
                'side': response.side,
                'price': float(response.price) if response.price else None,
                'size': response.size,
                'filled_size': response.filled_size,
                'status': response.status
            }
            
        except Exception as e:
            log_error(f"❌ Failed to get order: {e}")
            return None
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get current position for symbol."""
        try:
            position_req = (
                GetPositionDetailsReqBuilder()
                .set_symbol(symbol)
                .build()
            )
            
            response = self.position_api.get_position_details(position_req)
            
            if not response or not response.id:
                return None
            
            return {
                'symbol': response.symbol,
                'side': 'long' if response.current_qty > 0 else 'short',
                'size': abs(response.current_qty),
                'entry_price': float(response.avg_entry_price),
                'leverage': response.real_leverage,
                'unrealized_pnl': float(response.unrealised_pnl),
                'margin': float(response.position_margin),
                'liquidation_price': float(response.liquidation_price) if response.liquidation_price else None
            }
            
        except Exception as e:
            log_error(f"❌ Failed to get position: {e}")
            return None
    
    def fetch_ticker(self, symbol: str) -> Optional[Dict]:
        """Fetch current ticker data."""
        try:
            # Use market API to get ticker
            ticker = self.market_api.get_ticker(symbol)
            return {
                'symbol': symbol,
                'last': float(ticker.price),
                'bid': float(ticker.best_bid_price) if ticker.best_bid_price else None,
                'ask': float(ticker.best_ask_price) if ticker.best_ask_price else None,
                'volume': float(ticker.volume_24h) if ticker.volume_24h else None
            }
        except Exception as e:
            log_error(f"❌ Failed to fetch ticker: {e}")
            return None
    
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
            return []
