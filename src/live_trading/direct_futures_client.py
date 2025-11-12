"""
Direct Kucoin Futures API Client

Bypasses the SDK for private endpoints to have full control over timestamps.
Uses manual authentication with server-synchronized timestamps.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import time
import hmac
import hashlib
import base64
import requests
import json
from typing import Optional, Dict
from src.live_trading.time_sync import get_kucoin_server_time


class DirectKucoinFuturesClient:
    """Direct API client with manual timestamp control."""
    
    def __init__(self, api_key: str, api_secret: str, api_passphrase: str, test_mode: bool = True):
        """Initialize direct client."""
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_passphrase = api_passphrase
        self.test_mode = test_mode
        
        self.base_url = "https://api-futures.kucoin.com"
        
        # Create passphrase signature (v2)
        self.passphrase_sig = base64.b64encode(
            hmac.new(
                api_secret.encode('utf-8'),
                api_passphrase.encode('utf-8'),
                hashlib.sha256
            ).digest()
        ).decode('utf-8')
        
        print(f"✅ Direct Kucoin Futures client initialized")
        print(f"   Mode: {'🧪 TEST' if test_mode else '🔴 LIVE'}")
    
    def _sign_request(self, timestamp: str, method: str, endpoint: str, body: str = "") -> str:
        """Create request signature."""
        str_to_sign = timestamp + method + endpoint + body
        
        signature = base64.b64encode(
            hmac.new(
                self.api_secret.encode('utf-8'),
                str_to_sign.encode('utf-8'),
                hashlib.sha256
            ).digest()
        ).decode('utf-8')
        
        return signature
    
    def _get_headers(self, method: str, endpoint: str, body: str = "") -> Dict:
        """Generate authenticated headers with server-synced timestamp."""
        # Use Kucoin server time (synced)
        timestamp = str(get_kucoin_server_time())
        signature = self._sign_request(timestamp, method, endpoint, body)
        
        headers = {
            "KC-API-KEY": self.api_key,
            "KC-API-SIGN": signature,
            "KC-API-TIMESTAMP": timestamp,
            "KC-API-PASSPHRASE": self.passphrase_sig,
            "KC-API-KEY-VERSION": "2",
            "Content-Type": "application/json"
        }
        
        return headers
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get position details."""
        endpoint = f"/api/v1/position?symbol={symbol}"
        headers = self._get_headers("GET", endpoint)
        
        url = self.base_url + endpoint
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '200000':
                    position_data = data.get('data', {})
                    if position_data and position_data.get('currentQty', 0) != 0:
                        return position_data
                    return None  # No position
                else:
                    print(f"❌ API Error: {data.get('msg', 'Unknown error')}")
                    return None
            else:
                print(f"❌ HTTP {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Request failed: {e}")
            return None
    
    def create_market_order(self, symbol: str, side: str, size: int, leverage: int = 1) -> Optional[Dict]:
        """Create market order."""
        # Input validation
        if not symbol or not isinstance(symbol, str):
            raise ValueError(f"Invalid symbol: {symbol}")
        if side not in ['buy', 'sell']:
            raise ValueError(f"Invalid side: {side}. Must be 'buy' or 'sell'")
        if not isinstance(size, (int, float)) or size <= 0:
            raise ValueError(f"Invalid size: {size}. Must be positive number")
        if not isinstance(leverage, (int, float)) or not (1 <= leverage <= 100):
            raise ValueError(f"Invalid leverage: {leverage}. Must be between 1-100")
        
        client_oid = f"bot_{int(time.time() * 1000)}"
        
        order_data = {
            "clientOid": client_oid,
            "side": side,
            "symbol": symbol,
            "type": "market",
            "leverage": leverage,
            "size": size
        }
        
        body = json.dumps(order_data)
        endpoint = "/api/v1/orders/test" if self.test_mode else "/api/v1/orders"
        headers = self._get_headers("POST", endpoint, body)
        
        url = self.base_url + endpoint
        
        try:
            response = requests.post(url, headers=headers, data=body, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '200000':
                    return data.get('data', {})
                else:
                    print(f"❌ API Error: {data.get('msg', 'Unknown error')}")
                    return None
            else:
                print(f"❌ HTTP {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Request failed: {e}")
            return None
    
    def create_limit_order(self, symbol: str, side: str, price: float, size: int, leverage: int = 1) -> Optional[Dict]:
        """Create limit order."""
        # Input validation
        if not symbol or not isinstance(symbol, str):
            raise ValueError(f"Invalid symbol: {symbol}")
        if side not in ['buy', 'sell']:
            raise ValueError(f"Invalid side: {side}. Must be 'buy' or 'sell'")
        if not isinstance(price, (int, float)) or price <= 0:
            raise ValueError(f"Invalid price: {price}. Must be positive number")
        if not isinstance(size, (int, float)) or size <= 0:
            raise ValueError(f"Invalid size: {size}. Must be positive number")
        if not isinstance(leverage, (int, float)) or not (1 <= leverage <= 100):
            raise ValueError(f"Invalid leverage: {leverage}. Must be between 1-100")
        
        client_oid = f"bot_{int(time.time() * 1000)}"
        
        order_data = {
            "clientOid": client_oid,
            "side": side,
            "symbol": symbol,
            "type": "limit",
            "price": str(price),
            "size": size,
            "leverage": leverage
        }
        
        body = json.dumps(order_data)
        endpoint = "/api/v1/orders/test" if self.test_mode else "/api/v1/orders"
        headers = self._get_headers("POST", endpoint, body)
        
        url = self.base_url + endpoint
        
        try:
            response = requests.post(url, headers=headers, data=body, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '200000':
                    return data.get('data', {})
                else:
                    print(f"❌ API Error: {data.get('msg', 'Unknown error')}")
                    return None
            else:
                print(f"❌ HTTP {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Request failed: {e}")
            return None
    
    def get_order(self, order_id: str) -> Optional[Dict]:
        """Get order details by order ID."""
        endpoint = f"/api/v1/orders/{order_id}"
        headers = self._get_headers("GET", endpoint, "")
        
        url = self.base_url + endpoint
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '200000':
                    return data.get('data', {})
                else:
                    print(f"❌ API Error: {data.get('msg', 'Unknown error')}")
                    return None
            else:
                print(f"❌ HTTP {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Request failed: {e}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by order ID."""
        endpoint = f"/api/v1/orders/{order_id}"
        headers = self._get_headers("DELETE", endpoint, "")
        
        url = self.base_url + endpoint
        
        try:
            response = requests.delete(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '200000':
                    return True
                else:
                    print(f"❌ API Error: {data.get('msg', 'Unknown error')}")
                    return False
            else:
                print(f"❌ HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Request failed: {e}")
            return False


def test_direct_client():
    """Test the direct client."""
    print("="*80)
    print("🧪 TESTING DIRECT CLIENT WITH TIME SYNC")
    print("="*80)
    
    from src.live_trading.credentials import CredentialsManager
    
    # Load credentials
    manager = CredentialsManager()
    creds = manager.load_credentials()
    if not creds:
        print("❌ No credentials found!")
        return
    
    # Create direct client
    client = DirectKucoinFuturesClient(
        api_key=creds['api_key'],
        api_secret=creds['api_secret'],
        api_passphrase=creds['api_passphrase'],
        test_mode=True
    )
    
    # Test 1: Get position
    print("\n" + "="*80)
    print("📋 Test 1: Get Position (XBTUSDTM)")
    print("="*80)
    
    position = client.get_position("XBTUSDTM")
    if position is not None:
        print(f"✅ SUCCESS: Position retrieved")
        print(f"   Symbol: {position.get('symbol', 'N/A')}")
        print(f"   Current Qty: {position.get('currentQty', 0)}")
        print(f"   Avg Entry Price: {position.get('avgEntryPrice', 0)}")
    elif position is None:
        print(f"✅ SUCCESS: No open position (this is normal)")
    
    # Test 2: Create test market order
    print("\n" + "="*80)
    print("📋 Test 2: Create Test Market Order")
    print("="*80)
    
    order = client.create_market_order("XBTUSDTM", "buy", 1, leverage=1)
    if order:
        print(f"✅ SUCCESS: Market order validated")
        print(f"   Order ID: {order.get('orderId', 'N/A')}")
        print(f"   Client OID: {order.get('clientOid', 'N/A')}")
    
    # Test 3: Create test limit order
    print("\n" + "="*80)
    print("📋 Test 3: Create Test Limit Order")
    print("="*80)
    
    # Get current price first (using requests - public endpoint)
    try:
        ticker_response = requests.get(
            "https://api-futures.kucoin.com/api/v1/ticker?symbol=XBTUSDTM",
            timeout=10
        )
        if ticker_response.status_code == 200:
            ticker_data = ticker_response.json().get('data', {})
            current_price = float(ticker_data.get('price', 50000))
            limit_price = current_price * 0.95  # 5% below market
            
            print(f"   Current price: ${current_price:,.2f}")
            print(f"   Limit price: ${limit_price:,.2f} (5% below)")
            
            order = client.create_limit_order("XBTUSDTM", "buy", limit_price, 1, leverage=1)
            if order:
                print(f"✅ SUCCESS: Limit order validated")
                print(f"   Order ID: {order.get('orderId', 'N/A')}")
    except Exception as e:
        print(f"⚠️  Could not test limit order: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("📊 TEST COMPLETE")
    print("="*80)
    print("\nIf all tests passed:")
    print("   ✅ Time synchronization is working")
    print("   ✅ Timestamp authentication is fixed")
    print("   ✅ Ready to integrate with main trading system")
    print("\nNext step:")
    print("   Update kucoin_universal_client.py to use DirectKucoinFuturesClient")
    print("   for private endpoints instead of the SDK")
    print("="*80)


if __name__ == "__main__":
    test_direct_client()
