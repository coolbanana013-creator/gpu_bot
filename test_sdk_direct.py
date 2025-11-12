"""
Direct Kucoin SDK Test - Bypass Wrapper

This tests the SDK directly to see exactly what errors we're getting.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.live_trading.credentials import CredentialsManager
from kucoin_universal_sdk.generate.futures.market import GetTickerReqBuilder
from kucoin_universal_sdk.generate.futures.positions import GetPositionDetailsReqBuilder
from kucoin_universal_sdk.generate.futures.order import AddOrderReqBuilder
from kucoin_universal_sdk.model import ClientOptionBuilder
from kucoin_universal_sdk.model import GLOBAL_API_ENDPOINT, GLOBAL_FUTURES_API_ENDPOINT
from kucoin_universal_sdk.model import TransportOptionBuilder
from kucoin_universal_sdk.api import DefaultClient


def test_direct_sdk():
    """Test SDK directly without wrapper."""
    print("="*80)
    print("🔬 DIRECT SDK TEST (NO WRAPPER)")
    print("="*80)
    
    # Load credentials
    print("\n1. Loading credentials...")
    manager = CredentialsManager()
    creds = manager.load_credentials()
    if not creds:
        print("❌ No credentials found!")
        return
    
    print("✅ Credentials loaded")
    
    # Initialize SDK directly
    print("\n2. Initializing SDK...")
    
    http_transport_option = (
        TransportOptionBuilder()
        .set_keep_alive(True)
        .build()
    )
    
    client_option = (
        ClientOptionBuilder()
        .set_key(creds['api_key'])
        .set_secret(creds['api_secret'])
        .set_passphrase(creds['api_passphrase'])
        .set_spot_endpoint(GLOBAL_API_ENDPOINT)
        .set_futures_endpoint(GLOBAL_FUTURES_API_ENDPOINT)
        .set_transport_option(http_transport_option)
        .build()
    )
    
    client = DefaultClient(client_option)
    rest_service = client.rest_service()
    futures_service = rest_service.get_futures_service()
    
    print("✅ SDK initialized")
    
    # Test 1: Public endpoint (ticker)
    print("\n3. Testing PUBLIC endpoint (Get Ticker)...")
    try:
        market_api = futures_service.get_market_api()
        ticker_req = GetTickerReqBuilder().set_symbol("XBTUSDTM").build()
        ticker_resp = market_api.get_ticker(ticker_req)
        
        print(f"✅ SUCCESS!")
        print(f"   Price: ${ticker_resp.price}")
        print(f"   Symbol: {ticker_resp.symbol}")
        print(f"   Response type: {type(ticker_resp)}")
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        print(f"   Error type: {type(e)}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Private endpoint (position)
    print("\n4. Testing PRIVATE endpoint (Get Position)...")
    print("   This is where we expect to see the REAL error...")
    try:
        position_api = futures_service.get_positions_api()
        position_req = GetPositionDetailsReqBuilder().set_symbol("XBTUSDTM").build()
        position_resp = position_api.get_position_details(position_req)
        
        print(f"✅ SUCCESS!")
        print(f"   Position ID: {position_resp.id}")
        print(f"   Symbol: {position_resp.symbol}")
        print(f"   Quantity: {position_resp.current_qty}")
        print(f"   Response type: {type(position_resp)}")
        
    except Exception as e:
        print(f"❌ FAILED - HERE'S THE REAL ERROR:")
        print(f"   Error: {e}")
        print(f"   Error type: {type(e)}")
        
        error_str = str(e)
        if "timestamp" in error_str.lower():
            print(f"\n🔍 CONFIRMED: Timestamp authentication error")
            print(f"   This means the API key lacks Futures Trading permissions")
        elif "permission" in error_str.lower() or "forbidden" in error_str.lower():
            print(f"\n🔍 CONFIRMED: Permission error")
        else:
            print(f"\n🔍 Unknown error type")
        
        print(f"\n📋 Full traceback:")
        import traceback
        traceback.print_exc()
    
    # Test 3: Private endpoint (create test order)
    print("\n5. Testing PRIVATE endpoint (Test Order)...")
    try:
        order_api = futures_service.get_order_api()
        order_req = (
            AddOrderReqBuilder()
            .set_client_oid("test_" + str(int(__import__('time').time() * 1000)))
            .set_side('buy')
            .set_symbol("XBTUSDTM")
            .set_type('market')
            .set_size(1)
            .set_leverage(1)
            .build()
        )
        
        # Use test endpoint
        order_resp = order_api.add_order_test(order_req)
        
        print(f"✅ SUCCESS!")
        print(f"   Order ID: {order_resp.order_id}")
        print(f"   Client OID: {order_resp.client_oid}")
        
    except Exception as e:
        print(f"❌ FAILED:")
        print(f"   Error: {e}")
        print(f"   Error type: {type(e)}")
        
        if "timestamp" in str(e).lower():
            print(f"\n🔍 CONFIRMED: Timestamp error on orders too")
    
    # Summary
    print("\n" + "="*80)
    print("📊 DIRECT SDK TEST COMPLETE")
    print("="*80)
    print("\nIf you see timestamp errors above on private endpoints:")
    print("   → Your API key needs 'Futures Trading' permission")
    print("   → Go to: https://www.kucoin.com/account/api")
    print("   → Edit your API key")
    print("   → Enable: Futures Trading (Read + Trade)")
    print("   → Save and wait 5 minutes")
    print("="*80)


if __name__ == "__main__":
    test_direct_sdk()
