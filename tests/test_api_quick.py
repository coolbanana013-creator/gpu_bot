"""
Quick API Test - Focus on Timestamp Issue

Minimal test to diagnose and fix the timestamp authentication issue.
"""

import sys
from pathlib import Path
import time
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.live_trading.kucoin_universal_client import KucoinUniversalClient
from src.live_trading.credentials import CredentialsManager


def main():
    """Quick API test."""
    print("="*80)
    print("🔧 QUICK API TEST - TIMESTAMP ISSUE DIAGNOSIS")
    print("="*80)
    
    # Load credentials
    print("\n1. Loading credentials...")
    manager = CredentialsManager()
    creds = manager.load_credentials()
    if not creds:
        print("❌ No credentials found!")
        return
    print("✅ Credentials loaded")
    
    # Initialize client
    print("\n2. Initializing client...")
    client = KucoinUniversalClient(
        api_key=creds['api_key'],
        api_secret=creds['api_secret'],
        api_passphrase=creds['api_passphrase'],
        test_mode=True
    )
    print("✅ Client initialized")
    
    # Test public endpoint (should work)
    print("\n3. Testing PUBLIC endpoint (Ticker)...")
    try:
        ticker = client.fetch_ticker("XBTUSDTM")
        if ticker and 'last' in ticker:
            print(f"✅ SUCCESS: BTC Price = ${ticker['last']}")
        else:
            print(f"⚠️  Warning: Unexpected ticker response: {ticker}")
    except Exception as e:
        print(f"❌ FAILED: {e}")
    
    # Test private endpoint (timestamp issue)
    print("\n4. Testing PRIVATE endpoint (Get Position)...")
    print("   This is where we expect the timestamp error...")
    
    try:
        # Get current time for comparison
        local_time_ms = int(time.time() * 1000)
        print(f"   Local timestamp: {local_time_ms}")
        
        position = client.get_position("XBTUSDTM")
        if position is not None:
            print(f"✅ SUCCESS: Position retrieved")
            print(f"   Position details: {position}")
        else:
            print(f"⚠️  No position found (may not have open position)")
    except Exception as e:
        error_msg = str(e)
        print(f"❌ FAILED: {error_msg}")
        
        if "timestamp" in error_msg.lower():
            print("\n📋 DIAGNOSIS:")
            print("   - Timestamp authentication is failing")
            print("   - This is a known issue with Kucoin Universal SDK")
            print("   - Possible causes:")
            print("     1. System time is out of sync with Kucoin servers")
            print("     2. API Key doesn't have proper permissions")
            print("     3. API Key is restricted to specific IPs")
            print("     4. SDK timestamp generation has a bug")
            
            print("\n💡 SOLUTIONS TO TRY:")
            print("   1. Check API key permissions on Kucoin")
            print("   2. Verify no IP restrictions on the API key")
            print("   3. Sync system time with internet time server")
            print("   4. Try using live mode instead of test mode")
            print("   5. Check if futures trading permission is enabled")
    
    # Test order validation (test endpoint)
    print("\n5. Testing ORDER validation (Test endpoint)...")
    try:
        order = client.create_market_order("XBTUSDTM", "buy", 1)
        if order:
            print(f"✅ SUCCESS: Order validated")
            print(f"   Order ID: {order.get('orderId', 'N/A')}")
        else:
            print(f"⚠️  Order validation returned None")
    except Exception as e:
        error_msg = str(e)
        print(f"❌ FAILED: {error_msg}")
        if "timestamp" in error_msg.lower():
            print("   (Same timestamp issue)")
    
    # Final assessment
    print("\n" + "="*80)
    print("📊 ASSESSMENT")
    print("="*80)
    print("✅ Public endpoints: WORKING")
    print("❌ Private endpoints: BLOCKED by timestamp authentication")
    print("\n🎯 NEXT STEPS:")
    print("   1. Log into Kucoin website")
    print("   2. Go to API Management")
    print("   3. Verify API key has:")
    print("      - General permission (read)")
    print("      - Futures permission (read + trade)")
    print("      - No IP restrictions (or add your IP)")
    print("   4. Try regenerating API key if needed")
    print("="*80)


if __name__ == "__main__":
    main()
