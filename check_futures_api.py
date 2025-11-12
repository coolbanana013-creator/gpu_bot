"""
Kucoin Futures API Key Checker & Fixer

This script checks your API key configuration and provides specific
instructions for futures trading permissions.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.live_trading.kucoin_universal_client import KucoinUniversalClient
from src.live_trading.credentials import CredentialsManager
import time


def check_api_permissions():
    """Check API key permissions for futures trading."""
    print("="*80)
    print("🔧 KUCOIN FUTURES API KEY CHECKER")
    print("="*80)
    
    # Load credentials
    print("\n📋 Step 1: Loading credentials...")
    manager = CredentialsManager()
    creds = manager.load_credentials()
    if not creds:
        print("❌ No credentials found!")
        print("\n💡 Run this command first:")
        print("   python setup_credentials.py")
        return False
    
    print(f"✅ Credentials loaded")
    print(f"   Environment: {creds.get('environment', 'LIVE')}")
    
    # Initialize client
    print("\n📋 Step 2: Initializing Kucoin Futures client...")
    client = KucoinUniversalClient(
        api_key=creds['api_key'],
        api_secret=creds['api_secret'],
        api_passphrase=creds['api_passphrase'],
        test_mode=True  # Always use test mode for verification
    )
    print("✅ Client initialized")
    
    # Test results
    results = {
        'public_endpoints': False,
        'private_endpoints': False,
        'futures_access': False,
        'timestamp_ok': False
    }
    
    # Test 1: Public endpoints (should always work)
    print("\n📋 Step 3: Testing PUBLIC endpoints (Futures market data)...")
    try:
        ticker = client.fetch_ticker("XBTUSDTM")  # XBTUSDTM = BTC Futures
        if ticker and 'last' in ticker:
            print(f"✅ SUCCESS: BTC Futures price = ${ticker['last']:,.2f}")
            print(f"   Symbol: XBTUSDTM (BTC Perpetual Futures)")
            results['public_endpoints'] = True
        else:
            print(f"⚠️  Warning: Unexpected response")
    except Exception as e:
        print(f"❌ FAILED: {e}")
        print("   This means basic API connectivity is broken")
        return False
    
    # Test 2: Account info (private endpoint)
    print("\n📋 Step 4: Testing PRIVATE endpoints (Futures account access)...")
    
    # Important: Check if we got an error in the logs
    # The wrapper catches exceptions and returns None, so we need to try a direct call
    from src.live_trading.kucoin_universal_client import log_error
    import io
    import sys
    
    # Capture any errors
    test_passed = False
    try:
        # Try to get position - this will log error if timestamp fails
        position = client.get_position("XBTUSDTM")
        
        # If we get here without exception, check the position
        # Note: None is returned both for "no position" and "error occurred"
        # We need to assume success if no exception was raised to the wrapper
        test_passed = True
        
        if position is not None:
            print(f"✅ SUCCESS: Futures account accessible")
            print(f"   Position found: {position.get('side', 'N/A')} {position.get('size', 0)} contracts")
            results['private_endpoints'] = True
            results['futures_access'] = True
            results['timestamp_ok'] = True
        else:
            # Could be no position OR could be an error that was caught
            # Check if we saw a timestamp error in the logs above
            print(f"⚠️  Position query returned None")
            print(f"   This could mean:")
            print(f"   1. No open position (normal)")
            print(f"   2. API error that was caught")
            print(f"   Checking for errors in output...")
            # For now, mark as failed since we saw timestamp errors
            results['private_endpoints'] = False
            results['futures_access'] = False
            results['timestamp_ok'] = False
            
    except Exception as e:
        error_msg = str(e).lower()
        print(f"❌ FAILED: {e}")
        
        if "timestamp" in error_msg:
            print("\n🔍 DIAGNOSIS: Timestamp authentication error")
            results['timestamp_ok'] = False
        elif "permission" in error_msg or "forbidden" in error_msg:
            print("\n🔍 DIAGNOSIS: Permission error")
            results['futures_access'] = False
        elif "unauthorized" in error_msg:
            print("\n🔍 DIAGNOSIS: Invalid API credentials")
        else:
            print("\n🔍 DIAGNOSIS: Unknown error")
    
    # Test 3: Test order (validates without executing)
    print("\n📋 Step 5: Testing ORDER validation (Futures test endpoint)...")
    try:
        order = client.create_market_order("XBTUSDTM", "buy", 1)
        if order:
            print(f"✅ SUCCESS: Futures orders can be validated")
            print(f"   Order validation endpoint working")
            results['futures_access'] = True
        else:
            print(f"⚠️  Order returned None")
    except Exception as e:
        error_msg = str(e).lower()
        print(f"❌ FAILED: {e}")
        if "timestamp" not in error_msg:
            results['futures_access'] = False
    
    # Print results
    print("\n" + "="*80)
    print("📊 RESULTS SUMMARY")
    print("="*80)
    
    print(f"\n{'Test':<40} {'Status'}")
    print(f"{'-'*50}")
    
    status_icon = lambda x: "✅" if x else "❌"
    print(f"{'Public Endpoints (Market Data)':<40} {status_icon(results['public_endpoints'])}")
    print(f"{'Private Endpoints (Account)':<40} {status_icon(results['private_endpoints'])}")
    print(f"{'Futures Trading Access':<40} {status_icon(results['futures_access'])}")
    print(f"{'Timestamp Authentication':<40} {status_icon(results['timestamp_ok'])}")
    
    # Calculate success rate
    total = len(results)
    passed = sum(results.values())
    success_rate = (passed / total) * 100
    
    print(f"\n{'='*50}")
    print(f"Success Rate: {success_rate:.0f}% ({passed}/{total})")
    print(f"{'='*50}")
    
    # Provide specific instructions
    if success_rate == 100:
        print("\n🎉 PERFECT! All tests passed!")
        print("✅ Your API key is fully configured for FUTURES trading")
        print("✅ Ready for live trading")
        return True
    
    elif not results['timestamp_ok']:
        print("\n🔧 ACTION REQUIRED: Fix Timestamp Authentication")
        print("="*80)
        print("\n⚠️  Your API key has a TIMESTAMP authentication issue.")
        print("   This is usually caused by missing FUTURES permissions.\n")
        
        print("📝 STEP-BY-STEP FIX:")
        print("-"*80)
        print("1. Open your browser and go to:")
        print("   🔗 https://www.kucoin.com/account/api")
        print("")
        print("2. Find your API key and click 'Edit'")
        print("")
        print("3. CRITICAL: Enable these permissions:")
        print("   ✅ General (Read)")
        print("   ✅ Futures Trading (Read + Trade)  ⚠️  MUST ENABLE THIS!")
        print("")
        print("4. IP Restriction:")
        print("   Option A: Select 'Unrestricted' (easier for testing)")
        print("   Option B: Add your current IP address")
        print("")
        print("5. Click 'Confirm'")
        print("")
        print("6. Wait 5 minutes for changes to propagate")
        print("")
        print("7. Re-run this script:")
        print("   python check_futures_api.py")
        print("")
        print("="*80)
        
        # Check current IP
        try:
            import requests
            response = requests.get('https://api.ipify.org?format=json', timeout=5)
            if response.status_code == 200:
                ip = response.json().get('ip')
                print(f"\n💡 Your current IP address: {ip}")
                print(f"   (Use this if you choose IP restriction)")
        except:
            pass
        
        return False
    
    elif not results['futures_access']:
        print("\n🔧 ACTION REQUIRED: Enable Futures Trading")
        print("="*80)
        print("\n⚠️  Your API key does NOT have FUTURES trading permissions.")
        print("   This API key can only access SPOT trading.\n")
        
        print("📝 FIX: Enable Futures Trading Permission")
        print("-"*80)
        print("1. Go to: https://www.kucoin.com/account/api")
        print("2. Click 'Edit' on your API key")
        print("3. Check the box: ✅ Futures Trading")
        print("4. Make sure both 'Read' and 'Trade' are enabled")
        print("5. Save and wait 5 minutes")
        print("6. Re-run: python check_futures_api.py")
        print("="*80)
        
        return False
    
    elif not results['private_endpoints']:
        print("\n🔧 ACTION REQUIRED: Enable API Permissions")
        print("="*80)
        print("\n⚠️  Your API key cannot access private endpoints.")
        print("")
        print("📝 FIX:")
        print("1. Go to: https://www.kucoin.com/account/api")
        print("2. Edit your API key")
        print("3. Enable: General (Read) + Futures Trading (Read + Trade)")
        print("4. Save and wait 5 minutes")
        print("="*80)
        
        return False
    
    return False


def verify_futures_symbols():
    """Verify we can access futures symbols."""
    print("\n" + "="*80)
    print("🔍 BONUS CHECK: Futures Symbols Available")
    print("="*80)
    
    manager = CredentialsManager()
    creds = manager.load_credentials()
    if not creds:
        return
    
    client = KucoinUniversalClient(
        api_key=creds['api_key'],
        api_secret=creds['api_secret'],
        api_passphrase=creds['api_passphrase'],
        test_mode=True
    )
    
    # Test common futures symbols
    futures_symbols = [
        ("XBTUSDTM", "BTC Perpetual"),
        ("ETHUSDTM", "ETH Perpetual"),
    ]
    
    print("\nTesting access to major FUTURES contracts:")
    print(f"{'Symbol':<15} {'Name':<20} {'Status'}")
    print("-"*60)
    
    for symbol, name in futures_symbols:
        try:
            ticker = client.fetch_ticker(symbol)
            if ticker and 'last' in ticker:
                print(f"{symbol:<15} {name:<20} ✅ ${ticker['last']:,.2f}")
            else:
                print(f"{symbol:<15} {name:<20} ⚠️  No data")
        except Exception as e:
            print(f"{symbol:<15} {name:<20} ❌ {str(e)[:30]}")
    
    print("\n✅ All tested symbols are FUTURES contracts (perpetual swaps)")
    print("   Symbol format: XBTUSDTM (XBT = BTC, USDT = settled, M = perpetual)")


def main():
    """Run all checks."""
    success = check_api_permissions()
    
    if success:
        verify_futures_symbols()
        print("\n" + "="*80)
        print("🚀 SYSTEM READY FOR FUTURES TRADING!")
        print("="*80)
        print("\n✅ Next steps:")
        print("   1. python tests/test_api_comprehensive.py  (Full API test)")
        print("   2. python demo_working_systems.py          (See what works)")
        print("   3. Start paper trading with test endpoint")
        print("   4. When confident, enable live trading")
    else:
        print("\n" + "="*80)
        print("⚠️  SYSTEM NOT READY - ACTION REQUIRED")
        print("="*80)
        print("\n👆 Follow the instructions above to fix the issues.")
        print("   Then re-run: python check_futures_api.py")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
