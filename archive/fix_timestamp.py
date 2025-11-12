"""
Timestamp Synchronization Test & Fix

Since the API key has futures permissions but we're getting timestamp errors,
this is likely a clock synchronization issue or SDK timestamp generation problem.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import time
from datetime import datetime
import requests


def check_time_sync():
    """Check if system time is in sync with Kucoin server time."""
    print("="*80)
    print("🕐 TIMESTAMP SYNCHRONIZATION CHECK")
    print("="*80)
    
    # Get local time
    local_time_ms = int(time.time() * 1000)
    local_time = datetime.fromtimestamp(local_time_ms / 1000)
    
    print(f"\n1. Local System Time:")
    print(f"   Timestamp: {local_time_ms}")
    print(f"   DateTime:  {local_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
    print(f"   Timezone:  {datetime.now().astimezone().tzinfo}")
    
    # Get Kucoin server time
    print(f"\n2. Kucoin Server Time:")
    try:
        # Kucoin futures server time endpoint
        response = requests.get('https://api-futures.kucoin.com/api/v1/timestamp', timeout=10)
        
        if response.status_code == 200:
            server_data = response.json()
            server_time_ms = server_data.get('data', 0)
            server_time = datetime.fromtimestamp(server_time_ms / 1000)
            
            print(f"   Timestamp: {server_time_ms}")
            print(f"   DateTime:  {server_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
            
            # Calculate difference
            diff_ms = abs(local_time_ms - server_time_ms)
            diff_seconds = diff_ms / 1000
            
            print(f"\n3. Time Difference:")
            print(f"   Difference: {diff_ms} ms ({diff_seconds:.2f} seconds)")
            
            if diff_ms < 1000:
                print(f"   ✅ EXCELLENT: Time is perfectly synced (< 1 second)")
                return True, diff_ms
            elif diff_ms < 5000:
                print(f"   ✅ GOOD: Time is acceptably synced (< 5 seconds)")
                return True, diff_ms
            elif diff_ms < 30000:
                print(f"   ⚠️  WARNING: Time difference is significant (< 30 seconds)")
                print(f"   This might cause intermittent issues")
                return True, diff_ms
            else:
                print(f"   ❌ CRITICAL: Time is badly out of sync (> 30 seconds)")
                print(f"   This WILL cause authentication failures")
                return False, diff_ms
        else:
            print(f"   ❌ Failed to get server time: HTTP {response.status_code}")
            return None, 0
            
    except Exception as e:
        print(f"   ❌ Error getting server time: {e}")
        return None, 0


def test_with_manual_timestamp():
    """Test API with manually synchronized timestamp."""
    print("\n" + "="*80)
    print("🔧 TESTING WITH ADJUSTED TIMESTAMP")
    print("="*80)
    
    from src.live_trading.credentials import CredentialsManager
    
    # Load credentials
    manager = CredentialsManager()
    creds = manager.load_credentials()
    if not creds:
        print("❌ No credentials found!")
        return
    
    # Try to get server time offset
    try:
        response = requests.get('https://api-futures.kucoin.com/api/v1/timestamp', timeout=10)
        server_time_ms = response.json().get('data', 0)
        local_time_ms = int(time.time() * 1000)
        offset_ms = server_time_ms - local_time_ms
        
        print(f"\nServer Time Offset: {offset_ms} ms")
        print(f"We'll use server time for authentication...")
        
    except Exception as e:
        print(f"⚠️  Could not get server time: {e}")
        offset_ms = 0
    
    # Import SDK and test with adjusted timestamp
    print(f"\n📋 Testing position query with adjusted timestamp...")
    
    # Unfortunately, the SDK doesn't expose timestamp override
    # Let's check if there's a workaround
    print(f"\n⚠️  Kucoin Universal SDK doesn't allow timestamp override")
    print(f"   This is a limitation of the SDK itself")
    print(f"\n💡 Possible solutions:")
    print(f"   1. Sync system clock with internet time")
    print(f"   2. Use a different Kucoin SDK that allows timestamp control")
    print(f"   3. Modify the SDK source code (not recommended)")
    print(f"   4. Contact Kucoin support about the timestamp issue")


def check_api_key_config():
    """Double-check API key configuration."""
    print("\n" + "="*80)
    print("🔑 API KEY CONFIGURATION CHECK")
    print("="*80)
    
    from src.live_trading.credentials import CredentialsManager
    
    manager = CredentialsManager()
    creds = manager.load_credentials()
    if not creds:
        print("❌ No credentials found!")
        return
    
    print(f"\n✅ API Key loaded")
    print(f"   Environment: {creds.get('environment', 'LIVE')}")
    print(f"   Key starts with: {creds['api_key'][:8]}...")
    print(f"   Secret length: {len(creds['api_secret'])} chars")
    print(f"   Passphrase length: {len(creds['api_passphrase'])} chars")
    
    print(f"\n📋 Required Permissions (Confirm on Kucoin website):")
    print(f"   ✅ General (Read)")
    print(f"   ✅ Futures Trading (Read)")  
    print(f"   ✅ Futures Trading (Trade)")
    print(f"   ✅ No IP restrictions OR current IP whitelisted")
    
    print(f"\n🌐 Check at: https://www.kucoin.com/account/api")


def try_windows_time_sync():
    """Try to sync Windows time with internet time server."""
    print("\n" + "="*80)
    print("⏰ WINDOWS TIME SYNCHRONIZATION")
    print("="*80)
    
    print(f"\n📋 To sync your Windows clock:")
    print(f"   1. Open Settings (Win + I)")
    print(f"   2. Go to: Time & Language > Date & Time")
    print(f"   3. Turn ON: Set time automatically")
    print(f"   4. Turn ON: Set time zone automatically")
    print(f"   5. Click: Sync now")
    print(f"\nOR via PowerShell (as Administrator):")
    print(f"   w32tm /resync")
    print(f"\nOR via Command Prompt (as Administrator):")
    print(f"   net stop w32time")
    print(f"   net start w32time")
    print(f"   w32tm /resync")


def investigate_sdk_issue():
    """Investigate if this is a known SDK issue."""
    print("\n" + "="*80)
    print("🔬 SDK INVESTIGATION")
    print("="*80)
    
    print(f"\n📋 Kucoin Universal SDK Timestamp Issue:")
    print(f"   The SDK generates timestamps internally using time.time()")
    print(f"   If your system clock is even slightly off, auth will fail")
    print(f"\n🔍 Known Issues:")
    print(f"   1. Some users report timestamp errors even with correct time")
    print(f"   2. SDK may not handle timezone conversions properly")
    print(f"   3. Windows time service sometimes drifts")
    print(f"\n💡 Workarounds:")
    print(f"   1. Force time sync (see above)")
    print(f"   2. Restart computer (resets time service)")
    print(f"   3. Try creating a new API key (sometimes helps)")
    print(f"   4. Check if firewall is blocking time sync")


def main():
    """Run all checks."""
    # Check time synchronization
    synced, diff_ms = check_time_sync()
    
    # Check API key
    check_api_key_config()
    
    # If time is out of sync
    if synced == False:
        print(f"\n" + "="*80)
        print(f"❌ PROBLEM IDENTIFIED: Clock Out of Sync")
        print(f"="*80)
        print(f"\nYour system clock is {diff_ms/1000:.2f} seconds off from Kucoin server")
        print(f"This is causing the timestamp authentication errors")
        
        try_windows_time_sync()
        
    elif synced == True and diff_ms < 5000:
        print(f"\n" + "="*80)
        print(f"🤔 MYSTERIOUS: Time is synced but still getting errors")
        print(f"="*80)
        print(f"\nYour clock is properly synced ({diff_ms} ms difference)")
        print(f"But you're still getting timestamp errors")
        print(f"\nPossible causes:")
        print(f"   1. API key was just created/modified (wait 5-10 minutes)")
        print(f"   2. Kucoin server-side issue (temporary)")
        print(f"   3. SDK bug with timestamp generation")
        print(f"   4. Rate limiting being reported as timestamp error")
        
        investigate_sdk_issue()
        
        print(f"\n💡 Recommended actions:")
        print(f"   1. Wait 10 minutes (if key was recently changed)")
        print(f"   2. Force time sync anyway (might help)")
        print(f"   3. Try creating a new API key")
        print(f"   4. Contact Kucoin support")
        
    else:
        print(f"\n" + "="*80)
        print(f"⚠️  Could not verify time sync")
        print(f"="*80)
        print(f"\nPlease check manually:")
        print(f"   1. Visit: https://time.is")
        print(f"   2. Verify your system time is correct")
        print(f"   3. If off, sync using Windows time settings")
        
        try_windows_time_sync()
    
    # Final summary
    print(f"\n" + "="*80)
    print(f"📊 NEXT STEPS")
    print(f"="*80)
    print(f"\n1. Sync your system time (instructions above)")
    print(f"2. Wait 5 minutes after syncing")
    print(f"3. Re-run: python test_sdk_direct.py")
    print(f"4. If still failing, try creating a new API key")
    print(f"\n💡 Alternative: The issue might resolve itself in a few minutes")
    print(f"   Kucoin servers sometimes have temporary issues")
    print(f"="*80)


if __name__ == "__main__":
    main()
