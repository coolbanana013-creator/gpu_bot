"""
Interactive Credentials Setup

Prompts for Kucoin API credentials and stores them encrypted.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.live_trading.credentials import CredentialsManager
from src.utils.validation import log_info, log_success, log_error
import getpass


def setup_credentials():
    """Interactive credential setup."""
    print("\n" + "="*80)
    print("🔐 KUCOIN API CREDENTIALS SETUP")
    print("="*80)
    print("\nYour credentials will be encrypted and stored securely.")
    print("You'll need:")
    print("  1. API Key")
    print("  2. API Secret")
    print("  3. API Passphrase")
    print("  4. Choose Sandbox or Live environment")
    print("\n" + "="*80 + "\n")
    
    try:
        # Get credentials
        api_key = input("Enter your Kucoin API Key: ").strip()
        if not api_key:
            log_error("API Key cannot be empty!")
            return False
        
        api_secret = getpass.getpass("Enter your Kucoin API Secret: ").strip()
        if not api_secret:
            log_error("API Secret cannot be empty!")
            return False
        
        api_passphrase = getpass.getpass("Enter your Kucoin API Passphrase: ").strip()
        if not api_passphrase:
            log_error("API Passphrase cannot be empty!")
            return False
        
        # Choose environment
        print("\nChoose trading environment:")
        print("  1. Sandbox (Test environment - recommended)")
        print("  2. Live (Real money trading)")
        choice = input("Enter choice (1 or 2): ").strip()
        
        is_sandbox = choice == "1"
        env_name = "SANDBOX" if is_sandbox else "LIVE"
        
        print(f"\n⚠️  You selected: {env_name} environment")
        confirm = input("Is this correct? (yes/no): ").strip().lower()
        
        if confirm not in ['yes', 'y']:
            log_info("Setup cancelled.")
            return False
        
        # Save credentials
        log_info("\nSaving credentials...")
        manager = CredentialsManager()
        
        environment = "sandbox" if is_sandbox else "live"
        
        manager.save_credentials(
            api_key=api_key,
            api_secret=api_secret,
            api_passphrase=api_passphrase,
            environment=environment
        )
        
        log_success("\n✅ Credentials saved successfully!")
        log_info(f"Environment: {env_name}")
        log_info(f"Encrypted file: {manager.CREDENTIALS_FILE}")
        
        # Verify by loading
        log_info("\nVerifying credentials...")
        loaded = manager.load_credentials()
        if loaded:
            log_success("✅ Credentials verified - ready to use!")
            print("\n" + "="*80)
            print("You can now run API tests with:")
            print("  python tests/test_api_endpoints.py")
            print("="*80 + "\n")
            return True
        else:
            log_error("Failed to verify credentials!")
            return False
            
    except KeyboardInterrupt:
        log_info("\n\nSetup cancelled by user.")
        return False
    except Exception as e:
        log_error(f"\nSetup failed: {e}")
        return False


if __name__ == "__main__":
    success = setup_credentials()
    sys.exit(0 if success else 1)
