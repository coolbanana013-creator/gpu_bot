"""
Encrypted Credentials Manager

Stores Kucoin API credentials securely using Fernet encryption.
Auto-prompts for credentials if not present.
"""

import os
import json
from pathlib import Path
from typing import Dict, Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import getpass

from ..utils.validation import log_info, log_error, log_warning


class CredentialsManager:
    """Secure credential storage using encryption."""
    
    CREDENTIALS_FILE = Path.home() / ".gpu_bot" / "credentials.enc"
    KEY_FILE = Path.home() / ".gpu_bot" / ".key"
    
    def __init__(self):
        """Initialize credentials manager."""
        self.credentials_dir = Path.home() / ".gpu_bot"
        self.credentials_dir.mkdir(parents=True, exist_ok=True)
        
        self.cipher_suite = self._get_or_create_cipher()
    
    def _get_or_create_cipher(self) -> Fernet:
        """Get or create encryption cipher."""
        if self.KEY_FILE.exists():
            # Load existing key
            with open(self.KEY_FILE, 'rb') as f:
                key = f.read()
        else:
            # Create new key from user password
            log_info("First-time setup: Creating encryption key...")
            password = getpass.getpass("Enter a master password to encrypt your API keys: ")
            confirm = getpass.getpass("Confirm password: ")
            
            if password != confirm:
                raise ValueError("Passwords do not match!")
            
            # Derive key from password using PBKDF2HMAC
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b'gpu_bot_salt_v1',  # In production, use random salt
                iterations=100000,
                backend=default_backend()
            )
            key = kdf.derive(password.encode())
            
            # Encode for Fernet
            from base64 import urlsafe_b64encode
            key = urlsafe_b64encode(key)
            
            # Save key
            with open(self.KEY_FILE, 'wb') as f:
                f.write(key)
            
            # Secure file permissions (Unix-like systems)
            try:
                os.chmod(self.KEY_FILE, 0o600)
            except:
                pass
            
            log_info(f"Encryption key created and saved to: {self.KEY_FILE}")
            log_warning("⚠️  IMPORTANT: Keep your master password safe! You'll need it to access your API keys.")
        
        return Fernet(key)
    
    def save_credentials(self, api_key: str, api_secret: str, api_passphrase: str, environment: str = "live"):
        """
        Save encrypted credentials.
        
        Args:
            api_key: Kucoin API key
            api_secret: Kucoin API secret
            api_passphrase: Kucoin API passphrase
            environment: 'live' or 'sandbox'
        """
        credentials = {
            'api_key': api_key,
            'api_secret': api_secret,
            'api_passphrase': api_passphrase,
            'environment': environment
        }
        
        # Encrypt
        json_data = json.dumps(credentials).encode()
        encrypted_data = self.cipher_suite.encrypt(json_data)
        
        # Save
        with open(self.CREDENTIALS_FILE, 'wb') as f:
            f.write(encrypted_data)
        
        log_info(f"✅ Credentials saved securely to: {self.CREDENTIALS_FILE}")
    
    def load_credentials(self) -> Optional[Dict[str, str]]:
        """
        Load and decrypt credentials.
        
        Returns:
            Dict with api_key, api_secret, api_passphrase, environment or None
        """
        if not self.CREDENTIALS_FILE.exists():
            return None
        
        try:
            # Load encrypted data
            with open(self.CREDENTIALS_FILE, 'rb') as f:
                encrypted_data = f.read()
            
            # Decrypt
            decrypted_data = self.cipher_suite.decrypt(encrypted_data)
            credentials = json.loads(decrypted_data.decode())
            
            return credentials
        except Exception as e:
            log_error(f"Failed to decrypt credentials: {e}")
            return None
    
    def prompt_and_save_credentials(self):
        """Interactive prompt to save credentials."""
        print("\n" + "="*60)
        print("KUCOIN API CREDENTIALS SETUP")
        print("="*60)
        print("\nYou need Kucoin Futures API credentials to trade.")
        print("Get them at: https://www.kucoin.com/account/api")
        print("\n⚠️  SECURITY NOTES:")
        print("  - Use API keys with FUTURES trading permissions")
        print("  - Restrict to your IP address if possible")
        print("  - Enable withdrawals ONLY if needed (not recommended)")
        print("  - Keys will be encrypted and stored locally")
        print()
        
        api_key = input("Enter API Key: ").strip()
        api_secret = getpass.getpass("Enter API Secret: ").strip()
        api_passphrase = getpass.getpass("Enter API Passphrase: ").strip()
        
        print("\nChoose environment:")
        print("  1. Sandbox (testnet - fake money)")
        print("  2. Live (production - real money)")
        env_choice = input("Select [1-2]: ").strip()
        
        environment = "sandbox" if env_choice == "1" else "live"
        
        # Confirm
        print(f"\n⚠️  You are setting up {environment.upper()} environment")
        confirm = input("Confirm? (yes/no): ").strip().lower()
        
        if confirm != 'yes':
            log_warning("Setup cancelled")
            return False
        
        self.save_credentials(api_key, api_secret, api_passphrase, environment)
        return True
    
    def credentials_exist(self) -> bool:
        """Check if credentials file exists."""
        return self.CREDENTIALS_FILE.exists()
    
    def delete_credentials(self):
        """Delete stored credentials (for reset)."""
        if self.CREDENTIALS_FILE.exists():
            self.CREDENTIALS_FILE.unlink()
            log_info("Credentials deleted")
        
        if self.KEY_FILE.exists():
            self.KEY_FILE.unlink()
            log_info("Encryption key deleted")
