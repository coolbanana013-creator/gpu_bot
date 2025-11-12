"""
Kucoin Client with Server Time Synchronization

This wrapper fetches Kucoin's server time and adjusts for clock drift.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import time
import requests
from threading import Thread, Lock
from typing import Optional


class TimeSync:
    """Synchronizes with Kucoin server time."""
    
    def __init__(self):
        self.offset_ms = 0
        self.last_sync = 0
        self.sync_interval = 60000  # Sync every 60 seconds
        self.lock = Lock()
        self._sync_now()
    
    def _sync_now(self):
        """Fetch server time and calculate offset."""
        try:
            local_before = int(time.time() * 1000)
            response = requests.get('https://api-futures.kucoin.com/api/v1/timestamp', timeout=5)
            local_after = int(time.time() * 1000)
            
            if response.status_code == 200:
                server_time = response.json().get('data', 0)
                # Use average of before/after for better accuracy
                local_time = (local_before + local_after) // 2
                
                with self.lock:
                    self.offset_ms = server_time - local_time
                    self.last_sync = local_time
                    
                print(f"⏰ Time sync: offset = {self.offset_ms} ms")
                return True
        except Exception as e:
            print(f"⚠️  Time sync failed: {e}")
            with self.lock:
                self.offset_ms = 0
        return False
    
    def get_server_time(self) -> int:
        """Get current server time (with offset applied)."""
        # Check if we need to resync
        current_time = int(time.time() * 1000)
        
        with self.lock:
            if current_time - self.last_sync > self.sync_interval:
                # Resync in background
                Thread(target=self._sync_now, daemon=True).start()
            
            return current_time + self.offset_ms


# Global time sync instance
_time_sync = TimeSync()


def get_kucoin_server_time() -> int:
    """Get Kucoin server time."""
    return _time_sync.get_server_time()


def resync_time():
    """Force time resynchronization."""
    _time_sync._sync_now()


if __name__ == "__main__":
    print("Testing time synchronization...")
    
    # Initial sync
    server_time = get_kucoin_server_time()
    local_time = int(time.time() * 1000)
    
    print(f"\nLocal time:  {local_time}")
    print(f"Server time: {server_time}")
    print(f"Offset:      {_time_sync.offset_ms} ms")
    
    if abs(_time_sync.offset_ms) < 1000:
        print(f"\n✅ Excellent sync (< 1 second)")
    elif abs(_time_sync.offset_ms) < 5000:
        print(f"\n✅ Good sync (< 5 seconds)")
    else:
        print(f"\n⚠️  Significant offset ({abs(_time_sync.offset_ms)/1000:.2f} seconds)")
    
    print(f"\nTime sync is active and will update every {_time_sync.sync_interval/1000:.0f} seconds")
