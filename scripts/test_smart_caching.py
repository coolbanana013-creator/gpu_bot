"""
Test script to demonstrate smart data caching.
Run this twice - first run downloads, second run uses cache.
"""
import sys
from datetime import datetime
from src.data_provider.fetcher import DataFetcher
from src.utils.validation import log_info

def test_smart_caching():
    """Test smart caching - downloads only missing files."""
    
    log_info("=" * 60)
    log_info("SMART CACHING TEST")
    log_info("=" * 60)
    log_info("This test demonstrates smart caching:")
    log_info("  - First run: Downloads missing files")
    log_info("  - Subsequent runs: Uses cached files (no download)")
    log_info("=" * 60)
    
    # Initialize fetcher with 8 parallel workers for faster downloads (Spot data)
    fetcher = DataFetcher(max_workers=8, exchange_type='spot')
    
    # Fetch 30 days of 1m data (this is a lot of data)
    pair = "BTC/USDT"  # Spot format
    timeframe = "1m"
    days = 30
    end_date = datetime(2025, 11, 9)
    
    log_info(f"\nTesting with: {pair}, {timeframe}, {days} days")
    log_info(f"End date: {end_date.strftime('%Y-%m-%d')}")
    log_info("")
    
    try:
        file_paths = fetcher.fetch_data_range(
            pair=pair,
            timeframe=timeframe,
            total_days=days,
            end_date=end_date
        )
        
        log_info("")
        log_info("=" * 60)
        log_info("[OK] TEST COMPLETED SUCCESSFULLY")
        log_info(f"Total files available: {len(file_paths)}")
        log_info("=" * 60)
        log_info("\nNow run this script again to see caching in action!")
        log_info("The second run should show: '[OK] All X data file(s) already cached'")
        
    except Exception as e:
        log_info("")
        log_info("=" * 60)
        log_info(f"[ERROR] TEST FAILED: {e}")
        log_info("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    test_smart_caching()
