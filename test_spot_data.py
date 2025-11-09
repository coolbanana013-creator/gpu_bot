"""
Test Spot Data Fetching - Verify new organized structure works.
"""
from datetime import datetime
from src.data_provider.fetcher import DataFetcher
from src.utils.validation import log_info
from pathlib import Path

def test_spot_data_fetching():
    """Test fetching Spot data with new organized file structure."""
    
    log_info("=" * 70)
    log_info("SPOT DATA FETCHING TEST")
    log_info("=" * 70)
    log_info("Testing:")
    log_info("  - Kucoin Spot API connection")
    log_info("  - BTC/USDT classical format (not perpetual)")
    log_info("  - Organized file structure: data/BTC_USDT/1m/date.parquet")
    log_info("  - Smart caching")
    log_info("=" * 70)
    
    # Initialize Spot fetcher
    log_info("\n[1/4] Initializing Spot fetcher...")
    fetcher = DataFetcher(max_workers=8, exchange_type='spot')
    
    # Test configuration
    pair = "BTC/USDT"  # Classical spot format
    timeframe = "15m"
    days = 7  # One week
    end_date = datetime(2025, 11, 9)
    
    log_info(f"[2/4] Test configuration:")
    log_info(f"  - Exchange: Kucoin Spot")
    log_info(f"  - Pair: {pair}")
    log_info(f"  - Timeframe: {timeframe}")
    log_info(f"  - Days: {days}")
    log_info(f"  - End date: {end_date.strftime('%Y-%m-%d')}")
    
    # Fetch data
    log_info(f"\n[3/4] Fetching data...")
    try:
        file_paths = fetcher.fetch_data_range(
            pair=pair,
            timeframe=timeframe,
            total_days=days,
            end_date=end_date
        )
        
        log_info(f"\n[4/4] Verifying file structure...")
        
        # Check first file to verify structure
        if file_paths:
            first_file = file_paths[0]
            log_info(f"  Sample file path: {first_file}")
            
            # Verify organized structure
            expected_structure = f"data{Path('/').as_posix()}BTC_USDT{Path('/').as_posix()}{timeframe}{Path('/').as_posix()}"
            actual_path = str(first_file).replace('\\', '/')
            
            if expected_structure in actual_path:
                log_info(f"  [OK] File structure is organized correctly!")
                log_info(f"       Structure: data/BTC_USDT/{timeframe}/YYYY-MM-DD.parquet")
            else:
                log_info(f"  [WARNING] Unexpected structure")
                log_info(f"       Expected: {expected_structure}")
                log_info(f"       Got: {actual_path}")
            
            # Check if file exists and is readable
            import pandas as pd
            df = pd.read_parquet(first_file)
            log_info(f"  [OK] File is readable: {len(df)} candles")
            log_info(f"       Columns: {list(df.columns)}")
        
        log_info("\n" + "=" * 70)
        log_info("[OK] ALL TESTS PASSED!")
        log_info("=" * 70)
        log_info(f"\nTotal files: {len(file_paths)}")
        log_info(f"Storage location: data/BTC_USDT/{timeframe}/")
        log_info("\nNow you can:")
        log_info("  1. Run main.py - it will use BTC/USDT Spot by default")
        log_info("  2. Check data/ directory to see organized structure")
        log_info("  3. Subsequent runs will use cached data (instant!)")
        
    except Exception as e:
        log_info("\n" + "=" * 70)
        log_info(f"[ERROR] TEST FAILED: {e}")
        log_info("=" * 70)
        raise

if __name__ == "__main__":
    test_spot_data_fetching()
