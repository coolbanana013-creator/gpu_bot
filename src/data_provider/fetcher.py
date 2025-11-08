"""
Market data fetcher with smart downloading from Kucoin API.
Only downloads missing data files, uses Parquet for efficient storage.
"""
import os
import time
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
import pandas as pd
import ccxt
from pathlib import Path

from ..utils.validation import (
    validate_pair, validate_timeframe, validate_int,
    log_info, log_error, log_warning, log_debug
)
from ..utils.config import (
    DATA_DIR, DATA_FILE_PATTERN, OHLCV_COLUMNS,
    API_REQUEST_DELAY, MAX_API_RETRIES, TIMEFRAME_TO_MS,
    DATA_BUFFER_MULTIPLIER
)


class DataFetcher:
    """
    Fetches OHLCV data from Kucoin Futures API with smart caching.
    """
    
    def __init__(self, data_dir: str = DATA_DIR):
        """
        Initialize DataFetcher.
        
        Args:
            data_dir: Directory to store/load data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Kucoin exchange (futures)
        try:
            self.exchange = ccxt.kucoinfutures({
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',
                }
            })
            log_info("Initialized Kucoin Futures API connection")
        except Exception as e:
            log_error(f"Failed to initialize Kucoin API: {e}")
            raise RuntimeError(f"Cannot initialize Kucoin API: {e}")
    
    def _get_file_path(self, pair: str, timeframe: str, date: datetime) -> Path:
        """
        Generate file path for a given pair, timeframe, and date.
        
        Args:
            pair: Trading pair (e.g., "BTC/USDT")
            timeframe: Timeframe string (e.g., "1m")
            date: Date for the data
            
        Returns:
            Path to the data file
        """
        pair_str = pair.replace('/', '_').lower()
        date_str = date.strftime('%Y-%m-%d')
        filename = DATA_FILE_PATTERN.format(
            pair=pair_str,
            timeframe=timeframe,
            date=date_str
        )
        return self.data_dir / filename
    
    def _file_exists(self, file_path: Path) -> bool:
        """Check if data file exists and is valid."""
        if not file_path.exists():
            return False
        
        try:
            # Try to read the file to ensure it's valid
            df = pd.read_parquet(file_path)
            if df.empty or len(df) < 10:  # Sanity check
                log_warning(f"File {file_path.name} exists but has insufficient data")
                return False
            return True
        except Exception as e:
            log_warning(f"File {file_path.name} exists but is corrupted: {e}")
            return False
    
    def _fetch_ohlcv_for_date(
        self,
        pair: str,
        timeframe: str,
        date: datetime,
        retry_count: int = 0
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for a specific date from Kucoin API.
        
        Args:
            pair: Trading pair
            timeframe: Timeframe string
            date: Date to fetch data for
            retry_count: Current retry attempt
            
        Returns:
            DataFrame with OHLCV data, or None if failed
        """
        # Calculate start and end timestamps for the day (UTC)
        start_dt = datetime(date.year, date.month, date.day, 0, 0, 0)
        end_dt = start_dt + timedelta(days=1)
        
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)
        
        try:
            log_debug(f"Fetching {pair} {timeframe} data for {date.strftime('%Y-%m-%d')}")
            
            # Fetch data from exchange with pagination to get full day
            all_ohlcv = []
            current_start = start_ms
            max_iterations = 20  # Safety limit to prevent infinite loops
            iteration = 0
            
            while current_start < end_ms and iteration < max_iterations:
                iteration += 1
                
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol=pair,
                    timeframe=timeframe,
                    since=current_start,
                    limit=1500  # Kucoin limit (though they may return less)
                )
                
                if not ohlcv:
                    break
                
                # Filter to only include data for this specific day
                filtered = [
                    candle for candle in ohlcv
                    if start_ms <= candle[0] < end_ms
                ]
                
                if not filtered:
                    break
                
                all_ohlcv.extend(filtered)
                
                # Move to next batch - use last candle timestamp + 1 minute (60000ms for 1m timeframe)
                last_timestamp = ohlcv[-1][0]
                current_start = last_timestamp + 60000  # Move forward by 1 minute
                
                # If the last candle returned is beyond our end time, we're done
                if last_timestamp >= end_ms:
                    break
                
                # Add small delay to avoid rate limiting
                if iteration < max_iterations:
                    time.sleep(0.1)
            
            if not all_ohlcv:
                log_warning(f"No data returned for {pair} {timeframe} on {date.strftime('%Y-%m-%d')}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(
                all_ohlcv,
                columns=OHLCV_COLUMNS
            )
            
            # Validate data
            if df.empty:
                log_warning(f"Empty DataFrame for {pair} {timeframe} on {date.strftime('%Y-%m-%d')}")
                return None
            
            log_info(f"Fetched {len(df)} candles for {pair} {timeframe} on {date.strftime('%Y-%m-%d')}")
            return df
            
        except ccxt.NetworkError as e:
            if retry_count < MAX_API_RETRIES:
                log_warning(f"Network error, retrying ({retry_count + 1}/{MAX_API_RETRIES}): {e}")
                time.sleep(API_REQUEST_DELAY * 2)
                return self._fetch_ohlcv_for_date(pair, timeframe, date, retry_count + 1)
            else:
                log_error(f"Failed to fetch data after {MAX_API_RETRIES} retries: {e}")
                raise RuntimeError(f"Network error fetching data: {e}")
                
        except ccxt.ExchangeError as e:
            log_error(f"Exchange error fetching data for {date.strftime('%Y-%m-%d')}: {e}")
            # CRASH on exchange errors - don't skip silently
            raise RuntimeError(f"Exchange error for {date.strftime('%Y-%m-%d')}: {e}")
            
        except Exception as e:
            log_error(f"Unexpected error fetching data for {date.strftime('%Y-%m-%d')}: {e}")
            raise
    
    def _save_data(self, df: pd.DataFrame, file_path: Path) -> None:
        """
        Save DataFrame to Parquet file.
        
        Args:
            df: DataFrame to save
            file_path: Path to save to
        """
        try:
            df.to_parquet(file_path, index=False, compression='snappy')
            log_debug(f"Saved data to {file_path.name}")
        except Exception as e:
            log_error(f"Failed to save data to {file_path.name}: {e}")
            raise
    
    def fetch_data_range(
        self,
        pair: str,
        timeframe: str,
        total_days: int,
        end_date: Optional[datetime] = None
    ) -> List[Path]:
        """
        Fetch data for a range of days, using cached files when available.
        
        Args:
            pair: Trading pair (validated)
            timeframe: Timeframe string (validated)
            total_days: Number of days to fetch
            end_date: End date (exclusive), defaults to today
            
        Returns:
            List of file paths containing the data
            
        Raises:
            ValueError: If parameters invalid
            RuntimeError: If data fetching fails
        """
        # Validate inputs
        pair = validate_pair(pair)
        timeframe = validate_timeframe(timeframe)
        total_days = validate_int(total_days, "total_days", min_val=1, max_val=1000)
        
        # Default to yesterday (exclude today's incomplete data)
        if end_date is None:
            end_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        
        log_info(f"Fetching {total_days} days of {pair} {timeframe} data ending {end_date.strftime('%Y-%m-%d')}")
        
        file_paths = []
        dates_to_fetch = []
        
        # Check which files exist and which need to be fetched
        for i in range(total_days):
            date = end_date - timedelta(days=i + 1)
            file_path = self._get_file_path(pair, timeframe, date)
            
            if self._file_exists(file_path):
                log_debug(f"Found cached data for {date.strftime('%Y-%m-%d')}")
                file_paths.append(file_path)
            else:
                dates_to_fetch.append((date, file_path))
        
        # Fetch missing data
        if dates_to_fetch:
            log_info(f"Need to fetch {len(dates_to_fetch)} missing days")
            
            for date, file_path in dates_to_fetch:
                # Rate limiting
                time.sleep(API_REQUEST_DELAY)
                
                df = self._fetch_ohlcv_for_date(pair, timeframe, date)
                
                # CRASH if data fetch fails - no silent skipping
                if df is None or df.empty:
                    raise RuntimeError(f"Failed to fetch data for {date.strftime('%Y-%m-%d')}: No data returned")
                
                self._save_data(df, file_path)
                file_paths.append(file_path)
        
        # Sort file paths by date (ascending)
        file_paths.sort()
        
        if not file_paths:
            raise RuntimeError(f"No data available for {pair} {timeframe}")
        
        log_info(f"Total {len(file_paths)} data files available")
        return file_paths
    
    def calculate_required_days(
        self,
        backtest_days: int,
        cycles: int,
        buffer_multiplier: float = 1.2
    ) -> int:
        """
        Calculate total days of data needed for non-overlapping cycles + buffer.
        
        Args:
            backtest_days: Days per backtest cycle
            cycles: Number of non-overlapping cycles
            buffer_multiplier: Extra 20% buffer for indicator lookback (default 1.2)
            
        Returns:
            Total days required (cycles * days_per_cycle * 1.2)
        """
        backtest_days = validate_int(backtest_days, "backtest_days", min_val=1)
        cycles = validate_int(cycles, "cycles", min_val=1)
        
        # For non-overlapping cycles: need cycles × days_per_cycle + 20% buffer
        base_days = backtest_days * cycles
        total_days = int(base_days * buffer_multiplier)
        
        log_debug(f"Calculated required days: {total_days} (base: {base_days}, buffer: {buffer_multiplier}x)")
        return total_days
    
    def get_available_pairs(self) -> List[str]:
        """
        Get list of available trading pairs from Kucoin Futures.
        
        Returns:
            List of trading pair symbols
        """
        try:
            markets = self.exchange.load_markets()
            futures_pairs = [
                symbol for symbol, market in markets.items()
                if market.get('type') == 'future' and market.get('active', False)
            ]
            log_info(f"Found {len(futures_pairs)} active futures pairs")
            return sorted(futures_pairs)
        except Exception as e:
            log_error(f"Failed to fetch available pairs: {e}")
            raise RuntimeError(f"Cannot fetch available pairs: {e}")
    
    def validate_pair_exists(self, pair: str) -> bool:
        """
        Validate that a trading pair exists on Kucoin Futures.
        
        Args:
            pair: Trading pair to validate
            
        Returns:
            True if pair exists
            
        Raises:
            ValueError: If pair doesn't exist
        """
        available_pairs = self.get_available_pairs()
        
        if pair not in available_pairs:
            raise ValueError(
                f"Trading pair '{pair}' not available on Kucoin Futures. "
                f"Available pairs: {', '.join(available_pairs[:10])}..."
            )
        
        return True
