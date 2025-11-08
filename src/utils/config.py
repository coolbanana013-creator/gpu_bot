"""
Configuration constants for the trading bot.
Contains Kucoin Futures specifications and global settings.
"""
from typing import Dict, List

# ============================================================================
# EXCHANGE CONFIGURATION (Kucoin Futures)
# ============================================================================

# Trading fees
MAKER_FEE_RATE = 0.0002  # 0.02%
TAKER_FEE_RATE = 0.0002  # 0.02%
DEFAULT_FEE_RATE = TAKER_FEE_RATE  # Conservative assumption

# Slippage model
SLIPPAGE_RATE = 0.001  # 0.1% random slippage

# Funding rate (approximate, actual varies)
FUNDING_RATE_8H = 0.0001  # 0.01% every 8 hours
FUNDING_INTERVAL_HOURS = 8

# Leverage limits
MIN_LEVERAGE = 1
MAX_LEVERAGE = 125  # Kucoin Futures maximum

# Margin requirements
INITIAL_MARGIN_RATE = 0.01  # 1% for 100x leverage (1/leverage)
MAINTENANCE_MARGIN_RATE = 0.005  # 0.5% typical maintenance margin
LIQUIDATION_BUFFER = 0.001  # 0.1% buffer above maintenance margin

# Position limits
MAX_POSITIONS_PER_BOT = 10  # Maximum concurrent positions

# ============================================================================
# TIMEFRAMES
# ============================================================================

VALID_TIMEFRAMES: List[str] = [
    '1m',   # 1 minute
    '5m',   # 5 minutes
    '15m',  # 15 minutes
    '30m',  # 30 minutes
    '1h',   # 1 hour
    '4h',   # 4 hours
    '1d'    # 1 day
]

# Timeframe to milliseconds conversion
TIMEFRAME_TO_MS: Dict[str, int] = {
    '1m': 60 * 1000,
    '5m': 5 * 60 * 1000,
    '15m': 15 * 60 * 1000,
    '30m': 30 * 60 * 1000,
    '1h': 60 * 60 * 1000,
    '4h': 4 * 60 * 60 * 1000,
    '1d': 24 * 60 * 60 * 1000,
}

# Timeframe to seconds conversion
TIMEFRAME_TO_SECONDS: Dict[str, int] = {
    '1m': 60,
    '5m': 5 * 60,
    '15m': 15 * 60,
    '30m': 30 * 60,
    '1h': 60 * 60,
    '4h': 4 * 60 * 60,
    '1d': 24 * 60 * 60,
}

# ============================================================================
# GENETIC ALGORITHM CONFIGURATION
# ============================================================================

# Population constraints
MIN_POPULATION = 1000
MAX_POPULATION = 1000000
DEFAULT_POPULATION = 10000

# Generation constraints
MIN_GENERATIONS = 1
MAX_GENERATIONS = 100
DEFAULT_GENERATIONS = 10

# Cycle constraints
MIN_CYCLES = 1
MAX_CYCLES = 100
DEFAULT_CYCLES = 5

# Backtest period constraints
MIN_BACKTEST_DAYS = 1
MAX_BACKTEST_DAYS = 365
DEFAULT_BACKTEST_DAYS = 10  # Reasonable for data chunking testing

# Data buffer for indicator lookback
DATA_BUFFER_MULTIPLIER = 1.25  # 25% extra data for lookback periods

# ============================================================================
# INDICATOR CONFIGURATION
# ============================================================================

# Indicator count per bot
MIN_INDICATORS_PER_BOT = 1
MAX_INDICATORS_PER_BOT = 10
DEFAULT_MIN_INDICATORS = 1
DEFAULT_MAX_INDICATORS = 5

# Maximum lookback period for any indicator
MAX_INDICATOR_LOOKBACK = 200  # bars

# Consensus threshold for signals
SIGNAL_CONSENSUS_THRESHOLD = 0.75  # 75% of indicators must agree

# ============================================================================
# RISK MANAGEMENT CONFIGURATION
# ============================================================================

# Risk strategy count per bot
MIN_RISK_STRATEGIES_PER_BOT = 1
MAX_RISK_STRATEGIES_PER_BOT = 10
DEFAULT_MIN_RISK_STRATEGIES = 1
DEFAULT_MAX_RISK_STRATEGIES = 5

# Take Profit constraints (percentage)
MIN_TAKE_PROFIT_PCT = 1.0  # 1%
MAX_TAKE_PROFIT_PCT = 25.0  # 25%

# Stop Loss constraints (percentage)
MIN_STOP_LOSS_PCT = 0.5  # 0.5%
# MAX_STOP_LOSS_PCT = computed as min(TAKE_PROFIT / 2, reasonable_value)

# Position sizing constraints
MIN_POSITION_SIZE_PCT = 1.0  # 1% of balance
MAX_POSITION_SIZE_PCT = 100.0  # 100% of balance (with leverage)

# Minimum free balance to open position
MIN_FREE_BALANCE_PCT = 10.0  # 10% of initial balance must remain free

# ============================================================================
# DATA STORAGE CONFIGURATION
# ============================================================================

# Data directory
DATA_DIR = 'data'

# File naming pattern for market data
DATA_FILE_PATTERN = '{pair}_{timeframe}_{date}.parquet'

# Columns in OHLCV data
OHLCV_COLUMNS = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

# ============================================================================
# GPU CONFIGURATION
# ============================================================================

# Maximum VRAM usage target (bytes)
MAX_VRAM_USAGE_GB = 3.0
MAX_VRAM_USAGE_BYTES = int(MAX_VRAM_USAGE_GB * 1024 * 1024 * 1024)

# Batch sizes for GPU operations
GPU_BATCH_SIZE_BOTS = 10000  # Process 10k bots at once
GPU_BATCH_SIZE_DATA = 100000  # Process 100k data points at once

# OpenCL work group size (tunable per device)
OPENCL_WORK_GROUP_SIZE = 256

# ============================================================================
# VALIDATION CONFIGURATION
# ============================================================================

# Gap tolerance for data validation (multiplier of timeframe)
DATA_GAP_TOLERANCE = 1.5

# Minimum data points per day
MIN_DATA_POINTS_PER_DAY = {
    '1m': 1440 * 0.95,   # 95% of expected
    '5m': 288 * 0.95,
    '15m': 96 * 0.95,
    '30m': 48 * 0.95,
    '1h': 24 * 0.95,
    '4h': 6 * 0.95,
    '1d': 1,
}

# ============================================================================
# TRADING PAIRS
# ============================================================================

# Supported quote currencies
SUPPORTED_QUOTE_CURRENCIES = ['USDT', 'BUSD', 'USD']

# Default trading pair (KuCoin Futures perpetual swap - XBTUSDTM)
DEFAULT_TRADING_PAIR = 'XBTUSDTM'

# ============================================================================
# BALANCE CONFIGURATION
# ============================================================================

# Default initial balance
DEFAULT_INITIAL_BALANCE = 100.0

# Minimum initial balance
MIN_INITIAL_BALANCE = 10.0

# ============================================================================
# RESULTS CONFIGURATION
# ============================================================================

# Number of top bots to save
TOP_BOTS_COUNT = 10

# Results file
RESULTS_FILE = 'data/best_bots.json'

# ============================================================================
# RANDOM SEED CONFIGURATION
# ============================================================================

# Default random seed for reproducibility
DEFAULT_RANDOM_SEED = 42

# ============================================================================
# API RATE LIMITING
# ============================================================================

# Delay between API requests (seconds)
API_REQUEST_DELAY = 1.0

# Maximum retries for failed requests
MAX_API_RETRIES = 3

# ============================================================================
# MODE CONFIGURATION
# ============================================================================

AVAILABLE_MODES = [1, 2, 3, 4]
IMPLEMENTED_MODES = [1, 4]  # Mode 1 (GA) and Mode 4 (Single Bot) are implemented

MODE_DESCRIPTIONS = {
    1: "Genetic Algorithm Mode - Evolve trading bots over multiple generations",
    2: "Not Implemented",
    3: "Not Implemented",
    4: "Single Bot Backtest Mode - Test individual bots on historical data (supports loading saved bots)"
}
