"""
Strict validation utilities for parameter checking and input validation.
All functions raise ValueError on invalid inputs - no silent failures.
"""
import logging
from typing import Any, List, Union, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gpu_bot.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def validate_int(
    value: Any,
    name: str,
    min_val: Optional[int] = None,
    max_val: Optional[int] = None,
    allow_none: bool = False
) -> int:
    """
    Validate integer parameter with strict type and range checking.
    
    Args:
        value: Value to validate
        name: Parameter name for error messages
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        allow_none: Whether None is acceptable
        
    Returns:
        Validated integer value
        
    Raises:
        ValueError: If validation fails
    """
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{name} cannot be None")
    
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}")
    
    if min_val is not None and value < min_val:
        raise ValueError(f"{name} must be >= {min_val}, got {value}")
    
    if max_val is not None and value > max_val:
        raise ValueError(f"{name} must be <= {max_val}, got {value}")
    
    logger.debug(f"Validated {name}={value}")
    return value


def validate_float(
    value: Any,
    name: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    allow_none: bool = False,
    strict_positive: bool = False
) -> float:
    """
    Validate float parameter with strict type and range checking.
    
    Args:
        value: Value to validate
        name: Parameter name for error messages
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        allow_none: Whether None is acceptable
        strict_positive: If True, value must be > 0 (not just >= 0)
        
    Returns:
        Validated float value
        
    Raises:
        ValueError: If validation fails
    """
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{name} cannot be None")
    
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a number, got {type(value).__name__}")
    
    value = float(value)
    
    if strict_positive and value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    
    if min_val is not None and value < min_val:
        raise ValueError(f"{name} must be >= {min_val}, got {value}")
    
    if max_val is not None and value > max_val:
        raise ValueError(f"{name} must be <= {max_val}, got {value}")
    
    logger.debug(f"Validated {name}={value}")
    return value


def validate_enum(
    value: Any,
    name: str,
    allowed_values: List[Any],
    case_sensitive: bool = True
) -> Any:
    """
    Validate that value is in a list of allowed values.
    
    Args:
        value: Value to validate
        name: Parameter name for error messages
        allowed_values: List of acceptable values
        case_sensitive: For strings, whether to check case-sensitively
        
    Returns:
        Validated value
        
    Raises:
        ValueError: If value not in allowed_values
    """
    if value is None:
        raise ValueError(f"{name} cannot be None")
    
    check_value = value
    check_list = allowed_values
    
    # For strings, handle case sensitivity
    if isinstance(value, str) and not case_sensitive:
        check_value = value.lower()
        check_list = [v.lower() if isinstance(v, str) else v for v in allowed_values]
    
    if check_value not in check_list:
        raise ValueError(
            f"{name} must be one of {allowed_values}, got '{value}'"
        )
    
    logger.debug(f"Validated {name}={value}")
    return value


def validate_string(
    value: Any,
    name: str,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    allow_empty: bool = False,
    allow_none: bool = False
) -> str:
    """
    Validate string parameter.
    
    Args:
        value: Value to validate
        name: Parameter name for error messages
        min_length: Minimum string length
        max_length: Maximum string length
        allow_empty: Whether empty string is acceptable
        allow_none: Whether None is acceptable
        
    Returns:
        Validated string
        
    Raises:
        ValueError: If validation fails
    """
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{name} cannot be None")
    
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string, got {type(value).__name__}")
    
    if not allow_empty and len(value) == 0:
        raise ValueError(f"{name} cannot be empty")
    
    if min_length is not None and len(value) < min_length:
        raise ValueError(f"{name} must be at least {min_length} characters, got {len(value)}")
    
    if max_length is not None and len(value) > max_length:
        raise ValueError(f"{name} must be at most {max_length} characters, got {len(value)}")
    
    logger.debug(f"Validated {name}='{value}'")
    return value


def validate_range(
    min_val: Union[int, float],
    max_val: Union[int, float],
    name: str
) -> tuple:
    """
    Validate that min <= max for a range.
    
    Args:
        min_val: Minimum value
        max_val: Maximum value
        name: Range name for error messages
        
    Returns:
        Tuple of (min_val, max_val)
        
    Raises:
        ValueError: If min > max
    """
    if min_val > max_val:
        raise ValueError(
            f"{name}: min ({min_val}) cannot be greater than max ({max_val})"
        )
    
    logger.debug(f"Validated range {name}: [{min_val}, {max_val}]")
    return min_val, max_val


def validate_population_size(population: int) -> int:
    """
    Validate population size for genetic algorithm.
    
    Args:
        population: Population size
        
    Returns:
        Validated population
        
    Raises:
        ValueError: If population not in valid range [1000, 1000000]
    """
    return validate_int(
        population,
        "population",
        min_val=1000,
        max_val=1000000
    )


def validate_timeframe(timeframe: str) -> str:
    """
    Validate trading timeframe.
    
    Args:
        timeframe: Timeframe string
        
    Returns:
        Validated timeframe
        
    Raises:
        ValueError: If timeframe not supported
    """
    valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
    return validate_enum(
        timeframe,
        "timeframe",
        valid_timeframes,
        case_sensitive=False
    )


def validate_leverage(leverage: int) -> int:
    """
    Validate leverage value for Kucoin Futures.
    
    Args:
        leverage: Leverage multiplier
        
    Returns:
        Validated leverage
        
    Raises:
        ValueError: If leverage not in range [1, 125]
    """
    return validate_int(
        leverage,
        "leverage",
        min_val=1,
        max_val=125
    )


def validate_date(date_str: str, name: str = "date") -> datetime:
    """
    Validate and parse date string in ISO format (YYYY-MM-DD).
    
    Args:
        date_str: Date string to parse
        name: Parameter name for error messages
        
    Returns:
        Parsed datetime object
        
    Raises:
        ValueError: If date format invalid
    """
    try:
        dt = datetime.fromisoformat(date_str)
        logger.debug(f"Validated {name}={date_str}")
        return dt
    except (ValueError, TypeError) as e:
        raise ValueError(f"{name} must be in YYYY-MM-DD format, got '{date_str}': {e}")


def validate_percentage(
    value: float,
    name: str,
    min_val: float = 0.0,
    max_val: float = 100.0
) -> float:
    """
    Validate percentage value.
    
    Args:
        value: Percentage value
        name: Parameter name for error messages
        min_val: Minimum percentage (default 0)
        max_val: Maximum percentage (default 100)
        
    Returns:
        Validated percentage
        
    Raises:
        ValueError: If percentage out of range
    """
    return validate_float(
        value,
        name,
        min_val=min_val,
        max_val=max_val
    )


def validate_pair(pair: str) -> str:
    """
    Validate trading pair format.
    
    Args:
        pair: Trading pair (e.g., "BTC/USDT")
        
    Returns:
        Validated and normalized pair
        
    Raises:
        ValueError: If pair format invalid
    """
    pair = validate_string(pair, "pair", min_length=5)
    
    # Support both standard format (BTC/USDT) and KuCoin Futures format (XBTUSDTM)
    if '/' in pair:
        # Standard format: BASE/QUOTE or BASE/QUOTE:SETTLE
        parts = pair.split('/')
        if len(parts) != 2:
            raise ValueError(f"Trading pair must be in format BASE/QUOTE, got '{pair}'")
        
        base, quote = parts
        if not base or not quote:
            raise ValueError(f"Invalid trading pair format: '{pair}'")
        
        # Normalize to uppercase
        normalized = f"{base.upper()}/{quote.upper()}"
    else:
        # KuCoin Futures format: XBTUSDTM (no slash)
        # Just validate it's not empty and normalize to uppercase
        if not pair:
            raise ValueError(f"Invalid trading pair format: '{pair}'")
        normalized = pair.upper()
    
    logger.debug(f"Validated pair: {pair} -> {normalized}")
    return normalized


def log_info(message: str) -> None:
    """Log info message."""
    logger.info(message)


def log_error(message: str) -> None:
    """Log error message."""
    logger.error(message)


def log_warning(message: str) -> None:
    """Log warning message."""
    logger.warning(message)


def log_debug(message: str) -> None:
    """Log debug message."""
    logger.debug(message)
