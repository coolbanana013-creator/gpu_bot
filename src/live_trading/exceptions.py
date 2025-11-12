"""
Custom exceptions for GPU Bot trading system.
Provides specific exception types for better error handling and debugging.
"""


class TradingError(Exception):
    """Base exception for all trading-related errors."""
    pass


class OrderError(TradingError):
    """Exception raised for order-related errors."""
    
    def __init__(self, message: str, order_id: str = None, code: str = None):
        self.order_id = order_id
        self.code = code
        super().__init__(message)


class OrderCreationError(OrderError):
    """Exception raised when order creation fails."""
    pass


class OrderCancellationError(OrderError):
    """Exception raised when order cancellation fails."""
    pass


class AuthenticationError(TradingError):
    """Exception raised for authentication failures."""
    
    def __init__(self, message: str, code: str = None):
        self.code = code
        super().__init__(message)


class TimestampError(AuthenticationError):
    """Exception raised for timestamp-related auth failures."""
    pass


class NetworkError(TradingError):
    """Exception raised for network-related errors."""
    
    def __init__(self, message: str, retry_count: int = 0):
        self.retry_count = retry_count
        super().__init__(message)


class TimeoutError(NetworkError):
    """Exception raised when request times out."""
    pass


class ConnectionError(NetworkError):
    """Exception raised when connection fails."""
    pass


class RateLimitError(TradingError):
    """Exception raised when rate limit is exceeded."""
    
    def __init__(self, message: str, retry_after: int = None):
        self.retry_after = retry_after
        super().__init__(message)


class ValidationError(TradingError):
    """Exception raised for input validation failures."""
    
    def __init__(self, message: str, field: str = None, value=None):
        self.field = field
        self.value = value
        super().__init__(message)


class PositionError(TradingError):
    """Exception raised for position-related errors."""
    pass


class LiquidationRiskError(PositionError):
    """Exception raised when position is near liquidation."""
    
    def __init__(self, message: str, distance_pct: float = None):
        self.distance_pct = distance_pct
        super().__init__(message)


class InsufficientMarginError(TradingError):
    """Exception raised when insufficient margin for operation."""
    
    def __init__(self, message: str, required: float = None, available: float = None):
        self.required = required
        self.available = available
        super().__init__(message)


class RiskLimitError(TradingError):
    """Exception raised when risk limits are exceeded."""
    
    def __init__(self, message: str, limit_type: str = None, current_value=None, max_value=None):
        self.limit_type = limit_type
        self.current_value = current_value
        self.max_value = max_value
        super().__init__(message)


class CircuitBreakerError(TradingError):
    """Exception raised when circuit breaker is open."""
    
    def __init__(self, message: str, failure_count: int = None, retry_after: int = None):
        self.failure_count = failure_count
        self.retry_after = retry_after
        super().__init__(message)


class KucoinAPIError(TradingError):
    """Exception raised for Kucoin API errors."""
    
    def __init__(self, message: str, code: str = None, response: dict = None):
        self.code = code
        self.response = response
        super().__init__(f"Kucoin API Error {code}: {message}" if code else message)
