"""
Rate limiter for API requests.
Enforces Kucoin Futures API rate limits: 30 requests per 3 seconds.
"""

import time
import threading
from collections import deque
from typing import Callable, Any
from .exceptions import RateLimitError


class RateLimiter:
    """
    Token bucket rate limiter with sliding window.
    
    Kucoin Futures rate limits:
    - Order placement: 30 requests per 3 seconds per UID
    - Other endpoints: 100 requests per 10 seconds per IP
    """
    
    def __init__(self, max_calls: int = 30, period: float = 3.0):
        """
        Initialize rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed
            period: Time period in seconds
        """
        self.max_calls = max_calls
        self.period = period
        self.calls = deque()
        self.lock = threading.Lock()
    
    def _clean_old_calls(self, now: float):
        """Remove calls outside the current window."""
        cutoff = now - self.period
        while self.calls and self.calls[0] < cutoff:
            self.calls.popleft()
    
    def can_call(self) -> tuple[bool, float]:
        """
        Check if a call can be made now.
        
        Returns:
            Tuple of (can_call, wait_time)
            - can_call: True if call is allowed
            - wait_time: Seconds to wait if call is not allowed
        """
        with self.lock:
            now = time.time()
            self._clean_old_calls(now)
            
            if len(self.calls) < self.max_calls:
                return True, 0.0
            
            # Calculate wait time
            oldest_call = self.calls[0]
            wait_time = (oldest_call + self.period) - now
            return False, max(0.0, wait_time)
    
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded."""
        can_call, wait_time = self.can_call()
        if not can_call:
            print(f"⏳ Rate limit: waiting {wait_time:.2f}s...")
            time.sleep(wait_time + 0.01)  # Add small buffer
    
    def record_call(self):
        """Record that a call was made."""
        with self.lock:
            now = time.time()
            self._clean_old_calls(now)
            self.calls.append(now)
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with rate limiting.
        
        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            Result of function call
        
        Raises:
            RateLimitError: If rate limit exceeded and wait would be too long
        """
        can_call, wait_time = self.can_call()
        
        if not can_call:
            if wait_time > 10:  # Don't wait more than 10 seconds
                raise RateLimitError(
                    f"Rate limit exceeded. Would need to wait {wait_time:.1f}s",
                    retry_after=int(wait_time)
                )
            self.wait_if_needed()
        
        self.record_call()
        return func(*args, **kwargs)
    
    def get_stats(self) -> dict:
        """Get rate limiter statistics."""
        with self.lock:
            now = time.time()
            self._clean_old_calls(now)
            return {
                'current_calls': len(self.calls),
                'max_calls': self.max_calls,
                'period': self.period,
                'utilization_pct': (len(self.calls) / self.max_calls) * 100,
                'calls_remaining': self.max_calls - len(self.calls)
            }


# Global rate limiters for different endpoint types
order_rate_limiter = RateLimiter(max_calls=30, period=3.0)  # 30 req/3s for orders
general_rate_limiter = RateLimiter(max_calls=100, period=10.0)  # 100 req/10s for other


def rate_limit_order(func: Callable) -> Callable:
    """Decorator for order-related API calls."""
    def wrapper(*args, **kwargs):
        return order_rate_limiter.call(func, *args, **kwargs)
    return wrapper


def rate_limit_general(func: Callable) -> Callable:
    """Decorator for general API calls."""
    def wrapper(*args, **kwargs):
        return general_rate_limiter.call(func, *args, **kwargs)
    return wrapper
