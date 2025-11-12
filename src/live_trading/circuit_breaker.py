"""
Circuit breaker pattern implementation.
Prevents cascading failures by stopping requests after repeated failures.
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Callable, Any
from enum import Enum
from .exceptions import CircuitBreakerError


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "CLOSED"  # Normal operation
    OPEN = "OPEN"  # Blocking all requests
    HALF_OPEN = "HALF_OPEN"  # Testing if service recovered


class CircuitBreaker:
    """
    Circuit breaker to prevent cascading failures.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, blocking requests
    - HALF_OPEN: Testing recovery after timeout
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before trying again
            expected_exception: Exception type to count as failure
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self.lock = threading.Lock()
        
        # Statistics
        self.total_calls = 0
        self.total_failures = 0
        self.total_successes = 0
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        
        elapsed = time.time() - self.last_failure_time
        return elapsed >= self.recovery_timeout
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            Result of function call
        
        Raises:
            CircuitBreakerError: If circuit is open
            Exception: Any exception raised by func
        """
        with self.lock:
            self.total_calls += 1
            
            # Check if circuit is open
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    print(f"🔄 Circuit breaker: Attempting recovery...")
                    self.state = CircuitState.HALF_OPEN
                else:
                    wait_time = self.recovery_timeout - (time.time() - self.last_failure_time)
                    raise CircuitBreakerError(
                        f"Circuit breaker OPEN. Service unavailable. Retry in {wait_time:.0f}s",
                        failure_count=self.failure_count,
                        retry_after=int(wait_time)
                    )
        
        # Attempt the call
        try:
            result = func(*args, **kwargs)
            
            # Success - reset failure count
            with self.lock:
                if self.state == CircuitState.HALF_OPEN:
                    print(f"✅ Circuit breaker: Recovery successful, circuit CLOSED")
                
                self.failure_count = 0
                self.state = CircuitState.CLOSED
                self.total_successes += 1
            
            return result
            
        except self.expected_exception as e:
            # Failure - increment count
            with self.lock:
                self.failure_count += 1
                self.total_failures += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    if self.state != CircuitState.OPEN:
                        print(f"🔴 Circuit breaker OPEN after {self.failure_count} failures")
                    self.state = CircuitState.OPEN
                else:
                    print(f"⚠️  Circuit breaker: Failure {self.failure_count}/{self.failure_threshold}")
            
            # Re-raise the exception
            raise
    
    def reset(self):
        """Manually reset the circuit breaker."""
        with self.lock:
            print(f"🔄 Circuit breaker manually reset")
            self.failure_count = 0
            self.state = CircuitState.CLOSED
            self.last_failure_time = None
    
    def get_stats(self) -> dict:
        """Get circuit breaker statistics."""
        with self.lock:
            return {
                'state': self.state.value,
                'failure_count': self.failure_count,
                'failure_threshold': self.failure_threshold,
                'total_calls': self.total_calls,
                'total_successes': self.total_successes,
                'total_failures': self.total_failures,
                'success_rate': (self.total_successes / max(1, self.total_calls)) * 100,
                'last_failure': datetime.fromtimestamp(self.last_failure_time).isoformat() 
                    if self.last_failure_time else None
            }


# Global circuit breakers for different operations
api_circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
order_circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)


def circuit_break(breaker: CircuitBreaker = None):
    """
    Decorator to add circuit breaker protection.
    
    Args:
        breaker: CircuitBreaker instance to use (default: api_circuit_breaker)
    """
    if breaker is None:
        breaker = api_circuit_breaker
    
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)
        return wrapper
    return decorator
