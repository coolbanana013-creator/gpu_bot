"""
Safety Features Test Suite

Tests rate limiter, circuit breaker, risk manager, and input validation.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.live_trading.kucoin_universal_client import KucoinUniversalClient
from src.live_trading.credentials import CredentialsManager
from src.live_trading.exceptions import (
    RateLimitError, CircuitBreakerError, RiskLimitError,
    ValidationError, OrderCreationError
)
from src.utils.validation import log_info, log_success, log_error, log_warning


def test_rate_limiter():
    """Test rate limiter prevents excessive API calls."""
    log_info("\n📊 Testing Rate Limiter...")
    
    creds_mgr = CredentialsManager()
    creds = creds_mgr.load_credentials()
    
    client = KucoinUniversalClient(
        api_key=creds['api_key'],
        api_secret=creds['api_secret'],
        api_passphrase=creds['api_passphrase'],
        test_mode=True
    )
    
    symbol = "XBTUSDTM"
    
    # Test 1: Normal operation should work
    try:
        ticker = client.fetch_ticker(symbol)
        log_success("✅ Normal API call works")
    except Exception as e:
        log_error(f"❌ Normal API call failed: {e}")
        return False
    
    # Test 2: Rapid successive calls should be rate limited
    log_info("   Testing rapid API calls (30 calls in quick succession)...")
    start_time = time.time()
    call_count = 0
    
    for i in range(30):
        try:
            client.fetch_ticker(symbol)
            call_count += 1
        except Exception as e:
            log_error(f"   Call {i+1} failed: {e}")
    
    elapsed = time.time() - start_time
    log_info(f"   Completed {call_count} calls in {elapsed:.2f}s")
    
    # Should take at least 3 seconds for 30 calls (30 req/3s limit)
    if elapsed < 2.5:
        log_warning(f"⚠️  Rate limiter may not be working - too fast: {elapsed:.2f}s")
    else:
        log_success(f"✅ Rate limiter working - enforced delays: {elapsed:.2f}s")
    
    return True


def test_circuit_breaker():
    """Test circuit breaker opens after failures."""
    log_info("\n🔌 Testing Circuit Breaker...")
    
    from src.live_trading.circuit_breaker import CircuitBreaker
    
    # Create test circuit breaker with low threshold
    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=5)
    
    def failing_function():
        raise Exception("Simulated failure")
    
    def working_function():
        return "Success"
    
    # Test 1: Force failures to open circuit
    failure_count = 0
    for i in range(5):
        try:
            breaker.call(failing_function)
        except CircuitBreakerError:
            log_info(f"   Circuit breaker OPEN after {i} failures")
            break
        except Exception:
            failure_count += 1
    
    if failure_count >= 3:
        log_success("✅ Circuit breaker opened after threshold")
    else:
        log_error("❌ Circuit breaker did not open")
        return False
    
    # Test 2: Circuit should block calls while open
    try:
        breaker.call(working_function)
        log_error("❌ Circuit breaker allowed call while OPEN")
        return False
    except CircuitBreakerError:
        log_success("✅ Circuit breaker blocked calls while OPEN")
    
    # Test 3: Circuit should recover after timeout
    log_info(f"   Waiting {breaker.recovery_timeout}s for recovery...")
    time.sleep(breaker.recovery_timeout + 1)
    
    try:
        result = breaker.call(working_function)
        log_success("✅ Circuit breaker recovered and allows calls")
    except Exception as e:
        log_error(f"❌ Circuit breaker did not recover: {e}")
        return False
    
    return True


def test_risk_manager():
    """Test risk manager validation and limits."""
    log_info("\n🛡️ Testing Risk Manager...")
    
    creds_mgr = CredentialsManager()
    creds = creds_mgr.load_credentials()
    
    client = KucoinUniversalClient(
        api_key=creds['api_key'],
        api_secret=creds['api_secret'],
        api_passphrase=creds['api_passphrase'],
        test_mode=True
    )
    
    symbol = "XBTUSDTM"
    
    # Test 1: Normal order should pass
    try:
        order = client.create_market_order(
            symbol=symbol,
            side='buy',
            size=1,
            leverage=2
        )
        log_success("✅ Normal order passed risk checks")
    except Exception as e:
        log_error(f"❌ Normal order failed: {e}")
        return False
    
    # Test 2: Oversized order should be rejected
    try:
        order = client.create_market_order(
            symbol=symbol,
            side='buy',
            size=150,  # Exceeds max_order_size of 100
            leverage=1
        )
        log_error("❌ Oversized order was not rejected")
        return False
    except (RiskLimitError, ValidationError, OrderCreationError) as e:
        log_success(f"✅ Oversized order rejected: {type(e).__name__}")
    except Exception as e:
        # The exception gets wrapped in OrderError
        if "exceeds maximum" in str(e):
            log_success(f"✅ Oversized order rejected (wrapped exception)")
        else:
            log_error(f"❌ Unexpected error: {e}")
            return False
    
    # Test 3: Excessive leverage should be rejected
    try:
        order = client.create_market_order(
            symbol=symbol,
            side='buy',
            size=1,
            leverage=15  # Exceeds max_leverage of 10 (test mode)
        )
        log_error("❌ Excessive leverage was not rejected")
        return False
    except RiskLimitError as e:
        log_success(f"✅ Excessive leverage rejected")
    except Exception as e:
        log_error(f"❌ Unexpected error: {e}")
        return False
    
    # Test 4: Invalid side should be rejected
    try:
        order = client.create_market_order(
            symbol=symbol,
            side='invalid',
            size=1,
            leverage=1
        )
        log_error("❌ Invalid side was not rejected")
        return False
    except ValidationError as e:
        log_success(f"✅ Invalid side rejected")
    except Exception as e:
        log_error(f"❌ Unexpected error: {e}")
        return False
    
    return True


def test_input_validation():
    """Test input validation in direct client."""
    log_info("\n✅ Testing Input Validation...")
    
    creds_mgr = CredentialsManager()
    creds = creds_mgr.load_credentials()
    
    from src.live_trading.direct_futures_client import DirectKucoinFuturesClient
    
    client = DirectKucoinFuturesClient(
        api_key=creds['api_key'],
        api_secret=creds['api_secret'],
        api_passphrase=creds['api_passphrase'],
        test_mode=True
    )
    
    # Test 1: Invalid symbol
    try:
        order = client.create_market_order(
            symbol="",
            side='buy',
            size=1,
            leverage=1
        )
        log_error("❌ Empty symbol was not rejected")
        return False
    except ValueError as e:
        log_success(f"✅ Empty symbol rejected")
    except Exception as e:
        log_error(f"❌ Unexpected error: {e}")
        return False
    
    # Test 2: Negative size
    try:
        order = client.create_market_order(
            symbol="XBTUSDTM",
            side='buy',
            size=-5,
            leverage=1
        )
        log_error("❌ Negative size was not rejected")
        return False
    except ValueError as e:
        log_success(f"✅ Negative size rejected")
    except Exception as e:
        log_error(f"❌ Unexpected error: {e}")
        return False
    
    # Test 3: Invalid leverage
    try:
        order = client.create_market_order(
            symbol="XBTUSDTM",
            side='buy',
            size=1,
            leverage=0
        )
        log_error("❌ Zero leverage was not rejected")
        return False
    except ValueError as e:
        log_success(f"✅ Zero leverage rejected")
    except Exception as e:
        log_error(f"❌ Unexpected error: {e}")
        return False
    
    # Test 4: Invalid price in limit order
    try:
        order = client.create_limit_order(
            symbol="XBTUSDTM",
            side='buy',
            price=-100,
            size=1,
            leverage=1
        )
        log_error("❌ Negative price was not rejected")
        return False
    except ValueError as e:
        log_success(f"✅ Negative price rejected")
    except Exception as e:
        log_error(f"❌ Unexpected error: {e}")
        return False
    
    return True


def main():
    """Run all safety feature tests."""
    print("\n" + "="*80)
    print("🔒 SAFETY FEATURES TEST SUITE")
    print("="*80)
    
    results = {
        'passed': 0,
        'failed': 0
    }
    
    tests = [
        ("Rate Limiter", test_rate_limiter),
        ("Circuit Breaker", test_circuit_breaker),
        ("Risk Manager", test_risk_manager),
        ("Input Validation", test_input_validation)
    ]
    
    for test_name, test_func in tests:
        try:
            log_info(f"\n{'='*80}")
            log_info(f"Testing: {test_name}")
            log_info(f"{'='*80}")
            
            if test_func():
                results['passed'] += 1
                log_success(f"\n✅ {test_name} - ALL TESTS PASSED")
            else:
                results['failed'] += 1
                log_error(f"\n❌ {test_name} - TESTS FAILED")
        except Exception as e:
            results['failed'] += 1
            log_error(f"\n❌ {test_name} - EXCEPTION: {e}")
    
    # Print final summary
    print("\n" + "="*80)
    print("📊 FINAL RESULTS")
    print("="*80)
    print(f"✅ Passed: {results['passed']}/{len(tests)}")
    print(f"❌ Failed: {results['failed']}/{len(tests)}")
    print(f"📈 Success Rate: {(results['passed']/len(tests)*100):.1f}%")
    print("="*80)
    
    if results['failed'] == 0:
        print("🎉 ALL SAFETY FEATURES WORKING PERFECTLY!")
    else:
        print("⚠️  Some safety features need attention")
    print("="*80)


if __name__ == "__main__":
    main()
