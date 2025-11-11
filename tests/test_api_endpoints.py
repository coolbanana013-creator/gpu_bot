"""
Comprehensive API Endpoint Test Script

Tests all Kucoin API endpoints to ensure they work correctly.
Even if we get "insufficient balance" errors, we want to verify the API calls are functional.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.live_trading.kucoin_universal_client import KucoinUniversalClient
from src.utils.validation import log_info, log_error, log_success, log_warning
import json


class APIEndpointTester:
    """Test all Kucoin API endpoints."""
    
    def __init__(self, api_key: str, api_secret: str, api_passphrase: str, is_sandbox: bool = True):
        """
        Initialize tester.
        
        Args:
            api_key: Kucoin API key
            api_secret: Kucoin API secret
            api_passphrase: Kucoin API passphrase
            is_sandbox: Use sandbox environment (default True)
        """
        self.client = KucoinUniversalClient(
            api_key=api_key,
            api_secret=api_secret,
            api_passphrase=api_passphrase,
            is_sandbox=is_sandbox
        )
        self.symbol = "BTC-USDT"
        self.test_results = {}
    
    async def test_all_endpoints(self):
        """Run all endpoint tests."""
        log_info("="*80)
        log_info("KUCOIN API ENDPOINT COMPREHENSIVE TEST")
        log_info("="*80)
        
        # Test categories
        await self.test_account_endpoints()
        await self.test_market_data_endpoints()
        await self.test_trading_endpoints()
        
        # Print summary
        self.print_summary()
    
    async def test_account_endpoints(self):
        """Test account-related endpoints."""
        log_info("\n📊 TESTING ACCOUNT ENDPOINTS")
        log_info("-"*80)
        
        # Test 1: Get Account Info
        await self.run_test(
            "Get Account Info",
            self.client.get_account_info()
        )
        
        # Test 2: Get Account Balance
        await self.run_test(
            "Get Account Balance",
            self.client.get_account_balance()
        )
        
        # Test 3: Get Futures Account Overview
        await self.run_test(
            "Get Futures Account Overview",
            self.client.get_futures_account_overview()
        )
        
        # Test 4: Get Position Details
        await self.run_test(
            "Get Position Details",
            self.client.get_position_details(self.symbol)
        )
    
    async def test_market_data_endpoints(self):
        """Test market data endpoints."""
        log_info("\n📈 TESTING MARKET DATA ENDPOINTS")
        log_info("-"*80)
        
        # Test 1: Get Ticker
        await self.run_test(
            "Get Ticker",
            self.client.get_ticker(self.symbol)
        )
        
        # Test 2: Get 24h Stats
        await self.run_test(
            "Get 24h Stats",
            self.client.get_24h_stats(self.symbol)
        )
        
        # Test 3: Get Klines (OHLCV)
        await self.run_test(
            "Get Klines (1min, 100 bars)",
            self.client.get_klines(self.symbol, "1min", limit=100)
        )
        
        # Test 4: Get Order Book
        await self.run_test(
            "Get Order Book (depth 20)",
            self.client.get_order_book(self.symbol, depth=20)
        )
        
        # Test 5: Get Funding Rate
        await self.run_test(
            "Get Funding Rate",
            self.client.get_funding_rate(self.symbol)
        )
        
        # Test 6: Get Contract Details
        await self.run_test(
            "Get Contract Details",
            self.client.get_contract_details(self.symbol)
        )
    
    async def test_trading_endpoints(self):
        """Test trading endpoints (will likely fail with insufficient balance)."""
        log_info("\n⚡ TESTING TRADING ENDPOINTS")
        log_info("-"*80)
        log_warning("NOTE: These may fail with 'insufficient balance' - that's OK!")
        log_warning("We're testing if the API calls work, not if we have funds.")
        
        # Get current price for test orders
        ticker = await self.client.get_ticker(self.symbol)
        if ticker and 'price' in ticker:
            current_price = float(ticker['price'])
            
            # Test 1: Place Market Buy Order (tiny size)
            await self.run_test(
                "Place Market Buy Order (0.001 BTC)",
                self.client.place_market_order(
                    symbol=self.symbol,
                    side="buy",
                    size=0.001,
                    leverage=1
                ),
                expect_failure=True
            )
            
            # Test 2: Place Limit Buy Order (far from price)
            limit_price = current_price * 0.5  # 50% below current
            await self.run_test(
                f"Place Limit Buy Order (price: ${limit_price:.2f})",
                self.client.place_limit_order(
                    symbol=self.symbol,
                    side="buy",
                    price=limit_price,
                    size=0.001,
                    leverage=1
                ),
                expect_failure=True
            )
            
            # Test 3: Place Stop Market Order
            stop_price = current_price * 0.9  # 10% below
            await self.run_test(
                f"Place Stop Market Order (stop: ${stop_price:.2f})",
                self.client.place_stop_market_order(
                    symbol=self.symbol,
                    side="sell",
                    size=0.001,
                    stop_price=stop_price,
                    leverage=1
                ),
                expect_failure=True
            )
            
            # Test 4: Get Open Orders
            await self.run_test(
                "Get Open Orders",
                self.client.get_open_orders(self.symbol)
            )
            
            # Test 5: Get Order History
            await self.run_test(
                "Get Order History (last 10)",
                self.client.get_order_history(self.symbol, limit=10)
            )
        else:
            log_error("Could not get current price - skipping order tests")
    
    async def run_test(self, test_name: str, coroutine, expect_failure: bool = False):
        """
        Run a single test.
        
        Args:
            test_name: Name of the test
            coroutine: Async function to test
            expect_failure: If True, failure is acceptable (e.g., insufficient balance)
        """
        try:
            result = await coroutine
            
            if result is not None:
                log_success(f"✅ {test_name}: PASSED")
                
                # Show sample of result
                if isinstance(result, dict):
                    log_info(f"   Sample: {json.dumps(result, indent=2)[:200]}...")
                elif isinstance(result, list) and len(result) > 0:
                    log_info(f"   Returned {len(result)} items")
                
                self.test_results[test_name] = {"status": "PASSED", "result": result}
            else:
                if expect_failure:
                    log_warning(f"⚠️  {test_name}: No result (expected - insufficient funds?)")
                    self.test_results[test_name] = {"status": "EXPECTED_FAILURE", "result": None}
                else:
                    log_error(f"❌ {test_name}: FAILED (returned None)")
                    self.test_results[test_name] = {"status": "FAILED", "error": "Returned None"}
        
        except Exception as e:
            error_msg = str(e)
            
            # Check if it's an expected error
            expected_errors = [
                "insufficient balance",
                "insufficient funds",
                "balance not sufficient",
                "order size below minimum",
                "account has no permission"
            ]
            
            is_expected = any(err in error_msg.lower() for err in expected_errors)
            
            if is_expected and expect_failure:
                log_warning(f"⚠️  {test_name}: Expected failure - {error_msg}")
                self.test_results[test_name] = {"status": "EXPECTED_FAILURE", "error": error_msg}
            elif is_expected:
                log_warning(f"⚠️  {test_name}: API works but insufficient funds - {error_msg}")
                self.test_results[test_name] = {"status": "API_WORKS", "error": error_msg}
            else:
                log_error(f"❌ {test_name}: FAILED - {error_msg}")
                self.test_results[test_name] = {"status": "FAILED", "error": error_msg}
    
    def print_summary(self):
        """Print test summary."""
        log_info("\n" + "="*80)
        log_info("TEST SUMMARY")
        log_info("="*80)
        
        passed = sum(1 for r in self.test_results.values() if r['status'] == 'PASSED')
        api_works = sum(1 for r in self.test_results.values() if r['status'] == 'API_WORKS')
        expected_fail = sum(1 for r in self.test_results.values() if r['status'] == 'EXPECTED_FAILURE')
        failed = sum(1 for r in self.test_results.values() if r['status'] == 'FAILED')
        total = len(self.test_results)
        
        log_info(f"Total Tests: {total}")
        log_success(f"✅ Passed: {passed}")
        log_success(f"✅ API Works (insufficient funds): {api_works}")
        log_warning(f"⚠️  Expected Failures: {expected_fail}")
        log_error(f"❌ Failed: {failed}")
        
        # Overall verdict
        if failed == 0:
            log_success("\n🎉 ALL TESTS PASSED! API endpoints are functional.")
        elif failed <= 2:
            log_warning("\n⚠️  MOSTLY PASSED with minor issues.")
        else:
            log_error("\n❌ MULTIPLE FAILURES - API may have issues.")
        
        # Detailed results
        log_info("\n" + "="*80)
        log_info("DETAILED RESULTS")
        log_info("="*80)
        
        for test_name, result in self.test_results.items():
            status = result['status']
            if status == 'PASSED':
                log_success(f"✅ {test_name}")
            elif status == 'API_WORKS':
                log_success(f"✅ {test_name} (API functional)")
            elif status == 'EXPECTED_FAILURE':
                log_warning(f"⚠️  {test_name} (expected)")
            else:
                log_error(f"❌ {test_name}")
                if 'error' in result:
                    log_error(f"   Error: {result['error']}")


async def main():
    """Main test function."""
    import os
    
    # Get API credentials from environment
    api_key = os.getenv('KUCOIN_API_KEY', '')
    api_secret = os.getenv('KUCOIN_API_SECRET', '')
    api_passphrase = os.getenv('KUCOIN_API_PASSPHRASE', '')
    is_sandbox = os.getenv('KUCOIN_SANDBOX', 'true').lower() == 'true'
    
    if not all([api_key, api_secret, api_passphrase]):
        log_error("ERROR: API credentials not found in environment variables!")
        log_info("\nPlease set the following environment variables:")
        log_info("  KUCOIN_API_KEY")
        log_info("  KUCOIN_API_SECRET")
        log_info("  KUCOIN_API_PASSPHRASE")
        log_info("  KUCOIN_SANDBOX (optional, default: true)")
        log_info("\nExample:")
        log_info("  $env:KUCOIN_API_KEY='your_key'")
        log_info("  $env:KUCOIN_API_SECRET='your_secret'")
        log_info("  $env:KUCOIN_API_PASSPHRASE='your_passphrase'")
        return
    
    # Create tester
    tester = APIEndpointTester(
        api_key=api_key,
        api_secret=api_secret,
        api_passphrase=api_passphrase,
        is_sandbox=is_sandbox
    )
    
    # Run all tests
    await tester.test_all_endpoints()


if __name__ == "__main__":
    asyncio.run(main())
