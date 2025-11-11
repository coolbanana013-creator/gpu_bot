"""
Test Suite Runner

Runs all paper/live trading implementation tests:
1. Dynamic Slippage Test
2. Margin Trading Test
3. Signal Consensus Test
4. Integration Test

Reports overall success/failure.
"""

import sys
import subprocess
from pathlib import Path


def run_test(test_name: str, test_file: str) -> bool:
    """
    Run a single test script.
    
    Args:
        test_name: Display name of test
        test_file: Path to test script
    
    Returns:
        True if test passed, False otherwise
    """
    print("\n" + "="*80)
    print(f"RUNNING: {test_name}")
    print("="*80)
    
    try:
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print(f"\n✅ {test_name} PASSED")
            return True
        else:
            print(f"\n❌ {test_name} FAILED (exit code: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"\n❌ {test_name} ERROR: {e}")
        return False


def main():
    """Run all tests."""
    print("="*80)
    print("GPU BOT - PAPER/LIVE TRADING TEST SUITE")
    print("="*80)
    print("Running comprehensive tests to validate GPU kernel port...")
    print("="*80)
    
    # Get test directory
    test_dir = Path(__file__).parent
    
    # Define all tests
    tests = [
        ("Dynamic Slippage Test", test_dir / "test_slippage.py"),
        ("Margin Trading Test", test_dir / "test_margin.py"),
        ("Signal Consensus Test", test_dir / "test_signals.py"),
        ("Integration Test", test_dir / "test_integration.py")
    ]
    
    # Run all tests
    results = []
    for test_name, test_file in tests:
        if not test_file.exists():
            print(f"\n⚠️  {test_name} not found: {test_file}")
            results.append((test_name, False))
            continue
        
        success = run_test(test_name, str(test_file))
        results.append((test_name, success))
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, success in results if success)
    failed = len(results) - passed
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:10} {test_name}")
    
    print("="*80)
    print(f"Total: {len(results)} tests, {passed} passed, {failed} failed")
    print("="*80)
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED - GPU kernel port validated successfully!")
        print("Paper/live trading implementation ready for deployment.\n")
        return True
    else:
        print(f"\n⚠️  {failed} TEST(S) FAILED - Review errors above.")
        print("Fix issues before deploying paper/live trading.\n")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
