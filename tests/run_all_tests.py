"""
Test runner script - executes all tests without user input.
"""
import subprocess
import sys
from pathlib import Path

def run_tests():
    """Run all test suites."""
    test_dir = Path(__file__).parent
    
    print("="*70)
    print("RUNNING GPU BOT COMPREHENSIVE TEST SUITE")
    print("="*70 + "\n")
    
    test_files = [
        "test_generator.py",
        "test_simulator.py",
        "test_workflow.py"
    ]
    
    all_passed = True
    results = {}
    
    for test_file in test_files:
        print(f"\n{'='*70}")
        print(f"Running: {test_file}")
        print('='*70)
        
        test_path = test_dir / test_file
        
        # Run pytest
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_path), "-v", "--tb=short"],
            capture_output=False
        )
        
        results[test_file] = result.returncode == 0
        
        if result.returncode != 0:
            all_passed = False
            print(f"\n[FAILED] {test_file}")
        else:
            print(f"\n[PASSED] {test_file}")
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_file, passed in results.items():
        status = "[PASSED]" if passed else "[FAILED]"
        print(f"{test_file:30} {status}")
    
    print("="*70)
    
    if all_passed:
        print("\n[SUCCESS] ALL TESTS PASSED!")
        return 0
    else:
        print("\n[WARNING] SOME TESTS FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(run_tests())
