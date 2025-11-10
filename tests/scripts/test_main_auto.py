"""
Automated test for main.py - Mode 1 (Evolution).
Runs a minimal evolution to verify end-to-end workflow.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from pathlib import Path

def test_main_auto():
    """Test main.py with minimal parameters."""
    
    print("\n" + "="*80)
    print("AUTOMATED MAIN.PY TEST (Mode 1 - Evolution)")
    print("="*80 + "\n")
    
    # Create config for multi-cycle testing
    # 10 cycles × 1m timeframe × 7 days = excellent test of multi-cycle optimization
    # 7 days at 1m = 10,080 bars, plenty for 10 cycles
    # Fixed: Now only pre-computes 3-5 indicator combinations (not 6+)
    config = {
        "mode": 1,
        "pair": "BTC/USDT",
        "timeframe": "1m",  # 1m for high bar count
        "cycles": 10,  # Test multi-cycle optimization!
        "backtest_days": 1,  # Each cycle uses 1 day (1440 bars)
        "population": 1000,  # Minimum allowed
        "generations": 2,
        "min_indicators": 3,
        "max_indicators": 5,  # Only pre-computes up to 5 now (fast!)
        "min_risk_strategies": 2,
        "max_risk_strategies": 5,
        "min_leverage": 1,
        "max_leverage": 10,
        "initial_balance": 1000,
        "random_seed": 42,
        "days": 7  # 7 days at 1m = 10,080 bars
    }
    
    # Save config
    config_dir = Path(__file__).parent.parent / "config"
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / "test_config.json"
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Created test config: {config_path}")
    print(f"\nConfig:")
    print(f"  Population: {config['population']} bots")
    print(f"  Generations: {config['generations']}")
    print(f"  Cycles: {config['cycles']}")
    print(f"  Backtest days per cycle: {config['backtest_days']}")
    print(f"  Data: {config['days']} days of {config['pair']} {config['timeframe']}")
    print(f"\nThis will test the multi-cycle optimization with minimal data.")
    print(f"Running main.py with automated config...")
    print("="*80 + "\n")
    
    # Run main.py as module with automated input
    import subprocess
    
    # Prepare all inputs in correct order based on main.py prompts
    inputs = [
        "1",  # Mode 1
        config['pair'],  # Trading pair
        str(config['initial_balance']),  # Initial balance
        str(config['population']),  # Population size
        str(config['generations']),  # Generations
        str(config['cycles']),  # Cycles per generation
        str(config['backtest_days']),  # Days per backtest cycle
        config['timeframe'],  # Timeframe
        str(config['min_leverage']),  # Min leverage
        str(config['max_leverage']),  # Max leverage
        str(config['min_indicators']),  # Min indicators
        str(config['max_indicators']),  # Max indicators
        str(config['min_risk_strategies']),  # Min risk strategies
        str(config['max_risk_strategies']),  # Max risk strategies
        "y",  # Use random seed? (y/n)
        str(config['random_seed']),  # Random seed value
        "n",  # Interactive mode? (n for automated)
        "",  # Days to load (empty to use all available)
    ]
    
    input_str = '\n'.join(inputs) + '\n'
    
    result = subprocess.run(
        ["python", "-m", "main"],
        input=input_str,
        text=True,
        capture_output=False,
        cwd=Path(__file__).parent.parent
    )
    
    if result.returncode == 0:
        print("\n" + "="*80)
        print("✅ MAIN.PY TEST PASSED")
        print("="*80)
        return True
    else:
        print("\n" + "="*80)
        print("❌ MAIN.PY TEST FAILED")
        print("="*80)
        return False

if __name__ == "__main__":
    try:
        success = test_main_auto()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
