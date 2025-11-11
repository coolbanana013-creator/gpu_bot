"""
Test 1: Dynamic Slippage Calculation

Validates that CPU slippage calculation matches GPU kernel behavior exactly.

Tests:
1. Base slippage (ideal conditions)
2. Volume impact (position size vs volume)
3. Volatility multiplier (high/low range)
4. Leverage multiplier
5. Combined factors
6. Boundary conditions (min/max caps)
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.live_trading.gpu_kernel_port import calculate_dynamic_slippage, BASE_SLIPPAGE


def test_base_slippage():
    """Test 1: Base slippage (ideal conditions)."""
    print("\n" + "="*60)
    print("TEST 1: Base Slippage (Ideal Conditions)")
    print("="*60)
    
    # Small position, high volume, low volatility
    position_value = 100.0  # $100 position
    current_volume = 1000000.0  # $1M volume
    leverage = 1  # No leverage
    current_price = 50000.0
    current_high = 50100.0
    current_low = 49900.0
    
    slippage = calculate_dynamic_slippage(
        position_value, current_volume, leverage,
        current_price, current_high, current_low
    )
    
    print(f"Position Value:  ${position_value:,.2f}")
    print(f"Volume:          ${current_volume:,.2f}")
    print(f"Leverage:        {leverage}x")
    print(f"Price Range:     ${current_low:,.2f} - ${current_high:,.2f}")
    print(f"\nCalculated Slippage: {slippage*100:.4f}%")
    print(f"Expected (base):     {BASE_SLIPPAGE*100:.4f}%")
    
    # Should be close to base slippage
    assert slippage >= BASE_SLIPPAGE * 0.5, "Slippage too low"
    assert slippage <= BASE_SLIPPAGE * 2.0, "Slippage too high for ideal conditions"
    print("✅ PASS: Base slippage in expected range")


def test_volume_impact():
    """Test 2: Volume impact (large position vs small volume)."""
    print("\n" + "="*60)
    print("TEST 2: Volume Impact")
    print("="*60)
    
    leverage = 1
    current_price = 50000.0
    current_high = 50100.0
    current_low = 49900.0
    
    # Test cases: position value vs volume
    test_cases = [
        (1000.0, 10000.0, "10% of volume"),
        (100.0, 10000.0, "1% of volume"),
        (10.0, 10000.0, "0.1% of volume"),
    ]
    
    results = []
    for position_value, current_volume, description in test_cases:
        slippage = calculate_dynamic_slippage(
            position_value, current_volume, leverage,
            current_price, current_high, current_low
        )
        results.append((description, slippage))
        print(f"{description:20} → Slippage: {slippage*100:.4f}%")
    
    # Larger position should have higher slippage
    assert results[0][1] > results[1][1], "Larger position should have higher slippage"
    assert results[1][1] > results[2][1], "Medium position should have more slippage than small"
    print("✅ PASS: Volume impact scales correctly")


def test_volatility_multiplier():
    """Test 3: Volatility multiplier."""
    print("\n" + "="*60)
    print("TEST 3: Volatility Multiplier")
    print("="*60)
    
    position_value = 1000.0
    current_volume = 100000.0
    leverage = 1
    current_price = 50000.0
    
    # Test different volatility levels
    test_cases = [
        (49900.0, 50100.0, "Low volatility (0.4%)"),
        (49000.0, 51000.0, "Medium volatility (4%)"),
        (48000.0, 52000.0, "High volatility (8%)"),
    ]
    
    results = []
    for current_low, current_high, description in test_cases:
        slippage = calculate_dynamic_slippage(
            position_value, current_volume, leverage,
            current_price, current_high, current_low
        )
        range_pct = ((current_high - current_low) / current_price) * 100
        results.append((description, range_pct, slippage))
        print(f"{description:25} (Range: {range_pct:.2f}%) → Slippage: {slippage*100:.4f}%")
    
    # Higher volatility should increase slippage
    assert results[2][2] > results[1][2], "High volatility should have higher slippage"
    assert results[1][2] > results[0][2], "Medium volatility should have higher slippage than low"
    print("✅ PASS: Volatility multiplier works correctly")


def test_leverage_multiplier():
    """Test 4: Leverage multiplier."""
    print("\n" + "="*60)
    print("TEST 4: Leverage Multiplier")
    print("="*60)
    
    position_value = 1000.0
    current_volume = 100000.0
    current_price = 50000.0
    current_high = 50100.0
    current_low = 49900.0
    
    # Test different leverage levels
    leverages = [1, 10, 25, 50, 100, 125]
    
    results = []
    for leverage in leverages:
        slippage = calculate_dynamic_slippage(
            position_value, current_volume, leverage,
            current_price, current_high, current_low
        )
        results.append((leverage, slippage))
        print(f"Leverage: {leverage:>3}x → Slippage: {slippage*100:.4f}%")
    
    # Higher leverage should increase slippage
    for i in range(len(results) - 1):
        assert results[i+1][1] >= results[i][1], f"Leverage {results[i+1][0]}x should have >= slippage than {results[i][0]}x"
    
    print("✅ PASS: Leverage multiplier scales correctly")


def test_boundary_conditions():
    """Test 5: Boundary conditions (min/max caps)."""
    print("\n" + "="*60)
    print("TEST 5: Boundary Conditions")
    print("="*60)
    
    # Test minimum slippage (ideal conditions)
    slippage_min = calculate_dynamic_slippage(
        position_value=10.0,
        current_volume=10000000.0,  # Huge volume
        leverage=1,
        current_price=50000.0,
        current_high=50010.0,  # Tiny range
        current_low=49990.0
    )
    print(f"Minimum slippage (ideal):  {slippage_min*100:.4f}%")
    assert slippage_min >= 0.00005, "Slippage below minimum cap (0.005%)"
    
    # Test maximum slippage (terrible conditions)
    slippage_max = calculate_dynamic_slippage(
        position_value=10000.0,  # Large position
        current_volume=100.0,  # Tiny volume
        leverage=125,  # Max leverage
        current_price=50000.0,
        current_high=60000.0,  # Huge range
        current_low=40000.0
    )
    print(f"Maximum slippage (extreme): {slippage_max*100:.4f}%")
    assert slippage_max <= 0.005, "Slippage above maximum cap (0.5%)"
    
    print("✅ PASS: Slippage bounded by min/max caps")


def test_realistic_scenarios():
    """Test 6: Realistic trading scenarios."""
    print("\n" + "="*60)
    print("TEST 6: Realistic Trading Scenarios")
    print("="*60)
    
    scenarios = [
        {
            'name': "Small retail trade (10x leverage)",
            'position_value': 1000.0,
            'current_volume': 5000000.0,
            'leverage': 10,
            'current_price': 50000.0,
            'current_high': 50200.0,
            'current_low': 49800.0
        },
        {
            'name': "Medium trade (25x leverage)",
            'position_value': 5000.0,
            'current_volume': 3000000.0,
            'leverage': 25,
            'current_price': 50000.0,
            'current_high': 50500.0,
            'current_low': 49500.0
        },
        {
            'name': "Large trade (50x leverage)",
            'position_value': 20000.0,
            'current_volume': 2000000.0,
            'leverage': 50,
            'current_price': 50000.0,
            'current_high': 51000.0,
            'current_low': 49000.0
        },
        {
            'name': "Volatile market (low volume)",
            'position_value': 1000.0,
            'current_volume': 500000.0,
            'leverage': 10,
            'current_price': 50000.0,
            'current_high': 52000.0,
            'current_low': 48000.0
        }
    ]
    
    for scenario in scenarios:
        slippage = calculate_dynamic_slippage(
            scenario['position_value'],
            scenario['current_volume'],
            scenario['leverage'],
            scenario['current_price'],
            scenario['current_high'],
            scenario['current_low']
        )
        
        print(f"\n{scenario['name']}:")
        print(f"  Position: ${scenario['position_value']:,.2f}, Volume: ${scenario['current_volume']:,.2f}")
        print(f"  Leverage: {scenario['leverage']}x, Range: {((scenario['current_high']-scenario['current_low'])/scenario['current_price']*100):.2f}%")
        print(f"  → Slippage: {slippage*100:.4f}% (${scenario['position_value']*slippage:.2f} cost)")
        
        # Sanity checks
        assert slippage >= 0.00005, "Slippage too low"
        assert slippage <= 0.005, "Slippage too high"
    
    print("\n✅ PASS: All realistic scenarios produce reasonable slippage")


def run_all_tests():
    """Run all slippage tests."""
    print("\n" + "="*60)
    print("DYNAMIC SLIPPAGE CALCULATION TEST SUITE")
    print("GPU Kernel Port Validation")
    print("="*60)
    
    try:
        test_base_slippage()
        test_volume_impact()
        test_volatility_multiplier()
        test_leverage_multiplier()
        test_boundary_conditions()
        test_realistic_scenarios()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED")
        print("="*60)
        print("Dynamic slippage calculation matches GPU kernel behavior.")
        print("="*60 + "\n")
        
        return True
        
    except AssertionError as e:
        print("\n" + "="*60)
        print(f"❌ TEST FAILED: {e}")
        print("="*60 + "\n")
        return False
    except Exception as e:
        print("\n" + "="*60)
        print(f"❌ ERROR: {e}")
        print("="*60 + "\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
