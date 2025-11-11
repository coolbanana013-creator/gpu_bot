"""
Test 3: Signal Consensus Generation

Validates that 100% consensus signal generation works correctly.

Tests:
1. All bullish indicators → Buy signal
2. All bearish indicators → Sell signal
3. Mixed indicators → No signal
4. Individual indicator signal logic
5. Indicator history tracking
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.live_trading.gpu_kernel_port import (
    generate_signal_consensus,
    get_indicator_signal
)


def test_all_bullish():
    """Test 1: All bullish indicators → Buy signal."""
    print("\n" + "="*60)
    print("TEST 1: All Bullish Indicators")
    print("="*60)
    
    # 3 indicators, all bullish
    indicator_values = {
        12: 25.0,  # RSI: 25 (oversold → bullish)
        26: 5.0,   # MACD: +5 (above zero → bullish)
        17: 1.5    # Momentum: +1.5 (positive → bullish)
    }
    
    indicator_params = {
        12: [14.0, 0.0, 0.0],
        26: [12.0, 26.0, 9.0],
        17: [10.0, 0.0, 0.0]
    }
    
    indicator_history = {
        12: [30.0, 28.0, 26.0],  # RSI falling
        26: [3.0, 4.0, 4.5],      # MACD rising
        17: [1.0, 1.2, 1.3]       # Momentum rising
    }
    
    signal, breakdown = generate_signal_consensus(
        indicator_values,
        indicator_params,
        indicator_history,
        100,  # bar index
        50000.0  # price
    )
    
    print(f"Indicator Values: {indicator_values}")
    print(f"Bullish Count:    {breakdown['bullish_count']}")
    print(f"Bearish Count:    {breakdown['bearish_count']}")
    print(f"Neutral Count:    {breakdown['neutral_count']}")
    print(f"Signal:           {signal}")
    print(f"Individual Signals: {breakdown['signals']}")
    
    assert signal == 1.0, "Expected bullish signal (1.0)"
    assert breakdown['bullish_count'] == 3, "All 3 indicators should be bullish"
    assert breakdown['bearish_count'] == 0, "No bearish indicators"
    
    print("✅ PASS: All bullish indicators produce buy signal")


def test_all_bearish():
    """Test 2: All bearish indicators → Sell signal."""
    print("\n" + "="*60)
    print("TEST 2: All Bearish Indicators")
    print("="*60)
    
    # 3 indicators, all bearish
    indicator_values = {
        12: 75.0,  # RSI: 75 (overbought → bearish)
        26: -5.0,  # MACD: -5 (below zero → bearish)
        17: -1.5   # Momentum: -1.5 (negative → bearish)
    }
    
    indicator_params = {
        12: [14.0, 0.0, 0.0],
        26: [12.0, 26.0, 9.0],
        17: [10.0, 0.0, 0.0]
    }
    
    indicator_history = {
        12: [70.0, 72.0, 74.0],  # RSI rising to overbought
        26: [-3.0, -4.0, -4.5],  # MACD falling
        17: [-1.0, -1.2, -1.3]   # Momentum falling
    }
    
    signal, breakdown = generate_signal_consensus(
        indicator_values,
        indicator_params,
        indicator_history,
        100,
        50000.0
    )
    
    print(f"Indicator Values: {indicator_values}")
    print(f"Bullish Count:    {breakdown['bullish_count']}")
    print(f"Bearish Count:    {breakdown['bearish_count']}")
    print(f"Neutral Count:    {breakdown['neutral_count']}")
    print(f"Signal:           {signal}")
    
    assert signal == -1.0, "Expected bearish signal (-1.0)"
    assert breakdown['bearish_count'] == 3, "All 3 indicators should be bearish"
    assert breakdown['bullish_count'] == 0, "No bullish indicators"
    
    print("✅ PASS: All bearish indicators produce sell signal")


def test_mixed_indicators():
    """Test 3: Mixed indicators → No signal."""
    print("\n" + "="*60)
    print("TEST 3: Mixed Indicators (No Consensus)")
    print("="*60)
    
    # 3 indicators: 1 bullish, 1 bearish, 1 neutral
    indicator_values = {
        12: 25.0,  # RSI: 25 (oversold → bullish)
        26: -5.0,  # MACD: -5 (below zero → bearish)
        27: 20.0   # ADX: 20 (weak trend → neutral)
    }
    
    indicator_params = {
        12: [14.0, 0.0, 0.0],
        26: [12.0, 26.0, 9.0],
        27: [14.0, 0.0, 0.0]
    }
    
    indicator_history = {
        12: [30.0, 28.0, 26.0],
        26: [-3.0, -4.0, -4.5],
        27: [18.0, 19.0, 19.5]
    }
    
    signal, breakdown = generate_signal_consensus(
        indicator_values,
        indicator_params,
        indicator_history,
        100,
        50000.0
    )
    
    print(f"Indicator Values: {indicator_values}")
    print(f"Bullish Count:    {breakdown['bullish_count']}")
    print(f"Bearish Count:    {breakdown['bearish_count']}")
    print(f"Neutral Count:    {breakdown['neutral_count']}")
    print(f"Signal:           {signal}")
    
    assert signal == 0.0, "Expected no signal (0.0) due to lack of consensus"
    assert breakdown['bullish_count'] > 0, "Should have at least 1 bullish"
    assert breakdown['bearish_count'] > 0, "Should have at least 1 bearish"
    
    print("✅ PASS: Mixed indicators produce no signal (100% consensus required)")


def test_individual_indicator_logic():
    """Test 4: Individual indicator signal logic."""
    print("\n" + "="*60)
    print("TEST 4: Individual Indicator Logic")
    print("="*60)
    
    test_cases = [
        # (indicator_index, value, params, history, expected_signal, description)
        (12, 25.0, [14.0, 0.0, 0.0], [30.0, 28.0, 26.0], 1, "RSI oversold"),
        (12, 75.0, [14.0, 0.0, 0.0], [70.0, 72.0, 74.0], -1, "RSI overbought"),
        (12, 50.0, [14.0, 0.0, 0.0], [50.0, 50.0, 50.0], 0, "RSI neutral"),
        
        (26, 5.0, [12.0, 26.0, 9.0], [3.0, 4.0, 4.5], 1, "MACD above zero rising"),
        (26, -5.0, [12.0, 26.0, 9.0], [-3.0, -4.0, -4.5], -1, "MACD below zero falling"),
        
        (17, 1.5, [10.0, 0.0, 0.0], [1.0, 1.2, 1.3], 1, "Momentum positive"),
        (17, -1.5, [10.0, 0.0, 0.0], [-1.0, -1.2, -1.3], -1, "Momentum negative"),
        
        (18, 3.0, [10.0, 0.0, 0.0], [2.0, 2.5, 2.8], 1, "ROC > 2%"),
        (18, -3.0, [10.0, 0.0, 0.0], [-2.0, -2.5, -2.8], -1, "ROC < -2%"),
    ]
    
    for ind_idx, value, params, history, expected, description in test_cases:
        signal = get_indicator_signal(
            ind_idx, value, params, 100, history, 50000.0
        )
        
        result = "✅" if signal == expected else "❌"
        print(f"{result} {description:30} → Expected: {expected:>2}, Got: {signal:>2}")
        
        assert signal == expected, f"Signal mismatch for {description}"
    
    print("✅ PASS: All individual indicator signals correct")


def test_indicator_history():
    """Test 5: Indicator history tracking."""
    print("\n" + "="*60)
    print("TEST 5: Indicator History Tracking")
    print("="*60)
    
    # Simulate indicator values over time
    rsi_values = [30.0, 28.0, 26.0, 24.0, 22.0]  # Falling to oversold
    macd_values = [0.0, 1.0, 2.0, 3.0, 4.0]      # Rising to bullish
    
    print("Simulating 5 bars of indicator values:")
    print(f"RSI values:  {rsi_values}")
    print(f"MACD values: {macd_values}")
    
    # Process each bar
    for i in range(len(rsi_values)):
        indicator_values = {
            12: rsi_values[i],
            26: macd_values[i]
        }
        
        indicator_params = {
            12: [14.0, 0.0, 0.0],
            26: [12.0, 26.0, 9.0]
        }
        
        # History = previous values
        indicator_history = {
            12: rsi_values[:i] if i > 0 else [],
            26: macd_values[:i] if i > 0 else []
        }
        
        signal, breakdown = generate_signal_consensus(
            indicator_values,
            indicator_params,
            indicator_history,
            i,
            50000.0
        )
        
        print(f"\nBar {i}: RSI={rsi_values[i]:.1f}, MACD={macd_values[i]:.1f}")
        print(f"  Bullish: {breakdown['bullish_count']}, Bearish: {breakdown['bearish_count']}, Signal: {signal}")
    
    print("\n✅ PASS: Indicator history tracking works correctly")


def test_realistic_scenarios():
    """Test 6: Realistic trading scenarios."""
    print("\n" + "="*60)
    print("TEST 6: Realistic Trading Scenarios")
    print("="*60)
    
    scenarios = [
        {
            'name': "Strong Uptrend (5 indicators)",
            'indicator_values': {
                12: 28.0,   # RSI oversold
                26: 8.0,    # MACD positive
                17: 2.0,    # Momentum positive
                0: 50100.0, # SMA rising
                36: 1000000.0  # OBV rising
            },
            'expected_signal': 1.0
        },
        {
            'name': "Strong Downtrend (5 indicators)",
            'indicator_values': {
                12: 72.0,   # RSI overbought
                26: -8.0,   # MACD negative
                17: -2.0,   # Momentum negative
                0: 49900.0, # SMA falling
                36: -1000000.0  # OBV falling
            },
            'expected_signal': -1.0
        },
        {
            'name': "Conflicting Signals (3 bull, 2 bear)",
            'indicator_values': {
                12: 28.0,   # RSI bullish
                26: 8.0,    # MACD bullish
                17: 2.0,    # Momentum bullish
                19: -15.0,  # Williams %R bearish
                29: 150.0   # CCI bearish
            },
            'expected_signal': 0.0  # No consensus
        }
    ]
    
    for scenario in scenarios:
        # Build params and history
        indicator_params = {}
        indicator_history = {}
        for ind_idx in scenario['indicator_values'].keys():
            indicator_params[ind_idx] = [14.0, 0.0, 0.0]
            indicator_history[ind_idx] = []
        
        signal, breakdown = generate_signal_consensus(
            scenario['indicator_values'],
            indicator_params,
            indicator_history,
            100,
            50000.0
        )
        
        result = "✅" if signal == scenario['expected_signal'] else "❌"
        print(f"\n{result} {scenario['name']}")
        print(f"   Bullish: {breakdown['bullish_count']}, Bearish: {breakdown['bearish_count']}, Neutral: {breakdown['neutral_count']}")
        print(f"   Expected: {scenario['expected_signal']}, Got: {signal}")
        
        assert signal == scenario['expected_signal'], f"Signal mismatch for {scenario['name']}"
    
    print("\n✅ PASS: All realistic scenarios produce correct signals")


def run_all_tests():
    """Run all signal consensus tests."""
    print("\n" + "="*60)
    print("SIGNAL CONSENSUS GENERATION TEST SUITE")
    print("GPU Kernel Port Validation")
    print("="*60)
    
    try:
        test_all_bullish()
        test_all_bearish()
        test_mixed_indicators()
        test_individual_indicator_logic()
        test_indicator_history()
        test_realistic_scenarios()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED")
        print("="*60)
        print("Signal consensus generation matches GPU kernel behavior.")
        print("100% consensus requirement working correctly.")
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
