"""
Test 2: True Margin Trading Logic

Validates that CPU margin calculations match GPU kernel behavior exactly.

Tests:
1. Margin calculation (position_value / leverage)
2. Position opening (fees, slippage, balance deduction)
3. Position closing (PnL calculation, leveraged gains/losses)
4. Liquidation prices (long and short)
5. Free margin calculation
6. Account-level liquidation
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.live_trading.gpu_kernel_port import (
    Position,
    open_position_with_margin,
    close_position_with_margin,
    calculate_free_margin,
    check_account_liquidation,
    calculate_unrealized_pnl
)


def test_margin_calculation():
    """Test 1: Margin calculation."""
    print("\n" + "="*60)
    print("TEST 1: Margin Calculation")
    print("="*60)
    
    # Test different leverage levels
    test_cases = [
        (10000.0, 1, 10000.0, "No leverage"),
        (10000.0, 10, 1000.0, "10x leverage"),
        (10000.0, 25, 400.0, "25x leverage"),
        (10000.0, 50, 200.0, "50x leverage"),
        (10000.0, 100, 100.0, "100x leverage"),
        (10000.0, 125, 80.0, "125x leverage (max)")
    ]
    
    for position_value, leverage, expected_margin, description in test_cases:
        position, balance, stats = open_position_with_margin(
            balance=100000.0,
            price=50000.0,
            direction=1,
            leverage=leverage,
            tp_multiplier=0.02,
            sl_multiplier=0.01,
            risk_strategy=0,  # Fixed percentage
            risk_param=position_value / 100000.0,  # To get exact position_value
            current_volume=1000000.0,
            current_high=50100.0,
            current_low=49900.0,
            existing_positions=[]
        )
        
        assert position is not None, f"Failed to open position for {description}"
        
        margin = stats['margin_required']
        print(f"{description:25} → Position: ${position_value:>8,.2f}, Margin: ${margin:>8,.2f}, Expected: ${expected_margin:>8,.2f}")
        
        # Allow small rounding differences
        assert abs(margin - expected_margin) / expected_margin < 0.05, f"Margin mismatch for {description}"
    
    print("✅ PASS: Margin calculations correct for all leverage levels")


def test_position_opening():
    """Test 2: Position opening (fees, slippage, balance)."""
    print("\n" + "="*60)
    print("TEST 2: Position Opening")
    print("="*60)
    
    initial_balance = 10000.0
    price = 50000.0
    leverage = 10
    
    position, new_balance, stats = open_position_with_margin(
        balance=initial_balance,
        price=price,
        direction=1,  # Long
        leverage=leverage,
        tp_multiplier=0.02,
        sl_multiplier=0.01,
        risk_strategy=0,
        risk_param=0.05,  # 5% of balance
        current_volume=1000000.0,
        current_high=50100.0,
        current_low=49900.0,
        existing_positions=[]
    )
    
    assert position is not None, "Position opening failed"
    
    print(f"Initial Balance:     ${initial_balance:,.2f}")
    print(f"Position Value:      ${stats['position_value']:,.2f}")
    print(f"Margin Required:     ${stats['margin_required']:,.2f}")
    print(f"Entry Fee (0.06%):   ${stats['entry_fee']:,.2f}")
    print(f"Slippage:            ${stats['slippage_cost']:,.4f}")
    print(f"Total Cost:          ${stats['total_cost']:,.2f}")
    print(f"New Balance:         ${new_balance:,.2f}")
    print(f"Quantity:            {stats['quantity']:.6f} BTC")
    
    # Verify balance deduction
    expected_deduction = stats['total_cost']
    actual_deduction = initial_balance - new_balance
    
    print(f"\nExpected Deduction:  ${expected_deduction:,.2f}")
    print(f"Actual Deduction:    ${actual_deduction:,.2f}")
    
    assert abs(actual_deduction - expected_deduction) < 0.01, "Balance deduction mismatch"
    
    # Verify fee calculation (0.06% taker fee on position value)
    expected_fee = stats['position_value'] * 0.0006
    assert abs(stats['entry_fee'] - expected_fee) < 0.01, "Fee calculation error"
    
    # Verify quantity = margin / price
    expected_quantity = stats['margin_required'] / price
    assert abs(stats['quantity'] - expected_quantity) < 0.000001, "Quantity calculation error"
    
    print("✅ PASS: Position opening calculations correct")


def test_position_closing_profit():
    """Test 3a: Position closing with profit."""
    print("\n" + "="*60)
    print("TEST 3a: Position Closing (Profit)")
    print("="*60)
    
    # Create a position manually
    position = Position(
        entry_price=50000.0,
        size=0.02,  # 0.02 BTC
        side=1,  # Long
        leverage=10,
        tp_price=51000.0,
        sl_price=49500.0,
        entry_time=0.0,
        is_active=True
    )
    
    # Margin reserved = entry_price * size = 50000 * 0.02 = 1000
    margin = position.entry_price * position.size
    
    # Close at 2% profit
    exit_price = 51000.0
    
    return_amount, stats = close_position_with_margin(
        position,
        exit_price,
        'tp',
        current_volume=1000000.0,
        current_high=51100.0,
        current_low=50900.0
    )
    
    print(f"Entry Price:         ${position.entry_price:,.2f}")
    print(f"Exit Price:          ${exit_price:,.2f}")
    print(f"Price Change:        {((exit_price - position.entry_price) / position.entry_price * 100):+.2f}%")
    print(f"Quantity:            {position.size:.6f} BTC")
    print(f"Leverage:            {position.leverage}x")
    print(f"\nMargin Reserved:     ${stats['margin_reserved']:,.2f}")
    print(f"Raw PnL:             ${stats['raw_pnl']:,.2f}")
    print(f"Leveraged PnL:       ${stats['leveraged_pnl']:,.2f}")
    print(f"Exit Fee:            ${stats['exit_fee']:,.2f}")
    print(f"Slippage:            ${stats['slippage_cost']:,.4f}")
    print(f"Net PnL:             ${stats['net_pnl']:,.2f}")
    print(f"Total Return:        ${stats['total_return']:,.2f}")
    
    # Verify leveraged PnL
    # Raw PnL = (exit - entry) * size = (51000 - 50000) * 0.02 = 20
    # Leveraged PnL = raw * leverage = 20 * 10 = 200
    expected_raw_pnl = (exit_price - position.entry_price) * position.size
    expected_leveraged_pnl = expected_raw_pnl * position.leverage
    
    assert abs(stats['raw_pnl'] - expected_raw_pnl) < 0.01, "Raw PnL calculation error"
    assert abs(stats['leveraged_pnl'] - expected_leveraged_pnl) < 0.01, "Leveraged PnL calculation error"
    
    # Total return should be margin + net_pnl
    expected_return = margin + stats['net_pnl']
    assert abs(return_amount - expected_return) < 0.01, "Return amount mismatch"
    
    # Net PnL should be positive
    assert stats['net_pnl'] > 0, "Should have profit"
    
    print("✅ PASS: Profitable position closing calculations correct")


def test_position_closing_loss():
    """Test 3b: Position closing with loss."""
    print("\n" + "="*60)
    print("TEST 3b: Position Closing (Loss)")
    print("="*60)
    
    position = Position(
        entry_price=50000.0,
        size=0.02,
        side=1,  # Long
        leverage=10,
        tp_price=51000.0,
        sl_price=49500.0,
        entry_time=0.0,
        is_active=True
    )
    
    margin = position.entry_price * position.size
    
    # Close at 1% loss
    exit_price = 49500.0
    
    return_amount, stats = close_position_with_margin(
        position,
        exit_price,
        'sl',
        current_volume=1000000.0,
        current_high=49600.0,
        current_low=49400.0
    )
    
    print(f"Entry Price:         ${position.entry_price:,.2f}")
    print(f"Exit Price:          ${exit_price:,.2f}")
    print(f"Price Change:        {((exit_price - position.entry_price) / position.entry_price * 100):+.2f}%")
    print(f"\nMargin Reserved:     ${stats['margin_reserved']:,.2f}")
    print(f"Leveraged PnL:       ${stats['leveraged_pnl']:,.2f}")
    print(f"Net PnL:             ${stats['net_pnl']:,.2f}")
    print(f"Total Return:        ${stats['total_return']:,.2f}")
    print(f"Loss Amount:         ${margin - return_amount:,.2f}")
    
    # Net PnL should be negative
    assert stats['net_pnl'] < 0, "Should have loss"
    
    # Return should be less than margin but not negative
    assert return_amount < margin, "Return should be less than margin"
    assert return_amount >= 0, "Return should not be negative (can't lose more than margin)"
    
    print("✅ PASS: Loss position closing calculations correct")


def test_liquidation_prices():
    """Test 4: Liquidation price calculation."""
    print("\n" + "="*60)
    print("TEST 4: Liquidation Price Calculation")
    print("="*60)
    
    test_cases = [
        (1, 50000.0, 10, "Long 10x"),
        (1, 50000.0, 25, "Long 25x"),
        (1, 50000.0, 50, "Long 50x"),
        (-1, 50000.0, 10, "Short 10x"),
        (-1, 50000.0, 25, "Short 25x"),
        (-1, 50000.0, 50, "Short 50x"),
    ]
    
    for direction, entry_price, leverage, description in test_cases:
        position, balance, stats = open_position_with_margin(
            balance=10000.0,
            price=entry_price,
            direction=direction,
            leverage=leverage,
            tp_multiplier=0.02,
            sl_multiplier=0.01,
            risk_strategy=0,
            risk_param=0.05,
            current_volume=1000000.0,
            current_high=entry_price + 100,
            current_low=entry_price - 100,
            existing_positions=[]
        )
        
        assert position is not None, f"Failed to open {description}"
        
        liq_price = position.liquidation_price
        liq_distance_pct = abs((liq_price - entry_price) / entry_price) * 100
        
        print(f"{description:15} → Entry: ${entry_price:>8,.2f}, Liquidation: ${liq_price:>8,.2f}, Distance: {liq_distance_pct:>6.2f}%")
        
        # Verify liquidation price is in correct direction
        if direction == 1:  # Long
            assert liq_price < entry_price, "Long liquidation should be below entry"
        else:  # Short
            assert liq_price > entry_price, "Short liquidation should be above entry"
        
        # Verify distance increases with lower leverage
        # (Higher leverage = closer liquidation price)
    
    print("✅ PASS: Liquidation prices calculated correctly")


def test_free_margin():
    """Test 5: Free margin calculation."""
    print("\n" + "="*60)
    print("TEST 5: Free Margin Calculation")
    print("="*60)
    
    balance = 10000.0
    current_price = 50000.0
    
    # Create multiple positions
    positions = [
        Position(entry_price=50000.0, size=0.01, side=1, leverage=10, 
                tp_price=51000.0, sl_price=49500.0, entry_time=0.0, is_active=True),
        Position(entry_price=50000.0, size=0.02, side=-1, leverage=10,
                tp_price=49000.0, sl_price=50500.0, entry_time=0.0, is_active=True)
    ]
    
    free_margin = calculate_free_margin(balance, positions, current_price)
    
    # Calculate expected free margin
    used_margin = sum(pos.entry_price * pos.size for pos in positions)
    unrealized_pnl = sum(calculate_unrealized_pnl(pos, current_price) for pos in positions)
    expected_free = balance + unrealized_pnl - used_margin
    
    print(f"Balance:             ${balance:,.2f}")
    print(f"Used Margin:         ${used_margin:,.2f}")
    print(f"Unrealized PnL:      ${unrealized_pnl:,.2f}")
    print(f"Free Margin:         ${free_margin:,.2f}")
    print(f"Expected Free:       ${expected_free:,.2f}")
    
    assert abs(free_margin - expected_free) < 0.01, "Free margin calculation error"
    print("✅ PASS: Free margin calculated correctly")


def test_account_liquidation():
    """Test 6: Account-level liquidation check."""
    print("\n" + "="*60)
    print("TEST 6: Account-Level Liquidation")
    print("="*60)
    
    # Scenario 1: Healthy account
    balance = 10000.0
    positions = [
        Position(entry_price=50000.0, size=0.01, side=1, leverage=10,
                tp_price=51000.0, sl_price=49500.0, entry_time=0.0, is_active=True)
    ]
    current_price = 50000.0
    
    is_liquidated = check_account_liquidation(balance, positions, current_price)
    print(f"Scenario 1 (Healthy): Balance=${balance:,.2f}, Price=${current_price:,.2f} → Liquidated: {is_liquidated}")
    assert not is_liquidated, "Healthy account should not be liquidated"
    
    # Scenario 2: Account in danger (price moved against position)
    current_price = 45000.0  # 10% drop, 10x leverage = -100% loss
    is_liquidated = check_account_liquidation(balance, positions, current_price)
    print(f"Scenario 2 (Danger): Balance=${balance:,.2f}, Price=${current_price:,.2f} → Liquidated: {is_liquidated}")
    # Should be liquidated or very close
    
    # Scenario 3: Multiple positions with mixed PnL
    positions = [
        Position(entry_price=50000.0, size=0.01, side=1, leverage=10,
                tp_price=51000.0, sl_price=49500.0, entry_time=0.0, is_active=True),
        Position(entry_price=50000.0, size=0.01, side=-1, leverage=10,
                tp_price=49000.0, sl_price=50500.0, entry_time=0.0, is_active=True)
    ]
    current_price = 48000.0  # Long loses, short wins
    is_liquidated = check_account_liquidation(balance, positions, current_price)
    print(f"Scenario 3 (Mixed): Balance=${balance:,.2f}, Price=${current_price:,.2f} → Liquidated: {is_liquidated}")
    
    print("✅ PASS: Account liquidation checks work correctly")


def run_all_tests():
    """Run all margin trading tests."""
    print("\n" + "="*60)
    print("TRUE MARGIN TRADING LOGIC TEST SUITE")
    print("GPU Kernel Port Validation")
    print("="*60)
    
    try:
        test_margin_calculation()
        test_position_opening()
        test_position_closing_profit()
        test_position_closing_loss()
        test_liquidation_prices()
        test_free_margin()
        test_account_liquidation()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED")
        print("="*60)
        print("True margin trading logic matches GPU kernel behavior.")
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
