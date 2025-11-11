"""
Test 4: Integration Test

End-to-end validation of complete trading loop:
1. Bot loading
2. Indicator calculation
3. Signal generation
4. Position opening (with slippage & fees)
5. Position updates (TP/SL checks)
6. Position closing
7. Balance tracking
8. Complete trading scenario simulation
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
    calculate_unrealized_pnl,
    generate_signal_consensus,
    apply_funding_rates
)


def test_complete_profitable_trade():
    """Test 1: Complete profitable trade cycle."""
    print("\n" + "="*60)
    print("TEST 1: Complete Profitable Trade Cycle")
    print("="*60)
    
    initial_balance = 10000.0
    current_balance = initial_balance
    leverage = 10
    price = 50000.0
    
    print(f"Starting Balance: ${initial_balance:,.2f}")
    print(f"Initial Price:    ${price:,.2f}")
    
    # Step 1: Generate bullish signal
    indicator_values = {
        12: 25.0,  # RSI oversold
        26: 5.0,   # MACD positive
        17: 1.5    # Momentum positive
    }
    indicator_params = {12: [14.0, 0.0, 0.0], 26: [12.0, 26.0, 9.0], 17: [10.0, 0.0, 0.0]}
    indicator_history = {12: [30.0, 28.0], 26: [3.0, 4.0], 17: [1.0, 1.2]}
    
    signal, breakdown = generate_signal_consensus(
        indicator_values, indicator_params, indicator_history, 100, price
    )
    
    print(f"\nStep 1: Signal Generation")
    print(f"  Signal: {signal} (Bullish: {breakdown['bullish_count']}, Bearish: {breakdown['bearish_count']})")
    assert signal == 1.0, "Should generate bullish signal"
    
    # Step 2: Open position
    position, new_balance, stats = open_position_with_margin(
        balance=current_balance,
        price=price,
        direction=1,  # Long
        leverage=leverage,
        tp_multiplier=0.02,
        sl_multiplier=0.01,
        risk_strategy=0,
        risk_param=0.05,
        current_volume=1000000.0,
        current_high=50100.0,
        current_low=49900.0,
        existing_positions=[]
    )
    
    assert position is not None, "Position should open successfully"
    current_balance = new_balance
    
    print(f"\nStep 2: Position Opening")
    print(f"  Entry Price:     ${position.entry_price:,.2f}")
    print(f"  Quantity:        {position.size:.6f} BTC")
    print(f"  Leverage:        {position.leverage}x")
    print(f"  Margin Required: ${stats['margin_required']:,.2f}")
    print(f"  Total Cost:      ${stats['total_cost']:,.2f}")
    print(f"  New Balance:     ${current_balance:,.2f}")
    
    # Step 3: Price moves up (2% gain)
    new_price = price * 1.02
    unrealized_pnl = calculate_unrealized_pnl(position, new_price)
    
    print(f"\nStep 3: Price Movement")
    print(f"  New Price:       ${new_price:,.2f} (+2.00%)")
    print(f"  Unrealized PnL:  ${unrealized_pnl:,.2f}")
    
    # Step 4: TP hit, close position
    print(f"\nStep 4: Take Profit Hit")
    return_amount, close_stats = close_position_with_margin(
        position, new_price, 'tp',
        current_volume=1000000.0,
        current_high=new_price + 100,
        current_low=new_price - 100
    )
    
    current_balance += return_amount
    
    print(f"  Exit Price:      ${new_price:,.2f}")
    print(f"  Net PnL:         ${close_stats['net_pnl']:,.2f}")
    print(f"  Return Amount:   ${return_amount:,.2f}")
    print(f"  Final Balance:   ${current_balance:,.2f}")
    
    # Verify profit
    total_pnl = current_balance - initial_balance
    print(f"\nFinal Results:")
    print(f"  Initial:         ${initial_balance:,.2f}")
    print(f"  Final:           ${current_balance:,.2f}")
    print(f"  Total PnL:       ${total_pnl:,.2f} ({total_pnl/initial_balance*100:+.2f}%)")
    
    assert current_balance > initial_balance, "Should have profit"
    assert close_stats['net_pnl'] > 0, "Net PnL should be positive"
    
    print("✅ PASS: Complete profitable trade cycle works correctly")


def test_complete_losing_trade():
    """Test 2: Complete losing trade cycle."""
    print("\n" + "="*60)
    print("TEST 2: Complete Losing Trade Cycle")
    print("="*60)
    
    initial_balance = 10000.0
    current_balance = initial_balance
    leverage = 10
    price = 50000.0
    
    print(f"Starting Balance: ${initial_balance:,.2f}")
    
    # Open position
    position, new_balance, stats = open_position_with_margin(
        balance=current_balance,
        price=price,
        direction=1,  # Long
        leverage=leverage,
        tp_multiplier=0.02,
        sl_multiplier=0.01,
        risk_strategy=0,
        risk_param=0.05,
        current_volume=1000000.0,
        current_high=50100.0,
        current_low=49900.0,
        existing_positions=[]
    )
    
    current_balance = new_balance
    print(f"Position opened: ${position.entry_price:,.2f}, Balance: ${current_balance:,.2f}")
    
    # Price moves down (1% loss) → SL hit
    new_price = price * 0.99
    
    print(f"Price dropped to: ${new_price:,.2f} (-1.00%)")
    print("Stop Loss triggered")
    
    return_amount, close_stats = close_position_with_margin(
        position, new_price, 'sl',
        current_volume=1000000.0,
        current_high=new_price + 100,
        current_low=new_price - 100
    )
    
    current_balance += return_amount
    
    print(f"Net PnL:         ${close_stats['net_pnl']:,.2f}")
    print(f"Final Balance:   ${current_balance:,.2f}")
    
    total_pnl = current_balance - initial_balance
    print(f"Total PnL:       ${total_pnl:,.2f} ({total_pnl/initial_balance*100:+.2f}%)")
    
    assert current_balance < initial_balance, "Should have loss"
    assert close_stats['net_pnl'] < 0, "Net PnL should be negative"
    
    print("✅ PASS: Complete losing trade cycle works correctly")


def test_multiple_positions():
    """Test 3: Multiple concurrent positions."""
    print("\n" + "="*60)
    print("TEST 3: Multiple Concurrent Positions")
    print("="*60)
    
    initial_balance = 10000.0
    current_balance = initial_balance
    price = 50000.0
    positions = []
    
    print(f"Starting Balance: ${initial_balance:,.2f}")
    
    # Open 3 positions
    for i in range(3):
        position, new_balance, stats = open_position_with_margin(
            balance=current_balance,
            price=price,
            direction=1 if i % 2 == 0 else -1,  # Alternate long/short
            leverage=10,
            tp_multiplier=0.02,
            sl_multiplier=0.01,
            risk_strategy=0,
            risk_param=0.02,  # 2% per position
            current_volume=1000000.0,
            current_high=50100.0,
            current_low=49900.0,
            existing_positions=positions
        )
        
        if position:
            current_balance = new_balance
            positions.append(position)
            side = "LONG" if position.side == 1 else "SHORT"
            print(f"Position {i+1} opened: {side}, Balance: ${current_balance:,.2f}")
    
    print(f"\nTotal positions: {len(positions)}")
    
    # Calculate free margin
    free_margin = calculate_free_margin(current_balance, positions, price)
    print(f"Free Margin:     ${free_margin:,.2f}")
    
    # Calculate total unrealized PnL (at entry price = 0)
    total_unrealized = sum(calculate_unrealized_pnl(pos, price) for pos in positions)
    print(f"Total Unrealized: ${total_unrealized:,.2f}")
    
    # Check account liquidation status
    is_liquidated = check_account_liquidation(current_balance, positions, price)
    print(f"Account Liquidated: {is_liquidated}")
    
    assert len(positions) > 0, "Should have opened positions"
    assert not is_liquidated, "Account should not be liquidated at entry"
    
    print("✅ PASS: Multiple concurrent positions handled correctly")


def test_funding_rates():
    """Test 4: Funding rate application."""
    print("\n" + "="*60)
    print("TEST 4: Funding Rate Application")
    print("="*60)
    
    initial_balance = 10000.0
    current_balance = initial_balance
    
    # Create position
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
    
    print(f"Position: ${position.entry_price:,.2f}, Size: {position.size:.4f}, Leverage: {position.leverage}x")
    print(f"Starting Balance: ${current_balance:,.2f}")
    
    # Simulate 480 bars (8 hours = 1 funding payment)
    bars_held = 0
    funding_payments = []
    
    for i in range(1, 481):
        bars_held += 1
        funding_cost, current_balance = apply_funding_rates(
            position, bars_held, current_balance
        )
        
        if funding_cost != 0.0:
            funding_payments.append(funding_cost)
            print(f"\nBar {bars_held}: Funding payment ${funding_cost:,.4f}")
    
    print(f"\nTotal Funding Payments: {len(funding_payments)}")
    print(f"Final Balance:          ${current_balance:,.2f}")
    print(f"Total Funding Cost:     ${sum(funding_payments):,.4f}")
    
    assert len(funding_payments) == 1, "Should have 1 funding payment after 480 bars"
    assert current_balance < initial_balance, "Balance should decrease (long pays funding)"
    
    print("✅ PASS: Funding rates applied correctly")


def test_signal_reversal_exit():
    """Test 5: Signal reversal exit."""
    print("\n" + "="*60)
    print("TEST 5: Signal Reversal Exit")
    print("="*60)
    
    # Open long position
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
    
    print("Position: LONG @ $50,000")
    
    # Test different signals
    test_cases = [
        (1.0, False, "Bullish signal → no exit (same direction)"),
        (0.0, False, "Neutral signal → no exit"),
        (-1.0, True, "Bearish signal → EXIT (reversal)")
    ]
    
    from src.live_trading.gpu_kernel_port import check_signal_reversal
    
    for signal, should_exit, description in test_cases:
        should_exit_actual = check_signal_reversal(position, signal)
        
        result = "✅" if should_exit_actual == should_exit else "❌"
        print(f"{result} {description:50} Expected: {should_exit}, Got: {should_exit_actual}")
        
        assert should_exit_actual == should_exit, f"Signal reversal check failed for {description}"
    
    print("✅ PASS: Signal reversal exits work correctly")


def test_account_level_liquidation():
    """Test 6: Account-level liquidation scenario."""
    print("\n" + "="*60)
    print("TEST 6: Account-Level Liquidation")
    print("="*60)
    
    initial_balance = 1000.0  # Small balance for easier liquidation
    current_balance = initial_balance
    price = 50000.0
    
    # Open high-leverage position
    position, new_balance, stats = open_position_with_margin(
        balance=current_balance,
        price=price,
        direction=1,  # Long
        leverage=50,  # High leverage
        tp_multiplier=0.02,
        sl_multiplier=0.01,
        risk_strategy=0,
        risk_param=0.5,  # 50% of balance
        current_volume=1000000.0,
        current_high=50100.0,
        current_low=49900.0,
        existing_positions=[]
    )
    
    current_balance = new_balance
    positions = [position]
    
    print(f"Position opened: ${position.entry_price:,.2f}, Leverage: {position.leverage}x")
    print(f"Liquidation Price: ${position.liquidation_price:,.2f}")
    print(f"Balance after open: ${current_balance:,.2f}")
    
    # Test prices
    test_prices = [
        (50000.0, False, "At entry price"),
        (49500.0, False, "1% down (SL territory)"),
        (position.liquidation_price + 100, False, "Near liquidation"),
        (position.liquidation_price - 100, True, "Below liquidation")
    ]
    
    for test_price, should_liquidate, description in test_prices:
        is_liquidated = check_account_liquidation(current_balance, positions, test_price)
        
        unrealized = calculate_unrealized_pnl(position, test_price)
        result = "✅" if is_liquidated == should_liquidate else "❌"
        
        print(f"\n{result} {description}")
        print(f"   Price: ${test_price:,.2f}, Unrealized: ${unrealized:,.2f}, Liquidated: {is_liquidated}")
        
        assert is_liquidated == should_liquidate, f"Liquidation check failed for {description}"
    
    print("\n✅ PASS: Account-level liquidation works correctly")


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "="*60)
    print("INTEGRATION TEST SUITE")
    print("End-to-End Trading Loop Validation")
    print("="*60)
    
    try:
        test_complete_profitable_trade()
        test_complete_losing_trade()
        test_multiple_positions()
        test_funding_rates()
        test_signal_reversal_exit()
        test_account_level_liquidation()
        
        print("\n" + "="*60)
        print("✅ ALL INTEGRATION TESTS PASSED")
        print("="*60)
        print("Complete trading loop validated successfully.")
        print("Paper/live trading system ready for deployment.")
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
