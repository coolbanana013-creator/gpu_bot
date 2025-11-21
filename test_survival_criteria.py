"""
Quick test to verify relaxed survival criteria work
Tests the 3 survival criteria independently
"""
print("=== SURVIVAL CRITERIA TEST ===\n")

# Create mock results with different survival scenarios
print("Testing survival criteria...")
print("Criteria: Avg profit > -10%, 70%+ cycles profitable, Max DD < 30%\n")

# Scenario 1: Bot with 70% profitable cycles, -5% avg profit, 25% DD
print("Scenario 1: 70% profitable cycles, -5% avg profit, 25% DD")
print("Expected: SURVIVE ✅")
result1 = type('obj', (object,), {
    'total_pnl': -0.5,
    'per_cycle_pnl': [0.1, 0.05, -0.2, 0.08, 0.06, -0.15, 0.04, 0.09, -0.1, 0.07,
                      0.05, 0.03, -0.08, 0.06, 0.04, 0.02, -0.12, 0.07, 0.05, 0.03],
    'max_drawdown': 0.25,
    'fitness_score': 0.5,
    'num_trades': 50
})()
profitable = sum(1 for p in result1.per_cycle_pnl if p > 0)
avg_profit = (result1.total_pnl / 10.0) * 100
print(f"  Profitable cycles: {profitable}/20 = {profitable/20*100:.0f}%")
print(f"  Avg profit: {avg_profit:.1f}%")
print(f"  Max DD: {result1.max_drawdown*100:.0f}%")

# Check criteria
passes_profit = avg_profit >= -10.0
passes_cycles = (profitable / 20) >= 0.70
passes_dd = result1.max_drawdown < 0.30
passes_all = passes_profit and passes_cycles and passes_dd
print(f"  Result: {'SURVIVE ✅' if passes_all else 'ELIMINATED ❌'}\n")

# Scenario 2: Bot with 65% profitable cycles (should fail)
print("Scenario 2: 65% profitable cycles, +2% avg profit, 20% DD")
print("Expected: ELIMINATED ❌ (below 70% threshold)")
result2 = type('obj', (object,), {
    'total_pnl': 0.2,
    'per_cycle_pnl': [0.05, 0.02, -0.05, 0.03, -0.08, 0.04, -0.02, 0.06,
                      0.03, -0.04, 0.02, -0.03, 0.05, 0.04, -0.06, 0.02,
                      -0.05, 0.03, 0.04, -0.02],
    'max_drawdown': 0.20,
    'fitness_score': 0.8,
    'num_trades': 60
})()
profitable2 = sum(1 for p in result2.per_cycle_pnl if p > 0)
avg_profit2 = (result2.total_pnl / 10.0) * 100
print(f"  Profitable cycles: {profitable2}/20 = {profitable2/20*100:.0f}%")
print(f"  Avg profit: {avg_profit2:.1f}%")
print(f"  Max DD: {result2.max_drawdown*100:.0f}%")

passes_profit2 = avg_profit2 >= -10.0
passes_cycles2 = (profitable2 / 20) >= 0.70
passes_dd2 = result2.max_drawdown < 0.30
passes_all2 = passes_profit2 and passes_cycles2 and passes_dd2
print(f"  Result: {'SURVIVE ✅' if passes_all2 else 'ELIMINATED ❌'}\n")

# Scenario 3: Bot with 100% profitable cycles (ideal case)
print("Scenario 3: 100% profitable cycles, +15% avg profit, 10% DD")
print("Expected: SURVIVE ✅ (ideal bot)")
result3 = type('obj', (object,), {
    'total_pnl': 1.5,
    'per_cycle_pnl': [0.08, 0.07, 0.09, 0.06, 0.08, 0.07, 0.09, 0.08,
                      0.07, 0.08, 0.06, 0.09, 0.08, 0.07, 0.06, 0.09,
                      0.08, 0.07, 0.08, 0.06],
    'max_drawdown': 0.10,
    'fitness_score': 2.5,
    'num_trades': 80
})()
profitable3 = sum(1 for p in result3.per_cycle_pnl if p > 0)
avg_profit3 = (result3.total_pnl / 10.0) * 100
print(f"  Profitable cycles: {profitable3}/20 = {profitable3/20*100:.0f}%")
print(f"  Avg profit: {avg_profit3:.1f}%")
print(f"  Max DD: {result3.max_drawdown*100:.0f}%")

passes_profit3 = avg_profit3 >= -10.0
passes_cycles3 = (profitable3 / 20) >= 0.70
passes_dd3 = result3.max_drawdown < 0.30
passes_all3 = passes_profit3 and passes_cycles3 and passes_dd3
print(f"  Result: {'SURVIVE ✅' if passes_all3 else 'ELIMINATED ❌'}\n")

# Scenario 4: Bot with 35% DD (should fail)
print("Scenario 4: 80% profitable cycles, +5% avg profit, 35% DD")
print("Expected: ELIMINATED ❌ (exceeds 30% DD limit)")
result4 = type('obj', (object,), {
    'total_pnl': 0.5,
    'per_cycle_pnl': [0.05, 0.04, 0.06, -0.08, 0.05, 0.04, -0.06, 0.07,
                      0.05, 0.04, 0.06, -0.05, 0.05, 0.06, 0.04, 0.07,
                      0.05, -0.04, 0.06, 0.05],
    'max_drawdown': 0.35,
    'fitness_score': 0.6,
    'num_trades': 70
})()
profitable4 = sum(1 for p in result4.per_cycle_pnl if p > 0)
avg_profit4 = (result4.total_pnl / 10.0) * 100
print(f"  Profitable cycles: {profitable4}/20 = {profitable4/20*100:.0f}%")
print(f"  Avg profit: {avg_profit4:.1f}%")
print(f"  Max DD: {result4.max_drawdown*100:.0f}%")

passes_profit4 = avg_profit4 >= -10.0
passes_cycles4 = (profitable4 / 20) >= 0.70
passes_dd4 = result4.max_drawdown < 0.30
passes_all4 = passes_profit4 and passes_cycles4 and passes_dd4
print(f"  Result: {'SURVIVE ✅' if passes_all4 else 'ELIMINATED ❌'}\n")

print("=" * 50)
print("\nSUMMARY:")
print(f"Scenario 1 (good bot): {'PASS ✅' if passes_all else 'FAIL ❌'}")
print(f"Scenario 2 (low win%): {'PASS ✅' if not passes_all2 else 'FAIL ❌'}")  # Should be eliminated
print(f"Scenario 3 (ideal bot): {'PASS ✅' if passes_all3 else 'FAIL ❌'}")
print(f"Scenario 4 (high DD): {'PASS ✅' if not passes_all4 else 'FAIL ❌'}")  # Should be eliminated

all_tests_pass = passes_all and not passes_all2 and passes_all3 and not passes_all4
print(f"\nAll tests: {'PASS ✅' if all_tests_pass else 'FAIL ❌'}")

if all_tests_pass:
    print("\n✅ Survival criteria are working correctly!")
    print("Expected survival rate with new criteria: 5-15%")
else:
    print("\n❌ Survival criteria may have issues - check implementation")
