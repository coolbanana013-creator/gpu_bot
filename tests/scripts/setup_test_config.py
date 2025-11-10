"""Quick test to verify bot diversity during evolution."""
import json

# Update config to use smaller parameters for quick test
config = {
    "pair": "BTC/USDT:USDT",
    "population_size": 100,
    "generations": 5,
    "cycles": 1,
    "backtest_days": 15,
    "timeframe": "15m",
    "min_leverage": 1,
    "max_leverage": 1,
    "min_indicators": 1,
    "max_indicators": 5,
    "min_risk_strategies": 1,
    "max_risk_strategies": 5,
    "seed": 42
}

with open('last_run_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("Created test configuration:")
print(json.dumps(config, indent=2))
print("\nRun 'python main.py' and press Enter repeatedly to use these defaults.")
