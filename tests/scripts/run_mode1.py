"""
Automated script to run main.py mode 1 with last_run_config.json settings.
This bypasses the interactive prompts by pressing Enter for all defaults.
"""
import subprocess
import sys

# Prepare inputs for main.py:
# Mode: 1
# Then press Enter for all parameters (uses defaults from last_run_config.json)
# Need to send Enter for: pair, balance, population, generations, cycles, backtest_days,
# min_indicators, max_indicators, min_risk, max_risk, min_leverage, max_leverage, 
# mutation_rate, elite_pct, random_seed, timeframe
inputs = "1\n" + "\n" * 16  # Mode 1 + 16 Enter keypresses for all defaults

print("="*60)
print("Running main.py Mode 1 with last_run_config.json settings")
print("="*60)

# Run main.py with piped inputs
process = subprocess.Popen(
    [sys.executable, "main.py"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

# Send inputs
process.stdin.write(inputs)
process.stdin.flush()

# Stream output in real-time
try:
    for line in process.stdout:
        print(line, end='')
except KeyboardInterrupt:
    print("\n\n⚠️ Process interrupted by user")
    process.terminate()

process.wait()
exit_code = process.returncode

print(f"\n\nProcess completed with exit code: {exit_code}")
