"""Run main.py with all default parameters for testing."""
import subprocess
import sys

# All inputs for main.py with defaults
inputs = [
    "1",        # Mode 1
    "",         # Trading pair (default: XBTUSDTM)
    "",         # Initial balance (default: 100.0)
    "1000",     # Population size (minimum allowed)
    "2",        # Generations (smaller for testing)
    "2",        # Cycles (smaller for testing) 
    "3",        # Days per cycle (smaller for testing)
    "",         # Timeframe (default: 1m)
    "",         # Min leverage (default: 1)
    "",         # Max leverage (default: 10)
    "",         # Min indicators (default: 1)
    "",         # Max indicators (default: 5)
    "",         # Min risk strategies (default: 1)
    "",         # Max risk strategies (default: 5)
    "",         # Random seed (default: y)
    "",         # Seed value (default: 42)
]

# Join all inputs with newlines
input_str = "\n".join(inputs) + "\n"

print("Running main.py with test parameters...")
print("Population: 1000, Generations: 2, Cycles: 2, Days: 3")
print()

# Run main.py with inputs
process = subprocess.Popen(
    [sys.executable, "main.py"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

# Send inputs and capture output
try:
    output, _ = process.communicate(input=input_str, timeout=300)  # 5 min timeout
    print(output)
    sys.exit(process.returncode)
except subprocess.TimeoutExpired:
    process.kill()
    print("\nProcess timed out after 5 minutes")
    sys.exit(1)
