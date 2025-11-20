#!/usr/bin/env python3
"""
Quick test script - runs main.py with minimal parameters automatically.
Tests GPU kernel execution without user input.
"""

import subprocess
import sys

# Test configuration: small workload for quick testing
test_config = """1
BTC/USDT
1000
1000
1
5
7
1m
125
125
1
5
1
1
200
n
n
"""

print("=" * 60)
print("TESTING GPU KERNEL EXECUTION")
print("=" * 60)
print("\nConfiguration:")
print("- Mode: 1 (Genetic Algorithm)")
print("- Population: 1000 bots")
print("- Generations: 1")
print("- Cycles: 5")
print("- Days per cycle: 7")
print("- Timeframe: 1m")
print("\nMonitor GPU usage in Task Manager during execution:")
print("1. Spike to 80-90% (indicators precomputation) - WORKING")
print("2. Should stay at 60-100% (backtest kernel) - TESTING")
print("\nStarting in 3 seconds...")
print("=" * 60)

import time
time.sleep(3)

# Run main.py with piped input
process = subprocess.Popen(
    [sys.executable, "main.py"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

# Send inputs
process.stdin.write(test_config)
process.stdin.close()

# Stream output in real-time
print("\n[TEST OUTPUT]")
try:
    for line in process.stdout:
        print(line, end='')
except KeyboardInterrupt:
    print("\n[TEST INTERRUPTED]")
    process.terminate()
    sys.exit(1)

# Wait for completion
returncode = process.wait()

print("\n" + "=" * 60)
if returncode == 0:
    print("✓ TEST COMPLETED SUCCESSFULLY")
else:
    print(f"✗ TEST FAILED (exit code: {returncode})")
print("=" * 60)

sys.exit(returncode)
