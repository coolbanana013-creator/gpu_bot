#!/usr/bin/env python3
"""
Test GPU execution with full debug output
"""

import subprocess
import sys

# Test configuration
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

print("=" * 80)
print("GPU KERNEL DEBUG TEST")
print("=" * 80)
print("\nThis will:")
print("1. Enable OpenCL printf output")
print("2. Run with 1000 bots, 1 generation, 5 cycles")
print("3. Show kernel debug messages")
print("\nWatch for:")
print("- [KERNEL] Entry: ... (kernel starts)")
print("- [KERNEL] Bot 0 Cycle 0: ... (cycle loop starts)")
print("- [KERNEL] Bot 0 Cycle 0 Bar ...: ... (main bar loop starts)")
print("\nIf it hangs WITHOUT printing bar loop message, the issue is BEFORE the loop")
print("If it prints bar loop but hangs, the issue is INSIDE the loop")
print("=" * 80)

import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['PYOPENCL_NO_CACHE'] = '1'

import time
print("\nStarting in 3 seconds...")
time.sleep(3)

# Run main.py with debug
process = subprocess.Popen(
    [sys.executable, "main.py"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
    env=os.environ
)

# Send inputs
process.stdin.write(test_config)
process.stdin.close()

print("\n[DEBUG OUTPUT]")
print("=" * 80)

try:
    for line in process.stdout:
        print(line, end='')
        # Highlight debug lines
        if '[KERNEL]' in line or '[DEBUG]' in line:
            print("  << DEBUG", end='')
        print()
except KeyboardInterrupt:
    print("\n\n[INTERRUPTED]")
    process.terminate()
    sys.exit(1)

returncode = process.wait()
print("=" * 80)
print(f"\nExit code: {returncode}")
sys.exit(returncode)
