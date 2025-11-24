#!/usr/bin/env python3
"""
Stub wrapper: calls canonical script at scripts/debug/inspect_adx_values.py to avoid duplicate logic.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEBUG_SCRIPT = ROOT / 'scripts' / 'debug' / 'inspect_adx_values.py'

if __name__ == '__main__':
	if not DEBUG_SCRIPT.exists():
		print('Canonical debug script not found: ', DEBUG_SCRIPT)
		sys.exit(1)
	rc = subprocess.call([sys.executable, str(DEBUG_SCRIPT)], cwd=str(ROOT))
	sys.exit(rc)

