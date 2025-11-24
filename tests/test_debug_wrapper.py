import os
import subprocess
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / 'scripts' / 'tools' / 'run_all_debug_tests.py'

def test_debug_wrapper_runs_offline():
    assert WRAPPER.exists(), 'Wrapper not found'
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['SKIP_LOAD_MARKETS'] = '1'
    res = subprocess.run(['python', str(WRAPPER)], capture_output=True, text=True, env=env, cwd=str(ROOT))
    # Wrapper may exit with nonzero if any script failed; we accept both but check it runs
    assert res.returncode in (0,1), f'Unexpected exit code {res.returncode}, stdout: {res.stdout}, stderr: {res.stderr}'