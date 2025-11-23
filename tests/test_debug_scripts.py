import pytest
import pyopencl as cl
import subprocess
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEBUG_DIR = ROOT / 'scripts' / 'debug'

SCRIPTS_TO_RUN = [
    DEBUG_DIR / 'inspect_adx_values.py',
    DEBUG_DIR / 'inspect_aroon_values.py',
    DEBUG_DIR / 'inspect_rsi_values.py',
    DEBUG_DIR / 'inspect_vwap_values.py',
    DEBUG_DIR / 'inspect_sma_values.py',
]

@pytest.mark.skipif(not cl.get_platforms(), reason='No OpenCL platforms found')
@pytest.mark.parametrize('script_path', SCRIPTS_TO_RUN)
def test_debug_script_runs(script_path):
    # Ensure script exists
    assert script_path.exists(), f"Script {script_path} not found"
    # Run script and check it exits with code 0
    env = dict(os.environ)
    env['PYTHONIOENCODING'] = 'utf-8'
    res = subprocess.run(['python', str(script_path)], capture_output=True, text=True, env=env)
    assert res.returncode == 0, f"Script {script_path} failed: stdout:\n{res.stdout}\nstderr:\n{res.stderr}"