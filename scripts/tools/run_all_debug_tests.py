#!/usr/bin/env python3
"""
Run all debug/inspect/parity scripts and report results.
This wrapper enumerates Python scripts under scripts/debug, scripts/inspect, scripts/parity
and runs them as subprocesses with controlled environment to avoid UnicodeEncodeError and
to allow tests to run offline by default (SKIP_LOAD_MARKETS=1).
"""
from pathlib import Path
import subprocess
import sys
import os
import argparse

ROOT = Path(__file__).resolve().parents[2]

def discover_scripts(root: Path):
    scripts_dir = root / 'scripts'
    candidates = []
    for sub in ('debug', 'inspect', 'parity'):
        folder = scripts_dir / sub
        if not folder.exists():
            continue
        for p in folder.glob('*.py'):
            if p.name.startswith('__'):
                continue
            candidates.append(p)
    return candidates

def run_script(script_path: Path, live: bool = False):
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    if not live:
        env['SKIP_LOAD_MARKETS'] = '1'
        env['DATA_FETCHER_SKIP_LOAD_MARKETS'] = '1'
    # Use the same python executable
    cmd = [sys.executable, str(script_path)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, cwd=str(ROOT))
    out, err = proc.communicate()
    return proc.returncode, out.decode('utf-8', errors='replace'), err.decode('utf-8', errors='replace')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--live', action='store_true', help='Allow network operations (do not set SKIP_LOAD_MARKETS)')
    args = parser.parse_args()

    scripts = discover_scripts(ROOT)
    print(f'Found {len(scripts)} script(s) to run')

    results = []
    for s in scripts:
        print(f'Running {s}...')
        rc, out, err = run_script(s, live=args.live)
        results.append((s, rc, out, err))
        print(f'Exit {rc} | stdout: {len(out)} bytes | stderr: {len(err)} bytes')
        if rc != 0:
            print('---- STDERR ----')
            print(err)
            print('---- STDOUT ----')
            print(out)
            print('---- END ----')

    # Summary
    print('\nSummary:')
    failed = [r for r in results if r[1] != 0]
    for s, rc, out, err in results:
        status = 'OK' if rc == 0 else f'FAIL({rc})'
        print(f'{s.name.ljust(30)} {status}')

    if failed:
        print('\nFailures details:')
        for s, rc, out, err in failed:
            print(f'--- {s.name} | Exit {rc} ---')
            print(err.strip()[:1000])
            print('---')
        sys.exit(1)
    else:
        print('\nAll scripts ran successfully')

if __name__ == '__main__':
    main()
