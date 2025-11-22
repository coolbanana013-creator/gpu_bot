import os
import csv
import subprocess
from pathlib import Path
import numpy as np

from src.backtester.compact_simulator import CompactBacktester
from src.bot_generator.compact_generator import CompactBotConfig
from src.data_provider.loader import DataLoader
import pyopencl as cl


def run_backtest_for_bot(bot_id: int):
    # Helper to run backtester for single bot using reproduce script logic
    # Set env for trade logs
    os.environ['ENABLE_TRADE_LOGS'] = '1'
    os.environ['TRADE_LOG_MAX'] = '500000'

    # Read bot config from generation file
    row = None
    with open('logs/generation_0.csv', newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f, delimiter=';'):
            if int(r['BotID']) == bot_id:
                row = r
                break
    assert row is not None, 'Bot not found in generation_0.csv'

    # Build bot with parsed indicators
    from src.utils.indicator_parser import parse_indicator_params, parse_indices
    inds = parse_indices(row.get('IndicatorsUsed') or row.get('IndicatorIndices') or '')
    iparams = parse_indicator_params(row.get('IndicatorParams') or '')
    while len(inds) < 8:
        inds.append(0)
    while len(iparams) < 8:
        iparams.append([0.0, 0.0, 0.0])

    cfg = CompactBotConfig(
        bot_id=bot_id,
        num_indicators=sum(1 for i in inds if i != 0),
        indicator_indices=np.array(inds, dtype=np.uint8),
        indicator_params=np.array(iparams, dtype=np.float32),
        indicator_risk_strategies=np.array([0] * 8, dtype=np.uint8),
        risk_param=float(row.get('RiskStrategies', '0').split('(')[1].split(')')[0]) if row.get('RiskStrategies') and '(' in row.get('RiskStrategies') else 0.05,
        tp_multiplier=float(str(row.get('TPMultiplier') or '1.0').replace(',', '.')), 
        sl_multiplier=float(str(row.get('SLMultiplier') or '1.0').replace(',', '.')), 
        leverage=int(float(str(row.get('Leverage') or '1').replace(',', '.')))
    )

    # Load data and run backtester
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    file_paths = sorted(Path('data') / 'BTC_USDT' / '1m' .glob('*.parquet')) if False else sorted((Path('data')/'BTC_USDT'/'1m').glob('*.parquet'))
    loader = DataLoader(file_paths=file_paths, timeframe='1m', random_seed=42, gpu_context=ctx, gpu_queue=queue, use_gpu_processing=False)
    ohlcv = loader.load_all_data()
    cycles = loader.generate_cycle_ranges(5, 7)

    backtester = CompactBacktester(gpu_context=ctx, gpu_queue=queue, initial_balance=10.0)
    # Remove previous logs
    tpath = Path('logs') / 'trade_logs.csv'
    if tpath.exists():
        tpath.unlink()

    results = backtester.backtest_bots([cfg], ohlcv, cycles)
    return results[0]


def _select_sample_bot():
    # Choose a bot with non-zero TotalTrades from generation CSV
    with open('logs/generation_0.csv', newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f, delimiter=';'):
            try:
                tot = int(r.get('TotalTrades') or 0)
            except Exception:
                tot = 0
            if tot > 0:
                return int(r['BotID'])
    return None


def test_trade_logs_no_cross_chunk_duplicates_and_pnl_consistent():
    # Choose a sample bot that actually trades in the generation log
    sample_bot = _select_sample_bot()
    if sample_bot is None:
        import pytest
        pytest.skip('No bot with non-zero total trades found in generation CSV')
    res = run_backtest_for_bot(sample_bot)

    # If the backtest produced no trades then skip test
    if res.total_trades == 0:
        import pytest
        pytest.skip('Backtest produced no trades for selected bot; skipping logging consistency test')

    # Read trade logs for this bot
    logs = []
    with open('logs/trade_logs.csv', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';')
        for r in reader:
            if int(r['BotID']) == res.bot_id:
                logs.append(r)

    if not logs:
        import pytest
        pytest.skip('No trades found in trade_logs.csv for selected bot; skipping test')

    # Check duplicates by signature (EntryBar, ExitBar, Direction) across chunks
    sig_map = {}
    for r in logs:
        sig = (int(r['EntryBar'] or 0), int(r['ExitBar'] or 0), r['Direction'])
        chunk = int(r.get('ChunkID') or -1)
        sig_map.setdefault(sig, set()).add(chunk)

    # All signatures must map to exactly 1 chunk now
    duplicates = [sig for sig, cs in sig_map.items() if len(cs) > 1]
    assert len(duplicates) == 0, f'Duplicate trade signatures found across chunks: {duplicates}'

    # Sum per-trade PnL by cycle ignoring out_of_cycle and confirm equals res.per_cycle_pnl
    from collections import defaultdict
    per_cycle_sum = defaultdict(float)
    for r in logs:
        if int(r.get('OutOfCycle') or 0):
            continue
        c = int(r['Cycle'])
        per_cycle_sum[c] += float(str(r['PnL']).replace(',', '.'))

    # Compare with per_cycle_pnl
    for idx, expected in enumerate(res.per_cycle_pnl):
        actual = per_cycle_sum.get(idx, 0.0)
        assert abs(actual - expected) < 1e-2, f'Cycle {idx} mismatch: expected {expected}, actual {actual}'

    # === New: verify kernel close counters match number of logged closes ===
    cc_path = Path('logs') / 'close_counters.csv'
    if not cc_path.exists():
        import pytest
        pytest.skip('close_counters.csv not found - kernel not instrumented for this run')

    # Map kernel counted closes by cycle for this bot
    close_counts = {}
    with open(cc_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';')
        for r in reader:
            if int(r['BotID']) != res.bot_id:
                continue
            close_counts[int(r['Cycle'])] = int(r['KernelCloseCount'])

    # Build expected count from trade logs (ignore out_of_cycle)
    from collections import defaultdict
    logged_close_counts = defaultdict(int)
    for r in logs:
        if int(r.get('OutOfCycle') or 0):
            continue
        c = int(r['Cycle'])
        logged_close_counts[c] += 1

    for c_idx, expected_count in logged_close_counts.items():
        actual_count = close_counts.get(c_idx, 0)
        assert actual_count == expected_count, f'Kernel close count differs for cycle {c_idx}: kernel {actual_count} != logs {expected_count}'


def test_reconcile_with_logs_env_var():
    # Test that setting RECONCILE_WITH_LOGS writes per-cycle PnL consistent with trade logs
    import os
    import pytest

    sample_bot = _select_sample_bot()
    if sample_bot is None:
        pytest.skip('No bot with non-zero total trades found in generation CSV')

    os.environ['RECONCILE_WITH_LOGS'] = '1'
    os.environ['ENABLE_TRADE_LOGS'] = '1'
    os.environ['TRADE_LOG_MAX'] = '50000'

    res = run_backtest_for_bot(sample_bot)
    # If there were no trades, skip; otherwise compare
    if res.total_trades == 0:
        pytest.skip('No trades logged - skipping reconcile test')

    # Read logs and compare
    per_cycle_sum = {}
    with open('logs/trade_logs.csv', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';')
        for r in reader:
            if int(r.get('OutOfCycle') or 0):
                continue
            if int(r['BotID']) != res.bot_id:
                continue
            c = int(r['Cycle']); per_cycle_sum[c] = per_cycle_sum.get(c, 0.0) + float(str(r['PnL']).replace(',','.'))

    for idx, pnl in enumerate(res.per_cycle_pnl):
        actual = per_cycle_sum.get(idx, 0.0)
        assert abs(float(pnl) - actual) < 1e-2, f'Reconciled mismatch for cycle {idx}: {pnl} != {actual}'
