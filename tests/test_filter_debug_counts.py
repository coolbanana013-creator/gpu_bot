import os
import csv
from pathlib import Path
from importlib import reload

def test_analyze_filter_debug_counts(tmp_path, capsys):
    # Create a temporary logs directory
    logs_dir = tmp_path / 'logs'
    logs_dir.mkdir()

    # Create a sample filter_debug_counts.csv file
    fc_path = logs_dir / 'filter_debug_counts.csv'
    with open(fc_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['BotID', 'ADX', 'ATR', 'VOLUME', 'SR', 'RSI', 'NAN'])
        writer.writerow([1, 3, 1, 0, 0, 0, 0])
        writer.writerow([2, 0, 2, 0, 1, 0, 1])
        writer.writerow([3, 5, 0, 1, 0, 2, 0])

    # Create a dummy filter_reenable_summary.csv (with expected columns for the analyzer)
    summary_path = logs_dir / 'filter_reenable_summary.csv'
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        headers = ['bot_id', 'all_bypass', 'all_bypass_filter_bits', 'quality_on_srvol_bypass', 'quality_on_srvol_bypass_filter_bits', 'quality_on_volume_on', 'quality_on_volume_on_filter_bits', 'all_on', 'all_on_filter_bits']
        writer.writerow(headers)
        # Each config column contains a '|' separated string of trades per cycle (we use 5 cycles here)
        writer.writerow([1, '0|0|0|0|0', '0x0|0x0|0x0|0x0|0x0', '0|0|0|0|0', '0x0|0x0|0x0|0x0|0x0', '0|0|0|0|0', '0x0|0x0|0x0|0x0|0x0', '0|0|0|0|0', '0x01|0x00|0x02|0x00|0x00'])

    # Run the analyzer script with PYTHONPATH pointing to project root
    import sys
    sys.path.insert(0, os.path.abspath('.'))
    from scripts import analyze_filter_debug
    # Patch paths in the script to use tmp logs directory
    analyze_filter_debug.Path = Path
    # Change working directory to project root to match script expectations
    os.chdir(str(tmp_path))
    # Execute main and capture output
    reload(analyze_filter_debug)
    analyze_filter_debug.main()
    captured = capsys.readouterr()
    assert 'Aggregated per-bot filter counts' in captured.out
    assert 'ADX' in captured.out
