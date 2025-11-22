import os, csv, glob, sys
from pathlib import Path
repo_root = str(Path(__file__).resolve().parents[1])
sys.path.append(repo_root)
import numpy as np
import pyopencl as cl
from src.bot_generator.compact_generator import CompactBotConfig
from src.data_provider.loader import DataLoader
from src.backtester.compact_simulator import CompactBacktester

# Enable trade logs
os.environ['ENABLE_TRADE_LOGS'] = '1'
os.environ['TRADE_LOG_MAX'] = '50000'

BOT_ID = 5

# Load bot from generation_0.csv
row = None
with open('logs/generation_0.csv', newline='', encoding='utf-8') as f:
    for r in csv.DictReader(f, delimiter=';'):
        if r['BotID'] == str(BOT_ID):
            row = r
            break

if not row:
    print(f'Bot {BOT_ID} not found in generation_0.csv')
    raise SystemExit(1)

print(f"\n{'='*70}")
print(f"TRACING BOT {BOT_ID}")
print(f"{'='*70}")
print(f"Original Results from Generation CSV:")
print(f"  Total Trades: {row['TotalTrades']}")
print(f"  Total PnL: {row['TotalPnL']}")
print(f"  Win Rate: {row['TotalWinRate']}%")
print(f"  Sharpe Ratio: {row['SharpeRatio']}")
print(f"  Max Drawdown: {row['MaxDrawdown']}")
print(f"  Fitness Score: {row['FitnessScore']}")
print(f"\nPer-Cycle Results:")
for i in range(int(row['NumCycles'])):
    trades_key = f'Cycle{i}_Trades'
    pnl_key = f'Cycle{i}_TotalPnL'
    wr_key = f'Cycle{i}_WinRate'
    if trades_key in row:
        print(f"  Cycle {i}: {row[trades_key]} trades, PnL={row[pnl_key]}, WinRate={row[wr_key]}%")

# Parse bot configuration
params = [p.strip() for p in row['IndicatorParams'].split('|')]
inds = []
iparams = []

from src.indicators.gpu_indicators import get_gpu_indicator_name
from src.utils.indicator_parser import parse_indicator_params

for p in params:
    name = p.split('(')[0].strip()
    idx = None
    for i in range(50):
        if get_gpu_indicator_name(i).split('(')[0].lower().startswith(name.lower().split('(')[0]):
            idx = i
            break
    if idx is None:
        continue
    inds.append(idx)
    
    if '(' in p and ')' in p:
        inside = p[p.rfind('(')+1:p.rfind(')')]
        try:
            pars = [float(x) for x in inside.split(',')]
        except Exception:
            parsed = parse_indicator_params(p)
            pars = parsed[0] if parsed else [0.0, 0.0, 0.0]
    else:
        pars = [0.0, 0.0, 0.0]
    
    while len(pars) < 3:
        pars.append(0.0)
    iparams.append(pars[:3])

while len(inds) < 8:
    inds.append(0)
while len(iparams) < 8:
    iparams.append([0.0, 0.0, 0.0])

try:
    risk = float(row['RiskStrategies'].split('(')[1].split(')')[0])
except Exception:
    risk = 0.05

bot = CompactBotConfig(
    bot_id=BOT_ID,
    num_indicators=sum(1 for i in inds if i != 0),
    indicator_indices=np.array(inds, dtype=np.uint8),
    indicator_params=np.array(iparams, dtype=np.float32),
    indicator_risk_strategies=np.array([0]*8, dtype=np.uint8),
    risk_param=risk,
    tp_multiplier=float(row['TPMultiplier'].replace(',', '.')) if row.get('TPMultiplier') else 1.0,
    sl_multiplier=float(row['SLMultiplier'].replace(',', '.')) if row.get('SLMultiplier') else 1.0,
    leverage=int(float(row['Leverage'])),
    survival_generations=0
)

print(f"\n{'='*70}")
print(f"BOT CONFIGURATION:")
print(f"{'='*70}")
print(f"Bot ID: {bot.bot_id}")
print(f"Indicators: {bot.num_indicators}")
print(f"Indicator Indices: {inds[:bot.num_indicators]}")
print(f"Leverage: {bot.leverage}")
print(f"Risk Param: {bot.risk_param}")

# Load data - use same date range as original run
# Original run: 24 days total (2 cycles × 3 days = 6 days + buffer)
all_file_paths = sorted(Path(p) for p in glob.glob(str(Path('data')/'BTC_USDT'/'1m'/'*.parquet')))
num_cycles = int(row['NumCycles'])
days_per_cycle = 3  # From original run
total_days_needed = num_cycles * days_per_cycle + 18  # Add buffer for warmup

# Take only the most recent files to match original run
file_paths = all_file_paths[-total_days_needed:] if len(all_file_paths) > total_days_needed else all_file_paths

print(f"\nUsing {len(file_paths)} most recent data files (approx {total_days_needed} days)")

ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)
loader = DataLoader(
    file_paths=file_paths,
    timeframe='1m',
    random_seed=None,  # Match GA run
    gpu_context=ctx,
    gpu_queue=queue,
    use_gpu_processing=False
)

ohlcv = loader.load_all_data()

# Generate same cycles as original run
cycles = loader.generate_cycle_ranges(num_cycles, 3)  # 3 days per cycle
print(f"\n{'='*70}")
print(f"DATA LOADED:")
print(f"{'='*70}")
print(f"Total bars: {len(ohlcv)}")
print(f"Cycles: {num_cycles}")
for i, (start, end) in enumerate(cycles):
    print(f"  Cycle {i}: bars {start}-{end} ({end-start+1} bars)")

# Clear old logs
trade_logs_path = Path('logs') / 'trade_logs.csv'
close_counters_path = Path('logs') / 'close_counters.csv'
if trade_logs_path.exists():
    trade_logs_path.unlink()
if close_counters_path.exists():
    close_counters_path.unlink()

# Run backtest
print(f"\n{'='*70}")
print(f"RUNNING BACKTEST:")
print(f"{'='*70}")

try:
    backtester = CompactBacktester(
        gpu_context=ctx,
        gpu_queue=queue,
        initial_balance=10.0  # Match GA initial balance
    )
    results = backtester.backtest_bots([bot], ohlcv, cycles)
    res = results[0]
    
    print(f"\nBacktest Results:")
    print(f"  Total Trades: {res.total_trades}")
    print(f"  Total PnL: {res.total_pnl:.2f}")
    print(f"  Win Rate: {res.win_rate:.2f}%")
    print(f"  Sharpe Ratio: {res.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {res.max_drawdown:.2f}")
    print(f"  Fitness Score: {res.fitness_score:.2f}")
    print(f"\nPer-Cycle Results:")
    for i in range(len(res.per_cycle_pnl)):
        trades = res.per_cycle_trades[i] if hasattr(res, 'per_cycle_trades') else 'N/A'
        wins = res.per_cycle_wins[i] if hasattr(res, 'per_cycle_wins') else 'N/A'
        pnl = res.per_cycle_pnl[i]
        print(f"  Cycle {i}: {trades} trades, {wins} wins, PnL={pnl:.2f}")
    
    # Compare with original
    print(f"\n{'='*70}")
    print(f"COMPARISON WITH GENERATION LOG:")
    print(f"{'='*70}")
    
    orig_trades = int(row['TotalTrades'])
    orig_pnl = float(row['TotalPnL'].replace(',', '.'))
    orig_wr = float(row['TotalWinRate'].replace(',', '.'))
    
    trade_diff = res.total_trades - orig_trades
    pnl_diff = res.total_pnl - orig_pnl
    wr_diff = res.win_rate - orig_wr
    
    print(f"Total Trades: {res.total_trades} vs {orig_trades} (diff: {trade_diff:+d})")
    print(f"Total PnL: {res.total_pnl:.2f} vs {orig_pnl:.2f} (diff: {pnl_diff:+.2f})")
    print(f"Win Rate: {res.win_rate:.2f}% vs {orig_wr:.2f}% (diff: {wr_diff:+.2f}%)")
    
    if abs(trade_diff) > 0:
        print(f"\n⚠️  WARNING: Trade count mismatch detected!")
    if abs(pnl_diff) > 0.01:
        print(f"\n⚠️  WARNING: PnL mismatch detected!")
    if abs(wr_diff) > 0.1:
        print(f"\n⚠️  WARNING: Win rate mismatch detected!")
    
    if abs(trade_diff) == 0 and abs(pnl_diff) < 0.01 and abs(wr_diff) < 0.1:
        print(f"\n✅ PERFECT MATCH! Backtest reproduces generation results exactly.")
    
    # Analyze trade logs
    print(f"\n{'='*70}")
    print(f"TRADE LOG ANALYSIS:")
    print(f"{'='*70}")
    
    if trade_logs_path.exists():
        trades_by_cycle = {}
        pnl_by_cycle = {}
        
        with open(trade_logs_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=';')
            for r in reader:
                if int(r['BotID']) == BOT_ID:
                    cycle = int(r['Cycle'])
                    pnl = float(r['PnL'].replace(',', '.'))
                    
                    if cycle not in trades_by_cycle:
                        trades_by_cycle[cycle] = 0
                        pnl_by_cycle[cycle] = 0.0
                    
                    trades_by_cycle[cycle] += 1
                    pnl_by_cycle[cycle] += pnl
        
        print(f"Logged Trades by Cycle:")
        for cycle in sorted(trades_by_cycle.keys()):
            print(f"  Cycle {cycle}: {trades_by_cycle[cycle]} trades, PnL={pnl_by_cycle[cycle]:.2f}")
    
    # Analyze close counters
    if close_counters_path.exists():
        print(f"\nKernel Close Counters:")
        with open(close_counters_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=';')
            for r in reader:
                if int(r['BotID']) == BOT_ID:
                    print(f"  Cycle {r['Cycle']}: {r['KernelCloseCount']} closes (kernel-counted)")
    
    print(f"\n{'='*70}")
    print(f"TRACE COMPLETE")
    print(f"{'='*70}\n")

except Exception as e:
    import traceback
    traceback.print_exc()
    raise
