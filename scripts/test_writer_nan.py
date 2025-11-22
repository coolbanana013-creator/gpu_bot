import os, sys
sys.path.insert(0,'.')
from src.backtester.compact_simulator import TradeLogStreamWriter, TRADE_LOG_DTYPE
import numpy as np
from pathlib import Path
p=Path('logs')/('test_trade_log_writer.csv')
if p.exists(): p.unlink()
writer = TradeLogStreamWriter(p, batch_size=1)
writer.start()
arr = np.zeros(1, dtype=TRADE_LOG_DTYPE)
arr['bot_id']=1
arr['cycle']=0
arr['entry_price']=100.0
arr['exit_price']=101.0
arr['entry_bar']=10
arr['exit_bar']=11
arr['leverage']=1.0
arr['pnl']=float('nan')
arr['direction']=1
arr['chunk_id']=0
arr['out_of_cycle']=0
writer.enqueue(arr)
# wait for worker to flush
import time
while writer.get_stats()['written'] == 0:
    time.sleep(0.1)
writer.shutdown()
print('Done')
