"""
Live Trading Dashboard

Real-time display of trading status, indicators, positions, and PnL.
"""

import os
import time
from typing import Dict
from datetime import datetime

from ..utils.validation import log_info


class LiveDashboard:
    """Terminal-based live trading dashboard."""
    
    def __init__(self):
        """Initialize dashboard."""
        self.update_interval = 1.0  # Update every second
        self.last_update = 0
    
    def clear_screen(self):
        """Clear terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def render(self, state: Dict):
        """
        Render dashboard with current state.
        
        Args:
            state: State dict from RealTimeTradingEngine.get_current_state()
        """
        current_time = time.time()
        if current_time - self.last_update < self.update_interval:
            return
        
        self.last_update = current_time
        self.clear_screen()
        
        # Header
        print("=" * 80)
        print(f"{'GPU BOT - LIVE TRADING DASHBOARD':^80}")
        print("=" * 80)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Pair: {state['pair']} | Timeframe: {state['timeframe']} | Bot ID: {state['bot_id']}")
        print("=" * 80)
        
        # Current Price
        price = state['price']
        print(f"\n💰 CURRENT PRICE: ${price:,.2f}")
        
        # Signal Status
        signal = state['current_signal']
        signal_breakdown = state['signal_breakdown']
        
        print(f"\n📊 SIGNAL STATUS:")
        if signal > 0:
            print(f"   Status: 🟢 BUY SIGNAL")
        elif signal < 0:
            print(f"   Status: 🔴 SELL SIGNAL")
        else:
            print(f"   Status: ⚪ NO SIGNAL (No consensus)")
        
        print(f"   Bullish:  {signal_breakdown['bullish_count']}/{state['indicator_count']} indicators")
        print(f"   Bearish:  {signal_breakdown['bearish_count']}/{state['indicator_count']} indicators")
        print(f"   Neutral:  {signal_breakdown['neutral_count']}/{state['indicator_count']} indicators")
        print(f"   Total Signals Generated: {state['total_signals']} ({state['buy_signals']} buy, {state['sell_signals']} sell)")
        
        # Indicators
        print(f"\n📈 INDICATORS (Leverage: {state['leverage']}x, TP: {state['tp_multiplier']:.2f}x, SL: {state['sl_multiplier']:.2f}x):")
        print(f"{'Name':<20} {'Value':<15} {'Signal':<10} {'Parameters':<30}")
        print("-" * 80)
        
        for ind in state['indicator_details']:
            signal_str = "🟢 BUY" if ind['signal'] == 1 else ("🔴 SELL" if ind['signal'] == -1 else "⚪ NEUTRAL")
            params = f"[{ind['param0']:.2f}, {ind['param1']:.2f}, {ind['param2']:.2f}]"
            print(f"{ind['name']:<20} {ind['value']:<15.4f} {signal_str:<10} {params:<30}")
        
        # Positions
        pos_summary = state['position_summary']
        print(f"\n💼 POSITIONS:")
        print(f"   Open: {pos_summary.open_positions} | Closed: {pos_summary.closed_positions}")
        print(f"   Unrealized PnL: ${pos_summary.unrealized_pnl:+,.2f}")
        print(f"   Realized PnL:   ${pos_summary.realized_pnl:+,.2f}")
        print(f"   Total PnL:      ${pos_summary.total_pnl:+,.2f}")
        print(f"   Total Fees:     ${pos_summary.total_fees:,.2f}")
        print(f"   Win Rate:       {pos_summary.win_rate*100:.1f}%")
        
        # Balance
        pnl_pct = (pos_summary.total_pnl / pos_summary.current_balance * 100) if pos_summary.current_balance > 0 else 0
        print(f"\n💵 BALANCE:")
        print(f"   Current: ${pos_summary.current_balance:,.2f} ({pnl_pct:+.2f}%)")
        
        # Stats
        print(f"\n📊 STATISTICS:")
        print(f"   Candles Processed: {state['candles_processed']}")
        
        print("\n" + "=" * 80)
        print("Press Ctrl+C to stop trading")
        print("=" * 80)
    
    def render_simple(self, message: str):
        """Render simple message."""
        self.clear_screen()
        print("=" * 80)
        print(f"{'GPU BOT - LIVE TRADING':^80}")
        print("=" * 80)
        print(f"\n{message}\n")
        print("=" * 80)
