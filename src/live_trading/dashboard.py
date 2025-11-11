"""
Live Trading Dashboard - COMPLETE REWRITE

Real-time display with 7 comprehensive sections:
1. Mode Banner (PAPER vs LIVE)
2. Runtime & Price Info
3. Balance Breakdown (Initial + Realized + Unrealized = Current)
4. Leverage & Risk Strategy
5. INDICATOR THRESHOLD TABLE (Current Value vs Bullish Condition vs Bearish Condition)
6. Open Positions Detail
7. Closed Positions Detail (Last 5)

NO SIMPLIFICATIONS - Full GPU kernel parity.
"""

import os
import time
from typing import Dict
from datetime import datetime, timedelta


def format_seconds(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    return str(timedelta(seconds=int(seconds)))


class LiveDashboard:
    """
    Terminal-based live trading dashboard.
    
    Displays comprehensive real-time information matching GPU backtest detail level.
    """
    
    def __init__(self):
        """Initialize dashboard."""
        self.update_interval = 1.0  # Update every second
        self.last_update = 0
        self.term_width = 150  # Wide display for all columns
    
    def clear_screen(self):
        """Clear terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def render(self, state: Dict):
        """
        Render complete dashboard with current state.
        
        Args:
            state: State dict from RealTimeTradingEngine.get_current_state()
        """
        current_time = time.time()
        if current_time - self.last_update < self.update_interval:
            return
        
        self.last_update = current_time
        self.clear_screen()
        
        # ====================================================================
        # SECTION 1: MODE BANNER
        # ====================================================================
        mode_str = "🧪 PAPER TRADING (Test Endpoint)" if state['test_mode'] else "💰 LIVE TRADING (Real Money)"
        mode_color = "GREEN" if state['test_mode'] else "RED"
        
        print("=" * self.term_width)
        print(f"{mode_str:^{self.term_width}}")
        print("=" * self.term_width)
        
        # ====================================================================
        # SECTION 2: RUNTIME & PRICE INFO
        # ====================================================================
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        runtime = format_seconds(state['uptime_seconds'])
        
        print(f"Time: {now} | Runtime: {runtime} | Pair: {state['pair']} | Timeframe: {state['timeframe']}")
        print(f"Bot ID: {state['bot_id']} | Price: ${state['price']:,.2f} | Candles: {state['candles_processed']:,}")
        print("-" * self.term_width)
        
        # ====================================================================
        # SECTION 3: BALANCE BREAKDOWN
        # ====================================================================
        initial = state['initial_balance']
        realized = state['realized_pnl']
        unrealized = state['unrealized_pnl']
        current = state['current_balance']
        free_margin = state['free_margin']
        
        total_pnl = realized + unrealized
        total_pnl_pct = (total_pnl / initial * 100) if initial > 0 else 0.0
        
        print(f"\n� BALANCE BREAKDOWN:")
        print(f"   Initial:     ${initial:>12,.2f}")
        print(f"   Realized:    ${realized:>12,.2f}  {'+' if realized >= 0 else ''}{realized/initial*100:.2f}%")
        print(f"   Unrealized:  ${unrealized:>12,.2f}  {'+' if unrealized >= 0 else ''}{unrealized/initial*100 if initial > 0 else 0:.2f}%")
        print(f"   {'─'*35}")
        print(f"   Current:     ${current:>12,.2f}  {'+' if total_pnl >= 0 else ''}{total_pnl_pct:.2f}%")
        print(f"   Free Margin: ${free_margin:>12,.2f}")
        print(f"\n   Fees:        ${state['total_fees']:>12,.2f}")
        print(f"   Slippage:    ${state['total_slippage']:>12,.2f}")
        print(f"   Funding:     ${state['total_funding']:>12,.2f}  {'(received)' if state['total_funding'] > 0 else '(paid)'}")
        
        # ====================================================================
        # SECTION 4: LEVERAGE & RISK STRATEGY
        # ====================================================================
        print(f"\n⚙️  TRADING CONFIGURATION:")
        print(f"   Leverage:        {state['leverage']}x")
        print(f"   Take Profit:     {state['tp_multiplier']*100:.2f}%")
        print(f"   Stop Loss:       {state['sl_multiplier']*100:.2f}%")
        print(f"   Risk Strategy:   {state['risk_strategy']}")
        print(f"   Risk Parameter:  {state['risk_param']:.4f}")
        
        # ====================================================================
        # SECTION 5: INDICATOR THRESHOLD TABLE (CRITICAL FEATURE)
        # ====================================================================
        print(f"\n📊 INDICATOR THRESHOLDS (100% Consensus Required):")
        print(f"{'Indicator':<25} {'Current Value':<18} {'Bullish When':<22} {'Bearish When':<22} {'Signal':<8}")
        print("─" * self.term_width)
        
        for ind in state['indicator_details']:
            ind_name = ind['name'][:24]  # Truncate long names
            current_val = ind['value']
            
            # Determine bullish/bearish conditions based on indicator type
            bullish_condition, bearish_condition = self._get_indicator_conditions(ind)
            
            # Signal emoji
            if ind['signal_str'] == 'BULL':
                signal_emoji = "🟢 BULL"
            elif ind['signal_str'] == 'BEAR':
                signal_emoji = "🔴 BEAR"
            else:
                signal_emoji = "⚪ NEUT"
            
            print(f"{ind_name:<25} {current_val:<18.4f} {bullish_condition:<22} {bearish_condition:<22} {signal_emoji:<8}")
        
        # Overall signal consensus
        signal = state['current_signal']
        if signal > 0:
            consensus_str = "🟢 ALL BULLISH → BUY SIGNAL"
        elif signal < 0:
            consensus_str = "🔴 ALL BEARISH → SELL SIGNAL"
        else:
            consensus_str = "⚪ NO CONSENSUS → NO SIGNAL"
        
        print("─" * self.term_width)
        print(f"{'CONSENSUS:':<25} {consensus_str}")
        print(f"{'SIGNALS GENERATED:':<25} {state['total_signals']} total ({state['buy_signals']} buy, {state['sell_signals']} sell)")
        
        # ====================================================================
        # SECTION 6: OPEN POSITIONS DETAIL
        # ====================================================================
        open_positions = state['open_positions']
        print(f"\n📈 OPEN POSITIONS ({len(open_positions)}):")
        
        if len(open_positions) > 0:
            print(f"{'Side':<8} {'Entry':<12} {'Current':<12} {'Size':<12} {'Lev':<6} {'TP':<12} {'SL':<12} {'Liq':<12} {'PnL':<15} {'%':<10}")
            print("─" * self.term_width)
            
            for pos in open_positions:
                side_emoji = "🟢 LONG" if pos['side'] == 'LONG' else "🔴 SHORT"
                pnl_str = f"${pos['unrealized_pnl']:+,.2f}"
                pnl_pct_str = f"{pos['unrealized_pnl_pct']:+.2f}%"
                
                print(f"{side_emoji:<8} "
                      f"${pos['entry_price']:<11,.2f} "
                      f"${pos['current_price']:<11,.2f} "
                      f"{pos['quantity']:<12.6f} "
                      f"{pos['leverage']:<6}x "
                      f"${pos['tp_price']:<11,.2f} "
                      f"${pos['sl_price']:<11,.2f} "
                      f"${pos['liquidation_price']:<11,.2f} "
                      f"{pnl_str:<15} "
                      f"{pnl_pct_str:<10}")
        else:
            print("   No open positions")
        
        # ====================================================================
        # SECTION 7: CLOSED POSITIONS DETAIL (Last 5)
        # ====================================================================
        closed_positions = state['closed_positions']
        print(f"\n� CLOSED POSITIONS (Last 5 of {state['closed_positions_count']}):")
        
        if len(closed_positions) > 0:
            print(f"{'Side':<8} {'Entry':<12} {'Exit':<12} {'Size':<12} {'Lev':<6} {'PnL':<15} {'Fees':<12} {'Reason':<12}")
            print("─" * self.term_width)
            
            for pos in reversed(closed_positions):  # Most recent first
                side_emoji = "🟢 LONG" if pos['side'] == 'LONG' else "🔴 SHORT"
                pnl_str = f"${pos['net_pnl']:+,.2f}"
                result_emoji = "✅" if pos['net_pnl'] >= 0 else "❌"
                
                print(f"{side_emoji:<8} "
                      f"${pos['entry_price']:<11,.2f} "
                      f"${pos['exit_price']:<11,.2f} "
                      f"{pos['quantity']:<12.6f} "
                      f"{pos['leverage']:<6}x "
                      f"{pnl_str:<15} "
                      f"${pos['fees']:<11,.2f} "
                      f"{result_emoji} {pos['reason']:<10}")
        else:
            print("   No closed positions yet")
        
        # ====================================================================
        # FOOTER
        # ====================================================================
        print("\n" + "=" * self.term_width)
        if state['test_mode']:
            print(f"{'⚠️  PAPER TRADING MODE - No real money at risk':^{self.term_width}}")
        else:
            print(f"{'⚠️  LIVE TRADING MODE - Real money at risk!':^{self.term_width}}")
        print(f"{'Press Ctrl+C to stop trading':^{self.term_width}}")
        print("=" * self.term_width)
    
    def _get_indicator_conditions(self, ind: Dict) -> tuple:
        """
        Get bullish and bearish condition descriptions for an indicator.
        
        Args:
            ind: Indicator detail dict
        
        Returns:
            Tuple of (bullish_condition_str, bearish_condition_str)
        """
        ind_idx = ind['index']
        
        # Moving Averages (0-11): trend-following
        if 0 <= ind_idx <= 11:
            return ("Rising > +0.1%", "Falling < -0.1%")
        
        # RSI (12-14): overbought/oversold
        if 12 <= ind_idx <= 14:
            return ("< 30 (oversold)", "> 70 (overbought)")
        
        # Stochastic %K (15)
        if ind_idx == 15:
            return ("< 20 (oversold)", "> 80 (overbought)")
        
        # StochRSI (16)
        if ind_idx == 16:
            return ("< 20 (oversold)", "> 80 (overbought)")
        
        # Momentum (17)
        if ind_idx == 17:
            return ("> 0 (uptrend)", "< 0 (downtrend)")
        
        # ROC (18)
        if ind_idx == 18:
            return ("> 2% (strong up)", "< -2% (strong down)")
        
        # Williams %R (19)
        if ind_idx == 19:
            return ("< -80 (oversold)", "> -20 (overbought)")
        
        # ATR (20-21), NATR (22): volatility
        if 20 <= ind_idx <= 22:
            return ("Increasing", "Decreasing")
        
        # Bollinger Upper (23)
        if ind_idx == 23:
            return ("Expanding > +0.2%", "Price near band")
        
        # Bollinger Lower (24)
        if ind_idx == 24:
            return ("Price near band", "Expanding < -0.2%")
        
        # Keltner (25)
        if ind_idx == 25:
            return ("Expanding", "Contracting")
        
        # MACD (26)
        if ind_idx == 26:
            return ("> 0 or bullish cross", "< 0 or bearish cross")
        
        # ADX (27)
        if ind_idx == 27:
            return ("> 25 & rising", "< 25 or falling")
        
        # Aroon Up (28)
        if ind_idx == 28:
            return ("> 70 (recent high)", "< 30 (no recent high)")
        
        # CCI (29)
        if ind_idx == 29:
            return ("< -100 (oversold)", "> 100 (overbought)")
        
        # DPO (30)
        if ind_idx == 30:
            return ("> 0 (above trend)", "< 0 (below trend)")
        
        # Parabolic SAR (31)
        if ind_idx == 31:
            return ("Dropping (uptrend)", "Rising (downtrend)")
        
        # SuperTrend (32)
        if ind_idx == 32:
            return ("Rising > +0.1%", "Falling < -0.1%")
        
        # Trend Strength (33-35)
        if 33 <= ind_idx <= 35:
            return ("> 0 (upslope)", "< 0 (downslope)")
        
        # OBV (36)
        if ind_idx == 36:
            return ("Rising", "Falling")
        
        # VWAP (37)
        if ind_idx == 37:
            return ("Rising", "Falling")
        
        # MFI (38)
        if ind_idx == 38:
            return ("< 20 (oversold)", "> 80 (overbought)")
        
        # A/D (39)
        if ind_idx == 39:
            return ("Rising (accum)", "Falling (distrib)")
        
        # Volume SMA (40)
        if ind_idx == 40:
            return ("> +20% (surge)", "< -20% (low)")
        
        # Pivot Points (41)
        if ind_idx == 41:
            return ("Above pivot", "Below pivot")
        
        # Fractal High (42)
        if ind_idx == 42:
            return ("No fractal high", "Fractal high (resist)")
        
        # Fractal Low (43)
        if ind_idx == 43:
            return ("Fractal low (support)", "No fractal low")
        
        # Support/Resistance (44)
        if ind_idx == 44:
            return ("Breaking resistance", "Breaking support")
        
        # Price Channel (45)
        if ind_idx == 45:
            return ("Rising", "Falling")
        
        # High-Low Range (46)
        if ind_idx == 46:
            return ("> +50% (breakout)", "Normal range")
        
        # Close Position (47)
        if ind_idx == 47:
            return ("> 0.7 (near high)", "< 0.3 (near low)")
        
        # Price Acceleration (48)
        if ind_idx == 48:
            return ("> 0 (accelerating)", "< 0 (decelerating)")
        
        # Volume ROC (49)
        if ind_idx == 49:
            return ("> 10% (surge)", "< -10% (decline)")
        
        # Default
        return ("Rising", "Falling")
    
    def render_simple(self, message: str):
        """Render simple message."""
        self.clear_screen()
        print("=" * self.term_width)
        print(f"{'GPU BOT - LIVE TRADING':^{self.term_width}}")
        print("=" * self.term_width)
        print(f"\n{message}\n")
        print("=" * self.term_width)
