"""
Live Trading Dashboard - Real-time Indicator Display

Shows bot indicators, their current values, and signal conditions.
Displays what needs to happen for buy/sell signals.
"""

import os
import sys
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np

from ..utils.validation import log_info
from ..bot_generator.compact_generator import CompactBotConfig
from .gpu_kernel_port import RISK_STRATEGY_NAMES


# ANSI color codes for terminal
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    # Text colors
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    
    # Background colors
    BG_RED = '\033[101m'
    BG_GREEN = '\033[102m'
    BG_YELLOW = '\033[103m'
    BG_BLUE = '\033[104m'


# Indicator names mapping (index -> name)
INDICATOR_NAMES = {
    0: "SMA", 1: "EMA", 2: "WMA", 3: "DEMA", 4: "TEMA",
    5: "KAMA", 6: "TRIMA", 7: "T3", 8: "HMA", 9: "ZLEMA",
    10: "BBands_Upper", 11: "BBands_Middle", 12: "BBands_Lower",
    13: "BBands_Width", 14: "BBands_%B",
    15: "RSI", 16: "Stoch_K", 17: "Stoch_D", 18: "StochRSI",
    19: "Williams_%R", 20: "ROC", 21: "Momentum",
    22: "CCI", 23: "CMO", 24: "Aroon_Up", 25: "Aroon_Down",
    26: "MACD", 27: "MACD_Signal", 28: "MACD_Hist",
    29: "PPO", 30: "ADX", 31: "DI+", 32: "DI-",
    33: "ATR", 34: "NATR", 35: "TRANGE",
    36: "SAR", 37: "SuperTrend", 38: "Ichimoku_Tenkan",
    39: "Ichimoku_Kijun", 40: "Ichimoku_SenkouA", 41: "Ichimoku_SenkouB",
    42: "OBV", 43: "AD", 44: "ADOSC", 45: "MFI",
    46: "Chaikin", 47: "Force_Index", 48: "EOM", 49: "VWAP"
}


class LiveTradingDashboard:
    """
    Real-time trading dashboard showing indicators, values, and signal conditions.
    """
    
    def __init__(self, bot_config: CompactBotConfig, mode: str = "PAPER"):
        """
        Initialize dashboard.
        
        Args:
            bot_config: Bot configuration with indicators
            mode: "PAPER" or "LIVE"
        """
        self.bot_config = bot_config
        self.mode = mode
        self.last_update = 0
        self.update_interval = 1  # Update every second minimum
        
        # Indicator signal thresholds (from GPU kernel logic)
        self.thresholds = {
            'RSI': {'oversold': 30, 'overbought': 70},
            'Stoch_K': {'oversold': 20, 'overbought': 80},
            'Stoch_D': {'oversold': 20, 'overbought': 80},
            'StochRSI': {'oversold': 0.2, 'overbought': 0.8},
            'Williams_%R': {'oversold': -80, 'overbought': -20},
            'CCI': {'oversold': -100, 'overbought': 100},
            'MFI': {'oversold': 20, 'overbought': 80},
            'ADX': {'trending': 25, 'strong': 50}
        }
    
    def clear_screen(self):
        """Clear terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def get_indicator_name(self, idx: int) -> str:
        """Get indicator name from index."""
        return INDICATOR_NAMES.get(idx, f"Unknown_{idx}")
    
    def get_indicator_signal(self, indicator_name: str, value: float, price: float, position: int) -> Dict:
        """
        Determine indicator signal (BUY/SELL/NEUTRAL).
        
        This replicates the GPU kernel get_indicator_signal logic.
        
        Returns:
            Dict with 'signal', 'condition', and 'color'
        """
        
        # RSI
        if indicator_name == 'RSI':
            if value < self.thresholds['RSI']['oversold']:
                return {'signal': 'BUY', 'condition': f'RSI < {self.thresholds["RSI"]["oversold"]}', 'color': Colors.GREEN}
            elif value > self.thresholds['RSI']['overbought']:
                return {'signal': 'SELL', 'condition': f'RSI > {self.thresholds["RSI"]["overbought"]}', 'color': Colors.RED}
            else:
                return {'signal': 'NEUTRAL', 'condition': f'{self.thresholds["RSI"]["oversold"]} < RSI < {self.thresholds["RSI"]["overbought"]}', 'color': Colors.YELLOW}
        
        # Stochastic K
        elif indicator_name == 'Stoch_K':
            if value < self.thresholds['Stoch_K']['oversold']:
                return {'signal': 'BUY', 'condition': f'Stoch_K < {self.thresholds["Stoch_K"]["oversold"]}', 'color': Colors.GREEN}
            elif value > self.thresholds['Stoch_K']['overbought']:
                return {'signal': 'SELL', 'condition': f'Stoch_K > {self.thresholds["Stoch_K"]["overbought"]}', 'color': Colors.RED}
            else:
                return {'signal': 'NEUTRAL', 'condition': f'{self.thresholds["Stoch_K"]["oversold"]} < Stoch_K < {self.thresholds["Stoch_K"]["overbought"]}', 'color': Colors.YELLOW}
        
        # Stochastic D
        elif indicator_name == 'Stoch_D':
            if value < self.thresholds['Stoch_D']['oversold']:
                return {'signal': 'BUY', 'condition': f'Stoch_D < {self.thresholds["Stoch_D"]["oversold"]}', 'color': Colors.GREEN}
            elif value > self.thresholds['Stoch_D']['overbought']:
                return {'signal': 'SELL', 'condition': f'Stoch_D > {self.thresholds["Stoch_D"]["overbought"]}', 'color': Colors.RED}
            else:
                return {'signal': 'NEUTRAL', 'condition': f'{self.thresholds["Stoch_D"]["oversold"]} < Stoch_D < {self.thresholds["Stoch_D"]["overbought"]}', 'color': Colors.YELLOW}
        
        # Williams %R
        elif indicator_name == 'Williams_%R':
            if value < self.thresholds['Williams_%R']['oversold']:
                return {'signal': 'BUY', 'condition': f'Williams %R < {self.thresholds["Williams_%R"]["oversold"]}', 'color': Colors.GREEN}
            elif value > self.thresholds['Williams_%R']['overbought']:
                return {'signal': 'SELL', 'condition': f'Williams %R > {self.thresholds["Williams_%R"]["overbought"]}', 'color': Colors.RED}
            else:
                return {'signal': 'NEUTRAL', 'condition': f'{self.thresholds["Williams_%R"]["oversold"]} < Williams %R < {self.thresholds["Williams_%R"]["overbought"]}', 'color': Colors.YELLOW}
        
        # CCI
        elif indicator_name == 'CCI':
            if value < self.thresholds['CCI']['oversold']:
                return {'signal': 'BUY', 'condition': f'CCI < {self.thresholds["CCI"]["oversold"]}', 'color': Colors.GREEN}
            elif value > self.thresholds['CCI']['overbought']:
                return {'signal': 'SELL', 'condition': f'CCI > {self.thresholds["CCI"]["overbought"]}', 'color': Colors.RED}
            else:
                return {'signal': 'NEUTRAL', 'condition': f'{self.thresholds["CCI"]["oversold"]} < CCI < {self.thresholds["CCI"]["overbought"]}', 'color': Colors.YELLOW}
        
        # MFI
        elif indicator_name == 'MFI':
            if value < self.thresholds['MFI']['oversold']:
                return {'signal': 'BUY', 'condition': f'MFI < {self.thresholds["MFI"]["oversold"]}', 'color': Colors.GREEN}
            elif value > self.thresholds['MFI']['overbought']:
                return {'signal': 'SELL', 'condition': f'MFI > {self.thresholds["MFI"]["overbought"]}', 'color': Colors.RED}
            else:
                return {'signal': 'NEUTRAL', 'condition': f'{self.thresholds["MFI"]["oversold"]} < MFI < {self.thresholds["MFI"]["overbought"]}', 'color': Colors.YELLOW}
        
        # MACD Histogram
        elif indicator_name == 'MACD_Hist':
            if value > 0:
                return {'signal': 'BUY', 'condition': 'MACD Hist > 0 (bullish)', 'color': Colors.GREEN}
            elif value < 0:
                return {'signal': 'SELL', 'condition': 'MACD Hist < 0 (bearish)', 'color': Colors.RED}
            else:
                return {'signal': 'NEUTRAL', 'condition': 'MACD Hist = 0', 'color': Colors.YELLOW}
        
        # ADX (trend strength)
        elif indicator_name == 'ADX':
            if value > self.thresholds['ADX']['strong']:
                return {'signal': 'STRONG TREND', 'condition': f'ADX > {self.thresholds["ADX"]["strong"]} (very strong)', 'color': Colors.MAGENTA}
            elif value > self.thresholds['ADX']['trending']:
                return {'signal': 'TRENDING', 'condition': f'ADX > {self.thresholds["ADX"]["trending"]} (trending)', 'color': Colors.CYAN}
            else:
                return {'signal': 'WEAK', 'condition': f'ADX < {self.thresholds["ADX"]["trending"]} (weak/ranging)', 'color': Colors.YELLOW}
        
        # Moving averages (compare with price)
        elif indicator_name in ['SMA', 'EMA', 'WMA', 'DEMA', 'TEMA', 'KAMA', 'HMA', 'ZLEMA']:
            if price > value:
                return {'signal': 'BUY', 'condition': f'Price > {indicator_name} (bullish)', 'color': Colors.GREEN}
            elif price < value:
                return {'signal': 'SELL', 'condition': f'Price < {indicator_name} (bearish)', 'color': Colors.RED}
            else:
                return {'signal': 'NEUTRAL', 'condition': f'Price = {indicator_name}', 'color': Colors.YELLOW}
        
        # Bollinger Bands %B
        elif indicator_name == 'BBands_%B':
            if value < 0:
                return {'signal': 'BUY', 'condition': '%B < 0 (below lower band)', 'color': Colors.GREEN}
            elif value > 1:
                return {'signal': 'SELL', 'condition': '%B > 1 (above upper band)', 'color': Colors.RED}
            else:
                return {'signal': 'NEUTRAL', 'condition': '0 < %B < 1 (within bands)', 'color': Colors.YELLOW}
        
        # Default: compare with zero
        else:
            if value > 0:
                return {'signal': 'POSITIVE', 'condition': 'Value > 0', 'color': Colors.GREEN}
            elif value < 0:
                return {'signal': 'NEGATIVE', 'condition': 'Value < 0', 'color': Colors.RED}
            else:
                return {'signal': 'NEUTRAL', 'condition': 'Value = 0', 'color': Colors.YELLOW}
    
    def render(self, state: Dict):
        """
        Render live dashboard with indicator values and signal conditions.
        
        Args:
            state: Current engine state with indicators, price, positions, etc.
        """
        current_time = datetime.now().timestamp()
        
        # Only update if enough time has passed
        if current_time - self.last_update < self.update_interval:
            return
        
        self.last_update = current_time
        
        # Clear screen for fresh render
        self.clear_screen()
        
        # Header
        mode_color = Colors.YELLOW if self.mode == "PAPER" else Colors.RED
        print(f"\n{Colors.BOLD}{'='*100}{Colors.RESET}")
        print(f"{Colors.BOLD}{mode_color}{'📊 LIVE TRADING DASHBOARD - ' + self.mode + ' MODE':^100}{Colors.RESET}")
        print(f"{Colors.BOLD}{'='*100}{Colors.RESET}\n")
        
        # Current time and price
        timestamp = state.get('timestamp', current_time)
        time_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        current_price = state.get('price', 0.0)
        
        print(f"{Colors.CYAN}Time:{Colors.RESET} {time_str}  |  ", end='')
        print(f"{Colors.CYAN}Price:{Colors.RESET} ${current_price:,.2f}  |  ", end='')
        print(f"{Colors.CYAN}Balance:{Colors.RESET} ${state.get('balance', 0):,.2f}")
        print()
        
        # Account status
        print(f"{Colors.BOLD}{'ACCOUNT STATUS':^100}{Colors.RESET}")
        print(f"{'-'*100}")
        
        realized_pnl = state.get('realized_pnl', 0)
        unrealized_pnl = state.get('unrealized_pnl', 0)
        total_pnl = realized_pnl + unrealized_pnl
        pnl_color = Colors.GREEN if total_pnl >= 0 else Colors.RED
        
        print(f"Realized PnL: {pnl_color}${realized_pnl:+,.2f}{Colors.RESET}  |  ", end='')
        print(f"Unrealized PnL: {pnl_color}${unrealized_pnl:+,.2f}{Colors.RESET}  |  ", end='')
        print(f"Total PnL: {pnl_color}${total_pnl:+,.2f}{Colors.RESET}")
        print()
        
        # Position status
        open_positions = state.get('open_positions', [])
        position_side = state.get('position_side', 'NONE')
        position_size = state.get('position_size', 0)
        
        if len(open_positions) > 0:
            position_color = Colors.GREEN if position_side == 'LONG' else Colors.RED
            print(f"Position: {position_color}{position_side} {position_size:.4f} contracts{Colors.RESET}")
        else:
            print(f"Position: {Colors.YELLOW}NO POSITION{Colors.RESET}")
        print()
        
        # Bot configuration
        print(f"{Colors.BOLD}{'BOT CONFIGURATION':^100}{Colors.RESET}")
        print(f"{'-'*100}")
        print(f"Bot ID: {self.bot_config.bot_id}  |  ", end='')
        print(f"Leverage: {self.bot_config.leverage}x  |  ", end='')
        print(f"TP: {self.bot_config.tp_multiplier*100:.2f}%  |  ", end='')
        print(f"SL: {self.bot_config.sl_multiplier*100:.2f}%")
        
        risk_strategy_name = RISK_STRATEGY_NAMES.get(self.bot_config.risk_strategy, "Unknown")
        print(f"Risk Strategy: {risk_strategy_name} (param: {self.bot_config.risk_param:.4f})")
        print()
        
        # Indicators section
        print(f"{Colors.BOLD}{'INDICATORS & SIGNAL CONDITIONS':^100}{Colors.RESET}")
        print(f"{'-'*100}")
        print(f"{'Indicator':<20} {'Value':<15} {'Signal':<15} {'Condition':<50}")
        print(f"{'-'*100}")
        
        # Get indicator values from state
        indicator_values = state.get('indicator_values', {})
        
        # Display each bot indicator
        for i in range(self.bot_config.num_indicators):
            ind_idx = self.bot_config.indicator_indices[i]
            ind_name = self.get_indicator_name(ind_idx)
            ind_params = self.bot_config.indicator_params[i]
            
            # Get current value
            value = indicator_values.get(ind_idx, 0.0)
            
            # Get signal analysis
            signal_info = self.get_indicator_signal(
                ind_name, 
                value, 
                current_price,
                1 if position_side == 'LONG' else (-1 if position_side == 'SHORT' else 0)
            )
            
            # Format output
            signal_color = signal_info['color']
            params_str = f"({ind_params[0]:.1f}, {ind_params[1]:.1f}, {ind_params[2]:.1f})"
            
            print(f"{ind_name:<20} ", end='')
            print(f"{value:<15.4f} ", end='')
            print(f"{signal_color}{signal_info['signal']:<15}{Colors.RESET} ", end='')
            print(f"{signal_info['condition']:<50}")
        
        print()
        
        # Signal consensus
        print(f"{Colors.BOLD}{'SIGNAL CONSENSUS':^100}{Colors.RESET}")
        print(f"{'-'*100}")
        
        buy_signals = 0
        sell_signals = 0
        neutral_signals = 0
        
        for i in range(self.bot_config.num_indicators):
            ind_idx = self.bot_config.indicator_indices[i]
            ind_name = self.get_indicator_name(ind_idx)
            value = indicator_values.get(ind_idx, 0.0)
            
            signal_info = self.get_indicator_signal(ind_name, value, current_price, 0)
            
            if 'BUY' in signal_info['signal'] or 'POSITIVE' in signal_info['signal']:
                buy_signals += 1
            elif 'SELL' in signal_info['signal'] or 'NEGATIVE' in signal_info['signal']:
                sell_signals += 1
            else:
                neutral_signals += 1
        
        total_signals = self.bot_config.num_indicators
        
        print(f"{Colors.GREEN}BUY Signals: {buy_signals}/{total_signals} ({buy_signals/total_signals*100:.1f}%){Colors.RESET}  |  ", end='')
        print(f"{Colors.RED}SELL Signals: {sell_signals}/{total_signals} ({sell_signals/total_signals*100:.1f}%){Colors.RESET}  |  ", end='')
        print(f"{Colors.YELLOW}NEUTRAL: {neutral_signals}/{total_signals} ({neutral_signals/total_signals*100:.1f}%){Colors.RESET}")
        
        # Overall consensus
        if buy_signals > sell_signals and buy_signals > neutral_signals:
            consensus = f"{Colors.GREEN}BULLISH ↗{Colors.RESET}"
        elif sell_signals > buy_signals and sell_signals > neutral_signals:
            consensus = f"{Colors.RED}BEARISH ↘{Colors.RESET}"
        else:
            consensus = f"{Colors.YELLOW}NEUTRAL ↔{Colors.RESET}"
        
        print(f"\n{Colors.BOLD}Overall Consensus: {consensus}{Colors.RESET}")
        print()
        
        # Trading stats
        print(f"{Colors.BOLD}{'TRADING STATISTICS':^100}{Colors.RESET}")
        print(f"{'-'*100}")
        
        closed_positions = state.get('closed_positions', 0)
        wins = state.get('wins', 0)
        losses = state.get('losses', 0)
        win_rate = state.get('win_rate', 0)
        
        print(f"Closed Positions: {closed_positions}  |  ", end='')
        print(f"Wins: {Colors.GREEN}{wins}{Colors.RESET}  |  ", end='')
        print(f"Losses: {Colors.RED}{losses}{Colors.RESET}  |  ", end='')
        
        win_rate_color = Colors.GREEN if win_rate >= 50 else (Colors.YELLOW if win_rate >= 40 else Colors.RED)
        print(f"Win Rate: {win_rate_color}{win_rate:.1f}%{Colors.RESET}")
        
        total_fees = state.get('total_fees', 0)
        total_slippage = state.get('total_slippage', 0)
        
        print(f"Total Fees: ${total_fees:.2f}  |  ", end='')
        print(f"Total Slippage: ${total_slippage:.2f}")
        
        signals_generated = state.get('signals_generated', 0)
        buy_signal_count = state.get('buy_signals', 0)
        sell_signal_count = state.get('sell_signals', 0)
        candles_processed = state.get('candles_processed', 0)
        
        print(f"Signals Generated: {signals_generated} ({buy_signal_count} buy, {sell_signal_count} sell)  |  ", end='')
        print(f"Candles Processed: {candles_processed}")
        
        print()
        print(f"{Colors.BOLD}{'='*100}{Colors.RESET}")
        print(f"{Colors.CYAN}Press Ctrl+C to stop trading{Colors.RESET}\n")
