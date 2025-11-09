"""
Real-Time Trading Engine

Unified engine for both paper and live trading.
Replicates GPU kernel logic exactly on CPU with live data.
"""

import time
import threading
from typing import Optional, Dict, List
from datetime import datetime
import numpy as np

from ..bot_generator.compact_generator import CompactBotConfig
from ..utils.validation import log_info, log_error, log_warning
from .indicator_calculator import RealTimeIndicatorCalculator
from .signal_generator import SignalGenerator
from .position_manager import PositionManager, PaperPositionManager, LivePositionManager
from .risk_manager import RiskManager


class RealTimeTradingEngine:
    """
    Real-time trading engine that replicates GPU backtest logic on CPU.
    
    Supports both paper trading (Mode 2) and live trading (Mode 3).
    """
    
    def __init__(
        self,
        bot_config: CompactBotConfig,
        initial_balance: float,
        position_manager: PositionManager,
        pair: str = "BTC/USDT",
        timeframe: str = "1m"
    ):
        """
        Initialize trading engine.
        
        Args:
            bot_config: Bot configuration to trade with
            initial_balance: Starting balance
            position_manager: Paper or Live position manager
            pair: Trading pair
            timeframe: Candle timeframe
        """
        self.bot_config = bot_config
        self.initial_balance = initial_balance
        self.position_manager = position_manager
        self.pair = pair
        self.timeframe = timeframe
        
        # Components
        self.indicator_calculator = RealTimeIndicatorCalculator(lookback_bars=500)
        self.signal_generator = SignalGenerator()
        self.risk_manager = RiskManager(leverage=bot_config.leverage)
        
        # State
        self.is_running = False
        self.current_price = 0.0
        self.last_signal = 0.0
        self.last_signal_time = 0.0
        
        # Indicator tracking
        self.indicator_values: Dict[int, float] = {}
        self.indicator_history: Dict[int, List[float]] = {}
        
        # Initialize history buffers
        for i in range(bot_config.num_indicators):
            ind_idx = bot_config.indicator_indices[i]
            self.indicator_history[ind_idx] = []
        
        # Statistics
        self.total_signals = 0
        self.buy_signals = 0
        self.sell_signals = 0
        self.candles_processed = 0
        
        log_info(f"🤖 Trading Engine initialized for {pair}")
        log_info(f"   Bot ID: {bot_config.bot_id}")
        log_info(f"   Indicators: {bot_config.num_indicators}")
        log_info(f"   Leverage: {bot_config.leverage}x")
        log_info(f"   TP: {bot_config.tp_multiplier:.2f}x, SL: {bot_config.sl_multiplier:.2f}x")
    
    def process_candle(self, open_: float, high: float, low: float, close: float, volume: float, timestamp: float):
        """
        Process new candle (matches GPU kernel bar-by-bar processing).
        
        Args:
            open_, high, low, close, volume: OHLCV data
            timestamp: Candle timestamp
        """
        self.current_price = close
        self.candles_processed += 1
        
        # Update price data
        self.indicator_calculator.update_price_data(open_, high, low, close, volume)
        
        # Calculate all indicators for this bot
        self.indicator_values = {}
        
        for i in range(self.bot_config.num_indicators):
            ind_idx = self.bot_config.indicator_indices[i]
            param0 = self.bot_config.indicator_params[i][0]
            param1 = self.bot_config.indicator_params[i][1]
            param2 = self.bot_config.indicator_params[i][2]
            
            value = self.indicator_calculator.calculate_indicator(ind_idx, param0, param1, param2)
            self.indicator_values[ind_idx] = value
            
            # Update history
            if ind_idx not in self.indicator_history:
                self.indicator_history[ind_idx] = []
            
            self.indicator_history[ind_idx].append(value)
            
            # Keep last 100 values
            if len(self.indicator_history[ind_idx]) > 100:
                self.indicator_history[ind_idx].pop(0)
        
        # Generate signal (100% consensus)
        # Convert lists to numpy arrays for signal generator
        indicator_history_arrays = {
            ind_idx: np.array(hist) for ind_idx, hist in self.indicator_history.items()
        }
        
        signal = self.signal_generator.generate_signal(
            self.indicator_values,
            indicator_history_arrays
        )
        
        self.last_signal = signal
        self.last_signal_time = timestamp
        
        if signal != 0.0:
            self.total_signals += 1
            if signal > 0:
                self.buy_signals += 1
            else:
                self.sell_signals += 1
        
        # Update existing positions (check TP/SL)
        closed_positions = self.position_manager.update_positions(close)
        
        # Execute signal if generated
        if signal != 0.0:
            self._execute_signal(signal, close, timestamp)
    
    def _execute_signal(self, signal: float, price: float, timestamp: float):
        """
        Execute trading signal (open position).
        
        Args:
            signal: 1.0 (buy) or -1.0 (sell)
            price: Current price
            timestamp: Signal timestamp
        """
        side = 1 if signal > 0 else -1
        
        # Check if we already have a position in this direction
        for pos in self.position_manager.open_positions:
            if pos.side == side:
                return  # Already have position in this direction
        
        # Calculate position size using risk management
        size = self.risk_manager.calculate_position_size(
            balance=self.position_manager.current_balance,
            price=price,
            leverage=self.bot_config.leverage
        )
        
        if size <= 0:
            log_warning(f"Position size too small: {size}")
            return
        
        # Calculate TP/SL prices (matches GPU kernel logic)
        if side == 1:  # Long
            tp_price = price * (1 + self.bot_config.tp_multiplier)
            sl_price = price * (1 - self.bot_config.sl_multiplier)
        else:  # Short
            tp_price = price * (1 - self.bot_config.tp_multiplier)
            sl_price = price * (1 + self.bot_config.sl_multiplier)
        
        # Open position
        position = self.position_manager.open_position(
            side=side,
            price=price,
            size=size,
            leverage=self.bot_config.leverage,
            tp_price=tp_price,
            sl_price=sl_price
        )
        
        if position:
            signal_type = "BUY" if side == 1 else "SELL"
            log_info(f"🎯 SIGNAL: {signal_type} @ ${price:.2f}")
    
    def get_current_state(self) -> Dict:
        """
        Get current engine state for dashboard.
        
        Returns:
            Dict with all current state information
        """
        # Get signal breakdown
        indicator_history_arrays = {
            ind_idx: np.array(hist) for ind_idx, hist in self.indicator_history.items()
        }
        
        signal_breakdown = self.signal_generator.get_signal_breakdown(
            self.indicator_values,
            indicator_history_arrays
        )
        
        # Get position summary
        position_summary = self.position_manager.get_summary(self.current_price)
        
        # Build indicator details
        indicator_details = []
        for i in range(self.bot_config.num_indicators):
            ind_idx = self.bot_config.indicator_indices[i]
            
            # Get indicator name
            indicator_name = self.indicator_calculator.all_indicators[ind_idx] if ind_idx < len(self.indicator_calculator.all_indicators) else f"IND_{ind_idx}"
            
            detail = {
                'index': ind_idx,
                'name': indicator_name,
                'value': self.indicator_values.get(ind_idx, 0.0),
                'param0': self.bot_config.indicator_params[i][0],
                'param1': self.bot_config.indicator_params[i][1],
                'param2': self.bot_config.indicator_params[i][2],
                'signal': signal_breakdown['signals_by_indicator'].get(ind_idx, 0)
            }
            indicator_details.append(detail)
        
        return {
            'timestamp': time.time(),
            'price': self.current_price,
            'pair': self.pair,
            'timeframe': self.timeframe,
            
            # Bot config
            'bot_id': self.bot_config.bot_id,
            'leverage': self.bot_config.leverage,
            'tp_multiplier': self.bot_config.tp_multiplier,
            'sl_multiplier': self.bot_config.sl_multiplier,
            
            # Indicators
            'indicator_details': indicator_details,
            'indicator_count': self.bot_config.num_indicators,
            
            # Signal
            'current_signal': self.last_signal,
            'signal_breakdown': signal_breakdown,
            'total_signals': self.total_signals,
            'buy_signals': self.buy_signals,
            'sell_signals': self.sell_signals,
            
            # Positions
            'position_summary': position_summary,
            'open_positions': len(self.position_manager.open_positions),
            'closed_positions': len(self.position_manager.closed_positions),
            
            # Stats
            'candles_processed': self.candles_processed,
            'uptime': time.time() - (self.last_signal_time if self.last_signal_time > 0 else time.time())
        }
    
    def start(self):
        """Start the trading engine."""
        self.is_running = True
        log_info("🚀 Trading engine started")
    
    def stop(self):
        """Stop the trading engine."""
        self.is_running = False
        log_info("🛑 Trading engine stopped")
        
        # Close all open positions
        for position in self.position_manager.open_positions[:]:
            self.position_manager.close_position(position, self.current_price, "shutdown")
        
        # Print final summary
        summary = self.position_manager.get_summary(self.current_price)
        log_info(f"\n{'='*60}")
        log_info(f"FINAL SUMMARY")
        log_info(f"{'='*60}")
        log_info(f"Initial Balance: ${self.initial_balance:,.2f}")
        log_info(f"Final Balance:   ${summary.current_balance:,.2f}")
        log_info(f"Total PnL:       ${summary.total_pnl:,.2f} ({summary.total_pnl/self.initial_balance*100:+.2f}%)")
        log_info(f"Total Fees:      ${summary.total_fees:,.2f}")
        log_info(f"Positions:       {summary.closed_positions} closed")
        log_info(f"Win Rate:        {summary.win_rate*100:.1f}%")
        log_info(f"Signals:         {self.total_signals} ({self.buy_signals} buy, {self.sell_signals} sell)")
        log_info(f"{'='*60}\n")
