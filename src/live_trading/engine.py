"""
Real-Time Trading Engine

Unified engine for both paper and live trading.
Replicates GPU kernel logic EXACTLY on CPU with live data.

COMPLETE REWRITE using gpu_kernel_port.py functions.
"""

import time
import threading
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import numpy as np

from ..bot_generator.compact_generator import CompactBotConfig
from ..utils.validation import log_info, log_error, log_warning
from .indicator_calculator import RealTimeIndicatorCalculator
from .kucoin_universal_client import KucoinUniversalClient
from .gpu_kernel_port import (
    Position,
    calculate_dynamic_slippage,
    calculate_unrealized_pnl,
    calculate_free_margin,
    check_account_liquidation,
    check_signal_reversal,
    calculate_position_size,
    apply_funding_rates,
    open_position_with_margin,
    close_position_with_margin,
    generate_signal_consensus,
    get_indicator_signal,
    RISK_STRATEGY_NAMES
)


class RealTimeTradingEngine:
    """
    Real-time trading engine that replicates GPU backtest logic EXACTLY on CPU.
    
    Supports both paper trading (Mode 2) and live trading (Mode 3).
    Uses Kucoin Universal SDK and GPU kernel port functions.
    """
    
    def __init__(
        self,
        bot_config: CompactBotConfig,
        initial_balance: float,
        kucoin_client: KucoinUniversalClient,
        pair: str = "XBTUSDTM",  # Kucoin perpetual symbol
        timeframe: str = "1m",
        test_mode: bool = True
    ):
        """
        Initialize trading engine.
        
        Args:
            bot_config: Bot configuration to trade with
            initial_balance: Starting balance
            kucoin_client: Kucoin Universal SDK client
            pair: Trading pair (Kucoin perpetual symbol)
            timeframe: Candle timeframe
            test_mode: True for paper trading (test endpoint), False for live
        """
        self.bot_config = bot_config
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.kucoin_client = kucoin_client
        self.pair = pair
        self.timeframe = timeframe
        self.test_mode = test_mode
        
        # Components
        self.indicator_calculator = RealTimeIndicatorCalculator(lookback_bars=500)
        
        # State
        self.is_running = False
        self.start_time = time.time()
        self.current_price = 0.0
        self.current_volume = 0.0
        self.current_high = 0.0
        self.current_low = 0.0
        self.last_signal = 0.0
        self.last_signal_time = 0.0
        
        # Position management (using GPU kernel Position struct)
        self.open_positions: List[Position] = []
        self.closed_positions: List[Dict] = []
        
        # Indicator tracking (for signal generation)
        self.indicator_values: Dict[int, float] = {}
        self.indicator_params: Dict[int, List[float]] = {}
        self.indicator_history: Dict[int, List[float]] = {}
        
        # Initialize indicator structures
        for i in range(bot_config.num_indicators):
            ind_idx = bot_config.indicator_indices[i]
            self.indicator_history[ind_idx] = []
            self.indicator_params[ind_idx] = [
                bot_config.indicator_params[i][0],
                bot_config.indicator_params[i][1],
                bot_config.indicator_params[i][2]
            ]
        
        # Statistics
        self.total_signals = 0
        self.buy_signals = 0
        self.sell_signals = 0
        self.candles_processed = 0
        self.bars_held = 0  # For funding rate calculation
        
        # Fee and slippage tracking
        self.total_fees = 0.0
        self.total_slippage = 0.0
        self.total_funding = 0.0
        
        # Realized PnL tracking
        self.realized_pnl = 0.0
        
        mode_str = "PAPER (Test Endpoint)" if test_mode else "LIVE (Real Money)"
        risk_strategy_name = RISK_STRATEGY_NAMES.get(bot_config.risk_strategy, "Unknown")
        
        log_info(f"🤖 Trading Engine initialized - {mode_str}")
        log_info(f"   Pair: {pair}")
        log_info(f"   Bot ID: {bot_config.bot_id}")
        log_info(f"   Indicators: {bot_config.num_indicators}")
        log_info(f"   Leverage: {bot_config.leverage}x")
        log_info(f"   TP: {bot_config.tp_multiplier*100:.2f}%, SL: {bot_config.sl_multiplier*100:.2f}%")
        log_info(f"   Risk Strategy: {risk_strategy_name} (param: {bot_config.risk_param:.4f})")
    
    def process_candle(self, open_: float, high: float, low: float, close: float, volume: float, timestamp: float):
        """
        Process new candle (EXACT PORT of GPU kernel bar-by-bar processing).
        
        Args:
            open_, high, low, close, volume: OHLCV data
            timestamp: Candle timestamp
        """
        self.current_price = close
        self.current_volume = volume
        self.current_high = high
        self.current_low = low
        self.candles_processed += 1
        self.bars_held += 1
        
        # Update price data for indicator calculation
        self.indicator_calculator.update_price_data(open_, high, low, close, volume)
        
        # === STEP 1: CALCULATE ALL INDICATORS ===
        self.indicator_values = {}
        
        for i in range(self.bot_config.num_indicators):
            ind_idx = self.bot_config.indicator_indices[i]
            param0 = self.bot_config.indicator_params[i][0]
            param1 = self.bot_config.indicator_params[i][1]
            param2 = self.bot_config.indicator_params[i][2]
            
            value = self.indicator_calculator.calculate_indicator(ind_idx, param0, param1, param2)
            self.indicator_values[ind_idx] = value
            
            # Update history (for momentum indicators)
            if ind_idx not in self.indicator_history:
                self.indicator_history[ind_idx] = []
            
            self.indicator_history[ind_idx].append(value)
            
            # Keep last 100 values
            if len(self.indicator_history[ind_idx]) > 100:
                self.indicator_history[ind_idx].pop(0)
        
        # === STEP 2: GENERATE SIGNAL (100% CONSENSUS) ===
        signal, breakdown = generate_signal_consensus(
            self.indicator_values,
            self.indicator_params,
            self.indicator_history,
            self.candles_processed,
            close
        )
        
        self.last_signal = signal
        self.last_signal_time = timestamp
        
        if signal != 0.0:
            self.total_signals += 1
            if signal > 0:
                self.buy_signals += 1
            else:
                self.sell_signals += 1
        
        # === STEP 3: UPDATE EXISTING POSITIONS ===
        self._update_positions(close)
        
        # === STEP 4: CHECK ACCOUNT LIQUIDATION ===
        if check_account_liquidation(self.current_balance, self.open_positions, close):
            log_error("⚠️ ACCOUNT LIQUIDATION TRIGGERED!")
            self._liquidate_all_positions(close)
            return
        
        # === STEP 5: CHECK SIGNAL REVERSAL FOR EXISTING POSITIONS ===
        for position in self.open_positions[:]:
            if check_signal_reversal(position, signal):
                log_info(f"🔄 Signal reversal detected - closing {('LONG' if position.side == 1 else 'SHORT')}")
                self._close_position(position, close, 'reversal')
        
        # === STEP 6: EXECUTE NEW SIGNAL IF GENERATED ===
        if signal != 0.0:
            self._execute_signal(signal, close, timestamp)
    
    def _execute_signal(self, signal: float, price: float, timestamp: float):
        """
        Execute trading signal using GPU kernel logic.
        
        EXACT PORT of GPU kernel position opening logic.
        
        Args:
            signal: 1.0 (buy) or -1.0 (sell)
            price: Current price
            timestamp: Signal timestamp
        """
        direction = 1 if signal > 0 else -1
        
        # Check if we already have a position in this direction
        for pos in self.open_positions:
            if pos.side == direction:
                log_warning(f"Already have {'LONG' if direction == 1 else 'SHORT'} position - skipping")
                return
        
        # Open position using GPU kernel logic
        position, new_balance, stats = open_position_with_margin(
            balance=self.current_balance,
            price=price,
            direction=direction,
            leverage=self.bot_config.leverage,
            tp_multiplier=self.bot_config.tp_multiplier,
            sl_multiplier=self.bot_config.sl_multiplier,
            risk_strategy=self.bot_config.risk_strategy,
            risk_param=self.bot_config.risk_param,
            current_volume=self.current_volume,
            current_high=self.current_high,
            current_low=self.current_low,
            existing_positions=self.open_positions
        )
        
        if position is None:
            reason = stats.get('reason', 'unknown')
            log_warning(f"Failed to open position: {reason}")
            return
        
        # Update balance
        self.current_balance = new_balance
        
        # Track fees and slippage
        self.total_fees += stats['entry_fee']
        self.total_slippage += stats['slippage_cost']
        
        # Set position entry time
        position.entry_time = timestamp
        
        # Add to open positions
        self.open_positions.append(position)
        
        # Execute order via Kucoin API
        side_str = "buy" if direction == 1 else "sell"
        
        try:
            order_result = self.kucoin_client.create_market_order(
                symbol=self.pair,
                side=side_str,
                size=stats['quantity'],
                leverage=self.bot_config.leverage,
                margin_mode='ISOLATED',
                reduce_only=False
            )
            
            log_info(f"✅ {'LONG' if direction == 1 else 'SHORT'} position opened @ ${price:.2f}")
            log_info(f"   Quantity: {stats['quantity']:.6f}, Margin: ${stats['margin_required']:.2f}")
            log_info(f"   TP: ${position.tp_price:.2f}, SL: ${position.sl_price:.2f}")
            log_info(f"   Fees: ${stats['entry_fee']:.2f}, Slippage: ${stats['slippage_cost']:.4f}")
            if not self.test_mode:
                log_info(f"   Order ID: {order_result.get('orderId', 'N/A')}")
            
        except Exception as e:
            log_error(f"Failed to execute order: {e}")
            # Rollback position and balance
            self.open_positions.remove(position)
            self.current_balance += stats['total_cost']
            self.total_fees -= stats['entry_fee']
            self.total_slippage -= stats['slippage_cost']
    
    def _update_positions(self, current_price: float):
        """
        Update all open positions (check TP/SL/liquidation).
        
        EXACT PORT of GPU kernel position update logic.
        
        Args:
            current_price: Current market price
        """
        for position in self.open_positions[:]:
            # Apply funding rates (every 8 hours = 480 bars at 1m)
            funding_cost, self.current_balance = apply_funding_rates(
                position, self.bars_held, self.current_balance
            )
            if funding_cost != 0.0:
                self.total_funding += funding_cost
                log_info(f"💰 Funding {'paid' if funding_cost < 0 else 'received'}: ${abs(funding_cost):.2f}")
            
            # Update unrealized PnL
            position.unrealized_pnl = calculate_unrealized_pnl(position, current_price)
            
            # Check TP
            if position.side == 1 and current_price >= position.tp_price:
                log_info(f"🎯 Take Profit hit for LONG @ ${current_price:.2f}")
                self._close_position(position, current_price, 'tp')
                continue
            elif position.side == -1 and current_price <= position.tp_price:
                log_info(f"🎯 Take Profit hit for SHORT @ ${current_price:.2f}")
                self._close_position(position, current_price, 'tp')
                continue
            
            # Check SL
            if position.side == 1 and current_price <= position.sl_price:
                log_warning(f"🛑 Stop Loss hit for LONG @ ${current_price:.2f}")
                self._close_position(position, current_price, 'sl')
                continue
            elif position.side == -1 and current_price >= position.sl_price:
                log_warning(f"🛑 Stop Loss hit for SHORT @ ${current_price:.2f}")
                self._close_position(position, current_price, 'sl')
                continue
            
            # Check individual position liquidation
            if position.side == 1 and current_price <= position.liquidation_price:
                log_error(f"💥 LONG position liquidated @ ${current_price:.2f}")
                self._close_position(position, position.liquidation_price, 'liquidation')
                continue
            elif position.side == -1 and current_price >= position.liquidation_price:
                log_error(f"💥 SHORT position liquidated @ ${current_price:.2f}")
                self._close_position(position, position.liquidation_price, 'liquidation')
                continue
    
    def _close_position(self, position: Position, exit_price: float, reason: str):
        """
        Close position using GPU kernel logic.
        
        EXACT PORT of GPU kernel position closing logic.
        
        Args:
            position: Position to close
            exit_price: Exit price
            reason: 'tp', 'sl', 'liquidation', 'reversal', 'manual'
        """
        # Calculate return using GPU kernel logic
        return_amount, stats = close_position_with_margin(
            position,
            exit_price,
            reason,
            self.current_volume,
            self.current_high,
            self.current_low
        )
        
        # Update balance
        self.current_balance += return_amount
        
        # Track fees, slippage, PnL
        self.total_fees += stats['exit_fee']
        self.total_slippage += stats['slippage_cost']
        self.realized_pnl += stats['net_pnl']
        
        # Remove from open positions
        if position in self.open_positions:
            self.open_positions.remove(position)
        
        # Add to closed positions history
        self.closed_positions.append({
            'side': 'LONG' if position.side == 1 else 'SHORT',
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'quantity': position.size,
            'leverage': position.leverage,
            'margin': stats['margin_reserved'],
            'net_pnl': stats['net_pnl'],
            'fees': stats['exit_fee'],
            'slippage': stats['slippage_cost'],
            'reason': reason,
            'timestamp': time.time()
        })
        
        # Execute close order via Kucoin API (reduce_only=True)
        side_str = "sell" if position.side == 1 else "buy"  # Opposite side to close
        
        try:
            order_result = self.kucoin_client.create_market_order(
                symbol=self.pair,
                side=side_str,
                size=position.size,
                leverage=position.leverage,
                margin_mode='ISOLATED',
                reduce_only=True
            )
            
            result_emoji = "✅" if stats['net_pnl'] >= 0 else "❌"
            log_info(f"{result_emoji} Position closed ({reason.upper()}) @ ${exit_price:.2f}")
            log_info(f"   Net PnL: ${stats['net_pnl']:.2f} ({stats['net_pnl']/stats['margin_reserved']*100:+.2f}%)")
            log_info(f"   Balance: ${self.current_balance:.2f}")
            if not self.test_mode:
                log_info(f"   Order ID: {order_result.get('orderId', 'N/A')}")
            
        except Exception as e:
            log_error(f"Failed to execute close order: {e}")
            # Position still removed from tracking, but order failed
    
    def _liquidate_all_positions(self, current_price: float):
        """
        Liquidate all positions (account-level liquidation).
        
        Args:
            current_price: Current price
        """
        for position in self.open_positions[:]:
            self._close_position(position, current_price, 'liquidation')
        
        log_error("⚠️ All positions liquidated - account blown!")
        self.stop()
    
    def get_current_state(self) -> Dict:
        """
        Get current engine state for dashboard.
        
        Returns:
            Dict with all current state information
        """
        # Calculate total unrealized PnL
        total_unrealized_pnl = sum(
            calculate_unrealized_pnl(pos, self.current_price) for pos in self.open_positions
        )
        
        # Calculate free margin
        free_margin = calculate_free_margin(
            self.current_balance, self.open_positions, self.current_price
        )
        
        # Build indicator details with signal breakdown
        indicator_details = []
        for i in range(self.bot_config.num_indicators):
            ind_idx = self.bot_config.indicator_indices[i]
            
            # Get indicator name
            indicator_name = self.indicator_calculator.all_indicators[ind_idx] if ind_idx < len(self.indicator_calculator.all_indicators) else f"IND_{ind_idx}"
            
            # Get individual signal
            history = self.indicator_history.get(ind_idx, [])
            signal = get_indicator_signal(
                ind_idx,
                self.indicator_values.get(ind_idx, 0.0),
                self.indicator_params.get(ind_idx, [0.0, 0.0, 0.0]),
                self.candles_processed,
                history,
                self.current_price
            )
            
            detail = {
                'index': ind_idx,
                'name': indicator_name,
                'value': self.indicator_values.get(ind_idx, 0.0),
                'param0': self.indicator_params[ind_idx][0],
                'param1': self.indicator_params[ind_idx][1],
                'param2': self.indicator_params[ind_idx][2],
                'signal': signal,
                'signal_str': 'BULL' if signal == 1 else 'BEAR' if signal == -1 else 'NEUT'
            }
            indicator_details.append(detail)
        
        # Build open positions list
        open_positions_list = []
        for pos in self.open_positions:
            unrealized_pnl = calculate_unrealized_pnl(pos, self.current_price)
            open_positions_list.append({
                'side': 'LONG' if pos.side == 1 else 'SHORT',
                'entry_price': pos.entry_price,
                'current_price': self.current_price,
                'quantity': pos.size,
                'leverage': pos.leverage,
                'tp_price': pos.tp_price,
                'sl_price': pos.sl_price,
                'liquidation_price': pos.liquidation_price,
                'unrealized_pnl': unrealized_pnl,
                'unrealized_pnl_pct': (unrealized_pnl / (pos.entry_price * pos.size)) * 100
            })
        
        # Get last 5 closed positions
        recent_closed = self.closed_positions[-5:] if len(self.closed_positions) > 0 else []
        
        return {
            'timestamp': time.time(),
            'uptime_seconds': time.time() - self.start_time,
            'price': self.current_price,
            'pair': self.pair,
            'timeframe': self.timeframe,
            'test_mode': self.test_mode,
            
            # Bot config
            'bot_id': self.bot_config.bot_id,
            'leverage': self.bot_config.leverage,
            'tp_multiplier': self.bot_config.tp_multiplier,
            'sl_multiplier': self.bot_config.sl_multiplier,
            'risk_strategy': RISK_STRATEGY_NAMES.get(self.bot_config.risk_strategy, "Unknown"),
            'risk_param': self.bot_config.risk_param,
            
            # Balance breakdown
            'initial_balance': self.initial_balance,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': total_unrealized_pnl,
            'current_balance': self.current_balance + total_unrealized_pnl,
            'balance_before_unrealized': self.current_balance,
            'free_margin': free_margin,
            
            # Fees and costs
            'total_fees': self.total_fees,
            'total_slippage': self.total_slippage,
            'total_funding': self.total_funding,
            
            # Indicators
            'indicator_details': indicator_details,
            'indicator_count': self.bot_config.num_indicators,
            
            # Signal
            'current_signal': self.last_signal,
            'total_signals': self.total_signals,
            'buy_signals': self.buy_signals,
            'sell_signals': self.sell_signals,
            
            # Positions
            'open_positions': open_positions_list,
            'open_positions_count': len(self.open_positions),
            'closed_positions': recent_closed,
            'closed_positions_count': len(self.closed_positions),
            
            # Stats
            'candles_processed': self.candles_processed,
            'bars_held': self.bars_held
        }
    
    def start(self):
        """Start the trading engine."""
        self.is_running = True
        self.start_time = time.time()
        mode_str = "PAPER TRADING" if self.test_mode else "LIVE TRADING"
        log_info(f"🚀 {mode_str} ENGINE STARTED")
    
    def stop(self):
        """Stop the trading engine."""
        self.is_running = False
        log_info("🛑 Trading engine stopped")
        
        # Close all open positions
        for position in self.open_positions[:]:
            self._close_position(position, self.current_price, "shutdown")
        
        # Calculate win rate
        wins = sum(1 for p in self.closed_positions if p['net_pnl'] > 0)
        losses = sum(1 for p in self.closed_positions if p['net_pnl'] <= 0)
        win_rate = (wins / len(self.closed_positions) * 100) if len(self.closed_positions) > 0 else 0.0
        
        # Print final summary
        total_unrealized = sum(calculate_unrealized_pnl(pos, self.current_price) for pos in self.open_positions)
        final_balance = self.current_balance + total_unrealized
        total_pnl = final_balance - self.initial_balance
        
        log_info(f"\n{'='*70}")
        log_info(f"FINAL SUMMARY - {'PAPER MODE' if self.test_mode else 'LIVE MODE'}")
        log_info(f"{'='*70}")
        log_info(f"Initial Balance:     ${self.initial_balance:,.2f}")
        log_info(f"Realized PnL:        ${self.realized_pnl:,.2f}")
        log_info(f"Unrealized PnL:      ${total_unrealized:,.2f}")
        log_info(f"Final Balance:       ${final_balance:,.2f}")
        log_info(f"Total PnL:           ${total_pnl:,.2f} ({total_pnl/self.initial_balance*100:+.2f}%)")
        log_info(f"")
        log_info(f"Total Fees:          ${self.total_fees:,.2f}")
        log_info(f"Total Slippage:      ${self.total_slippage:,.2f}")
        log_info(f"Total Funding:       ${self.total_funding:,.2f}")
        log_info(f"")
        log_info(f"Positions Closed:    {len(self.closed_positions)}")
        log_info(f"Wins / Losses:       {wins} / {losses}")
        log_info(f"Win Rate:            {win_rate:.1f}%")
        log_info(f"")
        log_info(f"Signals Generated:   {self.total_signals} ({self.buy_signals} buy, {self.sell_signals} sell)")
        log_info(f"Candles Processed:   {self.candles_processed:,}")
        log_info(f"Uptime:              {time.time() - self.start_time:.0f} seconds")
        log_info(f"{'='*70}\n")
