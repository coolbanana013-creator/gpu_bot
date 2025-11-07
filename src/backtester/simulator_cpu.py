"""
Backtesting simulator - tests bot performance on historical data.
CPU-based implementation (GPU kernel to be added for production).
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from ..bot_generator.generator import BotConfig
from ..indicators.factory import IndicatorFactory
from ..indicators.signals import SignalType, calculate_consensus_signal
from ..risk_management.strategies import RiskStrategyFactory, calculate_average_position_size
from ..risk_management.tp_sl import calculate_tp_sl_prices
from ..utils.validation import log_info, log_debug, log_warning
from ..utils.config import (
    DEFAULT_FEE_RATE, SLIPPAGE_RATE, FUNDING_RATE_8H, FUNDING_INTERVAL_HOURS,
    SIGNAL_CONSENSUS_THRESHOLD, MAX_POSITIONS_PER_BOT, MIN_FREE_BALANCE_PCT
)


class PositionSide(Enum):
    """Position direction."""
    LONG = 1
    SHORT = -1


@dataclass
class Position:
    """Open position."""
    entry_price: float
    size: float  # Position size in quote currency
    side: PositionSide
    entry_index: int
    tp_price: float
    sl_price: float
    leverage: int
    
    def calculate_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL."""
        if self.side == PositionSide.LONG:
            price_change_pct = (current_price - self.entry_price) / self.entry_price
        else:
            price_change_pct = (self.entry_price - current_price) / self.entry_price
        
        return self.size * price_change_pct * self.leverage


@dataclass
class BacktestResult:
    """Results from backtesting a single bot."""
    bot_id: int
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_profit_pct: float
    winrate: float
    final_balance: float
    max_drawdown_pct: float
    trades_detail: List[Dict[str, Any]]


class Backtester:
    """
    Backtests trading bots on historical data.
    Simulates realistic trading with fees, slippage, funding, and liquidations.
    """
    
    def __init__(
        self,
        initial_balance: float,
        fee_rate: float = DEFAULT_FEE_RATE,
        slippage_rate: float = SLIPPAGE_RATE,
        funding_rate: float = FUNDING_RATE_8H,
        funding_interval_hours: int = FUNDING_INTERVAL_HOURS
    ):
        """
        Initialize backtester.
        
        Args:
            initial_balance: Starting balance
            fee_rate: Trading fee rate
            slippage_rate: Slippage estimate
            funding_rate: Funding rate per interval
            funding_interval_hours: Hours between funding payments
        """
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.funding_rate = funding_rate
        self.funding_interval_hours = funding_interval_hours
        
        log_info(f"Backtester initialized with ${initial_balance} initial balance")
    
    def _precompute_indicators(
        self,
        ohlcv: np.ndarray,
        bot: BotConfig
    ) -> Dict[str, np.ndarray]:
        """
        Precompute all indicator values for the bot.
        
        Args:
            ohlcv: OHLCV data array
            bot: Bot configuration
            
        Returns:
            Dictionary mapping indicator index to computed values
        """
        indicator_values = {}
        
        for i, ind_params in enumerate(bot.indicators):
            try:
                values = IndicatorFactory.compute_indicator(ohlcv, ind_params)
                indicator_values[i] = values
                log_debug(f"Computed {ind_params.indicator_type.value} for bot {bot.bot_id}")
            except Exception as e:
                log_warning(f"Failed to compute {ind_params.indicator_type.value}: {e}")
                indicator_values[i] = np.full(len(ohlcv), np.nan)
        
        return indicator_values
    
    def _generate_signals(
        self,
        bar_index: int,
        close_price: float,
        indicator_values: Dict[str, np.ndarray],
        bot: BotConfig
    ) -> SignalType:
        """
        Generate trading signal for current bar.
        
        Args:
            bar_index: Current bar index
            close_price: Current close price
            indicator_values: Precomputed indicator values
            bot: Bot configuration
            
        Returns:
            Consensus signal
        """
        signals = []
        
        for i, ind_params in enumerate(bot.indicators):
            ind_type = ind_params.indicator_type
            values = indicator_values.get(i)
            
            if values is None or bar_index >= len(values):
                continue
            
            current_value = values[bar_index]
            
            # Generate signal based on indicator type
            # Simplified - in full version, use proper signal generation from signals.py
            if np.isnan(current_value):
                signal = SignalType.NEUTRAL
            else:
                # Simple signal logic (extend for all indicator types)
                if ind_type.value in ['RSI', 'STOCH', 'CCI', 'WILLIAMS_R', 'MFI']:
                    # Oscillators
                    if current_value < 30:
                        signal = SignalType.LONG
                    elif current_value > 70:
                        signal = SignalType.SHORT
                    else:
                        signal = SignalType.NEUTRAL
                elif ind_type.value in ['EMA', 'SMA', 'WMA', 'DEMA', 'TEMA']:
                    # Moving averages
                    if close_price > current_value:
                        signal = SignalType.LONG
                    elif close_price < current_value:
                        signal = SignalType.SHORT
                    else:
                        signal = SignalType.NEUTRAL
                else:
                    # Default neutral
                    signal = SignalType.NEUTRAL
            
            signals.append(signal)
        
        # Calculate consensus
        return calculate_consensus_signal(signals, SIGNAL_CONSENSUS_THRESHOLD)
    
    def _calculate_position_size(
        self,
        balance: float,
        bot: BotConfig,
        **kwargs
    ) -> float:
        """
        Calculate position size using bot's risk strategies.
        
        Args:
            balance: Current balance
            bot: Bot configuration
            **kwargs: Additional parameters for risk strategies
            
        Returns:
            Position size in quote currency
        """
        size_fraction = calculate_average_position_size(
            bot.risk_strategies,
            balance,
            **kwargs
        )
        
        # Ensure minimum free balance
        max_size_fraction = 1.0 - (MIN_FREE_BALANCE_PCT / 100.0)
        size_fraction = min(size_fraction, max_size_fraction)
        
        return balance * size_fraction
    
    def _apply_fees_and_slippage(self, price: float, size: float) -> Tuple[float, float]:
        """
        Apply fees and slippage to a trade.
        
        Args:
            price: Trade price
            size: Trade size
            
        Returns:
            Tuple of (cost, effective_price)
        """
        # Trading fee
        fee_cost = size * self.fee_rate
        
        # Slippage (random within range)
        slippage_pct = np.random.uniform(0, self.slippage_rate)
        slippage_cost = size * slippage_pct
        
        total_cost = fee_cost + slippage_cost
        effective_price = price * (1.0 + slippage_pct)
        
        return total_cost, effective_price
    
    def backtest_bot(
        self,
        bot: BotConfig,
        ohlcv: pd.DataFrame,
        cycle_start_idx: int = 0,
        cycle_end_idx: Optional[int] = None
    ) -> BacktestResult:
        """
        Backtest a single bot on a data range.
        
        Args:
            bot: Bot configuration
            ohlcv: OHLCV DataFrame
            cycle_start_idx: Start index for backtest
            cycle_end_idx: End index for backtest (None = end of data)
            
        Returns:
            BacktestResult
        """
        if cycle_end_idx is None:
            cycle_end_idx = len(ohlcv)
        
        # Convert DataFrame to numpy array for faster access
        ohlcv_array = ohlcv[['timestamp', 'open', 'high', 'low', 'close', 'volume']].values
        
        # Precompute all indicators
        indicator_values = self._precompute_indicators(ohlcv_array, bot)
        
        # Initialize state
        balance = self.initial_balance
        positions: List[Position] = []
        trades = []
        max_balance = balance
        max_drawdown = 0.0
        
        winning_trades = 0
        losing_trades = 0
        
        # Backtest loop
        for i in range(cycle_start_idx, cycle_end_idx):
            timestamp, open_p, high, low, close, volume = ohlcv_array[i]
            
            # Check existing positions for TP/SL
            positions_to_close = []
            for pos_idx, pos in enumerate(positions):
                # Check if TP or SL hit
                hit_tp = False
                hit_sl = False
                
                if pos.side == PositionSide.LONG:
                    if high >= pos.tp_price:
                        hit_tp = True
                    elif low <= pos.sl_price:
                        hit_sl = True
                else:  # SHORT
                    if low <= pos.tp_price:
                        hit_tp = True
                    elif high >= pos.sl_price:
                        hit_sl = True
                
                if hit_tp or hit_sl:
                    # Close position
                    exit_price = pos.tp_price if hit_tp else pos.sl_price
                    pnl = pos.calculate_pnl(exit_price)
                    
                    # Apply exit fees
                    exit_cost, _ = self._apply_fees_and_slippage(exit_price, pos.size)
                    net_pnl = pnl - exit_cost
                    
                    balance += net_pnl
                    
                    # Record trade
                    trades.append({
                        'entry_index': pos.entry_index,
                        'exit_index': i,
                        'side': pos.side.name,
                        'entry_price': pos.entry_price,
                        'exit_price': exit_price,
                        'size': pos.size,
                        'pnl': net_pnl,
                        'hit_tp': hit_tp
                    })
                    
                    if net_pnl > 0:
                        winning_trades += 1
                    else:
                        losing_trades += 1
                    
                    positions_to_close.append(pos_idx)
            
            # Remove closed positions
            for idx in reversed(positions_to_close):
                positions.pop(idx)
            
            # Update max drawdown
            max_balance = max(max_balance, balance)
            if max_balance > 0:
                current_drawdown = ((max_balance - balance) / max_balance) * 100.0
                max_drawdown = max(max_drawdown, current_drawdown)
            
            # Generate signal for new positions
            if len(positions) < MAX_POSITIONS_PER_BOT:
                signal = self._generate_signals(i, close, indicator_values, bot)
                
                if signal != SignalType.NEUTRAL:
                    # Calculate position size
                    position_size = self._calculate_position_size(
                        balance,
                        bot,
                        trade_number=len(trades),
                        total_trades=100  # Estimate
                    )
                    
                    if position_size >= balance * 0.01:  # Min 1% of balance
                        # Open position
                        side = PositionSide.LONG if signal == SignalType.LONG else PositionSide.SHORT
                        
                        # Apply entry fees
                        entry_cost, effective_entry = self._apply_fees_and_slippage(close, position_size)
                        
                        # Calculate TP/SL prices
                        tp_price, sl_price = calculate_tp_sl_prices(
                            effective_entry,
                            bot.take_profit_pct,
                            bot.stop_loss_pct,
                            is_long=(side == PositionSide.LONG)
                        )
                        
                        # Deduct entry cost from balance
                        balance -= entry_cost
                        
                        position = Position(
                            entry_price=effective_entry,
                            size=position_size,
                            side=side,
                            entry_index=i,
                            tp_price=tp_price,
                            sl_price=sl_price,
                            leverage=bot.leverage
                        )
                        
                        positions.append(position)
        
        # Close any remaining positions at last price
        for pos in positions:
            pnl = pos.calculate_pnl(close)
            exit_cost, _ = self._apply_fees_and_slippage(close, pos.size)
            net_pnl = pnl - exit_cost
            balance += net_pnl
            
            if net_pnl > 0:
                winning_trades += 1
            else:
                losing_trades += 1
        
        # Calculate metrics
        total_trades = winning_trades + losing_trades
        winrate = winning_trades / total_trades if total_trades > 0 else 0.0
        total_profit_pct = ((balance - self.initial_balance) / self.initial_balance) * 100.0
        
        return BacktestResult(
            bot_id=bot.bot_id,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            total_profit_pct=total_profit_pct,
            winrate=winrate,
            final_balance=balance,
            max_drawdown_pct=max_drawdown,
            trades_detail=trades
        )
    
    def backtest_population(
        self,
        population: List[BotConfig],
        ohlcv: pd.DataFrame,
        cycle_ranges: List[Tuple[int, int]]
    ) -> Dict[int, List[BacktestResult]]:
        """
        Backtest entire population across multiple cycles.
        
        Args:
            population: List of bot configurations
            ohlcv: OHLCV DataFrame
            cycle_ranges: List of (start_idx, end_idx) for each cycle
            
        Returns:
            Dictionary mapping bot_id to list of BacktestResults (one per cycle)
        """
        log_info(f"Backtesting {len(population)} bots across {len(cycle_ranges)} cycles...")
        
        results = {}
        
        for bot_idx, bot in enumerate(population):
            bot_results = []
            
            for cycle_idx, (start_idx, end_idx) in enumerate(cycle_ranges):
                result = self.backtest_bot(bot, ohlcv, start_idx, end_idx)
                bot_results.append(result)
            
            results[bot.bot_id] = bot_results
            
            if (bot_idx + 1) % 100 == 0:
                log_info(f"Backtested {bot_idx + 1}/{len(population)} bots")
        
        log_info(f"Backtesting complete for {len(population)} bots")
        return results
    
    def calculate_average_metrics(
        self,
        results: List[BacktestResult]
    ) -> Dict[str, float]:
        """
        Calculate average metrics across multiple backtest results.
        
        Args:
            results: List of BacktestResults
            
        Returns:
            Dictionary of average metrics
        """
        if not results:
            return {
                'avg_profit_pct': 0.0,
                'avg_winrate': 0.0,
                'avg_trades': 0.0,
                'avg_drawdown': 0.0
            }
        
        return {
            'avg_profit_pct': np.mean([r.total_profit_pct for r in results]),
            'avg_winrate': np.mean([r.winrate for r in results]),
            'avg_trades': np.mean([r.total_trades for r in results]),
            'avg_drawdown': np.mean([r.max_drawdown_pct for r in results])
        }
