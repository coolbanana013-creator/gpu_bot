"""
GPU Kernel Logic Port - Exact CPU Implementations

This module contains exact CPU implementations of all GPU kernel functions.
Each function is a direct port from backtest_with_precomputed.cl with identical logic.

NO SIMPLIFICATIONS - Full feature parity with GPU kernel.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass


# ============================================================================
# CONSTANTS (from GPU kernel)
# ============================================================================

MAKER_FEE = 0.0002  # 0.02% Kucoin maker
TAKER_FEE = 0.0006  # 0.06% Kucoin taker
BASE_SLIPPAGE = 0.0001  # 0.01% base slippage
FUNDING_RATE_INTERVAL = 480  # 8 hours = 480 minutes at 1m timeframe
BASE_FUNDING_RATE = 0.0001  # 0.01% per 8 hours
MAINTENANCE_MARGIN_RATE = 0.005  # 0.5% for BTC

# Risk strategy constants (15 strategies)
RISK_FIXED_PCT = 0
RISK_FIXED_USD = 1
RISK_KELLY_FULL = 2
RISK_KELLY_HALF = 3
RISK_KELLY_QUARTER = 4
RISK_ATR_MULTIPLIER = 5
RISK_VOLATILITY_PCT = 6
RISK_EQUITY_CURVE = 7
RISK_FIXED_RISK_REWARD = 8
RISK_MARTINGALE = 9
RISK_ANTI_MARTINGALE = 10
RISK_FIXED_RATIO = 11
RISK_PERCENT_VOLATILITY = 12
RISK_WILLIAMS_FIXED = 13
RISK_OPTIMAL_F = 14

# Risk strategy names for display
RISK_STRATEGY_NAMES = {
    RISK_FIXED_PCT: "Fixed Percentage",
    RISK_FIXED_USD: "Fixed USD Amount",
    RISK_KELLY_FULL: "Kelly Criterion (Full)",
    RISK_KELLY_HALF: "Kelly Criterion (Half)",
    RISK_KELLY_QUARTER: "Kelly Criterion (Quarter)",
    RISK_ATR_MULTIPLIER: "ATR Multiplier",
    RISK_VOLATILITY_PCT: "Volatility Percentage",
    RISK_EQUITY_CURVE: "Equity Curve Adjustment",
    RISK_FIXED_RISK_REWARD: "Fixed Risk/Reward",
    RISK_MARTINGALE: "Martingale",
    RISK_ANTI_MARTINGALE: "Anti-Martingale",
    RISK_FIXED_RATIO: "Fixed Ratio (Ryan Jones)",
    RISK_PERCENT_VOLATILITY: "Percent Volatility",
    RISK_WILLIAMS_FIXED: "Williams Fixed Fractional",
    RISK_OPTIMAL_F: "Optimal f (Ralph Vince)"
}


@dataclass
class Position:
    """Position struct matching GPU kernel."""
    entry_price: float
    size: float  # Quantity in contracts
    side: int  # 1 = long, -1 = short
    leverage: int
    tp_price: float
    sl_price: float
    entry_time: float  # Timestamp or bar index
    liquidation_price: float = 0.0
    unrealized_pnl: float = 0.0
    is_active: bool = True


# ============================================================================
# CORE GPU KERNEL FUNCTIONS - EXACT PORTS
# ============================================================================

def calculate_dynamic_slippage(
    position_value: float,
    current_volume: float,
    leverage: int,
    current_price: float,
    current_high: float,
    current_low: float
) -> float:
    """
    Calculate dynamic slippage based on position size, volume, volatility, and leverage.
    
    EXACT PORT from GPU kernel lines 150-187.
    OPTIMIZED VERSION: Minimal memory footprint, no historical lookups.
    Uses current bar data only.
    
    Args:
        position_value: Total position value (with leverage)
        current_volume: Current bar's volume
        leverage: Leverage multiplier
        current_price: Current close price
        current_high: Current bar's high
        current_low: Current bar's low
    
    Returns:
        Dynamic slippage rate (e.g., 0.0001 = 0.01%)
    """
    # Base slippage (ideal conditions)
    slippage = BASE_SLIPPAGE
    
    # 1. Volume impact: position size as % of current volume
    volume_impact = 0.0
    if current_volume > 0.0:
        position_pct = position_value / (current_volume * current_price)
        volume_impact = position_pct * 0.01  # 1% of volume = 0.01% additional slippage
        volume_impact = min(volume_impact, 0.005)  # Cap at 0.5% additional
    
    # 2. Volatility multiplier: use current bar's high-low range
    volatility_multiplier = 1.0
    if current_price > 0.0:
        range_pct = (current_high - current_low) / current_price
        # If range is 2%, volatility is normal (1x)
        # If range is 4%, volatility is high (2x)
        volatility_multiplier = 1.0 + (range_pct / 0.02)
        volatility_multiplier = min(volatility_multiplier, 4.0)  # Cap at 4x
    
    # 3. Leverage multiplier: higher leverage = larger notional = more market impact
    leverage_multiplier = 1.0 + (leverage / 62.5)
    
    # Combine all factors
    total_slippage = (slippage + volume_impact) * volatility_multiplier * leverage_multiplier
    
    # Final bounds: min 0.005% (ideal conditions), max 0.5% (terrible conditions)
    total_slippage = max(0.00005, min(total_slippage, 0.005))
    
    return total_slippage


def calculate_unrealized_pnl(
    position: Position,
    current_price: float
) -> float:
    """
    Calculate unrealized PnL for a position.
    
    EXACT PORT from GPU kernel lines 227-243.
    
    Args:
        position: Position object
        current_price: Current market price
    
    Returns:
        Unrealized PnL (can be negative)
    """
    if not position.is_active:
        return 0.0
    
    if position.side == 1:
        # Long: profit when price rises
        price_diff = current_price - position.entry_price
    else:
        # Short: profit when price falls
        price_diff = position.entry_price - current_price
    
    # Leveraged PnL
    raw_pnl = price_diff * position.size
    return raw_pnl * position.leverage


def calculate_free_margin(
    balance: float,
    positions: List[Position],
    current_price: float
) -> float:
    """
    Calculate free margin (available for new positions).
    
    EXACT PORT from GPU kernel lines 245-269.
    
    Formula: Free Margin = Balance + Unrealized PnL - Used Margin
    
    Args:
        balance: Current account balance
        positions: List of all positions
        current_price: Current market price
    
    Returns:
        Available margin for new positions
    """
    used_margin = 0.0
    unrealized_pnl = 0.0
    
    for pos in positions:
        if pos.is_active:
            # Margin used = entry_price * quantity
            used_margin += pos.entry_price * pos.size
            
            # Add unrealized PnL
            unrealized_pnl += calculate_unrealized_pnl(pos, current_price)
    
    return balance + unrealized_pnl - used_margin


def check_account_liquidation(
    balance: float,
    positions: List[Position],
    current_price: float
) -> bool:
    """
    Check if account should be liquidated (ACCOUNT-LEVEL, not per-position).
    
    EXACT PORT from GPU kernel lines 271-309.
    
    Real liquidation checks total equity against maintenance margin:
    - Equity = Balance + Sum(Unrealized PnL)
    - Used Margin = Sum(entry_price * quantity for all positions)
    - Maintenance Margin = Used Margin * maintenance_rate (0.5% for BTC) * leverage
    - Liquidation occurs when: Equity < Maintenance Margin
    
    Args:
        balance: Current account balance
        positions: List of all positions
        current_price: Current market price
    
    Returns:
        True if account should be liquidated
    """
    total_unrealized_pnl = 0.0
    total_used_margin = 0.0
    max_leverage = 1  # Track maximum leverage used
    
    for pos in positions:
        if pos.is_active:
            # Calculate unrealized PnL
            total_unrealized_pnl += calculate_unrealized_pnl(pos, current_price)
            
            # Calculate used margin
            total_used_margin += pos.entry_price * pos.size
            
            # Track max leverage
            if pos.leverage > max_leverage:
                max_leverage = pos.leverage
    
    # No positions = no liquidation
    if total_used_margin <= 0.0:
        return False
    
    # Calculate equity
    equity = balance + total_unrealized_pnl
    
    # Maintenance margin: 0.5% of used margin for BTC
    maintenance_margin = total_used_margin * MAINTENANCE_MARGIN_RATE * max_leverage
    
    # Liquidation occurs when equity drops below maintenance margin
    return equity < maintenance_margin


def calculate_position_size(
    balance: float,
    price: float,
    risk_strategy: int,
    risk_param: float
) -> float:
    """
    Calculate position size based on risk strategy.
    
    EXACT PORT from GPU kernel lines 311-450.
    
    Each bot uses ONE strategy with ONE parameter.
    
    Args:
        balance: Current account balance
        price: Current market price
        risk_strategy: Strategy index (0-14)
        risk_param: Strategy parameter
    
    Returns:
        Position value (not quantity)
    """
    position_value = 0.0
    
    if risk_strategy == RISK_FIXED_PCT:
        # Fixed percentage of balance (risk_param: 0.01-0.20 = 1-20%)
        position_value = balance * risk_param
        
    elif risk_strategy == RISK_FIXED_USD:
        # Fixed USD amount (risk_param: 10-10000)
        position_value = risk_param
        
    elif risk_strategy == RISK_KELLY_FULL:
        # Full Kelly criterion (risk_param: 0.01-1.0 fraction)
        position_value = balance * risk_param
        
    elif risk_strategy == RISK_KELLY_HALF:
        # Half Kelly (risk_param: 0.01-1.0, applied as half)
        position_value = balance * (risk_param * 0.5)
        
    elif risk_strategy == RISK_KELLY_QUARTER:
        # Quarter Kelly (risk_param: 0.01-1.0, applied as quarter)
        position_value = balance * (risk_param * 0.25)
        
    elif risk_strategy == RISK_ATR_MULTIPLIER:
        # ATR-based sizing (risk_param: 1.0-5.0 multiplier)
        position_value = balance * 0.05 * risk_param
        
    elif risk_strategy == RISK_VOLATILITY_PCT:
        # Volatility-based percentage (risk_param: 0.01-0.20)
        position_value = balance * risk_param
        
    elif risk_strategy == RISK_EQUITY_CURVE:
        # Equity curve adjustment (risk_param: 0.5-2.0 multiplier)
        position_value = balance * 0.05 * risk_param
        
    elif risk_strategy == RISK_FIXED_RISK_REWARD:
        # Fixed risk/reward ratio (risk_param: 0.01-0.10 risk per trade)
        position_value = balance * risk_param
        
    elif risk_strategy == RISK_MARTINGALE:
        # Martingale (risk_param: 1.5-3.0 multiplier after loss) - DANGEROUS
        position_value = balance * 0.05 * risk_param
        
    elif risk_strategy == RISK_ANTI_MARTINGALE:
        # Anti-Martingale (risk_param: 1.2-2.0 multiplier after win)
        position_value = balance * 0.05 * risk_param
        
    elif risk_strategy == RISK_FIXED_RATIO:
        # Fixed Ratio (Ryan Jones) (risk_param: 1000-10000 delta)
        position_value = balance * 0.05
        
    elif risk_strategy == RISK_PERCENT_VOLATILITY:
        # Percent Volatility (risk_param: 0.01-0.20)
        position_value = balance * risk_param
        
    elif risk_strategy == RISK_WILLIAMS_FIXED:
        # Williams Fixed Fractional (risk_param: 0.01-0.10)
        position_value = balance * risk_param
        
    elif risk_strategy == RISK_OPTIMAL_F:
        # Optimal f (Ralph Vince) (risk_param: 0.01-0.30)
        position_value = balance * risk_param
        
    else:
        # Default: 5% of balance
        position_value = balance * 0.05
    
    # Ensure reasonable bounds
    # Min: $10, Max: 20% of balance
    position_value = max(10.0, min(position_value, balance * 0.2))
    
    return position_value


def apply_funding_rates(
    position: Position,
    bars_held: int,
    balance: float
) -> Tuple[float, float]:
    """
    Apply funding rates (perpetual futures charge every 8 hours).
    
    EXACT PORT from GPU kernel lines 1040-1056.
    
    Args:
        position: Position object
        bars_held: Number of bars position has been held
        balance: Current balance (to deduct from)
    
    Returns:
        Tuple of (funding_cost, updated_balance)
    """
    # Check if we crossed a funding rate boundary
    prev_funding_periods = (bars_held - 1) // FUNDING_RATE_INTERVAL
    curr_funding_periods = bars_held // FUNDING_RATE_INTERVAL
    
    if curr_funding_periods > prev_funding_periods:
        # Funding rate payment due
        position_value = position.entry_price * position.size * position.leverage
        funding_cost = position_value * BASE_FUNDING_RATE
        
        # In bull markets (positive funding), longs pay, shorts receive
        # In bear markets (negative funding), shorts pay, longs receive
        # We use neutral rate here
        if position.side == 1:
            # Long position pays funding
            balance -= funding_cost
            return (-funding_cost, balance)
        else:
            # Short position receives funding
            balance += funding_cost
            return (funding_cost, balance)
    
    return (0.0, balance)


def check_signal_reversal(
    position: Position,
    current_signal: float
) -> bool:
    """
    Check if signal reversed (exit trigger).
    
    EXACT PORT from GPU kernel lines 1093-1099.
    
    Args:
        position: Current position
        current_signal: Current signal (-1, 0, or 1)
    
    Returns:
        True if should exit due to signal reversal
    """
    # If long and signal turns bearish, exit
    if position.side == 1 and current_signal < 0:
        return True
    
    # If short and signal turns bullish, exit
    if position.side == -1 and current_signal > 0:
        return True
    
    return False


def open_position_with_margin(
    balance: float,
    price: float,
    direction: int,
    leverage: int,
    tp_multiplier: float,
    sl_multiplier: float,
    risk_strategy: int,
    risk_param: float,
    current_volume: float,
    current_high: float,
    current_low: float,
    existing_positions: List[Position]
) -> Tuple[Optional[Position], float, Dict]:
    """
    Open new position with TRUE MARGIN TRADING.
    
    EXACT PORT from GPU kernel lines 802-814 and surrounding logic.
    
    REALISTIC APPROACH:
    - Calculate position_value from desired exposure (strategy-specific)
    - Margin required = position_value / leverage (what we put up as collateral)
    - Quantity based on MARGIN, not full position value
    - Fees based on full position value (leverage amplifies)
    - PnL will be leveraged automatically through quantity calculation
    
    Args:
        balance: Current account balance
        price: Entry price
        direction: 1 for long, -1 for short
        leverage: Leverage multiplier
        tp_multiplier: Take profit multiplier
        sl_multiplier: Stop loss multiplier
        risk_strategy: Risk strategy index
        risk_param: Risk strategy parameter
        current_volume: Current bar volume
        current_high: Current bar high
        current_low: Current bar low
        existing_positions: List of existing positions
    
    Returns:
        Tuple of (Position object or None, updated_balance, stats_dict)
    """
    stats = {
        'position_value': 0.0,
        'margin_required': 0.0,
        'entry_fee': 0.0,
        'slippage_cost': 0.0,
        'total_cost': 0.0,
        'quantity': 0.0,
        'reason': None
    }
    
    # Calculate desired position value from risk strategy
    desired_position_value = calculate_position_size(
        balance, price, risk_strategy, risk_param
    )
    stats['position_value'] = desired_position_value
    
    # TRUE MARGIN TRADING CALCULATION
    # Margin = what we reserve from balance (collateral)
    margin_required = desired_position_value / leverage
    stats['margin_required'] = margin_required
    
    # DYNAMIC SLIPPAGE based on market conditions
    slippage_rate = calculate_dynamic_slippage(
        desired_position_value,
        current_volume,
        leverage,
        price,
        current_high,
        current_low
    )
    
    # Fees and slippage are based on FULL position value (leverage amplifies costs)
    entry_fee = desired_position_value * TAKER_FEE
    slippage_cost = desired_position_value * slippage_rate
    stats['entry_fee'] = entry_fee
    stats['slippage_cost'] = slippage_cost
    
    # Total cost = margin we reserve + fees we pay upfront
    total_cost = margin_required + entry_fee + slippage_cost
    stats['total_cost'] = total_cost
    
    # Check free margin (balance + unrealized PnL - used margin)
    free_margin = calculate_free_margin(balance, existing_positions, price)
    
    if free_margin < total_cost:
        stats['reason'] = 'insufficient_free_margin'
        return (None, balance, stats)
    
    # Deduct cost from balance
    new_balance = balance - total_cost
    
    # Ensure balance didn't go negative
    if new_balance < 0.0:
        stats['reason'] = 'negative_balance'
        return (None, balance, stats)
    
    # Calculate quantity based on MARGIN (not full position value)
    # This ensures PnL is automatically leveraged
    quantity = margin_required / price
    stats['quantity'] = quantity
    
    # Calculate TP/SL prices based on entry price
    if direction == 1:
        # Long
        tp_price = price * (1.0 + tp_multiplier)
        sl_price = price * (1.0 - sl_multiplier)
        
        # IMPROVED LIQUIDATION PRICE FORMULA WITH MAINTENANCE MARGIN
        liquidation_threshold = (1.0 - MAINTENANCE_MARGIN_RATE) / leverage
        liquidation_price = price * (1.0 - liquidation_threshold)
    else:
        # Short
        tp_price = price * (1.0 - tp_multiplier)
        sl_price = price * (1.0 + sl_multiplier)
        
        # IMPROVED LIQUIDATION PRICE FORMULA FOR SHORT
        liquidation_threshold = (1.0 - MAINTENANCE_MARGIN_RATE) / leverage
        liquidation_price = price * (1.0 + liquidation_threshold)
    
    # Create position
    position = Position(
        entry_price=price,
        size=quantity,
        side=direction,
        leverage=leverage,
        tp_price=tp_price,
        sl_price=sl_price,
        entry_time=0.0,  # Will be set by caller
        liquidation_price=liquidation_price,
        is_active=True
    )
    
    stats['reason'] = 'success'
    return (position, new_balance, stats)


def close_position_with_margin(
    position: Position,
    exit_price: float,
    reason: str,
    current_volume: float,
    current_high: float,
    current_low: float
) -> Tuple[float, Dict]:
    """
    Close position and calculate PnL with TRUE MARGIN TRADING.
    
    EXACT PORT from GPU kernel lines 913-920 and surrounding logic.
    
    REALISTIC APPROACH:
    - Quantity was calculated from margin (margin / entry_price)
    - PnL = price_diff * quantity * leverage (leverage amplification)
    - Return = margin + leveraged_pnl - fees - slippage
    - Liquidation = lose entire margin (but no more)
    
    Args:
        position: Position to close
        exit_price: Exit price
        reason: Reason for closing ('tp', 'sl', 'liquidation', 'reversal')
        current_volume: Current bar volume
        current_high: Current bar high
        current_low: Current bar low
    
    Returns:
        Tuple of (return_amount, stats_dict)
    """
    stats = {
        'margin_reserved': 0.0,
        'raw_pnl': 0.0,
        'leveraged_pnl': 0.0,
        'exit_fee': 0.0,
        'slippage_cost': 0.0,
        'net_pnl': 0.0,
        'total_return': 0.0,
        'reason': reason
    }
    
    if not position.is_active:
        return (0.0, stats)
    
    # Calculate raw price difference
    if position.side == 1:
        # Long: profit when price rises
        price_diff = exit_price - position.entry_price
    else:
        # Short: profit when price falls
        price_diff = position.entry_price - exit_price
    
    # TRUE MARGIN TRADING PnL CALCULATION
    margin_reserved = position.entry_price * position.size
    raw_pnl = price_diff * position.size
    leveraged_pnl = raw_pnl * position.leverage
    
    stats['margin_reserved'] = margin_reserved
    stats['raw_pnl'] = raw_pnl
    stats['leveraged_pnl'] = leveraged_pnl
    
    # Calculate position value for fees (full notional value)
    notional_position_value = position.entry_price * position.size * position.leverage
    
    # TP and SL are limit orders → maker fee
    # Signal reversals are market orders → taker fee
    # Liquidation loses all margin, no exit fee
    if reason == 'liquidation':
        exit_fee = 0.0
    elif reason in ['tp', 'sl']:
        exit_fee = notional_position_value * MAKER_FEE
    else:  # 'reversal' or 'manual'
        exit_fee = notional_position_value * TAKER_FEE
    
    stats['exit_fee'] = exit_fee
    
    # DYNAMIC SLIPPAGE on exit
    slippage_rate = calculate_dynamic_slippage(
        notional_position_value,
        current_volume,
        position.leverage,
        exit_price,
        current_high,
        current_low
    )
    slippage_cost = notional_position_value * slippage_rate
    stats['slippage_cost'] = slippage_cost
    
    # Net PnL after fees
    net_pnl = leveraged_pnl - exit_fee - slippage_cost
    stats['net_pnl'] = net_pnl
    
    # Total return = margin we get back + net PnL
    total_return = margin_reserved + net_pnl
    
    # LIQUIDATION HANDLING
    if reason == 'liquidation':
        # Liquidation = lose entire margin
        total_return = 0.0
    else:
        # Cap maximum loss at margin (can't lose more than we put up)
        if total_return < 0.0:
            total_return = 0.0
    
    stats['total_return'] = total_return
    position.is_active = False
    
    return (total_return, stats)


# ============================================================================
# INDICATOR SIGNAL LOGIC - ALL 50 INDICATORS
# ============================================================================

def get_indicator_signal(
    indicator_index: int,
    indicator_value: float,
    indicator_params: List[float],
    bar_index: int,
    indicator_history: List[float],
    price: float = 0.0
) -> int:
    """
    Get signal from single indicator.
    
    EXACT PORT from GPU kernel lines 540-780.
    
    Returns:
        1 for bullish, -1 for bearish, 0 for neutral
    """
    # Get previous value for momentum indicators
    prev_value = indicator_history[-1] if len(indicator_history) > 0 else indicator_value
    
    # === CATEGORY 1: MOVING AVERAGES (0-11) ===
    if 0 <= indicator_index <= 11:
        # Trend-following: MA rising = bullish, MA falling = bearish
        if indicator_value > prev_value * 1.001:
            return 1
        elif indicator_value < prev_value * 0.999:
            return -1
        return 0
    
    # === CATEGORY 2: MOMENTUM INDICATORS (12-19) ===
    
    # RSI (12-14): overbought/oversold
    if 12 <= indicator_index <= 14:
        if indicator_value < 30.0:
            return 1  # Oversold = buy signal
        elif indicator_value > 70.0:
            return -1  # Overbought = sell signal
        return 0
    
    # Stochastic %K (15)
    if indicator_index == 15:
        if indicator_value < 20.0:
            return 1
        elif indicator_value > 80.0:
            return -1
        return 0
    
    # StochRSI (16)
    if indicator_index == 16:
        if indicator_value < 20.0:
            return 1
        elif indicator_value > 80.0:
            return -1
        return 0
    
    # Momentum (17): rate of change
    if indicator_index == 17:
        if indicator_value > 0.0:
            return 1
        elif indicator_value < 0.0:
            return -1
        return 0
    
    # ROC (18): percentage rate of change
    if indicator_index == 18:
        if indicator_value > 2.0:
            return 1
        elif indicator_value < -2.0:
            return -1
        return 0
    
    # Williams %R (19)
    if indicator_index == 19:
        if indicator_value < -80.0:
            return 1  # Oversold
        elif indicator_value > -20.0:
            return -1  # Overbought
        return 0
    
    # === CATEGORY 3: VOLATILITY INDICATORS (20-25) ===
    
    # ATR (20-21)
    if 20 <= indicator_index <= 21:
        if len(indicator_history) >= 2:
            prev2 = indicator_history[-2] if len(indicator_history) > 1 else prev_value
            if prev_value > prev2:
                return 1  # Volatility trend up
            elif prev_value < prev2:
                return -1  # Volatility trend down
        return 0
    
    # NATR (22)
    if indicator_index == 22:
        if indicator_value > prev_value:
            return 1
        elif indicator_value < prev_value:
            return -1
        return 0
    
    # Bollinger Bands Upper (23)
    if indicator_index == 23:
        # Near upper band = potential breakout
        if len(indicator_history) > 0:
            if indicator_value > prev_value * 1.002:
                return 1  # Expanding bands = bullish
            else:
                return -1  # Near upper in normal = overbought
        return 0
    
    # Bollinger Bands Lower (24)
    if indicator_index == 24:
        # Near lower band = oversold
        if len(indicator_history) > 0:
            if indicator_value < prev_value * 0.998:
                return -1  # Expanding down = bearish
            else:
                return 1  # Near lower = oversold
        return 0
    
    # Keltner Channel (25)
    if indicator_index == 25:
        if indicator_value > prev_value:
            return 1
        elif indicator_value < prev_value:
            return -1
        return 0
    
    # === CATEGORY 4: TREND INDICATORS (26-35) ===
    
    # MACD (26)
    if indicator_index == 26:
        if indicator_value > 0.0 and prev_value <= 0.0:
            return 1  # Bullish crossover
        elif indicator_value < 0.0 and prev_value >= 0.0:
            return -1  # Bearish crossover
        elif indicator_value > 0.0:
            return 1  # Above zero = bullish
        elif indicator_value < 0.0:
            return -1  # Below zero = bearish
        return 0
    
    # ADX (27): trend strength
    if indicator_index == 27:
        if indicator_value > 25.0 and indicator_value > prev_value:
            # Strong trend strengthening - check direction
            # Assume bullish if ADX rising (real impl would check +DI/-DI)
            return 1
        return 0
    
    # Aroon Up (28)
    if indicator_index == 28:
        if indicator_value > 70.0:
            return 1  # Recent high = bullish
        elif indicator_value < 30.0:
            return -1  # No recent high = bearish
        return 0
    
    # CCI (29)
    if indicator_index == 29:
        if indicator_value < -100.0:
            return 1  # Oversold
        elif indicator_value > 100.0:
            return -1  # Overbought
        return 0
    
    # DPO (30)
    if indicator_index == 30:
        if indicator_value > 0.0:
            return 1
        elif indicator_value < 0.0:
            return -1
        return 0
    
    # Parabolic SAR (31)
    if indicator_index == 31:
        if indicator_value < prev_value:
            return 1  # SAR dropping = uptrend
        elif indicator_value > prev_value:
            return -1  # SAR rising = downtrend
        return 0
    
    # SuperTrend (32)
    if indicator_index == 32:
        if indicator_value > prev_value * 1.001:
            return 1  # Rising = bullish
        elif indicator_value < prev_value * 0.999:
            return -1  # Falling = bearish
        return 0
    
    # Trend Strength (33-35): linear regression slope
    if 33 <= indicator_index <= 35:
        if indicator_value > 0.0:
            return 1  # Positive slope = uptrend
        elif indicator_value < 0.0:
            return -1  # Negative slope = downtrend
        return 0
    
    # === CATEGORY 5: VOLUME INDICATORS (36-40) ===
    
    # OBV (36)
    if indicator_index == 36:
        if indicator_value > prev_value:
            return 1  # Volume supporting uptrend
        elif indicator_value < prev_value:
            return -1  # Volume supporting downtrend
        return 0
    
    # VWAP (37)
    if indicator_index == 37:
        if indicator_value > prev_value:
            return 1
        elif indicator_value < prev_value:
            return -1
        return 0
    
    # MFI (38): money flow index
    if indicator_index == 38:
        if indicator_value < 20.0:
            return 1  # Oversold with volume
        elif indicator_value > 80.0:
            return -1  # Overbought with volume
        return 0
    
    # A/D (39): accumulation/distribution
    if indicator_index == 39:
        if indicator_value > prev_value:
            return 1  # Accumulation
        elif indicator_value < prev_value:
            return -1  # Distribution
        return 0
    
    # Volume SMA (40)
    if indicator_index == 40:
        if indicator_value > prev_value * 1.2:
            return 1  # High volume = breakout
        elif indicator_value < prev_value * 0.8:
            return -1  # Low volume = breakdown
        return 0
    
    # === CATEGORY 6: PATTERN INDICATORS (41-45) ===
    
    # Pivot Points (41)
    if indicator_index == 41:
        if indicator_value > prev_value:
            return 1
        elif indicator_value < prev_value:
            return -1
        return 0
    
    # Fractal High (42)
    if indicator_index == 42:
        if indicator_value > 0.0:
            return -1  # Near resistance
        return 0
    
    # Fractal Low (43)
    if indicator_index == 43:
        if indicator_value > 0.0:
            return 1  # Near support
        return 0
    
    # Support/Resistance (44)
    if indicator_index == 44:
        if indicator_value > prev_value:
            return 1  # Breaking resistance = bullish
        elif indicator_value < prev_value:
            return -1  # Breaking support = bearish
        return 0
    
    # Price Channel (45)
    if indicator_index == 45:
        if indicator_value > prev_value:
            return 1
        elif indicator_value < prev_value:
            return -1
        return 0
    
    # === CATEGORY 7: SIMPLE INDICATORS (46-49) ===
    
    # High-Low Range (46)
    if indicator_index == 46:
        if len(indicator_history) > 0:
            if indicator_value > prev_value * 1.5:
                return 1  # Expanding range = breakout
        return 0
    
    # Close Position (47): where close is in bar range
    if indicator_index == 47:
        if indicator_value > 0.7:
            return 1  # Bullish close (near high)
        elif indicator_value < 0.3:
            return -1  # Bearish close (near low)
        return 0
    
    # Price Acceleration (48)
    if indicator_index == 48:
        if indicator_value > 0.0:
            return 1  # Accelerating uptrend
        elif indicator_value < 0.0:
            return -1  # Accelerating downtrend
        return 0
    
    # Volume ROC (49)
    if indicator_index == 49:
        if indicator_value > 10.0:
            return 1  # Strong volume increase = bullish
        elif indicator_value < -10.0:
            return -1  # Strong volume decrease = bearish
        return 0
    
    # Unknown indicator
    return 0


def generate_signal_consensus(
    indicator_values: Dict[int, float],
    indicator_params: Dict[int, List[float]],
    indicator_history: Dict[int, List[float]],
    bar_index: int,
    price: float = 0.0
) -> Tuple[float, Dict]:
    """
    Generate signal from indicators using 75% consensus.
    
    EXACT PORT from GPU kernel lines 540-780.
    
    STRONG: 75% of indicators must agree for a signal.
    
    Args:
        indicator_values: Dict of {indicator_index: current_value}
        indicator_params: Dict of {indicator_index: [param0, param1, param2]}
        indicator_history: Dict of {indicator_index: [prev_values]}
        bar_index: Current bar index
        price: Current price (for context)
    
    Returns:
        Tuple of (signal, breakdown_dict)
        signal: 1.0 (all bullish), -1.0 (all bearish), 0.0 (no consensus)
        breakdown: {'bullish_count', 'bearish_count', 'neutral_count', 'signals'}
    """
    if len(indicator_values) == 0:
        return (0.0, {
            'bullish_count': 0,
            'bearish_count': 0,
            'neutral_count': 0,
            'signals': {}
        })
    
    bullish_count = 0
    bearish_count = 0
    neutral_count = 0
    valid_indicators = 0
    signals = {}
    
    for ind_idx, ind_value in indicator_values.items():
        # Skip invalid indicator values (NaN, Inf, or during warmup)
        if ind_value is None or (isinstance(ind_value, float) and (np.isnan(ind_value) or np.isinf(ind_value))):
            continue  # Skip this indicator
        
        valid_indicators += 1
        
        params = indicator_params.get(ind_idx, [0.0, 0.0, 0.0])
        history = indicator_history.get(ind_idx, [])
        
        signal = get_indicator_signal(
            ind_idx, ind_value, params, bar_index, history, price
        )
        
        signals[ind_idx] = signal
        
        if signal == 1:
            bullish_count += 1
        elif signal == -1:
            bearish_count += 1
        else:
            neutral_count += 1
    
    # Need at least one valid indicator
    if valid_indicators == 0:
        return (0.0, {
            'bullish_count': 0,
            'bearish_count': 0,
            'neutral_count': 0,
            'signals': {}
        })
    
    # Calculate consensus percentage based on VALID indicators only
    bullish_pct = bullish_count / valid_indicators
    bearish_pct = bearish_count / valid_indicators
    
    # 75% consensus required (STRONG: 3 out of 4 indicators must agree)
    if bullish_pct >= 0.75:
        final_signal = 1.0  # 75%+ bullish
    elif bearish_pct >= 0.75:
        final_signal = -1.0  # 75%+ bearish
    else:
        final_signal = 0.0  # No consensus (below 75% threshold)
    
    breakdown = {
        'bullish_count': bullish_count,
        'bearish_count': bearish_count,
        'neutral_count': neutral_count,
        'signals': signals
    }
    
    return (final_signal, breakdown)
