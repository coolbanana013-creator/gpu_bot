"""
Take Profit and Stop Loss generation with leverage adjustments.
Validates against liquidation prices and fees.
"""
import numpy as np
from typing import Tuple, Optional

from ..utils.validation import validate_float, validate_int, log_debug
from ..utils.config import (
    MIN_TAKE_PROFIT_PCT, MAX_TAKE_PROFIT_PCT,
    MIN_STOP_LOSS_PCT, DEFAULT_FEE_RATE, SLIPPAGE_RATE,
    MAINTENANCE_MARGIN_RATE, LIQUIDATION_BUFFER
)


def calculate_liquidation_price(
    entry_price: float,
    leverage: int,
    is_long: bool,
    maintenance_margin_rate: float = MAINTENANCE_MARGIN_RATE
) -> float:
    """
    Calculate liquidation price for a leveraged position.
    
    For Kucoin Futures:
    Long liquidation = entry * (1 - (1/leverage - maintenance_margin))
    Short liquidation = entry * (1 + (1/leverage - maintenance_margin))
    
    Args:
        entry_price: Position entry price
        leverage: Leverage multiplier
        is_long: True for long position, False for short
        maintenance_margin_rate: Maintenance margin requirement
        
    Returns:
        Liquidation price
    """
    initial_margin_rate = 1.0 / leverage
    margin_diff = initial_margin_rate - maintenance_margin_rate
    
    if is_long:
        liq_price = entry_price * (1.0 - margin_diff)
    else:
        liq_price = entry_price * (1.0 + margin_diff)
    
    return liq_price


def calculate_effective_tp_sl(
    tp_pct: float,
    sl_pct: float,
    leverage: int,
    fee_rate: float = DEFAULT_FEE_RATE,
    slippage_rate: float = SLIPPAGE_RATE
) -> Tuple[float, float]:
    """
    Calculate effective TP/SL after accounting for leverage, fees, and slippage.
    
    Effective TP = (TP% / leverage) - fees*2 - slippage
    Effective SL = (SL% / leverage)
    
    Args:
        tp_pct: Take profit percentage (1-25%)
        sl_pct: Stop loss percentage (0.5% to TP/2)
        leverage: Leverage multiplier
        fee_rate: Trading fee rate
        slippage_rate: Slippage estimate
        
    Returns:
        Tuple of (effective_tp_pct, effective_sl_pct)
    """
    # Account for leverage
    leveraged_tp = tp_pct / leverage
    leveraged_sl = sl_pct / leverage
    
    # Subtract costs from TP (entry + exit fees, plus slippage)
    total_cost_pct = (fee_rate * 2 + slippage_rate) * 100
    effective_tp = leveraged_tp - total_cost_pct
    
    # SL is just leveraged (costs already hurt us)
    effective_sl = leveraged_sl
    
    return effective_tp, effective_sl


def generate_tp_sl(
    leverage: int,
    entry_price: float = 100.0,
    is_long: bool = True,
    fee_rate: float = DEFAULT_FEE_RATE,
    slippage_rate: float = SLIPPAGE_RATE,
    maintenance_margin_rate: float = MAINTENANCE_MARGIN_RATE
) -> Tuple[float, float, float, float]:
    """
    Generate valid TP and SL percentages with leverage adjustments.
    
    Constraints:
    - TP: 1-25% (before leverage)
    - SL: 0.5% to TP/2 (before leverage)
    - Effective TP > 0 after fees
    - SL must not trigger liquidation
    
    Args:
        leverage: Leverage multiplier (1-25)
        entry_price: Entry price for liquidation calculation
        is_long: True for long, False for short
        fee_rate: Trading fee rate
        slippage_rate: Slippage rate
        maintenance_margin_rate: Maintenance margin requirement
        
    Returns:
        Tuple of (tp_pct, sl_pct, effective_tp_pct, effective_sl_pct)
        
    Raises:
        ValueError: If parameters invalid
    """
    leverage = validate_int(leverage, "leverage", min_val=1, max_val=25)
    entry_price = validate_float(entry_price, "entry_price", strict_positive=True)
    
    # Calculate liquidation price
    liq_price = calculate_liquidation_price(entry_price, leverage, is_long, maintenance_margin_rate)
    
    # Calculate max SL before liquidation (with buffer)
    if is_long:
        max_sl_price = liq_price + (entry_price - liq_price) * (1.0 - LIQUIDATION_BUFFER)
        max_sl_pct = ((entry_price - max_sl_price) / entry_price) * 100.0
    else:
        max_sl_price = liq_price - (liq_price - entry_price) * (1.0 - LIQUIDATION_BUFFER)
        max_sl_pct = ((max_sl_price - entry_price) / entry_price) * 100.0
    
    # Ensure max_sl_pct is reasonable
    max_sl_pct = max(MIN_STOP_LOSS_PCT, min(max_sl_pct, 50.0))
    
    # Generate TP (1-25%)
    tp_pct = np.random.uniform(MIN_TAKE_PROFIT_PCT, MAX_TAKE_PROFIT_PCT)
    
    # Generate SL (0.5% to min(TP/2, max_sl_pct))
    max_sl_for_tp = tp_pct / 2.0
    actual_max_sl = min(max_sl_for_tp, max_sl_pct)
    
    sl_pct = np.random.uniform(MIN_STOP_LOSS_PCT, actual_max_sl)
    
    # Calculate effective TP/SL
    effective_tp, effective_sl = calculate_effective_tp_sl(
        tp_pct, sl_pct, leverage, fee_rate, slippage_rate
    )
    
    # Validate effective TP is positive
    min_effective_tp = 0.1  # At least 0.1% profit after all costs
    
    if effective_tp < min_effective_tp:
        # Adjust TP upward to ensure profitability
        total_cost_pct = (fee_rate * 2 + slippage_rate) * 100
        min_tp_needed = (min_effective_tp + total_cost_pct) * leverage
        tp_pct = max(tp_pct, min_tp_needed)
        tp_pct = min(tp_pct, MAX_TAKE_PROFIT_PCT)
        
        # Recalculate effective values
        effective_tp, effective_sl = calculate_effective_tp_sl(
            tp_pct, sl_pct, leverage, fee_rate, slippage_rate
        )
    
    log_debug(
        f"Generated TP/SL for {leverage}x leverage: "
        f"TP={tp_pct:.2f}% (eff: {effective_tp:.2f}%), "
        f"SL={sl_pct:.2f}% (eff: {effective_sl:.2f}%)"
    )
    
    return float(tp_pct), float(sl_pct), float(effective_tp), float(effective_sl)


def calculate_tp_sl_prices(
    entry_price: float,
    tp_pct: float,
    sl_pct: float,
    is_long: bool
) -> Tuple[float, float]:
    """
    Calculate actual TP and SL prices from percentages.
    
    Args:
        entry_price: Entry price
        tp_pct: Take profit percentage
        sl_pct: Stop loss percentage
        is_long: True for long, False for short
        
    Returns:
        Tuple of (tp_price, sl_price)
    """
    if is_long:
        tp_price = entry_price * (1.0 + tp_pct / 100.0)
        sl_price = entry_price * (1.0 - sl_pct / 100.0)
    else:
        tp_price = entry_price * (1.0 - tp_pct / 100.0)
        sl_price = entry_price * (1.0 + sl_pct / 100.0)
    
    return float(tp_price), float(sl_price)


def validate_tp_sl(
    entry_price: float,
    tp_price: float,
    sl_price: float,
    leverage: int,
    is_long: bool,
    maintenance_margin_rate: float = MAINTENANCE_MARGIN_RATE
) -> bool:
    """
    Validate that TP/SL prices are valid and won't cause immediate liquidation.
    
    Args:
        entry_price: Entry price
        tp_price: Take profit price
        sl_price: Stop loss price
        leverage: Leverage multiplier
        is_long: True for long, False for short
        maintenance_margin_rate: Maintenance margin requirement
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Calculate liquidation price
        liq_price = calculate_liquidation_price(
            entry_price, leverage, is_long, maintenance_margin_rate
        )
        
        if is_long:
            # For long: TP > entry > SL > liquidation
            if not (tp_price > entry_price):
                log_debug(f"Invalid long TP: {tp_price} <= {entry_price}")
                return False
            if not (entry_price > sl_price):
                log_debug(f"Invalid long SL: {entry_price} <= {sl_price}")
                return False
            if not (sl_price > liq_price):
                log_debug(f"SL too close to liquidation: {sl_price} <= {liq_price}")
                return False
        else:
            # For short: TP < entry < SL < liquidation
            if not (tp_price < entry_price):
                log_debug(f"Invalid short TP: {tp_price} >= {entry_price}")
                return False
            if not (entry_price < sl_price):
                log_debug(f"Invalid short SL: {entry_price} >= {sl_price}")
                return False
            if not (sl_price < liq_price):
                log_debug(f"SL too close to liquidation: {sl_price} >= {liq_price}")
                return False
        
        return True
        
    except Exception as e:
        log_debug(f"TP/SL validation error: {e}")
        return False


def calculate_risk_reward_ratio(tp_pct: float, sl_pct: float) -> float:
    """
    Calculate risk/reward ratio.
    
    Args:
        tp_pct: Take profit percentage
        sl_pct: Stop loss percentage
        
    Returns:
        Risk/reward ratio (TP / SL)
    """
    if sl_pct == 0:
        return 0.0
    return tp_pct / sl_pct
