"""
Signal generation rules for technical indicators.
Defines buy/sell conditions for each indicator type.
"""
import numpy as np
from typing import Tuple
from enum import Enum

from .factory import IndicatorType


class SignalType(Enum):
    """Signal types."""
    LONG = 1  # Buy/Long signal
    SHORT = -1  # Sell/Short signal
    NEUTRAL = 0  # No signal


def generate_rsi_signal(rsi_value: float, oversold: float = 30, overbought: float = 70) -> SignalType:
    """
    Generate signal from RSI.
    Long when RSI < oversold, Short when RSI > overbought.
    """
    if np.isnan(rsi_value):
        return SignalType.NEUTRAL
    if rsi_value < oversold:
        return SignalType.LONG
    elif rsi_value > overbought:
        return SignalType.SHORT
    return SignalType.NEUTRAL


def generate_stochastic_signal(stoch_value: float, oversold: float = 20, overbought: float = 80) -> SignalType:
    """
    Generate signal from Stochastic.
    Long when %K < oversold, Short when %K > overbought.
    """
    if np.isnan(stoch_value):
        return SignalType.NEUTRAL
    if stoch_value < oversold:
        return SignalType.LONG
    elif stoch_value > overbought:
        return SignalType.SHORT
    return SignalType.NEUTRAL


def generate_macd_signal(macd_current: float, macd_previous: float, signal_current: float, signal_previous: float) -> SignalType:
    """
    Generate signal from MACD.
    Long on bullish crossover, Short on bearish crossover.
    """
    if np.isnan(macd_current) or np.isnan(signal_current):
        return SignalType.NEUTRAL
    
    # Bullish crossover: MACD crosses above signal
    if macd_previous < signal_previous and macd_current > signal_current:
        return SignalType.LONG
    # Bearish crossover: MACD crosses below signal
    elif macd_previous > signal_previous and macd_current < signal_current:
        return SignalType.SHORT
    
    return SignalType.NEUTRAL


def generate_cci_signal(cci_value: float, oversold: float = -100, overbought: float = 100) -> SignalType:
    """
    Generate signal from CCI.
    Long when CCI < oversold, Short when CCI > overbought.
    """
    if np.isnan(cci_value):
        return SignalType.NEUTRAL
    if cci_value < oversold:
        return SignalType.LONG
    elif cci_value > overbought:
        return SignalType.SHORT
    return SignalType.NEUTRAL


def generate_momentum_signal(mom_value: float) -> SignalType:
    """
    Generate signal from Momentum.
    Long when positive, Short when negative.
    """
    if np.isnan(mom_value):
        return SignalType.NEUTRAL
    if mom_value > 0:
        return SignalType.LONG
    elif mom_value < 0:
        return SignalType.SHORT
    return SignalType.NEUTRAL


def generate_roc_signal(roc_value: float) -> SignalType:
    """
    Generate signal from ROC.
    Long when positive, Short when negative.
    """
    if np.isnan(roc_value):
        return SignalType.NEUTRAL
    if roc_value > 0:
        return SignalType.LONG
    elif roc_value < 0:
        return SignalType.SHORT
    return SignalType.NEUTRAL


def generate_williams_r_signal(willr_value: float, oversold: float = -80, overbought: float = -20) -> SignalType:
    """
    Generate signal from Williams %R.
    Long when < oversold, Short when > overbought.
    """
    if np.isnan(willr_value):
        return SignalType.NEUTRAL
    if willr_value < oversold:
        return SignalType.LONG
    elif willr_value > overbought:
        return SignalType.SHORT
    return SignalType.NEUTRAL


def generate_moving_average_signal(price: float, ma_value: float) -> SignalType:
    """
    Generate signal from Moving Average.
    Long when price > MA, Short when price < MA.
    """
    if np.isnan(ma_value):
        return SignalType.NEUTRAL
    if price > ma_value:
        return SignalType.LONG
    elif price < ma_value:
        return SignalType.SHORT
    return SignalType.NEUTRAL


def generate_adx_signal(adx_value: float, plus_di: float, minus_di: float, threshold: float = 25) -> SignalType:
    """
    Generate signal from ADX.
    Strong trend when ADX > threshold, direction from DI+/DI-.
    """
    if np.isnan(adx_value) or np.isnan(plus_di) or np.isnan(minus_di):
        return SignalType.NEUTRAL
    
    # Only generate signals in strong trends
    if adx_value > threshold:
        if plus_di > minus_di:
            return SignalType.LONG
        elif minus_di > plus_di:
            return SignalType.SHORT
    
    return SignalType.NEUTRAL


def generate_aroon_signal(aroon_up: float, aroon_down: float) -> SignalType:
    """
    Generate signal from Aroon.
    Long when Aroon Up > Aroon Down, Short when Aroon Down > Aroon Up.
    """
    if np.isnan(aroon_up) or np.isnan(aroon_down):
        return SignalType.NEUTRAL
    
    if aroon_up > aroon_down and aroon_up > 70:
        return SignalType.LONG
    elif aroon_down > aroon_up and aroon_down > 70:
        return SignalType.SHORT
    
    return SignalType.NEUTRAL


def generate_bollinger_signal(price: float, upper: float, lower: float, middle: float) -> SignalType:
    """
    Generate signal from Bollinger Bands.
    Long when price touches lower band, Short when price touches upper band.
    """
    if np.isnan(upper) or np.isnan(lower):
        return SignalType.NEUTRAL
    
    if price <= lower:
        return SignalType.LONG
    elif price >= upper:
        return SignalType.SHORT
    
    return SignalType.NEUTRAL


def generate_atr_signal(atr_current: float, atr_average: float) -> SignalType:
    """
    Generate signal from ATR.
    High volatility (ATR rising) might indicate trend continuation.
    """
    if np.isnan(atr_current) or np.isnan(atr_average):
        return SignalType.NEUTRAL
    
    # Rising volatility - maintain trend
    if atr_current > atr_average * 1.2:
        return SignalType.LONG  # Bias towards long in high volatility
    
    return SignalType.NEUTRAL


def generate_obv_signal(obv_current: float, obv_previous: float) -> SignalType:
    """
    Generate signal from OBV.
    Long when OBV rising, Short when OBV falling.
    """
    if np.isnan(obv_current) or np.isnan(obv_previous):
        return SignalType.NEUTRAL
    
    if obv_current > obv_previous:
        return SignalType.LONG
    elif obv_current < obv_previous:
        return SignalType.SHORT
    
    return SignalType.NEUTRAL


def generate_mfi_signal(mfi_value: float, oversold: float = 20, overbought: float = 80) -> SignalType:
    """
    Generate signal from MFI.
    Long when MFI < oversold, Short when MFI > overbought.
    """
    if np.isnan(mfi_value):
        return SignalType.NEUTRAL
    
    if mfi_value < oversold:
        return SignalType.LONG
    elif mfi_value > overbought:
        return SignalType.SHORT
    
    return SignalType.NEUTRAL


def generate_sar_signal(price: float, sar_value: float) -> SignalType:
    """
    Generate signal from Parabolic SAR.
    Long when price > SAR, Short when price < SAR.
    """
    if np.isnan(sar_value):
        return SignalType.NEUTRAL
    
    if price > sar_value:
        return SignalType.LONG
    elif price < sar_value:
        return SignalType.SHORT
    
    return SignalType.NEUTRAL


def generate_ppo_signal(ppo_value: float) -> SignalType:
    """
    Generate signal from PPO.
    Long when positive, Short when negative.
    """
    if np.isnan(ppo_value):
        return SignalType.NEUTRAL
    
    if ppo_value > 0:
        return SignalType.LONG
    elif ppo_value < 0:
        return SignalType.SHORT
    
    return SignalType.NEUTRAL


# Mapping of indicator types to their signal functions
INDICATOR_SIGNAL_GENERATORS = {
    IndicatorType.RSI: generate_rsi_signal,
    IndicatorType.STOCH: generate_stochastic_signal,
    IndicatorType.MACD: generate_macd_signal,
    IndicatorType.CCI: generate_cci_signal,
    IndicatorType.MOM: generate_momentum_signal,
    IndicatorType.ROC: generate_roc_signal,
    IndicatorType.WILLIAMS_R: generate_williams_r_signal,
    IndicatorType.EMA: generate_moving_average_signal,
    IndicatorType.SMA: generate_moving_average_signal,
    IndicatorType.WMA: generate_moving_average_signal,
    IndicatorType.DEMA: generate_moving_average_signal,
    IndicatorType.TEMA: generate_moving_average_signal,
    IndicatorType.ADX: generate_adx_signal,
    IndicatorType.AROON: generate_aroon_signal,
    IndicatorType.BBANDS: generate_bollinger_signal,
    IndicatorType.ATR: generate_atr_signal,
    IndicatorType.OBV: generate_obv_signal,
    IndicatorType.MFI: generate_mfi_signal,
    IndicatorType.SAR: generate_sar_signal,
    IndicatorType.PPO: generate_ppo_signal,
}


def calculate_consensus_signal(signals: list, threshold: float = 0.75) -> SignalType:
    """
    Calculate consensus signal from multiple indicator signals.
    
    Args:
        signals: List of SignalType values
        threshold: Percentage of indicators that must agree (0.0-1.0)
        
    Returns:
        Consensus signal (LONG, SHORT, or NEUTRAL)
    """
    if not signals:
        return SignalType.NEUTRAL
    
    long_count = sum(1 for s in signals if s == SignalType.LONG)
    short_count = sum(1 for s in signals if s == SignalType.SHORT)
    total_count = len(signals)
    
    long_pct = long_count / total_count
    short_pct = short_count / total_count
    
    if long_pct >= threshold:
        return SignalType.LONG
    elif short_pct >= threshold:
        return SignalType.SHORT
    
    return SignalType.NEUTRAL
