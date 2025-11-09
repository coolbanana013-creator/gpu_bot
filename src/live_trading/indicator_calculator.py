"""
Real-Time Indicator Calculator

Calculates indicators in real-time on CPU, replicating GPU kernel logic exactly.
Uses same formulas and parameters as GPU kernels.
"""

import numpy as np
from typing import Dict, List, Tuple
import talib

from ..indicators.factory import IndicatorFactory
from ..utils.validation import log_debug


class RealTimeIndicatorCalculator:
    """
    Calculate indicators in real-time matching GPU kernel logic.
    
    Supports all 50 indicators from GPU kernels.
    """
    
    def __init__(self, lookback_bars: int = 500):
        """
        Initialize calculator.
        
        Args:
            lookback_bars: Number of historical bars to maintain for calculations
        """
        self.lookback_bars = lookback_bars
        self.indicator_factory = IndicatorFactory()
        
        # Get all available indicators (matches GPU kernel order)
        self.all_indicators = self.indicator_factory.get_all_indicator_types()
        
        # Price data buffers (circular buffer)
        self.opens = np.zeros(lookback_bars, dtype=np.float32)
        self.highs = np.zeros(lookback_bars, dtype=np.float32)
        self.lows = np.zeros(lookback_bars, dtype=np.float32)
        self.closes = np.zeros(lookback_bars, dtype=np.float32)
        self.volumes = np.zeros(lookback_bars, dtype=np.float32)
        
        self.current_index = 0
        self.bars_count = 0
    
    def update_price_data(self, open_: float, high: float, low: float, close: float, volume: float):
        """
        Update price data with latest candle.
        
        Args:
            open_: Open price
            high: High price
            low: Low price
            close: Close price
            volume: Volume
        """
        idx = self.current_index % self.lookback_bars
        
        self.opens[idx] = open_
        self.highs[idx] = high
        self.lows[idx] = low
        self.closes[idx] = close
        self.volumes[idx] = volume
        
        self.current_index += 1
        self.bars_count = min(self.bars_count + 1, self.lookback_bars)
    
    def get_ordered_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get price data in chronological order.
        
        Returns:
            Tuple of (opens, highs, lows, closes, volumes) arrays
        """
        if self.bars_count < self.lookback_bars:
            # Haven't filled buffer yet
            return (
                self.opens[:self.bars_count],
                self.highs[:self.bars_count],
                self.lows[:self.bars_count],
                self.closes[:self.bars_count],
                self.volumes[:self.bars_count]
            )
        else:
            # Buffer full, need to reorder
            idx = self.current_index % self.lookback_bars
            return (
                np.roll(self.opens, -idx),
                np.roll(self.highs, -idx),
                np.roll(self.lows, -idx),
                np.roll(self.closes, -idx),
                np.roll(self.volumes, -idx)
            )
    
    def calculate_indicator(self, indicator_index: int, param0: float, param1: float, param2: float) -> float:
        """
        Calculate single indicator value (matches GPU kernel logic).
        
        Args:
            indicator_index: Index 0-49 (matches GPU kernel indicator order)
            param0, param1, param2: Indicator parameters
        
        Returns:
            Indicator value
        """
        if self.bars_count < 20:  # Need minimum data
            return 0.0
        
        opens, highs, lows, closes, volumes = self.get_ordered_data()
        
        try:
            # Get indicator name
            indicator_name = self.all_indicators[indicator_index]
            
            # Calculate based on indicator type (matches GPU kernel order)
            # Moving Averages (0-11)
            if 0 <= indicator_index <= 9:  # SMA, EMA variants
                period = int(param0)
                if indicator_index in [0, 1, 2]:  # SMA
                    return talib.SMA(closes, timeperiod=period)[-1]
                elif indicator_index in [3, 4, 5]:  # EMA
                    return talib.EMA(closes, timeperiod=period)[-1]
                elif indicator_index in [6, 7, 8]:  # WMA
                    return talib.WMA(closes, timeperiod=period)[-1]
                elif indicator_index == 9:  # DEMA
                    return talib.DEMA(closes, timeperiod=period)[-1]
            
            elif indicator_index in [10, 11]:  # TEMA, KAMA
                period = int(param0)
                if indicator_index == 10:
                    return talib.TEMA(closes, timeperiod=period)[-1]
                else:
                    return talib.KAMA(closes, timeperiod=period)[-1]
            
            # RSI (12-14)
            elif 12 <= indicator_index <= 14:
                period = int(param0)
                return talib.RSI(closes, timeperiod=period)[-1]
            
            # Stochastic (15-16)
            elif indicator_index == 15:
                k_period = int(param0)
                slowk, slowd = talib.STOCH(highs, lows, closes, 
                                           fastk_period=k_period,
                                           slowk_period=3, slowd_period=3)
                return slowk[-1]
            elif indicator_index == 16:
                k_period = int(param0)
                slowk, slowd = talib.STOCH(highs, lows, closes,
                                           fastk_period=k_period,
                                           slowk_period=3, slowd_period=3)
                return slowd[-1]
            
            # Momentum (17-19)
            elif indicator_index in [17, 18, 19]:
                period = int(param0)
                return talib.MOM(closes, timeperiod=period)[-1]
            
            # ROC (20-22)
            elif indicator_index in [20, 21, 22]:
                period = int(param0)
                return talib.ROC(closes, timeperiod=period)[-1]
            
            # Bollinger Bands (23-25)
            elif indicator_index in [23, 24, 25]:
                period = int(param0)
                stddev = param1 if param1 > 0 else 2.0
                upper, middle, lower = talib.BBANDS(closes, timeperiod=period, 
                                                    nbdevup=stddev, nbdevdn=stddev)
                if indicator_index == 23:
                    return upper[-1]
                elif indicator_index == 24:
                    return middle[-1]
                else:
                    return lower[-1]
            
            # MACD (26)
            elif indicator_index == 26:
                fast = int(param0) if param0 > 0 else 12
                slow = int(param1) if param1 > 0 else 26
                signal = int(param2) if param2 > 0 else 9
                macd, macdsignal, macdhist = talib.MACD(closes, fastperiod=fast,
                                                         slowperiod=slow, signalperiod=signal)
                return macd[-1]
            
            # ADX (27-28)
            elif indicator_index in [27, 28]:
                period = int(param0)
                return talib.ADX(highs, lows, closes, timeperiod=period)[-1]
            
            # CCI (29)
            elif indicator_index == 29:
                period = int(param0)
                return talib.CCI(highs, lows, closes, timeperiod=period)[-1]
            
            # Williams %R (30)
            elif indicator_index == 30:
                period = int(param0)
                return talib.WILLR(highs, lows, closes, timeperiod=period)[-1]
            
            # ATR (31-33)
            elif indicator_index in [31, 32, 33]:
                period = int(param0)
                return talib.ATR(highs, lows, closes, timeperiod=period)[-1]
            
            # SAR (34-35)
            elif indicator_index in [34, 35]:
                return talib.SAR(highs, lows, acceleration=0.02, maximum=0.2)[-1]
            
            # Volume indicators (36-40)
            elif indicator_index == 36:  # OBV
                return talib.OBV(closes, volumes)[-1]
            elif indicator_index == 37:  # AD
                return talib.AD(highs, lows, closes, volumes)[-1]
            elif indicator_index == 38:  # ADOSC
                return talib.ADOSC(highs, lows, closes, volumes, fastperiod=3, slowperiod=10)[-1]
            elif indicator_index in [39, 40]:  # Volume MA
                period = int(param0)
                return talib.SMA(volumes, timeperiod=period)[-1]
            
            # Price patterns (41-49)
            elif indicator_index == 41:  # Typical Price
                return (highs[-1] + lows[-1] + closes[-1]) / 3.0
            elif indicator_index == 42:  # Median Price
                return (highs[-1] + lows[-1]) / 2.0
            elif indicator_index == 43:  # Weighted Close
                return (highs[-1] + lows[-1] + closes[-1] * 2) / 4.0
            elif indicator_index == 44:  # Price change
                return closes[-1] - closes[-2] if len(closes) >= 2 else 0.0
            elif indicator_index == 45:  # Price change %
                return ((closes[-1] - closes[-2]) / closes[-2]) * 100 if len(closes) >= 2 else 0.0
            elif indicator_index == 46:  # High-Low range
                return highs[-1] - lows[-1]
            elif indicator_index == 47:  # High-Low %
                return ((highs[-1] - lows[-1]) / lows[-1]) * 100
            elif indicator_index == 48:  # Close vs Open
                return closes[-1] - opens[-1]
            elif indicator_index == 49:  # Volume change
                return volumes[-1] - volumes[-2] if len(volumes) >= 2 else 0.0
            
            else:
                return 0.0
                
        except Exception as e:
            log_debug(f"Indicator calculation error (index {indicator_index}): {e}")
            return 0.0
    
    def calculate_all_bot_indicators(
        self,
        indicator_indices: List[int],
        indicator_params: np.ndarray
    ) -> Dict[int, float]:
        """
        Calculate all indicators for a bot.
        
        Args:
            indicator_indices: List of indicator indices to calculate
            indicator_params: Array of shape (num_indicators, 3) with parameters
        
        Returns:
            Dict mapping indicator_index -> value
        """
        results = {}
        
        for i, ind_idx in enumerate(indicator_indices):
            param0 = indicator_params[i][0]
            param1 = indicator_params[i][1]
            param2 = indicator_params[i][2]
            
            value = self.calculate_indicator(ind_idx, param0, param1, param2)
            results[ind_idx] = value
        
        return results
