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
        
        # VWAP session tracking (resets daily)
        self.vwap_cumulative_tp_vol = 0.0
        self.vwap_cumulative_vol = 0.0
        self.last_vwap_date = None
        
        # SuperTrend state tracking
        self.supertrend_direction = {}  # {(period, multiplier): direction}
    
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
    
    def _check_and_reset_vwap_session(self, timestamp: float = None):
        """
        Check if new trading session started and reset VWAP if needed.
        Trading sessions typically reset at 00:00 UTC.
        
        Args:
            timestamp: Unix timestamp (optional, uses current time if not provided)
        """
        from datetime import datetime, timezone
        
        if timestamp is None:
            current_date = datetime.now(timezone.utc).date()
        else:
            current_date = datetime.fromtimestamp(timestamp, tz=timezone.utc).date()
        
        # Reset VWAP if new day
        if self.last_vwap_date is None or current_date != self.last_vwap_date:
            self.vwap_cumulative_tp_vol = 0.0
            self.vwap_cumulative_vol = 0.0
            self.last_vwap_date = current_date
    
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
            
            # Momentum (17)
            elif indicator_index == 17:
                period = int(param0)
                return talib.MOM(closes, timeperiod=period)[-1]
            
            # ROC (18)
            elif indicator_index == 18:
                period = int(param0)
                return talib.ROC(closes, timeperiod=period)[-1]
            
            # Williams %R (19)
            elif indicator_index == 19:
                period = int(param0)
                return talib.WILLR(highs, lows, closes, timeperiod=period)[-1]
            
            # ATR (20-21)
            elif indicator_index in [20, 21]:
                period = int(param0)
                return talib.ATR(highs, lows, closes, timeperiod=period)[-1]
            
            # NATR (22)
            elif indicator_index == 22:
                period = int(param0)
                return talib.NATR(highs, lows, closes, timeperiod=period)[-1]
            
            # Bollinger Bands Upper (23)
            elif indicator_index == 23:
                period = int(param0)
                stddev = param1 if param1 > 0 else 2.0
                upper, middle, lower = talib.BBANDS(closes, timeperiod=period, 
                                                    nbdevup=stddev, nbdevdn=stddev)
                return upper[-1]
            
            # Bollinger Bands Lower (24)
            elif indicator_index == 24:
                period = int(param0)
                stddev = param1 if param1 > 0 else 2.0
                upper, middle, lower = talib.BBANDS(closes, timeperiod=period, 
                                                    nbdevup=stddev, nbdevdn=stddev)
                return lower[-1]  # NOT MIDDLE!
            
            # Keltner Channel (25)
            elif indicator_index == 25:
                period = int(param0) if param0 > 0 else 20
                atr_period = int(param1) if param1 > 0 else 10
                multiplier = param2 if param2 > 0 else 2.0
                
                ema = talib.EMA(closes, timeperiod=period)
                atr = talib.ATR(highs, lows, closes, timeperiod=atr_period)
                
                # Keltner = EMA ± ATR * multiplier
                keltner_upper = ema[-1] + (atr[-1] * multiplier)
                return keltner_upper
            
            # MACD (26)
            elif indicator_index == 26:
                fast = int(param0) if param0 > 0 else 12
                slow = int(param1) if param1 > 0 else 26
                signal = int(param2) if param2 > 0 else 9
                macd, macdsignal, macdhist = talib.MACD(closes, fastperiod=fast,
                                                         slowperiod=slow, signalperiod=signal)
                return macd[-1]
            
            # ADX (27)
            elif indicator_index == 27:
                period = int(param0)
                return talib.ADX(highs, lows, closes, timeperiod=period)[-1]
            
            # Aroon Up (28)
            elif indicator_index == 28:
                period = int(param0)
                aroon_down, aroon_up = talib.AROON(highs, lows, timeperiod=period)
                return aroon_up[-1]
            
            # CCI (29)
            elif indicator_index == 29:
                period = int(param0)
                return talib.CCI(highs, lows, closes, timeperiod=period)[-1]
            
            # DPO (30) - Detrended Price Oscillator
            elif indicator_index == 30:
                period = int(param0) if param0 > 0 else 20
                shift = int(period / 2) + 1
                
                # DPO = Close - SMA(shifted back)
                sma = talib.SMA(closes, timeperiod=period)
                if len(sma) >= shift:
                    dpo = closes[-1] - sma[-shift]
                else:
                    dpo = 0.0
                return dpo
            
            # Parabolic SAR (31)
            elif indicator_index == 31:
                acceleration = param0 if param0 > 0 else 0.02
                maximum = param1 if param1 > 0 else 0.2
                return talib.SAR(highs, lows, acceleration=acceleration, maximum=maximum)[-1]
            
            # SuperTrend (32)
            elif indicator_index == 32:
                period = int(param0) if param0 > 0 else 10
                multiplier = param1 if param1 > 0 else 3.0
                
                # SuperTrend = HL_avg ± ATR * multiplier
                atr = talib.ATR(highs, lows, closes, timeperiod=period)
                hl_avg = (highs + lows) / 2
                
                basic_upper = hl_avg[-1] + (multiplier * atr[-1])
                basic_lower = hl_avg[-1] - (multiplier * atr[-1])
                
                # State tracking for trend direction
                key = (period, multiplier)
                if key not in self.supertrend_direction:
                    # Initialize: uptrend if price above HL_avg
                    self.supertrend_direction[key] = 1 if closes[-1] > hl_avg[-1] else -1
                
                # Update direction based on price crossing bands
                if closes[-1] > basic_upper:
                    self.supertrend_direction[key] = 1  # Uptrend
                elif closes[-1] < basic_lower:
                    self.supertrend_direction[key] = -1  # Downtrend
                
                # Return appropriate band based on trend
                return basic_lower if self.supertrend_direction[key] == 1 else basic_upper
            
            # Linear Regression Slope (33-35)
            elif indicator_index in [33, 34, 35]:
                period = int(param0) if param0 > 0 else 20
                slope = talib.LINEARREG_SLOPE(closes, timeperiod=period)[-1]
                return slope
            
            # OBV (36)
            elif indicator_index == 36:
                return talib.OBV(closes, volumes)[-1]
            
            # VWAP (37) - Volume-Weighted Average Price
            elif indicator_index == 37:
                # Session-based VWAP (resets daily)
                self._check_and_reset_vwap_session()
                
                # Update cumulative values
                typical_price_current = (highs[-1] + lows[-1] + closes[-1]) / 3
                self.vwap_cumulative_tp_vol += typical_price_current * volumes[-1]
                self.vwap_cumulative_vol += volumes[-1]
                
                # Calculate VWAP
                if self.vwap_cumulative_vol > 0:
                    vwap = self.vwap_cumulative_tp_vol / self.vwap_cumulative_vol
                else:
                    vwap = closes[-1]
                
                return vwap
            
            # MFI (38) - Money Flow Index
            elif indicator_index == 38:
                period = int(param0) if param0 > 0 else 14
                return talib.MFI(highs, lows, closes, volumes, timeperiod=period)[-1]
            
            # A/D (39) - Accumulation/Distribution
            elif indicator_index == 39:
                return talib.AD(highs, lows, closes, volumes)[-1]
            
            # Volume SMA (40)
            elif indicator_index == 40:
                period = int(param0) if param0 > 0 else 20
                return talib.SMA(volumes, timeperiod=period)[-1]
            
            # Pivot Points (41)
            elif indicator_index == 41:
                # Classic pivot = (High[-2] + Low[-2] + Close[-2]) / 3
                if len(closes) >= 2:
                    pivot = (highs[-2] + lows[-2] + closes[-2]) / 3
                    return pivot
                return closes[-1]
            
            # Fractal High (42)
            elif indicator_index == 42:
                # Fractal high: middle bar higher than 2 bars on each side
                if len(highs) >= 5:
                    if highs[-3] > highs[-5] and highs[-3] > highs[-4] and \
                       highs[-3] > highs[-2] and highs[-3] > highs[-1]:
                        return 1.0  # Fractal detected
                return 0.0
            
            # Fractal Low (43)
            elif indicator_index == 43:
                # Fractal low: middle bar lower than 2 bars on each side
                if len(lows) >= 5:
                    if lows[-3] < lows[-5] and lows[-3] < lows[-4] and \
                       lows[-3] < lows[-2] and lows[-3] < lows[-1]:
                        return 1.0  # Fractal detected
                return 0.0
            
            # Support/Resistance (44)
            elif indicator_index == 44:
                # Dynamic S/R using recent highs/lows
                lookback = int(param0) if param0 > 0 else 20
                if len(highs) >= lookback:
                    resistance = np.max(highs[-lookback:])
                    support = np.min(lows[-lookback:])
                    # Return which level is closer
                    dist_to_resistance = resistance - closes[-1]
                    dist_to_support = closes[-1] - support
                    return resistance if dist_to_resistance < dist_to_support else support
                return closes[-1]
            
            # Price Channel (45)
            elif indicator_index == 45:
                period = int(param0) if param0 > 0 else 20
                if len(highs) >= period:
                    highest = np.max(highs[-period:])
                    lowest = np.min(lows[-period:])
                    # Return midpoint
                    return (highest + lowest) / 2
                return closes[-1]
            
            # High-Low Range (46)
            elif indicator_index == 46:
                return highs[-1] - lows[-1]
            
            # Close Position in Range (47)
            elif indicator_index == 47:
                # (Close - Low) / (High - Low)
                range_size = highs[-1] - lows[-1]
                if range_size > 0:
                    return (closes[-1] - lows[-1]) / range_size
                return 0.5
            
            # Price Acceleration (48)
            elif indicator_index == 48:
                # Second derivative: rate of change of momentum
                if len(closes) >= 3:
                    mom1 = closes[-1] - closes[-2]
                    mom2 = closes[-2] - closes[-3]
                    acceleration = mom1 - mom2
                    return acceleration
                return 0.0
            
            # Volume ROC (49)
            elif indicator_index == 49:
                period = int(param0) if param0 > 0 else 1
                return talib.ROC(volumes, timeperiod=period)[-1]
            
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
