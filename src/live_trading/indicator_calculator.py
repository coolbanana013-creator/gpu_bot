"""
Real-Time Indicator Calculator

Calculates indicators in real-time on CPU, replicating GPU kernel logic exactly.
Uses same formulas and parameters as GPU kernels.
"""

import numpy as np
from typing import Dict, List, Tuple
import talib

from ..indicators.factory import IndicatorFactory
from ..indicators.gpu_default_params import get_gpu_default_params
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
        # Use float32 for VWAP cumulative to keep parity with GPU's float32 accumulation
        self.vwap_cumulative_tp_vol = np.float32(0.0)
        self.vwap_cumulative_vol = np.float32(0.0)
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
                self.opens[:self.bars_count].astype(np.float64),
                self.highs[:self.bars_count].astype(np.float64),
                self.lows[:self.bars_count].astype(np.float64),
                self.closes[:self.bars_count].astype(np.float64),
                self.volumes[:self.bars_count].astype(np.float64)
            )
        else:
            # Buffer full, need to reorder
            idx = self.current_index % self.lookback_bars
            return (
                np.roll(self.opens, -idx).astype(np.float64),
                np.roll(self.highs, -idx).astype(np.float64),
                np.roll(self.lows, -idx).astype(np.float64),
                np.roll(self.closes, -idx).astype(np.float64),
                np.roll(self.volumes, -idx).astype(np.float64)
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
            # Reset using float32
            self.vwap_cumulative_tp_vol = np.float32(0.0)
            self.vwap_cumulative_vol = np.float32(0.0)
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
        # NOTE: Removed an arbitrary early-exit warmup threshold (20 bars).
        # The GPU precompute kernel computes many indicators from the first bar
        # (or has its own per-indicator warmup semantics). Rely on each
        # per-indicator helper to decide whether the output is valid and
        # return the computed value (or 0.0/NaN) as the kernel would.
        # This keeps parity for indicators that are defined early (e.g., OBV, AD).
        
        opens, highs, lows, closes, volumes = self.get_ordered_data()
        # Keep kernels working arrays in float32 to match GPU calculations
        opens32 = opens.astype(np.float32)
        highs32 = highs.astype(np.float32)
        lows32 = lows.astype(np.float32)
        closes32 = closes.astype(np.float32)
        volumes32 = volumes.astype(np.float32)
        
        try:
            # Get indicator name
            indicator_name = self.all_indicators[indicator_index]

            # Use centralized default GPU params mapping
            def _default_params(idx: int):
                p0, p1, p2 = get_gpu_default_params(idx)
                return p0, p1, p2
            
            # Calculate based on indicator type (matches GPU kernel order)
            # Helper: Kernel-like SMA/EMA implementations to replicate GPU behavior precisely
            def _kernel_sma(arr, period):
                out = np.zeros(len(arr), dtype=np.float32)
                if period <= 0:
                    return out
                for i in range(len(arr)):
                    if i < period - 1:
                        out[i] = np.float32(0.0)
                    else:
                        # Accumulate with float32 semantics to match GPU
                        s = np.float32(0.0)
                        # Sum in the same order as GPU (most recent first) to avoid float32 associativity diff
                        for j in range(i, i - period, -1):
                            s = np.float32(s + np.float32(arr[j]))
                        out[i] = np.float32(s / np.float32(period))
                return out

            def _kernel_ema(arr, period):
                out = np.zeros(len(arr), dtype=np.float32)
                if period <= 0:
                    return out
                prev_ema = np.float32(0.0)
                k = np.float32(2.0) / np.float32(period + 1)
                for i in range(len(arr)):
                    if i < period - 1:
                        out[i] = np.float32(0.0)
                        continue
                    if i == period - 1:
                        # First ema is SMA computed with float32 accumulation
                        s = np.float32(0.0)
                        # Sum in same order as GPU compute_sma_helper (most recent first)
                        for j in range(i, i - period, -1):
                            s = np.float32(s + np.float32(arr[j]))
                        prev_ema = np.float32(s / np.float32(period))
                        out[i] = prev_ema
                        continue
                    prev_ema = np.float32((np.float32(arr[i]) - prev_ema) * k + prev_ema)
                    out[i] = prev_ema
                return out

            def _kernel_atr(highs, lows, closes, period):
                n = len(closes)
                out = np.zeros(n, dtype=np.float32)
                if period <= 0 or n == 0:
                    return out
                # Compute TR
                tr = np.zeros(n, dtype=np.float32)
                for i in range(n):
                    if i == 0:
                        tr[i] = highs[i] - lows[i]
                    else:
                        hl = highs[i] - lows[i]
                        hc = abs(highs[i] - closes[i - 1])
                        lc = abs(lows[i] - closes[i - 1])
                        tr[i] = max(hl, max(hc, lc))

                # RMA smoothing like GPU kernel
                if n >= period:
                    s = np.float32(0.0)
                    for j in range(period):
                        s = np.float32(s + tr[j])
                    out[period - 1] = np.float32(s / np.float32(period))
                    for i in range(period, n):
                        out[i] = np.float32((out[i - 1] * np.float32(period - 1) + tr[i]) / np.float32(period))
                else:
                    for i in range(n):
                        out[i] = np.float32(0.0)
                return out

            def _kernel_macd(closes, fast, slow, signal_period):
                n = len(closes)
                macd = np.zeros(n, dtype=np.float32)
                if n == 0:
                    return macd
                # compute fast and slow EMAs using existing helper
                fast_arr = _kernel_ema(closes, fast)
                slow_arr = _kernel_ema(closes, slow)
                signal_ema = np.float32(0.0)
                for i in range(n):
                    if i < slow:
                        macd[i] = np.float32(0.0)
                        continue
                    macd_line = np.float32(fast_arr[i] - slow_arr[i])
                    if i >= slow - 1:
                        if i == slow - 1:
                            signal_ema = macd_line
                        else:
                            k = np.float32(2.0) / np.float32(signal_period + 1)
                            signal_ema = np.float32((macd_line - signal_ema) * k + signal_ema)
                    macd[i] = macd_line
                return macd

            def _kernel_rsi(arr, period):
                # Re-implement kernel-style RSI with float32 accumulation to match GPU precisely
                n = len(arr)
                out = np.full(n, np.float32(50.0), dtype=np.float32)
                if period <= 0 or n == 0:
                    return out
                # Initial average gain/loss using forward sum as in the GPU kernel
                if n < period + 1:
                    return out
                avg_gain = np.float32(0.0)
                avg_loss = np.float32(0.0)
                for i in range(1, period + 1):
                    change = np.float32(arr[i] - arr[i - 1])
                    if change > np.float32(0.0):
                        avg_gain = np.float32(avg_gain + change)
                    else:
                        avg_loss = np.float32(avg_loss + np.abs(change))
                avg_gain = np.float32(avg_gain / np.float32(period))
                avg_loss = np.float32(avg_loss / np.float32(period))

                for i in range(n):
                    if i < period:
                        out[i] = np.float32(50.0)
                        continue
                    change = np.float32(arr[i] - arr[i - 1])
                    gain = np.float32(change if change > 0 else 0.0)
                    loss = np.float32(-change if change < 0 else 0.0)
                    # Wilder's smoothing
                    avg_gain = np.float32((avg_gain * np.float32(period - 1) + gain) / np.float32(period))
                    avg_loss = np.float32((avg_loss * np.float32(period - 1) + loss) / np.float32(period))
                    if avg_loss < np.float32(1e-10):
                        out[i] = np.float32(100.0)
                    else:
                        rs = np.float32(avg_gain / avg_loss)
                        out[i] = np.float32(100.0 - (100.0 / (np.float32(1.0) + rs)))
                return out

            def _kernel_bollinger(arr, period, stddev):
                # Return two arrays: upper, lower
                n = len(arr)
                upper = np.zeros(n, dtype=np.float32)
                lower = np.zeros(n, dtype=np.float32)
                if period <= 0:
                    return upper, lower
                for i in range(n):
                    # Kernel initializes pre-warmup bars to 0.0 for parity and
                    # writes computed bands only once enough bars are available
                    if i < period - 1:
                        upper[i] = np.float32(arr[i])
                        lower[i] = np.float32(arr[i])
                        continue
                    s = np.float32(0.0)
                    for j in range(i - period + 1, i + 1):
                        s = np.float32(s + np.float32(arr[j]))
                    mean = np.float32(s / np.float32(period))
                    # Compute variance with population denominator = period (GPU kernel uses / period)
                    var = np.float32(0.0)
                    for j in range(i - period + 1, i + 1):
                        d = np.float32(arr[j]) - mean
                        var = np.float32(var + d * d)
                    var = np.float32(var / np.float32(period))
                    std = np.float32(np.sqrt(var))
                    upper[i] = np.float32(mean + stddev * std)
                    lower[i] = np.float32(mean - stddev * std)
                return upper, lower

            def _kernel_psar(op, hi, lo, cl, period_af_start=0.02, af_max=0.2):
                n = len(cl)
                out = np.zeros(n, dtype=np.float32)
                if n == 0:
                    return out
                sar = lo[0]
                ep = hi[0]
                af = np.float32(period_af_start)
                is_long = True
                out[0] = np.float32(sar)
                for i in range(1, n):
                    sar = np.float32(sar + af * (ep - sar))
                    if is_long:
                        if lo[i] < sar:
                            is_long = False
                            sar = ep
                            ep = lo[i]
                            af = np.float32(period_af_start)
                        else:
                            if hi[i] > ep:
                                ep = hi[i]
                                af = np.float32(min(float(af + period_af_start), af_max))
                    else:
                        if hi[i] > sar:
                            is_long = True
                            sar = ep
                            ep = hi[i]
                            af = np.float32(period_af_start)
                        else:
                            if lo[i] < ep:
                                ep = lo[i]
                                af = np.float32(min(float(af + period_af_start), af_max))
                    out[i] = sar
                return out

            def _kernel_supertrend(highs, lows, closes, super_period, multiplier, atr_period=None):
                n = len(closes)
                out = np.zeros(n, dtype=np.float32)
                if n == 0:
                    return out
                # Compute ATR (if not provided) as per kernel (RMA-style implemented elsewhere).
                # Use provided atr_period if present; otherwise use default 14 (GPU ATR default).
                if atr_period is None:
                    atr_period = int(get_gpu_default_params(20)[0])
                tr = np.zeros(n, dtype=np.float32)
                for i in range(n):
                    if i == 0:
                        tr[i] = highs[i] - lows[i]
                    else:
                        hl = highs[i] - lows[i]
                        hc = abs(highs[i] - closes[i - 1])
                        lc = abs(lows[i] - closes[i - 1])
                        tr[i] = max(hl, max(hc, lc))
                # ATR: smoothing like kernel using RMA (Wilder), initial average over period
                atr = np.zeros(n, dtype=np.float32)
                if atr_period <= 0:
                    return out
                if n >= atr_period:
                    s = np.float32(0.0)
                    for j in range(0, atr_period):
                        s = np.float32(s + tr[j])
                    atr[atr_period - 1] = np.float32(s / np.float32(atr_period))
                    for i in range(atr_period, n):
                        atr[i] = np.float32((atr[i - 1] * np.float32(atr_period - 1) + tr[i]) / np.float32(atr_period))
                # Supertrend computation
                # GPU kernel uses 'period' (super_period) for warmup check, not atr_period
                trend = 1
                for i in range(n):
                    if i < super_period - 1:
                        out[i] = closes[i]
                        continue
                    hl_avg = np.float32((highs[i] + lows[i]) / 2.0)
                    atrv = atr[i]
                    upper = np.float32(hl_avg + multiplier * atrv)
                    lower = np.float32(hl_avg - multiplier * atrv)
                    if trend == 1:
                        if closes[i] < lower:
                            trend = -1
                            out[i] = upper
                        else:
                            out[i] = lower
                    else:
                        if closes[i] > upper:
                            trend = 1
                            out[i] = lower
                        else:
                            out[i] = upper
                return out

            # Moving Averages (0-11)
            if 0 <= indicator_index <= 5:  # SMA indices 0-5
                period = int(param0) if param0 > 0 else int(_default_params(indicator_index)[0])
                # All indices 0-5 are SMA in GPU kernel
                return float(_kernel_sma(closes, period)[-1])

            elif 6 <= indicator_index <= 11:  # EMA indices 6-11
                period = int(param0) if param0 > 0 else int(_default_params(indicator_index)[0])
                return float(_kernel_ema(closes, period)[-1])
            elif indicator_index in [10, 11]:  # TEMA, KAMA (original code used 10,11 incorrectly earlier)
                period = int(param0) if param0 > 0 else int(_default_params(indicator_index)[0])
                if indicator_index == 10:
                    return talib.TEMA(closes, timeperiod=period)[-1]
                else:
                    return talib.KAMA(closes, timeperiod=period)[-1]
            
            elif indicator_index in [10, 11]:  # TEMA, KAMA
                period = int(param0) if param0 > 0 else int(_default_params(indicator_index)[0])
                if indicator_index == 10:
                    return talib.TEMA(closes, timeperiod=period)[-1]
                else:
                    return talib.KAMA(closes, timeperiod=period)[-1]
            
            # RSI (12-14)
            elif 12 <= indicator_index <= 14:
                period = int(param0) if param0 > 0 else int(_default_params(indicator_index)[0])
                rsi_arr = _kernel_rsi(closes32, period)
                return float(rsi_arr[-1])
            
            # Stochastic (15-16)
            elif indicator_index == 15:
                k_period = int(param0) if param0 > 0 else int(_default_params(indicator_index)[0])
                # Kernel-like stochastic without TA-Lib to ensure parity
                def _kernel_stochastic(highs, lows, closes, period, smooth_k=3):
                    n = len(closes)
                    out = np.full(n, np.float32(50.0), dtype=np.float32)
                    if period <= 0 or n == 0:
                        return out
                    for i in range(n):
                        if i < period - 1:
                            out[i] = np.float32(50.0)
                            continue
                        hi = highs[i]
                        lo = lows[i]
                        for j in range(1, period):
                            hi = np.float32(max(hi, highs[i - j]))
                            lo = np.float32(min(lo, lows[i - j]))
                        rng = hi - lo
                        if rng < np.float32(1e-10):
                            out[i] = np.float32(50.0)
                        else:
                            out[i] = np.float32(((closes[i] - lo) / rng) * 100.0)
                    # Smooth with SMA
                    if smooth_k > 1:
                        smoothed = np.zeros(n, dtype=np.float32)
                        for i in range(n):
                            if i < smooth_k - 1:
                                smoothed[i] = out[i]
                                continue
                            s = np.float32(0.0)
                            for j in range(smooth_k):
                                s = np.float32(s + out[i - j])
                            smoothed[i] = np.float32(s / np.float32(smooth_k))
                        out = smoothed
                    return out
                slowk = _kernel_stochastic(highs32, lows32, closes32, k_period, smooth_k=3)
                return float(slowk[-1])
            elif indicator_index == 16:
                k_period = int(param0) if param0 > 0 else int(_default_params(indicator_index)[0])
                # Implement kernel-like StochRSI: compute RSI, then run stochastic on RSI values
                rsi_arr = None
                # Use our kernel_rsi helper from above by computing a small array and reusing method
                def _kernel_stochrsi(rsi_vals, period):
                    n = len(rsi_vals)
                    out = np.full(n, np.float32(50.0), dtype=np.float32)
                    if period <= 0 or n == 0:
                        return out
                    for i in range(n):
                        if i < period - 1:
                            out[i] = np.float32(50.0)
                            continue
                        hi = rsi_vals[i]
                        lo = rsi_vals[i]
                        for j in range(1, period):
                            hi = np.float32(max(hi, rsi_vals[i - j]))
                            lo = np.float32(min(lo, rsi_vals[i - j]))
                        rng = hi - lo
                        if rng < np.float32(1e-10):
                            out[i] = np.float32(50.0)
                        else:
                            out[i] = np.float32(((rsi_vals[i] - lo) / rng) * 100.0)
                    return out

                # Kernel uses RSI(14) buffer for StochRSI, so compute RSI with default of indicator 13
                rsi_period_for_stoch = int(_default_params(13)[0])
                rsi_arr = _kernel_rsi(closes32, rsi_period_for_stoch)
                stoch_rsi = _kernel_stochrsi(rsi_arr, k_period)
                return float(stoch_rsi[-1])
            
            # Momentum (17) - kernel implementation: close - close(period)
            elif indicator_index == 17:
                period = int(param0) if param0 > 0 else int(_default_params(indicator_index)[0])
                if len(closes) <= period:
                    return 0.0
                return float(closes[-1] - closes[-1 - period])
            
            # ROC (18)
            elif indicator_index == 18:
                period = int(param0) if param0 > 0 else int(_default_params(indicator_index)[0])
                return talib.ROC(closes, timeperiod=period)[-1]
            
            # Williams %R (19)
            elif indicator_index == 19:
                period = int(param0) if param0 > 0 else int(_default_params(indicator_index)[0])
                return talib.WILLR(highs, lows, closes, timeperiod=period)[-1]
            
            # ATR (20-21)
            elif indicator_index in [20, 21]:
                period = int(param0) if param0 > 0 else int(_default_params(indicator_index)[0])
                atr_arr = _kernel_atr(highs32, lows32, closes32, period)
                return float(atr_arr[-1])
            
            # NATR (22)
            elif indicator_index == 22:
                period = int(param0) if param0 > 0 else int(_default_params(indicator_index)[0])
                return talib.NATR(highs, lows, closes, timeperiod=period)[-1]
            
            # Bollinger Bands Upper (23)
            elif indicator_index == 23:
                period = int(param0) if param0 > 0 else int(_default_params(indicator_index)[0])
                stddev = float(param1) if param1 > 0 else 2.0
                upper, lower = _kernel_bollinger(closes32, period, stddev)
                return float(upper[-1])
            
            # Bollinger Bands Lower (24)
            elif indicator_index == 24:
                period = int(param0) if param0 > 0 else int(_default_params(indicator_index)[0])
                stddev = float(param1) if param1 > 0 else 2.0
                upper, lower = _kernel_bollinger(closes32, period, stddev)
                return float(lower[-1])
            
            # Keltner Channel (25)
            elif indicator_index == 25:
                # Keltner Channel uses EMA 20 and ATR 10 with multiplier 2.0 in GPU precompute
                period = int(param0) if param0 > 0 else int(_default_params(indicator_index)[0])
                atr_period = int(10)
                multiplier = 2.0
                
                ema = talib.EMA(closes, timeperiod=period)
                atr = talib.ATR(highs, lows, closes, timeperiod=atr_period)
                
                # Keltner = EMA ± ATR * multiplier
                keltner_upper = ema[-1] + (atr[-1] * multiplier)
                return keltner_upper
            
            # MACD (26)
            elif indicator_index == 26:
                fast = int(param0) if param0 > 0 else int(_default_params(indicator_index)[0])
                slow = int(param1) if param1 > 0 else int(_default_params(indicator_index)[1])
                signal = int(param2) if param2 > 0 else int(_default_params(indicator_index)[2])
                macd_arr = _kernel_macd(closes32, fast, slow, signal)
                return float(macd_arr[-1])
            
            # ADX (27)
            elif indicator_index == 27:
                period = int(param0) if param0 > 0 else int(_default_params(indicator_index)[0])
                # Implement kernel-like ADX computation to match GPU exactly
                def _kernel_adx(highs, lows, closes, period):
                    n = len(closes)
                    out = np.full(n, np.nan, dtype=np.float32)
                    if n < period * 2:
                        return out

                    smoothed_tr = np.float32(0.0)
                    smoothed_plus_dm = np.float32(0.0)
                    smoothed_minus_dm = np.float32(0.0)
                    dx_sum = np.float32(0.0)
                    prev_adx = np.float32(0.0)

                    # Initial smoothing using first 'period' bars (from 1..period inclusive)
                    for i in range(1, period + 1):
                        if i == 0:
                            tr = np.float32(highs[i] - lows[i])
                        else:
                            hl = np.float32(highs[i] - lows[i])
                            hc = np.float32(abs(highs[i] - closes[i - 1]))
                            lc = np.float32(abs(lows[i] - closes[i - 1]))
                            tr = np.float32(max(hl, max(hc, lc)))
                        plus_dm = np.float32(0.0)
                        minus_dm = np.float32(0.0)
                        if highs[i] - highs[i - 1] > lows[i - 1] - lows[i]:
                            plus_dm = np.float32(max(highs[i] - highs[i - 1], np.float32(0.0)))
                        if lows[i - 1] - lows[i] > highs[i] - highs[i - 1]:
                            minus_dm = np.float32(max(lows[i - 1] - lows[i], np.float32(0.0)))
                        smoothed_tr = np.float32(smoothed_tr + tr)
                        smoothed_plus_dm = np.float32(smoothed_plus_dm + plus_dm)
                        smoothed_minus_dm = np.float32(smoothed_minus_dm + minus_dm)

                    # Compute DX values and accumulate for SMA
                    for bar in range(period, period * 2):
                        if bar > period:
                            # Update smoothed values using Wilder's method
                            hl = np.float32(highs[bar] - lows[bar])
                            hc = np.float32(abs(highs[bar] - closes[bar - 1]))
                            lc = np.float32(abs(lows[bar] - closes[bar - 1]))
                            tr = np.float32(max(hl, max(hc, lc)))
                            plus_dm = np.float32(0.0)
                            minus_dm = np.float32(0.0)
                            if highs[bar] - highs[bar - 1] > lows[bar - 1] - lows[bar]:
                                plus_dm = np.float32(max(highs[bar] - highs[bar - 1], np.float32(0.0)))
                            if lows[bar - 1] - lows[bar] > highs[bar] - highs[bar - 1]:
                                minus_dm = np.float32(max(lows[bar - 1] - lows[bar], np.float32(0.0)))
                            smoothed_tr = np.float32(smoothed_tr - (smoothed_tr / np.float32(period)) + tr)
                            smoothed_plus_dm = np.float32(smoothed_plus_dm - (smoothed_plus_dm / np.float32(period)) + plus_dm)
                            smoothed_minus_dm = np.float32(smoothed_minus_dm - (smoothed_minus_dm / np.float32(period)) + minus_dm)

                        plus_di = np.float32((smoothed_plus_dm / smoothed_tr) * 100.0) if smoothed_tr > 0.0 else np.float32(0.0)
                        minus_di = np.float32((smoothed_minus_dm / smoothed_tr) * 100.0) if smoothed_tr > 0.0 else np.float32(0.0)
                        dx = np.float32(((abs(plus_di - minus_di) / (plus_di + minus_di)) * 100.0) if (plus_di + minus_di) > 0.0 else np.float32(0.0))
                        dx_sum = np.float32(dx_sum + dx)

                    prev_adx = np.float32(dx_sum / np.float32(period))

                    # Fill final output values
                    for bar in range(n):
                        if bar < period * 2 - 1:
                            out[bar] = np.float32(np.nan)
                        elif bar == period * 2 - 1:
                            out[bar] = prev_adx
                        else:
                            hl = np.float32(highs[bar] - lows[bar])
                            hc = np.float32(abs(highs[bar] - closes[bar - 1]))
                            lc = np.float32(abs(lows[bar] - closes[bar - 1]))
                            tr = np.float32(max(hl, max(hc, lc)))
                            plus_dm = np.float32(0.0)
                            minus_dm = np.float32(0.0)
                            if highs[bar] - highs[bar - 1] > lows[bar - 1] - lows[bar]:
                                plus_dm = np.float32(max(highs[bar] - highs[bar - 1], np.float32(0.0)))
                            if lows[bar - 1] - lows[bar] > highs[bar] - highs[bar - 1]:
                                minus_dm = np.float32(max(lows[bar - 1] - lows[bar], np.float32(0.0)))
                            smoothed_tr = np.float32(smoothed_tr - (smoothed_tr / np.float32(period)) + tr)
                            smoothed_plus_dm = np.float32(smoothed_plus_dm - (smoothed_plus_dm / np.float32(period)) + plus_dm)
                            smoothed_minus_dm = np.float32(smoothed_minus_dm - (smoothed_minus_dm / np.float32(period)) + minus_dm)
                            plus_di = np.float32((smoothed_plus_dm / smoothed_tr) * 100.0) if smoothed_tr > 0.0 else np.float32(0.0)
                            minus_di = np.float32((smoothed_minus_dm / smoothed_tr) * 100.0) if smoothed_tr > 0.0 else np.float32(0.0)
                            dx = np.float32(((abs(plus_di - minus_di) / (plus_di + minus_di)) * 100.0) if (plus_di + minus_di) > 0.0 else np.float32(0.0))
                            prev_adx = np.float32((prev_adx * np.float32(period - 1) + dx) / np.float32(period))
                            out[bar] = prev_adx
                    return out

                return float(_kernel_adx(highs32, lows32, closes32, period)[-1])
            
            # Aroon Up (28)
            elif indicator_index == 28:
                period = int(param0) if param0 > 0 else int(_default_params(indicator_index)[0])
                # Use local kernel-style implementation for parity with GPU
                # inclusive/exclusive semantics: GPU uses exclusive window [bar-period, bar-1]
                def _kernel_aroon_up(highs, period):
                    out = np.full(len(highs), np.nan, dtype=np.float32)
                    if len(highs) < period + 1:
                        return out
                    for bar in range(len(highs)):
                        if bar < period:
                            out[bar] = np.nan
                            continue
                        start = bar - period
                        window = highs[start:bar]  # exclusive current bar
                        # rightmost occurrence
                        maxv = np.max(window)
                        # find rightmost index where window == maxv
                        rel_idx = len(window) - 1 - np.argmax(window[::-1])
                        idx = start + rel_idx
                        bars_since_high = bar - idx
                        out[bar] = ((period - bars_since_high) / float(period)) * 100.0
                    return out

                return float(_kernel_aroon_up(highs32, period)[-1])
            
            # CCI (29)
            elif indicator_index == 29:
                period = int(param0) if param0 > 0 else int(_default_params(indicator_index)[0])
                return talib.CCI(highs, lows, closes, timeperiod=period)[-1]
            
            # DPO (30) - Detrended Price Oscillator
            elif indicator_index == 30:
                period = int(param0) if param0 > 0 else int(_default_params(indicator_index)[0])
                offset = int(period / 2) + 1
                # GPU: out[bar] = ohlcv[bar - offset].close - compute_sma_helper(ohlcv, bar - offset, period)
                # Need at least (period - 1 + offset) bars to compute
                if len(closes) < period - 1 + offset:
                    dpo = 0.0
                else:
                    # Compute SMA at the offset bar
                    historical_idx = len(closes) - 1 - offset
                    if historical_idx < period - 1:
                        dpo = 0.0
                    else:
                        # Compute SMA at historical_idx
                        s = np.float32(0.0)
                        for j in range(historical_idx - period + 1, historical_idx + 1):
                            s = np.float32(s + np.float32(closes[j]))
                        sma_val = np.float32(s / np.float32(period))
                        dpo = float(np.float32(closes[historical_idx]) - sma_val)
                return float(dpo)
            
            # Parabolic SAR (31)
            elif indicator_index == 31:
                acceleration = float(param0) if param0 > 0 else float(_default_params(indicator_index)[0])
                maximum = float(param1) if param1 > 0 else float(_default_params(indicator_index)[1])
                # Use kernel-like PSAR implementation
                psar_arr = _kernel_psar(opens32, highs32, lows32, closes32, acceleration, maximum)
                return float(psar_arr[-1])
            
            # SuperTrend (32)
            elif indicator_index == 32:
                period = int(param0) if param0 > 0 else int(_default_params(indicator_index)[0])
                multiplier = float(param1) if param1 > 0 else float(_default_params(indicator_index)[1])
                # Compute kernel-like SuperTrend using our helper
                atr_period_default = int(get_gpu_default_params(20)[0])
                super_arr = _kernel_supertrend(highs32, lows32, closes32, period, multiplier, atr_period=atr_period_default)
                # Return latest value (kernel computes full history every time with trend state tracking inside helper)
                return float(super_arr[-1])
            
            # Linear Regression Slope (33-35)
            elif indicator_index in [33, 34, 35]:
                period = int(param0) if param0 > 0 else int(_default_params(indicator_index)[0])
                slope = talib.LINEARREG_SLOPE(closes, timeperiod=period)[-1]
                return slope
            
            # OBV (36) - implement GPU kernel logic exactly
            elif indicator_index == 36:
                # On-balance volume: cumulative volume added/subtracted based on close direction
                if len(closes) == 0:
                    return 0.0
                obv = 0.0
                for i in range(len(closes)):
                    if i == 0:
                        # GPU kernel sets first bar to 0.0
                        continue
                    if closes[i] > closes[i-1]:
                        obv += volumes[i]
                    elif closes[i] < closes[i-1]:
                        obv -= volumes[i]
                return float(obv)
            
            # VWAP (37) - Volume-Weighted Average Price
            elif indicator_index == 37:
                # Session-based VWAP (resets daily)
                self._check_and_reset_vwap_session()
                
                # Update cumulative values
                typical_price_current = np.float32((highs[-1] + lows[-1] + closes[-1]) / 3.0)
                self.vwap_cumulative_tp_vol = np.float32(self.vwap_cumulative_tp_vol + np.float32(typical_price_current * np.float32(volumes[-1])))
                self.vwap_cumulative_vol = np.float32(self.vwap_cumulative_vol + np.float32(volumes[-1]))
                
                # Calculate VWAP
                if self.vwap_cumulative_vol > 0:
                    vwap = self.vwap_cumulative_tp_vol / self.vwap_cumulative_vol
                else:
                    vwap = closes[-1]
                
                return vwap
            
            # MFI (38) - Money Flow Index
            elif indicator_index == 38:
                period = int(param0) if param0 > 0 else int(_default_params(indicator_index)[0])
                return talib.MFI(highs, lows, closes, volumes, timeperiod=period)[-1]
            
            # A/D (39) - Accumulation/Distribution - implement GPU kernel logic
            elif indicator_index == 39:
                if len(closes32) == 0:
                    return 0.0
                ad = np.float32(0.0)
                for i in range(len(closes32)):
                    hl_range = np.float32(highs32[i] - lows32[i])
                    if hl_range < np.float32(1e-10):
                        # keep previous ad value
                        continue
                    clv = np.float32(((closes32[i] - lows32[i]) - (highs32[i] - closes32[i])) / hl_range)
                    ad = np.float32(ad + np.float32(clv * volumes32[i]))
                return float(ad)
            
            # Volume SMA (40)
            elif indicator_index == 40:
                period = int(param0) if param0 > 0 else int(_default_params(indicator_index)[0])
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
                # Support/Resistance (GPU: returns midpoint of highest/lowest in period)
                period = int(_default_params(indicator_index)[0])
                if len(highs) >= period:
                    highest = np.max(highs[-period:])
                    lowest = np.min(lows[-period:])
                    # GPU kernel: out[bar] = (highest + lowest) / 2.0f
                    return (highest + lowest) / 2.0
                return closes[-1]
            
            # Price Channel (45)
            elif indicator_index == 45:
                period = int(_default_params(indicator_index)[0])
                if len(highs) >= period:
                    highest = np.max(highs[-period:])
                    # GPU kernel: returns highest (not midpoint)
                    return highest
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
            
            # Price Acceleration (48) - kernel's second derivative of momentum
            elif indicator_index == 48:
                # Price Acceleration (GPU kernel uses period parameter)
                period = int(param0) if param0 > 0 else int(_default_params(indicator_index)[0])
                # GPU: velocity_now = close[bar] - close[bar - period]
                #      velocity_prev = close[bar-1] - close[bar - period - 1]
                #      out = velocity_now - velocity_prev
                bar = len(closes) - 1
                if bar < period + 1:
                    return 0.0
                velocity_now = float(closes[bar] - closes[bar - period])
                velocity_prev = float(closes[bar - 1] - closes[bar - period - 1])
                return float(velocity_now - velocity_prev)
            
            # Volume ROC (49)
            elif indicator_index == 49:
                period = int(_default_params(indicator_index)[0])
                return talib.ROC(volumes, timeperiod=period)[-1]
            
            else:
                return 0.0
                
        except Exception as e:
            # Include parameter and buffer context for parity debugging
            import traceback
            tb = traceback.format_exc()
            from ..utils.validation import log_error
            log_error(f"Indicator calculation error (index {indicator_index}, name {indicator_name}) - params: {param0}, {param1}, {param2}, bars_count={self.bars_count}\n{tb}")
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

    def compute_warmup_for_bot(self, bot_config) -> int:
        """
        Compute warmup bars required for a given bot config.
        Mirrors GPU kernel warmup computation logic used in backtest_with_precomputed.cl
        Returns the number of bars required to warm-up indicators.
        """
        warmup_bars = 0
        # Helper to resolve fixed periods used by GPU precompute, using centralized defaults
        for i in range(bot_config.num_indicators):
            idx = int(bot_config.indicator_indices[i])
            # Prefer GPU fixed precompute params to compute warmup lengths
            p = int(bot_config.indicator_params[i][0]) if bot_config.indicator_params[i][0] > 0 else int(get_gpu_default_params(idx)[0])
            p2 = int(bot_config.indicator_params[i][1]) if bot_config.indicator_params[i][1] > 0 else 0
            p3 = int(bot_config.indicator_params[i][2]) if bot_config.indicator_params[i][2] > 0 else 0

            indicator_warmup = 0
            # Moving Averages (SMA/EMA/WMA/DEMA/TEMA/KAMA)
            if 0 <= idx <= 5:
                indicator_warmup = p
            elif 6 <= idx <= 11:
                indicator_warmup = p * 5
            elif 12 <= idx <= 14:
                indicator_warmup = p * 2
            elif idx == 15:
                indicator_warmup = p + p2
            elif idx == 16:
                indicator_warmup = p * 3
            elif 17 <= idx <= 19:
                indicator_warmup = p + 10
            elif 20 <= idx <= 22:
                indicator_warmup = p * 2
            elif idx in (23, 24):
                indicator_warmup = p * 5
            elif idx == 25:
                indicator_warmup = int(p * 2.5) if p > 0 else 0
            elif idx == 26:
                indicator_warmup = p2 * 5 + p3 * 3
            elif idx == 27:
                indicator_warmup = p * 2
            elif 28 <= idx <= 35:
                indicator_warmup = int(p * 1.5) if p > 0 else 0
            elif 36 <= idx <= 40:
                indicator_warmup = p + 20
            elif 41 <= idx <= 45:
                indicator_warmup = p + 10
            elif 46 <= idx <= 49:
                indicator_warmup = 20

            if indicator_warmup > warmup_bars:
                warmup_bars = indicator_warmup

        return warmup_bars
