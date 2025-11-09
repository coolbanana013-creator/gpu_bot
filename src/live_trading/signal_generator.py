"""
Real-Time Signal Generator

Generates trading signals using 100% consensus logic (matches GPU kernel exactly).
"""

from typing import Dict, Optional
import numpy as np

from ..utils.validation import log_debug


class SignalGenerator:
    """
    Generate signals with 100% consensus (ALL indicators must agree).
    Replicates GPU kernel generate_signal_consensus() exactly.
    """
    
    def __init__(self):
        """Initialize signal generator."""
        self.consensus_threshold = 1.0  # 100% consensus required
    
    def classify_indicator_signal(
        self,
        indicator_index: int,
        current_value: float,
        previous_value: Optional[float] = None,
        indicator_history: Optional[np.ndarray] = None
    ) -> int:
        """
        Classify indicator as bullish (1), bearish (-1), or neutral (0).
        Matches GPU kernel logic exactly.
        
        Args:
            indicator_index: Index 0-49
            current_value: Current indicator value
            previous_value: Previous bar's value (for trend indicators)
            indicator_history: Last 5 values for moving average comparison
        
        Returns:
            Signal: 1 (bullish), -1 (bearish), 0 (neutral)
        """
        # Moving averages (0-11): compare with previous value
        if 0 <= indicator_index <= 11:
            if previous_value is not None:
                if current_value > previous_value:
                    return 1
                elif current_value < previous_value:
                    return -1
            return 0
        
        # RSI (12-14): overbought/oversold
        elif 12 <= indicator_index <= 14:
            if current_value < 30.0:
                return 1  # Oversold = bullish
            elif current_value > 70.0:
                return -1  # Overbought = bearish
            return 0
        
        # Stochastic (15-16): overbought/oversold
        elif 15 <= indicator_index <= 16:
            if current_value < 20.0:
                return 1
            elif current_value > 80.0:
                return -1
            return 0
        
        # Momentum indicators (17-19): positive/negative
        elif 17 <= indicator_index <= 19:
            if current_value > 0.0:
                return 1
            elif current_value < 0.0:
                return -1
            return 0
        
        # ROC (20-22): positive/negative
        elif 20 <= indicator_index <= 22:
            if current_value > 0.0:
                return 1
            elif current_value < 0.0:
                return -1
            return 0
        
        # Bollinger Bands (23-25): use with price comparison
        # For simplicity, compare current value with previous
        elif 23 <= indicator_index <= 25:
            if previous_value is not None:
                if current_value > previous_value:
                    return 1
                elif current_value < previous_value:
                    return -1
            return 0
        
        # MACD (26): positive/negative
        elif indicator_index == 26:
            if current_value > 0.0:
                return 1
            elif current_value < 0.0:
                return -1
            return 0
        
        # ADX (27-28): trend strength
        elif 27 <= indicator_index <= 28:
            if current_value > 25.0:
                # Strong trend - use direction
                if previous_value is not None:
                    if current_value > previous_value:
                        return 1
                    else:
                        return -1
            return 0
        
        # CCI (29): overbought/oversold
        elif indicator_index == 29:
            if current_value < -100.0:
                return 1
            elif current_value > 100.0:
                return -1
            return 0
        
        # Williams %R (30): overbought/oversold
        elif indicator_index == 30:
            if current_value < -80.0:
                return 1
            elif current_value > -20.0:
                return -1
            return 0
        
        # ATR (31-33): volatility (use with trend)
        elif 31 <= indicator_index <= 33:
            if previous_value is not None:
                if current_value > previous_value:
                    return 1
                elif current_value < previous_value:
                    return -1
            return 0
        
        # SAR (34-35): compare with price (needs price context)
        elif 34 <= indicator_index <= 35:
            # SAR interpretation needs price - skip for now
            return 0
        
        # Volume indicators (36-40): increasing volume
        elif 36 <= indicator_index <= 40:
            if previous_value is not None:
                if current_value > previous_value:
                    return 1
                elif current_value < previous_value:
                    return -1
            return 0
        
        # Price patterns (41-49): use moving average comparison
        else:
            if indicator_history is not None and len(indicator_history) >= 5:
                avg = np.mean(indicator_history[-5:])
                if current_value > avg * 1.01:
                    return 1
                elif current_value < avg * 0.99:
                    return -1
            return 0
    
    def generate_signal(
        self,
        indicator_values: Dict[int, float],
        indicator_history: Dict[int, np.ndarray]
    ) -> float:
        """
        Generate consensus signal (matches GPU kernel exactly).
        
        Args:
            indicator_values: Dict mapping indicator_index -> current_value
            indicator_history: Dict mapping indicator_index -> array of last N values
        
        Returns:
            Signal: 1.0 (buy), -1.0 (sell), 0.0 (no consensus)
        """
        if not indicator_values:
            return 0.0
        
        bullish_count = 0
        bearish_count = 0
        num_indicators = len(indicator_values)
        
        for ind_idx, current_value in indicator_values.items():
            # Get previous value and history
            previous_value = None
            history = None
            
            if ind_idx in indicator_history:
                hist = indicator_history[ind_idx]
                if len(hist) >= 2:
                    previous_value = hist[-2]
                if len(hist) >= 5:
                    history = hist
            
            # Classify signal
            signal = self.classify_indicator_signal(
                ind_idx,
                current_value,
                previous_value,
                history
            )
            
            if signal == 1:
                bullish_count += 1
            elif signal == -1:
                bearish_count += 1
        
        # Calculate consensus percentage
        total_signals = bullish_count + bearish_count
        if total_signals == 0:
            return 0.0
        
        bullish_pct = bullish_count / num_indicators
        bearish_pct = bearish_count / num_indicators
        
        # 100% consensus required (ALL indicators must agree)
        if bullish_pct >= self.consensus_threshold:
            return 1.0  # ALL bullish
        elif bearish_pct >= self.consensus_threshold:
            return -1.0  # ALL bearish
        else:
            return 0.0  # No unanimous consensus
    
    def get_signal_breakdown(
        self,
        indicator_values: Dict[int, float],
        indicator_history: Dict[int, np.ndarray]
    ) -> Dict:
        """
        Get detailed signal breakdown for dashboard display.
        
        Returns:
            Dict with bullish_count, bearish_count, neutral_count, signals_by_indicator
        """
        breakdown = {
            'bullish_count': 0,
            'bearish_count': 0,
            'neutral_count': 0,
            'signals_by_indicator': {}
        }
        
        for ind_idx, current_value in indicator_values.items():
            previous_value = None
            history = None
            
            if ind_idx in indicator_history:
                hist = indicator_history[ind_idx]
                if len(hist) >= 2:
                    previous_value = hist[-2]
                if len(hist) >= 5:
                    history = hist
            
            signal = self.classify_indicator_signal(
                ind_idx,
                current_value,
                previous_value,
                history
            )
            
            breakdown['signals_by_indicator'][ind_idx] = signal
            
            if signal == 1:
                breakdown['bullish_count'] += 1
            elif signal == -1:
                breakdown['bearish_count'] += 1
            else:
                breakdown['neutral_count'] += 1
        
        return breakdown
