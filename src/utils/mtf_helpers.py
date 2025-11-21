"""
Multi-Timeframe Helper Functions

Provides utilities for multi-timeframe analysis to improve signal quality
by ensuring trades align with higher timeframe trends.
"""

def get_higher_timeframes(base_timeframe: str) -> tuple:
    """
    Get the two higher timeframes for multi-timeframe analysis.
    
    Args:
        base_timeframe: The trading timeframe (e.g., "1m", "15m", "1h")
        
    Returns:
        Tuple of (htf1, htf2) - two higher timeframes
        Returns (None, None) if base_timeframe is already the highest
    """
    timeframe_hierarchy = {
        "1m": ("5m", "15m"),
        "5m": ("15m", "1h"),
        "15m": ("1h", "4h"),
        "30m": ("1h", "4h"),
        "1h": ("4h", "1d"),
        "4h": ("1d", None),
        "1d": (None, None)
    }
    
    return timeframe_hierarchy.get(base_timeframe, (None, None))


def get_timeframe_multiplier(from_tf: str, to_tf: str) -> int:
    """
    Get the multiplier between two timeframes.
    
    Args:
        from_tf: Source timeframe (e.g., "1m")
        to_tf: Target timeframe (e.g., "15m")
        
    Returns:
        Integer multiplier (e.g., 15 for 1m->15m)
    """
    # Convert timeframes to minutes
    tf_to_minutes = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "4h": 240,
        "1d": 1440
    }
    
    from_minutes = tf_to_minutes.get(from_tf, 1)
    to_minutes = tf_to_minutes.get(to_tf, 1)
    
    return to_minutes // from_minutes


def resample_ohlcv_to_higher_tf(ohlcv_data, multiplier: int):
    """
    Resample OHLCV data to a higher timeframe.
    
    Args:
        ohlcv_data: numpy array of shape (N, 5) with OHLCV data
        multiplier: How many lower TF bars make 1 higher TF bar
        
    Returns:
        Resampled OHLCV data as numpy array
    """
    import numpy as np
    
    num_bars = len(ohlcv_data)
    num_htf_bars = num_bars // multiplier
    
    # Pre-allocate output array
    htf_data = np.zeros((num_htf_bars, 5), dtype=np.float32)
    
    for i in range(num_htf_bars):
        start_idx = i * multiplier
        end_idx = start_idx + multiplier
        
        if end_idx > num_bars:
            break
            
        chunk = ohlcv_data[start_idx:end_idx]
        
        # OHLCV aggregation
        htf_data[i, 0] = chunk[0, 0]  # Open = first bar's open
        htf_data[i, 1] = np.max(chunk[:, 1])  # High = highest high
        htf_data[i, 2] = np.min(chunk[:, 2])  # Low = lowest low
        htf_data[i, 3] = chunk[-1, 3]  # Close = last bar's close
        htf_data[i, 4] = np.sum(chunk[:, 4])  # Volume = sum of volumes
    
    return htf_data


def get_mtf_config(base_timeframe: str) -> dict:
    """
    Get the complete multi-timeframe configuration.
    
    Args:
        base_timeframe: The trading timeframe
        
    Returns:
        Dictionary with MTF configuration
    """
    htf1, htf2 = get_higher_timeframes(base_timeframe)
    
    config = {
        "base_tf": base_timeframe,
        "htf1": htf1,
        "htf2": htf2,
        "enabled": htf1 is not None,  # Disable for 1d timeframe
    }
    
    if htf1:
        config["htf1_multiplier"] = get_timeframe_multiplier(base_timeframe, htf1)
    if htf2:
        config["htf2_multiplier"] = get_timeframe_multiplier(base_timeframe, htf2)
    
    return config


# Trend indicators for higher timeframe analysis
# These are the key indicators used to determine HTF trend
MTF_TREND_INDICATORS = {
    "ema_fast": 0,  # EMA(20) - indicator index
    "ema_slow": 1,  # EMA(50) - indicator index
    "macd": 26,     # MACD - indicator index
    "adx": 27,      # ADX - indicator index
}


def calculate_trend_strength(ema_fast, ema_slow, macd, adx):
    """
    Calculate trend direction and strength from HTF indicators.
    
    Args:
        ema_fast: Fast EMA value
        ema_slow: Slow EMA value
        macd: MACD value
        adx: ADX value
        
    Returns:
        -1 (bearish), 0 (neutral), +1 (bullish)
    """
    signals = []
    
    # EMA cross
    if ema_fast > ema_slow * 1.001:  # 0.1% threshold
        signals.append(1)
    elif ema_fast < ema_slow * 0.999:
        signals.append(-1)
    else:
        signals.append(0)
    
    # MACD
    if macd > 0:
        signals.append(1)
    elif macd < 0:
        signals.append(-1)
    else:
        signals.append(0)
    
    # ADX strength (only if >25, ignore if weak)
    if adx > 25:
        # Trend is strong, use EMA direction
        if signals[0] != 0:  # Use EMA signal
            signals.append(signals[0])
    
    # Consensus: majority vote
    bullish = sum(1 for s in signals if s == 1)
    bearish = sum(1 for s in signals if s == -1)
    
    if bullish > bearish and bullish >= 2:
        return 1  # Bullish trend
    elif bearish > bullish and bearish >= 2:
        return -1  # Bearish trend
    else:
        return 0  # Neutral / unclear
