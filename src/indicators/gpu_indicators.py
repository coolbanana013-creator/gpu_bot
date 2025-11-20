"""
GPU Indicator Mappings

This module defines the exact 50 indicators used in the GPU kernels (precompute_all_indicators.cl).
These must match the switch statement in the GPU kernel EXACTLY (indices 0-49).
"""

from enum import IntEnum
from typing import List, Tuple


class GPUIndicatorIndex(IntEnum):
    """
    GPU indicator indices matching precompute_all_indicators.cl switch statement.
    CRITICAL: These indices must match the OpenCL kernel implementation exactly!
    """
    # MOVING AVERAGES (0-11)
    SMA_5 = 0
    SMA_10 = 1
    SMA_20 = 2
    SMA_50 = 3
    SMA_100 = 4
    SMA_200 = 5
    EMA_5 = 6
    EMA_10 = 7
    EMA_20 = 8
    EMA_50 = 9
    EMA_100 = 10
    EMA_200 = 11
    
    # MOMENTUM (12-19)
    RSI_7 = 12
    RSI_14 = 13
    RSI_21 = 14
    STOCH_14 = 15
    STOCHRSI_14 = 16
    MOMENTUM_10 = 17
    ROC_10 = 18
    WILLIAMS_R_14 = 19
    
    # VOLATILITY (20-25)
    ATR_14 = 20
    ATR_20 = 21
    NATR_14 = 22
    BB_UPPER_20 = 23
    BB_LOWER_20 = 24
    KELTNER_20 = 25
    
    # TREND (26-35)
    MACD_12_26_9 = 26
    ADX_14 = 27
    AROON_UP_25 = 28
    CCI_20 = 29
    DPO_20 = 30
    PSAR = 31
    SUPERTREND_10 = 32
    TREND_STRENGTH_20 = 33
    TREND_STRENGTH_50 = 34
    TREND_STRENGTH_100 = 35
    
    # VOLUME (36-40)
    OBV = 36
    VWAP = 37
    MFI_14 = 38
    AD = 39
    VOLUME_SMA_20 = 40
    
    # PATTERN (41-45)
    PIVOT_POINTS = 41
    FRACTAL_HIGH_5 = 42
    FRACTAL_LOW_5 = 43
    SUPPORT_RESISTANCE_20 = 44
    PRICE_CHANNEL_20 = 45
    
    # SIMPLE (46-49)
    HIGH_LOW_RANGE = 46
    CLOSE_POSITION = 47
    PRICE_ACCELERATION_10 = 48
    VOLUME_ROC_10 = 49


# Human-readable names for indicators (for logging/CSV)
GPU_INDICATOR_NAMES = {
    # Moving Averages
    0: "SMA(5)",
    1: "SMA(10)",
    2: "SMA(20)",
    3: "SMA(50)",
    4: "SMA(100)",
    5: "SMA(200)",
    6: "EMA(5)",
    7: "EMA(10)",
    8: "EMA(20)",
    9: "EMA(50)",
    10: "EMA(100)",
    11: "EMA(200)",
    
    # Momentum
    12: "RSI(7)",
    13: "RSI(14)",
    14: "RSI(21)",
    15: "Stoch(14)",
    16: "StochRSI(14)",
    17: "Momentum(10)",
    18: "ROC(10)",
    19: "WilliamsR(14)",
    
    # Volatility
    20: "ATR(14)",
    21: "ATR(20)",
    22: "NATR(14)",
    23: "BB_Upper(20)",
    24: "BB_Lower(20)",
    25: "Keltner(20)",
    
    # Trend
    26: "MACD(12,26,9)",
    27: "ADX(14)",
    28: "AroonUp(25)",
    29: "CCI(20)",
    30: "DPO(20)",
    31: "PSAR",
    32: "SuperTrend(10)",
    33: "TrendStrength(20)",
    34: "TrendStrength(50)",
    35: "TrendStrength(100)",
    
    # Volume
    36: "OBV",
    37: "VWAP",
    38: "MFI(14)",
    39: "A/D",
    40: "VolumeSMA(20)",
    
    # Pattern
    41: "PivotPoints",
    42: "FractalHigh(5)",
    43: "FractalLow(5)",
    44: "Support/Resistance(20)",
    45: "PriceChannel(20)",
    
    # Simple
    46: "HighLowRange",
    47: "ClosePosition",
    48: "PriceAccel(10)",
    49: "VolumeROC(10)",
}


def get_all_gpu_indicators() -> List[int]:
    """
    Get all available GPU indicator indices (0-49), excluding problematic indicators.
    
    Excluded indicators (cause low trade frequency):
    - 40: Volume_SMA_20 (50.6% of low-trade bots use this, 0% of successful bots)
    - 49: VOLUME_ROC_10 (14.8% low-trade vs 0% high-trade, volume-based indicator issue)
    
    Analysis showed these volume indicators are associated with significantly lower trade frequency.
    """
    # Start with all indicators 0-49
    all_indicators = list(range(50))
    
    # Remove problematic volume indicators
    excluded = {40, 49}  # Volume_SMA_20, Volume_ROC_10
    
    return [i for i in all_indicators if i not in excluded]


def get_gpu_indicator_name(index: int) -> str:
    """Get human-readable name for a GPU indicator index."""
    return GPU_INDICATOR_NAMES.get(index, f"Unknown({index})")


def validate_indicator_index(index: int) -> bool:
    """Validate that an indicator index is in valid range."""
    return 0 <= index < 50


def get_indicator_category(index: int) -> str:
    """Get category name for an indicator."""
    if 0 <= index <= 11:
        return "Moving Averages"
    elif 12 <= index <= 19:
        return "Momentum"
    elif 20 <= index <= 25:
        return "Volatility"
    elif 26 <= index <= 35:
        return "Trend"
    elif 36 <= index <= 40:
        return "Volume"
    elif 41 <= index <= 45:
        return "Pattern"
    elif 46 <= index <= 49:
        return "Simple"
    else:
        return "Unknown"


# Export count for validation
GPU_INDICATOR_COUNT = 50


def verify_indicator_mapping():
    """Verify that all 50 indicators are properly mapped."""
    assert len(GPU_INDICATOR_NAMES) == GPU_INDICATOR_COUNT, \
        f"Expected {GPU_INDICATOR_COUNT} indicators, got {len(GPU_INDICATOR_NAMES)}"
    
    # Verify all indices 0-49 are present
    for i in range(GPU_INDICATOR_COUNT):
        assert i in GPU_INDICATOR_NAMES, f"Missing indicator index {i}"
    
    print(f"✓ GPU indicator mapping verified: {GPU_INDICATOR_COUNT} indicators")


if __name__ == "__main__":
    verify_indicator_mapping()
    print("\nGPU Indicators by Category:")
    for i in range(50):
        category = get_indicator_category(i)
        name = get_gpu_indicator_name(i)
        print(f"  [{i:2d}] {category:20s} {name}")
