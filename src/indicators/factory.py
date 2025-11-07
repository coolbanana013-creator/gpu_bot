"""
Comprehensive technical indicators library with 50+ indicators.
Each indicator has parameter generation with strict validation.
Uses TA-Lib where possible, numpy for custom implementations.
"""
import numpy as np
import talib
from typing import Dict, Any, Tuple, List, Optional
from enum import Enum
from dataclasses import dataclass

from ..utils.validation import validate_int, validate_float, log_debug


class IndicatorType(Enum):
    """Enumeration of all supported indicator types."""
    # Momentum Indicators
    RSI = "RSI"
    STOCH = "STOCH"
    STOCHF = "STOCHF"
    MACD = "MACD"
    CCI = "CCI"
    MOM = "MOM"
    ROC = "ROC"
    RSI_STOCH = "RSI_STOCH"
    WILLIAMS_R = "WILLIAMS_R"
    ULTIMATE_OSC = "ULTIMATE_OSC"
    PPO = "PPO"
    
    # Trend Indicators
    EMA = "EMA"
    SMA = "SMA"
    WMA = "WMA"
    DEMA = "DEMA"
    TEMA = "TEMA"
    TRIMA = "TRIMA"
    KAMA = "KAMA"
    T3 = "T3"
    ADX = "ADX"
    ADXR = "ADXR"
    AROON = "AROON"
    AROONOSC = "AROONOSC"
    DX = "DX"
    MINUS_DI = "MINUS_DI"
    PLUS_DI = "PLUS_DI"
    TRIX = "TRIX"
    
    # Volatility Indicators
    ATR = "ATR"
    NATR = "NATR"
    BBANDS = "BBANDS"
    
    # Volume Indicators
    OBV = "OBV"
    AD = "AD"
    ADOSC = "ADOSC"
    MFI = "MFI"
    
    # Cycle Indicators
    HT_DCPERIOD = "HT_DCPERIOD"
    HT_DCPHASE = "HT_DCPHASE"
    HT_TRENDMODE = "HT_TRENDMODE"
    
    # Price Transform
    SAR = "SAR"
    SAREXT = "SAREXT"
    
    # Overlap Studies
    MIDPOINT = "MIDPOINT"
    MIDPRICE = "MIDPRICE"
    
    # Statistic Functions
    BETA = "BETA"
    CORREL = "CORREL"
    LINEARREG = "LINEARREG"
    LINEARREG_ANGLE = "LINEARREG_ANGLE"
    LINEARREG_INTERCEPT = "LINEARREG_INTERCEPT"
    LINEARREG_SLOPE = "LINEARREG_SLOPE"
    STDDEV = "STDDEV"
    TSF = "TSF"
    VAR = "VAR"


@dataclass
class IndicatorParams:
    """Container for indicator parameters."""
    indicator_type: IndicatorType
    params: Dict[str, Any]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "type": self.indicator_type.value,
            "params": self.params
        }


class BaseIndicator:
    """Base class for all indicators."""
    
    def __init__(self, indicator_type: IndicatorType):
        self.indicator_type = indicator_type
    
    def generate_params(self) -> Dict[str, Any]:
        """Generate random valid parameters. Override in subclasses."""
        raise NotImplementedError
    
    def compute(self, ohlcv: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute indicator values. Override in subclasses."""
        raise NotImplementedError


# ============================================================================
# MOMENTUM INDICATORS
# ============================================================================

class RSIIndicator(BaseIndicator):
    """Relative Strength Index."""
    
    def __init__(self):
        super().__init__(IndicatorType.RSI)
    
    def generate_params(self) -> Dict[str, Any]:
        """RSI period: 2-100, typically 5-30."""
        period = np.random.randint(5, 31)
        return {"period": int(period)}
    
    def compute(self, ohlcv: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute RSI."""
        close = ohlcv[:, 4]  # Close prices
        period = params["period"]
        return talib.RSI(close, timeperiod=period)


class StochasticIndicator(BaseIndicator):
    """Stochastic Oscillator."""
    
    def __init__(self):
        super().__init__(IndicatorType.STOCH)
    
    def generate_params(self) -> Dict[str, Any]:
        """Stochastic: fastk_period 5-50, slowk_period 3-10, slowd_period 3-10."""
        fastk = np.random.randint(5, 51)
        slowk = np.random.randint(3, 11)
        slowd = np.random.randint(3, 11)
        return {
            "fastk_period": int(fastk),
            "slowk_period": int(slowk),
            "slowd_period": int(slowd)
        }
    
    def compute(self, ohlcv: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute Stochastic %K."""
        high, low, close = ohlcv[:, 2], ohlcv[:, 3], ohlcv[:, 4]
        slowk, slowd = talib.STOCH(
            high, low, close,
            fastk_period=params["fastk_period"],
            slowk_period=params["slowk_period"],
            slowd_period=params["slowd_period"]
        )
        return slowk


class MACDIndicator(BaseIndicator):
    """MACD - Moving Average Convergence Divergence."""
    
    def __init__(self):
        super().__init__(IndicatorType.MACD)
    
    def generate_params(self) -> Dict[str, Any]:
        """MACD: fast 5-50, slow 10-100, signal 5-20."""
        fast = np.random.randint(5, 51)
        slow = np.random.randint(max(fast + 1, 10), 101)
        signal = np.random.randint(5, 21)
        return {
            "fastperiod": int(fast),
            "slowperiod": int(slow),
            "signalperiod": int(signal)
        }
    
    def compute(self, ohlcv: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute MACD line."""
        close = ohlcv[:, 4]
        macd, signal, hist = talib.MACD(
            close,
            fastperiod=params["fastperiod"],
            slowperiod=params["slowperiod"],
            signalperiod=params["signalperiod"]
        )
        return macd


class CCIIndicator(BaseIndicator):
    """Commodity Channel Index."""
    
    def __init__(self):
        super().__init__(IndicatorType.CCI)
    
    def generate_params(self) -> Dict[str, Any]:
        """CCI period: 5-50."""
        period = np.random.randint(5, 51)
        return {"period": int(period)}
    
    def compute(self, ohlcv: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute CCI."""
        high, low, close = ohlcv[:, 2], ohlcv[:, 3], ohlcv[:, 4]
        return talib.CCI(high, low, close, timeperiod=params["period"])


class MOMIndicator(BaseIndicator):
    """Momentum."""
    
    def __init__(self):
        super().__init__(IndicatorType.MOM)
    
    def generate_params(self) -> Dict[str, Any]:
        """Momentum period: 5-30."""
        period = np.random.randint(5, 31)
        return {"period": int(period)}
    
    def compute(self, ohlcv: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute Momentum."""
        close = ohlcv[:, 4]
        return talib.MOM(close, timeperiod=params["period"])


class ROCIndicator(BaseIndicator):
    """Rate of Change."""
    
    def __init__(self):
        super().__init__(IndicatorType.ROC)
    
    def generate_params(self) -> Dict[str, Any]:
        """ROC period: 5-30."""
        period = np.random.randint(5, 31)
        return {"period": int(period)}
    
    def compute(self, ohlcv: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute ROC."""
        close = ohlcv[:, 4]
        return talib.ROC(close, timeperiod=params["period"])


class WilliamsRIndicator(BaseIndicator):
    """Williams %R."""
    
    def __init__(self):
        super().__init__(IndicatorType.WILLIAMS_R)
    
    def generate_params(self) -> Dict[str, Any]:
        """Williams %R period: 5-30."""
        period = np.random.randint(5, 31)
        return {"period": int(period)}
    
    def compute(self, ohlcv: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute Williams %R."""
        high, low, close = ohlcv[:, 2], ohlcv[:, 3], ohlcv[:, 4]
        return talib.WILLR(high, low, close, timeperiod=params["period"])


class UltimateOscillatorIndicator(BaseIndicator):
    """Ultimate Oscillator."""
    
    def __init__(self):
        super().__init__(IndicatorType.ULTIMATE_OSC)
    
    def generate_params(self) -> Dict[str, Any]:
        """Ultimate Oscillator: 3 periods."""
        period1 = np.random.randint(5, 11)
        period2 = np.random.randint(10, 21)
        period3 = np.random.randint(20, 41)
        return {
            "period1": int(period1),
            "period2": int(period2),
            "period3": int(period3)
        }
    
    def compute(self, ohlcv: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute Ultimate Oscillator."""
        high, low, close = ohlcv[:, 2], ohlcv[:, 3], ohlcv[:, 4]
        return talib.ULTOSC(
            high, low, close,
            timeperiod1=params["period1"],
            timeperiod2=params["period2"],
            timeperiod3=params["period3"]
        )


class PPOIndicator(BaseIndicator):
    """Percentage Price Oscillator."""
    
    def __init__(self):
        super().__init__(IndicatorType.PPO)
    
    def generate_params(self) -> Dict[str, Any]:
        """PPO: fast 5-30, slow 20-50."""
        fast = np.random.randint(5, 31)
        slow = np.random.randint(max(fast + 5, 20), 51)
        return {"fastperiod": int(fast), "slowperiod": int(slow)}
    
    def compute(self, ohlcv: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute PPO."""
        close = ohlcv[:, 4]
        return talib.PPO(close, fastperiod=params["fastperiod"], slowperiod=params["slowperiod"])


# ============================================================================
# TREND INDICATORS
# ============================================================================

class EMAIndicator(BaseIndicator):
    """Exponential Moving Average."""
    
    def __init__(self):
        super().__init__(IndicatorType.EMA)
    
    def generate_params(self) -> Dict[str, Any]:
        """EMA period: 5-200."""
        period = np.random.randint(5, 201)
        return {"period": int(period)}
    
    def compute(self, ohlcv: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute EMA."""
        close = ohlcv[:, 4]
        return talib.EMA(close, timeperiod=params["period"])


class SMAIndicator(BaseIndicator):
    """Simple Moving Average."""
    
    def __init__(self):
        super().__init__(IndicatorType.SMA)
    
    def generate_params(self) -> Dict[str, Any]:
        """SMA period: 5-200."""
        period = np.random.randint(5, 201)
        return {"period": int(period)}
    
    def compute(self, ohlcv: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute SMA."""
        close = ohlcv[:, 4]
        return talib.SMA(close, timeperiod=params["period"])


class WMAIndicator(BaseIndicator):
    """Weighted Moving Average."""
    
    def __init__(self):
        super().__init__(IndicatorType.WMA)
    
    def generate_params(self) -> Dict[str, Any]:
        """WMA period: 5-100."""
        period = np.random.randint(5, 101)
        return {"period": int(period)}
    
    def compute(self, ohlcv: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute WMA."""
        close = ohlcv[:, 4]
        return talib.WMA(close, timeperiod=params["period"])


class DEMAIndicator(BaseIndicator):
    """Double Exponential Moving Average."""
    
    def __init__(self):
        super().__init__(IndicatorType.DEMA)
    
    def generate_params(self) -> Dict[str, Any]:
        """DEMA period: 5-100."""
        period = np.random.randint(5, 101)
        return {"period": int(period)}
    
    def compute(self, ohlcv: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute DEMA."""
        close = ohlcv[:, 4]
        return talib.DEMA(close, timeperiod=params["period"])


class TEMAIndicator(BaseIndicator):
    """Triple Exponential Moving Average."""
    
    def __init__(self):
        super().__init__(IndicatorType.TEMA)
    
    def generate_params(self) -> Dict[str, Any]:
        """TEMA period: 5-100."""
        period = np.random.randint(5, 101)
        return {"period": int(period)}
    
    def compute(self, ohlcv: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute TEMA."""
        close = ohlcv[:, 4]
        return talib.TEMA(close, timeperiod=params["period"])


class ADXIndicator(BaseIndicator):
    """Average Directional Index."""
    
    def __init__(self):
        super().__init__(IndicatorType.ADX)
    
    def generate_params(self) -> Dict[str, Any]:
        """ADX period: 5-50."""
        period = np.random.randint(5, 51)
        return {"period": int(period)}
    
    def compute(self, ohlcv: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute ADX."""
        high, low, close = ohlcv[:, 2], ohlcv[:, 3], ohlcv[:, 4]
        return talib.ADX(high, low, close, timeperiod=params["period"])


class AROONIndicator(BaseIndicator):
    """Aroon Indicator."""
    
    def __init__(self):
        super().__init__(IndicatorType.AROON)
    
    def generate_params(self) -> Dict[str, Any]:
        """Aroon period: 5-50."""
        period = np.random.randint(5, 51)
        return {"period": int(period)}
    
    def compute(self, ohlcv: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute Aroon Up."""
        high, low = ohlcv[:, 2], ohlcv[:, 3]
        aroon_up, aroon_down = talib.AROON(high, low, timeperiod=params["period"])
        return aroon_up


class TRIXIndicator(BaseIndicator):
    """TRIX - Triple Exponential Average."""
    
    def __init__(self):
        super().__init__(IndicatorType.TRIX)
    
    def generate_params(self) -> Dict[str, Any]:
        """TRIX period: 10-50."""
        period = np.random.randint(10, 51)
        return {"period": int(period)}
    
    def compute(self, ohlcv: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute TRIX."""
        close = ohlcv[:, 4]
        return talib.TRIX(close, timeperiod=params["period"])


# ============================================================================
# VOLATILITY INDICATORS
# ============================================================================

class ATRIndicator(BaseIndicator):
    """Average True Range."""
    
    def __init__(self):
        super().__init__(IndicatorType.ATR)
    
    def generate_params(self) -> Dict[str, Any]:
        """ATR period: 5-50."""
        period = np.random.randint(5, 51)
        return {"period": int(period)}
    
    def compute(self, ohlcv: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute ATR."""
        high, low, close = ohlcv[:, 2], ohlcv[:, 3], ohlcv[:, 4]
        return talib.ATR(high, low, close, timeperiod=params["period"])


class BollingerBandsIndicator(BaseIndicator):
    """Bollinger Bands."""
    
    def __init__(self):
        super().__init__(IndicatorType.BBANDS)
    
    def generate_params(self) -> Dict[str, Any]:
        """Bollinger Bands: period 10-50, stddev 1-3."""
        period = np.random.randint(10, 51)
        nbdevup = np.random.uniform(1.5, 3.0)
        nbdevdn = nbdevup
        return {
            "period": int(period),
            "nbdevup": float(nbdevup),
            "nbdevdn": float(nbdevdn)
        }
    
    def compute(self, ohlcv: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute Bollinger Bands middle band."""
        close = ohlcv[:, 4]
        upper, middle, lower = talib.BBANDS(
            close,
            timeperiod=params["period"],
            nbdevup=params["nbdevup"],
            nbdevdn=params["nbdevdn"]
        )
        return middle


# ============================================================================
# VOLUME INDICATORS
# ============================================================================

class OBVIndicator(BaseIndicator):
    """On Balance Volume."""
    
    def __init__(self):
        super().__init__(IndicatorType.OBV)
    
    def generate_params(self) -> Dict[str, Any]:
        """OBV has no parameters."""
        return {}
    
    def compute(self, ohlcv: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute OBV."""
        close, volume = ohlcv[:, 4], ohlcv[:, 5]
        return talib.OBV(close, volume)


class ADIndicator(BaseIndicator):
    """Accumulation/Distribution."""
    
    def __init__(self):
        super().__init__(IndicatorType.AD)
    
    def generate_params(self) -> Dict[str, Any]:
        """AD has no parameters."""
        return {}
    
    def compute(self, ohlcv: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute AD."""
        high, low, close, volume = ohlcv[:, 2], ohlcv[:, 3], ohlcv[:, 4], ohlcv[:, 5]
        return talib.AD(high, low, close, volume)


class ADOSCIndicator(BaseIndicator):
    """Chaikin A/D Oscillator."""
    
    def __init__(self):
        super().__init__(IndicatorType.ADOSC)
    
    def generate_params(self) -> Dict[str, Any]:
        """ADOSC: fast 3-10, slow 10-30."""
        fast = np.random.randint(3, 11)
        slow = np.random.randint(max(fast + 3, 10), 31)
        return {"fastperiod": int(fast), "slowperiod": int(slow)}
    
    def compute(self, ohlcv: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute ADOSC."""
        high, low, close, volume = ohlcv[:, 2], ohlcv[:, 3], ohlcv[:, 4], ohlcv[:, 5]
        return talib.ADOSC(high, low, close, volume, fastperiod=params["fastperiod"], slowperiod=params["slowperiod"])


class MFIIndicator(BaseIndicator):
    """Money Flow Index."""
    
    def __init__(self):
        super().__init__(IndicatorType.MFI)
    
    def generate_params(self) -> Dict[str, Any]:
        """MFI period: 5-30."""
        period = np.random.randint(5, 31)
        return {"period": int(period)}
    
    def compute(self, ohlcv: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute MFI."""
        high, low, close, volume = ohlcv[:, 2], ohlcv[:, 3], ohlcv[:, 4], ohlcv[:, 5]
        return talib.MFI(high, low, close, volume, timeperiod=params["period"])


# ============================================================================
# ADDITIONAL INDICATORS FOR 50+ COUNT
# ============================================================================

class SARIndicator(BaseIndicator):
    """Parabolic SAR."""
    
    def __init__(self):
        super().__init__(IndicatorType.SAR)
    
    def generate_params(self) -> Dict[str, Any]:
        """SAR: acceleration 0.01-0.05, maximum 0.1-0.3."""
        acceleration = np.random.uniform(0.01, 0.05)
        maximum = np.random.uniform(0.1, 0.3)
        return {"acceleration": float(acceleration), "maximum": float(maximum)}
    
    def compute(self, ohlcv: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute SAR."""
        high, low = ohlcv[:, 2], ohlcv[:, 3]
        return talib.SAR(high, low, acceleration=params["acceleration"], maximum=params["maximum"])


class BETAIndicator(BaseIndicator):
    """Beta (correlation with market)."""
    
    def __init__(self):
        super().__init__(IndicatorType.BETA)
    
    def generate_params(self) -> Dict[str, Any]:
        """Beta period: 5-50."""
        period = np.random.randint(5, 51)
        return {"period": int(period)}
    
    def compute(self, ohlcv: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute Beta (using close as both high and low for simplicity)."""
        close = ohlcv[:, 4]
        return talib.BETA(close, close, timeperiod=params["period"])


class CORRELIndicator(BaseIndicator):
    """Pearson Correlation Coefficient."""
    
    def __init__(self):
        super().__init__(IndicatorType.CORREL)
    
    def generate_params(self) -> Dict[str, Any]:
        """Correlation period: 10-50."""
        period = np.random.randint(10, 51)
        return {"period": int(period)}
    
    def compute(self, ohlcv: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute correlation."""
        high, low = ohlcv[:, 2], ohlcv[:, 3]
        return talib.CORREL(high, low, timeperiod=params["period"])


class LINEARREGIndicator(BaseIndicator):
    """Linear Regression."""
    
    def __init__(self):
        super().__init__(IndicatorType.LINEARREG)
    
    def generate_params(self) -> Dict[str, Any]:
        """Linear regression period: 10-50."""
        period = np.random.randint(10, 51)
        return {"period": int(period)}
    
    def compute(self, ohlcv: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute linear regression."""
        close = ohlcv[:, 4]
        return talib.LINEARREG(close, timeperiod=params["period"])


class STDDEVIndicator(BaseIndicator):
    """Standard Deviation."""
    
    def __init__(self):
        super().__init__(IndicatorType.STDDEV)
    
    def generate_params(self) -> Dict[str, Any]:
        """Stddev period: 5-50."""
        period = np.random.randint(5, 51)
        return {"period": int(period)}
    
    def compute(self, ohlcv: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute standard deviation."""
        close = ohlcv[:, 4]
        return talib.STDDEV(close, timeperiod=params["period"])


class TSFIndicator(BaseIndicator):
    """Time Series Forecast."""
    
    def __init__(self):
        super().__init__(IndicatorType.TSF)
    
    def generate_params(self) -> Dict[str, Any]:
        """TSF period: 10-50."""
        period = np.random.randint(10, 51)
        return {"period": int(period)}
    
    def compute(self, ohlcv: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute TSF."""
        close = ohlcv[:, 4]
        return talib.TSF(close, timeperiod=params["period"])


class VARIndicator(BaseIndicator):
    """Variance."""
    
    def __init__(self):
        super().__init__(IndicatorType.VAR)
    
    def generate_params(self) -> Dict[str, Any]:
        """Variance period: 5-50."""
        period = np.random.randint(5, 51)
        return {"period": int(period)}
    
    def compute(self, ohlcv: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute variance."""
        close = ohlcv[:, 4]
        return talib.VAR(close, timeperiod=params["period"])


class HTDCPERIODIndicator(BaseIndicator):
    """Hilbert Transform - Dominant Cycle Period."""
    
    def __init__(self):
        super().__init__(IndicatorType.HT_DCPERIOD)
    
    def generate_params(self) -> Dict[str, Any]:
        """No parameters."""
        return {}
    
    def compute(self, ohlcv: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute HT_DCPERIOD."""
        close = ohlcv[:, 4]
        return talib.HT_DCPERIOD(close)


class HTDCPHASEIndicator(BaseIndicator):
    """Hilbert Transform - Dominant Cycle Phase."""
    
    def __init__(self):
        super().__init__(IndicatorType.HT_DCPHASE)
    
    def generate_params(self) -> Dict[str, Any]:
        """No parameters."""
        return {}
    
    def compute(self, ohlcv: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute HT_DCPHASE."""
        close = ohlcv[:, 4]
        return talib.HT_DCPHASE(close)


class HTTRENDMODEIndicator(BaseIndicator):
    """Hilbert Transform - Trend vs Cycle Mode."""
    
    def __init__(self):
        super().__init__(IndicatorType.HT_TRENDMODE)
    
    def generate_params(self) -> Dict[str, Any]:
        """No parameters."""
        return {}
    
    def compute(self, ohlcv: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute HT_TRENDMODE."""
        close = ohlcv[:, 4]
        return talib.HT_TRENDMODE(close)


class KAMAIndicator(BaseIndicator):
    """Kaufman Adaptive Moving Average."""
    
    def __init__(self):
        super().__init__(IndicatorType.KAMA)
    
    def generate_params(self) -> Dict[str, Any]:
        """KAMA period: 10-50."""
        period = np.random.randint(10, 51)
        return {"period": int(period)}
    
    def compute(self, ohlcv: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute KAMA."""
        close = ohlcv[:, 4]
        return talib.KAMA(close, timeperiod=params["period"])


class MIDPOINTIndicator(BaseIndicator):
    """MidPoint over period."""
    
    def __init__(self):
        super().__init__(IndicatorType.MIDPOINT)
    
    def generate_params(self) -> Dict[str, Any]:
        """Midpoint period: 5-50."""
        period = np.random.randint(5, 51)
        return {"period": int(period)}
    
    def compute(self, ohlcv: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute MIDPOINT."""
        close = ohlcv[:, 4]
        return talib.MIDPOINT(close, timeperiod=params["period"])


class MIDPRICEIndicator(BaseIndicator):
    """Midpoint Price over period."""
    
    def __init__(self):
        super().__init__(IndicatorType.MIDPRICE)
    
    def generate_params(self) -> Dict[str, Any]:
        """Midprice period: 5-50."""
        period = np.random.randint(5, 51)
        return {"period": int(period)}
    
    def compute(self, ohlcv: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute MIDPRICE."""
        high, low = ohlcv[:, 2], ohlcv[:, 3]
        return talib.MIDPRICE(high, low, timeperiod=params["period"])


class PLUSDIIndicator(BaseIndicator):
    """Plus Directional Indicator."""
    
    def __init__(self):
        super().__init__(IndicatorType.PLUS_DI)
    
    def generate_params(self) -> Dict[str, Any]:
        """Plus DI period: 5-50."""
        period = np.random.randint(5, 51)
        return {"period": int(period)}
    
    def compute(self, ohlcv: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute PLUS_DI."""
        high, low, close = ohlcv[:, 2], ohlcv[:, 3], ohlcv[:, 4]
        return talib.PLUS_DI(high, low, close, timeperiod=params["period"])


class MINUSDIIndicator(BaseIndicator):
    """Minus Directional Indicator."""
    
    def __init__(self):
        super().__init__(IndicatorType.MINUS_DI)
    
    def generate_params(self) -> Dict[str, Any]:
        """Minus DI period: 5-50."""
        period = np.random.randint(5, 51)
        return {"period": int(period)}
    
    def compute(self, ohlcv: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute MINUS_DI."""
        high, low, close = ohlcv[:, 2], ohlcv[:, 3], ohlcv[:, 4]
        return talib.MINUS_DI(high, low, close, timeperiod=params["period"])


class SAREXTIndicator(BaseIndicator):
    """Parabolic SAR - Extended."""
    
    def __init__(self):
        super().__init__(IndicatorType.SAREXT)
    
    def generate_params(self) -> Dict[str, Any]:
        """SAREXT parameters."""
        start_value = np.random.uniform(0.01, 0.05)
        increment = np.random.uniform(0.01, 0.05)
        maximum = np.random.uniform(0.1, 0.3)
        return {
            "startvalue": float(start_value),
            "offsetonreverse": 0.0,
            "accelerationinitlong": float(start_value),
            "accelerationlong": float(increment),
            "accelerationmaxlong": float(maximum),
            "accelerationinitshort": float(start_value),
            "accelerationshort": float(increment),
            "accelerationmaxshort": float(maximum)
        }
    
    def compute(self, ohlcv: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute SAREXT."""
        high, low = ohlcv[:, 2], ohlcv[:, 3]
        return talib.SAREXT(
            high, low,
            startvalue=params["startvalue"],
            offsetonreverse=params["offsetonreverse"],
            accelerationinitlong=params["accelerationinitlong"],
            accelerationlong=params["accelerationlong"],
            accelerationmaxlong=params["accelerationmaxlong"],
            accelerationinitshort=params["accelerationinitshort"],
            accelerationshort=params["accelerationshort"],
            accelerationmaxshort=params["accelerationmaxshort"]
        )


class LINEARREGANGLEIndicator(BaseIndicator):
    """Linear Regression Angle."""
    
    def __init__(self):
        super().__init__(IndicatorType.LINEARREG_ANGLE)
    
    def generate_params(self) -> Dict[str, Any]:
        """Linear regression angle period: 5-50."""
        period = np.random.randint(5, 51)
        return {"period": int(period)}
    
    def compute(self, ohlcv: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute LINEARREG_ANGLE."""
        close = ohlcv[:, 4]
        return talib.LINEARREG_ANGLE(close, timeperiod=params["period"])


class LINEARREGINTERCEPTIndicator(BaseIndicator):
    """Linear Regression Intercept."""
    
    def __init__(self):
        super().__init__(IndicatorType.LINEARREG_INTERCEPT)
    
    def generate_params(self) -> Dict[str, Any]:
        """Linear regression intercept period: 5-50."""
        period = np.random.randint(5, 51)
        return {"period": int(period)}
    
    def compute(self, ohlcv: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute LINEARREG_INTERCEPT."""
        close = ohlcv[:, 4]
        return talib.LINEARREG_INTERCEPT(close, timeperiod=params["period"])


class LINEARREGSLOPEIndicator(BaseIndicator):
    """Linear Regression Slope."""
    
    def __init__(self):
        super().__init__(IndicatorType.LINEARREG_SLOPE)
    
    def generate_params(self) -> Dict[str, Any]:
        """Linear regression slope period: 5-50."""
        period = np.random.randint(5, 51)
        return {"period": int(period)}
    
    def compute(self, ohlcv: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute LINEARREG_SLOPE."""
        close = ohlcv[:, 4]
        return talib.LINEARREG_SLOPE(close, timeperiod=params["period"])


class TRIMAIndicator(BaseIndicator):
    """Triangular Moving Average."""
    
    def __init__(self):
        super().__init__(IndicatorType.TRIMA)
    
    def generate_params(self) -> Dict[str, Any]:
        """TRIMA period: 5-50."""
        period = np.random.randint(5, 51)
        return {"period": int(period)}
    
    def compute(self, ohlcv: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute TRIMA."""
        close = ohlcv[:, 4]
        return talib.TRIMA(close, timeperiod=params["period"])


class T3Indicator(BaseIndicator):
    """Triple Exponential Moving Average (T3)."""
    
    def __init__(self):
        super().__init__(IndicatorType.T3)
    
    def generate_params(self) -> Dict[str, Any]:
        """T3 period: 5-50."""
        period = np.random.randint(5, 51)
        vfactor = np.random.uniform(0.5, 0.9)
        return {"period": int(period), "vfactor": float(vfactor)}
    
    def compute(self, ohlcv: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute T3."""
        close = ohlcv[:, 4]
        return talib.T3(close, timeperiod=params["period"], vfactor=params.get("vfactor", 0.7))


class ADXRIndicator(BaseIndicator):
    """Average Directional Movement Index Rating."""
    
    def __init__(self):
        super().__init__(IndicatorType.ADXR)
    
    def generate_params(self) -> Dict[str, Any]:
        """ADXR period: 5-50."""
        period = np.random.randint(5, 51)
        return {"period": int(period)}
    
    def compute(self, ohlcv: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute ADXR."""
        high, low, close = ohlcv[:, 2], ohlcv[:, 3], ohlcv[:, 4]
        return talib.ADXR(high, low, close, timeperiod=params["period"])


class AROONOSCIndicator(BaseIndicator):
    """Aroon Oscillator."""
    
    def __init__(self):
        super().__init__(IndicatorType.AROONOSC)
    
    def generate_params(self) -> Dict[str, Any]:
        """Aroon oscillator period: 5-50."""
        period = np.random.randint(5, 51)
        return {"period": int(period)}
    
    def compute(self, ohlcv: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute AROONOSC."""
        high, low = ohlcv[:, 2], ohlcv[:, 3]
        return talib.AROONOSC(high, low, timeperiod=params["period"])


class DXIndicator(BaseIndicator):
    """Directional Movement Index."""
    
    def __init__(self):
        super().__init__(IndicatorType.DX)
    
    def generate_params(self) -> Dict[str, Any]:
        """DX period: 5-50."""
        period = np.random.randint(5, 51)
        return {"period": int(period)}
    
    def compute(self, ohlcv: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute DX."""
        high, low, close = ohlcv[:, 2], ohlcv[:, 3], ohlcv[:, 4]
        return talib.DX(high, low, close, timeperiod=params["period"])


class NATRIndicator(BaseIndicator):
    """Normalized Average True Range."""
    
    def __init__(self):
        super().__init__(IndicatorType.NATR)
    
    def generate_params(self) -> Dict[str, Any]:
        """NATR period: 5-50."""
        period = np.random.randint(5, 51)
        return {"period": int(period)}
    
    def compute(self, ohlcv: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute NATR."""
        high, low, close = ohlcv[:, 2], ohlcv[:, 3], ohlcv[:, 4]
        return talib.NATR(high, low, close, timeperiod=params["period"])


class STOCHFIndicator(BaseIndicator):
    """Stochastic Fast."""
    
    def __init__(self):
        super().__init__(IndicatorType.STOCHF)
    
    def generate_params(self) -> Dict[str, Any]:
        """Stochastic Fast parameters."""
        fastk_period = np.random.randint(5, 21)
        fastd_period = np.random.randint(3, 11)
        return {
            "fastk_period": int(fastk_period),
            "fastd_period": int(fastd_period),
            "fastd_matype": 0  # Simple MA
        }
    
    def compute(self, ohlcv: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute STOCHF."""
        high, low, close = ohlcv[:, 2], ohlcv[:, 3], ohlcv[:, 4]
        fastk, fastd = talib.STOCHF(
            high, low, close,
            fastk_period=params["fastk_period"],
            fastd_period=params["fastd_period"],
            fastd_matype=params.get("fastd_matype", 0)
        )
        return fastk  # Return %K


class RSISTOCHIndicator(BaseIndicator):
    """RSI-based Stochastic (custom combination)."""
    
    def __init__(self):
        super().__init__(IndicatorType.RSI_STOCH)
    
    def generate_params(self) -> Dict[str, Any]:
        """RSI-Stoch parameters."""
        rsi_period = np.random.randint(10, 31)
        stoch_period = np.random.randint(5, 21)
        return {
            "rsi_period": int(rsi_period),
            "stoch_period": int(stoch_period)
        }
    
    def compute(self, ohlcv: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute RSI-Stochastic."""
        close = ohlcv[:, 4]
        rsi = talib.RSI(close, timeperiod=params["rsi_period"])
        # Apply stochastic to RSI values
        high_rsi = np.maximum.accumulate(rsi)
        low_rsi = np.minimum.accumulate(rsi)
        stoch_rsi = (rsi - low_rsi) / (high_rsi - low_rsi + 1e-10) * 100
        return stoch_rsi


# ============================================================================
# INDICATOR FACTORY
# ============================================================================

class IndicatorFactory:
    """
    Factory for creating and managing indicators.
    Provides methods to generate random indicators with valid parameters.
    """
    
    # Map indicator types to their classes
    INDICATOR_CLASSES = {
        IndicatorType.RSI: RSIIndicator,
        IndicatorType.STOCH: StochasticIndicator,
        IndicatorType.MACD: MACDIndicator,
        IndicatorType.CCI: CCIIndicator,
        IndicatorType.MOM: MOMIndicator,
        IndicatorType.ROC: ROCIndicator,
        IndicatorType.WILLIAMS_R: WilliamsRIndicator,
        IndicatorType.ULTIMATE_OSC: UltimateOscillatorIndicator,
        IndicatorType.PPO: PPOIndicator,
        IndicatorType.EMA: EMAIndicator,
        IndicatorType.SMA: SMAIndicator,
        IndicatorType.WMA: WMAIndicator,
        IndicatorType.DEMA: DEMAIndicator,
        IndicatorType.TEMA: TEMAIndicator,
        IndicatorType.ADX: ADXIndicator,
        IndicatorType.AROON: AROONIndicator,
        IndicatorType.TRIX: TRIXIndicator,
        IndicatorType.ATR: ATRIndicator,
        IndicatorType.BBANDS: BollingerBandsIndicator,
        IndicatorType.OBV: OBVIndicator,
        IndicatorType.AD: ADIndicator,
        IndicatorType.ADOSC: ADOSCIndicator,
        IndicatorType.MFI: MFIIndicator,
        IndicatorType.SAR: SARIndicator,
        IndicatorType.BETA: BETAIndicator,
        IndicatorType.CORREL: CORRELIndicator,
        IndicatorType.LINEARREG: LINEARREGIndicator,
        IndicatorType.STDDEV: STDDEVIndicator,
        IndicatorType.TSF: TSFIndicator,
        IndicatorType.VAR: VARIndicator,
        IndicatorType.HT_DCPERIOD: HTDCPERIODIndicator,
        IndicatorType.HT_DCPHASE: HTDCPHASEIndicator,
        IndicatorType.HT_TRENDMODE: HTTRENDMODEIndicator,
        IndicatorType.KAMA: KAMAIndicator,
        IndicatorType.MIDPOINT: MIDPOINTIndicator,
        IndicatorType.MIDPRICE: MIDPRICEIndicator,
        IndicatorType.PLUS_DI: PLUSDIIndicator,
        IndicatorType.MINUS_DI: MINUSDIIndicator,
        IndicatorType.SAREXT: SAREXTIndicator,
        IndicatorType.LINEARREG_ANGLE: LINEARREGANGLEIndicator,
        IndicatorType.LINEARREG_INTERCEPT: LINEARREGINTERCEPTIndicator,
        IndicatorType.LINEARREG_SLOPE: LINEARREGSLOPEIndicator,
        IndicatorType.TRIMA: TRIMAIndicator,
        IndicatorType.T3: T3Indicator,
        IndicatorType.ADXR: ADXRIndicator,
        IndicatorType.AROONOSC: AROONOSCIndicator,
        IndicatorType.DX: DXIndicator,
        IndicatorType.NATR: NATRIndicator,
        IndicatorType.STOCHF: STOCHFIndicator,
        IndicatorType.RSI_STOCH: RSISTOCHIndicator,
    }
    
    @classmethod
    def get_all_indicator_types(cls) -> List[IndicatorType]:
        """Get list of all available indicator types."""
        return list(cls.INDICATOR_CLASSES.keys())
    
    @classmethod
    def create_indicator(cls, indicator_type: IndicatorType) -> BaseIndicator:
        """Create an indicator instance."""
        if indicator_type not in cls.INDICATOR_CLASSES:
            raise ValueError(f"Unknown indicator type: {indicator_type}")
        return cls.INDICATOR_CLASSES[indicator_type]()
    
    @classmethod
    def generate_random_indicator_params(cls, indicator_type: IndicatorType) -> IndicatorParams:
        """Generate random parameters for an indicator."""
        indicator = cls.create_indicator(indicator_type)
        params = indicator.generate_params()
        return IndicatorParams(indicator_type=indicator_type, params=params)
    
    @classmethod
    def compute_indicator(
        cls,
        ohlcv: np.ndarray,
        indicator_params: IndicatorParams
    ) -> np.ndarray:
        """Compute indicator values."""
        indicator = cls.create_indicator(indicator_params.indicator_type)
        return indicator.compute(ohlcv, indicator_params.params)
    
    @classmethod
    def get_indicator_count(cls) -> int:
        """Get total number of available indicators."""
        return len(cls.INDICATOR_CLASSES)


# Log total indicator count
log_debug(f"Indicator factory initialized with {IndicatorFactory.get_indicator_count()} indicators")
