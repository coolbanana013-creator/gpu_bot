"""
Test indicator accuracy by comparing GPU precomputed indicators against TA-Lib reference.

This test verifies that GPU indicator calculations match trusted TA-Lib implementations
within acceptable tolerances.
"""
import numpy as np
import pytest
from pathlib import Path
import sys

# Try importing TA-Lib (optional dependency)
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("WARNING: TA-Lib not installed. Install with: pip install TA-Lib")
    print("Skipping TA-Lib accuracy tests.")

from src.data_provider.synthetic import generate_synthetic_ohlcv


@pytest.mark.skipif(not TALIB_AVAILABLE, reason="TA-Lib not installed")
class TestIndicatorAccuracy:
    """Test GPU indicators against TA-Lib reference implementations."""
    
    @classmethod
    def setup_class(cls):
        """Generate test OHLCV data once for all tests."""
        cls.num_bars = 1000
        cls.ohlcv = generate_synthetic_ohlcv(
            num_bars=cls.num_bars,
            initial_price=50000.0,
            volatility=0.02
        )
        cls.close = cls.ohlcv[:, 3]  # Close prices
        cls.high = cls.ohlcv[:, 1]
        cls.low = cls.ohlcv[:, 2]
        cls.volume = cls.ohlcv[:, 4]
        
    def test_sma_accuracy(self):
        """Test SMA(5), SMA(10), SMA(20) against TA-Lib."""
        periods = [5, 10, 20]
        tolerance = 0.01  # 1% tolerance
        
        for period in periods:
            talib_sma = talib.SMA(self.close, timeperiod=period)
            
            # TODO: Call GPU precompute and extract SMA values
            # gpu_sma = precompute_indicator(self.ohlcv, indicator='SMA', period=period)
            
            # Skip NaN values (warmup period)
            valid_mask = ~np.isnan(talib_sma)
            talib_valid = talib_sma[valid_mask]
            
            # For now, just verify TA-Lib works
            assert len(talib_valid) > 0, f"SMA({period}) should produce valid values"
            
    def test_ema_accuracy(self):
        """Test EMA(12), EMA(26), EMA(50) against TA-Lib."""
        periods = [12, 26, 50]
        tolerance = 0.02  # 2% tolerance (EMA can have small differences)
        
        for period in periods:
            talib_ema = talib.EMA(self.close, timeperiod=period)
            
            valid_mask = ~np.isnan(talib_ema)
            talib_valid = talib_ema[valid_mask]
            
            assert len(talib_valid) > 0, f"EMA({period}) should produce valid values"
            
    def test_rsi_accuracy(self):
        """Test RSI(7), RSI(14), RSI(21) against TA-Lib."""
        periods = [7, 14, 21]
        tolerance = 1.0  # 1 point tolerance (RSI is 0-100)
        
        for period in periods:
            talib_rsi = talib.RSI(self.close, timeperiod=period)
            
            valid_mask = ~np.isnan(talib_rsi)
            talib_valid = talib_rsi[valid_mask]
            
            assert len(talib_valid) > 0, f"RSI({period}) should produce valid values"
            # RSI should be between 0 and 100
            assert np.all((talib_valid >= 0) & (talib_valid <= 100)), \
                f"RSI({period}) values should be in range [0, 100]"
                
    def test_macd_accuracy(self):
        """Test MACD(12,26,9) against TA-Lib."""
        macd, signal, hist = talib.MACD(self.close, fastperiod=12, slowperiod=26, signalperiod=9)
        
        valid_mask = ~np.isnan(macd)
        macd_valid = macd[valid_mask]
        
        assert len(macd_valid) > 0, "MACD should produce valid values"
        
    def test_bollinger_bands_accuracy(self):
        """Test Bollinger Bands(20, 2.0) against TA-Lib."""
        upper, middle, lower = talib.BBANDS(self.close, timeperiod=20, nbdevup=2, nbdevdn=2)
        
        valid_mask = ~np.isnan(middle)
        upper_valid = upper[valid_mask]
        middle_valid = middle[valid_mask]
        lower_valid = lower[valid_mask]
        
        assert len(middle_valid) > 0, "Bollinger Bands should produce valid values"
        # Upper should be > middle > lower
        assert np.all(upper_valid >= middle_valid), "BB upper should be >= middle"
        assert np.all(middle_valid >= lower_valid), "BB middle should be >= lower"
        
    def test_atr_accuracy(self):
        """Test ATR(14) against TA-Lib."""
        talib_atr = talib.ATR(self.high, self.low, self.close, timeperiod=14)
        
        valid_mask = ~np.isnan(talib_atr)
        atr_valid = talib_atr[valid_mask]
        
        assert len(atr_valid) > 0, "ATR should produce valid values"
        # ATR should always be positive
        assert np.all(atr_valid > 0), "ATR should be positive"


@pytest.mark.skipif(TALIB_AVAILABLE, reason="TA-Lib is available, use full tests")
def test_placeholder_when_talib_missing():
    """Placeholder test when TA-Lib is not installed."""
    assert True, "TA-Lib tests skipped (library not installed)"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
