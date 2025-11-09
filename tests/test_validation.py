"""
Unit tests for validation utilities.
"""
import pytest
from src.utils.validation import (
    validate_int, validate_float, validate_enum, validate_string,
    validate_range, validate_population_size, validate_timeframe,
    validate_leverage, validate_percentage, validate_pair
)


class TestValidateInt:
    """Tests for validate_int function."""
    
    def test_valid_int(self):
        """Test valid integer."""
        assert validate_int(42, "test") == 42
    
    def test_valid_int_with_range(self):
        """Test valid integer within range."""
        assert validate_int(50, "test", min_val=0, max_val=100) == 50
    
    def test_invalid_type(self):
        """Test invalid type raises ValueError."""
        with pytest.raises(ValueError, match="must be an integer"):
            validate_int("not an int", "test")
    
    def test_bool_rejected(self):
        """Test that booleans are rejected (they're technically ints in Python)."""
        with pytest.raises(ValueError, match="must be an integer"):
            validate_int(True, "test")
    
    def test_below_minimum(self):
        """Test value below minimum raises ValueError."""
        with pytest.raises(ValueError, match="must be >= 10"):
            validate_int(5, "test", min_val=10)
    
    def test_above_maximum(self):
        """Test value above maximum raises ValueError."""
        with pytest.raises(ValueError, match="must be <= 100"):
            validate_int(150, "test", max_val=100)
    
    def test_none_not_allowed(self):
        """Test None raises ValueError by default."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_int(None, "test")
    
    def test_none_allowed(self):
        """Test None is accepted when allow_none=True."""
        assert validate_int(None, "test", allow_none=True) is None


class TestValidateFloat:
    """Tests for validate_float function."""
    
    def test_valid_float(self):
        """Test valid float."""
        assert validate_float(3.14, "test") == 3.14
    
    def test_int_converted_to_float(self):
        """Test integer is converted to float."""
        result = validate_float(42, "test")
        assert result == 42.0
        assert isinstance(result, float)
    
    def test_strict_positive(self):
        """Test strict_positive rejects zero and negatives."""
        with pytest.raises(ValueError, match="must be > 0"):
            validate_float(0.0, "test", strict_positive=True)
        
        with pytest.raises(ValueError, match="must be > 0"):
            validate_float(-1.0, "test", strict_positive=True)
    
    def test_range_validation(self):
        """Test range validation."""
        assert validate_float(5.5, "test", min_val=0.0, max_val=10.0) == 5.5
        
        with pytest.raises(ValueError, match="must be >= 1.0"):
            validate_float(0.5, "test", min_val=1.0)


class TestValidateEnum:
    """Tests for validate_enum function."""
    
    def test_valid_enum_value(self):
        """Test valid enum value."""
        allowed = ['a', 'b', 'c']
        assert validate_enum('b', "test", allowed) == 'b'
    
    def test_invalid_enum_value(self):
        """Test invalid enum value raises ValueError."""
        allowed = ['a', 'b', 'c']
        with pytest.raises(ValueError, match="must be one of"):
            validate_enum('d', "test", allowed)
    
    def test_case_insensitive(self):
        """Test case insensitive matching."""
        allowed = ['ABC', 'DEF']
        result = validate_enum('abc', "test", allowed, case_sensitive=False)
        assert result == 'abc'


class TestValidateString:
    """Tests for validate_string function."""
    
    def test_valid_string(self):
        """Test valid string."""
        assert validate_string("hello", "test") == "hello"
    
    def test_empty_not_allowed(self):
        """Test empty string raises ValueError by default."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_string("", "test")
    
    def test_empty_allowed(self):
        """Test empty string accepted when allow_empty=True."""
        assert validate_string("", "test", allow_empty=True) == ""
    
    def test_length_constraints(self):
        """Test length constraints."""
        assert validate_string("hello", "test", min_length=3, max_length=10) == "hello"
        
        with pytest.raises(ValueError, match="at least 10 characters"):
            validate_string("short", "test", min_length=10)


class TestValidateRange:
    """Tests for validate_range function."""
    
    def test_valid_range(self):
        """Test valid range."""
        min_val, max_val = validate_range(1, 10, "test")
        assert min_val == 1
        assert max_val == 10
    
    def test_invalid_range(self):
        """Test invalid range (min > max) raises ValueError."""
        with pytest.raises(ValueError, match="cannot be greater than"):
            validate_range(10, 5, "test")


class TestValidatePopulationSize:
    """Tests for validate_population_size function."""
    
    def test_valid_population(self):
        """Test valid population size."""
        assert validate_population_size(10000) == 10000
    
    def test_too_small(self):
        """Test population too small."""
        with pytest.raises(ValueError, match="must be >= 1000"):
            validate_population_size(500)
    
    def test_too_large(self):
        """Test population too large."""
        with pytest.raises(ValueError, match="must be <= 1000000"):
            validate_population_size(2000000)


class TestValidateTimeframe:
    """Tests for validate_timeframe function."""
    
    def test_valid_timeframes(self):
        """Test all valid timeframes."""
        valid = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
        for tf in valid:
            assert validate_timeframe(tf) == tf.lower()
    
    def test_invalid_timeframe(self):
        """Test invalid timeframe raises ValueError."""
        with pytest.raises(ValueError, match="must be one of"):
            validate_timeframe('2h')


class TestValidateLeverage:
    """Tests for validate_leverage function."""
    
    def test_valid_leverage(self):
        """Test valid leverage values."""
        assert validate_leverage(1) == 1
        assert validate_leverage(10) == 10
        assert validate_leverage(25) == 25  # Updated from 125 to 25 (safer limit)
    
    def test_invalid_leverage(self):
        """Test invalid leverage values."""
        with pytest.raises(ValueError, match="must be >= 1"):
            validate_leverage(0)
        
        with pytest.raises(ValueError, match="must be <= 25"):  # Updated from 125 to 25
            validate_leverage(200)


class TestValidatePercentage:
    """Tests for validate_percentage function."""
    
    def test_valid_percentage(self):
        """Test valid percentage."""
        assert validate_percentage(50.0, "test") == 50.0
    
    def test_out_of_range(self):
        """Test out of range percentage."""
        with pytest.raises(ValueError, match="must be >= 0"):
            validate_percentage(-10.0, "test")
        
        with pytest.raises(ValueError, match="must be <= 100"):
            validate_percentage(150.0, "test")
    
    def test_custom_range(self):
        """Test custom range."""
        assert validate_percentage(5.0, "test", min_val=1.0, max_val=10.0) == 5.0


class TestValidatePair:
    """Tests for validate_pair function."""
    
    def test_valid_pair(self):
        """Test valid trading pair."""
        assert validate_pair("BTC/USDT") == "BTC/USDT"
    
    def test_lowercase_normalized(self):
        """Test lowercase is normalized to uppercase."""
        assert validate_pair("btc/usdt") == "BTC/USDT"
    
    def test_missing_separator(self):
        """Test pair without separator - currently accepted."""
        # Note: validate_pair currently accepts pairs without '/' separator
        result = validate_pair("BTCUSDT")
        assert result == "BTCUSDT"
    
    def test_invalid_format(self):
        """Test invalid format raises ValueError."""
        with pytest.raises(ValueError):
            validate_pair("BTC/USDT/EXTRA")
    
    def test_empty_parts(self):
        """Test empty base or quote raises ValueError."""
        with pytest.raises(ValueError):
            validate_pair("/USDT")
        
        with pytest.raises(ValueError):
            validate_pair("BTC/")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
