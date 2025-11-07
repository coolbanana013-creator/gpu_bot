"""
Unit tests for GPU backtester.
Tests kernel compilation, backtest execution, and result parsing.
"""
import pytest
import numpy as np
import pyopencl as cl
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtester.simulator import GPUBacktester
from src.bot_generator.generator import BotGenerator, BotConfig
from src.indicators.factory import IndicatorParams, IndicatorType
from src.risk_management.strategies import RiskStrategyParams, RiskStrategyType


@pytest.fixture(scope="module")
def gpu_context():
    """Create GPU context for tests."""
    platforms = cl.get_platforms()
    if not platforms:
        pytest.skip("No OpenCL platforms available")
    
    gpu_device = None
    for platform in platforms:
        try:
            devices = platform.get_devices(device_type=cl.device_type.GPU)
            if devices:
                gpu_device = devices[0]
                break
        except cl.RuntimeError:
            continue
    
    if gpu_device is None:
        pytest.skip("No GPU device available")
    
    context = cl.Context([gpu_device])
    queue = cl.CommandQueue(context)
    
    return context, queue


@pytest.fixture
def sample_ohlcv():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    num_bars = 1000
    
    # Generate realistic price movement
    base_price = 50000.0
    returns = np.random.randn(num_bars) * 0.02  # 2% volatility
    prices = base_price * np.exp(np.cumsum(returns))
    
    ohlcv = np.zeros((num_bars, 5), dtype=np.float32)
    
    for i in range(num_bars):
        price = prices[i]
        high_offset = abs(np.random.randn()) * 0.01
        low_offset = abs(np.random.randn()) * 0.01
        
        ohlcv[i, 0] = price  # open
        ohlcv[i, 1] = price * (1 + high_offset)  # high
        ohlcv[i, 2] = price * (1 - low_offset)  # low
        ohlcv[i, 3] = price * (1 + np.random.randn() * 0.005)  # close
        ohlcv[i, 4] = np.random.uniform(100, 1000)  # volume
    
    return ohlcv


def test_backtester_initialization(gpu_context):
    """Test GPUBacktester initializes correctly."""
    context, queue = gpu_context
    
    backtester = GPUBacktester(
        gpu_context=context,
        gpu_queue=queue,
        initial_balance=10000.0
    )
    
    assert backtester.ctx is not None
    assert backtester.queue is not None
    assert backtester.initial_balance == 10000.0


def test_kernel_compilation(gpu_context):
    """Test backtest kernels compile successfully."""
    context, queue = gpu_context
    
    # This will compile kernels in constructor
    try:
        backtester = GPUBacktester(
            gpu_context=context,
            gpu_queue=queue,
            initial_balance=10000.0
        )
        assert backtester.backtest_program is not None
        assert backtester.precompute_program is not None
    except Exception as e:
        pytest.fail(f"Kernel compilation failed: {e}")


def test_backtest_execution(gpu_context, sample_ohlcv):
    """Test backtesting runs without errors."""
    context, queue = gpu_context
    
    # Generate test bots
    generator = BotGenerator(
        population_size=10,
        min_indicators=3,
        max_indicators=5,
        min_risk_strategies=2,
        max_risk_strategies=3,
        leverage=10,
        random_seed=42,
        gpu_context=context,
        gpu_queue=queue
    )
    bots = generator.generate_population()
    
    # Create backtester
    backtester = GPUBacktester(
        gpu_context=context,
        gpu_queue=queue,
        initial_balance=10000.0
    )
    
    # Run backtest
    cycle_starts = np.array([0, 500], dtype=np.int32)
    cycle_ends = np.array([499, 999], dtype=np.int32)
    
    results = backtester.backtest_bots(
        bots=bots,
        ohlcv_data=sample_ohlcv,
        cycle_starts=cycle_starts,
        cycle_ends=cycle_ends
    )
    
    # Check results
    assert len(results) == 10
    
    for result in results:
        assert hasattr(result, 'bot_id')
        assert hasattr(result, 'total_trades')
        assert hasattr(result, 'winning_trades')
        assert hasattr(result, 'losing_trades')
        assert hasattr(result, 'total_return_pct')
        assert hasattr(result, 'sharpe_ratio')
        assert hasattr(result, 'max_drawdown_pct')
        assert hasattr(result, 'final_balance')
        assert hasattr(result, 'win_rate')
        
        # Check values are reasonable
        assert result.total_trades >= 0
        assert result.winning_trades >= 0
        assert result.losing_trades >= 0
        assert result.final_balance > 0  # Should never go to zero in simulation


def test_precompute_indicators(gpu_context, sample_ohlcv):
    """Test indicator precomputation works."""
    context, queue = gpu_context
    
    backtester = GPUBacktester(
        gpu_context=context,
        gpu_queue=queue,
        initial_balance=10000.0
    )
    
    # Test with a few indicator types
    indicator_types = np.array([0, 1, 2, 4, 5], dtype=np.int32)  # RSI, MACD, STOCH, EMA, SMA
    
    try:
        precomputed = backtester._precompute_indicators(
            ohlcv_data=sample_ohlcv,
            indicator_types=indicator_types,
            num_indicator_types=5
        )
        
        # Check output shape
        assert precomputed.shape == (len(sample_ohlcv), 5, 10)
        
        # Check for NaN/Inf (should have validation)
        assert not np.any(np.isnan(precomputed))
        assert not np.any(np.isinf(precomputed))
        
    except Exception as e:
        pytest.fail(f"Indicator precomputation failed: {e}")


def test_result_parsing(gpu_context, sample_ohlcv):
    """Test BacktestResult parsing is correct."""
    context, queue = gpu_context
    
    generator = BotGenerator(
        population_size=5,
        min_indicators=3,
        max_indicators=4,
        min_risk_strategies=2,
        max_risk_strategies=2,
        leverage=5,
        random_seed=123,
        gpu_context=context,
        gpu_queue=queue
    )
    bots = generator.generate_population()
    
    backtester = GPUBacktester(
        gpu_context=context,
        gpu_queue=queue,
        initial_balance=5000.0
    )
    
    cycle_starts = np.array([0], dtype=np.int32)
    cycle_ends = np.array([len(sample_ohlcv) - 1], dtype=np.int32)
    
    results = backtester.backtest_bots(
        bots=bots,
        ohlcv_data=sample_ohlcv,
        cycle_starts=cycle_starts,
        cycle_ends=cycle_ends
    )
    
    # Verify result structure
    for i, result in enumerate(results):
        # Bot IDs should match
        assert result.bot_id == bots[i].bot_id
        
        # Win rate should be consistent
        if result.total_trades > 0:
            calculated_winrate = (result.winning_trades / result.total_trades) * 100
            assert abs(result.win_rate - calculated_winrate) < 0.01
        
        # Winning + losing should equal total
        assert result.winning_trades + result.losing_trades <= result.total_trades


def test_vram_validation(gpu_context, sample_ohlcv):
    """Test VRAM estimation runs before backtest."""
    context, queue = gpu_context
    
    # This should work with reasonable bot count
    generator = BotGenerator(
        population_size=100,
        min_indicators=5,
        max_indicators=5,
        min_risk_strategies=3,
        max_risk_strategies=3,
        leverage=10,
        random_seed=42,
        gpu_context=context,
        gpu_queue=queue
    )
    bots = generator.generate_population()
    
    backtester = GPUBacktester(
        gpu_context=context,
        gpu_queue=queue,
        initial_balance=10000.0
    )
    
    cycle_starts = np.array([0], dtype=np.int32)
    cycle_ends = np.array([len(sample_ohlcv) - 1], dtype=np.int32)
    
    # Should not raise VRAM error for 100 bots
    try:
        results = backtester.backtest_bots(
            bots=bots,
            ohlcv_data=sample_ohlcv,
            cycle_starts=cycle_starts,
            cycle_ends=cycle_ends
        )
        assert len(results) == 100
    except RuntimeError as e:
        if "VRAM" in str(e):
            pytest.skip(f"Insufficient VRAM for test: {e}")
        else:
            raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
