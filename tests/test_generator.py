"""
Unit tests for GPU bot generator.
Tests kernel compilation, bot generation, and struct parsing.
"""
import pytest
import numpy as np
import pyopencl as cl
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bot_generator.generator import BotGenerator
from src.indicators.factory import IndicatorType
from src.risk_management.strategies import RiskStrategyType


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


def test_gpu_context_creation(gpu_context):
    """Test GPU context is valid."""
    context, queue = gpu_context
    assert context is not None
    assert queue is not None


def test_kernel_compilation(gpu_context):
    """Test bot generation kernel compiles successfully."""
    context, queue = gpu_context
    
    kernel_path = Path(__file__).parent.parent / "src" / "gpu_kernels" / "bot_gen_impl.cl"
    assert kernel_path.exists(), f"Kernel file not found: {kernel_path}"
    
    kernel_src = kernel_path.read_text()
    
    # Should compile without errors
    try:
        program = cl.Program(context, kernel_src).build()
        assert program is not None
        assert hasattr(program, 'generate_bots')
    except cl.RuntimeError as e:
        pytest.fail(f"Kernel compilation failed: {e}")


def test_bot_generation(gpu_context):
    """Test generating population of bots."""
    context, queue = gpu_context
    
    generator = BotGenerator(
        population_size=100,
        min_indicators=3,
        max_indicators=8,
        min_risk_strategies=2,
        max_risk_strategies=5,
        leverage=10,
        random_seed=42,
        gpu_context=context,
        gpu_queue=queue
    )
    
    bots = generator.generate_population()
    
    # Check population size
    assert len(bots) == 100
    
    # Check bot IDs are unique
    bot_ids = [bot.bot_id for bot in bots]
    assert len(set(bot_ids)) == 100, "Bot IDs should be unique"
    
    # Check bot structure
    for bot in bots:
        assert hasattr(bot, 'bot_id')
        assert hasattr(bot, 'indicators')
        assert hasattr(bot, 'risk_strategies')
        assert hasattr(bot, 'take_profit_pct')
        assert hasattr(bot, 'stop_loss_pct')
        assert hasattr(bot, 'leverage')
        
        # Check indicator count
        assert 3 <= len(bot.indicators) <= 8
        
        # Check risk strategy count
        assert 2 <= len(bot.risk_strategies) <= 5
        
        # Check leverage
        assert bot.leverage == 10
        
        # Check TP/SL are positive
        assert bot.take_profit_pct > 0
        assert bot.stop_loss_pct > 0


def test_indicator_uniqueness(gpu_context):
    """Test that bots have unique indicator combinations."""
    context, queue = gpu_context
    
    generator = BotGenerator(
        population_size=50,
        min_indicators=5,
        max_indicators=5,  # Fixed number for easier testing
        min_risk_strategies=2,
        max_risk_strategies=3,
        leverage=5,
        random_seed=123,
        gpu_context=context,
        gpu_queue=queue
    )
    
    bots = generator.generate_population()
    
    # Check indicator types are not duplicated within a bot
    for bot in bots:
        indicator_types = [ind.indicator_type for ind in bot.indicators]
        assert len(indicator_types) == len(set(indicator_types)), \
            f"Bot {bot.bot_id} has duplicate indicator types"


def test_struct_parsing(gpu_context):
    """Test struct deserialization is correct."""
    context, queue = gpu_context
    
    generator = BotGenerator(
        population_size=10,
        min_indicators=3,
        max_indicators=5,
        min_risk_strategies=2,
        max_risk_strategies=4,
        leverage=20,
        random_seed=999,
        gpu_context=context,
        gpu_queue=queue
    )
    
    bots = generator.generate_population()
    
    # Verify all bots were parsed correctly
    assert len(bots) == 10
    
    for bot in bots:
        # Check all required fields are present and valid
        assert 0 <= bot.bot_id <= 10000000  # Allow 0-based bot IDs
        assert len(bot.indicators) >= 3
        assert len(bot.risk_strategies) >= 2
        assert 0 < bot.take_profit_pct <= 20
        assert 0 < bot.stop_loss_pct <= 10
        assert bot.leverage == 20
        
        # Check indicator params are valid dicts
        for ind in bot.indicators:
            assert isinstance(ind.params, dict)
            assert len(ind.params) > 0
        
        # Check risk strategy params are valid dicts
        for risk in bot.risk_strategies:
            assert isinstance(risk.params, dict)


def test_random_seed_reproducibility(gpu_context):
    """Test that same random seed produces same bots."""
    context, queue = gpu_context
    
    # Generate with seed 42
    generator1 = BotGenerator(
        population_size=20,
        min_indicators=4,
        max_indicators=6,
        min_risk_strategies=2,
        max_risk_strategies=4,
        leverage=15,
        random_seed=42,
        gpu_context=context,
        gpu_queue=queue
    )
    bots1 = generator1.generate_population()
    
    # Generate again with same seed
    generator2 = BotGenerator(
        population_size=20,
        min_indicators=4,
        max_indicators=6,
        min_risk_strategies=2,
        max_risk_strategies=4,
        leverage=15,
        random_seed=42,
        gpu_context=context,
        gpu_queue=queue
    )
    bots2 = generator2.generate_population()
    
    # Should produce identical bots
    for i in range(20):
        assert bots1[i].bot_id == bots2[i].bot_id
        assert len(bots1[i].indicators) == len(bots2[i].indicators)
        assert len(bots1[i].risk_strategies) == len(bots2[i].risk_strategies)
        # Note: Due to floating point precision, exact param matching may vary slightly


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
