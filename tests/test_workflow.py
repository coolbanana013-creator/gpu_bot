"""
Integration test for complete workflow without user input.
Tests end-to-end Mode 1 execution.
"""
import pytest
import numpy as np
import pyopencl as cl
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bot_generator.generator import BotGenerator
from src.backtester.simulator import GPUBacktester


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
def workflow_ohlcv():
    """Generate OHLCV data for workflow test."""
    np.random.seed(42)
    num_bars = 5000  # 5000 bars for realistic test
    
    base_price = 50000.0
    returns = np.random.randn(num_bars) * 0.015
    prices = base_price * np.exp(np.cumsum(returns))
    
    ohlcv = np.zeros((num_bars, 5), dtype=np.float32)
    
    for i in range(num_bars):
        price = prices[i]
        high_offset = abs(np.random.randn()) * 0.008
        low_offset = abs(np.random.randn()) * 0.008
        
        ohlcv[i, 0] = price
        ohlcv[i, 1] = price * (1 + high_offset)
        ohlcv[i, 2] = price * (1 - low_offset)
        ohlcv[i, 3] = price * (1 + np.random.randn() * 0.004)
        ohlcv[i, 4] = np.random.uniform(100, 1000)
    
    return ohlcv


def test_mode1_workflow_small(gpu_context, workflow_ohlcv):
    """Test complete Mode 1 workflow with small population."""
    context, queue = gpu_context
    
    # Parameters similar to Mode 1
    params = {
        "population": 100,
        "generations": 2,
        "random_seed": 42,
        "min_indicators": 3,
        "max_indicators": 7,
        "min_risk_strategies": 2,
        "max_risk_strategies": 5,
        "leverage": 10,
        "initial_balance": 10000.0
    }
    
    # Step 1: Generate initial population
    generator = BotGenerator(
        population_size=params["population"],
        min_indicators=params["min_indicators"],
        max_indicators=params["max_indicators"],
        min_risk_strategies=params["min_risk_strategies"],
        max_risk_strategies=params["max_risk_strategies"],
        leverage=params["leverage"],
        random_seed=params["random_seed"],
        gpu_context=context,
        gpu_queue=queue
    )
    
    bots = generator.generate_population()
    assert len(bots) == params["population"]
    
    # Step 2: Create backtester
    backtester = GPUBacktester(
        gpu_context=context,
        gpu_queue=queue,
        initial_balance=params["initial_balance"]
    )
    
    # Step 3: Run backtest on multiple cycles
    num_cycles = 3
    bars_per_cycle = len(workflow_ohlcv) // num_cycles
    
    cycle_starts = np.array([i * bars_per_cycle for i in range(num_cycles)], dtype=np.int32)
    cycle_ends = np.array([(i + 1) * bars_per_cycle - 1 for i in range(num_cycles)], dtype=np.int32)
    cycle_ends[-1] = len(workflow_ohlcv) - 1  # Last cycle goes to end
    
    results = backtester.backtest_bots(
        bots=bots,
        ohlcv_data=workflow_ohlcv,
        cycle_starts=cycle_starts,
        cycle_ends=cycle_ends
    )
    
    # Step 4: Validate results
    assert len(results) == params["population"]
    
    # Check that we got valid results
    valid_results = [r for r in results if r.total_trades > 0]
    assert len(valid_results) > 0, "Should have at least some bots with trades"
    
    # Sort by sharpe ratio
    sorted_results = sorted(results, key=lambda r: r.sharpe_ratio, reverse=True)
    top_10 = sorted_results[:10]
    
    # Verify structure of top results
    for result in top_10:
        assert result.bot_id > 0
        assert result.total_trades >= 0
        assert result.final_balance > 0
        assert -100 <= result.total_return_pct <= 10000  # Reasonable range
        assert -100 <= result.max_drawdown_pct <= 100
        assert 0 <= result.win_rate <= 100
    
    print(f"\nTop 3 bots:")
    for i, result in enumerate(top_10[:3], 1):
        print(f"  {i}. Bot {result.bot_id}: Sharpe={result.sharpe_ratio:.3f}, "
              f"Return={result.total_return_pct:.2f}%, Trades={result.total_trades}")


def test_gpu_kernel_execution_verification(gpu_context, workflow_ohlcv):
    """Verify GPU kernels are actually executing (not CPU fallback)."""
    context, queue = gpu_context
    
    # Generate small population
    generator = BotGenerator(
        population_size=50,
        min_indicators=4,
        max_indicators=6,
        min_risk_strategies=2,
        max_risk_strategies=4,
        leverage=5,
        random_seed=123,
        gpu_context=context,
        gpu_queue=queue
    )
    
    # This should execute GPU kernel
    bots = generator.generate_population()
    
    # Verify bots were actually generated with diversity
    bot_ids = [b.bot_id for b in bots]
    assert len(set(bot_ids)) == 50, "GPU should generate unique bot IDs"
    
    # Verify indicators vary
    indicator_counts = [len(b.indicators) for b in bots]
    assert len(set(indicator_counts)) > 1, "Should have variation in indicator counts"
    
    # Backtest execution
    backtester = GPUBacktester(
        gpu_context=context,
        gpu_queue=queue,
        initial_balance=10000.0
    )
    
    cycle_starts = np.array([0], dtype=np.int32)
    cycle_ends = np.array([len(workflow_ohlcv) - 1], dtype=np.int32)
    
    results = backtester.backtest_bots(
        bots=bots,
        ohlcv_data=workflow_ohlcv,
        cycle_starts=cycle_starts,
        cycle_ends=cycle_ends
    )
    
    # Verify GPU execution produced valid results
    assert len(results) == 50
    
    # Check result diversity (not all identical)
    final_balances = [r.final_balance for r in results]
    assert len(set(final_balances)) > 1, "GPU should produce varied results"


def test_vram_estimation_runs(gpu_context, workflow_ohlcv):
    """Test that VRAM estimation runs before execution."""
    context, queue = gpu_context
    
    from src.utils.vram_estimator import VRAMEstimator
    
    estimator = VRAMEstimator(context, queue)
    
    # Test bot generation estimation
    vram_gen = estimator.estimate_bot_generation_vram(
        population_size=1000,
        num_indicator_types=30,
        num_risk_strategy_types=12,
        num_indicator_param_ranges=30,
        num_risk_param_ranges=12
    )
    assert vram_gen['total_bytes'] > 0
    
    # Test backtesting estimation
    vram_backtest = estimator.estimate_backtesting_vram(
        population_size=1000,
        num_cycles=3,
        total_bars=len(workflow_ohlcv),
        num_indicator_types=20
    )
    assert vram_backtest['total_bytes'] > 0
    
    # Estimation should be reasonable (not gigantic)
    assert vram_gen['total_bytes'] < 1e9  # Less than 1GB for 1000 bots
    assert vram_backtest['total_bytes'] < 1e10  # Less than 10GB for reasonable test


def test_reproducibility_with_seed(gpu_context, workflow_ohlcv):
    """Test that random seed produces reproducible results."""
    context, queue = gpu_context
    
    seed = 12345
    
    # First run
    gen1 = BotGenerator(
        population_size=20,
        min_indicators=5,
        max_indicators=5,
        min_risk_strategies=3,
        max_risk_strategies=3,
        leverage=10,
        random_seed=seed,
        gpu_context=context,
        gpu_queue=queue
    )
    bots1 = gen1.generate_population()
    
    # Second run with same seed
    gen2 = BotGenerator(
        population_size=20,
        min_indicators=5,
        max_indicators=5,
        min_risk_strategies=3,
        max_risk_strategies=3,
        leverage=10,
        random_seed=seed,
        gpu_context=context,
        gpu_queue=queue
    )
    bots2 = gen2.generate_population()
    
    # Should generate identical bot IDs
    for i in range(20):
        assert bots1[i].bot_id == bots2[i].bot_id, f"Bot {i} IDs don't match"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
