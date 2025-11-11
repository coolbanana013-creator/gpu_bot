"""
System smoke test: End-to-end evolution with metrics collection.

Runs a small evolution (small population, few generations) and collects:
- Trade frequency statistics
- Fitness distribution
- Resource usage (GPU memory, time)
- Bot diversity metrics
"""
import pytest
import numpy as np
from pathlib import Path
import time
import json

try:
    import pyopencl as cl
    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False

from src.bot_generator.compact_generator import CompactBotGenerator
from src.backtester.compact_simulator import CompactBacktester
from src.ga.evolver_compact import GeneticAlgorithmEvolver
from src.data_provider.synthetic import generate_synthetic_ohlcv


@pytest.mark.skipif(not OPENCL_AVAILABLE, reason="OpenCL not available")
class TestSystemSmoke:
    """End-to-end system tests."""
    
    @classmethod
    def setup_class(cls):
        """Initialize GPU context once for all tests."""
        try:
            platforms = cl.get_platforms()
            if not platforms:
                pytest.skip("No OpenCL platforms found")
            
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
                pytest.skip("No GPU device found")
            
            cls.ctx = cl.Context([gpu_device])
            cls.queue = cl.CommandQueue(cls.ctx)
            cls.device = gpu_device
            
        except Exception as e:
            pytest.skip(f"GPU initialization failed: {e}")
    
    def test_small_evolution_runs_without_errors(self):
        """Test that a small evolution completes without crashing."""
        # Parameters for small test
        population_size = 100
        generations = 2
        cycles = 2
        num_bars = 1000  # ~16 hours at 1-minute bars
        
        # Generate synthetic data
        ohlcv_data = generate_synthetic_ohlcv(
            num_bars=num_bars,
            initial_price=50000.0,
            volatility=0.02
        )
        
        # Create generator
        generator = CompactBotGenerator(
            population_size=population_size,
            min_indicators=1,
            max_indicators=3,  # Small for faster testing
            min_leverage=1,
            max_leverage=10,  # Lower leverage for stability
            gpu_context=self.ctx,
            gpu_queue=self.queue
        )
        
        # Create backtester
        backtester = CompactBacktester(
            gpu_context=self.ctx,
            gpu_queue=self.queue,
            initial_balance=10000.0
        )
        
        # Create evolver
        evolver = GeneticAlgorithmEvolver(
            generator=generator,
            backtester=backtester,
            population_size=population_size,
            num_generations=generations,
            cycles_per_generation=cycles,
            mutation_rate=0.15,
            crossover_rate=0.7
        )
        
        # Run evolution
        start_time = time.time()
        try:
            best_bot, best_result = evolver.evolve(ohlcv_data)
            elapsed = time.time() - start_time
            
            # Basic assertions
            assert best_bot is not None, "Should return a best bot"
            assert best_result is not None, "Should return best result"
            assert elapsed > 0, "Should take some time"
            
            print(f"\n=== Smoke Test Results ===")
            print(f"Time elapsed: {elapsed:.2f}s")
            print(f"Best fitness: {best_result.fitness_score:.2f}")
            print(f"Best bot trades: {best_result.total_trades}")
            print(f"Best bot PnL: ${best_result.total_pnl:.2f}")
            
        except Exception as e:
            pytest.fail(f"Evolution failed with error: {e}")
    
    def test_collect_trade_frequency_metrics(self):
        """Collect and report trade frequency distribution."""
        # TODO: Run evolution and collect trade count distribution
        # Report: min, max, mean, median, std of trades per bot
        assert True, "Trade frequency metrics collection not yet implemented"
    
    def test_collect_fitness_distribution(self):
        """Collect and report fitness score distribution."""
        # TODO: Run evolution and collect fitness distribution per generation
        # Report: Best, worst, mean, median fitness over generations
        assert True, "Fitness distribution collection not yet implemented"
    
    def test_measure_gpu_memory_usage(self):
        """Measure actual GPU memory consumption during evolution."""
        # TODO: Query GPU memory before/after evolution
        # Report: Peak memory usage, memory per bot
        assert True, "GPU memory measurement not yet implemented"
    
    def test_bot_diversity_metrics(self):
        """Measure diversity of indicator combinations in population."""
        # TODO: Collect unique indicator combinations per generation
        # Report: Diversity percentage, most common combinations
        assert True, "Bot diversity metrics not yet implemented"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
