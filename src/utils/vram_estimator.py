"""
VRAM Estimation and Validation Utility

Calculates required GPU memory before kernel execution.
Crashes if estimated VRAM exceeds available GPU memory.

No CPU fallbacks - GPU requirement is mandatory.
"""
import numpy as np
from typing import Tuple


class VRAMEstimator:
    """Estimates and validates VRAM requirements for GPU kernels."""
    
    # Constants matching OpenCL kernels
    MAX_INDICATORS = 20
    MAX_RISK_STRATEGIES = 10
    MAX_PARAMS = 10
    MAX_POSITIONS = 100
    
    def __init__(self, gpu_context=None, gpu_queue=None):
        """
        Initialize VRAM estimator with actual GPU VRAM query.
        
        Args:
            gpu_context: Optional OpenCL context for device info
            gpu_queue: Optional OpenCL command queue
        """
        self.ctx = gpu_context
        self.queue = gpu_queue
        
        if gpu_context:
            # Get actual device VRAM (global memory size)
            devices = gpu_context.devices
            if devices:
                device = devices[0]
                self.device_vram = device.global_mem_size
                self.device_max_alloc = device.max_mem_alloc_size
                self.device_local_mem = device.local_mem_size
                
                # Log device memory info
                print(f"GPU Memory Info:")
                print(f"  Total VRAM: {self.device_vram / (1024**3):.2f} GB")
                print(f"  Max Single Allocation: {self.device_max_alloc / (1024**3):.2f} GB")
                print(f"  Local Memory: {self.device_local_mem / 1024:.2f} KB")
            else:
                self.device_vram = None
                self.device_max_alloc = None
                self.device_local_mem = None
        else:
            self.device_vram = None
            self.device_max_alloc = None
            self.device_local_mem = None
    
    # Constants matching OpenCL kernels (class variables)
    MAX_INDICATORS = 20
    MAX_RISK_STRATEGIES = 10
    MAX_PARAMS = 10
    MAX_POSITIONS = 100
    
    # Struct sizes in bytes (approximate)
    BOT_CONFIG_SIZE = (
        4 +  # bot_id
        4 +  # num_indicators
        (MAX_INDICATORS * 4) +  # indicator_types
        (MAX_INDICATORS * MAX_PARAMS * 4) +  # indicator_params
        4 +  # num_risk_strategies
        (MAX_RISK_STRATEGIES * 4) +  # risk_strategy_types
        (MAX_RISK_STRATEGIES * MAX_PARAMS * 4) +  # risk_strategy_params
        4 +  # take_profit_pct
        4 +  # stop_loss_pct
        4    # leverage
    )
    
    BACKTEST_RESULT_SIZE = (
        4 +  # bot_id
        4 +  # cycle_id
        4 +  # final_balance
        4 +  # profit_pct
        4 +  # total_trades
        4 +  # winning_trades
        4 +  # losing_trades
        4 +  # win_rate
        4 +  # max_drawdown_pct
        4 +  # sharpe_ratio
        4 +  # total_fees_paid
        4    # total_funding_paid
    )
    
    OHLCV_BAR_SIZE = 5 * 4  # 5 floats (O, H, L, C, V)
    
    @staticmethod
    def estimate_bot_generation_vram(
        population_size: int,
        num_indicator_types: int,
        num_risk_strategy_types: int,
        num_indicator_param_ranges: int,
        num_risk_param_ranges: int
    ) -> dict:
        """
        Estimate VRAM required for bot generation kernel.
        
        Args:
            population_size: Number of bots to generate
            num_indicator_types: Number of available indicator types
            num_risk_strategy_types: Number of available risk strategy types
            num_indicator_param_ranges: Number of indicator parameter ranges
            num_risk_param_ranges: Number of risk strategy parameter ranges
            
        Returns:
            Dictionary with VRAM breakdown and total
        """
        estimate = {}
        
        # Bot configs output
        estimate['bot_configs'] = population_size * VRAMEstimator.BOT_CONFIG_SIZE
        
        # Indicator types (constant memory, but still counted)
        estimate['indicator_types'] = num_indicator_types * 4
        
        # Risk strategy types
        estimate['risk_strategy_types'] = num_risk_strategy_types * 4
        
        # Indicator parameter ranges
        param_range_size = (
            4 +  # indicator_type
            4 +  # num_params
            (VRAMEstimator.MAX_PARAMS * 4) +  # param_mins
            (VRAMEstimator.MAX_PARAMS * 4)    # param_maxs
        )
        estimate['indicator_param_ranges'] = num_indicator_param_ranges * param_range_size
        
        # Risk strategy parameter ranges
        estimate['risk_param_ranges'] = num_risk_param_ranges * param_range_size
        
        # Random seeds
        estimate['random_seeds'] = population_size * 4
        
        # Total
        estimate['total_bytes'] = sum(estimate.values())
        estimate['total_mb'] = estimate['total_bytes'] / (1024 ** 2)
        estimate['total_gb'] = estimate['total_bytes'] / (1024 ** 3)
        
        return estimate
    
    @staticmethod
    def estimate_backtesting_vram(
        population_size: int,
        num_cycles: int,
        total_bars: int,
        num_indicator_types: int
    ) -> dict:
        """
        Estimate VRAM required for backtesting kernel.
        
        Args:
            population_size: Number of bots
            num_cycles: Number of backtest cycles
            total_bars: Total OHLCV bars in dataset
            num_indicator_types: Number of indicator types for precomputation
            
        Returns:
            Dictionary with VRAM breakdown and total
        """
        estimate = {}
        
        # Bot configs input
        estimate['bot_configs'] = population_size * VRAMEstimator.BOT_CONFIG_SIZE
        
        # Precomputed indicators [total_bars, num_indicator_types, MAX_PARAMS]
        estimate['precomputed_indicators'] = (
            total_bars * num_indicator_types * VRAMEstimator.MAX_PARAMS * 4
        )
        
        # OHLCV data
        estimate['ohlcv_data'] = total_bars * VRAMEstimator.OHLCV_BAR_SIZE
        
        # Cycle ranges
        estimate['cycle_starts'] = num_cycles * 4
        estimate['cycle_ends'] = num_cycles * 4
        
        # Results output [population_size * num_cycles]
        estimate['results'] = (
            population_size * num_cycles * VRAMEstimator.BACKTEST_RESULT_SIZE
        )
        
        # Total
        estimate['total_bytes'] = sum(estimate.values())
        estimate['total_mb'] = estimate['total_bytes'] / (1024 ** 2)
        estimate['total_gb'] = estimate['total_bytes'] / (1024 ** 3)
        
        return estimate
    
    @staticmethod
    def validate_vram_availability(
        required_vram_bytes: int,
        available_vram_bytes: int,
        safety_margin: float = 0.9
    ) -> None:
        """
        Validate that required VRAM is available.
        
        Args:
            required_vram_bytes: Required VRAM in bytes
            available_vram_bytes: Available GPU VRAM in bytes
            safety_margin: Use only this fraction of available VRAM (default 90%)
            
        Raises:
            RuntimeError: If required VRAM exceeds available VRAM
        """
        usable_vram = available_vram_bytes * safety_margin
        
        if required_vram_bytes > usable_vram:
            required_gb = required_vram_bytes / (1024 ** 3)
            available_gb = available_vram_bytes / (1024 ** 3)
            usable_gb = usable_vram / (1024 ** 3)
            
            raise RuntimeError(
                f"INSUFFICIENT VRAM\n"
                f"Required: {required_gb:.2f} GB\n"
                f"Available: {available_gb:.2f} GB\n"
                f"Usable (with {safety_margin*100:.0f}% margin): {usable_gb:.2f} GB\n"
                f"\nReduce population size, cycles, or backtest days to fit in available VRAM."
            )
    
    @staticmethod
    def estimate_and_validate_workflow(
        population_size: int,
        num_cycles: int,
        total_bars: int,
        num_indicator_types: int = 30,
        num_risk_strategy_types: int = 12,
        gpu_vram_bytes: int = None
    ) -> dict:
        """
        Estimate VRAM for full workflow and validate against available GPU memory.
        
        Args:
            population_size: Number of bots
            num_cycles: Number of backtest cycles
            total_bars: Total OHLCV bars
            num_indicator_types: Number of indicator types (default 30)
            num_risk_strategy_types: Number of risk strategies (default 12)
            gpu_vram_bytes: Available GPU VRAM in bytes (if None, skips validation)
            
        Returns:
            Dictionary with estimates for each stage
            
        Raises:
            RuntimeError: If required VRAM exceeds available
        """
        estimates = {}
        
        # Bot generation stage
        estimates['bot_generation'] = VRAMEstimator.estimate_bot_generation_vram(
            population_size=population_size,
            num_indicator_types=num_indicator_types,
            num_risk_strategy_types=num_risk_strategy_types,
            num_indicator_param_ranges=num_indicator_types,
            num_risk_param_ranges=num_risk_strategy_types
        )
        
        # Backtesting stage
        estimates['backtesting'] = VRAMEstimator.estimate_backtesting_vram(
            population_size=population_size,
            num_cycles=num_cycles,
            total_bars=total_bars,
            num_indicator_types=num_indicator_types
        )
        
        # Peak VRAM (backtesting is typically larger)
        estimates['peak_vram_bytes'] = max(
            estimates['bot_generation']['total_bytes'],
            estimates['backtesting']['total_bytes']
        )
        estimates['peak_vram_mb'] = estimates['peak_vram_bytes'] / (1024 ** 2)
        estimates['peak_vram_gb'] = estimates['peak_vram_bytes'] / (1024 ** 3)
        
        # Validate if GPU VRAM provided
        if gpu_vram_bytes is not None:
            VRAMEstimator.validate_vram_availability(
                required_vram_bytes=estimates['peak_vram_bytes'],
                available_vram_bytes=gpu_vram_bytes
            )
        
        return estimates
    
    @staticmethod
    def print_vram_report(estimates: dict) -> None:
        """
        Print formatted VRAM estimation report.
        
        Args:
            estimates: Estimates dictionary from estimate_and_validate_workflow
        """
        print("\n" + "="*60)
        print("VRAM ESTIMATION REPORT")
        print("="*60)
        
        print("\n--- Bot Generation Stage ---")
        bot_gen = estimates['bot_generation']
        print(f"  Bot Configs:      {bot_gen['bot_configs'] / (1024**2):>8.2f} MB")
        print(f"  Indicator Data:   {bot_gen['indicator_types'] / (1024**2):>8.2f} MB")
        print(f"  Risk Data:        {bot_gen['risk_strategy_types'] / (1024**2):>8.2f} MB")
        print(f"  Param Ranges:     {(bot_gen['indicator_param_ranges'] + bot_gen['risk_param_ranges']) / (1024**2):>8.2f} MB")
        print(f"  Random Seeds:     {bot_gen['random_seeds'] / (1024**2):>8.2f} MB")
        print(f"  TOTAL:            {bot_gen['total_mb']:>8.2f} MB ({bot_gen['total_gb']:.3f} GB)")
        
        print("\n--- Backtesting Stage ---")
        backtest = estimates['backtesting']
        print(f"  Bot Configs:      {backtest['bot_configs'] / (1024**2):>8.2f} MB")
        print(f"  Precomp Inds:     {backtest['precomputed_indicators'] / (1024**2):>8.2f} MB")
        print(f"  OHLCV Data:       {backtest['ohlcv_data'] / (1024**2):>8.2f} MB")
        print(f"  Cycle Ranges:     {(backtest['cycle_starts'] + backtest['cycle_ends']) / (1024**2):>8.2f} MB")
        print(f"  Results:          {backtest['results'] / (1024**2):>8.2f} MB")
        print(f"  TOTAL:            {backtest['total_mb']:>8.2f} MB ({backtest['total_gb']:.3f} GB)")
        
        print("\n--- Peak VRAM Usage ---")
        print(f"  Peak Required:    {estimates['peak_vram_mb']:>8.2f} MB ({estimates['peak_vram_gb']:.3f} GB)")
        print("="*60 + "\n")


def estimate_max_population(
    available_vram_gb: float,
    total_bars: int,
    num_cycles: int,
    safety_margin: float = 0.9
) -> int:
    """
    Estimate maximum population size that fits in available VRAM.
    
    Args:
        available_vram_gb: Available GPU VRAM in GB
        total_bars: Total OHLCV bars in dataset
        num_cycles: Number of backtest cycles
        safety_margin: Use only this fraction of VRAM
        
    Returns:
        Maximum safe population size
    """
    available_bytes = int(available_vram_gb * (1024 ** 3) * safety_margin)
    
    # Binary search for max population
    low, high = 1000, 10_000_000
    max_pop = 1000
    
    while low <= high:
        mid = (low + high) // 2
        
        try:
            estimates = VRAMEstimator.estimate_and_validate_workflow(
                population_size=mid,
                num_cycles=num_cycles,
                total_bars=total_bars,
                gpu_vram_bytes=int(available_bytes / safety_margin)
            )
            
            if estimates['peak_vram_bytes'] <= available_bytes:
                max_pop = mid
                low = mid + 1
            else:
                high = mid - 1
        except RuntimeError:
            high = mid - 1
    
    return max_pop


if __name__ == "__main__":
    # Example usage
    print("VRAM Estimator Test")
    
    # Estimate for typical configuration
    estimates = VRAMEstimator.estimate_and_validate_workflow(
        population_size=10_000,
        num_cycles=10,
        total_bars=50_000  # ~35 days of 1m data
    )
    
    VRAMEstimator.print_vram_report(estimates)
    
    # Test with 8GB GPU
    print("Testing fit on 8GB GPU...")
    try:
        VRAMEstimator.validate_vram_availability(
            required_vram_bytes=estimates['peak_vram_bytes'],
            available_vram_bytes=8 * (1024 ** 3)
        )
        print("✓ Fits in 8GB GPU with safety margin")
    except RuntimeError as e:
        print(f"✗ Does not fit: {e}")
    
    # Calculate max population for 8GB
    max_pop = estimate_max_population(
        available_vram_gb=8.0,
        total_bars=50_000,
        num_cycles=10
    )
    print(f"\nMax population for 8GB GPU: {max_pop:,} bots")
