"""
Bot Generator - GPU-Only PyOpenCL Implementation

CRITICAL: This module requires OpenCL-capable GPU. No CPU fallbacks.
Generates trading bot configurations in parallel using GPU kernels.
"""
import numpy as np
import pyopencl as cl
from typing import List, Set, Optional
from dataclasses import dataclass
import os

from ..indicators.factory import IndicatorType, IndicatorFactory, IndicatorParams
from ..risk_management.strategies import RiskStrategyType, RiskStrategyFactory, RiskStrategyParams
from ..utils.validation import validate_int, log_info, log_error, log_warning
from ..utils.vram_estimator import VRAMEstimator
from ..utils.config import (
    MIN_INDICATORS_PER_BOT, MAX_INDICATORS_PER_BOT,
    MIN_RISK_STRATEGIES_PER_BOT, MAX_RISK_STRATEGIES_PER_BOT
)


# Constants matching OpenCL kernel
MAX_INDICATORS = 20
MAX_RISK_STRATEGIES = 10
MAX_PARAMS = 10


@dataclass
class BotConfig:
    """Configuration for a single trading bot."""
    bot_id: int
    indicators: List[IndicatorParams]
    risk_strategies: List[RiskStrategyParams]
    take_profit_pct: float
    stop_loss_pct: float
    leverage: int
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "bot_id": self.bot_id,
            "indicators": [ind.to_dict() for ind in self.indicators],
            "risk_strategies": [strat.to_dict() for strat in self.risk_strategies],
            "take_profit_pct": self.take_profit_pct,
            "stop_loss_pct": self.stop_loss_pct,
            "leverage": self.leverage
        }
    
    def get_indicator_combo_signature(self) -> frozenset:
        """Get hashable signature for indicator combination."""
        return frozenset(ind.indicator_type for ind in self.indicators)


class BotGenerator:
    """
    GPU-accelerated bot configuration generator.
    Uses OpenCL kernel for parallel generation.
    
    CRITICAL: Requires GPU. Crashes if GPU unavailable or kernel compilation fails.
    """
    
    def __init__(
        self,
        population_size: int,
        min_indicators: int = MIN_INDICATORS_PER_BOT,
        max_indicators: int = MAX_INDICATORS_PER_BOT,
        min_risk_strategies: int = MIN_RISK_STRATEGIES_PER_BOT,
        max_risk_strategies: int = MAX_RISK_STRATEGIES_PER_BOT,
        leverage: int = 10,
        random_seed: Optional[int] = None,
        gpu_context: Optional[cl.Context] = None,
        gpu_queue: Optional[cl.CommandQueue] = None
    ):
        """
        Initialize GPU-based bot generator.
        
        Args:
            population_size: Number of bots to generate
            min_indicators: Minimum indicators per bot
            max_indicators: Maximum indicators per bot
            min_risk_strategies: Minimum risk strategies per bot
            max_risk_strategies: Maximum risk strategies per bot
            leverage: Leverage multiplier
            random_seed: Random seed for reproducibility
            gpu_context: PyOpenCL context (REQUIRED)
            gpu_queue: PyOpenCL command queue (REQUIRED)
            
        Raises:
            RuntimeError: If GPU context/queue not provided or kernel fails to compile
        """
        # Validate GPU context
        if gpu_context is None or gpu_queue is None:
            raise RuntimeError(
                "GPU context and queue are REQUIRED. No CPU fallbacks available.\n"
                "Initialize GPU before creating BotGenerator."
            )
        
        self.gpu_context = gpu_context
        self.gpu_queue = gpu_queue
        
        # Validate parameters (allow small populations for testing)
        self.population_size = validate_int(population_size, "population_size", min_val=1)
        self.min_indicators = validate_int(min_indicators, "min_indicators", 
                                          min_val=MIN_INDICATORS_PER_BOT, 
                                          max_val=MAX_INDICATORS_PER_BOT)
        self.max_indicators = validate_int(max_indicators, "max_indicators",
                                          min_val=self.min_indicators,
                                          max_val=MAX_INDICATORS_PER_BOT)
        self.min_risk_strategies = validate_int(min_risk_strategies, "min_risk_strategies",
                                               min_val=MIN_RISK_STRATEGIES_PER_BOT,
                                               max_val=MAX_RISK_STRATEGIES_PER_BOT)
        self.max_risk_strategies = validate_int(max_risk_strategies, "max_risk_strategies",
                                               min_val=self.min_risk_strategies,
                                               max_val=MAX_RISK_STRATEGIES_PER_BOT)
        self.leverage = validate_int(leverage, "leverage", min_val=1, max_val=125)
        self.random_seed = random_seed if random_seed is not None else 42
        
        # Get indicator and risk strategy metadata
        self.all_indicator_types = IndicatorFactory.get_all_indicator_types()
        self.all_risk_strategy_types = RiskStrategyFactory.get_all_strategy_types()
        
        # Track population
        self.population: List[BotConfig] = []
        self.used_combos: Set[frozenset] = set()
        
        # Compile OpenCL kernel
        self._compile_kernel()
        
        # Estimate VRAM
        self._estimate_vram()
        
        log_info(
            f"GPU Bot Generator initialized: pop={population_size}, "
            f"indicators={min_indicators}-{max_indicators}, "
            f"risk={min_risk_strategies}-{max_risk_strategies}, "
            f"leverage={leverage}x, seed={self.random_seed}"
        )
    
    def _compile_kernel(self) -> None:
        """
        Compile OpenCL kernel from source file.
        
        Raises:
            RuntimeError: If kernel file not found or compilation fails
        """
        try:
            # Get kernel source path
            kernel_dir = os.path.join(os.path.dirname(__file__), '..', 'gpu_kernels')
            kernel_path = os.path.join(kernel_dir, 'bot_gen_impl.cl')
            
            if not os.path.exists(kernel_path):
                raise RuntimeError(
                    f"Kernel source not found: {kernel_path}\n"
                    f"GPU implementation requires bot_gen_impl.cl"
                )
            
            # Load kernel source
            with open(kernel_path, 'r') as f:
                kernel_source = f.read()
            
            # Compile program
            log_info("Compiling bot generation OpenCL kernel...")
            self.program = cl.Program(self.gpu_context, kernel_source).build()
            log_info("✓ Kernel compiled successfully")
            
        except cl.RuntimeError as e:
            log_error(f"OpenCL kernel compilation FAILED:\n{e}")
            raise RuntimeError(
                f"Failed to compile bot generation kernel.\n"
                f"This is a FATAL error - no CPU fallback available.\n"
                f"Error: {e}"
            )
        except Exception as e:
            log_error(f"Kernel compilation error: {e}")
            raise RuntimeError(f"Kernel compilation failed: {e}")
    
    def _estimate_vram(self) -> None:
        """
        Estimate VRAM requirements and validate against available GPU memory.
        
        Raises:
            RuntimeError: If required VRAM exceeds available
        """
        try:
            # Get GPU global memory size
            device = self.gpu_context.devices[0]
            available_vram = device.global_mem_size
            
            # Estimate required VRAM
            estimates = VRAMEstimator.estimate_bot_generation_vram(
                population_size=self.population_size,
                num_indicator_types=len(self.all_indicator_types),
                num_risk_strategy_types=len(self.all_risk_strategy_types),
                num_indicator_param_ranges=len(self.all_indicator_types),
                num_risk_param_ranges=len(self.all_risk_strategy_types)
            )
            
            # Validate
            VRAMEstimator.validate_vram_availability(
                required_vram_bytes=estimates['total_bytes'],
                available_vram_bytes=available_vram
            )
            
            log_info(f"VRAM estimate: {estimates['total_mb']:.2f} MB / {available_vram/(1024**3):.2f} GB available")
            
        except RuntimeError as e:
            log_error(f"VRAM validation failed: {e}")
            raise
    
    def generate_population(self) -> List[BotConfig]:
        """
        Generate full population of bots using GPU kernel.
        
        Returns:
            List of BotConfig instances
            
        Raises:
            RuntimeError: If GPU execution fails
        """
        log_info(f"Generating {self.population_size} bots on GPU...")
        
        try:
            # Prepare input data
            indicator_types_array, risk_types_array = self._prepare_type_arrays()
            ind_param_ranges, risk_param_ranges = self._prepare_param_ranges()
            random_seeds = self._generate_random_seeds()
            
            # Create OpenCL buffers
            mf = cl.mem_flags
            
            # Output buffer for bot configs
            bot_config_size = self._calculate_bot_config_struct_size()
            bot_configs_buf = cl.Buffer(
                self.gpu_context,
                mf.WRITE_ONLY,
                size=bot_config_size * self.population_size
            )
            
            # Input buffers
            indicator_types_buf = cl.Buffer(
                self.gpu_context,
                mf.READ_ONLY | mf.COPY_HOST_PTR,
                hostbuf=indicator_types_array
            )
            
            risk_types_buf = cl.Buffer(
                self.gpu_context,
                mf.READ_ONLY | mf.COPY_HOST_PTR,
                hostbuf=risk_types_array
            )
            
            ind_param_ranges_buf = cl.Buffer(
                self.gpu_context,
                mf.READ_ONLY | mf.COPY_HOST_PTR,
                hostbuf=ind_param_ranges
            )
            
            risk_param_ranges_buf = cl.Buffer(
                self.gpu_context,
                mf.READ_ONLY | mf.COPY_HOST_PTR,
                hostbuf=risk_param_ranges
            )
            
            random_seeds_buf = cl.Buffer(
                self.gpu_context,
                mf.READ_ONLY | mf.COPY_HOST_PTR,
                hostbuf=random_seeds
            )
            
            # TP/SL ranges
            tp_min, tp_max, sl_min, sl_max = self._get_tp_sl_ranges()
            
            # Execute kernel
            log_info("Executing GPU kernel...")
            kernel = self.program.generate_bots
            kernel.set_args(
                bot_configs_buf,
                indicator_types_buf,
                np.int32(len(self.all_indicator_types)),
                risk_types_buf,
                np.int32(len(self.all_risk_strategy_types)),
                ind_param_ranges_buf,
                np.int32(len(self.all_indicator_types)),
                risk_param_ranges_buf,
                np.int32(len(self.all_risk_strategy_types)),
                random_seeds_buf,
                np.int32(self.population_size),
                np.int32(self.min_indicators),
                np.int32(self.max_indicators),
                np.int32(self.min_risk_strategies),
                np.int32(self.max_risk_strategies),
                np.int32(self.leverage),
                np.float32(tp_min),
                np.float32(tp_max),
                np.float32(sl_min),
                np.float32(sl_max)
            )
            
            # Enqueue kernel (one thread per bot)
            global_size = (self.population_size,)
            cl.enqueue_nd_range_kernel(self.gpu_queue, kernel, global_size, None)
            
            # Read results
            log_info("Reading results from GPU...")
            bot_configs_array = np.empty(self.population_size * bot_config_size, dtype=np.uint8)
            cl.enqueue_copy(self.gpu_queue, bot_configs_array, bot_configs_buf)
            self.gpu_queue.finish()
            
            # Parse results
            log_info("Parsing bot configurations...")
            self.population = self._parse_bot_configs(bot_configs_array)
            
            log_info(f"✓ Generated {len(self.population)} bots successfully")
            
            return self.population
            
        except Exception as e:
            log_error(f"GPU bot generation FAILED: {e}")
            raise RuntimeError(
                f"Bot generation on GPU failed.\n"
                f"No CPU fallback available per specification.\n"
                f"Error: {e}"
            )
    
    def _prepare_type_arrays(self):
        """Prepare numpy arrays for indicator and risk strategy types."""
        # Convert enums to integers using their indices in the list
        indicator_types = np.array(list(range(len(self.all_indicator_types))), dtype=np.int32)
        risk_types = np.array(list(range(len(self.all_risk_strategy_types))), dtype=np.int32)
        return indicator_types, risk_types
    
    def _prepare_param_ranges(self):
        """
        Prepare parameter range structures for GPU.
        This is a placeholder - actual implementation would serialize
        IndicatorFactory and RiskStrategyFactory parameter ranges.
        """
        # Simplified: Create dummy parameter ranges
        # In full implementation, query factories for actual ranges
        
        num_ind_types = len(self.all_indicator_types)
        num_risk_types = len(self.all_risk_strategy_types)
        
        # Structure: [type, num_params, param_mins[10], param_maxs[10]]
        ind_ranges = np.zeros((num_ind_types, 22), dtype=np.float32)
        risk_ranges = np.zeros((num_risk_types, 22), dtype=np.float32)
        
        # Populate with actual factory data (simplified here)
        for i in range(num_ind_types):
            ind_ranges[i, 0] = float(i)  # type ID (index)
            ind_ranges[i, 1] = 2.0  # num_params (example)
            # Mins and maxs would come from IndicatorFactory
        
        for i in range(num_risk_types):
            risk_ranges[i, 0] = float(i)  # type ID (index)
            risk_ranges[i, 1] = 2.0
        
        return ind_ranges, risk_ranges
    
    def _generate_random_seeds(self) -> np.ndarray:
        """Generate unique random seeds for each bot."""
        np.random.seed(self.random_seed)
        seeds = np.random.randint(0, 2**31, size=self.population_size, dtype=np.uint32)
        return seeds
    
    def _get_tp_sl_ranges(self):
        """Get TP/SL percentage ranges."""
        # Adjust for leverage
        lev_factor = np.sqrt(self.leverage)
        tp_min = 0.5 / lev_factor
        tp_max = 10.0 / lev_factor
        sl_min = 0.3 / lev_factor
        sl_max = 5.0 / lev_factor
        return tp_min, tp_max, sl_min, sl_max
    
    def _calculate_bot_config_struct_size(self) -> int:
        """Calculate size of BotConfig struct in bytes."""
        # Match OpenCL struct definition
        size = (
            4 +  # bot_id (int)
            4 +  # num_indicators (int)
            (MAX_INDICATORS * 4) +  # indicator_types
            (MAX_INDICATORS * MAX_PARAMS * 4) +  # indicator_params
            4 +  # num_risk_strategies
            (MAX_RISK_STRATEGIES * 4) +  # risk_strategy_types
            (MAX_RISK_STRATEGIES * MAX_PARAMS * 4) +  # risk_strategy_params
            4 +  # take_profit_pct
            4 +  # stop_loss_pct
            4    # leverage
        )
        return size
    
    def _parse_bot_configs(self, raw_data: np.ndarray) -> List[BotConfig]:
        """
        Parse raw byte array from GPU into BotConfig objects.
        
        Deserializes OpenCL BotConfig struct bytes into Python objects.
        Struct layout must match bot_gen_impl.cl exactly.
        
        Args:
            raw_data: Raw bytes from GPU buffer
            
        Returns:
            List of parsed BotConfig objects
            
        Raises:
            ValueError: If struct data is malformed or invalid
        """
        configs = []
        
        # Calculate struct size (must match OpenCL)
        struct_size = self._calculate_bot_config_struct_size()
        
        if len(raw_data) < self.population_size * struct_size:
            raise ValueError(
                f"Insufficient data: got {len(raw_data)} bytes, "
                f"expected {self.population_size * struct_size}"
            )
        
        log_info(f"Parsing {self.population_size} bot configurations from GPU...")
        
        offset = 0
        for i in range(self.population_size):
            try:
                # Extract struct bytes for this bot
                bot_bytes = raw_data[offset:offset + struct_size]
                
                # Parse fields according to struct layout
                pos = 0
                
                # int bot_id
                bot_id = np.frombuffer(bot_bytes[pos:pos+4], dtype=np.int32)[0]
                pos += 4
                
                # int num_indicators
                num_indicators = np.frombuffer(bot_bytes[pos:pos+4], dtype=np.int32)[0]
                pos += 4
                
                # int indicator_types[MAX_INDICATORS]
                indicator_types = np.frombuffer(
                    bot_bytes[pos:pos+(MAX_INDICATORS*4)], 
                    dtype=np.int32
                )[:num_indicators]
                pos += MAX_INDICATORS * 4
                
                # float indicator_params[MAX_INDICATORS * MAX_PARAMS] (flattened)
                indicator_params_flat = np.frombuffer(
                    bot_bytes[pos:pos+(MAX_INDICATORS*MAX_PARAMS*4)],
                    dtype=np.float32
                )
                indicator_params = indicator_params_flat.reshape(MAX_INDICATORS, MAX_PARAMS)[:num_indicators]
                pos += MAX_INDICATORS * MAX_PARAMS * 4
                
                # int num_risk_strategies
                num_risk_strategies = np.frombuffer(bot_bytes[pos:pos+4], dtype=np.int32)[0]
                pos += 4
                
                # int risk_strategy_types[MAX_RISK_STRATEGIES]
                risk_strategy_types = np.frombuffer(
                    bot_bytes[pos:pos+(MAX_RISK_STRATEGIES*4)],
                    dtype=np.int32
                )[:num_risk_strategies]
                pos += MAX_RISK_STRATEGIES * 4
                
                # float risk_strategy_params[MAX_RISK_STRATEGIES * MAX_PARAMS] (flattened)
                risk_params_flat = np.frombuffer(
                    bot_bytes[pos:pos+(MAX_RISK_STRATEGIES*MAX_PARAMS*4)],
                    dtype=np.float32
                )
                risk_params = risk_params_flat.reshape(MAX_RISK_STRATEGIES, MAX_PARAMS)[:num_risk_strategies]
                pos += MAX_RISK_STRATEGIES * MAX_PARAMS * 4
                
                # float take_profit_pct
                take_profit_pct = np.frombuffer(bot_bytes[pos:pos+4], dtype=np.float32)[0]
                pos += 4
                
                # float stop_loss_pct
                stop_loss_pct = np.frombuffer(bot_bytes[pos:pos+4], dtype=np.float32)[0]
                pos += 4
                
                # int leverage
                leverage = np.frombuffer(bot_bytes[pos:pos+4], dtype=np.int32)[0]
                pos += 4
                
                # Validate parsed data
                if bot_id < 0 or bot_id >= self.population_size:
                    raise ValueError(f"Invalid bot_id: {bot_id}")
                
                if num_indicators < self.min_indicators or num_indicators > self.max_indicators:
                    raise ValueError(
                        f"Invalid num_indicators: {num_indicators} "
                        f"(expected {self.min_indicators}-{self.max_indicators})"
                    )
                
                if num_risk_strategies < self.min_risk_strategies or num_risk_strategies > self.max_risk_strategies:
                    raise ValueError(
                        f"Invalid num_risk_strategies: {num_risk_strategies} "
                        f"(expected {self.min_risk_strategies}-{self.max_risk_strategies})"
                    )
                
                # Convert to Python objects
                indicators = []
                for j in range(num_indicators):
                    # Map integer type ID to IndicatorType enum
                    type_id = int(indicator_types[j])
                    if type_id < 0 or type_id >= len(self.all_indicator_types):
                        raise ValueError(f"Invalid indicator type ID: {type_id}")
                    
                    ind_type = self.all_indicator_types[type_id]
                    
                    # Create params dict from array
                    params_dict = {}
                    for k in range(MAX_PARAMS):
                        param_val = float(indicator_params[j, k])
                        if param_val != 0.0 or k < 3:  # Keep first 3 params even if zero
                            params_dict[f'param_{k}'] = param_val
                    
                    ind_params = IndicatorParams(
                        indicator_type=ind_type,
                        params=params_dict
                    )
                    indicators.append(ind_params)
                
                # Convert risk strategies
                risk_strategies = []
                for j in range(num_risk_strategies):
                    # Map integer type ID to RiskStrategyType enum
                    type_id = int(risk_strategy_types[j])
                    if type_id < 0 or type_id >= len(self.all_risk_strategy_types):
                        raise ValueError(f"Invalid risk strategy type ID: {type_id}")
                    
                    risk_type = self.all_risk_strategy_types[type_id]
                    
                    # Create params dict from array
                    params_dict = {}
                    for k in range(MAX_PARAMS):
                        param_val = float(risk_params[j, k])
                        if param_val != 0.0 or k < 3:
                            params_dict[f'param_{k}'] = param_val
                    
                    risk_params_obj = RiskStrategyParams(
                        strategy_type=risk_type,
                        params=params_dict
                    )
                    risk_strategies.append(risk_params_obj)
                
                # Create BotConfig
                bot = BotConfig(
                    bot_id=int(bot_id),
                    indicators=indicators,
                    risk_strategies=risk_strategies,
                    take_profit_pct=float(take_profit_pct),
                    stop_loss_pct=float(stop_loss_pct),
                    leverage=int(leverage)
                )
                
                configs.append(bot)
                
                offset += struct_size
                
            except Exception as e:
                raise ValueError(
                    f"Failed to parse bot {i} at offset {offset}: {e}\n"
                    f"This indicates struct layout mismatch between host and kernel."
                )
        
        log_info(f"✓ Successfully parsed {len(configs)} bot configurations")
        
        # Track unique combinations
        for bot in configs:
            self.used_combos.add(bot.get_indicator_combo_signature())
        
        return configs
    
    def refill_population(self, num_to_generate: int) -> List[BotConfig]:
        """
        Generate new bots to refill population.
        Uses same GPU kernel with different seed offset.
        """
        log_info(f"Refilling {num_to_generate} bots...")
        
        # For simplicity, regenerate with new seed
        old_seed = self.random_seed
        self.random_seed += 1000  # Offset seed
        
        original_pop_size = self.population_size
        self.population_size = num_to_generate
        
        new_bots = self.generate_population()
        
        # Restore original settings
        self.population_size = original_pop_size
        self.random_seed = old_seed
        
        return new_bots
