"""
COMPACT Bot Generator - Memory-Optimized GPU Implementation

Generates 128-byte bot configs (vs 1344 bytes old).
1M bots = 119MB (vs 1.25GB old) - 90.7% memory savings.
"""
import numpy as np
import pyopencl as cl
from typing import List, Optional
from dataclasses import dataclass
from pathlib import Path

from ..indicators.factory import IndicatorFactory
from ..indicators.gpu_indicators import get_all_gpu_indicators, get_gpu_indicator_name, GPU_INDICATOR_COUNT
from ..utils.validation import log_info, log_error


# Compact bot: 132 bytes (packed struct) - MUST match OpenCL kernel exactly!
COMPACT_BOT_SIZE = 132
MAX_INDICATORS_PER_BOT = 8
MAX_PARAMS_PER_INDICATOR = 3


@dataclass
class CompactBotConfig:
    """Compact bot configuration (132 bytes)."""
    bot_id: int
    num_indicators: int
    indicator_indices: np.ndarray          # uint8[8] - indices 0-49
    indicator_params: np.ndarray           # float32[8][3] - parameters
    indicator_risk_strategies: np.ndarray  # uint8[8] - risk strategy per indicator (0-14)
    risk_param: float                      # float32 - global risk parameter
    tp_multiplier: float
    sl_multiplier: float
    leverage: int
    survival_generations: int = 0  # Track how many generations this bot survived
    
    # Keep risk_strategy as deprecated alias for backward compatibility
    @property
    def risk_strategy(self) -> int:
        """Deprecated: Use indicator_risk_strategies[0] instead."""
        return int(self.indicator_risk_strategies[0]) if len(self.indicator_risk_strategies) > 0 else 0
    
    def __post_init__(self):
        """Ensure survival_generations is valid."""
        try:
            sg = int(self.survival_generations)
            # Strict validation: should be in range [0, 100]
            if sg < 0 or sg > 100:  # Any invalid value gets reset to 0
                sg = 0
        except (ValueError, TypeError, OverflowError, AttributeError):
            sg = 0
        self.survival_generations = sg
    
    def __deepcopy__(self, memo):
        """Override deepcopy to ensure survival_generations is properly preserved."""
        # Create a new instance with all fields copied
        import copy
        new_bot = CompactBotConfig(
            bot_id=self.bot_id,
            num_indicators=self.num_indicators,
            indicator_indices=copy.deepcopy(self.indicator_indices, memo),
            indicator_params=copy.deepcopy(self.indicator_params, memo),
            indicator_risk_strategies=copy.deepcopy(self.indicator_risk_strategies, memo),
            risk_param=self.risk_param,
            tp_multiplier=self.tp_multiplier,
            sl_multiplier=self.sl_multiplier,
            leverage=self.leverage,
            survival_generations=self.survival_generations  # Preserve the value, __post_init__ will validate
        )
        memo[id(self)] = new_bot
        return new_bot
    
    def __eq__(self, other):
        """Compare bots by bot_id for equality."""
        if not isinstance(other, CompactBotConfig):
            return NotImplemented
        return self.bot_id == other.bot_id
    
    def __hash__(self):
        """Hash based on bot_id for set operations."""
        return hash(self.bot_id)
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict with FULL indicator data."""
        return {
            "bot_id": self.bot_id,
            "num_indicators": self.num_indicators,
            "indicator_indices": self.indicator_indices[:self.num_indicators].tolist(),
            "indicator_params": self.indicator_params[:self.num_indicators].tolist(),
            "indicator_risk_strategies": self.indicator_risk_strategies[:self.num_indicators].tolist(),
            "risk_param": float(self.risk_param),
            "tp_multiplier": float(self.tp_multiplier),
            "sl_multiplier": float(self.sl_multiplier),
            "leverage": self.leverage,
            "survival_generations": self.survival_generations
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CompactBotConfig':
        """
        Reconstruct bot from JSON dict (exact GPU config reproduction).
        
        Args:
            data: Dict from to_dict() or saved bot file
            
        Returns:
            CompactBotConfig with exact same configuration
        """
        import numpy as np
        
        # Handle both formats: direct dict or nested 'config' dict
        if 'config' in data:
            config = data['config']
            bot_id = data.get('bot_id', config.get('bot_id', 1))
            survival_generations = data.get('survival_generations', 0)
        else:
            config = data
            bot_id = config.get('bot_id', 1)
            survival_generations = config.get('survival_generations', 0)
        
        # Pad indicator arrays to 8 elements
        indicator_indices = config['indicator_indices']
        indicator_params = config.get('indicator_params', [])
        indicator_risk_strategies = config.get('indicator_risk_strategies', [])
        
        # Backward compatibility: if indicator_risk_strategies is empty but risk_strategy exists
        if not indicator_risk_strategies and 'risk_strategy' in config:
            # Use the old single risk_strategy for all indicators
            indicator_risk_strategies = [config['risk_strategy']] * 8
        
        # Ensure arrays are exactly 8 elements (pad with zeros)
        while len(indicator_indices) < 8:
            indicator_indices.append(0)
        while len(indicator_params) < 8:
            indicator_params.append([0.0, 0.0, 0.0])
        while len(indicator_risk_strategies) < 8:
            indicator_risk_strategies.append(0)
        
        return cls(
            bot_id=bot_id,
            num_indicators=config['num_indicators'],
            indicator_indices=np.array(indicator_indices[:8], dtype=np.uint8),
            indicator_params=np.array(indicator_params[:8], dtype=np.float32),
            indicator_risk_strategies=np.array(indicator_risk_strategies[:8], dtype=np.uint8),
            risk_param=config['risk_param'],
            tp_multiplier=config['tp_multiplier'],
            sl_multiplier=config['sl_multiplier'],
            leverage=config['leverage'],
            survival_generations=survival_generations
        )


class CompactBotGenerator:
    """
    Memory-optimized bot generator using compact 128-byte structs.
    90.7% memory reduction vs old architecture.
    
    Supports leverage 1-25x for safer trading (reduced from 125x).
    """
    
    def __init__(
        self,
        gpu_context: cl.Context,
        gpu_queue: cl.CommandQueue,
        population_size: int,
        min_indicators: int = 3,
        max_indicators: int = 8,
        min_risk_strategies: int = 2,
        max_risk_strategies: int = 5,
        min_leverage: int = 1,
        max_leverage: int = 125,  # Maximum leverage for testing
        random_seed: int = 42
    ):
        """Initialize compact bot generator."""
        if gpu_context is None or gpu_queue is None:
            raise RuntimeError("GPU context and queue required")
        
        # Validate inputs
        if not (1 <= min_indicators <= 8):
            raise ValueError(f"min_indicators must be 1-8, got {min_indicators}")
        if not (min_indicators <= max_indicators <= 8):
            raise ValueError(f"max_indicators must be {min_indicators}-8, got {max_indicators}")
        if not (1 <= min_risk_strategies <= 15):
            raise ValueError(f"min_risk_strategies must be 1-15, got {min_risk_strategies}")
        if not (min_risk_strategies <= max_risk_strategies <= 15):
            raise ValueError(f"max_risk_strategies must be {min_risk_strategies}-15, got {max_risk_strategies}")
        if not (1 <= min_leverage <= 125):
            raise ValueError(f"min_leverage must be 1-125, got {min_leverage}")
        if not (min_leverage <= max_leverage <= 125):
            raise ValueError(f"max_leverage must be {min_leverage}-125, got {max_leverage}")
        
        self.ctx = gpu_context
        self.queue = gpu_queue
        self.population_size = population_size
        self.min_indicators = min_indicators
        self.max_indicators = max_indicators
        self.min_risk_strategies = min_risk_strategies
        self.max_risk_strategies = max_risk_strategies
        self.min_leverage = min_leverage
        self.max_leverage = max_leverage
        self.random_seed = random_seed
        self._rng = np.random.RandomState(random_seed)  # Dedicated RNG instance
        # Track used indicator combinations across generations for diversity
        self.used_combinations = set()
        
        # Compile kernel
        self._compile_kernel()
        
        # Get indicator metadata from GPU implementation (0-49)
        self.all_indicators = get_all_gpu_indicators()
        self.num_indicators = GPU_INDICATOR_COUNT  # Exactly 50
        
        log_info(f"CompactBotGenerator initialized:")
        log_info(f"  Population: {population_size} bots")
        log_info(f"  Indicators: {min_indicators}-{max_indicators} per bot ({self.num_indicators} available)")
        log_info(f"  Risk strategies: {min_risk_strategies}-{max_risk_strategies} per bot")
        log_info(f"  Leverage: {min_leverage}-{max_leverage}x")
    
    def _compile_kernel(self):
        """Compile compact bot generation kernel."""
        kernel_path = Path(__file__).parent.parent / "gpu_kernels" / "compact_bot_gen.cl"
        
        if not kernel_path.exists():
            raise FileNotFoundError(f"Kernel not found: {kernel_path}")
        
        kernel_src = kernel_path.read_text()
        
        try:
            self.program = cl.Program(self.ctx, kernel_src).build()
            # Cache the kernel instance to avoid repeated retrieval warning
            self.generate_bots_kernel = cl.Kernel(self.program, "generate_compact_bots")
            log_info("Compiled compact_bot_gen.cl")
        except cl.RuntimeError as e:
            log_error(f"Kernel compilation failed: {e}")
            raise
    
    def _create_param_ranges(self) -> np.ndarray:
        """Create parameter ranges for all 50 indicators."""
        # Each indicator has 3 parameters with min/max ranges
        param_ranges = np.zeros((50, 3, 2), dtype=np.float32)  # [indicator][param][min/max]
        
        # Hardcoded reasonable ranges for all indicators
        # Format: [[min1, max1], [min2, max2], [min3, max3]]
        default_ranges = {
            # Moving averages (0-9): period, 0, 0
            'ma': [[5, 200], [0, 0], [0, 0]],
            # Momentum (10-19): period, threshold1, threshold2
            'momentum': [[5, 50], [20, 40], [60, 80]],
            # Volatility (20-29): period, std_dev_mult, 0
            'volatility': [[5, 50], [1.5, 3.0], [0, 0]],
            # Trend (30-39): period, 0, 0
            'trend': [[5, 50], [0, 0], [0, 0]],
            # Cycle (40-49): period, 0, 0
            'cycle': [[10, 100], [0, 0], [0, 0]]
        }
        
        for i in range(50):
            if i < 10:
                ranges = default_ranges['ma']
            elif i < 20:
                ranges = default_ranges['momentum']
            elif i < 30:
                ranges = default_ranges['volatility']
            elif i < 40:
                ranges = default_ranges['trend']
            else:
                ranges = default_ranges['cycle']
            
            for j in range(3):
                param_ranges[i, j, 0] = ranges[j][0]  # min
                param_ranges[i, j, 1] = ranges[j][1]  # max
        
        return param_ranges
    
    def generate_population(self) -> List[CompactBotConfig]:
        """Generate population of compact bots on GPU with maximum parallelization."""
        log_info(f"Generating {self.population_size} compact bots on GPU...")
        
        # Get optimal work group size
        device = self.ctx.devices[0]
        max_work_group_size = device.max_work_group_size
        # Use maximum work group size for better GPU utilization
        work_group_size = min(512, max_work_group_size)  # Use 512 or device max
        
        log_info(f"GPU: {device.name} - Using work group size: {work_group_size}/{max_work_group_size}")
        
        # Prepare parameter ranges
        param_ranges = self._create_param_ranges()
        
        # Generate random seeds using dedicated RNG
        seeds = self._rng.randint(0, 2**31, size=self.population_size, dtype=np.uint32)
        
        # Create OpenCL buffers
        bot_configs_buf = cl.Buffer(
            self.ctx,
            cl.mem_flags.WRITE_ONLY,
            size=COMPACT_BOT_SIZE * self.population_size
        )
        
        param_ranges_buf = cl.Buffer(
            self.ctx,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=param_ranges.astype(np.float32)
        )
        
        seeds_buf = cl.Buffer(
            self.ctx,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=seeds
        )
        
        # Execute kernel with optimized work group size for maximum parallelization
        kernel = self.generate_bots_kernel  # Use cached kernel instance
        
        # Round up global size to be multiple of work group size for optimal GPU utilization
        global_size = ((self.population_size + work_group_size - 1) // work_group_size) * work_group_size
        local_size = work_group_size
        
        log_info(f"Launching GPU kernel: {global_size} work items, {local_size} per group ({global_size//local_size} groups)")
        
        kernel(
            self.queue,
            (global_size,),
            (local_size,),
            bot_configs_buf,
            param_ranges_buf,
            seeds_buf,
            np.int32(self.min_indicators),
            np.int32(self.max_indicators),
            np.int32(self.min_risk_strategies),
            np.int32(self.max_risk_strategies),
            np.int32(self.min_leverage),  # NEW: min leverage
            np.int32(self.max_leverage)   # UPDATED: max leverage
        )
        
        self.queue.finish()
        log_info(f"GPU kernel completed - processing results...")
        
        # Read results
        bot_configs_raw = np.empty(self.population_size * COMPACT_BOT_SIZE, dtype=np.uint8)
        cl.enqueue_copy(self.queue, bot_configs_raw, bot_configs_buf)
        
        # Parse bots
        bots = self._parse_bots(bot_configs_raw)
        
        # Remove or replace non-directional indicators and ensure uniqueness
        self._enforce_directional_and_unique(bots)
        log_info(f"[OK] Generated {len(bots)} compact bots (128 bytes each) - directional enforced, unique combos updated")
        
        return bots
    
    def generate_single_bot(self, bot_id: int) -> CompactBotConfig:
        """
        Generate a single compact bot on GPU.
        
        Args:
            bot_id: ID to assign to the bot
            
        Returns:
            Single CompactBotConfig
        """
        # Prepare parameter ranges
        param_ranges = self._create_param_ranges()
        
        # Generate random seed using dedicated RNG to ensure unique bots
        seed = self._rng.randint(0, 2**31, dtype=np.uint32)
        seeds = np.array([seed], dtype=np.uint32)
        
        # Create OpenCL buffers for single bot
        bot_configs_buf = cl.Buffer(
            self.ctx,
            cl.mem_flags.WRITE_ONLY,
            size=COMPACT_BOT_SIZE
        )
        
        param_ranges_buf = cl.Buffer(
            self.ctx,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=param_ranges.astype(np.float32)
        )
        
        seeds_buf = cl.Buffer(
            self.ctx,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=seeds
        )
        
        # Execute kernel for single bot
        kernel = self.generate_bots_kernel  # Use cached kernel instance
        global_size = (1,)  # Generate just 1 bot
        
        kernel(
            self.queue,
            global_size,
            None,
            bot_configs_buf,
            param_ranges_buf,
            seeds_buf,
            np.int32(self.min_indicators),
            np.int32(self.max_indicators),
            np.int32(self.min_risk_strategies),
            np.int32(self.max_risk_strategies),
            np.int32(self.min_leverage),
            np.int32(self.max_leverage)
        )
        
        self.queue.finish()
        
        # Read result
        bot_config_raw = np.empty(COMPACT_BOT_SIZE, dtype=np.uint8)
        cl.enqueue_copy(self.queue, bot_config_raw, bot_configs_buf)
        
        # Parse single bot
        bots = self._parse_bots(bot_config_raw)
        
        if not bots:
            raise RuntimeError("Failed to generate bot")
        
        # Update bot_id to requested value
        bot = bots[0]
        bot.bot_id = bot_id
        
        return bot
    
    def _parse_bots(self, raw_data: np.ndarray) -> List[CompactBotConfig]:
        """Parse compact bot structs from raw bytes."""
        bots = []
        
        # Define struct layout (must match OpenCL exactly - 132 bytes total)
        dt = np.dtype([
            ('bot_id', np.int32),                                      # 4 bytes (offset 0)
            ('num_indicators', np.uint8),                              # 1 byte (offset 4)
            ('indicator_indices', np.uint8, 8),                        # 8 bytes (offset 5)
            ('indicator_params', np.float32, (8, 3)),                  # 96 bytes (offset 13)
            ('indicator_risk_strategies', np.uint8, 8),                # 8 bytes (offset 109)
            ('risk_param', np.float32),                                # 4 bytes (offset 117)
            ('tp_multiplier', np.float32),                             # 4 bytes (offset 121)
            ('sl_multiplier', np.float32),                             # 4 bytes (offset 125)
            ('leverage', np.uint8),                                    # 1 byte (offset 129)
            ('padding', np.uint8, 2)                                   # 2 bytes (offset 130)
        ])  # Total: 132 bytes
        
        # Parse all bots
        structured = np.frombuffer(raw_data, dtype=dt)
        
        for bot_struct in structured:
            bot = CompactBotConfig(
                bot_id=int(bot_struct['bot_id']),
                num_indicators=int(bot_struct['num_indicators']),
                indicator_indices=bot_struct['indicator_indices'].copy(),
                indicator_params=bot_struct['indicator_params'].copy(),
                indicator_risk_strategies=bot_struct['indicator_risk_strategies'].copy(),
                risk_param=float(bot_struct['risk_param']),
                tp_multiplier=float(bot_struct['tp_multiplier']),
                sl_multiplier=float(bot_struct['sl_multiplier']),
                leverage=int(bot_struct['leverage']),
                survival_generations=0  # New bots start with 0
            )
            bots.append(bot)
        # Ensure every bot contains at least one directional-producing indicator
        self._ensure_directional_indicators(bots)

        return bots

    def _ensure_directional_indicators(self, bots: List[CompactBotConfig]):
        """
        Ensure every bot includes at least one directional-producing indicator.
        Directional indicators include moving averages, momentum, and trend indicators.
        This avoids generating bots that may produce zero trades across cycles due to
        neutral indicator combos.
        """
        # Directional indicator index ranges: MA-family (0-11), Momentum (12-19), Trend (26-35)
        directional_indices = set(list(range(0, 12)) + list(range(12, 20)) + list(range(26, 36)))
        for bot in bots:
            # Ensure min_indicators enforced (some CSV bots may have fewer)
            if bot.num_indicators < self.min_indicators:
                for i in range(bot.num_indicators, min(self.min_indicators, MAX_INDICATORS_PER_BOT)):
                    # set to directional indicator
                    rep = int(self._rng.choice(directional_choices))
                    bot.indicator_indices[i] = np.uint8(rep)
                    bot.indicator_params[i][0] = 10.0
                bot.num_indicators = min(self.min_indicators, MAX_INDICATORS_PER_BOT)
            has_directional = False
            for idx in bot.indicator_indices[:bot.num_indicators]:
                if int(idx) in directional_indices and idx != 0:
                    has_directional = True
                    break
            if not has_directional:
                # Replace the first indicator slot with a known directional indicator (MACD index 26)
                pos = 0
                if bot.num_indicators > 0:
                    pos = 0
                replacement = 26  # MACD index - trend-producing
                # Mutate indicator list in-place
                bot.indicator_indices[pos] = np.uint8(replacement)
                bot.indicator_params[pos] = np.array([12.0, 26.0, 9.0], dtype=np.float32)  # MACD params
        # Additional tuning: ensure at least one short-period indicator exists (increases activity on 1m)
        for bot in bots:
            tuned = False
            for i in range(bot.num_indicators):
                idx = int(bot.indicator_indices[i])
                # If indicator is moving average/EMA family, set a short period (e.g., 5-20)
                if 0 <= idx <= 11 and not tuned:
                    bot.indicator_params[i][0] = 10.0
                    tuned = True
                # If indicator is MACD, set to common sensitive fast/slow/signal
                if idx == 26 and not tuned:
                    bot.indicator_params[i] = np.array([12.0, 26.0, 9.0], dtype=np.float32)
                    tuned = True
            # If no tune applied, forcibly set first indicator to a short MA
            if not tuned and bot.num_indicators > 0:
                bot.indicator_indices[0] = np.uint8(0)  # SMA/EMA index 0
                bot.indicator_params[0][0] = 10.0
            # Set risk strategy to fixed percent if many bots show no activity
            if bot.num_indicators > 0:
                bot.indicator_risk_strategies[0] = np.uint8(0)  # RISK_FIXED_PCT
                bot.risk_param = max(bot.risk_param, 0.02)  # 2% risk per trade by default
                if bot.leverage > 50:
                    bot.leverage = min(10, bot.leverage)  # Reduce leverage for stability

    def _bot_signature(self, bot: CompactBotConfig):
        """Create a hashable signature for uniqueness checks based on indices and rounded parameters."""
        # Use first parameter rounded to integer for signature stability and indices list
        idxs = tuple(int(i) for i in bot.indicator_indices[:bot.num_indicators])
        params = tuple(int(round(float(bot.indicator_params[i][0]))) for i in range(bot.num_indicators))
        return (bot.num_indicators, idxs, params, int(bot.leverage), round(bot.risk_param, 3))

    def _enforce_directional_and_unique(self, bots: List[CompactBotConfig]):
        """Enforce directional-only indicators and uniqueness across generated bots.

        This mutates the bot list in-place, clears non-directional indicators by replacing them
        with directional ones, and ensures uniqueness of indicator combinations using a
        per-population `seen` set and the `used_combinations` cache.
        """
        # Directional indicator index ranges: MA-family (0-11), Momentum (12-19), Trend (26-35), MACD (26)
        directional_indices = list(range(0, 12)) + list(range(12, 20)) + list(range(26, 36))
        directional_choices = [np.uint8(i) for i in directional_indices if i < self.num_indicators]
        if not directional_choices:
            directional_choices = [np.uint8(26)]  # fallback to MACD if meta-data incorrect

        seen = set()
        for bot in bots:
            # Ensure min_indicators enforced (some CSV bots may have fewer)
            if bot.num_indicators < self.min_indicators:
                for i in range(bot.num_indicators, min(self.min_indicators, MAX_INDICATORS_PER_BOT)):
                    # set to directional indicator
                    rep = int(self._rng.choice(directional_choices))
                    bot.indicator_indices[i] = np.uint8(rep)
                    bot.indicator_params[i][0] = 10.0
                bot.num_indicators = min(self.min_indicators, MAX_INDICATORS_PER_BOT)

            # Replace any non-directional indicators with directional ones
            for i in range(bot.num_indicators):
                idx = int(bot.indicator_indices[i])
                if idx not in directional_indices or idx == 0:
                    # pick a random directional indicator to replace
                    replacement = int(self._rng.choice(directional_choices))
                    bot.indicator_indices[i] = np.uint8(replacement)
                    # Set a short period default where applicable
                    bot.indicator_params[i][0] = 10.0

            # Ensure at least one directional indicator and short period tuning
            self._ensure_directional_indicators([bot])

            # Now ensure uniqueness across population and global cache
            sig = self._bot_signature(bot)
            tries = 0
            while (sig in seen or sig in self.used_combinations) and tries < 50:
                # Apply mutation: change first indicator to a different directional one
                pos = 0 if bot.num_indicators > 0 else 0
                new_idx = int(self._rng.choice(directional_choices))
                bot.indicator_indices[pos] = np.uint8(new_idx)
                bot.indicator_params[pos][0] = float(max(5.0, int(self._rng.randint(5, 25))))
                sig = self._bot_signature(bot)
                tries += 1
            seen.add(sig)
            self.used_combinations.add(sig)
    
    def estimate_vram(self) -> dict:
        """Estimate VRAM usage."""
        bot_configs_mb = (self.population_size * COMPACT_BOT_SIZE) / (1024 * 1024)
        param_ranges_mb = (50 * 6 * 4) / (1024 * 1024)
        seeds_mb = (self.population_size * 4) / (1024 * 1024)
        total_mb = bot_configs_mb + param_ranges_mb + seeds_mb
        
        return {
            'bot_configs_mb': bot_configs_mb,
            'param_ranges_mb': param_ranges_mb,
            'seeds_mb': seeds_mb,
            'total_mb': total_mb,
            'total_bytes': int(total_mb * 1024 * 1024)
        }

    def clear_used_combinations(self):
        """Clear the global used combinations cache (for refilling or fresh generation)."""
        self.used_combinations.clear()
