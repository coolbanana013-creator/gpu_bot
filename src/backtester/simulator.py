"""
GPU-accelerated backtesting simulator using OpenCL.
100% GPU implementation - no CPU fallback.
"""
import numpy as np
import pyopencl as cl
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from ..bot_generator.generator import BotConfig
from ..utils.validation import log_info, log_debug, log_error
from ..utils.vram_estimator import VRAMEstimator


@dataclass
class BacktestResult:
    """Results from backtesting a single bot."""
    bot_id: int
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    final_balance: float
    win_rate: float


class GPUBacktester:
    """
    GPU-accelerated backtester using OpenCL.
    Executes backtest_impl.cl and precompute_indicators.cl kernels.
    """
    
    def __init__(
        self,
        gpu_context: cl.Context,
        gpu_queue: cl.CommandQueue,
        initial_balance: float = 10000.0
    ):
        """
        Initialize GPU backtester.
        
        Args:
            gpu_context: OpenCL context (mandatory)
            gpu_queue: OpenCL command queue (mandatory)
            initial_balance: Starting balance for backtests
        """
        if gpu_context is None or gpu_queue is None:
            raise RuntimeError("GPU context and queue are required. GPU-only mode.")
        
        self.ctx = gpu_context
        self.queue = gpu_queue
        self.initial_balance = initial_balance
        
        # Compile kernels
        self._compile_kernels()
        
        log_info("GPU Backtester initialized (100% GPU mode)")
    
    def _compile_kernels(self):
        """Compile OpenCL kernels."""
        kernel_dir = Path(__file__).parent.parent / "gpu_kernels"
        
        # Compile precompute indicators kernel
        precompute_path = kernel_dir / "precompute_indicators.cl"
        if not precompute_path.exists():
            raise FileNotFoundError(f"Precompute kernel not found: {precompute_path}")
        
        precompute_src = precompute_path.read_text()
        try:
            self.precompute_program = cl.Program(self.ctx, precompute_src).build()
            log_debug("Compiled precompute_indicators.cl")
        except cl.RuntimeError as e:
            log_error(f"Failed to compile precompute_indicators.cl: {e}")
            raise
        
        # Compile backtest kernel
        backtest_path = kernel_dir / "backtest_impl.cl"
        if not backtest_path.exists():
            raise FileNotFoundError(f"Backtest kernel not found: {backtest_path}")
        
        backtest_src = backtest_path.read_text()
        try:
            self.backtest_program = cl.Program(self.ctx, backtest_src).build()
            log_debug("Compiled backtest_impl.cl")
        except cl.RuntimeError as e:
            log_error(f"Failed to compile backtest_impl.cl: {e}")
            raise
    
    def _precompute_indicators(
        self,
        ohlcv_data: np.ndarray,
        indicator_types: np.ndarray,
        num_indicator_types: int
    ) -> np.ndarray:
        """
        Precompute all indicator values on GPU.
        
        Args:
            ohlcv_data: OHLCV bars [num_bars, 5] (open, high, low, close, volume)
            indicator_types: Unique indicator types [num_types]
            num_indicator_types: Number of unique types
            
        Returns:
            Precomputed indicator values [num_bars, num_types, MAX_PARAMS]
        """
        num_bars = ohlcv_data.shape[0]
        MAX_PARAMS = 10
        
        # Allocate output buffer
        indicator_values = np.zeros((num_bars, num_indicator_types, MAX_PARAMS), dtype=np.float32)
        
        # Create buffers
        mf = cl.mem_flags
        ohlcv_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ohlcv_data.astype(np.float32))
        types_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=indicator_types.astype(np.int32))
        output_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, indicator_values.nbytes)
        
        # Execute kernel
        kernel = self.precompute_program.precompute_indicators
        global_size = (num_bars,)
        kernel(
            self.queue, global_size, None,
            ohlcv_buf,
            types_buf,
            output_buf,
            np.int32(num_bars),
            np.int32(num_indicator_types)
        )
        
        # Read results
        cl.enqueue_copy(self.queue, indicator_values, output_buf)
        self.queue.finish()
        
        # Validate (check for NaN/Inf)
        error_count = np.zeros(1, dtype=np.int32)
        error_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=error_count)
        
        validate_kernel = self.precompute_program.validate_indicators
        validate_kernel(
            self.queue, global_size, None,
            output_buf,
            error_buf,
            np.int32(num_bars),
            np.int32(num_indicator_types)
        )
        
        cl.enqueue_copy(self.queue, error_count, error_buf)
        self.queue.finish()
        
        if error_count[0] > 0:
            log_error(f"Precomputed indicators contain {error_count[0]} NaN/Inf values!")
            raise RuntimeError("Indicator precomputation failed validation")
        
        log_debug(f"Precomputed {num_indicator_types} indicators for {num_bars} bars")
        return indicator_values
    
    def backtest_bots(
        self,
        bots: List[BotConfig],
        ohlcv_data: np.ndarray,
        cycle_starts: np.ndarray,
        cycle_ends: np.ndarray
    ) -> List[BacktestResult]:
        """
        Backtest multiple bots on GPU.
        
        Args:
            bots: List of BotConfig objects
            ohlcv_data: OHLCV bars [num_bars, 5]
            cycle_starts: Start indices for each cycle [num_cycles]
            cycle_ends: End indices for each cycle [num_cycles]
            
        Returns:
            List of BacktestResult objects
        """
        num_bots = len(bots)
        num_bars = ohlcv_data.shape[0]
        num_cycles = len(cycle_starts)
        
        log_info(f"Backtesting {num_bots} bots on {num_bars} bars with {num_cycles} cycles")
        
        # Validate VRAM
        estimator = VRAMEstimator(self.ctx, self.queue)
        required_vram = estimator.estimate_backtesting_vram(
            population_size=num_bots,
            num_cycles=num_cycles,
            total_bars=num_bars,
            num_indicator_types=20  # Conservative estimate
        )
        
        # Get available VRAM
        if estimator.device_vram:
            estimator.validate_vram_availability(required_vram['total_bytes'], estimator.device_vram)
        # else skip validation if device VRAM unknown
        
        # Serialize bot configs to GPU format
        bot_configs_bytes = self._serialize_bot_configs(bots)
        
        # Extract unique indicator types from all bots (as integers)
        all_indicator_type_values = set()
        for bot in bots:
            for ind in bot.indicators:
                # Map indicator type to integer using type_map
                ind_int = self._indicator_type_to_int(ind.indicator_type)
                all_indicator_type_values.add(ind_int)
        
        unique_types = np.array(sorted(all_indicator_type_values), dtype=np.int32)
        num_indicator_types = len(unique_types)
        
        # Precompute indicators
        log_debug("Precomputing indicators on GPU...")
        precomputed_indicators = self._precompute_indicators(
            ohlcv_data,
            unique_types,
            num_indicator_types
        )
        
        # Prepare buffers
        BOT_CONFIG_SIZE = 1344  # Bytes per bot config
        BACKTEST_RESULT_SIZE = 48  # Bytes per result
        
        results_bytes = np.zeros(num_bots * BACKTEST_RESULT_SIZE, dtype=np.uint8)
        
        mf = cl.mem_flags
        bot_configs_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bot_configs_bytes)
        ohlcv_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ohlcv_data.astype(np.float32))
        precomputed_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=precomputed_indicators)
        cycle_starts_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cycle_starts.astype(np.int32))
        cycle_ends_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cycle_ends.astype(np.int32))
        results_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, results_bytes.nbytes)
        
        # Execute backtest kernel
        log_debug("Executing backtest kernel...")
        kernel = self.backtest_program.backtest_bots
        global_size = (num_bots,)
        
        kernel(
            self.queue, global_size, None,
            bot_configs_buf,                    # 1. bot_configs
            precomputed_buf,                    # 2. precomputed_indicators
            ohlcv_buf,                          # 3. ohlcv_data
            cycle_starts_buf,                   # 4. cycle_starts
            cycle_ends_buf,                     # 5. cycle_ends
            np.int32(num_cycles),               # 6. num_cycles
            np.int32(num_bars),                 # 7. total_bars
            np.int32(num_indicator_types),      # 8. num_indicator_types
            np.float32(self.initial_balance),   # 9. initial_balance
            results_buf                         # 10. results
        )
        
        # Read results
        cl.enqueue_copy(self.queue, results_bytes, results_buf)
        self.queue.finish()
        
        log_info("Backtest kernel execution complete")
        
        # Parse results
        results = self._parse_backtest_results(results_bytes, num_bots)
        
        return results
    
    def _serialize_bot_configs(self, bots: List[BotConfig]) -> np.ndarray:
        """
        Serialize bot configs to GPU format.
        
        BotConfig struct layout (1,344 bytes):
        - bot_id: int (4 bytes)
        - num_indicators: int (4 bytes)
        - indicator_types: int[20] (80 bytes)
        - indicator_params: float[20][10] (800 bytes)
        - num_risk_strategies: int (4 bytes)
        - risk_strategy_types: int[10] (40 bytes)
        - risk_strategy_params: float[10][10] (400 bytes)
        - take_profit_pct: float (4 bytes)
        - stop_loss_pct: float (4 bytes)
        - leverage: int (4 bytes)
        """
        BOT_CONFIG_SIZE = 1344
        num_bots = len(bots)
        data = np.zeros(num_bots * BOT_CONFIG_SIZE, dtype=np.uint8)
        
        for i, bot in enumerate(bots):
            offset = i * BOT_CONFIG_SIZE
            
            # bot_id
            np.frombuffer(data, dtype=np.int32, count=1, offset=offset)[0] = bot.bot_id
            offset += 4
            
            # num_indicators
            num_indicators = len(bot.indicators)
            np.frombuffer(data, dtype=np.int32, count=1, offset=offset)[0] = num_indicators
            offset += 4
            
            # indicator_types (20 slots)
            ind_types = np.zeros(20, dtype=np.int32)
            for j, ind in enumerate(bot.indicators[:20]):
                ind_types[j] = self._indicator_type_to_int(ind.indicator_type)
            np.frombuffer(data, dtype=np.int32, count=20, offset=offset)[:] = ind_types
            offset += 80
            
            # indicator_params (20x10 floats)
            ind_params = np.zeros((20, 10), dtype=np.float32)
            for j, ind in enumerate(bot.indicators[:20]):
                params_list = self._params_dict_to_array(ind.params, 10)
                ind_params[j] = params_list
            np.frombuffer(data, dtype=np.float32, count=200, offset=offset)[:] = ind_params.flatten()
            offset += 800
            
            # num_risk_strategies
            num_risks = len(bot.risk_strategies)
            np.frombuffer(data, dtype=np.int32, count=1, offset=offset)[0] = num_risks
            offset += 4
            
            # risk_strategy_types (10 slots)
            risk_types = np.zeros(10, dtype=np.int32)
            for j, risk in enumerate(bot.risk_strategies[:10]):
                risk_types[j] = self._risk_type_to_int(risk.strategy_type)
            np.frombuffer(data, dtype=np.int32, count=10, offset=offset)[:] = risk_types
            offset += 40
            
            # risk_strategy_params (10x10 floats)
            risk_params = np.zeros((10, 10), dtype=np.float32)
            for j, risk in enumerate(bot.risk_strategies[:10]):
                params_list = self._params_dict_to_array(risk.params, 10)
                risk_params[j] = params_list
            np.frombuffer(data, dtype=np.float32, count=100, offset=offset)[:] = risk_params.flatten()
            offset += 400
            
            # take_profit_pct
            np.frombuffer(data, dtype=np.float32, count=1, offset=offset)[0] = bot.take_profit_pct
            offset += 4
            
            # stop_loss_pct
            np.frombuffer(data, dtype=np.float32, count=1, offset=offset)[0] = bot.stop_loss_pct
            offset += 4
            
            # leverage
            np.frombuffer(data, dtype=np.int32, count=1, offset=offset)[0] = bot.leverage
        
        return data
    
    def _indicator_type_to_int(self, ind_type) -> int:
        """Map IndicatorType to integer for GPU."""
        # Match the enum ordering in OpenCL kernel
        type_map = {
            "RSI": 0, "MACD": 1, "STOCH": 2, "BBANDS": 3,
            "EMA": 4, "SMA": 5, "ATR": 6, "CCI": 7,
            "MOM": 8, "ROC": 9, "WILLIAMS_R": 10, "ADX": 11,
            "AROON": 12, "TRIX": 13, "OBV": 14, "AD": 15,
            "ADOSC": 16, "MFI": 17, "SAR": 18, "BETA": 19,
            "CORREL": 20, "LINEARREG": 21, "STDDEV": 22, "TSF": 23,
            "WMA": 24, "DEMA": 25, "TEMA": 26, "PPO": 27,
            "ULTIMATE_OSC": 28, "STOCHF": 29, "RSI_STOCH": 30,
            "VAR": 31, "HT_DCPERIOD": 32, "HT_DCPHASE": 33,
            "HT_TRENDMODE": 34, "KAMA": 35, "MIDPOINT": 36,
            "MIDPRICE": 37, "PLUS_DI": 38, "MINUS_DI": 39,
            "SAREXT": 40, "LINEARREG_ANGLE": 41, "LINEARREG_INTERCEPT": 42,
            "LINEARREG_SLOPE": 43, "TRIMA": 44, "T3": 45,
            "ADXR": 46, "AROONOSC": 47, "DX": 48, "NATR": 49
        }
        return type_map.get(ind_type.value, 0)
    
    def _risk_type_to_int(self, risk_type) -> int:
        """Map RiskStrategyType to integer for GPU."""
        type_map = {
            "FIXED_PERCENT": 0, "FIXED_AMOUNT": 1, "KELLY_FULL": 2,
            "KELLY_HALF": 3, "KELLY_FRACTIONAL": 4, "VOLATILITY_ATR": 5,
            "VOLATILITY_STDDEV": 6, "MARTINGALE": 7, "ANTI_MARTINGALE": 8,
            "EQUITY_CURVE": 9, "WIN_STREAK": 10, "PROGRESSIVE": 11,
            "LOSS_STREAK": 12, "MAX_DRAWDOWN": 13, "RISK_REWARD_RATIO": 14
        }
        return type_map.get(risk_type.value, 0)
    
    def _params_dict_to_array(self, params: Dict[str, Any], max_len: int) -> np.ndarray:
        """Convert parameter dict to fixed-length float array."""
        arr = np.zeros(max_len, dtype=np.float32)
        for i, (key, value) in enumerate(params.items()):
            if i >= max_len:
                break
            arr[i] = float(value)
        return arr
    
    def _parse_backtest_results(self, results_bytes: np.ndarray, num_bots: int) -> List[BacktestResult]:
        """
        Parse BacktestResult structs from GPU.
        
        BacktestResult struct layout (48 bytes):
        - bot_id: int (4 bytes)
        - total_trades: int (4 bytes)
        - winning_trades: int (4 bytes)
        - losing_trades: int (4 bytes)
        - total_return_pct: float (4 bytes)
        - sharpe_ratio: float (4 bytes)
        - max_drawdown_pct: float (4 bytes)
        - final_balance: float (4 bytes)
        - win_rate: float (4 bytes)
        - padding: 12 bytes
        """
        RESULT_SIZE = 48
        results = []
        
        for i in range(num_bots):
            offset = i * RESULT_SIZE
            
            bot_id = np.frombuffer(results_bytes, dtype=np.int32, count=1, offset=offset)[0]
            offset += 4
            
            total_trades = np.frombuffer(results_bytes, dtype=np.int32, count=1, offset=offset)[0]
            offset += 4
            
            winning_trades = np.frombuffer(results_bytes, dtype=np.int32, count=1, offset=offset)[0]
            offset += 4
            
            losing_trades = np.frombuffer(results_bytes, dtype=np.int32, count=1, offset=offset)[0]
            offset += 4
            
            total_return_pct = np.frombuffer(results_bytes, dtype=np.float32, count=1, offset=offset)[0]
            offset += 4
            
            sharpe_ratio = np.frombuffer(results_bytes, dtype=np.float32, count=1, offset=offset)[0]
            offset += 4
            
            max_drawdown_pct = np.frombuffer(results_bytes, dtype=np.float32, count=1, offset=offset)[0]
            offset += 4
            
            final_balance = np.frombuffer(results_bytes, dtype=np.float32, count=1, offset=offset)[0]
            offset += 4
            
            win_rate = np.frombuffer(results_bytes, dtype=np.float32, count=1, offset=offset)[0]
            
            results.append(BacktestResult(
                bot_id=int(bot_id),
                total_trades=int(total_trades),
                winning_trades=int(winning_trades),
                losing_trades=int(losing_trades),
                total_return_pct=float(total_return_pct),
                sharpe_ratio=float(sharpe_ratio),
                max_drawdown_pct=float(max_drawdown_pct),
                final_balance=float(final_balance),
                win_rate=float(win_rate)
            ))
        
        log_debug(f"Parsed {num_bots} backtest results")
        return results
