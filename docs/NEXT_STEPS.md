# Next Steps - Quick Reference

## ⚡ Priority Tasks (in order)

### 1. Complete Bot Generator Struct Parser (2-3 hours)
**File**: `src/bot_generator/generator_gpu.py`
**Method**: `_parse_bot_configs(raw_data: np.ndarray)`

**Current**: Placeholder that creates dummy configs
**Need**: Parse OpenCL struct bytes into Python BotConfig objects

**Struct Layout** (from `bot_gen_impl.cl`):
```c
typedef struct {
    int bot_id;                                      // 4 bytes
    int num_indicators;                              // 4 bytes
    int indicator_types[20];                         // 80 bytes
    float indicator_params[20 * 10];                 // 800 bytes
    int num_risk_strategies;                         // 4 bytes
    int risk_strategy_types[10];                     // 40 bytes
    float risk_strategy_params[10 * 10];             // 400 bytes
    float take_profit_pct;                           // 4 bytes
    float stop_loss_pct;                             // 4 bytes
    int leverage;                                    // 4 bytes
} BotConfig;  // Total: 1,344 bytes
```

**Parsing Code**:
```python
def _parse_bot_configs(self, raw_data: np.ndarray) -> List[BotConfig]:
    configs = []
    offset = 0
    struct_size = 1344  # bytes
    
    for i in range(self.population_size):
        # Extract struct bytes
        bot_bytes = raw_data[offset:offset + struct_size]
        
        # Parse fields
        bot_id = np.frombuffer(bot_bytes[0:4], dtype=np.int32)[0]
        num_inds = np.frombuffer(bot_bytes[4:8], dtype=np.int32)[0]
        ind_types = np.frombuffer(bot_bytes[8:88], dtype=np.int32)[:num_inds]
        ind_params = np.frombuffer(bot_bytes[88:888], dtype=np.float32).reshape(20, 10)[:num_inds]
        
        num_risks = np.frombuffer(bot_bytes[888:892], dtype=np.int32)[0]
        risk_types = np.frombuffer(bot_bytes[892:932], dtype=np.int32)[:num_risks]
        risk_params = np.frombuffer(bot_bytes[932:1332], dtype=np.float32).reshape(10, 10)[:num_risks]
        
        tp_pct = np.frombuffer(bot_bytes[1332:1336], dtype=np.float32)[0]
        sl_pct = np.frombuffer(bot_bytes[1336:1340], dtype=np.float32)[0]
        leverage = np.frombuffer(bot_bytes[1340:1344], dtype=np.int32)[0]
        
        # Create IndicatorParams objects
        indicators = []
        for j in range(num_inds):
            ind_type = IndicatorType(ind_types[j])
            params = {f'param{k}': ind_params[j, k] for k in range(10)}
            indicators.append(IndicatorParams(ind_type, params))
        
        # Create RiskStrategyParams objects
        strategies = []
        for j in range(num_risks):
            risk_type = RiskStrategyType(risk_types[j])
            params = {f'param{k}': risk_params[j, k] for k in range(10)}
            strategies.append(RiskStrategyParams(risk_type, params))
        
        # Create BotConfig
        bot = BotConfig(
            bot_id=bot_id,
            indicators=indicators,
            risk_strategies=strategies,
            take_profit_pct=tp_pct,
            stop_loss_pct=sl_pct,
            leverage=leverage
        )
        configs.append(bot)
        
        offset += struct_size
    
    return configs
```

---

### 2. Create GPU Backtester Host (4-6 hours)
**File**: `src/backtester/simulator_gpu.py` (create new)

**Template** (based on generator_gpu.py):
```python
import pyopencl as cl
import numpy as np
from typing import List
from dataclasses import dataclass

@dataclass
class BacktestResult:
    bot_id: int
    cycle_id: int
    final_balance: float
    profit_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    max_drawdown_pct: float

class Backtester:
    def __init__(self, initial_balance: float, gpu_context, gpu_queue):
        if gpu_context is None or gpu_queue is None:
            raise RuntimeError("GPU context required")
        
        self.initial_balance = initial_balance
        self.gpu_context = gpu_context
        self.gpu_queue = gpu_queue
        self._compile_kernel()
    
    def _compile_kernel(self):
        kernel_path = 'src/gpu_kernels/backtest_impl.cl'
        with open(kernel_path, 'r') as f:
            source = f.read()
        self.program = cl.Program(self.gpu_context, source).build()
    
    def backtest_population(self, bots, ohlcv_data, cycle_ranges):
        # Create buffers
        # Execute kernel
        # Parse results
        pass
```

**Key Steps**:
1. Load `backtest_impl.cl` and compile
2. Convert Python BotConfig → OpenCL struct bytes
3. Convert OHLCV DataFrame → float array
4. Create cycle_starts/cycle_ends int arrays
5. Create buffers (bots, ohlcv, cycles, results)
6. Execute kernel (global size = num_bots)
7. Read results buffer
8. Parse BacktestResult structs

---

### 3. Create Precompute Indicators Kernel (2-3 hours)
**File**: `src/gpu_kernels/precompute_indicators.cl` (create new)

**Purpose**: Compute all indicators once before backtesting

**Kernel Signature**:
```c
__kernel void precompute_indicators(
    __global float *ohlcv_data,           // Input: [total_bars, 5] OHLCV
    __global float *indicator_values,     // Output: [total_bars, num_types, MAX_PARAMS]
    const int total_bars,
    const int num_indicator_types
) {
    int bar = get_global_id(0);
    int ind_type = get_global_id(1);
    
    if (bar >= total_bars || ind_type >= num_indicator_types) return;
    
    // Compute indicator based on type
    // Store in indicator_values[bar * num_types * MAX_PARAMS + ind_type * MAX_PARAMS + ...]
}
```

**2D Work Size**: `(total_bars, num_indicator_types)`

**Integration**: Call this in simulator_gpu before backtest kernel

---

### 4. Expand to 50+ Indicators (3-4 hours)

**Add to `src/indicators/factory.py`**:
```python
class HT_TRENDLINE(Indicator):
    """Hilbert Transform - Instantaneous Trendline"""
    def generate_params(self):
        return {}  # No parameters

class KAMA(Indicator):
    """Kaufman Adaptive Moving Average"""
    def generate_params(self):
        return {
            'timeperiod': np.random.randint(10, 50)
        }

# Add 18 more...
```

**Add to `src/indicators/signals.py`**:
```python
def compute_ht_trendline_signal(value, price):
    return Signal.LONG if price > value else Signal.SHORT

# Add 18 more...
```

**Add to `backtest_impl.cl`**:
```c
case 30:  // HT_TRENDLINE
    sig.direction = (current_price > precomputed[base_idx]) ? 1 : -1;
    break;

// Add 18 more cases...
```

---

### 5. Expand to 15+ Risk Strategies (1-2 hours)

**Add to `src/risk_management/strategies.py`**:
```python
class RuinBasedSizing(RiskStrategy):
    def generate_params(self):
        return {
            'max_ruin_probability': np.random.uniform(0.01, 0.10),
            'win_rate_estimate': np.random.uniform(0.45, 0.65)
        }

class OptimalF(RiskStrategy):
    def generate_params(self):
        return {
            'max_loss': np.random.uniform(0.01, 0.05),
            'fraction': np.random.uniform(0.1, 0.3)
        }

# Add 1-3 more...
```

---

### 6. Testing Suite (6-8 hours)

**Create `test/test_generator_gpu.py`**:
```python
import pytest
import pyopencl as cl
from src.bot_generator.generator_gpu import BotGenerator

@pytest.fixture
def gpu_context():
    return cl.create_some_context()

@pytest.fixture
def gpu_queue(gpu_context):
    return cl.CommandQueue(gpu_context)

def test_kernel_compilation(gpu_context, gpu_queue):
    gen = BotGenerator(
        population_size=100,
        gpu_context=gpu_context,
        gpu_queue=gpu_queue
    )
    assert gen.program is not None

def test_bot_generation(gpu_context, gpu_queue):
    gen = BotGenerator(
        population_size=1000,
        random_seed=42,
        gpu_context=gpu_context,
        gpu_queue=gpu_queue
    )
    bots = gen.generate_population()
    assert len(bots) == 1000
    assert all(b.bot_id >= 0 for b in bots)

def test_no_gpu_crashes():
    with pytest.raises(RuntimeError, match="GPU context"):
        BotGenerator(population_size=100, gpu_context=None, gpu_queue=None)
```

**Similarly**: Create test_simulator_gpu.py, test_workflow.py, etc.

---

### 7. Documentation Updates (2-3 hours)

**Update `README.md`** - Add GPU requirements section:
```markdown
## GPU Requirements

### Mandatory Requirements
- OpenCL-capable GPU (NVIDIA, AMD, or Intel)
- OpenCL drivers installed
- Minimum 2GB VRAM (8GB+ recommended for large populations)

### Installation

#### NVIDIA
```bash
# Install CUDA Toolkit (includes OpenCL)
# Download from: https://developer.nvidia.com/cuda-downloads
```

#### AMD
```bash
# Install ROCm (includes OpenCL)
sudo apt-get install rocm-opencl
```

#### Intel
```bash
# Install Intel OpenCL Runtime
# Download from: https://software.intel.com/content/www/us/en/develop/tools/opencl-sdk.html
```

### VRAM Guidelines
- 10k bots, 10 cycles, 7 days: ~450 MB
- 100k bots, 20 cycles, 30 days: ~4.5 GB
- 1M bots, 50 cycles, 90 days: ~45 GB (multi-GPU required)
```

---

## 🧪 Testing Checklist

Before considering implementation complete:

- [ ] Compile bot_gen_impl.cl on real GPU
- [ ] Compile backtest_impl.cl on real GPU
- [ ] Generate 100 bots, verify struct parsing
- [ ] Generate 10k bots, verify uniqueness
- [ ] Backtest 100 bots on 1000 bars, verify PnL calculation
- [ ] Backtest 10k bots, verify performance (should be <1 minute)
- [ ] Test VRAM estimator with various sizes
- [ ] Test crash on no GPU (should fail gracefully with error message)
- [ ] Test crash on insufficient VRAM
- [ ] Verify reproducibility (same seed → same results)

---

## 📊 Performance Benchmarks to Run

Once GPU code is working:

1. **Bot Generation Speed**:
   - Time 1k, 10k, 100k, 1M bots
   - Compare to CPU version (if still available)

2. **Backtesting Speed**:
   - Time 1k bots × 10 cycles × 50k bars
   - Time 10k bots × 10 cycles × 50k bars
   - Compare to CPU version

3. **VRAM Usage**:
   - Measure actual VRAM with nvidia-smi or rocm-smi
   - Verify against VRAMEstimator predictions

4. **Scaling**:
   - Test doubling population, measure speedup
   - Find maximum population for your GPU

---

## 🐛 Common Issues & Solutions

### Kernel Won't Compile
- Check OpenCL version (`clinfo`)
- Simplify kernel (comment out code sections)
- Add compiler flags: `-cl-std=CL1.2`

### VRAM Overflow
- Reduce population size
- Reduce backtest days
- Use larger timeframe (5m instead of 1m)

### Slow Performance
- Check work group size (should be multiple of 32/64)
- Verify coalesced memory access
- Profile with GPU tools

### Wrong Results
- Verify struct packing/alignment
- Check endianness
- Add debug prints in kernel

---

## 📁 File Organization

```
gpu_bot/
├── src/
│   ├── gpu_kernels/
│   │   ├── bot_gen_impl.cl          ✅ Complete
│   │   ├── backtest_impl.cl         ✅ Complete
│   │   └── precompute_indicators.cl ⏳ TODO
│   ├── bot_generator/
│   │   ├── generator.py             ⚠️  Deprecated (CPU version)
│   │   └── generator_gpu.py         ✅ Complete (needs struct parser)
│   ├── backtester/
│   │   ├── simulator.py             ⚠️  Deprecated (CPU version)
│   │   └── simulator_gpu.py         ⏳ TODO
│   ├── utils/
│   │   └── vram_estimator.py        ✅ Complete
│   └── ...
├── test/
│   ├── test_validation.py           ✅ Exists
│   ├── test_generator_gpu.py        ⏳ TODO
│   ├── test_simulator_gpu.py        ⏳ TODO
│   └── test_workflow.py             ⏳ TODO
├── GPU_IMPLEMENTATION_STATUS.md     ✅ Complete
├── GPU_COMPREHENSIVE_REVIEW_RESPONSE.md ✅ Complete
└── GPU_FINAL_SUMMARY.md             ✅ Complete
```

---

## ⚡ Quick Command Reference

```bash
# Run with GPU (once complete)
python main.py

# Test GPU availability
python -c "import pyopencl as cl; print(cl.get_platforms())"

# Check VRAM
nvidia-smi  # NVIDIA
rocm-smi    # AMD

# Run tests
pytest test/test_generator_gpu.py -v
pytest test/test_workflow.py -v

# Estimate VRAM
python -c "from src.utils.vram_estimator import *; VRAMEstimator.print_vram_report(VRAMEstimator.estimate_and_validate_workflow(10000, 10, 50000))"
```

---

## 🎯 Success Criteria

Implementation is **complete** when:

1. ✅ Application crashes on no GPU (already done)
2. ⏳ Bot generation runs on GPU and produces valid configs
3. ⏳ Backtesting runs on GPU and produces realistic PnL
4. ⏳ VRAM validation prevents overflow
5. ⏳ Results are reproducible with same seed
6. ⏳ 50+ indicators implemented
7. ⏳ 15+ risk strategies implemented
8. ⏳ Mode 4 works for single bot backtest
9. ⏳ Full test suite passes
10. ⏳ Documentation updated

**Current**: 1/10 complete (10%)
**After GPU integration**: 5/10 complete (50%)
**Full completion**: ~20-30 hours remaining

---

Good luck! 🚀
