# IMPLEMENTATION PLAN - COMPLETE GA TRADING BOT

**Date**: November 7, 2025  
**Based On**: CODE_REVIEW_DETAILED.md, STATUS.md, COMPLETE_TODO.md  
**Strategy**: Precomputed Indicators for Memory Efficiency  

---

## ARCHITECTURE DECISION: PRECOMPUTED INDICATORS

### Current Problem
- Inline indicator computation (unified_backtest.cl) causes OUT_OF_RESOURCES
- 50 indicators × 10K bots = massive register pressure
- Each bot recalculates same indicators independently

### Solution: Precompute Once, Read Many
```
┌─────────────────────────────────────────────────────────────┐
│ 1. Load OHLCV Data Once (5000 bars × 5 values = 100KB)    │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Compute ALL Indicators Once (50 × 5000 × 4 = 1MB)      │
│    - SMA(5,10,20,50,100,200)                                │
│    - EMA(5,10,20,50,100,200)                                │
│    - RSI(7,14,21)                                           │
│    - ATR(14,20)                                             │
│    - MACD(12,26,9)                                          │
│    - Stoch(14,3)                                            │
│    - CCI(20)                                                │
│    - BB(20,2.0)                                             │
│    - ... 42 more                                            │
│    Store in global buffer: [50 indicators][5000 bars]       │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Each Bot Reads What It Needs                            │
│    Bot 0: indicator_indices = [10, 23, 45, ...]            │
│           reads RSI[bar], BB_upper[bar], VWAP[bar]          │
│    Bot 1: indicator_indices = [0, 1, 14, ...]              │
│           reads SMA[bar], EMA[bar], MACD[bar]               │
│    ...                                                      │
│    10,000 bots × minimal reads = efficient                  │
└─────────────────────────────────────────────────────────────┘
```

### Memory Comparison
**Inline Computation (Current)**:
- Each bot computes own indicators: 10K bots × 50 indicators = 500K computations
- Register usage: HIGH (causes OUT_OF_RESOURCES)
- Memory: 128 bytes/bot + massive register pressure
- **FAILS at 10K bots**

**Precomputed (New)**:
- Compute once: 50 indicators × 5000 bars × 4 bytes = 1 MB
- Bots read from buffer: 10K bots × 8 indicators × 4 bytes = 320 KB
- Total: ~1.5 MB for all bots + indicators
- Register usage: LOW (just indexing)
- **Works with 1M+ bots**

---

## PHASE 1: CRITICAL KERNEL REWRITE (Priority: P0)

### Task 1.1: Create Indicator Precomputation Kernel
**File**: `src/gpu_kernels/precompute_all_indicators.cl`

**Structure**:
```c
// Output: Global buffer [NUM_INDICATORS][MAX_BARS] = [50][10000]
typedef struct {
    float values[MAX_BARS];  // Precomputed indicator values
} IndicatorBuffer;

__kernel void precompute_all_indicators(
    __global OHLCVBar *ohlcv,      // Input: OHLCV data
    const int num_bars,
    __global IndicatorBuffer *indicators_out  // Output: 50 indicator buffers
) {
    int indicator_id = get_global_id(0);  // 0-49
    
    if (indicator_id >= 50) return;
    
    // Compute this indicator for all bars
    switch(indicator_id) {
        case 0: compute_sma_5(ohlcv, num_bars, &indicators_out[0]); break;
        case 1: compute_sma_10(ohlcv, num_bars, &indicators_out[1]); break;
        case 2: compute_sma_20(ohlcv, num_bars, &indicators_out[2]); break;
        // ... all 50 indicators
    }
}
```

**Indicators to Implement** (50 total):
1. **Moving Averages** (12): SMA(5,10,20,50,100,200), EMA(5,10,20,50,100,200)
2. **Momentum** (8): RSI(7,14,21), Stoch(14,3), StochRSI(14), MOM(10), ROC(10), WILLR(14), CMO(14), TRIX(15)
3. **Volatility** (6): ATR(14,20), NATR(14), BB_upper(20,2), BB_lower(20,2), Keltner(20), Donchian(20)
4. **Trend** (10): MACD(12,26,9), ADX(14), Aroon(25), CCI(20), DPO(20), KST, PSAR, SuperTrend(10,3), Ichimoku, DMI(14)
5. **Volume** (5): OBV, VWAP, MFI(14), AD, ADOSC(3,10)
6. **Pattern** (4): SAR, ZigZag, FractalHigh(5), FractalLow(5)
7. **Cycle** (5): HT_DCPERIOD, HT_SINE, HT_TRENDMODE, HT_PHASOR, MESA

### Task 1.2: Create Real Backtest Kernel
**File**: `src/gpu_kernels/backtest_with_precomputed.cl`

**Structure**:
```c
__kernel void backtest_with_signals(
    __global CompactBotConfig *bots,
    __global OHLCVBar *ohlcv,
    __global IndicatorBuffer *precomputed_indicators,  // Pre-calculated
    __global int *cycle_starts,
    __global int *cycle_ends,
    const int num_cycles,
    const float initial_balance,
    __global BacktestResult *results
) {
    int bot_idx = get_global_id(0);
    CompactBotConfig bot = bots[bot_idx];
    
    // Validate inputs (STRICT)
    if (bot.leverage < 1 || bot.leverage > 125) {
        results[bot_idx].bot_id = -9999;  // Error code
        return;
    }
    
    // For each cycle
    for (int cycle = 0; cycle < num_cycles; cycle++) {
        int start = cycle_starts[cycle];
        int end = cycle_ends[cycle];
        
        // For each bar
        for (int bar = start; bar <= end; bar++) {
            // Read precomputed indicators for THIS bot's configuration
            float indicator_values[8];
            for (int i = 0; i < bot.num_indicators; i++) {
                int ind_idx = bot.indicator_indices[i];
                indicator_values[i] = precomputed_indicators[ind_idx].values[bar];
            }
            
            // Generate signal with 75% consensus
            float signal = generate_signal_consensus(
                indicator_values,
                bot.num_indicators,
                &bot
            );
            
            // Manage positions (multiple allowed until balance < 10%)
            manage_positions(
                &bot,
                signal,
                ohlcv[bar],
                &balance,
                positions,
                &num_positions
            );
        }
    }
    
    // Calculate results
    results[bot_idx] = calculate_metrics(...);
}
```

### Task 1.3: Update CompactBacktester
**File**: `src/backtester/compact_simulator.py`

**Changes**:
```python
class CompactBacktester:
    def __init__(self, ...):
        # Compile BOTH kernels
        self._compile_precompute_kernel()
        self._compile_backtest_kernel()
    
    def backtest_bots(self, bots, ohlcv_data, cycles):
        # Step 1: Precompute all indicators ONCE
        indicator_buffer = self._precompute_indicators(ohlcv_data)
        
        # Step 2: Run backtest with precomputed data
        results = self._run_backtest_kernel(
            bots, 
            ohlcv_data, 
            indicator_buffer,  # Pass precomputed
            cycles
        )
        
        return results
    
    def _precompute_indicators(self, ohlcv_data):
        """Compute all 50 indicators once."""
        num_bars = len(ohlcv_data)
        
        # Allocate output buffer: [50 indicators][num_bars]
        indicators_buf = cl.Buffer(
            self.ctx,
            cl.mem_flags.WRITE_ONLY,
            size=50 * num_bars * 4  # 50 indicators × bars × float32
        )
        
        # Run precompute kernel
        kernel = self.precompute_program.precompute_all_indicators
        kernel(
            self.queue,
            (50,),  # One work item per indicator
            None,
            ohlcv_buf,
            np.int32(num_bars),
            indicators_buf
        )
        
        self.queue.finish()
        return indicators_buf
```

---

## PHASE 2: USER INPUT VALIDATION (Priority: P0)

### Task 2.1: Update CompactBotConfig for 125x Leverage
**File**: `src/bot_generator/compact_generator.py`

**Changes**:
```python
class CompactBotConfig:
    # ... existing fields ...
    leverage: int  # 1-125 (update from 1-10)

class CompactBotGenerator:
    def __init__(
        self,
        population_size: int,
        min_indicators: int = 1,      # User input
        max_indicators: int = 8,      # User input  
        min_risk_strategies: int = 1, # User input
        max_risk_strategies: int = 5, # User input
        leverage: int = 10,           # User input (1-125)
        random_seed: Optional[int] = None,
        gpu_context: cl.Context = None,
        gpu_queue: cl.CommandQueue = None
    ):
        # VALIDATE ALL INPUTS
        if not (1 <= min_indicators <= 8):
            raise ValueError(f"min_indicators must be 1-8, got {min_indicators}")
        if not (min_indicators <= max_indicators <= 8):
            raise ValueError(f"max_indicators must be {min_indicators}-8, got {max_indicators}")
        if not (1 <= min_risk_strategies <= 15):
            raise ValueError(f"min_risk_strategies must be 1-15, got {min_risk_strategies}")
        if not (min_risk_strategies <= max_risk_strategies <= 15):
            raise ValueError(f"max_risk_strategies must be {min_risk_strategies}-15, got {max_risk_strategies}")
        if not (1 <= leverage <= 125):
            raise ValueError(f"leverage must be 1-125, got {leverage}")
        
        self.min_indicators = min_indicators
        self.max_indicators = max_indicators
        self.min_risk_strategies = min_risk_strategies
        self.max_risk_strategies = max_risk_strategies
        self.leverage = leverage
        
        # Pass to kernel
        self._update_kernel_params()
```

**Kernel Update**:
```c
// In compact_bot_gen.cl
__kernel void generate_bots(
    // ... existing params ...
    const int min_indicators,      // NEW
    const int max_indicators,      // NEW
    const int min_risk_strategies, // NEW
    const int max_risk_strategies, // NEW
    const int leverage             // NEW
) {
    // Generate num_indicators in range [min, max]
    bot.num_indicators = min_indicators + (xorshift32(&seed) % (max_indicators - min_indicators + 1));
    
    // Generate risk strategies in range
    int num_risk = min_risk_strategies + (xorshift32(&seed) % (max_risk_strategies - min_risk_strategies + 1));
    
    // Set leverage
    bot.leverage = leverage;  // Fixed per generation
}
```

### Task 2.2: Validate Throughout Pipeline
**Files**: `main.py`, `src/ga/evolver.py`

**Changes in main.py**:
```python
def get_mode1_parameters() -> dict:
    params = {}
    
    # ... existing code ...
    
    # Leverage (1-125)
    params['leverage'] = get_user_input(
        "Leverage (1-125)",
        10,
        lambda x: validate_int(int(x), "leverage", min_val=1, max_val=125)
    )
    
    # Min indicators (1-8)
    params['min_indicators'] = get_user_input(
        "Min indicators per bot (1-8)",
        1,
        lambda x: validate_int(int(x), "min_indicators", min_val=1, max_val=8)
    )
    
    # Max indicators (min-8)
    params['max_indicators'] = get_user_input(
        f"Max indicators per bot ({params['min_indicators']}-8)",
        5,
        lambda x: validate_int(int(x), "max_indicators", 
                              min_val=params['min_indicators'], max_val=8)
    )
    
    # Min risk strategies (1-15)
    params['min_risk_strategies'] = get_user_input(
        "Min risk strategies per bot (1-15)",
        1,
        lambda x: validate_int(int(x), "min_risk_strategies", min_val=1, max_val=15)
    )
    
    # Max risk strategies (min-15)
    params['max_risk_strategies'] = get_user_input(
        f"Max risk strategies per bot ({params['min_risk_strategies']}-15)",
        5,
        lambda x: validate_int(int(x), "max_risk_strategies",
                              min_val=params['min_risk_strategies'], max_val=15)
    )
    
    return params

def run_mode1(params: dict, ...):
    # Pass ALL user params to generator
    bot_generator = CompactBotGenerator(
        population_size=params['population'],
        min_indicators=params['min_indicators'],
        max_indicators=params['max_indicators'],
        min_risk_strategies=params['min_risk_strategies'],
        max_risk_strategies=params['max_risk_strategies'],
        leverage=params['leverage'],
        random_seed=params['random_seed'],
        gpu_context=gpu_context,
        gpu_queue=gpu_queue
    )
```

---

## PHASE 3: FIX GA EVOLVER (Priority: P0)

### Task 3.1: Fix Method Name
**File**: `src/ga/evolver.py`, line 93

**Change**:
```python
# OLD
population = self.bot_generator.generate_initial_population()

# NEW
population = self.bot_generator.generate_population()
```

### Task 3.2: Implement Mutation for Compact Bots
```python
def mutate_bot(self, bot: CompactBotConfig, mutation_rate: float = 0.15) -> CompactBotConfig:
    """Mutate compact bot configuration."""
    import copy
    import random
    import numpy as np
    
    mutated = copy.deepcopy(bot)
    
    # Mutation 1: Change indicator type (30% chance)
    if random.random() < mutation_rate * 2:
        idx = random.randint(0, mutated.num_indicators - 1)
        old_ind = mutated.indicator_indices[idx]
        new_ind = random.randint(0, 49)
        mutated.indicator_indices[idx] = new_ind
        log_debug(f"Bot {bot.bot_id}: Mutated indicator {idx}: {old_ind} -> {new_ind}")
    
    # Mutation 2: Adjust parameter (40% chance)
    if random.random() < mutation_rate * 3:
        idx = random.randint(0, mutated.num_indicators - 1)
        param_idx = random.randint(0, 2)
        old_val = mutated.indicator_params[idx][param_idx]
        mutated.indicator_params[idx][param_idx] *= random.uniform(0.8, 1.2)
        log_debug(f"Bot {bot.bot_id}: Mutated param {idx},{param_idx}: {old_val:.2f} -> {mutated.indicator_params[idx][param_idx]:.2f}")
    
    # Mutation 3: Flip risk strategy bit (30% chance)
    if random.random() < mutation_rate * 2:
        bit = random.randint(0, 14)
        old_bitmap = mutated.risk_strategy_bitmap
        mutated.risk_strategy_bitmap ^= (1 << bit)
        log_debug(f"Bot {bot.bot_id}: Flipped risk bit {bit}: {bin(old_bitmap)} -> {bin(mutated.risk_strategy_bitmap)}")
    
    # Mutation 4: Adjust TP (20% chance)
    if random.random() < mutation_rate:
        old_tp = mutated.tp_multiplier
        mutated.tp_multiplier *= random.uniform(0.9, 1.1)
        mutated.tp_multiplier = max(0.01, min(0.25, mutated.tp_multiplier))
        log_debug(f"Bot {bot.bot_id}: Mutated TP: {old_tp:.4f} -> {mutated.tp_multiplier:.4f}")
    
    # Mutation 5: Adjust SL (20% chance)
    if random.random() < mutation_rate:
        old_sl = mutated.sl_multiplier
        mutated.sl_multiplier *= random.uniform(0.9, 1.1)
        mutated.sl_multiplier = max(0.005, min(mutated.tp_multiplier/2, mutated.sl_multiplier))
        log_debug(f"Bot {bot.bot_id}: Mutated SL: {old_sl:.4f} -> {mutated.sl_multiplier:.4f}")
    
    return mutated

def crossover(self, parent1: CompactBotConfig, parent2: CompactBotConfig) -> CompactBotConfig:
    """Cross two compact bots."""
    import random
    import numpy as np
    
    # Create child
    child = CompactBotConfig(
        bot_id=-1,  # Will be assigned later
        num_indicators=0,
        indicator_indices=np.zeros(8, dtype=np.uint8),
        indicator_params=np.zeros((8, 3), dtype=np.float32),
        risk_strategy_bitmap=0,
        tp_multiplier=0.0,
        sl_multiplier=0.0,
        leverage=parent1.leverage
    )
    
    # Crossover indicators (mix from both parents)
    min_ind = min(parent1.num_indicators, parent2.num_indicators)
    crossover_point = random.randint(1, min_ind)
    
    # Take first part from parent1
    child.indicator_indices[:crossover_point] = parent1.indicator_indices[:crossover_point]
    child.indicator_params[:crossover_point] = parent1.indicator_params[:crossover_point]
    
    # Take second part from parent2
    child.indicator_indices[crossover_point:min_ind] = parent2.indicator_indices[crossover_point:min_ind]
    child.indicator_params[crossover_point:min_ind] = parent2.indicator_params[crossover_point:min_ind]
    
    child.num_indicators = min_ind
    
    # Combine risk strategies (OR operation - get all strategies from both)
    child.risk_strategy_bitmap = parent1.risk_strategy_bitmap | parent2.risk_strategy_bitmap
    
    # Average TP/SL
    child.tp_multiplier = (parent1.tp_multiplier + parent2.tp_multiplier) / 2
    child.sl_multiplier = (parent1.sl_multiplier + parent2.sl_multiplier) / 2
    
    log_debug(f"Crossover: P1({parent1.bot_id}) × P2({parent2.bot_id}) -> Child")
    
    return child
```

### Task 3.3: Implement Combination Tracking
```python
class CompactBotGenerator:
    def __init__(self, ...):
        # ... existing code ...
        self.used_combinations: Set[frozenset] = set()
        self.next_bot_id = 0
    
    def generate_population(self) -> List[CompactBotConfig]:
        """Generate population with unique indicator combinations."""
        # ... existing kernel generation ...
        
        # Track combinations
        for bot in bots:
            combo = frozenset(bot.indicator_indices[:bot.num_indicators])
            self.used_combinations.add(combo)
            bot.bot_id = self.next_bot_id
            self.next_bot_id += 1
        
        log_info(f"Generated {len(bots)} bots, {len(self.used_combinations)} unique combinations used")
        return bots
    
    def release_combinations(self, dead_bots: List[CompactBotConfig]):
        """Release indicator combinations from eliminated bots."""
        for bot in dead_bots:
            combo = frozenset(bot.indicator_indices[:bot.num_indicators])
            self.used_combinations.discard(combo)
        log_info(f"Released {len(dead_bots)} combinations, {len(self.used_combinations)} still in use")
```

---

## PHASE 4: DATA PROVIDER INTEGRATION (Priority: P1)

### Task 4.1: Update DataFetcher
**File**: `src/data_provider/fetcher.py`

```python
def calculate_required_days(self, backtest_days: int, cycles: int) -> int:
    """Calculate total days needed with 25% buffer."""
    base_days = backtest_days * cycles
    buffered = int(base_days * 1.25)  # +25%
    log_info(f"Required: {backtest_days}d × {cycles} cycles × 1.25 buffer = {buffered} days")
    return buffered

def fetch_and_store_daily(self, pair: str, timeframe: str, num_days: int) -> List[str]:
    """Fetch data and store in 1-day slices."""
    from datetime import datetime, timedelta
    
    end_date = datetime.now() - timedelta(days=1)  # Exclude incomplete today
    start_date = end_date - timedelta(days=num_days)
    
    file_paths = []
    current = start_date
    
    while current <= end_date:
        date_str = current.strftime("%Y-%m-%d")
        filename = f"data/{pair}_{timeframe}_{date_str}.npy"
        
        if not os.path.exists(filename):
            log_info(f"Fetching {date_str}...")
            ohlcv = self.fetch_ohlcv(
                pair=pair,
                timeframe=timeframe,
                start=current,
                end=current + timedelta(days=1)
            )
            np.save(filename, ohlcv)
        else:
            log_debug(f"Using cached {date_str}")
        
        file_paths.append(filename)
        current += timedelta(days=1)
    
    return file_paths
```

### Task 4.2: Update DataLoader for Non-Overlapping Cycles
**File**: `src/data_provider/loader.py`

```python
def create_non_overlapping_cycles(
    self,
    num_cycles: int,
    backtest_days: int,
    ohlcv_data: np.ndarray
) -> List[Tuple[int, int]]:
    """Create random non-overlapping cycles."""
    import random
    
    bars_per_day = self._get_bars_per_day()
    cycle_size = bars_per_day * backtest_days
    total_bars = len(ohlcv_data)
    
    # Calculate maximum possible cycles
    max_cycles = total_bars // cycle_size
    if num_cycles > max_cycles:
        raise ValueError(f"Cannot create {num_cycles} non-overlapping cycles of {backtest_days} days in {total_bars} bars")
    
    # Create cycle ranges
    all_possible_starts = list(range(0, total_bars - cycle_size + 1, cycle_size))
    random.shuffle(all_possible_starts)
    
    cycles = []
    for i in range(num_cycles):
        start = all_possible_starts[i]
        end = start + cycle_size - 1
        cycles.append((start, end))
    
    # Verify non-overlapping
    sorted_cycles = sorted(cycles)
    for i in range(len(sorted_cycles) - 1):
        if sorted_cycles[i][1] >= sorted_cycles[i+1][0]:
            raise RuntimeError("Cycles overlap! This should never happen.")
    
    log_info(f"Created {num_cycles} non-overlapping cycles of {backtest_days} days each")
    return cycles

def _get_bars_per_day(self) -> int:
    """Get number of bars per day based on timeframe."""
    tf_to_bars = {
        '1m': 1440,
        '5m': 288,
        '15m': 96,
        '1h': 24,
        '4h': 6,
        '1d': 1
    }
    return tf_to_bars.get(self.timeframe, 1440)
```

---

## PHASE 5: TP/SL VALIDATION (Priority: P1)

### Task 5.1: Implement Leverage-Aware Validation
**File**: `src/gpu_kernels/compact_bot_gen.cl`

```c
// After generating TP/SL, validate them
void validate_and_fix_tp_sl(CompactBotConfig *bot) {
    // Kucoin fees
    float maker_fee = 0.0002f;
    float taker_fee = 0.0006f;
    
    // Total fee cost (entry + exit)
    float total_fee_pct = (maker_fee + taker_fee) * (float)bot->leverage;
    
    // Minimum TP must cover fees + 0.5% profit
    float min_tp = total_fee_pct + 0.005f;
    if (bot->tp_multiplier < min_tp) {
        bot->tp_multiplier = min_tp;
    }
    
    // Maximum TP is 25%
    if (bot->tp_multiplier > 0.25f) {
        bot->tp_multiplier = 0.25f;
    }
    
    // SL must be at most TP/2
    float max_sl = bot->tp_multiplier / 2.0f;
    if (bot->sl_multiplier > max_sl) {
        bot->sl_multiplier = max_sl;
    }
    
    // SL must not trigger immediate liquidation
    // Liquidation at ~(1/leverage - 1% buffer)
    float liq_threshold = (1.0f / (float)bot->leverage) - 0.01f;
    if (bot->sl_multiplier > liq_threshold) {
        bot->sl_multiplier = liq_threshold;
    }
    
    // Minimum SL is 0.2% (avoid market noise)
    if (bot->sl_multiplier < 0.002f) {
        bot->sl_multiplier = 0.002f;
    }
}
```

---

## TESTING STRATEGY

### Unit Tests
1. **Test Each Indicator** (50 tests)
   ```python
   def test_sma_5_accuracy():
       # Generate test data
       ohlcv = create_test_ohlcv()
       
       # Compute with kernel
       kernel_result = precompute_sma_5(ohlcv)
       
       # Compute with TA-Lib reference
       talib_result = talib.SMA(ohlcv[:, 3], timeperiod=5)
       
       # Compare
       assert np.allclose(kernel_result, talib_result, rtol=1e-4)
   ```

2. **Test TP/SL Validation** (10 tests)
3. **Test Leverage Limits** (5 tests)
4. **Test Non-Overlapping Cycles** (5 tests)

### Integration Tests
1. **Complete GA Workflow** (1 test)
   ```python
   def test_full_ga_10_generations():
       # Run 10 generations with 100 bots
       params = {
           'population': 100,
           'generations': 10,
           'cycles': 5,
           'backtest_days': 7,
           # ... all params
       }
       
       result = run_complete_ga(params)
       
       assert len(result['top_bots']) == 10
       assert all(b.generation_survived >= 1 for b in result['top_bots'])
   ```

---

## IMPLEMENTATION ORDER

### Week 1: Critical Kernel
- [x] Day 1-2: Implement precompute_all_indicators.cl (50 indicators)
- [ ] Day 3-4: Implement backtest_with_precomputed.cl (real trading)
- [ ] Day 5: Update CompactBacktester for 2-kernel approach

### Week 2: Validation & GA
- [ ] Day 1: Fix leverage limits (1-125), user input validation
- [ ] Day 2: Fix GA evolver (mutation, crossover, combination tracking)
- [ ] Day 3: Implement TP/SL validation
- [ ] Day 4-5: Test all 50 indicators against TA-Lib

### Week 3: Data & Integration
- [ ] Day 1-2: Integrate data provider (1-day slices, buffer calc)
- [ ] Day 3: Implement non-overlapping cycles
- [ ] Day 4-5: Complete integration testing

---

## SUCCESS METRICS

- [ ] All 50 indicators match TA-Lib within 0.01%
- [ ] 1M bots backtest without OUT_OF_RESOURCES
- [ ] Memory usage < 500MB for 1M bots
- [ ] Complete GA runs 10 generations successfully
- [ ] All user inputs validated and flow correctly
- [ ] TP always > fees, SL never causes instant liquidation
- [ ] Cycles are proven non-overlapping
- [ ] Top 10 bots saved and can be reloaded

---

**NEXT STEP**: Start implementing precompute_all_indicators.cl kernel
