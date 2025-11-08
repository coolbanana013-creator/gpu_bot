# COMPLETE TODO: GENETIC ALGORITHM CRYPTO TRADING BOT

**Project**: GPU-Accelerated GA Trading Bot for Kucoin Futures  
**Current Status**: 45% Complete (Infrastructure solid, features incomplete)  
**Target**: Fully functional genetic algorithm with realistic backtesting  

---

## PRIORITY LEVELS

🔴 **P0 - CRITICAL**: Must fix before any production use  
🟠 **P1 - HIGH**: Required for basic functionality  
🟡 **P2 - MEDIUM**: Important for correct operation  
🟢 **P3 - LOW**: Nice to have, optimization  

---

## PHASE 1: CRITICAL FIXES (Est: 2-3 days)

### 🔴 P0-1: Implement Real Backtest Kernel
**File**: `src/gpu_kernels/unified_backtest_minimal.cl`  
**Current**: Fake test kernel generating random results  
**Required**: Full trading simulation with realistic logic  

**Tasks**:
- [ ] Implement 15 core indicators:
  - SMA (Simple Moving Average) - Period 5-200
  - EMA (Exponential Moving Average) - Period 5-200
  - RSI (Relative Strength Index) - Period 5-50, Thresholds 20-40/60-80
  - ATR (Average True Range) - Period 5-50
  - MACD (Moving Average Convergence Divergence) - Fast 5-20, Slow 20-50
  - Stochastic - Period 5-50
  - CCI (Commodity Channel Index) - Period 5-50
  - Bollinger Bands - Period 10-50, StdDev 1.5-3.0
  - ROC (Rate of Change) - Period 5-30
  - MOM (Momentum) - Period 5-30
  - Williams %R - Period 5-30
  - Aroon - Period 10-50
  - ADX (Average Directional Index) - Period 10-30
  - OBV (On-Balance Volume) - No parameters
  - VWAP (Volume Weighted Average Price) - Period 10-100

- [ ] Implement signal generation logic:
  ```c
  // For each indicator, calculate signal (-1 to +1)
  // SMA: Price crosses above = +1, below = -1
  // RSI: < oversold = +1, > overbought = -1
  // MACD: MACD > signal line = +1, < = -1
  // Bollinger: Price < lower = +1, > upper = -1
  // Etc.
  ```

- [ ] Implement 75% consensus threshold:
  ```c
  int bullish_signals = 0;
  int bearish_signals = 0;
  
  for (int i = 0; i < bot.num_indicators; i++) {
      float signal = calculate_indicator_signal(...);
      if (signal > 0.5f) bullish_signals++;
      else if (signal < -0.5f) bearish_signals++;
  }
  
  float consensus = (float)max(bullish_signals, bearish_signals) / (float)bot.num_indicators;
  
  if (consensus >= 0.75f) {
      // Open position
  }
  ```

- [ ] Implement position management:
  ```c
  // Track multiple positions
  Position positions[100];  // Up to 100 concurrent
  int num_open_positions = 0;
  
  // Open position only if:
  // 1. Consensus >= 75%
  // 2. Free balance > 10% of total
  // 3. Have available position slot
  
  float free_balance = balance - (position_value_sum);
  if (free_balance > total_balance * 0.10f && num_open_positions < 100) {
      // Calculate position size using risk strategies
      float pos_size = calculate_position_size(bot.risk_strategy_bitmap, ...);
      
      // Open position
      positions[num_open_positions++] = create_position(...);
  }
  ```

- [ ] Implement TP/SL execution:
  ```c
  // For each open position
  for (int p = 0; p < num_open_positions; p++) {
      Position *pos = &positions[p];
      
      // Check TP
      if (is_long && price >= pos->tp_price) {
          close_position(pos, price, TAKER_FEE);
      }
      else if (!is_long && price <= pos->tp_price) {
          close_position(pos, price, TAKER_FEE);
      }
      
      // Check SL
      if (is_long && price <= pos->sl_price) {
          close_position(pos, price, TAKER_FEE);
      }
      else if (!is_long && price >= pos->sl_price) {
          close_position(pos, price, TAKER_FEE);
      }
      
      // Check liquidation
      float unrealized_pnl = calculate_pnl(pos, price);
      float loss_pct = -unrealized_pnl / pos->initial_margin;
      if (loss_pct >= (1.0f / (float)bot.leverage - 0.01f)) {
          liquidate_position(pos);
          balance = 0.0f;  // Liquidated
      }
  }
  ```

- [ ] Implement realistic fee calculation:
  ```c
  // Kucoin Futures fees
  #define MAKER_FEE 0.0002f      // 0.02%
  #define TAKER_FEE 0.0006f      // 0.06%
  #define SLIPPAGE 0.0010f       // 0.10% estimated
  #define FUNDING_RATE 0.0001f   // 0.01% per 8h
  
  // On entry
  float entry_fee = entry_price * position_size * TAKER_FEE;
  float slippage_cost = entry_price * position_size * SLIPPAGE;
  balance -= (entry_fee + slippage_cost);
  
  // On exit
  float exit_fee = exit_price * position_size * TAKER_FEE;
  balance -= exit_fee;
  
  // Funding (every 8 hours if position held)
  int funding_periods = (exit_bar - entry_bar) / (8 * 60);  // For 1m timeframe
  float funding_cost = position_value * FUNDING_RATE * funding_periods;
  balance -= funding_cost;
  ```

- [ ] Implement risk management strategies:
  ```c
  float calculate_position_size(uint risk_bitmap, float balance, float atr, ...) {
      float base_size = balance * 0.02f;  // Default 2%
      float adjusted_size = 0.0f;
      int strategy_count = 0;
      
      // Strategy 0: Fixed Percentage
      if (risk_bitmap & (1 << 0)) {
          adjusted_size += balance * 0.05f;
          strategy_count++;
      }
      
      // Strategy 1: Kelly Criterion
      if (risk_bitmap & (1 << 1)) {
          float win_rate = estimate_win_rate(...);
          float avg_win = estimate_avg_win(...);
          float avg_loss = estimate_avg_loss(...);
          float kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win;
          kelly = fmax(0.0f, fmin(kelly, 0.25f));  // Cap at 25%
          adjusted_size += balance * kelly;
          strategy_count++;
      }
      
      // Strategy 2: Volatility-based (ATR)
      if (risk_bitmap & (1 << 2)) {
          float vol_pct = atr / current_price;
          float size_multiplier = fmax(0.5f, fmin(2.0f, 0.02f / vol_pct));
          adjusted_size += base_size * size_multiplier;
          strategy_count++;
      }
      
      // ... Implement all 15 strategies
      
      // Average if multiple strategies active
      if (strategy_count > 0) {
          adjusted_size /= (float)strategy_count;
      } else {
          adjusted_size = base_size;
      }
      
      // Apply leverage
      adjusted_size *= (float)bot.leverage;
      
      // Cap at 95% of available balance
      return fmin(adjusted_size, balance * 0.95f);
  }
  ```

- [ ] Implement proper statistics calculation:
  ```c
  // Track for Sharpe ratio
  float sum_returns = 0.0f;
  float sum_sq_returns = 0.0f;
  int num_returns = 0;
  
  // On each trade close
  float trade_return = pnl / initial_balance;
  sum_returns += trade_return;
  sum_sq_returns += trade_return * trade_return;
  num_returns++;
  
  // Calculate Sharpe
  float avg_return = sum_returns / (float)num_returns;
  float variance = (sum_sq_returns / (float)num_returns) - (avg_return * avg_return);
  float std_dev = sqrt(fmax(variance, 0.0f));
  float sharpe = (avg_return / std_dev) * sqrt(252.0f);  // Annualized
  
  // Track drawdown
  float peak_balance = initial_balance;
  if (balance > peak_balance) peak_balance = balance;
  float drawdown = (peak_balance - balance) / peak_balance * 100.0f;
  if (drawdown > max_drawdown) max_drawdown = drawdown;
  ```

**Estimated Time**: 1-2 days  
**Complexity**: HIGH  
**Impact**: CRITICAL - Without this, all results are fake  

---

### 🔴 P0-2: Fix GA Evolver for Compact Bots
**File**: `src/ga/evolver.py`  
**Current**: Calls non-existent method, mutation/crossover not adapted  
**Required**: Full GA workflow for compact bot structure  

**Tasks**:
- [ ] Fix method name (line 93):
  ```python
  # Change from:
  population = self.bot_generator.generate_initial_population()
  # To:
  population = self.bot_generator.generate_population()
  ```

- [ ] Implement mutation for compact bots:
  ```python
  def mutate_bot(self, bot: CompactBotConfig, mutation_rate: float = 0.1) -> CompactBotConfig:
      """Mutate compact bot configuration."""
      import copy
      import random
      
      mutated = copy.deepcopy(bot)
      
      # Randomly choose what to mutate
      mutations = []
      
      # 1. Indicator mutation (change one indicator type)
      if random.random() < mutation_rate:
          idx = random.randint(0, mutated.num_indicators - 1)
          new_indicator = random.randint(0, 49)
          mutated.indicator_indices[idx] = new_indicator
          # Re-generate params for new indicator type
          # (Get valid range from IndicatorFactory)
      
      # 2. Parameter mutation (adjust one parameter)
      if random.random() < mutation_rate:
          idx = random.randint(0, mutated.num_indicators - 1)
          param_idx = random.randint(0, 2)
          # Mutate by ±20%
          mutated.indicator_params[idx][param_idx] *= random.uniform(0.8, 1.2)
          # Clamp to valid range
      
      # 3. Risk strategy mutation (flip one bit)
      if random.random() < mutation_rate:
          bit = random.randint(0, 14)
          mutated.risk_strategy_bitmap ^= (1 << bit)
      
      # 4. TP/SL mutation
      if random.random() < mutation_rate:
          mutated.tp_multiplier *= random.uniform(0.9, 1.1)
          mutated.sl_multiplier *= random.uniform(0.9, 1.1)
          # Validate constraints
      
      # 5. Leverage mutation
      if random.random() < mutation_rate:
          mutated.leverage = random.randint(1, 10)
      
      return mutated
  ```

- [ ] Implement crossover for compact bots:
  ```python
  def crossover(self, parent1: CompactBotConfig, parent2: CompactBotConfig) -> CompactBotConfig:
      """Cross two compact bots."""
      import random
      
      child = CompactBotConfig(
          bot_id=self._next_bot_id(),
          num_indicators=0,  # Will set below
          indicator_indices=np.zeros(8, dtype=np.uint8),
          indicator_params=np.zeros((8, 3), dtype=np.float32),
          risk_strategy_bitmap=0,
          tp_multiplier=0.0,
          sl_multiplier=0.0,
          leverage=0
      )
      
      # Mix indicators
      num_from_p1 = random.randint(1, min(parent1.num_indicators, parent2.num_indicators))
      child.num_indicators = parent1.num_indicators  # Or average
      
      child.indicator_indices[:num_from_p1] = parent1.indicator_indices[:num_from_p1]
      child.indicator_indices[num_from_p1:child.num_indicators] = \
          parent2.indicator_indices[:child.num_indicators - num_from_p1]
      
      child.indicator_params[:num_from_p1] = parent1.indicator_params[:num_from_p1]
      child.indicator_params[num_from_p1:child.num_indicators] = \
          parent2.indicator_params[:child.num_indicators - num_from_p1]
      
      # Combine risk strategies (OR/AND/XOR)
      strategy = random.choice(['or', 'and', 'xor'])
      if strategy == 'or':
          child.risk_strategy_bitmap = parent1.risk_strategy_bitmap | parent2.risk_strategy_bitmap
      elif strategy == 'and':
          child.risk_strategy_bitmap = parent1.risk_strategy_bitmap & parent2.risk_strategy_bitmap
      else:
          child.risk_strategy_bitmap = parent1.risk_strategy_bitmap ^ parent2.risk_strategy_bitmap
      
      # Ensure at least one strategy
      if child.risk_strategy_bitmap == 0:
          child.risk_strategy_bitmap = parent1.risk_strategy_bitmap
      
      # Average TP/SL
      child.tp_multiplier = (parent1.tp_multiplier + parent2.tp_multiplier) / 2
      child.sl_multiplier = (parent1.sl_multiplier + parent2.sl_multiplier) / 2
      
      # Random leverage from either parent
      child.leverage = random.choice([parent1.leverage, parent2.leverage])
      
      return child
  ```

- [ ] Implement selection logic:
  ```python
  def select_survivors(self, results: List[BacktestResult], population: List[CompactBotConfig]) -> List[CompactBotConfig]:
      """Select bots that survive to next generation."""
      # Calculate average profit across all cycles for each bot
      bot_avg_profits = {}
      for result in results:
          if result.bot_id not in bot_avg_profits:
              bot_avg_profits[result.bot_id] = []
          bot_avg_profits[result.bot_id].append(result.total_return_pct)
      
      # Keep only bots with positive average profit
      survivors = []
      for bot in population:
          avg_profit = np.mean(bot_avg_profits[bot.bot_id])
          if avg_profit > 0:
              survivors.append(bot)
              # Track performance
              self.bot_performances[bot.bot_id].add_generation_result(
                  self.current_generation,
                  {
                      'avg_profit_pct': avg_profit,
                      'avg_winrate': ...,
                      'avg_trades': ...
                  }
              )
      
      log_info(f"Generation {self.current_generation}: {len(survivors)}/{len(population)} survived")
      return survivors
  ```

- [ ] Implement population refill:
  ```python
  def refill_population(self, survivors: List[CompactBotConfig], target_size: int) -> List[CompactBotConfig]:
      """Refill population to target size."""
      if len(survivors) >= target_size:
          return survivors[:target_size]
      
      new_population = survivors.copy()
      needed = target_size - len(survivors)
      
      # Strategy: 50% new random, 50% offspring from survivors
      num_new = needed // 2
      num_offspring = needed - num_new
      
      # Generate new random bots
      new_bots = self.bot_generator.generate_population()[:num_new]
      new_population.extend(new_bots)
      
      # Create offspring from survivors
      for _ in range(num_offspring):
          parent1 = random.choice(survivors)
          parent2 = random.choice(survivors)
          child = self.crossover(parent1, parent2)
          child = self.mutate_bot(child)
          new_population.append(child)
      
      return new_population
  ```

**Estimated Time**: 1 day  
**Complexity**: MEDIUM  
**Impact**: CRITICAL - GA won't work without this  

---

### 🟠 P1-1: Integrate Data Provider
**Files**: `src/data_provider/fetcher.py`, `loader.py`, `src/backtester/compact_simulator.py`  
**Current**: Backtester uses synthetic random data  
**Required**: Real Kucoin data fetched and loaded  

**Tasks**:
- [ ] Update DataFetcher to store in 1-day slices:
  ```python
  def fetch_and_store(self, pair: str, timeframe: str, start_date: str, end_date: str):
      """Fetch data and store in daily files."""
      # data/BTC-USDT/1m/2025-11-01.csv
      # data/BTC-USDT/1m/2025-11-02.csv
      # etc.
      
      base_dir = Path("data") / pair / timeframe
      base_dir.mkdir(parents=True, exist_ok=True)
      
      current_date = datetime.strptime(start_date, "%Y-%m-%d")
      end_dt = datetime.strptime(end_date, "%Y-%m-%d")
      
      while current_date <= end_dt:
          # Fetch one day
          day_str = current_date.strftime("%Y-%m-%d")
          file_path = base_dir / f"{day_str}.csv"
          
          if not file_path.exists():
              data = self.fetch_kucoin_ohlcv(pair, timeframe, day_str)
              data.to_csv(file_path)
          
          current_date += timedelta(days=1)
  ```

- [ ] Calculate required data range with buffer:
  ```python
  def calculate_data_needed(backtest_days: int, cycles: int) -> int:
      """Calculate total days needed with 25% buffer."""
      base_days = backtest_days * cycles
      buffered_days = int(base_days * 1.25)
      return buffered_days
  ```

- [ ] Exclude today's incomplete data:
  ```python
  def get_complete_days(pair: str, timeframe: str, num_days: int) -> List[str]:
      """Get list of complete day files, excluding today."""
      today = datetime.now().date()
      end_date = today - timedelta(days=1)  # Exclude today
      start_date = end_date - timedelta(days=num_days)
      
      days = []
      current = start_date
      while current <= end_date:
          days.append(current.strftime("%Y-%m-%d"))
          current += timedelta(days=1)
      
      return days
  ```

- [ ] Update CompactBacktester to use real data:
  ```python
  # In main.py Mode 1
  from src.data_provider.fetcher import DataFetcher
  from src.data_provider.loader import DataLoader
  
  # Calculate days needed
  total_days = calculate_data_needed(backtest_days=7, cycles=10)  # = 88 days
  
  # Fetch data
  fetcher = DataFetcher()
  fetcher.fetch_and_store(
      pair="BTC-USDT",
      timeframe="1m",
      start_date=...,
      end_date=...
  )
  
  # Load data
  loader = DataLoader(pair="BTC-USDT", timeframe="1m")
  ohlcv_data = loader.load_days(total_days)
  
  # Create non-overlapping cycles
  cycles = loader.create_random_cycles(
      num_cycles=10,
      cycle_days=7,
      total_data=ohlcv_data
  )
  ```

**Estimated Time**: 4-6 hours  
**Complexity**: LOW  
**Impact**: HIGH - Need real data for meaningful results  

---

### 🟠 P1-2: Implement TP/SL Validation
**Files**: `src/bot_generator/compact_generator.py`, `src/gpu_kernels/compact_bot_gen.cl`  
**Current**: No validation that TP/SL are viable  
**Required**: Ensure TP > fees, SL doesn't cause instant liquidation  

**Tasks**:
- [ ] Add TP/SL validation in GPU kernel:
  ```c
  // After generating TP/SL multipliers
  
  // Calculate minimum TP to cover fees
  float maker_fee = 0.0002f;
  float taker_fee = 0.0006f;
  float total_fees = (maker_fee + taker_fee) * (float)bot.leverage;
  float min_tp = total_fees + 0.005f;  // +0.5% profit minimum
  
  if (bot.tp_multiplier < min_tp) {
      bot.tp_multiplier = min_tp;
  }
  
  // Ensure SL is at most TP/2
  float max_sl = bot.tp_multiplier / 2.0f;
  if (bot.sl_multiplier > max_sl) {
      bot.sl_multiplier = max_sl;
  }
  
  // Ensure SL doesn't cause immediate liquidation
  // Liquidation occurs at ~(1/leverage - margin_buffer)
  float liq_threshold = (1.0f / (float)bot.leverage) - 0.01f;  // -1% buffer
  if (bot.sl_multiplier > liq_threshold) {
      bot.sl_multiplier = liq_threshold;
  }
  
  // Ensure minimum distance from entry
  if (bot.sl_multiplier < 0.002f) {  // Min 0.2% to avoid noise
      bot.sl_multiplier = 0.002f;
  }
  ```

- [ ] Add validation in Python (post-generation check):
  ```python
  def validate_bot_config(bot: CompactBotConfig) -> bool:
      """Validate bot configuration is viable."""
      # Check TP > fees
      total_fees = (0.0002 + 0.0006) * bot.leverage
      min_tp = total_fees + 0.005
      if bot.tp_multiplier < min_tp:
          return False
      
      # Check SL constraints
      if bot.sl_multiplier > bot.tp_multiplier / 2:
          return False
      
      liq_threshold = (1.0 / bot.leverage) - 0.01
      if bot.sl_multiplier > liq_threshold:
          return False
      
      if bot.sl_multiplier < 0.002:
          return False
      
      return True
  ```

**Estimated Time**: 2-3 hours  
**Complexity**: LOW  
**Impact**: HIGH - Prevents invalid bot configs  

---

## PHASE 2: CORE FEATURES (Est: 3-4 days)

### 🟡 P2-1: Implement Unique Combination Tracking
**File**: `src/bot_generator/compact_generator.py`  
**Current**: No tracking of which indicator combinations are used  
**Required**: Track and avoid duplicates, enable refill from unused  

**Tasks**:
- [ ] Add combination tracking data structure:
  ```python
  class CompactBotGenerator:
      def __init__(self, ...):
          # Track used combinations
          self.used_combinations: Set[frozenset] = set()
          
          # Pre-calculate total possible combinations
          self.total_possible = self._calculate_total_combinations()
      
      def _calculate_total_combinations(self) -> int:
          """Calculate total possible unique combinations."""
          from math import comb
          total = 0
          for k in range(self.min_indicators, self.max_indicators + 1):
              total += comb(50, k)  # C(50, k) combinations
          return total
  ```

- [ ] Track combinations during generation:
  ```python
  def generate_population(self) -> List[CompactBotConfig]:
      bots = # ... existing generation code ...
      
      # Track combinations
      for bot in bots:
          combo = frozenset(bot.indicator_indices[:bot.num_indicators])
          self.used_combinations.add(combo)
      
      log_info(f"Used combinations: {len(self.used_combinations)} / {self.total_possible}")
      return bots
  ```

- [ ] Implement combination release:
  ```python
  def release_combinations(self, dead_bots: List[CompactBotConfig]):
      """Release indicator combinations from eliminated bots."""
      for bot in dead_bots:
          combo = frozenset(bot.indicator_indices[:bot.num_indicators])
          self.used_combinations.discard(combo)
      
      log_info(f"Released {len(dead_bots)} combinations")
  ```

- [ ] Add method to check availability:
  ```python
  def can_generate_unique(self, count: int) -> bool:
      """Check if we can generate 'count' unique bots."""
      available = self.total_possible - len(self.used_combinations)
      return available >= count
  ```

**Estimated Time**: 4 hours  
**Complexity**: MEDIUM  
**Impact**: MEDIUM - Important for GA exploration  

---

### 🟡 P2-2: Implement Non-Overlapping Cycle Validation
**File**: `src/data_provider/loader.py`, `src/backtester/compact_simulator.py`  
**Current**: No validation that cycles don't overlap  
**Required**: Ensure statistical independence of cycles  

**Tasks**:
- [ ] Add cycle validation method:
  ```python
  def validate_cycles_non_overlapping(cycles: List[Tuple[int, int]]) -> bool:
      """Ensure cycles don't overlap."""
      sorted_cycles = sorted(cycles, key=lambda x: x[0])
      
      for i in range(len(sorted_cycles) - 1):
          if sorted_cycles[i][1] >= sorted_cycles[i+1][0]:
              raise ValueError(
                  f"Overlapping cycles detected: "
                  f"{sorted_cycles[i]} and {sorted_cycles[i+1]}"
              )
      
      return True
  ```

- [ ] Implement random non-overlapping cycle generation:
  ```python
  def create_random_cycles(
      num_cycles: int,
      cycle_days: int,
      timeframe: str,
      total_bars: int
  ) -> List[Tuple[int, int]]:
      """Create random non-overlapping cycles."""
      # Calculate bars per cycle
      if timeframe == "1m":
          bars_per_day = 24 * 60
      elif timeframe == "5m":
          bars_per_day = 24 * 12
      # ... etc
      
      bars_per_cycle = cycle_days * bars_per_day
      
      # Total bars needed
      total_needed = num_cycles * bars_per_cycle
      if total_needed > total_bars:
          raise ValueError(f"Need {total_needed} bars but only have {total_bars}")
      
      # Randomly partition
      available_bars = total_bars
      cycles = []
      
      for i in range(num_cycles):
          # Random start within available range
          max_start = available_bars - ((num_cycles - i) * bars_per_cycle)
          if i == 0:
              start = random.randint(0, max_start)
          else:
              start = cycles[-1][1] + 1  # Start after previous cycle
          
          end = start + bars_per_cycle - 1
          cycles.append((start, end))
      
      # Verify
      validate_cycles_non_overlapping(cycles)
      
      return cycles
  ```

**Estimated Time**: 3 hours  
**Complexity**: LOW  
**Impact**: MEDIUM - Ensures valid statistics  

---

### 🟡 P2-3: Implement Complete Main Loop
**File**: `main.py`  
**Current**: Mode 1 structure exists but GA loop incomplete  
**Required**: Full generation loop with selection, refill, repeat  

**Tasks**:
- [ ] Implement complete GA loop in Mode 1:
  ```python
  def run_mode1(params: dict, gpu_context, gpu_queue, gpu_info: dict):
      # ... existing data loading code ...
      
      # Initialize components
      bot_generator = CompactBotGenerator(...)
      backtester = CompactBacktester(...)
      evolver = GeneticAlgorithmEvolver(...)
      
      # Generation 0: Initial population
      log_info("="*60)
      log_info("GENERATION 0: Initial Population")
      log_info("="*60)
      
      population = bot_generator.generate_population()
      
      # Main evolution loop
      for generation in range(params['generations']):
          log_info(f"\n{'='*60}")
          log_info(f"GENERATION {generation + 1}")
          log_info(f"{'='*60}")
          
          # Backtest all bots on all cycles
          log_info(f"Backtesting {len(population)} bots on {params['cycles']} cycles...")
          results = backtester.backtest_bots(population, ohlcv_data, cycles)
          
          # Select survivors (positive avg profit)
          survivors = evolver.select_survivors(results, population)
          
          log_info(f"Survivors: {len(survivors)} / {len(population)}")
          
          if len(survivors) == 0:
              log_warning("No survivors! All bots unprofitable.")
              break
          
          # Release combinations from dead bots
          dead_bots = [b for b in population if b not in survivors]
          bot_generator.release_combinations(dead_bots)
          
          # Refill population
          population = evolver.refill_population(survivors, params['population'])
          
          log_info(f"New population: {len(population)} bots")
      
      # Final selection: Top 10 bots
      log_info("\n" + "="*60)
      log_info("FINAL SELECTION: Top 10 Bots")
      log_info("="*60)
      
      top_bots = evolver.select_top_bots(
          count=10,
          criteria=[
              ('generations_survived', 'desc'),
              ('avg_profit_pct', 'desc'),
              ('avg_winrate', 'desc')
          ]
      )
      
      # Save to file
      save_top_bots(top_bots, filename="results/top_10_bots.json")
      
      # Display results
      display_final_results(top_bots)
  ```

- [ ] Implement top bot selection:
  ```python
  def select_top_bots(self, count: int, criteria: List) -> List[Tuple[CompactBotConfig, BotPerformance]]:
      """Select top bots by multiple criteria."""
      # Get all bots with performance data
      candidates = [(bot_id, perf) for bot_id, perf in self.bot_performances.items()]
      
      # Sort by criteria (lexicographic)
      for criterion, direction in reversed(criteria):
          reverse = (direction == 'desc')
          
          if criterion == 'generations_survived':
              candidates.sort(key=lambda x: x[1].generations_survived, reverse=reverse)
          elif criterion == 'avg_profit_pct':
              candidates.sort(key=lambda x: x[1].get_average_metrics()['avg_profit_pct'], reverse=reverse)
          elif criterion == 'avg_winrate':
              candidates.sort(key=lambda x: x[1].get_average_metrics()['avg_winrate'], reverse=reverse)
      
      # Return top N
      top = candidates[:count]
      return [(self.bot_performances[bot_id].bot, self.bot_performances[bot_id]) for bot_id, _ in top]
  ```

- [ ] Implement bot saving:
  ```python
  def save_top_bots(bots: List[Tuple[CompactBotConfig, BotPerformance]], filename: str):
      """Save top bots to file for reuse in other modes."""
      data = {
          'timestamp': datetime.now().isoformat(),
          'bots': []
      }
      
      for bot, perf in bots:
          data['bots'].append({
              'config': bot.to_dict(),
              'performance': {
                  'generations_survived': perf.generations_survived,
                  'avg_metrics': perf.get_average_metrics(),
                  'generation_history': perf.generation_results
              }
          })
      
      with open(filename, 'w') as f:
          json.dump(data, f, indent=2)
  ```

**Estimated Time**: 6 hours  
**Complexity**: MEDIUM  
**Impact**: HIGH - Completes GA workflow  

---

## PHASE 3: OPTIMIZATION & POLISH (Est: 2-3 days)

### 🟢 P3-1: Optimize Full Kernel
**File**: `src/gpu_kernels/unified_backtest.cl`  
**Current**: Causes OUT_OF_RESOURCES  
**Required**: Optimize to run with 50 indicators  

**Tasks**:
- [ ] Implement local memory caching:
  ```c
  __kernel void unified_backtest(...) {
      // Cache OHLCV data in local memory (shared within work-group)
      __local OHLCVBar local_cache[256];
      
      int local_id = get_local_id(0);
      int group_id = get_group_id(0);
      
      // Load to local memory (cooperative loading)
      for (int i = local_id; i < 256; i += get_local_size(0)) {
          if (cycle_start + i < total_bars) {
              local_cache[i] = ohlcv_data[cycle_start + i];
          }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      
      // Now use local_cache instead of global ohlcv_data
      // Much faster access
  }
  ```

- [ ] Split into multi-kernel pipeline:
  ```c
  // Kernel 1: Calculate all indicators for all bars
  __kernel void compute_indicators(
      __global OHLCVBar *ohlcv,
      __global float *indicator_values,  // [num_bars * 50]
      ...
  );
  
  // Kernel 2: Generate signals per bot
  __kernel void generate_signals(
      __global CompactBotConfig *bots,
      __global float *indicator_values,
      __global float *signals,  // [num_bots * num_bars]
      ...
  );
  
  // Kernel 3: Execute trades
  __kernel void execute_trades(
      __global CompactBotConfig *bots,
      __global float *signals,
      __global BacktestResult *results,
      ...
  );
  ```

**Estimated Time**: 1-2 days (experimental)  
**Complexity**: HIGH  
**Impact**: LOW (minimal kernel works for now)  

---

### 🟢 P3-2: Add Parameter Config Files
**Files**: `config/indicator_params.json`, `config/risk_strategies.json`  
**Current**: Hardcoded ranges  
**Required**: Configurable parameter ranges  

**Tasks**:
- [ ] Create indicator config:
  ```json
  {
    "SMA": {
      "params": [
        {"name": "period", "min": 5, "max": 200, "type": "int"}
      ]
    },
    "RSI": {
      "params": [
        {"name": "period", "min": 5, "max": 50, "type": "int"},
        {"name": "oversold", "min": 20, "max": 40, "type": "float"},
        {"name": "overbought", "min": 60, "max": 80, "type": "float"}
      ]
    },
    ...
  }
  ```

- [ ] Load in generator:
  ```python
  def _load_param_ranges(self) -> np.ndarray:
      """Load parameter ranges from config."""
      with open('config/indicator_params.json') as f:
          config = json.load(f)
      
      ranges = np.zeros((50, 3, 2), dtype=np.float32)
      
      for indicator in self.all_indicators:
          idx = indicator.type_id
          params = config[indicator.name]['params']
          for i, param in enumerate(params):
              ranges[idx, i, 0] = param['min']
              ranges[idx, i, 1] = param['max']
      
      return ranges
  ```

**Estimated Time**: 4 hours  
**Complexity**: LOW  
**Impact**: LOW (nice to have)  

---

## PHASE 4: TESTING & VALIDATION (Est: 1-2 days)

### 🟡 P2-4: Comprehensive Integration Tests
**Files**: `tests/test_ga_workflow.py`, `tests/test_backtest_realism.py`  

**Tasks**:
- [ ] Test complete GA workflow:
  ```python
  def test_full_ga_workflow():
      """Test complete genetic algorithm workflow."""
      # Initialize with small params
      params = {
          'population': 100,
          'generations': 3,
          'cycles': 5,
          ...
      }
      
      # Run Mode 1
      results = run_mode1(params, ...)
      
      # Verify:
      # - Population size maintained
      # - Survivors have positive profit
      # - Combinations tracked correctly
      # - Top 10 selected correctly
      assert len(results['top_bots']) == 10
  ```

- [ ] Test backtest realism:
  ```python
  def test_backtest_fees_liquidation():
      """Test that backtesting applies fees and liquidation correctly."""
      # Create bot with high leverage
      bot = create_test_bot(leverage=20, sl_multiplier=0.04)
      
      # Create data with sharp price drop
      ohlcv = create_crash_data(drop_pct=0.05)
      
      # Backtest
      results = backtest([bot], ohlcv, ...)
      
      # Verify liquidation occurred
      assert results[0].final_balance == 0
  ```

- [ ] Test indicator calculations:
  ```python
  def test_indicator_accuracy():
      """Test indicators match TA-Lib reference."""
      import talib
      
      ohlcv = load_real_data()
      
      # Calculate SMA in kernel
      kernel_sma = calculate_in_kernel('SMA', ohlcv, period=20)
      
      # Calculate with TA-Lib
      talib_sma = talib.SMA(ohlcv['close'], timeperiod=20)
      
      # Should match within numerical precision
      assert np.allclose(kernel_sma, talib_sma, rtol=1e-5)
  ```

**Estimated Time**: 8 hours  
**Complexity**: MEDIUM  
**Impact**: HIGH - Ensures correctness  

---

## TOTAL EFFORT ESTIMATE

| Phase | Priority | Time | Tasks |
|-------|----------|------|-------|
| Phase 1: Critical Fixes | P0-P1 | 2-3 days | 3 major tasks |
| Phase 2: Core Features | P2 | 3-4 days | 4 major tasks |
| Phase 3: Optimization | P3 | 2-3 days | 2 major tasks |
| Phase 4: Testing | P2 | 1-2 days | 3 major tasks |
| **TOTAL** | | **8-12 days** | **12 major tasks** |

---

## SUCCESS CRITERIA

### Minimum Viable Product (MVP)
- [ ] Real backtest kernel with 15+ indicators
- [ ] GA evolver working with compact bots
- [ ] Real Kucoin data integrated
- [ ] Complete GA workflow (10 generations)
- [ ] Top 10 bots saved correctly
- [ ] All tests passing

### Production Ready
- [ ] 50 indicators working (full kernel optimized)
- [ ] Comprehensive test coverage (>80%)
- [ ] User documentation complete
- [ ] Performance meets targets (100K+ sims/sec)
- [ ] Validated against real trading results

---

## IMPLEMENTATION ORDER (Recommended)

1. **Week 1, Days 1-2**: P0-1 (Real backtest kernel) - CRITICAL
2. **Week 1, Days 3-4**: P0-2 (Fix GA evolver) - CRITICAL
3. **Week 1, Day 5**: P1-1 (Data provider integration) - HIGH
4. **Week 2, Day 1**: P1-2 (TP/SL validation) - HIGH
5. **Week 2, Days 2-3**: P2-3 (Complete main loop) - MEDIUM
6. **Week 2, Days 4-5**: P2-1 (Combination tracking) - MEDIUM
7. **Week 3, Day 1**: P2-2 (Cycle validation) - MEDIUM
8. **Week 3, Days 2-3**: P2-4 (Integration tests) - MEDIUM
9. **Week 3, Days 4-5**: P3-2 (Config files) + Documentation - LOW

Total: ~15 working days for complete implementation

---

**END OF TODO**

**Current Status**: 45% complete  
**Next Critical Step**: Implement real backtest kernel (P0-1)  
**Estimated Time to MVP**: 8-10 days  
**Estimated Time to Production**: 15-20 days
