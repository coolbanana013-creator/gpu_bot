# Paper & Live Trading Implementation Status

## ✅ COMPLETED

### Task 1: Kucoin Universal SDK Research
- Analyzed official SDK documentation
- Identified test endpoint: `/api/v1/orders/test` (paper trading)
- Identified live endpoint: `/api/v1/orders` (live trading)
- Located 7 example files showing proper SDK usage

### Task 2: New Kucoin Universal Client
**File Created**: `src/live_trading/kucoin_universal_client.py` (380 lines)
- Replaced CCXT with official Kucoin Universal SDK
- Implemented `test_mode` parameter for paper/live differentiation
- Methods implemented:
  - `create_market_order()` - Uses test/live endpoint based on mode
  - `create_limit_order()` - Supports TP/SL orders
  - `cancel_order()` - Order cancellation
  - `get_order()` - Order status lookup
  - `get_position()` - Position details
  - `fetch_ticker()` - Current price
  - `fetch_ohlcv()` - Historical candles

## 🔄 IN PROGRESS / TO DO

### Tasks 3-9: GPU Kernel Logic Porting (MASSIVE - 6-8 hours)

**Requires complete rewrite of** `src/live_trading/engine.py`

#### Task 3: Dynamic Slippage (Lines 150-187 from kernel)
```python
def calculate_dynamic_slippage(
    position_value, current_volume, leverage, 
    current_price, current_high, current_low
) -> float:
    # BASE_SLIPPAGE = 0.0001
    # Volume impact: position size % of current volume
    # Volatility multiplier: high-low range / price
    # Leverage multiplier: 1.0 + (leverage / 62.5)
    # Combine: (slippage + volume_impact) * volatility * leverage
    # Bounds: [0.00005, 0.005]
```

#### Task 4: Funding Rates (Lines 1040-1056 from kernel)
```python
def apply_funding_rates(position, bars_held) -> float:
    # FUNDING_INTERVAL = 480 bars (8 hours at 1m)
    # BASE_FUNDING_RATE = 0.0001 (0.01%)
    # prev_periods = (bars_held - 1) // FUNDING_INTERVAL
    # curr_periods = bars_held // FUNDING_INTERVAL
    # If curr_periods > prev_periods:
    #     position_value = entry_price * size * leverage
    #     funding_cost = position_value * BASE_FUNDING_RATE
    #     Long positions PAY, short positions RECEIVE
```

#### Task 5: Account Liquidation (Lines 271-309 from kernel)
```python
def check_account_liquidation(current_price) -> bool:
    # total_unrealized_pnl = sum(unrealized_pnl for all positions)
    # total_used_margin = sum(entry_price * size for all positions)
    # equity = balance + total_unrealized_pnl
    # maintenance_margin = total_used_margin * 0.005 * leverage
    # return equity < maintenance_margin
```

#### Task 6: Signal Reversal (Lines 1093-1099 from kernel)
```python
def check_signal_reversal(position, current_signal) -> bool:
    # If long position and signal turns bearish: close
    # If short position and signal turns bullish: close
    # return (position.side == 1 and current_signal < 0) or
    #        (position.side == -1 and current_signal > 0)
```

#### Task 7: True Margin Trading - open_position() (Lines 802-814 from kernel)
```python
def open_position(...):
    # position_value = from risk strategy
    # margin_required = position_value / leverage  # CRITICAL
    # slippage_rate = calculate_dynamic_slippage(...)
    # entry_fee = position_value * 0.0006  # Taker fee
    # slippage_cost = position_value * slippage_rate
    # total_cost = margin_required + entry_fee + slippage_cost
    # quantity = margin_required / price  # Based on MARGIN not full value
    # Deduct total_cost from balance
    # Set TP/SL prices
    # Calculate liquidation price
```

#### Task 8: 100% Consensus Signal Generation (Lines 540-780 from kernel)
**MASSIVE**: All 50 indicators with exact signal logic:
- Moving Averages (0-11): Price crosses, momentum
- Momentum (12-19): RSI, Stochastic, MACD overbought/oversold
- Volatility (20-25): ATR, Bollinger Bands expansion/contraction
- Trend (26-35): ADX, Aroon, CCI trend strength
- Volume (36-40): OBV, VWAP, MFI volume confirmation
- Pattern (41-45): Pivot points, fractals, support/resistance
- Simple (46-49): High-low range, close position, acceleration

**100% consensus**: ALL indicators must agree (bullish_pct >= 1.0 or bearish_pct >= 1.0)

#### Task 9: All 15 Risk Strategies (Lines 311-450 from kernel)
```python
def calculate_position_size(balance, price, risk_strategy, risk_param) -> float:
    # RISK_FIXED_PCT: balance * risk_param (0.01-0.20)
    # RISK_FIXED_USD: fixed amount (10-10000)
    # RISK_KELLY_FULL: balance * risk_param
    # RISK_KELLY_HALF: balance * (risk_param * 0.5)
    # RISK_KELLY_QUARTER: balance * (risk_param * 0.25)
    # RISK_ATR_MULTIPLIER: balance * 0.05 * risk_param
    # RISK_VOLATILITY_PCT: balance * risk_param
    # RISK_EQUITY_CURVE: balance * 0.05 * risk_param
    # RISK_FIXED_RISK_REWARD: balance * risk_param
    # RISK_MARTINGALE: balance * 0.05 * risk_param (DANGEROUS)
    # RISK_ANTI_MARTINGALE: balance * 0.05 * risk_param
    # RISK_FIXED_RATIO: balance * 0.05
    # RISK_PERCENT_VOLATILITY: balance * risk_param
    # RISK_WILLIAMS_FIXED: balance * risk_param
    # RISK_OPTIMAL_F: balance * risk_param
    # Bounds: min $10, max 20% of balance
```

### Tasks 10-16: Dashboard Enhancement (MASSIVE - 4 hours)

**Requires complete rewrite of** `src/live_trading/dashboard.py`

#### Task 10: Runtime Tracking
```python
def __init__(self):
    self.start_time = time.time()

def format_runtime(self, seconds) -> str:
    # Convert to HH:MM:SS format
```

#### Task 11: Mode Banner
```python
mode = "🟢 PAPER TRADING" if state['mode'] == 'paper' else "🔴 LIVE TRADING"
print(f"{mode:^80}")
```

#### Task 12: Balance Breakdown
```python
print(f"   Initial:    ${state['initial_balance']:,.2f}")
print(f"   Realized:   ${state['realized_pnl']:+,.2f}")
print(f"   Unrealized: ${state['unrealized_pnl']:+,.2f}")
print(f"   Current:    ${state['current_balance']:,.2f}")
```

#### Task 13: Leverage & Risk Display
```python
print(f"   Leverage: {state['leverage']}x")
print(f"   TP: {state['tp_multiplier']:.2f}x | SL: {state['sl_multiplier']:.2f}x")
print(f"   Risk Strategy: {RISK_STRATEGY_NAMES[state['risk_strategy']]}")
print(f"   Risk Parameter: {state['risk_param']:.4f}")
```

#### Task 14: Indicator Threshold Display (CRITICAL FEATURE)
```python
print(f"{'Indicator':<20} {'Current':<15} {'Bullish When':<25} {'Bearish When':<25} {'Signal':<10}")
print("-" * 100)

for ind in state['indicator_details']:
    # Example: RSI(14): Current=56.3, Bullish When: <30, Bearish When: >70, Signal: NEUTRAL
    signal_str = "🟢 BUY" if ind['signal'] == 1 else ("🔴 SELL" if ind['signal'] == -1 else "⚪ NEUTRAL")
    print(f"{ind['name']:<20} {ind['value']:<15.4f} {ind['bullish_condition']:<25} {ind['bearish_condition']:<25} {signal_str:<10}")
```

#### Task 15: Open Positions Detail
```python
print(f"{'Side':<8} {'Entry':<12} {'Size':<12} {'TP':<12} {'SL':<12} {'Current PnL':<15}")
for pos in state['open_positions']:
    side_str = "🟢 LONG" if pos['side'] == 1 else "🔴 SHORT"
    print(f"{side_str:<8} ${pos['entry_price']:<11,.2f} {pos['size']:<12.6f} ${pos['tp_price']:<11,.2f} ${pos['sl_price']:<11,.2f} ${pos['unrealized_pnl']:+,.2f}")
```

#### Task 16: Closed Positions Detail
```python
print(f"{'Side':<8} {'Entry':<12} {'Exit':<12} {'PnL':<15} {'Result':<10}")
for pos in state['closed_positions'][:5]:  # Last 5
    result_str = "WIN" if pos['pnl'] > 0 else "LOSS"
    print(f"{side_str:<8} ${pos['entry_price']:<11,.2f} ${pos['exit_price']:<11,.2f} ${pos['pnl']:+,.2f} {result_str:<10}")
```

### Tasks 17-18: Bot Config Loading (2 hours)

**Requires updates to** `main.py`

#### Task 17: Improved Bot Loading UI
```python
def load_saved_bot():
    bot_files = glob.glob("bots/**/*.json", recursive=True)
    bots_data = []
    for f in bot_files:
        with open(f, 'r') as file:
            bot_data = json.load(file)
            bot_data['file'] = f
            bots_data.append(bot_data)
    
    # Sort by fitness descending
    bots_data.sort(key=lambda x: x.get('fitness_score', 0), reverse=True)
    
    # Display top 20
    print(f"{'#':<4} {'Bot ID':<10} {'Fitness':<12} {'Survival':<10} {'Win Rate':<12} {'File':<40}")
    for i, bot in enumerate(bots_data[:20]):
        print(f"{i+1:<4} {bot.get('bot_id'):<10} {bot.get('fitness_score'):<12.2f} {bot.get('survival_generations'):<10} {bot.get('win_rate')*100:<12.1f}% {bot['file']:<40}")
```

#### Task 18: Config Validation
```python
def validate_bot_config(bot_data):
    required_fields = ['bot_id', 'leverage', 'tp_multiplier', 'sl_multiplier', 
                      'num_indicators', 'indicator_indices', 'indicator_params',
                      'risk_strategy', 'risk_param']
    
    for field in required_fields:
        if field not in bot_data:
            raise ValueError(f"Missing required field: {field}")
    
    # Validate ranges
    assert 1 <= bot_data['leverage'] <= 125
    assert 0 < bot_data['tp_multiplier'] <= 1.0
    assert 0 < bot_data['sl_multiplier'] <= 0.95 / bot_data['leverage']
    # ... more validation
```

### Task 19: SDK Method Testing (1 hour)

**Create new file**: `scripts/test_kucoin_sdk.py`

Test all methods:
1. `create_market_order()` - Test and live mode
2. `create_limit_order()` - Test and live mode
3. `cancel_order()`
4. `get_order()`
5. `get_position()`
6. `fetch_ticker()`
7. `fetch_ohlcv()`
8. Error handling for invalid inputs
9. Rate limit handling
10. Connection resilience

### Tasks 20-22: Integration Testing (2 hours)

#### Task 20: Paper Trading Integration Test
```python
def test_paper_trading_session():
    # 1. Load test bot config
    # 2. Initialize engine with test_mode=True
    # 3. Run for 100 candles
    # 4. Verify orders use /api/v1/orders/test endpoint
    # 5. Verify no real money deducted
    # 6. Verify positions tracked correctly
    # 7. Verify dashboard updates
```

#### Task 21: CPU vs GPU Comparison Test
```python
def test_cpu_vs_gpu_equivalence():
    # 1. Load same bot config
    # 2. Use same historical data (1000 bars)
    # 3. Run GPU backtest
    # 4. Run CPU engine bar-by-bar
    # 5. Compare:
    #    - Total trades (should match exactly)
    #    - Win rate (should match within 0.1%)
    #    - Total PnL (should match within 0.01%)
    #    - Final balance (should match within $0.01)
```

#### Task 22: Dashboard Integration Test
```python
def test_dashboard_display():
    # 1. Initialize trading engine
    # 2. Open 2 positions (1 long, 1 short)
    # 3. Close 1 position with profit
    # 4. Render dashboard
    # 5. Verify all sections present:
    #    - Mode banner
    #    - Runtime
    #    - Balance breakdown
    #    - Leverage/risk
    #    - Indicator thresholds (ALL 50 indicators)
    #    - Open positions (2 showing)
    #    - Closed positions (1 showing)
```

## 📊 IMPLEMENTATION STATISTICS

- **Total Tasks**: 22
- **Completed**: 2 (9%)
- **In Progress**: 0
- **To Do**: 20 (91%)

**Estimated Time Remaining**: 15-17 hours

**Files to Create/Modify**:
1. ✅ `src/live_trading/kucoin_universal_client.py` (NEW, 380 lines)
2. ⏳ `src/live_trading/engine.py` (REWRITE, ~800 lines)
3. ⏳ `src/live_trading/dashboard.py` (REWRITE, ~300 lines)
4. ⏳ `main.py` (UPDATE bot loading, ~100 lines changed)
5. ⏳ `scripts/test_kucoin_sdk.py` (NEW, ~400 lines)
6. ⏳ `scripts/test_paper_trading.py` (NEW, ~200 lines)
7. ⏳ `scripts/test_cpu_vs_gpu.py` (NEW, ~300 lines)

**Lines of Code**: ~2,480 lines total

## 🚀 NEXT STEPS TO COMPLETE

1. **Port GPU Kernel Logic** (Tasks 3-9) - 6-8 hours
   - Create `src/live_trading/gpu_logic_port.py` with all helper functions
   - Update `engine.py` to use these functions
   - Test each function individually against GPU kernel output

2. **Enhance Dashboard** (Tasks 10-16) - 4 hours
   - Rewrite `dashboard.py` with all new sections
   - Create indicator threshold mapping for all 50 indicators
   - Test with live data

3. **Improve Bot Loading** (Tasks 17-18) - 2 hours
   - Update `main.py` bot selection UI
   - Add comprehensive validation

4. **Create Test Suite** (Tasks 19-22) - 3 hours
   - SDK method tests
   - Paper trading integration test
   - CPU vs GPU equivalence test
   - Dashboard rendering test

**Total Remaining**: 15-17 hours of focused development

## ⚠️ CRITICAL DEPENDENCIES

- Official Kucoin Universal SDK must be installed: `pip install kucoin-universal-sdk`
- GPU kernel file must be accessible for logic reference
- Saved bot configs must exist in `bots/` directory
- Test API credentials must be configured

## 📝 RECOMMENDATION

Given the scope, implement in phases:

**Phase 1** (HIGH PRIORITY - 8 hours):
- Tasks 3-7: Core GPU logic (slippage, funding, liquidation, positions)
- Tasks 10-13: Basic dashboard enhancements
- Task 19: SDK testing

**Phase 2** (MEDIUM PRIORITY - 5 hours):
- Task 8: Signal consensus (all 50 indicators)
- Task 9: All 15 risk strategies
- Tasks 14-16: Advanced dashboard sections

**Phase 3** (LOW PRIORITY - 4 hours):
- Tasks 17-18: Bot loading improvements
- Tasks 20-22: Integration testing

This allows testing and validation at each phase before proceeding.
