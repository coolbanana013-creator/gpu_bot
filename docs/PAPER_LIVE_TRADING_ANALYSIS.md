# Paper & Live Trading System - Analysis & Implementation Plan

## Current State Analysis

### ✅ What EXISTS and WORKS:
1. **Basic Infrastructure** (`src/live_trading/`)
   - `credentials.py`: Kucoin API credential management
   - `kucoin_client.py`: CCXT-based API wrapper for Kucoin Futures
   - `engine.py`: Real-time trading engine framework
   - `position_manager.py`: Paper and Live position managers
   - `dashboard.py`: Terminal-based live dashboard
   - `indicator_calculator.py`: Real-time indicator computation
   - `signal_generator.py`: Signal generation from indicators
   - `risk_manager.py`: Risk management logic

2. **Main Entry Points** (`main.py`)
   - Mode 2: Paper trading entry point (lines 886-950)
   - Mode 3: Live trading entry point (lines 1094-1160)
   - Both can load saved bots from GA evolution

3. **Dashboard Features** (PARTIAL)
   - Current price display
   - Signal status (buy/sell/neutral)
   - Indicator breakdown with signals
   - Position summary (open/closed counts)
   - PnL tracking (realized/unrealized/total)
   - Balance display
   - Statistics counter

### ❌ What's MISSING or INCOMPLETE:

#### 1. **API Endpoint Differentiation**
**Problem**: Paper and live trading don't use different API endpoints
- Paper trading should use: `/api/v1/orders/test` (Kucoin test orders)
- Live trading should use: `/api/v1/orders` (real orders)
- **Current**: Both use same `create_market_order()` method

**Solution**: Add `test_mode` parameter to order functions:
```python
def create_market_order(self, symbol, side, size, leverage=1, test_mode=False):
    if test_mode:
        # Use /api/v1/orders/test endpoint
        order = self.exchange.create_order(
            symbol=symbol,
            type='market',
            side=side,
            amount=size,
            params={'test': True}  # CCXT test order flag
        )
    else:
        # Use /api/v1/orders endpoint (real trading)
        order = self.exchange.create_market_order(...)
```

#### 2. **GPU Kernel Logic Replication**
**Problem**: CPU trading engine doesn't perfectly replicate GPU backtest logic

**GPU Kernel Does** (`backtest_with_precomputed.cl`):
- Dynamic slippage calculation based on volume, volatility, leverage (lines 150-187)
- Funding rates every 8 hours (lines 1040-1056)
- Account-level liquidation check (lines 271-309)
- True margin trading: margin = position_value / leverage (lines 802-814)
- Signal reversal exits (lines 1093-1099)
- TP/SL with proper maker/taker fees (lines 913-920)
- 100% consensus signal generation (lines 540-780)
- 15 risk strategies for position sizing (lines 311-450)

**CPU Engine Currently Does** (`engine.py`):
- Basic indicator calculation
- Simple signal generation
- Basic position management
- ❌ NO dynamic slippage
- ❌ NO funding rates
- ❌ NO account-level liquidation
- ❌ Simplified margin calculation
- ❌ NO signal reversal logic
- ❌ Missing some risk strategies

**Action Required**: Port ALL GPU kernel logic to CPU engine

#### 3. **Enhanced Dashboard Requirements**

**MISSING Dashboard Features**:
1. **Pair Runtime**: Time bot has been running
2. **Leverage Display**: Current leverage used
3. **Open Positions Detail**: Each position's entry price, TP, SL, current PnL
4. **Closed Positions Detail**: Recent closed trades with entry/exit/PnL
5. **Risk Management Display**: Which strategy is active, parameters
6. **Indicator Threshold Display**: 
   ```
   RSI(14): Current=56.3, Need: >70 (BUY) or <30 (SELL) → NEUTRAL
   MACD: Current=+12.5, Need: >0 (BUY) → BULLISH
   EMA(20): Current=45231.2, Prev=45198.7 → BULLISH (rising)
   ```
7. **Mode Indicator**: Clear "PAPER TRADING" vs "LIVE TRADING" banner
8. **Total Balance Breakdown**: Initial + Realized + Unrealized

**Action Required**: Expand dashboard with these sections

#### 4. **Bot Config Loading**
**Problem**: Bot loading from JSON exists but needs verification

**Current**: Lines 946-976 in main.py load from `bots/**/*.json`
**Works**: Yes, but needs:
- Better error handling
- Validation of loaded config
- Display of bot's historical performance
- Ability to select from top N bots by fitness

#### 5. **Unified Trading Engine**
**Problem**: `PaperPositionManager` and `LivePositionManager` have different implementations

**Solution**: Create single base engine with only API endpoint difference:
```python
class UnifiedTradingEngine:
    def __init__(self, bot_config, mode='paper'):
        self.mode = mode  # 'paper' or 'live'
        self.api_client = KucoinFuturesClient(test_mode=(mode=='paper'))
        
    def open_position(self, ...):
        # Same logic for both
        pnl, fees, slippage = self._calculate_costs(...)
        
        if self.mode == 'paper':
            # Store fake position, use test API endpoint
            order = self.api_client.create_market_order(..., test_mode=True)
        else:
            # Real position, use real API endpoint
            order = self.api_client.create_market_order(..., test_mode=False)
```

---

## Implementation Plan

### Phase 1: Fix API Endpoint Differentiation (HIGH PRIORITY)

**File**: `src/live_trading/kucoin_client.py`

**Changes**:
1. Add `test_mode` parameter to `__init__`:
   ```python
   def __init__(self, credentials, testnet=False, test_orders=False):
       self.test_orders = test_orders  # NEW: use test endpoints
   ```

2. Modify `create_market_order()`:
   ```python
   def create_market_order(self, symbol, side, size, leverage=1):
       self.exchange.set_leverage(leverage, symbol)
       
       if self.test_orders:
           # Paper trading: use test endpoint
           order = self.exchange.create_order(
               symbol=symbol,
               type='market',
               side=side,
               amount=size,
               params={'test': True}
           )
       else:
           # Live trading: real order
           order = self.exchange.create_market_order(...)
   ```

3. Update Mode 2 (paper) and Mode 3 (live) to pass correct flag

**Estimated Time**: 1 hour

---

### Phase 2: Port GPU Kernel Logic to CPU Engine (CRITICAL)

**File**: `src/live_trading/engine.py`

**Missing Functions to Add**:

1. **Dynamic Slippage** (from kernel lines 150-187):
```python
def calculate_dynamic_slippage(
    self,
    position_value: float,
    current_volume: float,
    leverage: int,
    current_price: float,
    current_high: float,
    current_low: float
) -> float:
    """Calculate dynamic slippage (matches GPU kernel exactly)."""
    slippage = 0.0001  # Base slippage
    
    # Volume impact
    if current_volume > 0:
        position_pct = position_value / (current_volume * current_price)
        volume_impact = min(position_pct * 0.01, 0.005)
    else:
        volume_impact = 0.0
    
    # Volatility multiplier
    if current_price > 0:
        range_pct = (current_high - current_low) / current_price
        volatility_multiplier = min(1.0 + (range_pct / 0.02), 4.0)
    else:
        volatility_multiplier = 1.0
    
    # Leverage multiplier
    leverage_multiplier = 1.0 + (leverage / 62.5)
    
    # Combine
    total_slippage = (slippage + volume_impact) * volatility_multiplier * leverage_multiplier
    return max(0.00005, min(total_slippage, 0.005))
```

2. **Funding Rates** (from kernel lines 1040-1056):
```python
def apply_funding_rates(self, position: Position, bars_held: int) -> float:
    """Apply funding rate every 8 hours (480 bars at 1m)."""
    FUNDING_INTERVAL = 480
    BASE_FUNDING_RATE = 0.0001
    
    prev_periods = (bars_held - 1) // FUNDING_INTERVAL
    curr_periods = bars_held // FUNDING_INTERVAL
    
    if curr_periods > prev_periods:
        position_value = position.entry_price * position.size * position.leverage
        funding_cost = position_value * BASE_FUNDING_RATE
        
        if position.side == 1:  # Long pays
            return -funding_cost
        else:  # Short receives
            return funding_cost
    return 0.0
```

3. **Account-Level Liquidation** (from kernel lines 271-309):
```python
def check_account_liquidation(self, current_price: float) -> bool:
    """Check if account should be liquidated (all positions)."""
    total_unrealized_pnl = 0.0
    total_used_margin = 0.0
    
    for pos in self.position_manager.open_positions:
        total_unrealized_pnl += self._calculate_unrealized_pnl(pos, current_price)
        total_used_margin += pos.entry_price * pos.size
    
    if total_used_margin == 0:
        return False
    
    equity = self.current_balance + total_unrealized_pnl
    maintenance_margin = total_used_margin * 0.005 * self.bot_config.leverage
    
    return equity < maintenance_margin
```

4. **Signal Reversal Exits** (from kernel lines 1093-1099):
```python
def check_signal_reversal(self, position: Position, current_signal: float) -> bool:
    """Check if signal reversed (exit trigger)."""
    if position.side == 1 and current_signal < 0:  # Long but bearish signal
        return True
    if position.side == -1 and current_signal > 0:  # Short but bullish signal
        return True
    return False
```

5. **True Margin Trading Calculation** (from kernel lines 802-814):
```python
def open_position(self, direction: int, price: float, ...):
    """Open position with true margin trading logic."""
    # Calculate position value from risk strategy
    position_value = self._calculate_position_size(self.current_balance, price)
    
    # Margin required = position_value / leverage (CRITICAL)
    margin_required = position_value / self.bot_config.leverage
    
    # Dynamic slippage
    slippage_rate = self.calculate_dynamic_slippage(
        position_value, current_volume, self.bot_config.leverage,
        price, current_high, current_low
    )
    
    # Fees on FULL position value (leverage amplifies)
    entry_fee = position_value * 0.0006  # Taker fee
    slippage_cost = position_value * slippage_rate
    
    total_cost = margin_required + entry_fee + slippage_cost
    
    # Check free margin (balance + unrealized PnL - used margin)
    free_margin = self._calculate_free_margin(price)
    if free_margin < total_cost:
        return None
    
    # Calculate quantity based on MARGIN (not full position value)
    quantity = margin_required / price
    
    # Create position
    position = Position(
        entry_price=price,
        size=quantity,
        side=direction,
        leverage=self.bot_config.leverage,
        tp_price=self._calculate_tp(price, direction),
        sl_price=self._calculate_sl(price, direction)
    )
    
    self.current_balance -= total_cost
    return position
```

**Estimated Time**: 6-8 hours

---

### Phase 3: Enhanced Dashboard (HIGH PRIORITY)

**File**: `src/live_trading/dashboard.py`

**New Sections to Add**:

```python
def render(self, state: Dict):
    self.clear_screen()
    
    # Header with MODE banner
    mode = "🟢 PAPER TRADING" if state['mode'] == 'paper' else "🔴 LIVE TRADING"
    print("=" * 80)
    print(f"{mode:^80}")
    print(f"{'GPU BOT - LIVE TRADING DASHBOARD':^80}")
    print("=" * 80)
    
    # Runtime tracking
    runtime_seconds = time.time() - state['start_time']
    runtime_str = self._format_runtime(runtime_seconds)
    print(f"Runtime: {runtime_str} | Pair: {state['pair']} | Bot ID: {state['bot_id']}")
    
    # Current Price
    print(f"\n💰 CURRENT PRICE: ${state['price']:,.2f}")
    
    # Balance Breakdown
    print(f"\n💵 BALANCE:")
    print(f"   Initial:    ${state['initial_balance']:,.2f}")
    print(f"   Realized:   ${state['realized_pnl']:+,.2f}")
    print(f"   Unrealized: ${state['unrealized_pnl']:+,.2f}")
    print(f"   Current:    ${state['current_balance']:,.2f} ({state['pnl_pct']:+.2f}%)")
    
    # Leverage & Risk
    print(f"\n⚖️ LEVERAGE & RISK:")
    print(f"   Leverage: {state['leverage']}x")
    print(f"   TP: {state['tp_multiplier']:.2f}x | SL: {state['sl_multiplier']:.2f}x")
    print(f"   Risk Strategy: {state['risk_strategy_name']}")
    print(f"   Risk Parameter: {state['risk_param']:.4f}")
    
    # Signal Status
    signal = state['current_signal']
    print(f"\n📊 SIGNAL STATUS:")
    if signal > 0:
        print(f"   Status: 🟢 BUY SIGNAL (All {state['indicator_count']} indicators agree)")
    elif signal < 0:
        print(f"   Status: 🔴 SELL SIGNAL (All {state['indicator_count']} indicators agree)")
    else:
        print(f"   Status: ⚪ NO SIGNAL (No 100% consensus)")
    
    # CRITICAL: Indicator Thresholds
    print(f"\n📈 INDICATORS - REAL-TIME VALUES vs THRESHOLDS:")
    print(f"{'Indicator':<20} {'Current':<15} {'Bullish When':<25} {'Bearish When':<25} {'Signal':<10}")
    print("-" * 100)
    
    for ind in state['indicator_details']:
        signal_str = "🟢 BUY" if ind['signal'] == 1 else ("🔴 SELL" if ind['signal'] == -1 else "⚪ NEUTRAL")
        print(f"{ind['name']:<20} {ind['value']:<15.4f} {ind['bullish_condition']:<25} {ind['bearish_condition']:<25} {signal_str:<10}")
    
    # Open Positions Detail
    print(f"\n💼 OPEN POSITIONS ({state['open_positions_count']}):")
    if state['open_positions_count'] > 0:
        print(f"{'Side':<8} {'Entry':<12} {'Size':<12} {'TP':<12} {'SL':<12} {'Current PnL':<15}")
        print("-" * 80)
        for pos in state['open_positions']:
            side_str = "🟢 LONG" if pos['side'] == 1 else "🔴 SHORT"
            print(f"{side_str:<8} ${pos['entry_price']:<11,.2f} {pos['size']:<12.6f} ${pos['tp_price']:<11,.2f} ${pos['sl_price']:<11,.2f} ${pos['unrealized_pnl']:+,.2f}")
    else:
        print("   No open positions")
    
    # Closed Positions (last 5)
    print(f"\n📋 RECENT CLOSED POSITIONS (Last 5):")
    if len(state['closed_positions']) > 0:
        print(f"{'Side':<8} {'Entry':<12} {'Exit':<12} {'PnL':<15} {'Result':<10}")
        print("-" * 60)
        for pos in state['closed_positions'][:5]:
            side_str = "🟢 LONG" if pos['side'] == 1 else "🔴 SHORT"
            result_str = "WIN" if pos['pnl'] > 0 else "LOSS"
            print(f"{side_str:<8} ${pos['entry_price']:<11,.2f} ${pos['exit_price']:<11,.2f} ${pos['pnl']:+,.2f} {result_str:<10}")
    else:
        print("   No closed positions yet")
    
    # Statistics
    print(f"\n📊 STATISTICS:")
    print(f"   Total Trades: {state['total_trades']} | Win Rate: {state['win_rate']*100:.1f}%")
    print(f"   Wins: {state['wins']} | Losses: {state['losses']}")
    print(f"   Largest Win: ${state['largest_win']:+,.2f} | Largest Loss: ${state['largest_loss']:+,.2f}")
    print(f"   Candles Processed: {state['candles_processed']}")
    
    print("\n" + "=" * 80)
    print("Press Ctrl+C to stop trading")
```

**Estimated Time**: 4 hours

---

### Phase 4: Bot Config Loading Improvements

**File**: `main.py`

**Enhancements**:
1. Display bot's backtest performance when loading
2. Sort by fitness score
3. Show survival generations
4. Validate config before starting

```python
def load_saved_bot():
    """Load bot from evolution results with enhanced display."""
    import glob
    import json
    
    bot_files = glob.glob("bots/**/*.json", recursive=True)
    if not bot_files:
        log_error("No saved bots found")
        return None
    
    # Load and parse all bots
    bots_data = []
    for f in bot_files:
        try:
            with open(f, 'r') as file:
                bot_data = json.load(file)
                bot_data['file'] = f
                bots_data.append(bot_data)
        except:
            continue
    
    # Sort by fitness (descending)
    bots_data.sort(key=lambda x: x.get('fitness_score', 0), reverse=True)
    
    # Display top 20
    print("\n🏆 TOP BOTS FROM EVOLUTION:")
    print(f"{'#':<4} {'Bot ID':<10} {'Fitness':<12} {'Survival':<10} {'Win Rate':<12} {'File':<40}")
    print("-" * 100)
    
    for i, bot in enumerate(bots_data[:20]):
        print(f"{i+1:<4} {bot.get('bot_id', 'N/A'):<10} {bot.get('fitness_score', 0):<12.2f} "
              f"{bot.get('survival_generations', 0):<10} {bot.get('win_rate', 0)*100:<12.1f}% "
              f"{bot['file']:<40}")
    
    # Select bot
    choice = int(input("\nSelect bot number (1-20): ")) - 1
    if 0 <= choice < len(bots_data):
        return bots_data[choice]
    return None
```

**Estimated Time**: 2 hours

---

## Total Estimated Implementation Time: 13-15 hours

## Testing Plan

1. **Phase 1 Test**: Verify paper trading uses test endpoint, no real money moved
2. **Phase 2 Test**: Compare CPU engine output with GPU backtest on same data
3. **Phase 3 Test**: Visual inspection of dashboard, all fields populated correctly
4. **Phase 4 Test**: Load top bot, verify it matches saved config

---

## Critical Success Criteria

✅ Paper and live trading use DIFFERENT API endpoints
✅ CPU engine produces IDENTICAL results to GPU kernel on same data
✅ Dashboard shows ALL required information (especially indicator thresholds)
✅ Bot loading works flawlessly with performance display
✅ Mode indicator clearly shows PAPER vs LIVE
✅ No real money can be spent in paper mode

---

## Priority Order

1. **URGENT**: Phase 1 (API endpoints) - prevents accidental real trades
2. **CRITICAL**: Phase 2 (GPU logic replication) - ensures accuracy
3. **HIGH**: Phase 3 (dashboard) - enables monitoring and debugging
4. **MEDIUM**: Phase 4 (bot loading) - quality of life improvement
