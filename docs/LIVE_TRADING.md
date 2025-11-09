# Live Trading Modes - Implementation Guide

## Overview

Modes 2 and 3 provide real-time trading capabilities using CPU-based implementation that **exactly replicates** the GPU kernel logic used in backtesting. This ensures consistency between backtested performance and live trading results.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LIVE TRADING SYSTEM                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │           Real-Time Trading Engine                      │ │
│  │  - Replicates GPU kernel logic on CPU                  │ │
│  │  - Processes live candles bar-by-bar                   │ │
│  │  - Unified for paper and live trading                  │ │
│  └────────────────────────────────────────────────────────┘ │
│           │              │               │                   │
│           ▼              ▼               ▼                   │
│  ┌──────────────┐ ┌─────────────┐ ┌──────────────────┐     │
│  │  Indicator   │ │   Signal    │ │  Risk Manager    │     │
│  │  Calculator  │ │  Generator  │ │  (Position Size) │     │
│  │  (50 types)  │ │ (100% cons.)│ │                  │     │
│  └──────────────┘ └─────────────┘ └──────────────────┘     │
│           │              │               │                   │
│           └──────────────┴───────────────┘                   │
│                          │                                   │
│                          ▼                                   │
│              ┌──────────────────────┐                        │
│              │  Position Manager    │                        │
│              │  ├─ Paper (Mode 2)   │                        │
│              │  └─ Live (Mode 3)    │                        │
│              └──────────────────────┘                        │
│                          │                                   │
│          ┌───────────────┴───────────────┐                   │
│          ▼                               ▼                   │
│  ┌──────────────────┐          ┌──────────────────┐         │
│  │  Kucoin Client   │          │  Live Dashboard  │         │
│  │  (Data & Orders) │          │  (Terminal UI)   │         │
│  └──────────────────┘          └──────────────────┘         │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Mode 2: Paper Trading (Live Simulation)

### Purpose
- Test bots with live data without risking real money
- Track simulated positions with realistic fees and slippage
- Verify bot performance in real market conditions

### Features
- ✅ Fake positions (no real orders)
- ✅ Simulated PnL with Kucoin Futures fees (0.02% maker, 0.06% taker)
- ✅ Slippage modeling (0.1%)
- ✅ Real-time price data from Kucoin
- ✅ Identical logic to GPU backtest kernel
- ✅ Live dashboard with indicators and signals

### Usage
```bash
python main.py
# Select Mode 2
# Follow prompts for configuration
```

### Configuration
1. **Credentials**: First run will prompt for Kucoin API keys (stored encrypted)
2. **Trading Pair**: e.g., BTC/USDT
3. **Initial Balance**: Starting capital (simulated)
4. **Timeframe**: Candle interval (1m, 5m, 15m, etc.)
5. **Bot**: Load from saved results or use test configuration

## Mode 3: Live Trading (Real Money)

### Purpose
- Execute real trades on Kucoin Futures
- Use winning bots from genetic algorithm
- Automated trading with proper risk management

### Features
- ✅ Real order execution
- ✅ Live position tracking
- ✅ Same engine as Mode 2 (consistency guaranteed)
- ✅ Automatic TP/SL management
- ✅ Real-time PnL tracking
- ✅ Safety checks and risk limits

### Safety Features
- 🔒 Encrypted credential storage
- 🔒 Confirmation prompt before trading
- 🔒 Position size limits (2% risk per trade)
- 🔒 Leverage validation
- 🔒 Liquidation monitoring

### Usage
```bash
python main.py
# Select Mode 3
# Type 'I UNDERSTAND THE RISKS'
# Follow prompts for configuration
```

### ⚠️ CRITICAL WARNINGS
- **REAL MONEY**: All trades use real funds
- **CAN LOSE EVERYTHING**: Crypto trading is extremely risky
- **NOT GUARANTEED**: Past backtest performance ≠ future results
- **START SMALL**: Test with minimum position sizes first
- **MONITOR CONSTANTLY**: Check dashboard frequently
- **USE STOP LOSSES**: Always set proper SL levels

## Credential Management

### First-Time Setup
```
1. Run Mode 2 or Mode 3
2. System prompts for Kucoin API credentials
3. Enter master password for encryption
4. Credentials saved to ~/.gpu_bot/credentials.enc
```

### Get Kucoin API Keys
1. Visit https://www.kucoin.com/account/api
2. Create new API key with **Futures Trading** permissions
3. **Restrict to your IP** (recommended)
4. **Disable withdrawals** (highly recommended)
5. Save API Key, Secret, and Passphrase

### Security
- Credentials encrypted using Fernet (AES-128)
- Master password derived with PBKDF2 (100k iterations)
- Files stored in `~/.gpu_bot/` with restrictive permissions
- Never committed to version control

## Components

### 1. RealTimeTradingEngine
**File**: `src/live_trading/engine.py`

Processes live candles and replicates GPU kernel logic:
- Updates indicator calculator with OHLCV data
- Calculates all bot indicators
- Generates 100% consensus signals
- Manages positions (open/close based on signals and TP/SL)
- Tracks statistics and state

### 2. RealTimeIndicatorCalculator
**File**: `src/live_trading/indicator_calculator.py`

Calculates all 50 indicators matching GPU kernel:
- Uses TA-Lib for standard indicators
- Maintains circular buffer of price data
- Supports all GPU kernel indicators (MA, RSI, MACD, ADX, etc.)
- Returns real-time values for signal generation

### 3. SignalGenerator
**File**: `src/live_trading/signal_generator.py`

Generates signals with 100% consensus:
- Classifies each indicator as bullish/bearish/neutral
- Matches GPU kernel `generate_signal_consensus()` exactly
- Returns 1.0 (buy), -1.0 (sell), or 0.0 (no signal)
- Provides breakdown for dashboard display

### 4. Position Managers

#### PaperPositionManager
**File**: `src/live_trading/position_manager.py`

Simulates positions:
- Tracks fake positions in memory
- Applies realistic fees (0.02%/0.06%)
- Models slippage (0.1%)
- Checks TP/SL on every price update
- Calculates PnL exactly as GPU kernel

#### LivePositionManager
**File**: `src/live_trading/position_manager.py`

Executes real trades:
- Places market orders via Kucoin API
- Sets leverage per position
- Monitors positions from exchange
- Auto-closes on TP/SL triggers
- Handles liquidation checks

### 5. KucoinFuturesClient
**File**: `src/live_trading/kucoin_client.py`

Connects to Kucoin Futures API:
- Fetches live ticker data
- Streams OHLCV candles
- Places market orders
- Manages positions
- Supports testnet/live environments

### 6. LiveDashboard
**File**: `src/live_trading/dashboard.py`

Terminal-based real-time display:
- Current price
- All indicators with values and signals
- Signal status (buy/sell/neutral)
- Open/closed positions
- Unrealized/realized PnL
- Balance and stats

### 7. CredentialsManager
**File**: `src/live_trading/credentials.py`

Secure credential storage:
- Encrypts API keys with Fernet
- Derives key from master password
- Auto-prompts on first run
- Stores in `~/.gpu_bot/`

## Signal Generation Logic

### GPU Kernel (Backtest)
```c
// 100% consensus required
if (bullish_pct >= 1.0f) return 1.0f;   // ALL bullish
if (bearish_pct >= 1.0f) return -1.0f;  // ALL bearish
return 0.0f;  // No unanimous consensus
```

### CPU Engine (Live Trading)
```python
# 100% consensus required (matches GPU)
if bullish_pct >= 1.0:
    return 1.0  # ALL bullish
elif bearish_pct >= 1.0:
    return -1.0  # ALL bearish
else:
    return 0.0  # No unanimous consensus
```

### Indicator Classification
Both GPU and CPU use identical rules:
- **RSI**: <30 = bullish, >70 = bearish
- **MACD**: >0 = bullish, <0 = bearish
- **ADX**: >25 with trend = directional signal
- **Moving Averages**: Current > previous = bullish
- **Momentum**: >0 = bullish, <0 = bearish
- etc.

## Position Management

### Opening Positions
```python
1. Signal generated (1.0 or -1.0)
2. Check existing positions (avoid duplicates)
3. Calculate position size (2% risk)
4. Calculate TP/SL prices:
   - Long: TP = price * (1 + tp_multiplier), SL = price * (1 - sl_multiplier)
   - Short: TP = price * (1 - tp_multiplier), SL = price * (1 + sl_multiplier)
5. Execute:
   - Paper: Add to memory, deduct fees
   - Live: Place market order on exchange
```

### Closing Positions
```python
1. Check TP/SL on every candle
2. If triggered:
   - Calculate PnL (entry vs exit, with leverage)
   - Apply exit fees
   - Update balance
3. Execute:
   - Paper: Update memory, track stats
   - Live: Place closing order on exchange
```

### Liquidation Check
```python
pnl_pct = (current_price - entry_price) / entry_price * side
liquidation_threshold = -1.0 / leverage

if pnl_pct <= liquidation_threshold:
    # Position liquidated
    net_pnl = -margin  # Lose entire margin
```

## Testing Workflow

### Recommended Approach
1. **Backtest with Mode 1**: Evolve bots, find top performers
2. **Paper trade with Mode 2**: Test winners with live data (no money risk)
3. **Monitor for 1+ weeks**: Verify consistent performance
4. **Start small with Mode 3**: Use minimum position sizes
5. **Scale gradually**: Increase size only if profitable

### Dashboard Interpretation

**Signal Status**:
- 🟢 BUY SIGNAL: ALL indicators bullish (100% agreement)
- 🔴 SELL SIGNAL: ALL indicators bearish (100% agreement)
- ⚪ NO SIGNAL: Mixed signals (no unanimous consensus)

**Indicator Display**:
```
Name                 Value           Signal     Parameters
RSI_14               32.45           🟢 BUY     [14.00, 0.00, 0.00]
MACD_12_26_9         0.52            🟢 BUY     [12.00, 26.00, 9.00]
ADX_14               28.33           🟢 BUY     [14.00, 0.00, 0.00]
```

**Position Summary**:
```
Open: 1 | Closed: 5
Unrealized PnL: +$45.20
Realized PnL:   +$123.50
Total PnL:      +$168.70
Total Fees:     $12.30
Win Rate:       80.0%
```

## Troubleshooting

### "No OpenCL platforms found"
- Install GPU drivers (NVIDIA CUDA, AMD ROCm, Intel OpenCL)
- GPU still needed for Mode 1, but Modes 2/3 run on CPU

### "Failed to connect to Kucoin"
- Check API credentials
- Verify network connection
- Ensure API permissions include Futures trading
- Check IP whitelist if enabled

### "Position size too small"
- Increase initial balance
- Adjust risk_percent in position manager
- Check minimum order size for symbol

### "Insufficient balance"
- Margin required = (position_value / leverage) + fees
- Reduce position size or increase balance
- Check available balance on exchange (Mode 3)

### Dashboard not updating
- Check data stream connection
- Verify timeframe matches Kucoin support
- Look for errors in terminal output

## Dependencies

New for live trading:
```
cryptography>=41.0.0  # Credential encryption
ccxt>=4.0.0          # Kucoin API (already included)
```

Install:
```bash
pip install cryptography
```

## File Structure

```
src/live_trading/
├── __init__.py              # Module exports
├── credentials.py           # Encrypted credential manager
├── engine.py                # Real-time trading engine
├── indicator_calculator.py  # CPU indicator calculation
├── signal_generator.py      # 100% consensus signals
├── position_manager.py      # Paper & Live position handling
├── risk_manager.py          # Position sizing and safety
├── kucoin_client.py         # Kucoin API integration
└── dashboard.py             # Terminal UI

~/.gpu_bot/                  # User data directory
├── credentials.enc          # Encrypted API keys
└── .key                     # Encryption key
```

## Next Steps

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Get API keys**: Create Kucoin Futures API credentials
3. **Test with Mode 2**: Run paper trading for practice
4. **Monitor performance**: Use dashboard to track signals
5. **Consider Mode 3**: Only if confident with paper trading results

## Support

For issues or questions:
1. Check this README
2. Review error messages in terminal
3. Test with paper trading first
4. Start with minimum position sizes

---

**Remember**: Trading is risky. Never trade more than you can afford to lose. Past performance does not guarantee future results.
