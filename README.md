# GPU-Accelerated Genetic Algorithm Crypto Trading Bot

A high-performance trading bot that uses genetic algorithms and GPU acceleration (via PyOpenCL) to evolve profitable trading strategies for cryptocurrency futures markets.

## 🎯 Overview

This bot implements a sophisticated genetic algorithm that evolves a population of trading bots over multiple generations. Each bot contains:
- **1-5 Technical Indicators** (from a library of 30+) with randomized parameters
- **1-5 Risk Management Strategies** (from 12+ strategies) for position sizing
- **Dynamic TP/SL** with leverage adjustments

The system uses **GPU kernels** for massive parallelization, capable of handling 10,000+ bots simultaneously on modest hardware (3GB VRAM target).

## 🚀 Features

### Mode 1: Genetic Algorithm Evolution (Implemented)
- Evolve trading bots over multiple generations
- Non-overlapping backtest cycles for robust validation
- Automatic selection of profitable bots
- Population refill with unique indicator combinations
- Smart data caching (only downloads missing data)

### Data Management
- **Smart Fetcher**: Downloads only missing data, caches as Parquet files
- **Validation**: Strict data integrity checks (gaps, duplicates, invalid values)
- **Kucoin Futures API**: Real market data from Kucoin
- **Multiple Timeframes**: 1m, 5m, 15m, 30m, 1h, 4h, 1d

### Technical Indicators (30+ Implemented)
**Momentum**: RSI, MACD, Stochastic, CCI, MOM, ROC, Williams %R, Ultimate Oscillator, PPO

**Trend**: EMA, SMA, WMA, DEMA, TEMA, ADX, ADXR, Aroon, TRIX

**Volatility**: ATR, NATR, Bollinger Bands

**Volume**: OBV, AD, ADOSC, MFI

**Statistical**: Beta, Correlation, Linear Regression, StdDev, TSF

**Others**: Parabolic SAR

### Risk Management Strategies (12+ Implemented)
- **Fixed Size**: Fixed %, Fixed Amount
- **Kelly Criterion**: Full, Half, Fractional Kelly
- **Volatility-Based**: ATR-based, StdDev-based
- **Adaptive**: Martingale, Anti-Martingale, Equity Curve
- **Streak-Based**: Win Streak adjustment
- **Progressive**: Gradual size increases

### Realistic Trading Simulation
- **Kucoin Futures Specs**: 0.02% fees, 0.1% slippage model
- **Leverage Support**: 1-125x with proper margin calculations
- **Liquidation Logic**: Maintenance margin & liquidation prices
- **Funding Rates**: 0.01% every 8 hours
- **Multiple Positions**: Up to 10 concurrent positions per bot

## 📁 Project Structure

```
gpu_bot/
├── main.py                          # Entry point
├── requirements.txt                  # Dependencies
├── README.md                         # This file
│
├── src/
│   ├── data_provider/
│   │   ├── fetcher.py               # Smart data fetching from Kucoin
│   │   └── loader.py                # Data loading & cycle generation
│   │
│   ├── indicators/
│   │   ├── factory.py               # 30+ technical indicators
│   │   └── signals.py               # Signal generation rules
│   │
│   ├── risk_management/
│   │   ├── strategies.py            # 12+ position sizing strategies
│   │   └── tp_sl.py                 # TP/SL generation with leverage
│   │
│   ├── bot_generator/
│   │   └── generator.py             # Bot generation (CPU-based for now)
│   │
│   ├── backtester/
│   │   └── simulator.py             # Backtesting engine
│   │
│   ├── ga/
│   │   └── evolver.py               # Genetic algorithm logic
│   │
│   ├── gpu_kernels/
│   │   ├── bot_gen.cl               # OpenCL bot generation kernel
│   │   └── backtest.cl              # OpenCL backtesting kernel
│   │
│   └── utils/
│       ├── config.py                # Configuration constants
│       └── validation.py            # Strict parameter validation
│
├── data/                             # Market data (Parquet files)
└── test/                             # Unit & integration tests
```

## 🔧 Installation

### Prerequisites
- Python 3.12+
- OpenCL-capable GPU (AMD, NVIDIA, or Intel)
- TA-Lib C library

### Windows Installation

```powershell
# Install Python dependencies
pip install -r requirements.txt

# Install TA-Lib
# Download from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib-0.4.28-cp312-cp312-win_amd64.whl
```

### Linux Installation

```bash
# Install TA-Lib C library
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install

# Install Python dependencies
pip install -r requirements.txt
```

## 🎮 Usage

### Mode 1: Genetic Algorithm Evolution

```powershell
python main.py
```

You'll be prompted for parameters:
- **Trading Pair**: e.g., "BTC/USDT" (default)
- **Initial Balance**: Starting capital, e.g., 100 USDT
- **Population Size**: 1000-1,000,000 (default: 10,000)
- **Generations**: 1-100 (default: 10)
- **Cycles per Generation**: 1-100 (default: 10)
- **Backtest Days**: 1-365 (default: 7)
- **Timeframe**: 1m, 5m, 15m, 30m, 1h, 4h, 1d
- **Leverage**: 1-125x (default: 10x)
- **Indicators per Bot**: Min/Max 1-10
- **Risk Strategies per Bot**: Min/Max 1-10

### Example Session

```
Select mode (1-4): 1
Trading pair [BTC/USDT]: BTC/USDT
Initial balance [100.0]: 1000
Population size [10000]: 5000
Generations [10]: 15
Cycles per generation [10]: 10
Backtest days per cycle [7]: 7
Timeframe [1m]: 5m
Leverage [10]: 20
Min indicators per bot [1]: 2
Max indicators per bot [5]: 4
Min risk strategies per bot [1]: 1
Max risk strategies per bot [5]: 3
```

## 📊 Output

Results are saved to `data/best_bots.json`:

```json
{
  "top_bots": [
    {
      "rank": 1,
      "bot_id": 4532,
      "generations_survived": 15,
      "avg_profit_pct": 24.5,
      "avg_winrate": 0.68,
      "avg_trades": 142,
      "indicators": [
        {"type": "RSI", "params": {"period": 14}},
        {"type": "EMA", "params": {"period": 50}}
      ],
      "risk_strategies": [
        {"type": "KELLY_HALF", "params": {}}
      ],
      "take_profit_pct": 5.2,
      "stop_loss_pct": 2.1
    }
  ]
}
```

## 🔬 Technical Details

### GPU Optimization
- **Single Data Load**: Market data loaded once per generation
- **Batch Processing**: All bots processed in parallel
- **Indicator Precomputation**: Calculate all indicator values once
- **Memory Efficiency**: Target <1GB VRAM for 1M bots

### Validation & Error Handling
- **Strict Type Checking**: All parameters validated
- **No Silent Failures**: Raises exceptions on invalid inputs
- **No CPU Fallback**: Crashes if GPU unavailable (by design)
- **Comprehensive Logging**: Debug, Info, Warning, Error levels

### Data Integrity
- **Gap Detection**: Identifies missing candles
- **Duplicate Removal**: Automatic deduplication
- **OHLC Validation**: Ensures valid price relationships
- **Smart Caching**: Only downloads missing date ranges

## 🧪 Testing

```powershell
# Run all tests
pytest test/

# Run with coverage
pytest test/ --cov=src --cov-report=html

# Run specific test file
pytest test/test_validation.py -v
```

## ⚙️ Configuration

Key parameters in `src/utils/config.py`:
- `MAKER_FEE_RATE = 0.0002` (0.02%)
- `TAKER_FEE_RATE = 0.0002` (0.02%)
- `SLIPPAGE_RATE = 0.001` (0.1%)
- `FUNDING_RATE_8H = 0.0001` (0.01%)
- `MAINTENANCE_MARGIN_RATE = 0.005` (0.5%)
- `SIGNAL_CONSENSUS_THRESHOLD = 0.75` (75%)

## 🚧 Development Status

### ✅ Completed
- [x] Project structure & utilities
- [x] Data fetcher with smart caching
- [x] Data loader with validation
- [x] 30+ Technical indicators
- [x] Signal generation rules
- [x] 12+ Risk management strategies
- [x] TP/SL generation with leverage
- [x] Configuration & validation

### 🚧 In Progress
- [ ] GPU kernel for bot generation (bot_gen.cl)
- [ ] GPU kernel for backtesting (backtest.cl)
- [ ] CPU-based bot generator (fallback/testing)
- [ ] Backtester simulator host code
- [ ] Genetic algorithm evolver
- [ ] Main.py integration
- [ ] Unit tests
- [ ] Integration tests

### 📝 TODO
- [ ] Mode 4 implementation (Single bot backtest)
- [ ] Performance benchmarks
- [ ] Web dashboard for results
- [ ] Multi-pair support
- [ ] Paper trading mode

## 📜 License

MIT License - See LICENSE file for details

## ⚠️ Disclaimer

This software is for educational purposes only. Cryptocurrency trading carries significant risk of loss. Never trade with money you cannot afford to lose. Past performance does not guarantee future results.

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit a pull request

## 📞 Support

For issues, questions, or suggestions, please open a GitHub issue.

---

**Built with Python 3.12+ | PyOpenCL | TA-Lib | CCXT**
