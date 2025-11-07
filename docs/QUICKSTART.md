# Quick Start Guide

## 🚀 Get Running in 5 Minutes

### Step 1: Install Dependencies

```powershell
# Install Python packages
pip install numpy pandas pyarrow ccxt pytest

# Install TA-Lib
# Windows: Download wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib-0.4.28-cp312-cp312-win_amd64.whl

# OR use pip (may require compilation)
pip install TA-Lib
```

### Step 2: Run with Defaults

```powershell
python main.py
```

Then press Enter for all prompts to use defaults:
- Mode: 1
- Pair: BTC/USDT
- Balance: 100 USDT
- Population: 10,000 bots
- Generations: 10
- Cycles: 10
- Days: 7
- Timeframe: 1m
- Leverage: 10x
- Indicators: 1-5 per bot
- Strategies: 1-5 per bot
- Random seed: 42

### Step 3: Wait for Results

The bot will:
1. Download 88 days of BTC/USDT 1m data (~5 minutes first run, cached after)
2. Generate 10,000 bots with random configurations
3. Run 10 generations of evolution (~30-60 minutes)
4. Save top 10 bots to `data/best_bots.json`

### Step 4: Check Results

```powershell
# View results
cat data/best_bots.json

# Or open in text editor
notepad data/best_bots.json
```

## 🎯 Quick Test Run (Fast)

For a quick test to verify everything works:

```powershell
python main.py
```

Use these settings:
- Population: **1000** (instead of 10,000)
- Generations: **3** (instead of 10)
- Cycles: **3** (instead of 10)
- Days: **3** (instead of 7)
- Timeframe: **5m** (instead of 1m)

This will complete in ~5-10 minutes.

## 📊 Understanding Output

### Console Output
```
Generation 0 Summary:
  Population: 1000 bots
  Profitable: 247 bots
  Total bots tracked: 1000

TOP 10 BOTS
Rank #1
  Bot ID: 4532
  Generations Survived: 3
  Average Profit: 12.45%
  Average Winrate: 64.2%
  Average Trades: 127.3
  Indicators: 3
  Risk Strategies: 2
```

### JSON Output (`data/best_bots.json`)
```json
{
  "total_generations": 3,
  "total_bots_evaluated": 1000,
  "top_bots": [
    {
      "rank": 1,
      "bot_id": 4532,
      "generations_survived": 3,
      "avg_profit_pct": 12.45,
      "avg_winrate": 0.642,
      "avg_trades": 127.3,
      "config": {
        "indicators": [...],
        "risk_strategies": [...],
        "take_profit_pct": 5.2,
        "stop_loss_pct": 2.1,
        "leverage": 10
      }
    }
  ]
}
```

## 🔧 Troubleshooting

### TA-Lib Installation Issues

**Windows:**
```powershell
# Download from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
# Choose matching Python version (e.g., cp312 = Python 3.12)
pip install TA_Lib‑0.4.28‑cp312‑cp312‑win_amd64.whl
```

**Linux:**
```bash
# Install C library first
sudo apt-get install build-essential wget
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install

# Then install Python wrapper
pip install TA-Lib
```

### Kucoin API Issues

If data fetching fails:
1. Check internet connection
2. Verify Kucoin API is accessible: https://api.kucoin.com/api/v1/status
3. Wait 1 minute (rate limiting) and retry
4. Try different trading pair (e.g., ETH/USDT)

### Memory Issues

If you get memory errors:
- Reduce population size (e.g., 5000 instead of 10000)
- Reduce timeframe resolution (e.g., 5m instead of 1m)
- Reduce backtest days (e.g., 3 instead of 7)
- Close other applications

## 💡 Tips for Best Results

### For Faster Execution
- Use 5m or 15m timeframe (less data)
- Reduce backtest days (3-5 days sufficient)
- Smaller population (1000-5000 bots)
- Fewer generations (3-5 generations)

### For Better Trading Strategies
- More generations (10-20)
- More cycles (15-20) for robustness
- Longer backtest periods (14-30 days)
- Larger population (10000-50000)

### For Reproducible Results
- Always use same random seed
- Same parameters
- Same date range
- Note: Data changes over time, so results vary with date

## 📈 Interpreting Results

### Good Bot Characteristics
- **Profit**: >10% average profit
- **Winrate**: >55% winrate
- **Trades**: 50-200 trades (enough data, not overtrading)
- **Survived**: 5+ generations
- **Drawdown**: <20% max drawdown

### Red Flags
- **Too high profit** (>50%): Might be overfitted
- **Too few trades** (<10): Insufficient data
- **Low winrate** (<45%): Strategy not working
- **Survived only 1-2 gens**: Luck, not skill

## 🎓 Next Steps

1. **Analyze Top Bots**: Look at indicator combinations that work
2. **Test Different Pairs**: Try ETH/USDT, SOL/USDT, etc.
3. **Adjust Parameters**: Experiment with leverage, TP/SL ranges
4. **Paper Trade**: Test in real market (carefully!)
5. **Add More Indicators**: Implement additional TA-Lib indicators
6. **Optimize**: Profile code, find bottlenecks

## ⚠️ Important Warnings

1. **This is educational software** - Not financial advice
2. **Backtest results ≠ future performance** - Past success doesn't guarantee future profits
3. **Start small** - Test with minimal capital if going live
4. **Understand the code** - Don't use what you don't understand
5. **Crypto is risky** - Only trade what you can afford to lose

## 🆘 Getting Help

1. Check `README.md` for detailed documentation
2. Review `IMPLEMENTATION_SUMMARY.md` for technical details
3. Read inline code comments
4. Check test files for usage examples
5. Open GitHub issue if you find bugs

## 🎉 You're Ready!

Run `python main.py` and start evolving profitable trading bots!

Good luck! 🚀📈
