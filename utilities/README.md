# Utilities Directory

This directory contains debugging, testing, and analysis scripts used during development.

## Test Scripts

### GPU Testing
- **test_debug_kernel.py** - Debug GPU kernel execution with full output, runs 1000 bots × 1 generation × 5 cycles
- **test_gpu_execution.py** - Basic GPU execution tests
- **test_gpu_kernel_port.py** - GPU kernel porting validation tests  
- **test_kernel_fixes.py** - Kernel bug fix validation

### Trading System Tests
- **test_kucoin_fixes.py** - KuCoin API integration tests

## Analysis Scripts

- **analyze_bot5_logs.py** - Analyze trade logs for specific bot debugging
- **trace_bot5.py** - Detailed execution trace for bot #5
- **find_tradable_bot.py** - Search for bots meeting profitability criteria
- **check_gen_results.py** - Validate generation results
- **check_braces.py** - Code syntax validation utility

## Test Input Files

- **test_inputs.txt** - Standard test configuration inputs
- **test_quick.txt** - Quick test configuration
- **test_kernel_validation.txt** through **test_kernel_validation4.txt** - Kernel validation test configs

## Usage

These scripts are primarily for development and debugging. For normal operation, use `main.py` in the root directory.

### Example: Debug GPU Kernel Execution
```bash
cd utilities
python test_debug_kernel.py
```

### Example: Analyze Bot Logs
```bash
cd utilities  
python analyze_bot5_logs.py
```

## Environment Variables

Many test scripts support environment variables:
- `ENABLE_TRADE_LOGS=1` - Enable detailed trade logging
- `TRADE_LOG_MAX=20000` - Set maximum trade logs per chunk
- `DEBUG_DISABLE_TRADE_LOGS=1` - Force disable trade logging for debugging

See individual script headers for specific configuration options.
