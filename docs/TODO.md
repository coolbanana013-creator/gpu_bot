# TODO LIST - PRIORITY FIXES AND IMPROVEMENTS
**Generated**: November 11, 2025
**Status**: Main.py now runs successfully - Ready for improvements
**Current State**: ✅ All 14 code review tasks complete + VRAM validation working

---

## 🔴 CRITICAL (P0) - MUST FIX BEFORE PRODUCTION

### 1. ✅ VRAM Estimator Integration (FIXED)
**Status**: ✅ COMPLETE
- Fixed method signature mismatch
- Now validates GPU memory before evolution
**Files**: main.py
**Result**: Evolution runs without out-of-memory errors

### 2. ✅ Save Bot Configuration Fix (FIXED)
**Status**: ✅ COMPLETE
- Fixed `risk_strategy_bitmap` → `risk_strategy` 
- Added `risk_param` to saved bot data
**Files**: src/ga/evolver_compact.py
**Result**: Bots save successfully

### 3. ⚠️ Consensus Threshold - 100% Unanimity → 60% Majority
**Status**: ❌ NOT STARTED (User requested to leave at 100% for now)
**Priority**: P0 (but postponed per user request)
**Issue**: Requires ALL indicators to agree for signal
**Impact**: Very few trades (30-40 per bot over 25 days)
**Fix Required**:
- Add `consensus_threshold` field to CompactBotConfig (128→132 bytes)
- Update OpenCL kernel structure
- Modify signal generation logic
- Update bot generator
**Time**: 2 hours
**Files**: 
  - src/bot_generator/compact_generator.py
  - src/gpu_kernels/backtest_with_precomputed.cl
  - src/gpu_kernels/compact_bot_gen.cl
**Note**: User wants this postponed - leave consensus at 100% for now

### 4. ✅ TP/SL Ratio Validation (FIXED)
**Status**: ✅ COMPLETE
**Priority**: P0
**Issue**: Allows TP < SL (e.g., TP=0.01, SL=0.10) → guaranteed losses
**Impact**: Bots can have unprofitable configurations by design
**Fix Applied**: Added validation `if (tp < sl * 0.8) reject` with error code -9975
**Time**: 15 minutes
**Files**: src/gpu_kernels/backtest_with_precomputed.cl (line ~1160)

---

## 🟠 HIGH PRIORITY (P1) - SHOULD FIX SOON

### 5. ✅ Drawdown Circuit Breaker (FIXED)
**Status**: ✅ COMPLETE
**Priority**: P1
**Issue**: Bots continue trading after -99% loss
**Impact**: Unrealistic "death spirals"
**Fix Applied**: Stops trading and closes all positions if drawdown > 30%, exits cycle early
**Time**: 30 minutes
**Files**: src/gpu_kernels/backtest_with_precomputed.cl (line ~1293)

### 6. ✅ Maintenance Margin in Liquidation Formula (FIXED)
**Status**: ✅ COMPLETE
**Priority**: P1
**Issue**: Liquidation formula simplified (missing maintenance margin)
**Impact**: Liquidation prices slightly inaccurate at high leverage
**Fix Applied**: Changed to `(1.0f - 0.005f) / leverage` (0.5% maintenance margin for BTC)
**Time**: 15 minutes
**Files**: src/gpu_kernels/backtest_with_precomputed.cl (lines 742, 750)

### 7. ✅ SL Exit Fee Correction (FIXED)
**Status**: ✅ COMPLETE
**Priority**: P2 (medium-low impact)
**Issue**: SL exits charged taker fee (0.07%), should be maker (0.02%)
**Impact**: ~0.05% cost difference per SL exit
**Fix Applied**: TP/SL now use maker fee, only signal reversals use taker fee, liquidations have 0 exit fee
**Time**: 5 minutes
**Files**: src/gpu_kernels/backtest_with_precomputed.cl (line ~804)

### 8. ✅ Indicator Warmup Period (FIXED)
**Status**: ✅ COMPLETE
**Priority**: P2
**Issue**: Bots can generate signals before indicators fully warmed up
**Impact**: First 1-2% of backtest may have unreliable signals
**Fix Applied**: Calculates warmup period based on max indicator period, skips first N bars per cycle, skips cycle if too short
**Time**: 30 minutes
**Files**: src/gpu_kernels/backtest_with_precomputed.cl (line ~1246)

---

## 🟡 MEDIUM PRIORITY (P2) - IMPROVEMENTS

### 9. Signal Contradiction Handling
**Status**: ❌ NOT STARTED
**Priority**: P2
**Issue**: Mixing counter-trend (RSI, BB) with trend-following (MACD, MA) indicators
**Impact**: Conflicting signals reduce effectiveness
**Fix Required**: Classify indicators by type, separate consensus
**Time**: 2-3 hours
**Files**: Multiple (bot config, kernels, generator)

### 10. Dynamic Slippage Model
**Status**: ❌ NOT STARTED
**Priority**: P3 (low)
**Issue**: Fixed 0.01% slippage regardless of volatility/size
**Impact**: Minor - acceptable simplification
**Fix Required**: Scale slippage with ATR and position size
**Time**: 1 hour
**Files**: src/gpu_kernels/backtest_with_precomputed.cl

### 11. Test Suite Implementation
**Status**: ⚠️ PARTIALLY DONE (stubs created)
**Priority**: P2
**Files Created**:
- tests/test_indicator_accuracy.py (6 tests)
- tests/test_trading_logic.py (10 tests)
- tests/test_system_smoke.py (4 tests)
**Remaining Work**: Implement test bodies (currently TODOs)
**Time**: 4-6 hours

### 12. ✅ Kernel Warning Suppression (FIXED)
**Status**: ✅ COMPLETE
**Priority**: P3
**Issue**: RepeatedKernelRetrieval warning in compact_generator.py
**Impact**: Performance degradation (kernel recompiled)
**Fix Applied**: Cached kernel instance as `self.generate_bots_kernel` during compilation, reused in both methods
**Time**: 15 minutes
**Files**: src/bot_generator/compact_generator.py (lines 215, 298, 372)

---

## 🟢 LOW PRIORITY (P3) - POLISH

### 13. Duplicate Bot Save Logs
**Status**: ❌ NOT STARTED
**Priority**: P3
**Issue**: Each bot saved multiple times (duplicate log messages)
**Impact**: Log spam only
**Fix Required**: Remove duplicate save calls
**Time**: 5 minutes
**Files**: src/ga/evolver_compact.py

### 14. State Persistence Integration
**Status**: ✅ MODULE CREATED, ❌ NOT INTEGRATED
**Priority**: P3
**Files**: src/persistence/checkpoint.py
**Remaining Work**: Add checkpoint save/load calls to evolver
**Time**: 30 minutes
**Benefit**: Resume evolution after interruption

### 15. Performance Profiling Output
**Status**: ⚠️ INCOMPLETE
**Priority**: P3
**Issue**: "Phase Breakdown" and "Potential Bottlenecks" show numbers only (no labels)
**Impact**: Can't interpret performance data
**Fix Required**: Add labels to profiler output
**Time**: 15 minutes
**Files**: Check evolver_compact.py profiler

---

## 📊 TESTING CHECKLIST

### Unit Tests (Need Implementation)
- [ ] test_indicator_accuracy.py - Implement 6 TA-Lib comparison tests
- [ ] test_trading_logic.py - Implement 10 trading tests
- [ ] test_system_smoke.py - Implement metrics collection

### Integration Tests (Need Creation)
- [ ] Test consensus threshold (after implementing #3)
- [ ] Test TP/SL validation (after implementing #4)
- [ ] Test drawdown circuit breaker (after implementing #5)
- [ ] Test liquidation at 1x, 10x, 125x leverage

### System Tests (Recommended)
- [ ] Run 10,000 bots × 50 generations
- [ ] Collect trade frequency distribution
- [ ] Analyze fitness score distribution
- [ ] Monitor VRAM usage throughout evolution
- [ ] Verify no crashes over extended run

---

## 📈 OBSERVED ISSUES FROM LATEST RUN

### Trade Frequency Analysis
**Observation**: 30-40 trades per bot over 25 days (7-day cycles × 3)
**Expected**: 30-100 trades per 100-day period
**Status**: ⚠️ LOW (likely due to 100% consensus requirement)
**Action**: Monitor after consensus threshold fix (#3)

### Survival Rate
**Gen 0**: 202/10,000 survivors (2.0%)
**Gen 1**: 347/10,000 survivors (3.5%)
**Status**: ✓ ACCEPTABLE (strict filter: ALL cycles profitable)
**Note**: Survival increased in Gen 1 (good evolution signal)

### Performance
**Total Time**: 111.9 seconds for 2 generations
**Per Generation**: ~55 seconds average
**Bottleneck**: Backtesting (1.1-1.2s per chunk)
**Status**: ✓ GOOD (10k bots × 3 cycles = 30k workloads in ~1.2s)

---

## 🎯 RECOMMENDED IMPLEMENTATION ORDER

### Week 1 (Critical Fixes)
**Day 1-2**: 
1. Implement TP/SL ratio validation (#4) - 15 min
2. Test with 1000 bots to verify rejection
3. Implement drawdown circuit breaker (#5) - 30 min
4. Test to confirm stops at -30%

**Day 3-4**:
5. Implement maintenance margin fix (#6) - 15 min
6. Implement SL exit fee fix (#7) - 5 min
7. Run full evolution test (10k bots × 10 generations)
8. Analyze results and validate fixes

**Day 5**:
9. Decide on consensus threshold implementation (#3)
10. If proceeding: implement consensus threshold (2 hours)
11. Test trade frequency improvement

### Week 2 (Testing & Polish)
**Day 1-3**: Implement test suites (#11) - 4-6 hours
**Day 4**: Fix kernel warning (#12) - 15 min
**Day 5**: Clean up duplicate saves (#13), profiler output (#15)

### Week 3 (Advanced Features)
- Indicator warmup (#8)
- Signal contradiction handling (#9)
- State persistence integration (#14)
- Dynamic slippage (#10)

---

## ✅ COMPLETED ITEMS (From Previous Session)

1. ✅ Input validation with robust error handling
2. ✅ Risk-aware multi-objective fitness function
3. ✅ Thread-safety locks (memory, buffer, queue)
4. ✅ RNG verification (xorshift32)
5. ✅ Path sanitization (directory traversal prevention)
6. ✅ Kernel performance verification
7. ✅ Batch transfer verification
8. ✅ File logging with rotation (10MB, 5 backups)
9. ✅ State persistence module created
10. ✅ Overflow protection (counter clamping)
11. ✅ VRAM estimator query real device memory
12. ✅ Test infrastructure created (3 files)
13. ✅ Comprehensive flaw analysis documented
14. ✅ Implementation plan documented

---

## 📝 NOTES

### User Preferences
- ❗ **Consensus threshold**: Leave at 100% for now (don't modify bot config)
- ❗ **Fix everything else**: Implement all other fixes
- ✅ **VRAM fix**: Completed
- ✅ **Main.py runs successfully**: Confirmed

### Key Documents
- `FLAW_ANALYSIS.md` - Detailed flaw analysis
- `FIX_IMPLEMENTATION_PLAN.md` - Step-by-step implementation guide
- `COMPLETION_SUMMARY.md` - Full session report
- `QUICK_REFERENCE.md` - Quick start guide

### Estimated Total Time for Priority Fixes
- **P0 Critical** (excluding consensus): 15 min
- **P1 High**: 50 min
- **P2 Medium**: 35 min
- **Total**: ~1.5 hours (without consensus threshold)

### Next Session Goals
1. Fix TP/SL ratio validation (15 min)
2. Add drawdown circuit breaker (30 min)
3. Fix maintenance margin formula (15 min)
4. Run validation tests

---

**END OF TODO LIST**
