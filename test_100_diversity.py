"""Test 100% bot diversity guarantee."""
import pyopencl as cl
from src.bot_generator.compact_generator import CompactBotGenerator
from src.ga.evolver_compact import GeneticAlgorithmEvolver

# Initialize GPU
platforms = cl.get_platforms()
devices = platforms[0].get_devices(device_type=cl.device_type.GPU)
ctx = cl.Context(devices=[devices[0]])
queue = cl.CommandQueue(ctx)

# Create generator
generator = CompactBotGenerator(
    gpu_context=ctx,
    gpu_queue=queue,
    population_size=1000,  # Test with 1000 bots
    random_seed=42
)

print("="*70)
print("TESTING 100% BOT DIVERSITY GUARANTEE")
print("="*70)

# Test 1: Initial population
print("\n[TEST 1] Initial Population Diversity")
print("-" * 70)
population = generator.generate_population()
combinations = [frozenset(bot.indicator_indices[:bot.num_indicators]) for bot in population]
unique_combos = len(set(combinations))
diversity_pct = (unique_combos / len(population)) * 100

print(f"Population size: {len(population)}")
print(f"Unique combinations: {unique_combos}")
print(f"Diversity: {diversity_pct:.2f}%")

if diversity_pct == 100.0:
    print("✅ PASS: 100% unique combinations in initial population!")
else:
    print(f"⚠️  Initial diversity: {diversity_pct:.2f}% (will be fixed by evolver)")

# Test 2: Test evolver's uniqueness enforcement
print("\n[TEST 2] Evolver's Uniqueness Enforcement")
print("-" * 70)

# Create a dummy backtester (not used, just for evolver init)
class DummyBacktester:
    def backtest_bots(self, *args, **kwargs):
        return []

evolver = GeneticAlgorithmEvolver(
    bot_generator=generator,
    backtester=DummyBacktester(),
    gpu_context=ctx,
    gpu_queue=queue,
    mutation_rate=0.15,
    elite_pct=0.10,
    pair="BTC/USDT:USDT",
    timeframe="15m"
)

# Initialize population with uniqueness enforcement
evolver_population = evolver.initialize_population()
evolver_combinations = [frozenset(bot.indicator_indices[:bot.num_indicators]) 
                        for bot in evolver_population]
evolver_unique = len(set(evolver_combinations))
evolver_diversity = (evolver_unique / len(evolver_population)) * 100

print(f"Population size: {len(evolver_population)}")
print(f"Unique combinations: {evolver_unique}")
print(f"Diversity: {evolver_diversity:.2f}%")

if evolver_diversity == 100.0:
    print("✅ PASS: 100% unique combinations after evolver enforcement!")
else:
    print(f"❌ FAIL: Only {evolver_diversity:.2f}% unique")

# Test 3: Test generate_unique_bot method
print("\n[TEST 3] Generate Unique Bot (No Retries)")
print("-" * 70)

# Pre-populate with some bots
test_bots = []
for i in range(50):
    bot = evolver.generate_unique_bot(bot_id=1000 + i)
    test_bots.append(bot)

test_combos = [frozenset(bot.indicator_indices[:bot.num_indicators]) for bot in test_bots]
test_unique = len(set(test_combos))
test_diversity = (test_unique / len(test_bots)) * 100

print(f"Generated: {len(test_bots)} bots")
print(f"Unique combinations: {test_unique}")
print(f"Diversity: {test_diversity:.2f}%")

if test_diversity == 100.0:
    print("✅ PASS: 100% unique - no duplicates generated!")
else:
    print(f"❌ FAIL: Only {test_diversity:.2f}% unique")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
if evolver_diversity == 100.0 and test_diversity == 100.0:
    print("✅ ALL TESTS PASSED - 100% diversity guaranteed!")
else:
    print("⚠️  Some tests did not achieve 100% diversity")
print("="*70)
