"""Test script to verify bot diversity in generation."""
import pyopencl as cl
import numpy as np
from src.bot_generator.compact_generator import CompactBotGenerator

# Initialize GPU
platforms = cl.get_platforms()
devices = platforms[0].get_devices(device_type=cl.device_type.GPU)
ctx = cl.Context(devices=[devices[0]])
queue = cl.CommandQueue(ctx)

# Create generator
generator = CompactBotGenerator(
    gpu_context=ctx,
    gpu_queue=queue,
    population_size=10,
    random_seed=42
)

print("Testing bot diversity...")
print("=" * 60)

# Test 1: Generate initial population
print("\nTest 1: Initial population diversity")
population = generator.generate_population()
print(f"Generated {len(population)} bots")
bot_ids = [bot.bot_id for bot in population]
print(f"Bot IDs: {bot_ids}")
indicators = [tuple(bot.indicator_indices[:bot.num_indicators]) for bot in population]
unique_combos = len(set(indicators))
print(f"Unique indicator combinations: {unique_combos}/{len(population)}")

# Test 2: Generate individual bots
print("\nTest 2: Individual bot generation (10 bots)")
individual_bots = []
for i in range(10):
    bot = generator.generate_single_bot(bot_id=1000 + i)
    individual_bots.append(bot)

bot_configs = []
for bot in individual_bots:
    config = {
        'bot_id': bot.bot_id,
        'num_indicators': bot.num_indicators,
        'indicators': tuple(bot.indicator_indices[:bot.num_indicators]),
        'leverage': bot.leverage,
        'tp': round(bot.tp_multiplier, 4),
        'sl': round(bot.sl_multiplier, 4)
    }
    bot_configs.append(config)
    print(f"  Bot {bot.bot_id}: {bot.num_indicators} indicators, leverage={bot.leverage}, "
          f"TP={config['tp']}, SL={config['sl']}")

# Check for duplicates
unique_indicators = len(set(c['indicators'] for c in bot_configs))
unique_leverages = len(set(c['leverage'] for c in bot_configs))
unique_tps = len(set(c['tp'] for c in bot_configs))
unique_sls = len(set(c['sl'] for c in bot_configs))

print(f"\nDiversity metrics:")
print(f"  Unique indicator combinations: {unique_indicators}/10")
print(f"  Unique leverages: {unique_leverages}/10")
print(f"  Unique TPs: {unique_tps}/10")
print(f"  Unique SLs: {unique_sls}/10")

if unique_indicators >= 8:
    print("\n✅ PASS: Bots are diverse!")
else:
    print("\n❌ FAIL: Bots are too similar!")

print("=" * 60)
