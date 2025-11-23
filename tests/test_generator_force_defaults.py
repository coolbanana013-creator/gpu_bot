import pyopencl as cl
import numpy as np
from src.bot_generator.compact_generator import CompactBotGenerator
from src.indicators.gpu_default_params import get_gpu_default_params


def test_generator_force_defaults():
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    gen = CompactBotGenerator(gpu_context=ctx, gpu_queue=queue, population_size=5, force_gpu_default_params=True)
    bots = gen.generate_population()
    for bot in bots:
        for i in range(bot.num_indicators):
            idx = int(bot.indicator_indices[i])
            p0, p1, p2 = get_gpu_default_params(idx)
            # Use near-equality for floats with tolerance; bot may have ints or floats
            assert np.isclose(bot.indicator_params[i][0], p0, rtol=1e-9, atol=1e-9), f"param0 mismatch for idx {idx}: {bot.indicator_params[i][0]} vs {p0}"
            assert np.isclose(bot.indicator_params[i][1], p1, rtol=1e-9, atol=1e-9), f"param1 mismatch for idx {idx}: {bot.indicator_params[i][1]} vs {p1}"
            assert np.isclose(bot.indicator_params[i][2], p2, rtol=1e-9, atol=1e-9), f"param2 mismatch for idx {idx}: {bot.indicator_params[i][2]} vs {p2}"
