import pyopencl as cl
from pathlib import Path
import traceback

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
print('Context created')

# Compile precompute
try:
    pre = Path('src/gpu_kernels/precompute_all_indicators.cl').read_text()
    p = cl.Program(ctx, pre).build()
    print('Precompute built')
except Exception:
    traceback.print_exc()
    print('Precompute build failed')

# Compile backtest and print build log on failure
try:
    back = Path('src/gpu_kernels/backtest_with_precomputed.cl').read_text()
    p2 = cl.Program(ctx, back).build()
    print('Backtest built')
except Exception:
    import pyopencl as _cl
    traceback.print_exc()
    try:
        prog = _cl.Program(ctx, back)
        try:
            prog.build()
        except Exception:
            print('--- BACKTEST BUILD LOG ---')
            for d in ctx.devices:
                print(f'--- Build log for device: {d.name} ---')
                print(prog.get_build_info(d, _cl.program_build_info.LOG))
    except Exception:
        pass
    print('Backtest build failed')

# Compile aggregate
try:
    agg = Path('src/gpu_kernels/aggregate_results.cl').read_text()
    p3 = cl.Program(ctx, agg).build()
    print('Aggregate built')
except Exception:
    traceback.print_exc()
    print('Aggregate build failed')
