"""Check OpenCL GPU devices and simulate bot generation workload."""
import pyopencl as cl
import numpy as np
import time

print("=" * 60)
print("OpenCL Platform & Device Information")
print("=" * 60)

platforms = cl.get_platforms()
for i, platform in enumerate(platforms):
    print(f"\nPlatform {i}: {platform.name}")
    print(f"  Version: {platform.version}")
    
    devices = platform.get_devices()
    for j, device in enumerate(devices):
        print(f"\n  Device {j}: {device.name}")
        print(f"    Type: {cl.device_type.to_string(device.type)}")
        print(f"    Max Compute Units: {device.max_compute_units}")
        print(f"    Max Work Group Size: {device.max_work_group_size}")
        print(f"    Max Clock Frequency: {device.max_clock_frequency} MHz")
        print(f"    Global Memory: {device.global_mem_size / (1024**3):.2f} GB")
        print(f"    Local Memory: {device.local_mem_size / 1024:.2f} KB")

print("\n" + "=" * 60)
print("Testing GPU Computation (Intensive Workload)")
print("=" * 60)

# Create context with GPU device
ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)
device = ctx.devices[0]

print(f"\nUsing: {device.name}")
print(f"Max Work Group Size: {device.max_work_group_size}")

# Simple kernel that does real computation
kernel_code = """
__kernel void compute_intensive(
    __global float* output,
    const int iterations
) {
    int gid = get_global_id(0);
    
    // Do intensive computation to actually use GPU
    float result = (float)gid;
    for (int i = 0; i < iterations; i++) {
        result = sqrt(result * 3.14159f + 1.0f);
        result = sin(result) * cos(result);
        result = pow(result, 1.5f);
    }
    
    output[gid] = result;
}
"""

# Compile kernel
program = cl.Program(ctx, kernel_code).build()

# Test with increasing workload
test_sizes = [
    (100000, 100, "Light (100k items, 100 iterations)"),
    (1000000, 500, "Medium (1M items, 500 iterations)"),
    (1000000, 1000, "Heavy (1M items, 1000 iterations)"),
]

for size, iterations, desc in test_sizes:
    print(f"\n{desc}:")
    
    # Allocate buffers
    output = np.zeros(size, dtype=np.float32)
    output_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, output.nbytes)
    
    # Calculate work group size
    work_group_size = min(512, device.max_work_group_size)
    global_size = ((size + work_group_size - 1) // work_group_size) * work_group_size
    
    print(f"  Global size: {global_size}, Work group size: {work_group_size}")
    print(f"  Work groups: {global_size // work_group_size}")
    
    # Execute kernel and measure time
    start = time.perf_counter()
    
    program.compute_intensive(
        queue,
        (global_size,),
        (work_group_size,),
        output_buf,
        np.int32(iterations)
    )
    
    queue.finish()
    elapsed = time.perf_counter() - start
    
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Throughput: {size / elapsed / 1000000:.2f} M items/sec")
    print(f"  → Check GPU usage NOW with Task Manager!")
    
    # Brief pause to see GPU usage spike
    if desc.startswith("Heavy"):
        print("\n  Running 5 iterations to maintain GPU load...")
        for run in range(5):
            start = time.perf_counter()
            program.compute_intensive(queue, (global_size,), (work_group_size,), output_buf, np.int32(iterations))
            queue.finish()
            elapsed = time.perf_counter() - start
            print(f"    Run {run+1}: {elapsed:.3f}s ({size/elapsed/1000000:.2f} M items/sec)")

print("\n" + "=" * 60)
print("Test Complete")
print("=" * 60)
print("\nNOTE: If GPU usage stayed low (<5%), the workload might be too small")
print("or GPU isn't being used. Bot generation (100k items, simple ops) is")
print("extremely fast - GPU finishes in milliseconds, so Task Manager might")
print("not catch the brief spike.")
