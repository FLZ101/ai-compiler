import time

import torch
import torch._inductor.config as inductor_config

# Keep this example a fresh compilation, like the main compile-modes script.
inductor_config.force_disable_caches = True


def fn(x, matrices):
    # Many small GEMMs cannot be fused into one big GEMM. Each iteration is a
    # separate cuBLAS/Triton launch, so the graph becomes launch-bound rather
    # than dominated by a single large matmul.
    for w in matrices:
        x = torch.relu(x @ w)
    return x


torch.manual_seed(0)

size = 64
num_steps = 64

x = torch.randn(size, size, device="cuda")
matrices = [torch.randn(size, size, device="cuda") for _ in range(num_steps)]

variants = {
    "default": torch.compile(fn),
    "reduce-overhead": torch.compile(fn, mode="reduce-overhead"),
    "max-autotune": torch.compile(fn, mode="max-autotune"),
    "max-autotune-no-cudagraphs": torch.compile(fn, mode="max-autotune-no-cudagraphs"),
}

for name, compiled in variants.items():
    compile_start = time.perf_counter()
    for _ in range(3):
        compiled(x, matrices)
    torch.cuda.synchronize()
    compile_seconds = time.perf_counter() - compile_start

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(50):
        compiled(x, matrices)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end)
    print(f"{name:28s} compile={compile_seconds:5.2f}s " f"{elapsed_ms / 50:8.3f} ms/iter")
