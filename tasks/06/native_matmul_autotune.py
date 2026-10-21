import time

import torch
import torch._inductor.config as inductor_config

# Force a fresh compile so the mode comparison is not masked by a cache hit.
inductor_config.force_disable_caches = True

# By default Inductor sends fp32 matmul to cuBLAS. Native matmul makes Inductor
# emit a Triton GEMM whose block/warp/stage choices can be autotuned.
inductor_config.triton.native_matmul = True


def fn(a, b):
    return a @ b


torch.manual_seed(0)

m = 3072
n = 2048
k = 3072

a = torch.randn(m, k, device="cuda")
b = torch.randn(k, n, device="cuda")

variants = {
    "default": torch.compile(fn),
    "reduce-overhead": torch.compile(fn, mode="reduce-overhead"),
    "max-autotune": torch.compile(fn, mode="max-autotune"),
    "max-autotune-no-cudagraphs": torch.compile(fn, mode="max-autotune-no-cudagraphs"),
}

for name, compiled in variants.items():
    compile_start = time.perf_counter()
    for _ in range(3):
        compiled(a, b)
    torch.cuda.synchronize()
    compile_seconds = time.perf_counter() - compile_start

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(30):
        compiled(a, b)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end)
    print(f"{name:28s} compile={compile_seconds:6.2f}s " f"{elapsed_ms / 30:8.3f} ms/iter")
