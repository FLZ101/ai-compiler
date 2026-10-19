import time

import torch
import torch._inductor.config as inductor_config

from pathlib import Path

# Force fresh compilation for every mode. This keeps the benchmark from being
# dominated by cache hits and makes the TORCH_LOGS output complete.
inductor_config.force_disable_caches = True

# If TORCH_COMPILE_DEBUG=1 is used, keep the generated artifacts next to this
# script rather than in the process working directory.
inductor_config.trace.debug_dir = str(Path(__file__).resolve().parent / "debug")


def fn(x, w, b):
    return torch.relu(x @ w + b)


x = torch.randn(2048, 2048, device="cuda")
w = torch.randn(2048, 2048, device="cuda")
b = torch.randn(2048, device="cuda")

variants = {
    "default": torch.compile(fn),
    "reduce-overhead": torch.compile(fn, mode="reduce-overhead"),
    "max-autotune": torch.compile(fn, mode="max-autotune"),
    "max-autotune-no-cudagraphs": torch.compile(fn, mode="max-autotune-no-cudagraphs"),
}

# First calls trigger compilation. The measured wall time includes Dynamo,
# AOTAutograd, Inductor, and any autotuning for that compile mode.
for name, compiled in variants.items():
    compile_start = time.perf_counter()
    for _ in range(3):
        compiled(x, w, b)
    torch.cuda.synchronize()
    compile_seconds = time.perf_counter() - compile_start

    # Use CUDA events for the runtime measurement so Python launch overhead is
    # not included. CUDA graphs, when enabled, are recorded and replayed during
    # this loop.
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(30):
        compiled(x, w, b)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end)
    print(f"{name:28s} compile={compile_seconds:6.2f}s  " f"{elapsed_ms / 30:8.3f} ms/iter")
