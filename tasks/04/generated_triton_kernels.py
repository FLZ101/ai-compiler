import torch
import torch._inductor.config as inductor_config

from pathlib import Path

# Task 04 wants fresh generated-code logs, not a cache hit.
inductor_config.force_disable_caches = True

# Keep all Inductor debug artifacts next to this script.
inductor_config.trace.debug_dir = str(Path(__file__).resolve().parent / "debug")

# Keep the pointwise heuristic deterministic so the printed kernel is easier to
# compare across runs. The actual launch parameters are still chosen by the
# Inductor/Triton heuristic.
inductor_config.triton.autotune_pointwise = False

# Task 04 is about reading the generated Triton GEMM. By default Inductor sends
# this fp32 matmul to cuBLAS. Native matmul makes Inductor emit a Triton GEMM
# with tl.dot, plus the fused bias/ReLU epilogue, which is what this task asks
# us to annotate.
inductor_config.triton.native_matmul = True


@torch.compile
def matmul_bias_relu(x, w, b):
    return torch.relu(x @ w + b)


@torch.compile
def pointwise(x):
    return torch.sin(x) + torch.cos(x)


@torch.compile
def reduction(x):
    return x.sum(dim=1)


@torch.compile
def layernorm(x):
    return torch.nn.functional.layer_norm(x, x.shape[-1:])


x = torch.randn(2048, 2048, device="cuda")
w = torch.randn(2048, 2048, device="cuda")
b = torch.randn(2048, device="cuda")

matmul_bias_relu(x, w, b)
pointwise(x)
reduction(x)
layernorm(x)
