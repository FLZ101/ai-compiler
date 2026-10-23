from pathlib import Path

import torch
import torch._inductor.config as inductor_config

# Keep each log run from reusing an older compiled artifact.
inductor_config.force_disable_caches = True

# Put TORCH_COMPILE_DEBUG artifact trees next to this script, as with Task 03/04.
inductor_config.trace.debug_dir = str(Path(__file__).resolve().parent / "debug")

# Force a Triton GEMM instead of an external cuBLAS mm call so the trace ends at
# Inductor code generation rather than at an opaque vendor dispatch.
inductor_config.triton.native_matmul = True


@torch.compile
def matmul_bias_relu(x, w, b):
    # This single Python expression is enough to exercise:
    #   1. Dynamo: trace x @ w + b -> relu into an FX graph.
    #   2. AOTAutograd: decompose/normalize the forward graph.
    #   3. Inductor: schedule the matmul and pointwise epilogue.
    #   4. Triton codegen: emit a reduction kernel with tl.dot and fused epilogue.
    return torch.relu(x @ w + b)


torch.manual_seed(0)

x = torch.randn(256, 256, device="cuda")
w = torch.randn(256, 256, device="cuda")
b = torch.randn(256, device="cuda")

# The first call triggers tracing and compilation. Run it a few times so a
# cache hit path is visible when caches are not forcibly disabled.
for _ in range(3):
    out = matmul_bias_relu(x, w, b)
torch.cuda.synchronize()

print(out.shape)
print(float(out.sum()))
