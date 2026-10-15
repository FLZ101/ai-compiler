import torch
import torch._inductor.config as inductor_config
from pathlib import Path

# Generate fresh artifacts and put them next to this script instead of reusing
# the default torch_compile_debug directory.
inductor_config.force_disable_caches = True
inductor_config.trace.debug_dir = str(Path(__file__).resolve().parent / "fusion_debug")


@torch.compile
def fn(x):
    # Stage 1: elementwise add followed by a reduction along the columns.
    y = x + 1
    s = y.sum(dim=1)

    # Stage 2: separate pointwise multiply after the reduction. Before fusion
    # these stages are two scheduler nodes; afterward they become one
    # FusedSchedulerNode.
    return s * 2


x = torch.randn(64, 32, device="cuda")
fn(x)
