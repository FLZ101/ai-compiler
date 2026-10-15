import torch
import torch._inductor.config as inductor_config

from pathlib import Path

# Regenerate Inductor artifacts on every run instead of reusing cached code.
inductor_config.force_disable_caches = True

# Keep debug output next to this script rather than the process working dir.
inductor_config.trace.debug_dir = str(Path(__file__).resolve().parent / "debug")


@torch.compile
def fn(x, w, b):
    return torch.relu(x @ w + b)


x = torch.randn(2048, 2048, device="cuda")
w = torch.randn(2048, 2048, device="cuda")
b = torch.randn(2048, device="cuda")

fn(x, w, b)
