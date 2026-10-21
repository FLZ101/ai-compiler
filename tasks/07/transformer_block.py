import os

import torch
import torch.nn as nn


class Block(nn.Module):
    """Small transformer block used to collect realistic kernel names."""

    def __init__(self, dim=256, heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        return x


def warmup(fn, x, calls=3):
    for _ in range(calls):
        fn(x)
    torch.cuda.synchronize()


def benchmark(fn, x, iters=20):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn(x)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iters


torch.manual_seed(0)

model = Block(dim=256, heads=8).cuda().eval()
x = torch.randn(8, 128, 256, device="cuda")

variants = {
    "eager": model,
    "compiled": torch.compile(model),
    "reduce-overhead": torch.compile(model, mode="reduce-overhead"),
}

with torch.inference_mode():
    eager_out = model(x)
    torch.cuda.synchronize()

    for name, fn in variants.items():
        warmup(fn, x)
        out = fn(x)
        torch.cuda.synchronize()

        max_diff = (out - eager_out).abs().max().item()
        elapsed_ms = benchmark(fn, x)
        print(f"{name:18s} max_diff={max_diff:.3e}  {elapsed_ms:8.3f} ms/iter")

    if os.environ.get("PROFILE") == "1":
        from torch.profiler import ProfilerActivity, profile

        compiled_fn = variants["compiled"]
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=False,
        ) as prof:
            for _ in range(10):
                compiled_fn(x)
            torch.cuda.synchronize()

        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
