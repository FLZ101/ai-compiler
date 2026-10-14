import torch


def eager_fn(x, w, b):
    return torch.relu(x @ w + b)


@torch.compile(backend="inductor")
def compiled_fn(x, w, b):
    return torch.relu(x @ w + b)


x = torch.randn(2048, 2048, device="cuda")
w = torch.randn(2048, 2048, device="cuda")
b = torch.randn(2048, device="cuda")

# Compile and warm up
for _ in range(5):
    out = compiled_fn(x, w, b)

torch.cuda.synchronize()


# Compare eager and compiled execution
def time_fn(fn, *args, iters=20):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


eager_ms = time_fn(eager_fn, x, w, b)
compiled_ms = time_fn(compiled_fn, x, w, b)

print(f"eager:     {eager_ms:.3f} ms/iter")
print(f"compiled:  {compiled_ms:.3f} ms/iter")
print(f"speedup:   {eager_ms / compiled_ms:.2f}x")
