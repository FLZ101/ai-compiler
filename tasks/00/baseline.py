import torch


def eager_fn(x, w, b):
    """Run the matmul-bias-ReLU computation in eager PyTorch."""
    return torch.relu(x @ w + b)


@torch.compile(backend="inductor")
def compiled_fn(x, w, b):
    """Run the same computation through TorchInductor."""
    return torch.relu(x @ w + b)


# All tensors must be on the same device. `device="cuda"` moves the data to the
# current CUDA GPU, so every PyTorch operation below runs as a CUDA kernel.
x = torch.randn(2048, 2048, device="cuda")
w = torch.randn(2048, 2048, device="cuda")
b = torch.randn(2048, device="cuda")

# Compile and warm up the Inductor path so benchmarking excludes compilation.
for _ in range(5):
    out = compiled_fn(x, w, b)

# Wait until every queued CUDA kernel has finished before starting the timer.
torch.cuda.synchronize()


def time_fn(fn, *args, iters=20):
    """Measure average GPU time per call using CUDA events."""
    # CUDA events record positions in the GPU's work queue. `enable_timing=True`
    # lets us ask the driver for elapsed GPU time between two events.
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # `record()` inserts the event into the current CUDA stream. Kernel launches
    # are asynchronous, so the CPU keeps enqueueing work without waiting.
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()

    # Block the CPU until the GPU reaches the end event. Without this, we might
    # query the timer before the kernels have actually completed.
    torch.cuda.synchronize()

    # The driver reports milliseconds between the two event positions on the GPU.
    return start.elapsed_time(end) / iters


eager_ms = time_fn(eager_fn, x, w, b)
compiled_ms = time_fn(compiled_fn, x, w, b)

print(f"eager:     {eager_ms:.3f} ms/iter")
print(f"compiled:  {compiled_ms:.3f} ms/iter")
print(f"speedup:   {eager_ms / compiled_ms:.2f}x")
