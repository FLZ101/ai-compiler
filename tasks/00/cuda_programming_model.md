# PyTorch CUDA vs CUDA C programming model

PyTorch exposes CUDA as a high-level tensor/runtime abstraction. CUDA C gives you
direct control over kernels, memory, and streams. They use the same GPU hardware
and CUDA runtime underneath, but they operate at different levels.

## CUDA C model

- You write two kinds of code: host code on CPU and `__global__` kernels on GPU.
- You launch a kernel explicitly with a grid and block configuration.
- You manually allocate device memory and copy data with `cudaMalloc`,
  `cudaMemcpy`, and `cudaFree`.
- You compute thread indexes yourself: `blockIdx`, `blockDim`, `threadIdx`.
- You synchronize with `cudaDeviceSynchronize`, `cudaStreamSynchronize`, or events.
- `nvcc` compiles CUDA C into host code plus GPU code such as PTX/SASS.

Example mental model:

```cuda
__global__ void add(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
```

## PyTorch CUDA model

- You work with `torch.Tensor` objects that carry a `device`.
- A tensor operation such as `a + b` dispatches to an already-written or
  generated CUDA kernel.
- You normally do not write kernels or manage raw device pointers.
- Memory is allocated and freed by PyTorch's caching allocator.
- Kernels are launched asynchronously, and PyTorch manages the default stream.
- You can still use lower-level CUDA features through `torch.cuda.Stream`,
  `torch.cuda.Event`, and custom CUDA extensions.
- `torch.compile(backend="inductor")` generates Triton/CUDA kernels from
  Python/PyTorch operations.

Example:

```python
a = torch.randn(1000, device="cuda")
b = torch.randn(1000, device="cuda")
c = a + b
```

The `+` operation becomes a GPU kernel launch, but you do not write that kernel.

## Side-by-side

| Concern | CUDA C | PyTorch CUDA |
| --- | --- | --- |
| Kernel code | Explicit `__global__` functions | Implicit dispatch to built-in/generated kernels |
| Launch | `kernel<<<grid, block>>>(...)` | `torch` operator call |
| Memory | Manual `cudaMalloc/cudaMemcpy/cudaFree` | Tensor allocator and automatic transfers |
| Thread indexing | Manual `threadIdx/blockIdx` | Hidden inside kernels |
| Data model | Raw pointers | `torch.Tensor` with device/dtype/shape |
| Synchronization | Explicit CUDA APIs | Often automatic; explicit via `torch.cuda.synchronize` |
| Code generation | `nvcc` at build time | PyTorch runtime, Inductor/Triton at execution |
| Performance tuning | Manual shared memory, occupancy, kernels | Operator fusion, autotuning, compilation |
| Hardware access | Direct | Accessible through streams, events, custom ops |

## How they connect

- PyTorch CUDA ultimately calls the CUDA runtime and driver.
- `torch.cuda.Stream` maps to CUDA streams.
- `torch.cuda.Event` maps to CUDA events.
- Custom PyTorch operators can be written with CUDA C or CUDA extensions when
  you need direct control.
- The thread/SM/block model is the same; PyTorch just hides it behind tensor
  operations.

The main difference is level of abstraction: CUDA C gives direct control over
the GPU, while PyTorch CUDA gives a high-level, differentiable, tensor-based
programming model with CUDA managed for you.
