# Eager vs TorchInductor

Eager and TorchInductor both ultimately run PyTorch operations on the same
hardware, but they execute those operations very differently.

## Eager execution

- Each PyTorch operation runs immediately when it is called.
- PyTorch dispatches to a prewritten kernel, such as an ATen, cuBLAS, or cuDNN
  kernel.
- There is no optimization across the whole expression.
- Intermediate tensors are usually created for each operator.
- Python dispatch overhead happens for every operation.
- Dynamic Python behavior and changing shapes are easy to support.

For Task 0:

```python
torch.relu(x @ w + b)
```

Eager roughly does:

```text
x @ w     -> matmul kernel, allocates temporary matrix
+ b       -> bias-add kernel, allocates another temporary matrix
relu(...) -> ReLU kernel, allocates final output
```

## TorchInductor execution

- `torch.compile` first captures the computation into an FX graph.
- Inductor lowers that graph into an internal IR.
- It fuses operators, plans buffers, and schedules the work.
- It generates and compiles optimized Triton/CUDA kernels.
- The runtime wrapper launches fewer kernels and reuses memory.
- The compiled program is cached and reused on later calls.

The same computation can become:

```text
fused matmul + bias + relu kernel -> final output
```

That removes intermediate buffers, kernel launches, and much of the per-operator
Python/dispatch overhead.

## Comparison

| Aspect | Eager | TorchInductor |
| --- | --- | --- |
| Execution | Immediate per operator | Compiled whole program |
| Kernel source | Prewritten ATen/cuBLAS/cuDNN kernels | Generated Triton/CUDA kernels |
| Cross-op optimization | None | Fusion and scheduling |
| Intermediate buffers | Often created | Reused or eliminated |
| Python overhead | Per operation | Minimal at runtime |
| Shape flexibility | High | Guards determine reuse/recompile |
| First-call cost | Low | Higher because compilation |
| Steady-state speed | Usually slower | Usually faster |

## Key difference

Eager asks: what kernel should this single op use right now?

TorchInductor asks: how should this whole computation be lowered into the fewest,
most efficient kernels?

Eager is immediate and flexible. Inductor adds a compilation step to optimize
the entire graph before running it.
