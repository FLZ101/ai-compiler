# Graph Optimization Plan

This plan studies optimization at the graph level, not inside one CUDA kernel.
It covers both:

- **manual graph optimization:** editing FX graphs or writing fused kernels by
  hand,
- **compiler-performed optimization:** what Dynamo, AOTAutograd, and Inductor
  do automatically.

The central example is operator fusion, but the plan also covers the other
graph transformations needed to understand a production `torch.compile`
pipeline.

## Why graph optimization matters

A model is first expressed as many small PyTorch operations:

```python
def model(x, w, b):
    y = x @ w
    y = y + b
    y = torch.relu(y)
    return y
```

Eager execution launches one kernel or library call per operation. Graph
optimization removes intermediate tensors, kernel launches, and memory traffic
by rewriting the computation before code generation.

```text
eager graph
  -> normalize/decompose
  -> simplify
  -> fuse
  -> plan memory
  -> schedule
  -> generate kernels
```

## Optimization categories

| Category | Examples | Usually performed by |
| --- | --- | --- |
| Algebraic simplification | `x + 0 -> x`, `x * 1 -> x` | AOTAutograd/Inductor |
| Dead-code elimination | remove unused intermediate results | FX passes |
| Common-subexpression elimination | reuse repeated computation | AOTAutograd/Inductor |
| Constant folding | precompute fixed weights/biases | Inductor freezing |
| Decomposition | split composite ops into primitives | AOTAutograd |
| Layout optimization | propagate transposes into GEMMs | Inductor |
| Operator fusion | merge pointwise/reduction epilogues | Inductor scheduler |
| Memory planning | reuse buffers, remove copies | Inductor |
| Functionalization | remove in-place mutation | AOTAutograd |
| Recomputation | discard/recompute activations | AOTAutograd/Inductor |
| Guard specialization | compile only for observed shapes/dtypes | Dynamo |

## Operator fusion in detail

### Fusion types

- **Vertical pointwise fusion:** merge `add -> mul -> relu` into one kernel.
- **Reduction epilogue fusion:** merge `matmul -> bias -> relu` so bias/relu run
  in the GEMM epilogue.
- **Reduction fusion:** merge `x * w` into a following `sum`.
- **Horizontal fusion:** merge independent pointwise nodes with compatible
  shapes and inputs into one kernel.
- **Normalization fusion:** merge mean/variance computation with affine scale
  and residual add.

### Manual pointwise fusion

Before fusion, a three-op chain is:

```python
t = x + b
u = t * scale
y = torch.relu(u)
```

Manual fused implementation in Triton:

```python
@triton.jit
def fused_add_mul_relu(x_ptr, b_ptr, scale_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    scale = tl.load(scale_ptr + offs, mask=mask)
    y = tl.relu((x + b) * scale)
    tl.store(y_ptr + offs, y, mask=mask)
```

Manual fused implementation in CUDA C++:

```cuda
__global__ void fused_add_mul_relu(const float* x,
                                   const float* b,
                                   const float* scale,
                                   float* y,
                                   int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = fmaxf((x[i] + b[i]) * scale[i], 0.0f);
    }
}
```

The fused version reads each tensor once, computes in registers, and writes
once. The unfused eager version does three global-memory round trips.

### Torch-performed pointwise fusion

Compile the same function:

```python
@torch.compile
def fused(x, b, scale):
    return torch.relu((x + b) * scale)
```

Inspect the generated kernel:

```bash
TORCH_LOGS=output_code make run-log SCRIPT=graph_optimization_pointwise.py
```

Expected result:

- Inductor emits one pointwise kernel, often named
  `triton_poi_fused_add_mul_relu_0`.

### Manual vs compiler epilogue fusion

For `matmul + bias + relu`:

```python
def matmul_bias_relu(x, w, b):
    return torch.relu(x @ w + b)
```

Manual options:

- write a CUDA/Triton GEMM and apply bias+ReLU in the accumulator epilogue,
- call cuBLAS and then run a fused bias+ReLU kernel.

Compiler options:

- default Inductor often keeps `extern_kernels.mm` and emits
  `triton_poi_fused_add_relu_0`;
- with native matmul enabled, Inductor can fuse bias+ReLU directly into the
  Triton GEMM epilogue.

Record:

- number of kernels,
- kernel names,
- whether the intermediate GEMM output is written to global memory,
- end-to-end time.

## Non-fusion graph optimizations

### Dead-code elimination and CSE

Use FX to observe and remove unused nodes:

```python
import torch
from torch import fx

def fn(x):
    unused = x * 2
    return x + 1

gm = fx.symbolic_trace(fn)
print(gm.graph)
```

Manual DCE:

- find nodes with zero users,
- remove them unless they have side effects.

CSE:

- identify identical operations on identical inputs,
- replace later uses with the earlier result.

Inductor performs similar cleanup before scheduling.

### Constant folding

Manual constant folding:

```python
w = torch.randn(128, 128)
b = torch.randn(128)

def fn(x):
    return torch.relu(x @ w + b)
```

Compiler equivalent:

```python
inductor_config.freezing = True
compiled = torch.compile(fn)
```

Expected result:

- weights are captured as constants,
- Inductor may fold or specialize them into the generated code,
- recompilation occurs if the constants change.

### Layout optimization

Transposes are layout metadata, not always a separate kernel:

```python
def fn(x, w):
    return x.transpose(0, 1) @ w
```

Optimization:

- propagate the transpose into the GEMM instead of materializing a transposed
  copy,
- choose `torch.contiguous_format` or `torch.channels_last` for convolutions,
- use `torch._subclasses.fake_tensor` and FX to inspect layout choices.

Observe with:

```bash
TORCH_LOGS=+inductor make run-log SCRIPT=graph_optimization_layout.py
```

### Memory planning and buffer reuse

Inductor can reuse buffers whose lifetimes do not overlap.

Inspect:

```python
import torch._inductor.config as inductor_config

inductor_config.optimize_mem = True
```

Observe:

- generated `buf0`, `buf1` allocations,
- which buffers are reused,
- whether intermediates are allocated per call or pooled.

### Functionalization and in-place mutation

PyTorch user code often mutates tensors:

```python
x.add_(1)
x.relu_()
```

Graph optimizations prefer functional graphs:

- AOTAutograd functionalizes in-place ops,
- the compiler can then reorder and fuse them safely,
- selected output/input buffers may still be updated in-place at the end.

Observe with:

```bash
TORCH_LOGS=post_grad_graphs make run-log SCRIPT=graph_optimization_mutation.py
```

### Recomputation and activation checkpointing

Recomputation deliberately removes some saved activations and recomputes them
in the backward pass:

```python
torch.utils.checkpoint.checkpoint(layer, x)
```

Tradeoff:

- lower activation memory,
- extra forward compute,
- more useful for large transformers than small models.

## Observability commands

```bash
TORCH_LOGS=graph make run-log SCRIPT=script.py
TORCH_LOGS=graph_breaks,recompiles,guards make run-log SCRIPT=script.py
TORCH_LOGS=+inductor make run-log SCRIPT=script.py
TORCH_LOGS=output_code make run-log SCRIPT=script.py
TORCH_COMPILE_DEBUG=1 make run-log SCRIPT=script.py
```

Use `torch._dynamo.explain` to show graph breaks:

```python
explanation = torch._dynamo.explain(fn, x)
print(explanation.graph_break_count)
print(explanation.break_reasons)
```

## Hands-on phases

### Phase 0: Graph inspection

- Trace `x + b`, `(x + b) * scale`, and `relu` with `torch.fx`.
- Print the eager graph before and after `torch.compile`.
- Identify intermediate nodes and users.

### Phase 1: Manual pointwise fusion

- Implement the fused `add + mul + relu` in Triton and CUDA.
- Compare correctness and kernel count with eager and `torch.compile`.

### Phase 2: Vertical epilogue fusion

- Implement `matmul + bias + relu` with and without a GEMM epilogue.
- Compare Inductor default vs native matmul codegen.

### Phase 3: Horizontal fusion

- Construct independent pointwise ops sharing an input.
- Observe whether Inductor merges them into one kernel.
- Identify shapes/dtypes that prevent horizontal fusion.

### Phase 4: Algebraic and cleanup passes

- Manually implement DCE, CSE, and constant folding on small FX graphs.
- Compare with AOTAutograd/Inductor output.

### Phase 5: Layout and memory passes

- Benchmark explicit `transpose` vs layout propagation.
- Enable and disable `optimize_mem`.
- Inspect generated buffer reuse.

### Phase 6: Functionalization and mutation

- Trace in-place ops with and without `torch.compile`.
- Observe the functionalized graph.

### Phase 7: Dynamic shapes and guards

- Run the same compiled function with changing shapes.
- Record recompiles and guard failures.
- Test `dynamic=True` and constrained dynamic dimensions.

### Phase 8: End-to-end audit

For one representative model, produce:

- eager kernel count,
- compiled kernel count,
- fusion report,
- memory-buffer report,
- graph-break report,
- recompile report.

## Coverage checklist

- [ ] vertical pointwise fusion
- [ ] reduction epilogue fusion
- [ ] horizontal fusion
- [ ] dead-code elimination
- [ ] common-subexpression elimination
- [ ] constant folding
- [ ] decomposition
- [ ] layout propagation
- [ ] memory planning/buffer reuse
- [ ] functionalization
- [ ] activation recomputation
- [ ] dynamic-shape guard specialization

## Relation to other plans

- `torch_compile_study_plan.md`: end-to-end Dynamo/Inductor workbook.
- `cuda_operator_optimization_plan.md`: inside-kernel CUDA optimization.
- `custom_accelerator_backend_plan.md`: applying these graph optimizations to a
  new backend.
