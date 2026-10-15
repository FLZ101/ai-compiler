# Task 05: Fusion and scheduling

This task runs five small computation graphs and observes how Inductor fuses
them into different numbers of GPU kernels.

## Run

```bash
find tasks/05/debug -mindepth 1 -delete 2>/dev/null || true
TORCH_COMPILE_DEBUG=1 TORCH_LOGS=+fusion,+inductor,output_code \
  make run SCRIPT=tasks/05/fusion_matrix.py \
  > tasks/05/fusion_matrix.log 2>&1
```

The `+fusion` log shows scheduler fusion decisions. The `+inductor` and
`output_code` logs show the final kernel names and runtime calls.

## Actual result

PyTorch/Inductor version in this environment uses cuBLAS for normal fp32
matmuls, so a matmul appears as `extern_kernels.mm` rather than a generated
Triton GEMM.

| Example | Fused? | Number of kernels | Kernel names | Notes |
| --- | --- | --- | --- | --- |
| pointwise chain | yes | 1 | `triton_poi_fused_add_0` | all three adds are one pointwise kernel |
| matmul + bias + ReLU | partial | 2 | `extern_kernels.mm`, `triton_poi_fused_add_relu_0` | bias+ReLU fused into the GEMM epilogue |
| attention pattern | partial | 3 | `extern_kernels.bmm`, `triton_per_fused__softmax_prepare_softmax_online_0`, `extern_kernels.bmm` | softmax reduction fused internally; the two matmuls remain separate |
| single matmul | yes | 1 | `extern_kernels.mm` | one cuBLAS GEMM |
| chained matmul | no | 2 | `extern_kernels.mm`, `extern_kernels.mm` | two dependent matmuls cannot fuse |

## Why the counts differ

### Pointwise chain

```python
return x + 1 + 2 + 3
```

This is three dependent elementwise adds. Inductor folds all of them into one
loop body and emits a single `triton_poi_fused_add_0` kernel. There is no need
to write intermediate `x + 1` or `x + 1 + 2` tensors to global memory.

### Matmul + bias + ReLU

```python
return torch.relu(x @ w + b)
```

The matmul is an external cuBLAS kernel. The bias add and ReLU are both
pointwise, so Inductor fuses them into `triton_poi_fused_add_relu_0`.

The two kernels are:

```text
buf0 = mm(x, w)                 # extern_kernels.mm
buf1 = relu(buf0 + b)           # triton_poi_fused_add_relu_0
```

This is vertical/epilogue fusion: the pointwise work is fused with itself, but
the external GEMM is a separate stage. If Task 04 native matmul mode is enabled,
Inductor can emit the matmul, bias, and ReLU as one Triton GEMM kernel.

### Attention pattern

```python
scores = q @ k.transpose(-2, -1)
attn = torch.softmax(scores, dim=-1)
return attn @ v
```

Inductor lowers this to three kernels:

```text
scores = bmm(q, k^T)                        # extern_kernels.bmm
attn   = online_softmax(scores)             # persistent Triton reduction
out    = bmm(attn, v)                       # extern_kernels.bmm
```

The softmax kernel is `triton_per_fused__softmax_prepare_softmax_online_0`.
Its internal subtraction, exponential, and division are fused into the
reduction. The two batched matmuls remain separate because the softmax output
must be fully materialized before the second matmul can read it.

### Single matmul

```python
return a @ b
```

There is nothing to fuse. Inductor calls `extern_kernels.mm` once.

### Chained matmul

```python
left = a @ b
right = left @ c
return right
```

The second matmul consumes the full output of the first matmul, so both results
must be materialized:

```text
left  = a @ b
right = left @ c
```

That gives two separate `extern_kernels.mm` calls. Inductor does not merge two
dependent GEMMs into one kernel.

## Horizontal vs vertical fusion

The detailed comparison is in
[horizontal_vs_vertical_fusion.md](horizontal_vs_vertical_fusion.md).

In this task, the clearest vertical fusion is the bias+ReLU epilogue after the
matmul. The softmax kernel is a reduction plus pointwise epilogue fused into
one persistent reduction kernel.

## Check

- You can distinguish horizontal from vertical fusion.
- You can explain why attention usually requires multiple kernels: softmax is a
  reduction over the last dimension, and its result is an input to another
  matmul, so the schedule needs at least two matmuls plus the softmax stage.
- The exact kernel count can vary by PyTorch version; this table records what
  the installed version actually emitted.
