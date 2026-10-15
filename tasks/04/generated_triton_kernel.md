# Task 04: Annotate the generated Triton kernel

This task looks at the final Inductor output: a real Triton kernel generated for
the NVIDIA GPU. The goal is not to memorize the wrapper code around it, but to
be able to find the few lines that do the actual work.

## Run

```bash
find tasks/04/debug -mindepth 1 -delete 2>/dev/null || true
TORCH_COMPILE_DEBUG=1 TORCH_LOGS=output_code,+inductor \
  make run SCRIPT=tasks/04/generated_triton_kernels.py \
  > tasks/04/generated_code.log 2>&1
```

The `+inductor` logger is added because `output_code` prints the generated
Python, while `+inductor` also prints the launch configuration chosen by
Inductor's Triton heuristic.

`TORCH_COMPILE_DEBUG=1` makes Inductor write its complete artifact tree to
`tasks/04/debug/`. The annotated kernel snippets in this note are the readable
summary; the full generated code is in `tasks/04/generated_code.log` and the
debug tree.

## Why the debug tree has four directories

The script calls four separate `@torch.compile` functions, and Inductor writes
one debug trace directory per compiled graph:

| Debug directory | Compiled function |
| --- | --- |
| `model__0_inference_0.0` | `matmul_bias_relu(x, w, b)` |
| `model__1_inference_1.1` | `pointwise(x)` |
| `model__2_inference_2.2` | `reduction(x)` |
| `model__3_inference_3.3` | `layernorm(x)` |

The number before `inference` is the compile counter, in call order. Each
directory contains that function's FX graphs, pre/post-fusion IR, and
`output_code.py`.

This is per compiled function, not per Triton kernel. If one function produced
multiple kernels, those kernels would still live in that function's single
debug directory. If a function recompiles because of a guard miss, Inductor can
create an additional directory for the new graph.

The script enables `inductor_config.triton.native_matmul = True`. Without it,
this fp32 matmul is sent to `extern_kernels.mm` (cuBLAS), so the interesting
Triton GEMM body with `tl.dot` is not generated. Native matmul makes Inductor
lower `aten.mm` directly into Triton code.

## The graph being lowered

The first example is:

```python
return torch.relu(x @ w + b)
```

The logical graph is:

```text
%mm   = aten.mm(%arg0_1, %arg1_1)
%add  = aten.add.Tensor(%mm, %arg2_1)
%relu = aten.relu.default(%add)
```

The point is that the matmul, bias, and ReLU are three source nodes, but
Inductor is allowed to generate them as one fused Triton program.

## Annotated native matmul + bias + ReLU kernel

The generated name is:

```text
triton_red_fused_add_mm_relu_0
```

The name already tells the story:

- `triton_`: Inductor-generated Triton kernel.
- `red`: contains a reduction over the `K` dimension.
- `fused`: multiple ATen ops were fused.
- `add_mm_relu`: the ops are `add`, `mm`, and `relu`.

### The interesting body

The snippet below is trimmed to the important code. Inductor also emits
bookkeeping variables such as `roffset`, `rindex`, and `rbase`; they are not
needed for understanding the algorithm.

```python
@triton.jit
def triton_red_fused_add_mm_relu_0(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2,
    ynumel, xnumel, r0_numel,
    YBLOCK: tl.constexpr, XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr,
):
    ynumel = 2048
    xnumel = 2048
    r0_numel = 2048

    # One grid program computes a 2-D tile of the output.
    # program_id(1) selects the output-row tile.
    # program_id(0) selects the output-column tile.
    yoffset = tl.program_id(1).to(tl.int64) * YBLOCK
    # tl.arange(0, YBLOCK) -> [YBLOCK]
    # [:, None, None] -> [YBLOCK, 1, 1]; yoffset broadcasts to every row.
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None, None].to(tl.int64)

    xoffset = tl.program_id(0).to(tl.int64) * XBLOCK
    # tl.arange(0, XBLOCK) -> [XBLOCK]
    # [None, :, None] -> [1, XBLOCK, 1]; xoffset broadcasts to every column.
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :, None].to(tl.int64)
    xmask = xindex < xnumel

    # Reduction index for the inner K dimension.
    r0_base = tl.arange(0, R0_BLOCK)[None, None, :].to(tl.int64)

    y0 = yindex
    x1 = xindex

    # C[YBLOCK, XBLOCK] = A[YBLOCK, K] @ B[K, XBLOCK]
    accumulator = tl.full([YBLOCK, XBLOCK], 0, tl.float32)

    # This is the main reduction loop.
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        # r0_index and r0_mask both have shape [1, 1, R0_BLOCK].
        r0_mask = r0_index < r0_numel
        r0_2 = r0_index

        # Load an A tile: A[y0, k].
        # y0 [YBLOCK, 1, 1] + r0_2 [1, 1, R0_BLOCK] -> [YBLOCK, 1, R0_BLOCK]
        a_tile = tl.load(
            in_ptr0 + (r0_2 + 2048 * y0),
            r0_mask,
            eviction_policy="evict_last",
            other=0.0,
        )

        # Load the corresponding B tile. B is stored row-major as [K, X],
        # so Inductor loads it as [XBLOCK, R0_BLOCK] and transposes it.
        # x1 [1, XBLOCK, 1] + r0_2 [1, 1, R0_BLOCK] -> [1, XBLOCK, R0_BLOCK]
        b_tile = tl.load(
            in_ptr1 + (x1 + 2048 * r0_2),
            r0_mask & xmask,
            eviction_policy="evict_last",
            other=0.0,
        )

        a_tile = tl.reshape(a_tile, [YBLOCK, R0_BLOCK])
        b_tile = tl.trans(tl.reshape(b_tile, [XBLOCK, R0_BLOCK]))

        # The actual tensor-core / matrix instruction candidate.
        partial = tl.dot(a_tile, b_tile, input_precision="ieee")
        partial = tl.broadcast_to(partial, [YBLOCK, XBLOCK])
        accumulator = accumulator + partial

    # Epilogue 1: add the bias. The accumulator shape is [Y, X, 1] after the
    # reduction epilogue conversion, and bias is broadcast across rows.
    result = accumulator[:, :, None]
    bias = tl.load(in_ptr2 + (x1), xmask, eviction_policy="evict_last")
    result = result + bias

    # Epilogue 2: ReLU = max(0, result).
    zero = tl.full([1, 1, 1], 0, tl.int32)
    result = tl.maximum(zero, result, tl.PropagateNan.ALL)

    # Write one output tile.
    tl.store(in_out_ptr0 + (x1 + 2048 * y0), result, xmask)
```

### What the `tl.dot` / broadcast / accumulate lines mean

```python
# One K-block matrix multiply: [YBLOCK, R0_BLOCK] @ [R0_BLOCK, XBLOCK].
partial = tl.dot(a_tile, b_tile, input_precision="ieee")

# Align the result to the [YBLOCK, XBLOCK] output tile shape.
partial = tl.broadcast_to(partial, [YBLOCK, XBLOCK])

# Add this K slice into the running C tile.
accumulator = accumulator + partial
```

Before these lines:

- `a_tile` has shape `[YBLOCK, R0_BLOCK]`: one block of output rows by one block
  of the reduction dimension `K`.
- `b_tile` has shape `[R0_BLOCK, XBLOCK]`: the same `K` block by one block of
  output columns.

The first line computes the matrix product for this `K` tile:

```text
partial[y, x] = sum_k a_tile[y, k] * b_tile[k, x]
```

The result has shape `[YBLOCK, XBLOCK]`.

`input_precision="ieee"` means this fp32 dot product uses IEEE-compliant
floating-point math rather than TensorFloat-32 (`tf32`). That matters when the
result must match eager PyTorch fp32 numerics.

`tl.broadcast_to(partial, [YBLOCK, XBLOCK])` aligns the `tl.dot` result to the
expected output tile shape. In this exact kernel `partial` already has that
shape, so it is effectively a shape annotation rather than a data movement.
Inductor keeps this broadcast because the same code path also handles cases
where the dot result is smaller and must be expanded before accumulation.

Broadcasting means replicating values along size-1 dimensions, not filling
with zeros. For example, a `[YBLOCK, 1]` tensor broadcast to
`[YBLOCK, XBLOCK]` repeats the same column across every output column.
Zero-filling comes from a different mechanism: `tl.load(ptr, mask, other=0.0)`
uses `0.0` only where the load mask is false.

The final line is the reduction accumulator update:

```text
C_tile = C_tile + A_tile @ B_tile
```

Each trip through the `for r0_offset in tl.range(...)` loop adds one slice of
`K`. After the loop, `accumulator` contains the complete
`A[YBLOCK, K] @ B[K, XBLOCK]` tile, and only then does the kernel apply the
bias/ReLU epilogue.

`tl.dot` is a tile-level matrix multiply, not the entire `x @ w` GEMM. A full
matrix multiply is implemented as the tiling loop plus many `tl.dot` calls and
the accumulator update:

```text
full matmul = tile the operands + loop over K + tl.dot + accumulate
```

Triton does not expose `tl.matmul` or `tl.mm`. Its matrix-multiply primitive is
`tl.dot`; `tl.mm` / `torch.matmul` are PyTorch/ATen-level names.

### What to mark

`tl.load` appears three times:

- `in_ptr0` is `x`, the left matmul operand.
- `in_ptr1` is `w`, the right matmul operand.
- `in_ptr2` is the bias.

`tl.dot` is the matrix-multiply accumulator update.

`tl.store` is the final write of the fused output.

The epilogue is everything after the reduction loop:

```python
# Bias is broadcast over every row of the output tile.
bias + accumulator

# ReLU: clamp all negative values to zero.
max(0, bias + accumulator)
```

The block sizes are `YBLOCK`, `XBLOCK`, and `R0_BLOCK`. `R0_BLOCK` is the
reduction block for the `K` dimension. The launch heuristic logged for this
run chose:

```text
XBLOCK: 64, YBLOCK: 128, R0_BLOCK: 32
num_warps: 8, num_stages: 4
```

The kernel is a 2-D grid (`Grid2D`): one grid axis covers output rows, the
other covers output columns.

## Pointwise example

For:

```python
return torch.sin(x) + torch.cos(x)
```

Inductor emits a one-dimensional pointwise kernel:

```python
@triton.jit
def triton_poi_fused_add_cos_sin_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    # Total number of scalar elements in the flattened tensor.
    xnumel = 4194304

    # This program handles one contiguous XBLOCK-sized chunk.
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]

    # Shape is divisible by the block size, so no boundary masking is needed.
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex

    # One global-memory load for the input chunk.
    tmp0 = tl.load(in_ptr0 + (x0), None)

    # Elementwise math: sin(x) and cos(x).
    tmp1 = tl_math.sin(tmp0)
    tmp2 = tl_math.cos(tmp0)

    # Fused epilogue: add the two elementwise results.
    tmp3 = tmp1 + tmp2

    # One global-memory store for the output chunk.
    tl.store(out_ptr0 + (x0), tmp3, None)
```

There is no reduction loop and no `tl.dot`. The entire computation is:

```text
load -> sin -> cos -> add -> store
```

Launch configuration:

```text
XBLOCK: 1024, num_warps: 4, num_stages: 1
```

## Reduction example

For:

```python
return x.sum(dim=1)
```

Inductor emits `triton_red_fused_sum_0`. The important reduction parts are:

```python
@triton.jit
def triton_red_fused_sum_0(
    in_ptr0, out_ptr0, xnumel, r0_numel,
    XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr,
):
    # Output elements and reduction elements.
    xnumel = 2048
    r0_numel = 2048

    # One output element per row: shape [XBLOCK, 1].
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel

    # Reduction indices have shape [1, R0_BLOCK].
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex

    # Accumulate each row's sum across reduction blocks.
    accumulator = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)

    # Main reduction loop over dim=1.
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_1 = r0_index

        # Load a block of values from x[row, r0].
        value = tl.load(
            in_ptr0 + (r0_1 + 2048 * x0),
            r0_mask & xmask,
            eviction_policy="evict_first",
            other=0.0,
        )

        # Broadcast to the 2-D accumulator tile.
        value = tl.broadcast_to(value, [XBLOCK, R0_BLOCK])

        # Add the block, but keep old values where masks are false.
        next_accumulator = accumulator + value
        accumulator = tl.where(r0_mask & xmask, next_accumulator, accumulator)

    # Collapse the reduction axis and store one scalar per row.
    result = tl.sum(accumulator, 1)[:, None]
    tl.store(out_ptr0 + (x0), result, xmask)
```

The difference from pointwise:

- `xindex` is shaped `[XBLOCK, 1]`.
- `r0_base` is shaped `[1, R0_BLOCK]`.
- `tl.range(0, r0_numel, R0_BLOCK)` is the reduction loop.
- `tl.sum(accumulator, 1)` collapses the reduction axis.

Launch configuration:

```text
XBLOCK: 2, R0_BLOCK: 2048
num_warps: 16, num_stages: 1
```

## LayerNorm example

For:

```python
return torch.nn.functional.layer_norm(x, x.shape[-1:])
```

Inductor emits `triton_red_fused_native_layer_norm_0`. The kernel is more
interesting because it performs two reductions over the same data:

1. mean and variance using Welford's algorithm;
2. a second pass to normalize and store the output.

The important shape is:

```python
# Welford combines count, mean, and M2 incrementally while loading values.
tmp2_mean, tmp2_m2, tmp2_weight = triton_helpers.welford_reduce(
    value, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
)
...
# Convert the incremental Welford state into final mean and variance.
mean, m2, weight = triton_helpers.welford(tmp2_mean, tmp2_m2, tmp2_weight, 1)
...
# Normalize: (x - mean) / sqrt(variance + epsilon).
value = (value - mean) * libdevice.rsqrt(variance + epsilon)
```

This is still a `red` kernel, not a separate pointwise kernel. The two loops
share the same output tiling and keep intermediate statistics in registers
instead of writing them to global memory.

Launch configuration:

```text
XBLOCK: 2, R0_BLOCK: 2048
num_warps: 16, num_stages: 1
```

The block-level load primitive is summarized in [tl_load.md](tl_load.md).

## General anatomy of an Inductor Triton kernel

| Part | What it is |
| --- | --- |
| `@triton.jit` | Marks the function as Triton code. |
| `tl.program_id(...)` | Identifies this program in the launch grid. |
| `tl.arange(...)` | Creates the local vector of indices for one block. |
| `XBLOCK` / `YBLOCK` / `R0_BLOCK` | Compile-time block sizes. |
| masks | Guard against out-of-bounds indices. |
| `tl.load(..., other=0.0)` | Global-memory read with safe padding. |
| `tl.range(...)` | The reduction loop over a dimension. |
| `tl.dot` | Matrix-multiply tile instruction. |
| `tl.sum` / Welford helpers | Other reductions. |
| epilogue | Pointwise work after the main reduction. |
| `tl.store` | Global-memory write. |
| `num_warps` / `num_stages` | Launch/pipelining knobs chosen by the heuristic. |

## Check

- The matmul, bias, and ReLU are in one kernel because Inductor fuses the
  dependent pointwise epilogue into the matmul reduction. The matmul writes its
  tile into `accumulator`, then the same program adds bias, applies ReLU, and
  stores the final result.
- The main reduction loop is:

```python
# Loop over all K blocks.
for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
    ...

    # Matrix multiply this K slice.
    partial = tl.dot(a_tile, b_tile, input_precision="ieee")

    # Accumulate the slice into the output tile.
    accumulator = accumulator + partial
```

- Without `inductor_config.triton.native_matmul = True`, the default fp32 GEMM
  would instead show up as `extern_kernels.mm(...)` in `output_code.py`. In that
  mode the generated Triton kernel is only the bias+ReLU pointwise epilogue and
  contains `tl.load`/`tl.store`, but no `tl.dot`.
