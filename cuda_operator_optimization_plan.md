# CUDA Operator Implementation and Optimization Plan

This plan studies typical deep-learning operators by implementing each one in
three ways:

1. **PyTorch** eager or `torch.compile` as the correctness/performance
   reference.
2. **Triton** as the high-productivity block-level GPU language.
3. **CUDA C++** as the low-level implementation for optimization and profiling.

The focus is the CUDA optimization guide: start from a correct kernel, then
improve memory access, occupancy, parallelism, shared memory, and tensor-core
usage while measuring each change.

## Target operators

- **Elementwise / pointwise:** `relu`, `add`, `mul`, `sigmoid`, and fused
  `bias + relu`.
- **Reduction:** `sum`, `mean`, and the reduction part of `softmax`.
- **Matmul / GEMM:** `C = A @ B`, with optional bias + activation epilogue.
- **Convolution:** `conv1d`, `conv2d`, and `conv3d`, direct, GEMM-lowered, or
  cuDNN-backed.
- **Normalization:** `batch_norm`, `layer_norm`, `group_norm`, and `rms_norm`.
- **Attention:** scaled dot-product attention, softmax, causal masking, and
  flash-attention-style tiled kernels.
- **Data movement and indexing:** `transpose`, `permute`, `contiguous`,
  `gather`, `scatter`, and `embedding`.
- **Pooling and stochastic ops:** max/avg pooling, global pooling, and dropout.

These three families cover the most important CUDA patterns: memory-bound
kernels, cross-thread reductions, and compute/memory-bound GEMMs.

## Three implementation layers

| Layer | How it works | Best for |
| --- | --- | --- |
| PyTorch | `torch.relu`, `torch.matmul`, `torch.compile` | baseline, correctness, fast start |
| Triton | block programs, `tl.load`, `tl.dot`, masks | productive custom kernels, Inductor-style code |
| CUDA C++ | explicit threads, shared memory, warp primitives | maximum control, detailed profiling, learning |

Every optimized CUDA kernel should be compared against the equivalent PyTorch
and Triton version:

- correctness and numerical tolerance,
- kernel time from CUDA events or `ncu`,
- achieved memory throughput and occupancy,
- compile complexity and maintainability.

## Environment and profiling setup

Use these tools from the main workbook:

```bash
nvcc --version
nvidia-smi
make check
/usr/local/cuda/bin/ncu --version
/usr/local/cuda/bin/nsys --version
```

For every kernel, capture at least:

```bash
/usr/local/cuda/bin/ncu --set full .venv/bin/python script.py
/usr/local/cuda/bin/nsys profile .venv/bin/python script.py
```

Key `ncu` metrics:

- `gpu__time_duration.sum`
- `sm__throughput.avg.pct_of_peak_sustained_elapsed`
- `dram__throughput.avg.pct_of_peak_sustained_elapsed`
- `l1tex__throughput.avg.pct_of_peak_sustained_elapsed`
- `sm__warps_active.avg.pct_of_peak_sustained_active`
- `launch__registers_per_thread`
- `launch__occupancy_limit_registers`
- `lts__t_sectors_srcunit_tex_op_read.sum`

If `ncu` permissions are unavailable, use CUDA events for timing and
`torch.profiler` for coarse kernel names.

## CUDA optimization guide

Apply this checklist to every CUDA kernel:

1. **Expose enough parallelism.** Use enough blocks and a grid-stride loop when
   the grid is bounded.
2. **Coalesce global memory.** Adjacent threads should access adjacent
   addresses.
3. **Vectorize loads/stores.** Use `float2`, `float4`, or `int4` when alignment
   allows.
4. **Hide latency with occupancy.** Balance registers, shared memory, and
   threads per block against SM limits.
5. **Reuse data in shared memory.** Tile matmul inputs and reductions that are
   read more than once.
6. **Avoid shared-memory bank conflicts.** Pad tiles or use conflict-free
   access patterns.
7. **Use warp-level primitives.** `__shfl_down_sync`, `__shfl_xor_sync`, and
   `__activemask` avoid shared-memory round trips for small reductions.
8. **Increase instruction-level parallelism.** Unroll loops and keep multiple
   independent loads in flight.
9. **Use tensor cores.** `mma.sync`, WMMA, or CUTLASS are faster than FFMA for
   fp16/bf16 GEMM.
10. **Tune launch bounds.** `__launch_bounds__` can reduce register pressure
    and improve occupancy.
11. **Handle tails safely.** Mask or specialize boundary iterations without
    hurting the steady-state path.
12. **Minimize launch overhead.** Fuse epilogues and use fewer kernels for
    short operations.

## Optimization technique coverage matrix

Every essential technique should appear in at least one operator demo. This
matrix assigns a primary and secondary demo for each technique.

| Technique | Primary demo | Secondary demo |
| --- | --- | --- |
| Coalesced global memory | elementwise `relu`/fused add | matmul global tile load |
| Vectorized load/store | elementwise `float4` | transpose tile copy |
| Grid-stride loop | elementwise | embedding over variable rows |
| Occupancy and launch bounds | elementwise | matmul register tuning |
| Shared-memory tiling | matmul | convolution |
| Bank-conflict avoidance | transpose | matmul shared tile |
| Tree/warp reduction | `sum`/`mean` | layer norm |
| Multi-block reduction | `mean` over large tensor | softmax |
| Register blocking and ILP | matmul | direct convolution |
| Software pipelining | matmul | convolution/attention |
| Tensor cores | fp16 matmul | attention QKV GEMMs |
| Kernel fusion / epilogue | matmul + bias + relu | layer norm + residual |
| Boundary masking | matmul non-multiple shape | convolution padding |
| Random-access memory behavior | embedding/gather | KV-cache attention |

How to use the matrix:

- Implement the primary demo first; keep it small and measurable.
- Use the secondary demo later to confirm the technique transfers to a
  different memory/compute pattern.
- Record the `ncu` metric that proves the optimization worked, not just wall
  time.
- If a technique does not improve the primary demo, revisit occupancy, problem
  size, or measurement methodology before accepting the result.

At the end of the plan, each row should have a measured before/after result.

## Phase 0: Naive, PyTorch, and Triton references

Create reference implementations:

- NumPy or pure-Python naive loops for each operator family
- `torch.relu`, `torch.add`, `torch.sum`, `torch.softmax`, `torch.matmul`
- `torch.compile` variants
- hand-written Triton versions from the main Task 08 workbook

Measure and record the baseline kernel time and numerical output. These become
the target for the CUDA kernels.

### Naive reference implementations

The naive versions are deliberately simple. They define correctness first and
provide a floor that every optimized kernel must beat numerically.

Elementwise `relu` and fused `relu(x + b)`:

```python
def relu(x):
    return np.where(x > 0, x, 0)

def add_relu(x, b):
    return np.where(x + b > 0, x + b, 0)
```

Reduction `sum`, `mean`, and row-wise `softmax`:

```python
def reduce_sum(x, axis=None):
    out = 0.0
    for value in x:
        out += value
    return out

def row_softmax(x):
    out = []
    for row in x:
        m = max(row)
        e = [math.exp(v - m) for v in row]
        s = sum(e)
        out.append([v / s for v in e])
    return np.asarray(out)
```

Matmul:

```python
def matmul(a, b):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = np.zeros((M, N), dtype=a.dtype)
    for m in range(M):
        for n in range(N):
            for k in range(K):
                c[m, n] += a[m, k] * b[k, n]
    return c
```

Direct `conv2d`:

```python
def conv2d(x, w, stride=1, padding=0):
    N, C_in, H, W = x.shape
    C_out, _, KH, KW = w.shape
    H_out = (H + 2 * padding - KH) // stride + 1
    W_out = (W + 2 * padding - KW) // stride + 1
    out = np.zeros((N, C_out, H_out, W_out))
    for n in range(N):
        for co in range(C_out):
            for ho in range(H_out):
                for wo in range(W_out):
                    total = 0.0
                    for ci in range(C_in):
                        for kh in range(KH):
                            for kw in range(KW):
                                hi = ho * stride - padding + kh
                                wi = wo * stride - padding + kw
                                if 0 <= hi < H and 0 <= wi < W:
                                    total += x[n, ci, hi, wi] * w[co, ci, kh, kw]
                    out[n, co, ho, wo] = total
    return out
```

Layer norm:

```python
def layer_norm(x, weight, bias, eps=1e-5):
    out = np.empty_like(x)
    for i, row in enumerate(x):
        mean = row.mean()
        var = ((row - mean) ** 2).mean()
        normed = (row - mean) / math.sqrt(var + eps)
        out[i] = normed * weight + bias
    return out
```

Materialized attention:

```python
def attention(q, k, v, scale):
    scores = q @ k.T * scale
    probs = row_softmax(scores)
    return probs @ v
```

Store these references with fixed input tensors so every CUDA/Triton kernel can
be checked against the same values and tolerances.

Output:

- baseline results table,
- saved input/output tensors for deterministic comparisons.

Check:

- Can PyTorch and Triton outputs agree within the expected tolerance?

## Phase 1: Elementwise kernels

Implement the same elementwise operation in CUDA and optimize it through the
standard ladder.

### 1.1 Naive one-thread-per-element `relu`

```cuda
__global__ void relu_kernel(const float* __restrict__ x,
                            float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = fmaxf(x[i], 0.0f);
    }
}
```

### 1.2 Grid-stride loop

Use a grid-stride loop when input size can exceed the launch grid:

```cuda
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
     i += blockDim.x * gridDim.x) {
    y[i] = fmaxf(x[i], 0.0f);
}
```

### 1.3 Vectorized access

For aligned contiguous tensors, process four floats per thread:

```cuda
const float4* x4 = reinterpret_cast<const float4*>(x);
float4* y4 = reinterpret_cast<float4*>(y);
// one float4 load and store per element
```

### 1.4 Fused pointwise chain

Fuse `y = relu(x + b)` so the value is read once, computed in registers, and
written once:

```text
load x and b -> add -> relu -> store y
```

Measurements:

- achieved global memory bandwidth,
- occupancy,
- whether the kernel is memory-bound.

Expected result:

- vectorized fused kernel approaches peak DRAM bandwidth.

## Phase 2: Reduction kernels

Implement `sum`/`mean` and use the same techniques needed for `softmax`.

### 2.1 Naive global reduction

One thread per element and atomics into a single output:

```cuda
atomicAdd(out, x[i]);
```

This is correct but serializes all updates. It is a teaching baseline, not a
production kernel.

### 2.2 Shared-memory tree reduction

- Each block reduces its tile into shared memory.
- Threads stride down and combine values:

```cuda
for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    __syncthreads();
    if (threadIdx.x < stride) {
        sdata[threadIdx.x] += sdata[threadIdx.x + stride];
    }
}
```

- Write one partial sum per block.

### 2.3 Warp shuffle reduction

Reduce inside a warp without shared memory:

```cuda
for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
}
```

### 2.4 Softmax online reduction

For a row of size `K`:

- compute running max to improve numerical stability,
- accumulate `sum(exp(x - running_max))`,
- compute `exp(x - max) / sum`,
- fuse the normalization into the same kernel when possible.

Measurements:

- reduction time vs PyTorch `sum`/`softmax`,
- shared-memory bank conflicts,
- divergence and active mask correctness.

Expected result:

- warp-shuffle reduction beats atomics and shared-memory-only reduction.

## Phase 3: Matmul kernels

Implement GEMM from a naive triple loop to a tiled tensor-core kernel.

### 3.1 Naive matmul

One thread computes one output element:

```cuda
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
if (row < M && col < N) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
}
```

This version is usually memory/latency limited and far from cuBLAS.

### 3.2 Coalesced shared-memory tiling

- Load `A` and `B` tiles into shared memory.
- Each thread block computes a `BLOCK_M x BLOCK_N` output tile.
- Each thread accumulates a sub-tile in registers.
- Advance by `BLOCK_K`.

The shared-memory tiles should be loaded coalesced from global memory and
accessed conflict-free inside the block.

### 3.3 Register blocking and vectorization

- Let each thread compute a `TM x TN` register tile.
- Load `TM`/`TN` values with vectorized types when aligned.
- Keep multiple independent `acc` values to hide FMA latency.

### 3.4 Software pipelining and double buffering

Overlap global loads for iteration `k+1` with compute for iteration `k`:

```text
load tile[0]
for k:
    prefetch tile[k+1]
    compute tile[k]
    swap buffers
```

### 3.5 Tensor cores

For fp16/bf16 GEMM, use tensor cores rather than FFMA:

- CUDA WMMA API for a simple start,
- raw `mma.sync` for detailed control,
- CUTLASS for production-grade tiled tensor-core GEMM.

Measurements:

- FLOP/s vs naive kernel and cuBLAS,
- shared-memory throughput and bank conflicts,
- register pressure and occupancy,
- global-load reuse.

Expected result:

- shared-memory tiled GEMM is much faster than naive.
- tensor-core GEMM approaches cuBLAS on well-shaped square matrices.

## Phase 4: Fused epilogue

Do not write matmul output and then launch a separate `bias + relu` kernel.

- Add the epilogue inside the matmul kernel after the reduction loop.
- For tensor-core GEMM, pass the accumulator through a register epilogue.
- Compare with Inductor's fused `matmul + bias + relu` kernel from Task 04.

Expected result:

- fused epilogue removes a full global-memory round trip.

## Additional essential operator families

### Convolution

Implement the main strategies in order of difficulty:

1. **im2col + GEMM:** lower convolution to matrix multiplication and reuse the
   Phase 3 GEMM.
2. **Direct convolution:** compute each output element from the corresponding
   input patch and filter taps.
3. **Shared-memory tiling:** tile input and filter across the channel, spatial,
   and batch dimensions.
4. **Winograd/FFT:** reduce arithmetic for small filters, at the cost of more
   memory traffic and precision complexity.
5. **cuDNN:** use the vendor implementation as the production reference.

CUDA considerations:

- NCHW is usually easier to vectorize than NHWC, but NHWC can improve tensor-core
  compatibility for GEMM-lowered implementations.
- Overlap channel-reduction work with input/filter loads.
- Handle padding, stride, dilation, and groups without branching on every
  element.
- Compare with `torch.nn.functional.conv2d` and `cuDNN` FLOP/s.

### Normalization

Normalization combines elementwise math with reductions over a specific axis.

- `batch_norm` reduces over `(N, H, W)` per channel in training and can fold
  into an affine pointwise op at inference.
- `layer_norm` reduces over the feature dimension for each token.
- `group_norm` reduces over grouped channels per token.
- `rms_norm` replaces variance with root mean square and is common in LLMs.

Optimization guide:

- Reuse the warp-shuffle/shared-memory reduction from Phase 2.
- Fuse `x - mean`, variance, normalization, and affine scale/bias into one
  kernel.
- For small hidden sizes, one block per token with a warp reduction is often
  fastest.
- Keep mean/variance in registers or shared memory instead of round-tripping
  through global memory.
- Compare against `torch.nn.LayerNorm` and Inductor's generated
  `triton_per_fused_*_layer_norm` kernels.

### Attention and softmax

Scaled dot-product attention is the most important composed operator in modern
models:

```text
S = Q @ K^T
P = softmax(S / sqrt(d))
O = P @ V
```

Optimization ladder:

1. Separate GEMM + softmax + GEMM kernels using the earlier primitives.
2. Fuse the softmax normalization into the second GEMM epilogue.
3. Tile `Q`, `K`, and `V` in shared memory and compute attention one block of
   tokens at a time.
4. Add causal masking and RoPE.
5. Use flash-attention-style online softmax to avoid materializing `S`.
6. Integrate KV-cache layouts and PagedAttention for inference.

Key CUDA points:

- Keep the `Q` block resident in shared memory and stream `K`/`V` blocks.
- Use online max/sum updates to avoid a second pass over `S`.
- Avoid atomic contention in the multi-block softmax path.
- Compare against `torch.nn.functional.scaled_dot_product_attention` and
  Inductor's `triton_per_fused__softmax_prepare_softmax_online` kernels.

### Embedding, gather, and scatter

Embedding and indexing ops are usually memory-bound random accesses.

- `embedding`: one gather per token; optimize table layout and cache reuse.
- `gather`: load values from arbitrary indices; ensure coalesced output writes
  even when input reads are random.
- `scatter` / `index_add`: handle duplicate output indices with atomics or
  deterministic multi-pass accumulation.

Measure effective memory bandwidth and compare against `torch.nn.Embedding`,
`torch.gather`, and `torch.scatter_`.

### Transpose, permute, and contiguous

These ops often expose bad global-memory access patterns.

Optimization guide:

- For square tiles, stage data through shared memory with padded indices to
  avoid bank conflicts.
- For large contiguous copies, use vectorized `float4`/`int4` loads and stores.
- For `permute`/`transpose`, prefer writing coalesced output even when input
  reads are strided.
- Use shared memory only when the tile benefits from reuse; otherwise a direct
  strided copy may be faster.

### Pooling and dropout

- Max/avg pooling is a windowed reduction; reuse shared-memory tile reductions
  for overlapping windows.
- Global pooling is a full-tensor reduction; use the multi-block reduction
  strategy.
- Dropout is elementwise random masking followed by scale; fuse mask generation
  and scale into the surrounding elementwise kernel instead of launching a
  separate kernel.

### Additional pointwise activations

Add the common non-linear activations to the elementwise plan:

- `gelu` and its tanh approximation,
- `silu`/`swish`,
- `leaky_relu`,
- `softplus`,
- `hardsigmoid` and `hardswish`.

Use the same vectorization and fusion rules as Phase 1. Accuracy and throughput
often differ between exact and approximate variants, so record both.

## Transformer kernel and fusion plan

Transformers are composed from the operator families above. The optimization
work is therefore less about inventing new primitives and more about fusing
them around the model's data flow.

### Representative block

```text
x -> LayerNorm -> attention -> residual add
  -> LayerNorm -> MLP -> residual add
```

The main CUDA kernels are:

- QKV projection GEMMs,
- scaled dot-product attention,
- output projection GEMM,
- two LayerNorm/RMSNorm reductions,
- two MLP GEMMs,
- GELU/SiLU activation,
- residual adds and optional dropout.

### Attention variants

Start with the simple materialized attention from the attention section, then
add production variants:

- **Causal attention:** apply a triangular mask and avoid computing masked
  positions.
- **Flash attention:** tile `Q`, `K`, and `V` so the full score matrix is never
  written to global memory.
- **Fused QKV:** compute the three projection GEMMs in one kernel when memory
  layout allows.
- **RoPE:** fuse rotary position embedding into the Q/K load path rather than
  applying it as a separate tensor op.
- **KV cache:** support sequence-length dimensions that grow at inference time.
- **PagedAttention:** page KV-cache blocks to reduce fragmentation and support
  continuous batching.

### MLP fusion

The common MLP is:

```text
x -> linear(d, 4d) -> activation -> linear(4d, d)
```

Optimize by:

- fusing the activation into the first GEMM epilogue,
- fusing bias into the GEMM,
- fusing the residual add/dropout into the second GEMM epilogue when possible,
- using a fast approximate GELU when accuracy permits.

This is the same fused-epilogue pattern as Phase 4, applied twice per block.

### Normalization and residual fusion

Fuse the operations around normalization:

```text
residual -> layernorm -> scale -> bias -> residual_add
```

Inductor already emits fused `triton_per_fused_*_layer_norm` and pointwise
residual kernels. The CUDA version should follow the same rule: do not round-trip
the residual or normalization statistics through global memory more than
necessary.

### Training vs inference

Training:

- backward kernels for attention, softmax, and normalization are required.
- activation checkpointing trades recompute for memory.
- batch sizes and sequence lengths are usually smaller and more variable.

Inference:

- weights are frozen and can be pretransformed or folded.
- dynamic sequence length and KV-cache access dominate.
- CUDA graphs help fixed-shape serving; continuous batching needs more flexible
  launch patterns.
- FP8/INT8/INT4 quantization is usually applied first.

### Parallelism and multi-GPU boundaries

Single-kernel optimization is only part of transformer performance:

- tensor parallelism splits projection weights across GPUs,
- sequence parallelism splits LayerNorm/dropout across the sequence,
- pipeline parallelism overlaps transformer blocks,
- all-reduce communication competes with GEMM compute,
- scheduling must overlap attention/MLP compute with communication.

For the CUDA-focused portion of this plan, implement and profile the
single-GPU fused block first, then study multi-GPU effects with `nsys`.

### Transformer profiling workflow

1. Run eager PyTorch and record the kernel timeline with `torch.profiler`.
2. Run `torch.compile` and compare the fused kernel list.
3. Profile the top kernels with `ncu`.
4. Replace the largest unfused sequence with a hand-written CUDA/Triton kernel.
5. Measure end-to-end block time, not just individual kernel time.

Typical findings:

- GEMMs dominate in large models,
- LayerNorm/dropout/residual launches dominate small-model or short-sequence
  cases,
- unfused softmax and elementwise ops can be removed by flash attention and
  epilogue fusion,
- launch overhead matters for small batches and short sequences.

## Phase 5: Profiling and tuning loop

For each optimized kernel, run:

```bash
/usr/local/cuda/bin/ncu --set full .venv/bin/python benchmark.py
```

Record:

- kernel duration,
- achieved occupancy,
- DRAM and shared-memory throughput,
- register usage,
- achieved vs theoretical FLOP/s,
- number of kernels and launch overhead.

Only keep an optimization if it improves a real metric. Common traps:

- improving occupancy while lowering memory reuse,
- reducing registers but spilling to local memory,
- optimizing a tail path that is rarely executed,
- benchmarking without warming up the device,
- comparing asynchronous kernels without synchronization.

## Phase 6: Integrate CUDA kernels into PyTorch

Once kernels are correct, expose them as PyTorch ops.

Options:

- `torch.utils.cpp_extension.load_inline` for rapid experiments,
- a compiled C++/CUDA extension for reusable kernels,
- `torch.library` for custom ATen-level dispatch,
- a custom Inductor backend for `torch.compile` integration.

For the custom Inductor route, continue with
[custom_accelerator_backend_plan.md](custom_accelerator_backend_plan.md), but the
device is CUDA here and the generated backend can lower nodes to the hand-written
CUDA kernels.

## Milestones

- [ ] PyTorch and Triton references pass numerical parity
- [ ] fused elementwise CUDA kernel reaches high DRAM bandwidth
- [ ] warp-shuffle reduction beats atomics
- [ ] shared-memory tiled matmul beats naive matmul
- [ ] tensor-core matmul approaches cuBLAS
- [ ] matmul epilogue is fused
- [ ] direct or GEMM-lowered convolution is profiled against cuDNN
- [ ] layer norm is fused into one kernel
- [ ] tiled attention avoids materializing the full score matrix
- [ ] embedding/gather and transpose kernels reach high effective bandwidth
- [ ] a full transformer block is fused and profiled as one optimized kernel set
- [ ] `ncu` metrics explain each performance change
- [ ] kernels are callable from PyTorch

## Relation to the main workbook

This operator plan is the low-level counterpart to the main
`torch_compile_study_plan.md`:

- Task 00 establishes the eager/compiled baseline.
- Task 04 shows Inductor-generated Triton kernels for matmul, reduction, and
  pointwise ops.
- Task 08 compares a hand-written Triton matmul with Inductor.
- This plan replaces the generated Triton layer with hand-written CUDA C++ and
  profiles each optimization step.

The custom accelerator plan then generalizes the same operator and backend
boundaries to a non-NVIDIA device.
