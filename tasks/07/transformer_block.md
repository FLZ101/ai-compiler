# Task 07: Profile a realistic transformer block

This task runs a small transformer block through eager PyTorch and
`torch.compile`, records GPU timings, and uses Nsight tools to inspect which
kernels are taking time.

## Files

- `tasks/07/transformer_block.py`
- `tasks/07/transformer_block.md`

## Run

```bash
make run SCRIPT=tasks/07/transformer_block.py
```

Capture the generated-code log into the ignored debug directory:

```bash
TORCH_LOGS=output_code make run-log SCRIPT=tasks/07/transformer_block.py
```

## Model

The block contains:

- `nn.MultiheadAttention`
- two `LayerNorm`s
- an MLP with `Linear -> GELU -> Linear`

Input shape: `[8, 128, 256]`.

## Results

On the RTX 3060 Laptop GPU, a clean run produces:

```text
eager              max_diff=0.000e+00     0.675 ms/iter
compiled           max_diff=7.153e-07     0.589 ms/iter
reduce-overhead    max_diff=7.153e-07     0.592 ms/iter
```

The compiled graph is slightly faster than eager, but `reduce-overhead` does not
improve this particular block. The block is small enough that CUDA graph
replay does not remove a meaningful CPU bottleneck.

## Torch profiler result

Because `nsys` did not collect CUDA kernel rows in this environment, the
built-in PyTorch profiler is the reliable way to rank kernels:

```bash
PROFILE=1 make run SCRIPT=tasks/07/transformer_block.py
```

The top CUDA kernels for the compiled path were:

| Kernel | Self CUDA | Calls | Avg |
| --- | --- | --- | --- |
| `ampere_sgemm_64x64_tn` | 1.934 ms | 20 | 96.7 us |
| `ampere_sgemm_128x64_tn` | 1.008 ms | 10 | 100.8 us |
| `ampere_sgemm_128x128_nn` | 0.639 ms | 10 | 63.9 us |
| `triton_poi_fused_addmm_gelu_view_6` | 0.300 ms | 10 | 30.0 us |
| `triton_per_fused__softmax_prepare_softmax_online_3` | 0.250 ms | 10 | 25.0 us |
| `triton_poi_fused__unsafe_view_add_clone_squeeze_transpose_unsqueeze_view_1` | 0.182 ms | 10 | 18.2 us |
| `triton_per_fused_add_addmm_clone_native_layer_norm_transpose_view_5` | 0.083 ms | 10 | 8.3 us |

The `aten::mm` operator aggregates to about 3.365 ms of CUDA time and contains
the `ampere_sgemm_*` kernels. Those cuBLAS GEMMs dominate the block.

## Bottleneck

The bottleneck is GEMM work from the attention projections, attention
batched matmuls, and MLP linear layers. Inductor successfully fuses the
non-GEMM parts into small pointwise/reduction Triton kernels.

This block is mostly compute-bound at the current small batch size. CUDA graphs
do not help because the Python/CPU launch side is not the limiting factor.

## Nsight Systems

Profile the process:

```bash
mkdir -p tasks/07/debug
/usr/local/cuda/bin/nsys profile \
  --trace=cuda,nvtx,osrt --syscall=process-tree \
  --cuda-memory-usage=true \
  --pytorch=autograd-nvtx \
  --stats=true --force-overwrite=true \
  --output=tasks/07/debug/nsys_transformer \
  .venv/bin/python tasks/07/transformer_block.py
```

If running under `sudo`, `nsys` may not be on the secure `PATH`. Use the full
path above:

```bash
sudo /usr/local/cuda/bin/nsys ...
```

`--pytorch=autograd-nvtx` makes PyTorch emit NVTX ranges. Without it, `nsys`
reports:

```text
SKIPPED: ... does not contain NV Tools Extension (NVTX) data.
```

On this machine, `--syscall=process-tree` may be disabled by Nsight Systems:

```text
WARNING: --syscall is not available and will be disabled.
CAP_BPF and CAP_PERFMON capabilities ... are required to run syscall tracing.
```

Without those capabilities, the generated report will not contain syscall
data. The same environment also prevents `nsys` from collecting CUDA kernel
rows. Because GPU memory reports depend on CUDA kernel/memory trace data, the
`cuda_gpu_mem_*` reports are skipped too. In this environment, the PyTorch
profiler is used for kernel-level analysis instead.

Inspect the report:

```bash
/usr/local/cuda/bin/nsys stats tasks/07/debug/nsys_transformer.nsys-rep
```

Record the top three kernels by GPU time.

## Nsight Compute

Optional deeper analysis:

```bash
/usr/local/cuda/bin/ncu --set full .venv/bin/python tasks/07/transformer_block.py
```

This can be slow. A faster follow-up is to target one kernel after identifying
it with `nsys`.

In this environment, `ncu` requires performance-counter permissions and failed
with `ERR_NVGPUCTRPERM`. The PyTorch profiler table above was used instead.

## Check

- Which kernel is the bottleneck?
- Is this block launch-bound or compute-bound?
- Does `reduce-overhead` improve GPU time on this block?
