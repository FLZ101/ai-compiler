# Autotuning in TorchInductor

Autotuning means asking the compiler to try several ways to execute a kernel,
measure them, and keep the fastest one.

In the context of `torch.compile`, autotuning mostly applies to Triton kernels.
Inductor has several tunable choices for the same generated kernel:

- `XBLOCK`, `YBLOCK`, `RBLOCK`: tile/block sizes
- `num_warps`: number of warps per block
- `num_stages`: software pipelining stages
- `num_ctas`: thread block clusters
- `maxnreg`: register limits
- sometimes different algorithm implementations, such as cuBLAS versus
  generated Triton GEMM

During compilation, Inductor generates a kernel for each candidate
configuration, runs it briefly on the GPU, measures its latency, and records
the winner.

## What it looks like in the log

Task 06's pointwise kernel is tuned with coordinate descent:

```text
Baseline XBLOCK: 1024, num_warps: 4
Try XBLOCK: 2048, num_warps: 4
Try XBLOCK: 512, num_warps: 4
Try XBLOCK: 1024, num_warps: 8
...
best config XBLOCK: 1024, num_warps: 8
```

That is autotuning: a small benchmark of launch configurations at compile time.

## Why it matters

- The best tile/warp configuration depends on the GPU, tensor shape, dtype, and
  kernel type.
- A fixed heuristic is fast to choose but may not be optimal.
- Autotuning spends more compile time, but can reduce runtime.

## In the compile modes

- `default`: mostly uses Inductor's fast heuristics.
- `max-autotune`: explicitly searches more configurations and algorithms.
- `max-autotune-no-cudagraphs`: same autotuning, but without CUDA Graphs.
- `reduce-overhead`: no extra autotuning; it only adds CUDA Graphs.

## Task 06 limitation

The main matmul was sent to `extern_kernels.mm`, and the log printed:

```text
Not enough SMs to use max_autotune_gemm mode
```

So on this GPU the GEMM was not autotuned; only the smaller pointwise
`triton_poi_fused_add_relu_0` kernel was tuned. That is why the performance
differences between modes were small.
