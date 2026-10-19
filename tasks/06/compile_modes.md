# Task 06: Compare compile modes and autotuning

This task compares `torch.compile` modes on the same matmul + bias + ReLU
workload and connects the mode name to what Inductor changes at runtime.

## Files

- `tasks/06/compile_modes.py`
- `tasks/06/compile_modes.md`
- `tasks/06/cuda_graphs.md`
- `tasks/06/autotune.md`

## Run

```bash
make run SCRIPT=tasks/06/compile_modes.py
```

To see scheduler, Triton heuristic, and launch configuration logs:

```bash
TORCH_LOGS=+inductor make run-log SCRIPT=tasks/06/compile_modes.py
```

The script disables Inductor caches, so every mode compiles on every run.
The log is written under `tasks/06/debug/`, which is already git-ignored.

## Modes

- `default`: normal Inductor heuristics, no CUDA graphs.
- `reduce-overhead`: adds CUDA graph capture around small CPU launch work to
  reduce per-call launch overhead.
- `max-autotune`: spends extra compile time benchmarking candidate
  Triton/cuBLAS configurations and also enables CUDA graphs.
- `max-autotune-no-cudagraphs`: same autotuning as `max-autotune`, but does not
  capture CUDA graphs.

Detailed concept notes:

- `cuda_graphs.md`
- `autotune.md`

## Results

On this machine with the matmul + bias + ReLU workload, the log run produced:

```text
default                      compile=  4.07s     2.867 ms/iter
reduce-overhead              compile=  0.58s     3.080 ms/iter
max-autotune                 compile=  0.71s     2.891 ms/iter
max-autotune-no-cudagraphs   compile=  0.62s     2.691 ms/iter
```

These numbers are noisy because the workload is small enough that Python
launch overhead and GPU frequency/clock variations matter. The first compiled
mode also pays some one-time CUDA/Triton initialization cost, so `default`
shows a larger compile time than a clean repeated measurement would.

## What to look for in logs

With `TORCH_LOGS=+inductor`, search for:

- `num_warps`
- `num_stages`
- selected `triton_config`
- autotuning candidate evaluation
- CUDA graph capture/replay

The log shows the matmul stays as `extern_kernels.mm` and the bias+ReLU
pointwise kernel is `triton_poi_fused_add_relu_0`.

### CUDA graph partitions

The scheduler log records one partition per compiled graph:

| Mode | Cudagraphable | Non-cudagraphable |
| --- | --- | --- |
| `default` | 0 | 1 |
| `reduce-overhead` | 1 | 0 |
| `max-autotune` | 1 | 0 |
| `max-autotune-no-cudagraphs` | 0 | 1 |

This is the expected pattern: `reduce-overhead` and `max-autotune` capture the
whole graph with CUDA graphs, while `default` and the explicit no-CUDA-graph
mode do not.

### Triton launch configuration

For the pointwise `triton_poi_fused_add_relu_0` kernel, the selected
configurations were:

| Mode | `XBLOCK` | `num_warps` |
| --- | --- | --- |
| `default` | 1024 | 4 |
| `reduce-overhead` | 1024 | 4 |
| `max-autotune` | 1024 | 8 |
| `max-autotune-no-cudagraphs` | 512 | 4 |

The GEMM itself is not being autotuned in this environment. The log includes:

```text
Not enough SMs to use max_autotune_gemm mode
Max autotune selects from 1 choices.
```

So `max-autotune` spends its tuning effort on the pointwise epilogue, not on
the cuBLAS matmul. This is why the runtime differences are small.

## Check

- Can you explain the compile-time versus runtime tradeoff?
- Can you explain what CUDA graphs reduce?
- Can you explain when `max-autotune` helps and when `reduce-overhead` is
  enough?

### Conclusions

- `max-autotune` helps most when the kernel has meaningful launch choices,
  such as a generated Triton GEMM with several block/warp/stage candidates.
  On this 30-SM GPU, `max_autotune_gemm` is disabled, so the GEMM is already
  fixed to cuBLAS and only the epilogue is tuned.
- `reduce-overhead` helps launch-bound workloads by capturing the kernel
  sequence into a CUDA graph and replaying it with less Python/CPU overhead.
  Here the GPU GEMM dominates, so CUDA graphs do not improve wall time much.
- CUDA graphs do not make a single GEMM run faster; they remove repeated CPU
  launch overhead around the already-compiled kernels.
