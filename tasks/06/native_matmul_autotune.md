# Task 06 example: native matmul autotune isolation

This example isolates autotuning from CUDA Graphs on a generated Triton GEMM.

## Why this shape

Inductor normally sends fp32 matmul to cuBLAS. This script enables
`inductor_config.triton.native_matmul = True`, so Inductor generates a Triton
matmul kernel with tunable block sizes, warps, and stages.

The shape is `[3072, 3072] @ [3072, 2048]`. The goal is to compare:

- `default`: no CUDA Graphs and normal GEMM heuristic.
- `max-autotune-no-cudagraphs`: extra GEMM autotuning and no CUDA Graphs.

Because both modes disable CUDA Graphs, the difference isolates autotuning.

## Run

```bash
make run SCRIPT=tasks/06/native_matmul_autotune.py
```

Write output to the ignored debug directory:

```bash
make run-log SCRIPT=tasks/06/native_matmul_autotune.py
```

Compilation can take a while because `max-autotune` benchmarks multiple GEMM
configurations on the GPU.

## Observed result

```text
default                      6.707 ms/iter
reduce-overhead              6.791 ms/iter
max-autotune                 6.930 ms/iter
max-autotune-no-cudagraphs   9.455 ms/iter
```

On this RTX 3060 Laptop GPU, autotuning did **not** produce a larger positive
difference. The default Triton heuristic was already reasonable, and the
autotuned GEMM configuration was slower for this shape.

This is the honest result for this hardware: the large mode differences in this
task come from CUDA Graphs, not from `max-autotune` GEMM tuning. The 30-SM GPU
also logs `Not enough SMs to use max_autotune_gemm mode`, which limits the
search space.

## Profiling why the autotune difference is noise

Run the example with Inductor logging:

```bash
TORCH_LOGS=+inductor make run-log SCRIPT=tasks/06/native_matmul_autotune.py
```

The log shows that every mode generates the same kernel:

```text
Generating kernel code with kernel_name: triton_red_fused_mm_0
```

The selected launch configurations explain the result:

| Mode | `XBLOCK` | `YBLOCK` | `R0_BLOCK` | `num_warps` | `num_stages` |
| --- | --- | --- | --- | --- | --- |
| `default` | 64 | 128 | 32 | 8 | 4 |
| `reduce-overhead` | 64 | 128 | 32 | 8 | 4 |
| `max-autotune` | 64 | 128 | 32 | 8 | 4 |
| `max-autotune-no-cudagraphs` | 64 | 128 | 16 | 2 | 4 |

The first three modes all settle on the same GEMM config. Only
`max-autotune-no-cudagraphs` changes `R0_BLOCK` from 32 to 16 and `num_warps`
from 8 to 2.

The coordinate-descent tuner measured that changed config as faster:

```text
Improve from ... 5.904384 -> ... 5.158912, 1.145x
```

But the real 30-iteration benchmark measured it as 9.455 ms/iter. The one-shot
autotune measurement did not match steady-state execution. `max-autotune-no-cudagraphs`
also runs last in the script, after minutes of autotuning have warmed the GPU,
so its benchmark is more exposed to clock/thermal effects.

The log also shows many candidate configs being skipped:

```text
Skip config ... because of register spilling
Got exception out of resource: ... Hardware limit
```

So the useful search space is narrow. The default config is already near a
resource limit and performs well for this shape.

Conclusion: the profiling evidence says the modes are not producing a better
GEMM config in steady state. The large mode differences in this task come from
CUDA Graphs, not from native-GEMM autotuning on this 30-SM GPU.

## How the default config is determined

The `default` launch config is not one hard-coded value. From the Inductor log:

```text
CachingAutotuner gets 19 configs for triton_red_fused_mm_0
Best config for triton_red_fused_mm_0:
  XBLOCK: 64, YBLOCK: 128, R0_BLOCK: 32,
  num_warps: 8, num_ctas: 1, num_stages: 4
```

The selection follows four steps:

1. Inductor proposes candidate tile sizes from the GEMM dimensions:
   `M = 3072`, `N = 2048`, `K = 3072`.
2. It filters candidates against hardware limits such as shared memory,
   threads per block, registers, and `num_stages`.
3. `CachingAutotuner` measures the surviving candidates at compile time.
4. It keeps the fastest measured candidate.

So `default` already does a small, fast autotune. The winning config in this
example is:

```text
XBLOCK: 64, YBLOCK: 128, R0_BLOCK: 32,
num_warps: 8, num_stages: 4
```

It uses about 98,304 bytes of shared memory, close to the useful resource limit
for this shape on the RTX 3060.

## Why default is already close to autotune

Current Inductor gives `default` a compact `CachingAutotuner`, while
`max-autotune` performs additional search such as coordinate descent.

| | `default` | `max-autotune` |
| --- | --- | --- |
| Candidate count | small heuristic-generated set | larger, with refinement |
| Search method | measure initial candidates | coordinate descent around the best |
| GEMM algorithm search | limited | more aggressive when available |
| Compile time | lower | higher |

In this native matmul, both `default` and `max-autotune` settled on the same
GEMM config. Only `max-autotune-no-cudagraphs` moved to another config, and its
steady-state result was slower. That means the small default search was already
enough for this shape.

## Why max-autotune can be slower with the same config

The kernel is identical in `default` and `max-autotune`:

```text
triton_red_fused_mm_0
XBLOCK: 64, YBLOCK: 128, R0_BLOCK: 32,
num_warps: 8, num_stages: 4
```

Therefore the GPU-side GEMM work should take the same time. The small measured
gap, about 0.22 ms or 3%, comes from wrapper and measurement differences:

- `default` launches the compiled wrapper directly.
- `max-autotune` uses CUDA Graph replay.
- The modes run sequentially, so later modes can be warmer or clock-throttled.
- A one-kernel benchmark naturally has a few percent run-to-run variance.

The `max-autotune` wrapper is not making this kernel slower. If only the
`triton_red_fused_mm_0` kernel duration were profiled, both modes should show
essentially the same GPU time.
