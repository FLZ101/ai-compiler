# Task 06 example: many small GEMMs

This example is designed to show a much larger difference between
`torch.compile` modes than a single large matmul does.

## Why it works

The function runs 64 small `64 x 64` matrix multiplies in sequence:

```python
for w in matrices:
    x = torch.relu(x @ w)
```

Each matmul is too small for one large GEMM to dominate. The CPU must launch a
large number of small kernels, so the workload is launch-bound. CUDA graphs
remove most of that per-kernel launch overhead.

## Run

```bash
make run SCRIPT=tasks/06/many_small_gemms.py
```

Write output to the ignored debug directory:

```bash
make run-log SCRIPT=tasks/06/many_small_gemms.py
```

## Expected result

On the RTX 3060 Laptop GPU, this example produces roughly:

```text
default                      compile= 3.29s     2.007 ms/iter
reduce-overhead              compile= 1.40s     0.480 ms/iter
max-autotune                 compile= 1.54s     0.476 ms/iter
max-autotune-no-cudagraphs   compile= 1.33s     2.156 ms/iter
```

`reduce-overhead` and `max-autotune` are much faster because they capture the
long sequence of small kernels into a CUDA graph.

`max-autotune-no-cudagraphs` is as slow as `default`. That isolates the
improvement to CUDA graph replay, not GEMM autotuning.
