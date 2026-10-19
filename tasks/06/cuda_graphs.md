# CUDA Graphs

CUDA Graphs are a CUDA runtime feature that records a sequence of GPU
operations into a reusable graph object. After capture, the whole sequence can
be replayed with a single CUDA call instead of submitting every operation
separately from the CPU.

## Eager launches versus graph replay

Without CUDA Graphs, one call looks like:

```text
CPU -> launch matmul kernel
CPU -> launch bias/relu kernel
CPU -> wait for result
```

With CUDA Graphs:

```text
1. Start stream capture.
2. Record the same kernels/copies into a graph.
3. End capture.
4. Instantiate the graph.
5. On later calls, launch the entire graph once.
```

The recorded operations become one replayable unit.

## Why it helps

Each normal kernel launch has CPU and CUDA driver overhead. If a model contains
many small kernels, the CPU can become the bottleneck and the GPU may go idle
between launches.

CUDA Graphs remove most of that repeated launch overhead:

- one graph launch instead of many kernel launches
- less Python/CPU work per call
- useful for launch-bound workloads with many small kernels

It does not make any individual kernel compute faster. A heavy GEMM still takes
the same GPU time.

## Capture constraints

CUDA Graphs generally fix the captured state:

- input shapes
- memory addresses/pointers
- streams
- kernel launch parameters

That is why `torch.compile` normally captures after warmup with stable static
inputs. PyTorch can manage multiple graphs, or CUDA graph trees, for different
shapes and input addresses.

## Which compile modes use it

- `default`: no CUDA graphs.
- `reduce-overhead`: enables CUDA graphs to cut launch overhead.
- `max-autotune`: autotuning plus CUDA graphs.
- `max-autotune-no-cudagraphs`: autotuning only, no CUDA graphs.

Task 06 shows the expected partition difference:

```text
reduce-overhead              -> 1 cudagraphable, 0 non-cudagraphable
max-autotune                 -> 1 cudagraphable, 0 non-cudagraphable
default                      -> 0 cudagraphable, 1 non-cudagraphable
max-autotune-no-cudagraphs   -> 0 cudagraphable, 1 non-cudagraphable
```
