# NVIDIA GPU `torch.compile` Study Plan

This is a hands-on workbook for learning the NVIDIA GPU route:

```text
PyTorch eager
  -> Dynamo
  -> FX Graph
  -> AOTAutograd
  -> TorchInductor
  -> Triton kernels
  -> PTX/device code
  -> runtime
```

The plan is example-driven. Each task has:

- an objective
- one or more concrete scripts
- exact commands
- artifacts to inspect
- a self-check to confirm that you learned the right thing

## How to use this plan

Create a clean workspace for the study:

```bash
mkdir -p ~/torch-compile-study
cd ~/torch-compile-study
```

Run the tasks in order. Save the output for each task:

```text
torch-compile-study/
  tasks/
    00/
      baseline.py
      baseline.md
    01/
      dynamo_fx.py
      dynamo_fx.md
    02/
      aotautograd.py
      aotautograd.md
    03/
      inductor_artifacts.py
      inductor_artifacts.md
    04/
      fusion_matrix.py
      fusion_matrix.md
    05/
      compile_modes.py
      compile_modes.md
    06/
      transformer_block.py
      transformer_block.md
    07/
      triton_handwritten.py
      triton_handwritten.md
  artifacts/
```

Each task lives in a zero-padded `tasks/NN/` directory. Scripts and notes use
short names such as `baseline.py` and `baseline.md`, rather than repeating the
task number in the filename.

Before starting, verify the environment:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
python -c "import triton; print(triton.__version__)"
```

You need:

- an NVIDIA GPU
- PyTorch 2.6 or newer with CUDA support
- Triton
- CUDA toolkit if you want to inspect `nvcc` output

Recommended optional tools:

```bash
nsys --version
ncu --version
```

## Quick glossary

| Component | What it does |
| --- | --- |
| `TorchDynamo` | Traces eager Python into an FX graph and creates guards. |
| `FX Graph` | PyTorch's structured computation graph. |
| `AOTAutograd` | Traces forward and backward together, decomposes ops, and removes mutations. |
| `TorchInductor` | Lowers the FX graph, fuses operators, schedules buffers, and generates kernels. |
| `Triton` | Block-level GPU kernel language used by Inductor for NVIDIA GPU code generation. |
| `PTX` | NVIDIA virtual assembly produced by the Triton compiler. |
| `runtime wrapper` | Python/C++ code that launches the generated kernels and manages buffers. |

## Plan structure

The plan is split into six stages:

- **Stage 1:** core route, tasks 0-9. Start here.
- **Stage 2:** advanced and production-level details, tasks 10-26. This includes:
  - tasks 10-16: correctness, benchmarking, dynamic shapes, caching, CUDA graphs, full training loops, and graph-break mitigation;
  - tasks 17-23: Inductor lowering and memory planning, backend selection, mixed precision, attention kernel selection, Triton-to-PTX/SASS inspection, and AOTInductor;
  - tasks 24-26: TorchAO GPU quantization, `torch.compile` integration, PT2E export quantization, and quantized kernel inspection.
- **Stage 3:** edge cases and advanced diagnostics, tasks 27-31. These cover in-place mutations, data-dependent control flow, structured logging, memory measurement, and custom autograd.
- **Stage 4:** advanced NVIDIA precision and training, tasks 32-36. These cover full-graph compilation, cache limits, custom Triton op integration, FP8 precision, and activation checkpointing.
- **Stage 5:** compiler APIs, configuration, and deployment, tasks 37-42. These cover Inductor tuning knobs, compiler metrics, fake tensors, AOTInductor runtime loading, backend comparison, and asynchronous compilation.
- **Stage 6:** inference optimization and advanced caching, tasks 43-46. These cover weight freezing, C++ wrapper codegen, FX graph caching, and GEMM backend candidate inspection.

Complete the stages in order.

## Recommended natural order

The sections below are grouped by topic, but the most natural learning order is:

```text
0  baseline
1  Dynamo and FX
2  AOTAutograd
3  Inductor artifacts
4  generated Triton kernel
5  fusion and scheduling
6  compile modes and autotuning
7  end-to-end profiling
8  hand-written Triton comparison
9  source-code trace

10 correctness and numerical checks
11 rigorous benchmarking
12 dynamic shapes and recompilation
13 compilation cache
14 CUDA graphs
15 full training loop
16 graph breaks and custom ops

17 Inductor lowering and memory planning
18 backend selection: Triton vs cuBLAS/CUTLASS/cuDNN
19 mixed precision and autocast
20 attention and SDPA selection
21 Triton IR to PTX
22 SASS inspection
23 AOTInductor and export

24 TorchAO int8 quantization
25 quantized model with torch.compile
26 PT2E export quantization

27 in-place mutations and functionalization
28 data-dependent control flow
29 structured logging and debugging
30 memory measurement and buffer reuse
31 custom autograd functions

32 full-graph compilation and graph-break enforcement
33 Dynamo cache limits and recompilation policy
34 custom Triton kernel op integration
35 FP8 mixed precision
36 activation checkpointing and recompute

37 Inductor configuration and tuning knobs
38 compiler metrics and counters
39 fake tensors and meta tracing
40 AOTInductor artifact loading and execution
41 backend comparison: inductor, aot_eager, eager
42 asynchronous compilation and cache management

43 inference weight freezing
44 C++ wrapper codegen
45 FX graph caching
46 GEMM backend candidate inspection
```

Use this order when you work through the tasks.

## Task-card checklist

Each task now includes its detailed walkthrough. Tasks 0-9 are fully worked examples with goals, inputs, steps, expected output, and checks. Tasks 10-46 follow the same pattern with inline scripts and expected output.

### How to read a task

Every task follows the same pattern:

- **Goal** says what you should understand.
- **Input** lists what is required. If `Input` is omitted, the required input is the environment plus the script shown in the task.
- **Steps** are the concrete commands. If `Steps` is omitted, create the shown script and run the commands under `Run`.
- **Expected output** tells you what a successful run looks like.
- **Check** is the self-assessment question.

For tasks 10-46, keep the short script and note names under their task directory,
for example `tasks/10/correctness.py`, `tasks/23/aot_inductor.py`, and
`tasks/46/gemm_candidates.py`.

### Task 0: Establish the eager-vs-compiled baseline

**Goal:** prove that `torch.compile` works on the GPU and produces a measurable baseline.

**Input:**

- an NVIDIA GPU
- PyTorch with CUDA
- `tasks/00/baseline.py`

**Steps:**

1. Create `tasks/00/baseline.py` as shown below.
2. Run `make run SCRIPT=tasks/00/baseline.py`.
3. Record the eager time, compiled time, and speedup.
4. Run the script a second time and note whether compilation was cached.

**Expected output:**

```text
eager:     12.345 ms/iter
compiled:   7.890 ms/iter
speedup:    1.56x
```

**Check:**

- The first compiled call is slower than the next calls.
- The final `speedup` is greater than `1.0x`.
- You know that `backend="inductor"` was used.


**Detailed walkthrough**


Create `tasks/00/baseline.py`:

```python
import torch

def eager_fn(x, w, b):
    return torch.relu(x @ w + b)

@torch.compile(backend="inductor")
def compiled_fn(x, w, b):
    return torch.relu(x @ w + b)

x = torch.randn(2048, 2048, device="cuda")
w = torch.randn(2048, 2048, device="cuda")
b = torch.randn(2048, device="cuda")

# Compile and warm up
for _ in range(5):
    out = compiled_fn(x, w, b)

torch.cuda.synchronize()

# Compare eager and compiled execution
def time_fn(fn, *args, iters=20):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters

eager_ms = time_fn(eager_fn, x, w, b)
compiled_ms = time_fn(compiled_fn, x, w, b)

print(f"eager:     {eager_ms:.3f} ms/iter")
print(f"compiled:  {compiled_ms:.3f} ms/iter")
print(f"speedup:   {eager_ms / compiled_ms:.2f}x")
```

### Task 1: Inspect Dynamo capture, FX graph, guards, and graph breaks

**Goal:** see the first compiler stage from eager Python to FX.

**Input:**

- `01_dynamo_fx.py`

**Steps:**

1. Run `python 01_dynamo_fx.py`.
2. Copy the printed FX graph and generated Python code into `notes/01_fx_graph.md`.
3. Run `TORCH_LOGS="graph_breaks,guards" python 01_dynamo_fx.py`.
4. Find the line in the logs where the `print` call causes a graph break.
5. Find one guard for device, one for dtype, and one for shape.

**Expected output:**

- an FX graph containing `relu`, `add`, and `mul` nodes
- generated Python that no longer contains the original eager Python control flow
- a graph break caused by `print`
- guard entries for `cuda`, `float32`, and tensor shape

**Check:**

- You can explain which Python operations are traceable.
- You can explain why a graph break falls back to eager execution.


**Detailed walkthrough**


Create `01_dynamo_fx.py`:

```python
import torch

# Example 1: explain what Dynamo sees
def fn(x):
    return torch.relu(x + 1) * 2

x = torch.randn(8, device="cuda")
explanation = torch._dynamo.explain(fn)(x)
print("=== explain ===")
print(explanation)


# Example 2: print the captured FX graph using a custom backend
def inspect_backend(gm, example_inputs):
    print("\n=== Captured FX graph ===")
    print(gm.graph)
    print("\n=== Generated Python code ===")
    print(gm.code)
    return gm.forward

@torch.compile(backend=inspect_backend)
def captured_fn(x):
    return torch.relu(x + 1) * 2

captured_fn(torch.randn(8, device="cuda"))


# Example 3: observe a graph break
@torch.compile
def graph_break_fn(x):
    y = x + 1
    print("inside compiled function")  # Python I/O causes a graph break
    return y * 2

print("\n=== graph break example ===")
graph_break_fn(torch.randn(8, device="cuda"))
```

Inspect:

- `graph_count` and `graph_break_count`
- guards for device, dtype, shape, and stride
- how `print` forces a graph break and eager fallback

### Task 2: Inspect AOTAutograd forward/backward decomposition

**Goal:** observe the step where forward and backward are traced together and ops are decomposed.

**Input:**

- `02_aotautograd.py`

**Steps:**

1. Run `python 02_aotautograd.py`.
2. Copy the forward graph and backward graph into `notes/02_aot_graphs.md`.
3. Run `TORCH_LOGS="dynamo,aot,inductor" python 02_aotautograd.py`.
4. Identify at least one decomposed operator, for example `addmm`, `mm`, `t`, or `relu`.
5. Identify where the backward computation appears.

**Expected output:**

- two printed graphs: forward and backward
- a joint representation showing how gradients are computed
- evidence that `nn.Linear` was decomposed

**Check:**

- You can describe the difference between eager autograd and AOTAutograd.
- You can name one operator decomposition.

> `aot_function` is experimental and may move between PyTorch releases. If the import changes, use `torch._functorch.aot_autograd` through the installed package's current public path or use the AOTAutograd logs shown in Task 2.


**Detailed walkthrough**


Create `02_aotautograd.py`:

```python
import torch
from torch._functorch.aot_autograd import aot_function


def print_graph(name):
    def compiler(gm, example_inputs):
        print(f"\n=== {name} graph ===")
        print(gm.graph)
        return gm.forward

    return compiler


def fn(x):
    return (x * x).sum()


x = torch.randn(4, requires_grad=True, device="cuda")

# Directly inspect the AOTAutograd forward and backward graphs.
compiled_fn = aot_function(
    fn,
    fw_compiler=print_graph("forward"),
    bw_compiler=print_graph("backward"),
)

out = compiled_fn(x)
out.backward()


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(64, 32)

    def forward(self, x):
        return torch.relu(self.fc(x)).sum()

model = Net().cuda()
x = torch.randn(16, 64, device="cuda")

compiled_model = torch.compile(model)
loss = compiled_model(x)
loss.backward()
```

Run with joint-graph logging:

```bash
TORCH_LOGS="aot_joint_graph" python 02_aotautograd.py
```

Run with broader logs:

```bash
TORCH_LOGS="dynamo,aot,inductor" python 02_aotautograd.py
```

Inspect:

- how `nn.Linear` becomes `addmm` and related decomposed ops;
- how backward operations appear in the joint graph;
- how mutations are functionalized;
- where AOTAutograd hands the graph to Inductor.

### Task 3: Capture complete Inductor artifacts

**Goal:** inspect the actual IR files generated by TorchInductor.

**Input:**

- `03_inductor_artifacts.py`

**Steps:**

1. Run `rm -rf torch_compile_debug`.
2. Run `TORCH_COMPILE_DEBUG=1 TORCH_LOGS="output_code" python 03_inductor_artifacts.py`.
3. Run `find torch_compile_debug -maxdepth 3 -type f | sort`.
4. Open `fx_graph_readable.py`, `ir_pre_fusion.txt`, `ir_post_fusion.txt`, and `output_code.py`.
5. Write one paragraph in `notes/03_inductor_artifacts.md` explaining the role of each file.

**Expected output:**

```text
torch_compile_debug/
  model__0_forward_1.0/
    debug.log
    fx_graph_readable.py
    fx_graph_runnable.py
    ir_pre_fusion.txt
    ir_post_fusion.txt
    output_code.py
```

**Check:**

- You found the FX graph file.
- You found pre-fusion and post-fusion IR.
- You found the generated runtime code.

> The debug directory name such as `model__0_forward_1.0` can vary by PyTorch version. Use `find torch_compile_debug -maxdepth 2 -type d` to locate the actual directory before running the `sed` commands.


**Detailed walkthrough**


Create `03_inductor_artifacts.py`:

```python
import torch

@torch.compile
def fn(x, w, b):
    return torch.relu(x @ w + b)

x = torch.randn(2048, 2048, device="cuda")
w = torch.randn(2048, 2048, device="cuda")
b = torch.randn(2048, device="cuda")

fn(x, w, b)
```

Run with debug artifacts and code output:

```bash
rm -rf torch_compile_debug
TORCH_COMPILE_DEBUG=1 TORCH_LOGS="output_code" python 03_inductor_artifacts.py
```

Find the debug directory:

```bash
find torch_compile_debug -maxdepth 3 -type f | sort
```

Expected files:

```text
torch_compile_debug/
  model__0_forward_1.0/
    debug.log
    fx_graph_readable.py
    fx_graph_runnable.py
    ir_pre_fusion.txt
    ir_post_fusion.txt
    output_code.py
```

Inspect them in this order:

```bash
sed -n '1,120p' torch_compile_debug/model__0_forward_1.0/fx_graph_readable.py
sed -n '1,120p' torch_compile_debug/model__0_forward_1.0/ir_pre_fusion.txt
sed -n '1,160p' torch_compile_debug/model__0_forward_1.0/ir_post_fusion.txt
sed -n '1,240p' torch_compile_debug/model__0_forward_1.0/output_code.py
```

### Task 4: Annotate the generated Triton kernel

**Goal:** read a real generated NVIDIA kernel and identify its major parts.

**Input:**

- `03_inductor_artifacts.py`
- `TORCH_LOGS="output_code"`

**Steps:**

1. Run `TORCH_LOGS="output_code" python 03_inductor_artifacts.py > artifacts/generated_code.log 2>&1`.
2. Find the `@triton.jit` function.
3. Mark every `tl.load`, `tl.dot`, and `tl.store`.
4. Find the epilogue where bias and ReLU are applied.
5. Find the block-size constants and launch configuration.

**Expected output:**

An annotated kernel containing:

- input loads
- a matrix-multiply accumulator
- output store
- fused epilogue
- `num_warps` and `num_stages`

**Check:**

- You can explain why the matmul, bias, and ReLU are in one kernel.
- You can identify where the main reduction loop happens.


**Detailed walkthrough**


Use the same matmul example from Phase 3, but redirect output to a file:

```bash
TORCH_LOGS="output_code" python 03_inductor_artifacts.py > artifacts/generated_code.log 2>&1
```

Open the generated Triton code and label:

- `@triton.jit`
- `tl.load` for global-memory reads
- `tl.dot` for the matmul
- `tl.store` for the output
- ReLU/bias epilogue
- block sizes such as `BLOCK_M`, `BLOCK_N`, `BLOCK_K`
- `num_warps` and `num_stages`
- boundary masks

Try additional examples:

```python
# pointwise only
@torch.compile
def pointwise(x):
    return torch.sin(x) + torch.cos(x)

# reduction
@torch.compile
def reduction(x):
    return x.sum(dim=1)

# normalization
@torch.compile
def layernorm(x):
    return torch.nn.functional.layer_norm(x, x.shape[-1:])
```

Run each with:

```bash
TORCH_LOGS="output_code" python example.py
```

Look for kernel name patterns:

```text
triton_poi_fused_...
triton_red_fused_...
triton_per_fused_...
```

### Task 5: Explain fusion and scheduling

**Goal:** demonstrate that different computation graphs produce different kernel counts.

**Input:**

- `04_fusion_matrix.py`

**Steps:**

1. Run `TORCH_LOGS="+fusion" python 04_fusion_matrix.py`.
2. Run `TORCH_LOGS="+inductor" python 04_fusion_matrix.py`.
3. For each example, record the number of generated kernels and their names.
4. Fill in the fusion table in `notes/05_fusion_results.md`.

**Expected output:**

A completed table similar to:

| Example | Fused? | Number of kernels | Notes |
| --- | --- | --- | --- |
| pointwise chain | yes | 1 | all pointwise ops fused |
| matmul + bias + ReLU | yes | 1 | GEMM epilogue fusion |
| attention pattern | no | multiple | reductions and separate matmuls |
| single matmul | yes | 1 | GEMM kernel |
| chained matmul | no | 2 | two dependent matmuls |

**Check:**

- You can distinguish horizontal fusion from vertical fusion.
- You can explain why attention usually requires multiple kernels.

> The exact number of kernels can vary by PyTorch version. Use the table as a guide and record what your installed version actually produces.


**Detailed walkthrough**


Create `04_fusion_matrix.py`:

```python
import torch

def example_1(x):
    return x + 1 + 2 + 3

def example_2(x, w, b):
    return torch.relu(x @ w + b)

def example_3(q, k, v):
    scores = q @ k.transpose(-2, -1)
    attn = torch.softmax(scores, dim=-1)
    return attn @ v

def example_4(a, b):
    return a @ b

def example_5(a, b, c):
    left = a @ b
    right = left @ c
    return right

x = torch.randn(2048, 2048, device="cuda")
w = torch.randn(2048, 2048, device="cuda")
b = torch.randn(2048, device="cuda")
q = torch.randn(128, 64, 128, device="cuda")
k = torch.randn(128, 64, 128, device="cuda")
v = torch.randn(128, 64, 128, device="cuda")
a = torch.randn(2048, 2048, device="cuda")
c = torch.randn(2048, 2048, device="cuda")

functions = [
    ("pointwise_chain", example_1, (x,)),
    ("matmul_bias_relu", example_2, (x, w, b)),
    ("attention_pattern", example_3, (q, k, v)),
    ("single_matmul", example_4, (a, w)),
    ("chained_matmul", example_5, (a, w, c)),
]

for name, fn, args in functions:
    compiled = torch.compile(fn)
    compiled(*args)
    print(f"compiled {name}")
```

Run with fusion logs:

```bash
TORCH_LOGS="+fusion" python 04_fusion_matrix.py
```

Run with full Inductor logs:

```bash
TORCH_LOGS="+inductor" python 04_fusion_matrix.py
```

For one selected example, capture artifacts:

```bash
rm -rf torch_compile_debug
TORCH_COMPILE_DEBUG=1 python 03_inductor_artifacts.py
```

Fill in a table:

| Example | Fused? | Number of kernels | Kernel names | Notes |
| --- | --- | --- | --- | --- |
| pointwise chain | | | | |
| matmul + bias + ReLU | | | | |
| attention pattern | | | | |
| single matmul | | | | |
| chained matmul | | | | |

### Task 6: Compare compile modes and autotuning

**Goal:** observe how Inductor's configuration choices affect runtime.

**Input:**

- `05_compile_modes.py`

**Steps:**

1. Run `python 05_compile_modes.py`.
2. Record the measured time for each mode.
3. Run `TORCH_LOGS="+inductor" python 05_compile_modes.py`.
4. Find log entries for `num_warps`, `num_stages`, and `triton_config`.
5. Write a short conclusion: when does `max-autotune` help, and when does `reduce-overhead` help?

**Expected output:**

```text
default                        6.123 ms/iter
reduce-overhead                6.101 ms/iter
max-autotune                   5.456 ms/iter
max-autotune-no-cudagraphs     5.510 ms/iter
```

**Check:**

- You can explain the tradeoff between compile time and runtime.
- You can explain what CUDA graphs reduce.


**Detailed walkthrough**


Create `05_compile_modes.py`:

```python
import torch

def fn(x, w, b):
    return torch.relu(x @ w + b)

x = torch.randn(2048, 2048, device="cuda")
w = torch.randn(2048, 2048, device="cuda")
b = torch.randn(2048, device="cuda")

variants = {
    "default": torch.compile(fn),
    "reduce-overhead": torch.compile(fn, mode="reduce-overhead"),
    "max-autotune": torch.compile(fn, mode="max-autotune"),
    "max-autotune-no-cudagraphs": torch.compile(fn, mode="max-autotune-no-cudagraphs"),
}

for name, compiled in variants.items():
    for _ in range(3):
        compiled(x, w, b)

torch.cuda.synchronize()

for name, compiled in variants.items():
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(20):
        compiled(x, w, b)
    end.record()
    torch.cuda.synchronize()

    print(f"{name:28s} {start.elapsed_time(end) / 20:.3f} ms/iter")
```

Run with Inductor logs:

```bash
TORCH_LOGS="+inductor" python 05_compile_modes.py
```

Look for:

- `num_warps`
- `num_stages`
- selected `triton_config`
- autotuning candidate evaluation
- CUDA graph usage

### Task 7: Profile a realistic transformer block

**Goal:** connect generated source to actual GPU execution and find a bottleneck.

**Input:**

- `06_transformer_block.py`
- optional `nsys` and `ncu`

**Steps:**

1. Run `python 06_transformer_block.py`.
2. Run `nsys profile --stats=true python 06_transformer_block.py` if available.
3. Run `ncu --set full python 06_transformer_block.py` for deeper kernel analysis if available.
4. Record the top three kernels by GPU time.
5. Record whether CUDA graphs reduce launch overhead.

**Expected output:**

- a list of generated kernel names such as `triton_poi_fused_...`
- a ranked list of kernel durations
- memory throughput and occupancy if `ncu` was used

**Check:**

- You can identify the bottleneck kernel.
- You can state whether the workload is compute-bound or launch-bound.


**Detailed walkthrough**


Create `06_transformer_block.py`:

```python
import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, dim=256, heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        return x

model = Block().cuda()
x = torch.randn(8, 128, 256, device="cuda")

compiled = torch.compile(model)
out = compiled(x)

for _ in range(10):
    out = compiled(x)

torch.cuda.synchronize()
```

Profile the Python program:

```bash
python 06_transformer_block.py
```

Profile with NVIDIA Nsight Systems:

```bash
nsys profile --stats=true python 06_transformer_block.py
```

Profile kernels with Nsight Compute:

```bash
ncu --set full python 06_transformer_block.py
```

Collect:

- kernel names
- kernel duration
- memory throughput
- achieved occupancy
- launch overhead
- total GPU time

### Task 8: Compare with a hand-written Triton kernel

**Goal:** verify understanding by writing the same style of kernel that Inductor generates.

**Input:**

- `07_triton_handwritten.py`

**Steps:**

1. Run `python 07_triton_handwritten.py`.
2. Run `TORCH_LOGS="output_code" python 07_triton_handwritten.py`.
3. Compare your `matmul_kernel` with the generated Inductor kernel.
4. Write a table comparing block sizes, loops, masks, and `tl.dot`.

**Expected output:**

```text
max abs diff: 0.0
```

or a very small numerical difference.

**Check:**

- You can explain the purpose of each `BLOCK_*` constant.
- You can explain why the generated kernel and your kernel have the same high-level shape.

> In `07_triton_handwritten.py`, the example assumes that `M`, `N`, and `K` are divisible by the selected block sizes. For arbitrary shapes, add load/store masks for the K loop as well as the output tile.


**Detailed walkthrough**


Create `07_triton_handwritten.py`:

```python
import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(tl.float16)
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_matmul(a, b):
    assert a.is_cuda and b.is_cuda
    M, K = a.shape
    K2, N = b.shape
    assert K == K2

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
    )

    matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=16,
        BLOCK_N=16,
        BLOCK_K=16,
        num_warps=4,
    )
    return c


@torch.compile
def inductor_matmul(a, b):
    return a @ b

a = torch.randn(512, 512, device="cuda", dtype=torch.float16)
b = torch.randn(512, 512, device="cuda", dtype=torch.float16)

out_triton = triton_matmul(a, b)
out_inductor = inductor_matmul(a, b)

print("max abs diff:", (out_triton - out_inductor).abs().max().item())
```

Generate Inductor's matmul kernel for comparison:

```bash
TORCH_LOGS="output_code" python 07_triton_handwritten.py
```

Compare:

- block sizes
- loop structure
- masks
- accumulator initialization
- shared-memory behavior
- `tl.dot` use

### Task 9: Trace the implementation in source code

**Goal:** connect the examples to the real PyTorch and Triton source.

**Input:**

- installed PyTorch source files

**Steps:**

1. Find the PyTorch package path.
2. Read `torch/_dynamo/output_graph.py` and `guards.py`.
3. Read `torch/_functorch/aot_autograd.py`.
4. Read `torch/_inductor/graph.py`, `scheduler.py`, and `codegen/triton.py`.
5. Use `rg` to locate the `TritonKernel` class and `tl.dot` code generation.
6. Write one page explaining the route in your own words.

**Expected output:**

- source-file notes
- grep results showing the relevant functions
- a one-page summary of the complete route

**Check:**

- You can follow one matmul node from FX to Inductor IR to Triton source.
- You know where fusion decisions are made.

> For editable or source installs, the paths may differ. Use `python -c "import torch; print(torch.__file__)"` to locate the actual package root.


**Detailed walkthrough**


Find your PyTorch install path:

```bash
python -c "import torch, os; print(os.path.dirname(torch.__file__))"
```

Read these files in order:

```text
torch/_dynamo/output_graph.py
torch/_dynamo/guards.py
torch/_dynamo/eval_frame.py

torch/_functorch/aot_autograd.py

torch/_inductor/compile_fx.py
torch/_inductor/graph.py
torch/_inductor/scheduler.py
torch/_inductor/codegen/triton.py
torch/_inductor/codegen/wrapper.py
```

Triton source:

```text
python/triton/compiler/
python/triton/language/
third_party/nvidia/backend/
```

Useful search commands:

```bash
rg "class TritonKernel" torch/_inductor
rg "codegen" torch/_inductor/scheduler.py
rg "def codegen" torch/_inductor/codegen/triton.py
rg "tl.dot" torch/_inductor
```

## Stage 2: Advanced and production-level exercises

These tasks fill the remaining parts of the route. They are more complex or require a deeper compiler-system view, so run them after completing Stage 1.

### Task 10: Verify numerical correctness

**Goal:** prove that compiled output and gradients are close to eager execution.

Create `10_correctness.py`:

```python
import torch

def fn(x, w, b):
    return torch.relu(x @ w + b)

for dtype in (torch.float32, torch.float16, torch.bfloat16):
    x = torch.randn(256, 256, device="cuda", dtype=dtype)
    w = torch.randn(256, 256, device="cuda", dtype=dtype)
    b = torch.randn(256, device="cuda", dtype=dtype)

    compiled = torch.compile(fn)
    out_eager = fn(x, w, b)
    out_compiled = compiled(x, w, b)

    torch.testing.assert_close(
        out_compiled,
        out_eager,
        rtol=1e-2 if dtype != torch.float32 else 1e-5,
        atol=1e-2 if dtype != torch.float32 else 1e-5,
    )
    print(dtype, "output ok")
```

Run:

```bash
python 10_correctness.py
```

For gradients:

```python
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(32, 16)

    def forward(self, x):
        return torch.relu(self.fc(x)).sum()

model = Net().cuda()
x = torch.randn(8, 32, device="cuda")

eager_loss = model(x)
eager_loss.backward()
eager_grad = model.fc.weight.grad.clone()
model.zero_grad()

compiled_model = torch.compile(model)
compiled_loss = compiled_model(x)
compiled_loss.backward()
compiled_grad = model.fc.weight.grad.clone()

torch.testing.assert_close(compiled_grad, eager_grad, rtol=1e-4, atol=1e-4)
print("gradient ok")
```

**Expected output:**

```text
torch.float32 output ok
torch.float16 output ok
torch.bfloat16 output ok
gradient ok
```

**Check:**

- You know when fused kernels can change numerical order and why tolerance matters.
- You can verify both forward and backward paths.


### Task 11: Use rigorous benchmarking

**Goal:** replace informal timing with reliable measurements.

Create `11_benchmark.py`:

```python
import torch
from torch.utils.benchmark import Timer

def fn(x, w, b):
    return torch.relu(x @ w + b)

x = torch.randn(2048, 2048, device="cuda")
w = torch.randn(2048, 2048, device="cuda")
b = torch.randn(2048, device="cuda")

eager = fn
compiled = torch.compile(fn)

for _ in range(10):
    compiled(x, w, b)

torch.cuda.synchronize()

for name, callable_fn in (("eager", eager), ("compiled", compiled)):
    timer = Timer(
        stmt="callable_fn(x, w, b)",
        globals={"callable_fn": callable_fn, "x": x, "w": w, "b": b},
    )
    result = timer.blocked_autorange(min_run_time=1.0)
    print(name, result.median * 1e3, "ms")
```

Run:

```bash
python 11_benchmark.py
```

**Expected output:**

Median execution time with warmup and multiple repetitions.

**Check:**

- You separate compilation warmup from measured execution.
- You report median time instead of a single measurement.
- You understand that CPU-side overhead can contaminate small-kernel measurements.


### Task 12: Control dynamic shapes and recompilation

**Goal:** learn how shape specialization and guards affect recompilation.

**Input:**

- `10_dynamic_shapes.py`

Create:

```python
import torch
from torch._dynamo import mark_dynamic


@torch.compile
def static_fn(x):
    return x.sum(dim=0)


@torch.compile(dynamic=True)
def dynamic_fn(x):
    return x.sum(dim=0)


def run(name, fn):
    print(f"\n=== {name} ===")
    fn(torch.randn(4, 16, device="cuda"))
    fn(torch.randn(8, 16, device="cuda"))
    fn(torch.randn(16, 16, device="cuda"))


run("static", static_fn)
run("dynamic", dynamic_fn)

# Also demonstrate explicit dynamic marking.
x = torch.randn(4, 16, device="cuda")
mark_dynamic(x, 0)
dynamic_fn(x)
dynamic_fn(torch.randn(12, 16, device="cuda"))
```

Run:

```bash
TORCH_LOGS="guards" python 12_dynamic_shapes.py
```

**Expected output:**

- `static_fn` recompiles when the first dimension changes.
- `dynamic_fn` avoids those recompiles.
- Guard logs show shape guards and dynamic dimension guards.

**Check:**

- You can explain the difference between a static shape guard and a dynamic shape guard.
- You know when to use `dynamic=True` and `mark_dynamic`.

> `dynamic=True` enables dynamic shape support for the compiled function, while `mark_dynamic` marks a specific dimension on a specific tensor. They are complementary: `dynamic=True` is broader, `mark_dynamic` is more targeted.


### Task 13: Inspect compilation cache and cache keys

**Goal:** understand when Inductor reuses compiled artifacts and when it rebuilds them.

**Input:**

- a small matmul example

**Steps:**

1. Use a separate cache directory.
2. Compile and run a matmul.
3. Count generated files.
4. Change the input shape or dtype.
5. Count generated files again.

```bash
rm -rf /tmp/torchinductor_study
TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_study python 03_inductor_artifacts.py
find /tmp/torchinductor_study -type f | wc -l
```

Create `13_changed_shape.py`:

```python
import torch

@torch.compile
def fn(x, w, b):
    return torch.relu(x @ w + b)

x = torch.randn(1024, 1024, device="cuda")
w = torch.randn(1024, 1024, device="cuda")
b = torch.randn(1024, device="cuda")
fn(x, w, b)
```

Change the shape from `1024` to `2048` and run again:

```bash
TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_study python 13_changed_shape.py
find /tmp/torchinductor_study -type f | wc -l
```

Run guard logs:

```bash
TORCH_LOGS="guards" python 13_changed_shape.py
```

**Expected output:**

- A new cache entry appears after a shape or dtype change.
- Guard logs explain which specialization changed.

**Check:**

- You can list the inputs to the cache key: code, guards, shapes, dtypes, device, and Inductor config.
- You know how to clear or isolate the cache.


### Task 14: Demonstrate CUDA graphs explicitly

**Goal:** observe how `reduce-overhead` uses CUDA graphs to reduce kernel-launch overhead.

Create `14_cuda_graphs.py`:

```python
import torch

def fn(x):
    return torch.relu(x + 1) * 2

x = torch.randn(1024, 1024, device="cuda")

default = torch.compile(fn)
reduce_overhead = torch.compile(fn, mode="reduce-overhead")

for _ in range(20):
    default(x)
    reduce_overhead(x)

torch.cuda.synchronize()

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
    for _ in range(100):
        default(x)
    for _ in range(100):
        reduce_overhead(x)

torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
```

Run:

```bash
python 14_cuda_graphs.py
```

Optionally use Nsight Systems:

```bash
nsys profile --stats=true python 14_cuda_graphs.py
```

**Expected output:**

- `reduce-overhead` shows fewer CPU launch events and/or CUDA graph replay.
- For a pointwise function, launch overhead is a larger fraction of total time.

**Check:**

- You can explain the difference between kernel execution time and kernel launch overhead.
- You can explain when CUDA graphs help and when they do not.

> `torch.profiler` may not always label CUDA graph replay clearly. Use `nsys` if you need to distinguish individual launches from graph replay.

### Task 15: Run a complete training loop

**Goal:** observe compiled forward and backward graphs over multiple optimizer steps.

Create `15_training_loop.py`:

```python
import torch

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(64, 32)

    def forward(self, x):
        return torch.relu(self.fc(x)).sum()

model = Net().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
x = torch.randn(16, 64, device="cuda")

compiled_model = torch.compile(model)

for step in range(5):
    optimizer.zero_grad()
    loss = compiled_model(x)
    loss.backward()
    optimizer.step()
    print(step, loss.item())
```

Run:

```bash
python 15_training_loop.py
```

Run with logs:

```bash
TORCH_LOGS="aot_joint_graph,inductor" python 15_training_loop.py
```

**Expected output:**

- A loss value for each step.
- Compiled forward and backward graphs reused across steps.

**Check:**

- You understand that parameters are updated outside the compiled forward/backward graph.
- You can explain why recompilation should not occur every step for a fixed shape.


### Task 16: Diagnose and fix graph breaks with custom ops

**Goal:** move from observing graph breaks to removing them.

Start with a graph-breaking function:

```python
import torch

@torch.compile
def bad(x):
    y = x + 1
    print("Python side effect")
    return y * 2
```

Run:

```bash
TORCH_LOGS="graph_breaks" python 16_graph_break.py
```

Then create a traceable custom operator:

```python
import torch

@torch.library.custom_op("mylib::mul2", mutates_args=())
def mul2(x: torch.Tensor) -> torch.Tensor:
    return x * 2

@mul2.register_fake
def _(x):
    return torch.empty_like(x)

@torch.compile
def fixed(x):
    return torch.ops.mylib.mul2(x) + 1

x = torch.randn(8, device="cuda")
print(fixed(x))
```

Run again:

```bash
TORCH_LOGS="graph_breaks" python 16_custom_op_fixed.py
```

**Expected output:**

- The original function produces a graph break.
- The custom-op version does not break the graph.

**Check:**

- You can explain why arbitrary Python code breaks tracing.
- You can register a custom op and make it visible to `torch.compile`.


## Stage 2 continued: Inductor and NVIDIA backend internals

### Task 17: Inspect Inductor lowering and memory planning

**Goal:** understand buffer creation, reuse, and elimination before code generation.

Create `17_lowering_memory.py`:

```python
import torch

@torch.compile
def pointwise(x):
    return (x + 1).relu() * 2

@torch.compile
def matmul(x, w):
    return x @ w

x = torch.randn(2048, 2048, device="cuda")
w = torch.randn(2048, 2048, device="cuda")
pointwise(x)
matmul(x, w)
```

**Steps:**

1. Set Inductor debug options:

```python
import torch._inductor.config as inductor_config

inductor_config.debug = True
inductor_config.comment_origin = True
```

2. Run with:

```bash
TORCH_COMPILE_DEBUG=1 python 17_lowering_memory.py
```

4. Open `ir_pre_fusion.txt` and `ir_post_fusion.txt`.
5. Identify `buf0`, `buf1`, and any eliminated intermediate buffers.
6. Compare the pre-fusion and post-fusion graph to see buffer fusion.

Also search the source:

```bash
rg "class Buffer" torch/_inductor
rg "allocate" torch/_inductor
rg "free" torch/_inductor
```

**Expected output:**

- A note showing which intermediate tensors disappeared after fusion.
- A list of buffer names in pre-fusion versus post-fusion IR.

**Check:**

- You can explain why eliminating an intermediate buffer improves memory bandwidth.
- You know that Inductor schedules buffers, not just operations.


### Task 18: Compare Inductor Triton codegen with cuBLAS/CUTLASS/cuDNN

**Goal:** learn that Inductor does not always generate Triton kernels; it may select vendor libraries.

Create `18_backend_selection.py`:

```python
import torch
import torch.nn.functional as F
import torch._inductor.config as inductor_config

def raw_mm(a, b):
    return a @ b

@torch.compile
def conv(x, w):
    return F.conv2d(x, w)

a = torch.randn(2048, 2048, device="cuda")
b = torch.randn(2048, 2048, device="cuda")
x = torch.randn(8, 32, 64, 64, device="cuda")
w = torch.randn(32, 32, 3, 3, device="cuda")

mm = torch.compile(raw_mm)
mm(a, b)
conv(x, w)

inductor_config.max_autotune = True
inductor_config.max_autotune_gemm = True

mm_auto = torch.compile(raw_mm)
mm_auto(a, b)
```

Run:

```bash
TORCH_LOGS="output_code" python 18_backend_selection.py
```

Run with scheduler logs:

```bash
TORCH_LOGS="+inductor" python 18_backend_selection.py
```

**Expected output:**

Look for these patterns:

```text
triton_...
extern_kernels...
cutlass...
cublas...
cudnn...
```

**Check:**

- You can explain when Inductor emits Triton source versus calling cuBLAS or cuDNN.
- You can explain why `max-autotune` may benchmark CUTLASS templates.


### Task 19: Understand mixed precision and `torch.autocast`

**Goal:** observe how `torch.compile` handles FP16/BF16 and how precision changes generated kernels.

Create `19_autocast.py`:

```python
import torch

@torch.compile
def fn(x, w, b):
    return torch.relu(x @ w + b)

x = torch.randn(2048, 2048, device="cuda")
w = torch.randn(2048, 2048, device="cuda")
b = torch.randn(2048, device="cuda")

for dtype in (torch.float16, torch.bfloat16):
    print(f"\n=== {dtype} ===")
    with torch.autocast("cuda", dtype=dtype):
        out = fn(x, w, b)
    print(out.dtype, out.shape)
```

Run:

```bash
TORCH_LOGS="output_code" python 19_autocast.py
```

Also try:

```python
torch.set_float32_matmul_precision("high")
torch.set_float32_matmul_precision("highest")
```

**Expected output:**

- Generated Triton kernels show different input/output dtypes.
- Autocast changes which kernels or GEMM libraries are used.

**Check:**

- You can explain how `torch.autocast` affects the graph seen by Inductor.
- You know why BF16 and FP16 may choose different code paths.

> If autocast does not change the generated dtype, create inputs explicitly in `torch.float16` or `torch.bfloat16` to make the precision change visible.

### Task 20: Inspect attention and SDPA kernel selection

**Goal:** understand how a transformer attention step is lowered on NVIDIA GPUs.

Create `20_attention.py`:

```python
import torch
import torch.nn.functional as F

@torch.compile
def sdpa(q, k, v):
    return F.scaled_dot_product_attention(q, k, v)

q = torch.randn(8, 8, 512, 64, device="cuda", dtype=torch.float16)
k = torch.randn(8, 8, 512, 64, device="cuda", dtype=torch.float16)
v = torch.randn(8, 8, 512, 64, device="cuda", dtype=torch.float16)

out = sdpa(q, k, v)
```

Run:

```bash
TORCH_LOGS="output_code" python 20_attention.py
```

Also run:

```bash
TORCH_LOGS="+inductor" python 20_attention.py
```

Compare with `nn.MultiheadAttention` from Task 7.

**Expected output:**

- You can see whether Inductor generates a Triton attention kernel, uses an `aten` SDPA backend, or calls a vendor kernel.
- Kernel names or log lines reveal the selected attention implementation.

**Check:**

- You can explain why FlashAttention-style fused attention is preferred over separate matmul-softmax-matmul kernels.
- You can identify when the compiled route falls back to a less fused path.


### Task 21: Trace Triton down to PTX

**Goal:** follow a generated Triton kernel through the Triton compiler to device code.

For reliable IR files, use the hand-written Triton matmul. Inductor-generated Triton may already be cached and may not print every IR stage with these environment variables.

```bash
rm -f *.ttir *.ttgir *.llir *.ptx
TRITON_DUMP_IR=1 python 07_triton_handwritten.py
ls -1 *.ttir *.ttgir *.llir *.ptx
```

Also try:

```bash
TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 python 07_triton_handwritten.py
```

**Expected output:**

Files or printed IR such as:

```text
matmul.ttir
matmul.ttgir
matmul.llir
matmul.ptx
```

**Check:**

- You can explain the role of TTIR, TTGIR, LLVM IR, and PTX.
- You can find where block-level layout is lowered to warp-level and thread-level operations.


### Task 22: Inspect SASS with CUDA binary tools

**Goal:** go one level lower than PTX and inspect actual NVIDIA machine code.

**Steps:**

1. Generate a matmul kernel and locate its compiled artifact.
2. Use:

```bash
find /tmp/torchinductor* -type f \( -name '*.cubin' -o -name '*.so' \) 2>/dev/null
```

3. Disassemble a cubin:

```bash
cuobjdump --dump-sass <path_to_cubin>
```

or:

```bash
nvdisasm -c <path_to_cubin>
```

4. In Nsight Compute, open the kernel and switch to the source or SASS view.

**Expected output:**

- NVIDIA SASS instructions for the generated kernel.
- Evidence of memory, tensor-core, or FFMA instructions.

**Check:**

- You can distinguish PTX from SASS.
- You can identify tensor-core-related instructions in a matmul kernel.

> If `find` does not return a `.cubin`, look for a generated `.so` and disassemble the relevant kernel inside it, or use Nsight Compute's SASS view directly.

### Task 23: Export and compile ahead of time with AOTInductor

**Goal:** learn the production/deployment route that bypasses Dynamo tracing at runtime.

Create `23_aot_inductor.py`:

```python
import torch
from torch.export import Dim

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(64, 32)

    def forward(self, x):
        return torch.relu(self.fc(x))

model = Net().cuda().eval()
x = torch.randn(1, 64, device="cuda")

exported = torch.export.export(
    model,
    (x,),
    dynamic_shapes={"x": {0: Dim("batch", min=1, max=1024)}},
)

print(exported.graph_module.graph)

path = torch._export.aot_compile(
    exported.module(),
    (x,),
)
print("AOTInductor artifact:", path)
```

Run:

```bash
python 23_aot_inductor.py
```

The script prints the generated artifact path. Inspect that path directly, for example:

```bash
ls -lh <path_printed_by_script>
```

**Expected output:**

- an exported graph containing the traced model;
- an AOTInductor artifact such as a shared library or package.

**Check:**

- You can describe the difference between JIT `torch.compile` and AOT export.
- You understand why AOTInductor is better for fixed deployment graphs.

> Note: exact AOT API names can change between PyTorch releases. If `torch._export.aot_compile` is not available, check the installed PyTorch documentation for the current AOTInductor entry point.


## Stage 2 continued: Quantization

Quantization is part of the production `torch.compile` story. These tasks use TorchAO and PyTorch 2 export quantization to show how quantized models pass through Inductor.

### Task 24: Run TorchAO int8 quantization on GPU

**Goal:** quantize a small GPU model and verify that it still runs correctly.

Install TorchAO:

```bash
pip install torchao
```

Create `24_torchao_int8.py`:

```python
import torch
from torchao.quantization import quantize_
from torchao.quantization.quant_api import int8_dynamic_activation_int8_weight

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(128, 256)
        self.fc2 = torch.nn.Linear(256, 128)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = Net().cuda().eval()
quantize_(model, int8_dynamic_activation_int8_weight())

x = torch.randn(8, 128, device="cuda")
with torch.no_grad():
    out = model(x)

print(out.shape, out.dtype)
```

Run:

```bash
python 24_torchao_int8.py
```

**Expected output:**

- A tensor with shape `[8, 128]`.
- The model runs on CUDA with quantized linear layers.

**Check:**

- You understand that TorchAO swaps floating-point linear layers with quantized tensor subclasses or quantized operators.
- You can list which layers were quantized.


### Task 25: Combine TorchAO quantization with `torch.compile`

**Goal:** see how Inductor lowers a quantized TorchAO model.

Create `25_torchao_compile.py`:

```python
import torch
from torchao.quantization import quantize_
from torchao.quantization.quant_api import int8_dynamic_activation_int8_weight
from torchao.quantization.utils import recommended_inductor_config_setter

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(128, 256)
        self.fc2 = torch.nn.Linear(256, 128)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = Net().cuda().eval()
quantize_(model, int8_dynamic_activation_int8_weight())
recommended_inductor_config_setter()

compiled_model = torch.compile(model, mode="max-autotune")

x = torch.randn(8, 128, device="cuda")

with torch.no_grad():
    for _ in range(5):
        out = compiled_model(x)

torch.cuda.synchronize()
print(out.shape, out.dtype)
```

Run:

```bash
TORCH_LOGS="output_code" python 25_torchao_compile.py
```

Also run:

```bash
TORCH_LOGS="+inductor" python 25_torchao_compile.py
```

**Expected output:**

- Generated kernels may include int8 loads, dequantization, and `tl.dot`.
- Inductor logs show quantized operators or tensor subclasses being handled.

**Check:**

- You can explain why quantization is often used together with `torch.compile`.
- You know that `recommended_inductor_config_setter()` tunes Inductor settings for TorchAO workflows.


### Task 26: Run PyTorch 2 export post-training quantization

**Goal:** understand the export-based PTQ path that is often preferred for deployment.

Create `26_pt2e_ptq.py`:

```python
import torch
from torch.export import export
from torchao.quantization.pt2e.quantize_pt2e import (
    prepare_pt2e,
    convert_pt2e,
)
from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (
    X86InductorQuantizer,
)

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(128, 256)
        self.fc2 = torch.nn.Linear(256, 128)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = Net().eval()
x = torch.randn(8, 128)

exported = export(model, (x,))
quantizer = X86InductorQuantizer()
quantizer.set_global(
    quantizer.get_default_quantizer(),
)

prepared = prepare_pt2e(exported, quantizer)

with torch.no_grad():
    for _ in range(5):
        prepared(torch.randn(8, 128))

quantized = convert_pt2e(prepared)
print(quantized.graph_module.graph)
```

Run:

```bash
python 26_pt2e_ptq.py
```

Then compile and inspect the quantized exported model:

```python
compiled_quantized = torch.compile(quantized)
out = compiled_quantized(x)
```

Run with logs:

```bash
TORCH_LOGS="output_code" python 26_pt2e_ptq.py
```

**Expected output:**

- An exported graph with quantization/dequantization nodes and observers removed after conversion.
- Inductor-generated code that operates on quantized tensors.

**Check:**

- You can describe the difference between `prepare_pt2e` and `convert_pt2e`.
- You know why calibration happens before conversion.

> Note: this example uses the X86 Inductor quantizer to keep the script simple. For a CUDA-specific quantizer, use the appropriate TorchAO quantizer available in your installed version, such as a CUDA/XPU quantizer or `torchao.quantization.quant_api` GPU quantization path.


## Stage 3: Edge cases and advanced diagnostics

These tasks cover important behaviors that are easy to miss when first learning `torch.compile`.

### Task 27: Inspect in-place mutations and functionalization

**Goal:** understand how AOTAutograd handles mutation and makes the graph functional.

Create `27_inplace.py`:

```python
import torch

@torch.compile
def fn(x):
    x.add_(1)
    return x

x = torch.randn(8, device="cuda")
y = fn(x)

print("same tensor object:", y is x)
print("value after call:", x)
```

Run:

```bash
TORCH_COMPILE_DEBUG=1 python 27_inplace.py
```

Inspect:

```bash
sed -n '1,160p' torch_compile_debug/model__0_forward_1.0/fx_graph_readable.py
sed -n '1,160p' torch_compile_debug/model__0_forward_1.0/ir_pre_fusion.txt
```

**Expected output:**

- The FX graph shows a functional update rather than the original Python mutation.
- Inductor IR may contain a copy or an in-place buffer write.

**Check:**

- You can explain why AOTAutograd functionalizes mutations.
- You can describe what happens to the original input tensor.

> Exact identity behavior can vary by PyTorch version. The important observation is not whether `y is x`, but whether the graph has been rewritten from an in-place mutation into a functional form.

### Task 28: Handle data-dependent control flow

**Goal:** see how `torch.compile` handles control flow that depends on tensor values.

Create `28_control_flow.py`:

```python
import torch

class Net(torch.nn.Module):
    def forward(self, x):
        return torch.cond(
            x.sum() > 0,
            lambda x: x * 2,
            lambda x: -x,
            (x,),
        )

model = Net().cuda()
compiled = torch.compile(model)

print(compiled(torch.randn(8, device="cuda")))
print(compiled(-torch.randn(8, device="cuda")))
```

Run:

```bash
TORCH_LOGS="dynamo,guards" python 28_control_flow.py
```

**Expected output:**

- Both positive and negative branches execute correctly.
- Dynamo preserves the conditional structure instead of forcing an eager fallback.

**Check:**

- You can explain why Python `if` cannot be traced directly, while `torch.cond` can.
- You can identify which branch is selected at runtime.

### Task 29: Use structured logging instead of environment variables

**Goal:** control compiler diagnostics from inside Python.

Create `29_logging.py`:

```python
import torch
import torch._logging as logging

logging.set_logs(
    dynamo=True,
    guards=True,
    inductor=True,
    output_code=True,
)

@torch.compile
def fn(x, w, b):
    return torch.relu(x @ w + b)

x = torch.randn(1024, 1024, device="cuda")
w = torch.randn(1024, 1024, device="cuda")
b = torch.randn(1024, device="cuda")

fn(x, w, b)
```

Run:

```bash
python 29_logging.py
```

**Expected output:**

- Dynamo, guard, Inductor, and generated-code messages are printed.
- The same information is available without setting `TORCH_LOGS`.

**Check:**

- You can enable and disable individual logging categories programmatically.
- You understand that `TORCH_LOGS` and `torch._logging.set_logs` control the same underlying system.

### Task 30: Measure memory allocation and buffer reuse

**Goal:** quantify memory savings from fusion and buffer reuse.

Create `30_memory.py`:

```python
import torch

def fn(x):
    y = x + 1
    z = y * 2
    return z

compiled = torch.compile(fn)
x = torch.randn(4096, 4096, device="cuda")

torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()

# Warm up compilation and record a baseline after the input is allocated.
compiled(x)
torch.cuda.synchronize()
baseline = torch.cuda.memory_allocated()

out = compiled(x)
torch.cuda.synchronize()

print("baseline bytes:", baseline)
print("current bytes:", torch.cuda.memory_allocated())
print("peak bytes:", torch.cuda.max_memory_allocated())
```

Run:

```bash
python 30_memory.py
```

Compare with an eager version and with `TORCH_COMPILE_DEBUG=1` artifacts.

**Expected output:**

- A peak memory value for the compiled function.
- Inductor IR showing whether intermediate `y` was eliminated or reused.

**Check:**

- You can explain how fusion removes intermediate buffers.
- You can measure peak memory with `max_memory_allocated()`.

### Task 31: Test a custom `torch.autograd.Function`

**Goal:** understand how custom autograd functions interact with `torch.compile`.

Create `31_custom_autograd.py`:

```python
import torch

class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return grad * (x > 0)

@torch.compile
def fn(x):
    return MyReLU.apply(x).sum()

x = torch.randn(16, device="cuda", requires_grad=True)
loss = fn(x)
loss.backward()
print(loss.item(), x.grad is not None)
```

Run:

```bash
TORCH_LOGS="graph_breaks" python 31_custom_autograd.py
```

**Expected output:**

- The custom autograd function runs and produces gradients.
- The logs show whether Dynamo captured the op or fell back to eager execution.

**Check:**

- You can explain when custom autograd functions cause graph breaks.
- You know how to wrap them with `torch.compiler.allow_in_graph` or register them as custom ops when needed.

## Stage 4: Advanced NVIDIA precision and training

These tasks cover remaining production-relevant details for NVIDIA GPUs.

### Task 32: Enforce full-graph compilation

**Goal:** understand and use `fullgraph=True` to forbid eager fallback.

Create `32_fullgraph.py`:

```python
import torch

@torch.compile(fullgraph=True)
def fn(x):
    y = x + 1
    print("this will fail fullgraph")
    return y * 2

x = torch.randn(8, device="cuda")

try:
    fn(x)
except Exception as e:
    print("fullgraph error:", type(e).__name__)
```

Then remove the `print` call and rerun.

Run:

```bash
TORCH_LOGS="graph_breaks" python 32_fullgraph.py
```

**Expected output:**

- With `print`, compilation raises an error instead of falling back to eager.
- Without `print`, the graph compiles completely.

**Check:**

- You can explain the difference between `fullgraph=False` and `fullgraph=True`.
- You know how to use `torch._dynamo.explain` to find remaining graph breaks.

### Task 33: Control Dynamo cache and recompilation policy

**Goal:** understand the compiled-graph cache and what happens when it fills.

Create `33_cache_limit.py`:

```python
import torch
import torch._dynamo.config as dynamo_config

dynamo_config.cache_size_limit = 8

@torch.compile
def fn(x):
    return x.sum(dim=0)

for size in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
    x = torch.randn(size, 16, device="cuda")
    fn(x)
```

Run:

```bash
TORCH_LOGS="dynamo,guards" python 33_cache_limit.py
```

**Expected output:**

- New shapes create new compiled graphs until the cache limit is reached.
- Logs show recompilations and cache eviction/reuse behavior.

**Check:**

- You can explain why fixed-shape programs use more cache entries.
- You can decide when `dynamic=True` is better than a large `cache_size_limit`.

### Task 34: Integrate a custom Triton kernel as a compiled op

**Goal:** call a hand-written Triton kernel from inside a `torch.compile` graph.

Create `34_custom_triton_op.py`:

```python
import torch
import triton
import triton.language as tl
from torch.library import custom_op

@custom_op("mylib::triton_mul2", mutates_args=())
def triton_mul2(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)

    @triton.jit
    def kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        val = tl.load(x_ptr + offs, mask=mask)
        tl.store(y_ptr + offs, val * 2, mask=mask)

    n = x.numel()
    grid = (triton.cdiv(n, 1024),)
    kernel[grid](x, out, n, BLOCK=1024)
    return out

@triton_mul2.register_fake
def _(x):
    return torch.empty_like(x)

@torch.compile
def fn(x):
    return torch.ops.mylib.triton_mul2(x) + 1

x = torch.randn(4096, device="cuda")
print(fn(x))
```

Run:

```bash
TORCH_LOGS="output_code" python 34_custom_triton_op.py
```

**Expected output:**

- The custom Triton kernel is called inside the compiled graph.
- Generated wrapper code shows the custom op invocation.

**Check:**

- You understand the role of `register_fake` for tracing.
- You can integrate custom Triton kernels without causing graph breaks.

> `torch.library.custom_op` is available in recent PyTorch releases. On older versions, use `torch.library.impl` and register a fake implementation manually.

### Task 35: Test FP8 mixed precision

**Goal:** observe how `torch.compile` handles FP8 tensor types.

Create `35_fp8.py`:

```python
import torch

@torch.compile
def fn(x, w):
    return x @ w

x = torch.randn(1024, 1024, device="cuda", dtype=torch.float8_e4m3fn)
w = torch.randn(1024, 1024, device="cuda", dtype=torch.float8_e4m3fn)

out = fn(x, w)
print(out.dtype, out.shape)
```

Run:

```bash
TORCH_LOGS="output_code" python 35_fp8.py
```

**Expected output:**

- Inductor either lowers the FP8 matmul through a supported kernel or reports that the op is not supported.
- Logs show FP8-related type handling.

**Check:**

- You can explain the difference between FP8, FP16, BF16, and INT8.
- You know that FP8 support depends on both the PyTorch version and the NVIDIA GPU generation.

### Task 36: Test activation checkpointing with `torch.compile`

**Goal:** see how recomputation interacts with compiled training.

Create `36_checkpoint.py`:

```python
import torch
from torch.utils.checkpoint import checkpoint

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(256, 256)

    def forward(self, x):
        x = checkpoint(self.fc, x, use_reentrant=False)
        return x.sum()

model = Net().cuda()
x = torch.randn(16, 256, device="cuda", requires_grad=True)

compiled = torch.compile(model)
loss = compiled(x)
loss.backward()
print(loss.item())
```

Run:

```bash
TORCH_LOGS="aot_joint_graph" python 36_checkpoint.py
```

**Expected output:**

- The forward pass runs, then backward recomputes the checkpointed activation.
- AOTAutograd logs show the recomputation region.

**Check:**

- You can explain why checkpointing trades compute for memory.
- You understand that checkpointing changes the backward graph, not the forward graph.

## Stage 5: Compiler APIs, configuration, and deployment

These tasks cover the operational and programmatic side of the Inductor route.

### Task 37: Inspect and tune Inductor configuration

**Goal:** understand the most important `torch._inductor.config` knobs.

Create `37_inductor_config.py`:

```python
import torch
import torch._inductor.config as inductor_config

print("max_autotune:", inductor_config.max_autotune)
print("max_autotune_gemm:", inductor_config.max_autotune_gemm)
print("triton.cudagraphs:", inductor_config.triton.cudagraphs)
print("coordinate_descent_tuning:", inductor_config.coordinate_descent_tuning)
print("cpp_wrapper:", inductor_config.cpp_wrapper)

inductor_config.max_autotune = True
inductor_config.max_autotune_gemm = True
inductor_config.coordinate_descent_tuning = True

@torch.compile
def fn(x, w):
    return x @ w

x = torch.randn(1024, 1024, device="cuda")
w = torch.randn(1024, 1024, device="cuda")
fn(x, w)
```

Run:

```bash
TORCH_LOGS="+inductor" python 37_inductor_config.py
```

**Expected output:**

- Current Inductor configuration values.
- Additional autotuning activity after enabling `max_autotune` and coordinate descent.

**Check:**

- You can explain what `max_autotune` and `coordinate_descent_tuning` do.
- You know where to find more Inductor config options in `torch/_inductor/config.py`.

### Task 38: Read compiler metrics and counters

**Goal:** measure compilation behavior instead of only observing logs.

Create `38_metrics.py`:

```python
import torch
from torch._dynamo.utils import counters

@torch.compile
def fn(x, w, b):
    return torch.relu(x @ w + b)

x = torch.randn(1024, 1024, device="cuda")
w = torch.randn(1024, 1024, device="cuda")
b = torch.randn(1024, device="cuda")

for _ in range(3):
    fn(x, w, b)

print(counters)
```

Run:

```bash
python 38_metrics.py
```

**Expected output:**

- Counters such as `graph_break`, `compiled_autograd`, or cache-hit counters.
- A Python dictionary showing what Dynamo/Inductor recorded.

**Check:**

- You can use counters to detect unexpected graph breaks.
- You understand that counters are useful for automated regression checks.

> Counter names vary by PyTorch version. If a specific key is missing, print the full dictionary and look for analogous keys related to graph breaks, autograd, or cache hits.

### Task 39: Use fake tensors and meta tracing

**Goal:** understand how PyTorch traces without allocating real GPU tensors.

Create `39_fake_tensors.py`:

```python
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

def fn(x, w):
    return x @ w

x = torch.randn(4, 8, device="meta")
w = torch.randn(8, 16, device="meta")

with FakeTensorMode():
    out = fn(x, w)
    print(out.shape, out.device, out.dtype)
```

Run:

```bash
python 39_fake_tensors.py
```

**Expected output:**

- A fake/meta output tensor without any real GPU allocation.
- Shape and dtype information.

**Check:**

- You can explain why fake tensors are used for shape propagation.
- You know that AOTAutograd and Inductor use fake tensors during tracing.

### Task 40: Load and run an AOTInductor artifact

**Goal:** move from compiling an AOT artifact to executing it.

Create `40_aot_runtime.py`:

```python
import torch
from torch.export import export

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(64, 32)

    def forward(self, x):
        return torch.relu(self.fc(x))

model = Net().cuda().eval()
x = torch.randn(1, 64, device="cuda")

exported = export(model, (x,))
path = torch._export.aot_compile(exported.module(), (x,))
print("artifact path:", path)
```

Run:

```bash
python 40_aot_runtime.py
```

Then inspect and load the artifact with the PyTorch AOT runtime API available in your installed version.

**Expected output:**

- A generated artifact path.
- Successful loading/running of the compiled artifact if your PyTorch version exposes the runtime loader.

**Check:**

- You can describe the difference between compiling an artifact and executing it.
- You know that production deployment often uses a C++ runner or the provided Python loader.

> Note: AOT runtime APIs vary by release. Check the installed PyTorch documentation for the current artifact-loading API.

### Task 41: Compare compiler backends

**Goal:** understand that `torch.compile` can use different backends.

Create `41_backends.py`:

```python
import torch

def fn(x, w, b):
    return torch.relu(x @ w + b)

x = torch.randn(1024, 1024, device="cuda")
w = torch.randn(1024, 1024, device="cuda")
b = torch.randn(1024, device="cuda")

for backend in ("eager", "aot_eager", "inductor"):
    compiled = torch.compile(fn, backend=backend)
    compiled(x, w, b)
    print(backend, "ok")
```

Run:

```bash
python 41_backends.py
```

**Expected output:**

- Each backend runs successfully.
- Inductor generates kernels; eager and aot_eager execute the traced graph without Inductor lowering.

**Check:**

- You can explain the difference between eager execution, AOTAutograd-only execution, and full Inductor lowering.
- You know when using `aot_eager` is useful for debugging.

### Task 42: Test asynchronous compilation and cache management

**Goal:** understand compilation caching and asynchronous kernel compilation behavior.

Create `42_async_cache.py`:

```python
import torch
import torch._inductor.config as inductor_config

inductor_config.async_compile = True

@torch.compile
def fn(x, w):
    return x @ w

x = torch.randn(1024, 1024, device="cuda")
w = torch.randn(1024, 1024, device="cuda")

fn(x, w)
torch.cuda.synchronize()

print("compilation finished")
```

Run:

```bash
TORCH_LOGS="+inductor" python 42_async_cache.py
```

Inspect the cache directory:

```bash
find /tmp/torchinductor_* -maxdepth 2 -type f 2>/dev/null | head -20
```

**Expected output:**

- Compilation proceeds asynchronously or in subprocesses.
- Cache directories contain generated source, object files, or compiled artifacts.

**Check:**

- You can explain why async compilation can reduce startup latency.
- You can inspect `torch._inductor.config.cache_dir`.

## Stage 6: Inference optimization and advanced caching

These tasks cover important production-inference features of the Inductor route.

### Task 43: Enable inference weight freezing

**Goal:** understand how Inductor can constant-fold frozen parameters for inference.

Create `43_freezing.py`:

```python
import torch
import torch._inductor.config as inductor_config

inductor_config.freezing = True

w = torch.randn(128, 128, device="cuda")
b = torch.randn(128, device="cuda")

@torch.compile
def fn(x):
    return torch.relu(x @ w + b)

x = torch.randn(128, 128, device="cuda")
fn(x)
```

Run:

```bash
TORCH_LOGS="output_code" python 43_freezing.py
```

**Expected output:**

- Generated code or Inductor IR shows frozen weights treated as constants.
- Inductor may fold operations on constants.

**Check:**

- You can explain why freezing helps inference.
- You understand that frozen parameters must not require gradients.

### Task 44: Generate a C++ runtime wrapper

**Goal:** observe C++ wrapper codegen instead of the default Python wrapper.

Create `44_cpp_wrapper.py`:

```python
import torch
import torch._inductor.config as inductor_config

inductor_config.cpp_wrapper = True

@torch.compile
def fn(x, w, b):
    return torch.relu(x @ w + b)

x = torch.randn(1024, 1024, device="cuda")
w = torch.randn(1024, 1024, device="cuda")
b = torch.randn(1024, device="cuda")

fn(x, w, b)
```

Run:

```bash
TORCH_COMPILE_DEBUG=1 python 44_cpp_wrapper.py
```

Inspect:

```bash
sed -n '1,240p' torch_compile_debug/model__0_forward_1.0/output_code.py
```

**Expected output:**

- The generated wrapper contains C++ code or calls into a compiled C++ extension.
- Fewer Python-side kernel launches are required.

**Check:**

- You can explain why a C++ wrapper reduces launch overhead.
- You know that `cpp_wrapper` is especially useful for many small kernels.

### Task 45: Enable FX graph caching

**Goal:** see how Inductor caches the FX graph before lowering.

Create `45_fx_cache.py`:

```python
import torch
import torch._inductor.config as inductor_config

inductor_config.fx_graph_cache = True

@torch.compile
def fn(x, w, b):
    return torch.relu(x @ w + b)

x = torch.randn(1024, 1024, device="cuda")
w = torch.randn(1024, 1024, device="cuda")
b = torch.randn(1024, device="cuda")

fn(x, w, b)
fn(x, w, b)
```

Run:

```bash
TORCH_LOGS="+inductor" python 45_fx_cache.py
```

**Expected output:**

- Logs or counters show a cache hit on the second call.
- Lowering is skipped after the first compilation.

**Check:**

- You can distinguish FX graph cache from the generated-code cache.
- You understand why caching matters for repeated compilation.

### Task 46: Inspect GEMM backend candidates

**Goal:** see which GEMM implementations Inductor considers and selects.

Create `46_gemm_candidates.py`:

```python
import torch
import torch._inductor.config as inductor_config

inductor_config.max_autotune = True
inductor_config.max_autotune_gemm = True
inductor_config.coordinate_descent_tuning = True

@torch.compile
def fn(x, w):
    return x @ w

x = torch.randn(2048, 2048, device="cuda")
w = torch.randn(2048, 2048, device="cuda")
fn(x, w)
```

Run:

```bash
TORCH_LOGS="+inductor" python 46_gemm_candidates.py
```

**Expected output:**

- Logs showing candidate GEMM backends such as Triton, CUTLASS, cuBLAS, or ATen.
- The selected backend and its measured or estimated performance.

**Check:**

- You can explain why Inductor benchmarks multiple GEMM templates.
- You know that `max_autotune_gemm` is separate from pointwise autotuning.

## Coverage checklist

Use this checklist to confirm that the plan covers every essential part of the single-GPU NVIDIA `torch.compile` route.

| Route component | Covered by |
| --- | --- |
| eager versus compiled execution | Task 0 |
| Dynamo graph capture | Task 1 |
| FX graph | Task 1 |
| guards and graph breaks | Task 1, Task 12, Task 16, Task 32 |
| AOTAutograd forward/backward | Task 2 |
| operator decomposition | Task 2 |
| Inductor pre/post-fusion IR | Task 3 |
| generated Triton kernel | Task 4 |
| fusion and scheduling | Task 5 |
| compile modes and autotuning | Task 6 |
| profiling | Task 7 |
| hand-written Triton comparison | Task 8 |
| source-code tracing | Task 9 |
| correctness and tolerance | Task 10 |
| benchmarking methodology | Task 11 |
| dynamic shapes and recompilation | Task 12 |
| compilation cache | Task 13 |
| CUDA graphs | Task 14 |
| full training loop | Task 15 |
| graph-break mitigation and custom ops | Task 16 |
| Inductor lowering and memory planning | Task 17, Task 30 |
| backend selection | Task 18, Task 46 |
| mixed precision and autocast | Task 19 |
| attention and SDPA selection | Task 20 |
| Triton IR to PTX | Task 21 |
| SASS inspection | Task 22 |
| AOTInductor and export | Task 23, Task 40 |
| TorchAO GPU quantization | Task 24 |
| quantized `torch.compile` | Task 25 |
| PT2E export quantization | Task 26 |
| in-place mutation and functionalization | Task 27 |
| data-dependent control flow | Task 28 |
| structured logging | Task 29 |
| memory allocation and buffer reuse | Task 30 |
| custom autograd functions | Task 31 |
| full-graph compilation | Task 32 |
| Dynamo cache policy | Task 33 |
| custom Triton op integration | Task 34 |
| FP8 precision | Task 35 |
| activation checkpointing | Task 36 |
| Inductor configuration | Task 37 |
| compiler metrics and counters | Task 38 |
| fake tensors and meta tracing | Task 39 |
| AOT artifact loading | Task 40 |
| backend comparison | Task 41 |
| asynchronous compilation | Task 42 |
| inference weight freezing | Task 43 |
| C++ wrapper codegen | Task 44 |
| FX graph caching | Task 45 |
| GEMM backend candidates | Task 46 |

### Definition of done

The single-GPU NVIDIA `torch.compile` route is considered covered when you can:

1. Explain the entire route from eager PyTorch to SASS.
2. Capture and inspect FX, AOTAutograd, and Inductor IR artifacts.
3. Explain and demonstrate fusion, scheduling, autotuning, and CUDA graphs.
4. Verify correctness and performance with rigorous measurements.
5. Debug graph breaks, dynamic shapes, and compilation cache behavior.
6. Inspect generated Triton, PTX, and SASS.
7. Export an AOTInductor artifact and run a quantized compiled model.
8. Tune important Inductor and Dynamo configuration options.

## Task clarity audit

This table records the clarity status of every task after review.

| Task | Status | Remaining note |
| --- | --- | --- |
| 0 | Clear | |
| 1 | Clear | |
| 2 | Clear | `aot_function` is an experimental API |
| 3 | Clear | Debug directory name may vary |
| 4 | Clear | |
| 5 | Clear | Exact kernel count may vary by PyTorch version |
| 6 | Clear | |
| 7 | Clear | |
| 8 | Clear | Assume shapes divisible by block sizes |
| 9 | Clear | Source paths differ for editable installs |
| 10 | Clear | |
| 11 | Clear | |
| 12 | Clear | `mark_dynamic` and `dynamic=True` are complementary |
| 13 | Clear | Added missing changed-shape script |
| 14 | Clear | Use `nsys` if profiler does not clearly show graph replay |
| 15 | Clear | |
| 16 | Clear | |
| 17 | Clear | Removed unclear config option |
| 18 | Clear | Backend names may vary |
| 19 | Clear | Use explicit low-precision inputs if autocast is unclear |
| 20 | Clear | |
| 21 | Clear | Use hand-written Triton for reliable IR dumps |
| 22 | Clear | Locate the generated cubin before disassembly |
| 23 | Clear | AOTInductor API varies by release |
| 24 | Clear | Requires TorchAO |
| 25 | Clear | |
| 26 | Clear | Use a CUDA quantizer for CUDA-only experiments |
| 27 | Clear | Tensor identity may vary; observe functionalization |
| 28 | Clear | `torch.cond` signature can change between releases |
| 29 | Clear | |
| 30 | Clear | Added baseline memory measurement |
| 31 | Clear | |
| 32 | Clear | |
| 33 | Clear | |
| 34 | Clear | `torch.library.custom_op` requires recent PyTorch |
| 35 | Clear | FP8 requires supported NVIDIA hardware |
| 36 | Clear | |
| 37 | Clear | |
| 38 | Clear | Counter names vary |
| 39 | Clear | |
| 40 | Clear | Artifact-loading API varies by release |
| 41 | Clear | |
| 42 | Clear | |
| 43 | Clear | |
| 44 | Clear | |
| 45 | Clear | |
| 46 | Clear | |

## Explicitly out of scope

The following are important AI-compiler topics, but they need a separate plan because they go beyond the single-GPU `torch.compile` route:

1. **Multi-GPU and distributed compilation**
   - fully sharded data parallel, FSDP
   - tensor parallelism, TP
   - pipeline parallelism, PP
   - sequence parallelism

2. **Custom accelerator backend integration**
   - non-NVIDIA accelerator lowering
   - custom ISA/codegen backends
   - vendor-specific runtime integration

3. **Vendor-specific compiler internals**
   - TensorRT
   - OpenVINO
   - other closed or vendor-specific optimization stacks

The current plan deliberately stays focused on:

```text
PyTorch eager -> Dynamo -> FX -> AOTAutograd -> Inductor -> Triton/PTX/SASS -> CUDA runtime
```

## Final workbook deliverable

For each Stage 1, Stage 2, Stage 3, Stage 4, Stage 5, and Stage 6 task, create a note containing:

- input program
- captured FX graph
- pre-fusion Inductor IR
- post-fusion Inductor IR
- generated Triton kernel
- number of kernels and launch overhead
- profiling result
- one sentence describing the key optimization

Recommended example order:

1. pointwise chain
2. matmul + bias + activation
3. reduction
4. layer normalization
5. softmax
6. multi-head attention block
7. transformer block
8. training forward/backward
9. dynamic shapes
10. default mode versus `max-autotune`

## Common issues

### Triton is missing or does not find the GPU

```bash
python -c "import triton; print(triton.__version__)"
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Compiled code is cached and does not change

Use a fresh cache:

```bash
TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_study python script.py
```

### Too much log output

Limit logs to one category:

```bash
TORCH_LOGS="output_code" python script.py
TORCH_LOGS="graph_breaks" python script.py
TORCH_LOGS="+fusion" python script.py
```

### Version-sensitive APIs

Some advanced tasks depend on APIs that change between PyTorch and TorchAO releases:

- `torch._export.aot_compile`
- `torchao.quantization.pt2e`
- `torch._logging.set_logs`
- `torch.cond`
- AOTInductor artifact-loading APIs
- FP8 availability on the installed GPU

If one of these does not match the example exactly, check the installed package documentation and adapt the import or function name.

### Dynamic-shape example recompiles unexpectedly

Run with guard logs:

```bash
TORCH_LOGS="guards" python script.py
```

Look for guards that depend on exact input sizes and think about how to make the model shape-specialization-friendly.
