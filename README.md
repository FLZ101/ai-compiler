# torch.compile Study Workbook

Hands-on study workbook for the NVIDIA GPU path in PyTorch:

```text
PyTorch eager -> Dynamo -> FX Graph -> AOTAutograd -> TorchInductor -> Triton -> CUDA
```

The full task plan is in [torch_compile_study_plan.md](torch_compile_study_plan.md).
The separate [custom_accelerator_backend_plan.md](custom_accelerator_backend_plan.md)
describes how to extend the same compiler route to a non-NVIDIA accelerator.
The [cuda_operator_optimization_plan.md](cuda_operator_optimization_plan.md)
covers hand-implementing and profiling typical operators in PyTorch, Triton,
and CUDA C++.
The [graph_optimization_plan.md](graph_optimization_plan.md) covers graph-level
optimizations such as operator fusion, DCE/CSE, layout, memory planning, and
dynamic-shape guards, with manual and `torch.compile` examples.

## Requirements

- Python 3.10 or newer
- NVIDIA GPU and CUDA toolkit
- PyTorch with CUDA support
- Triton

The Makefile detects the local CUDA major version and installs the matching
PyTorch CUDA wheel.

## Setup

```bash
make venv
make check
```

`make venv` creates `.venv` and installs PyTorch, Triton, Black, pre-commit,
and NumPy. `make check` verifies the GPU, PyTorch CUDA availability, and Triton.

## Common commands

```bash
make check
make run SCRIPT=tasks/00/baseline.py
make run SCRIPT=tasks/01/dynamo_fx.py
make run-log SCRIPT=tasks/06/compile_modes.py TORCH_LOGS=+inductor
make format-python
make precommit
make clean
```

Run a task script with Dynamo logs:

```bash
TORCH_LOGS="guards" make run SCRIPT=tasks/01/dynamo_fx.py
TORCH_LOGS="graph_breaks,guards" make run SCRIPT=tasks/01/dynamo_fx.py
```

`make run-log` runs the same script but writes stdout/stderr to that task's
git-ignored debug directory, for example `tasks/06/debug/compile_modes.log`.

## Task layout

Each task lives in a zero-padded directory and uses short script/note names:

```text
tasks/
  00/
    baseline.py
    baseline.md
  01/
    dynamo_fx.py
    dynamo_fx.md
    guards.md
```

Scripts are the executable task. The matching Markdown files record observations
and explain the related concepts.

## Current progress

- Task 00: eager vs compiled baseline
- Task 01: Dynamo FX graph, guards, and graph breaks
- Task 02: AOTAutograd forward/backward decomposition
- Task 03: Inductor artifacts, fusion nodes, and reduction IR
- Task 04: annotated generated Triton kernels
- Task 05: fusion and scheduling matrix
- Task 06: compile modes, CUDA graphs, and autotuning
- Task 07: transformer block profiling with torch/nsys/ncu
- Task 08: hand-written Triton matmul vs Inductor
- Task 09: compiler source-code trace
