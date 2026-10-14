# torch.compile Study Workbook

Hands-on study workbook for the NVIDIA GPU path in PyTorch:

```text
PyTorch eager -> Dynamo -> FX Graph -> AOTAutograd -> TorchInductor -> Triton -> CUDA
```

The full task plan is in [torch_compile_study_plan.md](torch_compile_study_plan.md).

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
make format-python
make precommit
make clean
```

Run a task script with Dynamo logs:

```bash
TORCH_LOGS="guards" make run SCRIPT=tasks/01/dynamo_fx.py
TORCH_LOGS="graph_breaks,guards" make run SCRIPT=tasks/01/dynamo_fx.py
```

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
