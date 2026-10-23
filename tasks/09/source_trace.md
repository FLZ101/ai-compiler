# Task 09: trace the compiler implementation in source code

This task connects the earlier examples to the installed PyTorch/Triton source.

For the end-to-end example trace, see
[execution_trace.md](execution_trace.md).

For smaller examples that isolate bytecode tracing and FX graph generation, see
[tracing_examples.md](tracing_examples.md).

## Run

```bash
make run SCRIPT=tasks/09/source_trace.py
```

The script prints the installed PyTorch package root and checks the compiler
source files used below.

## Environment

On this machine, the installed package root is:

```text
/home/fenglei/work/ai-compiler/.venv/lib/python3.13/site-packages/torch
```

PyTorch reports `2.14.1+cu130`.

## Source route

Read in this order:

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

Relevant Triton sources are under `site-packages/triton`.

## Grep results

The main Triton code generator class is:

```text
torch/_inductor/codegen/triton.py:3274: class TritonKernel(SIMDKernel[TritonCSEVariable])
```

The top-level kernel source entry point is:

```text
torch/_inductor/codegen/triton.py:8008: def codegen(self) -> str
```

Inductor lowers a matmul dot to `tl.dot`:

```text
torch/_inductor/codegen/triton.py:1616: def dot(a, b):
torch/_inductor/codegen/triton.py:1800: return f'tl.dot({a}, {b}, input_precision="{input_precision}")'
```

Scheduler fusion nodes are defined in:

```text
torch/_inductor/scheduler.py:1205: class BaseSchedulerNode
torch/_inductor/scheduler.py:2257: class SchedulerNode(BaseSchedulerNode)
torch/_inductor/scheduler.py:2725: class FusedSchedulerNode(BaseSchedulerNode)
```

## One-page route summary

1. **Dynamo**
   - `eval_frame.py` installs the frame-evaluation hook.
   - `output_graph.py` records the Python execution as FX nodes and applies
     guards.
   - `guards.py` stores the runtime conditions that make a traced graph valid.

2. **AOTAutograd**
   - `_functorch/aot_autograd.py` traces forward and backward together,
     decomposes composite ops, and functionalizes mutations.
   - The output is a joint forward/backward graph of primitive ATen ops.

3. **Inductor front end**
   - `_inductor/compile_fx.py` accepts the FX graph and drives lowering.
   - `_inductor/graph.py` converts FX/ATen ops into Inductor IR nodes such as
     pointwise, reduction, and external kernels.

4. **Scheduling and fusion**
   - `scheduler.py` builds `SchedulerNode` and `FusedSchedulerNode` objects.
   - `FusedSchedulerNode` is where multiple IR nodes share one generated kernel.
   - Fusion decisions are made here, before Triton code generation.

5. **Triton codegen**
   - `codegen/triton.py` defines `TritonKernel`.
   - `dot()` lowers `ops.dot` to a `tl.dot` expression.
   - `codegen()` emits the complete Triton source for a scheduled kernel.

6. **Runtime wrapper**
   - `codegen/wrapper.py` generates Python/C++ code that allocates buffers and
     launches the compiled Triton kernels.

## Check

- [ ] Can trace one `aten.mm` node from FX to Inductor IR.
- [ ] Can explain that `TritonKernel` emits `tl.dot` for matmul.
- [ ] Can explain that `FusedSchedulerNode` is where fusion decisions become a
      single kernel.
- [ ] Can follow one kernel from `codegen()` back to the scheduler node that
      produced it.
