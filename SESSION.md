# Session Notes

## Current session

Focus: Task 9, tracing the PyTorch compiler from Python bytecode to FX graph
generation and Inductor lowering.

### Files

- `tasks/09/source_trace.py`
- `tasks/09/source_trace.md`
- `tasks/09/trace_example.py`
- `tasks/09/execution_trace.md`
- `tasks/09/tracing_examples.py`
- `tasks/09/tracing_examples.md`

### Key concepts covered

- `torch/_dynamo/output_graph.py`:
  - `OutputGraph` owns graph construction, guards, shape environment, and
    backend invocation.
  - `SubgraphTracer` creates FX proxies/nodes from `VariableTracker`s.
- `VariableTracker`:
  - symbolic representation of Python values during Dynamo tracing.
  - tensor operations create FX proxies; Python constants remain trackers.
- Bytecode tracing:
  - CPython bytecode is stack-oriented.
  - Dynamo symbolically interprets bytecode with `InstructionTranslator`.
  - Python `if`/`for`/`while` are tracing-time control flow.
- FX graph generation:
  - tensor inputs become placeholders.
  - ATen operations become `call_function` FX nodes.
  - outputs are rebuilt as a PyTree of tensor proxies and Python constants.
- `torch._dynamo.explain`:
  - diagnostic harness, not the interpreter.
  - reports graphs, graph breaks, break reasons, operation counts, and guards.

### Example trace

`tasks/09/trace_example.py` follows:

```text
matmul_bias_relu
  -> Dynamo FX graph: mm, add, relu
  -> guards
  -> AOTAutograd
  -> Inductor IR
  -> FusedSchedulerNode
  -> Triton kernel triton_red_fused_add_mm_relu_0
  -> wrapper launch
```

Relevant logs:

- `tasks/09/debug/trace_dynamo.log`
- `tasks/09/debug/trace_aot.log`
- `tasks/09/debug/trace_inductor.log`
- `tasks/09/debug/torchinductor/model__0_inference_0.0/`

### Commands

```bash
make run SCRIPT=tasks/09/source_trace.py
make run SCRIPT=tasks/09/trace_example.py
make run SCRIPT=tasks/09/tracing_examples.py
```

### Next steps

- Commit the Task 9 files when requested.
- Continue to Task 10: numerical correctness and gradient checks.
