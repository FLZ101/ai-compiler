# `torch._dynamo.explain` vs `torch.compile`

Both start with TorchDynamo, the component that traces eager Python into an FX
graph.

## `torch.compile`

- Main end-to-end compiler entry point.
- Wraps a function or `nn.Module`.
- Pipeline is roughly:
  - Dynamo captures Python into an FX graph.
  - AOTAutograd traces forward/backward and decomposes ops.
  - Inductor lowers the graph and generates optimized Triton/CUDA kernels.
- Returns a compiled callable that is cached and reused.
- This is the production path used to make code faster.

```python
@torch.compile
def f(x):
    return torch.relu(x + 1) * 2
```

## `torch._dynamo.explain`

- A debugging/inspection API focused only on the Dynamo capture stage.
- Runs the same tracing engine and returns information such as:
  - graph count
  - graph break count
  - op count
  - break reasons
  - generated guards
- Helps answer: did Dynamo capture this cleanly, and if not, where did it break?

```python
explanation = torch._dynamo.explain(f)(x)
print(explanation)
```

## How they relate

- `torch.compile` performs capture, lowering, code generation, and execution.
- `torch._dynamo.explain` performs only the first part: Dynamo capture and
  analysis.
- They share TorchDynamo's bytecode tracing and guard machinery.
- `explain` is diagnostic; `torch.compile` is the real compiler path.

In Task 1:

- `explain(fn)(x)` reports graph count, graph break count, and op count.
- `@torch.compile(backend=inspect_backend)` exposes the FX graph and generated
  Python that Dynamo passed to a backend.

## Comparison

| | `torch.compile` | `torch._dynamo.explain` |
| --- | --- | --- |
| Purpose | Compile and run faster | Inspect Dynamo tracing |
| Runs full backend | Yes | No |
| Generates CUDA/Triton code | Yes | No |
| Reports graph breaks | Via logs | Directly in return value |
| API stability | Stable main API | Internal/debugging API |

`torch._dynamo.explain` lets you look inside the first stage of what
`torch.compile` does.
