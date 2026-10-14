# Task 01: Dynamo capture, FX graph, guards, and graph breaks

## Files

- `tasks/01/dynamo_fx.py`
- `tasks/01/dynamo_fx.md`
- `tasks/01/dynamo_explain_vs_compile.md`
- `tasks/01/nested_torch_compile.md`
- `tasks/01/guards_static_shape.py`
- `tasks/01/guards_dynamic_all_args.py`
- `tasks/01/guards_mark_dynamic.py`
- `tasks/01/guards_device_dtype.py`
- `tasks/01/guards.md`

## Run

```bash
make run SCRIPT=tasks/01/dynamo_fx.py
```

For guard and graph-break logs:

```bash
TORCH_LOGS="graph_breaks,guards" make run SCRIPT=tasks/01/dynamo_fx.py
```

## Explain summary

```text
Graph Count: 1
Graph Break Count: 0
Op Count: 3
```

The traced ops are `add`, `relu`, and `mul`.

## Captured FX graph

```text
graph():
    %l_x_ : torch.Tensor [num_users=1] = placeholder[target=L_x_]
    %add : [num_users=1] = call_function[target=operator.add](args = (%l_x_, 1), kwargs = {})
    %relu : [num_users=1] = call_function[target=torch.relu](args = (%add,), kwargs = {})
    %mul : [num_users=1] = call_function[target=operator.mul](args = (%relu, 2), kwargs = {})
    return (mul,)
```

## Generated Python code

```python
def forward(self, L_x_ : torch.Tensor):
    l_x_ = L_x_
    add = l_x_ + 1;  l_x_ = None
    relu = torch.relu(add);  add = None
    mul = relu * 2;  relu = None
    return (mul,)
```

## Representative guards

- Device: `DispatchKeySet(CUDA, BackendSelect, ADInplaceOrView, AutogradCUDA)`, `device=0`
- Dtype: `torch.float32`
- Shape: `size=[8]`, `stride=[1]`

Full `TENSOR_MATCH` guard:

```text
check_tensor(L['x'], Tensor, DispatchKeySet(CUDA, BackendSelect, ADInplaceOrView,
AutogradCUDA), torch.float32, device=0, requires_grad=False, size=[8], stride=[1])
```

## Graph break

The break occurs at `tasks/01/dynamo_fx.py:36`, the `print` call inside
`graph_break_fn`.

Reason:

```text
Dynamo does not know how to trace builtin operator `print` with argument types
['str'] (has_kwargs False)
```

`print` is Python I/O rather than a tensor operation, so Dynamo ends the traced
subgraph, runs that Python code eagerly, and resumes tracing afterward.

## Check

- Traceable operations are tensor/torch operations such as `add`, `relu`, and
  `mul`.
- The generated Python code is a flat FX-graph function, not the original eager
  Python function.
- A graph break falls back to eager execution at the unsupported Python call,
  then resumes compilation afterward.
