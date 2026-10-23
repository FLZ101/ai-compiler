# Task 09 bytecode-tracing examples

`tracing_examples.py` contains small functions that isolate Dynamo's bytecode
interpreter and FX-graph generation from Inductor kernel code.

## Run

```bash
make run SCRIPT=tasks/09/tracing_examples.py
```

The script uses:

```python
torch._dynamo.explain(fn)(*args)
```

and prints `graph_count`, `graph_break_count`, `op_count`, and each FX graph.

## What each example exercises

| Example | Tracing feature |
| --- | --- |
| `arithmetic` | binary operators become FX arithmetic/ATen nodes |
| `matmul_bias_relu` | `aten.mm`, `aten.add`, `aten.relu` node sequence |
| `static_control` | Python `if` follows the known branch and traces only that path |
| `range_loop` | Python `range` is unrolled into repeated graph nodes |
| `tuple_unpack` | tuple/list/split stay as trackers; tensor add becomes one node |
| `nested_call` | nested Python function call is inlined/traced |
| `non_tensor_return` | shape/dtype are Python outputs while `x + 1` is a tensor output |
| `data_dependent_control` | tensor-valued condition causes a graph break |

## Expected observations

For `matmul_bias_relu`, the readable FX graph is:

```python
def forward(self, L_x_, L_w_, L_b_):
    matmul = L_x_ @ L_w_
    add = matmul + L_b_
    relu = torch.relu(add)
    return (relu,)
```

For `static_control(flag=True)`, only `x + 1` appears. With `flag=False`, only
`x - 1` appears. No `if` node exists in either graph.

For `range_loop`, the loop is unrolled into three add operations. The graph does
not contain a Python `for` node.

For `data_dependent_control`, `explain` reports a graph break because the
condition depends on `x.sum() > 0`.

## How to inspect bytecode

Add `dis.dis(fn)` to any example. The script already prints `arithmetic`:

```text
LOAD_FAST  x
LOAD_FAST  y
BINARY_OP  +
...
```

Compare the instructions with the printed FX graph to see how each stack
operation becomes a graph node.
