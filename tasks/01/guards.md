# Guards

A guard is a runtime condition Dynamo records while tracing a function. Before
reusing a compiled graph, PyTorch checks those conditions against the current
call. If every guard passes, the cached graph runs. If one fails, PyTorch
recompiles or falls back.

## Trace-to-call flow

```text
trace time  -> assumptions become guards
call time   -> guards compare current runtime state
pass        -> reuse compiled graph
fail        -> recompile
```

## Common guard categories

- Tensor metadata: device, dtype, shape, stride, `requires_grad`.
- Global state: grad mode, default device, deterministic algorithms.
- Object identity: the `torch` module or a callable did not change.
- Dynamic shape marking: whether a dimension was marked dynamic.
- Torch function mode and autograd hooks.

Example guard from the Task 1 trace:

```text
check_tensor(L['x'], Tensor, DispatchKeySet(CUDA, BackendSelect, ADInplaceOrView,
AutogradCUDA), torch.float32, device=0, requires_grad=False, size=[8], stride=[1])
```

## Why guards matter

- They prevent reusing a kernel compiled for the wrong device, dtype, or shape.
- They make compiled execution safe for changing inputs.
- Too many guard failures cause repeated recompilation and reduce speedups.

## Inspect guards

```bash
TORCH_LOGS="guards" .venv/bin/python tasks/01/dynamo_fx.py
TORCH_LOGS="guards,recompiles" .venv/bin/python tasks/01/guards_static_shape.py
```

## Demo files

- `guards_static_shape.py`: static shape specialization and recompilation.
- `guards_dynamic_all_args.py`: `dynamic=True` for one or more tensor arguments.
- `guards_mark_dynamic.py`: mark one dimension or every dimension of one tensor.
- `guards_device_dtype.py`: device and dtype guard assumptions.

Run one example:

```bash
TORCH_LOGS="guards,recompiles" .venv/bin/python tasks/01/guards_static_shape.py
```

## Dynamic guard choices

- `torch.compile(dynamic=True)`: all dimensions of all tensor arguments dynamic.
- `mark_dynamic(x, dim)`: one dimension of one tensor dynamic.
- `mark_dynamic(x, list(range(x.dim())))`: every dimension of one tensor dynamic.
- Default static: each tensor shape is specialized and guarded exactly.
