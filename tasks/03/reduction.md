# Inductor reduction ops

Inductor represents reductions with two related IR operations:
`ops.reduction` and `ops.store_reduction`.

## `ops.reduction`

```python
reduction = ops.reduction(dtype, src_dtype, reduction_type, value)
```

Meaning:

- `dtype`: output dtype of the final result.
- `src_dtype`: accumulation dtype.
- `reduction_type`: operation such as `sum`, `max`, or `argmax`.
- `value`: scalar value produced by the loop body for this iteration.

Example:

```python
ops.reduction(torch.float32, torch.float32, 'sum', add)
```

This means: accumulate all `add` values across the reduction loop using float32
sum and produce a float32 result.

Input contract: `value` must be a scalar produced inside the reduction loop.
Passing a normal tensor/buffer violates this contract and can fail later during
shape propagation, lowering, or codegen.

## `ops.store_reduction`

```python
ops.store_reduction('buf0', index, reduction)
```

This writes the accumulated result to `buf0[index]`. It is not a normal
overwrite. Multiple reduction iterations can contribute to the same output
index.

```text
normal store:      buf[index] = value
reduction store:   buf[index] = combine(value across reduction dims)
```

Input contract:

- `name`: the target reduction output buffer.
- `index`: the flattened output offset, usually a symbolic expression such as
  `p0` or `2048*d0 + d1`.
- `value`: the result returned by `ops.reduction`, not an ordinary computed
  expression.

`store_reduction` must be used with a reduction result. Using a normal value
with it would not create a regular tensor store and would produce incorrect
reduction semantics or fail during codegen.

## Scalar vs tensor result

- The full scheduler node produces a tensor buffer, for example shape `[64]`.
- The `ops.reduction(...)` expression produces a symbolic scalar for one output
  index.
- `ops.store_reduction(...)` writes that scalar into the output tensor.

## Why reduction is special

`ops.reduction` and `ops.store_reduction` are semantic IR operations, not
passive metadata. They affect scheduling, memory planning, and codegen.

Depending on the generated kernel, a reduction may become:

- a thread-local accumulator
- shared-memory reduction
- warp-level reduction
- atomic updates
