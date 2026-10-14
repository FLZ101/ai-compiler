# Nested `torch.compile` calls

When `f` is compiled with `torch.compile` and calls another compiled function
`g`, Dynamo does not necessarily produce two separate compiled kernels. What
happens depends on whether Dynamo can trace through the call.

## Typical case: `g` is inlined into `f`

```python
@torch.compile
def g(x):
    return x * 2

@torch.compile
def f(x):
    y = g(x)
    return y + 1
```

When Dynamo traces `f`, it sees the call to `g`. PyTorch often recognizes that
`g` is a compiled wrapper and unwraps it to its original Python function. Dynamo
then traces `g` as part of the same FX graph. The result is one compiled graph
containing both `mul` and `add`.

That is efficient because Inductor can fuse both operations together. The
separately compiled version of `g` may not even be called.

## Fallback case: `g` acts like a black box

If Dynamo cannot trace through `g`, such as when it cannot safely unwrap the
compiled wrapper or `g` contains an unsupported operation, it triggers a graph
break.

Then `f` is compiled in pieces:

```text
compiled part 1
  -> Python call to compiled g
  -> compiled part 2
```

The inner compiled function `g` runs as its own compiled program, and `f`
cannot fuse across the call boundary.

## Effects

- Single graph when inlined: better fusion, fewer kernels, less launch overhead.
- Graph break when not inlined: more subgraphs, separate caches, and reduced
  optimization.
- Guards: if inlined, guards from `g` become part of `f`. If black-boxed, the
  outer graph may guard on the inner callable/object.
- Caching: both `f` and `g` can have their own compiled caches. Nested
  compilation can create multiple cache entries and recompilations.

## Practical guidance

Usually compile at the outermost boundary:

```python
@torch.compile
def f(x):
    y = g_plain(x)
    return y + 1
```

Use nested `torch.compile` only when you explicitly want `g` to be a reusable,
separately compiled boundary. Otherwise, it often adds wrapper complexity
without improving performance and may prevent cross-function fusion.
