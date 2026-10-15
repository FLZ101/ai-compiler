# Task 04: `tl.load` summary

`tl.load` is a block-level load: it reads many values in parallel from a tensor
of pointers and returns a value tensor with the same shape.

```python
tl.load(pointer, mask=None, other=None, eviction_policy=None)
```

In the generated matmul kernel, the A tile is loaded like this:

```python
a_tile = tl.load(
    in_ptr0 + (r0_2 + 2048 * y0),
    r0_mask,
    eviction_policy="evict_last",
    other=0.0,
)
```

## How broadcasting creates the pointer tile

The address expression is:

```python
in_ptr0 + (r0_2 + 2048 * y0)
```

The two index tensors have different shapes:

```text
y0     [YBLOCK, 1, 1]
r0_2   [1, 1, R0_BLOCK]
```

Broadcasting aligns those shapes from the last dimension backward and expands
any size-1 dimension as needed. Here the result is:

```text
[YBLOCK, 1, R0_BLOCK]
```

Before the addition, the scalar values are:

```text
y0[y, 0, 0]   = yoffset + y
r0_2[0, 0, k] = r0_offset + k
```

After broadcasting, every `(y, k)` combination sees:

```text
y0[y, 1, k]   = yoffset + y
r0_2[y, 1, k] = r0_offset + k
```

So the full address expression becomes:

```text
pointer = base + r0_2 + 2048 * y0
        = base + (r0_offset + k) + 2048 * (yoffset + y)
```

Using absolute row and column indices:

```text
row = yoffset + y
col = r0_offset + k
```

this is simply:

```text
pointer[row, col] = base + row * 2048 + col
```

The `2048` is the row stride because `A` is a `2048 x 2048` row-major matrix.
Therefore the broadcast pointer tensor is exactly the address grid for one tile:

```text
A[yoffset : yoffset + YBLOCK, r0_offset : r0_offset + R0_BLOCK]
```

For example, with `YBLOCK=2` and `R0_BLOCK=3`, the pointer grid is:

```text
[[[y=0,k=0], [y=0,k=1], [y=0,k=2]],
 [[y=1,k=0], [y=1,k=1], [y=1,k=2]]]
```

Each entry is one address. `tl.load` reads all six entries in parallel and
returns a value tensor of the same `[YBLOCK, 1, R0_BLOCK]` shape.

The middle singleton is the output-column axis `X`. The A operand does not
depend on the output column, so that axis remains size 1. Inductor drops it
after the load:

```python
a_tile = tl.reshape(a_tile, [YBLOCK, R0_BLOCK])
```

## The other `tl.load` arguments

`mask` is an optional boolean tensor with the same shape as `pointer`. Lanes
where the mask is false do not read real memory.

`other` is the fill value used for masked-out lanes. These kernels usually use
`0.0`, so boundary lanes contribute zero to a reduction or matmul.

`eviction_policy` is a cache hint:

- `"evict_last"` means the data is likely to be reused and should stay in cache
  longer.
- `"evict_first"` means the data is streaming/use-once and can be evicted
  sooner.

In the matmul example, A and B tiles are reused across output tiles, so they
use `"evict_last"`. A plain reduction that reads each input value once can use
`"evict_first"`.
