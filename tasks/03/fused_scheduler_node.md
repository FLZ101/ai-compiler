# FusedSchedulerNode and scheduler stages

This note uses `tasks/03/fusion_example.py`:

```python
y = x + 1
s = y.sum(dim=1)
return s * 2
```

The interesting part is that Inductor treats the `x + 1` work as part of the
reduction node, then fuses the later `* 2` pointwise node into that same node.

## Before fusion

`ir_pre_fusion.txt` shows two separate `SchedulerNode`s:

```text
op0: SchedulerNode(ComputedBuffer)   # sum(x + 1)
op1: SchedulerNode(ComputedBuffer)   # result * 2
```

Their data flow is:

```text
arg0_1[64, 32] -> op0 -> buf0[64] -> op1 -> buf1[64]
```

`op0` is the reduction node. Its loop body contains the add and the reduction
together:

```python
load = ops.load('arg0_1', get_index)
add = ops.add(load, ops.constant(1.0, torch.float32))
reduction = ops.reduction(torch.float32, torch.float32, 'sum', add)
store_reduction = ops.store_reduction('buf0', get_index_1, reduction)
```

`op1` is the pointwise node:

```python
load = ops.load('buf0', get_index)
mul = ops.mul(load, ops.constant(2.0, torch.float32))
store = ops.store('buf1', get_index_1, mul, None)
```

So before fusion, Inductor would conceptually launch one reduction kernel to
produce `buf0`, then another pointwise kernel to read `buf0` and produce
`buf1`.

## How the nodes fuse

After fusion, `ir_post_fusion.txt` wraps both scheduler nodes:

```text
op0_op1: FusedSchedulerNode(SchedulerNode,SchedulerNode)
```

This does not remove the two original nodes. They remain as:

```text
op0_op1.snodes[0]  # op0: reduction
op0_op1.snodes[1]  # op1: pointwise multiply
```

The fused node now owns the writes of both `buf0` and `buf1`, so the scheduler
can plan them together instead of treating them as two independent launches.

In `output_code.py`, the result is one Triton kernel:

```text
triton_per_fused_add_mul_sum_0
```

The pointwise `* 2` becomes the epilogue of the reduction:

```python
tmp6 = tl.sum(tmp5, 1)[:, None]
tmp8 = tmp6 * 2
tl.store(in_out_ptr0 + x0, tmp8, xmask)
```

The generated runner also reuses storage:

```python
buf0 = empty_strided_cuda((64,), (1,), torch.float32)
buf1 = buf0; del buf0  # reuse
triton_per_fused_add_mul_sum_0.run(buf1, arg0_1, 64, 32, stream=raw_stream0)
```

So fusion produces:

- one kernel launch instead of two
- one `[64]` allocation instead of two
- a reduction kernel with the dependent pointwise op as its epilogue

## Why `y = x + 1` is not a stage itself

In Inductor's scheduler, a stage is not one Python assignment or one ATen op.
A `SchedulerNode` corresponds to a materialized output buffer, represented by
`ComputedBuffer` and ending in an `ops.store` or `ops.store_reduction`.

Here `y` never becomes such a buffer:

```python
y = x + 1
s = y.sum(dim=1)
```

`y` is consumed immediately by `sum`, so Inductor does not need to write `y` to
memory. The add is placed inside `op0_loop_body` and feeds directly into the
reduction:

```text
load arg0_1
add 1
reduction sum(add)
store_reduction buf0
```

Therefore `y` is an intermediate expression inside the reduction stage, not a
stage boundary.

The first materialized buffer is `buf0` for `s`, because:

- `s` has a different shape from `x`: `[64]` instead of `[64, 32]`.
- `s` is later consumed by `s * 2`.

The second materialized buffer is `buf1` for the final `s * 2` output.
Those two buffers create the two `SchedulerNode`s that Inductor then fuses into
`FusedSchedulerNode`.

Rule of thumb:

```text
SchedulerNode / stage = a buffer that must be written

Python/ATen intermediate = folded into the enclosing loop body unless Inductor
needs to materialize it
```
