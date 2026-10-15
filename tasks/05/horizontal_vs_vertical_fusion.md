# Horizontal vs vertical fusion

Vertical and horizontal fusion describe two different ways Inductor combines
scheduler nodes into fewer GPU kernels.

## Vertical fusion

Producer and consumer are fused so the consumer reads the producer's result
directly from registers or shared memory instead of global memory.

Example:

```python
y = x + 1
z = torch.relu(y)
```

These are dependent:

```text
x -> add -> relu -> output
```

Inductor can emit one kernel:

```text
load x
tmp = x + 1
out = relu(tmp)
store out
```

The intermediate `y` never needs its own global-memory buffer.

In Task 05, this is the `matmul + bias + ReLU` case:

```text
mm(x, w) -> +b -> relu
```

The bias+ReLU pointwise work becomes a vertical epilogue after the matmul.

## Horizontal fusion

Independent operations are combined into one kernel because they have
compatible loop shapes and can share a single program.

Example:

```python
out1 = x + 1
out2 = x * 2
```

There is no dependency between `out1` and `out2`:

```text
x -> out1
x -> out2
```

A single fused pointwise kernel can do:

```text
load x
store out1 = x + 1
store out2 = x * 2
```

That avoids two separate launches and two passes over `x`.

## Simple comparison

| | Direction | Relationship | Goal |
| --- | --- | --- | --- |
| Vertical fusion | producer -> consumer | dependent | keep intermediate in registers/shared memory |
| Horizontal fusion | side-by-side | independent | share one launch/loop and reduce memory traffic |

In Inductor's scheduler, vertical fusion is the most common and most impactful:
it eliminates whole intermediate buffers. Horizontal fusion is useful when two
independent outputs have matching shapes and can safely share a kernel.
