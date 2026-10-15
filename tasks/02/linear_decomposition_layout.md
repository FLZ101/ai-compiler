# `nn.Linear` decomposition and tensor memory layout

Notes on how a single `nn.Linear` module becomes a couple of ATen ops during
AOTAutograd tracing, why the resulting expression is shape-consistent, and how
PyTorch's row-major storage explains the `permute` view.

## `nn.Linear` forward decomposition

`nn.Linear(in_features, out_features)` stores:

- `weight` of shape `(out_features, in_features)` — e.g. `(32, 64)`
- `bias` of shape `(out_features,)` — e.g. `(32,)`

Its forward is effectively:

```python
return input @ weight.t() + bias
```

For the task-02 module `Net.forward = torch.relu(self.fc(x)).sum()`,
AOTAutograd traced:

```text
primals_1 = weight   # (32, 64)
primals_2 = bias     # (32,)
primals_3 = x        # (16, 64)

permute: (64, 32) = aten.permute.default(weight, [1, 0])   # weight.T
addmm:   (16, 32) = aten.addmm.default(bias, x, permute)   # bias + x @ weight.T
relu:    (16, 32) = aten.relu.default(addmm)
sum:     ()       = aten.sum.default(relu)
```

So the single high-level `nn.Linear` (which internally calls `F.linear`) becomes
two ops:

1. `aten.permute` transposes the weight, because `nn.Linear` stores it as
   `(out, in)` but the matmul needs `(in, out)`.
2. `aten.addmm` is a fused `bias + x @ weight.T`.

`addmm` is itself a composite op. AOTAutograd or Inductor can decompose it
further into `mm` plus a bias add, and its backward appears as `mm` for the
weight gradient plus `sum`/`view` for the bias gradient.

## Why `bias + x @ weight.T` works

Every shape in the expression lines up:

```text
weight:   (out, in)   = (32, 64)
bias:     (out,)      = (32,)
x:        (batch, in) = (16, 64)

x @ weight.T
  (16, 64) @ (64, 32) = (16, 32)     # batch x out

bias + (x @ weight.T)
  (32,) + (16, 32)    = (16, 32)     # bias broadcasts over batch
```

### The matmul needs the transpose

Matmul `A @ B` requires `A`'s last dimension to equal `B`'s first dimension:

```text
x        : (16, 64)
weight.T : (64, 32)      # transpose of (32, 64)
```

The `64` matches, so the result is `(16, 32)`. Without the transpose,
`x @ weight` would be `(16, 64) @ (32, 64)`, and `64 != 32` fails.

The weight is stored as `(32, 64)` because `weight[i]` is the row of
coefficients for output neuron `i`: there are 32 outputs, each with 64 inputs.
That layout is `(out, in)`, but the matmul wants `(in, out)`, hence `weight.T`.

### The bias broadcasts over the batch

The bias `(32,)` aligns with the matmul result's last dimension and repeats
across the 16 rows. The precise `(32,)` vs `(1, 32)` vs `(32, 1)` distinction is
covered next.

### One-row example

With a single input row `x = [x0, x1]`, weight
`[[w00, w01], [w10, w11]]`, and bias `[b0, b1]`:

```text
out[0] = b0 + x0*w00 + x1*w01
out[1] = b1 + x0*w10 + x1*w11
```

That is exactly `x @ weight.T + bias`.

## The bias shape: `(32,)` vs `(1, 32)` vs `(32, 1)`

These three are different shapes with different rank (`ndim`):

| Shape | Rank | Meaning |
| --- | --- | --- |
| `(32,)` | 1-D | a vector of 32 elements |
| `(1, 32)` | 2-D | one row, 32 columns |
| `(32, 1)` | 2-D | 32 rows, one column |

```python
a = torch.zeros(32)     # torch.Size([32])     ndim=1
b = torch.zeros(1, 32)  # torch.Size([1, 32])  ndim=2
c = torch.zeros(32, 1)  # torch.Size([32, 1])  ndim=2
```

Broadcasting right-aligns the dimensions, and a tensor with fewer dimensions
gets `1`s prepended. So:

- `(32,)` behaves like `(1, 32)` when compared against a 2-D tensor.
- `(32, 1)` behaves like a **column**, not a row.

For the linear output `(16, 32)`:

```python
(32,)   + (16, 32)   # OK  -> (16, 32)   (32,) aligns with the trailing 32
(1, 32) + (16, 32)   # OK  -> (16, 32)   (1, 32) broadcasts over the 16 rows
(32, 1) + (16, 32)   # ERROR             trailing dim is 1, then 32 vs 16
```

`(32, 1)` broadcasts correctly only against something like `(32, n)`:

```python
(32, 1) + (32, 4)    # OK -> (32, 4)
```

In the bias add, broadcasting treats `bias` as `(1, 32)`, aligns it to the last
dimension (the 32 output features), and repeats it across the 16 batch rows:

```text
(16, 32)
(   32)    -> effectively (1, 32) -> broadcast to (16, 32)
```

So `bias` is "the same 32 values for every row", but its actual shape stays
`(32,)`, not `(1, 32)`.

## `nn.Linear` backward decomposition

The backward of `nn.Linear` is the backward of `bias + x @ weight.T`, and
AOTAutograd decomposes it into a transpose + matmul (for the weight gradient)
and a sum + reshape (for the bias gradient).

Let `grad_z = dL/dz`, where `z = bias + x @ weight.T` has shape `(16, 32)`. The
three derivatives of the linear op are:

```text
grad_weight = grad_z.T @ x         (32,16) @ (16,64) -> (32, 64)
grad_bias   = sum(grad_z, dim=0)   (16,32)          -> (32,)
grad_x      = grad_z @ weight      (16,32) @ (32,64) -> (16, 64)
```

So:

- `grad_weight` is a `permute` (transpose `grad_z`) followed by an `mm` with `x`.
- `grad_bias` is a `sum` over the batch dimension (the reverse of the forward
  bias broadcast), then a `view` back to `(32,)`.
- `grad_x` is `grad_z @ weight`, but AOTAutograd returns `None` for it when
  `x.requires_grad=False`.

### The actual backward graph

With a `relu` in front, the task-02 backward graph was:

```text
def backward(self, primals_3, le, tangents_1):
    expand       = aten.expand.default(tangents_1, [16, 32])   # scalar -> (16,32)
    full_default = aten.full.default([], 0.0, ...)
    where        = aten.where.self(le, full_default, expand)   # grad_z = relu'(z)
    permute_1    = aten.permute.default(where, [1, 0])         # grad_z.T
    mm           = aten.mm.default(permute_1, primals_3)       # grad_weight
    sum_2        = aten.sum.dim_IntList(where, [0], True)      # sum over batch
    view         = aten.view.default(sum_2, [32])              # grad_bias
    return (mm, view, None)                                    # (grad_w, grad_b, grad_x)
```

Here `primals_3` is the saved input `x`, and `le` is the ReLU gate saved by the
forward graph. The `where` produces `grad_z`, then the Linear backward splits
into:

```text
permute + mm   -> grad_weight   (32, 64)
sum + view     -> grad_bias     (32,)
```

`aten.view.default` is the ATen reshape operator: it returns a tensor that
shares the same storage but with a new shape. Here it collapses `sum_2` from
`(1, 32)` (the `keepdim=True` sum kept a leading batch dimension) down to
`(32,)`, the required bias gradient shape. No data is copied; only the shape
metadata changes.

The `None` is the `grad_x` slot, skipped because `x` was not a leaf requiring
gradients.
