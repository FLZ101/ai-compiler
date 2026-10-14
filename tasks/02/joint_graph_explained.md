# AOTAutograd joint graph, tangents, and `expand`

Follow-up notes to `tasks/02/aotautograd.md`. They explain, with minimal
vector-add examples, why AOTAutograd builds a joint forward/backward graph and
what the `tangents_1` and `expand` pieces mean.

## Run

```bash
make run SCRIPT=tasks/02/joint_graph_vector_add_sum.py
make run SCRIPT=tasks/02/joint_graph_vector_add.py
make run SCRIPT=tasks/02/joint_graph_vector_dot.py
```

Each script enables `aot_joint_graph` logging programmatically and prints the
partitioned forward and backward graphs.

## Why a joint graph

AOTAutograd's job is to produce both the forward and the backward passes as one
compilable program, and the backward pass depends on values that only the
forward pass produces.

Eager autograd runs the forward pass while recording a tape, then replays that
tape during backward. AOTAutograd instead traces the forward and backward
together into a single **joint graph**, then splits that graph into a forward
graph and a backward graph. The joint graph is the intermediate step where the
whole training step is visible as one pure function:

```text
primals (inputs/weights), tangents (output grad)
        -> forward ops (add, sum, addmm, relu, ...)
        -> backward ops (expand, mm, where, ...)
        -> (loss, grad_inputs, grad_weights, grad_bias)
```

Concrete reasons for the joint representation:

1. **It knows the boundary.** Only by seeing forward and backward together can
   AOTAutograd decide which forward intermediates must be *saved* for backward
   versus *recomputed*. That decision happens in the partitioning step that
   follows the joint graph.
2. **Cross-boundary optimizations.** Some passes only work when forward and
   backward are in one graph, such as removing a saved activation that is
   actually unused, or fusing work across the forward/backward split.
3. **Functionalization and decomposition once.** The joint graph is where
   in-place/alias mutations are converted into pure functional ops and where
   composite ops such as `nn.Linear` are decomposed, once over the entire
   forward+backward span.
4. **Higher-order autograd.** Tracing forward+backward as a function lets
   `torch.compile` and AOTAutograd compose multiple levels of autograd, because
   each level is just another joint graph.

So the pipeline is:

```text
Dynamo FX graph
  -> AOTAutograd traces the JOINT graph (forward + backward together)
  -> functionalize + decompose
  -> partition into fw graph and bw graph
  -> hand each to Inductor
```

## Vector add with sum

```python
def fn(a, b):
    return (a + b).sum()
```

with `a, b` of shape `[3]` and `requires_grad=True`. The joint graph AOTAutograd
traces is:

```text
primals_1: a    "f32[3]"
primals_2: b    "f32[3]"
tangents_1: dL   "f32[]"     # incoming gradient of the scalar loss

add:    "f32[3]" = aten.add.Tensor(a, b)
sum_1:  "f32[]"  = aten.sum.default(add)
expand: "f32[3]" = aten.expand.default(tangents_1, [3])

return [sum_1, expand, expand]
```

One graph contains everything: the forward ops (`add`, `sum`), the backward op
(`expand`), and a single return list that mixes the loss (`sum_1`) with the two
input gradients (`expand`, `expand`).

AOTAutograd then partitions it into two graphs:

```text
=== forward ===
def forward(a, b):
    add = a + b
    return (add.sum(),)          # the loss

=== backward ===
def backward(tangents_1):
    expand = tangents_1.expand([3])
    return (expand, expand)      # grad_a, grad_b
```

The forward graph alone has no idea how to produce `grad_a` and `grad_b`, because
they depend on `tangents_1`, which only exists once the loss is differentiated.
The backward graph cannot be written without first knowing what the forward graph
computed. The joint graph is where that connection is made explicit.

### What `tangents_1` is

`tangents_1` is the incoming gradient: the derivative of the final loss with
respect to this function's output, which seeds the backward pass.

In the joint graph, inputs are split into two groups:

- **primals**: the actual forward inputs/values (`a`, `b`).
- **tangents**: the gradient values flowing backward into the graph.

For the example, the output is the scalar `sum_1`. Calling:

```python
out = compiled(a, b)   # out = (a + b).sum()
out.backward()
```

implicitly sets the incoming gradient to `1.0`:

```text
tangents_1 = dL / d(out) = 1.0
```

That `1.0` is `tangents_1`, the seed for reverse-mode autodiff. The joint graph
uses it:

```text
sum_1  = (a + b).sum()
expand = tangents_1.expand([3])   # broadcast scalar 1.0 to shape [3]
grad_a = expand
grad_b = expand
```

The result is `grad_a = grad_b = [1.0, 1.0, 1.0]`, matching:

```text
d((a+b).sum())/da_i = 1
d((a+b).sum())/db_i = 1
```

"Tangent" comes from the forward/reverse dual view used by functorch and
AOTAutograd: primals are the forward values, and tangents are the adjoint or
gradient values. If the loss were a vector instead of a scalar, `tangents_1`
would be a vector, and `backward()` would receive a matching upstream gradient.

### Why `expand`

The gradient must have the same shape as the value it is a gradient for.

The function output is a scalar:

```text
sum_1 = (a + b).sum()      # shape []
```

so the incoming gradient is also a scalar:

```text
tangents_1 = dL/d(sum_1)   # shape []
```

But the required gradients are with respect to `a` and `b`, which have shape
`[3]`. Propagating through the `sum` reduction uses:

```text
d(sum)/d(add_i) = 1  for every element i
```

so the gradient with respect to the intermediate `add` is:

```text
grad_add_i = tangents_1 * 1 = tangents_1,   for all i
```

That is exactly "the scalar tangent broadcast to shape `[3]`", which is what
`expand` does:

```text
expand = tangents_1.expand([3])
```

It turns `tangents_1` from `shape []` into `shape [3]`, with every element equal
to the scalar. Then:

```text
grad_a = expand   # d(a+b)/da = 1
grad_b = expand   # d(a+b)/db = 1
```

`expand` is a view with stride `0`, not a copy, so it is the zero-copy way of
repeating a scalar over several positions. The joint-graph log shows this as
`"f32[3][0]"`: shape `[3]`, stride `[0]`.

For vector add with sum, nothing needs to cross the forward/backward boundary:
the backward graph receives only `tangents_1` and broadcasts it to shape `[3]`.

## Vector add without sum

```python
def fn(a, b):
    return a + b
```

with `a, b` of shape `[3]` and `requires_grad=True`. The output is a vector, so
the incoming gradient is already shape `[3]` and no `expand` is needed. Joint
graph:

```text
primals_1: a    "f32[3]"
primals_2: b    "f32[3]"
tangents_1: "f32[3]"     # upstream gradient, same shape as the output

add: "f32[3]" = aten.add.Tensor(a, b)

return [add, tangents_1, tangents_1]      # output, grad_a, grad_b
```

Partitioned graphs:

```text
=== forward ===
def forward(a, b):
    return (a + b,)             # shape [3]

=== backward ===
def backward(tangents_1):
    return (tangents_1, tangents_1)   # grad_a, grad_b
```

Because the output is not a scalar, `backward()` needs an explicit gradient:

```python
out.backward(torch.ones_like(out))
```

That vector is exactly `tangents_1`, and it flows through unchanged because
`d(a+b)/da = 1` elementwise. There is no reduction, so there is no `expand`.

## Vector dot product: saving primals for backward

```python
def fn(a, b):
    return (a * b).sum()      # vector dot product
```

Joint graph:

```text
primals_1: a    "f32[3]"
primals_2: b    "f32[3]"
tangents_1: dL   "f32[]"

mul:    "f32[3]" = aten.mul.Tensor(a, b)
sum_1:  "f32[]"  = aten.sum.default(mul)
expand: "f32[3]" = aten.expand.default(tangents_1, [3])
mul_1:  "f32[3]" = aten.mul.Tensor(expand, a)
mul_2:  "f32[3]" = aten.mul.Tensor(expand, b)

return [sum_1, mul_2, mul_1]      # loss, grad_a, grad_b
```

Partitioned graphs:

```text
=== forward ===
def forward(a, b):
    mul = a * b
    return (mul.sum(), a, b)      # saves both primals for backward

=== backward ===
def backward(a, b, tangents_1):
    expand = tangents_1.expand([3])
    mul_1 = expand * a            # grad_b = d(a*b)/db = a
    mul_2 = expand * b            # grad_a = d(a*b)/da = b
    return (mul_2, mul_1)         # (grad_a, grad_b)
```

The key difference from vector add with sum is the forward return list. Because:

```text
d(a_i * b_i)/da_i = b_i
d(a_i * b_i)/db_i = a_i
```

the backward pass needs the values of `a` and `b`, not just the scalar tangent.
So the joint graph's forward half returns `(sum_1, a, b)`, explicitly carrying
both primals across the boundary into the backward half. This is the "save
forward intermediates/primals for backward" decision that only the joint graph
can make, because it sees the backward pass's needs before partitioning.
