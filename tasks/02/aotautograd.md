# Task 02: AOTAutograd forward/backward decomposition

## Files

- `tasks/02/aotautograd.py`
- `tasks/02/aotautograd.md`
- `tasks/02/chain_rule.md`
- `tasks/02/joint_graph_explained.md`
- `tasks/02/joint_graph_vector_add_sum.py`
- `tasks/02/joint_graph_vector_add.py`
- `tasks/02/joint_graph_vector_dot.py`

## Run

```bash
make run SCRIPT=tasks/02/aotautograd.py
TORCH_LOGS="aot_joint_graph" make run SCRIPT=tasks/02/aotautograd.py
TORCH_LOGS="dynamo,aot,inductor" make run SCRIPT=tasks/02/aotautograd.py
```

The `aot_function` part prints the partitioned forward and backward graphs. The
`torch.compile` part, when run with `TORCH_LOGS`, prints the joint graph and the
decomposed forward/backward graphs for the `nn.Linear` module.

## Tiny function: `(x * x).sum()`

Forward graph:

```text
graph():
    %primals_1 : [num_users=2] = placeholder[target=primals_1]
    %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_1, %primals_1), kwargs = {})
    %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul,), kwargs = {})
    return (sum_1, primals_1)
```

Backward graph:

```text
graph():
    %primals_1 : [num_users=2] = placeholder[target=primals_1]
    %tangents_1 : [num_users=1] = placeholder[target=tangents_1]
    %expand : [num_users=2] = call_function[target=torch.ops.aten.expand.default](args = (%tangents_1, [4]), kwargs = {})
    %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand, %primals_1), kwargs = {})
    %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand, %primals_1), kwargs = {})
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %mul_1), kwargs = {})
    return (add,)
```

The forward computes `x * x` then reduces with `sum`. The backward broadcasts
the scalar tangent back to shape `[4]`, then computes the elementwise gradient
`2 * x` as two multiplies and an add.

## Joint graph for the tiny function

From `TORCH_LOGS="aot_joint_graph"`:

```text
primals_1: "f32[4][1]cuda:0"; tangents_1: "f32[][]cuda:0";

mul:    "f32[4][1]cuda:0" = aten.mul.Tensor(primals_1, primals_1)
sum_1:  "f32[][]cuda:0"   = aten.sum.default(mul)
expand: "f32[4][0]cuda:0" = aten.expand.default(tangents_1, [4])
mul_1:  "f32[4][1]cuda:0" = aten.mul.Tensor(expand, primals_1)
mul_2:  "f32[4][1]cuda:0" = aten.mul.Tensor(expand, primals_1)
add:    "f32[4][1]cuda:0" = aten.add.Tensor(mul_2, mul_1)

return [sum_1, add]
```

This single graph contains both the forward computation (`mul`, `sum`) and the
backward computation (`expand`, `mul`, `add`) in one place. AOTAutograd traces
the joint graph first and then partitions it into the forward and backward
graphs shown above.

## `nn.Linear` decomposition

`Net.forward` is `torch.relu(self.fc(x)).sum()` where `self.fc` is
`nn.Linear(64, 32)`. AOTAutograd lowers the module into this decomposed forward
graph:

```text
def forward(self, primals_1: "f32[32, 64]", primals_2: "f32[32]", primals_3: "f32[16, 64]"):
    # self.fc.weight is stored as [32, 64]; the linear op needs weight.T.
    permute: "f32[64, 32]" = aten.permute.default(primals_1, [1, 0])
    # bias + x @ weight.T, expressed as one fused addmm op.
    addmm: "f32[16, 32]" = aten.addmm.default(primals_2, primals_3, permute)
    relu: "f32[16, 32]" = aten.relu.default(addmm)
    sum_1: "f32[]" = aten.sum.default(relu)
    # Saved for backward: the ReLU gate (x <= 0).
    le: "b8[16, 32]" = aten.le.Scalar(relu, 0)
    return (sum_1, primals_3, le)
```

Corresponding backward graph:

```text
def forward(self, primals_3: "f32[16, 64]", le: "b8[16, 32]", tangents_1: "f32[]"):
    expand: "f32[16, 32]" = aten.expand.default(tangents_1, [16, 32])
    full_default: "f32[]" = aten.full.default([], 0.0, ...)
    # ReLU backward as a mask gate instead of a single relu_backward op.
    where: "f32[16, 32]" = aten.where.self(le, full_default, expand)
    permute_1: "f32[32, 16]" = aten.permute.default(where, [1, 0])
    # Gradient w.r.t. weight = where.T @ x, expressed as a plain mm.
    mm: "f32[32, 64]" = aten.mm.default(permute_1, primals_3)
    # Gradient w.r.t. bias = sum of the gate over the batch dimension.
    sum_2: "f32[1, 32]" = aten.sum.dim_IntList(where, [0], True)
    view: "f32[32]" = aten.view.default(sum_2, [32])
    return (mm, view, None)
```

## Operator decompositions observed

- `nn.Linear` -> `aten.permute` (weight transpose) + `aten.addmm`.
- `aten.addmm` backward -> `aten.mm` plus `aten.sum.dim_IntList`/`aten.view` for
  the bias gradient.
- `relu` backward -> `aten.le.Scalar` + `aten.where.self`, rather than a single
  high-level backward node.

## Notes

- `aot_function` prints a harmless warning because the example returns the
  raw `gm.forward` callable instead of a boxed compiler function. It does not
  affect graph inspection.
- Functionalization inserts `aten.alias` nodes to model view/in-place aliasing
  in a pure, side-effect-free graph.

## Check

- Eager autograd runs backward by recording a tape of Python ops during the
  forward pass and replaying it. AOTAutograd instead traces forward and backward
  together into one joint graph, decomposes high-level ops, functionalizes
  mutations, and partitions the result into explicit forward and backward graphs.
- One operator decomposition is `nn.Linear` becoming `permute` + `addmm`, whose
  backward becomes `mm` and a summed `view` for the bias.
