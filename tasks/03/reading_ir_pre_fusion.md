# How to read `ir_pre_fusion.txt`

Read `ir_pre_fusion.txt` as a scheduler-level IR dump. It describes buffers,
kernels, and loop bodies before final indexing/fusion decisions are applied.

## Start with the nodes

For Task 3, there are two nodes:

```text
op0: ExternKernelSchedulerNode(ExternKernelOut)
op1: SchedulerNode(ComputedBuffer)
```

`op0` is the matmul. It is an external kernel rather than Inductor-generated
code:

```text
op0.node.kernel = extern_kernels.mm
```

`op1` is a computed pointwise buffer. It reads the matmul result and bias, then
produces the final output.

## Trace the buffers

```text
op0.writes = buf0
op0.outputs = buf0
op1.unmet_dependencies = buf0, arg2_1
op1.writes = buf1
```

So the data flow is:

```text
arg0_1, arg1_1 -> mm -> buf0 -> add + relu -> buf1
arg2_1 -------------------------^
```

## Understand the layout

```text
buf0.layout = FixedLayout('cuda:0', torch.float32, size=[2048, 2048], stride=[2048, 1])
```

This says `buf0` is a CUDA float32 tensor of shape `2048 x 2048` with row-major
stride.

## Read the loop body

```python
class op1_loop_body:
    var_ranges = {p0: 2048, p1: 2048}
    index0 = 2048*p0 + p1
    index1 = p1

    def body(self, ops):
        load = ops.load('buf0', get_index)
        load_1 = ops.load('arg2_1', get_index_1)
        add = ops.add(load, load_1)
        relu = ops.relu(add)
        store = ops.store('buf1', get_index_2, relu, None)
```

This means:

1. For each output element `(p0, p1)`.
2. Load `buf0[p0, p1]`.
3. Load bias `arg2_1[p1]`.
4. Compute `add = buf0[p0, p1] + arg2_1[p1]`.
5. Compute `relu = relu(add)`.
6. Store into `buf1[p0, p1]`.

## Key things to look for

- `ExternKernel*`: operations left to libraries such as cuBLAS/cuDNN.
- `ComputedBuffer`: operations Inductor plans to generate itself.
- `unmet_dependencies`: inputs a node still needs.
- `met_dependencies`: inputs already resolved.
- `MemoryDep` expressions such as `2048*d0 + d1`: flattened indexing into buffers.
- `var_ranges`: loop dimensions.
- Multiple `ops.load/add/relu` inside one body: potential fusion.

The file answers: what buffers exist, which kernels produce them, and what each
generated loop will do before final fusion/index lowering.

## `StarDep` vs `MemoryDep`

Both describe a scheduler node's relationship to a buffer, but at different
precision.

### `MemoryDep`

`MemoryDep` is an exact memory dependency with an index expression:

```text
MemoryDep('buf0', 2048*d0 + d1, {d0: 2048, d1: 2048})
```

It means:

- Buffer: `buf0`
- Flattened index: `2048*d0 + d1`
- Loop ranges: `d0` in `[0, 2048)`, `d1` in `[0, 2048)`

This gives Inductor detailed access information, so it can reason about element
dependencies, aliasing, reuse, and fusion.

### `StarDep`

`StarDep` is a wildcard dependency:

```text
StarDep(name='buf0', mode=None)
```

It means the node accesses the entire buffer, without Inductor knowing exactly
which elements or what indexing pattern is used.

It appears for opaque or external kernels, such as the matmul:

```text
op0.node.kernel = extern_kernels.mm
op0.writes = [StarDep(name='buf0')]
```

Because `extern_kernels.mm` is treated as a black-box external kernel, Inductor
conservatively says this node writes all of `buf0`.

### Summary

| | `MemoryDep` | `StarDep` |
| --- | --- | --- |
| Precision | Exact element index | Whole buffer |
| Index expression | Yes | No |
| Used for analysis | Yes | Limited |
| Typical source | Generated pointwise/reduction buffers | External/opaque kernels |

`MemoryDep` means Inductor knows the exact memory access. `StarDep` means Inductor
only knows the entire buffer is accessed.

## `ExternKernelOut` vs `ComputedBuffer`

These describe two kinds of output buffers in Inductor's scheduler IR.

### `ExternKernelOut`

- Output of an external/opaque kernel.
- Inductor does not generate the loop body itself.
- Uses a `StarDep` because exact indexing is unknown.
- Typical examples: cuBLAS matmul, cuDNN convolution, vendor library calls.
- In Task 3:

```text
op0: ExternKernelSchedulerNode(ExternKernelOut)
op0.node.kernel = extern_kernels.mm
op0.writes = [StarDep(name='buf0')]
```

Inductor knows the buffer exists and is produced by the external kernel, but it
treats the access as a whole-buffer dependency.

### `ComputedBuffer`

- Output of an Inductor-generated loop.
- Inductor creates and schedules the loop body itself.
- Uses `MemoryDep` with exact index expressions.
- Can be fused, reordered, and optimized by Inductor.
- In Task 3, the bias-add + ReLU pointwise operation is a `ComputedBuffer`:

```text
op1: SchedulerNode(ComputedBuffer)
op1.writes = [MemoryDep('buf1', 2048*d0 + d1, {d0: 2048, d1: 2048})]
```

### Comparison

| | `ExternKernelOut` | `ComputedBuffer` |
| --- | --- | --- |
| Kernel source | External library | Inductor-generated |
| Loop body | Hidden/opaque | Visible `loop_body` |
| Dependency type | `StarDep` | `MemoryDep` |
| Fusion control | Limited | Fully schedulable |
| Example | cuBLAS matmul | add + ReLU pointwise |

`ExternKernelOut` means Inductor calls an outside kernel and receives this
buffer. `ComputedBuffer` means Inductor will generate code to compute the buffer
itself.
