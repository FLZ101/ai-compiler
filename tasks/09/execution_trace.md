# Task 09 execution trace: matmul + bias + ReLU

This trace follows `tasks/09/trace_example.py` through the complete compiler
route and records both the runtime evidence and the corresponding installed
source-code location.

## Example

```python
@torch.compile
def matmul_bias_relu(x, w, b):
    return torch.relu(x @ w + b)
```

The generated debug artifacts and logs are in:

```text
tasks/09/debug/trace_dynamo.log
tasks/09/debug/trace_aot.log
tasks/09/debug/trace_inductor.log
tasks/09/debug/torchinductor/model__0_inference_0.0/
```

## Trace table

| Step | What happens | Source location | Runtime evidence |
| --- | --- | --- | --- |
| 1 | User calls the compiled function | `tasks/09/trace_example.py:25` | `torch.relu(x @ w + b)` |
| 2 | Dynamo installs a frame-evaluation hook | `torch/_dynamo/eval_frame.py` | Python execution enters Dynamo on first call |
| 3 | Dynamo records FX nodes and builds `OutputGraph` | `torch/_dynamo/output_graph.py:672` | `fx_graph_readable.py` contains `aten.mm`, `aten.add.Tensor`, `aten.relu.default` |
| 4 | Guards capture tensor dtype/shape/stride/device and global state | `torch/_dynamo/guards.py` | `trace_dynamo.log` shows `TENSOR_MATCH`, `NO_TENSOR_ALIASING`, `GLOBAL_STATE` |
| 5 | AOTAutograd exports a functional forward graph | `torch/_functorch/aot_autograd.py:1550` | `trace_aot.log` runs the export path |
| 6 | Inductor receives the FX graph and lowers it to IR | `torch/_inductor/compile_fx.py:2901`, `torch/_inductor/graph.py:386` | `ir_pre_fusion.txt` is produced |
| 7 | Scheduler creates one node for matmul and one for bias+ReLU | `torch/_inductor/scheduler.py:2257` | `ir_pre_fusion.txt` shows `op0` and `op1` |
| 8 | Scheduler fuses the dependent nodes | `torch/_inductor/scheduler.py:2725` | `ir_post_fusion.txt` shows `FusedSchedulerNode(SchedulerNode,SchedulerNode)` |
| 9 | Triton code generator emits one kernel | `torch/_inductor/codegen/triton.py:3274` | `output_code.py` contains `triton_red_fused_add_mm_relu_0` |
| 10 | The matmul is lowered to `tl.dot` | `torch/_inductor/codegen/triton.py:1616` | generated body contains `tl.dot(tmp2, tmp3, input_precision="ieee")` |
| 11 | The complete kernel source is emitted | `torch/_inductor/codegen/triton.py:8008` | generated body contains the reduction loop and final store |
| 12 | Wrapper allocates buffers and launches the kernel | `torch/_inductor/codegen/wrapper.py:1448` | generated wrapper calls `triton_red_fused_add_mm_relu_0.run(...)` |

## Stage details

### Dynamo FX graph

`tasks/09/debug/torchinductor/model__0_inference_0.0/fx_graph_readable.py`:

```python
mm: "f32[256, 256]" = torch.ops.aten.mm.default(arg0_1, arg1_1)
add: "f32[256, 256]" = torch.ops.aten.add.Tensor(mm, arg2_1)
relu: "f32[256, 256]" = torch.ops.aten.relu.default(add)
return (relu,)
```

### Guards

`tasks/09/debug/trace_dynamo.log` records guards for each input:

```text
TENSOR_MATCH: check_tensor(
    L['x'],
    Tensor,
    DispatchKeySet(CUDA, ...),
    torch.float32,
    device=0,
    requires_grad=False,
    size=[256, 256],
    stride=[256, 1],
)
NO_TENSOR_ALIASING: check_no_aliasing(L['b'], L['w'], L['x'])
```

The guards prove the traced graph is valid only for those tensor properties.

### Inductor IR before fusion

`ir_pre_fusion.txt` has two scheduler nodes:

```text
op0: SchedulerNode(ComputedBuffer)  # matmul -> buf0
op1: SchedulerNode(ComputedBuffer)  # add -> relu -> buf1
```

`op0` uses `ops.dot` and `ops.reduction`. `op1` uses `ops.add` and `ops.relu`.

### Inductor IR after fusion

`ir_post_fusion.txt` merges them:

```text
op0_op1: FusedSchedulerNode(SchedulerNode,SchedulerNode)
```

The matmul and the bias+ReLU epilogue become one scheduled kernel.

### Generated Triton kernel

`trace_inductor.log` emits `triton_red_fused_add_mm_relu_0`:

```python
_tmp6 = tl.full([YBLOCK, XBLOCK], 0, tl.float32)
for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
    ...
    tmp4 = tl.dot(tmp2, tmp3, input_precision="ieee")
    _tmp6 = _tmp6 + tmp4
...
tmp8 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
tmp9 = tmp6 + tmp8
tmp11 = tl.maximum(tmp10, tmp9, tl.PropagateNan.ALL)
tl.store(in_out_ptr0 + (x1 + 256*y0), tmp11, xmask & ymask)
```

This is the matmul reduction loop followed by the fused bias add and ReLU.

### Launch wrapper

The generated wrapper calls the kernel once:

```python
triton_red_fused_add_mm_relu_0.run(
    buf1, arg0_1, arg1_1, arg2_1,
    256, 256, 256,
    stream=raw_stream0,
)
```

## Conclusion

One user expression (`x @ w + b` followed by `relu`) travels through:

```text
Python
  -> Dynamo FX graph
  -> guards
  -> AOTAutograd functional graph
  -> Inductor IR nodes
  -> FusedSchedulerNode
  -> Triton kernel
  -> generated Python wrapper launch
```

The corresponding source locations are the implementation of each arrow in
that route.

## Essential steps not exercised by this example

This inference-only, static-shape, cache-disabled example does not demonstrate
every stage in the production compiler route.

| Missing step | Why it is absent here | How to observe it |
| --- | --- | --- |
| Backward/AOTAutograd joint graph | `matmul_bias_relu` is called without `loss.backward()` | run a training example with `TORCH_LOGS=aot_joint_graph` |
| Functionalization of in-place mutation | the example has no `add_`/`relu_` mutation | trace `x.relu_()` with `TORCH_LOGS=post_grad_graphs` |
| Composite-op decomposition | `mm`, `add`, and `relu` are already primitive | trace `torch.nn.functional.gelu` or `softmax` |
| Fake-tensor/meta tracing | this happens inside Dynamo/AOTAutograd but is not exposed in these logs | use `FakeTensorMode` or `torch.library.impl_abstract` |
| Dynamic-shape recompile and guard failure | inputs stay `256 x 256` | change shapes between calls and use `TORCH_LOGS=recompiles,guards` |
| Graph break handling | the function contains only supported tensor ops | add Python `if` or data-dependent control flow |
| Buffer-reuse/memory planning | one output buffer makes reuse hard to see | trace a chain with several non-overlapping intermediates |
| Compiled-artifact cache hit | the script sets `force_disable_caches = True` | remove that setting and run twice |
| Triton IR, PTX, and SASS lowering | `output_code` stops at Triton source | set `TRITON_DUMP_IR=1`, `TRITON_KERNEL_DUMP=1`, and inspect generated PTX |
| C++ wrapper / AOT artifact | this example uses the Python wrapper | enable `inductor_config.cpp_wrapper = True` or use AOTInductor |
| CUDA graph capture | `torch.compile(mode="reduce-overhead")` is not used | compile the same function with CUDA-graph mode |
| External backend selection | native Triton matmul is forced | remove `triton.native_matmul = True` and observe `extern_kernels.mm` |

The current trace is intentionally narrow: it isolates the happy path from FX
through fused Triton code generation. Each missing row should be added as a
separate trace when studying that subsystem.
