# Custom Accelerator Backend Integration Plan

This plan extends the main `torch.compile` workbook from an NVIDIA CUDA/Triton
device to a custom accelerator. The goal is to make a new device usable through
the same PyTorch path:

```text
eager PyTorch
  -> custom device runtime
  -> Dynamo / FX
  -> AOTAutograd
  -> Inductor
  -> custom codegen
  -> custom accelerator kernel
```

The custom accelerator may expose a C/C++ compiler, a vendor kernel library, a
Triton-like language, or only a small set of vendor primitives. The integration
strategy is the same in all cases: register the device, teach PyTorch how to
represent and trace tensors on it, implement kernels for the required operator
set, then connect Inductor code generation to that kernel interface.

## Objective

Run a `torch.compile`d model end-to-end on the custom device with:

- correct eager execution,
- Dynamo tracing without accidental CPU fallback,
- AOTAutograd forward/backward graph capture,
- Inductor lowering through a custom backend,
- kernel launch through the custom runtime,
- numerical and performance validation against the CPU or reference device.

## Architecture

```text
User model (torch.nn.Module)
        |
        v
PyTorch dispatcher / ATen ops
        |
        +-- Device module: allocator, streams, events, sync
        +-- Custom op kernels / vendor library
        |
        v
TorchDynamo
        |
        v
FX graph
        |
        v
AOTAutograd
        |
        v
TorchInductor scheduler
        |
        v
Custom Inductor backend / codegen
        |
        v
Custom accelerator compiler and runtime
```

## Integration surface

| Layer | PyTorch entry point | What must be implemented |
| --- | --- | --- |
| Device | `PrivateUse1` device registration | device type, allocator, stream, event, sync |
| Tensor APIs | device module hooks | create tensors, transfer, query device |
| Operators | `torch.library`, C++ extension | eager kernels for the required op set |
| Meta kernels | `torch.library.impl_abstract` | shape/dtype propagation without hardware |
| Tracing | Dynamo device guards | treat custom tensors as a first-class device |
| Compiler | custom Inductor backend | FX lower, scheduler hooks, codegen |
| Runtime | generated wrapper | allocation, kernel launch, stream ordering |
| Validation | op tests, `torch.compile` tests | numerical parity, fallback detection |

## Phase 0: Define the accelerator model

Before touching PyTorch, document the hardware and compiler model:

- memory spaces and allocator behavior,
- synchronous vs asynchronous execution,
- stream/event semantics,
- supported dtypes and layout,
- compiler command and code model,
- vendor library capabilities,
- minimum tensor rank, broadcast, and reduction support.

Output:

- one page describing the device model,
- a small C/C++ or vendor-language hello kernel,
- a command-line build/run recipe.

Check:

- Can a standalone tensor add and matmul kernel run outside PyTorch?

## Phase 1: Register the device

Make PyTorch recognize the device as a first-class backend.

Tasks:

- Register a `PrivateUse1` backend name.
- Provide a device module implementing memory allocation, deallocation,
  device guard, synchronization, streams, events, and current-device queries.
- Expose tensor factory functions such as `empty`, `rand`, and transfer helpers.
- Add a device context manager and `device_count`/`is_available`.
- Ensure `torch.device("custom:0")` creates valid tensors.

Output:

- `torch.ones(4, device="custom")` succeeds.
- `.to("custom")` and `.to("cpu")` copy correctly.
- `torch.cuda.synchronize()`-style synchronization is available for the custom
  device.

Check:

- Are allocations reused or cached?
- Does the current stream order memory operations correctly?
- Can two custom devices coexist?

## Phase 2: Implement the eager operator set

Choose the smallest operator set needed by the target models. A practical MVP
often starts with:

- constructors: `empty`, `empty_strided`, `zeros`, `ones`,
- data movement: `copy_`, `to`,
- unary pointwise: `relu`, `gelu`, `exp`, `sigmoid`, `neg`,
- binary pointwise: `add`, `sub`, `mul`, `div`,
- reductions: `sum`, `mean`, `max`,
- shape/layout: `view`, `reshape`, `transpose`, `permute`, `slice`,
- core compute: `addmm`/`mm`, `bmm`, `conv2d`, `softmax`, `layer_norm`.

Tasks:

- Implement each kernel in the custom compiler or vendor library.
- Bind kernels to ATen operators with `torch.library`.
- Register meta/abstract kernels so shape inference works without a real
  device.
- Mark unsupported ops to fail loudly instead of silently falling back.

Output:

- a conformance table showing op, dtype, device, and status,
- an eager numerical parity test against CPU.

Check:

- Does every op work with non-contiguous and strided inputs?
- Which ops still rely on CPU fallback?

## Phase 3: Make Dynamo trace custom tensors

Dynamo must understand tensors on the custom device and not treat them as
CPU tensors.

Tasks:

- Verify device guards record `custom` instead of falling back to CPU.
- Trace simple pointwise and matmul functions with `TORCH_LOGS=graph`.
- Check the FX graph for `prim.device`, `call_function` nodes, and no implicit
  `.to("cpu")`.
- Test dynamic shapes and data-dependent control flow.
- Build a small graph-break inventory.

Output:

- a traced FX graph for a representative model,
- a list of graph breaks and their causes.

Check:

- Does the trace contain only supported ATen ops?
- Do unsupported ops graph-break rather than silently move data?

## Phase 4: Teach AOTAutograd about the device

AOTAutograd decomposes ops and generates forward/backward graphs.

Tasks:

- Run `aot_eager` on a module with `requires_grad=True`.
- Add any required backward formulas or decompositions for custom ops.
- Ensure intermediate buffers and gradients use the custom device.
- Test functionalization and mutation removal.

Output:

- forward and backward FX graphs on the custom device,
- a successful `loss.backward()` with custom gradients.

Check:

- Are all gradient tensors allocated on the custom device?
- Does AOTAutograd insert CPU copies anywhere?

## Phase 5: Implement the Inductor backend

This is the core compiler work. There are two useful paths:

1. **Wrapper-only backend:** use Inductor scheduling and memory planning, but
   emit calls to vendor kernels for every node.
2. **Full codegen backend:** generate custom kernels for fused pointwise and
   reduction nodes.

Start with the wrapper-only backend, then add fused codegen.

Tasks:

- Register the backend with `torch._dynamo.backends.registry`.
- Provide a custom codegen class and scheduler hooks.
- Implement runtime wrapper functions for allocation, stream access, and kernel
  launch.
- Emit one vendor-library call per scheduled node first.
- Add fused pointwise codegen for a simple `add` + `relu` chain.
- Add reduction codegen for `sum` and `mean`.
- Add matmul codegen or map matmul to the vendor library.

Output:

- `torch.compile(model, backend="custom_ind")` runs on the custom device,
- `TORCH_LOGS=output_code` shows generated custom kernel code,
- a kernel inventory and fallback inventory.

Check:

- Does the generated wrapper preserve stream ordering?
- Can the backend allocate temporary buffers without CPU round-trips?
- Does a graph break fall back to eager custom ops?

## Phase 6: Build the runtime wrapper

Generated code needs a small runtime that owns execution.

Tasks:

- Generate code that allocates output and intermediate tensors.
- Expose the current stream to generated launches.
- Record and synchronize custom events.
- Add a memory planner for reused buffers.
- Support compile-time constants and weight buffers.
- Support static shapes first, then constrained dynamic shapes.

Output:

- a runnable generated artifact,
- repeated calls with no recompilation for unchanged shapes,
- clean release of temporary buffers.

Check:

- Where does allocation live: Python, C++, or the custom runtime?
- Can the runtime execute without a Python interpreter?

## Phase 7: Fallback and decomposition strategy

No custom backend implements every ATen op on day one. The integration must
explicitly choose between:

- custom kernel,
- decomposition into supported ops,
- eager custom op,
- CPU fallback with device transfer.

Tasks:

- Define a policy per unsupported op.
- Log every CPU fallback and device copy.
- Add decompositions for composite ops before they become unsupported nodes.
- Measure fallback cost and decide which ops to implement next.

Output:

- a fallback report per model,
- a prioritized op-implementation backlog.

Check:

- Are CPU fallbacks silent or observable?
- Does a CPU fallback break the CUDA/custom graph capture path?

## Phase 8: Validate correctness and performance

Use a layered test strategy:

1. Custom op unit tests on eager.
2. OpInfo-style numerical parity against CPU or reference device.
3. `aot_eager` correctness tests.
4. Inductor custom backend correctness tests.
5. End-to-end model accuracy tests.
6. Performance comparison against CPU fallback and vendor library.

Tasks:

- Build a reproducible benchmark suite.
- Measure compile time, first-call latency, steady-state latency, and
  throughput.
- Profile kernel time versus device transfer and Python overhead.
- Add regression tests for shape, dtype, and device changes.

Output:

- correctness report,
- benchmark table,
- known limitations.

Check:

- Is the custom backend faster than CPU fallback?
- Which op or graph break dominates compile time?
- Which kernel dominates steady-state latency?

## Phase 9: Production hardening

Turn the prototype into a deployable backend.

Tasks:

- Freeze weights and cache compiled artifacts.
- Add AOT export/loading if the runtime supports no-Python execution.
- Constrain dynamic shapes and warm up all served variants.
- Add error reporting for unsupported shapes and dtypes.
- Pin the compiler/runtime versions with the generated artifacts.
- Document the supported operator set and fallback policy.

Output:

- deployment artifact,
- operator support matrix,
- operations runbook.

Check:

- Can a cold process load and run the artifact?
- Does the backend fail clearly on unsupported inputs?

## Minimal viable integration checklist

The fastest path to a working custom `torch.compile` backend is:

- [ ] custom device recognized by `torch.device`
- [ ] `empty`, `copy_`, and synchronization work
- [ ] eager matmul and pointwise ops pass parity
- [ ] meta kernels return correct shapes
- [ ] Dynamo traces a small model without CPU fallback
- [ ] AOTAutograd runs forward and backward
- [ ] wrapper-only Inductor backend launches vendor kernels
- [ ] generated wrapper preserves stream ordering
- [ ] one fused pointwise kernel is generated
- [ ] static-shape benchmark runs without recompilation

## Risks

- PyTorch internal device/backend APIs can change between releases.
- Implicit CPU fallback hides correctness and performance problems.
- Missing meta kernels cause Dynamo/AOTAutograd to fail or fall back.
- Stream/event semantics differ across accelerators.
- Custom memory allocators may not support the strides PyTorch expects.
- Inductor's scheduler assumptions are biased toward CUDA/Triton execution.
- Vendor compilers may not support the same tiling/reduction model as Triton.

## Relation to the main workbook

The main `torch_compile_study_plan.md` should be completed through the CUDA
baseline, Inductor artifacts, fusion, compile modes, profiling, and hand-written
Triton tasks first. That gives the mental model needed to replace each NVIDIA
component with the custom equivalent.
