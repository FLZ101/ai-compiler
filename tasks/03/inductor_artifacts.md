# Task 03: Complete Inductor artifacts

## Files

- `tasks/03/inductor_artifacts.py`
- `tasks/03/inductor_artifacts.md`
- `tasks/03/reading_ir_pre_fusion.md`
- `tasks/03/fusion_example.py`
- `tasks/03/fusion_example.md`
- `tasks/03/fused_scheduler_node.md`
- `tasks/03/reduction.md`

## Run

```bash
find tasks/03/debug -mindepth 1 -delete 2>/dev/null || true
TORCH_COMPILE_DEBUG=1 TORCH_LOGS="output_code" \
  make run SCRIPT=tasks/03/inductor_artifacts.py
find tasks/03/debug -maxdepth 5 -type f | sort
```

The script sets `inductor_config.force_disable_caches = True`, so compilation
and artifact generation happen on every run.

The debug directory is derived from the Python file with `Path(__file__)`, so
artifacts are written to `tasks/03/debug/` regardless of the current working
directory.

## Configuration

The script applies two compiler settings before calling `torch.compile`:

- `inductor_config.force_disable_caches = True`
- `inductor_config.trace.debug_dir = tasks/03/debug`

The first forces fresh code generation. The second makes the debug trace local
to the task.

## Actual layout

```text
tasks/03/debug/
  torchinductor/
    model__0_inference_0.0/
      fx_graph_readable.py
      fx_graph_runnable.py
      fx_graph_transformed.py
      inductor_provenance_tracking_node_mappings.json
      ir_pre_fusion.txt
      ir_post_fusion.txt
      output_code.py
```

## Artifact roles

The model directory name can vary by PyTorch version. Locate it with:

```bash
find tasks/03/debug -type d -name 'model__*'
```

### `fx_graph_readable.py`

The captured and decomposed FX graph before Inductor lowering:

```python
mm = torch.ops.aten.mm.default(arg0_1, arg1_1)
add = torch.ops.aten.add.Tensor(mm, arg2_1)
relu = torch.ops.aten.relu.default(add)
return (relu,)
```

### `fx_graph_runnable.py`

The FX graph in an executable Python form, with concrete input descriptions.

### `fx_graph_transformed.py`

The FX graph after Inductor's graph transformations, showing the IR that is
closer to final lowering.

### `inductor_provenance_tracking_node_mappings.json`

Maps generated nodes back to source ATen/Inductor operations, useful for
debugging provenance.

### `ir_pre_fusion.txt`

Inductor's scheduler view before final memory/index fusion. The matmul is an
external kernel, while `add` and `relu` are already represented as one pointwise
loop body.

### `ir_post_fusion.txt`

The scheduler view after fusion and indexing decisions. In this small example,
the main visible change is flattened scheduling/index metadata.

### `output_code.py`

The final generated runtime program. It contains:

- `extern_kernels.mm` for the matrix multiply.
- `triton_poi_fused_add_relu_0` for the fused bias-add and ReLU pointwise kernel.
- Buffer allocation and reuse code.
- The `Runner.call` wrapper that launches the kernels.

## Observations

- The matmul remains a cuBLAS/external kernel because it is a heavy GEMM.
- Bias add and ReLU are fused into one Triton pointwise kernel.
- `buf1 = buf0` shows memory reuse instead of allocating a new output buffer.
- The final program is a generated Python wrapper around compiled kernels, not
  the original eager Python function.

## Check

- The FX graph is present.
- Pre-fusion and post-fusion IR are present.
- The generated runtime code is present.
