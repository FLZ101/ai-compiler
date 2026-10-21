# Task 08: hand-written Triton matmul vs Inductor

This task writes a Triton GEMM by hand and compares it with the kernel that
TorchInductor emits for `a @ b`.

For a focused explanation of the decorator and its relationship to
`torch.compile`, see [triton_jit.md](triton_jit.md).

By default, Inductor sends `aten.mm` to an external GEMM implementation such
as cuBLAS, so `TORCH_LOGS=output_code` would show `extern_kernels.mm(...)`.
The script therefore sets:

```python
inductor_config.triton.native_matmul = True
```

That makes Inductor generate its own Triton matmul kernel so the two sources
can be compared directly.

## Run

```bash
make run SCRIPT=tasks/08/triton_handwritten.py
```

Inspect the Inductor-generated Triton source:

```bash
TORCH_LOGS=output_code make run SCRIPT=tasks/08/triton_handwritten.py
```

To keep the compiler log in the task's ignored debug directory:

```bash
TORCH_LOGS=output_code make run-log SCRIPT=tasks/08/triton_handwritten.py
```

The generated source in the log is also copied to a temp file such as
`/tmp/torchinductor_*/.../....py`.

## Expected result

For `M = N = K = 512` with `torch.float16`, the hand-written kernel and the
compiled matmul should agree almost exactly:

```text
triton shape:    (512, 512)
inductor shape:  (512, 512)
max abs diff:    0.0
```

Small floating-point differences are normal when Inductor selects a different
tile shape or reduction order.

On the RTX 3060 Laptop GPU, this task produces:

```text
triton shape:    (512, 512)
inductor shape:  (512, 512)
max abs diff:    0.0
```

## Hand-written kernel structure

- `pid = tl.program_id(0)` assigns each program one output tile.
- `pid_m = pid // num_pid_n` and `pid_n = pid % num_pid_n` map the flat program
  id back to the `(M, N)` tile grid.
- `offs_m`, `offs_n`, and `offs_k` are offset vectors that become pointer
  arithmetic inside the generated GPU code.
- `BLOCK_M`, `BLOCK_N`, and `BLOCK_K` are `tl.constexpr` values. Triton
  specializes the kernel for these tile sizes at compilation time.
- `acc = tl.zeros(..., dtype=tl.float32)` keeps the tensor-core accumulator in
  fp32 even though the inputs and output are fp16.
- The serial `range(0, K, BLOCK_K)` loop performs the reduction across `K`.
- `tl.dot(a, b)` lowers to a tensor-core dot product for one K block.
- The output store uses a mask for the `M` and `N` boundaries. This example
  uses sizes divisible by the block dimensions, so the mask is not actively
  masking any element, but it makes the kernel general for arbitrary output
  edges.

## Hand-written vs generated kernel

| Aspect | Hand-written kernel | Inductor-generated matmul |
| --- | --- | --- |
| Program mapping | flat `tl.program_id(0)`, decoded to `(pid_m, pid_n)` | `Grid2D`: `program_id(0)` is columns/X, `program_id(1)` is rows/Y |
| Block sizes | fixed `16 x 16 x 16` in the script | selected by Inductor, often autotuned |
| K reduction | serial `range(0, K, BLOCK_K)` loop with `tl.dot` | `tl.range(0, r0_numel, R0_BLOCK)` loop with `tl.dot` |
| Accumulator | explicit fp32 `tl.zeros` | `tl.full([YBLOCK, XBLOCK], 0, tl.float32)` |
| Output store | `tl.store` with edge mask | equivalent masked store |
| Launch configuration | host picks `num_warps=4` | Inductor picks warps/stages from its config |

The high-level shape is identical because Inductor's Triton backend lowers
pointwise/reduction ops using the same block-program model. The differences are
configuration choices: Inductor owns tile sizes, warps, pipelining, and load
mask details.

## What the generated kernel looks like

With native matmul enabled, the output log names the kernel
`triton_red_fused_mm_0`. Its body follows the same reduction shape:

```text
yoffset = tl.program_id(1) * YBLOCK
xoffset = tl.program_id(0) * XBLOCK
_tmp10 = tl.full([YBLOCK, XBLOCK], 0, tl.float32)
for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
    ...
    tmp8 = tl.dot(tmp6, tmp7, input_precision="ieee")
    _tmp10 = _tmp10 + tmp8
tl.store(out_ptr0 + (x1 + 512*y0), tmp10, xmask & ymask)
```

This is the same tile-based GEMM as `matmul_kernel`, but expressed through
Inductor's generic reduction template. The main visual differences are:

- Inductor uses a 2D launch grid instead of manually decoding one program id.
- Inductor materializes three-dimensional index arrays and then reshapes them
  before `tl.dot`.
- Inductor adds masks and zero-filled out-of-range loads to make the template
  safe for arbitrary shapes, while the hand-written kernel only masks the store.
- Inductor owns the selected `YBLOCK`, `XBLOCK`, and `R0_BLOCK` values, plus
  `num_warps` and `num_stages`.
