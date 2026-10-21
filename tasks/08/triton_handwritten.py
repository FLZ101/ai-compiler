import torch
import triton
import triton.language as tl
import torch._inductor.config as inductor_config

# Inductor normally dispatches a matmul like this to cuBLAS. Enable its native
# Triton matmul path so TORCH_LOGS=output_code prints an actual Inductor GEMM
# kernel that can be compared with matmul_kernel below.
inductor_config.triton.native_matmul = True


@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Block sizes are compile-time constants. The Triton compiler specializes
    # this kernel for every concrete combination and can size registers and
    # tensor-core instructions accordingly.
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # One program computes one BLOCK_M x BLOCK_N output tile. Inductor's
    # generated matmul uses the same program/tile mapping, except that it
    # usually picks block sizes via autotuning instead of hardcoding them.
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # Programs are laid out along the N dimension fastest: pid_m * N + pid_n.
    # An equivalent 2D-grid form would be:
    #   pid_m = tl.program_id(0)
    #   pid_n = tl.program_id(1)
    # with grid = (num_pid_m, num_pid_n). This example intentionally uses the
    # flattened 1D form and decodes pid into (pid_m, pid_n).
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Offset vectors identify which rows/columns this tile owns:
    #   offs_m: shape (BLOCK_M,)
    #   offs_n: shape (BLOCK_N,)
    #   offs_k: shape (BLOCK_K,)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Build 2D pointer blocks with singleton axes and broadcasting.
    #
    # offs_m[:, None] has shape (BLOCK_M, 1): one row offset per row.
    # offs_k[None, :] has shape (1, BLOCK_K): one column offset per K chunk.
    # Their sum broadcasts to (BLOCK_M, BLOCK_K), so a_ptrs[m, k] points at
    # A[row=m, col=k] for every element in this program's A tile.
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)

    # The B tile is transposed relative to A:
    #   offs_k[:, None] -> (BLOCK_K, 1)
    #   offs_n[None, :] -> (1, BLOCK_N)
    # Their sum broadcasts to (BLOCK_K, BLOCK_N), giving B[k, n].
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # The dot-product accumulator is always float32, matching tensor-core
    # GEMM semantics: float16 inputs are multiplied and accumulated in fp32.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Walk K in BLOCK_K chunks. This is the serial reduction loop. Inductor's
    # generated matmul has the same loop, but often adds pipelining stages.
    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Cast back to the input/output dtype before the store.
    c = acc.to(tl.float16)

    # Rebuild the 2D row/column pointer block for the output tile, then mask
    # only the M and N edges. This script intentionally uses shapes that divide
    # evenly by BLOCK_M/BLOCK_N, so the mask is a correctness guard here rather
    # than an actively taken branch.
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_matmul(a, b):
    assert a.is_cuda and b.is_cuda
    M, K = a.shape
    K2, N = b.shape
    assert K == K2

    # Triton writes into an empty output tensor supplied by the caller. The
    # kernel therefore has no return value; the host only launches it.
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # The launch grid is a lambda so Triton can inspect the selected constexpr
    # block sizes and compute how many programs cover the output matrix.
    #
    # triton.cdiv is ceiling division: ceil(dim / BLOCK). It returns enough
    # blocks to cover dim even when dim is not divisible by BLOCK. For example,
    # cdiv(100, 16) -> 7 because 6 blocks cover only 96 elements.
    #
    # num_pid_m * num_pid_n is the total number of output tiles. The grid is
    # flattened to one dimension; matmul_kernel decodes pid back into pid_m and
    # pid_n. Boundary tiles beyond the tensor shape are handled by c_mask.
    # The equivalent 2D-grid form would be:
    #   grid = (num_pid_m, num_pid_n)
    #   pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    #
    # The lambda argument, meta, is Triton launch metadata. It contains every
    # tl.constexpr kernel parameter (BLOCK_M, BLOCK_N, BLOCK_K) plus launch
    # options such as num_warps. Normal kernel arguments like pointers, M, N, K,
    # and strides are not in meta; they are passed separately after [grid].
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

    # Strides are element counts, not bytes. Dimension 0 is the outer/row step
    # and dimension 1 is the inner, fastest-changing column step. For contiguous
    # row-major tensors, .stride(1) == 1 and .stride(0) == the row length.
    # Triton converts these element strides to byte offsets internally.
    matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=16,
        BLOCK_N=16,
        BLOCK_K=16,
        num_warps=4,
    )
    return c


@torch.compile
def inductor_matmul(a, b):
    # Inductor lowers this same expression, choosing a codegen backend and
    # tile configuration automatically. Use TORCH_LOGS=output_code to print the
    # Triton kernel it generated for comparison with matmul_kernel above.
    return a @ b


torch.manual_seed(0)

M, N, K = 512, 512, 512
a = torch.randn(M, K, device="cuda", dtype=torch.float16)
b = torch.randn(K, N, device="cuda", dtype=torch.float16)

out_triton = triton_matmul(a, b)
out_inductor = inductor_matmul(a, b)

# Both launches are asynchronous. This is a host/device-level barrier: Python
# blocks until all previously queued work on this CUDA device is complete. It
# is not the same as a block-level __syncthreads() inside a running kernel.
torch.cuda.synchronize()

print(f"triton shape:    {tuple(out_triton.shape)}")
print(f"inductor shape:  {tuple(out_inductor.shape)}")
print(f"max abs diff:    {(out_triton - out_inductor).abs().max().item()}")
