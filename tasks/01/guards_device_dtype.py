import torch


@torch.compile
def relu_f(x):
    return torch.relu(x)


# Trace with CUDA float32.
relu_f(torch.randn(4, device="cuda", dtype=torch.float32))

# Dtype changes: float32 -> float16 guard fails and causes recompilation.
relu_f(torch.randn(4, device="cuda", dtype=torch.float16))

# A CPU tensor would also fail the CUDA device guard, but that example is
# omitted here because CPU Inductor codegen currently requires `openssl`.
print("device/dtype demo done")
