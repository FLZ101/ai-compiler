import torch
from torch._dynamo import mark_dynamic


@torch.compile
def selective_dynamic(x, y):
    return x + y


# Mark only dimension 0 of `x` dynamic. `y` stays fully static and must keep
# the shape recorded on the first call, [1, 16], so it broadcasts across `x`.
x = torch.randn(8, 16, device="cuda")
y = torch.randn(1, 16, device="cuda")
mark_dynamic(x, 0)
selective_dynamic(x, y)

# This reuses the same compiled graph: `x` dimension 0 changed, but `y` did not.
x2 = torch.randn(16, 16, device="cuda")
selective_dynamic(x2, y)
print("selective dynamic demo done")


@torch.compile
def full_dynamic(x):
    return torch.relu(x)


# Mark every dimension of this tensor dynamic. Equivalent to repeated
# `mark_dynamic(x, dim)` calls for each dimension.
xa = torch.randn(8, 16, device="cuda")
mark_dynamic(xa, list(range(xa.dim())))
full_dynamic(xa)

# Both dimensions changed, so this call can reuse the same compiled graph.
xb = torch.randn(4, 32, device="cuda")
mark_dynamic(xb, list(range(xb.dim())))
full_dynamic(xb)
print("full dynamic x demo done")
