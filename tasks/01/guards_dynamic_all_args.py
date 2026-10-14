import torch


@torch.compile(dynamic=True)
def dynamic_f(x):
    return (x + 1) * 2


# These different lengths can reuse one compiled program because dimension 0
# is marked dynamic instead of specializing to a single concrete length.
dynamic_f(torch.randn(8, device="cuda"))
dynamic_f(torch.randn(16, device="cuda"))
dynamic_f(torch.randn(4, device="cuda"))
print("dynamic shape demo done")


@torch.compile(dynamic=True)
def two_arg_dynamic(x, y):
    return (x + 1) * (y + 2)


# `dynamic=True` marks all tensor arguments dynamic, not only the first one.
two_arg_dynamic(torch.randn(4, device="cuda"), torch.randn(4, device="cuda"))
two_arg_dynamic(torch.randn(8, device="cuda"), torch.randn(8, device="cuda"))
print("multi-argument dynamic demo done")
