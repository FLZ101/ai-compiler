import torch


@torch.compile
def f(x):
    return (x + 1) * 2


# First call: Dynamo specializes the compiled graph for shape [8].
f(torch.randn(8, device="cuda"))

# Same shape and metadata -> all guards pass -> cached graph is reused.
f(torch.randn(8, device="cuda"))

# New shape -> size=[8] guard fails -> Dynamo recompiles for [16].
f(torch.randn(16, device="cuda"))
print("static shape demo done")
