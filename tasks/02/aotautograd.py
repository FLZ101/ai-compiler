"""Inspect AOTAutograd forward and backward decomposition.

AOTAutograd is the stage that sits between Dynamo's FX graph and Inductor. It
traces the forward and backward passes together into a joint graph, then
partitions that joint graph back into forward and backward graphs. While doing
so it also decomposes high-level ATen operators (for example ``addmm``) into
smaller primitive ops.
"""

import torch
from torch._functorch.aot_autograd import aot_function


def print_graph(name):
    """Return a backend-style compiler that just prints the graph it receives."""

    def compiler(gm, example_inputs):
        print(f"\n=== {name} graph ===")
        print(gm.graph)
        print(f"\n=== {name} Python code ===")
        print(gm.code)
        # AOTAutograd expects the compiler to return a callable that runs the
        # graph. Returning the captured forward keeps this inspection pass
        # executable.
        return gm.forward

    return compiler


def fn(x):
    # A tiny differentiable function. Its forward graph is a square plus a sum
    # reduction, and its backward graph propagates the sum's broadcast gradient.
    return (x * x).sum()


x = torch.randn(4, requires_grad=True, device="cuda")

# aot_function traces forward and backward together, runs the AOTAutograd
# pipeline (including functionalization and decompositions), and hands the
# partitioned graphs to the supplied forward/backward compilers.
compiled_fn = aot_function(
    fn,
    fw_compiler=print_graph("forward"),
    bw_compiler=print_graph("backward"),
)

out = compiled_fn(x)
out.backward()


# A realistic module so we can observe how nn.Linear becomes decomposed ops.
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(64, 32)

    def forward(self, x):
        return torch.relu(self.fc(x)).sum()


model = Net().cuda()
x = torch.randn(16, 64, device="cuda")

compiled_model = torch.compile(model)
loss = compiled_model(x)
loss.backward()

print("\nAOTAutograd inspection complete.")
