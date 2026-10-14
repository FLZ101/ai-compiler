"""Inspect the AOTAutograd joint graph for a vector add followed by a sum.

The function is ``(a + b).sum()``. The ``sum`` reduces shape ``[3]`` to a
scalar, so the backward pass broadcasts the scalar gradient back to ``[3]``
with ``expand``.
"""

import torch
from torch._functorch.aot_autograd import aot_function

# Print the joint graph (forward + backward in one graph) to stderr.
torch._logging.set_logs(aot_joint_graph=True)


def print_graph(name):
    """Return a backend-style compiler that just prints the graph it receives."""

    def compiler(gm, example_inputs):
        print(f"\n=== {name} graph ===")
        print(gm.graph)
        return gm.forward

    return compiler


def fn(a, b):
    # Vector add followed by a sum, so there is a scalar loss and a backward
    # pass that broadcasts the scalar gradient back to shape [3].
    return (a + b).sum()


a = torch.randn(3, requires_grad=True, device="cuda")
b = torch.randn(3, requires_grad=True, device="cuda")

compiled_fn = aot_function(
    fn,
    fw_compiler=print_graph("forward"),
    bw_compiler=print_graph("backward"),
)

out = compiled_fn(a, b)
out.backward()
