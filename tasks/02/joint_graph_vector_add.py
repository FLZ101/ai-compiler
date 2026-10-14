"""Inspect the AOTAutograd joint graph for a vector add without a reduction.

The function is ``a + b``. Because the output is a vector, the incoming
gradient ``tangents_1`` is already shape ``[3]`` and flows through to both
inputs unchanged: no ``expand`` broadcast is needed.
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
    # Plain vector add. The output is shape [3], so backward passes the vector
    # tangent straight through to both inputs without an expand.
    return a + b


a = torch.randn(3, requires_grad=True, device="cuda")
b = torch.randn(3, requires_grad=True, device="cuda")

compiled_fn = aot_function(
    fn,
    fw_compiler=print_graph("forward"),
    bw_compiler=print_graph("backward"),
)

out = compiled_fn(a, b)
# The output is not a scalar, so backward needs an explicit upstream gradient.
out.backward(torch.ones_like(out))
