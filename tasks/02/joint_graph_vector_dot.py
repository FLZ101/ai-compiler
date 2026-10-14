"""Inspect the AOTAutograd joint graph for a vector dot product.

The function is ``(a * b).sum()``. Unlike vector add, the backward pass needs
the values of ``a`` and ``b``, so the forward graph saves both primals and the
backward graph receives them as inputs.
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
    # Vector dot product. Its backward pass is grad_a = b, grad_b = a, so the
    # forward graph must save both primals for the backward graph to use.
    return (a * b).sum()


a = torch.randn(3, requires_grad=True, device="cuda")
b = torch.randn(3, requires_grad=True, device="cuda")

compiled_fn = aot_function(
    fn,
    fw_compiler=print_graph("forward"),
    bw_compiler=print_graph("backward"),
)

out = compiled_fn(a, b)
out.backward()
