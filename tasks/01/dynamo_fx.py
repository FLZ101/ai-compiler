import torch


# Example 1: report what Dynamo traces, including op and graph-break counts.
def fn(x):
    return torch.relu(x + 1) * 2


# Dynamo needs real tensors to trace with. CUDA placement also makes the
# generated guards record the tensor's device, dtype, shape, and stride.
x = torch.randn(8, device="cuda")
explanation = torch._dynamo.explain(fn)(x)
print("=== explain ===")
print(explanation)


# Example 2: use a custom backend to inspect the graph before Inductor lowers it.
def inspect_backend(gm, example_inputs):
    # `gm` is the FX GraphModule captured by Dynamo.
    # `gm.graph` shows the symbolic node IR.
    # `gm.code` shows Python reconstructed from that graph.
    print("\n=== Captured FX graph ===")
    print(gm.graph)
    print("\n=== Generated Python code ===")
    print(gm.code)

    # A backend must return a callable that runs the graph. Returning the
    # captured forward keeps this inspection pass executable.
    return gm.forward


@torch.compile(backend=inspect_backend)
def captured_fn(x):
    return torch.relu(x + 1) * 2


captured_fn(torch.randn(8, device="cuda"))


# Example 3: observe a graph break caused by untraceable Python I/O.
@torch.compile
def graph_break_fn(x):
    # This tensor op is traceable, so it belongs to the first FX subgraph.
    y = x + 1

    # Dynamo cannot put this Python `print` into an FX graph, so it breaks the
    # graph here, runs the print eagerly, then resumes tracing below.
    print("inside compiled function")  # Python I/O causes a graph break

    # This tensor op is compiled in a second, resumed FX subgraph.
    return y * 2


print("\n=== graph break example ===")
graph_break_fn(torch.randn(8, device="cuda"))
