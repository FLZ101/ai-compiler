import dis

import torch


def show_graph(name, fn, *args):
    explanation = torch._dynamo.explain(fn)(*args)
    print(f"\n=== {name} ===")
    print(
        f"graphs={explanation.graph_count} "
        f"breaks={explanation.graph_break_count} "
        f"ops={explanation.op_count}"
    )
    if explanation.break_reasons:
        print("break reasons:", explanation.break_reasons)
    for index, graph in enumerate(explanation.graphs):
        print(f"--- graph {index} ---")
        graph.print_readable()


def arithmetic(x, y):
    # LOAD_FAST x, LOAD_FAST y, BINARY_OP +, multiply by Python 2.
    return (x + y) * 2


def matmul_bias_relu(x, w, b):
    # Three ATen nodes: mm, add, relu.
    return torch.relu(x @ w + b)


def static_control(x, flag):
    # Python bool is known at trace time, so only the taken branch is traced.
    if flag:
        return x + 1
    return x - 1


def range_loop(x):
    # Python range is unrolled during tracing.
    for i in range(3):
        x = x + i
    return x


def tuple_unpack(x):
    # PyTree/list operations stay as VariableTrackers; tensor nodes appear only
    # when tensor arithmetic is executed.
    a, b = x.split(1)
    return a + b


def nested_call(x):
    def helper(y):
        return y * 2

    return helper(x)


def non_tensor_return(x):
    # Shape/dtype metadata are Python values while x + 1 remains an FX tensor.
    return x.shape[0], x.dtype, x + 1


def data_dependent_control(x):
    # The condition depends on tensor data, so Dynamo must graph-break.
    if x.sum() > 0:
        return x + 1
    return x - 1


x = torch.randn(2, 2)
y = torch.randn(2, 2)
w = torch.randn(2, 2)
b = torch.randn(2)

print("bytecode for arithmetic:")
dis.dis(arithmetic)

show_graph("arithmetic", arithmetic, x, y)
show_graph("matmul_bias_relu", matmul_bias_relu, x, w, b)
show_graph("static_control true", static_control, x, True)
show_graph("static_control false", static_control, x, False)
show_graph("range_loop", range_loop, x)
show_graph("tuple_unpack", tuple_unpack, x)
show_graph("nested_call", nested_call, x)
show_graph("non_tensor_return", non_tensor_return, x)
show_graph("data_dependent_control", data_dependent_control, x)
