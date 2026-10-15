import torch
import torch._inductor.config as inductor_config

from pathlib import Path

# Keep logs deterministic and local to Task 05.
inductor_config.force_disable_caches = True
inductor_config.trace.debug_dir = str(Path(__file__).resolve().parent / "debug")


def pointwise_chain(x):
    return x + 1 + 2 + 3


def matmul_bias_relu(x, w, b):
    return torch.relu(x @ w + b)


def attention_pattern(q, k, v):
    scores = q @ k.transpose(-2, -1)
    attn = torch.softmax(scores, dim=-1)
    return attn @ v


def single_matmul(a, b):
    return a @ b


def chained_matmul(a, b, c):
    left = a @ b
    right = left @ c
    return right


x = torch.randn(2048, 2048, device="cuda")
w = torch.randn(2048, 2048, device="cuda")
b = torch.randn(2048, device="cuda")

q = torch.randn(128, 64, 128, device="cuda")
k = torch.randn(128, 64, 128, device="cuda")
v = torch.randn(128, 64, 128, device="cuda")

a = torch.randn(2048, 2048, device="cuda")
c = torch.randn(2048, 2048, device="cuda")

functions = [
    ("pointwise_chain", pointwise_chain, (x,)),
    ("matmul_bias_relu", matmul_bias_relu, (x, w, b)),
    ("attention_pattern", attention_pattern, (q, k, v)),
    ("single_matmul", single_matmul, (a, w)),
    ("chained_matmul", chained_matmul, (a, w, c)),
]

for name, fn, args in functions:
    compiled = torch.compile(fn)
    compiled(*args)
    print(f"compiled {name}")
