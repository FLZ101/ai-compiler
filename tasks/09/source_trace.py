from pathlib import Path

import torch

# PyTorch installed in this venv is a wheel. The Python source we need to trace
# is under site-packages/torch, with platform-specific code under lib64 on this
# system.
torch_root = Path(torch.__file__).resolve().parent
print(f"torch version: {torch.__version__}")
print(f"torch root:    {torch_root}")

# These files are the main stops along the compiler route. Read them in this
# order, from Python tracing to generated Triton source.
route_files = [
    "_dynamo/output_graph.py",
    "_dynamo/guards.py",
    "_dynamo/eval_frame.py",
    "_functorch/aot_autograd.py",
    "_inductor/compile_fx.py",
    "_inductor/graph.py",
    "_inductor/scheduler.py",
    "_inductor/codegen/triton.py",
    "_inductor/codegen/wrapper.py",
]

print("\ncompiler source route:")
for relative in route_files:
    path = torch_root / relative
    status = "ok" if path.exists() else "missing"
    print(f"  {status:7s} {path}")
