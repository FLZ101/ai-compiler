PYTHON ?= python3
VENV ?= .venv
VENV_PIP ?= $(VENV)/bin/pip
VENV_PYTHON ?= $(VENV)/bin/python
VENV_BLACK ?= $(VENV)/bin/black
VENV_PRE_COMMIT ?= $(VENV)/bin/pre-commit
VENV_MARKER ?= $(VENV)/.marker

# Detect the CUDA toolkit version and derive the PyTorch wheel tag. PyTorch
# publishes wheels per CUDA release line (CUDA 13 -> cu130), not per toolkit
# minor/patch. Override PYTORCH_CUDA_TAG for a different CUDA line.
NVCC ?= $(shell command -v nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)
CUDA_MAJOR := $(shell $(NVCC) --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\.[0-9]*.*/\1/p')
PYTORCH_CUDA_TAG ?= $(if $(filter 12,$(CUDA_MAJOR)),cu121,cu$(CUDA_MAJOR)0)
PYTORCH_INDEX_URL ?= https://download.pytorch.org/whl/$(PYTORCH_CUDA_TAG)
ifeq ($(PYTORCH_CUDA_TAG),cu121)
TORCH_VERSION ?= 2.5.1+$(PYTORCH_CUDA_TAG)
TRITON_VERSION ?= 3.1.0
PTXAS ?= $(shell command -v ptxas 2>/dev/null || echo /usr/local/cuda/bin/ptxas)
TRITON_ENV := TRITON_PTXAS_PATH=$(PTXAS)
PIP_INDEX_ARGS := --index-url $(PYTORCH_INDEX_URL) --extra-index-url https://pypi.org/simple
else
TORCH_VERSION ?= 2.14.1+$(PYTORCH_CUDA_TAG)
TRITON_VERSION ?= 3.8.0
TRITON_ENV :=
PIP_INDEX_ARGS := --index-url $(PYTORCH_INDEX_URL)
endif

SCRIPT ?=

.PHONY: venv check shell run torchao format-python precommit clean help

$(VENV_MARKER):
	$(PYTHON) -m venv $(VENV)
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install "torch==$(TORCH_VERSION)" "triton==$(TRITON_VERSION)" $(PIP_INDEX_ARGS)
	$(VENV_PIP) install -r requirements-dev.txt
	$(VENV_PRE_COMMIT) install
	touch "$(VENV_MARKER)"

venv: $(VENV_MARKER)

check: venv
	@nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
	$(VENV_PYTHON) -c 'import torch; print(f"torch {torch.__version__}, cuda {torch.version.cuda}, available {torch.cuda.is_available()}, device {torch.cuda.get_device_name(0)}")'
	$(VENV_PYTHON) -c 'import triton; print(f"triton {triton.__version__}")'

shell: venv
	$(VENV_PYTHON)

run: venv
	@test -n "$(SCRIPT)" || { echo "usage: make run SCRIPT=<path.py>"; exit 1; }
	$(TRITON_ENV) $(VENV_PYTHON) $(SCRIPT)

torchao: venv
	$(VENV_PIP) install torchao

format-python: venv
	find . -path ./.venv -prune -o -name '*.py' -print0 | xargs -0r $(VENV_BLACK)

precommit: venv
	$(VENV_PRE_COMMIT) run --all-files

clean:
	rm -rf $(VENV) .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

help:
	@echo "Targets:"
	@echo "  make venv          Bootstrap .venv and install all dependencies"
	@echo "  make check         Verify the CUDA/PyTorch/Triton environment"
	@echo "  make shell         Open a shell with the venv Python"
	@echo "  make run           Run a study script: make run SCRIPT=00_baseline.py"
	@echo "  make torchao       Install TorchAO for the quantization tasks"
	@echo "  make format-python Format Python sources with black from .venv"
	@echo "  make precommit      Run pre-commit on all files"
	@echo "  make clean      Remove the virtualenv and Python caches"
