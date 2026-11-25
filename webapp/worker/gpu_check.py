"""M0 check: can the worker container see the NVIDIA GPU? Exit 1 if not."""

import sys

import torch

ok = torch.cuda.is_available()
print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, GPU available: {ok}")
if ok:
    print(f"device: {torch.cuda.get_device_name(0)}, "
          f"VRAM {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB")
    x = torch.randn(1024, 1024, device="cuda")
    print(f"matmul on GPU ok: {bool(torch.isfinite(x @ x).all())}")
sys.exit(0 if ok else 1)
