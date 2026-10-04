"""Run on a GPU machine (e.g. Google Colab T4):  python python/tests/gpu_check.py"""
import torch

from axi.kernels.block_sparse_attention import HAS_TRITON
from test_block_sparse_attention import run_kernel_checks
from test_block_sparse_attention_backward import run_backward_checks

if not torch.cuda.is_available() or not HAS_TRITON:
    raise SystemExit("Needs a CUDA GPU and Triton installed.")

print("GPU:", torch.cuda.get_device_name(0))
print("--- forward vs reference ---")
fwd = run_kernel_checks(verbose=True)
print(f"FORWARD PASSED. Worst max-abs error: {fwd:.5f}")
print("--- backward vs PyTorch autograd ---")
bwd = run_backward_checks(verbose=True)
print(f"BACKWARD PASSED. Worst max-abs error: {bwd:.5f}")
