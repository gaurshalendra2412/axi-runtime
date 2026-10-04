"""
Backward-pass tests. The GPU test compares the Triton kernels' gradients with
PyTorch autograd run on the reference implementation (fp32). It is skipped
automatically without CUDA/Triton. The CPU test checks the safety validator.
"""

import torch

from axi.kernels.block_sparse_attention import (
    HAS_TRITON,
    assert_unique_blocks,
    block_sparse_attention_train,
    reference_block_sparse_attention,
)
from test_block_sparse_attention import make_case

CONFIGS = [
    # (B, H, M, D, lens_per_batch)
    (2, 2, 17, 64, [[64, 33], [10]]),
    (3, 2, 100, 64, [[64, 5], [], [64, 64, 1]]),  # includes an empty sequence
    (1, 4, 64, 128, [[64] * 6]),
]


def test_unique_block_validator():
    table = torch.tensor([[3, 5], [5, 1]], dtype=torch.int32)
    active = torch.tensor([2, 2], dtype=torch.int32)
    raised = False
    try:
        assert_unique_blocks(table, active)
    except ValueError:
        raised = True
    assert raised, "duplicate physical block should be rejected"

    ok_table = torch.tensor([[3, 5], [7, 1]], dtype=torch.int32)
    assert_unique_blocks(ok_table, active)  # must not raise

    # a duplicate sitting beyond num_active must be ignored
    pad_table = torch.tensor([[3, 9], [7, 9]], dtype=torch.int32)
    assert_unique_blocks(pad_table, torch.tensor([1, 1], dtype=torch.int32))


def _skip(reason):
    try:
        import pytest
    except ImportError:
        print(f"SKIPPED: {reason}")
        return True
    pytest.skip(reason)


def run_backward_checks(verbose=False):
    worst = 0.0
    for idx, (B, H, M, D, lens) in enumerate(CONFIGS):
        q, kp, vp, bt, bl, na = make_case(
            B, H, M, D, lens, device="cuda", dtype=torch.float16, seed=100 + idx)
        do = torch.randn(B, H, M, D, device="cuda", dtype=torch.float16)

        # Kernel gradients
        q_k, kp_k, vp_k = (t.clone().requires_grad_(True) for t in (q, kp, vp))
        out_k = block_sparse_attention_train(q_k, kp_k, vp_k, bt, bl, na)
        out_k.backward(do)
        assert not torch.isnan(q_k.grad).any(), f"NaN in dq, config {idx}"
        kernel_grads = [q_k.grad.float().cpu(), kp_k.grad.float().cpu(), vp_k.grad.float().cpu()]

        # Reference gradients: autograd on the fp32 reference, same inputs
        q_r, kp_r, vp_r = (t.float().cpu().requires_grad_(True) for t in (q, kp, vp))
        out_r = reference_block_sparse_attention(q_r, kp_r, vp_r, bt.cpu(), bl.cpu(), na.cpu())
        grads = torch.autograd.grad(
            out_r, [q_r, kp_r, vp_r], do.float().cpu(), allow_unused=True)
        ref_grads = [g if g is not None else torch.zeros_like(t)
                     for g, t in zip(grads, (q_r, kp_r, vp_r))]

        for name, gk, gr in zip(("dq", "dk_pool", "dv_pool"), kernel_grads, ref_grads):
            err = (gk - gr).abs().max().item()
            worst = max(worst, err)
            if verbose:
                print(f"config {idx} {name}: max_abs_err={err:.5f}  (ref max |g|={gr.abs().max().item():.3f})")
            assert torch.allclose(gk, gr, atol=3e-2, rtol=3e-2), (
                f"{name} mismatch in config {idx}: max err {err}")
    return worst


def test_backward_matches_autograd_on_gpu():
    if not torch.cuda.is_available():
        return _skip("CUDA not available")
    if not HAS_TRITON:
        return _skip("Triton not installed")
    run_backward_checks()


if __name__ == "__main__":
    test_unique_block_validator()
    print("CPU validator test: PASSED")
    test_backward_matches_autograd_on_gpu()
