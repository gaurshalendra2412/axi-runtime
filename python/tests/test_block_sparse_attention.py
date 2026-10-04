"""
CPU tests check the PyTorch reference against dense attention (run in CI).
The GPU test checks the Triton kernel against the reference; it is skipped
automatically when CUDA/Triton are unavailable. Run it on a GPU (e.g. Colab).
"""

import torch
import torch.nn.functional as F

from axi.kernels.block_sparse_attention import (
    BLOCK_N,
    HAS_TRITON,
    block_sparse_attention,
    reference_block_sparse_attention,
)


def make_case(B, H, M, D, lens_per_batch, device="cpu", dtype=torch.float32, seed=0):
    """lens_per_batch[b] = list of valid-token counts, one per active block.
    Padding slots of partially filled blocks are filled with large garbage so
    that any kernel that fails to mask them gives a wrong answer."""
    max_active = max(1, max(len(x) for x in lens_per_batch))
    total_blocks = B * max_active + 2
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, H, M, D, generator=g)
    k_pool = torch.randn(total_blocks, BLOCK_N, H, D, generator=g)
    v_pool = torch.randn(total_blocks, BLOCK_N, H, D, generator=g)
    perm = torch.randperm(total_blocks, generator=g)

    block_table = torch.zeros(B, max_active, dtype=torch.int32)
    block_lens = torch.zeros(B, max_active, dtype=torch.int32)
    num_active = torch.zeros(B, dtype=torch.int32)
    for b, lens in enumerate(lens_per_batch):
        num_active[b] = len(lens)
        for i, ln in enumerate(lens):
            blk = int(perm[b * max_active + i])
            block_table[b, i] = blk
            block_lens[b, i] = ln
            if ln < BLOCK_N:
                k_pool[blk, ln:] = 1e4
                v_pool[blk, ln:] = 1e4
    to = lambda t: t.to(device=device, dtype=dtype)
    return (
        to(q), to(k_pool), to(v_pool),
        block_table.to(device), block_lens.to(device), num_active.to(device),
    )


def _dense(q, k_pool, v_pool, block_table, block_lens, num_active, b):
    ks = [k_pool[int(block_table[b, i]), : int(block_lens[b, i])] for i in range(int(num_active[b]))]
    vs = [v_pool[int(block_table[b, i]), : int(block_lens[b, i])] for i in range(int(num_active[b]))]
    k = torch.cat(ks, 0).permute(1, 0, 2).unsqueeze(0)  # [1, H, L, D]
    v = torch.cat(vs, 0).permute(1, 0, 2).unsqueeze(0)
    return F.scaled_dot_product_attention(q[b : b + 1], k, v)[0]


def test_reference_matches_dense_attention():
    args = make_case(B=2, H=3, M=5, D=16, lens_per_batch=[[64, 64, 64], [64, 20]])
    out = reference_block_sparse_attention(*args)
    for b in range(2):
        expected = _dense(*args, b)
        assert torch.allclose(out[b], expected, atol=1e-5, rtol=1e-4)


def test_reference_zero_active_blocks_gives_zeros():
    args = make_case(B=2, H=2, M=3, D=16, lens_per_batch=[[64], []])
    out = reference_block_sparse_attention(*args)
    assert torch.count_nonzero(out[1]) == 0
    assert not torch.isnan(out).any()


def _skip(reason):
    try:
        import pytest
    except ImportError:
        print(f"SKIPPED: {reason}")
        return True
    pytest.skip(reason)


CONFIGS = [
    # (B, H, M, D, lens_per_batch)
    (2, 4, 1, 64, [[64, 64, 33], [64]]),
    (2, 4, 17, 64, [[10], [64, 64, 64, 64]]),
    (3, 2, 100, 128, [[64, 5], [], [64, 64, 1]]),  # includes an empty sequence
    (1, 8, 64, 64, [[64] * 12]),
]


def run_kernel_checks(verbose=False):
    worst = 0.0
    for idx, (B, H, M, D, lens) in enumerate(CONFIGS):
        args = make_case(B, H, M, D, lens, device="cuda", dtype=torch.float16, seed=idx)
        out = block_sparse_attention(*args).float().cpu()
        cpu_args = [t.cpu() for t in args]
        ref = reference_block_sparse_attention(*cpu_args).float()
        assert not torch.isnan(out).any(), f"NaN in kernel output, config {idx}"
        err = (out - ref).abs().max().item()
        worst = max(worst, err)
        if verbose:
            print(f"config {idx}: B={B} H={H} M={M} D={D} max_abs_err={err:.5f}")
        assert torch.allclose(out, ref, atol=1e-2, rtol=1e-2), (
            f"kernel != reference for config {idx}, max err {err}"
        )
    return worst


def test_kernel_matches_reference_on_gpu():
    if not torch.cuda.is_available():
        return _skip("CUDA not available")
    if not HAS_TRITON:
        return _skip("Triton not installed")
    run_kernel_checks()


if __name__ == "__main__":
    test_reference_matches_dense_attention()
    test_reference_zero_active_blocks_gives_zeros()
    print("CPU reference tests: PASSED")
    test_kernel_matches_reference_on_gpu()
