"""Property tests for the CICO attention layer (CPU, fast)."""

import torch
import torch.nn.functional as F

from axi.engine.cico_attention import DriftInvariantProjection, random_walk_pe

B, N, D, DS, K = 2, 7, 32, 8, 4


def _random_graph(seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.rand(B, N, N, generator=g)
    adj = (a + a.transpose(-1, -2)) / 2
    return adj * (1 - torch.eye(N))  # symmetric, zero diagonal


def _layer(seed=0):
    torch.manual_seed(seed)
    return DriftInvariantProjection(D, DS, k_eigen=K).eval()


def test_random_walk_pe_is_permutation_equivariant():
    adj = _random_graph()
    perm = torch.tensor([3, 0, 6, 1, 5, 2, 4])
    coords = random_walk_pe(adj, K)
    coords_perm = random_walk_pe(adj[:, perm][:, :, perm], K)
    assert torch.allclose(coords_perm, coords[:, perm], atol=1e-5)


def test_layer_is_permutation_equivariant():
    layer = _layer()
    adj = _random_graph()
    h = torch.randn(B, N, D, generator=torch.Generator().manual_seed(1))
    perm = torch.tensor([3, 0, 6, 1, 5, 2, 4])

    with torch.no_grad():
        out = layer(h, random_walk_pe(adj, K))
        out_perm = layer(h[:, perm], random_walk_pe(adj[:, perm][:, :, perm], K))
    assert torch.allclose(out_perm, out[:, perm], atol=1e-4)


def test_causal_attention_is_not_order_invariant():
    """Contrast: standard causal attention changes when tokens are reordered."""
    x = torch.randn(1, N, D, generator=torch.Generator().manual_seed(2))
    rev = torch.arange(N - 1, -1, -1)
    out = F.scaled_dot_product_attention(x, x, x, is_causal=True)
    out_rev = F.scaled_dot_product_attention(x[:, rev], x[:, rev], x[:, rev], is_causal=True)
    assert not torch.allclose(out_rev, out[:, rev], atol=1e-3)


def test_basis_is_orthonormal_and_projection_is_idempotent():
    layer = _layer()
    basis = layer.basis_state
    assert torch.allclose(basis.T @ basis, torch.eye(DS), atol=1e-5)
    proj = basis @ basis.T
    assert torch.allclose(proj @ proj, proj, atol=1e-5)


def test_fast_scores_equal_explicit_projection():
    layer = _layer()
    h = torch.randn(B, N, D, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        q, k = layer.q_proj(h), layer.k_proj(h)
        basis = layer.basis_state
        explicit = ((q @ basis) @ basis.T) @ (((k @ basis) @ basis.T).transpose(-1, -2))
        fast = (q @ basis) @ (k @ basis).transpose(-1, -2)
    assert torch.allclose(fast, explicit, atol=1e-4, rtol=1e-4)


def test_padding_mask_blocks_masked_keys():
    layer = _layer()
    adj = _random_graph()
    coords = random_walk_pe(adj, K)
    h = torch.randn(B, N, D, generator=torch.Generator().manual_seed(4))
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[:, -2:] = False

    h2 = h.clone()
    h2[:, -2:] = torch.randn(B, 2, D, generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
        a = layer(h, coords, key_padding_mask=mask)
        b = layer(h2, coords, key_padding_mask=mask)
    assert torch.allclose(a[:, :-2], b[:, :-2], atol=1e-6)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASSED", name)
