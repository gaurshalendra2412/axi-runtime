"""
Coordinate-based attention over graph-state tokens (CICO layer).

Attention scores come from (a) queries/keys projected onto a fixed orthonormal
subspace and (b) a bias from structural coordinates (e.g. random-walk
positional encodings). There is NO sequence position anywhere: no RoPE, no
causal mask. Consequences:

  * Exactly permutation-equivariant: shuffle the tokens (and recompute their
    coordinates from the shuffled graph) and the outputs shuffle identically.
    This is tested in python/tests/test_cico_attention.py.
  * It cannot represent word order. Use it for graph/state tokens, not text.
  * It does not by itself prevent attention dilution over many tokens, and
    whether it reduces "context drift" in a trained model is an empirical
    question this file does not answer.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def random_walk_pe(adj: torch.Tensor, k_steps: int) -> torch.Tensor:
    """Random-walk positional encoding: diag((D^-1 A)^t) for t = 1..k_steps.

    adj: [N, N] or [B, N, N] non-negative adjacency. Returns [..., N, k_steps].
    Permutation-equivariant and free of the sign ambiguity of eigenvectors.
    """
    deg = adj.sum(dim=-1).clamp(min=1e-12)
    walk = adj / deg.unsqueeze(-1)
    feats = []
    cur = walk
    for _ in range(k_steps):
        feats.append(torch.diagonal(cur, dim1=-2, dim2=-1))
        cur = cur @ walk
    return torch.stack(feats, dim=-1)


def laplacian_pe(adj: torch.Tensor, k: int) -> torch.Tensor:
    """Lowest non-trivial eigenvectors of the normalized Laplacian. adj: [N, N].

    WARNING: eigenvectors are only defined up to sign (and up to rotation
    inside repeated eigenvalues), so these coordinates are NOT invariant to
    relabeling the nodes. Prefer random_walk_pe, or use sign-invariant
    features of these.
    """
    n = adj.shape[0]
    deg = adj.sum(dim=-1).clamp(min=1e-6)
    d_inv_sqrt = torch.diag(deg.pow(-0.5))
    l_sym = torch.eye(n, device=adj.device, dtype=adj.dtype) - d_inv_sqrt @ adj @ d_inv_sqrt
    _, vecs = torch.linalg.eigh(l_sym)
    return vecs[:, 1 : k + 1]


class DriftInvariantProjection(nn.Module):
    def __init__(self, d_model: int, d_state: int, k_eigen: int = 0):
        super().__init__()
        if d_state > d_model:
            raise ValueError("d_state must be <= d_model")
        self.d_model = d_model
        self.d_state = d_state
        self.k_eigen = k_eigen

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.gamma = nn.Parameter(torch.tensor(0.5))

        # Default: a random orthonormal basis. Replace with calibrate_subspace.
        basis, _ = torch.linalg.qr(torch.randn(d_model, d_state))
        self.register_buffer("basis_state", basis)

    @torch.no_grad()
    def calibrate_subspace(self, anchor_states: torch.Tensor) -> None:
        """Set the basis to the top principal directions of anchor embeddings.
        anchor_states: [N_anchors, d_model]."""
        _, _, vh = torch.linalg.svd(anchor_states, full_matrices=False)
        self.basis_state.copy_(vh[: self.d_state, :].T)

    def forward(
        self,
        h: torch.Tensor,                       # [B, S, d_model]
        coords: torch.Tensor,                  # [B, S, K] structural coordinates
        key_padding_mask: Optional[torch.Tensor] = None,  # [B, S] True = real token
    ) -> torch.Tensor:
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)

        # (q B B^T)(k B B^T)^T == (q B)(k B)^T because B^T B = I.
        qs = q @ self.basis_state
        ks = k @ self.basis_state
        scores = (qs @ ks.transpose(-1, -2)) / math.sqrt(self.d_state)

        diff = coords.unsqueeze(2) - coords.unsqueeze(1)       # [B, S, S, K]
        dist_sq = diff.pow(2).sum(dim=-1)                      # [B, S, S]
        scores = scores - torch.abs(self.gamma) * dist_sq

        if key_padding_mask is not None:
            scores = scores.masked_fill(~key_padding_mask.unsqueeze(1), float("-inf"))

        weights = F.softmax(scores, dim=-1)
        return self.out_proj(weights @ v)
