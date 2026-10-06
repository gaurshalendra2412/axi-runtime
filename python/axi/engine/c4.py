"""Layer 4 reference: C4 (90-degree rotation) canonical coordinates + rotation-invariant attention (numpy).

Facts established by tests/test_c4.py:
  * cell -> (orbit representative, phase) is a bijection (nothing is lost)
  * rotating the grid leaves every (rep, payload) unchanged and adds +1 (mod 4) to every phase,
    EXCEPT the fixed center cell of odd grids (phase stays 0)  -> the 4-tuple stream itself is NOT invariant
  * attention that uses only RELATIVE phase (and canon coords) is exactly rotation-invariant,
    provided pairs involving a fixed center cell use no phase rotation.
"""
import numpy as np


def rot(n, i, j, k):
    k %= 4
    return [(i, j), (j, n - 1 - i), (n - 1 - i, n - 1 - j), (n - 1 - j, i)][k]


def canonicalize(n, i, j):
    """-> (rep_i, rep_j, phase, fixed). fixed=True only for the center of an odd grid."""
    orb = [rot(n, i, j, k) for k in range(4)]
    rep = min(orb)
    phase = next(k for k in range(4) if rot(n, rep[0], rep[1], k) == (i, j))
    return rep[0], rep[1], phase, len(set(orb)) == 1


def rotate_grid(n, grid):
    out = np.zeros_like(grid)
    for i in range(n):
        for j in range(n):
            a, b = rot(n, i, j, 1); out[a, b] = grid[i, j]
    return out


def compile_grid(grid, eps=0.0):
    """Dense n x n -> token list sorted by (rep, phase): (ri, rj, phase, fixed, value). Explicit-zero cells are dropped."""
    n = grid.shape[0]; toks = []
    for i in range(n):
        for j in range(n):
            v = grid[i, j]
            if abs(v) > eps:
                ri, rj, ph, fx = canonicalize(n, i, j); toks.append((ri, rj, ph, fx, float(v)))
    return sorted(toks)


class RelPhaseAttention:
    """Single-head attention. score_ij = q_i . R(theta_ij) k_j,  theta_ij = (phase_j - phase_i)*pi/2,
    theta_ij = 0 if either token is a fixed center. Values/payload carry no position."""
    def __init__(self, d_in, d_head, grid_dim, seed=0):
        r = np.random.default_rng(seed); s = d_in ** -0.5
        self.Wq, self.Wk, self.Wv = (r.normal(0, s, (d_in, d_head)) for _ in range(3))
        self.Ei = r.normal(0, 1, (grid_dim, d_head // 2)); self.Ej = r.normal(0, 1, (grid_dim, d_head // 2))
        self.d = d_head

    def _pos(self, ci, cj): return np.concatenate([self.Ei[ci], self.Ej[cj]], -1)

    def forward(self, x, ci, cj, phase, fixed, relative=True):
        pos = self._pos(ci, cj); q = x @ self.Wq + pos; k = x @ self.Wk + pos; v = x @ self.Wv
        h = self.d // 2; q1, q2, k1, k2 = q[:, :h], q[:, h:], k[:, :h], k[:, h:]
        A = q1 @ k1.T + q2 @ k2.T; B = q2 @ k1.T - q1 @ k2.T
        if relative:
            th = (phase[None, :] - phase[:, None]) * (np.pi / 2)
            th = np.where(fixed[:, None] | fixed[None, :], 0.0, th)
        else:   # "pasted" behaviour: every token incl. the center rotated by its own absolute phase
            th = (phase[None, :] - phase[:, None]) * (np.pi / 2)
        s = (np.cos(th) * A + np.sin(th) * B) / np.sqrt(self.d)
        s = s - s.max(1, keepdims=True); w = np.exp(s); w /= w.sum(1, keepdims=True)
        return w @ v
