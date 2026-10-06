import numpy as np
from axi.engine.c4 import rot, canonicalize, rotate_grid, compile_grid, RelPhaseAttention


def test_rotation_group_laws():
    for n in range(1, 9):
        for i in range(n):
            for j in range(n):
                assert rot(n, i, j, 4) == (i, j)
                assert rot(n, *rot(n, i, j, 1), 1) == rot(n, i, j, 2)


def test_cell_to_token_is_a_bijection():
    for n in range(1, 10):
        seen = {canonicalize(n, i, j)[:3] for i in range(n) for j in range(n)}
        assert len(seen) == n * n, n


def test_only_center_of_odd_grids_is_fixed():
    for n in range(1, 10):
        fixed = [(i, j) for i in range(n) for j in range(n) if canonicalize(n, i, j)[3]]
        assert fixed == ([((n - 1) // 2, (n - 1) // 2)] if n % 2 else [])


def test_rotation_keeps_rep_and_payload_and_shifts_phase_by_one():
    rng = np.random.default_rng(0)
    for n in (4, 5, 6, 7):
        g = rng.integers(0, 3, (n, n)).astype(float) * (rng.random((n, n)) < .6)
        a = compile_grid(g); b = compile_grid(rotate_grid(n, g))
        assert len(a) == len(b)
        da = {(t[0], t[1], t[2]): t for t in a}
        for (ri, rj, ph, fx, v) in b:
            src_phase = ph if fx else (ph - 1) % 4
            assert (ri, rj, src_phase) in da and da[(ri, rj, src_phase)][4] == v and da[(ri, rj, src_phase)][3] == fx


def test_pasted_test_was_too_weak_full_tokens_are_not_invariant():
    n = 4; g = np.zeros((n, n)); g[0, 1] = 42
    a, b = compile_grid(g), compile_grid(rotate_grid(n, g))
    assert a[0][:2] == b[0][:2] and a[0][4] == b[0][4]      # what the pasted Rust test checked
    assert a[0][2] != b[0][2]                               # but the phase differs, so the token differs


def _tokens(n, g):
    t = compile_grid(g)
    return (np.array([x[4] for x in t])[:, None], np.array([x[0] for x in t]), np.array([x[1] for x in t]),
            np.array([x[2] for x in t]), np.array([x[3] for x in t], bool), t)


def _key(t): return [(x[0], x[1], x[4]) for x in t]  # identity of a token independent of phase


def _by_cell(n, att, grid, relative=True):
    x, ci, cj, ph, fx, t = _tokens(n, grid)
    out = att.forward(x, ci, cj, ph, fx, relative=relative)
    return {rot(n, tt[0], tt[1], tt[2]): out[i] for i, tt in enumerate(t)}   # token -> the grid cell it came from


def test_relative_phase_attention_is_exactly_rotation_equivariant_even_grids():
    for n in (4, 6, 8):
        rng = np.random.default_rng(n)
        g = rng.integers(1, 5, (n, n)).astype(float) * (rng.random((n, n)) < .5)
        att = RelPhaseAttention(1, 8, n, seed=1)
        base = _by_cell(n, att, g); cur = g
        for k in (1, 2, 3):
            cur = rotate_grid(n, cur); out = _by_cell(n, att, cur)
            assert set(out) == {rot(n, i, j, k) for (i, j) in base}
            for (i, j), v in base.items():                     # output at the rotated cell == output at the original cell
                assert np.allclose(out[rot(n, i, j, k)], v, atol=1e-10), (n, k, i, j)


def test_odd_grid_center_breaks_pasted_style_but_not_fixed_version():
    n = 5; rng = np.random.default_rng(3)
    g = rng.integers(1, 5, (n, n)).astype(float); g[2, 2] = 9.0   # make the center token identifiable
    att = RelPhaseAttention(1, 8, n, seed=2)
    res = {}
    for rel in (False, True):
        o = []
        for gr in (g, rotate_grid(n, g)):
            x, ci, cj, ph, fx, t = _tokens(n, gr)
            out = att.forward(x, ci, cj, ph, fx, relative=rel)
            c = [i for i, tt in enumerate(t) if tt[3]][0]       # center token's output
            o.append(out[c])
        res[rel] = np.abs(o[0] - o[1]).max()
    print("   center-token change under rotation: pasted-style %.3g, relative+fixed-handling %.3g" % (res[False], res[True]))
    assert res[False] > 1e-3 and res[True] < 1e-10


def test_relative_form_equals_rotating_q_and_k_separately_for_free_tokens():
    rng = np.random.default_rng(0); d = 8; T = 7
    q = rng.normal(size=(T, d)); k = rng.normal(size=(T, d)); ph = rng.integers(0, 4, T)
    def R(x, p):
        th = p[:, None] * np.pi / 2; h = d // 2; x1, x2 = x[:, :h], x[:, h:]
        return np.concatenate([x1 * np.cos(th) - x2 * np.sin(th), x1 * np.sin(th) + x2 * np.cos(th)], 1)
    pasted = R(q, ph) @ R(k, ph).T
    h = d // 2; A = q[:, :h] @ k[:, :h].T + q[:, h:] @ k[:, h:].T; B = q[:, h:] @ k[:, :h].T - q[:, :h] @ k[:, h:].T
    th = (ph[None, :] - ph[:, None]) * np.pi / 2
    assert np.allclose(pasted, np.cos(th) * A + np.sin(th) * B)


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("PASS", n)
