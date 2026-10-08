"""Implements concept-map row 152 (14th scan, SECONDHAND): text pasted by the user on 8 Oct 2026, written by an AI in an earlier chat and naming the Medium piece
"Time over AXI" (the "169-dot Bhairavi mirror"). The five Medium pages named in that text (three handles that are not in our account notes) return
"410 Gone" from here, so NOTHING below was checked against the pages. The pasted text says, in its own words:
  * a fixed 13 x 13 coordinate system of 169 cells, with "perimeter + internal nodes" (boundary and inside);
  * "13 deliberately breaks 12-based / power-of-2 measurement" (an extra cell, "the 13th floor");
  * a 3 x 3 seed can be expanded to 13 x 13 "preserving symmetry and zero drift".
Those three sentences contain counting claims, and counting can be checked without the pages. Checked here by brute force (no closed forms are used in the asserts
except where a test says so):
  1. Symmetry classes of the 13 x 13 board under the eight symmetries (D4): 28 classes (sizes 1, 4 and 8: one centre, twelve classes of four on the axes and
     diagonals, fifteen classes of eight); under rotations only (C4): 43. The ring of 48 boundary cells holds 7 classes and the 121 inside cells 21.
     Only the centre cell (6, 6) is fixed by all eight.
  2. "Breaks 12-based measurement", in checkable form: 13 is prime, so only a 1 x 1 or a 13 x 13 seed can be enlarged to 13 x 13 by repeating cells (a 12-board is reached from seeds of size 2, 3, 4, 6; a 16-board from 2, 4, 8).
     So a 3 x 3 seed cannot be enlarged by repeating cells (13 is not a multiple of 3).
  3. "Expand a 3 x 3 seed to 13 x 13 preserving symmetry": of all 121 places a 3 x 3 block can sit in the 13 x 13 board, exactly ONE commutes with all eight symmetries
     (the centred one, 5 cells of margin each side), and copying it out again returns the seed exactly. More generally an n x n seed has a symmetric place iff n is odd:
     a 12 x 12 world cannot sit symmetrically inside 13 x 13. The extra "13th" row and column added to a 12 x 12 board is an L-shape of 25 cells whose only symmetries
     are the identity and the transpose: the 13th line is exactly what breaks the 12-board's symmetry.
  4. The numerology guard: 169 = 12^2 + 5^2 because the L-shape 2*12+1 = 25 is a square. That happens for n = 4, 12, 24, 40, 60, 84 (the odd-leg Pythagorean triples), so 13 is
     the second of an unbounded list and is not singled out by the count. Do not present 12^2 + 5^2 as evidence for anything.
  5. The rotation code (axi/engine/c4.py), tested before only up to 9 x 9, behaves the same on 13 x 13 (bijection to tokens, one fixed centre, +1 phase per quarter turn).
What these do NOT show: that a 13 x 13 board gives a model "higher reasoning density" or "lower drift" (no run measures it: that is the A/B harness, not started); that
the page's "Bhairavi mirror" is this structure (the page was not read); that 13 is better than any other odd size (every odd n has one centre, a ring and an inside; 11 and 15
behave the same way). The gate in this repository has no board at all (test_board_vocabulary.py), so none of this is enforced by the engine."""
import math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.d4 import apply as d4apply, IDS
from axi.engine.c4 import canonicalize, rotate_grid, compile_grid

N = 13


def perms(n):
    idx = np.arange(n * n).reshape(n, n)
    return [tuple(int(x) for x in d4apply(idx, t).ravel()) for t in IDS]


def orbits(n, cells=None, group=None):
    """Partition of cell indices into orbits of the given list of permutations (default all eight)."""
    group = group or perms(n); cells = set(range(n * n)) if cells is None else set(cells); out = []
    while cells:
        i = min(cells); orb = {p[i] for p in group}; out.append(orb); cells -= orb
    return out


def test_symmetry_classes_of_the_13_board():
    P = perms(N)
    assert len(set(P)) == 8                                                   # eight distinct symmetries
    orb = orbits(N)
    assert len(orb) == 28 == (7 * 8) // 2                                     # T(7)
    sizes = sorted(len(o) for o in orb)
    assert {s: sizes.count(s) for s in set(sizes)} == {1: 1, 4: 12, 8: 15} and sum(sizes) == 169
    fixed = [i for i in range(N * N) if all(p[i] == i for p in P)]
    assert fixed == [6 * N + 6]                                               # the centre (6, 6) is the only cell fixed by all eight
    ring = [r * N + c for r in range(N) for c in range(N) if r in (0, N - 1) or c in (0, N - 1)]
    inside = sorted(set(range(N * N)) - set(ring))
    assert (len(ring), len(inside)) == (48, 121)
    assert (len(orbits(N, ring)), len(orbits(N, inside))) == (7, 21)          # the ring holds 7 classes, the inside T(6) = 21
    # rotations only (C4): the cyclic subgroup {identity, three quarter turns}
    rots = [P[t] for t in range(4)]
    assert len(orbits(N, group=rots)) == 43 == (169 + 1 + 1 + 1) // 4         # Burnside: only the centre is fixed by a quarter turn


def test_thirteen_has_no_block_tilings_and_a_small_seed_has_one_symmetric_place():
    def seeds_reaching(n):                                                    # seed sizes s for which repeating every cell f x f times (f >= 1) gives exactly n x n
        return [s for s in range(1, n + 1) if any(np.kron(np.ones((s, s), dtype=int), np.ones((f, f), dtype=int)).shape == (n, n) for f in range(1, n + 1))]
    assert seeds_reaching(13) == [1, 13] and seeds_reaching(12) == [1, 2, 3, 4, 6, 12] and seeds_reaching(16) == [1, 2, 4, 8, 16]
    assert 13 % 3 == 1                                                        # a 3 x 3 seed cannot be repeated up to 13

    def embed(seed, r0, c0):
        g = np.zeros((N, N), dtype=int); k = seed.shape[0]; g[r0:r0 + k, c0:c0 + k] = seed; return g

    def symmetric_places(k):
        seed = np.arange(1, k * k + 1).reshape(k, k)                          # all labels distinct, so any asymmetry shows
        ok = []
        for r0 in range(N - k + 1):
            for c0 in range(N - k + 1):
                if all((embed(d4apply(seed, t), r0, c0) == d4apply(embed(seed, r0, c0), t)).all() for t in IDS): ok.append((r0, c0))
        return ok
    assert symmetric_places(3) == [(5, 5)]                                    # 121 places, one symmetric: 5 cells of margin each side
    assert {k: len(symmetric_places(k)) for k in range(1, N + 1)} == {k: (1 if k % 2 else 0) for k in range(1, N + 1)}
    assert symmetric_places(13) == [(0, 0)] and symmetric_places(12) == []    # the 12 world cannot sit symmetrically in the 13 board
    seed = np.arange(1, 10).reshape(3, 3)
    assert (embed(seed, 5, 5)[5:8, 5:8] == seed).all() and embed(seed, 5, 5).sum() == seed.sum()      # copy out returns the seed, nothing else is added
    mask = np.zeros((N, N), dtype=int); mask[12, :] = 1; mask[:, 12] = 1      # the 13th row and column added to a 12 x 12 board
    assert mask.sum() == 25
    stab = [t for t in IDS if (d4apply(mask, t) == mask).all()]
    assert len(stab) == 2 and 0 in stab and not ({1, 2, 3} & set(stab))       # identity and one reflection (the transpose); no rotation keeps the L-shape


def test_169_equals_12_squared_plus_5_squared_is_not_special_to_13():
    assert 13 ** 2 == 12 ** 2 + 5 ** 2 == 144 + 25 and 2 * 12 + 1 == 25 == 5 ** 2
    hits = [n for n in range(1, 101) if math.isqrt(2 * n + 1) ** 2 == 2 * n + 1]
    leg_triples = [(math.isqrt(2 * n + 1), n, n + 1) for n in hits]
    assert hits == [4, 12, 24, 40, 60, 84]                                    # 13 is the second member of a list that does not stop
    assert leg_triples[:3] == [(3, 4, 5), (5, 12, 13), (7, 24, 25)]


def test_c4_code_holds_on_the_13_board():
    seen = {canonicalize(N, i, j)[:3] for i in range(N) for j in range(N)}
    assert len(seen) == N * N                                                 # cell -> (representative, phase) is a bijection
    fixed = [(i, j) for i in range(N) for j in range(N) if canonicalize(N, i, j)[3]]
    assert fixed == [(6, 6)]
    rng = np.random.default_rng(13); g = rng.integers(1, 5, size=(N, N))
    a, b = compile_grid(g), compile_grid(rotate_grid(N, g))
    assert len(a) == len(b) == 169
    by_rep_a = {}; by_rep_b = {}
    for ri, rj, ph, fx, v in a: by_rep_a.setdefault((ri, rj), []).append((ph, fx, v))
    for ri, rj, ph, fx, v in b: by_rep_b.setdefault((ri, rj), []).append((ph, fx, v))
    assert by_rep_a.keys() == by_rep_b.keys() and len(by_rep_a) == 43
    for rep in by_rep_a:
        shifted = sorted(((ph + (0 if fx else 1)) % 4, fx, v) for ph, fx, v in by_rep_a[rep])
        assert shifted == sorted(by_rep_b[rep]), rep


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
