"""Implements concept-map row 64 (blog 14 Aug, "The Permutations of the Nine: Mapping the Emerging Market Landscape").
What the post does, as far as I could read it (one read, second read pending): names nine AI products, uses four 'market vectors'
(compute, distribution, ideology, capital) and computes NO count and states NO symmetry. So this file is ENGINEERING SUPPORT, not
Rajnish's claim: it supplies the count the title points at, for the grid the runtime actually uses.

What these tests show (3x3 grid, the 8 D4 transforms taken from axi.engine.d4, not re-derived):
  1. Nine DISTINCT labels on the 3x3 grid can be arranged in 9! = 362,880 ways. D4 acts FREELY on them (no non-identity transform
     fixes any such arrangement), so the canonical form collapses them into exactly 9!/8 = 45,360 classes of exactly 8 members.
  2. With REPEATED colours (ARC-style) symmetric grids have non-trivial stabilizers, so the saving is a little less than 8x.
     Brute force with the repo's own canonical() for 2 and 3 colours, and Burnside's count from the repo's own cell permutations for 10:
         colours  grids        D4 classes     saving
         2        512          102            5.0x     (symmetric grids are common when there are few colours)
         3        19,683       2,862          6.9x
         10       10^9         125,512,750    7.97x
  3. The 8 cell permutations really are a group of order 8 (closed under composition), so 'divide by 8' is legitimate.
What they do NOT show: that the post means D4 at all; that fewer distinct prompts help a model (that is an experiment); anything about
the nine products."""
import os, random, sys
from collections import Counter
from itertools import permutations, product
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.d4 import apply, canonical, IDS

IDX = np.arange(9).reshape(3, 3)
PERMS = [tuple(int(x) for x in apply(IDX, t).ravel()) for t in IDS]      # new position p shows old cell PERMS[t][p]


def act(a, perm): return tuple(a[perm[p]] for p in range(9))


def cycles(perm):
    seen, c = set(), 0
    for s in range(9):
        if s not in seen:
            c += 1
            while s not in seen: seen.add(s); s = perm[s]
    return c


def test_the_eight_cell_permutations_form_a_group():
    assert len(set(PERMS)) == 8
    comp = lambda p, q: tuple(p[q[i]] for i in range(9))
    assert all(comp(p, q) in set(PERMS) for p in PERMS for q in PERMS)
    assert sum(1 for p in PERMS if p == tuple(range(9))) == 1                 # only t = 0 is the identity


def test_nine_distinct_labels_9_factorial_over_8():
    classes = Counter()
    for a in permutations(range(9)):
        classes[min(act(a, p) for p in PERMS)] += 1
    assert len(classes) == 45360 == 362880 // 8
    assert set(classes.values()) == {8}                                      # every class has exactly 8 members: the action is free
    rng = random.Random(3)                                                   # and the repo's canonical() picks the same representative
    for _ in range(300):
        a = list(range(9)); rng.shuffle(a)
        g = np.array(a).reshape(3, 3)
        cg, _ = canonical(g)
        assert tuple(int(x) for x in cg.ravel()) == min(act(tuple(a), p) for p in PERMS)


def burnside(colours): return sum(colours ** cycles(p) for p in PERMS) // 8


def test_repeated_colours_brute_force_matches_burnside():
    for colours, expect in ((2, 102), (3, 2862)):
        seen = set()
        for cells in product(range(colours), repeat=9):
            cg, _ = canonical(np.array(cells).reshape(3, 3))
            seen.add(tuple(int(x) for x in cg.ravel()))
        assert len(seen) == expect == burnside(colours), (colours, len(seen), burnside(colours))
    assert burnside(10) == 125_512_750
    assert 7.96 < 10 ** 9 / burnside(10) < 8.0
    print("   saving factors (grids / D4 classes): 2 colours %.2f, 3 colours %.2f, 10 colours %.3f" %
          (512 / 102, 3 ** 9 / 2862, 10 ** 9 / burnside(10)))


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
