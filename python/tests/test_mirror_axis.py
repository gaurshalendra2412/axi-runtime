"""Implements concept-map row 81 (blog 24 Aug, "The Anatomy of the Integrated Mind"; read twice, byline "rajnish choubey", category BLACK).
The post describes an IMAGE (the image itself was not seen; only the post's description): a figure with bilateral symmetry. The left
half holds Shiva and Kali/Durga ("consciousness and manifest power"), the right half "a vast, shimmering matrix of digital code, pixels,
and binary data", and down the CENTRAL AXIS run "sacred scripts and primordial phonemes (Maa, Shiva, Durga)", where raw sound and
vibration "condense into symbols, language, and material reality". The post says: "The two halves are a false division."

The reading being tested (Rajnish to accept or reject): a mirror has an axis that does not move, and whatever is on the axis is the only
thing a left-right swap leaves alone. Three checkable facts about the 3x3 board and the 8 D4 transforms the runtime uses:
  1. FIXED CELLS. The identity fixes all 9 cells; the three rotations fix only the centre; each of the four reflections fixes exactly three
     cells, a line through the centre. The left-right mirror fixes the middle column and swaps (r,0) with (r,2). Of the 9 cells, the
     centre is on every axis; corners and edge-middles are on at most one.
  2. BURNSIDE. (9 + 1 + 1 + 1 + 3 + 3 + 3 + 3) / 8 = 3: the average number of fixed cells is the number of orbits, and it equals the
     centre / corners / edges split pinned in test_d4_orbits.py. Two independent routes to the same 3.
  3. THE HALVES ARE ONE HALF. A board that equals its own mirror image is fixed by the left half plus the axis (6 cells): with k colours
     there are k**6 such boards out of k**9 (brute force for k = 2 and 3). The right half adds no information once the left half and the axis
     are known. That is a precise sense in which "the two halves are a false division" holds for a symmetric state, and it fails for a
     state that is not symmetric.
What they do NOT show: that the image is a 3x3 board; that "digital" and "divine" halves are mirror images in any sense other than the
post's rhetoric; that language lives on the axis in the runtime. The nearest runtime fact is weaker: the delta text is the one thing both
the model and the host read the same way, but nothing here tests that."""
import os, sys
from itertools import product
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.d4 import apply, IDS

IDX = np.arange(9).reshape(3, 3)
CELLS = [(r, c) for r in range(3) for c in range(3)]


def fixed_cells(t):
    out = apply(IDX, t)
    return frozenset((p // 3, p % 3) for p in range(9) if int(out.ravel()[p]) == p)


def test_fixed_cells_of_the_eight_transforms():
    sizes = sorted(len(fixed_cells(t)) for t in IDS)
    assert sizes == [1, 1, 1, 3, 3, 3, 3, 9]                                  # three rotations, four reflections, identity
    lines = [fixed_cells(t) for t in IDS if len(fixed_cells(t)) == 3]
    assert set(lines) == {frozenset((r, 1) for r in range(3)), frozenset((1, c) for c in range(3)),
                          frozenset((i, i) for i in range(3)), frozenset((i, 2 - i) for i in range(3))}   # middle column, row, both diagonals
    assert all((1, 1) in f for f in lines) and all(fixed_cells(t) == {(1, 1)} for t in IDS if len(fixed_cells(t)) == 1)
    for p in CELLS:                                                           # every cell but the centre lies on exactly one axis
        assert sum(1 for f in lines if p in f) == (4 if p == (1, 1) else 1)


def test_the_left_right_mirror_swaps_the_halves_and_fixes_the_axis():
    mirror = [t for t in IDS if fixed_cells(t) == frozenset((r, 1) for r in range(3))]
    assert len(mirror) == 1
    a = np.arange(9).reshape(3, 3)
    b = apply(a, mirror[0])
    assert np.array_equal(b, np.fliplr(a))                                    # it is the plain left-right flip
    assert all(b[r, 1] == a[r, 1] and b[r, 0] == a[r, 2] and b[r, 2] == a[r, 0] for r in range(3))


def test_burnside_gives_the_same_three_orbits():
    assert sum(len(fixed_cells(t)) for t in IDS) == 24 and 24 // 8 == 3
    def goes_to(p, t):                                                        # where cell p lands when the board is transformed by t
        flat = [int(x) for x in apply(IDX, t).ravel()]
        return divmod(flat.index(3 * p[0] + p[1]), 3)
    orbits = {p: frozenset(goes_to(p, t) for t in IDS) for p in CELLS}
    assert set(orbits.values()) == {frozenset({(1, 1)}), frozenset({(0, 0), (0, 2), (2, 0), (2, 2)}), frozenset({(0, 1), (1, 0), (1, 2), (2, 1)})}


def test_the_two_halves_of_a_symmetric_board_are_one_half():
    for k in (2, 3):
        sym = sum(1 for cells in product(range(k), repeat=9) if np.array_equal(np.array(cells).reshape(3, 3), np.fliplr(np.array(cells).reshape(3, 3))))
        assert sym == k ** 6 and k ** 9 // sym == k ** 3                      # k^6 symmetric boards of k^9: the right half is free of choice only when the board is not symmetric
    mirror = [t for t in IDS if fixed_cells(t) == frozenset((r, 1) for r in range(3))][0]
    flat = [int(x) for x in apply(IDX, mirror).ravel()]
    cycles = {frozenset({p, flat[p]}) for p in range(9)}                       # orbits of the mirror alone: 3 fixed cells + 3 swapped pairs
    assert len(cycles) == 6                                                    # six free choices fix a symmetric board, which is why k**6


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
