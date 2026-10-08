"""Implements concept-map row 87 (blog 1 Sep, "We've got the full quadrant on the table now ..."; read twice, byline rajnish choubey) and the 3x3 grid of row 88.
The 1 Sep post names FOUR things (the American engine, Chinese depth, DeepSeek's lean architecture, the European voice) and calls them "the full quadrant".
Two reads agree it names NO axes, so nothing says which of them is opposite which. The 3 Aug / 4 Aug / 13 Sep posts use threes (a triad of figures),
and the 13 Sep post uses nine entries on a three-by-three grid.

The reading being tested (Rajnish to accept or reject): a SHAPE carries information only through the arrangements it distinguishes. How many bits can
the layout alone carry, once the shape's own symmetries are accounted for? Counted by brute force from the repository's D4 (taken from axi.engine.d4):
  * three labels on a triangle (all six rotations and reflections are symmetries): 6 arrangements, 1 class. A triangle layout carries 0 bits.
  * four labels on the corners of a square (D4): 24 arrangements, 3 classes of 8. A class is exactly "which two labels are opposite", so a square
    layout of four named things can carry only log2(3) = 1.58 bits: the choice of pairing. Without named axes (as on the 1 Sep page) it carries none.
  * nine labels on the 3x3 board (D4): 362,880 arrangements, 45,360 classes, 15.47 bits (test_nine_labelings.py has the same count by another route).
The square has NO centre cell: no cell is fixed by the rotations, so there is nowhere for a "fifth thing" to sit inside the quadrant. In the 3x3 board the
centre is fixed by all 8 transforms. (The 1 Sep page's missing item, "none of them can sit on a porch ... or look a class of thirty kids in the eye", is
not placed anywhere; whether it is the centre is a reading, not a finding.)
Tenth-scan extension (still row 87): NAMING THE AXES. A "quadrant" is two binary axes, and its symmetry group depends on what is named. With nothing named, all
eight symmetries of the square apply: 3 classes (1.585 bits). If the two axes are told apart (no symmetry may swap them) but their poles are not, only the four
"flip an axis" symmetries remain (the identity, left-right, up-down, half-turn: the Klein four-group): 6 classes (2.585 bits). If the poles are named too, no symmetry
is left: 24 classes (4.585 bits). So the same four items carry 1.6, 2.6 or 4.6 bits depending on how much of the frame the page names; the 1 Sep page names none.
What they do NOT show: that the four items are meant as a square; that the layout of the posts' items was ever intended to carry information."""
import math, os, sys
from itertools import permutations
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.d4 import apply as d4apply, IDS


def perms_of_grid(n):
    idx = np.arange(n * n).reshape(n, n)
    return [tuple(int(x) for x in d4apply(idx, t).ravel()) for t in IDS]      # new position p shows old cell perm[p]


def classes_of(labels_count, perms):
    seen, classes = set(), 0
    for a in permutations(range(labels_count)):
        if a in seen: continue
        classes += 1
        seen |= {tuple(a[p[i]] for i in range(labels_count)) for p in perms}
    return classes


def test_triangle_carries_no_layout_information():
    s3 = list(permutations(range(3)))                                        # rotations and reflections of a triangle act as every permutation of its corners
    assert classes_of(3, s3) == 1


def test_square_of_four_carries_only_the_choice_of_opposite_pairs():
    perms = perms_of_grid(2)                                                  # the 2 x 2 grid of cells is the square's four corners
    assert len(set(perms)) == 8
    assert classes_of(4, perms) == 3 and abs(math.log2(3) - 1.585) < 0.001
    opp = lambda a: frozenset({frozenset((a[0], a[3])), frozenset((a[1], a[2]))})        # cells 0,3 and 1,2 are the diagonals of the 2 x 2 grid
    pairings = {opp(a) for a in permutations(range(4))}
    assert len(pairings) == 3
    for a in permutations(range(4)):                                          # the opposite-pair structure is invariant under all 8 symmetries
        assert all(opp(tuple(a[p[i]] for i in range(4))) == opp(a) for p in perms)


def test_square_has_no_centre_and_the_three_by_three_board_has_one():
    sq, br = perms_of_grid(2), perms_of_grid(3)
    assert not any(all(p[i] == i for p in sq) for i in range(4))             # no cell of the 2 x 2 grid stays put under all 8 transforms
    assert [i for i in range(9) if all(p[i] == i for p in br)] == [4]        # the centre of the 3 x 3 board does


def test_nine_labels_on_the_board_by_burnside():
    perms = perms_of_grid(3)
    fixed = [math.factorial(9) if all(p[i] == i for i in range(9)) else 0 for p in perms]   # distinct labels: only the identity fixes any arrangement
    assert sum(1 for f in fixed if f) == 1 and len(perms) == 8
    assert sum(fixed) // 8 == 45360 and sum(fixed) % 8 == 0 and abs(math.log2(45360) - 15.469) < 0.001


def test_naming_the_axes_and_poles_raises_what_a_quadrant_can_carry():
    p = perms_of_grid(2)                                                      # p[t] is the arrangement map of transform id t on the 2x2 grid
    rows, cols = {frozenset((0, 1)), frozenset((2, 3))}, {frozenset((0, 2)), frozenset((1, 3))}
    keeps_axes = [t for t in IDS if {frozenset(p[t][i] for i in r) for r in rows} == rows and {frozenset(p[t][i] for i in c) for c in cols} == cols]
    assert keeps_axes == [0, 2, 4, 6]                                         # identity, half-turn, and the two flips: axes are never swapped
    v4 = [p[t] for t in keeps_axes]
    assert all(tuple(q[q[i]] for i in range(4)) == (0, 1, 2, 3) for q in v4)  # every element is its own inverse
    assert {tuple(a[b[i]] for i in range(4)) for a in v4 for b in v4} == set(v4)   # closed: a group of order 4, the Klein four-group
    assert classes_of(4, p) == 3 and classes_of(4, v4) == 6 and classes_of(4, [tuple(range(4))]) == 24
    assert [round(math.log2(c), 3) for c in (3, 6, 24)] == [1.585, 2.585, 4.585]


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
