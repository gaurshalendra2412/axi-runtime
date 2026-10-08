"""Implements concept-map rows 99 and 108 (tenth scan).
Row 99: blog 13 Sep "That geometry of three is a masterclass in stability and motion ... the universal threefold symmetry of the recycling and
radiation symbols" (Post 25, re-read; the page body is its title, two reads; the title list is: a triangle = "simple structural closure", a fan =
"dynamic spin", and the two symbols). Row 108: blog 13 Sep "Biology gives us the blueprint of the machinery - cells dividing, DNA strands mirroring each other, two bodies
colliding to form a third - but it leaves out the ghost in the machine" (two reads).

The reading being tested (Rajnish to accept or reject): "threefold symmetry" and "mirroring" each name TWO different things. A figure whose parts are
undirected has every rotation AND every reflection as a symmetry (the dihedral group); a figure whose parts have a direction (a fan's blades that all
lean one way, a ring of arrows that all point round the same way) keeps only the rotations (the cyclic group). Direction costs exactly half the group.
Counted here by brute force, with nothing assumed:
  1. A ring of n cells: as an undirected graph it has 2n automorphisms, as a directed ring it has n, and those n are exactly the rotations.
  2. Labels on a figure: the number of arrangements that cannot be turned into each other doubles when reflections are NOT symmetries.
     triangle, three labels: 2 classes (1 bit) with rotations only, 1 class (0 bits) with reflections too.
     square, four labels: 6 classes (2.585 bits) with rotations only, 3 classes (1.585 bits) with reflections too.
     3x3 board, nine labels: 90,720 classes (16.47 bits) rotations only, 45,360 (15.47 bits) with reflections: the one extra bit is the handedness.
  3. In our own language an edge is DIRECTED, `ADD[(r,c)->(r,c):rel#w]`. The clockwise ring round the outer eight cells of the board is kept by the
     four rotations and turned into the anticlockwise ring by the four reflections. So a frame group that includes reflections (the repository's D4)
     is only harmless for relations that are symmetric; for directed relations a reflection changes the meaning (this supports roadmap option B,
     separate ROT and FLIP operations, see FRAMEWORK_SYNTHESIS section 8; nothing is built).
  4. (row 108) Strings over {A, C, G, T}: `reverse` and `complement` are both involutions, they commute, and their composition `reverse complement` is the
     third non-identity element of a four-element group (the Klein four-group). A string equal to its own reverse complement exists only when
     its length is even (count 4^(n/2)): an odd-length string has a middle letter that would have to be its own complement.
What they do NOT show: that the posts mean a group by "symmetry"; that the recycling symbol is chiral and the radiation symbol is not (true as I know
the two pictures, but I did not look at them in this session); that the two strands of a DNA molecule are related by a half-turn, not by a mirror
(standard biochemistry, from memory, to check before print; the real molecule is a right-handed helix and its mirror image is a different, left-handed
molecule). Only the algebra above is tested."""
import math, os, sys
from itertools import permutations, product
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.d4 import apply as d4apply, IDS

ROT = (0, 1, 2, 3)          # ids 0..3 are the four rotations, 4..7 the four reflections (axi.engine.d4)


def grid_perms(n, ids):
    idx = np.arange(n * n).reshape(n, n)
    return [tuple(int(x) for x in d4apply(idx, t).ravel()) for t in ids]      # new position p shows old cell perm[p]


def classes_of(k, perms):
    seen, classes = set(), 0
    for a in permutations(range(k)):
        if a in seen: continue
        classes += 1
        seen |= {tuple(a[p[i]] for i in range(k)) for p in perms}
    return classes


def automorphisms(n, edges):
    E = set(edges)
    return [p for p in permutations(range(n)) if {(p[a], p[b]) for a, b in E} == E]


def test_direction_halves_the_symmetry_of_a_ring():
    for n in (3, 4, 5, 6):
        directed = [(i, (i + 1) % n) for i in range(n)]
        undirected = directed + [((i + 1) % n, i) for i in range(n)]
        a_dir, a_und = automorphisms(n, directed), automorphisms(n, undirected)
        assert len(a_und) == 2 * n and len(a_dir) == n
        assert set(a_dir) == {tuple((i + k) % n for i in range(n)) for k in range(n)}      # exactly the rotations
        assert set(a_dir) < set(a_und)


def test_rotation_only_classes_are_twice_the_full_group_classes():
    c3, s3 = [(0, 1, 2), (1, 2, 0), (2, 0, 1)], list(permutations(range(3)))
    assert (classes_of(3, c3), classes_of(3, s3)) == (2, 1)                                  # triangle: 1 bit vs 0 bits
    sq_rot, sq_full = grid_perms(2, ROT), grid_perms(2, IDS)
    assert (len(set(sq_rot)), len(set(sq_full))) == (4, 8)
    assert (classes_of(4, sq_rot), classes_of(4, sq_full)) == (6, 3)
    assert abs(math.log2(6) - 2.585) < 0.001 and abs(math.log2(3) - 1.585) < 0.001
    # 3x3 board, nine distinct labels, Burnside: only the identity fixes a labelling, so classes = 9! / |group|
    for ids, size in ((ROT, 4), (IDS, 8)):
        perms = grid_perms(3, ids)
        assert len(set(perms)) == size
        fixed = [math.factorial(9) if p == tuple(range(9)) else 0 for p in perms]
        assert sum(fixed) % size == 0
    assert math.factorial(9) // 4 == 90720 and math.factorial(9) // 8 == 45360
    assert abs(math.log2(90720) - math.log2(45360) - 1.0) < 1e-12


def test_a_directed_ring_on_the_board_is_kept_by_rotations_and_reversed_by_reflections():
    ring_cells = [0, 1, 2, 5, 8, 7, 6, 3]                                    # outer eight cells of the 3x3 board, clockwise from the top left
    ring = {(ring_cells[i], ring_cells[(i + 1) % 8]) for i in range(8)}
    reverse = {(b, a) for a, b in ring}
    assert ring != reverse
    for t in IDS:
        perm = grid_perms(3, [t])[0]                                         # new position p shows old cell perm[p]
        where = {old: new for new, old in enumerate(perm)}                   # where an old cell ends up
        image = {(where[a], where[b]) for a, b in ring}
        if t in ROT: assert image == ring, t
        else: assert image == reverse, t
    # a symmetric (undirected) relation does not care: both orientations of every ring edge are kept by all 8 transforms
    both = ring | reverse
    for t in IDS:
        perm = grid_perms(3, [t])[0]; where = {old: new for new, old in enumerate(perm)}
        assert {(where[a], where[b]) for a, b in both} == both


def test_strands_mirror_algebra_is_the_klein_four_group_and_odd_length_has_no_palindrome():
    comp = {"A": "T", "T": "A", "C": "G", "G": "C"}
    rev = lambda s: s[::-1]
    cmp_ = lambda s: "".join(comp[x] for x in s)
    rc = lambda s: rev(cmp_(s))
    for n in range(0, 6):
        for s in map("".join, product("ACGT", repeat=n)):
            assert rev(rev(s)) == s and cmp_(cmp_(s)) == s and rc(rc(s)) == s            # three involutions
            assert rev(cmp_(s)) == cmp_(rev(s)) == rc(s)                                  # reverse and complement commute; product = rc
            assert len({s, rev(s), cmp_(s), rc(s)}) in (1, 2, 4)                         # orbit sizes of a group of order 4
    for n in range(1, 9):
        pal = sum(1 for s in map("".join, product("ACGT", repeat=n)) if rc(s) == s)
        assert pal == (4 ** (n // 2) if n % 2 == 0 else 0), (n, pal)


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
