"""Implements concept-map row 100 (tenth scan): blog 13 Sep "Sacred texts across ancient traditions treat the number three as the architectural
blueprint of existence ..." (Post 32, re-read) and, in the teaser text that the theme appends to the "geometry of three" page (two reads, labelled
here as a teaser): "three points are required to define a plane, give an object volume, and make a structure truly unyielding."
The earlier log (BLOG_SCAN_LOG row 32) kept the first half, "three points are required to define a plane".

The reading being tested (Rajnish to accept or reject): "unyielding" is a counting statement. A framework of n points joined by bars is rigid when the
bars remove every freedom of movement except sliding and turning the whole thing. Counted by the rank of the rigidity matrix at generic positions:
  * in the plane, n points need 2n - 3 independent bars; in space, 3n - 6. (n >= 2 in the plane, n >= 3 in space.)
  * 3 points: the triangle (3 bars) is the smallest rigid figure in the plane. In space 3 points with 3 bars are still rigid (as a flat body).
  * 4 points in the plane, 4 bars (a square): rank 4 < 5, it flexes. One diagonal makes it rigid (5 bars).
  * 4 points in space need 6 bars (the tetrahedron); remove one bar and it flexes.
  * the page's "give an object volume" is off by one: any 3 points lie in a plane, so the volume (the determinant of three edge vectors) needs 4 points.
    The smallest rigid body that has volume is the tetrahedron, 4 points: three points settle the plane, four settle the space.
  * the 3x3 board drawn as a framework: with only the 12 up/down/left/right bars it has rank 12 of the 15 needed (three internal flexes); with one diagonal
    in each of the four small squares it has 16 bars, rank 15, rigid with one bar to spare; with both diagonals (20 bars, "eight surrounding" neighbours)
    still rank 15, rigid with five to spare. A mapping whose constraint graph is the plain grid does not pin the shape down; a triangulated one does.
What they do NOT show: that the posts mean a framework by "unyielding"; that our gate has anything to do with bars (it does not: INTERP only, section 2.2 of
STRUCTURE_EXTRACTION_2). All ranks are at generic (random) positions, as the theory defines them; at a perfectly regular lattice some extra flexes appear."""
import os, sys
from itertools import combinations
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))


def rank(points, edges, dim):
    P = np.asarray(points, float); R = np.zeros((len(edges), dim * len(P)))
    for k, (i, j) in enumerate(edges):
        d = P[i] - P[j]
        R[k, dim * i:dim * i + dim] = d; R[k, dim * j:dim * j + dim] = -d
    return int(np.linalg.matrix_rank(R))


def rigid_rank(n, dim):               # 2n-3 in the plane, 3n-6 in space
    return 2 * n - 3 if dim == 2 else 3 * n - 6


def test_triangle_is_the_smallest_rigid_figure_and_the_square_flexes():
    rng = np.random.default_rng(1)
    tri = rng.random((3, 2)); e = [(0, 1), (1, 2), (0, 2)]
    assert rank(tri, e, 2) == rigid_rank(3, 2) == 3
    assert rank(tri, e[:2], 2) == 2 < rigid_rank(3, 2)                      # drop a bar: flexes
    sq = rng.random((4, 2)); ring = [(0, 1), (1, 2), (2, 3), (3, 0)]
    assert rank(sq, ring, 2) == 4 < rigid_rank(4, 2) == 5                   # the square flexes
    assert rank(sq, ring + [(0, 2)], 2) == 5                                # one diagonal makes it rigid
    assert rank(sq, ring + [(0, 2), (1, 3)], 2) == 5                        # the second diagonal is redundant


def test_three_points_fix_a_plane_and_four_fix_the_space():
    rng = np.random.default_rng(2)
    p3 = rng.random((3, 3)); k3 = list(combinations(range(3), 2))
    assert rank(p3, k3, 3) == rigid_rank(3, 3) == 3                         # a triangle in space is a rigid flat body
    assert np.linalg.matrix_rank(p3[1:] - p3[0]) == 2                       # its three points span a plane: no volume
    p4 = rng.random((4, 3)); k4 = list(combinations(range(4), 2))
    assert rank(p4, k4, 3) == rigid_rank(4, 3) == 6                         # tetrahedron: 6 bars, rigid
    assert rank(p4, k4[:-1], 3) == 5 < 6                                    # 5 bars: flexes
    assert np.linalg.matrix_rank(p4[1:] - p4[0]) == 3 and abs(np.linalg.det(p4[1:] - p4[0])) > 1e-6     # four points have volume
    for n in (3, 4, 5, 6, 7):                                               # the general count: the complete graph is rigid exactly when it has >= the needed bars
        pts = rng.random((n, 3)); kn = list(combinations(range(n), 2))
        assert rank(pts, kn, 3) == rigid_rank(n, 3)
        pts2 = rng.random((n, 2))
        assert rank(pts2, kn, 2) == rigid_rank(n, 2)


def test_the_three_by_three_grid_as_a_framework():
    rng = np.random.default_rng(3); cell = lambda r, c: 3 * r + c
    P = rng.random((9, 2)); need = rigid_rank(9, 2)
    plain = [(cell(r, c), cell(r, c + 1)) for r in range(3) for c in range(2)] + [(cell(r, c), cell(r + 1, c)) for r in range(2) for c in range(3)]
    one = plain + [(cell(r, c), cell(r + 1, c + 1)) for r in range(2) for c in range(2)]
    both = one + [(cell(r, c + 1), cell(r + 1, c)) for r in range(2) for c in range(2)]
    assert (len(plain), len(one), len(both), need) == (12, 16, 20, 15)
    assert rank(P, plain, 2) == 12 and 2 * 9 - 3 - 12 == 3                   # 3 internal flexes
    assert rank(P, one, 2) == need                                           # rigid, one bar redundant
    assert rank(P, both, 2) == need                                          # rigid, five redundant


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
