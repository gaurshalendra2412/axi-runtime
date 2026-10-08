"""Implements concept-map row 89 (the earliest text, the collage dated 21 Jan 2026, last 20,000 characters; read twice) and supports the existing row 48 (blog
13 Sep, "Four intersecting lines carving out nine fields ..."). Together they give a small GEOMETRIC ONTOLOGY in the author's own words:
    "Shiva. Line one. A photo is the point. A video is a line like the life most of u enjoy." / "What is communication? It's simple." /
    "it's a line between two points."  and  "X = Two lines crossing. It's art. It's science. It's commerce."
i.e. point, line (a message between two points), and crossing. (These are the January lines verified on a second read; the page gives no board.)

The reading being tested (Rajnish to accept or reject): on the 3x3 board the runtime already uses, "a line between two points" is a pair of
cells, and the geometry the board itself can name is exactly what is in this file. Everything else has to be said with an explicit edge.
  1. VOCABULARY. The 36 unordered pairs of cells fall into FIVE kinds by distance pattern (12 side-by-side pairs, 8 diagonal neighbours, 6
     same-row-or-column pairs with a cell between, 2 corner-to-corner pairs, 8 knight-move pairs). D4 splits them into EIGHT classes of sizes 8, 8,
     4, 4, 4, 4, 2, 2, because the centre, the corners and the edge-middles are different places: side-by-side = centre-edge (4) + corner-edge (8);
     diagonal neighbours = corner-centre (4) + edge-edge (4); with a cell between = corner-corner (4) + edge-edge across the centre (2). 28 of the 36
     pairs lie on a straight line of the board (a queen's move); the other 8, the knight-move pairs, do not. So a picture of the board can say
     "next to", "diagonal", "in line" and "opposite corners", and it cannot say a knight-move relation, nor any relation that does not follow the
     geometry (a dependency from one cell to a distant one is a pair like any other and needs its own edge).
  2. LINES. There are exactly 8 straight lines of three cells (3 rows, 3 columns, 2 diagonals). They are the 8 pairs with a cell between
     (6 + 2), so 8 of the 36 "lines between two points" pass through a third point. Each cell lies on 4 (centre), 3 (corner) or 2 (edge) of them.
  3. THE FOUR LINES. Two horizontal and two vertical lines cross in 4 points and cut the plane into 9 fields. A field is bounded by 4 (centre),
     3 (edge) or 2 (corner) of the four lines. Both counts are constant on the D4 orbits (centre 1, edges 4, corners 4), which is why the
     "centre holds the core and eight gates surround it" picture and the symmetry group agree.
What they do NOT show: that the author's "point, line, X" is this board; that "knight move" or any class carries meaning in the work; that a model
reads a picture better than a list (that is an experiment, see GEOMETRY_EXTRACTION section 6). Design consequence (ours): if the board becomes part of
the language (roadmap item 1), it should supply these eight relation classes by itself and the delta should only carry the edges that geometry cannot."""
import os, sys
from itertools import combinations
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.d4 import apply as d4apply, IDS

CELLS = [(r, c) for r in range(3) for c in range(3)]
IDX = np.arange(9).reshape(3, 3)


def cell_map(t):
    flat = [int(x) for x in d4apply(IDX, t).ravel()]            # position p shows old cell flat[p]
    return {(q // 3, q % 3): (p // 3, p % 3) for p, q in enumerate(flat)}


MAPS = [cell_map(t) for t in IDS]
PAIRS = [frozenset(p) for p in combinations(CELLS, 2)]


def orbit(pair): return frozenset(frozenset(m[x] for x in pair) for m in MAPS)


def kind(pair):
    (a, b), (c, d) = sorted(pair)
    return tuple(sorted((abs(a - c), abs(b - d))))


def test_eight_classes_of_pairs_five_kinds_and_what_a_picture_can_say():
    orbits = {orbit(p) for p in PAIRS}
    assert len(PAIRS) == 36 and sorted(len(o) for o in orbits) == [2, 2, 4, 4, 4, 4, 8, 8]
    by_kind = {}
    for p in PAIRS: by_kind.setdefault(kind(p), []).append(p)
    assert {k: len(v) for k, v in by_kind.items()} == {(0, 1): 12, (1, 1): 8, (0, 2): 6, (2, 2): 2, (1, 2): 8}
    for k, v in by_kind.items():                                              # every kind is a union of D4 classes; D4 splits three of the five
        assert frozenset().union(*(orbit(p) for p in v)) == frozenset(v)
        assert len({orbit(p) for p in v}) == {(0, 1): 2, (1, 1): 2, (0, 2): 2, (2, 2): 1, (1, 2): 1}[k]
    queen = [p for p in PAIRS if kind(p) != (1, 2)]
    assert len(queen) == 28 and len(by_kind[(1, 2)]) == 8                    # all but the knight-move pairs lie on a straight line


def test_eight_lines_of_three_and_how_many_lines_pass_through_each_cell():
    lines = [frozenset(l) for l in ([(r, c) for c in range(3)] for r in range(3))] \
        + [frozenset(l) for l in ([(r, c) for r in range(3)] for c in range(3))] \
        + [frozenset((i, i) for i in range(3)), frozenset((i, 2 - i) for i in range(3))]
    collinear = lambda t: (lambda a, b, c: (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) == 0)(*sorted(t))
    trip = [frozenset(t) for t in combinations(CELLS, 3) if collinear(t)]
    assert set(trip) == set(lines) and len(lines) == 8                       # the only collinear triples are the 8 lines of tic-tac-toe
    ends = {frozenset((min(l), max(l))) for l in lines}
    assert ends == {p for p in PAIRS if kind(p) in ((0, 2), (2, 2))} and len(ends) == 8     # one pair of end cells per line: the pairs with a cell between
    through = {c: sum(1 for l in lines if c in l) for c in CELLS}
    assert through[(1, 1)] == 4 and {through[c] for c in CELLS if sum(c) % 2 == 0 and c != (1, 1)} == {3} and {through[c] for c in CELLS if sum(c) % 2 == 1} == {2}
    assert sum(through.values()) == 24


def test_four_lines_make_four_crossings_nine_fields_and_the_same_three_classes():
    xs, ys = (1, 2), (1, 2)                                                  # vertical lines x = 1, 2 and horizontal lines y = 1, 2 inside a 3 x 3 square
    assert len(xs) * len(ys) == 4 and (len(xs) + 1) * (len(ys) + 1) == 9
    bounded = {(r, c): sum(1 for x in xs if x in (c, c + 1)) + sum(1 for y in ys if y in (r, r + 1)) for (r, c) in CELLS}
    assert bounded[(1, 1)] == 4 and {bounded[c] for c in CELLS if c in ((0, 1), (1, 0), (1, 2), (2, 1))} == {3} and {bounded[c] for c in CELLS if sum(c) % 2 == 0 and c != (1, 1)} == {2}
    orbits = {frozenset(m[c] for m in MAPS) for c in CELLS}
    assert sorted(len(o) for o in orbits) == [1, 4, 4]
    assert all(len({bounded[c] for c in o}) == 1 for o in orbits)           # the count of bounding lines is constant on each D4 orbit


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
