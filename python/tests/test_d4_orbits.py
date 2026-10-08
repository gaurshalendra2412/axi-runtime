"""Implements concept-map row 48 (blog post 3, 13 Sep: "four intersecting lines carving out nine fields ... the center holds
the core and the eight surrounding gates face the quarters").
What this pins (a fact about the 3x3 grid and the 8 D4 transforms the code uses):
  * the centre cell is fixed by all 8 transforms;
  * the other eight cells split into TWO orbits of four (corners, edges); no transform maps a corner to an edge.
So under D4 the "eight gates" are two classes, not one. If the post means ONE class (a compass of eight directions), the symmetry it
needs is a 45-degree rotation of the compass, which is not a symmetry of a square grid and is not implemented here.
What it does NOT show: that the post means D4 at all. That is question 18 for Rajnish."""
import numpy as np
from axi.engine.d4 import apply, IDS

CENTRE = frozenset({(1, 1)})
CORNERS = frozenset({(0, 0), (0, 2), (2, 0), (2, 2)})
EDGES = frozenset({(0, 1), (1, 0), (1, 2), (2, 1)})


def orbit(cell):
    out = set()
    for t in IDS:
        g = np.zeros((3, 3), dtype=int)
        g[cell] = 1
        r, c = np.argwhere(apply(g, t) == 1)[0]
        out.add((int(r), int(c)))
    return frozenset(out)


def test_orbits_on_3x3_are_centre_corners_edges():
    cells = [(r, c) for r in range(3) for c in range(3)]
    assert {orbit(p) for p in cells} == {CENTRE, CORNERS, EDGES}


def test_no_transform_maps_a_corner_to_an_edge_and_centre_is_fixed():
    assert all(orbit(p) == CORNERS for p in CORNERS)
    assert all(orbit(p) == EDGES for p in EDGES)
    assert orbit((1, 1)) == CENTRE


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
