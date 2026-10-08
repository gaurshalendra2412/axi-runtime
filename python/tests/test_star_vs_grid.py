"""Implements concept-map row 146 (thirteenth scan): two 6 Sep pages that set a village against a map. Quotes verified on second reads (YES). Post 403, "That layout is the exact living blueprint of the organic village..." (about
145 words; published 07:17:21 UTC): "When your home sits at the center with the four orientations surrounding it, and friendship cuts across those traditional..." (the sentence is cut by the reader at 125 characters). Post 404, "An English map
tries to pin reality to a grid, carving the earth into hard squares, property lines, and right angles that ignore the terrain entirely, but a real village is an organic extension of the..." (the heading; about 105 words;
the reader returned the heading, not the body).

The reading being tested (Rajnish to accept or reject): a GRID is squares - every place has up to four neighbours and the lines close into small loops; a VILLAGE is a home at the centre with the four orientations around it
(a star), and "friendship cuts across" - a line between two places that the grid would not join. On the repository's own gate, with the 3x3 board as the grid and nothing assumed:
  1. The ball of radius 1 around the centre of the 3x3 grid (the home and the four orientations) is 5 places joined by 4 lines: the 4-STAR. The grid has 9 places, 12 lines and 4 independent loops (its four small squares);
     the star has 0 loops, a centre of degree 4, four ends of degree 1. At radius 2 the ball is the whole grid (distances from the centre: one place at 0, four at 1, four at 2).
  2. Cutting the four corners out of the grid is ONE delta (4 places, their 8 lines): the gate admits it and gives the star; leaving any one of the 8 lines out of the delta makes the gate refuse ("dangling edge"), 8 times out
     of 8. The inverse delta (taken before the step) restores the grid exactly.
  3. "Friendship cuts across": four lines between neighbouring orientations (north-east, east-south, south-west, west-north) added to the star make the WHEEL: 5 places, 8 lines, 4 independent loops - the same number of loops as
     the grid with 5 places instead of 9. Each of those four lines joins two places that are DIAGONAL neighbours in the grid (so the grid has no such line), and the corner cell that sits between each diagonal pair is one of
     the four cells that were cut (a one-to-one correspondence).
What they do NOT show: that the village the page describes is a star or that "friendship" means a line between orientations (the page says the home sits at the centre with four orientations around it and that friendship cuts
across "those traditional..." - the rest of the sentence is not in the reader's cut); that a grid is bad or a star better for anything; that either is a model of a model's memory. The counts are plain graph arithmetic."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.cico_parser import CICOParser
from axi.engine.gate import Graph, check, apply
from axi.engine.inverse import invert, to_text
from test_line_and_cycle import loop_count
from test_signed_slots import etext, same, copy

GRID = [(r, c) for r in range(3) for c in range(3)]
CENTRE, ORIENT, CORNERS = (1, 1), [(0, 1), (1, 2), (2, 1), (1, 0)], [(0, 0), (0, 2), (2, 2), (2, 0)]      # north, east, south, west in order around the centre


def parse(text): return CICOParser.parse_transition_delta(text)


def grid():
    g = Graph()
    items = [f"ADD[{r},{c}:0]" for (r, c) in GRID] + [etext("ADD", (a, b, "adj"), 1) for a in GRID for b in GRID if a < b and abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1]
    d = parse(" ".join(items))
    assert check(g, d).ok
    apply(g, d)
    return g


def loops(g): return loop_count(list(g.nodes), [(u, v) for (u, v, _) in g.edges])


def neighbours(g, n): return {v if u == n else u for (u, v, _) in g.edges if n in (u, v)}


def distances(g, start):
    dist, frontier = {start: 0}, [start]
    while frontier:
        nxt = []
        for n in frontier:
            for m in neighbours(g, n):
                if m not in dist: dist[m] = dist[n] + 1; nxt.append(m)
        frontier = nxt
    return dist


def test_the_ball_of_radius_1_around_the_centre_of_the_grid_is_the_four_star():
    g = grid()
    assert (len(g.nodes), len(g.edges), loops(g)) == (9, 12, 4)
    dist = distances(g, CENTRE)
    assert sorted(dist.values()) == [0, 1, 1, 1, 1, 2, 2, 2, 2]
    ball = {n for n, d in dist.items() if d <= 1}
    assert ball == {CENTRE} | set(ORIENT)
    inside = [(u, v) for (u, v, _) in g.edges if u in ball and v in ball]
    assert len(inside) == 4 and all(CENTRE in e for e in inside)                     # the four orientations are not joined to each other
    degree = lambda n: len(neighbours(g, n) & ball)
    assert degree(CENTRE) == 4 and all(degree(n) == 1 for n in ORIENT)
    assert loop_count(list(ball), inside) == 0
    assert {n for n, d in dist.items() if d <= 2} == set(GRID)


def test_cutting_the_four_corners_is_one_delta_the_gate_demands_all_eight_lines_and_the_inverse_restores_the_grid():
    g = grid(); before = copy(g)
    lines = sorted(k for k in g.edges if k[0] in CORNERS or k[1] in CORNERS)
    assert len(lines) == 8
    nodes = [f"DEL[{r},{c}]" for (r, c) in CORNERS]
    cut = parse(" ".join(nodes + [etext("DEL", k, g.edges[k]) for k in lines]))
    assert check(g, cut).ok
    for i in range(8):                                                             # leave out any one line: refused every time
        short = parse(" ".join(nodes + [etext("DEL", k, g.edges[k]) for j, k in enumerate(lines) if j != i]))
        r = check(g, short)
        assert not r.ok and r.reason.startswith("dangling edge")
    undo = to_text(invert(g, cut)); apply(g, cut)
    assert (len(g.nodes), len(g.edges), loops(g)) == (5, 4, 0) and set(g.nodes) == {CENTRE} | set(ORIENT)
    back = parse(undo)
    assert check(g, back).ok
    apply(g, back)
    assert same(g, before)


def test_four_friendship_lines_between_neighbouring_orientations_make_a_wheel_with_as_many_loops_as_the_grid():
    g = grid()
    apply(g, parse(" ".join(f"DEL[{r},{c}]" for (r, c) in CORNERS) + " " + " ".join(etext("DEL", k, g.edges[k]) for k in sorted(g.edges) if k[0] in CORNERS or k[1] in CORNERS)))
    rim = [(ORIENT[i], ORIENT[(i + 1) % 4]) for i in range(4)]
    friendship = parse(" ".join(etext("ADD", (a, b, "friend"), 1) for a, b in rim))
    assert check(g, friendship).ok
    apply(g, friendship)
    assert (len(g.nodes), len(g.edges), loops(g)) == (5, 8, 4)
    assert len(g.edges) - len(g.nodes) + 1 == loops(g) == 4                         # the grid also has 4, with 9 places and 12 lines
    assert 5 - 8 + 5 == 2 == 9 - 12 + 5                                             # Euler: faces 5 for the wheel (4 triangles and the outside) and 5 for the grid (4 squares and the outside)
    between = set()
    for a, b in rim:
        assert abs(a[0] - b[0]) == 1 and abs(a[1] - b[1]) == 1                      # diagonal neighbours: the grid has no line between them
        corner = (a[0] + b[0] - 1, a[1] + b[1] - 1)                                  # the cell completing the 2x2 square on a, b and the centre
        assert corner in CORNERS
        between.add(corner)
    assert between == set(CORNERS)                                                  # one friendship line per cut corner, none twice


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
