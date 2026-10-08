"""Implements concept-map row 117 (eleventh scan): blog 13 Sep, the page "A rocket ship is just a very expensive mechanical extension of the ego trying to outrun itself. The tech
industry treats space as a mileage ..." (Post 274; about 165 words, one paragraph; byline rajnish choubey; AI-voice by style, which proves nothing; two reads). Quotes verified
on a second read: "The ancient seers understood that you don't reach the infinite by moving across distance;" (curly apostrophe on the page) "you reach it by collapsing the
illusion of distance entirely."; "The entire cosmos fits inside the architecture of consciousness."; "You just had to stop running."

The reading being tested (Rajnish to accept or reject): in a graph, distance is not a fixed fact about the points, it is a count of steps along the edges that exist.
"Collapsing the illusion of distance" is then literal: add edges and the distances shrink; the points did not move. On the 3x3 board the same nine cells give four different
worlds depending on which neighbours are allowed. Counted by breadth-first search (the cells are (r,c), a pair is counted once; the "down-right" diagonal is the one used for the 16-edge row):
    neighbours                         edges (pairs)   pairs at distance 1 / 2 / 3 / 4     farthest apart   sum of distances   symmetries among the 8 board transforms
    up, down, left, right                   12           12 / 14 / 8 / 2                    4                 72               8
    plus one diagonal in each small square  16           16 / 15 / 4 / 1                    4                 62               4  (identity, half-turn and the two diagonal flips)
    both diagonals ("eight surrounding")    20           20 / 16 / - / -                    2                 52               8
    everyone with everyone                  36           36 / - / - / -                     1                 36               8
The 12, 16 and 20 are the same edge sets as in test_rigidity_counts.py: the plain grid does not pin the shape down, one diagonal per small square does (16 bars), and the
only choice of diagonals that keeps all 8 board symmetries is both (20 bars, five more than rigidity needs). So "rigid with the fewest bars" and "symmetric" pull apart on the board.
In the gate: the board has NO built-in neighbours (test_gate_frame_covariance.py); adjacency is whatever `adj` edges the task adds. Adding the eight diagonal pairs to the
plain board, as ordinary deltas, collapses the farthest distance from 4 to 2, and the inverse delta (test_undo_log.py) brings it back to 4.
What they do NOT show: that the page means a graph by "distance"; that the infinite or consciousness is anything like a complete graph; that a model reasons by distances on a
board. The page's sentence is a claim about attention; the arithmetic is only about which edges exist."""
import os, sys
from collections import Counter, deque
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.cico_parser import CICOParser
from axi.engine.d4 import IDS
from axi.engine.gate import Graph, check, apply
from axi.engine.inverse import invert, to_text
from test_gate_frame_covariance import d4_relabel
from test_signed_slots import etext

CELLS = [(r, c) for r in range(3) for c in range(3)]


def pairs_where(rule):
    return {frozenset((a, b)) for a in CELLS for b in CELLS if a < b and rule(a, b)}


PLAIN = pairs_where(lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1)
BOTH = PLAIN | pairs_where(lambda a, b: max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1)
ONE = PLAIN | pairs_where(lambda a, b: b[0] - a[0] == 1 and b[1] - a[1] == 1)            # one diagonal per small square: down-right
COMPLETE = pairs_where(lambda a, b: True)
DIAGONALS_BOTH = BOTH - PLAIN


def distances(pairs):
    adj = {c: set() for c in CELLS}
    for p in pairs:
        a, b = tuple(p); adj[a].add(b); adj[b].add(a)
    hist, total = Counter(), 0
    for s in CELLS:
        dist, q = {s: 0}, deque([s])
        while q:
            x = q.popleft()
            for y in adj[x]:
                if y not in dist: dist[y] = dist[x] + 1; q.append(y)
        for t in CELLS:
            if s < t:
                assert t in dist                                           # connected
                hist[dist[t]] += 1; total += dist[t]
    return dict(sorted(hist.items())), total


def symmetries_kept(pairs):
    out = []
    for t in IDS:
        m = d4_relabel(t)
        if {frozenset(m[x] for x in p) for p in pairs} == pairs: out.append(t)
    return out


def test_the_same_nine_cells_have_four_different_distance_tables():
    assert (len(PLAIN), len(ONE), len(BOTH), len(COMPLETE)) == (12, 16, 20, 36)         # the edge sets of test_rigidity_counts.py, plus the complete graph
    assert distances(PLAIN) == ({1: 12, 2: 14, 3: 8, 4: 2}, 72)
    assert distances(BOTH) == ({1: 20, 2: 16}, 52)
    assert distances(COMPLETE) == ({1: 36}, 36)
    assert distances(ONE) == ({1: 16, 2: 15, 3: 4, 4: 1}, 62)             # in between: the down-right diagonals shorten one pair of opposite corners, not the other
    assert [max(distances(p)[0]) for p in (PLAIN, ONE, BOTH, COMPLETE)] == [4, 4, 2, 1]
    assert [distances(p)[1] for p in (PLAIN, ONE, BOTH, COMPLETE)] == [72, 62, 52, 36]       # adding edges never lengthens a path, so the totals only fall
    corners = {(0, 0), (0, 2), (2, 0), (2, 2)}
    assert {(a, b) for a in corners for b in corners if a < b and abs(a[0] - b[0]) + abs(a[1] - b[1]) == 4} == {((0, 0), (2, 2)), ((0, 2), (2, 0))}      # the 2 farthest pairs of the plain board


def test_one_diagonal_per_square_is_rigid_but_keeps_only_4_of_the_8_symmetries():
    assert len(symmetries_kept(PLAIN)) == 8 and len(symmetries_kept(BOTH)) == 8 and len(symmetries_kept(COMPLETE)) == 8
    kept = symmetries_kept(ONE)
    assert len(kept) == 4 and set(kept) < set(IDS)                         # a diagonal has a direction: it keeps the half-turn, the identity and the two flips along the diagonals
    assert {0, 2} <= set(kept)                                             # identity and the half-turn (ids 0 and 2 in axi.engine.d4)


def board_with(pairs, rel="adj"):
    g = Graph()
    for c in CELLS: g.add_node(c, 0)
    d = CICOParser.parse_transition_delta(" ".join(etext("ADD", (a, b, rel), 1) for p in sorted(map(sorted, pairs)) for a, b in (p, p[::-1])))
    assert check(g, d).ok
    apply(g, d)
    return g


def graph_distances(g):
    return distances({frozenset((u, v)) for (u, v, _) in g.edges})


def test_in_the_gate_collapsing_distance_is_adding_edges_and_the_inverse_delta_restores_it():
    g = board_with(PLAIN)
    assert graph_distances(g) == distances(PLAIN) and len(g.edges) == 24                # 12 pairs, both directions
    d = CICOParser.parse_transition_delta(" ".join(etext("ADD", (a, b, "adj"), 1) for p in sorted(map(sorted, DIAGONALS_BOTH)) for a, b in (p, p[::-1])))
    assert len(DIAGONALS_BOTH) == 8 and check(g, d).ok
    undo = to_text(invert(g, d))                                           # computed before the step
    apply(g, d)
    assert graph_distances(g) == distances(BOTH) and max(graph_distances(g)[0]) == 2   # the farthest pair is now 2 steps away
    back = CICOParser.parse_transition_delta(undo)
    assert check(g, back).ok
    apply(g, back)
    assert graph_distances(g) == distances(PLAIN) and max(graph_distances(g)[0]) == 4


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
