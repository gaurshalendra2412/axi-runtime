"""Implements concept-map row 135 (thirteenth scan): the 16 Aug page "Most humans spend their entire lives looking away from those fractures..." (Post 325, byline rajnish choubey,
about 250 words, one read - the four bullets below are its whole structure). It lists FOUR PAIRS as bullets: "East/West", "Rich/Poor", "Educated/Uneducated", "Digital/Real". The page
never states how many combinations the four pairs make, and does not say whether the pairs are independent.

The reading being tested (Rajnish to accept or reject): each pair is a two-valued axis (one side or the other), the four pairs are independent, and a person or a situation is a choice
on every axis. Then there are 2^4 = 16 combinations, and they form the 4-CUBE (the tesseract graph Q4): two combinations are neighbours when they differ on exactly one axis. The
repository already has a 16-state object that is exactly this: the 2x2 board, where each of the four cells is present or absent. Counted with the gate, nothing assumed:
  1. For each of the 16 hosts (every subset of the four cells), the gate admits exactly four single-cell deltas - ADD on each absent cell, DEL on each present one - and refuses the
     other four (ADD on a present cell: "identification"; DEL on an absent cell: "match"). So the graph of admitted one-step moves has 16 vertices, degree 4, 32 edges, and is
     bipartite (every move changes the number of cells by one). Its distance is the Hamming distance (BFS) and its diameter is 4.
  2. Between the empty board and the full board the shortest paths have length 4 and there are 4! = 24 of them; every one of the 24 orders of the four ADDs is admitted step by step
     and ends at the same board, and so does the single delta that adds all four at once. The same holds for the four DELs coming back.
  3. Walks: the number of closed walks of length n from any start, summed over all starts, is trace(A^n) = 4^n + 4*2^n + 4*(-2)^n + (-4)^n for n >= 1 (the eigenvalues of Q4 are 4, 2, 0, -2, -4 with
     multiplicities 1, 4, 6, 4, 1), checked by matrix power for n = 1..8. It is ZERO for every odd n: no closed walk of odd length exists. A closed 5-step walk therefore needs a no-op step; of
     the 5^5 = 3,125 five-step sequences over {4 flips, one no-op} from a fixed start, exactly 241 return to the start, and every one of them contains an odd number of no-ops. (This is
     the parity point for the 11 Sep "five-step loop" page, whose step 5 "ends the confrontation without changing the behavior"; the page's own counts are checked in test_declared_counts.py.)
  4. Axes: n pairs have 2n sides but 2^n combinations (8 against 16 for n = 4; the two are equal only for n = 1 and 2). Three pairs make the cuboid of the 12 Aug "Cuboid of Consciousness" page
     (8 corners, 12 lines, 6 square faces, test_ring_to_circle.py); four pairs make the 4-cube with 16 corners, 32 lines, 24 square faces and 8 cubic cells, and 16 - 32 + 24 - 8 = 0. The
     group of one-step-preserving relabelings of the 16 combinations has order 384 = 2^4 * 4! (flip any axes, permute the axes), found here by exhaustive search, not assumed.
What they do NOT show: that the four pairs are independent or two-valued (East/West and Digital/Real are not obviously binary, and the page offers no data); that the 16 combinations
mean anything about people; that the 2x2 board is the intended carrier (the choice of four cells for four axes is arbitrary); that anything here concerns a model. The 4-cube, its spectrum,
Hamming distance and parity arguments are standard mathematics, not findings of ours."""
import os, sys
from itertools import permutations, product
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.cico_parser import CICOParser
from axi.engine.gate import Graph, check, apply
from test_signed_slots import same

CELLS4 = [(0, 0), (0, 1), (1, 0), (1, 1)]
AXES = ["East/West", "Rich/Poor", "Educated/Uneducated", "Digital/Real"]


def parse(text): return CICOParser.parse_transition_delta(text)


def board(mask):
    g = Graph()
    for i, c in enumerate(CELLS4):
        if mask >> i & 1: g.add_node(c, 0)
    return g


def mask_of(g): return sum(1 << i for i, c in enumerate(CELLS4) if c in g.nodes)


def add_text(i): return f"ADD[{CELLS4[i][0]},{CELLS4[i][1]}:0]"


def del_text(i): return f"DEL[{CELLS4[i][0]},{CELLS4[i][1]}]"


def admitted_moves():
    """{mask: [mask after each single-cell delta the gate admits]} and the refusal reasons seen."""
    moves, reasons = {}, set()
    for m in range(16):
        moves[m] = []
        for i in range(4):
            for text in (add_text(i), del_text(i)):
                g = board(m)
                r = check(g, parse(text))
                if r.ok:
                    apply(g, parse(text)); moves[m].append(mask_of(g))
                else:
                    reasons.add(r.reason.split(":")[0])
    return moves, reasons


def matmul(a, b): return [[sum(a[i][k] * b[k][j] for k in range(16)) for j in range(16)] for i in range(16)]


def test_the_gate_admits_exactly_four_one_cell_moves_per_board_and_they_form_the_4_cube():
    assert len(AXES) == 4
    moves, reasons = admitted_moves()
    assert all(len(v) == 4 for v in moves.values())                           # degree 4 everywhere
    assert reasons == {"identification", "match"}                             # the four refused deltas per board, and why
    edges = {frozenset((m, n)) for m, vs in moves.items() for n in vs}
    assert len(edges) == 32 == 16 * 4 // 2
    assert all(bin(m ^ n).count("1") == 1 for m, ns in moves.items() for n in ns)       # a move differs on exactly one axis
    assert all(bin(m).count("1") % 2 != bin(n).count("1") % 2 for m, ns in moves.items() for n in ns)    # bipartite by parity
    for start in (0, 5, 15):                                                  # BFS distance = Hamming distance
        dist, frontier = {start: 0}, [start]
        while frontier:
            nxt = []
            for m in frontier:
                for n in moves[m]:
                    if n not in dist: dist[n] = dist[m] + 1; nxt.append(n)
            frontier = nxt
        assert all(dist[m] == bin(m ^ start).count("1") for m in range(16)) and max(dist.values()) == 4


def test_the_24_shortest_paths_between_the_empty_and_the_full_board_are_all_admitted_and_end_in_the_same_board():
    full, empty = board(15), board(0)
    for order in permutations(range(4)):
        g = board(0)
        for i in order:
            d = parse(add_text(i)); assert check(g, d).ok; apply(g, d)
        assert same(g, full)
        for i in order:
            d = parse(del_text(i)); assert check(g, d).ok; apply(g, d)
        assert same(g, empty)
    assert len(list(permutations(range(4)))) == 24
    one_shot = parse(" ".join(add_text(i) for i in range(4)))
    g = board(0); assert check(g, one_shot).ok; apply(g, one_shot)
    assert same(g, full)
    # count shortest paths 0 -> 15 in the admitted-move graph by levels
    moves, _ = admitted_moves()
    level = {0: 1}
    for _ in range(4):
        nxt = {}
        for m, c in level.items():
            for n in moves[m]:
                if bin(n).count("1") == bin(m).count("1") + 1: nxt[n] = nxt.get(n, 0) + c
        level = nxt
    assert level == {15: 24}


def test_closed_walks_follow_the_cube_spectrum_odd_lengths_are_impossible_and_a_five_step_loop_needs_a_no_op():
    moves, _ = admitted_moves()
    A = [[1 if j in moves[i] else 0 for j in range(16)] for i in range(16)]
    P = [[int(i == j) for j in range(16)] for i in range(16)]
    for n in range(1, 9):
        P = matmul(P, A)
        trace = sum(P[i][i] for i in range(16))
        assert trace == 4 ** n + 4 * 2 ** n + 4 * (-2) ** n + (-4) ** n
        if n % 2: assert trace == 0                                            # no closed walk of odd length
    closed = []
    for seq in product(range(5), repeat=5):            # 0..3 = flip that cell, 4 = no-op
        m = 0
        for s in seq:
            if s < 4: m ^= 1 << s
        if m == 0: closed.append(seq)
    assert len(closed) == 241 == 200 + 40 + 1                                  # one no-op: 5 x 40; three no-ops: 10 x 4; five no-ops: 1
    assert all(sum(1 for s in seq if s == 4) % 2 == 1 for seq in closed)
    no_noop = [seq for seq in closed if 4 not in seq]
    assert no_noop == []                                                       # a five-step loop with no no-op does not exist
    g = board(0)                                                               # and the no-op is a delta the gate and parser really accept
    empty = parse("")
    assert check(g, empty).ok
    apply(g, empty)
    assert same(g, board(0))


def test_n_pairs_have_2n_sides_and_2_to_the_n_combinations_and_the_4_cube_has_16_32_24_8_and_384_symmetries():
    assert [(2 * n, 2 ** n) for n in range(1, 9)] == [(2, 2), (4, 4), (6, 8), (8, 16), (10, 32), (12, 64), (14, 128), (16, 256)]
    assert [n for n in range(1, 9) if 2 * n == 2 ** n] == [1, 2]
    def cube_counts(n):
        verts = range(2 ** n)
        edges = [(a, b) for a in verts for b in verts if a < b and bin(a ^ b).count("1") == 1]
        squares = [(a, a ^ (1 << i), a ^ (1 << j), a ^ (1 << i) ^ (1 << j)) for a in verts for i in range(n) for j in range(i + 1, n) if not (a >> i & 1) and not (a >> j & 1)]
        cubes = [a for a in verts for i in range(n) for j in range(i + 1, n) for k in range(j + 1, n) if not (a >> i & 1 or a >> j & 1 or a >> k & 1)]
        return len(verts), len(edges), len(squares), len(cubes)
    assert cube_counts(3) == (8, 12, 6, 1)                                     # the cuboid: 8 corners, 12 lines, 6 faces, 1 solid
    assert cube_counts(4) == (16, 32, 24, 8)
    assert 16 - 32 + 24 - 8 == 0
    moves, _ = admitted_moves()
    adj = {m: set(moves[m]) for m in moves}
    order = [0] + sorted(adj[0]) + [m for m in range(16) if m != 0 and m not in adj[0]]
    count = 0

    def extend(img, k):
        nonlocal count
        if k == 16: count += 1; return
        v = order[k]
        for w in range(16):
            if w in img.values(): continue
            if all((w in adj[img[u]]) == (u in adj[v]) for u in order[:k]):
                img[v] = w; extend(img, k + 1); del img[v]
    extend({}, 0)
    assert count == 384 == 2 ** 4 * 24


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
