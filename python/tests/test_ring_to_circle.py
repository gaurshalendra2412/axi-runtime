"""Implements concept-map row 133 (thirteenth scan): the 12 Aug page "THE JOURNEY FROM FRAME TO CIRCLE" (Post 307, byline rajnish choubey, about 2,500 words, two reads).
Quotes verified on the second read (YES): Truth 2 "The Cuboid of Consciousness" ("every rectangle is actually a cuboid"); Truth 3 "The journey is to go from the rectangle to the
circle"; Truth 5 "The Circle Has No Corners"; Truth 6 "All Lines Become One", where the four lines "began to curve", "began to bend" and "They became one continuous line."

The reading being tested (Rajnish to accept or reject): a RECTANGLE is a ring of four places (four corners, four lines); a CIRCLE is the same ring with the corners taken away until
one continuous line is left; "every rectangle is actually a cuboid" is the ring lifted one step to a frame of twelve lines. The page gives no method for "became one line", so this
file supplies one that can be counted, using the repository's own gate and nothing else:
  1. A ring of n places has ONE independent loop (edges - nodes + 1 = 1) and nodes - edges = 0, for every n from 1 (one place with a line back to itself) to 8. It has n symmetries
     (the rotations) when every line carries the same relation name. Built from the empty host by ONE delta each.
  2. The four lines can be made one line, step by step, by the gate: "contract" a line = delete the line and its dropped end, re-point the other lines of that end. Each step is one
     composite delta the gate admits (4 -> 3 -> 2 -> 1 place; the last step leaves one place and one line pointing back at it). At every step the loop count stays 1 and nodes - edges
     stays 0; only the symmetry count changes (4, 3, 2, 1). The gate refuses the same step when the old lines at the dropped end are not deleted ("dangling edge"). The inverse delta
     (taken before the step) restores the previous ring exactly, so "circle -> rectangle" is "subdivide a line", which is also admitted.
  3. Counted this way, "four lines become one line" is TRUE in the sense of loops and of nodes - edges (1 = 1, 0 = 0) and FALSE if "one line" is read as "no line": deleting the last
     self-line is the only step that changes the loop count (1 -> 0) and nodes - edges (0 -> 1). So "4 = 1" survives, "1 = 0" does not.
  4. "Every rectangle is actually a cuboid": ONE delta (4 new places, 4 new lines on top, 4 verticals) lifts the 4-ring to the 8-node, 12-line wire frame. It has 5 independent loops
     by four routes (edges - nodes + components; a spanning forest; the GF(2) rank of the six face boundaries; 12 minus the rank of the incidence matrix over GF(2)); 6 faces; every line
     lies on exactly 2 faces; nodes - edges + faces = 2. The six face boundaries have one dependency (their sum is zero), which is why 6 faces give 5 loops.
  5. The cuboid frame contracts to ONE place with FIVE self-lines (one per independent loop) by seven tree-edge contractions admitted by the gate, keeping the loop count 5 at every step.
     Distinct relation names are needed for this: with one shared name, two lines that land on the same pair of places collapse into one key and the gate refuses ("identification").
What they do NOT show: that the page means a ring by "rectangle" or that "became one continuous line" is a contraction; that a circle is a one-place ring (a topologist's circle has no
corners because corners are a property of a drawing, not of the graph - the corners here are only the count of places); that "every rectangle is a cuboid" is a claim about lifting a ring
(the page gives it as a sentence); that any of this applies to a model. Loop counting, Euler's formula and GF(2) cycle spaces are standard mathematics, not findings of ours."""
import os, sys
from itertools import combinations
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.cico_parser import CICOParser
from axi.engine.gate import Graph, check, apply
from axi.engine.inverse import invert, to_text
from test_line_and_cycle import CELLS, loop_count
from test_shape_of_four import graph_automorphisms
from test_signed_slots import etext, same


def parse(text): return CICOParser.parse_transition_delta(text)


def ring_text(n, rel="then"):
    nodes = " ".join(f"ADD[{CELLS[i][0]},{CELLS[i][1]}:0]" for i in range(n))
    edges = " ".join(etext("ADD", (CELLS[i], CELLS[(i + 1) % n], rel), 1) for i in range(n))
    return nodes + " " + edges


def build(text):
    g = Graph()
    d = parse(text)
    assert check(g, d).ok, check(g, d).reason
    apply(g, d)
    return g


def loops(g): return loop_count(list(g.nodes), [(u, v) for (u, v, _) in g.edges])


def contraction_text(g, key):
    """One composite delta: contract the line `key` = (a, b, rel). The place b is dropped and every other line at b is re-pointed to a."""
    a, b, _ = key
    assert a != b
    items, repointed = [], []
    for k in sorted(g.incident(b)):
        items.append(etext("DEL", k, g.edges[k]))
        if k != key:
            u, v, rel = k
            repointed.append(etext("ADD", (a if u == b else u, a if v == b else v, rel), g.edges[k]))
    return " ".join(items + [f"DEL[{b[0]},{b[1]}]"] + repointed)


def find_by_rel(g, rel): return [k for k in g.edges if k[2] == rel]


def test_every_ring_from_1_to_8_places_has_one_loop_and_as_many_lines_as_places():
    for n in range(1, 9):
        g = build(ring_text(n))
        assert (len(g.nodes), len(g.edges)) == (n, n)
        assert len(g.nodes) - len(g.edges) == 0 and loops(g) == 1
        if n <= 7: assert len(graph_automorphisms(g)) == n          # the n rotations (n = 1: the single self-line has only the identity)
    assert (1, 1) == (len(build(ring_text(1)).nodes), len(build(ring_text(1)).edges))
    assert CELLS[0] in build(ring_text(1)).nodes and (CELLS[0], CELLS[0], "then") in build(ring_text(1)).edges     # a self-line is a legal edge key


def test_the_four_ring_contracts_to_one_place_with_one_self_line_and_every_step_keeps_the_loop():
    g = build(ring_text(4))
    sizes, lps, syms = [(4, 4)], [1], [len(graph_automorphisms(g))]
    for _ in range(3):
        key = next(k for k in sorted(g.edges) if k[0] != k[1])               # any line that is not already a self-line
        step = parse(contraction_text(g, key))
        assert check(g, step).ok, check(g, step).reason
        undo = to_text(invert(g, step))
        snapshot = Graph()
        for n, v in g.nodes.items(): snapshot.add_node(n, v)
        for (u, v, r), w in g.edges.items(): snapshot.add_edge(u, v, r, w)
        apply(g, step)
        assert g.is_well_formed()
        sizes.append((len(g.nodes), len(g.edges))); lps.append(loops(g)); syms.append(len(graph_automorphisms(g)))
        back = parse(undo)
        assert check(g, back).ok                                              # circle -> rectangle: the inverse is a subdivision and is admitted
        again = Graph()
        for n, v in g.nodes.items(): again.add_node(n, v)
        for (u, v, r), w in g.edges.items(): again.add_edge(u, v, r, w)
        apply(again, back)
        assert same(again, snapshot)
    assert sizes == [(4, 4), (3, 3), (2, 2), (1, 1)] and lps == [1, 1, 1, 1]
    assert syms == [4, 3, 2, 1]                                              # only the symmetry changes
    (only,) = g.edges
    assert only[0] == only[1]                                                # the last line points back at its own place


def test_the_gate_refuses_a_contraction_that_leaves_a_line_hanging_and_deleting_the_last_self_line_is_the_only_step_that_changes_the_loop():
    g = build(ring_text(4))
    key = (CELLS[3], CELLS[0], "then")
    full = contraction_text(g, key).split(" ")
    other = etext("DEL", (CELLS[0], CELLS[1], "then"), 1)                    # the place CELLS[0] is dropped; its other line is CELLS[0] -> CELLS[1]
    assert other in full
    hanging = " ".join(x for x in full if x != other)
    assert hanging != " ".join(full)
    refused = check(g, parse(hanging))
    assert not refused.ok and refused.reason.startswith("dangling edge")
    while len(g.nodes) > 1:
        apply(g, parse(contraction_text(g, next(k for k in sorted(g.edges) if k[0] != k[1]))))
    assert (len(g.nodes), len(g.edges), loops(g)) == (1, 1, 1)
    (k,) = g.edges
    drop = parse(etext("DEL", k, g.edges[k]))
    assert check(g, drop).ok
    apply(g, drop)
    assert (len(g.nodes), len(g.edges), loops(g)) == (1, 0, 0)               # one place, no line: nodes - edges = 1, no loop
    assert len(g.nodes) - len(g.edges) == 1                                  # "4 = 1" kept both numbers; "1 = 0" changes both


def test_one_shared_relation_name_cannot_hold_two_parallel_lines_so_contraction_needs_names():
    x, y, z = CELLS[0], CELLS[1], CELLS[2]
    shared = build(f"ADD[{x[0]},{x[1]}:0] ADD[{y[0]},{y[1]}:0] ADD[{z[0]},{z[1]}:0] " + " ".join(
        etext("ADD", e, 1) for e in ((x, y, "r"), (x, z, "r"), (y, z, "r"))))
    key = (y, z, "r")
    step = parse(contraction_text(shared, key))                              # dropping z re-points x->z onto x->y, which already exists with the same name
    refused = check(shared, step)
    assert not refused.ok and refused.reason.startswith("identification")
    named = build(f"ADD[{x[0]},{x[1]}:0] ADD[{y[0]},{y[1]}:0] ADD[{z[0]},{z[1]}:0] " + " ".join(
        etext("ADD", e, 1) for e in ((x, y, "a"), (x, z, "b"), (y, z, "c"))))
    ok = parse(contraction_text(named, (y, z, "c")))
    assert check(named, ok).ok
    apply(named, ok)
    assert sorted(named.edges) == sorted([(x, y, "a"), (x, y, "b")]) and loops(named) == 1      # two parallel named lines = one loop


def cube():
    """Wire frame of a cuboid: place i has bits (x, y, z); a line joins places that differ in one bit. Returns (delta text, edge list [(i, j, rel)])."""
    edges = []
    for i, j in combinations(range(8), 2):
        if bin(i ^ j).count("1") == 1: edges.append((i, j, f"e{len(edges)}"))
    return edges


def gf2_rank(rows):
    rows, rank = list(rows), 0
    while rows:
        pivot = rows.pop()
        if pivot == 0: continue
        rank += 1
        low = pivot & -pivot
        rows = [r ^ pivot if r & low else r for r in rows]
    return rank


def test_one_delta_lifts_the_four_ring_to_the_cuboid_frame_with_five_loops_by_four_routes():
    edges = cube()
    assert len(edges) == 12
    bottom = [(i, j, r) for (i, j, r) in edges if i < 4 and j < 4]
    ring = build(" ".join(f"ADD[{CELLS[i][0]},{CELLS[i][1]}:0]" for i in range(4)) + " " + " ".join(
        etext("ADD", (CELLS[i], CELLS[j], r), 1) for (i, j, r) in bottom))
    assert (len(ring.nodes), len(ring.edges), loops(ring)) == (4, 4, 1)
    rest = [(i, j, r) for (i, j, r) in edges if not (i < 4 and j < 4)]
    lift = parse(" ".join(f"ADD[{CELLS[i][0]},{CELLS[i][1]}:0]" for i in range(4, 8)) + " " + " ".join(
        etext("ADD", (CELLS[i], CELLS[j], r), 1) for (i, j, r) in rest))
    assert check(ring, lift).ok
    undo = to_text(invert(ring, lift)); apply(ring, lift)
    assert (len(ring.nodes), len(ring.edges)) == (8, 12)
    route1 = len(ring.edges) - len(ring.nodes) + 1                           # edges - nodes + components (one component)
    route2 = loops(ring)                                                      # a spanning forest
    faces = []
    for axis in range(3):
        for val in (0, 1):
            members = {i for i in range(8) if (i >> axis) & 1 == val}
            faces.append([n for n, (i, j, r) in enumerate(edges) if i in members and j in members])
    assert len(faces) == 6 and all(len(f) == 4 for f in faces)
    route3 = gf2_rank(sum(1 << n for n in f) for f in faces)
    columns = []                                                              # incidence matrix over GF(2), one column per line, one bit per place
    for (i, j, r) in edges: columns.append((1 << i) | (1 << j))
    assert all(bin(c).count("1") == 2 for c in columns)                       # every line touches exactly two places
    assert gf2_rank(columns) == 7                                             # places minus components: the incidence matrix has rank 8 - 1
    route4 = len(edges) - gf2_rank(columns)
    assert route1 == route2 == route3 == route4 == 5
    total = 0
    for f in faces: total ^= sum(1 << n for n in f)
    assert total == 0                                                        # the six face boundaries add to zero: the one dependency
    per_line = [sum(1 for f in faces if n in f) for n in range(12)]
    assert per_line == [2] * 12                                              # every line lies on exactly two faces
    assert len(ring.nodes) - len(ring.edges) + len(faces) == 2
    back = parse(undo)
    assert check(ring, back).ok                                              # the inverse takes the cuboid frame back down to the rectangle
    again = Graph()
    apply(again, parse(" ".join(f"ADD[{CELLS[i][0]},{CELLS[i][1]}:0]" for i in range(4)) + " " + " ".join(
        etext("ADD", (CELLS[i], CELLS[j], r), 1) for (i, j, r) in bottom)))
    apply(ring, back)
    assert same(ring, again)


def test_the_cuboid_frame_contracts_to_one_place_and_five_self_lines_keeping_five_loops_at_every_step():
    edges = cube()
    g = build(" ".join(f"ADD[{CELLS[i][0]},{CELLS[i][1]}:0]" for i in range(8)) + " " + " ".join(
        etext("ADD", (CELLS[i], CELLS[j], r), 1) for (i, j, r) in edges))
    parent = list(range(8))

    def find(a):
        while parent[a] != a: a = parent[a]
        return a
    tree = []
    for (i, j, r) in edges:                                                   # a spanning tree by union-find (7 lines)
        if find(i) != find(j): parent[find(i)] = find(j); tree.append(r)
    assert len(tree) == 7
    seen = [(len(g.nodes), len(g.edges), loops(g))]
    for rel in tree:
        (key,) = find_by_rel(g, rel)
        assert key[0] != key[1]                                               # a tree line never becomes a self-line before its turn
        step = parse(contraction_text(g, key))
        assert check(g, step).ok, check(g, step).reason
        apply(g, step)
        assert g.is_well_formed()
        seen.append((len(g.nodes), len(g.edges), loops(g)))
    assert seen[0] == (8, 12, 5) and seen[-1] == (1, 5, 5)
    assert [s[2] for s in seen] == [5] * 8                                    # contracting a non-loop line never changes the loop count
    assert all(u == v for (u, v, _) in g.edges) and len({rel for (_, _, rel) in g.edges}) == 5     # five self-lines, five distinct names


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
