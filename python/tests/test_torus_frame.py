"""Implements concept-map item 121 (a wrap-around address frame in front of the gate; python/axi/engine/torus.py; docs/TORUS_FRAME.md), test 147.
Source: two of Rajnish Choubey's articles ("The India of Zero & Infinity", "The 99 Eyewitnesses of the Ramayana") as an AI summary pasted
on 24 Oct describes them, with code. Run here, that code accepted three wrong steps: an address outside its 108 points (200 was written
to point 92), a move onto an occupied point (the object there vanished), and a move from an empty point. It did so because, once every
number wraps, "the address exists" is always true. The module keeps the wrap and puts the gate behind it. This file checks:
  1. the three wrong steps are refused here (strict address; identification; match), and the state is left as it was;
  2. a move carries its edges, commits whole, and `invert` undoes it;
  3. the ring arithmetic: wrap, offset in [0, n), the short-way distance, the neighbours, the adjacency edges (108 points in a ring:
     108 edges), whole laps are no move, shifts compose like the group Z_rows x Z_cols;
  4. on a 5 x 7 frame, for every one of the 35 shifts and random states and deltas, the gate's verdict and kind of reason do not change and
     apply-then-shift equals shift-then-apply (the general fact for any relabelling is tests/test_gate_frame_covariance.py);
  5. the arithmetic of the 108-point mesh: 5,778 edges, and a surface with 910 handles to carry it without crossings (a torus has one;
     seven points is the most a torus carries fully joined);
  6. the page and the front pages (concept map item 121, VERIFIED row, the counter).
What it does NOT show: that a model proposes better on a wrap-around frame, anything about ARC, or anything about the articles beyond the
pasted summary. No model is called."""
import os, random, sys
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.cico_parser import CICOParser, EdgeDelta, NodeDelta, StateTransitionDelta
from axi.engine.gate import Graph, check, apply
from axi.engine.inverse import invert
from axi.engine.torus import AddressError, TorusFrame, complete_graph_edges, complete_graph_genus

HERE = os.path.dirname(__file__)
DOCS = os.path.join(HERE, "..", "..", "docs")
PAGE = os.path.join(DOCS, "TORUS_FRAME.md")
NEXT = "Next free: row 172, item 125, test 151, question 121"
P = CICOParser.parse_transition_delta


def same(a: Graph, b: Graph) -> bool: return a.nodes == b.nodes and a.edges == b.edges


def clone(g: Graph) -> Graph:
    h = Graph()
    for n, v in g.nodes.items(): h.add_node(n, v)
    for (u, v, rel), w in g.edges.items(): h.add_edge(u, v, rel, w)
    return h


def kind(reason: str) -> str: return reason.split(":")[0]


def ring108(strict=True):
    t = TorusFrame(1, 108, "strict" if strict else "wrap"); g = Graph(); return t, g


def test_the_three_wrong_steps_of_the_pasted_code_are_refused_here_and_the_state_is_left_as_it_was():
    t, g = ring108(); g.add_node((0, 5), 1)                                              # object A at point 5
    # 1. an address outside the 108 points
    try: t.place_delta(P("ADD[0,200:7]")); assert False, "strict mode must refuse 200"
    except AddressError as e: assert "outside the 1x108 frame" in str(e)
    tw = TorusFrame(1, 108, "wrap"); placed, wrapped = tw.place_delta(P("ADD[0,200:7]"))
    assert (placed.node_deltas[0].r, placed.node_deltas[0].c) == (0, 92) and wrapped == [((0, 200), (0, 92))]   # wrapped, and reported
    # 2. a move onto an occupied point: identification refuses, nothing changes
    g.add_node((0, 10), 2); before = clone(g)                                             # object B at point 10
    d = t.move_delta(g, (0, 5), 0, 5); r = check(g, d)
    assert not r.ok and kind(r.reason) == "identification" and same(g, before) and g.nodes[(0, 10)] == 2
    # 3. a move from an empty point: match refuses
    d = t.move_delta(g, (0, 50), 0, 3); r = check(g, d); assert not r.ok and kind(r.reason) == "match" and same(g, before)
    # and the legal move of the pasted demo: 5 + 105 wraps to 2
    d = t.move_delta(g, (0, 5), 0, 105); assert check(g, d).ok; apply(g, d)
    assert g.nodes.get((0, 2)) == 1 and (0, 5) not in g.nodes and g.nodes[(0, 10)] == 2


def test_a_move_carries_its_edges_commits_whole_and_invert_undoes_it():
    t = TorusFrame(4, 6); g = Graph()
    for n, v in (((0, 0), 1), ((0, 5), 2), ((3, 0), 3), ((2, 2), 4)): g.add_node(n, v)
    g.add_edge((0, 0), (0, 5), "dep", 3); g.add_edge((3, 0), (0, 0), "dep", -2); g.add_edge((0, 0), (0, 0), "self", 9)
    before = clone(g)
    d = t.move_delta(g, (0, 0), -1, -1)                                                  # goes off two edges at once: wraps to (3, 5)
    assert (d.node_deltas[1].r, d.node_deltas[1].c) == (3, 5) and check(g, d).ok
    undo = invert(g, d); apply(g, d)
    assert g.nodes[(3, 5)] == 1 and (0, 0) not in g.nodes
    assert g.edges == {((3, 5), (0, 5), "dep"): 3, ((3, 0), (3, 5), "dep"): -2, ((3, 5), (3, 5), "self"): 9, } and g.is_well_formed()
    assert check(g, undo).ok; apply(g, undo); assert same(g, before)
    assert t.move_delta(g, (0, 0), 4, 12) is None and t.move_delta(g, (0, 0), -8, 6) is None   # whole laps: no change


def test_the_ring_arithmetic():
    t = TorusFrame(5, 7)
    assert t.size == 35 and len(t.cells()) == 35 and t.wrap((-1, -1)) == (4, 6) and t.wrap((12, 15)) == (2, 1)
    for a in t.cells():
        for b in t.cells():
            dr, dc = t.offset(a, b); assert 0 <= dr < 5 and 0 <= dc < 7 and t.shift(a, dr, dc) == b
            assert t.distance(a, b) == t.distance(b, a) <= 2 + 3 and (t.distance(a, b) == 0) == (a == b)
        assert len(t.neighbours(a)) == 4 and all(t.distance(a, m) == 1 for m in t.neighbours(a))
    assert len(t.adjacency()) == 2 * 35 and len({(a, b) for a, b, _, _ in t.adjacency()}) == 70
    r = TorusFrame(1, 108); assert len(r.adjacency()) == 108 and r.neighbours((0, 0)) == [(0, 107), (0, 1)]     # a ring of 108: 108 edges
    assert TorusFrame(2, 2).neighbours((0, 0)) == [(1, 0), (0, 1)]                        # no repeats when the ring is short
    try: t.address((5, 0)); assert False
    except AddressError: pass
    assert TorusFrame(5, 7, "wrap").address((5, -1)) == (0, 6) and t.address((4, 6)) == (4, 6)
    for bad in ((0, 3), (3, 0)):
        try: TorusFrame(*bad); assert False
        except ValueError: pass
    try: TorusFrame(2, 2, "loose"); assert False
    except ValueError: pass
    # shifts compose like Z_5 x Z_7: shift by a then by b is shift by a + b, and every shift has an inverse
    rng = random.Random(1)
    for _ in range(200):
        n = rng.choice(t.cells()); a, b = (rng.randint(-20, 20), rng.randint(-20, 20)), (rng.randint(-20, 20), rng.randint(-20, 20))
        assert t.shift(t.shift(n, *a), *b) == t.shift(n, a[0] + b[0], a[1] + b[1]) and t.shift(t.shift(n, *a), -a[0], -a[1]) == n
    assert len({tuple(t.shift(n, dr, dc) for n in t.cells()) for dr in range(5) for dc in range(7)}) == 35   # 35 distinct actions


def random_state(t: TorusFrame, rng: random.Random) -> Graph:
    g = Graph()
    for n in rng.sample(t.cells(), rng.randint(3, 12)): g.add_node(n, rng.randint(-3, 3))
    ns = list(g.nodes)
    for _ in range(rng.randint(0, 10)):
        u, v = rng.choice(ns), rng.choice(ns); rel = rng.choice(["dep", "adj"])
        if (u, v, rel) not in g.edges: g.add_edge(u, v, rel, rng.randint(-2, 4))
    return g


def random_delta(t: TorusFrame, g: Graph, rng: random.Random) -> StateTransitionDelta:
    """Some steps are legal and many are not: deletes of present and absent nodes, adds onto free and taken cells, edges with right and wrong weights."""
    nodes, edges = [], []
    for _ in range(rng.randint(1, 3)):
        n = rng.choice(t.cells())
        nodes.append(NodeDelta("DEL", *n) if rng.random() < .5 else NodeDelta("ADD", *n, rng.randint(-3, 3)))
    for _ in range(rng.randint(0, 3)):
        if g.edges and rng.random() < .5:
            (u, v, rel), w = rng.choice(sorted(g.edges.items())); edges.append(EdgeDelta("DEL", u, v, rel, w if rng.random() < .8 else w + 1))
        else:
            edges.append(EdgeDelta("ADD", rng.choice(t.cells()), rng.choice(t.cells()), "adj", 1))
    return StateTransitionDelta(nodes, edges)


def test_on_a_5_by_7_frame_the_gate_does_not_care_where_round_the_torus_a_state_sits():
    t = TorusFrame(5, 7); rng = random.Random(20261024); seen = {"ok": 0, "refused": 0}
    for _ in range(60):
        g = random_state(t, rng); d = random_delta(t, g, rng); r = check(g, d); seen["ok" if r.ok else "refused"] += 1
        for dr in range(5):
            for dc in range(7):
                gs, ds = t.shift_graph(g, dr, dc), t.shift_delta(d, dr, dc); rs = check(gs, ds)
                assert rs.ok == r.ok and (r.ok or kind(rs.reason) == kind(r.reason))
                if r.ok:
                    a = clone(g); apply(a, d); b = clone(gs); apply(b, ds); assert same(t.shift_graph(a, dr, dc), b)
    assert seen["ok"] >= 5 and seen["refused"] >= 5, seen                                # both kinds of verdict were exercised


def test_the_arithmetic_of_the_108_point_mesh():
    assert complete_graph_edges(108) == 5778 and complete_graph_edges(7) == 21
    assert [complete_graph_genus(n) for n in (3, 4, 5, 6, 7, 8, 12)] == [0, 0, 1, 1, 1, 2, 6]
    assert complete_graph_genus(108) == 910


def test_the_page_and_the_front_pages_say_what_was_built_and_the_counter_moved():
    page = open(PAGE).read(); low = page.lower()
    for s in ("torus.py", "Rajnish Choubey", "The India of Zero & Infinity", "The 99 Eyewitnesses of the Ramayana", "200 was written to point 92",
              "identification", "match", "5,778", "910", "seven points", "108 edges", "no model", "test 147", "strict", "wrap"):
        assert s.lower() in low, s
    for bad in ("proven", "$", "dollar", "100 percent"): assert bad not in low, bad
    cm = open(os.path.join(DOCS, "RAJNISH_CONCEPT_MAP.md")).read(); vf = open(os.path.join(DOCS, "VERIFIED.md")).read()
    assert NEXT in cm and cm.count("Next free:") == 1 and "test 147 checks the wrap-around address frame, item 121" in cm
    j = cm.index("\n121. **") + 1; item = cm[j:cm.index("\n", j)]
    assert "torus.py" in item and "TORUS_FRAME.md" in item and "test 147" in item and "Rajnish Choubey" in item
    row = next(l for l in vf.split("\n") if "TORUS_FRAME.md" in l); assert "test_torus_frame.py" in row and "no model" in row.lower()


if __name__ == "__main__":
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]; bad = 0
    for n, f in fns:
        try: f(); print("ok  ", n)
        except Exception: bad += 1; print("FAIL", n); traceback.print_exc()
    print(f"{len(fns) - bad}/{len(fns)} passed"); sys.exit(1 if bad else 0)
