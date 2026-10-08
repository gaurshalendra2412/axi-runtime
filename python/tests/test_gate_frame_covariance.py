"""Implements concept-map rows 79 and 80.
  Row 79: blog 3 Aug, page "The Hinduism: Chapter 1" (section "The Relativistic Clash: Two Stationary Frames"). Read twice. The page says
          each person's own mind is "the default frame of reference" and that each mistakes local coordinates for the absolute universe;
          Rahi's Law ("Everyone thinks they have it") is the human version.
  Row 80: blog 4 Aug, page "The Hinduism: The Final Award", sections "The Mechanical Trap of the Three Subsets" and "Why Asimov's Laws Create
          the Mad Machine". The page calls Asimov's three laws "subsets" that fight each other and Rahi's Law "the absolute superset of all
          rules"; its stated failure is a machine with contradictory restrictions that has no defined answer (a "definition failure").
          (The page asserts this; it gives no derivation.)

The reading being tested (Rajnish to accept or reject): a verdict that is fit to be the witness must not depend on WHICH frame the speaker
uses, and must always have a defined answer. In this repository that is three checkable facts about `check` and `apply`:
  1. FRAME INDEPENDENCE. Rename the slots by any bijection of the 9 cells (the 8 D4 transforms of the 3x3 board, and random permutations
     of the cells) in the host AND in the delta. The verdict is the same, the kind of rejection is the same, and applying then renaming
     equals renaming then applying. So the gate sees slot IDENTITY only. It does not see where a slot is.
  2. ORDER INDEPENDENCE. Shuffling the items of a delta never changes admit/reject, and never changes the graph an admitted delta produces.
     The conditions are a conjunction (every one must hold), not a ranked list, so there is nothing to conflict.
  3. TOTALITY. On every parsed delta the gate returns an answer: admit, or reject with one of exactly four named kinds of reason. It never
     raises and never returns an empty reason on a rejection. (That includes slots far outside the 3x3 board.)

An honest consequence, which is the main point of this file: the gate has NO GEOMETRY of its own. A node at (99, 99) is admitted on a 3x3
board (test 3 below), and (0,0) and (0,1) are not "neighbours" unless the task supplies an `adj` edge between them. Shape, adjacency and
symmetry live in the canonicalizer (d4.py) and in the task, not in the gate or the grammar. Frame independence is the other face of that.
What they do NOT show: that the post's "frames" are D4 or slot renamings (they are about people's viewpoints); that a model respects any
frame; anything about contradictory rules in text (the gate's four conditions are fixed code and cannot contradict by construction, which is
a design property, not a measurement)."""
import os, random, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.cico_parser import CICOParser, NODE_ADD, NODE_DEL, EDGE
from axi.engine.d4 import apply as d4apply, IDS
from axi.engine.gate import Graph, check, apply
from test_signed_slots import same, copy, random_graph, random_delta_text, CELLS

P = CICOParser.parse_transition_delta
KINDS = {"match", "identification", "dangling edge", "well-formedness"}
IDX = np.arange(9).reshape(3, 3)


def d4_relabel(t):
    """old cell -> new cell when the board is transformed by D4 element t (taken from axi.engine.d4, not re-derived)."""
    flat = [int(x) for x in d4apply(IDX, t).ravel()]          # position p shows old cell flat[p]
    return {(q // 3, q % 3): (p // 3, p % 3) for p, q in enumerate(flat)}


def relabel_graph(g, m):
    h = Graph()
    for n, v in g.nodes.items(): h.add_node(m[n], v)
    for (u, v, rel), w in g.edges.items(): h.add_edge(m[u], m[v], rel, w)
    return h


def relabel_text(txt, m):
    if txt == "": return ""
    out = []
    for it in txt.split(" "):
        if (x := NODE_ADD.fullmatch(it)):
            r, c = m[(int(x[1]), int(x[2]))]; out.append(f"ADD[{r},{c}:{x[3]}]")
        elif (x := NODE_DEL.fullmatch(it)):
            r, c = m[(int(x[1]), int(x[2]))]; out.append(f"DEL[{r},{c}]")
        else:
            x = EDGE.fullmatch(it); assert x, it
            a, b = m[(int(x[2]), int(x[3]))], m[(int(x[4]), int(x[5]))]
            out.append(f"{x[1]}[({a[0]},{a[1]})->({b[0]},{b[1]}):{x[6]}#{x[7]}]")
    return " ".join(out)


def kind(reason): return reason.split(":")[0] if reason else ""


def test_d4_relabelling_really_is_the_eight_transforms():
    maps = [tuple(sorted(d4_relabel(t).items())) for t in IDS]
    assert len(set(maps)) == 8
    assert all(sorted(m.values()) == sorted(CELLS) for m in map(dict, maps))                  # each is a bijection of the 9 cells
    assert d4_relabel(0) == {c: c for c in CELLS}                                             # t = 0 is the identity


def test_verdict_and_result_do_not_depend_on_the_frame():
    rng = random.Random(36); ok = bad = checked = 0
    for _ in range(4000):
        g = random_graph(rng); txt = random_delta_text(g, rng)
        r0 = check(g, P(txt)); ok += r0.ok; bad += not r0.ok
        maps = [d4_relabel(t) for t in IDS]
        for _k in range(3):
            sh = list(CELLS); rng.shuffle(sh); maps.append(dict(zip(CELLS, sh)))               # arbitrary renamings, not only D4
        for m in maps:
            g2, d2 = relabel_graph(g, m), P(relabel_text(txt, m))
            r2 = check(g2, d2)
            assert r2.ok == r0.ok and kind(r2.reason) == kind(r0.reason), (txt, m)
            if r0.ok:
                h, h2 = copy(g), copy(g2); apply(h, P(txt)); apply(h2, d2)
                assert same(relabel_graph(h, m), h2)                                          # apply-then-rename == rename-then-apply
            checked += 1
    assert ok > 800 and bad > 800 and checked >= 4000 * 11, (ok, bad, checked)


def test_item_order_never_changes_the_verdict_or_the_result():
    rng = random.Random(37); ok = bad = 0
    for _ in range(6000):
        g = random_graph(rng); txt = random_delta_text(g, rng)
        items = txt.split(" ") if txt else []
        r0 = check(g, P(txt))
        ok += r0.ok; bad += not r0.ok
        for _k in range(4):
            rng.shuffle(items); t2 = " ".join(items)
            r2 = check(g, P(t2))
            assert r2.ok == r0.ok, (txt, t2)
            if r0.ok:
                h, h2 = copy(g), copy(g); apply(h, P(txt)); apply(h2, P(t2))
                assert same(h, h2)
    assert ok > 1000 and bad > 1000, (ok, bad)


def test_the_gate_always_answers_and_has_no_board_of_its_own():
    rng = random.Random(38); seen = set()
    for _ in range(20000):
        g = random_graph(rng); txt = random_delta_text(g, rng)
        r = check(g, P(txt))                                                                   # must not raise
        assert isinstance(r.ok, bool)
        if r.ok: assert r.reason == ""
        else:
            assert kind(r.reason) in KINDS, r.reason
            seen.add(kind(r.reason))
    assert seen == KINDS, seen                                                                 # all four kinds occur in the random mix
    g = Graph(); g.add_node((1, 1), 5)                                                         # a 3x3 board with a centre cell
    assert check(g, P("ADD[99,99:7]")).ok                                                      # slot far outside the board: admitted
    assert check(g, P("ADD[123456789012345678901234567890,0:1]")).ok                           # and arbitrarily large
    assert check(g, P("ADD[(1,1)->(99,99):adj#1]")).reason.startswith("well-formedness")      # an edge needs both ends to exist, nothing more
    h = Graph(); h.add_node((0, 0), 1); h.add_node((0, 1), 1)
    assert not h.edges                                                                         # neighbours exist only if the task adds an edge


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
