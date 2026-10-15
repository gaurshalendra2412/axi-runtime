"""Engineering support for docs/LAYOUT_TEST.md (concept-map item 98): does the ORDER the same graph is written in change what a stock model finds in it? No GPU, no torch, no real model. The reader,
the cache and the scoring of a reply are `kv_drop.py` (tested in test_kv_drop.py); this file tests what is new in `kv_layout.py`, on the numpy stand-in (`tests/kv_standin.py`):
  1. the quarter turn, and the request builder (`make_task` is `cascade.pick_task` on the identity frame and refuses an invalid request);
  2. the problems: 14 graphs, 42 delete problems on services with 1 to 3 edges, 8 control problems, and the figures the page quotes about them;
  3. the five layouts hold the same facts in the orders the page names (adjacency writes every edge twice, `rot90` turns state and request together, `shuffled` is reproducible);
  4. the scoring of a reply (recall, split by direction, invented, other; a reply cut off at the token limit counts for what it wrote) on scripted replies, and the exact sign test;
  5. a run writes one JSON line per problem, resumes, and refuses another set of problems; scripted replies go through the whole reader;
  6. the thirteen reading rules flip exactly at their thresholds; the page is locked and carries them;
  7. the cell of the notebook that runs this test."""
import hashlib, json, os, random, re, sys, tempfile
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from kv_standin import TinyEngine, byte_kit, byte_table
import axi.experiments.kv_drop as K
import axi.experiments.kv_large as L
import axi.experiments.kv_layout as Y
from axi.engine.cico_parser import CICOParser
from axi.engine.gate import Graph, check
from axi.experiments.agent_loop import SYSTEM, RULES_HINT, same_graph
from axi.experiments.cascade import pick_task

HERE = os.path.dirname(__file__)
PAGE = os.path.join(HERE, "..", "..", "docs", "LAYOUT_TEST.md")
PAGE_SHA256 = "c9c3cd72018428d55338b2f350b3605a4fc50a7091319badbbbe439ed2f142cc"
NB = os.path.join(HERE, "..", "colab", "axi_tensor_tests.ipynb")
RESULT_MARK = "\n\n---\n\n# Result"
KIT, TAB = byte_kit(True), byte_table()
SMALL = dict(seed=5, graphs=2, deps=1, controls=2, nodes=14, edges=22, grid=(5, 5))          # a problem set the numpy stand-in can read in a moment (about 1,500 byte tokens)
TIME_KEYS = {"seconds", "peak_mb"}


def _h(text): return hashlib.sha256((text.rstrip("\n") + "\n").encode()).hexdigest()


def strip(x):
    if isinstance(x, dict): return {k: strip(v) for k, v in x.items() if k not in TIME_KEYS}
    if isinstance(x, list): return [strip(v) for v in x]
    return x


def parse(ideal): return CICOParser.parse_transition_delta(ideal)


def raises(exc, fn, *a, **k):
    try: fn(*a, **k)
    except exc as ex: return ex
    raise AssertionError(f"{getattr(fn, '__name__', fn)} should have raised {exc.__name__}")


def hand_graph() -> Graph:
    """(0,0) has one edge that starts at it and one that ends at it; (2,2)->(1,1) is a real edge that is not (0,0)'s; (3,3) has no edge."""
    g = Graph()
    for n, v in (((0, 0), 1), ((0, 1), 2), ((1, 1), 3), ((2, 2), 4), ((3, 3), 5)): g.add_node(n, v)
    g.add_edge((0, 0), (0, 1), "dep", 3); g.add_edge((1, 1), (0, 0), "owns", 5); g.add_edge((2, 2), (1, 1), "dep", 1)
    return g


# ---------------------------------------------------------------- 1 the quarter turn and the request builder
def test_a_quarter_turn_is_a_bijection_four_turns_are_the_identity_and_a_turned_graph_is_the_same_graph():
    cells = [(r, c) for r in range(10) for c in range(10)]; once = [Y.turn90(x, 10) for x in cells]
    assert sorted(once) == cells and once != cells and Y.turn90((0, 0), 10) == (0, 9) and Y.turn90((0, 9), 10) == (9, 9) and Y.turn90((3, 4), 10) == (4, 6)
    for x in cells: assert Y.turn90(Y.turn90(Y.turn90(Y.turn90(x, 10), 10), 10), 10) == x
    g = L.random_large_graph(random.Random(3), 70, 120, (10, 10)); f = lambda rc: Y.turn90(rc, 10); h = Y.transform_graph(g, f)
    assert len(h.nodes) == 70 and len(h.edges) == 120 and h.is_well_formed() and not same_graph(g, h)
    assert all(h.nodes[f(n)] == v for n, v in g.nodes.items()) and all(h.edges[(f(u), f(v), r)] == w for (u, v, r), w in g.edges.items())
    assert all(len(h.incident(f(n))) == len(g.incident(n)) for n in g.nodes)                                       # the structure is the same, only the names of the places moved
    k = g
    for _ in range(4): k = Y.transform_graph(k, f)
    assert same_graph(k, g)
    assert Y.turn90((2, 5), 6) == (5, 3)                                                                           # the side comes from the grid, not from a constant


def test_make_task_builds_exactly_what_cascade_pick_task_builds_and_refuses_an_invalid_request():
    for seed in range(12):
        g = L.random_large_graph(random.Random(seed), 70, 120, (10, 10))
        t = pick_task(random.Random(seed + 7), g, "delete_dep"); n = (parse(t.ideal).node_deltas[0].r, parse(t.ideal).node_deltas[0].c); m = Y.make_task(g, "delete_dep", node=n)
        assert (m.kind, m.instruction, m.ideal) == (t.kind, t.instruction, t.ideal) and same_graph(m.expected, t.expected) and same_graph(m.graph, g), seed
        t = pick_task(random.Random(seed + 7), g, "add_edge"); e = parse(t.ideal).edge_deltas[0]; m = Y.make_task(g, "add_edge", u=e.u, v=e.v, weight=e.weight)
        assert (m.kind, m.instruction, m.ideal) == (t.kind, t.instruction, t.ideal) and same_graph(m.expected, t.expected), seed
        t = L.pick_large_task(random.Random(seed + 7), g, "add_node", (10, 10)); d = parse(t.ideal).node_deltas[0]; m = Y.make_task(g, "add_node", node=(d.r, d.c), value=d.val)
        assert (m.kind, m.instruction, m.ideal) == (t.kind, t.instruction, t.ideal) and same_graph(m.expected, t.expected), seed
    g = hand_graph(); ok = Y.make_task(g, "delete_dep", node=(0, 0))
    assert ok.ideal == "DEL[0,0] DEL[(0,0)->(0,1):dep#3] DEL[(1,1)->(0,0):owns#5]" and ok.instruction == "Delete the service at (0,0)."
    assert (0, 0) not in ok.expected.nodes and len(ok.expected.edges) == 1 and len(g.edges) == 3                                   # the expected state has the service and its two edges gone; the input is untouched
    for bad in (dict(kind="add_node", node=(0, 0), value=3), dict(kind="delete_dep", node=(9, 9)), dict(kind="add_edge", u=(0, 0), v=(0, 9), weight=1)):         # a service that exists, one that does not, an edge to a service that does not
        kind = bad.pop("kind"); raises(AssertionError, Y.make_task, g, kind, **bad)


# ---------------------------------------------------------------- 2 the problems
def test_the_problems_are_the_ones_the_page_describes():
    P = Y.make_problems(); assert len(P) == 50 and [p["id"] for p in P] == [p["id"] for p in Y.make_problems()] and len({p["id"] for p in P}) == 50
    dep, ctl = [p for p in P if p["kind"] == "delete_dep"], [p for p in P if p["kind"] != "delete_dep"]; assert len(dep) == 42 and len(ctl) == 8
    assert (Y.LAYOUT_NODES, Y.LAYOUT_EDGES, Y.LAYOUT_GRID, Y.LAYOUT_SEED, Y.LAYOUT_GRAPHS, Y.LAYOUT_DEPS, Y.LAYOUT_CONTROLS) == (70, 120, (10, 10), 31, 14, 3, 8)
    assert all(len(p["graph"].nodes) == 70 and len(p["graph"].edges) == 120 and p["side"] == 10 and p["graph"].is_well_formed() for p in P)
    assert all(0 <= r < 10 and 0 <= c < 10 for p in P for r, c in p["graph"].nodes) and sum(1 for n in P[0]["graph"].nodes if not P[0]["graph"].incident(n)) >= 3
    deg, outs, ins = {}, 0, 0
    for p in dep:
        n, g = p["params"]["node"], p["graph"]; inc = g.incident(n); assert 1 <= len(inc) <= 3 and n in g.nodes; deg[len(inc)] = deg.get(len(inc), 0) + 1
        outs += sum(1 for k in inc if k[0] == n); ins += sum(1 for k in inc if k[1] == n)
    assert deg == {1: 5, 2: 15, 3: 22} and (outs, ins) == (53, 48) and abs(sum(k * v for k, v in deg.items()) / 42 - 2.4) < 0.01             # the figures of the page
    for gi in range(14):
        ids = [p["id"] for p in P if p["id"].startswith(f"g{gi}d")]; nodes = [p["params"]["node"] for p in dep if p["id"] in ids]; assert ids == [f"g{gi}d{k}" for k in range(3)] and len(set(nodes)) == 3
        assert len({id(p["graph"]) for p in P if re.fullmatch(rf"g{gi}(d\d|c)", p["id"])}) == 1                                                       # a graph's problems share the graph
    assert [p["id"] for p in ctl] == [f"g{i}c" for i in range(8)] and [p["kind"] for p in ctl] == ["add_edge", "add_node"] * 4
    for p in ctl: t = Y.make_task(p["graph"], p["kind"], **p["params"]); assert check(p["graph"], parse(t.ideal)).ok
    other = Y.make_problems(seed=32); assert not same_graph(other[0]["graph"], P[0]["graph"]) and len(other) == 50
    s = Y.make_problems(**SMALL); assert len(s) == 4 and [p["kind"] for p in s] == ["delete_dep", "add_edge", "delete_dep", "add_node"] and all(p["side"] == 5 and len(p["graph"].nodes) == 14 for p in s)
    assert Y.fingerprint(P) == Y.fingerprint(Y.make_problems()) and Y.fingerprint(P) != Y.fingerprint(other) and Y.fingerprint(s) != Y.fingerprint(P)
    assert Y.fingerprint(P) == "802ec2d3cdb6f584", "the 50 problems of the run are the ones that were drawn when the page was locked (graphs, services, parameters, frame)"
    assert [(p["id"], p["kind"], tuple(sorted(p["params"].items()))) for p in ctl] == [
        ("g0c", "add_edge", (("u", (5, 2)), ("v", (4, 6)), ("weight", 2))), ("g1c", "add_node", (("node", (3, 8)), ("value", 3))), ("g2c", "add_edge", (("u", (7, 4)), ("v", (7, 9)), ("weight", 7))),
        ("g3c", "add_node", (("node", (6, 5)), ("value", 8))), ("g4c", "add_edge", (("u", (5, 9)), ("v", (5, 1)), ("weight", 4))), ("g5c", "add_node", (("node", (0, 9)), ("value", 3))),
        ("g6c", "add_edge", (("u", (9, 2)), ("v", (5, 4)), ("weight", 8))), ("g7c", "add_node", (("node", (9, 8)), ("value", 7)))]
    assert all(1 <= p["params"].get("weight", p["params"].get("value")) <= 9 for p in ctl)


def test_the_fingerprint_notices_every_part_of_a_problem():
    import copy
    P = Y.make_problems(**SMALL); base = Y.fingerprint(P); assert Y.fingerprint(copy.deepcopy(P)) == base
    def edited(fn):
        q = copy.deepcopy(P); fn(q); return Y.fingerprint(q)
    def a_node(q): n = sorted(q[0]["graph"].nodes)[0]; q[0]["graph"].nodes[n] = q[0]["graph"].nodes[n] % 9 + 1
    def an_edge(q): k = sorted(q[0]["graph"].edges)[0]; q[0]["graph"].edges[k] = q[0]["graph"].edges[k] % 9 + 1
    def a_param(q): q[0]["params"] = dict(q[0]["params"], node=(4, 4))
    def a_side(q): q[0]["side"] = 6
    def an_id(q): q[0]["id"] = "zz"
    def a_kind(q): q[0]["kind"] = "add_node"
    def order(q): q.reverse()
    def fewer(q): del q[-1]
    seen = {base}
    for fn in (a_node, an_edge, a_param, a_side, an_id, a_kind, order, fewer):
        f = edited(fn); assert f != base and f not in seen, fn.__name__; seen.add(f)
    q = copy.deepcopy(P); q[1]["params"] = dict(q[1]["params"]); assert Y.fingerprint(q) == base                                                                     # a copy of a part is not a change


def test_the_numbers_the_page_quotes_from_the_design_are_what_the_code_computes():
    head = _head(); P = Y.make_problems(); sysm = SYSTEM + RULES_HINT; size = {}
    for cnd in Y.CONDITIONS:
        sz = []
        for p in P:
            g, task, items, _ = Y.realize(p, cnd); sysx = sysm + (Y.NOTE_ADJ if cnd == "adjacency" else "")
            sz.append(L.crude_tokens(K.first_prefix_text(sysx)) + sum(L.crude_tokens(" " + i.text) for i in items) + L.crude_tokens(K.tail_text(task.instruction)))
        size[cnd] = sum(sz) / len(sz)
    assert round(size["sorted"]) == 2230 == round(size["shuffled"]) == round(size["grouped"]) == round(size["rot90"]) and round(size["adjacency"]) == 3695 and abs(size["adjacency"] / size["sorted"] - 1.65) < 0.01
    for needle in ("2,230 tokens in `sorted` and 3,695 in `adjacency`", "(5 with one edge, 15 with two, 22 with three; 2.4 on average; of their 101 edges, 53 start at the service and 48 end at it)", "About 1.65 times the tokens of `sorted`", "250 reads"):
        assert needle in head, needle
    assert size["sorted"] > Y.Y_MIN_PROMPT and size["adjacency"] * 1.03 < 3877                                                                    # the lines the design speaks to would read yes; adjacency stays under Run C's real prompts


# ---------------------------------------------------------------- 3 the layouts
def _keys(items): return [it.key for it in items]


def test_the_five_layouts_hold_the_same_facts_in_the_orders_the_page_names():
    P = Y.make_problems(**SMALL)
    for p in P:
        g = p["graph"]; base = K.graph_items(g); assert Y.layout_items(g, "sorted") == base and _keys(base)[:len(g.nodes)] == [("n", n) for n in sorted(g.nodes)]
        sh = Y.layout_items(g, "shuffled", p["id"]); assert sorted(_keys(sh)) == sorted(_keys(base)) and sh != base and sh == Y.layout_items(g, "shuffled", p["id"])          # same facts, another order, the same order every time
        assert all(it.key[0] == "n" for it in sh[:len(g.nodes)]) and all(it.key[0] == "e" for it in sh[len(g.nodes):])                                                           # services still before edges
        assert Y.layout_items(g, "shuffled", p["id"] + "x") != sh or len(g.edges) < 3
        assert _keys(sh)[:len(g.nodes)] != [("n", n) for n in sorted(g.nodes)] and _keys(sh)[len(g.nodes):] != [("e", k) for k in sorted(g.edges)], "the services and the edges are both shuffled"
        gr = Y.layout_items(g, "grouped"); assert sorted(_keys(gr)) == sorted(_keys(base)) and len(gr) == len(base)                                                                # every item once
        i = 0
        for n in sorted(g.nodes):
            assert gr[i].key == ("n", n); i += 1
            for k in sorted(k for k in g.edges if k[0] == n): assert gr[i].key == ("e", k) and gr[i].val == g.edges[k]; i += 1
        assert i == len(gr)
        ad = Y.layout_items(g, "adjacency"); assert len(ad) == len(g.nodes) + 2 * len(g.edges); i = 0
        for n in sorted(g.nodes):
            assert ad[i].key == ("n", n); i += 1; mine = [k for k in sorted(g.edges) if k[0] == n] + [k for k in sorted(g.edges) if k[1] == n]
            for k in mine: assert ad[i].key == ("e", k); i += 1
        assert i == len(ad) and all(_keys(ad).count(("e", k)) == 2 for k in g.edges) and all(_keys(ad).count(("n", n)) == 1 for n in g.nodes)                                 # every edge twice, after each of its two services
        assert all(sum(1 for it in ad if it.key == ("e", k)) == 2 for k in g.edges)
        assert str(raises(AssertionError, Y.layout_items, g, "diagonal")) == "diagonal"
    a, b = Y.layout_items(P[0]["graph"], "shuffled", "g0d0"), Y.layout_items(P[0]["graph"], "shuffled", "g0d1"); assert a != b                                                    # the order depends on the problem, so no layout is shared by accident


def test_rot90_turns_the_state_and_the_request_together_and_the_other_layouts_leave_the_coordinates_alone():
    for p in Y.make_problems(**SMALL):
        f = lambda rc: Y.turn90(rc, p["side"]); g0, t0, i0, p0 = Y.realize(p, "sorted"); g1, t1, i1, p1 = Y.realize(p, "rot90"); assert same_graph(g0, p["graph"]) and p0 == p["params"]
        assert same_graph(g1, Y.transform_graph(p["graph"], f)) and not same_graph(g0, g1) and i1 == K.graph_items(g1)                                             # written sorted, in the turned frame
        for key in ("node", "u", "v"):
            if key in p["params"]: assert p1[key] == f(p["params"][key])
        assert same_graph(t1.expected, Y.transform_graph(t0.expected, f)) and same_graph(t1.graph, g1) and t0.instruction != t1.instruction and check(g1, parse(t1.ideal)).ok
        if p["kind"] == "delete_dep": assert len(parse(t1.ideal).edge_deltas) == len(parse(t0.ideal).edge_deltas) and ("(%d,%d)" % p1["node"]) in t1.instruction
        for cnd in ("shuffled", "grouped", "adjacency"):
            g, t, items, pp = Y.realize(p, cnd); assert same_graph(g, p["graph"]) and t.instruction == t0.instruction and t.ideal == t0.ideal and pp == p["params"], cnd


def test_the_adjacency_kit_adds_one_sentence_to_the_system_prompt_and_nothing_else():
    base = KIT; k = Y.kit_for(base, "adjacency"); assert k is not base and k.system == base.system + Y.NOTE_ADJ and k.hint_rules is True and k.eos == base.eos
    assert k.first_prefix == base.enc(K.first_prefix_text(k.system)) and len(k.first_prefix) == len(base.first_prefix) + len(Y.NOTE_ADJ.encode()) and base.system == SYSTEM + RULES_HINT           # the base kit is untouched
    assert k.later_prefix == base.later_prefix and k.nl == base.nl
    for c in ("sorted", "shuffled", "grouped", "rot90"): assert Y.kit_for(base, c) is base, c
    p = Y.make_problems(**SMALL)[0]; g, t, items, _ = Y.realize(p, "adjacency"); assert k.canonical(items, t.instruction) and k.prompt_tokens(items, t.instruction)[:len(k.first_prefix)] == k.first_prefix
    assert "every edge is written twice" in Y.NOTE_ADJ and "write it once" in Y.NOTE_ADJ


# ---------------------------------------------------------------- 4 the scoring of a reply
def test_reply_metrics_count_what_the_page_says_on_scripted_replies():
    g = hand_graph(); node = (0, 0); out_e, in_e, oth = "DEL[(0,0)->(0,1):dep#3]", "DEL[(1,1)->(0,0):owns#5]", "DEL[(2,2)->(1,1):dep#1]"
    M = lambda *items: Y.reply_metrics(" ".join(items), g, node)
    m = M("DEL[0,0]", out_e, in_e); assert (m["recall"], m["recall_out"], m["recall_in"], m["invented"], m["other"], m["own"]) == (1.0, 1.0, 1.0, 0, 0, 2)
    m = M("DEL[0,0]", out_e); assert (m["recall"], m["recall_out"], m["recall_in"], m["invented"]) == (0.5, 1.0, 0.0, 0)                                           # the edge that ends at the service was not found
    m = M("DEL[0,0]", in_e); assert (m["recall"], m["recall_out"], m["recall_in"]) == (0.5, 0.0, 1.0)
    m = M("DEL[0,0]", out_e, "DEL[(1,1)->(0,0):owns#6]"); assert (m["recall"], m["invented"]) == (0.5, 1)                                                          # right edge, wrong weight: not found, and invented
    m = M("DEL[0,0]", out_e, "DEL[(0,1)->(0,0):dep#3]"); assert (m["recall"], m["invented"], m["other"]) == (0.5, 1, 0)                                              # an edge that never existed
    m = M("DEL[0,0]", out_e, in_e, oth); assert (m["recall"], m["invented"], m["other"]) == (1.0, 0, 1)                                                              # real, in the graph, but not the service's
    m = M("DEL[0,0]", out_e, out_e, in_e); assert (m["recall"], m["invented"], m["other"]) == (1.0, 0, 0)                                                            # the same edge twice is found once
    m = M("ADD[(0,0)->(0,1):dep#3]", "ADD[4,4:2]", "DEL[0,0]"); assert (m["recall"], m["invented"]) == (0.0, 0)                                                      # additions and node deletions are not edge deletions
    cut = "DEL[0,0] " + out_e + " DEL[(1,1)->(0,"; m = Y.reply_metrics(cut, g, node); assert (m["recall"], m["invented"]) == (0.5, 0) and K.strict_parse(cut) is None                    # a reply cut at the token limit: the items before the cut count, the whole does not parse
    assert Y.reply_metrics("", g, node)["recall"] == 0.0 and Y.reply_metrics("garbage ((", g, node)["invented"] == 0
    m = Y.reply_metrics("DEL[0,1] " + out_e, g, (0, 1)); assert (m["own"], m["recall"], m["recall_out"], m["recall_in"]) == (1, 1.0, None, 1.0)         # (0,1) has one edge, which ends at it: no recall of edges that start there
    m = Y.reply_metrics("DEL[3,3]", g, (3, 3)); assert m["own"] == 0 and m["recall"] is None and m["recall_out"] is None and m["recall_in"] is None
    assert Y.lenient_deleted_edges("DEL[0,0]\nDEL[(0,0)->(0,1):dep#3]\n  DEL[(10,12)->(3,4):owns#9] ADD[(1,1)->(2,2):dep#1] DEL[(1,1)->(0,0):owns#5") == [((0, 0), (0, 1), "dep", 3), ((10, 12), (3, 4), "owns", 9)]
    t = Y.make_task(g, "delete_dep", node=node); m = Y.reply_metrics(t.ideal, g, node); assert m["recall"] == 1.0 and m["invented"] == 0 and m["other"] == 0                       # the ideal reply of the builder is a perfect one


def test_the_exact_sign_test():
    assert Y.sign_test(0, 0) == 1.0 and Y.sign_test(5, 0) == 0.0625 == Y.sign_test(0, 5) and Y.sign_test(3, 3) == 1.0 and abs(Y.sign_test(8, 2) - 112 / 1024) < 1e-12
    assert abs(Y.sign_test(10, 0) - 2 / 1024) < 1e-12 and Y.sign_test(1, 0) == 1.0 and Y.sign_test(2, 1) == 1.0 and abs(Y.sign_test(9, 1) - 22 / 1024) < 1e-12


def _rec(**o):
    d = dict(exact_blind=False, exact_m6=False, admitted=False, recall=0.0, recall_out=None, recall_in=None, invented=0, other=0, tokens=[1, 2, 3], ended=True, prompt_tokens=100, canonical=True, seconds=1.0, peak_mb=10.0, text="t")
    d.update(o); return d


def test_the_summary_averages_delete_and_control_problems_separately_and_counts_against_sorted():
    rows = [dict(id="g0d0", kind="delete_dep", conds=dict(sorted=_rec(exact_blind=True, exact_m6=True, admitted=True, recall=0.5, recall_out=1.0, recall_in=0.0, invented=2, tokens=[0] * 4, text="a"),
                                                         adjacency=_rec(exact_blind=True, exact_m6=True, admitted=True, recall=1.0, recall_out=1.0, recall_in=1.0, tokens=[0] * 6, text="a", prompt_tokens=170, seconds=3.0, peak_mb=30.0))),
            dict(id="g0d1", kind="delete_dep", conds=dict(sorted=_rec(recall=0.0, recall_out=0.0, recall_in=None, invented=4, ended=False, tokens=[0] * 8, text="b"),
                                                         adjacency=_rec(exact_blind=True, exact_m6=True, admitted=True, recall=1.0, recall_out=1.0, recall_in=None, tokens=[0] * 2, text="c", prompt_tokens=170, seconds=3.0))),
            dict(id="g0d2", kind="delete_dep", conds=dict(sorted=_rec(exact_blind=True, exact_m6=True, admitted=True, recall=1.0, recall_out=None, recall_in=1.0, tokens=[0] * 6, text="d"),
                                                         adjacency=_rec(recall=0.5, recall_out=None, recall_in=0.5, invented=1, ended=False, tokens=[0] * 4, text="d", prompt_tokens=170, seconds=3.0))),
            dict(id="g0c", kind="add_edge", conds=dict(sorted=_rec(exact_blind=True, prompt_tokens=60, seconds=2.0), adjacency=_rec(exact_blind=False, prompt_tokens=100, seconds=4.0, peak_mb=50.0)))]
    s = Y.summarize_layout(rows); so, ad = s["sorted"], s["adjacency"]
    assert (so["n_delete"], so["n_control"]) == (3, 1) and abs(so["recall"] - 0.5) < 1e-12 and abs(so["recall_out"] - 0.5) < 1e-12 and so["recall_in"] == 0.5 and abs(so["invented"] - 2.0) < 1e-12        # None is left out of a mean
    assert abs(so["exact_blind"] - 2 / 3) < 1e-12 and abs(so["decoded"] - 6.0) < 1e-12 and abs(so["ran_on"] - 1 / 3) < 1e-12 and so["control_exact"] == 1.0 and abs(so["prompt_tokens"] - (100 * 3 + 60) / 4) < 1e-12   # control problems are not in the delete means but are in the prompt size
    assert ad["control_exact"] == 0.0 and abs(ad["recall"] - 2.5 / 3) < 1e-12 and abs(ad["ran_on"] - 1 / 3) < 1e-12 and ad["peak_mb"] == 50.0 and abs(ad["seconds"] - 3.25) < 1e-12 and abs(ad["exact_blind"] - 2 / 3) < 1e-12
    assert "exact_blind_vs_sorted" not in so and ad["exact_blind_vs_sorted"] == dict(gain=1, loss=1, p=1.0) and ad["exact_m6_vs_sorted"] == dict(gain=1, loss=1, p=1.0) and abs(ad["same_text_as_sorted"] - 2 / 3) < 1e-12   # a/a, b/c, d/d: the first and the last are the same text
    assert Y.summarize_layout([]) == {}
    only = Y.summarize_layout([dict(id="x", kind="delete_dep", conds=dict(shuffled=_rec(recall=0.5)))]); assert "same_text_as_sorted" not in only["shuffled"]                                              # without a sorted condition there is nothing to compare with
    asym = [dict(id=f"g0d{i}", kind="delete_dep", conds=dict(sorted=_rec(exact_blind=a, exact_m6=a), adjacency=_rec(exact_blind=b, exact_m6=(b if i < 2 else a)))) for i, (a, b) in enumerate([(False, True), (False, True), (False, True), (True, False)])]
    v = Y.summarize_layout(asym)["adjacency"]; assert v["exact_blind_vs_sorted"] == dict(gain=3, loss=1, p=Y.sign_test(3, 1)) and v["exact_m6_vs_sorted"] == dict(gain=2, loss=0, p=Y.sign_test(2, 0)) and Y.sign_test(3, 1) == 0.625


# ---------------------------------------------------------------- 5 the runner
def test_a_run_writes_one_line_per_problem_resumes_after_a_dropped_session_and_refuses_other_problems():
    P = Y.make_problems(**SMALL); d = tempfile.mkdtemp(); path = os.path.join(d, "l.jsonl"); e = TinyEngine(); seen = []
    r = Y.run_layout(e, KIT, TAB, path, P, meta=dict(model="stand-in"), progress=lambda i, n: seen.append((i, n)), max_new=4)
    head = json.loads(open(path).readline())["_meta"]; lines = open(path).read().strip().split("\n")
    assert len(lines) == 1 + len(P) and seen == [(1, 4), (2, 4), (3, 4), (4, 4)] and head["kind"] == "kv_layout" and head["problems"] == 4 and head["conditions"] == list(Y.CONDITIONS) and head["seed"] == 31 and head["max_new"] == 4
    assert head["ids"] == [p["id"] for p in P] and head["fp"] == Y.fingerprint(P) and head["model"] == "stand-in" and head["grid"] == [10, 10] and [x["id"] for x in r["rows"]] == head["ids"]
    for row in r["rows"]:
        assert list(row["conds"]) == list(Y.CONDITIONS) and row["kind"] in ("delete_dep", "add_edge", "add_node")
        for c, rec in row["conds"].items():
            assert rec["canonical"] is True and rec["prompt_tokens"] > 0 and len(rec["tokens"]) <= 4 and {"text", "exact_blind", "exact_m6", "admitted", "parsed", "ended", "seconds", "peak_mb"} <= set(rec)
            assert ("recall" in rec and "invented" in rec) == (row["kind"] == "delete_dep"), (row["id"], c)
        assert row["conds"]["adjacency"]["prompt_tokens"] > row["conds"]["sorted"]["prompt_tokens"] and row["conds"]["sorted"]["prompt_tokens"] == row["conds"]["shuffled"]["prompt_tokens"] == row["conds"]["rot90"]["prompt_tokens"]
    for pr, row in zip(P, r["rows"]):
        for c in Y.CONDITIONS:
            g, t, items, pp = Y.realize(pr, c); assert row["conds"][c]["prompt_tokens"] == len(KIT.prompt_tokens(items, t.instruction)) + (len(Y.NOTE_ADJ.encode()) if c == "adjacency" else 0), (pr["id"], c)    # one token per byte: the sentence is in adjacency's system prompt and in no other
    s = r["summary"]; assert set(s) == set(Y.CONDITIONS) and s["sorted"]["n_delete"] == 2 and s["sorted"]["n_control"] == 2
    calls = e.calls; again = Y.run_layout(e, KIT, TAB, path, P, meta=dict(model="stand-in"), max_new=4); assert e.calls == calls and strip(again["rows"]) == strip(r["rows"])                                  # a finished run is not run again
    keep = lines[:3]; open(path, "w").write("\n".join(keep) + "\n" + lines[3][:40]); e2 = TinyEngine(); again = Y.run_layout(e2, KIT, TAB, path, P, meta=dict(model="stand-in"), max_new=4)           # two problems saved and a half-written third
    assert strip(again["rows"]) == strip(r["rows"]) and 0 < e2.calls < calls
    for bad in (dict(problems=P[:3]), dict(problems=Y.make_problems(**dict(SMALL, seed=6))), dict(problems=P, conditions=("sorted", "adjacency")), dict(problems=P, max_new=5)):
        try: Y.run_layout(TinyEngine(), KIT, TAB, path, bad.get("problems", P), conditions=bad.get("conditions", Y.CONDITIONS), meta=dict(model="stand-in"), max_new=bad.get("max_new", 4)); raise AssertionError(("another run must be refused", bad.keys()))
        except ValueError as ex: assert "different run" in str(ex)
    try: Y.run_layout(TinyEngine(), byte_kit(False), TAB, os.path.join(d, "n.jsonl"), P, max_new=4); raise AssertionError("a prompt without the rule must be refused")
    except AssertionError as ex: assert "states the rule" in str(ex)


def test_scripted_replies_go_through_the_whole_reader_and_are_scored_as_the_page_says():
    p = [q for q in Y.make_problems(**SMALL) if q["kind"] == "delete_dep"][0]; g, task, items, pp = Y.realize(p, "sorted"); node = pp["node"]; inc = sorted(g.incident(node)); assert len(inc) >= 1
    e = TinyEngine(); say = lambda text, end=True: setattr(e, "force", list(text.encode()) + ([256] if end else []))
    say(task.ideal); row = Y.run_problem(e, KIT, TAB, p, ("sorted", "adjacency", "rot90"), max_new=400)
    assert row["conds"]["sorted"]["text"] == task.ideal and row["conds"]["sorted"]["exact_blind"] is True and row["conds"]["sorted"]["recall"] == 1.0 and row["conds"]["sorted"]["invented"] == 0 and row["conds"]["sorted"]["ended"] is True
    assert row["conds"]["adjacency"]["exact_blind"] is True and row["conds"]["adjacency"]["text"] == task.ideal                                                                                         # the same scripted reply, the same graph
    assert row["conds"]["rot90"]["exact_blind"] is False and row["conds"]["rot90"]["recall"] < 1.0                                                                                       # the turned problem has other coordinates: this reply no longer fits
    first = "DEL[%d,%d] " % node + "DEL[(%d,%d)->(%d,%d):%s#%d]" % (inc[0][0] + inc[0][1] + (inc[0][2], g.edges[inc[0]])); say(first)
    row = Y.run_problem(e, KIT, TAB, p, ("sorted",), max_new=400); r = row["conds"]["sorted"]; want = 1 / len(inc)
    assert abs(r["recall"] - want) < 1e-12 and r["exact_blind"] == (len(inc) == 1) and r["invented"] == 0
    wrong = first.replace("#%d]" % g.edges[inc[0]], "#%d]" % (g.edges[inc[0]] % 9 + 1)); say(wrong)
    r = Y.run_problem(e, KIT, TAB, p, ("sorted",), max_new=400)["conds"]["sorted"]; assert r["invented"] == 1 and r["recall"] == 0.0 and r["exact_blind"] is False
    say(task.ideal, end=False); r = Y.run_problem(e, KIT, TAB, p, ("sorted",), max_new=len(task.ideal) - 3)["conds"]["sorted"]                                                            # cut off at the token limit
    assert r["ended"] is False and r["exact_blind"] is False and r["parsed"] is False and abs(r["recall"] - (len(inc) - 1) / len(inc)) < 1e-12 and len(r["tokens"]) == len(task.ideal) - 3
    c = [q for q in Y.make_problems(**SMALL) if q["kind"] == "add_edge"][0]; gc, tc, _, _ = Y.realize(c, "sorted"); say(tc.ideal)
    rc = Y.run_problem(e, KIT, TAB, c, ("sorted", "adjacency"), max_new=200)["conds"]; assert rc["sorted"]["exact_blind"] and rc["adjacency"]["exact_blind"] and "recall" not in rc["sorted"]                  # a control problem: scored, with no recall


# ---------------------------------------------------------------- 6 the reading rules, the page
def _S(**over):
    full = dict(n_delete=42, n_control=8, prompt_tokens=2230.0, canonical=1.0, control_exact=1.0)
    base = dict(sorted=dict(full, recall=0.40, recall_out=0.60, recall_in=0.20, exact_blind=0.10, invented=2.0), shuffled=dict(full, recall=0.25, same_text_as_sorted=0.5), grouped=dict(full, recall=0.46),
                adjacency=dict(full, recall=0.60, exact_blind=0.30, invented=1.2), rot90=dict(full, recall=0.44))
    for c, o in over.items(): base[c] = dict(base[c], **o)
    return base


def _failing(**over): return [i + 1 for i, (_, ok) in enumerate(Y.verdict_layout(_S(**over))) if ok is not True]


def test_the_thirteen_reading_rules_flip_exactly_at_their_thresholds():
    base = Y.verdict_layout(_S()); assert len(base) == len(Y.LAYOUT_LINES) == 13 and all(ok is True for _, ok in base) and [t for t, _ in base] == [t for _, t in Y.LAYOUT_LINES]
    assert [tag for tag, _ in Y.LAYOUT_LINES].count("instrument") == 3 and [tag for tag, _ in Y.LAYOUT_LINES].count("guard") == 1 and [tag for tag, _ in Y.LAYOUT_LINES].count("claim") == 9
    cases = [(dict(sorted=dict(prompt_tokens=1999.9)), [1]), (dict(sorted=dict(prompt_tokens=2000.0)), []), (dict(adjacency=dict(canonical=0.999)), [2]), (dict(rot90=dict(canonical=0.0)), [2]),
             (dict(shuffled=dict(n_delete=41)), [3]), (dict(grouped=dict(n_control=7)), [3]), (dict(adjacency=dict(n_delete=43)), [3]),
             (dict(sorted=dict(recall=0.1499), shuffled=dict(recall=0.04), rot90=dict(recall=0.2)), [4]), (dict(sorted=dict(recall=0.15), shuffled=dict(recall=0.05), rot90=dict(recall=0.2)), []),
             (dict(sorted=dict(recall=0.90), shuffled=dict(recall=0.5), grouped=dict(recall=1.0), rot90=dict(recall=0.85), adjacency=dict(recall=0.95)), [7]),          # no room above: a gain of 0.15 is not possible
             (dict(sorted=dict(recall=0.9001), shuffled=dict(recall=0.5), grouped=dict(recall=1.0), rot90=dict(recall=0.85), adjacency=dict(recall=0.95)), [4, 7]),
             (dict(shuffled=dict(recall=0.30)), []), (dict(shuffled=dict(recall=0.3001)), [5]), (dict(sorted=dict(recall_out=0.30)), []), (dict(sorted=dict(recall_out=0.2999)), [6]),
             (dict(adjacency=dict(recall=0.55)), []), (dict(adjacency=dict(recall=0.5499)), [7]), (dict(adjacency=dict(exact_blind=0.25)), []), (dict(adjacency=dict(exact_blind=0.2499)), [8]),
             (dict(grouped=dict(recall=0.45)), []), (dict(grouped=dict(recall=0.4499)), [9]), (dict(rot90=dict(recall=0.50)), []), (dict(rot90=dict(recall=0.5001)), [10]), (dict(rot90=dict(recall=0.30)), []), (dict(rot90=dict(recall=0.2999)), [10]),
             (dict(adjacency=dict(invented=1.4)), []), (dict(adjacency=dict(invented=1.4001)), [11]), (dict(rot90=dict(control_exact=0.85)), []), (dict(rot90=dict(control_exact=0.8499)), [12]),
             (dict(sorted=dict(control_exact=0.8)), [12]), (dict(shuffled=dict(same_text_as_sorted=0.80)), []), (dict(shuffled=dict(same_text_as_sorted=0.8001)), [13])]
    for o, want in cases: assert _failing(**o) == want, (o, _failing(**o), want)
    assert Y.verdict_layout(_S(sorted=dict(invented=0.0)))[10][1] is None                                                                                      # nothing to reduce: the line cannot be read
    assert Y.verdict_layout({}) == [(t, None) for _, t in Y.LAYOUT_LINES]
    part = Y.verdict_layout({"sorted": _S()["sorted"]}); assert [ok for _, ok in part] == [True, True, True, True, None, True, None, None, None, None, None, True, None]            # only sorted was run: the lines about the others read 'not run'
    assert Y.LAYOUT_HEADLINE == (5, 7, 8)


def _head():
    text = open(PAGE).read(); assert text.count(RESULT_MARK) <= 1
    return text[:text.index(RESULT_MARK)] if RESULT_MARK in text else text


def test_the_page_is_locked_and_carries_the_thirteen_lines_and_the_settings_the_code_uses():
    head = _head(); assert _h(head) == PAGE_SHA256, "the predictions and reading rules must not change after the run; later notes go below them"
    for i, (tag, line) in enumerate(Y.LAYOUT_LINES, 1): assert f"\n{i}. [{tag}] {line} (" in head, i
    probs = [int(m.group(1)) for m in re.finditer(r"\n\d+\. \[\w+\] .*\((\d+) %\)(?=\n)", head)]; assert probs == [85, 85, 99, 85, 65, 92, 75, 55, 35, 70, 60, 80, 95]
    p = {i + 1: v for i, v in enumerate(probs)}; prod = 100 * np.prod([p[i] / 100 for i in Y.LAYOUT_HEADLINE]); assert Y.LAYOUT_HEADLINE == (5, 7, 8) and abs(prod - 27) <= 0.5 and "about 27 %" in head and "about **35 %**" in head
    assert "**Headline** (lines 5, 7 and 8)" in head and "Concept-map item 98" in head and "item 99" in head
    for needle in ("70 services", "10 x 10 grid", "120 distinct edges", "14 graphs, 3 services each", "42 delete problems", "8 control problems", "250 reads", "Seed 31", "at most 120 new tokens", "Qwen2.5-7B-Instruct in 4-bit (NF4)", "**Cost: $0.**",
                   "`sorted`", "`shuffled`", "`grouped`", "`adjacency`", "`rot90`", "`kv_layout.reply_metrics`", "`kv_layout.verdict_layout`", "`kv_drop.score_reply`", "`kv_large.random_large_graph`",
                   "axi-colab-tensor.zip", "axi_tensor_tests.ipynb", "test_kv_layout.py", "Cell 8", "KV_BLOCK_TEST.md", "Lost in the Middle", "Premise Order Matters", "Talk like a Graph", "None of those remedies is tested here",
                   "I have not checked any of them with him and do not attribute them to him", "4 (8 %)", "**0.23**", "**0.42**", "**0.04**", "2.85", "14 of 52 (27 %)", "not blind to it", "I did not add a \"sorted with the sentence\" condition",
                   "`rot90` is one of eight frames", "Run A and Run C each got every add reply right (89 of 89 in each)"):
        assert needle in head, needle
    assert (Y.Y_MIN_PROMPT, Y.Y_ROOM_LO, Y.Y_ROOM_HI, Y.Y_ORDER, Y.Y_CONTIG, Y.Y_GROUPED, Y.Y_ADJ_RECALL, Y.Y_ADJ_EXACT, Y.Y_ROT, Y.Y_CONTROL, Y.Y_INVENT, Y.Y_DIFFER) == (2000, 0.15, 0.90, 0.10, 0.10, 0.05, 0.15, 0.15, 0.10, 0.85, 0.30, 0.20)
    lines = "\n".join(l for _, l in Y.LAYOUT_LINES)
    for needle in ("at least 2,000 tokens", "between 0.15 and 0.90", "at least 0.10 below", "at least 0.10 above", "at least 0.15 above", "at least 15 points above", "at least 0.05 above", "within 0.10", "at least 30 % fewer", "at least 85 %", "at least 20 %"):
        assert needle in lines, needle
    assert Y.CONDITIONS == ("sorted", "shuffled", "grouped", "adjacency", "rot90") and K.KV_MAX_NEW == 120 and Y.NOTE_ADJ.strip() in head


# ---------------------------------------------------------------- 7 the notebook
def test_the_notebook_cell_of_this_test_runs_the_checks_first_states_the_rule_and_keeps_no_output():
    nb = json.load(open(NB)); cells = nb["cells"]; allsrc = "\n".join("".join(c["source"]) for c in cells); code = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]; joined = "\n".join(code)
    assert all(c["cell_type"] != "code" or (c.get("outputs") == [] and c.get("execution_count") is None) for c in cells) and not re.search(r"AIza[0-9A-Za-z_\-]{10,}|hf_[A-Za-z0-9]{20,}", allsrc) and "userdata" not in joined
    run = [c for c in code if "run_layout(" in c and "files.download" in c]; assert len(run) == 1 and len([c for c in code if "run_layout(" in c]) == 2; cell = run[0]            # the smoke run and the run
    assert "kit_rule" in cell and "verdict_layout(" in cell and "print_layout(" in cell and "axi_layout" in cell and "files.download" in cell and "store.restore" in cell and "store.progress" in cell
    assert joined.index("preflight(") < joined.index("run_layout(") and joined.index("preflight_blocks(") < joined.index("run_layout(") and "make_hf_kit(tok, hint_rules=True)" in joined
    assert 'kit_for(kit_rule, "adjacency").canonical(items_, t_.instruction)' in joined and "the order test's adjacency prompt (one more sentence in the system prompt) is the tokenizer's own tokens" in joined
    assert "docs/LAYOUT_TEST.md" in allsrc and "axi-colab-tensor.zip" in allsrc and "T4" in allsrc and "Do not paste" in allsrc and "$0" in allsrc
    src = open(Y.__file__).read(); assert "torch" not in src and "import transformers" not in src


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn): fn(); print("PASS", name); n += 1
    print(n, "tests passed")
