"""Engineering support for docs/SWARM_TEST.md (concept-map item 100): can many agents edit one shared state at the same time? CPU only, no model. This file tests the INSTRUMENT, not the finding:
  1. the footprint (L, K, C, D) of each kind of edit, and the independence rule on twelve hand-made pairs, each against the truth (both orders through the real gate): the rule, the pager detector and the
     naive detector each give the verdict the page says; a broken footprint is caught by the same table;
  2. the pager detector runs the real `Pager` and makes the same page calls as `Runtime.propose`; the verdict does not depend on which of the two commits first;
  3. the generator: every edit passes the gate, all kinds appear, the same seed gives the same edits, a hot spot makes edits overlap more; `blind_apply` is `gate.apply` when the edit is admissible;
  4. a small run of the pair experiment: the tables add up; deliberately broken rules are noticed (never clash, clash on any shared context, names only);
  5. the three schedulers on a hand-made round of three agents; merge chooses by agent number, not by arrival; a rule that passes everything makes the gate refuse (the counter has teeth);
  6. the stores: clash, stale snapshot, gate-only, last-writer (loses an update) and a store with no gate (ill-formed state) are told apart by the replay; real threads keep the right store correct;
  7. the summary and the nineteen reading rules, each flipping exactly at its threshold; the quick run, the printout and the command line;
  8. the page is locked (hash of the text above the Result), carries the nineteen lines with the probabilities, the settings and the thresholds the code uses."""
import contextlib, copy, hashlib, io, itertools, json, os, random, re, sys, tempfile, types
sys.path.insert(0, os.path.dirname(__file__))
import axi.engine.pager as PG
import axi.experiments.swarm as W
from axi.engine.cico_parser import CICOParser
from axi.engine.gate import Graph, apply, check
from axi.engine.inverse import to_text
from axi.engine.runtime import Runtime, vpn_of
from axi.experiments.agent_loop import copy_graph, same_graph

HERE = os.path.dirname(__file__)
PAGE = os.path.join(HERE, "..", "..", "docs", "SWARM_TEST.md")
PAGE_SHA256 = "e18d1a644f9b3394e8845df20f40c28a2cf86a4c6eeb7a7d94771a87656b2a04"
RESULT_MARK = "\n\n---\n\n# Result"
P = CICOParser.parse_transition_delta


def _h(text): return hashlib.sha256((text.rstrip("\n") + "\n").encode()).hexdigest()


def base_graph():
    """Seven services, five edges: (0,0)->(0,1)->(0,2) a chain, (1,0)->(1,1)->(1,2) another, (0,0)->(1,0), and a service (2,0) with no edge."""
    g = Graph()
    for k, n in enumerate([(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0)], 1): g.add_node(n, k)
    for u, v, rel, w in [((0, 0), (0, 1), "dep", 3), ((0, 1), (0, 2), "dep", 4), ((1, 0), (1, 1), "owns", 5), ((1, 1), (1, 2), "dep", 6), ((0, 0), (1, 0), "dep", 2)]: g.add_edge(u, v, rel, w)
    return g


# name: (a, b, commute, dpo clash, pager clash, names clash)
CASES = {
    "hub_adds": ("ADD[(0,0)->(1,2):dep#1]", "ADD[(0,0)->(0,2):owns#2]", True, False, True, False),
    "add_edge_vs_delete_service": ("DEL[1,2] DEL[(1,1)->(1,2):dep#6]", "ADD[(0,0)->(1,2):dep#9]", False, True, True, False),
    "double_delete_edge": ("DEL[(0,0)->(0,1):dep#3]", "DEL[(0,0)->(0,1):dep#3]", False, True, False, True),
    "delete_edge_vs_delete_its_service": ("DEL[(0,1)->(0,2):dep#4]", "DEL[0,2] DEL[(0,1)->(0,2):dep#4]", False, True, False, True),
    "add_same_service": ("ADD[2,2:5]", "ADD[2,2:9]", False, True, True, True),
    "add_same_edge_other_weight": ("ADD[(0,0)->(1,2):dep#1]", "ADD[(0,0)->(1,2):dep#2]", False, True, True, True),
    "disjoint_deletes": ("DEL[2,0]", "DEL[(1,0)->(1,1):owns#5]", True, False, False, False),
    "adjacent_service_deletes": ("DEL[0,0] DEL[(0,0)->(0,1):dep#3] DEL[(0,0)->(1,0):dep#2]", "DEL[0,1] DEL[(0,0)->(0,1):dep#3] DEL[(0,1)->(0,2):dep#4]", False, True, False, True),
    "wire_vs_delete_target": ("ADD[3,3:1] ADD[(3,3)->(0,2):dep#1]", "DEL[0,2] DEL[(0,1)->(0,2):dep#4]", False, True, True, False),
    "wire_vs_wire_same_target": ("ADD[3,3:1] ADD[(3,3)->(0,2):dep#1]", "ADD[3,4:1] ADD[(3,4)->(0,2):dep#1]", True, False, True, False),
    "rewire_and_add": ("DEL[(1,1)->(1,2):dep#6] ADD[(1,1)->(0,2):dep#1]", "ADD[(2,0)->(1,2):owns#1]", True, False, False, False),
    "same_edit_twice": ("DEL[2,0]", "DEL[2,0]", False, True, True, True),
}


def verdicts(g, a, b, fp=W.footprint, rule=W.independent):
    fa, fb = fp(a), fp(b); tmpl = W.pager_template(g)
    return dict(commute=W.commutes(g, a, b)["commute"], dpo=not rule(fa, fb), pager=W.pager_clash(tmpl, a, b)[1], names=W.names_clash(fa, fb))


def exactness_report(fp=W.footprint, rule=W.independent):
    """The cases on which the rule's verdict is not the truth (a clash exactly when the two edits do not commute)."""
    g = base_graph(); out = []
    for name, (ta, tb, *_) in CASES.items():
        v = verdicts(g, P(ta), P(tb), fp, rule)
        if v["dpo"] != (not v["commute"]): out.append(name)
    return out


# ---------------------------------------------------------------- 1 footprints and the rule on hand-made pairs
def test_the_footprint_of_each_kind_of_edit():
    n = lambda *x: ("n", tuple(x)); e = lambda u, v, r: ("e", (u, v, r))
    f = W.footprint(P("ADD[2,2:5]")); assert (f.L, f.K, f.C, f.D) == (frozenset(), frozenset(), frozenset({n(2, 2)}), frozenset())
    f = W.footprint(P("ADD[(0,0)->(1,2):dep#1]")); assert f.L == f.K == {n(0, 0), n(1, 2)} and f.C == {e((0, 0), (1, 2), "dep")} and f.D == frozenset()
    f = W.footprint(P("ADD[3,3:1] ADD[(3,3)->(0,2):dep#1]")); assert f.L == f.K == {n(0, 2)} and f.C == {n(3, 3), e((3, 3), (0, 2), "dep")}          # the end the same edit adds is not needed from outside
    f = W.footprint(P("DEL[(0,1)->(0,2):dep#4]")); assert f.L == {n(0, 1), n(0, 2), e((0, 1), (0, 2), "dep")} and f.K == {n(0, 1), n(0, 2)} and f.D == {e((0, 1), (0, 2), "dep")} and f.C == frozenset()
    f = W.footprint(P("DEL[0,2] DEL[(0,1)->(0,2):dep#4]")); assert f.D == {n(0, 2), e((0, 1), (0, 2), "dep")} and f.K == {n(0, 1)} and f.L == {n(0, 1), n(0, 2), e((0, 1), (0, 2), "dep")}
    f = W.footprint(P("DEL[2,0]")); assert f.L == f.D == {n(2, 0)} and f.K == frozenset()
    f = W.footprint(P("DEL[(1,1)->(1,2):dep#6] ADD[(1,1)->(0,2):dep#1]")); assert f.C == {e((1, 1), (0, 2), "dep")} and f.D == {e((1, 1), (1, 2), "dep")} and f.K == {n(1, 1), n(1, 2), n(0, 2)}
    for x in (f,): assert all(isinstance(s, frozenset) for s in x)


def test_the_rule_the_pager_and_the_names_give_the_verdicts_the_page_says_on_twelve_pairs():
    g = base_graph()
    for name, (ta, tb, commute, dpo, pager, names) in CASES.items():
        a, b = P(ta), P(tb); assert check(g, a).ok and check(g, b).ok, name
        v = verdicts(g, a, b); assert (v["commute"], v["dpo"], v["pager"], v["names"]) == (commute, dpo, pager, names), (name, v)
        w = verdicts(g, b, a); assert w == v, (name, "the verdict depends on which edit comes first")
    assert sum(1 for c in CASES.values() if c[3] != (not c[2])) == 0                                    # on all twelve the rule is exactly the truth
    assert sum(1 for c in CASES.values() if c[4] and c[2]) >= 2 and sum(1 for c in CASES.values() if not c[4] and not c[2]) >= 3        # the pager: false alarms and misses both occur on the table
    assert sum(1 for c in CASES.values() if not c[5] and not c[2]) >= 2                                  # names: misses occur
    a, b = W.footprint(P(CASES["hub_adds"][0])), W.footprint(P(CASES["hub_adds"][1])); assert W.independent(a, b) and W.independent(b, a)


def test_a_broken_footprint_or_rule_is_caught_by_the_same_table():
    assert exactness_report() == []
    def no_ends(delta):                                    # forgets that an added edge needs its ends
        f = W.footprint(delta); ends = {("n", x) for e in delta.edge_deltas if e.op == "ADD" for x in (e.u, e.v)}
        return W.Foot(f.L - ends, f.K - ends, f.C, f.D)
    assert "add_edge_vs_delete_service" in exactness_report(fp=no_ends) and "wire_vs_delete_target" in exactness_report(fp=no_ends)
    def no_create(delta):                                  # forgets what an edit creates
        f = W.footprint(delta); return W.Foot(f.L, f.K, frozenset(), f.D)
    assert {"add_same_service", "add_same_edge_other_weight"} <= set(exactness_report(fp=no_create))
    assert "hub_adds" in exactness_report(rule=lambda a, b: not (a.L & b.L))                              # every shared context a clash: false alarms
    assert len(exactness_report(rule=lambda a, b: True)) == 8 and set(exactness_report(rule=lambda a, b: True)) == {n for n, c in CASES.items() if not c[2]}   # never a clash: misses every real one
    assert set(exactness_report(rule=lambda a, b: not W.names_clash(a, b))) == {"add_edge_vs_delete_service", "wire_vs_delete_target"}


def test_the_rule_on_footprints_alone_and_the_truth_when_only_one_order_works_or_the_orders_differ():
    n = lambda *x: ("n", tuple(x)); F = lambda L=(), K=(), C=(), D=(): W.Foot(frozenset(L), frozenset(K), frozenset(C), frozenset(D)); x = n(5, 5); y = n(1, 1)
    makes, uses = F(C={x}), F(L={x}, K={x})
    assert not W.independent(makes, uses) and not W.independent(uses, makes)                      # one makes what the other needs (cannot happen on one snapshot; the rule says so all the same)
    assert not W.independent(makes, F(C={x})) and W.independent(makes, F(C={y})) and W.independent(uses, F(L={x}, K={x})) and W.independent(uses, F(L={y}, K={y}))
    assert not W.independent(uses, F(L={x}, D={x})) and not W.independent(F(L={x}, D={x}), uses) and not W.independent(F(L={x}, D={x}), F(L={x}, D={x}))
    assert W.names_clash(F(C={x}), F(D={x})) and not W.names_clash(F(L={x}, K={x}), F(L={x}, K={x})) and W.names_clash(F(D={x}), F(C={x}))
    g = base_graph(); a, b = P("ADD[3,3:1]"), P("ADD[(3,3)->(0,0):dep#1]")                      # b works only after a
    t = W.commutes(g, a, b); assert (t["ab"], t["ba"], t["same"], t["commute"]) == (True, False, False, False)
    t = W.commutes(g, b, a); assert (t["ab"], t["ba"], t["same"], t["commute"]) == (False, True, False, False)
    orig = (W.check, W.apply)
    try:                                                                                          # a gate under which the second write wins: both orders are accepted and the graphs differ
        W.check = lambda h, d: types.SimpleNamespace(ok=True)
        def overwrite(h, d):
            for z in d.node_deltas: h.nodes[(z.r, z.c)] = z.val
        W.apply = overwrite
        t = W.commutes(g, P("ADD[2,2:5]"), P("ADD[2,2:6]")); assert (t["ab"], t["ba"], t["same"], t["commute"]) == (True, True, False, False)
        t = W.commutes(g, P("ADD[2,2:5]"), P("ADD[2,2:5]")); assert (t["ab"], t["ba"], t["same"], t["commute"]) == (True, True, True, True)
    finally: W.check, W.apply = orig
    g1, g2, g3 = base_graph(), base_graph(), base_graph(); g2.nodes[(0, 0)] = 99; g3.edges[((0, 0), (0, 1), "dep")] = 99
    assert len({W.graph_hash(g1), W.graph_hash(g2), W.graph_hash(g3)}) == 3 and W.graph_hash(g1) == W.graph_hash(base_graph()) and not W.well_formed(_ill()) and W.well_formed(g1)


def _ill():
    g = base_graph(); del g.nodes[(0, 2)]; return g


# ---------------------------------------------------------------- 2 the pager detector
def test_the_pager_detector_makes_the_page_calls_runtime_makes_and_a_conflict_is_the_pagers():
    g = base_graph(); calls = []
    orig = (PG.Transaction.write_page, PG.Transaction.unmap_page, PG.Transaction.commit)
    def w(self, vpn): calls.append(("w", vpn)); return orig[0](self, vpn)
    def u(self, vpn): calls.append(("u", vpn)); return orig[1](self, vpn)
    try:
        PG.Transaction.write_page, PG.Transaction.unmap_page = w, u
        for name, (ta, tb, *_) in CASES.items():
            for t in (ta, tb):
                calls.clear(); p = PG.Pager(64); tx = W._pager_tx(p, P(t)); mine = list(calls); tx.rollback()
                calls.clear(); rt = Runtime(64); rt.graph = copy_graph(g); out = rt.propose(t); real = list(calls)
                assert out.committed, (name, t, out); assert mine == real, (name, t, mine, real)
    finally:
        PG.Transaction.write_page, PG.Transaction.unmap_page = orig[0], orig[1]
    raised = []; real_commit = PG.Transaction.commit
    def spy(self):
        try: return real_commit(self)
        except PG.Conflict as e: raised.append(e); raise
    try:
        PG.Transaction.commit = spy; tm = W.pager_template(g)
        assert W.pager_clash(tm, P(CASES["hub_adds"][0]), P(CASES["hub_adds"][1])) == (True, True) and len(raised) == 1 and isinstance(raised[0], PG.Conflict)
        assert W.pager_clash(tm, P(CASES["double_delete_edge"][0]), P(CASES["double_delete_edge"][1])) == (True, False) and len(raised) == 1
    finally:
        PG.Transaction.commit = real_commit
    tm = W.pager_template(g); assert len(tm._table) == 7 and tm.free_blocks() == 200 and tm._live == []
    W.pager_clash(tm, P("ADD[2,2:5]"), P("ADD[2,2:6]")); assert len(tm._table) == 7 and tm.free_blocks() == 200                          # the template is not changed by a detection


# ---------------------------------------------------------------- 3 the generator
def test_the_generator_makes_valid_varied_repeatable_edits_and_a_hot_spot_crowds_them():
    rng = random.Random(5); g = W.random_large_graph(rng, 40, 60, (10, 10)); hot, hot_free = W.hot_items(g, (10, 10))
    assert len(hot) == 3 and len(hot_free) == 3 and all(len(g.incident(n)) >= len(g.incident(m)) for n in hot for m in g.nodes if m not in hot) and all(c not in g.nodes for c in hot_free)
    assert W.hot_items(base_graph(), (10, 10)) == ([(0, 0), (0, 1), (1, 0)], [(0, 3), (0, 4), (0, 5)])                  # four edges tie at two: by coordinates; the free cells skip the seven that are taken
    shapes = set(); texts = []
    for _ in range(1500):
        d = W.random_delta(rng, g, (10, 10), hot, hot_free, 0.5); assert check(g, d).ok; shapes.add(W.shape(d)); texts.append(to_text(d))
    assert {"+n", "+e", "-e", "-n", "-e-n", "+e+n", "+e-e"} <= shapes, shapes
    for _ in range(1500):                                                                                            # what each kind promises
        d = W.random_delta(rng, g, (10, 10), hot, hot_free, 0.5); s = W.shape(d)
        if s == "-e-n": assert len(d.node_deltas) == 1 and 1 <= len(d.edge_deltas) <= 4 and set(g.incident((d.node_deltas[0].r, d.node_deltas[0].c))) == {(e.u, e.v, e.relation) for e in d.edge_deltas}
        if s == "-n": assert not g.incident((d.node_deltas[0].r, d.node_deltas[0].c)) and len(d.node_deltas) == 1
        if s == "+n": assert len(d.node_deltas) == 1 and d.node_deltas[0].val in range(1, 10)
        if s == "+e": assert len(d.edge_deltas) == 1 and d.edge_deltas[0].relation in ("dep", "owns") and d.edge_deltas[0].u != d.edge_deltas[0].v and 1 <= d.edge_deltas[0].weight <= 9
        if s == "-e": assert len(d.edge_deltas) == 1 and g.edges[(d.edge_deltas[0].u, d.edge_deltas[0].v, d.edge_deltas[0].relation)] == d.edge_deltas[0].weight
    rng2 = random.Random(5); g2 = W.random_large_graph(rng2, 40, 60, (10, 10)); h2 = W.hot_items(g2, (10, 10))
    assert [to_text(W.random_delta(rng2, g2, (10, 10), *h2, 0.5)) for _ in range(1500)] == texts
    def overlap(p_hot):
        r = random.Random(9); n = hit = 0
        for _ in range(3000):
            a, b = (W.footprint(W.random_delta(r, g, (10, 10), hot, hot_free, p_hot)) for _ in range(2)); n += 1; hit += bool((a.L | a.C) & (b.L | b.C))
        return hit / n
    assert overlap(0.5) > overlap(0.0) * 1.5
    assert hot == sorted(g.nodes, key=lambda n: (-len(g.incident(n)), n))[:3] and hot_free == [(r, c) for r in range(10) for c in range(10) if (r, c) not in g.nodes][:3]
    r = random.Random(1); pool, hs = list(range(40)), [1, 2, 3]; share = lambda p: sum(W._pick(r, pool, hs, p) in hs for _ in range(4000)) / 4000
    assert share(0.0) < 0.15 and 0.45 < share(0.5) < 0.65 and share(1.0) == 1.0 and W._pick(r, [], hs, 0.5) is None and W._pick(r, [7, 8], hs, 1.0) in (7, 8)
    orig, seq, calls = W._make, iter([P("DEL[9,9]")] * 3 + [P("ADD[2,2:5]")]), []
    try:                                                                                          # an edit the gate refuses is drawn again
        W._make = lambda *a, **k: (calls.append(1), next(seq))[1]; d = W.random_delta(random.Random(1), base_graph(), (10, 10)); assert to_text(d) == "ADD[2,2:5]" and len(calls) == 4
    finally: W._make = orig
    h = base_graph(); W.blind_apply(h, P("ADD[0,0:9] ADD[(0,0)->(0,1):dep#9]")); assert h.nodes[(0, 0)] == 9 and h.edges[((0, 0), (0, 1), "dep")] == 9         # blind: the write goes over what is there
    assert sum(w for _, w in W.KIND_WEIGHTS) == 100 and [k for k, _ in W.KIND_WEIGHTS] == ["add_node", "add_edge", "del_edge", "del_leaf", "del_dep", "multi"]
    try: W.random_delta(random.Random(1), Graph(), (0, 0)); raise AssertionError("an empty grid has no edit")
    except RuntimeError: pass
    for t in ("DEL[2,0]", "DEL[(0,0)->(0,1):dep#3]", "ADD[(0,0)->(1,2):dep#1]", "ADD[2,2:5]", "DEL[0,2] DEL[(0,1)->(0,2):dep#4]", "ADD[3,3:1] ADD[(3,3)->(0,2):dep#1]"):
        h1, h2_ = copy_graph(base_graph()), copy_graph(base_graph()); d = P(t); apply(h1, d); W.blind_apply(h2_, d); assert same_graph(h1, h2_), t
    assert W.shape(P("DEL[0,2] DEL[(0,1)->(0,2):dep#4]")) == "-e-n" and W.shape(P("ADD[2,2:5]")) == "+n"


# ---------------------------------------------------------------- 4 the pair experiment
def test_a_small_pair_run_adds_up_and_broken_rules_are_noticed():
    r = W.run_pairs(seed=3, graphs=3, per_graph=700)
    for name, d in r.items():
        assert d["pairs"] == 2100 and d["conflict"] + d["commute"] == d["pairs"] and d["both_valid"] == d["pairs"] and d["first_commit_ok"] == d["pairs"]
        assert sum(d["why"].values()) == d["conflict"]
        for det, c in d["detectors"].items():
            assert c["tp"] + c["fn"] == d["conflict"] and c["fp"] + c["tn"] == d["commute"], (name, det)
        assert d["detectors"]["names"]["fn"] > 0 and d["detectors"]["pager"]["fn"] > 0                                  # the structure of the table, not a finding: these two kinds exist
        for det in W.DETECTORS: assert all(k.startswith(("fn ", "fp ")) for k in d["misses"][det]) and len(d["examples"][det]) <= 3
    assert r["hot"]["p_hot"] == 0.5 and r["uniform"]["p_hot"] == 0.0 and json.dumps(r)
    never = W.run_pairs(seed=3, graphs=3, per_graph=700, rule=lambda a, b: True)
    for d in never.values(): assert d["detectors"]["dpo"]["tp"] == 0 and d["detectors"]["dpo"]["fn"] == d["conflict"] > 0
    anyctx = W.run_pairs(seed=3, graphs=3, per_graph=700, rule=lambda a, b: not (a.L & b.L))
    assert all(d["detectors"]["dpo"]["fp"] > r[k]["detectors"]["dpo"]["fp"] for k, d in anyctx.items())
    namesonly = W.run_pairs(seed=3, graphs=3, per_graph=700, rule=lambda a, b: not W.names_clash(a, b))
    for k, d in namesonly.items(): assert d["detectors"]["dpo"] == r[k]["detectors"]["names"]
    assert W.run_pairs(seed=3, graphs=3, per_graph=700) == r                                                            # repeatable


# ---------------------------------------------------------------- 5 the schedulers
def _round():
    return base_graph(), [P("DEL[1,2] DEL[(1,1)->(1,2):dep#6]"), P("ADD[(0,0)->(1,2):dep#9]"), P("ADD[2,2:5]")]


def test_the_three_schedulers_on_a_round_of_three_agents():
    import itertools
    g, ds = _round(); orders = [list(p) for p in itertools.permutations(range(3))]
    for arr in orders:
        h, ch, _ = W.schedule_blind(g, ds, arr); assert not W.well_formed(h) and ch == arr                                  # the edge to the deleted service stays
    gated = {tuple(arr): W.schedule_gated(g, ds, arr) for arr in orders}
    assert all(W.well_formed(h) for h, _, _ in gated.values()) and len({W.graph_hash(h) for h, _, _ in gated.values()}) == 2
    assert gated[(0, 1, 2)][1] == [0, 2] and gated[(1, 0, 2)][1] == [1, 2]                                              # whoever comes first wins
    ends = set()
    for arr in orders:
        h, ch, refused = W.schedule_merge(g, ds, arr); ends.add(W.graph_hash(h)); assert ch == [0, 2] and refused == 0 and W.well_formed(h)
    assert len(ends) == 1
    swapped = [ds[1], ds[0], ds[2]]
    for arr in orders:
        h, ch, refused = W.schedule_merge(g, swapped, arr); assert ch == [0, 2] and ((0, 0), (1, 2), "dep") in h.edges and (1, 2) in h.nodes and refused == 0        # agent 0 (the add) wins, by number
    for arr in orders:
        h, ch, refused = W.schedule_merge(g, ds, arr, rule=lambda a, b: True); assert ch == [0, 1, 2] and refused >= 1                    # a rule that passes everything: the gate refuses, and the counter says so
    ends2 = {W.graph_hash(W.schedule_merge(g, ds, arr, rule=lambda a, b: True)[0]) for arr in orders}; assert len(ends2) == 2           # and the end state depends on the order


def test_merge_keeps_an_edit_only_if_it_is_independent_of_every_edit_already_kept():
    g = base_graph(); ds = [P("ADD[2,2:5]"), P("ADD[2,3:5]"), P("ADD[2,2:6]")]                    # the third clashes with the first and not with the second
    for arr in itertools.permutations(range(3)):
        h, ch, refused = W.schedule_merge(g, ds, list(arr)); assert ch == [0, 1] and refused == 0 and h.nodes[(2, 2)] == 5 and (2, 3) in h.nodes


def test_the_round_the_sweep_and_the_pair_runs_on_scripted_edits_give_the_numbers_computed_by_hand():
    g, ds = _round(); it = itertools.cycle(ds); mk = lambda *a: base_graph(); dr = lambda *a: next(it)
    r = W.run_rounds(seed=1, rounds=1, n_agents=3, perms=20, regimes=(("x", 0.0),), make_graph=mk, draw=dr)["x"]["schedulers"]
    assert r["blind"]["same_graph"] == 1 and r["blind"]["ill_formed"] == 1 and r["blind"]["mean_taken"] == 1.0 and r["blind"]["gate_refusals"] == 0
    assert r["gated"]["same_graph"] == 0 and r["gated"]["ill_formed"] == 0 and abs(r["gated"]["mean_taken"] - 2 / 3) < 1e-9 and r["gated"]["gate_refusals"] == 0
    assert r["merge"]["same_graph"] == 1 and r["merge"]["ill_formed"] == 0 and abs(r["merge"]["mean_taken"] - 2 / 3) < 1e-9 and r["merge"]["gate_refusals"] == 0
    it = itertools.cycle(ds); every = lambda gg, d, a: W.schedule_merge(gg, d, a, rule=lambda x, y: True)
    r = W.run_rounds(seed=1, rounds=1, n_agents=3, perms=20, regimes=(("x", 0.0),), make_graph=mk, draw=dr, merge=every)["x"]["schedulers"]["merge"]
    assert r["gate_refusals"] == 20 and r["same_graph"] == 0 and r["mean_taken"] == 1.0 and r["ill_formed"] == 0
    three = [P("ADD[2,2:5]"), P("ADD[2,3:5]"), P("ADD[2,2:6]")]; it = itertools.cycle(three)
    s = W.run_sweep(seed=1, rounds=2, ns=(3,), sizes=((7, 5, (10, 10)),), make_graph=mk, draw=dr); assert set(s) == {"7/3"} and abs(s["7/3"]["mean_taken"] - 2 / 3) < 1e-9 and s["7/3"]["agents"] == 3 and s["7/3"]["services"] == 7
    it = iter([P(x) for ta, tb, *_ in CASES.values() for x in (ta, tb)])
    d = W.run_pairs(seed=1, graphs=1, per_graph=12, regimes=(("x", 0.0),), make_graph=mk, draw=dr)["x"]
    assert d["pairs"] == 12 and d["conflict"] == 8 and d["commute"] == 4 and d["both_valid"] == 12 and d["first_commit_ok"] == 12 and d["why"] == {"both orders refused": 8}
    assert d["detectors"] == {"dpo": dict(tp=8, fn=0, fp=0, tn=4), "pager": dict(tp=5, fn=3, fp=2, tn=2), "names": dict(tp=6, fn=2, fp=0, tn=4)}
    assert d["misses"]["dpo"] == {} and d["misses"]["pager"] == {"fn -e/-e": 1, "fn -e/-e-n": 1, "fn -e-n/-e-n": 1, "fp +e/+e": 1, "fp +e+n/+e+n": 1} and d["misses"]["names"] == {"fn +e/-e-n": 1, "fn +e+n/-e-n": 1}
    assert len(d["examples"]["pager"]) == 3 and d["examples"]["dpo"] == [] and all(set(e) == {"cell", "a", "b"} for e in d["examples"]["pager"])


def test_the_pair_run_counts_an_invalid_edit_a_one_order_clash_and_a_failed_first_commit():
    g0 = base_graph(); a, b = P("ADD[3,3:1]"), P("ADD[(3,3)->(0,0):dep#1]"); it = iter([a, b, b, a]); mk = lambda *x: base_graph(); dr = lambda *x: next(it)      # b needs a: invalid on the snapshot, accepted after a only
    d = W.run_pairs(seed=1, graphs=1, per_graph=2, regimes=(("x", 0.0),), make_graph=mk, draw=dr)["x"]
    assert d["pairs"] == 2 and d["both_valid"] == 0 and d["conflict"] == 2 and d["commute"] == 0 and d["why"] == {"one order only": 2}
    assert d["detectors"]["dpo"] == dict(tp=2, fn=0, fp=0, tn=0) and d["first_commit_ok"] == 2                                              # the rule sees: a creates what b needs
    it = iter([P("ADD[2,2:5]"), P("ADD[2,2:5]")]); d = W.run_pairs(seed=1, graphs=1, per_graph=1, regimes=(("x", 0.0),), make_graph=mk, draw=dr)["x"]
    assert d["both_valid"] == 1 and d["conflict"] == 1 and d["why"] == {"both orders refused": 1}                                           # the same edit twice
    orig = W.pager_clash
    try:
        W.pager_clash = lambda tmpl, x, y: (False, True); it = iter([a, a, a, a]); d = W.run_pairs(seed=1, graphs=1, per_graph=2, regimes=(("x", 0.0),), make_graph=mk, draw=dr)["x"]
        assert d["first_commit_ok"] == 0 and d["detectors"]["pager"]["tp"] == 2
    finally: W.pager_clash = orig
    assert check(g0, a).ok and not check(g0, b).ok


def test_a_small_round_run_keeps_the_gate_schedulers_well_formed():
    r = W.run_rounds(seed=3, rounds=10, n_agents=8, perms=4)
    for name, d in r.items():
        s = d["schedulers"]; assert d["rounds"] == 10 and set(s) == {"blind", "gated", "merge"}
        assert s["gated"]["ill_formed"] == 0 and s["merge"]["ill_formed"] == 0 and s["blind"]["gate_refusals"] == 0 and s["gated"]["gate_refusals"] == 0           # the gate keeps a state well formed whatever the rule says
        for v in s.values(): assert 0 < v["mean_taken"] <= 1 and 0 <= v["same_graph"] <= 10
        assert s["blind"]["mean_taken"] == 1.0
    s = W.run_sweep(seed=3, rounds=4, ns=(1, 8), sizes=((40, 60, (10, 10)),)); assert set(s) == {"40/1", "40/8"} and s["40/1"]["mean_taken"] == 1.0 and 0 < s["40/8"]["mean_taken"] <= 1
    assert W.run_rounds(seed=3, rounds=10, n_agents=8, perms=4) == r and W.run_sweep(seed=3, rounds=4, ns=(1, 8), sizes=((40, 60, (10, 10)),)) == s


# ---------------------------------------------------------------- 6 the stores
class NoGateStore(W.Store):
    """A store that checks nothing: the edit is written straight into the state."""
    name = "nogate"

    def commit(self, base, delta, snap):
        with self.lock: W.blind_apply(self.head, delta); self.log.append((delta, W.footprint(delta))); self.version += 1; self.stats["accepted"] += 1; return "ok"


def test_the_stores_and_the_replay_tell_a_clash_a_stale_snapshot_a_lost_update_and_a_missing_gate():
    g, (A, B, C) = _round()
    s = W.Store(g); v, snap = s.snapshot(); assert v == 0 and same_graph(snap, g) and snap is not s.head
    assert s.commit(0, A, snap) == "ok" and s.commit(0, B, snap) == "clash" and s.commit(0, C, snap) == "ok" and s.version == 2 and s.stats["clash"] == 1 and W.replay(g, s) and W.well_formed(s.head)
    t = W.GateOnlyStore(g); assert t.commit(0, A, snap) == "ok" and t.commit(0, B, snap) == "gate" and t.commit(0, C, snap) == "ok" and t.stats["clash"] == 0 and W.replay(g, t)
    assert same_graph(s.head, t.head)                                                                                  # same refusals, same state, by different routes
    l = W.LastWriterStore(g); assert l.commit(0, A, snap) == "ok" and l.commit(0, C, snap) == "ok" and (1, 2) in l.head.nodes and not W.replay(g, l)        # the second edit was made on the old snapshot: the first is lost
    n = NoGateStore(g); n.commit(0, A, snap); n.commit(0, B, snap); assert not W.well_formed(n.head) and not W.replay(g, n)
    u = W.Store(g); u.commit(0, A, snap); assert u.commit(0, P("ADD[(2,0)->(0,2):owns#1]"), snap) == "ok" and u.commit(1, B, u.snapshot()[1]) == "gate"        # B claims version 1 but was made on version 0: the rule passes it, the gate refuses it
    assert u.stats["gate_refused"] == 1


def test_real_threads_keep_the_checking_stores_correct_and_the_replay_catches_the_control():
    for cls in (W.Store, W.GateOnlyStore):
        r = W.run_threads(cls, seed=4, n_threads=4, rounds=40)
        assert r["attempts"] == 160 and r["finished"] and r["replay_ok"] and r["well_formed"] and r["accounted"] and not r["errors"] and r["accepted"] + r["clash"] + r["gate"] == 160 and r["accepted"] > 20, (cls.name, r)
        if cls is W.Store: assert r["clash"] > 0 and r["gate"] == r["gate_after_rule"]
    class RuleBlind(W.Store):                                    # keeps the name "rule" but passes every edit to the gate: the gate's refusals are then "after the rule"
        def _admit(self, base, fp): return True
    r = W.run_threads(RuleBlind, seed=4, n_threads=4, rounds=40); assert r["finished"] and r["replay_ok"] and r["gate"] > 0 and r["gate_after_rule"] == r["gate"] and r["clash"] == 0
    r = W.run_threads(W.GateOnlyStore, seed=4, n_threads=4, rounds=40); assert r["gate_after_rule"] == 0
    r = W.run_threads(W.LastWriterStore, seed=4, n_threads=4, rounds=40); assert r["finished"] and not r["replay_ok"]
    r = W.run_threads(NoGateStore, seed=4, n_threads=4, rounds=40); assert r["finished"] and (not r["replay_ok"] or not r["well_formed"])
    class Boom(W.Store):
        def commit(self, base, delta, snap): raise ValueError("boom")
    r = W.run_threads(Boom, seed=4, n_threads=3, rounds=5); assert not r["finished"] and r["errors"] and not r["accounted"]
    class BoomOnce(W.Store):                                     # one worker raises while the others wait for it at the barrier: the abort lets them out
        def commit(self, base, delta, snap):
            if not getattr(self, "boomed", False): self.boomed = True; raise ValueError("boom")
            return super().commit(base, delta, snap)
    r = W.run_threads(BoomOnce, seed=4, n_threads=3, rounds=5); assert not r["finished"] and any("boom" in e for e in r["errors"]) and not r["accounted"] and r["attempts"] == 15
    class Lying(W.Store):                                        # says "ok" and keeps no log
        def commit(self, base, delta, snap): return "ok"
    r = W.run_threads(Lying, seed=4, n_threads=3, rounds=5); assert r["finished"] and r["accepted"] == 15 and not r["accounted"] and r["well_formed"]


# ---------------------------------------------------------------- 7 summary, reading rules, quick run
def fake():
    side = lambda **k: dict(dict(pairs=130000, conflict=2500, commute=127500, both_valid=130000, first_commit_ok=130000), **k)
    det = lambda tp, fn, fp, tn: dict(tp=tp, fn=fn, fp=fp, tn=tn)
    P_ = {"uniform": side(detectors={"dpo": det(2500, 0, 0, 127500), "pager": det(1900, 600, 1000, 126500), "names": det(1500, 1000, 0, 127500)}),
          "hot": side(detectors={"dpo": det(2500, 0, 0, 127500), "pager": det(1900, 600, 4000, 123500), "names": det(1500, 1000, 0, 127500)})}
    sch = lambda same, bad, taken, ref=0: dict(same_graph=same, ill_formed=bad, mean_taken=taken, gate_refusals=ref)
    R_ = {r: dict(rounds=100, schedulers={"blind": sch(10, 70, 1.0), "gated": sch(10, 0, 0.62), "merge": sch(100, 0, 0.62)}) for r in ("uniform", "hot")}
    vals = {40: [.95, .80, .62], 160: [.99, .95, .86], 640: [.999, .99, .95]}
    S_ = {f"{s}/{n}": dict(services=s, agents=n, mean_taken=vals[s][i]) for s in vals for i, n in enumerate((4, 16, 64))}
    st = lambda **k: dict(dict(finished=True, replay_ok=True, well_formed=True, accounted=True, gate_after_rule=0, clash=200, attempts=2400, accepted=2000, gate=0, errors=[]), **k)
    T_ = {"rule": st(), "gate": st(clash=0, gate=150), "lww": st(replay_ok=False)}
    return dict(seed=41, pairs=P_, rounds=R_, sweep=S_, threads=T_, digest="x", seconds=1.0)


def _bad(mut=None):
    r = copy.deepcopy(fake())
    if mut: mut(r)
    return [i for i, (_, ok) in enumerate(W.verdict_swarm(W.summarize_swarm(r)), 1) if not ok]


def test_the_nineteen_reading_rules_flip_exactly_at_their_thresholds():
    assert _bad() == [] and len(W.SWARM_LINES) == 19 and W.SWARM_HEADLINE == (5, 6, 10, 16)
    def setp(r, reg, key, v): r["pairs"][reg][key] = v
    def setd(r, reg, det, cell, v): r["pairs"][reg]["detectors"][det][cell] = v
    def sets(r, key, v): r["sweep"][key]["mean_taken"] = v
    def pooled_conflict(r, c):
        for reg in r["pairs"]: setp(r, reg, "conflict", c)
    def pooled_commute(r, c):
        for reg in r["pairs"]: setp(r, reg, "commute", c)
    flips = [
        (lambda r: pooled_conflict(r, 1000), []), (lambda r: pooled_conflict(r, 999), [1]), (lambda r: pooled_commute(r, 10000), []), (lambda r: pooled_commute(r, 9999), [1]),
        (lambda r: setp(r, "hot", "both_valid", 129999), [2]), (lambda r: setp(r, "uniform", "first_commit_ok", 129999), [2]),
        (lambda r: r["threads"]["gate"].update(finished=False), [3]), (lambda r: r["threads"]["lww"].update(finished=False), [3]), (lambda r: r["threads"]["lww"].update(replay_ok=True), [4]),
        (lambda r: setd(r, "hot", "dpo", "fn", 1), [5]), (lambda r: setd(r, "uniform", "dpo", "fn", 1), [5]),
        (lambda r: setd(r, "hot", "dpo", "fp", 1275), []), (lambda r: setd(r, "hot", "dpo", "fp", 1276), [6]), (lambda r: setd(r, "uniform", "dpo", "fp", 1276), [6]),
        (lambda r: (setd(r, "uniform", "pager", "fn", 250), setd(r, "hot", "pager", "fn", 250)), []), (lambda r: (setd(r, "uniform", "pager", "fn", 250), setd(r, "hot", "pager", "fn", 249)), [7]),
        (lambda r: setd(r, "hot", "pager", "fp", 2550), []), (lambda r: setd(r, "hot", "pager", "fp", 2549), [8]), (lambda r: setd(r, "uniform", "pager", "fp", 0), []),
        (lambda r: (setd(r, "uniform", "names", "fn", 125), setd(r, "hot", "names", "fn", 125)), []), (lambda r: (setd(r, "uniform", "names", "fn", 125), setd(r, "hot", "names", "fn", 124)), [9]),
        (lambda r: r["rounds"]["hot"]["schedulers"]["merge"].update(same_graph=99), [10]), (lambda r: r["rounds"]["uniform"]["schedulers"]["merge"].update(same_graph=99), [10]), (lambda r: r["rounds"]["hot"]["schedulers"]["merge"].update(gate_refusals=1), [10]),
        (lambda r: r["rounds"]["hot"]["schedulers"]["gated"].update(same_graph=20), []), (lambda r: r["rounds"]["hot"]["schedulers"]["gated"].update(same_graph=21), [11]), (lambda r: r["rounds"]["uniform"]["schedulers"]["gated"].update(same_graph=90), []),
        (lambda r: r["rounds"]["hot"]["schedulers"]["blind"].update(ill_formed=10), []), (lambda r: r["rounds"]["hot"]["schedulers"]["blind"].update(ill_formed=9), [12]), (lambda r: r["rounds"]["uniform"]["schedulers"]["blind"].update(ill_formed=0), []),
        (lambda r: r["rounds"]["uniform"]["schedulers"]["gated"].update(ill_formed=1), [13]), (lambda r: r["rounds"]["hot"]["schedulers"]["merge"].update(ill_formed=1), [13]),
        (lambda r: sets(r, "160/64", .954), []), (lambda r: sets(r, "160/64", .956), [14]), (lambda r: [sets(r, f"640/{n}", .99) for n in (4, 16, 64)], [14]), (lambda r: sets(r, "40/16", .96), [14]),
        (lambda r: sets(r, "40/64", .45), []), (lambda r: sets(r, "40/64", .449), [15]), (lambda r: sets(r, "40/64", .80), []), (lambda r: sets(r, "40/64", .801), [15]), (lambda r: sets(r, "640/64", .90), []), (lambda r: sets(r, "640/64", .899), [15]),
        (lambda r: r["threads"]["rule"].update(replay_ok=False), [16]), (lambda r: r["threads"]["rule"].update(well_formed=False), [16]), (lambda r: r["threads"]["rule"].update(accounted=False), [16]), (lambda r: r["threads"]["rule"].update(gate_after_rule=1), [16]),
        (lambda r: r["threads"]["gate"].update(replay_ok=False), [17]), (lambda r: r["threads"]["gate"].update(accounted=False), [17]), (lambda r: r["threads"]["gate"].update(well_formed=False), [17]),
        (lambda r: r["threads"]["rule"].update(clash=72), []), (lambda r: r["threads"]["rule"].update(clash=71), [18]),
        (lambda r: r["rounds"]["hot"]["schedulers"]["merge"].update(mean_taken=.649), []), (lambda r: r["rounds"]["hot"]["schedulers"]["merge"].update(mean_taken=.652), [19]), (lambda r: r["rounds"]["uniform"]["schedulers"]["gated"].update(mean_taken=.589), [19]),
        (lambda r: (r["rounds"]["hot"]["schedulers"]["gated"].update(mean_taken=0.0), r["rounds"]["hot"]["schedulers"]["merge"].update(mean_taken=0.03)), []),
        (lambda r: (r["rounds"]["hot"]["schedulers"]["gated"].update(mean_taken=0.0), r["rounds"]["hot"]["schedulers"]["merge"].update(mean_taken=0.031)), [19])]
    for mut, want in flips: assert _bad(mut) == want, (_bad(mut), want)
    s = W.summarize_swarm(fake()); assert s["pairs"] == 260000 and abs(s["pager_missed"] - .24) < 1e-9 and abs(s["pager_false"]["hot"] - 4000 / 127500) < 1e-12 and abs(s["names_missed"] - .4) < 1e-9 and s["falls_with_n"] and s["rises_with_size"]


def test_the_quick_run_the_printout_and_the_command_line():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "x.json"); buf = io.StringIO()
        with contextlib.redirect_stdout(buf): W.main(["--quick", "--out", out, "--seed", "5"])
        res = json.load(open(out)); text = buf.getvalue()
        assert res["quick"] and res["seed"] == 5 and set(res) >= {"pairs", "rounds", "sweep", "threads", "digest"} and "digest" in text and "HEADLINE" in text and text.count("[instrument") == 4 and text.count("[claim") == 15
        checks = W.verdict_swarm(W.summarize_swarm(res)); assert len(checks) == 19 and [k for k, _ in checks] == [k for k, _ in W.SWARM_LINES]
        assert W.run_swarm(5, quick=True)["digest"] == res["digest"]
        assert hashlib.sha256(json.dumps([res["pairs"], res["rounds"], res["sweep"]], sort_keys=True, default=str).encode()).hexdigest()[:16] == res["digest"]
        res2 = copy.deepcopy(res); res2["sweep"]["40/4"]["mean_taken"] += 0.001; assert hashlib.sha256(json.dumps([res2["pairs"], res2["rounds"], res2["sweep"]], sort_keys=True, default=str).encode()).hexdigest()[:16] != res["digest"]
    for r in ("uniform", "hot"): assert f"pairs, regime {r}" in text and f"rounds, regime {r}" in text
    for s in ("rule", "gate", "lww"): assert f"threads, {s}" in text


# ---------------------------------------------------------------- 8 the page
def _head():
    text = open(PAGE).read(); assert text.count(RESULT_MARK) <= 1
    return text[:text.index(RESULT_MARK)] if RESULT_MARK in text else text


def test_the_page_is_locked_and_carries_the_nineteen_lines_and_the_settings_the_code_uses():
    head = _head(); assert _h(head) == PAGE_SHA256, "the predictions and reading rules must not change after the run; later notes go below them"
    for i, (tag, line) in enumerate(W.SWARM_LINES, 1): assert f"\n{i}. [{tag}] {line} (" in head, i
    probs = [int(m.group(1)) for m in re.finditer(r"\n\d+\. \[\w+\] .*\((\d+) %\)(?=\n)", head)]; assert probs == [92, 99, 97, 99, 90, 88, 90, 70, 93, 97, 80, 92, 99, 90, 75, 95, 93, 90, 95]
    prod = 100.0
    for i in W.SWARM_HEADLINE: prod *= probs[i - 1] / 100
    assert abs(prod - 73) <= 0.5 and "about 73 %" in head and "about **70 %**" in head and "**Headline** (lines 5, 6, 10 and 16)" in head
    assert (W.SWARM_SEED, W.PAIR_GRAPHS, W.PAIR_PER_GRAPH, W.PAIR_NODES, W.PAIR_EDGES, W.PAIR_GRID) == (41, 20, 6000, 40, 60, (10, 10)) and (W.ROUND_N, W.ROUND_ROUNDS, W.ROUND_PERMS) == (16, 100, 10)
    assert W.SWEEP_N == (4, 16, 64) and [s[0] for s in W.SWEEP_SIZES] == [40, 160, 640] and [s[1] for s in W.SWEEP_SIZES] == [60, 240, 960] and W.SWEEP_ROUNDS == 400 and (W.THREADS, W.THREAD_ROUNDS, W.THREAD_NODES, W.THREAD_EDGES) == (8, 300, 60, 90)
    assert dict(W.REGIMES) == {"uniform": 0.0, "hot": 0.5} and dict(W.KIND_WEIGHTS) == {"add_node": 15, "add_edge": 30, "del_edge": 20, "del_leaf": 10, "del_dep": 15, "multi": 10}
    assert (W.S_MIN_CONFLICT, W.S_MIN_COMMUTE, W.S_FALSE_ALARM, W.S_PAGER_MISS, W.S_PAGER_FALSE, W.S_NAMES_MISS, W.S_GATED_DIFFER, W.S_BLIND_BAD, W.S_THREAD_CONFLICT) == (2000, 20000, 0.01, 0.10, 0.02, 0.05, 0.80, 0.10, 0.03)
    assert (W.S_ACC_SMALL_LO, W.S_ACC_SMALL_HI, W.S_ACC_BIG, W.S_TREND_TOL, W.S_TREND_MIN, W.S_SAME_TAKEN) == (0.45, 0.80, 0.90, 0.005, 0.01, 0.03)
    for needle in ("Concept-map item 100", "`python/axi/experiments/swarm.py`", "python -m axi.experiments.swarm --out axi_swarm.json", "120,000 pairs per regime, 240,000 in all", "add a service 15, add an edge 30, delete an edge 20, delete a service without edges 10, delete a service with its edges (1 to 4) 15, and a compound edit 10",
                   "Seed 41", "100 rounds per regime", "400 rounds each", "Eight real threads, 300 rounds", "Graph of 60 services and 90 edges", "`lww`", "I did not print or record any rate", "I do not attribute them to him", "so I do not attribute them to him",
                   "10,000 AI agents can read and write to the exact same enterprise state graph simultaneously using Copy-on-Write (CoW) shadow paging", "Without AXI, multi-agent swarms inevitably overwrite each other's context and corrupt the shared environment",
                   "AXI acts as the atomic transaction manager for the swarm", "no test of this", "Church-Rosser", "Ehrig, Pfender and Schneider (1973)", "Kung and Robinson (1981)", "Shapiro et al. (2011)", "Berenson et al. (1995)", "(L_a ∩ L_b) is inside (K_a ∩ K_b)",
                   "item 101", "`swarm.verdict_swarm`", "the Python `Pager` is a model of its semantics, not the crate"):
        assert needle in head or needle.replace("so I do not", "I do not") in head, needle


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("PASS", n)
