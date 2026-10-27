"""Implements concept-map item 130 (meeting rings: the view for a request about two things is every shortest path between them;
python/axi/engine/focus.py, `focus_meet`; docs/FOCUSED_VIEW.md), test 156. Prompted by Rajnish Choubey's lines in "The Gen 2 Reset"
(each word a centre with expanding rings; meaning sparks where two words' rings intersect).
It checks, with no model:
  1. `focus_meet` on made-up graphs: a line, two shortest paths, no path ("not connected"), a cap on the rings ("no meeting within
     max_hops"), edges followed one way, a filter on relations, one node, a node not in the graph; the view is a copy;
  2. exact against a plain search from both ends: on every pair of 200 small service graphs (the scope tests' generator) and of a
     120-node graph (the large KV test's generator), both ways and one way, the view holds exactly the nodes and edges of the shortest
     paths, and the count of shortest paths is right;
  3. against the fixed rings of `focus_graph` (1, 2 and 3 hops from both nodes) on the four 120-node graphs of the large test: how
     often each view holds a path between the two, and its size; a fixed view holds one exactly when the distance is at most twice its
     hops; on the 8-node graphs the whole graph is already small;
  4. a made-up graph of 10,000 nodes: the views, the work done (rings from both ends against rings from one) and the time;
  5. the size of the views in the large test's crude tokens;
  6. the page and the front pages (concept map item 130, VERIFIED row, the counter).
What it does NOT show: that a model reads a connection better from this view; no test of AXI asks about a connection yet."""
import itertools, os, random, statistics, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.focus import focus_graph, focus_meet
from axi.engine.gate import Graph
from axi.experiments.agent_loop import _serialize, make_task
from axi.experiments.kv_large import crude_tokens, random_large_graph

HERE = os.path.dirname(__file__)
DOCS = os.path.join(HERE, "..", "..", "docs")
PAGE = os.path.join(DOCS, "FOCUSED_VIEW.md")
NEXT = "Next free: row 191, item 144, test 169, question 122"


def graph(nodes, edges):
    g = Graph()
    for n in nodes: g.add_node(n, 1)
    for u, v, r in edges: g.add_edge(u, v, r, 1)
    return g


def test_meeting_rings_on_made_up_graphs():
    a, b, c, d, e, x = (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (9, 9)
    line = graph([a, b, c, d, e, x], [(a, b, "dep"), (b, c, "dep"), (c, d, "dep"), (d, e, "dep")])
    v, why = focus_meet(line, a, e); assert set(v.nodes) == {a, b, c, d, e} and len(v.edges) == 4 and why["distance"] == 4 and why["rings"] == (2, 2) and why["paths"] == 1
    v, why = focus_meet(line, e, a); assert set(v.nodes) == {a, b, c, d, e} and why["distance"] == 4
    v, why = focus_meet(line, a, x); assert not why["met"] and why["reason"] == "not connected" and set(v.nodes) == {a, x} and v.edges == {}
    v, why = focus_meet(line, a, e, max_hops=3); assert not why["met"] and why["reason"] == "no meeting within max_hops"
    v, why = focus_meet(line, c, c); assert set(v.nodes) == {c} and why["distance"] == 0 and why["paths"] == 1
    assert focus_meet(line, a, (5, 5))[1]["reason"] == "a named node is not in the graph"
    p, q = (1, 0), (1, 1)                                                   # a diamond: two shortest paths, a longer one left out
    dia = graph([a, p, q, b, c, d], [(a, p, "dep"), (a, q, "dep"), (p, b, "dep"), (q, b, "owns"), (a, c, "dep"), (c, d, "dep"), (d, b, "dep")])
    v, why = focus_meet(dia, a, b); assert set(v.nodes) == {a, p, q, b} and why["paths"] == 2 and why["distance"] == 2
    v, why = focus_meet(dia, a, b, rels={"dep"}); assert set(v.nodes) == {a, p, b} and why["paths"] == 1
    one_way = graph([a, b, c], [(a, b, "dep"), (c, b, "dep")])
    assert focus_meet(one_way, a, c)[1]["distance"] == 2 and not focus_meet(one_way, a, c, directed=True)[1]["met"]
    assert focus_meet(one_way, a, b, directed=True)[1]["distance"] == 1 and not focus_meet(one_way, b, a, directed=True)[1]["met"]
    v, _ = focus_meet(line, a, e); v.nodes[a] = 7; assert line.nodes[a] == 1 and v.is_well_formed()


def dists(g, s, fwd=None):
    """A plain search from s: the distance of every node it reaches (fwd True: along edges, False: against them, None: either way)."""
    d = {s: 0}; f = [s]
    while f:
        nx = []
        for n in f:
            for (u, v, _) in g.incident(n):
                for m, ok in ((v, u == n and fwd is not False), (u, v == n and fwd is not True)):
                    if ok and m not in d: d[m] = d[n] + 1; nx.append(m)
        f = nx
    return d


def exact_on_every_pair(g, directed=False):
    fw = {n: dists(g, n, True if directed else None) for n in g.nodes}; bw = {n: dists(g, n, False if directed else None) for n in g.nodes}
    pairs = itertools.permutations(sorted(g.nodes), 2) if directed else itertools.combinations(sorted(g.nodes), 2); n_pairs = n_met = 0
    for a, b in pairs:
        v, why = focus_meet(g, a, b, directed=directed); da, db = fw[a], bw[b]; n_pairs += 1
        if b not in da: assert not why["met"] and why["reason"] == "not connected"; continue
        d = da[b]; n_met += 1; on = {x for x in da if x in db and da[x] + db[x] == d}
        assert why["met"] and why["distance"] == d and set(v.nodes) == on
        steps = {(u, w, r) for (u, w, r) in g.edges if u in on and w in on and (da[u] + 1 == da[w] or (not directed and da[w] + 1 == da[u]))}
        assert set(v.edges) == steps and v.is_well_formed()
        ways = {a: 1}                                                       # count the shortest paths layer by layer
        for x in sorted(on, key=lambda y: da[y]):
            if x != a: ways[x] = sum(ways[u if w == x else w] for (u, w, _) in steps if x in (u, w) and da[u if w == x else w] == da[x] - 1)
        assert why["paths"] == ways[b]
    return n_pairs, n_met


def small_graphs(): return [make_task(random.Random(i), "delete_dep").graph for i in range(200)]
def large_graphs(): return [random_large_graph(random.Random(s)) for s in (21000, 21001, 21002, 21003)]


def test_exact_against_a_plain_search_on_every_pair():
    tot = [0, 0]
    for g in small_graphs():
        n, m = exact_on_every_pair(g); tot[0] += n; tot[1] += m
    assert tot == [5600, 3591]
    g = large_graphs()[0]
    assert exact_on_every_pair(g) == (7140, 5886) and exact_on_every_pair(g, directed=True) == (14280, 7195)


def holds_path(view, a, b): return a in view.nodes and b in view.nodes and b in dists(view, a)


def survey(graphs):
    rows = []
    for g in graphs:
        fw = {n: dists(g, n) for n in g.nodes}
        for a, b in itertools.combinations(sorted(g.nodes), 2):
            v, why = focus_meet(g, a, b)
            if not why["met"]: continue
            d = why["distance"]; fixed = {h: focus_graph(g, [a, b], hops=h) for h in (1, 2, 3)}
            rows.append(dict(d=d, nodes=len(v.nodes), edges=len(v.edges), reached=why["reached"], one=sum(1 for x in fw[a].values() if x <= d),
                             fixed={h: (holds_path(f, a, b), len(f.nodes), len(f.edges)) for h, f in fixed.items()}))
    return rows


def test_against_fixed_rings_on_the_large_test_graphs():
    rows = survey(large_graphs()); n = len(rows); assert n == 24321
    assert all(r["fixed"][h][0] == (r["d"] <= 2 * h) for r in rows for h in (1, 2, 3))
    assert {d: sum(r["d"] == d for r in rows) for d in range(1, 10)} == {1: 790, 2: 2504, 3: 5951, 4: 7808, 5: 4997, 6: 1796, 7: 408, 8: 61, 9: 6}
    m = lambda k: round(statistics.mean(r[k] for r in rows), 2)
    assert (m("nodes"), m("edges"), round(m("reached"), 1), round(m("one"), 1)) == (6.72, 6.72, 27.9, 70.0)
    held = {h: sum(r["fixed"][h][0] for r in rows) for h in (1, 2, 3)}; assert held == {1: 3294, 2: 17053, 3: 23846}
    size = {h: (round(statistics.mean(r["fixed"][h][1] for r in rows), 1), round(statistics.mean(r["fixed"][h][2] for r in rows), 1)) for h in (1, 2, 3)}
    assert size == {1: (9.0, 7.2), 2: (29.1, 30.6), 3: (66.5, 88.2)}
    small = survey(small_graphs()); assert len(small) == 3591
    assert {h: sum(r["fixed"][h][0] for r in small) for h in (1, 2, 3)} == {1: 2823, 2: 3561, 3: 3591}
    assert round(statistics.mean(r["nodes"] for r in small), 2) == 3.01                    # of 8 nodes: the whole graph is already small


def test_a_made_up_graph_of_ten_thousand_nodes():
    g = random_large_graph(random.Random(90), n_nodes=10000, n_edges=16667, grid=(120, 120)); rng = random.Random(90); ns = sorted(g.nodes); rows = []
    for _ in range(300):
        a, b = rng.sample(ns, 2); v, why = focus_meet(g, a, b)
        if why["met"]:
            d = why["distance"]; one = sum(1 for x in dists(g, a).values() if x <= d)
            rows.append((d, len(v.nodes), why["reached"], one, {h: holds_path(focus_graph(g, [a, b], hops=h), a, b) for h in (1, 2, 3)}))
        else: assert why["reason"] == "not connected"
    assert len(rows) == 252 and round(statistics.mean(r[0] for r in rows), 2) == 7.38
    assert (round(statistics.mean(r[1] for r in rows), 1), round(statistics.mean(r[2] for r in rows)), round(statistics.mean(r[3] for r in rows))) == (14.5, 300, 5728)
    assert {h: sum(r[4][h] for r in rows) for h in (1, 2, 3)} == {1: 0, 2: 2, 3: 62}
    ms = []
    for _ in range(100):
        a, b = rng.sample(ns, 2); t = time.perf_counter(); focus_meet(g, a, b); ms.append((time.perf_counter() - t) * 1000)
    assert statistics.mean(ms) < 5, statistics.mean(ms)


def test_the_views_in_tokens():
    gs = large_graphs(); whole = statistics.mean(crude_tokens(_serialize(g)) for g in gs); rng = random.Random(7); meet, fixed3 = [], []
    for g in gs:
        ns = sorted(g.nodes)
        for _ in range(250):
            a, b = rng.sample(ns, 2); v, why = focus_meet(g, a, b)
            if why["met"]: meet.append(crude_tokens(_serialize(v))); fixed3.append(crude_tokens(_serialize(focus_graph(g, [a, b], hops=3))))
    assert len(meet) == 836 and (round(whole), round(statistics.mean(fixed3)), round(statistics.mean(meet))) == (3554, 1652, 151)


def test_the_page_and_the_front_pages():
    page = open(PAGE).read(); low = page.lower()
    for s in ("focus_meet", "The Gen 2 Reset", "rings intersect", "Meeting rings", "24,321 of 24,321", "6.7", "13.5 %", "70.1 %", "98.0 %", "9.0", "29.1",
              "66.5", "3,554", "1,652", "151", "14.5", "62 of 252", "5,728", "300", "not connected", "a standard method", "no test of AXI asks", "test 156",
              "whole exact state", "item 124"):
        assert s.lower() in low, s
    for bad in ("proven", "$", "dollar", "100 percent"): assert bad not in low, bad
    cm = open(os.path.join(DOCS, "RAJNISH_CONCEPT_MAP.md")).read(); vf = open(os.path.join(DOCS, "VERIFIED.md")).read()
    assert NEXT in cm and cm.count("Next free:") == 1 and "test 156 checks the meeting rings, item 130" in cm
    j = cm.index("\n130. **") + 1; item = cm[j:cm.index("\n", j)]
    assert "focus_meet" in item and "FOCUSED_VIEW.md" in item and "test 156" in item and "Rajnish Choubey" in item
    row = next(l for l in vf.split("\n") if "focus_meet" in l); assert "test_focus_meet.py" in row and "no model" in row.lower()


if __name__ == "__main__":
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]; bad = 0
    for n, f in fns:
        try: f(); print("ok  ", n)
        except Exception: bad += 1; print("FAIL", n); traceback.print_exc()
    print(f"{len(fns) - bad}/{len(fns)} passed"); sys.exit(1 if bad else 0)
