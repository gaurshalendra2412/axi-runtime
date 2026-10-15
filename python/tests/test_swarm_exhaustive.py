"""Engineering support for the exhaustive pass of docs/SWARM_TEST.md (concept-map item 100; added after the result, not pre-registered): is the independence rule exact on EVERY pair of edits of a tiny universe?
Tests the INSTRUMENT, `axi/experiments/swarm_exhaustive.py`; the saved result of the full pass (three operations) is checked by test_swarm_result.py:
  1. the universe: 80 graphs on three cells (1 + 3 + 12 + 64 by number of services), all well formed, no two the same;
  2. the edits, counted by hand: 6, 18 and 74 on the empty graph for one, two and three operations; 41 on the complete graph (no service can be deleted with its four edges in three operations); every edit passes the gate;
  3. a run at two operations (151,476 ordered pairs, 6 seconds): the counts, the rule exact, the kinds of error of the other two;
  4. broken rules are noticed (never clash; always clash; names only), and a run on part of the graphs is a part of the full one;
  5. the command line and the printout."""
import contextlib, io, json, os, sys, tempfile

sys.path.insert(0, os.path.dirname(__file__))
import axi.experiments.swarm as W
import axi.experiments.swarm_exhaustive as X
from axi.engine.gate import check


def test_the_universe_is_the_eighty_graphs_on_three_cells():
    gs = list(X.graphs()); assert len(gs) == 80 and all(g.is_well_formed() for g in gs)
    by = {}
    for g in gs: by[len(g.nodes)] = by.get(len(g.nodes), 0) + 1
    assert by == {0: 1, 1: 3, 2: 12, 3: 64} and len({(tuple(sorted(g.nodes)), tuple(sorted(g.edges))) for g in gs}) == 80 and all(w == 1 and k[2] == "dep" for g in gs for k, w in g.edges.items())
    assert all(set(g.nodes) <= set(X.CELLS) and all(v == 1 for v in g.nodes.values()) for g in gs) and max(len(g.edges) for g in gs) == 6
    assert len(list(X.graphs(cells=((0, 0), (0, 1))))) == 1 + 2 + 4 and len(list(X.graphs(cells=((0, 0),)))) == 2


def test_the_edits_are_counted_by_hand_and_all_pass_the_gate():
    gs = list(X.graphs()); empty = next(g for g in gs if not g.nodes); full = next(g for g in gs if len(g.edges) == 6)
    assert len(X.operations(empty)) == 6 + 12 and [len(X.edits(empty, k)) for k in (1, 2, 3)] == [6, 18, 74]           # 3 cells x 2 values; + 3 pairs x 4; + 8 + 3 pairs x 4 values x 2 directions x 2 weights
    assert len(X.operations(full)) == 3 + 6 and [len(X.edits(full, k)) for k in (1, 2, 3)] == [6, 6 + 15, 6 + 15 + 20]    # only edge deletions: no service can go (it has four edges)
    for g in gs[::7]:
        es = X.edits(g, 3); texts = {W.to_text(d) if hasattr(W, "to_text") else repr(d) for d in es}; assert all(check(g, d).ok for d in es) and len(texts) == len(es)
        assert all(1 <= len(d.node_deltas) + len(d.edge_deltas) <= 3 for d in es)
    one = next(g for g in gs if sorted(g.nodes) == [(0, 0)] and not g.edges); assert any(d.node_deltas and d.node_deltas[0].op == "DEL" for d in X.edits(one, 1)) and len(X.edits(one, 1)) == 1 + 2 + 2
    two = next(g for g in gs if sorted(g.nodes) == [(0, 0), (0, 1)] and len(g.edges) == 1)
    deletes = [d for d in X.edits(two, 3) if any(x.op == "DEL" and (x.r, x.c) == (0, 0) for x in d.node_deltas)]; assert deletes and all(any(e.op == "DEL" for e in d.edge_deltas) for d in deletes)       # a service with an edge goes with the edge


def test_a_run_at_two_operations_finds_the_rule_exact_and_the_others_wrong_in_the_ways_the_page_says():
    r = X.run_exhaustive(max_ops=2); D = r["detectors"]
    assert (r["graphs"], r["pairs"], r["conflict"], r["commute"], r["edits_per_graph_max"], r["max_ops"]) == (80, 151476, 87570, 63906, 90, 2) and r["why"] == {"both orders refused": 87570}
    assert D["dpo"] == dict(tp=87570, fn=0, fp=0, tn=63906) and D["pager"] == dict(tp=78723, fn=8847, fp=42408, tn=21498) and D["names"] == dict(tp=77658, fn=9912, fp=0, tn=63906) and r["misses"]["dpo"] == {} and r["examples"]["dpo"] == []
    assert "fn -e/-e" in r["misses"]["pager"] and "fp +e/+e" in r["misses"]["pager"] and "fn +e/-e-n" not in r["misses"]["pager"] and any(k.startswith("fn +e/") and k.endswith("-n") for k in r["misses"]["names"])
    assert len(r["examples"]["pager"]) == 3 and len(r["examples"]["names"]) == 3 and all(set(e) == {"cell", "graph", "a", "b"} for e in r["examples"]["pager"] + r["examples"]["names"])
    part = X.run_exhaustive(max_ops=2, limit_graphs=10); assert part["graphs"] == 10 and 0 < part["pairs"] < r["pairs"] and part["detectors"]["dpo"]["fn"] == part["detectors"]["dpo"]["fp"] == 0
    assert X.run_exhaustive(max_ops=1)["pairs"] == sum(len(X.edits(g, 1)) ** 2 for g in X.graphs())


def test_broken_rules_are_noticed():
    never = X.run_exhaustive(max_ops=2, limit_graphs=12, rule=lambda a, b: True); assert never["detectors"]["dpo"]["fn"] == never["conflict"] > 0 and never["detectors"]["dpo"]["fp"] == 0
    always = X.run_exhaustive(max_ops=2, limit_graphs=12, rule=lambda a, b: False); assert always["detectors"]["dpo"]["fp"] == always["commute"] > 0 and always["detectors"]["dpo"]["fn"] == 0
    names = X.run_exhaustive(max_ops=2, limit_graphs=40, rule=lambda a, b: not W.names_clash(a, b)); assert names["detectors"]["dpo"]["fn"] > 0 and names["detectors"]["dpo"]["fp"] == 0 and names["examples"]["dpo"]
    assert names["detectors"]["dpo"] == names["detectors"]["names"]                                                         # the rule "no written item in common" is the names detector


def test_the_command_line_saves_the_result_and_prints_the_table():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "x.json"); buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()): X.main(["--max-ops", "1", "--out", out])
        res = json.load(open(out)); text = buf.getvalue()
        assert res["max_ops"] == 1 and res["graphs"] == 80 and res["seconds"] >= 0 and set(res["detectors"]) == {"dpo", "pager", "names"} and "every pair on a tiny universe: 80 graphs" in text and "dpo" in text and "MISSED" in text and "saved" in text


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("PASS", n)
