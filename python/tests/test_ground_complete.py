"""Implements concept-map rows 25-26 ("she cuts away the garbage" / Durga closes the boundary): ground_and_complete().
Cases 1-3 are patterns actually seen in the Qwen2.5-3B transcripts (seed 2, rule in the prompt): ADD instead of DEL for the incident
edges, phantom edge deletes, wrong weights."""
import random, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.cico_parser import CICOParser
from axi.engine.gate import Graph, check, apply
from axi.engine.diagnostics import diagnose, ground_and_complete, propose_repair
from axi.experiments.agent_loop import make_tasks, copy_graph, same_graph, blind_apply


def G():
    g = Graph()
    for n, v in {(0, 0): 1, (1, 1): 2, (2, 0): 3, (2, 1): 4, (2, 3): 5}.items(): g.add_node(n, v)
    g.add_edge((2, 3), (2, 0), "dep", 4); g.add_edge((2, 3), (2, 1), "dep", 6); g.add_edge((0, 0), (1, 1), "owns", 2); g.add_edge((0, 0), (2, 3), "dep", 7)
    return g


def P(t): return CICOParser.parse_transition_delta(t)


def cascade_expected(g, n):
    h = copy_graph(g)
    for k in sorted(g.incident(n)): h.del_edge(k)
    h.del_node(n); return h


def run(g, text, **kw):
    r = ground_and_complete(g, P(text), **kw)
    return r


def test_add_instead_of_del_for_incident_edges_is_cut_then_completed():          # seed-2 transcript pattern
    g = G(); r = run(g, "ADD[(2,3)->(2,0):dep#4] ADD[(2,3)->(2,1):dep#6] DEL[2,3]")
    assert r.repairable and len(r.cut) == 2 and all("will not exist" in c[0] for c in r.cut)
    h = copy_graph(g); apply(h, r.repaired); assert same_graph(h, cascade_expected(g, (2, 3))) and h.is_well_formed()


def test_phantom_edge_is_cut_and_real_edges_are_completed():
    g = G(); r = run(g, "DEL[2,3] DEL[(2,3)->(9,9):dep#1] DEL[(2,3)->(2,0):dep#4]")
    assert r.repairable and [c[0] for c in r.cut] == ["no such edge in the graph"]
    h = copy_graph(g); apply(h, r.repaired); assert same_graph(h, cascade_expected(g, (2, 3)))
    assert "DEL[(0,0)->(2,3):dep#7]" in r.completed


def test_wrong_weight_is_grounded_to_the_real_one():
    g = G(); r = run(g, "DEL[2,3] DEL[(2,3)->(2,0):dep#9] DEL[(2,3)->(2,1):dep#6] DEL[(0,0)->(2,3):dep#7]")
    assert r.repairable and len(r.grounded) == 1 and r.grounded[0][2] == "DEL[(2,3)->(2,0):dep#4]" and r.cut == []
    h = copy_graph(g); apply(h, r.repaired); assert same_graph(h, cascade_expected(g, (2, 3)))


def test_add_then_delete_of_an_existing_node_means_delete():
    g = G(); r = run(g, "ADD[2,3:9] DEL[2,3]")
    assert r.repairable and "net effect" in r.cut[0][0]
    h = copy_graph(g); apply(h, r.repaired); assert same_graph(h, cascade_expected(g, (2, 3)))


def test_admissible_delta_is_passed_through_untouched():                          # "if your ledger is clean she is not an executioner"
    g = G(); assert ground_and_complete(g, P("ADD[4,4:3]")) is None
    assert ground_and_complete(g, P("DEL[2,3] DEL[(2,3)->(2,0):dep#4] DEL[(2,3)->(2,1):dep#6] DEL[(0,0)->(2,3):dep#7]")) is None


def test_what_cannot_be_repaired_without_guessing_is_left_alone():
    g = G()
    r = run(g, "ADD[2,3:9]"); assert not r.repairable and "ident_node" in r.reason           # an update: not expressible, not guessed
    r = run(g, "DEL[(9,9)->(8,8):dep#1]"); assert not r.repairable and "nothing true" in r.reason     # only phantoms
    r = run(g, "DEL[7,7]"); assert not r.repairable


def test_high_degree_node_follows_the_dual_policy_like_propose_repair():
    g = Graph()
    for i in range(7): g.add_node((0, i), i)
    for i in range(1, 7): g.add_edge((0, 0), (0, i), "dep", i)
    r = run(g, "DEL[0,0] DEL[(0,0)->(5,5):dep#1]", degree_threshold=4)
    assert r.repairable and r.strategies == {(0, 0): "dual"} and "DEL[0,0]" not in r.text


def _mangle(rng, g, ideal):
    items = ideal.split(" "); out = []
    for it in items:
        roll = rng.random()
        if it.startswith("DEL[(") and roll < 0.25: out.append(it.replace("DEL[", "ADD["))                    # wrong verb
        elif it.startswith("DEL[(") and roll < 0.45: continue                                                 # forgotten
        elif it.startswith("DEL[(") and roll < 0.6: out.append(it.rsplit("#", 1)[0] + "#" + str((int(it.rsplit("#", 1)[1][:-1]) % 9) + 1) + "]")   # wrong weight
        else: out.append(it)
        if rng.random() < 0.15: out.append("DEL[(%d,%d)->(%d,%d):dep#%d]" % (rng.randint(5, 8), rng.randint(5, 8), rng.randint(5, 8), rng.randint(5, 8), rng.randint(1, 9)))   # phantom
    return " ".join(out)


def test_fuzz_soundness_minimality_idempotence_and_recovery():
    rng = random.Random(7); n_rep = n_unrep = n_fix = 0
    for t in make_tasks(300, 11):
        if t.kind != "delete_dep": continue
        for _ in range(6):
            text = _mangle(rng, t.graph, t.ideal)
            try: d = P(text)
            except Exception: continue
            r = ground_and_complete(t.graph, d)
            if r is None: assert check(t.graph, d).ok; continue                                              # admissible stays untouched
            if not r.repairable: n_unrep += 1; continue
            n_rep += 1
            assert check(t.graph, r.repaired).ok                                                              # soundness
            assert {(x.op, x.r, x.c, x.val) for x in r.repaired.node_deltas} <= {(x.op, x.r, x.c, x.val) for x in d.node_deltas}   # never invents a node op
            adds_in = {(x.u, x.v, x.relation, x.weight) for x in d.edge_deltas if x.op == "ADD"}
            assert {(x.u, x.v, x.relation, x.weight) for x in r.repaired.edge_deltas if x.op == "ADD"} <= adds_in              # never invents an edge add
            assert all(x.op == "ADD" or (x.u, x.v, x.relation) in t.graph.edges and t.graph.edges[(x.u, x.v, x.relation)] == x.weight for x in r.repaired.edge_deltas)   # every DEL is a real edge
            assert ground_and_complete(t.graph, r.repaired) is None                                           # idempotent: the result is admissible
            h = copy_graph(t.graph); apply(h, r.repaired); assert h.is_well_formed()
            if same_graph(h, t.expected): n_fix += 1
    assert n_rep > 200 and n_fix > 0.9 * n_rep, (n_rep, n_fix, n_unrep)                                       # when it repairs, the intended change is what comes out


def test_it_repairs_strictly_more_than_propose_repair():
    rng = random.Random(3); more = 0; fewer = 0
    for t in make_tasks(200, 5):
        if t.kind != "delete_dep": continue
        d = P(_mangle(rng, t.graph, t.ideal)) if True else None
        a = propose_repair(t.graph, d); b = ground_and_complete(t.graph, d)
        ra = a is not None and a.repairable; rb = b is not None and b.repairable
        if rb and not ra: more += 1
        if ra and not rb: fewer += 1
    assert more > 0 and fewer == 0, (more, fewer)


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("PASS", n)
