import random
from axi.engine.cico_parser import CICOParser
from axi.engine.gate import Graph, check
from axi.engine.diagnostics import diagnose, propose_repair, format_delta
from axi.engine.runtime import Runtime


def rand_graph(rng, n=25, p=0.12):
    g = Graph(); nodes = [(i, 0) for i in range(n)]
    for x in nodes: g.add_node(x, 1)
    for a in nodes:
        for b in nodes:
            if a != b and rng.random() < p: g.add_edge(a, b, rng.choice("xyz"), rng.randrange(3))
    return g


def rand_delta_text(rng, g):
    ops = []
    for _ in range(rng.randint(1, 3)):
        k = rng.random(); ns = list(g.nodes)
        if k < .5:
            n = rng.choice(ns); ops.append(f"DEL[{n[0]},{n[1]}]")
            for (u, v, r) in list(g.incident(n)):
                if rng.random() < .4: ops.append(f"DEL[({u[0]},{u[1]})->({v[0]},{v[1]}):{r}#{g.edges[(u, v, r)]}]")
        elif k < .6: ops.append(f"DEL[{rng.randrange(40)},{rng.randrange(2)}]")
        elif k < .8: ops.append(f"ADD[{rng.randrange(40)},{rng.randrange(2)}:1]")
        else:
            a, b = rng.choice(ns), (rng.randrange(40), rng.randrange(2)); ops.append(f"ADD[({a[0]},{a[1]})->({b[0]},{b[1]}):x#1]")
    return " ".join(ops)


def test_format_roundtrip():
    s = "ADD[1,2:-3] DEL[4,5] ADD[(1,2)->(3,4):owns#5] DEL[(0,0)->(1,1):dep#-2]"
    d = CICOParser.parse_transition_delta(s); assert format_delta(d) == s


def test_diagnose_agrees_with_gate_fuzz():
    rng = random.Random(5); ok = bad = 0
    for _ in range(4000):
        g = rand_graph(rng, 12, .2); t = rand_delta_text(rng, g); d = CICOParser.parse_transition_delta(t)
        a = not diagnose(g, d); b = check(g, d).ok
        assert a == b, (t, diagnose(g, d), check(g, d)); ok += b; bad += (not b)
    assert ok > 300 and bad > 300


def test_dangling_obstruction_lists_exactly_the_attached_edges():
    g = Graph(); [g.add_node((i, 0)) for i in range(3)]; g.add_edge((0, 0), (1, 0), "dep", 5); g.add_edge((2, 0), (1, 0), "dep", 7)
    o = diagnose(g, CICOParser.parse_transition_delta("DEL[1,0]"))
    assert len(o) == 1 and o[0].kind == "dangling" and len(o[0].edges) == 2
    o = diagnose(g, CICOParser.parse_transition_delta("DEL[1,0] DEL[(0,0)->(1,0):dep#5]"))
    assert len(o[0].edges) == 1


def test_primal_repair_is_sound_and_minimal():
    g = Graph(); [g.add_node((i, 0)) for i in range(3)]; g.add_edge((0, 0), (1, 0), "dep", 5); g.add_edge((2, 0), (1, 0), "dep", 7)
    r = propose_repair(g, CICOParser.parse_transition_delta("DEL[1,0]"))
    assert r.repairable and r.strategies == {(1, 0): "primal"}
    assert r.added_ops == "DEL[((0,0))->((1,0)):dep#5]".replace("((", "(").replace("))", ")") + " " + "DEL[(2,0)->(1,0):dep#7]"
    assert check(g, r.repaired).ok and check(g, CICOParser.parse_transition_delta(r.text)).ok


def test_two_adjacent_deleted_nodes_share_one_edge_delete():
    g = Graph(); [g.add_node((i, 0)) for i in range(2)]; g.add_edge((0, 0), (1, 0), "dep", 1)
    r = propose_repair(g, CICOParser.parse_transition_delta("DEL[0,0] DEL[1,0]"))
    assert r.repairable and r.text.count("DEL[(0,0)->(1,0)") == 1 and check(g, r.repaired).ok   # pasted code would list the edge twice


def test_dual_repair_keeps_hub():
    g = Graph(); hub = (0, 0); g.add_node(hub)
    for i in range(1, 9): g.add_node((i, 0)); g.add_edge((i, 0), hub, "dep", 1)
    r = propose_repair(g, CICOParser.parse_transition_delta("DEL[0,0] ADD[99,0:1]"), degree_threshold=4)
    assert r.strategies == {hub: "dual"} and "DEL[0,0]" not in r.text and "ADD[99,0:1]" in r.text and check(g, r.repaired).ok


def test_unrepairable_is_reported_not_guessed():
    g = rand_graph(random.Random(1), 6, .3)
    r = propose_repair(g, CICOParser.parse_transition_delta("DEL[77,0]"))
    assert r is not None and not r.repairable and r.repaired is None and r.obstructions[0].kind == "match_node"
    assert propose_repair(g, CICOParser.parse_transition_delta("ADD[50,0:1]")) is None   # valid -> nothing to repair


def test_repair_soundness_fuzz():
    rng = random.Random(9); n_rep = 0
    for _ in range(4000):
        g = rand_graph(rng, 15, .2); d = CICOParser.parse_transition_delta(rand_delta_text(rng, g))
        r = propose_repair(g, d, degree_threshold=rng.choice([0, 2, 4]))
        if r is None: assert check(g, d).ok; continue
        assert not check(g, d).ok
        if r.repairable:
            n_rep += 1; assert check(g, r.repaired).ok, (format_delta(d), r.text)
            assert check(g, CICOParser.parse_transition_delta(r.text)).ok        # survives text round trip
    assert n_rep > 200


def test_closed_loop_through_runtime():
    rng = random.Random(3); rt = Runtime(400); healed = 0
    for i in range(8):
        rt.propose(f"ADD[{i},0:1]")
    for i in range(8):
        for j in range(8):
            if i != j and rng.random() < .35: rt.propose(f"ADD[({i},0)->({j},0):dep#1]")
    for _ in range(300):
        n = rng.randrange(8)
        if (n, 0) not in rt.graph.nodes: continue
        o = rt.propose(f"DEL[{n},0]")
        if o.committed: rt.propose(f"ADD[{n},0:1]"); continue
        assert o.stage == "gate" and o.repair is not None, o
        o2 = rt.propose(o.repair); assert o2.committed, (o, o2); healed += 1
        rt.propose(f"ADD[{n},0:1]")
        rt.pager.audit(); assert rt.graph.is_well_formed()
        for i in range(8):                       # re-grow edges so later rounds are non-trivial
            for j in range(8):
                if i != j and (i, 0) in rt.graph.nodes and (j, 0) in rt.graph.nodes and rng.random() < .15:
                    rt.propose(f"ADD[({i},0)->({j},0):dep#1]")
    assert healed > 20, healed


def test_redundant_add_is_dropped_and_dangling_still_repaired():
    g = Graph(); [g.add_node((i, 0), 9) for i in range(3)]; g.add_edge((0, 0), (1, 0), "dep", 5)
    # the 3B model's real failure shape: re-ADD of an existing node (same value) plus a node delete with dependents
    r = propose_repair(g, CICOParser.parse_transition_delta("ADD[1,0:9] DEL[1,0]"))
    assert r.repairable and "ADD" not in r.text and "DEL[(0,0)->(1,0):dep#5]" in r.text and len(r.dropped) == 1
    assert check(g, r.repaired).ok


def test_ambiguous_add_is_not_silently_dropped():
    g = Graph(); [g.add_node((i, 0), 9) for i in range(2)]
    r = propose_repair(g, CICOParser.parse_transition_delta("ADD[1,0:3]"))      # different value: could be an update intent
    assert r is not None and not r.repairable and r.dropped == []


def test_duplicate_items_and_idempotent_edge_add_dropped_semantics_preserved():
    rng = random.Random(4); n_fixed = 0
    for _ in range(1500):
        g = rand_graph(rng, 12, .25); n = rng.choice(list(g.nodes))
        ideal = [f"DEL[{n[0]},{n[1]}]"] + [f"DEL[({u[0]},{u[1]})->({v[0]},{v[1]}):{r}#{g.edges[(u, v, r)]}]" for (u, v, r) in sorted(g.incident(n))]
        ref = Graph(); [ref.add_node(k, v) for k, v in g.nodes.items()]; [ref.add_edge(*k, w) for k, w in g.edges.items()]
        from axi.engine.gate import apply
        apply(ref, CICOParser.parse_transition_delta(" ".join(ideal)))
        noisy = list(ideal[:1]) + ["ADD[%d,%d:%d]" % (*n, g.nodes[n])] + ideal[1:] + ideal[1:2]        # redundant ADD + duplicated item
        d = CICOParser.parse_transition_delta(" ".join(noisy)); r = propose_repair(g, d)
        if r is None: continue
        assert r.repairable, noisy
        out = Graph(); [out.add_node(k, v) for k, v in g.nodes.items()]; [out.add_edge(*k, w) for k, w in g.edges.items()]
        apply(out, r.repaired); assert out.nodes == ref.nodes and out.edges == ref.edges; n_fixed += 1
    assert n_fixed > 800


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("PASS", n)
