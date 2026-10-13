"""Implements concept-map item 91 (a third pasted audit; rows 3, 9, 27): what the repair does with a service that has MANY connections. Found while checking the audit's "deep graphs" claim.

The repair (`diagnostics.propose_repair`, used by `ground_and_complete`, so by repair_m6 and scope_m7) has two strategies for a service whose DEL leaves dangling edges:
  primal : add a DEL for every remaining incident edge, so the service goes together with its edges            (up to `degree_threshold` = 4 remaining edges)
  dual   : drop the DEL of the service, so the service STAYS with all its edges                                 (more than 4 remaining edges)
The dual choice is documented in the module header and is conservative: nothing is corrupted. But it means that asked to remove a hub, the held state keeps the hub.
Every task of every test so far deletes services with at most 3 connections (checked below), so the dual branch has never been exercised by any model run or any reading rule.

What this file pins, without any model:
  1. the exact boundary: 1 to 4 remaining edges are primal and the held state equals the intended state; 5 or more are dual and the service is kept (well formed, not corrupted, not what was asked);
  2. the count is of the edges the reply LEFT dangling: a reply that deletes 3 of 6 edges itself leaves 3, so the repair is primal;
  3. the same holds through the scope path (`admit_scoped`), where the request names the service, and `degree_threshold` lifts it: with a large threshold the intended state is reached at every size;
  4. the gate refuses the bare delete at every size (so a blind apply would leave dangling edges);
  5. the generators of the single requests (seeds 7 and 21) and of the 30 long episodes (seed 21) delete only services with at most 3 connections.
What it does NOT show: that lifting the threshold is the right default (it is a policy choice: a named delete of a hub is authorised by the request, an unnamed one is not), or anything about a model's behaviour on dense graphs."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from collections import Counter

from axi.engine.cico_parser import CICOParser
from axi.engine.diagnostics import ground_and_complete, propose_repair
from axi.engine.gate import Graph, apply, check
from axi.engine.scope import admit_scoped
from axi.experiments.long_horizon import oracle_script
from axi.experiments.scope_tasks import make_scope_tasks

T = CICOParser.parse_transition_delta
HUB = (0, 0)
BIG = 10 ** 6


def hub_graph(k):
    """A service at (0,0) with k connections (alternating out and in), k other services, and one unrelated pair that must never change."""
    g = Graph(); g.add_node(HUB, 1)
    for i in range(k):
        g.add_node((1, i), 1)
        if i % 2 == 0: g.add_edge(HUB, (1, i), "dep", 1 + i % 4)
        else: g.add_edge((1, i), HUB, "owns", 1 + i % 4)
    g.add_node((2, 0), 1); g.add_node((2, 1), 1); g.add_edge((2, 0), (2, 1), "dep", 3)
    return g


def clone(g):
    h = Graph()
    for n, v in g.nodes.items(): h.add_node(n, v)
    for (u, v, r), w in g.edges.items(): h.add_edge(u, v, r, w)
    return h


def intended(g, n=HUB):
    h = clone(g)
    for e in list(h.incident(n)): h.del_edge(e)
    h.del_node(n); return h


same = lambda a, b: a.nodes == b.nodes and a.edges == b.edges


def test_the_gate_refuses_the_bare_delete_of_a_connected_service_at_every_size():
    for k in range(1, 11):
        g = hub_graph(k); assert len(g.incident(HUB)) == k and not check(g, T("DEL[0,0]")).ok, k
    g = Graph(); g.add_node(HUB, 1); assert check(g, T("DEL[0,0]")).ok                          # no connections: nothing to repair, the bare delete is fine


def test_up_to_four_remaining_edges_the_service_goes_with_them_and_from_five_it_is_kept():
    for k in range(1, 11):
        g = hub_graph(k); h = clone(g); r = ground_and_complete(h, T("DEL[0,0]"))
        assert r is not None and r.repairable and check(g, r.repaired).ok
        apply(h, r.repaired); assert h.is_well_formed()                                       # never corrupted
        if k <= 4:
            assert r.strategies == {HUB: "primal"} and same(h, intended(g)), k
        else:
            assert r.strategies == {HUB: "dual"} and same(h, g) and HUB in h.nodes and not same(h, intended(g)), k         # kept, as documented
        p = propose_repair(g, T("DEL[0,0]")); assert p.strategies == r.strategies
    # the boundary is exactly 4 and 5
    assert ground_and_complete(hub_graph(4), T("DEL[0,0]")).strategies == {HUB: "primal"} and ground_and_complete(hub_graph(5), T("DEL[0,0]")).strategies == {HUB: "dual"}


def test_the_threshold_decides_it_and_a_large_one_reaches_the_intended_state_at_every_size():
    for k in range(1, 11):
        g = hub_graph(k)
        for thr, want_primal in ((BIG, True), (k, True), (k - 1, False)):
            h = clone(g); r = ground_and_complete(h, T("DEL[0,0]"), thr); apply(h, r.repaired)
            assert (r.strategies == {HUB: "primal"}) is want_primal and same(h, intended(g)) is want_primal, (k, thr)


def test_the_count_is_of_the_edges_the_reply_left_dangling():
    g = hub_graph(6); mine = sorted(g.edges)[:3]
    text = "DEL[0,0] " + " ".join(f"DEL[({u[0]},{u[1]})->({v[0]},{v[1]}):{rel}#{g.edges[(u, v, rel)]}]" for (u, v, rel) in mine)
    h = clone(g); r = ground_and_complete(h, T(text)); assert r is not None and r.strategies == {HUB: "primal"}                 # 3 left over
    apply(h, r.repaired); assert same(h, intended(g))
    g8 = hub_graph(8); mine8 = sorted(g8.edges)[:2]                                                                             # 6 left over: kept
    text8 = "DEL[0,0] " + " ".join(f"DEL[({u[0]},{u[1]})->({v[0]},{v[1]}):{rel}#{g8.edges[(u, v, rel)]}]" for (u, v, rel) in mine8)
    h8 = clone(g8); r8 = ground_and_complete(h8, T(text8)); assert r8.strategies == {HUB: "dual"}; apply(h8, r8.repaired); assert HUB in h8.nodes and h8.is_well_formed()


def test_through_the_scope_path_the_named_service_is_kept_above_four_unless_the_threshold_is_lifted():
    ask = "remove the service at (0,0)"
    for k in range(1, 11):
        g = hub_graph(k)
        r = admit_scoped(g, T("DEL[0,0]"), ask); h = clone(g); apply(h, r.final_delta)
        assert r.repaired and (HUB in h.nodes) is (k > 4) and same(h, intended(g)) is (k <= 4) and h.is_well_formed(), k
        r2 = admit_scoped(g, T("DEL[0,0]"), ask, degree_threshold=BIG); h2 = clone(g); apply(h2, r2.final_delta); assert same(h2, intended(g)), k
    # an unasked edge delete is still dropped, with the threshold lifted too
    g = hub_graph(7); stray = "DEL[(2,0)->(2,1):dep#3]"
    r = admit_scoped(g, T("DEL[0,0] " + stray), ask, degree_threshold=BIG); h = clone(g); apply(h, r.final_delta)
    assert same(h, intended(g)) and ((2, 0), (2, 1), "dep") in h.edges and any("does not name" in why for why, _ in r.dropped)


def test_two_named_services_that_share_edges_are_deleted_together_with_each_edge_once():
    """The fourth pasted text claimed a shared edge is deleted twice and that the scope filter tracks one target. Neither is so: edges are collected in a dict keyed by (from, to, relation)."""
    g = Graph()
    for n in [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 2)]: g.add_node(n, 1)
    g.add_edge((0, 0), (0, 1), "dep", 1); g.add_edge((0, 1), (0, 0), "owns", 2)                      # shared by the two targets, both directions
    g.add_edge((0, 0), (1, 0), "dep", 3); g.add_edge((0, 1), (0, 2), "dep", 4); g.add_edge((1, 1), (0, 1), "owns", 5)
    g.add_edge((2, 2), (1, 1), "dep", 6)                                                              # unrelated: must survive
    want = clone(g)
    for k in list(want.edges):
        if (0, 0) in k[:2] or (0, 1) in k[:2]: want.del_edge(k)
    want.del_node((0, 0)); want.del_node((0, 1))
    ask = "remove the services at (0,0) and (0,1)"
    r = admit_scoped(g, T("DEL[0,0] DEL[0,1] ADD[5,5:1]"), ask); h = clone(g)                         # forgets every edge, and adds a service nobody asked for
    assert r.final_delta is not None and r.repaired; apply(h, r.final_delta)
    keys = [(e.u, e.v, e.relation) for e in r.final_delta.edge_deltas]
    assert same(h, want) and len(keys) == len(set(keys)) == 5 and [d for _, d in r.dropped] == ["ADD[5,5:1]"] and ((2, 2), (1, 1), "dep") in h.edges
    r2 = ground_and_complete(clone(g), T("DEL[0,0] DEL[0,1]")); h2 = clone(g); apply(h2, r2.repaired); assert same(h2, want)         # also without the scope filter
    # a model that deletes one edge of the pair itself leaves the other to the repair, still once each
    one = "DEL[0,0] DEL[0,1] DEL[(0,0)->(0,1):dep#1]"; r3 = ground_and_complete(clone(g), T(one)); h3 = clone(g); apply(h3, r3.repaired)
    ks = [(e.u, e.v, e.relation) for e in r3.repaired.edge_deltas]; assert same(h3, want) and len(ks) == len(set(ks)) == 5


def test_no_task_of_any_test_so_far_deletes_a_service_with_more_than_three_connections():
    seen = Counter()
    for seed in (7, 21):
        for t in make_scope_tasks(280, seed):
            for nd in T(t.ideal).node_deltas:
                if nd.op == "DEL": seen[len(t.graph.incident((nd.r, nd.c)))] += 1
    for e in range(30):
        g, script, _ = oracle_script(21000 + e, 30, "scope")
        for st in script:
            d = T(st["ideal"])
            for nd in d.node_deltas:
                if nd.op == "DEL": seen[len(g.incident((nd.r, nd.c)))] += 1
            apply(g, d)
    assert seen and max(seen) == 3 and sum(seen.values()) > 500 and set(seen) == {0, 1, 2, 3}, seen              # the dual branch has never been exercised


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn): fn(); print("PASS", name); n += 1
    print(n, "tests passed")
