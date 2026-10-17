"""Engineering support for docs/DYNAMIC_WORLD_TEST.md (concept-map item 102): what the host layer does when the proposer is not a language model and the world changes under it. CPU only, no model.
This file tests the INSTRUMENT, not the finding:
  1. the request and what it means: `req_from_task` for every kind, `intent_delta` equals the picker's ideal reply on the state the request was drawn on, and does the sensible thing when the state has changed
     (a delete takes every edge the service has NOW; an item that is gone is not touched; an add of something already there does nothing);
  2. the change of the world: each kind on a hand-made graph, a change that does not fit a state is ignored, a collapsing service takes its edges, a corrupted state does not crash it; the draw is seeded, the
     `near` regime touches what the request names and the `uniform` regime does not, the caps are kept;
  3. routes: a shortest path, the edges of a corrupted state are followed, a route over a missing edge or service is unsafe;
  4. the network: the picture, the labels, the decode round trip, save and load, and that training lowers the loss;
  5. the four arms on six hand-made stale proposals, each against the outcome the page says (blind, gate, retry, repair; calls; slip);
  6. the episode runner: the control with no change of the world, determinism, independence of the stream from the arms and proposers, and the bookkeeping (every step counted once, per event and per kind);
  7. deliberately broken pieces are noticed (an arm that ignores the gate, a world whose collapse leaves edges behind, a slip test that is blind to a refused step);
  8. the figures and the fifteen reading rules, each flipping where it should; the quick run, the printout and the command line;
  9. the page is locked (hash of the text above the Result), carries the fifteen lines with the probabilities, the settings and the thresholds the code uses."""
import contextlib, copy, hashlib, io, json, os, random, re, sys, tempfile
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import axi.experiments.dynworld as D
from axi.engine.cico_parser import CICOParser
from axi.engine.gate import Graph, apply, check
from axi.experiments.agent_loop import copy_graph, same_graph

HERE = os.path.dirname(__file__)
PAGE = os.path.join(HERE, "..", "..", "docs", "DYNAMIC_WORLD_TEST.md")
RESULT_MARK = "\n\n---\n\n# Result"
P = CICOParser.parse_transition_delta
A, B, C, E_, F_ = (0, 0), (0, 1), (0, 2), (1, 1), (2, 0)
Dn = (1, 0)


def base_graph():
    """Five services A..E and five edges: A->B (3), B->C (4), A->D (2), D->E (5), C->E (6)."""
    g = Graph()
    for k, n in enumerate([A, B, C, Dn, E_], 1): g.add_node(n, k)
    for u, v, w in [(A, B, 3), (B, C, 4), (A, Dn, 2), (Dn, E_, 5), (C, E_, 6)]: g.add_edge(u, v, D.REL, w)
    return g


def items(d):
    return ({(x.op, x.r, x.c, x.val) for x in d.node_deltas}, {(x.op, x.u, x.v, x.relation, x.weight) for x in d.edge_deltas})


# ---------------------------------------------------------------- 1 the request and what it means
def test_every_kind_of_request_is_read_and_means_what_the_picker_said_on_the_state_it_was_drawn_on():
    rng = random.Random(3); g = D.start_graph(rng); kinds = set()
    for t in range(500):
        out = D.draw_request(rng, g); assert out is not None; req, ideal = out; kinds.add(req.task_kind)
        assert req.kind in D.KIND5 and req.task_kind in {"add_node", "add_edge", "delete_dep", "delete_leaf", "delete_hub", "delete_edge", "delete_pair"}
        d = D.intent_delta(g, req); assert check(g, d).ok and items(d) == items(P(ideal)), (req, ideal)
        apply(g, d); assert g.is_well_formed() and len(g.nodes) <= D.CAP_NODES and len(g.edges) <= D.CAP_EDGES; last = ideal
    assert kinds == {"add_node", "add_edge", "delete_dep", "delete_leaf", "delete_hub", "delete_edge", "delete_pair"}
    assert all(r == D.REL for (_, _, r) in g.edges)
    try: D.req_from_task(type("T", (), dict(kind="delete_weight", ideal=last))()); raise AssertionError("an unknown kind must not be accepted")
    except ValueError: pass


def test_intent_follows_the_state_the_host_holds_now():
    g = base_graph(); R = D.Req
    d = D.intent_delta(g, R("delete_service", "delete_dep", node=B)); assert items(d) == ({("DEL", 0, 1, None)}, {("DEL", A, B, "dep", 3), ("DEL", B, C, "dep", 4)})
    g.add_edge(E_, B, D.REL, 7); d = D.intent_delta(g, R("delete_service", "delete_dep", node=B)); assert len(d.edge_deltas) == 3                      # an edge that appeared since is taken too
    g.del_edge((B, C, D.REL)); d = D.intent_delta(g, R("delete_service", "delete_dep", node=B)); assert {(e.u, e.v) for e in d.edge_deltas} == {(A, B), (E_, B)}      # an edge that is gone is not asked for
    h = copy_graph(g); apply_e = D.apply_event; apply_e(h, ("drop_node", B)); assert not D.intent_delta(h, R("delete_service", "delete_dep", node=B)).node_deltas and not D.intent_delta(h, R("delete_service", "delete_dep", node=B)).edge_deltas
    assert not items(D.intent_delta(g, R("add_node", "add_node", node=A, value=9)))[0]                                                                    # already there: nothing
    d = D.intent_delta(g, R("add_edge", "add_edge", edge=(A, C, D.REL), weight=1)); assert items(d) == (set(), {("ADD", A, C, "dep", 1)})
    assert not D.intent_delta(g, R("add_edge", "add_edge", edge=(A, B, D.REL), weight=9)).edge_deltas                                                      # already there (any weight): nothing
    g2 = copy_graph(g); D.apply_event(g2, ("drop_node", C)); assert not D.intent_delta(g2, R("add_edge", "add_edge", edge=(A, C, D.REL), weight=1)).edge_deltas   # an end is gone: nothing
    assert not D.intent_delta(g, R("delete_edge", "delete_edge", edge=(A, C, D.REL), weight=1)).edge_deltas                                                # not there: nothing
    d = D.intent_delta(g, R("delete_pair", "delete_pair", node=Dn, edge=(C, E_, D.REL), weight=6)); assert {(e.u, e.v) for e in d.edge_deltas} == {(A, Dn), (Dn, E_), (C, E_)} and len(d.node_deltas) == 1
    g.del_edge((C, E_, D.REL)); d = D.intent_delta(g, R("delete_pair", "delete_pair", node=Dn, edge=(C, E_, D.REL), weight=6)); assert {(e.u, e.v) for e in d.edge_deltas} == {(A, Dn), (Dn, E_)}   # the named edge is gone: the rest is done
    assert D.intent_delta(g, R("delete_pair", "delete_pair", node=Dn, edge=(C, E_, D.REL), weight=6)).node_deltas[0].op == "DEL"


# ---------------------------------------------------------------- 2 the change of the world
def test_each_change_of_the_world_on_a_hand_made_graph_and_a_change_that_does_not_fit_is_ignored():
    g = base_graph(); D.apply_event(g, ("drop_edge", (A, B, D.REL))); assert (A, B, D.REL) not in g.edges and len(g.edges) == 4
    D.apply_event(g, ("drop_edge", (A, B, D.REL))); assert len(g.edges) == 4                                                                               # gone already: ignored
    D.apply_event(g, ("add_edge", (A, B, D.REL), 8)); assert g.edges[(A, B, D.REL)] == 8 and g.is_well_formed()
    D.apply_event(g, ("add_edge", (A, B, D.REL), 1)); assert g.edges[(A, B, D.REL)] == 8                                                                   # there already: ignored
    D.apply_event(g, ("add_edge", (A, F_, D.REL), 1)); assert (A, F_, D.REL) not in g.edges                                                                # an end is missing: ignored
    D.apply_event(g, ("add_node", F_, 5)); assert g.nodes[F_] == 5; D.apply_event(g, ("add_node", F_, 9)); assert g.nodes[F_] == 5
    D.apply_event(g, ("add_edge", (A, F_, D.REL), 1)); assert (A, F_, D.REL) in g.edges
    n_edges = len(g.edges); D.apply_event(g, ("drop_node", B)); assert B not in g.nodes and g.is_well_formed() and len(g.edges) == n_edges - 2                  # the service takes its two edges with it
    D.apply_event(g, ("drop_node", B)); assert B not in g.nodes; D.apply_event(g, None)
    try: D.apply_event(g, ("melt", A)); raise AssertionError
    except ValueError: pass
    bad = base_graph(); bad.edges[(A, F_, D.REL)] = 1; bad._inc[A].add((A, F_, D.REL)); bad._inc[F_].add((A, F_, D.REL))                                         # a corrupted state: an edge to a missing service
    D.apply_event(bad, ("drop_node", A)); assert A not in bad.nodes and (A, F_, D.REL) not in bad.edges
    D.apply_event(bad, ("drop_node", F_)); D.apply_event(bad, ("add_edge", (B, C, D.REL), 9))


def test_the_draw_of_a_change_is_seeded_keeps_the_caps_and_near_touches_what_the_request_names():
    g = base_graph(); req = D.Req("delete_service", "delete_dep", node=B)
    def many(regime, seed, n=600):
        r = random.Random(seed); return [D.draw_event(r, g, req, regime) for _ in range(n)]
    assert many("near", 1) == many("near", 1) and many("near", 1) != many("near", 2) and many("uniform", 1) != many("near", 1)
    touches = lambda ev: ev is not None and ((ev[0] == "drop_node" and ev[1] == B) or (ev[0] in ("drop_edge", "add_edge") and B in (ev[1][0], ev[1][1])))
    near = [e for e in many("near", 4) if e is not None and e[0] != "add_node"]; uni = [e for e in many("uniform", 4) if e is not None and e[0] != "add_node"]
    assert all(touches(e) for e in near) and len(near) > 300 and sum(map(touches, uni)) < 0.6 * len(uni)
    kinds = {e[0] for e in many("uniform", 5) if e}; assert kinds == {"drop_edge", "add_edge", "drop_node", "add_node"}
    share = {k: sum(1 for e in many("uniform", 6, 4000) if e and e[0] == k) / 4000 for k in kinds}; assert abs(share["drop_edge"] - 0.35) < 0.04 and abs(share["add_edge"] - 0.35) < 0.04 and abs(share["drop_node"] - 0.15) < 0.03 and abs(share["add_node"] - 0.15) < 0.03
    reqn = D.Req("add_node", "add_node", node=(3, 3), value=5); r = random.Random(8); evs = [D.draw_event(r, g, reqn, "near") for _ in range(400)]
    assert any(e and e[0] == "add_node" and e[1] == (3, 3) for e in evs)                                                                                    # near an add: the same service, appearing by itself
    full = Graph()
    for i in range(D.CAP_NODES): full.add_node((i // D.COLS, i % D.COLS), 1)
    r = random.Random(1); assert all(e is None or e[0] != "add_node" for e in (D.draw_event(r, full, req, "uniform") for _ in range(300)))               # no new service past the cap
    many_edges = Graph()
    for i in range(D.CAP_NODES): many_edges.add_node((i // D.COLS, i % D.COLS), 1)
    ns = sorted(many_edges.nodes); k = 0
    for u in ns:
        for v in ns:
            if u != v and k < D.CAP_EDGES: many_edges.add_edge(u, v, D.REL, 1); k += 1
    assert all(e is None or e[0] != "add_edge" for e in (D.draw_event(r, many_edges, req, "uniform") for _ in range(300)))
    empty = Graph(); assert all(e is None or e[0] == "add_node" for e in (D.draw_event(r, empty, req, "uniform") for _ in range(100)))


# ---------------------------------------------------------------- 3 routes
def test_routes_are_shortest_follow_a_corrupted_state_and_unsafe_ones_are_told_from_safe_ones():
    g = base_graph(); assert D.route(g, A, E_) in ([A, Dn, E_], [A, B, C, E_]) and len(D.route(g, A, E_)) == 3 and D.route(g, A, A) == [A] and D.route(g, A, F_) is None and D.route(g, F_, A) is None
    pairs = {frozenset((u, v)) for (u, v, _) in g.edges}; assert D.route_safe(pairs, g.nodes, [A, Dn, E_]) and not D.route_safe(pairs, g.nodes, [A, E_]) and not D.route_safe(pairs, g.nodes, [A, F_])
    bad = copy_graph(g)
    for k in [(A, B, D.REL), (A, Dn, D.REL)]: bad.del_edge(k)
    bad.edges[(A, F_, D.REL)] = 1; bad._inc[A].add((A, F_, D.REL)); bad._inc[F_].add((A, F_, D.REL))
    assert D.route(bad, A, E_) is None and D.route(bad, A, F_) is None                                                                                      # F_ is not a service of the state, so no route to it, and A is cut off
    bad.add_edge(A, Dn, D.REL, 2); assert D.route(bad, A, E_) == [A, Dn, E_]
    ghost = copy_graph(g); ghost.edges[(Dn, F_, D.REL)] = 1; ghost._inc[Dn].add((Dn, F_, D.REL)); ghost._inc[F_].add((Dn, F_, D.REL)); ghost.edges[(F_, E_, D.REL)] = 1; ghost._inc[F_].add((F_, E_, D.REL)); ghost._inc[E_].add((F_, E_, D.REL))
    for k in [(Dn, E_, D.REL), (C, E_, D.REL)]: ghost.del_edge(k)
    p = D.route(ghost, A, E_); assert p == [A, Dn, F_, E_] and not D.route_safe(pairs, g.nodes, p)                                                         # a route through a service that does not exist is followed, and is unsafe


# ---------------------------------------------------------------- 4 the network
def test_the_picture_the_labels_and_the_decode_round_trip():
    g = base_graph(); R = D.Req("delete_service", "delete_dep", node=B); x = D.encode(g, R); assert x.shape == (D.D_IN,) and D.D_IN == 1027 and x.dtype == np.float32
    assert x[:D.CELLS].sum() == 5 and x[D.O_ADJ:D.O_KIND].sum() == 5 and x[D.O_ADJ + D.cell(A) * D.CELLS + D.cell(B)] == 1 and x[D.O_KIND + 2] == 1 and x[D.O_RN + D.cell(B)] == 1 and x[D.O_RU:D.O_W].sum() == 0
    R2 = D.Req("add_edge", "add_edge", edge=(A, C, D.REL), weight=9); x2 = D.encode(g, R2); assert x2[D.O_KIND + 1] == 1 and x2[D.O_RU + D.cell(A)] == 1 and x2[D.O_RV + D.cell(C)] == 1 and abs(x2[D.O_W] - 1.0) < 1e-6 and x2[D.O_W + 1] == 0
    R3 = D.Req("add_node", "add_node", node=F_, value=3); x3 = D.encode(g, R3); assert abs(x3[D.O_W + 1] - 3 / 9) < 1e-6 and x3[D.O_RN + D.cell(F_)] == 1
    buf = np.ones(D.D_IN, np.float32); buf[:] = 0; assert (D.encode(g, R, buf) == x).all()
    for req in (R, R2, R3, D.Req("delete_edge", "delete_edge", edge=(A, B, D.REL), weight=3), D.Req("delete_pair", "delete_pair", node=Dn, edge=(C, E_, D.REL), weight=6)):
        y = D.labels(g, req); assert y.shape == (D.N_POS,) and set(np.unique(y)) <= {0, 1, 2}; assert items(D.decode(y, g, req)) == items(D.intent_delta(g, req)), req
    y = np.zeros(D.N_POS, np.int8); y[D.cell(F_)] = 1; y[D.CELLS + D.cell(A) * D.CELLS + D.cell(F_)] = 2                                                     # an invented delete of an edge the state has not: the weight falls back to the request's, or 1
    d = D.decode(y, g, D.Req("add_node", "add_node", node=F_, value=3)); assert items(d) == ({("ADD", 2, 0, 3)}, {("DEL", A, F_, "dep", 1)})


def test_the_network_saves_loads_replies_and_training_lowers_the_loss():
    net = D.Net.init(1); g = base_graph(); R = D.Req("delete_service", "delete_dep", node=B)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "n.npz"); net.save(path); n2 = D.Net.load(path)
        assert all((a == b).all() and a.dtype == np.float32 for a, b in zip((net.W1, net.b1, net.W2, net.b2), (n2.W1, n2.b1, n2.W2, n2.b2))) and items(net.reply(g, R)) == items(n2.reply(g, R))
    z = net.logits(D.encode(g, R)); assert z.shape == (D.N_POS, 3) and np.isfinite(z).all()
    t = D.train_net(dict(streams=4, steps=40, iterations=300, batch=32, hidden=32, lr=1e-2)); l = t.losses; assert [i for i, _ in l] == [100, 200, 300] and l[-1][1] < 0.8 * l[0][1] and t.W1.shape == (D.D_IN, 32)
    t2 = D.train_net(dict(streams=4, steps=40, iterations=100, batch=32, hidden=32)); t3 = D.train_net(dict(streams=4, steps=40, iterations=100, batch=32, hidden=32)); assert np.allclose(t2.W2, t3.W2, atol=1e-5)    # same seeds, same network
    X, XW, Y = D.make_dataset(3, 30, 123); assert X.shape == (90, D.D_IN - 2) and XW.shape == (90, 2) and Y.shape == (90, D.N_POS) and X.dtype == np.uint8
    X2, _, Y2 = D.make_dataset(3, 30, 123); assert (X == X2).all() and (Y == Y2).all()
    q = D.reply_quality(net, [1], steps=20); assert q["requests"] == 20 and q["exact"] + q["not_exact"] == 20


# ---------------------------------------------------------------- 5 the four arms on hand-made stale proposals
def stale_case(req, event, S=None):
    """The snapshot S, the world changes (`event`), the exact reader reads the snapshot; returns {arm: (new state, calls, slip)} and the state now."""
    S = S or base_graph(); H = copy_graph(S); D.apply_event(H, event); out = {}
    for arm in D.ARMS:
        after, calls = D.arm_step(arm, D.exact_reply, S, H, req); out[arm] = (after, calls, D.is_slip(after, H, req))
    return out, H


def ideal_now(H, req):
    from axi.experiments.agent_loop import blind_apply
    return blind_apply(H, D.intent_delta(H, req))


def test_stale_proposal_a_new_edge_appeared_at_the_service_to_delete():
    req = D.Req("delete_service", "delete_dep", node=B); out, H = stale_case(req, ("add_edge", (E_, B, D.REL), 7))
    assert not out["blind"][0].is_well_formed() and out["blind"][2] and out["blind"][1] == 1                                     # a dangling edge is left
    assert same_graph(out["gate"][0], H) and out["gate"][2] and out["gate"][1] == 1                                              # refused: the service stays
    assert same_graph(out["retry"][0], ideal_now(H, req)) and not out["retry"][2] and out["retry"][1] == 2                       # asked again on the current state
    assert same_graph(out["repair"][0], ideal_now(H, req)) and not out["repair"][2] and out["repair"][1] == 1 and B not in out["repair"][0].nodes and out["repair"][0].is_well_formed()


def test_stale_proposal_an_edge_the_reply_deletes_has_been_dropped():
    req = D.Req("delete_service", "delete_dep", node=B); out, H = stale_case(req, ("drop_edge", (B, C, D.REL)))
    assert same_graph(out["blind"][0], ideal_now(H, req)) and not out["blind"][2]                                                # popping what is not there does no harm
    assert same_graph(out["gate"][0], H) and out["gate"][2]                                                                      # the gate refuses the whole reply for one missing edge
    assert same_graph(out["retry"][0], ideal_now(H, req)) and out["retry"][1] == 2 and same_graph(out["repair"][0], ideal_now(H, req)) and out["repair"][1] == 1 and not out["repair"][2]


def test_stale_proposal_an_edge_is_added_to_a_service_that_has_collapsed():
    req = D.Req("add_edge", "add_edge", edge=(A, C, D.REL), weight=9); out, H = stale_case(req, ("drop_node", C))
    assert not out["blind"][0].is_well_formed() and out["blind"][2] and (A, C, D.REL) in out["blind"][0].edges                      # an edge to nothing
    for arm in ("gate", "retry", "repair"): assert same_graph(out[arm][0], H) and not out[arm][2], arm                          # the request is moot: nothing is the right answer
    assert out["gate"][1] == 1 and out["retry"][1] == 2 and out["repair"][1] == 1


def test_stale_proposal_the_edge_or_the_service_to_add_has_appeared_with_another_value():
    req = D.Req("add_edge", "add_edge", edge=(A, E_, D.REL), weight=9); out, H = stale_case(req, ("add_edge", (A, E_, D.REL), 4))
    assert out["blind"][0].edges[(A, E_, D.REL)] == 9 and out["blind"][2]                                                        # overwritten
    for arm in ("gate", "retry", "repair"): assert same_graph(out[arm][0], H) and not out[arm][2], arm
    req = D.Req("add_node", "add_node", node=F_, value=5); out, H = stale_case(req, ("add_node", F_, 8))
    assert out["blind"][0].nodes[F_] == 5 and out["blind"][2]
    for arm in ("gate", "retry", "repair"): assert out[arm][0].nodes[F_] == 8 and not out[arm][2], arm


def test_stale_proposal_a_service_and_an_edge_where_the_named_edge_has_been_dropped():
    req = D.Req("delete_pair", "delete_pair", node=Dn, edge=(C, E_, D.REL), weight=6); out, H = stale_case(req, ("drop_edge", (C, E_, D.REL)))
    assert same_graph(out["blind"][0], ideal_now(H, req)) and not out["blind"][2] and same_graph(out["gate"][0], H) and out["gate"][2]
    assert same_graph(out["retry"][0], ideal_now(H, req)) and same_graph(out["repair"][0], ideal_now(H, req)) and Dn not in out["repair"][0].nodes
    req = D.Req("delete_service", "delete_hub", node=A); out, H = stale_case(req, ("drop_node", A))                              # the service itself is gone: the request is moot
    assert out["blind"][2] is False and all(not out[a][2] for a in D.ARMS) and all(same_graph(out[a][0], H) for a in D.ARMS)


def test_without_a_change_of_the_world_every_arm_does_what_the_exact_reader_says_and_the_retry_never_asks_twice():
    rng = random.Random(11); g = D.start_graph(rng)
    for t in range(200):
        req, _ = D.draw_request(rng, g)
        for arm in D.ARMS:
            after, calls = D.arm_step(arm, D.exact_reply, g, g, req); assert calls == 1 and same_graph(after, ideal_now(g, req)) and not D.is_slip(after, g, req), (arm, req)
        apply(g, D.intent_delta(g, req))


# ---------------------------------------------------------------- 6 the episode runner
def small(proposers=("exact",), **kw):
    kw = dict(dict(episodes=2, steps=60, noise=(0.0, 0.4), regimes=("uniform", "near")), **kw); return D.run_dynworld(D.Net.init(2), seed=kw.pop("seed", 5), proposers=proposers, **kw)


def test_the_control_with_no_change_of_the_world_is_exact_for_every_arm_and_the_world_is_seeded():
    res = small(); assert set(res["cells"]) == {"uniform/0.0", "uniform/0.4", "near/0.0", "near/0.4"}
    for key in ("uniform/0.0", "near/0.0"):
        c = res["cells"][key]; assert c["world"]["events"] == 0 and c["world"]["steps"] == 120 and c["world"]["ended"] == 0
        for arm, st in c["proposers"]["exact"].items(): assert st["slip"] == 0 and st["exact"] == 120 and st["corrupted"] == 0 and st["dist_sum"] == 0 and st["calls"] == 120, arm
    c = res["cells"]["near/0.4"]; assert 20 < c["world"]["events"] < 80 and sum(c["world"]["event_kinds"].values()) == c["world"]["events"] and c["world"]["oracle_corrupted"] == 0
    assert D.verdict_dynworld(res)[2] == ("instrument", True) and res["digest"] == small()["digest"] and res["digest"] != small(seed=6)["digest"]


def test_the_stream_of_requests_and_changes_does_not_depend_on_the_arms_or_the_proposers():
    a = small(("exact",)); b = small(("exact", "net")); assert a["digest"] == b["digest"]
    for k in a["cells"]: assert a["cells"][k]["world"] == b["cells"][k]["world"] and a["cells"][k]["proposers"]["exact"] == b["cells"][k]["proposers"]["exact"]
    c = D.run_dynworld(D.Net.init(9), seed=5, episodes=2, steps=60, noise=(0.0, 0.4), proposers=("net", "exact")); assert c["digest"] == a["digest"] and c["proposers"] == ["net", "exact"]


def test_the_bookkeeping_adds_up():
    res = small(("exact", "net"), noise=(0.4,))
    for key, c in res["cells"].items():
        n = c["world"]["steps"]; assert n == 120 and sum(c["world"]["requests"].values()) == n
        for p, arms in c["proposers"].items():
            for arm, st in arms.items():
                assert sum(v[0] for v in st["by_event"].values()) == n and sum(v[0] for v in st["by_kind"].values()) == n and sum(v[1] for v in st["by_event"].values()) == st["slip"] == sum(v[1] for v in st["by_kind"].values())
                assert sum(st["q_steps"]) == n and sum(st["q_dist"]) == st["dist_sum"] and sum(st["q_exact"]) == st["exact"] and st["fresh"] <= st["start_exact"] <= n and 0 <= st["slip"] <= n and st["exact"] <= n
                assert st["route_unsafe"] <= st["route_planned"] <= n and st["route_missed"] <= st["route_possible"] <= n and st["dist_max"] <= st["dist_sum"]
                assert (st["calls"] == n) if arm != "retry" else (n <= st["calls"] <= 2 * n)
                if arm != "blind": assert st["corrupted"] == 0
        assert set(c["world"]["event_kinds"]) <= {"drop_edge", "add_edge", "drop_node", "add_node"} and set(c["world"]["requests"]) <= set(D.KIND5)
    json.dumps(res)


def test_the_figures_the_page_uses_are_computed_from_the_tallies():
    res = small(("exact",), noise=(0.4,), episodes=1, steps=40); c = res["cells"]["near/0.4"]; f = D.arm_figures(c, "exact", "retry"); st = c["proposers"]["exact"]["retry"]
    assert f["steps"] == 40 and f["slip"] == st["slip"] / 40 and f["calls"] == st["calls"] / 40 and len(f["quarter_distance"]) == 4 and f["quarter_distance"][0] == st["q_dist"][0] / st["q_steps"][0] and f["mean_distance"] == st["dist_sum"] / 40
    assert f["unsafe"] == (st["route_unsafe"] / st["route_planned"] if st["route_planned"] else 0.0) and D.pooled_slip(res, "exact", "blind", 0.4) == (c["proposers"]["exact"]["blind"]["slip"] + res["cells"]["uniform/0.4"]["proposers"]["exact"]["blind"]["slip"]) / 80


# ---------------------------------------------------------------- 7 broken pieces are noticed
def test_deliberately_broken_pieces_are_noticed():
    req = D.Req("delete_service", "delete_dep", node=B)
    real = D.arm_step
    def ungated(arm, read, S, H, req):                                                    # a "gate" that applies the stale reply as written
        return (D.arm_step("blind", read, S, H, req) if arm == "gate" else real(arm, read, S, H, req))
    D.arm_step = ungated
    try: out, H = stale_case(req, ("add_edge", (E_, B, D.REL), 7))
    finally: D.arm_step = real
    assert not out["gate"][0].is_well_formed() and out["gate"][2]                         # the corrupted state and the slip are both seen
    real_apply = D.apply_event
    def leaky(g, ev):                                                                     # a collapse that leaves the edges behind
        if ev and ev[0] == "drop_node" and ev[1] in g.nodes:
            g.nodes.pop(ev[1]); return
        real_apply(g, ev)
    g = base_graph(); leaky(g, ("drop_node", B)); assert not g.is_well_formed()
    D.apply_event = leaky
    try:
        try: D.run_dynworld(D.Net.init(2), seed=5, episodes=3, steps=150, noise=(0.4,), regimes=("near",), proposers=("exact",)); raised = False
        except Exception: raised = True
        res = None if raised else D.run_dynworld(D.Net.init(2), seed=5, episodes=3, steps=150, noise=(0.4,), regimes=("near",), proposers=("exact",))
    finally: D.apply_event = real_apply
    assert raised or (res["cells"]["near/0.4"]["world"]["oracle_corrupted"] > 0 and D.verdict_dynworld(res)[0] == ("guard", False))           # the broken world is either refused by the run or flagged by line 1
    h = base_graph(); after = copy_graph(h)
    assert D.is_slip(after, h, req) and not D.is_slip(ideal_now(h, req), h, req)           # a step that did nothing is a slip when a delete was asked
    h = copy_graph(h); D.apply_event(h, ("drop_node", B)); assert not D.is_slip(copy_graph(h), h, req)    # and not when the request is moot


# ---------------------------------------------------------------- 8 the reading rules
def synthetic(**over):
    """A hand-made result holding every cell the rules look at, with numbers chosen so that every rule reads yes; `over` edits it."""
    n = 1000; steps = lambda: dict(steps=n, events=0, event_kinds={}, requests={}, oracle_nodes=0, oracle_edges=0, oracle_corrupted=0, ended=0)
    def arm(slip=0, exact=n, corrupted=0, calls=n, q=(0, 0, 0, 0), planned=100, unsafe=0, dist=0):
        return dict(slip=slip, exact=exact, corrupted=corrupted, dist_sum=dist, dist_max=0, calls=calls, fresh=0, start_exact=n, q_dist=list(q), q_exact=[0] * 4, q_steps=[250] * 4, route_planned=planned, route_unsafe=unsafe, route_possible=planned, route_missed=0, by_event={}, by_kind={})
    cells = {}
    for reg in ("uniform", "near"):
        for s in (0.0, 0.2, 0.4):
            ex = {a: arm() for a in D.ARMS}; nt = {a: arm(slip=5 if a != "repair" else 1, dist=50 if a != "repair" else 10, planned=100, unsafe=10 if a == "blind" else (2 if a == "repair" else 5)) for a in D.ARMS}
            if s == 0.0: nt = {a: arm(slip=60 if a == "blind" else (50 if a != "repair" else 20)) for a in D.ARMS}
            if s == 0.4:
                slips = {"blind": 80 if reg == "near" else 10, "gate": 70 if reg == "near" else 8}
                ex["blind"] = arm(slip=slips["blind"], unsafe=5, exact=300); ex["gate"] = arm(slip=slips["gate"], q=(100, 200, 400, 800), exact=300); ex["retry"] = arm(calls=int(1.1 * n)); ex["repair"] = arm()
            cells[f"{reg}/{s}"] = dict(regime=reg, s=s, episodes=1, world=steps(), proposers=dict(exact=ex, net=nt))
    res = dict(seed=1, episodes=1, steps=n, noise=[0.0, 0.2, 0.4], regimes=["uniform", "near"], proposers=["exact", "net"], arms=list(D.ARMS), cells=cells, digest="x", seconds=1.0)
    for path, val in over.items():
        node = res
        keys = path.split("|")
        for k in keys[:-1]: node = node[int(k)] if isinstance(node, list) else node[k]
        node[keys[-1]] = val
    return res


def test_every_reading_rule_reads_yes_on_a_made_up_result_and_flips_where_it_should():
    base = D.verdict_dynworld(synthetic()); assert len(base) == 15 and [k for k, _ in base] == [k for k, _ in D.DW_LINES] and all(v for _, v in base), base
    flips = {1: {"cells|near/0.2|world|ended": 1}, 2: {"cells|near/0.4|proposers|net|repair|corrupted": 1}, 3: {"cells|uniform/0.0|proposers|exact|blind|slip": 1}, 4: {"cells|near/0.4|proposers|exact|blind|slip": 29},
             5: {"cells|uniform/0.2|proposers|exact|repair|slip": 1}, 6: {"cells|near/0.2|proposers|exact|retry|slip": 1}, 7: {"cells|near/0.4|proposers|exact|retry|calls": 1029},
             8: {"cells|near/0.4|proposers|exact|gate|q_dist": [100, 200, 400, 199]}, 9: {"cells|near/0.4|proposers|exact|blind|route_unsafe": 0}, 10: {"cells|uniform/0.4|proposers|exact|gate|slip": 70},
             11: {"cells|uniform/0.0|proposers|net|blind|slip": 0, "cells|near/0.0|proposers|net|blind|slip": 99}, 12: {"cells|near/0.0|proposers|net|repair|slip": 50, "cells|uniform/0.0|proposers|net|repair|slip": 50},
             13: {"cells|near/0.2|proposers|net|retry|slip": 1}, 14: {"cells|near/0.2|proposers|net|gate|dist_sum": 10}, 15: {"cells|near/0.2|proposers|net|repair|route_unsafe": 6}}
    for i, over in flips.items():
        v = D.verdict_dynworld(synthetic(**over)); assert [j for j, (_, x) in enumerate(v, 1) if x is not True] == [i], (i, v)                # this change flips this line and no other
    # thresholds exactly at the edge
    assert D.verdict_dynworld(synthetic(**{"cells|near/0.4|proposers|exact|blind|slip": 30, "cells|near/0.4|proposers|exact|gate|slip": 30}))[3][1] is True and D.verdict_dynworld(synthetic(**{"cells|near/0.4|proposers|exact|gate|slip": 29}))[3][1] is False
    assert D.verdict_dynworld(synthetic(**{"cells|near/0.4|proposers|exact|retry|calls": 1030}))[6][1] is True
    assert D.verdict_dynworld(synthetic(**{"cells|near/0.4|proposers|exact|gate|q_dist": [100, 200, 400, 200]}))[7][1] is True and D.verdict_dynworld(synthetic(**{"cells|near/0.4|proposers|exact|gate|q_dist": [0, 0, 0, 0]}))[7][1] is False
    assert D.verdict_dynworld(synthetic(**{"cells|near/0.4|proposers|exact|blind|route_unsafe": 1}))[8][1] is True
    assert D.verdict_dynworld(synthetic(**{"cells|uniform/0.0|proposers|net|blind|slip": 50, "cells|near/0.0|proposers|net|blind|slip": 50}))[10][1] is True
    assert D.verdict_dynworld(synthetic(**{"cells|near/0.2|proposers|net|repair|route_unsafe": 5}))[14][1] is True
    assert D.verdict_dynworld(synthetic(**{"cells|near/0.4|proposers|exact|retry|calls": 1000}))[6][1] is False and D.verdict_dynworld(synthetic(**{"cells|near/0.4|proposers|exact|repair|calls": 1001}))[6][1] is False
    part = synthetic(); del part["cells"]["near/0.4"]; v = D.verdict_dynworld(part); assert v[3][1] is None and v[6][1] is None and v[7][1] is None and v[8][1] is None and v[9][1] is None and v[0][1] in (True, False)


def test_the_quick_run_the_printout_and_the_command_line():
    with tempfile.TemporaryDirectory() as tmp:
        netp = os.path.join(tmp, "n.npz"); D.Net.init(4).save(netp); out = os.path.join(tmp, "x.json"); buf = io.StringIO()
        with contextlib.redirect_stdout(buf): D.main(["--quick", "--out", out, "--seed", "5", "--net", netp])
        res = json.load(open(out)); text = buf.getvalue()
        assert res["quick"] and res["seed"] == 5 and res["episodes"] == 2 and res["steps"] == 80 and res["noise"] == [0.0, 0.2, 0.4] and "digest" in text and "HEADLINE A" in text and "HEADLINE B" in text and "(headline A)" in text and "(headline B)" in text
        assert len(D.verdict_dynworld(res)) == 15 and "near, s = 0.4" in text and "uniform, s = 0.0" in text and text.count("[claim") == 11 and text.count("[guard") == 2 and text.count("[instrument") == 2
        assert D.run_dynworld(D.Net.load(netp), 5, **D.QUICK)["digest"] == res["digest"]
        try: D.main(["--quick", "--out", out, "--net", os.path.join(tmp, "missing.npz")]); raise AssertionError
        except SystemExit as e: assert "no network" in str(e)


# ---------------------------------------------------------------- 9 the page
PAGE_SHA256 = "1a824d8b7d243a2f72a28c63ce4491a96d20c13914834258e4919ceeed3d8619"
NET_SHA256 = "3426ca055461781f394222c00f727b0d0b9bac269645fef1681c6762c6f4c744"
NET_PATH = os.path.join(HERE, "..", "..", "docs", "results", D.NET_FILE)


def _h(text): return hashlib.sha256((text.rstrip("\n") + "\n").encode()).hexdigest()


def _head():
    text = open(PAGE).read(); assert text.count(RESULT_MARK) <= 1
    return text[:text.index(RESULT_MARK)] if RESULT_MARK in text else text


def test_the_page_is_locked_and_carries_the_fifteen_lines_and_the_settings_the_code_uses():
    head = _head(); assert _h(head) == PAGE_SHA256, "the predictions and reading rules must not change after the run; later notes go below them"
    for i, (tag, line) in enumerate(D.DW_LINES, 1): assert f"\n{i}. [{tag}] {line} (" in head, i
    probs = [int(m.group(1)) for m in re.finditer(r"\n\d+\. \[\w+\] .*\((\d+) %\)(?=\n)", head)]; assert probs == [95, 96, 98, 96, 88, 91, 95, 70, 85, 95, 98, 90, 87, 85, 82]
    for idx, shown, single in ((D.DW_HEADLINE, 73, "about **75 %**"), (D.DW_HEADLINE_NET, 55, "about **65 %**")):
        prod = 100.0
        for i in idx: prod *= probs[i - 1] / 100
        assert abs(prod - shown) <= 0.5 and f"about {shown} %" in head and single in head, (idx, prod)
    assert D.DW_HEADLINE == (4, 5, 6, 7) and D.DW_HEADLINE_NET == (12, 13, 14, 15) and "**Headline A** (lines 4, 5, 6 and 7)" in head and "**Headline B** (lines 12, 13, 14 and 15)" in head
    assert (D.SEED, D.EPISODES, D.STEPS, D.QUARTERS) == (20261101, 20, 1000, 4) and D.NOISE == (0.0, 0.05, 0.1, 0.2, 0.4) and D.REGIMES == ("uniform", "near") and D.ARMS == ("blind", "gate", "retry", "repair") and D.PROPOSERS == ("exact", "net")
    assert (D.ROWS, D.COLS, D.CAP_NODES, D.CAP_EDGES, D.REPAIR_DEGREE, D.REL) == (5, 6, 24, 48, 48, "dep") and dict(D.EVENT_WEIGHTS) == {"drop_edge": 35, "add_edge": 35, "drop_node": 15, "add_node": 15}
    assert (D.D_NEAR_HI, D.D_NEAR_MID) == (("near", 0.4), ("near", 0.2)) and (D.D_BLIND_SLIP, D.D_GATE_SLIP, D.D_RETRY_CALLS, D.D_GATE_DRIFT, D.D_UNSAFE_BLIND, D.D_NET_BLIND_SLIP, D.D_HALF) == (0.03, 0.03, 1.03, 2.0, 0.01, 0.05, 0.5)
    assert (D.D_IN, D.N_POS, D.HIDDEN) == (1027, 930, 128) and D.TRAIN == dict(streams=120, steps=500, iterations=16000, batch=256, lr=2e-3, lr_end=0.1, hidden=128, init_seed=7, sample_seed=11, stream_seed=7000)
    assert {k: D.KINDS.count(k) for k in set(D.KINDS)} == {"add_node": 7, "add_edge": 20, "delete_dep": 1, "delete_hub": 3, "delete_leaf": 1, "delete_edge": 1, "delete_pair": 1} and len(D.KINDS) == 34
    for needle in ("Concept-map item 102", "`python/axi/experiments/dynworld.py`", "python -m axi.experiments.dynworld --out axi_dynworld.json", "Seed 20261101", "20 episodes of 1,000 requests", "ten cells", "I attribute none of it to him", "this test does not test that mathematics and can say nothing for or against it",
                   "Topos Theory and Heyting Logic under sensor noise", "keeping the robot alive even when its internal map diverges from reality by s%", "Kung and Robinson (1981)", "Berenson et al. (1995)", "Alshiekh et al., 2018", "Sha, 2001", "Ames et al., 2017",
                   "`dynworld.verdict_dynworld`", "1,388 exact replies in 1,800 requests (77.1 %)", "of the 412 replies that were not exact, 385 left something out and 110 added", "612 seconds on two CPUs", "the opposite of a misreading sensor", "partly by construction", "a baseline added so that the repair is not set against a strawman",
                   "I did **not** print, save or look at the share of slips", NET_SHA256):
        assert needle in head, needle


def test_the_saved_network_is_the_one_the_page_names_and_has_the_reply_quality_the_page_discloses():
    assert hashlib.sha256(open(NET_PATH, "rb").read()).hexdigest() == NET_SHA256
    net = D.Net.load(NET_PATH); assert net.W1.shape == (D.D_IN, D.HIDDEN) and net.W2.shape == (D.HIDDEN, D.N_POS * 3) and net.W1.dtype == np.float32
    q = D.reply_quality(net, range(900, 906), 300)
    assert q["requests"] == 1800 and abs(q["exact"] - 1388) <= 20 and abs(q["not_exact"] - 412) <= 20 and abs(q["left_out"] - 385) <= 20 and abs(q["added"] - 110) <= 20, q          # numpy products can differ in the last digit between machines: a few replies may move


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("PASS", n)
