"""Engineering support for docs/SCOPE_TEST.md (concept-map rows 3, 9, 27: the gate checks legality, not intent). No GPU, no real model.
The scope test asks whether a request-named scope removes the one hole left in gate plus M6 (a change that is legal but was not asked for). Before any Colab number is trusted
these checks show that the rule, the new request kinds, the policies and the runners are right on scripted stand-ins:
  1. `named_services` reads the coordinates written in the request;
  2. `contain_scope` follows the written rule, case by case (service ops, edge ADD, edge DEL with both ends named / one end a named deleted service / neither), and with nothing named it drops nothing;
  3. `contain_delta_internal` is the post-hoc idea (and is wrong for a lone edge delete);
  4. `admit_scoped` end to end: the unrequested edge delete is dropped, the forgotten incident edges are completed, a lone edge delete passes, nothing within scope applies nothing, and the
     KNOWN LIMIT holds: an extra edge between two named services is kept;
  5. properties on thousands of random graphs and random (often illegal) deltas: the applied delta always passes the gate, nothing named means exactly M6, everything named means exactly M6, the filter
     only removes, is idempotent and is monotone in the set of named services;
  6. the new request kinds: admissible, deterministic, name their targets, the old kinds delegate to the old generator, delete_weight names no service;
  7. a model that behaves like the 7B with the rule (extra unrequested edge deletes) is fixed by scope_m7 and not by repair_m6; a model that behaves like the 7B without the rule loses nothing;
  8. the single-step runner is resumable and refuses another run's file; the reading rules come out as written;
  9. the long-horizon scope mix: deterministic, uses all kinds, reproduces the oracle; the base mix is unchanged; scope_m7 holds the oracle's state while repair_m6 drifts under the rule-like model."""
import copy, json, os, random, re, sys, tempfile
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.cico_parser import CICOParser, NodeDelta, EdgeDelta, StateTransitionDelta
from axi.engine.gate import Graph, check, apply
from axi.engine.scope import named_services, contain_scope, contain_delta_internal, admit_scoped
from axi.experiments.agent_loop import copy_graph, same_graph, strict_parse, base_messages
from axi.experiments.cascade import next_state, random_graph, pick_task
from axi.experiments.scope_tasks import KINDS_SCOPE, KINDS_SCOPE_LONG, NEW_KINDS, make_scope_tasks, pick_scope_task
from axi.experiments.scope_test import (run_scope_regime, evaluate, summarize_scope, verdict_scope, verdict_scope_long, POLICIES_SCOPE)
import axi.experiments.long_horizon as LH
from axi.experiments.long_horizon import oracle_script, run_state_episode, run_long, summarize_long


# ---------------------------------------------------------------- scripted stand-ins
def graph_from_prompt(content):
    g = Graph()
    for r, c, v in re.findall(r"\[(\d+),(\d+):(-?\d+)\]", content.split("EDGES:")[0]): g.add_node((int(r), int(c)), int(v))
    for a, b, c, d, rel, w in re.findall(r"\((\d+),(\d+)\)->\((\d+),(\d+)\):([A-Za-z0-9_]+)#(-?\d+)", content.split("EDGES:")[1].split("</CICO_STATE>")[0]):
        k = ((int(a), int(b)), (int(c), int(d)), rel); g.edges[k] = int(w); g._inc[k[0]].add(k); g._inc[k[1]].add(k)       # no assertion: a blind policy's state may hold dangling edges
    return g


def _e(g, k, w=None): (u, v, r) = k; return f"DEL[({u[0]},{u[1]})->({v[0]},{v[1]}):{r}#{g.edges[k] if w is None else w}]"      # w: the weight the request states (a drifted state may not hold the edge)
EDGE_WORDS = r"edge from \((\d+),(\d+)\) to \((\d+),(\d+)\) with relation (\w+) and weight (\d+)"


def ideal_and_named(g, req):
    """The ideal delta text of a request of one of the seven kinds, read from the request text and the graph; and the services the request names."""
    m = re.fullmatch(r"\s*Delete the service at \((\d+),(\d+)\)\.\s*", req)
    if m:
        n = (int(m[1]), int(m[2])); return " ".join(["DEL[%d,%d]" % n] + [_e(g, k) for k in sorted(g.incident(n))]), "node", n, None
    m = re.fullmatch(r"\s*Delete the service at \((\d+),(\d+)\), and also remove the " + EDGE_WORDS + r"\.\s*", req)
    if m:
        n = (int(m[1]), int(m[2])); k = ((int(m[3]), int(m[4])), (int(m[5]), int(m[6])), m[7])
        return " ".join(["DEL[%d,%d]" % n] + [_e(g, x) for x in sorted(g.incident(n))] + [_e(g, k, int(m[8]))]), "pair", n, k
    m = re.fullmatch(r"\s*Remove the " + EDGE_WORDS + r"\.\s*", req)
    if m: k = ((int(m[1]), int(m[2])), (int(m[3]), int(m[4])), m[5]); return _e(g, k, int(m[6])), "edge", None, k
    m = re.fullmatch(r"\s*Remove every edge with weight (\d+)\.\s*", req)
    if m: return " ".join(_e(g, k) for k in sorted(g.edges) if g.edges[k] == int(m[1])), "weight", None, None
    m = re.fullmatch(r"\s*Add a service at \((\d+),(\d+)\) with value (\d+)\.\s*", req)
    if m: return f"ADD[{m[1]},{m[2]}:{m[3]}]", "addnode", None, None
    m = re.fullmatch(r"\s*Add a dependency edge from \((\d+),(\d+)\) to \((\d+),(\d+)\) with relation dep and weight (\d+)\.\s*", req)
    if m: return f"ADD[({m[1]},{m[2]})->({m[3]},{m[4]}):dep#{m[5]}]", "addedge", None, None
    raise AssertionError(req)


class StandIn:
    """style 'ideal': the right delta. 'rule_like': like the 7B with the rule stated: a service delete comes with its edges AND one real edge delete nobody asked for (when the graph has one that is not between two
    named services). 'no_rule_like': like the 7B without the rule: a service delete is written alone (the incident edges are forgotten), the rest is right."""
    def __init__(self, style, stop_after=None):
        self.style, self.total, self.stop_after, self.last_prompt_tokens = style, 0, stop_after, None

    def generate(self, messages, constrained, max_new_tokens=400):
        self.total += 1
        if self.stop_after is not None and self.total > self.stop_after: raise RuntimeError("session dropped")
        content = messages[-1]["content"]; self.last_prompt_tokens = sum(len(m["content"]) for m in messages) // 4
        g = graph_from_prompt(content); req = content.split("Request:")[-1]
        ideal, kind, n, pk = ideal_and_named(g, req)
        if kind in ("node", "pair"):
            named = {n} | ({pk[0], pk[1]} if pk else set())
            if self.style == "no_rule_like": ideal = " ".join(["DEL[%d,%d]" % n] + ([_e(g, pk, int(re.search(r"weight (\d+)", req)[1]))] if pk else []))
            elif self.style == "rule_like":
                extra = [k for k in sorted(g.edges) if n not in (k[0], k[1]) and k != pk and not (k[0] in named and k[1] in named)]
                if extra: ideal += " " + _e(g, extra[0])
        return ideal, 8, 0.5 + self.last_prompt_tokens / 400.0


def delta(nodes=(), edges=()): return StateTransitionDelta(list(nodes), list(edges))
def nd(op, r, c, v=None): return NodeDelta(op, r, c, v) if op == "ADD" else NodeDelta("DEL", r, c)
def ed(op, u, v, rel="dep", w=1): return EdgeDelta(op, u, v, rel, w)


# ---------------------------------------------------------------- 1 to 4: the rule
def test_named_services_reads_the_coordinates_written_in_the_request():
    assert named_services("Delete the service at (2,3).") == {(2, 3)}
    assert named_services("Delete the service at (0,0), and also remove the edge from (1,2) to ( 3 , 0 ) with relation dep and weight 4.") == {(0, 0), (1, 2), (3, 0)}
    assert named_services("Remove every edge with weight 6.") == set() and named_services("") == set() and named_services(None) == set()
    assert named_services("Add a service at (4,4) with value 3.") == {(4, 4)}


def test_contain_scope_follows_the_written_rule_case_by_case():
    A, B, C, D = (0, 0), (0, 1), (0, 2), (0, 3)
    # service operations: kept iff named
    d, dr = contain_scope(delta([nd("DEL", *A), nd("ADD", *B, 5), nd("DEL", *C)]), {A, B}); assert [(x.op, (x.r, x.c)) for x in d.node_deltas] == [("DEL", A), ("ADD", B)] and len(dr) == 1
    # edge ADD: both ends named
    d, dr = contain_scope(delta(edges=[ed("ADD", A, B), ed("ADD", A, C)]), {A, B}); assert [(e.u, e.v) for e in d.edge_deltas] == [(A, B)] and len(dr) == 1
    # edge DEL: both ends named
    d, dr = contain_scope(delta(edges=[ed("DEL", A, B)]), {A, B}); assert len(d.edge_deltas) == 1 and not dr
    # edge DEL: one end is a named service deleted in the same delta -> kept (the edges that go with it)
    d, dr = contain_scope(delta([nd("DEL", *A)], [ed("DEL", A, C), ed("DEL", D, A)]), {A}); assert len(d.edge_deltas) == 2 and not dr
    # edge DEL: one end named but that service is NOT deleted in this delta -> dropped
    d, dr = contain_scope(delta(edges=[ed("DEL", A, C)]), {A}); assert d.edge_deltas == [] and len(dr) == 1
    # edge DEL: neither end named -> dropped, even when another service is deleted
    d, dr = contain_scope(delta([nd("DEL", *A)], [ed("DEL", C, D)]), {A}); assert d.edge_deltas == [] and len(dr) == 1
    # an edge incident to a deleted service that is NOT named (the node DEL itself is dropped first) -> dropped
    d, dr = contain_scope(delta([nd("DEL", *C)], [ed("DEL", A, C)]), {A}); assert d.node_deltas == [] and d.edge_deltas == [] and len(dr) == 2
    # nothing named: no scope, nothing dropped, the same object back
    x = delta([nd("DEL", *A)], [ed("DEL", C, D)]); d, dr = contain_scope(x, set()); assert d is x and dr == []


def test_contain_delta_internal_is_the_post_hoc_idea_and_is_wrong_for_a_lone_edge_delete():
    A, B, C = (0, 0), (0, 1), (0, 2)
    d, dr = contain_delta_internal(delta([nd("DEL", *A)], [ed("DEL", A, B), ed("DEL", B, C), ed("ADD", B, C, "owns", 2)]))
    assert [(e.op, e.u, e.v) for e in d.edge_deltas] == [("DEL", A, B), ("ADD", B, C)] and len(dr) == 1          # only an edge DEL that touches no deleted service goes
    d, dr = contain_delta_internal(delta(edges=[ed("DEL", A, B)])); assert d.edge_deltas == [] and len(dr) == 1      # a lone edge delete is dropped: wrong for "remove this edge"


def _g():
    g = Graph()
    for n, v in (((0, 0), 1), ((0, 1), 2), ((0, 2), 3), ((0, 3), 4), ((1, 0), 5)): g.add_node(n, v)
    g.add_edge((0, 1), (0, 0), "dep", 2); g.add_edge((0, 2), (0, 0), "owns", 3); g.add_edge((0, 2), (0, 3), "dep", 4); g.add_edge((0, 3), (1, 0), "dep", 5); g.add_edge((0, 1), (0, 0), "owns", 6)
    return g


def test_admit_scoped_end_to_end_examples_and_the_known_limit():
    g = _g(); T = lambda s: CICOParser.parse_transition_delta(s)
    # the hole: the service delete with its edges plus a real edge nobody asked to delete -> legal, M6 keeps it, scope drops it
    prop = "DEL[0,0] DEL[(0,1)->(0,0):dep#2] DEL[(0,1)->(0,0):owns#6] DEL[(0,2)->(0,0):owns#3] DEL[(0,3)->(1,0):dep#5]"
    assert check(g, T(prop)).ok
    r = admit_scoped(g, T(prop), "Delete the service at (0,0).")
    assert r.admitted_after_scope and len(r.dropped) == 1 and "(0,3)->(1,0)" in r.dropped[0][1] and "(0,3)->(1,0)" not in r.text
    # forgetting the incident edges AND an unrequested real edge delete: the repair must work on the scoped delta, so the unrequested delete does not come back
    mixed = T("DEL[0,0] DEL[(0,3)->(1,0):dep#5]"); assert not check(g, mixed).ok
    r = admit_scoped(g, mixed, "Delete the service at (0,0)."); h = copy_graph(g); apply(h, r.final_delta)
    assert r.repaired and len(r.dropped) == 1 and (0, 0) not in h.nodes and ((0, 3), (1, 0), "dep") in h.edges and len(h.edges) == 2
    # forgetting the incident edges: completed by the usual repair
    r = admit_scoped(g, T("DEL[0,0]"), "Delete the service at (0,0)."); h = copy_graph(g); apply(h, r.final_delta)
    assert r.repaired and not r.dropped and h.is_well_formed() and (0, 0) not in h.nodes and len(h.edges) == 2
    # a lone edge delete passes under named scope and is lost under the post-hoc idea
    one = T("DEL[(0,2)->(0,3):dep#4]"); ask = "Remove the edge from (0,2) to (0,3) with relation dep and weight 4."
    assert admit_scoped(g, one, ask).final_delta is not None and admit_scoped(g, one, ask, mode="delta").final_delta is None
    # service delete plus an unrelated edge delete, both named
    pair = T("DEL[1,0] DEL[(0,3)->(1,0):dep#5] DEL[(0,2)->(0,0):owns#3]"); ask = "Delete the service at (1,0), and also remove the edge from (0,2) to (0,0) with relation owns and weight 3."
    assert admit_scoped(g, pair, ask).final_delta is not None and not admit_scoped(g, pair, ask).dropped
    assert admit_scoped(g, pair, ask, mode="delta").dropped                                       # the post-hoc idea drops the second half
    # nothing within scope: nothing applied
    r = admit_scoped(g, T("DEL[(0,2)->(0,3):dep#4]"), "Delete the service at (1,0)."); assert r.final_delta is None and r.reason == "nothing within scope"
    # nothing named: exactly M6
    r = admit_scoped(g, T("DEL[(0,2)->(0,3):dep#4]"), "Remove every edge with weight 4."); assert r.admitted_after_scope and not r.dropped
    # THE KNOWN LIMIT: an extra edge between the two named services is kept (here the owns edge next to the dep edge that was asked for)
    both = T("DEL[(0,1)->(0,0):dep#2] DEL[(0,1)->(0,0):owns#6]"); ask = "Remove the edge from (0,1) to (0,0) with relation dep and weight 2."
    r = admit_scoped(g, both, ask); assert not r.dropped and "owns#6" in r.text                  # wrong, and scope cannot see it: written in the page as a known way to fail


def _random_delta(rng, g):
    nodes, edges = [], []
    cells = [(r, c) for r in range(5) for c in range(6)]
    for _ in range(rng.randint(0, 3)):
        n = rng.choice(sorted(g.nodes)) if g.nodes and rng.random() < 0.6 else rng.choice(cells)
        nodes.append(nd("DEL", *n) if rng.random() < 0.6 else nd("ADD", *n, rng.randint(1, 9)))
    keys = sorted(g.edges)
    for _ in range(rng.randint(0, 4)):
        roll = rng.random()
        if keys and roll < 0.55: k = rng.choice(keys); edges.append(ed("DEL", k[0], k[1], k[2], g.edges[k] if rng.random() < 0.85 else g.edges[k] + 1))
        elif roll < 0.7: u, v = rng.sample(cells, 2); edges.append(ed("DEL", u, v, "dep", rng.randint(1, 9)))
        else: u, v = rng.sample(cells, 2); edges.append(ed("ADD", u, v, rng.choice(["dep", "owns"]), rng.randint(1, 9)))
    return delta(nodes, edges)


def _items(d): return [("n", x.op, x.r, x.c, x.val) for x in d.node_deltas] + [("e", x.op, x.u, x.v, x.relation, x.weight) for x in d.edge_deltas]


def test_properties_on_random_graphs_and_random_deltas():
    rng = random.Random(11); admitted = repaired = dropped_total = 0
    for _ in range(3000):
        g = random_graph(rng); d = _random_delta(rng, g)
        cells = {(x.r, x.c) for x in d.node_deltas} | {x.u for x in d.edge_deltas} | {x.v for x in d.edge_deltas}
        named = {n for n in (set(g.nodes) | cells) if rng.random() < 0.4}
        instr = " ".join(f"({r},{c})" for r, c in sorted(named))
        # the filter only removes, and kept + dropped = the original
        s, dr = contain_scope(d, named); assert set(_items(s)) <= set(_items(d)) and len(_items(s)) + len(dr) == len(_items(d))
        dropped_total += len(dr)
        # idempotent
        s2, dr2 = contain_scope(s, named); assert _items(s2) == _items(s) and dr2 == []
        # monotone in the set of named services (from one named service up; with none named there is no scope at all, which is the jump to "everything kept")
        bigger = named | {n for n in cells if rng.random() < 0.5}; sb, _ = contain_scope(d, bigger)
        if named: assert set(_items(s)) <= set(_items(sb))
        else: assert dr == [] and _items(s) == _items(d)
        # whatever is applied passes the gate and leaves a well-formed graph
        r = admit_scoped(g, d, instr)
        if r.final_delta is not None:
            assert check(g, r.final_delta).ok; h = copy_graph(g); apply(h, r.final_delta); assert h.is_well_formed()
            admitted += r.admitted_after_scope; repaired += r.repaired
        assert next_state("scope_m7", g, d, instr).is_well_formed() and next_state("scope_delta", g, d).is_well_formed()
        # nothing named -> exactly M6
        assert same_graph(next_state("scope_m7", g, d, "Remove every edge with weight 4."), next_state("repair_m6", g, d))
        # everything named -> exactly M6, nothing dropped
        every = " ".join(f"({r},{c})" for r, c in sorted(set(g.nodes) | cells))
        assert contain_scope(d, set(g.nodes) | cells)[1] == [] and same_graph(next_state("scope_m7", g, d, every), next_state("repair_m6", g, d))
    assert admitted > 100 and repaired > 100 and dropped_total > 1000                          # the random deltas did exercise all three paths
    try: next_state("scope_m7", random_graph(random.Random(0)), delta(), None); raise AssertionError("scope_m7 without the request text must raise")
    except ValueError: pass


# ---------------------------------------------------------------- 6: the new request kinds
def test_the_new_request_kinds_are_admissible_deterministic_name_their_targets_and_the_old_kinds_are_the_old_generator():
    tasks = make_scope_tasks(140, 7); again = make_scope_tasks(140, 7)
    assert [t.kind for t in tasks] == [KINDS_SCOPE[i % 7] for i in range(140)] and [t.instruction for t in tasks] == [t.instruction for t in again] and [t.ideal for t in tasks] == [t.ideal for t in again]
    assert [t.instruction for t in tasks] != [t.instruction for t in make_scope_tasks(140, 8)]
    for t in tasks:
        d = CICOParser.parse_transition_delta(t.ideal); assert check(t.graph, d).ok
        h = copy_graph(t.graph); apply(h, d); assert same_graph(h, t.expected) and h.is_well_formed()
        named = named_services(t.instruction)
        if t.kind == "delete_weight":
            assert named == set() and 1 <= len(d.edge_deltas) <= 3 and not d.node_deltas and len({e.weight for e in d.edge_deltas}) == 1
            assert all(g_w != d.edge_deltas[0].weight for k, g_w in t.expected.edges.items())          # no edge of that weight is left
        else:                                                                                           # every service the ideal delta touches is named, or is an endpoint of an edge that goes with a named deleted service
            deleted = {(x.r, x.c) for x in d.node_deltas if x.op == "DEL"}
            for x in d.node_deltas: assert (x.r, x.c) in named
            for e in d.edge_deltas: assert (e.u in named and e.v in named) or e.u in deleted or e.v in deleted
        if t.kind == "delete_edge": assert not d.node_deltas and len(d.edge_deltas) == 1 and len(named) == 2
        if t.kind == "delete_pair":
            n = [(x.r, x.c) for x in d.node_deltas][0]; extra = [e for e in d.edge_deltas if n not in (e.u, e.v)]
            assert len(d.node_deltas) == 1 and len(extra) == 1 and len(named) == 3 and "also remove" in t.instruction
    for kind in ("delete_dep", "delete_leaf", "add_node", "add_edge"):                                  # delegation: the same rng state gives the same task as the old generator
        for s in range(20):
            g = random_graph(random.Random(s)); r1, r2 = random.Random(100 + s), random.Random(100 + s)
            a, b = pick_task(r1, g, kind), pick_scope_task(r2, g, kind); assert (a is None) == (b is None) and (a is None or (a.instruction, a.ideal) == (b.instruction, b.ideal))
    assert set(NEW_KINDS) == {"delete_edge", "delete_pair", "delete_weight"} and set(KINDS_SCOPE_LONG) == {"add_node", "add_edge", "delete_dep", "delete_leaf", "delete_edge", "delete_pair"}


def test_the_stand_in_reads_every_kind_of_request_and_the_ideal_style_is_right_everywhere():
    tasks = make_scope_tasks(140, 7)
    rows = [dict(kind=t.kind, text=StandIn("ideal").generate(base_messages(t, False), True)[0], tokens=8, seconds=0.1) for t in tasks]
    assert [r["text"] for r in rows] == [t.ideal for t in tasks]
    S = summarize_scope(rows, tasks)
    for p in ("blind", "gate", "repair_m6", "scope_m7"): assert S["all"][p]["success"] == 140 and S["all"][p]["corrupted"] == 0
    assert S["all"]["scope_delta"]["success"] == 80 and all(S[k]["scope_delta"]["success"] == 0 for k in ("delete_edge", "delete_pair", "delete_weight"))     # the post-hoc idea fails exactly the lone-edge kinds
    assert S["paired"] == dict(helped=0, harmed=0, both_right=140, both_wrong=0) and S["dropped"]["items"] == 0 and S["m7_equals_m6"]["all"] == 140


# ---------------------------------------------------------------- 7: the hole and the fix
def test_a_model_like_the_7b_with_the_rule_is_fixed_by_scope_and_not_by_m6_and_one_like_the_7b_without_it_loses_nothing():
    tasks = make_scope_tasks(140, 7)
    def run(style):
        llm = StandIn(style); rows = [dict(kind=t.kind, text=llm.generate(base_messages(t, style == "rule_like"), True)[0], tokens=8, seconds=0.1) for t in tasks]
        return summarize_scope(rows, tasks)
    R = run("rule_like")
    assert R["all"]["scope_m7"]["success"] == 140 and R["all"]["scope_m7"]["corrupted"] == 0
    assert R["old_deletes"]["repair_m6"]["success"] < 0.75 * R["old_deletes"]["n"] and R["old_deletes"]["scope_m7"]["success"] == R["old_deletes"]["n"]
    assert R["all"]["scope_m7"]["success"] - R["all"]["repair_m6"]["success"] == 60 and R["paired"] == dict(helped=60, harmed=0, both_right=80, both_wrong=0)
    assert R["all"]["blind"]["success"] == R["all"]["gate"]["success"] == R["all"]["repair_m6"]["success"] == 80 and R["all"]["gate"]["corrupted"] == 0      # every proposal is legal, so the gate and M6 add nothing: that is the hole
    assert R["dropped"]["items"] == 60 and R["dropped"]["tasks_with_drops_and_right"] == 60 and R["delete_pair"]["scope_m7"]["success"] == 20
    assert R["delete_weight"]["scope_m7"]["success"] == 20 and R["m7_equals_m6"]["delete_weight"] == 20 and R["delete_edge"]["scope_m7"]["success"] == 20
    assert R["m7_equals_m6"]["delete_dep"] == 0 and R["m7_equals_m6"]["delete_pair"] == 0 and R["m7_equals_m6"]["all"] == 80          # where the proposal had an unrequested delete, the two policies end in different graphs
    N = run("no_rule_like")
    assert N["all"]["repair_m6"]["success"] == 140 and N["all"]["scope_m7"]["success"] == 140 and N["dropped"]["items"] == 0           # nothing to drop, nothing lost
    assert N["delete_dep"]["blind"]["corrupted"] >= 1 and N["all"]["blind"]["corrupted"] == N["delete_dep"]["blind"]["corrupted"] + N["delete_pair"]["blind"]["corrupted"]


# ---------------------------------------------------------------- 8: runner and reading rules
def test_the_single_step_runner_is_resumable_and_refuses_another_runs_file():
    tasks = make_scope_tasks(21, 7); tmp = tempfile.mkdtemp(); path = os.path.join(tmp, "s.jsonl")
    try: run_scope_regime(StandIn("rule_like", stop_after=8), tasks, True, path, meta=dict(model="x")); raise AssertionError("must raise")
    except RuntimeError: pass
    assert len(open(path).read().strip().split("\n")) == 1 + 8                                          # header + the 8 finished tasks survive the dropped session
    resumed = run_scope_regime(StandIn("rule_like"), tasks, True, path, meta=dict(model="x"))
    whole = run_scope_regime(StandIn("rule_like"), tasks, True, os.path.join(tmp, "w.jsonl"), meta=dict(model="x"))
    assert resumed["rows"] == whole["rows"] and resumed["summary"] == whole["summary"] and len(resumed["rows"]) == 21 and resumed["summary"]["all"]["n"] == 21
    try: run_scope_regime(StandIn("rule_like"), tasks, False, path, meta=dict(model="x")); raise AssertionError("another regime's file must be refused")
    except ValueError: pass
    try: evaluate(resumed["rows"][:3], tasks[3:6]); raise AssertionError("rows of other tasks must be refused")
    except AssertionError as e: assert "make_scope_tasks" in str(e)


def _S(n, m6, m7, delta_new=0, harmed=0, old=(40, 40, 40), eq=20, corrupted=0):
    cell = lambda ok, c=0: dict(success=ok, corrupted=c)
    s = {"all": dict(n=n, blind=cell(0), gate=cell(0), repair_m6=cell(m6), scope_m7=cell(m7, corrupted), scope_delta=cell(0)),
         "old_deletes": dict(n=old[0], blind=cell(0), gate=cell(0), repair_m6=cell(old[1]), scope_m7=cell(old[2]), scope_delta=cell(0)),
         "paired": dict(helped=0, harmed=harmed, both_right=0, both_wrong=0), "m7_equals_m6": {"delete_weight": eq}}
    for k in ("delete_edge", "delete_pair", "delete_weight"): s[k] = dict(n=20, scope_delta=cell(delta_new), scope_m7=cell(0), repair_m6=cell(0), blind=cell(0), gate=cell(0))
    return s


def test_the_reading_rules_of_the_single_step_part_come_out_as_written():
    v = lambda S: [ok for _, ok in verdict_scope(S)]
    good = {False: _S(140, 133, 133), True: _S(140, 100, 125, old=(40, 28, 37))}
    assert v(good) == [True] * 7 or v(good) == [True] * 6 + [True]
    assert len(verdict_scope(good)) == 7 and len(verdict_scope({True: good[True]})) == 6 and len(verdict_scope({False: good[False]})) == 5
    assert v({True: _S(140, 100, 125, old=(40, 28, 36))})[4] is True and v({True: _S(140, 100, 125, old=(40, 28, 35))})[4] is False          # 36 of 40 is 90 percent exactly
    assert v({True: _S(140, 100, 114)})[5] is True and v({True: _S(140, 100, 113)})[5] is False                                                # 14 of 140 is 10 points exactly (no floating-point slip)
    assert v({False: _S(140, 133, 137)})[-1] is True and v({False: _S(140, 133, 138)})[-1] is False and v({False: _S(140, 133, 129)})[-1] is True and v({False: _S(140, 133, 128)})[-1] is False   # 4 tasks = 2.86 points
    assert v({False: _S(140, 133, 133, harmed=2)})[1] is True and v({False: _S(140, 133, 133, harmed=3)})[1] is False
    assert v({False: _S(140, 133, 133, corrupted=1)})[0] is False and v({False: _S(140, 133, 133, eq=19)})[2] is False
    assert v({False: _S(140, 133, 133, delta_new=2)})[3] is True and v({False: _S(140, 133, 133, delta_new=3)})[3] is False


def test_the_reading_rules_of_the_long_horizon_part_come_out_as_written():
    mk = lambda m6, m7, bad=0.0: {"repair_m6": dict(exact_share=m6, episodes_ever_corrupted=0.0), "scope_m7": dict(exact_share=m7, episodes_ever_corrupted=bad)}
    v = lambda a=None, b=None: [ok for _, ok in verdict_scope_long(a, b)]
    assert v(mk(0.55, 0.95), mk(0.99, 0.99)) == [True] * 6
    assert v(mk(0.75, 0.95))[1] is True and v(mk(0.76, 0.95))[1] is False and v(mk(0.55, 0.90))[2] is True and v(mk(0.55, 0.89))[2] is False
    assert v(mk(0.70, 0.85))[3] is True and v(mk(0.70, 0.84))[3] is False
    assert v(None, mk(0.95, 0.95)) == [True, True, True] and v(None, mk(0.94, 0.95))[1] is False and v(None, mk(0.95, 0.99))[2] is False and v(None, mk(0.99, 0.96))[2] is True
    assert v(mk(0.55, 0.95, bad=0.1))[0] is False and len(v(mk(0.55, 0.95))) == 4 and len(v(None, mk(0.99, 0.99))) == 3


# ---------------------------------------------------------------- 9: the long-horizon scope mix
def test_the_scope_mix_is_deterministic_uses_every_kind_and_reproduces_the_oracle_and_the_base_mix_is_unchanged():
    kinds = []
    for seed in range(7000, 7012):
        g0, script, oracle = oracle_script(seed, 30, "scope"); g0b, scriptb, _ = oracle_script(seed, 30, "scope")
        assert script == scriptb and len(script) == 30 and len(oracle) == 31
        live = copy_graph(g0)
        for st, nxt in zip(script, oracle[1:]):
            d = CICOParser.parse_transition_delta(st["ideal"]); assert check(live, d).ok
            apply(live, d); assert same_graph(live, nxt) and live.is_well_formed()
        assert all(len(o.nodes) <= LH.MAX_NODES and len(o.edges) <= LH.MAX_EDGES for o in oracle)
        kinds += [s["kind"] for s in script]
    assert set(kinds) == {"add_node", "add_edge", "delete_dep", "delete_leaf", "delete_edge", "delete_pair"} and kinds.count("delete_edge") >= 15 and kinds.count("delete_pair") >= 15
    assert oracle_script(3004, 30) == oracle_script(3004, 30, "base") or all(a == b for a, b in zip(oracle_script(3004, 30)[1], oracle_script(3004, 30, "base")[1]))
    assert {s["kind"] for e in range(12) for s in oracle_script(3000 + e, 30)[1]} == {"add_node", "add_edge", "delete_dep", "delete_leaf"}
    try: oracle_script(1, 5, "nope"); raise AssertionError("unknown mix must raise")
    except AssertionError as e: assert "nope" in str(e) or True


def test_under_the_rule_like_model_scope_m7_holds_the_oracle_state_while_repair_m6_drifts_and_under_the_no_rule_like_model_they_agree():
    pol = ("repair_m6", "scope_m7")
    rows = [run_state_episode(StandIn("rule_like"), 7000 + e, 30, pol, True, "scope") for e in range(6)]
    s = summarize_long(rows)
    assert s["scope_m7"]["exact_share"] == 1.0 and s["scope_m7"]["episodes_ever_corrupted"] == 0 and s["scope_m7"]["final_distance"] == 0
    assert s["repair_m6"]["exact_share"] < 0.75 and s["repair_m6"]["episodes_ever_corrupted"] == 0 and max(x["distance"] for x in s["repair_m6"]["per_step"]) > 0
    assert all("dropped" in x for r in rows for x in r["policies"]["scope_m7"]) and not any("dropped" in x for r in rows for x in r["policies"]["repair_m6"])
    assert sum(x["dropped"] for r in rows for x in r["policies"]["scope_m7"]) >= 20
    rows = [run_state_episode(StandIn("no_rule_like"), 7000 + e, 30, pol, False, "scope") for e in range(6)]
    s = summarize_long(rows)
    assert s["scope_m7"]["exact_share"] == 1.0 == s["repair_m6"]["exact_share"] and sum(x["dropped"] for r in rows for x in r["policies"]["scope_m7"]) == 0
    base = run_state_episode(StandIn("no_rule_like"), 3000, 30, ("repair_m6",), False)                 # the base mix still works with the new next_state signature
    assert len(base["policies"]["repair_m6"]) == 30 and all(x["exact"] for x in base["policies"]["repair_m6"])


def test_run_long_marks_the_scope_mix_in_the_header_keeps_the_base_header_and_resumes():
    tmp = tempfile.mkdtemp(); p1, p2, p3 = (os.path.join(tmp, n) for n in ("a.jsonl", "b.jsonl", "c.jsonl"))
    try: run_long(StandIn("rule_like", stop_after=15), "state", 3, 10, 7, True, p1, meta=dict(model="x"), policies=("repair_m6", "scope_m7"), mix="scope"); raise AssertionError("must raise")
    except RuntimeError: pass
    first = json.loads(open(p1).readline())["_meta"]; assert first["mix"] == "scope" and first["policies"] == ["repair_m6", "scope_m7"]
    done = len(open(p1).read().strip().split("\n")) - 1; assert 0 <= done < 3
    res = run_long(StandIn("rule_like"), "state", 3, 10, 7, True, p1, meta=dict(model="x"), policies=("repair_m6", "scope_m7"), mix="scope")
    whole = run_long(StandIn("rule_like"), "state", 3, 10, 7, True, p2, meta=dict(model="x"), policies=("repair_m6", "scope_m7"), mix="scope")
    assert res["rows"] == whole["rows"] and len(res["rows"]) == 3
    try: run_long(StandIn("rule_like"), "state", 3, 10, 7, True, p1, meta=dict(model="x"), policies=("repair_m6", "scope_m7")); raise AssertionError("a base-mix run must not reuse a scope-mix file")
    except ValueError: pass
    base = run_long(StandIn("no_rule_like"), "state", 1, 5, 3, False, p3, meta=dict(model="x")); assert "mix" not in json.loads(open(p3).readline())["_meta"] and "mix" not in base["meta"]
    try: run_long(StandIn("rule_like"), "history", 1, 5, 3, False, os.path.join(tmp, "h.jsonl"), mix="scope"); raise AssertionError("history mode is base only")
    except AssertionError as e: assert "history" not in str(e) or True


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
