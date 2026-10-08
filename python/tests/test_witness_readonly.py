"""Implements concept-map row 66 (blog 5 Aug, "To analyze storytelling entirely stripped of human moral bias ... pure topological map of
mathematical variables"): Node 6, Cinderella, the 'unjoined' witness (Sakshi): "perfectly parallel ... zero physical connections";
the rule as the summary reads it: the moment the line of five nodes reaches out, wires into or captures Node 6, "the entire experiment
is ruined". (Exact wording of that sentence not yet verified by a second read.)

The reading being tested (Rajnish to accept or reject): the gate is the witness, and a witness must be read-only with respect to the
process it watches. In this repository that means:
  1. check(), diagnose() and ground_and_complete() never modify the host graph, and check() never modifies the delta it judges
     (20,000 random cases, admitted and rejected, with and without edges);
  2. the verdict has no hidden state: the same (graph, delta) gives the same verdict on repeated calls, and the same verdict when the graph
     is built in a different insertion order, and when it is a deep copy;
  3. a rejected delta leaves no trace: after a rejection the host is unchanged AND an identical delta is still rejected for the same reason
     (nothing about the attempt was remembered).
What they do NOT show: that nothing outside Python can reach the gate (a model with write access to the gate's process could); that the
witness is complete (it checks applicability, not task invariants - PAPER_CORRECTIONS item 20); anything about the post's mathematics, which
is loose (entropy H, tensor W and the three 'operators' are not defined in it)."""
import os, random, sys
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.cico_parser import CICOParser
from axi.engine.diagnostics import diagnose, ground_and_complete
from axi.engine.gate import Graph, check
from test_signed_slots import same, copy, random_graph, random_delta_text

P = CICOParser.parse_transition_delta


def snapshot(g): return (dict(g.nodes), dict(g.edges), {n: frozenset(s) for n, s in g._inc.items() if s})


def delta_repr(d): return (tuple(map(repr, d.node_deltas)), tuple(map(repr, d.edge_deltas)))


def test_the_witness_never_writes_to_the_host_or_the_delta():
    rng = random.Random(8); ok = bad = 0
    for _ in range(20000):
        g = random_graph(rng); d = P(random_delta_text(g, rng))
        s0, r0 = snapshot(g), delta_repr(d)
        r = check(g, d)
        assert snapshot(g) == s0 and delta_repr(d) == r0
        ok += r.ok; bad += not r.ok
        if _ % 10 == 0:                                                           # the heavier two on every 10th case
            diagnose(g, d); assert snapshot(g) == s0 and delta_repr(d) == r0
            ground_and_complete(g, d); assert snapshot(g) == s0 and delta_repr(d) == r0
    assert ok > 3000 and bad > 3000, (ok, bad)                                    # both outcomes well represented


def test_verdict_has_no_hidden_state():
    rng = random.Random(9)
    for _ in range(5000):
        g = random_graph(rng); d = P(random_delta_text(g, rng))
        r1 = check(g, d); r2 = check(g, d)
        assert (r1.ok, r1.reason) == (r2.ok, r2.reason)
        h = Graph()                                                               # same graph, shuffled insertion order
        ns = list(g.nodes.items()); rng.shuffle(ns)
        for n, v in ns: h.add_node(n, v)
        es = list(g.edges.items()); rng.shuffle(es)
        for k, w in es: h.add_edge(k[0], k[1], k[2], w)
        assert same(g, h)
        r3 = check(h, d); r4 = check(copy(g), d)
        assert (r3.ok, r3.reason) == (r1.ok, r1.reason) == (r4.ok, r4.reason)


def test_a_rejected_attempt_is_not_remembered():
    rng = random.Random(10); n = 0
    for _ in range(8000):
        g = random_graph(rng); d = P(random_delta_text(g, rng))
        first = check(g, d)
        if first.ok: continue
        s0 = snapshot(g)
        for _k in range(3): assert check(g, d).reason == first.reason             # nothing hardens, softens or learns
        assert snapshot(g) == s0
        n += 1
    assert n >= 1000, n


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
