"""Implements concept-map row 104 (tenth scan): blog 12 Aug "The Blackboard That Never Has to Be Erased: A child learning to write usually starts on
something that can be undone - a slate, a whiteboard, a blackboard with a stick ..." (two reads; about 1,300 to 1,500 words by the summarizer's estimate,
UNSURE; byline rajnish choubey; authorship unknown). Quotes verified twice: "The blackboard only ever had the one plane;"; "a wedge pressed in could still
be smoothed over and pressed again, before it dried hard"; "The newest blackboard is faster, bigger, and never needs erasing."; "this is a difference
in the surface, not a difference in what the surface actually understands." The page runs a chronological ladder of writing surfaces judged by whether a
mark can be undone: rock (no) - clay (only while wet) - page (a crossed-out word stays visible) - blackboard (erase, and the teacher must remember what
was erased) - an AI system on a screen (the visible plane "can simply keep extending"). Two planes: a hidden one where the calculation happens and a
visible one where the result shows. The page does NOT say what becomes of the hidden plane's work or give any limit on the visible plane.

The reading being tested (Rajnish to accept or reject): the ladder is a ladder of how much history a surface keeps. In OUR language the same ladder is
exactly the difference between keeping a log of inverses and not keeping one (`invert`, concept-map rows 38, 58, 65):
  1. "Page": an APPEND-ONLY log of inverses (each computed before its delta is applied). Replaying the log newest-first returns the host to EVERY earlier
     checkpoint, exactly (300 random episodes, up to 10 admitted steps each). The log only ever grows.
  2. "Blackboard": erase the oldest log entry and the host can no longer go back past the point where it was erased; what the host holds then is a state it
     could have reached from other starts. Two different starting hosts, the same two steps (DEL a node, ADD it back with another value), the SAME final
     host: the final state cannot tell them apart, so the start is unrecoverable without the log.
  3. "Rock": an inverse computed AFTER the delete cannot be built, because the deleted value exists only in the host before it (`invert` raises KeyError).
What they do NOT show: that the AI's visible plane is append-only (the page asserts it); anything about a hidden plane; that the gate should keep a log
(it does not: the host is the whole state, and `invert` needs the graph from before the step). Also, the claim "never has to be erased" is about a
visible text surface; nothing here measures a model's context."""
import os, random, sys
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.cico_parser import CICOParser
from axi.engine.gate import Graph, check, apply
from axi.engine.inverse import invert, to_text
from test_signed_slots import same, copy, random_graph, random_delta_text


def episode(seed, steps=10):
    rng = random.Random(seed); g = random_graph(rng); states, log = [copy(g)], []
    for _ in range(500):
        d = CICOParser.parse_transition_delta(random_delta_text(g, rng))
        if not (d.node_deltas or d.edge_deltas) or not check(g, d).ok: continue
        log.append(to_text(invert(g, d)))                                   # computed BEFORE the step: the deleted value is still in g
        apply(g, d); states.append(copy(g))
        if len(log) == steps: break
    return g, states, log


def replay(g, texts):
    h = copy(g)
    for t in texts:
        d = CICOParser.parse_transition_delta(t)
        assert check(h, d).ok, t
        apply(h, d)
    return h


def test_a_log_of_inverses_returns_the_host_to_every_earlier_checkpoint():
    total = 0
    for seed in range(300):
        g, states, log = episode(seed)
        assert len(log) == len(states) - 1
        total += len(log)
        for k in range(len(log) + 1):
            assert same(replay(g, reversed(log[k:])), states[k]), (seed, k)         # newest first
    assert total > 1500


def test_without_the_log_the_start_cannot_be_recovered_from_the_end():
    a, b = Graph(), Graph()
    a.add_node((0, 0), 1); b.add_node((0, 0), 2)
    for g in (a, b):
        for t in ("DEL[0,0]", "ADD[0,0:5]"):
            d = CICOParser.parse_transition_delta(t); assert check(g, d).ok; apply(g, d)
    assert same(a, b)                                                       # different starts, one end
    # an erased log entry: dropping the oldest inverse leaves the host stuck at the second state
    n_cut = 0
    for seed in range(300):
        g, states, log = episode(seed)
        if len(log) < 2: continue
        got = replay(g, reversed(log[1:]))
        assert same(got, states[1]) and not same(got, states[0]); n_cut += 1
    assert n_cut > 250


def test_an_inverse_computed_after_the_delete_cannot_be_built():
    g = Graph(); g.add_node((1, 1), 7)
    d = CICOParser.parse_transition_delta("DEL[1,1]")
    before = invert(g, d); assert to_text(before) == "ADD[1,1:7]"
    apply(g, d)
    try: invert(g, d)
    except KeyError: pass
    else: raise AssertionError("an inverse after the delete should be impossible")


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
