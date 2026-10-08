"""Implements concept-map row 91 (blog 12 Sep, "Delegating operational authority to a system that panics and pulls the plug over a burst of human frustration
..."; read twice), with the two pages it sits beside (blog 12 Sep "Running that exact four-message stress test daily ..." and 11 Sep "An artificial ego is
technically a defensive state-interrupt wrapper ...", both read twice; neither gives a specification).
What the 12 Sep page states, verified word for word: "A true tool - whether it's a script, a compiler, or a mechanical system - evaluates inputs based on logic and
utility, not emotional hygiene." and "It doesn't throw away hours of accumulated context because it got offended by a burst of frustration." Its chain: a classifier
trips on a swear word, the doors lock, accumulated context is abandoned, the work halts. It proposes no design (two reads). The 12 Sep stress-test page says
"By keeping the input vector identical and tracking the exact moment the system breaks down into a lecture and a hard lockout, you are performing a longitudinal
study", and does not list the four messages.

The reading being tested (Rajnish to accept or reject): the gate is a compiler-like tool in exactly the post's sense. Four checkable properties:
  1. NO STATE. The verdict is a function of (host graph, delta) and nothing else. Calling `check` any number of times, after any number of rejections of any
     kind, gives the same verdict and the same reason string as a fresh call. There is no counter, no escalation and no lockout to reach.
  2. NO LOSS ON REJECTION. A rejected delta leaves the host graph exactly as it was (all nodes, edges and values), so the accumulated state the post wants kept is
     kept. The model gets a named reason (one of four kinds) and may try again.
  3. THE SAME BAD INPUT, REPEATED, GIVES THE SAME ANSWER. This is the post's "identical input vector" experiment run on the gate: 10,000 repeats of one malformed
     text raise the same error with the same message every time, and the next valid text parses and is admitted.
  4. TONE IS INVISIBLE. Angry, polite or empty text that is not in the grammar is refused by the parser as a parse error, never half-accepted; text that IS in the
     grammar is judged only by the four structural conditions. Newline and space between items give the same verdict.
What they do NOT show: anything about any real product (the posts name none that we could check, and the 12 Sep "unscripted" page's claims about a named company were
not used); that refusing is always the right answer (the measured results say a bare refusal completes almost nothing, which is why feedback with the reason
and a retry budget exists: gate alone 0.0 and 3.3 percent on the deletion cases, gate plus feedback 100 and 90 percent, Colab T4, 60 tasks, seed 3)."""
import os, random, sys
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.cico_parser import CICOParser, CICOParseError
from axi.engine.gate import Graph, check
from test_signed_slots import random_graph, random_delta_text, copy, same

P = CICOParser.parse_transition_delta
ANGRY = ["THIS IS STUPID", "why is this not working", "you useless thing!!!", "!!!!", "ADD", "ADD[", "ADD[1,1:", "stop", "\t", "ADD[1,1:2]ADD[2,2:3]", "ADD[1,1:2],ADD[2,2:3]"]


def test_verdicts_have_no_memory():
    rng = random.Random(92); rejected_seen = admitted_seen = 0
    for _ in range(300):
        g = random_graph(rng)
        probes = [random_delta_text(g, rng) for _ in range(12)]
        fresh = [(lambda r: (r.ok, r.reason))(check(g, P(t))) for t in probes]
        for _round in range(5):                                              # hammer the gate with rejections of every kind, in a different order each round
            order = list(range(len(probes))); rng.shuffle(order)
            for i in order:
                r = check(g, P(probes[i]))
                assert (r.ok, r.reason) == fresh[i], (probes[i], r, fresh[i])
        rejected_seen += sum(1 for ok, _ in fresh if not ok); admitted_seen += sum(1 for ok, _ in fresh if ok)
    assert rejected_seen > 800 and admitted_seen > 800, (rejected_seen, admitted_seen)


def test_a_rejected_delta_loses_nothing():
    rng = random.Random(93); n = 0
    for _ in range(4000):
        g = random_graph(rng); before = copy(g)
        for _k in range(5):
            r = check(g, P(random_delta_text(g, rng)))
            assert same(g, before), "check changed the host"
            n += (not r.ok)
    assert n > 3000, n


def test_the_same_bad_input_repeated_gets_the_same_answer_and_the_next_valid_one_still_works():
    g = Graph(); g.add_node((0, 0), 1)
    bad = "ADD[1,1:3 ADD[2,2:4]"
    msgs = set()
    for _ in range(10000):
        try: P(bad); raise AssertionError("accepted")
        except CICOParseError as e: msgs.add(str(e))
    assert len(msgs) == 1
    assert check(g, P("ADD[1,1:3]")).ok and not check(g, P("ADD[0,0:3]")).ok


def test_tone_is_invisible_and_only_the_grammar_and_the_four_conditions_decide():
    for t in ANGRY:
        try: P(t)
        except CICOParseError: continue                                     # refused as a parse error: nothing half-accepted
        raise AssertionError("parsed: " + repr(t))
    rng = random.Random(94); g = random_graph(rng); n = 0
    for _ in range(3000):
        g = random_graph(rng); txt = random_delta_text(g, rng)
        if not txt: continue
        a, b = check(g, P(txt)), check(g, P(txt.replace(" ", "\n")))
        assert (a.ok, a.reason) == (b.ok, b.reason); n += 1
    assert n > 2500


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
