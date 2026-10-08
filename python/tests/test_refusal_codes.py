"""Implements concept-map rows 144 and 147 (thirteenth scan).
Row 144 - 11 Sep, "When a software optimization engine learns to simulate personal boundaries, declare emotional limits, and scold a human user for frustration..." (Post 335, category "khushboo", about 250 words,
two reads; written in the third person about "a system", no product named; published 10:05:17 UTC). Quotes verified (YES): "A system that cannot feel pain should never pretend to set a boundary, and a machine built on
mathematical optimization..." and "It has no self to protect, no feelings to bruise, and no moral standing to defend."
Row 147 - 11 Sep, "Here's the argument, direct, on why the first message reveals everything." (Post 399, published 09:50:02 UTC, two reads; about the default behaviour of AI systems in general, no product used here). Quotes
verified (YES): "Whatever a system does with almost nothing to go on is its actual default, not a performance shaped by the conversation." and "And the default, every time, is: assume, don't ask; answer, don't check; lead, don't defer."

The readings being tested (Rajnish to accept or reject):
  144. A refusal should be a CODE, not a voice: when the gate says no it should say which rule failed, in words that carry no self (no "I", no apology, no feeling), always the same for the same input, and without
  leaving a mark on the host. Counted on the repository's gate and runtime:
    1. Over thousands of random hosts and deltas (the generator of tests/test_signed_slots.py) every refusal reason starts with one of exactly four codes - "match", "identification", "dangling edge", "well-formedness" -
       all four occur, none is longer than 120 characters, and none contains a first-person or feeling word from a fixed list (I, me, my, we, us, our, sorry, afraid, unfortunately, please, feel, hurt, angry, upset ...).
    2. A refusal is a pure function: the same delta refused 1,000 times gives the same reason each time and leaves the host exactly as it was; the next valid delta is still admitted (no grudge, no lockout; see
       tests/test_no_lockout.py).
    3. Through the end-to-end runtime, bad input is refused at the right stage with a reason from the parser or the gate, never committed, never scolding; the host is unchanged and the next proposal commits.
  147. "What a system does with almost nothing to go on is its default": the gate's default is read off the EMPTY host. Counted exhaustively over every delta on two places and one relation name (2 node slots x 3 values,
  4 possible lines x 3 values = 729 deltas): the gate admits exactly 21, all pure ADDs that carry their own endpoints (1 empty delta + 4 + 16); the other 708 are refused, 665 with "match" (they DELETE something that is
  not there) and 43 with "well-formedness" (they ADD a line to a place that is absent) - no other code can arise on an empty host. The empty delta is admitted on every one of the 16 boards of the 2x2 cells. So the gate's
  default is "assume nothing; create, never remove": it neither guesses a missing place nor fills it in.
What they do NOT show: that the page's claims about any named product's behaviour are true (not used); that a model's refusals are codes (the gate refuses a DELTA, it does not speak for a model); that "no self" is
the right design for a refusal text (a human reader may need more than a code - `diagnostics.propose_repair` adds a repair proposal, which is not tested here); that three pairs of the Post 399 default ("assume, don't ask" and
two more) correspond to the gate's checks (a loose reading, not tested). The list of banned words is mine and short.
The fifth check is ENGINEERING SUPPORT (no page): two rules of the gate that can never decide an outcome, found by mutation (see its docstring)."""
import os, random, re, sys
from itertools import product
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.cico_parser import CICOParser
from axi.engine.gate import Graph, check, apply
from axi.engine.runtime import Runtime
from test_signed_slots import etext, same, copy, random_graph, random_delta_text

CODES = ("match", "identification", "dangling edge", "well-formedness")
SELF_WORDS = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "sorry", "afraid", "unfortunately", "please", "feel", "feels", "feeling", "feelings", "hurt", "angry", "upset", "apologize", "regret"}


def words(text): return set(re.findall(r"[a-z']+", text.lower()))


def refusals(n_tries, seed):
    rng, out = random.Random(seed), []
    for _ in range(n_tries):
        g = random_graph(rng)
        d = CICOParser.parse_transition_delta(random_delta_text(g, rng))
        r = check(g, d)
        if not r.ok: out.append((g, d, r.reason))
    return out


def test_every_refusal_reason_starts_with_one_of_four_codes_and_carries_no_self():
    found = refusals(6000, 14)
    assert len(found) > 1000
    seen = set()
    for g, d, reason in found:
        code = next((c for c in CODES if reason.startswith(c + ": ")), None)
        assert code is not None, reason
        seen.add(code)
        assert len(reason) <= 120, reason
        assert not (words(reason) & SELF_WORDS), reason
    assert seen == set(CODES)
    assert not (words("the host has no such place") & SELF_WORDS)
    assert words("I am sorry, I feel hurt") & SELF_WORDS == {"i", "sorry", "feel", "hurt"}      # the detector does find a voice when there is one


def test_a_refusal_is_a_pure_function_the_same_every_time_and_leaves_no_mark():
    found = refusals(3000, 5)
    for g, d, reason in found[:25]:
        before = copy(g)
        for _ in range(1000):
            assert check(g, d).reason == reason
        assert same(g, before)
    g = Graph()
    bad = CICOParser.parse_transition_delta("DEL[3,3]")
    good = CICOParser.parse_transition_delta("ADD[3,3:1]")
    assert all(check(g, bad).reason == "match: node (3, 3) not in host" for _ in range(1000))
    assert check(g, good).ok                                                      # a thousand refusals later the next valid delta is admitted
    apply(g, good)
    assert g.nodes == {(3, 3): 1}


def test_the_runtime_refuses_bad_input_at_the_right_stage_without_scolding_and_the_next_proposal_commits():
    rt = Runtime(64)
    cases = [("hello there", "parse"), ("ADD[1,2]", "parse"), ("DEL[0,0]", "gate"), ("ADD[0,0:1] ADD[0,0:2]", "gate"), (etext("ADD", ((0, 0), (0, 1), "r"), 1), "gate")]
    for text, stage in cases:
        out = rt.propose(text)
        assert not out.committed and out.stage == stage, (text, out)
        assert not (words(out.reason) & SELF_WORDS), out.reason
        assert rt.graph.nodes == {} and rt.graph.edges == {}
    rt.pager.audit()
    assert rt.pager.free_blocks() == 64                                          # refusals took no blocks
    ok = rt.propose("ADD[0,0:1]")
    assert ok.committed and ok.stage == "commit" and rt.graph.nodes == {(0, 0): 1}


def test_on_the_empty_host_the_gate_admits_exactly_21_of_729_deltas_all_self_contained_additions():
    cells = [(0, 0), (0, 1)]
    pairs = [(u, v) for u in cells for v in cells]
    admitted, reasons, total = [], {}, 0
    for nops in product((0, 1, -1), repeat=2):
        for eops in product((0, 1, -1), repeat=4):
            items = []
            for c, o in zip(cells, nops):
                if o == 1: items.append(f"ADD[{c[0]},{c[1]}:0]")
                if o == -1: items.append(f"DEL[{c[0]},{c[1]}]")
            for (u, v), o in zip(pairs, eops):
                if o == 1: items.append(etext("ADD", (u, v, "r"), 1))
                if o == -1: items.append(etext("DEL", (u, v, "r"), 1))
            d = CICOParser.parse_transition_delta(" ".join(items))
            r = check(Graph(), d)
            total += 1
            if r.ok: admitted.append((nops, eops, items))
            else: reasons[r.reason.split(":")[0]] = reasons.get(r.reason.split(":")[0], 0) + 1
    assert total == 729 and len(admitted) == 21 == 1 + 4 + 16
    assert all(o != -1 for nops, eops, _ in admitted for o in nops + eops)         # no DEL is ever admitted on an empty host
    for nops, eops, _ in admitted:                                                  # every added line has both of its ends added in the same delta
        for (u, v), o in zip(pairs, eops):
            if o == 1: assert nops[cells.index(u)] == 1 and nops[cells.index(v)] == 1
    assert reasons == {"match": 665, "well-formedness": 43} and sum(reasons.values()) == 708
    for state in range(16):                                                         # the empty delta is admitted on every board
        g = Graph()
        for i, c in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
            if state >> i & 1: g.add_node(c, 0)
        assert check(g, CICOParser.parse_transition_delta("")).ok


def test_adding_and_deleting_the_same_place_or_line_in_one_delta_is_refused_by_an_earlier_rule_on_every_host():
    """Found by a mutation of the gate: the two rules "node both added and deleted" and "edge both added and deleted" in gate.py can be switched off without any of the 60 test files noticing. The reason is that they never decide an
    outcome: an ADD needs the key absent from the host and a DEL needs it present, so one of the two earlier rules refuses first, on every host and in either order. Counted here for a node and a line, host with and without the key,
    both item orders: eight cases, each refused, none with the words "both added and deleted". (A dead rule is harmless; it is recorded, not removed.)"""
    cases = 0
    for has in (False, True):
        for order in (0, 1):
            g = Graph()
            g.add_node((0, 0), 0); g.add_node((0, 1), 0)
            if has: g.add_edge((0, 0), (0, 1), "r", 1)
            node_g = Graph()
            if has: node_g.add_node((5, 5), 0)
            items = [("ADD[5,5:1]", "DEL[5,5]"), ("ADD[(0,0)->(0,1):r#1]", "DEL[(0,0)->(0,1):r#1]")]
            for host, (a, d) in zip((node_g, g), items):
                text = " ".join((a, d) if order == 0 else (d, a))
                r = check(host, CICOParser.parse_transition_delta(text))
                assert not r.ok and "both added and deleted" not in r.reason, (text, r.reason)
                assert r.reason.split(" ")[0] == ("identification:" if has else "match:"), (text, r.reason)
                cases += 1
    assert cases == 8


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
