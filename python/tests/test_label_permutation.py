"""Implements concept-map row 53 (blog post of 29 Aug, "the 13-goddess problem"): an image generator was asked for 13 labelled
portraits on a circle; the post says it fitted 12, then "skipped the number 5" and "duplicated 9", and states that 360/13 is
about 27.69 degrees per portrait.

What this pins (sandbox, no model involved) is a SCOPE statement about our own gate, which a careful reader will ask for:
  * the gate catches writing into a slot that is already filled (identification) - a duplicated SLOT;
  * the gate does NOT catch the same LABEL in two different slots, and does NOT catch a slot that is never filled. Those are
    task invariants (labels form a permutation of 1..n), not applicability conditions. A separate invariant check is needed,
    and the test shows a ten-line one catches both failures named in the post.
  * the arithmetic in the post is right (360/13 = 27.69...), and a program that places item i at 360*i/n degrees cannot skip or
    repeat a slot, because it has a counter.
What it does NOT show: anything about image models (the generated image is not in the repo and was not seen). The 'skipped 5,
duplicated 9' failure is modelled here as a label sequence, which is my reading of the post, not the post's data.
Design consequence (stated, not built): completeness and uniqueness belong in a task-level invariant checked at commit, next to
the gate; M6 'complete' only closes dangling edges and does not do this."""
import math
from collections import Counter
from axi.engine.cico_parser import CICOParser
from axi.engine.gate import Graph, check
from axi.experiments.agent_loop import blind_apply

N = 13


def delta_for(labels):
    """One ADD per slot: slot k (cell (0,k)) shows goddess number labels[k]."""
    return CICOParser.parse_transition_delta(" ".join(f"ADD[0,{k}:{v}]" for k, v in enumerate(labels)))


def permutation_report(g, n):
    """Task invariant: the labels on the n slots are exactly 1..n, each once. Returns (missing, duplicated, empty_slots)."""
    vals = [g.nodes.get((0, k)) for k in range(n)]
    empty = [k for k, v in enumerate(vals) if v is None]
    cnt = Counter(v for v in vals if v is not None)
    missing = sorted(set(range(1, n + 1)) - set(cnt))
    dup = sorted(v for v, c in cnt.items() if c > 1)
    return missing, dup, empty


# the failure as the post describes it: 12 fitted, then the 13th goes wrong: 5 is skipped, 9 appears twice
POST_FAILURE = [1, 2, 3, 4, 6, 7, 8, 9, 9, 10, 11, 12, 13]


def test_correct_layout_is_admitted_and_is_a_permutation():
    d = delta_for(list(range(1, N + 1)))
    assert check(Graph(), d).ok
    g = blind_apply(Graph(), d)
    assert permutation_report(g, N) == ([], [], [])


def test_gate_catches_a_filled_slot_written_twice():
    # same slot twice inside one delta
    d = CICOParser.parse_transition_delta("ADD[0,4:5] ADD[0,4:6]")
    r = check(Graph(), d); assert not r.ok and "identification" in r.reason, r
    # a later delta writing into a slot that is already filled
    g = blind_apply(Graph(), delta_for(list(range(1, N + 1))))
    r = check(g, CICOParser.parse_transition_delta("ADD[0,4:99]")); assert not r.ok and "already exists" in r.reason, r


def test_gate_does_not_catch_the_same_label_in_two_slots_or_a_skipped_label():
    d = delta_for(POST_FAILURE)
    assert check(Graph(), d).ok                        # every write is individually admissible: that is the point
    g = blind_apply(Graph(), d)
    assert g.is_well_formed()
    missing, dup, empty = permutation_report(g, N)
    assert missing == [5] and dup == [9] and empty == []   # label 5 skipped, label 9 doubled, all 13 slots filled


def test_gate_does_not_catch_an_unfilled_slot():
    d = delta_for(list(range(1, N)))                   # only 12 portraits placed
    assert check(Graph(), d).ok
    g = blind_apply(Graph(), d)
    assert permutation_report(g, N) == ([13], [], [12])


def test_the_invariant_check_is_what_catches_both():
    for labels in (POST_FAILURE, list(range(1, N))):
        g = blind_apply(Graph(), delta_for(labels))
        assert permutation_report(g, N) != ([], [], [])


def test_arithmetic_in_the_post_and_a_counter_cannot_skip():
    assert round(360 / 13, 2) == 27.69
    assert 360 / 12 == 30 and 360 / 9 == 40
    for n in (9, 12, 13):
        angles = [360 * i / n for i in range(n)]
        assert len(set(angles)) == n                              # no slot repeated
        gaps = {round(b - a, 9) for a, b in zip(angles, angles[1:] + [360.0])}
        assert gaps == {round(360 / n, 9)}                        # all gaps equal: evenly spaced
        assert all(math.isclose(a, 360 * i / n) for i, a in enumerate(angles))


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
