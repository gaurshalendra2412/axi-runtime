"""Implements concept-map row 115 (eleventh scan): blog 13 Sep, the page "The divine leela (play) of Lord Shiva washing dishes offers profound spiritual and moral lessons
regarding true devotion and humanity. The key takeaways from this story include:" (Post 263; about 175 words; byline rajnish choubey; it reads as the summary of a video;
two reads). The body is four bold lead-ins, each with a timestamp range that points into a source video. Verified on a second read, exactly as written:
  "True Worship is Service" (16:47-16:56); "Recognizing the Divine in All" (19:39-19:47); "Humility Over Ego" (20:26-20:32); "Compassion Matters More than Wealth" (17:42-17:49).
The page does not give the video's title or length. The four ranges are NOT in increasing order on the page: the last point comes from earlier in the video than the second
and the third.
This file is about the ORDER, not about Shiva. It is a small piece of the mapping checker that concept-map item C asks for (row 90: a mapping must be told which relation
to keep; the calendar page kept order but not successor). Here the relation is order, and it can be measured.

The reading being tested (Rajnish to accept or reject):
  1. A summary that points back into a source can be checked for order by counting inversions (pairs listed in the opposite order to the source). By video time the four
     points rank 1, 3, 4, 2 as the page lists them: 2 inversions out of a possible 6. An order-preserving summary has 0, a reversed one has 6. Of the 24 possible orders of four
     points, 1 has 0 inversions and 5 have exactly 2 (the counts for four items are 1, 3, 5, 6, 5, 3, 1).
  2. Why order matters in OUR language: a log of inverses (test_undo_log.py) must be replayed newest first. Replayed in any other order, the gate refuses most of the time,
     because an inverse no longer matches the host it meets; occasionally it applies and lands in a wrong state; and when the steps happen not to touch each other another
     order works by luck. Measured on 300 seeded episodes of 4 admitted steps (24 orders each, 7,200 replays): the newest-first order restores in all 300 episodes, and no other
     order restores in all of them. A summary that keeps the points but reorders them is in this sense not the same record.
What they do NOT show: that the page meant its list as a log; that the video's own order is "the" order of the lesson (the page does not give the video); anything about the
content of the four points. The shares in item 2 depend on the random generator of deltas used by the other tests (test_signed_slots.py); they are measurements of that
distribution, not laws, so the assertions below are bounds and not exact counts."""
import itertools, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.cico_parser import CICOParser
from axi.engine.gate import check, apply
from test_signed_slots import same, copy
from test_undo_log import episode

PAGE_ORDER = ["16:47", "19:39", "20:26", "17:42"]                          # the start of each point's range, in the order the page lists the points


def seconds(stamp):
    m, s = stamp.split(":")
    return int(m) * 60 + int(s)


def inversions(seq):
    return sum(1 for i in range(len(seq)) for j in range(i + 1, len(seq)) if seq[i] > seq[j])


def ranks(stamps):
    order = sorted(range(len(stamps)), key=lambda i: seconds(stamps[i]))
    r = [0] * len(stamps)
    for rank, i in enumerate(order, 1): r[i] = rank
    return r


def order_report(stamps):
    r = ranks(stamps); n = len(r)
    return {"ranks": r, "inversions": inversions(r), "max": n * (n - 1) // 2, "preserved": inversions(r) == 0}


def test_the_pages_four_points_are_not_in_video_order_two_inversions_out_of_six():
    assert [seconds(s) for s in PAGE_ORDER] == [1007, 1179, 1226, 1062]
    rep = order_report(PAGE_ORDER)
    assert rep == {"ranks": [1, 3, 4, 2], "inversions": 2, "max": 6, "preserved": False}
    assert order_report(sorted(PAGE_ORDER, key=seconds))["inversions"] == 0
    assert order_report(sorted(PAGE_ORDER, key=seconds, reverse=True))["inversions"] == 6
    counts = [sum(1 for p in itertools.permutations(range(4)) if inversions(p) == k) for k in range(7)]
    assert counts == [1, 3, 5, 6, 5, 3, 1] and sum(counts) == 24           # one order with 0 inversions, five with exactly 2


def replay_outcome(g, texts, order):
    h = copy(g)
    for i in order:
        d = CICOParser.parse_transition_delta(texts[i])
        if not check(h, d).ok: return "refused", h
        apply(h, d)
    return "applied", h


def test_replaying_a_log_of_inverses_in_any_order_but_newest_first_fails_somewhere():
    eps = []
    for seed in range(300):
        g, states, log = episode(seed, steps=4)
        if len(log) == 4: eps.append((g, states[0], log))
    assert len(eps) > 250
    restored = {o: 0 for o in itertools.permutations(range(4))}
    shares = {"refused": 0, "restored": 0, "wrong state": 0}
    for g, start, log in eps:
        for order in restored:
            how, h = replay_outcome(g, log, order)
            if how == "refused": shares["refused"] += 1
            elif same(h, start): shares["restored"] += 1; restored[order] += 1
            else: shares["wrong state"] += 1
    n = len(eps)
    assert restored[(3, 2, 1, 0)] == n                                     # newest first: every episode
    assert all(c < n for o, c in restored.items() if o != (3, 2, 1, 0))    # every other order fails somewhere
    page_like = (0, 2, 3, 1)                                               # the page's rank pattern 1,3,4,2 read as positions in a 4-step log
    assert restored[page_like] < n
    total = sum(shares.values()); assert total == 24 * n
    assert shares["refused"] / total > 0.5                                 # most wrong orders are refused by the gate
    assert shares["wrong state"] / total < 0.05                            # silently wrong is rare, but the gate checks admissibility, not intent
    assert shares["restored"] > n                                          # beyond the one right order, commuting steps let some other orders work by luck


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
