"""Implements concept-map row 102 (tenth scan): blog 4 Sep "Averaging ten thousand distinct quotes doesn't yield a profound universal truth, it
collapses every contradiction of human ..." (two reads: "Philosophically, however, averaging ten thousand quotes is the ultimate bureaucratic trap.";
"The outliers --- the massive multi-volume religious texts and the sprawling imperial manifestos --- cancel out"; "pulling the distribution toward a
predictable mean of roughly 14 to 15 words per statement", which is a PREDICTION on the page, no dataset), and blog 13 Sep "You've hit on the exact
flaw ..." (Post 40, re-read: "our numbers and units divide the infinite into neat little pieces"). Related repository idea: Shiva = canonical form
(concept-map row 6, FRAMEWORK_SYNTHESIS section 2).

The reading being tested (Rajnish to accept or reject): there are two ways to "forget" the frame of a picture. AVERAGING it (take the mean) and
CANONICALIZING it (pick one representative of everything that differs only by the frame's symmetry). They throw away very different amounts. On the
3x3 board with nine distinct labels (counted by brute force over all 362,880 arrangements, nothing assumed):
    all arrangements                      362,880   (18.47 bits)
    up to rotation only                    90,720
    up to the 8 symmetries (canonical)     45,360   (15.47 bits)   <- nothing but the frame is forgotten
    after averaging over the 8 symmetries     167   ( 7.38 bits)   <- (centre label, sum of the four corners) is all that survives
    the mean over every arrangement             1   ( 0 bits)      <- every cell is exactly 5
  The averaged board (the "Reynolds operator" of the group) is constant on each cell orbit: four equal corners, four equal edges, a centre.
  So "average away the differences" is not a smaller version of "canonicalize"; it deletes the arrangement itself, and canonicalization deletes
  only the orientation.
Also, on the page's arithmetic: outliers do NOT cancel in a mean, they move it. One 10,000-word text among 9,999 fourteen-word quotes lifts the
mean from 14 to 14.9986, which is "14 to 15 words" with no averaging magic at all; cancelling that single outlier would take 769 one-word quotes
(word counts cannot go below 1). The 14-15 figure itself is untested here: there is no dataset.
What they do NOT show: that any model averages or canonicalizes this way; anything about the 10,000 quotes (we do not have them); that the page means
the group average by "averaging"."""
import math, os, sys
from fractions import Fraction
from itertools import permutations
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.d4 import apply as d4apply, IDS

ALL = np.array(list(permutations(range(1, 10))), dtype=np.int64)        # all 362,880 arrangements of labels 1..9 on cells 0..8 (row major)
W = 10 ** np.arange(8, -1, -1, dtype=np.int64)                          # base-10 key: the 9 digits as one number
CORNERS, EDGES, CENTRE = [0, 2, 6, 8], [1, 3, 5, 7], 4


def perm_list(ids):
    idx = np.arange(9).reshape(3, 3)
    return [np.array(d4apply(idx, t).ravel()) for t in ids]            # new position j shows old cell p[j]


def classes(ids):
    keys = np.stack([ALL[:, p] @ W for p in perm_list(ids)])           # key of the arrangement as seen after each transform
    return len(np.unique(keys.min(axis=0)))                            # the smallest key is the representative of its class


def test_nothing_but_the_frame_is_forgotten_by_canonicalizing():
    assert len(ALL) == math.factorial(9) == 362_880
    assert classes((0,)) == 362_880                                     # the identity alone: no merging
    assert classes((0, 1, 2, 3)) == 90_720                              # rotations only
    assert classes(IDS) == 45_360                                       # all eight symmetries
    assert abs(math.log2(45_360) - 15.469) < 0.001


def test_averaging_over_the_symmetries_keeps_only_centre_and_corner_sum():
    keep = {(int(a[CENTRE]), int(a[CORNERS].sum())) for a in ALL}
    assert len(keep) == 167 and abs(math.log2(167) - 7.384) < 0.001
    rng = np.random.default_rng(7)
    for _ in range(50):                                                 # the true group average equals the formula built from (centre, corner sum)
        a = ALL[int(rng.integers(len(ALL)))].reshape(3, 3)
        avg = sum(d4apply(a, t).astype(float) for t in IDS) / 8
        avg_rot = sum(d4apply(a, t).astype(float) for t in IDS[:4]) / 4      # rotations alone give the same averaged board: on 3x3 the cell orbits are the same
        assert np.allclose(avg, avg_rot)                                       # (reflections halve the CLASS count, 90,720 to 45,360, but add nothing to the average)
        c, s = a[1, 1], a[0, 0] + a[0, 2] + a[2, 0] + a[2, 2]
        assert np.allclose(avg.ravel()[CORNERS], s / 4) and np.allclose(avg.ravel()[EDGES], (45 - c - s) / 4) and avg[1, 1] == c
    # the arrangement is gone: many canonical classes share one averaged board (every class sits under exactly one pair)
    canon = np.stack([ALL[:, p] @ W for p in perm_list(IDS)]).min(axis=0)
    pair = ALL[:, CENTRE] * 100 + ALL[:, CORNERS].sum(axis=1)
    per_pair = {}
    for k, c in zip(pair.tolist(), canon.tolist()): per_pair.setdefault(k, set()).add(c)
    assert len(per_pair) == 167 and sum(len(v) for v in per_pair.values()) == 45_360          # classes are not split between pairs
    assert max(len(v) for v in per_pair.values()) > 400 and min(len(v) for v in per_pair.values()) >= 1


def test_the_mean_over_every_arrangement_is_the_same_number_in_every_cell():
    assert (ALL.sum(axis=0) == 5 * math.factorial(9)).all()             # exactly 5 in all nine cells: one value, 0 bits
    assert 362_880 > 90_720 > 45_360 > 167 > 1


def test_outliers_move_a_mean_they_do_not_cancel_in_it():
    n, typical, outlier = 10_000, 14, 10_000
    mean = Fraction((n - 1) * typical + outlier, n)
    assert mean == Fraction(149_986, 10_000) and abs(float(mean) - 14.9986) < 1e-9                  # one outlier takes 14 to 15
    ones = Fraction(outlier - typical, typical - 1)                    # one-word quotes needed to cancel it (word counts cannot go below 1)
    assert math.ceil(ones) == 769 and abs(float(ones) - 768.15) < 0.01
    sample = [typical] * (n - 1) + [outlier]
    assert sorted(sample)[n // 2] == typical                            # the median does not move at all


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
