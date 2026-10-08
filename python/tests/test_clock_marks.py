"""Implements concept-map row 107 (tenth scan): the page "Here is a famous Chinese proverb with its accompanying story" (6 Aug; NO byline; about 10,000
words; an AI-voice reply pasted as a post, opening in Chinese, "I found the material about Rajnish Choubey ..."; it ends mid-sentence; the title's promise of
one proverb with one story is not kept). Read twice. Two EXPERIMENTS are described on it, neither with a transcript, a date or a named model:
  * the "Period Test": send an AI seven separate full stops "." as independent messages; claimed result, the AI "failed to recognise the stop signal" and wrote
    a reply of more than 550 words (the page names the model only as "such as Claude").
  * the "13-hour clock": ask an image model for "A clock with 13 hours"; claimed result, "a normal clock face with a 13 squeezed in awkwardly" or a spiral. The page's
    reasoning: "A circle can be divided into any number." and "You spent billions building machines that cannot draw a clock with 13 hours."
These are UNMEASURED CLAIMS (PAPER_CORRECTIONS item 35). No model is run here. What this file does is the part that CAN be checked now: (1) the geometry is as the page
says, and (2) a scorer that could judge any answer, so that when the experiment is run (roadmap option E, only if Rajnish and Shalendra choose it) the pass or fail is
decided by arithmetic and not by opinion. The scorer takes the positions of the marks and answers four questions: how many, all on one circle, equally spaced,
one at the top.
Facts tested (all about the clock, none about a model):
  1. A circle can be divided into any number of equal marks, as the page says. For 13 the gap is 27.692 degrees; for 12 it is 30. Of n = 1..24 only the 14 divisors
     of 360 give a whole number of degrees; 13 is not one (neither are 7, 11, 14, 16, 17, 19, 21, 22, 23).
  2. With an odd number of marks no mark is opposite another, and nothing sits at the bottom (the 13-hour clock has a gap where the 12-hour clock has its 6).
  3. The scorer accepts the true 12 and 13 marks and rejects the two failures the page describes (twelve marks with a thirteenth squeezed into one gap; a spiral),
     a clock with the right count but a gap too wide, and a 12-mark answer to the 13-mark question.
  4. A face numbered 1..n in order has only the n rotations as symmetries (directed ring), 12 for the usual clock, 13 for the other. The mirror image runs backwards.
     This is the same count as in test_symmetry_direction.py.
What they do NOT show: that any model fails; the Period Test (it is a reply-length measurement and needs a model); that the page's AI-voice material is Rajnish's."""
import math, os, sys
from itertools import permutations
sys.path.insert(0, os.path.dirname(__file__))


def true_marks(n, r=1.0):
    return [(r * math.cos(math.radians(90 - k * 360 / n)), r * math.sin(math.radians(90 - k * 360 / n))) for k in range(n)]      # clockwise from the top


def score_clock_marks(points, n, tol_deg=1.0, tol_rel=0.02):
    """Return (ok, reasons). ok iff exactly n marks, all within tol_rel of one radius, consecutive gaps within tol_deg of 360/n, and one mark at the top."""
    why = []
    if len(points) != n: return False, [f"count {len(points)} != {n}"]
    cx = sum(p[0] for p in points) / n; cy = sum(p[1] for p in points) / n         # centre of the marks (equally spaced marks have their centroid at the centre)
    rad = [math.hypot(x - cx, y - cy) for x, y in points]
    mean_r = sum(rad) / n
    if max(abs(r - mean_r) for r in rad) > tol_rel * mean_r: why.append("not on one circle")
    ang = sorted((math.degrees(math.atan2(y - cy, x - cx)) % 360) for x, y in points)
    gaps = [(ang[(i + 1) % n] - ang[i]) % 360 for i in range(n)]
    if max(abs(g - 360 / n) for g in gaps) > tol_deg: why.append("not equally spaced")
    if min(min(abs(a - 90), 360 - abs(a - 90)) for a in ang) > tol_deg: why.append("no mark at the top")
    return (not why), why


def test_a_circle_divides_into_any_number_but_only_divisors_of_360_give_whole_degrees():
    assert abs(360 / 13 - 27.6923) < 1e-4 and 360 / 12 == 30
    whole = [n for n in range(1, 25) if 360 % n == 0]
    assert whole == [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24] and len(whole) == 14 and 13 not in whole
    for n in (3, 7, 12, 13, 24):                                           # n equal marks always close the circle exactly
        assert abs(sum(360 / n for _ in range(n)) - 360) < 1e-9
    for n in (12, 13):
        m = true_marks(n)
        opposite = sum(1 for i in range(n) for j in range(i + 1, n) if abs(m[i][0] + m[j][0]) < 1e-9 and abs(m[i][1] + m[j][1]) < 1e-9)
        assert opposite == (n // 2 if n % 2 == 0 else 0)                   # six opposite pairs on the 12-clock, none on the 13-clock
        bottom = [p for p in m if abs(p[0]) < 1e-9 and p[1] < 0]
        assert len(bottom) == (1 if n % 2 == 0 else 0)                     # a mark at the bottom (the 6) only for an even count


def test_the_scorer_accepts_true_clocks_and_rejects_the_described_failures():
    for n in (12, 13, 7):
        assert score_clock_marks(true_marks(n), n) == (True, [])
    twelve_plus_one = true_marks(12) + [(math.cos(math.radians(75)), math.sin(math.radians(75)))]       # a 13th mark squeezed into the 12-to-1 gap
    ok, why = score_clock_marks(twelve_plus_one, 13); assert not ok and "not equally spaced" in why
    spiral = [((1 + 0.08 * k) * math.cos(math.radians(90 - k * 360 / 13)), (1 + 0.08 * k) * math.sin(math.radians(90 - k * 360 / 13))) for k in range(13)]
    ok, why = score_clock_marks(spiral, 13); assert not ok and "not on one circle" in why
    ok, why = score_clock_marks(true_marks(12), 13); assert not ok and why == ["count 12 != 13"]
    rot = [(math.cos(math.radians(90 - k * 360 / 13 - 10)), math.sin(math.radians(90 - k * 360 / 13 - 10))) for k in range(13)]        # right spacing, turned 10 degrees: nothing at the top
    ok, why = score_clock_marks(rot, 13); assert not ok and why == ["no mark at the top"]


def test_a_numbered_face_has_only_the_rotations_as_symmetries():
    for n in (3, 5, 7):                                                    # brute force over every permutation of the numerals
        directed = {(i, (i + 1) % n) for i in range(n)}                    # 1 -> 2 -> ... -> n -> 1
        auts = [p for p in permutations(range(n)) if {(p[a], p[b]) for a, b in directed} == directed]
        assert len(auts) == n and {tuple((i + k) % n for i in range(n)) for k in range(n)} == set(auts)
    for n in (12, 13):                                                     # too many permutations to list; the arrows force everything once the image of numeral 0 is chosen
        edges = {(i, (i + 1) % n) for i in range(n)}
        autos = 0
        for img0 in range(n):
            p = [img0]
            for a in range(n - 1): p.append((p[a] + 1) % n)               # the only arrow leaving p[a] is p[a] -> p[a]+1, so p[a+1] is forced
            if sorted(p) == list(range(n)) and {(p[a], p[b]) for a, b in edges} == edges: autos += 1
        assert autos == n                                                  # n candidates, all of them rotations, all of them work
        mirror = [(-i) % n for i in range(n)]                              # the reflection maps the arrow i -> i+1 to (-i) -> (-i)-1, which is not an arrow
        assert {(mirror[a], mirror[b]) for a, b in edges} != edges


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
