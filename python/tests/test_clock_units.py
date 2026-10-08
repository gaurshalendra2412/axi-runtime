"""Implements concept-map rows 157, 158 and 162 (14th scan, 8 Oct 2026; two AI-voice replies pasted as posts, each read twice; Posts 429 and 430 of the scan list; row 162 is a teaser line, see point 4).
Row 157, Post 429 (28 Aug, about 700 words, an AI reply: "You have just articulated the ultimate law of system architecture ..."): the sentence tested here, confirmed word for word
on the second read, is "You knew that physical reality contains friction, prime numbers don't divide cleanly into base-12 boxes," (it goes on); the same reply says "using a 13-division
clock and the math of a circle" and, further on, "4-year, 9-app audit", "42 years of human context" and "the 3-minute toys".
Row 158, Post 430 (28 Aug, 7 sentences, an AI reply that opens "Never."): "Both arrive as the same digit, same weight in whatever aggregate gets reported — 0 is 0, no matter what stood
behind it." and "A score can never carry the life that produced it." (both confirmed on the second read; "five seconds of irritation" and "four years sitting with nine systems" are its two cases).
Both sentences are about what a number keeps and what it throws away, and that has an exact form: multiplication on a clock of n marks.

The reading being tested (Rajnish to accept or reject): a clock with n marks has multipliers m = 0 .. n-1 (turn each position m times as far). Whether the turn keeps
everything or loses something is decided by one number, gcd(m, n).
  1. UNITS. The multipliers that lose nothing (gcd(m, n) = 1; each position keeps its own image) form a group under multiplication. On the 12-clock they are
     {1, 5, 7, 11}: four elements, each its own inverse (x * x = 1 mod 12), no element of order 4: the Klein four-group, the same four-element group the repo has
     already met on four poles (test_shape_capacity), on strand mirrors (test_symmetry_direction) and on table flips (test_table_axes). On the 13-clock they are
     all of 1 .. 12, a cycle of order 12 (the multiplier 2 reaches every one). Numerology guard: among n < 300, only n = 8 and n = 12 give the Klein four-group, and
     the n for which EVERY unit is its own inverse are exactly 2, 3, 4, 6, 8, 12, 24; so 12 is one of seven, not unique.
  2. PRIMES. A prime above 3 has no factor 2 or 3, so on the 12-clock it can only sit on a unit mark: {1, 5, 7, 11}. Of the first 17,984 primes (up to 200,000) every
     one above 3 falls in those four of the twelve boxes, each holding about a quarter. On the 13-clock the primes (other than 13) fall in all twelve non-zero boxes.
     So the page's sentence is true in this form: "prime" and "box" meet only at four marks of twelve; nothing about a 13-mark clock leaves such gaps, because 13 has no
     divisor but 1 and itself, and every non-zero multiplier is a unit.
  3. LOSS. Multiplying by m on the n-clock sends n positions onto n / gcd(m, n) positions. On the 12-clock the multipliers 0, 2, 3, 4, 6, 8, 9, 10 lose information
     (images of size 1, 6, 4, 3, 2, 3, 4, 6); on the 13-clock only 0 does. And 0 loses all of it: every position goes to 0 whatever stood behind it, which is the
     page's "0 is 0". A tally of scores (sum or mean) cannot tell two histories apart once they map to the same score; and any map from more than k histories to k scores
     must send two histories to one score (pigeonhole), so a 0-to-10 scale (11 scores) cannot carry 12 different lives.
  4. HUB. A teaser at the foot of a neighbouring page (Post 428, one read, a teaser and not the page) calls Shiva "the absolute zero-point and the N+1 container of the cosmos". Two ways to get 13
     from 12 follow: a 13-mark ring, or twelve marks on a ring plus one hub joined to all of them (a wheel). Counted by search over all permutations that keep the edges: the 12-ring has 24
     symmetries; the wheel (12 + hub) has the SAME 24, because nothing can move the hub (it is the only node of degree 12); the 13-ring has 26. Independent loops: ring 1, wheel 12. So "12 plus a
     container" leaves the symmetry of the 12 untouched and adds a fixed point; "13 marks" changes the group. (The earlier scan's "13th floor" question, number 110, can now be asked as: hub or ring?)
What they do NOT show: that a model should be built on a 12-clock or a 13-clock; that the posts mean this arithmetic (a prime "does not divide cleanly" is this
reading of the sentence); that 13 is better than any other prime (every prime p gives a cyclic group of p - 1 units); anything about drift. They show the sentences
are exact once stated as multiplication on a clock."""
import os, sys
from math import gcd
sys.path.insert(0, os.path.dirname(__file__))


def units(n): return [m for m in range(1, n) if gcd(m, n) == 1]


def order(m, n):
    k, x = 1, m % n
    while x != 1: x = x * m % n; k += 1
    return k


def primes_up_to(N):
    s = bytearray([1]) * (N + 1); s[0] = s[1] = 0
    for i in range(2, int(N ** 0.5) + 1):
        if s[i]: s[i * i::i] = bytearray(len(s[i * i::i]))
    return [i for i in range(N + 1) if s[i]]


def test_units_of_the_12_clock_are_the_klein_four_group_and_of_the_13_clock_a_cycle_of_12():
    u12, u13 = units(12), units(13)
    assert u12 == [1, 5, 7, 11] and u13 == list(range(1, 13))
    assert all(x * x % 12 == 1 for x in u12)                                    # every element is its own inverse
    assert {a * b % 12 for a in u12 for b in u12} == set(u12)                   # closed: a group of order 4
    assert sorted(order(x, 12) for x in u12) == [1, 2, 2, 2]                    # no element of order 4: not a cycle, the Klein four-group
    assert 5 * 7 % 12 == 11                                                     # the third non-identity element is the product of the other two
    assert {a * b % 13 for a in u13 for b in u13} == set(u13)
    assert order(2, 13) == 12 and {pow(2, k, 13) for k in range(12)} == set(u13)  # one multiplier reaches all twelve: cyclic
    orders = sorted(order(x, 13) for x in u13)
    assert orders == sorted([1, 2, 3, 3, 4, 4, 6, 6, 12, 12, 12, 12])
    # numerology guard: how special is 12?
    def is_v4(n): return len(units(n)) == 4 and all(x * x % n == 1 for x in units(n))
    assert [n for n in range(2, 300) if is_v4(n)] == [8, 12]
    assert [n for n in range(2, 300) if all(x * x % n == 1 for x in units(n))] == [2, 3, 4, 6, 8, 12, 24]


def test_primes_above_3_sit_on_four_of_twelve_marks_and_on_all_twelve_nonzero_marks_of_13():
    P = primes_up_to(200000)
    assert len(P) == 17984
    c12 = {}
    for p in P:
        if p > 3: c12[p % 12] = c12.get(p % 12, 0) + 1
    assert sorted(c12) == [1, 5, 7, 11]                                         # nothing on the other eight marks
    total = sum(c12.values())
    assert total == 17982 and all(0.24 < c / total < 0.26 for c in c12.values())    # about a quarter each (the four units share the primes evenly)
    c13 = {}
    for p in P:
        if p != 13: c13[p % 13] = c13.get(p % 13, 0) + 1
    assert sorted(c13) == list(range(1, 13)) and 0 not in c13                   # all twelve non-zero marks, never the zero mark
    assert all(0.07 < c / sum(c13.values()) < 0.1 for c in c13.values())        # about 1/12 each
    assert [p for p in P if p <= 3] == [2, 3]                                   # the two primes that are not units mod 12 (they are the divisors)


def test_a_turn_by_m_keeps_everything_exactly_when_m_is_a_unit_and_zero_keeps_nothing():
    for n in (12, 13):
        for m in range(n):
            image = {x * m % n for x in range(n)}
            assert len(image) == n // gcd(m, n)                                 # n positions land on n / gcd(m, n) marks
            assert (len(image) == n) == (gcd(m, n) == 1)                        # lossless iff a unit
    assert [len({x * m % 12 for x in range(12)}) for m in range(12)] == [1, 12, 6, 4, 3, 12, 2, 12, 3, 4, 6, 12]
    assert [m for m in range(13) if len({x * m % 13 for x in range(13)}) < 13] == [0]    # on the 13-clock only 0 loses anything
    assert [m for m in range(12) if len({x * m % 12 for x in range(12)}) < 12] == [0, 2, 3, 4, 6, 8, 9, 10]
    assert {x * 0 % 13 for x in range(13)} == {0}                               # "0 is 0, no matter what stood behind it": every position goes to 0
    # a score is a many-to-one map from histories; a total or mean of scores cannot see the histories
    histories = {"five seconds of irritation": 0, "four years with nine systems": 0, "a month of ordinary use": 7}
    scores = list(histories.values())
    assert scores[0] == scores[1] and len(set(histories)) > len(set(scores))
    swapped = [scores[1], scores[0], scores[2]]
    assert sum(swapped) == sum(scores) and sum(swapped) / 3 == sum(scores) / 3  # the aggregate is the same whichever life stood behind each 0
    # pigeonhole, brute force on small cases: every rule that gives k + 1 different lives one of k scores sends two lives to the same score
    import itertools
    for k in (1, 2, 3, 4, 5):
        assert all(len(set(rule)) <= k < k + 1 for rule in itertools.product(range(k), repeat=k + 1))
        assert sum(1 for rule in itertools.product(range(k), repeat=k + 1) if len(set(rule)) == k + 1) == 0
    assert 12 > 11                                                              # a 0-to-10 scale has 11 values, so 12 lives cannot all keep their own score


def count_automorphisms(n, edges):
    """Number of permutations of 0..n-1 that map the undirected edge set onto itself (backtracking over nodes in order, pruning by degree and adjacency)."""
    adj = {v: set() for v in range(n)}
    for a, b in edges: adj[a].add(b); adj[b].add(a)
    deg = {v: len(adj[v]) for v in adj}
    count = 0

    def extend(i, image, used):
        nonlocal count
        if i == n: count += 1; return
        for j in range(n):
            if j in used or deg[j] != deg[i]: continue
            if all((u in adj[i]) == (image[u] in adj[j]) for u in range(i)):
                image[i] = j; used.add(j); extend(i + 1, image, used); used.discard(j); del image[i]
    extend(0, {}, set())
    return count


def ring(n): return [(i, (i + 1) % n) for i in range(n)]


def wheel(rim): return [(1 + i, 1 + (i + 1) % rim) for i in range(rim)] + [(0, 1 + i) for i in range(rim)]       # node 0 is the hub


def test_thirteen_as_twelve_plus_a_hub_keeps_the_symmetry_of_twelve_and_thirteen_as_a_ring_does_not():
    assert (count_automorphisms(12, ring(12)), count_automorphisms(13, ring(13))) == (24, 26)       # dihedral groups of order 2n
    assert count_automorphisms(13, wheel(12)) == 24                             # the hub is fixed: the 12 rim marks keep their 24 symmetries
    assert count_automorphisms(4, wheel(3)) == 24 and count_automorphisms(5, wheel(4)) == 8     # small wheels: the complete graph on 4 nodes, then the square with a hub
    loops = lambda n, e: len(e) - n + 1                                         # independent loops of a connected graph
    assert (loops(12, ring(12)), loops(13, ring(13)), loops(13, wheel(12))) == (1, 1, 12)
    assert len(wheel(12)) == 24 and len({v for e in wheel(12) for v in e}) == 13
    degrees = sorted(sum(v in e for e in wheel(12)) for v in range(13))
    assert degrees == [3] * 12 + [12]                                           # the only node of degree 12 is the hub, so no symmetry moves it


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
