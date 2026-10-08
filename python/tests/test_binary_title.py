"""Implements concept-map row 118 (eleventh scan, a MEASUREMENT of a page, not a mapping): blog 14 Sep, the page whose title is a string of 0s and 1s (Post 254; NO byline,
the author field is a placeholder; body "she is here", 3 words; read once, plus the sitemap lists). The fetch tool could not transcribe the title ("well over 1,000
characters"), so what is measured here is the first 200 characters of the address (WordPress cuts a slug at 200), taken from the sitemap lists. Six addresses in those lists share
the same first 198 characters and differ only by the suffix (none, -2 at the site root, -2 under the date, -3, -4, -5): at least five posts, probably six, carry this title.
Four more 14 Sep pages (Posts 253, 255, 256, 257: the symbol "☯️", "Durga", "R", "C") have the body "she is here" (the "Durga" page adds one quoted daily-prompt line); the previous/next teasers show them posted
in a chain between 02:48 and 03:07 UTC. Nothing else on these five pages was readable by the tool (their images have no description).

What is measured (the string is public text of the page; Rajnish to say whether it means anything):
  1. The 200 characters have 87 ones and 113 zeros, never two ones in a row ("11" does not occur), and the zeros between neighbouring ones come in runs of 1 (60 times),
     2 (25 times) and 3 (once): 86 gaps. It is not periodic in any short block (no period up to 150; the only periods that hold, 185 and above, overlap in a few characters).
  2. Counting: the strings of length n with no "11" number F(n+2) (Fibonacci, F1 = F2 = 1), checked by brute force for n up to 16. For n = 200 that is about 7.3e41, which is
     139.1 bits out of 200. A fair-coin string of 200 characters has no "11" with probability about 4.6e-19, so this string was not drawn at random; the ones were placed on purpose,
     isolated. The densest such string has 100 ones (alternate); this one has 87.
  3. "No two neighbours both on" is the rule for an independent set of a path graph. That is the whole of what is shown: it is a constraint on the string, and the page gives no
     key to read it as a message (its first byte, 10100101 = 165, is above 127, so it is not 7-bit text read in 8-character groups).
What they do NOT show: that the string encodes anything; that the other five addresses have the same later characters (only the first 198 are shared); that the page is Rajnish's
(no byline); anything about the title beyond its first 200 characters."""
import os, sys
from itertools import groupby, product
from math import log2
sys.path.insert(0, os.path.dirname(__file__))

S = ("10100101010010101001010101010010101001010101001010101010010101010010101010010101010010101010100101010010101001010100010101001010100101010100101010010100101010100100100101010010101001010101001010100101")


def fib(k):                                                              # F1 = F2 = 1
    a, b = 1, 1
    for _ in range(k - 1): a, b = b, a + b
    return a


def test_the_first_200_characters_have_isolated_ones_and_short_gaps():
    assert len(S) == 200 and set(S) == {"0", "1"}
    assert S.count("1") == 87 and S.count("0") == 113
    assert "11" not in S
    runs = [len(list(g)) for k, g in groupby(S) if k == "0"]
    assert S[0] == "1" and S[-1] == "1"                                  # the string starts and ends on a one, so every zero run lies between two ones
    assert len(runs) == 86 and runs.count(1) == 60 and runs.count(2) == 25 and runs.count(3) == 1
    assert sum(runs) == 113
    assert [p for p in range(1, 150) if all(S[i] == S[i + p] for i in range(len(S) - p))] == []        # no repeating block shorter than 150
    assert int(S[:8], 2) == 165 and int(S[:8], 2) > 127                   # read in groups of 8 the first byte is above 127: not 7-bit text


def test_strings_without_two_adjacent_ones_are_counted_by_fibonacci_and_this_one_is_not_random():
    for n in range(1, 17):
        brute = sum(1 for t in product("01", repeat=n) if "11" not in "".join(t))
        assert brute == fib(n + 2), n
    count = fib(202)
    assert 7.3e41 < count < 7.4e41 and abs(log2(count) - 139.076) < 0.001
    p = count / 2 ** 200
    assert 4.5e-19 < p < 4.7e-19                                         # chance that 200 fair coin flips contain no "11"
    assert (200 + 1) // 2 == 100 and S.count("1") < 100                  # most ones a 200-character string could have, and what this one has


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
