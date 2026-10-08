"""Implements concept-map row 101 (tenth scan): blog 13 Sep "A standard three-by-three grid is transparent, you can hold the entire logic of the game in
your head ..." (Post 34, re-read, two reads): "the moment you expand it to a four-by-four or scale it" you "abandon human scale entirely and hand the board
over to brute-force computation". The page names no game, no threshold and no formula. (Sidebar teasers on the page: "Ten isn't an upgrade", "eight
surrounding gates".)

The reading being tested (Rajnish to accept or reject): is there a CLIFF between the 3x3 board and the 4x4 board? Counted exactly, for n x n boards:
  * cell orbits under the eight symmetries (D4): T(ceil(n/2)), the triangular numbers 1, 3, 3, 6, 6, 10, 10 for n = 2..8 (a quarter of the board, folded
    along its diagonal, is a triangle of cells). 3x3 and 4x4 both have three kinds of cell.
  * a cell fixed by every symmetry (a centre) exists exactly when n is odd. THIS is what changes between 3x3 and 4x4: the centre disappears.
  * lines of n cells (rows, columns, two diagonals): 2n + 2, so 8 for 3x3 and 10 for 4x4.
  * distinct arrangements of n*n distinct labels: (n*n)!. Classes under D4: (n*n)!/8 (only the identity fixes an arrangement of distinct labels).
    9! = 362,880 (18.47 bits), 16! = 20,922,789,888,000 (44.25 bits): a factor of 57,657,600 (25.78 bits).
  * there is NO cliff: the bits added by each step 2->3, 3->4, 4->5, 5->6 are 13.88, 25.78, 39.43, 54.41, growing smoothly (second differences
    11.9, 13.65, 14.98). 3->4 is not special in size. What is special is parity.
  * "ten": the cell orbits of the 7x7 and the 8x8 board number ten (T(4) = 1+2+3+4), the lines of a 4x4 board number ten (2*4+2), and the tetractys is
    1+2+3+4 = 10. These tens have three different causes (a folded quadrant; two diagonals plus rows and columns; a sum). Nothing links them. This is the
    numerology risk named in FRAMEWORK_SYNTHESIS: do not write the tens as one fact.
What they do NOT show: that humans cannot hold a 4x4 board (that is a statement about people, and no number here measures it); that the page means these
counts by "human scale" (it gives none). Nothing in the gate depends on board size: it has no board (test_board_vocabulary.py)."""
import math, os, sys
from itertools import permutations
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.d4 import apply as d4apply, IDS


def perms_of_grid(n):
    idx = np.arange(n * n).reshape(n, n)
    return [tuple(int(x) for x in d4apply(idx, t).ravel()) for t in IDS]


def orbit_count(n):
    perms, seen, k = perms_of_grid(n), set(), 0
    for i in range(n * n):
        if i in seen: continue
        k += 1; seen |= {p[i] for p in perms}
    return k


def has_centre(n):
    perms = perms_of_grid(n)
    return any(all(p[i] == i for p in perms) for i in range(n * n))


def lines_of_n(n):
    cells = [(r, c) for r in range(n) for c in range(n)]; out = set()
    for r, c in cells:
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            line = [(r + k * dr, c + k * dc) for k in range(n)]
            if all(0 <= a < n and 0 <= b < n for a, b in line): out.add(frozenset(line))
    return len(out)


def tri(k): return k * (k + 1) // 2


def test_cell_orbits_centre_and_lines_follow_closed_forms():
    for n in range(2, 9):
        assert orbit_count(n) == tri((n + 1) // 2), n
        assert has_centre(n) == (n % 2 == 1), n
        assert lines_of_n(n) == 2 * n + 2, n
    assert [orbit_count(n) for n in range(2, 9)] == [1, 3, 3, 6, 6, 10, 10]
    assert orbit_count(3) == orbit_count(4) == 3 and has_centre(3) and not has_centre(4)
    assert (lines_of_n(3), lines_of_n(4)) == (8, 10)


def test_states_grow_without_a_cliff_between_three_and_four():
    bits = {n: math.log2(math.factorial(n * n)) for n in range(2, 7)}
    assert abs(bits[3] - 18.469) < 0.001 and abs(bits[4] - 44.250) < 0.001
    assert math.factorial(16) // math.factorial(9) == 57_657_600 and math.factorial(16) == 20_922_789_888_000
    step = [bits[n + 1] - bits[n] for n in range(2, 6)]
    assert [round(s, 2) for s in step] == [13.88, 25.78, 39.43, 54.41]
    second = [step[i + 1] - step[i] for i in range(3)]
    assert all(s > 0 for s in second) and max(second) - min(second) < 3.2        # smooth acceleration, no jump at 3 -> 4
    # classes under the eight symmetries: all n*n labels distinct, only the identity fixes an arrangement
    assert math.factorial(9) // 8 == 45_360 and math.factorial(16) % 8 == 0
    perms = perms_of_grid(3)
    assert sum(1 for p in perms if p == tuple(range(9))) == 1 and len(set(perms)) == 8


def test_three_different_tens():
    assert tri(4) == 1 + 2 + 3 + 4 == 10 and orbit_count(7) == orbit_count(8) == 10
    assert lines_of_n(4) == 2 * 4 + 2 == 10
    # different causes: the orbit count does not move at 4x4 (it is 3), and the 4x4 line count does not equal its orbit count
    assert orbit_count(4) != lines_of_n(4)


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
