"""Implements concept-map row 103 (tenth scan): blog 12 Sep "A detective's wall captures the exact friction point. A human investigator pins up a
hundred fragments of evidence because those are the pieces they can hold in ..." (two reads, about 140 words, one paragraph). Quotes verified twice:
"a hundred fragments of evidence"; "connecting them with a few intentional strings"; "pixel, word, and metadata tag"; "mapping a trillion paths of
inference"; "Human intuition is the ultimate compression algorithm"; "The machine can generate every theoretical connection, but it cannot feel the
weight of truth." The page gives no method for either the "trillion" or the "few".

The reading being tested (Rajnish to accept or reject): the wall is a GRAPH. Fragments are nodes, strings are edges, and the human's skill is to
pick a sparse set of edges. Counted exactly (big integers, nothing sampled):
  * 100 fragments allow C(100,2) = 4,950 possible strings.
  * Any subset of them could be the wall: 2^4,950 (a 1,491-digit number).
  * Orderings of the fragments: 100! (about 9.3e157). Spanning trees (the smallest walls that connect everything): 100^98 (Cayley), about 1e196.
    The formula n^(n-2) is checked here against the matrix-tree theorem for n = 3..8.
  * So "a trillion paths" is a UNDERSTATEMENT of the space by hundreds of orders of magnitude, not an overstatement. Where the page's figure does fall:
    ordered chains of 6 fragments number 100*99*98*97*96*95 = 858,277,728,000 (8.6e11), chains of 5 number 9.03e9, chains of 7 number 8.1e13. The page
    gives no chain length, so this only locates the figure; it does not explain it.
  * One thread out of a trillion is log2(1e12) = 39.86 bits: the amount of information "reaches straight for the single thread" would be, if the
    page's own number is taken at face value.
What they do NOT show: that a detective, or an AI, works this way; that intuition is compression (the page has no measure); anything about what a model
does with 100 images. The connection to this repository is only the vocabulary: a CICO graph is exactly nodes plus chosen, typed, weighted edges."""
import math, os, sys
from itertools import combinations
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))


def spanning_trees(n):
    """Matrix-tree theorem: the number of spanning trees of K_n is any cofactor of its Laplacian."""
    L = n * np.eye(n) - np.ones((n, n))
    return int(round(np.linalg.det(L[1:, 1:])))


def test_cayley_formula_matches_the_matrix_tree_theorem():
    for n in range(3, 9):
        assert spanning_trees(n) == n ** (n - 2), n


def test_counts_for_a_hundred_fragments():
    assert math.comb(100, 2) == 4950 == len(list(combinations(range(100), 2)))
    assert len(str(2 ** 4950)) == 1491
    f = math.factorial(100)
    assert len(str(f)) == 158 and str(f).startswith("9332621544")
    assert len(str(100 ** 98)) == 197                                       # 1e196
    # ordered chains of k distinct fragments
    chains = lambda k: math.perm(100, k)
    assert chains(5) == 9_034_502_400 and chains(6) == 858_277_728_000 and chains(7) == 80_678_106_432_000
    assert chains(5) < 10 ** 12 < chains(7) and abs(chains(6) / 1e12 - 0.858) < 0.001
    # choosing a "few" strings out of the 4,950 possible: 3 strings give 2.0e10 different walls, 5 strings already give 2.5e16 (more than a trillion)
    assert math.comb(4950, 3) < 10 ** 11 < 10 ** 12 < math.comb(4950, 5)


def test_one_thread_out_of_a_trillion_is_about_forty_bits():
    assert abs(math.log2(10 ** 12) - 39.863) < 0.001
    assert abs(math.log2(math.perm(100, 6)) - 39.64) < 0.01                 # the 6-chain count lies close to it (no claim that this is the cause)


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
