"""Implements concept-map row 159 (14th scan, 8 Oct 2026): Post 423, blog 8 Sep, "The ancient proverb that barks do not bite and its human corollary that the talking man seldom ..."
(about 440 words, five paragraphs, byline not recorded; read twice, the second read asked only for the combinations it names). The post pairs two properties, loud or quiet
and acts (bites) or does not, and names these combinations (quotes from the second read, each under 125 characters):
  * loud, does not act (paragraph 2): "They are the human equivalents of the territorial cur, sounding the alarm to inflate their own courage"
  * quiet, acts (paragraph 3): "The predator that makes no sound is often simply calculating the angle of approach, saving its oxygen for the strike"
  * talks and acts (paragraph 4): "To complicate the probability further, there is an entire category of human actors who talk while they bite" (the sentence is longer; cut here)
  * paragraph 3 opens: "Yet, to rely on the inverse—to assume that absolute silence guarantees safety—is a fatal miscalculation of probability."
The combination the post never names is quiet and does not act: the empty street, the dog asleep. The second read says so in the same words ("the text never describes this case").
Three of four cells are stated and the fourth is left out. That is the shape of a table with a zero cell: what is not said is also one of the four places something could be.

The reading being tested (Rajnish to accept or reject): the post's sentences are claims about a 2 x 2 table of cells (loud/quiet) x (acts/idle), and the claims are exact:
  1. A world is the set of cells that someone occupies: 2^4 = 16 worlds. The proverb as a universal ("whoever barks does not bite") rules out the cell (loud, acts); the post's own
     paragraph 4 occupies that cell, so as a universal the proverb is false in every world the post describes, which is why the post calls it a matter of probability. The claim
     "silence guarantees safety" rules out (quiet, acts); paragraph 3 occupies that cell. The two claims together would rule out the whole "acts" row. The three named cells
     leave exactly two of the 16 worlds, and they differ in one cell only: the unstated one. One bit is open.
  2. The four forms of a conditional (the statement, its converse, its inverse, its contrapositive) fall into TWO meanings: the statement equals its contrapositive, the converse
     equals the inverse. The two operations "swap the sides" (converse) and "negate both sides" (inverse) commute, each is its own inverse, and together they give the third
     (contrapositive): the Klein four-group acting on the four forms; the repo has met this group three times already on poles, strands and table flips (units mod 12 in
     test_clock_units.py is a fourth place). In cell terms the statement and its contrapositive rule out one cell, the converse and the inverse rule out the cell diagonally opposite.
  3. The post's "inverse" is not the textbook inverse. The textbook inverse of "loud implies idle" is "quiet implies acts", which rules out the UNSTATED cell (quiet, idle).
     The post's "inverse" ("silence guarantees safety", quiet implies idle) rules out (quiet, acts): the same antecedent with the opposite consequent. Nothing in the post depends on
     the difference; it matters only if someone encodes the post's sentences: the word "inverse" would name two different cells.
What they do NOT show: how often loud people act or quiet ones do (the post gives no data and says "probability" four times); that the 2 x 2 is the post's whole argument (it also
speaks of noise against action and roar against silence); that people fall into four types. They show that a plain paragraph about two properties is a table, and that a
table makes the missing cell visible."""
import os, sys
from itertools import product
sys.path.insert(0, os.path.dirname(__file__))

CELLS = [(a, b) for a in (0, 1) for b in (0, 1)]                              # (loud, acts)
NAME = {(1, 0): "loud, idle", (0, 1): "quiet, acts", (1, 1): "loud, acts", (0, 0): "quiet, idle"}
WORLDS = [frozenset(c for c, keep in zip(CELLS, bits) if keep) for bits in product((0, 1), repeat=4)]


def ruled_out(antecedent, consequent):
    """Cells that 'if antecedent then consequent' forbids. A literal is (axis, value): axis 0 = loud, axis 1 = acts."""
    return frozenset(c for c in CELLS if c[antecedent[0]] == antecedent[1] and c[consequent[0]] != consequent[1])


def holds(ruled, world): return not (ruled & world)


LOUD, QUIET, ACTS, IDLE = (0, 1), (0, 0), (1, 1), (1, 0)
NEG = lambda lit: (lit[0], 1 - lit[1])
PROVERB = ruled_out(LOUD, IDLE)                                               # whoever barks does not bite
SILENCE_SAFE = ruled_out(QUIET, IDLE)                                         # the post's "inverse": silence guarantees safety
NAMED = [(1, 0), (0, 1), (1, 1)]                                              # loud-idle, quiet-acts, loud-acts


def test_three_of_four_cells_are_named_and_one_bit_is_left_open():
    assert len(WORLDS) == 16 and len(set(WORLDS)) == 16
    assert PROVERB == {(1, 1)} and SILENCE_SAFE == {(0, 1)}                      # each claim forbids exactly one cell
    assert sum(holds(PROVERB, w) for w in WORLDS) == 8 and sum(holds(SILENCE_SAFE, w) for w in WORLDS) == 8
    assert sum(holds(PROVERB, w) and holds(SILENCE_SAFE, w) for w in WORLDS) == 4
    both = PROVERB | SILENCE_SAFE
    assert {c for c in CELLS if c[1] == 1} == both                               # together they forbid the whole 'acts' row: nobody acts
    consistent = [w for w in WORLDS if all(c in w for c in NAMED)]               # worlds in which all three named cells are occupied
    assert len(consistent) == 2
    assert not any(holds(PROVERB, w) for w in consistent) and not any(holds(SILENCE_SAFE, w) for w in consistent)   # the post's own examples falsify both universals
    (small, big) = sorted(consistent, key=len)
    assert big - small == {(0, 0)} and NAME[(0, 0)] == "quiet, idle"             # the two worlds differ in the cell the post never names
    # only the loud-idle example agrees with the proverb; every cell the post names after that is an exception to one of the two sayings
    assert [holds(PROVERB, frozenset([c])) for c in NAMED] == [True, True, False]
    assert [holds(SILENCE_SAFE, frozenset([c])) for c in NAMED] == [True, False, True]


def test_the_four_forms_of_a_conditional_are_two_meanings_and_swap_and_negate_make_the_klein_four_group():
    p, q = LOUD, IDLE
    form = lambda s: ruled_out(*s)
    S, converse, inverse, contra = (p, q), (q, p), (NEG(p), NEG(q)), (NEG(q), NEG(p))
    assert [form(s) for s in (S, converse, inverse, contra)] == [{(1, 1)}, {(0, 0)}, {(0, 0)}, {(1, 1)}]
    assert form(S) == form(contra) and form(converse) == form(inverse) and form(S) != form(converse)    # four forms, two meanings
    # the diagonally opposite cell: the statement forbids (loud, acts), its converse and inverse forbid (quiet, idle)
    (cell,) = form(S); (opposite,) = form(converse)
    assert opposite == (1 - cell[0], 1 - cell[1])
    # the two operations on forms; the group they generate is the Klein four-group acting on the four forms
    swap = lambda s: (s[1], s[0])
    negate = lambda s: (NEG(s[0]), NEG(s[1]))
    forms = [S, converse, inverse, contra]
    perm = lambda f: tuple(forms.index(f(s)) for s in forms)
    P_swap, P_neg = perm(swap), perm(negate)
    comp = lambda a, b: tuple(a[b[i]] for i in range(4))
    ident = (0, 1, 2, 3)
    group = {ident, P_swap, P_neg, comp(P_swap, P_neg)}
    assert len(group) == 4 and comp(P_swap, P_neg) == comp(P_neg, P_swap)       # four elements, commuting
    assert all(comp(g, g) == ident for g in group)                             # every element is its own inverse: not a cycle of four
    assert {comp(a, b) for a in group for b in group} == group                 # closed
    assert comp(P_swap, P_neg) == perm(lambda s: (NEG(s[1]), NEG(s[0])))       # swap-then-negate is the contrapositive map
    assert {g[0] for g in group} == {0, 1, 2, 3}                               # the group moves the statement to every form: regular on the four forms


def test_the_posts_inverse_is_not_the_textbook_inverse_and_the_textbook_one_is_about_the_unstated_cell():
    textbook_inverse = ruled_out(NEG(LOUD), NEG(IDLE))                         # quiet implies acts
    assert textbook_inverse == {(0, 0)} and NAME[(0, 0)] == "quiet, idle"      # it forbids the one cell the post never names
    assert SILENCE_SAFE == {(0, 1)} and SILENCE_SAFE != textbook_inverse       # the post's 'inverse' forbids a named cell instead
    # same antecedent (quiet), opposite consequents: if both held, nobody would be quiet
    assert SILENCE_SAFE | textbook_inverse == {c for c in CELLS if c[0] == 0}
    # the converse of the proverb also forbids the unstated cell (the same meaning as the textbook inverse)
    assert ruled_out(IDLE, LOUD) == textbook_inverse
    # of the 16 worlds the post's three examples leave 2; the textbook inverse holds in exactly one of them (the one with the unstated cell empty)
    two = [w for w in WORLDS if all(c in w for c in NAMED)]
    assert [holds(textbook_inverse, w) for w in sorted(two, key=len)] == [True, False]


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
