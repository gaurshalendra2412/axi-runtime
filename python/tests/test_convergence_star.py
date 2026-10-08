"""Implements concept-map row 114 (eleventh scan): blog 13 Sep, the page "Deep in the ancient shadows of the primordial jungle, where the breath of the damp
earth mingled with the smoke of sacred fires, the earliest ..." (Post 267; about 360 to 380 words in four paragraphs; byline rajnish choubey; style reads as AI-voice,
which proves nothing either way; three reads). Quotes verified on a second and third read (YES or CLOSE; the page uses curly apostrophes):
  * paragraph 2: "Grandfather Brahma sat as the elder seed-sitter"; "Father Vishnu walked the long, winding paths between the clearings";
    "Mahesh, the brother of the wild wind and the burning pyres"; "this masculine line of grandfather, father, and brother was only half the story".
  * paragraph 3: "Beneath their roots burned Kali, the dark and primal mother of the undergrowth".
  * paragraph 4: "From the convergence of the grandfather's blueprint, the father's steady stewardship, the brother's wild silence, and the mother's fierce,
    untamed fire, stepped Durga." and "In her, the three prongs of her brother Mahesh's trident found a new home, uniting the past of the grandfather, the present of
    the father, and the infinite horizon of the mother into a single, breathing geometry of grace and power".  "She was the ultimate child of the jungle, riding the
    golden lion through the dense canopy with ten arms outstretched, holding every weapon of the cosmos."
  The page never uses the word "future", never says sister or daughter, and calls Kali (not Durga) the mother.
What is new compared with the earlier scans: this page lays the TIME triad (listed without an assignment on earlier pages) on kin figures and on the prongs. Time is laid on three kin figures (past: grandfather, present:
father, infinite horizon: mother), the trident's three prongs unite those three times, and the fourth figure (the brother) is the owner of the trident, not a
prong. Four contributions converge on one figure. The same sentence about the prongs also appears in Post 225 (ninth scan), and Post 23 (first scan) gives the three
prongs another triad (cognition, substance, power). So at least two different triads are laid on the same three prongs, and no page says which prong carries what.

The reading being tested (Rajnish to accept or reject): a convergence of four named contributions is a star with four arms, and what the star can say depends
on whether the arms are told apart. Counted by brute force, with nothing assumed:
  1. A star of four arms into one node, arms NOT told apart, has 24 symmetries (every shuffle of the arms). Told apart by four different edge names (blueprint,
     stewardship, silence, fire), it has exactly 1. The edge name (`rel` in `ADD[(r,c)->(r,c):rel#w]`) is what breaks the symmetry.
  2. In OUR language: Durga on the centre cell (1,1), the four contributions on the four arm cells (0,1), (1,0), (1,2), (2,1). Every one of the 24 ways to put the four
     names on the four arms is admitted by the gate and gives a different graph; the 8 symmetries of the board fold those 24 into 3 classes (4 rotations: 6 classes).
     This is the same ladder as `test_shape_capacity.py` (3, 6, 24): a quad carries 1.585 bits until its arms are named.
  3. The page fixes WHO is a prong (grandfather, father, mother, by their times) and who is not (the brother, the owner), so of the 24 ways to put four figures in three
     prong places and a holder place the words rule out 18 and leave 6. Those 6 differ only in WHERE each time sits on the trident: 3 once the trident's own left-right
     mirror is ignored (which time is in the middle). Each further triad laid on the same trident adds another factor of 6 that no page settles (2.585 bits); three
     triads leave 36 relative alignments (5.17 bits).
What they do NOT show: that the page means a graph; that the star is the "full quadrant" of Post 214 (a different page; row 87); that the brother is a "handle"
(the page says trident owner; reading him as the shaft is ours); that the prongs of Post 23 and of this page are meant to line up; anything about the symmetries of the
real object. The relations are the page's words; the cells are our choice."""
import math, os, sys
from itertools import permutations
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.cico_parser import CICOParser
from axi.engine.d4 import IDS
from axi.engine.gate import Graph, check, apply
from test_gate_frame_covariance import d4_relabel
from test_signed_slots import etext

NAMES = ["blueprint", "stewardship", "silence", "fire"]                # grandfather, father, brother, mother (the page's order of listing)
ARMS = [(0, 1), (1, 0), (1, 2), (2, 1)]
CENTRE = (1, 1)
ROT = (0, 1, 2, 3)


def automorphisms(n, edges):
    E = set(edges)
    return [p for p in permutations(range(n)) if {(p[a], p[b], l) for a, b, l in E} == E]


def build(assign):
    """assign[i] = relation name on the arm ARMS[i]; returns (host graph, delta) for 'the four arrive at the centre'."""
    g = Graph()
    for n in ARMS + [CENTRE]: g.add_node(n, 0)
    d = CICOParser.parse_transition_delta(" ".join(etext("ADD", (a, CENTRE, assign[i]), 1) for i, a in enumerate(ARMS)))
    return g, d


def arrangement(assign):
    g, d = build(assign)
    assert check(g, d).ok
    apply(g, d)
    return frozenset(g.edges)


def fold(arrangements, ids):
    maps = [d4_relabel(t) for t in ids]
    seen, classes = set(), 0
    for a in arrangements:
        if a in seen: continue
        classes += 1
        seen |= {frozenset((m[u], m[v], rel) for (u, v, rel) in a) for m in maps}
    return classes


def test_four_arms_into_one_node_have_24_symmetries_until_the_arms_are_named():
    unlabeled = [(i, 4, "") for i in range(4)]
    assert len(automorphisms(5, unlabeled)) == 24                          # any shuffle of the four arms
    labelled = [(i, 4, NAMES[i]) for i in range(4)]
    assert automorphisms(5, labelled) == [tuple(range(5))]                 # only the identity: each arm is told apart by its edge name
    two_named = [(0, 4, "blueprint"), (1, 4, "blueprint"), (2, 4, "silence"), (3, 4, "silence")]
    assert len(automorphisms(5, two_named)) == 4                           # two names used twice: only the shuffles inside each pair survive


def test_in_the_language_the_24_namings_are_all_admitted_and_fold_to_3_classes_under_the_board_symmetries():
    arrangements = {arrangement(p) for p in permutations(NAMES)}
    assert len(arrangements) == 24                                         # 24 different graphs
    assert fold(arrangements, IDS) == 3                                    # 8 board symmetries: 24 / 8
    assert fold(arrangements, ROT) == 6                                    # rotations only: 24 / 4
    assert [round(math.log2(c), 3) for c in (3, 6, 24)] == [1.585, 2.585, 4.585]
    same_name_twice = arrangement(["blueprint", "blueprint", "silence", "fire"])
    g, d = build(["blueprint", "blueprint", "silence", "fire"])
    assert check(g, d).ok and len(same_name_twice) == 4                    # the gate does not need the names to differ; that is a property of the page, not of the gate


def test_the_page_fixes_who_is_a_prong_but_not_where_and_each_extra_triad_adds_a_factor_of_6():
    figures = ["grandfather", "father", "brother", "mother"]
    places = ["prong1", "prong2", "prong3", "holder"]
    ways = list(permutations(figures))                                     # figure in place i = ways[k][i]
    assert len(ways) == 24
    times = {"grandfather": "past", "father": "present", "mother": "infinite horizon"}      # the page's words
    fit = [w for w in ways if w[3] == "brother" and {w[i] for i in range(3)} == set(times)]
    assert len(fit) == 6                                                   # holder fixed by "her brother Mahesh's trident"; the 3 prong places can still be filled 6 ways
    prong_orders = {tuple(w[:3]) for w in fit}
    assert len(prong_orders) == 6
    mirror = lambda t: (t[2], t[1], t[0])                                  # the trident's own left-right mirror
    classes = {frozenset((t, mirror(t))) for t in prong_orders}
    assert len(classes) == 3                                               # up to the mirror only "who is in the middle" is left
    assert {t[1] for t in prong_orders} == set(times)                      # each of the three times can be the middle one
    assert math.factorial(3) == 6 and abs(math.log2(6) - 2.585) < 0.001
    assert abs(math.log2(6 ** 2) - 5.17) < 0.001                           # three triads on one trident: two alignments no page gives


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
