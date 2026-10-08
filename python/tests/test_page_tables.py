"""Implements concept-map row 140 (thirteenth scan): tables that exist ACROSS pages, never on one page. Quotes verified on second reads (YES) unless marked.
  * Three consecutive numbered posts of 1 Sep (UTC): 3779 (00:28:39; heading "she is here"): "Krishna (Vedic / Sanatana Dharma)" / "Buddha (Buddhism)" / "Jesus (Christianity)" / "Muhammad (Islam)".
    3780 (00:31:07): "Durga / Kali / Shakti (Vedic / Sanatana Dharma)" / "Tara / Prajnaparamita (Buddhism)" / "Mary (Christianity)" / "Fatimah (Islam)"; its body says the male deities "often represent the static
    background". 3781 (00:34:35): "The Universal God (The Static Background — Purusha / Mahakala)" and "The Universal Goddess (The Kinetic Engine — Prakriti / Mahashakti)", each with "The Blueprint:" "The Form:"
    "The Name:"; it does not mention the four traditions. No page states a table or the number four.
  * 12 Aug Post 307 (two reads): four walls "The Wall of the Past - everything we have seen." / Future / Present / "The Wall of the Eternal - the awareness that sees all." and four corner questions "Where did I come
    from?" (past), "Where am I going?" (future), "Who am I?" (the self), "What is the meaning?" (the eternal). 12 Aug Post 365 (two reads): the user's four lines "past" "future" "present" "eternal" in the same ORDER,
    and six lists of four: the four headed Lines (The Frame; The Negative Space; The Mark; The Completion), the four Clean Lines (Creation; Preservation; Destruction; Grace), the four Rectangles (Page; Tablet;
    Mobile; Frame) which "are the four arms of Mahalakshmi" (pustaka, pot of gold, coconut, lotus), the user's four lines, Human Spirit (Birth, Life, Death, Eternity), Universe (Space, Time, Matter, Consciousness).
  * 5 Sep Post 326 (one read): four "structural attractors": "those who maintain the physical base"; "those who circulate resources"; "those who wield power"; "those who anchor the conceptual framework."
    6 Sep Post 404's previous-post teaser (a theme link, not the page's own text; one read): four roles - protecting and governing the perimeter; generating and managing resources; laboring to maintain infrastructure;
    holding knowledge or the sacred line.

The reading being tested (Rajnish to accept or reject): the author keeps FIXED-ORDER lists and reuses them, so several posts together make tables (rows = the items of one list, columns = the lists). Counted:
  1. The three 1 Sep posts make a 4 x 2 table: the same four traditions in the same order in 3779 and 3780 (Vedic, Buddhism, Christianity, Islam); one male figure per tradition and 11 female names over
     the 4 female cells (3 + 2 + 1 + 1); 3781 names the two columns (static background / kinetic engine). Eight cells; the two projections are 4 classes of 2 (forget the pole) and 2 classes of 4 (forget the tradition).
     Two independent random orders would agree once in 24; a list habit makes it likely, which is the point of the reading.
  2. Post 307 and Post 365 give the order past, future, present, eternal twice (not the order of time). Three of Post 307's four wall-question pairs are stated (past, future, eternal); with three pairs given the fourth
     is forced: of 24 matchings exactly 1 fits (2 fit two stated pairs, 6 fit one). The pairing "the self = the present wall" is the only one that is not stated.
  3. Post 365's six quads make a 6 x 4 table (24 cells). The page states ONE alignment across quads (rectangles = arms). Only two words appear in two quads: "Frame", at position 1 in the Lines and at
     position 4 in the Rectangles, and "Space", at position 2 in the Lines ("The Negative Space") and at position 1 in the Universe: so list position cannot be the identity map between quads. (The fourth position holds the completing item in every quad - Completion, Grace, Frame (the lotus), eternal, Eternity,
     Consciousness - but that is MY matching of meanings, marked INTERP.) Aligning all six quads by position is 1 choice out of 24^5 = 7,962,624 possible column alignments.
  4. Post 326's four roles and the four roles in the Post 404 teaser match one to one only by a meaning match that is mine (INTERP): written as a permutation of positions it is [2, 1, 0, 3], one swap with two roles
     in place, 3 inversions. A random order has at least two roles in place in 7 of 24 cases (29%), so the match carries no weight by position.
What they do NOT show: that the author meant any of these tables; that Krishna, Buddha, Jesus and Muhammad are "male counterparts" of the female names (3780's own words are "often represent" and it names only Shiva);
that the traditions are four on purpose; that the roles of Post 326 and Post 404 are the same four. Counting alignments is arithmetic; it is not a finding about the pages' meaning."""
import os, re, sys
from itertools import permutations
sys.path.insert(0, os.path.dirname(__file__))

P3779 = ["Krishna (Vedic / Sanatana Dharma)", "Buddha (Buddhism)", "Jesus (Christianity)", "Muhammad (Islam)"]
P3780 = ["Durga / Kali / Shakti (Vedic / Sanatana Dharma)", "Tara / Prajnaparamita (Buddhism)", "Mary (Christianity)", "Fatimah (Islam)"]
WALLS = ["past", "future", "present", "eternal"]
QUESTIONS = ["Where did I come from?", "Where am I going?", "Who am I?", "What is the meaning?"]
P21 = {
    "Lines": ["The Frame", "The Negative Space", "The Mark", "The Completion"],
    "Clean Lines": ["Creation", "Preservation", "Destruction", "Grace"],
    "Rectangles": ["Page", "Tablet", "Mobile", "Frame"],
    "User lines": ["past", "future", "present", "eternal"],
    "Human Spirit": ["Birth", "Life", "Death", "Eternity"],
    "Universe": ["Space", "Time", "Matter", "Consciousness"],
}
ARMS = ["pustaka", "pot of gold", "coconut", "lotus"]
# R21 = Post 326; P74 = the previous-post teaser on Post 404
R21 = ["maintain the physical base", "circulate resources", "wield power", "anchor the conceptual framework"]
P74 = ["protecting and governing the perimeter", "generating and managing resources", "laboring to maintain infrastructure", "holding knowledge or the sacred line"]


def tradition(entry): return re.search(r"\(([^)]*)\)$", entry).group(1)


def figures(entry): return [x.strip() for x in re.sub(r"\s*\([^)]*\)$", "", entry).split("/")]


def test_three_posts_make_a_4_by_2_table_with_the_traditions_in_the_same_order():
    rows_m, rows_f = [tradition(e) for e in P3779], [tradition(e) for e in P3780]
    assert rows_m == rows_f == ["Vedic / Sanatana Dharma", "Buddhism", "Christianity", "Islam"]
    male = [figures(e) for e in P3779]
    female = [figures(e) for e in P3780]
    assert [len(m) for m in male] == [1, 1, 1, 1] and [len(f) for f in female] == [3, 2, 1, 1] and sum(len(f) for f in female) == 7
    cells = [(t, pole) for t in rows_m for pole in ("static", "kinetic")]
    assert len(cells) == 8 == 4 * 2
    by_pole = {p: [c for c in cells if c[1] == p] for p in ("static", "kinetic")}
    by_tradition = {t: [c for c in cells if c[0] == t] for t in rows_m}
    assert sorted(len(v) for v in by_pole.values()) == [4, 4] and sorted(len(v) for v in by_tradition.values()) == [2, 2, 2, 2]
    assert len(list(permutations(range(4)))) == 24                                 # two independent random orders would agree once in 24
    names = [n for f in male + female for n in f]
    assert len(names) == 11 and len(set(names)) == 11                              # four male names and seven female names, no name twice


def test_the_order_past_future_present_eternal_is_shared_and_three_stated_pairs_force_the_fourth():
    assert WALLS == P21["User lines"] and WALLS != ["past", "present", "future", "eternal"]       # twice the same, and not the order of time
    stated = {0: 0, 1: 1, 3: 3}                                                    # question index -> wall index for past, future, the eternal
    matchings = list(permutations(range(4)))
    assert len(matchings) == 24
    consistent = lambda keep: [p for p in matchings if all(p[q] == w for q, w in keep.items())]
    assert [len(consistent(dict(list(stated.items())[:k]))) for k in (0, 1, 2, 3)] == [24, 6, 2, 1]
    (only,) = consistent(stated)
    assert only == (0, 1, 2, 3)                                                    # forced: the self gets the present wall, the identity matching
    assert QUESTIONS[only.index(2)] == "Who am I?" and WALLS[2] == "present"


def test_six_quads_make_a_6_by_4_table_whose_two_shared_words_sit_at_different_positions():
    assert len(P21) == 6 and all(len(v) == 4 for v in P21.values()) and sum(len(v) for v in P21.values()) == 24
    assert P21["Rectangles"] == ["Page", "Tablet", "Mobile", "Frame"] and len(ARMS) == 4        # rectangles = arms is the one stated alignment
    where = {}
    for quad, items in P21.items():
        for i, item in enumerate(items):
            words = set(re.findall(r"[A-Za-z]+", item.lower())) - {"the"}
            for w in words: where.setdefault(w, []).append((quad, i))
    shared = {w: v for w, v in where.items() if len({q for q, _ in v}) > 1}
    assert shared == {"frame": [("Lines", 0), ("Rectangles", 3)], "space": [("Lines", 1), ("Universe", 0)]}     # two repeated words, each at two different positions
    assert all(i != j for v in shared.values() for (_, i), (_, j) in [tuple(v)])
    assert 24 ** 5 == 7_962_624                                                    # column alignments of six quads (24 per added quad, 5 added)


def test_the_four_roles_of_two_posts_match_by_a_permutation_that_a_random_order_often_beats():
    match = {0: 2, 1: 1, 2: 0, 3: 3}                                               # R21 index -> P74 index, by meaning (mine)
    perm = [match[i] for i in range(4)]
    assert perm == [2, 1, 0, 3] and sorted(perm) == [0, 1, 2, 3]
    assert sum(1 for i, p in enumerate(perm) if i == p) == 2
    assert sum(1 for i in range(4) for j in range(i + 1, 4) if perm[i] > perm[j]) == 3
    at_least_two_fixed = sum(1 for p in permutations(range(4)) if sum(1 for i, x in enumerate(p) if i == x) >= 2)
    assert at_least_two_fixed == 7 and round(100 * 7 / 24) == 29
    assert "resources" in R21[1] and "resources" in P74[1]                         # the only shared word at the matched positions


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
