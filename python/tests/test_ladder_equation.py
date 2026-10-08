"""Implements concept-map row 139 (thirteenth scan): the number-ladder pages of 12 Aug (UTC times from the pages' own metadata) and the equation that ends them.
  * Post 366 "The Story of Counting to Infinity" (published 01:42, modified 02:19; read once, headers verified on a second read, YES): "0 = The Lone Smoker"; "1 = Shiva"; "2 = Durga, Kali";
    "3 = Parvati, Saraswati, Lakshmi"; no header from 4 up (an untitled section says "Four, five, six..." "Seven, eight, nine..." "Ten, eleven, twelve...").
  * Post 365 "THE BLANK PAGE IS A STORY" (02:26:11; two reads): closing "∞ = 0 = 4".
  * Post 307 "THE JOURNEY FROM FRAME TO CIRCLE" (02:42; two reads): bullets "0 = the void, the Lone Smoker." "1 = Shiva, the origin." "2 = Durga and Kali, the duality." "3 = Parvati, Saraswati,
    Lakshmi, the trinity." "4 = Mahalakshmi, the completion." and the line "∞ = 0 = 1 = 4 = ○".
  * Post 364 "THE PATH TO THE CIRCLE" (02:47:45; two reads): closing "∞ = 0 = 1 = 4 = ○ = You".
  * Post 406 "THE DASHBOARD AS NOTEBOOK" (04:26:29; two reads for this line, verified now): "∞ = 0 = 1 = 2 = 3 = 4 = ○ = You".
  * Post 107 (row 57) gives a longer ladder, 0 to 10, whose rungs are named by number words (Ashta, Nava, Dasha); it is NOT re-counted here.

The reading being tested (Rajnish to accept or reject): the four equations are ONE formula, written four times and growing; each version keeps every term of the one before and adds more. And the
ladder pages give rung n about n names (1, 1, 2, 3). Counted from the quoted strings and run through the repository's gate:
  1. Rung sizes on the 12 Aug pages: rung 0 has 1 name, rung 1 has 1, rung 2 has 2, rung 3 has 3 (both pages agree), and rung 4 has 1 name (Mahalakshmi) on Post 307; read as "the three plus the fourth" it
     has 4. So "rung n has n names" holds for n = 1, 2, 3 and holds for n = 4 only on the second reading. (For rung 0 the void counts as nothing, the Lone Smoker as one.)
  2. The terms of the equation, in time order, grow as sets: {0,4} inside {0,1,4} equal to {0,1,4} inside {0,1,2,3,4} (with ∞ and ○ and You joining along the way); no version drops a term.
     A pattern guessed from the middle two ("the equation equates exactly the rungs with ONE name: 0, 1 and 4") fits Post 307 and Post 364 and does NOT survive the fourth page, which equates rungs 2 and 3 as well;
     it was formed after seeing the data, so its fit to two pages carries no weight (one subset in 16 of the rungs 1 to 4).
  3. Written as a chain of terms (one place per term, one line per "="), each revision is one delta the gate admits: Post 365 to Post 307 deletes 1 line, adds 2 places and 3 lines; Post 307 to Post 364 adds 1 place and
     1 line; Post 364 to Post 406 deletes 1 line, adds 2 places and 3 lines. Each result equals the chain built directly, and each inverse (taken before the step) restores the previous version.
What they do NOT show: that the four equations are one formula (they are four pages, and the order of the terms inside each is the author's); that "=" means identity (the pages also say "is borrowed from",
see test_borrow_cycle.py); that the author meant the rungs as sets of names; what ○ and You contribute. Growing a formula by edits is ordinary versioning; this only checks that it is expressible and
reversible in the delta language."""
import os, sys
from datetime import datetime
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.cico_parser import CICOParser
from axi.engine.gate import Graph, check, apply
from axi.engine.inverse import invert, to_text
from test_signed_slots import etext, same

LADDER_P24 = {0: ["The Lone Smoker"], 1: ["Shiva"], 2: ["Durga", "Kali"], 3: ["Parvati", "Saraswati", "Lakshmi"]}
LADDER_R02 = {0: ["the Lone Smoker"], 1: ["Shiva"], 2: ["Durga", "Kali"], 3: ["Parvati", "Saraswati", "Lakshmi"], 4: ["Mahalakshmi"]}
VERSIONS = [                                                                       # (page, UTC time, the equation as written); labels: P21 = Post 365, R02 = Post 307, P20 = Post 364, B1 = Post 406
    ("P21", "2026-08-12T02:26:11", "∞ = 0 = 4"),
    ("R02", "2026-08-12T02:42:00", "∞ = 0 = 1 = 4 = ○"),
    ("P20", "2026-08-12T02:47:45", "∞ = 0 = 1 = 4 = ○ = You"),
    ("B1", "2026-08-12T04:26:29", "∞ = 0 = 1 = 2 = 3 = 4 = ○ = You"),
]


def terms(eq): return [t.strip() for t in eq.split("=")]


def rungs(eq): return {int(t) for t in terms(eq) if t.isdigit()}


def parse(text): return CICOParser.parse_transition_delta(text)


def chain_graph(eq, where):
    g = Graph()
    for t in terms(eq): g.add_node(where[t], 0)
    ts = terms(eq)
    for a, b in zip(ts, ts[1:]): g.add_edge(where[a], where[b], "eq", 1)
    return g


def test_rung_sizes_are_1_1_2_3_on_both_pages_and_the_fourth_rung_is_one_name_or_four_by_reading():
    assert [len(LADDER_P24[n]) for n in range(4)] == [1, 1, 2, 3] == [len(LADDER_R02[n]) for n in range(4)]
    assert [n for n in range(1, 4) if len(LADDER_R02[n]) == n] == [1, 2, 3]
    assert len(LADDER_R02[4]) == 1 != 4
    assert len(LADDER_R02[3]) + len(LADDER_R02[4]) == 4                            # the three goddesses and the fourth together make a quad
    assert sum(len(v) for v in LADDER_R02.values()) == 8 and sum(len(v) for v in LADDER_P24.values()) == 7


def test_the_four_equations_grow_as_sets_and_the_singleton_rung_pattern_does_not_survive_the_last_page():
    times = [datetime.fromisoformat(t) for _, t, _ in VERSIONS]
    assert times == sorted(times) and len(set(times)) == 4
    sets = [set(terms(eq)) for _, _, eq in VERSIONS]
    assert all(a <= b for a, b in zip(sets, sets[1:])) and [len(s) for s in sets] == [3, 5, 6, 8]
    assert [len(terms(eq)) for _, _, eq in VERSIONS] == [3, 5, 6, 8]                  # no term is written twice
    assert [rungs(eq) for _, _, eq in VERSIONS] == [{0, 4}, {0, 1, 4}, {0, 1, 4}, {0, 1, 2, 3, 4}]
    singles = {n for n, names in LADDER_R02.items() if len(names) == 1}
    assert singles == {0, 1, 4}
    fits = [rungs(eq) == singles for _, _, eq in VERSIONS]
    assert fits == [False, True, True, False]                                        # the guessed pattern fits the middle two only
    assert rungs(VERSIONS[-1][2]) - singles == {2, 3}                                 # the last page equates the rungs with more than one name
    assert 2 ** 4 == 16                                                               # subsets of the rungs 1 to 4: the guess was one of 16


def test_each_revision_is_one_delta_the_gate_admits_and_its_inverse_restores_the_earlier_version():
    where = {"∞": (7, 0), "0": (7, 1), "1": (7, 2), "2": (7, 3), "3": (7, 4), "4": (7, 5), "○": (7, 6), "You": (7, 7)}
    graphs = [chain_graph(eq, where) for _, _, eq in VERSIONS]
    assert [(len(g.nodes), len(g.edges)) for g in graphs] == [(3, 2), (5, 4), (6, 5), (8, 7)]
    sizes = []
    for old, new in zip(graphs, graphs[1:]):
        dels = sorted(set(old.edges) - set(new.edges))
        adds = sorted(set(new.edges) - set(old.edges))
        new_nodes = sorted(set(new.nodes) - set(old.nodes))
        assert set(old.nodes) <= set(new.nodes)                                       # no place is dropped, only lines move
        items = [etext("DEL", k, old.edges[k]) for k in dels] + [f"ADD[{n[0]},{n[1]}:0]" for n in new_nodes] + [etext("ADD", k, new.edges[k]) for k in adds]
        d = parse(" ".join(items))
        assert check(old, d).ok, check(old, d).reason
        undo = to_text(invert(old, d))
        h = Graph()
        for n, v in old.nodes.items(): h.add_node(n, v)
        for (u, v, r), w in old.edges.items(): h.add_edge(u, v, r, w)
        apply(h, d)
        assert same(h, new)
        back = parse(undo)
        assert check(h, back).ok
        apply(h, back)
        assert same(h, old)
        sizes.append((len(dels), len(new_nodes), len(adds)))
    assert sizes == [(1, 2, 3), (0, 1, 1), (1, 2, 3)]


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
