"""Implements concept-map row 116 (eleventh scan): blog 13 Sep, the page "Inner Engineering gives you the manual to keep your own internal engine from seizing up in modern
traffic, but it leaves you sitting alone in an ..." (Post 266; about 210 words; byline rajnish choubey; reads as AI-voice by style, which proves nothing; two reads).
Body, verified on a second read: "the woman standing in front of you moving through her Lakshmi-to-Kali spectrum"; "it's about holding your center, maintaining absolute
zero violence, dropping your ego when she walks away"; "It is a vital foundation, but it treats the human being as a closed system operating in a vacuum."
The page's PREVIOUS-POST teaser (theme text, NOT the body) is the excerpt of Post 48, which lists Lakshmi, Saraswati, Durga, Kali in that order. Post 48 itself (second
scan, concept-map row 41) is not a line: "She enters as Lakshmi", then "reads the room" (Saraswati), then Durga if he "steps up with absolute accountability", Kali if he does not.

The reading being tested (Rajnish to accept or reject): "spectrum" suggests a LINE of four stages, Post 48 gives a FORK after the second stage. The shape of four things
decides how many symmetries it has, and the symmetries decide how many different arrangements of four distinct names it can hold. Counted by brute force (n! labelings,
orbits under the automorphisms, which is n! / |Aut| when every name is different):
    shape of four                               symmetries   classes of 4 names   bits
    ring, undirected  (a "quad" square)             8              3             1.585
    ring, directed    (arrows all one way round)    4              6             2.585
    path, undirected  (a spectrum, no arrows)       2             12             3.585
    path, directed    (a spectrum with an order)    1             24             4.585
    fork, edges not told apart (Post 48 as a tree)  2             12             3.585
    fork, edges told apart ("steps_up", "does_not") 1             24             4.585
Each halving of the symmetry group adds exactly one bit. In OUR language the telling-apart is the edge: both the relation name and the weight (`rel#w`) count, so
two fork edges with the same name but different weights are already told apart. The ring rows repeat test_symmetry_direction.py and test_shape_capacity.py (3, 6); the
new rows are the path and the fork.
What they do NOT show: that the page's "spectrum" is a path (the page does not say), that Post 48 is the same story as this page (the teaser only shows the page-14 author
linked them), or that four is the right number of stages. Which shape Rajnish means is the question; the table says what each shape can carry."""
import math, os, sys
from itertools import permutations
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.cico_parser import CICOParser
from axi.engine.gate import Graph, check, apply
from test_signed_slots import etext


def automorphisms(n, edges):
    E = set(edges)
    return [p for p in permutations(range(n)) if {(p[a], p[b], l) for a, b, l in E} == E]


def classes_of_names(n, auts):
    """Orbits of the n! assignments of n different names to the n places under the automorphisms (brute force)."""
    seen, classes = set(), 0
    for lab in permutations(range(n)):
        if lab in seen: continue
        classes += 1
        seen |= {tuple(lab[p.index(i)] for i in range(n)) for p in auts}
    return classes


def lab(pairs, label=""): return [(a, b, label) for a, b in pairs]


def shapes(n):
    ring = [(i, (i + 1) % n) for i in range(n)]
    path = [(i, i + 1) for i in range(n - 1)]
    both = lambda es: es + [(b, a) for a, b in es]
    return {"ring undirected": lab(both(ring)), "ring directed": lab(ring), "path undirected": lab(both(path)), "path directed": lab(path)}


def test_the_ladder_for_four_names_is_3_6_12_24_and_classes_equal_factorial_over_symmetries():
    want = {"ring undirected": (8, 3), "ring directed": (4, 6), "path undirected": (2, 12), "path directed": (1, 24)}
    for name, edges in shapes(4).items():
        a = automorphisms(4, edges)
        assert (len(a), classes_of_names(4, a)) == want[name], name
    for n in (3, 4, 5):                                                    # the law n! / |Aut| for every shape and size, not only 4
        for name, edges in shapes(n).items():
            a = automorphisms(n, edges)
            assert classes_of_names(n, a) == math.factorial(n) // len(a), (n, name)
    bits = [round(math.log2(c), 3) for c in (3, 6, 12, 24)]
    assert bits == [1.585, 2.585, 3.585, 4.585]
    assert [round(b2 - b1, 6) for b1, b2 in zip([math.log2(c) for c in (3, 6, 12)], [math.log2(c) for c in (6, 12, 24)])] == [1.0, 1.0, 1.0]


def test_post_48_as_a_fork_has_a_swap_symmetry_until_its_two_branches_are_told_apart():
    fork_same = lab([(0, 1), (1, 2), (1, 3)], "then")                      # stages 0 -> 1, then 1 -> 2 and 1 -> 3 with one and the same relation
    a = automorphisms(4, fork_same)
    assert len(a) == 2 and classes_of_names(4, a) == 12                    # the two branches can be exchanged
    fork_named = [(0, 1, "then"), (1, 2, "steps_up"), (1, 3, "does_not")]
    a = automorphisms(4, fork_named)
    assert a == [(0, 1, 2, 3)] and classes_of_names(4, a) == 24            # the page's condition names the branches: nothing can be exchanged


def fork_graph(rel_to_durga, w_to_durga, rel_to_kali, w_to_kali):
    """Lakshmi (0,0) -> Saraswati (0,1) -> Durga (0,2) and Kali (1,1), through the real gate. Node values are all 0 so that only the structure is measured."""
    g = Graph()
    for n in ((0, 0), (0, 1), (0, 2), (1, 1)): g.add_node(n, 0)
    txt = " ".join([etext("ADD", ((0, 0), (0, 1), "then"), 1), etext("ADD", ((0, 1), (0, 2), rel_to_durga), w_to_durga), etext("ADD", ((0, 1), (1, 1), rel_to_kali), w_to_kali)])
    d = CICOParser.parse_transition_delta(txt)
    assert check(g, d).ok
    apply(g, d)
    return g


def graph_automorphisms(g):
    nodes = sorted(g.nodes)
    out = []
    for p in permutations(range(len(nodes))):
        m = {nodes[i]: nodes[p[i]] for i in range(len(nodes))}
        if {(m[u], m[v], rel): w for (u, v, rel), w in g.edges.items()} == g.edges and all(g.nodes[m[n]] == g.nodes[n] for n in nodes): out.append(p)
    return out


def test_in_the_language_the_relation_name_and_the_weight_each_break_the_fork_symmetry():
    assert len(graph_automorphisms(fork_graph("branch", 1, "branch", 1))) == 2         # same name, same weight: Durga and Kali are interchangeable
    assert len(graph_automorphisms(fork_graph("steps_up", 1, "does_not", 1))) == 1     # told apart by the relation name
    assert len(graph_automorphisms(fork_graph("branch", 1, "branch", 2))) == 1         # told apart by the weight alone
    assert len(graph_automorphisms(fork_graph("branch", 1, "branch", 1))) == 2 and len(graph_automorphisms(fork_graph("a", 3, "b", 3))) == 1


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
