"""Implements concept-map row 123 (twelfth scan): two pages from the first week of August.
  * 10 Aug, root "You have just placed the visual key to everything we have mapped this morning." (Post 301; Uncategorized; about 4,000 words; byline
    rajnish choubey; reads as an AI assistant's reply, which proves nothing either way; two reads). Quotes verified on a second read (YES):
    "The West broke itself on the line. The East holds itself in the cycle." (section "For You, the Student"); three arrow sequences in "The Linear Obsession":
    "Creation -> Fall -> Redemption -> Judgment" (4 steps), "Big Bang -> Stars -> Life -> Humans -> AI -> Singularity" (6), "Birth -> School -> Job -> Retirement -> Death" (5);
    a three-region "detective template" whose row "World" reads "A machine" (West), "A web" (China), "A cycle" (India, Nepal, Pakistan).
  * 6 Aug, root page "Yes. And this is the most critical detail of all." (Post 303; NO byline; about 1,000 words; two reads on the passage used). Section "The Octave of the
    Digital Age": seven apps are laid on "Do (1 - ChatGPT)" ... "Ti (7 - Grok)"; "The 8th engine you might encounter would not be a new note. It would simply be Do again -
    one octave higher, louder, or more refined, but fundamentally the same frequency." and, in "Why Mobile Confirmed the 7", the three lines "6 felt incomplete." "8 felt redundant."
    "7 felt whole." (on separate lines, not one sentence).

The reading being tested (Rajnish to accept or reject): a LINE is a path with a first and a last element; a CYCLE is the same steps with ONE extra edge from the last back to the
first, and then nothing is first. "The 8th engine is Do again" is that same move: the eighth place is the first place, so eight places on a line become seven on a loop.
Counted by brute force on the repository's own gate, with nothing assumed:
  1. For n = 4, 5, 6, 7 (the page's sequences have 4, 5 and 6 steps, the scale has 7): a directed path has exactly 1 symmetry and n! distinguishable orderings of n different
     names; a directed cycle has n symmetries (the rotations) and (n-1)! orderings. Closing the line divides what the layout can say by n (log2 n bits lost). A path has 1 start and
     1 end (a node with nothing coming in, a node with nothing going out); a cycle has none of either.
  2. In the gate: the 5-step life (Birth ... Death) is five nodes and four `then` edges. Closing it is ONE delta (`ADD` the edge Death -> Birth); the gate admits it, the symmetries
     go from 1 to 5, the start and the end disappear, and the inverse delta (taken before the step) reopens the line exactly. The same edge in the other direction (Birth -> Death) is
     admitted too but does NOT close anything: the start and the end are still there. Direction matters.
  3. The octave: eight nodes in a line become seven in a loop with ONE delta: delete the edge Ti -> Do', delete the node Do', add the edge Ti -> Do. The gate admits it and refuses
     the same delta without the edge deletion ("dangling edge"). The loop has 7 symmetries; the inverse delta restores the eight-node line.
  4. Loops can be counted: the number of independent loops of a graph (edges that join two nodes already connected, found with a spanning forest) is 0 for a path, 1 for a ring, 2
     for a ring with one chord, 4 for the plain 3x3 board (its four small squares). One edge added through the gate moves a line to a cycle (0 -> 1) and a cycle to a web (1 -> 2).
What they do NOT show: that the page means a path by "line" or a ring by "cycle"; that "a machine" is a line or "a web" a graph with two loops (the detective table only names them);
that the seven apps are in a cyclic order (the page gives them numbers 1 to 7 and names a return to Do, nothing more); that seven is a limit of the mind (the page says it, no data;
the working-memory figure usually quoted today is nearer four chunks than seven, as far as I know). Cycles also mean loops that can run forever; whether the AXI runtime
wants one is the question for Rajnish."""
import os, sys
from math import factorial, log2
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.cico_parser import CICOParser
from axi.engine.gate import Graph, check, apply
from axi.engine.inverse import invert, to_text
from test_shape_of_four import automorphisms, classes_of_names, graph_automorphisms
from test_signed_slots import etext, same

CELLS = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0), (2, 0), (2, 1), (2, 2)]


def path_edges(n): return [(i, i + 1, "") for i in range(n - 1)]


def cycle_edges(n): return path_edges(n) + [(n - 1, 0, "")]


def starts_and_ends(nodes, edges):
    """(nodes with nothing coming in, nodes with nothing going out) for a directed edge list."""
    has_in = {v for (_, v, *_) in edges}
    has_out = {u for (u, _, *_) in edges}
    return len([n for n in nodes if n not in has_in]), len([n for n in nodes if n not in has_out])


def loop_count(nodes, pairs):
    """Independent loops by a spanning forest: an edge that joins two nodes already connected closes one more loop."""
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x: x = parent[x]
        return x
    loops = 0
    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra == rb: loops += 1
        else: parent[ra] = rb
    return loops


def line_graph(n):
    g = Graph()
    for c in CELLS[:n]: g.add_node(c, 0)
    d = CICOParser.parse_transition_delta(" ".join(etext("ADD", (CELLS[i], CELLS[i + 1], "then"), 1) for i in range(n - 1)))
    assert check(g, d).ok
    apply(g, d)
    return g


def graph_starts_ends(g): return starts_and_ends(list(g.nodes), [(u, v) for (u, v, _) in g.edges])


def test_closing_a_line_into_a_cycle_divides_the_orderings_by_n_and_removes_the_start_and_the_end():
    for n in (4, 5, 6, 7):                                                  # 4, 5, 6 are the page's sequences, 7 is the scale
        nodes = list(range(n))
        ap, ac = automorphisms(n, path_edges(n)), automorphisms(n, cycle_edges(n))
        assert (len(ap), len(ac)) == (1, n)                                 # a path has no symmetry; a cycle has the n rotations
        assert classes_of_names(n, ap) == factorial(n) and classes_of_names(n, ac) == factorial(n - 1)
        assert abs(log2(factorial(n)) - log2(factorial(n - 1)) - log2(n)) < 1e-9        # closing the line loses log2(n) bits of layout
        assert starts_and_ends(nodes, path_edges(n)) == (1, 1)
        assert starts_and_ends(nodes, cycle_edges(n)) == (0, 0)
    assert [factorial(n) for n in (4, 5, 6)] == [24, 120, 720] and [factorial(n - 1) for n in (4, 5, 6)] == [6, 24, 120]


def test_in_the_gate_closing_a_five_step_life_is_one_delta_and_its_inverse_reopens_it_and_direction_matters():
    g = line_graph(5)                                                       # Birth, School, Job, Retirement, Death
    before = line_graph(5)
    assert len(g.edges) == 4 and len(graph_automorphisms(g)) == 1 and graph_starts_ends(g) == (1, 1)
    close = CICOParser.parse_transition_delta(etext("ADD", (CELLS[4], CELLS[0], "then"), 1))        # Death -> Birth
    assert check(g, close).ok
    undo = to_text(invert(g, close)); apply(g, close)
    assert len(g.edges) == 5 and len(graph_automorphisms(g)) == 5 and graph_starts_ends(g) == (0, 0)
    assert not check(g, close).ok and check(g, close).reason.startswith("identification")           # the same edge cannot be closed twice
    back = CICOParser.parse_transition_delta(undo)
    assert check(g, back).ok
    apply(g, back)
    assert same(g, before) and len(graph_automorphisms(g)) == 1
    wrong = CICOParser.parse_transition_delta(etext("ADD", (CELLS[0], CELLS[4], "then"), 1))        # Birth -> Death: a shortcut, not a loop
    assert check(g, wrong).ok
    apply(g, wrong)
    assert graph_starts_ends(g) == (1, 1) and len(graph_automorphisms(g)) == 1


def test_the_eighth_place_is_the_first_again_eight_in_a_line_become_seven_in_a_loop_by_one_delta():
    g = line_graph(8)                                                       # Do Re Mi Fa Sol La Ti Do'
    before = line_graph(8)
    assert len(g.nodes) == 8 and len(graph_automorphisms(g)) == 1
    ti, do2, do = CELLS[6], CELLS[7], CELLS[0]
    text = f"{etext('DEL', (ti, do2, 'then'), 1)} DEL[{do2[0]},{do2[1]}] {etext('ADD', (ti, do, 'then'), 1)}"
    octave = CICOParser.parse_transition_delta(text)
    assert check(g, octave).ok
    no_edge_deleted = CICOParser.parse_transition_delta(f"DEL[{do2[0]},{do2[1]}] {etext('ADD', (ti, do, 'then'), 1)}")
    refused = check(g, no_edge_deleted)
    assert not refused.ok and refused.reason.startswith("dangling edge")      # the eighth place cannot be dropped and leave an edge hanging
    undo = to_text(invert(g, octave)); apply(g, octave)
    assert len(g.nodes) == 7 and len(g.edges) == 7 and len(graph_automorphisms(g)) == 7 and graph_starts_ends(g) == (0, 0)
    back = CICOParser.parse_transition_delta(undo)
    assert check(g, back).ok
    apply(g, back)
    assert same(g, before) and len(g.nodes) == 8


def test_independent_loops_are_0_for_a_line_1_for_a_ring_2_with_a_chord_and_4_on_the_board():
    five = list(range(5))
    line = [(i, i + 1) for i in range(4)]
    ring = line + [(4, 0)]
    web = ring + [(0, 2)]
    assert [loop_count(five, p) for p in (line, ring, web)] == [0, 1, 2]
    for p in (line, ring, web):                                             # a second route to the same number: edges - nodes + components
        assert loop_count(five, p) == len(p) - 5 + 1
    plain = [(a, b) for a in CELLS for b in CELLS if a < b and abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1]
    assert len(plain) == 12 and loop_count(CELLS, plain) == 4               # the four small squares of the 3x3 board
    g = line_graph(5)
    pairs = lambda: [(u, v) for (u, v, _) in g.edges]
    assert loop_count(list(g.nodes), pairs()) == 0
    for (u, v) in ((CELLS[4], CELLS[0]), (CELLS[0], CELLS[2])):            # first the closing edge, then a chord
        d = CICOParser.parse_transition_delta(etext("ADD", (u, v, "then"), 1))
        assert check(g, d).ok
        apply(g, d)
    assert loop_count(list(g.nodes), pairs()) == 2


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
