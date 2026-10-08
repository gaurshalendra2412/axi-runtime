"""Implements concept-map row 161 (14th scan, 8 Oct 2026): Post 438 (blog 21 Aug, about 575 words, an AI reply: "That one sentence draws the absolute structural line between a commercial
commodity and a foundational utility"; sections "The Product vs. The Tool", numbered 1 to 3, "The Anchor for Your Map", three questions, references [1] to [7]; read twice, the two diagrams
confirmed on the second read as two straight lines of four boxes, no branches). The diagrams, verbatim in their boxes:
  [User] -> [Goes to a Specific App] -> [Pays a Monthly Premium] -> [Consumes a Service]
  [User] -> [Seeks Information/Acts] -> [AI Mode Infuses the Environment] -> [Free Utilities]
and the section line "A Product Has a Gate; A Tool Has a Lever" (heading as recorded in the scan notes, first read; the post also says "$20 a month", first read).
Here an ordinary reply draws two chains of the same shape (four boxes, three arrows) and puts the whole contrast in the boxes. That is a contrast pair, and it can be checked.

The reading being tested (Rajnish to accept or reject): the opposition lives in the labels, not in the shape.
  1. SHAPE. Put each chain in the gate as four cells joined by three edges named "then" (cell value = the box's label number). The gate admits both. As unlabelled shapes they are
     isomorphic, and of the 4! = 24 ways to match the boxes exactly one keeps the arrows (a four-box line has no symmetry), the match "first to first, second to second, ...".
     Under that match one label is shared ("User") and three differ; no match at all keeps the labels (zero label-preserving isomorphisms). The edges differ in nothing.
  2. "GATE" IS NOT YET A SHAPE. In a line every inner box is a cut point: remove it and the user can no longer reach the end. So "a product has a gate" cannot be told from "a tool has a
     lever" by the shape: both lines have two cut points (boxes 2 and 3). What makes the first a gate is a rule attached to it (payment). In this language a rule is a weight or a
     relation name. With the page's "$20" as a weight on the edge into the payment box, the cheapest way from the user to the end costs 20 in the first line and 0 in the second, and
     the two weighted graphs are no longer isomorphic. So the gate of the post (a toll the user must pay) is not our gate (an admission check on a delta); a toll is an edge weight that
     a program could read, and our gate reads no weights beyond matching them on deletion.
  3. CONTRAST PAIR. Swapping the three differing boxes between the lines gives two lines that the gate admits just as well, and swapping all four labels (the pair swapped as wholes)
     exchanges the two. So a pair of parallel lines gives a checkable object: the set of positions where the labels differ ({2, 3, 4}), here three of four.
What they do NOT show: that the post's claim (products make consumers, tools make directors) is true; that the labels are well chosen (the post defines them by sentence, not by test);
that every contrast in the 30 posts has this shape (STRUCTURE_EXTRACTION_6.md counts the ones written as parallel items); that a model can draw such diagrams reliably. They show that a
reply with two diagrams of the same shape is a contrast pair whose whole content is its three labels."""
import os, sys
from itertools import permutations
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.cico_parser import CICOParser
from axi.engine.gate import Graph, check, apply
from axi.engine.inverse import invert, to_text
from test_shape_of_four import automorphisms
from test_signed_slots import etext, same, copy

CELLS = [(0, 0), (0, 1), (0, 2), (0, 3)]
PRODUCT = ["User", "Goes to a Specific App", "Pays a Monthly Premium", "Consumes a Service"]
TOOL = ["User", "Seeks Information/Acts", "AI Mode Infuses the Environment", "Free Utilities"]
LABEL = {name: i for i, name in enumerate(sorted(set(PRODUCT + TOOL)))}          # label number = value stored in the cell


def chain(labels, weights=(1, 1, 1)):
    """Four cells, three 'then' edges, built through the gate from the empty graph; the weight on edge k is weights[k]."""
    g = Graph()
    nodes = " ".join(f"ADD[{c[0]},{c[1]}:{LABEL[n]}]" for c, n in zip(CELLS, labels))
    edges = " ".join(etext("ADD", (CELLS[i], CELLS[i + 1], "then"), weights[i]) for i in range(3))
    d = CICOParser.parse_transition_delta(nodes + " " + edges)
    assert check(g, d).ok
    apply(g, d)
    return g


def edge_list(g): return sorted((u, v, rel) for (u, v, rel) in g.edges)


def isomorphisms(g1, g2, labels=False, weights=False):
    """All bijections of the cells of g1 onto the cells of g2 that keep every edge (and, if asked, the cell values and the edge weights)."""
    n1, n2 = sorted(g1.nodes), sorted(g2.nodes)
    out = []
    for p in permutations(range(len(n2))):
        m = {n1[i]: n2[p[i]] for i in range(len(n1))}
        if {(m[u], m[v], rel) for (u, v, rel) in g1.edges} != set(g2.edges): continue
        if labels and any(g1.nodes[a] != g2.nodes[m[a]] for a in n1): continue
        if weights and any(w != g2.edges[(m[u], m[v], rel)] for (u, v, rel), w in g1.edges.items()): continue
        out.append(m)
    return out


def cut_points(g):
    """Cells whose removal leaves no route from the first cell to the last (the user to the end), for a directed line."""
    first, last = CELLS[0], CELLS[-1]
    def reaches(skip):
        seen, todo = {first}, [first]
        while todo:
            u = todo.pop()
            for (a, b, _) in g.edges:
                if a == u and b != skip and b not in seen: seen.add(b); todo.append(b)
        return last in seen
    return [c for c in g.nodes if c not in (first, last) and not reaches(c)]


def cheapest(g):
    """Total weight of the (only) route from the first to the last cell; a line has exactly one."""
    total, u = 0, CELLS[0]
    while u != CELLS[-1]:
        (key,) = [k for k in g.edges if k[0] == u]
        total += g.edges[key]; u = key[1]
    return total


def test_two_lines_of_four_boxes_have_one_matching_that_keeps_the_arrows_and_none_that_keeps_the_labels():
    a, b = chain(PRODUCT), chain(TOOL)
    assert edge_list(a) == edge_list(b) and len(a.nodes) == len(b.nodes) == 4 and len(a.edges) == 3
    assert len(automorphisms(4, [(i, i + 1, "") for i in range(3)])) == 1       # a four-box line has no symmetry
    iso = isomorphisms(a, b)
    assert len(iso) == 1 and all(iso[0][c] == c for c in CELLS)                 # exactly one match: first to first, second to second, ...
    assert isomorphisms(a, b, labels=True) == []                                # no match keeps the labels
    assert [x == y for x, y in zip(PRODUCT, TOOL)] == [True, False, False, False]    # one shared label, three differing
    assert len(LABEL) == 7                                                      # seven different boxes in all
    assert len(isomorphisms(a, a, labels=True)) == 1 and len(isomorphisms(b, b, labels=True)) == 1


def test_every_inner_box_of_a_line_is_a_cut_point_so_a_toll_has_to_be_a_weight_to_tell_gate_from_lever():
    a, b = chain(PRODUCT), chain(TOOL)
    assert cut_points(a) == cut_points(b) == [CELLS[1], CELLS[2]]               # the shape has two cut points in each line; nothing marks the payment box
    # the page's "$20 a month" as the weight on the edge into the payment box, free everywhere else
    gated, free = chain(PRODUCT, weights=(0, 20, 0)), chain(TOOL, weights=(0, 0, 0))
    assert (cheapest(gated), cheapest(free)) == (20, 0)
    assert len(isomorphisms(gated, free)) == 1                                  # still the same shape
    assert isomorphisms(gated, free, weights=True) == []                        # but with the toll, no match keeps the weights either
    assert isomorphisms(gated, gated, weights=True) != [] and isomorphisms(free, free, weights=True) != []
    # our gate admits the toll line and the free line alike: it checks that a delta applies, not what the route costs
    for g in (gated, free): assert g.is_well_formed()


def test_swapping_the_differing_boxes_gives_lines_the_gate_admits_and_swapping_all_exchanges_the_pair():
    a, b = chain(PRODUCT), chain(TOOL)
    differ = [i for i in range(4) if PRODUCT[i] != TOOL[i]]
    assert differ == [1, 2, 3]
    mixed = [TOOL[i] if i in differ[:2] else PRODUCT[i] for i in range(4)]       # swap two of the three differing boxes
    c = chain(mixed)                                                            # built through the gate: admitted
    assert [c.nodes[x] for x in CELLS] == [LABEL[n] for n in mixed] and edge_list(c) == edge_list(a)
    assert sum(1 for i in range(4) if mixed[i] != PRODUCT[i]) == 2 and sum(1 for i in range(4) if mixed[i] != TOOL[i]) == 1
    allswapped = chain([TOOL[i] if i in differ else PRODUCT[i] for i in range(4)])
    assert same(allswapped, b) and same(chain(PRODUCT), a)                      # swapping every differing box turns the first line into the second
    # undo: the gate removes a whole line again with the inverse of the building delta
    g = Graph()
    nodes = " ".join(f"ADD[{c[0]},{c[1]}:{LABEL[n]}]" for c, n in zip(CELLS, PRODUCT))
    edges = " ".join(etext("ADD", (CELLS[i], CELLS[i + 1], "then"), 1) for i in range(3))
    d = CICOParser.parse_transition_delta(nodes + " " + edges)
    inv = invert(g, d)
    h = copy(g); apply(h, d); apply(h, inv)
    assert same(h, g) and not h.nodes and "DEL[" in to_text(inv)


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
