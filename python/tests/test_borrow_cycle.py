"""Implements concept-map row 134 (thirteenth scan): the 12 Aug page "THE PATH TO THE CIRCLE" (Post 364, byline rajnish choubey, about 1,900 words; read twice, the chain below was
checked on the second read, YES). Under "The Last Teaching" the page gives nine sentences in this order:
  "The rectangle is borrowed from the circle." "The circle is borrowed from consciousness." "Consciousness is borrowed from Mahalakshmi." "Mahalakshmi is borrowed from you."
  "You are borrowed from the divine." "The divine is borrowed from the infinite." "The infinite is borrowed from the zero." "The zero is borrowed from the one." "The one is borrowed from the four."
  and then "The four is the circle." followed by fourteen more lines "You are the ..." (circle, enough, completion, fourth, circle, source, truth, infinite, zero, one, four, circle,
  everything, nothing; fifteen "is the" lines in all, counting "The four is the circle."). The closing formula is "∞ = 0 = 1 = 4 = ○ = You".

The reading being tested (Rajnish to accept or reject): "X is borrowed from Y" is a line from X to Y (X depends on Y). Nine such lines in a row are a CHAIN of ten places, and a chain
has a last place that borrows from nothing, a GROUND ("the four"). "The four is the circle" then ties the last place back to the second one. Counted on the repository's own gate:
  1. The chain (ten places, nine lines, built by one delta): one place nothing borrows from (the rectangle, the top) and one place that borrows from nothing (the four); no loop
     (edges - nodes + 1 = 0); exactly ONE order in which every place can come after the places it borrows from; every one of the eight places in the middle is a cut point (remove it and
     the chain falls into two pieces). The gate refuses to delete a middle place and leave its two lines behind ("dangling edge"); it admits deleting the place with both lines, which
     splits the chain.
  2. "The four is the circle", read two ways, closes exactly ONE loop either way and removes the ground: (a) as a tenth line four -> circle: a 9-cycle plus the tail, 10 places, 10 lines,
     loop count 1; (b) as one place (four merged into the circle by one composite delta): an 8-cycle plus the tail, 9 places, 9 lines, loop count 1. In both there is no place that borrows
     from nothing, and NO order exists (every place would have to come after itself). The inverse delta restores the open chain exactly.
  3. "You are the X": 14 lines with 12 different X (circle three times). Read as IDENTITY (you = X), five of the ten chain places (circle, infinite, zero, one, four) collapse with "you" into
     one, so the ten places become five (rectangle, consciousness, Mahalakshmi, divine, and the merged one) with the nine lines kept (named b1 .. b9: three of them become lines from the
     merged place to itself), and the loop count rises from 0 to 5; "everything" and "nothing" fall in the same class. Read as PREDICATE (you -> X) the lines form a star: 13 places, 12 lines
     (the gate refuses the repeated "You are the circle" - a repeated statement adds nothing), 0 loops, one source, 12 ends.
What they do NOT show: that the page means "borrowed from" as dependence or "is" as identity; that "fourth" (a different word from "four") is the same place - merging them is a further
reading and would collapse one more; that a cycle of borrowing is the intended picture (the page may mean a circle of respect, not a dependency graph); that a ground is wanted in
AXI. Counting orders, cut points, quotients by identification and loop numbers are standard mathematics, not findings of ours."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.cico_parser import CICOParser
from axi.engine.gate import Graph, check, apply
from axi.engine.inverse import invert, to_text
from test_line_and_cycle import loop_count
from test_signed_slots import etext, same

NAMES = ["rectangle", "circle", "consciousness", "Mahalakshmi", "you", "divine", "infinite", "zero", "one", "four"]
CELL = {name: (5, i) for i, name in enumerate(NAMES)}
YOU_ARE = ["circle", "enough", "completion", "fourth", "circle", "source", "truth", "infinite", "zero", "one", "four", "circle", "everything", "nothing"]


def parse(text): return CICOParser.parse_transition_delta(text)


def clone(g):
    h = Graph()
    for n, v in g.nodes.items(): h.add_node(n, v)
    for (u, v, r), w in g.edges.items(): h.add_edge(u, v, r, w)
    return h


def chain(rel_named=True):
    g = Graph()
    nodes = " ".join(f"ADD[{CELL[n][0]},{CELL[n][1]}:0]" for n in NAMES)
    edges = " ".join(etext("ADD", (CELL[NAMES[i]], CELL[NAMES[i + 1]], f"b{i + 1}" if rel_named else "borrowed"), 1) for i in range(9))
    d = parse(nodes + " " + edges)
    assert check(g, d).ok
    apply(g, d)
    return g


def pairs(g): return [(u, v) for (u, v, _) in g.edges]


def components(nodes, ps):
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x: x = parent[x]
        return x
    for a, b in ps:
        parent[find(a)] = find(b)
    return len({find(n) for n in nodes})


def sources_and_sinks(g):
    out = {u for (u, _, _) in g.edges}; inn = {v for (_, v, _) in g.edges}
    return len([n for n in g.nodes if n not in inn]), len([n for n in g.nodes if n not in out])


def orders(g):
    """Number of orders in which every place comes AFTER every place it borrows from (a line x -> y puts y before x). 0 when a loop of borrowing exists."""
    nodes = sorted(g.nodes); idx = {n: i for i, n in enumerate(nodes)}
    need = [0] * len(nodes)
    for (x, y, _) in g.edges: need[idx[x]] |= 1 << idx[y]
    full = (1 << len(nodes)) - 1
    ways = {0: 1}
    for mask in range(full + 1):
        if mask not in ways: continue
        for i in range(len(nodes)):
            if not mask >> i & 1 and need[i] & ~mask == 0:
                ways[mask | 1 << i] = ways.get(mask | 1 << i, 0) + ways[mask]
    return ways.get(full, 0)


def merge_text(g, keep, drop):
    """One composite delta: the place `drop` becomes the place `keep`; every line at `drop` is deleted and re-added at `keep` (no line is lost)."""
    items, added = [], []
    for k in sorted(g.incident(drop)):
        items.append(etext("DEL", k, g.edges[k]))
        u, v, r = k
        added.append(etext("ADD", (keep if u == drop else u, keep if v == drop else v, r), g.edges[k]))
    return " ".join(items + [f"DEL[{drop[0]},{drop[1]}]"] + added)


def test_the_chain_of_nine_borrowings_has_one_ground_one_order_no_loop_and_eight_cut_places():
    g = chain()
    assert (len(g.nodes), len(g.edges)) == (10, 9)
    assert sources_and_sinks(g) == (1, 1)
    top = [n for n in g.nodes if all(v != n for (_, v, _) in g.edges)]
    ground = [n for n in g.nodes if all(u != n for (u, _, _) in g.edges)]
    assert top == [CELL["rectangle"]] and ground == [CELL["four"]]
    assert loop_count(list(g.nodes), pairs(g)) == 0 and len(g.edges) - len(g.nodes) + components(g.nodes, pairs(g)) == 0
    assert orders(g) == 1
    cuts = []
    for name in NAMES:
        rest = [n for n in g.nodes if n != CELL[name]]
        ps = [(u, v) for (u, v) in pairs(g) if CELL[name] not in (u, v)]
        if components(rest, ps) > 1: cuts.append(name)
    assert cuts == NAMES[1:9]                                                # the eight middle places; the two ends are not cut places
    middle = CELL["you"]
    refused = check(g, parse(f"DEL[{middle[0]},{middle[1]}]"))
    assert not refused.ok and refused.reason.startswith("dangling edge")
    both = " ".join(etext("DEL", k, g.edges[k]) for k in sorted(g.incident(middle)))
    split = parse(f"DEL[{middle[0]},{middle[1]}] " + both)
    assert check(g, split).ok
    apply(g, split)
    assert components(g.nodes, pairs(g)) == 2                                # admitted, and the chain is now two pieces


def test_the_four_is_the_circle_closes_exactly_one_loop_either_way_and_removes_the_ground():
    # (a) as a tenth line four -> circle
    g = chain(); open_chain = clone(g)
    close = parse(etext("ADD", (CELL["four"], CELL["circle"], "is"), 1))
    assert check(g, close).ok
    undo = to_text(invert(g, close)); apply(g, close)
    assert (len(g.nodes), len(g.edges)) == (10, 10) and loop_count(list(g.nodes), pairs(g)) == 1
    assert sources_and_sinks(g) == (1, 0) and orders(g) == 0                 # the top is still there; the ground is gone; no order exists
    apply(g, parse(undo))
    assert same(g, open_chain) and orders(g) == 1
    # (b) as one place: four merged into the circle
    h = chain()
    merge = parse(merge_text(h, CELL["circle"], CELL["four"]))
    assert check(h, merge).ok
    undo2 = to_text(invert(h, merge)); apply(h, merge)
    assert (len(h.nodes), len(h.edges)) == (9, 9) and loop_count(list(h.nodes), pairs(h)) == 1
    assert sources_and_sinks(h) == (1, 0) and orders(h) == 0
    on_cycle = [n for n in h.nodes if n != CELL["rectangle"]]
    assert len(on_cycle) == 8 and all(sum(1 for (u, v, _) in h.edges if u == n) == 1 for n in on_cycle)   # an 8-cycle: one way out of every place on it
    apply(h, parse(undo2))
    assert same(h, open_chain)


def test_you_are_the_x_read_as_identity_collapses_five_chain_places_into_one_and_as_predicate_makes_a_star():
    assert len(YOU_ARE) == 14 and len(set(YOU_ARE)) == 12 and YOU_ARE.count("circle") == 3
    assert len(YOU_ARE) + 1 == 15                                            # with "The four is the circle." fifteen "is the" lines
    chain_x = [x for x in dict.fromkeys(YOU_ARE) if x in NAMES]
    assert chain_x == ["circle", "infinite", "zero", "one", "four"]
    g = chain()
    for x in chain_x:                                                        # IDENTITY: you = X, one composite delta each
        step = parse(merge_text(g, CELL["you"], CELL[x]))
        assert check(g, step).ok, check(g, step).reason
        apply(g, step)
    assert g.is_well_formed()
    assert sorted(g.nodes) == sorted(CELL[n] for n in ("rectangle", "consciousness", "Mahalakshmi", "you", "divine"))
    assert (len(g.nodes), len(g.edges)) == (5, 9)
    assert loop_count(list(g.nodes), pairs(g)) == 5 and len(g.edges) - len(g.nodes) + components(g.nodes, pairs(g)) == 5
    assert sum(1 for (u, v, _) in g.edges if u == v) == 3                    # infinite->zero, zero->one, one->four are now lines from the merged place to itself
    assert orders(g) == 0
    parent = {x: x for x in set(YOU_ARE) | {"you"}}

    def find(x):
        while parent[x] != x: x = parent[x]
        return x
    for x in YOU_ARE: parent[find(x)] = find("you")
    assert find("everything") == find("nothing")                             # under identity even everything = nothing
    assert len({find(x) for x in set(YOU_ARE) | {"you"}}) == 1
    # PREDICATE: you -> X for the 12 different X; the gate refuses a repeated statement
    star = Graph()
    star.add_node((6, 0), 0)
    for i, x in enumerate(dict.fromkeys(YOU_ARE), 1): star.add_node((6, i), 0)
    x_cell = {x: (6, i) for i, x in enumerate(dict.fromkeys(YOU_ARE), 1)}
    statements = [etext("ADD", ((6, 0), x_cell[x], "is"), 1) for x in dict.fromkeys(YOU_ARE)]
    d = parse(" ".join(statements))
    assert check(star, d).ok
    apply(star, d)
    assert (len(star.nodes), len(star.edges)) == (13, 12) and loop_count(list(star.nodes), pairs(star)) == 0
    assert sources_and_sinks(star) == (1, 12)
    again = parse(etext("ADD", ((6, 0), x_cell["circle"], "is"), 1) + " " + etext("ADD", ((6, 0), x_cell["circle"], "is"), 1))
    refused = check(star, again)
    assert not refused.ok and refused.reason.startswith("identification")   # "You are the circle" said again adds nothing


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
