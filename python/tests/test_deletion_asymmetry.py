"""Implements concept-map row 65 (blog 14 Aug, "That habit - writing for months in notebooks, showing it to a single person, and then
deleting the file the moment it was seen ..." - deletion as 'evolutionary armor': "they can't index what isn't there") and sharpens
row 38 (inverse closure, test_signed_slots.py). Earlier source of the same idea: 23 Jul "sacrifice what u love most" (row 52).

The reading being tested (Rajnish to accept or reject): DEL is a real erasure, and an erasure you can undo is an erasure you kept a copy of.
Facts about OUR language and gate (all four are about the code in this repository, none are about the post):
  1. DEL of an EDGE carries its weight and the gate checks it (match condition). DEL of a NODE carries no value and the gate cannot see one:
     any value 0..9 is deleted without complaint. So the language is asymmetric: edges are matched on content, nodes only on identity.
  2. Therefore apply() is not injective. Two hosts that differ only in the deleted node's value become the SAME graph after the same
     DEL[r,c]. The value is gone from the state. Whatever restores it must have been saved BEFORE the delete (`invert(g, d)` is exactly that
     saved copy). Retention (to undo) and armor (to leave no trace) pull in opposite directions; this is a design choice, not a bug.
  3. An edge-only delta is invertible from its own text (no host needed); a delta with a node DEL is not.
  4. After a delete the host holds no residue of the item: nodes, edges and the incidence index equal those of a graph built without it.
What they do NOT show: that the post's deletion is this operation; any claim about surveillance or privacy; that the gate SHOULD match
node values (that would catch stale node beliefs, a possible upgrade `DEL[r,c:val]`, not implemented, not measured)."""
import os, random, sys
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.cico_parser import CICOParser
from axi.engine.gate import Graph, check, apply
from axi.engine.inverse import invert, to_text
from test_signed_slots import same, copy, random_graph, random_delta_text

P = CICOParser.parse_transition_delta


def test_node_delete_is_blind_to_the_value_edge_delete_is_not():
    for v in range(10):
        g = Graph(); g.add_node((1, 1), v)
        assert check(g, P("DEL[1,1]")).ok                                         # every stored value is deleted without a question
    g = Graph(); g.add_node((0, 0), 1); g.add_node((1, 1), 1); g.add_edge((0, 0), (1, 1), "adj", 3)
    ok = [w for w in range(-2, 9) if check(g, P(f"DEL[(0,0)->(1,1):adj#{w}]")).ok]
    assert ok == [3]                                                              # only the true weight matches
    assert check(g, P("DEL[(0,0)->(1,1):adj#4]")).reason.startswith("match: edge")


def test_erasure_is_not_injective_so_undo_needs_a_copy_taken_before():
    g1, g2 = Graph(), Graph()
    for g, v in ((g1, 3), (g2, 7)):
        g.add_node((0, 0), 5); g.add_node((1, 1), v); g.add_edge((0, 0), (1, 1), "dep", 2)
    d = P("DEL[(0,0)->(1,1):dep#2] DEL[1,1]")
    assert check(g1, d).ok and check(g2, d).ok
    i1, i2 = invert(g1, d), invert(g2, d)                                         # taken BEFORE applying
    h1, h2 = copy(g1), copy(g2); apply(h1, d); apply(h2, d)
    assert same(h1, h2)                                                           # the value (3 vs 7) is gone from the state
    assert to_text(i1) != to_text(i2) and "ADD[1,1:3]" in to_text(i1) and "ADD[1,1:7]" in to_text(i2)
    apply(h1, i1); apply(h2, i2)
    assert same(h1, g1) and same(h2, g2)                                          # with the saved copy both come back exactly


def edge_only_text(g, rng):
    items = [f"DEL[({k[0][0]},{k[0][1]})->({k[1][0]},{k[1][1]}):{k[2]}#{w}]" for k, w in g.edges.items() if rng.random() < 0.5]
    nodes = list(g.nodes)
    for _ in range(rng.randint(0, 3)):
        if len(nodes) >= 1:
            u, v, rel = rng.choice(nodes), rng.choice(nodes), rng.choice(["adj", "dep"])
            if (u, v, rel) not in g.edges:
                items.append(f"ADD[({u[0]},{u[1]})->({v[0]},{v[1]}):{rel}#{rng.randint(-2, 5)}]")
    return " ".join(dict.fromkeys(items))


def test_edge_only_delta_inverts_from_its_text_a_node_delete_does_not():
    rng = random.Random(21); edge_only = node_del = 0
    for _ in range(6000):                                                         # (a) deltas made only of edge items
        g = random_graph(rng); txt = edge_only_text(g, rng)
        if not txt: continue
        d = P(txt)
        if not check(g, d).ok: continue
        edge_only += 1
        assert to_text(invert(Graph(), d)) == to_text(invert(g, d))               # no host needed
    for _ in range(6000):                                                         # (b) deltas that delete a node
        g = random_graph(rng); d = P(random_delta_text(g, rng))
        if not check(g, d).ok or not any(x.op == "DEL" for x in d.node_deltas): continue
        node_del += 1
        try: invert(Graph(), d); raise AssertionError("inverse of a node delete worked without the host")
        except KeyError: pass                                                     # the deleted value lives only in the host
    assert edge_only >= 1000 and node_del >= 300, (edge_only, node_del)


def test_no_residue_after_delete():
    rng = random.Random(5); n_checked = 0
    for _ in range(4000):
        g = random_graph(rng); d = P(random_delta_text(g, rng))
        dead = [(x.r, x.c) for x in d.node_deltas if x.op == "DEL"]
        if not dead or not check(g, d).ok: continue
        h = copy(g); apply(h, d)
        for n in dead:
            assert n not in h.nodes and not h.incident(n) and n not in h._inc
            assert all(n not in (u, v) for (u, v, _) in h.edges)
        ref = Graph()                                                              # a graph built without the deleted items
        gone_e = {(e.u, e.v, e.relation) for e in d.edge_deltas if e.op == "DEL"}
        for n, v in g.nodes.items():
            if n not in dead: ref.add_node(n, v)
        for k, w in g.edges.items():
            if k not in gone_e: ref.add_edge(k[0], k[1], k[2], w)
        for x in d.node_deltas:
            if x.op == "ADD": ref.add_node((x.r, x.c), x.val)
        for e in d.edge_deltas:
            if e.op == "ADD": ref.add_edge(e.u, e.v, e.relation, e.weight)
        assert same(h, ref), (g.nodes, d)
        n_checked += 1
    assert n_checked >= 300, n_checked


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
