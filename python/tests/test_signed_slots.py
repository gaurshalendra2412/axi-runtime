"""Implements concept-map row 58 (blog 2 Aug "Frequency of the Flesh" and 30 Aug "+1 / -1" posts) and closes the TEST of row 38
(inverse closure). It also pins row 59's "one counterexample refutes a 100% claim" as a property of the gate.

The reading being tested (Rajnish to accept or reject):
    2 Aug: action / deletion / anchor      30 Aug: Durga +1, Kali -1, Shiva 0 (the ground)
    ->  in a delta, every slot is +1 (ADD), -1 (DEL) or 0 (untouched), and the host state is the ground that does not move.

What these tests show (all on the gate and the new `invert`, in this repository):
  1. Slot arithmetic is exact. For a node-only delta, the gate admits it if and only if, for every slot, present(0 or 1) + (+1/-1/0)
     stays in {0, 1}. Checked exhaustively on a 2x2 grid (16 states x 81 deltas) and on 20,000 random 3x3 cases.
  2. Inverse closure. For ~thousands of random graphs (with edges) and random deltas, every delta the gate admits has an inverse that is
     (a) admitted on the new graph and (b) restores the old graph exactly. The inverse of the inverse is the original delta.
  3. Balance. For every admitted delta, the change in node count is #ADD - #DEL, the same for edges, and delta + inverse sums to zero
     slot by slot. (This is bookkeeping that follows from rule 1; it is not an independent law.)
  4. Veto. Adding ONE violating item to an admitted delta makes the whole delta rejected, for four kinds of violation.
What they do NOT show: that the posts mean ADD/DEL by +1/-1; that the gate is complete (it checks applicability, not task invariants,
see PAPER_CORRECTIONS item 20); anything about the model. SET (in-place update) is still missing from the language (row 17)."""
import os, random, sys
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.cico_parser import CICOParser
from axi.engine.gate import Graph, check, apply
from axi.engine.inverse import invert, to_text

CELLS = [(r, c) for r in range(3) for c in range(3)]
RELS = ["adj", "dep"]


def same(a: Graph, b: Graph) -> bool:
    inc = lambda g: {n: set(g.incident(n)) for n in g.nodes if g.incident(n)}
    return a.nodes == b.nodes and a.edges == b.edges and inc(a) == inc(b)


def copy(g: Graph) -> Graph:
    h = Graph()
    for n, v in g.nodes.items(): h.add_node(n, v)
    for (u, v, rel), w in g.edges.items(): h.add_edge(u, v, rel, w)
    return h


def etext(op, k, w): return f"{op}[({k[0][0]},{k[0][1]})->({k[1][0]},{k[1][1]}):{k[2]}#{w}]"


def random_graph(rng):
    g = Graph()
    for n in CELLS:
        if rng.random() < 0.6: g.add_node(n, rng.randint(0, 9))
    nodes = list(g.nodes)
    for _ in range(rng.randint(0, 9)):
        if nodes:
            u, v, rel = rng.choice(nodes), rng.choice(nodes), rng.choice(RELS)
            if (u, v, rel) not in g.edges: g.add_edge(u, v, rel, rng.randint(-2, 5))
    return g


def random_delta_text(g, rng):
    items = []
    for n in CELLS:
        if rng.random() < 0.25:
            if n in g.nodes:
                items.append(f"DEL[{n[0]},{n[1]}]")
                if rng.random() < 0.85:
                    items += [etext("DEL", k, g.edges[k]) for k in g.incident(n)]
            else:
                items.append(f"ADD[{n[0]},{n[1]}:{rng.randint(0, 9)}]")
    for _ in range(rng.randint(0, 3)):
        k = (rng.choice(CELLS), rng.choice(CELLS), rng.choice(RELS))
        if k in g.edges:
            if rng.random() < 0.7: items.append(etext("DEL", k, g.edges[k] if rng.random() < 0.9 else g.edges[k] + 1))
        else:
            items.append(etext("ADD", k, rng.randint(-2, 5)))
    items = list(dict.fromkeys(items)) if rng.random() < 0.9 else items
    rng.shuffle(items)
    return " ".join(items)


def admitted_samples(n_tries, seed):
    rng = random.Random(seed); out = []
    for _ in range(n_tries):
        g = random_graph(rng); txt = random_delta_text(g, rng)
        d = CICOParser.parse_transition_delta(txt)
        if check(g, d).ok: out.append((g, d))
    return out


def test_slot_arithmetic_exhaustive_2x2():
    cells = CELLS[:0] + [(0, 0), (0, 1), (1, 0), (1, 1)]
    n = 0
    for state in range(16):
        present = [(state >> i) & 1 for i in range(4)]
        g = Graph()
        for i, cell in enumerate(cells):
            if present[i]: g.add_node(cell, 7)
        for ops in range(81):
            d_i, items, x = [], [], ops
            for i, cell in enumerate(cells):
                d = x % 3 - 1; x //= 3; d_i.append(d)
                if d == 1: items.append(f"ADD[{cell[0]},{cell[1]}:5]")
                elif d == -1: items.append(f"DEL[{cell[0]},{cell[1]}]")
            r = check(g, CICOParser.parse_transition_delta(" ".join(items)))
            expect = all(0 <= present[i] + d_i[i] <= 1 for i in range(4))
            assert r.ok == expect, (present, d_i, r)
            n += 1
    assert n == 1296


def test_slot_arithmetic_random_3x3():
    rng = random.Random(11); n = ok = 0
    for _ in range(20000):
        g = Graph(); present = {}
        for c in CELLS:
            present[c] = int(rng.random() < 0.5)
            if present[c]: g.add_node(c, 3)
        items, dd = [], {}
        for c in CELLS:
            d = rng.choice([-1, 0, 0, 1]); dd[c] = d
            if d == 1: items.append(f"ADD[{c[0]},{c[1]}:2]")
            if d == -1: items.append(f"DEL[{c[0]},{c[1]}]")
        r = check(g, CICOParser.parse_transition_delta(" ".join(items)))
        expect = all(0 <= present[c] + dd[c] <= 1 for c in CELLS)
        assert r.ok == expect, (present, dd, r)
        n += 1; ok += r.ok
    assert 1000 < ok < n  # both outcomes are well represented, so the equality above is not trivial


def test_inverse_closure_and_involution():
    S = admitted_samples(30000, 5)
    assert len(S) >= 2000, len(S)
    kinds = {"ADDn": 0, "DELn": 0, "ADDe": 0, "DELe": 0, "DELn+DELe": 0}
    for g, d in S:
        for x in d.node_deltas: kinds[x.op + "n"] += 1
        for x in d.edge_deltas: kinds[x.op + "e"] += 1
        if any(x.op == "DEL" for x in d.node_deltas) and any(x.op == "DEL" for x in d.edge_deltas): kinds["DELn+DELe"] += 1
        inv = invert(g, d)
        h = copy(g); apply(h, d)
        r = check(h, inv)
        assert r.ok, (to_text(d), to_text(inv), r)           # (a) the inverse is admitted on the new graph
        apply(h, inv)
        assert same(h, g), (to_text(d), to_text(inv))         # (b) and restores the old graph exactly
        g2 = copy(g); apply(g2, d)
        back = invert(g2, inv)                                # involution: inverse of the inverse is the original delta
        assert sorted(map(str, back.node_deltas)) == sorted(map(str, d.node_deltas))
        assert sorted(map(str, back.edge_deltas)) == sorted(map(str, d.edge_deltas))
    assert all(v >= 200 for v in kinds.values()), kinds       # every item kind, and node deletes with their edges, were exercised
    print("   admitted:", len(S), kinds)


def test_balance_follows_from_the_gate():
    for g, d in admitted_samples(15000, 6):
        h = copy(g); apply(h, d)
        sn = sum(1 if x.op == "ADD" else -1 for x in d.node_deltas)
        se = sum(1 if x.op == "ADD" else -1 for x in d.edge_deltas)
        assert len(h.nodes) - len(g.nodes) == sn and len(h.edges) - len(g.edges) == se
        inv = invert(g, d)
        sign = lambda x: 1 if x.op == "ADD" else -1
        assert sum(map(sign, d.node_deltas)) + sum(map(sign, inv.node_deltas)) == 0
        assert sum(map(sign, d.edge_deltas)) + sum(map(sign, inv.edge_deltas)) == 0


def test_one_bad_item_vetoes_the_whole_delta():
    S = [(g, d) for g, d in admitted_samples(20000, 7) if len(d.node_deltas) + len(d.edge_deltas) >= 2][:600]
    assert len(S) >= 300, len(S)
    made = {"missing": 0, "exists": 0, "repeat": 0, "dangling": 0}
    for g, d in S:
        txt = to_text(d)
        assert check(g, CICOParser.parse_transition_delta(txt)).ok                       # control: the good delta is admitted
        touched = {(x.r, x.c) for x in d.node_deltas}
        absent = [c for c in CELLS if c not in g.nodes and c not in touched]
        if absent:
            c = absent[0]; made["missing"] += 1
            assert not check(g, CICOParser.parse_transition_delta(txt + f" DEL[{c[0]},{c[1]}]")).ok
        present = [c for c in CELLS if c in g.nodes and c not in touched]
        if present:
            c = present[0]; made["exists"] += 1
            assert not check(g, CICOParser.parse_transition_delta(txt + f" ADD[{c[0]},{c[1]}:1]")).ok
        made["repeat"] += 1
        first = txt.split(" ")[0]
        assert not check(g, CICOParser.parse_transition_delta(txt + " " + first)).ok
        del_edges = {(x.u, x.v, x.relation) for x in d.edge_deltas if x.op == "DEL"}
        for c in present:
            if g.incident(c) and not (g.incident(c) & del_edges):
                made["dangling"] += 1
                assert not check(g, CICOParser.parse_transition_delta(txt + f" DEL[{c[0]},{c[1]}]")).ok
                break
    assert all(v >= 100 for v in made.values()), made
    print("   veto cases built:", made)


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
