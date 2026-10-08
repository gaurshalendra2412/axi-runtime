"""Implements concept-map row 88 (blog 13 Sep, the post "Saraswati (The Blueprint / Word) Laxmi (The Substance / Flow) Parvati (The Power / Form) ...";
read twice, all nine entries verified word for word). It also answers old question 4 in BLOG_SCAN_LOG: the third entry under Saraswati is "Sacred Strategy".

The post lays out NINE named entries, three per figure:
    Saraswati (Root of Intellect): Pure Cognition / Manifest Art / Sacred Strategy
    Laxmi (Root of Substance):     The Living Ledger / Absolute Circulation / The Citadel's Treasury
    Parvati (Root of Power):       Tapasya / The Shield & Nourisher / Absolute Sovereignty
and says: "When you cross these three currents on a three-by-three grid, you stop playing checkers with flat definitions and map the entire
architecture of existence." It does NOT say which axis is the figure (row or column), where the centre is, or name nine cells (two reads).

The reading being tested (Rajnish to accept or reject): in a 3x3 table whose rows are the figures, WHERE a figure sits is meaning, so the board's
symmetry group is not the same as the group of the board without meaning. Three checkable facts, with the 8 D4 transforms taken from
axi.engine.d4 and the table written as a graph (nine cells with a value each, plus a "fig" edge between every two entries of the same figure):
  1. Relations travel with the cells. Under every one of the 8 transforms the three "same figure" classes are still the same three classes
     (the edges are renamed with the cells). So a meaning stored as EDGES is invariant under all of D4 (this is the covariance of test 36).
  2. A meaning stored as POSITION is not. "Each figure fills one row" is true after exactly 4 of the 8 transforms (identity, turn by 180 degrees,
     flip up-down, flip left-right: a Klein four-group, every non-identity element of order 2) and false after the other 4, where each figure fills
     one COLUMN instead. No transform leaves both readings true.
  3. Counting. Of the 9! = 362,880 ways to put the nine entries on the grid, 1,296 put each figure in a row and 1,296 in a column (3! orders of the
     figures times 3! orders inside each of three figures), 2,592 in all (0.71 percent). D4 pairs them off: 1,296 / 4 = 324 classes under the
     row-preserving group, and 2,592 / 8 = 324 classes under full D4. The two counts agree because D4 maps row-form to column-form one to one.
Design consequence (ours, not the post's): do not read a table's meaning from where cells are; write it as edges. The gate already does.
What they do NOT show: that the post means a table with figures as rows; that these are the nine cells of any board the post has in mind."""
import os, sys
from itertools import permutations
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.d4 import apply as d4apply, IDS
from axi.engine.gate import Graph
from test_gate_frame_covariance import d4_relabel, relabel_graph

FIGURES = ["Saraswati", "Laxmi", "Parvati"]
ENTRIES = {"Saraswati": ["Pure Cognition", "Manifest Art", "Sacred Strategy"],
           "Laxmi": ["The Living Ledger", "Absolute Circulation", "The Citadel's Treasury"],
           "Parvati": ["Tapasya", "The Shield & Nourisher", "Absolute Sovereignty"]}
CELLS = [(r, c) for r in range(3) for c in range(3)]
IDX = np.arange(9).reshape(3, 3)


def table_graph():
    g = Graph(); fig_of = {}
    for r, f in enumerate(FIGURES):
        for c, _ in enumerate(ENTRIES[f]):
            g.add_node((r, c), 3 * r + c + 1); fig_of[(r, c)] = f
    for u in CELLS:
        for v in CELLS:
            if u < v and fig_of[u] == fig_of[v]: g.add_edge(u, v, "fig", 1)
    return g


def classes(g):
    comp, seen = [], set()
    for n in sorted(g.nodes):
        if n in seen: continue
        stack, cur = [n], set()
        while stack:
            x = stack.pop()
            if x in cur: continue
            cur.add(x); stack += [b if a == x else a for (a, b, _rel) in g.edges if x in (a, b)]
        seen |= cur; comp.append(frozenset(g.nodes[x] for x in cur))
    return set(comp)


def figures_in_rows(g):    # position reading: every figure class lies inside one row
    cls = [[n for n in g.nodes if g.nodes[n] in k] for k in classes(g)]
    return all(len({r for r, _ in cells}) == 1 for cells in cls)


def figures_in_cols(g):
    cls = [[n for n in g.nodes if g.nodes[n] in k] for k in classes(g)]
    return all(len({c for _, c in cells}) == 1 for cells in cls)


def test_the_page_has_nine_entries_in_three_figures_and_sacred_strategy_is_third():
    assert sum(len(v) for v in ENTRIES.values()) == 9 and all(len(v) == 3 for v in ENTRIES.values())
    assert ENTRIES["Saraswati"][2] == "Sacred Strategy" and len({e for v in ENTRIES.values() for e in v}) == 9


def test_relations_stored_as_edges_survive_all_eight_transforms():
    g = table_graph(); base = classes(g)
    assert base == {frozenset({1, 2, 3}), frozenset({4, 5, 6}), frozenset({7, 8, 9})}
    for t in IDS:
        h = relabel_graph(g, d4_relabel(t))
        assert classes(h) == base, t                                         # the same three groups of entries, wherever the cells went
        assert len(h.nodes) == 9 and len(h.edges) == 9                       # 3 figures x 3 pairs


def test_a_meaning_stored_as_position_survives_only_four_of_the_eight():
    g = table_graph(); rows, cols = [], []
    for t in IDS:
        h = relabel_graph(g, d4_relabel(t))
        r, c = figures_in_rows(h), figures_in_cols(h)
        assert r != c, t                                                     # never both, never neither
        (rows if r else cols).append(t)
    assert len(rows) == 4 and len(cols) == 4
    perm = lambda t: tuple(int(x) for x in d4apply(IDX, t).ravel())
    comp = lambda p, q: tuple(p[q[i]] for i in range(9))
    P = {perm(t) for t in rows}
    assert all(comp(p, q) in P for p in P for q in P)                        # closed under composition: a subgroup of order 4
    assert all(comp(p, p) == tuple(range(9)) for p in P)                     # every element is its own inverse: the Klein four-group
    assert 0 in rows and 2 in rows                                           # identity and the half turn (t = 2) are among them


def test_how_many_arrangements_read_as_rows_or_columns():
    fig = {i: i // 3 for i in range(9)}                                      # entry index 0..8 -> figure 0..2, three entries per figure
    rows = cols = 0
    for p in permutations(range(9)):                                         # p[cell] = entry placed on that cell (cell = 3 r + c)
        if all(len({fig[p[3 * r + c]] for c in range(3)}) == 1 for r in range(3)): rows += 1
        elif all(len({fig[p[3 * r + c]] for r in range(3)}) == 1 for c in range(3)): cols += 1
    assert rows == 6 * 6 ** 3 == 1296 and cols == 1296 and rows + cols == 2592
    assert abs((rows + cols) / 362880 - 0.0071429) < 1e-6
    assert rows // 4 == 324 and (rows + cols) // 8 == 324


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
