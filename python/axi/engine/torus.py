"""A wrap-around (torus) address frame in front of the gate (concept-map item 121, docs/TORUS_FRAME.md, test 147).

Source: two of Rajnish Choubey's articles, "The India of Zero & Infinity" and "The 99 Eyewitnesses of the Ramayana", as an AI summary the
owner pasted on 24 Oct describes them (the articles themselves were not read here): an omnidirectional origin where every direction is
presence, a fixed closed set of points whose movement wraps instead of stopping at an edge, and any point naming any other directly.

What it adds. The gate has no geometry of its own: it sees which cell, never where the cell is (tests/test_gate_frame_covariance.py).
Shape lives in front of it, as the D4 symmetry does in d4.py. This module is such a layer for jobs whose space is circular (days of the
week, shift rotas, ring buffers, rotations, maps that wrap at the edges):

  * `address` turns a coordinate the model wrote into the host's name for it. In `strict` mode (the default) a coordinate outside the
    frame is refused (AddressError); in `wrap` mode it is wrapped and the wrap is reported, never silent.
  * `offset` and `distance` measure on the ring: every offset is a forward step in [0, n), so no direction is negative.
  * `neighbours` and `adjacency` give the wrap-around neighbours as `adj` edges, which the gate cannot supply itself.
  * `move_delta` writes a move as an ordinary CICO delta (delete the node and its edges, add it at the wrapped target with the same
    edges). The gate decides it as it decides anything: an empty source is refused (match), an occupied target is refused
    (identification), and the move commits whole or not at all. `axi.engine.inverse.invert` undoes it.
  * `shift_graph` and `shift_delta` move a whole state or delta round the torus (the group Z_rows x Z_cols acting on the cells).

It edits no framework file: the parser, the gate, `apply` and `invert` are used as they are.
What it does NOT show: that a model proposes better moves on a wrap-around frame, or anything about ARC; no model is called here."""
from __future__ import annotations
from typing import Iterable, List, Optional, Tuple

from axi.engine.cico_parser import EdgeDelta, NodeDelta, StateTransitionDelta
from axi.engine.gate import Graph

Node = Tuple[int, int]
Edge = Tuple[Node, Node, str, int]


class AddressError(ValueError):
    """A coordinate outside the frame, in strict mode."""


class TorusFrame:
    def __init__(self, rows: int, cols: int, mode: str = "strict"):
        if rows < 1 or cols < 1: raise ValueError("a frame has at least one row and one column")
        if mode not in ("strict", "wrap"): raise ValueError("mode is 'strict' or 'wrap'")
        self.rows, self.cols, self.mode = rows, cols, mode

    # ------------------------------------------------------------ addresses
    @property
    def size(self) -> int: return self.rows * self.cols

    def cells(self) -> List[Node]: return [(r, c) for r in range(self.rows) for c in range(self.cols)]

    def contains(self, n: Node) -> bool: return 0 <= n[0] < self.rows and 0 <= n[1] < self.cols

    def wrap(self, n: Node) -> Node: return (n[0] % self.rows, n[1] % self.cols)

    def address(self, n: Node) -> Node:
        """The host's name for a coordinate the model wrote. Strict: outside the frame raises AddressError. Wrap: wrapped."""
        n = (int(n[0]), int(n[1]))
        if self.contains(n): return n
        if self.mode == "wrap": return self.wrap(n)
        raise AddressError(f"address {n} is outside the {self.rows}x{self.cols} frame")

    # ------------------------------------------------------------ measure on the ring
    def offset(self, a: Node, b: Node) -> Tuple[int, int]:
        """The step from a to b, each part in [0, n): every direction is a forward step that wraps."""
        return ((b[0] - a[0]) % self.rows, (b[1] - a[1]) % self.cols)

    def distance(self, a: Node, b: Node) -> int:
        """Steps on the torus, the short way round in each direction."""
        dr, dc = self.offset(a, b)
        return min(dr, self.rows - dr) + min(dc, self.cols - dc)

    def shift(self, n: Node, dr: int, dc: int) -> Node: return self.wrap((n[0] + dr, n[1] + dc))

    def neighbours(self, n: Node) -> List[Node]:
        """The four wrap-around neighbours, without repeats and without the cell itself (a frame one row high is a ring)."""
        n = self.wrap(n); out: List[Node] = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            m = self.shift(n, dr, dc)
            if m != n and m not in out: out.append(m)
        return out

    def adjacency(self, rel: str = "adj", weight: int = 1) -> List[Edge]:
        """Every neighbouring pair once, as an edge the task can add to the host (the gate supplies no adjacency itself)."""
        return [(a, b, rel, weight) for a in self.cells() for b in self.neighbours(a) if a < b]

    # ------------------------------------------------------------ deltas
    def place_delta(self, delta: StateTransitionDelta) -> Tuple[StateTransitionDelta, List[Tuple[Node, Node]]]:
        """Every coordinate of a parsed delta through `address`. Returns the placed delta and the list of (written, placed) pairs that were
        wrapped, so the host can count them. Strict mode raises AddressError on the first coordinate outside the frame."""
        wrapped: List[Tuple[Node, Node]] = []
        def put(n: Node) -> Node:
            p = self.address(n)
            if p != tuple(n): wrapped.append((tuple(n), p))
            return p
        nodes = []
        for d in delta.node_deltas:
            r, c = put((d.r, d.c)); nodes.append(NodeDelta(d.op, r, c, d.val))
        edges = [EdgeDelta(e.op, put(e.u), put(e.v), e.relation, e.weight) for e in delta.edge_deltas]
        return StateTransitionDelta(nodes, edges), wrapped

    def move_delta(self, g: Graph, src: Node, dr: int, dc: int) -> Optional[StateTransitionDelta]:
        """The delta that moves the node at `src` by (dr, dc) round the torus, its edges going with it. The gate decides it: an empty source
        is refused (match), an occupied target is refused (identification). A move by whole laps is no change and returns None."""
        src = self.address(src); dst = self.shift(src, dr, dc)
        if dst == src: return None
        inc = sorted(g.incident(src)) if src in g.nodes else []
        m = lambda x: dst if x == src else x
        nodes = [NodeDelta("DEL", src[0], src[1]), NodeDelta("ADD", dst[0], dst[1], g.nodes.get(src, 0))]
        edges = [EdgeDelta("DEL", u, v, rel, g.edges[(u, v, rel)]) for (u, v, rel) in inc]
        edges += [EdgeDelta("ADD", m(u), m(v), rel, g.edges[(u, v, rel)]) for (u, v, rel) in inc]
        return StateTransitionDelta(nodes, edges)

    def shift_delta(self, delta: StateTransitionDelta, dr: int, dc: int) -> StateTransitionDelta:
        s = lambda n: self.shift(n, dr, dc)
        return StateTransitionDelta([NodeDelta(d.op, *s((d.r, d.c)), d.val) for d in delta.node_deltas],
                                    [EdgeDelta(e.op, s(e.u), s(e.v), e.relation, e.weight) for e in delta.edge_deltas])

    def shift_graph(self, g: Graph, dr: int, dc: int) -> Graph:
        """The whole state moved by (dr, dc) round the torus. Every node must be inside the frame."""
        out = Graph()
        for n, v in g.nodes.items():
            if not self.contains(n): raise AddressError(f"node {n} is outside the {self.rows}x{self.cols} frame")
            out.add_node(self.shift(n, dr, dc), v)
        for (u, v, rel), w in g.edges.items(): out.add_edge(self.shift(u, dr, dc), self.shift(v, dr, dc), rel, w)
        return out


def complete_graph_edges(n: int) -> int:
    """Edges of n points each joined to every other: n (n - 1) / 2 (108 points: 5,778)."""
    return n * (n - 1) // 2


def complete_graph_genus(n: int) -> int:
    """The fewest handles a surface needs to carry the complete graph on n points without crossings (Ringel and Youngs, 1968):
    ceil((n - 3)(n - 4) / 12) for n >= 3. A torus has one handle, so seven points is the most it carries fully joined."""
    if n < 3: return 0
    return -(-((n - 3) * (n - 4)) // 12)
