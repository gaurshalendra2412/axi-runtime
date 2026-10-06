"""DPO-style gluing gate with match validation (adjacency-indexed).

Host graph: nodes {(r,c): val}, edges {(u,v,rel): weight}.
A delta is accepted iff ALL hold:
  match (morphism):   every DEL node exists; every DEL edge exists with the same relation and weight
  identification:     no ADD of an existing node/edge, no item repeated within the delta
  dangling edge:      every edge incident to a deleted node is deleted in the same delta
  well-formedness:    every added edge has both endpoints present after the rewrite
"""
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple

Node = Tuple[int, int]
EdgeKey = Tuple[Node, Node, str]


class Graph:
    def __init__(self):
        self.nodes: Dict[Node, int] = {}
        self.edges: Dict[EdgeKey, int] = {}
        self._inc: Dict[Node, Set[EdgeKey]] = defaultdict(set)

    def add_node(self, n, val=0): self.nodes[n] = val

    def add_edge(self, u, v, rel, w):
        assert u in self.nodes and v in self.nodes
        k = (u, v, rel); self.edges[k] = w; self._inc[u].add(k); self._inc[v].add(k)

    def del_edge(self, k):
        del self.edges[k]; self._inc[k[0]].discard(k); self._inc[k[1]].discard(k)

    def del_node(self, n):
        assert not self._inc.get(n), "dangling"
        del self.nodes[n]; self._inc.pop(n, None)

    def incident(self, n): return self._inc.get(n, set())

    def is_well_formed(self):
        return all(u in self.nodes and v in self.nodes for (u, v, _) in self.edges)


@dataclass
class GateResult:
    ok: bool
    reason: str = ""


def check(g: Graph, delta) -> GateResult:
    seen = set()
    del_nodes, add_nodes, del_edges, add_edges = set(), set(), set(), set()
    for d in delta.node_deltas:
        n = (d.r, d.c)
        if ("n", d.op, n) in seen: return GateResult(False, f"identification: repeated node op {d.op}{n}")
        seen.add(("n", d.op, n))
        if d.op == "DEL":
            if n not in g.nodes: return GateResult(False, f"match: node {n} not in host")
            del_nodes.add(n)
        else:
            if n in g.nodes: return GateResult(False, f"identification: node {n} already exists")
            add_nodes.add(n)
    if del_nodes & add_nodes: return GateResult(False, "identification: node both added and deleted")
    for d in delta.edge_deltas:
        k = (d.u, d.v, d.relation)
        if ("e", d.op, k) in seen: return GateResult(False, f"identification: repeated edge op {d.op}{k}")
        seen.add(("e", d.op, k))
        if d.op == "DEL":
            if k not in g.edges: return GateResult(False, f"match: edge {k} not in host")
            if g.edges[k] != d.weight: return GateResult(False, f"match: edge {k} weight {g.edges[k]} != {d.weight}")
            del_edges.add(k)
        else:
            if k in g.edges: return GateResult(False, f"identification: edge {k} already exists")
            add_edges.add(k)
    if del_edges & add_edges: return GateResult(False, "identification: edge both added and deleted")
    for n in del_nodes:
        dangling = g.incident(n) - del_edges
        if dangling: return GateResult(False, f"dangling edge: {n} still has {len(dangling)} incident edge(s)")
    alive = (set(g.nodes) - del_nodes) | add_nodes
    for (u, v, rel) in add_edges:
        if u not in alive or v not in alive: return GateResult(False, f"well-formedness: edge {(u, v, rel)} endpoint missing")
    return GateResult(True)


def apply(g: Graph, delta) -> None:
    """Apply an already-checked delta. Order: delete edges, delete nodes, add nodes, add edges."""
    for d in delta.edge_deltas:
        if d.op == "DEL": g.del_edge((d.u, d.v, d.relation))
    for d in delta.node_deltas:
        if d.op == "DEL": g.del_node((d.r, d.c))
    for d in delta.node_deltas:
        if d.op == "ADD": g.add_node((d.r, d.c), d.val)
    for d in delta.edge_deltas:
        if d.op == "ADD": g.add_edge(d.u, d.v, d.relation, d.weight)
