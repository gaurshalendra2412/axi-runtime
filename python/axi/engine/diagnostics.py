"""Layer 1: structured obstruction diagnostics + deterministic repair deltas (no prose).

diagnose() collects ALL gluing/match obstructions (the gate stops at the first).
propose_repair() turns dangling-edge obstructions into a repaired proposal in the CICO language:
  primal : add DEL for every undeleted incident edge          (small degree)
  dual   : drop the DEL of the node (keep it)                 (degree > threshold)
Only dangling obstructions are auto-repairable; anything else is reported, not guessed.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .cico_parser import NodeDelta, EdgeDelta, StateTransitionDelta
from .gate import Graph


@dataclass(frozen=True)
class Obstruction:
    kind: str                      # dangling | match_node | match_edge | match_weight | ident_node | ident_edge | ident_conflict | malformed
    node: Optional[Tuple[int, int]] = None
    edge: Optional[tuple] = None   # (u, v, rel)
    edges: tuple = ()              # dangling: ((u, v, rel, weight), ...) still attached to `node`


def diagnose(g: Graph, delta) -> List[Obstruction]:
    obs: List[Obstruction] = []
    seen = set(); del_nodes, add_nodes, del_edges, add_edges = set(), set(), set(), set()
    for d in delta.node_deltas:
        n = (d.r, d.c)
        if ("n", d.op, n) in seen:
            obs.append(Obstruction("ident_node", node=n)); continue
        seen.add(("n", d.op, n))
        if d.op == "DEL":
            if n not in g.nodes: obs.append(Obstruction("match_node", node=n))
            else: del_nodes.add(n)
        else:
            if n in g.nodes: obs.append(Obstruction("ident_node", node=n))
            else: add_nodes.add(n)
    for n in del_nodes & add_nodes: obs.append(Obstruction("ident_conflict", node=n))
    for d in delta.edge_deltas:
        k = (d.u, d.v, d.relation)
        if ("e", d.op, k) in seen:
            obs.append(Obstruction("ident_edge", edge=k)); continue
        seen.add(("e", d.op, k))
        if d.op == "DEL":
            if k not in g.edges: obs.append(Obstruction("match_edge", edge=k))
            elif g.edges[k] != d.weight: obs.append(Obstruction("match_weight", edge=k))
            else: del_edges.add(k)
        else:
            if k in g.edges: obs.append(Obstruction("ident_edge", edge=k))
            else: add_edges.add(k)
    for k in del_edges & add_edges: obs.append(Obstruction("ident_conflict", edge=k))
    for n in sorted(del_nodes):
        left = g.incident(n) - del_edges
        if left:
            obs.append(Obstruction("dangling", node=n, edges=tuple(sorted((u, v, r, g.edges[(u, v, r)]) for (u, v, r) in left))))
    alive = (set(g.nodes) - del_nodes) | add_nodes
    for (u, v, r) in sorted(add_edges):
        if u not in alive or v not in alive: obs.append(Obstruction("malformed", edge=(u, v, r)))
    return obs


def format_delta(delta: StateTransitionDelta) -> str:
    items = []
    for d in delta.node_deltas:
        items.append(f"ADD[{d.r},{d.c}:{d.val}]" if d.op == "ADD" else f"DEL[{d.r},{d.c}]")
    for e in delta.edge_deltas:
        items.append(f"{e.op}[({e.u[0]},{e.u[1]})->({e.v[0]},{e.v[1]}):{e.relation}#{e.weight}]")
    return " ".join(items)


@dataclass
class Repair:
    obstructions: List[Obstruction]
    repairable: bool
    strategies: dict = field(default_factory=dict)       # node -> 'primal' | 'dual'
    added_ops: str = ""                                  # CICO text of the ops the repair ADDED (the hint)
    repaired: Optional[StateTransitionDelta] = None
    text: Optional[str] = None                           # full repaired proposal, ready to resubmit


def propose_repair(g: Graph, delta: StateTransitionDelta, degree_threshold: int = 4) -> Optional[Repair]:
    obs = diagnose(g, delta)
    if not obs: return None
    if any(o.kind != "dangling" for o in obs): return Repair(obs, False)
    dual_nodes, new_edges, strat = set(), {}, {}
    for o in obs:
        if len(o.edges) <= degree_threshold:
            strat[o.node] = "primal"
            for (u, v, r, w) in o.edges: new_edges[(u, v, r)] = w      # dict: edge shared by two deleted nodes is added once
        else:
            strat[o.node] = "dual"; dual_nodes.add(o.node)
    nodes = [d for d in delta.node_deltas if not (d.op == "DEL" and (d.r, d.c) in dual_nodes)]
    # a dual-kept node's edges need no deletion; but an edge shared with a primal node must still go (primal wins for that edge)
    added = [EdgeDelta("DEL", u, v, r, w) for (u, v, r), w in sorted(new_edges.items())]
    repaired = StateTransitionDelta(nodes, list(delta.edge_deltas) + added)
    return Repair(obs, True, strat, format_delta(StateTransitionDelta([], added)), repaired, format_delta(repaired))
