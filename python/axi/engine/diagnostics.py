"""Layer 1: structured obstruction diagnostics + deterministic repair deltas (no prose).

diagnose() collects ALL gluing/match obstructions (the gate stops at the first).
propose_repair() turns dangling-edge obstructions into a repaired proposal in the CICO language:
  primal : add DEL for every undeleted incident edge          (small degree)
  dual   : drop the DEL of the node (keep it)                 (degree > threshold)
Only dangling obstructions are auto-repairable by propose_repair(); anything else is reported, not guessed.

ground_and_complete() (added after reading the Kali / Durga posts, see docs/RAJNISH_CONCEPT_MAP.md rows 25-26) handles more:
  cut      drop the operations that name things which are not in the real graph or contradict the delta (the "garbage")
  ground   correct an edge weight to the real one (the edge is named correctly, only the number is wrong)
  complete then close the boundary: delete every remaining incident edge of a deleted node (primal) or keep the node (dual)
It never invents a node operation and never adds an edge that the model did not propose, except the DELs of REAL incident edges.
Updates (ADD of an existing node with another value) cannot be expressed in this language, so they stay unrepaired.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .cico_parser import NodeDelta, EdgeDelta, StateTransitionDelta
from .gate import Graph, check as _check


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


def normalize(g: Graph, delta: StateTransitionDelta):
    """Drop only PROVABLY redundant items: exact duplicates, and ADDs of a node/edge that already exists with the
    identical value/weight (a no-op in sequential semantics). Anything ambiguous (e.g. different value) is kept."""
    seen, nodes, edges, dropped = set(), [], [], []
    for d in delta.node_deltas:
        key = ("n", d.op, d.r, d.c, d.val)
        redundant = key in seen or (d.op == "ADD" and (d.r, d.c) in g.nodes and g.nodes[(d.r, d.c)] == d.val)
        seen.add(key); (dropped if redundant else nodes).append(d)
    for e in delta.edge_deltas:
        key = ("e", e.op, e.u, e.v, e.relation, e.weight); k = (e.u, e.v, e.relation)
        redundant = key in seen or (e.op == "ADD" and k in g.edges and g.edges[k] == e.weight)
        seen.add(key); (dropped if redundant else edges).append(e)
    return StateTransitionDelta(nodes, edges), dropped


@dataclass
class Repair:
    obstructions: List[Obstruction]
    repairable: bool
    strategies: dict = field(default_factory=dict)       # node -> 'primal' | 'dual'
    added_ops: str = ""                                  # CICO text of the ops the repair ADDED (the hint)
    repaired: Optional[StateTransitionDelta] = None
    text: Optional[str] = None                           # full repaired proposal, ready to resubmit
    dropped: list = field(default_factory=list)          # redundant items removed by normalize()


def propose_repair(g: Graph, delta: StateTransitionDelta, degree_threshold: int = 4) -> Optional[Repair]:
    obs0 = diagnose(g, delta)
    if not obs0: return None
    base, dropped = normalize(g, delta)
    obs = diagnose(g, base) if dropped else obs0
    if not obs: return Repair(obs0, True, {}, "", base, format_delta(base), dropped)
    if any(o.kind != "dangling" for o in obs): return Repair(obs0, False)
    delta = base
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
    return Repair(obs0, True, strat, format_delta(StateTransitionDelta([], added)), repaired, format_delta(repaired), dropped)


@dataclass
class Grounding:
    obstructions: List[Obstruction]
    repairable: bool
    reason: str = ""                                    # why it is not repairable
    cut: list = field(default_factory=list)             # [(reason, item text)] operations removed
    grounded: list = field(default_factory=list)        # [(reason, before, after)] values corrected to the real graph
    completed: str = ""                                 # CICO text of the DELs added to close dangling edges
    strategies: dict = field(default_factory=dict)      # node -> 'primal' | 'dual'
    repaired: Optional[StateTransitionDelta] = None
    text: Optional[str] = None


def _txt_n(d): return f"ADD[{d.r},{d.c}:{d.val}]" if d.op == "ADD" else f"DEL[{d.r},{d.c}]"
def _txt_e(e): return f"{e.op}[({e.u[0]},{e.u[1]})->({e.v[0]},{e.v[1]}):{e.relation}#{e.weight}]"


def ground_and_complete(g: Graph, delta: StateTransitionDelta, degree_threshold: int = 4) -> Optional[Grounding]:
    """Deterministic repair that keeps what is true, cuts what is false, and closes the boundary.
    Returns None when the delta is already admissible (a clean proposal is passed through untouched)."""
    obs0 = diagnose(g, delta)
    if not obs0: return None
    cut, grounded = [], []
    node_ops = list(delta.node_deltas)
    # 1. node operations
    nodes, seen = [], set()
    for d in node_ops:
        n, key = (d.r, d.c), (d.op, d.r, d.c)
        if key in seen: cut.append(("duplicate", _txt_n(d))); continue
        seen.add(key)
        if d.op == "DEL" and n not in g.nodes: cut.append(("no such service in the graph", _txt_n(d))); continue
        if d.op == "ADD" and n in g.nodes:
            if g.nodes[n] == d.val: cut.append(("service already present with this value", _txt_n(d))); continue
            if any(x.op == "DEL" and (x.r, x.c) == n for x in node_ops):
                cut.append(("add-then-delete of an existing service: the net effect is the delete", _txt_n(d))); continue
        nodes.append(d)
    del_nodes = {(d.r, d.c) for d in nodes if d.op == "DEL"}
    add_nodes = {(d.r, d.c) for d in nodes if d.op == "ADD"}
    alive = (set(g.nodes) - del_nodes) | add_nodes
    # 2. edge operations
    edges, seen = [], set()
    for e in delta.edge_deltas:
        k = (e.u, e.v, e.relation); key = (e.op, k)
        if key in seen: cut.append(("duplicate", _txt_e(e))); continue
        seen.add(key)
        if e.op == "DEL":
            if k not in g.edges: cut.append(("no such edge in the graph", _txt_e(e))); continue
            if g.edges[k] != e.weight:
                e2 = EdgeDelta("DEL", e.u, e.v, e.relation, g.edges[k]); grounded.append(("edge weight corrected to the real one", _txt_e(e), _txt_e(e2))); e = e2
        else:
            if e.u not in alive or e.v not in alive:
                cut.append(("an edge cannot be added to a service that will not exist", _txt_e(e))); continue
            if k in g.edges and g.edges[k] == e.weight: cut.append(("edge already present with this weight", _txt_e(e))); continue
        edges.append(e)
    # edge ADD/DEL of the same key would still conflict: leave it to the check below (unrepaired, reported)
    cleaned = StateTransitionDelta(nodes, edges)
    if not nodes and not edges:
        return Grounding(obs0, False, "nothing true is left after removing the claims that are not in the graph", cut, grounded)
    obs = diagnose(g, cleaned)
    if not obs:
        repaired, completed, strat = cleaned, "", {}
    else:
        if any(o.kind != "dangling" for o in obs):
            return Grounding(obs0, False, "obstructions remain that cannot be repaired without guessing: " + ", ".join(sorted({o.kind for o in obs})), cut, grounded)
        rep = propose_repair(g, cleaned, degree_threshold)
        if rep is None or not rep.repairable:
            return Grounding(obs0, False, "closing the boundary failed", cut, grounded)
        repaired, completed, strat = rep.repaired, rep.added_ops, rep.strategies
    if not _check(g, repaired).ok:
        return Grounding(obs0, False, "the repaired delta still fails the gate", cut, grounded)
    return Grounding(obs0, True, "", cut, grounded, completed, strat, repaired, format_delta(repaired))
