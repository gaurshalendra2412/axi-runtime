"""Inverse of a delta (concept-map rows 38 and 58).

Given a host graph g and a delta d that the gate admits on g, `invert(g, d)` returns the delta d' that undoes it:
    ADD node  ->  DEL node                      (the value is not needed to delete)
    DEL node  ->  ADD node with the value it had in g
    ADD edge  ->  DEL edge with the weight that was added
    DEL edge  ->  ADD edge with the weight that was matched
Read as signed counts per item (+1 for ADD, -1 for DEL, 0 for untouched), d' is the negative of d, so d followed by d' adds up to
zero. `invert` must be called with the graph BEFORE d is applied, because a deleted node's value exists only there.
It does not check that d is admissible; call `gate.check` first. SET (an in-place update) is not part of the language yet (row 17);
when it is added, its inverse has to be added here as well.
"""
from axi.engine.cico_parser import EdgeDelta, NodeDelta, StateTransitionDelta
from axi.engine.gate import Graph


def invert(g: Graph, delta: StateTransitionDelta) -> StateTransitionDelta:
    nodes = []
    for d in delta.node_deltas:
        if d.op == "ADD":
            nodes.append(NodeDelta("DEL", d.r, d.c))
        else:
            nodes.append(NodeDelta("ADD", d.r, d.c, g.nodes[(d.r, d.c)]))
    edges = [EdgeDelta("DEL" if e.op == "ADD" else "ADD", e.u, e.v, e.relation, e.weight) for e in delta.edge_deltas]
    return StateTransitionDelta(nodes, edges)


def to_text(delta: StateTransitionDelta) -> str:
    """Serialize back to the CICO wire format (items separated by one space), so an inverse can be logged or replayed."""
    items = []
    for d in delta.node_deltas:
        items.append(f"ADD[{d.r},{d.c}:{d.val}]" if d.op == "ADD" else f"DEL[{d.r},{d.c}]")
    for e in delta.edge_deltas:
        items.append(f"{e.op}[({e.u[0]},{e.u[1]})->({e.v[0]},{e.v[1]}):{e.relation}#{e.weight}]")
    return " ".join(items)
