"""Scope containment: keep only the changes the request is about (docs/SCOPE_TEST.md; concept-map rows 3, 9, 27).

What the gate cannot see. The gate (`gate.check`) answers "is this delta legal on this graph?". A delta can be legal and still not be what was asked: the 7B model, told the
rule, deletes real edges nobody asked it to delete (docs/SCALE_TEST_7B.md, docs/LONG_HORIZON_TEST.md). The gate admits them, ground-and-complete keeps them, the held state drifts.

Named scope. The request names the things it is about. Here "name" is the crudest possible reading: a coordinate (r,c) written in the request text. A change is IN SCOPE if
  * a service operation (ADD / DEL) is on a named service;
  * an edge ADD has both endpoints named;
  * an edge DEL has both endpoints named, OR has one endpoint that is a named service which this same delta deletes (the edges that must go with it).
Everything else is dropped and reported. If the request names nothing (for example "remove every edge with weight 4"), there is no scope to apply and nothing is dropped: the
policy is then exactly ground-and-complete (M6). After the filter the usual path runs: an admissible delta is applied, an inadmissible one is grounded and completed, and if
nothing is left, nothing is applied.

`contain_delta_internal` is the earlier, post-hoc idea (found after reading the 7B outputs): drop every edge DEL that does not touch a service deleted in the same delta. It needs
no request, and it is wrong for any request that is a lone edge delete. It is kept as a baseline so the test can show that.

What this does NOT do: it cannot tell a wrong edge between two named services from a right one, it trusts that the request names its targets, and a request that names a service
it does not intend to touch widens the scope. It reads coordinates, not meaning."""
import re
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

from .cico_parser import StateTransitionDelta
from .diagnostics import ground_and_complete, format_delta
from .gate import Graph, check

COORD = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")


def named_services(instruction: Optional[str]) -> Set[Tuple[int, int]]:
    """The coordinates written in the request, as a set of (r, c)."""
    return {(int(a), int(b)) for a, b in COORD.findall(instruction or "")}


def _txt_n(d): return f"ADD[{d.r},{d.c}:{d.val}]" if d.op == "ADD" else f"DEL[{d.r},{d.c}]"
def _txt_e(e): return f"{e.op}[({e.u[0]},{e.u[1]})->({e.v[0]},{e.v[1]}):{e.relation}#{e.weight}]"


def contain_scope(delta: StateTransitionDelta, named: Set[Tuple[int, int]]):
    """(scoped delta, dropped) where dropped = [(reason, item text)]. With no named service nothing is dropped."""
    if not named: return delta, []
    dropped, nodes = [], []
    for d in delta.node_deltas:
        if (d.r, d.c) in named: nodes.append(d)
        else: dropped.append(("service not named in the request", _txt_n(d)))
    deleted = {(d.r, d.c) for d in nodes if d.op == "DEL"}
    edges = []
    for e in delta.edge_deltas:
        if e.op == "ADD":
            if e.u in named and e.v in named: edges.append(e)
            else: dropped.append(("edge added between services not both named in the request", _txt_e(e)))
        elif (e.u in named and e.v in named) or e.u in deleted or e.v in deleted: edges.append(e)
        else: dropped.append(("edge removed that the request does not name and that no deleted named service owns", _txt_e(e)))
    return StateTransitionDelta(nodes, edges), dropped


def contain_delta_internal(delta: StateTransitionDelta):
    """The post-hoc baseline: an edge DEL is kept only if it touches a service deleted in the same delta. Everything else is kept."""
    gone = {(d.r, d.c) for d in delta.node_deltas if d.op == "DEL"}
    keep, dropped = [], []
    for e in delta.edge_deltas:
        if e.op == "DEL" and e.u not in gone and e.v not in gone: dropped.append(("edge removed that touches no service deleted in this delta", _txt_e(e)))
        else: keep.append(e)
    return StateTransitionDelta(list(delta.node_deltas), keep), dropped


@dataclass
class ScopeResult:
    mode: str
    named: Optional[Set[Tuple[int, int]]]
    dropped: list = field(default_factory=list)              # [(reason, item text)] removed by the scope filter
    admitted_after_scope: bool = False                       # the filtered delta passed the gate as it stood
    repaired: bool = False                                   # ground-and-complete had to run and succeeded
    final_delta: Optional[StateTransitionDelta] = None       # what to apply; None = apply nothing
    reason: str = ""

    @property
    def text(self) -> str: return format_delta(self.final_delta) if self.final_delta is not None else ""


def admit_scoped(g: Graph, delta: StateTransitionDelta, instruction: Optional[str] = None, mode: str = "named", degree_threshold: int = 4) -> ScopeResult:
    """Scope filter, then the M6 path. mode 'named' needs the request text; mode 'delta' is the post-hoc baseline."""
    assert mode in ("named", "delta")
    if mode == "named":
        named = named_services(instruction); scoped, dropped = contain_scope(delta, named)
    else:
        named = None; scoped, dropped = contain_delta_internal(delta)
    res = ScopeResult(mode, named, dropped)
    if not scoped.node_deltas and not scoped.edge_deltas:
        res.reason = "nothing within scope" if dropped else "empty proposal"; return res
    if check(g, scoped).ok:
        res.admitted_after_scope, res.final_delta = True, scoped; return res
    gr = ground_and_complete(g, scoped, degree_threshold)
    if gr is not None and gr.repairable and check(g, gr.repaired).ok:
        res.repaired, res.final_delta = True, gr.repaired; return res
    res.reason = gr.reason if gr is not None else "not admissible"
    return res
