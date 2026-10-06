"""Strict CICO delta parser.

Grammar (items separated by exactly one space or newline; one optional trailing separator):
    ADD[r,c:val]                      node add     (val may be negative)
    DEL[r,c]                          node delete
    ADD[(r,c)->(r,c):rel#w]           edge add     (rel = [A-Za-z0-9_]+, w may be negative)
    DEL[(r,c)->(r,c):rel#w]           edge delete
Anything else raises CICOParseError. Nothing is silently skipped.
"""
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

_N = r"[0-9]+"
NODE_ADD = re.compile(rf"ADD\[({_N}),({_N}):(-?{_N})\]")
NODE_DEL = re.compile(rf"DEL\[({_N}),({_N})\]")
EDGE = re.compile(rf"(ADD|DEL)\[\(({_N}),({_N})\)->\(({_N}),({_N})\):([A-Za-z0-9_]+)#(-?{_N})\]")


class CICOParseError(ValueError):
    pass


@dataclass(frozen=True)
class NodeDelta:
    op: str
    r: int
    c: int
    val: Optional[int] = None


@dataclass(frozen=True)
class EdgeDelta:
    op: str
    u: Tuple[int, int]
    v: Tuple[int, int]
    relation: str
    weight: int


@dataclass
class StateTransitionDelta:
    node_deltas: List[NodeDelta]
    edge_deltas: List[EdgeDelta]


class CICOParser:
    @staticmethod
    def serialize_state(nodes, edges) -> str:
        ns = " ".join(f"[{r},{c}:{v}]" for r, c, v in nodes)
        es = " ".join(f"({a},{b})->({c},{d}):{rel}#{w}" for a, b, c, d, rel, w in edges)
        return f"<CICO_STATE>\nNODES: {ns}\nEDGES: {es}\n</CICO_STATE>"

    @staticmethod
    def parse_transition_delta(raw: str) -> StateTransitionDelta:
        nodes: List[NodeDelta] = []
        edges: List[EdgeDelta] = []
        if raw == "":
            return StateTransitionDelta(nodes, edges)
        items = re.split(r"[ \n]", raw)
        if len(items) > 1 and items[-1] == "":
            items.pop()  # one trailing separator allowed
        for it in items:
            m = NODE_ADD.fullmatch(it)
            if m:
                nodes.append(NodeDelta("ADD", int(m[1]), int(m[2]), int(m[3])))
                continue
            m = NODE_DEL.fullmatch(it)
            if m:
                nodes.append(NodeDelta("DEL", int(m[1]), int(m[2])))
                continue
            m = EDGE.fullmatch(it)
            if m:
                edges.append(EdgeDelta(m[1], (int(m[2]), int(m[3])), (int(m[4]), int(m[5])), m[6], int(m[7])))
                continue
            raise CICOParseError(f"invalid CICO item: {it!r}")
        return StateTransitionDelta(nodes, edges)
