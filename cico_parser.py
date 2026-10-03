"""
CICO (Coordinate-In, Coordinate-Out) Structural Language Parser.

Grammar (one delta per whitespace-separated token):
  ADD[r,c:val]                     add node at (r,c) with integer value
  DEL[r,c]                         delete node at (r,c)
  ADD[(r1,c1)->(r2,c2):rel#w]      add edge
  DEL[(r1,c1)->(r2,c2):rel#w]      delete edge

The parser is STRICT: any token that does not match the grammar raises
CICOParseError. Silently skipping malformed text would defeat the purpose
of a verification gate.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


class CICOParseError(ValueError):
    pass


@dataclass
class NodeDelta:
    op: str  # 'ADD' or 'DEL'
    r: int
    c: int
    val: Optional[int] = None


@dataclass
class EdgeDelta:
    op: str  # 'ADD' or 'DEL'
    u: Tuple[int, int]
    v: Tuple[int, int]
    relation: str
    weight: int


@dataclass
class StateTransitionDelta:
    node_deltas: List[NodeDelta]
    edge_deltas: List[EdgeDelta]


class CICOParser:
    NODE_ADD = re.compile(r"ADD\[(\d+),(\d+):(-?\d+)\]")
    NODE_DEL = re.compile(r"DEL\[(\d+),(\d+)\]")
    EDGE_MUT = re.compile(
        r"(ADD|DEL)\[\((\d+),(\d+)\)->\((\d+),(\d+)\):([A-Za-z0-9_]+)#(-?\d+)\]"
    )

    @classmethod
    def serialize_state(
        cls,
        nodes: List[Tuple[int, int, int]],
        edges: List[Tuple[int, int, int, int, str, int]],
    ) -> str:
        """Serialize host graph G into CICO state tokens."""
        node_strs = [f"[{r},{c}:{val}]" for r, c, val in nodes]
        edge_strs = [
            f"({u_r},{u_c})->({v_r},{v_c}):{rel}#{w}"
            for u_r, u_c, v_r, v_c, rel, w in edges
        ]
        return (
            "<CICO_STATE>\n"
            f"NODES: {' '.join(node_strs)}\n"
            f"EDGES: {' '.join(edge_strs)}\n"
            "</CICO_STATE>"
        )

    @classmethod
    def parse_transition_delta(cls, raw: str) -> StateTransitionDelta:
        """Parse a proposed delta. Raises CICOParseError on any invalid token."""
        node_deltas: List[NodeDelta] = []
        edge_deltas: List[EdgeDelta] = []

        for token in raw.split():
            m = cls.NODE_ADD.fullmatch(token)
            if m:
                node_deltas.append(NodeDelta("ADD", int(m[1]), int(m[2]), int(m[3])))
                continue
            m = cls.NODE_DEL.fullmatch(token)
            if m:
                node_deltas.append(NodeDelta("DEL", int(m[1]), int(m[2])))
                continue
            m = cls.EDGE_MUT.fullmatch(token)
            if m:
                edge_deltas.append(
                    EdgeDelta(
                        m[1],
                        (int(m[2]), int(m[3])),
                        (int(m[4]), int(m[5])),
                        m[6],
                        int(m[7]),
                    )
                )
                continue
            raise CICOParseError(f"Invalid CICO token: {token!r}")

        return StateTransitionDelta(node_deltas, edge_deltas)
