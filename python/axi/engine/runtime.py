"""End-to-end pipeline: raw model text -> strict parse -> speculative page writes -> gluing gate -> commit/rollback."""
from dataclasses import dataclass
from .cico_parser import CICOParser, CICOParseError
from .gate import Graph, check, apply
from .pager import Pager, OutOfPhysicalBlocks, Conflict

W = 1 << 16


def vpn_of(n): return n[0] * W + n[1]


@dataclass
class Outcome:
    committed: bool
    stage: str   # parse | memory | gate | commit
    reason: str = ""


class Runtime:
    def __init__(self, capacity: int):
        self.graph, self.pager = Graph(), Pager(capacity)

    def propose(self, raw: str) -> Outcome:
        try:
            delta = CICOParser.parse_transition_delta(raw)
        except CICOParseError as e:
            return Outcome(False, "parse", str(e))
        tx = self.pager.begin_transaction()
        try:
            touched = {(d.r, d.c) for d in delta.node_deltas if d.op == "ADD"}
            for e in delta.edge_deltas:
                if e.op == "ADD": touched |= {e.u, e.v}
            for n in sorted(touched): tx.write_page(vpn_of(n))     # speculative "prefill"
        except OutOfPhysicalBlocks:
            tx.rollback(); return Outcome(False, "memory", "out of physical blocks")
        res = check(self.graph, delta)
        if not res.ok:
            tx.rollback(); return Outcome(False, "gate", res.reason)
        for d in delta.node_deltas:
            if d.op == "DEL": tx.unmap_page(vpn_of((d.r, d.c)))
        try:
            tx.commit()
        except Conflict as e:
            return Outcome(False, "commit", f"conflict on page {e}")
        apply(self.graph, delta)
        return Outcome(True, "commit")
