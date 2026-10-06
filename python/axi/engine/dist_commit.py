"""Layer 3 reference model: all-or-nothing page commit across tensor-parallel ranks.

Each rank owns a Pager (its HBM shard). A proposal is: speculative writes on EVERY rank -> each rank
votes (alloc ok? prepare ok?) -> consensus = gate_ok AND all votes (== all_reduce MIN over uint8)
-> commit on all ranks or rollback on all ranks. Proposals are serialized by the coordinator.
"""
from typing import List, Sequence
from .pager import Pager, OutOfPhysicalBlocks


class DistributedPager:
    def __init__(self, capacities: Sequence[int]):
        self.ranks: List[Pager] = [Pager(c) for c in capacities]

    def propose(self, writes: Sequence[int], unmaps: Sequence[int], gate_ok: bool = True) -> bool:
        txs, votes = [], []
        for p in self.ranks:                                   # phase 1: speculative isolation on every shard
            tx = p.begin_transaction(); ok = True
            try:
                for v in writes: tx.write_page(v)
                for v in unmaps: tx.unmap_page(v)
                ok = tx.prepare()
            except OutOfPhysicalBlocks:
                ok = False
            txs.append(tx); votes.append(1 if ok else 0)
        beta = int(gate_ok) & min(votes)                       # consensus: any 0 -> abort everywhere
        for tx in txs:                                         # phase 2: same decision on every rank
            tx.commit() if beta else tx.rollback()
        return bool(beta)

    def mapped_sets(self): return [frozenset(p.mapped_pages()) for p in self.ranks]

    def check(self):
        for p in self.ranks: p.audit()
        sets = self.mapped_sets()
        assert all(s == sets[0] for s in sets), "ranks diverged"


class NaiveIndependentRanks(DistributedPager):
    """Negative control: every rank decides on its own (no consensus)."""
    def propose(self, writes, unmaps, gate_ok=True):
        out = []
        for p in self.ranks:
            tx = p.begin_transaction()
            try:
                for v in writes: tx.write_page(v)
                for v in unmaps: tx.unmap_page(v)
                tx.commit() if gate_ok else tx.rollback(); out.append(gate_ok)
            except OutOfPhysicalBlocks:
                tx.rollback(); out.append(False)
        return all(out)
