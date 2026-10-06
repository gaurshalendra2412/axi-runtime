"""Pure-Python reference model of the copy-on-write pager (mirrors crates/axi-cache semantics).

Ownership accounting: every physical block is in exactly one place: free, mapped, or held by a live
transaction. audit() checks free + mapped + in_flight == capacity with no overlaps.
begin O(1), write O(1), commit/rollback O(t) in blocks touched. Commit is a short locked update.
"""
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Set


class PagerError(Exception): pass
class OutOfPhysicalBlocks(PagerError): pass
class Conflict(PagerError): pass


@dataclass
class PageWrite:
    new_block: int
    copy_from: Optional[int]


class Pager:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._free: List[int] = list(range(capacity - 1, -1, -1))
        self._table: Dict[int, int] = {}
        self._live: List["Transaction"] = []
        self._lock = threading.Lock()

    def begin_transaction(self) -> "Transaction":
        tx = Transaction(self); self._live.append(tx); return tx

    def read_page(self, vpn): return self._table.get(vpn)
    def free_blocks(self): return len(self._free)
    def mapped_pages(self): return dict(self._table)

    def audit(self):
        with self._lock:
            free, mapped = set(self._free), set(self._table.values())
            held = [b for t in self._live for b in t._held()]
            assert len(free) == len(self._free), "duplicate in free list"
            assert len(mapped) == len(self._table), "block mapped twice"
            assert len(set(held)) == len(held), "block held twice by transactions"
            assert not (free & mapped) and not (free & set(held)) and not (mapped & set(held)), "block in two places"
            assert len(free) + len(mapped) + len(held) == self.capacity, "leak or phantom block"


class Transaction:
    def __init__(self, pager: Pager):
        self._p = pager; self._writes: Dict[int, int] = {}; self._base: Dict[int, Optional[int]] = {}
        self._unmaps: Dict[int, Optional[int]] = {}; self._open = True

    def _held(self): return list(self._writes.values())

    def write_page(self, vpn: int) -> PageWrite:
        assert self._open
        if vpn in self._writes: return PageWrite(self._writes[vpn], None)  # already private to this tx
        p = self._p
        with p._lock:
            if not p._free: raise OutOfPhysicalBlocks()
            b = p._free.pop()
            self._base[vpn] = p._table.get(vpn)
            self._writes[vpn] = b
            self._unmaps.pop(vpn, None)
            return PageWrite(b, self._base[vpn])

    def unmap_page(self, vpn: int):
        """Python-model addition: drop a page at commit (node deleted). Not yet in the Rust crate."""
        assert self._open
        with self._p._lock:
            if vpn in self._writes:  # discard private block
                self._p._free.append(self._writes.pop(vpn))
                base = self._base.pop(vpn)
            else:
                base = self._p._table.get(vpn)
            self._unmaps[vpn] = base

    def prepare(self) -> bool:
        """2PC phase 1 check: True iff commit() would not hit a Conflict right now. Valid for commit
        only if proposals are serialized by a coordinator (nothing else commits in between)."""
        p = self._p
        with p._lock:
            return all(p._table.get(v) == b for v, b in list(self._base.items()) + list(self._unmaps.items()))

    def commit(self):
        assert self._open; p = self._p
        with p._lock:
            for vpn, base in list(self._base.items()) + list(self._unmaps.items()):
                if p._table.get(vpn) != base:
                    self._release_locked(); raise Conflict(vpn)
            for vpn, b in self._writes.items():
                old = p._table.get(vpn)
                if old is not None: p._free.append(old)
                p._table[vpn] = b
            for vpn in self._unmaps:
                old = p._table.pop(vpn, None)
                if old is not None: p._free.append(old)
            self._writes = {}; self._open = False; p._live.remove(self)

    def rollback(self):
        if not self._open: return
        with self._p._lock: self._release_locked()

    def _release_locked(self):
        self._p._free.extend(self._writes.values())
        self._writes = {}; self._open = False; self._p._live.remove(self)
