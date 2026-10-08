"""Implements concept-map row 136 (thirteenth scan): four 11 Aug pages that talk about the same container - "The delusion of maths and page and knowledge" (Post 348, two reads),
"That is the sacred geometry of the final page" (Post 369, two reads), "You have just shown me the physical, digital footprint of the last 10 days" (Post 371, one read) and "To the high-parameter
excited people..." (Post 368, one read). Quotes verified (YES) on the second reads: "the 1,024 MB of digital memory you have reserved" and "the 784 MB of silence you have kept"; "The
story of AI is found in the exact 240 MB you have already used"; "And the 784 MB of empty space?" "That is the silence between the folds."; "In a 2D world, empty space is wasted.";
"It is 2 raised to the power of 10." The page never states 240 + 784. Post 371 says the figures are a blog host's storage dashboard ("The storage: 1,024 MB total, with 240 MB used."), so
1,024 / 240 / 784 are a hosting quota and what it showed on the day, not a design number of AXI, and 784 = 1,024 - 240 is a subtraction, not a pattern.

The reading being tested (Rajnish to accept or reject): "a container of 1,024 with 240 used and 784 kept empty" is the shape of the copy-on-write pager in `axi/engine/pager.py`, where
the empty space is not waste but what makes a commit safe: a write never overwrites a page in place, it takes a FREE block, and the old block is released only at commit. Counted on the
repository's reference pager, one block standing for one MB (a free choice):
  1. Pager(1024) with 240 pages mapped has 784 free; audit() (free + mapped + held == capacity) holds before, during and after.
  2. Rewriting all 240 pages in one transaction takes 240 more blocks while the old 240 stay mapped (free 544 in flight, 240 held), and after commit the old 240 are released:
     free is 784 again. Rollback instead returns every held block. The largest amount any transaction can hold is the free count: 784 writes succeed, the 785th raises OutOfPhysicalBlocks.
  3. A container that is FULL cannot change at all: with 0 free even a one-page rewrite of an existing page raises OutOfPhysicalBlocks. It can still DELETE (unmap needs no free block);
     after one delete commits, one write is possible again. So "empty space is wasted" (the page) and "empty space is what lets a write happen" (the code) are opposite statements about the same blocks.
  4. The general rule, brute-forced on a small container (capacity 64, every M from 0 to 64 mapped pages): a transaction can rewrite ALL M mapped pages if and only if M <= capacity - M,
     that is, if the container is at most HALF full. At 1,024 with 240 used the container is 23.4% full, so a whole-container rewrite fits with 544 blocks to spare.
What they do NOT show: that the page means a copy-on-write reserve (it calls the 784 "silence" and a space "to fill with exactly the right amount of truth"); that a KV cache or a graph store
for a model needs a reserve of this size (no measurement here, and no model is involved); that half-full is a good operating point (it is the limit for rewriting everything at once, not a
recommendation). Copy-on-write needing headroom is standard; the Rust crate has no unmap yet (see the docstring of the reference model)."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.pager import Pager, OutOfPhysicalBlocks


def filled(capacity, n):
    p = Pager(capacity)
    tx = p.begin_transaction()
    for v in range(n): tx.write_page(v)
    tx.commit()
    return p


def test_1024_with_240_mapped_has_784_free_and_the_ledger_balances():
    p = filled(1024, 240)
    p.audit()
    assert p.free_blocks() == 784 == 1024 - 240 and len(p.mapped_pages()) == 240
    assert 240 + 784 == 1024 == 2 ** 10                                       # the sum the pages imply but never state
    assert (round(100 * 240 / 1024), round(100 * 784 / 1024)) == (23, 77)


def test_rewriting_all_240_pages_holds_240_more_blocks_until_commit_and_returns_to_784_free():
    p = filled(1024, 240)
    old = dict(p.mapped_pages())
    tx = p.begin_transaction()
    for v in range(240): tx.write_page(v)
    p.audit()
    assert p.free_blocks() == 544 and p.mapped_pages() == old                 # the old blocks are still the mapped ones while the new ones are held
    assert tx.prepare()
    tx.commit()
    p.audit()
    new = p.mapped_pages()
    assert p.free_blocks() == 784 and len(new) == 240 and not (set(new.values()) & set(old.values()))
    assert all(new[v] != old[v] for v in range(240))                          # every page moved to a fresh block
    rb = p.begin_transaction()
    for v in range(240): rb.write_page(v)
    rb.rollback()
    p.audit()
    assert p.free_blocks() == 784 and p.mapped_pages() == new                 # rollback gives every held block back and changes nothing
    most = p.begin_transaction()
    for v in range(1000, 1000 + 784): most.write_page(v)
    try:
        most.write_page(5000)
        raise AssertionError("the 785th block should not exist")
    except OutOfPhysicalBlocks:
        pass
    p.audit()
    most.rollback()
    assert p.free_blocks() == 784


def test_a_full_container_cannot_rewrite_even_one_page_but_can_delete_and_then_write_again():
    p = filled(1024, 1024)
    p.audit()
    assert p.free_blocks() == 0
    tx = p.begin_transaction()
    try:
        tx.write_page(0)
        raise AssertionError("a full container accepted a write")
    except OutOfPhysicalBlocks:
        pass
    tx.rollback()
    dele = p.begin_transaction()
    dele.unmap_page(0)                                                        # deleting needs no free block
    dele.commit()
    p.audit()
    assert p.free_blocks() == 1 and 0 not in p.mapped_pages()
    again = p.begin_transaction()
    again.write_page(1)                                                       # one block is enough for one rewrite
    again.commit()
    p.audit()
    assert p.free_blocks() == 1
    scratch = p.begin_transaction()
    scratch.write_page(2000)                                                  # a page written and then dropped in the same transaction hands its private block straight back
    assert p.free_blocks() == 0
    scratch.unmap_page(2000)
    p.audit()
    assert p.free_blocks() == 1
    scratch.commit()
    p.audit()
    assert p.free_blocks() == 1 and 2000 not in p.mapped_pages()


def test_a_whole_container_rewrite_fits_exactly_when_the_container_is_at_most_half_full():
    cap = 64
    for m in range(cap + 1):
        p = filled(cap, m)
        tx = p.begin_transaction()
        done = 0
        try:
            for v in range(m):
                tx.write_page(v); done += 1
            ok = True
        except OutOfPhysicalBlocks:
            ok = False
        assert ok == (m <= cap - m), m
        if not ok: assert done == cap - m                                     # it stops exactly when the free blocks run out
        tx.rollback()
        p.audit()
        assert p.free_blocks() == cap - m
    assert 240 <= 1024 - 240 and 1024 - 2 * 240 == 544


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
