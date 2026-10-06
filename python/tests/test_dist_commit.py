import random
from axi.engine.dist_commit import DistributedPager, NaiveIndependentRanks


def test_one_rank_oom_aborts_everywhere():
    d = DistributedPager([8, 8, 2, 8])                     # rank 2 is small
    assert d.propose([1, 2], [], True); d.check()          # fits everywhere (2 blocks)
    before = d.mapped_sets()
    assert not d.propose([3], [], True)                    # rank 2 full -> all abort
    assert d.mapped_sets() == before; d.check()


def test_gate_failure_aborts_everywhere():
    d = DistributedPager([4] * 4); d.propose([1], [], True)
    before = d.mapped_sets(); assert not d.propose([2], [], gate_ok=False)
    assert d.mapped_sets() == before; d.check()
    assert all(p.free_blocks() == 3 for p in d.ranks)


def test_naive_independent_ranks_diverge():
    n = NaiveIndependentRanks([8, 8, 2, 8])
    n.propose([1, 2], [], True); n.propose([3], [], True)
    sets = n.mapped_sets(); assert any(s != sets[0] for s in sets), "expected split-brain without consensus"


def test_fault_injection_fuzz_never_diverges():
    rng = random.Random(21); commits = aborts = 0
    d = DistributedPager([30, 30, 9, 30, 14])
    for i in range(6000):
        mapped = set(d.ranks[0].mapped_pages())
        writes = rng.sample(range(25), rng.randint(0, 3))
        unmaps = [v for v in rng.sample(range(25), rng.randint(0, 2)) if v in mapped and v not in writes]
        ok = d.propose(writes, unmaps, gate_ok=rng.random() > .2)
        commits += ok; aborts += (not ok); d.check()
        if not ok:   # aborted proposals leave nothing in flight
            assert all(len(p._live) == 0 for p in d.ranks)
    assert commits > 500 and aborts > 500, (commits, aborts)
    print("   fuzz: commits", commits, "aborts", aborts)


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("PASS", n)
