import random
from axi.engine.runtime import Runtime, vpn_of
from axi.engine.cico_decoder import CICOGrammarDFA, TokenMasker, build_vocab, sample_delta
from axi.engine.pager import Pager, OutOfPhysicalBlocks, Conflict


def consistent(rt):
    rt.pager.audit()
    assert rt.graph.is_well_formed(), "dangling edge in committed state"
    assert set(rt.pager.mapped_pages()) == {vpn_of(n) for n in rt.graph.nodes}, "pages != nodes"
    assert rt.pager.free_blocks() == rt.pager.capacity - len(rt.graph.nodes), "leak"


def snapshot(rt):
    return dict(rt.graph.nodes), dict(rt.graph.edges), rt.pager.mapped_pages(), rt.pager.free_blocks()


def build(rt):
    for s in ("ADD[1,1:10] ADD[2,2:20] ADD[3,3:30]", "ADD[(1,1)->(2,2):dep#5] ADD[(3,3)->(2,2):dep#7]"):
        assert rt.propose(s).committed, s
    consistent(rt)


def test_happy_path():
    rt = Runtime(16); build(rt)
    assert len(rt.graph.nodes) == 3 and len(rt.graph.edges) == 2


def test_dangling_delete_rejected_and_state_untouched():
    rt = Runtime(16); build(rt); before = snapshot(rt)
    o = rt.propose("DEL[2,2]")
    assert (o.committed, o.stage) == (False, "gate") and "dangling" in o.reason
    assert snapshot(rt) == before; consistent(rt)


def test_delete_with_all_incident_edges_commits():
    rt = Runtime(16); build(rt)
    o = rt.propose("DEL[(1,1)->(2,2):dep#5] DEL[(3,3)->(2,2):dep#7] DEL[2,2]")
    assert o.committed, o.reason
    assert (2, 2) not in rt.graph.nodes and not rt.graph.edges; consistent(rt)


def test_match_must_be_a_real_morphism():
    rt = Runtime(16); build(rt)
    for s, kw in [("DEL[9,9]", "match"), ("DEL[(1,1)->(2,2):dep#99]", "weight"), ("DEL[(1,1)->(2,2):other#5]", "match"),
                  ("ADD[1,1:0]", "identification"), ("ADD[(1,1)->(2,2):dep#5]", "identification"),
                  ("ADD[(1,1)->(8,8):dep#1]", "well-formedness"), ("ADD[4,4:1] ADD[4,4:2]", "identification"),
                  ("ADD[4,4:1] DEL[4,4]", "match")]:
        before = snapshot(rt); o = rt.propose(s)
        assert not o.committed and o.stage == "gate" and kw in o.reason, (s, o)
        assert snapshot(rt) == before; consistent(rt)


def test_prose_and_malformed_rejected_at_parse():
    rt = Runtime(16); build(rt); before = snapshot(rt)
    for s in ["Sure! DEL[2,2]", "ADD[1,2]", "```DEL[1,1]```", '{"action":"delete"}']:
        o = rt.propose(s); assert not o.committed and o.stage == "parse", s
    assert snapshot(rt) == before; consistent(rt)


def test_oom_rolls_back_without_leak():
    # COW needs headroom: an edge add speculatively rewrites its endpoint pages (3 nodes -> 3 spare blocks)
    rt = Runtime(6); build(rt)
    assert rt.propose("ADD[4,4:1] ADD[5,5:1] ADD[6,6:1]").committed   # pool now full
    before = snapshot(rt); o = rt.propose("ADD[7,7:1]")
    assert (o.committed, o.stage) == (False, "memory"); assert snapshot(rt) == before; consistent(rt)
    # edge add on a full pool also fails cleanly (no spare block for the COW copy)
    o = rt.propose("ADD[(4,4)->(5,5):x#1]"); assert o.stage == "memory"; assert snapshot(rt) == before; consistent(rt)
    # partial speculative prefill then failure: no leak
    rt2 = Runtime(4); assert rt2.propose("ADD[1,1:1] ADD[2,2:1]").committed
    o = rt2.propose("ADD[4,4:1] ADD[5,5:1] ADD[6,6:1]"); assert o.stage == "memory"; consistent(rt2)


def test_pager_conflict_detected():
    p = Pager(4); t1 = p.begin_transaction(); t1.write_page(0); t1.commit()
    a, b = p.begin_transaction(), p.begin_transaction()
    a.write_page(0); b.write_page(0); a.commit()
    try: b.commit()
    except Conflict: pass
    else: raise AssertionError("expected Conflict")
    p.audit(); assert p.free_blocks() == 3


def test_cow_isolation():
    p = Pager(4); t = p.begin_transaction(); b0 = t.write_page(0).new_block; t.commit()
    t2 = p.begin_transaction(); w = t2.write_page(0)
    assert w.new_block != b0 and w.copy_from == b0 and p.read_page(0) == b0
    t2.rollback(); assert p.read_page(0) == b0 and p.free_blocks() == 3; p.audit()


def test_stress_decoder_proposals_never_break_invariants():
    """Mask-sampled deltas on a small grid + hand-built dangerous ones; invariants after EVERY proposal."""
    rng = random.Random(11)
    dfa = CICOGrammarDFA(); vocab, eos = build_vocab(); m = TokenMasker(dfa, vocab, eos)
    rt = Runtime(40); commits = {"gate": 0, "parse": 0, "memory": 0, "commit": 0}; ok = 0
    nodes = [f"ADD[{r},{c}:{r + c}]" for r in range(3) for c in range(3)]
    for i in range(4000):
        k = rng.random()
        if k < .4:
            raw, _ = sample_delta(m, rng, max_tokens=30)
        else:  # structured proposals over a tiny coordinate space so collisions/dangling actually happen
            def n(): return f"({rng.randrange(3)},{rng.randrange(3)})"
            if k < .6: raw = rng.choice(nodes)
            elif k < .75: raw = f"DEL[{rng.randrange(3)},{rng.randrange(3)}]"
            elif k < .9: raw = f"ADD[{n()}->{n()}:r{rng.randrange(2)}#{rng.randrange(3)}]"
            else: raw = f"DEL[{n()}->{n()}:r{rng.randrange(2)}#{rng.randrange(3)}]"
        o = rt.propose(raw)
        if o.committed: ok += 1
        else: commits[o.stage] += 1
        consistent(rt)
    assert ok > 200 and commits["gate"] > 200, (ok, commits)   # non-vacuous
    print("   stress:", ok, "committed;", commits, "rejected; final nodes", len(rt.graph.nodes), "edges", len(rt.graph.edges))


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("PASS", n)
