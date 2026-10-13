"""Implements concept-map item 92 (endurance and cost; rows 3, 9, 27): the INSTRUMENT of docs/ENDURANCE_TEST.md, written before the Part C run. It checks the instrument, not a result:
  1. the request stream is the old generator's, one step at a time and never stored: its first n requests and states equal `oracle_script`'s for both earlier mixes, it does not stop where the old one would
     (an empty graph gets an add_node request, after every kind was tried once in a fixed order), and the `steady` mix stays inside the caps for thousands of steps;
  2. each of the 15 kinds of damaged reply does exactly what the page says (and nothing when there is nothing to damage), is deterministic given its generator, and the ones the page calls unreadable are refused by
     the strict parser while the others are not;
  3. the stand-in model deals the kinds from a shuffled deck (each equally often), is seeded, and writes the ideal reply when it is not damaging;
  4. the run: a clean run leaves every policy exact; damage corrupts blind apply and never the gated policies; policies do not influence each other; a crash is counted and the run goes on (in a policy and in the
     parser); time is the policy's own update (checked with a fake clock); blocks, counters and memory keys are exact; the same seed gives the same counts;
  5. every reading-rule line flips exactly at its boundary (50 percent, 30 services and 60 edges, 1.5 times, 100 microseconds, 1.25 times, 3 and 10 times) and when one number is doctored;
  6. the page and the module agree (the ten lines, tags, seeds, sizes, limits, the 15 kinds) and the page's text above its Result is byte for byte what was written before the run (a hash);
  7. Part A: the straight line through the history growth, the total of a growing context and the step at which a window is passed are exact on hand-built numbers;
  8. Part B: the replay reproduces the `exact` flag of every step of hand-built runs, detects a doctored flag, splits requests into applicable and not, counts the same wasted tokens as `confirm_test.wasted_tokens`.
What it does NOT show: anything about a real model's replies, or the result of the run."""
import contextlib, hashlib, io, os, re, sys
from unittest import mock
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.gate import Graph
from axi.experiments import endurance as E
from axi.experiments.agent_loop import copy_graph, same_graph, strict_parse
from axi.experiments.cascade import next_state
from axi.experiments.confirm_test import wasted_tokens
from axi.experiments.long_horizon import oracle_script

HERE = os.path.dirname(__file__)
PAGE = os.path.join(HERE, "..", "..", "docs", "ENDURANCE_TEST.md")
PAGE_SHA256 = "952b1d27c2e2c5610e641e2690a8f3541e3e874aa0d78b095b9dc6209a84f4ab"          # filled in when the page was written, before any Part C run
RESULT_MARK = "\n\n---\n\n# Result"


def _h(text): return hashlib.sha256((text.rstrip("\n") + "\n").encode()).hexdigest()


# ---------------------------------------------------------------- 1 the stream
def test_the_stream_is_the_old_generators_prefix_and_goes_on_where_the_old_one_stopped():
    stopped_early = []
    for mix, seeds, n in (("base", list(range(8)) + [21000, 3000, 150], 200), ("scope", list(range(8)) + [21000, 21001, 74, 85], 60)):
        for seed in seeds:
            g0, script, oracle = oracle_script(seed, n, mix); S = E.Stream(seed, mix); assert same_graph(S.start, g0)
            for t, st in enumerate(script):
                got = S.next(); assert got is not None and got[0] == st, (mix, seed, t)
                assert same_graph(got[1], oracle[t]) and same_graph(S.live, oracle[t + 1]), (mix, seed, t)
            if len(script) < n:
                stopped_early.append((mix, seed, len(script))); nxt = S.next(); assert nxt is not None and same_graph(nxt[1], oracle[-1])      # where the old one stopped, this one goes on from the same state
    assert ("scope", 74, 22) in stopped_early and ("base", 150, 116) in stopped_early and ("scope", 85, 43) in stopped_early, stopped_early


def test_a_stream_that_would_have_stopped_gets_an_add_node_request_and_the_fallback_is_in_fixed_order():
    S = E.Stream(5, "steady"); S.live = Graph(); calls = []
    real = S.picker
    def stingy(rng, g, kind):
        calls.append(kind); return None if len(calls) <= 20 else real(rng, g, kind)
    S.picker = stingy; st, before = S.next()
    assert st["kind"] == "add_node" and len(before.nodes) == 0 and len(S.live.nodes) == 1 and len(calls) > 20
    assert calls[20:] == ["add_edge", "add_node"], calls[20:]                                                         # sorted kinds: add_edge cannot be made on an empty graph, add_node can
    S2 = E.Stream(5, "steady"); S2.live = Graph(); S2.picker = lambda rng, g, kind: None; assert S2.next() is None     # nothing fits at all: the stream ends
    assert E.Stream(77, "steady").start.nodes == oracle_script(77, 1, "scope")[0].nodes                               # the same starting graph as the old generator for the same seed


def test_the_steady_mix_stays_inside_the_caps_and_uses_every_kind_for_thousands_of_steps():
    S = E.Stream(77000, "steady"); kinds = set(); maxn = maxe = 0; emptied = 0
    for i in range(5000):
        got = S.next(); assert got is not None, i; kinds.add(got[0]["kind"]); g = S.live
        assert g.is_well_formed() and len(g.nodes) <= E.MAX_NODES and len(g.edges) <= E.MAX_EDGES, i
        maxn, maxe = max(maxn, len(g.nodes)), max(maxe, len(g.edges)); emptied += (len(g.nodes) == 0)
    assert kinds == {"add_node", "add_edge", "delete_dep", "delete_leaf", "delete_edge", "delete_pair"} and maxn >= 8 and maxe >= 3
    assert E.KINDS_STEADY.count("add_node") == 3 and E.KINDS_STEADY.count("add_edge") == 3 and len(E.KINDS_STEADY) == 11


# ---------------------------------------------------------------- 2 the damage
def _case():
    g = Graph()
    for n in [(0, 0), (0, 1), (1, 1), (2, 2)]: g.add_node(n, 5)
    g.add_edge((0, 0), (0, 1), "dep", 3); g.add_edge((1, 1), (0, 0), "owns", 4); g.add_edge((2, 2), (1, 1), "dep", 7)
    return g, "DEL[0,0] DEL[(0,0)->(0,1):dep#3] DEL[(1,1)->(0,0):owns#4]"


def _damage(kind, seed=1, previous="ADD[3,3:1]"):
    g, ideal = _case(); import random
    return E.damage(kind, random.Random(seed), ideal, g, previous), ideal, g


def test_each_kind_of_damage_does_what_the_page_says():
    r, ideal, g = _damage("forget_dependents"); assert r == "DEL[0,0]" and strict_parse(r) is not None
    r, ideal, g = _damage("extra_edge_delete"); extra = E.ITEM.findall(r)[3:]; assert r.startswith(ideal + " ") and len(extra) == 1 and extra[0] == "DEL[(2,2)->(1,1):dep#7]"
    r, _, _ = _damage("wrong_coordinate"); a, b = re.findall(r"\d+", ideal), re.findall(r"\d+", r); assert len(a) == len(b) and sum(x != y for x, y in zip(a, b)) <= 1 and all(0 <= int(x) <= 30 for x in b)
    for seed in range(40):
        r, _, _ = _damage("truncate", seed); assert 0 < len(r) < len(ideal) and ideal.startswith(r)
    r, _, _ = _damage("drop_item"); assert len(E.ITEM.findall(r)) == 2 and set(E.ITEM.findall(r)) <= set(E.ITEM.findall(ideal))
    r, _, _ = _damage("duplicate_item"); its = E.ITEM.findall(r); assert len(its) == 4 and set(its) == set(E.ITEM.findall(ideal))
    r, _, _ = _damage("extra_add"); assert r.startswith(ideal + " ADD[") and len(E.ITEM.findall(r)) == 4 and strict_parse(r) is not None
    for seed in range(200):
        r, _, _ = _damage("garbage", seed); assert 5 <= len(r) <= 80 and strict_parse(r) is None, (seed, r)
    assert _damage("empty")[0] == "" and strict_parse("") is None
    r, _, _ = _damage("huge"); d = strict_parse(r); assert len(E.ITEM.findall(r)) == E.HUGE_ITEMS == 1000 and d is not None and len(d.node_deltas) + len(d.edge_deltas) == 1000
    r, _, _ = _damage("bigint"); d = strict_parse(r); assert "9" * 40 in r and "9" * 41 not in r and d is not None and r.count("9" * 40) == 1
    assert _damage("stale", previous="DEL[7,7]")[0] == "DEL[7,7]" and _damage("stale")[0] == "ADD[3,3:1]"
    for seed in range(20):
        r, _, _ = _damage("format_slip", seed); assert re.search(r"DEL\[0,0:[1-9]\]", r) and strict_parse(r) is None
    r, _, _ = _damage("unicode_digits"); assert any(ord(ch) > 127 and ch.isdigit() for ch in r) and sum(1 for ch in r if ord(ch) > 127) == 1
    r, _, _ = _damage("negative"); assert r.startswith("DEL[-0,0]") and strict_parse(r) is None


def test_a_kind_with_nothing_to_damage_leaves_the_reply_alone_and_every_kind_is_deterministic():
    import random
    g, _ = _case(); edge_only = "DEL[(2,2)->(1,1):dep#7]"; r = random.Random(3)
    for kind in ("forget_dependents", "format_slip", "negative"): assert E.damage(kind, r, edge_only, g, "") == edge_only, kind
    assert E.damage("extra_edge_delete", r, "", Graph(), "") == "" and E.damage("drop_item", r, "", g, "") == "" and E.damage("duplicate_item", r, "", g, "") == ""
    assert E.damage("wrong_coordinate", r, "garbage", g, "") == "garbage" and E.damage("unicode_digits", r, "garbage", g, "") == "garbage" and E.damage("bigint", r, "garbage", g, "") == "garbage"
    assert E.damage("truncate", r, "x", g, "") == "x"
    assert len(E.FAULTS) == 15 and len(set(E.FAULTS)) == 15
    for kind in E.FAULTS:
        a = _damage(kind, 9)[0]; assert a == _damage(kind, 9)[0], kind
    try: E.damage("nonsense", random.Random(1), "DEL[0,0]", g, ""); raise AssertionError("an unknown kind was accepted")
    except ValueError: pass


# ---------------------------------------------------------------- 3 the stand-in
def test_the_stand_in_deals_each_kind_equally_often_and_is_seeded():
    g, ideal = _case()
    G = E.Garbler(1, 1.0); kinds = [G.reply(ideal, g)[1] for _ in range(150)]
    assert all(k in E.FAULTS for k in kinds) and all(kinds.count(f) == 10 for f in E.FAULTS) and all(len(set(kinds[i:i + 15])) == 15 for i in range(0, 150, 15))
    assert kinds[:15] != kinds[15:30] and kinds[:15] != list(E.FAULTS) and sorted(kinds[:15]) == sorted(E.FAULTS)                      # each deck is shuffled, and shuffled anew
    G = E.Garbler(1, 0.0); assert all(G.reply(ideal, g) == (ideal, None) for _ in range(50))
    G = E.Garbler(2, 0.5); n = [G.reply(ideal, g)[1] is not None for _ in range(4000)]; assert 0.45 < sum(n) / 4000 < 0.55
    a = [E.Garbler(5).reply(ideal, g) for _ in range(1)]; b = [E.Garbler(5).reply(ideal, g) for _ in range(1)]; assert a == b
    A, B = E.Garbler(6), E.Garbler(6); assert [A.reply(ideal, g) for _ in range(200)] == [B.reply(ideal, g) for _ in range(200)]
    assert [E.Garbler(6).reply(ideal, g) for _ in range(40)] != [E.Garbler(7).reply(ideal, g) for _ in range(40)]
    G = E.Garbler(1, 1.0, faults=("stale",)); assert G.reply("DEL[1,1]", g) == ("", "stale") and G.reply("DEL[2,2]", g) == ("DEL[1,1]", "stale")          # `previous` is the ideal reply of the step before
    assert E.P_FAULT == 0.5


# ---------------------------------------------------------------- 4 the run
def _strip(r):
    r = {k: v for k, v in r.items() if k != "peak_rss_mb"}; r["policies"] = {p: {k: v for k, v in s.items() if k not in ("ns",)} for p, s in r["policies"].items()}; return r


def test_a_clean_run_is_exact_for_every_policy_and_damage_corrupts_only_blind_apply():
    r = E.run_endurance(4, 300, 11, p=0.0)
    assert r["steps_run"] == 1200 and r["clean_replies"] == 1200 and r["damaged_replies"] == 0 and r["dealt"] == 0 and r["unparsed"] == 0 and r["parse_crashes"] == 0 and r["crashes"] == {}
    for p, s in r["policies"].items(): assert s["steps"] == 1200 and s["exact"] == 1200 and s["corrupted"] == 0 and s["crashes"] == 0 and s["max_distance"] == 0, p
    d = E.run_endurance(6, 300, 12, p=0.5)
    assert d["steps_run"] == 1800 and d["clean_replies"] + d["damaged_replies"] == 1800 and d["damaged_replies"] > 300 and d["dealt"] > d["damaged_replies"] > 0.8 * d["dealt"] and sum(d["fault_counts"].values()) == d["dealt"]
    assert d["unparsed"] > 0 and d["parse_crashes"] == 0 and d["crashes"] == {}
    for p in E.GATED: assert d["policies"][p]["corrupted"] == 0 and d["policies"][p]["steps"] == 1800, p
    assert d["policies"]["blind"]["corrupted"] > 900 and d["policies"]["blind"]["max_edges"] > d["policies"]["scope_m7"]["max_edges"]
    assert d["policies"]["scope_m7"]["max_edges"] <= E.LIMIT_EDGES and d["policies"]["scope_m7"]["max_nodes"] <= E.LIMIT_NODES
    assert set(d["peak_rss_mb"]) == {"after_10_percent", "end"} and d["peak_rss_mb"]["end"] >= d["peak_rss_mb"]["after_10_percent"] > 0
    assert d["meta"] == dict(episodes=6, steps=300, seed=12, p_fault=0.5, faults=list(E.FAULTS), mix="steady", policies=list(E.POLICIES_C))


def test_policies_do_not_influence_each_other_and_the_same_seed_gives_the_same_counts():
    a = E.run_endurance(3, 200, 13, policies=E.GATED); b = E.run_endurance(3, 200, 13, policies=("scope_m7",)); c = E.run_endurance(3, 200, 13, policies=("blind", "scope_m7"))
    assert _strip(a)["policies"]["scope_m7"] == _strip(b)["policies"]["scope_m7"] == _strip(c)["policies"]["scope_m7"]
    for k in ("steps_run", "clean_replies", "damaged_replies", "unparsed", "fault_counts"): assert a[k] == b[k] == c[k], k
    assert _strip(a) == _strip(E.run_endurance(3, 200, 13, policies=E.GATED)) and _strip(a) != _strip(E.run_endurance(3, 200, 14, policies=E.GATED))
    first = E.run_endurance(5, 150, 15, policies=("blind",)); more = E.run_endurance(3, 150, 15, policies=("blind",))
    assert more["steps_run"] == 450 and first["steps_run"] == 750                                                         # the first episodes of a longer run are the shorter run's episodes (same stream seeds)


def test_a_second_implementation_agrees_with_the_run():
    from axi.experiments.long_horizon import distance
    ep, n, seed, pols = 3, 120, 31, E.GATED
    r = E.run_endurance(ep, n, seed, policies=pols)
    tot = {p: dict(corrupted=0, exact=0, dsum=0, maxn=0, maxe=0, maxd=0) for p in pols}; unparsed = damaged = 0
    for e in range(ep):
        S = E.Stream(seed * 1000 + e); G = E.Garbler(seed * 1000 + e); held = {p: copy_graph(S.start) for p in pols}
        for _ in range(n):
            st, before = S.next(); text, kind = G.reply(st["ideal"], before); damaged += (text != st["ideal"]); d = strict_parse(text); unparsed += (d is None)
            for p in pols:
                held[p] = next_state(p, held[p], d, st["instruction"]); g = held[p]; dist = distance(g, S.live); t = tot[p]
                t["corrupted"] += (not g.is_well_formed()); t["exact"] += (dist == 0); t["dsum"] += dist; t["maxn"] = max(t["maxn"], len(g.nodes)); t["maxe"] = max(t["maxe"], len(g.edges)); t["maxd"] = max(t["maxd"], dist)
    assert r["damaged_replies"] == damaged > 0 and r["unparsed"] == unparsed > 0
    for p in pols:
        s_ = r["policies"][p]; t = tot[p]
        assert (s_["corrupted"], s_["exact"], sum(s_["dist_sum"]), s_["max_nodes"], s_["max_edges"], s_["max_distance"]) == (t["corrupted"], t["exact"], t["dsum"], t["maxn"], t["maxe"], t["maxd"]), p
    assert tot["gate"]["maxn"] != tot["gate"]["maxe"] and tot["repair_m6"]["maxd"] > 0


def test_the_replies_are_made_from_the_oracles_state_before_the_step():
    seen = []; real = E.Garbler.reply
    def spy(self, ideal, before): seen.append((ideal, copy_graph(before))); return real(self, ideal, before)
    with mock.patch.object(E.Garbler, "reply", spy): E.run_endurance(2, 60, 33, policies=("scope_m7",))
    exp = []
    for e in range(2):
        S = E.Stream(33 * 1000 + e)
        for _ in range(60): st, before = S.next(); exp.append((st["ideal"], before))
    assert len(seen) == 120 == len(exp) and all(a[0] == b[0] and same_graph(a[1], b[1]) for a, b in zip(seen, exp))
    assert sum(1 for (_, g), (_, h) in zip(seen[:-1], seen[1:]) if not same_graph(g, h)) > 40                       # the state does move, so "before" and "after" are different things


def test_exact_counts_steps_at_distance_zero_only():
    seq = {"i": 0}
    def fake(a, b): seq["i"] += 1; return seq["i"] % 3
    with mock.patch.object(E, "distance", fake): r = E.run_endurance(1, 30, 5, policies=("gate",))
    s = r["policies"]["gate"]; assert s["exact"] == 10 and sum(s["dist_sum"]) == sum(i % 3 for i in range(1, 31)) and s["max_distance"] == 2


def test_a_crash_is_counted_and_the_run_goes_on():
    real = E.next_state; seen = {"n": 0}
    def flaky(pol, g, d, instr=None, names=None):
        if pol == "gate":
            seen["n"] += 1
            if seen["n"] % 50 == 0: raise RuntimeError("boom")
        return real(pol, g, d, instr)
    with mock.patch.object(E, "next_state", flaky): r = E.run_endurance(2, 200, 16, policies=E.GATED)
    assert r["steps_run"] == 400 and r["policies"]["gate"]["crashes"] == 400 // 50 and r["crashes"]["gate:RuntimeError"]["count"] == 8 and r["crashes"]["gate:RuntimeError"]["text"] is not None
    assert "boom" in r["crashes"]["gate:RuntimeError"]["trace"] and r["policies"]["scope_m7"]["crashes"] == 0 and r["policies"]["gate"]["steps"] == 400 and r["policies"]["gate"]["corrupted"] == 0
    def bad_parse(text):
        if "DEL" in text and len(text) % 7 == 0: raise ValueError("bad")
        return strict_parse(text)
    with mock.patch.object(E, "strict_parse", bad_parse): r = E.run_endurance(2, 200, 17, policies=E.GATED)
    assert r["parse_crashes"] > 0 and r["crashes"]["parse:ValueError"]["count"] == r["parse_crashes"] and r["unparsed"] >= r["parse_crashes"] and r["steps_run"] == 400
    ok = E.run_endurance(2, 200, 17, policies=E.GATED); assert ok["parse_crashes"] == 0 and ok["crashes"] == {}


def test_a_crash_leaves_the_held_state_as_it_was():
    real = E.next_state
    def make(mode):
        cnt = {"n": 0}
        def f(pol, g, d, instr=None, names=None):
            if pol == "gate":
                cnt["n"] += 1
                if cnt["n"] % 50 == 0:
                    if mode == "raise": raise RuntimeError("boom")
                    return copy_graph(g)                                                                          # what a refused reply does: nothing changes
            return real(pol, g, d, instr)
        return f
    with mock.patch.object(E, "next_state", make("raise")): a = E.run_endurance(2, 200, 16, policies=E.GATED)
    with mock.patch.object(E, "next_state", make("keep")): b = E.run_endurance(2, 200, 16, policies=E.GATED)
    drop = lambda r: {p: {k: v for k, v in s.items() if k not in ("ns", "crashes")} for p, s in r["policies"].items()}
    assert a["policies"]["gate"]["crashes"] == 8 and b["policies"]["gate"]["crashes"] == 0 and drop(a) == drop(b)


def test_memory_is_read_after_the_first_tenth_of_the_episodes_and_at_the_end():
    class RU:
        def __init__(self, v): self.ru_maxrss = v * 1024
    for episodes, want_steps in ((20, 10), (3, 5), (10, 5)):
        seen = []; vals = iter(range(1, 100)); n = {"steps": 0}; real_next = E.Stream.next
        def counting(self): n["steps"] += 1; return real_next(self)
        def fake(who): seen.append(n["steps"]); return RU(next(vals))
        with mock.patch.object(E.Stream, "next", counting), mock.patch.object(E.resource, "getrusage", fake):
            r = E.run_endurance(episodes, 5, 21, policies=("scope_m7",))
        assert r["peak_rss_mb"] == {"after_10_percent": 1.0, "end": 2.0} and seen == [want_steps, 5 * episodes], (episodes, seen)


def test_time_is_the_policys_own_update_and_the_blocks_add_up():
    ticks = {"n": 0}
    def clock(): ticks["n"] += 1; return ticks["n"] * 1000
    r = E.run_endurance(2, 100, 18, policies=("scope_m7", "gate"), clock=clock)
    for p, s in r["policies"].items():
        assert s["steps"] == 200 and sum(s["ns_n"]) == 200 and sum(s["dist_n"]) == 200 and len(s["ns"]) == E.BLOCKS == 10
        assert abs(sum(s["ns"]) - 200 * 1000) < 1e-6 and all(abs(x - 1.0) < 1e-9 for x in E.block_us(s)), p                  # two clock readings a step, 1000 ns apart
        assert s["ns_n"] == [20] * 10 and len(E.block_distance(s)) == 10
    assert E.block_us(dict(ns=[0] * 10, ns_n=[0] * 10)) == [None] * 10 and E.block_distance(dict(dist_sum=[0] * 10, dist_n=[0] * 10)) == [None] * 10
    r = E.run_endurance(1, 7, 19, policies=("gate",)); assert len(r["policies"]["gate"]["ns"]) == 10 and sum(r["policies"]["gate"]["ns_n"]) == 7           # fewer steps than blocks: nothing is lost


# ---------------------------------------------------------------- 5 the reading rules
def _pol(steps=1000, corrupted=0, exact=0, max_nodes=10, max_edges=10, dist=10.0, us=30.0, crashes=0):
    k = steps // 10; d = dist if isinstance(dist, (list, tuple)) else [dist] * 10; u = us if isinstance(us, (list, tuple)) else [us] * 10
    return dict(steps=steps, corrupted=corrupted, exact=exact, crashes=crashes, dist_sum=[x * k for x in d], dist_n=[k] * 10, ns=[x * 1000 * k for x in u], ns_n=[k] * 10, max_distance=50, max_nodes=max_nodes, max_edges=max_edges, dangling_max=0, corrupted_by_fault={})


def _run(policies, steps_run=1000, crashes=None, parse_crashes=0, rss=(20.0, 20.0)):
    return dict(meta={}, steps_run=steps_run, clean_replies=500, damaged_replies=500, dealt=500, unparsed=100, parse_crashes=parse_crashes, fault_counts={}, crashes=crashes or {}, peak_rss_mb=dict(after_10_percent=rss[0], end=rss[1]), policies=policies)


def _res():
    gated = lambda **kw: {"gate": _pol(dist=120.0, **kw), "repair_m6": _pol(dist=40.0, **kw), "scope_m7": _pol(**kw)}
    return dict(control=_run({p: _pol(exact=1000) for p in E.POLICIES_C}), million=_run(gated()), blind=_run({"blind": _pol(corrupted=600)}), long=[_run(gated(), steps_run=250000), _run(gated(), steps_run=250000)])


def _v(res): return [ok for _, ok in E.verdict_endurance(res)]


def test_every_reading_rule_line_flips_at_its_boundary():
    import copy
    assert _v(_res()) == [True] * 10 and [t for t, _ in E.verdict_endurance(_res())] == [t for _, t in E.LINES_C]
    r = _res(); r["blind"]["policies"]["blind"]["corrupted"] = 500; assert _v(r)[3] and _v(r)[:3] == [True] * 3                 # exactly 50 percent
    r["blind"]["policies"]["blind"]["corrupted"] = 499; assert _v(r)[3] is False and sum(_v(r)) == 9
    r = _res(); r["million"]["parse_crashes"] = 1; assert _v(r)[0] is False and sum(_v(r)) == 9
    r = _res(); r["long"][1]["crashes"] = {"gate:KeyError": dict(count=1)}; assert _v(r)[0] is False and sum(_v(r)) == 9
    r = _res(); r["control"]["parse_crashes"] = 2; assert _v(r)[0] is False
    r = _res(); r["blind"]["parse_crashes"] = 1; assert _v(r)[0] is False
    for where in (("million",), ("long", 0), ("long", 1)):
        for pol in E.GATED:
            r = _res(); t = r[where[0]] if len(where) == 1 else r[where[0]][where[1]]; t["policies"][pol]["corrupted"] = 1; assert _v(r)[1] is False and sum(_v(r)) == 9, (where, pol)
    r = _res(); r["blind"]["policies"]["blind"]["corrupted"] = 1000; assert _v(r)[1] is True                                     # blind apply being corrupted is expected and is not line 2
    r = _res(); r["control"]["policies"]["blind"]["exact"] = 999; assert _v(r)[2] is False and sum(_v(r)) == 9
    r = _res(); del r["control"]["policies"]["gate"]; assert _v(r)[2] is False
    r = _res(); r["control"]["policies"]["gate"]["steps"] = 0; r["control"]["policies"]["gate"]["exact"] = 0; assert _v(r)[2] is False
    for field, ok_v, bad_v in (("max_nodes", 30, 31), ("max_edges", 60, 61)):
        for where in (("million",), ("long", 0), ("long", 1)):
            r = _res(); t = r[where[0]] if len(where) == 1 else r[where[0]][where[1]]; t["policies"]["scope_m7"][field] = ok_v; assert _v(r)[4]
            t["policies"]["scope_m7"][field] = bad_v; assert _v(r)[4] is False and sum(_v(r)) == 9, (field, where)
    r = _res(); r["million"]["policies"]["gate"]["max_edges"] = 5000; r["million"]["policies"]["repair_m6"]["max_nodes"] = 500; assert _v(r)[4]               # only scope_m7 is held to the limits
    ramp = lambda last, base=10.0: [base] * 9 + [last]
    for i in (0, 1):
        r = _res(); r["long"][i]["policies"]["scope_m7"] = _pol(dist=ramp(15.0)); assert _v(r)[5] and _v(r)[6]
        r["long"][i]["policies"]["scope_m7"] = _pol(dist=ramp(15.01)); assert {k for k, x in enumerate(_v(r)) if not x} == {5, 9}      # line 10 compares with scope_m7's last tenth, which grew
        r = _res(); r["long"][i]["policies"]["scope_m7"] = _pol(us=ramp(45.0, 30.0)); assert _v(r)[6]
        r["long"][i]["policies"]["scope_m7"] = _pol(us=ramp(45.1, 30.0)); assert _v(r)[6] is False and _v(r)[5] and sum(_v(r)) == 9
    r = _res(); r["million"]["policies"]["scope_m7"] = _pol(us=ramp(1000.0, 30.0)); assert _v(r)[6]                                   # the flatness lines look at the two very long episodes only
    r = _res(); r["long"][0]["policies"]["scope_m7"] = _pol(dist=[20.0] + [10.0] * 9); assert _v(r)[5] is True                  # first tenth higher: not growing
    m = lambda us: dict(million=_run({"gate": _pol(), "repair_m6": _pol(), "scope_m7": _pol(us=us)}))
    r = _res(); r["million"] = m(100.0)["million"]; assert _v(r)[7]
    r["million"] = m(100.1)["million"]; assert _v(r)[7] is False and sum(_v(r)) == 9
    r = _res(); r["million"]["policies"]["scope_m7"]["ns"] = [0] * 10; assert _v(r)[7]
    r = _res(); r["million"]["peak_rss_mb"] = dict(after_10_percent=20.0, end=25.0); assert _v(r)[8]
    r["million"]["peak_rss_mb"] = dict(after_10_percent=20.0, end=25.01); assert _v(r)[8] is False and sum(_v(r)) == 9
    r = _res(); r["long"][0]["peak_rss_mb"] = dict(after_10_percent=1.0, end=900.0); assert _v(r)[8]                            # memory is read on the million-step run
    for i in (0, 1):
        r = _res(); r["long"][i]["policies"]["repair_m6"] = _pol(dist=30.0); assert _v(r)[9]
        r["long"][i]["policies"]["repair_m6"] = _pol(dist=29.9); assert _v(r)[9] is False and sum(_v(r)) == 9
        r = _res(); r["long"][i]["policies"]["gate"] = _pol(dist=100.0); assert _v(r)[9]
        r["long"][i]["policies"]["gate"] = _pol(dist=99.9); assert _v(r)[9] is False and sum(_v(r)) == 9
    r = _res(); r["long"][0]["policies"]["gate"] = _pol(dist=[1.0] * 9 + [500.0]); r["long"][0]["policies"]["repair_m6"] = _pol(dist=[1.0] * 9 + [500.0]); assert _v(r)[9]          # the last tenth is what counts


def test_the_printed_lines_are_the_registered_ones_and_the_tokens_are_arithmetic():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): E.print_verdict_endurance(E.verdict_endurance(_res()))
    out = buf.getvalue(); assert out.count("yes [") == 10 and "NO " not in out
    for tag, text in E.LINES_C: assert f"[{tag}] {text}" in out
    r = _res(); r["blind"]["policies"]["blind"]["corrupted"] = 0
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): E.print_verdict_endurance(E.verdict_endurance(r))
    assert buf.getvalue().count("NO  [") == 1 and "NO  [instrument]" in buf.getvalue()
    t = E.tokens_equivalent(_res(), 323.2); assert t["steps"] == 1000 + 1000 + 250000 * 2 and abs(t["tokens"] - 502000 * 323.2) < 1e-6 and t["million_run_steps"] == 1000
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): E.print_endurance(_res(), dict(per_step_state_gemini=323.2))
    assert "nothing was sent to a model" in buf.getvalue() and "million-step run" in buf.getvalue()


# ---------------------------------------------------------------- 6 the page and the module
def test_the_page_and_the_module_agree_and_the_predictions_are_locked():
    text = open(PAGE).read(); assert text.count(RESULT_MARK) <= 1; head = text[:text.index(RESULT_MARK)] if RESULT_MARK in text else text
    assert _h(head) == PAGE_SHA256, "the predictions and reading rules must not change after the run; later notes go below them"
    for i, (tag, line) in enumerate(E.LINES_C, 1): assert f"\n{i}. [{tag}] {line}\n" in head, i
    assert len(E.LINES_C) == 10 and "Ten lines" in head and "Overall: that all ten lines" in head
    for k in E.FAULTS: assert k in head, k
    assert "15 kinds" in head and head.count("15 kinds") == 3
    for needle in (str(E.SEED_CONTROL), str(E.SEED_C), str(E.LONG_SEEDS[0]), str(E.LONG_SEEDS[1]), f"{E.SEED_C * 1000} to {E.SEED_C * 1000 + E.EPISODES_C - 1}", f"{E.LONG_SEEDS[0] * 1000} and {E.LONG_SEEDS[1] * 1000}",
                   "20 episodes of 2,000 steps", "500 episodes of 2,000 steps", "250,000 steps", "first 50 of those episodes", "probability 0.5", "1,000 random items", "40 digits", "at most 1.5 times", "100 microseconds", "1.25 times",
                   "30 services or 60 edges", "at least 3 times", "at least 10 times", "50 percent"):
        assert needle in head, needle
    assert (E.EPISODES_CONTROL, E.STEPS_CONTROL, E.EPISODES_C, E.STEPS_C, E.STEPS_L, E.BLIND_EPISODES, E.HUGE_ITEMS, E.P_FAULT, E.BLOCKS, E.LIMIT_NODES, E.LIMIT_EDGES) == (20, 2000, 500, 2000, 250000, 50, 1000, 0.5, 10, 30, 60)
    assert E.GATED == ("gate", "repair_m6", "scope_m7") and E.POLICIES_C == ("blind",) + E.GATED and (E.SEED_CONTROL, E.SEED_C, E.LONG_SEEDS) == (20261011, 20261012, (20261013, 20261014))
    assert "axi_endurance_cpu.json" in head and E.RESULT_FILE == "axi_endurance_cpu.json" and "python3 -m axi.experiments.endurance" in head
    for ok_p in ("(80 %)", "(95 %)", "(99 %)", "(98 %)", "(90 %)", "(85 %)"): assert ok_p in head
    assert "about **35 %**" in head and "about **75 %**" in head and head.count("(Part C written BEFORE its run") == 1


# ---------------------------------------------------------------- 7 Part A
def _hist_rows(a, b, lengths):
    return [dict(seed=i, steps=L, policies={"repair_m6": [dict(prompt_tokens=a + b * (t + 1), tokens=7) for t in range(L)]}) for i, L in enumerate(lengths)]


def test_the_token_bill_formulas_are_exact_on_hand_built_numbers():
    h = E.history_growth(_hist_rows(100, 50, [10, 8, 12])); assert h["steps"] == 8 and h["lengths"] == [10, 8, 12] and abs(h["intercept"] - 100) < 1e-9 and abs(h["slope"] - 50) < 1e-9
    assert h["first"] == 150 and h["last"] == 500 and len(h["per_step"]) == 8 and h["episodes"] == 3
    for n in range(1, 25): assert abs(E.history_total(n, 100, 50) - sum(100 + 50 * t for t in range(1, n + 1))) < 1e-9, n
    for w in (200, 5500, 32768, 10 ** 6):
        t = E.steps_until(w, 100, 50); assert 100 + 50 * t > w and 100 + 50 * (t - 1) <= w, w
    assert E.WINDOWS["1,000,000"] == 10 ** 6 and E.WINDOWS["2,000,000"] == 2 * 10 ** 6 and E.EXTENSION_STEPS == (100, 300, 1000, 3000, 10000)
    st = E.state_tokens([dict(policies={"scope_m7": [dict(prompt_tokens=300, tokens=10), dict(prompt_tokens=400, tokens=20)]})], "scope_m7"); assert st == dict(steps=2, prompt=350.0, output=15.0)


def test_the_token_bill_reads_the_saved_files_and_labels_the_extension():
    b = E.token_bill()
    assert b["measured_steps"] == sum(p["steps"] for p in b["pairs"]) and len(b["pairs"]) == 4 and all(p["state"] < p["history"] for p in b["pairs"])
    assert abs(b["measured_saving"] - (1 - b["measured_state_tokens"] / b["measured_history_tokens"])) < 1e-12 and [e["steps"] for e in b["extension"]] == [100, 300, 1000, 3000, 10000]
    assert all(e["saving"] > b["measured_saving"] for e in b["extension"]) and b["extension"][0]["history_tokens"] < b["extension"][-1]["history_tokens"]
    h = b["history"]; assert abs(b["extension"][2]["history_tokens"] - (E.history_total(1000, h["intercept"], h["slope"]) + 1000 * b["state_qwen"]["output"])) < 1e-6
    w = b["windows"]; assert w["32,768"] < w["128,000"] < w["1,000,000"] < w["2,000,000"] and abs(b["state_steps_for"]["2,000,000 tokens"] - 2 * b["state_steps_for"]["1,000,000 tokens"]) < 1e-9


# ---------------------------------------------------------------- 8 Part B
def _built_rows(policies=("blind", "repair_m6", "scope_m7"), seeds=(0, 1, 2), steps=25, mix="scope", damage_at=None):
    """Rows as the long runs save them, made by running the stand-in policies on the ideal reply (and on an empty reply at the (seed, step) pairs in `damage_at`)."""
    rows = []
    for sd in seeds:
        g0, script, oracle = oracle_script(sd, steps, mix); pol = {}
        for p in policies:
            live = copy_graph(g0); log = []
            for t, st in enumerate(script):
                text = "" if damage_at and (sd, t) in damage_at else st["ideal"]; after = next_state(p, live, strict_parse(text), st["instruction"])
                log.append(dict(t=t + 1, kind=st["kind"], text=text, tokens=10, prompt_tokens=100, exact=same_graph(after, oracle[t + 1]))); live = after
            pol[p] = log
        rows.append(dict(seed=sd, steps=len(script), policies=pol))
    return rows


def test_the_replay_reproduces_a_clean_run_and_finds_a_doctored_flag():
    rows = _built_rows()
    for p in ("blind", "repair_m6", "scope_m7"):
        c = E.replay_policy(rows, p, "scope"); n = sum(r["steps"] for r in rows)
        assert c["steps"] == n and c["mismatch"] == 0 and c["unparsed"] == 0 and c["start_off"] == 0 and c["start_same"] == n and c["wasted"] == 0 and c["tokens"] == 110 * n
        assert c["applicable_same"] == n and c["wrong_same"] == 0 and c["inapplicable_same"] + c["inapplicable_off"] == 0
    rows[1]["policies"]["blind"][3]["exact"] = False; assert E.replay_policy(rows, "blind", "scope")["mismatch"] == 1
    rows[1]["policies"]["blind"][3]["exact"] = True; rows[0]["policies"]["blind"][0]["exact"] = False; rows[2]["policies"]["blind"][9]["exact"] = False; assert E.replay_policy(rows, "blind", "scope")["mismatch"] == 2


def test_the_replay_splits_a_wrong_step_into_persistence_inapplicable_requests_and_fresh_errors():
    clean = _built_rows(seeds=range(12), steps=30); total = {"off": 0, "inap": 0, "wrong_off": 0, "wasted": 0, "unparsed": 0}
    damage = {(sd, 2) for sd in range(12)}                                                  # the third reply of every episode is unreadable and is never retried
    rows = _built_rows(seeds=range(12), steps=30, damage_at=damage)
    for p in ("repair_m6", "scope_m7", "blind"):
        c = E.replay_policy(rows, p, "scope"); n = c["steps"]
        assert c["mismatch"] == 0 and c["unparsed"] == 12 and c["wrong_same"] == 12 and c["applicable_same"] + c["inapplicable_same"] == c["start_same"] and c["applicable_off"] + c["inapplicable_off"] == c["start_off"]
        assert c["start_same"] + c["start_off"] == n == c["applicable_same"] + c["applicable_off"] + c["inapplicable_same"] + c["inapplicable_off"]
        w, t = wasted_tokens(rows, p); assert (c["wasted"], c["tokens"]) == (w, t) and 0 < w < t
        total["off"] += c["start_off"]; total["inap"] += c["inapplicable_off"]; total["wrong_off"] += c["wrong_off"]; total["wasted"] += c["wasted"]
    assert total["off"] > 0 and total["wasted"] > 0 and total["inap"] > 0, total                    # the unread reply leaves later requests that cannot be applied as written
    c = E.replay_policy(clean, "scope_m7", "scope"); assert c["start_off"] == 0 and c["wasted"] == 0


def test_the_cascade_report_covers_all_saved_runs_and_every_replay_matches_its_saved_flags():
    rep = E.cascade_report()
    assert len(E.SAVED_RUNS) == 8 and set(rep) == {n for _, n in E.SAVED_RUNS} and sum(len(d["policies"]) for d in rep.values()) == 22
    for name, d in rep.items():
        for p, c in d["policies"].items():
            assert c["mismatch"] == 0, (name, p)
            assert c["steps"] == c["start_same"] + c["start_off"] == c["applicable_same"] + c["applicable_off"] + c["inapplicable_same"] + c["inapplicable_off"], (name, p)
            w, t = wasted_tokens(__import__("json").load(open(os.path.join(E.RESULTS, name + ".json")))["rows"], p); assert (c["wasted"], c["tokens"]) == (w, t), (name, p)


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn): fn(); print("PASS", name); n += 1
    print(n, "tests passed")
