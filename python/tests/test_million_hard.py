"""Implements concept-map item 95 (the hard million-token test on Gemini 3.8 Flash; rows 3, 9, 27): the INSTRUMENT of docs/MILLION_TOKEN_TEST_HARD.md, written before the run. It checks the instrument, not a result:
  1. the dense request stream (`million_hard.HardStream`): deterministic, the registered kinds and weights, inside 24 services and 48 edges and not ending early for 4,000 steps on both registered seeds, and of the sizes
     the page says it has; the hub delete names 4 to 8 dependants and nothing else changes in the earlier stream;
  2. the re-ask: a refused reply is asked again up to twice with a different note each time, every attempt is billed (an attempt that came from the saved reply file is not), the step is stored with the first
     reply, the other attempts and the final one, the replay checks it and refuses a doctored one, a run without re-asks is unchanged;
  3. the dollar guard in `api_llm.DiskCachedLLM`: it refuses the next call once the saved answers cost the budget at the typed-in prices, an answer already saved is still free, raising the budget carries on;
  4. every reading-rule line of the hard test flips exactly at its boundary, the headline is called passed only when all nine headline lines hold, and the reports (new offsets by cause, the re-asks, the control arm);
  5. the page and the module agree (the fourteen lines, the numbers, seeds, kinds, settings) and the page's text above its Result is byte for byte what was written before the run (a hash); the notebook never turns
     billing on, holds no key, and stops by itself.
What it does NOT show: anything about a real model's replies, or the result of the run."""
import contextlib, hashlib, io, json, os, re, sys, tempfile
sys.path.insert(0, os.path.dirname(__file__))
import test_million as TM
from axi.engine.gate import Graph
from axi.experiments import api_llm as A
from axi.experiments import million as M
from axi.experiments import million_hard as H
from axi.experiments.agent_loop import copy_graph, same_graph, strict_parse
from axi.experiments.endurance import Stream

HERE = os.path.dirname(__file__)
PAGE = os.path.join(HERE, "..", "..", "docs", "MILLION_TOKEN_TEST_HARD.md")
PAGE_SHA256 = "9f786621d81c62eb717cdf518a3e4aca1657c2aae09266030f5145f20ce00b59"          # filled in when the page was written, before any run
NB = os.path.join(HERE, "..", "colab", "axi_million_hard.ipynb")
RESULT_MARK = TM.RESULT_MARK
_h = TM._h
tmp = TM.tmp
REFUSED = "DEL[1,2:7]"       # the slip of the saved Flash Lite runs: the id with a number after it


class HEcho(TM.Echo):
    """TM.Echo over the hard stream: it writes the ideal reply of each request (or what `tweak(step, kind, ideal)` makes of it)."""
    def __init__(self, seed, skip=0, tweak=None):
        self.s, self.tweak, self.calls, self.seen, self.last_prompt_tokens, self.n = H.HardStream(seed), tweak, 0, [], 100, 0
        for _ in range(skip): self.s.next(); self.n += 1


class Asker(HEcho):
    """Answers a first attempt with a refused reply on the steps in `bad` (step -> how many attempts are refused before the good one); a re-ask is recognised by its note and does not advance the stream."""
    def __init__(self, seed, bad=None, cached=(), **kw):
        super().__init__(seed, **kw); self.bad, self.cur, self.cached, self.last_think_tokens, self.from_disk, self.attempts = dict(bad or {}), None, set(cached), 5, False, []

    def generate(self, messages, constrained, max_new_tokens=400):
        note = next((i + 1 for i, n in enumerate(M.REASK_NOTES) if messages[-1]["content"].endswith(n)), 0)
        if note == 0: self.cur, _ = self.s.next()[0], None; self.n += 1
        self.calls += 1; self.seen.append(messages); self.attempts.append((self.n, note)); self.last_prompt_tokens = (len(messages[0]["content"]) + len(messages[1]["content"])) // 3
        self.from_disk = (self.n, note) in self.cached
        return (REFUSED if note < self.bad.get(self.n, 0) else self.cur["ideal"]), 11, 0.5


def hrun(seed=5, steps=60, policy="scope_m7", llm=None, path=None, **kw):
    path = path or tmp(); llm = llm or HEcho(seed)
    return M.run_million(llm, seed, steps, policy, path, meta=dict(model="stand-in"), stream="HardStream", **kw), llm, path


def forgetful_hub(n, kind, ideal):
    """What a model does that forgets the dependants: only the service is deleted (for the three kinds that have dependants)."""
    items = ideal.split(" ")
    if kind in ("delete_dep", "delete_hub"): return items[0]
    if kind == "delete_pair": return items[0] + " " + items[-1]
    return ideal


# ---------------------------------------------------------------- 1 the stream
def test_the_hard_stream_has_the_registered_kinds_caps_and_picker_and_leaves_the_earlier_stream_alone():
    assert [H.KINDS_HARD.count(k) for k in ("add_node", "add_edge", "delete_dep", "delete_hub", "delete_leaf", "delete_edge", "delete_pair")] == [7, 20, 1, 3, 1, 1, 1] and len(H.KINDS_HARD) == 34
    assert (H.MAX_NODES_H, H.MAX_EDGES_H, H.HUB_MIN, H.HUB_MAX) == (24, 48, 4, 8) and H.HardStream.max_nodes == 24 and H.HardStream.max_edges == 48
    assert M.LiveStream.max_nodes == 14 and M.LiveStream.max_edges == 26 and Stream.max_nodes == 14 and M.STREAMS["HardStream"] is H.HardStream and M.STREAMS["LiveStream"] is M.LiveStream
    a, b = H.HardStream(11), H.HardStream(11); xs = [a.next() for _ in range(300)]; ys = [b.next() for _ in range(300)]
    assert all(x[0] == y[0] and same_graph(x[1], y[1]) for x, y in zip(xs, ys)) and all(x is not None for x in xs)
    assert isinstance(a, Stream) and a.kinds_all == H.KINDS_HARD and a.kinds_all is not H.KINDS_HARD and a.picker is H.pick_hard_task and same_graph(a.start, Stream(11, "steady").start)
    live = M.LiveStream(11); zs = [live.next() for _ in range(50)]; assert [z[0] for z in zs] != [x[0] for x in xs[:50]]
    assert M.make_stream("HardStream", 3).kinds_all == H.KINDS_HARD and M.make_stream("LiveStream", 3).kinds_all == M.KINDS_LIVE
    try: M.make_stream("nope", 1); assert False
    except ValueError as e: assert "unknown request stream" in str(e)


def test_both_hard_seeds_run_4000_steps_inside_the_caps_and_have_the_sizes_the_page_states():
    assert H.HARD_SEEDS == (20261031, 20261032) and H.HARD_STEPS == 4000
    nodes, edges, hub_items, share, risky_items = [], [], [], [], []
    for seed in H.HARD_SEEDS:
        S = H.HardStream(seed); sn, se, kinds, items = [], [], {}, {}
        for i in range(H.HARD_STEPS):
            out = S.next(); assert out is not None, (seed, i)
            st, before = out; sn.append(len(before.nodes)); se.append(len(before.edges)); kinds[st["kind"]] = kinds.get(st["kind"], 0) + 1; items[st["kind"]] = items.get(st["kind"], 0) + st["ideal"].count("[")
            assert len(S.live.nodes) <= 24 and len(S.live.edges) <= 48 and S.live.is_well_formed(), (seed, i)
        for j in range(10): nodes.append(sum(sn[j * 400:(j + 1) * 400]) / 400); edges.append(sum(se[j * 400:(j + 1) * 400]) / 400)
        assert max(sn) == 24 and max(se) == 48 and set(kinds) == {"add_node", "add_edge", "delete_dep", "delete_hub", "delete_leaf", "delete_edge", "delete_pair"}
        hub_items.append(items["delete_hub"] / kinds["delete_hub"]); share.append(100 * (kinds["delete_dep"] + kinds["delete_hub"] + kinds["delete_pair"]) / H.HARD_STEPS)
        risky_items.append((items["delete_dep"] + items["delete_hub"] + items["delete_pair"]) / H.HARD_STEPS)
        assert 300 <= kinds["delete_hub"] <= 360 and 55 <= 100 * kinds["add_edge"] / H.HARD_STEPS <= 65
    assert 17.0 <= min(nodes) and max(nodes) <= 22.3 and 18.3 <= min(edges) and max(edges) <= 37.1, (min(nodes), max(nodes), min(edges), max(edges))            # "17 to 22 services and 18 to 37 edges" on the page
    assert all(5.8 <= x <= 6.0 for x in hub_items) and all(14.9 <= x <= 15.5 for x in share) and all(0.71 <= x <= 0.73 for x in risky_items), (hub_items, share, risky_items)


def _graph(edges, nodes=12):
    g = Graph()
    cells = [(r, c) for r in range(3) for c in range(5)][:nodes]
    for i, n in enumerate(cells): g.add_node(n, i % 9 + 1)
    for k, (a, b) in enumerate(edges): g.add_edge(cells[a], cells[b], "dep", k % 9 + 1)
    return g, cells


def test_a_hub_delete_names_every_one_of_four_to_eight_dependants_and_nothing_else_is_a_hub():
    import random
    for degree, hub in ((3, False), (4, True), (8, True), (9, False)):
        g, cells = _graph([(0, i) for i in range(1, degree + 1)])
        rng = random.Random(1); t = H.pick_hard_task(rng, g, "delete_hub")
        if hub:
            assert t is not None and t.kind == "delete_hub" and t.instruction == "Delete the service at (0,0)." and t.ideal.count("DEL[(0,0)->") == degree and t.ideal.startswith("DEL[0,0] ")
            assert same_graph(t.graph, g) and t.expected.is_well_formed() and (0, 0) not in t.expected.nodes and len(t.expected.edges) == 0
        else:
            others = [n for n in g.nodes if H.HUB_MIN <= len(g.incident(n)) <= H.HUB_MAX]
            assert (t is None) == (not others), degree                                  # the star's centre is too small or too big; its leaves have one edge: no hub
    g, _ = _graph([(0, 1), (0, 2)]); assert H.pick_hard_task(random.Random(2), g, "delete_hub") is None
    g, _ = _graph([(0, i) for i in range(1, 6)])
    from axi.experiments.scope_tasks import pick_scope_task
    for kind in ("delete_dep", "delete_leaf", "delete_edge", "delete_pair", "add_node", "add_edge"):
        a = H.pick_hard_task(random.Random(7), g, kind); b = pick_scope_task(random.Random(7), g, kind)
        assert (a is None) == (b is None) and (a is None or (a.ideal == b.ideal and a.instruction == b.instruction)), kind                              # the other kinds are the scope test's, unchanged


def test_the_degree_threshold_also_reaches_the_words_policy_and_a_fresh_process_finds_the_hard_stream():
    import random, subprocess
    from axi.engine.grounding import NameIndex
    from axi.experiments.cascade import next_state
    g, cells = _graph([(0, i) for i in range(1, 7)]); t = H.pick_hard_task(random.Random(1), g, "delete_hub"); assert t is not None
    d = strict_parse(t.ideal.split(" ")[0]); assert d is not None                                                                          # only the service itself: the dependants are forgotten
    ix = NameIndex({c: n for c, n in zip(cells, "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima".split())})
    wide = next_state("scope_words", g, d, "Delete the service alpha.", names=ix, degree_threshold=8); narrow = next_state("scope_words", g, d, "Delete the service alpha.", names=ix)
    assert same_graph(wide, t.expected) and not same_graph(narrow, t.expected) and cells[0] in narrow.nodes                                       # 6 dependants: deleted with them at 8, kept at the default 4
    root = os.path.dirname(HERE)
    code = f"import sys; sys.path.insert(0, {root!r}); from axi.experiments.million import make_stream; print(type(make_stream('HardStream', 5)).__name__)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120); assert out.returncode == 0 and out.stdout.strip() == "HardStream", out.stderr[-300:]      # the notebook may import million first


def test_a_clean_hard_run_is_exact_and_a_model_that_forgets_the_dependants_is_mended_only_when_the_repair_deletes_up_to_eight_of_them():
    assert H.HARD_DEGREE == 8 == H.HUB_MAX
    r, llm, path = hrun(5, 150, degree_threshold=H.HARD_DEGREE)
    assert len(r["rows"]) == 150 and llm.calls == 150 and all(x["exact"] and x["parsed"] and not x["corrupted"] and x["start_same"] and x["applicable"] and x["wrong"] is False for x in r["rows"])
    assert any(x["kind"] == "delete_hub" for x in r["rows"]) and all(x.get("dropped", 0) == 0 for x in r["rows"])                      # hub deletes are admitted whole by the scope filter
    meta = json.loads(open(path).readline())["_meta"]; assert meta["stream"] == "HardStream" and meta["kinds"] == H.KINDS_HARD and "reasks" not in meta and meta["degree_threshold"] == 8
    assert "degree_threshold" not in json.loads(open(hrun(5, 5)[2]).readline())["_meta"]                                               # the default (4) is not written: the files of the first two pages keep their header
    res, rows = {}, {}
    for p in M.CONTRAST_POLICIES + (M.MAIN_POLICY,): res[p] = hrun(21, 400, p, llm=HEcho(21, tweak=forgetful_hub), degree_threshold=8)[0]; rows[p] = res[p]["rows"]
    assert sum(x["corrupted"] for x in rows["blind"]) >= 0.8 * 400 and all(not x["corrupted"] for p in ("gate", "repair_m6", "scope_m7") for x in rows[p])
    mean = lambda p, a, b: sum(x["distance"] for x in rows[p][a:b]) / (b - a)
    assert mean("gate", 200, 400) > 0.5 and mean("blind", 200, 400) > 0.5 and mean("scope_m7", 200, 400) == 0 and mean("repair_m6", 200, 400) == 0 and all(x["exact"] for x in rows["scope_m7"] + rows["repair_m6"])
    assert all(a["kind"] == b["kind"] for a, b in zip(rows["blind"], rows["gate"]))
    kept = hrun(21, 400, "scope_m7", llm=HEcho(21, tweak=forgetful_hub))[0]["rows"]                                                  # the framework default (4): a service with 5 to 8 dependants is KEPT, not deleted
    assert sum(1 for x in kept if not x["exact"]) > 100 and max(x["distance"] for x in kept) > 10 and all(not x["corrupted"] for x in kept)
    odd = hrun(21, 400, "repair_m5", llm=HEcho(21, tweak=forgetful_hub), degree_threshold=8)[0]["rows"]; assert all(x["exact"] for x in odd)         # repair_m5 takes the threshold too
    sc = res["scope_m7"]; M.replay(sc["rows"], 21, "scope_m7", stream="HardStream", degree_threshold=8)                                       # the replay needs the same threshold as the run
    try: M.replay(sc["rows"], 21, "scope_m7", stream="HardStream"); assert False
    except M.ReplayMismatch: pass
    assert M._run_lines(dict(rows=sc["rows"], meta=sc["meta"]))[1] is True and sc["meta"]["degree_threshold"] == 8                              # line 1 replays with the threshold written in the header
    nohead = dict(sc["meta"]); nohead.pop("degree_threshold"); assert M._run_lines(dict(rows=sc["rows"], meta=nohead))[1] is False             # a header without it is read as the default 4, which does not reproduce the rows
    p = tmp(); full = hrun(21, 400, llm=HEcho(21, tweak=forgetful_hub), path=p, degree_threshold=8)[0]; lines = open(p).read().splitlines(); open(p, "w").write("\n".join(lines[:231]) + "\n" + lines[231][:20])      # cut a run that had hub deletes
    assert any(x["kind"] == "delete_hub" for x in full["rows"][:230])
    again = M.run_million(HEcho(21, skip=230, tweak=forgetful_hub), 21, 400, "scope_m7", p, meta=dict(model="stand-in"), stream="HardStream", degree_threshold=8)
    assert [json.dumps(x, sort_keys=True) for x in again["rows"]] == [json.dumps(x, sort_keys=True) for x in full["rows"]]                       # resumed with the replay under the same threshold


# ---------------------------------------------------------------- 2 the re-ask
def test_a_refused_reply_is_asked_again_with_a_different_note_each_time_and_every_attempt_is_stored_and_billed():
    assert strict_parse(REFUSED) is None and len(M.REASK_NOTES) == 2 and M.REASK_NOTES[0] != M.REASK_NOTES[1]
    llm = Asker(5, bad={4: 1, 9: 2}); r, _, path = hrun(5, 20, llm=llm, reasks=2); rows = r["rows"]
    assert llm.calls == 20 + 1 + 2 and llm.attempts.count((4, 1)) == 1 and llm.attempts.count((9, 2)) == 1 and [a for a in llm.attempts if a[0] == 9] == [(9, 0), (9, 1), (9, 2)]
    assert all(x["exact"] for x in rows) and rows[3]["asks"] == 2 and rows[8]["asks"] == 3 and all("asks" not in x for i, x in enumerate(rows) if i not in (3, 8))
    s3, s8 = rows[3], rows[8]; assert s3["first_text"] == REFUSED and len(s3["reasks"]) == 1 and s3["text"] == s3["reasks"][-1]["text"] != REFUSED and strict_parse(s3["text"]) is not None
    assert s8["first_text"] == REFUSED and [x["text"] for x in s8["reasks"]][0] == REFUSED and s8["text"] == s8["reasks"][1]["text"] and s8["parsed"] is True
    base = llm.seen[0][1]["content"]; k = [i for i, a in enumerate(llm.attempts) if a == (4, 0)][0]
    first, second = llm.seen[k][1]["content"], llm.seen[k + 1][1]["content"]; assert second == first + "\n\n" + M.REASK_NOTES[0] and llm.seen[k + 1][0] == llm.seen[k][0]
    k9 = [i for i, a in enumerate(llm.attempts) if a == (9, 0)][0]; assert llm.seen[k9 + 2][1]["content"] == llm.seen[k9][1]["content"] + "\n\n" + M.REASK_NOTES[1]
    assert s3["seconds"] == 1.0 and s8["seconds"] == 1.5 and rows[0]["seconds"] == 0.5                                                      # the seconds of every attempt are the step's
    p = s3["prompt_tokens"]; assert M._billed(s3) == p + 11 + 5 + s3["reasks"][0]["prompt_tokens"] + 11 + 5 and M._billed(rows[0]) == rows[0]["prompt_tokens"] + 16
    s = r["summary"]; assert s["tokens"] == M.tokens_sent(rows) and s["reasked_steps"] == 2 and s["calls"] == 23 and s["output_tokens"] == 11 * 23 and s["think_tokens"] == 5 * 23 and s["cached_steps"] == 0
    assert s["prompt_tokens"] + s["output_tokens"] + s["think_tokens"] == s["tokens"]
    meta = json.loads(open(path).readline())["_meta"]; assert meta["reasks"] == 2
    M.replay(rows, 5, "scope_m7", stream="HardStream", reasks=2)                                                                              # the replay accepts it
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): M.print_million(s, "hard")
    assert "2 steps needed a re-ask" in buf.getvalue()
    plain = hrun(5, 10)[0]["summary"]; buf = io.StringIO()
    with contextlib.redirect_stdout(buf): M.print_million(plain, "plain")
    assert "re-ask" not in buf.getvalue() and plain["reasked_steps"] == 0 and plain["calls"] == 10


def test_the_re_ask_gives_up_after_the_allowed_number_and_the_step_then_ends_refused_with_the_state_unchanged():
    class Never(Asker):
        def generate(self, messages, constrained, max_new_tokens=400):
            note = next((i + 1 for i, n in enumerate(M.REASK_NOTES) if messages[-1]["content"].endswith(n)), 0)
            if note == 0: self.s.next(); self.n += 1
            self.calls += 1; self.last_prompt_tokens = 10; return REFUSED, 4, 0.1
    for k in (0, 1, 2):
        llm = Never(5); r, _, _ = hrun(5, 6, llm=llm, reasks=k)
        assert llm.calls == 6 * (1 + k) and all(not x["parsed"] and x["distance"] >= 0 for x in r["rows"]) and all(x.get("asks", 1) == (1 + k if k else 1) for x in r["rows"])
        assert r["summary"]["refused"] == 6 and all(x["text"] == REFUSED for x in r["rows"]) and r["summary"]["tokens"] == 6 * (1 + k) * 19
        assert (["reasks" in x for x in r["rows"]] == [True] * 6) == (k > 0) and r["rows"][0]["start_same"] and not r["rows"][-1]["start_same"]
        M.replay(r["rows"], 5, "scope_m7", stream="HardStream", reasks=k)
    try: hrun(5, 3, reasks=3); assert False
    except ValueError as e: assert "reasks must be" in str(e)


def test_a_run_without_re_asks_keeps_the_header_and_the_refused_reply_it_always_had():
    llm = Asker(5, bad={3: 1}); p = tmp(); r = M.run_million(llm, 5, 8, "scope_m7", p, meta={}, stream="HardStream")
    assert llm.calls == 8 and not r["rows"][2]["parsed"] and "asks" not in r["rows"][2] and "reasks" not in json.loads(open(p).readline())["_meta"]
    p2 = tmp(); r2 = M.run_million(TM.Echo(5), 5, 8, "scope_m7", p2, meta={})
    assert json.loads(open(p2).readline())["_meta"]["stream"] == "LiveStream" and json.loads(open(p2).readline())["_meta"]["kinds"] == M.KINDS_LIVE


def test_the_replay_refuses_a_doctored_re_ask_and_a_refused_reply_that_was_not_asked_again():
    r, _, _ = hrun(5, 20, llm=Asker(5, bad={4: 1, 9: 2}), reasks=2); good = r["rows"]
    def doctored(mut):
        rows = json.loads(json.dumps(good)); mut(rows); return rows
    cases = {
        "a readable first reply": lambda rows: rows[3].__setitem__("first_text", rows[3]["text"]),
        "no first reply": lambda rows: rows[3].pop("first_text"),
        "asks too many": lambda rows: rows[3].__setitem__("asks", 3),
        "asks too few": lambda rows: rows[8].__setitem__("asks", 2),
        "last re-ask is not the reply used": lambda rows: rows[3]["reasks"][-1].__setitem__("text", REFUSED),
        "a readable middle attempt": lambda rows: rows[8]["reasks"][0].__setitem__("text", rows[8]["text"]),
        "a refused reply left alone": lambda rows: (rows[3].pop("asks"), rows[3].pop("first_text"), rows[3].pop("reasks"), rows[3].__setitem__("text", REFUSED)),
    }
    for name, mut in cases.items():
        try: M.replay(doctored(mut), 5, "scope_m7", stream="HardStream", reasks=2); assert False, name
        except M.ReplayMismatch: pass
    try: M.replay(good, 5, "scope_m7", stream="HardStream", reasks=1); assert False                                                           # three attempts in a run that allowed two
    except M.ReplayMismatch: pass
    rows = doctored(lambda rows: (rows[3].pop("asks"), rows[3].pop("first_text"), rows[3].pop("reasks"))); M.replay(rows, 5, "scope_m7", stream="HardStream", reasks=2)   # a step that was not re-asked and is readable: fine
    r0 = hrun(5, 20, llm=Asker(5, bad={4: 1}))[0]["rows"]; M.replay(r0, 5, "scope_m7", stream="HardStream", reasks=0)                       # a refused reply with re-asks off: fine
    try: M.replay(r0, 5, "scope_m7", stream="HardStream", reasks=2); assert False                                                             # the same file read as a run that re-asked: it was not
    except M.ReplayMismatch: pass
    try: M.replay(r0, 5, "scope_m7", stream="HardStream", reasks=1); assert False                                                             # also when only one re-ask was allowed
    except M.ReplayMismatch: pass
    assert len(M.replay(good, 5, "scope_m7", strict=False, stream="HardStream", reasks=2)[2]) == 0 and M.replay(doctored(cases["asks too many"]), 5, "scope_m7", strict=False, stream="HardStream", reasks=2)[2] == [4]


def test_a_cut_re_asking_run_resumes_to_the_same_rows_and_a_file_of_another_setting_is_refused():
    full, _, p = hrun(5, 30, llm=Asker(5, bad={4: 1, 9: 2, 22: 1}), reasks=2)
    lines = open(p).read().splitlines(); open(p, "w").write("\n".join(lines[:8]) + "\n" + lines[8][:15])                                       # cut in the middle of a line
    llm = Asker(5, bad={4: 1, 9: 2, 22: 1}, skip=7); res = M.run_million(llm, 5, 30, "scope_m7", p, meta=dict(model="stand-in"), stream="HardStream", reasks=2)
    assert [json.dumps(x, sort_keys=True) for x in res["rows"]] == [json.dumps(x, sort_keys=True) for x in full["rows"]]
    for kw in (dict(reasks=1), dict(reasks=0), dict(stream="LiveStream", reasks=2)):
        try: M.run_million(Asker(5), 5, 30, "scope_m7", p, meta=dict(model="stand-in"), **{"stream": "HardStream", **kw}); assert False, kw
        except ValueError: pass


def test_an_attempt_from_the_saved_reply_file_is_not_billed_and_a_step_is_cached_only_if_all_its_calls_were():
    llm = Asker(5, bad={4: 1}, cached=((4, 0),)); r, _, _ = hrun(5, 10, llm=llm, reasks=2); rows = r["rows"]; s4 = rows[3]
    assert s4["cached"] is True and s4["reasks"][0]["cached"] is False and M._billed(s4) == s4["reasks"][0]["prompt_tokens"] + 11 + 5 and r["summary"]["cached_steps"] == 0
    both = Asker(5, bad={4: 1}, cached=((4, 0), (4, 1))); r2, _, _ = hrun(5, 10, llm=both, reasks=2)
    assert M._billed(r2["rows"][3]) == 0 and r2["summary"]["cached_steps"] == 1 and r2["summary"]["calls"] == 11
    sm = r2["summary"]; live = [x for i, x in enumerate(r2["rows"]) if i != 3]; assert sm["prompt_tokens"] == sum(x["prompt_tokens"] for x in live) and sm["output_tokens"] == 11 * 9
    assert r2["rows"][3]["seconds"] > 0 and abs(sm["live_seconds"] - (sm["seconds"] - r2["rows"][3]["seconds"])) < 1e-9 < sm["seconds"] - sm["live_seconds"]     # the seconds of a step answered from the saved file are not live time
    only_second = Asker(5, bad={4: 1}, cached=((4, 1),)); r3, _, _ = hrun(5, 10, llm=only_second, reasks=2)
    assert r3["rows"][3]["cached"] is False and r3["rows"][3]["reasks"][0]["cached"] is True and M._billed(r3["rows"][3]) == r3["rows"][3]["prompt_tokens"] + 16 and r3["summary"]["cached_steps"] == 0


def test_a_token_cap_in_the_middle_of_a_re_asking_run_stops_at_a_step_boundary_and_the_run_carries_on():
    full, _, _ = hrun(5, 30, llm=Asker(5, bad={4: 1, 9: 2}), reasks=2); per = [M._billed(x) for x in full["rows"]]; cap = sum(per[:12])
    p = tmp(); a = Asker(5, bad={4: 1, 9: 2}); r = hrun(5, 30, llm=a, path=p, reasks=2, max_tokens=cap)[0]
    assert len(r["rows"]) == 12 and r["stopped"] == "token_cap" and a.calls == 12 + 3                                                           # 12 steps, three of the calls were re-asks
    b = Asker(5, bad={4: 1, 9: 2}, skip=12); r2 = hrun(5, 30, llm=b, path=p, reasks=2)[0]; assert len(r2["rows"]) == 30 and r2["stopped"] is None and r2["summary"] == full["summary"]


# ---------------------------------------------------------------- 2b the scripted start of the 20-step check
def test_a_scripted_start_answers_the_first_steps_without_the_model_and_such_a_file_is_not_the_registered_run():
    llm = HEcho(7, skip=30); r, _, path = hrun(7, 40, llm=llm, scripted=30, reasks=2, degree_threshold=8); rows = r["rows"]
    assert llm.calls == 10 and len(rows) == 40 and all(x["exact"] and x["parsed"] for x in rows)
    assert all(x["cached"] and x["tokens"] == 0 and x["think"] == 0 and x["prompt_tokens"] == 0 and x["seconds"] == 0.0 and "asks" not in x for x in rows[:30]) and all(not x["cached"] and x["tokens"] == 11 for x in rows[30:])
    ideal = H.HardStream(7); assert all(x["text"] == ideal.next()[0]["ideal"] for x in rows[:30])                                  # the scripted reply is the oracle's own
    s = r["summary"]; assert s["cached_steps"] == 30 and s["tokens"] == M.tokens_sent(rows) == sum(x["prompt_tokens"] + 11 for x in rows[30:]) and s["calls"] == 40 and abs(s["live_seconds"] - 5.0) < 1e-9 and abs(s["seconds"] - 5.0) < 1e-9
    meta = json.loads(open(path).readline())["_meta"]; assert meta["scripted"] == 30 and "scripted" not in json.loads(open(hrun(7, 5)[2]).readline())["_meta"]
    M.replay(rows, 7, "scope_m7", stream="HardStream", reasks=2, degree_threshold=8)
    full = dict(stream="HardStream", policy="scope_m7", steps=4000, seed=H.HARD_SEEDS[0], reasks=2, degree_threshold=8)
    for key, val in (("stream", "LiveStream"), ("seed", 5), ("seed", H.HARD_SEEDS[1] + 1), ("policy", "gate"), ("steps", 3999), ("reasks", 1), ("degree_threshold", 4)): assert not H.registered(dict(full, **{key: val}), "scope_m7", 4000), (key, val)
    assert H.registered(dict(full, seed=H.HARD_SEEDS[1]), "scope_m7", 4000)
    assert H.registered(full, "scope_m7", 4000) and not H.registered(dict(full, scripted=30), "scope_m7", 4000) and H.registered(dict(full, scripted=0), "scope_m7", 4000)
    lines = open(path).read().splitlines(); open(path, "w").write("\n".join(lines[:20]) + "\n" + lines[20][:9])                                       # cut inside the scripted part
    again = M.run_million(HEcho(7, skip=30), 7, 40, "scope_m7", path, meta=dict(model="stand-in"), stream="HardStream", reasks=2, degree_threshold=8, scripted=30)
    assert [json.dumps(x, sort_keys=True) for x in again["rows"]] == [json.dumps(x, sort_keys=True) for x in rows]
    whole = HEcho(7); rw = hrun(7, 5, llm=whole, scripted=5)[0]; assert whole.calls == 0 and len(rw["rows"]) == 5 and all(x["cached"] and x["exact"] for x in rw["rows"])      # all the steps scripted is allowed
    for bad in (-1, 41):
        try: hrun(7, 40, scripted=bad); assert False, bad
        except ValueError as e: assert "scripted must be" in str(e)
    try: M.run_million(HEcho(7, skip=30), 7, 40, "scope_m7", path, meta=dict(model="stand-in"), stream="HardStream", reasks=2, degree_threshold=8, scripted=10); assert False       # another setting: another file
    except ValueError: pass
    sm = M.smoke_report(s, 4000, 1.0, 4.0, lowest="low"); assert sm["steps"] == 10 and abs(sm["seconds_per_step"] - 0.5) < 1e-9 and abs(sm["tokens_per_step"] - s["tokens"] / 10) < 1e-9          # the 30 scripted steps do not dilute the check
    plain = hrun(7, 10)[0]["summary"]; assert plain["live_seconds"] == plain["seconds"] and abs(M.smoke_report(plain, 4000)["seconds_per_step"] - 0.5) < 1e-9


# ---------------------------------------------------------------- 3 the dollar guard
class Bill:
    """A backend that answers every call with the same text and the token counts it is told."""
    def __init__(self, prompt=1000, out=10, think=40): self.calls, self.p, self.o, self.t = 0, prompt, out, think
    def call(self, messages): self.calls += 1; return dict(text="ADD[0,0:1]", tokens=self.o, think_tokens=self.t, prompt_tokens=self.p, finish="STOP")


def _msgs(i): return [{"role": "system", "content": "s"}, {"role": "user", "content": f"u{i}"}]


def test_the_dollar_guard_refuses_the_next_call_once_the_saved_answers_cost_the_budget():
    b = Bill(); llm = A.DiskCachedLLM(b, tmp(), dict(m=1), budget_usd=0.01, price_in=1.0, price_out=4.0)
    assert llm.spent_usd() == 0.0
    per = (1000 * 1.0 + 50 * 4.0) / 1e6; assert abs(per - 0.0012) < 1e-12
    for i in range(9): llm.generate(_msgs(i), True)                                                  # 9 x 0.0012 = 0.0108 > 0.01 happens after the 9th call; the 9th was still allowed (0.0096 before it)
    assert b.calls == 9 and abs(llm.spent_usd() - 9 * per) < 1e-12
    try: llm.generate(_msgs(99), True); assert False
    except A.BudgetStop as e:
        assert isinstance(e, A.QuotaStop) and "Budget reached" in str(e) and "$0.01" in str(e) and "raise BUDGET_USD" in str(e) and "Everything finished is saved" in str(e)
    assert b.calls == 9
    text, n, sec = llm.generate(_msgs(3), True); assert text == "ADD[0,0:1]" and llm.from_disk is True and b.calls == 9                          # an answer already paid for is still free
    llm.budget_usd = 0.02; llm.generate(_msgs(99), True); assert b.calls == 10 and llm.from_disk is False                                          # raising the budget carries on
    again = A.DiskCachedLLM(Bill(), llm.path, dict(m=1), budget_usd=0.011, price_in=1.0, price_out=4.0); assert abs(again.spent_usd() - 10 * per) < 1e-12      # the spend is read back from the file
    try: again.generate(_msgs(100), True); assert False
    except A.BudgetStop: pass


def test_the_dollar_guard_needs_both_prices_counts_thinking_as_output_and_is_off_without_a_budget():
    for kw in (dict(budget_usd=5.0), dict(budget_usd=5.0, price_in=1.0), dict(budget_usd=5.0, price_out=1.0)):
        try: A.DiskCachedLLM(Bill(), tmp(), dict(m=1), **kw); assert False, kw
        except ValueError as e: assert "both prices" in str(e)
    free = A.DiskCachedLLM(Bill(), tmp(), dict(m=1)); free.generate(_msgs(1), True); assert free.spent_usd() is None and free.budget_usd is None
    free.generate(_msgs(2), True); assert free.backend.calls == 2                                                                                  # no budget, no stop
    for one in (dict(price_in=1.0), dict(price_out=1.0)):
        half = A.DiskCachedLLM(Bill(), tmp(), dict(m=1), **one); half.generate(_msgs(1), True); assert half.spent_usd() is None                       # one price alone gives no cost
    think = A.DiskCachedLLM(Bill(prompt=0, out=0, think=1_000_000), tmp(), dict(m=1), price_in=0.0, price_out=3.75); think.generate(_msgs(1), True); assert abs(think.spent_usd() - 3.75) < 1e-9      # thinking tokens are priced as output
    prices = A.DiskCachedLLM(Bill(prompt=1_000_000, out=0, think=0), tmp(), dict(m=1), price_in=0.75, price_out=3.75); prices.generate(_msgs(1), True); assert abs(prices.spent_usd() - 0.75) < 1e-9
    zero = A.DiskCachedLLM(Bill(), tmp(), dict(m=1), budget_usd=0.0, price_in=1.0, price_out=1.0)
    try: zero.generate(_msgs(1), True); assert False                                                                                                # a budget of nothing stops before the first call
    except A.BudgetStop: assert zero.backend.calls == 0


def test_the_dollar_guard_stops_a_run_between_steps_and_the_run_carries_on_when_the_budget_is_raised():
    class HBackend:
        def __init__(self, seed): self.s, self.calls = H.HardStream(seed), 0
        def call(self, messages):
            st, _ = self.s.next(); self.calls += 1
            return dict(text=st["ideal"], tokens=11, think_tokens=4, prompt_tokens=(len(messages[0]["content"]) + len(messages[1]["content"])) // 3, finish="STOP")
    cache = tmp(); b = HBackend(5); llm = A.DiskCachedLLM(b, cache, dict(m=1), budget_usd=0.005, price_in=1.0, price_out=1.0)
    p = tmp()
    try: M.run_million(llm, 5, 400, "scope_m7", p, meta=dict(model="stand-in"), stream="HardStream"); assert False
    except A.BudgetStop: pass
    done = len(open(p).read().splitlines()) - 1; assert 0 < done < 400 and b.calls == done and llm.spent_usd() >= 0.005
    assert llm.spent_usd() - 0.005 < 0.0015                                                                                                           # overshoot: at most the last call
    llm.budget_usd = 5.0; b2 = HBackend(5)
    for _ in range(done): b2.s.next()
    llm.backend = b2; res = M.run_million(llm, 5, 400, "scope_m7", p, meta=dict(model="stand-in"), stream="HardStream")
    assert len(res["rows"]) == 400 and res["stopped"] is None and all(x["exact"] for x in res["rows"])


# ---------------------------------------------------------------- 4 the reading rules, the headline and the reports
_BASE = {}


def base(i=0):
    """A clean registered run (4,000 steps, seed i of the two, two re-asks allowed, the repair deleting up to 8 dependants) made by the stand-in; a fresh copy each time."""
    if i not in _BASE: _BASE[i] = json.dumps(hrun(H.HARD_SEEDS[i], H.HARD_STEPS, reasks=2, degree_threshold=H.HARD_DEGREE)[0])
    return json.loads(_BASE[i])


def total(run, n):
    """Spread the prompt tokens evenly so that the run has sent exactly n tokens (the other tokens of the steps stay as they were)."""
    rows = run["rows"]; other = M.tokens_sent(rows) - sum(r["prompt_tokens"] for r in rows); want = n - other; q, rem = divmod(want, len(rows)); assert q > 0
    for i, r in enumerate(rows): r["prompt_tokens"] = q + (1 if i < rem else 0)
    assert M.tokens_sent(rows) == n; return run


def ready(i=0):
    run = base(i)
    for r in run["rows"]: r["prompt_tokens"] = 700
    return run


@contextlib.contextmanager
def no_replay():
    """Line 1 reads the saved flags back through a replay; the other lines are tested with that part switched off, so that a doctored flag changes one line only."""
    keep = M.replay; M.replay = lambda *a, **k: (None, None, [])
    try: yield
    finally: M.replay = keep


@contextlib.contextmanager
def broken_replay():
    keep = M.replay
    def boom(*a, **k): raise M.ReplayMismatch("the stream ended before the saved steps did")
    M.replay = boom
    try: yield
    finally: M.replay = keep


def line(n, edit, i=0):
    run = ready(i); edit(run["rows"]); return H._hard_run_lines(run)[n]


def setrows(rows, key, idx, val):
    for j in idx: rows[j][key] = val


def test_a_clean_run_holds_every_line_of_one_run_and_a_file_made_with_other_settings_is_not_the_registered_run():
    run = ready(); a = H._hard_run_lines(run); assert sorted(a) == [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12] and all(a.values()), a
    assert M.wrong_rate(run["rows"]) == (0, H.HARD_STEPS) and run["meta"]["reasks"] == 2 and run["meta"]["degree_threshold"] == 8 and run["meta"]["stream"] == "HardStream"
    with broken_replay(): z = H._hard_run_lines(ready()); assert z[1] is False and all(z[k] for k in z if k != 1)                                    # a replay that cannot follow the saved steps: not the saved run, and only line 1 says so
    for key, val in (("reasks", 1), ("reasks", 0), ("degree_threshold", 4), ("stream", "LiveStream"), ("policy", "gate"), ("steps", 3999), ("steps", 4001), ("seed", 5)):
        r = ready(); r["meta"][key] = val; assert H._hard_run_lines(r)[1] is False, (key, val)
    for key in ("reasks", "degree_threshold"):
        r = ready(); del r["meta"][key]; assert H._hard_run_lines(r)[1] is False, key                               # the default is not the registered setting
    r = ready(); r["rows"].pop(); assert H._hard_run_lines(r)[1] is False                                           # a step missing
    r = ready(); r["rows"][10]["t"] = 99; assert H._hard_run_lines(r)[1] is False                                   # steps out of order
    r = ready(); r["rows"][7]["exact"] = False; assert H._hard_run_lines(r)[1] is False and H._hard_run_lines(r)[7] is True               # a doctored flag: the replay disagrees (line 7 alone would not notice)
    r = ready(); r["rows"][7]["asks"] = 2; assert H._hard_run_lines(r)[1] is False                                  # a doctored re-ask
    r = ready(); r["rows"][7]["text"] = "DEL[0,0]"; assert H._hard_run_lines(r)[1] is False                         # a doctored reply
    for i in (0, 1): assert H._hard_run_lines(ready(i))[1] is True
    assert H.registered(run["meta"], "scope_m7", 4000) and not H.registered(run["meta"], "blind", 4000)
    r = ready(); r["meta"]["stream"] = "LiveStream"; r["meta"]["kinds"] = M.KINDS_LIVE; assert H._hard_run_lines(r)[1] is False


def test_lines_two_and_three_flip_at_a_single_corrupted_step_and_at_twenty_one_refused_replies():
    with no_replay():
        assert line(2, lambda rows: None) is True and line(2, lambda rows: setrows(rows, "corrupted", [2999], True)) is False and line(2, lambda rows: setrows(rows, "corrupted", [0], True)) is False
        assert line(3, lambda rows: setrows(rows, "parsed", range(20), False)) is True and line(3, lambda rows: setrows(rows, "parsed", range(21), False)) is False         # 0.5 % of 4,000 is 20


def test_line_five_flips_between_2499999_and_2500000_tokens_and_the_tokens_are_those_of_every_call():
    with no_replay():
        for n, want in ((2_500_000, True), (2_499_999, False), (2_500_001, True)):
            r = ready(); total(r, n); assert H._hard_run_lines(r)[5] is want, n
        r = ready(); r["rows"][3].update(asks=2, first_text="x", reasks=[dict(text=r["rows"][3]["text"], tokens=11, think=0, cached=False, prompt_tokens=700)]); total(r, 2_500_000)
        assert M.tokens_sent(r["rows"]) == 2_500_000 and H._hard_run_lines(r)[5] is True and H.H_TOKENS_MIN == 2_500_000
        r["rows"][3]["reasks"][0]["cached"] = True; assert M.tokens_sent(r["rows"]) == 2_500_000 - 711 and H._hard_run_lines(r)[5] is False                  # an attempt answered from the saved file is not counted


def test_lines_six_and_seven_flip_at_eighty_percent_of_a_tenth_and_ninety_percent_of_the_run():
    with no_replay():
        for t in range(10):
            lo, k = t * 400, 80
            assert line(6, lambda rows: setrows(rows, "exact", range(lo, lo + k), False)) is True and line(6, lambda rows: setrows(rows, "exact", range(lo, lo + k + 1), False)) is False, t
        spread = lambda m: (lambda rows: [setrows(rows, "exact", range(t * 400, t * 400 + m), False) for t in range(10)])           # m non-exact steps in each tenth
        assert line(7, spread(40)) is True and line(7, spread(41)) is False and line(6, spread(41)) is True                          # 90 % of the run is 3,600 steps; 41 per tenth is still 89.75 % in every tenth
        assert line(7, lambda rows: setrows(rows, "exact", range(0, 400), False)) is True                                           # a whole bad tenth: line 7 holds (90 %), line 6 does not
        assert line(6, lambda rows: setrows(rows, "exact", range(0, 400), False)) is False


def test_lines_eight_and_nine_flip_at_distance_four_and_at_a_half_more_than_the_first_tenth():
    with no_replay():
        assert line(8, lambda rows: setrows(rows, "distance", [1234], 4)) is True and line(8, lambda rows: setrows(rows, "distance", [1234], 5)) is False and line(8, lambda rows: setrows(rows, "distance", [3999], 5)) is False
        assert line(9, lambda rows: setrows(rows, "distance", range(3600, 3800), 1)) is True and line(9, lambda rows: setrows(rows, "distance", range(3600, 3801), 1)) is False        # 0.5 of 400 steps is 200 steps of distance 1
        def shifted(extra):
            def edit(rows):
                setrows(rows, "distance", range(0, 400), 1); setrows(rows, "distance", range(3600, 4000), 1); setrows(rows, "distance", range(3600, 3600 + extra), 2)
            return edit
        assert line(9, shifted(200)) is True and line(9, shifted(201)) is False                                                  # first tenth 1.0, last 1.5 or a little more
        assert line(9, lambda rows: setrows(rows, "distance", range(0, 400), 3)) is True                                         # a first tenth worse than the last one is fine


def test_line_ten_counts_only_steps_that_start_from_the_oracle_and_ask_for_a_change_that_can_be_applied():
    with no_replay():
        assert line(10, lambda rows: setrows(rows, "wrong", range(40), True)) is True and line(10, lambda rows: setrows(rows, "wrong", range(41), True)) is False
        def fewer(k):
            def edit(rows): setrows(rows, "applicable", range(1000, 4000), False); setrows(rows, "wrong", range(1000, 4000), None); setrows(rows, "wrong", range(k), True)
            return edit
        assert line(10, fewer(10)) is True and line(10, fewer(11)) is False                                                       # 1,000 steps counted: 10 may be wrong
        def moved(k):
            def edit(rows): setrows(rows, "start_same", range(1000, 4000), False); setrows(rows, "wrong", range(k), True)
            return edit
        assert line(10, moved(10)) is True and line(10, moved(11)) is False
        assert line(10, lambda rows: (setrows(rows, "applicable", range(4000), False), setrows(rows, "wrong", range(4000), None))) is False          # nothing counted: not a yes


def test_lines_eleven_and_twelve_flip_at_one_and_a_half_times_the_first_tenth_and_need_a_first_tenth_to_compare_with():
    with no_replay():
        def last(key, v): return lambda rows: (setrows(rows, key, range(4000), 100), setrows(rows, key, range(3600, 4000), v))
        assert line(11, last("prompt_tokens", 150)) is True and line(11, last("prompt_tokens", 151)) is False and line(11, last("prompt_tokens", 100)) is True
        assert line(11, lambda rows: setrows(rows, "prompt_tokens", range(0, 400), 0)) is False and line(11, lambda rows: setrows(rows, "prompt_tokens", range(0, 400), None)) is False
        assert line(11, lambda rows: setrows(rows, "prompt_tokens", range(4000), 0)) is False                                                  # no prompt in the first tenth AND none in the last: nothing to compare, not a pass
        assert line(12, last("seconds", 150.0)) is True and line(12, last("seconds", 151.0)) is False                                   # 100 s a step in the first tenth, 150 s or a little more in the last
        assert line(12, lambda rows: setrows(rows, "seconds", range(4000), 0.0)) is False                                                      # no seconds in the first tenth AND none in the last: nothing to compare, not a pass
        run = ready(); setrows(run["rows"], "seconds", range(4000), 1.0); setrows(run["rows"], "seconds", range(3600, 4000), 1.5); assert H._hard_run_lines(run)[12] is True
        setrows(run["rows"], "seconds", range(3600, 4000), 1.51); assert H._hard_run_lines(run)[12] is False
        setrows(run["rows"], "seconds", range(3600, 4000), 0.1); assert H._hard_run_lines(run)[12] is True
        setrows(run["rows"], "seconds", range(0, 400), 0.0); assert H._hard_run_lines(run)[12] is False                                  # a first tenth of zero seconds is no yardstick


def _contrast(seed, **kw):
    kw = dict(dict(reasks=2, degree_threshold=H.HARD_DEGREE), **kw)
    return {p: json.loads(json.dumps(hrun(seed, 400, p, llm=HEcho(seed, tweak=forgetful_hub), **kw)[0])) for p in M.CONTRAST_POLICIES}


def test_lines_four_and_thirteen_read_the_contrast_and_are_not_run_when_it_was_made_with_other_settings():
    run1 = ready(); ctr = _contrast(H.HARD_SEEDS[0]); made = {"made with " + str(kw): _contrast(H.HARD_SEEDS[0], **kw) for kw in (dict(reasks=1), dict(reasks=0), dict(degree_threshold=4))}
    made["another seed"] = _contrast(H.HARD_SEEDS[1]); made["a seed that is not registered"] = _contrast(5)                          # made here: a stand-in run needs the replay that no_replay() switches off
    with no_replay():
        v = H.verdict_hard(run1, ctr); assert [x[1] for x in v] == [t for _, t in H.HARD_LINES] and [x[0] for x in v] == [t for t, _ in H.HARD_LINES] and v[3][2] == "yes" and v[12][2] == "yes" and v[13][2] == "not run"
        bl = lambda c: sum(1 for r in c["blind"]["rows"] if r["corrupted"])
        assert bl(ctr) >= 320 and sum(x["distance"] for x in ctr["gate"]["rows"][200:400]) / 200 > 0.5 and sum(x["distance"] for x in ctr["blind"]["rows"][200:400]) / 200 > 0.5
        def with_blind(k):                                              # exactly k corrupted steps among the 400 of blind apply
            c = json.loads(json.dumps(ctr)); setrows(c["blind"]["rows"], "corrupted", range(400), False); setrows(c["blind"]["rows"], "corrupted", range(k), True); return H.verdict_hard(run1, c)[3][2]
        assert with_blind(320) == "yes" and with_blind(319) == "NO"
        def with_scope(k):                                              # scope_m7's distance in steps 201 to 400 is k/200
            r = ready(); setrows(r["rows"], "distance", range(200, 400), 0); setrows(r["rows"], "distance", range(200, 200 + k), 1); return H.verdict_hard(r, ctr)[12][2]
        assert with_scope(100) == "yes" and with_scope(101) == "NO" and with_scope(0) == "yes"
        def with_gate(mean_x200, scope_x200):                           # "smaller than the gate's": gate and scope_m7 get these sums over the 200 steps
            c = json.loads(json.dumps(ctr)); r = ready()
            for key, rows, k in (("gate", c["gate"]["rows"], mean_x200), ("scope", r["rows"], scope_x200)):
                setrows(rows, "distance", range(200, 400), 0); setrows(rows, "distance", range(200, 200 + k), 1)
            return H.verdict_hard(r, c)[12][2]
        assert with_gate(41, 40) == "yes" and with_gate(40, 40) == "NO" and with_gate(0, 0) == "NO" and with_gate(39, 40) == "NO"
        def edge(which, idx):                                          # a single offset of the gate (or blind apply) at the first or the last of the 200 steps read must still count
            c = json.loads(json.dumps(ctr)); r = ready()
            for rows in (c[which]["rows"], r["rows"]): setrows(rows, "distance", range(200, 400), 0)
            c[which]["rows"][idx]["distance"] = 1; return H.verdict_hard(r, c)[12][2]
        assert edge("gate", 200) == edge("gate", 399) == edge("blind", 200) == edge("blind", 399) == "yes"                                    # scope_m7 has 0, the other arm has 1/200 more
        def with_blind_mean(k, scope_k):
            c = json.loads(json.dumps(ctr)); r = ready(); setrows(c["blind"]["rows"], "distance", range(200, 400), 0); setrows(c["blind"]["rows"], "distance", range(200, 200 + k), 1)
            setrows(r["rows"], "distance", range(200, 400), 0); setrows(r["rows"], "distance", range(200, 200 + scope_k), 1); return H.verdict_hard(r, c)[12][2]
        assert with_blind_mean(60, 50) == "yes" and with_blind_mean(50, 50) == "NO" and with_blind_mean(40, 50) == "NO"
        cases = {"none": None, "a policy missing": {k: v for k, v in ctr.items() if k != "gate"}, "repair_m6 missing": {k: v for k, v in ctr.items() if k != "repair_m6"}}
        short = json.loads(json.dumps(ctr)); short["gate"]["rows"] = short["gate"]["rows"][:399]; cases["399 steps"] = short
        longer = json.loads(json.dumps(ctr)); longer["gate"]["rows"].append(dict(longer["gate"]["rows"][-1])); cases["401 steps"] = longer
        cases.update(made)
        odd = json.loads(json.dumps(ctr)); odd["blind"]["meta"]["policy"] = "gate"; cases["a file labelled with another policy"] = odd
        for name, c in cases.items():
            v = H.verdict_hard(run1, c); assert v[3][2] == v[12][2] == "not run", name
        assert H.headline(H.verdict_hard(run1, None)) == "PASSED"                          # the contrast is not a headline line


def test_line_fourteen_needs_the_second_seed_five_million_tokens_together_and_every_one_of_the_lines_it_names():
    run1 = ready(0)
    with no_replay():
        assert H.verdict_hard(run1)[13][2] == "not run" and H.verdict_hard(run1, None, ready(1))[13][2] == "yes"
        def two(n, edit=lambda rows: None, meta=None, first=None):
            a, b = ready(0), ready(1); edit(b["rows"]); b["meta"].update(meta or {}); total(a, n // 2); total(b, n - n // 2); return H.verdict_hard(a, None, b)[13][2]
        assert two(5_000_000) == "yes" and two(4_999_999) == "NO" and two(5_000_001) == "yes"
        assert H.verdict_hard(run1, None, ready(0))[13][2] == "NO"                                          # the same seed twice is not a second run
        for ln, edit in ((2, lambda rows: setrows(rows, "corrupted", [5], True)), (3, lambda rows: setrows(rows, "parsed", range(21), False)), (6, lambda rows: setrows(rows, "exact", range(80 + 1), False)), (7, lambda rows: setrows(rows, "exact", list(range(0, 4000, 10)) + [1], False)),
                         (8, lambda rows: setrows(rows, "distance", [9], 5)), (9, lambda rows: setrows(rows, "distance", range(3600, 3801), 1)), (10, lambda rows: setrows(rows, "wrong", range(41), True)),
                         (11, lambda rows: setrows(rows, "prompt_tokens", range(3600, 4000), 1400)), (12, lambda rows: setrows(rows, "seconds", range(3600, 4000), 2.0))):
            a, b = ready(0), ready(1); edit(b["rows"]); assert H._hard_run_lines(b)[ln] is False, ln; assert H.verdict_hard(a, None, b)[13][2] == "NO", ln
        for key, val in (("reasks", 1), ("degree_threshold", 4), ("steps", 3000), ("policy", "gate")):
            assert two(6_000_000, meta={key: val}) == "NO", key                                           # run 2 made with other settings (line 1 of run 2)
    b = ready(1); b["rows"][5]["exact"] = False; assert H.verdict_hard(run1, None, b)[13][2] == "NO"          # and with the replay on, a doctored run 2 does not count either


def test_the_headline_is_passed_only_when_all_nine_lines_hold_and_not_decided_when_one_was_not_run():
    mk = lambda **ov: [(tag, text, ov.get(f"l{i}", "yes")) for i, (tag, text) in enumerate(H.HARD_LINES, 1)]
    assert H.HEADLINE_LINES == (1, 2, 3, 5, 6, 7, 8, 9, 10) and len(H.HARD_LINES) == 14 and H.headline(mk()) == "PASSED"
    for i in H.HEADLINE_LINES: assert H.headline(mk(**{f"l{i}": "NO"})) == "NOT PASSED" and H.headline(mk(**{f"l{i}": "not run"})) == "NOT DECIDED", i
    for i in (4, 11, 12, 13, 14): assert H.headline(mk(**{f"l{i}": "NO"})) == "PASSED", i                       # reported with the headline, not part of it
    assert H.headline(mk(l5="not run", l7="NO")) == "NOT PASSED"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): H.print_verdict_hard(mk(l7="NO", l14="not run"))
    out = buf.getvalue(); assert "NOT PASSED" in out and out.count("<- headline line") == 9 and "not run 14. [claim]" in out and "NO      7. [claim]" in out


def test_the_reports_count_new_offsets_by_cause_the_re_asks_and_the_control_arm():
    llm = Asker(5, bad={5: 3, 9: 1, 14: 1}); r = hrun(5, 60, llm=llm, reasks=2, degree_threshold=8)[0]; rep = H.report_hard(r); rows = r["rows"]
    assert rep["steps"] == 60 and rep["reasked_steps"] == 3 and rep["rescued"] == 2 and rep["still_refused"] == 1 and rep["extra_calls"] == 4 and rep["from_refused"] == 1          # two rescued, one not: told apart
    off = H.report_hard(hrun(5, 60, llm=Asker(5, bad={5: 3, 9: 1}), reasks=0, degree_threshold=8)[0]); assert off["reasked_steps"] == 0 and off["rescued"] == 0 and off["still_refused"] == 2 and off["extra_calls"] == 0     # without re-asks every refused reply is a refused step
    new = [x for x in rows if x["start_same"] and not x["exact"]]; assert rep["new_offsets"] == len(new) == rep["from_refused"] + rep["from_readable"] and not rows[4]["parsed"] and rows[4]["start_same"] and rows[8]["parsed"] and rows[8]["asks"] == 2 and not rows[8]["start_same"]
    assert sum(n for w, n in rep["wrong_by_kind"].values()) == sum(1 for x in rows if x["start_same"] and x["applicable"]) and all(w <= n for w, n in rep["wrong_by_kind"].values())
    assert abs(rep["exact_share"] - sum(x["exact"] for x in rows) / 60) < 1e-12 and len(rep["tenth_exact"]) == 10 and rep["dropped_items"] == sum(x.get("dropped", 0) for x in rows)
    f = hrun(21, 400, llm=HEcho(21, tweak=forgetful_hub), degree_threshold=4)[0]; fr = H.report_hard(f)                          # the default repair keeps a service with more than 4 dependants: one slip, and the offset stays
    assert fr["new_offsets"] == 1 and fr["from_readable"] == 1 and fr["from_refused"] == 0 and fr["reasked_steps"] == 0 and fr["wrong_by_kind"]["delete_hub"] == (1, 1) and sum(w for w, n in fr["wrong_by_kind"].values()) == 1 and fr["exact_share"] < 0.2
    chatty = lambda n, kind, ideal: ideal + (" ADD[9,9:1]" if kind == "add_node" else "")                                              # an extra item the request did not name: the scope filter removes it
    c = hrun(21, 100, llm=HEcho(21, tweak=chatty), degree_threshold=8)[0]; cr = H.report_hard(c); adds = sum(1 for x in c["rows"] if x["kind"] == "add_node")
    assert adds > 3 and cr["steps_with_dropped"] == adds and cr["dropped_items"] == adds and cr["new_offsets"] == 0 and cr["exact_share"] == 1.0
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): H.print_hard_report(rep, "main run")
    out = buf.getvalue(); assert "3 steps needed one (4 extra calls); 2 of them ended with a readable reply" in out and "add_edge 1/8" in out and "share exact in each tenth" in out
    main = hrun(5, 100, llm=Asker(5, bad={5: 3, 9: 1}), reasks=2, degree_threshold=8)[0]; ctl = hrun(5, 60, llm=Asker(5, bad={5: 3, 9: 1}), reasks=0, degree_threshold=8)[0]
    c = H.compare_control(main, ctl); assert c["steps"] == 60 and c["with_reask"]["steps"] == 60 and c["without_reask"]["steps"] == 60 and c["without_reask"]["refused"] == 2 and c["with_reask"]["refused"] == 1
    assert c["without_reask"]["exact"] < c["with_reask"]["exact"] + 1e-9 and c["with_reask"]["steps"] == c["without_reask"]["steps"]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): H.print_control(c)
    assert "with re-ask" in buf.getvalue() and "without re-ask" in buf.getvalue() and "first 60 steps" in buf.getvalue()


def test_the_twenty_step_check_advises_the_lowest_thinking_level_and_names_a_cost_a_time_and_a_cap_that_are_too_big():
    class Thinker(HEcho):
        last_think_tokens = 300
    r = hrun(99, 20, llm=Thinker(99), reasks=2, degree_threshold=8)[0]["summary"]
    sm = M.smoke_report(r, H.HARD_STEPS, 1.0, 4.0, think_limit=H.HARD_THINKING_LIMIT, lowest=H.HARD_LOWEST_THINKING, budget_usd=1.0, minutes_limit=10, token_cap=1_000_000)
    adv = " | ".join(sm["advice"]); assert 'THINKING = "low"' in adv and "300 thinking tokens a step" in adv and "above your budget" in adv and "hours of calls" in adv and "above the cap" in adv and "no prices" not in adv
    assert abs(sm["think_per_step"] - 300) < 1e-9 and sm["projected_cost"] > 1.0 and abs(sm["projected_tokens"] - sm["tokens_per_step"] * 4000) < 1e-6
    calm = M.smoke_report(hrun(99, 20, reasks=2, degree_threshold=8)[0]["summary"], 4000, 1.0, 4.0, think_limit=150, lowest="low", budget_usd=1000.0, minutes_limit=1e6, token_cap=10 ** 9); assert calm["advice"] == []
    half = hrun(7, 10)[0]["summary"]                                                                                                        # 0.5 s a step: 1200 steps are exactly 10 minutes
    assert not any("hours of calls" in a for a in M.smoke_report(half, 1200, 1.0, 4.0, minutes_limit=10)["advice"]) and any("hours of calls" in a for a in M.smoke_report(half, 1200, 1.0, 4.0, minutes_limit=9.99)["advice"])
    cost = M.smoke_report(half, 1200, 1.0, 4.0)["projected_cost"]; assert cost > 0                                                              # a budget exactly equal to the projected cost is not exceeded; a hair under it is
    assert not any("above your budget" in a for a in M.smoke_report(half, 1200, 1.0, 4.0, budget_usd=cost)["advice"]) and any("above your budget" in a for a in M.smoke_report(half, 1200, 1.0, 4.0, budget_usd=cost * 0.999)["advice"])
    none = M.smoke_report(r, 4000, lowest="low"); assert none["projected_cost"] is None and any("no prices" in a for a in none["advice"])
    assert H.HARD_THINKING_LIMIT == 150 and H.HARD_LOWEST_THINKING == "low" and "minimal" in M.smoke_report(r, 4000)["advice"][0]


# ---------------------------------------------------------------- 5 the page and the notebook
def _head():
    text = open(PAGE).read(); assert text.count(RESULT_MARK) <= 1
    return text[:text.index(RESULT_MARK)] if RESULT_MARK in text else text


def test_the_page_is_locked_and_carries_the_fourteen_lines_and_the_settings_the_code_uses():
    head = _head(); assert _h(head) == PAGE_SHA256, "the predictions and reading rules must not change after the run; later notes go below them"
    for i, (tag, line) in enumerate(H.HARD_LINES, 1): assert f"\n{i}. [{tag}] {line}\n" in head, i
    assert "Fourteen lines" in head and len(H.HARD_LINES) == 14 and [t for t, _ in H.HARD_LINES].count("guard") == 2 and [t for t, _ in H.HARD_LINES].count("instrument") == 2 and [t for t, _ in H.HARD_LINES].count("claim") == 10
    for needle in (f"stream seed {H.HARD_SEEDS[0]}", f"stream seed {H.HARD_SEEDS[1]}", "4,000 requests", f"first {H.HARD_CONTRAST_STEPS} requests", f"first {H.HARD_CONTROL_STEPS:,} requests", f"`{H.HARD_MODEL}`", f"{H.HARD_TOKEN_CAP:,} tokens",
                   f"{H.HARD_FREE_DAILY} calls a day", f"more than {H.HARD_THINKING_LIMIT} thinking tokens a step", f"more than {H.HARD_MINUTES_LIMIT // 60} hours", f'"{H.HARD_LOWEST_THINKING}"', "seed 99", "`million_hard.verdict_hard`", "`million_hard.registered`",
                   "`million_hard.headline`", "`million.replay`", "`million_hard.HardStream`", "`million.reask_messages`", f"`degree_threshold={H.HARD_DEGREE}`", "Up to 24 services and 48 edges",
                   "at most 8,192 output tokens including thinking", "UNVERIFIED", "$0.75 per million input tokens and $3.75 per million output tokens", "1 January 2027", "refuses to start unless two prices and a budget are typed in",
                   "never turns billing on", "test_million_hard_result.py", "(85 %)", "(99 %)", "(93 %)", "(55 %)", "(90 %)", "(30 %)", "(40 %)", "(45 %)", "(65 %)", "(80 %)", "(97 %)", "(60 %)", "(8 %)",
                   "about **20 %**", "about **43 %**", "**under 3 %**", "I expect a NOT PASSED", "the line I expect to fail first", "3.8 Flash"):
        assert needle in head, needle
    assert H.HARD_MODEL == "gemini-3.8-flash" and H.HARD_TOKEN_CAP == 8_000_000 and H.HARD_FREE_DAILY == 20 and H.HARD_THINKING_LIMIT == 150 and H.HARD_MINUTES_LIMIT == 240 and H.HARD_LOWEST_THINKING == "low" and H.HARD_REASKS == 2
    assert H.HARD_DEGREE == 8 and (H.HARD_STEPS, H.HARD_CONTRAST_STEPS, H.HARD_CONTROL_STEPS) == (4000, 400, 1000) and M.MAIN_POLICY == "scope_m7" and M.CONTRAST_POLICIES == ("blind", "gate", "repair_m6")
    assert "not** a Pro model" in head and head.count("frontier") == 0 and "mid-tier" not in head
    lines_text = "\n".join(l for _, l in H.HARD_LINES)
    consts = [("at most 0.5 percent", H.H_REFUSED_MAX == 0.005), ("at least 80 percent of its first 400", H.H_BLIND_MIN == 0.8), ("at least 2,500,000", H.H_TOKENS_MIN == 2_500_000), ("at least 5,000,000", H.H_TOKENS_MIN_TWO == 5_000_000),
              ("at least 80 percent of the steps hold exactly", H.H_TENTH_EXACT == 0.8), ("at least 90 percent of all steps", H.H_ALL_EXACT == 0.9), ("exceeded 4 items", H.H_DISTANCE == 4), ("at most 0.5 above", H.H_DRIFT_ADD == 0.5),
              ("at most 1.0 percent", H.H_WRONG_MAX == 0.01), ("at most 1.5 times", H.H_FLAT == 1.5), ("at most 0.5, and smaller", H.H_CONTRAST_MEAN == 0.5), ("steps 201 to 400", H.HARD_CONTRAST_STEPS == 400), ("step 4,000", H.HARD_STEPS == 4000),
              ("ten tenths", M.BLOCKS == 10), ("up to two re-asks", H.HARD_REASKS == 2), ("up to 8 dependants", H.HARD_DEGREE == 8), ("one of the two registered seeds", len(H.HARD_SEEDS) == 2)]
    for needle, ok in consts: assert ok, needle; assert needle in lines_text, needle
    assert lines_text.count("at most 1.5 times") == 2 and H.HEADLINE_LINES == (1, 2, 3, 5, 6, 7, 8, 9, 10) and "if lines 1, 2, 3, 5, 6, 7, 8, 9 and 10 all read \"yes\"" in head and "(lines 1, 2, 3, 5, 6, 7, 8, 9, 10)" in head


def test_the_page_predictions_are_numbered_like_the_lines_and_the_overall_figures_follow_from_them():
    head = _head(); pred = head.split("## Predictions")[1].split("## Reading rules")[0]
    nums = [int(m.group(1)) for l in pred.splitlines() for m in [re.match(r"^(\d+)\. ", l)] if m]; assert nums == list(range(1, 16)), nums
    for i in range(1, 15):
        line = [l for l in pred.splitlines() if l.startswith(f"{i}. [")]; assert len(line) == 1 and re.search(r"\(\d+ %\)", line[0]), i
        assert f"{i}. [{H.HARD_LINES[i - 1][0]}]" in line[0], i                                                            # the tag is the line's
    assert "\n15. Not reading-rule lines." in pred
    p = {i: int(re.search(r"\((\d+) %\)", [l for l in pred.splitlines() if l.startswith(f"{i}. [")][0]).group(1)) for i in range(1, 15)}
    assert abs(100 * (p[1] / 100) * (p[2] / 100) * (p[3] / 100) * (p[4] / 100) - 43) <= 2 and "about **43 %**" in head                           # lines 1 to 4 multiplied
    assert p[6] == min(p[i] for i in range(1, 14)) and "line I expect to fail first" in head and 10 <= min(p[i] for i in range(1, 14))
    assert p[14] < 10 and p[14] == min(p.values())
    assert 15 <= 100 * (p[1] / 100) * (p[2] / 100) * (p[3] / 100) * (p[5] / 100) * 0.30 <= 25 and "about **20 %**" in head
    assert head.count("(97 %)") == 1 and p[11] == 97 and p[2] == 99 and p[1] == 85


def test_the_numbers_the_page_cites_are_the_saved_flash_lite_files_the_oracle_stream_and_the_arithmetic_of_the_cost_table():
    head = _head(); base = os.path.join(HERE, "..", "..", "docs", "results")
    d = json.load(open(os.path.join(base, "axi_gemini_long_norule_gemini-3.5-flash-lite_seed7.json")))
    rows = lambda p: [r for e in d["rows"] for r in e["policies"][p]]
    s, b, r6 = rows("scope_m7"), rows("blind"), rows("repair_m6")
    assert (len(s), sum(r["exact"] for r in s), sum(r["corrupted"] for r in s), sum(not r["parsed"] for r in s)) == (120, 120, 0, 0) and (sum(r["exact"] for r in r6), sum(r["corrupted"] for r in r6)) == (120, 0)
    assert (sum(r["exact"] for r in b), sum(r["corrupted"] for r in b), sum(any(x["corrupted"] for x in e["policies"]["blind"]) for e in d["rows"])) == (61, 59, 3)
    assert sorted(min(x["t"] for x in e["policies"]["blind"] if x["corrupted"]) for e in d["rows"] if any(x["corrupted"] for x in e["policies"]["blind"])) == [3, 13, 14]
    assert round(sum(r["prompt_tokens"] for r in s) / 120, 1) == 308.6
    q = json.load(open(os.path.join(base, "axi_scope_long_norule_Qwen2.5-7B_seed7.json"))); qs = [r for e in q["rows"] for r in e["policies"]["scope_m7"]]; assert round(sum(r["prompt_tokens"] for r in qs) / len(qs), 1) == 307.3
    api = open(os.path.join(HERE, "..", "..", "docs", "API_TEST_GEMINI.md")).read()                                                  # the figures that come from the written-up Part A
    for needle in ("3 of 140", "0 of 177", "20 of 20 without help", "15 of 20 graphs left with a dangling edge"): assert needle in api, needle
    for needle in ("120 of 120 steps", "blind apply held it on 61 of 120", "a corrupted graph on 59", "3 of 4 episodes", "(first at step 3, 13 and 14)", "15 of 20 requests", "20 of 20 without help", "3 of 140 single replies",
                   "0 of 177", "75 of 334", "308.6 prompt tokens a step against 307.3"): assert needle in head, needle
    assert "75 of 334" in open(os.path.join(HERE, "..", "..", "docs", "MILLION_TOKEN_TEST_GEMINI.md")).read() and "3 of 140" in open(os.path.join(HERE, "..", "..", "docs", "MILLION_TOKEN_TEST_GEMINI.md")).read()
    # the oracle's stream
    chars, kinds, items, tn, te = [], {}, {}, [], []
    for seed in H.HARD_SEEDS:
        st_ = H.HardStream(seed); tot = 0; n = 0; k = {}; it = {}; N, E = [], []
        for i in range(H.HARD_STEPS):
            st, ob = st_.next(); oa = st_.live
            tot += sum(len(m["content"]) for m in M.base_messages(M.Task(st["kind"], copy_graph(ob), st["instruction"], st["ideal"], oa), False)); k[st["kind"]] = k.get(st["kind"], 0) + 1; it[st["kind"]] = it.get(st["kind"], 0) + st["ideal"].count("["); N.append(len(ob.nodes)); E.append(len(ob.edges))
        chars.append(tot / H.HARD_STEPS); kinds[seed] = {x: 100 * v / H.HARD_STEPS for x, v in k.items()}; items[seed] = {x: it[x] / k[x] for x in k}
        tn += [sum(N[j * 400:(j + 1) * 400]) / 400 for j in range(10)]; te += [sum(E[j * 400:(j + 1) * 400]) / 400 for j in range(10)]
    assert [round(c) for c in chars] == [1415, 1442] and [round(-256.8 + 0.6709 * c) for c in chars] == [693, 711] and "1,415 characters (seed 20261031) and 1,442 characters (seed 20261032)" in head and "about 693 and 711 prompt tokens" in head
    assert (round(min(tn), 1), round(max(tn), 1), round(min(te), 1), round(max(te), 1)) == (17.1, 22.2, 18.4, 37.0) and "17.1 to 22.2 services and 18.4 to 37.0 edges" in head
    hub = [kinds[s_]["delete_hub"] for s_ in H.HARD_SEEDS]; assert (round(min(hub), 1), round(max(hub), 1)) == (8.1, 8.7) and "8.1 to 8.7 percent" in head
    hi = [items[s_]["delete_hub"] for s_ in H.HARD_SEEDS]; assert (round(min(hi), 1), round(max(hi), 1)) == (5.8, 6.0) and "5.8 to 6.0 edges" in head
    assert [round(min(kinds[s_][x] for s_ in H.HARD_SEEDS), 1) for x in ("delete_dep", "delete_leaf", "delete_pair")] == [3.1, 2.8, 3.2] and [round(max(kinds[s_][x] for s_ in H.HARD_SEEDS), 1) for x in ("delete_dep", "delete_leaf", "delete_pair")] == [3.5, 3.2, 3.7]
    assert all(round(kinds[s_]["delete_edge"], 1) == 3.0 and round(kinds[s_]["add_node"], 1) == 18.5 and 60 <= kinds[s_]["add_edge"] <= 60.5 for s_ in H.HARD_SEEDS)
    allk = [sum(kinds[s_][x] for x in ("delete_dep", "delete_hub", "delete_leaf", "delete_edge", "delete_pair")) for s_ in H.HARD_SEEDS]; dep = [sum(kinds[s_][x] for x in ("delete_dep", "delete_hub", "delete_pair")) for s_ in H.HARD_SEEDS]
    assert all(20.9 <= x <= 21.4 for x in allk) and all(14.9 <= x <= 15.4 for x in dep) and "all deletes together about 21 percent" in head and "about 15 percent" in head
    assert "(1,302 against 1,451 characters on seed 20261031, 1,309 against 1,525 on seed 20261032)" in head                           # the first and last tenth, as the prediction for line 11 says
    for seed, a, b2 in ((H.HARD_SEEDS[0], 1302, 1451), (H.HARD_SEEDS[1], 1309, 1525)):
        st_ = H.HardStream(seed); cs = []
        for i in range(H.HARD_STEPS):
            st, ob = st_.next(); oa = st_.live; cs.append(sum(len(m["content"]) for m in M.base_messages(M.Task(st["kind"], copy_graph(ob), st["instruction"], st["ideal"], oa), False)))
        assert (round(sum(cs[:400]) / 400), round(sum(cs[-400:]) / 400)) == (a, b2) and 1.1 <= b2 / a <= 1.17
    chk = H.HardStream(99); crow = []                                                                                                    # the 20-step check: 200 requests answered by the oracle, then 20 asked of the model
    for i in range(H.HARD_WARM + 20):
        st, ob = chk.next(); oa = chk.live
        if i >= H.HARD_WARM: crow.append((st["kind"], len(ob.nodes), len(ob.edges), sum(len(m["content"]) for m in M.base_messages(M.Task(st["kind"], copy_graph(ob), st["instruction"], st["ideal"], oa), False))))
    assert H.HARD_WARM == 200 and (round(sum(r[1] for r in crow) / 20), round(sum(r[2] for r in crow) / 20, 1), round(sum(r[3] for r in crow) / 20), sum(r[0] == "delete_hub" for r in crow), sum(r[0] == "delete_dep" for r in crow)) == (20, 25.9, 1406, 2, 2)
    assert "The first 200 requests of it are answered by the oracle itself" in head and "on average 20 services, 25.9 edges and 1,406 characters a prompt" in head and "with 2 hub deletes and 2 deletes with dependants among them" in head
    # the cost table: the arithmetic of the page, on the prices it names as unverified
    P_IN, P_OUT, PROMPT, REPLY = 0.75, 3.75, 702, 20
    per = lambda think: PROMPT * P_IN / 1e6 + (REPLY + think) * P_OUT / 1e6
    for think in (0, 100, 300, 1000):
        r1, ctr, ctl = per(think) * 4000, per(think) * 1200, per(think) * 1000
        row = f"| {think} | ${r1:.2f} | ${ctr:.2f} | ${ctl:.2f} | ${r1 + ctr + ctl:.2f} | ${2 * r1 + ctr + ctl:.2f} |"; assert row in head, row
    assert "about $4 for run 1, about $6 with the contrast and the control, and about $10 with run 2 as well" in head and "$7, $11 and $18 at 300" in head and "a budget of 15 dollars" in head
    assert round((693 + 711) / 2) == 702 and "(4,000 steps)" in head and "(1,200 steps)" in head and "(1,000 steps)" in head


def test_the_notebook_never_turns_billing_on_holds_no_key_needs_typed_prices_and_a_budget_and_stops_by_itself():
    nb = json.load(open(NB)); code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"); allsrc = "\n".join("".join(c["source"]) for c in nb["cells"])
    assert re.search(r"^PAID = None\b", code, flags=re.M) and re.search(r"^THINKING = None\b", code, flags=re.M) and re.search(r"^PRICE_IN = None\b", code, flags=re.M) and re.search(r"^PRICE_OUT = None\b", code, flags=re.M) and re.search(r"^BUDGET_USD = None\b", code, flags=re.M)
    assert 'userdata.get("GEMINI_API_KEY")' in code and not re.search(r"AIza[0-9A-Za-z_\-]{10,}", allsrc) and "print(KEY" not in code and "print(key" not in code
    assert "docs/MILLION_TOKEN_TEST_HARD.md" in allsrc and "axi-colab-million.zip" in allsrc and "8,000,000 tokens" in allsrc and "never turns billing on" in allsrc
    assert "max_tokens=HARD_TOKEN_CAP" in code and "budget_usd=BUDGET_USD, price_in=PRICE_IN, price_out=PRICE_OUT" in code and "stream=\"HardStream\"" in code and "reasks=HARD_REASKS" in code and "degree_threshold=HARD_DEGREE" in code
    assert "run_file(tag, 99, HARD_WARM + 20, MAIN_POLICY, scripted=HARD_WARM)" in code and "smoke_report(" in code and 'lowest=HARD_LOWEST_THINKING' in code and "minutes_limit=HARD_MINUTES_LIMIT" in code and "token_cap=HARD_TOKEN_CAP" in code and "BudgetStop" in code
    assert "assert PAID in (True, False)" in code and "assert PRICE_IN" in code and "assert BUDGET_USD" in code and "daily_limit=None if PAID else HARD_FREE_DAILY" in code
    assert all(c["cell_type"] != "code" or (c.get("outputs") == [] and c.get("execution_count") is None) for c in nb["cells"])                           # no outputs saved in the notebook


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn): fn(); print("PASS", name); n += 1
    print(n, "tests passed")
