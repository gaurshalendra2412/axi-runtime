"""Implements concept-map item 93 (million-token test on a real model; rows 3, 9, 27): the INSTRUMENT of docs/MILLION_TOKEN_TEST.md, written before the run. It checks the instrument, not a result:
  1. the request stream: deterministic, only the registered kinds, inside the caps and not ending early for 3,600 steps on both registered seeds, and of the sizes the page says it has;
  2. the run with a stand-in that writes the ideal reply: exact at every step, one real call per step (no cache), the file holds a header and one line per step, tokens add up; a stand-in that forgets the
     dependants (what the model did in the 30-step runs) corrupts blind apply, leaves the gate off and leaves scope_m7 exact; the model is shown the HELD state, not the oracle's;
  3. resume: a file cut in the middle of a line continues to the same rows, a finished file makes no call, a file of another run, a doctored flag and a doctored kind are refused, a stream that ends is reported;
  4. the flags of a step (applicable, wrong, start_same, dropped) on hand-built cases; the tenths and halves; the probe; the GPU probe with a fake torch;
  5. every reading-rule line flips exactly at its boundary (2 percent, 1,000,000 tokens, 1.0 item, 10 items, 80 percent, 2 points, 1.5 times twice, 50 percent, strictly smaller, 2,000,000 tokens);
  6. the page and the module agree (the thirteen lines, the numbers, seeds, kinds) and the page's text above its Result is byte for byte what was written before the run (a hash).
What it does NOT show: anything about a real model's replies, or the result of the run."""
import contextlib, hashlib, io, json, os, re, sys, tempfile
from unittest import mock
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.gate import Graph
from axi.experiments import million as M
from axi.experiments.agent_loop import _serialize, copy_graph, same_graph, strict_parse
from axi.experiments.endurance import Stream

HERE = os.path.dirname(__file__)
PAGE = os.path.join(HERE, "..", "..", "docs", "MILLION_TOKEN_TEST.md")
PAGE_SHA256 = "ff90163e2fc1f3378c7d4f30f7cdcb0d96094b0c44f641a0528ba276ff1b7f83"          # filled in when the page was written, before any run
RESULT_MARK = "\n\n---\n\n# Result"


def _h(text): return hashlib.sha256((text.rstrip("\n") + "\n").encode()).hexdigest()


class Echo:
    """A stand-in model: it runs its own copy of the stream and writes the ideal reply of each request (or what `tweak(step, kind, ideal)` makes of it). It records every call."""
    def __init__(self, seed, skip=0, tweak=None):
        self.s, self.tweak, self.calls, self.seen, self.last_prompt_tokens, self.n = M.LiveStream(seed), tweak, 0, [], 100, 0
        for _ in range(skip): self.s.next(); self.n += 1

    def generate(self, messages, constrained, max_new_tokens=400):
        st, _ = self.s.next(); self.n += 1; self.calls += 1; self.seen.append(messages); self.last_prompt_tokens = (len(messages[0]["content"]) + len(messages[1]["content"])) // 3
        text = st["ideal"] if self.tweak is None else self.tweak(self.n, st["kind"], st["ideal"])
        return text, 11, 0.5


def forgetful(n, kind, ideal):
    """What Qwen 7B did without the rule: the service delete without its dependants (for a delete-with-dependants and for the service half of a delete-and-edge request)."""
    items = ideal.split(" ")
    if kind == "delete_dep": return items[0]
    if kind == "delete_pair": return items[0] + " " + items[-1]
    return ideal


def tmp(): return os.path.join(tempfile.mkdtemp(), "r.jsonl")


def run(seed=5, steps=60, policy="scope_m7", path=None, llm=None, **kw):
    path = path or tmp(); llm = llm or Echo(seed)
    return run_with(llm, seed, steps, policy, path, **kw), llm, path


def run_with(llm, seed, steps, policy, path, **kw): return M.run_million(llm, seed, steps, policy, path, meta=dict(model="stand-in"), **kw)


# ---------------------------------------------------------------- 1 the stream
def test_the_stream_is_deterministic_uses_only_the_registered_kinds_and_is_the_steady_streams_prefix_with_other_weights():
    assert sorted(set(M.KINDS_LIVE)) == ["add_edge", "add_node", "delete_dep", "delete_edge", "delete_leaf", "delete_pair"]
    assert [M.KINDS_LIVE.count(k) for k in ("add_node", "add_edge", "delete_dep", "delete_leaf", "delete_edge", "delete_pair")] == [3, 4, 1, 1, 1, 1]
    a, b = M.LiveStream(11), M.LiveStream(11); xs = [a.next() for _ in range(200)]; ys = [b.next() for _ in range(200)]
    assert all(x[0] == y[0] and same_graph(x[1], y[1]) for x, y in zip(xs, ys)) and all(x is not None for x in xs)
    assert same_graph(a.start, Stream(11, "steady").start) and M.LiveStream(12).start.nodes != M.LiveStream(11).start.nodes
    other = Stream(11, "steady"); zs = [other.next()[0] for _ in range(200)]
    assert [x[0] for x in xs] != zs                                              # the weights are not the steady mix's
    assert isinstance(M.LiveStream(1), Stream) and M.LiveStream(1).kinds_all == M.KINDS_LIVE and M.LiveStream(1).kinds_all is not M.KINDS_LIVE


def test_both_registered_seeds_run_3600_steps_inside_the_caps_and_have_the_sizes_the_page_states():
    assert M.SEEDS == (20261021, 20261022) and M.STEPS == 3600
    nodes, edges, deps = [], [], 0
    for seed in M.SEEDS:
        S = M.LiveStream(seed); sn, se = [], []
        for i in range(M.STEPS):
            out = S.next(); assert out is not None, (seed, i)
            st, before = out; sn.append(len(before.nodes)); se.append(len(before.edges)); deps += st["kind"] in ("delete_dep", "delete_pair")
            assert len(S.live.nodes) <= 14 and len(S.live.edges) <= 26 and S.live.is_well_formed(), (seed, i)
        for j in range(10):
            nodes.append(sum(sn[j * 360:(j + 1) * 360]) / 360); edges.append(sum(se[j * 360:(j + 1) * 360]) / 360)
        assert max(sn) == 14 and max(se) >= 25
    assert 7.45 <= min(nodes) and max(nodes) <= 11.75 and 2.65 <= min(edges) and max(edges) <= 9.85, (min(nodes), max(nodes), min(edges), max(edges))       # "7.5 to 11.7 and 2.7 to 9.8" on the page
    assert 0.16 <= deps / (2 * M.STEPS) <= 0.19, deps / (2 * M.STEPS)                                                                                       # "about 17 percent"


# ---------------------------------------------------------------- 2 the run
def test_a_clean_run_is_exact_at_every_step_asks_the_model_once_per_step_and_the_file_holds_one_line_per_step():
    r, llm, path = run(5, 120)
    rows, s = r["rows"], r["summary"]
    assert len(rows) == 120 and llm.calls == 120 and r["stopped"] is None and [x["t"] for x in rows] == list(range(1, 121))
    assert all(x["exact"] and x["distance"] == 0 and x["parsed"] and not x["corrupted"] and x["start_same"] for x in rows) and s["exact"] == 120 and s["max_distance"] == 0
    lines = open(path).read().splitlines(); assert len(lines) == 121 and json.loads(lines[0])["_meta"]["seed"] == 5 and [json.loads(l)["t"] for l in lines[1:]] == list(range(1, 121))
    meta = json.loads(lines[0])["_meta"]; assert meta["policy"] == "scope_m7" and meta["steps"] == 120 and meta["kinds"] == M.KINDS_LIVE and meta["model"] == "stand-in" and meta["hint_rules"] is False
    assert s["prompt_tokens"] == sum(x["prompt_tokens"] for x in rows) and s["output_tokens"] == 11 * 120 and s["tokens"] == s["prompt_tokens"] + s["output_tokens"] == M.tokens_sent(rows)
    assert abs(s["seconds"] - 60.0) < 1e-6 and all(x["tokens"] == 11 and x["seconds"] == 0.5 for x in rows) and s["corrupted"] == 0 and s["refused"] == 0
    assert all(x["applicable"] and x["wrong"] is False for x in rows) and all("dropped" in x and x["dropped"] == 0 for x in rows)


def test_no_cache_a_repeated_prompt_is_asked_again():
    class Same:
        def __init__(self): self.calls, self.last_prompt_tokens = 0, 7
        def generate(self, messages, constrained, max_new_tokens=400): self.calls += 1; return "ADD[0,0:1]", 3, 0.1          # the same reply to every prompt, many prompts repeat
    llm = Same(); r = M.run_million(llm, 3, 40, "gate", tmp(), meta={})
    assert llm.calls == 40 and len(r["rows"]) == 40 and r["summary"]["tokens"] == 40 * 10


def test_the_model_is_shown_the_held_state_not_the_oracles_after_a_miss():
    def omit(n, kind, ideal): return "ADD[4,5:1]" if n == 7 else ideal                  # step 7: an unasked service, which the scope filter drops; the requested change does not happen
    r, llm, _ = run(9, 30, llm=Echo(9, tweak=omit)); rows = r["rows"]
    assert rows[6]["dropped"] == 1 and rows[6]["distance"] > 0 and rows[6]["exact"] is False and rows[6]["wrong"] is True        # the unasked service was dropped and the asked change did not happen
    held = M.replay(rows[:7], 9, "scope_m7")[1]; stream = M.replay(rows[:7], 9, "scope_m7")[0]
    assert not same_graph(held, stream.live), "the miss has to have left the held state off the oracle's"
    assert llm.seen[7][1]["content"].startswith(_serialize(held)) and not llm.seen[7][1]["content"].startswith(_serialize(stream.live))
    assert rows[6]["dropped"] == 1 and rows[7]["start_same"] is False and rows[5]["start_same"] is True


def test_what_the_policies_do_with_a_model_that_forgets_the_dependants_over_the_contrast_length():
    res = {}
    for p in M.CONTRAST_POLICIES + (M.MAIN_POLICY,): res[p] = run(21, M.CONTRAST_STEPS, p, llm=Echo(21, tweak=forgetful))[0]
    rows = {p: res[p]["rows"] for p in res}
    assert sum(r["corrupted"] for r in rows["blind"]) >= 0.5 * M.CONTRAST_STEPS and all(not r["corrupted"] for p in ("gate", "repair_m6", "scope_m7") for r in rows[p])
    mean = lambda p, a, b: sum(r["distance"] for r in rows[p][a:b]) / (b - a)
    assert mean("gate", 200, 300) > 0 and mean("blind", 200, 300) > 0 and mean("scope_m7", 200, 300) == 0 and mean("repair_m6", 200, 300) == 0
    assert all(r["exact"] for r in rows["scope_m7"]) and sum(r["parsed"] for r in rows["blind"]) == M.CONTRAST_STEPS
    v = M.verdict_million(res["scope_m7"], {p: res[p] for p in M.CONTRAST_POLICIES})
    assert v[3][2] == "yes" and v[11][2] == "yes"                                           # lines 4 and 12 on the stand-in
    assert all(res[p]["rows"][0]["kind"] == res["scope_m7"]["rows"][0]["kind"] for p in res) and all(a["kind"] == b["kind"] for a, b in zip(rows["blind"], rows["gate"]))   # the same requests


def test_the_model_is_asked_for_constrained_output_with_the_rule_only_when_asked_and_seconds_are_kept_to_four_places():
    class Spy(Echo):
        flags = None
        def generate(self, messages, constrained, max_new_tokens=400):
            self.flags = (self.flags or []) + [constrained]; text, n, sec = super().generate(messages, constrained, max_new_tokens); return text, n, 1.23456789
    llm = Spy(5); r, _, _ = run(5, 12, llm=llm)
    assert llm.flags == [True] * 12 and all("Rule:" not in m[0]["content"] for m in llm.seen) and all((m[0]["role"], m[1]["role"]) == ("system", "user") for m in llm.seen)
    assert all(x["seconds"] == 1.2346 for x in r["rows"])
    llm2 = Spy(5); r2, _, p2 = run(5, 5, llm=llm2, hint_rules=True)
    assert all("Rule:" in m[0]["content"] for m in llm2.seen) and json.loads(open(p2).readline())["_meta"]["hint_rules"] is True


class Hosted(Echo):
    """A stand-in with what the hosted-model wrapper adds: thinking tokens, and answers that come from the saved reply file."""
    def __init__(self, seed, cached_at=(), think=7, **kw): super().__init__(seed, **kw); self.cached_at, self.think, self.last_think_tokens, self.from_disk = set(cached_at), think, 0, False
    def generate(self, messages, constrained, max_new_tokens=400):
        out = super().generate(messages, constrained, max_new_tokens); self.last_think_tokens = self.think; self.from_disk = self.n in self.cached_at; return out


def test_thinking_tokens_are_billed_and_answers_from_the_saved_reply_file_are_not():
    llm = Hosted(5, cached_at=(3, 4)); r, _, path = run(5, 30, llm=llm); rows, s = r["rows"], r["summary"]
    assert [x["cached"] for x in rows] == [t in (3, 4) for t in range(1, 31)] and all(x["think"] == 7 for x in rows)
    live = [x for x in rows if not x["cached"]]
    assert M.tokens_sent(rows) == sum(x["prompt_tokens"] + 11 + 7 for x in live) == s["tokens"] and s["cached_steps"] == 2 and s["think_tokens"] == 7 * 28 and s["output_tokens"] == 11 * 28
    assert s["prompt_tokens"] == sum(x["prompt_tokens"] for x in live) and s["tokens"] == s["prompt_tokens"] + s["output_tokens"] + s["think_tokens"]
    assert M._billed(rows[2]) == 0 and M._billed(rows[0]) == rows[0]["prompt_tokens"] + 18
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): M.print_million(s, "hosted")
    out = buf.getvalue(); assert f"{s['think_tokens']:,} thinking" in out and "2 steps answered from the saved reply file, not counted" in out
    plain, _, _ = run(5, 12); buf = io.StringIO()
    with contextlib.redirect_stdout(buf): M.print_million(plain["summary"], "plain")
    assert "thinking" not in buf.getvalue() and "saved reply file" not in buf.getvalue() and plain["summary"]["think_tokens"] == 0 and plain["summary"]["cached_steps"] == 0
    assert all(x["think"] == 0 and x["cached"] is False for x in plain["rows"])                             # a model without the two attributes
    v = M.verdict_million(r); assert v[0][2] == "yes"                                                       # the replay does not read the two fields


def test_the_token_cap_stops_before_the_next_call_resumes_cleanly_and_does_not_flag_a_finished_run():
    full, _, _ = run(5, 40, llm=Hosted(5)); per = [M._billed(x) for x in full["rows"]]
    cap = sum(per[:10]); p = tmp(); llm = Hosted(5)
    r = run_with(llm, 5, 40, "scope_m7", p, max_tokens=cap)
    assert len(r["rows"]) == 10 and r["stopped"] == "token_cap" and llm.calls == 10                          # reached exactly: no 11th call
    llm_b = Hosted(5, skip=10); r_b = run_with(llm_b, 5, 40, "scope_m7", p, max_tokens=cap)
    assert len(r_b["rows"]) == 10 and r_b["stopped"] == "token_cap" and llm_b.calls == 0                      # the same cap again: nothing more is asked
    r_c = run_with(Hosted(5, skip=10), 5, 40, "scope_m7", p, max_tokens=cap + 1)
    assert len(r_c["rows"]) == 11 and r_c["stopped"] == "token_cap"                                           # one token more: one more step, then it stops
    r_d = run_with(Hosted(5, skip=11), 5, 40, "scope_m7", p); assert len(r_d["rows"]) == 40 and r_d["stopped"] is None and [x["text"] for x in r_d["rows"]] == [x["text"] for x in full["rows"]]
    q = tmp(); r_e = run_with(Hosted(5), 5, 40, "scope_m7", q, max_tokens=sum(per)); assert len(r_e["rows"]) == 40 and r_e["stopped"] is None             # the cap is reached by the last step itself: not a stop
    q2 = tmp(); r_f = run_with(Hosted(5), 5, 40, "scope_m7", q2, max_tokens=sum(per) - 1); assert len(r_f["rows"]) == 40 and r_f["stopped"] is None         # reached before the last step only if a step is left to refuse
    q3 = tmp(); r_g = run_with(Hosted(5), 5, 40, "scope_m7", q3, max_tokens=sum(per[:39])); assert len(r_g["rows"]) == 39 and r_g["stopped"] == "token_cap"


def test_a_quota_stop_leaves_the_file_as_it_was_after_the_last_finished_step_and_the_next_day_carries_on():
    from axi.experiments.api_llm import QuotaStop
    class Limited(Echo):
        def generate(self, messages, constrained, max_new_tokens=400):
            if self.calls == 6: raise QuotaStop("Daily budget reached")
            return super().generate(messages, constrained, max_new_tokens)
    p = tmp(); llm = Limited(5)
    try: run_with(llm, 5, 20, "scope_m7", p); assert False
    except QuotaStop as e: assert "Daily budget" in str(e)
    assert len(open(p).read().splitlines()) == 7 and [json.loads(l)["t"] for l in open(p).read().splitlines()[1:]] == list(range(1, 7))
    r = run_with(Echo(5, skip=6), 5, 20, "scope_m7", p); full, _, _ = run(5, 20)
    assert [x["text"] for x in r["rows"]] == [x["text"] for x in full["rows"]] and r["summary"] == full["summary"] and r["stopped"] is None


def test_the_harness_over_the_hosted_model_wrapper_counts_what_was_asked_and_a_second_pass_from_the_reply_file_counts_nothing():
    from axi.experiments.api_llm import DiskCachedLLM
    class Backend:
        def __init__(self, seed): self.s, self.calls = M.LiveStream(seed), 0
        def call(self, messages):
            st, _ = self.s.next(); self.calls += 1
            return dict(text=st["ideal"], tokens=11, think_tokens=4, prompt_tokens=(len(messages[0]["content"]) + len(messages[1]["content"])) // 3, finish="STOP")
    cache = tmp(); b = Backend(5); wrapper = DiskCachedLLM(b, cache, dict(model="stand-in"), daily_limit=None)
    r = M.run_million(wrapper, 5, 30, "scope_m7", tmp(), meta=dict(model="stand-in"))
    live = [x for x in r["rows"] if not x["cached"]]
    assert b.calls == len(live) == wrapper.calls_this_session and all(x["think"] == 4 for x in r["rows"]) and r["summary"]["think_tokens"] == 4 * len(live) and wrapper.calls_today() == b.calls
    assert r["summary"]["tokens"] == r["summary"]["prompt_tokens"] + 11 * len(live) + 4 * len(live)
    wrapper2 = DiskCachedLLM(Backend(5), cache, dict(model="stand-in"), daily_limit=None)                       # another day: the reply file is there, the harness file is not
    r2 = M.run_million(wrapper2, 5, 30, "scope_m7", tmp(), meta=dict(model="stand-in"))
    assert all(x["cached"] for x in r2["rows"]) and r2["summary"]["tokens"] == 0 and r2["summary"]["cached_steps"] == 30 and wrapper2.calls_this_session == 0
    assert [x["text"] for x in r2["rows"]] == [x["text"] for x in r["rows"]] and M.verdict_million(r2)[0][2] == "yes" and M.verdict_million(r2)[4][2] == "NO"
    assert all(x["think"] == 4 for x in r2["rows"]) and r2["summary"]["think_tokens"] == 0                       # a reply read back keeps the thinking count it was saved with, and still counts nothing
    limited = DiskCachedLLM(Backend(5), tmp(), dict(model="stand-in"), daily_limit=5)
    from axi.experiments.api_llm import QuotaStop
    path = tmp()
    try: M.run_million(limited, 5, 30, "scope_m7", path, meta=dict(model="stand-in")); assert False
    except QuotaStop as e: assert "Daily budget reached" in str(e)
    assert len(open(path).read().splitlines()) == 6                                                              # 5 steps, then the stop


def test_the_cost_estimate_and_the_short_check_on_hand_built_numbers():
    assert abs(M.estimate_cost(1_000_000, 1_000_000, 0.3, 2.5) - 2.8) < 1e-12 and M.estimate_cost(1e6, 1e6, None, 2.5) is None and M.estimate_cost(1e6, 1e6, 0.3, None) is None
    assert abs(M.estimate_cost(1_150_000, 50_000, 0.30, 2.50) - 0.47) < 1e-9 and M.estimate_cost(0, 0, 1.0, 1.0) == 0.0
    ok = M.summarize_million(_rows(20, prompt_tokens=300, tokens=12, seconds=0.5)); r = M.smoke_report(ok, 3600, 0.3, 2.5)
    assert r["steps"] == 20 and r["tokens_per_step"] == 312 and r["think_per_step"] == 0 and r["seconds_per_step"] == 0.5 and r["projected_tokens"] == 312 * 3600 and r["projected_minutes"] == 30.0
    assert abs(r["projected_cost"] - (300 * 3600 * 0.3 + 12 * 3600 * 2.5) / 1e6) < 1e-9 and r["advice"] == []
    assert M.smoke_report(ok)["projected_tokens"] == 312 * M.STEPS and any("PRICE_IN" in a for a in M.smoke_report(ok)["advice"]) and M.smoke_report(ok)["projected_cost"] is None
    t20 = M.smoke_report(M.summarize_million(_rows(20, prompt_tokens=300, tokens=12, think=20)), 3600, 0.3, 2.5); assert t20["advice"] == [] and t20["think_per_step"] == 20 and t20["tokens_per_step"] == 332
    t21 = M.smoke_report(M.summarize_million(_rows(20, prompt_tokens=300, tokens=12, think=21)), 3600, 0.3, 2.5)
    assert len(t21["advice"]) == 1 and "21 thinking tokens a step" in t21["advice"][0] and "minimal" in t21["advice"][0]
    assert abs(t21["projected_cost"] - (300 * 3600 * 0.3 + (12 + 21) * 3600 * 2.5) / 1e6) < 1e-9                       # thinking is billed as output
    big = M.smoke_report(M.summarize_million(_rows(20, prompt_tokens=550, tokens=12)), 3600, 0.3, 2.5); assert any("above the cap" in a for a in big["advice"]) and big["projected_tokens"] > M.GEMINI_TOKEN_CAP
    edge = M.smoke_report(M.summarize_million(_rows(20, prompt_tokens=M.GEMINI_TOKEN_CAP // 3600, tokens=0)), 3600, 0.3, 2.5); assert not any("above the cap" in a for a in edge["advice"])       # exactly at the cap is not above it
    at = M.smoke_report(M.summarize_million(_rows(20, prompt_tokens=1000, tokens=0)), 2000, 0.3, 2.5); assert at["projected_tokens"] == M.GEMINI_TOKEN_CAP and not any("above the cap" in a for a in at["advice"])      # exactly the cap
    over = M.smoke_report(M.summarize_million(_rows(20, prompt_tokens=1001, tokens=0)), 2000, 0.3, 2.5); assert any("above the cap" in a for a in over["advice"])                                                           # one token a step more
    cached = _rows(20, prompt_tokens=300, tokens=12); 
    for x in cached[:5]: x["cached"] = True
    c = M.smoke_report(M.summarize_million(cached), 3600, 0.3, 2.5); assert c["steps"] == 15 and c["tokens_per_step"] == 312
    allc = _rows(5, cached=True); a = M.smoke_report(M.summarize_million(allc)); assert a == dict(steps=0, advice=a["advice"]) and "saved reply file" in a["advice"][0]
    assert M.GEMINI_MODEL == "gemini-3.5-flash-lite" and M.GEMINI_TOKEN_CAP == 2_000_000 and M.GEMINI_FREE_DAILY == 480 and M.THINKING_LIMIT == 20


def test_the_scope_filter_trims_an_unasked_edge_delete_and_a_wrong_coordinate_leaves_an_omission():
    def slip(n, kind, ideal): return ideal + " DEL[(0,0)->(0,1):dep#1]" if n % 9 == 0 else ideal
    r, _, _ = run(13, 90, llm=Echo(13, tweak=slip)); rows = r["rows"]
    assert sum(1 for x in rows if x["dropped"] > 0) > 0 and all(not x["corrupted"] for x in rows) and r["summary"]["corrupted"] == 0
    junk, _, _ = run(13, 40, llm=Echo(13, tweak=lambda n, k, i: "not a delta" if n == 5 else i)); jr = junk["rows"]
    assert jr[4]["parsed"] is False and "dropped" not in jr[4] and jr[4]["wrong"] is True and jr[4]["distance"] > 0 and junk["summary"]["refused"] == 1 and all(x["parsed"] for i, x in enumerate(jr) if i != 4)


# ---------------------------------------------------------------- 3 resume
def test_a_file_cut_in_the_middle_of_a_line_continues_to_the_same_rows_and_makes_only_the_missing_calls():
    full, _, p0 = run(5, 80, llm=Echo(5, tweak=forgetful))
    p = tmp(); lines = open(p0).read().splitlines(); open(p, "w").write("\n".join(lines[:41]) + "\n" + lines[41][:25])        # 40 steps, then half of the 41st
    llm = Echo(5, skip=40, tweak=forgetful); r = run_with(llm, 5, 80, "scope_m7", p)
    assert llm.calls == 40 and [x["text"] for x in r["rows"]] == [x["text"] for x in full["rows"]] and r["summary"] == full["summary"]
    assert [json.loads(l) for l in open(p).read().splitlines()[1:]] == r["rows"]
    llm2 = Echo(5, skip=80); r2 = run_with(llm2, 5, 80, "scope_m7", p); assert llm2.calls == 0 and r2["rows"] == r["rows"]                    # a finished file makes no call


def test_a_file_of_another_run_a_longer_file_a_doctored_flag_and_a_doctored_kind_are_refused():
    _, _, p = run(5, 30)
    for kw in (dict(seed=6), dict(policy="gate"), dict(steps=31)):
        a = dict(seed=5, steps=30, policy="scope_m7"); a.update(kw)
        try: M.run_million(Echo(a["seed"]), a["seed"], a["steps"], a["policy"], p, meta=dict(model="stand-in")); assert False, kw
        except ValueError as e: assert "different run" in str(e)
    try: M.run_million(Echo(5), 5, 30, "scope_m7", p, meta=dict(model="other")); assert False
    except ValueError: pass
    try: M.run_million(Echo(5), 5, 30, "scope_m7", p, meta=dict(model="stand-in"), hint_rules=True); assert False
    except ValueError: pass
    lines = open(p).read().splitlines(); meta = dict(model="stand-in")
    for field, new, err in (("distance", 3, "distance"), ("exact", False, "exact"), ("kind", "add_node" if json.loads(lines[1])["kind"] != "add_node" else "delete_leaf", "kind")):
        q = tmp(); ls = list(lines[:31]); rec = json.loads(ls[10]); rec[field] = new; ls[10] = json.dumps(rec); open(q, "w").write("\n".join(ls) + "\n")
        try: M.run_million(Echo(5), 5, 30, "scope_m7", q, meta=meta); assert False, field
        except M.ReplayMismatch as e: assert err in str(e) and "step 10" in str(e), str(e)
    q = tmp(); ls = list(lines[:31]); rec = json.loads(ls[10]); rec["text"] = "garbage"; ls[10] = json.dumps(rec); open(q, "w").write("\n".join(ls) + "\n")
    try: M.run_million(Echo(5), 5, 30, "scope_m7", q, meta=meta); assert False
    except M.ReplayMismatch: pass                                                                            # the saved reply no longer gives the saved flags
    rows = [json.loads(l) for l in lines[1:31]]
    s, held, bad = M.replay(rows, 5, "scope_m7", strict=True); assert bad == [] and s.live is not None
    rows[3]["distance"] = 99; rows[8]["kind"] = "x"; _, _, bad = M.replay(rows, 5, "scope_m7", strict=False); assert bad == [4, 9]


def test_a_stream_that_ends_stops_the_run_and_says_so_and_a_replay_past_the_end_is_refused():
    with mock.patch.object(M.LiveStream, "next", autospec=True, side_effect=lambda self, _c=[0]: None):
        r = M.run_million(Echo(1), 1, 10, "gate", tmp(), meta={}); assert r["stopped"] == "stream" and r["rows"] == []
    real = M.LiveStream.next; count = {"n": 0}
    def once_none(self):
        count["n"] += 1
        return None if count["n"] == 4 else real(self)
    class Dummy:
        last_prompt_tokens = 5
        def generate(self, messages, constrained, max_new_tokens=400): return "ADD[0,0:1]", 3, 0.1
    with mock.patch.object(M.LiveStream, "next", once_none):
        r = M.run_million(Dummy(), 1, 10, "gate", tmp(), meta={}); assert r["stopped"] == "stream" and len(r["rows"]) == 3, (r["stopped"], len(r["rows"]))      # it stops at the first end, it does not go on
    p = tmp(); run(5, 6, path=p)
    rows = [json.loads(l) for l in open(p).read().splitlines()[1:]]
    with mock.patch.object(M.LiveStream, "next", autospec=True, side_effect=lambda self: None):
        try: M.replay(rows, 5, "scope_m7"); assert False
        except M.ReplayMismatch as e: assert "ended before step 1" in str(e)


def test_the_probe_is_stored_every_nth_step_and_the_progress_callback_sees_every_step():
    seen = []; r, _, _ = run(5, 35, probe=lambda: {"allocated_mb": 1.5}, probe_every=10, progress=lambda t, n, rec: seen.append((t, n, rec["t"])))
    assert [x["t"] for x in r["rows"] if "probe" in x] == [10, 20, 30] and seen == [(t, 35, t) for t in range(1, 36)]
    assert r["summary"]["probes"] == [(10, {"allocated_mb": 1.5}), (20, {"allocated_mb": 1.5}), (30, {"allocated_mb": 1.5})]
    r2, _, _ = run(5, 12); assert all("probe" not in x for x in r2["rows"]) and r2["summary"]["probes"] == []
    r4, _, p4 = run(5, 25, probe=lambda: {"allocated_mb": 1.0}, probe_every=10); llm4 = Echo(5, skip=25); r5 = run_with(llm4, 5, 25, "scope_m7", p4, probe=lambda: {"allocated_mb": 9.0}, probe_every=10)
    assert llm4.calls == 0 and r5["rows"] == r4["rows"]                                                       # a file with probe records replays: the probe is not one of the replayed flags
    rows = json.loads(json.dumps(r4["rows"])); del rows[4]["dropped"]
    try: M.replay(rows, 5, "scope_m7"); assert False
    except M.ReplayMismatch as e: assert "dropped" in str(e) and "step 5" in str(e)
    r3, _, _ = run(5, 205, probe=lambda: {"allocated_mb": 2.0}); assert [x["t"] for x in r3["rows"] if "probe" in x] == [100, 200]          # default: every 100 steps


def test_the_gpu_probe_frees_what_it_can_and_reports_megabytes():
    calls = []
    class Cuda:
        def memory_allocated(self): return 5_000_000_000
        def memory_reserved(self): return 6_000_000_000
        def empty_cache(self): calls.append("empty")
    fake = mock.MagicMock(); fake.cuda = Cuda()
    with mock.patch.dict(sys.modules, {"torch": fake}): out = M.gpu_probe()
    assert out == {"allocated_mb": 5000.0, "reserved_mb": 6000.0} and calls == ["empty"]


# ---------------------------------------------------------------- 4 the numbers
def test_the_flags_of_a_step_on_hand_built_cases():
    g = Graph()
    for n, v in (((0, 0), 1), ((0, 1), 2), ((1, 1), 3)): g.add_node(n, v)
    g.add_edge((0, 0), (0, 1), "dep", 4)
    oracle = copy_graph(g)
    st = dict(kind="delete_dep", instruction="Delete the service at (0,1).", ideal="DEL[0,1] DEL[(0,0)->(0,1):dep#4]")
    after_o = copy_graph(g); after_o.edges.clear(); after_o._inc.clear(); del after_o.nodes[(0, 1)]
    good = strict_parse("DEL[0,1] DEL[(0,0)->(0,1):dep#4]"); a, f = M.step_fields("scope_m7", g, st, oracle, after_o, good)
    assert same_graph(a, after_o) and f["applicable"] and f["wrong"] is False and f["exact"] and f["distance"] == 0 and f["start_same"] and f["parsed"] and not f["corrupted"] and f["dropped"] == 0
    assert (f["nodes"], f["edges"], f["oracle_nodes"], f["oracle_edges"]) == (2, 0, 2, 0)
    forget = strict_parse("DEL[0,1]"); a, f = M.step_fields("blind", g, st, oracle, after_o, forget)
    assert f["corrupted"] and f["wrong"] is True and f["distance"] == 1 and not f["exact"] and f["edges"] == 1 and "dropped" not in f                       # a dangling edge: one item off
    a, f = M.step_fields("gate", g, st, oracle, after_o, forget); assert same_graph(a, g) and f["wrong"] is True and f["distance"] == 2 and not f["corrupted"]
    a, f = M.step_fields("scope_m7", g, st, oracle, after_o, None); assert f["parsed"] is False and f["wrong"] is True and "dropped" not in f and same_graph(a, g)
    held = copy_graph(g); del held.nodes[(1, 1)]                                                  # a held state that lacks a service the oracle has
    a, f = M.step_fields("scope_m7", held, st, oracle, after_o, good); assert f["start_same"] is False and f["applicable"] and f["wrong"] is False and f["distance"] == 1
    st2 = dict(kind="delete_edge", instruction="Remove the edge from (0,0) to (0,1) with relation dep and weight 4.", ideal="DEL[(0,0)->(0,1):dep#4]")
    h2 = copy_graph(g); h2.edges.clear(); h2._inc.clear()                                          # the held state no longer has that edge: the request cannot be applied as written
    a, f = M.step_fields("scope_m7", h2, st2, oracle, oracle, strict_parse("DEL[(0,0)->(0,1):dep#4]")); assert f["applicable"] is False and f["wrong"] is None
    a, f = M.step_fields("scope_m7", g, st, oracle, after_o, strict_parse("DEL[0,1] DEL[(0,0)->(0,1):dep#4] DEL[1,1]")); assert f["dropped"] == 1 and f["exact"]


def _rows(n, **kw):
    rows = []
    for i in range(n):
        r = dict(t=i + 1, kind="add_node", text="", tokens=10, prompt_tokens=300, seconds=1.0, parsed=True, corrupted=False, distance=0, exact=True, start_same=True, applicable=True, wrong=False,
                 nodes=9, edges=5, oracle_nodes=9, oracle_edges=5, dropped=0)
        r.update({k: (v(i) if callable(v) else v) for k, v in kw.items()}); rows.append(r)
    return rows


def test_tenths_halves_wrong_rate_and_totals_on_hand_built_rows():
    for n in (0, 5, 99, 100, 105, 3600):
        rows = _rows(n); parts = [M.block(rows, i) for i in range(10)]
        assert sum(len(p) for p in parts) == n and [r["t"] for p in parts for r in p] == list(range(1, n + 1))
    rows = _rows(100, distance=lambda i: i, exact=lambda i: i % 4 != 0, prompt_tokens=lambda i: 100 + i, seconds=lambda i: i / 10, parsed=lambda i: i != 3, start_same=lambda i: i % 2 == 0,
                 applicable=lambda i: i % 5 != 0, wrong=lambda i: i % 10 == 2, dropped=lambda i: i % 3 == 0)
    s = M.summarize_million(rows); t0, t9 = s["tenths"][0], s["tenths"][9]
    assert (t0["first"], t0["last"], t0["steps"], t9["first"], t9["last"]) == (1, 10, 10, 91, 100) and t0["distance"] == 4.5 and t9["distance"] == 94.5
    assert t0["exact"] == 0.7 and t0["prompt_tokens"] == 104.5 and abs(t0["seconds"] - 0.45) < 1e-9 and t0["refused"] == 0.1 and t9["refused"] == 0 and abs(t0["dropped"] - 0.4) < 1e-9
    assert M.wrong_rate(rows[:10]) == (1, 4) and M.wrong_rate(rows) == (10, 40) and s["halves"] == [M.wrong_rate(rows[:50]), M.wrong_rate(rows[50:])] and s["halves"] == [(5, 20), (5, 20)]
    assert s["steps"] == 100 and s["tokens"] == sum(100 + i for i in range(100)) + 1000 == M.tokens_sent(rows) and s["refused"] == 1 and s["exact"] == 75 and s["max_distance"] == 99
    assert s["prompt_tokens"] == sum(100 + i for i in range(100)) and s["output_tokens"] == 1000 and s["corrupted"] == 0 and s["max_nodes"] == 9 and s["max_edges"] == 5
    assert M.summarize_million(_rows(100, wrong=lambda i: i == 49))["halves"] == [(1, 50), (0, 50)] and M.summarize_million(_rows(100, wrong=lambda i: i == 50))["halves"] == [(0, 50), (1, 50)]         # row 49 ends the first half, row 50 starts the second
    assert M.summarize_million(_rows(101, wrong=lambda i: i == 50))["halves"] == [(0, 50), (1, 51)] and M.summarize_million(_rows(101, wrong=lambda i: i == 49))["halves"] == [(1, 50), (0, 51)]
    nod = [{k: v for k, v in r.items() if k != "dropped"} for r in _rows(20)]; assert M.summarize_million(nod)["tenths"][0]["dropped"] == 0.0                 # a policy without a scope filter has no `dropped`
    assert M.tokens_sent(_rows(3, prompt_tokens=None)) == 30                                                                         # a missing prompt count adds nothing
    e = M.summarize_million([]); assert e["steps"] == 0 and e["tokens"] == 0 and e["max_distance"] == 0 and len(e["tenths"]) == 10 and e["tenths"][0]["distance"] is None
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): M.print_million(s, "unit"); M.print_million(e, "empty")
    skew = M.summarize_million(_rows(100, wrong=lambda i: i in (3, 60, 61), seconds=60.0)); buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2): M.print_million(skew, "skew")
    o2 = buf2.getvalue(); assert "first half 1/50, second half 2/50" in o2 and f"{skew['prompt_tokens']:,} prompt + {skew['output_tokens']:,} reply" in o2 and "100 model minutes" in o2 and "30,000 prompt + 1,000 reply" in o2
    out = buf.getvalue(); assert "unit: 100 steps" in out and "1-10" in out and "91-100" in out and "first half 5/20, second half 5/20" in out and "empty: 0 steps" in out


# ---------------------------------------------------------------- 5 the reading rules
def _genuine(seed=5, steps=100, **kw):
    """A genuine run of a stand-in, made small; its rows are then doctored where a test needs a boundary."""
    r, _, _ = run(seed, steps, **kw); return json.loads(json.dumps(r))


def _lines(run1, contrast=None, run2=None): return {i + 1: ans for i, (_, _, ans) in enumerate(M.verdict_million(run1, contrast, run2))}


def _set(run_, idx, **kw):
    for i in idx: run_["rows"][i].update(kw)


def test_an_untouched_genuine_run_passes_the_lines_that_are_about_one_run_and_the_others_say_not_run():
    base = _genuine(); v = _lines(base)
    assert [v[i] for i in (1, 2, 3, 6, 7, 8, 9, 10, 11)] == ["yes"] * 9 and v[5] == "NO" and v[4] == v[12] == v[13] == "not run"          # 100 steps cannot send a million tokens
    assert [t for t, _, _ in M.verdict_million(base)] == ["guard", "guard", "instrument", "instrument", "claim", "claim", "claim", "claim", "claim", "claim", "claim", "claim", "claim"]
    assert [l for _, l, _ in M.verdict_million(base)] == [l for _, l in M.LINES_M]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): M.print_verdict_million(M.verdict_million(base))
    out = buf.getvalue(); assert out.count("\nyes ") == 9 and "NO      5. [claim] tokens sent" in out and "not run 13. [claim]" in out
    e = _lines({"rows": [], "meta": base["meta"]}); assert all(e[i] == "NO" for i in (1, 2, 3, 5, 6, 7, 8, 9, 10, 11))


def test_line_1_flips_on_a_missing_step_a_doctored_flag_and_a_changed_step_number():
    base = _genuine(); assert _lines(base)[1] == "yes"
    short = json.loads(json.dumps(base)); short["rows"].pop(); assert _lines(short)[1] == "NO"
    for field, new in (("distance", 5), ("exact", False), ("wrong", True), ("start_same", False), ("kind", "x")):
        r = json.loads(json.dumps(base)); r["rows"][40][field] = new; assert _lines(r)[1] == "NO", field
    r = json.loads(json.dumps(base)); r["rows"][40]["t"] = 99; assert _lines(r)[1] == "NO"
    r = json.loads(json.dumps(base)); r["meta"]["steps"] = 101; assert _lines(r)[1] == "NO"
    r = json.loads(json.dumps(base)); r["rows"][40]["text"] = "ADD[0,0:1]"; assert _lines(r)[1] == "NO"                  # another reply: the saved flags no longer follow from it


def test_line_2_and_3_corruption_and_the_two_percent_of_refused_replies():
    base = _genuine(); r = json.loads(json.dumps(base)); _set(r, [57], corrupted=True); assert _lines(r)[2] == "NO"
    for k in (0, 99): r = json.loads(json.dumps(base)); _set(r, [k], corrupted=True); assert _lines(r)[2] == "NO", k                                      # the first and the last step are read too
    r = json.loads(json.dumps(base)); _set(r, [1, 50], parsed=False); assert _lines(r)[3] == "yes"                       # 2 of 100: exactly the limit
    r = json.loads(json.dumps(base)); _set(r, [1, 50, 99], parsed=False); assert _lines(r)[3] == "NO"


def test_line_5_tokens_flip_at_exactly_one_million():
    base = _genuine(steps=100); r = json.loads(json.dumps(base))
    for x in r["rows"]: x["prompt_tokens"], x["tokens"] = 9990, 10
    assert M.tokens_sent(r["rows"]) == 1_000_000 and _lines(r)[5] == "yes"
    r["rows"][0]["prompt_tokens"] -= 1; assert M.tokens_sent(r["rows"]) == 999_999 and _lines(r)[5] == "NO"


def test_line_6_the_distance_may_rise_by_at_most_one_item_between_the_first_and_the_last_tenth():
    base = _genuine(); r = json.loads(json.dumps(base)); _set(r, range(90, 100), distance=1); assert _lines(r)[6] == "yes"                 # 0.0 -> 1.0
    r["rows"][95]["distance"] = 2; assert _lines(r)[6] == "NO"                                                                           # 0.0 -> 1.1
    _set(r, range(0, 10), distance=1); assert _lines(r)[6] == "yes"                                                                      # 1.0 -> 1.1
    _set(r, range(0, 10), distance=3); _set(r, range(90, 100), distance=4); assert _lines(r)[6] == "yes"                                 # 3.0 -> 4.0
    r2 = json.loads(json.dumps(base)); _set(r2, range(10, 90), distance=50); assert _lines(r2)[6] == "yes"                                # the middle is not read by this line


def test_line_7_the_distance_may_reach_ten_and_not_eleven_at_any_step():
    base = _genuine(); r = json.loads(json.dumps(base)); _set(r, [33], distance=10); assert _lines(r)[7] == "yes"
    _set(r, [33], distance=11); assert _lines(r)[7] == "NO"


def test_line_8_eight_of_ten_steps_of_the_last_tenth_may_differ_from_the_oracle_two_may_not_seven_may_not():
    base = _genuine(); r = json.loads(json.dumps(base)); _set(r, [90, 91], exact=False); assert _lines(r)[8] == "yes"
    _set(r, [92], exact=False); assert _lines(r)[8] == "NO"
    r2 = json.loads(json.dumps(base)); _set(r2, [89], exact=False); _set(r2, range(0, 90), exact=False); assert _lines(r2)[8] == "yes"        # only the last tenth is read


def test_line_9_the_wrong_asks_of_the_second_half_may_exceed_the_first_halfs_share_by_two_points():
    base = _genuine(); r = json.loads(json.dumps(base)); _set(r, range(0, 5), wrong=True); _set(r, range(50, 56), wrong=True); assert _lines(r)[9] == "yes"        # 5/50 and 6/50
    _set(r, [56], wrong=True); assert _lines(r)[9] == "NO"                                                                                                       # 7/50
    r = json.loads(json.dumps(base)); _set(r, range(0, 100), start_same=False); assert _lines(r)[9] == "NO"                                                       # nothing counted: not read as yes
    r = json.loads(json.dumps(base)); _set(r, range(0, 100), applicable=False, wrong=None); assert _lines(r)[9] == "NO"
    r = json.loads(json.dumps(base)); _set(r, range(0, 50), start_same=False); assert _lines(r)[9] == "NO"                                                         # nothing in the first half
    r = json.loads(json.dumps(base)); _set(r, range(50, 100), start_same=False); _set(r, [3], wrong=True); assert _lines(r)[9] == "NO"                           # nothing in the second half
    r = json.loads(json.dumps(base)); _set(r, [0, 1, 2, 3, 4], start_same=False); _set(r, range(50, 55), wrong=True); assert _lines(r)[9] == "NO"                  # 0/45 first, 5/50 second: 10 points more
    r = json.loads(json.dumps(base)); _set(r, [0, 1, 2, 3, 4], start_same=False); _set(r, range(0, 5), wrong=True); _set(r, range(50, 51), wrong=True); assert _lines(r)[9] == "yes"      # wrong rows that do not start from the oracle's state are not counted


def test_line_10_and_11_the_prompt_tokens_and_the_seconds_may_grow_to_one_and_a_half_times():
    base = _genuine(); r = json.loads(json.dumps(base)); _set(r, range(0, 10), prompt_tokens=200, seconds=2.0); _set(r, range(90, 100), prompt_tokens=300, seconds=3.0)
    v = _lines(r); assert v[10] == "yes" and v[11] == "yes"
    _set(r, [95], prompt_tokens=310, seconds=3.1); v = _lines(r); assert v[10] == "NO" and v[11] == "NO"
    r2 = json.loads(json.dumps(base)); _set(r2, range(10, 90), prompt_tokens=5000, seconds=50.0); v = _lines(r2); assert v[10] == "yes" and v[11] == "yes"                 # the middle is not read
    r3 = json.loads(json.dumps(base)); _set(r3, range(0, 10), prompt_tokens=0); v = _lines(r3); assert v[10] == "NO"                                                      # no first-tenth tokens: not read as yes
    r4 = json.loads(json.dumps(base)); _set(r4, range(0, 10), seconds=0.0); assert _lines(r4)[11] == "NO"
    r5 = json.loads(json.dumps(base)); _set(r5, range(0, 10), prompt_tokens=0); _set(r5, range(90, 100), prompt_tokens=0); assert _lines(r5)[10] == "NO"                  # nothing counted at either end: not read as flat
    r6 = json.loads(json.dumps(base)); _set(r6, range(0, 10), seconds=0.0); _set(r6, range(90, 100), seconds=0.0); assert _lines(r6)[11] == "NO"


def _contrast(corrupted_blind=150, d_scope=0.0, d_gate=1.0, d_blind=1.0):
    mk = lambda **kw: {"rows": _rows(300, **kw), "meta": {}}
    return {"blind": mk(corrupted=lambda i: i < corrupted_blind, distance=d_blind), "gate": mk(distance=d_gate), "repair_m6": mk()}


def test_line_4_blind_apply_must_be_corrupted_on_half_of_its_steps_and_line_12_scope_must_be_strictly_closer():
    base = _genuine(steps=300); assert _lines(base, _contrast())[4] == "yes" and _lines(base, _contrast(149))[4] == "NO"
    assert _lines(base, _contrast())[12] == "yes"
    assert _lines(base, _contrast(d_gate=0.0))[12] == "NO" and _lines(base, _contrast(d_blind=0.0))[12] == "NO"                       # equal is not smaller
    r = json.loads(json.dumps(base)); _set(r, range(200, 300), distance=1); assert _lines(r, _contrast())[12] == "NO"                  # 1.0 against 1.0
    r = json.loads(json.dumps(base)); _set(r, range(0, 200), distance=40); assert _lines(r, _contrast())[12] == "yes"                   # only steps 201 to 300 are read
    def windows(gate_early, gate_late, blind_early, blind_late):
        mk = lambda e, l: {"rows": _rows(300, distance=lambda i: e if i < 200 else l), "meta": {}}
        c = {"blind": mk(blind_early, blind_late), "gate": mk(gate_early, gate_late), "repair_m6": {"rows": _rows(300), "meta": {}}}
        for x in c["blind"]["rows"]: x["corrupted"] = True
        return c
    one = json.loads(json.dumps(base)); _set(one, range(200, 300), distance=1)
    assert _lines(one, windows(0, 5, 0, 5))[12] == "yes"                                                                              # large only in steps 201 to 300: read
    assert _lines(one, windows(5, 0, 0, 5))[12] == "NO" and _lines(one, windows(0, 5, 5, 0))[12] == "NO"                              # large only in steps 1 to 200: not read
    part = _contrast(); part["gate"]["rows"].pop(); v = _lines(base, part); assert v[4] == v[12] == "not run"                          # 299 steps of one policy: not a complete contrast
    missing = _contrast(); del missing["repair_m6"]; v = _lines(base, missing); assert v[4] == v[12] == "not run"
    short = _genuine(steps=250); assert _lines(short, _contrast())[12] == "NO"


def test_line_13_needs_run_2_to_pass_the_per_run_lines_and_to_bring_the_total_to_two_million():
    r1, r2 = _genuine(5), _genuine(6)
    for r in (r1, r2):
        for x in r["rows"]: x["prompt_tokens"], x["tokens"] = 9990, 10
    assert _lines(r1, run2=r2)[13] == "yes" and _lines(r1)[13] == "not run"
    r2["rows"][0]["prompt_tokens"] -= 1; assert _lines(r1, run2=r2)[13] == "NO"                                                    # 1,999,999
    r2["rows"][0]["prompt_tokens"] += 1
    for line, mut in ((1, dict(kind="x")), (2, dict(corrupted=True)), (6, None), (7, dict(distance=11)), (8, None), (9, None), (10, None), (11, None)):
        r = json.loads(json.dumps(r2))
        if line == 1: r["rows"][3].update(mut)
        elif line == 2: r["rows"][3].update(mut)
        elif line == 7: r["rows"][3].update(mut)
        elif line == 6: _set(r, range(90, 100), distance=2)
        elif line == 8: _set(r, range(90, 100), exact=False)
        elif line == 9: _set(r, range(50, 100), wrong=True)
        elif line == 10: _set(r, range(90, 100), prompt_tokens=20000)
        elif line == 11: _set(r, range(90, 100), seconds=9.0)
        assert _lines(r1, run2=r)[13] == "NO", line
    r = json.loads(json.dumps(r2)); _set(r, [3, 4, 5], parsed=False); assert _lines(r1, run2=r)[13] == "NO"                                      # line 3 on run 2
    weak1 = json.loads(json.dumps(r1))
    for x in weak1["rows"]: x["prompt_tokens"], x["tokens"] = 100, 10
    assert _lines(weak1, run2=r2)[13] == "NO" and _lines(weak1, run2=r2)[5] == "NO"                                                 # run 1 alone sent too little: the pair does not reach two million
    only2 = json.loads(json.dumps(r2))
    for x in only2["rows"]: x["prompt_tokens"], x["tokens"] = 100, 10
    big = json.loads(json.dumps(r1))
    for x in big["rows"]: x["prompt_tokens"], x["tokens"] = 19990, 10
    assert _lines(big, run2=only2)[13] == "yes"                                                                                       # line 5 is not required of run 2 on its own


# ---------------------------------------------------------------- 6 the page
def test_the_page_and_the_module_agree_and_the_text_above_the_result_is_what_was_written_before_the_run():
    text = open(PAGE).read(); assert text.count(RESULT_MARK) <= 1; head = text[:text.index(RESULT_MARK)] if RESULT_MARK in text else text
    assert _h(head) == PAGE_SHA256, "the predictions and reading rules must not change after the run; later notes go below them"
    for i, (tag, line) in enumerate(M.LINES_M, 1): assert f"\n{i}. [{tag}] {line}\n" in head, i
    assert head.count("\n1. [guard] Run 1 reaches step 3,600") == 1 and "Thirteen lines" in head and len(M.LINES_M) == 13
    for needle in (f"stream seed {M.SEEDS[0]}", f"stream seed {M.SEEDS[1]}", "3,600 requests", f"first {M.CONTRAST_STEPS} requests", "Qwen/Qwen2.5-7B-Instruct", "at most 400 new tokens", "NF4 4-bit",
                   "add service x3, add edge x4, delete with dependants x1, delete without x1, delete edge x1, delete service and edge x1", "`million.verdict_million`", "`million.replay`", "`million.LiveStream`",
                   "GPU memory every 100 steps", "about **30 %**", "about **85 %**", "at most 14 and 26", "(97 %)", "(99 %)", "(90 %)", "(98 %)", "(80 %)", "(85 %)", "(50 %)", "(95 %)", "(40 %)"):
        assert needle in head, needle
    lines_text = "\n".join(l for _, l in M.LINES_M)
    consts = [("at most 2 percent", M.REFUSED_MAX == 0.02), ("at least 50 percent", M.BLIND_MIN == 0.5), ("at least 1,000,000", M.TOKENS_MIN == 10 ** 6), ("at most 1.0 above", M.DRIFT_ADD == 1.0),
              ("never exceeded 10 items", M.LIMIT_DISTANCE == 10), ("at least 80 percent", M.EXACT_MIN == 0.8), ("at most 2 percentage points", M.WORSE_ADD == 0.02), ("at most 1.5 times", M.FLAT_TOKENS == M.FLAT_SECONDS == 1.5),
              ("at least 2,000,000", M.TOKENS_MIN_TWO == 2 * 10 ** 6), ("steps 201 to 300", M.CONTRAST_STEPS == 300), ("step 3,600", M.STEPS == 3600), ("last tenth", M.BLOCKS == 10), ("first 300 steps", M.CONTRAST_STEPS == 300)]
    for needle, ok in consts: assert ok, needle; assert needle in lines_text, needle
    assert lines_text.count("at most 1.5 times") == 2 and "every 100" in head.replace("every 100 steps", "every 100") and M.PROBE_EVERY == 100


def test_the_page_predictions_are_numbered_like_the_lines_and_the_lines_and_probabilities_are_complete():
    head = open(PAGE).read().split(RESULT_MARK)[0]; pred = head.split("## Predictions")[1].split("## Reading rules")[0]
    nums = [int(m.group(1)) for l in pred.splitlines() for m in [re.match(r"^(\d+)\. ", l)] if m]
    assert nums == list(range(1, 15)), nums
    for i in range(1, 14): assert f"\n{i}. [" in pred, i
    assert "\n14. Not reading-rule lines." in pred and pred.count("[guard]") == 2 and pred.count("[instrument]") == 2 and pred.count("[claim]") == 9
    assert M.MODEL == "Qwen/Qwen2.5-7B-Instruct" and M.MAIN_POLICY == "scope_m7" and M.CONTRAST_POLICIES == ("blind", "gate", "repair_m6")


# ---------------------------------------------------------------- 7 the hosted-model page (docs/MILLION_TOKEN_TEST_GEMINI.md) and its notebook
PAGE_G = os.path.join(HERE, "..", "..", "docs", "MILLION_TOKEN_TEST_GEMINI.md")
PAGE_G_SHA256 = "92732083b26327a6e173ca3494c943b888cd87afe3df9e2bfe0576932fc7967c"          # filled in when the page was written, before any run
NB_G = os.path.join(HERE, "..", "colab", "axi_million_gemini.ipynb")


def _head_g():
    text = open(PAGE_G).read(); assert text.count(RESULT_MARK) <= 1
    return text[:text.index(RESULT_MARK)] if RESULT_MARK in text else text


def test_the_hosted_page_is_locked_and_carries_the_same_thirteen_lines_and_the_settings_the_code_uses():
    head = _head_g(); assert _h(head) == PAGE_G_SHA256, "the predictions and reading rules must not change after the run; later notes go below them"
    for i, (tag, line) in enumerate(M.LINES_M, 1): assert f"\n{i}. [{tag}] {line}\n" in head, i
    assert "Thirteen lines" in head and len(M.LINES_M) == 13 and "the same as on the Qwen page" in head
    for needle in (f"stream seed {M.SEEDS[0]}", f"stream seed {M.SEEDS[1]}", "3,600 requests", f"first {M.CONTRAST_STEPS} requests", f"`{M.GEMINI_MODEL}`", M.GEMINI_MODEL, f"{M.GEMINI_TOKEN_CAP:,} tokens",
                   f"{M.GEMINI_FREE_DAILY} calls a day", f"more than {M.THINKING_LIMIT} thinking tokens a step", '"minimal"', "seed 99", "`million.verdict_million`", "`million.replay`", "`million.LiveStream`",
                   "at most 8,192 output tokens including thinking", "the owner's choice", "No price is assumed on this page", "(85 %)", "(99 %)", "(80 %)", "(96 %)", "(95 %)", "(35 %)", "(88 %)", "(65 %)", "(25 %)",
                   "about **15 %**", "about **65 %**", "Flash Lite", "test_million_gemini_result.py"):
        assert needle in head, needle
    assert M.GEMINI_TOKEN_CAP == 2_000_000 and M.GEMINI_FREE_DAILY == 480 and M.THINKING_LIMIT == 20 and M.GEMINI_MODEL == "gemini-3.5-flash-lite" and M.MAIN_POLICY == "scope_m7"
    assert "it is **not** a Pro model and not a frontier model" in head and head.count("frontier") == 1           # the page says what the model is not, and nowhere calls it frontier


def test_the_hosted_page_predictions_are_numbered_like_the_lines_and_complete():
    head = _head_g(); pred = head.split("## Predictions")[1].split("## Reading rules")[0]
    nums = [int(m.group(1)) for l in pred.splitlines() for m in [re.match(r"^(\d+)\. ", l)] if m]
    assert nums == list(range(1, 15)), nums
    for i in range(1, 14):
        line = [l for l in pred.splitlines() if l.startswith(f"{i}. [")]; assert len(line) == 1 and re.search(r"\(\d+ %\)", line[0]), i           # every line carries its probability
    assert "\n14. Not reading-rule lines." in pred and pred.count("[guard]") == 2 and pred.count("[instrument]") == 2 and pred.count("[claim]") == 9
    for k in range(1, 14): assert f"\n{k}. [{M.LINES_M[k - 1][0]}]" in pred, k                                                                      # the tags are those of the lines
    p = {i: int(re.search(r"\((\d+) %\)", [l for l in pred.splitlines() if l.startswith(f"{i}. [")][0]).group(1)) for i in range(1, 14)}
    assert abs(100 * (p[1] / 100) * (p[2] / 100) * (p[3] / 100) * (p[4] / 100) - 65) <= 3 and "about **65 %**" in head          # the 'guard and instrument' figure is what the four lines' own figures give
    assert p[8] == min(p[i] for i in range(1, 13)) and "the line I am least sure of" in head and 15 <= min(p[i] for i in range(1, 13))          # line 8 is the one the page calls its weakest; the overall figure is below every single line


def test_the_hosted_page_numbers_it_cites_are_the_saved_flash_lite_files():
    """The facts of 'What shaped the probabilities' are recomputed from docs/results, so a page edit that disagrees with them is caught."""
    base = os.path.join(HERE, "..", "..", "docs", "results")
    def load(rule): return json.load(open(os.path.join(base, f"axi_gemini_long_{rule}_gemini-3.5-flash-lite_seed7.json")))
    d = load("norule"); head = _head_g()
    def rows(p): return [r for e in d["rows"] for r in e["policies"][p]]
    s, b = rows("scope_m7"), rows("blind"); r6 = rows("repair_m6")
    assert (len(s), sum(r["exact"] for r in s), sum(r["corrupted"] for r in s), sum(not r["parsed"] for r in s)) == (120, 120, 0, 0)
    assert (sum(r["exact"] for r in r6), sum(r["corrupted"] for r in r6)) == (120, 0)
    assert (sum(r["exact"] for r in b), sum(r["corrupted"] for r in b), sum(any(x["corrupted"] for x in e["policies"]["blind"]) for e in d["rows"])) == (61, 59, 3)
    firsts = sorted(min(x["t"] for x in e["policies"]["blind"] if x["corrupted"]) for e in d["rows"] if any(x["corrupted"] for x in e["policies"]["blind"])); assert firsts == [3, 13, 14]
    assert round(sum(r["prompt_tokens"] for r in s) / 120, 1) == 308.6 and round(sum(r["tokens"] for r in s) / 120, 1) == 14.6
    secs = [sum(r["seconds"] for r in rows(p)) / 120 for p in ("blind", "repair_m6", "scope_m7")]; assert [round(x, 2) for x in (min(secs), max(secs))] == [0.53, 0.56]
    q = json.load(open(os.path.join(base, "axi_scope_long_norule_Qwen2.5-7B_seed7.json"))); qs = [r for e in q["rows"] for r in e["policies"]["scope_m7"]]
    assert round(sum(r["prompt_tokens"] for r in qs) / len(qs), 1) == 307.3 and round(sum(r["tokens"] for r in qs) / len(qs), 1) == 11.3
    assert (round(sum(r["nodes"] for r in s) / 120, 1), round(sum(r["nodes"] for r in qs) / len(qs), 1)) == (4.9, 5.8)
    rr = load("rule"); n_unp = sum(not r["parsed"] for e in rr["rows"] for r in e["policies"]["scope_m7"]); assert n_unp == 42                                    # (the page's 75 of 334 counts distinct replies, in docs/API_TEST_GEMINI.md)
    for needle in ("120 of 120 steps", "blind apply held it on 61 of 120", "a corrupted graph on 59", "3 of 4 episodes", "(first at step 3, 13 and 14)", "15 of 20 requests", "20 of 20 without help", "0.53 to 0.56 s a call",
                   "3 of 140 single replies", "0 of 177", "75 of 334", "308.6 prompt tokens", "307.3 for Qwen", "4.9 services on average against 5.8", "14.6 tokens against 11.3", "3 of 317"):
        assert needle in head, needle
    # the stream's own numbers quoted on the page
    ch = []
    for seed in M.SEEDS:
        st = M.LiveStream(seed); tot = n = 0
        while n < M.STEPS:
            out = st.next(); s_, ob = out; oa = st.live
            msgs = M.base_messages(M.Task(s_["kind"], M.copy_graph(ob), s_["instruction"], s_["ideal"], oa), False); tot += sum(len(m["content"]) for m in msgs); n += 1
        ch.append(tot / n)
    assert [round(c) for c in ch] == [927, 920] and round(-256.8 + 0.6709 * ch[0]) == 365 and "on average 927 characters (920 for the second seed)" in head and "about 365 prompt tokens" in head


def test_the_hosted_notebook_never_turns_billing_on_and_holds_no_key():
    nb = json.load(open(NB_G)); code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"); allsrc = "\n".join("".join(c["source"]) for c in nb["cells"])
    assert re.search(r"^PAID = False\b", code, flags=re.M) and re.search(r"^THINKING = None\b", code, flags=re.M) and re.search(r"^PRICE_IN = None\b", code, flags=re.M) and re.search(r"^PRICE_OUT = None\b", code, flags=re.M)
    assert 'userdata.get("GEMINI_API_KEY")' in code and not re.search(r"AIza[0-9A-Za-z_\-]{10,}", allsrc) and "print(KEY" not in code and "print(key" not in code
    assert "docs/MILLION_TOKEN_TEST_GEMINI.md" in allsrc and "axi-colab-million.zip" in allsrc and "2,000,000 tokens" in allsrc
    assert "max_tokens=GEMINI_TOKEN_CAP" in code and "daily_limit=None if PAID else GEMINI_FREE_DAILY" in code and "smoke_report(" in code and "run_file(tag, 99, 20, MAIN_POLICY)" in code
    assert all(c["cell_type"] != "code" or (c.get("outputs") == [] and c.get("execution_count") is None) for c in nb["cells"])                           # no outputs saved in the notebook


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn): fn(); print("PASS", name); n += 1
    print(n, "tests passed")
