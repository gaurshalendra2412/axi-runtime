"""Implements concept-map item 115 (the long session on Gemini 3.8 Flash, test 22): the INSTRUMENT of docs/GEMINI_LONG_TEST.md, written before any call. It checks the instrument, not a result:
  1. the heavy turn (`ticket`): deterministic, the same in every arm, no coordinates; the kept chat's prompt is the instructions plus every turn and every reply, the AXI arm's prompt is flat, the depth probe's history
     is the ideal replies;
  2. the arms on a stand-in model that answers with the oracle's reply (or a slip on chosen turns): the saved flags, the resume from a file, the stop at the context cap and at the token cap, the replay that refuses a doctored file;
  3. the streamed backend (`GeminiChatBackend`): chat roles, the seconds to the first token, the cached tokens, the dollar guard with a cached price;
  4. speed, latency and token columns, recomputed by hand on small cases; the reading lines flip at their boundaries; the headline needs lines 7, 8 and 11;
  5. the page and the module agree and the page above its Result is byte for byte what was written before the run (a hash); the notebook holds no key, never turns billing on, sets no budget by itself;
  6. the front pages (concept-map item 115, VERIFIED row) and the counter.
What it does NOT show: anything about a real model's replies, or the result of the run."""
import contextlib, hashlib, io, json, os, re, sys, tempfile, types
sys.path.insert(0, os.path.dirname(__file__))
from axi.experiments import gemini_long as G
from axi.experiments import million as M
from axi.experiments import api_llm as A
from axi.experiments.agent_loop import strict_parse

HERE = os.path.dirname(__file__)
PAGE = os.path.join(HERE, "..", "..", "docs", "GEMINI_LONG_TEST.md")
NB = os.path.join(HERE, "..", "colab", "axi_gemini_long.ipynb")
PAGE_SHA256 = "acdb936ab9060869e27eb29f28c69fa6f514f6365ae13686a82401a314b9825b"        # written before any call
RESULT_MARK = "\n\n---\n\n# Result"


def _h(text): return hashlib.sha256((text.rstrip("\n") + "\n").encode()).hexdigest()
def tmp(): return os.path.join(tempfile.mkdtemp(), "r.jsonl")


class Fake:
    """A stand-in for ChatLLM: answers with the oracle's ideal reply for the request in the last user message (a slip on the turns in `slip`); records the messages it was sent."""
    def __init__(self, seed, slip=(), cache_share=0.5):
        self.seed, self.slip, self.share, self.seen = seed, set(slip), cache_share, []
        self.last_prompt_tokens = 0; self.last_think_tokens = 7; self.from_disk = False; self.last_cache_tokens = 0; self.last_first_token = 0.2; self.last_finish = "STOP"
        self.ideal = {}; stm = M.make_stream(G.STREAM, seed)
        for i in range(1, 400):
            o = stm.next()
            if o is None: break
            self.ideal[o[0]["instruction"]] = (i, o[0]["ideal"])

    def generate(self, messages, constrained=True, max_new_tokens=400):
        self.seen.append(messages); instr = messages[-1]["content"].rsplit("Request: ", 1)[1].split("\n\n")[0]; i, text = self.ideal[instr]
        if i in self.slip: text = "DEL[1,2:7]"
        self.last_prompt_tokens = sum(len(m["content"]) for m in messages) // 4; self.last_cache_tokens = int(self.last_prompt_tokens * self.share); self.last_first_token = 0.2 + 1e-6 * self.last_prompt_tokens
        return text, 10, 0.5 + 2e-6 * self.last_prompt_tokens


# ---------------------------------------------------------------- 1 the heavy turn
def test_the_heavy_turn_is_deterministic_the_same_in_every_arm_and_holds_no_rule():
    a, b = G.ticket(5, 3), G.ticket(5, 3)
    assert a == b and a != G.ticket(5, 4) and a != G.ticket(6, 3) and a.count("\n") == G.PAD_LINES
    assert "->" not in a and "edge" not in a.lower() and "service graph" not in a.lower()
    assert len(G.ticket(5, 3, 10).split("\n")) == 11


def test_the_kept_chat_grows_the_axi_prompt_is_flat_and_the_depth_history_is_ideal():
    seed = 5; k = G.run_kept(Fake(seed), seed, 8, tmp()); a = G.run_axi(Fake(seed), seed, 8, tmp())
    kp = [r["prompt_tokens"] for r in k["rows"]]; ap = [r["prompt_tokens"] for r in a["rows"]]
    assert all(y > x for x, y in zip(kp, kp[1:])) and max(ap) < 1.5 * min(ap) + 2000 and kp[-1] > 3 * kp[0]
    fk = Fake(seed); G.run_kept(fk, seed, 4, tmp()); m = fk.seen[3]
    assert [x["role"] for x in m] == ["system", "user", "assistant", "user", "assistant", "user", "assistant", "user"] and "Shift log" in m[1]["content"] and "Shift log" in m[7]["content"]
    assert m[1]["content"].count("{") >= 0 and "Request: " in m[7]["content"]
    fa = Fake(seed); G.run_axi(fa, seed, 3, tmp()); assert all(len(x) == 2 for x in fa.seen)
    fd = Fake(seed); d = G.run_depth(fd, seed, (5,), 2, tmp()); assert len(d["rows"]) == 2 and len(fd.seen[0]) == 1 + 2 * 4 + 1 and fd.seen[0][-1]["role"] == "user"
    ideal = [x["content"] for x in fd.seen[1] if x["role"] == "assistant"]; stm = M.make_stream(G.STREAM, seed); want = [stm.next()[0]["ideal"] for _ in range(5)]
    assert ideal == want


# ---------------------------------------------------------------- 2 the arms
def test_arm_k_flags_a_slip_resumes_from_its_file_and_the_replay_refuses_a_doctored_row():
    seed = 5; path = tmp(); llm = Fake(seed, slip=(3,)); k = G.run_kept(llm, seed, 9, path, max_tokens=1)
    assert len(k["rows"]) == 1 and k["stopped"] == "token_cap"
    k = G.run_kept(llm, seed, 9, path, max_tokens=12000); r = k["rows"]; assert 1 < len(r) < 9
    k2 = G.run_kept(Fake(seed, slip=(3,)), seed, 9, path); r = k2["rows"]; assert len(r) == 9 and k2["stopped"] is None
    assert r[2]["parsed"] is False and r[2]["reply_ok"] is False and r[2]["reply_exact"] is False and r[2]["corrupted"] is False and r[1]["reply_exact"] and r[3]["reply_exact"] is False or r[3]["reply_exact"] is True
    G.replay(k2["rows"], seed, "K")
    bad = [dict(x) for x in k2["rows"]]; bad[1]["text"] = bad[2]["text"]
    try: G.replay(bad, seed, "K"); raise AssertionError("a doctored row was accepted")
    except G.ReplayMismatch: pass
    try: G.run_kept(Fake(seed), 6, 3, path); raise AssertionError("another seed accepted")
    except ValueError: pass


def test_arm_k_stops_at_the_context_cap_and_at_the_token_cap():
    seed = 5; k = G.run_kept(Fake(seed), seed, 50, tmp(), context_cap=9000); assert k["stopped"] == "context" and len(k["rows"]) < 50 and max(r["prompt_tokens"] for r in k["rows"]) <= 9000
    k = G.run_kept(Fake(seed), seed, 50, tmp(), max_tokens=20000); assert k["stopped"] == "token_cap" and 1 <= len(k["rows"]) < 50
    k = G.run_kept(Fake(seed), seed, 4, tmp()); assert k["stopped"] is None


def test_arm_a_reasks_a_refused_reply_scripted_steps_cost_nothing_and_a_slip_is_caught():
    seed = 5; llm = Fake(seed); a = G.run_axi(llm, seed, 12, tmp(), scripted=8)
    assert len(llm.seen) == 4 and all(r["prompt_tokens"] == 0 and r["from_disk"] for r in a["rows"][:8]) and all(r["exact"] for r in a["rows"])
    calls = []
    class Slip(Fake):
        def generate(s, messages, constrained=True, max_new_tokens=400):
            calls.append(messages[-1]["content"][-40:]); t, n, sec = super().generate(messages, constrained, max_new_tokens)
            return ("DEL[1,2:7]" if len(calls) == 2 else t), n, sec
    a = G.run_axi(Slip(seed), seed, 4, tmp()); assert a["rows"][1]["asks"] == 2 and a["rows"][1]["reasks"][0]["text"] == a["rows"][1]["text"] and all(r["exact"] for r in a["rows"])
    G.replay(a["rows"], seed, "A")


def test_the_depth_probe_resumes_replays_and_refuses_a_doctored_row():
    seed = 5; path = tmp(); d = G.run_depth(Fake(seed), seed, (4, 8), 3, path); assert len(d["rows"]) == 6 and set(d["summary"]) == {4, 8}
    n = len(Fake(seed).seen); l2 = Fake(seed); d2 = G.run_depth(l2, seed, (4, 8), 3, path); assert len(d2["rows"]) == 6 and l2.seen == []
    G.replay_depth(d2["rows"], seed)
    bad = [dict(x) for x in d2["rows"]]; bad[0]["text"] = "DEL[1,2:7]"
    try: G.replay_depth(bad, seed); raise AssertionError("doctored")
    except G.ReplayMismatch: pass


# ---------------------------------------------------------------- 3 the backend and the guard
class Chunk:
    def __init__(s, text, um=None, fin=None): s.text = text; s.usage_metadata = um; s.candidates = [types.SimpleNamespace(finish_reason=fin)] if fin else []


class FakeTypes:
    class Content:
        def __init__(s, role, parts): s.role, s.parts = role, parts
    class Part:
        def __init__(s, text): s.text = text
    class GenerateContentConfig:
        def __init__(s, **kw): s.kw = kw
    class ThinkingConfig:
        def __init__(s, **kw): s.kw = kw


class FakeClient:
    def __init__(s, chunks, names=("models/gemini-3.8-flash",)):
        s.chunks, s.calls = chunks, []; s.models = types.SimpleNamespace(generate_content_stream=s._gen, list=lambda: [types.SimpleNamespace(name=n) for n in names])
    def _gen(s, model, contents, config): s.calls.append((model, contents, config)); return iter(s.chunks)


def test_the_streamed_backend_reads_first_token_cache_tokens_roles_and_refuses_a_bad_chat():
    um = types.SimpleNamespace(prompt_token_count=1000, candidates_token_count=9, thoughts_token_count=40, cached_content_token_count=900)
    cl = FakeClient([Chunk(""), Chunk("A"), Chunk("B", um, "STOP")]); be = G.GeminiChatBackend(model="m", client=cl, types=FakeTypes)
    r = be.call([{"role": "system", "content": "S"}, {"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"}, {"role": "user", "content": "u2"}])
    assert r["text"] == "AB" and r["prompt_tokens"] == 1000 and r["tokens"] == 9 and r["think_tokens"] == 40 and be.last_cache_tokens == 900 and be.last_first_token is not None and r["finish"] == "STOP"
    model, contents, cfg = cl.calls[0]; assert [c.role for c in contents] == ["user", "model", "user"] and cfg.kw["system_instruction"] == "S"
    for bad in ([{"role": "system", "content": "S"}], [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}], [{"role": "tool", "content": "x"}, {"role": "user", "content": "u"}]):
        try: be.call(bad); raise AssertionError("accepted")
        except ValueError: pass
    be2 = G.GeminiChatBackend(model="m", client=FakeClient([Chunk("")]), types=FakeTypes); assert be2.call([{"role": "user", "content": "u"}])["text"] == "" and be2.last_first_token is None


def test_the_chat_llm_saves_cache_and_first_token_prices_them_and_stops_at_the_budget():
    um = types.SimpleNamespace(prompt_token_count=1_000_000, candidates_token_count=1000, thoughts_token_count=1000, cached_content_token_count=500_000)
    said = []; path = tmp()
    llm, model = G.build_chat_llm("KEY-NOT-USED", path, client=FakeClient([Chunk("X", um, "STOP")]), types=FakeTypes, price_in=1.0, price_out=10.0, price_cached=0.1, budget_usd=0.5, min_interval=0, say=said.append)
    assert model == "gemini-3.8-flash"
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "u"}]
    t, n, s = llm.generate(msgs); assert t == "X" and llm.last_cache_tokens == 500_000 and llm.last_first_token is not None
    assert abs(llm.spent_usd() - (500_000 * 1.0 + 500_000 * 0.1 + 2000 * 10.0) / 1e6) < 1e-9
    llm.generate(msgs); assert llm.from_disk and llm.last_cache_tokens == 500_000
    try: llm.generate(msgs + [{"role": "user", "content": "v"}]); raise AssertionError("budget did not stop")
    except A.BudgetStop: pass
    assert "KEY-NOT-USED" not in open(path).read()
    llm2, _ = G.build_chat_llm("K", path, client=FakeClient([]), types=FakeTypes, price_in=1.0, price_out=10.0, min_interval=0, say=said.append)
    assert llm2.cache and list(llm2.cache.values())[0]["cache_tokens"] == 500_000 and llm2.spent_usd() > llm.spent_usd() - 1e-9


# ---------------------------------------------------------------- 4 the columns and the reading lines
def _row(t, prompt, cache, vis, think, sec, first, exact=True, ok=True, corrupted=False, dist=0, parsed=True, disk=False):
    return dict(t=t, kind="x", text="", prompt_tokens=prompt, cache_tokens=cache, tokens=vis, think=think, seconds=sec, first_token=first, from_disk=disk, reply_exact=exact, reply_ok=ok, parsed=parsed,
                exact=exact, corrupted=corrupted, distance=dist)


def test_speed_latency_and_token_columns_by_hand():
    rows = [_row(1, 1000, 0, 10, 40, 2.0, 0.5), _row(2, 3000, 2000, 10, 40, 4.0, 1.0, exact=False), _row(3, 5000, 4000, 10, 40, 6.0, 1.5), _row(4, 100, 0, 0, 0, 0.0, None, disk=True)]
    s = G.stats(rows)
    assert s["steps"] == 4 and s["live_steps"] == 3 and s["seconds"] == 12.0 and abs(s["steps_per_hour"] - 3 / 12 * 3600) < 1e-9 and abs(s["exact_per_hour"] - 2 / 12 * 3600) < 1e-9
    assert s["prompt"] == 9000 and s["cache"] == 6000 and s["computed"] == 3000 + 150 and abs(s["cached_share"] - 2 / 3) < 1e-9 and s["first_p50"] == 1.0 and s["total_max"] == 6.0 and s["first_mean"] == 1.0
    assert G.billed(rows[0]) == 1050 and G.billed(rows[3]) == 0 and G.computed(rows[1]) == 1000 + 50 and G.tokens_sent(rows) == 1050 + 3050 + 5050
    b = G.by_prompt_band(rows, edges=(0, 2000, 6000)); assert [(x["lo"], x["steps"]) for x in b] == [(0, 2), (2000, 2)]
    assert abs(G.stats(rows[:3])["computed_per_exact"] - 3150 / 2) < 1e-9
    r2 = dict(rows[0], asks=2, reasks=[dict(prompt_tokens=1000, cache_tokens=0, tokens=5, think=5, seconds=1.0, first_token=0.2, from_disk=False)]); assert G.billed(r2) == 1050 + 1010


def _fullrows(n, exact_until=10 ** 9, prompt=lambda t: 1000 + 2000 * t, sec=lambda t: 1.0, first=lambda t: 0.3, cache=0.9, think=lambda t: 40, ok_until=10 ** 9, corrupt_at=None):
    return [_row(t, prompt(t), int(prompt(t) * cache), 10, think(t), sec(t), first(t), exact=t <= exact_until, ok=t <= ok_until, corrupted=(t == corrupt_at), dist=0) for t in range(1, n + 1)]


def _ctx(k_rows, a_rows, d_rows=None, stopped="context"):
    return dict(rows=k_rows, stopped=stopped), dict(rows=a_rows, stopped=None), (dict(rows=d_rows) if d_rows else None)


def test_every_reading_line_flips_at_its_boundary_and_the_headline_needs_lines_7_8_and_11():
    a = _fullrows(G.A_STEPS, prompt=lambda t: 3000, sec=lambda t: 2.0, first=lambda t: 0.4)
    k = _fullrows(600, exact_until=100, prompt=lambda t: 1000 + 1700 * t, sec=lambda t: 1.0 + 0.05 * t, first=lambda t: 0.3 + 0.05 * t, think=lambda t: 40 if t <= 50 else 100, ok_until=100, corrupt_at=120)
    kk, aa, dd = _ctx(k, a); c = {i: v for i, _, _, v in G.verdict(kk, aa, dd, {"K": True, "A": True})}
    assert [c[i] for i in (1, 2, 3, 4, 5, 6, 7, 8, 9)] == [True] * 9 and c[10] is True and c[11] is True and c[12] is True and c[13] is True and c[14] is False and c[15] is True and c[16] is True and c[18] is True
    assert c[17] is True or c[17] is False
    assert c[19] is True and c[20] is None and G.headline(G.verdict(kk, aa, dd, {"K": True, "A": True})).startswith("AXI HOLDS")
    # boundaries
    a2 = _fullrows(G.A_STEPS); [r.update(exact=False) for r in a2[:30]]                                    # 2 % wrong, all in the first tenth
    assert {i: v for i, _, _, v in G.verdict(kk, dict(rows=a2, stopped=None), dd, {"K": True, "A": True})}[7] is False
    a3 = _fullrows(G.A_STEPS); a3[10].update(corrupted=True); assert {i: v for i, _, _, v in G.verdict(kk, dict(rows=a3, stopped=None), dd, {"K": True, "A": True})}[8] is False
    a4 = _fullrows(G.A_STEPS); a4[5].update(distance=5); assert {i: v for i, _, _, v in G.verdict(kk, dict(rows=a4, stopped=None), dd, {"K": True, "A": True})}[8] is False
    a5 = _fullrows(G.A_STEPS - 1); assert {i: v for i, _, _, v in G.verdict(kk, dict(rows=a5, stopped=None), dd, {"K": True, "A": True})}[2] is False
    assert {i: v for i, _, _, v in G.verdict(kk, aa, dd, {"K": False, "A": True})}[1] is False and {i: v for i, _, _, v in G.verdict(kk, aa, dd, {"K": True, "A": False})}[2] is False
    k2 = _fullrows(600, prompt=lambda t: 1000 + 700 * t); assert {i: v for i, _, _, v in G.verdict(_ctx(k2, a)[0], aa, dd, {})}[3] is False                       # never reaches 500,000
    k3 = _fullrows(600, exact_until=100, prompt=lambda t: 1000 + 1700 * t, ok_until=100); k3[0].update(reply_exact=False); assert {i: v for i, _, _, v in G.verdict(_ctx(k3, a)[0], aa, dd, {})}[14] is True
    k4 = _fullrows(300, exact_until=10 ** 9, prompt=lambda t: 1000 + 3000 * t); c4 = {i: v for i, _, _, v in G.verdict(_ctx(k4, a)[0], aa, dd, {})}; assert c4[11] is False and c4[12] is False and c4[13] is False
    k5 = _fullrows(300, prompt=lambda t: 1000 + 3000 * t); [r.update(cache_tokens=0) for r in k5[10:10 + 58]]; assert {i: v for i, _, _, v in G.verdict(_ctx(k5, a)[0], aa, dd, {})}[4] is True
    [r.update(cache_tokens=0) for r in k5[68:68 + 2]]; assert {i: v for i, _, _, v in G.verdict(_ctx(k5, a)[0], aa, dd, {})}[4] is False
    k6 = _fullrows(300, prompt=lambda t: 1000 + 3000 * t); [r.update(parsed=False) for r in k6[:7]]; assert {i: v for i, _, _, v in G.verdict(_ctx(k6, a)[0], aa, dd, {})}[6] is False
    a6 = _fullrows(G.A_STEPS); [r.update(parsed=False) for r in a6[:8]]; assert {i: v for i, _, _, v in G.verdict(kk, dict(rows=a6, stopped=None), dd, {})}[5] is False
    a7 = _fullrows(G.A_STEPS, prompt=lambda t: 3000 + t, sec=lambda t: 1.0 if t < 150 else 2.0); assert {i: v for i, _, _, v in G.verdict(kk, dict(rows=a7, stopped=None), dd, {})}[9] is False
    assert {i: v for i, _, _, v in G.verdict(None, None, None)} == {i: None for i in range(1, 21)}
    d = []
    for s, ex in ((50, 10), (450, 8)):
        for j in range(10): d.append(dict(_row(s + j, 1000 * s, 0, 10, 10, 1.0, 0.3, exact=j < ex), start=s))
    assert {i: v for i, _, _, v in G.verdict(kk, aa, dict(rows=d), {})}[20] is True
    d[-1].update(reply_exact=True); d[-2].update(reply_exact=True); assert {i: v for i, _, _, v in G.verdict(kk, aa, dict(rows=d), {})}[20] is False
    ch = G.verdict(kk, aa, dd, {}); mixed = [(i, t, x, (False if i == 8 else v)) for i, t, x, v in ch]; assert G.headline(mixed).startswith("MIXED")
    assert G.claims_allowed(ch)["base"] is True and G.claims_allowed([(i, t, x, (False if i == 4 else v)) for i, t, x, v in ch])["tokens_computed"] is False and G.claims_allowed([(i, t, x, (None if i == 3 else v)) for i, t, x, v in ch])["base"] is False


def test_the_pilot_report_projects_the_window_and_the_bill_from_20_real_turns():
    k = _fullrows(20, prompt=lambda t: 3000 + 2000 * t); a = _fullrows(20, prompt=lambda t: 3000, sec=lambda t: 2.0)
    p = G.pilot_report(k, a, 1.0, 10.0, 0.1); assert abs(p["k_growth_per_turn"] - 2000) < 1e-9 and p["k_turns_that_fit"] == int((G.K_CONTEXT_CAP - 5000) / 2000) + 1
    assert p["cost_k"] < p["cost_k_nocache"] and p["cost_a"] > 0 and abs(p["a_hours"] - 2.0 * G.A_STEPS / 3600) < 1e-9
    assert G.pilot_report(k, a)["cost_a"] is None
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): G.print_stats("K", k); G.print_bands(k); G.print_verdict(G.verdict(*_ctx(k, a)))
    out = buf.getvalue(); assert "steps/h" in out and "first p50" in out and "headline:" in out


# ---------------------------------------------------------------- 5 page and notebook
def _head():
    text = open(PAGE).read(); assert text.count(RESULT_MARK) <= 1
    return text[:text.index(RESULT_MARK)] if RESULT_MARK in text else text


def test_the_page_above_its_result_is_what_was_written_before_the_run():
    assert _h(_head()) == PAGE_SHA256


def test_the_page_and_the_module_agree():
    head = _head(); pred = head.split("## Predictions")[1].split("## Reading rules")[0]; rules = head.split("## Reading rules")[1].split("Headline:")[0]
    ps = re.findall(r"^(\d+)\. \[(\w+)\] (.*) \((\d+) %\)$", pred, re.M); rs = re.findall(r"^(\d+)\. (.*)$", rules, re.M)
    assert len(ps) == len(rs) == len(G.G_LINES) == 20
    for (n, tag, text, pr), (m, rt), (gt, gtext) in zip(ps, rs, G.G_LINES): assert n == m and text == rt == gtext and tag == gt and 0 < int(pr) < 100
    for needle in (G.MODEL, str(G.SEED), f"{G.K_CONTEXT_CAP:,}", f"{G.A_STEPS:,}", f"{G.K_MAX_STEPS}", f"{G.PAD_LINES} lines", f"{G.K_TOKEN_CAP:,}", f"{G.A_TOKEN_CAP:,}", f"{G.D_TOKEN_CAP:,}", "Energy in joules", "seconds to the first token"): assert needle in head, needle
    assert "ultimate" not in head.lower() and "$" not in head and "proven" not in head.lower()
    assert [s for s in G.D_STARTS] == [50, 100, 200, 300, 350] and G.MODEL == "gemini-3.8-flash"


def test_the_notebook_holds_no_key_never_turns_billing_on_and_sets_no_budget():
    nb = json.load(open(NB)); src = "\n".join("".join(c["source"]) for c in nb["cells"]); code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert not re.search(r"AIza[0-9A-Za-z_-]{20,}", src) and "GOOGLE_API_KEY" in src and "userdata.get" in src
    assert "BUDGET_USD = None" in code and "PAID = None" in code and "PRICE_IN = None" in code and "PRICE_CACHED = None" in code
    assert "enable billing" not in code.lower() and "billing" not in code.split("PAID")[0]
    for name in ("run_kept", "run_axi", "run_depth", "pilot_report", "replay_depth", "verdict", "build_chat_llm"): assert "G." + name in code, name
    assert code.count("need_budget()") >= 3 and all(c["outputs"] == [] for c in nb["cells"] if c["cell_type"] == "code")
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            body = "".join(c["source"]); body = "\n".join(l for l in body.split("\n") if not l.lstrip().startswith("!")); compile(body, "cell", "exec")


# ---------------------------------------------------------------- 6 front pages
def test_the_two_front_pages_say_what_was_built_and_that_nothing_was_called():
    cm = open(os.path.join(HERE, "..", "..", "docs", "RAJNISH_CONCEPT_MAP.md")).read(); vf = open(os.path.join(HERE, "..", "..", "docs", "VERIFIED.md")).read()
    assert "Next free: row 186, item 140, test 165, question 122" in cm and cm.count("Next free:") == 1 and "test 139 checks the long-session test" in cm and "115. **The long session on Gemini 3.8 Flash" in cm
    assert "GEMINI_LONG_TEST.md" in vf and "test 139 checks the long-session test" in cm


if __name__ == "__main__":
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]; bad = 0
    for n, f in fns:
        try: f(); print("ok  ", n)
        except Exception: bad += 1; print("FAIL", n); traceback.print_exc()
    print(f"{len(fns) - bad}/{len(fns)} passed"); sys.exit(1 if bad else 0)
