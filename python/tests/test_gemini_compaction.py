"""The INSTRUMENT of docs/COMPACTION_TEST.md (the head-to-head test: the kept chat with compaction against AXI), written before any call. It checks the instrument, not a result:
  1. the compaction rule: a summary is asked for when the last prompt reached the trigger; the summary call is the chat as it stands with its own system text and the request; the chat is emptied and the summary leads the next
     turn's message; the verbatim arm asks for the graph; the arms S, L and V are the ones the page names;
  2. the arms on a stand-in model that answers with the oracle's reply (or a slip on chosen turns) and writes a short summary: the saved flags, the resume from a file, the stop at the token cap, an empty summary refused,
     the replay that refuses a doctored reply, a compaction made when none was due, a compaction missing when due, a blank summary;
  3. counting: the summary call is part of the step after it (tokens, cached tokens, seconds, first token), the cost at typed prices recomputed by hand, with and without a cached price;
  4. the figures of an arm (held-exact share, first slip, never exact again, cost per exact step) recomputed by hand, and every reading line flipping at its boundary, the cost lines staying 'not run' without the three prices;
  5. the projection of a pilot recomputed by hand;
  6. the page and the module agree and the page above its Result is byte for byte what was written before the run (a hash); the notebook holds no key, never turns billing on, sets no budget by itself;
  7. the existing files are untouched by this test (it only reads them).
What it does NOT show: anything about a real model's summaries, or the result of the run."""
import hashlib, json, os, re, sys, tempfile
sys.path.insert(0, os.path.dirname(__file__))
from axi.experiments import gemini_compaction as C
from axi.experiments import gemini_long as G
from axi.experiments import million as M

HERE = os.path.dirname(__file__)
PAGE = os.path.join(HERE, "..", "..", "docs", "COMPACTION_TEST.md")
NB = os.path.join(HERE, "..", "colab", "axi_gemini_compaction.ipynb")
PAGE_SHA256 = "dac1514c8deaaf49df6527431f74b4adbadbd30661b2e3767d108c38a6cb1b5f"        # written before any call
RESULT_MARK = "\n\n---\n\n# Result"


def _h(text): return hashlib.sha256((text.rstrip("\n") + "\n").encode()).hexdigest()
def tmp(): return os.path.join(tempfile.mkdtemp(), "r.jsonl")


class Fake:
    """A stand-in for ChatLLM: answers with the oracle's ideal reply for the request in the last user message (a slip on the turns in `slip`); for a summary call (the compaction system text) it writes 'SUMMARY n'.
    Records the messages it was sent. The prompt size is the number of characters over four."""
    def __init__(self, seed, slip=(), summary=None, cache_share=0.5):
        self.seed, self.slip, self.share, self.seen, self.n_sum, self.summary = seed, set(slip), cache_share, [], 0, summary
        self.last_prompt_tokens = 0; self.last_think_tokens = 7; self.from_disk = False; self.last_cache_tokens = 0; self.last_first_token = 0.2; self.last_finish = "STOP"
        self.ideal = {}; stm = M.make_stream(G.STREAM, seed)
        for i in range(1, 400):
            o = stm.next()
            if o is None: break
            self.ideal[o[0]["instruction"]] = (i, o[0]["ideal"])

    def generate(self, messages, constrained=True, max_new_tokens=400):
        self.seen.append(messages); self.last_prompt_tokens = sum(len(m["content"]) for m in messages) // 4; self.last_cache_tokens = int(self.last_prompt_tokens * self.share); self.last_first_token = 0.2 + 1e-6 * self.last_prompt_tokens
        if messages[0]["content"] == C.COMPACT_SYSTEM:
            self.n_sum += 1; text = self.summary if self.summary is not None else f"SUMMARY {self.n_sum}: the graph so far"; return text, 30, 1.0 + 1e-6 * self.last_prompt_tokens
        instr = messages[-1]["content"].rsplit("Request: ", 1)[1].split("\n\n")[0]; i, text = self.ideal[instr]
        if i in self.slip: text = "DEL[1,2:7]"
        return text, 10, 0.5 + 2e-6 * self.last_prompt_tokens


SEED = 5
TRG = 3000


def run(arm="S", steps=14, llm=None, path=None, trigger=TRG, **kw):
    llm = llm or Fake(SEED); return C.run_compact(llm, SEED, steps, arm, path or tmp(), trigger=trigger, **kw), llm


# ---------------------------------------------------------------- 1 the compaction rule
def test_the_arms_are_the_ones_the_page_names():
    assert C.ARMS == {"S": dict(trigger=32_000, style="generic"), "L": dict(trigger=128_000, style="generic"), "V": dict(trigger=32_000, style="verbatim")}
    assert C.STEPS == 600 and C.SEED == G.SEED == 20261111 and C.G.STREAM == "HardStream" and C.A_FILE_SHA256 == "7ba26d2946ec20ddd8a2085f18c6a2eee9ac3fb02025209158ce67a91faf29c6"
    assert "complete current service graph" in C.COMPACT_VERBATIM and "service graph" not in C.COMPACT_GENERIC and "graph" not in C.COMPACT_GENERIC.lower()


def test_a_summary_is_due_when_the_last_prompt_reaches_the_trigger_and_not_before():
    assert C.compaction_due([], 100) is False and C.compaction_due([dict(prompt_tokens=99)], 100) is False and C.compaction_due([dict(prompt_tokens=100)], 100) is True
    assert C.compaction_due([dict(prompt_tokens=500), dict(prompt_tokens=99)], 100) is False and C.compaction_due([dict(prompt_tokens=None)], 100) is False


def test_the_chat_restarts_with_the_summary_and_the_summary_call_is_the_chat_as_it_stands():
    r, llm = run("S", 14); rows = r["rows"]; comp_t = [x["t"] for x in rows if "compaction" in x]; assert len(comp_t) >= 2 and comp_t[0] > 2
    first = comp_t[0]; assert rows[first - 2]["prompt_tokens"] >= TRG and all(x["prompt_tokens"] < TRG for x in rows[:first - 2])
    call = next(m for m in llm.seen if m[0]["content"] == C.COMPACT_SYSTEM)                                                   # the first summary call
    assert [m["role"] for m in call] == ["system"] + ["user", "assistant"] * (first - 1) + ["user"] and call[-1]["content"] == C.COMPACT_GENERIC and "Shift log" in call[1]["content"]
    turn = llm.seen[llm.seen.index(call) + 1]                                                                                    # the turn right after it: system + one message
    assert len(turn) == 2 and turn[0]["content"] == C.SYSTEM and turn[1]["content"].startswith(C.SUMMARY_HEAD + "SUMMARY 1: the graph so far\n\n") and "Request: " in turn[1]["content"]
    assert rows[first - 1]["compaction"]["text"] == "SUMMARY 1: the graph so far" and rows[first - 1]["prompt_tokens"] < rows[first - 2]["prompt_tokens"]
    nxt = llm.seen[llm.seen.index(call) + 2]; assert [m["role"] for m in nxt] == ["system", "user", "assistant", "user"] and nxt[1]["content"].startswith(C.SUMMARY_HEAD)       # the summary stays in the chat until the next one
    assert sum(1 for m in llm.seen if m[0]["content"] == C.COMPACT_SYSTEM) == len(comp_t)


def test_the_verbatim_arm_asks_for_the_graph_and_the_first_state_is_in_the_first_turn_only():
    r, llm = run("V", 8, trigger=2000); call = next(m for m in llm.seen if m[0]["content"] == C.COMPACT_SYSTEM)
    assert call[-1]["content"] == C.COMPACT_VERBATIM and r["meta"]["style"] == "verbatim" and r["meta"]["trigger"] == 2000
    first_turn = llm.seen[0][1]["content"]; assert "[" in first_turn.split("Shift log")[0] and first_turn.split("Shift log")[0].strip() != ""
    second = llm.seen[1][3]["content"]; assert second.split("Shift log")[0].strip() == ""


# ---------------------------------------------------------------- 2 the arms
def test_a_slip_is_flagged_and_a_blind_state_stays_wrong():
    r, _ = run("S", 9, llm=Fake(SEED, slip=(3,))); x = r["rows"]
    assert x[2]["parsed"] is False and x[2]["reply_exact"] is False and x[1]["reply_exact"] and x[0]["exact"] and x[1]["exact"] and all("reply_ok" in y for y in x)


def test_resume_from_the_file_the_token_cap_and_an_empty_summary_refused():
    path = tmp(); llm = Fake(SEED); r = C.run_compact(llm, SEED, 14, "S", path, trigger=TRG, max_tokens=1); assert len(r["rows"]) == 1 and r["stopped"] == "token_cap"
    r = C.run_compact(llm, SEED, 14, "S", path, trigger=TRG, max_tokens=9000); assert 1 < len(r["rows"]) < 14
    llm2 = Fake(SEED); r2 = C.run_compact(llm2, SEED, 14, "S", path, trigger=TRG); assert len(r2["rows"]) == 14 and r2["stopped"] is None
    again = Fake(SEED); r3 = C.run_compact(again, SEED, 14, "S", path, trigger=TRG); assert len(r3["rows"]) == 14 and again.seen == []
    try: C.run_compact(Fake(SEED), 6, 3, "S", path, trigger=TRG); raise AssertionError("another seed accepted")
    except ValueError: pass
    try: C.run_compact(Fake(SEED), SEED, 14, "L", path, trigger=TRG); raise AssertionError("another arm accepted")
    except ValueError: pass
    p2 = tmp()
    try: C.run_compact(Fake(SEED, summary="  "), SEED, 14, "S", p2, trigger=TRG); raise AssertionError("an empty summary was accepted")
    except ValueError as e: assert "empty summary" in str(e)
    saved = [json.loads(l) for l in open(p2)][1:]; assert saved and all("compaction" not in x for x in saved) and len(saved) < 14


def test_the_replay_refuses_a_doctored_file():
    r, _ = run("S", 14); rows = r["rows"]; C.replay(rows, SEED, TRG, "generic")
    i = next(k for k, x in enumerate(rows) if "compaction" in x)
    def bad(f):
        b = [json.loads(json.dumps(x)) for x in rows]; f(b)
        try: C.replay(b, SEED, TRG, "generic")
        except G.ReplayMismatch as e: return str(e)
        raise AssertionError("a doctored file was accepted")
    assert "not made" in bad(lambda b: b[i].pop("compaction")) and "not due" in bad(lambda b: b[1].update(compaction=dict(b[i]["compaction"])))
    assert "no summary" in bad(lambda b: b[i]["compaction"].update(text="  ")) and "flags" in bad(lambda b: b[1].update(text="DEL[1,2:7]"))
    assert "flags" in bad(lambda b: b[2].update(distance=b[2]["distance"] + 1))


# ---------------------------------------------------------------- 3 counting
def comp(prompt=30000, cache=0, text="s", tokens=50, think=10, sec=2.0, first=1.0):
    return dict(text=text, tokens=tokens, think=think, prompt_tokens=prompt, cache_tokens=cache, first_token=first, seconds=sec, from_disk=False, finish="STOP")


def row(t, exact=True, reply_exact=None, prompt=1000, sec=3.0, cache=0, tokens=10, think=5, parsed=True, c=None, first=0.5):
    r = dict(t=t, kind="add_node", text="ADD[0,0:1]", seconds=sec, tokens=tokens, think=think, prompt_tokens=prompt, cache_tokens=cache, first_token=first, from_disk=False, finish="STOP", parsed=parsed, reply_ok=True,
             reply_exact=exact if reply_exact is None else reply_exact, corrupted=False, distance=0 if exact else 2, exact=exact, nodes=5, edges=5, oracle_nodes=5, oracle_edges=5)
    if c: r["compaction"] = c
    return r


def test_the_summary_call_is_counted_with_the_step_after_it():
    a, b = row(1, prompt=1000, cache=0), row(2, prompt=500, cache=0, c=comp(prompt=30000, cache=0))
    L = C.as_long(b); assert len(G.parts_of(L)) == 2 and L["seconds"] == 5.0 and G.billed(L) == 500 + 10 + 5 + 30000 + 50 + 10 and C.billed(a) == 1015
    assert C.tokens_sent([a, b]) == 1015 + 30575 and C.stats([a, b])["steps"] == 2 and C.stats([a, b])["seconds"] == 3.0 + 5.0 and C.stats([a, b])["prompt"] == 1000 + 500 + 30000
    assert C.as_long(a) is a and "compaction" in b and "reasks" not in b


def test_the_cost_at_typed_prices_is_recomputed_by_hand_with_and_without_a_cached_price():
    a, b = row(1, prompt=1000, cache=800, tokens=10, think=5), row(2, prompt=500, cache=0, tokens=10, think=5, c=comp(prompt=30000, cache=0, tokens=50, think=10))
    pin, pout, pc = 0.75, 3.75, 0.075
    want = ((200 * pin + 800 * pc + 15 * pout) + (500 * pin + 15 * pout) + (30000 * pin + 60 * pout)) / 1e6
    assert abs(C.cost_usd([a, b], pin, pout, pc) - want) < 1e-12
    want0 = ((1000 * pin + 15 * pout) + (500 * pin + 15 * pout) + (30000 * pin + 60 * pout)) / 1e6
    assert abs(C.cost_usd([a, b], pin, pout) - want0) < 1e-12 and C.cost_usd([a, b], None, pout) is None and C.cost_usd([a, b], pin, None, pc) is None


# ---------------------------------------------------------------- 4 the figures and the reading lines
def test_the_figures_of_an_arm_by_hand():
    rows = [row(1), row(2), row(3, exact=False, reply_exact=False), row(4, exact=False, reply_exact=True), row(5, exact=True, reply_exact=True), row(6, exact=False, reply_exact=False)]
    f = C.arm_figures(rows, 0.75, 3.75, 0.075)
    assert f["steps"] == 6 and f["held_exact_steps"] == 3 and f["held_exact"] == 0.5 and f["final_exact"] is False and f["first_slip"] == 3 and f["first_held_wrong"] == 3 and f["held_wrong_turns"] == 3
    assert f["lost_for_good"] == 6 and abs(f["reply_exact"] - 4 / 6) < 1e-12 and f["max_distance"] == 2
    per = (1000 * 0.75 + 15 * 3.75) / 1e6; assert abs(f["cost"] - 6 * per) < 1e-12 and abs(f["cost_per_step"] - per) < 1e-12 and abs(f["cost_per_exact"] - 2 * per) < 1e-12
    g = C.arm_figures([row(1), row(2)]); assert g["lost_for_good"] is None and g["cost"] is None and g["cost_per_step"] is None and g["held_exact"] == 1.0
    z = C.arm_figures([row(1, exact=False)], 0.75, 3.75, 0.075); assert z["cost_per_exact"] == float("inf") and z["lost_for_good"] == 1


def arm(n_exact, n=600, comps=12, final_exact=None, first_slip=None, parsed_share=1.0, empty=False, sec=3.0, cost_scale=1, prompt=1000):
    """600 rows, the first n_exact held-exact then wrong (final row forced by `final_exact`), `comps` compactions spread over the arm, the first reply slip at `first_slip`."""
    rows = []
    for t in range(1, n + 1):
        ex = t <= n_exact if final_exact is None or t < n else final_exact
        c = comp(text="" if empty and t == 50 else "sum", prompt=30000 * cost_scale) if comps and t > 1 and t % (n // comps) == 0 else None
        rows.append(row(t, exact=ex, reply_exact=(t != first_slip) and ex or (first_slip is None and ex), sec=sec, prompt=prompt, parsed=(t > parsed_share * n * 0 + (1 - parsed_share) * n), c=c))
    return dict(rows=rows, summary={}, meta={}, stopped=None)


PRICES = (0.75, 3.75, 0.075)


def verdict(S, L, V, A, **kw):
    return C.verdict(dict(S=S, L=L, V=V), A["rows"] if A else None, kw.get("replays", dict(S=True, L=True, V=True, A=True)), kw.get("a_file_ok", True), kw.get("prices", PRICES))


def res(v): return {i: c for i, _, _, c in v}


def test_every_reading_line_flips_at_its_boundary():
    A = arm(600, comps=0, prompt=2800); S = arm(588, comps=12); L = arm(600, comps=4); V = arm(500, comps=12)
    r = res(verdict(S, L, V, A))
    assert r[1] and r[2] and r[3] and r[4] and r[5] and r[6]
    assert r[7] is True and r[8] is True and r[9] is False                                                                 # 588/600 = 98.0 % reads yes; 500/600 does not
    assert r[10] is False and r[11] is False and r[12] is True                                                             # AXI 100 % against S 98.0 % (2 points, not 5), L 100 %, V 83.3 %
    assert r[13] is True and r[14] is False and r[15] is True                                                              # the last step: S's (588 exact steps) is wrong, L's is exact, V's is wrong
    assert res(verdict(arm(587, comps=12), L, V, A))[7] is False
    assert res(verdict(arm(570, comps=12), L, V, A))[10] is True and res(verdict(arm(571, comps=12), L, V, A))[10] is False       # 100 % - 95.0 % = 5 points reads yes, 4.8 does not
    assert res(verdict(arm(600, comps=12, final_exact=False), L, V, A))[13] is True and res(verdict(arm(600, comps=12), L, V, A))[13] is False
    assert res(verdict(arm(600, comps=12, first_slip=99), L, V, A))[16] is True and res(verdict(arm(600, comps=12, first_slip=100), L, V, A))[16] is False and res(verdict(arm(600, comps=12), L, V, A))[16] is False
    assert res(verdict(S, L, V, A, replays=dict(S=False, L=True, V=True, A=True)))[1] is False and res(verdict(S, L, V, A, a_file_ok=False))[4] is False
    assert res(verdict(S, L, V, A, replays=dict(S=True, L=True, V=True, A=False)))[4] is False
    assert res(verdict(arm(588, comps=9), L, V, A))[5] is False and res(verdict(S, arm(600, comps=2), V, A))[5] is False and res(verdict(arm(588, comps=12, empty=True), L, V, A))[5] is False
    assert res(verdict(arm(588, comps=12, parsed_share=0.97), L, V, A))[6] is False and res(verdict(arm(588, comps=12, parsed_share=0.98), L, V, A))[6] is True
    short = dict(S, rows=S["rows"][:599]); assert res(verdict(short, L, V, A))[1] is False
    assert res(verdict(S, L, V, None))[4] is None and res(verdict(S, L, V, None))[10] is None


def test_the_time_line_and_the_cost_lines_and_the_headline():
    A = arm(600, comps=0, prompt=2800, sec=3.0); S = arm(588, comps=12, sec=3.0, prompt=17000); L = arm(600, comps=4, sec=4.0, prompt=64000); V = arm(500, comps=12, sec=3.0, prompt=17000)      # the chats are longer than AXI's flat prompt
    r = res(verdict(S, L, V, A)); assert r[17] is False                                                                      # S: 3.0 s a step plus 12 summary calls of 2.0 s over 600 steps = 3.04 s, against 3.0 s for AXI
    assert res(verdict(arm(588, comps=12, sec=2.9), L, V, A))[17] is True and res(verdict(arm(588, comps=12, sec=2.97), L, V, A))[17] is False         # 2.9 s + 0.04 = 2.94 s reads yes; 2.97 s + 0.04 = 3.01 s does not
    # the cost lines: not run without the three prices; with them, the cheaper arm decides
    assert res(verdict(S, L, V, A, prices=(0.75, 3.75, None)))[18] is None and res(verdict(S, L, V, A, prices=(None, None, None)))[19] is None
    r = res(verdict(S, L, V, A)); assert r[18] is True and r[19] is True
    big = arm(600, comps=0, prompt=40000); r = res(verdict(S, L, V, big)); assert r[18] is False and r[19] is False                # a dearer AXI arm reads NO
    one = arm(600, comps=0, prompt=20000); r = res(verdict(S, L, V, one)); assert r[18] is False                                     # dearer than arm S alone is enough
    r = res(verdict(S, L, V, arm(588, comps=0, prompt=2800))); assert r[18] is True and r[19] is True                              # cheaper per step and per exact step (S has 588 exact steps and a dearer step)
    assert C.headline(verdict(S, L, V, A)).startswith("MIXED") and "line 11 NO" in C.headline(verdict(S, L, V, A))
    S3, L3, V3 = arm(500, comps=12), arm(500, comps=4), arm(500, comps=12); h = C.headline(verdict(S3, L3, V3, A)); assert h == "AXI HOLDS THE STATE BETTER THAN EACH COMPACTION ARM"
    assert "arm S" in C.headline(verdict(arm(595, comps=12), arm(500, comps=4), V3, A)) or C.headline(verdict(arm(595, comps=12), arm(500, comps=4), V3, A)).startswith("MIXED")
    assert C.headline(verdict(S, L, V, None)).startswith("NOT READ")
    assert C.claims_allowed(verdict(S, L, V, A))["base"] is True and C.claims_allowed(verdict(S, L, V, A, a_file_ok=False))["base"] is False


# ---------------------------------------------------------------- 5 the projection of a pilot
def test_the_projection_of_a_pilot_by_hand():
    rows = [row(i + 1, prompt=1000 + 100 * i, cache=int((1000 + 100 * i) * 0.5), tokens=10, think=10) for i in range(5)]
    p = C.project(rows, 10, 10 ** 9, 0.75, 3.75, 0.075)            # no trigger reached: a prompt that grows by 100 a turn from 1,000, half of it cached
    tot = sum(1000 + 100 * i for i in range(10)); cached = 0.5 * (tot - 1000)                      # the first turn is cold: no cache
    assert p["compactions"] == 0 and p["growth_per_turn"] == 100 and abs(p["cached_share"] - 0.5) < 1e-9 and abs(p["prompt_tokens"] - tot) < 1e-9 and abs(p["output_tokens"] - 200) < 1e-9
    want = ((tot - cached) * 0.75 + cached * 0.075 + 200 * 3.75) / 1e6; assert abs(p["cost"] - want) < 1e-9 and abs(p["cost_nodiscount"] - (tot * 0.75 + 200 * 3.75) / 1e6) < 1e-9
    q = C.project(rows, 10, 1500, 0.75, 3.75); assert q["compactions"] >= 1 and q["cost"] is not None and C.project(rows, 10, 1500, None, None)["cost"] is None and "note" in C.project([row(1)], 10, 1000, 1, 1)
    assert p["cost"] < p["cost_nodiscount"]


# ---------------------------------------------------------------- 6 the page and the notebook
def _page():
    return open(PAGE).read()


def test_the_page_above_its_result_is_the_text_hashed_before_the_run_and_agrees_with_the_module():
    head = _page().split(RESULT_MARK)[0]; assert _h(head) == PAGE_SHA256
    rules = head.split("## Reading rules")[1].split("Headline:")[0]; listed = re.findall(r"^(\d+)\. (.*)$", rules, re.M)
    assert [t for _, t in listed] == [text for _, text in C.LINES] and [int(n) for n, _ in listed] == list(range(1, len(C.LINES) + 1))
    pred = head.split("## Predictions")[1].split("## Reading rules")[0]; got = re.findall(r"^(\d+)\. \[(\w+)\] (.*) \((\d+) %\)$", pred, re.M)
    assert [(int(n), tag, text) for n, tag, text, _ in got] == [(i + 1, tag, text) for i, (tag, text) in enumerate(C.LINES)] and all(0 < int(p) < 100 for *_, p in got)
    for name in ("S", "L", "V"): assert f"`{name}`" in head
    for must in ("32,000", "128,000", "600 steps", "20261111", "gemini-3.8-flash", C.A_FILE_SHA256[:12], "Nothing on this page has been run", "host side", "Rajnish"): assert must in head, must
    for banned in ("open source", "open-source", "proven", "we cracked", "revolution"): assert banned not in head.lower(), banned


def test_the_notebook_holds_no_key_never_turns_billing_on_and_sets_no_budget():
    nb = json.load(open(NB)); src = "\n".join("".join(c["source"]) for c in nb["cells"]); code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert not re.search(r"AIza[0-9A-Za-z_-]{20,}", src) and "GOOGLE_API_KEY" in src and "userdata.get" in src
    assert "BUDGET_USD = None" in code and "PAID = None" in code and "PRICE_IN = None" in code and "PRICE_OUT = None" in code and "PRICE_CACHED = None" in code
    assert "enable billing" not in code.lower() and "billing" not in code.split("PAID")[0]
    for name in ("run_compact", "project", "replay", "verdict", "print_arm", "print_verdict"): assert "C." + name in code, name
    assert "G.build_chat_llm" in code and code.count("need_budget()") >= 3 and all(c["outputs"] == [] for c in nb["cells"] if c["cell_type"] == "code")
    assert "A_FILE_SHA256" in code and "hashlib" in code
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            body = "".join(c["source"]); body = "\n".join(l for l in body.split("\n") if not l.lstrip().startswith("!")); compile(body, "cell", "exec")


# ---------------------------------------------------------------- 7 the saved arm A is only read
def test_the_saved_arm_a_is_the_file_with_the_hash_written_here_and_replays():
    p = os.path.join(HERE, "..", "..", "docs", "results", "axi_gemini_long_session_seed20261111_default.json")
    assert hashlib.sha256(open(p, "rb").read()).hexdigest() == C.A_FILE_SHA256
    a = json.load(open(p))["A"]; rows = a["rows"][:C.STEPS]; assert len(rows) == 600 and all(r["exact"] for r in rows) and "compaction" not in rows[0]
    G.replay(rows, C.SEED, "A")
    f = C.arm_figures(rows, 0.75, 3.75, 0.075); assert f["held_exact"] == 1.0 and f["lost_for_good"] is None and f["compactions"] == 0 and f["cached_share"] == 0.0


if __name__ == "__main__":
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]; bad = 0
    for n, f in fns:
        try: f(); print("ok  ", n)
        except Exception: bad += 1; print("FAIL", n); traceback.print_exc()
    print(f"{len(fns) - bad}/{len(fns)} passed"); sys.exit(1 if bad else 0)
