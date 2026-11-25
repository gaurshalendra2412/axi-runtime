"""The INSTRUMENT of docs/LITE_COMPACTION_TEST.md (test 46: the kept chat with compaction against AXI, on Gemini 3.5 Flash Lite, inside the free quota), written before any call. It checks the instrument, not a result:
  1. the settings the page names (model, seed, 300 steps, the two compaction arms, the free-quota guards) and that the module reuses the existing code instead of changing it;
  2. the three arms on a stand-in model that answers with the oracle's reply: the same rows as the existing runners give, the replay of each arm, an unknown arm refused;
  3. the model setup: the id is picked from the key's list, the quota guards are the page's, no price and no budget exist;
  4. every reading line flipping at its boundary, and the headline;
  5. the page and the module agree and the page above its Result is byte for byte what was written before the run (a hash); the notebook holds no key, no price, no budget and never turns billing on;
  6. the existing files are untouched by this test (it only imports them).
What it does NOT show: anything about a real model's summaries, or the result of the run."""
import hashlib, json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
from axi.experiments import gemini_compaction as C
from axi.experiments import gemini_long as G
from axi.experiments import gemini_lite_compaction as L
from test_gemini_compaction import Fake, arm, tmp

HERE = os.path.dirname(__file__)
PAGE = os.path.join(HERE, "..", "..", "docs", "LITE_COMPACTION_TEST.md")
NB = os.path.join(HERE, "..", "colab", "axi_gemini_lite_compaction.ipynb")
PAGE_SHA256 = "837f3ef3b0495a30ee3916a727141a3a94bfd3ff6ab4a808037050bd4a4f02ad"        # written before any call
RESULT_MARK = "\n\n---\n\n# Result"
SEED, TRG = L.SEED, 3000


def _h(text): return hashlib.sha256((text.rstrip("\n") + "\n").encode()).hexdigest()


# ---------------------------------------------------------------- 1 the settings
def test_the_settings_are_the_ones_the_page_names():
    assert L.MODEL == "gemini-3.5-flash-lite" and L.SEED == G.SEED == 20261111 and L.STEPS == 300 and G.STREAM == "HardStream"
    assert L.ARMS == {"S": dict(trigger=32_000, style="generic"), "V": dict(trigger=32_000, style="verbatim")}
    assert L.ARMS["S"] == C.ARMS["S"] and L.ARMS["V"] == C.ARMS["V"] and "L" not in L.ARMS
    assert L.DAILY_CALLS == 480 and L.MIN_INTERVAL == 6.0 and L.DAILY_CALLS < 500 and 60 / L.MIN_INTERVAL <= 15
    assert L.TOKEN_CAP == {"S": 12_000_000, "V": 12_000_000, "A": 3_000_000}


def test_the_module_changes_nothing_that_exists():
    assert G.MODEL == "gemini-3.8-flash" and C.STEPS == 600 and set(C.ARMS) == {"S", "L", "V"} and L.G is G and L.C is C


# ---------------------------------------------------------------- 2 the arms
def test_the_compaction_arms_give_the_rows_of_the_existing_runner():
    for a in "SV":
        r = L.run_arm(Fake(SEED), a, tmp(), steps=14, trigger=TRG, max_tokens=10 ** 9); d = C.run_compact(Fake(SEED), SEED, 14, a, tmp(), trigger=TRG)
        assert r["rows"] == d["rows"] and r["meta"]["arm"] == a and r["meta"]["steps"] == 14 and len(r["rows"]) == 14 and any("compaction" in x for x in r["rows"])
        L.replay_arm(r["rows"], a, trigger=TRG)


def test_arm_a_is_the_framework_arm_and_replays():
    r = L.run_arm(Fake(SEED), "A", tmp(), steps=8, max_tokens=10 ** 9); d = G.run_axi(Fake(SEED), SEED, 8, tmp())
    assert r["rows"] == d["rows"] and r["meta"]["arm"] == "A" and r["meta"]["policy"] == "scope_m7" and all(x["exact"] and x["parsed"] for x in r["rows"])
    L.replay_arm(r["rows"], "A")


def test_a_wrong_arm_and_a_doctored_file_are_refused():
    for f in (lambda: L.run_arm(Fake(SEED), "L", tmp()), lambda: L.replay_arm([], "L")):
        try: f(); raise AssertionError("arm L was accepted")
        except ValueError: pass
    r = L.run_arm(Fake(SEED), "S", tmp(), steps=14, trigger=TRG); rows = [json.loads(json.dumps(x)) for x in r["rows"]]; rows[1]["text"] = "DEL[1,2:7]"
    try: L.replay_arm(rows, "S", trigger=TRG); raise AssertionError("a doctored file was accepted")
    except G.ReplayMismatch: pass


def test_the_token_cap_stops_an_arm_and_the_same_call_carries_on():
    path = tmp(); llm = Fake(SEED); r = L.run_arm(llm, "S", path, steps=14, trigger=TRG, max_tokens=1); assert len(r["rows"]) == 1 and r["stopped"] == "token_cap"
    r = L.run_arm(llm, "S", path, steps=14, trigger=TRG, max_tokens=10 ** 9); assert len(r["rows"]) == 14 and r["stopped"] is None


# ---------------------------------------------------------------- 3 the model setup
class _M:
    def __init__(self, name): self.name = name


class _Models:
    def list(self): return [_M("models/gemini-3.8-flash"), _M("models/gemini-3.5-flash-lite-preview-tts"), _M("models/gemini-3.5-flash-lite-preview-09-2026"), _M("models/gemini-3.1-flash-lite")]


class _Client: models = _Models()


def test_the_model_is_picked_from_the_keys_list_with_the_free_quota_guards_and_no_price():
    lines = []; llm, model = L.build_lite_llm("key", tmp(), client=_Client(), types=object(), say=lines.append)
    assert model == "gemini-3.5-flash-lite-preview-09-2026" and llm.daily_limit == 480 and llm.min_interval == 6.0 and llm.spent_usd() is None and llm.price_in is None and llm.price_out is None
    assert getattr(llm, "budget_usd", None) is None and any(model in x for x in lines)


# ---------------------------------------------------------------- 4 the reading rules
def v(S, V, A, **kw): return L.verdict(dict(S=S, V=V, A=A), kw.get("replays", dict(S=True, V=True, A=True)))
def res(x): return {i: c for i, _, _, c in x}
def mk(n_exact, **kw): return arm(n_exact, n=300, **kw)


def test_every_reading_line_flips_at_its_boundary():
    A = mk(300, comps=0, prompt=2800); S = mk(294, comps=12, prompt=17000); V = mk(250, comps=12)
    r = res(v(S, V, A)); assert all(r[i] for i in range(1, 6)) and len(r) == len(L.LINES) == 14
    assert r[6] is True and r[7] is True and r[8] is False                                          # 294/300 = 98.0 % reads yes, 250/300 does not
    assert r[9] is False and r[10] is True                                                          # AXI 100 % against S 98.0 % (2 points), against V 83.3 %
    assert r[11] is True and r[12] is False                                                         # the last step of S is wrong; its first slip is turn 295
    assert res(v(mk(293, comps=12), V, A))[7] is False
    assert res(v(mk(285, comps=12), V, A))[9] is True and res(v(mk(286, comps=12), V, A))[9] is False   # 100 % - 95.0 % = 5 points reads yes, 4.7 does not
    assert res(v(mk(300, comps=12, final_exact=False), V, A))[11] is True and res(v(mk(300, comps=12), V, A))[11] is False
    assert res(v(mk(300, comps=12, first_slip=99), V, A))[12] is True and res(v(mk(300, comps=12, first_slip=100), V, A))[12] is False and res(v(mk(300, comps=12), V, A))[12] is False
    assert res(v(S, V, A, replays=dict(S=False, V=True, A=True)))[1] is False and res(v(S, V, A, replays=dict(S=True, V=False, A=True)))[2] is False and res(v(S, V, A, replays=dict(S=True, V=True, A=False)))[3] is False
    assert res(v(dict(S, rows=S["rows"][:299]), V, A))[1] is False and res(v(S, V, dict(A, rows=A["rows"][:299])))[3] is False
    assert res(v(mk(294, comps=8), V, A))[4] is True and res(v(mk(294, comps=7), V, A))[4] is False and res(v(S, mk(250, comps=7), A))[4] is False and res(v(mk(294, comps=12, empty=True), V, A))[4] is False
    assert res(v(mk(294, comps=12, parsed_share=0.98), V, A))[5] is True and res(v(mk(294, comps=12, parsed_share=0.97), V, A))[5] is False
    assert res(v(S, mk(250, comps=12, parsed_share=0.97), A))[5] is False
    refused = lambda k: dict(A, rows=[dict(x, parsed=(i >= k)) for i, x in enumerate(A["rows"])])
    assert res(v(S, V, refused(6)))[5] is True and res(v(S, V, refused(7)))[5] is False             # 6 refused of 300 = 2.0 % reads yes, 7 does not
    assert res(v(S, V, mk(294, comps=0)))[6] is True and res(v(S, V, mk(293, comps=0)))[6] is False
    none = res(v(S, V, None)); assert none[3] is None and none[5] is None and none[6] is None and none[9] is None and none[13] is None and none[1] is True


def test_the_token_line_and_the_time_line_at_their_boundaries():
    A = mk(300, comps=0, prompt=1000)
    assert res(v(mk(300, comps=0, prompt=3000), mk(300, comps=0), A))[13] is True and res(v(mk(300, comps=0, prompt=2999), mk(300, comps=0), A))[13] is False        # 3.0 times reads yes
    # the summary calls count in the step's prompt: 12 summaries of 30,000 tokens over 300 steps add 1,200 a step
    assert res(v(mk(300, comps=12, prompt=1800), mk(300, comps=0), A))[13] is True and res(v(mk(300, comps=12, prompt=1799), mk(300, comps=0), A))[13] is False
    assert res(v(mk(300, comps=0, sec=3.0), mk(300, comps=0), mk(300, comps=0, sec=3.0)))[14] is True and res(v(mk(300, comps=0, sec=3.001), mk(300, comps=0), mk(300, comps=0, sec=3.0)))[14] is False
    assert res(v(mk(300, comps=12, sec=2.9), mk(300, comps=0), mk(300, comps=0, sec=3.0)))[14] is True       # 2.9 s + 12 summary calls of 2.0 s over 300 steps = 2.98 s
    assert res(v(mk(300, comps=12, sec=2.93), mk(300, comps=0), mk(300, comps=0, sec=3.0)))[14] is False     # 2.93 + 0.08 = 3.01 s


def test_the_headline_and_the_rule_on_claims():
    A = mk(300, comps=0, prompt=2800)
    assert L.headline(v(mk(250, comps=12), mk(250, comps=12), A)) == "AXI HOLDS THE STATE BETTER THAN EACH COMPACTION ARM"
    h = L.headline(v(mk(300, comps=12), mk(250, comps=12), A)); assert h.startswith("MIXED") and "line 9 NO" in h and "line 6 yes" in h and "line 10 yes" in h
    assert "arm S" in L.headline(v(mk(299, comps=12), mk(250, comps=12), mk(300, comps=0))) or L.headline(v(mk(299, comps=12), mk(250, comps=12), mk(300, comps=0))).startswith("MIXED")
    t = L.headline(v(mk(100, comps=12), mk(100, comps=12), mk(300, comps=0))); assert t == "AXI HOLDS THE STATE BETTER THAN EACH COMPACTION ARM"
    w = L.headline(v(mk(280, comps=12), mk(100, comps=12), mk(300, comps=0))); assert w == "AXI HOLDS THE STATE BETTER THAN EACH COMPACTION ARM"
    assert L.headline(v(mk(250, comps=12), mk(250, comps=12), None)).startswith("NOT READ")
    assert L.claims_allowed(v(mk(294, comps=12), mk(250, comps=12), A))["base"] is True and L.claims_allowed(v(mk(294, comps=12), mk(250, comps=12), A, replays=dict(S=True, V=True, A=False)))["base"] is False
    assert L.claims_allowed(v(mk(294, comps=12), mk(250, comps=12), None))["base"] is False


# ---------------------------------------------------------------- 5 the page and the notebook
def _page(): return open(PAGE).read()


def test_the_page_above_its_result_is_the_text_hashed_before_the_run_and_agrees_with_the_module():
    head = _page().split(RESULT_MARK)[0]; assert _h(head) == PAGE_SHA256
    rules = head.split("## Reading rules")[1].split("Headline:")[0]; listed = re.findall(r"^(\d+)\. (.*)$", rules, re.M)
    assert [t for _, t in listed] == [text for _, text in L.LINES] and [int(n) for n, _ in listed] == list(range(1, len(L.LINES) + 1))
    pred = head.split("## Predictions")[1].split("## Reading rules")[0]; got = re.findall(r"^(\d+)\. \[(\w+)\] (.*) \((\d+) %\)$", pred, re.M)
    assert [(int(n), tag, text) for n, tag, text, _ in got] == [(i + 1, tag, text) for i, (tag, text) in enumerate(L.LINES)] and all(0 < int(p) < 100 for *_, p in got)
    for name in ("S", "V", "A"): assert f"`{name}`" in head
    for must in ("32,000", "300 steps", "20261111", "gemini-3.5-flash-lite", "Nothing on this page has been run", "host side", "Rajnish", "500 calls a day", "480 calls", "6 seconds", "against AXI"): assert must in head, must
    assert "L" not in re.findall(r"\| `(\w)` \|", head) and "(`COMPACT_GENERIC`" in head and "(`COMPACT_VERBATIM`" in head
    for banned in ("open source", "open-source", "proven", "we cracked", "revolution", "prototype", "blueprint"): assert banned not in head.lower(), banned
    assert all(ord(c) < 128 for c in head), "a non-ASCII character (a dash?) is in the page"


def test_the_compaction_prompts_the_page_names_are_the_ones_in_the_code():
    assert C.COMPACT_GENERIC and C.COMPACT_VERBATIM and C.COMPACT_GENERIC != C.COMPACT_VERBATIM and "complete current service graph" in C.COMPACT_VERBATIM


def test_the_notebook_holds_no_key_no_price_no_budget_and_never_turns_billing_on():
    nb = json.load(open(NB)); src = "\n".join("".join(c["source"]) for c in nb["cells"]); code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert not re.search(r"AIza[0-9A-Za-z_-]{20,}", src) and "GOOGLE_API_KEY" in src and "userdata.get" in src
    for banned in ("PAID", "PRICE_", "BUDGET", "billing", "need_budget"): assert banned not in code, banned
    assert "L.build_lite_llm" in code and 'L.run_arm(llm, "S"' in code and 'L.run_arm(llm, "V"' in code and 'L.run_arm(llm, "A"' in code and "L.verdict" in code and "L.replay_arm" in code and "L.print_verdict" in code
    assert all(c["outputs"] == [] for c in nb["cells"] if c["cell_type"] == "code") and "cannot turn billing on" in src and "gemini_lite_compaction.py" in code and "LITE_COMPACTION_TEST.md" in code
    assert all(ord(c) < 128 for c in src), "a non-ASCII character (a dash?) is in the notebook"
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            body = "".join(c["source"]); body = "\n".join(l for l in body.split("\n") if not l.lstrip().startswith("!")); compile(body, "cell", "exec")
