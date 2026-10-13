"""Implements concept-map item 90 (the confirmation test; rows 3, 9, 27): the INSTRUMENT of docs/CONFIRM_TEST.md, written BEFORE any model run. It checks the instrument, not a result:
  1. the prompt wrapper puts the one added example right after the system text (before the rule), changes nothing else and never alters the caller's messages;
  2. the task lists are exactly the ones the page fixed by hash (280 single requests, 30 long scripts of 30 steps), none of them occurs under the seeds of earlier runs, and the hash changes if a seed changes;
  3. the statistics: Wilson intervals at known values, the episode bootstrap is deterministic, resamples whole episodes (one episode gives a zero-width interval), and narrows with more episodes; the wasted-token
     count is exact on hand-built rows;
  4. the analysis of real runs on stand-in models: a careful model leaves every policy exact; a model that forgets a service's edges corrupts blind apply in every episode and nobody else; a model that adds an
     unasked edge delete costs repair_m6 exact steps and not scope_m7; and every prompt the model saw carried the added example;
  5. every reading-rule line is exact at its boundary (95 and 90 percent, 3 points, the Wilson bound at 14 of 30, 10 points and a lower bound above 0, 10 and 2 percent of tokens, 5 points of 280, 358 and 658 per 1000,
     5 percent unparsed), a regime that was not run is skipped, and a line flips when one number is doctored;
  6. the page and the module agree (the same lines, hashes, seeds, resamples) and the page's text above its Result is byte for byte what was written before any run (a hash);
  7. the notebook runs end to end on stand-ins for Colab and Gemini, stops at the daily limit in the middle of a part, finishes the next day with the same rows as a run that never stopped, asks nothing twice,
     sends every prompt with the added example, prints the reading-rule lines, downloads its files and never writes the key; `PAID` switches the daily limit and the pacing.
What it does NOT show: that Google's service behaves like the stand-in, or anything about a real model's replies (the pre-registered run)."""
import contextlib, hashlib, io, json, os, re, sys, tempfile, time
from unittest import mock
sys.path.insert(0, os.path.dirname(__file__))
from axi.experiments import confirm_test as C
from axi.experiments.agent_loop import RULES_HINT, SYSTEM, _serialize, base_messages
from axi.experiments.long_horizon import oracle_script
from axi.experiments.scope_tasks import KINDS_SCOPE, make_scope_tasks
from test_api_gemini import Clock, _fake_world, KEY, read, ts, _no_bang, _src
from test_colab_persist import patched_modules
from test_scope import StandIn

HERE = os.path.dirname(__file__)
PAGE = os.path.join(HERE, "..", "..", "docs", "CONFIRM_TEST.md")
NB = os.path.join(HERE, "..", "colab", "axi_confirm_test.ipynb")
PAGE_SHA256 = "47b0c4e457c4fbd462279dd8d25228cbad3036450c06b4a0af7849d0d4c938a6"          # filled in when the page is written, before any run
RESULT_MARK = "\n\n---\n\n# Result"


def _h(text): return hashlib.sha256((text.rstrip("\n") + "\n").encode()).hexdigest()


# ---------------------------------------------------------------- 1 the wrapper
def test_the_added_example_goes_right_after_the_system_text_and_nothing_else_changes():
    seen, flags = [], []
    class Rec:
        last_prompt_tokens = 77
        def generate(self, messages, constrained): seen.append(messages); flags.append(constrained); return "DEL[0,0]", 3, 0.1
    n = C.Noted(Rec()); t = make_scope_tasks(7, 21)[0]
    for hint in (False, True):
        m = base_messages(t, hint); before = json.loads(json.dumps(m)); n.generate(m, True); got = seen[-1]
        assert m == before, "the caller's messages must not change"
        assert got[0]["content"] == SYSTEM + C.NOTE + (RULES_HINT if hint else "") and got[1] == m[1] and len(got) == len(m)
    n.generate(m); assert flags[-1] is True; n.generate(m, False); assert flags[-1] is False                          # the flag the caller gives (or its default) is passed on unchanged
    assert n.last_prompt_tokens == 77 and getattr(n, "last_cached", "absent") == "absent"
    assert C.NOTE.count("\n") == 1 and "which has no edges" in C.NOTE and C.NOTE.rstrip().endswith("DEL[2,1]")
    try: n.generate([{"role": "system", "content": "something else"}, {"role": "user", "content": "x"}], True); raise AssertionError("an unexpected system text was accepted")
    except AssertionError as e:
        assert "unexpected system text" in str(e)


# ---------------------------------------------------------------- 2 the task lists
def test_the_task_lists_are_the_fixed_ones_and_fresh():
    fp = C.fingerprints(); assert (fp["n_single"], fp["episodes"], fp["steps"]) == (280, 30, 900) and C.fingerprints() == fp
    ts_ = make_scope_tasks(280, 21); assert [t.kind for t in ts_] == [KINDS_SCOPE[i % 7] for i in range(280)] and all(sum(t.kind == k for t in ts_) == 40 for k in KINDS_SCOPE)
    assert [t.instruction for t in make_scope_tasks(14, 21)] == [t.instruction for t in ts_[:14]]                       # a shorter list is a prefix (the notebook test relies on it)
    new = {(_serialize(t.graph), t.instruction, t.ideal) for t in ts_}; g_new = {_serialize(oracle_script(C.SEED_LONG * 1000 + e, C.STEPS, "scope")[0]) for e in range(C.EPISODES)}
    assert len(new) == 280 and len(g_new) == 30
    for seed in (0, 1, 2, 3, 7):
        assert not (new & {(_serialize(t.graph), t.instruction, t.ideal) for t in make_scope_tasks(280, seed)}), seed
        assert not (g_new & {_serialize(oracle_script(seed * 1000 + e, 30, "scope")[0]) for e in range(30)}), seed
    with mock.patch.object(C, "SEED_SINGLE", 22): assert C.fingerprints()["single"] != fp["single"] and C.fingerprints()["long"] == fp["long"]
    with mock.patch.object(C, "SEED_LONG", 22): assert C.fingerprints()["long"] != fp["long"] and C.fingerprints()["single"] == fp["single"]
    assert (C.SEED_SINGLE, C.N_SINGLE, C.SEED_LONG, C.EPISODES, C.STEPS, C.BOOT_RESAMPLES, C.BOOT_SEED) == (21, 280, 21, 30, 30, 2000, 20261010) and C.POLICIES_LONG == ("blind", "repair_m6", "scope_m7")


# ---------------------------------------------------------------- 3 statistics
def test_wilson_at_known_values():
    lo, hi = C.wilson(15, 30); assert abs(lo - 0.3315) < 2e-3 and abs(hi - 0.6685) < 2e-3
    lo, hi = C.wilson(0, 30); assert lo == 0.0 and abs(hi - 0.1135) < 2e-3
    lo, hi = C.wilson(30, 30); assert hi == 1.0 and abs(lo - 0.8865) < 2e-3
    assert C.wilson(0, 0) == (0.0, 1.0)
    assert C.wilson(5, 10)[1] - C.wilson(5, 10)[0] > C.wilson(50, 100)[1] - C.wilson(50, 100)[0]            # more data, narrower
    assert C.wilson(14, 30)[0] >= 0.30 > C.wilson(13, 30)[0]                                                      # the boundary the reading rule sits on


def test_the_bootstrap_resamples_whole_episodes_and_is_deterministic():
    pairs = [(30, 30)] * 20 + [(0, 30)] * 10
    a = C.boot_ratio(pairs); assert a == C.boot_ratio(pairs) and abs(a[0] - 20 / 30) < 1e-12 and a[1] < a[0] < a[2]
    fine = [(i % 29, 30) for i in range(30)]; assert C.boot_ratio(fine, seed=1) != C.boot_ratio(fine) and C.boot_ratio(fine, seed=1) == C.boot_ratio(fine, seed=1)      # the generator seed matters (two-valued episodes give the same percentiles under any seed)
    one = C.boot_ratio([(7, 10)]); assert one == (0.7, 0.7, 0.7)                                                  # a single episode cannot be resampled into spread
    same = C.boot_ratio([(10, 10)] * 12); assert same == (1.0, 1.0, 1.0)
    wide, narrow = C.boot_ratio(([(30, 30)] * 2 + [(0, 30)] * 1)), C.boot_ratio(([(30, 30)] * 20 + [(0, 30)] * 10)); assert wide[2] - wide[1] > narrow[2] - narrow[1]
    assert all(x != x for x in C.boot_ratio([])) and all(x != x for x in C.boot_ratio([(0, 0)]))               # nan, no division by zero
    lo, hi = C.boot_ratio([(0, 10), (10, 10)] * 3)[1:]; assert 0.0 <= lo < hi <= 1.0


def test_bootstrap_percentiles_point_estimate_and_empty_resamples_are_exact():
    seq = [0, 0] * 26 + [0, 1] * 949 + [1, 1] * 25                # 1000 resamples of two episodes: 26 give 0.0, 949 give 0.5, 25 give 1.0
    class Scripted:
        def __init__(self, seed): self.it = iter(seq)
        def randrange(self, n): return next(self.it)
    with mock.patch.object(C.random, "Random", Scripted): assert C.boot_ratio([(0, 10), (10, 10)], resamples=1000) == (0.5, 0.0, 1.0)      # the 2.5 and 97.5 percent places of the sorted list
    assert C.boot_ratio([(1, 2), (3, 10)])[0] == 4 / 12                                                          # a ratio of sums, not a mean of ratios
    assert C.boot_ratio([(0, 0)] * 3 + [(5, 5)]) == (1.0, 1.0, 1.0)                                              # a resample with no steps at all is left out, not counted as 0


def _step(exact, corrupted=False, tokens=10, prompt=100): return dict(exact=exact, corrupted=corrupted, tokens=tokens, prompt_tokens=prompt)


def test_wasted_tokens_are_counted_from_the_step_after_the_first_wrong_one():
    rows = [dict(policies={"blind": [_step(True), _step(False), _step(False, tokens=20), _step(True, prompt=50)], "repair_m6": [_step(True)] * 4}),
            dict(policies={"blind": [_step(False), _step(True, prompt=None, tokens=None)], "repair_m6": [_step(True)] * 2})]
    # episode 1: steps 3 and 4 start from a wrong state (the step before was wrong): (100+20) + (50+10); episode 2: step 2 starts from a wrong state: 0 tokens known (None counts as 0)
    assert C.wasted_tokens(rows, "blind") == (120 + 60, 110 + 110 + 120 + 60 + 110 + 0) and C.wasted_tokens(rows, "repair_m6") == (0, 110 * 6)
    assert C.wasted_tokens([], "blind") == (0, 0)


# ---------------------------------------------------------------- 4 the analysis of real runs on stand-ins
class Counting:
    def __init__(self, inner): self.inner, self.n, self.systems, self.last_prompt_tokens = inner, 0, [], None
    def generate(self, messages, constrained, max_new_tokens=400):
        self.n += 1; self.systems.append(messages[0]["content"]); out = self.inner.generate(messages, constrained); self.last_prompt_tokens = self.inner.last_prompt_tokens; return out


def test_the_analysis_of_runs_on_stand_in_models():
    tmp = tempfile.mkdtemp(); res = {}
    for style in ("ideal", "no_rule_like", "rule_like"):
        llm = Counting(StandIn(style)); res[style] = C.run_long_confirm(llm, False, os.path.join(tmp, style + ".jsonl"), episodes=4); a = res[style]["analysis"]
        assert llm.n > 0 and all(s.startswith(SYSTEM + C.NOTE) for s in llm.systems), "every prompt the model saw carries the added example"
        assert json.loads(json.dumps(a)) == json.loads(json.dumps(a)) and all(a[p]["episodes"] == 4 and a[p]["steps"] == 120 for p in C.POLICIES_LONG)
    a = res["ideal"]["analysis"]
    for p in C.POLICIES_LONG: assert a[p]["exact"] == 120 and a[p]["ever_corrupted"] == 0 and a[p]["corrupted_steps"] == 0 and a[p]["wasted_tokens"] == 0 and a[p]["exact_ci"] == [1.0, 1.0]
    assert a["repair_m6"]["gap_vs_blind"]["num"] == 0
    a = res["no_rule_like"]["analysis"]                           # forgets the edges of a deleted service: only blind apply is hurt
    assert a["blind"]["ever_corrupted"] == 4 and a["blind"]["corrupted_steps"] > 0 and a["blind"]["exact"] < 120 and a["blind"]["wasted_tokens"] > 0
    for p in ("repair_m6", "scope_m7"): assert a[p]["exact"] == 120 and a[p]["corrupted_steps"] == 0 and a[p]["ever_corrupted"] == 0 and a[p]["wasted_tokens"] == 0
    g = a["repair_m6"]["gap_vs_blind"]; assert g["num"] == 120 - a["blind"]["exact"] + 120 * 3 - 120 * 3 and g["den"] == 120 and g["lower"] <= g["num"] / g["den"] <= g["upper"] and g["lower"] > 0
    a = res["rule_like"]["analysis"]                              # adds one unasked edge delete: repair_m6 keeps it (and drifts), scope_m7 drops it
    assert a["scope_m7"]["exact"] == 120 and a["repair_m6"]["exact"] < 120 and a["repair_m6"]["corrupted_steps"] == 0 and a["blind"]["corrupted_steps"] == 0
    assert a["repair_m6"]["wasted_tokens"] > 0 and a["scope_m7"]["wasted_tokens"] == 0
    # the single runner carries the example too, and a longer list is the prefix-extension of a shorter one
    llm = Counting(StandIn("ideal")); r = C.run_single(llm, True, os.path.join(tmp, "s.jsonl"), tasks=make_scope_tasks(14, 21)); assert len(r["rows"]) == 14 and r["meta"]["confirm"] is True
    assert all(s == SYSTEM + C.NOTE + RULES_HINT for s in llm.systems) and r["summary"]["all"]["repair_m6"]["success"] == 14


def test_long_analysis_counts_exactly_on_hand_built_rows():
    st = lambda exact, corrupted=False: _step(exact, corrupted)
    rows = [dict(policies={"blind": [st(True), st(False, True), st(False, True)], "repair_m6": [st(True), st(True), st(False, True)], "scope_m7": [st(True)] * 3}),
            dict(policies={"blind": [st(True)] * 3, "repair_m6": [st(True)] * 3, "scope_m7": [st(False), st(True), st(True)]})]
    a = C.long_analysis(rows)
    b, m, c = a["blind"], a["repair_m6"], a["scope_m7"]
    assert (b["episodes"], b["steps"], b["exact"], b["ever_corrupted"], b["corrupted_steps"], b["wasted_tokens"], b["tokens"]) == (2, 6, 4, 1, 2, 110, 660)
    assert (m["episodes"], m["steps"], m["exact"], m["ever_corrupted"], m["corrupted_steps"], m["wasted_tokens"], m["tokens"]) == (2, 6, 5, 1, 1, 0, 660), "a corruption at the last step of an episode counts"
    assert (c["steps"], c["exact"], c["ever_corrupted"], c["corrupted_steps"], c["wasted_tokens"]) == (6, 5, 0, 0, 110)
    assert (m["gap_vs_blind"]["num"], m["gap_vs_blind"]["den"]) == (1, 6) and (c["gap_vs_blind"]["num"], c["gap_vs_blind"]["den"]) == (1, 6) and "gap_vs_blind" not in b
    for p in (b, m, c): assert 0 <= p["exact_ci"][0] <= p["exact"] / p["steps"] <= p["exact_ci"][1] <= 1
    for p in (m, c): g = p["gap_vs_blind"]; assert g["lower"] <= g["num"] / g["den"] <= g["upper"]


def test_the_runners_use_the_fixed_seeds_the_fixed_scripts_and_the_chosen_regime():
    tmp = tempfile.mkdtemp()
    for hint in (False, True):
        llm = Counting(StandIn("ideal")); r = C.run_long_confirm(llm, hint, os.path.join(tmp, f"l{hint}.jsonl"), episodes=3); meta = r["meta"]
        assert (meta["mode"], meta["seed"], meta["steps"], meta["episodes"], meta["mix"], meta["policies"]) == ("state", 21, 30, 3, "scope", list(C.POLICIES_LONG)) and meta["hint_rules"] is hint and meta["confirm"] is True
        assert [x["seed"] for x in r["rows"]] == [21000, 21001, 21002]
        for e, row in enumerate(r["rows"]):
            _, script, _ = oracle_script(21000 + e, 30, "scope"); assert [x["kind"] for x in row["policies"]["blind"]] == [x["kind"] for x in script] and len(script) == 30
        assert llm.n > 0 and all((RULES_HINT in x) is hint for x in llm.systems)
        llm = Counting(StandIn("ideal")); C.run_single(llm, hint, os.path.join(tmp, f"s{hint}.jsonl"), tasks=make_scope_tasks(14, 21)); assert llm.n > 0 and all((RULES_HINT in x) is hint for x in llm.systems)
    with mock.patch.object(C, "N_SINGLE", 14):
        r = C.run_single(StandIn("ideal"), False, os.path.join(tmp, "d.jsonl")); assert [x["kind"] for x in r["rows"]] == [t.kind for t in make_scope_tasks(14, 21)]


def test_the_printed_tables_show_the_numbers_and_the_verdict_marks():
    a = _L(blind=_pol(exact=450, lo=0.40, ever=25, cs=400, w=2000, t=10000), repair_m6=_pol(exact=880, lo=0.97, w=0, t=10000, gap=dict(num=450, den=900, lower=0.4, upper=0.6)),
           scope_m7=_pol(exact=890, lo=0.98, w=50, t=10000, gap=dict(num=-18, den=900, lower=-0.05, upper=0.01)))
    a["blind"]["exact_ci"] = [0.40, 0.55]
    out = io.StringIO()
    with contextlib.redirect_stdout(out): C.print_confirm_long(a, "PART X")
    t = out.getvalue()
    assert "== long runs, PART X (30 episodes) ==" in t
    assert re.search(r"blind\s+50\.0%\s+40\.0 to\s+55\.0%\s+25 of 30\s+\(\s*66 to\s+93%\)\s+400\s+20\.0% of 10,000", t), t
    assert re.search(r"repair_m6\s+97\.8%\s+97\.0 to\s+98\.0%\s+0 of 30 .*\s+0\s+0\.0% of 10,000", t), t
    assert "repair_m6 minus blind apply, exact steps: +50.0 points (95% bound +40.0 to +60.0)" in t and "scope_m7 minus blind apply, exact steps: -2.0 points (95% bound -5.0 to +1.0)" in t
    out = io.StringIO()
    with contextlib.redirect_stdout(out): C.print_verdict([("alpha line", True), ("beta line", False)], "all four")
    t = out.getvalue()
    assert "-- all four (reading rule fixed before the run) --" in t and re.search(r"^  yes  alpha line$", t, re.M) and re.search(r"^  NO   beta line$", t, re.M), t


def test_the_waste_line_needs_tokens_to_have_been_counted():
    l = {False: _L(blind=_pol(exact=450, lo=0.4, ever=25, cs=400, w=0, t=0), repair_m6=_pol(w=0, t=0, gap=dict(num=450, den=900, lower=0.4, upper=0.6)))}
    assert _v(None, l)[C.L_WASTE] is False


# ---------------------------------------------------------------- 5 the reading rules
def _pol(exact=900, lo=0.99, ever=0, cs=0, w=0, t=10000, steps=900, gap=None):
    d = dict(episodes=30, steps=steps, exact=exact, exact_ci=[lo, min(1.0, lo + 0.01)], ever_corrupted=ever, corrupted_steps=cs, wasted_tokens=w, tokens=t)
    if gap is not None: d["gap_vs_blind"] = gap
    return d


def _L(**over):
    a = dict(blind=_pol(exact=450, lo=0.40, ever=25, cs=400, w=2000, t=10000), repair_m6=_pol(gap=dict(num=450, den=900, lower=0.4, upper=0.6)), scope_m7=_pol(gap=dict(num=450, den=900, lower=0.4, upper=0.6)))
    a.update(over); return a


def _S(**over):
    s = dict(n=280, blind=dict(success=250, corrupted=10), gate=dict(success=268, corrupted=0), repair_m6=dict(success=270, corrupted=0), scope_m7=dict(success=270, corrupted=0)); s.update(over); return dict(all=s)


def _v(single=None, long_=None, **kw): return dict(C.verdict_confirm(single, long_, **kw))
def _only(checks, text): return [ok for t, ok in checks if t == text]


def test_every_reading_rule_is_exact_at_its_boundary():
    both = lambda x: {False: x, True: x}
    ok = C.verdict_confirm({False: _S(), True: _S()}, both(_L()))
    assert [t for t, _ in ok] == [C.L_SINGLE_GUARD, C.L_LONG_GUARD, C.L_M6, C.L_M7, C.L_M7_M6, C.L_EVER.format(r="no rule"), C.L_EVER.format(r="rule stated"), C.L_GAP.format(r="no rule"), C.L_GAP.format(r="rule stated"),
                                  C.L_WASTE, C.L_SINGLE_NEED, C.L_REPLICATE] and all(x for _, x in ok), ok
    # single guard: any corrupted graph under gate, repair_m6 or scope_m7, in either regime
    for p in ("gate", "repair_m6", "scope_m7"):
        for h in (False, True):
            s = {False: _S(), True: _S()}; s[h] = _S(**{p: dict(success=270, corrupted=1)}); assert _v(s, both(_L()))[C.L_SINGLE_GUARD] is False
    s = {False: _S(blind=dict(success=250, corrupted=99)), True: _S()}; assert _v(s, both(_L()))[C.L_SINGLE_GUARD] is True                  # blind apply may corrupt
    # long guard: one corrupted step of repair_m6 or scope_m7 in any regime
    for p in ("repair_m6", "scope_m7"):
        for h in (False, True):
            l = {False: _L(), True: _L()}; l[h] = _L(**{p: _pol(cs=1, gap=dict(num=450, den=900, lower=0.4, upper=0.6))}); assert _v({False: _S()}, l)[C.L_LONG_GUARD] is False
    # 95 percent of 900 steps = 855; the lower bound 0.90
    for line, p in ((C.L_M6, "repair_m6"), (C.L_M7, "scope_m7")):
        g = dict(num=450, den=900, lower=0.4, upper=0.6)
        for exact, lo, want in ((855, 0.90, True), (854, 0.95, False), (900, 0.8999, False), (900, 0.90, True)):
            assert _v(None, both(_L(**{p: _pol(exact=exact, lo=lo, gap=g)})))[line] is want, (p, exact, lo)
        l = {False: _L(), True: _L(**{p: _pol(exact=854, gap=g)})}; assert _v(None, l)[line] is False                                      # one failing regime fails the line
    # scope_m7 behind repair_m6 by 3 points of 900 = 27 steps
    g = dict(num=450, den=900, lower=0.4, upper=0.6)
    assert _v(None, both(_L(scope_m7=_pol(exact=873, gap=g))))[C.L_M7_M6] is True and _v(None, both(_L(scope_m7=_pol(exact=872, gap=g))))[C.L_M7_M6] is False
    assert _v(None, both(_L(scope_m7=_pol(exact=900, gap=g), repair_m6=_pol(exact=873, gap=g))))[C.L_M7_M6] is True                                 # ahead is never a problem
    # episodes ever corrupted: Wilson lower bound at least 30 percent (14 of 30)
    for h, name in ((False, "no rule"), (True, "rule stated")):
        line = C.L_EVER.format(r=name)
        for k, want in ((14, True), (13, False)):
            l = {False: _L(), True: _L()}; l[h] = _L(blind=_pol(exact=450, lo=0.4, ever=k, cs=400, w=2000)); assert _v(None, l)[line] is want, (name, k)
    # gap: at least 10 points (90 of 900) and a lower bound above 0
    for h, name in ((False, "no rule"), (True, "rule stated")):
        line = C.L_GAP.format(r=name)
        for num, lower, want in ((90, 0.01, True), (89, 0.5, False), (500, 0.0, False), (500, 1e-9, True)):
            gp = dict(num=num, den=900, lower=lower, upper=1.0); l = {False: _L(), True: _L()}; l[h] = _L(repair_m6=_pol(gap=gp), scope_m7=_pol(gap=gp)); assert _v(None, l)[line] is want, (name, num, lower)
    # wasted tokens (no rule only): blind at least 10 percent, repair_m6 at most 2 percent
    for bw, mw, want in ((1000, 20, True), (999, 20, False), (1000, 21, False), (5000, 0, True)):
        l = {False: _L(blind=_pol(exact=450, lo=0.4, ever=25, cs=400, w=bw, t=10000), repair_m6=_pol(w=mw, t=1000, gap=dict(num=450, den=900, lower=0.4, upper=0.6)))}
        assert _v(None, l)[C.L_WASTE] is want, (bw, mw)
    bad = {False: _L(), True: _L(blind=_pol(exact=450, lo=0.4, ever=25, cs=400, w=0, t=10000))}; assert _v(None, bad)[C.L_WASTE] is True                    # the rule-stated regime does not enter this line
    # single requests: repair_m6 ahead of blind by 5 points of 280 = 14
    for blind, want in ((256, True), (257, False)): assert _v({False: _S(blind=dict(success=blind, corrupted=0), repair_m6=dict(success=270, corrupted=0))}, None)[C.L_SINGLE_NEED] is want
    assert C.L_SINGLE_NEED not in _v({True: _S()}, None)                                                                                            # only the no-rule regime has this line
    # replicate: 358 to 658 of 1000 steps
    for exact, want in ((358, True), (357, False), (658, True), (659, False)):
        assert _v(None, {False: _L(blind=_pol(exact=exact, steps=1000, lo=0.4, ever=25, cs=400, w=2000, t=10000))})[C.L_REPLICATE] is want, exact
    # format: at most 5 percent do not parse
    rows = lambda bad, n: [dict(text="garbage") if i < bad else dict(text="DEL[0,0]") for i in range(n)]
    for bad, want in ((5, True), (6, False)):
        v = C.verdict_confirm(None, None, {False: rows(bad, 100)}, {}); assert len(v) == 1 and v[0][1] is want and v[0][0].startswith("[format] at most 5 percent of the model's replies do not parse (")
    assert C.verdict_confirm() == []


def test_a_regime_that_was_not_run_is_skipped():
    v = C.verdict_confirm({False: _S()}, {False: _L()}); texts = [t for t, _ in v]
    assert not any("rule stated" in t for t in texts) and C.L_EVER.format(r="no rule") in texts and C.L_WASTE in texts and C.L_REPLICATE in texts
    v = C.verdict_confirm({True: _S()}, {True: _L()}); texts = [t for t, _ in v]
    assert C.L_WASTE not in texts and C.L_REPLICATE not in texts and C.L_SINGLE_NEED not in texts and C.L_EVER.format(r="no rule") not in texts and C.L_EVER.format(r="rule stated") in texts


# ---------------------------------------------------------------- 6 the page
def test_page_and_module_agree_and_the_page_text_is_locked():
    if not os.path.exists(PAGE): return                                          # the page is written right after the instrument; the test then goes live
    text = open(PAGE).read(); assert text.count(RESULT_MARK) <= 1; head = text[:text.index(RESULT_MARK)] if RESULT_MARK in text else text
    assert PAGE_SHA256 and _h(head) == PAGE_SHA256, "the predictions and reading rules must not change after the run; later notes go below them"
    fp = C.fingerprints(); assert fp["single"] in head and fp["long"] in head
    for must in ("make_scope_tasks(280, 21)", "21000 to 21029", "2,000 resamples", "generator seed 20261010", "Gemini 3.5 Flash Lite", "remove the service at (2,1), which has no edges -> DEL[2,1]",
                 "480 calls a day", "Thirteen lines", "## Predictions", "## Reading rules", "## What this cannot show", "falsifies"):
        assert must in head, must
    lines = [C.L_SINGLE_GUARD, C.L_LONG_GUARD, C.L_M6, C.L_M7, C.L_M7_M6] + [C.L_EVER.format(r=r) for r in ("no rule", "rule stated")] + [C.L_GAP.format(r=r) for r in ("no rule", "rule stated")] + [C.L_WASTE, C.L_SINGLE_NEED, C.L_REPLICATE]
    for l in lines: assert l in head, l
    assert "[format] at most 5 percent of the model's replies do not parse" in head
    assert len(lines) + 1 == 13
    preds, rules = head.split("## Reading rules")[0], head.split("## Reading rules")[1].split("## What this cannot show")[0]
    tagged = lambda part: [l for l in part.split("\n") if re.match(r"^\d+\. \[(guard|need|replicate|format)\]", l)]
    assert len(tagged(rules)) == 13 and len(tagged(preds)) == 13, (len(tagged(rules)), len(tagged(preds)))
    assert "50.8 percent" in head and "46.3" in head and all(f"({p} %)" in head or f"{p} %" in head for p in (99, 60, 90, 93, 92, 80, 55, 65))
    assert "12 %" in head and "40 %" in head and str(C.BOOT_SEED) in head and "44 replies" in head and C.NOTE.strip().split("->")[0].strip() in head


# ---------------------------------------------------------------- 7 the notebook
def _code_index(contains):
    nb = json.load(open(NB)); return [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code" and contains in _src(c)][0]


def test_the_notebook_says_what_the_page_says():
    nb = json.load(open(NB)); cells = nb["cells"]; code = "".join(_src(c) for c in cells if c["cell_type"] == "code"); text = "".join(_src(c) for c in cells if c["cell_type"] == "markdown")
    assert 'WANTED = "gemini-3.5-flash-lite"' in code and "PAID = False" in code and "DAILY_FREE = 480" in code and "daily_limit=(100000 if PAID else DAILY_FREE)" in code and "min_interval=(1.0 if PAID else 4.5)" in code
    assert code.count("run_single(llm,") == 3 and code.count("run_long_confirm(llm,") == 2 and "run_single(llm, False" in code and "run_single(llm, True" in code and "run_long_confirm(llm, False" in code and "run_long_confirm(llm, True" in code
    assert "verdict_confirm(single, long_," in code and "print_verdict(" in code and "print_confirm_long(" in code and "single_set()[:6]" in code
    for f in ("axi_confirm_single_norule_", "axi_confirm_single_rule_", "axi_confirm_long_norule_", "axi_confirm_long_rule_", "axi_confirm_cache_"): assert f in code and f in text, f
    assert "30 episodes" in text and "3,200 calls" in text and "480 calls a day" in text and "PAID = True" in text and "axi-colab-confirm.zip" in text and "Never paste it in the chat" in text and "gemini-3.8-flash" in text
    assert code.count("The cell above stopped for today. Run it again first.") == 3, "each of the later cells refuses to start until the one before it has finished"
    assert KEY not in json.dumps(nb) and "AIza" not in json.dumps(nb) and "userdata.get" in code


def _run_notebook(cells_to_run, work, drive_root, clock, daily=None, paid=None, ns=None, world=None):
    nb = json.load(open(NB)); out = io.StringIO(); mods, st = world or _fake_world(clock); ns = {} if ns is None else ns; early = False
    with patched_modules(mods), contextlib.redirect_stdout(out), mock.patch.object(time, "time", clock.now), mock.patch.object(time, "sleep", clock.advance):
        for i in cells_to_run:
            c = nb["cells"][i]; assert c["cell_type"] == "code", i
            src = _no_bang(_src(c)).replace('open_store("/content/out", "axi_confirm")', f'open_store("{work}", "axi_confirm", "{drive_root}")').replace('"/content/smoke_confirm.jsonl"', f'"{work}/smoke_confirm.jsonl"')
            src = src.replace('sys.path.insert(0, "/content/axi/python")', "")
            if daily is not None: src = src.replace("DAILY_FREE = 480", f"DAILY_FREE = {daily}")
            if paid is not None: src = src.replace("PAID = False", f"PAID = {paid}")
            assert "/content" not in src, src
            try: exec(compile(src, f"cell{i}", "exec"), ns)
            except AssertionError as e:
                if "stopped for today" in str(e): early = True; break
                raise
    return ns, out.getvalue(), st, early


def test_the_notebook_runs_end_to_end_stops_at_the_limit_and_finishes_the_next_day_like_an_unstopped_run():
    cells = [_code_index(k) for k in ("files.upload()", "open_store(", "userdata.get", 'name = f"axi_confirm_single_norule_', 'name = f"axi_confirm_single_rule_', 'name = f"axi_confirm_long_norule_', 'name = f"axi_confirm_long_rule_')]
    tmp = tempfile.mkdtemp()
    with mock.patch.object(C, "EPISODES", 3), mock.patch.object(C, "N_SINGLE", 14):
        c = Clock(); work = os.path.join(tmp, "ref"); os.makedirs(work)
        ref, out_ref, st_ref, early = _run_notebook(cells, work, os.path.join(tmp, "drive_ref"), c, daily=10 ** 6); assert not early
        assert len(ref["smoke"]["rows"]) == 6 and ref["smoke"]["meta"]["smoke"] is True and "SMOKE OK" in out_ref
        total = st_ref.calls; assert total > 150 and ref["res_l_rule"] is not None
        assert all(sysi.startswith(SYSTEM + C.NOTE) for (_, sysi, _) in st_ref.prompts), "every prompt that reached the service carries the added example"
        assert sum(1 for (_, sysi, _) in st_ref.prompts if RULES_HINT in sysi) > 0 and sum(1 for (_, sysi, _) in st_ref.prompts if RULES_HINT not in sysi) > 0
        for limit in (60, total - 30):
            drive = os.path.join(tmp, f"drive_{limit}"); saved = os.path.join(drive, "MyDrive", "axi_confirm")
            c = Clock(); w1 = os.path.join(tmp, f"d1_{limit}"); os.makedirs(w1)
            ns1, out1, st1, early1 = _run_notebook(cells, w1, drive, c, daily=limit)
            assert "STOPPED FOR NOW" in out1 and "Daily budget reached" in out1 and "12:30 pm in India until 1 November, 1:30 pm after" in out1 and st1.calls == limit and ns1["res_l_rule"] is None
            assert KEY not in out1 and not any(KEY in read(os.path.join(d, f)) for d, _, fs in os.walk(tmp) for f in fs if f.endswith((".json", ".jsonl")))
            calls, outs, downloaded, days = st1.calls, [out1], list(st1.downloaded), 1; models, keys = set(st1.models), [st1.keys]
            while True:                                                                      # one new session per Pacific day until the last cell has finished
                days += 1; assert days <= 12, "did not finish"
                c.t = ts(2026, 10, 9 + days, 20); w2 = os.path.join(tmp, f"d{days}_{limit}"); os.makedirs(w2)
                ns2, out2, st2, early2 = _run_notebook(cells, w2, drive, c, daily=limit)
                calls += st2.calls; outs.append(out2); downloaded += st2.downloaded; models |= set(st2.models); keys.append(st2.keys)
                assert "reply file: restored" in out2 and "Drive connected" in out2 and st2.calls <= limit
                if ns2["res_l_rule"] is not None: break
            out2 = "".join(outs); assert calls == total and days >= 2, ("no prompt was asked twice across the days", limit, total, calls, days)
            for k in ("res_s_norule", "res_s_rule", "res_l_norule", "res_l_rule"): assert ns2[k]["rows"] == ref[k]["rows"] and ns2[k]["summary"] == ref[k]["summary"], (limit, k)
            for k in ("res_l_norule", "res_l_rule"): assert ns2[k]["analysis"] == ref[k]["analysis"]
            assert "all four parts (reading rule fixed before the run)" in out2 and "[format] at most 5 percent" in out2 and "[guard]" in out2 and "[need]" in out2 and "[replicate]" in out2 and "PART 2, no rule" in out2
            for f in ("axi_confirm_single_norule_gemini-3_5-flash-lite.json", "axi_confirm_single_rule_gemini-3_5-flash-lite.json", "axi_confirm_long_norule_gemini-3_5-flash-lite.json",
                      "axi_confirm_long_rule_gemini-3_5-flash-lite.json", "axi_confirm_cache_gemini-3_5-flash-lite.jsonl"): assert f in downloaded, f
            assert models == {"gemini-3.5-flash-lite"} and all(k == [KEY] for k in keys if k)
            assert os.path.exists(os.path.join(saved, "axi_confirm_cache_gemini-3_5-flash-lite.jsonl"))
        c.t = ts(2026, 10, 28, 20); w4 = os.path.join(tmp, "d4"); os.makedirs(w4)
        ns4, out4, st4, early4 = _run_notebook(cells, w4, drive, c, daily=0); assert not early4 and st4.calls == 0 and ns4["res_l_rule"]["rows"] == ref["res_l_rule"]["rows"]


def test_paid_switches_the_daily_limit_and_the_pacing():
    first = [_code_index(k) for k in ("files.upload()", "open_store(", "userdata.get")]
    for paid, limit, gap in ((False, 480, 4.5), (True, 100000, 1.0)):
        tmp = tempfile.mkdtemp(); c = Clock(); work = os.path.join(tmp, "w"); os.makedirs(work)
        ns, out, st, early = _run_notebook(first, work, os.path.join(tmp, "d"), c, paid=paid)
        assert ns["llm"].daily_limit == limit and ns["llm"].min_interval == gap and ns["PAID"] is paid, (paid, ns["llm"].daily_limit, ns["llm"].min_interval)


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn): fn(); print("PASS", name); n += 1
    print(f"{n} tests passed")
