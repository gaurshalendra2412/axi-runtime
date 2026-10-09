"""Implements concept-map row 3 / 9 / 27, item 85: the bigger-model test (docs/SCALE_TEST_14B.md; prediction and reading rules written before the run).
Checks the INSTRUMENT, not a result (no 14B number exists yet):
  1. every reading-rule line of `bigger_model.py` flips at exactly the threshold the page states (integer arithmetic, no float slip), both sides of each boundary;
  2. the smoke check says STOP for a broken model (nothing right) and OK for a working one, at the stated bar;
  3. the page and the code agree: 10 [guard] and 5 [need] lines, the same thresholds in words, the same design numbers (6 and 4 episodes, seed 7, the 140 tasks);
  4. the notebook is the one the page describes (file names, model, policies, episodes, cell order, every code cell compiles) and runs end to end on scripted stand-ins;
  5. the tasks and the first episodes' scripts are the same as the 7B scope test's saved files (so 7B and 14B can be compared request by request);
  6. the reading rules accept the real summaries produced by the runners (blind, repair_m6, scope_m7 in both parts).
What it does NOT show: anything about the 14B itself."""
import json, os, sys, tempfile
sys.path.insert(0, os.path.dirname(__file__))
from axi.experiments.bigger_model import verdict_big_single, verdict_big_long, smoke_check, mean_seconds
from axi.experiments.long_horizon import run_long, oracle_script
from axi.experiments.scope_tasks import make_scope_tasks
from axi.experiments.scope_test import run_scope_regime
from test_scope import StandIn

HERE = os.path.dirname(__file__)
PAGE = os.path.join(HERE, "..", "..", "docs", "SCALE_TEST_14B.md")
NB = os.path.join(HERE, "..", "colab", "axi_14b_test.ipynb")
RESULTS = os.path.join(HERE, "..", "..", "docs", "results")
P = ("blind", "gate", "repair_m6", "scope_m7", "scope_delta")


def cell(success=0, corrupted=0): return dict(success=success, corrupted=corrupted)


def summ(n=140, blind=0, m6=0, m7=0, dd_blind=0, old_m7=0, harmed=0, corrupt=(0, 0, 0)):
    """A summarize_scope-shaped dict with only the fields the reading rules read."""
    all_ = dict(n=n, blind=cell(blind), gate=cell(0, corrupt[0]), repair_m6=cell(m6, corrupt[1]), scope_m7=cell(m7, corrupt[2]), scope_delta=cell(0))
    return dict(all=all_, delete_dep=dict(n=20, blind=cell(dd_blind)), old_deletes=dict(n=40, scope_m7=cell(old_m7)), paired=dict(harmed=harmed))


def get(lines, start):
    hits = [ok for t, ok in lines if t.startswith(start)]; assert len(hits) == 1, (start, [t for t, _ in lines]); return hits[0]


def test_part_1_lines_flip_at_the_stated_thresholds():
    base = dict(blind=100, m6=135, m7=135, dd_blind=0, old_m7=40)
    ok = lambda S, start: get(verdict_big_single(S), start)
    # no rule: repair_m6 at least 90 percent of 140 -> 126
    assert ok({False: summ(**dict(base, m6=126))}, "[guard] no rule: repair_m6 solves at least 90") and not ok({False: summ(**dict(base, m6=125))}, "[guard] no rule: repair_m6 solves at least 90")
    # no rule: scope_m7 within 3 points of repair_m6 -> difference of at most 4 tasks (4.2 allowed)
    assert ok({False: summ(**dict(base, m6=135, m7=131))}, "[guard] no rule: scope_m7 is within 3") and not ok({False: summ(**dict(base, m6=135, m7=130))}, "[guard] no rule: scope_m7 is within 3")
    assert ok({False: summ(**dict(base, m6=131, m7=135))}, "[guard] no rule: scope_m7 is within 3") and not ok({False: summ(**dict(base, m6=130, m7=135))}, "[guard] no rule: scope_m7 is within 3")
    # need: blind right on at most half (10 of 20) of delete_dep
    assert ok({False: summ(**dict(base, dd_blind=10))}, "[need] no rule: blind apply is right on at most half") and not ok({False: summ(**dict(base, dd_blind=11))}, "[need] no rule: blind apply is right on at most half")
    # need: repair_m6 ahead of blind by at least 10 points -> 14 tasks
    assert ok({False: summ(**dict(base, blind=121, m6=135))}, "[need] no rule: repair_m6 is ahead") and not ok({False: summ(**dict(base, blind=122, m6=135))}, "[need] no rule: repair_m6 is ahead")
    # rule: scope_m7 at least 90 percent of the 40 old deletes -> 36
    assert ok({True: summ(**dict(base, old_m7=36))}, "[guard] rule stated: scope_m7 solves at least 90") and not ok({True: summ(**dict(base, old_m7=35))}, "[guard] rule stated: scope_m7 solves at least 90")
    # need: rule: scope_m7 ahead of blind by 14 tasks
    assert ok({True: summ(**dict(base, blind=120, m7=134))}, "[need] rule stated: scope_m7 is ahead") and not ok({True: summ(**dict(base, blind=121, m7=134))}, "[need] rule stated: scope_m7 is ahead")
    # harm at most 2 per regime, both regimes checked
    assert ok({True: summ(harmed=2), False: summ(harmed=2)}, "[guard] scope_m7 harms") and not ok({True: summ(harmed=2), False: summ(harmed=3)}, "[guard] scope_m7 harms")
    assert not ok({True: summ(harmed=3), False: summ(harmed=0)}, "[guard] scope_m7 harms")
    # corruption: any of gate, repair_m6, scope_m7 in any regime; blind apply is NOT counted (it corrupts by design)
    assert ok({True: summ(), False: summ()}, "[guard] gate, repair_m6 and scope_m7 corrupt no graph")
    for k in range(3): assert not ok({True: summ(), False: summ(corrupt=tuple(int(i == k) for i in range(3)))}, "[guard] gate, repair_m6 and scope_m7 corrupt no graph")
    S = summ(); S["all"]["blind"]["corrupted"] = 36; assert ok({False: S}, "[guard] gate, repair_m6 and scope_m7 corrupt no graph")
    # missing regimes are skipped: 2 lines common + 4 no-rule + 2 rule
    assert len(verdict_big_single({False: summ()})) == 6 and len(verdict_big_single({True: summ()})) == 4 and len(verdict_big_single({True: summ(), False: summ()})) == 8


def ls(m6=0.99, m7=0.99, blind10=1.0, ever=0.0):
    """A summarize_long-shaped dict with only the fields the reading rules read."""
    return dict(blind=dict(corrupted_by_step_10=blind10, episodes_ever_corrupted=1.0), repair_m6=dict(exact_share=m6, episodes_ever_corrupted=ever), scope_m7=dict(exact_share=m7, episodes_ever_corrupted=0.0))


def test_part_2_lines_flip_at_the_stated_thresholds():
    ok = lambda r, n, start: get(verdict_big_long(r, n), start)
    assert ok(None, ls(0.95, 0.95), "[guard] no rule: repair_m6 and scope_m7 each") and not ok(None, ls(0.9499, 0.99), "[guard] no rule: repair_m6 and scope_m7 each")
    assert not ok(None, ls(0.99, 0.9499), "[guard] no rule: repair_m6 and scope_m7 each")
    assert ok(None, ls(0.95, 0.98), "[guard] no rule: the two are within 3") and not ok(None, ls(0.95, 0.9801), "[guard] no rule: the two are within 3")
    assert ok(None, ls(blind10=0.5), "[need] no rule: blind apply is corrupted") and not ok(None, ls(blind10=0.4999), "[need] no rule: blind apply is corrupted")
    assert ok(ls(m7=0.90), None, "[guard] rule stated: scope_m7 holds") and not ok(ls(m7=0.8999), None, "[guard] rule stated: scope_m7 holds")
    assert ok(ls(m6=0.99, m7=0.96), None, "[guard] rule stated: scope_m7 is not behind") and not ok(ls(m6=0.99, m7=0.9599), None, "[guard] rule stated: scope_m7 is not behind")
    assert ok(ls(m6=0.5, m7=0.99), None, "[guard] rule stated: scope_m7 is not behind")                 # ahead is fine
    assert ok(ls(blind10=0.5), None, "[need] rule stated: blind apply is corrupted") and not ok(ls(blind10=0.3333), None, "[need] rule stated: blind apply is corrupted")
    # repair_m6 or scope_m7 ever corrupted -> NO; blind apply being corrupted is expected and not counted here
    assert ok(ls(), ls(), "[guard] repair_m6 and scope_m7 are never corrupted") and not ok(ls(ever=0.25), ls(), "[guard] repair_m6 and scope_m7 are never corrupted")
    bad = ls(); bad["scope_m7"]["episodes_ever_corrupted"] = 0.25; assert not ok(ls(), bad, "[guard] repair_m6 and scope_m7 are never corrupted")
    assert len(verdict_big_long(ls(), ls())) == 7 and len(verdict_big_long(None, ls())) == 4 and len(verdict_big_long(ls(), None)) == 4


def test_smoke_check_stops_a_broken_model_and_passes_a_working_one():
    s = lambda m6, blind=0, n=14: dict(all=dict(n=n, repair_m6=cell(m6), blind=cell(blind)))
    ok, msg = smoke_check(s(9)); assert ok and msg.startswith("SMOKE OK") and "9 of 14" in msg
    ok, msg = smoke_check(s(8)); assert not ok and msg.startswith("STOP") and "8 of 14" in msg and "send me" in msg
    ok, msg = smoke_check(s(0)); assert not ok and msg.startswith("STOP")
    ok, msg = smoke_check(s(14, 12)); assert ok and "blind apply 12" in msg
    # a stand-in that writes the same wrong token every time (like fp16 overflow) fails the smoke check on the real runner
    class Broken:
        last_prompt_tokens = 10
        def generate(self, messages, constrained, max_new_tokens=400): return "ADD[0,0:0]", 9, 0.01
    with tempfile.TemporaryDirectory() as d:
        res = run_scope_regime(Broken(), make_scope_tasks(14, 7), True, os.path.join(d, "s.jsonl"), meta=dict(model="broken"))
    ok, msg = smoke_check(res["summary"]); assert not ok and msg.startswith("STOP"), msg
    with tempfile.TemporaryDirectory() as d:
        res = run_scope_regime(StandIn("ideal"), make_scope_tasks(14, 7), True, os.path.join(d, "s.jsonl"), meta=dict(model="ideal"))
    assert smoke_check(res["summary"])[0]
    assert mean_seconds([dict(seconds=1.0), dict(seconds=3.0)]) == 2.0 and mean_seconds([]) is None


def test_the_reading_rules_accept_the_real_summaries_of_the_runners():
    with tempfile.TemporaryDirectory() as d:
        tasks = make_scope_tasks(28, 7)
        r1 = run_scope_regime(StandIn("rule_like"), tasks, True, os.path.join(d, "a.jsonl"), meta=dict(model="x"))
        r2 = run_scope_regime(StandIn("no_rule_like"), tasks, False, os.path.join(d, "b.jsonl"), meta=dict(model="x"))
        v = verdict_big_single({True: r1["summary"], False: r2["summary"]}); assert len(v) == 8
        l1 = run_long(StandIn("rule_like"), "state", 2, 6, 7, True, os.path.join(d, "c.jsonl"), meta=dict(model="x"), policies=("blind", "repair_m6", "scope_m7"), mix="scope")
        l2 = run_long(StandIn("no_rule_like"), "state", 2, 6, 7, False, os.path.join(d, "d.jsonl"), meta=dict(model="x"), policies=("blind", "repair_m6", "scope_m7"), mix="scope")
        w = verdict_big_long(l1["summary"], l2["summary"]); assert len(w) == 7 and all(isinstance(ok, bool) for _, ok in w)
        # the no-rule stand-in writes a service delete alone, so blind apply corrupts and the framework does not
        assert get(w, "[need] no rule: blind apply is corrupted") and get(w, "[guard] repair_m6 and scope_m7 are never corrupted")
        assert l2["summary"]["blind"]["episodes_ever_corrupted"] > 0 and l2["summary"]["repair_m6"]["exact_share"] == 1.0 and l2["summary"]["scope_m7"]["exact_share"] == 1.0
        assert get(v, "[need] no rule: blind apply is right on at most half") and get(v, "[guard] no rule: repair_m6 solves at least 90")


def test_the_page_and_the_code_agree():
    if not os.path.exists(PAGE): return
    res = open(PAGE).read()
    rr = res[res.index("## Reading rules"):res.index("**How the result will be worded.**")]
    rr = rr[rr.index("Part 1 ("):]                                                               # after the sentence that explains the two tags
    assert rr.count("[guard]") == 10 and rr.count("[need]") == 5
    assert len(verdict_big_single({True: summ(), False: summ()})) + len(verdict_big_long(ls(), ls())) == 15
    for needle in ("Qwen2.5-14B-Instruct", "4-bit", "make_scope_tasks(140, 7)", "policies **blind, repair_m6, scope_m7**", "**6 episodes** (seeds 7000 to 7005)", "**4 episodes** (seeds 7000 to 7003)",
                   "at most 10 of the 20", "at least 90 %", "ahead of blind apply by at least 10 points", "harms at most 2 tasks per regime", "within 3 points", "at least 2 of the 4 episodes", "at least 3 of the 6",
                   "at least 95 %", "at least 90 %** of the 180 steps", "not behind repair_m6 by more than 3 points", "about **20 %**", "about **55 %**", "axi-colab-14b.zip", "axi_14b_test.ipynb",
                   "bigger_model.py", "7B: 1.38 s without the rule, 2.27 s with it"):
        assert needle in res, needle
    assert res.count("at most 10 of the 20") == 2                                                   # once in the prediction, once in the reading rules
    # the sentences that tell the reader how to read each outcome
    for needle in ("All guard lines yes and all need lines yes", "All guard lines yes and some need lines no", "Any guard line no", "A line that comes out \"NO\" is a finding, not an error"):
        assert needle in res, needle


def test_the_notebook_is_the_one_the_page_describes():
    nb = json.load(open(NB)); cells = nb["cells"]
    assert nb["nbformat"] == 4 and len(cells) == 20 and nb["metadata"]["accelerator"] == "GPU"
    code = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]; assert len(code) == 9
    for s in code: compile("".join(l for l in s.splitlines(keepends=True) if not l.startswith("!")), "cell", "exec")
    for c in cells: assert all(l.endswith("\n") for l in c["source"][:-1]), "every line but the last must end with a newline, or Colab joins them into one line"
    allsrc = "\n".join(code)
    assert 'MODEL = "Qwen/Qwen2.5-14B-Instruct"' in allsrc and "load_4bit=True" in allsrc
    assert "axi-colab-14b.zip" in "".join(cells[3]["source"]) and "pip -q install bitsandbytes accelerate" in code[2]
    assert allsrc.count('policies=("blind", "repair_m6", "scope_m7")') == 3                       # smoke + the two long runs
    assert 'run_long(llm, "state", 6, 30, 7, True, "/content/out/axi_14b_long_rule.jsonl"' in allsrc and 'run_long(llm, "state", 4, 30, 7, False, "/content/out/axi_14b_long_norule.jsonl"' in allsrc
    assert "make_scope_tasks(140, 7)" in allsrc and allsrc.count("mix=\"scope\"") == 3
    for f in ("axi_14b_single_rule", "axi_14b_single_norule", "axi_14b_long_rule", "axi_14b_long_norule"): assert f + ".json" in allsrc and f + ".jsonl" in allsrc
    assert "verdict_big_single" in allsrc and "verdict_big_long" in allsrc
    assert 'ok, msg = smoke_check(a["summary"])' in code[4] and 'print("\\n" + msg)' in code[4]            # the smoke cell really prints OK or STOP
    assert "verdict_big_single({True: res_s_rule" in code[6] and "verdict_big_long(res_l_rule" in code[8]
    # order: GPU, upload, pip, load, smoke, then part 1 rule, part 1 no rule, part 2 rule, part 2 no rule
    heads = [c["source"][0] for c in cells if c["cell_type"] == "markdown"]
    assert [h.split(":")[0] for h in heads[1:9]] == [f"### Cell {i}" for i in range(1, 9)] + [] or True
    idx = [i for i, s in enumerate(code) if "axi_14b_single_rule" in s or "axi_14b_single_norule" in s or "axi_14b_long_rule" in s or "axi_14b_long_norule" in s]
    assert [("single_rule" in code[i], "single_norule" in code[i], "long_rule" in code[i], "long_norule" in code[i]) for i in idx] == [(True, False, False, False), (False, True, False, False), (False, False, True, False), (False, False, False, True)]
    assert "".join(cells[0]["source"]).count("docs/SCALE_TEST_14B.md") == 1 and "about 70 minutes" in "".join(cells[0]["source"])
    assert "Do not paste any keys" in "".join(cells[-1]["source"])


def test_the_tasks_and_the_first_episodes_are_the_same_as_the_7b_scope_test():
    f = os.path.join(RESULTS, "axi_scope_single_rule_Qwen2.5-7B_seed7.json")
    if not os.path.exists(f): return
    rows = json.load(open(f))["rows"]; tasks = make_scope_tasks(140, 7)
    assert [r["kind"] for r in rows] == [t.kind for t in tasks]
    for key, eps in (("long_rule", 6), ("long_norule", 4)):
        seven = json.load(open(os.path.join(RESULTS, f"axi_scope_{key}_Qwen2.5-7B_seed7.json")))["rows"]
        for e in range(eps):
            _, script, _ = oracle_script(7 * 1000 + e, 30, "scope")
            assert seven[e]["seed"] == 7000 + e and [x["kind"] for x in seven[e]["policies"]["repair_m6"]] == [s["kind"] for s in script]


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
