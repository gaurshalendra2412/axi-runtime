"""Implements concept-map item 131 (escalate when nothing is committed: the next model is asked only on a step the host itself sees fail;
python/axi/experiments/escalate.py, a measurement NOT KEPT in the engine; docs/ESCALATE_ON_REFUSAL.md), test 157. Prompted by Rajnish Choubey's line quoted in "How Perplexity Made
Google a backend team of brick layers" ("The only thing that scales is AI-to-AI communication + human-centric routing").
It checks, with no model and no new call:
  1. `escalate` with scripted models: the first reply that commits a change ends it, the later models are not asked, a reply that
     commits nothing passes the step on, none committing leaves the state as it was;
  2. replayed from the saved replies of the seed-7 single-step scope runs (the same 140 prompts sent to Qwen2.5-7B, Qwen2.5-14B and
     gemini-3.5-flash-lite, with the rule stated and without it): each model alone, and every order of two, through the framework's own
     repair (`repair_m6`) and scope (`scope_m7`), unchanged;
  3. the trigger compared with escalating on the gate's raw refusal (before the repair): more calls, and steps lost;
  4. what no router of this kind sees: the Qwen replies always commit something, so their wrong ones are never passed on;
  5. the saved seed-7 long runs, as far as they can say: the step of each episode where flash-lite first failed, replayed with the next
     model's reply to the same prompt; after that step the saved runs cannot say;
  6. why it is not kept: all 33 replies of flash-lite's that nothing could commit are one slip (a node delete written with a value), and
     a reader that drops that value gives flash-lite alone the router's 138 of 140 with no second call;
  7. the page and the front pages (concept map item 131, VERIFIED row, the counter).
What it does NOT show: a live run of the router; the seconds were measured on different machines."""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
from axi.experiments.escalate import escalate
from axi.engine.gate import check
from axi.experiments.agent_loop import copy_graph, same_graph, strict_parse
from axi.experiments.cascade import next_state
from axi.experiments.long_horizon import oracle_script
from axi.experiments.scope_tasks import make_scope_tasks

HERE = os.path.dirname(__file__)
DOCS = os.path.join(HERE, "..", "..", "docs")
RES = os.path.join(DOCS, "results")
PAGE = os.path.join(DOCS, "ESCALATE_ON_REFUSAL.md")
NEXT = "Next free: row 183, item 137, test 162, question 122"
MODELS = ("7B", "14B", "fl")
SINGLE = {"7B": "axi_scope_single_{}_Qwen2.5-7B_seed7.json", "14B": "axi_14b_single_{}_Qwen2.5-14B_seed7.json", "fl": "axi_gemini_single_{}_gemini-3.5-flash-lite_seed7.json"}
LONG = {"7B": "axi_scope_long_{}_Qwen2.5-7B_seed7.json", "14B": "axi_14b_long_{}_Qwen2.5-14B_seed7.json", "fl": "axi_gemini_long_{}_gemini-3.5-flash-lite_seed7.json"}


def test_escalate_with_scripted_models():
    asked = []
    def model(name, reply): return lambda: (asked.append(name), reply)[1]
    commit = lambda reply: reply                                  # the state the host would commit is the reply itself here
    same = lambda x, y: x == y
    assert escalate([model("a", "S1"), model("b", "S2")], commit, "S0", same)[0] == "S1" and asked == ["a"]
    asked.clear(); after, why = escalate([model("a", "S0"), model("b", "S2"), model("c", "S3")], commit, "S0", same)
    assert after == "S2" and why["asked"] == 2 and why["by"] == 1 and asked == ["a", "b"] and why["replies"] == ["S0", "S2"]
    asked.clear(); after, why = escalate([model("a", "S0"), model("b", "S0")], commit, "S0", same)
    assert after == "S0" and why["by"] is None and why["asked"] == 2
    assert escalate([], commit, "S0", same) == ("S0", dict(asked=0, by=None, replies=[]))


def single(model, rule):
    return json.load(open(os.path.join(RES, SINGLE[model].format("rule" if rule else "norule"))))["rows"]


def replay_single(order, rule, policy, trigger="nothing committed"):
    """Exact tasks, models asked after the first, seconds and output tokens of every call, for the models in `order`."""
    tasks = make_scope_tasks(140, 7); rows = {m: single(m, rule) for m in order}
    for m in order: assert [r["kind"] for r in rows[m]] == [t.kind for t in tasks]
    out = dict(exact=0, passed_on=0, seconds=0.0, tokens=0)
    for i, t in enumerate(tasks):
        commit = lambda text: next_state(policy, t.graph, strict_parse(text), t.instruction)
        if trigger == "nothing committed":
            askers = [lambda m=m: rows[m][i]["text"] for m in order]
            after, why = escalate(askers, commit, t.graph, same_graph)
            if why["by"] is None: after = t.graph
            used = order[:why["asked"]]
        else:                                                     # the gate's raw refusal of the first reply, before the repair
            d = strict_parse(rows[order[0]][i]["text"]); refused = d is None or not check(t.graph, d).ok
            used = order[:2] if refused and len(order) > 1 else order[:1]; after = commit(rows[used[-1]][i]["text"])
        out["exact"] += same_graph(after, t.expected); out["passed_on"] += len(used) - 1
        out["seconds"] += sum(rows[m][i]["seconds"] for m in used); out["tokens"] += sum(rows[m][i]["tokens"] for m in used)
    out["seconds"] = round(out["seconds"], 1)
    return out


def test_replayed_from_the_saved_single_step_replies():
    exact = lambda o, r, p: replay_single(o, r, p)["exact"]
    # rule stated, scope_m7 (the current best policy)
    assert [exact((m,), True, "scope_m7") for m in MODELS] == [134, 133, 108]
    assert replay_single(("fl", "7B"), True, "scope_m7") == dict(exact=138, passed_on=30, seconds=226.8, tokens=4660)
    assert replay_single(("fl", "14B"), True, "scope_m7") == dict(exact=138, passed_on=30, seconds=267.5, tokens=4165)
    assert replay_single(("7B",), True, "scope_m7") == dict(exact=134, passed_on=0, seconds=317.3, tokens=3975)
    assert replay_single(("14B",), True, "scope_m7") == dict(exact=133, passed_on=0, seconds=538.7, tokens=3510)
    assert replay_single(("fl",), True, "scope_m7") == dict(exact=108, passed_on=0, seconds=110.9, tokens=3025)
    assert replay_single(("7B", "14B"), True, "scope_m7")["passed_on"] == 0 and replay_single(("14B", "fl"), True, "scope_m7")["passed_on"] == 0
    # rule stated, repair_m6
    assert [exact((m,), True, "repair_m6") for m in MODELS] == [118, 117, 108]
    assert (exact(("fl", "7B"), True, "repair_m6"), exact(("fl", "14B"), True, "repair_m6")) == (126, 128)
    # no rule
    for p, alone in (("scope_m7", [135, 136, 135]), ("repair_m6", [135, 135, 135])):
        assert [exact((m,), False, p) for m in MODELS] == alone
        for second in ("7B", "14B"):
            o = replay_single(("fl", second), False, p); assert (o["exact"], o["passed_on"]) == (138, 3)
    assert replay_single(("fl", "7B"), False, "scope_m7")["seconds"] == 84.5 and replay_single(("fl",), False, "scope_m7")["seconds"] == 79.5
    assert all(replay_single(("fl", "7B", "14B"), r, "scope_m7")["exact"] == 138 for r in (True, False))   # a third model adds nothing here


def test_escalating_on_the_raw_refusal_costs_more_and_loses_steps():
    raw = lambda o, r, p: replay_single(o, r, p, trigger="raw refusal")
    assert raw(("7B", "14B"), True, "scope_m7") == dict(exact=135, passed_on=34, seconds=467.3, tokens=5017)
    assert raw(("7B", "fl"), True, "scope_m7")["exact"] == 120                 # 14 fewer than 7B alone: the repair would have fixed them
    assert raw(("14B", "fl"), True, "scope_m7")["exact"] == 119
    assert raw(("fl", "7B"), True, "scope_m7") == dict(exact=138, passed_on=32, seconds=230.1, tokens=4698)   # same gain, two calls more


def test_what_the_host_cannot_see_is_never_passed_on():
    tasks = make_scope_tasks(140, 7)
    for m in ("7B", "14B"):
        for rule in (True, False):
            rows = single(m, rule)
            for p in ("repair_m6", "scope_m7"):
                fins = [next_state(p, t.graph, strict_parse(r["text"]), t.instruction) for r, t in zip(rows, tasks)]
                assert not any(same_graph(f, t.graph) for f, t in zip(fins, tasks))           # the repair commits something every time
    rows = single("7B", True); fins = [next_state("scope_m7", t.graph, strict_parse(r["text"]), t.instruction) for r, t in zip(rows, tasks)]
    assert sum(not same_graph(f, t.expected) for f, t in zip(fins, tasks)) == 6               # 6 wrong, all committed, none passed on


def long_rows(model, rule):
    return json.load(open(os.path.join(RES, LONG[model].format("rule" if rule else "norule"))))["rows"]


def first_miss_replayed(first, second, rule, policy):
    """For each episode both runs share: at the step where `first` first failed, (nothing committed?, the second model's reply to the same
    prompt exact?, or None when the second model's own state had already left the oracle's)."""
    out = []
    for ef, es in zip(long_rows(first, rule), long_rows(second, rule)):
        assert ef["seed"] == es["seed"]; g0, script, oracle = oracle_script(ef["seed"], ef["steps"], "scope")
        F, S = ef["policies"][policy], es["policies"][policy]
        t = next((i for i, r in enumerate(F) if not r["exact"]), None)
        if t is None: continue
        commit = lambda text: next_state(policy, copy_graph(oracle[t]), strict_parse(text), script[t]["instruction"])
        assert same_graph(commit(F[t]["text"]), oracle[t + 1]) == F[t]["exact"]
        nothing = same_graph(commit(F[t]["text"]), oracle[t])
        avail = t == 0 or S[t - 1]["exact"]
        out.append((nothing, same_graph(commit(S[t]["text"]), oracle[t + 1]) if avail else None))
    return out


def test_the_long_runs_as_far_as_they_can_say():
    for second in ("7B", "14B"):
        assert first_miss_replayed("fl", second, True, "scope_m7") == [(True, True)] * 6
    assert first_miss_replayed("fl", "14B", True, "repair_m6") == [(True, True), (True, True), (True, True), (True, None), (True, True), (True, True)]
    assert sorted(first_miss_replayed("fl", "7B", True, "repair_m6"), key=str) == sorted([(True, False), (True, True), (True, False), (True, None), (True, True), (True, True)], key=str)
    assert first_miss_replayed("fl", "14B", False, "scope_m7") == []                      # no rule: flash-lite never failed in the 4 shared episodes
    fl = long_rows("fl", True)
    assert sum(r["exact"] for e in fl for r in e["policies"]["scope_m7"]) == 14            # alone it held 14 of 180 steps, all before its first miss


def test_why_it_is_not_kept_one_slip_that_a_reader_also_recovers():
    tasks = make_scope_tasks(140, 7); slip = re.compile(r"DEL\[(\d+),(\d+):\d+\]")
    for rule, n in ((True, 30), (False, 3)):
        rows = single("fl", rule); bad = [r["text"] for r in rows if strict_parse(r["text"]) is None]
        assert len(bad) == n and all(slip.search(t) for t in bad)
        for p in ("scope_m7", "repair_m6"):
            fixed = sum(same_graph(next_state(p, t.graph, strict_parse(slip.sub(r"DEL[\1,\2]", r["text"])), t.instruction), t.expected) for r, t in zip(rows, tasks))
            assert fixed == 138, (rule, p, fixed)                             # the router's 138 of 140, with no second call
    assert not os.path.exists(os.path.join(HERE, "..", "axi", "engine", "escalate.py"))   # not kept in the engine


def test_the_page_and_the_front_pages():
    page = open(PAGE).read(); low = page.lower()
    for s in ("escalate.py", "Rajnish Choubey", "How Perplexity Made Google a backend team of brick layers", "human-centric routing", "item 109",
              "108", "138 of 140", "134", "133", "30 of the 140", "226.8", "317.3", "538.7", "110.9", "135", "3 of the 140", "raw refusal", "120",
              "cannot see", "6 of 6", "14 of 180", "different machines", "test 157", "needs a run", "may be", "not kept", "33", "one slip",
              "no second call", "experiments/escalate.py"):
        assert s.lower() in low, s
    for bad in ("proven", "$", "dollar", "100 percent"): assert bad not in low, bad
    cm = open(os.path.join(DOCS, "RAJNISH_CONCEPT_MAP.md")).read(); vf = open(os.path.join(DOCS, "VERIFIED.md")).read()
    assert NEXT in cm and cm.count("Next free:") == 1 and "test 157 checks escalating when nothing is committed, item 131" in cm
    j = cm.index("\n131. **") + 1; item = cm[j:cm.index("\n", j)]
    assert "escalate.py" in item and "ESCALATE_ON_REFUSAL.md" in item and "test 157" in item and "Rajnish Choubey" in item and "not kept" in item
    row = next(l for l in vf.split("\n") if "ESCALATE_ON_REFUSAL.md" in l); assert "test_escalate.py" in row and "no model" in row.lower()


if __name__ == "__main__":
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]; bad = 0
    for n, f in fns:
        try: f(); print("ok  ", n)
        except Exception: bad += 1; print("FAIL", n); traceback.print_exc()
    print(f"{len(fns) - bad}/{len(fns)} passed"); sys.exit(1 if bad else 0)
