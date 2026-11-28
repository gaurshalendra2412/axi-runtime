"""Offline checks of the Test 47 summary (docs/TAU2_PILOT.md) on synthetic result files. No model call, no tau2 import."""
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from axi.experiments import tau2_pilot_run as R  # noqa: E402


def sim(task, reward, prompts, completions=5, cached=0, axi=None, term="user_stop"):
    msgs = []
    for i, p in enumerate(prompts):
        raw = {"usage": {"prompt_tokens_details": {"cached_tokens": cached}}}
        if axi is not None:
            raw["axi"] = dict(axi, cached_tokens=cached)
        msgs.append({"role": "assistant", "content": "x", "usage": {"prompt_tokens": p, "completion_tokens": completions}, "raw_data": raw})
        msgs.append({"role": "user", "content": "y", "usage": {"prompt_tokens": 100, "completion_tokens": 7}})
    return {"task_id": task, "termination_reason": term, "reward_info": {"reward": reward}, "messages": msgs}


def write(d, tag, arm, sims):
    p = os.path.join(d, f"{tag}_{arm}")
    os.makedirs(p)
    json.dump({"simulations": sims}, open(os.path.join(p, "results.json"), "w"))
    return os.path.join(p, "results.json")


def test_rows_and_totals():
    with tempfile.TemporaryDirectory() as d:
        f = write(d, "t", "B", [sim("0", 1.0, [1000, 2000, 3000], cached=500), sim("1", 0.0, [4000])])
        rows = R.summarize_run(f)
        assert [r["task"] for r in rows] == ["0", "1"]
        assert rows[0]["prompt"] == 6000 and rows[0]["completion"] == 15 and rows[0]["cached"] == 1500
        assert rows[0]["user_prompt"] == 300 and rows[0]["agent_calls"] == 3
        t = R.totals(rows)
        assert t["tasks"] == 2 and t["passed"] == 1 and t["agent_calls"] == 4 and t["prompt"] == 10000
        assert abs(t["mean_prompt_per_call"] - 2500) < 1e-9
        assert abs(t["usd_list"] - (10000 * 0.75 + 20 * 3.75) / 1e6) < 1e-12
        assert t["usd_cached_rate"] < t["usd_list"]
        assert abs(t["usd_per_pass_list"] - t["usd_list"]) < 1e-12  # one pass


def test_gate_columns_are_read_from_the_agents_records():
    ax = {"llm_calls": 2, "refusals": [["CONFIRM", "cancel_pending_order"]], "fallbacks": 1, "dropped_calls": 1, "dropped_text": 0, "empty_retries": 0}
    with tempfile.TemporaryDirectory() as d:
        rows = R.summarize_run(write(d, "t", "A", [sim("0", 1.0, [800, 900], axi=ax)]))
        assert rows[0]["refusals"] == [["CONFIRM", "cancel_pending_order"]] * 2 and rows[0]["fallbacks"] == 2
        assert rows[0]["llm_calls"] == 4 and rows[0]["dropped_calls"] == 2


def test_readings_follow_the_plan_page():
    tasks = ["0", "1"]
    with tempfile.TemporaryDirectory() as d:
        ax = {"llm_calls": 1, "refusals": [], "fallbacks": 0}
        by = {"B": R.summarize_run(write(d, "t", "B", [sim("0", 1.0, [1000] * 4), sim("1", 0.0, [1000] * 4)])),
              "G": R.summarize_run(write(d, "t", "G", [sim("0", 1.0, [1000] * 4), sim("1", 1.0, [1000] * 4)])),
              "A": R.summarize_run(write(d, "t", "A", [sim("0", 1.0, [300] * 4, axi=dict(ax, refusals=[["CONFIRM", "x"]])), sim("1", 0.0, [300] * 4, axi=ax)]))}
    tot = {k: R.totals(v) for k, v in by.items()}
    rd = {r[0].split()[0]: r for r in R.readings(by, tot, tasks)}
    assert rd["1"][1] is True  # two tasks and three arms: six runs expected, six present
    assert [rd[k][1] for k in "2345678"] == [True] * 7
    assert R.readings({"B": by["B"][:1], "G": by["G"], "A": by["A"]}, tot, tasks)[0][1] is False  # a missing run fails the guard
    assert "A 300 vs B 1000" in rd["2"][2]


def test_guard_fails_on_infrastructure_error_or_missing_reward():
    tasks = ["0"]
    with tempfile.TemporaryDirectory() as d:
        mk = lambda arm, **kw: R.summarize_run(write(d, "t", arm, [sim("0", **kw)]))  # noqa: E731
        by = {"B": mk("B", reward=1.0, prompts=[10]), "G": mk("G", reward=1.0, prompts=[10]),
              "A": mk("A", reward=1.0, prompts=[10], term="infrastructure_error")}
    tot = {k: R.totals(v) for k, v in by.items()}
    assert R.readings(by, tot, tasks)[0][1] is False


def test_line_8_does_not_apply_without_a_pass_on_both_sides():
    tasks = ["0"]
    with tempfile.TemporaryDirectory() as d:
        by = {a: R.summarize_run(write(d, "t", a, [sim("0", 0.0 if a == "B" else 1.0, [10])])) for a in "BGA"}
    tot = {k: R.totals(v) for k, v in by.items()}
    assert R.readings(by, tot, tasks)[-1][1] is None


class FakeClock:
    def __init__(self):
        self.t = 0.0
        self.sleeps = []

    def now(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


class RateLimitError(Exception):
    pass


def test_throttle_limits_calls_a_minute_and_retries_one_call():
    c = FakeClock()
    seen = []

    def real(*a, **k):
        seen.append(c.t)
        return "ok"

    th = R.make_throttled(real, 4, clock=c.now, sleep=c.sleep)
    assert [th(i) for i in range(10)] == ["ok"] * 10
    for i in range(len(seen)):
        assert sum(1 for t in seen if seen[i] - 60.0 < t <= seen[i]) <= 4  # never more than 4 in any 60 s
    fails = {"n": 0}

    def flaky(*a, **k):
        fails["n"] += 1
        if fails["n"] < 3:
            raise RateLimitError("429")
        return "done"

    c2 = FakeClock()
    assert R.make_throttled(flaky, 4, clock=c2.now, sleep=c2.sleep)() == "done" and fails["n"] == 3  # the same call, asked again
    try:
        R.make_throttled(lambda: (_ for _ in ()).throw(ValueError("bad request")), 4, clock=c2.now, sleep=c2.sleep)()
        raise AssertionError("a non-transient error must pass through")
    except ValueError:
        pass


def test_a_spent_daily_quota_stops_the_run_and_is_not_asked_again():
    calls = {"n": 0}
    said = []

    def real(*a, **k):
        calls["n"] += 1
        raise RateLimitError("Quota exceeded for metric: generate_content_free_tier_requests, limit: 20, "
                             "description: Generate content requests per day limit exceeded on Free Tier. quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier")

    c = FakeClock()
    th = R.make_throttled(real, 4, clock=c.now, sleep=c.sleep, stop=said.append)
    try:
        th()
        raise AssertionError("the error must pass through")
    except RateLimitError:
        pass
    assert calls["n"] == 1 and c.sleeps == [] and said == [R.DAILY_QUOTA_MESSAGE]  # asked once, no waiting, the stop message said
    assert R.is_daily_quota(RateLimitError("limit exceeded ... PerDay ...")) and not R.is_daily_quota(RateLimitError("429 per minute"))
    assert "billing" in R.DAILY_QUOTA_MESSAGE and "no result" in R.DAILY_QUOTA_MESSAGE
    c0 = FakeClock()  # rpm 0: no waiting between calls
    th0 = R.make_throttled(lambda: "ok", 0, clock=c0.now, sleep=c0.sleep)
    assert [th0() for _ in range(50)] == ["ok"] * 50 and c0.sleeps == []


def test_arms_and_tasks_of_the_plan_page():
    assert R.TASKS == ["0", "1", "2", "3", "4", "6", "7", "8", "10", "11"]
    assert R.AGENT_OF == {"B": "llm_agent", "G": "gate_chat", "A": "held_state", "S": "summary_chat", "A2": "held_state_dlg"}
    assert [a for a in R.AGENT_OF if a != "A2"] == ["B", "G", "A", "S"]  # the default arms of a run; A2 is added with --arms A2
    assert (R.P_IN, R.P_CACHED, R.P_OUT) == (0.75, 0.075, 3.75) and R.SEED == 300 and R.MODEL == "gemini/gemini-3.8-flash"
    plan = open(os.path.join(HERE, "..", "..", "docs", "TAU2_PILOT.md"), encoding="utf-8").read()
    for s in ("0, 1, 2, 3, 4, 6, 7, 8, 10, 11", "seed 300", "gemini/gemini-3.8-flash", "0.75 input, 3.75 output",
              "768f3d8fe8528b4f32a5096f9ed008ae50f48ead"):
        assert s in plan, s


def test_the_frozen_retail_50_set_is_the_benchmarks_own_and_is_named_on_the_plan_page():
    ids = R.RETAIL_50
    assert len(ids) == 50 and len(set(ids)) == 50 and R.TASKSETS["retail_50"] is ids and R.TASKSETS["first10"] == R.TASKS
    plan = open(os.path.join(HERE, "..", "..", "docs", "TAU2_PILOT.md"), encoding="utf-8").read()
    assert ", ".join(ids) in plan  # the page lists exactly the ids the code runs
    src = os.environ.get("TAU2_SRC")
    f = os.path.join(src, "..", "data", "tau2", "task_subsets", "retail_50.json") if src else ""
    if os.path.exists(f):
        with open(f) as fh:
            d = json.load(fh)
        assert [str(t) for t in d["task_ids"]] == ids and d["seed"] == 42 and d["domain"] == "retail"


def test_choose_tasks_explicit_ids_win_and_the_default_is_the_frozen_set():
    assert R.choose_tasks("retail_50", ["3", "1"]) == ["3", "1"]
    assert R.choose_tasks("first10", None) == R.TASKS
    assert R.choose_tasks("retail_50", None, check=False) == R.RETAIL_50


def test_command_line_of_an_arm_resumes_and_carries_the_task_ids():
    argv = R.build_argv("S", R.RETAIL_50, "t47", 2, 0.0)
    assert argv[:2] == ["tau2", "run"] and "--auto-resume" in argv and argv[argv.index("--agent") + 1] == "summary_chat"
    i = argv.index("--task-ids") + 1
    assert argv[i:i + 50] == R.RETAIL_50 and argv[i + 50] == "--num-trials"
    assert argv[argv.index("--save-to") + 1] == "t47_S" and argv[argv.index("--seed") + 1] == "300"


def test_judge_is_pointed_at_the_runs_model_and_its_reply_parses():
    import types

    assert R.strip_fence('```json\n{"results": []}\n```') == '{"results": []}' and R.strip_fence('{"a": 1}') == '{"a": 1}'
    assert R.strip_fence("```\n[1]\n```") == "[1]" and R.strip_fence(None) is None

    class Msg:
        def __init__(self, content):
            self.content = content

    E = types.SimpleNamespace(DEFAULT_LLM_NL_ASSERTIONS="gpt-5.4-mini-2026-03-17", DEFAULT_LLM_NL_ASSERTIONS_ARGS={"reasoning_effort": "low"},
                              generate=lambda *a, **kw: Msg('```json\n{"results": []}\n```'))
    R.patch_judge(E, R.MODEL, R.JUDGE_ARGS)
    assert E.DEFAULT_LLM_NL_ASSERTIONS == "gemini/gemini-3.8-flash" and E.DEFAULT_LLM_NL_ASSERTIONS_ARGS == {"temperature": 0.0}
    assert json.loads(E.generate(model="m", messages=[]).content) == {"results": []}  # the fenced reply now parses


def test_the_16_judged_tasks_are_those_with_written_assertions_and_are_named_on_the_plan_page():
    assert len(R.JUDGED) == 16 and set(R.JUDGED) <= set(R.RETAIL_50) and len(set(R.JUDGED)) == 16
    plan = open(os.path.join(HERE, "..", "..", "docs", "TAU2_PILOT.md"), encoding="utf-8").read()
    assert ", ".join(R.JUDGED) in plan
    src = os.environ.get("TAU2_SRC")
    f = os.path.join(src, "..", "data", "tau2", "domains", "retail", "tasks.json") if src else ""
    if os.path.exists(f):
        with open(f) as fh:
            tasks = {str(t["id"]): t for t in json.load(fh)}
        with_nl = [i for i in R.RETAIL_50 if (tasks[i].get("evaluation_criteria") or {}).get("nl_assertions")]
        assert with_nl == R.JUDGED


def test_summary_reports_the_34_tasks_without_a_judge_separately():
    import contextlib
    import io

    with tempfile.TemporaryDirectory() as d:
        cwd = os.getcwd()
        os.chdir(d)
        try:
            for arm in "BGA":
                sims = [sim(t, 0.0 if (arm == "A" and t in ("6", "8")) else 1.0, [100]) for t in R.RETAIL_50]
                write(d, "t", arm, sims)
            R.run_dir = lambda tag, arm: os.path.join(d, f"{tag}_{arm}")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                out = R.summary("t", R.RETAIL_50)
        finally:
            os.chdir(cwd)
    assert out["no_judge_passed"]["B"] == {"tasks": 34, "passed": 34} and out["no_judge_passed"]["A"] == {"tasks": 34, "passed": 32}
    assert out["paired"]["A_vs_B_no_judge"]["b_only"] == 2 and out["paired"]["A_vs_B"]["b_only"] == 2
    assert "without written assertions" in buf.getvalue()


def test_sign_test_is_exact_and_two_sided():
    assert R.sign_test_p(0, 0) == 1.0 and R.sign_test_p(3, 3) == 1.0
    assert abs(R.sign_test_p(5, 0) - 2 / 32) < 1e-12 and abs(R.sign_test_p(0, 6) - 2 / 64) < 1e-12
    assert abs(R.sign_test_p(2, 10) - 2 * sum(__import__("math").comb(12, i) for i in range(3)) / 4096) < 1e-12
    assert R.sign_test_p(1, 0) == 1.0  # one discordant task can never be a result


def test_paired_counts_tasks_only_one_arm_passed():
    mk = lambda pairs: [{"task": t, "reward": r} for t, r in pairs]  # noqa: E731
    a = mk([("1", 1.0), ("2", 1.0), ("3", 0.0), ("4", 0.0), ("5", 1.0)])
    b = mk([("1", 1.0), ("2", 0.0), ("3", 0.0), ("4", 1.0), ("6", 1.0)])
    c = R.paired(a, b)
    assert (c["tasks"], c["a_only"], c["b_only"], c["both"], c["neither"]) == (4, 1, 1, 1, 1) and c["p"] == 1.0


def test_reading_7_scales_with_the_number_of_gated_runs():
    def labels(n):
        tasks = [str(i) for i in range(n)]
        with tempfile.TemporaryDirectory() as d:
            by = {a: R.summarize_run(write(d, "t", a, [sim(t, 1.0, [100]) for t in tasks])) for a in "BGA"}
        tot = {k: R.totals(v) for k, v in by.items()}
        return {r[0].split()[0]: r for r in R.readings(by, tot, tasks)}

    assert "at most 3 of the 20 gated runs" in labels(10)["7"][0]  # the ten-task plan: 3 of 20
    r50 = labels(50)
    assert "at most 15 of the 100 gated runs" in r50["7"][0] and "150 runs on file" in r50["1"][0]  # 50 tasks, three plan-page arms


def test_arm_s_is_reported_next_to_the_plan_page_arms_and_adds_no_reading_line():
    ax_s = {"llm_calls": 2, "summary_calls": 1, "summary_applied": True, "refusals": [], "fallbacks": 0}
    with tempfile.TemporaryDirectory() as d:
        write(d, "t", "B", [sim("0", 1.0, [1000] * 4)])
        write(d, "t", "G", [sim("0", 1.0, [1000] * 4)])
        write(d, "t", "A", [sim("0", 1.0, [300] * 4, axi={"llm_calls": 1, "refusals": []})])
        write(d, "t", "S", [sim("0", 1.0, [1500] * 3, axi=ax_s)])
        rows = R.summarize_run(os.path.join(d, "t_S", "results.json"))
        assert rows[0]["summary_calls"] == 3 and rows[0]["summary_applied"] == 3 and rows[0]["llm_calls"] == 6
        assert R.totals(rows)["summary_calls"] == 3
        old_run_dir, old_cwd = R.run_dir, os.getcwd()
        R.run_dir = lambda tag, arm: os.path.join(d, f"{tag}_{arm}")
        os.chdir(d)
        try:
            out = R.summary("t", ["0"])
        finally:
            R.run_dir = old_run_dir
            os.chdir(old_cwd)
    assert set(out["rows"]) == {"B", "G", "A", "S"} and len(out["readings"]) == 8
    assert out["readings"][0]["yes"] is True  # the guard counts the three plan-page arms only: one task, three runs


def test_arm_a2_is_reported_with_its_own_six_lines_and_the_plan_page_readings_do_not_change():
    ax = {"llm_calls": 1, "refusals": [], "fallbacks": 0}
    tasks = ["0", "1"]
    with tempfile.TemporaryDirectory() as d:
        write(d, "t", "B", [sim(t, 1.0, [1000] * 4, cached=800) for t in tasks])
        write(d, "t", "G", [sim(t, 1.0, [1000] * 4, cached=800, axi=ax) for t in tasks])
        write(d, "t", "A", [sim(t, 1.0, [900] * 4, cached=50, axi=ax) for t in tasks])
        write(d, "t", "A2", [sim(t, 1.0, [900] * 4, cached=500, axi=ax) for t in tasks])
        old_run_dir, old_cwd = R.run_dir, os.getcwd()
        R.run_dir = lambda tag, arm: os.path.join(d, f"{tag}_{arm}")
        os.chdir(d)
        try:
            out = R.summary("t", tasks)
        finally:
            R.run_dir = old_run_dir
            os.chdir(old_cwd)
    assert R.AGENT_OF["A2"] == "held_state_dlg" and "A2" in R.PAIRS[-1] + R.PAIRS[-2]
    assert set(out["rows"]) == {"B", "G", "A", "A2"} and len(out["readings"]) == 8 and len(out["readings_a2"]) == 6
    assert out["readings"][0]["yes"] is True  # the guard is about B, G and A: two tasks, three arms
    assert {"A2_vs_B", "A2_vs_A"} <= set(out["paired"])
    rd = {r["line"].split()[0]: r for r in out["readings_a2"]}
    assert rd["A2-1"]["yes"] is True and rd["A2-3"]["yes"] is False  # two tasks passed: fewer than 42
    assert rd["A2-2"]["yes"] is True  # 500 of 900 per call cached: 56 %
    assert rd["A2-4"]["yes"] is True  # more cached tokens than A: cheaper per pass at the cached price
    assert rd["A2-5"]["yes"] is False  # B has 800 of 1,000 per call cached, A2 500 of 900: B is cheaper per pass at the cached price
    assert rd["A2-6"]["yes"] is False  # 900 against 1000 per call is not 0.7


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            n += 1
            print("PASS", name)
    print(f"{n} tests PASS")
