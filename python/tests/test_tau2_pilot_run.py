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
    assert R.AGENT_OF == {"B": "llm_agent", "G": "gate_chat", "A": "held_state", "S": "summary_chat"}
    assert (R.P_IN, R.P_CACHED, R.P_OUT) == (0.75, 0.075, 3.75) and R.SEED == 300 and R.MODEL == "gemini/gemini-3.8-flash"
    plan = open(os.path.join(HERE, "..", "..", "docs", "TAU2_PILOT.md"), encoding="utf-8").read()
    for s in ("0, 1, 2, 3, 4, 6, 7, 8, 10, 11", "seed 300", "gemini/gemini-3.8-flash", "0.75 input, 3.75 output",
              "768f3d8fe8528b4f32a5096f9ed008ae50f48ead"):
        assert s in plan, s


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


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            n += 1
            print("PASS", name)
    print(f"{n} tests PASS")
