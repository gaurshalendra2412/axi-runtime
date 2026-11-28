"""Test 47 runner: tau2-bench retail tasks (the benchmark's own frozen 50-task set), four arms, one model (docs/TAU2_PILOT.md).

Usage (inside the notebook `python/colab/axi_tau2_pilot.ipynb`, after `pip install` of tau2-bench
and with GEMINI_API_KEY set):

    python tau2_pilot_run.py run     --arms B G A S [--taskset retail_50 | --tasks 0 1 2 ...] [--concurrency 2] [--tag t47]
    python tau2_pilot_run.py summary [--taskset retail_50 | --tasks 0 1 2 ...] [--tag t47]

`run` calls the benchmark's own command line with the two registered agents; the results land where
the benchmark puts them (`<data dir>/simulations/<tag>_<arm>/results.json`). `summary` reads those
files, prints the tables of the plan page, the eight readings and the paired comparisons, and writes `<tag>_summary.json`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

MODEL = "gemini/gemini-3.8-flash"
TASKS = ["0", "1", "2", "3", "4", "6", "7", "8", "10", "11"]  # the first ten of the base split: the smoke-test and pilot set
# The benchmark's own frozen, stratified 50-task retail subset (data/tau2/task_subsets/retail_50.json at the pinned commit;
# seed 42, frame = the 114 retail base tasks). Held here so the plan page can be checked against the code, and checked
# against the installed benchmark's file by `frozen_ids` before every run.
RETAIL_50 = ["6", "8", "11", "13", "14", "16", "17", "18", "19", "21", "22", "23", "25", "26", "30", "32", "36", "38", "39", "40",
             "42", "47", "49", "50", "52", "53", "56", "58", "59", "63", "64", "65", "66", "68", "70", "71", "74", "78", "79", "80",
             "83", "85", "86", "89", "97", "102", "104", "105", "107", "112"]
TASKSETS = {"retail_50": RETAIL_50, "first10": TASKS}
# 16 of the 50 tasks carry written ("natural-language") assertions, which the benchmark checks with an LLM judge (default
# model: an OpenAI one, for which this run has no key). The runner points that judge at the run's own model (`install_judge`);
# the other 34 tasks are scored by the benchmark's database check alone, with no judge.
JUDGED = ["16", "19", "21", "36", "38", "39", "40", "47", "59", "63", "68", "70", "89", "104", "105", "107"]
JUDGE_ARGS = {"temperature": 0.0}
GATED_FALLBACK_SHARE = 0.15  # reading 7: at most this share of the gated runs may end in the host's fallback message (3 of 20 on ten tasks)
AGENT_OF = {"B": "llm_agent", "G": "gate_chat", "A": "held_state", "S": "summary_chat", "A2": "held_state_dlg"}  # S: the chat rolled up by the model (extra arm, docs/TAU2_PILOT.md)
SEED = 300
# Google's list prices for the model, dollars per million tokens (docs/PRICE_CHECK.md)
P_IN, P_CACHED, P_OUT = 0.75, 0.075, 3.75


TRANSIENT = ("RateLimitError", "ServiceUnavailableError", "InternalServerError", "APIConnectionError", "Timeout")


DAILY_QUOTA_MESSAGE = (
    "STOP: the key's daily request quota is used up (the error says 'per day'). Asking again today cannot work, "
    "and the benchmark would only retry the conversation. Use a key with billing on (Colab Secrets, name "
    "GEMINI_API_KEY, then RPM = 0 in Cell 2), or wait for the quota to reset. Nothing was scored: runs that "
    "stopped here have no result."
)


def _stop_run(message: str) -> None:
    """Default reaction to a spent daily quota: say so and end the process (the benchmark would retry for a long time)."""
    import os
    import sys

    print(message, file=sys.stderr, flush=True)
    os._exit(3)


def is_daily_quota(error: Exception) -> bool:
    text = str(error)
    return "PerDay" in text or "per day" in text.lower()


def make_throttled(real, rpm: int, clock=None, sleep=None, max_attempts: int = 12, stop=None):
    """Wrap a completion function: at most `rpm` calls in any 60 seconds, and a transient failure
    (rate limit, 503, connection) waits and asks again for that one call. A call that failed returned
    nothing and used no tokens, so the conversation, the prompts and the counts are the same as without
    the wrapper; only the timing changes. The benchmark itself would restart the whole conversation."""
    import threading
    import time

    clock = clock or time.monotonic
    sleep = sleep or time.sleep
    lock = threading.Lock()
    stamps: list = []

    def take_slot() -> None:
        while True:
            with lock:
                now = clock()
                stamps[:] = [t for t in stamps if now - t < 60.0]
                if not rpm or len(stamps) < rpm:
                    stamps.append(now)
                    return
                wait = 60.0 - (now - stamps[0]) + 0.5
            sleep(wait)

    def throttled(*args, **kwargs):
        for attempt in range(max_attempts):
            take_slot()
            try:
                return real(*args, **kwargs)
            except Exception as e:  # noqa: BLE001
                if type(e).__name__ in TRANSIENT and is_daily_quota(e):
                    (stop or _stop_run)(DAILY_QUOTA_MESSAGE)
                    raise
                if type(e).__name__ not in TRANSIENT or attempt == max_attempts - 1:
                    raise
                sleep(min(60, 10 * (attempt + 1)))

    throttled.waits = stamps
    return throttled


def install_throttle(rpm: int) -> None:
    import tau2.utils.llm_utils as LU  # noqa: WPS433

    LU.completion = make_throttled(LU.completion, rpm)


_FENCE = re.compile(r"^\s*```[A-Za-z]*\s*(.*?)\s*```\s*$", re.S)


def strip_fence(text):
    """The benchmark's judge reads the model's reply with json.loads; take off a ```json fence if the model added one."""
    m = _FENCE.match(text or "")
    return m.group(1) if m else text


def patch_judge(E, model: str, args: dict) -> None:
    """Point the written-assertion judge module `E` at `model` and make its reply parse (no change to its prompt or rule)."""
    E.DEFAULT_LLM_NL_ASSERTIONS = model
    E.DEFAULT_LLM_NL_ASSERTIONS_ARGS = dict(args)
    real = E.generate

    def generate(*a, **kw):
        m = real(*a, **kw)
        cleaned = strip_fence(getattr(m, "content", None))
        if cleaned != getattr(m, "content", None):
            try:
                m.content = cleaned
            except Exception:  # a message object that refuses assignment
                m = m.model_copy(update={"content": cleaned})
        return m

    E.generate = generate


def install_judge(model: str = MODEL) -> None:
    import tau2.evaluator.evaluator_nl_assertions as E  # noqa: WPS433

    patch_judge(E, model, JUDGE_ARGS)


def run_dir(tag: str, arm: str) -> str:
    from tau2.utils.utils import DATA_DIR  # noqa: WPS433

    return os.path.join(str(DATA_DIR), "simulations", f"{tag}_{arm}")


def frozen_ids(name: str = "retail_50") -> list:
    """The task ids of the benchmark's own frozen subset `name`, read from the installed benchmark's data directory."""
    from tau2.utils.utils import DATA_DIR  # noqa: WPS433

    with open(os.path.join(str(DATA_DIR), "tau2", "task_subsets", f"{name}.json")) as f:
        return [str(t) for t in json.load(f)["task_ids"]]


def choose_tasks(taskset, tasks, check=True) -> list:
    """`--tasks` wins when given; otherwise the named set. A frozen set is compared with the benchmark's file, and a
    difference stops the run (the plan page names these ids)."""
    if tasks:
        return [str(t) for t in tasks]
    ids = TASKSETS[taskset]
    if check and taskset == "retail_50":
        on_file = frozen_ids("retail_50")
        if on_file != ids:
            raise SystemExit(f"retail_50 in the installed benchmark differs from the ids on the plan page: {on_file}")
    return list(ids)


def build_argv(arm: str, tasks, tag: str, concurrency: int, temperature: float, model: str = MODEL) -> list:
    """The benchmark's own command line for one arm."""
    args = json.dumps({"temperature": temperature})
    return ["tau2", "run", "--domain", "retail", "--agent", AGENT_OF[arm], "--user", "user_simulator",
            "--agent-llm", model, "--user-llm", model, "--agent-llm-args", args, "--user-llm-args", args,
            "--task-ids", *tasks, "--num-trials", "1", "--seed", str(SEED), "--max-steps", "200",
            "--max-concurrency", str(concurrency), "--save-to", f"{tag}_{arm}", "--log-level", "ERROR",
            "--auto-resume"]  # a stopped run (quota, closed tab) continues from its saved file instead of asking


def run_arm(arm: str, tasks, tag: str, concurrency: int, temperature: float, model: str = MODEL, rpm: int = 0) -> None:
    from axi.experiments import tau2_retail_agents

    tau2_retail_agents.register()
    install_throttle(rpm)  # rpm 0: no waiting between calls, but a failed call is still asked again, and a spent daily quota stops the run
    install_judge(model)  # the judge of the 16 tasks with written assertions is the run's own model (no OpenAI key here)
    from tau2.cli import main

    argv = build_argv(arm, tasks, tag, concurrency, temperature, model)
    print("RUN", arm, " ".join(argv[2:]))
    sys.argv = argv
    try:
        main()
    except SystemExit as e:  # the benchmark's command line may exit when it is done
        if e.code not in (None, 0):
            raise


# ---------------------------------------------------------------------------
def _usage_of(m: dict):
    u = m.get("usage") or {}
    return u.get("prompt_tokens", 0) or 0, u.get("completion_tokens", 0) or 0


def _cached_of(m: dict) -> int:
    ax = (m.get("raw_data") or {}).get("axi")
    if ax is not None:
        return ax.get("cached_tokens", 0) or 0
    try:
        return int(((m.get("raw_data") or {}).get("usage", {}).get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    except Exception:
        return 0


def dollars(prompt: int, completion: int, cached: int = 0) -> float:
    """List-price dollars; cached prompt tokens at the cached rate when `cached` is given."""
    return ((prompt - cached) * P_IN + cached * P_CACHED + completion * P_OUT) / 1e6


def summarize_run(path: str) -> list:
    data = json.load(open(path, encoding="utf-8"))
    rows = []
    for s in data["simulations"]:
        msgs = s.get("messages") or []
        agent = [m for m in msgs if m.get("role") == "assistant"]
        user = [m for m in msgs if m.get("role") == "user"]
        pt = sum(_usage_of(m)[0] for m in agent)
        ct = sum(_usage_of(m)[1] for m in agent)
        cached = sum(_cached_of(m) for m in agent)
        ax = [(m.get("raw_data") or {}).get("axi") or {} for m in agent]
        rows.append({
            "task": s["task_id"],
            "reward": (s.get("reward_info") or {}).get("reward"),
            "termination": s.get("termination_reason"),
            "agent_calls": len(agent),
            "llm_calls": sum(a.get("llm_calls", 1) for a in ax) if ax and any(ax) else len(agent),
            "prompt": pt, "completion": ct, "cached": cached,
            "user_prompt": sum(_usage_of(m)[0] for m in user), "user_completion": sum(_usage_of(m)[1] for m in user),
            "refusals": [r for a in ax for r in a.get("refusals", [])],
            "fallbacks": sum(a.get("fallbacks", 0) for a in ax),
            "dropped_calls": sum(a.get("dropped_calls", 0) for a in ax),
            "dropped_text": sum(a.get("dropped_text", 0) for a in ax),
            "empty_retries": sum(a.get("empty_retries", 0) for a in ax),
            "summary_calls": sum(a.get("summary_calls", 0) for a in ax),
            "summary_applied": sum(1 for a in ax if a.get("summary_applied")),
            "agent_cost_reported": s.get("agent_cost"),
        })
    rows.sort(key=lambda r: int(r["task"]) if str(r["task"]).isdigit() else 0)
    return rows


def totals(rows: list) -> dict:
    calls = sum(r["agent_calls"] for r in rows)
    pt, ct, cached = (sum(r[k] for r in rows) for k in ("prompt", "completion", "cached"))
    passed = sum(1 for r in rows if (r["reward"] or 0) >= 1.0)
    return {
        "tasks": len(rows), "passed": passed, "agent_calls": calls, "prompt": pt, "completion": ct, "cached": cached,
        "mean_prompt_per_call": (pt / calls) if calls else 0.0,
        "usd_list": dollars(pt, ct), "usd_cached_rate": dollars(pt, ct, cached),
        "usd_per_pass_list": (dollars(pt, ct) / passed) if passed else None,
        "usd_per_pass_cached_rate": (dollars(pt, ct, cached) / passed) if passed else None,
        "refusals": sum(len(r["refusals"]) for r in rows),
        "runs_with_fallback": sum(1 for r in rows if r["fallbacks"]),
        "summary_calls": sum(r.get("summary_calls", 0) for r in rows),
        "user_prompt": sum(r["user_prompt"] for r in rows), "user_completion": sum(r["user_completion"] for r in rows),
    }


def sign_test_p(x: int, y: int) -> float:
    """Exact two-sided sign test: x and y are the tasks that exactly one of two arms passed; no ties to adjust."""
    from math import comb

    n = x + y
    if n == 0:
        return 1.0
    k = min(x, y)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired(rows_a: list, rows_b: list, skip=()) -> dict:
    """Pass/fail of two arms task by task (only tasks on file in both, minus `skip`): who passed what alone, and the sign-test p."""
    ra = {r["task"]: (r["reward"] or 0) >= 1.0 for r in rows_a if r["task"] not in skip}
    rb = {r["task"]: (r["reward"] or 0) >= 1.0 for r in rows_b if r["task"] not in skip}
    common = [t for t in ra if t in rb]
    a_only = sum(1 for t in common if ra[t] and not rb[t])
    b_only = sum(1 for t in common if rb[t] and not ra[t])
    both = sum(1 for t in common if ra[t] and rb[t])
    return {"tasks": len(common), "a_only": a_only, "b_only": b_only, "both": both, "neither": len(common) - a_only - b_only - both,
            "p": sign_test_p(a_only, b_only)}


PAIRS = [("A", "B"), ("G", "B"), ("S", "B"), ("A", "S"), ("A2", "B"), ("A2", "A")]  # S and A2 are extra arms: a pair is reported when both arms were run


def readings(rows_by_arm: dict, tot: dict, tasks=None) -> list:
    """The eight lines of the plan page, in the same order: (label, yes/no/None, detail)."""
    out = []
    tasks = tasks or TASKS
    every = [r for rows in rows_by_arm.values() for r in rows]
    ok = len(every) == 3 * len(tasks) and all(r["reward"] is not None and r["termination"] != "infrastructure_error" for r in every)
    out.append((f"1 [guard] {3 * len(tasks)} runs on file, none an infrastructure error, every reward present", ok, f"{len(every)} runs"))
    B, G, A = tot["B"], tot["G"], tot["A"]
    out.append(("2 mean prompt tokens per agent call lower in A than in B", A["mean_prompt_per_call"] < B["mean_prompt_per_call"],
                f"A {A['mean_prompt_per_call']:.0f} vs B {B['mean_prompt_per_call']:.0f}"))
    ratio = (A["prompt"] / B["prompt"]) if B["prompt"] else None
    out.append(("3 total agent prompt tokens of A at most 0.7 of B's", ratio is not None and ratio <= 0.7,
                f"A {A['prompt']:,} / B {B['prompt']:,} = {ratio:.3f}" if ratio is not None else "no B tokens"))
    out.append(("4 A passes at least as many tasks as B", A["passed"] >= B["passed"], f"A {A['passed']} vs B {B['passed']} of {A['tasks']}"))
    out.append(("5 G passes at least as many tasks as B", G["passed"] >= B["passed"], f"G {G['passed']} vs B {B['passed']} of {G['tasks']}"))
    out.append(("6 the gate refuses at least one call in G or A", (G["refusals"] + A["refusals"]) >= 1,
                f"G {G['refusals']}, A {A['refusals']} refusals"))
    fb = G["runs_with_fallback"] + A["runs_with_fallback"]
    gated = 2 * len(tasks)  # arms G and A
    cap = int(GATED_FALLBACK_SHARE * gated + 1e-9)
    out.append((f"7 the host's fallback message replaces the model's in at most {cap} of the {gated} gated runs", fb <= cap, f"{fb} of {gated} runs"))
    if B["passed"] and A["passed"]:
        out.append(("8 cost per passed task of A at most that of B", A["usd_per_pass_list"] <= B["usd_per_pass_list"],
                    f"A ${A['usd_per_pass_list']:.4f} vs B ${B['usd_per_pass_list']:.4f} per pass (list price, agent side)"))
    else:
        out.append(("8 cost per passed task of A at most that of B", None, "a pass count of 0 on one side: the line does not apply"))
    return out


def readings_a2(rows_a2: list, tot: dict, tasks=None) -> list:
    """The six lines written for arm A2 on the plan page before it was run: (label, yes/no, detail). Needs the totals of B, A and A2."""
    tasks = tasks or TASKS
    A2, A, B = tot["A2"], tot["A"], tot["B"]
    out = []
    ok = len(rows_a2) == len(tasks) and all(r["reward"] is not None and r["termination"] != "infrastructure_error" for r in rows_a2)
    out.append((f"A2-1 [guard] {len(tasks)} runs of arm A2 on file, none an infrastructure error, every reward present", ok, f"{len(rows_a2)} runs"))
    share = (A2["cached"] / A2["prompt"]) if A2["prompt"] else 0.0
    out.append(("A2-2 at least 40 % of the prompt tokens of A2 are cached tokens", share >= 0.40,
                f"A2 {100 * share:.0f} % (A {100 * A['cached'] / A['prompt']:.0f} %, B {100 * B['cached'] / B['prompt']:.0f} %)"))
    out.append(("A2-3 A2 passes at least 42 of the 50 tasks", A2["passed"] >= 42, f"A2 {A2['passed']} (A {A['passed']}, B {B['passed']}) of {A2['tasks']}"))
    pa2, pa, pb = (T["usd_per_pass_cached_rate"] for T in (A2, A, B))
    if A2["passed"]:
        out.append(("A2-4 cost per passed task of A2 at the cached price is at most that of A", pa2 <= pa, f"A2 ${pa2:.4f} vs A ${pa:.4f}"))
        out.append(("A2-5 cost per passed task of A2 at the cached price is at most that of B", pa2 <= pb, f"A2 ${pa2:.4f} vs B ${pb:.4f}"))
    else:
        out.append(("A2-4 cost per passed task of A2 at the cached price is at most that of A", None, "a pass count of 0: the line does not apply"))
        out.append(("A2-5 cost per passed task of A2 at the cached price is at most that of B", None, "a pass count of 0: the line does not apply"))
    ratio = (A2["prompt"] / B["prompt"]) if B["prompt"] else None
    out.append(("A2-6 total agent prompt tokens of A2 at most 0.7 of B's", ratio is not None and ratio <= 0.7,
                f"A2 {A2['prompt']:,} / B {B['prompt']:,} = {ratio:.3f}" if ratio is not None else "no B tokens"))
    return out


def summary(tag: str, tasks=None) -> dict:
    tasks = tasks or TASKS
    rows_by_arm, tot = {}, {}
    for arm in ("B", "G", "A"):
        rows_by_arm[arm] = summarize_run(os.path.join(run_dir(tag, arm), "results.json"))
        tot[arm] = totals(rows_by_arm[arm])
    core = dict(rows_by_arm)  # the eight readings of the plan page are about B, G and A only
    s_path = os.path.join(run_dir(tag, "S"), "results.json")
    if os.path.exists(s_path):  # the extra arm S is reported next to them when it was run; it has no reading line of its own
        rows_by_arm["S"] = summarize_run(s_path)
        tot["S"] = totals(rows_by_arm["S"])
    a2_path = os.path.join(run_dir(tag, "A2"), "results.json")
    if os.path.exists(a2_path):  # the extra arm A2 (dialogue first in the prompt), designed after the result of arm A; it has its own six lines
        rows_by_arm["A2"] = summarize_run(a2_path)
        tot["A2"] = totals(rows_by_arm["A2"])
    arms = list(rows_by_arm)
    print("\nper task (reward, agent calls, prompt tokens, gate refusals, fallback)")
    for t in tasks:
        cells = []
        for arm in arms:
            r = next((x for x in rows_by_arm[arm] if x["task"] == t), None)
            cells.append("-" if r is None else f"{arm}: {r['reward']} {r['agent_calls']}c {r['prompt']:,}p {len(r['refusals'])}r {r['fallbacks']}f [{r['termination']}]")
        print(f"task {t:>2}  " + "  |  ".join(cells))
    print("\ntotals")
    for arm in arms:
        T = tot[arm]
        print(f"{arm}: passed {T['passed']}/{T['tasks']}, calls {T['agent_calls']}, prompt {T['prompt']:,} (cached {T['cached']:,}), "
              f"completion {T['completion']:,}, mean prompt/call {T['mean_prompt_per_call']:.0f}, "
              f"list ${T['usd_list']:.2f} (cached rate ${T['usd_cached_rate']:.2f}), refusals {T['refusals']}, runs with fallback {T['runs_with_fallback']}"
              + (f", summary calls {T['summary_calls']}" if arm == "S" else ""))
    print("cached share of the prompt tokens: " + ", ".join(f"{arm} {100 * tot[arm]['cached'] / tot[arm]['prompt']:.0f} %" for arm in arms if tot[arm]["prompt"]))
    pairs = {}
    print("\npaired pass/fail, task by task (X only / Y only = tasks that only that arm passed; exact two-sided sign test)")
    for x, y in PAIRS:
        if x in rows_by_arm and y in rows_by_arm:
            c = paired(rows_by_arm[x], rows_by_arm[y])
            pairs[f"{x}_vs_{y}"] = c
            print(f"{x} vs {y}: {x} only {c['a_only']}, {y} only {c['b_only']}, both {c['both']}, neither {c['neither']} of {c['tasks']}; p = {c['p']:.3f}")
    nojudge = {}
    if set(tasks) == set(RETAIL_50):  # the same pairs on the 34 tasks that no judge scores
        print(f"\non the {len(RETAIL_50) - len(JUDGED)} tasks without written assertions (database check only, no judge):")
        for arm in arms:
            kept = [r for r in rows_by_arm[arm] if r["task"] not in JUDGED]
            nojudge[arm] = {"tasks": len(kept), "passed": sum(1 for r in kept if (r["reward"] or 0) >= 1.0)}
            print(f"{arm}: passed {nojudge[arm]['passed']}/{nojudge[arm]['tasks']}")
        for x, y in PAIRS:
            if x in rows_by_arm and y in rows_by_arm:
                c = paired(rows_by_arm[x], rows_by_arm[y], skip=set(JUDGED))
                pairs[f"{x}_vs_{y}_no_judge"] = c
                print(f"{x} vs {y}: {x} only {c['a_only']}, {y} only {c['b_only']}, both {c['both']}, neither {c['neither']} of {c['tasks']}; p = {c['p']:.3f}")
    rd = readings(core, {k: tot[k] for k in core}, tasks)
    print("\nreadings (plan page order)")
    for label, val, detail in rd:
        print(("YES " if val else "NO  " if val is not None else "n/a ") + label + "  --  " + detail)
    out = {"tag": tag, "model": MODEL, "tasks": tasks, "rows": rows_by_arm, "totals": tot, "paired": pairs, "no_judge_passed": nojudge,
           "readings": [{"line": l, "yes": v, "detail": d} for l, v, d in rd]}
    if "A2" in rows_by_arm:
        rd2 = readings_a2(rows_by_arm["A2"], tot, tasks)
        print("\nreadings of arm A2 (plan page order)")
        for label, val, detail in rd2:
            print(("YES " if val else "NO  " if val is not None else "n/a ") + label + "  --  " + detail)
        out["readings_a2"] = [{"line": l, "yes": v, "detail": d} for l, v, d in rd2]
    json.dump(out, open(f"{tag}_summary.json", "w"), indent=1)
    print(f"\nwrote {tag}_summary.json")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arms", nargs="+", default=["B", "G", "A", "S"], choices=list(AGENT_OF))
    r.add_argument("--taskset", default="retail_50", choices=list(TASKSETS))
    r.add_argument("--tasks", nargs="+", default=None, help="explicit ids; they win over --taskset")
    r.add_argument("--concurrency", type=int, default=2)
    r.add_argument("--temperature", type=float, default=0.0)
    r.add_argument("--tag", default="t47")
    r.add_argument("--rpm", type=int, default=0, help="client-side limit on model calls a minute (0: no waiting); for a key with a per-minute cap")
    s = sub.add_parser("summary")
    s.add_argument("--tag", default="t47")
    s.add_argument("--taskset", default="retail_50", choices=list(TASKSETS))
    s.add_argument("--tasks", nargs="+", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        tasks = choose_tasks(a.taskset, a.tasks)
        for arm in a.arms:
            run_arm(arm, tasks, a.tag, a.concurrency, a.temperature, rpm=a.rpm)
    else:
        summary(a.tag, choose_tasks(a.taskset, a.tasks, check=False))


if __name__ == "__main__":
    main()
