"""Test 47 runner: ten tau2-bench retail tasks, three arms, one model (docs/TAU2_PILOT.md).

Usage (inside the notebook `python/colab/axi_tau2_pilot.ipynb`, after `pip install` of tau2-bench
and with GEMINI_API_KEY set):

    python tau2_pilot_run.py run     --arms B G A [--tasks 0 1 2 ...] [--concurrency 2] [--tag t47]
    python tau2_pilot_run.py summary [--tag t47]

`run` calls the benchmark's own command line with the two registered agents; the results land where
the benchmark puts them (`<data dir>/simulations/<tag>_<arm>/results.json`). `summary` reads those
files, prints the tables of the plan page and the eight readings, and writes `<tag>_summary.json`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

MODEL = "gemini/gemini-3.8-flash"
TASKS = ["0", "1", "2", "3", "4", "6", "7", "8", "10", "11"]
AGENT_OF = {"B": "llm_agent", "G": "gate_chat", "A": "held_state"}
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


def run_dir(tag: str, arm: str) -> str:
    from tau2.utils.utils import DATA_DIR  # noqa: WPS433

    return os.path.join(str(DATA_DIR), "simulations", f"{tag}_{arm}")


def run_arm(arm: str, tasks, tag: str, concurrency: int, temperature: float, model: str = MODEL, rpm: int = 0) -> None:
    from axi.experiments import tau2_retail_agents

    tau2_retail_agents.register()
    install_throttle(rpm)  # rpm 0: no waiting between calls, but a failed call is still asked again, and a spent daily quota stops the run
    from tau2.cli import main

    args = json.dumps({"temperature": temperature})
    argv = ["tau2", "run", "--domain", "retail", "--agent", AGENT_OF[arm], "--user", "user_simulator",
            "--agent-llm", model, "--user-llm", model, "--agent-llm-args", args, "--user-llm-args", args,
            "--task-ids", *tasks, "--num-trials", "1", "--seed", str(SEED), "--max-steps", "200",
            "--max-concurrency", str(concurrency), "--save-to", f"{tag}_{arm}", "--log-level", "ERROR"]
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
        "user_prompt": sum(r["user_prompt"] for r in rows), "user_completion": sum(r["user_completion"] for r in rows),
    }


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
    out.append(("7 the host's fallback message replaces the model's in at most 3 of the 20 gated runs", fb <= 3, f"{fb} of 20 runs"))
    if B["passed"] and A["passed"]:
        out.append(("8 cost per passed task of A at most that of B", A["usd_per_pass_list"] <= B["usd_per_pass_list"],
                    f"A ${A['usd_per_pass_list']:.4f} vs B ${B['usd_per_pass_list']:.4f} per pass (list price, agent side)"))
    else:
        out.append(("8 cost per passed task of A at most that of B", None, "a pass count of 0 on one side: the line does not apply"))
    return out


def summary(tag: str, tasks=None) -> dict:
    tasks = tasks or TASKS
    rows_by_arm, tot = {}, {}
    for arm in ("B", "G", "A"):
        rows_by_arm[arm] = summarize_run(os.path.join(run_dir(tag, arm), "results.json"))
        tot[arm] = totals(rows_by_arm[arm])
    print("\nper task (reward, agent calls, prompt tokens, gate refusals, fallback)")
    for t in tasks:
        cells = []
        for arm in ("B", "G", "A"):
            r = next((x for x in rows_by_arm[arm] if x["task"] == t), None)
            cells.append("-" if r is None else f"{arm}: {r['reward']} {r['agent_calls']}c {r['prompt']:,}p {len(r['refusals'])}r {r['fallbacks']}f [{r['termination']}]")
        print(f"task {t:>2}  " + "  |  ".join(cells))
    print("\ntotals")
    for arm in ("B", "G", "A"):
        T = tot[arm]
        print(f"{arm}: passed {T['passed']}/{T['tasks']}, calls {T['agent_calls']}, prompt {T['prompt']:,} (cached {T['cached']:,}), "
              f"completion {T['completion']:,}, mean prompt/call {T['mean_prompt_per_call']:.0f}, "
              f"list ${T['usd_list']:.2f} (cached rate ${T['usd_cached_rate']:.2f}), refusals {T['refusals']}, runs with fallback {T['runs_with_fallback']}")
    rd = readings(rows_by_arm, tot, tasks)
    print("\nreadings (plan page order)")
    for label, val, detail in rd:
        print(("YES " if val else "NO  " if val is not None else "n/a ") + label + "  --  " + detail)
    out = {"tag": tag, "model": MODEL, "tasks": tasks, "rows": rows_by_arm, "totals": tot,
           "readings": [{"line": l, "yes": v, "detail": d} for l, v, d in rd]}
    json.dump(out, open(f"{tag}_summary.json", "w"), indent=1)
    print(f"\nwrote {tag}_summary.json")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arms", nargs="+", default=["B", "G", "A"], choices=list(AGENT_OF))
    r.add_argument("--tasks", nargs="+", default=TASKS)
    r.add_argument("--concurrency", type=int, default=2)
    r.add_argument("--temperature", type=float, default=0.0)
    r.add_argument("--tag", default="t47")
    r.add_argument("--rpm", type=int, default=0, help="client-side limit on model calls a minute (0: no waiting); for a key with a per-minute cap")
    s = sub.add_parser("summary")
    s.add_argument("--tag", default="t47")
    s.add_argument("--tasks", nargs="+", default=TASKS)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        for arm in a.arms:
            run_arm(arm, a.tasks, a.tag, a.concurrency, a.temperature, rpm=a.rpm)
    else:
        summary(a.tag, a.tasks)


if __name__ == "__main__":
    main()
