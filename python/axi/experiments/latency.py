"""Latency of the correctness modes, from a finished agent-loop result (the `rows` of an agent_loop / colab_scale_run JSON). No model is run here.

Two kinds of time, kept apart on purpose:
  model seconds   what the GPU spent generating, as recorded in the rows (`seconds` of the M0 / M1 call and of each M3 retry call; torch.cuda.synchronize before and after).
  runtime micros  what the CPU spends in the runtime on the SAME proposals: gate `check`, `diagnose`, the M5 `propose_repair`, the M6 `ground_and_complete`, and the inverse `invert`.
                  Measured here, warm, as the mean of REPS calls per proposal; the machine it ran on is whatever runs this file, so quote it with `cpu_description()`.
Per-task cost of each mode (model seconds, in the pipeline order of agent_loop.run_task):
  M0 the unconstrained call; M1, M2, M5, M6 the one constrained call (the gate and the repairs add no model call); M3 the constrained call plus, when the gate rejected a
  parseable proposal, one more constrained call whose recorded seconds are the retry's.
Also reported: the cost of grammar-constrained decoding (M1 seconds against M0 seconds on the same prompt; the texts are compared, so a ratio is only quoted for identical texts).
"""
import platform, statistics, time
from typing import Dict, List

from axi.engine.gate import check
from axi.engine.diagnostics import diagnose, propose_repair, ground_and_complete
from axi.engine.inverse import invert
from axi.experiments.agent_loop import make_tasks, strict_parse

RETRY = ("prose_plain", "prose_detailed", "structured", "structured_imperative")
ROW_KEY = {"M0": "M0_unconstrained", "M1": "M1_constrained", "M2": "M2_gate", "M5": "M5_auto_repair", "M6": "M6_ground_complete",
           **{"M3_" + v: f"M3_retry_{v}" for v in RETRY}}
REPS = 200


def cpu_description() -> str:
    model = platform.processor() or platform.machine()
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"): model = line.split(":", 1)[1].strip(); break
    except OSError: pass
    return f"{model} (python {platform.python_version()})"


def _mean_us(fn, reps=REPS) -> float:
    fn()
    t = time.perf_counter_ns()
    for _ in range(reps): fn()
    return (time.perf_counter_ns() - t) / reps / 1000.0


def model_seconds_per_task(rows) -> Dict[str, List[float]]:
    """Model seconds spent on each task, per mode, using the recorded call times."""
    out = {m: [] for m in ("M0", "M1", "M2", "M3_" + RETRY[0], "M3_" + RETRY[1], "M3_" + RETRY[2], "M3_" + RETRY[3], "M5", "M6")}
    for r in rows:
        m0, m1 = r["M0_unconstrained"]["seconds"], r["M1_constrained"]["seconds"]
        out["M0"].append(m0)
        for k in ("M1", "M2", "M5", "M6"): out[k].append(m1)
        for v in RETRY:
            x = r[f"M3_retry_{v}"]
            out["M3_" + v].append(m1 + (x["seconds"] if x.get("retried") else 0.0))
    return out


def runtime_micros(rows, tasks) -> Dict[str, List[float]]:
    """CPU microseconds per proposal for each runtime step; repairs are timed only where they run (the gate rejected a parseable proposal)."""
    out = {"check": [], "diagnose": [], "propose_repair_M5": [], "ground_and_complete_M6": [], "invert": []}
    for r, t in zip(rows, tasks):
        d = strict_parse(r["M1_constrained"]["text"])
        if d is None: continue
        ok = check(t.graph, d).ok
        out["check"].append(_mean_us(lambda: check(t.graph, d)))
        out["diagnose"].append(_mean_us(lambda: diagnose(t.graph, d)))
        if ok: out["invert"].append(_mean_us(lambda: invert(t.graph, d)))
        else:
            out["propose_repair_M5"].append(_mean_us(lambda: propose_repair(t.graph, d)))
            out["ground_and_complete_M6"].append(_mean_us(lambda: ground_and_complete(t.graph, d)))
    return out


def _p(xs, q): xs = sorted(xs); return xs[min(len(xs) - 1, int(q * len(xs)))]


def latency_report(rows, seed=3) -> dict:
    tasks = make_tasks(len(rows), seed)
    assert [r["kind"] for r in rows] == [t.kind for t in tasks], "rows are not the tasks of make_tasks(n, seed)"
    secs = model_seconds_per_task(rows)
    dd = [i for i, r in enumerate(rows) if r["kind"] == "delete_dep"]
    def wins(m): return sum(bool(r[ROW_KEY[m]]["success"]) for r in rows)
    mode = {m: dict(mean_all=statistics.mean(v), mean_delete_dep=statistics.mean(v[i] for i in dd), total=sum(v), successes=wins(m),
                    seconds_per_success=(sum(v) / wins(m) if wins(m) else None)) for m, v in secs.items()}
    retry = {}
    for v in RETRY:
        rs = [r[f"M3_retry_{v}"]["seconds"] for r in rows if r[f"M3_retry_{v}"].get("retried")]
        retry[v] = dict(n=len(rs), mean_s=statistics.mean(rs) if rs else 0.0, max_s=max(rs) if rs else 0.0)
    same = [r for r in rows if r["M0_unconstrained"]["text"] == r["M1_constrained"]["text"]]
    ratio = [r["M1_constrained"]["seconds"] / r["M0_unconstrained"]["seconds"] for r in same]
    cons = dict(n=len(rows), identical_texts=len(same), mean_ratio_M1_over_M0=statistics.mean(ratio), median_ratio=statistics.median(ratio),
                m0_total_s=sum(r["M0_unconstrained"]["seconds"] for r in same), m1_total_s=sum(r["M1_constrained"]["seconds"] for r in same))
    micro = runtime_micros(rows, tasks)
    runtime = {k: dict(n=len(v), median_us=statistics.median(v), p95_us=_p(v, 0.95), max_us=max(v)) for k, v in micro.items() if v}
    tokens = sum(r["M1_constrained"]["tokens"] for r in rows); seconds = sum(r["M1_constrained"]["seconds"] for r in rows)
    per_task_runtime_us = {k: statistics.mean(micro[k]) for k in micro if micro[k]}
    worst_task_us = max(c + d for c, d in zip(micro["check"], micro["diagnose"]))                  # check + diagnose on the same proposal
    return dict(n=len(rows), cpu=cpu_description(), model_seconds=mode, retry_call=retry, constrained_decoding=cons, runtime=runtime,
                tokens_per_second=tokens / seconds, mean_model_seconds_per_task=seconds / len(rows),
                runtime_share_of_one_call=(runtime["check"]["median_us"] + runtime["diagnose"]["median_us"]) / 1e6 / (seconds / len(rows)),
                worst_check_plus_diagnose_us=worst_task_us)


def print_latency(rep, title=""):
    print(f"\n== latency {title} (n={rep['n']} tasks; CPU: {rep['cpu']}) ==")
    print(f"generation: {rep['tokens_per_second']:.1f} tokens/s overall, {rep['mean_model_seconds_per_task']:.2f} s per constrained call on average")
    print(f"{'mode':32s} {'model s / task (all)':>22s} {'(delete_dep)':>14s} {'correct':>8s} {'model s per correct task':>26s}")
    for m, v in rep["model_seconds"].items():
        sps = "-" if v["seconds_per_success"] is None else f"{v['seconds_per_success']:.2f}"
        print(f"{m:32s} {v['mean_all']:22.2f} {v['mean_delete_dep']:14.2f} {v['successes']:8d} {sps:>26s}")
    print("second model call in the M3 retry modes (only on rejected proposals):")
    for k, v in rep["retry_call"].items(): print(f"  {k:24s} {v['n']:3d} retries, mean {v['mean_s']:.2f} s, max {v['max_s']:.2f} s")
    c = rep["constrained_decoding"]
    print(f"constrained against unconstrained decoding: {c['identical_texts']}/{c['n']} identical texts; M1/M0 time ratio mean {c['mean_ratio_M1_over_M0']:.3f}, median {c['median_ratio']:.3f} (total {c['m1_total_s']:.1f} s against {c['m0_total_s']:.1f} s)")
    print(f"{'runtime step (CPU, per proposal)':36s} {'n':>4s} {'median us':>10s} {'p95 us':>9s} {'max us':>9s}")
    for k, v in rep["runtime"].items(): print(f"{k:36s} {v['n']:4d} {v['median_us']:10.1f} {v['p95_us']:9.1f} {v['max_us']:9.1f}")
    print(f"gate check + diagnose is {rep['runtime_share_of_one_call']*100:.4f} % of one average model call")
