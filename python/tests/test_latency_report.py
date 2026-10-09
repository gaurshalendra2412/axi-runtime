"""Implements the latency half of docs/LATENCY_7B.md (concept-map rows 6 and 9; VERIFIED.md 'speed claims'): what the correctness modes cost in model seconds and what the runtime costs in CPU microseconds,
computed from the two saved Qwen2.5-7B result files (docs/results/, 60 tasks each). No model is run.
Checks: the per-task model seconds of every mode follow the pipeline (M2, M5 and M6 add no model call; an M3 mode adds the retry call exactly on the retried tasks); the totals and 'seconds per correct task'
are consistent with the rows; constrained decoding gave the same text as unconstrained in all 120 tasks and its time ratio is within 3 percent of 1; the runtime steps cost microseconds (bounds only, because the
speed of the CPU that runs the test varies): check and diagnose under 1 ms each, M6 under 5 ms, and together under 0.01 percent of one model call; the page quotes the model-seconds tables as computed.
What it does NOT show: timings on the Colab CPU, long contexts, a cache-reusing engine, or the Rust crates."""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
from axi.experiments.latency import latency_report, model_seconds_per_task, ROW_KEY, RETRY, cpu_description

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "..", "docs", "results")


def load():
    paths = {n: os.path.join(RES, f"axi_Qwen2.5-7B-Instruct_seed3_{n}.json") for n in ("norule", "rule")}
    if not all(os.path.exists(p) for p in paths.values()): return None
    return {n: json.load(open(p))["rows"] for n, p in paths.items()}


def test_model_seconds_follow_the_pipeline_and_the_retry_call_is_paid_only_by_retried_tasks():
    R = load()
    if R is None: return
    for name, rows in R.items():
        secs = model_seconds_per_task(rows)
        assert all(len(v) == 60 for v in secs.values())
        assert secs["M1"] == secs["M2"] == secs["M5"] == secs["M6"] == [r["M1_constrained"]["seconds"] for r in rows]       # the gate and the repairs add no model call
        assert secs["M0"] == [r["M0_unconstrained"]["seconds"] for r in rows]
        for v in RETRY:
            for r, s in zip(rows, secs["M3_" + v]):
                x = r[f"M3_retry_{v}"]
                assert abs(s - (r["M1_constrained"]["seconds"] + (x["seconds"] if x["retried"] else 0.0))) < 1e-12
                if not x["retried"]: assert s == r["M1_constrained"]["seconds"] and x["tokens"] == 0
        retried = sum(r["M3_retry_prose_detailed"]["retried"] for r in rows)
        assert retried == (30 if name == "norule" else 25)                                                                     # = the proposals the gate rejected and that parsed
        assert retried == sum(bool(r["M2_gate"]["rejected"]) for r in rows)


def test_totals_seconds_per_correct_task_and_constrained_decoding_overhead_match_the_rows():
    R = load()
    if R is None: return
    for name, rows in R.items():
        rep = latency_report(rows)
        for m, v in rep["model_seconds"].items():
            assert v["successes"] == sum(bool(r[ROW_KEY[m]]["success"]) for r in rows)
            assert abs(v["seconds_per_success"] - v["total"] / v["successes"]) < 1e-9 if v["successes"] else v["seconds_per_success"] is None
        sm = rep["model_seconds"]
        assert sm["M6"]["seconds_per_success"] <= min(sm[k]["seconds_per_success"] for k in sm if k.startswith("M3_"))        # M6 is the cheapest per correct task in both regimes
        c = rep["constrained_decoding"]
        assert abs(c["mean_ratio_M1_over_M0"] - sum(r["M1_constrained"]["seconds"] / r["M0_unconstrained"]["seconds"] for r in rows) / 60) < 1e-12      # not its inverse
        assert c["identical_texts"] == c["n"] == 60 and 0.97 < c["mean_ratio_M1_over_M0"] < 1.03 and abs(c["m1_total_s"] - c["m0_total_s"]) / c["m0_total_s"] < 0.03
    for name, rows in R.items():                                                                                        # the retry call statistics, and the identical-text guard on a doctored copy
        rep = latency_report(rows)
        for v in RETRY:
            xs = [r[f"M3_retry_{v}"]["seconds"] for r in rows if r[f"M3_retry_{v}"]["retried"]]
            assert rep["retry_call"][v]["n"] == len(xs) and abs(rep["retry_call"][v]["mean_s"] - sum(xs) / len(xs)) < 1e-12 and rep["retry_call"][v]["max_s"] == max(xs)
        doctored = json.loads(json.dumps(rows)); doctored[40]["M1_constrained"]["text"] = "DEL[0,0]"
        c = latency_report(doctored)["constrained_decoding"]
        assert c["identical_texts"] == 59 and c["n"] == 60
        others = [r for i, r in enumerate(doctored) if i != 40]
        assert abs(c["mean_ratio_M1_over_M0"] - sum(r["M1_constrained"]["seconds"] / r["M0_unconstrained"]["seconds"] for r in others) / 59) < 1e-12      # a pair with different texts is not compared
    a, b = latency_report(R["norule"]), latency_report(R["rule"])
    assert b["mean_model_seconds_per_task"] > 2 * a["mean_model_seconds_per_task"]                                             # the rule regime writes longer deltas
    assert a["model_seconds"]["M3_prose_plain"]["mean_delete_dep"] > 4 * a["model_seconds"]["M1"]["mean_delete_dep"]


def test_the_runtime_costs_microseconds_next_to_seconds_for_the_model_and_the_page_quotes_the_tables():
    R = load()
    if R is None: return
    for name, rows in R.items():
        rep = latency_report(rows)
        rt = rep["runtime"]
        assert rt["check"]["n"] == rt["diagnose"]["n"] == 60 and rt["propose_repair_M5"]["n"] == rt["ground_and_complete_M6"]["n"] == (30 if name == "norule" else 25) and rt["invert"]["n"] == 60 - rt["propose_repair_M5"]["n"]
        assert rt["check"]["max_us"] < 1000 and rt["diagnose"]["max_us"] < 1000 and rt["ground_and_complete_M6"]["max_us"] < 5000
        assert rep["runtime_share_of_one_call"] < 1e-4 and rep["worst_check_plus_diagnose_us"] < 2000
        assert rt["ground_and_complete_M6"]["median_us"] > rt["propose_repair_M5"]["median_us"] > rt["check"]["median_us"]        # more work, more microseconds
    assert cpu_description()
    page = os.path.join(HERE, "..", "..", "docs", "LATENCY_7B.md")
    if not os.path.exists(page): return
    text = open(page).read()
    for name, heading in (("norule", "**No rule in the prompt**"), ("rule", "**Rule stated in the prompt**")):
        chunk = text[text.index(heading):]; chunk = chunk[:chunk.index("\n\n", chunk.index("|---"))] if heading.startswith("**No") else chunk[:chunk.index("\n\n", chunk.index("|---"))]
        rep = latency_report(R[name]); found = {}
        for line in chunk.splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) == 5 and re.match(r"^M\d", cells[0]): found[cells[0]] = cells[1:]
        assert len(found) == 9
        labels = {"M0 unconstrained": "M0", "M1 constrained": "M1", "M2 gate only": "M2", "M3 retry, vague prose": "M3_prose_plain", "M3 retry, detailed prose": "M3_prose_detailed",
                  "M3 retry, structured (v1)": "M3_structured", "M3 retry, structured imperative": "M3_structured_imperative", "M5 deterministic repair": "M5", "M6 ground-and-complete": "M6"}
        for label, cells in found.items():
            v = rep["model_seconds"][labels[label]]
            assert cells == [f"{v['mean_all']:.2f}", f"{v['mean_delete_dep']:.2f}", f"{v['successes']} of 60", f"{v['seconds_per_success']:.2f}"], (name, label)


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
