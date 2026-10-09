"""Helpers for the 'structure or scale' run (concept-map row 9; roadmap: first step of the scale sweep). Used by the Colab notebook `axi_7b_test.ipynb`.
The question: the same 60 graph-edit tasks (seed 3, half of them 'delete a service that still has dependents'), the same code, a bigger model. Does the bigger
model, left alone, stop corrupting the graph? The 3B numbers on this very seed are in docs/EXPERIMENT_RESULTS.md ('Seed 3'); REFERENCE_3B copies them so the
notebook can print the two side by side. Nothing here needs a GPU: the model is passed in, so the logic can be checked with a scripted stand-in."""
import json, os, time
from axi.experiments.agent_loop import make_tasks, print_summary, MODES
from axi.experiments.resumable import run_resumable

MODE_LABEL = {"M0_unconstrained": "M0 unconstrained", "M1_constrained": "M1 constrained", "M2_gate": "M2 gate only", "M3_retry_prose_plain": "M3 retry, vague prose",
              "M3_retry_prose_detailed": "M3 retry, detailed prose", "M3_retry_structured": "M3 retry, structured (v1)", "M3_retry_structured_imperative": "M3 retry, structured imperative",
              "M5_auto_repair": "M5 deterministic repair", "M6_ground_complete": "M6 ground-and-complete"}
# Qwen2.5-3B-Instruct, seed 3, max_new_tokens 400, success in percent: (all tasks, delete_dep tasks). Copied from docs/EXPERIMENT_RESULTS.md, 'Seed 3'.
REFERENCE_3B = {
    False: {"M0_unconstrained": (50.0, 0.0), "M1_constrained": (50.0, 0.0), "M2_gate": (50.0, 0.0), "M3_retry_prose_plain": (51.7, 3.3), "M3_retry_prose_detailed": (98.3, 96.7),
            "M3_retry_structured": (65.0, 30.0), "M3_retry_structured_imperative": (100.0, 100.0), "M5_auto_repair": (100.0, 100.0), "M6_ground_complete": (100.0, 100.0)},
    True: {"M0_unconstrained": (53.3, 10.0), "M1_constrained": (55.0, 13.3), "M2_gate": (41.7, 3.3), "M3_retry_prose_plain": (51.7, 16.7), "M3_retry_prose_detailed": (53.3, 26.7),
           "M3_retry_structured": (46.7, 13.3), "M3_retry_structured_imperative": (48.3, 16.7), "M5_auto_repair": (51.7, 23.3), "M6_ground_complete": (95.0, 90.0)}}
REFERENCE_3B_M1_CORRUPTED_DELETE_DEP = {False: 100.0, True: 76.7}      # blind apply of the constrained output corrupts this share of delete_dep tasks


def run_regime(llm, model_name, outdir, hint_rules, seed=3, n=60, load_4bit=True, progress=True):
    """Run one regime to a resumable file and write the finished result as JSON. Returns the result dict. Re-running after a dropped session carries on."""
    os.makedirs(outdir, exist_ok=True)
    tag = ("rule" if hint_rules else "norule")
    base = os.path.join(outdir, f"axi_{model_name.split('/')[-1]}_seed{seed}_{tag}")
    meta = dict(model=model_name, load_4bit=load_4bit, seed=seed, max_new_tokens=400)
    t0 = time.time()
    def prog(i, total):
        if progress: print(f"  task {i}/{total}   {time.time() - t0:6.0f} s", end="\r")
    res = run_resumable(llm, make_tasks(n, seed), hint_rules, base + ".jsonl", meta=meta, progress=prog)
    json.dump(res, open(base + ".json", "w"), indent=1)
    return res, base + ".json"


def comparison_table(results):
    """results: {hint_rules(bool): result dict of the bigger model}. A text table: delete_dep success, 3B against the bigger model, then the blind-apply corruption line."""
    lines = []
    for hint, res in sorted(results.items()):
        S = res["summary"]; dd, al = S["delete_dep"], S["all"]
        lines.append(f"\n== {'rule stated in the prompt' if hint else 'no rule in the prompt'}  (delete_dep n={dd['n']}, all n={al['n']}) ==")
        lines.append(f"{'mode':34s} | {'3B all':>7s} {'3B del':>7s} | {'big all':>7s} {'big del':>7s} | {'big corrupted (del)':>19s}")
        for m in MODES:
            r3 = REFERENCE_3B[hint][m]
            lines.append(f"{MODE_LABEL[m]:34s} | {r3[0]:6.1f}% {r3[1]:6.1f}% | {al[m]['success']*100:6.1f}% {dd[m]['success']*100:6.1f}% | {dd[m]['corrupted']*100:18.1f}%")
        lines.append(f"blind apply (M1) corrupts the graph on delete_dep: 3B {REFERENCE_3B_M1_CORRUPTED_DELETE_DEP[hint]:.1f}%   bigger model {dd['M1_constrained']['corrupted']*100:.1f}%")
    return "\n".join(lines)


def verdict(results):
    """Plain-words reading, using the rule written down BEFORE the run (docs/SCALE_TEST_7B.md): no-rule regime, M1 corruption on delete_dep."""
    if False not in results: return "no-rule regime not run yet"
    dd = results[False]["summary"]["delete_dep"]; c = dd["M1_constrained"]["corrupted"] * 100; g = max(dd[m]["corrupted"] for m in MODES[2:]) * 100
    if c >= 50: head = f"The bigger model, left alone, still corrupts {c:.0f}% of delete-with-dependents tasks (3B: 100%). Scale alone did not fix it."
    elif c >= 20: head = f"The bigger model corrupts {c:.0f}% (3B: 100%): scale helped a lot but did not remove the problem."
    else: head = f"The bigger model corrupts only {c:.0f}% (3B: 100%). On this task scale alone nearly fixes it, and the 'structure beats scale' reading is much weaker here."
    return head + f" Worst corruption in any gated mode: {g:.0f}%."
