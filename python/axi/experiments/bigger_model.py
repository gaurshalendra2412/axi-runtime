"""Reading rules of the bigger-model test (docs/SCALE_TEST_14B.md; written BEFORE the run). No model code here: the runners are the unchanged ones of scope_test.py and long_horizon.py.

Every line is tagged [guard] (the framework holds the state / never corrupts) or [need] (the raw model still fails at this size). All thresholds are integers or share
comparisons with a 1e-12 epsilon, so that no floating-point slip can move a verdict (0.10 * 140 = 14.000000000000002 was one in an earlier test)."""
from typing import List, Optional


def verdict_big_single(summaries) -> List[tuple]:
    """summaries: {hint_rules(bool): summarize_scope(...)}. Missing regimes are skipped."""
    out = []; regs = sorted(summaries)
    out.append(("[guard] gate, repair_m6 and scope_m7 corrupt no graph in any regime", all(summaries[h]["all"][p]["corrupted"] == 0 for h in regs for p in ("gate", "repair_m6", "scope_m7"))))
    out.append(("[guard] scope_m7 harms at most 2 tasks per regime (repair_m6 right, scope_m7 wrong)", all(summaries[h]["paired"]["harmed"] <= 2 for h in regs)))
    if False in summaries:
        s = summaries[False]; n = s["all"]["n"]; dd = s["delete_dep"]
        out.append((f"[guard] no rule: repair_m6 solves at least 90 percent of the {n} tasks", 10 * s["all"]["repair_m6"]["success"] >= 9 * n))
        out.append(("[guard] no rule: scope_m7 is within 3 points of repair_m6 overall", 100 * abs(s["all"]["scope_m7"]["success"] - s["all"]["repair_m6"]["success"]) <= 3 * n))
        out.append((f"[need] no rule: blind apply is right on at most half of the {dd['n']} delete_dep requests", 2 * dd["blind"]["success"] <= dd["n"]))
        out.append(("[need] no rule: repair_m6 is ahead of blind apply by at least 10 points", 10 * (s["all"]["repair_m6"]["success"] - s["all"]["blind"]["success"]) >= n))
    if True in summaries:
        s = summaries[True]; n = s["all"]["n"]; od = s["old_deletes"]
        out.append((f"[guard] rule stated: scope_m7 solves at least 90 percent of the old delete requests (delete_dep + delete_leaf, n={od['n']})", 10 * od["scope_m7"]["success"] >= 9 * od["n"]))
        out.append(("[need] rule stated: scope_m7 is ahead of blind apply by at least 10 points", 10 * (s["all"]["scope_m7"]["success"] - s["all"]["blind"]["success"]) >= n))
    return out


def verdict_big_long(rule_summary=None, norule_summary=None) -> List[tuple]:
    """Reading rules for the long-horizon part (summarize_long output of the scope-mix state runs with policies blind, repair_m6, scope_m7)."""
    out = []; ss = [x for x in (rule_summary, norule_summary) if x]
    out.append(("[guard] repair_m6 and scope_m7 are never corrupted at any step", all(x[p]["episodes_ever_corrupted"] == 0 for x in ss for p in ("repair_m6", "scope_m7"))))
    if norule_summary:
        x = norule_summary; m6, m7 = x["repair_m6"]["exact_share"], x["scope_m7"]["exact_share"]
        out.append(("[guard] no rule: repair_m6 and scope_m7 each hold the oracle's state on at least 95 percent of steps", m6 >= 0.95 - 1e-12 and m7 >= 0.95 - 1e-12))
        out.append(("[guard] no rule: the two are within 3 points of each other", abs(m7 - m6) <= 0.03 + 1e-12))
        out.append(("[need] no rule: blind apply is corrupted by step 10 in at least half of the episodes", x["blind"]["corrupted_by_step_10"] >= 0.5 - 1e-12))
    if rule_summary:
        x = rule_summary; m6, m7 = x["repair_m6"]["exact_share"], x["scope_m7"]["exact_share"]
        out.append(("[guard] rule stated: scope_m7 holds the oracle's state on at least 90 percent of steps", m7 >= 0.90 - 1e-12))
        out.append(("[guard] rule stated: scope_m7 is not behind repair_m6 by more than 3 points", m6 - m7 <= 0.03 + 1e-12))
        out.append(("[need] rule stated: blind apply is corrupted by step 10 in at least half of the episodes", x["blind"]["corrupted_by_step_10"] >= 0.5 - 1e-12))
    return out


def smoke_check(summary, min_right: int = 9) -> tuple:
    """Stop-early test for the 14-task smoke run (rule stated). A model whose output is broken (for example fp16 overflow giving one repeated token) solves almost nothing;
    a working one solves most. The bar is deliberately low (the 7B's rate on the same kind of task was about 85 percent)."""
    n = summary["all"]["n"]; m6 = summary["all"]["repair_m6"]["success"]; blind = summary["all"]["blind"]["success"]
    ok = m6 >= min_right
    msg = (f"SMOKE OK: repair_m6 right on {m6} of {n} (blind apply {blind}). Go on to the next cell." if ok else
           f"STOP: repair_m6 right on only {m6} of {n} (blind apply {blind}). The model's output looks broken. Do not run the long cells; send me this printout and the lines above it.")
    return ok, msg


def mean_seconds(rows) -> Optional[float]:
    """Mean model seconds per request in a single-step file (used by the notebook and by the result check)."""
    return sum(r["seconds"] for r in rows) / len(rows) if rows else None
