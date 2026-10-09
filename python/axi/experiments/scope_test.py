"""Scope test, single-step part (docs/SCOPE_TEST.md; prediction and reading rules written BEFORE any run).

One constrained model call per task (the same prompts as every earlier run: `SYSTEM`, optional `RULES_HINT`; the prompt shows the graph and the request). Every policy is a
deterministic function of that one text, so all of them are evaluated on the SAME model outputs, offline:
  blind        apply what the model wrote                    repair_m6    gate + ground-and-complete (the current best)
  gate         refuse an inadmissible proposal               scope_m7     named scope (axi/engine/scope.py) then repair_m6
  scope_delta  the post-hoc idea: keep an edge delete only if it touches a service deleted in the same delta (the baseline to beat, and to show where it is wrong)
Seven kinds of request, 20 of each at n = 140 (`scope_tasks.make_scope_tasks`): delete_dep, delete_leaf, add_node, add_edge (as before) and delete_edge, delete_pair, delete_weight (new,
chosen to be where a scope rule could go wrong). Success = the final graph equals the expected graph exactly.

The long-horizon part (drift over 30 steps) reuses `long_horizon.run_long(..., mix='scope', policies=('repair_m6', 'scope_m7'))`; its reading rules are `verdict_scope_long` below."""
import json, os
from typing import List

from axi.engine.scope import admit_scoped
from axi.experiments.agent_loop import base_messages, copy_graph, same_graph, strict_parse
from axi.experiments.cascade import next_state
from axi.experiments.resumable import load_rows, _roundtrip
from axi.experiments.scope_tasks import KINDS_SCOPE, make_scope_tasks

POLICIES_SCOPE = ("blind", "gate", "repair_m6", "scope_m7", "scope_delta")
OLD_DELETES = ("delete_dep", "delete_leaf")


# ---------------------------------------------------------------- the one model call per task, resumable
def run_scope_regime(llm, tasks, hint_rules: bool, path: str, meta=None, progress=None) -> dict:
    """One JSON line per finished task; run again with the same arguments to carry on after a dropped session."""
    meta = dict(meta or {}, n=len(tasks), hint_rules=bool(hint_rules), test="scope_single_step")
    rows = load_rows(path, meta)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w") as f: f.write(json.dumps({"_meta": _roundtrip(meta)}) + "\n")
    with open(path, "a") as f:
        for i in range(len(rows), len(tasks)):
            t = tasks[i]; text, ntok, sec = llm.generate(base_messages(t, hint_rules), constrained=True)
            row = _roundtrip(dict(kind=t.kind, text=text, tokens=ntok, seconds=sec)); f.write(json.dumps(row) + "\n"); f.flush(); os.fsync(f.fileno()); rows.append(row)
            if progress: progress(i + 1, len(tasks))
    return {"rows": rows, "summary": summarize_scope(rows, tasks), "meta": meta}


# ---------------------------------------------------------------- evaluation (no model)
def evaluate(rows, tasks) -> List[dict]:
    assert [r["kind"] for r in rows] == [t.kind for t in tasks], "rows are not the tasks of make_scope_tasks(n, seed)"
    out = []
    for row, t in zip(rows, tasks):
        d = strict_parse(row["text"]); r = dict(kind=t.kind, parsed=d is not None); fin = {}
        for p in POLICIES_SCOPE:
            fin[p] = next_state(p, t.graph, d, t.instruction)
            r[p] = dict(success=same_graph(fin[p], t.expected), corrupted=not fin[p].is_well_formed())
        r["m7_equals_m6"] = same_graph(fin["scope_m7"], fin["repair_m6"])
        r["dropped"] = len(admit_scoped(t.graph, d, t.instruction, "named").dropped) if d is not None else 0
        out.append(r)
    return out


def _cell(rs, p): return dict(success=sum(r[p]["success"] for r in rs), corrupted=sum(r[p]["corrupted"] for r in rs))


def summarize_scope(rows, tasks) -> dict:
    ev = evaluate(rows, tasks); groups = {k: [r for r in ev if r["kind"] == k] for k in KINDS_SCOPE}
    groups["old_deletes"] = [r for r in ev if r["kind"] in OLD_DELETES]; groups["all"] = ev
    out = {g: dict(n=len(rs), **{p: _cell(rs, p) for p in POLICIES_SCOPE}) for g, rs in groups.items()}
    out["paired"] = dict(helped=sum(r["scope_m7"]["success"] and not r["repair_m6"]["success"] for r in ev), harmed=sum(r["repair_m6"]["success"] and not r["scope_m7"]["success"] for r in ev),
                         both_right=sum(r["repair_m6"]["success"] and r["scope_m7"]["success"] for r in ev), both_wrong=sum(not r["repair_m6"]["success"] and not r["scope_m7"]["success"] for r in ev))
    out["m7_equals_m6"] = {k: sum(r["m7_equals_m6"] for r in rs) for k, rs in groups.items()}
    out["dropped"] = dict(tasks_with_drops=sum(r["dropped"] > 0 for r in ev), items=sum(r["dropped"] for r in ev),
                          tasks_with_drops_and_right=sum(r["dropped"] > 0 and r["scope_m7"]["success"] for r in ev))
    return out


# ---------------------------------------------------------------- reading rules (fixed in docs/SCOPE_TEST.md before the run)
def verdict_scope(summaries) -> List[tuple]:
    """summaries: {hint_rules(bool): summarize_scope(...)}. Missing regimes are skipped."""
    out = []; regs = sorted(summaries)
    name = lambda h: "rule stated" if h else "no rule"
    out.append(("scope_m7 corrupts no graph in any regime", all(summaries[h]["all"]["scope_m7"]["corrupted"] == 0 for h in regs)))
    out.append(("scope_m7 harms at most 2 tasks per regime (repair_m6 right, scope_m7 wrong)", all(summaries[h]["paired"]["harmed"] <= 2 for h in regs)))
    out.append(("delete_weight (the request names no service): scope_m7 gives the same final graph as repair_m6 on every task", all(summaries[h]["m7_equals_m6"]["delete_weight"] == summaries[h]["delete_weight"]["n"] for h in regs)))
    out.append(("the post-hoc filter scope_delta solves at most 10 percent of delete_edge, delete_pair and delete_weight requests, in every regime",
                all(10 * summaries[h][k]["scope_delta"]["success"] <= summaries[h][k]["n"] for h in regs for k in ("delete_edge", "delete_pair", "delete_weight"))))
    if True in summaries:
        s = summaries[True]
        out.append((f"rule stated: scope_m7 solves at least 90 percent of the old delete requests (delete_dep + delete_leaf, n={s['old_deletes']['n']})", 10 * s["old_deletes"]["scope_m7"]["success"] >= 9 * s["old_deletes"]["n"]))
        out.append(("rule stated: scope_m7 beats repair_m6 overall by at least 10 points", 10 * (s["all"]["scope_m7"]["success"] - s["all"]["repair_m6"]["success"]) >= s["all"]["n"]))
    if False in summaries:
        s = summaries[False]
        out.append(("no rule: scope_m7 is within 3 points of repair_m6 overall", 100 * abs(s["all"]["scope_m7"]["success"] - s["all"]["repair_m6"]["success"]) <= 3 * s["all"]["n"]))
    return out


def verdict_scope_long(rule_summary=None, norule_summary=None) -> List[tuple]:
    """Reading rules for the long-horizon part (summarize_long output of the scope-mix state runs, policies repair_m6 and scope_m7)."""
    out = []
    ss = [x for x in (rule_summary, norule_summary) if x]
    out.append(("scope_m7 is never corrupted at any step", all(x["scope_m7"]["episodes_ever_corrupted"] == 0 for x in ss)))
    if rule_summary:
        m6, m7 = rule_summary["repair_m6"]["exact_share"], rule_summary["scope_m7"]["exact_share"]
        out.append(("rule stated: repair_m6 holds the oracle's state on at most 75 percent of steps", m6 <= 0.75 + 1e-12))
        out.append(("rule stated: scope_m7 holds the oracle's state on at least 90 percent of steps", m7 >= 0.90 - 1e-12))
        out.append(("rule stated: scope_m7 is ahead of repair_m6 by at least 15 points", m7 - m6 >= 0.15 - 1e-12))
    if norule_summary:
        m6, m7 = norule_summary["repair_m6"]["exact_share"], norule_summary["scope_m7"]["exact_share"]
        out.append(("no rule: both policies hold the oracle's state on at least 95 percent of steps", m6 >= 0.95 - 1e-12 and m7 >= 0.95 - 1e-12))
        out.append(("no rule: the two are within 3 points of each other", abs(m7 - m6) <= 0.03 + 1e-12))
    return out


# ---------------------------------------------------------------- printing
def print_scope(summary, title=""):
    print(f"\n== scope test, single step {title} (success = final graph equals the expected graph) ==")
    names = {"blind": "blind", "gate": "gate", "repair_m6": "repair_m6", "scope_m7": "scope_m7", "scope_delta": "scope_delta (post-hoc)"}
    print(f"{'kind':14s} {'n':>3s} | " + " ".join(f"{names[p][:12]:>12s}" for p in POLICIES_SCOPE) + " | scope_m7 = repair_m6")
    for g in list(KINDS_SCOPE) + ["old_deletes", "all"]:
        s = summary[g]; print(f"{g:14s} {s['n']:3d} | " + " ".join(f"{s[p]['success']:12d}" for p in POLICIES_SCOPE) + f" | {summary['m7_equals_m6'][g]:3d} of {s['n']}")
    a = summary["all"]; pr = summary["paired"]
    print(f"graphs left corrupted: " + ", ".join(f"{p} {a[p]['corrupted']}" for p in POLICIES_SCOPE))
    print(f"scope_m7 against repair_m6, task by task: helped {pr['helped']}, harmed {pr['harmed']}, both right {pr['both_right']}, both wrong {pr['both_wrong']}")
    d = summary["dropped"]; print(f"scope filter: dropped something on {d['tasks_with_drops']} tasks ({d['items']} items); {d['tasks_with_drops_and_right']} of those tasks still ended right")


def print_exact(summary, title=""):
    print(f"\n== exact steps, {title} ==")
    for p, v in summary.items(): print(f"{p:11s} episodes {v['episodes']:3d}  exact steps {v['exact_share']*100:5.1f} %  final distance {v['final_distance']:.2f}  ever corrupted {v['episodes_ever_corrupted']*100:.0f} %")


def print_verdict(checks, title):
    print(f"\n-- {title} (reading rule fixed before the run) --")
    for text, ok in checks: print(("  yes  " if ok else "  NO   ") + text)
