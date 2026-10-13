"""Ordinary-words test, single-step (docs/WORDS_TEST.md; prediction and reading rules written BEFORE any model run).

One model call per task, 160 tasks per regime: the 140 requests of `word_tasks.make_word_tasks(140, 7)` (the same graphs, kinds and correct changes as the scope test, asked in words) and the 20
ambiguity requests of `word_tasks.make_ambiguity_tasks(20, 7)` (the correct reaction is to change nothing). Every policy is a deterministic function of that one text and is evaluated offline on
the SAME model outputs:
  blind, gate, repair_m6   as before (they never read the request)
  scope_m7                 named scope by COORDINATES only: on a request in words it sees nothing named and is exactly repair_m6 (the hole this test is about)
  scope_words              named scope read from words as well as coordinates (`grounding.ground`); an ambiguous request applies nothing
Success = the final graph equals the expected graph exactly (for an ambiguity request: the graph as it was). All thresholds are integers."""
import json, os
from typing import List

from axi.experiments.agent_loop import copy_graph, same_graph, strict_parse
from axi.experiments.cascade import next_state
from axi.experiments.resumable import load_rows, _roundtrip
from axi.experiments.scope_tasks import KINDS_SCOPE
from axi.experiments.word_tasks import make_ambiguity_tasks, make_word_tasks, word_messages

POLICIES_WORDS = ("blind", "gate", "repair_m6", "scope_m7", "scope_words")
OLD_DELETES = ("delete_dep", "delete_leaf")
N_WORDS, N_AMBIG, SEED = 140, 20, 7


def word_set(n_words: int = N_WORDS, n_ambig: int = N_AMBIG, seed: int = SEED):
    """The 160 tasks of one regime: the word requests first, then the ambiguity requests."""
    return make_word_tasks(n_words, seed) + make_ambiguity_tasks(n_ambig, seed)


def run_word_regime(llm, tasks, hint_rules: bool, path: str, meta=None, progress=None) -> dict:
    """One JSON line per finished task; run again with the same arguments to carry on after a dropped session."""
    meta = dict(meta or {}, n=len(tasks), hint_rules=bool(hint_rules), test="words_single_step")
    rows = load_rows(path, meta)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w") as f: f.write(json.dumps({"_meta": _roundtrip(meta)}) + "\n")
    with open(path, "a") as f:
        for i in range(len(rows), len(tasks)):
            t = tasks[i]; text, ntok, sec = llm.generate(word_messages(t, hint_rules), constrained=True)
            row = _roundtrip(dict(kind=t.kind, text=text, tokens=ntok, seconds=sec)); f.write(json.dumps(row) + "\n"); f.flush(); os.fsync(f.fileno()); rows.append(row)
            if progress: progress(i + 1, len(tasks))
    return {"rows": rows, "summary": summarize_words(rows, tasks), "meta": meta}


def evaluate_words(rows, tasks) -> List[dict]:
    assert [r["kind"] for r in rows] == [t.kind for t in tasks], "rows are not the tasks of word_set()"
    out = []
    for row, t in zip(rows, tasks):
        d = strict_parse(row["text"]); r = dict(kind=t.kind, parsed=d is not None); ix = t.index(); fin = {}
        for p in POLICIES_WORDS:
            fin[p] = next_state(p, t.graph, d, t.instruction, names=ix)
            r[p] = dict(success=same_graph(fin[p], t.expected), corrupted=not fin[p].is_well_formed(), changed=not same_graph(fin[p], t.graph))
        r["m7_equals_m6"] = same_graph(fin["scope_m7"], fin["repair_m6"])
        out.append(r)
    return out


def _cell(rs, p): return dict(success=sum(r[p]["success"] for r in rs), corrupted=sum(r[p]["corrupted"] for r in rs), changed=sum(r[p]["changed"] for r in rs))


def summarize_words(rows, tasks) -> dict:
    ev = evaluate_words(rows, tasks); words = [r for r in ev if r["kind"] != "ambiguous"]
    groups = {k: [r for r in words if r["kind"] == k] for k in KINDS_SCOPE}
    groups["old_deletes"] = [r for r in words if r["kind"] in OLD_DELETES]; groups["all"] = words; groups["ambiguous"] = [r for r in ev if r["kind"] == "ambiguous"]
    out = {g: dict(n=len(rs), **{p: _cell(rs, p) for p in POLICIES_WORDS}) for g, rs in groups.items()}
    out["paired"] = dict(helped=sum(r["scope_words"]["success"] and not r["repair_m6"]["success"] for r in words), harmed=sum(r["repair_m6"]["success"] and not r["scope_words"]["success"] for r in words),
                         both_right=sum(r["repair_m6"]["success"] and r["scope_words"]["success"] for r in words), both_wrong=sum(not r["repair_m6"]["success"] and not r["scope_words"]["success"] for r in words))
    out["m7_equals_m6"] = sum(r["m7_equals_m6"] for r in words)
    out["unparsed"] = dict(words=sum(not r["parsed"] for r in words), ambiguous=sum(not r["parsed"] for r in groups["ambiguous"]))
    return out


# ---------------------------------------------------------------- reading rules (fixed in docs/WORDS_TEST.md before the run)
def verdict_words(summaries) -> List[tuple]:
    """summaries: {hint_rules(bool): summarize_words(...)}. Lines are tagged [guard] (the framework holds) or [need] (the raw model still fails)."""
    out = []; regs = sorted(summaries); names = {False: "no rule", True: "rule stated"}
    out.append(("[guard] gate, repair_m6, scope_m7 and scope_words corrupt no graph in any regime (the 140 and the 20)",
                all(summaries[h][g][p]["corrupted"] == 0 for h in regs for g in ("all", "ambiguous") for p in ("gate", "repair_m6", "scope_m7", "scope_words"))))
    out.append(("[guard] ambiguous requests: scope_words changes the graph on none of the 20, in every regime", all(summaries[h]["ambiguous"]["scope_words"]["changed"] == 0 for h in regs)))
    out.append(("[guard] scope_words harms at most 2 of the 140 per regime (repair_m6 right, scope_words wrong)", all(summaries[h]["paired"]["harmed"] <= 2 for h in regs)))
    out.append(("[guard] scope_m7 (coordinates only) is within 2 tasks of repair_m6 on the 140 word requests in each regime",
                all(abs(summaries[h]["all"]["scope_m7"]["success"] - summaries[h]["all"]["repair_m6"]["success"]) <= 2 for h in regs)))
    out.append(("[need] ambiguous requests: blind apply changes the graph on at least half of the 20 in each regime", all(2 * summaries[h]["ambiguous"]["blind"]["changed"] >= summaries[h]["ambiguous"]["n"] for h in regs)))
    if True in summaries:
        s = summaries[True]; out.append((f"[need] {names[True]}: scope_words is ahead of repair_m6 by at least 5 of the 140", s["all"]["scope_words"]["success"] - s["all"]["repair_m6"]["success"] >= 5))
    return out


def verdict_words_format(rows_by_regime) -> List[tuple]:
    """rows_by_regime: {hint_rules: rows}. [format] at most 5 percent of all replies do not parse (integer arithmetic: 20 times the unparsed count at most the number of replies)."""
    bad = sum(1 for rs in rows_by_regime.values() for r in rs if strict_parse(r["text"]) is None); tot = sum(len(rs) for rs in rows_by_regime.values())
    return [(f"[format] at most 5 percent of the model's replies do not parse ({bad} of {tot})", tot > 0 and 20 * bad <= tot)]


def print_words(summary, title=""):
    cols = POLICIES_WORDS
    print(f"== words test, single step {title} (success = final graph equals the expected graph; for 'ambiguous' = the graph as it was) ==")
    print(f"{'kind':<14}{'n':>3} | " + " ".join(f"{c:>12}" for c in cols))
    for g in list(KINDS_SCOPE) + ["old_deletes", "all", "ambiguous"]:
        s = summary[g]; print(f"{g:<14}{s['n']:>3} | " + " ".join(f"{s[c]['success']:>12}" for c in cols))
    a = summary["all"]; amb = summary["ambiguous"]
    print("graphs left corrupted (the 140): " + ", ".join(f"{c} {a[c]['corrupted']}" for c in cols))
    print("ambiguous requests where the graph was changed: " + ", ".join(f"{c} {amb[c]['changed']} of {amb['n']}" for c in cols))
    p = summary["paired"]; print(f"scope_words against repair_m6, task by task: helped {p['helped']}, harmed {p['harmed']}, both right {p['both_right']}, both wrong {p['both_wrong']}")
    print(f"unparsed replies: {summary['unparsed']['words']} of the 140, {summary['unparsed']['ambiguous']} of the 20")


def print_verdict(checks, title):
    print(f"\n-- {title} (reading rule fixed before the run) --")
    for text, ok in checks: print(f"  {'yes' if ok else 'NO ':<3}  {text}")
