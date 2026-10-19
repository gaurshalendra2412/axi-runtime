"""Provider-side serving test, third run: the second run's constraint with the length of a relation name capped at 16 characters. Page with the predictions written BEFORE any run:
docs/PROVIDER_SIDE_BOUNDED_TEST.md (concept-map item 108). The second run (docs/PROVIDER_SIDE_CONSTRAINED_TEST.md, its Result) read line 2 NO: one reply of the exact-state arm (job 19, step 12) ran to
the 96-token limit inside a name, because the regular expression of that run, `[A-Za-z0-9_]+`, has no limit on the length of a name and greedy decoding looped. By the page's own rule neither the
plain time ratios nor the exact-steps-per-hour figures were claimed from that run. This run caps the name and runs the cells again.

What is the same (nothing below is changed; `provider_side.py` and `provider_side_constrained.py` are imported and not edited: their sha256 are pinned on the page): the model, the engine class, the
80,000-token cache, the 30-step jobs and their seeds, the lockstep rounds, the policy repair_m6, the measures, the runner and its checkpoint, the summaries.
What is new:
  * the regular expression: the same delta language, with a name of 1 to 16 characters (`{1,16}`) in place of one or more;
  * the cells: `S` at 8, 32 and 64 jobs, and `KC` at 8 jobs as a check that the kept chat is unchanged by the cap. The kept chat at 32 and 64 jobs (5 and 3 hours of GPU) is NOT run again: its cells
    are those of the second run, read from the saved file (its sha256 is pinned on the page), and the page says why that is allowed and what it does not show;
  * a replay check: every reply of the new cells is compared with the reply of the same job at the same step in the second run's file (a reply the grammar allows is not changed by it, with greedy decoding);
  * 13 reading lines.

Honest limits: as the second run (one model, one small GPU, one engine and its version, synthetic graphs, greedy decoding, lockstep rounds, GPU seconds and not dollars, no energy; the grammar makes
a reply well formed, not right), plus: the kept chat at 32 and 64 jobs ran under the uncapped expression. `BoundedStandIn` (a model of a prefix cache) is used to test the harness only; its numbers
are never a result."""
import hashlib, json, os
from typing import Dict, List, Optional, Tuple

import axi.experiments.provider_side as PS
import axi.experiments.provider_side_constrained as C
from axi.experiments.provider_side import _ge, _le

# ---------------------------------------------------------------- the settings written on the page
NAME_CAP = 16
ARMS = ("S", "KC")
LEVELS = (8, 32, 64)
MODEL, STEPS, SEED, POLICY, HINT_RULES, MAX_NEW, CAPACITY_TOKENS = C.MODEL, C.STEPS, C.SEED, C.POLICY, C.HINT_RULES, C.MAX_NEW, C.CAPACITY_TOKENS

_NAME = r"[A-Za-z0-9_]{1,%d}" % NAME_CAP
_ITEM = r"(?:ADD\[[0-9]+,[0-9]+:-?[0-9]+\]|DEL\[[0-9]+,[0-9]+\]|(?:ADD|DEL)\[\([0-9]+,[0-9]+\)->\([0-9]+,[0-9]+\):" + _NAME + r"#-?[0-9]+\])"
REGEX = _ITEM + r"(?:[ \n]" + _ITEM + r")*[ \n]?"
REGEX_SHA256 = hashlib.sha256(REGEX.encode()).hexdigest()

LOOP_JOB, LOOP_STEP = 19, 12                      # the reply that ran to 96 tokens in the second run: job 19 (0-based), step 12 (1-based)
B_PARSED = 0.995                                  # line 2
B_SHARE_TOL = 0.005                               # line 6: the exact share of an `S` cell within half a point of the second run's
B_TIME_TOL = 0.30                                 # lines 7 and 8: seconds per step within 30 percent of the second run's
B_QA_LIGHT, B_QA_MID, B_QA_HEAVY = 2.0, 10.0, 10.0
B_EXACT_GAP, B_EXACT_HIGH = 0.05, 0.98

BL_LINES = [
    ("instrument", "the engine reported how many prompt tokens were served from its cache, for every request of every cell that was run"),
    ("instrument", "the constraint held: in every cell that was run at least 99.5 percent of the replies were accepted by the strict parser, and no reply used all 96 new tokens"),
    ("guard", "every reply of every `S` cell is the same text as the reply of the same job at the same step in the second run, except the steps of job 19 from step 12 on"),
    ("guard", "every reply of the `KC` cell at 8 jobs is the same text as in the second run, and that cell is 100 percent exact"),
    ("guard", "the reply that ran to 96 tokens in the second run (job 19, step 12) is now accepted by the strict parser, is not admitted by the gate, and uses fewer than 96 new tokens"),
    ("claim", "the exact share of every `S` cell is within half a point of the second run's (88.3, 94.6 and 97.2 percent)"),
    ("claim", "the seconds per step of every `S` cell are within 30 percent of the second run's (0.152, 0.118 and 0.106)"),
    ("claim", "the seconds per step of the `KC` cell at 8 jobs are within 30 percent of the second run's (0.608)"),
    ("claim", "at K = 8 exact-state does at least 2 times as many exact steps per GPU hour as the kept chat (both cells of this run)"),
    ("claim", "at K = 32 exact-state (this run) does at least 10 times as many exact steps per GPU hour as the kept chat (the second run's cell)"),
    ("claim", "at K = 64 exact-state (this run) does at least 10 times as many exact steps per GPU hour as the kept chat (the second run's cell)"),
    ("claim", "exact-state was not less exact than the kept chat: at every K its share of exact steps is at least the kept chat's minus 5 points"),
    ("claim", "exact-state held the right state on at least 98 percent of the steps in every one of its cells"),
]
BL_HEADLINE = (9, 11)


# ---------------------------------------------------------------- the engines
def constraint_info(kind: str) -> dict:
    return dict(kind=kind, api="structured_outputs(regex)", regex=REGEX, regex_sha256=REGEX_SHA256, name_cap=NAME_CAP)


class BoundedVLLMEngine(PS.VLLMEngine):
    """The first run's engine with the reply constrained by the capped REGEX. Not testable without a GPU; tests/test_provider_side_bounded.py runs it against a fake `vllm` module, and the notebook's
    safety cell feeds it the prompt of the reply that ran to 96 tokens in the second run."""
    def __init__(self, *a, **kw):
        params = C._structured_params(REGEX)                                 # before the model is loaded: a version that cannot constrain fails in a second, not after ten minutes
        super().__init__(*a, **kw)
        from vllm import SamplingParams
        self.sp = SamplingParams(temperature=0.0, max_tokens=MAX_NEW, structured_outputs=params)

    def info(self) -> dict:
        d = super().info(); d["constraint"] = constraint_info("regex"); return d


class BoundedStandIn(PS.StandInEngine):
    """The stand-in with the constraint recorded (it replies with the oracle's own reply, which is in the language). Tests the harness only. Never a result."""
    def info(self) -> dict:
        d = super().info(); d["constraint"] = constraint_info("stand_in"); return d


# ---------------------------------------------------------------- planned cells, the runner (the second run's, unchanged)
def cells_planned() -> List[Tuple[str, int]]:
    """In the order they are run: the cheap ones, all of them. About ten minutes of GPU in all."""
    return [("S", 8), ("KC", 8), ("S", 32), ("S", 64)]


run_cell = C.run_cell                              # same jobs, same rounds, same checkpoint; the engine decides the constraint
save_cell = C.save_cell
summarize_cell2 = C.summarize_cell2
summarize_all = C.summarize_all


def loop_probe(engine, seed: int = SEED) -> dict:
    """One request: the exact-state prompt of step 12 of job 19, rebuilt with the oracle's own replies for steps 1 to 11 (the job held the right state up to step 11 in the second run). In the second run
    the engine answered `ADD[(3,0)->(3,0):4tworowsdown4colsright...` there, 96 tokens without a closing bracket."""
    return C.slip_probe(engine, episode=LOOP_JOB, step=LOOP_STEP, seed=seed)


# ---------------------------------------------------------------- the second run's file
def default_update58_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "docs", "results", "axi_provider_constrained.json")


def load_second_run(path: Optional[str] = None) -> dict:
    return json.load(open(path or default_update58_path()))


def sha256_file(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def compare_with_second_run(new: dict, old: dict) -> dict:
    """The same job at the same step: is the reply the same text? `allowed` are the differences the page allows (job 19 from step 12 on), `unexpected` the others, `length_mismatch` the jobs that
    did not take the same number of steps."""
    assert (new["arm"], new["K"]) == (old["arm"], old["K"]), ((new["arm"], new["K"]), (old["arm"], old["K"]))
    steps = same = 0; allowed: List[Tuple[int, int]] = []; unexpected: List[Tuple[int, int]] = []; mismatch: List[int] = []
    for j, (a, b) in enumerate(zip(new["sessions"], old["sessions"])):
        if len(a["log"]) != len(b["log"]): mismatch.append(j)
        for x, y in zip(a["log"], b["log"]):
            steps += 1
            if x["text"] == y["text"]: same += 1
            else: (allowed if (j == LOOP_JOB and x["t"] >= LOOP_STEP) else unexpected).append((j, x["t"]))
    if len(new["sessions"]) != len(old["sessions"]): mismatch.append(-1)
    return dict(steps=steps, same=same, allowed=allowed, unexpected=unexpected, length_mismatch=mismatch)


def loop_step(cell: dict) -> Optional[dict]:
    """The record of job 19, step 12, if the cell has that job and step."""
    if len(cell["sessions"]) <= LOOP_JOB or len(cell["sessions"][LOOP_JOB]["log"]) < LOOP_STEP: return None
    return cell["sessions"][LOOP_JOB]["log"][LOOP_STEP - 1]


# ---------------------------------------------------------------- merge
def merge_cells(cells: List[dict]) -> dict:
    """Puts cell files together. Refuses cells that were not run under the same settings (the constraint included), a cell twice, cells that carry another constraint or none."""
    assert cells, "nothing to merge"
    ref = cells[0]
    key = lambda c: (c["steps"], c["seed"], c["policy"], c["hint_rules"], c["engine"]["kind"], c["engine"].get("model"), c["engine"].get("capacity_tokens"), c["engine"].get("version"), c["engine"].get("gpu"),
                     (c["engine"].get("constraint") or {}).get("regex_sha256"))
    seen = set()
    for c in cells:
        if (c["engine"].get("constraint") or {}).get("regex_sha256") != REGEX_SHA256: raise ValueError(f"cell {c['arm']}/K={c['K']} was not run with this test's constraint")
        if key(c) != key(ref): raise ValueError(f"cell {c['arm']}/K={c['K']} was run under other settings: {key(c)} against {key(ref)}")
        if (c["arm"], c["K"]) in seen: raise ValueError(f"cell {c['arm']}/K={c['K']} twice")
        if (c["arm"], c["K"]) not in cells_planned(): raise ValueError(f"cell {c['arm']}/K={c['K']} is not one of this test's cells")
        seen.add((c["arm"], c["K"]))
    order = {k: i for i, k in enumerate(cells_planned())}
    return dict(meta=dict(model=ref["engine"].get("model"), engine=ref["engine"], steps=ref["steps"], seed=ref["seed"], policy=ref["policy"], hint_rules=ref["hint_rules"], name_cap=NAME_CAP, cells=[list(k) for k in cells_planned()]),
                cells=sorted(cells, key=lambda c: order[(c["arm"], c["K"])]))


# ---------------------------------------------------------------- reading rules (fixed in docs/PROVIDER_SIDE_BOUNDED_TEST.md before the run)
def _rate_ratio(a, b):
    if a is None or b is None: return None
    if b == 0: return float("inf") if a > 0 else None
    return a / b


def verdict_bounded(new_cells: List[dict], old_cells: List[dict]) -> List[Tuple[int, str, str, Optional[bool]]]:
    """(number, tag, text, yes / no / None = not run). `new_cells` are the cells of this run, `old_cells` those of the second run (the kept chat at 32 and 64 jobs is read from them). A line that needs a
    cell that was not run is None, never 'no'."""
    new = {(c["arm"], c["K"]): c for c in new_cells if any(s["log"] for s in c["sessions"])}; old = {(c["arm"], c["K"]): c for c in old_cells}
    Sn = {k: C.summarize_cell2(c) for k, c in new.items()}; So = {k: C.summarize_cell2(c) for k, c in old.items()}
    s_new = [k for k in new if k[0] == "S"]
    cmp_ = {k: compare_with_second_run(new[k], old[k]) for k in new if k in old}
    out: List[Optional[bool]] = []
    out.append(all(s["cached_tokens"] is not None for s in Sn.values()) if Sn else None)                                                                                              # 1
    out.append(all(_le(s["unparsed"] / s["steps_done"], 1 - B_PARSED) and s["truncated"] == 0 for s in Sn.values()) if Sn else None)                                                 # 2
    out.append(all(not cmp_[k]["unexpected"] and not cmp_[k]["length_mismatch"] for k in s_new) if s_new else None)                                                                    # 3
    out.append(None if ("KC", 8) not in new else (cmp_[("KC", 8)]["same"] == cmp_[("KC", 8)]["steps"] and not cmp_[("KC", 8)]["length_mismatch"] and _ge(Sn[("KC", 8)]["exact_share"], 1.0)))      # 4
    recs = [loop_step(new[k]) for k in s_new if loop_step(new[k]) is not None]
    out.append(all(r["parsed"] and not r["admitted"] and r["n_new"] < MAX_NEW for r in recs) if recs else None)                                                                        # 5
    out.append(all(abs(Sn[k]["exact_share"] - So[k]["exact_share"]) <= B_SHARE_TOL + 1e-12 for k in s_new) if s_new else None)                                                         # 6
    out.append(all(abs(Sn[k]["seconds_per_step"] / So[k]["seconds_per_step"] - 1) <= B_TIME_TOL + 1e-12 for k in s_new) if s_new else None)                                           # 7
    out.append(None if ("KC", 8) not in new else abs(Sn[("KC", 8)]["seconds_per_step"] / So[("KC", 8)]["seconds_per_step"] - 1) <= B_TIME_TOL + 1e-12)                                 # 8
    kc = {8: Sn.get(("KC", 8)), 32: So.get(("KC", 32)), 64: So.get(("KC", 64))}                                                                                                         # the kept chat: 8 from this run, 32 and 64 from the second
    for K, lim in ((8, B_QA_LIGHT), (32, B_QA_MID), (64, B_QA_HEAVY)):                                                                                                                  # 9 to 11
        s_, k_ = Sn.get(("S", K)), kc[K]
        r = _rate_ratio(s_["exact_per_hour"], k_["exact_per_hour"]) if s_ and k_ else None; out.append(None if r is None else _ge(r, lim))
    pairs = [(Sn[("S", K)], kc[K]) for K in LEVELS if ("S", K) in Sn and kc[K]]
    out.append(all(_ge(a["exact_share"], b["exact_share"] - B_EXACT_GAP) for a, b in pairs) if pairs else None)                                                                         # 12
    out.append(all(_ge(Sn[k]["exact_share"], B_EXACT_HIGH) for k in s_new) if s_new else None)                                                                                         # 13
    assert len(out) == len(BL_LINES)
    return [(i + 1, BL_LINES[i][0], BL_LINES[i][1], out[i]) for i in range(len(BL_LINES))]


def claims_allowed(checks) -> dict:
    """What the page lets the Result claim, from the lines: the exact-steps-per-hour lines (9 to 11) are claimed as read, with the exactness of both arms next to every figure, only if line 2 (the
    constraint held) and line 3 (the replies are the second run's) read yes; the point ratios of lines 10 and 11 (the kept chat's cells are those of another Colab session) are quoted as numbers only
    if lines 7 and 8 read yes too (the new session was no more than 30 percent from the second run's in speed, for both arms), otherwise only as 'at least' the line's limit; the plain time ratios are
    claimed as ratios at equal exactness only if line 12 reads yes. A line that was not run claims nothing."""
    ok = {i: c for i, _, _, c in checks}; base = ok[2] is True and ok[3] is True
    return dict(constraint_held=ok[2], replies_as_in_second_run=ok[3], base=base, quality_adjusted={i: (ok[i] if base else None) for i in (9, 10, 11)},
                point_ratios=bool(base and ok[7] is True and ok[8] is True), plain_time_at_equal_exactness=(ok[12] if base else None))


# ---------------------------------------------------------------- printing
def print_cells(S: Dict[Tuple[str, int], dict]) -> None:
    C.print_cells(S)


def print_verdict(checks, title="provider-side test, name capped at 16") -> None:
    print(f"\n-- {title} (reading rules fixed before the run) --")
    for i, tag, text, ok in checks: print(("  yes      " if ok else "  NO       " if ok is False else "  not run  ") + f"{i:2d}. [{tag}] {text}")
    ran = [ok for _, _, _, ok in checks if ok is not None]
    print(f"  ({len(ran)} of {len(checks)} lines could be read; headline lines {BL_HEADLINE[0]} and {BL_HEADLINE[1]}: " + ", ".join("yes" if checks[i - 1][3] else "NO" if checks[i - 1][3] is False else "not run" for i in BL_HEADLINE) + ")")
    a = claims_allowed(checks)
    if not a["base"]: print("  rule on the page: line 2 or line 3 did not read yes, so NO claim is made from this run until the cause is found.")
    else:
        print("  rule on the page: lines 2 and 3 read yes, so the exact-steps-per-hour lines (9 to 11) are claimed as read, with the exactness of both arms next to every figure; " +
              ("line 12 reads yes, so the plain time ratios may be claimed at equal exactness." if a["plain_time_at_equal_exactness"] else "line 12 reads NO, so the plain time ratios are NOT claimed as equal-exactness ratios."))
        print("  " + ("lines 7 and 8 read yes, so the point ratios of lines 10 and 11 may be quoted as numbers, with the two sessions named." if a["point_ratios"] else
                      "line 7 or 8 did not read yes, so lines 10 and 11 are quoted only as 'at least' their limit, with the speed of both sessions beside it."))
        print("  never: a multiple of capacity, dollars, a statement about any provider.")


def fingerprint() -> str:
    """sha256 of this module's source, recorded in the merged file so a result says which code made it."""
    return hashlib.sha256(open(__file__, "rb").read()).hexdigest()


def merge_files(paths: List[str], out: str, second_run: Optional[str] = None) -> dict:
    cells = [json.load(open(p)) for p in paths]; m = merge_cells(cells); sr = second_run or default_update58_path()
    m["second_run_file"] = os.path.basename(sr); m["second_run_sha256"] = sha256_file(sr)
    m["module_sha256"] = fingerprint(); m["constrained_sha256"] = C.fingerprint(); m["provider_side_sha256"] = PS.fingerprint()
    with open(out, "w") as f: json.dump(m, f)
    return m


def dry_run(steps: int = STEPS, levels=None) -> Dict[Tuple[str, int], dict]:
    """Every planned cell on the stand-in, with the cache sized to match the proxy's token count. Prints the tables. NOT a result (the stand-in replies with the oracle's own reply)."""
    eng = BoundedStandIn(capacity_tokens=int(CAPACITY_TOKENS * PS.PROXY_SCALE)); cells = []
    for arm, k in cells_planned():
        if levels is None or k in levels: cells.append(run_cell(eng, arm, k, steps))
    S = summarize_all(cells); print("STAND-IN (a model of a block cache), NOT a result"); print_cells(S); return S


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="provider-side test, name capped (docs/PROVIDER_SIDE_BOUNDED_TEST.md)")
    ap.add_argument("--dry-run", action="store_true", help="run every cell on the stand-in engine (no GPU); prints, does not save")
    ap.add_argument("--merge", nargs="+", metavar="CELL.json", help="put cell files together")
    ap.add_argument("--out", default="axi_provider_bounded.json"); a = ap.parse_args()
    if a.dry_run: dry_run()
    elif a.merge:
        m = merge_files(a.merge, a.out); old = load_second_run()["cells"]; print_cells(summarize_all(m["cells"])); print_verdict(verdict_bounded(m["cells"], old)); print("\nwritten to", a.out)
    else: ap.print_help()
