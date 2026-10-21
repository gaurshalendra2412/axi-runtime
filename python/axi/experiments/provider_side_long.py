"""Provider-side serving test, sixth run: jobs of 100 steps, the first long ones. Page with the predictions written BEFORE any run: docs/PROVIDER_SIDE_LONG_TEST.md (concept-map item 112).

Every job of every serving run so far was 30 steps. The claim that matters to the owner is about long tasks: a model that holds its own picture of a task loses it as the task goes on, and the exact state
kept by the host should not be lost. 30 steps is a short task. This test runs the same exact-state jobs for 100 steps (a job is the same script with more steps: its first 30 requests are the 30-step
job's), on 64 jobs that no run has seen (jobs 192 to 255), and asks four things:
  * does the exact-state arm still hold the right state after 100 steps, with the grammar that names the coordinates (`LN`) and without a grammar's help (`L0`, the capped expression of the third run)?
  * does the 100 percent of the fifth run (`FN`: 1,920 of 1,920 steps) come back on jobs nobody has run? That is read from steps 1 to 30 of `LN`, which are the 30-step jobs of the fifth run's kind;
  * does the prompt of the exact-state arm stay the same size while the job goes on, and does the held graph stay well formed?
  * what does a chat that keeps everything do over 100 steps on the same 8 jobs, with the same grammar (`LC` against `L8`)? Its prompt grows, the model's context is 8,192 tokens and a kept chat stops
    before its next prompt would pass 7,600 (the rule of `provider_side.py`).

What is the same (nothing below is changed; the files are imported and not edited, their sha256 are pinned on the page and in the test): the model, the engine class, the 80,000-token cache, the seeds
(job e has seed 20261017 x 1000 + e), the prompts of the earlier runs, the lockstep rounds, the policy repair_m6, the gate, the repair, the measures, the capped expression of the third run, the
`named` grammar of the fifth run, and the typed session of the fifth run (`provider_side_typed.TypedSession`, used here with 100 steps). What is new, and only this: the number of steps (100), a kept-chat
session that carries a grammar per request (`KeptChat`), the cells, the reading lines, the windows (steps 1 to 30, 1 to 33, 67 to 100), and a pre-flight check that can fail.

Honest limits: as the fifth run (one model, one small GPU, one engine and its version, synthetic graphs, greedy decoding, lockstep rounds, GPU seconds and not dollars, no energy), plus: the host reads the
request from the sentence the job was built from (a request in free words is another test); 100 steps is longer than 30 and is not a long task in the sense of days; a slip stays in the held state until a
later request removes it, so the share of exact steps after a slip is a consequence of the slip and not a new one; the kept chat is the plain kept chat (no summary, no window), and its stop is a property of
the context length of this setup (8,192 tokens), not of the model. `TypedStandIn` and `ScriptedStandIn` (models of a prefix cache that obey the grammar) test the harness only; their numbers are never a result."""
import hashlib, json, os, re, subprocess
from typing import Dict, List, Optional, Tuple

import axi.experiments.provider_side as PS
import axi.experiments.provider_side_typed as T
from axi.experiments.provider_side import _ge, _le

# ---------------------------------------------------------------- the settings written on the page
MODEL, SEED, POLICY, HINT_RULES, MAX_NEW, CAPACITY_TOKENS = T.MODEL, T.SEED, T.POLICY, T.HINT_RULES, T.MAX_NEW, T.CAPACITY_TOKENS
REGEX, REGEX_SHA256, NAME_CAP = T.REGEX, T.REGEX_SHA256, T.NAME_CAP
STEPS = 100
FIRST = 192
SEEN_JOBS = range(0, 192)                 # jobs 0 to 63 (third, fourth and fifth run), 64 to 127 (fourth run), 128 to 191 (fifth run)
SHORT_STEPS = T.STEPS                     # 30, the length of every job so far

# label: (jobs side by side, first job number, arm, grammar). Job e has seed SEED * 1000 + e. S = the exact state is shown at every step; KC = the whole chat is kept (cache on).
CELLS: Dict[str, Tuple[int, int, str, str]] = {
    "L0": (64, FIRST, "S", "capped"),
    "LN": (64, FIRST, "S", "named"),
    "L8": (8, FIRST, "S", "named"),
    "LC": (8, FIRST, "KC", "named"),
}
ORDER = ["L0", "LN", "L8", "LC"]

PINS = dict(provider_side="bfb4734e171d9244cd47d10402be46528a557095a8275d02b38481a0e2b0b1d3", constrained="03452148e28c9f31372f12cc11ef1808ac67308adff972fa19d4aeeef7c96f7d",
            bounded="7427a686d151c4294809aebaa9664d0b5fa420854817eb79e910372c3f57b6bc", gate="b6a50408788791f76bc4d4129dad84debb6e09c9c5e804673cdf04abec224ce7",
            diagnostics="255f3c11db989b645077d10706a187e2c913576538a826e6dbdb8892b014b4f2", agent_loop="d658afaa29e3bce7e1fb7f43d88281fc40ee68ca59864421a65284e9973516b6",
            cascade="b0533316bc094e80a1d68eddb7e137c7c176af9a342341288360c18d61f74fe7", long_horizon="5718435de2178de110ddbb59dd3bf32e1bec9f1fd8625a27119d789399fa20ac")
TYPED_MODULE_SHA256 = "c5f05d5840374fb77a1c67c6fd33f166b26342e93733bd9a31aa474eacbc1420"          # provider_side_typed.py, the module the fifth run's Result was read with
GRAMMAR_SHA = {"L0": REGEX_SHA256, "LN": "85d17bbffb556c21acf180b989fecf3b2c92ca419fce1516a675fb24dd9929af",
               "L8": "9603a4c10ec798e0d8098b00f02fd15831a79f4fcd58d8beafbce6463e1c3ddb", "LC": "9603a4c10ec798e0d8098b00f02fd15831a79f4fcd58d8beafbce6463e1c3ddb"}

T_REPEAT, T_HIGH, T_GAIN, T_KEPT_TOL, T_TIME, T_FLAT = 0.995, 0.99, 0.03, 0.01, 0.15, 1.3           # lines 8, 9 and 10, 11, 14, 16, 6
WIN_FIRST30, WIN_EARLY, WIN_LATE = (1, 30), (1, 33), (67, 100)
GPU_FREE_MIB = 500                                                                                  # the pre-flight wants less than this in use before the engine is loaded

LL_LINES = [
    ("instrument", "the engine reported how many prompt tokens were served from its cache, for every request of every cell that was run"),
    ("instrument", "the constraint held: of all the replies of all the cells that were run at least 99.5 percent were accepted by the strict parser, none used all 96 new tokens, and every reply is in the language of its own request"),
    ("guard", "the grammar changed only what it forbids: every job of LN writes the same text as the same job of L0 up to the job's first step whose L0 reply is outside the named language, and the reply at that step is inside it"),
    ("guard", "the cells used the expressions this test defines: each cell's recorded set of expressions has the sha256 recomputed from its jobs, and the oracle's own reply is inside the expression of every request"),
    ("guard", "every job of the exact-state cells (L0, LN, L8) ran all 100 steps; none ended early"),
    ("claim", "the exact-state prompt stays flat while the job goes on: in L0 the mean prompt of the last quarter of the steps (76 to 100) is at most 1.3 times that of the first quarter (1 to 25)"),
    ("claim", "the held graph was never malformed: no step of any cell that was run has a corrupted state"),
    ("claim", "the fifth run's 100 percent comes back on jobs nobody has run: on steps 1 to 30 of the 64 new jobs the exact share of LN is at least 99.5 percent"),
    ("claim", "over all 100 steps the exact share of LN is at least 99 percent"),
    ("claim", "over the last third of the steps (67 to 100) the exact share of LN is at least 99 percent"),
    ("claim", "over all 100 steps the exact share of LN is at least three points above that of L0"),
    ("claim", "the job without a grammar's help does lose ground as it goes on: the exact share of L0 over steps 67 to 100 is lower than over steps 1 to 33"),
    ("claim", "every kept-chat job (LC) stopped before its 100th step, because its next prompt would have passed 7,600 tokens"),
    ("claim", "on the steps it ran, the kept chat held the state no less exactly than the exact-state arm on the same 8 jobs: the exact share of LC is at least that of L8 minus one point"),
    ("claim", "the kept chat does fewer exact steps per GPU hour than the exact-state arm on the same 8 jobs: LC lower than L8 (both in this session)"),
    ("claim", "the seconds per step of LN are at most 15 percent above those of L0 (both in this session)"),
    ("claim", "the exact steps per GPU hour of LN are higher than those of L0 (both in this session)"),
]
LL_HEADLINE = (8, 9)


# ---------------------------------------------------------------- the sessions
class KeptChat(PS.Session):
    """The kept chat of the earlier runs (`provider_side.Session`, arm KC, unchanged: it adds, it overrides nothing that decides a step), asked with the grammar of its cell. Without it the kept chat would
    be asked under a different language than the exact-state cell it is compared with."""
    def __init__(self, seed: int, steps: int = STEPS, shape: str = "named"):
        assert shape in T.SHAPES, shape
        super().__init__("KC", seed, steps, HINT_RULES); self.shape = shape; self._rx = REGEX

    def request(self) -> T.TRequest:
        rq = super().request(); st = self.script[self.t]; self._rx = T.regex_for(self.shape, st["instruction"])
        assert re.fullmatch(self._rx, st["ideal"]), f"the expression of this request excludes the right answer: {st['instruction']!r} {st['ideal']!r}"
        return T.TRequest(rq.messages, rq.stand_in_reply, seed=self.seed, step=self.t + 1, prompt="old", shape=self.shape, regex=self._rx)

    def accept(self, rep: PS.Reply) -> None:
        super().accept(rep); self.log[-1]["in_language"] = re.fullmatch(self._rx, rep.text) is not None


def make_sessions(label: str, steps: int = STEPS, seed: int = SEED) -> list:
    K, first, arm, shape = CELLS[label]
    if arm == "S": return [T.TypedSession(seed * 1000 + e, steps, "old", shape) for e in range(first, first + K)]
    return [KeptChat(seed * 1000 + e, steps, shape) for e in range(first, first + K)]


def run_sessions(engine, sess: list, progress=None) -> dict:
    """The lockstep rounds of the earlier runs (`provider_side_typed.run_jobs`), for sessions that are handed in: a round for every unfinished job's request, then every job applies its reply."""
    engine.reset_cache(); rounds: List[dict] = []; wall = 0.0; seen: set = set()                   # every cell starts with a cold cache
    while True:
        active = [s for s in sess if not s.done]
        if not active: break
        rqs = [s.request() for s in active]; fresh = {getattr(r, "regex", REGEX) for r in rqs} - seen; seen |= fresh
        reps, secs = engine.generate_round(rqs, reset_cache=False); assert len(reps) == len(active)
        for s, r in zip(active, reps): s.accept(r)
        rounds.append(T._round(len(rounds) + 1, len(active), secs, reps, len(fresh))); wall += secs
        if progress: progress(len(rounds))
    return dict(wall_seconds=wall, rounds=rounds, distinct_regexes=len(seen), sessions=[dict(seed=s.seed, script_steps=len(s.script), stopped=s.stopped, log=s.log) for s in sess])


def run_cell(engine, label: str, steps: int = STEPS, seed: int = SEED, progress=None) -> dict:
    K, first, arm, shape = CELLS[label]
    r = run_sessions(engine, make_sessions(label, steps, seed), progress)
    return dict(label=label, arm=arm, K=K, first_episode=first, prompt="old", shape=shape, steps=steps, seed=seed, policy=POLICY, hint_rules=HINT_RULES, engine=engine.info(),
                grammar_sha256=T.grammar_sha256(shape, first, K, steps, seed), **r)


# ---------------------------------------------------------------- summaries
def window_share(c: dict, lo: int, hi: int) -> Optional[float]:
    """The share of the steps lo to hi (counted from 1, both included) of all jobs of a cell after which the held state was exactly the oracle's. None if no job has a step in the window."""
    xs = [x for s in c["sessions"] for x in s["log"] if lo <= x["t"] <= hi]
    return sum(1 for x in xs if x["exact"]) / len(xs) if xs else None


def first_slip_thirds(c: dict) -> List[int]:
    """How many jobs had their first inexact step in steps 1 to 33, 34 to 66 and 67 to 100."""
    out = [0, 0, 0]
    for s in c["sessions"]:
        x = next((y for y in s["log"] if not y["exact"]), None)
        if x is not None: out[0 if x["t"] <= 33 else 1 if x["t"] <= 66 else 2] += 1
    return out


def summarize_cell(c: dict) -> dict:
    s = T.summarize_cell4(c)
    s.update(label=c["label"], arm=c["arm"], first30=window_share(c, *WIN_FIRST30), early=window_share(c, *WIN_EARLY), late=window_share(c, *WIN_LATE), slips_by_third=first_slip_thirds(c),
             steps_per_job=[len(x["log"]) for x in c["sessions"]], stopped_jobs=sum(1 for x in c["sessions"] if x["stopped"] is not None))
    return s


def summarize_all(cells: List[dict]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for c in cells:
        assert c["label"] not in out, f"cell {c['label']} twice"; out[c["label"]] = summarize_cell(c)
    return out


# ---------------------------------------------------------------- merge
def fingerprint() -> str:
    """sha256 of this module's source, recorded in the merged file so a result says which code made it."""
    return hashlib.sha256(open(__file__, "rb").read()).hexdigest()


def cells_planned() -> List[str]:
    """In the order they are run: the control and the grammar on 64 jobs first (the pair that carries the headline), then the two cells of 8 jobs. About 35 minutes of GPU."""
    return list(ORDER)


def merge_cells(cells: List[dict]) -> dict:
    """Puts cell files together. Refuses cells that were not run under the same settings, a cell twice, a cell that is not one of this test's, a cell whose settings are not those of its label,
    a cell of another length, a cell that carries another base constraint, and the stand-in mixed with a real engine."""
    assert cells, "nothing to merge"
    ref = cells[0]
    key = lambda c: (c["steps"], c["seed"], c["policy"], c["hint_rules"], c["engine"]["kind"], c["engine"].get("model"), c["engine"].get("capacity_tokens"), c["engine"].get("version"), c["engine"].get("gpu"),
                     (c["engine"].get("constraint") or {}).get("regex_sha256"))
    seen = set()
    for c in cells:
        lab = c.get("label")
        if lab not in CELLS: raise ValueError(f"cell {lab!r} is not one of this test's cells")
        if (c["K"], c["first_episode"], c["arm"], c["shape"]) != CELLS[lab]: raise ValueError(f"cell {lab} was not run with the settings of its label: {(c['K'], c['first_episode'], c['arm'], c['shape'])} against {CELLS[lab]}")
        if c["steps"] != STEPS: raise ValueError(f"cell {lab} has {c['steps']} steps, this test has {STEPS}")
        if (c["engine"].get("constraint") or {}).get("regex_sha256") != REGEX_SHA256: raise ValueError(f"cell {lab} was not run with this test's constraint")
        if key(c) != key(ref): raise ValueError(f"cell {lab} was run under other settings: {key(c)} against {key(ref)}")
        if lab in seen: raise ValueError(f"cell {lab} twice")
        seen.add(lab)
    return dict(meta=dict(model=ref["engine"].get("model"), engine=ref["engine"], steps=ref["steps"], seed=ref["seed"], policy=ref["policy"], hint_rules=ref["hint_rules"], name_cap=NAME_CAP,
                          cells={k: list(CELLS[k]) for k in ORDER}), cells=sorted(cells, key=lambda c: ORDER.index(c["label"])))


def merge_files(paths: List[str], out: str) -> dict:
    cells = [json.load(open(p)) for p in paths]; m = merge_cells(cells)
    m["module_sha256"] = fingerprint(); m["typed_module_sha256"] = T.fingerprint(); m["pinned_sha256"] = T.pinned_hashes()
    with open(out, "w") as f: json.dump(m, f)
    return m


def save_cell(cell: dict, path: str) -> str:
    with open(path, "w") as f: json.dump(cell, f)
    return path


# ---------------------------------------------------------------- reading rules (fixed in docs/PROVIDER_SIDE_LONG_TEST.md before the run)
def verdict_long(cells: List[dict]) -> List[Tuple[int, str, str, Optional[bool]]]:
    """(number, tag, text, yes / no / None = not run). A line that needs a cell that was not run is None, never 'no'."""
    new = {c["label"]: c for c in cells if any(s["log"] for s in c["sessions"])}; Sn = {k: summarize_cell(c) for k, c in new.items()}
    has = lambda *ks: all(k in Sn for k in ks)
    out: List[Optional[bool]] = []
    allrep = [r for k in new for r in T.replies_of(new[k])]
    out.append(all(r[2] is not None for r in allrep) if allrep else None)                                                                                                        # 1
    out.append((sum(1 for r in allrep if not r[0]) <= (1 - T.T_PARSED) * len(allrep) + 1e-9 and all(r[1] < MAX_NEW for r in allrep) and all(r[3] for r in allrep)) if allrep else None)  # 2
    out.append(None if not has("L0", "LN") else T.same_until_forced(new["LN"], new["L0"], "named"))                                                                              # 3
    out.append(all(T.cell_grammar_ok(c) and c["grammar_sha256"] == GRAMMAR_SHA[k] for k, c in new.items()) if new else None)                                                   # 4
    s_cells = [k for k in ("L0", "LN", "L8") if k in new]
    out.append(all(len(s["log"]) == STEPS and s["stopped"] is None for k in s_cells for s in new[k]["sessions"]) if s_cells else None)                                           # 5
    out.append(None if not has("L0") else _le(Sn["L0"]["prompt_last_quarter"], Sn["L0"]["prompt_first_quarter"] * T_FLAT))                                                      # 6
    out.append(all(Sn[k]["corrupted_steps"] == 0 for k in Sn) if Sn else None)                                                                                                  # 7
    out.append(None if not has("LN") or Sn["LN"]["first30"] is None else _ge(Sn["LN"]["first30"], T_REPEAT))                                                                    # 8
    out.append(None if not has("LN") else _ge(Sn["LN"]["exact_share"], T_HIGH))                                                                                                  # 9
    out.append(None if not has("LN") or Sn["LN"]["late"] is None else _ge(Sn["LN"]["late"], T_HIGH))                                                                            # 10
    out.append(None if not has("L0", "LN") else _ge(Sn["LN"]["exact_share"], Sn["L0"]["exact_share"] + T_GAIN))                                                                  # 11
    out.append(None if not has("L0") or Sn["L0"]["late"] is None or Sn["L0"]["early"] is None else Sn["L0"]["late"] < Sn["L0"]["early"] - 1e-12)                                # 12
    out.append(None if not has("LC") else all(s["stopped"] == "context" and len(s["log"]) < STEPS for s in new["LC"]["sessions"]))                                                # 13
    out.append(None if not has("L8", "LC") else _ge(Sn["LC"]["exact_share"], Sn["L8"]["exact_share"] - T_KEPT_TOL))                                                               # 14
    out.append(None if not has("L8", "LC") else Sn["LC"]["exact_per_hour"] < Sn["L8"]["exact_per_hour"])                                                                         # 15
    out.append(None if not has("L0", "LN") else _le(Sn["LN"]["seconds_per_step"], Sn["L0"]["seconds_per_step"] * (1 + T_TIME)))                                                  # 16
    out.append(None if not has("L0", "LN") else Sn["LN"]["exact_per_hour"] > Sn["L0"]["exact_per_hour"])                                                                         # 17
    assert len(out) == len(LL_LINES)
    return [(i + 1, LL_LINES[i][0], LL_LINES[i][1], out[i]) for i in range(len(LL_LINES))]


def claims_allowed(checks) -> dict:
    """What the page lets the Result claim, from the lines: with lines 2, 3, 4 and 5 yes the figures of the cells are claimed as read. That the 100 percent comes back is claimed from line 8, what happens over
    100 steps from lines 9 to 12, what the kept chat does from lines 13 to 15, a statement about time or cost only from lines 16 and 17 (cells of one session). A line that was not run claims nothing."""
    ok = {i: c for i, _, _, c in checks}; base = all(ok[i] is True for i in (2, 3, 4, 5))
    pick = lambda ks: ({i: ok[i] for i in ks} if base else {i: None for i in ks})
    return dict(base=base, constraint_held=ok[2], grammar_changed_only_what_it_forbids=ok[3], expressions_as_defined=ok[4], jobs_ran_all_steps=ok[5], repeat_of_the_fifth_run=pick((8,)),
                over_100_steps=pick((9, 10, 11, 12)), prompt_and_graph=pick((6, 7)), kept_chat=pick((13, 14, 15)), time_and_cost_in_one_session=pick((16, 17)))


# ---------------------------------------------------------------- printing
def print_cells(S: Dict[str, dict]) -> None:
    f = lambda x, k=1: "-" if x is None else f"{x:.{k}f}"; pc = lambda x: "-" if x is None else f"{x * 100:.1f}%"
    print(f"\n{'cell':4s} {'K':>3s} {'jobs':>9s} {'arm':>3s} {'grammar':>7s} | {'steps':>6s} {'seconds':>8s} {'s/step':>7s} {'comp/step':>9s} | {'exact':>6s} {'1-30':>6s} {'67-100':>7s} {'jobs ok':>7s} {'corrupt':>7s} | {'stopped':>7s} {'outside':>7s} {'exact/h':>8s}")
    for lab in ORDER:
        if lab not in S: continue
        s = S[lab]; K, first, arm, shape = CELLS[lab]
        print(f"{lab:4s} {K:3d} {str(first) + '-' + str(first + K - 1):>9s} {arm:>3s} {shape:>7s} | {s['steps_done']:6d} {f(s['wall_seconds'], 0):>8s} {f(s['seconds_per_step'], 3):>7s} {f(s['computed_per_step'], 0):>9s} | "
              f"{pc(s['exact_share']):>6s} {pc(s['first30']):>6s} {pc(s['late']):>7s} {s['jobs_exact_to_end']:7d} {s['corrupted_steps']:7d} | {s['stopped_jobs']:7d} {s['out_of_language']:7d} {f(s['exact_per_hour'], 0):>8s}")


def print_windows(S: Dict[str, dict]) -> None:
    """Where in the job the first slips are, and how long the prompts are at the start and at the end (descriptive)."""
    for lab in ORDER:
        if lab not in S: continue
        s = S[lab]; steps = s["steps_per_job"]
        print(f"{lab}: first slips of jobs in steps 1-33 / 34-66 / 67-100: {s['slips_by_third'][0]} / {s['slips_by_third'][1]} / {s['slips_by_third'][2]};  prompt tokens first quarter {s['prompt_first_quarter']:.0f}, last quarter {s['prompt_last_quarter']:.0f}, longest {s['longest_prompt']};  steps per job {min(steps)} to {max(steps)}")


def print_misses(S: Dict[str, dict]) -> None:
    """The first miss of every job that missed, per cell (descriptive)."""
    for lab in ORDER:
        if lab in S and S[lab]["first_misses"]:
            print(f"first misses in {lab}: " + "; ".join(f"job {j} step {t} ({k}) {x!r}" for j, t, k, x in S[lab]["first_misses"]))


def print_verdict(checks, title="provider-side test, jobs of 100 steps") -> None:
    print(f"\n-- {title} (reading rules fixed before the run) --")
    for i, tag, text, ok in checks: print(("  yes      " if ok else "  NO       " if ok is False else "  not run  ") + f"{i:2d}. [{tag}] {text}")
    ran = [ok for _, _, _, ok in checks if ok is not None]
    print(f"  ({len(ran)} of {len(checks)} lines could be read; headline lines {LL_HEADLINE[0]} and {LL_HEADLINE[1]}: " + ", ".join("yes" if checks[i - 1][3] else "NO" if checks[i - 1][3] is False else "not run" for i in LL_HEADLINE) + ")")
    a = claims_allowed(checks)
    if not a["base"]: print("  rule on the page: line 2, 3, 4 or 5 did not read yes, so NO claim is made from this run until the cause is found.")
    else:
        print("  rule on the page: lines 2, 3, 4 and 5 read yes, so the figures of the cells are claimed as read. A statement about time or cost comes only from lines 16 and 17 (cells of one session).")
        print("  never: a multiple of capacity, dollars, '100 percent' for a grammar, a statement about any provider, anything about requests in free words, anything about a chat with a summary or a window, anything about a model that was not run.")


# ---------------------------------------------------------------- the pre-flight check: it can fail
def gpu_state(run=None) -> Optional[dict]:
    """The GPU as `nvidia-smi` sees it (all processes), or None if it cannot be read. `torch` is not used: starting CUDA in the notebook before the engine would be a change to the run."""
    run = run or subprocess.run
    try: p = run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
    except Exception: return None
    if getattr(p, "returncode", 1) != 0 or not (p.stdout or "").strip(): return None
    try: name, used, total = [x.strip() for x in p.stdout.strip().splitlines()[0].split(",")]; return dict(name=name, used_mib=int(float(used)), total_mib=int(float(total)))
    except ValueError: return None


def preflight(gpu=None, steps: int = STEPS, seed: int = SEED) -> List[Tuple[str, bool, str]]:
    """Checks that cost seconds and save a Colab run: (name, ok, what was found). Every check can read NO: the jobs, the files, the expressions, a real first request of each cell, the GPU. A check that
    cannot be made (an error while it runs) reads NO with the error, and the others are still made."""
    out: List[Tuple[str, bool, str]] = []
    def one(name, fn):
        try: ok, detail = fn()
        except Exception as ex: ok, detail = False, f"could not be checked: {type(ex).__name__}: {ex}"
        out.append((name, bool(ok), detail))
    jobs = sorted({e for K, first, _, _ in CELLS.values() for e in range(first, first + K)}); clash = [e for e in jobs if e in SEEN_JOBS]
    one("the jobs are new", lambda: (not clash and jobs == list(range(FIRST, FIRST + 64)), f"jobs {jobs[0]} to {jobs[-1]} ({len(jobs)}); jobs of earlier runs among them: {clash if clash else 'none'}"))
    def steps_and_prefix():
        lens = {e: len(T.script_of(seed * 1000 + e, steps)) for e in jobs}                       # compared with 100, the page's number, and not with `steps`: a script of 99 steps reads NO even if 99 was asked for
        pref = all([x["instruction"] for x in T.script_of(seed * 1000 + e, steps)[:SHORT_STEPS]] == [x["instruction"] for x in T.script_of(seed * 1000 + e, SHORT_STEPS)] for e in jobs)
        return all(n == 100 for n in lens.values()) and pref, f"steps per job {min(lens.values())} to {max(lens.values())}; first 30 requests equal the 30-step job's: {pref}"
    one("every job has 100 steps and starts with its 30-step job", steps_and_prefix)
    def files():
        pins = T.pinned_hashes(); bad = sorted(k for k, v in PINS.items() if pins.get(k) != v); mod_ok = T.fingerprint() == TYPED_MODULE_SHA256 and T.REGEX_SHA256 == REGEX_SHA256
        return not bad and mod_ok, f"changed: {bad if bad else 'none'}; provider_side_typed.py as pinned: {T.fingerprint() == TYPED_MODULE_SHA256}; capped expression as pinned: {T.REGEX_SHA256 == REGEX_SHA256}"
    one("the files this test imports are the pinned ones", files)
    def expressions():
        shas = {lab: T.grammar_sha256(shape, first, K, steps, seed) for lab, (K, first, _, shape) in CELLS.items()}; badg = sorted(l for l in CELLS if shas[l] != GRAMMAR_SHA[l])
        return not badg, "; ".join(f"{l} {shas[l][:8]}" for l in ORDER) + (f"; wrong: {badg}" if badg else "")
    one("the expressions are the ones on the page", expressions)
    def oracle():
        n_req = 0; outside = 0
        for lab, (K, first, _, shape) in CELLS.items():
            for e in range(first, first + K):
                for st in T.script_of(seed * 1000 + e, steps): n_req += 1; outside += 0 if T.in_language(shape, st["instruction"], st["ideal"]) else 1
        return outside == 0, f"{n_req - outside} of {n_req} requests"
    one("the oracle's reply is inside the expression of every request", oracle)
    def first_request(lab):
        s = make_sessions(lab, steps, seed)[0]; rq = s.request(); user = rq.messages[-1]["content"]
        ok = s.script[0]["instruction"] in user and re.fullmatch(rq.regex, s.script[0]["ideal"]) is not None and len(rq.messages) == 2 and rq.messages[0]["role"] == "system"
        return ok, f"user message {len(user)} characters, expression {len(rq.regex)} characters, job {s.seed - seed * 1000}"
    for lab in ORDER: one(f"cell {lab}: its first request is built", lambda lab=lab: first_request(lab))
    def card():
        g = gpu_state() if gpu is None else gpu()
        if g is None: return False, "nvidia-smi could not be read (is the runtime a GPU runtime?)"
        return "T4" in g["name"] and g["used_mib"] <= GPU_FREE_MIB, f"{g['name']}, {g['used_mib']} of {g['total_mib']} MiB in use (at most {GPU_FREE_MIB} before the engine is loaded)"
    one("the GPU is a free T4", card)
    return out


def print_preflight(res) -> bool:
    print("-- pre-flight --")
    for name, ok, detail in res: print(("  ok    " if ok else "  FAIL  ") + f"{name}: {detail}")
    good = all(ok for _, ok, _ in res); print("  " + ("GREEN: nothing found that would waste the run. Go on to the next cell." if good else "RED: do not run the next cells. Send me this printout.")); return good


# ---------------------------------------------------------------- the stand-in run (harness only)
def dry_run(steps: int = STEPS, labels=None) -> Dict[str, dict]:
    """Every planned cell on the stand-in (a model of a block cache that answers with the oracle's own reply). Prints the table. NOT a result."""
    eng = T.TypedStandIn(capacity_tokens=int(CAPACITY_TOKENS * PS.PROXY_SCALE)); cells = []
    for lab in ORDER:
        if labels is None or lab in labels: cells.append(run_cell(eng, lab, steps))
    S = summarize_all(cells); print("STAND-IN (a model of a block cache; it answers with the oracle's own reply), NOT a result"); print_cells(S); return S


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="provider-side test, jobs of 100 steps (docs/PROVIDER_SIDE_LONG_TEST.md)")
    ap.add_argument("--dry-run", action="store_true", help="run every cell on the stand-in engine (no GPU); prints, does not save")
    ap.add_argument("--preflight", action="store_true", help="run the pre-flight check (reads nvidia-smi)")
    ap.add_argument("--merge", nargs="+", metavar="CELL.json", help="put cell files together")
    ap.add_argument("--out", default="axi_provider_long.json"); a = ap.parse_args()
    if a.dry_run: dry_run()
    elif a.preflight: raise SystemExit(0 if print_preflight(preflight()) else 1)
    elif a.merge:
        m = merge_files(a.merge, a.out); S = summarize_all(m["cells"]); print_cells(S); print_windows(S); print_misses(S); print_verdict(verdict_long(m["cells"])); print("\nwritten to", a.out)
    else: ap.print_help()
