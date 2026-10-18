"""Provider-side serving test, second run: every reply of every arm is constrained by the repository's grammar. Page with the predictions written BEFORE any run:
docs/PROVIDER_SIDE_CONSTRAINED_TEST.md (concept-map item 107). The first run (docs/PROVIDER_SIDE_TEST.md, its Result) left replies unconstrained and the exact-state arm held the right
state on 69 to 78 percent of its steps against 97 to 100 for the kept chat; 82 of the 89 replies the strict parser refused were the same slip, `DEL[r,c:v]` (a value after a delete).

What is the same as in the first run (nothing below is changed, `provider_side.py` is imported and not edited: its sha256 is pinned in the first run's merged file): the model, the engine
class, the 80,000-token cache, the 30-step jobs and their seeds, the lockstep rounds, the policy repair_m6, the measures, the arms `S` and `KC`.
What is new:
  * the reply must match a regular expression that accepts exactly the delta language of `axi.engine.cico_parser` (at least one item; items separated by one space or newline),
    handed to the engine as structured output; it is the same for both arms;
  * a cell can be checkpointed after every round and carried on in a later Colab session (a cell of three hours can then be finished over several sessions; the first round after a
    resume starts with a cold cache, and the cell records where it was resumed);
  * the planned cells are S and KC at 8, 32 and 64 jobs, cheap cells first (the arm with the cache emptied each round and the random-order arm are not repeated);
  * the reading lines are 17, with a quality-adjusted measure (exact steps per GPU hour) as the headline, because the saved data of the first run say that exact-state will not reach the
    kept chat's exactness at 8 jobs even with the replies constrained (the page has the reasoning).

Honest limits: as the first run (one model, one small GPU, one engine and its version, synthetic graphs, greedy decoding, lockstep rounds, GPU seconds and not dollars, no energy), plus:
the grammar cannot make a reply right, only well formed; a reply that is well formed and wrong stays wrong. `StandInEngine` (a model of a prefix cache) is used to test the harness only; its
numbers are never a result."""
import hashlib, json, os, platform
from typing import Dict, List, Optional, Tuple

import axi.experiments.provider_side as PS
from axi.experiments.provider_side import Reply, Request, Session, StandInEngine, VLLMEngine, _ge, _le, _lt, _mean, _ratio, summarize_cell

# ---------------------------------------------------------------- the settings written on the page
ARMS = ("S", "KC")
ARM_TEXT = {"S": "exact state each step, cache on", "KC": "whole chat kept, cache on"}
LEVELS = (8, 32, 64)
MODEL, STEPS, SEED, POLICY, HINT_RULES, MAX_NEW = PS.MODEL, PS.STEPS, PS.SEED, PS.POLICY, PS.HINT_RULES, PS.MAX_NEW
CAPACITY_TOKENS = PS.CAPACITY_TOKENS

# The delta language of axi.engine.cico_parser (ADD[r,c:v], DEL[r,c], ADD/DEL[(r,c)->(r,c):relation#weight]); at least one item, items separated by exactly one space or newline, one optional
# separator at the end. The empty reply is not allowed (the strict parser reads an empty reply as "no reply").
_ITEM = r"(?:ADD\[[0-9]+,[0-9]+:-?[0-9]+\]|DEL\[[0-9]+,[0-9]+\]|(?:ADD|DEL)\[\([0-9]+,[0-9]+\)->\([0-9]+,[0-9]+\):[A-Za-z0-9_]+#-?[0-9]+\])"
REGEX = _ITEM + r"(?:[ \n]" + _ITEM + r")*[ \n]?"
REGEX_SHA256 = hashlib.sha256(REGEX.encode()).hexdigest()

C_PARSED = 0.995                                  # line 2: at least this share of the replies is accepted by the strict parser
C_MIN_EXACT, C_EXACT_GAP, C_EXACT_HIGH = 0.80, 0.05, 0.98
C_FLAT, C_GROWTH, C_CACHE_FITS = PS.PL_FLAT, PS.PL_GROWTH, PS.PL_CACHE_FITS
C_LIGHT_TOKENS, C_LIGHT_TIME, C_OVERRUN, C_PRESSURE_RATIO, C_MID_RATIO = PS.PL_LIGHT_TOKENS, PS.PL_LIGHT_TIME, PS.PL_OVERRUN, PS.PL_PRESSURE_RATIO, PS.PL_MID_RATIO
C_QA_LIGHT, C_QA_MID, C_QA_HEAVY = 2.0, 10.0, 10.0   # lines 15 to 17: exact steps per GPU hour of exact-state against the kept chat, at 8, 32 and 64 jobs

CL_LINES = [
    ("instrument", "the engine reported how many prompt tokens were served from its cache, for every request of every cell that was run"),
    ("instrument", "the constraint held: in every cell that was run at least 99.5 percent of the replies were accepted by the strict parser, and no reply used all 96 new tokens"),
    ("instrument", "the setup made the pressure it was meant to: 64 times the longest kept-chat prompt at K = 64 is at least twice the cache size, and 8 times the longest at K = 8 fits in it"),
    ("guard", "in every cell the held state was exactly the oracle's state on at least 80 percent of the steps"),
    ("claim", "exact-state was not less exact than the kept chat: at every K its share of exact steps is at least the kept chat's minus 5 points"),
    ("claim", "with the replies constrained, exact-state held the right state on at least 98 percent of the steps in every one of its cells"),
    ("claim", "exact-state prompts stay flat: the mean prompt of the last quarter of the steps is at most 1.3 times that of the first quarter (K = 8)"),
    ("claim", "kept-chat prompts grow: the mean prompt of the last quarter is at least 3.5 times that of the first quarter (K = 8)"),
    ("claim", "at K = 8 the cache works for the kept chat: at least 80 percent of its prompt tokens came from the cache"),
    ("claim", "at K = 8 with the cache on, exact-state computes at least 30 percent fewer prompt tokens than the kept chat"),
    ("claim", "at K = 8 with the cache on, exact-state takes at least 20 percent less GPU time per step than the kept chat"),
    ("claim", "at K = 32 the kept chat takes at least 1.3 times as much GPU time per step as exact-state"),
    ("claim", "at K = 64 the kept chat overruns the cache: in the second half of the steps less than 50 percent of its prompt tokens came from the cache"),
    ("claim", "at K = 64 the kept chat takes at least twice as much GPU time per step as exact-state"),
    ("claim", "at K = 8 exact-state does at least 2 times as many exact steps per GPU hour as the kept chat"),
    ("claim", "at K = 32 exact-state does at least 10 times as many exact steps per GPU hour as the kept chat"),
    ("claim", "at K = 64 exact-state does at least 10 times as many exact steps per GPU hour as the kept chat"),
]
CL_HEADLINE = (15, 17)
CL_PLAIN_TIME_LINES = (11, 12, 14)                # read as a plain time ratio only when line 5 reads yes; otherwise the claim rests on lines 15 to 17


# ---------------------------------------------------------------- the engines
def constraint_info(kind: str) -> dict:
    return dict(kind=kind, api="structured_outputs(regex)", regex=REGEX, regex_sha256=REGEX_SHA256)


def _structured_params(regex: str):
    """vLLM's structured-output parameter object. There is deliberately no fallback to replies that are not constrained: a run that would silently drop the constraint is not this test."""
    try: from vllm.sampling_params import StructuredOutputsParams
    except ImportError:
        try: from vllm import StructuredOutputsParams
        except ImportError as e:
            raise RuntimeError("this vLLM version has no StructuredOutputsParams (it came in with 0.12); install vllm==0.31.0 as the notebook does. The test does not fall back to replies that are not constrained.") from e
    return StructuredOutputsParams(regex=regex)


class ConstrainedVLLMEngine(VLLMEngine):
    """The first run's engine, with the reply constrained by REGEX. Not testable without a GPU; tests/test_provider_side_constrained.py runs it against a fake `vllm` module, and the
    notebook's safety cell shows within minutes whether the installed version takes the constraint (it feeds the prompt of a reply that was refused in the first run)."""
    def __init__(self, *a, **kw):
        params = _structured_params(REGEX)                                   # before the model is loaded: a version that cannot constrain fails in a second, not after ten minutes
        super().__init__(*a, **kw)
        from vllm import SamplingParams
        self.sp = SamplingParams(temperature=0.0, max_tokens=MAX_NEW, structured_outputs=params)

    def info(self) -> dict:
        d = super().info(); d["constraint"] = constraint_info("regex"); return d


class ConstrainedStandIn(StandInEngine):
    """The stand-in with the constraint recorded (it replies with the oracle's own reply, which is in the language). Tests the harness only. Never a result."""
    def info(self) -> dict:
        d = super().info(); d["constraint"] = constraint_info("stand_in"); return d


# ---------------------------------------------------------------- planned cells
def cells_planned() -> List[Tuple[str, int]]:
    """In the order they are run: the cheap cells first, then the kept chat at 32 jobs (about an hour and a half), then the kept chat at 64 jobs (three hours or more)."""
    return [("S", 8), ("KC", 8), ("S", 32), ("S", 64), ("KC", 32), ("KC", 64)]


# ---------------------------------------------------------------- one cell, with a checkpoint after every round
CK_FORMAT = 1


def _engine_key(info: dict) -> dict:
    return {k: info.get(k) for k in ("kind", "model", "version", "gpu", "capacity_tokens")} | {"constraint": (info.get("constraint") or {}).get("regex_sha256")}


def _header(arm: str, K: int, steps: int, seed: int, info: dict) -> dict:
    return dict(format=CK_FORMAT, arm=arm, K=K, steps=steps, seed=seed, policy=POLICY, hint_rules=HINT_RULES, engine=_engine_key(info))


def _save_checkpoint(path: str, header: dict, rounds: List[dict], sess: List[Session], resumed: List[int]) -> None:
    data = dict(header, rounds=rounds, resumed_at=resumed, replies=[[[x["text"], x["n_prompt"], x["n_cached"], x["n_new"]] for x in s.log] for s in sess])
    tmp = path + ".tmp"
    with open(tmp, "w") as f: json.dump(data, f)
    os.replace(tmp, path)                                                    # a session that dies while writing leaves the last good file


def _replay(sess: List[Session], replies: List[list], rounds: List[dict]) -> None:
    """Brings the jobs back to where they were by feeding them their own saved replies again (a job is a function of its script and its replies). Refuses a checkpoint that does not add up."""
    if len(replies) != len(sess): raise ValueError(f"the checkpoint has {len(replies)} jobs, this cell has {len(sess)}")
    for s, rec in zip(sess, replies):
        for text, n_prompt, n_cached, n_new in rec:
            if s.done: raise ValueError("the checkpoint has a reply for a job that was already finished")
            s.request(); s.accept(Reply(text, n_prompt, n_cached, n_new))
    for j, r in enumerate(rounds):
        if r["active"] != sum(1 for rec in replies if len(rec) > j): raise ValueError(f"the checkpoint does not add up at round {j + 1}")
    if rounds and max(len(rec) for rec in replies) != len(rounds): raise ValueError("the checkpoint's rounds and replies disagree")


def run_cell(engine, arm: str, K: int, steps: int = STEPS, seed: int = SEED, progress=None, checkpoint: Optional[str] = None) -> dict:
    """The first run's `run_cell` (same jobs, same rounds, same cold start), plus a checkpoint: after every round the replies so far are written to `checkpoint`; if that file is there when the
    cell starts, the cell carries on from it. The seconds of a resumed cell are the sum of its rounds (the time between sessions is not counted), and `resumed_at` lists the rounds at which a
    session was resumed; the first round after a resume starts with a cold cache."""
    assert arm in ARMS and K >= 1, (arm, K)
    info = engine.info(); header = _header(arm, K, steps, seed, info)
    sess = [Session(arm, seed * 1000 + e, steps) for e in range(K)]
    rounds: List[dict] = []; resumed: List[int] = []
    if checkpoint and os.path.exists(checkpoint):
        ck = json.load(open(checkpoint))
        if {k: ck.get(k) for k in header} != header: raise ValueError(f"the checkpoint {checkpoint} belongs to another cell or other settings: {ck.get('arm')}/{ck.get('K')} {ck.get('engine')} against {header}")
        rounds = list(ck["rounds"]); resumed = list(ck["resumed_at"]) + [len(rounds)]
        _replay(sess, ck["replies"], rounds)
    wall = sum(r["seconds"] for r in rounds)
    engine.reset_cache()                                                     # every cell, and every resumed session, starts with a cold cache
    while True:
        active = [s for s in sess if not s.done]
        if not active: break
        reps, secs = engine.generate_round([s.request() for s in active], reset_cache=False)
        assert len(reps) == len(active)
        for s, r in zip(active, reps): s.accept(r)
        cached = None if any(r.n_cached is None for r in reps) else sum(r.n_cached for r in reps)
        rounds.append(dict(round=len(rounds) + 1, active=len(active), seconds=secs, prompt_tokens=sum(r.n_prompt for r in reps), cached_tokens=cached, new_tokens=sum(r.n_new for r in reps)))
        wall += secs
        if checkpoint: _save_checkpoint(checkpoint, header, rounds, sess, resumed)
        if progress: progress(len(rounds), arm, K)
    return dict(arm=arm, K=K, steps=steps, seed=seed, policy=POLICY, hint_rules=HINT_RULES, engine=info, wall_seconds=wall, rounds=rounds, resumed_at=resumed,
                sessions=[dict(seed=s.seed, script_steps=len(s.script), stopped=s.stopped, log=s.log) for s in sess])


# ---------------------------------------------------------------- the probe: the prompt of a reply the first run refused
def slip_probe(engine, episode: int = 0, step: int = 20, seed: int = SEED) -> dict:
    """One request: the exact-state prompt of step `step` of job `episode`. In the first run job 0 answered `DEL[4,5:8]` at step 20 (a value after a delete) and the strict parser refused it.
    That job held the right state up to step 19, so feeding it the oracle's own replies for steps 1 to 19 rebuilds the prompt the first run's job saw. Returns the oracle's reply and what the
    engine said."""
    s = Session("S", seed * 1000 + episode, STEPS)
    if not 1 <= step <= len(s.script): raise ValueError(f"step {step} is not in this job")
    while s.t < step - 1:
        rq = s.request(); s.accept(Reply(rq.stand_in_reply, 0, 0, 0))
    rq = s.request(); reps, secs = engine.generate_round([rq]); text = reps[0].text
    return dict(episode=episode, step=step, kind=s.script[s.t]["kind"], oracle=rq.stand_in_reply, reply=text, parsed=PS.strict_parse(text) is not None, n_new=reps[0].n_new, seconds=secs)


# ---------------------------------------------------------------- summaries
def summarize_cell2(c: dict) -> dict:
    s = summarize_cell(c); steps = [x for ss in c["sessions"] for x in ss["log"]]; n = len(steps)
    s["unparsed"] = sum(1 for x in steps if not x["parsed"]); s["truncated"] = sum(1 for x in steps if x["n_new"] >= MAX_NEW)
    s["exact_steps"] = sum(1 for x in steps if x["exact"]); s["exact_per_hour"] = (3600.0 * s["exact_steps"] / c["wall_seconds"]) if c["wall_seconds"] else None
    s["resumed_at"] = list(c.get("resumed_at", [])); s["constraint"] = (c["engine"].get("constraint") or {}).get("regex_sha256")
    return s


def summarize_all(cells: List[dict]) -> Dict[Tuple[str, int], dict]:
    out: Dict[Tuple[str, int], dict] = {}
    for c in cells:
        k = (c["arm"], c["K"]); assert k not in out, f"cell {k} twice"; out[k] = summarize_cell2(c)
    return out


def merge_cells(cells: List[dict]) -> dict:
    """Puts cell files together. Refuses cells that were not run under the same settings (the constraint included), a cell twice, cells that carry no constraint, and the stand-in mixed with a real engine."""
    assert cells, "nothing to merge"
    ref = cells[0]
    key = lambda c: (c["steps"], c["seed"], c["policy"], c["hint_rules"], c["engine"]["kind"], c["engine"].get("model"), c["engine"].get("capacity_tokens"), c["engine"].get("version"), c["engine"].get("gpu"),
                     (c["engine"].get("constraint") or {}).get("regex_sha256"))
    seen = set()
    for c in cells:
        if (c["engine"].get("constraint") or {}).get("regex_sha256") != REGEX_SHA256: raise ValueError(f"cell {c['arm']}/K={c['K']} was not run with this test's constraint")
        if key(c) != key(ref): raise ValueError(f"cell {c['arm']}/K={c['K']} was run under other settings: {key(c)} against {key(ref)}")
        if (c["arm"], c["K"]) in seen: raise ValueError(f"cell {c['arm']}/K={c['K']} twice")
        seen.add((c["arm"], c["K"]))
    order = {k: i for i, k in enumerate(cells_planned())}
    return dict(meta=dict(model=ref["engine"].get("model"), engine=ref["engine"], steps=ref["steps"], seed=ref["seed"], policy=ref["policy"], hint_rules=ref["hint_rules"], levels=list(LEVELS), arms=list(ARMS)),
                cells=sorted(cells, key=lambda c: (order.get((c["arm"], c["K"]), 99), c["K"], ARMS.index(c["arm"]))))


# ---------------------------------------------------------------- reading rules (fixed in docs/PROVIDER_SIDE_CONSTRAINED_TEST.md before the run)
def _rate_ratio(a, b):
    """a / b for two rates; a kept chat with no exact step at all gives an infinite ratio if exact-state had some."""
    if a is None or b is None: return None
    if b == 0: return float("inf") if a > 0 else None
    return a / b


def verdict_constrained(S: Dict[Tuple[str, int], dict]) -> List[Tuple[int, str, str, Optional[bool]]]:
    """(number, tag, text, yes / no / None = not run). `S` is summarize_all's table. A line that needs a cell that was not run is None, never 'no'."""
    S = {k: v for k, v in S.items() if v["steps_done"] > 0}
    g = lambda arm, k: S.get((arm, k))
    kc8, kc32, kc64, s8, s32, s64 = g("KC", 8), g("KC", 32), g("KC", 64), g("S", 8), g("S", 32), g("S", 64)
    out: List[Optional[bool]] = []
    cached_all = [s["cached_tokens"] is not None for s in S.values()]; out.append(all(cached_all) if cached_all else None)                                                           # 1
    out.append(all(_le(s["unparsed"] / s["steps_done"], 1 - C_PARSED) and s["truncated"] == 0 for s in S.values()) if S else None)                                                     # 2
    out.append(None if kc8 is None or kc64 is None or not kc8["capacity_tokens"] or not kc64["capacity_tokens"] else
               (64 * kc64["longest_prompt"] >= 2 * kc64["capacity_tokens"] and 8 * kc8["longest_prompt"] <= kc8["capacity_tokens"]))                                                  # 3
    out.append(all(_ge(s["exact_share"], C_MIN_EXACT) for s in S.values()) if S else None)                                                                                            # 4
    pairs = [(g("S", k), g("KC", k)) for k in LEVELS if g("S", k) and g("KC", k)]
    out.append(all(_ge(a["exact_share"], b["exact_share"] - C_EXACT_GAP) for a, b in pairs) if pairs else None)                                                                      # 5
    sc = [s for (arm, _), s in S.items() if arm == "S"]; out.append(all(_ge(s["exact_share"], C_EXACT_HIGH) for s in sc) if sc else None)                                              # 6
    r = _ratio(s8["prompt_last_quarter"], s8["prompt_first_quarter"]) if s8 else None; out.append(None if r is None else _le(r, C_FLAT))                                              # 7
    r = _ratio(kc8["prompt_last_quarter"], kc8["prompt_first_quarter"]) if kc8 else None; out.append(None if r is None else _ge(r, C_GROWTH))                                          # 8
    out.append(None if kc8 is None or kc8["cached_share"] is None else _ge(kc8["cached_share"], C_CACHE_FITS))                                                                        # 9
    r = _ratio(s8["computed_per_step"], kc8["computed_per_step"]) if s8 and kc8 else None; out.append(None if r is None else _le(r, 1 - C_LIGHT_TOKENS))                               # 10
    r = _ratio(s8["seconds_per_step"], kc8["seconds_per_step"]) if s8 and kc8 else None; out.append(None if r is None else _le(r, 1 - C_LIGHT_TIME))                                  # 11
    r = _ratio(kc32["seconds_per_step"], s32["seconds_per_step"]) if kc32 and s32 else None; out.append(None if r is None else _ge(r, C_MID_RATIO))                                   # 12
    out.append(None if kc64 is None or kc64["cached_share_second_half"] is None else _lt(kc64["cached_share_second_half"], C_OVERRUN))                                                 # 13
    r = _ratio(kc64["seconds_per_step"], s64["seconds_per_step"]) if kc64 and s64 else None; out.append(None if r is None else _ge(r, C_PRESSURE_RATIO))                              # 14
    for s_, k_, lim in ((s8, kc8, C_QA_LIGHT), (s32, kc32, C_QA_MID), (s64, kc64, C_QA_HEAVY)):                                                                                         # 15 to 17
        r = _rate_ratio(s_["exact_per_hour"], k_["exact_per_hour"]) if s_ and k_ else None; out.append(None if r is None else _ge(r, lim))
    assert len(out) == len(CL_LINES)
    return [(i + 1, CL_LINES[i][0], CL_LINES[i][1], out[i]) for i in range(len(CL_LINES))]


def claims_allowed(checks) -> dict:
    """What the page lets the Result claim, from the lines (the rule on the page): the plain time ratios (lines 11, 12, 14) are claimed as ratios at equal exactness only if line 5 reads yes;
    the exact-steps-per-hour lines (15 to 17) are claimed as read, with the exactness gap stated. A line that was not run claims nothing."""
    ok = {i: c for i, _, _, c in checks}
    return dict(plain_time_at_equal_exactness=ok[5], quality_adjusted={i: ok[i] for i in (15, 16, 17)})


# ---------------------------------------------------------------- printing
def print_cells(S: Dict[Tuple[str, int], dict]) -> None:
    f = lambda x, k=1: "-" if x is None else f"{x:.{k}f}"; pc = lambda x: "-" if x is None else f"{x * 100:.0f}%"
    print(f"\n{'arm':3s} {'K':>3s} | {'steps':>5s} {'seconds':>8s} {'s/step':>7s} | {'prompt tok':>11s} {'computed':>10s} {'cached':>7s} {'late cached':>11s} | {'prompt q1':>9s} {'q4':>6s} {'longest':>8s} | {'exact':>6s} {'unparsed':>8s} {'exact/h':>8s}")
    for (arm, k), s in sorted(S.items(), key=lambda kv: (kv[0][1], ARMS.index(kv[0][0]))):
        print(f"{arm:3s} {k:3d} | {s['steps_done']:5d} {f(s['wall_seconds'], 0):>8s} {f(s['seconds_per_step'], 2):>7s} | {s['prompt_tokens']:11d} {f(s['computed_prefill'], 0):>10s} {pc(s['cached_share']):>7s} {pc(s['cached_share_second_half']):>11s} | "
              f"{f(s['prompt_first_quarter'], 0):>9s} {f(s['prompt_last_quarter'], 0):>6s} {s['longest_prompt'] or 0:8d} | {pc(s['exact_share']):>6s} {s['unparsed']:8d} {f(s['exact_per_hour'], 0):>8s}" + (f"   resumed at round {s['resumed_at']}" if s["resumed_at"] else ""))


def print_verdict(checks, title="provider-side test, constrained replies") -> None:
    print(f"\n-- {title} (reading rules fixed before the run) --")
    for i, tag, text, ok in checks: print(("  yes      " if ok else "  NO       " if ok is False else "  not run  ") + f"{i:2d}. [{tag}] {text}")
    ran = [ok for _, _, _, ok in checks if ok is not None]
    print(f"  ({len(ran)} of {len(checks)} lines could be read; headline lines {CL_HEADLINE[0]} and {CL_HEADLINE[1]}: " + ", ".join("yes" if checks[i - 1][3] else "NO" if checks[i - 1][3] is False else "not run" for i in CL_HEADLINE) + ")")
    a = claims_allowed(checks)
    print("  rule on the page: " + ("line 5 reads yes, so the plain time ratios (lines 11, 12, 14) may be claimed at equal exactness." if a["plain_time_at_equal_exactness"] else
          "line 5 reads NO, so the plain time ratios are NOT claimed as equal-exactness ratios; the claim rests on lines 15 to 17, with the exactness gap stated." if a["plain_time_at_equal_exactness"] is False else "line 5 was not run."))


def save_cell(cell: dict, path: str) -> str:
    with open(path, "w") as f: json.dump(cell, f)
    return path


def fingerprint() -> str:
    """sha256 of this module's source, recorded in the merged file so a result says which code made it."""
    return hashlib.sha256(open(__file__, "rb").read()).hexdigest()


def merge_files(paths: List[str], out: str) -> dict:
    cells = [json.load(open(p)) for p in paths]; m = merge_cells(cells); m["module_sha256"] = fingerprint(); m["provider_side_sha256"] = PS.fingerprint()
    with open(out, "w") as f: json.dump(m, f)
    return m


def dry_run(steps: int = STEPS, levels=None) -> Dict[Tuple[str, int], dict]:
    """Every planned cell on the stand-in, with the cache sized to match the proxy's token count. Prints the tables and the lines. NOT a result."""
    eng = ConstrainedStandIn(capacity_tokens=int(CAPACITY_TOKENS * PS.PROXY_SCALE)); cells = []
    for arm, k in cells_planned():
        if levels is None or k in levels: cells.append(run_cell(eng, arm, k, steps))
    S = summarize_all(cells); print("STAND-IN (a model of a block cache), NOT a result"); print_cells(S); print_verdict(verdict_constrained(S), "STAND-IN, not a result"); return S


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="provider-side test, constrained replies (docs/PROVIDER_SIDE_CONSTRAINED_TEST.md)")
    ap.add_argument("--dry-run", action="store_true", help="run every cell on the stand-in engine (no GPU); prints, does not save")
    ap.add_argument("--merge", nargs="+", metavar="CELL.json", help="put cell files together")
    ap.add_argument("--out", default="axi_provider_constrained.json"); a = ap.parse_args()
    if a.dry_run: dry_run()
    elif a.merge:
        m = merge_files(a.merge, a.out); S = summarize_all(m["cells"]); print_cells(S); print_verdict(verdict_constrained(S)); print("\nwritten to", a.out)
    else: ap.print_help()
