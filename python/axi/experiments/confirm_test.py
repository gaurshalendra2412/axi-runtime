"""Confirmation test (docs/CONFIRM_TEST.md; prediction, reading rules and the exact task lists written BEFORE any run).

Everything before this page was built and read on a few episodes and one seed. This test asks one question: do the findings of the first hosted-model run repeat on tasks nobody has looked at, with
enough episodes to put error margins on every number? Same model (Gemini 3.5 Flash Lite), same policies (blind, repair_m6, scope_m7), same request kinds (the 'scope' mix), same evaluation; what changes:
  * fresh seeds: 280 single requests from `make_scope_tasks(280, 21)` and 30 long episodes of 30 requests from seed 21 (episode seeds 21000 to 21029); earlier runs used seeds 3 and 7 only;
  * 30 episodes instead of 4 to 6, and every share that matters comes with a 95 percent bound (Wilson for episode shares; a bootstrap that resamples whole episodes for step shares);
  * one added line in the prompt (a worked example of a node delete with no edges), because the first hosted run lost 12 percent of its replies to the slip `DEL[2,0:7]` in place of `DEL[2,0]`.
All policies are evaluated offline on the same saved replies. All thresholds are integers or are compared with a 1e-12 margin."""
import hashlib, json, math, random
from typing import List

from axi.experiments.agent_loop import SYSTEM, _serialize
from axi.experiments.api_test import verdict_api_format
from axi.experiments.long_horizon import oracle_script, run_long
from axi.experiments.scope_tasks import make_scope_tasks
from axi.experiments.scope_test import run_scope_regime

SEED_SINGLE, N_SINGLE = 21, 280
SEED_LONG, EPISODES, STEPS = 21, 30, 30
POLICIES_LONG = ("blind", "repair_m6", "scope_m7")
BOOT_RESAMPLES, BOOT_SEED = 2000, 20261010
FIRST_RUN_BLIND_EXACT = (508, 1000)                       # the first hosted run, no rule: blind apply held the oracle's state on 50.8 percent of steps (docs/API_TEST_GEMINI.md)
NOTE = "\n  remove the service at (2,1), which has no edges -> DEL[2,1]"
EPS = 1e-12


class Noted:
    """Puts NOTE right after the system text of every prompt (before the rule, when the rule is stated). Everything else is passed through to the wrapped model."""
    def __init__(self, llm, note=NOTE): self.llm, self.note = llm, note

    def generate(self, messages, constrained=True, max_new_tokens=400):
        m = [dict(x) for x in messages]
        assert m[0]["role"] == "system" and m[0]["content"].startswith(SYSTEM), "unexpected system text"
        m[0]["content"] = SYSTEM + self.note + m[0]["content"][len(SYSTEM):]
        return self.llm.generate(m, constrained)

    def __getattr__(self, name): return getattr(self.__dict__["llm"], name)


# ---------------------------------------------------------------- the fixed task lists
def single_set(n=None): return make_scope_tasks(n or N_SINGLE, SEED_SINGLE)


def fingerprints() -> dict:
    """sha256 of the exact tasks the test asks: the single requests (kind, request, correct delta) and the 30 long scripts (start graph, then kind, request, correct delta per step)."""
    single = [[t.kind, t.instruction, t.ideal] for t in make_scope_tasks(N_SINGLE, SEED_SINGLE)]
    long_ = []
    for e in range(EPISODES):
        g0, script, _ = oracle_script(SEED_LONG * 1000 + e, STEPS, "scope")
        long_.append([_serialize(g0), [[s["kind"], s["instruction"], s["ideal"]] for s in script]])
    h = lambda x: hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()
    return dict(single=h(single), long=h(long_), n_single=len(single), episodes=len(long_), steps=sum(len(s) for _, s in long_))


# ---------------------------------------------------------------- the runs
def run_single(llm, hint_rules, path, meta=None, progress=None, tasks=None):
    return run_scope_regime(Noted(llm), tasks if tasks is not None else single_set(), hint_rules, path, meta=dict(meta or {}, confirm=True), progress=progress)


def run_long_confirm(llm, hint_rules, path, meta=None, progress=None, episodes=None):
    res = run_long(Noted(llm), "state", episodes or EPISODES, STEPS, SEED_LONG, hint_rules, path, meta=dict(meta or {}, confirm=True), policies=POLICIES_LONG, progress=progress, mix="scope")
    res["analysis"] = long_analysis(res["rows"]); return res


# ---------------------------------------------------------------- statistics (deterministic)
def wilson(k: int, n: int, z: float = 1.96):
    """95 percent Wilson interval for k of n (lower, upper)."""
    if n <= 0: return (0.0, 1.0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))


def boot_ratio(pairs, resamples=BOOT_RESAMPLES, seed=BOOT_SEED):
    """pairs: one (numerator, denominator) per episode. Returns (point, lower 2.5 %, upper 97.5 %) of sum(num) / sum(den), resampling whole episodes with replacement."""
    n = len(pairs); den = sum(b for _, b in pairs)
    if n == 0 or den == 0: return (float("nan"), float("nan"), float("nan"))
    rng = random.Random(seed); stats = []
    for _ in range(resamples):
        s = [pairs[rng.randrange(n)] for _ in range(n)]; d = sum(b for _, b in s)
        if d: stats.append(sum(a for a, _ in s) / d)
    stats.sort(); lo = stats[int(0.025 * len(stats))]; hi = stats[min(len(stats) - 1, int(0.975 * len(stats)))]
    return (sum(a for a, _ in pairs) / den, lo, hi)


def wasted_tokens(rows, policy):
    """(tokens spent on steps that START from a state already different from the oracle's, all tokens): prompt tokens plus visible output tokens of every step the policy takes, counting each step
    whether or not its reply was shared with another policy (what the policy would spend if it ran alone). Thinking tokens are not in the step records and are not counted."""
    wasted = total = 0
    for r in rows:
        steps = r["policies"][policy]
        for i, s in enumerate(steps):
            tk = (s.get("prompt_tokens") or 0) + (s.get("tokens") or 0); total += tk
            if i > 0 and not steps[i - 1]["exact"]: wasted += tk
    return wasted, total


def long_analysis(rows) -> dict:
    """Per policy: exact steps with a bootstrap bound, episodes ever corrupted, corrupted steps, wasted tokens; for repair_m6 and scope_m7 also the gap in exact steps over blind apply, paired by episode."""
    out = {}
    for p in POLICIES_LONG:
        pairs = [(sum(bool(s["exact"]) for s in r["policies"][p]), len(r["policies"][p])) for r in rows]; pt, lo, hi = boot_ratio(pairs); w, t = wasted_tokens(rows, p)
        out[p] = dict(episodes=len(rows), steps=sum(b for _, b in pairs), exact=sum(a for a, _ in pairs), exact_ci=[lo, hi], ever_corrupted=sum(any(s["corrupted"] for s in r["policies"][p]) for r in rows),
                      corrupted_steps=sum(bool(s["corrupted"]) for r in rows for s in r["policies"][p]), wasted_tokens=w, tokens=t)
    for p in ("repair_m6", "scope_m7"):
        diff = [(sum(bool(s["exact"]) for s in r["policies"][p]) - sum(bool(s["exact"]) for s in r["policies"]["blind"]), len(r["policies"]["blind"])) for r in rows]; pt, lo, hi = boot_ratio(diff)
        out[p]["gap_vs_blind"] = dict(num=sum(a for a, _ in diff), den=sum(b for _, b in diff), lower=lo, upper=hi)
    return out


# ---------------------------------------------------------------- reading rules (fixed in docs/CONFIRM_TEST.md before any run)
L_SINGLE_GUARD = "[guard] gate, repair_m6 and scope_m7 corrupt no graph in the 280 single requests, in either regime"
L_LONG_GUARD = "[guard] repair_m6 and scope_m7 are never corrupted at any long-run step, in either regime"
L_M6 = "[guard] repair_m6 holds the oracle's state on at least 95 percent of long-run steps, with the lower 95 percent bound at least 90 percent, in every regime run"
L_M7 = "[guard] scope_m7: the same, in every regime run"
L_M7_M6 = "[guard] scope_m7 is not behind repair_m6 by more than 3 points of exact steps, in every regime run"
L_EVER = "[need] {r}: blind apply corrupts the state at some step in a share of the episodes whose lower 95 percent bound is at least 30 percent"
L_GAP = "[need] {r}: repair_m6 is ahead of blind apply in exact steps by at least 10 points, and the lower 95 percent bound of the gap is above 0"
L_WASTE = "[need] no rule: blind apply spends at least 10 percent of its tokens on steps that start from an already wrong state, repair_m6 at most 2 percent"
L_SINGLE_NEED = "[need] no rule: on the 280 single requests repair_m6 is ahead of blind apply by at least 5 points"
L_REPLICATE = "[replicate] no rule: blind apply's exact-step share is within 15 points of the first run's 50.8 percent (35.8 to 65.8 percent)"
NAMES = {False: "no rule", True: "rule stated"}


def verdict_confirm(single=None, long_=None, single_rows=None, long_rows=None) -> List[tuple]:
    """single: {hint_rules: summarize_scope(...)}; long_: {hint_rules: long_analysis(rows)}; single_rows, long_rows: {hint_rules: rows} (for the [format] line). Regimes that were not run are skipped."""
    out = []; sr = sorted(single or {}); lr = sorted(long_ or {})
    if sr: out.append((L_SINGLE_GUARD, all(single[h]["all"][p]["corrupted"] == 0 for h in sr for p in ("gate", "repair_m6", "scope_m7"))))
    if lr:
        out.append((L_LONG_GUARD, all(long_[h][p]["corrupted_steps"] == 0 for h in lr for p in ("repair_m6", "scope_m7"))))
        for line, p in ((L_M6, "repair_m6"), (L_M7, "scope_m7")):
            out.append((line, all(100 * long_[h][p]["exact"] >= 95 * long_[h][p]["steps"] and long_[h][p]["exact_ci"][0] >= 0.90 - EPS for h in lr)))
        out.append((L_M7_M6, all(100 * (long_[h]["repair_m6"]["exact"] - long_[h]["scope_m7"]["exact"]) <= 3 * long_[h]["repair_m6"]["steps"] for h in lr)))
        for h in (False, True):
            if h in long_:
                b = long_[h]["blind"]; out.append((L_EVER.format(r=NAMES[h]), wilson(b["ever_corrupted"], b["episodes"])[0] >= 0.30 - EPS))
        for h in (False, True):
            if h in long_:
                g = long_[h]["repair_m6"]["gap_vs_blind"]; out.append((L_GAP.format(r=NAMES[h]), 10 * g["num"] >= g["den"] and g["lower"] > 0))
        if False in long_:
            b, m = long_[False]["blind"], long_[False]["repair_m6"]
            out.append((L_WASTE, b["tokens"] > 0 and 10 * b["wasted_tokens"] >= b["tokens"] and 50 * m["wasted_tokens"] <= m["tokens"]))
    if False in (single or {}):
        s = single[False]["all"]; out.append((L_SINGLE_NEED, 20 * (s["repair_m6"]["success"] - s["blind"]["success"]) >= s["n"]))
    if False in (long_ or {}):
        b = long_[False]["blind"]; out.append((L_REPLICATE, 1000 * b["exact"] >= 358 * b["steps"] and 1000 * b["exact"] <= 658 * b["steps"]))
    if single_rows or long_rows: out += verdict_api_format(single_rows, long_rows)
    return out


# ---------------------------------------------------------------- printing
def print_confirm_long(a, title=""):
    print(f"\n== long runs, {title} ({a['blind']['episodes']} episodes) ==")
    print(f"{'policy':10s} {'exact steps':>12s} {'95% bound':>16s} {'episodes ever corrupted':>26s} {'corrupted steps':>16s} {'tokens from a wrong state':>27s}")
    for p in POLICIES_LONG:
        v = a[p]; lo, hi = wilson(v["ever_corrupted"], v["episodes"])
        print(f"{p:10s} {100 * v['exact'] / v['steps']:11.1f}% {100 * v['exact_ci'][0]:6.1f} to {100 * v['exact_ci'][1]:5.1f}% {v['ever_corrupted']:>9d} of {v['episodes']:<3d} ({100 * lo:3.0f} to {100 * hi:3.0f}%) {v['corrupted_steps']:>10d}"
              f" {100 * v['wasted_tokens'] / max(v['tokens'], 1):>10.1f}% of {v['tokens']:,}")
    for p in ("repair_m6", "scope_m7"):
        g = a[p]["gap_vs_blind"]; print(f"{p} minus blind apply, exact steps: {100 * g['num'] / g['den']:+.1f} points (95% bound {100 * g['lower']:+.1f} to {100 * g['upper']:+.1f})")


def print_verdict(checks, title):
    print(f"\n-- {title} (reading rule fixed before the run) --")
    for text, ok in checks: print(f"  {'yes' if ok else 'NO ':<3}  {text}")
