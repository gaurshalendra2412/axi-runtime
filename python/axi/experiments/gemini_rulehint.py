"""Test 34: does the repair rule have to be coded? Gemini 3.8 Flash, the hard stream, a second seed, the rule stated in the prompt, with the repair on or off.
Page: docs/GEMINI_RULEHINT_TEST.md, written and hash-locked BEFORE any run.

The question comes from a reader of the article '98% Right, 21% Correct': the cascade-delete rule was given to the framework, not discovered, so how many real slips have a rule that tidy? This test does not
answer that in general. It asks three smaller things on the same job as the 4,000-request run (`million_hard.HardStream`, the same model, the same re-asks, the same repair threshold of 8):
  (a) does the first result repeat on a second seed (20261032, never run before);
  (b) if the rule is STATED in the prompt (`agent_loop.RULES_HINT`) and nothing repairs, does the model stop slipping (arm `hint_gate`);
  (c) if the rule is stated AND the repair is on (arm `hint_repair`), is the repair still needed.
Five arms, 400 requests each, one seed: repair (scope_m7, no hint), hint_gate (gate, hint), hint_repair (scope_m7, hint), blind (no check) and gate (check only), the last two being the contrast.
Nothing here edits the framework's core; it uses `million.run_million` as it is. What this file does not do: it calls no model."""
from typing import Dict, List, Optional

from axi.experiments import million as M
from axi.experiments import million_hard as H
from axi.experiments.agent_loop import copy_graph, same_graph, strict_parse
from axi.experiments.cascade import next_state

RH_SEED = H.HARD_SEEDS[1]                  # 20261032: the second seed of the hard test, never run
RH_STEPS = 400
RH_MODEL = H.HARD_MODEL
# arm name -> (policy, hint_rules)
ARMS = {"repair": ("scope_m7", False), "hint_gate": ("gate", True), "hint_repair": ("scope_m7", True), "blind": ("blind", False), "gate": ("gate", False)}
ARM_ORDER = ("repair", "hint_gate", "hint_repair", "blind", "gate")


def registered(meta: dict, arm: str) -> bool:
    """The settings of a run are the ones the page fixed: the hard stream, this arm's policy and hint, 400 steps, the second seed, two re-asks, the repair deleting up to 8 dependants, every reply asked of the model."""
    pol, hint = ARMS[arm]
    return (meta.get("stream") == "HardStream" and meta.get("policy") == pol and bool(meta.get("hint_rules")) == hint and meta.get("steps") == RH_STEPS and meta.get("seed") == RH_SEED
            and meta.get("reasks", 0) == H.HARD_REASKS and meta.get("degree_threshold", 4) == H.HARD_DEGREE and meta.get("scripted", 0) == 0)


def raw_slips(rows: list, meta: dict) -> dict:
    """The model's own slips, counted from the saved replies alone: replay the stream under the arm's policy and, at every step that starts on the oracle's state, apply the reply AS WRITTEN (blind) to that state; a slip is a
    reply whose blind result is not the oracle's state. Returns {slips, steps_counted, by_kind: {kind: (slips, steps)}}. This is what the model did, whatever the policy then made of it."""
    s = M.make_stream(meta.get("stream", "LiveStream"), meta["seed"]); live = copy_graph(s.start); n = 0; k = 0; by = {}
    pol, dt = meta["policy"], meta.get("degree_threshold", 4)
    for rec in rows:
        out = s.next()
        if out is None: raise M.ReplayMismatch(f"the stream ended before step {rec['t']}")
        st, ob = out; oa = s.live; d = strict_parse(rec["text"])
        if same_graph(live, ob):
            slip = not same_graph(next_state("blind", live, d, st["instruction"], degree_threshold=dt), oa)
            n += 1; k += slip; c = by.setdefault(rec["kind"], [0, 0]); c[1] += 1; c[0] += slip
        live = next_state(pol, live, d, st["instruction"], degree_threshold=dt)
    return dict(slips=k, steps_counted=n, by_kind={a: tuple(v) for a, v in sorted(by.items())})


def slip_rate(s: dict) -> Optional[float]:
    """Slips of the model over the steps that start on the oracle's state (a step that starts off the oracle's state is not counted: the model was shown a different state)."""
    r = s["raw"]; return r["slips"] / r["steps_counted"] if r["steps_counted"] else None


def row_cost(r: dict, price_in: float, price_out: float) -> float:
    """Dollars of one step at the typed prices (per million tokens), every call counted, as the article counted them: prompt tokens at the input price, visible reply and thinking tokens at the output price."""
    return sum((p["prompt_tokens"] * price_in + (p["tokens"] + p["think"]) * price_out) / 1e6 for p in M._parts(r))


def arm_stats(run: dict, price_in: Optional[float] = None, price_out: Optional[float] = None) -> dict:
    rows, meta = run["rows"], run["meta"]; n = len(rows); last = rows[-100:] if n else []
    exact = sum(1 for r in rows if r["exact"]); st = dict(steps=n, exact=exact, exact_share=exact / n if n else None, corrupted=sum(1 for r in rows if r["corrupted"]), refused=sum(1 for r in rows if not r["parsed"]),
                                                           max_distance=max((r["distance"] for r in rows), default=0), last100_distance=M._mean(r["distance"] for r in last), mean_distance=M._mean(r["distance"] for r in rows),
                                                           raw=raw_slips(rows, meta) if n else dict(slips=0, steps_counted=0, by_kind={}), reasked_steps=sum(1 for r in rows if r.get("asks", 1) > 1), tokens=M.tokens_sent(rows))
    if price_in is not None and price_out is not None:
        c = sum(row_cost(r, price_in, price_out) for r in rows); st["cost"] = c; st["cost_per_exact"] = c / exact if exact else None
    return st


# ---------------------------------------------------------------- the reading rules (fixed in the page before the run)
R1 = "all five arms reached step 400 with every step on file under the registered settings (second seed), and the replay of the saved replies reproduces every saved flag of every step of every arm"
R2 = "arm repair (scope_m7, no hint) was never corrupted (no edge pointing at a missing service) at any step"
R3 = "arm blind (no check, no hint) was corrupted on at least 40 percent of its 400 steps"
R4 = "arm repair: at least 95 percent of the 400 steps hold exactly the oracle's state"
R5 = "arm repair: the distance to the oracle never exceeded 4 items at any step"
R6 = "arm repair: the model slipped (a reply that, applied as written to the oracle's state, gives a wrong state) at least 10 times in the 400 steps, so that the repair had work to do"
R7 = "arm gate (check only, no hint) ends farther from the oracle than arm blind: its mean distance over the last 100 steps is larger than blind's"
R8 = "stating the rule: the share of steps (counted from the oracle's state) on which the model slipped is, in arm hint_gate, at most half of what it is in arm repair"
R9 = "arm hint_gate (rule stated, check only, no repair): at least 90 percent of the 400 steps hold exactly the oracle's state"
R10 = "arm hint_gate: the mean distance to the oracle over the last 100 steps is at most 0.5"
R11 = "arm hint_repair (rule stated and repair on): at least 99 percent of the 400 steps hold exactly the oracle's state"
R12 = "the share of steps on which the model slipped is, in arm hint_repair, no larger than in arm repair"
R13 = "cost of one exact step (dollars at the prices typed in the notebook, every call counted): arm repair costs at most 0.4 times what arm blind costs"
RH_LINES = [("guard", R1), ("guard", R2), ("instrument", R3), ("claim", R4), ("claim", R5), ("instrument", R6), ("claim", R7), ("claim", R8), ("claim", R9), ("claim", R10), ("claim", R11), ("claim", R12), ("claim", R13)]
HEADLINE_LINES = (1, 2, 4, 5)              # decide whether the first result repeats on a second seed; lines 6 to 13 are reported with it
EXACT_REPAIR, BLIND_CORRUPT, MIN_SLIPS, HALF, EXACT_HINT_GATE, DIST_HINT_GATE, EXACT_HINT_REPAIR, COST_RATIO = 0.95, 0.4, 10, 0.5, 0.9, 0.5, 0.99, 0.4


def verdict_rulehint(runs: Dict[str, dict], price_in: Optional[float] = None, price_out: Optional[float] = None) -> List[tuple]:
    """(tag, line, answer) for the 13 lines; answer is 'yes', 'NO' or 'not run' (a line that needs an arm that was not run, or the prices for line 13)."""
    have = {a: runs[a] for a in ARM_ORDER if a in runs}
    full = len(have) == len(ARM_ORDER) and all(len(r["rows"]) == RH_STEPS and registered(r["meta"], a) for a, r in have.items())
    S = {a: arm_stats(r, price_in, price_out) for a, r in have.items() if r["rows"]}
    ans = {}
    def yn(b): return "yes" if b else "NO"
    if full:
        bad = False
        for a, r in have.items():
            try: _, _, b = M.replay(r["rows"], r["meta"]["seed"], r["meta"]["policy"], strict=False, stream="HardStream", reasks=H.HARD_REASKS, degree_threshold=H.HARD_DEGREE); bad = bad or bool(b)
            except M.ReplayMismatch: bad = True
        ans[1] = yn(not bad)
    else: ans[1] = "NO" if have else "not run"
    ans[2] = yn(S["repair"]["corrupted"] == 0) if "repair" in S else "not run"
    ans[3] = yn(S["blind"]["corrupted"] >= BLIND_CORRUPT * S["blind"]["steps"]) if "blind" in S else "not run"
    if "repair" in S:
        ans[4] = yn(S["repair"]["exact"] >= EXACT_REPAIR * S["repair"]["steps"]); ans[5] = yn(S["repair"]["max_distance"] <= 4); ans[6] = yn(S["repair"]["raw"]["slips"] >= MIN_SLIPS)
    else: ans[4] = ans[5] = ans[6] = "not run"
    ans[7] = yn(S["gate"]["last100_distance"] > S["blind"]["last100_distance"]) if "gate" in S and "blind" in S else "not run"
    ans[8] = yn(slip_rate(S["hint_gate"]) <= HALF * slip_rate(S["repair"])) if "hint_gate" in S and "repair" in S and slip_rate(S["hint_gate"]) is not None and slip_rate(S["repair"]) is not None else "not run"
    if "hint_gate" in S: ans[9] = yn(S["hint_gate"]["exact"] >= EXACT_HINT_GATE * S["hint_gate"]["steps"]); ans[10] = yn(S["hint_gate"]["last100_distance"] <= DIST_HINT_GATE)
    else: ans[9] = ans[10] = "not run"
    ans[11] = yn(S["hint_repair"]["exact"] >= EXACT_HINT_REPAIR * S["hint_repair"]["steps"]) if "hint_repair" in S else "not run"
    ans[12] = yn(slip_rate(S["hint_repair"]) <= slip_rate(S["repair"])) if "hint_repair" in S and "repair" in S and slip_rate(S["hint_repair"]) is not None and slip_rate(S["repair"]) is not None else "not run"
    if "repair" in S and "blind" in S and "cost_per_exact" in S["repair"] and S["repair"]["cost_per_exact"] is not None and S["blind"].get("cost_per_exact"):
        ans[13] = yn(S["repair"]["cost_per_exact"] <= COST_RATIO * S["blind"]["cost_per_exact"])
    else: ans[13] = "not run"
    return [(tag, text, ans[i + 1]) for i, (tag, text) in enumerate(RH_LINES)]


def headline(v: List[tuple]) -> str:
    """'REPEATS' if the four headline lines are all 'yes', 'DOES NOT REPEAT' if one of them is 'NO', else 'NOT DECIDED'."""
    a = [v[i - 1][2] for i in HEADLINE_LINES]
    return "REPEATS" if all(x == "yes" for x in a) else "DOES NOT REPEAT" if any(x == "NO" for x in a) else "NOT DECIDED"


def print_arms(runs: Dict[str, dict], price_in: Optional[float] = None, price_out: Optional[float] = None):
    print("\n== the five arms, first 400 requests of the hard stream, seed %d ==" % RH_SEED)
    print(f"{'arm':12s} {'steps':>5s} {'exact':>6s} {'corrupt':>7s} {'refused':>7s} {'max dist':>8s} {'last100 dist':>12s} {'model slips':>11s} {'re-asks':>7s} {'cost $':>7s} {'$ / exact step':>14s}")
    for a in ARM_ORDER:
        if a not in runs or not runs[a]["rows"]: continue
        s = arm_stats(runs[a], price_in, price_out); c = s.get("cost"); ce = s.get("cost_per_exact")
        print(f"{a:12s} {s['steps']:5d} {100 * s['exact_share']:5.1f}% {s['corrupted']:7d} {s['refused']:7d} {s['max_distance']:8d} {s['last100_distance']:12.2f} {s['raw']['slips']:5d}/{s['raw']['steps_counted']:<5d} {s['reasked_steps']:7d} "
              f"{('%.2f' % c) if c is not None else '-':>7s} {('%.4f' % ce) if ce is not None else '-':>14s}")
    for a in ARM_ORDER:
        if a in runs and runs[a]["rows"]:
            by = arm_stats(runs[a])["raw"]["by_kind"]; print(f"  slips of the model by kind, {a}: " + (", ".join(f"{k} {w}/{n}" for k, (w, n) in by.items() if w) or "none"))


def print_verdict_rulehint(v: List[tuple], title: str = "RULE-IN-THE-PROMPT TEST (Test 34)"):
    print(f"\n== reading rules, {title} ==")
    for i, (tag, text, ans) in enumerate(v, 1): print(f"{ans:7s} {i}. [{tag}] {text}" + ("   <- headline line" if i in HEADLINE_LINES else ""))
    print(f"\nHEADLINE (lines {', '.join(str(i) for i in HEADLINE_LINES)}): {headline(v)}")
