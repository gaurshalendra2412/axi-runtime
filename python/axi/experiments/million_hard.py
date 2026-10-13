"""The hard million-token test (concept-map item 95; rows 3, 9, 27). Page: docs/MILLION_TOKEN_TEST_HARD.md, written and hash-locked BEFORE any run.

The pages before it (MILLION_TOKEN_TEST.md on Qwen2.5-7B, MILLION_TOKEN_TEST_GEMINI.md on Gemini 3.5 Flash Lite) asked whether the held state survives about 1.4 million tokens of an easy job. This one asks the
same question of a current model (Gemini 3.8 Flash) on a job made harder in four ways, all fixed in the page before any run:
  * a DENSE graph: up to 24 services and 48 edges (the earlier pages: 14 and 26), so a prompt is about twice as long and most services have several dependants;
  * a new request kind, `delete_hub`: delete a service that has 4 to 8 dependants (the earlier kinds stop at 3), where the reply must name every one of them and the rule about dependants is NOT stated
    (the repair deletes dependants for a service with up to `HARD_DEGREE` = 8 of them; the framework's own default is 4, which keeps a bigger service rather than delete it, and would disagree with the oracle by design);
  * a re-ask: when the strict parser refuses a reply, the model is asked again (up to twice, `million.reask_messages`), because the saved Flash Lite runs showed unreadable replies on deletes and a reply that
    says nothing is the one slip the scope filter cannot see;
  * much stricter reading rules (`HARD_LINES`): every tenth of the run at least 80 percent exact, at least 90 percent exact overall, a distance that never passes 4 items, at most one step in a hundred that does not
    do what was asked, and the headline (`HEADLINE_LINES`) is only called passed if all nine core lines hold.
The money is guarded in `api_llm.DiskCachedLLM` (`budget_usd` with the prices the person typed in). Everything else (the run, the replay, the resume, the contrast) is `million.py`.
What this file does not do: it calls no model."""
import random
from typing import Dict, List, Optional

from axi.engine.cico_parser import CICOParser
from axi.engine.gate import apply, check
from axi.experiments import million as M
from axi.experiments.agent_loop import Task, copy_graph
from axi.experiments.endurance import Stream
from axi.experiments.scope_tasks import pick_scope_task

T = CICOParser.parse_transition_delta
HARD_MODEL = "gemini-3.8-flash"
HARD_SEEDS = (20261031, 20261032)
HARD_STEPS = 4000
HARD_CONTRAST_STEPS = 400
HARD_CONTROL_STEPS = 1000                  # the control arm: scope_m7 WITHOUT the re-ask, on the first 1,000 requests (reported, not a reading-rule line)
HARD_REASKS = 2
HARD_THINKING_LIMIT = 150                  # thinking tokens a step above which the 20-step check advises the lowest thinking level the model accepts ('low')
HARD_LOWEST_THINKING = "low"               # Gemini 3.8 Flash has no 'minimal' (its levels are low, medium, high)
HARD_TOKEN_CAP = 8_000_000                 # a second guard behind the dollar budget: no single run sends more than this
HARD_MINUTES_LIMIT = 240                   # projected minutes of calls for run 1 above which the 20-step check advises the lowest thinking level (4 hours)
HARD_WARM = 200                            # steps of the 20-step check that the oracle answers itself (not counted as sent) so that the graph has grown before the model is asked
HARD_FREE_DAILY = 20                       # calls a day the free tier is said to allow for this model (a free key can do the 20-step check and nothing more)
HARD_DEGREE = 8                            # `degree_threshold` of the repair in this test (the framework default is 4): a service with up to 8 dependants is deleted together with them, as the oracle does
MAX_NODES_H, MAX_EDGES_H = 24, 48
HUB_MIN, HUB_MAX = 4, 8
KINDS_HARD = ["add_node"] * 7 + ["add_edge"] * 20 + ["delete_dep"] + ["delete_hub"] * 3 + ["delete_leaf"] + ["delete_edge"] + ["delete_pair"]          # chosen by simulating the ORACLE only (page): about 20 services, 18 to 38 edges, 15 % deletes with dependants


def pick_hard_task(rng: random.Random, g, kind: str) -> Optional[Task]:
    """The picker of the scope test, plus `delete_hub`: the service at (r, c) has 4 to 8 edges in or out (its dependants), the request is the plain 'Delete the service at (r, c).', and the ideal reply
    deletes the service and every one of its edges. None if no service of the live graph has that many."""
    if kind != "delete_hub": return pick_scope_task(rng, g, kind)
    cand = [n for n in sorted(g.nodes) if HUB_MIN <= len(g.incident(n)) <= HUB_MAX]
    if not cand: return None
    n = rng.choice(cand)
    ideal = "DEL[%d,%d] " % n + " ".join(f"DEL[({u[0]},{u[1]})->({v[0]},{v[1]}):{r}#{g.edges[(u, v, r)]}]" for (u, v, r) in sorted(g.incident(n)))
    d = T(ideal)
    if not check(g, d).ok: return None
    exp = copy_graph(g); apply(exp, d)
    return Task(kind, copy_graph(g), f"Delete the service at ({n[0]},{n[1]}).", ideal, exp)


class HardStream(Stream):
    """`endurance.Stream` with the hard test's caps, kinds and picker. Never stored; deterministic in the seed."""
    max_nodes, max_edges = MAX_NODES_H, MAX_EDGES_H

    def __init__(self, seed: int):
        super().__init__(seed, "steady"); self.kinds_all = list(KINDS_HARD); self.picker = pick_hard_task


M.STREAMS["HardStream"] = HardStream


# ---------------------------------------------------------------- the reading rules (fixed in the page before the run)
H_FINISH = "the main run (scope_m7, the hard stream, up to two re-asks, the repair deleting up to 8 dependants, one of the two registered seeds) reached step 4,000 with every step on file, and the replay of its saved replies reproduces every saved flag of every step and checks every re-ask"
H_CORRUPT = "scope_m7 was never corrupted (no edge pointing at a missing service) at any step of the main run"
H_REFUSED = "at most 0.5 percent of the steps of the main run ended with a reply the strict parser refused (after the re-asks)"
H_BLIND = "blind apply, on the same requests, was corrupted on at least 80 percent of its first 400 steps"
H_TOKENS = "tokens sent in the main run (prompt, visible reply and thinking tokens of every call, as the service counted them) are at least 2,500,000"
H_TENTHS = "in every one of the ten tenths of the main run, at least 80 percent of the steps hold exactly the oracle's state"
H_ALL = "at least 90 percent of all steps of the main run hold exactly the oracle's state"
H_BOUND = "the distance to the oracle never exceeded 4 items at any step of the main run"
H_DRIFT = "the mean distance to the oracle in the last tenth of the main run is at most 0.5 above that of the first tenth"
H_WRONG = "among the steps of the main run that start from the oracle's state and ask for a change that can be applied, at most 1.0 percent did not do what was asked"
H_TOKFLAT = "the mean prompt tokens of the last tenth of the main run are at most 1.5 times those of the first tenth"
H_TIMEFLAT = "the mean seconds a step takes in the last tenth of the main run are at most 1.5 times those of the first tenth"
H_CONTRAST = "in steps 201 to 400, scope_m7's mean distance to the oracle is at most 0.5, and smaller than blind apply's and smaller than the gate's, on the same requests"
H_TWO = "the main run on the second seed also reached step 4,000, and lines 2, 3 and 6 to 12 also hold for it, so that the two runs together sent at least 5,000,000 tokens (only if run 2 was made)"
HARD_LINES = [("guard", H_FINISH), ("guard", H_CORRUPT), ("instrument", H_REFUSED), ("instrument", H_BLIND), ("claim", H_TOKENS), ("claim", H_TENTHS), ("claim", H_ALL), ("claim", H_BOUND),
              ("claim", H_DRIFT), ("claim", H_WRONG), ("claim", H_TOKFLAT), ("claim", H_TIMEFLAT), ("claim", H_CONTRAST), ("claim", H_TWO)]
HEADLINE_LINES = (1, 2, 3, 5, 6, 7, 8, 9, 10)          # the lines that decide whether the test is called passed; the others are reported with it
H_REFUSED_MAX, H_BLIND_MIN, H_TOKENS_MIN, H_TOKENS_MIN_TWO = 0.005, 0.8, 2_500_000, 5_000_000
H_TENTH_EXACT, H_ALL_EXACT, H_DISTANCE, H_DRIFT_ADD, H_WRONG_MAX, H_FLAT, H_CONTRAST_MEAN = 0.8, 0.9, 4, 0.5, 0.01, 1.5, 0.5


def registered(meta: dict, policy: str, steps: int) -> bool:
    """The settings of a run are the ones the page fixed: the hard stream, this policy, this many steps, one of the two seeds, two re-asks, the repair deleting up to 8 dependants, every reply asked of the model (none scripted). A file made with other settings
    cannot be read as the registered run (line 1 says NO, and a contrast made with other settings is 'not run')."""
    return (meta.get("stream") == "HardStream" and meta.get("policy") == policy and meta.get("steps") == steps and meta.get("seed") in HARD_SEEDS
            and meta.get("reasks", 0) == HARD_REASKS and meta.get("degree_threshold", 4) == HARD_DEGREE and meta.get("scripted", 0) == 0)


def _hard_run_lines(run: dict) -> dict:
    """The lines that are about one run, as booleans keyed by line number (1, 2, 3, 5 to 12)."""
    rows, meta = run["rows"], run["meta"]; n = len(rows); steps = HARD_STEPS; keys = (1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12)
    if n == 0: return {k: False for k in keys}
    try: _, _, bad = M.replay(rows, meta["seed"], meta["policy"], strict=False, stream=meta.get("stream", "LiveStream"), reasks=meta.get("reasks", 0), degree_threshold=meta.get("degree_threshold", 4))
    except M.ReplayMismatch: bad = [-1]                                  # the stream ended before the saved steps did: not the run that was saved
    first, last = M.block(rows, 0), M.block(rows, M.BLOCKS - 1)
    w, d = M.wrong_rate(rows)
    pf, pl = M._mean(r["prompt_tokens"] or 0 for r in first), M._mean(r["prompt_tokens"] or 0 for r in last)
    sf, sl = M._mean(r["seconds"] for r in first), M._mean(r["seconds"] for r in last)
    return {1: registered(meta, M.MAIN_POLICY, steps) and n == steps and [r["t"] for r in rows] == list(range(1, steps + 1)) and not bad,
            2: all(not r["corrupted"] for r in rows),
            3: sum(1 for r in rows if not r["parsed"]) <= H_REFUSED_MAX * n,
            5: M.tokens_sent(rows) >= H_TOKENS_MIN,
            6: all(M._mean(r["exact"] for r in M.block(rows, i)) >= H_TENTH_EXACT for i in range(M.BLOCKS)),
            7: M._mean(r["exact"] for r in rows) >= H_ALL_EXACT,
            8: max(r["distance"] for r in rows) <= H_DISTANCE,
            9: M._mean(r["distance"] for r in last) <= M._mean(r["distance"] for r in first) + H_DRIFT_ADD,
            10: d > 0 and w <= H_WRONG_MAX * d,
            11: pf > 0 and pl <= H_FLAT * pf,
            12: sf > 0 and sl <= H_FLAT * sf}


def verdict_hard(run1: dict, contrast: Optional[Dict[str, dict]] = None, run2: Optional[dict] = None) -> List[tuple]:
    """(tag, line, answer) for the 14 lines; answer is 'yes', 'NO' or 'not run' (lines 4 and 13 need the contrast: blind, gate and repair_m6, 400 steps each, made with the registered settings on the main run's seed; line 14 needs run 2, on the other seed)."""
    a = _hard_run_lines(run1); out = {k: ("yes" if v else "NO") for k, v in a.items()}
    seed = run1["meta"].get("seed")
    full = contrast is not None and all(p in contrast and len(contrast[p]["rows"]) == HARD_CONTRAST_STEPS and registered(contrast[p]["meta"], p, HARD_CONTRAST_STEPS) and contrast[p]["meta"].get("seed") == seed for p in M.CONTRAST_POLICIES)
    if full:
        b = contrast["blind"]["rows"]; out[4] = "yes" if sum(1 for r in b if r["corrupted"]) >= H_BLIND_MIN * len(b) else "NO"
        s = M._mean(r["distance"] for r in run1["rows"][200:400]); g = M._mean(r["distance"] for r in contrast["gate"]["rows"][200:400]); bl = M._mean(r["distance"] for r in b[200:400])
        out[13] = "yes" if len(run1["rows"]) >= 400 and s is not None and s <= H_CONTRAST_MEAN and s < g and s < bl else "NO"
    else: out[4] = out[13] = "not run"
    if run2 is None: out[14] = "not run"
    else:
        b = _hard_run_lines(run2); ok = run2["meta"].get("seed") != seed and b[1] and all(b[k] for k in (2, 3, 6, 7, 8, 9, 10, 11, 12)) and M.tokens_sent(run1["rows"]) + M.tokens_sent(run2["rows"]) >= H_TOKENS_MIN_TWO
        out[14] = "yes" if ok else "NO"
    return [(tag, text, out[i + 1]) for i, (tag, text) in enumerate(HARD_LINES)]


def headline(v: List[tuple]) -> str:
    """'PASSED' if every headline line is 'yes', 'NOT PASSED' if one of them is 'NO'."""
    ans = [v[i - 1][2] for i in HEADLINE_LINES]
    return "PASSED" if all(a == "yes" for a in ans) else "NOT PASSED" if any(a == "NO" for a in ans) else "NOT DECIDED"


# ---------------------------------------------------------------- what the page also reports (not reading-rule lines)
def report_hard(run: dict) -> dict:
    """Counted from the saved rows: new offsets (a step that starts on the oracle's state and ends off it) and what they followed; the re-asks; the slips by request kind."""
    rows = run["rows"]; new = [r for r in rows if r["start_same"] and not r["exact"]]
    by_kind: Dict[str, list] = {}
    for r in rows:
        if r["start_same"] and r["applicable"]:
            k = by_kind.setdefault(r["kind"], [0, 0]); k[1] += 1; k[0] += bool(r["wrong"])
    asked = [r for r in rows if r.get("asks", 1) > 1]
    return dict(steps=len(rows), new_offsets=len(new), from_refused=sum(1 for r in new if not r["parsed"]), from_readable=sum(1 for r in new if r["parsed"]),
                reasked_steps=len(asked), rescued=sum(1 for r in asked if r["parsed"]), still_refused=sum(1 for r in rows if not r["parsed"]), extra_calls=sum(r.get("asks", 1) - 1 for r in rows),
                dropped_items=sum(r.get("dropped", 0) for r in rows), steps_with_dropped=sum(1 for r in rows if r.get("dropped", 0) > 0),
                wrong_by_kind={k: tuple(v) for k, v in sorted(by_kind.items())}, exact_share=M._mean(r["exact"] for r in rows) if rows else None,
                tenth_exact=[M._mean(r["exact"] for r in M.block(rows, i)) for i in range(M.BLOCKS)] if rows else [])


def compare_control(main: dict, control: dict) -> dict:
    """The first `len(control rows)` steps of the main run against the control arm (scope_m7 without a re-ask, same requests)."""
    n = len(control["rows"]); a = main["rows"][:n]; b = control["rows"]
    f = lambda rows: dict(steps=len(rows), refused=sum(1 for r in rows if not r["parsed"]), exact=M._mean(r["exact"] for r in rows), distance=M._mean(r["distance"] for r in rows), max_distance=max((r["distance"] for r in rows), default=0))
    return dict(steps=n, with_reask=f(a), without_reask=f(b))


def print_hard_report(rep: dict, title: str = "main run"):
    print(f"\n== {title}: what the reading rules do not show ==")
    print(f"new offsets (a step that starts on the oracle's state and ends off it): {rep['new_offsets']}, of which after a reply the parser refused {rep['from_refused']}, after a readable reply {rep['from_readable']}")
    print(f"re-asks: {rep['reasked_steps']} steps needed one ({rep['extra_calls']} extra calls); {rep['rescued']} of them ended with a readable reply; steps that still ended refused: {rep['still_refused']}")
    print(f"scope filter: removed {rep['dropped_items']} items in {rep['steps_with_dropped']} steps; share of steps equal to the oracle {100 * rep['exact_share']:.1f}%" if rep["exact_share"] is not None else "no steps")
    print("steps that did not do what was asked / asked, by request kind: " + ", ".join(f"{k} {w}/{n}" for k, (w, n) in rep["wrong_by_kind"].items()))
    print("share exact in each tenth: " + " ".join(f"{100 * x:.0f}%" for x in rep["tenth_exact"]))


def print_control(c: dict):
    a, b = c["with_reask"], c["without_reask"]
    print(f"\n== control: the first {c['steps']} steps, scope_m7 with the re-ask against without it ==")
    for name, x in (("with re-ask", a), ("without re-ask", b)): print(f"{name:15s} refused {x['refused']:3d}  exact {100 * x['exact']:.1f}%  mean distance {x['distance']:.2f}  largest {x['max_distance']}")


def print_verdict_hard(v: List[tuple], title: str = "HARD MILLION-TOKEN TEST"):
    print(f"\n== reading rules, {title} ==")
    for i, (tag, text, ans) in enumerate(v, 1): print(f"{ans:7s} {i}. [{tag}] {text}" + ("   <- headline line" if i in HEADLINE_LINES else ""))
    print(f"\nHEADLINE (lines {', '.join(str(i) for i in HEADLINE_LINES)}): {headline(v)}")
