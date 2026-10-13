"""Endurance and cost report (concept-map item 92; rows 3, 9, 27). Page: docs/ENDURANCE_TEST.md (Part C is pre-registered there; Parts A and B are descriptive and say so).

Three parts, none of which calls a model.

PART A, THE TOKEN BILL. Real token counts out of the saved runs (`docs/results/`): what one step costs when the held state is shown fresh at every step (state mode) against what it costs when the
whole conversation is kept and re-sent (history mode, the naive chat loop), and what that adds up to for longer jobs. The counts up to 30 steps are measured; everything beyond is a straight-line
extension of the measured growth per turn, and is labelled so wherever it is printed.

PART B, WHAT A WRONG STEP COSTS. For every saved 30-step run, the whole episode is replayed from the saved raw replies (the replay has to reproduce the saved `exact` flag of every step; it does), and the cost of an
early mistake is split into three things that are easy to mix up:
  1. persistence   : tokens spent on steps that START from a state different from the oracle's (`confirm_test.wasted_tokens`), i.e. the job is already wrong;
  2. inapplicable  : requests whose ideal change cannot be applied as written to the state the policy holds (it names a service the held state lacks, or would leave an edge dangling);
  3. fresh errors  : among the applicable requests, how often the step itself did not do what was asked, for steps that start from the oracle's state and for steps that start from a different state.
Only (3) can show a model getting WORSE after a mistake. (1) and (2) are bookkeeping: the state stayed wrong.

PART C, ENDURANCE OF THE HOST LAYER. A stand-in that writes the ideal reply, or one of 15 kinds of damaged reply (the kinds are listed in `FAULTS`), is run for a million steps through blind apply, the gate, repair_m6 and
scope_m7 on a stream of requests that is never stored. It counts crashes, corrupted states, time per step and memory. It is a test of the code that sits between a model and the held state; it says nothing about a model.

Honest limits are in the page. In particular: the damaged replies are made from the ORACLE's state and every policy receives the same text, so a policy that has drifted is sometimes given a reply meant for another state;
the token counts of Part A come from three tokenizers (Qwen, Gemini) and one task family; "tokens" in Part C are steps times the measured tokens per step, not tokens that were sent anywhere."""
import json, os, random, re, resource, statistics, time, traceback
from typing import Dict, List, Optional

from axi.engine.cico_parser import CICOParser
from axi.engine.gate import Graph, apply, check
from axi.experiments.agent_loop import copy_graph, same_graph, strict_parse
from axi.experiments.cascade import next_state, random_graph
from axi.experiments.long_horizon import MAX_EDGES, MAX_NODES, _mix, distance, oracle_script

RESULTS = os.path.join(os.path.dirname(__file__), "..", "..", "..", "docs", "results")
T = CICOParser.parse_transition_delta


# ================================================================ PART A: the token bill
def _load(name: str) -> dict:
    with open(os.path.join(RESULTS, name + ".json")) as f: return json.load(f)


def _mean(xs): xs = list(xs); return sum(xs) / len(xs) if xs else None


def state_tokens(rows, policy: str) -> dict:
    """Mean prompt tokens, mean output tokens and number of steps over every step of one policy in a state-mode file."""
    st = [s for r in rows for s in r["policies"][policy]]
    return dict(steps=len(st), prompt=_mean(s["prompt_tokens"] for s in st), output=_mean(s["tokens"] for s in st))


def history_growth(rows, policy: str = "repair_m6") -> dict:
    """Prompt tokens of a history-mode run by step number, over the steps ALL episodes reached (`steps`): the mean per step and the least-squares line (intercept, slope per turn). `lengths` are the
    steps each episode finished (an episode stops cleanly if the next turn would pass the context cap, or when the GPU ran out of memory)."""
    eps = [r["policies"][policy] for r in rows]; T_ = min(len(e) for e in eps)
    mean_by_t = [_mean(e[t]["prompt_tokens"] for e in eps) for t in range(T_)]
    xs = list(range(1, T_ + 1)); mx, my = _mean(xs), _mean(mean_by_t)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, mean_by_t)) / sum((x - mx) ** 2 for x in xs)
    return dict(steps=T_, episodes=len(eps), lengths=[len(e) for e in eps], first=mean_by_t[0], last=mean_by_t[-1], intercept=my - slope * mx, slope=slope, per_step=mean_by_t)


def history_total(n: int, intercept: float, slope: float) -> float:
    """Prompt tokens re-sent over n steps when step t carries intercept + slope * t tokens (the naive chat loop)."""
    return n * intercept + slope * n * (n + 1) / 2


def steps_until(window: float, intercept: float, slope: float) -> int:
    """First step whose prompt would pass `window` tokens."""
    return int((window - intercept) / slope) + 1


HISTORY_FILE, STATE_FILE, GEMINI_FILE = "axi_long_history_norule_Qwen2.5-7B_seed3", "axi_long_state_norule_Qwen2.5-7B_seed3", "axi_gemini_long_norule_gemini-3.5-flash-lite_seed7"
WINDOWS = {"5,500 (the T4 GPU ran out of memory near here)": 5500, "32,768": 32768, "128,000": 128000, "1,000,000": 10 ** 6, "2,000,000": 2 * 10 ** 6}
EXTENSION_STEPS = (100, 300, 1000, 3000, 10000)


def token_bill() -> dict:
    """Everything of Part A, from the saved files. MEASURED: the 4 episodes that ran in both modes (same seeds), every step each of them finished in history mode, prompt plus output tokens, Qwen 7B.
    EXTENSION: a straight line through the measured growth per turn, carried to longer jobs; nothing past the longest measured episode was run."""
    hrows = _load(HISTORY_FILE)["rows"]; hist = history_growth(hrows)
    srows = {r["seed"]: r for r in _load(STATE_FILE)["rows"]}; pairs = []
    for h in hrows:
        L = len(h["policies"]["repair_m6"]); s_ = srows[h["seed"]]["policies"]["repair_m6"][:L]; hs = h["policies"]["repair_m6"]
        pairs.append(dict(seed=h["seed"], steps=L, history=sum(x["prompt_tokens"] + x["tokens"] for x in hs), state=sum(x["prompt_tokens"] + x["tokens"] for x in s_), last_prompt=hs[-1]["prompt_tokens"]))
    sq = state_tokens([srows[h["seed"]] for h in hrows], "repair_m6"); sg = state_tokens(_load(GEMINI_FILE)["rows"], "scope_m7")
    hs_, ss_ = sum(p["history"] for p in pairs), sum(p["state"] for p in pairs); per_state = sq["prompt"] + sq["output"]; per_gem = sg["prompt"] + sg["output"]
    ext = []
    for n_ in EXTENSION_STEPS:
        h = history_total(n_, hist["intercept"], hist["slope"]) + n_ * sq["output"]; s = n_ * per_state
        ext.append(dict(steps=n_, state_tokens=s, history_tokens=h, saving=1 - s / h))
    return dict(history=hist, pairs=pairs, measured_steps=sum(p["steps"] for p in pairs), measured_state_tokens=ss_, measured_history_tokens=hs_, measured_saving=1 - ss_ / hs_,
                state_qwen=sq, state_gemini=sg, per_step_state_qwen=per_state, per_step_state_gemini=per_gem,
                windows={k: steps_until(v, hist["intercept"], hist["slope"]) for k, v in WINDOWS.items()}, extension=ext,
                state_steps_for={"1,000,000 tokens": 10 ** 6 / per_gem, "2,000,000 tokens": 2 * 10 ** 6 / per_gem})


# ================================================================ PART B: what a wrong step costs
SAVED_RUNS = [("Qwen 7B, rule not stated (base mix)", "axi_long_state_norule_Qwen2.5-7B_seed3"), ("Qwen 7B, rule stated (base mix)", "axi_long_state_rule_Qwen2.5-7B_seed3"),
              ("Qwen 7B, rule not stated (scope mix)", "axi_scope_long_norule_Qwen2.5-7B_seed7"), ("Qwen 7B, rule stated (scope mix)", "axi_scope_long_rule_Qwen2.5-7B_seed7"),
              ("Qwen 14B, rule not stated", "axi_14b_long_norule_Qwen2.5-14B_seed7"), ("Qwen 14B, rule stated", "axi_14b_long_rule_Qwen2.5-14B_seed7"),
              ("Flash Lite, rule not stated", "axi_gemini_long_norule_gemini-3.5-flash-lite_seed7"), ("Flash Lite, rule stated", "axi_gemini_long_rule_gemini-3.5-flash-lite_seed7")]


def replay_policy(rows, policy: str, mix: str) -> dict:
    """Replay every episode of one policy from the saved raw replies. Counts: steps, replay mismatches against the saved `exact` flag, unparsed replies, steps that start from the oracle's state or not,
    inapplicable requests (the ideal change is not admissible on the held state), applicable ones and how many of those the step did not do (per starting condition), and the tokens."""
    c = dict(steps=0, mismatch=0, unparsed=0, start_same=0, start_off=0, inapplicable_same=0, inapplicable_off=0, applicable_same=0, applicable_off=0, wrong_same=0, wrong_off=0, wasted=0, tokens=0)
    for r in rows:
        g0, script, oracle = oracle_script(r["seed"], r["steps"], mix); live = copy_graph(g0); prev_exact = True
        for t, (st, rec) in enumerate(zip(script, r["policies"][policy])):
            d = strict_parse(rec["text"]); after = next_state(policy, live, d, st["instruction"]); c["steps"] += 1
            if same_graph(after, oracle[t + 1]) != bool(rec["exact"]): c["mismatch"] += 1
            if d is None: c["unparsed"] += 1
            tk = (rec.get("prompt_tokens") or 0) + (rec.get("tokens") or 0); c["tokens"] += tk
            same = same_graph(live, oracle[t]); c["start_same" if same else "start_off"] += 1
            if t > 0 and not prev_exact: c["wasted"] += tk
            suf = "same" if same else "off"; ideal = T(st["ideal"])
            if check(live, ideal).ok:
                want = copy_graph(live); apply(want, ideal); c["applicable_" + suf] += 1; c["wrong_" + suf] += (not same_graph(after, want))
            else: c["inapplicable_" + suf] += 1
            prev_exact = bool(rec["exact"]); live = after
    return c


def cascade_report() -> dict:
    out = {}
    for label, name in SAVED_RUNS:
        d = _load(name); rows = d["rows"]; mix = d["meta"].get("mix") or "base"
        out[name] = dict(label=label, episodes=len(rows), mix=mix, policies={p: replay_policy(rows, p, mix) for p in rows[0]["policies"]})
    return out


# ================================================================ PART C: the endurance of the host layer
POLICIES_C = ("blind", "gate", "repair_m6", "scope_m7")
FAULTS = ("forget_dependents", "extra_edge_delete", "wrong_coordinate", "truncate", "drop_item", "duplicate_item", "extra_add", "garbage", "empty", "huge", "bigint", "stale", "format_slip", "unicode_digits", "negative")
P_FAULT = 0.5
HUGE_ITEMS = 1000
ITEM = re.compile(r"(?:DEL|ADD)\[[^\]]*\]")
SEED_C, EPISODES_C, STEPS_C = 20261012, 500, 2000                 # the million-step run: 500 episodes of 2,000 steps
SEED_L, EPISODES_L, STEPS_L = 20261013, 2, 250000                 # the two very long episodes
BLOCKS = 10                                                       # a run is cut in 10 equal blocks for time and distance


KINDS_STEADY = ["add_node"] * 3 + ["add_edge"] * 3 + ["delete_dep"] * 2 + ["delete_leaf"] + ["delete_edge"] + ["delete_pair"]       # adds outweigh deletes, so a long stream keeps a graph near the caps instead of an empty one


class Stream:
    """The requests of `long_horizon.oracle_script` (same picker, same caps, same random draws), produced one at a time and never stored, so an episode can be a million steps long. Two differences, both
    only where oracle_script would have STOPPED: when 20 random tries find no request that fits, every kind is tried once in a fixed order before the stream gives up (an empty graph then gets an add_node request),
    and mix 'steady' (KINDS_STEADY, the scope picker) exists so that long streams do not drift to an empty graph. For the mixes of the earlier tests the first n requests of Stream(seed, mix) are exactly those of
    oracle_script(seed, n, mix) whenever the latter does not stop early (a test pins this). `max_nodes` and `max_edges` are the caps (14 and 26); a subclass may raise them (`million_hard.HardStream`)."""
    max_nodes, max_edges = MAX_NODES, MAX_EDGES

    def __init__(self, seed: int, mix: str = "steady"):
        self.rng = random.Random(seed); self.start = random_graph(self.rng); self.live = copy_graph(self.start)
        if mix == "steady":
            from axi.experiments.scope_tasks import pick_scope_task
            self.kinds_all, self.picker = KINDS_STEADY, pick_scope_task
        else: self.kinds_all, self.picker = _mix(mix)

    def next(self):
        g = self.live; kinds = [k for k in self.kinds_all if not (k == "add_node" and len(g.nodes) >= self.max_nodes) and not (k == "add_edge" and len(g.edges) >= self.max_edges)]
        task = None
        for _ in range(20):
            task = self.picker(self.rng, g, self.rng.choice(kinds))
            if task is not None: break
        if task is None:
            for k in sorted(set(kinds)):
                task = self.picker(self.rng, g, k)
                if task is not None: break
        if task is None: return None
        before = self.live; self.live = copy_graph(task.expected)
        return dict(kind=task.kind, instruction=task.instruction, ideal=task.ideal), before


def _edge_txt(k, w): (u, v, r) = k; return f"DEL[({u[0]},{u[1]})->({v[0]},{v[1]}):{r}#{w}]"


def damage(fault: str, rng: random.Random, ideal: str, before: Graph, previous: str) -> str:
    """One kind of damaged reply for the request whose ideal reply is `ideal` on the oracle's state `before`. `previous` is the ideal reply of the step before. Every kind is deterministic given rng."""
    items = ITEM.findall(ideal)
    if fault == "forget_dependents": return " ".join(i for i in items if "->" not in i) or ideal
    if fault == "extra_edge_delete":
        asked = set(items); extra = [k for k in sorted(before.edges) if _edge_txt(k, before.edges[k]) not in asked]
        return ideal if not extra else ideal + " " + _edge_txt((k := rng.choice(extra)), before.edges[k])
    if fault == "wrong_coordinate":
        spans = [m.span() for m in re.finditer(r"\d+", ideal)]
        if not spans: return ideal
        a, b = rng.choice(spans); return ideal[:a] + str(rng.randint(0, 30)) + ideal[b:]
    if fault == "truncate": return ideal[:rng.randint(1, len(ideal) - 1)] if len(ideal) > 1 else ideal
    if fault == "drop_item":
        if not items: return ideal
        i = rng.randrange(len(items)); return " ".join(items[:i] + items[i + 1:])
    if fault == "duplicate_item":
        if not items: return ideal
        i = rng.randrange(len(items)); return " ".join(items[:i + 1] + items[i:])
    if fault == "extra_add": return ideal + f" ADD[{rng.randint(0, 6)},{rng.randint(0, 6)}:{rng.randint(1, 9)}]"
    if fault == "garbage": return "".join(rng.choice("abcdefgxyz []()->:#,.0123456789\n\tDELADD") for _ in range(rng.randint(5, 80)))
    if fault == "empty": return ""
    if fault == "huge":
        out = []
        for _ in range(HUGE_ITEMS):
            k = rng.randrange(3); a, b, c, d = (rng.randint(0, 9) for _ in range(4))
            out.append(f"DEL[{a},{b}]" if k == 0 else f"ADD[{a},{b}:{rng.randint(1, 9)}]" if k == 1 else f"ADD[({a},{b})->({c},{d}):dep#{rng.randint(1, 9)}]")
        return " ".join(out)
    if fault == "bigint": return re.sub(r"\d+", "9" * 40, ideal, count=1) if re.search(r"\d", ideal) else ideal
    if fault == "stale": return previous
    if fault == "format_slip": return re.sub(r"DEL\[(\d+),(\d+)\]", lambda m: f"DEL[{m[1]},{m[2]}:{rng.randint(1, 9)}]", ideal, count=1)
    if fault == "unicode_digits":
        table = {"0": "٠", "1": "١", "2": "٢", "3": "٣", "4": "٤", "5": "٥", "6": "٦", "7": "٧", "8": "٨", "9": "٩"}
        spans = [m.span() for m in re.finditer(r"\d", ideal)]
        if not spans: return ideal
        a, b = rng.choice(spans); return ideal[:a] + table[ideal[a]] + ideal[b:]
    if fault == "negative": return re.sub(r"\[(\d+)", lambda m: "[-" + m[1], ideal, count=1)
    raise ValueError(fault)


class Garbler:
    """Stand-in for a model: the ideal reply, or with probability `p` one kind of damaged reply (kinds chosen in turn from a shuffled deck, so every kind is used equally often). Seeded."""
    def __init__(self, seed: int, p: float = P_FAULT, faults=FAULTS):
        self.rng = random.Random(seed); self.p, self.faults, self.deck, self.previous = p, tuple(faults), [], ""

    def reply(self, ideal: str, before: Graph):
        kind = None
        if self.p > 0 and self.rng.random() < self.p:
            if not self.deck: self.deck = list(self.faults); self.rng.shuffle(self.deck)
            kind = self.deck.pop()
        text = ideal if kind is None else damage(kind, self.rng, ideal, before, self.previous); self.previous = ideal
        return text, kind


def run_endurance(episodes: int, steps: int, seed: int, p: float = P_FAULT, policies=POLICIES_C, faults=FAULTS, mix: str = "steady", clock=time.perf_counter_ns) -> dict:
    """Run `episodes` x `steps` steps through every policy. Returns counts per policy and per block of the run (time, distance), the crashes with the first example of each kind, and the peak memory.
    A crash (an exception that is not a parse refusal) is counted, the step is treated as an unreadable reply (nothing applied) and the run goes on."""
    nb = BLOCKS; per = {pol: dict(steps=0, corrupted=0, exact=0, crashes=0, dist_sum=[0] * nb, dist_n=[0] * nb, ns=[0] * nb, ns_n=[0] * nb, max_distance=0, max_nodes=0, max_edges=0, dangling_max=0, corrupted_by_fault={}) for pol in policies}
    crashes = {}; parse_crash = 0; fault_counts = {f: 0 for f in faults}; clean = 0; unparsed = 0; total = 0; rss = {}; done_steps = 0
    block_of = lambda t: min(nb - 1, t * nb // steps)
    for e in range(episodes):
        S = Stream(seed * 1000 + e, mix); G = Garbler(seed * 1000 + e, p, faults); held = {pol: copy_graph(S.start) for pol in policies}
        for t in range(steps):
            nxt = S.next()
            if nxt is None: break
            st, before = nxt; text, kind = G.reply(st["ideal"], before); total += 1
            if text == st["ideal"]: clean += 1                                                                        # a dealt kind can find nothing to damage (a format slip on a request with no service delete): the reply is then clean
            if kind is not None: fault_counts[kind] += 1
            try: d = strict_parse(text)
            except Exception as ex:                                                                                   # not a parse refusal: a crash in the parser
                parse_crash += 1; crashes.setdefault("parse:" + type(ex).__name__, dict(count=0, fault=kind, text=text[:200], trace=traceback.format_exc()[-400:]))["count"] += 1; d = None
            if d is None: unparsed += 1
            b = block_of(t)
            for pol in policies:
                s = per[pol]; t0 = clock()
                try: after = next_state(pol, held[pol], d, st["instruction"])
                except Exception as ex:
                    crashes.setdefault(f"{pol}:{type(ex).__name__}", dict(count=0, fault=kind, text=text[:200], trace=traceback.format_exc()[-400:]))["count"] += 1; s["crashes"] += 1; after = copy_graph(held[pol])
                s["ns"][b] += clock() - t0; s["ns_n"][b] += 1; held[pol] = after
                dist = distance(after, S.live); s["steps"] += 1; s["dist_sum"][b] += dist; s["dist_n"][b] += 1; s["max_distance"] = max(s["max_distance"], dist)
                s["exact"] += (dist == 0); wf = after.is_well_formed(); s["corrupted"] += (not wf)
                if not wf and kind is not None: s["corrupted_by_fault"][kind] = s["corrupted_by_fault"].get(kind, 0) + 1
                s["max_nodes"] = max(s["max_nodes"], len(after.nodes)); s["max_edges"] = max(s["max_edges"], len(after.edges))
                if not wf: s["dangling_max"] = max(s["dangling_max"], sum(1 for (u, v, _) in after.edges if u not in after.nodes or v not in after.nodes))
            done_steps += 1
        if e == max(0, episodes // 10 - 1): rss["after_10_percent"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    rss["end"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return dict(meta=dict(episodes=episodes, steps=steps, seed=seed, p_fault=p, faults=list(faults), mix=mix, policies=list(policies)), steps_run=total, clean_replies=clean, damaged_replies=total - clean,
                unparsed=unparsed, parse_crashes=parse_crash, fault_counts=fault_counts, dealt=sum(fault_counts.values()), crashes=crashes, peak_rss_mb=rss, policies=per)


def block_us(per_policy: dict) -> List[float]:
    """Mean microseconds per step in each tenth of the run, for one policy."""
    return [ns / n / 1000.0 if n else None for ns, n in zip(per_policy["ns"], per_policy["ns_n"])]


def block_distance(per_policy: dict) -> List[float]:
    return [s / n if n else None for s, n in zip(per_policy["dist_sum"], per_policy["dist_n"])]


# ================================================================ PART C: the run and its reading rules (fixed in docs/ENDURANCE_TEST.md before the run)
SEED_CONTROL, EPISODES_CONTROL, STEPS_CONTROL = 20261011, 20, 2000
BLIND_EPISODES = 50
LONG_SEEDS = (20261013, 20261014)
RESULT_FILE = "axi_endurance_cpu.json"
GATED = ("gate", "repair_m6", "scope_m7")
LIMIT_NODES, LIMIT_EDGES = 30, 60
L_CRASH = "no exception other than a parse refusal escaped from the parser or from any policy, in any part of the run"
L_CORRUPT = "gate, repair_m6 and scope_m7 were never corrupted at any step of the million-step run or of the two very long episodes"
L_CONTROL = "control, no damaged replies: every policy, blind apply included, held exactly the oracle's state at every step"
L_BLIND = "blind apply, given the damaged replies, was corrupted on at least 50 percent of its steps (the damage has teeth)"
L_SMALL = "scope_m7 never held more than 30 services or 60 edges (the oracle's own caps are 14 and 26)"
L_DRIFT = "scope_m7's mean distance to the oracle in the last tenth of each very long episode is at most 1.5 times that of the first tenth"
L_TIME = "scope_m7's host time per step in the last tenth of each very long episode is at most 1.5 times that of the first tenth"
L_FAST = "scope_m7's host time over the million steps is at most 100 microseconds per step on one core"
L_MEMORY = "peak memory at the end of the million-step run is at most 1.25 times the peak after its first tenth"
L_ORDER = "in the last tenth of each very long episode repair_m6's mean distance is at least 3 times scope_m7's and the gate's at least 10 times"
LINES_C = [("guard", L_CRASH), ("guard", L_CORRUPT), ("guard", L_CONTROL), ("instrument", L_BLIND), ("claim", L_SMALL), ("claim", L_DRIFT), ("claim", L_TIME), ("claim", L_FAST), ("claim", L_MEMORY), ("claim", L_ORDER)]


def run_all(say=print) -> dict:
    """The whole of Part C. Control (no damage, four policies), the million-step run (three gated policies), blind apply on the first 50 episodes of the same streams, two very long episodes."""
    out = {}
    say("control ..."); out["control"] = run_endurance(EPISODES_CONTROL, STEPS_CONTROL, SEED_CONTROL, p=0.0)
    say("million-step run ..."); out["million"] = run_endurance(EPISODES_C, STEPS_C, SEED_C, policies=GATED)
    say("blind apply, first 50 episodes ..."); out["blind"] = run_endurance(BLIND_EPISODES, STEPS_C, SEED_C, policies=("blind",))
    out["long"] = []
    for sd in LONG_SEEDS:
        say(f"very long episode, seed {sd} ..."); out["long"].append(run_endurance(1, STEPS_L, sd, policies=GATED))
    return out


def _ratio_ok(xs, k): return xs[0] is not None and xs[-1] is not None and xs[-1] <= k * xs[0]


def verdict_endurance(res: dict) -> List[tuple]:
    parts = [res["control"], res["million"], res["blind"]] + list(res["long"])
    crash = sum(r["parse_crashes"] + sum(c["count"] for c in r["crashes"].values()) for r in parts)
    runs_g = [res["million"]] + list(res["long"])
    corrupted = sum(r["policies"][p]["corrupted"] for r in runs_g for p in GATED)
    control_ok = all(s["steps"] > 0 and s["exact"] == s["steps"] for s in res["control"]["policies"].values()) and len(res["control"]["policies"]) == 4
    b = res["blind"]["policies"]["blind"]
    scope = [r["policies"]["scope_m7"] for r in runs_g]
    small = all(s["max_nodes"] <= LIMIT_NODES and s["max_edges"] <= LIMIT_EDGES for s in scope)
    drift = all(_ratio_ok(block_distance(r["policies"]["scope_m7"]), 1.5) for r in res["long"])
    tm = all(_ratio_ok(block_us(r["policies"]["scope_m7"]), 1.5) for r in res["long"])
    m = res["million"]["policies"]["scope_m7"]; fast = sum(m["ns"]) / m["steps"] / 1000.0 <= 100.0
    rss = res["million"]["peak_rss_mb"]; mem = rss["end"] <= 1.25 * rss["after_10_percent"]
    order = all(block_distance(r["policies"]["repair_m6"])[-1] >= 3 * block_distance(r["policies"]["scope_m7"])[-1] and block_distance(r["policies"]["gate"])[-1] >= 10 * block_distance(r["policies"]["scope_m7"])[-1] for r in res["long"])
    return [(L_CRASH, crash == 0), (L_CORRUPT, corrupted == 0), (L_CONTROL, control_ok), (L_BLIND, b["steps"] > 0 and b["corrupted"] / b["steps"] >= 0.5),
            (L_SMALL, small), (L_DRIFT, drift), (L_TIME, tm), (L_FAST, fast), (L_MEMORY, mem), (L_ORDER, order)]


def tokens_equivalent(res: dict, per_step: float) -> dict:
    """Steps the host layer handled and the tokens that many steps would carry at `per_step` tokens (prompt plus reply) a step. Arithmetic: nothing was sent to a model."""
    steps = sum(r["steps_run"] for r in [res["million"], res["blind"]] + list(res["long"]))
    return dict(steps=steps, million_run_steps=res["million"]["steps_run"], tokens=steps * per_step, per_step=per_step)


def print_endurance(res: dict, bill: Optional[dict] = None):
    print("\n== endurance of the host layer (Part C) ==")
    for key, title in (("control", "control, no damage"), ("million", "million-step run"), ("blind", "blind apply, first 50 episodes")):
        r = res[key]; print(f"\n{title}: {r['steps_run']} steps, {r['damaged_replies']} damaged replies, {r['unparsed']} unreadable, crashes {r['parse_crashes'] + sum(c['count'] for c in r['crashes'].values())}, peak memory {r['peak_rss_mb']['end']:.0f} MB")
        print(f"{'policy':10s} {'corrupted':>10s} {'exact':>8s} {'max dist':>9s} {'max svc':>8s} {'max edges':>10s} {'us/step':>8s}   mean distance by tenth of an episode")
        for p, s in r["policies"].items():
            us = sum(s["ns"]) / max(s["steps"], 1) / 1000.0
            print(f"{p:10s} {s['corrupted']:10d} {s['exact']:8d} {s['max_distance']:9d} {s['max_nodes']:8d} {s['max_edges']:10d} {us:8.1f}   " + " ".join(f"{x:6.1f}" for x in block_distance(s)))
    for i, r in enumerate(res["long"]):
        print(f"\nvery long episode {i + 1}: {r['steps_run']} steps, peak memory {r['peak_rss_mb']['end']:.0f} MB")
        print(f"{'policy':10s} {'corrupted':>10s} {'max svc':>8s} {'max edges':>10s}   mean distance by tenth                                  | host microseconds by tenth")
        for p, s in r["policies"].items():
            print(f"{p:10s} {s['corrupted']:10d} {s['max_nodes']:8d} {s['max_edges']:10d}   " + " ".join(f"{x:6.1f}" for x in block_distance(s)) + "  | " + " ".join(f"{x:5.0f}" for x in block_us(s)))
    if bill:
        t = tokens_equivalent(res, bill["per_step_state_gemini"]); print(f"\nsteps handled by the host layer: {t['steps']:,}; at the measured {t['per_step']:.0f} tokens a step that is about {t['tokens'] / 1e6:,.0f} million tokens of model traffic (arithmetic: nothing was sent to a model)")


def print_verdict_endurance(v, title="PART C"):
    print(f"\n== reading rules, {title} ==")
    for (tag, text), (_, ok) in zip(LINES_C, v): print(("yes " if ok else "NO  ") + f"[{tag}] " + text)


if __name__ == "__main__":
    import sys
    out_dir = sys.argv[1] if len(sys.argv) > 1 else RESULTS
    res = run_all(); bill = token_bill(); v = verdict_endurance(res); print_endurance(res, bill); print_verdict_endurance(v)
    import hashlib, platform
    res["_python"] = sys.version.split()[0]; res["_platform"] = platform.platform(); res["_cpus"] = os.cpu_count()
    with open(__file__, "rb") as f: res["_code_sha256"] = hashlib.sha256(f.read()).hexdigest()
    path = os.path.join(out_dir, RESULT_FILE)
    with open(path, "w") as f: json.dump(res, f, indent=1, sort_keys=True)
    print("saved", path)


# ================================================================ the Result tables (rendered here, so that a test can check the page against the saved data)
def _pc(a, b, k=1): return "-" if not b else f"{100.0 * a / b:.{k}f}%"


def _frac(a, b): return "-" if not b else f"{a}/{b} = {100.0 * a / b:.1f}%"


def _crashes(r): return r["parse_crashes"] + sum(c["count"] for c in r["crashes"].values())


def table_c_overview(res: dict) -> str:
    rows = ["| run | policy | steps | corrupted steps | steps equal to the oracle | largest distance | most services held | most edges held | host microseconds per step |", "|---|---|---|---|---|---|---|---|---|"]
    runs = [("control, no damage", res["control"])] + [("million-step run", res["million"]), ("blind apply, first 50 episodes", res["blind"])] + [(f"very long episode {i + 1}", r) for i, r in enumerate(res["long"])]
    for title, r in runs:
        for p, s in r["policies"].items():
            us = sum(s["ns"]) / max(s["steps"], 1) / 1000.0
            rows.append(f"| {title} | {p} | {s['steps']:,} | {s['corrupted']:,} | {s['exact']:,} | {s['max_distance']:,} | {s['max_nodes']:,} | {s['max_edges']:,} | {us:.1f} |")
    return "\n".join(rows)


def table_c_replies(res: dict) -> str:
    rows = ["| run | steps | clean replies | changed replies | refused by the strict parser | crashes | peak memory after the first tenth of the episodes, MB | peak memory at the end, MB |", "|---|---|---|---|---|---|---|---|"]
    runs = [("control", res["control"]), ("million-step run", res["million"]), ("blind apply", res["blind"])] + [(f"very long episode {i + 1}", r) for i, r in enumerate(res["long"])]
    for title, r in runs:
        rows.append(f"| {title} | {r['steps_run']:,} | {r['clean_replies']:,} | {r['damaged_replies']:,} | {r['unparsed']:,} ({_pc(r['unparsed'], r['steps_run'])}) | {_crashes(r)} | {r['peak_rss_mb']['after_10_percent']:.1f} | {r['peak_rss_mb']['end']:.1f} |")
    return "\n".join(rows)


def table_c_tenths(res: dict) -> str:
    head = "| episode | policy | " + " | ".join(f"{i}" for i in range(1, 11)) + " |"
    rows = ["Mean distance to the oracle, by tenth of the 250,000 steps:", "", head, "|---|---|" + "---|" * 10]
    for i, r in enumerate(res["long"], 1):
        for p in ("scope_m7", "repair_m6", "gate"): rows.append(f"| {i} | {p} | " + " | ".join(f"{x:.1f}" for x in block_distance(r["policies"][p])) + " |")
    rows += ["", "Host microseconds per step, by tenth:", "", head, "|---|---|" + "---|" * 10]
    for i, r in enumerate(res["long"], 1):
        for p in ("scope_m7", "repair_m6", "gate"): rows.append(f"| {i} | {p} | " + " | ".join(f"{x:.0f}" for x in block_us(r["policies"][p])) + " |")
    return "\n".join(rows)


def table_a(bill: dict) -> str:
    rows = ["Measured (Qwen2.5-7B, the four episodes that ran in both modes; prompt plus reply tokens of every step each episode finished):", "",
            "| episode seed | steps finished | tokens, whole chat kept | tokens, exact state shown each step | prompt at the last step, chat kept |", "|---|---|---|---|---|"]
    for p in bill["pairs"]: rows.append(f"| {p['seed']} | {p['steps']} | {p['history']:,} | {p['state']:,} | {p['last_prompt']:,} |")
    rows.append(f"| all four | {bill['measured_steps']} | {bill['measured_history_tokens']:,} | {bill['measured_state_tokens']:,} | |")
    h = bill["history"]
    rows += ["", f"The chat-kept prompt grew by {h['slope']:.1f} tokens a step (straight line through the mean of the four episodes over the {h['steps']} steps all of them reached: {h['intercept']:.1f} + {h['slope']:.1f} x step). "
             f"Measured saving over the {bill['measured_steps']} steps: **{100 * bill['measured_saving']:.1f} percent**. The two episodes that stopped early did so cleanly because the next turn would have passed the context cap or the GPU ran out of memory.",
             "", "Extension of that straight line to longer jobs (**not measured**; nothing beyond 30 steps was run):", "", f"| steps | tokens, exact state shown each step (Qwen: {bill['per_step_state_qwen']:.1f} a step) | tokens, whole chat re-sent each step | saving |", "|---|---|---|---|"]
    for e in bill["extension"]: rows.append(f"| {e['steps']:,} | {e['state_tokens']:,.0f} | {e['history_tokens']:,.0f} | {100 * e['saving']:.2f}% |")
    rows += ["", "Step at which the chat-kept prompt would pass a window (extension): " + "; ".join(f"{k}: step {v:,}" for k, v in bill["windows"].items()) + ".",
             "", f"With the exact state shown each step, Gemini 3.5 Flash Lite used {bill['per_step_state_gemini']:.1f} tokens a step (prompt {bill['state_gemini']['prompt']:.1f}, reply {bill['state_gemini']['output']:.1f}; the 120 steps of the scope_m7 policy in `axi_gemini_long_norule`): "
             f"1 million tokens of work is about {bill['state_steps_for']['1,000,000 tokens']:,.0f} steps and 2 million about {bill['state_steps_for']['2,000,000 tokens']:,.0f}."]
    return "\n".join(rows)


def table_b(rep: dict) -> str:
    rows = ["| setting | policy | steps | refused by the parser | tokens spent from a state different from the oracle's | requests that could not be applied as written | step did not do what was asked, starting from the oracle's state | same, starting from a different state |", "|---|---|---|---|---|---|---|---|"]
    for name, d in rep.items():
        for p, c in d["policies"].items():
            inap = c["inapplicable_same"] + c["inapplicable_off"]
            rows.append(f"| {d['label']} | {p} | {c['steps']} | {c['unparsed']} | {_pc(c['wasted'], c['tokens'])} | {_pc(inap, c['steps'])} | {_frac(c['wrong_same'], c['applicable_same'])} | {_frac(c['wrong_off'], c['applicable_off'])} |")
    return "\n".join(rows)
