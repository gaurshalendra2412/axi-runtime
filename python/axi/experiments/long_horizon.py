"""Long-horizon test: how far does the held state drift from the intended state over many steps, and what does a growing chat history cost?
Page with the prediction written BEFORE any run: docs/LONG_HORIZON_TEST.md.

Drift, as measured here (the word has no single meaning in the posts; this is the operational one): the DISTANCE between the state a policy holds after step t and the state
an oracle holds after the same t requests. Distance = number of differing items (a service present in one and not the other, or with another value; an edge present in one
and not the other, or with another weight). Two ways to be far: CORRUPTION (an edge points at a missing service) and OMISSION (a requested change did not happen).

State mode. An episode is a fixed script of T requests made on the ORACLE's graph (so every policy gets the same requests). Each policy keeps its own live graph, shows
THAT graph to the model at every step (fresh context: system, exact current state, request), and updates it with its own rule (`cascade.next_state`):
  blind        apply what the model wrote, no checks            gate        refuse an inadmissible proposal, leave the graph as it was
  repair_m5    gate + propose_repair                            repair_m6   gate + ground_and_complete
Greedy decoding gives the same text for the same prompt, so identical prompts are asked once (`Memo`); a repeated prompt reports the seconds of its first call.

History mode. Same script, one policy (repair_m6), but the conversation ACCUMULATES: every earlier request, state and raw reply stays in the context and only the last turn is
new. The information is the same as in state mode plus the past; what changes is the length of the context. No KV cache is reused between steps (each step is a fresh generate),
which is the naive chat loop; a serving stack that reuses the cache would pay less. Per step: seconds, prompt tokens, whether the model's raw proposal was admissible as it stood.

Honest limits: one model family, synthetic graphs, greedy decoding; the oracle's requests never refer to something the oracle lacks, but a policy that has diverged can be asked
about something it does not hold (that is part of what drift is here). Under repair_m6 the held state is repaired whatever the model wrote, so state accuracy is near its ceiling in
history mode too; the raw-proposal columns (`admitted`) are where a change in the model's own behaviour would show."""
import json, os, random
from typing import Dict, List, Optional

from axi.engine.gate import Graph, check
from axi.experiments.agent_loop import Task, base_messages, copy_graph, same_graph, strict_parse
from axi.experiments.cascade import random_graph, pick_task, next_state, dangling_count
from axi.experiments.resumable import load_rows, _roundtrip

KINDS_LONG = ["add_node"] * 3 + ["add_edge"] * 3 + ["delete_dep"] * 3 + ["delete_leaf"]      # balanced so a long script does not empty the graph
MAX_NODES, MAX_EDGES = 14, 26
POLICIES = ("blind", "gate", "repair_m6")
MIXES = ("base", "scope")        # 'base': the mix of docs/LONG_HORIZON_TEST.md; 'scope': also delete_edge and delete_pair requests (docs/SCOPE_TEST.md)


def _mix(mix: str):
    assert mix in MIXES, mix
    if mix == "base": return KINDS_LONG, pick_task
    from axi.experiments.scope_tasks import KINDS_SCOPE_LONG, pick_scope_task
    return KINDS_SCOPE_LONG, pick_scope_task


# ---------------------------------------------------------------- oracle, distance
def oracle_script(seed: int, steps: int, mix: str = "base"):
    """(start graph, requests, oracle graphs O_0..O_T). The requests are chosen on the oracle's graph, so each is meaningful there; the script may end early if no request fits."""
    rng = random.Random(seed); g0 = random_graph(rng); live = copy_graph(g0); script, oracle = [], [copy_graph(g0)]
    kinds_all, picker = _mix(mix)
    for _ in range(steps):
        kinds = [k for k in kinds_all if not (k == "add_node" and len(live.nodes) >= MAX_NODES) and not (k == "add_edge" and len(live.edges) >= MAX_EDGES)]
        task = None
        for _ in range(20):
            task = picker(rng, live, rng.choice(kinds))
            if task is not None: break
        if task is None: break
        script.append(dict(kind=task.kind, instruction=task.instruction, ideal=task.ideal))
        live = copy_graph(task.expected); oracle.append(copy_graph(live))
    return g0, script, oracle


def distance(a: Graph, b: Graph) -> int:
    """Number of differing items between two graphs: services absent in one or with another value, edges absent in one or with another weight."""
    dn = sum(1 for n in set(a.nodes) | set(b.nodes) if a.nodes.get(n) != b.nodes.get(n))
    de = sum(1 for e in set(a.edges) | set(b.edges) if a.edges.get(e) != b.edges.get(e))
    return dn + de


class Memo:
    """Asks the model once per identical prompt (greedy decoding is deterministic). `last_cached` and `last_prompt_tokens` describe the latest call."""
    def __init__(self, llm): self.llm, self.cache, self.hits, self.calls, self.last_cached, self.last_prompt_tokens = llm, {}, 0, 0, False, None

    def generate(self, messages, constrained, max_new_tokens=400):
        key = json.dumps([messages, bool(constrained)], sort_keys=True)
        if key in self.cache:
            self.hits += 1; self.last_cached = True; out, self.last_prompt_tokens = self.cache[key]; return out
        out = self.llm.generate(messages, constrained); self.calls += 1; self.last_cached = False
        self.last_prompt_tokens = getattr(self.llm, "last_prompt_tokens", None); self.cache[key] = (out, self.last_prompt_tokens); return out


def _step_record(t, st, text, ntok, sec, d, live_before, live_after, expected, cached, prompt_tokens, dropped=None):
    rec = dict(t=t, kind=st["kind"], text=text, tokens=ntok, seconds=sec, cached=cached, prompt_tokens=prompt_tokens, parsed=d is not None,
               admitted=bool(d is not None and check(live_before, d).ok), corrupted=not live_after.is_well_formed(), dangling=dangling_count(live_after),
               distance=distance(live_after, expected), exact=same_graph(live_after, expected), nodes=len(live_after.nodes), edges=len(live_after.edges))
    if dropped is not None: rec["dropped"] = dropped                      # scope policies: how many items the scope filter removed at this step
    return rec


# ---------------------------------------------------------------- one episode
def run_state_episode(llm, seed: int, steps: int, policies=POLICIES, hint_rules=False, mix: str = "base") -> dict:
    g0, script, oracle = oracle_script(seed, steps, mix); out = {p: [] for p in policies}
    for p in policies:
        live = copy_graph(g0)
        for t, st in enumerate(script):
            task = Task(st["kind"], copy_graph(live), st["instruction"], st["ideal"], oracle[t + 1])
            text, ntok, sec = llm.generate(base_messages(task, hint_rules), constrained=True); d = strict_parse(text)
            after = next_state(p, live, d, st["instruction"])
            dropped = None
            if p in ("scope_m7", "scope_delta") and d is not None:
                from axi.engine.scope import admit_scoped
                dropped = len(admit_scoped(live, d, st["instruction"], "named" if p == "scope_m7" else "delta").dropped)
            out[p].append(_step_record(t + 1, st, text, ntok, sec, d, live, after, oracle[t + 1], getattr(llm, "last_cached", False), getattr(llm, "last_prompt_tokens", None), dropped))
            live = after
    return dict(seed=seed, steps=len(script), policies=out, stopped={p: None for p in policies})


def _is_oom(e: Exception) -> bool: return "OutOfMemory" in type(e).__name__ or "out of memory" in str(e).lower()


def run_history_episode(llm, seed: int, steps: int, hint_rules=False, policy="repair_m6", max_context_tokens=14000) -> dict:
    g0, script, oracle = oracle_script(seed, steps); live = copy_graph(g0); history: List[dict] = []; log = []; stopped = None
    for t, st in enumerate(script):
        task = Task(st["kind"], copy_graph(live), st["instruction"], st["ideal"], oracle[t + 1]); base = base_messages(task, hint_rules)
        messages = [base[0]] + history + [base[1]]
        try: text, ntok, sec = llm.generate(messages, constrained=True)
        except Exception as e:
            if _is_oom(e): stopped = "oom"; break
            raise
        d = strict_parse(text); after = next_state(policy, live, d); pt = getattr(llm, "last_prompt_tokens", None)
        log.append(_step_record(t + 1, st, text, ntok, sec, d, live, after, oracle[t + 1], False, pt))
        live = after; history += [base[1], {"role": "assistant", "content": text}]
        if pt and pt * (len(log) + 1) / len(log) > max_context_tokens and t + 1 < len(script): stopped = "context"; break      # the next turn would pass the cap
    return dict(seed=seed, steps=len(script), policies={policy: log}, stopped={policy: stopped})


# ---------------------------------------------------------------- resumable driver
def run_long(llm, mode: str, episodes: int, steps: int, seed: int, hint_rules: bool, path: str, meta=None, policies=POLICIES, max_context_tokens=14000, progress=None, mix: str = "base") -> dict:
    """mode 'state' or 'history'. One JSON line per finished episode; run again with the same arguments to carry on after a dropped session."""
    assert mode in ("state", "history") and (mode == "state" or mix == "base")
    pol = list(policies) if mode == "state" else ["repair_m6"]
    meta = dict(meta or {}, mode=mode, episodes=episodes, steps=steps, seed=seed, hint_rules=bool(hint_rules), policies=pol, max_context_tokens=max_context_tokens)
    if mix != "base": meta["mix"] = mix                                    # the base runs keep the meta they always had
    rows = load_rows(path, meta)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w") as f: f.write(json.dumps({"_meta": _roundtrip(meta)}) + "\n")
    memo = Memo(llm) if mode == "state" else llm
    with open(path, "a") as f:
        for e in range(len(rows), episodes):
            ep_seed = seed * 1000 + e
            row = run_state_episode(memo, ep_seed, steps, pol, hint_rules, mix) if mode == "state" else run_history_episode(llm, ep_seed, steps, hint_rules, "repair_m6", max_context_tokens)
            row = _roundtrip(row); f.write(json.dumps(row) + "\n"); f.flush(); os.fsync(f.fileno()); rows.append(row)
            if progress: progress(e + 1, episodes)
    return {"rows": rows, "summary": summarize_long(rows), "meta": meta}


# ---------------------------------------------------------------- summaries
def _share(xs): xs = list(xs); return sum(xs) / len(xs) if xs else None


def summarize_long(rows) -> dict:
    out = {}
    for p in (rows[0]["policies"] if rows else {}):
        eps = [r["policies"][p] for r in rows]; T = max((len(e) for e in eps), default=0); per_step = []
        for t in range(T):
            rs = [e[t] for e in eps if len(e) > t]
            pts = [r["prompt_tokens"] for r in rs if r.get("prompt_tokens") is not None]
            per_step.append(dict(t=t + 1, n=len(rs), corrupted=_share(r["corrupted"] for r in rs), dangling=_share(r["dangling"] for r in rs), distance=_share(r["distance"] for r in rs),
                                 exact=_share(r["exact"] for r in rs), seconds=_share(r["seconds"] for r in rs), prompt_tokens=_share(pts)))
        steps = [s for e in eps for s in e]; dd = [s for s in steps if s["kind"] == "delete_dep"]; last = [e[-1] for e in eps if e]
        out[p] = dict(per_step=per_step, episodes=len(eps), episodes_ever_corrupted=_share(any(s["corrupted"] for s in e) for e in eps),
                      corrupted_by_step_10=_share(any(s["corrupted"] for s in e[:10]) for e in eps), exact_share=_share(s["exact"] for s in steps),
                      final_distance=_share(s["distance"] for s in last), final_dangling=_share(s["dangling"] for s in last), episodes_exact_at_end=_share(s["exact"] for s in last),
                      raw_admitted=_share(s["admitted"] for s in steps), raw_admitted_delete_dep=_share(s["admitted"] for s in dd), n_delete_dep=len(dd),
                      stopped=[r["stopped"][p] for r in rows])
    return out


def _window(per_step, lo, hi, key): return _share(x[key] for x in per_step[lo:hi] if x[key] is not None)


def compare_context(state_rows, history_rows, policy="repair_m6") -> dict:
    """State against history on the SAME episodes (same seeds), for one policy: seconds and raw-proposal quality in the first and last quarter of the steps both reached."""
    by_seed = {r["seed"]: r for r in state_rows}; pairs = [(by_seed[h["seed"]], h) for h in history_rows if h["seed"] in by_seed]
    T = min((min(len(s["policies"][policy]), len(h["policies"][policy])) for s, h in pairs), default=0); q = max(T // 4, 1); out = dict(episodes=len(pairs), steps_compared=T)
    for name, sel in (("state", lambda s, h: s), ("history", lambda s, h: h)):
        eps = [sel(s, h)["policies"][policy][:T] for s, h in pairs]
        first, last = [x for e in eps for x in e[:q]], [x for e in eps for x in e[T - q:]]
        dd = lambda xs: [x["admitted"] for x in xs if x["kind"] == "delete_dep"]
        out[name] = dict(seconds_first=_share(x["seconds"] for x in first), seconds_last=_share(x["seconds"] for x in last),
                         prompt_tokens_first=_share(x["prompt_tokens"] for x in first if x["prompt_tokens"] is not None), prompt_tokens_last=_share(x["prompt_tokens"] for x in last if x["prompt_tokens"] is not None),
                         admitted_dd_first=_share(dd(first)), admitted_dd_last=_share(dd(last)), n_dd_first=len(dd(first)), n_dd_last=len(dd(last)),
                         exact_share=_share(x["exact"] for e in eps for x in e))
        if out[name]["seconds_first"]: out[name]["growth"] = out[name]["seconds_last"] / out[name]["seconds_first"]
    return out


# ---------------------------------------------------------------- reading rules (fixed in docs/LONG_HORIZON_TEST.md before the run)
def verdict_state(summary) -> List[tuple]:
    bl, ga, m6 = summary["blind"], summary["gate"], summary["repair_m6"]
    never = all(x["corrupted"] == 0 for p in (ga, m6) for x in p["per_step"])
    return [("gate and repair_m6 are never corrupted at any step", never),
            ("blind: at least 90 % of episodes are corrupted by step 10", (bl["corrupted_by_step_10"] or 0) >= 0.9),
            ("repair_m6 holds exactly the oracle's state on at least 95 % of steps", (m6["exact_share"] or 0) >= 0.95)]


def verdict_context(cmp) -> List[tuple]:
    s, h = cmp["state"], cmp["history"]
    out = [("state mode is flat: last-quarter seconds at most 1.3 times the first quarter", s.get("growth", 99) <= 1.3),
           ("history costs time: last-quarter seconds at least 3 times the first quarter", h.get("growth", 0) >= 3.0)]
    a, b = s.get("admitted_dd_last"), h.get("admitted_dd_last")
    out.append(("the raw proposals on delete_dep differ by at least 10 points between history and state (last quarter)", a is not None and b is not None and abs(a - b) >= 0.10))
    return out


def print_long(summary, title=""):
    print(f"\n== long horizon {title} ==")
    T = max(len(v["per_step"]) for v in summary.values())
    marks = [t for t in (1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 40) if t <= T]
    print(f"{'policy':11s} {'episodes':>8s} | " + " ".join(f"t={t:<3d}" for t in marks) + "   (mean distance to the oracle)")
    for p, v in summary.items(): print(f"{p:11s} {v['episodes']:8d} | " + " ".join(f"{v['per_step'][t-1]['distance']:5.1f}" if t <= len(v['per_step']) and v['per_step'][t-1]['distance'] is not None else "    -" for t in marks))
    print(f"{'policy':11s} {'corrupted by step 10':>21s} {'ever corrupted':>15s} {'exact steps':>12s} {'final dangling':>15s} {'raw admitted (delete_dep)':>26s}")
    for p, v in summary.items():
        f = lambda x: "-" if x is None else f"{x*100:.0f}%"
        print(f"{p:11s} {f(v['corrupted_by_step_10']):>21s} {f(v['episodes_ever_corrupted']):>15s} {f(v['exact_share']):>12s} {v['final_dangling'] if v['final_dangling'] is None else round(v['final_dangling'], 2):>15} {f(v['raw_admitted_delete_dep']):>26s}  (n={v['n_delete_dep']})")


def print_context(cmp):
    s, h = cmp["state"], cmp["history"]; f = lambda x, k=1: "-" if x is None else f"{x:.{k}f}"; pc = lambda x: "-" if x is None else f"{x*100:.0f}%"
    print(f"\n== state against history, repair_m6, {cmp['episodes']} episodes, first and last quarter of {cmp['steps_compared']} steps ==")
    print(f"{'':8s} {'seconds first':>14s} {'seconds last':>13s} {'growth':>7s} {'prompt tokens first':>20s} {'last':>8s} {'raw admitted dd first':>22s} {'last':>6s} {'exact steps':>12s}")
    for name, x in (("state", s), ("history", h)):
        print(f"{name:8s} {f(x['seconds_first'],2):>14s} {f(x['seconds_last'],2):>13s} {f(x.get('growth'),2):>7s} {f(x['prompt_tokens_first'],0):>20s} {f(x['prompt_tokens_last'],0):>8s} {pc(x['admitted_dd_first']):>22s} {pc(x['admitted_dd_last']):>6s} {pc(x['exact_share']):>12s}")


def print_verdict(checks, title):
    print(f"\n-- {title} (reading rule fixed before the run) --")
    for text, ok in checks: print(("  yes  " if ok else "  NO   ") + text)
