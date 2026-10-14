"""The KV-cache test on a LARGE graph: thousands of tokens in the state instead of a few hundred. Page with the predictions written BEFORE any run: docs/KV_LARGE_TEST.md (concept-map item 97).

It is the test of docs/KV_DROP_TEST.md (`kv_drop.py`, same arms, same cache protocol, same prompt layout, same scoring) with three differences:
  * the request script comes from here: a start graph of about 120 services and 200 edges on a 14 x 14 grid (the small test: 8 services and about 9 edges on a 3 x 4 grid), the same four kinds of
    request, at most 150 services and 260 edges;
  * the `keep` arm (a chat that never drops anything) is not run: one state is already thousands of tokens, so a chat would pass 20,000 tokens within a few steps and say nothing new;
  * the reading rules are its own (`LARGE_LINES`, `verdict_large`), written for this size. `summarize_large` adds what only this test needs (prompt size, first third against last third, the share of
    stale rows at the last step).
Everything else (arms, teacher forcing, KL, scoring, resumable runner, preflight) is `kv_drop.py`; nothing in that file changed meaning, it only gained an optional `script_fn` argument.

How the request script is made: the same way as `long_horizon.oracle_script` (requests are chosen on the oracle's graph, so each is meaningful there; balanced kinds so the graph neither empties nor
fills). `add_node` needs the grid, so it is written here; the other three kinds do not depend on the grid and are `cascade.pick_task`, unchanged."""
import random, re
from typing import Callable, List, Optional, Tuple

from axi.engine.gate import Graph, check, apply
from axi.engine.cico_parser import CICOParser
from axi.experiments.agent_loop import Task, copy_graph
from axi.experiments.cascade import pick_task
from axi.experiments.agent_loop import SYSTEM
from axi.experiments.kv_drop import KV_MAX_NEW, K_EXACT_POINTS, K_KL_SHARE, _ge, _le, _mean, apply_delta_items, first_prefix_text, graph_items, run_kv, summarize_kv, tail_text
from axi.experiments.long_horizon import KINDS_LONG

LARGE_NODES, LARGE_EDGES, LARGE_GRID = 120, 200, (14, 14)
LARGE_MAX_NODES, LARGE_MAX_EDGES = 150, 260
XL_NODES, XL_EDGES, XL_GRID = 240, 400, (20, 20)
XL_MAX_NODES, XL_MAX_EDGES = 300, 520
LARGE_SEED, LARGE_C_SEED, XL_SEED = 21, 21, 22          # run A and run C: episode seeds 21000 to 21003 (the same scripts, without and with the rule); run X: 22000 and 22001
LARGE_A_EPISODES, LARGE_A_STEPS = 4, 40
LARGE_C_EPISODES, LARGE_C_STEPS = 4, 40
XL_EPISODES, XL_STEPS = 2, 30
LARGE_ARMS = ("fresh", "exact", "drop_compact", "drop_gap")

# thresholds of the reading rules (docs/KV_LARGE_TEST.md)
L_MIN_PROMPT, L_FRESH_M6, L_EXACT_SAME, L_DC_SAME, L_DG_SAME, L_PREFILL_D, L_PREFILL_RATIO, L_SEC_D, L_LATE, L_C_SAME, L_STALE, L_X_SAME = 3000, 0.90, 0.98, 0.97, 0.95, 0.10, 5.0, 0.50, 0.95, 0.95, 0.80, 0.95
LARGE_HEADLINE = (5, 7, 8, 13, 14)
LARGE_LINES = [
    ("instrument", "The graph is large: the prompt of `fresh` holds on average at least 3,000 tokens over the steps of run A."),
    ("instrument", "The tokens built piece by piece (header, items, request) equal the tokenizer's own tokens for the whole prompt text on every step of run A."),
    ("guard", "fresh: the share of steps on which repair_m6 turns the reply into the right next state is at least 90 % (run A)."),
    ("guard", "exact: the reply is identical to fresh's, token for token, on at least 98 % of the steps of run A."),
    ("claim", "drop_compact: the reply is identical to fresh's on at least 97 % of the steps of run A."),
    ("claim", "drop_gap: the reply is identical to fresh's on at least 95 % of the steps of run A."),
    ("claim", "drop_compact: the share of steps on which repair_m6 turns the reply into the right next state is within 1 point of fresh's (run A)."),
    ("claim", "drop_compact: on at least 95 % of the steps of run A the largest KL along the reply, against fresh's distribution, is below 0.05 nats."),
    ("claim", "drop_compact reads at most 10 % of the tokens that fresh reads (run A, whole episodes, the first build of each included)."),
    ("claim", "exact reads at least 5 times as many tokens as drop_compact (run A)."),
    ("claim", "drop_compact: mean seconds per step at most 0.5 times fresh's (run A)."),
    ("claim", "drop_compact: mean seconds per step below exact's (run A)."),
    ("claim", "drop_compact: the reply is identical to fresh's on at least 95 % of the steps in the last third of the steps of run A."),
    ("claim", "run C (rule stated): drop_compact's reply is identical to fresh's on at least 95 % of the steps."),
    ("load", "drop_compact: at the last step of run A, at least 80 % of the state rows in its cache are stale."),
    ("claim", "run X (about 8,000 tokens): drop_compact's reply is identical to fresh's on at least 95 % of the steps."),
]


# ---------------------------------------------------------------- the large graph and its request script
def random_large_graph(rng: random.Random, n_nodes: int = LARGE_NODES, n_edges: int = LARGE_EDGES, grid: Tuple[int, int] = LARGE_GRID) -> Graph:
    """`n_nodes` services on random cells of the grid (values 1 to 9) and `n_edges` distinct edges (relation dep or owns, weight 1 to 9) between random pairs of services. One service in twenty
    gets no edge at all, so that 'delete a service without edges' has candidates from the start."""
    cells = [(r, c) for r in range(grid[0]) for c in range(grid[1])]; assert n_nodes <= len(cells); rng.shuffle(cells); nodes = cells[:n_nodes]
    g = Graph()
    for n in nodes: g.add_node(n, rng.randint(1, 9))
    core = nodes[:n_nodes - max(1, n_nodes // 20)]; tries = 0
    while len(g.edges) < n_edges and tries < 50 * n_edges:
        tries += 1; u, v = rng.sample(core, 2); rel = rng.choice(["dep", "owns"])
        if (u, v, rel) not in g.edges: g.add_edge(u, v, rel, rng.randint(1, 9))
    return g


def pick_large_task(rng: random.Random, g: Graph, kind: str, grid: Tuple[int, int] = LARGE_GRID) -> Optional[Task]:
    """`cascade.pick_task` with the free cells taken from `grid` for add_node (the other kinds do not look at the grid)."""
    if kind != "add_node": return pick_task(rng, g, kind)
    free = [(r, c) for r in range(grid[0]) for c in range(grid[1]) if (r, c) not in g.nodes]
    if not free: return None
    n = rng.choice(free); v = rng.randint(1, 9); ideal = f"ADD[{n[0]},{n[1]}:{v}]"; instr = f"Add a service at ({n[0]},{n[1]}) with value {v}."
    d = CICOParser.parse_transition_delta(ideal)
    if not check(g, d).ok: return None
    exp = copy_graph(g); apply(exp, d); return Task(kind, copy_graph(g), instr, ideal, exp)


def make_script_fn(n_nodes: int = LARGE_NODES, n_edges: int = LARGE_EDGES, grid: Tuple[int, int] = LARGE_GRID, max_nodes: int = LARGE_MAX_NODES, max_edges: int = LARGE_MAX_EDGES) -> Callable:
    """A request script like `long_horizon.oracle_script(seed, steps)` for a graph of this size: returns (start graph, requests, oracle graphs O_0..O_T)."""
    def script_fn(seed: int, steps: int):
        rng = random.Random(seed); g0 = random_large_graph(rng, n_nodes, n_edges, grid); live = copy_graph(g0); script, oracle = [], [copy_graph(g0)]
        for _ in range(steps):
            kinds = [k for k in KINDS_LONG if not (k == "add_node" and len(live.nodes) >= max_nodes) and not (k == "add_edge" and len(live.edges) >= max_edges)]; task = None
            for _ in range(20):
                task = pick_large_task(rng, live, rng.choice(kinds), grid)
                if task is not None: break
            if task is None: break
            script.append(dict(kind=task.kind, instruction=task.instruction, ideal=task.ideal)); live = copy_graph(task.expected); oracle.append(copy_graph(live))
        return g0, script, oracle
    script_fn.__name__ = f"large_{n_nodes}_{n_edges}_{grid[0]}x{grid[1]}"
    return script_fn


large_script = make_script_fn()
xl_script = make_script_fn(XL_NODES, XL_EDGES, XL_GRID, XL_MAX_NODES, XL_MAX_EDGES)


# ---------------------------------------------------------------- what the script alone fixes (no model, no tokenizer of Qwen)
def crude_tokens(text: str) -> int:
    """A rough token count in the manner of Qwen's pre-tokenizer: a word, a single digit, or a run of punctuation (each with the space before it) is one token. It is NOT Qwen's tokenizer; it is for
    the dry run only, and the real count is measured by the notebook."""
    return len(re.findall(r"\s?[A-Za-z]+|\s?\d|\s?[^\sA-Za-z\d]+|\s+", text))


def dry_run(script_fn: Callable, seeds, steps: int, count: Callable[[str], int] = crude_tokens) -> dict:
    """The parts of the reading rules that follow from the script and the design alone, with `count` (default `crude_tokens`) for the token counts: mean tokens of the first prompt; the tokens read over
    whole episodes by `fresh`, `exact` and the drop arms (the same bookkeeping as the arms: the first build, then at each step the items to append, for `exact` the items after the first deleted one,
    and the request); the share of the state rows (header included) that are stale in the cache that serves the LAST step (a row behind the first hole of any earlier step stays stale; items appended later are not)."""
    first, tot, stale_share = [], dict(fresh=0, exact=0, drop=0), []
    for seed in seeds:
        g0, script, _ = script_fn(seed, steps); items = graph_items(g0); stale = [False] * len(items); hdr = count(first_prefix_text(SYSTEM)); sz = lambda its: sum(count(" " + i.text) for i in its)
        change, prev = None, None
        for t, st in enumerate(script):
            tail = count(tail_text(st["instruction"])); tot["fresh"] += hdr + sz(items) + tail
            if t == 0: first.append(hdr + sz(items) + tail); tot["exact"] += hdr + sz(items) + tail; tot["drop"] += hdr + sz(items) + tail
            else:
                deleted, added = change; dset = set(deleted); i0 = min(deleted) if deleted else None
                reread = sum(count(" " + it.text) for j, it in enumerate(prev) if i0 is not None and j > i0 and j not in dset)
                tot["exact"] += reread + sz(added) + tail; tot["drop"] += sz(added) + tail
            w = [count(" " + it.text) for it in items]; share = sum(x for x, f in zip(w, stale) if f) / (hdr + sum(w))          # the cache that serves this step; the header rows are never stale
            new, deleted, added = apply_delta_items(items, CICOParser.parse_transition_delta(st["ideal"]))
            if deleted: stale = [stale[j] or j > min(deleted) for j in range(len(items)) if j not in set(deleted)]
            stale = stale + [False] * (len(new) - len(stale)); prev, items, change = items, new, (deleted, added)
        stale_share.append(share)
    return dict(prompt_first=sum(first) / len(first), drop_over_fresh=tot["drop"] / tot["fresh"], exact_over_fresh=tot["exact"] / tot["fresh"], exact_over_drop=tot["exact"] / tot["drop"],
                stale_share=sum(stale_share) / len(stale_share), tokens=tot)


# ---------------------------------------------------------------- summaries and the reading rules
def summarize_large(rows) -> dict:
    """`kv_drop.summarize_kv` plus: the mean prompt tokens of fresh, the agreement with fresh in the first and in the last third of the steps, and, for the drop arms, the share of the state
    rows that are stale at the last step of each episode."""
    out = summarize_kv(rows)
    for a, v in out.items():
        if not v.get("n"): continue
        eps = [r["arms"][a] for r in rows]; first = [s for e in eps for s in e if s["t"] <= len(e) // 3]
        v["same_first"] = _mean(s["same_as_fresh"] for s in first if s["same_as_fresh"] is not None)
        v["prompt_tokens_mean"] = _mean(s.get("prompt_tokens") for e in eps for s in e) if a == "fresh" else None
        v["stale_share_last"] = _mean(e[-1]["stale"] / e[-1]["resident_state"] for e in eps if e and e[-1].get("stale") is not None and e[-1]["resident_state"])
        v["state_rows_last"] = _mean(e[-1]["resident_state"] for e in eps if e)
    return out


def verdict_large(a: dict, c: Optional[dict] = None, x: Optional[dict] = None) -> List[Tuple[str, Optional[bool]]]:
    """The sixteen reading rules, in the order of LARGE_LINES. None = the run the line needs has not been made."""
    f, e, dc, dg = (a.get(n, {}) for n in ("fresh", "exact", "drop_compact", "drop_gap")); ratio = lambda p, q: None if not p or not q else p / q
    ok = [_ge(f.get("prompt_tokens_mean"), L_MIN_PROMPT), _ge(f.get("canonical"), 1.0), _ge(f.get("exact_m6"), L_FRESH_M6), _ge(e.get("same"), L_EXACT_SAME), _ge(dc.get("same"), L_DC_SAME),
          _ge(dg.get("same"), L_DG_SAME), None if dc.get("exact_m6") is None or f.get("exact_m6") is None else abs(dc["exact_m6"] - f["exact_m6"]) <= K_EXACT_POINTS + 1e-12, _ge(dc.get("kl_below"), K_KL_SHARE),
          _le(ratio(dc.get("prefilled"), f.get("prefilled")), L_PREFILL_D), _ge(ratio(e.get("prefilled"), dc.get("prefilled")), L_PREFILL_RATIO), _le(ratio(dc.get("seconds"), f.get("seconds")), L_SEC_D),
          None if dc.get("seconds") is None or e.get("seconds") is None else dc["seconds"] < e["seconds"], _ge(dc.get("same_late"), L_LATE), _ge((c or {}).get("drop_compact", {}).get("same"), L_C_SAME) if c else None,
          _ge(dc.get("stale_share_last"), L_STALE), _ge((x or {}).get("drop_compact", {}).get("same"), L_X_SAME) if x else None]
    return [(text, v) for (_, text), v in zip(LARGE_LINES, ok)]


def run_large(engine, kit, table, episodes: int, steps: int, seed: int, hint_rules: bool, path: str, script_fn: Callable, meta=None, progress=None, max_new: int = KV_MAX_NEW) -> dict:
    """`kv_drop.run_kv` with the four arms of this page and the script `script_fn`, and the summary of `summarize_large`. Resumable like `run_kv`."""
    res = run_kv(engine, kit, table, episodes, steps, seed, hint_rules, path, meta=meta, arms=LARGE_ARMS, max_new=max_new, progress=progress, script_fn=script_fn)
    res["summary"] = summarize_large(res["rows"]); return res


def print_large(summary: dict, title: str = ""):
    """The table of `kv_drop.print_kv` and, below it, what is particular to this test."""
    from axi.experiments.kv_drop import print_kv, _fx, _pc
    print_kv(summary, title)
    f = summary.get("fresh", {})
    if f.get("n"): print(f"\nprompt of fresh: {_fx(f.get('prompt_tokens_mean'), 0)} tokens on average over {f['n']} steps")
    print(f"{'arm':13s} {'reply = fresh, first third':>27s} {'last third':>11s} {'state rows at the last step':>28s} {'of them stale':>14s}")
    for arm, v in summary.items():
        if v.get("n") and arm != "fresh": print(f"{arm:13s} {_pc(v.get('same_first')):>27s} {_pc(v.get('same_late')):>11s} {_fx(v.get('state_rows_last'), 0):>28s} {_pc(v.get('stale_share_last')):>14s}")
