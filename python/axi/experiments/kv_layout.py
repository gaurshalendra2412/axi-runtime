"""Does the ORDER the same graph is written in change what a stock model finds in it? Page with the predictions written BEFORE any run: docs/LAYOUT_TEST.md (concept-map item 98).

Where the idea comes from: a text the owner shared on 15 Oct 2026 (sections 1 and 3: coordinates are read by position in the text, so "the model treats the same graph written in another order, or turned,
as another problem"; the remedy there needs a trained model). The first half is a claim about a stock model and can be measured now: the SAME graph and the SAME request, written in different orders.
Run C of the large test found the reason to look: with the rule stated, even a model that reads the whole prompt afresh gets only 4 of 52 "delete a service and list what hangs on it" replies exactly right
(it writes edges that never existed). If the trouble is finding the edges of one service in a long list, the order the edges are written in should matter.

One problem = one graph (70 services, 120 edges, a 10 x 10 grid, about 2,300 tokens by Qwen's count) and one request, read afresh under each condition by the same model with the same rule stated:
  sorted      services by coordinates, then edges by (from, to, relation): the layout of every earlier KV test. The edges that START at a service are next to each other.
  shuffled    the same services and edges in a random order (services still before edges).
  grouped     each service followed by the edges that start at it.
  adjacency   each service followed by ALL its edges, those that start at it and those that end at it; every edge therefore appears twice. One sentence in the prompt says so ("write it once in a reply").
  rot90       the whole problem turned a quarter turn (every coordinate, in the state and in the request), then written sorted. One of eight frames, so it cannot show "rotation blindness" in general;
              it is a probe.
The requests are "delete the service at (r,c)" for a service with 1 to 3 edges (42 problems; the reply must list every edge that hangs on it) and, as a control that needs no search, "add an edge" and
"add a service" (8 problems). Scoring is `kv_drop.score_reply` (applied as written; with repair_m6) plus what the page needs: the share of the service's own edges that the reply lists (recall, split into
edges that start at the service and edges that end at it), and the edges the reply invents."""
import copy
import hashlib
import json
import os
import random
import re
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from math import comb

from axi.engine.cico_parser import CICOParser, EDGE
from axi.engine.gate import Graph, apply, check
from axi.experiments.agent_loop import Task, copy_graph
from axi.experiments.kv_drop import KV_MAX_NEW, Item, Tracked, _fx, _ge, _mean, _pc, decode, first_prefix_text, score_reply
from axi.experiments.kv_large import random_large_graph
from axi.experiments.resumable import _roundtrip, load_rows

LAYOUT_NODES, LAYOUT_EDGES, LAYOUT_GRID = 70, 120, (10, 10)
LAYOUT_SEED, LAYOUT_GRAPHS, LAYOUT_DEPS, LAYOUT_CONTROLS = 31, 14, 3, 8          # 14 graphs x 3 delete problems = 42, plus 8 control problems on the first 8 graphs
CONDITIONS = ("sorted", "shuffled", "grouped", "adjacency", "rot90")
NOTE_ADJ = " In the state below every edge is written twice, once after each of its two services; in a reply write it once."

# thresholds of the reading rules (docs/LAYOUT_TEST.md)
Y_MIN_PROMPT, Y_ROOM_LO, Y_ROOM_HI, Y_ORDER, Y_CONTIG, Y_GROUPED, Y_ADJ_RECALL, Y_ADJ_EXACT, Y_ROT, Y_CONTROL, Y_INVENT, Y_DIFFER = 2000, 0.15, 0.90, 0.10, 0.10, 0.05, 0.15, 0.15, 0.10, 0.85, 0.30, 0.20
LAYOUT_HEADLINE = (5, 7, 8)
LAYOUT_LINES = [
    ("instrument", "The graph is large enough to matter: the prompt of `sorted` holds on average at least 2,000 tokens."),
    ("instrument", "The tokens built piece by piece (header, items, request) equal the tokenizer's own tokens for the whole prompt text on every probe of every condition."),
    ("instrument", "Every condition answered the same 42 delete problems and the same 8 control problems."),
    ("guard", "sorted: the recall (the share of a service's own edges that the reply lists) is between 0.15 and 0.90, on the delete problems (there is room to see a change in either direction)."),
    ("claim", "shuffled: the recall is at least 0.10 below sorted's, on the delete problems."),
    ("claim", "sorted: the recall of the edges that START at the service is at least 0.10 above the recall of the edges that END at it."),
    ("claim", "adjacency: the recall is at least 0.15 above sorted's."),
    ("claim", "adjacency: the share of delete replies that are exactly right when applied as written is at least 15 points above sorted's."),
    ("claim", "grouped: the recall is at least 0.05 above sorted's."),
    ("claim", "rot90: the recall is within 0.10 of sorted's."),
    ("claim", "adjacency: the edges a delete reply invents (edges that are not in the graph), per reply, are at least 30 % fewer than sorted's."),
    ("claim", "control: in every condition at least 85 % of the add replies are exactly right when applied as written."),
    ("claim", "shuffled: the reply is not the same text as sorted's on at least 20 % of the delete problems (the order changes what the model writes)."),
]


# ---------------------------------------------------------------- coordinates, frames, tasks
def turn90(rc: Tuple[int, int], side: int) -> Tuple[int, int]:
    """A quarter turn of a side x side grid: (r, c) -> (c, side - 1 - r)."""
    return (rc[1], side - 1 - rc[0])


def transform_graph(g: Graph, f: Callable[[Tuple[int, int]], Tuple[int, int]]) -> Graph:
    h = Graph()
    for n, v in g.nodes.items(): h.add_node(f(n), v)
    for (u, v, rel), w in g.edges.items(): h.add_edge(f(u), f(v), rel, w)
    return h


def make_task(g: Graph, kind: str, **p) -> Task:
    """The request, its ideal reply and the expected next state, built exactly as `cascade.pick_task` builds them (the test compares the two)."""
    if kind == "delete_dep":
        n = p["node"]; ideal = "DEL[%d,%d] " % n + " ".join(f"DEL[({u[0]},{u[1]})->({v[0]},{v[1]}):{r}#{g.edges[(u, v, r)]}]" for (u, v, r) in sorted(g.incident(n))); instr = f"Delete the service at ({n[0]},{n[1]})."
    elif kind == "add_node":
        n, v = p["node"], p["value"]; ideal = f"ADD[{n[0]},{n[1]}:{v}]"; instr = f"Add a service at ({n[0]},{n[1]}) with value {v}."
    else:
        u, v, w = p["u"], p["v"], p["weight"]; ideal = f"ADD[({u[0]},{u[1]})->({v[0]},{v[1]}):dep#{w}]"
        instr = f"Add a dependency edge from ({u[0]},{u[1]}) to ({v[0]},{v[1]}) with relation dep and weight {w}."
    d = CICOParser.parse_transition_delta(ideal); assert check(g, d).ok, (kind, p)
    exp = copy_graph(g); apply(exp, d); return Task(kind, copy_graph(g), instr, ideal, exp)


def make_problems(seed: int = LAYOUT_SEED, graphs: int = LAYOUT_GRAPHS, deps: int = LAYOUT_DEPS, controls: int = LAYOUT_CONTROLS,
                  nodes: int = LAYOUT_NODES, edges: int = LAYOUT_EDGES, grid: Tuple[int, int] = LAYOUT_GRID) -> List[dict]:
    """Per graph: `deps` delete problems on different services with 1 to 3 edges; the first `controls` graphs also get one add problem (an edge, then a service, alternating). Each problem is
    {id, kind, graph, params} with the coordinates of the identity frame."""
    rng = random.Random(seed * 1000); out = []
    for gi in range(graphs):
        g = random_large_graph(rng, nodes, edges, grid); cand = sorted(n for n in g.nodes if 1 <= len(g.incident(n)) <= 3); chosen = rng.sample(cand, deps)
        for k, n in enumerate(chosen): out.append(dict(id=f"g{gi}d{k}", kind="delete_dep", graph=g, params=dict(node=n), side=grid[0]))
        if gi < controls:
            if gi % 2 == 0:
                nd = sorted(g.nodes); pairs = [(u, v) for u in nd for v in nd if u != v and (u, v, "dep") not in g.edges]; u, v = rng.choice(pairs); out.append(dict(id=f"g{gi}c", kind="add_edge", graph=g, params=dict(u=u, v=v, weight=rng.randint(1, 9)), side=grid[0]))
            else:
                free = [(r, c) for r in range(grid[0]) for c in range(grid[1]) if (r, c) not in g.nodes]; out.append(dict(id=f"g{gi}c", kind="add_node", graph=g, params=dict(node=rng.choice(free), value=rng.randint(1, 9)), side=grid[0]))
    return out


def fingerprint(problems: Sequence[dict]) -> str:
    """A short hash of what the problems contain (graphs, kinds, parameters), written into the file header: a file made with other problems under the same ids is refused."""
    h = hashlib.sha256()
    for p in problems: h.update(repr((p["id"], p["kind"], sorted(p["graph"].nodes.items()), sorted(p["graph"].edges.items()), sorted(p["params"].items()), p["side"])).encode())
    return h.hexdigest()[:16]


def realize(problem: dict, condition: str) -> Tuple[Graph, Task, List[Item], dict]:
    """(the graph as the condition writes it, the task on it, the item list in the condition's order, the task's parameters in that frame). Only `rot90` changes the coordinates."""
    g, p, kind = problem["graph"], dict(problem["params"]), problem["kind"]
    if condition == "rot90":
        f = lambda rc: turn90(rc, problem["side"]); g = transform_graph(g, f)
        for key in ("node", "u", "v"):
            if key in p: p[key] = f(p[key])
    task = make_task(g, kind, **p); return g, task, layout_items(g, condition, problem["id"]), p


def layout_items(g: Graph, condition: str, pid: str = "") -> List[Item]:
    assert condition in CONDITIONS, condition
    nodes = [Item(("n", n), v) for n, v in sorted(g.nodes.items())]; edges = [Item(("e", k), w) for k, w in sorted(g.edges.items())]
    if condition in ("sorted", "rot90"): return nodes + edges
    if condition == "shuffled":
        rng = random.Random(f"{LAYOUT_SEED}/{pid}"); a, b = list(nodes), list(edges); rng.shuffle(a); rng.shuffle(b); return a + b
    out_e: Dict[tuple, List[Item]] = {n: [] for n, _ in sorted(g.nodes.items())}; in_e: Dict[tuple, List[Item]] = {n: [] for n in out_e}
    for it in edges:
        u, v, _ = it.key[1]; out_e[u].append(it); in_e[v].append(it)
    out = []
    for it in nodes:
        n = it.key[1]; out.append(it); out += out_e[n]
        if condition == "adjacency": out += in_e[n]
    return out


def kit_for(kit, condition: str):
    """The prompt kit for a condition: the base kit, or (adjacency) the same kit with one more sentence in the system prompt."""
    if condition != "adjacency": return kit
    k = copy.copy(kit); k.system = kit.system + NOTE_ADJ; k.first_prefix = kit.enc(first_prefix_text(k.system)); return k


# ---------------------------------------------------------------- scoring
def lenient_deleted_edges(text: str) -> List[tuple]:
    """The edge deletions a reply spells out completely, as (u, v, relation, weight), read item by item. A reply that ran into the token limit is not parseable as a whole (its last item is cut), but the
    items before the cut were written; the recall below counts them. `strict_parse` (the exactness scores) still refuses such a reply."""
    out = []
    for it in re.split(r"[ \n]", text.strip()):
        m = EDGE.fullmatch(it)
        if m and m[1] == "DEL": out.append(((int(m[2]), int(m[3])), (int(m[4]), int(m[5])), m[6], int(m[7])))
    return out


def reply_metrics(text: str, g: Graph, node: Tuple[int, int]) -> dict:
    """For a delete reply: of the edges that hang on `node` (its own), the share the reply lists (recall), the same for the edges that start at it and those that end at it, the number of edges the reply
    lists that are not in the graph exactly as written (invented: no such edge, or the weight is wrong) and the number that are in the graph but are not the service's own (other). Items are read one by one
    (`lenient_deleted_edges`), so a reply cut off at the token limit counts for what it wrote before the cut."""
    own = sorted(g.incident(node)); out_e = [k for k in own if k[0] == node]; in_e = [k for k in own if k[1] == node]; part = lambda ks, got: None if not ks else sum(k in got for k in ks) / len(ks)
    dels = lenient_deleted_edges(text); real = lambda e: g.edges.get((e[0], e[1], e[2])) == e[3]; got = {(e[0], e[1], e[2]) for e in dels if real(e)}; own_set = set(own)
    return dict(recall=part(own, got), recall_out=part(out_e, got), recall_in=part(in_e, got), invented=sum(1 for e in dels if not real(e)), other=sum(1 for e in dels if real(e) and (e[0], e[1], e[2]) not in own_set), own=len(own))


def read_one(engine, kit, table, items: Sequence[Item], instruction: str, max_new: int = KV_MAX_NEW) -> dict:
    """The model reads the prompt afresh and answers (greedy, under the grammar of every earlier test)."""
    toks = kit.prompt_tokens(items, instruction); cache = Tracked(engine.new_cache()); engine.reset_peak(); engine.sync(); t0 = time.perf_counter(); logits = cache.forward(toks, range(len(toks))); engine.sync(); t1 = time.perf_counter()
    rep = decode(cache, logits, table, kit.eos, len(toks), max_new); engine.sync(); t2 = time.perf_counter(); out = dict(tokens=list(rep.tokens), ended=rep.ended, prompt_tokens=len(toks), canonical=kit.canonical(items, instruction), seconds=t2 - t0, peak_mb=engine.peak_mb())
    del cache; return out


def run_problem(engine, kit, table, problem: dict, conditions: Sequence[str] = CONDITIONS, max_new: int = KV_MAX_NEW) -> dict:
    row = dict(id=problem["id"], kind=problem["kind"], conds={})
    for cnd in conditions:
        g, task, items, p = realize(problem, cnd); k = kit_for(kit, cnd); rec = read_one(engine, k, table, items, task.instruction, max_new); sc = score_reply(k, rec["tokens"], g, task.expected, task.instruction); rec.update(sc)
        if problem["kind"] == "delete_dep": rec.update(reply_metrics(sc["text"], g, p["node"]))
        row["conds"][cnd] = rec
    return row


def run_layout(engine, kit, table, path: str, problems: Optional[List[dict]] = None, conditions: Sequence[str] = CONDITIONS, meta=None, progress=None, max_new: int = KV_MAX_NEW) -> dict:
    """One JSON line per finished problem (all conditions); run again with the same arguments to carry on after a dropped session. The file header says the settings, and a file made with other settings is refused."""
    assert kit.hint_rules, "this test states the rule"
    problems = problems if problems is not None else make_problems()
    meta = dict(meta or {}, kind="kv_layout", problems=len(problems), conditions=list(conditions), seed=LAYOUT_SEED, nodes=LAYOUT_NODES, edges=LAYOUT_EDGES, grid=list(LAYOUT_GRID), max_new=max_new, ids=[p["id"] for p in problems], fp=fingerprint(problems))
    rows = load_rows(path, meta)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w") as f: f.write(json.dumps({"_meta": _roundtrip(meta)}) + "\n")
    with open(path, "a") as f:
        for i in range(len(rows), len(problems)):
            row = _roundtrip(run_problem(engine, kit, table, problems[i], conditions, max_new)); assert row["id"] == problems[i]["id"]
            f.write(json.dumps(row) + "\n"); f.flush(); os.fsync(f.fileno()); rows.append(row)
            if progress: progress(i + 1, len(problems))
    return {"rows": rows, "summary": summarize_layout(rows), "meta": meta}


# ---------------------------------------------------------------- summaries and the reading rules
def sign_test(gain: int, loss: int) -> float:
    """Two-sided exact sign test p-value for `gain` problems better and `loss` worse (ties left out)."""
    n = gain + loss
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(0, min(gain, loss) + 1)) / 2 ** n)


def summarize_layout(rows) -> dict:
    conds = list(rows[0]["conds"]) if rows else []; out = {}
    for c in conds:
        dep = [r["conds"][c] for r in rows if r["kind"] == "delete_dep"]; ctl = [r["conds"][c] for r in rows if r["kind"] != "delete_dep"]
        out[c] = dict(n_delete=len(dep), n_control=len(ctl), exact_blind=_mean(x["exact_blind"] for x in dep), exact_m6=_mean(x["exact_m6"] for x in dep), admitted=_mean(x["admitted"] for x in dep),
                      recall=_mean(x["recall"] for x in dep), recall_out=_mean(x["recall_out"] for x in dep), recall_in=_mean(x["recall_in"] for x in dep), invented=_mean(x["invented"] for x in dep), other=_mean(x["other"] for x in dep),
                      decoded=_mean(len(x["tokens"]) for x in dep), ran_on=_mean(not x["ended"] for x in dep), control_exact=_mean(x["exact_blind"] for x in ctl), prompt_tokens=_mean(x["prompt_tokens"] for x in dep + ctl), canonical=_mean(x["canonical"] for x in dep + ctl),
                      seconds=_mean(x["seconds"] for x in dep + ctl), peak_mb=max([x["peak_mb"] for x in dep + ctl] or [0]))
        if c != "sorted" and "sorted" in conds:
            base = [r["conds"]["sorted"] for r in rows if r["kind"] == "delete_dep"]
            for m in ("exact_blind", "exact_m6"):
                gain = sum(1 for a, b in zip(dep, base) if a[m] and not b[m]); loss = sum(1 for a, b in zip(dep, base) if b[m] and not a[m]); out[c][m + "_vs_sorted"] = dict(gain=gain, loss=loss, p=sign_test(gain, loss))
            out[c]["same_text_as_sorted"] = _mean(a["text"] == b["text"] for a, b in zip(dep, base))
    return out


def verdict_layout(s: dict) -> List[Tuple[str, Optional[bool]]]:
    """The thirteen reading rules, in the order of LAYOUT_LINES."""
    so, sh, gr, ad, ro = (s.get(n, {}) for n in ("sorted", "shuffled", "grouped", "adjacency", "rot90")); diff = lambda a, b: None if a is None or b is None else a - b
    ge = lambda x, y, d: None if x is None or y is None else x - y >= d - 1e-12
    ok = [_ge(so.get("prompt_tokens"), Y_MIN_PROMPT), None if not s else all(c.get("canonical") == 1.0 for c in s.values()),
          None if not s else bool({c.get("n_delete") for c in s.values()} == {42} and {c.get("n_control") for c in s.values()} == {8}),
          None if so.get("recall") is None else Y_ROOM_LO - 1e-12 <= so["recall"] <= Y_ROOM_HI + 1e-12,
          ge(so.get("recall"), sh.get("recall"), Y_ORDER), ge(so.get("recall_out"), so.get("recall_in"), Y_CONTIG), ge(ad.get("recall"), so.get("recall"), Y_ADJ_RECALL), ge(ad.get("exact_blind"), so.get("exact_blind"), Y_ADJ_EXACT),
          ge(gr.get("recall"), so.get("recall"), Y_GROUPED), None if ro.get("recall") is None or so.get("recall") is None else abs(ro["recall"] - so["recall"]) <= Y_ROT + 1e-12,
          None if ad.get("invented") is None or so.get("invented") is None or not so["invented"] else ad["invented"] <= (1 - Y_INVENT) * so["invented"] + 1e-12,
          None if not s else all(c.get("control_exact") is not None and c["control_exact"] >= Y_CONTROL - 1e-12 for c in s.values()),
          _ge(None if sh.get("same_text_as_sorted") is None else 1 - sh["same_text_as_sorted"], Y_DIFFER)]
    return [(text, v) for (_, text), v in zip(LAYOUT_LINES, ok)]


def print_layout(summary: dict, title: str = ""):
    print(f"\n== layout test {title} ==")
    print(f"{'condition':10s} {'delete n':>8s} {'exact':>7s} {'+repair':>8s} {'recall':>7s} {'starts':>7s} {'ends':>6s} {'invented':>9s} {'cut off':>7s} {'control':>8s} {'prompt':>7s} {'sec':>6s}")
    for c, v in summary.items():
        print(f"{c:10s} {v['n_delete']:8d} {_pc(v['exact_blind']):>7s} {_pc(v['exact_m6']):>8s} {_fx(v['recall']):>7s} {_fx(v['recall_out']):>7s} {_fx(v['recall_in']):>6s} {_fx(v['invented']):>9s} {_pc(v['ran_on'], 0):>7s} {_pc(v['control_exact'], 0):>8s} {_fx(v['prompt_tokens'], 0):>7s} {_fx(v['seconds'], 1):>6s}")
    print("\nagainst sorted, problem by problem (better / worse, exact sign test):")
    for c, v in summary.items():
        if c == "sorted": continue
        a, b = v["exact_blind_vs_sorted"], v["exact_m6_vs_sorted"]
        print(f"  {c:10s} exact as written: {a['gain']} better, {a['loss']} worse (p={a['p']:.2f})   with repair: {b['gain']} better, {b['loss']} worse (p={b['p']:.2f})   same reply as sorted on {_pc(v['same_text_as_sorted'], 0)} of problems")


def print_verdict_layout(checks, title: str):
    print(f"\n-- {title} (reading rules fixed before the run) --")
    for i, (text, ok) in enumerate(checks, 1): print(("  yes      " if ok else "  not run  " if ok is None else "  NO       ") + f"{i:2d}. {text}")
