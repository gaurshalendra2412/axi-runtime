"""The RESULT of docs/MILLION_TOKEN_TEST_HARD.md (concept-map item 95), recomputed from the saved files (run 1, the saved-reply file, the contrast), and the renderers of its tables. It calls no model and uses no key.

It was written AFTER run 1 and is not part of the run: `million.py` and `million_hard.py` (what the run used) were not touched by it. What it adds to the page's own reading rules
(`million_hard.verdict_hard`, which it only calls) is what those rules do not show, counted from the saved rows:
  * `analyse`: the stream is replayed over the saved replies and, for every step, the saved reply is compared with the ideal reply of the request (the same text, the same items in another order,
    or other items: left out, added or changed), and what each policy would have done with that reply STARTING FROM THE ORACLE'S STATE (`counterfactual`: one step, not the contrast, which is a
    separate live run of those policies on their own held states);
  * `cache_facts`: what the saved reply file says about the calls (how they ended, how many were retried after a busy server);
  * `analyse_contrast`: the contrast (blind, gate and repair_m6 on the first 400 requests, live runs on their own held states) read next to the first 400 steps of run 1, and `stretches` (the runs of steps for which a flag holds);
  * `table_*`: the tables of the Result, rendered from the above, so a number on the page cannot be edited without `tests/test_million_hard_result.py` failing.
"""
import json
import os
from typing import Dict, List

from axi.engine.cico_parser import CICOParser
from axi.experiments import million as M
from axi.experiments import million_hard as H
from axi.experiments.agent_loop import copy_graph, same_graph, strict_parse

T = CICOParser.parse_transition_delta
RESULTS = os.path.join(os.path.dirname(__file__), "..", "..", "..", "docs", "results")
RUN1_FILE = "axi_million_hard_run1_seed20261031_default.json"
CACHE1_FILE = "axi_million_hard_cache_after_run1.jsonl"       # the saved-reply file as it was when run 1 ended (before the contrast)
PRICE_IN, PRICE_OUT = 0.75, 3.75                              # dollars per million tokens (input; output including thinking): the Standard prices the owner read on Google's page and typed in; double from 1 Jan 2027
CONTRAST_FILE = "axi_million_hard_contrast_seed20261031_default.json"          # {"blind": run, "gate": run, "repair_m6": run}, 400 steps each, the registered settings, the main run's seed
POLICIES = ("blind", "gate", "repair_m6", "scope_m7")
ARMS = ("scope_m7", "blind", "gate", "repair_m6")                           # the order of the contrast table
KIND_ORDER = ("add_edge", "add_node", "delete_hub", "delete_dep", "delete_leaf", "delete_edge", "delete_pair")
KIND_NAMES = {"add_node": "add a service", "add_edge": "add an edge", "delete_leaf": "delete a service without dependants", "delete_edge": "delete an edge",
              "delete_dep": "delete a service with 1 to 3 dependants", "delete_hub": "delete a service with 4 to 8 dependants (hub)", "delete_pair": "delete a service and an edge"}


def load(name: str = RUN1_FILE) -> dict:
    with open(os.path.join(RESULTS, name)) as f: return json.load(f)


def load_contrast() -> dict:
    return load(CONTRAST_FILE)


def stretches(rows: list, key) -> list:
    """The runs of consecutive steps for which key(row) holds, as (first step, last step); a run still going at the last row ends there."""
    out, start = [], None
    for r in rows:
        if key(r) and start is None: start = r["t"]
        if not key(r) and start is not None: out.append((start, r["t"] - 1)); start = None
    if start is not None: out.append((start, rows[-1]["t"]))
    return out


def _items(d) -> list:
    """The items of a parsed reply, sorted: two replies with the same items in another order are equal."""
    return sorted(repr(x) for x in d.node_deltas) + sorted(repr(x) for x in d.edge_deltas)


def _mean(xs):
    xs = list(xs); return sum(xs) / len(xs) if xs else None


def money(prompt_tokens: int, output_tokens: int) -> float:
    """Dollars at the typed-in prices (arithmetic on the owner's prices, not a result of the page)."""
    return (prompt_tokens * PRICE_IN + output_tokens * PRICE_OUT) / 1e6


def analyse(run: dict, degree_threshold: int = H.HARD_DEGREE) -> dict:
    """Everything the Result says about one run that the reading rules do not, from the saved rows (the stream is replayed; no call is made)."""
    rows, meta = run["rows"], run["meta"]; stream = M.make_stream(meta["stream"], meta["seed"])
    kinds: Dict[str, dict] = {}; pair: Dict[int, list] = {}
    cf = {p: [0, 0] for p in POLICIES}; cf_default = {p: 0 for p in ("repair_m6", "scope_m7")}; only_left_out = 0
    for rec in rows:
        st, ob = stream.next(); oa = stream.live
        d = strict_parse(rec["text"]); ideal = T(st["ideal"])
        k = kinds.setdefault(rec["kind"], dict(requests=0, same_text=0, other_order=0, other_items=0, off=0, think=0))
        k["requests"] += 1; k["off"] += 0 if rec["exact"] else 1; k["think"] += rec["think"]
        if rec["text"].strip() == st["ideal"].strip(): k["same_text"] += 1
        elif d is not None and _items(d) == _items(ideal): k["other_order"] += 1
        else:
            k["other_items"] += 1
            if d is not None and set(_items(d)) < set(_items(ideal)): only_left_out += 1               # every item of the reply is in the ideal reply, some are missing
        if rec["kind"] == "delete_pair":
            n = ideal.node_deltas[0]; deps = len(ob.incident((n.r, n.c))); p = pair.setdefault(deps, [0, 0])
            p[0] += 1; p[1] += 0 if (rec["text"].strip() == st["ideal"].strip() or (d is not None and _items(d) == _items(ideal))) else 1
        for p in POLICIES:                                                                               # one step, from the oracle's state, with this reply
            res = M.next_state(p, copy_graph(ob), d, st["instruction"], degree_threshold=degree_threshold)
            cf[p][0] += 0 if same_graph(res, oa) else 1; cf[p][1] += 0 if res.is_well_formed() else 1
        for p in cf_default:                                                                             # the same with the framework's own default of 4 dependants
            res = M.next_state(p, copy_graph(ob), d, st["instruction"], degree_threshold=4); cf_default[p] += 0 if same_graph(res, oa) else 1
    th = sorted(r["think"] for r in rows); sec = [r["seconds"] for r in rows]
    tenths = []
    for i in range(M.BLOCKS):
        b = M.block(rows, i)
        tenths.append(dict(first=b[0]["t"], last=b[-1]["t"], steps=len(b), exact=_mean(r["exact"] for r in b), distance=_mean(r["distance"] for r in b), prompt=_mean(r["prompt_tokens"] for r in b),
                           think=_mean(r["think"] for r in b), seconds=_mean(r["seconds"] for r in b), nodes=_mean(r["nodes"] for r in b), edges=_mean(r["edges"] for r in b)))
    prompt, visible, think = (sum(r[k] for r in rows) for k in ("prompt_tokens", "tokens", "think"))
    return dict(steps=len(rows), kinds=kinds, pair=pair, counterfactual=cf, counterfactual_default=cf_default, only_left_out=only_left_out, tenths=tenths,
                tokens=dict(prompt=prompt, visible=visible, think=think, total=M.tokens_sent(rows), calls=sum(r.get("asks", 1) for r in rows)),
                think=dict(mean=sum(th) / len(th), median=th[len(th) // 2], p95=th[int(0.95 * len(th))], max=th[-1], over_1000=sum(1 for x in th if x > 1000)),
                seconds=dict(total=sum(sec), mean=sum(sec) / len(sec), max=max(sec), over_10=sum(1 for x in sec if x > 10)),
                dollars=dict(total=money(prompt, visible + think), thinking=money(0, think)))


def cache_facts(path: str) -> dict:
    """What the saved-reply file says: the model and settings it belongs to, how many calls it holds and how they ended, the retries after a busy server (a line with a 'fail' key)."""
    with open(path) as f: lines = [json.loads(x) for x in f if x.strip()]
    ok = [x for x in lines[1:] if "fail" not in x]
    return dict(meta=lines[0]["_meta"], calls=len(ok), distinct=len({x["k"] for x in ok}), stop=sum(1 for x in ok if x["finish"] == "FinishReason.STOP"), normalized=sum(1 for x in ok if x["normalized"]),
                retries=sum(1 for x in lines[1:] if x.get("fail") == "retry"), prompt=sum(x["prompt_tokens"] for x in ok), visible=sum(x["tokens"] for x in ok), think=sum(x["think"] for x in ok))


def analyse_contrast(run1: dict, contrast: Dict[str, dict]) -> dict:
    """Each arm of the contrast over the first H.HARD_CONTRAST_STEPS requests: scope_m7 is run 1's own first steps; blind, gate and repair_m6 are the live runs. Counted from the saved rows. The replay of each arm's replies
    (every saved flag reproduced) is not done here; the test does it."""
    n = H.HARD_CONTRAST_STEPS; out = {}
    for arm in ARMS:
        rows = run1["rows"][:n] if arm == "scope_m7" else contrast[arm]["rows"]; assert len(rows) == n
        live = [r for r in rows if not r["cached"]]
        tail = rows[n - 100:]
        out[arm] = dict(steps=n, exact=sum(1 for r in rows if r["exact"]), corrupted=sum(1 for r in rows if r["corrupted"]), refused=sum(1 for r in rows if not r["parsed"]),
                        mean_all=_mean(r["distance"] for r in rows), mean_201_400=_mean(r["distance"] for r in rows[200:n]), max_distance=max(r["distance"] for r in rows), last100_distance=_mean(r["distance"] for r in tail),
                        last100_exact=sum(1 for r in tail if r["exact"]), new_offsets=[r["t"] for r in rows if r["start_same"] and not r["exact"]], new_offset_kinds=sorted({r["kind"] for r in rows if r["start_same"] and not r["exact"]}),
                        inapplicable=sum(1 for r in rows if not r["applicable"]), reused=None if arm == "scope_m7" else n - len(live), live_calls=None if arm == "scope_m7" else len(live),
                        tokens=None if arm == "scope_m7" else M.tokens_sent(rows), dollars=None if arm == "scope_m7" else money(sum(r["prompt_tokens"] for r in live), sum(r["tokens"] + r["think"] for r in live)),
                        off=stretches(rows, lambda r: not r["exact"]), corrupted_stretches=stretches(rows, lambda r: r["corrupted"]))
    return out


def analyse_cost(run1: dict, contrast: Dict[str, dict]) -> dict:
    """What each arm of the contrast cost over the first H.HARD_CONTRAST_STEPS requests, with every call counted, and what a step that held the right state cost. Added after the contrast was read; arithmetic on the saved rows, not a reading rule.
    A reused reply is a call that was really made and billed, in run 1 (its own token counts are in the row and in the saved-reply file: the test checks it), so each arm is priced as if all of its calls had been billed, not only the live ones.
    Prices: PRICE_IN and PRICE_OUT, thinking counted as output. 'times_framework' is the arm's dollars per step with the right state divided by scope_m7's."""
    n = H.HARD_CONTRAST_STEPS; out = {}
    for arm in ARMS:
        rows = run1["rows"][:n] if arm == "scope_m7" else contrast[arm]["rows"]; parts = [q for r in rows for q in M._parts(r)]
        prompt, visible, think = (sum(q[k] for q in parts) for k in ("prompt_tokens", "tokens", "think")); exact = sum(1 for r in rows if r["exact"]); assert exact > 0
        tokens = prompt + visible + think; dollars = money(prompt, visible + think)
        out[arm] = dict(steps=len(rows), calls=len(parts), prompt=prompt, visible=visible, think=think, tokens=tokens, dollars=dollars, exact=exact, tokens_per_exact=tokens / exact, dollars_per_exact=dollars / exact)
    for arm in ARMS: out[arm]["times_framework"] = out[arm]["dollars_per_exact"] / out["scope_m7"]["dollars_per_exact"]
    return out


def blind_stretch_ends(contrast: Dict[str, dict]) -> list:
    """How each stretch of corrupted steps of the blind arm ended: replay the arm over its saved replies and, at each step after which no edge points at a missing service (and some did before it), say what the request was and why:
    'service added back' (the request added a service at a coordinate the leftover edges pointed at: the state is well formed again and still wrong) or 'leftover edges deleted' (the reply deleted every leftover edge).
    Returns [(step, kind of request, cause)]; a stretch still going at the last step has no entry."""
    d = contrast["blind"]; rows, meta = d["rows"], d["meta"]; stream = M.make_stream(meta["stream"], meta["seed"]); live = copy_graph(stream.start); out = []
    dangling = lambda g: sorted((u, v, r) for (u, v, r) in g.edges if u not in g.nodes or v not in g.nodes)
    for rec in rows:
        st, ob = stream.next(); delta = strict_parse(rec["text"]); before = dangling(live)
        live = M.next_state("blind", copy_graph(live), delta, st["instruction"], degree_threshold=meta["degree_threshold"]); after = dangling(live)
        if before and not after:
            ends = {x for (u, v, r) in before for x in (u, v)}                                                    # the coordinates the leftover edges touch
            added = {(n.r, n.c) for n in delta.node_deltas if n.op == "ADD"}
            deleted = {(e.u, e.v, e.relation) for e in delta.edge_deltas if e.op == "DEL"}
            cause = "service added back" if added & ends else "leftover edges deleted" if set(before) <= deleted else "other"
            out.append((rec["t"], rec["kind"], cause))
    return out


# ================================================================ the Result tables
def _pc(x, k=1): return f"{100 * x:.{k}f}%"


def lines_block(v: List[tuple]) -> str:
    return "\n".join(f"{i}. {ans} [{tag}] {text}" for i, (tag, text, ans) in enumerate(v, 1))


def table_tenths(a: dict) -> str:
    out = ["| tenth | steps | share exact | mean distance | mean prompt tokens | mean thinking tokens | mean seconds | mean services | mean edges |", "|---|---|---|---|---|---|---|---|---|"]
    for i, t in enumerate(a["tenths"], 1):
        out.append(f"| {i} ({t['first']:,} to {t['last']:,}) | {t['steps']} | {_pc(t['exact'])} | {t['distance']:.2f} | {t['prompt']:.1f} | {t['think']:.1f} | {t['seconds']:.2f} | {t['nodes']:.1f} | {t['edges']:.1f} |")
    return "\n".join(out)


def table_kinds(a: dict) -> str:
    out = ["| request | requests | reply = the ideal reply | same items, other order | other items | steps ending off the oracle's state | mean thinking tokens |", "|---|---|---|---|---|---|---|"]
    tot = dict(requests=0, same_text=0, other_order=0, other_items=0, off=0, think=0)
    for kind in KIND_ORDER:
        k = a["kinds"][kind]
        out.append(f"| {KIND_NAMES[kind]} | {k['requests']:,} | {k['same_text']:,} | {k['other_order']} | {k['other_items']} | {k['off']} | {k['think'] / k['requests']:.1f} |")
        for key in tot: tot[key] += k[key]
    out.append(f"| all | {tot['requests']:,} | {tot['same_text']:,} | {tot['other_order']} | {tot['other_items']} | {tot['off']} | {tot['think'] / tot['requests']:.1f} |")
    return "\n".join(out)


def table_dependants(a: dict) -> str:
    out = ["| dependants of the service to delete | delete-service-and-edge requests | replies that left some of them out | share |", "|---|---|---|---|"]
    for deps in sorted(a["pair"]):
        n, bad = a["pair"][deps]; out.append(f"| {deps} | {n} | {bad} | {_pc(bad / n, 0)} |")
    n, bad = sum(v[0] for v in a["pair"].values()), sum(v[1] for v in a["pair"].values()); out.append(f"| all | {n} | {bad} | {_pc(bad / n, 0)} |")
    return "\n".join(out)


ARM_NAMES = {"scope_m7": "scope_m7, the framework (run 1's first 400 steps)", "blind": "blind: the replies applied as written", "gate": "gate: applied only if the gate accepts, otherwise nothing", "repair_m6": "repair_m6: the repair alone, without the scope filter"}


def table_contrast(c: dict) -> str:
    names = ARM_NAMES
    out = ["| policy, first 400 requests | steps exactly the oracle's | steps corrupted | mean distance, steps 201 to 400 | mean distance, all 400 | largest distance | steps that started right and ended off | requests that could not be applied as written | replies reused from run 1 | tokens sent by the arm |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        x = c[arm]; out.append(f"| {names[arm]} | {x['exact']} | {x['corrupted']} | {x['mean_201_400']:.2f} | {x['mean_all']:.2f} | {x['max_distance']} | {len(x['new_offsets'])} | {x['inapplicable']} | "
                               + ("-" if x["reused"] is None else str(x["reused"])) + " | " + ("-" if x["tokens"] is None else f"{x['tokens']:,}") + " |")
    return "\n".join(out)


def table_counterfactual(a: dict) -> str:
    out = ["| what is done with the reply | steps (of 4,000) ending off the oracle's state | of which with a corrupted state |", "|---|---|---|"]
    names = {"blind": "applied as written (blind)", "gate": "applied only if the gate accepts it (gate)", "repair_m6": "repaired by the deterministic completion, deleting up to 8 dependants (repair_m6)",
             "scope_m7": "the scope filter, then the same repair (scope_m7, the framework as run)"}
    for p in POLICIES: out.append(f"| {names[p]} | {a['counterfactual'][p][0]} | {a['counterfactual'][p][1]} |")
    return "\n".join(out)


def table_cost(c: dict) -> str:
    out = ["| policy, first 400 requests | steps exactly the oracle's | tokens, every call counted | dollars, every call counted | tokens per step with the right state | dollars per step with the right state | times the framework's dollars per step with the right state |", "|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        x = c[arm]; out.append(f"| {ARM_NAMES[arm]} | {x['exact']} | {x['tokens']:,} | ${x['dollars']:.2f} | {x['tokens_per_exact']:,.0f} | ${x['dollars_per_exact']:.5f} | {x['times_framework']:.1f} |")
    return "\n".join(out)
