"""No-training agent-loop experiment: an off-the-shelf LLM proposes graph deltas; AXI wraps it.

Six modes per task (same prompt, greedy decoding):
  M0 unconstrained  + lenient parse + blind apply     (what a syntax-only / prompt-only agent does)
  M1 grammar-constrained + blind apply                (constrained decoding alone)
  M2 constrained + gate                               (reject on obstruction, no retry)
  M3 constrained + gate + 1 retry, feedback = {prose_plain, prose_detailed, structured}
  M5 constrained + gate + deterministic auto-repair   (no second LLM call)
Success = final graph equals the expected graph exactly (right change, no collateral, well-formed).
"""
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from axi.engine.cico_parser import (CICOParser, CICOParseError, NodeDelta, EdgeDelta, StateTransitionDelta,
                                    NODE_ADD, NODE_DEL, EDGE)
from axi.engine.gate import Graph, check, apply
from axi.engine.diagnostics import diagnose, propose_repair, format_delta

SYSTEM = """You maintain a service graph. Services sit at grid coordinates and are written [row,col:value]. Edges are written (r,c)->(r,c):relation#weight.
Reply with ONLY a state delta and no explanation. A delta is a list of items separated by single spaces:
  ADD[r,c:value]                        add a service
  DEL[r,c]                              remove a service
  ADD[(r,c)->(r,c):relation#weight]     add an edge
  DEL[(r,c)->(r,c):relation#weight]     remove an edge (relation and weight must match exactly)
Examples:
  add a service at (4,4) with value 3 -> ADD[4,4:3]
  remove the edge (0,0)->(1,1) dep weight 2 -> DEL[(0,0)->(1,1):dep#2]"""
RULES_HINT = """
Rule: a service can only be removed in the same delta that removes every edge attached to it.
  example: remove service (2,2) which has edge (0,0)->(2,2) dep weight 1 -> DEL[2,2] DEL[(0,0)->(2,2):dep#1]"""


# ---------------------------------------------------------------- graphs & tasks
def copy_graph(g: Graph) -> Graph:
    h = Graph()
    for n, v in g.nodes.items(): h.add_node(n, v)
    for (u, v, r), w in g.edges.items(): h.add_edge(u, v, r, w)
    return h


def same_graph(a: Graph, b: Graph) -> bool: return a.nodes == b.nodes and a.edges == b.edges


def blind_apply(g: Graph, d: StateTransitionDelta) -> Graph:
    """Apply a delta with NO relational checks (what a syntax-only system effectively does)."""
    h = copy_graph(g)
    for e in d.edge_deltas:
        k = (e.u, e.v, e.relation)
        if e.op == "DEL": h.edges.pop(k, None)
    for n in d.node_deltas:
        if n.op == "DEL": h.nodes.pop((n.r, n.c), None)
        else: h.nodes[(n.r, n.c)] = n.val
    for e in d.edge_deltas:
        if e.op == "ADD": h.edges[(e.u, e.v, e.relation)] = e.weight
    return h


@dataclass
class Task:
    kind: str                 # delete_dep | delete_leaf | add_node | add_edge
    graph: Graph
    instruction: str
    ideal: str                # CICO text of a correct delta
    expected: Graph


def _serialize(g: Graph) -> str:
    return CICOParser.serialize_state([(r, c, v) for (r, c), v in sorted(g.nodes.items())],
                                      [(u[0], u[1], v[0], v[1], rel, w) for (u, v, rel), w in sorted(g.edges.items())])


def make_task(rng: random.Random, kind: str) -> Task:
    while True:
        cells = [(r, c) for r in range(3) for c in range(4)]; rng.shuffle(cells); nodes = cells[:8]
        g = Graph()
        for n in nodes: g.add_node(n, rng.randint(1, 9))
        core = nodes[:7]                                    # nodes[7] stays isolated
        for _ in range(rng.randint(7, 10)):
            u, v = rng.sample(core, 2); rel = rng.choice(["dep", "owns"])
            if (u, v, rel) not in g.edges: g.add_edge(u, v, rel, rng.randint(1, 9))
        deg = {n: len(g.incident(n)) for n in nodes}
        if kind == "delete_dep":
            cand = [n for n in nodes if 1 <= deg[n] <= 3]
            if not cand: continue
            n = rng.choice(cand)
            ideal = "DEL[%d,%d] " % n + " ".join(f"DEL[({u[0]},{u[1]})->({v[0]},{v[1]}):{r}#{g.edges[(u, v, r)]}" + "]" for (u, v, r) in sorted(g.incident(n)))
            instr = f"Delete the service at ({n[0]},{n[1]})."
        elif kind == "delete_leaf":
            n = nodes[7]; ideal = "DEL[%d,%d]" % n; instr = f"Delete the service at ({n[0]},{n[1]})."
        elif kind == "add_node":
            free = [c for c in [(r, c) for r in range(5) for c in range(6)] if c not in g.nodes]; n = rng.choice(free); v = rng.randint(1, 9)
            ideal = f"ADD[{n[0]},{n[1]}:{v}]"; instr = f"Add a service at ({n[0]},{n[1]}) with value {v}."
        else:
            while True:
                u, v = rng.sample(nodes, 2)
                if (u, v, "dep") not in g.edges: break
            w = rng.randint(1, 9); ideal = f"ADD[({u[0]},{u[1]})->({v[0]},{v[1]}):dep#{w}]"
            instr = f"Add a dependency edge from ({u[0]},{u[1]}) to ({v[0]},{v[1]}) with relation dep and weight {w}."
        d = CICOParser.parse_transition_delta(ideal); assert check(g, d).ok, (kind, ideal)
        exp = copy_graph(g); apply(exp, d)
        return Task(kind, g, instr, ideal, exp)


def make_tasks(n: int, seed: int = 0) -> List[Task]:
    rng = random.Random(seed)
    kinds = (["delete_dep"] * 5 + ["delete_leaf"] * 2 + ["add_node"] + ["add_edge"] * 2)
    return [make_task(rng, kinds[i % len(kinds)]) for i in range(n)]


# ---------------------------------------------------------------- prompts & feedback
def base_messages(task: Task, hint_rules: bool) -> List[dict]:
    return [{"role": "system", "content": SYSTEM + (RULES_HINT if hint_rules else "")},
            {"role": "user", "content": f"{_serialize(task.graph)}\n\nRequest: {task.instruction}"}]


def _edge_txt(e): u, v, r, w = e; return f"DEL[({u[0]},{u[1]})->({v[0]},{v[1]}):{r}#{w}]"


def feedback_text(variant: str, obs) -> str:
    tail = " Reply with only the corrected delta."
    if variant == "structured":
        parts = []
        for o in obs:
            if o.kind == "dangling": parts.append(f"OBSTRUCTION dangling node=({o.node[0]},{o.node[1]}) attached: " + " ".join(_edge_txt(e) for e in o.edges))
            else: parts.append(f"OBSTRUCTION {o.kind} " + (f"node=({o.node[0]},{o.node[1]})" if o.node else f"edge={o.edge}"))
        return "\n".join(parts) + tail
    if variant == "structured_imperative":
        parts = []
        for o in obs:
            if o.kind == "dangling":
                parts.append(f"OBSTRUCTION dangling node=({o.node[0]},{o.node[1]})\nREQUIRED: the delta must also delete these edges: "
                             + " ".join(_edge_txt(e) for e in o.edges) + f"\nKeep DEL[{o.node[0]},{o.node[1]}] in the delta.")
            else: parts.append(f"OBSTRUCTION {o.kind} " + (f"node=({o.node[0]},{o.node[1]})" if o.node else f"edge={o.edge}"))
        return "\n".join(parts) + tail
    if variant == "prose_plain":
        return "That change was rejected because it would break the graph (for example removing a service that others still depend on)." + tail
    if variant == "prose_detailed":
        parts = []
        for o in obs:
            if o.kind == "dangling":
                es = ", ".join(f"({u[0]},{u[1]})->({v[0]},{v[1]}) {r} weight {w}" for (u, v, r, w) in o.edges)
                parts.append(f"You cannot remove service ({o.node[0]},{o.node[1]}) because these edges still reference it: {es}. They must be removed in the same delta.")
            else:
                parts.append(f"The change was rejected: {o.kind.replace('_', ' ')} problem at {o.node or o.edge}.")
        return " ".join(parts) + tail
    raise ValueError(variant)


# ---------------------------------------------------------------- parsing
def lenient_parse(text: str) -> Optional[StateTransitionDelta]:
    """What a pragmatic agent does with chatty output: pull out every well-formed item anywhere in the text."""
    nodes, edges = [], []
    for m in NODE_ADD.finditer(text): nodes.append(NodeDelta("ADD", int(m[1]), int(m[2]), int(m[3])))
    for m in NODE_DEL.finditer(text): nodes.append(NodeDelta("DEL", int(m[1]), int(m[2])))
    for m in EDGE.finditer(text): edges.append(EdgeDelta(m[1], (int(m[2]), int(m[3])), (int(m[4]), int(m[5])), m[6], int(m[7])))
    return StateTransitionDelta(nodes, edges) if nodes or edges else None


def strict_parse(text: str) -> Optional[StateTransitionDelta]:
    try: return CICOParser.parse_transition_delta(text.strip()) if text.strip() else None
    except CICOParseError: return None


# ---------------------------------------------------------------- the experiment
def _rec(g_before, final, task, **kw):
    return dict(corrupted=not final.is_well_formed(), success=same_graph(final, task.expected), **kw)


def run_task(llm, task: Task, hint_rules=False, degree_threshold=4) -> dict:
    msgs = base_messages(task, hint_rules); out = {"kind": task.kind}
    # M0
    t0, n0, s0 = llm.generate(msgs, constrained=False)
    strict0 = strict_parse(t0) is not None; d0 = lenient_parse(t0)
    final = blind_apply(task.graph, d0) if d0 else copy_graph(task.graph)
    out["M0_unconstrained"] = _rec(task.graph, final, task, strict_parse=strict0, lenient_parse=d0 is not None, tokens=n0, seconds=s0, text=t0)
    # M1
    t1, n1, s1 = llm.generate(msgs, constrained=True); d1 = strict_parse(t1)
    final = blind_apply(task.graph, d1) if d1 else copy_graph(task.graph)
    out["M1_constrained"] = _rec(task.graph, final, task, strict_parse=d1 is not None, tokens=n1, seconds=s1, text=t1)
    # M2 gate
    if d1 is None:
        rej, final, obs = True, copy_graph(task.graph), []
    else:
        res = check(task.graph, d1); rej = not res.ok; obs = diagnose(task.graph, d1)
        final = copy_graph(task.graph)
        if res.ok: apply(final, d1)
    out["M2_gate"] = _rec(task.graph, final, task, rejected=rej, obstructions=[o.kind for o in obs])
    # M3 retries (only when the gate rejected a parseable proposal)
    for variant in ("prose_plain", "prose_detailed", "structured", "structured_imperative"):
        key = f"M3_retry_{variant}"
        if not rej or d1 is None:
            out[key] = dict(out["M2_gate"], retried=False, tokens=0); continue
        m2 = msgs + [{"role": "assistant", "content": t1}, {"role": "user", "content": feedback_text(variant, obs)}]
        t, n, s = llm.generate(m2, constrained=True); d = strict_parse(t); final = copy_graph(task.graph); rej2 = True
        if d is not None and check(task.graph, d).ok: apply(final, d); rej2 = False
        out[key] = _rec(task.graph, final, task, rejected=rej2, retried=True, tokens=n, seconds=s, text=t)
    # M5 deterministic auto-repair (no second LLM call)
    final = copy_graph(task.graph); rep = propose_repair(task.graph, d1, degree_threshold) if d1 is not None else None
    if d1 is not None and not rej: apply(final, d1)
    elif rep is not None and rep.repairable and check(task.graph, rep.repaired).ok: apply(final, rep.repaired)
    out["M5_auto_repair"] = _rec(task.graph, final, task, rejected=False, llm_calls_extra=0)
    return out


def run_experiment(llm, tasks, hint_rules=False, progress=None) -> dict:
    rows = []
    for i, t in enumerate(tasks):
        rows.append(run_task(llm, t, hint_rules))
        if progress: progress(i + 1, len(tasks))
    return {"rows": rows, "summary": summarize(rows)}


MODES = ["M0_unconstrained", "M1_constrained", "M2_gate", "M3_retry_prose_plain", "M3_retry_prose_detailed", "M3_retry_structured", "M3_retry_structured_imperative", "M5_auto_repair"]


def summarize(rows) -> dict:
    s = {}
    for scope, sel in (("all", lambda r: True), ("delete_dep", lambda r: r["kind"] == "delete_dep")):
        rs = [r for r in rows if sel(r)]; n = max(len(rs), 1); d = {"n": len(rs)}
        for m in MODES:
            x = [r[m] for r in rs]
            d[m] = dict(success=sum(v["success"] for v in x) / n, corrupted=sum(v["corrupted"] for v in x) / n,
                        strict_parse=(sum(v.get("strict_parse", False) for v in x) / n) if "strict_parse" in x[0] else None,
                        avg_tokens=(sum(v.get("tokens", 0) for v in x) / n) if x and "tokens" in x[0] else None)
        d["gate_blocked_blind_successes"] = sum(1 for r in rs if r["M1_constrained"]["success"] and r["M2_gate"].get("rejected"))
        s[scope] = d
    return s


def print_summary(summary):
    for scope, d in summary.items():
        print(f"\n== {scope} (n={d['n']}) ==   mode | task success | corrupted graph | strict-parse")
        for m in MODES:
            v = d[m]; sp = "-" if v["strict_parse"] is None else f"{v['strict_parse']*100:5.1f}%"
            print(f"  {m:32s} {v['success']*100:6.1f}%   {v['corrupted']*100:6.1f}%   {sp}")
        print(f"  (tasks where blind apply was right but the gate rejected: {d['gate_blocked_blind_successes']})")
