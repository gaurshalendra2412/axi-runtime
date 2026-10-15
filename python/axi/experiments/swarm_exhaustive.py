"""Added AFTER the result of docs/SWARM_TEST.md was in (so NOT pre-registered; the page says so): is the independence rule exact when EVERY pair of edits on a tiny universe is checked, instead of a random draw?

The universe: three cells (0,0) (0,1) (1,0); every graph on them (80: each subset of the cells as services, each set of directed `dep` edges between them, all weights 1); on each graph every edit of one to
`max_ops` operations that the gate accepts (add a service with value 1 or 2 where there is none, delete one that is there, add a `dep` edge with weight 1 or 2 where there is none, delete one that is
there with its weight), in every combination; every ordered pair of two such edits (the two may be the same edit). For each pair the truth is the one of the page (apply both orders through the real gate;
they commute when both are accepted and the graphs are equal) and the three detectors of the page (`dpo`, `pager`, `names`) say clash or fine. Nothing is drawn: the numbers are counts over all pairs.

What this can and cannot show: exactness of the rule on THIS universe, with three services and edits of at most `max_ops` operations (a service with three edges cannot be deleted in one edit of three
operations, so the biggest compound edits of the random runs are out of reach); it is not a proof for the language, and a bigger universe would be more convincing. Run: python -m axi.experiments.swarm_exhaustive."""
import argparse
import json
import sys
import time
from collections import Counter
from itertools import combinations
from typing import Iterator, List, Sequence, Tuple

from axi.engine.cico_parser import StateTransitionDelta
from axi.engine.gate import Graph, check
from axi.engine.inverse import to_text
from axi.experiments import swarm as W

CELLS = ((0, 0), (0, 1), (1, 0))
MAX_OPS = 3
NODE_VALUES, EDGE_WEIGHTS = (1, 2), (1, 2)
DETECTORS = ("dpo", "pager", "names")


def graphs(cells: Sequence[tuple] = CELLS) -> Iterator[Graph]:
    """Every graph whose services are a subset of `cells` and whose edges are any set of directed `dep` edges (weight 1) between distinct services."""
    for k in range(len(cells) + 1):
        for services in combinations(cells, k):
            pairs = [(u, v) for u in services for v in services if u != v]
            for mask in range(2 ** len(pairs)):
                g = Graph()
                for n in services: g.add_node(n, 1)
                for i, (u, v) in enumerate(pairs):
                    if (mask >> i) & 1: g.add_edge(u, v, "dep", 1)
                yield g


def operations(g: Graph, cells: Sequence[tuple] = CELLS) -> List[tuple]:
    """The single operations that can be part of an edit the gate accepts on g: ('n', ...) and ('e', ...)."""
    out = []
    for n in cells:
        if n in g.nodes: out.append(("n", W._nd("DEL", n)))
        else: out += [("n", W._nd("ADD", n, v)) for v in NODE_VALUES]
    for u in cells:
        for v in cells:
            if u == v: continue
            k = (u, v, "dep")
            if k in g.edges: out.append(("e", W._ed("DEL", k, g.edges[k])))
            else: out += [("e", W._ed("ADD", k, w)) for w in EDGE_WEIGHTS]
    return out


def edits(g: Graph, max_ops: int = MAX_OPS, cells: Sequence[tuple] = CELLS) -> List[StateTransitionDelta]:
    """Every edit of 1 to max_ops operations that the gate accepts on g."""
    ops = operations(g, cells); res = []
    for r in range(1, max_ops + 1):
        for combo in combinations(ops, r):
            d = StateTransitionDelta([x for t, x in combo if t == "n"], [x for t, x in combo if t == "e"])
            if check(g, d).ok: res.append(d)
    return res


def run_exhaustive(max_ops: int = MAX_OPS, cells: Sequence[tuple] = CELLS, rule=W.independent, progress=None, limit_graphs: int = 0) -> dict:
    """Counts over every ordered pair of accepted edits on every graph of the universe; `rule` is the independence rule under test."""
    tally = {d: Counter() for d in DETECTORS}; misses = {d: Counter() for d in DETECTORS}; examples = {d: [] for d in DETECTORS}; why = Counter(); graphs_seen = pairs = 0; sizes = []
    for g in graphs(cells):
        if limit_graphs and graphs_seen >= limit_graphs: break
        graphs_seen += 1; es = edits(g, max_ops, cells); sizes.append(len(es)); feet = [W.footprint(d) for d in es]; tmpl = W.pager_template(g)
        for i, a in enumerate(es):
            for j, b in enumerate(es):
                pairs += 1; t = W.commutes(g, a, b); conflict = not t["commute"]
                if conflict: why["one order only" if t["ab"] != t["ba"] else "both orders refused" if not t["ab"] else "orders differ"] += 1
                said = {"dpo": not rule(feet[i], feet[j]), "pager": W.pager_clash(tmpl, a, b)[1], "names": W.names_clash(feet[i], feet[j])}
                for det, flagged in said.items():
                    cell = ("tp" if flagged else "fn") if conflict else ("fp" if flagged else "tn"); tally[det][cell] += 1
                    if cell in ("fn", "fp"):
                        misses[det][cell + " " + "/".join(sorted([W.shape(a), W.shape(b)]))] += 1
                        if len(examples[det]) < 3: examples[det].append(dict(cell=cell, graph=sorted(g.nodes) + sorted((k[0], k[1]) for k in g.edges), a=to_text(a), b=to_text(b)))
        if progress: progress(graphs_seen)
    conflict = sum(tally["dpo"][c] for c in ("tp", "fn"))
    return dict(max_ops=max_ops, cells=[list(c) for c in cells], graphs=graphs_seen, edits_per_graph_max=max(sizes or [0]), pairs=pairs, conflict=conflict, commute=pairs - conflict, why=dict(why),
                detectors={d: {c: tally[d].get(c, 0) for c in ("tp", "fn", "fp", "tn")} for d in DETECTORS}, misses={d: dict(misses[d].most_common(10)) for d in DETECTORS}, examples=examples)


def print_exhaustive(res: dict):
    print(f"\n== every pair on a tiny universe: {res['graphs']} graphs, edits of 1 to {res['max_ops']} operations, {res['pairs']} ordered pairs, {res['conflict']} do not commute, {res['commute']} commute; why: {res['why']}")
    for d in DETECTORS:
        t = res["detectors"][d]; print(f"   {d:6s} caught {t['tp']:7d}  MISSED {t['fn']:7d}  false alarms {t['fp']:7d}  right 'fine' {t['tn']:7d}")
    for d in DETECTORS:
        if res["misses"][d]: print(f"   {d}: " + ", ".join(f"{k}: {v}" for k, v in list(res["misses"][d].items())[:6]))
    for d in DETECTORS:
        for ex in res["examples"][d][:1]: print(f"   first {d} {ex['cell']}: on {ex['graph']}: {ex['a']}   |   {ex['b']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Every pair of edits on a tiny universe (added after the result of docs/SWARM_TEST.md)"); ap.add_argument("--out", default="axi_swarm_exhaustive.json"); ap.add_argument("--max-ops", type=int, default=MAX_OPS)
    a = ap.parse_args(argv); t0 = time.time()
    res = run_exhaustive(a.max_ops, progress=lambda k: print(f"  graph {k}/80", file=sys.stderr) if k % 10 == 0 else None); res["seconds"] = round(time.time() - t0, 1); print_exhaustive(res)
    with open(a.out, "w") as f: json.dump(res, f, indent=1, sort_keys=True)
    print("saved", a.out)


if __name__ == "__main__": main()
