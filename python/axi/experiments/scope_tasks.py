"""Task kinds for the scope test (docs/SCOPE_TEST.md). The four old kinds are those of `cascade.pick_task`; three are new, chosen to be where a scope rule could go wrong:

  delete_edge    "Remove the edge from (a,b) to (c,d) with relation r and weight w."          the correct delta is a lone edge delete, no service is deleted
                 (the post-hoc idea `contain_delta_internal` drops it and does nothing)
  delete_pair    "Delete the service at (x,y), and also remove the edge from (a,b) to (c,d) ..."   a service delete plus an edge delete that does not touch it
                 (a rule that only keeps edges of the deleted service would drop the second half)
  delete_weight  "Remove every edge with weight w."                                           names no service at all, so no scope can be derived from the request
                 (named scope applies nothing here and is exactly M6)
Every request is checked: the ideal delta is admissible on the graph and the expected graph is that delta applied.
`make_scope_tasks(n, seed)` is the single-step set (kinds in turn); `KINDS_SCOPE_LONG` is the mix of the long-horizon variant (no delete_weight: an implicit target has no oracle edit count to keep fixed)."""
import random
from collections import Counter
from typing import List, Optional

from axi.engine.cico_parser import CICOParser
from axi.engine.gate import Graph, check, apply
from axi.experiments.agent_loop import Task, copy_graph
from axi.experiments.cascade import pick_task, random_graph

KINDS_SCOPE = ["delete_dep", "delete_leaf", "delete_edge", "delete_pair", "delete_weight", "add_node", "add_edge"]
NEW_KINDS = ("delete_edge", "delete_pair", "delete_weight")
KINDS_SCOPE_LONG = ["add_node"] * 2 + ["add_edge"] * 2 + ["delete_dep"] * 3 + ["delete_leaf"] + ["delete_edge"] + ["delete_pair"]


def _edge_txt(g: Graph, k) -> str: (u, v, r) = k; return f"DEL[({u[0]},{u[1]})->({v[0]},{v[1]}):{r}#{g.edges[k]}]"
def _edge_words(g: Graph, k) -> str: (u, v, r) = k; return f"the edge from ({u[0]},{u[1]}) to ({v[0]},{v[1]}) with relation {r} and weight {g.edges[k]}"


def pick_scope_task(rng: random.Random, g: Graph, kind: str) -> Optional[Task]:
    """A request of this kind that is meaningful on the live graph; None if impossible there."""
    if kind not in NEW_KINDS: return pick_task(rng, g, kind)
    nodes = sorted(g.nodes)
    if kind == "delete_edge":
        keys = sorted(g.edges)
        if not keys: return None
        k = rng.choice(keys); ideal = _edge_txt(g, k); instr = "Remove " + _edge_words(g, k) + "."
    elif kind == "delete_pair":
        cand = [n for n in nodes if len(g.incident(n)) <= 3 and any(n not in (e[0], e[1]) for e in g.edges)]
        if not cand: return None
        n = rng.choice(cand); other = sorted(k for k in g.edges if n not in (k[0], k[1])); k = rng.choice(other)
        ideal = " ".join(["DEL[%d,%d]" % n] + [_edge_txt(g, e) for e in sorted(g.incident(n))] + [_edge_txt(g, k)])
        instr = f"Delete the service at ({n[0]},{n[1]}), and also remove " + _edge_words(g, k) + "."
    else:                                                                           # delete_weight
        count = Counter(g.edges.values()); ws = sorted(w for w, c in count.items() if 1 <= c <= 3)
        if not ws: return None
        w = rng.choice(ws); ks = sorted(k for k in g.edges if g.edges[k] == w)
        ideal = " ".join(_edge_txt(g, k) for k in ks); instr = f"Remove every edge with weight {w}."
    d = CICOParser.parse_transition_delta(ideal)
    if not check(g, d).ok: return None
    exp = copy_graph(g); apply(exp, d)
    return Task(kind, copy_graph(g), instr, ideal, exp)


def make_scope_tasks(n: int, seed: int) -> List[Task]:
    """n tasks, kinds in turn (n = 140 gives 20 of each), each on a fresh random graph of the same shape as every earlier test."""
    rng = random.Random(seed); out = []
    for i in range(n):
        kind = KINDS_SCOPE[i % len(KINDS_SCOPE)]
        while True:
            t = pick_scope_task(rng, random_graph(rng), kind)
            if t is not None: out.append(t); break
    return out
