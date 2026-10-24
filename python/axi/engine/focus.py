"""The 1 % view: show the model only the part of the exact state a request needs (concept-map item 124, docs/FOCUSED_VIEW.md, test 150).

Source: Rajnish Choubey's text "The Trilogy of the Canvas: The Art of the Void" (the owner pasted it on 24 Oct). Creation and editing are
joined by a third act, deliberate deletion: of everything made, 99 % is deleted and 1 % is shown, and the value of an idea is the noise
cleared around it.

AXI already deletes the history: each step shows the exact state, not the chat. But it shows the WHOLE state, so its prompt still grows
with the size of the state (Test 23: 959 to 1,337 tokens a step as the list grew), and a state of thousands of items would not fit. This
module deletes from the VIEW the part of the state a request does not touch:

  * `focus_items(state, request, ...)`: for a list of items with facts, the items the request names, the items a description in it
    points to (a value of a describing field, such as an owner: every item that has it, so an ambiguous description stays visible as
    ambiguous), and the most recently added item when the request says "last". A request that names a NEW item (one the host may see
    added) needs no other item. A request the host cannot place gets the whole state (the fallback).
  * `focus_graph(g, named, hops)`: for a graph, the named nodes, every edge that touches them and the nodes at the other ends, out to
    `hops` steps (a delete needs every edge of the deleted node; the repair adds nothing beyond that).

The view changes what the model READS, never what the host CHECKS: the check, the repair and the commit still run on the whole exact state.
It edits no framework file. What it does NOT show: that a model is as exact with the view as with the whole state; that needs a run."""
from __future__ import annotations
import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from axi.engine.gate import Graph


def mentioned(text: str, vocab: Iterable[str]) -> List[str]:
    """The phrases of `vocab` that occur in `text` as whole words, case ignored, longest first, spans never overlapping."""
    found, taken = [], []
    for v in sorted(set(vocab), key=lambda s: (-len(s), s)):
        for m in re.finditer(r"(?<!\w)" + re.escape(v) + r"(?!\w)", text, re.I):
            if not any(m.start() < b and a < m.end() for a, b in taken):
                taken.append((m.start(), m.end())); found.append(v); break
    return found


def focus_items(state: Mapping[str, Mapping], request: str, describers: Optional[Mapping[str, Iterable[str]]] = None,
                new_names: Iterable[str] = (), recency: Sequence[str] = ("last",)) -> Tuple[Dict[str, dict], dict]:
    """(view, why). `state` keeps its items in the order they were added; the view keeps that order. `describers`: field -> the values a
    description may use (for example owner -> first names). `new_names`: names the host may see added."""
    names = list(state)
    named = [n for n in mentioned(request, names)]
    described: List[str] = []
    for f, values in (describers or {}).items():
        for v in mentioned(request, [str(x) for x in values]):
            described += [n for n in names if str(state[n].get(f)) == v]
    recent = [names[-1]] if names and any(re.search(r"\b" + re.escape(w) + r"\b", request, re.I) for w in recency) else []
    new = [n for n in mentioned(request, [x for x in new_names if x not in state])]
    keep = set(named) | set(described) | set(recent)
    fallback = not keep and not new
    view = {n: dict(state[n]) for n in names if fallback or n in keep}
    return view, dict(named=named, described=sorted(set(described), key=names.index), recent=recent, new=new, fallback=fallback, shown=len(view), hidden=len(state) - len(view))


def focus_graph(g: Graph, named: Iterable[Tuple[int, int]], hops: int = 1) -> Graph:
    """The named nodes that exist, every edge within `hops` steps of them, and the nodes those edges reach."""
    frontier = {n for n in named if n in g.nodes}; seen = set(frontier); edges = set()
    for _ in range(max(0, hops)):
        nxt = set()
        for n in frontier:
            for k in g.incident(n):
                edges.add(k); u, v, _ = k
                for m in (u, v):
                    if m not in seen: seen.add(m); nxt.add(m)
        frontier = nxt
    out = Graph()
    for n in sorted(seen): out.add_node(n, g.nodes[n])
    for k in sorted(edges): out.add_edge(k[0], k[1], k[2], g.edges[k])
    return out
