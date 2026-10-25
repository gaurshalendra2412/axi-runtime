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
  * `focus_meet(g, a, b, ...)`: for a request about two things (how is a linked to b, does a depend on b), rings grown from both until
    they meet, and only the nodes and edges of the shortest paths between them (item 130, test 156). Prompted by Rajnish Choubey's lines
    in "The Gen 2 Reset": each word a centre with expanding rings; meaning sparks where two words' rings intersect. The search is the
    standard one from both ends; what is new is using it to choose the view.

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


def _steps(g: Graph, n, forward: Optional[bool], rels):
    """(edge, the node at its other end) for every edge of `n` the rings may follow: both ways (forward None), out (True) or in (False)."""
    for k in sorted(g.incident(n)):
        u, v, r = k
        if rels is not None and r not in rels: continue
        if u == n and forward is not False: yield k, v
        if v == n and forward is not True: yield k, u


def focus_meet(g: Graph, a: Tuple[int, int], b: Tuple[int, int], directed: bool = False, rels: Optional[Iterable[str]] = None,
               max_hops: Optional[int] = None) -> Tuple[Graph, dict]:
    """Meeting rings (concept-map item 130, docs/FOCUSED_VIEW.md, test 156): rings grow from `a` and from `b`, one whole ring at a time and
    the smaller side first, until they meet. The view is every node and edge on a shortest path between the two, nothing else.

    `directed`: follow edges forward from `a` and backward from `b` (does a reach b along its edges), else either way. `rels`: the
    relations the rings may follow (all when None). `max_hops`: stop when the two radii add up to this without a meeting.
    why: met, distance (the length of the shortest path), rings (the radius from a, the radius from b), reached (the nodes the rings
    touched, the work done), paths (how many shortest paths), reason (when they did not meet: "not connected" means the host knows
    exactly that there is no path; "no meeting within max_hops" means it does not know)."""
    rels = None if rels is None else set(rels)
    if a not in g.nodes or b not in g.nodes:
        return Graph(), dict(met=False, distance=None, rings=(0, 0), reached=0, paths=0, reason="a named node is not in the graph")
    da, db = {a: 0}, {b: 0}; fa, fb = [a], [b]; ra = rb = 0
    fwd_a, fwd_b = (True, False) if directed else (None, None)
    while not (set(da) & set(db)):
        if (max_hops is not None and ra + rb >= max_hops) or not fa or not fb:
            out = Graph()
            for n in sorted({a, b}): out.add_node(n, g.nodes[n])
            why = "no meeting within max_hops" if fa and fb else "not connected"
            return out, dict(met=False, distance=None, rings=(ra, rb), reached=len(set(da) | set(db)), paths=0, reason=why)
        grow_a = len(fa) < len(fb) or (len(fa) == len(fb) and ra <= rb)          # the smaller ring first; on a tie, the shorter radius
        dist, front, fwd = (da, fa, fwd_a) if grow_a else (db, fb, fwd_b)
        nxt = []
        for n in front:
            for _, m in _steps(g, n, fwd, rels):
                if m not in dist: dist[m] = dist[n] + 1; nxt.append(m)
        if grow_a: fa, ra = nxt, ra + 1
        else: fb, rb = nxt, rb + 1
    meet = sorted(set(da) & set(db)); d = ra + rb
    nodes, edges = set(meet), set()
    for dist, fwd_back in ((da, False if directed else None), (db, True if directed else None)):     # walk back from the meeting to each end
        layer = list(meet)
        while layer:
            prev = []
            for n in layer:
                for k, p in _steps(g, n, fwd_back, rels):
                    if dist.get(p) == dist[n] - 1:
                        edges.add(k)
                        if p not in nodes: nodes.add(p); prev.append(p)
            layer = sorted(set(prev))
    out = Graph()
    for n in sorted(nodes): out.add_node(n, g.nodes[n])
    for k in sorted(edges): out.add_edge(k[0], k[1], k[2], g.edges[k])
    return out, dict(met=True, distance=d, rings=(ra, rb), reached=len(set(da) | set(db)), paths=_count_paths(out, a, b, directed, rels, d), reason="")


def _count_paths(view: Graph, a, b, directed, rels, d) -> int:
    """The number of shortest paths from a to b inside the view (every walk of length d in it is one)."""
    ways = {a: 1}; layer = {a}
    for _ in range(d):
        nxt: Dict = {}
        for n in layer:
            for _, m in _steps(view, n, True if directed else None, rels):
                nxt[m] = nxt.get(m, 0) + ways[n]
        ways = nxt; layer = set(nxt)
    return ways.get(b, 0)
