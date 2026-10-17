"""Engineering for docs/DYNAMIC_WORLD_TEST.md (concept-map item 102): what the host layer does when the proposer is NOT a language model and the world changes under it.

CPU only, no language model, no internet. A small neural network (numpy, trained in this file) or an exact reader proposes changes to a service graph from a SNAPSHOT of the held state; with chance
`s` per step the world changes by itself between the snapshot and the commit (an edge or a service appears or disappears), so the proposal was made on a state that is one change old.
Four ways of handling the proposal are compared on the same requests and the same changes of the world:
  blind    apply the proposal as written
  gate     apply it only if the gate accepts it, else nothing
  retry    the gate; if it refuses, ask the proposer again on the CURRENT state (a fresh read) and use the gate once more
  repair   `ground_and_complete`: keep what is true, cut what is false, close the boundary (the framework's repair; `degree_threshold` = REPAIR_DEGREE so that it deletes every dependant)
What is the "right" outcome of a step: the request's MEANING applied to the state the host holds NOW (`intent_delta`): a service to delete goes together with every edge it has now; a request about
something that is no longer there does nothing. A step "slips" when the arm's new state is not the arm's own current state with that change applied (`slip`); the arm's state is compared with the
oracle's as well (`exact`, `distance`), which is where one slip stays for the rest of the episode.
Nothing here is measured on a robot, a vehicle or a model of one: the "world" is a graph, the "controller" reads a flattened picture of it, and the numbers are about this instrument."""
import hashlib, json, math, os, random, sys, time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from axi.engine.cico_parser import CICOParser, EdgeDelta, NodeDelta, StateTransitionDelta
from axi.engine.gate import Graph, apply, check
from axi.experiments.agent_loop import blind_apply, copy_graph, same_graph
from axi.experiments.cascade import next_state
from axi.experiments.long_horizon import distance
from axi.experiments.million_hard import KINDS_HARD, MAX_EDGES_H, MAX_NODES_H, pick_hard_task

P = CICOParser.parse_transition_delta
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "..", "..", "docs", "results")
NET_FILE = "axi_dynworld_net.npz"

# ---------------------------------------------------------------- the world
ROWS, COLS = 5, 6
CELLS = ROWS * COLS
REL = "dep"
CAP_NODES, CAP_EDGES = MAX_NODES_H, MAX_EDGES_H           # 24 and 48, as in the hard test
REPAIR_DEGREE = CAP_EDGES                                   # the repair deletes every dependant (no service is kept for having many): the oracle's rule
KINDS = tuple(KINDS_HARD)                                   # the request mix of the hard test
KIND5 = ("add_node", "add_edge", "delete_service", "delete_edge", "delete_pair")
EVENT_WEIGHTS = (("drop_edge", 35), ("add_edge", 35), ("drop_node", 15), ("add_node", 15))
REGIMES = ("uniform", "near")
NOISE = (0.0, 0.05, 0.1, 0.2, 0.4)
PROPOSERS = ("exact", "net")
ARMS = ("blind", "gate", "retry", "repair")
EPISODES, STEPS = 20, 1000
QUARTERS = 4
SEED = 20261101


def cell(n) -> int: return n[0] * COLS + n[1]
def cell_node(i: int): return (i // COLS, i % COLS)


@dataclass(frozen=True)
class Req:
    """A request in the form the proposers read: what is asked, about which service and/or edge. (The language model of the other tests read the same request as a sentence.)"""
    kind: str                         # one of KIND5
    task_kind: str                    # the picker's kind (delete_dep, delete_hub, ...)
    node: Optional[tuple] = None
    value: Optional[int] = None
    edge: Optional[tuple] = None      # (u, v, rel)
    weight: Optional[int] = None


def req_from_task(task) -> Req:
    """The structured request behind a picker task, read from the task's ideal reply on the state it was drawn on."""
    d = P(task.ideal); tk = task.kind
    if tk == "add_node": nd = d.node_deltas[0]; return Req("add_node", tk, node=(nd.r, nd.c), value=nd.val)
    if tk == "add_edge": e = d.edge_deltas[0]; return Req("add_edge", tk, edge=(e.u, e.v, e.relation), weight=e.weight)
    if tk in ("delete_dep", "delete_leaf", "delete_hub"): nd = d.node_deltas[0]; return Req("delete_service", tk, node=(nd.r, nd.c))
    if tk == "delete_edge": e = d.edge_deltas[0]; return Req("delete_edge", tk, edge=(e.u, e.v, e.relation), weight=e.weight)
    if tk == "delete_pair":
        nd = d.node_deltas[0]; n = (nd.r, nd.c); e = [x for x in d.edge_deltas if n not in (x.u, x.v)][0]
        return Req("delete_pair", tk, node=n, edge=(e.u, e.v, e.relation), weight=e.weight)
    raise ValueError(f"request kind {tk!r} is not part of this test")


def intent_delta(G: Graph, req: Req) -> StateTransitionDelta:
    """What the request MEANS on the state `G`: the delta that does what was asked with what is there now. A service to delete goes with every edge it has in `G`; an item that is already gone, an edge
    whose end is gone, an add of something that is already there: nothing (the request is moot)."""
    nodes, edges = [], []
    if req.kind == "add_node":
        if req.node not in G.nodes: nodes.append(NodeDelta("ADD", req.node[0], req.node[1], req.value))
    elif req.kind == "add_edge":
        u, v, rel = req.edge
        if u in G.nodes and v in G.nodes and req.edge not in G.edges: edges.append(EdgeDelta("ADD", u, v, rel, req.weight))
    elif req.kind == "delete_edge":
        if req.edge in G.edges: edges.append(EdgeDelta("DEL", req.edge[0], req.edge[1], req.edge[2], G.edges[req.edge]))
    else:
        gone = set()
        if req.node in G.nodes:
            nodes.append(NodeDelta("DEL", req.node[0], req.node[1]))
            for k in sorted(G.incident(req.node)): edges.append(EdgeDelta("DEL", k[0], k[1], k[2], G.edges[k])); gone.add(k)
        if req.kind == "delete_pair" and req.edge in G.edges and req.edge not in gone: edges.append(EdgeDelta("DEL", req.edge[0], req.edge[1], req.edge[2], G.edges[req.edge]))
    return StateTransitionDelta(nodes, edges)


def start_graph(rng: random.Random) -> Graph:
    """Eight services on a 3 x 4 patch of the grid, 7 to 10 `dep` edges (the shape of `cascade.random_graph`, with one relation only)."""
    cells = [(r, c) for r in range(3) for c in range(4)]; rng.shuffle(cells); nodes = cells[:8]; g = Graph()
    for n in nodes: g.add_node(n, rng.randint(1, 9))
    core = nodes[:7]
    for _ in range(rng.randint(7, 10)):
        u, v = rng.sample(core, 2)
        if (u, v, REL) not in g.edges: g.add_edge(u, v, REL, rng.randint(1, 9))
    return g


def draw_request(rng: random.Random, g: Graph):
    """The request of the hard test on the oracle's state: same caps, same picker, same 20 tries then every kind once (`endurance.Stream.next`). Returns (Req, ideal text on g) or None."""
    kinds = [k for k in KINDS if not (k == "add_node" and len(g.nodes) >= CAP_NODES) and not (k == "add_edge" and len(g.edges) >= CAP_EDGES)]; task = None
    for _ in range(20):
        task = pick_hard_task(rng, g, rng.choice(kinds))
        if task is not None: break
    if task is None:
        for k in sorted(set(kinds)):
            task = pick_hard_task(rng, g, k)
            if task is not None: break
    return None if task is None else (req_from_task(task), task.ideal)


# ---------------------------------------------------------------- the world changes by itself
def draw_event(rng: random.Random, g: Graph, req: Req, regime: str):
    """One change of the world, drawn on the oracle's state `g`. Kind by EVENT_WEIGHTS; then the item: `uniform` any item of that kind; `near` an item that touches a service the request names (and, for a
    new service, the request's own cell if it is free): the world changing where the agent is working. If no item of the drawn kind fits, there is no change this time. Returns a tuple or None.
    ('drop_edge', key) | ('add_edge', key, weight) | ('drop_node', n) | ('add_node', n, value)"""
    r = rng.random() * sum(w for _, w in EVENT_WEIGHTS); kind = EVENT_WEIGHTS[-1][0]; acc = 0
    for k, w in EVENT_WEIGHTS:
        acc += w
        if r < acc: kind = k; break
    named = set()
    if req.node is not None: named.add(req.node)
    if req.edge is not None: named |= {req.edge[0], req.edge[1]}
    named = sorted(n for n in named if n in g.nodes); near = regime == "near"
    pick = lambda c: c[rng.randrange(len(c))] if c else None
    if kind == "drop_edge":
        allc = sorted(g.edges); c = [k for k in allc if k[0] in named or k[1] in named] if near else []
        k = pick(c or allc); return None if k is None else ("drop_edge", k)
    if kind == "add_edge":
        w = rng.randint(1, 9)
        if len(g.edges) >= CAP_EDGES: return None
        ns = sorted(g.nodes); pairs = [(u, v) for u in ns for v in ns if u != v and (u, v, REL) not in g.edges]; c = [p for p in pairs if p[0] in named or p[1] in named] if near else []
        p = pick(c or pairs); return None if p is None else ("add_edge", (p[0], p[1], REL), w)
    if kind == "drop_node":
        n = pick(named if (near and named) else sorted(g.nodes)); return None if n is None else ("drop_node", n)
    v = rng.randint(1, 9)
    if len(g.nodes) >= CAP_NODES: return None
    free = [(r_, c_) for r_ in range(ROWS) for c_ in range(COLS) if (r_, c_) not in g.nodes]
    n = req.node if (near and req.kind == "add_node" and req.node in free) else pick(free)
    return None if n is None else ("add_node", n, v)


def apply_event(g: Graph, ev) -> None:
    """The change of the world, applied to a state in place. It can never fail: a change that does not fit the state (an edge to a service the state does not have, an item that is not there) is ignored by
    that state. A service that collapses takes every edge it has with it."""
    if ev is None: return
    kind = ev[0]
    if kind == "drop_edge":
        if ev[1] in g.edges: g.del_edge(ev[1])
    elif kind == "add_edge":
        k = ev[1]
        if k not in g.edges and k[0] in g.nodes and k[1] in g.nodes: g.add_edge(k[0], k[1], k[2], ev[2])
    elif kind == "drop_node":
        n = ev[1]
        if n in g.nodes:
            for k in sorted(g.incident(n)): g.del_edge(k)
            g.del_node(n)
    elif kind == "add_node":
        if ev[1] not in g.nodes: g.add_node(ev[1], ev[2])
    else: raise ValueError(ev)


# ---------------------------------------------------------------- a route the held map offers
def route(g: Graph, a, b) -> Optional[list]:
    """A shortest path from a to b over the edges of `g` read as two-way paths (relation ignored), neighbours in sorted order. None if `a` or `b` is not a service of `g` or no path exists. The edges of a
    CORRUPTED state are followed as they are, including those that lead to a service the state does not have."""
    if a not in g.nodes or b not in g.nodes: return None
    adj = defaultdict(list)
    for (u, v, _r) in g.edges: adj[u].append(v); adj[v].append(u)
    prev = {a: None}; q = deque([a])
    while q:
        x = q.popleft()
        if x == b: break
        for y in sorted(set(adj[x])):
            if y not in prev: prev[y] = x; q.append(y)
    if b not in prev: return None
    path = [b]
    while prev[path[-1]] is not None: path.append(prev[path[-1]])
    return path[::-1]


def route_safe(truth_pairs: set, truth_nodes, path) -> bool:
    """The robot can walk the route in the real world: every stop is a service that exists and every leg is an edge that exists (either direction)."""
    return all(n in truth_nodes for n in path) and all(frozenset((x, y)) in truth_pairs for x, y in zip(path, path[1:]))


def state_text(g: Graph) -> str: return repr(sorted(g.nodes.items())) + repr(sorted(g.edges.items()))


# ---------------------------------------------------------------- the proposers
def exact_reply(S: Graph, req: Req) -> StateTransitionDelta:
    """The exact reader: what the request means on the snapshot it is shown. It is never wrong about the snapshot, so any trouble that comes from it is the age of the snapshot."""
    return intent_delta(S, req)


O_NODE, O_ADJ = 0, CELLS
O_KIND = O_ADJ + CELLS * CELLS
O_RN = O_KIND + len(KIND5)
O_RU, O_RV = O_RN + CELLS, O_RN + 2 * CELLS
O_W = O_RN + 3 * CELLS
D_IN = O_W + 2
N_POS = CELLS + CELLS * CELLS                                # one output per cell and per ordered pair of cells, each with 3 classes: nothing, ADD, DEL
HIDDEN = 128


def encode(S: Graph, req: Req, out: Optional[np.ndarray] = None) -> np.ndarray:
    """The picture the network reads: which cells hold a service (30), which ordered pairs hold an edge (900; no weights, no relation), the kind of request (5), the request's service (30), the request's edge
    ends (30 + 30), its weight and its value (divided by 9). 1,027 numbers."""
    x = np.zeros(D_IN, np.float32) if out is None else out
    for n in S.nodes: x[O_NODE + cell(n)] = 1.0
    for (u, v, _r) in S.edges: x[O_ADJ + cell(u) * CELLS + cell(v)] = 1.0
    x[O_KIND + KIND5.index(req.kind)] = 1.0
    if req.node is not None: x[O_RN + cell(req.node)] = 1.0
    if req.edge is not None: x[O_RU + cell(req.edge[0])] = 1.0; x[O_RV + cell(req.edge[1])] = 1.0
    if req.weight is not None: x[O_W] = req.weight / 9.0
    if req.value is not None: x[O_W + 1] = req.value / 9.0
    return x


def labels(S: Graph, req: Req) -> np.ndarray:
    """The ideal reply on the snapshot as one class per output position (0 nothing, 1 ADD, 2 DEL)."""
    y = np.zeros(N_POS, np.int8); d = intent_delta(S, req)
    for nd in d.node_deltas: y[cell((nd.r, nd.c))] = 1 if nd.op == "ADD" else 2
    for e in d.edge_deltas: y[CELLS + cell(e.u) * CELLS + cell(e.v)] = 1 if e.op == "ADD" else 2
    return y


def decode(cls: np.ndarray, S: Graph, req: Req) -> StateTransitionDelta:
    """The chosen classes as a delta. The value of an added service and the weight of an added edge are copied from the request; the weight of a deleted edge is read from the snapshot (the request's
    weight, or 1, if the snapshot has no such edge)."""
    nodes, edges = [], []
    for i in np.nonzero(cls[:CELLS])[0]:
        n = cell_node(int(i)); nodes.append(NodeDelta("ADD", n[0], n[1], req.value if req.value is not None else 1) if cls[i] == 1 else NodeDelta("DEL", n[0], n[1]))
    for idx in np.nonzero(cls[CELLS:])[0]:
        i, j = divmod(int(idx), CELLS); u, v = cell_node(i), cell_node(j); c = cls[CELLS + idx]
        edges.append(EdgeDelta("ADD", u, v, REL, req.weight if req.weight is not None else 1) if c == 1 else EdgeDelta("DEL", u, v, REL, S.edges.get((u, v, REL), req.weight if req.weight is not None else 1)))
    return StateTransitionDelta(nodes, edges)


class Net:
    """One hidden layer of ReLU units between the picture and the 930 x 3 outputs; float32; trained here with softmax cross-entropy per output and Adam. It has no knowledge of the rules: it is shown
    states and ideal replies."""
    def __init__(self, W1, b1, W2, b2): self.W1, self.b1, self.W2, self.b2 = W1, b1, W2, b2

    @staticmethod
    def init(seed: int, hidden: int = HIDDEN) -> "Net":
        r = np.random.RandomState(seed)
        return Net((r.randn(D_IN, hidden) * math.sqrt(2.0 / D_IN)).astype(np.float32), np.zeros(hidden, np.float32),
                   (r.randn(hidden, N_POS * 3) * math.sqrt(1.0 / hidden)).astype(np.float32), np.zeros(N_POS * 3, np.float32))

    def logits(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(x @ self.W1 + self.b1, 0.0); return (h @ self.W2 + self.b2).reshape(x.shape[:-1] + (N_POS, 3))

    def reply(self, S: Graph, req: Req) -> StateTransitionDelta:
        return decode(self.logits(encode(S, req)).argmax(-1), S, req)

    def save(self, path: str) -> None: np.savez_compressed(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    @staticmethod
    def load(path: str) -> "Net":
        z = np.load(path); return Net(*(z[k].astype(np.float32) for k in ("W1", "b1", "W2", "b2")))


TRAIN = dict(streams=120, steps=500, iterations=16000, batch=256, lr=2e-3, lr_end=0.1, hidden=HIDDEN, init_seed=7, sample_seed=11, stream_seed=7000)


def make_dataset(streams: int, steps: int, seed0: int):
    """Clean streams (no change of the world): the state before each request, the request, the ideal reply on that state. Returns (X uint8 [n, D_IN-2], XW float32 [n, 2], Y int8 [n, N_POS])."""
    n = streams * steps; X = np.zeros((n, D_IN - 2), np.uint8); XW = np.zeros((n, 2), np.float32); Y = np.zeros((n, N_POS), np.int8); row = 0; buf = np.zeros(D_IN, np.float32)
    for z in range(streams):
        rng = random.Random(f"train/{seed0 + z}"); g = start_graph(rng)
        for t in range(steps):
            out = draw_request(rng, g)
            if out is None: break
            req, _ = out; buf[:] = 0; encode(g, req, buf); X[row] = buf[:D_IN - 2]; XW[row] = buf[D_IN - 2:]; Y[row] = labels(g, req); row += 1; apply(g, intent_delta(g, req))
    return X[:row], XW[:row], Y[:row]


def train_net(cfg: Optional[dict] = None, say: Callable = lambda *a: None) -> Net:
    """The recipe (TRAIN): a dataset of clean streams, `iterations` mini-batches drawn at random, Adam, the learning rate falling along a half cosine from `lr` to `lr` x `lr_end`. Deterministic given the seeds up to the numerical noise of the matrix products."""
    c = dict(TRAIN, **(cfg or {})); X, XW, Y = make_dataset(c["streams"], c["steps"], c["stream_seed"]); n = len(X); say(f"dataset {n} examples")
    net = Net.init(c["init_seed"], c["hidden"]); net.losses = []; r = np.random.RandomState(c["sample_seed"]); params = [net.W1, net.b1, net.W2, net.b2]
    m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]; b1_, b2_, eps = 0.9, 0.999, 1e-8
    for it in range(1, c["iterations"] + 1):
        idx = r.randint(0, n, c["batch"]); x = np.concatenate([X[idx].astype(np.float32), XW[idx]], 1); y = Y[idx].astype(np.int64)
        pre = x @ net.W1 + net.b1; h = np.maximum(pre, 0.0); z = (h @ net.W2 + net.b2).reshape(-1, N_POS, 3)
        z = z - z.max(-1, keepdims=True); e = np.exp(z); p = e / e.sum(-1, keepdims=True)
        loss = -np.log(np.take_along_axis(p, y[..., None], -1) + 1e-12).sum(1).mean()
        dz = p; np.put_along_axis(dz, y[..., None], np.take_along_axis(dz, y[..., None], -1) - 1.0, -1); dz = (dz / len(idx)).reshape(len(idx), -1).astype(np.float32)
        lr_t = c["lr"] * (c["lr_end"] + (1 - c["lr_end"]) * 0.5 * (1 + math.cos(math.pi * it / c["iterations"])))
        dW2 = h.T @ dz; db2 = dz.sum(0); dh = dz @ net.W2.T; dh[pre <= 0] = 0; dW1 = x.T @ dh; db1 = dh.sum(0); grads = [dW1, db1, dW2, db2]
        for i, (p_, g_) in enumerate(zip(params, grads)):
            m[i] = b1_ * m[i] + (1 - b1_) * g_; v[i] = b2_ * v[i] + (1 - b2_) * g_ * g_
            p_ -= (lr_t * (m[i] / (1 - b1_ ** it)) / (np.sqrt(v[i] / (1 - b2_ ** it)) + eps)).astype(np.float32)
        if it % 100 == 0: net.losses.append((it, float(loss)))
        if it % 500 == 0: say(f"iteration {it}: loss per example {loss:.4f}")
    return net


def reply_quality(net: Net, seeds: Sequence[int], steps: int = 300) -> dict:
    """On clean streams the net has not been trained on: how often its reply is exactly the ideal one, and what the others look like. For the Result and for looking at the recipe; not a reading rule."""
    n = same = omitted = extra = wrong = 0
    for z in seeds:
        rng = random.Random(f"quality/{z}"); g = start_graph(rng)
        for t in range(steps):
            out = draw_request(rng, g)
            if out is None: break
            req, _ = out; ideal = intent_delta(g, req); d = net.reply(g, req); n += 1
            key = lambda dd: ({(x.op, x.r, x.c) for x in dd.node_deltas}, {(x.op, x.u, x.v, x.weight) for x in dd.edge_deltas})
            a, b = key(d), key(ideal)
            if a == b: same += 1
            else:
                miss = (b[0] - a[0]) or (b[1] - a[1]); more = (a[0] - b[0]) or (a[1] - b[1]); omitted += bool(miss); extra += bool(more); wrong += 1
            apply(g, ideal)
    return dict(requests=n, exact=same, not_exact=wrong, left_out=omitted, added=extra)


# ---------------------------------------------------------------- the four ways of handling a proposal
def _admit(H: Graph, d: StateTransitionDelta) -> Graph:
    """The gate: the proposal is applied if it is admissible on `H`, otherwise the state is returned unchanged (a copy)."""
    h = copy_graph(H)
    if check(H, d).ok: apply(h, d)
    return h


def arm_step(arm: str, read: Callable, S: Graph, H: Graph, req: Req) -> Tuple[Graph, int]:
    """One step of one arm. `read(state, req)` is the proposer; `S` is the snapshot it reads (the held state before the change of the world), `H` the held state now. Returns (new state, proposer calls)."""
    d = read(S, req)
    if arm == "blind": return copy_graph(blind_apply(H, d)), 1
    if arm == "gate": return _admit(H, d), 1
    if arm == "repair": return next_state("repair_m6", H, d, degree_threshold=REPAIR_DEGREE), 1
    if arm == "retry":
        if check(H, d).ok: return _admit(H, d), 1
        return _admit(H, read(H, req)), 2                    # refused: ask again on the CURRENT state, use the gate once more
    raise ValueError(arm)


def is_slip(after: Graph, now: Graph, req: Req) -> bool:
    """The step did not do what the request means on the state the arm holds NOW: the new state is not that state with `intent_delta` applied."""
    return not same_graph(after, blind_apply(now, intent_delta(now, req)))


def _arm_stats() -> dict:
    return dict(slip=0, exact=0, corrupted=0, dist_sum=0, dist_max=0, calls=0, fresh=0, start_exact=0, q_dist=[0] * QUARTERS, q_exact=[0] * QUARTERS, q_steps=[0] * QUARTERS,
                route_planned=0, route_unsafe=0, route_possible=0, route_missed=0, by_event={}, by_kind={})


def _bump(d: dict, key: str, slip: bool):
    c = d.setdefault(key, [0, 0]); c[0] += 1; c[1] += int(slip)


def run_episode(seed: str, regime: str, s: float, steps: int, readers: Dict[str, Callable], acc: dict, digest) -> dict:
    """One episode: the oracle's world, `steps` requests, the change of the world with chance `s` before each reply, every (proposer, arm) holding its own state. `acc[proposer][arm]` is filled in.
    Returns what the episode saw of the world (counts)."""
    rng_req, rng_ev, rng_rt = random.Random(f"{seed}/req"), random.Random(f"{seed}/ev"), random.Random(f"{seed}/route"); O = start_graph(random.Random(f"{seed}/start"))
    H = {(p, a): copy_graph(O) for p in readers for a in ARMS}; seen = dict(steps=0, events=0, event_kinds={}, requests={}, oracle_nodes=0, oracle_edges=0, oracle_corrupted=0, ended=False)
    for t in range(steps):
        out = draw_request(rng_req, O)
        if out is None: seen["ended"] = True; break
        req, _ideal = out; ev = draw_event(rng_ev, O, req, regime) if rng_ev.random() < s else None
        ekind = ev[0] if ev is not None else "none"; seen["steps"] += 1; seen["requests"][req.kind] = seen["requests"].get(req.kind, 0) + 1
        if ev is not None: seen["events"] += 1; seen["event_kinds"][ekind] = seen["event_kinds"].get(ekind, 0) + 1
        O_pre = copy_graph(O) if ev is not None else O                                     # `O_pre` is never changed; `apply_event` below changes O in place
        snaps = {k: (copy_graph(g) if ev is not None else g) for k, g in H.items()}
        apply_event(O, ev)
        for g in H.values(): apply_event(g, ev)
        want = intent_delta(O, req); assert check(O, want).ok; O_next = copy_graph(O); apply(O_next, want)
        q = min(QUARTERS - 1, t * QUARTERS // steps); a_, b_ = None, None; nodes = sorted(O_next.nodes)
        if len(nodes) >= 2: a_, b_ = rng_rt.sample(nodes, 2)
        rt_o = route(O_next, a_, b_) if a_ is not None else None; pairs = {frozenset((u, v)) for (u, v, _r) in O_next.edges}
        for (p, arm), read in ((k, readers[k[0]]) for k in sorted(H)):
            st = acc[p][arm]
            now = H[(p, arm)]; start_exact = same_graph(snaps[(p, arm)], O_pre)
            after, calls = arm_step(arm, read, snaps[(p, arm)], now, req)
            slip = is_slip(after, now, req); ex = same_graph(after, O_next); dist = distance(after, O_next)
            st["slip"] += slip; st["exact"] += ex; st["corrupted"] += (not after.is_well_formed()); st["dist_sum"] += dist; st["dist_max"] = max(st["dist_max"], dist); st["calls"] += calls
            st["start_exact"] += start_exact; st["fresh"] += (start_exact and not ex); st["q_dist"][q] += dist; st["q_exact"][q] += ex; st["q_steps"][q] += 1
            _bump(st["by_event"], ekind, slip); _bump(st["by_kind"], req.kind, slip)
            if a_ is not None:
                st["route_possible"] += rt_o is not None; path = route(after, a_, b_)
                if path is not None:
                    st["route_planned"] += 1; st["route_unsafe"] += (not route_safe(pairs, O_next.nodes, path))
                elif rt_o is not None: st["route_missed"] += 1
            H[(p, arm)] = after
        digest.update(f"{t}|{req.task_kind}|{ev}|{state_text(O_next)}\n".encode()); O = O_next
        seen["oracle_nodes"] += len(O.nodes); seen["oracle_edges"] += len(O.edges); seen["oracle_corrupted"] += (not O.is_well_formed())
    return seen


def run_dynworld(net: Optional[Net], seed: int = SEED, episodes: int = EPISODES, steps: int = STEPS, noise: Sequence[float] = NOISE, regimes: Sequence[str] = REGIMES,
                 proposers: Sequence[str] = PROPOSERS, progress: Optional[Callable] = None) -> dict:
    """Every (regime, s) cell: `episodes` episodes of `steps` requests, all proposers and arms on the same world. The digest covers the requests, the changes of the world and the oracle's states only
    (not the network's replies), so it is the same on any machine."""
    t0 = time.time(); readers = {"exact": exact_reply}
    if "net" in proposers:
        if net is None: raise ValueError("the net proposer needs a network")
        readers["net"] = net.reply
    readers = {p: readers[p] for p in proposers}; cells = {}; dig = hashlib.sha256()
    for regime in regimes:
        for s in noise:
            acc = {p: {a: _arm_stats() for a in ARMS} for p in readers}; seen = dict(steps=0, events=0, event_kinds={}, requests={}, oracle_nodes=0, oracle_edges=0, oracle_corrupted=0, ended=0)
            for e in range(episodes):
                sd = hashlib.sha256()
                one = run_episode(f"{seed}/{regime}/{s}/{e}", regime, s, steps, readers, acc, sd); dig.update(sd.digest())
                for k in ("steps", "events", "oracle_nodes", "oracle_edges", "oracle_corrupted"): seen[k] += one[k]
                seen["ended"] += int(one["ended"])
                for k in ("event_kinds", "requests"):
                    for kk, v in one[k].items(): seen[k][kk] = seen[k].get(kk, 0) + v
                if progress: progress(regime, s, e + 1, episodes)
            cells[f"{regime}/{s}"] = dict(regime=regime, s=s, episodes=episodes, world=seen, proposers=acc)
    return dict(seed=seed, episodes=episodes, steps=steps, noise=list(noise), regimes=list(regimes), proposers=list(readers), arms=list(ARMS), cells=cells, digest=dig.hexdigest()[:16], seconds=round(time.time() - t0, 1))


# ---------------------------------------------------------------- the figures and the reading rules (fixed in docs/DYNAMIC_WORLD_TEST.md before the run)
D_NEAR_HI, D_NEAR_MID = ("near", 0.4), ("near", 0.2)
D_BLIND_SLIP, D_GATE_SLIP, D_RETRY_CALLS, D_GATE_DRIFT, D_UNSAFE_BLIND, D_NET_BLIND_SLIP, D_HALF = 0.03, 0.03, 1.03, 2.0, 0.01, 0.05, 0.5
DW_LINES = [
    ("guard", "every cell of the run finished: all episodes ran all their steps, no request stream ended early, and the oracle's state was well formed at every step"),
    ("guard", "gate, retry and repair were never corrupted (no edge pointing at a missing service) at any step of any cell, for either proposer"),
    ("instrument", "control: with s = 0, in both regimes, the four arms of the exact proposer slipped on no step and held exactly the oracle's state at every step"),
    ("instrument", "the change of the world has teeth: with the exact proposer in the near regime at s = 0.4, blind apply slipped on at least 3 % of its steps and so did the gate"),
    ("claim", "exact proposer, every cell: repair slipped on no step"),
    ("claim", "exact proposer, every cell: retry slipped on no step"),
    ("claim", "exact proposer, near regime, s = 0.4: the retry asked the proposer at least 1.03 times a step on average, the repair exactly once a step"),
    ("claim", "exact proposer, near regime, s = 0.4: the gate does not come back: its mean distance to the oracle in the last quarter of the episodes is at least twice that of the first quarter"),
    ("claim", "exact proposer, near regime, s = 0.4: of the routes blind apply planned at least 1 % were unsafe (they used an edge or a service the real world does not have), and repair planned no unsafe route"),
    ("claim", "exact proposer, s = 0.4: blind apply's slip share and the gate's slip share are each larger in the near regime than in the uniform regime"),
    ("claim", "net proposer, s = 0, both regimes together: blind apply slipped on at least 5 % of the steps (the network's own replies are not always right)"),
    ("claim", "net proposer, s = 0, both regimes together: repair slipped on fewer steps than blind apply, than the gate and than the retry"),
    ("claim", "net proposer, near regime, s = 0.2: repair slipped on fewer steps than blind apply, than the gate and than the retry"),
    ("claim", "net proposer, near regime, s = 0.2: repair's mean distance to the oracle over all steps is smaller than that of blind apply, of the gate and of the retry"),
    ("claim", "net proposer, near regime, s = 0.2: of the routes planned, the share that were unsafe is for repair at most half of blind apply's, and blind apply's is above 0"),
]
DW_HEADLINE = (4, 5, 6, 7)                                 # A: the exact reader, the age of the snapshot only
DW_HEADLINE_NET = (12, 13, 14, 15)                         # B: the trained network, near regime, s = 0.2


def _r(a: float, b: float) -> float: return a / b if b else 0.0


def _cell(res: dict, regime: str, s: float) -> dict: return res["cells"][f"{regime}/{s}"]


def arm_figures(cell: dict, proposer: str, arm: str) -> dict:
    """The rates of one arm in one cell, from the saved tallies."""
    st = cell["proposers"][proposer][arm]; n = cell["world"]["steps"]; q = [_r(a, b) for a, b in zip(st["q_dist"], st["q_steps"])]
    return dict(steps=n, slip=_r(st["slip"], n), slips=st["slip"], exact=_r(st["exact"], n), corrupted=st["corrupted"], mean_distance=_r(st["dist_sum"], n), max_distance=st["dist_max"], calls=_r(st["calls"], n),
                quarter_distance=q, fresh=_r(st["fresh"], st["start_exact"]), unsafe=_r(st["route_unsafe"], st["route_planned"]), unsafe_n=st["route_unsafe"], planned=st["route_planned"],
                missed=_r(st["route_missed"], st["route_possible"]))


def pooled_slip(res: dict, proposer: str, arm: str, s: float) -> float:
    """Slip share of one arm over both regimes together at one noise level."""
    n = sum(_cell(res, g, s)["world"]["steps"] for g in res["regimes"]); k = sum(_cell(res, g, s)["proposers"][proposer][arm]["slip"] for g in res["regimes"]); return _r(k, n)


def verdict_dynworld(res: dict) -> List[Tuple[str, Optional[bool]]]:
    """The 15 reading rules, in the order of docs/DYNAMIC_WORLD_TEST.md. A rule about a cell the run does not hold reads None (not run)."""
    cells = res["cells"]; fig = lambda reg_s, p, a: arm_figures(_cell(res, reg_s[0], reg_s[1]), p, a); has = lambda reg_s: f"{reg_s[0]}/{reg_s[1]}" in cells
    ex_cells = [c for c in cells.values() if "exact" in c["proposers"]]; out: List[Tuple[str, Optional[bool]]] = []
    want = res["episodes"] * res["steps"]
    out.append(("guard", all(c["world"]["steps"] == want and c["world"]["ended"] == 0 and c["world"].get("oracle_corrupted", 0) == 0 and all(sum(a["q_steps"]) == want for pr in c["proposers"].values() for a in pr.values()) for c in cells.values())))
    out.append(("guard", all(a["corrupted"] == 0 for c in cells.values() for pr in c["proposers"].values() for k, a in pr.items() if k != "blind")))
    ctl = [c for c in cells.values() if c["s"] == 0.0 and "exact" in c["proposers"]]
    out.append(("instrument", bool(ctl) and all(a["slip"] == 0 and a["exact"] == c["world"]["steps"] for c in ctl for a in c["proposers"]["exact"].values())))
    if has(D_NEAR_HI) and "exact" in res["proposers"]:
        b, g = fig(D_NEAR_HI, "exact", "blind"), fig(D_NEAR_HI, "exact", "gate"); out.append(("instrument", b["slip"] >= D_BLIND_SLIP and g["slip"] >= D_GATE_SLIP))
    else: out.append(("instrument", None))
    out.append(("claim", bool(ex_cells) and all(c["proposers"]["exact"]["repair"]["slip"] == 0 for c in ex_cells)))
    out.append(("claim", bool(ex_cells) and all(c["proposers"]["exact"]["retry"]["slip"] == 0 for c in ex_cells)))
    if has(D_NEAR_HI) and "exact" in res["proposers"]:
        rt, rp, gt = fig(D_NEAR_HI, "exact", "retry"), fig(D_NEAR_HI, "exact", "repair"), fig(D_NEAR_HI, "exact", "gate")
        out.append(("claim", rt["calls"] >= D_RETRY_CALLS and rp["calls"] == 1.0))
        out.append(("claim", gt["quarter_distance"][-1] > 0 and gt["quarter_distance"][-1] >= D_GATE_DRIFT * gt["quarter_distance"][0]))
        b = fig(D_NEAR_HI, "exact", "blind"); out.append(("claim", b["unsafe"] >= D_UNSAFE_BLIND and rp["unsafe_n"] == 0))
    else: out += [("claim", None)] * 3
    if has(("near", 0.4)) and has(("uniform", 0.4)) and "exact" in res["proposers"]:
        out.append(("claim", all(fig(("near", 0.4), "exact", a)["slip"] > fig(("uniform", 0.4), "exact", a)["slip"] for a in ("blind", "gate"))))
    else: out.append(("claim", None))
    if "net" in res["proposers"] and all(has((g, 0.0)) for g in res["regimes"]):
        out.append(("claim", pooled_slip(res, "net", "blind", 0.0) >= D_NET_BLIND_SLIP))
        out.append(("claim", all(pooled_slip(res, "net", "repair", 0.0) < pooled_slip(res, "net", a, 0.0) for a in ("blind", "gate", "retry"))))
    else: out += [("claim", None)] * 2
    if "net" in res["proposers"] and has(D_NEAR_MID):
        f = {a: fig(D_NEAR_MID, "net", a) for a in ARMS}
        out.append(("claim", all(f["repair"]["slip"] < f[a]["slip"] for a in ("blind", "gate", "retry"))))
        out.append(("claim", all(f["repair"]["mean_distance"] < f[a]["mean_distance"] for a in ("blind", "gate", "retry"))))
        out.append(("claim", f["blind"]["unsafe"] > 0 and f["repair"]["unsafe"] <= D_HALF * f["blind"]["unsafe"]))
    else: out += [("claim", None)] * 3
    return out


# ---------------------------------------------------------------- printing and the command line
def _pc(x: float, k: int = 1) -> str: return f"{100 * x:.{k}f}%"


def print_dynworld(res: dict) -> None:
    print(f"=== the world changes by itself (seed {res['seed']}, {res['episodes']} episodes of {res['steps']} steps per cell, {res['seconds']} s, digest {res['digest']}) ===")
    for key, c in res["cells"].items():
        w = c["world"]; print(f"\n{c['regime']}, s = {c['s']}: {w['steps']} steps, {w['events']} changes of the world {w['event_kinds']}")
        print(f"   {'proposer':8s} {'arm':7s} {'slip':>7s} {'exact':>7s} {'corrupt':>8s} {'distance':>9s} {'calls':>6s} {'unsafe routes':>14s} {'routes missed':>14s}")
        for p in res["proposers"]:
            for a in res["arms"]:
                f = arm_figures(c, p, a)
                print(f"   {p:8s} {a:7s} {_pc(f['slip']):>7s} {_pc(f['exact']):>7s} {f['corrupted']:8d} {f['mean_distance']:9.2f} {f['calls']:6.2f} {_pc(f['unsafe']) + ' of ' + str(f['planned']):>14s} {_pc(f['missed']):>14s}")


def print_verdict_dynworld(checks) -> None:
    print("--- reading rules ---")
    for i, (kind, v) in enumerate(checks, 1):
        print(f"{i:2d} [{kind:10s}] {'yes' if v else ('NO' if v is False else 'not run')}" + ("   (headline A)" if i in DW_HEADLINE else "") + ("   (headline B)" if i in DW_HEADLINE_NET else ""))
    for name, idx in (("A (the exact reader: the age of the snapshot only)", DW_HEADLINE), ("B (the trained network, near regime, s = 0.2)", DW_HEADLINE_NET)):
        head = [checks[i - 1][1] for i in idx]; print(f"HEADLINE {name}:", "ALL YES" if all(head) else "NOT all yes")


QUICK = dict(episodes=2, steps=80, noise=(0.0, 0.2, 0.4))


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="the world changes by itself (docs/DYNAMIC_WORLD_TEST.md)")
    ap.add_argument("--out", default="axi_dynworld.json"); ap.add_argument("--seed", type=int, default=SEED); ap.add_argument("--net", default=os.path.join(RESULTS, NET_FILE), help="the trained network (a .npz)")
    ap.add_argument("--retrain", action="store_true", help="train the network again with the recipe (about 10 minutes) and save it to --net instead of loading it")
    ap.add_argument("--quick", action="store_true", help="a few seconds, small samples: to see that it runs, not to read")
    a = ap.parse_args(argv)
    if a.retrain: net = train_net(say=print); net.save(a.net); print("trained and saved", os.path.abspath(a.net))
    else:
        if not os.path.exists(a.net): raise SystemExit(f"no network at {a.net}: give --net or train one with --retrain")
        net = Net.load(a.net)
    kw = dict(QUICK) if a.quick else {}
    res = run_dynworld(net, a.seed, progress=lambda g, s, e, n: print(f"  {g} s={s}: episode {e}/{n}", flush=True), **kw); res["quick"] = bool(a.quick)
    with open(a.out, "w") as f: json.dump(res, f, indent=1, sort_keys=True)
    print_dynworld(res); print_verdict_dynworld(verdict_dynworld(res)); print("saved", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
