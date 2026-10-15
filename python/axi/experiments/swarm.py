"""Many agents, one shared state: can several stateless agents propose edits to the SAME graph at the same moment and still leave a correct state? Page with the predictions written BEFORE the
run: docs/SWARM_TEST.md (concept-map item 100). CPU only: no model, no torch, about a minute.

Where the idea comes from: a text the owner pasted on 15 Oct 2026 says that, because an AXI agent needs only the current state and not the history, "10,000 AI agents can read and write to the exact
same enterprise state graph simultaneously" using copy-on-write shadow paging, and that without it "swarms inevitably overwrite each other". The text gives no test. This file makes the part that
can be tested without a model into measurable questions; the part about 10,000 agents, and about what an agent proposes, is NOT tested here.

The rule that is tested (a classical result of graph rewriting, parallel independence for double-pushout rules): two edits made on the same snapshot can be applied in either order, with the same
result, when what both of them MATCH is only what both of them KEEP, and neither makes an item the other makes or uses. For the delta language of the gate:
  L (what an edit matches)   the items it deletes, plus the items it needs to exist: the two ends of every edge it deletes or adds (an end that the same edit adds is not needed from outside);
  K (what it keeps)          L without the items it deletes;
  C (what it creates)        the nodes and edges it adds;
  independent(a, b)          (L_a & L_b) is inside (K_a & K_b), and C_a & C_b, C_a & L_b, C_b & L_a are empty.
Three ways to decide "these two edits clash" are compared with the truth (apply them to the snapshot in both orders with the real gate, `gate.check` / `gate.apply`: they commute iff both orders
are accepted and give the same graph):
  dpo    `independent` above;
  pager  the real `Pager`: two transactions open on the same table, each doing what `Runtime.propose` does for its edit (write the page of every added node and of the ends of every added edge,
         unmap the page of every deleted node); commit one, then the other; a `Conflict` on the second commit means "clash";
  names  the items two edits WRITE (create or delete) overlap.
On top of the pair test: a round of N agents on one snapshot under three schedulers (blind: apply everything, no gate; gated: first come, first served through the gate; merge: pick a set of
mutually independent edits by agent number, apply it in the order of arrival), and a store shared by real threads."""
import hashlib
import json
import random
import threading
import time
from collections import Counter, namedtuple
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from axi.engine.cico_parser import EdgeDelta, NodeDelta, StateTransitionDelta
from axi.engine.gate import Graph, apply, check
from axi.engine.inverse import to_text
from axi.engine.pager import Conflict, Pager
from axi.engine.runtime import vpn_of
from axi.experiments.agent_loop import copy_graph, same_graph
from axi.experiments.kv_large import random_large_graph

SWARM_SEED = 41
PAIR_GRAPHS, PAIR_PER_GRAPH = 20, 6000
PAIR_NODES, PAIR_EDGES, PAIR_GRID = 40, 60, (10, 10)
REGIMES = (("uniform", 0.0), ("hot", 0.5))                       # share of the choices that go to the 3 hub services / 3 hot free cells
ROUND_N, ROUND_ROUNDS, ROUND_PERMS = 16, 100, 10
SWEEP_N = (4, 16, 64)
SWEEP_SIZES = ((40, 60, (10, 10)), (160, 240, (16, 16)), (640, 960, (30, 30)))      # (services, edges, grid)
SWEEP_ROUNDS = 400
THREADS, THREAD_ROUNDS, THREAD_P_HOT = 8, 300, 0.5
THREAD_NODES, THREAD_EDGES, THREAD_GRID = 60, 90, (10, 10)
KIND_WEIGHTS = (("add_node", 15), ("add_edge", 30), ("del_edge", 20), ("del_leaf", 10), ("del_dep", 15), ("multi", 10))
DETECTORS = ("dpo", "pager", "names")

# thresholds of the reading rules (docs/SWARM_TEST.md)
S_MIN_CONFLICT, S_MIN_COMMUTE, S_FALSE_ALARM, S_PAGER_MISS, S_PAGER_FALSE, S_NAMES_MISS, S_GATED_DIFFER, S_BLIND_BAD, S_THREAD_CONFLICT = 2000, 20000, 0.01, 0.10, 0.02, 0.05, 0.80, 0.10, 0.03
S_ACC_SMALL_LO, S_ACC_SMALL_HI, S_ACC_BIG, S_TREND_TOL, S_TREND_MIN, S_SAME_TAKEN = 0.45, 0.80, 0.90, 0.005, 0.01, 0.03
SWARM_HEADLINE = (5, 6, 10, 16)
SWARM_LINES = [
    ("instrument", "Enough pairs of both kinds: over the two regimes at least 2,000 pairs do not commute and at least 20,000 do."),
    ("instrument", "In every pair both edits pass the gate on the shared snapshot, and the first commit of the two pager transactions succeeds (so a Conflict can come only from the second)."),
    ("instrument", "Every thread of the threaded runs finished every round (no worker raised, no barrier broke), for all three stores."),
    ("instrument", "The control store that lets the last writer replace the whole state loses updates: its log replayed through the gate does not give its final state."),
    ("claim", "dpo: no pair that does not commute is called independent (0 missed), in both regimes."),
    ("claim", "dpo: at most 1 % of the pairs that commute are called a clash (false alarms), in both regimes."),
    ("claim", "pager: at least 10 % of the pairs that do not commute raise no Conflict (missed), over the two regimes."),
    ("claim", "pager: at least 2 % of the pairs that commute raise a Conflict (false alarms), in the hot regime."),
    ("claim", "names: at least 5 % of the pairs that do not commute have no written item in common (missed), over the two regimes."),
    ("claim", "merge: in every round of the order test (both regimes) the chosen set gives the same final graph in all 10 arrival orders, and every step of it passes the gate."),
    ("claim", "gated (first come, first served): in the hot regime at least 80 % of the rounds end in different graphs for different arrival orders."),
    ("claim", "blind (no gate): in the hot regime at least 10 % of the rounds end in a graph with an edge whose end is missing, for at least one arrival order."),
    ("claim", "gated and merge: no round ends in a graph with an edge whose end is missing."),
    ("claim", "merge: the share of agents whose edit is chosen falls as the number of agents grows (4, 16, 64) for every graph size, and rises with the graph size (40, 160, 640 services) for every number of agents (no rise or fall against the trend by more than half a point, and at least a point from end to end)."),
    ("claim", "merge: with 64 agents the share chosen is between 45 % and 80 % on the graph of 40 services and at least 90 % on the graph of 640 services."),
    ("claim", "threaded store with the independence check: the log replayed through the gate gives the final state, the state is well formed, every attempt is accounted for, and the gate never refuses an edit that the check passed."),
    ("claim", "threaded store with the gate alone (no independence check): the same four properties hold."),
    ("claim", "threaded store with the independence check: at least 3 % of the attempts are refused as clashes (the clash path was exercised)."),
    ("claim", "gated and merge take about the same share of the edits: within 0.03 of each other in both regimes."),
]


# ---------------------------------------------------------------- the edits and who may share
Foot = namedtuple("Foot", "L K C D")


def footprint(delta: StateTransitionDelta) -> Foot:
    """(L, K, C, D) of an edit as the page defines them. Items are ("n", node) and ("e", (u, v, relation))."""
    made = {(d.r, d.c) for d in delta.node_deltas if d.op == "ADD"}
    C = {("n", (d.r, d.c)) for d in delta.node_deltas if d.op == "ADD"}
    D = {("n", (d.r, d.c)) for d in delta.node_deltas if d.op == "DEL"}
    L = set(D)
    for e in delta.edge_deltas:
        key = ("e", (e.u, e.v, e.relation))
        if e.op == "DEL": L.add(key); D.add(key)
        else: C.add(key)
        for end in (e.u, e.v):
            if end not in made: L.add(("n", end))
    return Foot(frozenset(L), frozenset(L - D), frozenset(C), frozenset(D))


def independent(a: Foot, b: Foot) -> bool:
    """Parallel independence: what both match must be kept by both, and neither makes what the other makes or uses."""
    if (a.L & b.L) - (a.K & b.K): return False
    if (a.C & b.C) or (a.C & b.L) or (b.C & a.L): return False
    return True


def names_clash(a: Foot, b: Foot) -> bool:
    """The naive rule: the two edits write (create or delete) a common item."""
    return bool((a.C | a.D) & (b.C | b.D))


def commutes(g: Graph, a: StateTransitionDelta, b: StateTransitionDelta) -> dict:
    """The truth: apply the two edits to the snapshot in both orders with the real gate. -> {ab, ba, same, commute}: the order a-then-b is accepted, b-then-a is accepted, the two end graphs are equal;
    commute = both orders accepted and equal."""
    ok, ends = [], []
    for first, second in ((a, b), (b, a)):
        h = copy_graph(g); good = check(h, first).ok
        if good: apply(h, first); good = check(h, second).ok
        if good: apply(h, second)
        ok.append(good); ends.append(h)
    same = ok[0] and ok[1] and same_graph(ends[0], ends[1])
    return dict(ab=ok[0], ba=ok[1], same=bool(same), commute=bool(same))


def graph_hash(g: Graph) -> str:
    return hashlib.sha256(repr((sorted(g.nodes.items()), sorted(g.edges.items()))).encode()).hexdigest()[:16]


def well_formed(g: Graph) -> bool:
    return all(u in g.nodes and v in g.nodes for (u, v, _) in g.edges)


# ---------------------------------------------------------------- the real pager as a detector
def pager_template(g: Graph) -> Pager:
    """A pager in which every service of g already has a page, as if each had been added through `Runtime`."""
    p = Pager(len(g.nodes) + 200); tx = p.begin_transaction()
    for n in sorted(g.nodes): tx.write_page(vpn_of(n))
    tx.commit(); return p


def _pager_tx(p: Pager, delta: StateTransitionDelta):
    """What `Runtime.propose` does with the pager for an edit (writes for added services and the ends of added edges, unmaps for deleted services), left uncommitted."""
    tx = p.begin_transaction()
    touched = {(d.r, d.c) for d in delta.node_deltas if d.op == "ADD"}
    for e in delta.edge_deltas:
        if e.op == "ADD": touched |= {e.u, e.v}
    for n in sorted(touched): tx.write_page(vpn_of(n))
    for d in delta.node_deltas:
        if d.op == "DEL": tx.unmap_page(vpn_of((d.r, d.c)))
    return tx


def pager_clash(template: Pager, a: StateTransitionDelta, b: StateTransitionDelta) -> Tuple[bool, bool]:
    """-> (the first commit succeeded, the second commit raised Conflict). Two transactions are open on the same table before either commits."""
    p = Pager(template.capacity); p._free = list(template._free); p._table = dict(template._table)
    ta, tb = _pager_tx(p, a), _pager_tx(p, b)
    ta.commit()
    try: tb.commit()
    except Conflict: return True, True
    return True, False


# ---------------------------------------------------------------- random edits
def hot_items(g: Graph, grid: Tuple[int, int], k: int = 3) -> Tuple[list, list]:
    """The k services with the most edges (ties by coordinates) and the k first free cells in row order: where a crowd of agents would crowd."""
    hubs = sorted(g.nodes, key=lambda n: (-len(g.incident(n)), n))[:k]
    free = [(r, c) for r in range(grid[0]) for c in range(grid[1]) if (r, c) not in g.nodes][:k]
    return hubs, free


def _pick(rng: random.Random, pool: list, hot, p_hot: float):
    if not pool: return None
    if p_hot and rng.random() < p_hot:
        h = [x for x in pool if x in hot]
        if h: return rng.choice(h)
    return rng.choice(pool)


def _nd(op, n, val=None): return NodeDelta(op, n[0], n[1], val)
def _ed(op, k, w): return EdgeDelta(op, k[0], k[1], k[2], w)


def _make(rng: random.Random, g: Graph, grid, kind: str, hot, hot_free, p_hot: float) -> Optional[StateTransitionDelta]:
    nodes = sorted(g.nodes); edges = sorted(g.edges)
    free = [(r, c) for r in range(grid[0]) for c in range(grid[1]) if (r, c) not in g.nodes]
    def edge_from(pool_nodes):
        if len(pool_nodes) < 2: return None
        u = _pick(rng, pool_nodes, hot, p_hot); v = rng.choice(pool_nodes)
        if u is None or u == v: return None
        if rng.random() < .5: u, v = v, u
        k = (u, v, rng.choice(["dep", "owns"]))
        return None if k in g.edges else k
    if kind == "add_node":
        n = _pick(rng, free, hot_free, p_hot)
        return None if n is None else StateTransitionDelta([_nd("ADD", n, rng.randint(1, 9))], [])
    if kind == "add_edge":
        k = edge_from(nodes)
        return None if k is None else StateTransitionDelta([], [_ed("ADD", k, rng.randint(1, 9))])
    if kind == "del_edge":
        pool = edges
        if p_hot and rng.random() < p_hot:
            h = [k for k in edges if k[0] in hot or k[1] in hot]; pool = h or edges
        if not pool: return None
        k = rng.choice(pool); return StateTransitionDelta([], [_ed("DEL", k, g.edges[k])])
    if kind == "del_leaf":
        leaves = [n for n in nodes if not g.incident(n)]
        return None if not leaves else StateTransitionDelta([_nd("DEL", rng.choice(leaves))], [])
    if kind == "del_dep":
        pool = [n for n in nodes if 1 <= len(g.incident(n)) <= 4]; n = _pick(rng, pool, hot, p_hot)
        if n is None: return None
        inc = sorted(g.incident(n)); return StateTransitionDelta([_nd("DEL", n)], [_ed("DEL", k, g.edges[k]) for k in inc])
    if kind == "multi":
        if rng.random() < .5:                                    # a new service wired to an existing one
            n = _pick(rng, free, hot_free, p_hot); v = _pick(rng, nodes, hot, p_hot)
            if n is None or v is None: return None
            k = (n, v, "dep") if rng.random() < .5 else (v, n, "owns")
            return StateTransitionDelta([_nd("ADD", n, rng.randint(1, 9))], [_ed("ADD", k, rng.randint(1, 9))])
        if not edges: return None                                # rewire: delete one edge, add another
        old = rng.choice(edges); new = edge_from(nodes)
        if new is None or new == old: return None
        return StateTransitionDelta([], [_ed("DEL", old, g.edges[old]), _ed("ADD", new, rng.randint(1, 9))])
    raise ValueError(kind)


def random_delta(rng: random.Random, g: Graph, grid: Tuple[int, int], hot=(), hot_free=(), p_hot: float = 0.0) -> StateTransitionDelta:
    """An edit that the gate accepts on `g`: one of six kinds (KIND_WEIGHTS), drawn again until the gate accepts it."""
    kinds = [k for k, w in KIND_WEIGHTS for _ in range(w)]
    for _ in range(80):
        d = _make(rng, g, grid, rng.choice(kinds), hot, hot_free, p_hot)
        if d is not None and check(g, d).ok: return d
    for n in [(r, c) for r in range(grid[0]) for c in range(grid[1]) if (r, c) not in g.nodes]:
        d = StateTransitionDelta([_nd("ADD", n, 1)], [])
        if check(g, d).ok: return d
    raise RuntimeError("the grid is full")


def shape(d: StateTransitionDelta) -> str:
    """A coarse name for an edit: which of +n -n +e -e it contains, e.g. '-e-n' for a service deleted with its edges."""
    toks = {("+" if x.op == "ADD" else "-") + "n" for x in d.node_deltas} | {("+" if x.op == "ADD" else "-") + "e" for x in d.edge_deltas}
    return "".join(sorted(toks, key=lambda t: (t[1], t[0])))


# ---------------------------------------------------------------- 1. pairs
def run_pairs(seed: int = SWARM_SEED, graphs: int = PAIR_GRAPHS, per_graph: int = PAIR_PER_GRAPH, regimes=REGIMES, nodes: int = PAIR_NODES, edges: int = PAIR_EDGES, grid=PAIR_GRID,
              rule: Callable = independent, progress=None, make_graph: Optional[Callable] = None, draw: Optional[Callable] = None) -> dict:
    """Pairs of random edits made on the same snapshot; each detector's verdict against the truth. `rule` is the independence rule under test, `make_graph(rng, nodes, edges, grid)` and `draw(rng, g, grid, hot,
    hot_free, p_hot)` the sources of graphs and edits (parameters so that the checks can try broken rules and scripted pairs)."""
    out = {}; make_graph = make_graph or random_large_graph; draw = draw or random_delta
    for name, p_hot in regimes:
        rng = random.Random(f"{seed}/pairs/{name}")
        tally = {d: Counter() for d in DETECTORS}; why = Counter(); misses = {d: Counter() for d in DETECTORS}; examples = {d: [] for d in DETECTORS}
        first_ok = both_valid = 0; pairs = 0
        for gi in range(graphs):
            g = make_graph(rng, nodes, edges, grid); hot, hot_free = hot_items(g, grid); tmpl = pager_template(g)
            for _ in range(per_graph):
                a = draw(rng, g, grid, hot, hot_free, p_hot); b = draw(rng, g, grid, hot, hot_free, p_hot); pairs += 1
                both_valid += check(g, a).ok and check(g, b).ok
                t = commutes(g, a, b); conflict = not t["commute"]
                if conflict: why["one order only" if t["ab"] != t["ba"] else "both orders refused" if not t["ab"] else "orders differ"] += 1
                fa, fb = footprint(a), footprint(b)
                pf, pc = pager_clash(tmpl, a, b); first_ok += pf
                said = {"dpo": not rule(fa, fb), "pager": pc, "names": names_clash(fa, fb)}
                for det, flagged in said.items():
                    cell = ("tp" if flagged else "fn") if conflict else ("fp" if flagged else "tn"); tally[det][cell] += 1
                    if cell in ("fn", "fp"):
                        misses[det][cell + " " + "/".join(sorted([shape(a), shape(b)]))] += 1
                        if len(examples[det]) < 3: examples[det].append(dict(cell=cell, a=to_text(a), b=to_text(b)))
            if progress: progress(name, gi + 1, graphs)
        n_conf = sum(tally["dpo"][c] for c in ("tp", "fn")); n_comm = pairs - n_conf
        out[name] = dict(p_hot=p_hot, pairs=pairs, conflict=n_conf, commute=n_comm, why=dict(why), both_valid=both_valid, first_commit_ok=first_ok,
                         detectors={d: dict(tally[d]) | {c: tally[d].get(c, 0) for c in ("tp", "fn", "fp", "tn")} for d in DETECTORS},
                         misses={d: dict(misses[d].most_common(8)) for d in DETECTORS}, examples=examples)
    return out


# ---------------------------------------------------------------- 2. rounds: N agents on one snapshot
def blind_apply(g: Graph, delta: StateTransitionDelta) -> None:
    """No gate: an ADD writes its key (over whatever is there), a DEL removes it if it is there. Same order of kinds as `gate.apply`."""
    for e in delta.edge_deltas:
        if e.op == "DEL": g.edges.pop((e.u, e.v, e.relation), None)
    for d in delta.node_deltas:
        if d.op == "DEL": g.nodes.pop((d.r, d.c), None)
    for d in delta.node_deltas:
        if d.op == "ADD": g.nodes[(d.r, d.c)] = d.val
    for e in delta.edge_deltas:
        if e.op == "ADD": g.edges[(e.u, e.v, e.relation)] = e.weight


def schedule_blind(g: Graph, deltas: Sequence[StateTransitionDelta], arrival: Sequence[int]):
    h = copy_graph(g)
    for i in arrival: blind_apply(h, deltas[i])
    return h, list(arrival), 0


def schedule_gated(g: Graph, deltas: Sequence[StateTransitionDelta], arrival: Sequence[int]):
    """First come, first served: each edit goes through the gate against the state as it is when its turn comes."""
    h = copy_graph(g); acc = []
    for i in arrival:
        if check(h, deltas[i]).ok: apply(h, deltas[i]); acc.append(i)
    return h, acc, 0


def schedule_merge(g: Graph, deltas: Sequence[StateTransitionDelta], arrival: Sequence[int], rule: Callable = independent):
    """Choose by agent number (not by arrival) a set of mutually independent edits; apply the chosen ones in the order they arrived, each through the gate. -> (graph, chosen, gate refusals)."""
    foots = {i: footprint(deltas[i]) for i in arrival}; chosen = []
    for i in sorted(arrival):
        if all(rule(foots[i], foots[j]) for j in chosen): chosen.append(i)
    keep = set(chosen); h = copy_graph(g); refused = 0
    for i in arrival:
        if i in keep:
            if check(h, deltas[i]).ok: apply(h, deltas[i])
            else: refused += 1
    return h, chosen, refused


SCHEDULERS = (("blind", schedule_blind), ("gated", schedule_gated), ("merge", schedule_merge))


def run_rounds(seed: int = SWARM_SEED, rounds: int = ROUND_ROUNDS, n_agents: int = ROUND_N, perms: int = ROUND_PERMS, regimes=REGIMES, nodes: int = PAIR_NODES, edges: int = PAIR_EDGES, grid=PAIR_GRID,
               merge: Callable = schedule_merge, make_graph: Optional[Callable] = None, draw: Optional[Callable] = None) -> dict:
    """Rounds of `n_agents` edits on one snapshot, each scheduler run under `perms` random arrival orders. Per scheduler: the share of rounds whose final graph is the same for every order, the share with an
    edge whose end is missing (for any order), the mean share of the edits accepted, and (merge) the gate refusals of edits that the rule had passed."""
    out = {}; make_graph = make_graph or random_large_graph; draw = draw or random_delta
    for name, p_hot in regimes:
        rng = random.Random(f"{seed}/rounds/{name}"); acc = {s: dict(same=0, bad=0, taken=0.0, refused=0, bad_steps=0) for s, _ in SCHEDULERS}
        for r in range(rounds):
            g = make_graph(rng, nodes, edges, grid); hot, hot_free = hot_items(g, grid)
            deltas = [draw(rng, g, grid, hot, hot_free, p_hot) for _ in range(n_agents)]
            arrivals = [rng.sample(range(n_agents), n_agents) for _ in range(perms)]
            for s, fn in SCHEDULERS:
                f = merge if s == "merge" else fn; hashes, taken, bad = set(), 0.0, False
                for arr in arrivals:
                    h, chosen, refused = f(g, deltas, arr); hashes.add(graph_hash(h)); taken += len(chosen) / n_agents; bad |= not well_formed(h); acc[s]["refused"] += refused
                acc[s]["same"] += len(hashes) == 1; acc[s]["bad"] += bad; acc[s]["taken"] += taken / perms
        out[name] = dict(p_hot=p_hot, rounds=rounds, agents=n_agents, perms=perms, schedulers={s: dict(same_graph=a["same"], ill_formed=a["bad"], mean_taken=a["taken"] / rounds, gate_refusals=a["refused"]) for s, a in acc.items()})
    return out


def run_sweep(seed: int = SWARM_SEED, rounds: int = SWEEP_ROUNDS, ns: Sequence[int] = SWEEP_N, sizes=SWEEP_SIZES, make_graph: Optional[Callable] = None, draw: Optional[Callable] = None) -> dict:
    """The share of agents whose edit the merge chooses, for N agents on a graph of S services (no hot spot)."""
    out = {}; make_graph = make_graph or random_large_graph; draw = draw or random_delta
    for (nodes, edges, grid) in sizes:
        for n in ns:
            rng = random.Random(f"{seed}/sweep/{nodes}/{n}"); taken = 0.0
            for _ in range(rounds):
                g = make_graph(rng, nodes, edges, grid); deltas = [draw(rng, g, grid, (), (), 0.0) for _ in range(n)]
                foots = [footprint(d) for d in deltas]; chosen = []
                for i in range(n):
                    if all(independent(foots[i], foots[j]) for j in chosen): chosen.append(i)
                taken += len(chosen) / n
            out[f"{nodes}/{n}"] = dict(services=nodes, agents=n, rounds=rounds, mean_taken=taken / rounds)
    return out


# ---------------------------------------------------------------- 3. stores shared by real threads
class Store:
    """The shared state. `snapshot()` -> (version, copy of the state); `commit(base, delta, snap)` applies the edit if the independence check (against every edit committed after `base`) and the gate (against
    the state as it is) both pass. The check and the commit are one step under one lock."""
    name = "rule"

    def __init__(self, g: Graph):
        self.head = copy_graph(g); self.version = 0; self.log: list = []; self.lock = threading.Lock(); self.stats = Counter()

    def snapshot(self):
        with self.lock: return self.version, copy_graph(self.head)

    def _admit(self, base: int, fp: Foot) -> bool:
        return all(independent(fp, other) for _, other in self.log[base:])

    def commit(self, base: int, delta: StateTransitionDelta, snap: Graph) -> str:
        fp = footprint(delta)
        with self.lock:
            if not self._admit(base, fp): self.stats["clash"] += 1; return "clash"
            if not check(self.head, delta).ok: self.stats["gate_refused"] += 1; return "gate"
            apply(self.head, delta); self.log.append((delta, fp)); self.version += 1; self.stats["accepted"] += 1; return "ok"


class GateOnlyStore(Store):
    """No independence check: the gate against the current state is the only judge."""
    name = "gate"

    def _admit(self, base, fp): return True


class LastWriterStore(Store):
    """Control: the edit is applied to the agent's OWN snapshot and that becomes the state (whole-state last-writer-wins)."""
    name = "lww"

    def commit(self, base, delta, snap):
        with self.lock:
            h = copy_graph(snap)
            if not check(h, delta).ok: self.stats["gate_refused"] += 1; return "gate"
            apply(h, delta); self.head = h; self.log.append((delta, footprint(delta))); self.version += 1; self.stats["accepted"] += 1; return "ok"


def replay(g0: Graph, store: Store) -> bool:
    """The log of accepted edits, applied from the start through the gate one by one, gives the store's final state (no edit was lost, none was made up, every one was admissible when its turn came)."""
    h = copy_graph(g0)
    for delta, _ in store.log:
        if not check(h, delta).ok: return False
        apply(h, delta)
    return same_graph(h, store.head)


def run_threads(store_cls, seed: int = SWARM_SEED, n_threads: int = THREADS, rounds: int = THREAD_ROUNDS, p_hot: float = THREAD_P_HOT, nodes: int = THREAD_NODES, edges: int = THREAD_EDGES, grid=THREAD_GRID) -> dict:
    """`n_threads` threads, `rounds` rounds. In a round every thread takes a snapshot, all wait for each other (so that they all hold the same snapshot), each makes an edit from its snapshot and commits it, racing
    for the lock. After an even round all wait again; after an odd round they do not, so that some snapshots are stale. Afterwards the log is replayed from the start through the gate."""
    rng = random.Random(f"{seed}/threads"); g0 = random_large_graph(rng, nodes, edges, grid); store = store_cls(g0)
    barrier = threading.Barrier(n_threads); results = Counter(); errors: list = []; rlock = threading.Lock()

    def worker(t: int):
        r = random.Random(f"{seed}/thread/{t}")
        try:
            for k in range(rounds):
                base, snap = store.snapshot(); barrier.wait(); hot, hot_free = hot_items(snap, grid)
                d = random_delta(r, snap, grid, hot, hot_free, p_hot); res = store.commit(base, d, snap)
                with rlock: results[res] += 1
                if k % 2 == 0: barrier.wait()
        except Exception as e:                                   # a broken barrier or a bug: record it and let the others finish
            errors.append(repr(e)); barrier.abort()

    ths = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for th in ths: th.start()
    for th in ths: th.join()
    replay_ok = replay(g0, store)
    attempts = n_threads * rounds
    return dict(store=store.name, threads=n_threads, rounds=rounds, attempts=attempts, accepted=results["ok"], clash=results["clash"], gate=results["gate"], errors=errors,
                finished=not errors and sum(results.values()) == attempts, replay_ok=replay_ok, well_formed=well_formed(store.head), accounted=sum(results.values()) == attempts and len(store.log) == store.version == results["ok"],
                gate_after_rule=store.stats["gate_refused"] if store.name == "rule" else 0)


# ---------------------------------------------------------------- summary, reading rules, printout
QUICK = dict(graphs=2, per_graph=120, rounds=6, perms=3, sweep_rounds=6, thread_rounds=24)        # a few seconds: to see that everything runs, not to read


def run_swarm(seed: int = SWARM_SEED, progress=None, quick: bool = False) -> dict:
    t0 = time.time(); q = QUICK if quick else {}
    pairs = run_pairs(seed, graphs=q.get("graphs", PAIR_GRAPHS), per_graph=q.get("per_graph", PAIR_PER_GRAPH), progress=progress)
    rounds = run_rounds(seed, rounds=q.get("rounds", ROUND_ROUNDS), perms=q.get("perms", ROUND_PERMS)); sweep = run_sweep(seed, rounds=q.get("sweep_rounds", SWEEP_ROUNDS))
    threads = {c.name: run_threads(c, seed, rounds=q.get("thread_rounds", THREAD_ROUNDS)) for c in (Store, GateOnlyStore, LastWriterStore)}
    digest = hashlib.sha256(json.dumps([pairs, rounds, sweep], sort_keys=True, default=str).encode()).hexdigest()[:16]
    return dict(seed=seed, quick=quick, pairs=pairs, rounds=rounds, sweep=sweep, threads=threads, digest=digest, seconds=round(time.time() - t0, 1))


def _rate(a: int, b: int) -> float: return a / b if b else 0.0


def _trend(xs: Sequence[float], sign: int) -> bool:
    """xs moves in direction `sign` (+1 up, -1 down) with no step against it by more than S_TREND_TOL, and by at least S_TREND_MIN from end to end."""
    steps = [sign * (b - a) for a, b in zip(xs, xs[1:])]
    return all(s >= -S_TREND_TOL for s in steps) and sign * (xs[-1] - xs[0]) >= S_TREND_MIN


def summarize_swarm(res: dict) -> dict:
    """The figures the reading rules use, computed from the saved result (nothing else)."""
    P, R, S, T = res["pairs"], res["rounds"], res["sweep"], res["threads"]
    pooled = lambda det, cell: sum(P[r]["detectors"][det][cell] for r in P)
    conflict = sum(P[r]["conflict"] for r in P); commute = sum(P[r]["commute"] for r in P)
    acc = lambda n, size: S[f"{size}/{n}"]["mean_taken"]; sizes = sorted({v["services"] for v in S.values()}); ns = sorted({v["agents"] for v in S.values()})
    return dict(
        conflict=conflict, commute=commute, pairs=conflict + commute,
        both_valid=all(P[r]["both_valid"] == P[r]["pairs"] for r in P), first_commit_ok=all(P[r]["first_commit_ok"] == P[r]["pairs"] for r in P),
        dpo_missed={r: P[r]["detectors"]["dpo"]["fn"] for r in P}, dpo_false={r: _rate(P[r]["detectors"]["dpo"]["fp"], P[r]["commute"]) for r in P},
        pager_missed=_rate(pooled("pager", "fn"), conflict), pager_false={r: _rate(P[r]["detectors"]["pager"]["fp"], P[r]["commute"]) for r in P},
        names_missed=_rate(pooled("names", "fn"), conflict),
        merge_same={r: R[r]["schedulers"]["merge"]["same_graph"] == R[r]["rounds"] for r in R}, merge_refusals=sum(R[r]["schedulers"]["merge"]["gate_refusals"] for r in R),
        gated_differ={r: 1 - _rate(R[r]["schedulers"]["gated"]["same_graph"], R[r]["rounds"]) for r in R}, blind_bad={r: _rate(R[r]["schedulers"]["blind"]["ill_formed"], R[r]["rounds"]) for r in R},
        gated_bad=sum(R[r]["schedulers"]["gated"]["ill_formed"] for r in R), merge_bad=sum(R[r]["schedulers"]["merge"]["ill_formed"] for r in R),
        taken={k: v["mean_taken"] for k, v in S.items()},
        falls_with_n=all(_trend([acc(n, s) for n in ns], -1) for s in sizes), rises_with_size=all(_trend([acc(n, s) for s in sizes], +1) for n in ns),
        taken_small=acc(max(ns), min(sizes)), taken_big=acc(max(ns), max(sizes)),
        taken_gap={r: abs(R[r]["schedulers"]["gated"]["mean_taken"] - R[r]["schedulers"]["merge"]["mean_taken"]) for r in R},
        threads={k: dict(finished=v["finished"], replay_ok=v["replay_ok"], well_formed=v["well_formed"], accounted=v["accounted"], gate_after_rule=v["gate_after_rule"], clash_share=_rate(v["clash"], v["attempts"]),
                         accepted=v["accepted"], attempts=v["attempts"]) for k, v in T.items()})


def verdict_swarm(s: dict) -> List[Tuple[str, Optional[bool]]]:
    """The 19 reading rules, in the order of docs/SWARM_TEST.md."""
    t = s["threads"]; ok = lambda x: x["replay_ok"] and x["well_formed"] and x["accounted"]
    return [
        ("instrument", s["conflict"] >= S_MIN_CONFLICT and s["commute"] >= S_MIN_COMMUTE),
        ("instrument", bool(s["both_valid"] and s["first_commit_ok"])),
        ("instrument", all(t[k]["finished"] for k in t)),
        ("instrument", not t["lww"]["replay_ok"]),
        ("claim", all(v == 0 for v in s["dpo_missed"].values())),
        ("claim", all(v <= S_FALSE_ALARM for v in s["dpo_false"].values())),
        ("claim", s["pager_missed"] >= S_PAGER_MISS),
        ("claim", s["pager_false"]["hot"] >= S_PAGER_FALSE),
        ("claim", s["names_missed"] >= S_NAMES_MISS),
        ("claim", all(s["merge_same"].values()) and s["merge_refusals"] == 0),
        ("claim", s["gated_differ"]["hot"] >= S_GATED_DIFFER),
        ("claim", s["blind_bad"]["hot"] >= S_BLIND_BAD),
        ("claim", s["gated_bad"] == 0 and s["merge_bad"] == 0),
        ("claim", bool(s["falls_with_n"] and s["rises_with_size"])),
        ("claim", S_ACC_SMALL_LO <= s["taken_small"] <= S_ACC_SMALL_HI and s["taken_big"] >= S_ACC_BIG),
        ("claim", ok(t["rule"]) and t["rule"]["gate_after_rule"] == 0),
        ("claim", ok(t["gate"])),
        ("claim", t["rule"]["clash_share"] >= S_THREAD_CONFLICT),
        ("claim", all(v <= S_SAME_TAKEN for v in s["taken_gap"].values())),
    ]


def print_swarm(res: dict, title: str = "") -> None:
    P, R, S, T = res["pairs"], res["rounds"], res["sweep"], res["threads"]
    print(f"=== many agents, one state {title} (seed {res['seed']}, {res['seconds']} s, digest {res['digest']}) ===")
    for r, d in P.items():
        print(f"pairs, regime {r}: {d['pairs']} pairs, {d['conflict']} do not commute ({d['why']}), {d['commute']} commute")
        for det in DETECTORS:
            c = d["detectors"][det]; print(f"   {det:6s} caught {c['tp']:5d}  MISSED {c['fn']:5d}  false alarms {c['fp']:5d}  right 'fine' {c['tn']:5d}")
    for r, d in R.items():
        print(f"rounds, regime {r}: {d['agents']} agents, {d['rounds']} rounds, {d['perms']} arrival orders each")
        for s, v in d["schedulers"].items(): print(f"   {s:6s} same graph for every order in {v['same_graph']:3d} rounds, ill-formed in {v['ill_formed']:3d}, mean share taken {v['mean_taken']:.3f}, gate refusals after the rule {v['gate_refusals']}")
    print("share of agents the merge chooses (agents/services):", {k: round(v["mean_taken"], 3) for k, v in S.items()})
    for k, v in T.items(): print(f"threads, {k:5s}: attempts {v['attempts']}, accepted {v['accepted']}, clash {v['clash']}, gate {v['gate']}, replay ok {v['replay_ok']}, well formed {v['well_formed']}, accounted {v['accounted']}, errors {v['errors']}")


def print_verdict_swarm(checks, title: str = "") -> None:
    print(f"--- reading rules {title} ---")
    for i, (kind, v) in enumerate(checks, 1): print(f"{i:2d} [{kind:10s}] {'yes' if v else ('NO' if v is False else 'not run')}   (headline)" if i in SWARM_HEADLINE else f"{i:2d} [{kind:10s}] {'yes' if v else ('NO' if v is False else 'not run')}")
    head = [checks[i - 1][1] for i in SWARM_HEADLINE]; print("HEADLINE:", "ALL YES" if all(head) else "NOT all yes")


def main(argv=None) -> None:
    import argparse, os
    ap = argparse.ArgumentParser(description="many agents, one shared state (docs/SWARM_TEST.md)"); ap.add_argument("--out", default="axi_swarm.json"); ap.add_argument("--seed", type=int, default=SWARM_SEED)
    ap.add_argument("--quick", action="store_true", help="a few seconds, small samples: to see that it runs, not to read")
    a = ap.parse_args(argv)
    res = run_swarm(a.seed, progress=lambda n, i, k: print(f"  pairs {n}: graph {i}/{k}", flush=True), quick=a.quick)
    with open(a.out, "w") as f: json.dump(res, f, indent=1, sort_keys=True, default=str)
    print_swarm(res); print_verdict_swarm(verdict_swarm(summarize_swarm(res))); print("saved", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
