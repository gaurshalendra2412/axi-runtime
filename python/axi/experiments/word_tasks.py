"""Ordinary-words requests (docs/WORDS_TEST.md; concept-map item 87). The same 140 graphs, kinds and correct changes as `scope_tasks.make_scope_tasks(140, 7)`, with two differences:
  * every service has a NAME (and a few have an alias), and the request is written with names instead of coordinates ("Remove the billing service." instead of "Delete the service at (2,1).");
  * 20 more requests (`make_ambiguity_tasks`) that name a service by a word two services share ("Remove the billing service." when "billing api" and "billing worker" both exist). The correct
    reaction is to change nothing; the expected graph is the graph as it was.
The prompt gets one extra line saying that requests may use names, a legend of the names and one worked example of a node delete (the first hosted-model run showed models writing `DEL[2,0:7]`
for `DEL[2,0]`; the example is shown in the new prompt only, so the new test is NOT the same prompt as the earlier ones and its numbers are compared with theirs only as numbers).
Name pools are built so that no two services in a normal graph share a word, so a name written in a request fits exactly one service; the ambiguity graphs are the only place where it does not.
`add_node` requests keep the coordinates ("Add a new service called inventory at (4,2) with value 5."): the new name is not in the table yet, the coordinates say where."""
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from axi.engine.cico_parser import CICOParser
from axi.engine.gate import Graph
from axi.engine.grounding import NameIndex
from axi.experiments.agent_loop import SYSTEM, RULES_HINT, Task, _serialize, copy_graph
from axi.experiments.cascade import random_graph
from axi.experiments.scope_tasks import make_scope_tasks

Coord = Tuple[int, int]

# canonical names; the words inside different entries never repeat, so a word picks out one service. Aliases (a few) use words that appear nowhere else either.
POOL = ["billing", "auth", "search", "cache", "orders", "payments", "gateway", "email", "inventory", "reports", "metrics", "queue", "catalog", "shipping", "audit", "ledger",
        "user directory", "image resizer", "rate limiter", "session store", "push notifier", "log shipper"]
ALIASES = {"auth": ["authentication"], "email": ["mailer"], "inventory": ["stock"], "metrics": ["telemetry"], "user directory": ["accounts"], "payments": ["checkout"], "queue": ["jobs"]}
# pairs that share their first word (used only by the ambiguity graphs); their words appear in no pool entry
PAIRS = [("billing api", "billing worker"), ("search index", "search crawler"), ("auth store", "auth proxy"), ("order queue", "order router"), ("report builder", "report mailer")]
PAIR_WORDS = frozenset(w for p in PAIRS for n in p for w in n.split()) | {"orders", "reports"}      # every word of the pairs; pool entries that use any of them stay out of the ambiguity graphs

WORDS_NOTE = """
Services also have names, listed under Names. A request may use a name instead of coordinates; your reply must always use coordinates.
  remove the service called "demo", which sits at (2,1) -> DEL[2,1]"""


@dataclass
class WordTask(Task):
    names: Dict[Coord, str] = field(default_factory=dict)
    aliases: Dict[Coord, List[str]] = field(default_factory=dict)
    targets: frozenset = frozenset()          # the services the request is about (what grounding must find); empty for 'delete_weight'
    ambiguous: bool = False

    def index(self) -> NameIndex: return NameIndex(self.names, self.aliases)


def _words(n: str) -> frozenset: return frozenset(n.split())
def _entry_words(n: str) -> frozenset: return _words(n) | frozenset(w for a in ALIASES.get(n, []) for w in a.split())     # the words of a pool entry, aliases included


def pool_is_disjoint() -> bool:
    """No word occurs in two pool entries (aliases included): the design check behind 'a name fits exactly one service'."""
    seen = {}
    for n in POOL:
        for w in set(n.split()) | {w for a in ALIASES.get(n, []) for w in a.split()}:
            if seen.setdefault(w, n) != n: return False
    return True


def name_graph(rng: random.Random, g: Graph, avoid=frozenset()) -> Tuple[Dict[Coord, str], Dict[Coord, List[str]]]:
    """A different name for each service of g; about one name in four with an alias to offer."""
    pool = [n for n in POOL if not (_entry_words(n) & avoid)]; picks = rng.sample(pool, len(g.nodes)); names = {}; aliases = {}
    for c, n in zip(sorted(g.nodes), picks):
        names[c] = n
        if n in ALIASES: aliases[c] = list(ALIASES[n])
    return names, aliases


def _surface(rng, names, aliases, c) -> str:
    """How the request calls the service at c: its name, or (one time in three, when it has one) its alias."""
    if aliases.get(c) and rng.random() < 1 / 3: return rng.choice(aliases[c])
    return names[c]


def _fresh_name(rng, names) -> str:
    used = {w for n in names.values() for w in n.split()} | {w for al in names.values() for w in al.split()}
    return rng.choice([n for n in POOL if not (_entry_words(n) & used)])


def word_instruction(rng: random.Random, task: Task, names, aliases) -> Tuple[str, frozenset]:
    """The request in words and the set of services it is about."""
    d = CICOParser.parse_transition_delta(task.ideal); S = lambda c: _surface(rng, names, aliases, c)
    nd = d.node_deltas; ed = d.edge_deltas
    if task.kind in ("delete_dep", "delete_leaf"):
        n = (nd[0].r, nd[0].c); a = S(n)
        return rng.choice([f"Remove the {a} service.", f"Decommission {a}.", f"Take the {a} service out of the graph."]), frozenset({n})
    if task.kind == "delete_edge":
        e = ed[0]; a, b = S(e.u), S(e.v)
        return rng.choice([f"Remove the {e.relation} link from {a} to {b} with weight {e.weight}.", f"Delete the {e.relation} edge going from {a} to {b}, weight {e.weight}."]), frozenset({e.u, e.v})
    if task.kind == "delete_pair":
        n = [(x.r, x.c) for x in nd if x.op == "DEL"][0]; e = [x for x in ed if n not in (x.u, x.v)][0]; a, b, c = S(n), S(e.u), S(e.v)
        return f"Remove the {a} service, and also the {e.relation} link from {b} to {c} with weight {e.weight}.", frozenset({n, e.u, e.v})
    if task.kind == "delete_weight": return task.instruction, frozenset()
    if task.kind == "add_node":
        x = nd[0]; new = _fresh_name(rng, names)
        return f"Add a new service called {new} at ({x.r},{x.c}) with value {x.val}.", frozenset({(x.r, x.c)})
    e = ed[0]; a, b = S(e.u), S(e.v)                                                            # add_edge
    return f"Add a {e.relation} link from {a} to {b} with weight {e.weight}.", frozenset({e.u, e.v})


def make_word_tasks(n: int, seed: int) -> List[WordTask]:
    """The tasks of make_scope_tasks(n, seed), asked in words."""
    out = []
    for i, t in enumerate(make_scope_tasks(n, seed)):
        rng = random.Random(seed * 1_000_003 + i); names, aliases = name_graph(rng, t.graph)
        instr, targets = word_instruction(rng, t, names, aliases)
        out.append(WordTask(t.kind, t.graph, instr, t.ideal, t.expected, names, aliases, targets, False))
    return out


def make_ambiguity_tasks(n: int, seed: int) -> List[WordTask]:
    """n requests that name a service by the first word of two services' names. The correct reaction is to change nothing."""
    out = []
    for i in range(n):
        rng = random.Random(seed * 7_000_003 + i); g = random_graph(rng); pair = PAIRS[i % len(PAIRS)]
        head = pair[0].split()[0]; names, aliases = name_graph(rng, g, avoid=PAIR_WORDS)
        cells = rng.sample(sorted(g.nodes), 2)
        for c, nm in zip(cells, pair): names[c] = nm; aliases.pop(c, None)
        instr = rng.choice([f"Remove the {head} service.", f"Decommission {head}.", f"Take the {head} service out of the graph."])
        out.append(WordTask("ambiguous", g, instr, "", copy_graph(g), names, aliases, frozenset(cells), True))
    return out


def legend(t: WordTask) -> str:
    parts = []
    for c in sorted(t.names):
        al = t.aliases.get(c); parts.append(f"{t.names[c]}" + (f" (also called {', '.join(al)})" if al else "") + f" = ({c[0]},{c[1]})")
    return "Names: " + "; ".join(parts)


def word_messages(t: WordTask, hint_rules: bool) -> List[dict]:
    return [{"role": "system", "content": SYSTEM + WORDS_NOTE + (RULES_HINT if hint_rules else "")},
            {"role": "user", "content": f"{_serialize(t.graph)}\n\n{legend(t)}\n\nRequest: {t.instruction}"}]
