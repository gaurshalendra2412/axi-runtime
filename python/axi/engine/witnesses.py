"""Many eyewitnesses, one entity (concept-map item 122, docs/WITNESSES.md, test 148).

Source: Rajnish Choubey's article "The 99 Eyewitnesses of the Ramayana" (Medium; the text the owner pasted on 24 Oct). Ninety-nine figures of
the epic each describe Sita by one adjective; no single adjective is Sita, and all of them together pin her down. Two structures are taken
from it, and both say the same thing: many independent views, one exact entity.

1. A witness quorum for a change. The gate answers "is this change legal on this state", never "is it the change that was asked": a legal
   change can be the wrong one, the gate's one stated blind spot (docs/AXI_TECHNICAL_REFERENCE.md, 2.4). Here several witnesses (independent
   proposals for the same request) are heard. Each admissible proposal is put into canonical form (its items as a set, so the order and
   exact repeats do not matter); a refused proposal is set aside, or counted in its repaired form when a repair is given. A change is
   committed only when at least `need` witnesses proposed exactly the same canonical change. A tie, or too few in agreement, commits
   nothing. One witness that is wrong but legal cannot commit; a wrong change needs `need` witnesses wrong in exactly the same way.

2. A field of descriptions for an item. An item can be named by where several descriptions meet ("whatever Priya is handling", "the
   Tuesday one at 400") instead of by its name. `resolve` intersects a description with the exact facts the host holds: exactly one item
   fits -> bound; two or more -> ambiguous; none -> unresolved. As in grounding.py, ambiguous and unresolved apply nothing. grounding.py
   reads names and aliases; this reads facts. The model writes the description; the host binds it.

`majority_error` and `quorum_miss` give the arithmetic of the quorum under the assumption that witnesses err independently. Samples of one
model are not independent, so these are the ceiling of the benefit, not a measurement.
It edits no framework file: `check` and `apply` are used as they are, and `quorum` never changes the host graph (the caller applies).
What it does NOT show: that a model's witnesses are independent enough for the quorum to pay; its cost (n witnesses are n calls a step);
anything measured on a model."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, field
from math import comb
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple

from axi.engine.cico_parser import EdgeDelta, NodeDelta, StateTransitionDelta
from axi.engine.gate import Graph, check

Key = Tuple[FrozenSet[tuple], FrozenSet[tuple]]


# ---------------------------------------------------------------- 1. the witness quorum
def canonical(delta: StateTransitionDelta) -> Key:
    """The change a delta makes, as two sets: the order of its items and exact repeats do not matter."""
    return (frozenset((d.op, d.r, d.c, d.val) for d in delta.node_deltas),
            frozenset((e.op, e.u, e.v, e.relation, e.weight) for e in delta.edge_deltas))


def from_key(key: Key) -> StateTransitionDelta:
    """A delta for a canonical change, its items in a fixed order."""
    nodes, edges = key
    return StateTransitionDelta([NodeDelta(*x) for x in sorted(nodes, key=repr)], [EdgeDelta(*x) for x in sorted(edges, key=repr)])


@dataclass
class QuorumResult:
    delta: Optional[StateTransitionDelta]          # the change to commit, or None: nothing is committed
    reason: str                                    # "agreed", "no quorum", "tie", "none admissible"
    votes: int                                     # witnesses behind the leading change
    heard: int                                     # witnesses heard
    admissible: int                                # witnesses whose proposal (or its repair) the gate admits
    tally: List[Tuple[Key, int]] = field(default_factory=list)


def quorum(g: Graph, proposals: Sequence[Optional[StateTransitionDelta]], need: Optional[int] = None,
           repair: Optional[Callable[[Graph, StateTransitionDelta], Optional[StateTransitionDelta]]] = None) -> QuorumResult:
    """Hear every witness; commit the change at least `need` of them (default: a majority of those heard) proposed exactly, else nothing.
    A proposal that did not parse is None. `repair(g, d)` may turn a refused proposal into an admissible one, which then counts as that."""
    n = len(proposals); need = n // 2 + 1 if need is None else need
    keys = []
    for p in proposals:
        if p is None: continue
        d = p
        if not check(g, d).ok:
            if repair is None: continue
            d = repair(g, d)
            if d is None or not check(g, d).ok: continue
        keys.append(canonical(d))
    if not keys: return QuorumResult(None, "none admissible", 0, n, 0)
    tally = sorted(Counter(keys).items(), key=lambda kv: (-kv[1], repr(sorted(kv[0][0], key=repr)), repr(sorted(kv[0][1], key=repr))))
    top = tally[0][1]
    if top < need: return QuorumResult(None, "no quorum", top, n, len(keys), tally)
    if sum(1 for _, c in tally if c == top) > 1: return QuorumResult(None, "tie", top, n, len(keys), tally)
    return QuorumResult(from_key(tally[0][0]), "agreed", top, n, len(keys), tally)


def tally_keys(keys: Sequence[Optional[Any]], need: Optional[int] = None) -> Tuple[Optional[Any], str, int, List[Tuple[Any, int]]]:
    """The same rule as `quorum` for proposals already reduced to a canonical key by the caller (None = refused): returns (winning key or
    None, reason, votes of the leader, tally). `need` defaults to a majority of the keys given. Used where the state is not a CICO graph
    (the list of the fuzzy-state job, whose canonical key is the list a proposal would leave)."""
    n = len(keys); need = n // 2 + 1 if need is None else need
    ks = [k for k in keys if k is not None]
    if not ks: return None, "none admissible", 0, []
    tally = sorted(Counter(ks).items(), key=lambda kv: (-kv[1], repr(kv[0])))
    top = tally[0][1]
    if top < need: return None, "no quorum", top, tally
    if sum(1 for _, c in tally if c == top) > 1: return None, "tie", top, tally
    return tally[0][0], "agreed", top, tally


def gate_repair(g: Graph, d: StateTransitionDelta) -> Optional[StateTransitionDelta]:
    """The framework's own repair (diagnostics.ground_and_complete) as a `repair` for `quorum`: the repaired delta, or None."""
    from axi.engine.diagnostics import ground_and_complete
    r = ground_and_complete(g, d)
    return r.repaired if r is not None and r.repairable else None


def majority_error(p: float, n: int, need: Optional[int] = None) -> float:
    """With n witnesses each wrong with probability p, independently: the chance that at least `need` are wrong. A wrong commit needs that
    AND the wrong ones agreeing exactly, so this is an upper limit of the wrong-commit rate under independence."""
    need = n // 2 + 1 if need is None else need
    return sum(comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(need, n + 1))


def quorum_miss(p: float, n: int, need: Optional[int] = None) -> float:
    """The chance that fewer than `need` witnesses are right, so that the right change is not committed at this ask (independence assumed)."""
    need = n // 2 + 1 if need is None else need
    return sum(comb(n, k) * (1 - p) ** k * p ** (n - k) for k in range(0, need))


# ---------------------------------------------------------------- 2. the field of descriptions
def resolve(items: Mapping[str, Mapping[str, Any]], description: Mapping[str, Any]) -> Tuple[str, List[str]]:
    """The items whose held facts equal every value in the description. ('bound', [name]) for exactly one, ('ambiguous', names) for two or
    more, ('unresolved', []) for none, ('none', []) for an empty description (it names nothing)."""
    if not description: return "none", []
    hits = sorted(n for n, facts in items.items() if all(f in facts and facts[f] == v for f, v in description.items()))
    return ("bound" if len(hits) == 1 else "ambiguous" if hits else "unresolved"), hits


def smallest_field(items: Mapping[str, Mapping[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    """The fewest of an item's own facts that pin it down among the others (fields in a fixed order), or None if all its facts together
    still fit another item: then only its name can name it."""
    from itertools import combinations
    facts = items[name]; fs = sorted(facts)
    for k in range(1, len(fs) + 1):
        for sub in combinations(fs, k):
            d = {f: facts[f] for f in sub}
            if resolve(items, d) == ("bound", [name]): return d
    return None
