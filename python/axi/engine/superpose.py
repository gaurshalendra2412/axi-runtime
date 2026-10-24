"""Hold every reading, choose none (concept-map item 125, docs/SUPERPOSITION.md, test 151).

Prompted by Rajnish Choubey's essay "The Wave, the Unicorn, and the End of Western Psychology": a particle when observed, a wave when not;
"How can one select one? That would be foolish." The method is a standard one: a set of possible states (possible worlds).

Today a change request that fits more than one item applies NOTHING (grounding.py, witnesses.resolve): safe, but the change is lost. Here
the host keeps every reading instead:

  * `apply(change)`: a change that names its item is made in every world; a world where it cannot apply drops out (the request
    presupposes it away); if it applies in none, nothing changes.
  * `branch(readings)`: an ambiguous change splits every world into one world per reading, plus one where it applied nowhere (today's
    rule). Above `cap` worlds the host does what it does today and applies nothing.
  * `clarify(t, item)`: a later message that says which item step t meant keeps only the worlds that read step t that way; `observe(holds)`
    keeps the worlds where a stated fact holds.
  * `expire(now, window)`: a reading still open `window` steps after its change falls back to today's rule (applied nowhere), so a
    change that is never clarified ends exactly as it ends today.
  * `ask(question)`: an answer is given only when every world gives the same one; otherwise the question is not settled yet.

With one world (no ambiguity) every call is the plain change, so a job without ambiguity runs exactly as before. The check before each
change runs in every world, on that world's exact state. It edits no framework file.
What it does NOT show: how often real requests are ambiguous, or that a model writes the description instead of guessing a name."""
from __future__ import annotations
import json
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple

Path = Tuple[Tuple[int, str], ...]           # the readings a world has taken: ((step, item), ...)


def _key(state) -> str: return json.dumps(state, sort_keys=True)


class Readings:
    def __init__(self, state, cap: int = 4):
        self.cap = cap; self.worlds: Dict[str, Tuple[Any, FrozenSet[Path]]] = {_key(state): (state, frozenset({()}))}
        self.max_seen, self.over_cap, self.contradictions, self.expired = 1, 0, 0, 0; self.pending: Dict[int, int] = {}

    # ------------------------------------------------------------ what is held
    @property
    def settled(self) -> bool: return len(self.worlds) == 1

    def state(self):
        """The one state, when it is settled; else None."""
        return next(iter(self.worlds.values()))[0] if self.settled else None

    def states(self) -> List[Any]: return [s for s, _ in self.worlds.values()]

    def _set(self, items: List[Tuple[Any, Path]]):
        merged: Dict[str, Tuple[Any, set]] = {}
        for s, p in items:
            k = _key(s)
            if k in merged: merged[k][1].add(p)
            else: merged[k] = (s, {p})
        self.worlds = {k: (s, frozenset(ps)) for k, (s, ps) in merged.items()}; self.max_seen = max(self.max_seen, len(self.worlds))

    # ------------------------------------------------------------ changes
    def apply(self, change: Callable[[Any], Optional[Any]]) -> str:
        out = [(change(s), p) for s, ps in self.worlds.values() for p in ps]; ok = [(s, p) for s, p in out if s is not None]
        if not ok: return "refused"
        self._set(ok); return "applied"

    def branch(self, t: int, readings: Callable[[Any], List[Tuple[str, Any]]]) -> str:
        """`readings(state)` -> [(item, state after reading the change as about that item), ...] for the items that fit and pass the check.
        Each world also keeps the reading in which the change applied nowhere (item None), today's rule, for `expire`."""
        out = [(after, p + ((t, item),)) for s, ps in self.worlds.values() for p in ps for item, after in readings(s)]
        if not out: return "refused"
        out += [(s, p + ((t, None),)) for s, ps in self.worlds.values() for p in ps]
        if len({_key(s) for s, _ in out}) > self.cap: self.over_cap += 1; return "over cap"
        self._set(out); self.pending[t] = t; return "branched"

    # ------------------------------------------------------------ observations
    def clarify(self, t: int, item: Optional[str]) -> str:
        keep = [(s, p) for s, ps in self.worlds.values() for p in ps if (t, item) in p]
        if not keep: self.contradictions += 1; return "contradicts"
        self._set(keep); self.pending.pop(t, None); return "settled" if self.settled else "narrowed"

    def expire(self, now: int, window: int) -> int:
        """Every reading still open `window` or more steps after its change falls back to today's rule. Returns how many fell back."""
        old = [t for t in self.pending if now - t >= window]
        for t in old: self.clarify(t, None); self.expired += 1
        return len(old)

    def observe(self, holds: Callable[[Any], bool]) -> str:
        keep = [(s, p) for s, ps in self.worlds.values() if holds(s) for p in ps]
        if not keep: self.contradictions += 1; return "contradicts"
        self._set(keep); return "settled" if self.settled else "narrowed"

    def ask(self, question: Callable[[Any], Any]) -> Tuple[Any, bool]:
        answers = {json.dumps(question(s), sort_keys=True, default=str) for s in self.states()}
        return (question(self.states()[0]), True) if len(answers) == 1 else (None, False)
