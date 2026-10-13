"""Grounding: from words to services (docs/WORDS_TEST.md; concept-map rows 3, 9, 27 and item 87).

The scope filter (`scope.py`) keeps only the changes a request is about, and it finds out what the request is about by reading COORDINATES written in the text. A request in ordinary
words ("remove the billing service") has no coordinates, so the filter sees nothing named and falls back to plain ground-and-complete (docs/SCOPE_TEST.md: "the largest open question").
This module closes that hole in the simplest way that can be tested: the graph carries a table of service NAMES, and the request is read against the table.

What it does, in this order:
  1. every coordinate written in the request, as before;
  2. every service NAME or ALIAS that appears in the request as whole words (longest match first, spans never overlap), so "billing worker" is not read as "billing";
  3. words left over that are not on the stop list: if they are the whole or part of exactly one service's name, that service is named ("partial"); if they fit two or more services
     the request is AMBIGUOUS; if they fit none they are UNRESOLVED.
Status of a request: 'ambiguous' (some phrase fits several services), 'unresolved' (a phrase fits nothing and nothing else is named), 'bound' (at least one service named), 'none' (the request
names no service: "remove every edge with weight 4"; the filter then applies no scope, exactly as before).

The one safety rule built in: a request that is ambiguous, or unresolved with nothing else named, applies NOTHING. The filter does not guess which of two services was meant.

What this is NOT: it is not language understanding. It is exact and token-overlap matching against a name table that the host supplies; a paraphrase it has not been given ("kill the thing that
charges cards") is unresolved. An unknown verb is read as a name that fits nothing, so a request in a style the stop list does not know is refused when it names nothing else. It trusts the
name table; if the table is wrong, the filter is wrong. The model never sees this code; it only changes which changes the host keeps."""
import re
from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, Optional, Sequence, Set, Tuple

Coord = Tuple[int, int]
COORD = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")

STOP = frozenset("""
a an the of to from and or also with without at in on by for as it its that this these those then than into onto s
delete remove decommission drop retire add create connect disconnect detach cut take out new going
service services node nodes component components link links edge edges dependency dependencies dep owns relation weight value
every all any each one both between called named name graph system please
""".split())


def normalize(text: str) -> str:
    """Lower case, coordinates and punctuation removed, one space between words."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", COORD.sub(" ", (text or "").lower())).split())


def _content(tokens: Iterable[str]):
    return [t for t in tokens if t not in STOP and not t.isdigit()]


@dataclass(frozen=True)
class Mention:
    phrase: str
    coords: FrozenSet[Coord]
    how: str                          # 'coordinate' | 'name' | 'partial'


@dataclass(frozen=True)
class Grounding:
    named: FrozenSet[Coord]
    mentions: Tuple[Mention, ...]
    ambiguous: Tuple[Tuple[str, FrozenSet[Coord]], ...]
    unresolved: Tuple[str, ...]

    @property
    def status(self) -> str:
        if self.ambiguous: return "ambiguous"
        if self.unresolved and not self.named: return "unresolved"
        return "bound" if self.named else "none"


class NameIndex:
    """names: {(r, c): 'billing worker'}; aliases: {(r, c): ['payments', 'pay']}. Two services may not have the same canonical name (the host's table is wrong if they do)."""

    def __init__(self, names: Dict[Coord, str], aliases: Optional[Dict[Coord, Sequence[str]]] = None):
        canon = [normalize(n) for n in names.values()]
        if len(set(canon)) != len(canon) or "" in canon: raise ValueError("service names must be non-empty and different from each other")
        self.names = dict(names); self.aliases = {c: tuple(a) for c, a in (aliases or {}).items()}
        forms: Dict[str, Set[Coord]] = {}
        self.tokens: Dict[Coord, FrozenSet[str]] = {}
        for c, n in names.items():
            surface = [n] + list(self.aliases.get(c, ())); toks: Set[str] = set()
            for f in surface:
                nf = normalize(f)
                if nf: forms.setdefault(nf, set()).add(c); toks.update(_content(nf.split()))
            self.tokens[c] = frozenset(toks)
        self.forms = {f: frozenset(cs) for f, cs in forms.items()}
        self._order = sorted(self.forms, key=lambda f: (-len(f.split()), f))


def ground(text: str, index: NameIndex) -> Grounding:
    """Read the request against the name table (see the module doc for the order of the steps)."""
    coords = {(int(a), int(b)) for a, b in COORD.findall(text or "")}
    mentions = [Mention(f"({r},{c})", frozenset({(r, c)}), "coordinate") for r, c in sorted(coords)]
    toks = normalize(text).split(); used = [False] * len(toks); named: Set[Coord] = set(coords); ambiguous = []; unresolved = []
    for f in index._order:                                               # 2. whole-word names and aliases, longest first
        ft = f.split(); n = len(ft)
        for i in range(len(toks) - n + 1):
            if toks[i:i + n] == ft and not any(used[i:i + n]):
                for j in range(i, i + n): used[j] = True
                cs = index.forms[f]
                if len(cs) == 1: mentions.append(Mention(f, cs, "name")); named |= cs
                else: ambiguous.append((f, cs))
    group = []                                                           # 3. leftover content words, grouped when they are next to each other
    for i, t in enumerate(toks + [None]):
        if t is not None and not used[i] and t not in STOP and not t.isdigit(): group.append(t); continue
        if group:
            g = set(group); cands = frozenset(c for c, tk in index.tokens.items() if g <= tk); phrase = " ".join(group)
            if len(cands) == 1: mentions.append(Mention(phrase, cands, "partial")); named |= cands
            elif len(cands) > 1: ambiguous.append((phrase, cands))
            else: unresolved.append(phrase)
            group = []
    return Grounding(frozenset(named), tuple(mentions), tuple(ambiguous), tuple(unresolved))
