"""The transition ledger: what the host can know exactly about a world it was not told the rules of (concept-map item 132, rows 172 to 178,
docs/HISTORY_LEDGER.md, test 158).

Comes from the structure of Rajnish Choubey's RFC-001, read into game terms (the wording in game terms is for Sir to confirm):
  * a thing is a tuple of WHAT IT IS (z, its identity) and WHAT STATE IT IS IN (x, its payload): the ledger follows an object by what it is
    and reports what changed in its state or place;
  * a rewrite is L <- K -> R, with K the part that is PRESERVED: in an observed move, K is everything that did not change, and what the
    move consumed and made is the rest;
  * the gate checks a change against the held state; the premise check (item 128) checks a sentence against the held state. This is the
    third of the same kind: a claim about the RULES is checked against the held HISTORY.
  * "a rule is a conservation law or a symmetry" is a line quoted in a pasted AI chat as Sir's; its source was not read. It is not relied on.

What it does, from frames only (rows of symbols), with no model and no rule given:
  * `record(action, before, after)` turns two screens into an object-level change: which object moved by how much, which changed colour,
    which vanished or appeared (rows A); cells that differ are kept exactly as well, so nothing depends on the matching being right;
  * counts that have never changed (objects and cells per symbol) are kept as laws once they have held for MIN_LAW moves; a later break is
    reported, and if the count comes back it is marked "came back" (something hidden behind something else, not destroyed) (row B);
  * `hidden_state`: the same screen and the same action gave two different results. The screen does not decide the outcome, so something
    that is not on the screen changed. The ledger cannot say what (a mode, a counter, chance, an animation) (row C);
  * `relations`, `completion`, `no_return_seen`: which actions move the same object by the same, the opposite or a quarter-turned step;
    which of the four unit steps no action has been seen to make; which changes no recorded move was seen to undo (row D);
  * `check_rule`: a claim about the rules, in one of three closed forms, is replayed over every recorded move and the moves that
    contradict it are returned (row E).
An effect that is not next to anything that moved in the same move (a gate that changes far from the pad that was stepped on) is listed as
`apart`; when the count of objects of a symbol changed but the count of its cells did not, two objects touched or one came apart (`touch`),
and that is not an effect.
It edits no framework file and calls no model. What it does NOT do: invent a rule, name the goal, say why a screen is not enough. Touching
objects of one symbol merge into one (the exact changed cells are still kept); something hidden behind something else looks as if it had
gone. It reports; it decides nothing."""
from __future__ import annotations
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

MIN_LAW = 3
UNIT = ((0, -1), (0, 1), (-1, 0), (1, 0))                       # the four unit steps (x to the right, y downwards)
_ORDER = {"move": 0, "recolour": 1, "gone": 2, "new": 3, "reshape": 4, "resize": 5}


def key(frame: Sequence) -> tuple:
    return tuple(r if isinstance(r, str) else tuple(r) for r in frame)


def background(frame: Sequence) -> Any:
    c = Counter(s for r in frame for s in r)
    return min(c, key=lambda s: (-c[s], str(s))) if c else None


def components(frame: Sequence, bg: Any = None) -> List[Tuple[Any, frozenset]]:
    """Same-symbol cells that touch along an edge, background left out: [(symbol, frozenset of (x, y))]."""
    h = len(frame); w = len(frame[0]) if h else 0
    if bg is None: bg = background(frame)
    seen = set(); out = []
    for y in range(h):
        row = frame[y]
        for x in range(w):
            s = row[x]
            if s == bg or (x, y) in seen: continue
            stack = [(x, y)]; seen.add((x, y)); cells = []
            while stack:
                cx, cy = stack.pop(); cells.append((cx, cy))
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in seen and frame[ny][nx] == s:
                        seen.add((nx, ny)); stack.append((nx, ny))
            out.append((s, frozenset(cells)))
    return out


def quantities(comps: Iterable[Tuple[Any, frozenset]]) -> Counter:
    q: Counter = Counter()
    for s, c in comps:
        q[("objects", s)] += 1; q[("cells", s)] += len(c)
    return q


def diff_cells(a: Sequence, b: Sequence) -> List[Tuple[int, int, Any, Any]]:
    out = []
    for y, (ra, rb) in enumerate(zip(a, b)):
        if ra == rb: continue
        for x, (p, q) in enumerate(zip(ra, rb)):
            if p != q: out.append((x, y, p, q))
    return out


def _bbox(cells) -> Tuple[int, int, int, int]:
    xs = [c[0] for c in cells]; ys = [c[1] for c in cells]
    return (min(xs), min(ys), max(xs), max(ys))


def _shape(cells) -> tuple:
    x0, y0, _, _ = _bbox(cells)
    return tuple(sorted((x - x0, y - y0) for x, y in cells))


def _ck(item) -> tuple:
    return (str(item[0]), tuple(sorted(item[1])))


def match(cb: List[Tuple[Any, frozenset]], ca: List[Tuple[Any, frozenset]]) -> Tuple[tuple, bool]:
    """-> (ops, ambiguous). Objects that are in both frames cell for cell cancel; what is left is matched by symbol and shape (a move,
    nearest first), then by cells (a recolour); the rest vanished or appeared. `ambiguous`: two or more objects of one symbol and shape
    on both sides, so which one went where is a choice. Ops:
      ("move", symbol, bbox_before, (dx, dy)); ("recolour", from, to, bbox, size); ("gone", symbol, bbox, size); ("new", symbol, bbox, size);
      ("resize", symbol, bbox_before, bbox_after, cells_after - cells_before): the same symbol, and every cell of the smaller one is in the larger one."""
    nb = Counter(cb); na = Counter(ca); common = nb & na
    lb = sorted(((s, c) for (s, c), n in (nb - common).items() for _ in range(n)), key=_ck)
    la = sorted(((s, c) for (s, c), n in (na - common).items() for _ in range(n)), key=_ck)
    gb = defaultdict(list); ga = defaultdict(list)
    for i, (s, c) in enumerate(lb): gb[(s, _shape(c))].append(i)
    for i, (s, c) in enumerate(la): ga[(s, _shape(c))].append(i)
    ops = []; amb = False; ub = set(); ua = set()
    for k in sorted(gb, key=lambda k: (str(k[0]), k[1])):
        if k not in ga: continue
        B, A = gb[k], ga[k]
        if len(B) > 1 and len(A) > 1: amb = True
        pairs = []
        for bi in B:
            bx, by = _bbox(lb[bi][1])[:2]
            for ai in A:
                ax, ay = _bbox(la[ai][1])[:2]; pairs.append((abs(ax - bx) + abs(ay - by), bi, ai, ax - bx, ay - by))
        for _, bi, ai, dx, dy in sorted(pairs):
            if bi in ub or ai in ua: continue
            ub.add(bi); ua.add(ai); ops.append(("move", k[0], _bbox(lb[bi][1]), (dx, dy)))
    rest_a: Dict[frozenset, List[int]] = defaultdict(list)
    for i, (s, c) in enumerate(la):
        if i not in ua: rest_a[c].append(i)
    for i, (s, c) in enumerate(lb):
        if i in ub: continue
        if rest_a.get(c):
            j = rest_a[c].pop(0); ub.add(i); ua.add(j); ops.append(("recolour", s, la[j][0], _bbox(c), len(c)))
    for i, (s, c) in enumerate(lb):                                   # the same thing with fewer or more cells (a strip that is used up, a gauge that fills): identity kept, state changed
        if i in ub: continue
        rel = [j for j, (s2, c2) in enumerate(la) if j not in ua and s2 == s and (c2 < c or c < c2)]
        if len(rel) == 1:                                           # exactly one candidate, and it is not shared: one thing got smaller or larger (two things that merged or came apart are left as gone and new, for `touch`)
            j = rel[0]; c2 = la[j][1]
            if sum(1 for k, (s3, c3) in enumerate(lb) if k not in ub and s3 == s and (c3 < c2 or c2 < c3)) == 1:
                ub.add(i); ua.add(j); ops.append(("resize", s, _bbox(c), _bbox(c2), len(c2) - len(c)))
    for i, (s, c) in enumerate(lb):
        if i not in ub: ops.append(("gone", s, _bbox(c), len(c)))
    for i, (s, c) in enumerate(la):
        if i not in ua: ops.append(("new", s, _bbox(c), len(c)))
    return tuple(sorted(ops, key=lambda o: (_ORDER[o[0]], repr(o)))), amb


def _zone(op: tuple, radius: int) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = op[2]
    return (x0 - radius, y0 - radius, x1 + radius, y1 + radius)


def _meets(a, b) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def apart_effects(ops: tuple, radius: int = 1) -> tuple:
    """Effects (recolour, gone, new, resize) that sit farther than `radius` cells from every object that moved in the same move; with nothing
    moved, all of them."""
    zones = []
    for o in ops:
        if o[0] == "move":
            x0, y0, x1, y1 = o[2]; dx, dy = o[3]
            zones += [(x0 - radius, y0 - radius, x1 + radius, y1 + radius), (x0 + dx - radius, y0 + dy - radius, x1 + dx + radius, y1 + dy + radius)]
    out = []
    for o in ops:
        if o[0] in ("recolour", "gone", "new", "resize"):
            box = o[2] if o[0] in ("gone", "new") else o[3] if o[0] == "recolour" else (min(o[2][0], o[3][0]), min(o[2][1], o[3][1]), max(o[2][2], o[3][2]), max(o[2][3], o[3][3]))
            if not any(_meets(box, z) for z in zones): out.append(o)
    return tuple(out)


def relation_word(va: Tuple[int, int], vb: Tuple[int, int]) -> Optional[str]:
    """How two steps relate: "same", "opposite", "quarter-turn" (same length, at right angles), or None."""
    if va == vb: return "same"
    if vb == (-va[0], -va[1]): return "opposite"
    if va[0] * vb[0] + va[1] * vb[1] == 0 and va[0] ** 2 + va[1] ** 2 == vb[0] ** 2 + vb[1] ** 2: return "quarter-turn"
    return None


class Record:
    __slots__ = ("i", "action", "kb", "ka", "ops", "diff", "qdelta", "ambiguous", "apart", "broke", "reshaped", "touch")

    def __init__(self, **kw):
        for k in self.__slots__: setattr(self, k, kw.get(k))

    def changed(self) -> bool: return self.kb != self.ka


class Ledger:
    def __init__(self, min_law: int = MIN_LAW, background: Any = None):
        self.min_law = min_law; self.bg = background; self.rows: List[Record] = []; self.first = None; self.current: Counter = Counter()
        self.never: Optional[set] = None; self.changed_ever: set = set(); self.breaks: List[dict] = []; self.outcomes: Dict[tuple, Dict[tuple, list]] = {}

    # ---- recording ----
    def record(self, action: Any, before: Sequence, after: Sequence) -> Record:
        kb, ka = key(before), key(after); i = len(self.rows)
        if i == 0:
            if self.bg is None: self.bg = background(before)             # fixed from the first frame, so the floor is never taken for an object
            self.first = kb; self.current = quantities(components(before, self.bg))
            self.never = {(x, y) for y in range(len(before)) for x in range(len(before[0]) if before else 0)}
        same_dims = len(before) == len(after) and (not before or len(before[0]) == len(after[0]))
        diff = tuple(diff_cells(before, after)) if same_dims else ()
        reshaped = not same_dims; ambiguous = False; qd: Dict[tuple, Tuple[int, int]] = {}
        if reshaped:
            ops = (("reshape", (len(before[0]) if before else 0, len(before)), (len(after[0]) if after else 0, len(after))),); self.never = set()
            qb, qa = quantities(components(before, self.bg)), quantities(components(after, self.bg))
        elif not diff:
            ops = (); qb = qa = None
        else:
            cb = components(before, self.bg); ca = components(after, self.bg)
            ops, ambiguous = match(cb, ca); qb, qa = quantities(cb), quantities(ca)
            if self.never is not None: self.never -= {(x, y) for x, y, _, _ in diff}
        if qb is not None:
            for q in set(qb) | set(qa):
                if qb[q] != qa[q]: qd[q] = (qb[q], qa[q])
        broke = tuple(sorted((q for q in qd if q in self.current and q not in self.changed_ever and i >= self.min_law), key=str))
        for q in broke:
            self.breaks.append(dict(quantity=q, at=i, action=action, was=qd[q][0], now=qd[q][1], held=i, back=False, back_at=None))
        for q, (b, a) in qd.items():
            self.changed_ever.add(q); self.current[q] = a
        for br in self.breaks:
            if not br["back"] and br["at"] < i and self.current[br["quantity"]] == br["was"]: br["back"] = True; br["back_at"] = i
        touch = tuple(sorted({q[1] for q in qd if q[0] == "objects" and ("cells", q[1]) not in qd}, key=str))     # the object count moved, the cell count did not
        apart = tuple(o for o in apart_effects(ops) if not (o[0] in ("gone", "new", "resize") and o[1] in touch))
        rec = Record(i=i, action=action, kb=kb, ka=ka, ops=ops, diff=diff, qdelta=qd, ambiguous=ambiguous, apart=apart, broke=broke, reshaped=reshaped, touch=touch)
        self.rows.append(rec)
        o = self.outcomes.setdefault((kb, action), {})
        if ka in o: o[ka][1] += 1
        else: o[ka] = [i, 1]
        return rec

    # ---- row C: the screen does not decide the outcome ----
    def hidden_state(self, ignore_ticks: bool = False) -> List[dict]:
        """The same screen and the same action gave two different results. With `ignore_ticks`, the screen is compared without the boxes of the
        strips that change size in most moves (`ticks`): a strip that is used up every move makes every screen different, so the strict
        comparison can almost never find a repeated screen. What is left out is a choice, so it is named (`ignored`), and the strip may itself be
        the reason for a difference: it is not compared, not excluded as a cause."""
        if not ignore_ticks: outcomes = self.outcomes; boxes: list = []; cells: frozenset = frozenset()
        else:
            cells = self.tick_cells()
            if not cells: return []
            boxes = [_bbox(cells)]
            outcomes = {}
            for r in self.rows:
                o = outcomes.setdefault((self._mask_cells(r.kb, cells), r.action), {}); ka = self._mask_cells(r.ka, cells)
                if ka in o: o[ka][1] += 1
                else: o[ka] = [r.i, 1]
        out = []
        for (kb, action), o in outcomes.items():
            if len(o) >= 2:
                first = sorted(v[0] for v in o.values())
                d = dict(action=action, results=len(o), moves=first, tries=sum(v[1] for v in o.values()))
                if boxes: d["ignored"] = boxes; d["ignored_cells"] = len(cells)
                out.append(d)
        return sorted(out, key=lambda d: d["moves"][0])

    def tick_cells(self) -> frozenset:
        """The cells that a strip (`ticks`) really changed: in a move where its symbol got smaller or larger, the cells that went to or came from that symbol. Only these are left out of
        the comparison in `hidden_state(ignore_ticks=True)`, not the rectangle around them (the first real run showed that the rectangle of a big object with a small notch covers most of the game)."""
        syms = {t["symbol"] for t in self.ticks()}; out = set()
        for r in self.rows:
            boxes = [b for o in r.ops if o[0] == "move" for b in (o[2], (o[2][0] + o[3][0], o[2][1] + o[3][1], o[2][2] + o[3][0], o[2][3] + o[3][1]))]       # what moved, where it was and where it is
            for s in {o[1] for o in r.ops if o[0] == "resize" and o[1] in syms}:
                out.update((x, y) for x, y, p, q in r.diff if (p == s or q == s) and not any(b[0] <= x <= b[2] and b[1] <= y <= b[3] for b in boxes))
        return frozenset(out)

    @staticmethod
    def _mask_cells(k: tuple, cells) -> tuple:
        by: Dict[int, List[int]] = defaultdict(list)
        for x, y in cells: by[y].append(x)
        rows = list(k)
        for y, xs in by.items():
            if 0 <= y < len(rows):
                r = rows[y]; ls = list(r)
                for x in xs:
                    if 0 <= x < len(ls): ls[x] = "?"
                rows[y] = "".join(ls) if isinstance(r, str) else tuple(ls)
        return tuple(rows)

    @staticmethod
    def _mask(k: tuple, boxes: list) -> tuple:
        rows = list(k)
        for y in sorted({y for b in boxes for y in range(max(0, b[1]), min(len(rows) - 1, b[3]) + 1)}):
            r = rows[y]
            for (x0, y0, x1, y1) in boxes:
                if y0 <= y <= y1:
                    a, b = max(0, x0), min(len(r) - 1, x1) + 1
                    if b > a: r = r[:a] + ("?" * (b - a) if isinstance(r, str) else ("?",) * (b - a)) + r[b:]
            rows[y] = r
        return tuple(rows)

    def ticks(self, share: float = 0.7) -> List[dict]:
        """Symbols whose object gets smaller or larger, in at least `share` of the moves that changed the screen between its first and its last
        resize, and in at least twice `min_law` of them (or in at least half of them if it is the same change every time: a strip that ticks every second move): a strip along an edge that is used up, a gauge that fills. (Counted over the time the strip
        lasted, so one that runs out half way still shows; and nearly always in the same direction.) `box` is the area the object has covered; `delta` the usual change in cells; `moves` the
        number of moves it changed in, `of` how many moves that changed the screen lie between its first and last change."""
        changed = [r for r in self.rows if r.changed()]
        per: Dict[Any, dict] = {}
        for k, r in enumerate(changed):
            hit = set()
            for o in r.ops:
                if o[0] != "resize": continue
                d = per.setdefault(o[1], dict(moves=0, first=k, last=k, deltas=Counter(), signs=Counter(), box=None)); d["deltas"][o[4]] += 1; d["signs"][o[4] > 0] += 1; d["last"] = k
                b = (min(o[2][0], o[3][0]), min(o[2][1], o[3][1]), max(o[2][2], o[3][2]), max(o[2][3], o[3][3]))
                d["box"] = b if d["box"] is None else (min(d["box"][0], b[0]), min(d["box"][1], b[1]), max(d["box"][2], b[2]), max(d["box"][3], b[3]))
                if o[1] not in hit: hit.add(o[1]); d["moves"] += 1
        out = []
        for s, d in sorted(per.items(), key=lambda kv: str(kv[0])):
            span = d["last"] - d["first"] + 1
            n = sum(d["deltas"].values()); steady = max(d["deltas"].values()) >= 0.9 * n                                                       # the same change every time, so also the same direction
            if d["moves"] >= 2 * self.min_law and ((d["moves"] >= share * span and max(d["signs"].values()) >= share * sum(d["signs"].values())) or (d["moves"] >= 0.5 * span and steady)):      # it keeps changing the same way (always smaller, or always larger: a gate that something walks over alternates); one that is exactly the same each time may skip moves (g50t: every second move, first real run)
                out.append(dict(symbol=s, moves=d["moves"], of=span, delta=min(d["deltas"].items(), key=lambda kv: (-kv[1], kv[0]))[0], box=d["box"]))
        return out

    # ---- row B: counts that never changed ----
    def laws(self, min_n: Optional[int] = None) -> List[dict]:
        n = len(self.rows); m = self.min_law if min_n is None else min_n
        if n < m: return []
        return [dict(quantity=q, value=v, held=n) for q, v in sorted(self.current.items(), key=lambda kv: str(kv[0])) if q not in self.changed_ever and v > 0]

    def broken_laws(self) -> List[dict]:
        return [dict(b) for b in self.breaks]

    def signature(self) -> Dict[Any, Dict[tuple, Counter]]:
        out: Dict[Any, Dict[tuple, Counter]] = {}
        for r in self.rows:
            for q, ch in r.qdelta.items(): out.setdefault(r.action, {}).setdefault(q, Counter())[ch] += 1
        return out

    def fixed_cells(self) -> frozenset:
        return frozenset(self.never or ())

    def rigid_symbols(self) -> List[Any]:
        if self.first is None or not self.never: return []
        bg = self.bg; by: Dict[Any, List[tuple]] = defaultdict(list)
        for y, row in enumerate(self.first):
            for x, s in enumerate(row):
                if s != bg: by[s].append((x, y))
        return sorted((s for s, cells in by.items() if all(c in self.never for c in cells)), key=str)

    # ---- row D: the actions ----
    def moves_by_action(self) -> Dict[Any, Counter]:
        out: Dict[Any, Counter] = {}
        for r in self.rows:
            for sv in {(o[1], o[3]) for o in r.ops if o[0] == "move"}: out.setdefault(r.action, Counter())[sv] += 1        # once per try, however many pieces of the symbol moved
        return out

    def tries(self, action: Any) -> int:
        return sum(1 for r in self.rows if r.action == action)

    def dominant(self) -> Dict[Any, Tuple[Any, Tuple[int, int], int, int]]:
        """action -> (symbol, step, times, tries): the step of the object an action moves most often."""
        out = {}
        for a, c in self.moves_by_action().items():
            (s, v), n = min(c.items(), key=lambda kv: (-kv[1], str(kv[0])))
            out[a] = (s, v, n, self.tries(a))
        return out

    def relations(self) -> List[Tuple[Any, Any, str]]:
        d = self.dominant(); acts = sorted(d, key=str); out = []
        for i, a in enumerate(acts):
            for b in acts[i + 1:]:
                (sa, va, _, _), (sb, vb, _, _) = d[a], d[b]
                if sa != sb: continue
                w = relation_word(va, vb)
                if w: out.append((a, b, w))
        return out

    def completion(self) -> List[Tuple[Any, Tuple[int, int]]]:
        """The four unit steps no action has been seen to make, for each symbol that some action moves by a unit step."""
        seen: Dict[Any, set] = defaultdict(set)
        for s, v, _, _ in self.dominant().values():
            if v in UNIT: seen[s].add(v)
        return [(s, u) for s in sorted(seen, key=str) for u in UNIT if u not in seen[s]]

    def no_return_seen(self) -> dict:
        firsts = {}
        for r in self.rows:
            if r.kb != r.ka: firsts.setdefault((r.kb, r.ka), r.i)
        no, back = [], 0
        for r in self.rows:
            if r.kb == r.ka: continue
            if (r.ka, r.kb) in firsts: back += 1
            else: no.append(r.i)
        return dict(changed=back + len(no), returned=back, no_return=no)

    # ---- row E: a claim about the rules, replayed over the history ----
    def check_rule(self, rule: tuple) -> dict:
        """Closed forms: ("moves", action, symbol, (dx, dy)); ("changes_cell", action, (x, y)); ("keeps", ("objects" | "cells", symbol));
        ("on_arrival", symbol, (x, y), (cx, cy)): whenever the symbol's object arrives inside the box that covers (x, y), the cell (cx, cy) changes.
        Returns tested, supported, idle (the action was played and the thing did not move: not a counterexample), counter (the moves that
        contradict it) and a verdict: "refuted", "no counterexample" (and at least one move supports it) or "untested"."""
        kind = rule[0]; counter: List[int] = []; supported = idle = tested = 0
        if kind == "moves":
            _, action, sym, vec = rule
            for r in self.rows:
                if r.action != action: continue
                tested += 1; mv = [o[3] for o in r.ops if o[0] == "move" and o[1] == sym]
                if not mv: idle += 1
                elif tuple(vec) in mv and all(v == tuple(vec) for v in mv): supported += 1
                else: counter.append(r.i)
        elif kind == "changes_cell":
            _, action, cell = rule
            for r in self.rows:
                if r.action != action: continue
                tested += 1
                if any((x, y) == tuple(cell) for x, y, _, _ in r.diff): supported += 1
                else: counter.append(r.i)
        elif kind == "on_arrival":
            _, sym, at, cell = rule; at = tuple(at); cell = tuple(cell)
            for r in self.rows:
                if not any(o[0] == "move" and o[1] == sym and o[2][0] + o[3][0] <= at[0] <= o[2][2] + o[3][0] and o[2][1] + o[3][1] <= at[1] <= o[2][3] + o[3][1]
                           and not (o[2][0] <= at[0] <= o[2][2] and o[2][1] <= at[1] <= o[2][3]) for o in r.ops): continue
                tested += 1
                if any((x, y) == cell for x, y, _, _ in r.diff): supported += 1
                else: counter.append(r.i)
        elif kind == "keeps":
            q = tuple(rule[1])
            for r in self.rows:
                tested += 1
                if q in r.qdelta: counter.append(r.i)
                else: supported += 1
        else:
            raise ValueError(f"unknown rule form: {kind!r}")
        verdict = "refuted" if counter else ("no counterexample" if supported else "untested")
        return dict(rule=rule, tested=tested, supported=supported, idle=idle, counter=counter[:5], counter_n=len(counter), verdict=verdict)

    # ---- the lines the host can add to what the model is shown ----
    def report(self, max_items: int = 6) -> List[str]:
        out: List[str] = []
        strict = self.hidden_state(); tk = self.ticks(); ts = {x["symbol"] for x in tk}
        for h in strict[:max_items]:
            out.append(f"From the same screen, '{h['action']}' gave {h['results']} different results (moves {', '.join(str(m) for m in h['moves'])}): the screen does not decide the outcome.")
        for x in tk[:max_items]:
            b = x["box"]; out.append(f"Something of {x['symbol']} gets smaller or larger in {x['moves']} of the {x['of']} moves that changed the screen while it lasted (usually by {x['delta']:+d} cells), in the area x {b[0]} to {b[2]}, y {b[1]} to {b[3]}: a strip that is used up or a gauge that fills" + (", every move." if x["moves"] >= 0.9 * x["of"] else ", but not in every move (the same change each time)."))
        seen = {(h["action"], h["moves"][0]) for h in strict}
        for h in [h for h in self.hidden_state(ignore_ticks=True) if (h["action"], h["moves"][0]) not in seen][:max_items]:
            out.append(f"From the same screen, not counting the strip that changes every move, '{h['action']}' gave {h['results']} different results (moves {', '.join(str(m) for m in h['moves'])}): the rest of the screen does not decide the outcome (the strip may).")
        lost = [b for b in self.breaks if not b["back"]]
        for b in lost[:max_items]:
            out.append(f"{b['quantity'][0]} of {b['quantity'][1]} had stayed {b['was']} for {b['held']} moves; at move {b['at']} ('{b['action']}') it became {b['now']} and has not come back.")
        if len(self.breaks) > len(lost): out.append(f"{len(self.breaks) - len(lost)} counts that had stayed the same changed and later came back (hidden behind something, or switched).")
        ls = self.laws()
        if ls: out.append("Never changed in " + str(len(self.rows)) + " moves: " + ", ".join(f"{l['quantity'][0]} of {l['quantity'][1]} = {l['value']}" for l in ls[:max_items]) + ".")
        for a, (s, v, n, t) in sorted(self.dominant().items(), key=lambda kv: str(kv[0])):
            out.append(f"'{a}' moves {s} by {v} ({n} of {t} tries).")
        for s, u in self.completion()[:max_items]: out.append(f"No action has been seen to move {s} by {u}.")
        for r in [r for r in self.rows if r.touch][:max_items]:
            out.append(f"Move {r.i} ('{r.action}'): objects of {', '.join(str(t) for t in r.touch)} touched or came apart (the count of objects changed, the cells did not).")
        ap = [(r.i, r.action, o) for r in self.rows for o in r.apart if o[1] not in ts][:max_items]
        for i, a, o in ap: out.append(f"Move {i} ('{a}'): a change away from what moved: {o[0]} {o[1:]}.")
        return out
