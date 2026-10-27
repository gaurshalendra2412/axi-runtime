"""Item 144 (test 169): three small readers on the transition ledger, for the question the ledger could not answer ("what am I supposed to do, and what is new here?"). They read a `Ledger` and change nothing in it; the ledger, its laws and the framework's core are untouched.

Where they come from: the owner's request of 27 Oct 2026, after check 8, to build what fits from the gaps still open (the goal, the actions that do not move anything, what to carry to a new level). The three AI-written texts the owner pasted describe those gaps in Rajnish Choubey's architecture (row 177 "candidate rules, and the probe on which they disagree"; an action known by its effect, not its name; a rule that is carried as a suggestion and is dropped at its first counterexample; a flag when a level shows something not seen before). Whose words they are, and whether this is the reading Sir intends, is for Sir to confirm. Nothing from those texts is used that the saved runs do not show; their claims that were not shown (collision prediction at 100 percent, symmetry or a closed shape as the goal, the 48 percent) are NOT used.

  1. `action_kinds(L)`: each action is told by what it did in this level: a mover (a piece moved in at least half of its tries), a changer (something else on the board changed without a move: a colour, a thing gone or new, a size), idle (nothing but the clock), or mixed. A click is one action ("ACTION6"). The ticking strip is left out.
  2. `novelty(earlier, L)`: the colours and the shapes on this level's first screen that no earlier level's first screen of the same game had.
  3. `carry_over(earlier, L)`: for each plain action, the direction it moved in the earlier levels (firm when it was the same in every earlier level that had a law) and the step it made last time (a suggestion): kept, open, or cancelled by the first move of that colour with another step in this level.
  4. `hypotheses(clears, L)`: row 177's list of candidate goals, from what the earlier clears of the same game changed on the board (the strip and the background left out): repeat that change, empty a colour that went down. They are guesses, none has been tested, and with no clear seen in the game there are none.
  `read_out` puts the four in at most MAX_LINES lines; `show` prints them for every level of the saved runs.

What it does NOT show: that a model plays better with any of these lines (a paid test), that a clear repeats its change in the next level (six clears, four distinct levels), that a changer is how a level is cleared. It is a reader of what happened."""
from __future__ import annotations
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from axi.engine.ledger import Ledger, MIN_LAW, components, _shape
from axi.experiments.foresee import MIN_TRIES

__all__ = ["action_kinds", "novelty", "carry_over", "clear_delta", "hypotheses", "read_out", "show", "MAX_LINES", "KINDS"]

MAX_LINES = 8
KINDS = ("mover", "changer", "idle", "mixed", "once")
_BG_LIKE = ("?",)
_WHAT = {"new": "something new appeared", "gone": "something vanished", "recolour": "a colour changed", "resize": "something grew or shrank", "reshape": "the screen changed size"}


def _base(action) -> str: return str(action).split("@")[0]


def _word(step: Tuple[int, int]) -> str:
    dx, dy = step
    if dx == 0 and dy != 0: return "up" if dy < 0 else "down"
    if dy == 0 and dx != 0: return "left" if dx < 0 else "right"
    return "diagonally"


def _ticks(L: Ledger) -> set: return {t["symbol"] for t in L.ticks()}


def _symbols(frame: Sequence) -> set: return {v for row in frame for v in row}


# ---------------------------------------------------------------- 1 what each action did
def action_kinds(L: Ledger) -> Dict[str, dict]:
    """{action (a click is "ACTION6"): dict(kind, tries, moved, changed, null, with_change, what)}; `what` counts the kinds of change a non-move made ("gone", "new", "recolour", "resize", "reshape")."""
    tsyms = _ticks(L); per: Dict[str, Counter] = {}
    for r in L.rows:
        d = per.setdefault(_base(r.action), Counter()); d["tries"] += 1
        moved = any(o[0] == "move" and o[1] not in tsyms for o in r.ops)
        other = [o for o in r.ops if o[0] != "move" and not (o[0] in ("resize", "gone", "new") and o[1] in tsyms) and not (o[0] == "recolour" and (o[1] in tsyms or o[2] in tsyms))]
        if moved:
            d["moved"] += 1
            if other: d["with_change"] += 1
        elif other:
            d["changed"] += 1; d["what:" + other[0][0]] += 1
        else: d["null"] += 1
    out: Dict[str, dict] = {}
    for a, d in per.items():
        t = d["tries"]
        kind = "once" if t < MIN_TRIES else ("mover" if 2 * d["moved"] >= t else ("changer" if 2 * d["changed"] >= t else ("idle" if d["null"] == t else "mixed")))
        out[a] = dict(kind=kind, tries=t, moved=d["moved"], changed=d["changed"], null=d["null"], with_change=d["with_change"], what={k[5:]: v for k, v in d.items() if k.startswith("what:")})
    return out


# ---------------------------------------------------------------- 2 what is new on this level
def novelty(earlier: Sequence[Ledger], L: Ledger) -> dict:
    """The colours and the shapes of this level's first screen that no earlier level's first screen of the game had. `first_level` is True when there is no earlier level."""
    if L.first is None: return dict(first_level=not earlier, symbols=[], shapes=0)
    if not earlier: return dict(first_level=True, symbols=[], shapes=0)
    seen_sym: set = set(); seen_shape: set = set()
    for E in earlier:
        if E.first is None: continue
        seen_sym |= _symbols(E.first)
        for s, c in components(E.first, E.bg): seen_shape.add((s, _shape(c)))
    new_sym = sorted(_symbols(L.first) - seen_sym, key=str)
    new_shapes = sum(1 for s, c in components(L.first, L.bg) if s not in new_sym and (s, _shape(c)) not in seen_shape)
    return dict(first_level=False, symbols=new_sym, shapes=new_shapes)


# ---------------------------------------------------------------- 3 what to carry from the earlier levels
def carry_over(earlier: Sequence[Ledger], L: Ledger) -> Dict[str, dict]:
    """{plain action: dict(direction, levels, firm, step, symbol, status, direction_changed)}. The direction is the commonest direction word over the earlier levels in which the action had a law of at least MIN_LAW moves; `firm`: it was the same
    in every one of them. The step (a suggestion) is the one of the latest such level. `status`: "open" when this level has not yet moved that colour with the action, "kept" when every move of it so far made the suggested step, "cancelled" at the first move of it with another step."""
    dirs: Dict[str, Counter] = {}; last: Dict[str, tuple] = {}
    for E in earlier:
        for a, (sym, step, n, tot) in E.dominant().items():
            if "@" in str(a) or n < MIN_LAW: continue
            dirs.setdefault(str(a), Counter())[_word(step)] += 1; last[str(a)] = (sym, step)
    now = {str(a): cn for a, cn in L.moves_by_action().items()}; out: Dict[str, dict] = {}
    for a, cn in dirs.items():
        word, k = min(cn.items(), key=lambda kv: (-kv[1], kv[0])); sym, step = last[a]
        seen_now = [(s, v) for (s, v) in now.get(a, {}) if s == sym]
        status = "open" if not seen_now else ("kept" if all(v == step for _, v in seen_now) else "cancelled")
        out[a] = dict(direction=word, levels=sum(cn.values()), firm=(k == sum(cn.values())), step=step, symbol=sym, status=status, direction_changed=any(_word(v) != word for _, v in seen_now))
    return out


# ---------------------------------------------------------------- 4 the candidate goals
def _counts(frame: Sequence) -> Counter:
    c: Counter = Counter()
    for row in frame:
        for v in row: c[v] += 1
    return c


def clear_delta(L: Ledger, last: Sequence) -> Dict[object, int]:
    """What a cleared level changed on the board: the number of cells of each colour on the screen `last` (just before the clearing move) minus on the level's first screen, with the cells the strip changed in that level left out and the strip's
    colour and the background left out. {} for an empty ledger."""
    if L is None or not L.rows: return {}
    cells = L.tick_cells(); tsyms = _ticks(L); first = L.rows[0].kb
    s0, s1 = _counts(L._mask_cells(first, cells)), _counts(L._mask_cells(tuple(r if isinstance(r, str) else tuple(r) for r in last), cells))
    return {s: s1[s] - s0[s] for s in set(s0) | set(s1) if s1[s] != s0[s] and s not in tsyms and s != L.bg and s not in _BG_LIKE}


def hypotheses(clears: Sequence[Dict[object, int]], L: Ledger) -> List[str]:
    """Row 177's candidates, as sentences: from the clears already seen in this game (each a `clear_delta`), the colours that are on this level's first screen. Guesses; none has been tested. [] when no clear has been seen or none of its colours is here."""
    if not clears or L.first is None: return []
    here = _symbols(L.first); out: List[str] = []; last = {s: d for s, d in clears[-1].items() if s in here}
    if last:
        out.append("repeat the change of the last clear: " + ", ".join(f"colour {s} {'+' if d > 0 else ''}{d}" for s, d in sorted(last.items(), key=lambda kv: str(kv[0]))) + " cells")
    down = sorted((s for s in {s for c in clears for s in c} if s in here and all(c.get(s, 0) < 0 for c in clears)), key=str)
    if down and len(clears) > 1: out.append(f"empty or collect colour {', '.join(str(s) for s in down)} (it went down in each of the {len(clears)} earlier clears)")      # with one clear it would say the same as the first guess
    return out[:3]


# ---------------------------------------------------------------- the read-out and the printout
def read_out(earlier: Sequence[Ledger], clears: Sequence[Dict[object, int]], L: Ledger, max_lines: int = MAX_LINES) -> List[str]:
    """At most `max_lines` lines of plain statements of what happened, what is new, what is carried as a suggestion, and the candidate goals. Empty when there is nothing to say."""
    lines: List[str] = []
    for a, k in sorted(action_kinds(L).items()):
        if k["kind"] == "changer": lines.append(f"- {a} did not move anything but changed the board in {k['changed']} of its {k['tries']} tries here ({', '.join(f'{_WHAT.get(w, w)} {n}x' for w, n in sorted(k['what'].items()))}).")
        elif k["kind"] == "idle": lines.append(f"- {a} changed nothing on the board in its {k['tries']} tries here.")
        elif k["kind"] == "mover" and k["with_change"]: lines.append(f"- {a} moved a piece in {k['moved']} of {k['tries']} tries, and something else on the board changed in {k['with_change']} of those.")
    nv = novelty(earlier, L)
    if nv["symbols"] or nv["shapes"]:
        parts = ([f"colour{'s' if len(nv['symbols']) > 1 else ''} " + ", ".join(str(s) for s in nv["symbols"])] if nv["symbols"] else []) + ([f"{nv['shapes']} shape{'s' if nv['shapes'] > 1 else ''}"] if nv["shapes"] else [])
        lines.append("- on this level's first screen and not on an earlier level's: " + " and ".join(parts) + ".")
    co = carry_over(earlier, L)
    if co:
        for status, said in (("cancelled", "this level's first move of it made another step, so that step is dropped"), ("kept", "this level so far agrees"), ("open", "not yet tried here")):
            sel = sorted(a for a, v in co.items() if v["status"] == status)
            if sel: lines.append("- earlier levels' step, a suggestion and not a rule (" + said + "): " + "; ".join(f"{a} {co[a]['direction']} by {max(abs(co[a]['step'][0]), abs(co[a]['step'][1]))}" + ("" if co[a]["firm"] else " (the direction varied)") for a in sel) + ".")
        ch = sorted(a for a, v in co.items() if v["direction_changed"])
        if ch: lines.append("- direction differs from the earlier levels for: " + ", ".join(ch) + ".")
    for h in hypotheses(clears, L): lines.append("- a guess at what clears a level, not tested: " + h + ".")
    return lines[:max_lines]


def show(sessions: Dict[str, object]) -> str:
    """For each arm, game and level of a replayed session (`foresee8.Session8`: `ledgers` and `clears`), the lines `read_out` gives at the end of the level, with the earlier levels and clears of the same game and arm. A printout, not a check: it has no verdict."""
    out: List[str] = []
    for arm, s in sessions.items():
        by_game: Dict[str, List[int]] = {}
        for (g, lv) in s.ledgers: by_game.setdefault(g, []).append(lv)
        cl = {(c["game"], c["level"]): c for c in getattr(s, "clears", [])}
        for g, lvs in sorted(by_game.items()):
            lvs = sorted(lvs)
            for lv in lvs:
                L = s.ledgers[(g, lv)]; earlier = [s.ledgers[(g, e)] for e in lvs if e < lv]
                cd = [clear_delta(s.ledgers[(g, e)], cl[(g, e)]["before"]) for e in lvs if e < lv and (g, e) in cl]
                lines = read_out(earlier, cd, L); mark = ("; cleared by " + str(cl[(g, lv)]["action"]) + (", the change: " + ", ".join(f"{k} {v:+d}" for k, v in sorted(clear_delta(L, cl[(g, lv)]["before"]).items(), key=lambda kv: str(kv[0]))) if clear_delta(L, cl[(g, lv)]["before"]) else ", no colour changed")) if (g, lv) in cl else ""
                out.append(f"== arm {arm} {g} level {lv}: {len(L.rows)} moves recorded{mark}")
                out.extend("   " + x for x in (lines or ["(nothing to say)"]))
    return "\n".join(out)
