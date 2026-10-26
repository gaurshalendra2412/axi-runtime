"""Item 137 (test 162): the second forward check on the saved ARC-AGI-3 runs, after the first one could not look at walls in three of the four games (docs/FORESEE_CHECK.md; its numbers are in docs/results/foresee_check_all.json).

The first check took the moving piece to be exactly one object of one colour; in g50t, ka59 and ls20 it was not, and the wall check could not be made in 553 of 729 predictions. Two changes, both written on
docs/FORESEE_CHECK_2.md before this is run, and nothing else:
  * the mover is the whole group that moves together (every colour, every piece that translated by the same step in the moves of that action), placed where it was last seen to move, not "the one object of
    one colour". An AI chat's phrase for it is displacement covariance; here it is only that: the symbols that moved by the same step as the main mover in at least half of the tries in which it moved;
  * a wall is learned from a move that did nothing, BY POSITION, not by colour: when the piece did not move and exactly one of the cells it stepped onto is not floor and has never been stepped onto, that
    cell is held as blocking. A cell the piece has stepped onto is never held as blocking. The edge of the screen blocks.
It also counts, for every move in which the piece did not move, whether a cell of a colour the piece has never stepped onto was in front of it (the floor is whatever colour it has walked on, not the screen's most common colour), so that "the piece did not move" is not simply taken to mean "a wall".

Comes from pillar II / III of Rajnish Choubey's RFC-001 read into game terms (a refused change is kept as held state; for Sir to confirm). What it does NOT show: that a path-finder clears a level, or what the goal
of a level is. A pure function of the saved moves; no model, no cost."""
from __future__ import annotations
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

from axi.engine.ledger import Ledger, MIN_LAW
from axi.experiments.foresee import MIN_TRIES, EDGE, combine, _actual

__all__ = ["evaluate", "from_session", "rates", "text", "verdict", "headline", "claims_allowed", "G_LINES", "HEADLINE", "combine"]


def _group(L: Ledger, action, sym, step, n) -> set:
    """The symbols that moved by `step` together with `sym` in at least half of the tries of `action` in which `sym` moved by it."""
    c: Counter = Counter()
    for r in L.rows:
        if r.action != action: continue
        ops = [o for o in r.ops if o[0] == "move" and o[3] == step]
        if any(o[1] == sym for o in ops):
            for s in {o[1] for o in ops}: c[s] += 1
    return {s for s, k in c.items() if 2 * k >= n} | {sym}


def _footprint(frame, group, last) -> set:
    cells = set()
    for s in group:
        b = last.get(s)
        if b is None: continue
        for y in range(max(0, b[1]), min(len(frame) - 1, b[3]) + 1):
            row = frame[y]
            for x in range(max(0, b[0]), min(len(row) - 1, b[2]) + 1):
                if row[x] == s: cells.add((x, y))
    return cells


def evaluate(moves: Iterable[Tuple[object, object, object]], min_law: int = MIN_LAW) -> dict:
    """`moves`: (action, before, after) of one level in the order played. Each move is predicted from the moves before it (fresh ledger, fresh held walls), then scored. Pure."""
    L = Ledger(min_law=min_law); c: Counter = Counter(); last: Dict[object, tuple] = {}; rigid: set = set(); passed: set = set(); walked: set = set()      # walked: the colours of the cells the piece has stepped onto
    for action, before, after in moves:
        d = None if "@" in str(action) else L.dominant().get(action)
        if d is not None and d[3] < MIN_TRIES: d = None
        pred = None
        if d is not None:
            sym, step, n, tot = d; group = _group(L, action, sym, step, n); fp = _footprint(before, group, last)
            known = bool(_footprint(before, {sym}, last)); h = len(before); w = len(before[0]) if h else 0; dx, dy = step
            T = {(x + dx, y + dy) for x, y in fp} - fp if known else set()
            outside = {t for t in T if not (0 <= t[0] < w and 0 <= t[1] < h)}; inside = T - outside
            blocked = known and (bool(outside) or bool(inside & rigid))
            ahead_object = known and bool(walked) and any(before[y][x] not in walked for x, y in inside)
            ahead_unknown = known and not walked
            pred = dict(symbol=sym, step=step, known=known, blocked=blocked, inside=inside, object=ahead_object, unknown=ahead_unknown)
        rec = L.record(action, before, after)
        c["moves"] += 1
        if pred is None:
            c["abstained"] += 1
        else:
            got = _actual(rec, pred["symbol"]); c["predicted"] += 1; hit = got == pred["step"]
            c["step_hit"] += int(hit); c["cell_hit"] += int((got is None) if pred["blocked"] else hit)
            if pred["known"]: c["known"] += 1; c["known_step_hit"] += int(hit); c["known_cell_hit"] += int((got is None) if pred["blocked"] else hit)
            if pred["blocked"]: c["cell_said"] += 1; c["cell_right"] += int(got is None)
            if got is None and pred["known"]: c["dnm_unknown" if pred["unknown"] else ("dnm_object" if pred["object"] else "dnm_floor")] += 1
            if got is None: c["did_not_move"] += 1
            elif not hit: c["other_step"] += 1
            if pred["known"]:
                if hit:
                    passed |= pred["inside"]; rigid -= pred["inside"]; walked |= {before[y][x] for x, y in pred["inside"]}
                elif got is None and not pred["blocked"]:
                    cand = {t for t in pred["inside"] if t not in passed and before[t[1]][t[0]] not in walked}
                    if len(cand) == 1: rigid |= cand
        for o in rec.ops:
            if o[0] == "move": last[o[1]] = (o[2][0] + o[3][0], o[2][1] + o[3][1], o[2][2] + o[3][0], o[2][3] + o[3][1])
    return dict(c)


def from_session(session) -> Dict[str, dict]:
    by: Dict[str, List[dict]] = {}
    for (game, lv), L in sorted(session.ledgers.items()):
        by.setdefault(game, []).append(evaluate((r.action, r.kb, r.ka) for r in L.rows))
    return {g: combine(parts) for g, parts in by.items()}


def rates(c: dict) -> dict:
    n = c.get("predicted", 0); f = lambda a, b: (a / b) if b else None
    return dict(moves=c.get("moves", 0), predicted=n, abstained=c.get("abstained", 0), coverage=f(n, c.get("moves", 0)), known_share=f(c.get("known", 0), n),
                step_only=f(c.get("step_hit", 0), n), cell_learned=f(c.get("cell_hit", 0), n), cell_said=c.get("cell_said", 0), cell_precision=f(c.get("cell_right", 0), c.get("cell_said", 0)),
                known=c.get("known", 0), known_step_only=f(c.get("known_step_hit", 0), c.get("known", 0)), known_cell_learned=f(c.get("known_cell_hit", 0), c.get("known", 0)),
                did_not_move=c.get("did_not_move", 0), other_step=c.get("other_step", 0), dnm_object=c.get("dnm_object", 0), dnm_floor=c.get("dnm_floor", 0), dnm_unknown=c.get("dnm_unknown", 0),
                object_share=f(c.get("dnm_object", 0), c.get("dnm_object", 0) + c.get("dnm_floor", 0)))


def text(per_game: Dict[str, dict]) -> str:
    pct = lambda x: "-" if x is None else f"{100 * x:.1f}%"; lines = []; tot = combine(per_game.values()) if per_game else {}
    for g, c in list(per_game.items()) + ([("ALL GAMES", tot)] if per_game else []):
        r = rates(c)
        lines.append(f"{g}: {r['moves']} moves, {r['predicted']} predicted; the mover was found in {pct(r['known_share'])} of them; step only right {pct(r['step_only'])}, with walls learned by position right {pct(r['cell_learned'])} "
                     f"(where the mover was found: {pct(r['known_step_only'])} against {pct(r['known_cell_learned'])}); 'does not move' said {r['cell_said']} times, right {pct(r['cell_precision'])}; "
                     f"the piece did not move {r['did_not_move']} times: a colour it had never stepped onto was in front of it {r['dnm_object']} times, only colours it had walked on {r['dnm_floor']} times ({pct(r['object_share'])} with something in front; not yet known {r['dnm_unknown']}); moved otherwise {r['other_step']}")
    return "\n".join(lines)


# ---------------------------------------------------------------- the reading lines (fixed before the run, docs/FORESEE_CHECK_2.md)
G_LINES = [
    ("guard", "the saved moves of all four arms (K, A, B and C) replay into fresh copies of the games and reproduce every saved flag"),
    ("instrument", "the mover (the group that moves together) is found in at least 60 percent of the predicted moves in each of at least 3 of the 4 games"),
    ("claim", "the prediction with walls learned by position is right more often than step only in at least 3 of the 4 games"),
    ("claim", "the learned prediction said that the piece does not move at least 20 times in all, and was right in at least 85 percent of them"),
    ("claim", "the learned prediction is right in at least 80 percent of the predicted moves, over all four arms and games"),
    ("claim", "the learned prediction is right in at least 90 percent of the predicted moves, over all four arms and games"),
    ("claim", "of the moves in which the piece did not move, at least half had a colour it had never stepped onto in front of it"),
    ("claim", "in no game is the learned prediction more than 2 points worse than step only"),
]
HEADLINE = "THE HOST'S LEARNED MODEL OF WHERE THE PIECE CAN GO PREDICTS WELL ENOUGH TO BUILD A PATH-FINDER ON"


def verdict(per_game: Optional[Dict[str, dict]], replays_ok: Optional[bool]) -> List[Tuple[int, str, str, Optional[bool]]]:
    v: List[Optional[bool]] = [None] * len(G_LINES)
    if replays_ok is not None: v[0] = bool(replays_ok)
    if per_game:
        tot = rates(combine(per_game.values())); rs = {g: rates(c) for g, c in per_game.items()}
        v[1] = sum(1 for r in rs.values() if r["known_share"] is not None and r["known_share"] >= 0.60) >= 3
        v[2] = sum(1 for r in rs.values() if r["cell_learned"] is not None and r["step_only"] is not None and r["cell_learned"] > r["step_only"]) >= 3
        v[3] = tot["cell_said"] >= 20 and tot["cell_precision"] is not None and tot["cell_precision"] >= 0.85
        v[4] = tot["cell_learned"] is not None and tot["cell_learned"] >= 0.80
        v[5] = tot["cell_learned"] is not None and tot["cell_learned"] >= 0.90
        v[6] = tot["object_share"] is not None and tot["object_share"] >= 0.50
        v[7] = all(r["cell_learned"] is not None and r["step_only"] is not None and r["cell_learned"] >= r["step_only"] - 0.02 for r in rs.values())
    return [(i + 1, t, x, v[i]) for i, (t, x) in enumerate(G_LINES)]


def claims_allowed(checks) -> bool:
    return bool(checks[0][3]) and bool(checks[1][3])


def headline(checks) -> str:
    if not claims_allowed(checks): return "NOT READ: a guard or instrument line is not yes, no figure is claimed until the cause is found"
    r = {i: v for i, _, _, v in checks}
    return HEADLINE if r[3] and r[5] else "MIXED: line 3 " + ("yes" if r[3] else "NO") + ", line 5 " + ("yes" if r[5] else "NO")
