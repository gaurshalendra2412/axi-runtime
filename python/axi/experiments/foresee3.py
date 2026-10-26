"""Item 138 (test 163): the third forward check on the saved ARC-AGI-3 runs, after the second one (docs/FORESEE_CHECK_2.md; its numbers are in the owner's saved foresee_check_2_all.json) found the mover (97.4 % of the predictions) but
learned almost no walls (said "does not move" 7 times in 729 predictions) and showed two different kinds of stop: in ls20 65 of 68 stops had a colour the piece had never stepped onto in front of it, in sp80 and g50t
82 of 88 stops had only colours it had walked on in front of it. Two changes, both written on docs/FORESEE_CHECK_3.md before this is run, and nothing else:
  * the WHOLE FRONT is held: when the piece did not move and the floor is known (the piece has stepped somewhere), every cell in front of it that is not a colour it has walked on and has never been stepped onto is held as
    blocking, not only when exactly one such cell is left (a piece that covers several cells almost never leaves exactly one). A cell the piece later steps onto is released at once, so a wrongly held cell costs one wrong
    "does not move" and no more;
  * the ACTION IS TYPED by its own record on open floor: an action is called stationary when, in at least 4 earlier tries of it with only colours the piece has walked on in front of it, the piece moved in fewer than half;
    it is then predicted "does not move", and a stop of such an action is not taken to say anything about a wall. A stop in front of something is not held against the action (a first trial on the invented worlds, which
    counted every stop, called ordinary actions stationary because they had bumped into walls, and was right in 7 percent of what it said).
Each change can be switched off, so that the effect of each is measured on its own, and with both off the code is the second check exactly (a test holds that). Two AI-written texts the owner pasted suggested these two; they
are not Sir's, and their other claims (a "Layer V" gauge, a computed 92.45 %) are not used.

Comes from pillar II / III of Rajnish Choubey's RFC-001 read into game terms (a refused change is kept as held state, released when the change goes through; for Sir to confirm). What it does NOT show: that a path-finder clears a
level, or what the goal of a level is. A pure function of the saved moves; no model, no cost."""
from __future__ import annotations
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

from axi.engine.ledger import Ledger, MIN_LAW
from axi.experiments.foresee import MIN_TRIES, combine, _actual
from axi.experiments import foresee2 as F2

__all__ = ["evaluate", "from_session", "rates", "by_action", "text", "verdict", "headline", "claims_allowed", "G_LINES", "HEADLINE", "combine", "VARIANTS", "TYPE_MIN", "SECOND"]

TYPE_MIN = 4                       # tries of an action on open floor before it can be called stationary
SECOND = (527, 729)                # what the second check got on the same runs: hits, predicted (owner's foresee_check_2_all.json)
VARIANTS = {"off": (False, False), "front": (True, False), "typed": (False, True), "full": (True, True)}


def evaluate(moves: Iterable[Tuple[object, object, object]], front: bool = True, typed: bool = True, min_law: int = MIN_LAW, carry: Optional[dict] = None) -> dict:
    """`moves`: (action, before, after) of one level in the order played. Each move is predicted from the moves before it (fresh ledger, fresh held cells), then scored. Pure.
    front=False, typed=False is the second check (foresee2.evaluate) exactly. `carry` (item 139, foresee4): None is a fresh start; a dict {"laws": {}, "walked": set(), "use_laws": bool, "use_floor": bool} is what the earlier levels of the same game left
    behind; it is read at the start, used only where it says so, and updated in place at the end of the level."""
    L = Ledger(min_law=min_law); c: Counter = Counter(); last: Dict[object, tuple] = {}; rigid: set = set(); passed: set = set(); walked: set = set()
    if carry and carry.get("use_floor"): walked |= set(carry.get("walked", ()))
    mv: Counter = Counter(); fl: Counter = Counter()      # per action: tries that moved the piece, tries that did not with only walked colours in front
    for action, before, after in moves:
        d = None if "@" in str(action) else L.dominant().get(action)
        if d is not None and d[3] < MIN_TRIES: d = None
        carried = False
        if d is None and carry and carry.get("use_laws") and "@" not in str(action):
            pl = carry["laws"].get(action)
            if pl is not None and any(pl[0] in row for row in before): d = pl; carried = True        # the step this action made in an earlier level, only if the same symbol is on this screen
        pred = None
        if d is not None:
            sym, step, n, tot = d; group = F2._group(L, action, sym, step, n); fp = F2._footprint(before, group, last)
            known = bool(F2._footprint(before, {sym}, last)); h = len(before); w = len(before[0]) if h else 0; dx, dy = step
            T = {(x + dx, y + dy) for x, y in fp} - fp if known else set()
            outside = {t for t in T if not (0 <= t[0] < w and 0 <= t[1] < h)}; inside = T - outside
            wall = known and (bool(outside) or bool(inside & rigid))
            stationary = mv[action] + fl[action] >= TYPE_MIN and 2 * mv[action] < mv[action] + fl[action]
            why = "typed" if (typed and stationary) else ("wall" if wall else None)
            pred = dict(carried=carried, symbol=sym, step=step, known=known, wall=wall, stationary=stationary, why=why, inside=inside,
                        object=known and bool(walked) and any(before[y][x] not in walked for x, y in inside), unknown=known and not walked)
        rec = L.record(action, before, after)
        c["moves"] += 1
        if pred is None:
            c["abstained"] += 1
        else:
            got = _actual(rec, pred["symbol"]); c["predicted"] += 1; hit = got == pred["step"]; said = pred["why"] is not None; a = str(action)
            if carried: c["carried_pred"] += 1; c["carried_hit"] += int((got is None) if said else hit)
            c["step_hit"] += int(hit); c["cell_hit"] += int((got is None) if said else hit); c["act_pred:" + a] += 1
            if pred["known"]: c["known"] += 1; c["known_step_hit"] += int(hit); c["known_cell_hit"] += int((got is None) if said else hit)
            if said: c[pred["why"] + "_said"] += 1; c[pred["why"] + "_right"] += int(got is None); c["cell_said"] += 1; c["cell_right"] += int(got is None)
            if got is None:
                c["did_not_move"] += 1; c["act_dnm:" + a] += 1
                if pred["known"]:
                    k = "dnm_unknown" if pred["unknown"] else ("dnm_object" if pred["object"] else "dnm_floor"); c[k] += 1
                    if k == "dnm_floor": fl[action] += 1; c["dnm_floor_stat"] += int(pred["stationary"])
            elif not hit: c["other_step"] += 1
            if hit: mv[action] += 1
            if pred["known"]:
                if hit:
                    passed |= pred["inside"]; rigid -= pred["inside"]; walked |= {before[y][x] for x, y in pred["inside"]}
                elif got is None and not pred["wall"] and not (typed and pred["stationary"]):
                    cand = {t for t in pred["inside"] if t not in passed and before[t[1]][t[0]] not in walked}
                    if front:
                        if walked and cand: rigid |= cand; c["held_cells"] += len(cand)
                    elif len(cand) == 1: rigid |= cand
        for o in rec.ops:
            if o[0] == "move": last[o[1]] = (o[2][0] + o[3][0], o[2][1] + o[3][1], o[2][2] + o[3][0], o[2][3] + o[3][1])
    if carry is not None:
        for a, dd in L.dominant().items():
            if dd[2] >= min_law and (a not in carry["laws"] or dd[2] >= carry["laws"][a][2]): carry["laws"][a] = dd
        carry["walked"] = set(carry.get("walked", ())) | walked
    return dict(c)


def from_session(session) -> Dict[str, Dict[str, dict]]:
    """variant -> game -> counters, the four variants on the same replayed levels."""
    out: Dict[str, Dict[str, dict]] = {}
    for name, (fr, ty) in VARIANTS.items():
        by: Dict[str, List[dict]] = {}
        for (game, lv), L in sorted(session.ledgers.items()):
            by.setdefault(game, []).append(evaluate(((r.action, r.kb, r.ka) for r in L.rows), fr, ty))
        out[name] = {g: combine(parts) for g, parts in by.items()}
    return out


def rates(c: dict) -> dict:
    r = F2.rates(c); f = lambda a, b: (a / b) if b else None
    r.update(wall_said=c.get("wall_said", 0), wall_precision=f(c.get("wall_right", 0), c.get("wall_said", 0)), typed_said=c.get("typed_said", 0), typed_precision=f(c.get("typed_right", 0), c.get("typed_said", 0)),
             floor_stationary_share=f(c.get("dnm_floor_stat", 0), c.get("dnm_floor", 0)), hits=c.get("cell_hit", 0), held_cells=c.get("held_cells", 0))
    return r


def by_action(c: dict) -> Dict[str, Tuple[int, int]]:
    """action -> (predicted, did not move)."""
    acts = sorted({k.split(":", 1)[1] for k in c if k.startswith("act_pred:")}, key=str)
    return {a: (c.get("act_pred:" + a, 0), c.get("act_dnm:" + a, 0)) for a in acts}


def text(variants: Dict[str, Dict[str, dict]]) -> str:
    pct = lambda x: "-" if x is None else f"{100 * x:.1f}%"; lines = []
    for name in ("off", "front", "typed", "full"):
        pg = variants.get(name) or {}
        if not pg: continue
        lines.append(f"== {name} ({'whole front held' if VARIANTS[name][0] else 'one cell learned'}, {'actions typed' if VARIANTS[name][1] else 'actions not typed'})")
        tot = combine(pg.values())
        for g, c in list(pg.items()) + [("ALL GAMES", tot)]:
            r = rates(c)
            lines.append(f"  {g}: {r['predicted']} predicted; step only right {pct(r['step_only'])}, this version right {pct(r['cell_learned'])} ({r['hits']}); 'does not move' said {r['cell_said']} times "
                         f"(by a held wall {r['wall_said']}, right {pct(r['wall_precision'])}; by action type {r['typed_said']}, right {pct(r['typed_precision'])}); did not move {r['did_not_move']}: "
                         f"unwalked colour in front {r['dnm_object']}, only walked colours {r['dnm_floor']} (on actions called stationary: {pct(r['floor_stationary_share'])}); cells held {r['held_cells']}")
    if variants.get("full"):
        lines.append("== stops by action, full version (predicted moves / did not move):")
        for g, c in variants["full"].items(): lines.append(f"  {g}: " + ", ".join(f"action {a}: {p}/{s}" for a, (p, s) in by_action(c).items()))
    return "\n".join(lines)


# ---------------------------------------------------------------- the reading lines (fixed before the run, docs/FORESEE_CHECK_3.md)
G_LINES = [
    ("guard", "the saved moves of all four arms (K, A, B and C) replay into fresh copies of the games and reproduce every saved flag"),
    ("instrument", "with both changes switched off the check gives the second check's figure on the same runs: 527 right of 729 predicted"),
    ("claim", "with both changes the prediction is right more often than with both off in at least 3 of the 4 games"),
    ("claim", "in ls20 the prediction with both changes is right in at least 75 percent of the predicted moves (the second check: 64.0 percent)"),
    ("claim", "with both changes the prediction said that the piece does not move at least 40 times in all, and was right in at least 85 percent of them"),
    ("claim", "with both changes the prediction is right in at least 80 percent of the predicted moves, over all four arms and games"),
    ("claim", "with both changes the prediction is right in at least 90 percent of the predicted moves, over all four arms and games"),
    ("claim", "at least half of the stops with only walked colours in front happened on an action that its own record called stationary at that moment"),
]
HEADLINE = "THE HOST'S MODEL WITH THE WHOLE FRONT HELD AND THE ACTIONS TYPED PREDICTS WELL ENOUGH TO BUILD A PATH-FINDER ON"


def verdict(variants: Optional[Dict[str, Dict[str, dict]]], replays_ok: Optional[bool]) -> List[Tuple[int, str, str, Optional[bool]]]:
    v: List[Optional[bool]] = [None] * len(G_LINES)
    if replays_ok is not None: v[0] = bool(replays_ok)
    if variants and variants.get("off") and variants.get("full"):
        off, full = variants["off"], variants["full"]; to, tf = rates(combine(off.values())), rates(combine(full.values()))
        v[1] = (to["hits"], to["predicted"]) == SECOND
        v[2] = sum(1 for g in full if g in off and rates(full[g])["cell_learned"] is not None and rates(off[g])["cell_learned"] is not None and rates(full[g])["cell_learned"] > rates(off[g])["cell_learned"]) >= 3
        ls = [rates(c) for g, c in full.items() if g.startswith("ls20")]
        v[3] = bool(ls) and ls[0]["cell_learned"] is not None and ls[0]["cell_learned"] >= 0.75
        v[4] = tf["cell_said"] >= 40 and tf["cell_precision"] is not None and tf["cell_precision"] >= 0.85
        v[5] = tf["cell_learned"] is not None and tf["cell_learned"] >= 0.80
        v[6] = tf["cell_learned"] is not None and tf["cell_learned"] >= 0.90
        v[7] = tf["floor_stationary_share"] is not None and tf["floor_stationary_share"] >= 0.50
    return [(i + 1, t, x, v[i]) for i, (t, x) in enumerate(G_LINES)]


def claims_allowed(checks) -> bool:
    return bool(checks[0][3]) and bool(checks[1][3])


def headline(checks) -> str:
    if not claims_allowed(checks): return "NOT READ: a guard or instrument line is not yes, no figure is claimed until the cause is found"
    r = {i: v for i, _, _, v in checks}
    return HEADLINE if r[3] and r[6] else "MIXED: line 3 " + ("yes" if r[3] else "NO") + ", line 6 " + ("yes" if r[6] else "NO")
