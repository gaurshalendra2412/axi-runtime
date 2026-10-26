"""Item 140 (test 165): the fifth forward check on the saved ARC-AGI-3 runs. The first four checks asked whether the host can say where the piece goes BEFORE a move. This one asks what the host's record says about the moves that
were played and about the moves it still cannot answer. Four measurements on the same saved moves, nothing new played, no model, no cost:
  1. REPEATS ("reset, repeat, pull again"): of the moves in which the piece did not move, how many were the same action tried before from the same place of the piece, which had not moved it then either. The repeat of the whole
     SCREEN is not the measure: the screen almost never repeats (208 to 227 of 240 screens in every arm were new), so the repo's "repeat no-ops" column (0, 0 and 1 in 240 moves) cannot see a repeat that is only the same place.
  2. THE GAP ("subset performed, superset claimed; the gap is where the human works"): every move the check could not predict is put under one reason: a click, the action's first try, an action tried once that did move the piece,
     an action tried once that did not, an action tried twice or more that never moved anything. The list is what the ledger hands over.
  3. FILLING THE GAP: two additions, each off by default, each switched on alone and together. `once`: an action that moved the piece in its one earlier try is predicted from that try (the third check waits for two). `elim`: an
     action never tried in this level is predicted to make the one unit step no other action has made, when exactly one of the four is missing for the main mover (`Ledger.completion`, row D of the ledger, which until now only
     reported it). Both are read from the ledger as it is; the ledger is not changed.
  4. THE DELETE TEST: the third check's predictor with each of its five parts deleted alone (the whole front held, the action typed, the edge of the screen, the floor colours, the mover as a group), to see which parts carry the
     result and which are only there. Also counted, for the reading of the action names: the unit step each action made on the main mover in every level.
With every part on and both additions off the counters are the third check's full version, counter for counter (a test holds that). The additions use only the moves before the move they predict; `earlier` (the laws of earlier
levels of the same game) is read only to tag which moves a carried law (the fourth check) would have answered anyway, never to predict.

Comes from Rajnish Choubey's writing as read in the owner's reading of the Cinderella Project pages ("AI Maths: Reset. Repeat. Pull Again. The Mechanism #1": a pull that keeps nothing between shots and does not aggregate; the
subset that is done against the superset that is claimed; the circle that returns to its first point), the Delete Test of his 99 % / 1 % essays, and row D of the ledger (`relations`, `completion`). Whose words those pages are
(the pages carry the byline of Saraswati Choubey and text signed by an AI) is for Sir to confirm; here they are read for structure only. What it does NOT show: that a veto of the repeats, or a filled gap, makes a playing model
use fewer moves (that is a paid test), that a path-finder clears a level, or what the goal of a level is. A pure function of the saved moves."""
from __future__ import annotations
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

from axi.engine.ledger import Ledger, MIN_LAW, UNIT, key
from axi.experiments.foresee import MIN_TRIES, combine, _actual
from axi.experiments import foresee2 as F2
from axi.experiments import foresee3 as F3

__all__ = ["evaluate", "from_session", "rates", "text", "verdict", "headline", "claims_allowed", "G_LINES", "HEADLINE", "combine", "VARIANTS", "PARTS", "REASONS", "THIRD_FULL", "ACTIONS"]

TYPE_MIN = F3.TYPE_MIN
THIRD_FULL = (603, 729)                         # the third check's full version on the same runs: right, predicted (owner's saved foresee_check_3_all.json)
PARTS = ("front", "typed", "edge", "floor", "group")
REASONS = ("click", "new", "once_moved", "once_nomove", "nolaw")
ACTIONS = ("ACTION1", "ACTION2", "ACTION3", "ACTION4")
VARIANTS = {"full": {}, "-front": dict(front=False), "-typed": dict(typed=False), "-edge": dict(edge=False), "-floor": dict(floor=False), "-group": dict(group=False),
            "once": dict(once=True), "elim": dict(elim=True), "once+elim": dict(once=True, elim=True)}


def _main_steps(L: Ledger) -> Dict[object, Tuple[int, int]]:
    """action -> the unit step it made most often on the main mover (the symbol with most unit steps in all), for actions that made it at least twice."""
    dom = L.dominant(); by: Counter = Counter()
    for a in sorted(dom, key=str):
        s, v, n, t = dom[a]
        if v in UNIT: by[s] += n
    if not by: return {}
    main = min(by, key=lambda s: (-by[s], str(s)))
    return {a: dom[a][1] for a in sorted(dom, key=str) if dom[a][0] == main and dom[a][1] in UNIT and dom[a][2] >= MIN_TRIES}


def _elimination(L: Ledger):
    """(symbol, the one missing unit step, reference action, its step, its count) when exactly one of the four unit steps is missing for the main mover, else None."""
    dom = L.dominant(); by: Counter = Counter(); ref: Dict[object, tuple] = {}
    for a in sorted(dom, key=str):
        s, v, n, t = dom[a]
        if v in UNIT:
            by[s] += n
            if s not in ref or n > ref[s][2]: ref[s] = (a, v, n)
    if not by: return None
    main = min(by, key=lambda s: (-by[s], str(s)))
    missing = [u for s, u in L.completion() if s == main]
    if len(missing) != 1: return None
    return (main, missing[0]) + ref[main]


def evaluate(moves: Iterable[Tuple[object, object, object]], front: bool = True, typed: bool = True, edge: bool = True, floor: bool = True, group: bool = True,
             once: bool = False, elim: bool = False, min_law: int = MIN_LAW, earlier: Optional[dict] = None) -> dict:
    """`moves`: (action, before, after) of one level in the order played. Each move is predicted from the moves before it (a fresh ledger), then scored. Pure. With the five parts on and once, elim off it is the third check's
    full version (foresee3.evaluate(moves, True, True)) counter for counter. `earlier`: the laws of earlier levels of the same game, read only to tag, never to predict."""
    L = Ledger(min_law=min_law); c: Counter = Counter(); last: Dict[object, tuple] = {}; rigid: set = set(); passed: set = set(); walked: set = set()
    mv: Counter = Counter(); fl: Counter = Counter(); seen: Dict[tuple, tuple] = {}; posseen: Dict[tuple, bool] = {}
    covers = lambda action, before: earlier is not None and earlier.get(action) is not None and any(earlier[action][0] in row for row in before)
    for action, before, after in moves:
        a = str(action); click = "@" in a; dom = None if click else L.dominant().get(action); tries = L.tries(action)
        d = dom; used = None; eref = None
        if d is not None and d[3] < MIN_TRIES:
            if once: used = "once"
            else: d = None
        if d is None and elim and not click and tries == 0:
            e = _elimination(L)
            if e is not None: d = (e[0], e[1], 0, 0); eref = e[2:]; used = "elim"
        kb, ka = key(before), key(after)
        if (action, kb) in seen:                                              # the same action on the very same screen: its earlier outcome
            pc, pa = seen[(action, kb)]; c["rep_screen"] += 1
            if pc: c["rep_screen_changed"] += 1; c["rep_screen_same"] += int(ka == pa)
            else: c["rep_screen_null"] += 1; c["rep_screen_null_null"] += int(kb == ka)
        seen[(action, kb)] = (kb != ka, ka); c["null_moves"] += int(kb == ka)
        pred = None; pk = None
        if d is not None:
            sym, step, n, tot = d
            grp = (F2._group(L, eref[0], sym, eref[1], eref[2]) if eref else F2._group(L, action, sym, step, n)) if group else {sym}
            fp = F2._footprint(before, grp, last)
            known = bool(F2._footprint(before, {sym}, last)); h = len(before); w = len(before[0]) if h else 0; dx, dy = step
            T = {(x + dx, y + dy) for x, y in fp} - fp if known else set()
            outside = {t for t in T if not (0 <= t[0] < w and 0 <= t[1] < h)}; inside = T - outside
            wall = known and ((edge and bool(outside)) or bool(inside & rigid))
            stationary = mv[action] + fl[action] >= TYPE_MIN and 2 * mv[action] < mv[action] + fl[action]
            why = "typed" if (typed and stationary) else ("wall" if wall else None)
            pred = dict(symbol=sym, step=step, known=known, wall=wall, stationary=stationary, why=why, inside=inside,
                        object=known and bool(walked) and any(before[y][x] not in walked for x, y in inside), unknown=known and not walked)
            if known and fp: pk = (action, frozenset(fp))
        rec = L.record(action, before, after)
        c["moves"] += 1
        if pred is None:
            r = "click" if click else ("new" if tries == 0 else ("once_nomove" if tries == 1 else "nolaw")) if dom is None else "once_moved"
            c["abstained"] += 1; c["abs:" + r] += 1; c["abs_cov:" + r] += int(covers(action, before))
        else:
            got = _actual(rec, pred["symbol"]); c["predicted"] += 1; hit = got == pred["step"]; said = pred["why"] is not None
            ok = (got is None) if said else hit
            c["step_hit"] += int(hit); c["cell_hit"] += int(ok); c["act_pred:" + a] += 1
            if used: c[used + "_pred"] += 1; c[used + "_hit"] += int(ok); c[used + "_covered"] += int(covers(action, before))
            if pred["known"]: c["known"] += 1; c["known_step_hit"] += int(hit); c["known_cell_hit"] += int(ok)
            if said: c[pred["why"] + "_said"] += 1; c[pred["why"] + "_right"] += int(got is None); c["cell_said"] += 1; c["cell_right"] += int(got is None)
            if got is None:
                c["did_not_move"] += 1; c["act_dnm:" + a] += 1
                if pred["known"]:
                    k = "dnm_unknown" if pred["unknown"] else ("dnm_object" if pred["object"] else "dnm_floor"); c[k] += 1
                    if k == "dnm_floor": fl[action] += 1; c["dnm_floor_stat"] += int(pred["stationary"])
            elif not hit: c["other_step"] += 1
            if hit: mv[action] += 1
            if pk is not None:                                                # the same action from the same place of the piece
                if pk in posseen:
                    c["rep_pos"] += 1; c["rep_pos_null"] += int(posseen[pk]); c["rep_pos_null_null"] += int(posseen[pk] and got is None)
                posseen[pk] = got is None
            if pred["known"]:
                if hit:
                    passed |= pred["inside"]; rigid -= pred["inside"]; walked |= {before[y][x] for x, y in pred["inside"]}
                elif got is None and not pred["wall"] and not (typed and pred["stationary"]):
                    wset = walked if floor else frozenset()
                    cand = {t for t in pred["inside"] if t not in passed and before[t[1]][t[0]] not in wset}
                    if front:
                        if walked and cand: rigid |= cand; c["held_cells"] += len(cand)
                    elif len(cand) == 1: rigid |= cand
        for o in rec.ops:
            if o[0] == "move": last[o[1]] = (o[2][0] + o[3][0], o[2][1] + o[3][1], o[2][2] + o[3][0], o[2][3] + o[3][1])
    for act, v in _main_steps(L).items(): c[f"step:{act}:{v[0]},{v[1]}"] += 1
    for _, _, wd in L.relations(): c["rel:" + wd] += 1
    return dict(c)


def from_session(session) -> Dict[str, Dict[str, dict]]:
    """variant -> game -> counters, the nine variants on the same replayed levels. The levels of a game are taken in order; `earlier` only tags."""
    out: Dict[str, Dict[str, dict]] = {}
    for name, flags in VARIANTS.items():
        by: Dict[str, List[dict]] = {}; ear_by: Dict[str, dict] = {}
        for (game, lv), L in sorted(session.ledgers.items()):
            ear = ear_by.setdefault(game, {})
            by.setdefault(game, []).append(evaluate(((r.action, r.kb, r.ka) for r in L.rows), earlier=dict(ear), **flags))
            for act, dd in L.dominant().items():
                if dd[2] >= MIN_LAW and (act not in ear or dd[2] >= ear[act][2]): ear[act] = dd
        out[name] = {g: combine(parts) for g, parts in by.items()}
    return out


def rates(c: dict) -> dict:
    r = F3.rates(c); f = lambda a, b: (a / b) if b else None
    r.update(null_moves=c.get("null_moves", 0), rep_screen=c.get("rep_screen", 0), rep_screen_null=c.get("rep_screen_null", 0), rep_pos=c.get("rep_pos", 0), rep_pos_null=c.get("rep_pos_null", 0),
             rep_pos_null_null=c.get("rep_pos_null_null", 0), rep_share=f(c.get("rep_pos_null_null", 0), c.get("did_not_move", 0)), abstained=c.get("abstained", 0),
             reasons={k: c.get("abs:" + k, 0) for k in REASONS}, reasons_covered={k: c.get("abs_cov:" + k, 0) for k in REASONS},
             once_pred=c.get("once_pred", 0), once_hit=c.get("once_hit", 0), elim_pred=c.get("elim_pred", 0), elim_hit=c.get("elim_hit", 0),
             elim_precision=f(c.get("elim_hit", 0), c.get("elim_pred", 0)), once_precision=f(c.get("once_hit", 0), c.get("once_pred", 0)), hits=c.get("cell_hit", 0))
    return r


def action_steps(c: dict) -> Dict[str, Dict[str, int]]:
    """action -> {step: levels in which it made that step on the main mover}."""
    out: Dict[str, Dict[str, int]] = {}
    for k, v in c.items():
        if k.startswith("step:"): _, a, st = k.split(":", 2); out.setdefault(a, {})[st] = out.setdefault(a, {}).get(st, 0) + v
    return out


def text(variants: Dict[str, Dict[str, dict]]) -> str:
    pct = lambda x: "-" if x is None else f"{100 * x:.1f}%"; lines = []
    full = variants.get("full") or {}
    if full:
        lines.append("== 1 repeats: moves in which the piece did not move, and how many were the same action from the same place that had not moved it before")
        for g, c in list(full.items()) + [("ALL GAMES", combine(full.values()))]:
            r = rates(c); lines.append(f"  {g}: {r['moves']} moves, {r['null_moves']} changed no cell; piece did not move in {r['did_not_move']} of the {r['predicted']} predicted; of those a repeat from the same place: {r['rep_pos_null_null']} ({pct(r['rep_share'])}); "
                                       f"same action from a place seen before {r['rep_pos']} (the piece had not moved there before: {r['rep_pos_null']}); the whole screen repeated {r['rep_screen']} times (earlier outcome no change: {r['rep_screen_null']})")
        lines.append("== 2 the gap: moves not predicted, by reason (and, in brackets, how many an earlier level's law would have answered)")
        for g, c in list(full.items()) + [("ALL GAMES", combine(full.values()))]:
            r = rates(c); lines.append(f"  {g}: {r['abstained']} not predicted: " + ", ".join(f"{k} {r['reasons'][k]} ({r['reasons_covered'][k]})" for k in REASONS))
    for nm in ("once", "elim", "once+elim"):
        pg = variants.get(nm) or {}
        if not pg or not full: continue
        t, tf = rates(combine(pg.values())), rates(combine(full.values()))
        lines.append(f"== 3 {nm}: {t['predicted']} predicted ({t['predicted'] - tf['predicted']:+d} against full), right {t['hits']} ({t['hits'] - tf['hits']:+d}); added by once {t['once_pred']} right {t['once_hit']} ({pct(t['once_precision'])}), by elim {t['elim_pred']} right {t['elim_hit']} ({pct(t['elim_precision'])}); "
                     f"right among predicted {pct(t['cell_learned'])} (full: {pct(tf['cell_learned'])})")
    if full:
        tf = rates(combine(full.values())); lines.append("== 4 the delete test: right moves with one part deleted alone (full: %d of %d)" % (tf["hits"], tf["predicted"]))
        for p in PARTS:
            pg = variants.get("-" + p) or {}
            if pg: t = rates(combine(pg.values())); lines.append(f"  without {p}: right {t['hits']} of {t['predicted']} ({t['hits'] - tf['hits']:+d}); right among predicted {pct(t['cell_learned'])}")
        st = action_steps(combine(full.values())); rel = {k[4:]: v for k, v in combine(full.values()).items() if k.startswith("rel:")}
        lines.append("== the step each action made on the main mover, in how many levels (all arms and games): " + "; ".join(f"{a}: " + ", ".join(f"({s}) in {n}" for s, n in sorted(st[a].items())) for a in sorted(st)))
        lines.append("== pairs of actions that moved the same piece, in levels: " + ", ".join(f"{k} {v}" for k, v in sorted(rel.items())))
    return "\n".join(lines)


# ---------------------------------------------------------------- the reading lines (fixed before the run, docs/FORESEE_CHECK_5.md)
G_LINES = [
    ("guard", "the saved moves of all four arms (K, A, B and C) replay into fresh copies of the games and reproduce every saved flag"),
    ("instrument", "with every part on and both additions off the check gives the third check's full figure on the same runs: 603 right of 729 predicted"),
    ("claim", "of the moves in which the piece did not move, at least 20 percent were the same action tried before from the same place of the piece, which had not moved it then either"),
    ("claim", "the share of those repeats among the moves in which the piece did not move is lower in arm C (the ledger's read-out in the prompt) than in arm K (the kept chat)"),
    ("claim", "with once and elim together at least 30 more moves are predicted than with them off, and at least 70 percent of the moves added are right"),
    ("claim", "elim alone predicts at least 8 moves, and at least 70 percent of them are right"),
    ("claim", "each of the five parts (the whole front held, the action typed, the edge of the screen, the floor colours, the mover as a group), deleted alone, costs at least 5 right moves over the four arms and games"),
    ("claim", "for each of ACTION1 to ACTION4 the unit step it makes on the main mover is the same in at least 90 percent of the levels in which it made one (at least 4 levels each), over the four arms and games"),
]
HEADLINE = "THE LEDGER KNOWS MOVES THAT WERE WASTED, AND CAN ANSWER MOVES IT NOW LEAVES TO THE MODEL"


def _combined(variants, name) -> Optional[dict]:
    pg = (variants or {}).get(name)
    return rates(combine(pg.values())) if pg else None


def verdict(variants: Optional[Dict[str, Dict[str, dict]]], replays_ok: Optional[bool], per_arm: Optional[dict] = None) -> List[Tuple[int, str, str, Optional[bool]]]:
    v: List[Optional[bool]] = [None] * len(G_LINES)
    if replays_ok is not None: v[0] = bool(replays_ok)
    full = _combined(variants, "full")
    if full is not None:
        v[1] = (full["hits"], full["predicted"]) == THIRD_FULL
        v[2] = full["rep_share"] is not None and full["rep_share"] >= 0.20
        both = _combined(variants, "once+elim")
        if both is not None:
            add = both["predicted"] - full["predicted"]; addh = both["hits"] - full["hits"]
            v[4] = add >= 30 and addh >= 0.70 * add
        el = _combined(variants, "elim")
        if el is not None: v[5] = el["elim_pred"] >= 8 and el["elim_precision"] is not None and el["elim_precision"] >= 0.70
        dels = [_combined(variants, "-" + p) for p in PARTS]
        if all(x is not None for x in dels): v[6] = all(full["hits"] - x["hits"] >= 5 for x in dels)
        st = action_steps(combine(variants["full"].values()))
        ok = []
        for a in ACTIONS:
            s = st.get(a, {}); tot = sum(s.values()); ok.append(tot >= 4 and max(s.values()) >= 0.90 * tot)
        v[7] = all(ok)
    if per_arm and isinstance(per_arm.get("K"), dict) and isinstance(per_arm.get("C"), dict) and "full" in per_arm["K"] and "full" in per_arm["C"]:
        k, cc = _combined(per_arm["K"], "full"), _combined(per_arm["C"], "full")
        if k and cc and k["rep_share"] is not None and cc["rep_share"] is not None: v[3] = cc["rep_share"] < k["rep_share"]
    return [(i + 1, t, x, v[i]) for i, (t, x) in enumerate(G_LINES)]


def claims_allowed(checks) -> bool:
    return bool(checks[0][3]) and bool(checks[1][3])


def headline(checks) -> str:
    if not claims_allowed(checks): return "NOT READ: a guard or instrument line is not yes, no figure is claimed until the cause is found"
    r = {i: v for i, _, _, v in checks}
    return HEADLINE if r[3] and r[5] else "MIXED: line 3 " + ("yes" if r[3] else "NO") + ", line 5 " + ("yes" if r[5] else "NO")
