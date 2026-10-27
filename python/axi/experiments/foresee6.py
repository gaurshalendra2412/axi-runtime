"""Item 141 (test 166): the sixth forward check on the saved ARC-AGI-3 runs. The fifth check (docs/FORESEE_CHECK_5.md; numbers in docs/results/foresee_check_5_printout.txt) found 88 of the 183 moves in which the piece did not move
to be the same action from the same place that had already done nothing, that `once` (predict from one earlier try that moved the piece) adds 68 predicted moves of which 60 are right, and that 96 of the 207 moves still not predicted are an
action's FIRST try in a level. Its `elim` (the one unit step no action has made) predicted nothing, and its line on the names of the actions stayed empty: the check only knew steps of ONE cell, and the real games move 5, 6, 3 and 4 cells.
That was a blind spot of the check, not a finding about the games. This check removes it and asks the questions that follow, on the same saved moves, with no model and no cost:
  1. A FIRST TRY, answered from the shape of the actions already seen, not from their names. Two ways, each switched on alone and together, both read from the ledger as it is:
       `rot`  the four actions of a mover make the four turns of one step of ANY length ((0,-L), (0,L), (-L,0), (L,0)); when three of them have made three different steps of one length and the fourth action is tried for
              the first time, it is predicted to make the one turn that is missing (the any-length version of the fifth check's `elim`; a press is the case in which it is wrong);
       `rel`  the RELATION between two actions carried from the earlier levels of the same game ("this action's step was the other's turned by a quarter / by a half / not turned"), applied to the actions of this level: when an action is
              tried for the first time and an action of this level has a known step, the first is predicted to make the other's step turned by the relation seen before. If two known actions disagree, nothing is predicted.
     The earlier levels give the RELATION, never a step, never a name's meaning: a carried step (the fourth check) answers 17 of these first tries; a carried relation is a different thing.
  2. THE THREE TIERS: every prediction is tagged with where it came from (this level's own law of at least two tries, one earlier try, the rotation set, the carried relation), so the precision of each tier is read separately:
     a claim, a suggestion, a first-try guess.
  3. ACTION NAMES: for ACTION1 to ACTION4, the direction of the step in every level (is "ACTION1" the same direction in every game?) and, for every pair, the quarter-turns from one to the other (is the arrangement the same even when
     the directions are not?).
  4. THE DELETE TEST, again: the fifth check found that deleting the floor colours gained 15 right moves and deleting the mover as a group gained 5. Here: each alone and both together.
  5. THE SCREEN WITHOUT ITS CLOCK: the fifth check counted 1 whole-screen repeat in 936 moves because a strip along the edge changes in nearly every move ("the number is always a step behind"). The ledger already finds that strip
     (`Ledger.tick_cells`, row C). Here the same action on the same screen is counted again with only those cells left out.
With every new part off the counters are the fifth check's `full` version, counter for counter (a test holds that). Nothing is predicted from a later move, and `earlier_rel` is built from earlier levels only.

Comes from Rajnish Choubey's writing as read in the owner's reading of the nine Cinderella Project pages of 27 Oct 2026 (the Spherical Mind letter: one radius from "minimal, for your own" to "exhaustive, for show"; "every theory fails when it
claims, every theory is great if it suggests"; the question and the answer born at the same instant; the 24-hour clock of 300,000 years; "a name defines scope"; the four things a system loses: who speaks, why, when silence is right, whether it
is inside or outside; "A Family That Shares a Phone": nothing deleted because nothing hidden) and from the fifth check's own numbers. Whose words those pages are (bylines of Saraswati Choubey, Radha Gupta, Rahul Chauhan and FUTURE OF AI,
and text signed by an AI) is for Sir to confirm; here they are read for structure only. What it does NOT show: that the tiers, a veto of repeats or a first-try guess make a playing model use fewer moves (that is a paid test), that a
path-finder clears a level, or what the goal of a level is. A pure function of the saved moves."""
from __future__ import annotations
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

from axi.engine.ledger import Ledger, MIN_LAW, key
from axi.experiments.foresee import MIN_TRIES, combine, _actual
from axi.experiments import foresee2 as F2
from axi.experiments import foresee3 as F3
from axi.experiments import foresee5 as F5

__all__ = ["evaluate", "from_session", "rates", "text", "verdict", "headline", "claims_allowed", "G_LINES", "HEADLINE", "combine", "VARIANTS", "THIRD_FULL", "CHECK5_ONCE", "CHECK5_NOFLOOR", "ACTIONS", "SOURCES"]

TYPE_MIN = F3.TYPE_MIN
THIRD_FULL = (603, 729)            # the third check's full version on the same runs: right, predicted
CHECK5_ONCE = (663, 797)           # the fifth check's `once` on the same runs
CHECK5_NOFLOOR = (618, 729)        # the fifth check's `-floor` on the same runs
ACTIONS = ("ACTION1", "ACTION2", "ACTION3", "ACTION4")
SOURCES = ("law", "once", "rot", "rel")
VARIANTS = {"full": {}, "-floor": dict(floor=False), "-group": dict(group=False), "-floor-group": dict(floor=False, group=False),
            "once": dict(once=True), "rot": dict(rot=True), "rel": dict(rel=True), "rot+rel": dict(rot=True, rel=True), "all": dict(once=True, rot=True, rel=True)}


# ---------------------------------------------------------------- steps of any length, and the quarter-turns between them
def _axis(v) -> bool: return (v[0] == 0) != (v[1] == 0)                      # a step along one axis only
def _len(v) -> int: return abs(v[0]) + abs(v[1])
def _sgn(x) -> int: return (x > 0) - (x < 0)


def turn(v: Tuple[int, int], k: int) -> Tuple[int, int]:
    """The step `v` turned by `k` quarter-turns ((dx, dy) -> (-dy, dx), the same way every time)."""
    for _ in range(k % 4): v = (-v[1], v[0])
    return v


def turns_between(va, vb) -> Optional[int]:
    """k with turn(vb, k) == va, or None when the two are not the same step turned by quarters."""
    for k in range(4):
        if turn(vb, k) == va: return k
    return None


def axis_steps(L: Ledger) -> Tuple[object, Dict[object, Tuple[Tuple[int, int], int]]]:
    """(the main mover, {action: (step, times)}): the symbol with most moves by a step along one axis (any length) in all, and for it every action (not a click) that made such a step at least MIN_TRIES times."""
    dom = L.dominant(); by: Counter = Counter()
    for a in sorted(dom, key=str):
        s, v, n, t = dom[a]
        if _axis(v) and "@" not in str(a): by[s] += n
    if not by: return None, {}
    main = min(by, key=lambda s: (-by[s], str(s)))
    return main, {a: (dom[a][1], dom[a][2]) for a in sorted(dom, key=str) if dom[a][0] == main and _axis(dom[a][1]) and dom[a][2] >= MIN_TRIES and "@" not in str(a)}


def level_relations(L: Ledger) -> Dict[tuple, int]:
    """{(action a, action b): k} for the actions of the main mover whose steps have the same length: a's step is b's step turned by k quarters. Read at the end of a level, to be carried to the later levels."""
    main, st = axis_steps(L); out: Dict[tuple, int] = {}
    for a in st:
        for b in st:
            if a == b or _len(st[a][0]) != _len(st[b][0]): continue
            k = turns_between(st[a][0], st[b][0])
            if k is not None: out[(a, b)] = k
    return out


def rotation_set(L: Ledger):
    """(symbol, the missing step, reference action, its step, its count) when exactly three actions of the main mover have made three different steps of one length and the fourth turn of it is the one missing, else None."""
    main, st = axis_steps(L)
    if main is None or len(st) < 3: return None
    lens = Counter(_len(v) for v, _ in st.values()); ln = min(lens, key=lambda x: (-lens[x], x))
    known = {a: (v, n) for a, (v, n) in st.items() if _len(v) == ln}; vals = {v for v, _ in known.values()}
    full = {(0, -ln), (0, ln), (-ln, 0), (ln, 0)}
    if len(known) != 3 or len(vals) != 3 or not vals <= full: return None
    miss = next(iter(full - vals)); ref = max(known, key=lambda a: (known[a][1], str(a)))
    return (main, miss, ref, known[ref][0], known[ref][1])


def from_relation(L: Ledger, action, rel: Dict[tuple, int]):
    """(symbol, step, reference action, its step, its count) for an action never tried: every action of this level with a known step that has a carried relation to `action` gives the step turned by that relation; if they all agree, that step."""
    main, st = axis_steps(L); cands = {}
    for b, (v, n) in st.items():
        k = rel.get((action, b))
        if k is not None: cands[b] = turn(v, k)
    if not cands or len(set(cands.values())) != 1: return None
    ref = max(cands, key=lambda b: (st[b][1], str(b)))
    return (main, next(iter(cands.values())), ref, st[ref][0], st[ref][1])


# ---------------------------------------------------------------- the check
def evaluate(moves: Iterable[Tuple[object, object, object]], front: bool = True, typed: bool = True, edge: bool = True, floor: bool = True, group: bool = True,
             once: bool = False, rot: bool = False, rel: bool = False, min_law: int = MIN_LAW, earlier: Optional[dict] = None, earlier_rel: Optional[dict] = None) -> dict:
    """`moves`: (action, before, after) of one level in the order played. Each move is predicted from the moves before it (a fresh ledger), then scored. Pure. With once, rot, rel off it is the fifth check's `full`
    (foresee5.evaluate(moves)) counter for counter. `earlier`: the laws of earlier levels of the same game, read only to tag. `earlier_rel`: {(action, action): quarter-turns} of earlier levels of the same game, used only when `rel`."""
    L = Ledger(min_law=min_law); c: Counter = Counter(); last: Dict[object, tuple] = {}; rigid: set = set(); passed: set = set(); walked: set = set()
    mv: Counter = Counter(); fl: Counter = Counter(); seen: Dict[tuple, tuple] = {}; posseen: Dict[tuple, bool] = {}; earlier_rel = earlier_rel or {}
    covers = lambda action, before: earlier is not None and earlier.get(action) is not None and any(earlier[action][0] in row for row in before)
    for action, before, after in moves:
        a = str(action); click = "@" in a; dom = None if click else L.dominant().get(action); tries = L.tries(action)
        d = dom; used = None; eref = None
        if d is not None and d[3] < MIN_TRIES:
            if once: used = "once"
            else: d = None
        if d is None and not click and tries == 0 and (rel or rot):
            e = from_relation(L, action, earlier_rel) if rel else None
            if e is not None: used = "rel"
            elif rot:
                e = rotation_set(L)
                if e is not None: used = "rot"
            if e is not None: d = (e[0], e[1], 0, 0); eref = e[2:]
        kb, ka = key(before), key(after)
        if (action, kb) in seen:
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
            src = used or "law"; c[src + "_pred"] += 1; c[src + "_hit"] += int(ok)
            if used: c[used + "_covered"] += int(covers(action, before))
            if pred["known"]: c["known"] += 1; c["known_step_hit"] += int(hit); c["known_cell_hit"] += int(ok)
            if said: c[pred["why"] + "_said"] += 1; c[pred["why"] + "_right"] += int(got is None); c["cell_said"] += 1; c["cell_right"] += int(got is None)
            if got is None:
                c["did_not_move"] += 1; c["act_dnm:" + a] += 1
                if pred["known"]:
                    k = "dnm_unknown" if pred["unknown"] else ("dnm_object" if pred["object"] else "dnm_floor"); c[k] += 1
                    if k == "dnm_floor": fl[action] += 1; c["dnm_floor_stat"] += int(pred["stationary"])
            elif not hit: c["other_step"] += 1
            if hit: mv[action] += 1
            if pk is not None:
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
    main, st = axis_steps(L)                                                  # what each action did, for the names: the direction and the length of its step, and the quarter-turns between actions
    for act, (v, n) in st.items(): c[f"dir:{act}:{_sgn(v[0])},{_sgn(v[1])}"] += 1; c[f"len:{act}:{_len(v)}"] += 1
    for (x, y), k in level_relations(L).items(): c[f"relk:{x}>{y}:{k}"] += 1
    cells = L.tick_cells()                                                    # the same action on the same screen, with the strip that ticks left out
    if cells:
        c["tick_levels"] += 1; sm: Dict[tuple, bool] = {}
        for r in L.rows:
            km, am = L._mask_cells(r.kb, cells), L._mask_cells(r.ka, cells); kk = (km, r.action)
            if kk in sm:
                c["mrep"] += 1
                if not sm[kk]: c["mrep_null"] += 1; c["mrep_null_null"] += int(km == am)
            sm[kk] = km != am
    return dict(c)


def from_session(session) -> Dict[str, Dict[str, dict]]:
    """variant -> game -> counters, the nine variants on the same replayed levels. The levels of a game are taken in order; `earlier` only tags, `earlier_rel` is the relations of the levels before."""
    out: Dict[str, Dict[str, dict]] = {}
    for name, flags in VARIANTS.items():
        by: Dict[str, List[dict]] = {}; ear_by: Dict[str, dict] = {}; rel_by: Dict[str, dict] = {}
        for (game, lv), L in sorted(session.ledgers.items()):
            ear = ear_by.setdefault(game, {}); rl = rel_by.setdefault(game, {})
            by.setdefault(game, []).append(evaluate(((r.action, r.kb, r.ka) for r in L.rows), earlier=dict(ear), earlier_rel=dict(rl), **flags))
            for act, dd in L.dominant().items():
                if dd[2] >= MIN_LAW and (act not in ear or dd[2] >= ear[act][2]): ear[act] = dd
            rl.update(level_relations(L))
        out[name] = {g: combine(parts) for g, parts in by.items()}
    return out


def rates(c: dict) -> dict:
    r = F3.rates(c); f = lambda a, b: (a / b) if b else None
    r.update(null_moves=c.get("null_moves", 0), rep_screen=c.get("rep_screen", 0), rep_pos=c.get("rep_pos", 0), rep_pos_null_null=c.get("rep_pos_null_null", 0), rep_share=f(c.get("rep_pos_null_null", 0), c.get("did_not_move", 0)),
             abstained=c.get("abstained", 0), reasons={k: c.get("abs:" + k, 0) for k in F5.REASONS}, hits=c.get("cell_hit", 0), mrep=c.get("mrep", 0), mrep_null=c.get("mrep_null", 0), mrep_null_null=c.get("mrep_null_null", 0),
             tick_levels=c.get("tick_levels", 0))
    for s in SOURCES: r[s + "_pred"] = c.get(s + "_pred", 0); r[s + "_hit"] = c.get(s + "_hit", 0); r[s + "_precision"] = f(c.get(s + "_hit", 0), c.get(s + "_pred", 0))
    return r


def name_counts(c: dict) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[int, int]], Dict[str, Dict[int, int]]]:
    """(direction: action -> {'dx,dy' sign: levels}, length: action -> {length: levels}, quarter-turns: 'a>b' -> {k: levels})."""
    dr: Dict[str, Dict[str, int]] = {}; ln: Dict[str, Dict[int, int]] = {}; rl: Dict[str, Dict[int, int]] = {}
    for k, v in c.items():
        if k.startswith("dir:"): _, a, s = k.split(":", 2); dr.setdefault(a, {})[s] = dr.setdefault(a, {}).get(s, 0) + v
        elif k.startswith("len:"): _, a, s = k.split(":", 2); ln.setdefault(a, {})[int(s)] = ln.setdefault(a, {}).get(int(s), 0) + v
        elif k.startswith("relk:"): _, p, s = k.split(":", 2); rl.setdefault(p, {})[int(s)] = rl.setdefault(p, {}).get(int(s), 0) + v
    return dr, ln, rl


def _consistent(d: Dict[object, int], need: int = 4, share: float = 0.90) -> bool:
    tot = sum(d.values()); return tot >= need and max(d.values()) >= share * tot


PAIRS = tuple(f"{a}>{b}" for i, a in enumerate(ACTIONS) for b in ACTIONS[i + 1:])


def text(variants: Dict[str, Dict[str, dict]]) -> str:
    pct = lambda x: "-" if x is None else f"{100 * x:.1f}%"; lines = []; full = variants.get("full") or {}
    if full:
        tf = rates(combine(full.values()))
        lines.append("== 1 first tries answered from the shape of the actions already seen (against the full version: %d right of %d predicted)" % (tf["hits"], tf["predicted"]))
        for nm in ("once", "rot", "rel", "rot+rel", "all"):
            pg = variants.get(nm) or {}
            if not pg: continue
            t = rates(combine(pg.values()))
            lines.append(f"  {nm}: {t['predicted']} predicted ({t['predicted'] - tf['predicted']:+d}), right {t['hits']} ({t['hits'] - tf['hits']:+d}); right among predicted {pct(t['cell_learned'])}; by rotation set {t['rot_pred']} right {t['rot_hit']} ({pct(t['rot_precision'])}), "
                         f"by carried relation {t['rel_pred']} right {t['rel_hit']} ({pct(t['rel_precision'])}), by one earlier try {t['once_pred']} right {t['once_hit']} ({pct(t['once_precision'])})")
    if variants.get("all"):
        a = variants["all"]; lines.append("== 2 the tiers in the `all` way: where each prediction came from (predicted, right, right share)")
        for g, c in list(a.items()) + [("ALL GAMES", combine(a.values()))]:
            r = rates(c); lines.append(f"  {g}: " + "; ".join(f"{s} {r[s + '_pred']}, {r[s + '_hit']}, {pct(r[s + '_precision'])}" for s in SOURCES) + f"; not predicted {r['abstained']}: " + ", ".join(f"{k} {r['reasons'][k]}" for k in F5.REASONS))
    if full:
        lines.append("== 3 the delete test again: right moves with the floor colours or the mover as a group deleted (full: %d of %d)" % (tf["hits"], tf["predicted"]))
        for p in ("-floor", "-group", "-floor-group"):
            pg = variants.get(p) or {}
            if pg: t = rates(combine(pg.values())); lines.append(f"  {p}: right {t['hits']} of {t['predicted']} ({t['hits'] - tf['hits']:+d}); right among predicted {pct(t['cell_learned'])}")
        dr, ln, rl = name_counts(combine(full.values()))
        lines.append("== 4 the names: the direction of each action's step on the main mover, in how many levels (all arms and games): " + "; ".join(f"{a}: " + ", ".join(f"({s}) in {n}" for s, n in sorted(dr[a].items())) for a in sorted(dr)))
        lines.append("   the length of the step: " + "; ".join(f"{a}: " + ", ".join(f"{s} cells in {n}" for s, n in sorted(ln[a].items())) for a in sorted(ln)))
        lines.append("   quarter-turns from the first action of a pair to the second (0 = the same step, 2 = opposite), levels: " + "; ".join(f"{p}: " + ", ".join(f"{k} in {n}" for k, n in sorted(rl[p].items())) for p in sorted(rl)))
        lines.append("== 5 the screen without its clock: the same action on the same screen, with the cells of the ticking strip left out")
        for g, c in list(full.items()) + [("ALL GAMES", combine(full.values()))]:
            r = rates(c); lines.append(f"  {g}: {r['moves']} moves, strip found in {r['tick_levels']} levels; whole-screen repeats counting the strip {r['rep_screen']}, without it {r['mrep']} (earlier outcome no change: {r['mrep_null']}, and no change again: {r['mrep_null_null']})")
    return "\n".join(lines)


# ---------------------------------------------------------------- the reading lines (fixed before the run, docs/FORESEE_CHECK_6.md)
G_LINES = [
    ("guard", "the saved moves of all four arms (K, A, B and C) replay into fresh copies of the games and reproduce every saved flag"),
    ("instrument", "with the new parts off the check gives the third check's figure (603 right of 729 predicted), the fifth check's once (663 right of 797) and the fifth check's deletion of the floor colours (618 right of 729) on the same runs"),
    ("claim", "the rotation set alone (rot) predicts at least 8 moves, and at least 70 percent of them are right"),
    ("claim", "the carried relation alone (rel) predicts at least 10 moves, and at least 70 percent of them are right"),
    ("claim", "with once, rot and rel together (all) at least 90 more moves are predicted than in the full version, and at least 80 percent of the moves added are right"),
    ("claim", "for each of ACTION1 to ACTION4 the direction of its step on the main mover is the same in at least 90 percent of the levels in which it made one (at least 4 levels each), over the four arms and games"),
    ("claim", "for each of the six pairs of ACTION1 to ACTION4 the quarter-turns from the first to the second are the same in at least 90 percent of the levels in which both made a step of the same length (at least 4 levels each)"),
    ("claim", "deleting the floor colours and the mover as a group together gains at least 10 right moves over the full version, over the four arms and games"),
    ("claim", "with the cells of the ticking strip left out, the same action on the same screen repeats at least 30 times in the 936 moves (the fifth check, with the strip: 1)"),
]
HEADLINE = "A FIRST TRY OF AN ACTION CAN BE ANSWERED FROM THE SHAPE OF THE ACTIONS ALREADY SEEN, NOT FROM THEIR NAMES"


def _combined(variants, name) -> Optional[dict]:
    pg = (variants or {}).get(name)
    return rates(combine(pg.values())) if pg else None


def verdict(variants: Optional[Dict[str, Dict[str, dict]]], replays_ok: Optional[bool]) -> List[Tuple[int, str, str, Optional[bool]]]:
    v: List[Optional[bool]] = [None] * len(G_LINES)
    if replays_ok is not None: v[0] = bool(replays_ok)
    full = _combined(variants, "full")
    if full is not None:
        on, nf = _combined(variants, "once"), _combined(variants, "-floor")
        v[1] = (full["hits"], full["predicted"]) == THIRD_FULL and on is not None and (on["hits"], on["predicted"]) == CHECK5_ONCE and nf is not None and (nf["hits"], nf["predicted"]) == CHECK5_NOFLOOR
        r = _combined(variants, "rot")
        if r is not None: v[2] = r["rot_pred"] >= 8 and r["rot_precision"] is not None and r["rot_precision"] >= 0.70
        r = _combined(variants, "rel")
        if r is not None: v[3] = r["rel_pred"] >= 10 and r["rel_precision"] is not None and r["rel_precision"] >= 0.70
        al = _combined(variants, "all")
        if al is not None:
            add = al["predicted"] - full["predicted"]; addh = al["hits"] - full["hits"]; v[4] = add >= 90 and addh >= 0.80 * add
        dr, ln, rl = name_counts(combine(variants["full"].values()))
        v[5] = all(_consistent(dr.get(a, {})) for a in ACTIONS)
        v[6] = all(_consistent(rl.get(p, {})) for p in PAIRS)
        both = _combined(variants, "-floor-group")
        if both is not None: v[7] = both["hits"] - full["hits"] >= 10
        v[8] = full["mrep"] >= 30
    return [(i + 1, t, x, v[i]) for i, (t, x) in enumerate(G_LINES)]


def claims_allowed(checks) -> bool:
    return bool(checks[0][3]) and bool(checks[1][3])


def headline(checks) -> str:
    if not claims_allowed(checks): return "NOT READ: a guard or instrument line is not yes, no figure is claimed until the cause is found"
    r = {i: v for i, _, _, v in checks}
    return HEADLINE if r[4] and r[5] else "MIXED: line 4 " + ("yes" if r[4] else "NO") + ", line 5 " + ("yes" if r[5] else "NO")
