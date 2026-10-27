"""Item 142 (test 167): the seventh forward check on the saved ARC-AGI-3 runs. The sixth check (docs/FORESEE_CHECK_6.md; its printout was read by the owner on 27 Oct 2026) found, on the 936 saved moves of the four games:
  - the NAME of an action held its direction in every level of every game: ACTION1 (0,-1) in 19 of 19 levels, ACTION2 (0,1) in 20 of 20, ACTION3 (-1,0) in 15 of 15, ACTION4 (1,0) in 18 of 18; and the quarter-turns between every pair were the
    same in every level (line 6 and line 7 read yes). So the arrangement of the four actions is one arrangement in these four games. The length of the step is not the same everywhere (3, 4, 5 or 6 cells);
  - the rotation set (`rot`) predicted 2 moves, both wrong; the carried relation (`rel`) predicted 10, 9 right; of 127 moves not predicted, 84 were an action's first try in a level;
  - with the strip that ticks left out, the same action on the same screen repeated 268 times (the whole screen counted 1), and where the earlier outcome had been "no change", 89 of 93 were "no change" again.
This check asks what those findings allow, on the same saved moves, with no model and no cost, and changes neither the ledger nor the fifth and sixth checks' predictor (with every new part off the counters are the sixth check's, counter for counter; a test holds that):
  1. `nm`  a FIRST TRY answered from the NAME: the direction of the action's step is the direction that name had in the OTHER games (a game never reads its own levels for it: leave one game out; only when the other games agree
     in at least 90 percent of at least 4 levels), the length is the length of a step already seen. Two tiers: `nm` (the length from a step this level has already shown) and `nc` (the length of the latest earlier level of the same game).
  2. THE ARRANGEMENT CLASS: the four actions as four turns of one step: up to a turn of the whole board there are 6 arrangements (3 pairings of opposites x 2 handedness). Each level that shows all four is counted into its class.
  3. THE SCREEN MEMORY with the strip found ONLINE: the sixth check found the strip that ticks from the whole level and then counted repeats, which an actor playing the level could not do. Here the strip is found from the moves before, at every move; the same action
     on the same screen (strip left out) is counted, what it did last time is compared with what it does now, and the moves that no other tier predicted are given that earlier outcome.
  4. THE RETIRE QUESTION: the ledger has no step that takes a law away once it has failed. After a law-tier prediction has been wrong by a different step, how right are the later law-tier predictions for that action?
  5. THE EDGE: when the step would leave the board, did the piece stay (a block) and not come in on the other side (a wrap)?
With the new parts off, nothing is predicted from a later move; the name prior and the carried length never read the level being predicted.

Comes from Rajnish Choubey's writing as read in the owner's reading of the Cinderella Project pages (the six of 27 Oct 2026 and the nine before): meaning set by relation and not by word (a name's direction is known by how it stands to the others); the "101st day" (a rule is trusted after it has held, which is `tries`
and `MIN_TRIES`); the Delete Test (the iceberg, the filter of least weight); the enter and the exit of one gate (the opposite action as the undo); the question and the answer at the same instant (the screen memory: what was said and what happened); "where the map tears, the map broke" (where the
ledger has no law it says so). Whose words those pages are is for Sir to confirm; here they are read for structure only. What it does NOT show: that a playing model uses fewer moves with any of this (a paid test), that a path-finder clears a level, that the names are the same in games
other than these four. A pure function of the saved moves."""
from __future__ import annotations
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

from axi.engine.ledger import Ledger, MIN_LAW, key
from axi.experiments.foresee import MIN_TRIES, combine, _actual
from axi.experiments import foresee2 as F2
from axi.experiments import foresee3 as F3
from axi.experiments import foresee5 as F5
from axi.experiments import foresee6 as F6
from axi.experiments.foresee6 import turn, turns_between, axis_steps, level_relations, rotation_set, from_relation, name_counts, _axis, _len, _sgn, ACTIONS, PAIRS

__all__ = ["evaluate", "from_session", "rates", "text", "verdict", "headline", "claims_allowed", "G_LINES", "HEADLINE", "combine", "VARIANTS", "THIRD_FULL", "CHECK5_ONCE", "CHECK6_ALL", "ACTIONS", "SOURCES", "NORTH_CLASSES"]

THIRD_FULL = F6.THIRD_FULL         # (603, 729) the third check's full version on the same runs: right, predicted
CHECK5_ONCE = F6.CHECK5_ONCE       # (663, 797)
CHECK6_ALL = (672, 809)            # the sixth check's `all` (once, rot, rel) on the same runs
TYPE_MIN = F3.TYPE_MIN
SOURCES = ("law", "once", "rot", "rel", "nm", "nc")
PRIOR_MIN_LEVELS, PRIOR_SHARE = 4, 0.90
N, E, S, W = (0, -1), (1, 0), (0, 1), (-1, 0)
NORTH_CLASSES = tuple(sorted({(a, b, c) for a in (S, E, W) for b in (S, E, W) for c in (S, E, W) if len({a, b, c}) == 3}))      # the 6 arrangements once ACTION1 is turned to face north
VARIANTS = {"full": {}, "once": dict(once=True), "all6": dict(once=True, rot=True, rel=True), "nm": dict(nm=True),
            "all7": dict(once=True, rot=True, rel=True, nm=True, scr=True), "best": dict(once=True, rot=True, rel=True, nm=True, scr=True, floor=False, group=False)}


# ---------------------------------------------------------------- the name prior and the length
def loose_steps(L: Ledger) -> Tuple[object, Dict[object, Tuple[Tuple[int, int], int]]]:
    """(the main mover, {action: (step, times)}) like `axis_steps` but from ONE move on: an action that made a step along one axis once is enough to read a length from."""
    dom = L.dominant(); by: Counter = Counter()
    for a in sorted(dom, key=str):
        s, v, n, t = dom[a]
        if _axis(v) and "@" not in str(a): by[s] += n
    if not by: return None, {}
    main = min(by, key=lambda s: (-by[s], str(s)))
    return main, {a: (dom[a][1], dom[a][2]) for a in sorted(dom, key=str) if dom[a][0] == main and _axis(dom[a][1]) and "@" not in str(a)}


def from_name(L: Ledger, action, names: Optional[dict], carry: Optional[tuple]):
    """(tier, (symbol, step, reference action, its step, its count)) for an action never tried: the direction is the name's direction in the other games; the length is the one length the steps already seen in this level
    share (tier `nm`), else the length of the latest earlier level of the game (tier `nc`, with the mover's symbol it had). Nothing if the name has no agreed direction, or the steps seen disagree in length."""
    dv = (names or {}).get(action)
    if dv is None: return None
    main, st = loose_steps(L)
    if st:
        lens = {_len(v) for v, _ in st.values()}
        if len(lens) != 1: return None
        ln = next(iter(lens)); ref = max(st, key=lambda a: (st[a][1], str(a)))
        return "nm", (main, (dv[0] * ln, dv[1] * ln), ref, st[ref][0], st[ref][1])
    if carry is not None:
        sym, ln = carry
        return "nc", (sym, (dv[0] * ln, dv[1] * ln), None, None, 0)
    return None


def arrangement_class(st: Dict[object, Tuple[Tuple[int, int], int]]) -> Optional[tuple]:
    """The class of a level once all four of ACTION1 to ACTION4 made a step of one length: ACTION1 turned to face north, and the directions of ACTION2, ACTION3 and ACTION4 after the same turn (one of 6). None otherwise."""
    if not all(a in st for a in ACTIONS): return None
    v = [st[a][0] for a in ACTIONS]
    if len({_len(x) for x in v}) != 1: return None
    d = [(_sgn(x[0]), _sgn(x[1])) for x in v]; k = turns_between(N, d[0])
    return None if k is None else tuple(turn(x, k) for x in d[1:])


# ---------------------------------------------------------------- the check
def evaluate(moves: Iterable[Tuple[object, object, object]], front: bool = True, typed: bool = True, edge: bool = True, floor: bool = True, group: bool = True,
             once: bool = False, rot: bool = False, rel: bool = False, nm: bool = False, scr: bool = False, min_law: int = MIN_LAW, earlier: Optional[dict] = None,
             earlier_rel: Optional[dict] = None, names: Optional[dict] = None, carry: Optional[tuple] = None) -> dict:
    """`moves`: (action, before, after) of one level in the order played. Each move is predicted from the moves before it (a fresh ledger), then scored. Pure. With nm and scr off it is the sixth check's evaluate counter for counter.
    `names`: {action: (sign dx, sign dy)} read from OTHER games only, used only when `nm`. `carry`: (symbol, length) of the latest earlier level of the same game, used only when `nm`. `scr` switches the screen-memory counters on
    (they never change what is predicted)."""
    L = Ledger(min_law=min_law); c: Counter = Counter(); last: Dict[object, tuple] = {}; rigid: set = set(); passed: set = set(); walked: set = set()
    mv: Counter = Counter(); fl: Counter = Counter(); seen: Dict[tuple, tuple] = {}; posseen: Dict[tuple, bool] = {}; earlier_rel = earlier_rel or {}; failed: set = set()
    covers = lambda action, before: earlier is not None and earlier.get(action) is not None and any(earlier[action][0] in row for row in before)
    for action, before, after in moves:
        a = str(action); click = "@" in a; dom = None if click else L.dominant().get(action); tries = L.tries(action)
        d = dom; used = None; eref = None
        if d is not None and d[3] < MIN_TRIES:
            if once: used = "once"
            else: d = None
        if d is None and not click and tries == 0 and (rel or rot or nm):
            e = from_relation(L, action, earlier_rel) if rel else None
            if e is not None: used = "rel"
            if e is None and nm:
                r = from_name(L, action, names, carry)
                if r is not None: used, e = r
            if e is None and rot:
                e = rotation_set(L)
                if e is not None: used = "rot"
            if e is not None: d = (e[0], e[1], 0, 0); eref = e[2:] if e[2] is not None else None
        kb, ka = key(before), key(after)
        prevs: Dict[str, tuple] = {}
        if scr:                                                                  # the same action on the same screen, with the strip found from the moves BEFORE this one left out
            base = L.tick_cells() if L.rows else frozenset()
            syms = {t["symbol"] for t in L.ticks()} if base else set()
            wide = (base | {(x, y) for y, rw in enumerate(kb) for x, v in enumerate(rw) if v in syms}) if base else base      # and every cell of the ticking symbol on THIS screen (the cell a strip is about to lose has not changed yet)
            for tag, cells in (("scr", base), ("scx", wide)):
                mask = (lambda k, cells=cells: L._mask_cells(k, cells)) if cells else (lambda k: k)
                mem: Dict[tuple, tuple] = {}
                for r in L.rows: mem[(r.action, mask(r.kb))] = (r.kb != r.ka and mask(r.kb) != mask(r.ka), mask(r.ka))
                prevs[tag] = (mask, mem.get((action, mask(kb))))
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
            pred = dict(symbol=sym, step=step, known=known, wall=wall, stationary=stationary, why=why, inside=inside, edge_only=known and edge and bool(outside) and not bool(inside & rigid),
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
            if src == "law":                                                     # the retire question: how right is a law after it has been wrong by a different step?
                c[("postfail_" if a in failed else "prefail_") + "pred"] += 1; c[("postfail_" if a in failed else "prefail_") + "hit"] += int(ok)
                if got is not None and not hit: failed.add(a); c["fail_events"] += 1
            if pred["known"]: c["known"] += 1; c["known_step_hit"] += int(hit); c["known_cell_hit"] += int(ok)
            if said:
                c[pred["why"] + "_said"] += 1; c[pred["why"] + "_right"] += int(got is None); c["cell_said"] += 1; c["cell_right"] += int(got is None)
                if pred["why"] == "wall" and pred["edge_only"]: c["edge_said"] += 1; c["edge_right"] += int(got is None); c["edge_moved"] += int(got is not None)
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
        for tag, (mask, prev) in prevs.items():                                  # the screen memory: what it did last time against what it does now (the strip left out; scr = the cells it has changed, scx = and its symbol on this screen)
            if prev is None: continue
            same = mask(ka) == prev[1]; c[tag + "_n"] += 1; c[tag + "_same"] += int(same)
            if not prev[0]: c[tag + "_null"] += 1; c[tag + "_null_null"] += int(mask(kb) == mask(ka))
            if pred is None: c[tag + "_fill"] += 1; c[tag + "_fill_right"] += int(same)
            else: c[tag + "_over"] += 1; c[tag + "_over_main_right"] += int(ok); c[tag + "_over_same"] += int(same)
        for o in rec.ops:
            if o[0] == "move": last[o[1]] = (o[2][0] + o[3][0], o[2][1] + o[3][1], o[2][2] + o[3][0], o[2][3] + o[3][1])
    main, st = axis_steps(L)                                                  # what each action did, for the names: the direction and the length of its step, and the quarter-turns between actions
    for act, (v, n) in st.items():
        c[f"dir:{act}:{_sgn(v[0])},{_sgn(v[1])}"] += 1; c[f"len:{act}:{_len(v)}"] += 1
        if names is not None: c[f"pri:{act}:n"] += 1; c[f"pri:{act}:ok"] += int((names or {}).get(act) == (_sgn(v[0]), _sgn(v[1])))
    for (x, y), k in level_relations(L).items(): c[f"relk:{x}>{y}:{k}"] += 1
    cl = arrangement_class(st)
    if cl is not None: c["cls:" + "|".join(f"{x},{y}" for x, y in cl)] += 1
    cells = L.tick_cells()                                                    # the same action on the same screen, with the strip that ticks left out (found from the whole level, as in the sixth check)
    if cells:
        c["tick_levels"] += 1; sm: Dict[tuple, bool] = {}
        for r in L.rows:
            km, am = L._mask_cells(r.kb, cells), L._mask_cells(r.ka, cells); kk = (km, r.action)
            if kk in sm:
                c["mrep"] += 1
                if not sm[kk]: c["mrep_null"] += 1; c["mrep_null_null"] += int(km == am)
            sm[kk] = km != am
    return dict(c)


# ---------------------------------------------------------------- the name prior, from the other games only
def name_prior(dirs: Dict[str, Dict[object, Counter]], game: str) -> dict:
    """{action: (sign dx, sign dy)} from the levels of every game EXCEPT `game`: the direction at least 90 percent of at least 4 levels agree on."""
    out = {}; acts = {a for g, d in dirs.items() if g != game for a in d}
    for a in acts:
        tot: Counter = Counter()
        for g, d in dirs.items():
            if g != game and a in d: tot.update(d[a])
        n = sum(tot.values()); dv, k = max(tot.items(), key=lambda kv: (kv[1], kv[0]))
        if n >= PRIOR_MIN_LEVELS and k >= PRIOR_SHARE * n: out[a] = dv
    return out


def from_session(session) -> Dict[str, Dict[str, dict]]:
    """variant -> game -> counters, the six variants on the same replayed levels. The levels of a game are taken in order; `earlier` only tags, `earlier_rel` and `carry` come from the levels before, `names` from the other games."""
    ledgers = sorted(session.ledgers.items()); dirs: Dict[str, Dict[object, Counter]] = {}
    for (game, lv), L in ledgers:
        for act, (v, n) in axis_steps(L)[1].items(): dirs.setdefault(game, {}).setdefault(act, Counter())[(_sgn(v[0]), _sgn(v[1]))] += 1
    prior = {g: name_prior(dirs, g) for g in {g for (g, _), _ in ledgers}}
    out: Dict[str, Dict[str, dict]] = {}
    for name, flags in VARIANTS.items():
        by: Dict[str, List[dict]] = {}; ear_by: Dict[str, dict] = {}; rel_by: Dict[str, dict] = {}; car_by: Dict[str, Optional[tuple]] = {}
        for (game, lv), L in ledgers:
            ear = ear_by.setdefault(game, {}); rl = rel_by.setdefault(game, {})
            by.setdefault(game, []).append(evaluate(((r.action, r.kb, r.ka) for r in L.rows), earlier=dict(ear), earlier_rel=dict(rl), names=prior[game], carry=car_by.get(game), **flags))
            for act, dd in L.dominant().items():
                if dd[2] >= MIN_LAW and (act not in ear or dd[2] >= ear[act][2]): ear[act] = dd
            rl.update(level_relations(L))
            main, st = axis_steps(L)
            if main is not None and st:
                lens = Counter(_len(v) for v, _ in st.values()); car_by[game] = (main, min(lens, key=lambda x: (-lens[x], x)))
        out[name] = {g: combine(parts) for g, parts in by.items()}
    return out


def rates(c: dict) -> dict:
    r = F6.rates(c); f = lambda a, b: (a / b) if b else None
    for s in SOURCES: r[s + "_pred"] = c.get(s + "_pred", 0); r[s + "_hit"] = c.get(s + "_hit", 0); r[s + "_precision"] = f(c.get(s + "_hit", 0), c.get(s + "_pred", 0))
    r.update(name_pred=c.get("nm_pred", 0) + c.get("nc_pred", 0), name_hit=c.get("nm_hit", 0) + c.get("nc_hit", 0))
    r["name_precision"] = f(r["name_hit"], r["name_pred"])
    for tg in ("scr", "scx"):
        r.update({tg + "_" + k: c.get(tg + "_" + k, 0) for k in ("n", "same", "null", "null_null", "fill", "fill_right", "over", "over_main_right", "over_same")})
        r[tg + "_null_share"] = f(r[tg + "_null_null"], r[tg + "_null"]); r[tg + "_fill_share"] = f(r[tg + "_fill_right"], r[tg + "_fill"])
    r.update(prefail_pred=c.get("prefail_pred", 0), prefail_hit=c.get("prefail_hit", 0), postfail_pred=c.get("postfail_pred", 0), postfail_hit=c.get("postfail_hit", 0), fail_events=c.get("fail_events", 0),
             edge_said=c.get("edge_said", 0), edge_right=c.get("edge_right", 0), edge_moved=c.get("edge_moved", 0))
    r["prefail_precision"] = f(r["prefail_hit"], r["prefail_pred"]); r["postfail_precision"] = f(r["postfail_hit"], r["postfail_pred"]); r["edge_share"] = f(r["edge_right"], r["edge_said"])
    return r


def prior_counts(c: dict) -> Dict[str, Tuple[int, int]]:
    """action -> (levels in which it made a step, levels in which the name prior from the other games named that direction)."""
    acts = sorted({k.split(":")[1] for k in c if k.startswith("pri:")}, key=str)
    return {a: (c.get(f"pri:{a}:n", 0), c.get(f"pri:{a}:ok", 0)) for a in acts}


def class_counts(c: dict) -> Dict[str, int]:
    return {k[4:]: v for k, v in sorted(c.items()) if k.startswith("cls:")}


def text(variants: Dict[str, Dict[str, dict]]) -> str:
    pct = lambda x: "-" if x is None else f"{100 * x:.1f}%"; lines = []; full = variants.get("full") or {}
    if full:
        tf = rates(combine(full.values()))
        lines.append("== 1 first tries answered from the NAME (direction from the other games, length from a step seen) and the tiers (against the full version: %d right of %d predicted)" % (tf["hits"], tf["predicted"]))
        for nm in ("once", "nm", "all6", "all7", "best"):
            pg = variants.get(nm) or {}
            if not pg: continue
            t = rates(combine(pg.values()))
            lines.append(f"  {nm}: {t['predicted']} predicted ({t['predicted'] - tf['predicted']:+d}), right {t['hits']} ({t['hits'] - tf['hits']:+d}); right among predicted {pct(t['cell_learned'])}; by name+seen length {t['nm_pred']} right {t['nm_hit']} ({pct(t['nm_precision'])}), "
                         f"by name+earlier level's length {t['nc_pred']} right {t['nc_hit']} ({pct(t['nc_precision'])}), carried relation {t['rel_pred']} right {t['rel_hit']}, rotation set {t['rot_pred']} right {t['rot_hit']}, one earlier try {t['once_pred']} right {t['once_hit']}")
        if variants.get("all7"):
            lines.append("   by game (all7): " + "; ".join(f"{g}: {r['predicted']} predicted, {r['hits']} right, not predicted {r['abstained']} (first tries {r['reasons']['new']})" for g, r in ((g, rates(c)) for g, c in variants["all7"].items())))
        lines.append("== 2 the names and the arrangement (all arms and games)")
        pc = prior_counts(combine(full.values()))
        lines.append("   levels in which the name prior from the OTHER games named the direction the action really made (levels with a step, levels agreeing): " + "; ".join(f"{a}: {n}, {k}" for a, (n, k) in pc.items()))
        cc = class_counts(combine(full.values()))
        lines.append("   arrangement classes (ACTION1 turned north; the directions of ACTION2, ACTION3, ACTION4), levels showing all four: " + (", ".join(f"({k.replace('|', ' ')}) in {v}" for k, v in cc.items()) or "none") + f"; classes seen {len(cc)} of 6")
    if variants.get("all7"):
        t = rates(combine(variants["all7"].values()))
        lines.append("== 3 the screen memory, the strip found from the moves before (all7): scr = the cells the strip has already changed left out; scx = and the cells of its symbol on this screen")
        for tg, nm in (("scr", "scr"), ("scx", "scx")):
            lines.append(f"   {nm}: same action on the same screen {t[tg + '_n']} repeats in {t['moves']} moves; earlier outcome no change: {t[tg + '_null']}, and no change again: {t[tg + '_null_null']} ({pct(t[tg + '_null_share'])}); the whole screen after identical to last time: {t[tg + '_same']} of {t[tg + '_n']}; "
                         f"moves no other tier predicted that it covers: {t[tg + '_fill']}, identical to last time: {t[tg + '_fill_right']} ({pct(t[tg + '_fill_share'])}); moves predicted AND remembered: {t[tg + '_over']}: the tiers right {t[tg + '_over_main_right']}, the memory identical {t[tg + '_over_same']}")
        fc = combine(full.values())
        lines.append(f"   (sixth check, strip found from the whole level: {fc.get('mrep', 0)} repeats; earlier no change {fc.get('mrep_null', 0)}, again {fc.get('mrep_null_null', 0)})")
    if full:
        t = tf
        lines.append("== 4 the retire question (full): law-tier predictions of an action before and after one of its predictions was wrong by a different step")
        lines.append(f"   before any such failure: {t['prefail_pred']} predicted, {t['prefail_hit']} right ({pct(t['prefail_precision'])}); after: {t['postfail_pred']} predicted, {t['postfail_hit']} right ({pct(t['postfail_precision'])}); failures {t['fail_events']}")
        lines.append(f"== 5 the edge (full): predictions that the piece stays because the step leaves the board: {t['edge_said']}; stayed {t['edge_right']} ({pct(t['edge_share'])}), moved {t['edge_moved']}")
    return "\n".join(lines)


# ---------------------------------------------------------------- the reading lines (fixed before the run, docs/FORESEE_CHECK_7.md)
G_LINES = [
    ("guard", "the saved moves of all four arms (K, A, B and C) replay into fresh copies of the games and reproduce every saved flag"),
    ("instrument", "with the new parts off the check gives the sixth check's figures on the same runs: the full version 603 right of 729 predicted, once 663 of 797 and all6 (once, rot and rel) 672 of 809"),
    ("claim", "for each of ACTION1 to ACTION4 the direction the other games give (a game never reads its own levels) is the direction the action made in at least 90 percent of the levels in which it made a step (at least 4 levels each)"),
    ("claim", "the name (nm alone, both lengths) predicts at least 35 moves, and at least 75 percent of them are right"),
    ("claim", "with once, rot, rel and nm together (all7) at least 110 more moves are predicted than in the full version, and at least 80 percent of the moves added are right"),
    ("claim", "every level in which all four actions made a step of one length is of one arrangement class, and there are at least 10 such levels"),
    ("claim", "with the strip found from the moves before and the cells of its symbol on the screen left out, the same action on the same screen repeats at least 150 times in the 936 moves (the sixth check, strip found from the whole level: 268)"),
    ("claim", "of those repeats the ones in which the earlier outcome was no change are no change again in at least 90 percent (at least 30 such repeats)"),
    ("claim", "that screen memory covers at least 10 moves that no other tier predicted, and at least 80 percent of them end on exactly the screen they ended on last time"),
    ("claim", "after a law-tier prediction for an action has been wrong by a different step, the later law-tier predictions for that action are right in at most 70 percent (at least 15 of them)"),
    ("claim", "when the step would leave the board and the piece is predicted to stay, it stays in at least 90 percent (at least 5 such predictions)"),
    ("claim", "the best way (all7 with the floor colours and the mover as a group deleted) has at least 40 more right moves than the sixth check's all (672) and at least 82 percent of its predicted moves right"),
]
HEADLINE = "IN THESE FOUR GAMES A FIRST TRY OF AN ACTION CAN BE ANSWERED FROM ITS NAME AND THE LENGTH OF A STEP ALREADY SEEN"


def _combined(variants, name) -> Optional[dict]:
    pg = (variants or {}).get(name)
    return rates(combine(pg.values())) if pg else None


def verdict(variants: Optional[Dict[str, Dict[str, dict]]], replays_ok: Optional[bool]) -> List[Tuple[int, str, str, Optional[bool]]]:
    v: List[Optional[bool]] = [None] * len(G_LINES)
    if replays_ok is not None: v[0] = bool(replays_ok)
    full = _combined(variants, "full")
    if full is not None:
        on, a6 = _combined(variants, "once"), _combined(variants, "all6")
        v[1] = (full["hits"], full["predicted"]) == THIRD_FULL and on is not None and (on["hits"], on["predicted"]) == CHECK5_ONCE and a6 is not None and (a6["hits"], a6["predicted"]) == CHECK6_ALL
        pc = prior_counts(combine(variants["full"].values()))
        v[2] = all(pc.get(a, (0, 0))[0] >= 4 and pc[a][1] >= 0.90 * pc[a][0] for a in ACTIONS)
        nm = _combined(variants, "nm")
        if nm is not None: v[3] = nm["name_pred"] >= 35 and nm["name_precision"] is not None and nm["name_precision"] >= 0.75
        al = _combined(variants, "all7")
        if al is not None:
            add = al["predicted"] - full["predicted"]; addh = al["hits"] - full["hits"]; v[4] = add >= 110 and addh >= 0.80 * add
            v[6] = al["scx_n"] >= 150
            v[7] = al["scx_null"] >= 30 and al["scx_null_null"] >= 0.90 * al["scx_null"]
            v[8] = al["scx_fill"] >= 10 and al["scx_fill_right"] >= 0.80 * al["scx_fill"]
        cc = class_counts(combine(variants["full"].values())); v[5] = len(cc) == 1 and sum(cc.values()) >= 10
        v[9] = full["postfail_pred"] >= 15 and full["postfail_hit"] <= 0.70 * full["postfail_pred"]
        v[10] = full["edge_said"] >= 5 and full["edge_right"] >= 0.90 * full["edge_said"]
        b = _combined(variants, "best")
        if b is not None: v[11] = b["hits"] - CHECK6_ALL[0] >= 40 and b["predicted"] > 0 and b["hits"] >= 0.82 * b["predicted"]
    return [(i + 1, t, x, v[i]) for i, (t, x) in enumerate(G_LINES)]


def claims_allowed(checks) -> bool:
    return bool(checks[0][3]) and bool(checks[1][3])


def headline(checks) -> str:
    if not claims_allowed(checks): return "NOT READ: a guard or instrument line is not yes, no figure is claimed until the cause is found"
    r = {i: v for i, _, _, v in checks}
    return HEADLINE if r[4] and r[5] else "MIXED: line 4 " + ("yes" if r[4] else "NO") + ", line 5 " + ("yes" if r[5] else "NO")
