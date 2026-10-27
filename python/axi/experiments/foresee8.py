"""Item 143 (test 168): the eighth forward check on the saved ARC-AGI-3 runs. The seventh check (docs/FORESEE_CHECK_7.md; numbers in docs/results/foresee_check_7_printout.txt) read MIXED: line 4 NO, line 5 NO, and showed three things this check follows up, on the same saved moves, with no model and no cost:
  - Its name prior (the direction an action's name had in the OTHER games) was built inside ONE arm at a time, from the levels of that arm's three other games, and was absent in 40 of the 72 levels in which ACTION1 to ACTION4 made a step (it needs at least 4 agreeing
    levels from the other games of the same arm), and right in the other 32 (the sixth check had found each name's direction the same in every level, so a prior that exists cannot be wrong). Line 3 (levels in which the prior equals the direction made) therefore measured how often the prior existed, not how right it was. That was a flaw of the method, not a finding about the games.
  - After a law-tier prediction had been wrong by a different step, the later law-tier predictions for that action were right in 56 of 92 (60.9 %) against 547 of 637 (85.9 %) before it (24 failures). The ledger has no step that takes a law away; the seventh check only counted.
  - Six levels were cleared in the saved runs (K 2, A 1, B 1, C 2); nothing in the checks looks at them.
This check asks:
  1. THE NAME PRIOR POOLED: the same leave-one-game-out prior (at least 90 percent of at least 4 levels agree), but taken from the levels of ALL FOUR arms of the other three games. A game still never reads its own levels for it. Is the prior now present, and is it right?
  2. THE RETIRE RULE: a law that has been wrong by a different step `retire` times is not used again in that level (its move is left to the model, counted as not predicted). `retire` = 1 and 2. What does it cost in right moves, what does it avoid in wrong ones, and what would a wrong move have to cost,
     compared with a right one, for the rule to pay (rights lost divided by wrongs avoided)? No cost is assumed here; the ratio is reported and the cost-of-being-wrong page is where a number would come from.
  3. THE SIX CLEARS: for each level the saved runs cleared, and for every level of 20 or more moves they did not, what the screen shows just before the end against the level's first screen: does a symbol that is not a ticking strip have a different number of cells, and was the clearing move made with an action that already had a law in that level? A description for the goal question (row 177), no more.
With `retire` 0 and the priors taken per arm the counters are the seventh check's, counter for counter (a test holds that). Nothing is predicted from a later move.

Comes from Rajnish Choubey's writing as read in the owner's reading of the Cinderella Project pages (the eight of 27 Oct 2026 after the seventh check were read for structure and gave nothing to build; two things recur and are asked here): "not this, not this" (a rule is known by what has been taken away: the retire question), and "the maths broke, not the universe" /
"valid only inside its own set-up; do not extrapolate past where the set-up changed" (a law that has failed marks the edge of its own domain). Whose words those pages are is for Sir to confirm; they are read for structure only. What it does NOT show: that a playing model uses fewer moves with any of this (a paid test), that a path-finder clears a level, that the priors hold in other games. A pure function of the saved moves."""
from __future__ import annotations
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

from axi.engine.ledger import Ledger, MIN_LAW, key
from axi.experiments.foresee import MIN_TRIES, combine, _actual
from axi.experiments import foresee2 as F2
from axi.experiments import foresee3 as F3
from axi.experiments import foresee7 as F7
from axi.experiments import ledger_replay as LR
from axi.experiments import gemini_arc as G
from axi.experiments import gemini_arc_b as B
from axi.experiments import gemini_arc_c as C
from axi.experiments.foresee6 import axis_steps, level_relations, rotation_set, from_relation, _axis, _len, _sgn, ACTIONS
from axi.experiments.foresee7 import from_name, name_prior, prior_counts, class_counts, arrangement_class, TYPE_MIN, PRIOR_MIN_LEVELS, PRIOR_SHARE

__all__ = ["evaluate", "from_sessions", "pool", "rates", "text", "verdict", "headline", "claims_allowed", "G_LINES", "HEADLINE", "combine", "VARIANTS", "CHECK7", "ACTIONS", "Session8", "feed8", "clear_study", "level_facts", "retire_cost"]

CHECK7 = {"full": (603, 729), "once": (663, 797), "all6": (672, 809), "all7": (690, 832), "best": (703, 832)}      # the seventh check's figures on the same runs: right, predicted
CLEARS_SEEN = 6                    # K 2, A 1, B 1, C 2
VARIANTS = {"full": {}, "once": dict(once=True), "all6": dict(once=True, rot=True, rel=True), "nm": dict(nm=True),
            "all7": dict(once=True, rot=True, rel=True, nm=True, scr=True), "best": dict(once=True, rot=True, rel=True, nm=True, scr=True, floor=False, group=False),
            "all7_r1": dict(once=True, rot=True, rel=True, nm=True, scr=True, retire=1), "all7_r2": dict(once=True, rot=True, rel=True, nm=True, scr=True, retire=2),
            "best_r1": dict(once=True, rot=True, rel=True, nm=True, scr=True, floor=False, group=False, retire=1)}


# ---------------------------------------------------------------- the check (the seventh check's evaluate, plus the retire rule and the prior-present counters)
def evaluate(moves: Iterable[Tuple[object, object, object]], front: bool = True, typed: bool = True, edge: bool = True, floor: bool = True, group: bool = True,
             once: bool = False, rot: bool = False, rel: bool = False, nm: bool = False, scr: bool = False, min_law: int = MIN_LAW, earlier: Optional[dict] = None,
             earlier_rel: Optional[dict] = None, names: Optional[dict] = None, carry: Optional[tuple] = None, retire: int = 0) -> dict:
    """`moves`: (action, before, after) of one level in the order played. Each move is predicted from the moves before it (a fresh ledger), then scored. Pure. With `retire` 0 it is the seventh check's evaluate counter for counter (plus the prior-present counters).
    `names`: {action: (sign dx, sign dy)} read from OTHER games only, used only when `nm`. `carry`: (symbol, length) of the latest earlier level of the same game, used only when `nm`. `scr` switches the screen-memory counters on
    (they never change what is predicted)."""
    L = Ledger(min_law=min_law); c: Counter = Counter(); last: Dict[object, tuple] = {}; rigid: set = set(); passed: set = set(); walked: set = set()
    mv: Counter = Counter(); fl: Counter = Counter(); seen: Dict[tuple, tuple] = {}; posseen: Dict[tuple, bool] = {}; earlier_rel = earlier_rel or {}; failed: set = set(); fails: Counter = Counter()
    covers = lambda action, before: earlier is not None and earlier.get(action) is not None and any(earlier[action][0] in row for row in before)
    for action, before, after in moves:
        a = str(action); click = "@" in a; dom = None if click else L.dominant().get(action); tries = L.tries(action)
        d = dom; used = None; eref = None
        if d is not None and d[3] < MIN_TRIES:
            if once: used = "once"
            else: d = None
        retired_now = bool(retire) and used is None and d is not None and fails[a] >= retire     # the retire question as a rule: a law that has been wrong by a different step `retire` times is not used again in this level
        if retired_now: d = None; c["retired_withheld"] += 1
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
            r = "retired" if retired_now else ("click" if click else ("new" if tries == 0 else ("once_nomove" if tries == 1 else "nolaw")) if dom is None else "once_moved")
            c["abstained"] += 1; c["abs:" + r] += 1; c["abs_cov:" + r] += int(covers(action, before))
        else:
            got = _actual(rec, pred["symbol"]); c["predicted"] += 1; hit = got == pred["step"]; said = pred["why"] is not None
            ok = (got is None) if said else hit
            c["step_hit"] += int(hit); c["cell_hit"] += int(ok); c["act_pred:" + a] += 1
            src = used or "law"; c[src + "_pred"] += 1; c[src + "_hit"] += int(ok)
            if used: c[used + "_covered"] += int(covers(action, before))
            if src == "law":                                                     # the retire question: how right is a law after it has been wrong by a different step?
                c[("postfail_" if a in failed else "prefail_") + "pred"] += 1; c[("postfail_" if a in failed else "prefail_") + "hit"] += int(ok)
                if got is not None and not hit: failed.add(a); fails[a] += 1; c["fail_events"] += 1
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
        if names is not None:
            c[f"pri:{act}:n"] += 1; c[f"pri:{act}:ok"] += int((names or {}).get(act) == (_sgn(v[0]), _sgn(v[1])))
            c[f"pri:{act}:has"] += int((names or {}).get(act) is not None); c[f"pri:{act}:has_ok"] += int((names or {}).get(act) is not None and (names or {}).get(act) == (_sgn(v[0]), _sgn(v[1])))
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


# ---------------------------------------------------------------- the pooled prior and the runs
def _dirs(session) -> Dict[str, Dict[object, Counter]]:
    out: Dict[str, Dict[object, Counter]] = {}
    for (game, lv), L in sorted(session.ledgers.items()):
        for act, (v, n) in axis_steps(L)[1].items(): out.setdefault(game, {}).setdefault(act, Counter())[(_sgn(v[0]), _sgn(v[1]))] += 1
    return out


def _run(session, prior: Dict[str, dict]) -> Dict[str, Dict[str, dict]]:
    ledgers = sorted(session.ledgers.items()); out: Dict[str, Dict[str, dict]] = {}
    for name, flags in VARIANTS.items():
        by: Dict[str, List[dict]] = {}; ear_by: Dict[str, dict] = {}; rel_by: Dict[str, dict] = {}; car_by: Dict[str, Optional[tuple]] = {}
        for (game, lv), L in ledgers:
            ear = ear_by.setdefault(game, {}); rl = rel_by.setdefault(game, {})
            by.setdefault(game, []).append(evaluate(((r.action, r.kb, r.ka) for r in L.rows), earlier=dict(ear), earlier_rel=dict(rl), names=prior.get(game, {}), carry=car_by.get(game), **flags))
            for act, dd in L.dominant().items():
                if dd[2] >= MIN_LAW and (act not in ear or dd[2] >= ear[act][2]): ear[act] = dd
            rl.update(level_relations(L))
            main, st = axis_steps(L)
            if main is not None and st:
                lens = Counter(_len(v) for v, _ in st.values()); car_by[game] = (main, min(lens, key=lambda x: (-lens[x], x)))
        out[name] = {g: combine(parts) for g, parts in by.items()}
    return out


def from_sessions(sessions: Dict[str, object], pooled: bool = True) -> Dict[str, Dict[str, Dict[str, dict]]]:
    """arm -> variant -> game -> counters. `pooled` True: the name prior of a game is read from the levels of the OTHER games in ALL the arms given (a game never reads its own levels); False: from the other games of the same arm only, as in the seventh check."""
    per_arm = {arm: _dirs(s) for arm, s in sessions.items()}; games = {g for d in per_arm.values() for g in d} | {g for s in sessions.values() for (g, _) in s.ledgers}
    out = {}
    for arm, s in sessions.items():
        if pooled:
            allg: Dict[str, Dict[object, Counter]] = {}
            for d in per_arm.values():
                for g, acts in d.items():
                    for act, cn in acts.items(): allg.setdefault(g, {}).setdefault(act, Counter()).update(cn)
            prior = {g: name_prior(allg, g) for g in games}
        else: prior = {g: name_prior(per_arm[arm], g) for g in games}
        out[arm] = _run(s, prior)
    return out


def pool(per_arm: Dict[str, Dict[str, Dict[str, dict]]]) -> Dict[str, Dict[str, dict]]:
    """variant -> game -> counters over the arms."""
    out: Dict[str, Dict[str, dict]] = {v: {} for v in VARIANTS}
    for arm, vs in per_arm.items():
        if "error" in vs: continue
        for v, pg in vs.items():
            for g, c in pg.items(): out[v][g] = combine([out[v].get(g, {}), c])
    return out


def rates(c: dict) -> dict:
    r = F7.rates(c); r["retired_withheld"] = c.get("retired_withheld", 0)
    r["precision"] = (r["hits"] / r["predicted"]) if r["predicted"] else None
    return r


def retire_cost(base: dict, ret: dict) -> dict:
    """What a retire rule did against the same way without it: right moves lost, wrong moves avoided, and the ratio of the two (what one wrong move must cost, in right ones, for the rule to pay)."""
    lost = base["hits"] - ret["hits"]; avoided = (base["predicted"] - base["hits"]) - (ret["predicted"] - ret["hits"])
    return dict(lost=lost, avoided=avoided, ratio=(lost / avoided) if avoided > 0 else None, gain=(ret["precision"] - base["precision"]) if ret["precision"] is not None and base["precision"] is not None else None)


# ---------------------------------------------------------------- the clears
class Session8(LR.Session):
    """The replay's session that also keeps the screen just before every move that cleared a level (the plain session counts those moves and drops them)."""
    def __init__(self, min_law: int = 3):
        super().__init__(min_law); self.clears: List[dict] = []

    def note(self, game, a, before, after, lv0, lv1):
        if a["action"] != "RESET" and lv1 > lv0: self.clears.append(dict(game=game, level=lv0, action=G.action_key(a), before=key(before)))
        super().note(game, a, before, after, lv0, lv1)


def feed8(rows: list, make_env, arm: str, games=G.GAMES, moves: int = G.MOVES, action_of=None, min_law: int = 3) -> Session8:
    """`ledger_replay.feed` with the session that keeps the clears. Same replay, same checks of every saved flag."""
    s = Session8(min_law)
    if arm == "C":
        with LR.watching(C.PlayC, s): C.replay(rows, make_env, games, moves, action_of)
    elif arm == "B":
        with LR.watching(B.PlayB, s): B.replay(rows, make_env, games, moves, action_of)
    else:
        with LR.watching(G.Play, s): G.replay(rows, make_env, arm, games, moves, action_of)
    return s


def _counts(frame) -> Counter:
    c: Counter = Counter()
    for row in frame:
        for v in row: c[v] += 1
    return c


def level_facts(L: Optional[Ledger], last, action=None) -> dict:
    """What the screen `last` (the one just before the end of the level) shows against the level's first screen, with the cells a ticking strip changed in this level left out of both: moves recorded, how many changed the screen, the ticking strips, the symbols other than a strip whose number of cells differs, and whether `action` already had a law of at least MIN_LAW tries."""
    if L is None or not L.rows: return dict(moves=0, changed=0, strips=[], delta={}, other_change=False, other_symbols=0, law=False, click=("@" in str(action)) if action is not None else False)
    cells = L.tick_cells(); s0, s1 = _counts(L._mask_cells(L.rows[0].kb, cells)), _counts(L._mask_cells(key(last), cells)); strips = {t["symbol"] for t in L.ticks()}      # the cells a strip changed in this level are left out of both screens: a strip turning into background is not a change of the board
    delta = {str(x): s1[x] - s0[x] for x in set(s0) | set(s1) if s1[x] != s0[x]}
    other = {x: d for x, d in delta.items() if x not in {str(t) for t in strips}}
    dom = L.dominant().get(action) if (action is not None and "@" not in str(action)) else None
    return dict(moves=len(L.rows), changed=sum(1 for r in L.rows if r.changed()), strips=sorted(str(t) for t in strips), delta=delta, other_change=bool(other), other_symbols=len(other),
                law=bool(dom is not None and dom[2] >= MIN_LAW), click=("@" in str(action)) if action is not None else False)


def clear_study(sessions: Dict[str, Session8], min_moves: int = 20) -> dict:
    """{'cleared': [facts of each clear], 'uncleared': [facts of each level of at least `min_moves` recorded moves that the run did not clear]}."""
    cl, un = [], []
    for arm, s in sessions.items():
        done = set()
        for c in s.clears:
            L = s.ledgers.get((c["game"], c["level"])); f = level_facts(L, c["before"], c["action"]); f.update(arm=arm, game=c["game"], level=c["level"], action=str(c["action"]), moves=(f["moves"] + 1)); cl.append(f); done.add((c["game"], c["level"]))
        for (game, lv), L in sorted(s.ledgers.items()):
            if (game, lv) in done or len(L.rows) < min_moves: continue
            f = level_facts(L, L.rows[-1].ka); f.update(arm=arm, game=game, level=lv); un.append(f)
    return dict(cleared=cl, uncleared=un)


# ---------------------------------------------------------------- the printout
def text(pooled: Optional[Dict[str, Dict[str, dict]]], per_arm_prior: Optional[Dict[str, Dict[str, dict]]] = None, clears: Optional[dict] = None) -> str:
    pct = lambda x: "-" if x is None else f"{100 * x:.1f}%"; lines = []; full = (pooled or {}).get("full") or {}
    if full:
        tf = rates(combine(full.values()))
        lines.append("== 1 first tries with the name prior pooled over the four arms (against the full version: %d right of %d predicted; the seventh check, prior per arm: all7 %d of %d, best %d of %d)" % (tf["hits"], tf["predicted"], CHECK7["all7"][0], CHECK7["all7"][1], CHECK7["best"][0], CHECK7["best"][1]))
        for nm in ("once", "nm", "all6", "all7", "best"):
            pg = pooled.get(nm) or {}
            if not pg: continue
            t = rates(combine(pg.values()))
            lines.append(f"  {nm}: {t['predicted']} predicted ({t['predicted'] - tf['predicted']:+d}), right {t['hits']} ({t['hits'] - tf['hits']:+d}); right among predicted {pct(t['precision'])}; by name+seen length {t['nm_pred']} right {t['nm_hit']} ({pct(t['nm_precision'])}), "
                         f"by name+earlier level's length {t['nc_pred']} right {t['nc_hit']} ({pct(t['nc_precision'])}), carried relation {t['rel_pred']} right {t['rel_hit']}, rotation set {t['rot_pred']} right {t['rot_hit']}, one earlier try {t['once_pred']} right {t['once_hit']}")
        if per_arm_prior and per_arm_prior.get("all7"):
            t = rates(combine(per_arm_prior["all7"].values())); u = rates(combine(per_arm_prior["best"].values()))
            lines.append(f"  (prior per arm, as in the seventh check: all7 {t['predicted']} predicted, {t['hits']} right; best {u['predicted']} predicted, {u['hits']} right)")
        lines.append("== 2 the name prior from the other games, pooled over the arms: levels in which the action made a step, levels in which a prior existed, levels in which the prior named the direction made")
        c = combine(full.values()); acts = sorted({k.split(":")[1] for k in c if k.startswith("pri:")}, key=str)
        lines.append("   " + "; ".join(f"{a}: {c.get(f'pri:{a}:n', 0)}, {c.get(f'pri:{a}:has', 0)}, {c.get(f'pri:{a}:has_ok', 0)}" for a in acts))
        for g, cg in sorted(full.items()):
            lines.append(f"   {g}: " + "; ".join(f"{a}: {cg.get(f'pri:{a}:n', 0)}, {cg.get(f'pri:{a}:has', 0)}, {cg.get(f'pri:{a}:has_ok', 0)}" for a in acts))
        lines.append("== 3 the retire rule (all7 way): a law wrong by a different step `retire` times is not used again in that level")
        a7 = rates(combine(pooled["all7"].values()))
        lines.append(f"   no rule: {a7['predicted']} predicted, {a7['hits']} right ({pct(a7['precision'])})")
        for nm, lab in (("all7_r1", "retire after 1"), ("all7_r2", "retire after 2"), ("best_r1", "best way, retire after 1")):
            if nm not in pooled: continue
            r = rates(combine(pooled[nm].values())); base = a7 if nm != "best_r1" else rates(combine(pooled["best"].values())); k = retire_cost(base, r)
            lines.append(f"   {lab}: {r['predicted']} predicted, {r['hits']} right ({pct(r['precision'])}); moves left to the model {r['retired_withheld']}; right moves lost {k['lost']}, wrong moves avoided {k['avoided']}, "
                         f"a wrong move must cost {'-' if k['ratio'] is None else format(k['ratio'], '.2f')} right ones for the rule to pay; precision {'-' if k['gain'] is None else format(100 * k['gain'], '+.1f')} points")
    if clears is not None:
        cl, un = clears["cleared"], clears["uncleared"]
        lines.append(f"== 4 the clears: {len(cl)} levels cleared in the four arms; for each, the screen just before the clearing move against the level's first screen")
        for f in cl:
            lines.append(f"   arm {f['arm']} {f['game']} level {f['level']}: {f['moves']} moves, {f['changed']} changed the screen; clearing action {f['action']} ({'a click' if f['click'] else ('already had a law' if f['law'] else 'no law yet')}); strips {f['strips'] or 'none'}; "
                         f"symbols other than a strip with a different number of cells: {f['other_symbols']}" + (f" {f['delta']}" if f["other_symbols"] else ""))
        k = sum(1 for f in cl if f["other_change"]); m = sum(1 for f in un if f["other_change"])
        lines.append(f"   levels of at least 20 recorded moves not cleared: {len(un)}; with such a symbol: {m}. Cleared with such a symbol: {k} of {len(cl)}")
    return "\n".join(lines)


# ---------------------------------------------------------------- the reading lines (fixed before the run, docs/FORESEE_CHECK_8.md)
G_LINES = [
    ("guard", "the saved moves of all four arms (K, A, B and C) replay into fresh copies of the games and reproduce every saved flag"),
    ("instrument", "with the name prior taken per arm and no retire rule the check gives the seventh check's figures on the same runs: full 603 right of 729 predicted, once 663 of 797, all6 672 of 809, all7 690 of 832, best 703 of 832; and the pooled full, once and all6 give the same"),
    ("claim", "with the prior pooled over the four arms, for each of ACTION1 to ACTION4 (at least 4 levels with a step) a prior exists in at least 90 percent of the levels with a step, it names the direction made in at least 90 percent of the levels in which it exists, and every game has a prior for each of them in at least one level"),
    ("claim", "the name (nm alone, both lengths, prior pooled) predicts at least 50 moves, and at least 75 percent of them are right"),
    ("claim", "with once, rot, rel and nm together (all7, prior pooled) at least 110 more moves are predicted than in the full version, and at least 80 percent of the moves added are right"),
    ("claim", "retiring a law after one wrong step (all7 with retire 1) raises the share right among the predicted moves by at least 2.0 points and loses at most 60 right moves"),
    ("claim", "for retire 1 at least 20 wrong moves are avoided and a wrong move would have to cost at most 2.0 right ones for the rule to pay (right moves lost divided by wrong moves avoided)"),
    ("claim", "retiring after two wrong steps (retire 2) avoids at least 10 wrong moves and needs a lower cost ratio than retire 1"),
    ("claim", "the replays show exactly 6 level clears over the four arms"),
    ("claim", "in at least 4 of the clears, a symbol that is not a ticking strip has a different number of cells on the screen just before the clearing move than on the level's first screen"),
    ("claim", "that sign is at least 25 points more common in the cleared levels than in the levels of 20 or more recorded moves that were not cleared (at least 5 such levels)"),
    ("claim", "in at least 4 of the clears the clearing move was made with an action that already had a law of at least 3 tries in that level"),
]
HEADLINE = "WITH THE NAME READ FROM ALL THE OTHER GAMES IN ALL FOUR ARMS, A FIRST TRY CAN BE ANSWERED FROM ITS NAME AND THE LENGTH OF A STEP ALREADY SEEN"


def _combined(variants, name) -> Optional[dict]:
    pg = (variants or {}).get(name)
    return rates(combine(pg.values())) if pg else None


def verdict(pooled: Optional[Dict[str, Dict[str, dict]]], per_arm_prior: Optional[Dict[str, Dict[str, dict]]], clears: Optional[dict], replays_ok: Optional[bool]) -> List[Tuple[int, str, str, Optional[bool]]]:
    v: List[Optional[bool]] = [None] * len(G_LINES)
    if replays_ok is not None: v[0] = bool(replays_ok)
    if pooled and per_arm_prior:
        ok = True
        for nm in ("full", "once", "all6", "all7", "best"):
            r = _combined(per_arm_prior, nm)
            if r is None or (r["hits"], r["predicted"]) != CHECK7[nm]: ok = False
        for nm in ("full", "once", "all6"):
            r = _combined(pooled, nm)
            if r is None or (r["hits"], r["predicted"]) != CHECK7[nm]: ok = False
        v[1] = ok
    if pooled and pooled.get("full"):
        c = combine(pooled["full"].values())
        v[2] = all(c.get(f"pri:{a}:n", 0) >= 4 and c.get(f"pri:{a}:has", 0) >= 0.90 * c.get(f"pri:{a}:n", 0) and c.get(f"pri:{a}:has_ok", 0) >= 0.90 * c.get(f"pri:{a}:has", 0) for a in ACTIONS) \
               and all(all(cg.get(f"pri:{a}:has", 0) > 0 for a in ACTIONS if cg.get(f"pri:{a}:n", 0) > 0) for cg in pooled["full"].values())
        full, nm, a7 = _combined(pooled, "full"), _combined(pooled, "nm"), _combined(pooled, "all7")
        if nm is not None: v[3] = nm["name_pred"] >= 50 and nm["name_precision"] is not None and nm["name_precision"] >= 0.75
        if a7 is not None and full is not None:
            add = a7["predicted"] - full["predicted"]; addh = a7["hits"] - full["hits"]; v[4] = add >= 110 and addh >= 0.80 * add
            r1, r2 = _combined(pooled, "all7_r1"), _combined(pooled, "all7_r2")
            if r1 is not None:
                k1 = retire_cost(a7, r1); v[5] = k1["gain"] is not None and k1["gain"] >= 0.02 and k1["lost"] <= 60
                v[6] = k1["avoided"] >= 20 and k1["ratio"] is not None and k1["ratio"] <= 2.0
                if r2 is not None:
                    k2 = retire_cost(a7, r2); v[7] = k2["avoided"] >= 10 and k2["ratio"] is not None and k1["ratio"] is not None and k2["ratio"] < k1["ratio"]
    if clears is not None:
        cl, un = clears["cleared"], clears["uncleared"]; v[8] = len(cl) == CLEARS_SEEN
        v[9] = sum(1 for f in cl if f["other_change"]) >= 4
        if len(un) >= 5 and cl: v[10] = (sum(1 for f in cl if f["other_change"]) / len(cl)) - (sum(1 for f in un if f["other_change"]) / len(un)) >= 0.25
        elif cl: v[10] = False
        v[11] = sum(1 for f in cl if f["law"]) >= 4
    return [(i + 1, t, x, v[i]) for i, (t, x) in enumerate(G_LINES)]


def claims_allowed(checks) -> bool:
    return bool(checks[0][3]) and bool(checks[1][3])


def headline(checks) -> str:
    if not claims_allowed(checks): return "NOT READ: a guard or instrument line is not yes, no figure is claimed until the cause is found"
    r = {i: v for i, _, _, v in checks}
    return HEADLINE if r[4] and r[5] else "MIXED: line 4 " + ("yes" if r[4] else "NO") + ", line 5 " + ("yes" if r[5] else "NO")
