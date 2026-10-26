"""Item 136 (test 161): can the host say where the piece will go BEFORE the move, using only what the ledger has seen in earlier moves? A free check on saved runs: no model, no cost.

Why. The pasted AI summaries of Rajnish Choubey's work ask for a forward look: "propose a candidate, check it on a shadow copy, commit or drop it" is pillar III of his RFC-001 (copy-on-write shadow
paging: a candidate writes to a shadow block, and a rejected one is dropped with nothing to undo) and pillar II is the gate that rejects a change the held state does not allow. The game itself cannot
be copied to look ahead in (a real game is played in the real game), so the shadow has to be a model the host LEARNED: what the ledger has seen. Whether that learned model is good enough to plan on is a
measurable thing, and this module measures it. How the structure is read into game terms is for Sir to confirm; nothing here is his text.

What is predicted, from the earlier moves of one level only (a fresh ledger per level, exactly as the read-out of arm C):
  * `step_only`: the action moves the object of symbol S by the step the ledger has seen it make most often (`Ledger.dominant`, needs two tries of that action);
  * `learned`: the same, except that "does not move" is predicted when a colour among the cells the piece would step onto has, in the earlier moves of this level, stopped this piece (alone among the
    non-floor colours there) at least once and has never been stepped onto (the host learns its own gate from what it has seen refused: pillar II read as a held record, not as a given wall);
  * `wall_aware`: the same, except that when the cells the piece would step onto are outside the screen or hold a colour the ledger has seen never change (`rigid_symbols`, after MIN_LAW moves), the
    prediction is "does not move". When there is not exactly one object of that symbol, the wall check is not made and the prediction is the same as step_only (counted as `wall_unknown`).
A move of a click action ("@" in its name) or one the ledger has no step for is not predicted (counted as `abstained`).
Scored against the move that was really played: the prediction is right when the object of that symbol moved by exactly the predicted step, or, when "does not move" was predicted, did not move.

What it does NOT show: that a planner built on this would clear a level; how the model chooses a goal (nothing here does); anything about a game it was not run on. A pure function of the saved moves."""
from __future__ import annotations
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

from axi.engine.ledger import Ledger, components, MIN_LAW

MIN_TRIES = 2
EDGE = "<edge of the screen>"


def foresee(L: Ledger, frame, action) -> Optional[dict]:
    """The prediction for `action` on `frame`, from the ledger's earlier moves: None (abstain) or dict(symbol, step, share, tries, blocked, wall_known)."""
    if "@" in str(action): return None
    d = L.dominant().get(action)
    if d is None or d[3] < MIN_TRIES: return None
    sym, step, n, tot = d
    out = dict(symbol=sym, step=step, share=n / tot, tries=tot, blocked=False, wall_known=False)
    rigid = set(L.rigid_symbols()) if len(L.rows) >= MIN_LAW else set()
    mine = [c for s, c in components(frame, L.bg) if s == sym]
    if len(mine) != 1: return out
    out["wall_known"] = True; cells = mine[0]; dx, dy = step; h = len(frame); w = len(frame[0]) if h else 0; tc = set()
    for x, y in cells:
        nx, ny = x + dx, y + dy
        if (nx, ny) in cells: continue
        if not (0 <= nx < w and 0 <= ny < h): tc.add(EDGE); out["blocked"] = True
        else:
            tc.add(frame[ny][nx])
            if frame[ny][nx] in rigid and frame[ny][nx] != sym: out["blocked"] = True
    out["target"] = frozenset(tc)
    return out


def _actual(rec, sym) -> Optional[Tuple[int, int]]:
    for o in rec.ops:
        if o[0] == "move" and o[1] == sym: return o[3]
    return None


def evaluate(moves: Iterable[Tuple[object, object, object]], min_law: int = MIN_LAW) -> dict:
    """`moves`: (action, before, after) of one level in the order played. A fresh ledger is fed; each move is predicted from the moves before it, then scored. Pure."""
    L = Ledger(min_law=min_law); c = Counter(); mem: Dict[tuple, Counter] = {}
    for action, before, after in moves:
        p = foresee(L, before, action)
        rec = L.record(action, before, after)
        c["moves"] += 1
        if p is None: c["abstained"] += 1; continue
        got = _actual(rec, p["symbol"]); c["predicted"] += 1
        c["step_hit"] += int(got == p["step"])
        if not p["wall_known"]: c["wall_unknown"] += 1
        if p["blocked"]:
            c["blocked_said"] += 1; c["blocked_right"] += int(got is None); c["wall_hit"] += int(got is None)
        else: c["wall_hit"] += int(got == p["step"])
        if got is None: c["did_not_move"] += 1
        elif got != p["step"]: c["other_step"] += 1
        tg = p.get("target")
        if tg is not None:
            sym = p["symbol"]; bad = [x for x in tg if mem.get((sym, x), Counter())["stopped"] and not mem.get((sym, x), Counter())["passed"]]
            say = bool(bad) if p["wall_known"] else False
            if say: c["learned_said"] += 1; c["learned_right"] += int(got is None); c["learned_hit"] += int(got is None)
            else: c["learned_hit"] += int(got == p["step"])
            if got == p["step"]:
                for x in tg: mem.setdefault((sym, x), Counter())["passed"] += 1
            elif got is None:
                hard = [x for x in tg if x != L.bg]
                if len(hard) == 1: mem.setdefault((sym, hard[0]), Counter())["stopped"] += 1
        elif p["wall_known"] is False: c["learned_hit"] += int(got == p["step"])
    return dict(c)


def combine(parts: Iterable[dict]) -> dict:
    t: Counter = Counter()
    for p in parts: t.update(p)
    return dict(t)


def rates(c: dict) -> dict:
    n = c.get("predicted", 0)
    f = lambda a, b: (a / b) if b else None
    return dict(moves=c.get("moves", 0), predicted=n, abstained=c.get("abstained", 0), coverage=f(n, c.get("moves", 0)),
                step_only=f(c.get("step_hit", 0), n), wall_aware=f(c.get("wall_hit", 0), n), learned=f(c.get("learned_hit", 0), n),
                learned_said=c.get("learned_said", 0), learned_precision=f(c.get("learned_right", 0), c.get("learned_said", 0)),
                blocked_said=c.get("blocked_said", 0), blocked_precision=f(c.get("blocked_right", 0), c.get("blocked_said", 0)),
                did_not_move=c.get("did_not_move", 0), other_step=c.get("other_step", 0), wall_unknown=c.get("wall_unknown", 0))


def from_session(session) -> Dict[str, dict]:
    """Per game, the counts over every level of a `ledger_replay.Session` (the rows hold the frames, so the ledger is simply fed again)."""
    by: Dict[str, List[dict]] = {}
    for (game, lv), L in sorted(session.ledgers.items()):
        by.setdefault(game, []).append(evaluate((r.action, r.kb, r.ka) for r in L.rows))
    return {g: combine(parts) for g, parts in by.items()}


def text(per_game: Dict[str, dict]) -> str:
    pct = lambda x: "-" if x is None else f"{100 * x:.1f}%"
    lines = []; tot = combine(per_game.values()) if per_game else {}
    for g, c in list(per_game.items()) + ([("ALL GAMES", tot)] if per_game else []):
        r = rates(c)
        lines.append(f"{g}: {r['moves']} moves, {r['predicted']} predicted (coverage {pct(r['coverage'])}), step only right {pct(r['step_only'])}, with the colours that never changed as walls right {pct(r['wall_aware'])} ('does not move' said {r['blocked_said']} times, right {pct(r['blocked_precision'])}), "
                     f"with the colours that stopped the piece before {pct(r['learned'])} ('does not move' said {r['learned_said']} times, right {pct(r['learned_precision'])}); wrong because the piece did not move {r['did_not_move']}, because it moved otherwise {r['other_step']}; wall check not possible {r['wall_unknown']}")
    return "\n".join(lines)


# ---------------------------------------------------------------- the reading lines (fixed before the run, docs/FORESEE_CHECK.md)
G_LINES = [
    ("guard", "the saved moves of all four arms (K, A, B and C) replay into fresh copies of the games and reproduce every saved flag"),
    ("instrument", "a move is predicted (an action with a step seen in at least two earlier tries of the level) in at least 60 percent of the recorded moves (a move that wins a level, and a reset, is not a transition and is not recorded), over the four arms"),
    ("claim", "step only is right in at least 70 percent of the predicted moves, over all four arms and games"),
    ("claim", "the prediction that learns what stopped the piece is right more often than step only in at least 3 of the 4 games"),
    ("claim", "when the learned prediction says the piece does not move it is right in at least 80 percent of those cases"),
    ("claim", "the learned prediction is right in at least 90 percent of the predicted moves, over all four arms and games"),
    ("claim", "the colours that never changed, taken as walls, are right no more often than the learned prediction in every one of the 4 games"),
    ("claim", "in no game is the learned prediction more than 2 points worse than step only"),
]
HEADLINE = "THE HOST'S LEARNED MODEL OF THE MOVES PREDICTS THE NEXT STEP WELL ENOUGH TO BUILD A PATH-FINDER ON"


def verdict(per_game: Optional[Dict[str, dict]], replays_ok: Optional[bool]) -> List[Tuple[int, str, str, Optional[bool]]]:
    """(number, tag, text, True / False / None) per reading line; None = not read (nothing given). `per_game`: game -> counts of `evaluate` summed over the four arms."""
    v: List[Optional[bool]] = [None] * len(G_LINES)
    if replays_ok is not None: v[0] = bool(replays_ok)
    if per_game:
        tot = rates(combine(per_game.values())); rs = {g: rates(c) for g, c in per_game.items()}
        v[1] = tot["coverage"] is not None and tot["coverage"] >= 0.60
        v[2] = tot["step_only"] is not None and tot["step_only"] >= 0.70
        better = sum(1 for r in rs.values() if r["learned"] is not None and r["step_only"] is not None and r["learned"] > r["step_only"]); v[3] = better >= 3
        v[4] = tot["learned_precision"] is not None and tot["learned_precision"] >= 0.80
        v[5] = tot["learned"] is not None and tot["learned"] >= 0.90
        v[6] = len(rs) == 4 and all(r["wall_aware"] is not None and r["learned"] is not None and r["wall_aware"] <= r["learned"] for r in rs.values())
        v[7] = all(r["learned"] is not None and r["step_only"] is not None and r["learned"] >= r["step_only"] - 0.02 for r in rs.values())
    return [(i + 1, t, x, v[i]) for i, (t, x) in enumerate(G_LINES)]


def claims_allowed(checks) -> bool:
    """No figure is claimed unless the guard and the instrument line read yes."""
    return bool(checks[0][3]) and bool(checks[1][3])


def headline(checks) -> str:
    if not claims_allowed(checks): return "NOT READ: a guard or instrument line is not yes, no figure is claimed until the cause is found"
    r = {i: v for i, _, _, v in checks}
    return HEADLINE if r[4] and r[6] else "MIXED: line 4 " + ("yes" if r[4] else "NO") + ", line 6 " + ("yes" if r[6] else "NO")
