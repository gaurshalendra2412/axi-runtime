"""Item 139 (test 164): the fourth forward check on the saved ARC-AGI-3 runs: the board carried from level to level.

In the first three checks every level starts from nothing: a fresh ledger, no floor, no law for any action. In the second check 207 of 936 moves (22 %) were not predicted at all, most of them because an action needs two earlier tries in each
level before the check will speak. An action means the same thing in the next level of the same game, and so does the floor the piece walked on. The pattern is the one in the owner's reading of Sir's writing: one board that every view reads
at once, and nothing held in a silo (the nine-profile essay, the "tab sync" of the other); here it is read in game terms, for Sir to confirm. Two things are carried, both from the EARLIER levels of the SAME game and the same arm only, never
from a later level and never from another game or arm:
  * the LAW of each action: the step it made most often (the symbol that moved and the step), kept when it was seen at least 3 times; used for a move only when the level has no law of its own for that action yet, and only when that
    symbol is on the screen (otherwise the move is not predicted, as before);
  * the FLOOR: the colours the piece has walked on, so that the whole-front rule can start earlier.
Nothing else is carried: not a cell, not a wall, not where the piece is (the layout of a new level is new). Three ways on the same moves: fresh (the third check with both changes, unchanged), laws carried, laws and floor carried.

What it does NOT show: that a path-finder clears a level, or what the goal of a level is. A pure function of the saved moves; no model, no cost."""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

from axi.experiments import foresee2 as F2
from axi.experiments import foresee3 as F3
from axi.experiments.foresee import combine

__all__ = ["from_session", "rates", "text", "verdict", "headline", "claims_allowed", "G_LINES", "HEADLINE", "combine", "WAYS", "FRESH_PREDICTED"]

FRESH_PREDICTED = 729                      # predicted moves of the third check (and the second) on the same runs: owner's foresee_check_2_all.json
WAYS = {"fresh": (False, False), "laws": (True, False), "laws+floor": (True, True)}          # name -> (carry the laws, carry the floor)


def from_session(session) -> Dict[str, Dict[str, dict]]:
    """way -> game -> counters. The levels of a game are played in order, the board carried only from the earlier ones."""
    out: Dict[str, Dict[str, dict]] = {}
    for name, (ul, uf) in WAYS.items():
        by: Dict[str, List[dict]] = {}; carry: Dict[str, dict] = {}
        for (game, lv), L in sorted(session.ledgers.items()):
            cy = None if not (ul or uf) else carry.setdefault(game, dict(laws={}, walked=set(), use_laws=ul, use_floor=uf))
            by.setdefault(game, []).append(F3.evaluate(((r.action, r.kb, r.ka) for r in L.rows), True, True, carry=cy))
        out[name] = {g: combine(parts) for g, parts in by.items()}
    return out


def rates(c: dict) -> dict:
    r = F3.rates(c); f = lambda a, b: (a / b) if b else None
    r.update(per_move=f(c.get("cell_hit", 0), c.get("moves", 0)), carried=c.get("carried_pred", 0), carried_right=f(c.get("carried_hit", 0), c.get("carried_pred", 0)))
    return r


def text(ways: Dict[str, Dict[str, dict]]) -> str:
    pct = lambda x: "-" if x is None else f"{100 * x:.1f}%"; lines = []
    for name in WAYS:
        pg = ways.get(name) or {}
        if not pg: continue
        lines.append(f"== {name} ({'laws carried' if WAYS[name][0] else 'fresh start'}, {'floor carried' if WAYS[name][1] else 'floor not carried'})")
        for g, c in list(pg.items()) + [("ALL GAMES", combine(pg.values()))]:
            r = rates(c)
            lines.append(f"  {g}: {r['moves']} moves, {r['predicted']} predicted ({pct(r['coverage'])}); right among those predicted {pct(r['cell_learned'])}; right per move played {pct(r['per_move'])}; "
                         f"predicted from an earlier level's law {r['carried']}, right {pct(r['carried_right'])}; 'does not move' said {r['cell_said']} times")
    return "\n".join(lines)


G_LINES = [
    ("guard", "the saved moves of all four arms (K, A, B and C) replay into fresh copies of the games and reproduce every saved flag"),
    ("instrument", "with nothing carried the check predicts 729 moves, as the second and third checks do on the same runs"),
    ("claim", "with the laws carried, at least 85 percent of the moves played are predicted (fresh: 77.9 percent)"),
    ("claim", "with the laws carried, the share of moves played that were predicted right is higher than fresh in at least 3 of the 4 games"),
    ("claim", "the moves predicted only because of a carried law are right in at least 60 percent of them, and there are at least 50 of them"),
    ("claim", "with the laws carried, the moves predicted right among those predicted are no more than 2 points fewer than fresh, over all four arms and games"),
    ("claim", "with the floor carried as well, the share of moves played that were predicted right is not lower than with the laws only"),
    ("claim", "with the laws carried, at least 70 percent of the moves played are predicted right (the second check: 56.3 percent)"),
]
HEADLINE = "THE BOARD CARRIED FROM LEVEL TO LEVEL PREDICTS MORE OF THE MOVES, AND RIGHT, THAN A FRESH START"


def verdict(ways: Optional[Dict[str, Dict[str, dict]]], replays_ok: Optional[bool]) -> List[Tuple[int, str, str, Optional[bool]]]:
    v: List[Optional[bool]] = [None] * len(G_LINES)
    if replays_ok is not None: v[0] = bool(replays_ok)
    if ways and ways.get("fresh") and ways.get("laws") and ways.get("laws+floor"):
        fr, lw, lf = ways["fresh"], ways["laws"], ways["laws+floor"]; tf, tl, tlf = (rates(combine(x.values())) for x in (fr, lw, lf))
        v[1] = tf["predicted"] == FRESH_PREDICTED
        v[2] = tl["coverage"] is not None and tl["coverage"] >= 0.85
        v[3] = sum(1 for g in lw if g in fr and rates(lw[g])["per_move"] is not None and rates(fr[g])["per_move"] is not None and rates(lw[g])["per_move"] > rates(fr[g])["per_move"]) >= 3
        v[4] = tl["carried"] >= 50 and tl["carried_right"] is not None and tl["carried_right"] >= 0.60
        v[5] = tl["cell_learned"] is not None and tf["cell_learned"] is not None and tl["cell_learned"] >= tf["cell_learned"] - 0.02
        v[6] = tlf["per_move"] is not None and tl["per_move"] is not None and tlf["per_move"] >= tl["per_move"]
        v[7] = tl["per_move"] is not None and tl["per_move"] >= 0.70
    return [(i + 1, t, x, v[i]) for i, (t, x) in enumerate(G_LINES)]


def claims_allowed(checks) -> bool:
    return bool(checks[0][3]) and bool(checks[1][3])


def headline(checks) -> str:
    if not claims_allowed(checks): return "NOT READ: a guard or instrument line is not yes, no figure is claimed until the cause is found"
    r = {i: v for i, _, _, v in checks}
    return HEADLINE if r[4] and r[6] else "MIXED: line 4 " + ("yes" if r[4] else "NO") + ", line 6 " + ("yes" if r[6] else "NO")
