"""Item 133: the transition ledger (axi/engine/ledger.py, Update 92) run over the SAVED ARC-AGI-3 runs of Test 24 and Test 25, to see what it does on real screens.

No model is called and nothing is paid for. The saved result files hold the model's replies, not the screens, so the replies are played back into fresh copies of the games (the existing `replay` of
gemini_arc / gemini_arc_b, which still checks every saved flag and refuses a doctored row), and the ledger is fed from the same moves. It adds nothing to the replay and changes nothing in it: `apply` is wrapped
for the length of one call and put back afterwards.

A move that completes a level and a RESET are NOT recorded as a transition (the whole screen changes there), they are counted. Each (game, level) has its own ledger, so a new level is a new world.

What it reports per game (host-side facts only, the model's rules are not asked for): moves recorded, moves that changed the screen, the operations found (move / recolour / reshape / gone / new), moves where one symbol is both
'gone' and 'new' (identity may have been lost: an honest count of how often the matching gave up), moves where the object-level change does not add up to the count change, the hidden-state flags and the moves where they fire,
counts that never changed and counts that broke, what each action does, effects apart from what moved, touches, and the host time per move.

Needs the toolkit only for a real run (`pip install arc-agi arcengine`, on a machine that can reach PyPI); the tests use the stand-in game of test 143. Run:  python -m axi.experiments.ledger_replay <result.json> --arm K|A|B
Comes from the structure of Rajnish Choubey's RFC-001, as read in docs/HISTORY_LEDGER.md (the reading in game terms is for Sir to confirm)."""
from __future__ import annotations
import argparse, contextlib, json, time
from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple

from axi.engine.ledger import Ledger
from axi.experiments import gemini_arc as G
from axi.experiments import gemini_arc_b as B


class Session:
    """The ledgers of one replay: one per (game, level), plus what was not recorded and how long the host took."""
    def __init__(self, min_law: int = 3):
        self.min_law = min_law; self.ledgers: Dict[Tuple[str, int], Ledger] = {}; self.skipped: Dict[str, Counter] = {}; self.seconds: Dict[str, float] = {}; self.moves: Dict[str, int] = {}

    def note(self, game: str, a: dict, before, after, lv0: int, lv1: int) -> None:
        sk = self.skipped.setdefault(game, Counter())
        if a["action"] == "RESET": sk["reset"] += 1; return
        if lv1 != lv0: sk["level_change"] += 1; return
        t = time.perf_counter(); L = self.ledgers.setdefault((game, lv0), Ledger(min_law=self.min_law)); L.record(G.action_key(a), before, after)
        self.seconds[game] = self.seconds.get(game, 0.0) + time.perf_counter() - t; self.moves[game] = self.moves.get(game, 0) + 1


@contextlib.contextmanager
def watching(cls, session: Session):
    """Wrap `cls.apply` so that every move the replay plays is also shown to the ledger. Put back on exit, even on an error."""
    orig = cls.apply

    def wrapped(self, a, *args, **kw):
        before, lv0 = self.screen, self.levels()
        out = orig(self, a, *args, **kw)
        session.note(self.game, a, before, self.screen, lv0, self.levels())
        return out
    cls.apply = wrapped
    try: yield
    finally: cls.apply = orig


def feed(rows: list, make_env: Callable, arm: str, games=G.GAMES, moves: int = G.MOVES, action_of: Optional[Callable] = None, min_law: int = 3) -> Session:
    """Replay the saved rows of one arm into fresh games and feed the ledger. Raises ReplayMismatch if a saved flag differs, exactly as the plain replay does."""
    s = Session(min_law)
    if arm == "B":
        with watching(B.PlayB, s): B.replay(rows, make_env, games, moves, action_of)
    else:
        with watching(G.Play, s): G.replay(rows, make_env, arm, games, moves, action_of)
    return s


def _balance_bad(r) -> bool:
    """True when the object-level change of one move does not add up to the change in the counts (the check of test 158)."""
    d = Counter()
    for o in r.ops:
        if o[0] == "gone": d[("cells", o[1])] -= o[3]; d[("objects", o[1])] -= 1
        elif o[0] == "new": d[("cells", o[1])] += o[3]; d[("objects", o[1])] += 1
        elif o[0] == "recolour": d[("cells", o[1])] -= o[4]; d[("cells", o[2])] += o[4]; d[("objects", o[1])] -= 1; d[("objects", o[2])] += 1
    return {q: v for q, v in d.items() if v} != {q: a - b for q, (b, a) in r.qdelta.items()}


def summarize(s: Session) -> Dict[str, dict]:
    """Per game, plain numbers (no text). Pure: the same session gives the same answer."""
    out: Dict[str, dict] = {}
    for (game, lv), L in sorted(s.ledgers.items()):
        g = out.setdefault(game, dict(levels_seen=0, moves=0, changed=0, ops=Counter(), identity_maybe_lost=0, balance_bad=0, ambiguous=0, reshaped=0, hidden_state=[], laws=0, lost_laws=0, came_back=0, apart=0, touch=0, actions={}))
        g["levels_seen"] += 1
        for r in L.rows:
            g["moves"] += 1; g["changed"] += int(r.changed()); g["ambiguous"] += int(bool(r.ambiguous)); g["reshaped"] += int(bool(r.reshaped)); g["apart"] += len(r.apart); g["touch"] += int(bool(r.touch))
            for o in r.ops: g["ops"][o[0]] += 1
            gone = {o[1] for o in r.ops if o[0] == "gone"}; new = {o[1] for o in r.ops if o[0] == "new"}
            g["identity_maybe_lost"] += int(bool(gone & new)); g["balance_bad"] += int(_balance_bad(r))
        g["hidden_state"] += [dict(level=lv, **h) for h in L.hidden_state()]
        g["laws"] += len(L.laws()); g["lost_laws"] += sum(1 for b in L.breaks if not b["back"]); g["came_back"] += sum(1 for b in L.breaks if b["back"])
        for a, (sym, v, n, t) in L.dominant().items(): g["actions"].setdefault(a, []).append(dict(level=lv, symbol=sym, step=v, n=n, of=t))
    for game, g in out.items():
        g["ops"] = dict(g["ops"]); g["skipped"] = dict(s.skipped.get(game, {}))
        g["ms_per_move"] = round(1000 * s.seconds.get(game, 0.0) / max(1, s.moves.get(game, 0)), 3)
    return out


def text(s: Session, max_items: int = 6) -> str:
    """The summary and the ledger's own reading lines, as plain text for the person."""
    lines: List[str] = []
    for game, g in summarize(s).items():
        lines.append(f"== {game}: {g['moves']} moves recorded over {g['levels_seen']} level(s), {g['changed']} changed the screen; not recorded: {g['skipped'] or 'none'}; host time {g['ms_per_move']} ms per move")
        lines.append(f"   operations found: {g['ops'] or 'none'}; one symbol both gone and new in {g['identity_maybe_lost']} moves; does not add up in {g['balance_bad']}; ambiguous {g['ambiguous']}; reshaped {g['reshaped']}; touches {g['touch']}; apart {g['apart']}")
        lines.append(f"   hidden-state flags: {len(g['hidden_state'])}" + ("".join(f" [level {h['level']}, '{h['action']}', moves {h['moves']}]" for h in g["hidden_state"][:max_items])))
        lines.append(f"   counts that never changed {g['laws']}, broken and not back {g['lost_laws']}, broken and came back {g['came_back']}")
        for (gm, lv), L in sorted(s.ledgers.items()):
            if gm == game:
                for x in L.report(max_items): lines.append(f"   level {lv}: {x}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("result", help="a saved result file, e.g. docs/results/axi_gemini_arc_default.json"); ap.add_argument("--arm", required=True, choices=["K", "A", "B"])
    ap.add_argument("--json", action="store_true", help="print the numbers as JSON instead of text")
    a = ap.parse_args(argv)
    d = json.load(open(a.result)); arm = d[a.arm]; rows = arm["rows"]; meta = arm.get("meta", {}); games = tuple(meta.get("games", G.GAMES)); moves = int(meta.get("moves", G.MOVES))
    import arc_agi

    def make_env(game): return arc_agi.Arcade().make(game, seed=G.GAME_SEED)
    s = feed(rows, make_env, a.arm, games, moves)
    print(json.dumps(summarize(s), indent=1, default=str) if a.json else text(s)); return 0


if __name__ == "__main__": raise SystemExit(main())
