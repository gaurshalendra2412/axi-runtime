"""The bigger ARC-AGI-3 test (docs/GEMINI_ARC_BIG_TEST.md, test 27): the same model, eight games, 60 moves each, three arms: the plain model (K), the model with the AXI ledger read-out (C), and the model with the read-out plus the goal-readers (D).

Tests 24 to 26 played four games for 60 moves. They could say little: the clears were 2, 1, 1 and 2 levels, all of them level 0. This test has twice the games (the four of the earlier tests and four more that the toolkit offers, chosen by a fixed rule),
the same 60 moves in each (the owner asked for a cost of about 20 dollars, and the plain model's chat grows with the square of the moves), and the three arms below, each played once on every game.
  K  the plain model: one chat per game, every screen and every reply kept in the chat (`gemini_arc.py`, arm K). What a firm gets when it just keeps talking to the model.
  C  arm C of Test 26 (`gemini_arc_c.py`): the Record, the check with re-asks, the object view, the untried list, the conductor and the ledger read-out. It is arm B plus the ledger, nothing else.
  D  arm C plus ONE more section in each turn, the NOTES, written by the host from the ledgers and the clears of THIS run only (`goalbook.py`, Update 106; item 144): which actions changed the board without moving a piece, what is on this level's first screen
     and was not on an earlier level's, what the earlier levels of this game suggest (a direction, a step) as a suggestion that this level can cancel, and guesses at what clears a level (marked as guesses, none tested). The section is left out when it has nothing to say.
It tests whether the framework's read-outs make the same model play better than the plain model on more games, and whether the goal-readers add anything to the ledger read-out. It does NOT use the official scoring (RHAE), it is one run per arm and game, and it is not an entry
on the ARC-AGI-3 leaderboard (its rules, scorecards and hidden games are not used).

Nothing of Tests 24 to 26, of `goalbook.py`, of `gemini_arc.py`, `gemini_arc_b.py`, `gemini_arc_c.py` or of the framework's core is changed: this module imports them and edits none. The notes of move m are a function of the moves before m (the replay rebuilds the ledgers and the
clears move by move and refuses a row whose `goal_lines` differs)."""
from __future__ import annotations
import hashlib, json, os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from axi.engine.ledger import Ledger, background, key as _key
from axi.experiments import gemini_arc as R
from axi.experiments import gemini_arc_b as B
from axi.experiments import gemini_arc_c as C
from axi.experiments import gemini_long as GL
from axi.experiments import goalbook as GB
from axi.experiments.resumable import load_rows, _roundtrip

MODEL, GAME_SEED, A_REASKS = R.MODEL, R.GAME_SEED, R.A_REASKS
BASE_GAMES = R.GAMES                          # the four games of Tests 24 to 26 come first
BIG_GAMES, BIG_MOVES = 8, 60
K_TOKEN_CAP, C_TOKEN_CAP, D_TOKEN_CAP = 120_000_000, 8_000_000, 8_000_000
PILOT_GAME, PILOT_MOVES = R.PILOT_GAME, 12
MIN_LAW, GOAL_MAX_LINES = C.MIN_LAW, GB.MAX_LINES
GOAL_HEAD = "Notes on this level from the framework (computed from the moves and the levels of this game so far; each line is a count or a suggestion, not a rule, and a guess is marked as one):"

SYSTEM_D = (C.SYSTEM_C + " A turn may also hold a Notes section, computed by the framework from this game's own moves and levels: which actions changed the board without moving a piece, what is new on this level's first screen, "
            "what earlier levels of this game suggest (a suggestion that this level can cancel) and guesses about what completes a level. A guess is marked as one; none has been tested.")


# ---------------------------------------------------------------- the games: a fixed rule, written before the run
def pick_games(available: Sequence, n: int = BIG_GAMES, base: Sequence[str] = BASE_GAMES) -> List[str]:
    """The games of the run: the four of the earlier tests (those the toolkit offers), then the others in the order of the sha256 of their id (a fixed order that does not depend on how the toolkit lists them), up to `n`."""
    ids = sorted({str(x) for x in available}); chosen = [g for g in base if g in ids]
    rest = sorted((g for g in ids if g not in chosen), key=lambda g: (hashlib.sha256(g.encode()).hexdigest(), g))
    return chosen + rest[:max(0, n - len(chosen))]


# ---------------------------------------------------------------- one game with the notes
class PlayD(C.PlayC):
    """Arm C's game plus the clears: for each level completed, what the clearing move changed on the board (`goalbook.clear_delta`), and the notes for the screen the model is about to see."""
    def __init__(self, env, game: str, moves: int = BIG_MOVES, action_of: Optional[Callable] = None):
        super().__init__(env, game, moves, action_of); self.clears: Dict[int, dict] = {}

    def apply(self, a: dict, forced: bool = False) -> dict:
        before, lv0 = self.screen, self.levels()
        out = super().apply(a, forced)
        if a.get("action") != "RESET" and self.levels() > lv0 and lv0 in self.ledgers and lv0 not in self.clears:
            self.clears[lv0] = dict(delta=GB.clear_delta(self.ledgers[lv0], before), action=R.action_key(a))
        return out

    def _start_ledger(self) -> Ledger:
        """A ledger with no move in it, holding only this level's first screen: it lets the notes say what is new on a level before its first move."""
        L = Ledger(min_law=MIN_LAW); L.bg = background(self.screen); L.first = _key(self.screen); return L

    def goal_text(self) -> List[str]:
        lv = self.levels(); L = self.ledgers.get(lv) or self._start_ledger()
        earlier = [self.ledgers[e] for e in sorted(self.ledgers) if e < lv]; deltas = [self.clears[e]["delta"] for e in sorted(self.clears) if e < lv]
        return GB.read_out(earlier, deltas, L, GOAL_MAX_LINES)

    def goal_view(self) -> str:
        lines = self.goal_text()
        return (GOAL_HEAD + "\n" + "\n".join(lines) + "\n\n") if lines else ""

    def turn(self, arm: str = "D") -> str:
        return self.record() + "\n\n" + self.view() + "\n\n" + self.ledger_view() + "\n\n" + self.goal_view() + R.Play.turn(self, "K")


# ---------------------------------------------------------------- the flags of a move and the replay
def _flags(p: PlayD, text: str, ok: bool, out: Optional[dict], forced: bool, shown: int, goal: int) -> dict:
    f = C._flags(p, text, ok, out, forced, shown); f["goal_lines"] = goal; return f


def replay(rows: list, make_env: Callable, games, moves: int = BIG_MOVES, action_of: Optional[Callable] = None) -> Dict[int, PlayD]:
    """Play the saved rows into fresh games, rebuild the ledgers and the clears move by move and refuse a row whose flags, or whose number of ledger lines or of note lines, differ from the saved ones."""
    plays: Dict[int, PlayD] = {}
    for idx, rec in enumerate(rows, 1):
        gi = rec["g"]
        if gi not in plays: plays[gi] = PlayD(make_env(games[gi]), games[gi], moves, action_of)
        p = plays[gi]
        if rec["game"] != games[gi] or rec["m"] != p.m + 1 or rec["t"] != idx: raise R.ReplayMismatch(f"move {rec['t']}: out of order")
        shown = len(p.ledger_text()); goal = len(p.goal_text()); f = p.forced(); bad = []
        if bool(rec.get("forced")) != (f is not None): bad.append("forced")
        if f is not None:
            if rec["text"] != json.dumps(f): bad.append("forced move")
            a = R.parse_action(rec["text"]); ok = True; out = p.apply(f, forced=True)
        else:
            a = R.parse_action(rec["text"]); ok, _ = p.legal(a); texts = ([rec["first_text"]] if rec.get("reasks") else []) + [x["text"] for x in rec.get("reasks", [])[:-1]]
            if rec.get("asks", 1) != 1 + len(rec.get("reasks", [])) or len(rec.get("reasks", [])) > A_REASKS: bad.append("asks")
            if any(p.legal(R.parse_action(x))[0] for x in texts): bad.append("asked again after a usable move")
            if rec.get("reasks") and len(rec["reasks"]) < A_REASKS and not ok: bad.append("stopped asking before a usable move")
            out = p.apply(a) if ok else None
            if not ok: p.m += 1
        fl = _flags(p, rec["text"], ok, out, f is not None, shown, goal)
        bad += [k for k, v in fl.items() if rec.get(k) != v]
        if bad: raise R.ReplayMismatch(f"move {rec['t']} (game {games[gi]}): flags {sorted(set(bad))} differ from the saved ones")
    return plays


def run(llm, make_env: Callable, path: str, games, moves: int = BIG_MOVES, meta=None, max_tokens: Optional[int] = None, progress=None, action_of: Optional[Callable] = None) -> dict:
    """Arm D. One JSON line per move after a header; the same call again carries on. `stopped`: 'token_cap' or None."""
    meta = dict(meta or {}, arm="D", games=list(games), moves=moves, game_seed=GAME_SEED, policy="record+check+explorer+ledger+notes", reasks=A_REASKS, stall_moves=B.STALL_MOVES, seen_max=B.SEEN_MAX, obj_max=B.OBJ_MAX,
                min_law=MIN_LAW, step_min_tries=C.STEP_MIN_TRIES, ledger_max_lines=C.LEDGER_MAX_LINES, goal_max_lines=GOAL_MAX_LINES)
    rows = load_rows(path, meta); stopped = None; spent = GL.tokens_sent(rows); plays = replay(rows, make_env, games, moves, action_of)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w") as f: f.write(json.dumps({"_meta": _roundtrip(meta)}) + "\n")
    with open(path, "a") as f:
        for gi, game in enumerate(games):
            if R._game_done(rows, gi, moves): continue
            if gi not in plays: plays[gi] = PlayD(make_env(game), game, moves, action_of)
            p = plays[gi]
            while p.m < moves and not p.done:
                if max_tokens is not None and spent >= max_tokens: stopped = "token_cap"; break
                shown = len(p.ledger_text()); goal = len(p.goal_text()); fm = p.forced(); m_before = p.m
                if fm is not None:
                    text = json.dumps(fm); out = p.apply(fm, forced=True)
                    rec = dict(t=len(rows) + 1, g=gi, game=game, m=m_before + 1, text=text, seconds=0.0, tokens=0, think=0, prompt_tokens=0, cache_tokens=0, first_token=None, from_disk=True, finish="FRAMEWORK", **_flags(p, text, True, out, True, shown, goal))
                else:
                    msgs = [{"role": "system", "content": SYSTEM_D}, {"role": "user", "content": p.turn("D")}]
                    text, ntok, sec = llm.generate(msgs, constrained=True); secs = float(sec); first = GL._att(llm, text, ntok, sec); extra = []
                    a = R.parse_action(text); ok, why = p.legal(a)
                    while not ok and len(extra) < A_REASKS:
                        text, ntok, sec = llm.generate(R.reask_messages(msgs, text, why), constrained=True); secs += float(sec); extra.append(GL._att(llm, text, ntok, sec)); a = R.parse_action(text); ok, why = p.legal(a)
                    out = p.apply(a) if ok else None
                    if not ok: p.m += 1
                    top = {k: first[k] for k in ("tokens", "think", "prompt_tokens", "cache_tokens", "first_token", "from_disk", "finish")}
                    rec = dict(t=len(rows) + 1, g=gi, game=game, m=m_before + 1, text=text, seconds=round(secs, 4), **top, **_flags(p, text, ok, out, False, shown, goal))
                    if extra: rec.update(asks=1 + len(extra), first_text=first["text"], reasks=extra)
                rec = _roundtrip(rec); f.write(json.dumps(rec) + "\n"); f.flush(); os.fsync(f.fileno()); rows.append(rec); spent += GL.billed(rec)
                if progress: progress(rec["t"], len(games) * moves, rec)
            if stopped: break
    return dict(rows=rows, summary=summarize(rows), meta=_roundtrip(meta), stopped=stopped)


# ---------------------------------------------------------------- summaries and the reading rules
load_arm, model_rows, moves_per_hour = B.load_arm, B.model_rows, B.moves_per_hour


def levels_by_game(rows: list, n_games: int) -> List[int]:
    """The most levels completed in each game (0 for a game with no row), in the order of the games."""
    out = [0] * n_games
    for r in rows: out[r["g"]] = max(out[r["g"]], r.get("levels") or 0)
    return out


def summarize(rows: list) -> dict:
    s = C.summarize(rows); mr = model_rows(rows); goal = [r["goal_lines"] for r in mr if "goal_lines" in r]
    s["goal_shown"] = (sum(1 for x in goal if x > 0) / len(goal)) if goal else None; s["goal_lines_mean"] = (sum(goal) / len(goal)) if goal else None
    s["games_with_level"] = sum(1 for b in s["games"] if b["levels"] >= 1); s["games_with_two"] = sum(1 for b in s["games"] if b["levels"] >= 2)
    return s


def beats(x: Sequence[int], y: Sequence[int]) -> int: return sum(1 for a, b in zip(x, y) if a > b)


G_FINISH_K = "the plain model (arm K) played every game to 60 moves or to a win, with every move on file, and the replay of its saved moves into fresh copies of the games reproduces every saved flag"
G_FINISH_C = "the arm with the ledger read-out (arm C) played every game to 60 moves or to a win, with every move on file, and the replay of its saved moves rebuilds the ledger and reproduces every saved flag, every framework move, every re-ask and the number of ledger lines"
G_FINISH_D = "the arm with the read-out and the notes (arm D) played every game to 60 moves or to a win, with every move on file, and the replay of its saved moves rebuilds the ledgers and the clears and reproduces every saved flag, every framework move, every re-ask, the number of ledger lines and the number of note lines"
G_FORCED = "at most 40 percent of the moves of arm C, and of arm D, were played by the framework itself (otherwise the explorer plays instead of the model)"
G_UNUSABLE = "at most 2 percent of the model's moves in arm C, and in arm D, ended with a reply that could not be used (after the re-asks)"
G_NOTES = "the notes were there: at least 25 percent of the model's moves in arm D were shown at least one line of notes"
G_C_K = "levels: arm C completed more levels than arm K, over the eight games"
G_D_K = "levels: arm D completed more levels than arm K, over the eight games"
G_D_C_GE = "levels: arm D completed at least as many levels as arm C"
G_D_C = "levels: arm D completed more levels than arm C"
G_WINS_K = "game by game: the games in which arm D completed more levels than arm K are more than the games in which arm K completed more than arm D"
G_WINS_C = "game by game: the games in which arm D completed more levels than arm C are more than the games in which arm C completed more than arm D"
G_BREADTH = "breadth: arm D completed a level in at least as many games as arm K and as arm C"
G_SECOND = "arm D completed a second level (levels completed 2 or more) in at least one game"
G_EFFECTIVE = "wasted moves: the share of arm D's moves that changed the screen is at least 3 points above that of arm K"
G_SCORE = "progress per token: the screens never seen before that arm D reached, per 1,000 computed tokens, are at least 1.5 times those of arm K"
G_TOKENS = "tokens sent: arm D sent at most one third of the tokens arm K sent (the kept chat grows with the square of the moves)"
G_THINK = "thinking: the mean thinking tokens per call of arm D are no more than those of arm K"
G_LINES = [("guard", G_FINISH_K), ("guard", G_FINISH_C), ("guard", G_FINISH_D), ("instrument", G_FORCED), ("instrument", G_UNUSABLE), ("instrument", G_NOTES), ("claim", G_C_K), ("claim", G_D_K), ("claim", G_D_C_GE), ("claim", G_D_C),
           ("claim", G_WINS_K), ("claim", G_WINS_C), ("claim", G_BREADTH), ("claim", G_SECOND), ("claim", G_EFFECTIVE), ("claim", G_SCORE), ("claim", G_TOKENS), ("claim", G_THINK)]
G_HEAD = "headline: lines 8 and 11 both read yes (the same model, with the framework's read-outs and notes, completes more levels than the plain model, over the eight games and game by game)"
HEADLINE = "THE FRAMEWORK'S READ-OUTS MAKE THE SAME MODEL COMPLETE MORE LEVELS THAN THE PLAIN MODEL"
_gt, _le = GL._gt, GL._le


def _complete(res: Optional[dict], games, moves) -> bool: return B._complete(res, games, moves)


def verdict(k: Optional[dict], c: Optional[dict], d: Optional[dict], replays: Optional[dict] = None, games=None, moves: int = BIG_MOVES) -> List[Tuple[int, str, str, Optional[bool]]]:
    games = list(games) if games is not None else []; replays = replays or {}; n = len(games); out: List[Optional[bool]] = []
    out.append(None if not k else bool(_complete(k, games, moves) and replays.get("K") is True))                                                                                   # 1
    out.append(None if not c else bool(_complete(c, games, moves) and replays.get("C") is True))                                                                                   # 2
    out.append(None if not d else bool(_complete(d, games, moves) and replays.get("D") is True))                                                                                   # 3
    sk = summarize(k["rows"]) if k else None; sc = summarize(c["rows"]) if c else None; sd = summarize(d["rows"]) if d else None
    lk = levels_by_game(k["rows"], n) if k else None; lc = levels_by_game(c["rows"], n) if c else None; ld = levels_by_game(d["rows"], n) if d else None
    out.append(None if not (sc and sd) else bool(_le(sc["forced"] / max(1, sc["moves"]), 0.40) and _le(sd["forced"] / max(1, sd["moves"]), 0.40)))                                  # 4
    out.append(None if not (sc and sd and model_rows(c["rows"]) and model_rows(d["rows"])) else bool(_le(sc["unusable"] / len(model_rows(c["rows"])), 0.02) and _le(sd["unusable"] / len(model_rows(d["rows"])), 0.02)))   # 5
    out.append(None if not sd or sd["goal_shown"] is None else _gt(sd["goal_shown"], 0.25 - 1e-9))                                                                                 # 6
    out.append(_gt(sum(lc), sum(lk) + 0.5) if lc is not None and lk is not None else None)                                                                                               # 7
    out.append(_gt(sum(ld), sum(lk) + 0.5) if ld is not None and lk is not None else None)                                                                                               # 8
    out.append(_gt(sum(ld), sum(lc) - 0.5) if ld is not None and lc is not None else None)                                                                                         # 9
    out.append(_gt(sum(ld), sum(lc) + 0.5) if ld is not None and lc is not None else None)                                                                                               # 10
    out.append(_gt(beats(ld, lk), beats(lk, ld) + 0.5) if ld is not None and lk is not None else None)                                                                                   # 11
    out.append(_gt(beats(ld, lc), beats(lc, ld) + 0.5) if ld is not None and lc is not None else None)                                                                                   # 12
    out.append(bool(sd["games_with_level"] >= sk["games_with_level"] and sd["games_with_level"] >= sc["games_with_level"]) if sd and sk and sc else None)                           # 13
    out.append(None if not sd else sd["games_with_two"] >= 1)                                                                                                                      # 14
    out.append(_gt(sd["effective"], sk["effective"] + 0.03) if sd and sk else None)                                                                                                # 15
    out.append(_gt(sd["score"], 1.5 * sk["score"]) if sd and sk and sd["score"] is not None and sk["score"] else None)                                                             # 16
    out.append(_le(sd["sent"], sk["sent"] / 3) if sd and sk else None)                                                                                                             # 17
    out.append(_le(sd["think_per_call"], sk["think_per_call"]) if sd and sk and sd["think_per_call"] is not None and sk["think_per_call"] is not None else None)                  # 18
    assert len(out) == len(G_LINES)
    return [(i + 1, G_LINES[i][0], G_LINES[i][1], out[i]) for i in range(len(G_LINES))]


def claims_allowed(checks) -> dict:
    c = {i: v for i, _, _, v in checks}
    return dict(base=bool(c.get(1) and c.get(2) and c.get(3) and c.get(5)), tokens_computed=bool(c.get(1) and c.get(2) and c.get(3)))


def headline(checks) -> str:
    c = {i: v for i, _, _, v in checks}
    if c.get(8) is None or c.get(11) is None: return "NOT READ: an arm is missing"
    if c[8] and c[11]: return HEADLINE
    return "MIXED: line 8 " + ("yes" if c[8] else "NO") + ", line 11 " + ("yes" if c[11] else "NO")


def pilot_report(k_rows: list, c_rows: list, d_rows: list, price_in=None, price_out=None, price_cached=None, games: int = BIG_GAMES, moves: int = BIG_MOVES) -> dict:
    """What a few real moves of each arm in one game say about the whole run (8 games x 60 moves): the kept chat's growth per move and its total prompt tokens, the speed and the projected cost of each arm and of the three together."""
    live = [r for r in c_rows if not r.get("forced")] or c_rows
    rk = R.pilot_report(k_rows, live, price_in, price_out, price_cached, games=games, moves=moves)
    out = {k: v for k, v in rk.items() if k.startswith("k_") or k in ("cost_k", "cost_k_nocache")}
    for tag, rows in (("c", c_rows), ("d", d_rows)):
        rep = C.pilot_report(rows, price_in, price_out, price_cached, games=games, moves=moves)
        out.update({k.replace("c_", tag + "_", 1) if k.startswith("c_") else k.replace("cost_c", "cost_" + tag): v for k, v in rep.items()})
    s = summarize(d_rows); out["d_notes_shown_share"] = s["goal_shown"]; out["d_notes_lines_mean"] = s["goal_lines_mean"]
    costs = [out.get(x) for x in ("cost_k", "cost_c", "cost_d")]
    out["cost_total"] = sum(costs) if all(isinstance(x, (int, float)) for x in costs) else None
    return out


def print_stats(title: str, rows: list):
    C.print_stats(title, rows); s = summarize(rows)
    if s["goal_shown"] is not None: print(f"   notes shown in {GL._f(s['goal_shown'] * 100)}% of the model's moves, {GL._f(s['goal_lines_mean'], 1)} lines on average")
    print(f"   games with a level: {s['games_with_level']} | games with two levels: {s['games_with_two']}")


def print_verdict(checks, title="THE BIGGER ARC-AGI-3 TEST: 8 GAMES, 60 MOVES, THREE ARMS (reading rules fixed before the run)"):
    print(f"\n-- {title} --")
    for i, tag, text, c in checks: print(f"  {'not run' if c is None else ('yes' if c else 'NO '):8} {i:2d}. [{tag}] {text}")
    print("\n  headline:", headline(checks)); a = claims_allowed(checks)
    print("  rule on the page:", "lines 1, 2, 3 and 5 read yes: the figures of the arms are claimed as read." if a["base"] else "a guard or instrument line is not yes: no claim is made until the cause is found.")


def print_levels(games: Sequence[str], **arms) -> None:
    """Levels completed in each game, one column per arm."""
    names = list(arms); print(f"{'game':>16} " + " ".join(f"{k:>4}" for k in names))
    cols = {k: levels_by_game(v["rows"], len(games)) if v else [None] * len(games) for k, v in arms.items()}
    for i, g in enumerate(games): print(f"{g:>16} " + " ".join(f"{('-' if cols[k][i] is None else cols[k][i]):>4}" for k in names))
    print(f"{'total':>16} " + " ".join(f"{(sum(cols[k]) if arms[k] else '-'):>4}" for k in names))


print_games = R.print_games
