"""The ARC-AGI-3 test with the ledger's read-out in the prompt (docs/GEMINI_ARC_C_TEST.md, test 160): arm C, the same model and the same four games as Tests 24 and 25.

Arm B (gemini_arc_b.py) is arm A (the exact Record and the check) plus the object view, the untried list and the conductor. Arm C is arm B plus ONE more section in each turn: the LEDGER read-out, written by the host from the
transition ledger (axi/engine/ledger.py, Update 92 and following; from the structure of Rajnish Choubey's RFC-001, the reading in game terms is for Sir to confirm), from the moves of THIS run played so far:
  - for each plain action, the step the same-coloured object moved by, and in how many of its tries (exact counts, never a rule);
  - the strips that get smaller or larger in most moves (a bar or counter, with its area);
  - the colours that have never changed anywhere in the moves of this level;
  - what the last move did, object by object (moved, changed colour).
It tells what happened, not why, and nothing is taken from any other run: the read-out of move m is made from the moves before m only (the replay checks it: each saved row holds `ledger_lines`, the number of lines the model was shown, and the replay
rebuilds the ledger move by move and refuses a row whose number differs). A new level is a new ledger (a new world), as in ledger_replay.py.

Nothing of Test 24, of Test 25, of `gemini_arc.py`, of `gemini_arc_b.py` or of the framework's core is changed: this module imports them and edits none. Arm B and arm A are the rows saved by the earlier tests (replayed, never run again); only arm C costs anything."""
from __future__ import annotations
import json, os
from typing import Callable, Dict, List, Optional, Tuple

from axi.engine.ledger import Ledger
from axi.experiments import gemini_arc as R
from axi.experiments import gemini_arc_b as B
from axi.experiments import gemini_long as GL
from axi.experiments.resumable import load_rows, _roundtrip

MODEL, GAMES, MOVES, GAME_SEED, A_REASKS = R.MODEL, R.GAMES, R.MOVES, R.GAME_SEED, R.A_REASKS
C_TOKEN_CAP = 8_000_000
PILOT_GAME, PILOT_MOVES = R.PILOT_GAME, 12
MIN_LAW, STEP_MIN_TRIES, LEDGER_MAX_LINES, RIGID_MAX, LAST_MAX = 3, 2, 12, 6, 4
LEDGER_HEAD = "Ledger of this level so far (framework, exact counts from this game's own screens and the moves played so far; it says what happened, not why; x grows to the right, y downwards):"

SYSTEM_C = (B.SYSTEM_B + " Each turn also holds a Ledger section, computed by the framework from the moves played so far in this level: what each action did, bars that get smaller or larger, colours that never changed, and what the last move did. "
            "They are counts of what happened, not rules: an action that did something in 15 of 21 tries did nothing in the others.")


class PlayC(B.PlayB):
    """Arm B's game with a ledger per level, fed by every move that is played (the model's and the framework's)."""
    def __init__(self, env, game: str, moves: int = MOVES, action_of: Optional[Callable] = None):
        super().__init__(env, game, moves, action_of); self.ledgers: Dict[int, Ledger] = {}

    def apply(self, a: dict, forced: bool = False) -> dict:
        before, lv0 = self.screen, self.levels()
        out = super().apply(a, forced)
        if a.get("action") != "RESET" and self.levels() == lv0: self.ledgers.setdefault(lv0, Ledger(min_law=MIN_LAW)).record(R.action_key(a), before, self.screen)      # a level change and a RESET are not a transition
        return out

    def ledger_text(self) -> List[str]:
        """The lines of the read-out for the screen the model is about to see. Empty before the first move of a level."""
        L = self.ledgers.get(self.levels())
        if L is None or not L.rows: return []
        lines: List[str] = []
        for act, (sym, step, n, tot) in sorted(L.dominant().items(), key=lambda kv: str(kv[0])):
            if "@" in str(act) or tot < STEP_MIN_TRIES: continue
            lines.append(f"- {act}: the object of colour {sym} moved by ({step[0]},{step[1]}) in {n} of its {tot} tries.")
        ticks = L.ticks(); tsyms = {t["symbol"] for t in ticks}
        for t in ticks[:3]:
            b = t["box"]; lines.append(f"- colour {t['symbol']} gets {'larger' if t['delta'] > 0 else 'smaller'} by {abs(t['delta'])} cell(s) in {t['moves']} of the {t['of']} moves that changed the screen (area x {b[0]}-{b[2]}, y {b[1]}-{b[3]}): probably a bar or counter that fills or is used up.")
        rigid = [s for s in L.rigid_symbols() if s not in tsyms] if len(L.rows) >= MIN_LAW else []                                  # a count is said only after MIN_LAW moves, as the ledger's own laws are
        if rigid: lines.append(f"- colours that have not changed anywhere in the {len(L.rows)} moves of this level: " + ", ".join(str(s) for s in rigid[:RIGID_MAX]) + ".")
        last = L.rows[-1]; said = []
        for o in last.ops:
            if o[0] == "move": said.append(f"colour {o[1]} moved by ({o[3][0]},{o[3][1]})")
            elif o[0] == "recolour" and o[1] not in tsyms and o[2] not in tsyms: said.append(f"{o[4]} cell(s) changed colour from {o[1]} to {o[2]}")
        if said: lines.append("- last move: " + "; ".join(said[:LAST_MAX]) + ".")
        elif not last.changed(): lines.append("- last move: nothing changed.")
        return lines[:LEDGER_MAX_LINES]

    def ledger_view(self) -> str:
        lines = self.ledger_text()
        return LEDGER_HEAD + "\n" + "\n".join(lines) if lines else LEDGER_HEAD + "\n(no move played yet in this level.)"

    def turn(self, arm: str = "C") -> str:
        return self.record() + "\n\n" + self.view() + "\n\n" + self.ledger_view() + "\n\n" + R.Play.turn(self, "K")


# ---------------------------------------------------------------- the flags of a move and the replay
def _flags(p: PlayC, text: str, ok: bool, out: Optional[dict], forced: bool, shown: int) -> dict:
    f = B._flags(p, text, ok, out, forced); f["ledger_lines"] = shown; return f


def replay(rows: list, make_env: Callable, games=GAMES, moves: int = MOVES, action_of: Optional[Callable] = None) -> Dict[int, PlayC]:
    """Play the saved rows into fresh games, rebuild the ledger move by move and refuse a row whose flags, or whose number of ledger lines, differ from the saved ones."""
    plays: Dict[int, PlayC] = {}
    for idx, rec in enumerate(rows, 1):
        gi = rec["g"]
        if gi not in plays: plays[gi] = PlayC(make_env(games[gi]), games[gi], moves, action_of)
        p = plays[gi]
        if rec["game"] != games[gi] or rec["m"] != p.m + 1 or rec["t"] != idx: raise R.ReplayMismatch(f"move {rec['t']}: out of order")
        shown = len(p.ledger_text()); f = p.forced(); bad = []
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
        fl = _flags(p, rec["text"], ok, out, f is not None, shown)
        bad += [k for k, v in fl.items() if rec.get(k) != v]
        if bad: raise R.ReplayMismatch(f"move {rec['t']} (game {games[gi]}): flags {sorted(set(bad))} differ from the saved ones")
    return plays


def run(llm, make_env: Callable, path: str, games=GAMES, moves: int = MOVES, meta=None, max_tokens: Optional[int] = None, progress=None, action_of: Optional[Callable] = None) -> dict:
    """Arm C. One JSON line per move after a header; the same call again carries on. `stopped`: 'token_cap' or None."""
    meta = dict(meta or {}, arm="C", games=list(games), moves=moves, game_seed=GAME_SEED, policy="record+check+explorer+ledger", reasks=A_REASKS, stall_moves=B.STALL_MOVES, seen_max=B.SEEN_MAX, obj_max=B.OBJ_MAX,
                min_law=MIN_LAW, step_min_tries=STEP_MIN_TRIES, ledger_max_lines=LEDGER_MAX_LINES)
    rows = load_rows(path, meta); stopped = None; spent = GL.tokens_sent(rows); plays = replay(rows, make_env, games, moves, action_of)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w") as f: f.write(json.dumps({"_meta": _roundtrip(meta)}) + "\n")
    with open(path, "a") as f:
        for gi, game in enumerate(games):
            if R._game_done(rows, gi, moves): continue
            if gi not in plays: plays[gi] = PlayC(make_env(game), game, moves, action_of)
            p = plays[gi]
            while p.m < moves and not p.done:
                if max_tokens is not None and spent >= max_tokens: stopped = "token_cap"; break
                shown = len(p.ledger_text()); fm = p.forced(); m_before = p.m
                if fm is not None:
                    text = json.dumps(fm); out = p.apply(fm, forced=True)
                    rec = dict(t=len(rows) + 1, g=gi, game=game, m=m_before + 1, text=text, seconds=0.0, tokens=0, think=0, prompt_tokens=0, cache_tokens=0, first_token=None, from_disk=True, finish="FRAMEWORK", **_flags(p, text, True, out, True, shown))
                else:
                    msgs = [{"role": "system", "content": SYSTEM_C}, {"role": "user", "content": p.turn("C")}]
                    text, ntok, sec = llm.generate(msgs, constrained=True); secs = float(sec); first = GL._att(llm, text, ntok, sec); extra = []
                    a = R.parse_action(text); ok, why = p.legal(a)
                    while not ok and len(extra) < A_REASKS:
                        text, ntok, sec = llm.generate(R.reask_messages(msgs, text, why), constrained=True); secs += float(sec); extra.append(GL._att(llm, text, ntok, sec)); a = R.parse_action(text); ok, why = p.legal(a)
                    out = p.apply(a) if ok else None
                    if not ok: p.m += 1
                    top = {k: first[k] for k in ("tokens", "think", "prompt_tokens", "cache_tokens", "first_token", "from_disk", "finish")}
                    rec = dict(t=len(rows) + 1, g=gi, game=game, m=m_before + 1, text=text, seconds=round(secs, 4), **top, **_flags(p, text, ok, out, False, shown))
                    if extra: rec.update(asks=1 + len(extra), first_text=first["text"], reasks=extra)
                rec = _roundtrip(rec); f.write(json.dumps(rec) + "\n"); f.flush(); os.fsync(f.fileno()); rows.append(rec); spent += GL.billed(rec)
                if progress: progress(rec["t"], len(games) * moves, rec)
            if stopped: break
    return dict(rows=rows, summary=summarize(rows), meta=_roundtrip(meta), stopped=stopped)


# ---------------------------------------------------------------- summaries and the reading rules
load_arm, model_rows, moves_per_hour = B.load_arm, B.model_rows, B.moves_per_hour


def summarize(rows: list) -> dict:
    s = B.summarize(rows); mr = model_rows(rows); shown = [r["ledger_lines"] for r in mr if "ledger_lines" in r]                     # rows of arms A, B and K have none: nothing to say about them
    s["ledger_shown"] = (sum(1 for x in shown if x > 0) / len(shown)) if shown else None; s["ledger_lines_mean"] = (sum(shown) / len(shown)) if shown else None
    return s


G_FINISH_C = "the arm with the ledger read-out (arm C) played every game to 60 moves or to a win, with every move on file, and the replay of its saved moves into fresh copies of the games rebuilds the ledger and reproduces every saved flag, every framework move, every re-ask and the number of ledger lines each move was shown"
G_FINISH_B = "the arm B of Test 25 played every game to 60 moves or to a win, with every move on file, and the replay of its saved moves reproduces every saved flag"
G_FORCED = "at most 40 percent of the moves of arm C were played by the framework itself (otherwise the explorer plays instead of the model)"
G_UNUSABLE = "at most 2 percent of the model's moves in arm C ended with a reply that could not be used (after the re-asks)"
G_SHOWN = "the read-out was there: at least 80 percent of the model's moves in arm C were shown at least one ledger line"
G_EFFECTIVE = "wasted moves: the share of arm C's moves that changed the screen is at least that of arm B"
G_REPEAT = "memory of what did nothing: arm C repeated a move that had already done nothing from the same screen no more often than arm B did"
G_SCORE = "progress per token: the screens never seen before that arm C reached, per 1,000 computed tokens, are at least 1.1 times those of arm B"
G_TOTAL = "reach: arm C reached at least as many screens never seen before as arm B, over the same number of moves"
G_LEVELS = "levels: arm C completed at least as many levels as arm B, over the games"
G_C_LEVEL = "arm C completed at least one level in at least one game"
G_MORE = "arm C completed more levels than arm B"
G_THINK = "thinking: the mean thinking tokens per call of arm C are at most 1.1 times those of arm B"
G_TOKENS = "tokens (the stand-in for energy): arm C computes at most 1.2 times the tokens per effective move that arm B computes (the read-out makes each prompt longer)"
G_A_LEVELS = "against arm A of Test 24: arm C completed at least as many levels as arm A"
G_K_LEVELS = "against the kept chat: arm C completed at least as many levels as arm K (Test 24)"
G_LINES = [("guard", G_FINISH_C), ("guard", G_FINISH_B), ("instrument", G_FORCED), ("instrument", G_UNUSABLE), ("instrument", G_SHOWN), ("claim", G_EFFECTIVE), ("claim", G_REPEAT), ("claim", G_SCORE), ("claim", G_TOTAL), ("claim", G_LEVELS),
           ("claim", G_C_LEVEL), ("claim", G_MORE), ("claim", G_THINK), ("claim", G_TOKENS), ("claim", G_A_LEVELS), ("claim", G_K_LEVELS)]
G_HEAD = "headline: lines 6 and 8 both read yes (with the read-out the model wastes no more moves and reaches more new screens per token)"
HEADLINE = "THE LEDGER READ-OUT WASTES NO MORE MOVES AND REACHES MORE NEW SCREENS PER TOKEN"
_gt, _le = GL._gt, GL._le


def _complete(res: Optional[dict], games, moves) -> bool: return B._complete(res, games, moves)


def verdict(c: Optional[dict], b: Optional[dict], a: Optional[dict] = None, k: Optional[dict] = None, replays: Optional[dict] = None, games=GAMES, moves: int = MOVES) -> List[Tuple[int, str, str, Optional[bool]]]:
    replays = replays or {}; out: List[Optional[bool]] = []
    out.append(None if not c else bool(_complete(c, games, moves) and replays.get("C") is True))                                                                                   # 1
    out.append(None if not b else bool(_complete(b, games, moves) and replays.get("B") is True))                                                                                   # 2
    sc = summarize(c["rows"]) if c else None; sb = summarize(b["rows"]) if b else None; sa = summarize(a["rows"]) if a else None; sk = summarize(k["rows"]) if k else None
    out.append(None if not sc else _le(sc["forced"] / max(1, sc["moves"]), 0.40))                                                                                                  # 3
    out.append(None if not sc or not model_rows(c["rows"]) else _le(sc["unusable"] / len(model_rows(c["rows"])), 0.02))                                                           # 4
    out.append(None if not sc or sc["ledger_shown"] is None else _gt(sc["ledger_shown"], 0.80 - 1e-9))                                                                             # 5
    both = bool(sc and sb and sc["moves"] == sb["moves"] and sc["moves"] > 0)
    out.append(_gt(sc["effective"], sb["effective"] - 1e-9) if both else None)                                                                                                     # 6
    out.append(_le(sc["repeat_noop"], sb["repeat_noop"]) if both else None)                                                                                                        # 7
    out.append(_gt(sc["score"], 1.1 * sb["score"]) if both and sc["score"] is not None and sb["score"] else None)                                                                  # 8
    out.append(_gt(sc["discovered"], sb["discovered"] - 0.5) if both else None)                                                                                                    # 9
    out.append(_gt(sc["levels"], sb["levels"] - 0.5) if both else None)                                                                                                            # 10
    out.append(None if not sc else sc["levels"] >= 1)                                                                                                                              # 11
    out.append(_gt(sc["levels"], sb["levels"] + 0.5) if both else None)                                                                                                            # 12
    out.append(_le(sc["think_per_call"], 1.1 * sb["think_per_call"]) if both and sc["think_per_call"] is not None and sb["think_per_call"] is not None else None)                 # 13
    eff = lambda s: s["computed"] / max(1, round(s["effective"] * s["moves"]))
    out.append(_le(eff(sc), 1.2 * eff(sb)) if both else None)                                                                                                                      # 14
    out.append(_gt(sc["levels"], sa["levels"] - 0.5) if sc and sa else None)                                                                                                       # 15
    out.append(_gt(sc["levels"], sk["levels"] - 0.5) if sc and sk else None)                                                                                                       # 16
    assert len(out) == len(G_LINES)
    return [(i + 1, G_LINES[i][0], G_LINES[i][1], out[i]) for i in range(len(G_LINES))]


def claims_allowed(checks) -> dict:
    c = {i: v for i, _, _, v in checks}
    return dict(base=bool(c.get(1) and c.get(2) and c.get(4) and c.get(5)), tokens_computed=bool(c.get(1) and c.get(2)))


def headline(checks) -> str:
    c = {i: v for i, _, _, v in checks}
    if c.get(6) is None or c.get(8) is None: return "NOT READ: arm C or arm B is missing"
    if c[6] and c[8]: return HEADLINE
    return "MIXED: line 6 " + ("yes" if c[6] else "NO") + ", line 8 " + ("yes" if c[8] else "NO")


def pilot_report(c_rows: list, price_in=None, price_out=None, price_cached=None, games: int = len(GAMES), moves: int = MOVES) -> dict:
    """What a few real moves of arm C in one game say about the whole run."""
    rep = B.pilot_report(c_rows, price_in, price_out, price_cached, games, moves); s = summarize(c_rows)
    out = {k.replace("b_", "c_", 1) if k.startswith("b_") else k.replace("cost_b", "cost_c"): v for k, v in rep.items()}; out["c_ledger_lines_mean"] = s["ledger_lines_mean"]; return out


def print_stats(title: str, rows: list):
    B.print_stats(title, rows); s = summarize(rows)
    if s["ledger_shown"] is not None: print(f"   ledger read-out shown in {GL._f(s['ledger_shown'] * 100)}% of the model's moves, {GL._f(s['ledger_lines_mean'], 1)} lines on average")


def print_verdict(checks, title="ARC-AGI-3 TEST WITH THE LEDGER READ-OUT (reading rules fixed before the run)"):
    print(f"\n-- {title} --")
    for i, tag, text, c in checks: print(f"  {'not run' if c is None else ('yes' if c else 'NO '):8} {i:2d}. [{tag}] {text}")
    print("\n  headline:", headline(checks)); a = claims_allowed(checks)
    print("  rule on the page:", "lines 1, 2, 4 and 5 read yes: the figures of the arms are claimed as read." if a["base"] else "a guard or instrument line is not yes: no claim is made until the cause is found.")


print_games = R.print_games
