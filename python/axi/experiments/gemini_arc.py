"""The ARC-AGI-3 test on a hosted model (docs/GEMINI_ARC_TEST.md, test 24): Gemini 3.8 Flash playing the public ARC-AGI-3 games, alone and with AXI.

ARC-AGI-3 games are small interactive puzzles whose rules are not told to the player. The screen is a 64 x 64 grid of colour numbers 0 to 15; a move is one of the actions the game offers (ACTION1 to ACTION7; ACTION6 is a click
with x and y from 0 to 63) or RESET. The toolkit (`arc-agi`, `arcengine`) runs the game on the machine in under a millisecond a step, so everything the test costs is the model.

Two arms, the same model, the same games, the same number of moves:
  K  the kept chat: one chat per game; every screen and every reply stays in the chat and is sent again at every move. What a firm gets when it just keeps talking to the model.
  A  with AXI: nothing is kept in the chat. Every move the model gets the current screen and a RECORD that the framework keeps exactly from the engine's frames (what each action did, the last moves, which actions did nothing
     from this very screen, how often this screen was seen, the levels). The record is computed, never asked of the model. A move goes through a check before it reaches the game (the action must be one the game offers, a
     click must be inside the grid), with up to two re-asks; a move that still fails is not played.
This is a test of what the framework's exact record and check do to the cost and the steadiness of a model on a hard game. It is NOT a test of how well the model works out new games, and it is not an entry on the ARC-AGI-3
leaderboard (its rules, scorecards and hidden games are not used).
Measured per call as in Tests 22 and 23 (prompt tokens, tokens served from the cache, visible and thinking tokens, seconds, seconds to the first token), and per move: legal, effective (legal and the screen changed), repeated
no-op (a move that already did nothing from the same screen), revisit (a screen seen before), levels. It imports `gemini_long.py` and `api_llm.py` and edits none of them; it imports no ARC package: the notebook hands it the
toolkit's `make` (so that the tests can use a stand-in game)."""
from __future__ import annotations
import hashlib, json, os
from typing import Callable, Dict, List, Optional, Tuple

from axi.experiments import api_llm as API
from axi.experiments import gemini_long as GL
from axi.experiments.resumable import load_rows, _roundtrip

# ---------------------------------------------------------------- the settings written on the page
MODEL = GL.MODEL
GAMES = ("ls20-9607627b", "g50t-5849a774", "ka59-38d34dbb", "sp80-589a99af")        # all four were played by the probe notebook on 22 Oct
MOVES = 60                                                                              # moves (calls of the model, one per move) per game and arm
GAME_SEED = 0
A_REASKS = 2
K_TOKEN_CAP, A_TOKEN_CAP = 120_000_000, 6_000_000
PILOT_GAME, PILOT_MOVES = GAMES[0], 12
MAX_OUT = GL.MAX_OUT
BANDS = (0, 20_000, 50_000, 100_000, 150_000, 200_000, 300_000, 1_000_001)
HEX = "0123456789abcdef"

SYSTEM = (
    "You are playing an unknown puzzle game. You are not told its rules; find them by playing. The screen is a grid of 64 rows of 64 characters, one character per cell, each a colour number from 0 to f (hex). "
    "Each turn you pick ONE action: ACTION1, ACTION2, ACTION3, ACTION4, ACTION5, ACTION7 (not all are offered in every game or every screen: the turn tells you which are offered now), ACTION6 (a click on a cell: it needs x and y, each from 0 to 63, "
    "x across, y down), or RESET (restart the level; the only move that works when the state is GAME_OVER). The goal is to complete levels; the turn tells you how many are completed and how many there are.\n"
    "Reply with ONE JSON object and nothing else: {\"action\": \"ACTION1\"} or, for a click, {\"action\": \"ACTION6\", \"x\": 10, \"y\": 20}.")
SYSTEM_K = SYSTEM + "\nThe earlier turns of this game stay in this chat."
SYSTEM_A = SYSTEM + "\nThe earlier turns are not in this chat. Instead each turn starts with a Record kept exactly by the framework from the game's own screens: use it."


# ---------------------------------------------------------------- the screen, the action, the check
def rows_of(frame) -> Tuple[str, ...]:
    """The 64 rows of a frame (an ndarray or nested lists of ints 0 to 15) as strings of hex digits."""
    return tuple("".join(HEX[int(v) & 15] for v in row) for row in (frame.tolist() if hasattr(frame, "tolist") else frame))


def screen_key(rows: Tuple[str, ...]) -> str: return hashlib.sha256("\n".join(rows).encode()).hexdigest()[:16]


def cells_changed(a: Tuple[str, ...], b: Tuple[str, ...]) -> int:
    return sum(1 for ra, rb in zip(a, b) for x, y in zip(ra, rb) if x != y) + 64 * max(0, abs(len(a) - len(b)))


def parse_action(text: str) -> Optional[dict]:
    """One JSON object {'action': NAME} or {'action': 'ACTION6', 'x': int, 'y': int} (a code fence around it is removed); anything else is None."""
    s = (text or "").strip()
    if s.startswith("```"):
        lines = s.split("\n")[1:]
        if lines and lines[-1].strip().startswith("```"): lines = lines[:-1]
        s = "\n".join(lines).strip()
    try: obj = json.loads(s)
    except (ValueError, TypeError): return None
    if not isinstance(obj, dict) or not isinstance(obj.get("action"), str): return None
    out = dict(action=obj["action"].strip().upper())
    if "x" in obj or "y" in obj: out.update(x=obj.get("x"), y=obj.get("y"))
    return out


def action_key(a: dict) -> str: return f"ACTION6@{a['x']},{a['y']}" if a["action"] == "ACTION6" else a["action"]


class Play:
    """One game: the engine's environment, the exact record kept from its frames, and the move counter. `env` follows the toolkit's wrapper: `observation_space`, `step(action, data=...)`, `reset()`.
    `action_of(name)` turns a name into the toolkit's action object (the default imports `arcengine.GameAction`)."""
    def __init__(self, env, game: str, moves: int = MOVES, action_of: Optional[Callable] = None):
        self.env, self.game, self.moves, self.action_of = env, game, moves, action_of
        self.obs = env.observation_space; self.screen = rows_of(self.obs.frame[-1]); self.m = 0; self.done = False
        self.seen = {screen_key(self.screen): 1}; self.noop = set(); self.table: Dict[str, List[int]] = {}; self.hit: List[str] = []; self.miss: List[str] = []; self.log: List[Tuple[int, str, bool, int]] = []; self.level_moves: List[int] = []

    # -- what the game offers now
    def state(self) -> str: return getattr(self.obs.state, "name", str(self.obs.state))
    def levels(self) -> int: return int(self.obs.levels_completed)
    def win_levels(self) -> int: return int(self.obs.win_levels)
    def offered(self) -> List[str]: return ["ACTION%d" % i for i in self.obs.available_actions if i != 0]

    def legal(self, a: Optional[dict]) -> Tuple[bool, str]:
        """The check before a move reaches the game."""
        if a is None: return False, "it is not one JSON object with an action"
        name = a["action"]
        if name == "RESET": return True, ""
        if name not in self.offered(): return False, f"{name} is not offered now (offered: {', '.join(self.offered()) or 'none'}; RESET always works)"
        if name == "ACTION6":
            for k in ("x", "y"):
                v = a.get(k)
                if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 63: return False, f"a click needs {k} as a whole number from 0 to 63"
        return True, ""

    def apply(self, a: dict) -> dict:
        """Play a legal move. Returns what it did to the screen; updates the exact record."""
        if self.action_of is None:
            from arcengine import GameAction
            self.action_of = lambda n: GameAction[n]
        before, ak = self.screen, action_key(a); bk = screen_key(before); self.m += 1
        repeat = (bk, ak) in self.noop
        if a["action"] == "RESET": r = self.env.reset()
        else: r = self.env.step(self.action_of(a["action"]), data=({"x": a["x"], "y": a["y"]} if a["action"] == "ACTION6" else {}))
        lv0 = self.levels()
        if r is not None:
            self.obs = r
            if len(r.frame): self.screen = rows_of(r.frame[-1])
        n = cells_changed(before, self.screen); changed = n > 0; k = screen_key(self.screen); revisit = k in self.seen and k != bk
        self.seen[k] = self.seen.get(k, 0) + 1
        t = self.table.setdefault(a["action"], [0, 0, 0]); t[0] += 1; t[1] += int(changed); t[2] += n
        if not changed: self.noop.add((bk, ak))
        if a["action"] == "ACTION6": (self.hit if changed else self.miss).append(f"({a['x']},{a['y']})")
        self.log.append((self.m, ak, changed, n))
        if self.levels() > lv0: self.level_moves.append(self.m)
        if self.state() == "WIN": self.done = True
        return dict(moved=True, changed=changed, cells=n, repeat_noop=repeat, revisit=revisit, state=self.state(), levels=self.levels(), win_levels=self.win_levels())

    def record(self) -> str:
        """The Record of arm A: exact, computed from the engine's frames, never asked of the model; its size is bounded."""
        out = [f"Record kept by the framework (exact; your earlier turns are not in this chat). Move {self.m + 1} of {self.moves}."]
        if self.table:
            out.append("Actions tried so far (times tried / times the screen changed / mean cells changed): " + "; ".join(f"{k} {v[0]}/{v[1]}/{v[2] / v[0]:.1f}" for k, v in sorted(self.table.items())) + ".")
        else: out.append("No action tried yet.")
        if self.hit: out.append("Clicks that changed the screen: " + " ".join(self.hit[-8:]) + ".")
        if self.miss: out.append("Clicks that did nothing: " + " ".join(self.miss[-8:]) + ".")
        if self.log:
            out.append("Last moves: " + "; ".join(f"#{m} {k} -> " + (f"changed {n} cells" if c else "nothing happened") for m, k, c, n in self.log[-6:]) + ".")
        k = screen_key(self.screen); tried = sorted({a for (s, a) in self.noop if s == k})
        out.append(f"This exact screen has been seen {self.seen[k]} time(s) in this game." + (f" Moves already tried from it that did nothing: {', '.join(tried)}." if tried else ""))
        if self.level_moves: out.append("Levels completed at moves: " + ", ".join(str(m) for m in self.level_moves) + ".")
        return "\n".join(out)

    def turn(self, arm: str) -> str:
        """The user message of the next move: (arm A: the Record), the screen, the state line, the offered actions."""
        head = (self.record() + "\n\n") if arm == "A" else ""
        return (f"{head}Screen:\n" + "\n".join(self.screen) + f"\nState: {self.state()}. Levels completed: {self.levels()} of {self.win_levels()}. Offered actions: {', '.join(self.offered()) or 'RESET only'}. RESET always works.\n"
                "Reply with one JSON object.")


def _system(arm: str) -> dict: return {"role": "system", "content": SYSTEM_K if arm == "K" else SYSTEM_A}


# ---------------------------------------------------------------- what a move did
_att, parts_of, billed, computed, tokens_sent = GL._att, GL.parts_of, GL.billed, GL.computed, GL.tokens_sent


def _flags(play: Play, text: str, ok: bool, outcome: Optional[dict]) -> dict:
    a = parse_action(text)
    out = dict(parsed=a is not None, reply_ok=bool(ok), reply_exact=bool(ok and outcome and outcome["changed"]), moved=bool(outcome), kind=(a["action"] if a else "invalid"))
    if outcome: out.update(changed=outcome["changed"], cells=outcome["cells"], repeat_noop=outcome["repeat_noop"], revisit=outcome["revisit"], state=outcome["state"], levels=outcome["levels"], win_levels=outcome["win_levels"])
    else: out.update(changed=False, cells=0, repeat_noop=False, revisit=False, state=play.state(), levels=play.levels(), win_levels=play.win_levels())
    return out


class ReplayMismatch(Exception): pass


def reask_messages(msgs: list, text: str, why: str) -> list:
    return msgs + [{"role": "assistant", "content": text}, {"role": "user", "content": f"That move could not be used: {why}. Reply again with the one JSON object only."}]


def _game_done(rows: list, gi: int, moves: int) -> bool:
    g = [r for r in rows if r["g"] == gi]
    return bool(g) and (len(g) >= moves or g[-1]["state"] == "WIN")


def replay(rows: list, make_env: Callable, arm: str, games=GAMES, moves: int = MOVES, action_of: Optional[Callable] = None):
    """Play the saved replies back into fresh copies of the games (no call to the model). Returns (the Play of each game started, the chat history of the unfinished game of arm K). Raises at the first flag that differs.
    The chat history of arm K is rebuilt turn by turn, so a resumed run sends exactly what the first run sent."""
    plays: Dict[int, Play] = {}; hist: Dict[int, List[dict]] = {}
    for idx, rec in enumerate(rows, 1):
        gi = rec["g"]
        if gi not in plays: plays[gi] = Play(make_env(games[gi]), games[gi], moves, action_of); hist[gi] = []
        p = plays[gi]
        if rec["game"] != games[gi] or rec["m"] != p.m + 1 or rec["t"] != idx: raise ReplayMismatch(f"move {rec['t']}: out of order")
        a = parse_action(rec["text"]); ok, _ = p.legal(a)
        user = p.turn(arm); texts = ([rec["first_text"]] if rec.get("reasks") else []) + [x["text"] for x in rec.get("reasks", [])[:-1]]
        bad = []
        if arm == "A":
            if rec.get("asks", 1) != 1 + len(rec.get("reasks", [])) or len(rec.get("reasks", [])) > A_REASKS: bad.append("asks")
            if any(p.legal(parse_action(x))[0] for x in texts): bad.append("asked again after a usable move")
            if rec.get("reasks") and len(rec["reasks"]) < A_REASKS and not ok: bad.append("stopped asking before a usable move")
        out = p.apply(a) if ok else None
        if not ok: p.m += 1
        f = _flags(p, rec["text"], ok, out)
        diff = [k for k, v in f.items() if rec.get(k) != v] + bad
        if diff: raise ReplayMismatch(f"move {rec['t']} (game {games[gi]}): flags {diff} differ from the saved ones")
        hist[gi] += [{"role": "user", "content": user}, {"role": "assistant", "content": rec["text"]}]
    return plays, hist


def run(llm, make_env: Callable, arm: str, path: str, games=GAMES, moves: int = MOVES, meta=None, max_tokens: Optional[int] = None, progress=None, action_of: Optional[Callable] = None) -> dict:
    """Play `moves` moves in each game (or until the game is won). One JSON line per move after a header; the same call again carries on (the games are replayed from the saved moves; the chat of arm K is rebuilt).
    `stopped`: 'token_cap' or None."""
    meta = dict(meta or {}, arm=arm, games=list(games), moves=moves, game_seed=GAME_SEED, policy="blind" if arm == "K" else "record+check", reasks=A_REASKS if arm == "A" else 0)
    rows = load_rows(path, meta); stopped = None; spent = tokens_sent(rows)
    plays, hist = replay(rows, make_env, arm, games, moves, action_of)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w") as f: f.write(json.dumps({"_meta": _roundtrip(meta)}) + "\n")
    with open(path, "a") as f:
        for gi, game in enumerate(games):
            if _game_done(rows, gi, moves): continue
            if gi not in plays: plays[gi] = Play(make_env(game), game, moves, action_of); hist[gi] = []
            p = plays[gi]
            while p.m < moves and not p.done:
                if max_tokens is not None and spent >= max_tokens: stopped = "token_cap"; break
                msgs = [_system(arm)] + (hist[gi] if arm == "K" else []) + [{"role": "user", "content": p.turn(arm)}]
                text, ntok, sec = llm.generate(msgs, constrained=True); secs = float(sec); first = _att(llm, text, ntok, sec); extra = []
                a = parse_action(text); ok, why = p.legal(a)
                if arm == "A":
                    while not ok and len(extra) < A_REASKS:
                        text, ntok, sec = llm.generate(reask_messages(msgs, text, why), constrained=True); secs += float(sec); extra.append(_att(llm, text, ntok, sec)); a = parse_action(text); ok, why = p.legal(a)
                user = msgs[-1]; m_before = p.m
                out = p.apply(a) if ok else None
                if not ok: p.m += 1
                top = {k: first[k] for k in ("tokens", "think", "prompt_tokens", "cache_tokens", "first_token", "from_disk", "finish")}
                rec = dict(t=len(rows) + 1, g=gi, game=game, m=m_before + 1, text=text, seconds=round(secs, 4), **top, **_flags(p, text, ok, out))
                if extra: rec.update(asks=1 + len(extra), first_text=first["text"], reasks=extra)
                rec = _roundtrip(rec); f.write(json.dumps(rec) + "\n"); f.flush(); os.fsync(f.fileno()); rows.append(rec); spent += billed(rec)
                hist[gi] += [user, {"role": "assistant", "content": text}]
                if progress: progress(rec["t"], len(games) * moves, rec)
            if stopped: break
    return dict(rows=rows, summary=summarize(rows), meta=_roundtrip(meta), stopped=stopped)


# ---------------------------------------------------------------- summaries
stats, window, _gt, _le, _mean, _pct, estimate_cost = GL.stats, GL.window, GL._gt, GL._le, GL._mean, GL._pct, GL.estimate_cost
print_stats, print_bands, _f = GL.print_stats, GL.print_bands, GL._f


def by_prompt_band(rows: list, edges=BANDS) -> List[dict]: return GL.by_prompt_band(rows, edges)


def _share(rows, key): return (sum(1 for r in rows if r.get(key)) / len(rows)) if rows else None


def levels_total(rows: list) -> int:
    best: Dict[int, int] = {}
    for r in rows: best[r["g"]] = max(best.get(r["g"], 0), r.get("levels") or 0)
    return sum(best.values())


def by_game(rows: list) -> List[dict]:
    out = []
    for gi in sorted({r["g"] for r in rows}):
        b = [r for r in rows if r["g"] == gi]
        out.append(dict(g=gi, game=b[0]["game"], moves=len(b), levels=max(r["levels"] for r in b), win_levels=b[-1]["win_levels"], state=b[-1]["state"], legal=_share(b, "reply_ok"), effective=_share(b, "reply_exact"),
                        repeat_noop=_share(b, "repeat_noop"), revisit=_share(b, "revisit"), prompt_max=max((r["prompt_tokens"] or 0) for r in b)))
    return out


def summarize(rows: list) -> dict:
    return dict(all=stats(rows), legal=_share(rows, "reply_ok"), effective=_share(rows, "reply_exact"), repeat_noop=_share(rows, "repeat_noop"), revisit=_share(rows, "revisit"), levels=levels_total(rows), games=by_game(rows),
                max_prompt=max((r.get("prompt_tokens") or 0 for r in rows), default=0), reasked=sum(1 for r in rows if r.get("asks", 1) > 1), unusable=sum(1 for r in rows if not r["reply_ok"]))


def _m_between(rows, lo, hi): return [r for r in rows if lo <= r["m"] <= hi]


# ---------------------------------------------------------------- the reading rules (fixed in the page before any call)
G_FINISH_K = "the kept chat (arm K) played every game to 60 moves or to a win, with every move on file, and the replay of its saved replies into fresh copies of the games reproduces every saved flag of every move"
G_FINISH_A = "the arm with AXI (arm A) played every game to 60 moves or to a win, with every move on file, and the replay of its saved replies reproduces every saved flag and every re-ask"
G_DEEP = "the kept chat grew to a prompt of at least 100,000 tokens in at least one game"
G_CACHE = "the service reported cached tokens for at least 80 percent of the calls of arm K that had a prompt of 10,000 tokens or more"
G_UNUSABLE = "at most 2 percent of the moves of arm A ended with a reply that could not be used (after the re-asks)"
G_PARSED = "at least 95 percent of the replies of arm K were one JSON object with an action"
G_A_FLAT = "arm A: the mean prompt tokens of moves 41 to 60 are at most 1.5 times those of moves 6 to 25, over the games"
G_TOKENS_SENT = "tokens sent (prompt, reply and thinking): arm A sent at most one third of what arm K sent, over the same number of moves"
G_K_DRIFT = "arm K drifts: its share of effective moves (a move the game offered that changed the screen) over moves 41 to 60 is at least 5 points below that over moves 1 to 20"
G_REPEAT = "memory of what did nothing: arm A repeated a move that had already done nothing from the same screen at most half as often as arm K, and arm K did so on at least 2 percent of its moves"
G_K_ILLEGAL = "arm K plays illegal moves: at least 1 percent of its replies are not one of the actions the game offered (or a click outside the grid, or not a move at all)"
G_A_EFFECTIVE = "arm A: its share of effective moves is at least 5 points above that of arm K"
G_LEVELS = "levels: arm A completed at least as many levels as arm K, over the games (read it together with lines 14 and 15: if both completed none, this line reads yes without meaning anything)"
G_A_LEVEL = "arm A completed at least one level in at least one game"
G_K_LEVEL = "arm K completed at least one level in at least one game"
G_LATENCY = "latency: the mean seconds of a call of arm K at a prompt of 100,000 tokens or more are at least 1.5 times the mean seconds of a move of arm A"
G_FIRST_TOKEN = "latency to the first token: the median seconds to the first token of arm K at a prompt of 100,000 tokens or more are at least 1.5 times those of arm A"
G_SPEED = "speed: the effective moves per hour of calls of arm A are at least 1.5 times those of arm K"
G_COMPUTED = "tokens (the stand-in for energy): arm K computes (prompt tokens not served from the cache, plus reply and thinking tokens) at least 1.5 times as many tokens per effective move as arm A, over the moves of arm K at a prompt of 100,000 tokens or more"
G_LINES = [("guard", G_FINISH_K), ("guard", G_FINISH_A), ("guard", G_DEEP), ("instrument", G_CACHE), ("instrument", G_UNUSABLE), ("instrument", G_PARSED), ("claim", G_A_FLAT), ("claim", G_TOKENS_SENT), ("claim", G_K_DRIFT),
           ("claim", G_REPEAT), ("claim", G_K_ILLEGAL), ("claim", G_A_EFFECTIVE), ("claim", G_LEVELS), ("claim", G_A_LEVEL), ("claim", G_K_LEVEL), ("claim", G_LATENCY), ("claim", G_FIRST_TOKEN), ("claim", G_SPEED), ("claim", G_COMPUTED)]
G_HEAD = "headline: lines 8 and 13 both read yes (with AXI far fewer tokens are sent and no fewer levels are completed)"


def verdict(k: Optional[dict], a: Optional[dict], replays: Optional[dict] = None, games=GAMES, moves: int = MOVES) -> List[Tuple[int, str, str, Optional[bool]]]:
    """(number, tag, text, yes / no / None = not run). `k`, `a`: the dicts `run` returns (or None). `replays`: {'K': bool, 'A': bool}."""
    replays = replays or {}; out: List[Optional[bool]] = []; kr = k["rows"] if k else None; ar = a["rows"] if a else None
    full = lambda rows: all(_game_done(rows, gi, moves) for gi in range(len(games)))
    out.append(None if not k else bool(replays.get("K", True)) and full(kr))                                                                                           # 1
    out.append(None if not a else bool(replays.get("A", True)) and full(ar))                                                                                           # 2
    out.append(None if not k else max((r.get("prompt_tokens") or 0 for r in kr), default=0) >= 100_000)                                                               # 3
    big = [r for r in kr if (r.get("prompt_tokens") or 0) >= 10_000] if k else []
    out.append(None if not k or not big else sum(1 for r in big if (r.get("cache_tokens") or 0) > 0) / len(big) >= 0.8)                                               # 4
    out.append(None if not a else sum(1 for r in ar if not r["reply_ok"]) <= 0.02 * len(ar))                                                                           # 5
    out.append(None if not k else sum(1 for r in kr if r["parsed"]) >= 0.95 * len(kr))                                                                                 # 6
    if a:
        early, late = _m_between(ar, 6, 25), _m_between(ar, 41, 60)
        out.append(None if not early or not late else _le(_mean(r["prompt_tokens"] or 0 for r in late), 1.5 * _mean(r["prompt_tokens"] or 0 for r in early)))                # 7
    else: out.append(None)
    out.append(None if not k or not a or len(ar) != len(kr) else tokens_sent(ar) * 3 <= tokens_sent(kr))                                                              # 8
    if k:
        e, l = _m_between(kr, 1, 20), _m_between(kr, 41, 60)
        out.append(None if not e or not l else _gt(_share(e, "reply_exact") - _share(l, "reply_exact"), 0.05))                                                         # 9
    else: out.append(None)
    out.append(None if not k or not a else (lambda rk, ra: _share(kr, "repeat_noop") >= 0.02 - 1e-12 and ra <= 0.5 * rk + 1e-12)(_share(kr, "repeat_noop"), _share(ar, "repeat_noop")))     # 10
    out.append(None if not k else (1 - _share(kr, "reply_ok")) >= 0.01 - 1e-12)                                                                                          # 11
    out.append(None if not k or not a else _gt(_share(ar, "reply_exact") - _share(kr, "reply_exact"), 0.05))                                                          # 12
    out.append(None if not k or not a else levels_total(ar) >= levels_total(kr))                                                                                        # 13
    out.append(None if not a else levels_total(ar) >= 1)                                                                                                                # 14
    out.append(None if not k else levels_total(kr) >= 1)                                                                                                                # 15
    deep = window(kr, 100_000) if kr else []; sd = stats(deep) if deep else None; sa = stats(ar) if ar else None
    out.append(None if not deep or not a else _gt(sd["seconds_per_step"], 1.5 * sa["seconds_per_step"]))                                                               # 16
    out.append(None if not deep or not a or sd["first_p50"] is None or sa["first_p50"] is None else _gt(sd["first_p50"], 1.5 * sa["first_p50"]))                      # 17
    if k and a:
        sk = stats(kr); out.append(_gt(sa["exact_per_hour"], 1.5 * sk["exact_per_hour"]) if sk["exact_per_hour"] and sa["exact_per_hour"] else None)                  # 18
    else: out.append(None)
    out.append(None if not deep or not a or sd["computed_per_exact"] is None or sa["computed_per_exact"] is None else _gt(sd["computed_per_exact"], 1.5 * sa["computed_per_exact"]))      # 19
    assert len(out) == len(G_LINES)
    return [(i + 1, G_LINES[i][0], G_LINES[i][1], out[i]) for i in range(len(G_LINES))]


def claims_allowed(checks) -> dict:
    """Lines 1, 2, 3, 5 and 6 read yes: the figures are claimed as read. Line 4 NO: no computed-token (energy stand-in) figure is claimed, only prompt tokens."""
    ok = {i: c for i, _, _, c in checks}; base = all(ok[i] is True for i in (1, 2, 3, 5, 6))
    return dict(base=base, tokens_computed=bool(base and ok[4] is True))


def headline(checks) -> str:
    ok = {i: c for i, _, _, c in checks}; need = (8, 13)
    if any(ok[i] is None for i in need): return "NOT READ (a headline line was not run)"
    return "AXI SENDS FAR FEWER TOKENS AND LOSES NO LEVELS" if all(ok[i] for i in need) else "MIXED: " + ", ".join(f"line {i} {'yes' if ok[i] else 'NO'}" for i in need)


# ---------------------------------------------------------------- the check before the run, and the printout
def pilot_report(k_rows: list, a_rows: list, price_in=None, price_out=None, price_cached=None, cached_share_assumed: float = 0.9, games: int = len(GAMES), moves: int = MOVES) -> dict:
    """What a few real moves of each arm in one game say about the whole run: tokens of a screen, growth of the kept chat per move, thinking tokens, seconds, the total prompt tokens of arm K over all games (they grow
    with the square of the moves in a game), and the projected cost of the two arms at the prices typed in."""
    sk, sa = stats(k_rows), stats(a_rows); grow = (k_rows[-1]["prompt_tokens"] - k_rows[0]["prompt_tokens"]) / max(1, len(k_rows) - 1) if len(k_rows) > 1 else None; base = k_rows[0]["prompt_tokens"] if k_rows else None
    if grow:
        per_game = sum((base or 0) + grow * i for i in range(moves)); k_prompt = per_game * games; share = sk["cached_share"] if sk["cached_share"] is not None else cached_share_assumed
        k_tok = dict(prompt=k_prompt, cache=k_prompt * share, vis=games * moves * sk["vis"] / max(1, sk["live_steps"]), think=games * moves * sk["think"] / max(1, sk["live_steps"])); k_nc = dict(k_tok, cache=0)
    else: k_tok = k_nc = None
    n = games * moves; a_tok = dict(prompt=sa["prompt"] / max(1, sa["live_steps"]) * n, cache=0, vis=sa["vis"] / max(1, sa["live_steps"]) * n, think=sa["think"] / max(1, sa["live_steps"]) * n)
    return dict(k_first_prompt=base, k_growth_per_move=grow, k_last_prompt_projected=(base + grow * (moves - 1)) if grow else None, k_total_prompt_tokens=(k_tok or {}).get("prompt"), k_think_per_move=sk["think_per_step"],
                k_seconds_per_move=sk["seconds_per_step"], a_prompt_per_move=sa["prompt_per_step"], a_think_per_move=sa["think_per_step"], a_seconds_per_move=sa["seconds_per_step"], a_total_tokens=a_tok["prompt"] + a_tok["vis"] + a_tok["think"],
                cost_a=estimate_cost(a_tok, price_in, price_out), cost_k=estimate_cost(k_tok, price_in, price_out, price_cached) if k_tok else None, cost_k_nocache=estimate_cost(k_nc, price_in, price_out) if k_nc else None)


def print_games(rows: list):
    print(f"{'game':>16} {'moves':>5} | {'legal':>6} {'effective':>9} {'repeat no-op':>12} {'revisit':>8} | {'levels':>7} {'state':>13} | {'max prompt':>10}")
    for b in by_game(rows):
        print(f"{b['game']:>16} {b['moves']:>5} | {_f(b['legal'] * 100):>5}% {_f(b['effective'] * 100):>8}% {_f(b['repeat_noop'] * 100):>11}% {_f(b['revisit'] * 100):>7}% | {b['levels']:>3}/{b['win_levels']:<3} {b['state']:>13} | {b['prompt_max']:>10,}")


def print_verdict(checks, title="ARC-AGI-3 TEST (reading rules fixed before the run)"):
    print(f"\n-- {title} --")
    for i, tag, text, c in checks: print(f"  {'not run' if c is None else ('yes' if c else 'NO '):8} {i:2d}. [{tag}] {text}")
    print("\n  headline:", headline(checks))
    a = claims_allowed(checks)
    print("  rule on the page:", "lines 1, 2, 3, 5 and 6 read yes: the figures of the arms are claimed as read." if a["base"] else "a guard or instrument line is not yes: no claim is made until the cause is found.")
