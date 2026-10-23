"""The ARC-AGI-3 test with a host-owned explorer (docs/GEMINI_ARC_B_TEST.md, test 25): arm B, the same model and the same four games as Test 24, with AXI's exact Record AND three more things the framework owns.

Test 24 (gemini_arc.py) played arm K (the kept chat) and arm A (AXI's exact Record and check). Arm A never made an illegal move, but it still repeated one action again and again in one game, and it spent about 3,900 thinking
tokens a move. Arm B keeps everything of arm A (the Record, the check, the two re-asks) and adds, all computed by the framework from the engine's own frames and never asked of the model:
  1. an OBJECT VIEW (the fixed address frame): the objects on the screen (same-colour cells that touch), each with its colour, size, box and one cell inside it, and the exact box of what the last move changed;
  2. an UNTRIED LIST: the offered actions not yet tried from this exact screen, and the objects whose inside cell has not yet been clicked from it;
  3. a CONDUCTOR (the host decides, not the model): when the last 4 moves found no new screen, or this screen has been seen 3 times, the framework itself plays the first untried move from this screen, in a fixed order
     (ACTION1, 2, 3, 4, 5, 7, then the clicks), with no call to the model. A move played this way costs no tokens and no seconds and is marked (framework) in the Record and in the rows.
A progress score is kept as a reading: screens never seen before that a move reached, per 1,000 computed tokens. It decides nothing.
Nothing of Test 24, of `gemini_arc.py` or of the framework's core is changed: this module imports `gemini_arc.py`, `gemini_long.py` and `api_llm.py` and edits none of them. The model's thinking is left at its default, as in
Test 24, so that arm B differs from arm A only by the three things above."""
from __future__ import annotations
import json, os
from typing import Callable, Dict, List, Optional, Tuple

from axi.experiments import gemini_arc as R
from axi.experiments import gemini_long as GL
from axi.experiments.resumable import load_rows, _roundtrip

MODEL, GAMES, MOVES, GAME_SEED, A_REASKS = R.MODEL, R.GAMES, R.MOVES, R.GAME_SEED, R.A_REASKS
B_TOKEN_CAP = 8_000_000
PILOT_GAME, PILOT_MOVES = R.PILOT_GAME, 12
STALL_MOVES, SEEN_MAX, OBJ_MAX, CLICK_MAX = 4, 3, 12, 6
ORDER = ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION7")

SYSTEM_B = (R.SYSTEM + "\nThe earlier turns are not in this chat. Instead each turn starts with a Record kept exactly by the framework from the game's own screens, then the Objects on the screen and the Untried moves, "
            "both computed by the framework: use them. When you have been stuck, the framework itself plays one of the untried moves; that move is in the Record, marked (framework).")


# ---------------------------------------------------------------- the object view (the fixed address frame)
def objects(rows: Tuple[str, ...]) -> Tuple[str, List[dict]]:
    """The background (the most common colour) and the objects: 4-connected cells of one colour that are not the background. Each: colour, size, box, one cell inside it nearest the box's middle.
    Largest first (ties: top, left)."""
    h, w = len(rows), len(rows[0]); count: Dict[str, int] = {}
    for r in rows:
        for ch in r: count[ch] = count.get(ch, 0) + 1
    bg = sorted(count, key=lambda c: (-count[c], c))[0]; seen = [[False] * w for _ in range(h)]; out = []
    for y in range(h):
        for x in range(w):
            if seen[y][x] or rows[y][x] == bg: continue
            col = rows[y][x]; stack = [(x, y)]; seen[y][x] = True; cells = []
            while stack:
                cx, cy = stack.pop(); cells.append((cx, cy))
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if 0 <= nx < w and 0 <= ny < h and not seen[ny][nx] and rows[ny][nx] == col: seen[ny][nx] = True; stack.append((nx, ny))
            xs, ys = [c[0] for c in cells], [c[1] for c in cells]; x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys); mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            ix, iy = min(cells, key=lambda c: ((c[0] - mx) ** 2 + (c[1] - my) ** 2, c[1], c[0]))
            out.append(dict(c=col, n=len(cells), x0=x0, x1=x1, y0=y0, y1=y1, cx=ix, cy=iy))
    out.sort(key=lambda o: (-o["n"], o["y0"], o["x0"]))
    return bg, out


def changed_box(a: Tuple[str, ...], b: Tuple[str, ...]) -> Optional[dict]:
    xs, ys = [], []
    for y, (ra, rb) in enumerate(zip(a, b)):
        for x, (ca, cb) in enumerate(zip(ra, rb)):
            if ca != cb: xs.append(x); ys.append(y)
    return dict(n=len(xs), x0=min(xs), x1=max(xs), y0=min(ys), y1=max(ys)) if xs else None


# ---------------------------------------------------------------- one game with the explorer
class PlayB(R.Play):
    def __init__(self, env, game: str, moves: int = MOVES, action_of: Optional[Callable] = None):
        super().__init__(env, game, moves, action_of)
        self.tried_here: Dict[str, set] = {}; self.new_flags: List[bool] = []; self.last_box: Optional[dict] = None; self.have_moved = False

    def apply(self, a: dict, forced: bool = False) -> dict:
        before, seen0, k0 = self.screen, set(self.seen), R.screen_key(self.screen)
        out = super().apply(a)
        self.tried_here.setdefault(k0, set()).add(R.action_key(a))
        new = R.screen_key(self.screen) not in seen0; self.new_flags.append(new); out["new_screen"] = new; self.have_moved = True
        self.last_box = changed_box(before, self.screen) if out["changed"] else None
        if forced:
            m, ak, c, n = self.log[-1]; self.log[-1] = (m, ak + " (framework)", c, n)
        return out

    def untried(self) -> Tuple[List[str], List[Tuple[int, int]]]:
        done = self.tried_here.get(R.screen_key(self.screen), set()); off = self.offered()
        acts = [a for a in ORDER if a in off and a not in done]; clicks: List[Tuple[int, int]] = []
        if "ACTION6" in off:
            for o in objects(self.screen)[1]:
                if f"ACTION6@{o['cx']},{o['cy']}" not in done: clicks.append((o["cx"], o["cy"]))
        return acts, clicks[:CLICK_MAX]

    def stalled(self) -> bool:
        if self.state() != "NOT_FINISHED": return False
        if self.seen[R.screen_key(self.screen)] >= SEEN_MAX: return True
        last = self.new_flags[-STALL_MOVES:]
        return len(last) >= STALL_MOVES and not any(last)

    def forced(self) -> Optional[dict]:
        """The conductor: the move the framework plays itself, or None (then the model is asked)."""
        if self.m >= self.moves or self.done or not self.stalled(): return None
        acts, clicks = self.untried()
        if acts: return dict(action=acts[0])
        if clicks: return dict(action="ACTION6", x=clicks[0][0], y=clicks[0][1])
        return None

    def view(self) -> str:
        bg, objs = objects(self.screen); lines = [f"Objects on the screen (framework, exact; same-colour cells that touch; background colour {bg}): colour, number of cells, box, a cell inside it."]
        for o in objs[:OBJ_MAX]: lines.append(f"- colour {o['c']}: {o['n']} cells, x {o['x0']}-{o['x1']}, y {o['y0']}-{o['y1']}, inside cell ({o['cx']},{o['cy']})")
        if len(objs) > OBJ_MAX: lines.append(f"({len(objs) - OBJ_MAX} smaller objects not listed.)")
        if not objs: lines.append("(none: the screen is all background colour.)")
        if self.have_moved: lines.append(f"Change made by the last move: {self.last_box['n']} cells inside x {self.last_box['x0']}-{self.last_box['x1']}, y {self.last_box['y0']}-{self.last_box['y1']}." if self.last_box else "The last move changed nothing.")
        acts, clicks = self.untried()
        if acts or clicks:
            lines.append("Untried from this exact screen: " + (", ".join(acts) if acts else "no plain action") + "." + (" Objects whose inside cell has not been clicked from it: " + " ".join(f"({x},{y})" for x, y in clicks) + "." if clicks else ""))
        else: lines.append("Every offered move has been tried from this exact screen.")
        return "\n".join(lines)

    def turn(self, arm: str = "B") -> str:
        return self.record() + "\n\n" + self.view() + "\n\n" + R.Play.turn(self, "K")


def progress_score(rows: list) -> Optional[float]:
    """Screens never seen before that a move reached, per 1,000 computed tokens (a reading; it decides nothing)."""
    comp = sum(GL.computed(r) for r in rows); disc = discovered(rows)
    return (disc / comp * 1000) if comp else None


def discovered(rows: list) -> int: return sum(1 for r in rows if r.get("changed") and not r.get("revisit"))


# ---------------------------------------------------------------- the flags of a move and the replay
def _flags(p: PlayB, text: str, ok: bool, out: Optional[dict], forced: bool) -> dict:
    f = R._flags(p, text, ok, out); f["forced"] = forced; f["new_screen"] = bool(out and out.get("new_screen")); return f


def replay(rows: list, make_env: Callable, games=GAMES, moves: int = MOVES, action_of: Optional[Callable] = None) -> Dict[int, PlayB]:
    plays: Dict[int, PlayB] = {}
    for idx, rec in enumerate(rows, 1):
        gi = rec["g"]
        if gi not in plays: plays[gi] = PlayB(make_env(games[gi]), games[gi], moves, action_of)
        p = plays[gi]
        if rec["game"] != games[gi] or rec["m"] != p.m + 1 or rec["t"] != idx: raise R.ReplayMismatch(f"move {rec['t']}: out of order")
        f = p.forced(); bad = []
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
        fl = _flags(p, rec["text"], ok, out, f is not None)
        bad += [k for k, v in fl.items() if rec.get(k) != v]
        if bad: raise R.ReplayMismatch(f"move {rec['t']} (game {games[gi]}): flags {sorted(set(bad))} differ from the saved ones")
    return plays


def run(llm, make_env: Callable, path: str, games=GAMES, moves: int = MOVES, meta=None, max_tokens: Optional[int] = None, progress=None, action_of: Optional[Callable] = None) -> dict:
    """Arm B. One JSON line per move after a header; the same call again carries on. `stopped`: 'token_cap' or None."""
    meta = dict(meta or {}, arm="B", games=list(games), moves=moves, game_seed=GAME_SEED, policy="record+check+explorer", reasks=A_REASKS, stall_moves=STALL_MOVES, seen_max=SEEN_MAX, obj_max=OBJ_MAX)
    rows = load_rows(path, meta); stopped = None; spent = GL.tokens_sent(rows); plays = replay(rows, make_env, games, moves, action_of)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w") as f: f.write(json.dumps({"_meta": _roundtrip(meta)}) + "\n")
    with open(path, "a") as f:
        for gi, game in enumerate(games):
            if R._game_done(rows, gi, moves): continue
            if gi not in plays: plays[gi] = PlayB(make_env(game), game, moves, action_of)
            p = plays[gi]
            while p.m < moves and not p.done:
                if max_tokens is not None and spent >= max_tokens: stopped = "token_cap"; break
                fm = p.forced(); m_before = p.m
                if fm is not None:
                    text = json.dumps(fm); out = p.apply(fm, forced=True)
                    rec = dict(t=len(rows) + 1, g=gi, game=game, m=m_before + 1, text=text, seconds=0.0, tokens=0, think=0, prompt_tokens=0, cache_tokens=0, first_token=None, from_disk=True, finish="FRAMEWORK", **_flags(p, text, True, out, True))
                else:
                    msgs = [{"role": "system", "content": SYSTEM_B}, {"role": "user", "content": p.turn("B")}]
                    text, ntok, sec = llm.generate(msgs, constrained=True); secs = float(sec); first = GL._att(llm, text, ntok, sec); extra = []
                    a = R.parse_action(text); ok, why = p.legal(a)
                    while not ok and len(extra) < A_REASKS:
                        text, ntok, sec = llm.generate(R.reask_messages(msgs, text, why), constrained=True); secs += float(sec); extra.append(GL._att(llm, text, ntok, sec)); a = R.parse_action(text); ok, why = p.legal(a)
                    out = p.apply(a) if ok else None
                    if not ok: p.m += 1
                    top = {k: first[k] for k in ("tokens", "think", "prompt_tokens", "cache_tokens", "first_token", "from_disk", "finish")}
                    rec = dict(t=len(rows) + 1, g=gi, game=game, m=m_before + 1, text=text, seconds=round(secs, 4), **top, **_flags(p, text, ok, out, False))
                    if extra: rec.update(asks=1 + len(extra), first_text=first["text"], reasks=extra)
                rec = _roundtrip(rec); f.write(json.dumps(rec) + "\n"); f.flush(); os.fsync(f.fileno()); rows.append(rec); spent += GL.billed(rec)
                if progress: progress(rec["t"], len(games) * moves, rec)
            if stopped: break
    return dict(rows=rows, summary=summarize(rows), meta=_roundtrip(meta), stopped=stopped)


# ---------------------------------------------------------------- summaries and the reading rules
def load_arm(path: str) -> dict:
    """The saved rows of an arm of Test 24 or 25 (no call to the model, whatever the file holds)."""
    rows = []
    for line in open(path):
        if line.strip():
            d = json.loads(line)
            if "_meta" not in d: rows.append(d)
    return dict(rows=rows, stopped=None)


def model_rows(rows: list) -> list: return [r for r in rows if not r.get("forced")]


def _share(rows, key): return (sum(1 for r in rows if r.get(key)) / len(rows)) if rows else None


def moves_per_hour(rows: list, key: str = "reply_exact") -> Optional[float]:
    """Effective moves per hour of calls: every move counts, a move the framework played costs 0 seconds."""
    secs = sum(r["seconds"] for r in rows)
    return sum(1 for r in rows if r.get(key)) / secs * 3600 if secs else None


def summarize(rows: list) -> dict:
    mr = model_rows(rows); s = GL.stats(mr) if mr else None; calls = sum(len(GL.parts_of(r)) for r in mr)
    return dict(all=s, moves=len(rows), forced=sum(1 for r in rows if r.get("forced")), forced_changed=sum(1 for r in rows if r.get("forced") and r.get("changed")), legal=_share(mr, "reply_ok"), effective=_share(rows, "reply_exact"),
                repeat_noop=sum(1 for r in rows if r.get("repeat_noop")), discovered=discovered(rows), score=progress_score(rows), levels=R.levels_total(rows), games=R.by_game(rows), calls=calls,
                think_per_call=(sum(p["think"] for r in mr for p in GL.parts_of(r)) / calls if calls else None), per_hour=moves_per_hour(rows), computed=sum(GL.computed(r) for r in rows), sent=GL.tokens_sent(rows),
                unusable=sum(1 for r in mr if not r["reply_ok"]), reasked=sum(1 for r in mr if r.get("asks", 1) > 1))


G_FINISH_B = "the arm with the explorer (arm B) played every game to 60 moves or to a win, with every move on file, and the replay of its saved moves into fresh copies of the games reproduces every saved flag, every framework move and every re-ask"
G_FINISH_A = "the arm A of Test 24 played every game to 60 moves or to a win, with every move on file, and the replay of its saved replies reproduces every saved flag"
G_FORCED = "at most 40 percent of the moves of arm B were played by the framework itself (otherwise the explorer plays instead of the model)"
G_UNUSABLE = "at most 2 percent of the model's moves in arm B ended with a reply that could not be used (after the re-asks)"
G_EFFECTIVE = "wasted moves: the share of arm B's moves that changed the screen is at least 3 points above that of arm A"
G_REPEAT = "memory of what did nothing: arm B repeated a move that had already done nothing from the same screen no more often than arm A did"
G_SCORE = "progress per token: the screens never seen before that arm B reached, per 1,000 computed tokens, are at least 1.25 times those of arm A"
G_TOTAL = "reach: arm B reached at least as many screens never seen before as arm A, over the same number of moves"
G_LEVELS = "levels: arm B completed at least as many levels as arm A, over the games"
G_B_LEVEL = "arm B completed at least one level in at least one game"
G_MORE = "arm B completed more levels than arm A"
G_SPEED = "speed: the effective moves per hour of calls of arm B (every move counts; a move the framework plays takes no time) are at least 1.3 times those of arm A"
G_THINK = "thinking: the mean thinking tokens per call of arm B are no more than those of arm A"
G_TOKENS = "tokens (the stand-in for energy): arm B computes no more tokens per effective move than arm A"
G_FORCED_OK = "the explorer's own moves work: at least 60 percent of the moves the framework played itself changed the screen"
G_K_LEVELS = "against the kept chat: arm B completed at least as many levels as arm K (Test 24)"
G_K_TOKENS = "against the kept chat: arm B sent at most one third of the tokens arm K sent (Test 24)"
G_LINES = [("guard", G_FINISH_B), ("guard", G_FINISH_A), ("instrument", G_FORCED), ("instrument", G_UNUSABLE), ("claim", G_EFFECTIVE), ("claim", G_REPEAT), ("claim", G_SCORE), ("claim", G_TOTAL), ("claim", G_LEVELS),
           ("claim", G_B_LEVEL), ("claim", G_MORE), ("claim", G_SPEED), ("claim", G_THINK), ("claim", G_TOKENS), ("claim", G_FORCED_OK), ("claim", G_K_LEVELS), ("claim", G_K_TOKENS)]
G_HEAD = "headline: lines 5 and 7 both read yes (the explorer wastes fewer moves and reaches more new screens per token)"
HEADLINE = "THE HOST-OWNED EXPLORER WASTES FEWER MOVES AND REACHES MORE NEW SCREENS PER TOKEN"
_gt, _le = GL._gt, GL._le


def _complete(res: Optional[dict], games, moves) -> bool:
    if not res: return False
    rows = res["rows"]
    for gi in range(len(games)):
        g = [r for r in rows if r["g"] == gi]
        if not g or not (len(g) >= moves or g[-1]["state"] == "WIN"): return False
    return True


def verdict(b: Optional[dict], a: Optional[dict], k: Optional[dict] = None, replays: Optional[dict] = None, games=GAMES, moves: int = MOVES) -> List[Tuple[int, str, str, Optional[bool]]]:
    replays = replays or {}; out: List[Optional[bool]] = []
    out.append(None if not b else bool(_complete(b, games, moves) and replays.get("B") is True))                                                                                   # 1
    out.append(None if not a else bool(_complete(a, games, moves) and replays.get("A") is True))                                                                                   # 2
    sb = summarize(b["rows"]) if b else None; sa = summarize(a["rows"]) if a else None; sk = summarize(k["rows"]) if k else None
    out.append(None if not sb else _le(sb["forced"] / max(1, sb["moves"]), 0.40))                                                                                                  # 3
    out.append(None if not sb or not model_rows(b["rows"]) else _le(sb["unusable"] / len(model_rows(b["rows"])), 0.02))                                                           # 4
    both = bool(sb and sa and sb["moves"] == sa["moves"] and sb["moves"] > 0)
    out.append(_gt(sb["effective"], sa["effective"] + 0.03) if both else None)                                                                                                     # 5
    out.append(_le(sb["repeat_noop"], sa["repeat_noop"]) if both else None)                                                                                                        # 6
    out.append(_gt(sb["score"], 1.25 * sa["score"]) if both and sb["score"] is not None and sa["score"] else None)                                                                 # 7
    out.append(_gt(sb["discovered"], sa["discovered"] - 0.5) if both else None)                                                                                                    # 8
    out.append(_gt(sb["levels"], sa["levels"] - 0.5) if both else None)                                                                                                            # 9
    out.append(None if not sb else sb["levels"] >= 1)                                                                                                                              # 10
    out.append(_gt(sb["levels"], sa["levels"] + 0.5) if both else None)                                                                                                            # 11
    out.append(_gt(sb["per_hour"], 1.3 * sa["per_hour"]) if both and sb["per_hour"] and sa["per_hour"] else None)                                                                  # 12
    out.append(_le(sb["think_per_call"], sa["think_per_call"]) if both and sb["think_per_call"] is not None and sa["think_per_call"] is not None else None)                       # 13
    eff = lambda s: s["computed"] / max(1, round(s["effective"] * s["moves"]))
    out.append(_le(eff(sb), eff(sa)) if both else None)                                                                                                                            # 14
    out.append(None if not sb or not sb["forced"] else _gt(sb["forced_changed"] / sb["forced"], 0.60 - 1e-9))                                                                      # 15
    out.append(_gt(sb["levels"], sk["levels"] - 0.5) if sb and sk else None)                                                                                                       # 16
    out.append(_le(sb["sent"], sk["sent"] / 3) if sb and sk else None)                                                                                                             # 17
    assert len(out) == len(G_LINES)
    return [(i + 1, G_LINES[i][0], G_LINES[i][1], out[i]) for i in range(len(G_LINES))]


def claims_allowed(checks) -> dict:
    c = {i: v for i, _, _, v in checks}
    return dict(base=bool(c.get(1) and c.get(2) and c.get(4)), tokens_computed=bool(c.get(1) and c.get(2)))


def headline(checks) -> str:
    c = {i: v for i, _, _, v in checks}
    if c.get(5) is None or c.get(7) is None: return "NOT READ: arm B or arm A is missing"
    if c[5] and c[7]: return HEADLINE
    return "MIXED: line 5 " + ("yes" if c[5] else "NO") + ", line 7 " + ("yes" if c[7] else "NO")


def pilot_report(b_rows: list, price_in=None, price_out=None, price_cached=None, games: int = len(GAMES), moves: int = MOVES) -> dict:
    """What a few real moves of arm B in one game say about the whole run."""
    s = summarize(b_rows); mr = model_rows(b_rows); n = games * moves; share = len(mr) / max(1, len(b_rows)); st = s["all"]
    if not st: return dict(cost_b=None)
    per = dict(prompt=st["prompt_per_step"], vis=st["vis"] / max(1, st["live_steps"]), think=st["think_per_step"]); tot = dict(prompt=per["prompt"] * n * share, cache=0, vis=per["vis"] * n * share, think=per["think"] * n * share)
    return dict(b_forced_share=s["forced"] / max(1, s["moves"]), b_prompt_per_model_move=per["prompt"], b_think_per_model_move=per["think"], b_seconds_per_model_move=st["seconds_per_step"], b_total_tokens=tot["prompt"] + tot["vis"] + tot["think"],
                cost_b=GL.estimate_cost(tot, price_in, price_out))


def print_stats(title: str, rows: list):
    s = summarize(rows)
    print(f"{title}: {s['moves']} moves ({s['forced']} played by the framework) | legal {GL._f(None if s['legal'] is None else s['legal'] * 100)}% of model moves | effective {GL._f(s['effective'] * 100)}% | repeat no-op {s['repeat_noop']} | new screens {s['discovered']} "
          f"| per 1,000 computed tokens {GL._f(s['score'], 2)} | levels {s['levels']} | thinking per call {GL._f(s['think_per_call'], 0)} | effective moves per hour {GL._f(s['per_hour'], 0)} | computed {s['computed']:,} | sent {s['sent']:,}")


def print_verdict(checks, title="ARC-AGI-3 TEST WITH THE EXPLORER (reading rules fixed before the run)"):
    print(f"\n-- {title} --")
    for i, tag, text, c in checks: print(f"  {'not run' if c is None else ('yes' if c else 'NO '):8} {i:2d}. [{tag}] {text}")
    print("\n  headline:", headline(checks)); a = claims_allowed(checks)
    print("  rule on the page:", "lines 1, 2 and 4 read yes: the figures of the arms are claimed as read." if a["base"] else "a guard or instrument line is not yes: no claim is made until the cause is found.")


print_games = R.print_games
