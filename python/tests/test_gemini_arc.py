"""Implements concept-map item 117 (Gemini 3.8 Flash on ARC-AGI-3 games, test 24): the INSTRUMENT of docs/GEMINI_ARC_TEST.md, written before any call. It checks the instrument, not a result:
  1. the screen (hex rows, cells changed, the key), the action parser and the check before a move reaches the game;
  2. the exact record kept from the engine's frames (what each action did, repeated no-ops, revisits, the levels), on a stand-in game that follows the toolkit's interface;
  3. the two arms on a stand-in model: the flags of every move, the illegal moves (blind in K, re-asked in A), the chat of K (per game, growing) against the flat prompt of A, the resume from a file, the token cap,
     the replay into fresh copies of the games that refuses a doctored row;
  4. the reading lines flip at their boundaries; the headline needs lines 8 and 13;
  5. the page and the module agree and the page above its Result is byte for byte what was written before the run (a hash); the notebook holds no key, never turns billing on, sets no budget by itself;
  6. the front pages (concept-map item 117, VERIFIED row) and the counter.
What it does NOT show: anything about a real model's moves, about the real games, or the result of the run. No ARC package is needed to run these tests."""
import contextlib, hashlib, io, json, os, re, sys, tempfile
sys.path.insert(0, os.path.dirname(__file__))
from axi.experiments import gemini_arc as R

HERE = os.path.dirname(__file__)
PAGE = os.path.join(HERE, "..", "..", "docs", "GEMINI_ARC_TEST.md")
NB = os.path.join(HERE, "..", "colab", "axi_gemini_arc.ipynb")
PAGE_SHA256 = "0e9e5bed8ba0b86e14a3da45f7eae666c8f4e2a96d09875397348dde817c22dd"        # written before any call
RESULT_MARK = "\n\n---\n\n# Result"
IDENT = lambda n: n


def _h(text): return hashlib.sha256((text.rstrip("\n") + "\n").encode()).hexdigest()
def tmp(): return os.path.join(tempfile.mkdtemp(), "r.jsonl")


from types import SimpleNamespace as NS
class Obs:
    def __init__(s, frames, state, lv, win, avail): s.frame, s.state, s.levels_completed, s.win_levels, s.available_actions = frames, NS(name=state), lv, win, avail
class FakeEnv:
    """A tiny deterministic game: a player cell moves with ACTION1-4 on a 64x64 board, ACTION5 does nothing, ACTION6 paints a cell, reaching the target completes a level; 40 moves without a win is GAME_OVER."""
    def __init__(s, seed=0, target=(3, 0), levels=2):
        s.seed, s.target0, s.win = seed, target, levels; s._init()
    def _init(s):
        s.px, s.py, s.paint, s.lv, s.n, s.st, s.target = 0, 0, set(), 0, 0, "NOT_FINISHED", s.target0; s._obs()
    def _grid(s):
        g = [[0] * 64 for _ in range(64)]
        for (x, y) in s.paint: g[y][x] = 9
        g[s.target[1]][s.target[0]] = 4; g[s.py][s.px] = 7; return g
    def _obs(s): s.o = Obs([s._grid()], s.st, s.lv, s.win, [1, 2, 3, 4, 5, 6] if s.st != "WIN" else [])
    @property
    def observation_space(s): return s.o
    def reset(s): s.px, s.py, s.n = 0, 0, 0; s.st = "NOT_FINISHED"; s._obs(); return s.o
    def step(s, action, data=None):
        if s.st in ("GAME_OVER", "WIN"): return Obs([], s.st, s.lv, s.win, s.o.available_actions)
        s.n += 1
        if action == "ACTION1": s.py = max(0, s.py - 1)
        elif action == "ACTION2": s.py = min(63, s.py + 1)
        elif action == "ACTION3": s.px = max(0, s.px - 1)
        elif action == "ACTION4": s.px = min(63, s.px + 1)
        elif action == "ACTION6": s.paint ^= {(data["x"], data["y"])}
        if (s.px, s.py) == s.target:
            s.lv += 1
            if s.lv >= s.win: s.st = "WIN"
            else: s.target = (s.target[0] + 2, s.target[1] + 1)
        elif s.n >= 40: s.st = "GAME_OVER"
        s._obs(); return s.o
def make_env(game): return FakeEnv(seed=0, target=(3, 0) if game.startswith("a") else (2, 1))
class FakeLLM:
    def __init__(s, bad=()): s.seen, s.bad, s.n = [], set(bad), 0; s.last_prompt_tokens=0; s.last_think_tokens=5; s.from_disk=False; s.last_cache_tokens=0; s.last_first_token=0.3; s.last_finish="STOP"
    def generate(s, messages, constrained=True, max_new_tokens=400):
        s.seen.append(messages); s.n += 1; u = messages[-1]["content"]; reask = u.startswith("That move")
        txt = [m for m in messages if m["role"] == "user" and "Screen:" in m["content"]][-1]["content"]
        h = int(hashlib.sha256(txt.encode()).hexdigest(), 16); offered = txt.split("Offered actions: ")[1].split(".")[0].split(", ")
        pick = ["ACTION4", "ACTION4", "ACTION2", "ACTION5", "ACTION3", "ACTION1"][h % 6]
        if pick not in offered: pick = "RESET"
        if "ACTION6" in offered and h % 11 == 0: out = '{"action": "ACTION6", "x": %d, "y": %d}' % (h % 64, (h // 64) % 64)
        else: out = '{"action": "%s"}' % pick
        if s.n in s.bad and not reask: out = '{"action": "ACTION9"}'
        s.last_prompt_tokens = sum(len(m["content"]) for m in messages) // 4; s.last_cache_tokens = s.last_prompt_tokens // 2; s.last_first_token = 0.2 + 1e-6 * s.last_prompt_tokens
        return out, 8, 0.4 + 2e-6 * s.last_prompt_tokens

def make_env1(game): return FakeEnv(seed=0, target=(3, 0), levels=1)


class Walker(FakeLLM):
    """Always plays ACTION4."""
    def generate(self, messages, constrained=True, max_new_tokens=400):
        out, n, s = super().generate(messages, constrained, max_new_tokens); return '{"action": "ACTION4"}', n, s


class BadRun(FakeLLM):
    """Returns an action the game does not offer on the calls in `bad`, re-asks included."""
    def generate(self, messages, constrained=True, max_new_tokens=400):
        out, n, s = super().generate(messages, constrained, max_new_tokens)
        return ('{"action": "ACTION9"}' if self.n in self.bad else out), n, s


GAMES2 = ("alpha-1", "beta-2")


def run(llm, arm, path=None, **kw): return R.run(llm, kw.pop("make", make_env), arm, path or tmp(), games=kw.pop("games", GAMES2), moves=kw.pop("moves", 30), action_of=IDENT, **kw)
def rep(rows, arm, make=None, games=GAMES2, moves=30): return R.replay(rows, make or make_env, arm, games, moves, IDENT)


# ---------------------------------------------------------------- 1 the screen, the parser, the check
class Arr:
    def __init__(s, rows): s.rows = rows
    def tolist(s): return s.rows


def test_the_screen_is_64_hex_rows_and_cells_changed_and_the_key_are_counted_by_hand():
    g = [[0] * 64 for _ in range(64)]; g[0][0] = 15; g[63][63] = 10
    a = R.rows_of(g); assert len(a) == 64 and all(len(r) == 64 for r in a) and a[0].startswith("f0") and a[63].endswith("a") and a == R.rows_of(Arr(g))
    h = [list(r) for r in g]; h[5][5] = 3; h[6][6] = 3; b = R.rows_of(h); assert R.cells_changed(a, b) == 2 and R.cells_changed(a, a) == 0 and R.screen_key(a) == R.screen_key(R.rows_of(g)) and R.screen_key(a) != R.screen_key(b)
    assert R.rows_of([[16, 17]]) == ("01",)                                                                                                       # a value is cut to 4 bits, never an error


def test_the_action_parser_reads_one_json_object_and_the_check_refuses_what_the_game_does_not_offer():
    P = R.parse_action
    assert P('{"action": "ACTION1"}') == dict(action="ACTION1") and P(' {"action": "action3"} ') == dict(action="ACTION3") and P('```json\n{"action": "ACTION6", "x": 10, "y": 20}\n```') == dict(action="ACTION6", x=10, y=20)
    for bad in ("", None, "ACTION1", '[1]', '{"action": 5}', '{"act": "ACTION1"}', 'Sure: {"action": "ACTION1"}', '{"action": '): assert P(bad) is None, bad
    assert R.action_key(dict(action="ACTION6", x=3, y=4)) == "ACTION6@3,4" and R.action_key(dict(action="ACTION2")) == "ACTION2"
    p = R.Play(make_env("alpha"), "alpha", 30, IDENT); assert p.offered() == ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6"] and p.state() == "NOT_FINISHED" and p.levels() == 0 and p.win_levels() == 2
    ok = lambda d: p.legal(d)[0]
    assert ok(dict(action="ACTION4")) and ok(dict(action="RESET")) and not ok(dict(action="ACTION7")) and not ok(dict(action="ACTION9")) and not ok(None)
    assert ok(dict(action="ACTION6", x=0, y=63)) and ok(dict(action="ACTION6", x=63, y=0)) and not ok(dict(action="ACTION6")) and not ok(dict(action="ACTION6", x=64, y=0)) and not ok(dict(action="ACTION6", x=0, y=-1))
    assert not ok(dict(action="ACTION6", x=True, y=1)) and not ok(dict(action="ACTION6", x="5", y=1)) and not ok(dict(action="ACTION6", x=1.5, y=1)) and "not offered" in p.legal(dict(action="ACTION7"))[1] and "0 to 63" in p.legal(dict(action="ACTION6", x=99, y=1))[1]


# ---------------------------------------------------------------- 2 the record kept from the engine's frames
def test_the_record_is_computed_from_the_screens_and_counts_what_each_move_did():
    p = R.Play(make_env("alpha"), "alpha", 30, IDENT); r1 = p.apply(dict(action="ACTION4"))
    assert r1 == dict(moved=True, changed=True, cells=2, repeat_noop=False, revisit=False, state="NOT_FINISHED", levels=0, win_levels=2) and p.m == 1
    r2 = p.apply(dict(action="ACTION5")); assert r2["changed"] is False and r2["cells"] == 0 and r2["repeat_noop"] is False
    r3 = p.apply(dict(action="ACTION5")); assert r3["repeat_noop"] is True and r3["changed"] is False
    r4 = p.apply(dict(action="ACTION3")); assert r4["revisit"] is True and r4["changed"] is True                                                       # back to the very first screen
    r5 = p.apply(dict(action="ACTION5")); assert r5["repeat_noop"] is False                                                                          # a no-op from a DIFFERENT screen is not a repeat
    c1 = p.apply(dict(action="ACTION6", x=5, y=5)); assert c1["changed"] and p.hit == ["(5,5)"]
    rec = p.record(); assert "Move 7 of 30" in rec and "ACTION4 1/1/2.0" in rec and "ACTION5 3/0/0.0" in rec and "ACTION3 1/1/2.0" in rec and "Clicks that changed the screen: (5,5)" in rec
    assert "nothing happened" in rec and "changed 2 cells" in rec and "seen 1 time(s)" in rec and "#6 ACTION6@5,5 -> changed 1 cells" in rec
    q = R.Play(make_env("alpha"), "alpha", 30, IDENT); q.apply(dict(action="ACTION5")); assert "Moves already tried from it that did nothing: ACTION5" in q.record() and "No action tried yet" in R.Play(make_env("alpha"), "alpha", 30, IDENT).record()
    w = R.Play(make_env("alpha"), "alpha", 30, IDENT)
    for _ in range(3): w.apply(dict(action="ACTION4"))
    assert w.levels() == 1 and w.level_moves == [3] and "Levels completed at moves: 3." in w.record() and w.done is False
    w1 = R.Play(make_env1("alpha"), "alpha", 30, IDENT)
    for _ in range(3): w1.apply(dict(action="ACTION4"))
    assert w1.state() == "WIN" and w1.done is True and w1.offered() == []
    k = R.Play(make_env("alpha"), "alpha", 60, IDENT)
    for i in range(45): k.apply(dict(action="ACTION5"))
    assert k.state() == "GAME_OVER" and k.legal(dict(action="RESET"))[0] and k.apply(dict(action="RESET"))["state"] == "NOT_FINISHED"
    big = R.Play(make_env("alpha"), "alpha", 300, IDENT); import random as _r; rr = _r.Random(3)
    for _ in range(250): big.apply(rr.choice([dict(action="ACTION1"), dict(action="ACTION2"), dict(action="ACTION3"), dict(action="ACTION4"), dict(action="ACTION5"), dict(action="ACTION6", x=rr.randrange(64), y=rr.randrange(64)), dict(action="RESET")]))
    assert len(big.record()) < 2600                                                                                                                  # the record is bounded
    assert p.turn("K").startswith("Screen:\n") and p.turn("A").startswith("Record kept by the framework") and p.turn("K").count("\n") == 64 + 2 and "Offered actions: ACTION1" in p.turn("K") and p.turn("K").endswith("Reply with one JSON object.")


# ---------------------------------------------------------------- 3 the arms
def test_the_two_arms_play_the_same_moves_flag_them_and_handle_an_illegal_move_blind_or_re_asked():
    k = run(FakeLLM(bad=(5,)), "K"); a = run(FakeLLM(bad=(5,)), "A")
    for res in (k, a):
        rows = res["rows"]; assert len(rows) == 60 and res["stopped"] is None and [r["t"] for r in rows] == list(range(1, 61)) and [r["m"] for r in rows[:30]] == list(range(1, 31)) and rows[30]["m"] == 1 and rows[30]["g"] == 1
        for r in rows:
            assert r["reply_exact"] == (r["reply_ok"] and r["changed"]) and r["moved"] == r["reply_ok"] and r["parsed"] is True
    k5 = k["rows"][4]; assert k5["kind"] == "ACTION9" and k5["reply_ok"] is False and k5["moved"] is False and k5["changed"] is False and k5["m"] == 5 and k["rows"][5]["m"] == 6 and "asks" not in k5            # blind: not played, the move is used
    a5 = a["rows"][4]; assert a5["asks"] == 2 and a5["first_text"] == '{"action": "ACTION9"}' and a5["reply_ok"] is True and a5["moved"] is True and len(a5["reasks"]) == 1                                  # AXI: asked again, played
    ab = run(BadRun(bad=(7, 8, 9)), "A"); r7 = ab["rows"][6]; assert r7["asks"] == 3 and len(r7["reasks"]) == 2 and r7["reply_ok"] is False and r7["moved"] is False and r7["m"] == 7 and ab["rows"][7]["m"] == 8 and ab["summary"]["unusable"] == 1
    s = k["summary"]; assert s["legal"] == 59 / 60 and abs(s["effective"] - sum(1 for r in k["rows"] if r["reply_exact"]) / 60) < 1e-12 and s["levels"] == sum(g["levels"] for g in R.by_game(k["rows"])) and len(s["games"]) == 2
    rep(k["rows"], "K"); rep(a["rows"], "A"); rep(ab["rows"], "A")


def test_a_won_game_ends_early_and_the_next_game_starts():
    w = run(Walker(), "A", make=make_env1); g = R.by_game(w["rows"]); assert [x["moves"] for x in g] == [3, 3] and all(x["state"] == "WIN" and x["levels"] == 1 for x in g) and len(w["rows"]) == 6
    assert R.levels_total(w["rows"]) == 2 and [r["g"] for r in w["rows"]] == [0, 0, 0, 1, 1, 1]
    rep(w["rows"], "A", make=make_env1); wk = run(Walker(), "K", make=make_env1); assert len(wk["rows"]) == 6; rep(wk["rows"], "K", make=make_env1)


def test_the_kept_chat_grows_inside_a_game_and_restarts_in_the_next_while_the_axi_prompt_stays_flat():
    fk, fa = FakeLLM(), FakeLLM(); k = run(fk, "K"); a = run(fa, "A")
    kp = [r["prompt_tokens"] for r in k["rows"]]; ap = [r["prompt_tokens"] for r in a["rows"]]
    assert all(y > x for x, y in zip(kp[:30], kp[1:30])) and kp[30] < kp[29] / 5 and all(y > x for x, y in zip(kp[30:], kp[31:])) and kp[29] > 20 * kp[0]
    assert max(ap) < 1.5 * min(ap) and kp[29] > 8 * ap[29]
    m = fk.seen[3]; assert [x["role"] for x in m] == ["system", "user", "assistant", "user", "assistant", "user", "assistant", "user"] and m[0]["content"] == R.SYSTEM_K and fk.seen[30][0]["role"] == "system" and len(fk.seen[30]) == 2
    assert all(len(x) == 2 and x[0]["content"] == R.SYSTEM_A and x[1]["content"].startswith("Record kept by the framework") for x in fa.seen) and not any("Record kept" in x[-1]["content"] for x in fk.seen)
    assert R.SYSTEM_K.startswith(R.SYSTEM) and R.SYSTEM_A.startswith(R.SYSTEM) and R.SYSTEM_K[len(R.SYSTEM):].count("\n") == 1 and R.SYSTEM_A[len(R.SYSTEM):].count("\n") == 1 and "stay in this chat" in R.SYSTEM_K and "not in this chat" in R.SYSTEM_A
    fb = BadRun(bad=(5,)); run(fb, "A", moves=8); re = [x for x in fb.seen if x[-1]["content"].startswith("That move could not be used")][0]
    assert [x["role"] for x in re] == ["system", "user", "assistant", "user"] and re[2]["content"] == '{"action": "ACTION9"}' and "not offered" in re[3]["content"] and "one JSON object only" in re[3]["content"]


def test_a_stopped_run_resumes_from_its_file_to_the_same_rows_and_the_same_chat_and_a_foreign_file_is_refused():
    for arm in ("K", "A"):
        full = run(FakeLLM(), arm); fu = FakeLLM(); run(fu, arm)
        p = tmp(); part = run(FakeLLM(), arm, p, max_tokens=25_000); n = len(part["rows"]); assert part["stopped"] == "token_cap" and 0 < n < 60
        f2 = FakeLLM(); rest = run(f2, arm, p); assert rest["rows"] == full["rows"] and rest["stopped"] is None and f2.seen == fu.seen[n:]                       # the resumed run sends exactly what the first run sent
        try: run(FakeLLM(), arm, p, moves=29); raise AssertionError("another run's file accepted")
        except ValueError: pass
    p = tmp(); run(FakeLLM(), "K", p, moves=5, games=("alpha-1",))
    try: run(FakeLLM(), "A", p, moves=5, games=("alpha-1",)); raise AssertionError("an arm K file accepted for arm A")
    except ValueError: pass


def test_the_replay_refuses_a_doctored_file_and_a_doctored_re_ask():
    k = run(FakeLLM(), "K", moves=12); a = run(BadRun(bad=(4, 7, 8, 9)), "A", moves=12)
    for arm, res, key in (("K", k, "reply_exact"), ("K", k, "changed"), ("K", k, "cells"), ("K", k, "repeat_noop"), ("K", k, "revisit"), ("K", k, "levels"), ("A", a, "reply_ok"), ("A", a, "moved"), ("A", a, "reply_exact")):
        rows = [dict(r) for r in res["rows"]]; v = rows[8][key]; rows[8][key] = (not v) if isinstance(v, bool) else v + 1
        try: rep(rows, arm, games=GAMES2, moves=12); raise AssertionError(f"accepted a doctored {key}")
        except R.ReplayMismatch: pass
    rows = [dict(r) for r in a["rows"]]; rows[3]["asks"] = 3
    try: rep(rows, "A", moves=12); raise AssertionError("accepted a doctored ask count")
    except R.ReplayMismatch: pass
    rows = [dict(r) for r in a["rows"]]; rows[2]["asks"] = 2; rows[2]["first_text"] = rows[2]["text"]; rows[2]["reasks"] = [dict(rows[2])]
    try: rep(rows, "A", moves=12); raise AssertionError("accepted a re-ask after a usable move")
    except R.ReplayMismatch: pass
    rows = [dict(r) for r in k["rows"]]; rows[3]["text"] = '{"action": "ACTION1"}' if rows[3]["text"] != '{"action": "ACTION1"}' else '{"action": "ACTION2"}'
    try: rep(rows, "K", moves=12); raise AssertionError("accepted a doctored reply")
    except R.ReplayMismatch: pass
    rows = [dict(r) for r in k["rows"]]; rows[3], rows[4] = rows[4], rows[3]
    try: rep(rows, "K", moves=12); raise AssertionError("accepted rows out of order")
    except R.ReplayMismatch: pass


# ---------------------------------------------------------------- 4 the reading lines
def _row(t, g, m, prompt=2000, cache=0, sec=1.0, first=0.3, eff=True, ok=True, parsed=True, rep=False, rev=False, lv=0, vis=0, think=0):
    return dict(t=t, g=g, game=R.GAMES[g], m=m, text="", kind="ACTION1", tokens=vis, think=think, prompt_tokens=prompt, cache_tokens=cache, first_token=first, seconds=sec, from_disk=False, finish="STOP", parsed=parsed, reply_ok=ok,
                reply_exact=eff, moved=ok, changed=eff, cells=1, repeat_noop=rep, revisit=rev, state="NOT_FINISHED", levels=lv, win_levels=3)


def _arm(**kw):
    """4 games of 60 moves; `kw` is passed to every row, a callable value is called with (g, m)."""
    rows = []
    for g in range(4):
        for m in range(1, 61): rows.append(_row(len(rows) + 1, g, m, **{k: (v(g, m) if callable(v) else v) for k, v in kw.items()}))
    return rows


def L(checks): return {i: v for i, _, _, v in checks}
def V(k=None, a=None, rp=None): return L(R.verdict(None if k is None else dict(rows=k, stopped=None), None if a is None else dict(rows=a, stopped=None), rp))


def test_the_reading_lines_flip_at_their_boundaries_and_the_headline_needs_lines_8_and_13():
    assert len(R.G_LINES) == 19 and [t for t, _ in R.G_LINES[:6]] == ["guard"] * 3 + ["instrument"] * 3
    K = lambda **kw: _arm(prompt=120_000, cache=60_000, **kw); A = lambda **kw: _arm(**kw); k, a = K(), A()
    base = V(k, a, {"K": True, "A": True}); assert [base[i] for i in (1, 2, 3, 4, 5, 6, 7, 8, 13)] == [True] * 9 and base[9] is False and base[14] is False and base[15] is False and base[12] is False
    assert V(None, None) == {i: None for i in range(1, 20)}
    assert V(k, a, {"K": False})[1] is False and V(k, a, {"A": False})[2] is False and V(k[:-1], a)[1] is False and V(k, a[:-1])[2] is False
    short = [dict(r) for r in k]; [r.update(state="WIN") for r in short if r["g"] == 3 and r["m"] == 20]; short = [r for r in short if not (r["g"] == 3 and r["m"] > 20)]
    assert V(short, a)[1] is True                                                                                                                      # a game that was won ends early and is complete
    assert V(_arm(prompt=99_999, cache=50_000), a)[3] is False and V(_arm(prompt=100_000, cache=50_000), a)[3] is True
    big = lambda n: _arm(prompt=20_000, cache=lambda g, m: 5 if (g * 60 + m) <= n else 0)                                                              # 240 calls of 20,000 tokens
    assert V(big(192), a)[4] is True and V(big(191), a)[4] is False and V(_arm(prompt=5000), a)[4] is None
    assert V(k, _arm(ok=lambda g, m: not (g == 0 and m <= 4)))[5] is True and V(k, _arm(ok=lambda g, m: not (g == 0 and m <= 5)))[5] is False                 # 4.8 of 240 allowed
    assert V(_arm(parsed=lambda g, m: not (g == 0 and m <= 12)), a)[6] is True and V(_arm(parsed=lambda g, m: not (g == 0 and m <= 13)), a)[6] is False
    ap = lambda late: _arm(prompt=lambda g, m: late if m >= 41 else 1000); assert V(k, ap(1500))[7] is True and V(k, ap(1501))[7] is False
    assert V(_arm(prompt=3000), _arm(prompt=1000))[8] is True and V(_arm(prompt=3000), _arm(prompt=1001))[8] is False and V(k[:-1], a)[8] is None
    dr = lambda n: _arm(eff=lambda g, m: not (41 <= m <= 60 and g * 20 + (m - 41) < n)); assert V(dr(4), a)[9] is True and V(dr(3), a)[9] is False                  # 4 of 80 late moves lost = 5 points
    rp = lambda n: _arm(rep=lambda g, m: g * 60 + m <= n); assert V(rp(5), rp(2))[10] is True and V(rp(5), rp(3))[10] is False and V(rp(4), rp(0))[10] is False and V(rp(5), rp(0))[10] is True
    il = lambda n: _arm(ok=lambda g, m: not (g * 60 + m <= n)); assert V(il(3), a)[11] is True and V(il(2), a)[11] is False
    ef = lambda n: _arm(eff=lambda g, m: (g * 60 + m) > 240 - n); assert V(ef(200), ef(212))[12] is True and V(ef(200), ef(211))[12] is False
    lv = lambda x, y: _arm(lv=lambda g, m: x if g == 0 else (y if g == 1 else 0)); assert V(lv(1, 0), lv(1, 0))[13] is True and V(lv(2, 0), lv(1, 0))[13] is False and V(lv(0, 0), lv(0, 0))[13] is True and V(lv(0, 0), lv(0, 0))[14] is False
    assert V(k, lv(0, 1))[14] is True and V(lv(0, 1), a)[15] is True and V(lv(0, 0), a)[15] is False
    sec = lambda x: _arm(prompt=120_000, cache=60_000, sec=x, first=x / 10); assert V(sec(3.0), a=_arm(sec=2.0, first=0.2))[16] is True and V(sec(2.99), a=_arm(sec=2.0, first=0.2))[16] is False
    assert V(sec(3.0), a=_arm(sec=2.0, first=0.2))[17] is True and V(sec(2.99), a=_arm(sec=2.0, first=0.2))[17] is False
    assert V(sec(3.0), _arm(sec=2.0))[18] is True and V(sec(3.0), _arm(sec=2.01))[18] is False and V(sec(3.0), _arm(sec=1.0))[18] is True                  # equal moves, 1.5 times as fast is needed
    cp = lambda v: _arm(prompt=120_000, cache=60_000, vis=v); assert V(cp(10), _arm(prompt=30_000, vis=10))[19] is True and V(cp(10), _arm(prompt=40_000, vis=10))[19] is False and V(cp(10), _arm(prompt=120_000, vis=10))[19] is False
    ck = L(R.verdict(dict(rows=k, stopped=None), dict(rows=a, stopped=None), {})); mixed = [(i, t, x, (False if i == 13 else v)) for i, t, x, v in R.verdict(dict(rows=k, stopped=None), dict(rows=a, stopped=None), {})]
    chk = R.verdict(dict(rows=k, stopped=None), dict(rows=a, stopped=None), {}); assert R.headline(chk) == "AXI SENDS FAR FEWER TOKENS AND LOSES NO LEVELS" and R.headline(mixed) == "MIXED: line 8 yes, line 13 NO" and R.headline(R.verdict(dict(rows=k, stopped=None), None)).startswith("NOT READ")
    assert R.claims_allowed(chk)["base"] is True and R.claims_allowed([(i, t, x, (False if i == 4 else v)) for i, t, x, v in chk])["tokens_computed"] is False and R.claims_allowed([(i, t, x, (None if i == 3 else v)) for i, t, x, v in chk])["base"] is False


def test_the_tables_and_the_pilot_report_are_recounted_by_hand():
    rows = [_row(1, 0, 1, lv=0), _row(2, 0, 2, eff=False, rep=True, lv=0), _row(3, 0, 3, lv=1), _row(4, 1, 1, ok=False, eff=False, lv=0), _row(5, 1, 2, lv=2)]
    bg = R.by_game(rows); assert [(b["g"], b["moves"], b["levels"]) for b in bg] == [(0, 3, 1), (1, 2, 2)] and abs(bg[0]["effective"] - 2 / 3) < 1e-12 and bg[0]["repeat_noop"] == 1 / 3 and bg[1]["legal"] == 0.5
    assert R.levels_total(rows) == 3 and R.summarize(rows)["unusable"] == 1 and R.summarize(rows)["levels"] == 3 and R.summarize(rows)["reasked"] == 0
    k = [_row(t, 0, t, prompt=3000 + 2000 * t, vis=10) for t in range(1, 13)]; a = [_row(t, 0, t, prompt=1500, vis=10) for t in range(1, 13)]
    p = R.pilot_report(k, a, 1.0, 10.0, 0.1); assert abs(p["k_growth_per_move"] - 2000) < 1e-9 and p["k_first_prompt"] == 5000 and abs(p["k_last_prompt_projected"] - (5000 + 2000 * (R.MOVES - 1))) < 1e-6
    assert abs(p["k_total_prompt_tokens"] - len(R.GAMES) * sum(5000 + 2000 * i for i in range(R.MOVES))) < 1e-3 and p["cost_k"] <= p["cost_k_nocache"] and p["cost_a"] > 0 and R.pilot_report(k, a)["cost_a"] is None
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): R.print_stats("K", k); R.print_games(rows); R.print_bands(k); R.print_verdict(R.verdict(dict(rows=_arm(prompt=120_000), stopped=None), dict(rows=_arm(), stopped=None)))
    out = buf.getvalue(); assert "steps/h" in out and "headline:" in out and "repeat no-op" in out


# ---------------------------------------------------------------- 5 page and notebook
def _head():
    text = open(PAGE).read(); assert text.count(RESULT_MARK) <= 1
    return text[:text.index(RESULT_MARK)] if RESULT_MARK in text else text


def test_the_page_above_its_result_is_what_was_written_before_the_run():
    assert _h(_head()) == PAGE_SHA256


def test_the_page_and_the_module_agree():
    head = _head(); pred = head.split("## Predictions")[1].split("## Reading rules")[0]; rules = head.split("## Reading rules")[1].split("Headline:")[0]
    ps = re.findall(r"^(\d+)\. \[(\w+)\] (.*) \((\d+) %\)$", pred, re.M); rs = re.findall(r"^(\d+)\. (.*)$", rules, re.M)
    assert len(ps) == len(rs) == len(R.G_LINES) == 19
    for (n, tag, text, pr), (m, rt), (gt, gtext) in zip(ps, rs, R.G_LINES): assert n == m and text == rt == gtext and tag == gt and 0 < int(pr) < 100
    for needle in (R.MODEL, f"{R.MOVES} moves per game", f"{R.K_TOKEN_CAP:,}", f"{R.A_TOKEN_CAP:,}", "Energy in joules", "seconds to the first token", "AXI SENDS FAR FEWER TOKENS AND LOSES NO LEVELS", "not an entry on the ARC-AGI-3 leaderboard", "Record kept by the framework", *R.GAMES): assert needle in head, needle
    assert "ultimate" not in head.lower() and "$" not in head and "proven" not in head.lower() and "100 percent" not in head
    assert R.GAMES == ("ls20-9607627b", "g50t-5849a774", "ka59-38d34dbb", "sp80-589a99af") and R.MOVES == 60 and R.MODEL == "gemini-3.8-flash"


def test_the_notebook_holds_no_key_never_turns_billing_on_and_sets_no_budget():
    nb = json.load(open(NB)); src = "\n".join("".join(c["source"]) for c in nb["cells"]); code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert len(nb["cells"]) == 19 and not re.search(r"AIza[0-9A-Za-z_-]{20,}", src) and "GOOGLE_API_KEY" in src and "userdata.get" in src
    assert "BUDGET_USD = None" in code and "PAID = None" in code and "PRICE_IN = None" in code and "PRICE_CACHED = None" in code and "PRICE_OUT = None" in code
    assert "enable billing" not in code.lower() and "billing" not in code.split("PAID")[0]
    for name in ("run(", "pilot_report", "verdict", "GL.build_chat_llm", "print_games", "replay("): assert "G." + name in code, name
    assert code.count("need_budget()") >= 3 and all(c["outputs"] == [] for c in nb["cells"] if c["cell_type"] == "code") and "gemini_arc.py" in code and "arc_agi.Arcade()" in code and "G.GAME_SEED" in code and "arc-agi arcengine" in code
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            body = "".join(c["source"]); body = "\n".join(l for l in body.split("\n") if not l.lstrip().startswith("!")); compile(body, "cell", "exec")


# ---------------------------------------------------------------- 6 front pages
def test_the_front_pages_say_what_was_built_and_that_nothing_was_called():
    cm = open(os.path.join(HERE, "..", "..", "docs", "RAJNISH_CONCEPT_MAP.md")).read(); vf = open(os.path.join(HERE, "..", "..", "docs", "VERIFIED.md")).read()
    assert "Next free: row 183, item 137, test 162, question 122" in cm and cm.count("Next free:") == 1 and "test 143 checks the ARC-AGI-3 test on Gemini 3.8 Flash, item 117" in cm and "117. **Gemini 3.8 Flash on ARC-AGI-3 games" in cm
    assert "GEMINI_ARC_TEST.md" in vf and "gemini_arc.py" in vf and PAGE_SHA256 in vf
    i = cm.index("117. **"); item = cm[i:cm.index("\n", i)]; assert "Status: run once on Colab" in item and "docs/GEMINI_ARC_TEST.md" in item and "test 143" in item and "not** of how well the model solves new games" in item


if __name__ == "__main__":
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]; bad = 0
    for n, f in fns:
        try: f(); print("ok  ", n)
        except Exception: bad += 1; print("FAIL", n); traceback.print_exc()
    print(f"{len(fns) - bad}/{len(fns)} passed"); sys.exit(1 if bad else 0)
