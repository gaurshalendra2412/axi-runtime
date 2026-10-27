"""Test 170 (item 145): the instrument of the bigger ARC-AGI-3 test, eight games, 60 moves, three arms (docs/GEMINI_ARC_BIG_TEST.md, axi/experiments/gemini_arc_d.py, colab/axi_gemini_arc_big.ipynb).

No model is called and no ARC package is imported: the stand-in game and stand-in model of test 143. The tests check the rule that picks the eight games, the Notes of arm D against lines made by hand (and that they are made from the moves before the move only: the
replay refuses a row whose number of note lines differs), the clears recorded at a level change, the turn and the system text, the arm D rows, the resume, the replay into fresh games (it refuses a doctored row and the rows of arm C), the 18 reading lines at their
boundaries, the pilot report, the page hash, the agreement of page and module, the notebook (no key, no billing, no budget, one arm per tab, no arm run by the reading cell) and the front pages. The Result of the real run will be its own test."""
import hashlib, io, json, os, re, sys, tempfile, contextlib
from axi.experiments import gemini_arc_d as D
from axi.experiments import gemini_arc_c as C
from axi.experiments import gemini_arc_b as B
from axi.experiments import gemini_arc as R
from tests import test_gemini_arc as T

HERE = os.path.dirname(os.path.abspath(__file__)); DOCS = os.path.join(HERE, "..", "..", "docs")
PAGE = os.path.join(DOCS, "GEMINI_ARC_BIG_TEST.md"); NB = os.path.join(HERE, "..", "colab", "axi_gemini_arc_big.ipynb")
PAGE_SHA256 = "2fe67c072f190faac1fbccd02e56dd33a04f2a209cdb3bc1ed0735176f72a595"; RESULT_MARK = "\n## Result"
NEXT = "Next free: row 193, item 146, test 171, question 122"
G2 = ("alpha-1", "beta-2"); IDENT = T.IDENT; make_env = T.make_env
NG, NM = D.BIG_GAMES, D.BIG_MOVES; GAMESN = tuple("g%02d" % i for i in range(NG))


def tmp(): return tempfile.mktemp(suffix=".jsonl")
def run_d(llm, path=None, games=G2, moves=30, **kw): return D.run(llm, make_env, path or tmp(), games=games, moves=moves, action_of=IDENT, **kw)


# ---------------------------------------------------------------- 1 the eight games
def test_the_eight_games_follow_a_fixed_rule_that_does_not_depend_on_the_order_offered():
    offered = [f"game{i:02d}-{hashlib.sha256(str(i).encode()).hexdigest()[:8]}" for i in range(30)]; base = list(R.GAMES)
    got = D.pick_games(offered + base); assert got[:4] == base and len(got) == 8 and len(set(got)) == 8
    assert got == D.pick_games(list(reversed(offered + base))) == D.pick_games(sorted(offered + base)) and got == D.pick_games(offered + base + offered)                    # order and repeats do not matter
    rest = sorted(offered, key=lambda g: (hashlib.sha256(g.encode()).hexdigest(), g))[:8]; assert got[4:] == rest[:4]
    assert D.pick_games(offered) == rest and D.pick_games(offered, n=5) == rest[:5]                                                                                  # without the four, the first by the hash
    assert D.pick_games(base[:2] + offered[:3]) [:2] == base[:2] and len(D.pick_games(base[:2] + offered[:3])) == 5                                                              # fewer than eight offered: as many as there are
    assert D.BIG_GAMES == 8 and D.BIG_MOVES == 60 and D.BASE_GAMES == R.GAMES


# ---------------------------------------------------------------- 2 the notes
def _won_level_with_paint(moves=120):
    p = D.PlayD(make_env("alpha-1"), "alpha-1", moves, IDENT); p.apply(dict(action="ACTION6", x=5, y=5))
    for _ in range(3): p.apply(dict(action="ACTION4"))
    return p


def test_the_notes_are_made_by_hand_from_this_runs_ledgers_and_clears_only():
    p = D.PlayD(make_env("alpha-1"), "alpha-1", 30, IDENT); assert p.goal_text() == [] and p.goal_view() == "" and p.clears == {}
    p.apply(dict(action="ACTION5")); p.apply(dict(action="ACTION5")); assert p.goal_text() == ["- ACTION5 changed nothing on the board in its 2 tries here."]
    assert p.goal_view() == D.GOAL_HEAD + "\n- ACTION5 changed nothing on the board in its 2 tries here.\n\n"
    q = _won_level_with_paint(); assert q.levels() == 1 and list(q.clears) == [0] and q.clears[0]["delta"] == {"9": 1} and q.clears[0]["action"] == "ACTION4"                          # the clear: one more cell of colour 9 than at the first screen
    t = q.goal_text(); assert t[0] == "- on this level's first screen and not on an earlier level's: colour 9." and t[1] == "- a guess at what clears a level, not tested: repeat the change of the last clear: colour 9 +1 cells."
    assert len(t) <= D.GOAL_MAX_LINES and all(l.startswith("- ") for l in t) and "rule" not in " ".join(t).lower().replace("not a rule", "")
    r = D.PlayD(make_env("alpha-1"), "alpha-1", 30, IDENT)
    for a in ("ACTION4", "ACTION4", "ACTION4"): r.apply(dict(action=a))
    assert r.levels() == 1 and r.clears[0]["delta"] == {} and r.goal_text() == []                                                                                              # nothing new and nothing to guess: the section is left out
    r.apply(dict(action="RESET")); assert r.clears.keys() == {0}                                                                                                                # a RESET is not a clear


def test_a_clear_is_recorded_once_and_only_when_a_level_is_completed():
    p = _won_level_with_paint(); before = dict(p.clears)
    p.apply(dict(action="ACTION4")); p.apply(dict(action="ACTION5")); assert p.clears[0] == before[0] and 1 not in p.clears and p.levels() == 1
    for a in ("ACTION4", "ACTION4", "ACTION4", "ACTION4", "ACTION2"): p.apply(dict(action=a))
    assert p.levels() >= 1 and p.clears[0] == before[0]


def test_the_notes_of_a_move_are_a_function_of_the_moves_before_it():
    moves = [dict(action=a) for a in ("ACTION5", "ACTION6", "ACTION4", "ACTION4", "ACTION4", "ACTION5", "ACTION5", "ACTION2")]; moves[1] = dict(action="ACTION6", x=5, y=5)
    for k in range(len(moves) + 1):
        a = D.PlayD(make_env("alpha-1"), "alpha-1", 60, IDENT); b = D.PlayD(make_env("alpha-1"), "alpha-1", 60, IDENT)
        for m in moves[:k]: a.apply(m); b.apply(m)
        for m in (dict(action="ACTION3"), dict(action="ACTION3"), dict(action="ACTION1")): b.apply(m)                                                                          # b goes on; a stops: what a is shown at move k
        again = D.PlayD(make_env("alpha-1"), "alpha-1", 60, IDENT)
        for m in moves[:k]: again.apply(m)
        assert a.goal_text() == again.goal_text() and a.ledger_text() == again.ledger_text()


def test_the_turn_holds_the_notes_after_the_ledger_and_before_the_screen_and_only_when_there_are_some():
    p = D.PlayD(make_env("alpha-1"), "alpha-1", 30, IDENT)
    for a in ("ACTION5", "ACTION5"): p.apply(dict(action=a))
    t = p.turn("D"); i = [t.index(x) for x in ("Record kept by the framework", "Objects on the screen", "Untried from this exact screen", "Ledger of this level so far", "Notes on this level from the framework", "Screen:\n", "Offered actions: ACTION1")]
    assert i == sorted(i) and t.endswith("Reply with one JSON object.") and "- ACTION5 changed nothing on the board in its 2 tries here." in t
    q = D.PlayD(make_env("alpha-1"), "alpha-1", 30, IDENT); q.apply(dict(action="ACTION4")); u = q.turn("D"); assert "Notes on this level" not in u and u == q.record() + "\n\n" + q.view() + "\n\n" + q.ledger_view() + "\n\n" + R.Play.turn(q, "K")   # arm D = arm C when there is nothing to say
    assert D.SYSTEM_D.startswith(C.SYSTEM_C) and "Notes section" in D.SYSTEM_D and "none has been tested" in D.SYSTEM_D and D.SYSTEM_D != C.SYSTEM_C


# ---------------------------------------------------------------- 3 the arm
def test_arm_d_flags_every_move_holds_the_number_of_note_lines_and_replays():
    llm = T.FakeLLM(bad=(5,)); res = run_d(llm); rows = res["rows"]; assert len(rows) == 60 and res["stopped"] is None and [r["t"] for r in rows] == list(range(1, 61))
    assert all(isinstance(r["goal_lines"], int) and 0 <= r["goal_lines"] <= D.GOAL_MAX_LINES and "ledger_lines" in r for r in rows) and rows[0]["goal_lines"] == 0 and any(r["goal_lines"] >= 2 for r in rows)
    mr = [x for x in llm.seen if not x[-1]["content"].startswith("That move")]; assert all(x[0]["content"] == D.SYSTEM_D for x in mr) and any("Notes on this level from the framework" in x[-1]["content"] for x in mr)
    s = D.summarize(rows); assert 0.2 < s["goal_shown"] < 1.0 and s["goal_lines_mean"] > 0.5 and s["moves"] == 60 and s["games_with_level"] >= 1
    assert D.summarize([{k: v for k, v in r.items() if k != "goal_lines"} for r in rows])["goal_shown"] is None
    D.replay(rows, make_env, G2, 30, IDENT)


def test_a_stopped_run_resumes_to_the_same_rows_and_a_foreign_file_is_refused():
    full = run_d(T.FakeLLM()); p = tmp(); part = run_d(T.FakeLLM(), p, max_tokens=12_000); n = len(part["rows"]); assert part["stopped"] == "token_cap" and 0 < n < 60
    rest = run_d(T.FakeLLM(), p); assert rest["rows"] == full["rows"] and rest["stopped"] is None
    again = T.FakeLLM(); assert run_d(again, p)["rows"] == full["rows"] and again.n == 0
    try: run_d(T.FakeLLM(), p, moves=29); raise AssertionError("another run's file accepted")
    except ValueError: pass
    q = tmp(); C.run(T.FakeLLM(), make_env, q, games=G2, moves=30, action_of=IDENT)
    try: run_d(T.FakeLLM(), q); raise AssertionError("arm C's file accepted as arm D's")
    except ValueError: pass


def test_the_replay_refuses_a_doctored_file_a_doctored_note_count_and_the_rows_of_arm_c():
    res = run_d(T.FakeLLM(), moves=20); rows = res["rows"]; ix = next(i for i, r in enumerate(rows) if r["forced"]); jx = next(i for i, r in enumerate(rows) if not r["forced"] and i > ix and r["goal_lines"] > 0)
    def refuses(rs, why, fn=None):
        try: (fn or (lambda r: D.replay(r, make_env, G2, 20, IDENT)))(rs); raise AssertionError("accepted " + why)
        except R.ReplayMismatch: pass
    for key in ("changed", "cells", "new_screen", "levels", "ledger_lines", "goal_lines"):
        rs = [dict(r) for r in rows]; v = rs[jx][key]; rs[jx][key] = (not v) if isinstance(v, bool) else v + 1; refuses(rs, key)
    rs = [dict(r) for r in rows]; rs[jx]["goal_lines"] = 0; refuses(rs, "notes that were not shown")
    rs = [dict(r) for r in rows]; del rs[jx]["goal_lines"]; refuses(rs, "a row with no count")
    rs = [dict(r) for r in rows]; rs[ix]["forced"] = False; refuses(rs, "a framework move marked as the model's")
    rs = [dict(r) for r in rows]; rs[jx], rs[jx + 1] = rs[jx + 1], rs[jx]; refuses(rs, "rows out of order")
    refuses([dict(r) for r in C.run(T.FakeLLM(), make_env, tmp(), games=G2, moves=20, action_of=IDENT)["rows"]], "the rows of arm C (they hold no note count)")


def test_the_plain_arm_and_the_other_arms_run_on_the_big_settings_and_the_level_table_reads_them():
    p = tmp(); k = R.run(T.FakeLLM(), make_env, "K", p, games=G2, moves=40, action_of=IDENT); c = C.run(T.FakeLLM(), make_env, tmp(), games=G2, moves=40, action_of=IDENT); d = run_d(T.FakeLLM(), moves=40)
    assert D.levels_by_game(k["rows"], 2) == [max(r["levels"] for r in k["rows"] if r["g"] == i) for i in (0, 1)] and D.levels_by_game([], 3) == [0, 0, 0] and D.beats([1, 0, 2], [0, 0, 3]) == 1
    R.replay(k["rows"], make_env, "K", G2, 40, IDENT); C.replay(c["rows"], make_env, G2, 40, IDENT); D.replay(d["rows"], make_env, G2, 40, IDENT)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): D.print_levels(G2, K=k, C=c, D=None); D.print_stats("D", d["rows"]); D.print_games(d["rows"])
    out = buf.getvalue(); assert "alpha-1" in out and "total" in out and "notes shown in" in out and "games with a level" in out and "   -" in out
    pr = D.pilot_report(R.run(T.FakeLLM(), make_env, "K", tmp(), games=("alpha-1",), moves=12, action_of=IDENT)["rows"], C.run(T.FakeLLM(), make_env, tmp(), games=("alpha-1",), moves=12, action_of=IDENT)["rows"],
                        run_d(T.FakeLLM(), games=("alpha-1",), moves=12)["rows"], 1.0, 10.0, 0.1)
    assert pr["k_growth_per_move"] > 0 and pr["k_total_prompt_tokens"] > 0 and pr["cost_k"] > 0 and pr["cost_c"] > 0 and pr["cost_d"] > 0 and abs(pr["cost_total"] - (pr["cost_k"] + pr["cost_c"] + pr["cost_d"])) < 1e-9
    assert pr["d_notes_shown_share"] is not None and "c_forced_share" in pr and "d_forced_share" in pr and "d_ledger_lines_mean" in pr
    assert D.pilot_report(R.run(T.FakeLLM(), make_env, "K", tmp(), games=("alpha-1",), moves=12, action_of=IDENT)["rows"], c["rows"][:12], d["rows"][:12])["cost_total"] is None


# ---------------------------------------------------------------- 4 the reading lines
def _row(t, g, m, eff=True, ok=True, rep=False, rev=False, forced=False, lv=0, sec=3.0, think=0, prompt=3000, new=None, ll=3, gl=2):
    new = (eff and not rev) if new is None else new
    d = dict(t=t, g=g, game=GAMESN[g], m=m, text="{}", kind="ACTION1", tokens=10, think=think, prompt_tokens=prompt, cache_tokens=0, first_token=1.0, seconds=sec, from_disk=False, finish="STOP", parsed=True, reply_ok=ok,
             reply_exact=eff and ok, moved=ok, changed=eff, cells=1, repeat_noop=rep, revisit=rev, state="NOT_FINISHED", levels=lv, win_levels=3, forced=forced, new_screen=new, ledger_lines=ll, goal_lines=gl)
    if forced: d.update(tokens=0, think=0, prompt_tokens=0, seconds=0.0, from_disk=True, finish="FRAMEWORK")
    return d


def arm(**kw):
    rows = []
    for g in range(NG):
        for m in range(1, NM + 1): rows.append(_row(len(rows) + 1, g, m, **{k: (v(g, m) if callable(v) else v) for k, v in kw.items()}))
    return dict(rows=rows, stopped=None)


OK = {"K": True, "C": True, "D": True}
def V(k=None, c=None, d=None, rp=OK): return {i: v for i, _, _, v in D.verdict(k, c, d, rp, games=GAMESN, moves=NM)}
def lv(arr): return arm(lv=lambda g, m: arr[g])
def idx(g, m): return g * NM + m


def test_the_reading_lines_flip_at_their_boundaries_and_the_headline_needs_lines_8_and_11():
    assert len(D.G_LINES) == 18 and [t for t, _ in D.G_LINES[:6]] == ["guard", "guard", "guard", "instrument", "instrument", "instrument"] and all(t == "claim" for t, _ in D.G_LINES[6:])
    z = lv([0] * NG); base = V(z, z, z); assert [base[i] for i in (1, 2, 3, 4, 5, 6)] == [True] * 6 and V() == {i: None for i in range(1, 19)} and base[7] is False and base[9] is True
    assert V(z, z, z, {"K": False, "C": True, "D": True})[1] is False and V(z, z, z, {"K": True, "C": False, "D": True})[2] is False and V(z, z, z, {"K": True, "C": True, "D": False})[3] is False and V(z, z, z, {"K": True})[2] is False
    cut = dict(rows=z["rows"][:-1], stopped=None); assert V(cut, z, z)[1] is False and V(z, cut, z)[2] is False and V(z, z, cut)[3] is False
    fz = lambda n: arm(forced=lambda g, m: idx(g, m) <= n); assert V(z, fz(192), fz(192))[4] is True and V(z, fz(192), fz(193))[4] is False and V(z, fz(193), fz(192))[4] is False                     # 40 percent of 480
    ok = lambda n: arm(ok=lambda g, m: idx(g, m) > n); assert V(z, ok(9), ok(9))[5] is True and V(z, ok(9), ok(10))[5] is False and V(z, ok(10), ok(9))[5] is False                              # 9.6 allowed
    gl = lambda n: arm(gl=lambda g, m: 0 if idx(g, m) <= n else 2); assert V(z, z, gl(360))[6] is True and V(z, z, gl(361))[6] is False                                                          # 25 percent of 480 shown
    one = lambda i: lv([1 if j == i else 0 for j in range(NG)])
    assert V(z, one(0), z)[7] is True and V(one(0), one(1), z)[7] is False and V(one(0), z, z)[7] is False
    assert V(z, z, one(0))[8] is True and V(one(0), z, one(1))[8] is False
    assert V(z, one(0), one(1))[9] is True and V(z, one(0), z)[9] is False and V(z, z, one(0))[10] is True and V(z, one(0), one(1))[10] is False
    two = lv([1, 1] + [0] * (NG - 2))
    assert V(one(1), z, two)[11] is True and V(lv([0, 1] + [0] * (NG - 2)), z, one(0))[11] is False and V(lv([0, 1] + [0] * (NG - 2)), z, two)[11] is True                                                              # 1 win against 0; 1 against 1; 1 against 0
    assert V(z, one(1), two)[12] is True and V(z, lv([1, 0] + [0] * (NG - 2)), one(1))[12] is False
    assert V(two, two, two)[13] is True and V(lv([1, 1, 1] + [0] * (NG - 3)), two, two)[13] is False and V(two, lv([1, 1, 1] + [0] * (NG - 3)), two)[13] is False
    assert V(z, z, lv([2] + [0] * (NG - 1)))[14] is True and V(z, z, two)[14] is False
    ae = lambda n: arm(eff=lambda g, m: idx(g, m) > n); assert V(ae(15), z, ae(0))[15] is True and V(ae(14), z, ae(0))[15] is False and V(ae(15), z, ae(0))[15] is True
    sc = lambda p: arm(prompt=p, new=True); assert V(sc(3000), z, sc(1996))[16] is True and V(sc(3000), z, sc(1997))[16] is False                                                              # 1.5 times
    sn = lambda p: arm(prompt=p, new=True); assert V(sn(3000), z, sn(993))[17] is True and V(sn(3000), z, sn(994))[17] is False                                                                    # one third of the tokens sent
    th = lambda x: arm(think=x); assert V(th(100), z, th(100))[18] is True and V(th(100), z, th(101))[18] is False
    chk = D.verdict(z, z, z, OK, games=GAMESN, moves=NM); full = [(i, t, x, True) for i, t, x, v in chk]; assert D.headline(full) == D.HEADLINE and D.claims_allowed(full)["base"] is True
    assert D.headline([(i, t, x, (False if i == 8 else True)) for i, t, x, v in chk]).startswith("MIXED: line 8 NO, line 11 yes") and D.headline([(i, t, x, (False if i == 11 else True)) for i, t, x, v in chk]).startswith("MIXED: line 8 yes, line 11 NO")
    assert D.headline(D.verdict(z, z, None, OK, games=GAMESN)).startswith("NOT READ") and D.headline(chk).startswith("MIXED")
    for bad in (1, 2, 3, 5): assert D.claims_allowed([(i, t, x, (False if i == bad else True)) for i, t, x, v in chk])["base"] is False, bad
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): D.print_verdict(full); D.print_levels(GAMESN, K=z, C=one(0), D=two)
    assert "headline:" in buf.getvalue() and "total" in buf.getvalue()


# ---------------------------------------------------------------- 5 page and notebook
def _head():
    text = open(PAGE, encoding="utf-8").read(); assert text.count(RESULT_MARK) <= 1
    return text[:text.index(RESULT_MARK)] if RESULT_MARK in text else text


def test_the_page_above_its_result_is_what_was_written_before_the_run():
    assert hashlib.sha256(_head().encode()).hexdigest() == PAGE_SHA256


def test_the_page_and_the_module_agree():
    head = _head(); pred = head.split("## Predictions")[1].split("## Reading rules")[0]; rules = head.split("## Reading rules")[1].split("Headline:")[0]
    ps = re.findall(r"^(\d+)\. \[(\w+)\] (.*) \((\d+) %\)$", pred, re.M); rs = re.findall(r"^(\d+)\. (.*)$", rules, re.M)
    assert len(ps) == len(rs) == len(D.G_LINES) == 18
    for (n, tag, text, pr), (m, rt), (gt, gtext) in zip(ps, rs, D.G_LINES): assert n == m and text == rt == gtext and tag == gt and 0 < int(pr) < 100
    for needle in (D.MODEL, f"{D.BIG_MOVES} moves", f"{D.K_TOKEN_CAP:,}", f"{D.C_TOKEN_CAP:,}", D.HEADLINE, "Energy in joules", "not an entry on the ARC-AGI-3 leaderboard", "Sir to confirm", "RHAE", "sha256 of the id", "at most 8 lines", *R.GAMES): assert needle in head, needle
    assert "$" not in head and "proven" not in head.lower() and "100 percent" not in head and "ultimate" not in head.lower() and D.D_TOKEN_CAP == D.C_TOKEN_CAP and D.GOAL_MAX_LINES == 8
    assert head.index("## Predictions") < head.index("## Reading rules") and "one run per arm" in head and "separate sessions" in head and "Each is marked as a guess" in head.replace("**", "").replace("Each is marked", "Each is marked") or "marked as a guess" in head


def test_the_notebook_holds_no_key_never_turns_billing_on_sets_no_budget_and_runs_one_arm_per_tab():
    nb = json.load(open(NB)); src = "\n".join("".join(c["source"]) for c in nb["cells"]); code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert len(nb["cells"]) == 17 and not re.search(r"AIza[0-9A-Za-z_-]{20,}", src) and "GOOGLE_API_KEY" in src and "userdata.get" in src and "gemini_arc_d.py" in code
    for v in ("ARM", "PAID", "PRICE_IN", "PRICE_OUT", "PRICE_CACHED", "BUDGET_USD"): assert f"{v} = None" in code, v
    assert "enable billing" not in code.lower() and code.count("need_budget()") >= 2 and all(c["outputs"] == [] for c in nb["cells"] if c["cell_type"] == "code")
    for needle in ("D.run(", "C.run(", 'G.run(llm, make_env, "K"', "D.pilot_report", "D.verdict", "D.load_arm", "D.replay(", "C.replay(", "G.replay(", "D.print_verdict", "D.print_levels", "gemini_arc_big_all_", "D.pick_games", "D.PILOT_GAME",
                   "D.K_TOKEN_CAP", "D.C_TOKEN_CAP", "D.D_TOKEN_CAP", "axi_gemini_big", "arc_big_games.json", "CACHE = \"axi_gemini_big_cache_%s_%s.jsonl\" % (ARM, TAG)", "assert ARM in"): assert needle in code, needle
    reading = [("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code" and "D.verdict" in "".join(c["source"])]; assert len(reading) == 1 and "RUN[" not in reading[0] and ".run(" not in reading[0]          # the reading cell calls no arm
    runcell = [("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code" and "RUN[ARM]" in "".join(c["source"])]; assert len(runcell) == 1 and runcell[0].lstrip().startswith("need_budget()")
    pilot = [("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code" and "D.pilot_report" in "".join(c["source"])]; assert len(pilot) == 1 and "assert PAID" in pilot[0] and "need_budget" not in pilot[0]
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            body = "\n".join(l for l in "".join(c["source"]).split("\n") if not l.lstrip().startswith("!")); compile(body, "cell", "exec")


# ---------------------------------------------------------------- 5b the terminal runner (for when Colab will not give a machine)
def _runner():
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_gemini_arc_big", os.path.join(HERE, "..", "colab", "run_gemini_arc_big.py")); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def test_the_terminal_runner_runs_the_same_arms_resumes_reads_and_refuses_without_a_key_prices_or_budget():
    M = _runner(); out = tempfile.mkdtemp(); res = tempfile.mkdtemp(); buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        for arm in ("K", "C", "D"): assert M.do_run(T.FakeLLM(), make_env, G2, arm, out, moves=30, action_of=IDENT)["stopped"] is None
        first = {a: open(os.path.join(out, f"arc_big_{a}_default.jsonl")).read() for a in "KCD"}
        for arm in ("K", "C", "D"): llm = T.FakeLLM(); M.do_run(llm, make_env, G2, arm, out, moves=30, action_of=IDENT); assert llm.n == 0                                          # the same command again: nothing is asked again
        assert first == {a: open(os.path.join(out, f"arc_big_{a}_default.jsonl")).read() for a in "KCD"}
        got = M.do_read(make_env, G2, out, res, moves=30, action_of=IDENT)
        pil = M.do_pilot(T.FakeLLM(), make_env, tempfile.mkdtemp(), 1.0, 10.0, 0.1, action_of=IDENT)
    assert got["replays"] == {"K": True, "C": True, "D": True} and [i for i, _, _, v in got["checks"][:3] if v] == [1, 2, 3] and len(got["checks"]) == 18
    saved = json.load(open(os.path.join(res, "gemini_arc_big_all_default.json"))); assert len(saved["checks"]) == 18 and saved["games"] == list(G2) and "headline:" in open(os.path.join(res, "gemini_arc_big_printout.txt")).read()
    assert pil["rep"]["cost_total"] > 0 and set(pil["rows"]) == {"K", "C", "D"} and "headline:" in buf.getvalue() and "levels completed in each game" in buf.getvalue()
    mid = tempfile.mkdtemp(); part = M.do_run(T.FakeLLM(), make_env, G2, "D", mid, moves=30, max_tokens=12_000, action_of=IDENT); assert part["stopped"] == "token_cap"                                  # a stopped run is saved and carries on
    assert M.do_run(T.FakeLLM(), make_env, G2, "D", mid, moves=30, action_of=IDENT)["rows"] == D.run(T.FakeLLM(), make_env, tmp(), games=G2, moves=30, action_of=IDENT)["rows"]
    keep = {k: os.environ.pop(k, None) for k in ("GOOGLE_API_KEY", "GEMINI_API_KEY")}
    try:
        assert M.main(["pilot", "--out", out]) == 2 and M.main(["run", "K", "--out", out]) == 2
        os.environ["GOOGLE_API_KEY"] = "x"; assert M.main(["run", "K", "--out", out]) == 2 and M.main(["run", "--out", out, "--budget", "5"]) == 2                                                 # no budget, no arm: it stops before any call
        try: M.guard(type("A", (), dict(price_in=None, price_out=None, price_cached=None))()); raise AssertionError("prices not needed")
        except AssertionError as e: assert "price" in str(e).lower()
    finally:
        os.environ.pop("GOOGLE_API_KEY", None)
        for k, v in keep.items():
            if v is not None: os.environ[k] = v
    src = open(os.path.join(HERE, "..", "colab", "run_gemini_arc_big.py")).read(); assert not re.search(r"AIza[0-9A-Za-z_-]{20,}", src) and "GOOGLE_API_KEY" in src and "enable billing" not in src.lower() and "D.pick_games" in src
    assert M.CAPS == dict(K=D.K_TOKEN_CAP, C=D.C_TOKEN_CAP, D=D.D_TOKEN_CAP) and "--budget" in src and "docs/results/gemini_arc_big_all_default.json" in src


# ---------------------------------------------------------------- 6 front pages
def test_the_front_pages_say_what_was_built_and_that_nothing_was_called():
    cm = open(os.path.join(DOCS, "RAJNISH_CONCEPT_MAP.md"), encoding="utf-8").read(); vf = open(os.path.join(DOCS, "VERIFIED.md"), encoding="utf-8").read(); hl = open(os.path.join(DOCS, "HISTORY_LEDGER.md"), encoding="utf-8").read()
    assert NEXT in cm and cm.count("Next free:") == 1 and "\n145. **" in cm
    i = cm.index("\n145. **") + 1; item = cm[i:cm.index("\n", i)]
    assert "GEMINI_ARC_BIG_TEST.md" in item and "gemini_arc_d.py" in item and "test 170" in item and "Rajnish Choubey" in item and "Sir to confirm" in item and "NOT RUN" in item and "18 reading lines" in item and PAGE_SHA256 in item and "no model" in item.lower()
    assert any("GEMINI_ARC_BIG_TEST.md" in l and "gemini_arc_d.py" in l and PAGE_SHA256 in l and "NOT RUN" in l for l in vf.split("\n"))
    assert "| 192 (U) |" in hl and "eight games" in hl and "not run on the real games" in hl.lower()


if __name__ == "__main__":
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]; bad = 0
    for n, f in fns:
        try: f(); print("ok  ", n)
        except Exception: bad += 1; print("FAIL", n); traceback.print_exc()
    print(f"{len(fns) - bad}/{len(fns)} passed"); sys.exit(1 if bad else 0)
