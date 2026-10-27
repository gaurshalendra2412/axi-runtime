"""Test 144 (item 118): the instrument of the ARC-AGI-3 test with a host-owned explorer (docs/GEMINI_ARC_B_TEST.md, axi/experiments/gemini_arc_b.py, colab/axi_gemini_arc_b.ipynb).

No model is called and no ARC package is imported: a small stand-in game (the one of test 143) and a stand-in model. The tests check the object view against counts made by hand, the untried list, the conductor's rule at its
boundaries, that a framework move costs nothing and is marked, the arm B rows, the replay into fresh games (it refuses a doctored row), the resume, the 17 reading lines at their boundaries, the page hash, the agreement of
page and module, the notebook (no key, no budget, no billing) and the front pages. The Result of the real run will be its own test."""
import hashlib, io, json, os, re, sys, tempfile, contextlib
from axi.experiments import gemini_arc_b as B
from axi.experiments import gemini_arc as R
from tests import test_gemini_arc as T

HERE = os.path.dirname(os.path.abspath(__file__)); DOCS = os.path.join(HERE, "..", "..", "docs")
PAGE = os.path.join(DOCS, "GEMINI_ARC_B_TEST.md"); NB = os.path.join(HERE, "..", "colab", "axi_gemini_arc_b.ipynb")
PAGE_SHA256 = "5ebb7bb1feb1625e283c3184c217812385939b9bcc42247543df14e835e78f00"; RESULT_MARK = "\n## Result"
G2 = ("alpha-1", "beta-2"); IDENT = T.IDENT; make_env = T.make_env


def tmp(): return tempfile.mktemp(suffix=".jsonl")
def run_b(llm, path=None, games=G2, moves=30, make=None, **kw): return B.run(llm, make or make_env, path or tmp(), games=games, moves=moves, action_of=IDENT, **kw)
def rows_of(lines): return tuple(lines)


def screen(cells, bg="0"):
    g = [[bg] * 64 for _ in range(64)]
    for (x, y), c in cells.items(): g[y][x] = c
    return tuple("".join(r) for r in g)


# ---------------------------------------------------------------- 1 the object view
def test_the_objects_are_counted_by_hand_and_the_background_is_the_most_common_colour():
    cells = {(2, 3): "7", (3, 3): "7", (4, 3): "7", (3, 4): "7", (10, 10): "4", (20, 20): "7", (21, 21): "7"}   # a plus-like piece of 4 cells, a lone 4, two 7s that only touch at a corner
    bg, objs = B.objects(screen(cells)); assert bg == "0" and len(objs) == 4
    big = objs[0]; assert (big["c"], big["n"], big["x0"], big["x1"], big["y0"], big["y1"]) == ("7", 4, 2, 4, 3, 4) and (big["cx"], big["cy"]) == (3, 3)
    assert [(o["n"], o["x0"], o["y0"]) for o in objs[1:]] == [(1, 10, 10), (1, 20, 20), (1, 21, 21)]                                   # corners do not join; ties: top, left
    for o in objs: assert screen(cells)[o["cy"]][o["cx"]] == o["c"]                                                                       # the inside cell is on the object
    ring = {(x, y): "5" for x in range(10, 15) for y in range(10, 15) if x in (10, 14) or y in (10, 14)}
    (o,) = B.objects(screen(ring))[1]; assert o["n"] == 16 and screen(ring)[o["cy"]][o["cx"]] == "5"                                      # a ring: the middle is empty, the cell is still on the ring
    assert B.objects(screen({}))[1] == [] and B.objects(screen({(0, 0): "3"}, bg="3"))[0] == "3"


def test_the_box_of_a_change_is_exact():
    a = screen({(5, 5): "1"}); b = screen({(5, 5): "2", (9, 7): "3"}); assert B.changed_box(a, b) == dict(n=2, x0=5, x1=9, y0=5, y1=7) and B.changed_box(a, a) is None


# ---------------------------------------------------------------- 2 the untried list and the conductor
def test_the_untried_list_follows_the_exact_screen_and_the_conductor_waits_for_a_stall():
    p = B.PlayB(make_env("alpha-1"), "alpha-1", 30, IDENT); acts, clicks = p.untried()
    assert acts == ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"] and clicks == [(0, 0), (3, 0)] and p.forced() is None and "The last move" not in p.view()
    p.apply(dict(action="ACTION5")); assert "The last move changed nothing." in p.view() and p.untried()[0] == ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]                 # ACTION5 was tried here
    p.apply(dict(action="ACTION4")); assert p.untried()[0] == ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"]                                                       # a new screen: nothing tried there yet
    assert "Change made by the last move: 2 cells inside x 0-1, y 0-0." in p.view()
    q = B.PlayB(make_env("alpha-1"), "alpha-1", 30, IDENT)
    for i in range(3): q.apply(dict(action="ACTION5"))
    assert q.new_flags == [False] * 3 and q.forced() == dict(action="ACTION1") and q.seen[R.screen_key(q.screen)] == 4                      # seen 4 times already: stalled
    q = B.PlayB(make_env("alpha-1"), "alpha-1", 30, IDENT)
    for i in range(2): q.apply(dict(action="ACTION5"))
    assert q.seen[R.screen_key(q.screen)] == 3 and q.forced() == dict(action="ACTION1")                                                     # SEEN_MAX = 3
    z = B.PlayB(make_env("alpha-1"), "alpha-1", 30, IDENT)
    for a in ("ACTION4", "ACTION2", "ACTION3", "ACTION1", "ACTION4", "ACTION2"): z.apply(dict(action=a))                                    # round a square: three new screens, then three seen ones
    assert z.new_flags == [True, True, True, False, False, False] and max(z.seen.values()) == 2 and z.forced() is None                      # one new screen inside the last 4 moves
    z.apply(dict(action="ACTION3")); assert z.new_flags[-4:] == [False] * 4 and max(z.seen.values()) == 2 and z.forced() == dict(action="ACTION2")        # 4 moves and no new screen (ACTION1 was tried from here)
    z.apply(dict(action="ACTION2"), forced=True); assert z.log[-1][1] == "ACTION2 (framework)" and "ACTION2 (framework) ->" in z.record()


def test_the_conductor_runs_through_the_actions_then_the_clicks_then_hands_back_to_the_model():
    p = B.PlayB(make_env("alpha-1"), "alpha-1", 60, IDENT)
    for i in range(2): p.apply(dict(action="ACTION5"))
    played = []
    while p.forced() is not None and len(played) < 20:
        f = p.forced(); played.append(f); p.apply(f, forced=True)
        if len(p.new_flags) > 2 and p.new_flags[-1]: break
    assert played and played[0] == dict(action="ACTION1")
    k = B.PlayB(make_env("alpha-1"), "alpha-1", 60, IDENT)
    for i in range(2): k.apply(dict(action="ACTION5"))
    k.tried_here[R.screen_key(k.screen)] |= {"ACTION1", "ACTION2", "ACTION3", "ACTION4"}; assert k.forced() == dict(action="ACTION6", x=0, y=0)                                      # all plain actions tried: the first click
    k.tried_here[R.screen_key(k.screen)] |= {"ACTION6@0,0", "ACTION6@3,0"}; assert k.forced() is None                                                                              # nothing untried: the model decides
    g = B.PlayB(make_env("alpha-1"), "alpha-1", 60, IDENT)
    for i in range(45): g.apply(dict(action="ACTION5"))
    assert g.state() == "GAME_OVER" and g.forced() is None                                                                                  # after GAME_OVER only the model (RESET) moves


def test_the_turn_holds_the_record_the_objects_the_untried_and_then_the_screen():
    p = B.PlayB(make_env("alpha-1"), "alpha-1", 30, IDENT); t = p.turn("B"); i = [t.index(x) for x in ("Record kept by the framework", "Objects on the screen", "Untried from this exact screen", "Screen:\n", "Offered actions: ACTION1")]
    assert i == sorted(i) and t.endswith("Reply with one JSON object.") and t.count("\n") > 64 and "background colour 0" in t and "(3,0)" in t
    assert B.SYSTEM_B.startswith(R.SYSTEM) and "marked (framework)" in B.SYSTEM_B and B.SYSTEM_B != R.SYSTEM_A


# ---------------------------------------------------------------- 3 the arm
def test_arm_b_flags_every_move_a_framework_move_costs_nothing_and_the_model_is_asked_again_after_a_bad_reply():
    llm = T.FakeLLM(bad=(5,)); res = run_b(llm); rows = res["rows"]; assert len(rows) == 60 and res["stopped"] is None and [r["t"] for r in rows] == list(range(1, 61))
    forced = [r for r in rows if r["forced"]]; assert forced and llm.n < 60 and all(r["seconds"] == 0.0 and r["tokens"] == 0 and r["prompt_tokens"] == 0 and r["from_disk"] and r["finish"] == "FRAMEWORK" and r["reply_ok"] and r["parsed"] for r in forced)
    assert all(r["text"] == json.dumps(json.loads(r["text"])) for r in forced) and llm.n == sum(1 for r in rows if not r["forced"]) + sum(len(r.get("reasks", [])) for r in rows)
    five = [r for r in rows if r.get("asks")][0]; assert five["asks"] == 2 and five["first_text"] == '{"action": "ACTION9"}' and five["reply_ok"] is True and not five["forced"]
    for r in rows: assert r["reply_exact"] == (r["reply_ok"] and r["changed"]) and isinstance(r["new_screen"], bool) and (not r["new_screen"] or r["changed"])
    s = B.summarize(rows); assert s["moves"] == 60 and s["forced"] == len(forced) and s["legal"] == 1.0 and s["discovered"] == B.discovered(rows) and s["unusable"] == 0 and s["reasked"] == 1
    assert abs(s["effective"] - sum(1 for r in rows if r["reply_exact"]) / 60) < 1e-12 and B.model_rows(rows) == [r for r in rows if not r["forced"]]
    assert all("Objects on the screen" in x[-1]["content"] and x[0]["content"] == B.SYSTEM_B for x in llm.seen if not x[-1]["content"].startswith("That move"))
    B.replay(rows, make_env, G2, 30, IDENT)


def test_a_stopped_run_resumes_to_the_same_rows_and_a_foreign_file_is_refused():
    full = run_b(T.FakeLLM()); p = tmp(); part = run_b(T.FakeLLM(), p, max_tokens=12_000); n = len(part["rows"]); assert part["stopped"] == "token_cap" and 0 < n < 60
    rest = run_b(T.FakeLLM(), p); assert rest["rows"] == full["rows"] and rest["stopped"] is None
    again = T.FakeLLM(); assert run_b(again, p)["rows"] == full["rows"] and again.n == 0                                                       # nothing left to ask: no call
    try: run_b(T.FakeLLM(), p, moves=29); raise AssertionError("another run's file accepted")
    except ValueError: pass


def test_the_replay_refuses_a_doctored_file_a_move_the_conductor_would_not_play_and_a_missing_framework_move():
    res = run_b(T.FakeLLM(), moves=20); rows = res["rows"]; ix = next(i for i, r in enumerate(rows) if r["forced"]); jx = next(i for i, r in enumerate(rows) if not r["forced"] and i > ix)
    def refuses(rs, why):
        try: B.replay(rs, make_env, G2, 20, IDENT); raise AssertionError("accepted " + why)
        except R.ReplayMismatch: pass
    for key in ("changed", "cells", "new_screen", "revisit", "repeat_noop", "levels"):
        rs = [dict(r) for r in rows]; v = rs[jx][key]; rs[jx][key] = (not v) if isinstance(v, bool) else v + 1; refuses(rs, key)
    rs = [dict(r) for r in rows]; rs[ix]["forced"] = False; refuses(rs, "a framework move marked as the model's")
    rs = [dict(r) for r in rows]; rs[jx]["forced"] = True; refuses(rs, "a model move marked as the framework's")
    rs = [dict(r) for r in rows]; rs[ix]["text"] = '{"action": "ACTION3"}' if rs[ix]["text"] != '{"action": "ACTION3"}' else '{"action": "ACTION2"}'; refuses(rs, "another framework move")
    rs = [dict(r) for r in rows]; rs[jx], rs[jx + 1] = rs[jx + 1], rs[jx]; refuses(rs, "rows out of order")
    llm = T.BadRun(bad=(3,)); bad = run_b(llm, moves=20)["rows"]; rs = [dict(r) for r in bad]; k = next(i for i, r in enumerate(rs) if r.get("asks")); rs[k]["asks"] = 3; refuses(rs, "a doctored ask count")


def test_a_won_game_ends_early_and_progress_is_a_reading_that_decides_nothing():
    w = run_b(T.Walker(), make=T.make_env1); g = R.by_game(w["rows"]); assert [x["moves"] for x in g] == [3, 3] and R.levels_total(w["rows"]) == 2
    rows = [dict(changed=True, revisit=False, tokens=10, think=5, prompt_tokens=1000, cache_tokens=400, from_disk=False)] * 2 + [dict(changed=True, revisit=True, tokens=0, think=0, prompt_tokens=0, cache_tokens=0, from_disk=True)] + [dict(changed=False, revisit=False, tokens=10, think=0, prompt_tokens=1000, cache_tokens=0, from_disk=False)]
    assert B.discovered(rows) == 2 and abs(B.progress_score(rows) - 2 / (2 * 615 + 1010) * 1000) < 1e-9 and B.progress_score([]) is None


# ---------------------------------------------------------------- 4 the reading lines
def _row(t, g, m, eff=True, ok=True, rep=False, rev=False, forced=False, lv=0, sec=3.0, think=100, prompt=5000, new=None):
    new = (eff and not rev) if new is None else new
    d = dict(t=t, g=g, game=B.GAMES[g], m=m, text="{}", kind="ACTION1", tokens=10, think=think, prompt_tokens=prompt, cache_tokens=0, first_token=1.0, seconds=sec, from_disk=False, finish="STOP", parsed=True, reply_ok=ok,
             reply_exact=eff and ok, moved=ok, changed=eff, cells=1, repeat_noop=rep, revisit=rev, state="NOT_FINISHED", levels=lv, win_levels=3, forced=forced, new_screen=new)
    if forced: d.update(tokens=0, think=0, prompt_tokens=0, seconds=0.0, from_disk=True, finish="FRAMEWORK")
    return d


def arm(**kw):
    rows = []
    for g in range(4):
        for m in range(1, 61):
            rows.append(_row(len(rows) + 1, g, m, **{k: (v(g, m) if callable(v) else v) for k, v in kw.items()}))
    return dict(rows=rows, stopped=None)


def V(b=None, a=None, k=None, rp=None): return {i: v for i, _, _, v in B.verdict(b, a, k, rp)}
OKR = {"B": True, "A": True}


def test_the_reading_lines_flip_at_their_boundaries_and_the_headline_needs_lines_5_and_7():
    assert len(B.G_LINES) == 17 and [t for t, _ in B.G_LINES[:4]] == ["guard", "guard", "instrument", "instrument"]
    a = arm(eff=lambda g, m: m % 20 != 0, rev=lambda g, m: m % 7 == 0); b = arm(eff=True); base = V(b, a, None, OKR)
    assert [base[i] for i in (1, 2, 3, 4)] == [True] * 4 and base[16] is None and base[17] is None and V(None, None) == {i: None for i in range(1, 18)}
    assert V(b, a, None, {"B": False, "A": True})[1] is False and V(b, a, None, {"B": True, "A": False})[2] is False and V(b, a, None, {"B": True})[2] is False
    cut = dict(rows=b["rows"][:-1], stopped=None); assert V(cut, a, None, OKR)[1] is False
    assert V(arm(forced=lambda g, m: m <= 24), a, None, OKR)[3] is True and V(arm(forced=lambda g, m: m <= 25), a, None, OKR)[3] is False                                          # 40 percent of 240 = 96
    fz = lambda n: arm(forced=lambda g, m: g * 60 + m <= n); assert V(fz(96), a, None, OKR)[3] is True and V(fz(97), a, None, OKR)[3] is False
    assert V(arm(ok=lambda g, m: not (g == 0 and m <= 4)), a, None, OKR)[4] is True and V(arm(ok=lambda g, m: not (g == 0 and m <= 5)), a, None, OKR)[4] is False              # 4.8 of 240 allowed
    ae = lambda n: arm(eff=lambda g, m: (g * 60 + m) > n); assert V(ae(0), ae(8), None, OKR)[5] is True and V(ae(0), ae(7), None, OKR)[5] is False              # 3 points of 240 = 7.2 moves
    rp = lambda n: arm(rep=lambda g, m: g * 60 + m <= n); assert V(rp(3), rp(3), None, OKR)[6] is True and V(rp(4), rp(3), None, OKR)[6] is False
    sc = lambda c: arm(prompt=c, new=True, think=0); assert V(sc(1000), sc(1253), None, OKR)[7] is True and V(sc(1000), sc(1252), None, OKR)[7] is False         # computed tokens per move 1,010 against 1,262.5 (1.25 times)
    ds = lambda n: arm(rev=lambda g, m: g * 60 + m <= n); assert V(ds(10), ds(10), None, OKR)[8] is True and V(ds(11), ds(10), None, OKR)[8] is False
    lv = lambda x: arm(lv=lambda g, m: x if g == 0 else 0); assert V(lv(1), lv(1), None, OKR)[9] is True and V(lv(0), lv(1), None, OKR)[9] is False and V(lv(1), lv(0), None, OKR)[11] is True and V(lv(1), lv(1), None, OKR)[11] is False
    assert V(lv(1), lv(0), None, OKR)[10] is True and V(lv(0), lv(0), None, OKR)[10] is False
    sp = lambda s: arm(sec=s); assert V(sp(2.0), sp(2.6), None, OKR)[12] is True and V(sp(2.0), sp(2.59), None, OKR)[12] is False
    th = lambda x: arm(think=x); assert V(th(100), th(100), None, OKR)[13] is True and V(th(101), th(100), None, OKR)[13] is False
    cm = lambda c: arm(prompt=c, think=0); assert V(cm(1000), cm(1000), None, OKR)[14] is True and V(cm(1001), cm(1000), None, OKR)[14] is False
    fc = lambda n: arm(forced=lambda g, m: m <= 10, eff=lambda g, m: not (m <= 10 and g * 10 + m <= n)); assert V(fc(16), a, None, OKR)[15] is True and V(fc(17), a, None, OKR)[15] is False and V(b, a, None, OKR)[15] is None
    k = arm(lv=lambda g, m: 1 if g == 0 else 0, prompt=120_000); assert V(lv(1), a, k, OKR)[16] is True and V(lv(0), a, k, OKR)[16] is False and V(b, a, k, OKR)[17] is True
    k2 = arm(prompt=7000, think=100); assert V(b, a, k2, OKR)[17] is False                                                                                                              # 5,000 vs 7,000 per move: not a third
    chk = B.verdict(arm(forced=lambda g, m: m <= 5, prompt=1000, new=True), arm(eff=lambda g, m: m % 5 != 0, prompt=3000), None, OKR); h = B.headline(chk)
    assert h == B.HEADLINE or h.startswith("MIXED"); assert B.headline([(i, t, x, (False if i == 7 else (v if v is not None else True))) for i, t, x, v in chk]).startswith("MIXED: line 5") and B.headline(B.verdict(b, None)).startswith("NOT READ")
    full = [(i, t, x, True) for i, t, x, v in chk]; assert B.headline(full) == B.HEADLINE and B.claims_allowed(full)["base"] is True and B.claims_allowed([(i, t, x, (False if i == 4 else True)) for i, t, x, v in chk])["base"] is False
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): B.print_stats("B", b["rows"]); B.print_games(b["rows"]); B.print_verdict(full)
    assert "headline:" in buf.getvalue() and "played by the framework" in buf.getvalue()


def test_the_pilot_report_and_the_arm_loader():
    rows = arm(forced=lambda g, m: m % 4 == 0)["rows"][:12]; r = B.pilot_report(rows, 1.0, 10.0, 0.1)
    assert abs(r["b_forced_share"] - 0.25) < 1e-12 and r["b_prompt_per_model_move"] == 5000 and r["cost_b"] > 0 and B.pilot_report(rows)["cost_b"] is None
    p = tmp(); res = run_b(T.FakeLLM(), p, moves=5, games=("alpha-1",)); got = B.load_arm(p); assert got["rows"] == res["rows"] and got["stopped"] is None


# ---------------------------------------------------------------- 5 page and notebook
def _head():
    text = open(PAGE).read(); assert text.count(RESULT_MARK) <= 1
    return text[:text.index(RESULT_MARK)] if RESULT_MARK in text else text


def test_the_page_above_its_result_is_what_was_written_before_the_run():
    assert hashlib.sha256(_head().encode()).hexdigest() == PAGE_SHA256


def test_the_page_and_the_module_agree():
    head = _head(); pred = head.split("## Predictions")[1].split("## Reading rules")[0]; rules = head.split("## Reading rules")[1].split("Headline:")[0]
    ps = re.findall(r"^(\d+)\. \[(\w+)\] (.*) \((\d+) %\)$", pred, re.M); rs = re.findall(r"^(\d+)\. (.*)$", rules, re.M)
    assert len(ps) == len(rs) == len(B.G_LINES) == 17
    for (n, tag, text, pr), (m, rt), (gt, gtext) in zip(ps, rs, B.G_LINES): assert n == m and text == rt == gtext and tag == gt and 0 < int(pr) < 100
    for needle in (B.MODEL, f"{B.MOVES} moves each", f"{B.B_TOKEN_CAP:,}", B.HEADLINE, "Energy in joules", "not an entry on the ARC-AGI-3 leaderboard", "marked \"(framework)\"", "Sir to confirm", "edits none of them", *B.GAMES): assert needle in head, needle
    assert "$" not in head and "proven" not in head.lower() and "100 percent" not in head and "ultimate" not in head.lower()
    assert B.STALL_MOVES == 4 and B.SEEN_MAX == 3 and B.OBJ_MAX == 12 and B.CLICK_MAX == 6 and f"last {B.STALL_MOVES} moves" in head and f"seen {B.SEEN_MAX} times" in head


def test_the_notebook_holds_no_key_never_turns_billing_on_sets_no_budget_and_never_runs_the_other_arms():
    nb = json.load(open(NB)); src = "\n".join("".join(c["source"]) for c in nb["cells"]); code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert len(nb["cells"]) == 17 and not re.search(r"AIza[0-9A-Za-z_-]{20,}", src) and "GOOGLE_API_KEY" in src and "userdata.get" in src and "gemini_arc_b.py" in code
    for v in ("PAID", "PRICE_IN", "PRICE_OUT", "PRICE_CACHED", "BUDGET_USD"): assert f"{v} = None" in code, v
    assert "enable billing" not in code.lower() and code.count("need_budget()") >= 2 and all(c["outputs"] == [] for c in nb["cells"] if c["cell_type"] == "code")
    for needle in ("B.run(", "B.pilot_report", "B.verdict", "B.load_arm", "B.replay(", "G.replay(", "B.print_verdict", "gemini_arc_b_all_"): assert needle in code, needle
    assert "G.run(" not in code and 'run(llm, make_env, "A"' not in code and 'run(llm, make_env, "K"' not in code                                # arms A and K are only read
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            body = "\n".join(l for l in "".join(c["source"]).split("\n") if not l.lstrip().startswith("!")); compile(body, "cell", "exec")


# ---------------------------------------------------------------- 6 front pages
def test_the_front_pages_say_what_was_built_and_that_nothing_was_called():
    cm = open(os.path.join(DOCS, "RAJNISH_CONCEPT_MAP.md")).read(); vf = open(os.path.join(DOCS, "VERIFIED.md")).read()
    assert "Next free: row 192, item 145, test 170, question 122" in cm and cm.count("Next free:") == 1 and "test 144 checks the ARC-AGI-3 test with the host-owned explorer, item 118" in cm and "\n118. **Gemini 3.8 Flash on ARC-AGI-3 games with a host-owned explorer" in cm
    assert "GEMINI_ARC_B_TEST.md" in vf and "gemini_arc_b.py" in vf and PAGE_SHA256 in vf
    i = cm.index("\n118. **") + 1; item = cm[i:cm.index("\n", i)]; assert "Status: run once on Colab" in item and "docs/GEMINI_ARC_B_TEST.md" in item and "test 144" in item and "Sir's writing is for Sir to confirm" in item and "17 reading lines" in item


if __name__ == "__main__":
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]; bad = 0
    for n, f in fns:
        try: f(); print("ok  ", n)
        except Exception: bad += 1; print("FAIL", n); traceback.print_exc()
    print(f"{len(fns) - bad}/{len(fns)} passed"); sys.exit(1 if bad else 0)
