"""Test 160 (item 135): the instrument of the ARC-AGI-3 test with the ledger's read-out in the prompt (docs/GEMINI_ARC_C_TEST.md, axi/experiments/gemini_arc_c.py, colab/axi_gemini_arc_c.ipynb).

No model is called and no ARC package is imported: the stand-in game and stand-in model of test 143. The tests check the read-out against lines made by hand, that it is made from the moves before the move only (the replay
refuses a row whose number of ledger lines differs), that a level change or a RESET starts a new ledger, the turn and the system text, the arm C rows, the resume, the replay into fresh games (it refuses a doctored row), the 16 reading
lines at their boundaries, the page hash, the agreement of page and module, the notebook (no key, no budget, no billing, no other arm run) and the front pages. The Result of the real run will be its own test."""
import hashlib, io, json, os, re, sys, tempfile, contextlib
from axi.experiments import gemini_arc_c as C
from axi.experiments import gemini_arc_b as B
from axi.experiments import gemini_arc as R
from tests import test_gemini_arc as T

HERE = os.path.dirname(os.path.abspath(__file__)); DOCS = os.path.join(HERE, "..", "..", "docs")
PAGE = os.path.join(DOCS, "GEMINI_ARC_C_TEST.md"); NB = os.path.join(HERE, "..", "colab", "axi_gemini_arc_c.ipynb")
PAGE_SHA256 = "424fbb3f2f41e734df5f49d950c1c019703e2d84fb4d24e7c09b16962fd35a56"; RESULT_MARK = "\n## Result"
NEXT = "Next free: row 190, item 143, test 168, question 122"
G2 = ("alpha-1", "beta-2"); IDENT = T.IDENT; make_env = T.make_env


def tmp(): return tempfile.mktemp(suffix=".jsonl")
def run_c(llm, path=None, games=G2, moves=30, make=None, **kw): return C.run(llm, make or make_env, path or tmp(), games=games, moves=moves, action_of=IDENT, **kw)


# ---------------------------------------------------------------- 1 the read-out
def test_the_readout_is_made_by_hand_and_says_counts_not_rules():
    p = C.PlayC(make_env("alpha-1"), "alpha-1", 30, IDENT); assert p.ledger_text() == [] and p.ledger_view() == C.LEDGER_HEAD + "\n(no move played yet in this level.)"
    p.apply(dict(action="ACTION4")); assert p.ledger_text() == ["- last move: colour 7 moved by (1,0)."]                                                              # one move: no step is given (two tries needed), no count of unchanged colours (three moves)
    p.apply(dict(action="ACTION4")); assert p.ledger_text() == ["- ACTION4: the object of colour 7 moved by (1,0) in 2 of its 2 tries.", "- last move: colour 7 moved by (1,0)."]
    p.apply(dict(action="ACTION5")); assert p.ledger_text() == ["- ACTION4: the object of colour 7 moved by (1,0) in 2 of its 2 tries.", "- colours that have not changed anywhere in the 3 moves of this level: 4.", "- last move: nothing changed."]
    p.apply(dict(action="ACTION2")); p.apply(dict(action="ACTION2")); t = p.ledger_text()
    assert t[:2] == ["- ACTION2: the object of colour 7 moved by (0,1) in 2 of its 2 tries.", "- ACTION4: the object of colour 7 moved by (1,0) in 2 of its 2 tries."] and t[-1] == "- last move: colour 7 moved by (0,1)."
    assert all(l.startswith("- ") for l in t) and len(t) <= C.LEDGER_MAX_LINES and "rule" not in " ".join(t).lower()
    p.apply(dict(action="ACTION1")); assert any(l.startswith("- ACTION1") and "in 1 of its 1 tries" not in l for l in p.ledger_text()) is False                          # one try: no step is given
    q = C.PlayC(make_env("alpha-1"), "alpha-1", 30, IDENT)
    for a in ("ACTION4", "ACTION3", "ACTION4"): q.apply(dict(action=a))
    assert "- ACTION3" not in " ".join(q.ledger_text())                                                                                                                  # ACTION3 tried once only
    c = C.PlayC(make_env("alpha-1"), "alpha-1", 30, IDENT)
    for a in ("ACTION4", "ACTION4"): c.apply(dict(action=a))
    c.apply(dict(action="ACTION6", x=0, y=0)); assert "ACTION6" not in " ".join(c.ledger_text())                                                                          # clicks are left out


def test_the_readout_names_a_bar_and_leaves_out_its_own_change():
    p = C.PlayC(make_env("alpha-1"), "alpha-1", 60, IDENT); L = p.ledgers.setdefault(0, C.Ledger(min_law=C.MIN_LAW))
    z = lambda k: ("." * 40, "z" * k + "." * (40 - k))                                                                                         # a strip along the bottom that is used up one cell a move
    for k in range(20, 8, -1): L.record("ACTION1", z(k), z(k - 1))
    t = p.ledger_text(); assert any("colour z gets smaller by 1 cell(s) in 12 of the 12 moves that changed the screen" in l and "probably a bar or counter" in l for l in t), t
    assert not any("changed colour" in l for l in t)


def test_a_new_level_and_a_reset_start_or_skip_the_ledger():
    class Walk(T.FakeLLM):
        def generate(self, messages, constrained=True, max_new_tokens=400):
            super().generate(messages, constrained, max_new_tokens); n = self.n % 12
            return ('{"action": "RESET"}' if n == 8 else '{"action": "ACTION4"}' if n % 5 else '{"action": "ACTION2"}'), 8, 0.4
    res = run_c(Walk(), moves=30); rows = res["rows"]; plays = C.replay(rows, make_env, G2, 30, IDENT); p = plays[0]
    assert len(p.ledgers) >= 2 and sorted(p.ledgers) == list(range(len(p.ledgers))) and all(r.action != "RESET" for L in p.ledgers.values() for r in L.rows)
    assert p.ledgers[0].rows and all(r.action != "RESET" for r in p.ledgers[0].rows)
    q = C.PlayC(make_env("alpha-1"), "alpha-1", 30, IDENT)
    for a in ("ACTION4", "ACTION4", "ACTION4"): q.apply(dict(action=a))
    assert q.levels() == 1 and q.ledger_text() == [] and 0 in q.ledgers and len(q.ledgers[0].rows) == 2                                                         # the move that won is not a transition; the new level is empty


def test_the_readout_of_a_move_is_a_function_of_the_moves_before_it():
    moves = [dict(action=a) for a in ("ACTION4", "ACTION2", "ACTION4", "ACTION5", "ACTION2", "ACTION1", "ACTION2")]
    for k in range(len(moves) + 1):
        a = C.PlayC(make_env("alpha-1"), "alpha-1", 60, IDENT); b = C.PlayC(make_env("alpha-1"), "alpha-1", 60, IDENT)
        for m in moves[:k]: a.apply(m); b.apply(m)
        for m in (dict(action="ACTION3"), dict(action="ACTION3"), dict(action="ACTION4")): b.apply(m)                                                                    # b goes on; a stops: what a is shown at move k
        again = C.PlayC(make_env("alpha-1"), "alpha-1", 60, IDENT)
        for m in moves[:k]: again.apply(m)
        assert a.ledger_text() == again.ledger_text()


def test_the_turn_holds_the_record_the_objects_the_untried_the_ledger_and_then_the_screen():
    p = C.PlayC(make_env("alpha-1"), "alpha-1", 30, IDENT)
    for a in ("ACTION4", "ACTION4", "ACTION5"): p.apply(dict(action=a))
    t = p.turn("C"); i = [t.index(x) for x in ("Record kept by the framework", "Objects on the screen", "Untried from this exact screen", "Ledger of this level so far", "Screen:\n", "Offered actions: ACTION1")]
    assert i == sorted(i) and t.endswith("Reply with one JSON object.") and "- ACTION4: the object of colour 7 moved by (1,0) in 2 of its 2 tries." in t
    assert C.SYSTEM_C.startswith(B.SYSTEM_B) and "Ledger section" in C.SYSTEM_C and "not rules" in C.SYSTEM_C and C.SYSTEM_C != B.SYSTEM_B


# ---------------------------------------------------------------- 2 the arm
def test_arm_c_flags_every_move_holds_the_number_of_ledger_lines_and_replays():
    llm = T.FakeLLM(bad=(5,)); res = run_c(llm); rows = res["rows"]; assert len(rows) == 60 and res["stopped"] is None and [r["t"] for r in rows] == list(range(1, 61))
    forced = [r for r in rows if r["forced"]]; assert forced and all(r["seconds"] == 0.0 and r["tokens"] == 0 and r["finish"] == "FRAMEWORK" and "ledger_lines" in r for r in forced)
    assert all(isinstance(r["ledger_lines"], int) and 0 <= r["ledger_lines"] <= C.LEDGER_MAX_LINES for r in rows) and rows[0]["ledger_lines"] == 0 and any(r["ledger_lines"] >= 3 for r in rows)
    mr = [x for x in llm.seen if not x[-1]["content"].startswith("That move")]; assert all(x[0]["content"] == C.SYSTEM_C and "Ledger of this level so far" in x[-1]["content"] for x in mr)
    s = C.summarize(rows); assert s["ledger_shown"] is not None and 0.9 < s["ledger_shown"] <= 1.0 and s["ledger_lines_mean"] > 2 and s["moves"] == 60
    assert C.summarize([{k: v for k, v in r.items() if k != "ledger_lines"} for r in rows])["ledger_shown"] is None                                                 # rows of the older arms say nothing about a read-out
    C.replay(rows, make_env, G2, 30, IDENT)


def test_a_stopped_run_resumes_to_the_same_rows_and_a_foreign_file_is_refused():
    full = run_c(T.FakeLLM()); p = tmp(); part = run_c(T.FakeLLM(), p, max_tokens=12_000); n = len(part["rows"]); assert part["stopped"] == "token_cap" and 0 < n < 60
    rest = run_c(T.FakeLLM(), p); assert rest["rows"] == full["rows"] and rest["stopped"] is None
    again = T.FakeLLM(); assert run_c(again, p)["rows"] == full["rows"] and again.n == 0
    try: run_c(T.FakeLLM(), p, moves=29); raise AssertionError("another run's file accepted")
    except ValueError: pass
    q = tmp(); b = B.run(T.FakeLLM(), make_env, q, games=G2, moves=30, action_of=IDENT)
    try: run_c(T.FakeLLM(), q); raise AssertionError("arm B's file accepted as arm C's")
    except ValueError: pass


def test_the_replay_refuses_a_doctored_file_a_doctored_ledger_count_and_a_missing_framework_move():
    res = run_c(T.FakeLLM(), moves=20); rows = res["rows"]; ix = next(i for i, r in enumerate(rows) if r["forced"]); jx = next(i for i, r in enumerate(rows) if not r["forced"] and i > ix and r["ledger_lines"] > 0)
    def refuses(rs, why):
        try: C.replay(rs, make_env, G2, 20, IDENT); raise AssertionError("accepted " + why)
        except R.ReplayMismatch: pass
    for key in ("changed", "cells", "new_screen", "revisit", "repeat_noop", "levels", "ledger_lines"):
        rs = [dict(r) for r in rows]; v = rs[jx][key]; rs[jx][key] = (not v) if isinstance(v, bool) else v + 1; refuses(rs, key)
    rs = [dict(r) for r in rows]; rs[jx]["ledger_lines"] = 0; refuses(rs, "a read-out that was not shown")
    rs = [dict(r) for r in rows]; del rs[jx]["ledger_lines"]; refuses(rs, "a row with no count")
    rs = [dict(r) for r in rows]; rs[ix]["forced"] = False; refuses(rs, "a framework move marked as the model's")
    rs = [dict(r) for r in rows]; rs[jx], rs[jx + 1] = rs[jx + 1], rs[jx]; refuses(rs, "rows out of order")
    refuses([dict(r) for r in B.run(T.FakeLLM(), make_env, tmp(), games=G2, moves=20, action_of=IDENT)["rows"]], "the rows of arm B (they hold no ledger count)")


# ---------------------------------------------------------------- 3 the reading lines
def _row(t, g, m, eff=True, ok=True, rep=False, rev=False, forced=False, lv=0, sec=3.0, think=100, prompt=5000, new=None, ll=3):
    new = (eff and not rev) if new is None else new
    d = dict(t=t, g=g, game=C.GAMES[g], m=m, text="{}", kind="ACTION1", tokens=10, think=think, prompt_tokens=prompt, cache_tokens=0, first_token=1.0, seconds=sec, from_disk=False, finish="STOP", parsed=True, reply_ok=ok,
             reply_exact=eff and ok, moved=ok, changed=eff, cells=1, repeat_noop=rep, revisit=rev, state="NOT_FINISHED", levels=lv, win_levels=3, forced=forced, new_screen=new, ledger_lines=ll)
    if forced: d.update(tokens=0, think=0, prompt_tokens=0, seconds=0.0, from_disk=True, finish="FRAMEWORK")
    return d


def arm(**kw):
    rows = []
    for g in range(4):
        for m in range(1, 61): rows.append(_row(len(rows) + 1, g, m, **{k: (v(g, m) if callable(v) else v) for k, v in kw.items()}))
    return dict(rows=rows, stopped=None)


def V(c=None, b=None, a=None, k=None, rp=None): return {i: v for i, _, _, v in C.verdict(c, b, a, k, rp)}
OKR = {"C": True, "B": True}


def test_the_reading_lines_flip_at_their_boundaries_and_the_headline_needs_lines_6_and_8():
    assert len(C.G_LINES) == 16 and [t for t, _ in C.G_LINES[:5]] == ["guard", "guard", "instrument", "instrument", "instrument"]
    b = arm(eff=lambda g, m: m % 20 != 0, rev=lambda g, m: m % 7 == 0); c = arm(eff=True); base = V(c, b, None, None, OKR)
    assert [base[i] for i in (1, 2, 3, 4, 5)] == [True] * 5 and base[15] is None and base[16] is None and V(None, None) == {i: None for i in range(1, 17)}
    assert V(c, b, None, None, {"C": False, "B": True})[1] is False and V(c, b, None, None, {"C": True, "B": False})[2] is False and V(c, b, None, None, {"C": True})[2] is False
    cut = dict(rows=c["rows"][:-1], stopped=None); assert V(cut, b, None, None, OKR)[1] is False
    fz = lambda n: arm(forced=lambda g, m: g * 60 + m <= n); assert V(fz(96), b, None, None, OKR)[3] is True and V(fz(97), b, None, None, OKR)[3] is False                          # 40 percent of 240
    assert V(arm(ok=lambda g, m: not (g == 0 and m <= 4)), b, None, None, OKR)[4] is True and V(arm(ok=lambda g, m: not (g == 0 and m <= 5)), b, None, None, OKR)[4] is False         # 4.8 of 240 allowed
    sh = lambda n: arm(ll=lambda g, m: 0 if g * 60 + m <= n else 3); assert V(sh(48), b, None, None, OKR)[5] is True and V(sh(49), b, None, None, OKR)[5] is False                  # 80 percent of 240 shown
    ae = lambda n: arm(eff=lambda g, m: (g * 60 + m) > n); assert V(ae(5), ae(5), None, None, OKR)[6] is True and V(ae(6), ae(5), None, None, OKR)[6] is False and V(ae(4), ae(5), None, None, OKR)[6] is True
    rp = lambda n: arm(rep=lambda g, m: g * 60 + m <= n); assert V(rp(3), rp(3), None, None, OKR)[7] is True and V(rp(4), rp(3), None, None, OKR)[7] is False
    sc = lambda p: arm(prompt=p, new=True, think=0); assert V(sc(989), sc(1090), None, None, OKR)[8] is True and V(sc(991), sc(1090), None, None, OKR)[8] is False                    # 1.1 times
    ds = lambda n: arm(rev=lambda g, m: g * 60 + m <= n); assert V(ds(10), ds(10), None, None, OKR)[9] is True and V(ds(11), ds(10), None, None, OKR)[9] is False
    lv = lambda x: arm(lv=lambda g, m: x if g == 0 else 0)
    assert V(lv(1), lv(1), None, None, OKR)[10] is True and V(lv(0), lv(1), None, None, OKR)[10] is False and V(lv(1), lv(0), None, None, OKR)[11] is True and V(lv(0), lv(1), None, None, OKR)[11] is False
    assert V(lv(2), lv(1), None, None, OKR)[12] is True and V(lv(1), lv(1), None, None, OKR)[12] is False
    th = lambda x: arm(think=x); assert V(th(110), th(100), None, None, OKR)[13] is True and V(th(112), th(100), None, None, OKR)[13] is False
    cm = lambda p: arm(prompt=p, think=0); assert V(cm(1190), cm(1000), None, None, OKR)[14] is True and V(cm(1210), cm(1000), None, None, OKR)[14] is False                        # 1.2 times
    assert V(lv(1), b, lv(1), None, OKR)[15] is True and V(lv(0), b, lv(1), None, OKR)[15] is False and V(c, b, None, None, OKR)[15] is None
    kk = lv(1); assert V(lv(1), b, None, kk, OKR)[16] is True and V(lv(0), b, None, kk, OKR)[16] is False
    chk = C.verdict(arm(forced=lambda g, m: m <= 5, prompt=1000, new=True), arm(eff=lambda g, m: m % 5 != 0, prompt=3000), None, None, OKR); h = C.headline(chk)
    assert h == C.HEADLINE or h.startswith("MIXED")
    full = [(i, t, x, True) for i, t, x, v in chk]; assert C.headline(full) == C.HEADLINE and C.claims_allowed(full)["base"] is True
    assert C.headline([(i, t, x, (False if i == 8 else True)) for i, t, x, v in chk]).startswith("MIXED: line 6 yes, line 8 NO") and C.headline([(i, t, x, (False if i == 6 else True)) for i, t, x, v in chk]).startswith("MIXED: line 6 NO")
    assert C.headline(C.verdict(c, None)).startswith("NOT READ")
    for bad in (1, 2, 4, 5): assert C.claims_allowed([(i, t, x, (False if i == bad else True)) for i, t, x, v in chk])["base"] is False, bad
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): C.print_stats("C", c["rows"]); C.print_games(c["rows"]); C.print_verdict(full)
    assert "headline:" in buf.getvalue() and "ledger read-out shown in 100.0% of the model's moves" in buf.getvalue()


def test_the_pilot_report_and_the_arm_loader():
    rows = arm(forced=lambda g, m: m % 4 == 0)["rows"][:12]; r = C.pilot_report(rows, 1.0, 10.0, 0.1)
    assert abs(r["c_forced_share"] - 0.25) < 1e-12 and r["c_prompt_per_model_move"] == 5000 and r["cost_c"] > 0 and C.pilot_report(rows)["cost_c"] is None and r["c_ledger_lines_mean"] == 3
    p = tmp(); res = run_c(T.FakeLLM(), p, moves=5, games=("alpha-1",)); got = C.load_arm(p); assert got["rows"] == res["rows"] and got["stopped"] is None


# ---------------------------------------------------------------- 4 page and notebook
def _head():
    text = open(PAGE, encoding="utf-8").read(); assert text.count(RESULT_MARK) <= 1
    return text[:text.index(RESULT_MARK)] if RESULT_MARK in text else text


def test_the_page_above_its_result_is_what_was_written_before_the_run():
    assert hashlib.sha256(_head().encode()).hexdigest() == PAGE_SHA256


def test_the_page_and_the_module_agree():
    head = _head(); pred = head.split("## Predictions")[1].split("## Reading rules")[0]; rules = head.split("## Reading rules")[1].split("Headline:")[0]
    ps = re.findall(r"^(\d+)\. \[(\w+)\] (.*) \((\d+) %\)$", pred, re.M); rs = re.findall(r"^(\d+)\. (.*)$", rules, re.M)
    assert len(ps) == len(rs) == len(C.G_LINES) == 16
    for (n, tag, text, pr), (m, rt), (gt, gtext) in zip(ps, rs, C.G_LINES): assert n == m and text == rt == gtext and tag == gt and 0 < int(pr) < 100
    for needle in (C.MODEL, f"{C.MOVES} moves each", f"{C.C_TOKEN_CAP:,}", C.HEADLINE, "Energy in joules", "not an entry on the ARC-AGI-3 leaderboard", "marked \"(framework)\"", "Sir to confirm", "edits none of them", "At most 12 lines", *C.GAMES): assert needle in head, needle
    assert "$" not in head and "proven" not in head.lower() and "100 percent" not in head and "ultimate" not in head.lower()
    assert C.LEDGER_MAX_LINES == 12 and C.STEP_MIN_TRIES == 2 and C.MIN_LAW == 3 and "at least twice" in head and "last 4 moves" in head and "seen 3 times" in head and "Clicks are left out" in head
    assert head.index("## Predictions") < head.index("## Reading rules") and "nothing from any other run" in head.lower().replace("**", "") or "holds nothing from any other run" in head


def test_the_notebook_holds_no_key_never_turns_billing_on_sets_no_budget_and_never_runs_the_other_arms():
    nb = json.load(open(NB)); src = "\n".join("".join(c["source"]) for c in nb["cells"]); code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert len(nb["cells"]) == 17 and not re.search(r"AIza[0-9A-Za-z_-]{20,}", src) and "GOOGLE_API_KEY" in src and "userdata.get" in src and "gemini_arc_c.py" in code
    for v in ("PAID", "PRICE_IN", "PRICE_OUT", "PRICE_CACHED", "BUDGET_USD"): assert f"{v} = None" in code, v
    assert "enable billing" not in code.lower() and code.count("need_budget()") >= 2 and all(c["outputs"] == [] for c in nb["cells"] if c["cell_type"] == "code")
    for needle in ("C.run(", "C.pilot_report", "C.verdict", "C.load_arm", "C.replay(", "B.replay(", "G.replay(", "C.print_verdict", "gemini_arc_c_all_", "C.PILOT_GAME", "C.C_TOKEN_CAP"): assert needle in code, needle
    assert "G.run(" not in code and "B.run(" not in code and 'run(llm, make_env, "A"' not in code and 'run(llm, make_env, "K"' not in code                       # arms A, B and K are only read
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            body = "\n".join(l for l in "".join(c["source"]).split("\n") if not l.lstrip().startswith("!")); compile(body, "cell", "exec")


# ---------------------------------------------------------------- 5 front pages
def test_the_front_pages_say_what_was_built_and_that_nothing_was_called():
    cm = open(os.path.join(DOCS, "RAJNISH_CONCEPT_MAP.md"), encoding="utf-8").read(); vf = open(os.path.join(DOCS, "VERIFIED.md"), encoding="utf-8").read(); hl = open(os.path.join(DOCS, "HISTORY_LEDGER.md"), encoding="utf-8").read()
    assert NEXT in cm and cm.count("Next free:") == 1 and "\n135. **" in cm
    i = cm.index("\n135. **") + 1; item = cm[i:cm.index("\n", i)]
    assert "GEMINI_ARC_C_TEST.md" in item and "gemini_arc_c.py" in item and "test 160" in item and "Rajnish Choubey" in item and "Sir to confirm" in item and "NOT RUN" in item and "16 reading lines" in item and "no model" in item.lower()
    assert any("GEMINI_ARC_C_TEST.md" in l and "gemini_arc_c.py" in l and PAGE_SHA256 in l and "NOT RUN" in l for l in vf.split("\n"))
    assert "| 181 (J) |" in hl and "read-out" in hl and "arm C" in hl


if __name__ == "__main__":
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]; bad = 0
    for n, f in fns:
        try: f(); print("ok  ", n)
        except Exception: bad += 1; print("FAIL", n); traceback.print_exc()
    print(f"{len(fns) - bad}/{len(fns)} passed"); sys.exit(1 if bad else 0)
