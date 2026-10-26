"""Test 165 (item 140): the fifth forward check (docs/FORESEE_CHECK_5.md, axi/experiments/foresee5.py, colab/axi_foresee_check_5.ipynb): repeats, the gap, filling the gap, the delete test.

The first four checks asked where the piece goes before a move. This one reads what the record says about the moves that were played and the moves it cannot answer. This test checks, by hand and on small worlds built for it:
  1. with every part on and both additions off the counters are the third check's (test 163), counter for counter, on the invented worlds and on the small worlds;
  2. a repeat is counted by the PLACE of the piece, not by the screen (a world whose screen always changes has no whole-screen repeat and has the repeats by place);
  3. every move not predicted falls under exactly one reason (a click, a first try, tried once and moved, tried once and did not, tried twice or more and never moved anything);
  4. `once` predicts from one earlier try and `elim` predicts the one unit step no other action has made, only for an action never tried, only when exactly one is missing, and is wrong for a press;
  5. each of the five parts, deleted alone, changes the counts where it should, and `earlier` only tags (it never changes a prediction);
  6. the 8 reading lines at their boundaries, the page (its hash, its lines, what it says it will not show), the notebook (no key, no model, nine ways), the replay of arm C on the stand-in game, the front pages.
What it does NOT show: how the check reads on the real games. That is the Result of the run (Colab)."""
import hashlib, json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.ledger import Ledger
from axi.experiments import foresee3 as F3
from axi.experiments import foresee5 as F5
from axi.experiments import ledger_replay as LR
from axi.experiments import gemini_arc_c as C
from axi.experiments.mini_worlds import World, walk, PUSH_LEVEL, PAD_LEVEL
from tests import test_gemini_arc as T

HERE = os.path.dirname(os.path.abspath(__file__)); DOCS = os.path.join(HERE, "..", "..", "docs")
PAGE = os.path.join(DOCS, "FORESEE_CHECK_5.md"); NB = os.path.join(HERE, "..", "colab", "axi_foresee_check_5.ipynb")
PAGE_SHA256 = "ea25e024206aeb2882ec1bdd949e3fa3c0c11ddb824b7dbc129fc62dd858bff3"; RESULT_MARK = "\n## Result"
NEXT = "Next free: row 187, item 141, test 166, question 122"
KEYS = ("moves", "predicted", "step_hit", "cell_hit", "cell_said", "cell_right", "once_pred", "once_hit", "elim_pred", "elim_hit", "rep_pos", "rep_pos_null", "rep_pos_null_null", "rep_screen", "did_not_move", "null_moves", "abstained")
get = lambda c: {k: c.get(k, 0) for k in KEYS}


def row(x, n=0, w=12):
    """One row world: the piece 'P' at column x of row 1, and a counter bar of n '#' along row 0 (so the screen is new every move when n changes)."""
    return tuple("".join("P" if (c, r) == (x, 1) else ("#" if r == 0 and c < n else ".") for c in range(w)) for r in range(3))


def tall(x, w=9, wall=6, door=True):
    return tuple("".join("a" if (c == x and r in (1, 2)) else ("#" if (c == wall and r in (1, 2) and door) else ".") for c in range(w)) for r in range(4))


def tall_seq(stops=5): return [("R", tall(i), tall(i + 1)) for i in range(1, 5)] + [("R", tall(5), tall(5))] * stops


def grid(x, y, w=8, h=8):
    return tuple("".join("P" if (c, r) == (x, y) else "." for c in range(w)) for r in range(h))


# ---------------------------------------------------------------- 1 every part on, both additions off: the third check
def test_with_every_part_on_and_both_additions_off_the_counters_are_the_third_checks_on_the_invented_worlds():
    for mk in (lambda: World(PUSH_LEVEL, pads=False), lambda: World(PAD_LEVEL)):
        for lv in range(1, 6):
            seq = walk(mk(), lv, 60); a = F3.evaluate(seq, True, True); b = F5.evaluate(seq)
            assert a and all(b.get(k, 0) == v for k, v in a.items())
    for seq in (tall_seq(), tall_seq(12), [("S", row(i), row(i + 1)) for i in range(1, 5)] + [("S", row(5), row(5))] * 8):
        a = F3.evaluate(seq, True, True); b = F5.evaluate(seq); assert all(b.get(k, 0) == v for k, v in a.items())


def test_the_check_is_a_pure_function_and_changes_nothing_it_is_given():
    seq = walk(World(PAD_LEVEL), 3, 120); snap = json.dumps(seq); a = F5.evaluate(seq, once=True, elim=True); assert json.dumps(seq) == snap
    assert a == F5.evaluate(list(seq), once=True, elim=True) and F5.text({}) == "" and F5.rates({})["rep_share"] is None and F5.rates({})["elim_precision"] is None
    assert list(F5.VARIANTS) == ["full", "-front", "-typed", "-edge", "-floor", "-group", "once", "elim", "once+elim"] and F5.THIRD_FULL == (603, 729) and F5.PARTS == ("front", "typed", "edge", "floor", "group")
    led = Ledger(); [led.record(*m) for m in seq]; before = (len(led.rows), led.dominant()); F5.evaluate(seq); assert (len(led.rows), led.dominant()) == before     # the ledger of the caller is not touched


# ---------------------------------------------------------------- 2 a repeat is counted by place, not by screen
def test_the_same_action_from_the_same_place_is_a_repeat_even_when_the_screen_is_new_every_move():
    seq = [("S", row(i, n=i), row(i + 1, n=i + 1)) for i in range(1, 5)] + [("S", row(5, n=n), row(5, n=n + 1)) for n in range(5, 11)]        # four steps, then six stops at the same place, a counter bar growing all the time
    c = get(F5.evaluate(seq))
    assert c["moves"] == 10 and c["predicted"] == 8 and c["did_not_move"] == 6 and c["rep_screen"] == 0                                    # the screen is new every move: no whole-screen repeat at all
    assert c["rep_pos"] == 5 and c["rep_pos_null"] == 5 and c["rep_pos_null_null"] == 5                                                      # the first stop is new; the next five repeat it
    assert abs(F5.rates(F5.evaluate(seq))["rep_share"] - 5 / 6) < 1e-9
    same = [("S", row(i), row(i + 1)) for i in range(1, 5)] + [("S", row(5), row(5))] * 6                                                    # the same stops with a screen that never changes
    d = get(F5.evaluate(same)); assert d["rep_screen"] == 5 and d["rep_pos_null_null"] == 5 and d["null_moves"] == 6


def test_a_place_the_piece_left_is_not_a_repeat_of_a_stop_and_a_move_from_a_stop_place_is_counted_once():
    # the piece stops at x=5 (twice), then the door of the world opens and it moves on from the same place: that move is a repeat of a place where it had not moved, but the piece did move
    seq = [("S", row(i), row(i + 1)) for i in range(1, 5)] + [("S", row(5), row(5))] * 2 + [("S", row(5), row(6))]
    c = get(F5.evaluate(seq)); assert (c["rep_pos"], c["rep_pos_null"], c["rep_pos_null_null"]) == (2, 2, 1)                                   # 2nd stop: repeat and null again; 3rd try (it moves): a repeat of a null place, not null again
    assert c["did_not_move"] == 2 and c["predicted"] == 5


# ---------------------------------------------------------------- 3 the gap, by reason
def test_every_move_not_predicted_is_under_exactly_one_reason():
    f0 = row(1); f1 = row(2); f2 = row(3); f3 = row(4)
    seq = [("S", f0, f1), ("S", f1, f2), ("S", f2, f3), ("T", f3, f3), ("T", f3, f3), ("T", f3, f3), ("C@1,1", f3, f3), ("S", f3, row(5))]
    c = get(F5.evaluate(seq)); raw = F5.evaluate(seq)
    assert c["moves"] == 8 and c["predicted"] == 2 and c["abstained"] == 6 and sum(raw.get("abs:" + k, 0) for k in F5.REASONS) == c["abstained"]       # S's third and fourth tries are predicted
    assert {k: raw.get("abs:" + k, 0) for k in F5.REASONS} == {"click": 1, "new": 2, "once_moved": 1, "once_nomove": 1, "nolaw": 1}                  # S new, S tried once and moved, T new, T tried once and did not, T tried twice and never moved, the click
    t = F5.evaluate(seq[:3] + [("T", f3, f3)] * 4); assert {k: t.get("abs:" + k, 0) for k in F5.REASONS} == {"click": 0, "new": 2, "once_moved": 1, "once_nomove": 1, "nolaw": 2}


def test_the_reasons_say_how_many_an_earlier_levels_law_would_have_answered_but_never_change_a_prediction():
    seq = [("S", row(i), row(i + 1)) for i in range(1, 5)]; earlier = {"S": ("P", (1, 0), 6, 6)}
    a = F5.evaluate(seq); b = F5.evaluate(seq, earlier=earlier)
    assert {k: v for k, v in b.items() if not k.startswith("abs_cov:")} == {k: v for k, v in a.items() if not k.startswith("abs_cov:")}
    assert b["abs_cov:new"] == 1 and b["abs_cov:once_moved"] == 1 and a.get("abs_cov:new", 0) == 0                                              # the first two tries would have been answered by the carried law
    other = F5.evaluate(seq, earlier={"S": ("Q", (1, 0), 6, 6)}); assert other.get("abs_cov:new", 0) == 0                                      # a law whose symbol is not on the screen answers nothing


# ---------------------------------------------------------------- 4 the two additions
def test_once_predicts_from_one_earlier_try_and_the_third_check_waits_for_two():
    seq = [("S", row(i), row(i + 1)) for i in range(1, 5)]
    off = get(F5.evaluate(seq)); on = get(F5.evaluate(seq, once=True))
    assert (off["predicted"], off["cell_hit"], off["once_pred"]) == (2, 2, 0) and (on["predicted"], on["cell_hit"], on["once_pred"], on["once_hit"]) == (3, 3, 1, 1)
    assert F5.evaluate(seq, once=True).get("abs:once_moved", 0) == 0 and F5.evaluate(seq).get("abs:once_moved", 0) == 1
    nomove = [("T", row(1), row(1)), ("T", row(1), row(1))]; assert get(F5.evaluate(nomove, once=True))["predicted"] == 0                          # an action that moved nothing has no step to predict from


def _four_way(new):
    """U, D and L each tried once (the piece ends at column 2, row 3), then a new action `new` = (name, screen after): the one unit step missing is (1, 0)."""
    return [("U", grid(3, 3), grid(3, 2)), ("D", grid(3, 2), grid(3, 3)), ("L", grid(3, 3), grid(2, 3)), (new[0], grid(2, 3), new[1])]


def test_elim_predicts_the_one_missing_unit_step_for_an_action_never_tried_and_is_wrong_for_a_press():
    mv = _four_way(("R", grid(3, 3)))                                                      # the new action R really steps right
    c = get(F5.evaluate(mv, elim=True)); assert (c["predicted"], c["elim_pred"], c["elim_hit"], c["cell_hit"]) == (1, 1, 1, 1) and get(F5.evaluate(mv))["predicted"] == 0
    press = _four_way(("A", grid(2, 3)))                                                   # the new action A is a press: nothing moves
    p = get(F5.evaluate(press, elim=True)); assert (p["predicted"], p["elim_pred"], p["elim_hit"], p["cell_hit"]) == (1, 1, 0, 0)
    two = mv[:2] + [("R", grid(3, 3), grid(4, 3))]                                          # only U and D seen: two steps missing, nothing is guessed
    assert get(F5.evaluate(two, elim=True))["elim_pred"] == 0
    tried = mv[:3] + [("R", grid(2, 3), grid(2, 3)), ("R", grid(2, 3), grid(2, 3))]       # R is eliminated at its first try only (wrong: it did nothing); at its second try it has been tried, so it is not new
    assert (get(F5.evaluate(tried, elim=True))["elim_pred"], get(F5.evaluate(tried, elim=True))["elim_hit"]) == (1, 0)
    e = F5._elimination(Ledger()); assert e is None
    led = Ledger(); [led.record(*m) for m in mv[:3]]; assert F5._elimination(led)[:2] == ("P", (1, 0))


def test_the_main_steps_of_a_level_are_the_unit_steps_each_action_made_on_the_main_mover():
    led = Ledger(); seq = [("U", grid(3, 3), grid(3, 2)), ("U", grid(3, 2), grid(3, 1)), ("D", grid(3, 1), grid(3, 2)), ("D", grid(3, 2), grid(3, 3)), ("X", grid(3, 3), grid(3, 3))]
    [led.record(*m) for m in seq]; assert F5._main_steps(led) == {"U": (0, -1), "D": (0, 1)}                                                    # X moved nothing; a step seen once is not enough
    c = F5.evaluate(seq); assert c["step:U:0,-1"] == 1 and c["step:D:0,1"] == 1 and c["rel:opposite"] == 1
    st = F5.action_steps(F5.combine([c, c])); assert st == {"U": {"0,-1": 2}, "D": {"0,1": 2}}


# ---------------------------------------------------------------- 5 the delete test
def test_deleting_the_edge_the_front_or_the_typing_alone_changes_the_counts_where_it_should():
    edge = [("S", row(i, w=6), row(i + 1, w=6)) for i in range(1, 5)] + [("S", row(5, w=6), row(5, w=6))] * 4                                    # the piece reaches the last column, then keeps pressing
    full = F5.evaluate(edge); no = F5.evaluate(edge, edge=False); g = lambda c: tuple(c.get(k, 0) for k in ("cell_said", "cell_hit", "wall_said", "typed_said"))
    assert g(full) == (4, 6, 3, 1) and g(no) == (1, 3, 0, 1)                                       # the edge makes three of the four statements "does not move" (all right); the fourth is the action typed. Without the edge those three stops are predicted as steps and missed
    f = F5.evaluate(tall_seq()); nf = F5.evaluate(tall_seq(), front=False); assert f["cell_hit"] == 6 and nf["cell_hit"] == 2 and f["held_cells"] == 2 and nf.get("held_cells", 0) == 0
    stop = [("S", row(i), row(i + 1)) for i in range(1, 5)] + [("S", row(5), row(5))] * 8
    t = get(F5.evaluate(stop)); nt = get(F5.evaluate(stop, typed=False)); assert t["cell_hit"] == 7 and nt["cell_hit"] == 2 and t["cell_said"] == 5 and nt["cell_said"] == 0


def test_deleting_the_floor_or_the_group_changes_the_counts_on_the_invented_worlds_and_nothing_else_is_touched():
    out = {}
    for nm, mk in (("pad", lambda: World(PAD_LEVEL)), ("push", lambda: World(PUSH_LEVEL, pads=False))):
        seqs = [walk(mk(), lv, 60) for lv in range(1, 6)]
        out[nm] = {v: F5.combine([F5.evaluate(s, **fl) for s in seqs]) for v, fl in F5.VARIANTS.items()}
    hits = lambda nm: {v: c["cell_hit"] for v, c in out[nm].items()}
    assert hits("pad") == {"full": 164, "-front": 166, "-typed": 164, "-edge": 164, "-floor": 165, "-group": 166, "once": 178, "elim": 164, "once+elim": 178}
    assert hits("push") == {"full": 162, "-front": 164, "-typed": 162, "-edge": 162, "-floor": 165, "-group": 159, "once": 176, "elim": 162, "once+elim": 176}
    for nm in out:
        assert all(c["predicted"] == 193 for v, c in out[nm].items() if v in ("full", "-front", "-typed", "-edge", "-floor", "-group"))                  # a deletion changes what is said, not which moves are predicted
        assert out[nm]["once"]["predicted"] == 206 and out[nm]["once"]["once_pred"] == 13 and out[nm]["elim"]["elim_pred"] == 1 and out[nm]["elim"]["elim_hit"] == 0


# ---------------------------------------------------------------- 6 the reading lines
GAMES = ("g50t-1", "ka59-1", "ls20-1", "sp80-1")


def _c(predicted=182, hit=150, dnm=40, rep=10, **kw):
    d = dict(moves=234, predicted=predicted, cell_hit=hit, did_not_move=dnm, rep_pos_null_null=rep, step_hit=hit, cell_said=0, cell_right=0, known=predicted)
    d.update(kw); return d


def _steps(**kw):
    base = {f"step:ACTION{i}:{s}": 10 for i, s in ((1, "0,-1"), (2, "0,1"), (3, "-1,0"), (4, "1,0"))}; base.update(kw); return base


def _full(predicted=182, hit=150, dnm=40, rep=10):
    out = {g: _c(predicted=predicted if i < 3 else predicted + 1, hit=hit, dnm=dnm, rep=rep) for i, g in enumerate(GAMES)}
    out["g50t-1"].update(_steps()); return out


def _v(full=None, both=None, elim=None, dels=None, k=None, c=None, ok=True):
    full = full or _full(); v = {"full": full, "once+elim": both or full, "elim": elim or full}
    for p in F5.PARTS: v["-" + p] = (dels or {}).get(p) or {g: dict(x, cell_hit=x["cell_hit"] - 6) for g, x in full.items()}
    arms = None
    if k is not None and c is not None: arms = {"K": {"full": {"g": k}}, "C": {"full": {"g": c}}}
    return {i: x for i, _, _, x in F5.verdict(v, ok, arms)}


def test_the_eight_lines_flip_at_their_boundaries_and_the_headline_needs_lines_3_and_5():
    assert len(F5.G_LINES) == 8 and [t for t, _ in F5.G_LINES] == ["guard", "instrument"] + ["claim"] * 6
    assert {i: x for i, _, _, x in F5.verdict(None, None)} == {i: None for i in range(1, 9)} and _v(ok=False)[1] is False
    assert _v(full=_full(predicted=182, hit=150))[2] is False                                                                                      # 729 predicted but 600 right: the instrument is 603 of 729
    good = {g: _c(predicted=182 if i < 3 else 183, hit=151 if i < 3 else 150) for i, g in enumerate(GAMES)}; assert _v(full=good)[2] is True            # 151 * 3 + 150 = 603 of 729
    assert _v(full=_full(rep=10, dnm=50))[3] is True and _v(full=_full(rep=9, dnm=50))[3] is False                                                 # 40 of 200 = 20 percent; 36 of 200 is not
    assert _v(full=_full(rep=0, dnm=0))[3] is False                                                                                                # no stop at all: nothing to read
    kk, cc = _c(dnm=50, rep=20), _c(dnm=50, rep=10); assert _v(k=kk, c=cc)[4] is True and _v(k=cc, c=kk)[4] is False and _v(k=kk, c=kk)[4] is False        # strictly lower in arm C
    base = _full(); mk = lambda add, addh: {g: _c(predicted=x["predicted"] + add, hit=x["cell_hit"] + addh) for g, x in base.items()}                    # per game: the figures below are 4 times these
    assert _v(both=mk(8, 6))[5] is True                                                                                                            # 32 more predicted, 24 right: 75 percent
    assert _v(both=mk(8, 5))[5] is False                                                                                                           # 32 more, 20 right: 62.5 percent
    assert _v(both=mk(7, 7))[5] is False                                                                                                           # 28 more: below 30
    el = lambda n, h: {g: dict(_c(), elim_pred=n if i == 0 else 0, elim_hit=h if i == 0 else 0) for i, g in enumerate(GAMES)}
    assert _v(elim=el(8, 6))[6] is True and _v(elim=el(8, 5))[6] is False and _v(elim=el(7, 7))[6] is False
    drop = lambda n: {g: dict(x, cell_hit=x["cell_hit"] - (n if i == 0 else 0)) for i, (g, x) in enumerate(base.items())}
    assert _v(dels={p: drop(5) for p in F5.PARTS})[7] is True and _v(dels={**{p: drop(5) for p in F5.PARTS}, "group": drop(4)})[7] is False        # each part must cost at least 5
    assert _v()[8] is True
    bad = _full(); bad["g50t-1"].update(_steps(**{"step:ACTION2:1,0": 2})); assert _v(full=bad)[8] is False                                          # 10 of 12 = 83 percent for ACTION2
    few = _full(); few["g50t-1"].update(_steps(**{"step:ACTION1:0,-1": 3})); assert _v(full=few)[8] is False                                                                                                                  # ACTION1 made a step in only 3 levels: fewer than 4
    one = _full(); one["g50t-1"] = {k: v for k, v in one["g50t-1"].items() if not k.startswith("step:ACTION4")}; assert _v(full=one)[8] is False                # an action never seen making a step: not yes
    allyes = [(i, t, x, True) for i, t, x, _ in F5.verdict(None, None)]
    assert F5.headline(allyes) == F5.HEADLINE and F5.claims_allowed(allyes)
    assert F5.headline([(i, t, x, (False if i == 5 else v)) for i, t, x, v in allyes]) == "MIXED: line 3 yes, line 5 NO"
    assert F5.headline([(i, t, x, (False if i == 3 else v)) for i, t, x, v in allyes]) == "MIXED: line 3 NO, line 5 yes"
    assert F5.headline([(i, t, x, (False if i == 2 else v)) for i, t, x, v in allyes]).startswith("NOT READ") and F5.headline(F5.verdict(None, None)).startswith("NOT READ")


# ---------------------------------------------------------------- 7 page, notebook, replay of arm C, front pages
def _head():
    text = open(PAGE, encoding="utf-8").read(); assert text.count(RESULT_MARK) <= 1
    return text[:text.index(RESULT_MARK)] if RESULT_MARK in text else text


def test_the_page_above_its_result_is_what_was_written_before_the_run():
    assert hashlib.sha256(_head().encode()).hexdigest() == PAGE_SHA256


def test_the_page_and_the_module_agree_and_it_says_which_text_is_whose():
    head = _head(); pred = head.split("## Predictions")[1].split("## Reading rules")[0]; rules = head.split("## Reading rules")[1].split("Headline:")[0]
    ps = re.findall(r"^(\d+)\. \[(\w+)\] (.*) \((\d+) %\)$", pred, re.M); rs = re.findall(r"^(\d+)\. (.*)$", rules, re.M)
    assert len(ps) == len(rs) == len(F5.G_LINES) == 8
    for (n, tag, text, pr), (m, rt), (gt, gtext) in zip(ps, rs, F5.G_LINES): assert n == m and text == rt == gtext and tag == gt and 0 < int(pr) < 100
    for needle in (F5.HEADLINE, "Sir to confirm", "Reset. Repeat. Pull again", "Saraswati Choubey", "steered the AI", "Delete Test", "603 of the 729", "207", "(0, 0 and 1 in 240", "208 to 227", "row D", "It is not", "will not show", "paid test", "$3.65",
                   "Result", "were run once on Colab", "not written yet", "Nine ways", "invented worlds"): assert needle in head, needle
    assert "$" in head and "proven" not in head.lower() and "ledger and the predictor of the third check are not changed" in head


def test_the_notebook_reads_the_saved_files_calls_no_model_holds_no_key_and_runs_nine_ways():
    nb = json.load(open(NB)); src = "\n".join("".join(c["source"]) for c in nb["cells"]); code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert len(nb["cells"]) == 12 and all(c["outputs"] == [] for c in nb["cells"] if c["cell_type"] == "code")
    assert not re.search(r"AIza[0-9A-Za-z_-]{20,}", src) and "GOOGLE_API_KEY" not in src and "userdata" not in src and "drive.mount" not in src and "open_store" not in src
    for needle in ("foresee5 as F", "F.from_session", "LR.run_file", "F.verdict", "F.headline", "F.VARIANTS", "per_arm if allok", "axi_gemini_arc_c_default.json", "axi_gemini_arc_b_default.json", "axi_gemini_arc_default.json", "foresee_check_5_all.json", "foresee5.py"): assert needle in code, needle
    assert "FORESEE_CHECK_5.md" in src and "foresee4" not in code and "generate" not in code and ".run(" not in code.replace("LR.run_file(", "")
    for c in nb["cells"]:
        if c["cell_type"] == "code": compile("\n".join(l for l in "".join(c["source"]).split("\n") if not l.lstrip().startswith("!")), "cell", "exec")


def test_the_fifth_check_runs_over_a_replay_of_arm_c_on_the_stand_in_game_nine_ways():
    res = C.run(T.FakeLLM(), T.make_env, os.path.join(os.environ.get("TMPDIR", "/tmp"), "foresee5_c_test.jsonl"), games=("alpha-1", "beta-2"), moves=30, action_of=lambda n: n)
    s = LR.feed(res["rows"], T.make_env, "C", games=("alpha-1", "beta-2"), moves=30, action_of=lambda n: n); w = F5.from_session(s)
    assert set(w) == set(F5.VARIANTS) and all(set(pg) == {"alpha-1", "beta-2"} for pg in w.values())
    assert all(sum(c["moves"] for c in pg.values()) == sum(s.moves.values()) for pg in w.values()) and "ALL GAMES" in F5.text(w) and "the delete test" in F5.text(w) and "the gap" in F5.text(w)
    for g in w["full"]:
        a = F3.combine([F3.evaluate(((r.action, r.kb, r.ka) for r in L.rows), True, True) for (gg, lv), L in sorted(s.ledgers.items()) if gg == g])
        assert all(w["full"][g].get(k, 0) == v for k, v in a.items())                                                                            # the full way is the third check on the replayed levels


def test_the_front_pages_say_what_was_built_and_that_it_was_not_run_on_the_real_games():
    cm = open(os.path.join(DOCS, "RAJNISH_CONCEPT_MAP.md"), encoding="utf-8").read(); vf = open(os.path.join(DOCS, "VERIFIED.md"), encoding="utf-8").read(); hl = open(os.path.join(DOCS, "HISTORY_LEDGER.md"), encoding="utf-8").read()
    assert NEXT in cm and cm.count("Next free:") == 1 and "\n140. **" in cm and "\n141. **" not in cm
    i = cm.index("\n140. **") + 1; item = cm[i:cm.index("\n", i)]
    for n in ("FORESEE_CHECK_5.md", "foresee5.py", "test 165", "Saraswati Choubey", "Sir to confirm", "NOT RUN", "8 reading lines", "row D", "Sir steered", "not a new idea", PAGE_SHA256): assert n in item, n
    assert any("FORESEE_CHECK_5.md" in l and "foresee5.py" in l and PAGE_SHA256 in l and "NOT RUN" in l for l in vf.split("\n"))
    assert "| 186 (O) |" in hl and "Sir" in hl and "elim" in hl


if __name__ == "__main__":
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]; bad = 0
    for n, f in fns:
        try: f(); print("ok  ", n)
        except Exception: bad += 1; print("FAIL", n); traceback.print_exc()
    print(f"{len(fns) - bad}/{len(fns)} passed"); sys.exit(1 if bad else 0)
