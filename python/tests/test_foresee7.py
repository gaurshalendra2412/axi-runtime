"""Test 167 (item 142): the seventh forward check (docs/FORESEE_CHECK_7.md, axi/experiments/foresee7.py, colab/axi_foresee_check_7.ipynb, colab/run_foresee_check_7.py): a first try from the name, the arrangement of the four actions, the screen memory with its strip found online, the retire question, the edge.

The sixth check found that ACTION1 to ACTION4 kept their direction in every level of every game, that the rotation set predicted 2 moves (both wrong) and the carried relation 10, and that with the strip left out the same action on the same screen repeats 268 times in 936 moves. This one asks what that allows. This test checks, by hand and on small worlds built for it:
  1. with every new part off the counters are the sixth check's `evaluate`, counter for counter (and so the fifth's and the third's), on the invented worlds and on the small worlds; the screen memory changes no prediction;
  2. the helpers: the steps from one move on, the name tier (`nm`: direction from the name, length from a step seen) and the carried-length tier (`nc`), the arrangement class (one of 6, mirrored when two actions swap, none unless all four made a step of one length);
  3. `nm` and `nc` predict the first try of an action, are wrong for a press, predict nothing without a name's direction, nothing when the steps seen disagree in length, and are read after `rel` and before `rot`; the tiers add up to the predicted moves;
  4. the name prior is read from the OTHER games only, from at least 4 levels and 90 percent; a game never reads its own levels; the carried length comes from earlier levels only;
  5. the screen memory with the two masks (the strict one leaks the cell about to change, the symbol-wide one does not), the retire counters, the edge counters;
  6. the 12 reading lines at their boundaries, the page (its hash, its lines, its probabilities, what it says it will not show), the notebook and the runner (no key, no model, six ways), the replay of arm C on the stand-in game, the front pages.
What it does NOT show: how the check reads on the real games. That is the Result of the run (Codespaces or Colab)."""
import hashlib, json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.ledger import Ledger
from axi.experiments import foresee3 as F3
from axi.experiments import foresee5 as F5
from axi.experiments import foresee6 as F6
from axi.experiments import foresee7 as F7
from axi.experiments import ledger_replay as LR
from axi.experiments import gemini_arc_c as C
from axi.experiments.mini_worlds import World, walk, PUSH_LEVEL, PAD_LEVEL
from tests import test_gemini_arc as T
from tests.test_foresee6 import grid, row, three, led, Sess, tall_seq

HERE = os.path.dirname(os.path.abspath(__file__)); DOCS = os.path.join(HERE, "..", "..", "docs")
PAGE = os.path.join(DOCS, "FORESEE_CHECK_7.md"); NB = os.path.join(HERE, "..", "colab", "axi_foresee_check_7.ipynb"); RUN = os.path.join(HERE, "..", "colab", "run_foresee_check_7.py")
PAGE_SHA256 = "fbfe7110f9d738ac64c7fd2f95c19f06b5590ca0ee81a8ced4081c3b3aa14fbd"; RESULT_MARK = "\n## Result"
NEXT = "Next free: row 192, item 145, test 170, question 122"
KEYS = ("moves", "predicted", "cell_hit", "once_pred", "once_hit", "rot_pred", "rot_hit", "rel_pred", "rel_hit", "nm_pred", "nm_hit", "nc_pred", "nc_hit", "law_pred", "law_hit", "abstained", "scx_n", "scx_same", "scr_n", "scr_same", "scx_fill", "scx_fill_right")
get = lambda c: {k: c.get(k, 0) for k in KEYS}
A = ("ACTION1", "ACTION2", "ACTION3", "ACTION4")
UP, DOWN, LEFT, RIGHT = (0, -1), (0, 1), (-1, 0), (1, 0)
NAMES = {"U": UP, "D": DOWN, "L": LEFT, "R": RIGHT}


def big(x, y): return grid(x, y, 16, 16)


def four(dirs=(UP, DOWN, LEFT, RIGHT), lens=(3, 3, 3, 3), names=A, each=2):
    """Each of four actions `each` times from the middle of a 16 x 16 board, its step `lens[i]` cells in `dirs[i]`."""
    mv = []
    for nm, (dx, dy), ln in zip(names, dirs, lens):
        for _ in range(each): mv.append((nm, big(7, 7), big(7 + dx * ln, 7 + dy * ln)))
    return mv


# ---------------------------------------------------------------- 1 every new part off: the sixth check, counter for counter
def test_with_every_new_part_off_the_counters_are_the_sixth_checks_evaluate_on_the_invented_worlds_and_the_small_worlds():
    for mk in (lambda: World(PUSH_LEVEL, pads=False), lambda: World(PAD_LEVEL)):
        for lv in range(1, 6):
            seq = walk(mk(), lv, 60); a = F6.evaluate(seq); b = F7.evaluate(seq)
            assert a and all(b.get(k, 0) == v for k, v in a.items())
            assert all(b.get(k, 0) == v for k, v in F5.evaluate(seq).items() if not k.startswith(("step:", "rel:")))
            assert all(b.get(k, 0) == v for k, v in F3.evaluate(seq, True, True).items())
    for seq in (tall_seq(), tall_seq(12), [("S", row(i), row(i + 1)) for i in range(1, 5)] + [("S", row(5), row(5))] * 8, three(("R", grid(0, 9), grid(3, 9))), four()):
        a = F6.evaluate(seq); b = F7.evaluate(seq); assert all(b.get(k, 0) == v for k, v in a.items())
    for kw in (dict(floor=False), dict(group=False), dict(once=True), dict(once=True, rot=True, rel=True, earlier_rel={("D", "U"): 2})):
        seq = walk(World(PAD_LEVEL), 3, 60); a = F6.evaluate(seq, **kw); b = F7.evaluate(seq, **kw); assert all(b.get(k, 0) == v for k, v in a.items())
    seq = walk(World(PAD_LEVEL), 3, 60)                                                                         # names given but nm off: nothing changes
    assert all(F7.evaluate(seq, names=NAMES, carry=("P", 3)).get(k, 0) == v for k, v in F6.evaluate(seq).items())


def test_the_screen_memory_and_the_counters_around_it_change_no_prediction():
    for seq in (walk(World(PAD_LEVEL), 3, 80), walk(World(PUSH_LEVEL, pads=False), 4, 80), [("S", row(5, n), row(5, n + 1)) for n in range(14)], four()):
        off = F7.evaluate(seq, once=True, rot=True, rel=True, nm=True, names=NAMES); on = F7.evaluate(seq, once=True, rot=True, rel=True, nm=True, names=NAMES, scr=True)
        assert all(on.get(k, 0) == v for k, v in off.items()) and not any(k.startswith(("scr_", "scx_")) for k in off)


def test_the_check_is_a_pure_function_and_changes_nothing_it_is_given():
    seq = walk(World(PAD_LEVEL), 3, 120); snap = json.dumps(seq); names = dict(NAMES); carry = ("P", 3)
    kw = dict(once=True, rot=True, rel=True, nm=True, scr=True, earlier_rel={("D", "U"): 2}, names=names, carry=carry); a = F7.evaluate(seq, **kw); assert json.dumps(seq) == snap and names == NAMES
    assert a == F7.evaluate(list(seq), **kw) and F7.text({}) == "" and F7.rates({})["nm_precision"] is None and F7.rates({})["postfail_precision"] is None and F7.rates({})["edge_share"] is None and F7.rates({})["scx_null_share"] is None
    assert list(F7.VARIANTS) == ["full", "once", "all6", "nm", "all7", "best"] and F7.THIRD_FULL == (603, 729) and F7.CHECK5_ONCE == (663, 797) and F7.CHECK6_ALL == (672, 809)
    assert F7.SOURCES == ("law", "once", "rot", "rel", "nm", "nc") and F7.ACTIONS == A and len(F7.G_LINES) == 12 and len(F7.NORTH_CLASSES) == 6
    L = led(four()); snapshot = (len(L.rows), L.dominant()); F7.from_name(L, "X", NAMES, ("P", 3)); F7.loose_steps(L); F7.arrangement_class({}); assert (len(L.rows), L.dominant()) == snapshot


# ---------------------------------------------------------------- 2 the helpers
def test_the_steps_from_one_move_on_are_read_by_loose_steps_and_by_axis_steps_only_from_two():
    L = led(three(None)[:1]); assert F7.loose_steps(L) == ("P", {"U": ((0, -3), 1)}) and F7.axis_steps(L)[1] == {}                      # one try: enough for a length, not for a step of the sixth check
    L = led(three(None)[:6]); main, st = F7.loose_steps(L); assert main == "P" and st == {"U": ((0, -3), 2), "D": ((0, 3), 2), "L": ((-3, 0), 2)}
    assert F7.loose_steps(Ledger()) == (None, {}) and F7.loose_steps(led([("A", grid(2, 2), grid(3, 3))])) == (None, {}) and F7.loose_steps(led([("Z@1,1", grid(2, 2), grid(2, 3))])) == (None, {})   # a diagonal step and a click are not a step along one axis


def test_a_name_gives_the_direction_and_a_step_seen_gives_the_length():
    L = led(three(None)[:1])                                                                                         # U moved 3 cells once
    t, e = F7.from_name(L, "D", NAMES, None); assert t == "nm" and e == ("P", (0, 3), "U", (0, -3), 1)
    assert F7.from_name(L, "R", NAMES, ("Q", 9))[0] == "nm" and F7.from_name(L, "R", NAMES, ("Q", 9))[1][:2] == ("P", (3, 0))                  # the length seen wins over the carried one
    assert F7.from_name(L, "D", {}, None) is None and F7.from_name(L, "D", None, None) is None and F7.from_name(L, "X", NAMES, None) is None      # no name, no direction
    mixed = led([("U", grid(6, 9), grid(6, 6)), ("L", grid(6, 6), grid(4, 6))]); assert F7.from_name(mixed, "D", NAMES, ("P", 3)) is None            # a step of 3 and a step of 2: no one length
    t, e = F7.from_name(Ledger(), "L", NAMES, ("P", 4)); assert t == "nc" and e == ("P", (-4, 0), None, None, 0)                                       # nothing seen in this level: the earlier level's length
    assert F7.from_name(Ledger(), "L", NAMES, None) is None


def test_the_arrangement_class_is_one_of_six_turned_to_face_north_and_none_unless_all_four_made_a_step_of_one_length():
    st = lambda **kw: F7.axis_steps(led(four(**kw)))[1]
    c = F7.arrangement_class(st()); assert c == (DOWN, LEFT, RIGHT) and c in F7.NORTH_CLASSES
    assert F7.arrangement_class(st(dirs=(DOWN, UP, RIGHT, LEFT))) == c                                                  # the whole arrangement turned half a turn is the same class
    assert F7.arrangement_class(st(dirs=(RIGHT, LEFT, DOWN, UP))) == F7.arrangement_class(st(dirs=(UP, DOWN, RIGHT, LEFT))) == (DOWN, RIGHT, LEFT) and c != (DOWN, RIGHT, LEFT)       # two actions swapped: the other handedness
    assert F7.arrangement_class(st(dirs=(UP, RIGHT, DOWN, LEFT))) == (RIGHT, DOWN, LEFT) and len({F7.arrangement_class(st(dirs=d)) for d in ((UP, DOWN, LEFT, RIGHT), (UP, DOWN, RIGHT, LEFT), (UP, LEFT, DOWN, RIGHT), (UP, LEFT, RIGHT, DOWN), (UP, RIGHT, DOWN, LEFT), (UP, RIGHT, LEFT, DOWN))}) == 6
    assert F7.arrangement_class(st(lens=(3, 3, 3, 4))) is None and F7.arrangement_class(st(each=1)) is None and F7.arrangement_class({}) is None          # one length; each action tried twice
    assert F7.arrangement_class(F7.axis_steps(led(four(names=("A", "B", "C", "D"))))[1]) is None                         # other names: not ACTION1 to ACTION4


# ---------------------------------------------------------------- 3 nm and nc
def test_nm_answers_the_first_try_of_an_action_from_the_name_and_the_length_seen_and_is_wrong_for_a_press():
    mv = three(("R", grid(0, 9), grid(3, 9)))                                                                          # U twice, D twice, L twice, then R for the first time (really (3, 0))
    off = get(F7.evaluate(mv, names=NAMES)); on = get(F7.evaluate(mv, nm=True, names=NAMES))
    assert off["predicted"] == 0 and (on["predicted"], on["nm_pred"], on["nm_hit"], on["cell_hit"], on["nc_pred"]) == (3, 3, 3, 3, 0)               # the first tries of D, L and R
    press = three(("A", grid(0, 9), grid(0, 9))); p = F7.evaluate(press, nm=True, names=dict(NAMES, A=RIGHT)); assert (p["nm_pred"], p["nm_hit"]) == (3, 2)           # A is named a step right and presses
    assert F7.evaluate(mv, nm=True, names={"U": UP}).get("nm_pred", 0) == 0 and F7.evaluate(mv, nm=True).get("nm_pred", 0) == 0                    # D, L and R have no name's direction
    assert F7.evaluate(three(("R@2,3", grid(0, 9), grid(0, 9))), nm=True, names=dict(NAMES, **{"R@2,3": RIGHT})).get("nm_pred", 0) == 3 - 1           # a click is never guessed (D and L still are)
    odd = [("U", grid(6, 9), grid(6, 6)), ("L", grid(6, 6), grid(4, 6)), ("D", grid(4, 6), grid(4, 9))]; o = F7.evaluate(odd, nm=True, names=NAMES); assert (o["nm_pred"], o["nm_hit"]) == (1, 0)     # L is guessed 3 from U (it made 2); D then finds steps of 3 and 2: no one length, nothing
    tried = mv[:6] + [("R", grid(0, 9), grid(0, 9)), ("R", grid(0, 9), grid(3, 9))]; t = F7.evaluate(tried, nm=True, names=NAMES); assert t["nm_pred"] == 3 and t["nm_hit"] == 2 and t["abs:once_nomove"] == 1   # only the first try of R is guessed (it did nothing); its second try is not
    both = get(F7.evaluate(mv, nm=True, once=True, names=NAMES)); assert (both["nm_pred"], both["once_pred"], both["predicted"]) == (3, 3, 6)               # the tiers add


def test_nc_answers_from_the_latest_earlier_levels_length_when_this_level_has_shown_no_step_and_never_from_the_level_itself():
    first = [("U", big(7, 7), big(7, 4))]                                                                              # the very first move of the level
    assert get(F7.evaluate(first, nm=True, names=NAMES)).get("nm_pred") == 0 and get(F7.evaluate(first, nm=True, names=NAMES, carry=("P", 3)))["nc_pred"] == 1 and get(F7.evaluate(first, nm=True, names=NAMES, carry=("P", 3)))["nc_hit"] == 1
    wrong = F7.evaluate(first, nm=True, names=NAMES, carry=("P", 4)); assert (wrong["nc_pred"], wrong["nc_hit"]) == (1, 0)                              # the length changed in this level
    other = F7.evaluate(first, nm=True, names=NAMES, carry=("Q", 3)); assert (other["nc_pred"], other["nc_hit"]) == (1, 0)                              # the mover's symbol is not on the board
    seen = [("U", big(7, 7), big(7, 4)), ("D", big(7, 4), big(7, 7))]; s = F7.evaluate(seen, nm=True, names=NAMES, carry=("P", 5)); assert (s["nc_pred"], s["nm_pred"], s["nm_hit"]) == (1, 1, 1)   # D's length is U's, not the carried 5


def test_the_order_is_the_carried_relation_then_the_name_then_the_rotation_set_and_the_tiers_add_up():
    mv = three(("R", grid(0, 9), grid(3, 9)))
    b = get(F7.evaluate(mv, rel=True, nm=True, rot=True, names=NAMES, earlier_rel={("R", "U"): 0}))                                                      # the relation says R = U (wrong), the name and the rotation set say (3, 0)
    assert (b["rel_pred"], b["rel_hit"], b["nm_pred"], b["rot_pred"]) == (1, 0, 2, 0)                                                                    # D and L by name; R by the relation
    n = get(F7.evaluate(mv, nm=True, rot=True, names=NAMES)); assert (n["nm_pred"], n["rot_pred"]) == (3, 0)                                            # the name is read before the rotation set
    r = get(F7.evaluate(mv, nm=True, rot=True, names={"U": UP})); assert (r["nm_pred"], r["rot_pred"], r["rot_hit"]) == (0, 1, 1)                          # no name for R: the rotation set answers
    for kw in (dict(), dict(once=True), dict(once=True, rot=True, rel=True, nm=True, names=NAMES), dict(nm=True, names=NAMES, carry=("P", 3))):
        c = F7.evaluate(mv, **kw); assert sum(c.get(s + "_pred", 0) for s in F7.SOURCES) == c.get("predicted", 0) and sum(c.get(s + "_hit", 0) for s in F7.SOURCES) == c.get("cell_hit", 0)
    r = F7.rates(F7.combine([F7.evaluate(mv, nm=True, names=NAMES), F7.evaluate(three(("A", grid(0, 9), grid(0, 9))), nm=True, names=dict(NAMES, A=RIGHT))])); assert (r["name_pred"], r["name_hit"], r["name_precision"]) == (6, 5, 5 / 6)


# ---------------------------------------------------------------- 4 the name prior from the other games, the carried length
def test_the_name_prior_is_read_from_the_other_games_only_from_four_levels_and_ninety_percent():
    from collections import Counter
    dirs = {"a": {"X": Counter({UP: 5})}, "b": {"X": Counter({UP: 5})}, "c": {"X": Counter({DOWN: 1})}}
    assert F7.name_prior(dirs, "c") == {"X": UP} and F7.name_prior(dirs, "a") == {}                    # for c: 10 levels of UP; for a: 5 of 6 levels agree, 83 percent
    assert F7.name_prior({"a": {"X": Counter({UP: 5})}}, "a") == {}                                                     # a game never reads its own levels
    assert F7.name_prior({"a": {"X": Counter({UP: 3})}, "b": {}}, "b") == {}                                            # 3 levels: not enough
    assert F7.name_prior({"a": {"X": Counter({UP: 4})}, "b": {}}, "b") == {"X": UP}
    assert F7.name_prior({"a": {"X": Counter({UP: 9, DOWN: 1})}, "b": {}}, "b") == {"X": UP} and F7.name_prior({"a": {"X": Counter({UP: 8, DOWN: 2})}, "b": {}}, "b") == {}      # exactly 90 percent; 80 percent


def _games(levels_a=5, levels_b=5):
    a = {("a", i): led(four(lens=(3, 3, 3, 3))) for i in range(1, levels_a + 1)}
    b = {("b", i): led([("ACTION1", big(7, 7), big(7, 4)), ("ACTION1", big(7, 4), big(7, 1)), ("ACTION2", big(7, 1), big(7, 4))]) for i in range(1, levels_b + 1)}
    return Sess({**a, **b})


def test_a_game_takes_the_names_from_the_other_games_and_the_length_from_its_own_earlier_levels():
    w = F7.from_session(_games()); assert set(w) == set(F7.VARIANTS) and set(w["nm"]) == {"a", "b"}
    nb = w["nm"]["b"]; assert nb["nm_pred"] == nb["nm_hit"] == 5 and nb.get("nc_pred", 0) == nb.get("nc_hit", 0) == 4              # b: ACTION2's first try (nm) in all 5 levels; ACTION1's first move from level 2 on (nc, the length of level 1)
    na = w["nm"]["a"]; assert na.get("nm_pred", 0) == na.get("nm_hit", 0) == 0 and na["nc_pred"] == na["nc_hit"] == 4                  # a: the other game only knows ACTION1; its first move is carried from level 2 on
    assert w["full"]["a"].get("nm_pred", 0) == 0 and w["full"]["b"].get("nc_pred", 0) == 0                                              # with nm off nothing is guessed
    alone = F7.from_session(Sess({("a", i): led(four()) for i in range(1, 6)})); assert alone["nm"]["a"].get("nm_pred", 0) == 0 and alone["nm"]["a"].get("nc_pred", 0) == 0     # a single game has no other game to read the names from
    few = F7.from_session(_games(levels_a=3)); assert few["nm"]["b"].get("nm_pred", 0) == 0                                              # 3 levels in the other game: no prior
    pc = F7.prior_counts(F7.combine(w["full"].values())); assert pc["ACTION1"] == (10, 10) and pc["ACTION2"] == (5, 0) and pc["ACTION3"] == (5, 0) and pc["ACTION4"] == (5, 0)      # ACTION1 made a step in 10 levels (5 of a, 5 of b) and both games name it; ACTION2 to 4 made one only in a, whose prior from b has only ACTION1


def test_the_names_the_classes_and_the_carried_length_are_counted_for_the_text():
    w = F7.from_session(_games()); full = F7.combine(w["full"].values())
    assert F7.class_counts(full) == {"0,1|-1,0|1,0": 5} and full["cls:0,1|-1,0|1,0"] == 5                                                   # the five full levels of game a, all of one class
    t = F7.text(w)
    for n in ("== 1 first tries answered from the NAME", "== 2 the names and the arrangement", "== 3 the screen memory", "== 4 the retire question", "== 5 the edge", "all7:", "best:", "nm:", "classes seen 1 of 6"): assert n in t, n
    assert F7.text({"full": {}}) == ""


# ---------------------------------------------------------------- 5 the screen memory, the retire question, the edge
def _bar(x, n, w=12): return tuple("".join("P" if (c, r) == (x, 1) else ("#" if r == 0 and c < n else ".") for c in range(w)) for r in range(3))


def test_the_screen_memory_with_a_strip_that_is_used_up_finds_its_repeats_only_with_the_symbol_wide_mask():
    shr = [("S", _bar(5, 11 - n), _bar(5, 10 - n)) for n in range(11)]
    c = F7.evaluate(shr, scr=True)
    assert c["tick_levels"] == 1 and c["scr_n"] == c["scx_n"] == 5 and c["scr_same"] == 0 and c["scx_same"] == 5 and c["scx_null"] == 5 and c["scx_null_null"] == 5      # the strict mask leaks the cell about to go; the wide one does not
    assert c["scx_fill"] == 5 and c["scx_fill_right"] == 5 and c["scr_fill"] == 5 and c["scr_fill_right"] == 0 and c.get("scx_over", 0) == 0                                               # nothing else predicted these moves
    assert F7.evaluate(shr).get("scx_n", 0) == 0                                                                                                                                      # off: nothing counted
    grow = [("S", row(5, n), row(5, n + 1)) for n in range(14)]; g = F7.evaluate(grow, scr=True); assert g["scx_n"] == 7 and g["scx_same"] == g["scx_null_null"] == 2           # a strip that grows leaks its new cell under both masks
    still = [("S", row(5), row(5))] * 6; s = F7.evaluate(still, scr=True); assert s["scx_n"] == s["scr_n"] == 5 and s["scx_same"] == 5 and s["scx_null_null"] == 5 and s.get("tick_levels", 0) == 0   # no strip: the whole screen repeats
    moving = [("S", row(i, i), row(i + 1, i + 1)) for i in range(1, 6)]; assert F7.evaluate(moving, scr=True).get("scx_n", 0) == 0
    over = F7.evaluate(tall_seq(6), scr=True); assert over["scx_over"] > 0 and over["scx_over_main_right"] > 0                                                                       # where a tier predicted too, both are counted side by side
    r = F7.rates(F7.combine([c])); assert r["scx_null_share"] == 1.0 and r["scx_fill_share"] == 1.0 and r["scr_fill_share"] == 0.0


def _tall(ys, h=40): return [("U", grid(6, a, 14, h), grid(6, b, 14, h)) for a, b in zip(ys, ys[1:])]


def test_the_retire_counters_separate_the_law_tier_before_and_after_a_wrong_step():
    seq = _tall([36, 33, 30, 27, 25, 22, 19])                                                                           # (0,-3) three times: the law from the third move on; then (0,-2): wrong by a different step; then (0,-3), (0,-3) again right
    c = F7.evaluate(seq); assert (c["law_pred"], c["prefail_pred"], c["prefail_hit"], c["fail_events"], c["postfail_pred"], c["postfail_hit"]) == (4, 2, 1, 1, 2, 2)
    again = F7.evaluate(_tall([36, 33, 30, 27, 25, 22, 20])); assert (again["prefail_pred"], again["prefail_hit"], again["fail_events"], again["postfail_pred"], again["postfail_hit"]) == (2, 1, 2, 2, 1)     # a second wrong step after the first
    steady = F7.evaluate(_tall([36, 33, 30, 27, 24, 21])); assert steady["prefail_pred"] == steady["prefail_hit"] == 3 and steady.get("postfail_pred", 0) == 0 and steady.get("fail_events", 0) == 0
    r = F7.rates(c); assert r["prefail_precision"] == 0.5 and r["postfail_precision"] == 1.0
    p = F7.evaluate(_tall([36, 33, 30, 27, 25, 22]), once=True, nm=True, names=NAMES); assert p.get("postfail_pred", 0) + p.get("prefail_pred", 0) == p["law_pred"]          # only the law tier


def test_the_edge_counters_say_whether_a_piece_predicted_to_stay_at_the_edge_stays_or_moves():
    stay = [("U", grid(6, 12), grid(6, 9)), ("U", grid(6, 9), grid(6, 6)), ("U", grid(6, 6), grid(6, 3)), ("U", grid(6, 3), grid(6, 0)), ("U", grid(6, 0), grid(6, 0))]
    c = F7.evaluate(stay); assert (c["edge_said"], c["edge_right"], c.get("edge_moved", 0), c["wall_said"]) == (1, 1, 0, 1)
    wrap = stay[:4] + [("U", grid(6, 0), grid(6, 11))]; w = F7.evaluate(wrap); assert (w["edge_said"], w.get("edge_right", 0), w["edge_moved"]) == (1, 0, 1)
    off = F7.evaluate(stay, edge=False); assert off.get("edge_said", 0) == 0                                              # without the edge nothing is said for it
    assert F7.rates(c)["edge_share"] == 1.0 and F7.rates(w)["edge_share"] == 0.0


# ---------------------------------------------------------------- 6 the reading lines
GAMES = ("g50t-1", "ka59-1", "ls20-1", "sp80-1")
SPLIT = lambda total: [total // 4 + (1 if i < total % 4 else 0) for i in range(4)]


def _c(predicted=182, hit=150, **kw):
    d = dict(moves=234, predicted=predicted, cell_hit=hit, mrep=8, cell_said=0, cell_right=0, known=predicted, did_not_move=40, rep_pos_null_null=10)
    d.update(kw); return d


def _w(hits, preds, **extra):
    out = {g: _c(predicted=p, hit=h) for g, p, h in zip(GAMES, SPLIT(preds), SPLIT(hits))}
    for k, tot in extra.items():
        for g, v in zip(GAMES, SPLIT(tot)): out[g][k] = v
    return out


def _full(post=(20, 14), edge=(10, 9), pri=(10, 10), cls=None, gone=()):
    f = _w(603, 729, postfail_pred=post[0], postfail_hit=post[1], edge_said=edge[0], edge_right=edge[1]); g = f["g50t-1"]
    for i in range(1, 5): g[f"pri:ACTION{i}:n"] = pri[0]; g[f"pri:ACTION{i}:ok"] = pri[1]
    for k in gone: del g[k]
    g.update(cls if cls is not None else {"cls:0,1|-1,0|1,0": 12}); return f


def _v(full=None, once=None, a6=None, nm=None, a7=None, best=None, ok=True):
    v = {"full": full or _full(), "once": once or _w(663, 797), "all6": a6 or _w(672, 809), "nm": nm or _w(603, 729, nm_pred=36, nm_hit=28, nc_pred=0, nc_hit=0),
         "all7": a7 or _w(603 + 90, 729 + 112, scx_n=160, scx_null=32, scx_null_null=30, scx_fill=12, scx_fill_right=10), "best": best or _w(712, 850)}
    return {i: x for i, _, _, x in F7.verdict(v, ok)}


def test_the_twelve_lines_flip_at_their_boundaries_and_the_headline_needs_lines_4_and_5():
    assert len(F7.G_LINES) == 12 and [t for t, _ in F7.G_LINES] == ["guard", "instrument"] + ["claim"] * 10
    assert {i: x for i, _, _, x in F7.verdict(None, None)} == {i: None for i in range(1, 13)} and _v(ok=False)[1] is False and all(_v().values())
    assert _v(full=_w(603, 728))[2] is False and _v(full=_w(602, 729))[2] is False and _v(once=_w(663, 798))[2] is False and _v(a6=_w(671, 809))[2] is False and _v(a6=_w(672, 810))[2] is False
    assert _v(full=_full(pri=(10, 9)))[3] is True and _v(full=_full(pri=(10, 8)))[3] is False and _v(full=_full(pri=(4, 4)))[3] is True and _v(full=_full(pri=(3, 3)))[3] is False        # 90 percent exactly; at least 4 levels
    assert _v(full=_full(gone=("pri:ACTION4:n", "pri:ACTION4:ok")))[3] is False                                                                                                       # an action never seen making a step is not yes
    n = lambda pr, h: _w(603, 729, nm_pred=pr, nm_hit=h)
    assert _v(nm=n(35, 27))[4] is True and _v(nm=n(35, 26))[4] is False and _v(nm=n(34, 34))[4] is False and _v(nm=n(0, 0))[4] is False                          # 77 percent of 35; 74 percent; 34 is below 35
    assert _v(nm=_w(603, 729, nm_pred=20, nm_hit=16, nc_pred=15, nc_hit=11))[4] is True                                                                    # both lengths count
    a = lambda add, right, **kw: _w(603 + right, 729 + add, scx_n=160, scx_null=32, scx_null_null=30, scx_fill=12, scx_fill_right=10, **kw)
    assert _v(a7=a(110, 88))[5] is True and _v(a7=a(110, 87))[5] is False and _v(a7=a(109, 109))[5] is False and _v(a7=a(0, 0))[5] is False                    # 110 more, 88 right is 80 percent
    cl = lambda **kw: _full(cls=kw)
    assert _v(full=cl(**{"cls:0,1|-1,0|1,0": 10}))[6] is True and _v(full=cl(**{"cls:0,1|-1,0|1,0": 9}))[6] is False and _v(full=cl(**{"cls:0,1|-1,0|1,0": 7, "cls:0,1|1,0|-1,0": 6}))[6] is False and _v(full=cl())[6] is False
    s = lambda **kw: _w(603 + 90, 729 + 112, **{**dict(scx_n=160, scx_null=32, scx_null_null=30, scx_fill=12, scx_fill_right=10), **kw})
    assert _v(a7=s(scx_n=150))[7] is True and _v(a7=s(scx_n=149))[7] is False and _v(a7=s(scx_n=0))[7] is False
    assert _v(a7=s(scx_null=30, scx_null_null=27))[8] is True and _v(a7=s(scx_null=30, scx_null_null=26))[8] is False and _v(a7=s(scx_null=29, scx_null_null=29))[8] is False
    assert _v(a7=s(scx_fill=10, scx_fill_right=8))[9] is True and _v(a7=s(scx_fill=10, scx_fill_right=7))[9] is False and _v(a7=s(scx_fill=9, scx_fill_right=9))[9] is False
    assert _v(full=_full(post=(20, 14)))[10] is True and _v(full=_full(post=(20, 15)))[10] is False and _v(full=_full(post=(14, 2)))[10] is False and _v(full=_full(post=(15, 10)))[10] is True       # 70 percent exactly; 75 percent; 14 is below 15
    assert _v(full=_full(edge=(10, 9)))[11] is True and _v(full=_full(edge=(10, 8)))[11] is False and _v(full=_full(edge=(5, 5)))[11] is True and _v(full=_full(edge=(4, 4)))[11] is False
    assert _v(best=_w(712, 850))[12] is True and _v(best=_w(711, 850))[12] is False and _v(best=_w(712, 870))[12] is False and _v(best=_w(0, 0))[12] is False                  # 40 more than 672, and 82 percent of the predicted
    allyes = [(i, t, x, True) for i, t, x, _ in F7.verdict(None, None)]
    assert F7.headline(allyes) == F7.HEADLINE and F7.claims_allowed(allyes)
    assert F7.headline([(i, t, x, (False if i == 5 else v)) for i, t, x, v in allyes]) == "MIXED: line 4 yes, line 5 NO"
    assert F7.headline([(i, t, x, (False if i == 4 else v)) for i, t, x, v in allyes]) == "MIXED: line 4 NO, line 5 yes"
    assert F7.headline([(i, t, x, (False if i == 2 else v)) for i, t, x, v in allyes]).startswith("NOT READ") and F7.headline(F7.verdict(None, None)).startswith("NOT READ")
    assert F7.headline([(i, t, x, (False if i == 1 else v)) for i, t, x, v in allyes]).startswith("NOT READ")


# ---------------------------------------------------------------- 7 page, notebook, runner, replay of arm C, front pages
def _head():
    text = open(PAGE, encoding="utf-8").read(); assert text.count(RESULT_MARK) <= 1
    return text[:text.index(RESULT_MARK)] if RESULT_MARK in text else text


def test_the_page_above_its_result_is_what_was_written_before_the_run():
    assert hashlib.sha256(_head().encode()).hexdigest() == PAGE_SHA256


def test_the_page_and_the_module_agree_and_it_says_which_text_is_whose():
    head = _head(); pred = head.split("## Predictions")[1].split("## Reading rules")[0]; rules = head.split("## Reading rules")[1].split("Headline:")[0]
    ps = re.findall(r"^(\d+)\. \[(\w+)\] (.*) \((\d+) %\)$", pred, re.M); rs = re.findall(r"^(\d+)\. (.*)$", rules, re.M)
    assert len(ps) == len(rs) == len(F7.G_LINES) == 12
    for (n, tag, text, pr), (m, rt), (gt, gtext) in zip(ps, rs, F7.G_LINES): assert n == m and text == rt == gtext and tag == gt and 0 < int(pr) < 100
    assert [int(p[3]) for p in ps] == [95, 85, 85, 50, 40, 65, 55, 70, 40, 35, 55, 40]
    for needle in (F7.HEADLINE, "Sir to confirm", "also run by Sir", "Delete Test", "603 right of 729", "672 right", "19 of 19", "20 of 20", "15 of 15", "18 of 18", "268 repeats", "89 of 93", "127 moves", "will not show", "paid test", "$3.65", "scx", "scr", "Result", "not written yet",
                   "Concept-map item 142", "Six ways", "invented worlds", "other games", "leak", "retire", "wrap", "arrangement class", "6 possible arrangements"): assert needle in head, needle
    assert "proven" not in head.lower() and "ledger and the predictor of the sixth check are not changed" in head and "Nothing on this page has been run on the real games" in head
    assert "the most common first guess" not in head


def test_the_notebook_and_the_runner_read_the_saved_files_call_no_model_hold_no_key_and_run_six_ways():
    nb = json.load(open(NB)); src = "\n".join("".join(c["source"]) for c in nb["cells"]); code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert len(nb["cells"]) == 12 and all(c["outputs"] == [] for c in nb["cells"] if c["cell_type"] == "code")
    run = open(RUN, encoding="utf-8").read()
    for text in (src, run):
        assert not re.search(r"AIza[0-9A-Za-z_-]{20,}", text) and "GOOGLE_API_KEY" not in text and "userdata" not in text and "drive.mount" not in text and "open_store" not in text
        assert "foresee6" not in text and "foresee5" not in text and "generate" not in text and ".run(" not in text.replace("LR.run_file(", "")
    for needle in ("foresee7 as F", "F.from_session", "LR.run_file", "F.verdict", "F.headline", "F.VARIANTS", "axi_gemini_arc_c_default.json", "axi_gemini_arc_b_default.json", "axi_gemini_arc_default.json", "foresee_check_7_all.json", "foresee7.py"): assert needle in code, needle
    for needle in ("foresee7 as F", "F.from_session", "LR.run_file", "F.verdict", "F.headline", "foresee_check_7_all.json", "foresee_check_7_printout.txt", "axi_gemini_arc_c_default.json"): assert needle in run, needle
    assert "FORESEE_CHECK_7.md" in src and "12 reading lines" in src and "verdict(per_game if allok else None, allok if ok else None)" in code and "verdict(per_game if allok else None, allok if ok else None)" in run
    for c in nb["cells"]:
        if c["cell_type"] == "code": compile("\n".join(l for l in "".join(c["source"]).split("\n") if not l.lstrip().startswith("!")), "cell", "exec")
    compile(run, "run_foresee_check_7.py", "exec")


def test_the_seventh_check_runs_over_a_replay_of_arm_c_on_the_stand_in_game_six_ways():
    res = C.run(T.FakeLLM(), T.make_env, os.path.join(os.environ.get("TMPDIR", "/tmp"), "foresee7_c_test.jsonl"), games=("alpha-1", "beta-2"), moves=30, action_of=lambda n: n)
    s = LR.feed(res["rows"], T.make_env, "C", games=("alpha-1", "beta-2"), moves=30, action_of=lambda n: n); w = F7.from_session(s)
    assert set(w) == set(F7.VARIANTS) and all(set(pg) == {"alpha-1", "beta-2"} for pg in w.values())
    assert all(sum(c["moves"] for c in pg.values()) == sum(s.moves.values()) for pg in w.values()) and "ALL GAMES" not in F7.text(w) and "== 2 the names and the arrangement" in F7.text(w) and "== 3 the screen memory" in F7.text(w)
    w6 = F6.from_session(s)
    for g in w["full"]:
        assert all(w["full"][g].get(k, 0) == v for k, v in w6["full"][g].items()) and all(w["once"][g].get(k, 0) == v for k, v in w6["once"][g].items()) and all(w["all6"][g].get(k, 0) == v for k, v in w6["all"][g].items())   # the sixth check's full, once and all
        assert w["all7"][g]["predicted"] >= w["all6"][g]["predicted"] >= w["full"][g]["predicted"]                              # more tiers never predict fewer
    checks = F7.verdict(w, True); assert len(checks) == 12 and checks[0][3] is True and F7.headline(checks)


def test_the_front_pages_say_what_was_built_and_that_it_was_not_run_on_the_real_games():
    cm = open(os.path.join(DOCS, "RAJNISH_CONCEPT_MAP.md"), encoding="utf-8").read(); vf = open(os.path.join(DOCS, "VERIFIED.md"), encoding="utf-8").read(); hl = open(os.path.join(DOCS, "HISTORY_LEDGER.md"), encoding="utf-8").read()
    assert NEXT in cm and cm.count("Next free:") == 1 and "\n141. **" in cm and "\n142. **" in cm and "\n145. **" not in cm
    i = cm.index("\n142. **") + 1; item = cm[i:cm.index("\n", i)]
    for n in ("FORESEE_CHECK_7.md", "foresee7.py", "test 167", "also run by Sir", "Sir to confirm", "Not run on the real games", "12 reading lines", "other three of four games", PAGE_SHA256): assert n in item, n
    assert any("FORESEE_CHECK_7.md" in l and "foresee7.py" in l and PAGE_SHA256 in l and "RUN by the owner" in l for l in vf.split("\n"))
    assert "| 188 (Q) |" in hl and "Sir" in hl and "not run on the real games" in hl and "the retire question" in hl


if __name__ == "__main__":
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]; bad = 0
    for n, f in fns:
        try: f(); print("ok  ", n)
        except Exception: bad += 1; print("FAIL", n); traceback.print_exc()
    print(f"{len(fns) - bad}/{len(fns)} passed"); sys.exit(1 if bad else 0)
