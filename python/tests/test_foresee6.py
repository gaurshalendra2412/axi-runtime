"""Test 166 (item 141): the sixth forward check (docs/FORESEE_CHECK_6.md, axi/experiments/foresee6.py, colab/axi_foresee_check_6.ipynb, colab/run_foresee_check_6.py): a first try from the shape of the actions, the three tiers, the names, the delete test again, the screen without its clock.

The fifth check could only see steps of one cell, and the real games move 5, 6, 3 and 4 cells; its `elim` predicted nothing and its line on the names stayed empty. This one reads steps of any length. This test checks, by hand and on small worlds built for it:
  1. with every new part off the counters are the fifth check's `full` (and so the third check's), counter for counter, on the invented worlds and on the small worlds;
  2. the helpers: a quarter-turn, the steps of any length each action made, the relations between actions of the same step length, the rotation set (exactly three known, three different, one length);
  3. `rot` predicts the one missing turn of a step of three cells for an action never tried and is wrong for a press; `rel` predicts from a relation carried from earlier levels, nothing when two known actions disagree, and is read before `rot`;
  4. the carried relation comes from earlier levels only (a level never predicts from itself or from a later one);
  5. the tier of every prediction (law, once, rot, rel), the direction, length and quarter-turn counters, the whole screen with the strip that ticks left out;
  6. the floor colours and the group deleted together, on the invented worlds;
  7. the 9 reading lines at their boundaries, the page (its hash, its lines, what it says it will not show), the notebook and the runner (no key, no model, nine ways), the replay of arm C on the stand-in game, the front pages.
What it does NOT show: how the check reads on the real games. That is the Result of the run (Codespaces or Colab)."""
import hashlib, json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.ledger import Ledger
from axi.experiments import foresee3 as F3
from axi.experiments import foresee5 as F5
from axi.experiments import foresee6 as F6
from axi.experiments import ledger_replay as LR
from axi.experiments import gemini_arc_c as C
from axi.experiments.mini_worlds import World, walk, PUSH_LEVEL, PAD_LEVEL
from tests import test_gemini_arc as T

HERE = os.path.dirname(os.path.abspath(__file__)); DOCS = os.path.join(HERE, "..", "..", "docs")
PAGE = os.path.join(DOCS, "FORESEE_CHECK_6.md"); NB = os.path.join(HERE, "..", "colab", "axi_foresee_check_6.ipynb"); RUN = os.path.join(HERE, "..", "colab", "run_foresee_check_6.py")
PAGE_SHA256 = "c87fc2296b2afae14ee7a1c0a813c3d433940124c8ada9b26e5b5882682544fc"; RESULT_MARK = "\n## Result"
NEXT = "Next free: row 192, item 145, test 170, question 122"
KEYS = ("moves", "predicted", "cell_hit", "once_pred", "once_hit", "rot_pred", "rot_hit", "rel_pred", "rel_hit", "law_pred", "law_hit", "abstained", "tick_levels", "mrep", "mrep_null", "mrep_null_null", "rep_screen")
get = lambda c: {k: c.get(k, 0) for k in KEYS}


class Sess:
    """A stand-in for a replayed session: only `.ledgers` is read by `from_session`."""
    def __init__(self, ledgers): self.ledgers = ledgers


def grid(x, y, w=14, h=14):
    return tuple("".join("P" if (c, r) == (x, y) else "." for c in range(w)) for r in range(h))


def row(x, n=0, w=12):
    """One row world: the piece 'P' at column x of row 1, and a counter bar of n '#' along row 0 (the screen is new every move when n changes)."""
    return tuple("".join("P" if (c, r) == (x, 1) else ("#" if r == 0 and c < n else ".") for c in range(w)) for r in range(3))


def tall(x, w=9, wall=6, door=True):
    return tuple("".join("a" if (c == x and r in (1, 2)) else ("#" if (c == wall and r in (1, 2) and door) else ".") for c in range(w)) for r in range(4))


def tall_seq(stops=5): return [("R", tall(i), tall(i + 1)) for i in range(1, 5)] + [("R", tall(5), tall(5))] * stops


# three-cell steps on a 14 x 14 board: U (0,-3), D (0,3), L (-3,0), R (3,0); each of U, D and L twice, then the new action.
def three(new):
    return [("U", grid(6, 9), grid(6, 6)), ("U", grid(6, 6), grid(6, 3)), ("D", grid(6, 3), grid(6, 6)), ("D", grid(6, 6), grid(6, 9)), ("L", grid(6, 9), grid(3, 9)), ("L", grid(3, 9), grid(0, 9))] + [new]


def led(moves):
    L = Ledger(); [L.record(*m) for m in moves]; return L


# ---------------------------------------------------------------- 1 every new part off: the fifth check, counter for counter
def test_with_every_new_part_off_the_counters_are_the_fifth_checks_full_on_the_invented_worlds_and_the_small_worlds():
    skip = ("step:", "rel:")                                                      # the fifth check's own names-of-actions counters; this check has dir:, len: and relk: instead
    for mk in (lambda: World(PUSH_LEVEL, pads=False), lambda: World(PAD_LEVEL)):
        for lv in range(1, 6):
            seq = walk(mk(), lv, 60); a = F5.evaluate(seq); b = F6.evaluate(seq)
            assert a and all(b.get(k, 0) == v for k, v in a.items() if not k.startswith(skip))
            assert all(b.get(k, 0) == v for k, v in F3.evaluate(seq, True, True).items())
    for seq in (tall_seq(), tall_seq(12), [("S", row(i), row(i + 1)) for i in range(1, 5)] + [("S", row(5), row(5))] * 8, three(("R", grid(0, 9), grid(3, 9)))):
        a = F5.evaluate(seq); b = F6.evaluate(seq); assert all(b.get(k, 0) == v for k, v in a.items() if not k.startswith(skip))
    for kw in (dict(floor=False), dict(group=False), dict(once=True)):                  # the old switches still mean what they meant
        seq = walk(World(PAD_LEVEL), 3, 60); a = F5.evaluate(seq, **kw); b = F6.evaluate(seq, **kw); assert all(b.get(k, 0) == v for k, v in a.items() if not k.startswith(skip))


def test_the_check_is_a_pure_function_and_changes_nothing_it_is_given():
    seq = walk(World(PAD_LEVEL), 3, 120); snap = json.dumps(seq); kw = dict(once=True, rot=True, rel=True, earlier_rel={("D", "U"): 2}); a = F6.evaluate(seq, **kw); assert json.dumps(seq) == snap
    assert a == F6.evaluate(list(seq), **kw) and F6.text({}) == "" and F6.rates({})["rot_precision"] is None and F6.rates({})["rel_precision"] is None and F6.rates({})["rep_share"] is None
    assert list(F6.VARIANTS) == ["full", "-floor", "-group", "-floor-group", "once", "rot", "rel", "rot+rel", "all"] and F6.THIRD_FULL == (603, 729) and F6.CHECK5_ONCE == (663, 797) and F6.CHECK5_NOFLOOR == (618, 729)
    assert F6.ACTIONS == ("ACTION1", "ACTION2", "ACTION3", "ACTION4") and F6.SOURCES == ("law", "once", "rot", "rel") and len(F6.PAIRS) == 6 and F6.PAIRS[0] == "ACTION1>ACTION2" and F6.PAIRS[-1] == "ACTION3>ACTION4"
    rel = {("D", "U"): 2}; before = dict(rel); L = led(seq); snapshot = (len(L.rows), L.dominant()); F6.evaluate(seq, earlier_rel=rel, rel=True); assert rel == before and (len(L.rows), L.dominant()) == snapshot


# ---------------------------------------------------------------- 2 the helpers
def test_a_quarter_turn_is_the_same_way_every_time_and_two_steps_are_related_only_when_they_are_turns_of_each_other():
    assert F6.turn((0, -3), 0) == (0, -3) and F6.turn((0, -3), 1) == (3, 0) and F6.turn((0, -3), 2) == (0, 3) and F6.turn((0, -3), 3) == (-3, 0) and F6.turn((0, -3), 5) == (3, 0)
    for v in ((0, -3), (4, 0), (0, 5)):
        for k in range(4): assert F6.turns_between(F6.turn(v, k), v) == k
    assert F6.turns_between((0, 3), (0, -3)) == 2 and F6.turns_between((0, 3), (0, -2)) is None and F6.turns_between((1, 1), (0, 1)) is None


def test_the_steps_of_a_level_are_the_axis_steps_of_any_length_each_action_made_on_the_main_mover():
    L = led(three(("R", grid(0, 9), grid(3, 9)))); main, st = F6.axis_steps(L)
    assert main == "P" and st == {"U": ((0, -3), 2), "D": ((0, 3), 2), "L": ((-3, 0), 2)}                          # R was tried once: not enough
    L = led(three(("R", grid(0, 9), grid(3, 9))) + [("R", grid(3, 9), grid(6, 9)), ("X", grid(6, 9), grid(6, 9)), ("X", grid(6, 9), grid(6, 9)), ("Z@1,1", grid(6, 9), grid(6, 9))])
    main, st = F6.axis_steps(L); assert set(st) == {"U", "D", "L", "R"} and st["R"] == ((3, 0), 2)                    # X moved nothing and the click has no step
    diag = led([("A", grid(2, 2), grid(3, 3)), ("A", grid(3, 3), grid(4, 4))]); assert F6.axis_steps(diag) == (None, {})                   # a step along both axes is not a step on one axis
    assert F6.axis_steps(Ledger()) == (None, {})


def test_the_relations_are_read_between_actions_whose_steps_have_the_same_length_and_are_quarter_turns_of_each_other():
    L = led(three(("R", grid(0, 9), grid(3, 9)))); r = F6.level_relations(L)
    assert r[("D", "U")] == 2 and r[("U", "D")] == 2 and r[("L", "U")] == 3 and r[("U", "L")] == 1 and r[("L", "D")] == 1 and r[("D", "L")] == 3 and len(r) == 6
    mix = led([("U", grid(6, 9), grid(6, 6)), ("U", grid(6, 6), grid(6, 3)), ("L", grid(6, 3), grid(4, 3)), ("L", grid(4, 3), grid(2, 3))]); assert F6.level_relations(mix) == {}      # steps of 3 and 2 cells: no relation
    assert F6.level_relations(Ledger()) == {}


def test_the_rotation_set_needs_exactly_three_known_actions_with_three_different_steps_of_one_length():
    rs = F6.rotation_set(led(three(("R", grid(0, 9), grid(3, 9)))[:6]))
    assert rs is not None and rs[:2] == ("P", (3, 0)) and rs[3] in {(0, -3), (0, 3), (-3, 0)} and rs[4] == 2
    assert F6.rotation_set(led(three(None)[:4])) is None                                                          # only U and D known
    four = led(three(("R", grid(0, 9), grid(3, 9))) + [("R", grid(3, 9), grid(6, 9))]); assert F6.rotation_set(four) is None            # four known: nothing is missing
    twin = led([("U", grid(6, 9), grid(6, 6)), ("U", grid(6, 6), grid(6, 3)), ("V", grid(6, 3), grid(6, 0)), ("V", grid(6, 9), grid(6, 6)), ("L", grid(6, 9), grid(3, 9)), ("L", grid(3, 9), grid(0, 9))]); assert F6.rotation_set(twin) is None   # two actions made the same step
    mixed = led([("U", grid(6, 9), grid(6, 6)), ("U", grid(6, 6), grid(6, 3)), ("D", grid(6, 3), grid(6, 6)), ("D", grid(6, 6), grid(6, 9)), ("L", grid(6, 9), grid(4, 9)), ("L", grid(4, 9), grid(2, 9))]); assert F6.rotation_set(mixed) is None   # not one length
    assert F6.rotation_set(Ledger()) is None


def test_a_carried_relation_gives_the_step_turned_and_gives_nothing_when_two_known_actions_disagree():
    L = led(three(None)[:4])                                                                                      # U and D known
    assert F6.from_relation(L, "X", {("X", "U"): 2})[:2] == ("P", (0, 3)) and F6.from_relation(L, "X", {("X", "U"): 0})[:2] == ("P", (0, -3))
    assert F6.from_relation(L, "X", {("X", "U"): 1, ("X", "D"): 1}) is None                                       # U turned once is (3, 0), D turned once is (-3, 0): they disagree
    assert F6.from_relation(L, "X", {("X", "U"): 2, ("X", "D"): 0})[:2] == ("P", (0, 3))                         # they agree: (0,-3) turned twice and (0,3) not turned
    assert F6.from_relation(L, "X", {("Y", "U"): 2}) is None and F6.from_relation(L, "X", {}) is None and F6.from_relation(Ledger(), "X", {("X", "U"): 2}) is None


# ---------------------------------------------------------------- 3 rot and rel
def test_rot_predicts_the_one_missing_turn_for_an_action_never_tried_and_is_wrong_for_a_press():
    mv = three(("R", grid(0, 9), grid(3, 9)))                                                                      # the new action R really steps right by 3
    off = get(F6.evaluate(mv)); on = get(F6.evaluate(mv, rot=True))
    assert off["predicted"] == 0 and (on["predicted"], on["rot_pred"], on["rot_hit"], on["cell_hit"]) == (1, 1, 1, 1) and F6.evaluate(mv, rot=True)["abs:new"] == 3
    press = three(("A", grid(0, 9), grid(0, 9)))                                                                   # the new action A is a press
    p = get(F6.evaluate(press, rot=True)); assert (p["predicted"], p["rot_pred"], p["rot_hit"], p["cell_hit"]) == (1, 1, 0, 0)
    two = mv[:4] + [("L", grid(6, 9), grid(3, 9))]; assert get(F6.evaluate(two, rot=True))["rot_pred"] == 0         # only U and D known: two turns missing, nothing is guessed
    tried = mv[:6] + [("R", grid(0, 9), grid(0, 9)), ("R", grid(0, 9), grid(0, 9))]                                # R was tried (and did nothing): its second try is not a first try
    t = get(F6.evaluate(tried, rot=True)); assert (t["rot_pred"], t["rot_hit"]) == (1, 0)                          # only the first try is guessed
    click = three(("R@2,3", grid(0, 9), grid(0, 9))); assert get(F6.evaluate(click, rot=True))["rot_pred"] == 0   # a click is never guessed
    once = F6.evaluate(three(("U", grid(0, 9), grid(0, 6))), rot=True); assert once.get("rot_pred", 0) == 0       # U was tried before: it has its own law
    both = get(F6.evaluate(mv, rot=True, once=True)); assert (both["rot_pred"], both["once_pred"], both["predicted"]) == (1, 3, 4) and get(F6.evaluate(mv, once=True))["predicted"] == 3      # the tiers add


def test_rel_predicts_from_a_relation_carried_from_earlier_levels_and_never_from_the_level_itself():
    s = [("U", grid(6, 9), grid(6, 6)), ("U", grid(6, 6), grid(6, 3)), ("D", grid(6, 3), grid(6, 6))]               # U known; D tried for the first time and steps (0, 3)
    rel = {("D", "U"): 2, ("U", "D"): 2}
    c = get(F6.evaluate(s, rel=True, earlier_rel=rel)); assert (c["predicted"], c["rel_pred"], c["rel_hit"], c["cell_hit"]) == (1, 1, 1, 1)
    assert get(F6.evaluate(s, rel=True))["predicted"] == 0 and get(F6.evaluate(s, earlier_rel=rel))["predicted"] == 0                                       # no carried relation, or the switch off: nothing
    wrong = F6.evaluate(s[:2] + [("D", grid(6, 3), grid(6, 3))], rel=True, earlier_rel=rel); assert (wrong["rel_pred"], wrong["rel_hit"]) == (1, 0)           # D did nothing this time: the relation was wrong here
    other = get(F6.evaluate(s, rel=True, earlier_rel={("D", "U"): 0})); assert (other["rel_pred"], other["rel_hit"]) == (1, 0)                                # a relation that does not hold in this level
    both = three(("R", grid(0, 9), grid(3, 9)))                                                                      # rot would say (3, 0); the carried relation R = U not turned says (0, -3): the relation is read first
    b = get(F6.evaluate(both, rot=True, rel=True, earlier_rel={("R", "U"): 0})); assert (b["rel_pred"], b["rel_hit"], b["rot_pred"]) == (1, 0, 0)
    r = get(F6.evaluate(both, rot=True, rel=True, earlier_rel={("R", "Q"): 0})); assert (r["rel_pred"], r["rot_pred"], r["rot_hit"]) == (0, 1, 1)             # a relation to an action that is not here: the rotation set answers


def test_the_carried_relation_comes_from_earlier_levels_only():
    lv1 = three(("R", grid(0, 9), grid(3, 9)))                                                                       # level 1: the relations among U, D, L
    lv2 = [("U", grid(6, 9), grid(6, 6)), ("U", grid(6, 6), grid(6, 3)), ("D", grid(6, 3), grid(6, 6))]               # level 2: D's first try, answerable from level 1's relation D > U = 2
    s = Sess({("g", 1): led(lv1), ("g", 2): led(lv2)}); w = F6.from_session(s)
    assert set(w) == set(F6.VARIANTS) and set(w["rel"]) == {"g"}
    c = get(w["rel"]["g"]); assert (c["rel_pred"], c["rel_hit"]) == (1, 1)                                           # only level 2 predicts: level 1 has no earlier level
    lv1_alone = Sess({("g", 1): led(lv1 + [("D", grid(3, 9), grid(3, 6))])}); assert get(F6.from_session(lv1_alone)["rel"]["g"])["rel_pred"] == 0               # a level does not use its own relations
    swapped = Sess({("g", 1): led(lv2), ("g", 2): led(lv1)}); assert get(F6.from_session(swapped)["rel"]["g"])["rel_pred"] == 0                              # level 1 knows too little: nothing carried
    other = Sess({("h", 1): led(lv1), ("g", 2): led(lv2)}); assert get(F6.from_session(other)["rel"]["g"])["rel_pred"] == 0                                   # a relation from another game is not carried
    assert get(w["full"]["g"])["predicted"] == 0 and get(w["all"]["g"])["rel_pred"] == 1


# ---------------------------------------------------------------- 4 the tiers, the names, the screen without its clock
def test_every_prediction_carries_its_tier_and_the_tiers_add_up_to_the_predicted_moves():
    seq = [("S", row(i), row(i + 1)) for i in range(1, 5)] + [("S", row(5), row(5))] * 3
    for kw in (dict(), dict(once=True), dict(once=True, rot=True, rel=True)):
        c = F6.evaluate(seq, **kw); assert sum(c.get(s + "_pred", 0) for s in F6.SOURCES) == c["predicted"] and sum(c.get(s + "_hit", 0) for s in F6.SOURCES) == c["cell_hit"]
    c = get(F6.evaluate(seq, once=True)); assert (c["once_pred"], c["once_hit"], c["law_pred"]) == (1, 1, 5) and get(F6.evaluate(seq))["law_pred"] == 5
    mv = three(("R", grid(0, 9), grid(3, 9))); a = F6.evaluate(mv, once=True, rot=True, rel=True, earlier_rel={}); assert sum(a.get(s + "_pred", 0) for s in F6.SOURCES) == a["predicted"] == 4 and a["rot_pred"] == 1 and a["once_pred"] == 3
    r = F6.rates(F6.combine([F6.evaluate(mv, rot=True), F6.evaluate(three(("A", grid(0, 9), grid(0, 9))), rot=True)])); assert r["rot_pred"] == 2 and r["rot_hit"] == 1 and r["rot_precision"] == 0.5


def test_the_direction_the_length_and_the_quarter_turns_of_each_level_are_counted_for_the_names():
    c = F6.evaluate(three(("R", grid(0, 9), grid(3, 9))) + [("R", grid(3, 9), grid(6, 9))])
    assert c["dir:U:0,-1"] == c["dir:D:0,1"] == c["dir:L:-1,0"] == c["dir:R:1,0"] == 1 and c["len:U:3"] == c["len:R:3"] == 1
    assert c["relk:D>U:2"] == 1 and c["relk:R>U:1"] == 1 and c["relk:U>R:3"] == 1 and sum(v for k, v in c.items() if k.startswith("relk:")) == 12
    dr, ln, rl = F6.name_counts(F6.combine([c, c])); assert dr["U"] == {"0,-1": 2} and ln["R"] == {3: 2} and rl["D>U"] == {2: 2} and rl["R>U"] == {1: 2}
    far = F6.evaluate([("A", grid(2, 9), grid(7, 9)), ("A", grid(7, 9), grid(12, 9))]); assert far["dir:A:1,0"] == 1 and far["len:A:5"] == 1                       # a step of five cells


def test_the_screen_is_counted_again_with_the_strip_that_ticks_left_out():
    grow = [("S", row(5, n), row(5, n + 1)) for n in range(10)]                                                      # the piece stays; a bar along the edge grows every move
    c = get(F6.evaluate(grow)); assert c["moves"] == 10 and c["rep_screen"] == 0 and c["tick_levels"] == 1 and c["mrep"] == 8 and c["mrep_null"] == 8 and c["mrep_null_null"] == 8
    still = [("S", row(5), row(5))] * 6; d = get(F6.evaluate(still)); assert d["rep_screen"] == 5 and d["tick_levels"] == 0 and d["mrep"] == 0                  # no ticking strip: nothing to leave out
    moving = [("S", row(i, i), row(i + 1, i + 1)) for i in range(1, 6)]; assert get(F6.evaluate(moving))["mrep"] == 0                                       # the piece moves every time: the masked screen is new every time
    osc = [("S", row(5, n % 4), row(5, (n + 1) % 4)) for n in range(12)]; o = get(F6.evaluate(osc)); assert o["rep_screen"] == 8 and o["mrep"] == 10 and o["tick_levels"] == 1       # the strip leaves out two repeats the whole screen already caught, and more


# ---------------------------------------------------------------- 5 the delete test again
def test_deleting_the_floor_and_the_group_together_changes_the_counts_on_the_invented_worlds_and_nothing_else_is_touched():
    out = {}
    for nm, mk in (("pad", lambda: World(PAD_LEVEL)), ("push", lambda: World(PUSH_LEVEL, pads=False))):
        seqs = [walk(mk(), lv, 60) for lv in range(1, 6)]
        out[nm] = {v: F6.combine([F6.evaluate(s, **fl) for s in seqs]) for v, fl in F6.VARIANTS.items()}
        ref = {v: F5.combine([F5.evaluate(s, **fl) for s in seqs]) for v, fl in F5.VARIANTS.items()}
        assert all(out[nm][v]["cell_hit"] == ref[v]["cell_hit"] and out[nm][v]["predicted"] == ref[v]["predicted"] for v in ("full", "-floor", "-group", "once"))     # the single deletions and once are the fifth check's
    hits = lambda nm: {v: c["cell_hit"] for v, c in out[nm].items()}
    assert hits("pad") == {"full": 164, "-floor": 165, "-group": 166, "-floor-group": 165, "once": 178, "rot": 164, "rel": 164, "rot+rel": 164, "all": 178}
    assert hits("push") == {"full": 162, "-floor": 165, "-group": 159, "-floor-group": 164, "once": 176, "rot": 162, "rel": 162, "rot+rel": 162, "all": 176}
    for nm in out:
        assert all(c["predicted"] == 193 for v, c in out[nm].items() if v in ("full", "-floor", "-group", "-floor-group")) and out[nm]["once"]["predicted"] == 206 and out[nm]["all"]["once_pred"] == 13
    assert out["pad"]["rot"]["rot_pred"] == 1 and out["pad"]["rot"].get("rot_hit", 0) == 0 and out["push"]["rot"].get("rot_pred", 0) == 0 and out["pad"]["rel"].get("rel_pred", 0) == 0      # on the invented worlds the new parts predict almost nothing


# ---------------------------------------------------------------- 6 the reading lines
GAMES = ("g50t-1", "ka59-1", "ls20-1", "sp80-1")
SPLIT = lambda total: [total // 4 + (1 if i < total % 4 else 0) for i in range(4)]


def _c(predicted=182, hit=150, **kw):
    d = dict(moves=234, predicted=predicted, cell_hit=hit, mrep=8, cell_said=0, cell_right=0, known=predicted, did_not_move=40, rep_pos_null_null=10)
    d.update(kw); return d


def _names(**kw):
    base = {f"dir:ACTION{i}:{s}": 10 for i, s in ((1, "0,-1"), (2, "0,1"), (3, "-1,0"), (4, "1,0"))}
    base.update({f"relk:{p}:{k}": 10 for p, k in zip(F6.PAIRS, (2, 1, 1, 3, 3, 2))}); base.update(kw); return base


def _w(hits, preds, **extra):
    """One variant across the four games from totals (right, predicted); `extra` is {key: total} spread over the games."""
    out = {g: _c(predicted=p, hit=h) for g, p, h in zip(GAMES, SPLIT(preds), SPLIT(hits))}
    for k, tot in extra.items():
        for g, v in zip(GAMES, SPLIT(tot)): out[g][k] = v
    return out


def _v(full=None, nofloor=None, once=None, rot=None, rel=None, allv=None, both=None, names=None, mrep=None, ok=True):
    fv = full or _w(603, 729, mrep=mrep if mrep is not None else 40); fv["g50t-1"].update(names or _names())
    v = {"full": fv, "-floor": nofloor or _w(618, 729), "once": once or _w(663, 797), "rot": rot or _w(603, 729, rot_pred=10, rot_hit=8), "rel": rel or _w(603, 729, rel_pred=12, rel_hit=10),
         "all": allv or _w(603 + 100, 729 + 120), "-floor-group": both or _w(620, 729)}
    return {i: x for i, _, _, x in F6.verdict(v, ok)}


def test_the_nine_lines_flip_at_their_boundaries_and_the_headline_needs_lines_4_and_5():
    assert len(F6.G_LINES) == 9 and [t for t, _ in F6.G_LINES] == ["guard", "instrument"] + ["claim"] * 7
    assert {i: x for i, _, _, x in F6.verdict(None, None)} == {i: None for i in range(1, 10)} and _v(ok=False)[1] is False and _v()[1] is True
    assert _v()[2] is True and _v(full=_w(603, 728))[2] is False and _v(full=_w(602, 729))[2] is False and _v(nofloor=_w(617, 729))[2] is False and _v(once=_w(663, 798))[2] is False   # the three figures of the earlier checks
    r = lambda n, h: _w(603, 729, rot_pred=n, rot_hit=h)
    assert _v(rot=r(8, 6))[3] is True and _v(rot=r(8, 5))[3] is False and _v(rot=r(7, 7))[3] is False and _v(rot=r(0, 0))[3] is False                      # 75 percent of 8; 62.5 percent of 8; 7 is below 8; nothing predicted
    l = lambda n, h: _w(603, 729, rel_pred=n, rel_hit=h)
    assert _v(rel=l(10, 7))[4] is True and _v(rel=l(10, 6))[4] is False and _v(rel=l(9, 9))[4] is False and _v(rel=l(0, 0))[4] is False
    assert _v(allv=_w(603 + 72, 729 + 90))[5] is True and _v(allv=_w(603 + 71, 729 + 90))[5] is False and _v(allv=_w(603 + 89, 729 + 89))[5] is False and _v(allv=_w(603, 729))[5] is False   # 90 more, 72 right is 80 percent
    assert _v(names=_names(**{"dir:ACTION1:0,1": 1}))[6] is True                                                                                 # 10 of 11 = 91 percent
    assert _v(names=_names(**{"dir:ACTION1:0,1": 2}))[6] is False                                                                                # 10 of 12 = 83 percent
    assert _v(names=_names(**{"dir:ACTION2:0,1": 3, "dir:ACTION2:1,0": 0}))[6] is False and _v(names=_names(**{"dir:ACTION3:-1,0": 3}))[6] is False  # made a step in only 3 levels: fewer than 4
    gone = _names(); del gone["dir:ACTION4:1,0"]; assert _v(names=gone)[6] is False                                                               # an action never seen making a step is not yes
    assert _v(names=_names(**{"dir:ACTION1:0,-1": 9, "dir:ACTION1:1,0": 1}))[6] is True                                                           # exactly 90 percent
    assert _v(names=_names(**{"relk:ACTION1>ACTION2:1": 4}))[7] is False and _v(names=_names(**{"relk:ACTION1>ACTION2:1": 1}))[7] is True        # 10 of 11 yes; 10 of 14 NO
    nopair = _names(); del nopair["relk:ACTION3>ACTION4:2"]; assert _v(names=nopair)[7] is False
    assert _v(both=_w(613, 729))[8] is True and _v(both=_w(612, 729))[8] is False and _v(both=_w(603, 729))[8] is False                            # 10 more right than the full 603
    assert _v(mrep=30)[9] is True and _v(mrep=29)[9] is False and _v(mrep=0)[9] is False
    allyes = [(i, t, x, True) for i, t, x, _ in F6.verdict(None, None)]
    assert F6.headline(allyes) == F6.HEADLINE and F6.claims_allowed(allyes)
    assert F6.headline([(i, t, x, (False if i == 5 else v)) for i, t, x, v in allyes]) == "MIXED: line 4 yes, line 5 NO"
    assert F6.headline([(i, t, x, (False if i == 4 else v)) for i, t, x, v in allyes]) == "MIXED: line 4 NO, line 5 yes"
    assert F6.headline([(i, t, x, (False if i == 2 else v)) for i, t, x, v in allyes]).startswith("NOT READ") and F6.headline(F6.verdict(None, None)).startswith("NOT READ")
    assert F6.headline([(i, t, x, (False if i == 1 else v)) for i, t, x, v in allyes]).startswith("NOT READ")


def test_the_text_of_a_run_says_what_each_section_found():
    mv = three(("R", grid(0, 9), grid(3, 9))) + [("R", grid(3, 9), grid(6, 9))]
    s = Sess({("g", 1): led(mv), ("g", 2): led(mv)}); w = F6.from_session(s); t = F6.text(w)
    for n in ("== 1 first tries answered", "== 2 the tiers", "== 3 the delete test again", "== 4 the names", "== 5 the screen without its clock", "rot: ", "rel: ", "all: ", "-floor-group", "ALL GAMES", "U: (0,-1) in 2"): assert n in t, n
    assert F6.text({"full": {}}) == ""


# ---------------------------------------------------------------- 7 page, notebook, runner, replay of arm C, front pages
def _head():
    text = open(PAGE, encoding="utf-8").read(); assert text.count(RESULT_MARK) <= 1
    return text[:text.index(RESULT_MARK)] if RESULT_MARK in text else text


def test_the_page_above_its_result_is_what_was_written_before_the_run():
    assert hashlib.sha256(_head().encode()).hexdigest() == PAGE_SHA256


def test_the_page_and_the_module_agree_and_it_says_which_text_is_whose():
    head = _head(); pred = head.split("## Predictions")[1].split("## Reading rules")[0]; rules = head.split("## Reading rules")[1].split("Headline:")[0]
    ps = re.findall(r"^(\d+)\. \[(\w+)\] (.*) \((\d+) %\)$", pred, re.M); rs = re.findall(r"^(\d+)\. (.*)$", rules, re.M)
    assert len(ps) == len(rs) == len(F6.G_LINES) == 9
    for (n, tag, text, pr), (m, rt), (gt, gtext) in zip(ps, rs, F6.G_LINES): assert n == m and text == rt == gtext and tag == gt and 0 < int(pr) < 100
    assert [int(p[3]) for p in ps] == [95, 85, 35, 50, 35, 50, 60, 50, 55]
    for needle in (F6.HEADLINE, "Sir to confirm", "Saraswati Choubey", "Radha Gupta", "Rahul Chauhan", "FUTURE OF AI", "steered the AI", "Delete Test", "603 right of 729", "663 of 797", "618 right of 729", "88 of the 183", "207", "blind spot", "will not show", "paid test", "$3.65",
                   "Result", "not written yet", "Nine ways", "invented worlds", "Concept-map item 141", "which sentences are Sir's own", "Spherical Mind", "24-hour clock", "A name defines scope", "the delete button you chose not to use"): assert needle in head, needle
    assert "proven" not in head.lower() and "ledger and the predictor of the third check are not changed" in head and "Nothing on this page has been run on the real games" in head


def test_the_notebook_and_the_runner_read_the_saved_files_call_no_model_hold_no_key_and_run_nine_ways():
    nb = json.load(open(NB)); src = "\n".join("".join(c["source"]) for c in nb["cells"]); code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert len(nb["cells"]) == 12 and all(c["outputs"] == [] for c in nb["cells"] if c["cell_type"] == "code")
    run = open(RUN, encoding="utf-8").read()
    for text in (src, run):
        assert not re.search(r"AIza[0-9A-Za-z_-]{20,}", text) and "GOOGLE_API_KEY" not in text and "userdata" not in text and "drive.mount" not in text and "open_store" not in text
        assert "foresee5" not in text and "foresee4" not in text and "generate" not in text and ".run(" not in text.replace("LR.run_file(", "")
    for needle in ("foresee6 as F", "F.from_session", "LR.run_file", "F.verdict", "F.headline", "F.VARIANTS", "axi_gemini_arc_c_default.json", "axi_gemini_arc_b_default.json", "axi_gemini_arc_default.json", "foresee_check_6_all.json", "foresee6.py"): assert needle in code, needle
    for needle in ("foresee6 as F", "F.from_session", "LR.run_file", "F.verdict", "F.headline", "foresee_check_6_all.json", "foresee_check_6_printout.txt", "axi_gemini_arc_c_default.json"): assert needle in run, needle
    assert "FORESEE_CHECK_6.md" in src and "verdict(per_game if allok else None, allok if ok else None)" in code and "verdict(per_game if allok else None, allok if ok else None)" in run
    for c in nb["cells"]:
        if c["cell_type"] == "code": compile("\n".join(l for l in "".join(c["source"]).split("\n") if not l.lstrip().startswith("!")), "cell", "exec")
    compile(run, "run_foresee_check_6.py", "exec")


def test_the_sixth_check_runs_over_a_replay_of_arm_c_on_the_stand_in_game_nine_ways():
    res = C.run(T.FakeLLM(), T.make_env, os.path.join(os.environ.get("TMPDIR", "/tmp"), "foresee6_c_test.jsonl"), games=("alpha-1", "beta-2"), moves=30, action_of=lambda n: n)
    s = LR.feed(res["rows"], T.make_env, "C", games=("alpha-1", "beta-2"), moves=30, action_of=lambda n: n); w = F6.from_session(s)
    assert set(w) == set(F6.VARIANTS) and all(set(pg) == {"alpha-1", "beta-2"} for pg in w.values())
    assert all(sum(c["moves"] for c in pg.values()) == sum(s.moves.values()) for pg in w.values()) and "ALL GAMES" in F6.text(w) and "the delete test" in F6.text(w) and "the names" in F6.text(w)
    f5 = F5.from_session(s)
    for g in w["full"]:
        a = F3.combine([F3.evaluate(((r.action, r.kb, r.ka) for r in L.rows), True, True) for (gg, lv), L in sorted(s.ledgers.items()) if gg == g])
        assert all(w["full"][g].get(k, 0) == v for k, v in a.items())                                                                              # the full way is the third check on the replayed levels
        assert all(w["full"][g].get(k, 0) == v for k, v in f5["full"][g].items() if not k.startswith(("step:", "rel:")))                         # and the fifth check's full way
        assert w["-floor"][g]["cell_hit"] == f5["-floor"][g]["cell_hit"] and w["-group"][g]["cell_hit"] == f5["-group"][g]["cell_hit"] and w["once"][g]["cell_hit"] == f5["once"][g]["cell_hit"]
    checks = F6.verdict(w, True); assert len(checks) == 9 and checks[0][3] is True and F6.headline(checks)


def test_the_front_pages_say_what_was_built_and_that_it_was_not_run_on_the_real_games():
    cm = open(os.path.join(DOCS, "RAJNISH_CONCEPT_MAP.md"), encoding="utf-8").read(); vf = open(os.path.join(DOCS, "VERIFIED.md"), encoding="utf-8").read(); hl = open(os.path.join(DOCS, "HISTORY_LEDGER.md"), encoding="utf-8").read()
    assert NEXT in cm and cm.count("Next free:") == 1 and "\n140. **" in cm and "\n141. **" in cm and "\n145. **" not in cm
    i = cm.index("\n141. **") + 1; item = cm[i:cm.index("\n", i)]
    for n in ("FORESEE_CHECK_6.md", "foresee6.py", "test 166", "Saraswati Choubey", "Radha Gupta", "Rahul Chauhan", "FUTURE OF AI", "Sir to confirm", "Not run on the real games", "9 reading lines", "blind spot", "Sir steered", PAGE_SHA256): assert n in item, n
    assert any("FORESEE_CHECK_6.md" in l and "foresee6.py" in l and PAGE_SHA256 in l and "RUN by the owner" in l for l in vf.split("\n"))
    assert "| 187 (P) |" in hl and "Sir" in hl and "blind spot" in hl and "foresee_check_5_printout.txt" in cm
    assert os.path.exists(os.path.join(DOCS, "results", "foresee_check_5_printout.txt"))


if __name__ == "__main__":
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]; bad = 0
    for n, f in fns:
        try: f(); print("ok  ", n)
        except Exception: bad += 1; print("FAIL", n); traceback.print_exc()
    print(f"{len(fns) - bad}/{len(fns)} passed"); sys.exit(1 if bad else 0)
