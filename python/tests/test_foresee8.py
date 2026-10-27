"""Test 168 (item 143): the eighth forward check (docs/FORESEE_CHECK_8.md, axi/experiments/foresee8.py, colab/axi_foresee_check_8.ipynb, colab/run_foresee_check_8.py): the name prior pooled over the four arms, the retire rule with its break-even cost, the six level clears.

The seventh check read MIXED (line 4 NO, line 5 NO). Its name prior was built inside one arm at a time and was absent in 40 of the 72 levels with a step, so its line 3 counted how often a prior existed; after a law had been wrong by a different step the later law predictions were right in 60.9 percent against 85.9 before; six levels were cleared and nothing looked at them. This test checks, by hand and on small worlds built for it:
  1. with `retire` 0 the counters are the seventh check's `evaluate`, counter for counter (and so the sixth's, the fifth's and the third's), on the invented worlds and on the small worlds; the only new counters are the prior-present ones;
  2. the retire rule: a law-tier prediction that has been wrong by a different step `retire` times is left to the model, per action, in that level only; the once tier and the other tiers are not touched; what it withholds is counted and is shown as "retired";
  3. the prior-present counters (`pri:*:has`, `pri:*:has_ok`), the pooled prior (read from the other games of ALL the arms, never from the game itself; per arm it is the seventh check's), the cost of a retire rule (right moves lost, wrong moves avoided, the ratio);
  4. the clears: the session that keeps the screen before a clearing move, what the screen shows against the first screen with the strip's cells left out, the study over cleared and uncleared levels;
  5. the 12 reading lines at their boundaries, the page (its hash, its lines, its probabilities, what it says it will not show), the notebook and the runner (no key, no model), the replay of the stand-in game, the front pages.
What it does NOT show: how the check reads on the real games. That is the Result of the run (Codespaces or Colab)."""
import hashlib, json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
from collections import Counter
from axi.engine.ledger import Ledger, key
from axi.experiments import foresee3 as F3
from axi.experiments import foresee5 as F5
from axi.experiments import foresee6 as F6
from axi.experiments import foresee7 as F7
from axi.experiments import foresee8 as F8
from axi.experiments import ledger_replay as LR
from axi.experiments import gemini_arc_c as C
from axi.experiments.mini_worlds import World, walk, PUSH_LEVEL, PAD_LEVEL
from tests import test_gemini_arc as T
from tests.test_foresee6 import grid, row, three, led, Sess, tall_seq
from tests.test_foresee7 import four, big, _tall, _bar, NAMES, A, UP, DOWN, LEFT, RIGHT, _games

HERE = os.path.dirname(os.path.abspath(__file__)); DOCS = os.path.join(HERE, "..", "..", "docs")
PAGE = os.path.join(DOCS, "FORESEE_CHECK_8.md"); NB = os.path.join(HERE, "..", "colab", "axi_foresee_check_8.ipynb"); RUN = os.path.join(HERE, "..", "colab", "run_foresee_check_8.py")
PAGE_SHA256 = "97a8e1d6da1f330073d0652133c942cb677a2780c25b43de5cc9645e172e1491"; RESULT_MARK = "\n## Result"
NEXT = "Next free: row 193, item 146, test 171, question 122"
NEW_KEYS = ("retired_withheld",)


def same_but_new(b, a):
    """Every counter of `a` is in `b` with the same value, and `b` has nothing else but the prior-present counters."""
    return all(b.get(k, 0) == v for k, v in a.items()) and all(k.startswith("pri:") and k.endswith((":has", ":has_ok")) for k in b if k not in a)


# ---------------------------------------------------------------- 1 retire 0: the seventh check, counter for counter
def test_with_retire_off_the_counters_are_the_seventh_checks_evaluate_on_the_invented_worlds_and_the_small_worlds():
    kws = (dict(), dict(once=True), dict(once=True, rot=True, rel=True), dict(once=True, rot=True, rel=True, nm=True, scr=True, names=NAMES, carry=("P", 3)),
           dict(once=True, rot=True, rel=True, nm=True, scr=True, floor=False, group=False, names=NAMES, carry=("P", 3)), dict(floor=False), dict(group=False), dict(edge=False))
    for mk in (lambda: World(PUSH_LEVEL, pads=False), lambda: World(PAD_LEVEL)):
        for lv in range(1, 6):
            seq = walk(mk(), lv, 60)
            for kw in kws: assert same_but_new(F8.evaluate(seq, **kw), F7.evaluate(seq, **kw))
            assert all(F8.evaluate(seq).get(k, 0) == v for k, v in F6.evaluate(seq).items()) and all(F8.evaluate(seq).get(k, 0) == v for k, v in F3.evaluate(seq, True, True).items())
            assert all(F8.evaluate(seq).get(k, 0) == v for k, v in F5.evaluate(seq).items() if not k.startswith(("step:", "rel:")))
    for seq in (tall_seq(), tall_seq(12), [("S", row(i), row(i + 1)) for i in range(1, 5)] + [("S", row(5), row(5))] * 8, three(("R", grid(0, 9), grid(3, 9))), four(), _tall([36, 33, 30, 27, 25, 22, 19])):
        for kw in kws[:4]: assert same_but_new(F8.evaluate(seq, **kw), F7.evaluate(seq, **kw))
    seq = walk(World(PAD_LEVEL), 3, 60); assert same_but_new(F8.evaluate(seq, earlier_rel={("D", "U"): 2}, once=True, rot=True, rel=True), F7.evaluate(seq, earlier_rel={("D", "U"): 2}, once=True, rot=True, rel=True))
    assert not any(k in F8.evaluate(seq) for k in NEW_KEYS) and not any(k.startswith("pri:") for k in F8.evaluate(seq))                                  # no names given: no prior counters at all


def test_the_check_is_a_pure_function_and_changes_nothing_it_is_given():
    seq = walk(World(PAD_LEVEL), 3, 120); snap = json.dumps(seq); names = dict(NAMES); carry = ("P", 3)
    kw = dict(once=True, rot=True, rel=True, nm=True, scr=True, retire=1, earlier_rel={("D", "U"): 2}, names=names, carry=carry); a = F8.evaluate(seq, **kw); assert json.dumps(seq) == snap and names == NAMES
    assert a == F8.evaluate(list(seq), **kw) and F8.text(None, None, None) == "" and F8.rates({})["precision"] is None and F8.rates({})["retired_withheld"] == 0
    assert list(F8.VARIANTS) == ["full", "once", "all6", "nm", "all7", "best", "all7_r1", "all7_r2", "best_r1"] and len(F8.G_LINES) == 12 and F8.CLEARS_SEEN == 6 and F8.ACTIONS == A
    assert F8.CHECK7 == {"full": (603, 729), "once": (663, 797), "all6": (672, 809), "all7": (690, 832), "best": (703, 832)}
    assert F8.VARIANTS["all7_r1"] == dict(F8.VARIANTS["all7"], retire=1) and F8.VARIANTS["all7_r2"] == dict(F8.VARIANTS["all7"], retire=2) and F8.VARIANTS["best_r1"] == dict(F8.VARIANTS["best"], retire=1)
    assert all(F8.VARIANTS[v] == F7.VARIANTS[v] for v in F7.VARIANTS)                                                                     # the seven's six ways are untouched
    L = led(four()); snapshot = (len(L.rows), L.dominant()); F8.level_facts(L, L.rows[-1].ka, "ACTION1"); assert (len(L.rows), L.dominant()) == snapshot


# ---------------------------------------------------------------- 2 the retire rule
def _r(seq, **kw): return F8.evaluate(seq, **kw)


def test_a_law_wrong_by_a_different_step_is_not_used_again_in_that_level_with_retire_1():
    seq = _tall([36, 33, 30, 27, 25, 22, 19, 16, 13])                                   # (0,-3) x3, then (0,-2), then (0,-3) x4
    off = _r(seq); one = _r(seq, retire=1)
    assert (off["moves"], off["predicted"], off["cell_hit"], off["fail_events"], off.get("retired_withheld", 0)) == (8, 6, 5, 1, 0)           # moves 3 to 8 are predicted; the 4th is wrong
    assert (one["predicted"], one["cell_hit"], one["fail_events"], one["retired_withheld"], one["abs:retired"], one["abstained"]) == (2, 1, 1, 4, 4, 6)    # after the failure the law is not used: 4 moves left to the model
    assert one["moves"] == off["moves"] and one["law_pred"] == 2 and one.get("postfail_pred", 0) == 0 and one["prefail_pred"] == 2 and one["prefail_hit"] == 1
    assert F8.rates(one)["precision"] == 0.5 and abs(F8.rates(off)["precision"] - 5 / 6) < 1e-12 and F8.rates(one)["retired_withheld"] == 4
    steady = _tall([36, 33, 30, 27, 24, 21]); assert _r(steady, retire=1) == _r(steady) and _r(steady, retire=1).get("retired_withheld", 0) == 0          # no failure, nothing withheld


def test_retire_2_waits_for_a_second_wrong_step():
    one_fail = _tall([36, 33, 30, 27, 25, 22, 19, 16, 13]); assert _r(one_fail, retire=2) == _r(one_fail)                                    # one failure is not two
    two = _tall([48, 45, 42, 39, 37, 34, 32, 29, 26, 23])                              # (0,-3) x3, (0,-2), (0,-3), (0,-2), (0,-3) x3: the law is wrong by a different step three times
    off, r1, r2 = _r(two), _r(two, retire=1), _r(two, retire=2)
    assert (off["fail_events"], off["predicted"], off.get("retired_withheld", 0)) == (3, 5, 0)
    assert (r1["fail_events"], r1["predicted"], r1["retired_withheld"]) == (1, 1, 4) and (r2["fail_events"], r2["predicted"], r2["retired_withheld"]) == (2, 3, 2)      # retire 1 stops at the first failure, retire 2 at the second
    assert r1["predicted"] < r2["predicted"] < off["predicted"] and r2["predicted"] + r2["retired_withheld"] == off["predicted"] and r1["predicted"] + r1["retired_withheld"] == off["predicted"]


def test_the_rule_retires_the_law_tier_of_one_action_only_and_in_that_level_only():
    mixed = [("U", grid(6, 12), grid(6, 9)), ("U", grid(6, 9), grid(6, 6)), ("U", grid(6, 6), grid(6, 3)), ("U", grid(6, 3), grid(6, 1)),          # U fails once
             ("D", grid(6, 1), grid(6, 4)), ("D", grid(6, 4), grid(6, 7)), ("D", grid(6, 7), grid(6, 10)), ("D", grid(6, 10), grid(6, 13)),         # D has its own law
             ("U", grid(6, 13), grid(6, 10)), ("D", grid(6, 10), grid(6, 13))]
    off, one = _r(mixed), _r(mixed, retire=1)
    assert one["retired_withheld"] == 1 and one["abs:retired"] == 1 and one["act_pred:U"] == off["act_pred:U"] - 1 and one["act_pred:D"] == off["act_pred:D"]        # only U, and only its last move
    again = _r(mixed[:4], retire=1); assert again.get("retired_withheld", 0) == 0                                                              # a fresh level starts with nothing retired
    assert _r(mixed, retire=1, names=NAMES).get("retired_withheld") == 1                                                                          # names given: no change here


def test_the_retire_rule_leaves_the_once_tier_and_the_other_tiers_alone_and_never_adds_a_prediction():
    seq = walk(World(PAD_LEVEL), 3, 80)
    for kw in (dict(once=True), dict(once=True, rot=True, rel=True, nm=True, names=NAMES, carry=("P", 3)), dict(once=True, rot=True, rel=True, nm=True, scr=True, floor=False, group=False, names=NAMES)):
        off, on = _r(seq, **kw), _r(seq, retire=1, **kw)
        assert on["predicted"] + on.get("retired_withheld", 0) == off["predicted"] and on["moves"] == off["moves"]                           # what it withholds is what the other way predicted
        assert all(on.get(s + "_pred", 0) <= off.get(s + "_pred", 0) for s in F7.SOURCES)
    c = _r(_tall([36, 33, 30, 27, 25, 22, 19, 16]), once=True, nm=True, names=NAMES, retire=1); assert sum(c.get(s + "_pred", 0) for s in F7.SOURCES) == c["predicted"] and sum(c.get(s + "_hit", 0) for s in F7.SOURCES) == c["cell_hit"]


# ---------------------------------------------------------------- 3 the prior-present counters, the pooled prior, the cost
def test_the_prior_present_counters_say_whether_a_prior_existed_and_whether_it_named_the_direction_made():
    mv = four(); g = F8.evaluate(mv, names={"ACTION1": UP, "ACTION2": UP})
    assert (g["pri:ACTION1:n"], g["pri:ACTION1:has"], g["pri:ACTION1:has_ok"]) == (1, 1, 1) and (g["pri:ACTION2:n"], g["pri:ACTION2:has"], g["pri:ACTION2:has_ok"]) == (1, 1, 0)
    assert (g["pri:ACTION3:n"], g.get("pri:ACTION3:has", 0), g.get("pri:ACTION3:has_ok", 0)) == (1, 0, 0) and g["pri:ACTION4:ok"] == 0
    none = F8.evaluate(mv, names={}); assert none["pri:ACTION1:n"] == 1 and none.get("pri:ACTION1:has", 0) == 0 and none["pri:ACTION1:ok"] == 0
    assert not any(k.startswith("pri:") for k in F8.evaluate(mv, names=None))                                                                # names None: not counted
    full = F8.evaluate(mv, names=dict(zip(A, (UP, DOWN, LEFT, RIGHT)))); assert all(full[f"pri:{a}:has"] == full[f"pri:{a}:has_ok"] == full[f"pri:{a}:n"] == 1 for a in A)


def _arms(n_levels=2):
    """Two arms that each hold game a (the four actions, 3 cells) and game b (ACTION1 twice, then ACTION2 once, 3 cells)."""
    def b(): return led([("ACTION1", big(7, 7), big(7, 4)), ("ACTION1", big(7, 4), big(7, 1)), ("ACTION2", big(7, 1), big(7, 4))])
    return {arm: Sess({**{("a", i): led(four(names=A)) for i in range(1, n_levels + 1)}, **{("b", i): b() for i in range(1, n_levels + 1)}}) for arm in ("X", "Y")}


def test_the_prior_pooled_over_the_arms_gives_a_game_what_one_arm_has_too_little_of_and_a_game_never_reads_its_own_levels():
    s = _arms(); per = F8.from_sessions(s, False); pooled = F8.from_sessions(s, True)
    assert set(pooled) == {"X", "Y"} and set(pooled["X"]) == set(F8.VARIANTS) and set(pooled["X"]["nm"]) == {"a", "b"}
    assert per["X"]["nm"]["b"].get("nm_pred", 0) == 0                                                                    # one arm: 2 levels of game a, below the 4 the prior needs
    assert all(pooled[arm]["nm"]["b"]["nm_pred"] == pooled[arm]["nm"]["b"]["nm_hit"] == 2 for arm in ("X", "Y"))        # two arms: 4 levels of ACTION2 = down, so ACTION2's first try in every level of b is named
    assert all(per[arm]["full"][g].get(f"pri:{a}:has", 0) == 0 for arm in per for g in ("a", "b") for a in A)                  # one arm: no prior in any level
    p = F8.pool(pooled); assert set(p) == set(F8.VARIANTS) and set(p["nm"]) == {"a", "b"} and p["nm"]["b"]["nm_pred"] == 4 and p["full"]["a"]["moves"] == 2 * 2 * 8
    c = p["full"]["b"]; assert (c["pri:ACTION1:n"], c["pri:ACTION1:has"], c["pri:ACTION1:has_ok"]) == (4, 4, 4) and "pri:ACTION2:n" not in c                  # b's ACTION1 is named by the 4 levels of UP in game a; b's ACTION2 made no step (one try)
    c = p["full"]["a"]; assert (c["pri:ACTION1:n"], c["pri:ACTION1:has"], c["pri:ACTION1:has_ok"]) == (4, 4, 4) and (c["pri:ACTION2:n"], c["pri:ACTION2:has"]) == (4, 0)      # a's prior comes from b, which only knows ACTION1
    alone = F8.from_sessions({"X": Sess({("a", i): led(four(names=A)) for i in range(1, 6)})}, True); assert alone["X"]["nm"]["a"].get("nm_pred", 0) == 0 and alone["X"]["full"]["a"]["pri:ACTION1:has"] == 0


def test_with_one_arm_the_pooled_prior_is_the_per_arm_prior_and_the_variants_without_a_retire_rule_are_the_sevenths():
    s = {"X": _arms()["X"]}; a = F8.from_sessions(s, True); b = F8.from_sessions(s, False); assert a == b
    w7 = F7.from_session(s["X"])
    for v in F7.VARIANTS:
        for g in w7[v]: assert same_but_new(a["X"][v][g], w7[v][g]), (v, g)


def test_the_cost_of_a_retire_rule_is_the_right_moves_lost_against_the_wrong_moves_avoided():
    base = F8.rates(dict(predicted=832, cell_hit=690, moves=936)); ret = F8.rates(dict(predicted=832 - 30 - 30, cell_hit=690 - 30, moves=936))
    k = F8.retire_cost(base, ret); assert (k["lost"], k["avoided"], k["ratio"]) == (30, 30, 1.0) and abs(k["gain"] - (660 / 772 - 690 / 832)) < 1e-12
    assert F8.retire_cost(base, base)["ratio"] is None and F8.retire_cost(base, base)["avoided"] == 0 and F8.retire_cost(base, base)["lost"] == 0 and F8.retire_cost(base, base)["gain"] == 0.0
    worse = F8.rates(dict(predicted=832 - 10, cell_hit=690 - 10, moves=936)); assert F8.retire_cost(base, worse)["avoided"] == 0 and F8.retire_cost(base, worse)["ratio"] is None      # nothing avoided: no ratio
    assert F8.retire_cost(F8.rates({}), F8.rates({}))["gain"] is None


# ---------------------------------------------------------------- 4 the clears
def test_the_session_keeps_the_screen_before_every_move_that_cleared_a_level_and_nothing_else():
    s = F8.Session8(); f0, f1 = grid(3, 3), grid(3, 4)
    s.note("g", {"action": "ACTION1"}, f0, f1, 0, 0); s.note("g", {"action": "ACTION1"}, f1, f0, 0, 1); s.note("g", {"action": "RESET"}, f0, f1, 1, 2)
    s.note("g", {"action": "ACTION6", "x": 2, "y": 3}, f0, f1, 1, 2); s.note("g", {"action": "ACTION2"}, f1, f0, 2, 1)
    assert s.clears == [dict(game="g", level=0, action="ACTION1", before=key(f1)), dict(game="g", level=1, action="ACTION6@2,3", before=key(f0))]              # a reset is not a clear; a drop in level is not a clear
    assert len(s.ledgers[("g", 0)].rows) == 1 and s.moves == {"g": 1} and s.skipped["g"]["level_change"] == 3 and s.skipped["g"]["reset"] == 1                       # the plain session's counts are kept


def _gk(x, y, k=True, w=14, h=14): return tuple("".join("P" if (c, r) == (x, y) else ("k" if k and (c, r) == (1, 1) else ".") for c in range(w)) for r in range(h))


def test_a_level_shows_against_its_first_screen_a_symbol_with_another_number_of_cells_and_the_strip_is_left_out():
    mv = [("U", _gk(6, 9), _gk(6, 6)), ("U", _gk(6, 6), _gk(6, 3)), ("U", _gk(6, 3), _gk(6, 0, False))]
    f = F8.level_facts(led(mv[:2]), mv[1][2], "U"); assert f["other_change"] is False and f["delta"] == {} and f["law"] is False and f["moves"] == 2 and f["changed"] == 2 and f["strips"] == [] and f["click"] is False
    L = led(mv); f = F8.level_facts(L, mv[2][2], "U"); assert f["other_change"] is True and f["delta"] == {"k": -1, ".": 1} and f["other_symbols"] == 2 and f["law"] is True    # the k was collected; U had 3 tries
    assert F8.level_facts(L, mv[2][2], "D")["law"] is False and F8.level_facts(L, mv[2][2], "U@1,2")["click"] is True and F8.level_facts(L, mv[2][2], "U@1,2")["law"] is False
    shr = [("S", _bar(5, 11 - n), _bar(5, 10 - n)) for n in range(9)]; Ls = led(shr); f = F8.level_facts(Ls, Ls.rows[-1].ka, "S")
    assert f["strips"] == ["#"] and f["other_change"] is False and f["delta"] == {}                                      # a strip turning into background is not a change of the board
    def bar2(n, kk): return tuple("".join("P" if (c, r) == (5, 1) else ("#" if r == 0 and c < n else ("k" if kk and (c, r) == (9, 2) else ".")) for c in range(12)) for r in range(3))
    both = [("S", bar2(11 - n, True), bar2(10 - n, True)) for n in range(8)] + [("S", bar2(3, True), bar2(2, False))]; Lb = led(both); f = F8.level_facts(Lb, Lb.rows[-1].ka, "S")
    assert f["strips"] == ["#"] and f["delta"] == {"k": -1, ".": 1} and f["other_change"] is True                       # the strip and something else: only the something else counts
    e = F8.level_facts(None, None); assert e["moves"] == 0 and e["other_change"] is False and e["law"] is False and F8.level_facts(Ledger(), None)["moves"] == 0


def test_the_study_describes_each_clear_and_each_long_uncleared_level():
    mv = [("U", _gk(6, 9), _gk(6, 6)), ("U", _gk(6, 6), _gk(6, 3)), ("U", _gk(6, 3), _gk(6, 0))]
    s = F8.Session8(); L = led(mv[:2] + [("U", _gk(6, 3), _gk(6, 3))] * 3)
    s.ledgers[("g", 0)] = L; s.clears.append(dict(game="g", level=0, action="U", before=key(_gk(6, 3, False)))); s.ledgers[("g", 1)] = led([("U", _gk(6, 9), _gk(6, 6))] * 2)
    s.ledgers[("g", 2)] = led([("U", _gk(6, 9), _gk(6, 6) if n % 2 == 0 else _gk(6, 9)) for n in range(20)]); s.ledgers[("h", 0)] = led([("U", _gk(6, 9), _gk(6, 9, False))] * 19)
    st = F8.clear_study({"Z": s}); assert len(st["cleared"]) == 1 and len(st["uncleared"]) == 1                          # level 1 (2 moves) and h (19 moves) are too short; level 2 has 20
    c = st["cleared"][0]; assert (c["arm"], c["game"], c["level"], c["action"], c["moves"], c["other_change"], c["law"]) == ("Z", "g", 0, "U", 6, True, False) and c["delta"] == {"k": -1, ".": 1}   # 5 recorded + the clearing move
    u = st["uncleared"][0]; assert (u["arm"], u["game"], u["level"], u["moves"], u["other_change"]) == ("Z", "g", 2, 20, False)
    assert len(F8.clear_study({"Z": s}, min_moves=19)["uncleared"]) == 2 and len(F8.clear_study({"Z": s}, min_moves=2)["uncleared"]) == 3 and F8.clear_study({}) == dict(cleared=[], uncleared=[])


# ---------------------------------------------------------------- 5 the reading lines
GAMES = ("g50t-1", "ka59-1", "ls20-1", "sp80-1")
SPLIT = lambda total: [total // 4 + (1 if i < total % 4 else 0) for i in range(4)]


def _w(hits, preds, **extra):
    out = {g: dict(moves=234, predicted=p, cell_hit=h) for g, p, h in zip(GAMES, SPLIT(preds), SPLIT(hits))}
    for k, tot in extra.items():
        for g, v in zip(GAMES, SPLIT(tot)): out[g][k] = v
    return out


def _pri(n=10, has=10, ok=10, one=None, drop=()):
    """Prior counters of the full version: the same in each of the four games; `one` replaces the counters of g50t."""
    w = _w(603, 729)
    for g in GAMES:
        for a in A: w[g][f"pri:{a}:n"] = n; w[g][f"pri:{a}:has"] = has; w[g][f"pri:{a}:has_ok"] = ok
    if one:
        for k, v in one.items(): w["g50t-1"][k] = v
    for k in drop:
        for g in GAMES: w[g].pop(k, None)
    return w


def _ret(a7, lost, avoided):
    h, p = a7; return _w(h - lost, p - lost - avoided)


def _pooled(**over):
    a7 = (693, 841)
    p = {"full": _pri(), "once": _w(663, 797), "all6": _w(672, 809), "nm": _w(610, 760, nm_pred=40, nm_hit=32, nc_pred=20, nc_hit=15), "best": _w(712, 850), "all7": _w(*a7),
         "all7_r1": _ret(a7, 30, 30), "all7_r2": _ret(a7, 10, 12), "best_r1": _ret((712, 850), 30, 30)}
    p.update(over); return p


def _prior(**over):
    p = {"full": _w(603, 729), "once": _w(663, 797), "all6": _w(672, 809), "all7": _w(690, 832), "best": _w(703, 832)}; p.update(over); return p


def _cl(other, law, n=6): return [dict(other_change=i < other, law=i < law, click=False, arm="K", game="g", level=i, action="ACTION1", moves=10, changed=8, strips=[], delta={}, other_symbols=int(i < other)) for i in range(n)]


def _un(other, n=6): return [dict(other_change=i < other, arm="K", game="g", level=i, moves=30) for i in range(n)]


def _clears(cl=None, un=None): return dict(cleared=cl if cl is not None else _cl(4, 4), uncleared=un if un is not None else _un(1))


def _v(pooled=None, prior=None, clears=None, ok=True):
    return {i: x for i, _, _, x in F8.verdict(pooled if pooled is not None else _pooled(), prior if prior is not None else _prior(), clears if clears is not None else _clears(), ok)}


def test_the_twelve_lines_flip_at_their_boundaries_and_the_headline_needs_lines_4_and_5():
    assert len(F8.G_LINES) == 12 and [t for t, _ in F8.G_LINES] == ["guard", "instrument"] + ["claim"] * 10
    assert {i: x for i, _, _, x in F8.verdict(None, None, None, None)} == {i: None for i in range(1, 13)} and _v(ok=False)[1] is False and all(_v().values())
    # 2 the instrument: five figures per arm prior, three pooled
    assert _v(prior=_prior(full=_w(603, 728)))[2] is False and _v(prior=_prior(all7=_w(691, 832)))[2] is False and _v(prior=_prior(best=_w(703, 833)))[2] is False and _v(prior=_prior(once=_w(662, 797)))[2] is False
    assert _v(pooled=_pooled(full=_pri(), all6=_w(672, 810)))[2] is False and _v(pooled=_pooled(once=_w(664, 797)))[2] is False and _v(pooled=_pooled(all7=_w(600, 800)))[2] is True        # pooled all7 is not held to the seventh's
    # 3 the pooled prior: present, agreeing, in every game
    assert _v(pooled=_pooled(full=_pri(n=10, has=9, ok=9)))[3] is True and _v(pooled=_pooled(full=_pri(n=10, has=8, ok=8)))[3] is False          # 90 percent exactly; 80
    assert _v(pooled=_pooled(full=_pri(n=10, has=10, ok=9)))[3] is True and _v(pooled=_pooled(full=_pri(n=10, has=10, ok=8)))[3] is False        # 9 of 10 agree; 8 of 10
    assert _v(pooled=_pooled(full=_pri(n=1, has=1, ok=1)))[3] is True and _v(pooled=_pooled(full=_pri(n=0, has=0, ok=0)))[3] is False and _v(pooled=_pooled(full=_pri(drop=("pri:ACTION4:n", "pri:ACTION4:has", "pri:ACTION4:has_ok"))))[3] is False   # 4 levels in all; none; an action never seen
    lone = {"pri:ACTION4:n": 1, "pri:ACTION4:has": 0, "pri:ACTION4:has_ok": 0}
    mixed = _pri(n=20, has=20, ok=20, one=lone); assert _v(pooled=_pooled(full=mixed))[3] is False                           # 61 levels, 60 with a prior, but one game has none for ACTION4
    mixed["g50t-1"]["pri:ACTION4:n"] = 0; assert _v(pooled=_pooled(full=mixed))[3] is True                                  # that game never made a step with it: not asked
    # 4 the name
    n = lambda a, b, c, d: _pooled(nm=_w(610, 760, nm_pred=a, nm_hit=b, nc_pred=c, nc_hit=d))
    assert _v(pooled=n(50, 38, 0, 0))[4] is True and _v(pooled=n(50, 37, 0, 0))[4] is False and _v(pooled=n(49, 49, 0, 0))[4] is False and _v(pooled=n(30, 24, 20, 14))[4] is True and _v(pooled=n(0, 0, 0, 0))[4] is False      # 76 percent; 74; 49; both lengths count
    # 5 all7 against the full version
    a = lambda add, right: _pooled(all7=_w(603 + right, 729 + add))
    assert _v(pooled=a(110, 88))[5] is True and _v(pooled=a(110, 87))[5] is False and _v(pooled=a(109, 109))[5] is False and _v(pooled=a(0, 0))[5] is False
    # 6 retire 1: precision up 2 points, at most 60 rights lost
    r = lambda lost, avoided: _pooled(all7_r1=_ret((693, 841), lost, avoided))
    assert _v(pooled=r(60, 80))[6] is True and _v(pooled=r(61, 80))[6] is False and _v(pooled=r(10, 2))[6] is False and _v(pooled=r(0, 0))[6] is False
    # 7 at least 20 avoided, ratio at most 2.0
    assert _v(pooled=r(40, 20))[7] is True and _v(pooled=r(41, 20))[7] is False and _v(pooled=r(30, 19))[7] is False and _v(pooled=r(30, 0))[7] is False
    # 8 retire 2 avoids 10 and has a lower ratio than retire 1
    r2 = lambda lost, avoided: _pooled(all7_r1=_ret((693, 841), 40, 25), all7_r2=_ret((693, 841), lost, avoided))
    assert _v(pooled=r2(15, 10))[8] is True and _v(pooled=r2(16, 10))[8] is False and _v(pooled=r2(14, 9))[8] is False and _v(pooled=r2(5, 0))[8] is False                   # 1.5 against 1.6; 1.6 is not lower
    assert _v(pooled=_pooled(all7_r1=_ret((693, 841), 30, 0), all7_r2=_ret((693, 841), 5, 10)))[8] is False                     # retire 1 avoids nothing: no ratio to be lower than
    # 9 to 12 the clears
    assert _v(clears=_clears(cl=_cl(4, 4, 6)))[9] is True and _v(clears=_clears(cl=_cl(4, 4, 5)))[9] is False and _v(clears=_clears(cl=_cl(4, 4, 7)))[9] is False
    assert _v(clears=_clears(cl=_cl(4, 4)))[10] is True and _v(clears=_clears(cl=_cl(3, 4)))[10] is False
    full6 = _cl(6, 4); assert _v(clears=_clears(cl=full6, un=_un(15, 20)))[11] is True and _v(clears=_clears(cl=full6, un=_un(16, 20)))[11] is False and _v(clears=_clears(cl=full6, un=_un(0, 4)))[11] is False        # 1.00 against 0.75: 25 points; 0.80; fewer than 5 uncleared
    assert _v(clears=_clears(cl=_cl(4, 4)))[12] is True and _v(clears=_clears(cl=_cl(4, 3)))[12] is False
    allyes = [(i, t, x, True) for i, t, x, _ in F8.verdict(None, None, None, None)]
    assert F8.headline(allyes) == F8.HEADLINE and F8.claims_allowed(allyes)
    assert F8.headline([(i, t, x, (False if i == 5 else v)) for i, t, x, v in allyes]) == "MIXED: line 4 yes, line 5 NO"
    assert F8.headline([(i, t, x, (False if i == 4 else v)) for i, t, x, v in allyes]) == "MIXED: line 4 NO, line 5 yes"
    assert F8.headline([(i, t, x, (False if i == 4 else (False if i == 5 else v))) for i, t, x, v in allyes]) == "MIXED: line 4 NO, line 5 NO"
    assert F8.headline([(i, t, x, (False if i in (3, 6, 12) else v)) for i, t, x, v in allyes]) == F8.HEADLINE                    # the other lines do not move the headline
    assert F8.headline([(i, t, x, (False if i == 2 else v)) for i, t, x, v in allyes]).startswith("NOT READ") and F8.headline(F8.verdict(None, None, None, None)).startswith("NOT READ")
    assert F8.headline([(i, t, x, (False if i == 1 else v)) for i, t, x, v in allyes]).startswith("NOT READ")


def test_the_printout_names_its_four_parts_and_says_what_each_way_did():
    t = F8.text(_pooled(), _prior(), _clears())
    for n in ("== 1 first tries with the name prior pooled over the four arms", "== 2 the name prior from the other games, pooled over the arms", "== 3 the retire rule", "== 4 the clears: 6 levels cleared", "no rule: 841 predicted, 693 right",
              "retire after 1:", "retire after 2:", "best way, retire after 1:", "right moves lost 30, wrong moves avoided 30", "must cost 1.00 right ones", "(prior per arm, as in the seventh check: all7 832 predicted, 690 right; best 832 predicted, 703 right)",
              "levels of at least 20 recorded moves not cleared: 6; with such a symbol: 1. Cleared with such a symbol: 4 of 6", "arm K g level 0: 10 moves", "already had a law", "no law yet"): assert n in t, n
    assert F8.text(None, None, None) == "" and F8.text({}, {}, None) == "" and "== 4" not in F8.text(_pooled(), None, None) and "== 3" in F8.text(_pooled(), None, None)
    assert "== 1" not in F8.text(None, None, _clears()) and "== 4" in F8.text(None, None, _clears())


# ---------------------------------------------------------------- 6 page, notebook, runner, replay of the stand-in game, front pages
def _head():
    text = open(PAGE, encoding="utf-8").read(); assert text.count(RESULT_MARK) <= 1
    return text[:text.index(RESULT_MARK)] if RESULT_MARK in text else text


def test_the_page_above_its_result_is_what_was_written_before_the_run():
    assert hashlib.sha256(_head().encode()).hexdigest() == PAGE_SHA256


def test_the_page_and_the_module_agree_and_it_says_which_text_is_whose():
    head = _head(); pred = head.split("## Predictions")[1].split("## Reading rules")[0]; rules = head.split("## Reading rules")[1].split("Headline:")[0]
    ps = re.findall(r"^(\d+)\. \[(\w+)\] (.*) \((\d+) %\)$", pred, re.M); rs = re.findall(r"^(\d+)\. (.*)$", rules, re.M)
    assert len(ps) == len(rs) == len(F8.G_LINES) == 12
    for (n, tag, text, pr), (m, rt), (gt, gtext) in zip(ps, rs, F8.G_LINES): assert n == m and text == rt == gtext and tag == gt and 0 < int(pr) < 100
    assert [int(p[3]) for p in ps] == [95, 85, 70, 45, 45, 55, 60, 35, 85, 50, 35, 60]
    for needle in (F8.HEADLINE, "Sir to confirm", "also run by Sir", "Concept-map item 143", "603 right of 729", "40 and right in the other 32", "60.9 %", "85.9 %", "Six levels were cleared", "will not show", "paid test", "$3.65", "Result", "break-even ratio",
                   "Nine ways", "all four arms", "not changed", "left out of both screens", "No cost is assumed", "not written yet" if False else "Nothing on this page has been run on the real games", "foresee8.py", "axi_foresee_check_8.ipynb", "run_foresee_check_8.py"): assert needle in head, needle
    assert "proven" not in head.lower() and "a flaw of the method, not a finding about the games" in head and "adds no line to it" in head


def test_the_notebook_and_the_runner_read_the_saved_files_call_no_model_hold_no_key_and_replay_the_four_arms():
    nb = json.load(open(NB)); src = "\n".join("".join(c["source"]) for c in nb["cells"]); code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert len(nb["cells"]) == 14 and all(c["outputs"] == [] for c in nb["cells"] if c["cell_type"] == "code")
    run = open(RUN, encoding="utf-8").read()
    for text in (src, run):
        assert not re.search(r"AIza[0-9A-Za-z_-]{20,}", text) and "GOOGLE_API_KEY" not in text and "userdata" not in text and "drive.mount" not in text and "open_store" not in text
        assert "foresee7" not in text and "foresee6" not in text and "generate" not in text and ".run(" not in text.replace("LR.run_file(", "")
    for text in (code, run):
        for needle in ("foresee8 as F", "F.feed8", "LR.load", "F.from_sessions(sessions, True)", "F.from_sessions(sessions, False)", "F.pool", "F.clear_study", "F.verdict(pooled, per_arm_prior, clears, allok if ok else None)", "F.headline", "F.text(pooled, per_arm_prior, clears)",
                       "axi_gemini_arc_c_default.json", "axi_gemini_arc_b_default.json", "axi_gemini_arc_default.json", "foresee_check_8_all.json"): assert needle in text, needle
    assert "foresee8.py" in code and "foresee_check_8_printout.txt" in run and "FORESEE_CHECK_8.md" in src and "12 reading lines" in src and "does not show" in src
    for c in nb["cells"]:
        if c["cell_type"] == "code": compile("\n".join(l for l in "".join(c["source"]).split("\n") if not l.lstrip().startswith("!")), "cell", "exec")
    compile(run, "run_foresee_check_8.py", "exec")


def test_the_eighth_check_runs_over_a_replay_of_the_stand_in_game_and_finds_its_clear():
    games = ("alpha-1", "beta-2"); idn = lambda n: n; rep = {}
    for name, llm in (("walker", T.Walker()), ("fake", T.FakeLLM())):
        res = C.run(llm, T.make_env, os.path.join(os.environ.get("TMPDIR", "/tmp"), f"foresee8_{name}_test.jsonl"), games=games, moves=30, action_of=idn)
        s = F8.feed8(res["rows"], T.make_env, "C", games=games, moves=30, action_of=idn); plain = LR.feed(res["rows"], T.make_env, "C", games=games, moves=30, action_of=idn); rep[name] = s
        assert isinstance(s, F8.Session8) and s.moves == plain.moves and {k: len(L.rows) for k, L in s.ledgers.items()} == {k: len(L.rows) for k, L in plain.ledgers.items()}      # the same replay, the same ledgers
    w = rep["walker"]; assert [(c["game"], c["level"], c["action"]) for c in w.clears] == [("alpha-1", 0, "ACTION4")] and rep["fake"].clears[0]["game"] == "beta-2"
    st = F8.clear_study({"C": w}); assert len(st["cleared"]) == 1 and st["cleared"][0]["moves"] == 3 and st["cleared"][0]["law"] is False and len(st["uncleared"]) == 2
    sessions = {"C": w}; pooled = F8.pool(F8.from_sessions(sessions, True)); prior = F8.pool(F8.from_sessions(sessions, False))
    assert set(pooled) == set(F8.VARIANTS) and all(set(pg) == set(games) for pg in pooled.values()) and pooled == prior                                    # one arm: pooled is per arm
    assert all(sum(c["moves"] for c in pg.values()) == sum(w.moves.values()) for pg in pooled.values()) and all(pooled["all7"][g]["predicted"] >= pooled["full"][g]["predicted"] for g in games)
    assert all(pooled["all7_r1"][g]["predicted"] <= pooled["all7"][g]["predicted"] and pooled["all7_r2"][g]["predicted"] >= pooled["all7_r1"][g]["predicted"] for g in games)
    checks = F8.verdict(pooled, prior, st, True); assert len(checks) == 12 and checks[0][3] is True and checks[1][3] is False and F8.headline(checks).startswith("NOT READ")      # the stand-in has not the seventh check's figures
    t = F8.text(pooled, prior, st); assert "== 1" in t and "== 4 the clears: 1 levels cleared" in t and "arm C alpha-1 level 0: 3 moves" in t


def test_the_front_pages_say_what_was_built_and_that_it_was_not_run_on_the_real_games():
    cm = open(os.path.join(DOCS, "RAJNISH_CONCEPT_MAP.md"), encoding="utf-8").read(); vf = open(os.path.join(DOCS, "VERIFIED.md"), encoding="utf-8").read(); hl = open(os.path.join(DOCS, "HISTORY_LEDGER.md"), encoding="utf-8").read()
    assert NEXT in cm and cm.count("Next free:") == 1 and "\n142. **" in cm and "\n143. **" in cm and "\n146. **" not in cm
    i = cm.index("\n143. **") + 1; item = cm[i:cm.index("\n", i)]
    for n in ("FORESEE_CHECK_8.md", "foresee8.py", "test 168", "also run by Sir", "Sir to confirm", "Not run on the real games", "12 reading lines", "retire", "six levels", "break-even", PAGE_SHA256): assert n in item, n
    assert any("FORESEE_CHECK_8.md" in l and "foresee8.py" in l and PAGE_SHA256 in l and "NOT RUN" in l for l in vf.split("\n"))
    assert "| 190 (S) |" in hl and "Sir" in hl and "not run on the real games" in hl and "retire" in hl


if __name__ == "__main__":
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]; bad = 0
    for n, f in fns:
        try: f(); print("ok  ", n)
        except Exception: bad += 1; print("FAIL", n); traceback.print_exc()
    print(f"{len(fns) - bad}/{len(fns)} passed"); sys.exit(1 if bad else 0)
