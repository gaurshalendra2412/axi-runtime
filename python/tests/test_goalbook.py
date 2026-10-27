"""Test 169 (item 144): the three readers on the ledger (axi/experiments/goalbook.py, colab/run_goalbook_show.py): what each action did (mover, changer, idle), what is new on a level's first screen, what to carry from the earlier levels as a suggestion, and the candidate goals from the clears already seen (row 177's list).

They read a `Ledger` and change nothing in it. This test checks, by hand and on small worlds built for it:
  1. the kinds of action: a mover, a changer (something else on the board changed, nothing moved), idle (only the clock ticked), mixed, once; a click is one action; the ticking strip is left out;
  2. the novelty of a level's first screen (colours and shapes not seen on an earlier level's), none on the first level;
  3. the carry-over: the direction (firm when the same in every earlier level with a law), the step as a suggestion that is open, kept or cancelled at the first move of that colour with another step; clicks and laws under MIN_LAW are not carried;
  4. the change a clear made (the strip and the background left out) and the candidate goals from it;
  5. the read-out (at most MAX_LINES, plain statements), the printout over a replayed session, the runner, the front pages.
What it does NOT show: that a model plays better with any of it (a paid test), or that a clear repeats its change in the next level."""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.ledger import Ledger, key
from axi.experiments import foresee8 as F8
from axi.experiments import goalbook as GB
from axi.experiments import gemini_arc_c as C
from tests import test_gemini_arc as T
from tests.test_foresee6 import grid, led, row
from tests.test_foresee7 import _bar
from tests.test_foresee8 import _gk

HERE = os.path.dirname(os.path.abspath(__file__)); DOCS = os.path.join(HERE, "..", "..", "docs"); RUN = os.path.join(HERE, "..", "colab", "run_goalbook_show.py")
NEXT = "Next free: row 192, item 145, test 170, question 122"


def _col(y, step_cols=None, z=False, w=14, h=14):
    return tuple("".join("P" if (c, r) == (6, y) else ("z" if z and (c, r) == (2, 2) else ".") for c in range(w)) for r in range(h))


def _lvl(step, n=3, z=False, start=12): ys = [start - step * i for i in range(n + 1)]; return led([("U", _col(a, z=z), _col(b, z=z)) for a, b in zip(ys, ys[1:])])


def _paint(cells):
    return tuple("".join("P" if (c, r) == (6, 6) else ("x" if (c, r) in cells else ".") for c in range(14)) for r in range(14))


def _snapshot(L): return (len(L.rows), L.dominant(), L.first)


# ---------------------------------------------------------------- 1 the kinds of action
def test_an_action_is_a_mover_a_changer_idle_mixed_or_seen_once_and_a_click_is_one_action():
    mv = [("U", _gk(6, 9), _gk(6, 6)), ("U", _gk(6, 6), _gk(6, 3)), ("U", _gk(6, 3), _gk(6, 0, False)), ("A", _gk(6, 0, False), _gk(6, 0, False)), ("A", _gk(6, 0, False), _gk(6, 0, False)), ("B", _gk(6, 0, False), _gk(6, 0, False))]
    k = GB.action_kinds(led(mv))
    assert (k["U"]["kind"], k["U"]["tries"], k["U"]["moved"], k["U"]["with_change"]) == ("mover", 3, 3, 1)                       # the third move also took the k away
    assert (k["A"]["kind"], k["A"]["null"]) == ("idle", 2) and (k["B"]["kind"], k["B"]["tries"]) == ("once", 1)
    ch = [("ACTION6@1,1", _paint(set()), _paint({(1, 1)})), ("ACTION6@2,1", _paint({(1, 1)}), _paint({(1, 1), (2, 1)})), ("ACTION6@3,1", _paint({(1, 1), (2, 1)}), _paint({(1, 1), (2, 1), (3, 1)}))]
    c = GB.action_kinds(led(ch)); assert list(c) == ["ACTION6"] and c["ACTION6"]["kind"] == "changer" and (c["ACTION6"]["tries"], c["ACTION6"]["changed"], c["ACTION6"]["moved"]) == (3, 3, 0) and sum(c["ACTION6"]["what"].values()) == 3
    mixed = [("M", _paint(set()), _paint({(1, 1)})), ("M", _paint({(1, 1)}), _paint({(1, 1)})), ("M", _paint({(1, 1)}), _paint({(1, 1)})), ("M", _paint({(1, 1)}), _paint({(1, 1)}))]
    assert GB.action_kinds(led(mixed))["M"]["kind"] == "mixed" and GB.action_kinds(Ledger()) == {}                                  # 1 change in 4 tries, 3 nothing


def test_the_ticking_strip_does_not_make_an_idle_action_a_changer_and_does_not_hide_a_mover():
    shr = [("S", _bar(5, 11 - n), _bar(5, 10 - n)) for n in range(9)]; L = led(shr); assert L.ticks() and GB.action_kinds(L)["S"]["kind"] == "idle"          # only the clock moved
    def bar(x, n, w=12): return tuple("".join("P" if (c, r) == (x, 1) else ("#" if r == 0 and c < n else ".") for c in range(w)) for r in range(3))
    run = [("R", bar(i, 11 - i), bar(i + 1, 10 - i)) for i in range(1, 9)]; k = GB.action_kinds(led(run))["R"]; assert k["kind"] == "mover" and k["moved"] == 8 and k["with_change"] == 0


# ---------------------------------------------------------------- 2 what is new on a level
def test_a_level_shows_the_colours_and_shapes_that_no_earlier_level_started_with():
    e1, e2 = _lvl(3), _lvl(3); here = _lvl(4, n=2, z=True)
    assert GB.novelty([e1, e2], here) == dict(first_level=False, symbols=["z"], shapes=0) and GB.novelty([], here) == dict(first_level=True, symbols=[], shapes=0)
    assert GB.novelty([e1], _lvl(3)) == dict(first_level=False, symbols=[], shapes=0) and GB.novelty([e1, here], here)["symbols"] == []                    # once seen, not new
    def blob(): return tuple("".join("P" if (c, r) == (6, 12) else ("P" if (c, r) in ((3, 3), (4, 3)) else ".") for c in range(14)) for r in range(14))
    new_shape = led([("U", blob(), blob())]); assert GB.novelty([e1], new_shape) == dict(first_level=False, symbols=[], shapes=1)                          # the same colour in a shape of two cells
    assert GB.novelty([e1], Ledger()) == dict(first_level=False, symbols=[], shapes=0)


# ---------------------------------------------------------------- 3 what to carry
def test_the_direction_is_firm_when_every_earlier_level_agrees_and_the_step_is_a_suggestion_that_the_first_other_step_cancels():
    e1, e2 = _lvl(3), _lvl(3)
    never = GB.carry_over([e1, e2], _lvl(3, n=0)); assert never == {} or never["U"]["status"] == "open"
    open_ = GB.carry_over([e1, e2], Ledger()); assert open_["U"] == dict(direction="up", levels=2, firm=True, step=(0, -3), symbol="P", status="open", direction_changed=False)
    kept = GB.carry_over([e1, e2], _lvl(3, n=2)); assert kept["U"]["status"] == "kept" and kept["U"]["firm"] is True
    cancelled = GB.carry_over([e1, e2], _lvl(4, n=2)); assert cancelled["U"]["status"] == "cancelled" and cancelled["U"]["step"] == (0, -3) and cancelled["U"]["direction_changed"] is False
    one_other = led([("U", _col(12), _col(9)), ("U", _col(9), _col(5))]); assert GB.carry_over([e1], one_other)["U"]["status"] == "cancelled"               # one move with another step is enough
    down = led([("U", _col(2), _col(5)), ("U", _col(5), _col(8))]); d = GB.carry_over([e1, e2], down)["U"]; assert d["direction_changed"] is True and d["direction"] == "up"          # the action went the other way here
    flipped = led([("U", _col(12 - 3 * i), _col(9 - 3 * i)) for i in range(3)]); updown = led([("U", _col(2 + 3 * i), _col(5 + 3 * i)) for i in range(3)])
    v = GB.carry_over([flipped, updown, flipped], Ledger())["U"]; assert v["direction"] == "up" and v["firm"] is False and v["levels"] == 3 and v["step"] == (0, -3)      # 2 of 3 levels up: not firm
    two = led([("U", _col(12), _col(9)), ("U", _col(9), _col(6))]); assert GB.carry_over([two], Ledger()) == {}                                      # a law needs MIN_LAW moves
    click = led([("ACTION6@1,1", _col(12), _col(9))] * 3); assert GB.carry_over([click], Ledger()) == {}                                          # a click is not carried
    other = led([("U", _col(12), _col(9)), ("U", _col(9), _col(6)), ("U", _col(6), _col(3))]); z = tuple(r.replace("P", "q") for r in _col(12)); zz = tuple(r.replace("P", "q") for r in _col(9))
    assert GB.carry_over([other], led([("U", z, zz)]))["U"]["status"] == "open"                                                                        # another colour moved: not a counterexample of the suggestion


# ---------------------------------------------------------------- 4 the clear and the candidate goals
def test_the_change_a_clear_made_leaves_out_the_strip_and_the_background_and_gives_the_guesses():
    mv = [("U", _gk(6, 9), _gk(6, 6)), ("U", _gk(6, 6), _gk(6, 3)), ("U", _gk(6, 3), _gk(6, 3))]; L = led(mv); last = _gk(6, 0, False)
    d = GB.clear_delta(L, last); assert d == {"k": -1} and GB.clear_delta(Ledger(), last) == {} and GB.clear_delta(None, last) == {}                       # the k was collected; the background gained a cell and is left out
    shr = [("S", _bar(5, 11 - n), _bar(5, 10 - n)) for n in range(9)]; Ls = led(shr); assert GB.clear_delta(Ls, Ls.rows[-1].ka) == {}                     # the strip alone is no change
    L1 = led([("U", _gk(6, 9), _gk(6, 6))]); L0 = led([("U", _col(9), _col(6))])
    assert GB.hypotheses([], L1) == [] and GB.hypotheses([d], L0) == [] and GB.hypotheses([{"v": -2}], L1) == []                                         # no clear seen; the colour is not on this level; a colour that is not here
    assert GB.hypotheses([d], L1) == ["repeat the change of the last clear: colour k -1 cells"]
    two = GB.hypotheses([d, {"k": -2}], L1); assert two[0] == "repeat the change of the last clear: colour k -2 cells" and "empty or collect colour k" in two[1] and "each of the 2 earlier clears" in two[1]
    assert len(GB.hypotheses([d, {"k": 1}], L1)) == 1 and len(GB.hypotheses([d] * 4, L1)) <= 3                                                       # it went down in only one of the two: no second guess
    grow = GB.hypotheses([{"k": 3}], L1); assert grow == ["repeat the change of the last clear: colour k +3 cells"]


# ---------------------------------------------------------------- 5 the read-out, the printout, the runner, the front pages
def test_the_read_out_says_only_what_happened_what_is_new_what_is_carried_and_what_is_guessed_in_at_most_eight_lines():
    e1, e2 = _lvl(3), _lvl(3); here = _lvl(4, n=2, z=True); snap = _snapshot(here); snaps = [_snapshot(e1), _snapshot(e2)]; clears = [{"z": -1}]
    lines = GB.read_out([e1, e2], clears, here)
    assert lines == ["- on this level's first screen and not on an earlier level's: colour z.", "- earlier levels' step, a suggestion and not a rule (this level's first move of it made another step, so that step is dropped): U up by 3.",
                     "- a guess at what clears a level, not tested: repeat the change of the last clear: colour z -1 cells."]
    assert _snapshot(here) == snap and [_snapshot(e1), _snapshot(e2)] == snaps and clears == [{"z": -1}]                                                  # nothing is changed
    assert GB.read_out([], [], _lvl(3)) == [] and GB.read_out([], [], Ledger()) == []                                                                   # a first level with only a mover: nothing to say
    ch = [("ACTION6@1,1", _paint(set()), _paint({(1, 1)})), ("ACTION6@2,1", _paint({(1, 1)}), _paint({(1, 1), (2, 1)})), ("ACTION6@3,1", _paint({(1, 1), (2, 1)}), _paint({(1, 1), (2, 1), (3, 1)}))]
    ro = GB.read_out([], [], led(ch)); assert ro[0].startswith("- ACTION6 did not move anything but changed the board in 3 of its 3 tries here (") and ("something new appeared" in ro[0] and "something grew or shrank" in ro[0])
    four = led([("ACTION%d" % i, _paint({(j, 1) for j in range(n)}), _paint({(j, 1) for j in range(n + 1)})) for i in range(1, 5) for n in (0, 1)])                      # four actions that each only change the board
    assert len(GB.read_out([e1] * 3, [{"z": -1}] * 3, four)) <= GB.MAX_LINES == 8 and len(GB.read_out([e1] * 3, [{"z": -1}] * 3, _lvl(4, n=2, z=True), max_lines=2)) == 2


def test_the_printout_over_a_replayed_session_gives_every_level_its_earlier_levels_and_clears():
    s = F8.Session8(); mv = [("U", _gk(6, 12), _gk(6, 9)), ("U", _gk(6, 9), _gk(6, 6)), ("U", _gk(6, 6), _gk(6, 3))]
    s.ledgers[("g", 0)] = led(mv); s.clears.append(dict(game="g", level=0, action="U", before=key(_gk(6, 0, False))))
    s.ledgers[("g", 1)] = led([("U", _gk(6, 9), _gk(6, 5))]); t = GB.show({"Z": s})
    assert "== arm Z g level 0: 3 moves recorded; cleared by U, the change: k -1" in t and "== arm Z g level 1: 1 moves recorded" in t and "(nothing to say)" in t
    assert "a guess at what clears a level, not tested: repeat the change of the last clear: colour k -1 cells" in t and "U up by 3" in t and GB.show({}) == ""
    w = C.run(T.Walker(), T.make_env, os.path.join(os.environ.get("TMPDIR", "/tmp"), "goalbook_walker_test.jsonl"), games=("alpha-1", "beta-2"), moves=30, action_of=lambda n: n)
    s2 = F8.feed8(w["rows"], T.make_env, "C", games=("alpha-1", "beta-2"), moves=30, action_of=lambda n: n); t2 = GB.show({"C": s2})
    assert "== arm C alpha-1 level 0: 2 moves recorded; cleared by ACTION4" in t2 and "== arm C alpha-1 level 1:" in t2 and "== arm C beta-2 level 0: 30 moves recorded" in t2


def test_the_runner_replays_the_four_arms_calls_no_model_holds_no_key_and_prints_without_a_verdict():
    run = open(RUN, encoding="utf-8").read()
    assert not re.search(r"AIza[0-9A-Za-z_-]{20,}", run) and "GOOGLE_API_KEY" not in run and "userdata" not in run and "generate" not in run and "verdict" not in run.replace("not a verdict", "")
    for needle in ("goalbook as GB", "F.feed8", "LR.load", "GB.show(sessions)", "axi_gemini_arc_c_default.json", "axi_gemini_arc_b_default.json", "axi_gemini_arc_default.json", "goalbook_show_printout.txt"): assert needle in run, needle
    compile(run, "run_goalbook_show.py", "exec")


def test_the_readers_are_pure_and_the_ledger_is_not_changed():
    L = _lvl(3); snap = _snapshot(L); before = json.dumps([r.ka for r in L.rows])
    GB.action_kinds(L); GB.novelty([_lvl(3)], L); GB.carry_over([_lvl(3)], L); GB.clear_delta(L, L.rows[-1].ka); GB.hypotheses([{"P": -1}], L); GB.read_out([_lvl(3)], [{"P": -1}], L)
    assert _snapshot(L) == snap and json.dumps([r.ka for r in L.rows]) == before and GB.KINDS == ("mover", "changer", "idle", "mixed", "once")
    src = open(os.path.join(HERE, "..", "axi", "experiments", "goalbook.py"), encoding="utf-8").read(); assert "foresee7" not in src and "L.rows.append" not in src and ".record(" not in src


def test_the_front_pages_say_what_was_built_and_that_it_was_not_run_on_the_real_games():
    cm = open(os.path.join(DOCS, "RAJNISH_CONCEPT_MAP.md"), encoding="utf-8").read(); vf = open(os.path.join(DOCS, "VERIFIED.md"), encoding="utf-8").read(); hl = open(os.path.join(DOCS, "HISTORY_LEDGER.md"), encoding="utf-8").read()
    assert NEXT in cm and cm.count("Next free:") == 1 and "\n143. **" in cm and "\n144. **" in cm and "\n145. **" not in cm
    i = cm.index("\n144. **") + 1; item = cm[i:cm.index("\n", i)]
    for n in ("goalbook.py", "run_goalbook_show.py", "test 169", "Sir to confirm", "Not run on the real games", "row 177", "action_kinds", "carry_over", "paid test"): assert n in item, n
    assert any("goalbook.py" in l and "NOT RUN" in l for l in vf.split("\n"))
    assert "| 191 (T) |" in hl and "Sir" in hl and "not run on the real games" in hl and "row 177" in hl


if __name__ == "__main__":
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]; bad = 0
    for n, f in fns:
        try: f(); print("ok  ", n)
        except Exception: bad += 1; print("FAIL", n); traceback.print_exc()
    print(f"{len(fns) - bad}/{len(fns)} passed"); sys.exit(1 if bad else 0)
