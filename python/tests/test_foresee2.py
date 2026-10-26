"""Test 162 (item 137): the second forward check (docs/FORESEE_CHECK_2.md, axi/experiments/foresee2.py, colab/axi_foresee_check_2.ipynb).

The first check (test 161) could not look at walls in 553 of 729 predictions because it took the mover to be one object of one colour. This one finds the mover as the group that moves together and learns walls by
position from a move that did nothing; the structure it reads is Rajnish Choubey's RFC-001 (a refused change is held state), in game terms, for Sir to confirm. No model and no benchmark game here. It checks:
  1. a sprite of two colours that moves together is one mover (the first check could not use it), and a second piece that does not move with it is not part of it;
  2. a wall is learned by position from a move that did nothing, a cell the piece has stepped onto is never held as blocking, the edge of the screen blocks;
  3. what is floor is what the piece has walked on (not the most common colour of the screen), and the count of "did not move" with something in front of the piece;
  4. on the invented worlds (not benchmark games) the exact counts of the page; a pure function;
  5. the 8 reading lines at their boundaries (including that a prediction that speaks less than 20 times cannot read yes), the page (hash, agreement with the module), the notebook, the replay of arm C into the stand-in game, the front pages.
What it does NOT show: how well it predicts on the real games. That is the Result of the run (Colab)."""
import hashlib, json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.ledger import Ledger
from axi.experiments import foresee2 as F2
from axi.experiments import foresee as F1
from axi.experiments import ledger_replay as LR
from axi.experiments import gemini_arc_c as C
from axi.experiments.mini_worlds import World, walk, PUSH_LEVEL, PAD_LEVEL
from tests import test_gemini_arc as T

HERE = os.path.dirname(os.path.abspath(__file__)); DOCS = os.path.join(HERE, "..", "..", "docs")
PAGE = os.path.join(DOCS, "FORESEE_CHECK_2.md"); NB = os.path.join(HERE, "..", "colab", "axi_foresee_check_2.ipynb")
PAGE_SHA256 = "382d0a7fbff8c961d060b6147c3a6708d3c8523dc87a9b216f0f1d74b554cb5a"; RESULT_MARK = "\n## Result"
NEXT = "Next free: row 187, item 141, test 166, question 122"


def sprite_world(n, wall_col=None, w=10, other=True):
    """A two-colour sprite 'ab' on row 1 of a 3-row screen; ACTION4 moves it one right (it stops in front of '#' at wall_col). 'x' is another piece that never moves."""
    frames = []; x = 1
    for _ in range(n + 1):
        rows = []
        for r in range(3):
            row = ["." for _ in range(w)]
            if wall_col is not None:
                for rr in range(3): pass
                row[wall_col] = "#"
            if other and r == 2: row[w - 1] = "x"
            if r == 1: row[x] = "a"; row[x + 1] = "b"
            rows.append("".join(row))
        frames.append(tuple(rows))
        if not (wall_col is not None and x + 2 == wall_col): x += 1
    return frames


def moves_of(frames, action="ACTION4"): return [(action, frames[i], frames[i + 1]) for i in range(len(frames) - 1)]


# ---------------------------------------------------------------- 1 the mover as a group
def test_a_sprite_of_two_colours_is_one_mover_and_a_piece_that_does_not_move_with_it_is_not():
    fr = sprite_world(8, wall_col=7); c = F2.evaluate(moves_of(fr)); assert c["moves"] == 8
    # the first check needs exactly one object of one colour: here the dominant symbol ('a' or 'b') is one object, so it also works; the difference is in the group
    L = Ledger()
    for i in range(3): L.record("ACTION4", fr[i], fr[i + 1])
    d = L.dominant()["ACTION4"]; g = F2._group(L, "ACTION4", d[0], d[1], d[2]); assert g == {"a", "b"} and "x" not in g
    last = {}
    for r in L.rows:
        for o in r.ops:
            if o[0] == "move": last[o[1]] = (o[2][0] + o[3][0], o[2][1] + o[3][1], o[2][2] + o[3][0], o[2][3] + o[3][1])
    fp = F2._footprint(fr[3], g, last); assert fp == {(4, 1), (5, 1)}                                       # the two cells of the sprite where it was last seen


def test_the_first_check_goes_blind_on_a_mover_that_is_two_objects_of_one_colour_and_the_group_does_not():
    w = 10
    def frame(x): return tuple(("".join("a" if c in (x, x + 3) and r == 1 else ("#" if c == 8 else ".") for c in range(w))) for r in range(3))
    seq = [("ACTION4", frame(i), frame(i + 1)) for i in range(4)]                                              # two cells of colour 'a' at x and x+3 move together; wall at column 8
    c1 = F1.evaluate(seq); assert c1.get("wall_unknown", 0) == c1["predicted"] > 0                               # two objects of the symbol: the first check cannot check walls
    c2 = F2.evaluate(seq); assert c2["known"] == c2["predicted"] > 0                                           # found by where it was last seen to move


# ---------------------------------------------------------------- 2 walls by position
def test_a_wall_is_learned_by_position_from_a_move_that_did_nothing_and_the_edge_blocks():
    # P walks right on a row, is stopped by a wall 'W' it has never stepped onto, tries again: the second time "does not move" is said
    w = 8
    def fr(x): return tuple("".join("P" if (c, r) == (x, 1) else ("W" if c == 6 else ".") for c in range(w)) for r in range(3))
    seq = [("R", fr(1), fr(2)), ("R", fr(2), fr(3)), ("R", fr(3), fr(4)), ("R", fr(4), fr(5)), ("R", fr(5), fr(5)), ("R", fr(5), fr(5)), ("R", fr(5), fr(5))]
    c = F2.evaluate(seq)
    assert c["predicted"] == 5 and c["step_hit"] == 2 and c["cell_said"] == 2 and c["cell_right"] == 2 and c["cell_hit"] == 4               # the first stop is a miss; after it the cell is held and "does not move" is right twice
    assert c["dnm_object"] == 3 and c.get("dnm_floor", 0) == 0                                                  # 'W' is a colour the piece never walked on
    edge = tuple("".join("P" if (c, r) == (3, 1) else "." for c in range(4)) for r in range(3)); e0 = tuple("".join("P" if (c, r) == (1, 1) else "." for c in range(4)) for r in range(3)); e2 = tuple("".join("P" if (c, r) == (2, 1) else "." for c in range(4)) for r in range(3))
    ce = F2.evaluate([("R", e0, e2), ("R", e2, edge), ("R", edge, edge)]); assert ce["predicted"] == 1 and ce["cell_said"] == 1 and ce["cell_right"] == 1   # at the last column the next step is off the screen


def test_a_cell_the_piece_has_stepped_onto_is_never_held_as_blocking_a_door_that_opens():
    w = 8
    def fr(x, door): return tuple("".join("P" if (c, r) == (x, 1) else (("D" if door else ".") if c == 5 else ".") for c in range(w)) for r in range(3))
    seq = [("R", fr(1, 1), fr(2, 1)), ("R", fr(2, 1), fr(3, 1)), ("R", fr(3, 1), fr(4, 1)), ("R", fr(4, 1), fr(4, 1)),            # stopped in front of the closed door: held
           ("R", fr(4, 1), fr(4, 1)),                                                                                      # said "does not move": right
           ("R", fr(4, 0), fr(5, 0))]                                                                                      # the door has opened and the piece steps onto the cell: it was not blocking
    c = F2.evaluate(seq); assert c["cell_said"] == 2 and c["cell_right"] == 1                                       # the open door: said "does not move", it moved: wrong once, and the cell is released after it
    again = F2.evaluate(seq + [("R", fr(5, 0), fr(6, 0))]); assert again["cell_said"] == 2                                # (no new statement: the cell was released)


# ---------------------------------------------------------------- 3 floor is what has been walked on
def test_floor_is_the_colour_the_piece_has_walked_on_not_the_screens_most_common_colour():
    # a screen in which '#' is the most common colour (as in the invented push world): the walls must still count as "something in front"
    c = F2.combine([F2.evaluate(walk(World(PUSH_LEVEL, pads=False), s, 300)) for s in range(10)])
    assert (c["moves"], c["predicted"], c["step_hit"], c["cell_hit"], c["cell_said"], c["cell_right"]) == (3000, 2294, 1670, 1966, 300, 298)
    assert (c["did_not_move"], c["dnm_object"], c["dnm_floor"], c["dnm_unknown"]) == (624, 371, 246, 7) and c["dnm_object"] + c["dnm_floor"] + c["dnm_unknown"] == c["did_not_move"]


def test_on_the_pad_world_the_counts_are_exact_and_the_rates_are_those_of_the_page():
    p = F2.combine([F2.evaluate(walk(World(PAD_LEVEL), s, 300)) for s in range(10)])
    assert (p["moves"], p["predicted"], p["step_hit"], p["cell_hit"], p["cell_said"], p["cell_right"], p["dnm_object"], p["dnm_floor"], p["dnm_unknown"]) == (3000, 2295, 1851, 2087, 252, 244, 382, 55, 7)
    r = F2.rates(p); assert abs(r["step_only"] - 0.807) < 0.001 and abs(r["cell_learned"] - 0.909) < 0.001 and r["known_share"] == 1.0
    assert abs(F2.rates(F2.combine([F2.evaluate(walk(World(PUSH_LEVEL, pads=False), s, 300)) for s in range(10)]))["cell_learned"] - 0.857) < 0.001


def test_the_check_is_a_pure_function_and_changes_nothing_it_is_given():
    moves = walk(World(PAD_LEVEL), 3, 120); snap = json.dumps(moves); a = F2.evaluate(moves); b = F2.evaluate(list(moves)); assert a == b and json.dumps(moves) == snap
    assert F2.text({}) == "" and F2.rates({})["known_share"] is None and F2.rates({})["object_share"] is None


# ---------------------------------------------------------------- 5 the reading lines
def _pg(moves=200, predicted=150, known=120, step=100, cell=120, said=30, right=28, obj=10, floor=5):
    return dict(moves=moves, predicted=predicted, known=known, step_hit=step, cell_hit=cell, cell_said=said, cell_right=right, dnm_object=obj, dnm_floor=floor, did_not_move=obj + floor)


def _v(games, ok=True): return {i: v for i, _, _, v in F2.verdict(games, ok)}


def test_the_eight_lines_flip_at_their_boundaries_and_the_headline_needs_lines_3_and_5():
    assert len(F2.G_LINES) == 8 and [t for t, _ in F2.G_LINES] == ["guard", "instrument"] + ["claim"] * 6
    four = lambda **kw: {f"g{i}": _pg(**kw) for i in range(4)}
    assert _v(None, None) == {i: None for i in range(1, 9)} and _v(four(), False)[1] is False
    g = four(); g["g0"] = _pg(known=89); assert _v(g)[2] is True                                       # 60 percent of 150 = 90: the mover found in 3 of the 4 games is enough
    g["g1"] = _pg(known=89); assert _v(g)[2] is False
    assert _v(four())[3] is True; g = four(); g["g0"] = _pg(cell=100); g["g1"] = _pg(cell=100); assert _v(g)[3] is False        # better than step only in 2 games only
    assert _v(four(said=4, right=4))[4] is False                                                                 # 16 statements: not enough, however right
    assert _v(four(said=5, right=5))[4] is True and _v(four(said=5, right=4))[4] is False                       # 20 statements, 85 percent: 17 of 20 right
    assert _v(four(said=10, right=8))[4] is False and _v(four(said=20, right=17))[4] is True and _v(four(said=20, right=16))[4] is False
    assert _v(four(predicted=100, cell=80))[5] is True and _v(four(predicted=100, cell=79))[5] is False          # 80 percent
    assert _v(four(predicted=100, cell=90))[6] is True and _v(four(predicted=100, cell=89))[6] is False          # 90 percent
    assert _v(four(obj=5, floor=5))[7] is True and _v(four(obj=4, floor=5))[7] is False                          # half
    assert _v(four(step=100, cell=98))[8] is True and _v(four(step=100, cell=96))[8] is False                   # 2 points of 150: 3 moves
    good = four(predicted=100, step=70, cell=95, known=100); ch = F2.verdict(good, True); assert F2.headline(ch) == F2.HEADLINE and F2.claims_allowed(ch)
    ch2 = F2.verdict(four(predicted=100, step=70, cell=75, known=100), True); assert F2.headline(ch2).startswith("MIXED: line 3 yes, line 5 NO")
    assert F2.headline(F2.verdict(None, None)).startswith("NOT READ") and F2.headline(F2.verdict(good, False)).startswith("NOT READ")


# ---------------------------------------------------------------- 6 page, notebook, replay of arm C, front pages
def _head():
    text = open(PAGE, encoding="utf-8").read(); assert text.count(RESULT_MARK) <= 1
    return text[:text.index(RESULT_MARK)] if RESULT_MARK in text else text


def test_the_page_above_its_result_is_what_was_written_before_the_run():
    assert hashlib.sha256(_head().encode()).hexdigest() == PAGE_SHA256


def test_the_page_and_the_module_agree_and_the_page_does_not_claim_the_pasted_texts_as_sirs():
    head = _head(); pred = head.split("## Predictions")[1].split("## Reading rules")[0]; rules = head.split("## Reading rules")[1].split("Headline:")[0]
    ps = re.findall(r"^(\d+)\. \[(\w+)\] (.*) \((\d+) %\)$", pred, re.M); rs = re.findall(r"^(\d+)\. (.*)$", rules, re.M)
    assert len(ps) == len(rs) == len(F2.G_LINES) == 8
    for (n, tag, text, pr), (m, rt), (gt, gtext) in zip(ps, rs, F2.G_LINES): assert n == m and text == rt == gtext and tag == gt and 0 < int(pr) < 100
    for needle in (F2.HEADLINE, "Sir to confirm", "RFC-001", "553", "183", "not assumed", "Neither text is Sir's", "85.7 %", "90.9 %", "will not show", "replay"): assert needle in head, needle
    assert "$" not in head and "proven" not in head.lower()
    p = F2.combine([F2.evaluate(walk(World(PAD_LEVEL), s, 300)) for s in range(10)]); q = F2.combine([F2.evaluate(walk(World(PUSH_LEVEL, pads=False), s, 300)) for s in range(10)])
    assert "%.1f %%" % (100 * F2.rates(q)["cell_learned"]) in head and "%.1f %%" % (100 * F2.rates(p)["cell_learned"]) in head


def test_the_notebook_reads_the_saved_files_calls_no_model_and_holds_no_key():
    nb = json.load(open(NB)); src = "\n".join("".join(c["source"]) for c in nb["cells"]); code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert len(nb["cells"]) == 12 and all(c["outputs"] == [] for c in nb["cells"] if c["cell_type"] == "code")
    assert not re.search(r"AIza[0-9A-Za-z_-]{20,}", src) and "GOOGLE_API_KEY" not in src and "userdata" not in src and "drive.mount" not in src and "open_store" not in src
    for needle in ("foresee2 as F", "F.from_session", "LR.run_file", "F.verdict", "F.headline", "axi_gemini_arc_c_default.json", "axi_gemini_arc_b_default.json", "axi_gemini_arc_default.json", "foresee_check_2_all.json", "foresee2.py"): assert needle in code, needle
    assert "FORESEE_CHECK_2.md" in src and "generate" not in code and ".run(" not in code.replace("LR.run_file(", "")
    for c in nb["cells"]:
        if c["cell_type"] == "code": compile("\n".join(l for l in "".join(c["source"]).split("\n") if not l.lstrip().startswith("!")), "cell", "exec")


def test_the_second_check_runs_over_a_replay_of_arm_c_on_the_stand_in_game():
    res = C.run(T.FakeLLM(), T.make_env, os.path.join(os.environ.get("TMPDIR", "/tmp"), "foresee2_c_test.jsonl"), games=("alpha-1", "beta-2"), moves=30, action_of=lambda n: n)
    s = LR.feed(res["rows"], T.make_env, "C", games=("alpha-1", "beta-2"), moves=30, action_of=lambda n: n); pg = F2.from_session(s)
    assert set(pg) == {"alpha-1", "beta-2"} and sum(c["moves"] for c in pg.values()) == sum(s.moves.values()) and "ALL GAMES" in F2.text(pg) and "found in" in F2.text(pg)


def test_the_front_pages_say_what_was_built_and_that_it_was_run_once_with_the_result_not_written():
    cm = open(os.path.join(DOCS, "RAJNISH_CONCEPT_MAP.md"), encoding="utf-8").read(); vf = open(os.path.join(DOCS, "VERIFIED.md"), encoding="utf-8").read(); hl = open(os.path.join(DOCS, "HISTORY_LEDGER.md"), encoding="utf-8").read()
    assert NEXT in cm and cm.count("Next free:") == 1 and "\n137. **" in cm and "\n138. **" in cm and "\n139. **" in cm and "\n140. **" in cm and "\n141. **" not in cm
    i = cm.index("\n137. **") + 1; item = cm[i:cm.index("\n", i)]
    for n in ("FORESEE_CHECK_2.md", "foresee2.py", "test 162", "Rajnish Choubey", "Sir to confirm", "Result not written", "8 reading lines", PAGE_SHA256): assert n in item, n
    assert any("FORESEE_CHECK_2.md" in l and "foresee2.py" in l and PAGE_SHA256 in l and "Result is not written" in l for l in vf.split("\n"))
    assert "| 183 (L) |" in hl and "position" in hl


if __name__ == "__main__":
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]; bad = 0
    for n, f in fns:
        try: f(); print("ok  ", n)
        except Exception: bad += 1; print("FAIL", n); traceback.print_exc()
    print(f"{len(fns) - bad}/{len(fns)} passed"); sys.exit(1 if bad else 0)
