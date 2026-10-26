"""Test 163 (item 138): the third forward check (docs/FORESEE_CHECK_3.md, axi/experiments/foresee3.py, colab/axi_foresee_check_3.ipynb).

The second check (test 162) found the mover (97.4 %) but learned almost no walls, and its stops were of two kinds: in ls20 something was in front of the piece, in sp80 and g50t nothing was. This test checks, by hand and on small worlds built for it:
  1. the whole front is held when the piece did not move (a tall piece in front of a two-cell wall: the second check learns nothing, the third learns it), and a held cell is released the moment the piece steps onto it;
  2. an action is typed stationary only by what it does on open floor (an action that stops on open floor is called stationary after 4 tries; a piece that bumps into a wall many times is NOT, which was the fault of a first trial);
  3. with both changes switched off the code is the second check exactly, on the invented worlds, counter for counter;
  4. the 8 reading lines at their boundaries, the page (its hash, its lines, what it says it will not show), the notebook (no key, no model, 4 ways), the front pages.
What it does NOT show: how well it predicts on the real games. That is the Result of the run (Colab)."""
import hashlib, json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
from axi.experiments import foresee2 as F2
from axi.experiments import foresee3 as F3
from axi.experiments import ledger_replay as LR
from axi.experiments import gemini_arc_c as C
from axi.experiments.mini_worlds import World, walk, PUSH_LEVEL, PAD_LEVEL
from tests import test_gemini_arc as T

HERE = os.path.dirname(os.path.abspath(__file__)); DOCS = os.path.join(HERE, "..", "..", "docs")
PAGE = os.path.join(DOCS, "FORESEE_CHECK_3.md"); NB = os.path.join(HERE, "..", "colab", "axi_foresee_check_3.ipynb")
PAGE_SHA256 = "ed6976b51d68394029e82d58ba6c046a0ef8c0084374390d37695854ab69895d"; RESULT_MARK = "\n## Result"
NEXT = "Next free: row 187, item 141, test 166, question 122"
KEYS = ("predicted", "step_hit", "cell_hit", "cell_said", "cell_right", "typed_said", "typed_right", "wall_said", "held_cells", "dnm_object", "dnm_floor", "dnm_floor_stat")
get = lambda c: {k: c.get(k, 0) for k in KEYS}


def tall(x, w=9, wall=6, door=True):
    """A piece two cells tall ('a', rows 1 and 2 of 4) at column x; a wall '#' two cells tall at column `wall` (an open door leaves it floor)."""
    return tuple("".join("a" if (c == x and r in (1, 2)) else ("#" if (c == wall and r in (1, 2) and door) else ".") for c in range(w)) for r in range(4))


def tall_seq(stops=5): return [("R", tall(i), tall(i + 1)) for i in range(1, 5)] + [("R", tall(5), tall(5))] * stops


def row(x, w=12): return tuple("".join("P" if (c, r) == (x, 1) else "." for c in range(w)) for r in range(3))


# ---------------------------------------------------------------- 1 the whole front is held, and released
def test_a_piece_two_cells_tall_in_front_of_a_two_cell_wall_the_second_check_learns_nothing_and_the_third_does():
    off = get(F3.evaluate(tall_seq(), False, False)); front = get(F3.evaluate(tall_seq(), True, False))
    assert off == dict(predicted=7, step_hit=2, cell_hit=2, cell_said=0, cell_right=0, typed_said=0, typed_right=0, wall_said=0, held_cells=0, dnm_object=5, dnm_floor=0, dnm_floor_stat=0)      # two candidate cells: the one-cell rule holds none
    assert front == dict(predicted=7, step_hit=2, cell_hit=6, cell_said=4, cell_right=4, typed_said=0, typed_right=0, wall_said=4, held_cells=2, dnm_object=5, dnm_floor=0, dnm_floor_stat=0)  # the first stop is a miss; the next four are said, and right
    assert F2.evaluate(tall_seq()).get("cell_said", 0) == 0                                                                  # the second check itself says nothing here


def test_a_held_cell_is_released_the_moment_the_piece_steps_onto_it_a_door_that_opens():
    seq = tall_seq(3) + [("R", tall(5, door=False), tall(6, door=False))]                                              # three stops (one miss, two said and right), then the door opens and the piece goes through
    c = get(F3.evaluate(seq, True, False)); assert (c["cell_said"], c["cell_right"], c["step_hit"], c["cell_hit"], c["held_cells"]) == (3, 2, 3, 4, 2)         # said once more at the open door: wrong once
    more = seq + [("R", tall(6, door=False), tall(7, door=False))]; d = get(F3.evaluate(more, True, False))
    assert d["cell_said"] == 3 and d["step_hit"] == 4 and d["cell_hit"] == 5                                         # released: no statement from the held cells any more, and the next step is predicted as a step


def test_the_floor_must_be_known_before_a_front_is_held():
    # the first predicted move is a stop: nothing has been stepped onto yet, so no floor is known and nothing is held (the front could be all floor)
    seq = [("R", tall(1, wall=4), tall(2, wall=4)), ("R", tall(2, wall=4), tall(3, wall=4)), ("R", tall(3, wall=4), tall(3, wall=4)), ("R", tall(3, wall=4), tall(3, wall=4))]
    c = F3.evaluate(seq, True, False); assert c["predicted"] == 2 and c["dnm_unknown"] == 2 and c.get("held_cells", 0) == 0 and c.get("cell_said", 0) == 0


# ---------------------------------------------------------------- 2 the action is typed by what it does on open floor
def test_an_action_that_stops_on_open_floor_is_called_stationary_after_four_tries_and_not_before():
    seq = [("S", row(i), row(i + 1)) for i in range(1, 5)] + [("S", row(5), row(5))] * 8
    typed = get(F3.evaluate(seq, False, True)); off = get(F3.evaluate(seq, False, False))
    assert off["cell_said"] == 0 and off["cell_hit"] == 2 == off["step_hit"] and off["dnm_floor"] == 8 and off["dnm_floor_stat"] == 5
    assert typed == dict(predicted=10, step_hit=2, cell_hit=7, cell_said=5, cell_right=5, typed_said=5, typed_right=5, wall_said=0, held_cells=0, dnm_object=0, dnm_floor=8, dnm_floor_stat=5)   # 3 stops to learn, then 5 statements, all right
    assert F3.evaluate(seq[:7], False, True).get("typed_said", 0) == 0                                                  # up to the third stop on open floor: not enough tries, nothing is called stationary


def test_a_piece_that_keeps_bumping_into_a_wall_is_not_called_stationary_by_the_typing():
    typed = get(F3.evaluate(tall_seq(12), False, True)); off = get(F3.evaluate(tall_seq(12), False, False)); assert typed == off and typed["typed_said"] == 0 and typed["dnm_object"] == 12
    # the same bumps with the front held: said by the held wall, never by the action type
    full = get(F3.evaluate(tall_seq(12), True, True)); assert full["typed_said"] == 0 and full["wall_said"] == 11


# ---------------------------------------------------------------- 3 both off is the second check
def test_with_both_changes_off_the_code_is_the_second_check_counter_for_counter_on_the_invented_worlds():
    for mk in (lambda: World(PUSH_LEVEL, pads=False), lambda: World(PAD_LEVEL)):
        sess = [walk(mk(), s, 300) for s in range(10)]
        a = F2.combine([F2.evaluate(m) for m in sess]); b = F3.combine([F3.evaluate(m, False, False) for m in sess])
        assert a and all(b.get(k, 0) == a[k] for k in a) and b.get("typed_said", 0) == 0 and b.get("held_cells", 0) == 0
        assert {k: v for k, v in b.items() if v and not k.startswith("act_") and k not in ("wall_said", "wall_right")} == {k: v for k, v in a.items() if v}                      # nothing else is counted that the second check does not count
    t = F3.evaluate(tall_seq(), False, False); assert all(t.get(k, 0) == v for k, v in F2.evaluate(tall_seq()).items())


def test_on_the_invented_worlds_the_four_ways_give_the_counts_and_rates_of_the_page():
    def run(mk): 
        sess = [walk(mk(), s, 300) for s in range(10)]; return {v: F3.combine([F3.evaluate(m, f, t) for m in sess]) for v, (f, t) in F3.VARIANTS.items()}
    push = run(lambda: World(PUSH_LEVEL, pads=False)); pad = run(lambda: World(PAD_LEVEL))
    assert [push[v]["cell_hit"] for v in ("off", "front", "typed", "full")] == [1966, 1958, 1963, 1955] and [pad[v]["cell_hit"] for v in ("off", "front", "typed", "full")] == [2087, 2081, 2086, 2080]
    assert push["off"]["predicted"] == push["full"]["predicted"] == 2294 and pad["off"]["predicted"] == pad["full"]["predicted"] == 2295
    f = lambda c: "%.1f %%" % (100 * F3.rates(c)["cell_learned"])
    assert (f(push["full"]), f(pad["full"]), f(push["off"]), f(pad["off"])) == ("85.2 %", "90.6 %", "85.7 %", "90.9 %")


def test_the_check_is_a_pure_function_and_changes_nothing_it_is_given():
    moves = walk(World(PAD_LEVEL), 3, 120); snap = json.dumps(moves); a = F3.evaluate(moves); b = F3.evaluate(list(moves)); assert a == b and json.dumps(moves) == snap
    assert F3.text({}) == "" and F3.rates({})["floor_stationary_share"] is None and F3.by_action({}) == {}
    assert F3.VARIANTS == {"off": (False, False), "front": (True, False), "typed": (False, True), "full": (True, True)}


# ---------------------------------------------------------------- 4 the reading lines
def _pg(predicted=100, hit=80, said=10, right=9, floor=10, stat=6):
    return dict(moves=predicted + 20, predicted=predicted, known=predicted, step_hit=hit - 5, cell_hit=hit, cell_said=said, cell_right=right, dnm_floor=floor, dnm_floor_stat=stat, did_not_move=floor)


GAMES = ("g50t-1", "ka59-1", "ls20-1", "sp80-1")


def _v(full, off=None, ok=True):
    off = off if off is not None else {g: _pg(predicted=180 if i < 3 else 189, hit=130 if i < 3 else 137) for i, g in enumerate(GAMES)}
    return {i: v for i, _, _, v in F3.verdict({"off": off, "full": full}, ok)}


def _four(**kw): return {g: _pg(**kw) for g in GAMES}


def test_the_eight_lines_flip_at_their_boundaries_and_the_headline_needs_lines_3_and_6():
    assert len(F3.G_LINES) == 8 and [t for t, _ in F3.G_LINES] == ["guard", "instrument"] + ["claim"] * 6
    assert {i: v for i, _, _, v in F3.verdict(None, None)} == {i: None for i in range(1, 9)} and _v(_four(), ok=False)[1] is False
    assert _v(_four())[2] is True                                                                                    # off: 3 games of 180 with 130 right and one of 189 with 137: 527 of 729
    bad = {g: _pg(predicted=180 if i < 3 else 189, hit=130 if i < 3 else 137) for i, g in enumerate(GAMES)}; bad["sp80-1"] = _pg(predicted=189, hit=138)
    assert _v(_four(), off=bad)[2] is False                                                                           # 528 of 729 is not the second check's figure: the instrument line fails
    off = {g: _pg(predicted=100, hit=50) for g in GAMES}
    full = _four(predicted=100, hit=51); full["g50t-1"] = _pg(predicted=100, hit=50); assert _v(full, off=off)[3] is True and not F3.claims_allowed(F3.verdict({"off": off, "full": full}, True))   # 3 games better, but line 2 fails (not the second check's total)
    full["ka59-1"] = _pg(predicted=100, hit=50); assert _v(full, off=off)[3] is False                                  # 2 games better only
    ls = _four(predicted=100, hit=80); ls["ls20-1"] = _pg(predicted=100, hit=75); assert _v(ls)[4] is True; ls["ls20-1"] = _pg(predicted=100, hit=74); assert _v(ls)[4] is False
    assert _v(_four(said=10, right=9))[5] is True and _v(_four(said=9, right=9))[5] is False                           # 40 statements needed (36 is too few, however right)
    assert _v(_four(said=10, right=9))[5] is True and _v(_four(said=10, right=8))[5] is False                          # 36 of 40 is 90; 32 of 40 is 80 percent
    r85 = _four(said=10, right=9); r85["g50t-1"] = _pg(said=10, right=7); r85["ka59-1"] = _pg(said=10, right=9); assert _v(r85)[5] is True      # 34 of 40 = 85 percent
    r85["ka59-1"] = _pg(said=10, right=8); assert _v(r85)[5] is False
    assert _v(_four(predicted=100, hit=80))[6] is True and _v(_four(predicted=100, hit=79))[6] is False
    assert _v(_four(predicted=100, hit=90))[7] is True and _v(_four(predicted=100, hit=89))[7] is False
    assert _v(_four(floor=10, stat=5))[8] is True and _v(_four(floor=10, stat=4))[8] is False
    good = _four(predicted=100, hit=95); ch = F3.verdict({"off": {g: _pg(predicted=180 if i < 3 else 189, hit=130 if i < 3 else 137) for i, g in enumerate(GAMES)}, "full": good}, True)
    assert F3.headline(ch) == F3.HEADLINE and F3.claims_allowed(ch)
    ch2 = F3.verdict({"off": {g: _pg(predicted=180 if i < 3 else 189, hit=130 if i < 3 else 137) for i, g in enumerate(GAMES)}, "full": _four(predicted=100, hit=75)}, True); assert F3.headline(ch2).startswith("MIXED: line 3 yes, line 6 NO")
    assert F3.headline(F3.verdict(None, None)).startswith("NOT READ") and F3.headline(F3.verdict({"off": good, "full": good}, True)).startswith("NOT READ")


# ---------------------------------------------------------------- 5 page, notebook, replay of arm C, front pages
def _head():
    text = open(PAGE, encoding="utf-8").read(); assert text.count(RESULT_MARK) <= 1
    return text[:text.index(RESULT_MARK)] if RESULT_MARK in text else text


def test_the_page_above_its_result_is_what_was_written_before_the_run():
    assert hashlib.sha256(_head().encode()).hexdigest() == PAGE_SHA256


def test_the_page_and_the_module_agree_and_the_page_does_not_claim_the_pasted_texts_as_sirs():
    head = _head(); pred = head.split("## Predictions")[1].split("## Reading rules")[0]; rules = head.split("## Reading rules")[1].split("Headline:")[0]
    ps = re.findall(r"^(\d+)\. \[(\w+)\] (.*) \((\d+) %\)$", pred, re.M); rs = re.findall(r"^(\d+)\. (.*)$", rules, re.M)
    assert len(ps) == len(rs) == len(F3.G_LINES) == 8
    for (n, tag, text, pr), (m, rt), (gt, gtext) in zip(ps, rs, F3.G_LINES): assert n == m and text == rt == gtext and tag == gt and 0 < int(pr) < 100
    for needle in (F3.HEADLINE, "Sir to confirm", "RFC-001", "527", "729", "65 of 68", "82 of 88", "Neither text is Sir's", "92.45 %", "will not show", "85.2 %", "90.6 %", "85.7 %", "90.9 %", "7 %", "Result is not written yet"): assert needle in head, needle
    assert "$" not in head and "proven" not in head.lower() and str(F3.TYPE_MIN) in head and F3.SECOND == (527, 729)


def test_the_notebook_reads_the_saved_files_calls_no_model_holds_no_key_and_runs_four_ways():
    nb = json.load(open(NB)); src = "\n".join("".join(c["source"]) for c in nb["cells"]); code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert len(nb["cells"]) == 12 and all(c["outputs"] == [] for c in nb["cells"] if c["cell_type"] == "code")
    assert not re.search(r"AIza[0-9A-Za-z_-]{20,}", src) and "GOOGLE_API_KEY" not in src and "userdata" not in src and "drive.mount" not in src and "open_store" not in src
    for needle in ("foresee3 as F", "F.from_session", "LR.run_file", "F.verdict", "F.headline", "F.VARIANTS", "axi_gemini_arc_c_default.json", "axi_gemini_arc_b_default.json", "axi_gemini_arc_default.json", "foresee_check_3_all.json", "foresee3.py"): assert needle in code, needle
    assert "FORESEE_CHECK_3.md" in src and "foresee2" not in code and "generate" not in code and ".run(" not in code.replace("LR.run_file(", "")
    for c in nb["cells"]:
        if c["cell_type"] == "code": compile("\n".join(l for l in "".join(c["source"]).split("\n") if not l.lstrip().startswith("!")), "cell", "exec")


def test_the_third_check_runs_over_a_replay_of_arm_c_on_the_stand_in_game_four_ways():
    res = C.run(T.FakeLLM(), T.make_env, os.path.join(os.environ.get("TMPDIR", "/tmp"), "foresee3_c_test.jsonl"), games=("alpha-1", "beta-2"), moves=30, action_of=lambda n: n)
    s = LR.feed(res["rows"], T.make_env, "C", games=("alpha-1", "beta-2"), moves=30, action_of=lambda n: n); vs = F3.from_session(s)
    assert set(vs) == {"off", "front", "typed", "full"} and all(set(pg) == {"alpha-1", "beta-2"} for pg in vs.values())
    assert all(sum(c["moves"] for c in pg.values()) == sum(s.moves.values()) for pg in vs.values()) and "stops by action" in F3.text(vs) and "ALL GAMES" in F3.text(vs)
    assert all(vs["off"][g]["predicted"] == vs["full"][g]["predicted"] for g in vs["off"])                              # the same moves are predicted four ways


def test_the_front_pages_say_what_was_built_and_that_it_was_run_once_with_the_result_not_written():
    cm = open(os.path.join(DOCS, "RAJNISH_CONCEPT_MAP.md"), encoding="utf-8").read(); vf = open(os.path.join(DOCS, "VERIFIED.md"), encoding="utf-8").read(); hl = open(os.path.join(DOCS, "HISTORY_LEDGER.md"), encoding="utf-8").read()
    assert NEXT in cm and cm.count("Next free:") == 1 and "\n138. **" in cm and "\n139. **" in cm and "\n140. **" in cm and "\n141. **" not in cm
    i = cm.index("\n138. **") + 1; item = cm[i:cm.index("\n", i)]
    for n in ("FORESEE_CHECK_3.md", "foresee3.py", "test 163", "Rajnish Choubey", "Sir to confirm", "Result not written", "8 reading lines", "not Sir's", "603 of 729", PAGE_SHA256): assert n in item, n
    assert any("FORESEE_CHECK_3.md" in l and "foresee3.py" in l and PAGE_SHA256 in l and "Result is not written" in l and "603 of 729" in l for l in vf.split("\n"))
    assert "| 184 (M) |" in hl and "front" in hl and "foresee_check_3_all.json" in hl


if __name__ == "__main__":
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]; bad = 0
    for n, f in fns:
        try: f(); print("ok  ", n)
        except Exception: bad += 1; print("FAIL", n); traceback.print_exc()
    print(f"{len(fns) - bad}/{len(fns)} passed"); sys.exit(1 if bad else 0)
