"""Test 164 (item 139): the fourth forward check (docs/FORESEE_CHECK_4.md, axi/experiments/foresee4.py, colab/axi_foresee_check_4.ipynb): the board carried from level to level.

The first three checks start every level from nothing; 207 of 936 moves (22 %) of the second check were not predicted at all. This test checks, by hand and on small worlds built for it:
  1. a law carried from an earlier level of the same game predicts the first moves of the next level (a plain step, as the mover is not yet located), and is used only if its symbol is on the screen;
  2. the floor carried lets the whole-front rule hold a front at the first stop of a new level, which the laws alone cannot;
  3. nothing is carried from a later level, from another game, or when a way does not say so; the fresh way is the third check (test 163) counter for counter;
  4. the counts on the invented worlds that the page quotes, the 8 reading lines at their boundaries, the page (its hash, its lines, what it says it will not show), the notebook and the front pages.
What it does NOT show: how well the carry predicts on the real games, nor whether a playing model does better with it. That is the Result of the run (Colab) and a paid test."""
import hashlib, json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.ledger import Ledger
from axi.experiments import foresee3 as F3
from axi.experiments import foresee4 as F4
from axi.experiments import ledger_replay as LR
from axi.experiments import gemini_arc_c as C
from axi.experiments.mini_worlds import World, walk, PUSH_LEVEL, PAD_LEVEL
from tests import test_gemini_arc as T

HERE = os.path.dirname(os.path.abspath(__file__)); DOCS = os.path.join(HERE, "..", "..", "docs")
PAGE = os.path.join(DOCS, "FORESEE_CHECK_4.md"); NB = os.path.join(HERE, "..", "colab", "axi_foresee_check_4.ipynb")
PAGE_SHA256 = "aec9788e7defdc121743c1c0abe218c4be89d53995b2f66ed1e3037d042816eb"; RESULT_MARK = "\n## Result"
NEXT = "Next free: row 186, item 140, test 165, question 122"
KEYS = ("moves", "predicted", "step_hit", "cell_hit", "carried_pred", "carried_hit", "known", "cell_said", "cell_right", "held_cells", "dnm_object", "dnm_unknown")
get = lambda c: {k: c.get(k, 0) for k in KEYS}


def rowW(x, w=12, s="P", wall=None): return tuple("".join(s if (c, r) == (x, 1) else ("W" if (wall is not None and c == wall) else ".") for c in range(w)) for r in range(3))


LV1 = [("S", rowW(i), rowW(i + 1)) for i in range(1, 7)]
def carry(laws=True, floor=False): return dict(laws={}, walked=set(), use_laws=laws, use_floor=floor)


# ---------------------------------------------------------------- 1 a carried law
def test_a_law_from_an_earlier_level_predicts_the_first_moves_of_the_next_level_as_a_plain_step():
    lv2 = [("S", rowW(i), rowW(i + 1)) for i in range(2, 6)]; cy = carry(); F3.evaluate(LV1, True, True, carry=cy)
    assert cy["laws"] == {"S": ("P", (1, 0), 6, 6)} and cy["walked"] == {"."}
    with_c = get(F3.evaluate(lv2, True, True, carry=cy)); fresh = get(F3.evaluate(lv2, True, True))
    assert fresh["predicted"] == 2 and fresh["carried_pred"] == 0                                                         # two tries before the check speaks
    assert (with_c["predicted"], with_c["carried_pred"], with_c["carried_hit"], with_c["step_hit"], with_c["cell_hit"]) == (4, 2, 2, 4, 4)


def test_a_carried_law_is_used_only_when_its_symbol_is_on_the_screen_and_never_without_the_switch():
    cy = carry(); F3.evaluate(LV1, True, True, carry=cy); other = [("S", rowW(i, s="Q"), rowW(i + 1, s="Q")) for i in range(2, 6)]
    assert F3.evaluate(other, True, True, carry=dict(cy)).get("carried_pred", 0) == 0                                      # 'Q' is not 'P': nothing is predicted from the old law
    cy2 = carry(laws=False); F3.evaluate(LV1, True, True, carry=cy2)
    assert F3.evaluate([("S", rowW(i), rowW(i + 1)) for i in range(2, 6)], True, True, carry=cy2).get("carried_pred", 0) == 0     # laws are kept but not used when the way does not say so


# ---------------------------------------------------------------- 2 the carried floor
def test_the_floor_carried_lets_the_front_be_held_at_the_first_stop_of_a_new_level():
    lv2 = [("S", rowW(4, wall=6), rowW(5, wall=6))] + [("S", rowW(5, wall=6), rowW(5, wall=6))] * 4
    out = {}
    for uf in (False, True): cy = carry(floor=uf); F3.evaluate(LV1, True, True, carry=cy); out[uf] = get(F3.evaluate(lv2, True, True, carry=cy))
    assert (out[False]["cell_hit"], out[False]["cell_said"], out[False]["dnm_unknown"], out[False]["held_cells"]) == (1, 0, 4, 0)       # no floor known yet: nothing is held
    assert (out[True]["cell_hit"], out[True]["cell_said"], out[True]["cell_right"], out[True]["dnm_object"], out[True]["held_cells"]) == (4, 3, 3, 4, 1)


# ---------------------------------------------------------------- 3 nothing from later levels or other games; the fresh way is the third check
def _session(levels):
    class S: pass
    s = S(); s.ledgers = {}
    for (g, lv, seq) in levels:
        L = Ledger()
        for m in seq: L.record(*m)
        s.ledgers[(g, lv)] = L
    return s


def test_nothing_is_carried_from_a_later_level_or_from_another_game():
    lv2 = [("S", rowW(3, wall=6), rowW(4, wall=6)), ("S", rowW(4, wall=6), rowW(5, wall=6))] + [("S", rowW(5, wall=6), rowW(5, wall=6))] * 4
    w = F4.from_session(_session([("a", 1, LV1), ("a", 2, lv2), ("b", 1, lv2)]))
    assert {n: (w[n]["a"]["moves"], w[n]["a"]["predicted"], w[n]["a"].get("carried_pred", 0), w[n]["b"]["predicted"], w[n]["b"].get("carried_pred", 0)) for n in F4.WAYS} == {
        "fresh": (12, 8, 0, 4, 0), "laws": (12, 10, 2, 4, 0), "laws+floor": (12, 10, 2, 4, 0)}                       # game b has no earlier level of its own: nothing carried, whatever game a taught
    rev = F4.from_session(_session([("a", 2, lv2), ("a", 1, LV1)]))                                                      # the order of the levels is by level number, not by insertion
    assert rev["laws"]["a"]["predicted"] == 10 and rev["laws"]["a"]["carried_pred"] == 2


def test_the_fresh_way_is_the_third_check_with_both_changes_counter_for_counter_on_the_invented_worlds():
    for mk in (lambda: World(PUSH_LEVEL, pads=False), lambda: World(PAD_LEVEL)):
        seqs = [walk(mk(), lv, 60) for lv in range(1, 6)]; w = F4.from_session(_session([("g", lv + 1, s) for lv, s in enumerate(seqs)]))
        a = F3.combine([F3.evaluate(s, True, True) for s in seqs]); assert w["fresh"]["g"] == a


def test_on_the_invented_worlds_the_counts_and_rates_are_those_of_the_page():
    out = {}
    for nm, mk in (("push", lambda: World(PUSH_LEVEL, pads=False)), ("pad", lambda: World(PAD_LEVEL))):
        seqs = [walk(mk(), lv, 60) for lv in range(1, 6)]; w = F4.from_session(_session([("g", lv + 1, s) for lv, s in enumerate(seqs)])); out[nm] = {k: F4.rates(v["g"]) for k, v in w.items()}
    f = lambda x: "%.1f %%" % (100 * x)
    assert [f(out["pad"][k]["coverage"]) for k in F4.WAYS] == ["64.3 %", "76.7 %", "76.7 %"] and [f(out["push"][k]["coverage"]) for k in F4.WAYS] == ["64.3 %", "76.7 %", "76.7 %"]
    assert [f(out["pad"][k]["per_move"]) for k in F4.WAYS] == ["54.7 %", "64.0 %", "59.7 %"] and [f(out["push"][k]["per_move"]) for k in F4.WAYS] == ["54.0 %", "63.3 %", "54.0 %"]


def test_the_check_is_a_pure_function_and_changes_nothing_it_is_given():
    seq = walk(World(PAD_LEVEL), 3, 120); snap = json.dumps(seq); cy = carry(); a = F3.evaluate(seq, True, True, carry=cy); assert json.dumps(seq) == snap and cy["laws"]
    b = F3.evaluate(list(seq), True, True, carry=carry()); assert a == b and F4.text({}) == "" and F4.rates({})["per_move"] is None and F4.rates({})["carried_right"] is None
    assert F4.WAYS == {"fresh": (False, False), "laws": (True, False), "laws+floor": (True, True)} and F4.FRESH_PREDICTED == 729


# ---------------------------------------------------------------- 4 the reading lines
GAMES = ("g50t-1", "ka59-1", "ls20-1", "sp80-1")


def _pg(moves=234, predicted=182, hit=140, carried=0, chit=0, said=10, right=9):
    return dict(moves=moves, predicted=predicted, known=predicted, step_hit=hit - 5, cell_hit=hit, cell_said=said, cell_right=right, carried_pred=carried, carried_hit=chit)


def _v(fresh=None, laws=None, lf=None, ok=True):
    fresh = fresh or {g: _pg(predicted=182 if i < 3 else 183) for i, g in enumerate(GAMES)}            # 182 * 3 + 183 = 729
    laws = laws or {g: _pg(predicted=210, hit=160, carried=15, chit=10) for g in GAMES}; lf = lf or laws
    return {i: v for i, _, _, v in F4.verdict({"fresh": fresh, "laws": laws, "laws+floor": lf}, ok)}


def test_the_eight_lines_flip_at_their_boundaries_and_the_headline_needs_lines_4_and_6():
    assert len(F4.G_LINES) == 8 and [t for t, _ in F4.G_LINES] == ["guard", "instrument"] + ["claim"] * 6
    assert {i: v for i, _, _, v in F4.verdict(None, None)} == {i: None for i in range(1, 9)} and _v(ok=False)[1] is False
    assert _v()[2] is True and _v(fresh={g: _pg(predicted=182) for g in GAMES})[2] is False                          # 728 predicted is not the 729 of the earlier checks
    assert _v(laws={g: _pg(moves=100, predicted=85, hit=60) for g in GAMES})[3] is True and _v(laws={g: _pg(moves=100, predicted=84, hit=60) for g in GAMES})[3] is False     # 85 percent
    fr = {g: _pg(predicted=182 if i < 3 else 183, hit=100) for i, g in enumerate(GAMES)}
    lw = {g: _pg(predicted=210, hit=101) for g in GAMES}; lw["g50t-1"] = _pg(predicted=210, hit=100); assert _v(fresh=fr, laws=lw)[4] is True                   # better in 3 games, equal in one: 3 of 4 is enough
    lw2 = dict(lw); lw2["ka59-1"] = _pg(predicted=210, hit=100); assert _v(fresh=fr, laws=lw2)[4] is False                   # better in 2 games only
    five = lambda carried, chit: {g: _pg(predicted=210, hit=160, carried=carried, chit=chit) for g in GAMES}
    assert _v(laws=five(13, 8))[5] is True and _v(laws=five(12, 8))[5] is False                                       # 52 carried moves, 32 right (61.5 percent): fine; 48 carried: too few
    assert _v(laws=five(15, 9))[5] is True and _v(laws=five(15, 8))[5] is False                                       # 36 of 60 = 60 percent; 32 of 60 is not
    base = {g: _pg(predicted=182 if i < 3 else 183, hit=140) for i, g in enumerate(GAMES)}                              # fresh: 560 of 729 = 76.8 percent
    same = {g: _pg(predicted=182 if i < 3 else 183, hit=140) for i, g in enumerate(GAMES)}
    assert _v(fresh=base, laws=same)[6] is True                                                                       # no worse
    worse = {g: _pg(predicted=182 if i < 3 else 183, hit=136) for i, g in enumerate(GAMES)}; assert _v(fresh=base, laws=worse)[6] is False   # more than 2 points fewer right among predicted
    just = {g: _pg(predicted=182 if i < 3 else 183, hit=138) for i, g in enumerate(GAMES)}; assert _v(fresh=base, laws=just)[6] is True
    assert _v(laws={g: _pg(predicted=210, hit=160) for g in GAMES}, lf={g: _pg(predicted=210, hit=160) for g in GAMES})[7] is True
    assert _v(laws={g: _pg(predicted=210, hit=160) for g in GAMES}, lf={g: _pg(predicted=210, hit=159) for g in GAMES})[7] is False
    assert _v(laws={g: _pg(moves=100, predicted=90, hit=70) for g in GAMES})[8] is True and _v(laws={g: _pg(moves=100, predicted=90, hit=69) for g in GAMES})[8] is False     # 70 percent of the moves played
    good = {g: _pg(predicted=210, hit=170, carried=15, chit=12) for g in GAMES}
    ch = F4.verdict({"fresh": {g: _pg(predicted=182 if i < 3 else 183) for i, g in enumerate(GAMES)}, "laws": good, "laws+floor": good}, True); assert F4.headline(ch) == F4.HEADLINE and F4.claims_allowed(ch)
    weak = {g: _pg(predicted=210, hit=150, carried=15, chit=2) for g in GAMES}
    ch2 = F4.verdict({"fresh": {g: _pg(predicted=182 if i < 3 else 183) for i, g in enumerate(GAMES)}, "laws": weak, "laws+floor": weak}, True); assert F4.headline(ch2).startswith("MIXED: line 4 yes, line 6 NO")
    assert F4.headline(F4.verdict(None, None)).startswith("NOT READ")


# ---------------------------------------------------------------- 5 page, notebook, replay of arm C, front pages
def _head():
    text = open(PAGE, encoding="utf-8").read(); assert text.count(RESULT_MARK) <= 1
    return text[:text.index(RESULT_MARK)] if RESULT_MARK in text else text


def test_the_page_above_its_result_is_what_was_written_before_the_run():
    assert hashlib.sha256(_head().encode()).hexdigest() == PAGE_SHA256


def test_the_page_and_the_module_agree_and_it_says_which_text_is_whose():
    head = _head(); pred = head.split("## Predictions")[1].split("## Reading rules")[0]; rules = head.split("## Reading rules")[1].split("Headline:")[0]
    ps = re.findall(r"^(\d+)\. \[(\w+)\] (.*) \((\d+) %\)$", pred, re.M); rs = re.findall(r"^(\d+)\. (.*)$", rules, re.M)
    assert len(ps) == len(rs) == len(F4.G_LINES) == 8
    for (n, tag, text, pr), (m, rt), (gt, gtext) in zip(ps, rs, F4.G_LINES): assert n == m and text == rt == gtext and tag == gt and 0 < int(pr) < 100
    for needle in (F4.HEADLINE, "Sir to confirm", "one board that every view reads at once", "pasted AI answers", "207", "729", "77.9 %", "56.3", "64.3 %", "76.7 %", "54.7 %", "64.0 %", "63.3 %", "59.7 %", "will not show", "paid test", "$3.65", "Result is not written", "has not been run either"): assert needle in head or needle == "Result is not written", needle
    assert "$" in head and "proven" not in head.lower()


def test_the_notebook_reads_the_saved_files_calls_no_model_holds_no_key_and_runs_three_ways():
    nb = json.load(open(NB)); src = "\n".join("".join(c["source"]) for c in nb["cells"]); code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert len(nb["cells"]) == 12 and all(c["outputs"] == [] for c in nb["cells"] if c["cell_type"] == "code")
    assert not re.search(r"AIza[0-9A-Za-z_-]{20,}", src) and "GOOGLE_API_KEY" not in src and "userdata" not in src and "drive.mount" not in src and "open_store" not in src
    for needle in ("foresee4 as F", "F.from_session", "LR.run_file", "F.verdict", "F.headline", "F.WAYS", "axi_gemini_arc_c_default.json", "axi_gemini_arc_b_default.json", "axi_gemini_arc_default.json", "foresee_check_4_all.json", "foresee4.py"): assert needle in code, needle
    assert "FORESEE_CHECK_4.md" in src and "foresee3" not in code and "generate" not in code and ".run(" not in code.replace("LR.run_file(", "")
    for c in nb["cells"]:
        if c["cell_type"] == "code": compile("\n".join(l for l in "".join(c["source"]).split("\n") if not l.lstrip().startswith("!")), "cell", "exec")


def test_the_fourth_check_runs_over_a_replay_of_arm_c_on_the_stand_in_game_three_ways():
    res = C.run(T.FakeLLM(), T.make_env, os.path.join(os.environ.get("TMPDIR", "/tmp"), "foresee4_c_test.jsonl"), games=("alpha-1", "beta-2"), moves=30, action_of=lambda n: n)
    s = LR.feed(res["rows"], T.make_env, "C", games=("alpha-1", "beta-2"), moves=30, action_of=lambda n: n); w = F4.from_session(s)
    assert set(w) == {"fresh", "laws", "laws+floor"} and all(set(pg) == {"alpha-1", "beta-2"} for pg in w.values())
    assert all(sum(c["moves"] for c in pg.values()) == sum(s.moves.values()) for pg in w.values()) and "ALL GAMES" in F4.text(w) and "earlier level's law" in F4.text(w)
    assert all(w["fresh"][g].get("carried_pred", 0) == 0 for g in w["fresh"])


def test_the_front_pages_say_what_was_built_and_that_it_was_not_run_on_the_real_games():
    cm = open(os.path.join(DOCS, "RAJNISH_CONCEPT_MAP.md"), encoding="utf-8").read(); vf = open(os.path.join(DOCS, "VERIFIED.md"), encoding="utf-8").read(); hl = open(os.path.join(DOCS, "HISTORY_LEDGER.md"), encoding="utf-8").read()
    assert NEXT in cm and cm.count("Next free:") == 1 and "\n139. **" in cm and "\n140. **" not in cm
    i = cm.index("\n139. **") + 1; item = cm[i:cm.index("\n", i)]
    for n in ("FORESEE_CHECK_4.md", "foresee4.py", "test 164", "Rajnish Choubey", "Sir to confirm", "NOT RUN", "8 reading lines", "one board", PAGE_SHA256): assert n in item, n
    assert any("FORESEE_CHECK_4.md" in l and "foresee4.py" in l and PAGE_SHA256 in l and "NOT RUN" in l for l in vf.split("\n"))
    assert "| 185 (N) |" in hl and "carried" in hl


if __name__ == "__main__":
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]; bad = 0
    for n, f in fns:
        try: f(); print("ok  ", n)
        except Exception: bad += 1; print("FAIL", n); traceback.print_exc()
    print(f"{len(fns) - bad}/{len(fns)} passed"); sys.exit(1 if bad else 0)
