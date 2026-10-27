"""Test 161 (item 136): the forward check on the saved ARC-AGI-3 runs (docs/FORESEE_CHECK.md, axi/experiments/foresee.py, colab/axi_foresee_check.ipynb).

Comes from pillar III of Rajnish Choubey's RFC-001 (a candidate is checked on a shadow before it is committed), read for game terms, for Sir to confirm. No model is called and no benchmark game is used here. It checks:
  1. the prediction made by hand on a tiny world: the step of an action, "does not move" at a wall and at the edge of the screen, no prediction for a click or an action tried once;
  2. that the colours that never changed are not always walls: a block that has not been pushed yet is taken for a wall (the invented push world: said 10 times, right 0 times), and what stopped the piece before is learned from what was seen, never from the screen it is about to predict;
  3. on the invented worlds (not benchmark games; 10 seeds of 300 random moves each, exact counts), the learned prediction against step only;
  4. a pure function: the same moves give the same counts and the inputs are not changed;
  5. the 8 reading lines at their boundaries, the page (hash, agreement with the module), the notebook, the replay tool reading arm C, the saved result file of Test 26, the front pages.
What it does NOT show: how well it predicts on the real games. That is the Result of the run (Colab)."""
import hashlib, json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.ledger import Ledger
from axi.experiments import foresee as F
from axi.experiments import ledger_replay as LR
from axi.experiments.mini_worlds import World, walk, PUSH_LEVEL, PAD_LEVEL
from tests import test_gemini_arc as T
from axi.experiments import gemini_arc_c as C

HERE = os.path.dirname(os.path.abspath(__file__)); DOCS = os.path.join(HERE, "..", "..", "docs")
PAGE = os.path.join(DOCS, "FORESEE_CHECK.md"); NB = os.path.join(HERE, "..", "colab", "axi_foresee_check.ipynb"); RESULT_C = os.path.join(DOCS, "results", "axi_gemini_arc_c_default.json")
PAGE_SHA256 = "d9d1e853efeb51d7bb8fb7eb449063f67773efab223404756974b5ed0a5e8b1b"; RESULT_MARK = "\n## Result"
NEXT = "Next free: row 188, item 142, test 167, question 122"


def row(*cells): return tuple(cells)


def step_world(n, wall_at=None, w=8):
    """A piece 'P' on a row that ACTION4 moves one cell right; '#' at wall_at (a column) in every row of a 3-row screen. Frames after n moves (stops at the wall)."""
    frames = []; x = 1
    for k in range(n + 1):
        frames.append(tuple(("".join("P" if (c == x and r == 1) else ("#" if c == wall_at else ".") for c in range(w))) for r in range(3)))
        if not (wall_at is not None and x + 1 == wall_at): x += 1
    return frames


# ---------------------------------------------------------------- 1 by hand
def test_the_step_the_wall_the_edge_and_the_two_tries():
    fr = step_world(6, wall_at=6); L = Ledger()
    assert F.foresee(L, fr[0], "ACTION4") is None                                           # nothing seen
    L.record("ACTION4", fr[0], fr[1]); assert F.foresee(L, fr[1], "ACTION4") is None                      # one try is not enough
    L.record("ACTION4", fr[1], fr[2]); p = F.foresee(L, fr[2], "ACTION4")
    assert p["symbol"] == "P" and p["step"] == (1, 0) and p["share"] == 1.0 and p["tries"] == 2 and p["blocked"] is False and p["wall_known"] is True
    L.record("ACTION4", fr[2], fr[3]); p = F.foresee(L, fr[3], "ACTION4")                                # P at x=4, wall at 6: x=5 is free
    assert p["blocked"] is False
    L.record("ACTION4", fr[3], fr[4]); p = F.foresee(L, fr[4], "ACTION4")                                # P at x=5, next cell is the wall '#': rigid (3+ moves, never changed)
    assert p["blocked"] is True and "#" in p["target"]
    assert F.foresee(L, fr[4], "ACTION6@3,3") is None and F.foresee(L, fr[4], "ACTION1") is None            # a click and an action never seen
    edge = step_world(7, wall_at=None, w=4); L2 = Ledger()
    for i in range(2): L2.record("ACTION4", edge[i], edge[i + 1])
    q = F.foresee(L2, edge[2], "ACTION4"); assert q["blocked"] is True and F.EDGE in q["target"]          # P at the last column: the next cell is outside the screen


def test_a_block_that_has_not_been_pushed_yet_is_taken_for_a_wall_and_the_learned_prediction_does_not():
    m = [("R",) + (x,) for x in range(0)]                                                          # (placeholder to keep the style of the other tests)
    rows = lambda px, bx: tuple("".join("P" if (c, r) == (px, 1) else "B" if (c, r) == (bx, 1) else "." for c in range(9)) for r in range(3))
    seq = [("R", rows(1, 4), rows(2, 4)), ("R", rows(2, 4), rows(3, 4)), ("R", rows(3, 4), rows(3, 4))]      # the third move: P stands next to B and does not move (blocked by it)
    c = F.evaluate(seq + [("R", rows(3, 4), rows(3, 4))])
    assert c["predicted"] == 2 and c["blocked_said"] == 1 and c["blocked_right"] == 1             # B never changed: taken as a wall, and here it was right
    # now B is pushed: it moves, and P with it. The wall reading said "blocked" in front of it, the learned one has never seen a stop.
    seq2 = [("R", rows(1, 6), rows(2, 6)), ("R", rows(2, 6), rows(3, 6)), ("R", rows(3, 6), rows(4, 6)), ("R", rows(4, 6), rows(5, 6)), ("R", rows(5, 6), rows(6, 7))]
    c2 = F.evaluate(seq2)
    assert c2["blocked_said"] == 1 and c2["blocked_right"] == 0 and c2.get("learned_said", 0) == 0 and c2["wall_hit"] < c2["learned_hit"]


def test_what_stopped_the_piece_is_learned_from_earlier_moves_only():
    rows = lambda px: tuple("".join("P" if (c, r) == (px, 1) else "X" if c == 5 else "." for c in range(7)) for r in range(3))
    # X never changes at x=5 so it is rigid after 3 moves; in this run P is stopped by it first and only then predicted
    seq = [("R", rows(1), rows(2)), ("R", rows(2), rows(3)), ("R", rows(3), rows(4)), ("R", rows(4), rows(4)), ("R", rows(4), rows(4)), ("R", rows(4), rows(4))]
    c = F.evaluate(seq)
    assert c["predicted"] == 4 and c["learned_said"] == 2 and c["learned_right"] == 2            # the move that was stopped first is a miss; after it the stop is learned and said


# ---------------------------------------------------------------- 3 the invented worlds
def _worlds(level, **kw): return F.combine([F.evaluate(walk(World(level, **kw), s, 300)) for s in range(10)])


def test_on_the_invented_worlds_the_counts_are_exact_and_the_learned_prediction_is_the_best_where_a_wall_can_be_learned():
    c = _worlds(PUSH_LEVEL, pads=False)
    assert (c["moves"], c["predicted"], c["step_hit"], c["wall_hit"], c["learned_hit"], c["blocked_said"], c["blocked_right"]) == (3000, 2294, 1670, 1660, 1670, 10, 0)     # the push world: a block is not a wall; nothing to learn from
    p = _worlds(PAD_LEVEL)
    assert (p["moves"], p["predicted"], p["step_hit"], p["wall_hit"], p["learned_hit"], p["learned_said"], p["learned_right"]) == (3000, 2295, 1851, 2187, 2246, 399, 397)
    r = F.rates(p); assert r["step_only"] < 0.81 < 0.95 < r["wall_aware"] < r["learned"] and r["learned_precision"] > 0.99 and abs(r["coverage"] - 0.765) < 1e-9


def test_the_check_is_a_pure_function_and_changes_nothing_it_is_given():
    moves = walk(World(PAD_LEVEL), 3, 120); snap = json.dumps(moves)
    a = F.evaluate(moves); b = F.evaluate(list(moves)); assert a == b and json.dumps(moves) == snap
    assert F.rates({}) == dict(moves=0, predicted=0, abstained=0, coverage=None, step_only=None, wall_aware=None, learned=None, learned_said=0, learned_precision=None, blocked_said=0, blocked_precision=None, did_not_move=0, other_step=0, wall_unknown=0)
    assert F.text({}) == ""


# ---------------------------------------------------------------- 5 the reading lines
def _pg(moves=240, predicted=200, step=150, learned=190, wall=180, said=20, right=19):
    return dict(moves=moves, predicted=predicted, step_hit=step, learned_hit=learned, wall_hit=wall, learned_said=said, learned_right=right)


def _v(games, ok=True): return {i: v for i, _, _, v in F.verdict(games, ok)}


def test_the_eight_lines_flip_at_their_boundaries_and_the_headline_needs_lines_4_and_6():
    assert len(F.G_LINES) == 8 and [t for t, _ in F.G_LINES] == ["guard", "instrument"] + ["claim"] * 6
    four = lambda **kw: {f"g{i}": _pg(**kw) for i in range(4)}
    assert _v(None, None) == {i: None for i in range(1, 9)} and _v(four(), False)[1] is False
    assert _v(four(predicted=144))[2] is True and _v(four(predicted=143))[2] is False                                  # 60 percent of the recorded moves
    assert _v(four(predicted=100, step=70, learned=90, wall=80))[3] is True and _v(four(predicted=100, step=69, learned=90, wall=80))[3] is False        # 70 percent
    g = four(); g["g0"] = _pg(step=190, learned=190); assert _v(g)[4] is True                                          # 3 of the 4 games better is enough
    g["g1"] = _pg(step=190, learned=190); assert _v(g)[4] is False
    assert _v(four(said=100, right=80))[5] is True and _v(four(said=100, right=79))[5] is False                         # 80 percent of "does not move"
    assert _v(four(predicted=100, learned=90, step=60, wall=70))[6] is True and _v(four(predicted=100, learned=89, step=60, wall=70))[6] is False       # 90 percent over all
    assert _v(four(wall=190, learned=190))[7] is True and _v(four(wall=191, learned=190))[7] is False
    assert _v(four(predicted=100, step=90, learned=88, wall=80))[8] is True and _v(four(predicted=100, step=90, learned=87, wall=80))[8] is False   # 2 points
    good = four(moves=150, predicted=100, step=60, learned=95, wall=80, said=10, right=10); ch = F.verdict(good, True)
    assert F.headline(ch) == F.HEADLINE and F.claims_allowed(ch)
    ch2 = F.verdict(four(moves=150, predicted=100, step=60, learned=85, wall=80, said=10, right=10), True); assert F.headline(ch2).startswith("MIXED: line 4 yes, line 6 NO")
    assert F.headline(F.verdict(None, None)).startswith("NOT READ") and F.headline(F.verdict(good, False)).startswith("NOT READ")


# ---------------------------------------------------------------- 6 page, notebook, replay tool, saved result
def _head():
    text = open(PAGE, encoding="utf-8").read(); assert text.count(RESULT_MARK) <= 1
    return text[:text.index(RESULT_MARK)] if RESULT_MARK in text else text


def test_the_page_above_its_result_is_what_was_written_before_the_run():
    assert hashlib.sha256(_head().encode()).hexdigest() == PAGE_SHA256


def test_the_page_and_the_module_agree_and_the_page_does_not_claim_the_pasted_texts_as_sirs():
    head = _head(); pred = head.split("## Predictions")[1].split("## Reading rules")[0]; rules = head.split("## Reading rules")[1].split("Headline:")[0]
    ps = re.findall(r"^(\d+)\. \[(\w+)\] (.*) \((\d+) %\)$", pred, re.M); rs = re.findall(r"^(\d+)\. (.*)$", rules, re.M)
    assert len(ps) == len(rs) == len(F.G_LINES) == 8
    for (n, tag, text, pr), (m, rt), (gt, gtext) in zip(ps, rs, F.G_LINES): assert n == m and text == rt == gtext and tag == gt and 0 < int(pr) < 100
    for needle in (F.HEADLINE, "Sir to confirm", "pillar III", "RFC-001", "No model", "not text of his", "will not show", "960", "Click actions"): assert needle.lower() in head.lower() or needle in head, needle
    assert "$" not in head and "proven" not in head.lower()


def test_the_notebook_reads_the_saved_files_calls_no_model_and_holds_no_key():
    nb = json.load(open(NB)); src = "\n".join("".join(c["source"]) for c in nb["cells"]); code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert len(nb["cells"]) == 12 and all(c["outputs"] == [] for c in nb["cells"] if c["cell_type"] == "code")
    assert not re.search(r"AIza[0-9A-Za-z_-]{20,}", src) and "GOOGLE_API_KEY" not in src and "userdata" not in src and "drive.mount" not in src and "open_store" not in src
    for needle in ("F.from_session", "LR.run_file", "F.verdict", "F.headline", "axi_gemini_arc_c_default.json", "axi_gemini_arc_b_default.json", "axi_gemini_arc_default.json", "foresee_check_all.json", '"K"', '"A"', '"B"', '"C"'): assert needle in code, needle
    assert ".run(" not in code.replace("LR.run_file(", "") and "generate" not in code
    for c in nb["cells"]:
        if c["cell_type"] == "code": compile("\n".join(l for l in "".join(c["source"]).split("\n") if not l.lstrip().startswith("!")), "cell", "exec")


def test_the_replay_tool_reads_arm_c_and_the_saved_result_of_test_26_is_in_the_repo():
    res = C.run(T.FakeLLM(), T.make_env, os.path.join(os.environ.get("TMPDIR", "/tmp"), "foresee_c_test.jsonl"), games=("alpha-1", "beta-2"), moves=30, action_of=lambda n: n); rows = res["rows"]
    s = LR.feed(rows, T.make_env, "C", games=("alpha-1", "beta-2"), moves=30, action_of=lambda n: n)
    assert s.ledgers and sum(s.moves.values()) > 40 and set(s.moves) == {"alpha-1", "beta-2"}
    pg = F.from_session(s); assert set(pg) == {"alpha-1", "beta-2"} and sum(c["moves"] for c in pg.values()) == sum(s.moves.values()) and "ALL GAMES" in F.text(pg)
    rows2, games, moves = LR.load(RESULT_C, "C"); assert len(rows2) == 240 and games == C.GAMES and moves == 60 and sum(r["levels"] > 0 for r in rows2) >= 2 and all("ledger_lines" in r for r in rows2)
    d = json.load(open(RESULT_C)); assert set(d) == {"C", "B", "A", "K", "checks", "replays"} and all(d["replays"].values()) and [c[3] for c in d["checks"]].count(False) == 1 and d["checks"][7][3] is False


# ---------------------------------------------------------------- 7 front pages
def test_the_front_pages_say_what_was_built_and_that_it_was_run_once_with_the_result_not_written():
    cm = open(os.path.join(DOCS, "RAJNISH_CONCEPT_MAP.md"), encoding="utf-8").read(); vf = open(os.path.join(DOCS, "VERIFIED.md"), encoding="utf-8").read(); hl = open(os.path.join(DOCS, "HISTORY_LEDGER.md"), encoding="utf-8").read()
    assert NEXT in cm and cm.count("Next free:") == 1 and "\n136. **" in cm and "\n137. **" in cm and "\n138. **" in cm and "\n139. **" in cm and "\n140. **" in cm and "\n142. **" not in cm
    i = cm.index("\n136. **") + 1; item = cm[i:cm.index("\n", i)]
    for n in ("FORESEE_CHECK.md", "foresee.py", "test 161", "Rajnish Choubey", "Sir to confirm", "Result not written", "8 reading lines", "pillar III", PAGE_SHA256): assert n in item, n
    assert any("FORESEE_CHECK.md" in l and "foresee.py" in l and PAGE_SHA256 in l and "Result is not written" in l for l in vf.split("\n"))
    assert "| 182 (K) |" in hl and "forward" in hl.lower()


if __name__ == "__main__":
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]; bad = 0
    for n, f in fns:
        try: f(); print("ok  ", n)
        except Exception: bad += 1; print("FAIL", n); traceback.print_exc()
    print(f"{len(fns) - bad}/{len(fns)} passed"); sys.exit(1 if bad else 0)
