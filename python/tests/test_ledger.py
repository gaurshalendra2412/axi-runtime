"""Implements concept-map item 132 (the transition ledger: what the host can know exactly about a world it was not told the rules of;
python/axi/engine/ledger.py; docs/HISTORY_LEDGER.md; rows 172 to 178), test 158. It comes from the structure of Rajnish Choubey's RFC-001
(a thing is identity plus state, a rewrite keeps an interface K, the gate checks a change against the held state), read into game terms;
the reading is for Sir to confirm. It checks, with no model and no game from the benchmark:
  1. two screens become an object-level change: a step, a push (two things move), a recolour, a thing gone, a thing new, a screen of
     another size, an unchanged screen, two things of one symbol and shape (flagged ambiguous), two that touch (an object count that moves
     with the cell count still: `touch`); the exact changed cells are kept as well; the inputs are not changed; lists of numbers work;
  2. on small INVENTED worlds with one mechanic each (axi/experiments/mini_worlds.py; not benchmark games): 20 seeds of 1,000 random
     moves in each of a push world, a pad world whose mode shows and a pad world whose mode is hidden:
       hidden-state flag: 0 of 20, 0 of 20 and 16 of 20 (it needs the same screen twice under both modes); a scripted visit finds it every time;
       effects apart from what moved: 0, 732 and 5, equal to the worlds' own count of pad entries (732) and of crushed blocks (5);
       the object-level change adds up to the count change in every one of 60,000 moves;
  3. counts that never changed are kept as laws after 3 moves; a crushed block breaks one and it does not come back; a block hidden
     behind another thing breaks one and it comes back;
  4. the actions: the step of each, which are opposite or a quarter turn of each other (checked against `d4.py` for all 8 transforms),
     the unit steps no action has been seen to make, the changes no recorded move was seen to undo;
  5. claims about the rules, replayed over the history: a true one has no counterexample, a false one returns the moves that contradict
     it, an unplayed action is untested, an unknown form is refused;
  6. the cost: about 2 ms for a move that changed a 64 x 64 screen, 0.02 ms for one that did not;
  7. the page and the front pages (concept map item 132 and rows 172 to 178, the VERIFIED row, the counter).
What it does NOT show: that any of this makes a model clear more levels; anything on a real benchmark game (the packages could not be
installed where this was written, and the saved runs hold the model's replies, not the screens); that real games behave like these worlds."""
import json, os, sys, time, random
sys.path.insert(0, os.path.dirname(__file__))
from collections import Counter
from axi.engine.ledger import Ledger, match, components, relation_word, UNIT, MIN_LAW
from axi.experiments.mini_worlds import World, walk, PUSH_LEVEL, PAD_LEVEL

HERE = os.path.dirname(__file__)
DOCS = os.path.join(HERE, "..", "..", "docs")
PAGE = os.path.join(DOCS, "HISTORY_LEDGER.md")
NEXT = "Next free: row 179, item 134, test 160, question 122"


def ops(a, b, bg="."):
    L = Ledger(background=bg); r = L.record("x", a, b); return r.ops


def test_two_screens_become_an_object_level_change():
    assert ops(("......", "..P...", "......"), ("......", "...P..", "......")) == (("move", "P", (2, 1, 2, 1), (1, 0)),)
    push = ops((".PB...",), ("..PB..",)); assert sorted(push) == [("move", "B", (2, 0, 2, 0), (1, 0)), ("move", "P", (1, 0, 1, 0), (1, 0))]
    assert ops((".r.",), (".g.",)) == (("recolour", "r", "g", (1, 0, 1, 0), 1),)
    assert ops((".B.",), ("...",)) == (("gone", "B", (1, 0, 1, 0), 1),) and ops(("...",), (".B.",)) == (("new", "B", (1, 0, 1, 0), 1),)
    assert ops(("..P",), ("P..",)) == (("move", "P", (2, 0, 2, 0), (-2, 0)),)                      # a jump is a move of that size
    assert ops(("PP.",), (".PP",)) == (("move", "P", (0, 0, 1, 0), (1, 0)),)                      # an object of two cells that overlaps itself
    assert ops(("...", "..."), ("...", "...")) == ()
    L = Ledger(background="."); r = L.record("x", ("..", ".."), ("...", "...")); assert r.reshaped and r.ops[0][0] == "reshape" and r.diff == ()


def test_ambiguity_and_touching_and_exact_cells():
    L = Ledger(background="."); r = L.record("x", ("B.B...",), (".B.B..",)); assert r.ambiguous and [o[3] for o in r.ops] == [(1, 0), (1, 0)]
    L = Ledger(background="."); r = L.record("x", ("B..B",), ("..BB",))                         # one block moves up against the other: one object of two cells
    assert not r.ambiguous and r.touch == ("B",) and r.apart == () and [o[0] for o in r.ops] == ["gone", "gone", "new"]
    assert sorted(r.diff) == [(0, 0, "B", "."), (2, 0, ".", "B")]                                # the exact cells are right whatever the matching says
    L = Ledger(background="."); r = L.record("x", ("B.B.",), ("B...",))
    assert r.ops == (("gone", "B", (2, 0, 2, 0), 1),) and r.diff == ((2, 0, "B", "."),) and r.touch == ()


def test_inputs_are_not_changed_and_numbers_work():
    a = [[0, 0, 0], [0, 7, 0]]; b = [[0, 0, 0], [0, 0, 7]]; ca = [r[:] for r in a]; cb = [r[:] for r in b]
    L = Ledger(background=0); r = L.record(3, a, b); assert a == ca and b == cb
    assert r.ops == (("move", 7, (1, 1, 1, 1), (1, 0)),) and L.dominant() == {3: (7, (1, 0), 1, 1)}


def push_laws(seed=1, moves=400):
    w = World(PUSH_LEVEL, pads=False); L = Ledger(background="."); walk(w, seed, moves, L); return L


def test_hidden_state_scripted_and_by_seed():
    # to the pad (the mode flips), to the gate, in, out, back to the pad (the mode flips back), to the gate again: blocked
    for visible, found in ((False, 1), (True, 0)):
        w = World(PAD_LEVEL, visible); L = Ledger(background=".")
        for a in "RR" + "RR" + "R" + "LLL" + "RR" + "R": L.record(a, w.frame(), w.step(a))
        h = L.hidden_state(); assert len(h) == found
        if found: assert h[0]["action"] == "R" and h[0]["results"] == 2 and len(h[0]["moves"]) == 2
    out = {}
    for name, (lv, vis, pads) in {"push": (PUSH_LEVEL, True, False), "visible": (PAD_LEVEL, True, True), "hidden": (PAD_LEVEL, False, True)}.items():
        out[name] = sum(bool(l.hidden_state()) for l in (walk_ledger(lv, vis, pads, s) for s in range(20)))
    assert out == {"push": 0, "visible": 0, "hidden": 16}


def walk_ledger(lv, vis, pads, seed, moves=1000):
    w = World(lv, vis, pads); L = Ledger(background="."); walk(w, seed, moves, L); L.world = w; return L


def test_effects_apart_equal_the_worlds_own_count_and_the_change_adds_up():
    got = {}
    for name, (lv, vis, pads) in {"push": (PUSH_LEVEL, True, False), "visible": (PAD_LEVEL, True, True), "hidden": (PAD_LEVEL, False, True)}.items():
        apart = entries = crushed = recs = bad = 0
        for s in range(20):
            L = walk_ledger(lv, vis, pads, s); apart += sum(1 for r in L.rows if r.apart); entries += L.world.entries; crushed += L.world.crushed
            for r in L.rows:
                recs += 1; d = Counter()
                for o in r.ops:
                    if o[0] == "gone": d[("cells", o[1])] -= o[3]; d[("objects", o[1])] -= 1
                    elif o[0] == "new": d[("cells", o[1])] += o[3]; d[("objects", o[1])] += 1
                    elif o[0] == "recolour":
                        d[("cells", o[1])] -= o[4]; d[("cells", o[2])] += o[4]; d[("objects", o[1])] -= 1; d[("objects", o[2])] += 1
                if {q: v for q, v in d.items() if v} != {q: a - b for q, (b, a) in r.qdelta.items()}: bad += 1
        got[name] = (apart, entries, crushed, recs, bad)
    assert got == {"push": (0, 0, 0, 20000, 0), "visible": (732, 732, 5, 20000, 0), "hidden": (5, 732, 5, 20000, 0)}
    # in the world that shows its mode the far effects are the pad entries (gauge and gates); in the hidden one only a crush shows


def test_laws_and_breaks():
    L = push_laws(); laws = {l["quantity"]: l["value"] for l in L.laws()}
    assert laws[("objects", "B")] == 2 and laws[("objects", "P")] == 1 and laws[("cells", "P")] == 1 and L.rigid_symbols() == ["#"]
    assert Ledger(background=".").laws() == [] and len(push_laws(moves=2).laws()) == 0 and len(push_laws(moves=MIN_LAW).laws()) > 0   # three moves are needed
    w = World(PAD_LEVEL, True); L = Ledger(background=".")
    for a in "RRLDRRRULL": L.record(a, w.frame(), w.step(a))                                  # the last move steps on the pad: the block on the gate is crushed
    lost = [b for b in L.broken_laws() if not b["back"]]; assert w.crushed == 1
    assert [(b["quantity"], b["was"], b["now"], b["at"], b["action"]) for b in lost] == [(("cells", "B"), 2, 1, 9, "L"), (("objects", "B"), 2, 1, 9, "L")]
    assert L.rows[9].broke == (("cells", "B"), ("objects", "B")) and L.check_rule(("keeps", ("objects", "B")))["counter"] == [9]
    assert any("has not come back" in line for line in L.report())
    # a count that breaks and comes back is marked so: in the world that hides its mode, the pad hides under the player and reappears
    w = World(PAD_LEVEL, False); L = Ledger(background=".")
    for a in "DDUURRR": L.record(a, w.frame(), w.step(a))
    came = [(b["quantity"], b["was"], b["now"], b["at"], b["back_at"]) for b in L.broken_laws() if b["back"]]
    assert came == [(("cells", "T"), 2, 1, 5, 6), (("objects", "T"), 2, 1, 5, 6)] and any("came back" in line for line in L.report())


def test_the_actions_and_the_symmetry_group():
    L = push_laws(moves=1000); d = L.dominant()
    assert {a: (s, v) for a, (s, v, n, t) in d.items()} == {"U": ("P", (0, -1)), "D": ("P", (0, 1)), "L": ("P", (-1, 0)), "R": ("P", (1, 0))} and "A" not in d
    rel = Counter(w for _, _, w in L.relations()); assert rel == {"opposite": 2, "quarter-turn": 4} and L.completion() == []
    w = World(PUSH_LEVEL, pads=False); L = Ledger(background=".")
    for a in "DUDU": L.record(a, w.frame(), w.step(a))
    assert sorted(L.completion()) == [("P", (-1, 0)), ("P", (1, 0))] and [r for r in L.relations()] == [("D", "U", "opposite")]
    from axi.engine import d4
    import numpy as np
    for a in UNIT:
        words = []; images = set()
        for t in d4.IDS:
            g = np.zeros((5, 5), dtype=int); g[2 + a[1], 2 + a[0]] = 1; h = d4.apply(g, t); (row,), (col,) = np.nonzero(h)
            b = (int(col) - 2, int(row) - 2); images.add(b); words.append(relation_word(a, b))
        assert images == set(UNIT) and Counter(words) == {"same": 2, "opposite": 2, "quarter-turn": 4}          # the four steps are one orbit of D4


def test_no_return_seen():
    w = World(PUSH_LEVEL, pads=False); L = Ledger(background=".")
    for a in "RLRRRLR": L.record(a, w.frame(), w.step(a))
    n = L.no_return_seen(); assert n == dict(changed=7, returned=5, no_return=[3, 4])
    # move 0 was undone by move 1; move 5 by move 6. Moves 3 and 4 (the step up to the block, and the push) have no recorded move that puts the screen back:
    # that is "not seen", for the moves that were tried, and not "cannot be undone"
    assert w.blocks == {(5, 1), (4, 3)} and {o[1] for o in L.rows[4].ops} == {"B", "P"}


def test_apart_effects_and_a_push_are_told_apart():
    w = World(PAD_LEVEL, True); L = Ledger(background=".")
    for a in "RRD": L.record(a, w.frame(), w.step(a))                                        # a step, onto the pad, a push of the block below
    assert L.rows[0].apart == () and len(L.rows[1].apart) == 3 and L.rows[2].apart == () and {o[1] for o in L.rows[2].ops if o[0] == "move"} == {"B", "P"}
    assert {o[0] for o in L.rows[1].apart} == {"recolour", "gone", "new"}


def test_claims_about_the_rules_replayed_over_the_history():
    L = push_laws(moves=600)
    ok = L.check_rule(("moves", "R", "P", (1, 0))); assert ok["verdict"] == "no counterexample" and ok["counter_n"] == 0 and ok["idle"] > 0 and ok["supported"] > 0
    bad = L.check_rule(("moves", "R", "P", (0, 1))); assert bad["verdict"] == "refuted" and bad["counter_n"] == ok["supported"] and len(bad["counter"]) == 5
    assert L.check_rule(("moves", "X", "P", (0, 1)))["verdict"] == "untested" and L.check_rule(("moves", "X", "P", (0, 1)))["tested"] == 0
    assert L.check_rule(("keeps", ("objects", "P")))["verdict"] == "no counterexample" and L.check_rule(("changes_cell", "A", (1, 1)))["verdict"] == "refuted"
    w = World(PAD_LEVEL, True); L = Ledger(background="."); walk(w, 3, 600, L)
    on = L.check_rule(("on_arrival", "P", (3, 1), (0, 0))); assert (on["tested"], on["supported"], on["verdict"]) == (27, 27, "no counterexample")
    off = L.check_rule(("on_arrival", "P", (3, 1), (5, 5))); assert off["verdict"] == "refuted" and off["counter_n"] == 27
    assert L.check_rule(("changes_cell", "R", (0, 0)))["verdict"] == "refuted"                   # a rule with no condition says more than the world does
    two = Ledger(background="."); two.record("x", ("B....B",), (".B..B.",))                   # one move, two blocks, two different steps
    assert two.check_rule(("moves", "x", "B", (1, 0)))["verdict"] == "refuted" and two.check_rule(("moves", "x", "B", (-1, 0)))["verdict"] == "refuted"
    one = Ledger(background="."); one.record("x", ("B..",), (".B.",)); assert one.check_rule(("moves", "x", "B", (1, 0)))["verdict"] == "no counterexample"
    try: L.check_rule(("teleports", "R")); assert False
    except ValueError: pass


def test_the_cost_on_64_by_64_screens():
    rng = random.Random(5); g = [["0"] * 64 for _ in range(64)]
    for i in range(64): g[0][i] = g[63][i] = g[i][0] = g[i][63] = "5"
    for _ in range(300): g[rng.randrange(2, 62)][rng.randrange(2, 62)] = "5"
    def put(x, y, c):
        for dy in range(3):
            for dx in range(3): g[y + dy][x + dx] = c
    def fr(): return tuple("".join(r) for r in g)
    x, y = 10, 10; put(x, y, "3"); f0 = fr(); L = Ledger(background="0"); ch = []; eq = []
    for i in range(100):
        put(x, y, "0"); x = 10 + (i % 40); y = 10 + ((i // 40) % 30); put(x, y, "3"); f1 = fr()
        t0 = time.perf_counter(); r = L.record("A", f0, f1); ch.append(time.perf_counter() - t0); f0 = f1
        t0 = time.perf_counter(); L.record("B", f0, f0); eq.append(time.perf_counter() - t0)
    assert r.ops[0][0] == "move" and r.ops[0][1] == "3"
    assert sum(ch) / 100 < 0.02 and sum(eq) / 100 < 0.002                  # a model call took 15.9 to 28.7 s a move in Test 24


def test_the_page_and_the_front_pages():
    page = open(PAGE).read(); low = page.lower()
    for s in ("ledger.py", "Rajnish Choubey", "RFC-001", "for Sir to confirm", "hidden-state", "16 of 20", "732", "60,000", "invented", "not benchmark",
              "on_arrival", "touch", "came back", "no return", "d4.py", "2 ms", "0.02 ms", "does not show", "test 158", "item 132", "question 121"):
        assert s.lower() in low, s
    for bad in ("proven", "$", "dollar", "clears more levels", "solves"): assert bad not in low, bad
    cm = open(os.path.join(DOCS, "RAJNISH_CONCEPT_MAP.md")).read(); vf = open(os.path.join(DOCS, "VERIFIED.md")).read()
    assert NEXT in cm and cm.count("Next free:") == 1 and "test 158 checks the transition ledger, item 132" in cm
    for n in range(172, 179): assert f"\n| {n} |" in cm, n
    j = cm.index("\n132. **") + 1; item = cm[j:cm.index("\n", j)]
    assert "ledger.py" in item and "HISTORY_LEDGER.md" in item and "test 158" in item and "Rajnish Choubey" in item and "RFC-001" in item
    row = next(l for l in vf.split("\n") if "HISTORY_LEDGER.md" in l); assert "test_ledger.py" in row and "no model" in row.lower()


if __name__ == "__main__":
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]; bad = 0
    for n, f in fns:
        try: f(); print("ok  ", n)
        except Exception: bad += 1; print("FAIL", n); traceback.print_exc()
    print(f"{len(fns) - bad}/{len(fns)} passed"); sys.exit(1 if bad else 0)
