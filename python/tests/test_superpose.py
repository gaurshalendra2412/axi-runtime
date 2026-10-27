"""Implements concept-map item 125 (hold every reading, choose none; python/axi/engine/superpose.py, python/axi/experiments/ambiguity_desk.py;
docs/SUPERPOSITION.md), test 151. Prompted by Rajnish Choubey's essay "The Wave, the Unicorn, and the End of Western Psychology".
It checks, with no model:
  1. the readings on made-up states: a named change in every world, a branch with the applied-nowhere reading, a clarification, an
     expiry back to today's rule, an observation, a question answered only when every reading agrees, the cap, worlds that become equal merge;
  2. NO HARM where there is no ambiguity: Test 23's saved replies run through the readings give the same list as the plain path at every
     one of the 400 steps, one world throughout, and the host's time stays far under a millisecond a step;
  3. on the ambiguity stream (5 seeds x 400 steps, written for this comparison): the three rules, today's (apply nothing), guessing the
     first item, and holding every reading; how often each is worse or better than today's rule;
  4. the page and the front pages (concept map item 125, VERIFIED row, the counter).
What it does NOT show: how often real requests are ambiguous, or that a model writes a description instead of guessing a name."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.superpose import Readings
from axi.experiments import ambiguity_desk as A
from axi.experiments import gemini_fuzzy as F

HERE = os.path.dirname(__file__)
DOCS = os.path.join(HERE, "..", "..", "docs")
PAGE = os.path.join(DOCS, "SUPERPOSITION.md")
NEXT = "Next free: row 191, item 144, test 169, question 122"


def put(state, n, f, v):
    if n not in state: return None
    new = {k: dict(x) for k, x in state.items()}; new[n][f] = v; return new


def test_the_readings_on_made_up_states():
    s0 = {"A": {"owner": "Priya", "day": "Monday"}, "B": {"owner": "Priya", "day": "Tuesday"}, "C": {"owner": "Hiro", "day": "Monday"}}
    R = Readings(s0, cap=9); assert R.settled and R.state() == s0
    assert R.apply(lambda s: put(s, "C", "day", "Friday")) == "applied" and R.state()["C"]["day"] == "Friday"
    assert R.apply(lambda s: put(s, "Z", "day", "Friday")) == "refused" and R.settled
    fits = lambda s: [(n, put(s, n, "day", "Sunday")) for n in sorted(s) if s[n]["owner"] == "Priya"]
    assert R.branch(5, fits) == "branched" and len(R.worlds) == 3 and not R.settled and R.state() is None          # A, B, and applied nowhere
    assert R.ask(lambda s: s["C"]["day"]) == ("Friday", True) and R.ask(lambda s: s["A"]["day"]) == (None, False)    # agreed facts still answer
    assert R.clarify(5, "B") == "settled" and R.state()["B"]["day"] == "Sunday" and R.state()["A"]["day"] == "Monday" and 5 not in R.pending
    assert R.clarify(5, "A") == "contradicts" and R.contradictions == 1 and R.settled
    R.branch(9, fits); assert R.expire(16, 8) == 0 and R.expire(17, 8) == 1 and R.settled and R.state()["A"]["day"] == "Monday" and R.state()["B"]["day"] == "Sunday"
    R.branch(20, fits); assert R.observe(lambda s: s["A"]["day"] == "Sunday") == "settled" and R.state()["A"]["day"] == "Sunday"
    many = {f"I{i}": {"owner": "Priya", "day": "Monday"} for i in range(12)}; M = Readings(many, cap=9)
    assert M.branch(1, lambda s: [(n, put(s, n, "day", "Friday")) for n in sorted(s)]) == "over cap" and M.settled and M.over_cap == 1 and M.state() == many
    same = Readings(s0, cap=9); same.branch(3, lambda s: [(n, put(s, n, "owner", "Priya")) for n in ("A", "B")])
    assert same.settled and same.state() == s0                                                    # readings that leave the same list are one world
    assert same.clarify(3, "B") == "settled" and same.clarify(3, "A") == "contradicts"            # its paths are kept: B was one of them


def test_no_harm_where_there_is_no_ambiguity_test_23_runs_exactly_as_before():
    d = json.load(open(os.path.join(DOCS, "results", "axi_gemini_fuzzy_seed20261122_default.json"))); rows = d["A"]["rows"]
    s = F.make_stream(F.SEED); live = {}; plain = []; t0 = time.perf_counter()
    for r in rows:
        st, ob = s.next(); ok, _ = F.usable(live, st, r["text"])
        if ok and st["kind"] != "query": live = F.apply_ops(live, F.check_ops(live, F.parse_reply(r["text"])["ops"])[2])
        plain.append(live)
    t1 = time.perf_counter(); s = F.make_stream(F.SEED); R = Readings({}); held = []; worlds = []
    for r in rows:
        st, ob = s.next()
        def ch(w, r=r, st=st):
            ok, _ = F.usable(w, st, r["text"]); return F.apply_ops(w, F.check_ops(w, F.parse_reply(r["text"])["ops"])[2]) if ok else w
        if st["kind"] != "query": R.apply(ch)
        held.append(R.state()); worlds.append(len(R.worlds))
    t2 = time.perf_counter()
    assert sum(a == b for a, b in zip(plain, held)) == 400 and max(worlds) == 1 and R.max_seen == 1
    assert (t2 - t1) / 400 < 0.002 and (t2 - t1) / 400 < 2.757 / 1000             # far under a millisecond a step; a model call took 2.76 s


def test_on_the_ambiguity_stream_holding_every_reading_against_today_and_guessing():
    c = A.compare(); N, G, H = c["NOTHING"], c["GUESS"], c["HOLD"]
    assert (N["steps"], N["ambiguous"], N["clarified"], N["questions"]) == (2000, 94, 76, 517) and G["ambiguous"] == H["ambiguous"] == 94
    assert (N["exact"], G["exact"], H["exact"]) == (314, 350, 827) and H["truth_held"] == 995
    assert (N["answers_exact"], N["answers_wrong"], N["not_settled"]) == (490, 27, 0) and (G["answers_exact"], G["answers_wrong"]) == (482, 35)
    assert (H["answers_exact"], H["answers_wrong"], H["not_settled"]) == (503, 6, 8) and H["max_worlds"] == 9 and H["over_cap"] == 10
    worse = better = expired = 0
    for seed in A.SEEDS:
        st = A.make_stream(seed); n, h = A.run(st, "NOTHING"), A.run(st, "HOLD"); expired += h[-1]["expired"]
        worse += sum(1 for a, b in zip(n, h) if a["exact"] and not b["exact"]); better += sum(1 for a, b in zip(n, h) if b["exact"] and not a["exact"])
    assert (worse, better, expired) == (6, 519, 18)
    assert A.CAP == 9 and A.WINDOW == 8 and A.WEIGHTS == (65, 5, 5, 25)


def test_the_page_and_the_front_pages_say_what_was_built_and_the_counter_moved():
    page = open(PAGE).read(); low = page.lower()
    for s in ("superpose.py", "ambiguity_desk.py", "Rajnish Choubey", "The Wave, the Unicorn", "How can one select one", "400 of 400", "one world", "0.14", "2.76",
              "314", "350", "827", "490", "482", "503", "27", "35", "6", "8", "519", "18", "written for this comparison", "test 151", "standard"):
        assert s.lower() in low, s
    for bad in ("proven", "$", "dollar", "100 percent"): assert bad not in low, bad
    cm = open(os.path.join(DOCS, "RAJNISH_CONCEPT_MAP.md")).read(); vf = open(os.path.join(DOCS, "VERIFIED.md")).read()
    assert NEXT in cm and cm.count("Next free:") == 1 and "test 151 checks holding every reading, item 125" in cm
    j = cm.index("\n125. **") + 1; item = cm[j:cm.index("\n", j)]
    assert "superpose.py" in item and "SUPERPOSITION.md" in item and "test 151" in item and "Rajnish Choubey" in item
    row = next(l for l in vf.split("\n") if "SUPERPOSITION.md" in l); assert "test_superpose.py" in row and "no model" in row.lower()


if __name__ == "__main__":
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]; bad = 0
    for n, f in fns:
        try: f(); print("ok  ", n)
        except Exception: bad += 1; print("FAIL", n); traceback.print_exc()
    print(f"{len(fns) - bad}/{len(fns)} passed"); sys.exit(1 if bad else 0)
