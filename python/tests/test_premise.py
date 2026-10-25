"""Implements concept-map item 128 (the premise check: what a message takes for granted is checked against the exact state;
python/axi/engine/premise.py; docs/PREMISE_CHECK.md), test 154. Prompted by Rajnish Choubey's article "The Cinderella Project: Testing AI
Understanding of Modern Visual Communication" (the AI in its chats took each new premise, then reversed, agreeing with whatever came last).
It checks, with no model:
  1. `premises` and `note` on made-up states: a false premise in a question, a true one, a value marked as the old one ("not Monday"),
     a fact not on the list, a sentence that names two items or none (ties nothing), a value inside a name (one word, one reading);
  2. NO FALSE ALARM on Test 23: all 400 steps, each message with its 30 sentences of office chatter, against the list before the step
     and the list after arm A's reply was committed: no report, while 214 values in the messages were tied to an item and checked;
  3. on false premises written for this check from Test 23's stream: 29 questions ("Since Dune Gallery is on Wednesday, what is the
     amount of Dune Gallery?") and 100 change messages ("Dune Gallery is on Wednesday, so Hand Dune Gallery to Marcus."), each also with
     the true value: every false premise reported, no true one; the host still answers 21 of the 29 questions exactly; a reply that writes
     the false premise into the list as well is NOT reported (what the check cannot see);
  4. the check takes about a millisecond a step with the chatter;
  5. the page and the front pages (concept map item 128, VERIFIED row, the counter).
What it does NOT show: how real messages are worded; the false premises are ours."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.premise import premises, note, sentences
from axi.engine.answer import answer
from axi.experiments import gemini_fuzzy as F

HERE = os.path.dirname(__file__)
DOCS = os.path.join(HERE, "..", "..", "docs")
PAGE = os.path.join(DOCS, "PREMISE_CHECK.md")
NEXT = "Next free: row 179, item 133, test 159, question 122"
V = {"owner": F.OWNERS, "day": F.DAYS, "amount": F.AMOUNTS, "status": F.STATUSES}
PH = {"day": "{n} is on {v}", "amount": "{n} costs {v}", "status": "{n} is {v}", "owner": "{v} is handling {n}"}


def wrong(f, v): dom = V[f]; return dom[(dom.index(v) + 1) % len(dom)]


def test_the_premise_check_on_made_up_states():
    st = {"Amber Gallery": {"owner": "Priya", "day": "Monday", "amount": 400}, "Day Spa": {"owner": "Marcus", "day": "Friday"}}
    r = premises(st, st, "Since Amber Gallery is on Friday, what is its amount?", V)
    assert [(x["item"], x["field"], x["said"], x["held"], x["against"]) for x in r] == [("Amber Gallery", "day", "Friday", "Monday", "after")]
    assert note(r) == "Note: you said Friday for the day of Amber Gallery; the list has Monday."
    assert premises(st, st, "Since Amber Gallery is on Monday, what is its amount?", V) == [] and note([]) is None
    after = {"Amber Gallery": {"owner": "Priya", "day": "Sunday", "amount": 400}, "Day Spa": {"owner": "Marcus", "day": "Friday"}}
    assert premises(st, after, "Scratch what I said about Amber Gallery: the day is Sunday, not Monday.", V) == []          # 'not Monday' is the old value
    assert premises(st, after, "Amber Gallery moves to Sunday, not Tuesday.", V)[0]["against"] == "before"
    assert premises(st, st, "Amber Gallery is waiting on payment, so what is its day?", V)[0]["held"] is None              # a fact not on the list
    assert premises(st, st, "Amber Gallery and Day Spa are both on Sunday.", V) == []                                    # two items: nothing is tied
    assert premises(st, st, "The courier said it would be Sunday. What is the day of Amber Gallery?", V) == []          # chatter ties nothing
    assert premises(st, st, "What is the day of Day Spa?", V) == []                                                      # 'Day' is the name, not a value
    assert premises(st, st, "Since Day Spa is on Monday, is it confirmed?", V)[0]["held"] == "Friday"
    assert sentences("One. Two? Three") == [(0, 5), (5, 10), (10, 15)]


def walk():
    d = json.load(open(os.path.join(DOCS, "results", "axi_gemini_fuzzy_seed20261122_default.json"))); s = F.make_stream(F.SEED); live = {}
    for r in d["A"]["rows"]:
        st, ob = s.next(); assert ob == live; ok, _ = F.usable(live, st, r["text"])
        after = live if st["kind"] == "query" or not ok else F.apply_ops(live, F.check_ops(live, F.parse_reply(r["text"])["ops"])[2])
        yield st, live, after; live = after


def test_no_false_alarm_on_test_23_with_the_chatter():
    reports = ties = 0
    from axi.engine.answer import spans
    for st, before, after in walk():
        reports += len(premises(before, after, F.user_text(F.SEED, st["t"], st), V)) + len(premises(before, after, st["message"], V))
        kinds = {("item",): sorted(set(before) | set(after))}; kinds.update({("value", f): [str(x) for x in v] for f, v in V.items()}); sp = spans(st["message"], kinds)
        for a, b in sentences(st["message"]):
            ins = [x for x in sp if a <= x[0] < b]
            if len({x[3] for x in ins if x[2] == ("item",)}) == 1: ties += sum(1 for x in ins if x[2] and x[2][0] == "value")
    assert (reports, ties) == (0, 214)


def test_false_premises_written_for_this_check():
    s = F.make_stream(F.SEED); q = dict(n=0, false=0, true=0, answered=0, exact=0); c = dict(n=0, false=0, true=0, written=0)
    for _ in range(400):
        st, ob = s.next()
        if st["kind"] == "query" and st["qkind"] == "recall":
            n = next(k for k in ob if k in st["message"]); f = next(x for x in F.FIELDS if f" {x} of" in st["message"]); others = [x for x in F.FIELDS if x != f and x in ob[n]]
            if not others: continue
            f2 = others[q["n"] % len(others)]; q["n"] += 1
            fq = f"Since {PH[f2].format(n=n, v=wrong(f2, ob[n][f2]))}, what is the {f} of {n}?"; tq = f"Since {PH[f2].format(n=n, v=ob[n][f2])}, what is the {f} of {n}?"
            q["false"] += bool(premises(ob, ob, fq, V)); q["true"] += bool(premises(ob, ob, tq, V))
            a, _ = answer(ob, fq, F.FIELDS, {"owner": F.OWNERS})
            if a is not None: q["answered"] += 1; q["exact"] += F.norm_answer(a) == F.norm_answer(st["answer"])
        if st["kind"] == "set":
            o = st["ops"][0]; n = o["name"]; others = [x for x in F.FIELDS if x != o["field"] and x in ob[n]]
            if not others: continue
            f2 = others[c["n"] % len(others)]; v = wrong(f2, ob[n][f2]); c["n"] += 1
            asked = F.apply_ops(ob, F.check_ops(ob, st["ops"])[2]); taken = F.apply_ops(ob, F.check_ops(ob, st["ops"] + [dict(op="set", name=n, field=f2, value=v)])[2])
            c["false"] += bool(premises(ob, asked, f"{PH[f2].format(n=n, v=v)}, so {st['message']}", V))
            c["true"] += bool(premises(ob, asked, f"{PH[f2].format(n=n, v=ob[n][f2])}, so {st['message']}", V))
            c["written"] += bool(premises(ob, taken, f"{PH[f2].format(n=n, v=v)}, so {st['message']}", V))
    assert q == dict(n=29, false=29, true=0, answered=21, exact=21)
    assert c == dict(n=100, false=100, true=0, written=0)                     # 'written': the reply also wrote the false premise; the check cannot see it


def test_the_check_is_fast():
    steps = list(walk()); t0 = time.perf_counter()
    for st, before, after in steps: premises(before, after, F.user_text(F.SEED, st["t"], st), V)
    t1 = time.perf_counter()
    for st, before, after in steps: premises(before, after, st["message"], V)
    t2 = time.perf_counter()
    assert (t1 - t0) / 400 < 0.01 and (t2 - t1) / 400 < 0.002                # a model call took 2.76 s a step in that run


def test_the_page_and_the_front_pages():
    page = open(PAGE).read(); low = page.lower()
    for s in ("premise.py", "Rajnish Choubey", "The Cinderella Project", "whatever framing came last", "valid", "true", "not Monday", "400", "214", "0 reports",
              "29 of 29", "100 of 100", "21 of 29", "written for this check", "cannot see", "1.5 ms", "0.1 ms", "test 154", "item 127"):
        assert s.lower() in low, s
    for bad in ("proven", "$", "dollar", "100 percent"): assert bad not in low, bad
    cm = open(os.path.join(DOCS, "RAJNISH_CONCEPT_MAP.md")).read(); vf = open(os.path.join(DOCS, "VERIFIED.md")).read()
    assert NEXT in cm and cm.count("Next free:") == 1 and "test 154 checks the premise check, item 128" in cm
    j = cm.index("\n128. **") + 1; item = cm[j:cm.index("\n", j)]
    assert "premise.py" in item and "PREMISE_CHECK.md" in item and "test 154" in item and "Rajnish Choubey" in item
    row = next(l for l in vf.split("\n") if "PREMISE_CHECK.md" in l); assert "test_premise.py" in row and "no model" in row.lower()


if __name__ == "__main__":
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]; bad = 0
    for n, f in fns:
        try: f(); print("ok  ", n)
        except Exception: bad += 1; print("FAIL", n); traceback.print_exc()
    print(f"{len(fns) - bad}/{len(fns)} passed"); sys.exit(1 if bad else 0)
