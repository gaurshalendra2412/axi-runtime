"""Implements concept-map item 126 (R = 0 for questions: the host answers a question about the state it holds; python/axi/engine/answer.py;
docs/HOST_ANSWERS.md), test 152. Prompted by Rajnish Choubey's open letter "The Failure of Thought Imposition" (the Spherical Mind).
It checks, with no model:
  1. `answer` on made-up states: a held fact, a fact never given ("unknown"), a description that fits one item and one that fits two
     ("ambiguous"), and every case it hands to the model (no field, two fields, two items, a description that fits nothing, two values);
  2. offline over Test 23's 39 questions (seed 20261122, no call): the host answers every one, exactly; the share of arm A's computed
     tokens and seconds those questions took, which a host answer saves;
  3. the page, the notes added to the readings page and the 1 % view page, the front pages and the counter.
What it does NOT show: how real questions are worded; Test 23's questions follow the stream's templates."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.answer import answer
from axi.experiments import gemini_fuzzy as F
from axi.experiments import gemini_long as GL

HERE = os.path.dirname(__file__)
DOCS = os.path.join(HERE, "..", "..", "docs")
PAGE = os.path.join(DOCS, "HOST_ANSWERS.md")
NEXT = "Next free: row 186, item 140, test 165, question 122"
DESC = {"owner": F.OWNERS}


def test_answer_on_made_up_states():
    st = {"Amber Gallery": {"owner": "Priya", "day": "Monday", "amount": 400}, "Birch Studio": {"owner": "Priya"}, "Cobalt Lounge": {"owner": "Marcus", "status": "pending"}}
    assert answer(st, "what is the day of Amber Gallery?", F.FIELDS, DESC) == ("Monday", "named")
    assert answer(st, "what is the amount of Amber Gallery?", F.FIELDS, DESC) == ("400", "named")
    assert answer(st, "what is the day of Birch Studio? It has not been given.", F.FIELDS, DESC) == ("unknown", "named")
    assert answer(st, "what is the status of the item Marcus is handling?", F.FIELDS, DESC) == ("pending", "described")
    assert answer(st, "what is the day of the item Priya is handling?", F.FIELDS, DESC) == ("ambiguous", "described, fits several")
    assert answer(st, "what is the day of the item Elena is handling?", F.FIELDS, DESC) == (None, "description fits nothing")
    assert answer(st, "tell me about Amber Gallery", F.FIELDS, DESC) == (None, "field not placed")
    assert answer(st, "what is the day and the status of Amber Gallery?", F.FIELDS, DESC) == (None, "field not placed")
    assert answer(st, "is the day of Amber Gallery the same as Birch Studio?", F.FIELDS, DESC) == (None, "item not placed")
    assert answer(st, "what is the day of the item Priya or Marcus is handling?", F.FIELDS, DESC) == (None, "two values for one description")
    assert answer(st, "what is the amount of Amber Gallery on Monday?", F.FIELDS, DESC) == ("400", "named")         # 'Monday' is not the word 'day'


def offline(arm):
    d = json.load(open(os.path.join(DOCS, "results", "axi_gemini_fuzzy_seed20261122_default.json"))); rows = d[arm]["rows"]
    s = F.make_stream(F.SEED); out = dict(q=0, host=0, exact=0, named=0, ambiguous=0, unknown=0, model_exact=0, q_comp=0, q_sec=0.0)
    for r in rows:
        st, ob = s.next()
        if st["kind"] != "query": continue
        a, how = answer(ob, st["message"], F.FIELDS, DESC); out["q"] += 1
        if a is not None:
            out["host"] += 1; out["exact"] += F.norm_answer(a) == F.norm_answer(st["answer"]); out["named"] += how == "named"; out["ambiguous"] += a == "ambiguous"; out["unknown"] += a == "unknown"
        out["model_exact"] += r["reply_exact"]; out["q_comp"] += GL.computed(r); out["q_sec"] += r["seconds"]
    out["comp"] = sum(GL.computed(r) for r in rows); out["sec"] = sum(r["seconds"] for r in rows)
    return out


def test_offline_on_test_23_the_host_answers_every_question_exactly():
    a = offline("A"); k = offline("K")
    assert (a["q"], a["host"], a["exact"], a["named"], a["ambiguous"], a["unknown"]) == (39, 39, 39, 37, 2, 5)
    assert a["model_exact"] == 39 and k["model_exact"] == 38 and k["exact"] == 39
    assert (a["q_comp"], a["comp"]) == (48858, 528836) and round(100 * a["q_comp"] / a["comp"], 1) == 9.2
    assert (round(a["q_sec"]), round(a["sec"])) == (85, 1103) and round(100 * a["q_sec"] / a["sec"], 1) == 7.7


def test_the_pages_and_the_front_pages():
    page = open(PAGE).read(); low = page.lower()
    for s in ("answer.py", "Rajnish Choubey", "The Failure of Thought Imposition", "Spherical Mind", "the question is the answer", "silence", "39 of 39", "37", "unknown",
              "ambiguous", "9.2 %", "7.7 %", "never invents", "templates", "test 152", "plain lookup"):
        assert s.lower() in low, s
    for bad in ("proven", "$", "dollar", "100 percent"): assert bad not in low, bad
    sup = open(os.path.join(DOCS, "SUPERPOSITION.md")).read(); assert "Fund Paradox" in sup and "hold contradictory truths simultaneously" in sup and "The Failure of Thought Imposition" in sup
    fv = open(os.path.join(DOCS, "FOCUSED_VIEW.md")).read(); assert "Widen on refusal" in fv and "Spherical Mind" in fv and "not measured" in fv.lower()
    cm = open(os.path.join(DOCS, "RAJNISH_CONCEPT_MAP.md")).read(); vf = open(os.path.join(DOCS, "VERIFIED.md")).read()
    assert NEXT in cm and cm.count("Next free:") == 1 and "test 152 checks the host's answers, item 126" in cm
    j = cm.index("\n126. **") + 1; item = cm[j:cm.index("\n", j)]
    assert "answer.py" in item and "HOST_ANSWERS.md" in item and "test 152" in item and "Rajnish Choubey" in item
    row = next(l for l in vf.split("\n") if "HOST_ANSWERS.md" in l); assert "test_answer.py" in row and "no model" in row.lower()


if __name__ == "__main__":
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]; bad = 0
    for n, f in fns:
        try: f(); print("ok  ", n)
        except Exception: bad += 1; print("FAIL", n); traceback.print_exc()
    print(f"{len(fns) - bad}/{len(fns)} passed"); sys.exit(1 if bad else 0)
