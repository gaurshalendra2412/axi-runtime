"""Implements concept-map item 127 (one word, one reading: the host reads every word of a question once; python/axi/engine/answer.py,
`read`; docs/ONE_READING.md), test 153. Prompted by the text on Devanagari and the Latin alphabet from Rajnish Choubey's conversation
(one sound, one symbol; "their / there / they're", "read / read").
It checks, with no model:
  1. `read` and `answer` on made-up states: a name that holds a field word ("Day Spa"), a name that holds an owner's name ("Priya Hall"),
     an item named like an owner ("Priya", read two ways: handed to the model), a name inside a longer name, every occurrence closed;
  2. NO CHANGE on Test 23: its 39 questions get the same answer as with Update 86's separate reading, 39 of 39 exact, and the 32 recall
     questions reworded without the field word ("how much is ...?") are all handed to the model, as before;
  3. the same questions with names that hold field words and owners' names (written for this check: the first word of every name
     replaced by one of Day, Status, Amount, Owner or an owner's first name): the separate reading answered 6 of 39 and gave 13 wrong
     answers to the reworded questions; one reading answers 39 of 39 exactly and gives no wrong answer;
  4. the host's reading stays far under a millisecond a question;
  5. the page and the front pages (concept map item 127, VERIFIED row, the counter).
What it does NOT show: how real names and questions are worded; the renamed stream is ours."""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.answer import answer, read
from axi.engine.focus import mentioned
from axi.engine.witnesses import resolve
from axi.experiments import gemini_fuzzy as F

HERE = os.path.dirname(__file__)
DOCS = os.path.join(HERE, "..", "..", "docs")
PAGE = os.path.join(DOCS, "ONE_READING.md")
NEXT = "Next free: row 172, item 132, test 158, question 121"
D = {"owner": F.OWNERS}
COLLIDE = ["Day", "Status", "Amount", "Owner", "Priya", "Marcus", "Elena", "Tomas", "Aiko", "Dmitri", "Fatima", "Lucas"]
RENAME = {f"{a} {n}": f"{COLLIDE[i]} {n}" for i, a in enumerate(F._ADJ) for n in F._NOUN}
REWORD = {"amount": "how much is {n}?", "owner": "who is handling {n}?", "day": "when is {n}?", "status": "where does {n} stand?"}


def separate(state, question, fields, describers=None):
    """Update 86's reading, kept here to compare: field words, names and describing values each read on their own."""
    fs = mentioned(question, list(fields))
    if len(fs) != 1: return None, "field not placed"
    f = fs[0]; names = mentioned(question, list(state)); desc, clash = {}, False
    for df, values in (describers or {}).items():
        hits = mentioned(question, [str(v) for v in values])
        if len(hits) > 1: clash = True
        elif hits: desc[df] = hits[0]
    if clash: return None, "two values for one description"
    if len(names) == 1 and not desc:
        facts = state[names[0]]; return (str(facts[f]) if f in facts else "unknown"), "named"
    if not names and desc:
        status, items = resolve({n: {k: str(v) for k, v in x.items()} for n, x in state.items()}, desc)
        if status == "bound": facts = state[items[0]]; return (str(facts[f]) if f in facts else "unknown"), "described"
        if status == "ambiguous": return "ambiguous", "described, fits several"
        return None, "description fits nothing"
    return None, "item not placed"


def test_one_reading_on_made_up_states():
    st = {"Day Spa": {"owner": "Priya", "day": "Monday", "amount": 400}, "Priya Hall": {"owner": "Marcus", "day": "Friday"}, "Amber": {"owner": "Elena", "day": "Sunday"},
          "Amber Gallery": {"owner": "Hiro", "day": "Tuesday"}}
    assert separate(st, "how much is Day Spa?", F.FIELDS, D) == ("Monday", "named")                 # Update 86: 'Day' read as the name AND the field
    assert answer(st, "how much is Day Spa?", F.FIELDS, D) == (None, "field not placed")
    assert separate(st, "who is handling Day Spa?", F.FIELDS, D) == ("Monday", "named") and answer(st, "who is handling Day Spa?", F.FIELDS, D)[0] is None
    assert separate(st, "what is the amount of Day Spa?", F.FIELDS, D)[0] is None and answer(st, "what is the amount of Day Spa?", F.FIELDS, D) == ("400", "named")
    assert answer(st, "what is the owner of Day Spa?", F.FIELDS, D) == ("Priya", "named")
    assert separate(st, "what is the day of Priya Hall?", F.FIELDS, D)[0] is None and answer(st, "what is the day of Priya Hall?", F.FIELDS, D) == ("Friday", "named")
    assert answer(st, "what is the day of the item Priya is handling?", F.FIELDS, D) == ("Monday", "described")
    assert answer(st, "what is the day of Amber Gallery?", F.FIELDS, D) == ("Tuesday", "named") and answer(st, "what is the day of Amber?", F.FIELDS, D) == ("Sunday", "named")
    two = {"Priya": {"owner": "Marcus", "day": "Monday"}, "Birch Studio": {"owner": "Priya", "day": "Friday"}}
    assert answer(two, "what is the day of Priya?", F.FIELDS, D) == (None, "one word, two readings")   # the item, or the item Priya handles?
    found, double = read("the day of Day Spa, then Day Spa again", {"f": ["day"], "i": ["Day Spa"]})
    assert found == {"f": ["day"], "i": ["Day Spa"]} and double == []                                # 'day' stands alone once; both 'Day Spa' are closed
    found, double = read("Day Spa and Day Spa", {"f": ["day"], "i": ["Day Spa"]}); assert found == {"f": [], "i": ["Day Spa"]}
    found, double = read("what of amber gallery", {"i": ["Amber Gallery", "amber gallery"]}); assert double == ["amber gallery"]


def questions():
    s = F.make_stream(F.SEED); out = []
    for _ in range(400):
        st, ob = s.next()
        if st["kind"] == "query": out.append((st, ob))
    return out


def rename(text):
    for a, b in sorted(RENAME.items(), key=lambda x: -len(x[0])): text = text.replace(a, b)
    return text


def score(fn, collided):
    tpl = dict(given=0, exact=0, wrong=0); rew = dict(given=0, exact=0, wrong=0, handed=0)
    for st, ob in questions():
        o = {rename(k): v for k, v in ob.items()} if collided else ob; q = rename(st["message"]) if collided else st["message"]
        a, _ = fn(o, q, F.FIELDS, D)
        if a is not None: ok = F.norm_answer(a) == F.norm_answer(st["answer"]); tpl["given"] += 1; tpl["exact"] += ok; tpl["wrong"] += not ok
        if st["qkind"] == "recall":
            n = next(k for k in ob if k in st["message"]); f = next(f for f in F.FIELDS if f" {f} of" in st["message"])
            a, _ = fn(o, REWORD[f].format(n=rename(n) if collided else n), F.FIELDS, D)
            if a is None: rew["handed"] += 1
            else: ok = F.norm_answer(a) == F.norm_answer(st["answer"]); rew["given"] += 1; rew["exact"] += ok; rew["wrong"] += not ok
    return tpl, rew


def test_no_change_on_test_23():
    for st, ob in questions(): assert answer(ob, st["message"], F.FIELDS, D) == separate(ob, st["message"], F.FIELDS, D)
    assert score(answer, False) == score(separate, False) == (dict(given=39, exact=39, wrong=0), dict(given=0, exact=0, wrong=0, handed=32))


def test_names_that_hold_field_words_and_owners_names():
    assert all(any(w.lower() in n.lower().split() for w in COLLIDE) for n in RENAME.values()) and len(set(RENAME.values())) == 144
    assert score(separate, True) == (dict(given=6, exact=6, wrong=0), dict(given=17, exact=4, wrong=13, handed=15))
    assert score(answer, True) == (dict(given=39, exact=39, wrong=0), dict(given=0, exact=0, wrong=0, handed=32))


def test_the_reading_is_fast():
    qs = questions(); t0 = time.perf_counter()
    for st, ob in qs: answer(ob, st["message"], F.FIELDS, D)
    assert (time.perf_counter() - t0) / len(qs) < 0.001


def test_the_page_and_the_front_pages():
    page = open(PAGE).read(); low = page.lower()
    for s in ("answer.py", "read", "Rajnish Choubey", "Devanagari", "one sound", "their / there / they're", "read / read", "Day Spa", "Monday", "39 of 39",
              "6 of 39", "13", "32", "written for this check", "no change", "test 153", "item 126", "two readings"):
        assert s.lower() in low, s
    for bad in ("proven", "$", "dollar", "100 percent"): assert bad not in low, bad
    ha = open(os.path.join(DOCS, "HOST_ANSWERS.md")).read(); assert "ONE_READING.md" in ha and "item 127" in ha
    cm = open(os.path.join(DOCS, "RAJNISH_CONCEPT_MAP.md")).read(); vf = open(os.path.join(DOCS, "VERIFIED.md")).read()
    assert NEXT in cm and cm.count("Next free:") == 1 and "test 153 checks one word, one reading, item 127" in cm
    j = cm.index("\n127. **") + 1; item = cm[j:cm.index("\n", j)]
    assert "answer.py" in item and "ONE_READING.md" in item and "test 153" in item and "Rajnish Choubey" in item
    row = next(l for l in vf.split("\n") if "ONE_READING.md" in l); assert "test_one_reading.py" in row and "no model" in row.lower()


if __name__ == "__main__":
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]; bad = 0
    for n, f in fns:
        try: f(); print("ok  ", n)
        except Exception: bad += 1; print("FAIL", n); traceback.print_exc()
    print(f"{len(fns) - bad}/{len(fns)} passed"); sys.exit(1 if bad else 0)
