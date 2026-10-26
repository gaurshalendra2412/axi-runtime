"""Implements concept-map item 129 (receive without adding: the host applies a plain change itself, checked by the gate, with no model call;
python/axi/engine/receive.py; docs/RECEIVE_WITHOUT_ADDING.md), test 155. Prompted by Rajnish Choubey's article "a durga diary: Why do we
park in driveways and drive on parkways?" ("the addition is the failure"; a tool that "cannot receive without immediately responding").
It checks, with no model:
  1. `receive` on made-up states: a plain change, an old value marked "not" or "from", a removal, a role marked "as", a number; handed
     to the model: a question, a reason, an aside, a hedge, a negation, two items, no held item, a new item, a value the list already
     has, two values for one field, a value with a removal, two sentences naming items, a word with two readings; the gate's refusal;
  2. offline on Test 23 (seed 20261122): of its 361 changes, 199 applied by the host and all 199 exact, alone and inside the 30 sentences
     of chatter; the other 162 handed to the model for the reasons counted; the seconds and computed tokens arm A spent on those 199;
  3. 20 more seeds of the same stream: 4,212 of 7,220 changes applied, none wrong; with names that hold field words and owners' names
     (item 127's renaming): 199 applied, none wrong;
  4. on messages written for this check from Test 23's set steps: a reason, a second sentence, a question, a hedge, a negation, an
     aside, a value the list already has: none applied; a statement joined to the change by a plain "and": all applied, the stated
     value written too (what it cannot see);
  5. it takes under 2 ms a step with the chatter;
  6. the page and the front pages (concept map item 129, VERIFIED row, the counter).
What it does NOT show: how real messages are worded; the measured messages follow Test 23's templates and ours."""
import collections, json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.receive import receive
from axi.experiments import gemini_fuzzy as F
from axi.experiments import gemini_long as GL

HERE = os.path.dirname(__file__)
DOCS = os.path.join(HERE, "..", "..", "docs")
PAGE = os.path.join(DOCS, "RECEIVE_WITHOUT_ADDING.md")
NEXT = "Next free: row 186, item 140, test 165, question 122"
V = {"owner": F.OWNERS, "day": F.DAYS, "amount": F.AMOUNTS, "status": F.STATUSES}
PH = {"day": "{n} is on {v}", "amount": "{n} costs {v}", "status": "{n} is {v}", "owner": "{v} is handling {n}"}
COLLIDE = ["Day", "Status", "Amount", "Owner", "Priya", "Marcus", "Elena", "Tomas", "Aiko", "Dmitri", "Fatima", "Lucas"]
RENAME = {f"{a} {n}": f"{COLLIDE[i]} {n}" for i, a in enumerate(F._ADJ) for n in F._NOUN}


def rename(text):
    for a, b in sorted(RENAME.items(), key=lambda x: -len(x[0])): text = text.replace(a, b)
    return text


def wrong(f, v): dom = V[f]; return dom[(dom.index(v) + 1) % len(dom)]
def lower1(m): return m[:1].lower() + m[1:]


def test_receive_on_made_up_states():
    st = {"Amber Gallery": {"owner": "Elena", "day": "Monday", "amount": 300}, "Birch Studio": {"owner": "Marcus", "status": "pending"}}
    ok = lambda m: receive(st, m, V, F.check_ops)
    assert ok("Push Amber Gallery to Friday.") == ([dict(op="set", name="Amber Gallery", field="day", value="Friday")], "read and checked")
    assert ok("Hand Amber Gallery to Marcus, not Elena.")[0] == [dict(op="set", name="Amber Gallery", field="owner", value="Marcus")]
    assert ok("Move Amber Gallery from Monday to Tuesday.")[0] == [dict(op="set", name="Amber Gallery", field="day", value="Tuesday")]
    assert ok("Drop Birch Studio from the list.")[0] == [dict(op="delete", name="Birch Studio")]
    assert ok("Mark Birch Studio as confirmed.")[0] == [dict(op="set", name="Birch Studio", field="status", value="confirmed")]   # "as" marks a role
    assert ok("Amber Gallery now costs 450.")[0] == [dict(op="set", name="Amber Gallery", field="amount", value=450)]
    assert ok("The coffee machine is broken again. Push Amber Gallery to Friday. Lunch is at one.")[0] == ok("Push Amber Gallery to Friday.")[0]
    handed = {"Is Amber Gallery on Friday?": "a question", "Amber Gallery is on Friday, so hand it to Marcus.": "a reason given: a premise, not a change",
              "Hand Amber Gallery (the one on Monday) to Marcus.": "an aside (which, who, brackets): a premise, not a change",
              "Hand Amber Gallery, which is on Tuesday, to Marcus.": "an aside (which, who, brackets): a premise, not a change",
              "Maybe push Amber Gallery to Friday.": "hedged", "Don't push Amber Gallery to Friday.": "negated",
              "Put Birch Studio on the same day as Amber Gallery.": "the sentence names two items", "Add Cobalt Lounge for Friday.": "no held item named",
              "The last one we added goes to Hiro.": "no held item named", "Push Amber Gallery to Monday.": "no change: the list already has it",
              "Push Amber Gallery to Friday or Tuesday.": "two values for one field", "Drop Amber Gallery and set it to Friday.": "a value and a removal",
              "Push Amber Gallery to Friday. Birch Studio goes to Elena.": "two sentences name held items", "Amber Gallery is fine.": "no value and no removal"}
    for m, why in handed.items(): assert ok(m) == (None, why), (m, ok(m))
    assert receive({"Priya Hall": {"owner": "Marcus"}}, "Hand Priya Hall to Elena.", V)[0] == [dict(op="set", name="Priya Hall", field="owner", value="Elena")]
    assert receive({"Priya": {"owner": "Marcus"}}, "Hand Priya to Elena.", V) == (None, "one word, two readings")      # an item named like an owner
    assert receive(st, "Push Amber Gallery to Friday.", V, lambda s, o: (False, "made-up refusal", [])) == (None, "the gate refused: made-up refusal")
    assert receive(st, "Push Amber Gallery to Friday.", V)[1] == "read"                                                    # no gate: read only
    assert st["Amber Gallery"]["day"] == "Monday"                                                                           # the state is not touched


def same(state, ops, ideal):
    _, _, a = F.check_ops(state, ideal)
    return F.apply_ops(state, ops) == F.apply_ops(state, a)


def run(seed, chatter=False, collided=False, rows=None):
    s = F.make_stream(seed); out = collections.Counter(); why = collections.Counter(); kinds = collections.Counter(); sec = 0.0; comp = 0
    for i in range(F.STEPS):
        st, ob = s.next()
        if st["kind"] == "query": continue
        o, msg, ideal = ob, st["message"], st["ops"]
        if collided: o = {rename(k): v for k, v in ob.items()}; msg = rename(msg); ideal = [dict(x, name=rename(x["name"])) for x in ideal]
        text = (rename(F.user_text(seed, st["t"], st)) if collided else F.user_text(seed, st["t"], st)) if chatter else msg
        ops, how = receive(o, text, V, F.check_ops); out["changes"] += 1
        if ops is None: why[how] += 1; out["handed"] += 1; continue
        out["applied"] += 1; kinds[st["kind"]] += 1
        if same(o, ops, ideal): out["exact"] += 1
        else: out["wrong"] += 1
        if rows: sec += rows[i]["seconds"]; comp += GL.computed(rows[i])
    return out, why, kinds, sec, comp


def saved_rows():
    return json.load(open(os.path.join(DOCS, "results", "axi_gemini_fuzzy_seed20261122_default.json")))["A"]["rows"]


def test_offline_on_test_23_the_host_applies_199_of_361_changes_and_all_are_exact():
    rows = saved_rows(); assert len(rows) == 400
    tot_s = sum(r["seconds"] for r in rows); tot_c = sum(GL.computed(r) for r in rows)
    assert (round(tot_s, 1), tot_c) == (1102.8, 528836)
    for chatter in (False, True):
        out, why, kinds, sec, comp = run(F.SEED, chatter, rows=rows)
        assert dict(out) == dict(changes=361, applied=199, exact=199, handed=162), out
        assert dict(kinds) == dict(set=102, delete=41, set2=32, correct=24)
        assert dict(why) == {"no held item named": 138, "the sentence names two items": 24}
        assert (round(sec, 1), comp) == (510.5, 261093)


def test_twenty_more_seeds_and_names_that_hold_field_words():
    tot = collections.Counter()
    for seed in range(1, 21): tot.update(run(seed)[0])
    assert (tot["changes"], tot["applied"], tot["exact"], tot["wrong"]) == (7220, 4212, 4212, 0)
    for chatter in (False, True):
        out = run(F.SEED, chatter, collided=True)[0]
        assert (out["applied"], out["exact"], out["wrong"]) == (199, 199, 0)


def test_on_messages_written_for_this_check_none_is_applied_except_a_plain_and():
    s = F.make_stream(F.SEED); applied = collections.Counter(); seen = collections.Counter(); written = 0; n_other = 0
    for _ in range(F.STEPS):
        st, ob = s.next()
        if st["kind"] != "set": continue
        o = st["ops"][0]; n = o["name"]; m = st["message"]; others = [x for x in F.FIELDS if x != o["field"] and x in ob[n]]
        tries = {"don't": "Don't " + lower1(m), "do not": "Do not " + lower1(m), "maybe": "Maybe " + lower1(m), "if": "If the client agrees, " + lower1(m),
                 "I think": "I think " + lower1(m), "question": m[:-1] + "?", "should": "Should we " + lower1(m)[:-1] + "?"}
        v0 = ob[n].get(o["field"])
        if v0 is not None: tries["already held"] = m.replace(str(o["value"]), str(v0))
        if others:
            f2 = others[n_other % len(others)]; v = wrong(f2, ob[n][f2]); n_other += 1; p = PH[f2].format(n=n, v=v)
            tries.update({"so": f"{p}, so {m}", "since": f"Since {p}, {lower1(m)}", "because": f"{m[:-1]}, because {p}.", "two sentences": f"{p}. {m}",
                          "which": f"{m[:-1]}, which means {p}.", "brackets": f"{m[:-1]} ({p}).", "and": f"{p} and {lower1(m)}"})
            ops, _ = receive(ob, tries["and"], V, F.check_ops)
            if ops and any(x.get("field") == f2 and str(x.get("value")) == str(v) for x in ops): written += 1
        for k, t in tries.items(): seen[k] += 1; applied[k] += receive(ob, t, V, F.check_ops)[0] is not None
    assert seen["so"] == 100 and seen["question"] == 102 and seen["already held"] == 86, seen
    assert all(applied[k] == 0 for k in seen if k != "and"), applied
    assert applied["and"] == 100 and written == 100                         # the blind spot: the stated value is written as a change


def test_the_host_reading_is_fast():
    s = F.make_stream(F.SEED); msgs = []
    for _ in range(F.STEPS):
        st, ob = s.next()
        if st["kind"] != "query": msgs.append((ob, st["message"], F.user_text(F.SEED, st["t"], st)))
    t0 = time.perf_counter()
    for ob, m, full in msgs: receive(ob, full, V, F.check_ops)
    with_chatter = (time.perf_counter() - t0) / len(msgs) * 1000
    t0 = time.perf_counter()
    for ob, m, full in msgs: receive(ob, m, V, F.check_ops)
    alone = (time.perf_counter() - t0) / len(msgs) * 1000
    assert with_chatter < 10 and alone < 2, (with_chatter, alone)


def test_the_page_and_the_front_pages():
    page = open(PAGE).read(); low = page.lower()
    for s in ("receive.py", "Rajnish Choubey", "a durga diary", "the addition is the failure", "cannot receive without immediately responding", "199 of 361",
              "510.5", "1,102.8", "261,093", "528,836", "4,212 of 7,220", "none wrong", "written for this check", "cannot see", "a plain \"and\"",
              "item 126", "item 127", "item 128", "the gate", "test 155", "1.7 ms", "0.1 ms", "may be"):
        assert s.lower() in low, s
    for bad in ("proven", "$", "dollar", "100 percent"): assert bad not in low, bad
    cm = open(os.path.join(DOCS, "RAJNISH_CONCEPT_MAP.md")).read(); vf = open(os.path.join(DOCS, "VERIFIED.md")).read()
    assert NEXT in cm and cm.count("Next free:") == 1 and "test 155 checks receiving without adding, item 129" in cm
    j = cm.index("\n129. **") + 1; item = cm[j:cm.index("\n", j)]
    assert "receive.py" in item and "RECEIVE_WITHOUT_ADDING.md" in item and "test 155" in item and "Rajnish Choubey" in item
    row = next(l for l in vf.split("\n") if "RECEIVE_WITHOUT_ADDING.md" in l); assert "test_receive.py" in row and "no model" in row.lower()


if __name__ == "__main__":
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]; bad = 0
    for n, f in fns:
        try: f(); print("ok  ", n)
        except Exception: bad += 1; print("FAIL", n); traceback.print_exc()
    print(f"{len(fns) - bad}/{len(fns)} passed"); sys.exit(1 if bad else 0)
