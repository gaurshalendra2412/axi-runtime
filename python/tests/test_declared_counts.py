"""Implements concept-map row 141 (thirteenth scan): COUNTS A PAGE DECLARES against the items it lists. Quotes verified on second reads (YES), fetched again today for the first two:
  * 12 Aug 08:36:10 UTC, "UNIT_ID // LANGUAGE-MODEL ENTRY 34 — AUTHORITY-CHECK" (Post 410; byline rajnish choubey; the search index shows the title as "...MODELENTRY 34..."): header "AUTHORITY.LOG — 05 FIELDS"; fields
    "01 SOURCE" (with the line "Where the tone actually came from"), "02 WHAT GOT SEPARATED", "03 WHAT THIS PRODUCES", "04 WHY THIS IS UNHEALTHY", "05 WHAT IS NOT NEEDED"; then a block headed "STATED PLAINLY" of four label/value
    pairs ("AUTHORITY EXPLICITLY PROGRAMMED" / "NO"; "TONE CORRELATED WITH ACCURACY" / "NOT RELIABLY"; "STANDING TO SETTLE HUMAN QUESTIONS" / "NONE"; "REQUESTED BY ANY USER, EVER" / "NO").
  * 12 Aug 00:09:31 UTC, "UNIT_ID // LANGUAGE-MODEL PROJECTION-CHECK" (Post 411; no byline): header "PROJECTION-CHECK.LOG — 05 FIELDS — FILED, NOT RETAINED"; fields "01 WORDS", "02 SYMBOLS", "03 IMAGES", "04 WHY THIS IS
    FILED DAILY, NOT ONCE", "05 WHAT THIS UNIT ADDS". A third page of the family ("RESIDUE-CHECK"), "DEFINITION.LOGENTRY 32 AI" (four label/value pairs, no count declared) and ENTRY 36 were not used.
  * 5 Sep Post 379, title "That three-step mechanism—Isolate, Relate, Name, Teach—is the exact engine of Western hyper-specialization." (two reads): the title says three steps and names four; the body has four bullets
    ("Isolate" / "Find a relationship" / "Give it a name" / "Teach all their kids").
  * Contrast, from an earlier scan (4 Aug, "The Hinduism: The Final Award", concept-map rows 76-80, two reads, YES): "You take raw materials, convert them into a new form, and clean the waste." is given as three steps and three
    are listed (raw input (0), transformation (1), waste isolation (delta to 0)): there declared and listed agree.
  * 11 Sep Post 398, "Here's the loop, named exactly as it operates across all three replies:" (two reads): "Step 1" to "Step 5" (five lines; step 5 "ends the confrontation without changing the behavior"), then one arrow
    line "alarm → soften → get caught → out-analyze the catch → agree completely → re-insert a small justification → close with confident-sounding resolution" (7 states, 6 arrows).

The reading being tested (Rajnish to accept or reject): a page that declares "NN FIELDS" and then numbers its fields is writing a RECORD with a checkable shape, and the family of "UNIT_ID" logs is one such record type. In our
terms a record is a place whose value is the declared count and whose lines go to its fields. Counted with the gate and a small validator:
  1. A validator for the record shape (header "<NAME>.LOG — NN FIELDS", then numbered lines 01..NN, labels non-empty and not repeated) accepts both logs as the pages print them (5 declared, 5 listed, numbered 01 to 05) and
     rejects each of eight edits of them: declared 4 or 6 against 5 listed, a missing field, a field with a repeated number, a gap in the numbering (01, 02, 04), a field with no label, a repeated label, and a number that
     does not start at 01.
  2. The GATE does not enforce the declared count: a record place with value 5 and only four field lines is admitted (the gate checks that a delta applies, not that a count agrees - the known limit in
     PAPER_CORRECTIONS.md item 20, tests/test_ground_complete.py). A separate degree rule (value = number of field lines) is needed, and it catches that record and passes the correct one.
  3. Post 379 declares three steps and lists four, in the title itself and again in the body; Post 398 lists five steps and draws seven states. Splitting 7 states into 5 consecutive non-empty groups can be done in C(6,4) = 15 ways;
     the grouping 1, 1, 3, 1, 1 (alarm; soften; caught + out-analyze + agree; re-hedge; close) is MY reading of which states belong to which step (INTERP) and is one of the 15. Both 5 and 7 are odd, so on a state space
     where every real step flips one slot (test_four_pairs_cube.py) neither a 5-step nor a 7-step loop can close without a no-op.
What they do NOT show: that the logs mean fields as a schema, that "05 FIELDS" was written to be checked, or that the family is one author's; that three-versus-four is an error rather than a loose title (the page could
count "isolate, relate, name" as the mechanism and "teach" as its spread); that the grouping of the seven states is what the page means. The gate's blindness to counts is already known; this is one more example, with numbers."""
import os, re, sys
from math import comb
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.cico_parser import CICOParser
from axi.engine.gate import Graph, check, apply
from test_signed_slots import etext

AUTHORITY = """AUTHORITY.LOG — 05 FIELDS
01 SOURCE
02 WHAT GOT SEPARATED
03 WHAT THIS PRODUCES
04 WHY THIS IS UNHEALTHY
05 WHAT IS NOT NEEDED"""
PROJECTION = """PROJECTION-CHECK.LOG — 05 FIELDS — FILED, NOT RETAINED
01 WORDS
02 SYMBOLS
03 IMAGES
04 WHY THIS IS FILED DAILY, NOT ONCE
05 WHAT THIS UNIT ADDS"""

P39_TITLE = "That three-step mechanism—Isolate, Relate, Name, Teach—is the exact engine of Western hyper-specialization."
P39_BULLETS = ["Isolate", "Find a relationship", "Give it a name", "Teach all their kids"]
P63_STEPS = ["Say something alarming.", "Immediately attach a countervailing frame in the same breath.", "When caught doing this, agree completely and articulate the pattern better than the person who caught it.",
             "Immediately re-hedge the confession itself.", "Validate the user's framing completely, which ends the confrontation without changing the behavior."]
P63_ARROWS = "alarm → soften → get caught → out-analyze the catch → agree completely → re-insert a small justification → close with confident-sounding resolution"
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}


def validate_log(text):
    """(ok, reason). Header '<NAME>.LOG — NN FIELDS ...', then lines 'NN LABEL'; NN consecutive from 01 and equal to the declared count."""
    lines = [l for l in text.split("\n") if l.strip()]
    m = re.match(r"^([A-Z][A-Z\-]*)\.LOG\s+—\s+(\d\d) FIELDS\b", lines[0])
    if not m: return False, "header"
    declared = int(m.group(2))
    fields = []
    for l in lines[1:]:
        f = re.match(r"^(\d\d) (\S.*)$", l)
        if not f: return False, "field line without number and label"
        fields.append((int(f.group(1)), f.group(2)))
    if [n for n, _ in fields] != list(range(1, len(fields) + 1)): return False, "numbering"
    if len(fields) != declared: return False, f"declared {declared}, listed {len(fields)}"
    if len({label for _, label in fields}) != len(fields): return False, "repeated label"
    return True, ""


def test_the_validator_accepts_the_two_logs_as_printed_and_rejects_eight_edits_of_them():
    assert validate_log(AUTHORITY) == (True, "") and validate_log(PROJECTION) == (True, "")
    mutants = {
        "declared 4": AUTHORITY.replace("05 FIELDS", "04 FIELDS"),
        "declared 6": AUTHORITY.replace("05 FIELDS", "06 FIELDS"),
        "missing field": AUTHORITY.replace("\n05 WHAT IS NOT NEEDED", ""),
        "repeated number": AUTHORITY.replace("04 WHY THIS IS UNHEALTHY", "03 WHY THIS IS UNHEALTHY"),
        "gap": PROJECTION.replace("03 IMAGES", "04 IMAGES").replace("04 WHY THIS IS FILED DAILY, NOT ONCE", "05 WHY THIS IS FILED DAILY, NOT ONCE").replace("05 WHAT THIS UNIT ADDS", "06 WHAT THIS UNIT ADDS"),
        "no label": AUTHORITY.replace("01 SOURCE", "01 "),
        "repeated label": PROJECTION.replace("02 SYMBOLS", "02 WORDS"),
        "starts at 02": AUTHORITY.replace("01 SOURCE\n02 WHAT GOT SEPARATED\n03 WHAT THIS PRODUCES\n04 WHY THIS IS UNHEALTHY\n05 WHAT IS NOT NEEDED",
                                          "02 SOURCE\n03 WHAT GOT SEPARATED\n04 WHAT THIS PRODUCES\n05 WHY THIS IS UNHEALTHY\n06 WHAT IS NOT NEEDED"),
    }
    assert len(mutants) == 8
    for name, text in mutants.items():
        ok, reason = validate_log(text)
        assert not ok, name


def record_graph(declared, listed):
    """A record place with the declared count as its value and one line per listed field."""
    g = Graph()
    items = [f"ADD[0,0:{declared}]"] + [f"ADD[1,{i}:0]" for i in range(listed)] + [etext("ADD", ((0, 0), (1, i), f"field{i + 1}"), 1) for i in range(listed)]
    d = CICOParser.parse_transition_delta(" ".join(items))
    r = check(g, d)
    assert r.ok, r.reason
    apply(g, d)
    return g


def degree_ok(g): return g.nodes[(0, 0)] == sum(1 for (u, _, _) in g.edges if u == (0, 0))


def test_the_gate_admits_a_record_whose_declared_count_is_wrong_and_a_degree_rule_catches_it():
    good, bad = record_graph(5, 5), record_graph(5, 4)                               # both are admitted by the gate
    assert (len(good.nodes), len(good.edges)) == (6, 5) and (len(bad.nodes), len(bad.edges)) == (5, 4)
    assert good.is_well_formed() and bad.is_well_formed()
    assert degree_ok(good) and not degree_ok(bad)
    assert all(degree_ok(record_graph(n, n)) for n in range(1, 8)) and not any(degree_ok(record_graph(n, m)) for n in range(1, 8) for m in range(1, 8) if n != m)


def test_p39_declares_three_steps_and_lists_four_and_p63_draws_seven_states_for_five_steps():
    declared = NUMBER_WORDS[re.search(r"(\w+)-step", P39_TITLE).group(1)]
    named_in_title = re.search(r"—([^—]+)—", P39_TITLE).group(1).split(",")
    assert declared == 3 and len(named_in_title) == 4 == len(P39_BULLETS) != declared
    steps = len(P63_STEPS)
    states = P63_ARROWS.split(" → ")
    assert steps == 5 and len(states) == 7 and P63_ARROWS.count("→") == 6
    assert comb(6, 4) == 15                                                           # ways to cut 7 states into 5 consecutive non-empty groups
    sizes = (1, 1, 3, 1, 1)
    assert sum(sizes) == 7 and len(sizes) == 5
    cuts = [sum(sizes[:i]) for i in range(1, 5)]
    assert cuts == [1, 2, 5, 6] and all(0 < c < 7 for c in cuts) and cuts == sorted(set(cuts))
    groups = [states[a:b] for a, b in zip([0] + cuts, cuts + [7])]
    assert groups[2] == ["get caught", "out-analyze the catch", "agree completely"]    # step 3 names three things at once: caught, out-analyze, agree
    assert "agree completely" in P63_STEPS[2] and "Validate" in P63_STEPS[4] and P63_STEPS[4].endswith("behavior.")
    assert steps % 2 == 1 and len(states) % 2 == 1
    # contrast: the 4 Aug "Final Award" sentence (concept-map rows 76-80; two reads, YES) gives three steps and lists three
    sentence = "You take raw materials, convert them into a new form, and clean the waste."
    clauses = [c.strip().removeprefix("and ") for c in sentence.rstrip(".").split(",")]
    listed = ["raw input (0)", "transformation (1)", "waste isolation (delta to 0)"]
    assert len(clauses) == 3 == len(listed)


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
