"""Implements concept-map item 122 (many eyewitnesses, one entity; python/axi/engine/witnesses.py; docs/WITNESSES.md), test 148.
Source: Rajnish Choubey's article "The 99 Eyewitnesses of the Ramayana", the text the owner pasted on 24 Oct. This file checks, with no model:
  1. the canonical form of a change: the order of its items and exact repeats do not matter, anything else does;
  2. the quorum on a stand-in graph: two witnesses in agreement commit, a single wrong-but-legal witness cannot, a tie or too few commits
     nothing, refused proposals are set aside or counted in their repaired form, and the host graph is never changed by hearing witnesses;
  3. the quorum's arithmetic, by exact enumeration of every outcome of three witnesses through the real `quorum`: with one way of being
     wrong the wrong-commit rate equals `majority_error`; with three ways it is lower; when the witnesses copy each other it falls back to
     a single witness's rate (so the formula is a ceiling of the benefit under independence, not a measurement);
  4. the field of descriptions: bound, ambiguous, unresolved, none; and, run offline over the Test 23 stream (seed 20261122, 400 turns, no
     call), the host binds all 23 "whatever <owner> is handling" changes to the right item and flags both ambiguous questions; of the 294
     changes to an existing item, how many could be named by one fact, by two, or only by their name;
  5. the page and the front pages (concept map item 122, VERIFIED row, the counter).
What it does NOT show: that a model's witnesses are independent enough for the quorum to pay, its cost in calls, or any gain on a model:
in Test 23 the model itself got all 23 owner references and both ambiguous questions right."""
import itertools, os, re, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.cico_parser import CICOParser, StateTransitionDelta
from axi.engine.gate import Graph, check, apply
from axi.engine.witnesses import canonical, from_key, gate_repair, quorum, majority_error, quorum_miss, resolve, smallest_field
from axi.experiments import gemini_fuzzy as F

HERE = os.path.dirname(__file__)
DOCS = os.path.join(HERE, "..", "..", "docs")
PAGE = os.path.join(DOCS, "WITNESSES.md")
NEXT = "Next free: row 193, item 146, test 171, question 122"
P = CICOParser.parse_transition_delta


def host() -> Graph:
    g = Graph()
    for n, v in (((0, 0), 1), ((0, 1), 2), ((0, 2), 3), ((1, 0), 4)): g.add_node(n, v)
    g.add_edge((0, 0), (0, 1), "dep", 2); g.add_edge((0, 1), (0, 2), "dep", 5)
    return g


def snap(g: Graph): return (dict(g.nodes), dict(g.edges))


RIGHT = "DEL[(0,1)->(0,2):dep#5] DEL[0,2]"                      # the change that was asked: remove (0,2)
WRONG = ["DEL[1,0]", "ADD[2,2:9]", "DEL[(0,0)->(0,1):dep#2]"]   # three legal changes that were not asked
ILLEGAL = "DEL[0,2]"                                            # leaves a dangling edge: refused by the gate


def test_the_canonical_form_ignores_order_and_repeats_and_nothing_else():
    a = P("DEL[(0,1)->(0,2):dep#5] DEL[0,2]"); b = P("DEL[0,2] DEL[(0,1)->(0,2):dep#5]"); c = P("DEL[0,2] DEL[0,2] DEL[(0,1)->(0,2):dep#5]")
    assert canonical(a) == canonical(b) == canonical(c) and canonical(a) != canonical(P("DEL[(0,1)->(0,2):dep#4] DEL[0,2]"))
    assert canonical(from_key(canonical(a))) == canonical(a) and canonical(P("ADD[1,1:3]")) != canonical(P("ADD[1,1:4]"))


def test_the_quorum_commits_only_what_enough_witnesses_proposed_exactly():
    g = host(); before = snap(g); r, w0, w1 = P(RIGHT), P(WRONG[0]), P(WRONG[1])
    assert all(check(g, d).ok for d in (r, w0, w1)) and not check(g, P(ILLEGAL)).ok
    q = quorum(g, [r, P("DEL[0,2] DEL[(0,1)->(0,2):dep#5]"), w0]); assert q.reason == "agreed" and q.votes == 2 and canonical(q.delta) == canonical(r)
    q = quorum(g, [r, w0, w1]); assert q.delta is None and q.reason == "no quorum" and q.votes == 1        # one witness each: nothing
    q = quorum(g, [r, w0]); assert q.delta is None and q.reason == "no quorum"
    q = quorum(g, [r, w0, r, w0], need=2); assert q.delta is None and q.reason == "tie" and q.votes == 2      # two changes with enough votes each: nothing
    q = quorum(g, [w0, None, P(ILLEGAL)]); assert q.delta is None and q.admissible == 1 and q.heard == 3
    q = quorum(g, [None, P(ILLEGAL)]); assert q.delta is None and q.reason == "none admissible"
    q = quorum(g, [w0, w0, r]); assert q.reason == "agreed" and canonical(q.delta) == canonical(w0)       # the limit: agreement is what commits
    q = quorum(g, [r, r, r, w0, w1], need=4); assert q.delta is None and q.reason == "no quorum" and q.votes == 3
    # a refused proposal, repaired, counts as its repair: the dangling DEL[0,2] becomes the full deletion
    q = quorum(g, [P(ILLEGAL), r, w0], repair=gate_repair); assert q.reason == "agreed" and q.votes == 2 and canonical(q.delta) == canonical(r)
    assert snap(g) == before                                                                                # hearing witnesses changes nothing
    apply(g, quorum(g, [r, r, w1]).delta); assert (0, 2) not in g.nodes and ((0, 1), (0, 2), "dep") not in g.edges


def enumerate_wrong_commit(p: float, ways: int, copy: float = 0.0) -> float:
    """Exact probability, through the real `quorum` on the stand-in graph, that three witnesses commit a wrong change. Each witness is right
    with probability 1 - p, else wrong in one of `ways` ways, uniformly. With probability `copy` all three copy the first witness."""
    g = host(); props = [P(RIGHT)] + [P(w) for w in WRONG[:ways]]; pr = [1 - p] + [p / ways] * ways; total = 0.0
    for combo in itertools.product(range(len(props)), repeat=3):
        q = quorum(g, [props[i] for i in combo]); prob = pr[combo[0]] * pr[combo[1]] * pr[combo[2]]
        if q.delta is not None and canonical(q.delta) != canonical(props[0]): total += prob
    return (1 - copy) * total + copy * p


def test_the_arithmetic_of_the_quorum_by_exact_enumeration_through_the_real_function():
    p = 0.0115                                                     # the share of wrong replies from a correct state in the 4,000-step run
    assert abs(majority_error(p, 1, 1) - p) < 1e-15 and abs(majority_error(p, 3) - (3 * p * p * (1 - p) + p ** 3)) < 1e-15
    assert abs(majority_error(p, 3) - 0.000394) < 5e-7 and abs(quorum_miss(p, 3) - majority_error(p, 3)) < 1e-15
    assert majority_error(p, 5) < 2e-5 < majority_error(p, 3)
    one, three = enumerate_wrong_commit(p, 1), enumerate_wrong_commit(p, 3)
    assert abs(one - majority_error(p, 3)) < 1e-12 and three < one and abs(three - (p * p * (1 - p) + 21 * (p / 3) ** 3)) < 1e-15
    assert abs(enumerate_wrong_commit(p, 1, copy=1.0) - p) < 1e-15                   # witnesses that copy each other are one witness
    half = enumerate_wrong_commit(p, 1, copy=0.5); assert one < half < p


def test_the_field_of_descriptions():
    items = {"Indigo Annex": {"owner": "Priya", "day": "Monday"}, "Blue Hall": {"owner": "Priya", "day": "Friday"}, "Red Room": {"owner": "Marcus"}}
    assert resolve(items, {"owner": "Marcus"}) == ("bound", ["Red Room"])
    assert resolve(items, {"owner": "Priya"}) == ("ambiguous", ["Blue Hall", "Indigo Annex"])
    assert resolve(items, {"owner": "Priya", "day": "Friday"}) == ("bound", ["Blue Hall"])      # two descriptions meet in one item
    assert resolve(items, {"owner": "Elena"}) == ("unresolved", []) and resolve(items, {"day": "Monday", "owner": "Marcus"}) == ("unresolved", [])
    assert resolve(items, {}) == ("none", []) and resolve(items, {"status": "pending"}) == ("unresolved", [])     # a fact not held fits nothing
    assert smallest_field(items, "Red Room") == {"owner": "Marcus"} and smallest_field(items, "Blue Hall") == {"day": "Friday"}
    twin = {"A": {"owner": "Priya"}, "B": {"owner": "Priya"}}; assert smallest_field(twin, "A") is None          # only its name names it


def stream_counts():
    s = F.make_stream(20261122); own = own_ok = amb = amb_ok = 0; sizes = Counter(); kinds = Counter()
    while True:
        x = s.next()
        if x is None: break
        st, ob = x; kinds[st["kind"] if st["kind"] != "query" else "q_" + st["qkind"]] += 1
        if st["kind"] == "ref_owner":
            o = re.search(r"Whatever (\w+) is handling", st["message"]).group(1); own += 1; own_ok += resolve(ob, {"owner": o}) == ("bound", [st["ops"][0]["name"]])
        if st["kind"] == "query" and st["qkind"] == "ambiguous":
            o = re.search(r"the item (\w+) is handling", st["message"]).group(1); amb += 1; amb_ok += resolve(ob, {"owner": o})[0] == "ambiguous"
        if st["kind"] in ("set", "set2", "correct", "ref_last", "ref_owner", "ref_same", "delete"):
            sf = smallest_field(ob, st["ops"][0]["name"]); sizes[len(sf) if sf else 0] += 1
    return dict(own=own, own_ok=own_ok, amb=amb, amb_ok=amb_ok, sizes=sizes, kinds=kinds)


def test_offline_on_the_test_23_stream_the_host_binds_every_owner_reference_and_flags_every_ambiguous_question():
    c = stream_counts()
    assert (c["own"], c["own_ok"], c["amb"], c["amb_ok"]) == (23, 23, 2, 2)
    assert sum(c["sizes"].values()) == 294 and (c["sizes"][1], c["sizes"][2], c["sizes"][0]) == (238, 34, 22) and set(c["sizes"]) <= {0, 1, 2}
    import json
    d = json.load(open(os.path.join(DOCS, "results", "axi_gemini_fuzzy_seed20261122_default.json")))   # the saved run had the same stream
    for arm in "KA":
        rows = d[arm]["rows"]
        assert sum(1 for r in rows if r["kind"] == "ref_owner") == 23 and sum(1 for r in rows if r.get("qkind") == "ambiguous") == 2
        assert all(r["reply_exact"] for r in rows if r["kind"] == "ref_owner" or r.get("qkind") == "ambiguous")   # the model got them too


def test_the_page_and_the_front_pages_say_what_was_built_and_the_counter_moved():
    page = open(PAGE).read(); low = page.lower()
    for s in ("witnesses.py", "Rajnish Choubey", "The 99 Eyewitnesses of the Ramayana", "legal but wrong", "0.0115", "0.04 %", "23 of 23", "2 of 2",
              "294", "238", "34", "22", "independen", "ceiling", "no model", "test 148", "already", "the model itself"):
        assert s.lower() in low, s
    for bad in ("proven", "$", "dollar", "100 percent"): assert bad not in low, bad
    cm = open(os.path.join(DOCS, "RAJNISH_CONCEPT_MAP.md")).read(); vf = open(os.path.join(DOCS, "VERIFIED.md")).read()
    assert NEXT in cm and cm.count("Next free:") == 1 and "test 148 checks the witness quorum and the field of descriptions, item 122" in cm
    j = cm.index("\n122. **") + 1; item = cm[j:cm.index("\n", j)]
    assert "witnesses.py" in item and "WITNESSES.md" in item and "test 148" in item and "Rajnish Choubey" in item
    row = next(l for l in vf.split("\n") if "WITNESSES.md" in l); assert "test_witnesses.py" in row and "no model" in row.lower()


if __name__ == "__main__":
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]; bad = 0
    for n, f in fns:
        try: f(); print("ok  ", n)
        except Exception: bad += 1; print("FAIL", n); traceback.print_exc()
    print(f"{len(fns) - bad}/{len(fns)} passed"); sys.exit(1 if bad else 0)
