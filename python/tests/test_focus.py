"""Implements concept-map item 124 (the 1 % view; python/axi/engine/focus.py; docs/FOCUSED_VIEW.md), test 150.
Source: Rajnish Choubey's text "The Trilogy of the Canvas: The Art of the Void" (pasted 24 Oct): deliberate deletion, 99 % deleted and 1 % shown.
It checks, with no model:
  1. `mentioned`, `focus_items` and `focus_graph` on made-up states: names, descriptions (an ambiguous one keeps every candidate), "last",
     a new item, the fallback to the whole state, the order kept; the graph view holds every edge a delete needs;
  2. offline over the Test 23 stream (seed 20261122, 400 steps, no call): the view held every item the right reply depends on at every
     step, with no fallback; the items shown against the items held;
  3. the prompt with the view, estimated from the saved Test 23 calls (tokens per character of the prompts that were really sent): its size
     over the job, and that it no longer grows with the list;
  4. the arithmetic for large states (20, 1,000 and 10,000 items): the whole list grows with the state, the view does not;
  5. the page and the front pages (concept map item 124, VERIFIED row, the counter).
What it does NOT show: that a model is as exact with the view as with the whole state; that needs a run."""
import json, os, re, statistics, sys
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.focus import focus_graph, focus_items, mentioned
from axi.engine.gate import Graph
from axi.experiments import gemini_fuzzy as F

HERE = os.path.dirname(__file__)
DOCS = os.path.join(HERE, "..", "..", "docs")
PAGE = os.path.join(DOCS, "FOCUSED_VIEW.md")
NEXT = "Next free: row 183, item 137, test 162, question 122"
DESC = {"owner": F.OWNERS}


def test_the_view_on_made_up_states():
    state = {"Amber Gallery": {"owner": "Priya", "day": "Monday"}, "Birch Studio": {"owner": "Marcus"}, "Cobalt Lounge": {"owner": "Priya"}, "Dune Depot": {"owner": "Elena"}}
    assert mentioned("Push amber gallery to Friday", state) == ["Amber Gallery"] and mentioned("Amber Galleryx", state) == []
    v, why = focus_items(state, "Push Amber Gallery to Friday.", DESC); assert list(v) == ["Amber Gallery"] and why["hidden"] == 3 and not why["fallback"]
    v, why = focus_items(state, "Whatever Marcus is handling: the day is now Tuesday.", DESC); assert list(v) == ["Birch Studio"]
    v, why = focus_items(state, "What is the status of the item Priya is handling?", DESC); assert list(v) == ["Amber Gallery", "Cobalt Lounge"]      # ambiguous stays visible
    v, why = focus_items(state, "The last one we added goes to Hiro.", DESC); assert list(v) == ["Dune Depot"] and why["recent"] == ["Dune Depot"]
    v, why = focus_items(state, "Put Birch Studio on the same day as Amber Gallery.", DESC); assert list(v) == ["Amber Gallery", "Birch Studio"]  # the order of the list is kept
    v, why = focus_items(state, "New item: Ember Foyer. It is set for Monday.", DESC, new_names=F.NAMES); assert v == {} and why["new"] == ["Ember Foyer"] and not why["fallback"]
    v, why = focus_items(state, "Please tidy up whatever needs it.", DESC, new_names=F.NAMES); assert why["fallback"] and list(v) == list(state) and why["hidden"] == 0
    v, why = focus_items({}, "anything", DESC); assert v == {} and why["fallback"]
    v, _ = focus_items(state, "Push Amber Gallery to Friday.", DESC); v["Amber Gallery"]["day"] = "Sunday"; assert state["Amber Gallery"]["day"] == "Monday"   # a copy
    g = Graph()
    for n, val in (((0, 0), 1), ((0, 1), 2), ((0, 2), 3), ((1, 1), 4), ((2, 2), 5)): g.add_node(n, val)
    g.add_edge((0, 0), (0, 1), "dep", 2); g.add_edge((0, 1), (0, 2), "dep", 5); g.add_edge((1, 1), (0, 1), "dep", 1); g.add_edge((2, 2), (1, 1), "dep", 7)
    one = focus_graph(g, [(0, 1)]); assert set(one.nodes) == {(0, 0), (0, 1), (0, 2), (1, 1)} and set(one.edges) == {k for k in g.edges if (0, 1) in k[:2]}
    assert set(one.edges) >= set(g.incident((0, 1))) and one.is_well_formed()                  # every edge a delete of (0, 1) needs
    two = focus_graph(g, [(0, 1)], hops=2); assert set(two.nodes) == set(g.nodes) and set(two.edges) == set(g.edges)
    assert focus_graph(g, [(9, 9)]).nodes == {} and focus_graph(g, [(2, 2)], hops=0).edges == {}


def depends_on(st, ob):
    """The items whose facts decide the right reply of a step of the Test 23 stream (written from the stream's own templates)."""
    k = st["kind"]
    if k == "query":
        if st["qkind"] == "ambiguous":
            o = re.search(r"the item (\w+) is handling", st["message"]).group(1); return {x for x in ob if ob[x].get("owner") == o}
        return {x for x in ob if x in st["message"]}
    d = {o["name"] for o in json.loads(st["ideal"])["ops"] if o["op"] != "add"}
    if k == "ref_same":
        m = re.search(r"same day as (.+)\.$|costs the same as (.+)\.$", st["message"]); d.add((m.group(1) or m.group(2)).strip())
    return d


def stream_view():
    s = F.make_stream(F.SEED); out = []
    while True:
        x = s.next()
        if x is None: break
        st, ob = x; view, why = focus_items(ob, st["message"], DESC, new_names=F.NAMES); out.append((st, ob, view, why))
    return out


def test_offline_on_the_test_23_stream_the_view_holds_what_every_right_reply_needs():
    sv = stream_view(); assert len(sv) == 400
    assert all(depends_on(st, ob) <= set(view) for st, ob, view, why in sv) and sum(why["fallback"] for *_, why in sv) == 0
    held = statistics.mean(len(ob) for _, ob, _, _ in sv); shown = statistics.mean(why["shown"] for *_, why in sv)
    assert round(held, 1) == 17.5 and round(shown, 1) == 1.2
    for st, ob, view, why in sv:                                                   # nothing the right reply touches is hidden
        if st["kind"] != "query": assert {o["name"] for o in json.loads(st["ideal"])["ops"] if o["op"] != "add"} <= set(view)


def prompt_estimate():
    d = json.load(open(os.path.join(DOCS, "results", "axi_gemini_fuzzy_seed20261122_default.json"))); rows = d["A"]["rows"]
    s = F.make_stream(F.SEED); full_c, view_c, toks = [], [], []
    for r in rows:
        st, ob = s.next()                                          # arm A held the list exactly at every step, so its list is the true one
        user = F.user_text(F.SEED, r["t"], st, ob); full = json.dumps([F._system("A"), {"role": "user", "content": user}])
        view, why = focus_items(ob, st["message"], DESC, new_names=F.NAMES)
        vt = F.show_state(view) + (f"\n({why['hidden']} other items not shown)" if why["hidden"] else "")
        v = json.dumps([F._system("A"), {"role": "user", "content": user.replace(F.show_state(ob), vt, 1)}])
        full_c.append(len(full)); view_c.append(len(v)); toks.append(r["prompt_tokens"])
    ratio = sum(toks) / sum(full_c)
    est = lambda xs: statistics.mean(xs) * ratio
    return dict(ratio=ratio, full=statistics.mean(toks), view=est(view_c), full50=statistics.mean(toks[:50]), view50=est(view_c[:50]), full100=statistics.mean(toks[-100:]), view100=est(view_c[-100:]))


def test_the_prompt_with_the_view_no_longer_grows_with_the_list():
    e = prompt_estimate()
    assert (round(e["full"]), round(e["view"]), round(e["full50"]), round(e["view50"]), round(e["full100"]), round(e["view100"])) == (1177, 920, 959, 912, 1337, 921)
    assert e["full100"] / e["full50"] > 1.39 and e["view100"] / e["view50"] < 1.01


def large(n):
    state = {f"Item {i:05d}": {"owner": F.OWNERS[i % 10], "day": F.DAYS[i % 7], "amount": 100 + 10 * (i % 90), "status": F.STATUSES[i % 4]} for i in range(n)}
    view, why = focus_items(state, f"Item {n // 2:05d} moves to Friday.", DESC)
    return len(F.show_state(state)), len(F.show_state(view) + f"\n({why['hidden']} other items not shown)"), why["shown"]


def test_for_large_states_the_whole_list_grows_and_the_view_does_not():
    r = prompt_estimate()["ratio"]; got = {n: large(n) for n in (20, 1000, 10000)}
    assert all(s == 1 for _, _, s in got.values())
    assert [(round(f * r), round(v * r)) for f, v, _ in (got[20], got[1000], got[10000])] == [(354, 23), (17686, 24), (176856, 24)]


def test_the_page_and_the_front_pages_say_what_was_built_and_the_counter_moved():
    page = open(PAGE).read(); low = page.lower()
    for s in ("focus.py", "Rajnish Choubey", "The Trilogy of the Canvas", "deliberate deletion", "400 of 400", "no fallback", "17.5", "1.2", "1,177", "920", "959", "912",
              "1,337", "921", "354", "17,686", "176,856", "24", "whole exact state", "test 150", "needs a run", "already"):
        assert s.lower() in low, s
    for bad in ("proven", "$", "dollar", "100 percent"): assert bad not in low, bad
    cm = open(os.path.join(DOCS, "RAJNISH_CONCEPT_MAP.md")).read(); vf = open(os.path.join(DOCS, "VERIFIED.md")).read()
    assert NEXT in cm and cm.count("Next free:") == 1 and "test 150 checks the 1 % view, item 124" in cm
    j = cm.index("\n124. **") + 1; item = cm[j:cm.index("\n", j)]
    assert "focus.py" in item and "FOCUSED_VIEW.md" in item and "test 150" in item and "Rajnish Choubey" in item
    row = next(l for l in vf.split("\n") if "FOCUSED_VIEW.md" in l); assert "test_focus.py" in row and "no model" in row.lower()


if __name__ == "__main__":
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]; bad = 0
    for n, f in fns:
        try: f(); print("ok  ", n)
        except Exception: bad += 1; print("FAIL", n); traceback.print_exc()
    print(f"{len(fns) - bad}/{len(fns)} passed"); sys.exit(1 if bad else 0)
