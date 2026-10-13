"""Implements concept-map item 87 (ordinary-words requests; rows 3, 9, 27): the grounding module `axi/engine/grounding.py`, its hook in `axi/engine/scope.py` (`names=`), the policy `scope_words` of
`cascade.next_state`, and the word tasks of `experiments/word_tasks.py`. No model is run. What this file establishes:
  1. reading a request against a name table: whole-word names and aliases, longest match first (so "billing worker" is not "billing"), coordinates still read, a unique partial name is bound,
     a phrase that fits two services is AMBIGUOUS, one that fits none is UNRESOLVED, a request that names nothing is 'none'; case, punctuation and stop words do not matter;
  2. the safety rule: ambiguous (or unresolved with nothing else named) applies NOTHING, and without a name table `admit_scoped` behaves exactly as before;
  3. the word tasks: same graphs, kinds, correct changes as `make_scope_tasks`; no coordinates in the request (except the new-service coordinates of add_node); grounding finds exactly the
     services the request is about on all 140 (seeds 7 to 12: 840 requests); the name pool has no word shared by two entries; the 20 ambiguity requests are all read as ambiguous with the two services;
  4. with the correct change as the model's reply, scope_words reaches the expected graph on all 140 and leaves all 20 ambiguity graphs unchanged; a model that guesses one of the two services
     (blind apply, gate, repair_m6, scope_m7) changes the graph on all 20, scope_words on none;
  5. a model that adds an unasked-for edge delete: scope_words drops it (the named scope works from words), scope_m7 (coordinates only) cannot, which is the hole this closes.
What it does NOT show: that a real model's replies behave this way (that is the pre-registered run), paraphrases outside the stop list, or names a host mis-spells."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.cico_parser import CICOParser
from axi.engine.gate import Graph
from axi.engine.grounding import NameIndex, ground, normalize, STOP
from axi.engine.scope import admit_scoped
from axi.experiments.agent_loop import copy_graph, same_graph, blind_apply
from axi.experiments.cascade import next_state
from axi.experiments.scope_tasks import make_scope_tasks
from axi.experiments.word_tasks import (ALIASES, POOL, PAIRS, WordTask, legend, make_ambiguity_tasks, make_word_tasks, pool_is_disjoint, word_messages)

NAMES = {(0, 0): "billing worker", (0, 1): "billing api", (1, 0): "auth", (1, 1): "image resizer", (2, 0): "payments"}
ALIAS = {(1, 0): ["authentication"], (2, 0): ["checkout", "pay"]}


def idx(): return NameIndex(NAMES, ALIAS)


def test_names_aliases_longest_first_coordinates_and_the_status_of_a_request():
    ix = idx()
    g = ground("Remove the billing worker service.", ix); assert g.named == {(0, 0)} and g.status == "bound" and g.mentions[0].how == "name"       # not (0,1): longest first
    g = ground("decommission AUTH!", ix); assert g.named == {(1, 0)}                                                                                  # case and punctuation do not matter
    g = ground("Decommission authentication", ix); assert g.named == {(1, 0)} and g.mentions[0].phrase == "authentication"                              # alias
    g = ground("Take the pay service out of the graph.", ix); assert g.named == {(2, 0)}                                                              # alias, stop words around it
    g = ground("Remove the dep link from auth to image resizer with weight 3.", ix); assert g.named == {(1, 0), (1, 1)} and g.status == "bound"
    g = ground("Delete the service at (4,4).", ix); assert g.named == {(4, 4)} and g.mentions[0].how == "coordinate" and g.status == "bound"            # coordinates are still read
    g = ground("Remove billing worker and auth", ix); assert g.named == {(0, 0), (1, 0)}
    g = ground("Remove every edge with weight 4.", ix); assert g.named == frozenset() and g.status == "none" and not g.ambiguous and not g.unresolved   # names nothing
    g = ground("Remove the resizer.", ix); assert g.named == {(1, 1)} and g.mentions[0].how == "partial"                                              # a unique part of a name
    g = ground("Remove the billing service.", ix); assert g.status == "ambiguous" and g.ambiguous == (("billing", frozenset({(0, 0), (0, 1)})),) and g.named == frozenset()
    g = ground("Remove the payments2 service.", ix); assert g.status == "unresolved" and g.unresolved == ("payments2",)
    g = ground("Remove the payments2 service at (0,0).", ix); assert g.status == "bound" and g.named == {(0, 0)} and g.unresolved == ("payments2",)       # unresolved with something named: proceeds
    g = ground("Remove billing and also the ledger.", ix); assert g.status == "ambiguous"                                                             # one ambiguous phrase makes the whole request ambiguous
    assert normalize("  The Billing-API, (2,1)!! ") == "the billing api" and "billing" not in STOP and "service" in STOP


def test_the_table_must_have_different_non_empty_names_and_an_exact_name_beats_a_partial_one():
    for bad in ({(0, 0): "auth", (0, 1): "Auth"}, {(0, 0): ""}, {(0, 0): "!!"}):
        try: NameIndex(bad); raise AssertionError("accepted")
        except ValueError: pass
    ix = NameIndex({(0, 0): "billing", (0, 1): "billing worker"})
    assert ground("Remove billing.", ix).named == {(0, 0)} and ground("Remove billing worker.", ix).named == {(0, 1)}                                  # 'billing' is an exact name here, not ambiguous
    ix = NameIndex({(0, 0): "queue"}, {(0, 0): ["jobs"]}); assert ground("Remove jobs.", ix).named == {(0, 0)}
    assert ground("Remove the resizer banana.", idx()).status == "unresolved"                                                                         # every leftover word must belong to the same name (not just one of them)
    ix = NameIndex({(0, 0): "payments"}, {(0, 0): ["credit card gateway"]}); g = ground("Remove the gateway.", ix); assert g.named == {(0, 0)} and g.mentions[0].how == "partial"   # a word of an alias binds too
    ix = NameIndex({(0, 0): "alpha", (0, 1): "foo baz"}, {(0, 0): ["foo"]}); g = ground("Remove foo.", ix); assert g.named == {(0, 0)} and g.mentions[0].how == "name"          # an exact alias beats a partial word of another name
    ix = NameIndex({(0, 0): "a b", (0, 1): "c d"}, {(0, 0): ["shared"], (0, 1): ["shared"]}); g = ground("Remove shared.", ix); assert g.status == "ambiguous" and g.ambiguous[0][1] == {(0, 0), (0, 1)}


def _graph():
    g = Graph()
    for c in NAMES: g.add_node(c, 5)
    g.add_edge((1, 0), (2, 0), "dep", 3); g.add_edge((1, 1), (2, 0), "dep", 2); g.add_edge((0, 0), (1, 0), "owns", 1)
    return g


def test_an_ambiguous_request_applies_nothing_and_without_a_table_the_filter_is_as_before():
    g = _graph(); ix = idx()
    d = CICOParser.parse_transition_delta("DEL[0,1]")
    r = admit_scoped(g, d, "Remove the billing service.", "named", names=ix)
    assert r.final_delta is None and r.grounding.status == "ambiguous" and "ambiguous" in r.reason and "nothing applied" in r.reason
    assert same_graph(next_state("scope_words", g, d, "Remove the billing service.", names=ix), g)
    r = admit_scoped(g, d, "Remove the ledger service.", "named", names=ix); assert r.final_delta is None and "unresolved" in r.reason
    # a clear request goes through the usual path
    d = CICOParser.parse_transition_delta("DEL[2,0] DEL[(1,0)->(2,0):dep#3] DEL[(1,1)->(2,0):dep#2]")
    r = admit_scoped(g, d, "Remove the payments service.", "named", names=ix); assert r.final_delta is not None and r.admitted_after_scope and r.grounding.named == {(2, 0)}
    exp = copy_graph(g); [exp.edges.pop(k) for k in list(exp.edges) if (2, 0) in (k[0], k[1])]; exp.nodes.pop((2, 0)); exp = copy_graph(exp)
    assert same_graph(next_state("scope_words", g, d, "Remove the payments service.", names=ix), exp)
    # no table: exactly the earlier behaviour (coordinates only), for every request text
    for text in ("Delete the service at (2,0).", "Remove the payments service.", "Remove the billing service.", None):
        a = admit_scoped(g, d, text, "named"); b = admit_scoped(g, d, text, "named", names=None)
        assert (a.named, a.dropped, a.reason, a.final_delta and a.final_delta.node_deltas) == (b.named, b.dropped, b.reason, b.final_delta and b.final_delta.node_deltas) and a.grounding is None
    for bad in ((None, ix), ("x", None)):
        try: next_state("scope_words", g, d, bad[0], names=bad[1]); raise AssertionError("accepted")
        except ValueError: pass


def test_a_word_request_gives_the_filter_something_to_scope_by_and_coordinates_only_cannot():
    g = _graph(); ix = idx()
    extra = "DEL[2,0] DEL[(1,0)->(2,0):dep#3] DEL[(1,1)->(2,0):dep#2] DEL[(0,0)->(1,0):owns#1]"           # the model also deletes an edge nobody asked about
    d = CICOParser.parse_transition_delta(extra); text = "Remove the payments service."
    w = next_state("scope_words", g, d, text, names=ix); c = next_state("scope_m7", g, d, text); m6 = next_state("repair_m6", g, d)
    assert ((0, 0), (1, 0), "owns") in w.edges and ((0, 0), (1, 0), "owns") not in c.edges                    # words: the unasked edge delete is dropped; coordinates only: kept
    assert same_graph(c, m6) and not same_graph(w, c)                                                         # the old filter was exactly M6 on a word request
    assert ((0, 0), (1, 0), "owns") in next_state("scope_m7", g, d, "Delete the service at (2,0).").edges                  # the same filter on a coordinate request: dropped it too, by coordinates


def test_the_name_pool_and_the_word_tasks_are_what_the_page_says():
    assert pool_is_disjoint() and len(set(POOL)) == len(POOL) == 22 and set(ALIASES) <= set(POOL)
    allw = [w for n in POOL for w in n.split()] + [w for a in ALIASES.values() for al in a for w in al.split()]; assert not (set(allw) & STOP)       # no name word is a stop word
    for seed in (7, 8, 9, 10, 11, 12):
        base = make_scope_tasks(140, seed); ws = make_word_tasks(140, seed)
        assert [t.kind for t in ws] == [t.kind for t in base]
        for b, w in zip(base, ws):
            assert same_graph(b.graph, w.graph) and same_graph(b.expected, w.expected) and w.ideal == b.ideal and sorted(w.names) == sorted(w.graph.nodes) and len(set(w.names.values())) == len(w.names)
            if w.kind == "delete_weight": assert w.instruction == b.instruction
            elif w.kind != "add_node": assert "(" not in w.instruction                                                                              # no coordinates written
            gr = ground(w.instruction, w.index())
            assert gr.named == w.targets and gr.status == ("none" if w.kind == "delete_weight" else "bound"), (seed, w.kind, w.instruction)
            if w.kind != "add_node": assert not any(m.how == "coordinate" for m in gr.mentions), w.instruction                                 # the request carries no coordinates
            else: assert [m.how for m in gr.mentions if m.how == "coordinate"] == ["coordinate"] and gr.unresolved             # only the place of the new service; its new name is not in the table
    ws = make_word_tasks(140, 7); assert make_word_tasks(140, 7)[5].instruction == ws[5].instruction                                            # deterministic
    assert sum(1 for t in ws if t.aliases and any(any(a in t.instruction for a in al) for al in t.aliases.values())) >= 8                       # some requests call a service by its alias


def test_the_ambiguity_requests_and_what_each_policy_does_with_a_guess():
    for seed in (7, 8, 9):
        am = make_ambiguity_tasks(20, seed); assert len(am) == 20 and all(isinstance(t, WordTask) and t.kind == "ambiguous" and t.ambiguous for t in am)
        assert len({tuple(sorted(t.targets)) for t in am}) >= 8 and len({t.instruction for t in am}) >= 5 and {t.names[c] for t in am for c in t.targets} == {n for p in PAIRS for n in p}
        for t in am:
            gr = ground(t.instruction, t.index()); assert gr.status == "ambiguous" and len(gr.ambiguous) == 1 and gr.ambiguous[0][1] == t.targets and len(t.targets) == 2
            assert same_graph(t.expected, t.graph) and t.ideal == ""
            a, b = sorted(t.targets); assert t.names[a].split()[0] == t.names[b].split()[0]                                                       # they share their first word, nobody else does
            assert sum(1 for n in t.names.values() if n.split()[0] == t.names[a].split()[0]) == 2
            for pick in (a, b):                                                                                                                   # the model guesses either one
                d = CICOParser.parse_transition_delta("DEL[%d,%d] " % pick + " ".join(f"DEL[({u[0]},{u[1]})->({v[0]},{v[1]}):{r}#{w}]" for (u, v, r), w in sorted(t.graph.edges.items()) if pick in (u, v)))
                assert not same_graph(blind_apply(t.graph, d), t.graph)
                for pol in ("gate", "repair_m6", "scope_m7"): assert not same_graph(next_state(pol, t.graph, d, t.instruction), t.graph), pol
                assert same_graph(next_state("scope_words", t.graph, d, t.instruction, names=t.index()), t.graph)


def test_the_correct_change_reaches_the_expected_graph_under_scope_words_on_all_140_and_the_prompt_shows_the_names():
    for seed in (7, 8):
        for t in make_word_tasks(140, seed):
            d = CICOParser.parse_transition_delta(t.ideal)
            assert same_graph(next_state("scope_words", t.graph, d, t.instruction, names=t.index()), t.expected), (seed, t.kind, t.instruction)
            assert same_graph(next_state("repair_m6", t.graph, d, t.instruction), t.expected)
    t = make_word_tasks(140, 7)[0]; m = word_messages(t, False); mr = word_messages(t, True)
    assert m[0]["role"] == "system" and "listed under Names" in m[0]["content"] and "DEL[2,1]" in m[0]["content"] and "Rule: a service can only be removed" not in m[0]["content"]
    assert "Rule: a service can only be removed" in mr[0]["content"] and mr[1] == m[1]
    assert legend(t) in m[1]["content"] and m[1]["content"].endswith("Request: " + t.instruction)
    assert all(f"{n.split()[0]}" in legend(t) for n in t.names.values()) and legend(t).startswith("Names: ")
    for c, n in t.names.items(): assert f"{n}" in legend(t) and f"= ({c[0]},{c[1]})" in legend(t)
    aliased = [x for x in make_word_tasks(140, 7) if x.aliases]; assert len(aliased) > 50
    for x in aliased:
        for c, al in x.aliases.items(): assert f"{x.names[c]} (also called {', '.join(al)}) = ({c[0]},{c[1]})" in legend(x)                          # the model is shown the aliases the filter knows


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn): fn(); print("PASS", name); n += 1
    print(f"{n} tests passed")
