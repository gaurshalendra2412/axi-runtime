"""Engineering support for docs/LONG_HORIZON_TEST.md (concept-map rows 3, 9, 27: containment over many steps, structure against scale, errors that compound). No GPU, no real model.
The long-horizon runner measures drift as the distance between the state a policy holds and the state an oracle holds after the same requests. These checks use scripted stand-ins to show
that the instrument itself is right before any Colab number is trusted:
  1. the oracle script is deterministic, every request is admissible on the oracle's graph, applying the ideal deltas in order reproduces the oracle, and a long script keeps the graph bounded
     and uses all four request kinds;
  2. `distance` counts differing services and edges (value and weight included), is symmetric, and is 0 exactly when the graphs are equal;
  3. with a model that forgets the incident edges (the 3B and 7B behaviour without the rule): blind apply is corrupted early and accumulates dangling edges, gate never corrupts but falls
     behind (omission), repair_m6 stays exactly on the oracle;
  4. identical prompts are asked once (Memo) and the answers are unchanged by it;
  5. history mode: the context grows turn by turn, prompt tokens and seconds grow with it while state mode stays flat, the context cap stops an episode cleanly, and an out-of-memory error
     is recorded as a stopped episode instead of a crash;
  6. the runner is resumable and a resumed run equals an uninterrupted one;
  7. the reading rules of docs/LONG_HORIZON_TEST.md come out as written on constructed summaries."""
import json, os, re, sys, tempfile
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.cico_parser import CICOParser
from axi.engine.gate import Graph, check, apply
from axi.experiments.agent_loop import copy_graph, same_graph
import axi.experiments.long_horizon as LH
from axi.experiments.long_horizon import (oracle_script, distance, Memo, run_state_episode, run_history_episode, run_long, summarize_long, compare_context,
                                          verdict_state, verdict_context, POLICIES, MAX_NODES, MAX_EDGES)


class NaiveLLM:
    """Writes only the service delete (forgets incident edges), like the models without the rule; adds are right. Time and prompt size are a function of the prompt length."""
    def __init__(self, stop_after=None, oom_after=None):
        self.total = 0; self.stop_after = stop_after; self.oom_after = oom_after; self.last_prompt_tokens = None

    def generate(self, messages, constrained, max_new_tokens=400):
        self.total += 1
        if self.stop_after is not None and self.total > self.stop_after: raise RuntimeError("session dropped")
        if self.oom_after is not None and self.total > self.oom_after: raise type("OutOfMemoryError", (RuntimeError,), {})("CUDA out of memory")
        self.last_prompt_tokens = sum(len(m["content"]) for m in messages) // 4
        req = messages[-1]["content"].split("Request:")[-1]
        m = re.search(r"Delete the service at \((\d+),(\d+)\)", req)
        if m: text = f"DEL[{m[1]},{m[2]}]"
        else:
            m = re.search(r"Add a service at \((\d+),(\d+)\) with value (\d+)", req)
            if m: text = f"ADD[{m[1]},{m[2]}:{m[3]}]"
            else:
                m = re.search(r"edge from \((\d+),(\d+)\) to \((\d+),(\d+)\) with relation dep and weight (\d+)", req)
                text = f"ADD[({m[1]},{m[2]})->({m[3]},{m[4]}):dep#{m[5]}]"
        return text, 8, 0.3 + self.last_prompt_tokens / 300.0


def test_the_oracle_script_is_deterministic_admissible_and_reproduces_the_oracle():
    for seed in (0, 1, 2, 3000, 3001):
        g0, script, oracle = oracle_script(seed, 30)
        g0b, scriptb, oracleb = oracle_script(seed, 30)
        assert script == scriptb and all(same_graph(a, b) for a, b in zip(oracle, oracleb)) and same_graph(g0, g0b)
        assert len(script) == 30 and len(oracle) == 31 and same_graph(oracle[0], g0)
        live = copy_graph(g0)
        for st, nxt in zip(script, oracle[1:]):
            d = CICOParser.parse_transition_delta(st["ideal"]); assert check(live, d).ok
            apply(live, d); assert same_graph(live, nxt) and live.is_well_formed()
        assert (MAX_NODES, MAX_EDGES) == (14, 26) and all(len(o.nodes) <= 14 and len(o.edges) <= 26 for o in oracle)     # the bounds written on the page
    kinds = [s["kind"] for e in range(12) for s in oracle_script(3000 + e, 30)[1]]
    assert set(kinds) == {"add_node", "add_edge", "delete_dep", "delete_leaf"} and kinds.count("delete_dep") >= 60
    assert oracle_script(7, 30)[1] != oracle_script(8, 30)[1]


def test_distance_counts_differing_services_and_edges_values_and_weights_included():
    a = Graph(); a.add_node((0, 0), 1); a.add_node((0, 1), 2); a.add_edge((0, 0), (0, 1), "dep", 3)
    b = copy_graph(a)
    assert distance(a, b) == 0 == distance(b, a)
    b.nodes[(0, 1)] = 5; assert distance(a, b) == 1                                 # another value
    c = copy_graph(a); c.edges[((0, 0), (0, 1), "dep")] = 4; assert distance(a, c) == 1       # another weight
    d = Graph(); d.add_node((0, 0), 1); assert distance(a, d) == 2 == distance(d, a)       # a service and the edge to it are missing
    e = copy_graph(a); del e.nodes[(0, 1)]; assert distance(a, e) == 1 and not e.is_well_formed()     # the edge is left dangling: the node is the only difference... and the edge stays equal
    assert distance(Graph(), Graph()) == 0


def test_a_model_that_forgets_the_edges_corrupts_blind_apply_early_gets_held_back_by_the_gate_and_is_kept_exact_by_m6():
    rows = [run_state_episode(NaiveLLM(), 3000 + e, 30) for e in range(6)]
    s = summarize_long(rows)
    bl, ga, m6 = s["blind"], s["gate"], s["repair_m6"]
    assert bl["corrupted_by_step_10"] >= 0.8 and bl["episodes_ever_corrupted"] == 1.0
    assert bl["final_dangling"] > 0 and bl["per_step"][-1]["dangling"] >= bl["per_step"][9]["dangling"]       # dangling edges are never removed by blind apply
    assert all(x["corrupted"] == 0 for x in ga["per_step"]) and all(x["corrupted"] == 0 for x in m6["per_step"])
    assert m6["exact_share"] == 1.0 and m6["final_distance"] == 0 and m6["episodes_exact_at_end"] == 1.0
    assert ga["exact_share"] < 0.9 and ga["final_distance"] > 0                                         # the gate alone never does the deletes the model got wrong: omission
    assert bl["raw_admitted_delete_dep"] == 0.0 or bl["raw_admitted_delete_dep"] < 0.5                  # the raw proposals were mostly not admissible
    assert all(v for _, v in verdict_state(s))                                                          # the reading rule of the page, on this stand-in


def test_identical_prompts_are_asked_once_and_the_answers_do_not_change():
    plain = run_state_episode(NaiveLLM(), 3000, 30)
    base = NaiveLLM(); memo = Memo(base); cached = run_state_episode(memo, 3000, 30)
    strip = lambda r: {p: [{k: v for k, v in x.items() if k != "cached"} for x in xs] for p, xs in r["policies"].items()}
    assert strip(plain) == strip(cached)
    assert memo.hits > 0 and base.total == memo.calls and base.total < 3 * 30                          # the three policies share the prompts until they part
    assert any(x["cached"] for xs in cached["policies"].values() for x in xs) and not any(x["cached"] for x in plain["policies"]["blind"])
    first = cached["policies"]["blind"]; assert not any(x["cached"] for x in first)                     # the first policy asks for everything


def test_history_mode_grows_the_context_and_the_cost_while_state_mode_stays_flat_and_failures_are_recorded():
    state = [_rt(run_state_episode(Memo(NaiveLLM()), 3000 + e, 24, ("repair_m6",))) for e in range(3)]
    hist = [_rt(run_history_episode(NaiveLLM(), 3000 + e, 24, max_context_tokens=10 ** 9)) for e in range(3)]
    cmp = compare_context(state, hist)
    assert cmp["episodes"] == 3 and cmp["steps_compared"] == 24
    assert cmp["history"]["prompt_tokens_last"] > 3 * cmp["history"]["prompt_tokens_first"]            # the context grows turn by turn
    assert cmp["history"]["growth"] > 3.0 and cmp["state"]["growth"] < 1.3                              # so does the time; state mode does not grow with the step
    assert [h["policies"]["repair_m6"][0]["prompt_tokens"] for h in hist] == [s["policies"]["repair_m6"][0]["prompt_tokens"] for s in state]    # step 1 is the same prompt in both modes
    assert all(x["exact"] for h in hist for x in h["policies"]["repair_m6"])                            # the held state is repaired whatever the model wrote
    # the cap stops an episode cleanly
    capped = run_history_episode(NaiveLLM(), 3000, 24, max_context_tokens=1000)
    n = len(capped["policies"]["repair_m6"]); assert capped["stopped"]["repair_m6"] == "context" and 1 < n < 24
    # an out-of-memory error ends the episode and keeps what was done
    oom = run_history_episode(NaiveLLM(oom_after=5), 3000, 24, max_context_tokens=10 ** 9)
    assert oom["stopped"]["repair_m6"] == "oom" and len(oom["policies"]["repair_m6"]) == 5
    # any other error still propagates
    try: run_history_episode(NaiveLLM(stop_after=3), 3000, 24, max_context_tokens=10 ** 9); raise AssertionError("should have raised")
    except RuntimeError as e: assert "dropped" in str(e)
    assert all(v for _, v in verdict_context(cmp)[:2])


def _rt(row): return json.loads(json.dumps(row))


def _step(t, kind="add_node", corrupted=False, exact=True, dist=0, dangling=0, admitted=True, sec=1.0, pt=100):
    return dict(t=t, kind=kind, text="", tokens=1, seconds=sec, cached=False, prompt_tokens=pt, parsed=True, admitted=admitted, corrupted=corrupted, dangling=dangling, distance=dist, exact=exact, nodes=1, edges=0)


def test_the_summary_counts_what_the_page_says_it_counts():
    ep = lambda first_bad: [_step(t, corrupted=(first_bad is not None and t >= first_bad), exact=not (first_bad is not None and t >= first_bad), dist=(t - first_bad + 1 if first_bad is not None and t >= first_bad else 0),
                                  dangling=(1 if first_bad is not None and t >= first_bad else 0), kind="delete_dep", admitted=first_bad is None) for t in range(1, 13)]
    rows = [dict(seed=i, steps=12, policies={"blind": ep(fb)}, stopped={"blind": None}) for i, fb in enumerate([None, 10, 11, 3])]      # corrupted from step 10, 11, 3 or never
    s = summarize_long(rows)["blind"]
    assert s["episodes"] == 4 and s["episodes_ever_corrupted"] == 0.75
    assert s["corrupted_by_step_10"] == 0.5                                             # step 10 counts, step 11 does not
    assert abs(s["exact_share"] - (12 + 9 + 10 + 2) / 48) < 1e-12
    assert s["final_dangling"] == 0.75 and s["final_distance"] == (0 + 3 + 2 + 10) / 4 and s["episodes_exact_at_end"] == 0.25
    assert s["raw_admitted_delete_dep"] == 12 / 48 and s["n_delete_dep"] == 48
    assert s["per_step"][2]["corrupted"] == 0.25 and s["per_step"][9]["corrupted"] == 0.5 and s["per_step"][11]["n"] == 4
    assert s["per_step"][11]["distance"] == (0 + 3 + 2 + 10) / 4


def test_the_runner_resumes_after_a_dropped_session_and_refuses_another_runs_file():
    for mode in ("state", "history"):
        def new(): return os.path.join(tempfile.mkdtemp(), "long.jsonl")
        meta = {"model": "fake"}; kw = dict(episodes=4, steps=12, seed=3, hint_rules=False, meta=meta, max_context_tokens=10 ** 9)
        full = run_long(NaiveLLM(), mode, path=new(), **kw)
        assert len(full["rows"]) == 4 and full["summary"] == json.loads(json.dumps(full["summary"]))
        path = new()
        try: run_long(NaiveLLM(stop_after=60 if mode == "state" else 30), mode, path=path, **kw); raise AssertionError("should have dropped")
        except RuntimeError: pass
        done = len(open(path).read().splitlines()) - 1; assert 0 < done < 4
        llm = NaiveLLM(); resumed = run_long(llm, mode, path=path, **kw)
        assert resumed["rows"] == full["rows"] and resumed["summary"] == full["summary"]
        again = NaiveLLM(); run_long(again, mode, path=path, **kw); assert again.total == 0               # a finished file asks the model for nothing
        with open(path, "a") as f: f.write('{"seed": 1, "pol')                                           # a half-written last line is dropped and redone
        assert run_long(NaiveLLM(), mode, path=path, **kw)["rows"] == full["rows"]
        try: run_long(NaiveLLM(), mode, path=path, **dict(kw, steps=13)); raise AssertionError("should refuse")
        except ValueError as e: assert "different run" in str(e)


def _summary(corrupted_by_10, exact, never_corrupt=True):
    ps = lambda c: [dict(t=1, n=1, corrupted=c, dangling=0, distance=0, exact=1, seconds=1, prompt_tokens=1)]
    return {"blind": dict(per_step=ps(1.0), corrupted_by_step_10=corrupted_by_10, exact_share=0.5),
            "gate": dict(per_step=ps(0.0 if never_corrupt else 0.1), corrupted_by_step_10=0.0, exact_share=0.7),
            "repair_m6": dict(per_step=ps(0.0), corrupted_by_step_10=0.0, exact_share=exact)}


def test_the_reading_rules_come_out_as_written():
    ok = lambda s: [v for _, v in verdict_state(s)]
    assert ok(_summary(0.9, 0.95)) == [True, True, True]
    assert ok(_summary(0.89, 0.95)) == [True, False, True] and ok(_summary(0.9, 0.949)) == [True, True, False] and ok(_summary(0.95, 0.99, never_corrupt=False))[0] is False
    mk = lambda g_s, g_h, a_s, a_h: {"state": dict(growth=g_s, admitted_dd_last=a_s), "history": dict(growth=g_h, admitted_dd_last=a_h)}
    v = lambda c: [x for _, x in verdict_context(c)]
    assert v(mk(1.3, 3.0, 0.0, 0.1)) == [True, True, True]
    assert v(mk(1.31, 3.0, 0.0, 0.09)) == [False, True, False] and v(mk(1.0, 2.99, 0.5, 0.5)) == [True, False, False]


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
