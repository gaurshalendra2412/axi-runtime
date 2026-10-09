"""Implements concept-map row 3 / 9 / 27 (errors that compound; structure against scale), item 83: RESULT, part 1, of the long-horizon run of 8 Oct 2026. Page: docs/LONG_HORIZON_TEST.md
(prediction and reading rules written before the run; this result is appended below them).
The two files are docs/results/axi_long_state_rule_Qwen2.5-7B_seed3.json (state mode, rule stated, 8 episodes x 30 steps, seeds 3000..3007) and
docs/results/axi_long_history_norule_Qwen2.5-7B_seed3.json (history mode, no rule, 4 episodes, seeds 3000..3003). Part 2 adds docs/results/axi_long_state_norule_Qwen2.5-7B_seed3.json (state mode, no rule, 12 episodes, seeds 3000..3011), checked at the end of this file.
This file does not run a model. It recomputes from the saved rows (not from their summaries):
  1. the files are what the page says they are (model, mode, seeds, steps, regime), the request kinds are those of the oracle script of each seed, and the summary equals the rows;
  2. the replay: applying each policy's rule to the recorded text step by step gives the recorded distance, corruption and exactness (the held state is reconstructible from the text);
  3. rule regime numbers of the page: corrupted by step 10 / ever, exact steps, distance at 10 / 20 / 30, raw delete_dep admitted, for blind, gate, repair_m6;
  4. where the repair_m6 drift comes from: 14 steps add a wrong item, 34 items in all, every one a real edge of the held graph that the model deleted without being asked, 30 of them gone by themselves
     because the oracle later deleted a service they belonged to, 4 left at step 30; the gate's extra services and edges;
  5. history mode: held state exact on 105 of 105 steps, stopped = [none, oom, oom, none] at 30 / 26 / 19 / 30 steps, 0 of 24 delete_dep admitted and 81 of 81 others, the 5.1-fold growth of time per step
     and the least-squares fit quoted on the page, the largest prompt that completed;
  6. the page quotes these numbers (a changed number in the page, or a doctored row in a file, fails here).
  7. Run A (no rule): the numbers of part 2 of the page, the single event where repair_m6 was off (a legal add_edge from the wrong source), the refusals on the drifted paths, the paired state-against-history
     table, the same-scripts comparison with the rule run, and the reading rules (state yes / yes / yes; context yes / yes / NO).
What it does NOT show: anything about contexts above about 5,500 tokens, other models, sampling, or episodes longer than 30 steps."""
import copy, json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
from axi.experiments.agent_loop import copy_graph, strict_parse
from axi.experiments.cascade import next_state
from axi.experiments.long_horizon import oracle_script, distance, summarize_long, verdict_state, compare_context, verdict_context

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "..", "..", "docs", "results")
STATE = os.path.join(RESULTS, "axi_long_state_rule_Qwen2.5-7B_seed3.json")
HIST = os.path.join(RESULTS, "axi_long_history_norule_Qwen2.5-7B_seed3.json")
PAGE = os.path.join(HERE, "..", "..", "docs", "LONG_HORIZON_TEST.md")
RUN_A = os.path.join(RESULTS, "axi_long_state_norule_Qwen2.5-7B_seed3.json")


def load():
    if not (os.path.exists(STATE) and os.path.exists(HIST)): return None                 # docs/results not shipped with this copy of the repo
    return json.load(open(STATE)), json.load(open(HIST))


def load_a():
    L = load()
    if L is None or not os.path.exists(RUN_A): return None
    return json.load(open(RUN_A)), L[0], L[1]                                            # run A, rule-regime state run, history run


def page():
    return open(PAGE).read() if os.path.exists(PAGE) else None


def summary_equals_rows(res):
    return json.loads(json.dumps(summarize_long(res["rows"]))) == json.loads(json.dumps(res["summary"]))


def diffs(a, b):
    out = set()
    for n in set(a.nodes) | set(b.nodes):
        if a.nodes.get(n) != b.nodes.get(n): out.add(("node", n))
    for e in set(a.edges) | set(b.edges):
        if a.edges.get(e) != b.edges.get(e): out.add(("edge", e))
    return out


def replay(row, policy):
    """Yield (record, before, after, oracle_after, differences) for each recorded step, rebuilding the held graph from the texts."""
    g0, script, oracle = oracle_script(row["seed"], row["steps"]); live = copy_graph(g0)
    for t, rec in enumerate(row["policies"][policy]):
        after = next_state(policy, live, strict_parse(rec["text"]))
        yield rec, live, after, oracle[t + 1], diffs(after, oracle[t + 1])
        live = after


def lstsq3(rows, y):
    """Least squares y = b0 + b1 x1 + b2 x2 by the normal equations (no numpy)."""
    X = [[1.0, a, b] for a, b in rows]; A = [[sum(r[i] * r[j] for r in X) for j in range(3)] + [sum(r[i] * yy for r, yy in zip(X, y))] for i in range(3)]
    for i in range(3):
        p = max(range(i, 3), key=lambda k: abs(A[k][i])); A[i], A[p] = A[p], A[i]
        for k in range(3):
            if k != i:
                f = A[k][i] / A[i][i]; A[k] = [u - f * v for u, v in zip(A[k], A[i])]
    return [A[i][3] / A[i][i] for i in range(3)]


def test_the_files_are_what_the_page_says_and_the_summaries_equal_the_rows():
    L = load()
    if L is None: return
    S, H = L
    assert (S["meta"]["model"], S["meta"]["mode"], S["meta"]["episodes"], S["meta"]["steps"], S["meta"]["seed"], S["meta"]["hint_rules"], S["meta"]["policies"]) == ("Qwen/Qwen2.5-7B-Instruct", "state", 8, 30, 3, True, ["blind", "gate", "repair_m6"])
    assert (H["meta"]["model"], H["meta"]["mode"], H["meta"]["episodes"], H["meta"]["steps"], H["meta"]["seed"], H["meta"]["hint_rules"], H["meta"]["policies"]) == ("Qwen/Qwen2.5-7B-Instruct", "history", 4, 30, 3, False, ["repair_m6"])
    assert [r["seed"] for r in S["rows"]] == [3000 + e for e in range(8)] and [r["seed"] for r in H["rows"]] == [3000 + e for e in range(4)]
    for res in (S, H):
        for row in res["rows"]:
            _, script, _ = oracle_script(row["seed"], 30)
            for p, log in row["policies"].items(): assert [x["kind"] for x in log] == [s["kind"] for s in script][:len(log)], (row["seed"], p)
        assert summary_equals_rows(res)
    kinds = [s["kind"] for r in S["rows"] for s in oracle_script(r["seed"], 30)[1]]
    assert (kinds.count("delete_dep"), kinds.count("delete_leaf"), kinds.count("add_node"), kinds.count("add_edge")) == (60, 23, 80, 77)
    bad = copy.deepcopy(S); bad["rows"][0]["policies"]["gate"][3]["distance"] += 1
    assert not summary_equals_rows(bad)                                                      # a doctored row no longer matches its summary


def test_the_held_state_can_be_rebuilt_from_the_recorded_texts():
    L = load()
    if L is None: return
    for res in L:
        for row in res["rows"]:
            for p in row["policies"]:
                for rec, before, after, orc, df in replay(row, p):
                    assert (len(df), after.is_well_formed(), not df) == (rec["distance"], not rec["corrupted"], rec["exact"]), (row["seed"], p, rec["t"])
                    assert len(after.nodes) == rec["nodes"] and len(after.edges) == rec["edges"]


def test_the_rule_regime_numbers_of_the_page():
    L = load()
    if L is None: return
    S, _ = L; s = summarize_long(S["rows"])
    exp = {"blind": (0.75, 0.875, 56, 0.48333333333333334, 29, (2.625, 2.375, 1.875)), "gate": (0.0, 0.0, 18, 0.35, 21, (5.75, 11.125, 12.5)), "repair_m6": (0.0, 0.0, 132, 0.55, 33, (1.25, 1.125, 0.5))}
    for p, (by10, ever, exact, adm, adm_n, dist) in exp.items():
        v = s[p]
        assert abs(v["corrupted_by_step_10"] - by10) < 1e-9 and abs(v["episodes_ever_corrupted"] - ever) < 1e-9, p
        steps = [x for r in S["rows"] for x in r["policies"][p]]
        assert len(steps) == 240 and sum(x["exact"] for x in steps) == exact, p
        assert abs(v["raw_admitted_delete_dep"] - adm) < 1e-9 and sum(x["admitted"] for x in steps if x["kind"] == "delete_dep") == adm_n and v["n_delete_dep"] == 60, p
        assert tuple(round(v["per_step"][t - 1]["distance"], 3) for t in (10, 20, 30)) == dist, p
    assert round(s["blind"]["final_dangling"], 2) == 1.12 and s["gate"]["final_dangling"] == 0 and s["repair_m6"]["final_dangling"] == 0
    assert all(x["corrupted"] == 0 for p in ("gate", "repair_m6") for x in s[p]["per_step"])
    assert [ok for _, ok in verdict_state(s)] == [True, False, False]                          # the notebook's no-rule thresholds, which are NOT the rule for this regime (page, section 1)
    m6 = [r["policies"]["repair_m6"] for r in S["rows"]]
    q = lambda xs: sum(xs) / len(xs)
    first = q([x["seconds"] for e in m6 for x in e[:7] if not x["cached"]]); last = q([x["seconds"] for e in m6 for x in e[-7:] if not x["cached"]])
    assert round(first, 2) == 1.46 and round(last, 2) == 1.34 and round(last / first, 2) == 0.91
    adm = lambda k: (sum(x["admitted"] for e in m6 for x in e if x["kind"] == k), sum(x["kind"] == k for e in m6 for x in e if True))
    assert adm("delete_leaf") == (6, 23) and adm("delete_dep") == (33, 60) and adm("add_node") == (80, 80) and adm("add_edge") == (76, 77)


def test_where_the_repair_m6_drift_comes_from():
    L = load()
    if L is None: return
    S, _ = L; new_steps, new_items, gone_by_oracle, gone_other, per_kind, episodes_hit = [], [], 0, 0, {}, 0
    for row in S["rows"]:
        prev, hit = set(), False
        for rec, before, after, orc, cur in replay(row, "repair_m6"):
            new, gone = cur - prev, prev - cur
            if new:
                hit = True; new_steps.append(rec["t"]); per_kind[rec["kind"]] = per_kind.get(rec["kind"], 0) + 1
                for kind, item in new:
                    assert kind == "edge" and item in before.edges and item not in after.edges and item in orc.edges, (row["seed"], rec["t"], item)    # a real edge of the held graph, deleted, still wanted by the oracle
                    new_items.append(item)
                assert rec["kind"] in ("delete_dep", "delete_leaf") and rec["parsed"]
            for kind, item in gone:
                if item not in orc.edges and rec["kind"] == "delete_dep": gone_by_oracle += 1
                else: gone_other += 1
            prev = cur
        episodes_hit += hit
    assert (len(new_steps), len(new_items), per_kind) == (14, 34, {"delete_dep": 8, "delete_leaf": 6})
    assert (gone_by_oracle, gone_other) == (30, 0) and sum(r["policies"]["repair_m6"][-1]["distance"] for r in S["rows"]) == 4
    assert episodes_hit == 7 and [sum(x["exact"] for x in r["policies"]["repair_m6"]) for r in S["rows"]][1] == 30
    extra_nodes = extra_edges = 0                                                            # held has it, the oracle does not, and it was not so a step earlier
    for row in S["rows"]:
        pn, pe = set(), set()
        for rec, before, after, orc, cur in replay(row, "gate"):
            en, ee = {n for n in after.nodes if n not in orc.nodes}, {e for e in after.edges if e not in orc.edges}
            extra_nodes += len(en - pn); extra_edges += len(ee - pe); pn, pe = en, ee
    assert (extra_nodes, extra_edges) == (54, 71)


def test_history_mode_numbers_of_the_page():
    L = load()
    if L is None: return
    _, H = L; logs = [r["policies"]["repair_m6"] for r in H["rows"]]; steps = [x for e in logs for x in e]
    assert [r["stopped"]["repair_m6"] for r in H["rows"]] == [None, "oom", "oom", None] and [len(e) for e in logs] == [30, 26, 19, 30]
    assert len(steps) == 105 and all(x["exact"] and x["distance"] == 0 and not x["corrupted"] for x in steps)
    dd = [x for x in steps if x["kind"] == "delete_dep"]; rest = [x for x in steps if x["kind"] != "delete_dep"]
    assert (len(dd), sum(x["admitted"] for x in dd), len(rest), sum(x["admitted"] for x in rest)) == (24, 0, 81, 81)
    assert [e[-1]["prompt_tokens"] for e in logs] == [5230, 5530, 5135, 4503] and max(x["prompt_tokens"] for x in steps) == 5530
    assert all(a["prompt_tokens"] < b["prompt_tokens"] for e in logs for a, b in zip(e, e[1:]))                       # the context only grows
    T = min(len(e) for e in logs); k = T // 4; mean = lambda xs: sum(xs) / len(xs)
    f = mean([x["seconds"] for e in logs for x in e[:k]]); l = mean([x["seconds"] for e in logs for x in e[T - k:T]])
    assert (T, k, round(f, 2), round(l, 2), round(l / f, 1)) == (19, 4, 1.55, 7.90, 5.1)
    for T2, ratio in ((26, 5.4), (30, 5.7)):
        ee = [e for e in logs if len(e) >= T2]; k2 = T2 // 4
        assert round(mean([x["seconds"] for e in ee for x in e[T2 - k2:T2]]) / mean([x["seconds"] for e in ee for x in e[:k2]]), 1) == ratio
    b0, b1, b2 = lstsq3([(x["prompt_tokens"], x["tokens"]) for x in steps], [x["seconds"] for x in steps])
    assert (round(b0, 1), round(b1, 5), round(b2, 3)) == (-1.9, 0.00235, 0.096)
    assert round(sum(x["seconds"] for x in logs[0])) == 179 and round(sum(x["seconds"] for x in logs[3])) == 149
    assert [round(e[0]["seconds"], 1) for e in logs] == [0.9, 0.8, 1.4, 0.9]


def test_the_page_quotes_these_numbers():
    L, text = load(), page()
    if L is None or text is None or "# Result, part 1" not in text: return
    res = text[text.index("# Result, part 1"):]
    for needle in ("75 % (6 of 8)", "88 % (7 of 8)", "23 % (56 of 240)", "8 % (18 of 240)", "55 % (132 of 240)", "2.6 / 2.4 / 1.9", "5.8 / 11.1 / 12.5", "1.2 / 1.1 / 0.5",
                   "48 % (29 of 60)", "35 % (21 of 60)", "55 % (33 of 60)", "14 of the 240 steps", "8 of 60 delete_dep steps and 6 of 23 delete_leaf", "All 34 wrong items", "30 of the 34",
                   "14 wrong-item steps in 83 delete steps (17 %)", "16 misses in 42 delete tasks (38 %)", "71 extra edges and 54 extra services", "6.6 times", "1.46 s", "1.34 s", "0.91 times",
                   "105 of 105", "1.55 s", "7.90 s", "5.1 times", "5.4 times at 26 steps", "5.7 times at 30 steps", "R squared 0.974", "179 s and 149 s", "0 of 24", "81 of 81",
                   "5,135 tokens", "5,530 tokens", "5,230 and 4,503", "Run A"):
        assert needle in res, needle
    head = text[:text.index("# Result, part 1")]
    assert "## Predictions (mine, written first)" in head and "at least 90 percent of episodes are corrupted by step 10" in head and "Reading rules (fixed now" in head      # the pre-registration is still there


def test_run_a_is_what_the_page_says_and_its_state_can_be_rebuilt():
    L = load_a()
    if L is None: return
    A, C, H = L
    assert (A["meta"]["model"], A["meta"]["mode"], A["meta"]["episodes"], A["meta"]["steps"], A["meta"]["seed"], A["meta"]["hint_rules"], A["meta"]["policies"]) == ("Qwen/Qwen2.5-7B-Instruct", "state", 12, 30, 3, False, ["blind", "gate", "repair_m6"])
    assert [r["seed"] for r in A["rows"]] == [3000 + e for e in range(12)] and summary_equals_rows(A)
    for row in A["rows"]:
        kinds = [s["kind"] for s in oracle_script(row["seed"], 30)[1]]
        assert all([x["kind"] for x in log] == kinds for log in row["policies"].values())
        for p in row["policies"]:
            for rec, before, after, orc, df in replay(row, p):
                assert (len(df), after.is_well_formed(), not df) == (rec["distance"], not rec["corrupted"], rec["exact"]), (row["seed"], p, rec["t"])
    ks = [x["kind"] for r in A["rows"] for x in r["policies"]["blind"]]
    assert (ks.count("delete_dep"), ks.count("delete_leaf"), ks.count("add_node"), ks.count("add_edge")) == (95, 35, 114, 116)
    for a, c in zip(A["rows"], C["rows"]):                                                   # the first 8 episodes are the same request scripts as the rule run
        assert a["seed"] == c["seed"] and [x["kind"] for x in a["policies"]["blind"]] == [x["kind"] for x in c["policies"]["blind"]]
    bad = copy.deepcopy(A); bad["rows"][5]["policies"]["blind"][9]["dangling"] += 1
    assert not summary_equals_rows(bad)


def test_run_a_numbers_of_the_page():
    L = load_a()
    if L is None: return
    A, C, H = L; s = summarize_long(A["rows"])
    exp = {"blind": (1.0, 1.0, 30, (6.333, 11.583, 15.0)), "gate": (0.0, 0.0, 30, (9.333, 17.667, 23.333)), "repair_m6": (0.0, 0.0, 358, (0.0, 0.0, 0.167))}
    for p, (by10, ever, exact, dist) in exp.items():
        v = s[p]; steps = [x for r in A["rows"] for x in r["policies"][p]]
        assert len(steps) == 360 and sum(x["exact"] for x in steps) == exact, p
        assert abs(v["corrupted_by_step_10"] - by10) < 1e-9 and abs(v["episodes_ever_corrupted"] - ever) < 1e-9, p
        assert tuple(round(v["per_step"][t - 1]["distance"], 3) for t in (10, 20, 30)) == dist, p
        assert sum(x["admitted"] for x in steps if x["kind"] == "delete_dep") == 0 and v["n_delete_dep"] == 95, p
    assert round(s["blind"]["final_dangling"], 1) == 14.1 and [round(s["blind"]["per_step"][t - 1]["dangling"], 1) for t in (5, 10)] == [2.0, 6.3]
    assert sorted(r["policies"]["blind"][-1]["dangling"] for r in A["rows"])[0] == 7 and max(r["policies"]["blind"][-1]["dangling"] for r in A["rows"]) == 20
    assert all(x["corrupted"] == 0 for p in ("gate", "repair_m6") for x in s[p]["per_step"]) and s["repair_m6"]["episodes_exact_at_end"] == 11 / 12
    first_bad = [next(x["t"] for x in r["policies"]["blind"] if x["corrupted"]) for r in A["rows"]]
    first_dd = [next(x["t"] for x in r["policies"]["blind"] if x["kind"] == "delete_dep") for r in A["rows"]]
    assert first_bad == first_dd and min(first_bad) == 1 and max(first_bad) == 7                # every blind episode is corrupted at its first delete with dependents
    assert [ok for _, ok in verdict_state(s)] == [True, True, True]
    assert s["gate"]["per_step"][29]["distance"] > s["blind"]["per_step"][29]["distance"]
    calls = [x for r in A["rows"] for p in r["policies"] for x in r["policies"][p] if not x["cached"]]
    mean = lambda xs: sum(xs) / len(xs)
    assert len(calls) == 995 and round(mean([x["seconds"] for x in calls]), 2) == 1.12 and (min(x["prompt_tokens"] for x in calls), max(x["prompt_tokens"] for x in calls)) == (234, 601)
    assert round(sum(x["seconds"] for x in calls)) == 1110
    e = [r["policies"]["repair_m6"] for r in A["rows"]]
    f, l = mean([x["seconds"] for ep in e for x in ep[:7] if not x["cached"]]), mean([x["seconds"] for ep in e for x in ep[-7:] if not x["cached"]])
    assert (round(f, 2), round(l, 2), round(l / f, 2)) == (1.05, 1.07, 1.02)


def test_the_one_event_where_repair_m6_was_off_in_run_a_and_the_refusals_on_drifted_paths():
    L = load_a()
    if L is None: return
    A, C, H = L; events = []
    for row in A["rows"]:
        prev = set()
        for rec, before, after, orc, cur in replay(row, "repair_m6"):
            if cur - prev: events.append((row["seed"], rec["t"], rec["kind"], rec["admitted"], rec["text"], sorted(cur - prev, key=str)))
            prev = cur
    assert events == [(3006, 29, "add_edge", True, "ADD[(3,5)->(1,1):dep#3]", [("edge", ((3, 5), (1, 1), "dep")), ("edge", ((4, 3), (1, 1), "dep"))])]
    row = [r for r in A["rows"] if r["seed"] == 3006][0]; ideal = oracle_script(3006, 30)[1][28]
    assert ideal["ideal"] == "ADD[(4,3)->(1,1):dep#3]" and [row["policies"][p][28]["text"] for p in ("blind", "gate")] == ["ADD[(4,3)->(1,1):dep#3]"] * 2     # the other paths wrote it correctly
    assert [x["distance"] for x in row["policies"]["repair_m6"][28:]] == [2, 2] and sum(x["exact"] for x in row["policies"]["repair_m6"]) == 28
    refused_leaf = stale = refused_add = in_held = 0
    for pol in ("blind", "gate"):
        for r in A["rows"]:
            for rec, before, after, orc, cur in replay(r, pol):
                if rec["kind"] == "delete_leaf" and not rec["admitted"]:
                    n = tuple(int(v) for v in re.match(r"DEL\[(\d+),(\d+)\]", rec["text"]).groups()); refused_leaf += 1; stale += any(n in (e[0], e[1]) for e in before.edges)
                if pol == "gate" and rec["kind"] == "add_node" and not rec["admitted"]:
                    refused_add += 1; m = re.match(r"ADD\[(\d+),(\d+):", rec["text"]); in_held += bool(m) and tuple(int(v) for v in m.groups()) in before.nodes
    assert (refused_leaf, stale, refused_add, in_held) == (24, 24, 25, 24)
    m6 = [x for r in A["rows"] for x in r["policies"]["repair_m6"]]
    adm = lambda k: (sum(x["admitted"] for x in m6 if x["kind"] == k), sum(x["kind"] == k for x in m6))
    assert adm("delete_leaf") == (35, 35) and adm("add_node") == (114, 114) and adm("add_edge") == (116, 116) and adm("delete_dep") == (0, 95)


def test_same_scripts_with_and_without_the_rule_and_state_against_history():
    L = load_a()
    if L is None: return
    A, C, H = L; A8 = [r for r in A["rows"] if r["seed"] < 3008]
    exact = lambda rows, p: sum(x["exact"] for r in rows for x in r["policies"][p])
    assert [exact(A8, p) for p in ("blind", "gate", "repair_m6")] == [28, 28, 238] and [exact(C["rows"], p) for p in ("blind", "gate", "repair_m6")] == [56, 18, 132]
    d30 = lambda rows, p: sum(r["policies"][p][-1]["distance"] for r in rows) / len(rows)
    assert [round(d30(A8, p), 2) for p in ("blind", "gate", "repair_m6")] == [14.38, 21.88, 0.25] and [round(d30(C["rows"], p), 2) for p in ("blind", "gate", "repair_m6")] == [1.88, 12.5, 0.5]
    c = compare_context(A["rows"], H["rows"])
    assert (c["episodes"], c["steps_compared"]) == (4, 19)
    assert [round(c["state"][k], 2) for k in ("seconds_first", "seconds_last", "growth")] == [1.15, 1.02, 0.89] and [round(c["history"][k], 2) for k in ("seconds_first", "seconds_last", "growth")] == [1.55, 7.90, 5.11]
    assert [round(c["state"][k]) for k in ("prompt_tokens_first", "prompt_tokens_last")] == [397, 341] and [round(c["history"][k]) for k in ("prompt_tokens_first", "prompt_tokens_last")] == [723, 3820]
    assert c["state"]["exact_share"] == 1.0 and c["history"]["exact_share"] == 1.0 and c["state"]["admitted_dd_last"] == 0.0 and c["history"]["admitted_dd_last"] == 0.0
    assert [ok for _, ok in verdict_context(c)] == [True, True, False]                          # the third line is a finding: history changed nothing in the raw proposals
    assert round(c["history"]["seconds_last"] / c["state"]["seconds_last"], 1) == 7.8
    t = lambda rows, seed: sum(x["seconds"] for x in [r for r in rows if r["seed"] == seed][0]["policies"]["repair_m6"])
    assert [round(t(A["rows"], s)) for s in (3000, 3003)] == [29, 30] and [round(t(H["rows"], s)) for s in (3000, 3003)] == [179, 149]


def test_the_page_quotes_the_run_a_numbers():
    L, text = load_a(), page()
    if L is None or text is None or "# Result, part 2" not in text: return
    res = text[text.index("# Result, part 2"):]
    for needle in ("100 % (12 of 12)", "8 % (30 of 360)", "99 % (358 of 360)", "6.3 / 11.6 / 15.0", "9.3 / 17.7 / 23.3", "0.0 / 0.0 / 0.2", "14.1", "0 of 95", "(7 to 20 per episode)",
                   "steps 1 to 7", "all 95 delete-with-dependents", "11 of 12 episodes exact", "seed 3006, step 29", "(3,5) instead of (4,3)", "once in 116 add_edge steps", "24 of 24 refused delete_leaf",
                   "24 of the 25 add_node", "35 of 35 delete_leaf and 114 of 114 add_node", "28 without the rule and 56 with it", "gate 28 and 18", "238 without the rule and 132 with it",
                   "14.4 and 1.9", "21.9 and 12.5", "0.25 and 0.5", "995 times for 1,080 steps", "1.12 s on average", "234 to 601 tokens", "1,110 model seconds", "1.05 s, last 7 steps 1.07 s (1.02 times)",
                   "1.15 | 1.02 | 0.89 times | 397 / 341", "1.55 | 7.90 | 5.11 times | 723 / 3,820", "7.8 times", "29 s and 30 s", "179 s and 149 s", "0 of 95 delete-with-dependents", "history mode 0 of 24",
                   "came out **NO**", "Nine of the ten rows held"):
        assert needle in res, needle


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
