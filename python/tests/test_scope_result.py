"""Implements concept-map row 3 / 9 / 27 (errors that compound; structure against scale), item 84: RESULT of the scope test of 8 Oct 2026. Page: docs/SCOPE_TEST.md
(prediction and reading rules written before the run; the result is appended below them).
The four files are docs/results/axi_scope_{single_rule, single_norule, long_rule, long_norule}_Qwen2.5-7B_seed7.json.
This file does not run a model. It recomputes from the saved rows (not from their summaries):
  1. the files are what the page says they are (model, regime, seed, counts, request kinds equal make_scope_tasks(140, 7) and the oracle scripts of seeds 7000 to 7011), no output is unparsed,
     and every saved summary equals the summary recomputed from the rows;
  2. Part 1: every cell of the page's table (kind x policy x regime), corrupted counts, helped / harmed, scope_m7 equal to repair_m6 on 140 of 140 without the rule and 124 of 140 with it,
     the filter's drops (16 tasks, 80 items, 79 real edges, 1 not in the graph, none of them in the correct change), the 11 misses all delete_weight, 9 of them with a real edge of another weight deleted;
  3. Part 2: the held state can be rebuilt from the recorded texts (distance, exactness, corruption, drops); 263 / 354 / 237 / 237 exact steps; the off steps of scope_m7 are ONE event per regime
     (seed 7007 step 18, seed 7002 step 28); scope_m7 exact where repair_m6 was not on 91 steps and never the reverse; drops at 24 steps, 23 of them exact;
  4. cost: tokens and model seconds with and without the rule; the filter's time per request;
  5. the notebook's reading rules, 7 lines for Part 1 and 6 for Part 2, all yes;
  6. the page quotes these numbers, and the part of the page above the Result line is byte for byte what was written before the run (hash).
What it does NOT show: requests in ordinary words, other models, sampling, longer episodes, or the two failure modes written down before the run (they never occurred)."""
import copy, hashlib, json, os, re, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.cico_parser import CICOParser
from axi.engine.scope import admit_scoped
from axi.experiments.agent_loop import copy_graph, same_graph, strict_parse
from axi.experiments.cascade import next_state
from axi.experiments.long_horizon import oracle_script, distance, summarize_long
from axi.experiments.scope_tasks import make_scope_tasks, KINDS_SCOPE
from axi.experiments.scope_test import summarize_scope, evaluate, verdict_scope, verdict_scope_long

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "..", "..", "docs", "results")
PAGE = os.path.join(HERE, "..", "..", "docs", "SCOPE_TEST.md")
NAMES = {"sr": "single_rule", "sn": "single_norule", "lr": "long_rule", "ln": "long_norule"}
PRE_REGISTRATION_SHA256 = "d19dd3adbbee24ead18eeed8ed66083273e5d537155f3948753ecb08d8ecf5fb"     # of the page text above the Result line, as sent before the run
RESULT_MARKER = "\n\n---\n\n# Result (run 8 Oct 2026"


def load():
    paths = {k: os.path.join(RESULTS, f"axi_scope_{n}_Qwen2.5-7B_seed7.json") for k, n in NAMES.items()}
    if not all(os.path.exists(p) for p in paths.values()): return None                   # docs/results not shipped with this copy of the repo
    return {k: json.load(open(p)) for k, p in paths.items()}


def page():
    return open(PAGE).read() if os.path.exists(PAGE) else None


def diffs(a, b):
    out = set()
    for n in set(a.nodes) | set(b.nodes):
        if a.nodes.get(n) != b.nodes.get(n): out.add(("node", n))
    for e in set(a.edges) | set(b.edges):
        if a.edges.get(e) != b.edges.get(e): out.add(("edge", e))
    return out


def replay(row, policy):
    """Yield (record, request, before, after, oracle_after, differences) per recorded step, rebuilding the held graph from the texts."""
    g0, script, oracle = oracle_script(row["seed"], row["steps"], "scope"); live = copy_graph(g0)
    for t, rec in enumerate(row["policies"][policy]):
        d = strict_parse(rec["text"]); after = next_state(policy, live, d, script[t]["instruction"])
        yield rec, script[t], live, after, oracle[t + 1], diffs(after, oracle[t + 1])
        live = after


TASKS = None
def tasks():
    global TASKS
    if TASKS is None: TASKS = make_scope_tasks(140, 7)
    return TASKS


def test_the_files_are_what_the_page_says_and_the_summaries_equal_the_rows():
    R = load()
    if R is None: return
    for k, hint in (("sr", True), ("sn", False)):
        m = R[k]["meta"]
        assert (m["model"], m["seed"], m["n"], m["hint_rules"], m["test"]) == ("Qwen/Qwen2.5-7B-Instruct", 7, 140, hint, "scope_single_step"), k
        rows = R[k]["rows"]
        assert [r["kind"] for r in rows] == [t.kind for t in tasks()] and len(rows) == 140
        assert all(strict_parse(r["text"]) is not None for r in rows)                           # nothing unparsed
        assert json.loads(json.dumps(summarize_scope(rows, tasks()))) == R[k]["summary"], k
    for k, hint, eps in (("lr", True, 12), ("ln", False, 8)):
        m = R[k]["meta"]
        assert (m["model"], m["mode"], m["episodes"], m["steps"], m["seed"], m["hint_rules"], m["policies"], m["mix"]) == ("Qwen/Qwen2.5-7B-Instruct", "state", eps, 30, 7, hint, ["repair_m6", "scope_m7"], "scope"), k
        assert [r["seed"] for r in R[k]["rows"]] == [7000 + e for e in range(eps)]
        for row in R[k]["rows"]:
            _, script, _ = oracle_script(row["seed"], 30, "scope"); assert len(script) == 30
            for p, log in row["policies"].items():
                assert [x["kind"] for x in log] == [s["kind"] for s in script], (row["seed"], p)
                assert all(x["parsed"] for x in log)
        assert json.loads(json.dumps(summarize_long(R[k]["rows"]))) == R[k]["summary"], k
    # a doctored row no longer matches its summary
    bad = copy.deepcopy(R["lr"]); bad["rows"][0]["policies"]["scope_m7"][3]["distance"] += 1
    assert json.loads(json.dumps(summarize_long(bad["rows"]))) != bad["summary"]
    bad = copy.deepcopy(R["sr"]); bad["rows"][5]["text"] = "ADD[0,0:1]"
    assert json.loads(json.dumps(summarize_scope(bad["rows"], tasks()))) != bad["summary"]


def test_part_1_the_table_of_the_page():
    R = load()
    if R is None: return
    P = ("blind", "gate", "repair_m6", "scope_m7", "scope_delta")
    exp = {  # kind: (no rule cells, rule cells), each in the order blind, gate, repair_m6, scope_m7, scope_delta
        "delete_dep": ((0, 0, 20, 20, 20), (11, 9, 19, 20, 20)), "delete_leaf": ((20, 20, 20, 20, 20), (5, 0, 5, 20, 20)),
        "delete_edge": ((20, 20, 20, 20, 0), (20, 20, 20, 20, 0)), "delete_pair": ((4, 4, 20, 20, 0), (4, 4, 20, 20, 0)),
        "delete_weight": ((15, 15, 15, 15, 0), (14, 14, 14, 14, 0)), "add_node": ((20,) * 5, (20,) * 5), "add_edge": ((20,) * 5, (20,) * 5),
        "old_deletes": ((20, 20, 40, 40, 40), (16, 9, 24, 40, 40)), "all": ((99, 99, 135, 135, 80), (94, 87, 118, 134, 80))}
    for g, (nr, ru) in exp.items():
        for key, want in (("sn", nr), ("sr", ru)):
            s = summarize_scope(R[key]["rows"], tasks())[g]
            assert tuple(s[p]["success"] for p in P) == want, (g, key)
            assert s["n"] == (40 if g == "old_deletes" else 140 if g == "all" else 20)
    for key, corrupted in (("sn", (36, 0, 0, 0, 0)), ("sr", (24, 0, 0, 0, 0))):
        s = summarize_scope(R[key]["rows"], tasks())
        assert tuple(s["all"][p]["corrupted"] for p in P) == corrupted, key
    sn, sr = summarize_scope(R["sn"]["rows"], tasks()), summarize_scope(R["sr"]["rows"], tasks())
    assert sn["paired"] == dict(helped=0, harmed=0, both_right=135, both_wrong=5) and sr["paired"] == dict(helped=16, harmed=0, both_right=118, both_wrong=6)
    assert sn["m7_equals_m6"]["all"] == 140 and sr["m7_equals_m6"]["all"] == 124 and sr["m7_equals_m6"]["delete_weight"] == 20 and sn["m7_equals_m6"]["delete_weight"] == 20
    assert sn["dropped"] == dict(tasks_with_drops=0, items=0, tasks_with_drops_and_right=0)
    assert sr["dropped"] == dict(tasks_with_drops=16, items=80, tasks_with_drops_and_right=16)
    # 10 * (134 - 118) >= 140 with room of 2 tasks: 16 against 14 needed
    assert 10 * (134 - 118) >= 140 and 10 * (134 - 118 - 2) >= 140 and 10 * (134 - 118 - 3) < 140     # a lead of 14 would still pass, 13 would not
    assert 10 * 24 <= 75 * 4 and 10 * 24 == 240 and 24 * 100 // 40 == 60                         # repair_m6 on the old deletes: 60 percent, line at 75
    assert round(100 * 135 / 140, 1) == 96.4 and round(100 * 16 / 140, 1) == 11.4


def _ideal_texts(t):
    ideal = CICOParser.parse_transition_delta(t.ideal)
    return {f"{e.op}[({e.u[0]},{e.u[1]})->({e.v[0]},{e.v[1]}):{e.relation}#{e.weight}]" for e in ideal.edge_deltas} | {f"{x.op}[{x.r},{x.c}]" for x in ideal.node_deltas if x.op == "DEL"} \
           | {f"{x.op}[{x.r},{x.c}:{x.val}]" for x in ideal.node_deltas if x.op == "ADD"}


def test_part_1_what_the_filter_dropped_and_where_scope_m7_missed():
    R = load()
    if R is None: return
    rows = R["sr"]["rows"]; items = real = phantom = needed = 0; kinds = {}
    for r, t in zip(rows, tasks()):
        res = admit_scoped(t.graph, strict_parse(r["text"]), t.instruction, "named")
        for reason, txt in res.dropped:
            items += 1; kinds[t.kind] = kinds.get(t.kind, 0) + 1
            assert reason == "edge removed that the request does not name and that no deleted named service owns"
            needed += txt in _ideal_texts(t)
            m = re.match(r"DEL\[\((\d+),(\d+)\)->\((\d+),(\d+)\):(\w+)#(\d+)\]", txt); k = ((int(m[1]), int(m[2])), (int(m[3]), int(m[4])), m[5])
            if k in t.graph.edges: real += 1
            else: phantom += 1
    assert (items, real, phantom, needed, kinds) == (80, 79, 1, 0, {"delete_leaf": 77, "delete_dep": 3})
    # the misses: every one of the 11 is a delete_weight; 9 of them deleted a real edge whose weight was not the one asked for
    tot = other_weight = 0; missed = {}
    for key in ("sn", "sr"):
        ev = evaluate(R[key]["rows"], tasks()); missed[key] = {i for i, r in enumerate(ev) if not r["scope_m7"]["success"]}
        assert {ev[i]["kind"] for i in missed[key]} == {"delete_weight"}
        for i in missed[key]:
            t = tasks()[i]; d = strict_parse(R[key]["rows"][i]["text"]); tot += 1
            exp = {(e.u, e.v, e.relation) for e in CICOParser.parse_transition_delta(t.ideal).edge_deltas}; got = {(e.u, e.v, e.relation) for e in d.edge_deltas}
            other_weight += bool([k for k in got if k in t.graph.edges and k not in exp])
    assert (len(missed["sn"]), len(missed["sr"]), tot, other_weight) == (5, 6, 11, 9)
    assert sorted(missed["sn"] & missed["sr"]) == [67, 102, 109]                               # the same three tasks fail in both regimes
    # scope_delta (the post-hoc baseline) is wrong exactly where the page said: 0 of 60 per regime on the three kinds
    for key in ("sn", "sr"):
        s = summarize_scope(R[key]["rows"], tasks())
        assert sum(s[k]["scope_delta"]["success"] for k in ("delete_edge", "delete_pair", "delete_weight")) == 0


def test_part_2_the_held_state_can_be_rebuilt_and_the_percentages_of_the_page():
    R = load()
    if R is None: return
    for key in ("lr", "ln"):
        for row in R[key]["rows"]:
            for p in row["policies"]:
                for rec, st, before, after, orc, df in replay(row, p):
                    assert (len(df), after.is_well_formed(), not df) == (rec["distance"], not rec["corrupted"], rec["exact"]), (key, row["seed"], p, rec["t"])
                    assert len(after.nodes) == rec["nodes"] and len(after.edges) == rec["edges"]
                    if p == "scope_m7":
                        assert rec["dropped"] == len(admit_scoped(before, strict_parse(rec["text"]), st["instruction"], "named").dropped)
    want = {("lr", "repair_m6"): (263, 11, 6), ("lr", "scope_m7"): (354, 1, 1), ("ln", "repair_m6"): (237, 1, 3), ("ln", "scope_m7"): (237, 1, 2)}
    for (key, p), (exact, eps_off, maxd) in want.items():
        steps = [x for r in R[key]["rows"] for x in r["policies"][p]]
        assert sum(x["exact"] for x in steps) == exact and len(steps) == (360 if key == "lr" else 240), (key, p)
        assert sum(any(not x["exact"] for x in r["policies"][p]) for r in R[key]["rows"]) == eps_off, (key, p)
        assert max(x["distance"] for x in steps) == maxd and not any(x["corrupted"] for x in steps), (key, p)
    assert round(100 * 263 / 360, 1) == 73.1 and round(100 * 354 / 360, 1) == 98.3 and round(100 * 237 / 240, 1) == 98.8 and round(100 * (354 - 263) / 360, 1) == 25.3
    s = summarize_long(R["lr"]["rows"]); assert round(s["scope_m7"]["exact_share"] - s["repair_m6"]["exact_share"], 3) == 0.253 and s["scope_m7"]["episodes_ever_corrupted"] == 0
    # step by step, rule regime: scope_m7 exact where repair_m6 was not, and the reverse
    ahead = behind = drops = drops_exact = items = 0; kinds = {}
    for row in R["lr"]["rows"]:
        for a, b in zip(row["policies"]["repair_m6"], row["policies"]["scope_m7"]):
            ahead += b["exact"] and not a["exact"]; behind += a["exact"] and not b["exact"]
            if b["dropped"]: drops += 1; drops_exact += b["exact"]; items += b["dropped"]; kinds[b["kind"]] = kinds.get(b["kind"], 0) + 1
    assert (ahead, behind, drops, drops_exact, items, kinds) == (91, 0, 24, 23, 51, {"delete_dep": 8, "delete_leaf": 13, "add_edge": 3})
    ahead = behind = drops = 0
    for row in R["ln"]["rows"]:
        for a, b in zip(row["policies"]["repair_m6"], row["policies"]["scope_m7"]):
            ahead += b["exact"] and not a["exact"]; behind += a["exact"] and not b["exact"]; drops += b["dropped"]
    assert (ahead, behind, drops) == (0, 0, 1)


def test_part_2_the_off_steps_are_one_event_per_regime():
    R = load()
    if R is None: return
    row = [r for r in R["lr"]["rows"] if r["seed"] == 7007][0]
    off = [x["t"] for x in row["policies"]["scope_m7"] if not x["exact"]]
    assert off == [18, 19, 20, 21, 22, 23]                                                       # the only off steps of scope_m7 in the rule regime
    assert sum(not x["exact"] for r in R["lr"]["rows"] for x in r["policies"]["scope_m7"]) == 6
    g0, script, oracle = oracle_script(7007, 30, "scope")
    assert script[17]["instruction"] == "Add a dependency edge from (0,0) to (0,3) with relation dep and weight 5." and row["policies"]["scope_m7"][17]["text"] == "ADD[(1,2)->(0,3):dep#5]"
    assert row["policies"]["scope_m7"][17]["dropped"] == 1 and row["policies"]["scope_m7"][17]["distance"] == 1
    k = ((0, 0), (0, 3), "dep")
    assert [t for t in range(0, 31) if k in oracle[t].edges] == [18, 19, 20, 21, 22, 23]          # the edge exists in the oracle exactly during the off steps
    assert script[23]["instruction"] == "Delete the service at (0,0)."                          # request 24: the service delete that removes the edge from the oracle
    # the same slip, repair_m6: wrong edge applied (the filter is what turned it into a missing edge)
    m6 = [x for x in row["policies"]["repair_m6"]]
    assert m6[17]["text"] == "ADD[(1,2)->(0,3):dep#5]" and m6[17]["distance"] == 2
    # no rule
    row = [r for r in R["ln"]["rows"] if r["seed"] == 7002][0]
    for p, dist in (("repair_m6", [3, 2, 2]), ("scope_m7", [2, 2, 2])):
        log = row["policies"][p]; assert [x["t"] for x in log if not x["exact"]] == [28, 29, 30] and [x["distance"] for x in log[27:]] == dist, p
    _, script, _ = oracle_script(7002, 30, "scope")
    assert script[27]["instruction"] == "Delete the service at (1,2)." and row["policies"]["scope_m7"][27]["text"] == "DEL[0,2]" and row["policies"]["scope_m7"][27]["dropped"] == 1
    assert sum(not x["exact"] for r in R["ln"]["rows"] for p in ("repair_m6", "scope_m7") for x in r["policies"][p]) == 6    # 3 + 3, all in seed 7002


def test_cost_numbers_of_the_page():
    R = load()
    if R is None: return
    tok = {k: sum(r["tokens"] for r in R[k]["rows"]) for k in ("sn", "sr")}; sec = {k: sum(r["seconds"] for r in R[k]["rows"]) for k in ("sn", "sr")}
    assert (tok["sn"], tok["sr"]) == (2040, 3975) and round(tok["sr"] / tok["sn"], 1) == 1.9
    assert (round(sec["sn"]), round(sec["sr"])) == (193, 317) and round(sec["sr"] / sec["sn"], 1) == 1.6
    for key, (t_, s_) in (("sn", (7.0, 0.91)), ("sr", (72.2, 4.91))):
        rs = [r for r in R[key]["rows"] if r["kind"] == "delete_leaf"]
        assert round(sum(r["tokens"] for r in rs) / len(rs), 1) == t_ and round(sum(r["seconds"] for r in rs) / len(rs), 2) == s_, key
    mean_call = sum(r["seconds"] for r in R["sr"]["rows"]) / 140; assert round(mean_call, 2) == 2.27
    ds = [(t, strict_parse(r["text"])) for r, t in zip(R["sr"]["rows"], tasks())]
    t0 = time.perf_counter()
    for t, d in ds: admit_scoped(t.graph, d, t.instruction, "named")
    per = (time.perf_counter() - t0) / len(ds)
    assert per < 0.005                                                                           # the page says about 11 microseconds; the bound is loose on purpose (machines differ)
    # the best arrangement on this model: no rule with plain M6 (135 of 140, 193 s) against rule plus scope_m7 (134 of 140, 317 s)
    sn, sr = summarize_scope(R["sn"]["rows"], tasks()), summarize_scope(R["sr"]["rows"], tasks())
    assert sn["all"]["repair_m6"]["success"] == 135 and sr["all"]["scope_m7"]["success"] == 134 and sec["sn"] < sec["sr"]


def test_the_notebook_reading_rules_all_come_out_yes():
    R = load()
    if R is None: return
    S = {True: summarize_scope(R["sr"]["rows"], tasks()), False: summarize_scope(R["sn"]["rows"], tasks())}
    v1 = verdict_scope(S); assert len(v1) == 7 and all(ok for _, ok in v1), [t for t, ok in v1 if not ok]
    v2 = verdict_scope_long(summarize_long(R["lr"]["rows"]), summarize_long(R["ln"]["rows"])); assert len(v2) == 6 and all(ok for _, ok in v2), [t for t, ok in v2 if not ok]
    # the lines are not vacuous: a doctored regime flips them
    bad = copy.deepcopy(S); bad[True]["old_deletes"]["scope_m7"]["success"] = 35; assert not all(ok for _, ok in verdict_scope(bad))
    bad = copy.deepcopy(S); bad[True]["all"]["scope_m7"]["success"] = 131; assert not all(ok for _, ok in verdict_scope(bad))      # 131 - 118 = 13 < 14
    ls = summarize_long(R["lr"]["rows"]); ls["repair_m6"]["exact_share"] = 0.80; assert not all(ok for _, ok in verdict_scope_long(ls, summarize_long(R["ln"]["rows"])))


def test_the_page_quotes_the_numbers_and_the_pre_registration_is_unchanged():
    res = page()
    if res is None: return
    assert RESULT_MARKER in res
    pre = res[:res.index(RESULT_MARKER)]
    assert hashlib.sha256(pre.rstrip("\n").encode()).hexdigest() == PRE_REGISTRATION_SHA256        # nothing above the Result line was edited after the run
    assert "## Predictions (mine, written first)" in pre and "## Reading rules (fixed now; the notebook prints exactly these)" in pre
    body = res[res.index(RESULT_MARKER):]
    for needle in ("All 13 reading-rule lines came out \"yes\" and all 6 predictions held", "from 118 to 134 correct graphs out of 140", "ahead by 16 tasks, 11.4 points", "98.3 % of steps", "73.1 %",
                   "0 of 280 single-step outputs", "0 of 1,200 recorded policy steps", "**helped 16, harmed 0**", "both right 118, both wrong 6", "80 items: 77 on delete_leaf, 3 on delete_dep",
                   "79 of the 80 were real edges", "1 was an edge that does not exist", "none of the 80 was part of the correct change", "5 of 20 delete_leaf", "72 tokens on average for a request whose correct answer is 7",
                   "0 of 60 per regime", "80 of 140 overall", "5 without the rule, 6 with it", "In 9 of the 11", "three of them", "263 / 360 = **73.1 %**", "354 / 360 = **98.3 %**", "237 / 240 = 98.8 %",
                   "11 of 12", "1 of 12", "1 of 8", "exact on 91 steps", "no step where M6 was exact", "Lead 25.3 points", "24 steps (51 items: delete_leaf 13 steps, delete_dep 8, add_edge 3); 23 of those 24",
                   "seed 7007, step 18", "(0,0) to (0,3)", "(1,2)", "seed 7002, step 28", "`DEL[0,2]`", "2 items off) rather than a wrong delete (3 items off)", "16 against 14 needed", "only 1.9 points under the line",
                   "3,975 tokens", "2,040", "1.9 times", "317 s", "193 s", "72.2 tokens against 7.0", "0.91 s against 4.91 s", "11 microseconds", "2.27 s", "135 of 140 (193 s)", "134 of 140 (317 s)",
                   "ticked for this setup, including unrequested changes", "Not ticked: requests in ordinary words"):
        assert needle in body, needle


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
