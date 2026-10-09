"""Implements concept-map row 3 / 9 / 27 (structure against scale), item 85: RESULT of the bigger-model test, Qwen2.5-14B, files received 9 Oct 2026. Page: docs/SCALE_TEST_14B.md
(prediction and reading rules written before the run, an addendum about the Colab restart, and the result appended below them).
The four files are docs/results/axi_14b_{single_rule, single_norule, long_rule, long_norule}_Qwen2.5-14B_seed7.json; the 7B files of docs/results are read for the comparison.
This file does not run a model. It recomputes from the saved rows (not from their summaries):
  1. the files are what the page says (model, regime, seed, counts, request kinds equal make_scope_tasks(140, 7) and the oracle scripts of seeds 7000 to 7005 / 7000 to 7003), no output is unparsed, every
     saved summary equals the summary recomputed from the rows, and the bytes are the ones whose sha256 the page lists;
  2. Part 1: every cell of the page's table, corrupted counts, helped / harmed, the 14B-against-7B contrasts task by task, and the gain table (integer arithmetic, then rounded);
  3. Part 1 details: what the filter dropped (16 tasks, 37 items, 34 real edges, 3 not in the graph, none in the correct change), the one no-rule task it changed, the 11 delete_weight misses, delete_pair texts;
  4. Part 2: the held state can be rebuilt from the recorded texts for all three policies; exact steps, off episodes, corrupted steps, blind apply's first corrupted step; scope_m7 exact where M6 was not and
     never the reverse; the 14B against the 7B step by step on the same episodes;
  5. cost: tokens and model seconds, the ratio to the 7B, the filter's time per request;
  6. the notebook's reading rules: 15 lines, 14 yes, the one no is the line the page names; the lines flip when a number is doctored;
  7. the page quotes these numbers, and the text above the Result line is byte for byte what was written before the run (two hashes: the pre-registration, and the addendum).
What it does NOT show: other model families, sizes above 14B, requests in ordinary words, sampling, or that the two Colab sessions would give the same text (not tested)."""
import copy, hashlib, json, os, re, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.cico_parser import CICOParser
from axi.engine.scope import admit_scoped
from axi.experiments.agent_loop import strict_parse
from axi.experiments.bigger_model import verdict_big_single, verdict_big_long
from axi.experiments.long_horizon import oracle_script, summarize_long
from axi.experiments.scope_tasks import make_scope_tasks
from axi.experiments.scope_test import summarize_scope, evaluate
from test_scope_result import replay

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "..", "..", "docs", "results")
PAGE = os.path.join(HERE, "..", "..", "docs", "SCALE_TEST_14B.md")
NAMES = {"sr": "single_rule", "sn": "single_norule", "lr": "long_rule", "ln": "long_norule"}
SHA = {"sr": "ed4bf1ee1fa664e70cfac656a1a35e77d1ebd192434933bd58446d691d04aad8", "sn": "93cec487b72b8c43617a2a947c8959bea870ee571ac14d5228e9b1f6a79bf2fd",
       "lr": "48667bca5dee84bb9e558d1cf7c7adaa1e0b7e43693e22de5b9fe659890f96c6", "ln": "4428ee1dd06e3279aae9764aa94022a7144c78448fd364e220ef97ec950922ff"}
PREREG_SHA256 = "e51b2a8b44941b03b7df6e727029bb650f97a223cc639d5010e0b802c471c29b"                # the page as delivered in update 25, before the addendum (text + the final newline)
ADDENDUM_SHA256 = "4f900b5cbe71166e1be5de1d51c0cf3e75a6fd825142c5d990b2721e35ffbc4f"              # the addendum text (after its heading separator, with the final newline) as it was when the page ended there
ADDENDUM_MARKER = "\n\n---\n\n# Addendum (9 Oct 2026)"
RESULT_MARKER = "\n\n---\n\n# Result (files received 9 Oct 2026"
P5 = ("blind", "gate", "repair_m6", "scope_m7", "scope_delta")


def load(prefix, model):
    paths = {k: os.path.join(RESULTS, f"axi_{prefix}_{n}_{model}_seed7.json") for k, n in NAMES.items()}
    if not all(os.path.exists(p) for p in paths.values()): return None                            # docs/results not shipped with this copy of the repo
    return {k: json.load(open(p)) for k, p in paths.items()}


def r14(): return load("14b", "Qwen2.5-14B")
def r7(): return load("scope", "Qwen2.5-7B")


def page():
    return open(PAGE).read() if os.path.exists(PAGE) else None


TASKS = None
def tasks():
    global TASKS
    if TASKS is None: TASKS = make_scope_tasks(140, 7)
    return TASKS


def pct(a, n): return round(100 * a / n, 1)


def test_the_files_are_what_the_page_says_and_the_summaries_equal_the_rows():
    R = r14()
    if R is None: return
    for k, hint in (("sr", True), ("sn", False)):
        m = R[k]["meta"]; rows = R[k]["rows"]
        assert (m["model"], m["seed"], m["n"], m["hint_rules"], m["test"]) == ("Qwen/Qwen2.5-14B-Instruct", 7, 140, hint, "scope_single_step"), k
        assert [r["kind"] for r in rows] == [t.kind for t in tasks()] and len(rows) == 140
        assert all(strict_parse(r["text"]) is not None for r in rows)                           # nothing unparsed
        assert json.loads(json.dumps(summarize_scope(rows, tasks()))) == R[k]["summary"], k
    steps = 0
    for k, hint, eps in (("lr", True, 6), ("ln", False, 4)):
        m = R[k]["meta"]
        assert (m["model"], m["mode"], m["episodes"], m["steps"], m["seed"], m["hint_rules"], m["policies"], m["mix"]) == ("Qwen/Qwen2.5-14B-Instruct", "state", eps, 30, 7, hint, ["blind", "repair_m6", "scope_m7"], "scope"), k
        assert [r["seed"] for r in R[k]["rows"]] == [7000 + e for e in range(eps)]
        for row in R[k]["rows"]:
            _, script, _ = oracle_script(row["seed"], 30, "scope"); assert len(script) == 30
            assert row["stopped"] == {"blind": None, "repair_m6": None, "scope_m7": None}
            for p, log in row["policies"].items():
                assert [x["kind"] for x in log] == [s["kind"] for s in script], (row["seed"], p)
                assert all(x["parsed"] for x in log); steps += len(log)
        assert json.loads(json.dumps(summarize_long(R[k]["rows"]))) == R[k]["summary"], k
    assert steps == 900                                                                            # 0 of 900 recorded policy steps unparsed
    for k, n in NAMES.items():                                                                     # the bytes are the ones the page lists
        data = open(os.path.join(RESULTS, f"axi_14b_{n}_Qwen2.5-14B_seed7.json"), "rb").read(); assert hashlib.sha256(data).hexdigest() == SHA[k], k
    res = page()
    if res is not None: assert all(f"`{h}`" in res for h in SHA.values())
    bad = copy.deepcopy(R["lr"]); bad["rows"][0]["policies"]["scope_m7"][3]["distance"] += 1
    assert json.loads(json.dumps(summarize_long(bad["rows"]))) != bad["summary"]                  # a doctored row no longer matches its summary
    bad = copy.deepcopy(R["sr"]); bad["rows"][5]["text"] = "ADD[0,0:1]"
    assert json.loads(json.dumps(summarize_scope(bad["rows"], tasks()))) != bad["summary"]


def test_part_1_the_table_of_the_page_and_the_comparison_with_the_7b():
    R, Q = r14(), r7()
    if R is None or Q is None: return
    exp = {  # kind: (no rule cells, rule cells), each in the order blind, gate, repair_m6, scope_m7, scope_delta
        "delete_dep": ((11, 10, 19, 20, 20), (17, 16, 18, 20, 20)), "delete_leaf": ((20, 17, 20, 20, 20), (6, 0, 6, 20, 20)),
        "delete_edge": ((20, 20, 20, 20, 0), (20, 20, 20, 20, 0)), "delete_pair": ((4, 4, 20, 20, 0), (4, 4, 20, 20, 0)),
        "delete_weight": ((16, 16, 16, 16, 0), (13, 13, 13, 13, 0)), "add_node": ((20,) * 5, (20,) * 5), "add_edge": ((20,) * 5, (20,) * 5),
        "old_deletes": ((31, 27, 39, 40, 40), (23, 16, 24, 40, 40)), "all": ((111, 107, 135, 136, 80), (100, 93, 117, 133, 80))}
    S = {k: summarize_scope(R[k]["rows"], tasks()) for k in ("sn", "sr")}; S7 = {k: summarize_scope(Q[k]["rows"], tasks()) for k in ("sn", "sr")}
    for g, (nr, ru) in exp.items():
        for key, want in (("sn", nr), ("sr", ru)):
            s = S[key][g]; assert tuple(s[p]["success"] for p in P5) == want, (g, key)
            assert s["n"] == (40 if g == "old_deletes" else 140 if g == "all" else 20)
    assert tuple(S7["sn"]["all"][p]["success"] for p in P5) == (99, 99, 135, 135, 80) and tuple(S7["sr"]["all"][p]["success"] for p in P5) == (94, 87, 118, 134, 80)
    for key, blind, others in (("sn", 24, 0), ("sr", 17, 0)):
        assert S[key]["all"]["blind"]["corrupted"] == blind and all(S[key]["all"][p]["corrupted"] == others for p in P5[1:]), key
    assert (S7["sn"]["all"]["blind"]["corrupted"], S7["sr"]["all"]["blind"]["corrupted"]) == (36, 24)
    assert sum(S[k]["all"][p]["corrupted"] for k in S for p in ("gate", "repair_m6", "scope_m7")) == 0 and 3 * 140 * 2 == 840        # 0 of 840 single-request graphs
    assert S["sn"]["paired"] == dict(helped=1, harmed=0, both_right=135, both_wrong=4) and S["sr"]["paired"] == dict(helped=16, harmed=0, both_right=117, both_wrong=7)
    assert S["sn"]["m7_equals_m6"]["all"] == 139 and S["sr"]["m7_equals_m6"]["all"] == 124
    assert S["sn"]["dropped"] == dict(tasks_with_drops=2, items=3, tasks_with_drops_and_right=2) and S["sr"]["dropped"] == dict(tasks_with_drops=16, items=37, tasks_with_drops_and_right=16)
    assert sum(S[k][g]["scope_delta"]["success"] for k in S for g in ("delete_edge", "delete_pair", "delete_weight")) == 0
    # the raw 14B against the raw 7B, task by task
    for key, want in (("sn", (98, 13, 1, 28)), ("sr", (87, 13, 7, 33))):
        a = [x["blind"]["success"] for x in evaluate(R[key]["rows"], tasks())]; b = [x["blind"]["success"] for x in evaluate(Q[key]["rows"], tasks())]
        assert (sum(x and y for x, y in zip(a, b)), sum(x and not y for x, y in zip(a, b)), sum(y and not x for x, y in zip(a, b)), sum(not x and not y for x, y in zip(a, b))) == want, key
    # the gain table, integer counts first
    gains = {"m6 over blind, no rule": (135 - 111, 135 - 99), "m7 over blind, rule": (133 - 100, 134 - 94), "m7 over m6, rule": (133 - 117, 134 - 118)}
    assert gains == {"m6 over blind, no rule": (24, 36), "m7 over blind, rule": (33, 40), "m7 over m6, rule": (16, 16)}
    assert [pct(x, 140) for x in (36, 24, 40, 33, 16)] == [25.7, 17.1, 28.6, 23.6, 11.4] and [pct(x, 180) for x in (56, 28)] == [31.1, 15.6] and pct(112, 120) == 93.3
    assert 10 * (135 - 111) >= 140 and 10 * (133 - 100) >= 140 and 111 - 99 > 0 and 100 - 94 > 0
    assert 11 > 10 and 2 * 11 > 20                                                                   # the one no: 11 of 20 against "10 or fewer"


def _ideal_texts(t):
    ideal = CICOParser.parse_transition_delta(t.ideal)
    return {f"{e.op}[({e.u[0]},{e.u[1]})->({e.v[0]},{e.v[1]}):{e.relation}#{e.weight}]" for e in ideal.edge_deltas} | {f"{x.op}[{x.r},{x.c}]" for x in ideal.node_deltas if x.op == "DEL"} \
           | {f"{x.op}[{x.r},{x.c}:{x.val}]" for x in ideal.node_deltas if x.op == "ADD"}


def _dropped(rows):
    out = {}
    for i, (r, t) in enumerate(zip(rows, tasks())):
        res = admit_scoped(t.graph, strict_parse(r["text"]), t.instruction, "named")
        if res.dropped: out[i] = res.dropped
    return out


def test_part_1_what_the_filter_dropped_what_it_changed_without_the_rule_and_where_it_missed():
    R, Q = r14(), r7()
    if R is None or Q is None: return
    rows = R["sr"]["rows"]; dr = _dropped(rows); items = real = phantom = needed = 0; by_task, by_item = {}, {}
    for i, drops in dr.items():
        t = tasks()[i]; by_task[t.kind] = by_task.get(t.kind, 0) + 1
        for reason, txt in drops:
            assert reason == "edge removed that the request does not name and that no deleted named service owns"
            items += 1; by_item[t.kind] = by_item.get(t.kind, 0) + 1; needed += txt in _ideal_texts(t)
            m = re.match(r"DEL\[\((\d+),(\d+)\)->\((\d+),(\d+)\):(\w+)#(\d+)\]", txt); k = ((int(m[1]), int(m[2])), (int(m[3]), int(m[4])), m[5])
            real += k in t.graph.edges; phantom += k not in t.graph.edges
    assert (len(dr), items, real, phantom, needed, by_task, by_item) == (16, 37, 34, 3, 0, {"delete_leaf": 14, "delete_dep": 2}, {"delete_leaf": 34, "delete_dep": 3})
    assert len(set(dr) & set(_dropped(Q["sr"]["rows"]))) == 13 and len(_dropped(Q["sr"]["rows"])) == 16          # as many tasks as the 7B, 13 of them the same
    leaf = [i for i, t in enumerate(tasks()) if t.kind == "delete_leaf"]
    n14 = [len(strict_parse(rows[i]["text"]).edge_deltas) for i in leaf]; n7 = [len(strict_parse(Q["sr"]["rows"][i]["text"]).edge_deltas) for i in leaf]
    assert (len(leaf), sum(x > 0 for x in n14), sum(n14), sum(x > 0 for x in n7), sum(n7)) == (20, 20, 49, 20, 87)                  # every delete_leaf got edge deletions, 49 against 87
    # no rule: the one task the filter changed
    sn = R["sn"]["rows"]; ev = evaluate(sn, tasks()); helped = [i for i, x in enumerate(ev) if x["scope_m7"]["success"] and not x["repair_m6"]["success"]]
    assert helped == [56] and tasks()[56].instruction == "Delete the service at (2,3)." and tasks()[56].kind == "delete_dep"
    d0 = _dropped(sn); assert sorted(d0) == [56, 126] and [txt for _, txt in d0[56]] == ["DEL[(2,0)->(2,1):owns#9]"]
    assert [txt for _, txt in d0[126]] == ["ADD[(0,2)->(1,1):owns#8]", "ADD[(0,2)->(1,2):dep#6]"] and ev[126]["repair_m6"]["success"] and ev[126]["scope_m7"]["success"]
    assert _dropped(Q["sn"]["rows"]) == {}                                                                                           # the 7B's 140 no-rule outputs: nothing dropped
    # the misses: all 11 are delete_weight; each deleted a real edge that is not in the correct answer
    tot = other = 0; missed = {}
    for key in ("sn", "sr"):
        e = evaluate(R[key]["rows"], tasks()); missed[key] = {i for i, x in enumerate(e) if not x["scope_m7"]["success"]}
        assert {e[i]["kind"] for i in missed[key]} == {"delete_weight"}
        for i in missed[key]:
            t = tasks()[i]; d = strict_parse(R[key]["rows"][i]["text"]); tot += 1
            exp = {(x.u, x.v, x.relation) for x in CICOParser.parse_transition_delta(t.ideal).edge_deltas}; got = {(x.u, x.v, x.relation) for x in d.edge_deltas}
            other += bool([k for k in got if k in t.graph.edges and k not in exp])
    assert (sorted(missed["sn"]), sorted(missed["sr"]), tot, other) == ([32, 67, 109, 123], [11, 18, 25, 95, 109, 123, 137], 11, 11)
    m7 = {key: {i for i, x in enumerate(evaluate(Q[key]["rows"], tasks())) if not x["scope_m7"]["success"]} for key in ("sn", "sr")}
    assert (len(m7["sn"]), len(m7["sr"])) == (5, 6) and 109 in missed["sn"] & missed["sr"] & m7["sn"] & m7["sr"]
    assert sorted(missed["sr"] - m7["sr"]) == [11, 25, 95, 123, 137] and sorted(m7["sr"] - missed["sr"]) == [32, 53, 67, 102]
    for i in (25, 95, 123, 137):                                                                                                      # four wrote the right edges and then an extra real edge
        t = tasks()[i]; exp = {(x.u, x.v, x.relation, x.weight) for x in CICOParser.parse_transition_delta(t.ideal).edge_deltas}
        got = {(x.u, x.v, x.relation, x.weight) for x in strict_parse(R["sr"]["rows"][i]["text"]).edge_deltas}
        assert exp < got, i
    t = tasks()[11]; got = {(x.u, x.v, x.relation, x.weight) for x in strict_parse(R["sr"]["rows"][11]["text"]).edge_deltas}
    assert not got & {(x.u, x.v, x.relation, x.weight) for x in CICOParser.parse_transition_delta(t.ideal).edge_deltas}                # one listed the wrong edges
    # delete_pair: same wrong text as the 7B in 8 of 16 (no rule) and 16 of 16 (rule)
    for key, same in (("sn", 8), ("sr", 16)):
        e = evaluate(R[key]["rows"], tasks()); wrong = [i for i, t in enumerate(tasks()) if t.kind == "delete_pair" and not e[i]["blind"]["success"]]
        assert len(wrong) == 16 and sum(R[key]["rows"][i]["text"] == Q[key]["rows"][i]["text"] for i in wrong) == same, key


def test_part_2_the_held_state_can_be_rebuilt_and_the_numbers_of_the_page():
    R, Q = r14(), r7()
    if R is None or Q is None: return
    for key in ("lr", "ln"):
        for row in R[key]["rows"]:
            for p in row["policies"]:
                for rec, st, before, after, orc, df in replay(row, p):
                    assert (len(df), after.is_well_formed(), not df) == (rec["distance"], not rec["corrupted"], rec["exact"]), (key, row["seed"], p, rec["t"])
                    assert len(after.nodes) == rec["nodes"] and len(after.edges) == rec["edges"]
                    if p == "scope_m7": assert rec["dropped"] == len(admit_scoped(before, strict_parse(rec["text"]), st["instruction"], "named").dropped)
    want = {("lr", "blind"): (76, 6, 3, 76), ("lr", "repair_m6"): (152, 6, 3, 0), ("lr", "scope_m7"): (180, 0, 0, 0),
            ("ln", "blind"): (8, 4, 7, 112), ("ln", "repair_m6"): (120, 0, 0, 0), ("ln", "scope_m7"): (120, 0, 0, 0)}
    for (key, p), (exact, eps_off, maxd, corrupted) in want.items():
        steps = [x for r in R[key]["rows"] for x in r["policies"][p]]
        assert len(steps) == (180 if key == "lr" else 120) and sum(x["exact"] for x in steps) == exact, (key, p)
        assert sum(any(not x["exact"] for x in r["policies"][p]) for r in R[key]["rows"]) == eps_off, (key, p)
        assert max(x["distance"] for x in steps) == maxd and sum(x["corrupted"] for x in steps) == corrupted, (key, p)
    assert [pct(a, n) for a, n in ((76, 180), (152, 180), (180, 180), (8, 120), (120, 120))] == [42.2, 84.4, 100.0, 6.7, 100.0]
    for key in ("lr", "ln"):                                                                       # blind apply: corrupted in every episode, by step 10 in 4 of 6 and 4 of 4, first at step 1 to 3 in 6 of 10
        first = [next((x["t"] for x in r["policies"]["blind"] if x["corrupted"]), None) for r in R[key]["rows"]]
        assert all(f is not None for f in first) and sum(f <= 10 for f in first) == (4 if key == "lr" else 4), key
    first = [next(x["t"] for x in r["policies"]["blind"] if x["corrupted"]) for k in ("lr", "ln") for r in R[k]["rows"]]
    assert first == [2, 2, 3, 13, 21, 5, 2, 2, 1, 7] and sum(f <= 3 for f in first) == 6
    sm = {k: summarize_long(R[k]["rows"]) for k in ("lr", "ln")}
    assert round(sm["lr"]["blind"]["corrupted_by_step_10"], 3) == 0.667 and sm["ln"]["blind"]["corrupted_by_step_10"] == 1.0
    assert all(sm[k][p]["episodes_ever_corrupted"] == 0 for k in sm for p in ("repair_m6", "scope_m7")) and all(sm[k]["blind"]["episodes_ever_corrupted"] == 1.0 for k in sm)
    # step by step, rule regime: scope_m7 exact where M6 was not and never the reverse; the filter's drops
    ahead = behind = drops = drops_exact = items = 0; kinds = {}
    for row in R["lr"]["rows"]:
        for a, b in zip(row["policies"]["repair_m6"], row["policies"]["scope_m7"]):
            ahead += b["exact"] and not a["exact"]; behind += a["exact"] and not b["exact"]
            if b["dropped"]: drops += 1; drops_exact += b["exact"]; items += b["dropped"]; kinds[b["kind"]] = kinds.get(b["kind"], 0) + 1
    assert (ahead, behind, drops, drops_exact, items, kinds) == (28, 0, 8, 8, 13, {"delete_leaf": 7, "delete_dep": 1})
    ahead = behind = drops = 0
    for row in R["ln"]["rows"]:
        for a, b in zip(row["policies"]["repair_m6"], row["policies"]["scope_m7"]):
            ahead += b["exact"] and not a["exact"]; behind += a["exact"] and not b["exact"]; drops += b["dropped"]
    assert (ahead, behind, drops) == (0, 0, 0)
    # against the 7B, same episodes, same steps
    for key, n, p, want in (("lr", 6, "repair_m6", (122, 30, 2, 26)), ("lr", 6, "scope_m7", (180, 0, 0, 0)), ("ln", 4, "repair_m6", (117, 3, 0, 0)), ("ln", 4, "scope_m7", (117, 3, 0, 0))):
        c = [0, 0, 0, 0]
        for a, b in zip(R[key]["rows"], Q[key]["rows"][:n]):
            assert a["seed"] == b["seed"]
            for x, y in zip(a["policies"][p], b["policies"][p]):
                c[0] += x["exact"] and y["exact"]; c[1] += x["exact"] and not y["exact"]; c[2] += y["exact"] and not x["exact"]; c[3] += (not x["exact"]) and not y["exact"]
        assert tuple(c) == want, (key, p)
    assert 124 + 56 == 180 and 152 + 28 == 180 and pct(124, 180) == 68.9 and 117 + 3 == 120
    # the one event of the 7B in the no-rule episodes, and what the 14B wrote at the same step
    r7_ = [r for r in Q["ln"]["rows"] if r["seed"] == 7002][0]["policies"]["scope_m7"][27]; r14_ = [r for r in R["ln"]["rows"] if r["seed"] == 7002][0]["policies"]["scope_m7"][27]
    assert (r7_["text"], r7_["exact"], r14_["text"], r14_["exact"]) == ("DEL[0,2]", False, "DEL[1,2]\nDEL[(2,2)->(1,2):dep#5]", True)
    _, script, _ = oracle_script(7002, 30, "scope"); assert script[27]["instruction"] == "Delete the service at (1,2)."


def test_cost_numbers_of_the_page():
    R, Q = r14(), r7()
    if R is None or Q is None: return
    tok = {k: sum(r["tokens"] for r in R[k]["rows"]) for k in ("sn", "sr")}; sec = {k: sum(r["seconds"] for r in R[k]["rows"]) for k in ("sn", "sr")}
    tok7 = {k: sum(r["tokens"] for r in Q[k]["rows"]) for k in ("sn", "sr")}; sec7 = {k: sum(r["seconds"] for r in Q[k]["rows"]) for k in ("sn", "sr")}
    assert (tok["sn"], tok["sr"], tok7["sn"], tok7["sr"]) == (2685, 3510, 2040, 3975) and round(tok["sr"] / tok["sn"], 1) == 1.3 and round(tok7["sr"] / tok7["sn"], 1) == 1.9
    assert (round(sec["sn"]), round(sec["sr"]), round(sec7["sn"]), round(sec7["sr"])) == (448, 539, 193, 317)
    assert round(sec["sr"] / sec["sn"], 1) == 1.2 and round(sec7["sr"] / sec7["sn"], 1) == 1.6
    assert (round(sec["sn"] / 140, 2), round(sec["sr"] / 140, 2), round(sec7["sn"] / 140, 2), round(sec7["sr"] / 140, 2)) == (3.2, 3.85, 1.38, 2.27)
    assert round(sec["sn"] / sec7["sn"], 1) == 2.3 and round(sec["sr"] / sec7["sr"], 1) == 1.7 and all(1.5 <= x <= 3 for x in (sec["sn"] / sec7["sn"], sec["sr"] / sec7["sr"]))      # prediction 9
    for key, (t_, s_) in (("sn", (11.5, 2.36)), ("sr", (43.8, 5.86))):
        rs = [r for r in R[key]["rows"] if r["kind"] == "delete_leaf"]
        assert round(sum(r["tokens"] for r in rs) / len(rs), 1) == t_ and round(sum(r["seconds"] for r in rs) / len(rs), 2) == s_, key
    ds = [(t, strict_parse(r["text"])) for r, t in zip(R["sr"]["rows"], tasks())]
    t0 = time.perf_counter()
    for t, d in ds: admit_scoped(t.graph, d, t.instruction, "named")
    per = (time.perf_counter() - t0) / len(ds)
    assert per < 0.005                                                                           # the page says about 13 microseconds; the bound is loose on purpose (machines differ)
    S = {k: summarize_scope(R[k]["rows"], tasks()) for k in ("sn", "sr")}
    assert S["sn"]["all"]["scope_m7"]["success"] == 136 and S["sn"]["all"]["repair_m6"]["success"] == 135 and S["sr"]["all"]["scope_m7"]["success"] == 133 and sec["sn"] < sec["sr"]


def test_the_reading_rules_come_out_as_the_page_says():
    R = r14()
    if R is None: return
    S = {True: summarize_scope(R["sr"]["rows"], tasks()), False: summarize_scope(R["sn"]["rows"], tasks())}
    ls_r, ls_n = summarize_long(R["lr"]["rows"]), summarize_long(R["ln"]["rows"])
    v1, v2 = verdict_big_single(S), verdict_big_long(ls_r, ls_n)
    assert len(v1) == 8 and len(v2) == 7
    no = [t for t, ok in v1 + v2 if not ok]
    assert len(no) == 1 and no[0].startswith("[need] no rule: blind apply is right on at most half of the 20 delete_dep"), no
    assert all(ok for t, ok in v1 + v2 if t.startswith("[guard]")) and sum(t.startswith("[guard]") for t, _ in v1 + v2) == 10
    assert sum(ok for t, ok in v1 + v2 if t.startswith("[need]")) == 4 and sum(t.startswith("[need]") for t, _ in v1 + v2) == 5
    # the lines are not vacuous: doctor one number at a time, and look at the one line that should change
    def line(vs, start):
        hit = [ok for t, ok in vs if t.startswith(start)]; assert len(hit) == 1, start; return hit[0]
    bad = copy.deepcopy(S); bad[False]["delete_dep"]["blind"]["success"] = 10; assert line(verdict_big_single(bad), "[need] no rule: blind apply is right on at most half")          # 10 of 20 would have been a yes: the miss is exactly one task
    bad = copy.deepcopy(S); bad[True]["old_deletes"]["scope_m7"]["success"] = 35; assert not line(verdict_big_single(bad), "[guard] rule stated: scope_m7 solves at least 90")
    bad = copy.deepcopy(S); bad[True]["old_deletes"]["scope_m7"]["success"] = 36; assert line(verdict_big_single(bad), "[guard] rule stated: scope_m7 solves at least 90")
    bad = copy.deepcopy(S)
    for p in ("repair_m6", "scope_m7"): bad[False]["all"][p]["success"] = 125                  # both at 125: only the "at least 90 percent" line should fall (126 is the bar)
    assert not line(verdict_big_single(bad), "[guard] no rule: repair_m6 solves at least 90") and line(verdict_big_single(bad), "[guard] no rule: scope_m7 is within 3")
    for p in ("repair_m6", "scope_m7"): bad[False]["all"][p]["success"] = 126
    assert line(verdict_big_single(bad), "[guard] no rule: repair_m6 solves at least 90")
    bad = copy.deepcopy(ls_r); bad["scope_m7"]["exact_share"] = 0.89; assert not line(verdict_big_long(bad, ls_n), "[guard] rule stated: scope_m7 holds the oracle's state on at least 90")
    bad = copy.deepcopy(ls_n)
    for p in ("repair_m6", "scope_m7"): bad[p]["exact_share"] = 0.948                         # both just under 95 percent and equal: only the "each at least 95 percent" line should fall
    assert not line(verdict_big_long(ls_r, bad), "[guard] no rule: repair_m6 and scope_m7 each hold the oracle's state on at least 95") and line(verdict_big_long(ls_r, bad), "[guard] no rule: the two are within 3")
    for p in ("repair_m6", "scope_m7"): bad[p]["exact_share"] = 0.95
    assert line(verdict_big_long(ls_r, bad), "[guard] no rule: repair_m6 and scope_m7 each hold the oracle's state on at least 95")
    bad = copy.deepcopy(ls_n); bad["blind"]["corrupted_by_step_10"] = 0.25; assert not line(verdict_big_long(ls_r, bad), "[need] no rule: blind apply is corrupted by step 10")
    bad = copy.deepcopy(ls_r); bad["blind"]["corrupted_by_step_10"] = 0.5; assert line(verdict_big_long(bad, ls_n), "[need] rule stated: blind apply is corrupted by step 10")
    bad = copy.deepcopy(ls_r); bad["blind"]["corrupted_by_step_10"] = 1 / 3; assert not line(verdict_big_long(bad, ls_n), "[need] rule stated: blind apply is corrupted by step 10")


def test_the_page_quotes_the_numbers_and_the_text_above_the_result_is_unchanged():
    res = page()
    if res is None: return
    assert RESULT_MARKER in res and res.count(RESULT_MARKER) == 1 and ADDENDUM_MARKER in res
    head, rest = res.split(RESULT_MARKER)[0], res[res.index(RESULT_MARKER):]
    pre, addendum = head.split(ADDENDUM_MARKER)
    assert hashlib.sha256((pre + "\n").encode()).hexdigest() == PREREG_SHA256, "the pre-registration must not change"
    assert hashlib.sha256((addendum + "\n").encode()).hexdigest() == ADDENDUM_SHA256, "the addendum must not change"
    assert "## Predictions (mine, written first; the number is how sure I am)" in pre and "## Reading rules (fixed now; the notebook prints exactly these)" in pre
    for needle in ("All 10 [guard] lines came out \"yes\". 4 of the 5 [need] lines came out \"yes\".", "The one \"no\" missed by a single task", "11 of the 20 delete_dep requests", "0 of 280 single-step outputs, 0 of 900 recorded policy steps",
                   "24 of 140 single requests without the rule and 17 of 140 with it", "36 tasks (25.7 points) | 24 tasks (17.1 points)", "40 tasks (28.6 points) | 33 tasks (23.6 points)", "16 tasks (11.4 points) | 16 tasks (11.4 points)",
                   "56 of 180 steps (31.1 points) | 28 of 180 steps (15.6 points)", "112 of 120 steps (93.3 points)", "8 without the rule and all 16 with it", "as many tasks as the 7B did (16, 13 of them the same tasks)",
                   "99 / 99 / 135 / 135 / 80", "94 / 87 / 118 / 134 / 80", "| delete_dep | 11 / 10 / 19 / 20 / 20 | 17 / 16 / 18 / 20 / 20 |", "| delete_leaf | 20 / 17 / 20 / 20 / 20 | 6 / 0 / 6 / 20 / 20 |", "| delete_edge (lone edge delete) | 20 / 20 / 20 / 20 / **0** | 20 / 20 / 20 / 20 / **0** |",
                   "| delete_pair (service delete plus a separate edge delete) | 4 / 4 / 20 / 20 / **0** | 4 / 4 / 20 / 20 / **0** |", "| delete_weight (names no service) | 16 / 16 / 16 / 16 / 0 | 13 / 13 / 13 / 13 / 0 |",
                   "| add_node | 20 / 20 / 20 / 20 / 20 | 20 / 20 / 20 / 20 / 20 |", "| add_edge | 20 / 20 / 20 / 20 / 20 | 20 / 20 / 20 / 20 / 20 |", "| **old deletes (delete_dep + delete_leaf, n = 40)** | 31 / 27 / 39 / 40 / 40 | 23 / 16 / **24** / **40** / 40 |",
                   "| **all (n = 140)** | 111 / 107 / 135 / 136 / 80 | 100 / 93 / **117** / **133** / 80 |", "98 right in both, 13 right only for the 14B, 1 only for the 7B, 28 right in neither (111 against 99)",
                   "87 / 13 / 7 / 33 (100 against 94)", "scope_m7 helped 1, harmed 0", "both right 135, both wrong 4", "\"Delete the service at (2,3).\"", "the edge (2,0) to (2,1)", "scope_m7 helped 16, harmed 0", "both right 117, both wrong 7",
                   "49 in all; the 7B 87", "37 items; the 7B 80 items on 16 tasks", "14 delete_leaf tasks (34 items) and 2 delete_dep tasks (3 items)", "34 of the 37 were real edges of the graph, 3 were edges that do not exist, and none of the 37",
                   "M6 was right on 6 of 20 delete_leaf requests (7B 5)", "16 and 16", "All 11 misses of scope_m7 (4 without the rule, 7 with it)", "(the 7B also had 11, 5 and 6)", "Task 109", "missed five requests the 7B got right", "the 7B four that the 14B got right",
                   "| rule stated, blind apply | 6 | 76 / 180 = 42.2 % | 6 of 6 | 3 | 76 |", "| rule stated, repair_m6 | 6 | 152 / 180 = **84.4 %** | 6 of 6 | 3 | 0 |", "| rule stated, scope_m7 | 6 | 180 / 180 = **100 %** | 0 of 6 | 0 | 0 |",
                   "| no rule, blind apply | 4 | 8 / 120 = 6.7 % | 4 of 4 | 7 | 112 |", "| no rule, repair_m6 | 4 | 120 / 120 = 100 % | 0 of 4 | 0 | 0 |", "| no rule, scope_m7 | 4 | 120 / 120 = 100 % | 0 of 4 | 0 | 0 |",
                   "rule stated, repair_m6 124 / 180 = 68.9 %", "117 / 120 for both", "by step 10 in 4 of 6 (rule stated) and 4 of 4 (no rule)", "step 1 to 3 in 6 of the 10 episodes", "scope_m7 was exact on 28 steps where M6 was not",
                   "The filter dropped something at 8 steps (13 items: delete_leaf 7 steps, delete_dep 1); all 8 steps were exact", "exact on 30 steps where the 7B's M6 was not and the reverse on 2 steps (both exact on 122, neither on 26)",
                   "from 56 steps to 28", "seed 7002, step 28", "`DEL[1,2]`", "missed** (11 of 20", "(135 of 140 = 96.4 %)", "(17.1)", "(40 of 40)", "(23.6)", "(0 and 0)", "(136 against 135)", "0 of 840 single-request graphs, 0 of 600 long steps",
                   "(111 against 99, 100 against 94)", "**missed** (16, equal)", "(4 of 4)", "(4 of 6)", "(15.6 ahead)", "3.20 s against 1.38 s = 2.3 times", "3.85 s against 2.27 s = 1.7 times", "about 13 microseconds against 3.85 s",
                   "Part 1 eight lines, seven yes and one no; Part 2 seven lines, all yes", "3,510 tokens", "2,685", "539 s of model time against 448 s", "43.8 tokens against 11.5 (5.86 s against 2.36 s per request)",
                   "136 of 140 (448 s)", "135 (448 s)", "133 of 140 (539 s)", "The one \"no\" is one task", "ten clean episodes cannot separate the two sizes", "Two sessions",
                   "ticked for this setup, including unrequested changes", "now on two sizes of one family"):
        assert needle in rest, needle
    assert "at most 10 of the 20" not in rest                                                      # the test_bigger_model count of that phrase (2, in the pre-registration) stays true


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
