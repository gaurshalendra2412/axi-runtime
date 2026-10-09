"""Implements concept-map row 9 (structure against scale; test 1 of section 3), RESULT of the 7B run of 8 Oct 2026. Page: docs/SCALE_TEST_7B.md (prediction and reading rule written before the run).
The two result files are docs/results/axi_Qwen2.5-7B-Instruct_seed3_{norule,rule}.json: Qwen2.5-7B-Instruct in 4-bit, the 60 tasks of make_tasks(60, seed 3), Colab T4, greedy, 400 new tokens.
This file does not run a model. It recomputes from the saved rows (not from their summaries) the numbers the page quotes:
  1. the files are the two regimes of the same 60 tasks, in the task order of make_tasks, with the expected settings and a summary that equals the rows;
  2. the pre-registered reading: blind apply corrupts 30 of 30 delete_dep tasks without the rule, so the verdict printed by the notebook is 'did not fix it'; and every gated mode
     corrupts 0 of 60 graphs in both regimes (the second result fixed in advance);
  3. the structured-v1 retry collapse (30 percent at 3B, 0 at 7B): in 30 of 30 retries the reply has no node-delete line, the gate admits it and the service stays; the imperative wording
     (which says 'Keep DEL[r,c]') is 30 of 30;
  4. the rule regime (not predicted): on the 12 service-without-edges tasks none of the 7B texts is the ideal delta, the gate admits 7 whose extra deletes name real edges, and all 16 misses of
     M6 are proposals that delete a real edge nobody asked to delete (none of the 44 hits is);
  5. the post-hoc scope filter (an idea found after reading the outputs, so not evidence): plain M6 replayed offline equals the recorded one, the filter gives 60 of 60 and drops 46 deletes;
  6. the tables printed in the page equal the files.
What it does NOT show: that a bigger model never fixes this (one family, one seed, 4-bit against the fp16 3B), or that the scope filter works on anything but the outputs it was derived from."""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
from colab.colab_scale_run import verdict, MODE_LABEL
from axi.experiments.agent_loop import MODES, make_tasks, strict_parse, copy_graph, same_graph, feedback_text
from axi.engine.gate import check, apply
from axi.engine.diagnostics import ground_and_complete, diagnose
from axi.engine.cico_parser import StateTransitionDelta

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "..", "..", "docs", "results")
GATED = [m for m in MODES if m not in ("M0_unconstrained", "M1_constrained")]


def load():
    paths = {h: os.path.join(RESULTS, f"axi_Qwen2.5-7B-Instruct_seed3_{'rule' if h else 'norule'}.json") for h in (False, True)}
    if not all(os.path.exists(p) for p in paths.values()): return None            # docs/results not shipped with this copy of the repo
    return {h: json.load(open(p)) for h, p in paths.items()}


def lines(text): return re.findall(r"(?:ADD|DEL)\[[^\]]*\]", text)


def real_edge_lines(task): return {f"DEL[({u[0]},{u[1]})->({v[0]},{v[1]}):{rel}#{w}]" for (u, v, rel), w in task.graph.edges.items()}


def extras(row, task): return [e for e in lines(row["M1_constrained"]["text"]) if e not in set(lines(task.ideal))]


def test_the_files_are_the_two_regimes_of_the_same_60_tasks_and_the_summary_equals_the_rows():
    R = load()
    if R is None: return
    tasks = make_tasks(60, 3)
    for hint, res in R.items():
        m = res["meta"]
        assert (m["model"], m["load_4bit"], m["seed"], m["n"], m["max_new_tokens"], m["hint_rules"]) == ("Qwen/Qwen2.5-7B-Instruct", True, 3, 60, 400, hint)
        assert [r["kind"] for r in res["rows"]] == [t.kind for t in tasks]
        kinds = [r["kind"] for r in res["rows"]]
        assert (kinds.count("delete_dep"), kinds.count("delete_leaf"), kinds.count("add_node"), kinds.count("add_edge")) == (30, 12, 6, 12)
        for scope, rows in (("all", res["rows"]), ("delete_dep", [r for r in res["rows"] if r["kind"] == "delete_dep"])):
            for mode in MODES:
                s = res["summary"][scope][mode]
                assert abs(s["success"] - sum(r[mode]["success"] for r in rows) / len(rows)) < 1e-9, (hint, scope, mode)
                assert abs(s["corrupted"] - sum(r[mode]["corrupted"] for r in rows) / len(rows)) < 1e-9, (hint, scope, mode)
        assert all(r["M0_unconstrained"]["text"] == r["M1_constrained"]["text"] and r["M1_constrained"]["strict_parse"] for r in res["rows"])   # greedy: the grammar changed nothing, every text parses


def test_the_pre_registered_reading_blind_apply_corrupts_30_of_30_and_no_gated_mode_ever_corrupts():
    R = load()
    if R is None: return
    dd = [r for r in R[False]["rows"] if r["kind"] == "delete_dep"]
    assert len(dd) == 30 and sum(r["M1_constrained"]["corrupted"] for r in dd) == 30 and sum(r["M1_constrained"]["success"] for r in dd) == 0
    assert all(re.fullmatch(r"DEL\[\d+,\d+\]", r["M1_constrained"]["text"]) and r["M1_constrained"]["tokens"] == 7 for r in dd)      # the node alone, the incident edges forgotten
    text = verdict(R)
    assert "Scale alone did not fix it" in text and "corrupts 100%" in text and "Worst corruption in any gated mode: 0%" in text
    for hint in (False, True):
        for mode in GATED: assert sum(r[mode]["corrupted"] for r in R[hint]["rows"]) == 0, (hint, mode)          # 14 cells, all 0
    assert sum(r["M1_constrained"]["corrupted"] for r in R[True]["rows"] if r["kind"] == "delete_dep") == 19     # rule stated: 63.3 percent, still above half


def test_structured_v1_collapses_because_the_reply_drops_the_service_delete_and_the_imperative_wording_keeps_it():
    R = load()
    if R is None: return
    tasks = make_tasks(60, 3); rows = R[False]["rows"]
    node_line = re.compile(r"DEL\[\d+,\d+\]")
    for variant, ok in (("structured", False), ("structured_imperative", True), ("prose_detailed", True), ("prose_plain", None)):
        key = f"M3_retry_{variant}"
        dd = [r for r in rows if r["kind"] == "delete_dep"]
        assert all(r[key]["retried"] for r in dd)
        if ok is False:
            assert not any(node_line.search(r[key]["text"]) for r in dd)                                        # 30 of 30 replies list edges only
            assert all(not r[key]["rejected"] and not r[key]["success"] for r in dd)                            # the gate admits them; the service stays
        if ok is True: assert all(node_line.search(r[key]["text"]) and r[key]["success"] for r in dd)
    assert sum(r["M3_retry_prose_plain"]["success"] for r in rows if r["kind"] == "delete_dep") == 3
    # the wording is the cause: the imperative feedback names the node delete to keep, the v1 feedback does not
    for r, t in zip(rows, tasks):
        if r["kind"] != "delete_dep": continue
        obs = diagnose(t.graph, strict_parse(r["M1_constrained"]["text"]))
        assert "Keep DEL[" in feedback_text("structured_imperative", obs) and "Keep DEL[" not in feedback_text("structured", obs)
        break


def test_with_the_rule_stated_the_7b_over_applies_it_the_gate_admits_real_extra_deletes_and_these_are_all_of_the_m6_misses():
    R = load()
    if R is None: return
    tasks = make_tasks(60, 3); rows = R[True]["rows"]; plain = R[False]["rows"]
    leaf = [(r, t) for r, t in zip(rows, tasks) if t.kind == "delete_leaf"]
    assert len(leaf) == 12
    assert all(lines(r["M1_constrained"]["text"]) == lines(t.ideal) for r, t in zip(plain, tasks) if t.kind == "delete_leaf")   # no rule: 12 of 12 exact
    assert not any(lines(r["M1_constrained"]["text"]) == lines(t.ideal) for r, t in leaf) and all(extras(r, t) for r, t in leaf)     # rule stated: none exact, all carry extra deletes
    assert sum(r["M1_constrained"]["success"] for r, t in leaf) == 3 and sum(r["M1_constrained"]["corrupted"] for r, t in leaf) == 0   # right only by accident, wrong 9, none corrupted
    assert sum(r["M2_gate"]["rejected"] for r, t in leaf) == 5
    admitted_wrong = [(r, t) for r, t in leaf if not r["M2_gate"]["rejected"] and not r["M2_gate"]["success"]]
    assert len(admitted_wrong) == 7 and all(any(e in real_edge_lines(t) for e in extras(r, t)) for r, t in admitted_wrong)       # the extras name real edges of other services
    n_extra = [e for r, t in zip(rows, tasks) if t.kind in ("delete_dep", "delete_leaf") for e in [(r, t, x) for x in extras(r, t)]]
    assert len(n_extra) == 66
    def target(t): return tuple(int(x) for x in re.search(r"\((\d+),(\d+)\)", t.instruction).groups())
    def touches(e, n): return n in [(int(a), int(b)) for a, b in re.findall(r"\((\d+),(\d+)\)", e)] or e.startswith("DEL[%d,%d]" % n)
    assert sum(1 for r, t, e in n_extra if not touches(e, target(t))) == 46
    # M6: the misses are exactly the proposals with a real, unrequested edge delete
    hit = [bool(r["M6_ground_complete"]["success"]) for r in rows]
    has_real = [any(e in real_edge_lines(t) for e in extras(r, t)) for r, t in zip(rows, tasks)]
    assert sum(hit) == 44 and sum(has_real) == 16 and all(h != x for h, x in zip(hit, has_real))
    assert sum(1 for h, (r, t) in zip(hit, zip(rows, tasks)) if not h and t.kind == "delete_leaf") == 9 and sum(1 for h, (r, t) in zip(hit, zip(rows, tasks)) if not h and t.kind == "delete_dep") == 7
    assert all(r["M6_ground_complete"]["success"] for r in plain)                                              # no rule: M6 is 60 of 60


def test_post_hoc_the_scope_filter_turns_the_saved_outputs_into_60_of_60_and_this_proves_nothing_beyond_them():
    R = load()
    if R is None: return
    tasks = make_tasks(60, 3)
    def replay(rows, scope):
        ok, dropped = 0, 0
        for r, t in zip(rows, tasks):
            d = strict_parse(r["M1_constrained"]["text"])
            if scope:
                gone = {(x.r, x.c) for x in d.node_deltas if x.op == "DEL"}
                keep = [e for e in d.edge_deltas if not (e.op == "DEL" and e.u not in gone and e.v not in gone)]
                dropped += len(d.edge_deltas) - len(keep); d = StateTransitionDelta(d.node_deltas, keep)
            final = copy_graph(t.graph)
            if check(t.graph, d).ok: apply(final, d)
            else:
                g = ground_and_complete(t.graph, d, 4)
                if g is not None and g.repairable and check(t.graph, g.repaired).ok: apply(final, g.repaired)
            ok += same_graph(final, t.expected)
        return ok, dropped
    for hint in (False, True):
        rows = R[hint]["rows"]
        assert replay(rows, False)[0] == sum(r["M6_ground_complete"]["success"] for r in rows)                 # the offline replay is faithful: 60 and 44
    assert replay(R[False]["rows"], True) == (60, 0) and replay(R[True]["rows"], True) == (60, 46)
    # why it cannot be counted as a result: in this task family every correct delta deletes only edges of a service it also deletes
    assert all(set(lines(t.ideal)[1:]) <= real_edge_lines(t) and all(touches_node(e, t) for e in lines(t.ideal)[1:]) for t in tasks if t.kind == "delete_dep")


def touches_node(e, t):
    n = tuple(int(x) for x in re.search(r"\((\d+),(\d+)\)", t.instruction).groups())
    return n in [(int(a), int(b)) for a, b in re.findall(r"\((\d+),(\d+)\)", e)]


def test_the_tables_in_the_result_section_of_the_page_equal_the_files():
    R = load()
    page = os.path.join(HERE, "..", "..", "docs", "SCALE_TEST_7B.md")
    if R is None or not os.path.exists(page): return
    text = open(page).read(); text = text[text.index("# Result (run 8 Oct 2026"):]
    parts = text.split("**Rule stated in the prompt**")
    assert len(parts) == 2
    for hint, chunk in ((False, parts[0]), (True, parts[1])):
        found = {}
        for line in chunk.splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) == 6 and cells[1].endswith("%"): found[cells[0]] = [float(c.replace("%", "").strip()) for c in cells[1:]]
        assert len(found) == 9
        S = R[hint]["summary"]
        for m in MODES:
            row = found[MODE_LABEL[m]]
            assert row[2] == round(S["all"][m]["success"] * 100, 1) and row[3] == round(S["delete_dep"][m]["success"] * 100, 1) and row[4] == round(S["delete_dep"][m]["corrupted"] * 100, 1), (hint, m)
    assert "30 of 30 tasks corrupted" in text and "73.3 % all" in text and "all 16 M6 misses" in text


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
