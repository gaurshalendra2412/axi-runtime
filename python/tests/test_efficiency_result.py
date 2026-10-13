"""Implements concept-map item 89 (rows 3, 9, 27): the RESULT CHECK of docs/EFFICIENCY_MEASURES.md. Every number in the page's Result section is recomputed from the three saved files in docs/results/:
  1. the files are the ones the page names (sha256), and they carry the metadata of the pre-registered run (400 ARC tasks at the named commit, 140 graphs, seed 7);
  2. transport: the facts that do not depend on the machine (kind, nodes, edges, text length, JSON size, frame size) are recomputed from `make_scope_tasks(140, 7)` and must match row for row; the summary is
     the median of the rows; the model-call reference is the median of the saved hosted-model run; the four reading-rule lines read yes / NO / NO / yes on the saved summary;
  3. ARC: the summary is recomputed from the 3,004 saved grid rows and the 400 task rows; every row obeys its own arithmetic (plain length = cells + rows - 1, lit cells at most all cells, 80 percent
     shorter means five times the sparse length at most the plain length); the nine reading-rule lines read as the page says; sparse text is never shorter once one cell in five is lit; the exploratory file
     has the same grids, never more lit cells than the colour-0 file, and the same arithmetic;
  4. the predictions the page lists as held do hold on the files (10: median empty share, mean distinct raw prompts, median text path);
  5. every figure quoted in the Result section is the figure computed from the files.
What it does NOT show: that the timings repeat on another machine (only orders of magnitude are read), or anything about tokens of a real tokenizer, accuracy, or the Rust code."""
import hashlib, json, os, statistics, sys
sys.path.insert(0, os.path.dirname(__file__))
from axi.experiments import efficiency as E
from axi.experiments.agent_loop import _serialize
from axi.experiments.scope_tasks import make_scope_tasks

HERE = os.path.dirname(__file__); ROOT = os.path.join(HERE, "..", "..")
RES = os.path.join(ROOT, "docs", "results"); PAGE = os.path.join(ROOT, "docs", "EFFICIENCY_MEASURES.md")
F = {"t": "axi_efficiency_transport.json", "a": "axi_efficiency_arc.json", "m": "axi_efficiency_arc_modal.json"}
RESULT_MARK = "\n\n---\n\n# Result"
load = lambda k: json.load(open(os.path.join(RES, F[k])))
rt = lambda x: json.loads(json.dumps(x))
c = lambda x: f"{x:,}"
pct = lambda x, n: f"{100 * x / n:.1f} %"


def result_text():
    text = open(PAGE).read(); assert text.count(RESULT_MARK) == 1; return text[text.index(RESULT_MARK):]


def test_files_are_the_ones_the_page_names_and_carry_the_preregistered_metadata():
    res = result_text()
    for k, name in F.items():
        h = hashlib.sha256(open(os.path.join(RES, name), "rb").read()).hexdigest(); assert h in res and f"docs/results/{name}" in res, name
    t, a, m = load("t"), load("a"), load("m")
    assert t["meta"]["test"] == "efficiency_transport" and t["meta"]["seed"] == 7 and t["meta"]["n"] == 140 and t["meta"]["inner"] == 200 and t["meta"]["repeats"] == 5
    assert t["meta"]["model_run"] == "axi_gemini_single_norule_gemini-3.5-flash-lite_seed7.json"
    assert a["meta"]["test"] == "efficiency_arc" and a["meta"]["arc_commit"] == E.ARC_COMMIT and a["meta"]["n_tasks"] == 400 == E.N_ARC_TASKS
    assert m["meta"]["test"] == "efficiency_arc_modal" and m["meta"]["exploratory"] is True and m["meta"]["arc_commit"] == E.ARC_COMMIT


def test_transport_rows_match_the_tasks_and_the_summary_is_their_median():
    t = load("t"); tasks = make_scope_tasks(140, 7); rows = t["rows"]; assert len(rows) == 140
    for r, task in zip(rows, tasks):
        g = task.graph
        assert (r["kind"], r["nodes"], r["edges"]) == (task.kind, len(g.nodes), len(g.edges)) and r["text_chars"] == len(_serialize(g))
        assert r["json_bytes"] == len(json.dumps(E.graph_dict(g))) and r["frame_bytes"] == len(E.wire_frame(g))
        assert all(isinstance(r[k], (int, float)) and r[k] > 0 for k in ("text_ns", "json_ns", "wire_total_ns", "wire_decode_ns", "filter_ns")) and r["wire_decode_ns"] <= r["wire_total_ns"]
    model_ns = E.model_call_ns(os.path.join(RES, t["meta"]["model_run"])); assert t["summary"]["model_call_ns"] == model_ns
    assert rt(E.summarize_transport(rows, model_ns)) == t["summary"]


def test_transport_reading_rules_and_the_page_quotes():
    t = load("t"); s = t["summary"]; assert [ok for _, ok in E.verdict_transport(s)] == [True, False, False, True]
    res = result_text(); share = 100 * (s["text_ns"] + s["json_ns"]) / s["model_call_ns"]
    for q in (f"Prompt text {c(s['text_ns'])} ns", f"JSON dumps plus loads {c(s['json_ns'])} ns", f"decode) {c(s['wire_total_ns'])} ns", f"decoding alone {c(s['wire_decode_ns'])} ns",
              f"`repair_m6`) {c(s['filter_ns'])} ns", f"{c(s['model_call_ns'])} ns, about {s['model_call_ns'] / 1e9:.2f} s",
              f"Text plus JSON together are {share:.4f} percent of one model call", f"would shorten it by {share:.4f} percent", f"bounds the transport at {share:.4f} percent of a step",
              f"{s['text_chars']} characters of text, {s['json_bytes']} bytes of JSON, {s['frame_bytes']} bytes of frame",
              f"is {s['wire_total_ns'] / s['json_ns']:.1f} times slower than the JSON round trip", f"decode is {s['wire_decode_ns'] / 350:.0f} times the paste's 350 ns",
              f"{s['text_ns'] / 1e6:.3f} ms and {s['json_ns'] / 1e6:.3f} ms", f"about {c(int(round(s['model_call_ns'] / (s['text_ns'] + s['json_ns']), -2)))} times smaller"):
        assert q in res, q
    assert "Transport: yes, NO, NO, yes." in res


def _arithmetic(grids):
    for r in grids:
        cells = r["h"] * r["w"]
        assert r["dense"] == cells + r["h"] - 1 and 0 <= r["nonzero"] <= cells and r["lossless"] is True and r["sparse"] > 0
        if 5 * r["nonzero"] >= cells: assert r["sparse"] >= r["dense"], r             # one cell in five lit: never shorter


def test_arc_summary_is_recomputed_from_the_rows_and_every_row_obeys_its_arithmetic():
    a = load("a"); grids, tasks = a["grids"], a["tasks"]; assert len(grids) == 3004 and len(tasks) == 400 and len({t["task"] for t in tasks}) == 400
    assert rt(E.summarize_arc(grids, tasks)) == a["summary"]; _arithmetic(grids)
    per_task = {}
    for r in grids: per_task.setdefault(r["task"], []).append(r["role"])
    assert set(per_task) == {t["task"] for t in tasks}
    for roles in per_task.values():
        n = sum(1 for x in roles if x.endswith("_in")) - 1; assert len(roles) == 2 * n + 1 and "test_in" in roles and len(set(roles)) == len(roles)        # pairs, plus the test input
    assert all(1 <= t["raw_distinct"] <= 8 and t["canon_distinct"] == 1 for t in tasks)


def test_arc_reading_rules_and_the_predictions_that_held():
    a = load("a"); s = a["summary"]; assert [ok for _, ok in E.verdict_arc(s)] == [True, True, False, False, False]
    assert 0.5 <= s["median_empty_share"] <= 0.8 and s["raw_distinct_sum"] / s["tasks"] >= 7.5
    t = load("t")["summary"]; assert 5_000 <= t["text_ns"] <= 200_000
    assert 4 * s["ge80"] < s["grids"] and 2 * s["sparse_shorter"] < s["grids"] and s["total_sparse"] > s["total_dense"]
    res = result_text(); assert "ARC: yes, yes, NO, NO, NO." in res


def test_arc_page_quotes():
    a = load("a")["summary"]; res = result_text(); lo = list(a["bands"].items())
    below20 = sum(e["n"] for k, e in lo[:2]); short20 = sum(e["sparse_shorter"] for k, e in lo[:2]); at20 = a["grids"] - below20; assert sum(e["sparse_shorter"] for k, e in lo[2:]) == 0
    for q in (f"restores {c(a['lossless'])} of {c(a['grids'])} grids exactly", f"on {a['canon_identical']} of {a['tasks']} tasks", f"give {a['raw_distinct_sum'] / a['tasks']:.2f} different prompts",
              f"The median grid is {a['median_empty_share']:.3f} empty", f"on {c(a['sparse_shorter'])} of {c(a['grids'])} grids ({pct(a['sparse_shorter'], a['grids'])})",
              f"or more on {c(a['ge50'])} ({pct(a['ge50'], a['grids'])})", f"or more on {c(a['ge80'])} ({pct(a['ge80'], a['grids'])})**",
              f"is {a['total_sparse'] / a['total_dense']:.2f} times the plain text ({c(a['total_sparse'])} against {c(a['total_dense'])} characters)",
              f"Of the {c(below20)} grids with under 20 percent of cells non-empty, the sparse text is shorter on {c(short20)}; of the {c(at20)} grids"):
        assert q in res, q


def test_exploratory_file_is_the_same_grids_with_the_most_frequent_colour_left_out():
    a, m = load("a"), load("m"); assert len(m["grids"]) == len(a["grids"]) and "tasks" not in m
    assert rt(E.summarize_arc(m["grids"], [])) == m["summary"]; _arithmetic(m["grids"])
    for ra, rm in zip(a["grids"], m["grids"]):
        assert (ra["task"], ra["role"], ra["h"], ra["w"], ra["dense"]) == (rm["task"], rm["role"], rm["h"], rm["w"], rm["dense"])
        assert rm["nonzero"] <= ra["nonzero"] and 0 <= rm["background"] <= 9                                   # the most frequent colour covers at least as many cells as colour 0
    s = m["summary"]; res = result_text()
    for q in (f"shorter on {c(s['sparse_shorter'])} of {c(s['grids'])} grids ({pct(s['sparse_shorter'], s['grids'])})", f"or more on {c(s['ge80'])} ({pct(s['ge80'], s['grids'])})",
              f"and {s['total_sparse'] / s['total_dense']:.2f} times the plain text in total", f"Lossless on {c(s['lossless'])} of {c(s['grids'])}"):
        assert q in res, q
    assert s["sparse_shorter"] * 4 < s["grids"] and s["total_sparse"] > s["total_dense"]


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("PASS", n)
    print(sum(1 for n in globals() if n.startswith("test_")), "tests passed")
