"""Implements concept-map item 98: RESULT of the layout test of 15 Oct 2026. Page: docs/LAYOUT_TEST.md (predictions and reading rules written before the run; the result is appended below them).
The file is docs/results/axi_layout_Qwen2.5-7B_seed31.json (Colab T4, Qwen2.5-7B-Instruct 4-bit, seed 31, 50 problems read under 5 conditions = 250 reads, saved by `axi_tensor_tests.ipynb` Cell 8).
This file does not run a model. It recomputes from the saved rows (not from their summaries):
  1. the file is what the page says (model, seed, counts, the problem ids and the fingerprint of the problem set equal what the code builds from seed 31, 42 delete problems and 8 controls, all five conditions in every row) and
     the saved summary equals the summary recomputed from the rows, by the code and by an independent count here;
  2. every reply is scored again from its saved text with `kv_layout.reply_metrics` and the scoring functions of `kv_drop.score_reply` (parse, gate, blind, repair_m6) on the graph the code rebuilds from the seed:
     the saved scores are the ones the text gives;
  3. the thirteen reading rules: seven yes (1, 2, 3, 4, 6, 12, 13), six NO (5, 7, 8, 9, 10, 11), the headline (5, 7, 8) all NO; the table of the page (probability and reading of every line) equals the probabilities written
     before the run and the recomputed readings; a tampered row changes the readings;
  4. the table of the page (exact as written, with the repair step, cut off, recall and its two halves, invented, other, prompt, seconds) and the paired comparisons against `sorted` (counts, sign tests);
  5. the looks taken afterwards, which the page says are NOT pre-registered: the paired differences with a bootstrap interval, what the repair step does with unparsed and parsed replies, the edges written with the arrow
     turned around;
  6. the part of the page above the Result line is byte for byte what was written before the run (hash), and the Result quotes these figures.
What it does NOT show: that any layout is better (the looks afterwards are reasons to test again), other models, sampling, bigger graphs, anything about the position codes of the shared text."""
import copy, hashlib, json, os, random, re, sys
sys.path.insert(0, os.path.dirname(__file__))
import axi.experiments.kv_layout as Y
from axi.engine.gate import check
from axi.experiments.agent_loop import same_graph, strict_parse
from axi.experiments.cascade import next_state

HERE = os.path.dirname(__file__)
PAGE = os.path.join(HERE, "..", "..", "docs", "LAYOUT_TEST.md")
FILE = os.path.join(HERE, "..", "..", "docs", "results", "axi_layout_Qwen2.5-7B_seed31.json")
RESULT_MARKER = "\n\n---\n\n# Result (run 15 Oct 2026"
PAGE_SHA256 = "c9c3cd72018428d55338b2f350b3605a4fc50a7091319badbbbe439ed2f142cc"       # the whole page as it was before the run (one newline at the end)
CONDS = ("sorted", "shuffled", "grouped", "adjacency", "rot90")
PROBS = [85, 85, 99, 85, 65, 92, 75, 55, 35, 70, 60, 80, 95]
READS = [True, True, True, True, False, True, False, False, False, False, False, True, True]
_cache = {}


def _load():
    if "d" not in _cache: _cache["d"] = json.load(open(FILE))
    return _cache["d"]


def _problems():
    if "ps" not in _cache: _cache["ps"] = {p["id"]: p for p in Y.make_problems()}
    return _cache["ps"]


def _dep(rows=None): return [r for r in (rows or _load()["rows"]) if r["kind"] == "delete_dep"]
def _ctl(rows=None): return [r for r in (rows or _load()["rows"]) if r["kind"] != "delete_dep"]
def _mean(xs):
    xs = [x for x in xs if x is not None]; return sum(xs) / len(xs) if xs else None


def _summ(rows):
    """The summary of the printout, counted again here from the per-reply fields (independent of `summarize_layout`)."""
    out = {}
    for c in CONDS:
        dep = [r["conds"][c] for r in _dep(rows)]; ctl = [r["conds"][c] for r in _ctl(rows)]; both = dep + ctl
        out[c] = dict(n_delete=len(dep), n_control=len(ctl), exact_blind=sum(1 for x in dep if x["exact_blind"]) / len(dep), exact_m6=sum(1 for x in dep if x["exact_m6"]) / len(dep), recall=_mean(x["recall"] for x in dep),
                      recall_out=_mean(x["recall_out"] for x in dep), recall_in=_mean(x["recall_in"] for x in dep), invented=_mean(x["invented"] for x in dep), other=_mean(x["other"] for x in dep), ran_on=sum(1 for x in dep if not x["ended"]) / len(dep),
                      control_exact=sum(1 for x in ctl if x["exact_blind"]) / len(ctl), prompt_tokens=_mean(x["prompt_tokens"] for x in both), canonical=_mean(1.0 if x["canonical"] else 0.0 for x in both), seconds=_mean(x["seconds"] for x in both))
    return out


def _page():
    text = open(PAGE).read(); assert text.count(RESULT_MARKER) == 1, "the page must carry exactly one Result"
    i = text.index(RESULT_MARKER); return text[:i], text[i:]


def _lines(S):
    """The thirteen reading rules, counted again here from the independent summary."""
    so, sh, gr, ad, ro = (S[c] for c in CONDS)
    return [so["prompt_tokens"] >= 2000, all(S[c]["canonical"] == 1.0 for c in CONDS), all(S[c]["n_delete"] == 42 and S[c]["n_control"] == 8 for c in CONDS), 0.15 <= so["recall"] <= 0.90, so["recall"] - sh["recall"] >= 0.10 - 1e-12,
            so["recall_out"] - so["recall_in"] >= 0.10 - 1e-12, ad["recall"] - so["recall"] >= 0.15 - 1e-12, ad["exact_blind"] - so["exact_blind"] >= 0.15 - 1e-12, gr["recall"] - so["recall"] >= 0.05 - 1e-12,
            abs(ro["recall"] - so["recall"]) <= 0.10 + 1e-12, ad["invented"] <= 0.7 * so["invented"] + 1e-12, all(S[c]["control_exact"] >= 0.85 for c in CONDS),
            sum(1 for r in _dep() if r["conds"]["shuffled"]["text"] != r["conds"]["sorted"]["text"]) / 42 >= 0.20 - 1e-12]


def _boot(diffs, key):
    """A 95 % bootstrap interval of the mean, 4,000 resamples, drawn with the Mersenne Twister's `random()` only (the same on every Python 3)."""
    rng = random.Random(key); n = len(diffs); bs = sorted(sum(diffs[int(rng.random() * n)] for _ in range(n)) / n for _ in range(4000)); return bs[100], bs[3899]


def _paired(c, m, rows=None):
    base = [r["conds"]["sorted"] for r in _dep(rows)]; x = [r["conds"][c] for r in _dep(rows)]
    diffs = [a[m] - b[m] for a, b in zip(x, base) if a[m] is not None and b[m] is not None]; better = sum(1 for z in diffs if z > 1e-12); worse = sum(1 for z in diffs if z < -1e-12); lo, hi = _boot(diffs, f"{c}/{m}")
    return dict(n=len(diffs), mean=sum(diffs) / len(diffs), lo=lo, hi=hi, better=better, worse=worse, p=Y.sign_test(better, worse))


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- 1. the file
def test_the_file_is_the_run_the_page_describes_and_its_saved_summary_is_the_recomputed_one():
    d = _load(); m = d["meta"]; assert set(d) == {"rows", "summary", "meta"}
    assert (m["model"], m["kind"], m["problems"], m["conditions"], m["seed"], m["nodes"], m["edges"], m["grid"], m["max_new"]) == ("Qwen/Qwen2.5-7B-Instruct", "kv_layout", 50, list(CONDS), 31, 70, 120, [10, 10], 120), m
    ps = Y.make_problems(); assert m["ids"] == [p["id"] for p in ps] == [r["id"] for r in d["rows"]] and m["fp"] == Y.fingerprint(ps) == "802ec2d3cdb6f584", "the problems of the file are the ones the code builds from seed 31"
    assert len(_dep()) == 42 and len(_ctl()) == 8 and sorted(r["kind"] for r in _ctl()) == ["add_edge"] * 4 + ["add_node"] * 4 and all(list(r["conds"]) == list(CONDS) for r in d["rows"])
    assert d["summary"] == Y.summarize_layout(d["rows"]), "the saved summary is the one the code computes from the rows"
    S = _summ(d["rows"])
    for c in CONDS:
        for k, v in S[c].items(): assert abs(v - d["summary"][c][k]) < 1e-9, (c, k)
    assert all(abs(S[c]["prompt_tokens"] - 2335.6) < 1e-9 for c in CONDS if c != "adjacency") and abs(S["adjacency"]["prompt_tokens"] - 3920.6) < 1e-9
    assert all(r["conds"][c]["ended"] == (len(r["conds"][c]["tokens"]) < 120) for r in d["rows"] for c in CONDS), "a reply ended exactly when it used fewer than the 120 tokens it was allowed"
    assert sum(1 for r in d["rows"] for c in CONDS if not r["conds"][c]["ended"]) == 44 and all(r["conds"][c]["ended"] for r in _ctl() for c in CONDS), "the 44 cut-off replies are all on delete problems"


def test_a_tampered_row_is_noticed_by_the_summary_and_the_lines():
    d = copy.deepcopy(_load()); rows = d["rows"]; assert Y.summarize_layout(rows) == _load()["summary"]
    wrong = [r for r in _dep(rows) if not r["conds"]["adjacency"]["exact_blind"]]
    for r in wrong[:7]: r["conds"]["adjacency"]["exact_blind"] = True
    s = Y.summarize_layout(rows); assert s["adjacency"]["exact_blind"] != _load()["summary"]["adjacency"]["exact_blind"] and abs(_summ(rows)["adjacency"]["exact_blind"] - s["adjacency"]["exact_blind"]) < 1e-12
    assert Y.verdict_layout(s)[7][1] is False and _lines(_summ(rows))[7] is False, "seven more exact replies for adjacency (11 of 42 against 9) still read line 8 NO"
    for r in wrong[7:11]: r["conds"]["adjacency"]["exact_blind"] = True
    assert Y.verdict_layout(Y.summarize_layout(rows))[7][1] is False, "15 of 42 against 9 is 14.3 points: still NO"
    wrong[11]["conds"]["adjacency"]["exact_blind"] = True; s = Y.summarize_layout(rows)
    assert Y.verdict_layout(s)[7][1] is True and _lines(_summ(rows))[7] is True and Y.verdict_layout(_load()["summary"])[7][1] is False, "16 of 42 against 9 is 16.7 points: line 8 would have read yes"
    d = copy.deepcopy(_load()); rows = d["rows"]
    for r in _dep(rows): r["conds"]["shuffled"]["recall"] = 0.0
    assert Y.verdict_layout(Y.summarize_layout(rows))[4][1] is True and _lines(_summ(rows))[4] is True, "a shuffled recall of 0 would have read line 5 yes"


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- 2. every reply scored again
def test_every_saved_reply_scored_again_from_its_text_gives_the_saved_scores():
    ps = _problems(); n_meta = 0
    for r in _load()["rows"]:
        for c in CONDS:
            v = r["conds"][c]; g, task, items, p = Y.realize(ps[r["id"]], c); d = strict_parse(v["text"])
            got = dict(parsed=d is not None, admitted=bool(d is not None and check(g, d).ok), exact_blind=same_graph(next_state("blind", g, d, task.instruction), task.expected), exact_m6=same_graph(next_state("repair_m6", g, d, task.instruction), task.expected))
            assert {k: v[k] for k in got} == got, (r["id"], c)
            if r["kind"] == "delete_dep":
                m = Y.reply_metrics(v["text"], g, p["node"]); assert {k: v[k] for k in m} == m, (r["id"], c); n_meta += 1
            else: assert "recall" not in v
    assert n_meta == 210


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- 3. the thirteen lines
def test_the_thirteen_lines_seven_yes_six_no_the_headline_all_no_and_the_page_table_says_the_same():
    d = _load(); mine = _lines(_summ(d["rows"])); got = Y.verdict_layout(d["summary"]); assert mine == READS == [v for _, v in got] and [t for t, _ in got] == [t for _, t in Y.LAYOUT_LINES]
    assert Y.LAYOUT_HEADLINE == (5, 7, 8) and not any(READS[i - 1] for i in Y.LAYOUT_HEADLINE) and sum(READS) == 7
    head, res = _page(); table = re.findall(r"^\| (\d+) \| (\d+) % \| (yes|\*\*NO\*\*) \|", res, re.M); assert [int(a) for a, _, _ in table] == list(range(1, 14)) and [int(b) for _, b, _ in table] == PROBS
    assert [c == "yes" for _, _, c in table] == READS, "the page says what the rows say"
    probs = [int(m.group(1)) for m in re.finditer(r"\n\d+\. \[\w+\] .*\((\d+) %\)(?=\n)", head)]; assert probs == PROBS, "the probabilities of the table are the ones written before the run"
    s = d["summary"]; so = s["sorted"]
    assert abs(so["recall_out"] - so["recall_in"] - 0.4310) < 5e-4 and abs(s["shuffled"]["recall"] - so["recall"] + 0.0040) < 5e-4 and abs(s["adjacency"]["recall"] - so["recall"] - 0.0397) < 5e-4 and abs(s["grouped"]["recall"] - so["recall"] + 0.0317) < 5e-4
    assert abs(s["rot90"]["recall"] - so["recall"] - 0.1071) < 5e-4 and abs(s["adjacency"]["exact_blind"] - so["exact_blind"] + 0.1190) < 5e-4 and abs(s["adjacency"]["invented"] / so["invented"] - 1.652) < 1e-3
    assert sum(1 for r in _dep() if r["conds"]["shuffled"]["text"] != r["conds"]["sorted"]["text"]) == 37 and all(s[c]["control_exact"] == 1.0 for c in CONDS)


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- 4. the table and the paired comparisons
def test_the_table_of_the_page_and_the_sign_tests_against_sorted():
    d = _load(); rows = _dep(); head, res = _page(); exact = {c: sum(1 for r in rows if r["conds"][c]["exact_blind"]) for c in CONDS}; m6 = {c: sum(1 for r in rows if r["conds"][c]["exact_m6"]) for c in CONDS}
    ran = {c: sum(1 for r in rows if not r["conds"][c]["ended"]) for c in CONDS}
    assert exact == dict(sorted=9, shuffled=2, grouped=8, adjacency=4, rot90=8) and m6 == dict(sorted=33, shuffled=19, grouped=41, adjacency=35, rot90=26) and ran == dict(sorted=5, shuffled=21, grouped=0, adjacency=7, rot90=11)
    for c in CONDS:
        s = d["summary"][c]; line = (f"| `{c}` | {exact[c]} ({100 * exact[c] / 42:.1f} %) | {m6[c]} ({100 * m6[c] / 42:.1f} %) | {ran[c]} ({100 * ran[c] / 42:.0f} %) | {s['recall']:.3f} | {s['recall_out']:.3f} | {s['recall_in']:.3f} | {s['invented']:.2f} | {s['prompt_tokens']:,.0f} | {s['seconds']:.1f} |")
        assert line in res, line
    want = dict(shuffled=((1, 8), (3, 17)), grouped=((2, 3), (8, 0)), adjacency=((3, 8), (6, 4)), rot90=((3, 4), (5, 12))); base = [r["conds"]["sorted"] for r in rows]
    for c, ((g1, l1), (g2, l2)) in want.items():
        x = [r["conds"][c] for r in rows]
        for m, (gg, ll) in (("exact_blind", (g1, l1)), ("exact_m6", (g2, l2))):
            gain = sum(1 for a, b in zip(x, base) if a[m] and not b[m]); loss = sum(1 for a, b in zip(x, base) if b[m] and not a[m]); assert (gain, loss) == (gg, ll), (c, m); assert d["summary"][c][m + "_vs_sorted"] == dict(gain=gg, loss=ll, p=Y.sign_test(gg, ll))
    same = {c: sum(1 for r in rows if r["conds"][c]["text"] == r["conds"]["sorted"]["text"]) for c in CONDS[1:]}; assert same == dict(shuffled=5, grouped=12, adjacency=0, rot90=0)
    p = {c: {m: Y.sign_test(*w) for m, w in zip(("exact", "m6"), ws)} for c, ws in want.items()}
    assert abs(p["shuffled"]["exact"] - 0.0391) < 5e-5 and abs(p["shuffled"]["m6"] - 0.0026) < 5e-5 and abs(p["grouped"]["m6"] - 0.0078) < 5e-5 and abs(p["adjacency"]["exact"] - 0.2266) < 5e-5 and abs(p["adjacency"]["m6"] - 0.7539) < 5e-5 and abs(p["rot90"]["m6"] - 0.1435) < 5e-5
    for q in ("`shuffled` 1 / 8 (p = 0.04)", "`grouped` 2 / 3 (p = 1.0)", "`adjacency` 3 / 8 (p = 0.23)", "`rot90` 3 / 4 (p = 1.0)", "`shuffled` 3 / 17 (p = 0.003)", "`grouped` 8 / 0 (p = 0.008)", "`adjacency` 6 / 4 (p = 0.75)", "`rot90` 5 / 12 (p = 0.14)"): assert q in res, q
    assert abs(d["summary"]["adjacency"]["seconds"] / d["summary"]["sorted"]["seconds"] - 1.838) < 1e-3 and abs(d["summary"]["adjacency"]["prompt_tokens"] / d["summary"]["sorted"]["prompt_tokens"] - 1.679) < 1e-3
    assert round(d["summary"]["adjacency"]["peak_mb"] / 1000, 1) == 10.1 and round(d["summary"]["sorted"]["peak_mb"] / 1000, 1) == 7.3
    assert round(sum(v["seconds"] for r in d["rows"] for v in r["conds"].values()) / 60, 1) == 37.5, "the reads add up to 37.5 minutes (the counter said 2,254 s with the cell overhead: 37.6)"


def test_the_looks_taken_afterwards_the_paired_differences_with_their_intervals():
    want = {("shuffled", "recall"): (42, -0.004, 10, 8), ("shuffled", "recall_out"): (29, -0.006, 9, 9), ("shuffled", "recall_in"): (29, -0.006, 4, 2), ("grouped", "recall"): (42, -0.032, 4, 6), ("grouped", "recall_out"): (29, 0.017, 4, 4),
            ("grouped", "recall_in"): (29, -0.069, 1, 3), ("adjacency", "recall"): (42, 0.040, 17, 10), ("adjacency", "recall_out"): (29, -0.155, 5, 10), ("adjacency", "recall_in"): (29, 0.287, 14, 2), ("rot90", "recall"): (42, 0.107, 9, 5),
            ("rot90", "recall_out"): (29, 0.161, 8, 4), ("rot90", "recall_in"): (29, 0.000, 2, 2)}
    for (c, m), (n, mean, better, worse) in want.items():
        r = _paired(c, m); assert (r["n"], r["better"], r["worse"]) == (n, better, worse) and abs(r["mean"] - mean) < 5e-4, (c, m, r)
    a = _paired("adjacency", "recall_in"); assert abs(a["lo"] - 0.09) < 5.1e-3 and abs(a["hi"] - 0.48) < 5.1e-3 and abs(a["p"] - 0.0042) < 5e-5
    a = _paired("adjacency", "recall_out"); assert abs(a["lo"] + 0.39) < 5.1e-3 and abs(a["hi"] - 0.07) < 5.1e-3
    a = _paired("rot90", "recall"); assert abs(a["lo"] + 0.01) < 5.1e-3 and abs(a["hi"] - 0.23) < 5.1e-3 and abs(a["p"] - 0.424) < 5e-4
    head, res = _page()
    for q in ("+0.287, 95 % interval +0.09 to +0.48, 14 problems better and 2 worse, p = 0.004", "0.379 against 0.534 (-0.155, interval -0.39 to +0.07, 5 better and 10 worse)", "95 % interval -0.01 to +0.23, 9 problems better and 5 worse, p = 0.42",
              "p = 0.003 x 8 = 0.02", "which does NOT survive a correction for eight tests: 0.06"): assert q in res, q
    assert len(res.split("## What I read from the rows afterwards (NOT pre-registered")) == 2


def test_with_the_repair_step_the_outcome_follows_the_ending_and_the_edges_of_other_services():
    rows = _dep(); parsed = {c: sum(1 for r in rows if r["conds"][c]["parsed"]) for c in CONDS}; assert parsed == dict(sorted=37, shuffled=21, grouped=42, adjacency=35, rot90=31) and sum(42 - v for v in parsed.values()) == 44
    xs = [v for r in rows for v in r["conds"].values()]; assert sum(1 for v in xs if not v["parsed"]) == 44 and not any(v["exact_m6"] for v in xs if not v["parsed"]) and all(not v["parsed"] for v in xs if not v["ended"])
    ok = {c: sum(1 for r in rows if r["conds"][c]["parsed"] and r["conds"][c]["exact_m6"]) for c in CONDS}; assert ok == dict(sorted=33, shuffled=19, grouped=41, adjacency=35, rot90=26)
    left = [(c, r["id"], r["conds"][c]["other"]) for r in rows for c in CONDS if r["conds"][c]["parsed"] and not r["conds"][c]["exact_m6"]]; assert len(left) == 12 and all(o >= 1 for _, _, o in left)
    assert {c: sum(1 for x in left if x[0] == c) for c in CONDS} == dict(sorted=4, shuffled=2, grouped=1, adjacency=0, rot90=5)
    other = {c: sum(r["conds"][c]["other"] for r in rows) for c in CONDS}; inv = {c: sum(r["conds"][c]["invented"] for r in rows) for c in CONDS}
    assert other == dict(sorted=24, shuffled=11, grouped=1, adjacency=0, rot90=36) and inv == dict(sorted=69, shuffled=118, grouped=61, adjacency=114, rot90=87)
    gains = [r["id"] for r in rows if r["conds"]["grouped"]["exact_m6"] and not r["conds"]["sorted"]["exact_m6"]]; assert len(gains) == 8 and sum(1 for r in rows if r["id"] in gains and not r["conds"]["sorted"]["ended"]) == 4
    assert all(r["conds"]["sorted"]["other"] >= 1 for r in rows if r["id"] in gains and r["conds"]["sorted"]["ended"])
    head, res = _page()
    for q in ("None of the 44 replies that failed to parse", "33 of 37 for `sorted`, 41 of 42 for `grouped`, 35 of 35 for `adjacency`, 19 of 21 for `shuffled`, 26 of 31 for `rot90`", "Each of the 12 replies that parsed and were still wrong after the repair had deleted at least one real edge that belongs to another service",
              "4, 2, 1, 0 and 5 of them", "names 1 real edge that is not the service's against 24 for `sorted` (11 for `shuffled`, 36 for `rot90`, 0 for `adjacency`)", "Four of its eight gains are replies of `sorted` that ran into the limit"): assert q in res, q


def test_edges_written_with_the_arrow_turned_around():
    ps = _problems(); out = {}
    for c in CONDS:
        o = dict(own=0, exact=0, rev=0, out_n=0, out_rev=0, in_n=0, in_exact=0, in_rev=0); blind = []
        for r in _dep():
            g, task, items, p = Y.realize(ps[r["id"]], c); node = p["node"]; own = sorted(g.incident(node)); dels = Y.lenient_deleted_edges(r["conds"][c]["text"]); real = lambda e: g.edges.get((e[0], e[1], e[2])) == e[3]
            got = {(e[0], e[1], e[2]) for e in dels if real(e)}; rev = {(e[1], e[0], e[2]) for e in dels if not real(e) and g.edges.get((e[1], e[0], e[2])) == e[3]} & set(own)
            for k in own:
                ex = k in got; rv = (k in rev) and not ex; o["own"] += 1; o["exact"] += ex; o["rev"] += rv
                if k[0] == node: o["out_n"] += 1; o["out_rev"] += rv
                else: o["in_n"] += 1; o["in_exact"] += ex; o["in_rev"] += rv
            blind.append(sum((k in got) or (k in rev) for k in own) / len(own))
        o["blind"] = blind; out[c] = o
    assert {c: (o["own"], o["exact"], o["rev"], o["out_n"], o["out_rev"], o["in_n"], o["in_exact"], o["in_rev"]) for c, o in out.items()} == {
        "sorted": (101, 35, 13, 53, 12, 48, 4, 1), "shuffled": (101, 36, 1, 53, 1, 48, 6, 0), "grouped": (101, 32, 14, 53, 13, 48, 2, 1), "adjacency": (101, 39, 22, 53, 16, 48, 21, 6), "rot90": (101, 43, 8, 53, 5, 48, 4, 3)}
    assert [round((o["exact"] + o["rev"]) / o["own"], 3) for o in out.values()] == [0.475, 0.366, 0.455, 0.604, 0.505]
    diffs = [a - b for a, b in zip(out["adjacency"]["blind"], out["sorted"]["blind"])]; better = sum(1 for z in diffs if z > 1e-12); worse = sum(1 for z in diffs if z < -1e-12); lo, hi = _boot(diffs, "adjacency/blind")
    assert (better, worse) == (19, 8) and abs(sum(diffs) / 42 - 0.151) < 5e-4 and abs(Y.sign_test(better, worse) - 0.0522) < 5e-4 and abs(lo + 0.03) < 5.1e-3 and abs(hi - 0.33) < 5.1e-3
    head, res = _page()
    for q in ("13 of the 101 edges that hang on the 42 services in `sorted` were written that way (12 of the 53 edges that start at the service, 1 of the 48 that end at it), 1 in `shuffled`, 14 in `grouped`, **22 in `adjacency`** (16 and 6), 8 in `rot90`",
              "0.475, 0.366, 0.455, **0.604** and 0.505", "+0.151, 95 % interval -0.03 to +0.33, 19 problems better and 8 worse, p = 0.052", "backwards 16 times in 53", "(4 of 48 as written)"): assert q in res, q


def test_the_item_by_item_reader_on_a_made_up_reply():
    got = Y.lenient_deleted_edges("DEL[3,4] ADD[(0,0)->(0,1):dep#2] DEL[(0,0)->(0,1):dep#12]\nDEL[(1,1)->(2,2):owns#3] DEL[(5,5)->(")
    assert got == [((0, 0), (0, 1), "dep", 12), ((1, 1), (2, 2), "owns", 3)], "an ADD item and a node deletion are not edge deletions, a weight may have two digits, a cut-off item is not read"
    assert Y.lenient_deleted_edges("") == [] and Y.lenient_deleted_edges("   \n ") == []


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- 5. the page
def test_the_page_above_the_result_is_what_was_written_before_the_run_and_the_result_quotes_the_figures():
    head, res = _page(); assert hashlib.sha256((head.rstrip("\n") + "\n").encode()).hexdigest() == PAGE_SHA256, "the predictions and reading rules must not change after the run; later notes go below them"
    assert PAGE_SHA256 in res and "test 111" in res and "docs/results/axi_layout_Qwen2.5-7B_seed31.json" in res
    for q in ("The headline failed, on all three of its lines", "seven read yes (1, 2, 3, 4, 6, 12, 13) and six read NO (5, 7, 8, 9, 10, 11)", "about 27 % and my single number was 35 %", "37.6 minutes", "The block test (Cells 9 and 10) was not run",
              "the prompt of `sorted` held 2,336 tokens", "`shuffled` recall 0.329 against 0.333: lower by 0.004", "gap of 0.431", "`adjacency` recall 0.373 against 0.333: higher by 0.040", "4 of 42 replies (9.5 %), `sorted` on 9 of 42 (21.4 %): **11.9 points worse**",
              "`grouped` recall 0.302 against 0.333: lower by 0.032", "higher by 0.107, outside the 0.10 band by 0.007", "2.71 edges per reply, `sorted` 1.64: **65 % more**", "8 of 8 add replies", "37 of 42 delete problems (88 %)",
              "order-insensitive in the large and direction-sensitive in the small", "not supported here", "one frame is not eight", "NOT pre-registered", "reasons to test again, not as findings", "1.68 times the tokens (3,921 against 2,336)",
              "1.8 times the time per read (13.7 s against 7.5 s)", "10.1 GB of GPU memory against 7.3 GB", "I did not run `sorted` with the sentence", "`diagnostics.ground_and_complete`", "a pre-registered replication of `grouped`", "(sha256 `c9c3cd72"): assert q.lower() in res.lower(), q
    assert "[X]" not in res and "TODO" not in res and "The block test (`KV_BLOCK_TEST.md`, Cells 9 and 10) is still waiting" in res, "the block test is still to run and the Result says so"


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn): fn(); print("PASS", name); n += 1
    print(n, "tests passed")
