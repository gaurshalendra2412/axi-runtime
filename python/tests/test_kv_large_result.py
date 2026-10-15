"""Implements concept-map item 97: RESULT of the large KV-cache test of 15 Oct 2026. Page: docs/KV_LARGE_TEST.md (predictions and reading rules written before the run; the result is appended below them).
The two files are docs/results/axi_large_{a,c}_Qwen2.5-7B_seed21.json (Colab T4, Qwen2.5-7B-Instruct 4-bit, seed 21, 4 episodes of 40 steps, four arms). Run X was not made.
This file does not run a model. It recomputes from the saved rows (not from their summaries):
  1. the files are what the page says (model, seed, counts, the request kinds equal `kv_large.large_script` of seeds 21000 to 21003, the same script in run A and run C, no step unparsed or missing) and every saved
     summary equals the summary recomputed from the rows;
  2. the sixteen reading rules: fourteen yes, line 14 NO, line 16 not run; the table of the page (probability and reading of every line) equals the probabilities written before the run and the recomputed readings;
  3. every figure of run A the page quotes (tokens, ratios, seconds, identical replies, KL, stale share) and of run C (identical replies by arm and by kind, KL, repaired exactness, the paired comparison and its sign test);
  4. the scoring of the delete replies of run C with `kv_layout.reply_metrics` (recall of the service's edges, of the edges that start and end at it, invented edges, ghosts: none), the numbers the layout page quotes;
  5. the afterwards, not pre-registered comparison of the first and the last third (counts and Fisher's exact p), which the page says is a reason to test and not a finding;
  6. the page quotes these numbers, says run X was not made, and the part above the Result line is byte for byte what was written before the run (hash).
What it does NOT show: that the changed replies are worse (the rows cannot tell), other models, sampling, a longer state, run X."""
import copy, hashlib, json, os, re, sys
from fractions import Fraction
from math import comb
sys.path.insert(0, os.path.dirname(__file__))
import axi.experiments.kv_large as L
from axi.engine.cico_parser import CICOParser
from axi.experiments.kv_layout import lenient_deleted_edges, reply_metrics, sign_test

HERE = os.path.dirname(__file__)
PAGE = os.path.join(HERE, "..", "..", "docs", "KV_LARGE_TEST.md")
RES = os.path.join(HERE, "..", "..", "docs", "results")
RESULT_MARKER = "\n\n---\n\n# Result (run 15 Oct 2026"
PAGE_SHA256 = "999e01007464fddb2de4aac7acd533353589d14f3f4e9587ea9ade1424695aef"       # the whole page as it was before the run (one newline at the end)
ARMS = ("fresh", "exact", "drop_compact", "drop_gap")
SEEDS = (21000, 21001, 21002, 21003)
_cache = {}


def _load(k):
    if k not in _cache: _cache[k] = json.load(open(os.path.join(RES, f"axi_large_{k}_Qwen2.5-7B_seed21.json")))
    return _cache[k]


def _steps(k, arm): return [s for r in _load(k)["rows"] for s in r["arms"][arm]]


def _fisher(a, b, c, d):
    """Two-sided exact p for the 2 x 2 table [[a, b], [c, d]] (the sum of the tables not more likely than the observed one)."""
    n, r1, c1 = a + b + c + d, a + b, a + c; p = lambda x: Fraction(comb(r1, x) * comb(n - r1, c1 - x), comb(n, c1)); p0 = p(a)
    return float(sum((p(x) for x in range(max(0, c1 - (n - r1)), min(r1, c1) + 1) if p(x) <= p0), Fraction(0)))


def _page():
    text = open(PAGE).read(); assert text.count(RESULT_MARKER) == 1, "the page must carry exactly one Result"
    i = text.index(RESULT_MARKER); return text[:i], text[i:]


def _delete_replies(k, arm):
    """(metrics, text, step) of every 'delete a service with dependents' reply of `arm`, scored against the oracle's graph at that step, and the number of ghost items (an edge that existed at an earlier step, is
    gone from the graph now, and is listed by the reply)."""
    out = []
    for r in _load(k)["rows"]:
        g0, script, oracle = L.large_script(r["seed"], r["steps"]); past = set()
        for t, st in enumerate(script):
            live = oracle[t]; x = r["arms"][arm][t]
            if st["kind"] == "delete_dep":
                nd = CICOParser.parse_transition_delta(st["ideal"]).node_deltas[0]; m = reply_metrics(x["text"], live, (nd.r, nd.c))
                m["ghosts"] = sum(1 for e in lenient_deleted_edges(x["text"]) if (e[0], e[1], e[2]) not in live.edges and (e[0], e[1], e[2]) in past); out.append((m, x))
            past |= set(live.edges)
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- 1. the files
def test_the_files_are_the_two_runs_the_page_describes_and_their_saved_summaries_are_the_recomputed_ones():
    for k, rule in (("a", False), ("c", True)):
        d = _load(k); m = d["meta"]
        assert (m["model"], m["kind"], m["seed"], m["episodes"], m["steps"], m["hint_rules"], m["script"], m["max_new"]) == ("Qwen/Qwen2.5-7B-Instruct", "kv_drop", 21, 4, 40, rule, "large_120_200_14x14", 120), m
        assert m["arms"] == list(ARMS) and [r["seed"] for r in d["rows"]] == list(SEEDS) and all(r["steps"] == 40 and not any(r["stopped"].values() if isinstance(r["stopped"], dict) else r["stopped"]) for r in d["rows"])
        assert set(d) == {"rows", "summary", "meta"}
        for r in d["rows"]:
            g0, script, oracle = L.large_script(r["seed"], 40); assert len(script) == 40 and set(r["arms"]) == set(ARMS)
            for a in ARMS:
                xs = r["arms"][a]; assert len(xs) == 40 and [x["t"] for x in xs] == list(range(1, 41)) and [x["kind"] for x in xs] == [s["kind"] for s in script], (k, r["seed"], a)
                assert all(isinstance(x["text"], str) and x["text"] and x["parsed"] is not None for x in xs)
        s = L.summarize_large(d["rows"]); saved = d["summary"]; assert set(s) == set(saved)
        for a in s:
            for key, v in s[a].items():
                w = saved[a][key]; assert (v is None and w is None) or (v is not None and w is not None and (abs(v - w) < 1e-9 if isinstance(v, float) or isinstance(w, float) else v == w)), (k, a, key, v, w)
    kinds = lambda k: [x["kind"] for x in _steps(k, "fresh")]; assert kinds("a") == kinds("c")       # the same request script in both runs
    count = {}; [count.__setitem__(x, count.get(x, 0) + 1) for x in kinds("a")]; assert count == {"add_edge": 54, "add_node": 35, "delete_dep": 52, "delete_leaf": 19}, count


def test_a_tampered_row_is_noticed_by_the_summary_check():
    d = copy.deepcopy(_load("c")); d["rows"][0]["arms"]["drop_compact"][12]["same_as_fresh"] = not d["rows"][0]["arms"]["drop_compact"][12]["same_as_fresh"]
    assert L.summarize_large(d["rows"])["drop_compact"]["same"] != _load("c")["summary"]["drop_compact"]["same"]
    d = copy.deepcopy(_load("a")); [x.__setitem__("same_as_fresh", False) for x in d["rows"][3]["arms"]["drop_compact"][-3:]]
    v = L.verdict_large(L.summarize_large(d["rows"]), L.summarize_large(_load("c")["rows"]))
    assert v[4][1] is True and v[12][1] is False, "three late disagreements out of 160 (98.1 %) leave line 5 yes (97 %) and turn line 13 (95 % of the last third: 53 of 56) NO"


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- 2. the sixteen lines
def test_the_sixteen_lines_fourteen_yes_line_14_no_line_16_not_run_and_the_page_table_says_the_same():
    sa, sc = L.summarize_large(_load("a")["rows"]), L.summarize_large(_load("c")["rows"]); v = [x for _, x in L.verdict_large(sa, sc)]
    assert v == [True] * 13 + [False, True, None], v
    head, res = _page(); probs = [int(m) for m in re.findall(r"^\d+\. \[[a-z]+\] .*?\((\d+) %\)\s*$", head, re.M)]; assert len(probs) == 16, probs
    rows = re.findall(r"^\| (\d+) \| (\d+) % \| (\*\*)?(yes|NO|not run)(\*\*)? \|", res, re.M); assert [int(r[0]) for r in rows] == list(range(1, 17))
    assert [int(r[1]) for r in rows] == probs, "the probabilities of the table are the ones written before the run"
    assert [{"yes": True, "NO": False, "not run": None}[r[3]] for r in rows] == v
    assert L.LARGE_HEADLINE == (5, 7, 8, 13, 14) and [v[i - 1] for i in L.LARGE_HEADLINE] == [True, True, True, True, False]       # the headline fails on line 14 alone


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- 3. run A
def test_run_a_every_figure_the_page_quotes():
    s = L.summarize_large(_load("a")["rows"]); st = {a: _steps("a", a) for a in ARMS}
    assert all(len(st[a]) == 160 for a in ARMS)
    assert round(s["fresh"]["prompt_tokens_mean"]) == 3808 and s["fresh"]["canonical"] == 1.0 and s["fresh"]["exact_m6"] == 1.0
    for a in ("exact", "drop_compact", "drop_gap"): assert sum(x["same_as_fresh"] for x in st[a]) == 160 and s[a]["kl_below"] == 1.0 and s[a]["exact_m6"] == 1.0 and s[a]["exact_blind"] == 0.675, a
    assert round(s["drop_compact"]["kl_max"], 4) == 0.0047 and round(s["drop_gap"]["kl_max"], 4) == 0.0177 and s["exact"]["kl_max"] < 1e-3
    tok = {a: sum(x["prefilled"] for x in st[a]) for a in ARMS}; assert tok == {"fresh": 609270, "exact": 224483, "drop_compact": 21221, "drop_gap": 21221}, tok
    assert round(tok["drop_compact"] / tok["fresh"], 4) == 0.0348 and round(tok["exact"] / tok["fresh"], 3) == 0.368 and round(tok["exact"] / tok["drop_compact"], 2) == 10.58
    sec = {a: sum(x["seconds"] for x in st[a]) / 160 for a in ARMS}; assert [round(sec[a], 2) for a in ARMS] == [8.25, 2.81, 1.50, 1.49] and round(sec["drop_compact"] / sec["fresh"], 2) == 0.18 and sec["drop_compact"] < sec["exact"]
    assert round(s["drop_compact"]["stale_share_last"], 3) == 0.935 and round(s["drop_compact"]["decoded_mean"], 1) == 11.2
    assert all(x["ended"] for a in ARMS for x in st[a]), "no reply of run A ran into the token limit"
    last = [x for x in st["drop_compact"] if x["t"] == 40]; assert [(x["stale"], x["resident_state"]) for x in last] == [(3199, 3435), (3410, 3633), (3527, 3750), (3466, 3734)]
    assert sum(1 for x in st["drop_compact"] if x["t"] > 26 and x["same_as_fresh"]) == 56            # the last third of the steps: 27 to 40, four episodes


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- 3. run C
def test_run_c_every_figure_the_page_quotes():
    s = L.summarize_large(_load("c")["rows"]); st = {a: _steps("c", a) for a in ARMS}
    same = {a: sum(x["same_as_fresh"] for x in st[a]) for a in ("exact", "drop_compact", "drop_gap")}; assert same == {"exact": 156, "drop_compact": 115, "drop_gap": 112}, same
    assert round(115 / 160, 3) == 0.719 and round(112 / 160, 3) == 0.7
    diff = [x for x in st["drop_compact"] if not x["same_as_fresh"]]; kinds = {}; [kinds.__setitem__(x["kind"], kinds.get(x["kind"], 0) + 1) for x in diff]
    assert len(diff) == 45 and kinds == {"delete_dep": 34, "delete_leaf": 11}, kinds
    assert sum(1 for x in st["drop_gap"] if not x["same_as_fresh"]) == 48
    ex = [x for x in st["exact"] if not x["same_as_fresh"]]; assert len(ex) == 4 and sorted(x["kind"] for x in ex) == ["delete_dep", "delete_dep", "delete_leaf", "delete_leaf"] and round(s["exact"]["kl_max"], 3) == 0.011
    k = [x for x in st["drop_compact"] if x["kl_max"] is not None]; assert len(k) == 160 and sum(1 for x in k if x["kl_max"] >= 0.05) == 51 and round(s["drop_compact"]["kl_below"], 3) == 0.681 and round(s["drop_compact"]["kl_max"], 1) == 8.2
    m6 = {a: sum(bool(x["exact_m6"]) for x in st[a]) for a in ARMS}; assert m6 == {"fresh": 132, "exact": 131, "drop_compact": 127, "drop_gap": 128}, m6
    assert [round(s[a]["exact_m6"], 3) for a in ARMS] == [0.825, 0.819, 0.794, 0.8]
    assert sum(1 for x in st["fresh"] if x["exact_blind"]) == 101
    gain = sum(1 for f, d in zip(st["fresh"], st["drop_compact"]) if d["exact_m6"] and not f["exact_m6"]); loss = sum(1 for f, d in zip(st["fresh"], st["drop_compact"]) if f["exact_m6"] and not d["exact_m6"])
    assert (gain, loss) == (6, 11) and round(sign_test(gain, loss), 2) == 0.33
    sec = {a: sum(x["seconds"] for x in st[a]) / 160 for a in ARMS}; assert [round(sec[a], 2) for a in ARMS] == [11.33, 5.70, 4.42, 4.31]
    assert sum(1 for x in st["fresh"] if not x["ended"]) == 19 and sum(1 for x in _steps("a", "fresh") if not x["ended"]) == 0
    assert round(s["drop_compact"]["stale_share_last"], 3) == 0.917 and round(s["fresh"]["decoded_mean"], 1) == 39.2


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- 4. the delete replies
def test_the_delete_replies_of_run_c_are_equally_wrong_in_every_arm_and_no_ghost_edge_appears():
    mean = lambda v: sum(v) / len(v); got = {}
    for a in ARMS:
        rs = _delete_replies("c", a); assert len(rs) == 52
        ms = [m for m, _ in rs]
        got[a] = dict(recall=round(mean([m["recall"] for m in ms]), 3), out=round(mean([m["recall_out"] for m in ms if m["recall_out"] is not None]), 3), inn=round(mean([m["recall_in"] for m in ms if m["recall_in"] is not None]), 3),
                      invented=sum(m["invented"] for m in ms), other=sum(m["other"] for m in ms), ghosts=sum(m["ghosts"] for m in ms), blind=sum(bool(x["exact_blind"]) for _, x in rs), cut=sum(1 for _, x in rs if not x["ended"]),
                      n_out=sum(1 for m in ms if m["recall_out"] is not None), n_in=sum(1 for m in ms if m["recall_in"] is not None))
    assert got["fresh"] == dict(recall=0.234, out=0.423, inn=0.035, invented=148, other=29, ghosts=0, blind=4, cut=14, n_out=41, n_in=38), got["fresh"]
    assert got["exact"] == dict(recall=0.234, out=0.423, inn=0.035, invented=144, other=30, ghosts=0, blind=4, cut=14, n_out=41, n_in=38), got["exact"]
    assert got["drop_compact"] == dict(recall=0.24, out=0.411, inn=0.035, invented=147, other=29, ghosts=0, blind=4, cut=15, n_out=41, n_in=38), got["drop_compact"]
    assert got["drop_gap"] == dict(recall=0.237, out=0.386, inn=0.061, invented=140, other=21, ghosts=0, blind=6, cut=13, n_out=41, n_in=38), got["drop_gap"]
    assert round(148 / 52, 2) == 2.85 and round(147 / 52, 2) == 2.83
    # pooled counts: of the 124 edges that hang on the 52 services, 66 start and 58 end at them; fresh lists 28 (26 and 2)
    own = 0; g = [0, 0, 0, 0, 0]
    for r in _load("c")["rows"]:
        g0, script, oracle = L.large_script(r["seed"], 40)
        for t, st in enumerate(script):
            if st["kind"] != "delete_dep": continue
            nd = CICOParser.parse_transition_delta(st["ideal"]).node_deltas[0]; n = (nd.r, nd.c); live = oracle[t]; es = sorted(live.incident(n)); x = r["arms"]["fresh"][t]
            have = {(e[0], e[1], e[2]) for e in lenient_deleted_edges(x["text"]) if live.edges.get((e[0], e[1], e[2])) == e[3]}
            g[0] += len(es); g[1] += sum(e in have for e in es); g[2] += sum(1 for e in es if e[0] == n); g[3] += sum(1 for e in es if e[0] == n and e in have); g[4] += sum(1 for e in es if e[1] == n and e in have)
    assert g == [124, 28, 66, 26, 2], g


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- 5. seen afterwards
def test_the_first_and_the_last_third_of_run_c_are_the_post_hoc_counts_the_page_gives():
    st = _steps("c", "drop_compact"); first = [x for x in st if x["t"] <= 13]; last = [x for x in st if x["t"] >= 27]; nd = lambda xs: sum(1 for x in xs if not x["same_as_fresh"])
    assert (len(first), nd(first), len(last), nd(last)) == (52, 8, 56, 18) and round(1 - 8 / 52, 3) == 0.846 and round(1 - 18 / 56, 3) == 0.679
    assert round(_fisher(8, 44, 18, 38), 3) == 0.046
    df = [x for x in first if x["kind"].startswith("delete")]; dl = [x for x in last if x["kind"].startswith("delete")]; assert (len(df), nd(df), len(dl), nd(dl)) == (23, 8, 24, 18) and round(_fisher(8, 15, 18, 6), 3) == 0.008
    assert nd([x for x in _steps("a", "drop_compact") if x["t"] <= 13 or x["t"] >= 27]) == 0, "run A never differs, so it cannot show the growth"
    assert round(_fisher(10, 10, 10, 10), 6) == 1.0 and _fisher(0, 10, 10, 0) < 1e-4            # the helper itself


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- 6. the page
def test_the_page_above_the_result_is_what_was_written_before_the_run_and_the_result_quotes_the_figures():
    head, res = _page(); assert hashlib.sha256((head.rstrip("\n") + "\n").encode()).hexdigest() == PAGE_SHA256, "the predictions and reading rules must not change after the run; later notes go below them"
    assert hashlib.sha256(head.rstrip("\n").encode()).hexdigest() == "3c101140aed34186f4af0ef7a7ac72b93972fa5b48ed2622ee03254094242096"
    for q in ["Run X (the bigger graph) was not made", "line 16 reads \"not run\"", "The headline failed", "fourteen read yes", "one reads NO (14)", "115 of 160 steps (71.9 %)", "3,808 tokens", "160 of 160",
              "21,221 tokens", "609,270", "3.5 %", "224,483", "10.6 times", "1.50, `fresh` 8.25", "0.18 times", "2.81", "93.5 %", "45 of 160 steps", "34 of the 52", "11 of the 19", "none of the 89",
              "`drop_gap` differed on 48 steps", "4 steps (2.5 %", "largest KL 0.011", "51 of 160 steps", "reached 8.2", "0.23", "0.42", "0.04", "2.85", "14 of the 52", "0.24 / 0.41 / 0.04 and 2.83",
              "82.5 %", "81.9 %", "79.4 %", "80.0 %", "6 times", "11 times", "p = 0.33", "0 times in any arm", "84.6 %", "67.9 %", "8 of 52", "18 of 56", "8 of 23", "18 of 24", "p = 0.046", "p = 0.008",
              "19 of the 160", "not pre-registered", "recall 0.04"]:
        assert q in res, q
    assert "Run X" in res and not re.search(r"run X (was|were) (made|run)\b", res.replace("was not made", "")), "the page must not claim run X"
    assert len(re.findall(r"^\| (?:[1-9]|1[0-6]) \|", res, re.M)) == 16


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn): fn(); print("PASS", name); n += 1
    print(n, "tests passed")
