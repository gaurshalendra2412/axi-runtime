"""Implements concept-map item 99: RESULT of runs B1 and B2 of the block test of 16 Oct 2026. Page: docs/KV_BLOCK_TEST.md (predictions and reading rules written before the runs; the result is appended below them).
Run B1: docs/results/axi_block_b1_Qwen2.5-7B_seed21.json (Colab T4, Qwen2.5-7B-Instruct 4-bit, seed 21, 3 episodes of 30 steps, no rule in the prompt, four arms = 360 replies, saved by `axi_tensor_tests.ipynb` Cell 9).
Run B2: docs/results/axi_block_b2_Qwen2.5-7B_seed21.json (the same, the rule stated, 3 episodes of 20 steps, four arms = 240 replies, Cell 10); it is read in the section "Run B2: the rule stated" of the page.
This file does not run a model. Everything below holds for run B1 unless it says B2; the B2 tests are the ones with b2 in their names. It recomputes from the saved rows (not from their summaries):
  1. the file is what the page says (model, seed, counts, episode seeds 21000 to 21002, the request kinds equal `kv_large.large_script`, no arm stopped early, every reply ended, `same_as_fresh` equals the comparison of the token lists)
     and the saved summary equals the summary recomputed from the rows, by the code and by an independent count here;
  2. every one of the 360 replies is scored again from its saved text with the functions of `kv_drop.score_reply` (parse, gate, blind, repair_m6) on the oracle graph the code rebuilds from the seed: the saved scores are the ones the text gives;
  3. the fourteen reading rules: eight yes (1, 2, 3, 4, 8, 9, 10, 11), three NO (5, 6, 7), three not run (12 to 14); the headline (5, 6, 10, 11) failed; the table of the page (probability and reading of every line) equals the probabilities
     written before the run and the recomputed readings; the instrument moves exactly at the boundaries of lines 5, 6 and 7 when rows are changed;
  4. every figure the page quotes: the table of the notebook, the table by kind of request, the paired sign tests, the times by phase, the peak memory;
  5. the looks taken afterwards, which the page says are NOT pre-registered: the second token of every delete reply, the 43 identical replies are all adds, the six add-edge differences, the first step of each episode
     (no hole in the positions yet), the thirds, the four differing `block16` replies, `block` against `drop_compact`, the first build;
  6. the part of the page above the Result line is byte for byte what was written before the run (hash), and the Result quotes these figures.
What it does NOT show: why the second token flips, anything about a trained model, other block sizes, blocks that follow the graph, run B2, other models, sampling."""
import collections, copy, hashlib, json, os, re, sys, types
sys.path.insert(0, os.path.dirname(__file__))
import axi.experiments.kv_block as B
import axi.experiments.kv_large as L
from axi.engine.cico_parser import CICOParser
from axi.engine.gate import check
from axi.experiments.agent_loop import same_graph, strict_parse
from axi.experiments.cascade import next_state
import axi.experiments.kv_drop as K
from axi.experiments.kv_layout import lenient_deleted_edges, reply_metrics, sign_test

HERE = os.path.dirname(__file__)
PAGE = os.path.join(HERE, "..", "..", "docs", "KV_BLOCK_TEST.md")
FILE = os.path.join(HERE, "..", "..", "docs", "results", "axi_block_b1_Qwen2.5-7B_seed21.json")
FILE2 = os.path.join(HERE, "..", "..", "docs", "results", "axi_block_b2_Qwen2.5-7B_seed21.json")
RESULT_MARKER = "\n\n---\n\n# Result (run 16 Oct 2026"
PAGE_SHA256 = "8f893996d9df330886651ac4cecdd957da76d41a501e5a72e5bd7ecaa38d6b04"       # the whole page as it was before the run (one newline at the end)
ARMS = ("fresh", "drop_compact", "block", "block16")
PROBS = [97, 95, 93, 95, 55, 65, 50, 65, 85, 95, 85, 40, 55, 40]
READS_B1 = [True, True, True, True, False, False, False, True, True, True, True, None, None, None]       # run B1 alone: lines 12 to 14 need run B2
READS = [True, True, True, True, False, False, False, True, True, True, True, False, False, False]      # runs B1 and B2
SEEDS = (21000, 21001, 21002)
_cache = {}


def _load():
    if "d" not in _cache: _cache["d"] = json.load(open(FILE))
    return _cache["d"]


def _load2():
    if "d2" not in _cache: _cache["d2"] = json.load(open(FILE2))
    return _cache["d2"]


def _rows(rows=None): return rows or _load()["rows"]
def _steps(arm, rows=None): return [s for r in _rows(rows) for s in r["arms"][arm]]
def _pairs(arm, rows=None): return [(f, b) for r in _rows(rows) for f, b in zip(r["arms"]["fresh"], r["arms"][arm])]
def _mean(xs):
    xs = [x for x in xs if x is not None]; return sum(xs) / len(xs) if xs else None
def _kind(s): return s["kind"]
def _is_del(s): return s["kind"].startswith("delete")


def _scripts():
    if "sc" not in _cache: _cache["sc"] = {s: L.large_script(s, 30) for s in SEEDS}
    return _cache["sc"]


def _scripts2():
    if "sc2" not in _cache: _cache["sc2"] = {s: L.large_script(s, 20) for s in SEEDS}
    return _cache["sc2"]


def _page():
    text = open(PAGE).read(); assert text.count(RESULT_MARKER) == 1 and text.count("\n# Result") == 1, "the page must carry exactly one Result"
    i = text.index(RESULT_MARKER); return text[:i], text[i:]


def _summ(rows=None):
    """The summary of the printout, counted again here from the per-step fields (independent of `summarize_block`): the identical share is counted from the token lists, not from the saved flag."""
    out = {}
    for a in ARMS:
        st = _steps(a, rows); pr = _pairs(a, rows); n = len(st); kls = [s["kl_max"] for s in st if s["kl_max"] is not None]
        out[a] = dict(n=n, same=None if a == "fresh" else sum(1 for f, b in pr if f["tokens"] == b["tokens"]) / n, exact_m6=sum(1 for s in st if s["exact_m6"]) / n, exact_blind=sum(1 for s in st if s["exact_blind"]) / n,
                      admitted=sum(1 for s in st if s["admitted"]) / n, kl_below_b=(sum(1 for k in kls if k < 0.5) / len(kls)) if kls else None, kl_max=max(kls) if kls else None, prefilled=sum(s["prefilled"] for s in st),
                      seconds=_mean(s["seconds"] for s in st), update_s=_mean(s["update_s"] for s in st), prefill_s=_mean(s["prefill_s"] for s in st), decode_s=_mean(s["decode_s"] for s in st), decoded=_mean(s["decoded"] for s in st),
                      v_dev=_mean(s.get("v_dev") for s in st), peak_mb=max(s["peak_mb"] for s in st), prompt_tokens=_mean(s.get("prompt_tokens") for s in st))
    return out


def _lines(S, rows=None):
    """The fourteen reading rules, counted again here from the independent summary (lines 12 to 14 need run B2)."""
    f, dc, bk, b16 = (S[a] for a in ARMS); rebuilt = [(s["rebuilt_dev_v"], s["rebuilt_dev_k"]) for s in _steps("block", rows) if s.get("rebuilt_dev_v") is not None]
    ok = [f["prompt_tokens"] >= 3000, all(s["canonical"] for s in _steps("fresh", rows)), bool(rebuilt) and all(v <= 0.01 + 1e-12 and k <= 0.05 + 1e-12 for v, k in rebuilt), f["exact_m6"] >= 0.90 - 1e-12,
          bk["same"] >= 0.90 - 1e-12, bk["exact_m6"] >= f["exact_m6"] - 0.03 - 1e-12, bk["kl_below_b"] >= 0.90 - 1e-12, b16["same"] >= 0.90 - 1e-12, bk["v_dev"] > dc["v_dev"], bk["prefilled"] / f["prefilled"] <= 0.10 + 1e-12,
          bk["seconds"] / f["seconds"] <= 0.50 + 1e-12]
    return ok + [None, None, None]


def _lines_b2(S1, S2):
    """Lines 12 to 14, counted again here from the independent summary of run B2."""
    f, bk = S2["fresh"], S2["block"]
    return [bk["exact_m6"] >= f["exact_m6"] - 0.05 - 1e-12, bk["exact_blind"] >= f["exact_blind"] - 0.05 - 1e-12, bk["same"] >= 0.70 - 1e-12]


def _flip(rows, n, key, value, where):
    """Set `key` of the first `n` steps of `block` for which `where(step, fresh_step)` holds."""
    k = 0
    for r in rows:
        for f, b in zip(r["arms"]["fresh"], r["arms"]["block"]):
            if k < n and where(b, f): b[key] = value; k += 1
    assert k == n, (k, n)


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- 1. the file
def test_the_file_is_the_run_the_page_describes_and_its_saved_summary_is_the_recomputed_one():
    d = _load(); m = d["meta"]; assert set(d) == {"rows", "summary", "meta"}
    assert m == dict(model="Qwen/Qwen2.5-7B-Instruct", kind="kv_drop", episodes=3, steps=30, seed=21, hint_rules=False, arms=list(ARMS), k_cap=20000, max_new=120, script="large_120_200_14x14"), m
    assert (B.BLOCK_SEED, B.BLOCK_B1_EPISODES, B.BLOCK_B1_STEPS, B.BLOCK_ARMS) == (21, 3, 30, ARMS) and L.large_script.__name__ == m["script"]
    rows = d["rows"]; assert [r["seed"] for r in rows] == list(SEEDS) and all(r["steps"] == 30 and list(r["arms"]) == list(ARMS) and all(len(r["arms"][a]) == 30 for a in ARMS) for r in rows)
    assert all(v is None for r in rows for v in r["stopped"].values()), "no arm stopped early"
    for r in rows:
        g0, script, oracle = _scripts()[r["seed"]]
        for a in ARMS: assert [s["kind"] for s in r["arms"][a]] == [st["kind"] for st in script[:30]] and [s["t"] for s in r["arms"][a]] == list(range(1, 31)), (r["seed"], a)
    steps = [s for a in ARMS for s in _steps(a)]; assert len(steps) == 360 and all(s["ended"] and s["decoded"] == len(s["tokens"]) < 120 for s in steps), "every reply ended on its own and the counts agree with the token lists"
    assert all(s["same_as_fresh"] == (s["tokens"] == f["tokens"]) for a in ARMS[1:] for f, s in _pairs(a)) and all(s["same_as_fresh"] is None for s in _steps("fresh"))
    assert all(s["prefilled"] == s["prompt_tokens"] for s in _steps("fresh")), "fresh reads the whole prompt every step"
    assert d["summary"] == B.summarize_block(rows), "the saved summary is the one the code computes from the rows"
    S = _summ(); sm = d["summary"]
    for a in ARMS:
        assert abs(S[a]["exact_m6"] - sm[a]["exact_m6"]) < 1e-9 and abs(S[a]["exact_blind"] - sm[a]["exact_blind"]) < 1e-9 and abs(S[a]["admitted"] - sm[a]["admitted"]) < 1e-9 and S[a]["prefilled"] == sm[a]["prefilled"]
        assert abs(S[a]["seconds"] - sm[a]["seconds"]) < 1e-9 and abs(S[a]["decoded"] - sm[a]["decoded_mean"]) < 1e-9 and S[a]["peak_mb"] == sm[a]["peak_mb"]
        if a != "fresh": assert abs(S[a]["same"] - sm[a]["same"]) < 1e-9 and abs(S[a]["kl_below_b"] - sm[a]["kl_below_b"]) < 1e-9 and abs(S[a]["kl_max"] - sm[a]["kl_max"]) < 1e-9
    assert [sm[a]["n"] for a in ARMS] == [90] * 4


def test_a_tampered_row_is_noticed_by_the_summary_and_the_lines_at_the_boundaries_of_lines_5_6_and_7():
    d = copy.deepcopy(_load()); rows = d["rows"]; assert B.summarize_block(rows) == _load()["summary"]
    wrong = lambda b, f: b["same_as_fresh"] is False; assert sum(1 for f, b in _pairs("block") if not b["same_as_fresh"]) == 47
    # line 5: 'identical on at least 90 %' = 81 of 90; the file has 43; 38 more
    _flip(rows, 37, "same_as_fresh", True, wrong); s = B.summarize_block(rows); assert abs(s["block"]["same"] - 80 / 90) < 1e-12 and B.verdict_block(s)[4][1] is False
    rows2 = copy.deepcopy(rows); _flip(rows2, 1, "same_as_fresh", True, wrong); s2 = B.summarize_block(rows2); assert abs(s2["block"]["same"] - 81 / 90) < 1e-12 and B.verdict_block(s2)[4][1] is True, "81 of 90 is exactly 90 %: line 5 reads yes"
    # line 6: 'at most 3 points below fresh's 100 %' = 88 of 90 repaired; the file has 43
    d = copy.deepcopy(_load()); rows = d["rows"]; nowrong = lambda b, f: not b["exact_m6"]
    _flip(rows, 44, "exact_m6", True, nowrong); s = B.summarize_block(rows); assert abs(s["block"]["exact_m6"] - 87 / 90) < 1e-12 and B.verdict_block(s)[5][1] is False and _lines(_summ(rows), rows)[5] is False
    rows2 = copy.deepcopy(rows); _flip(rows2, 1, "exact_m6", True, nowrong); s2 = B.summarize_block(rows2); assert abs(s2["block"]["exact_m6"] - 88 / 90) < 1e-12 and B.verdict_block(s2)[5][1] is True and _lines(_summ(rows2), rows2)[5] is True
    # line 7: 'largest KL below 0.5 on at least 90 %' = 81 of 90; the file has 36
    d = copy.deepcopy(_load()); rows = d["rows"]; high = lambda b, f: b["kl_max"] is not None and b["kl_max"] >= 0.5
    _flip(rows, 44, "kl_max", 0.1, high); s = B.summarize_block(rows); assert abs(s["block"]["kl_below_b"] - 80 / 90) < 1e-12 and B.verdict_block(s)[6][1] is False and _lines(_summ(rows), rows)[6] is False
    rows2 = copy.deepcopy(rows); _flip(rows2, 1, "kl_max", 0.1, high); s2 = B.summarize_block(rows2); assert abs(s2["block"]["kl_below_b"] - 81 / 90) < 1e-12 and B.verdict_block(s2)[6][1] is True and _lines(_summ(rows2), rows2)[6] is True
    # a kept cache that differs from the rebuild by 0.02 in the values reads line 3 NO
    d = copy.deepcopy(_load()); rows = d["rows"]; rows[1]["arms"]["block"][10]["rebuilt_dev_v"] = 0.02; s = B.summarize_block(rows)
    assert B.verdict_block(s)[2][1] is False and _lines(_summ(rows), rows)[2] is False and B.verdict_block(_load()["summary"])[2][1] is True


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- 2. every reply scored again
def test_every_saved_reply_scored_again_from_its_text_gives_the_saved_scores():
    n = 0
    for r in _load()["rows"]:
        g0, script, oracle = _scripts()[r["seed"]]
        for a in ARMS:
            for t, x in enumerate(r["arms"][a]):
                live, expected, instr = oracle[t], oracle[t + 1], script[t]["instruction"]; d = strict_parse(x["text"])
                got = dict(parsed=d is not None, admitted=bool(d is not None and check(live, d).ok), exact_blind=same_graph(next_state("blind", live, d, instr), expected), exact_m6=same_graph(next_state("repair_m6", live, d, instr), expected))
                assert {k: x[k] for k in got} == got, (r["seed"], a, t + 1)
                kit = types.SimpleNamespace(eos=151645, dec=lambda ids, text=x["text"]: text); assert x["tokens"][-1] == kit.eos, "every reply ends with the end token"
                sc = K.score_reply(kit, x["tokens"], live, expected, instr); assert sc == dict(text=x["text"], **got), "`kv_drop.score_reply`, the function the run used, gives the same record"; n += 1
    assert n == 360


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- 3. the fourteen lines
def test_the_fourteen_lines_eight_yes_six_no_the_headline_failed_and_the_page_table_says_the_same():
    d = _load(); d2 = _load2(); mine = _lines(_summ()); got = B.verdict_block(d["summary"]); assert mine == READS_B1 == [v for _, v in got] and [t for t, _ in got] == [t for _, t in B.BLOCK_LINES], "run B1 alone reads eleven lines"
    both = B.verdict_block(d["summary"], d2["summary"]); assert [v for _, v in both] == READS == mine[:11] + _lines_b2(_summ(), _summ(d2["rows"])), "with run B2 all fourteen lines read"
    assert B.BLOCK_HEADLINE == (5, 6, 10, 11) and [READS[i - 1] for i in B.BLOCK_HEADLINE] == [False, False, True, True] and sum(1 for v in READS if v is True) == 8 and sum(1 for v in READS if v is False) == 6
    head, res = _page(); table = re.findall(r"^\| (\d+) \| (\d+) % \| (yes|\*\*NO\*\*) \|", res, re.M)
    assert [int(a) for a, _, _ in table] == list(range(1, 15)) and [int(b) for _, b, _ in table] == PROBS
    assert [{"yes": True, "**NO**": False}[c] for _, _, c in table] == READS, "the page says what the rows say"
    probs = [int(m.group(1)) for m in re.finditer(r"\n\d+\. \[\w+\] .*\((\d+) %\)(?=\n)", head)]; assert probs == PROBS, "the probabilities of the table are the ones written before the run"
    S = d["summary"]; f, dc, bk, b16 = (S[a] for a in ARMS)
    assert d2["meta"]["hint_rules"] is True and d["meta"]["hint_rules"] is False
    assert f["prompt_tokens_mean"] == _summ()["fresh"]["prompt_tokens"] and abs(f["prompt_tokens_mean"] - 3849.8111) < 1e-4 and min(s["prompt_tokens"] for s in _steps("fresh")) == 3545 and max(s["prompt_tokens"] for s in _steps("fresh")) == 4008
    assert f["canonical"] == 1.0 and f["exact_m6"] == 1.0 and (bk["rebuilt_checks"], bk["rebuilt_dev_v_max"], bk["rebuilt_dev_k_max"]) == (9, 0.0, 0.0)
    assert bk["same"] == 43 / 90 and bk["exact_m6"] == 43 / 90 and bk["kl_below_b"] == 36 / 90 and abs(bk["kl_max"] - 5.325) < 5e-4 and b16["same"] == 86 / 90 and abs(bk["v_dev"] - 0.9363) < 5e-5 and abs(dc["v_dev"] - 0.1342) < 5e-5
    assert (bk["prefilled"], f["prefilled"]) == (14827, 346483) and abs(bk["prefilled"] / f["prefilled"] - 0.0428) < 5e-5 and abs(bk["seconds"] / f["seconds"] - 0.3251) < 5e-5
    assert abs(bk["same"] - 0.90) > 0.4 and abs(bk["exact_m6"] - (f["exact_m6"] - 0.03)) > 0.49, "lines 5 and 6 were not near misses"
    rebuilt = [s for s in _steps("block") if s.get("rebuilt_dev_v") is not None]; assert [s["t"] for s in rebuilt] == [1, 11, 21] * 3 and all(s["rebuilt_dev_v"] == 0.0 and s["rebuilt_dev_k"] == 0.0 for s in rebuilt)
    assert sum(1 for s in _steps("block") if s.get("rebuilt_dev_k") is not None) == 9 and all(s.get("rebuilt_dev_v") is None for s in _steps("fresh") + _steps("drop_compact") + _steps("block16"))


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- 4. the tables
def test_the_tables_of_the_page_the_kinds_the_sign_tests_the_phases_and_the_memory():
    d = _load(); S = d["summary"]; head, res = _page(); pc = lambda x: f"{100 * x:.1f} %"
    rows = {"fresh": "| `fresh` | - | %s | %s | - | - | %s | %.2f | - |", "drop_compact": "| `drop_compact` | %s | %s | %s | %s | %.3f | %s | %.2f | %.3f |", "block": "| `block` | %s | %s | %s | %s | %.3f | %s | %.2f | %.3f |", "block16": "| `block16` | %s | %s | %s | %s | %.3f | %s | %.2f | %.3f |"}
    fmt = lambda x: f"{x:,.1f}"
    s = S["fresh"]; assert rows["fresh"] % (pc(s["exact_m6"]), pc(s["exact_blind"]), fmt(s["prefilled_mean"]), s["seconds"]) in res
    for a in ARMS[1:]:
        s = S[a]; line = rows[a] % (pc(s["same"]), pc(s["exact_m6"]), pc(s["exact_blind"]), pc(s["kl_below_b"]), s["kl_max"], fmt(s["prefilled_mean"]), s["seconds"], s["v_dev"]); assert line in res, line
    kinds = {"add_edge": 28, "add_node": 22, "delete_dep": 31, "delete_leaf": 9}
    for k, n in kinds.items():
        for a in ARMS[1:]:
            bk = S[a]["by_kind"][k]; assert bk["n"] == n and sum(1 for f, b in _pairs(a) if b["kind"] == k and f["tokens"] == b["tokens"]) == round(bk["same"] * n)
    names = {"add edge": "add_edge", "add service": "add_node", "delete a service with dependants": "delete_dep", "delete a leaf service": "delete_leaf"}
    rows_ = re.findall(r"^\| (add edge|add service|delete a service with dependants|delete a leaf service) \((\d+)\) \| ([^|]+) \| ([^|]+) \| ([^|]+) \|$", res, re.M); assert len(rows_) == 4
    for name, n, c1, c2, c3 in rows_:
        k = names[name]; assert int(n) == kinds[k]
        for a, cell in zip(ARMS[1:], (c1, c2, c3)):
            same = S[a]["by_kind"][k]["same"]; got = float(re.search(r"(\d+(?:\.\d+)?) %", cell).group(1)); assert abs(got - 100 * same) < 0.051, (k, a, cell, same)
    for q in ("| add edge (28) | 100 % | 78.6 % (22) | 100 % |", "| add service (22) | 100 % | 95.5 % (21) | 100 % |", "| delete a service with dependants (31) | 100 % | **0 %** | 87.1 % (27) |", "| delete a leaf service (9) | 100 % | **0 %** | 100 % |"): assert q in res, q
    for a, (g, w, p) in {"block": (0, 47, 1.4e-14), "block16": (0, 3, 0.25), "drop_compact": (0, 0, 1.0)}.items():
        gain = sum(1 for f, b in _pairs(a) if b["exact_m6"] and not f["exact_m6"]); loss = sum(1 for f, b in _pairs(a) if f["exact_m6"] and not b["exact_m6"]); assert (gain, loss) == (g, w) and abs(sign_test(gain, loss) - p) <= max(1e-12, 1e-3 * p)
    assert "`block` 0 steps better and 47 worse (exact two-sided sign test, p < 0.0001), `block16` 0 better and 3 worse (p = 0.25), `drop_compact` 0 and 0" in res
    bk, dc = S["block"], S["drop_compact"]; assert (round(bk["update_s"], 2), round(bk["prefill_s"], 2), round(bk["decode_s"], 2), round(dc["update_s"], 2), round(dc["prefill_s"], 2), round(dc["decode_s"], 2)) == (1.07, 0.15, 1.59, 0.32, 0.15, 1.07)
    assert "`block` 1.07 s of cache update per step" in res and "1.59 s of decoding; `drop_compact` 0.32, 0.15 and 1.07" in res
    assert [round(S[a]["peak_mb"] / 1000, 1) for a in ARMS] == [11.0, 10.4, 6.9, 7.0] and "`fresh` 11.0 GB, `drop_compact` 10.4 GB, `block` and `block16` 6.9 GB and 7.0 GB" in res
    total = sum(s["seconds"] for a in ARMS for s in _steps(a)); assert round(total) == 1914 and "add up to 1,914 s" in res and "2,196 s" in res
    assert round(S["drop_compact"]["stale_share_last"], 3) == 0.938 and "93.8 % of its state rows stale at the last step" in res


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- 5. the looks taken afterwards
def test_the_second_token_of_every_delete_reply_and_the_43_identical_replies_are_all_adds():
    pr = _pairs("block"); dels = [(f, b) for f, b in pr if _is_del(f)]; adds = [(f, b) for f, b in pr if not _is_del(f)]; assert (len(dels), len(adds)) == (40, 50)
    assert all(f["tokens"][:2] == [38332, 58] and re.fullmatch(r"DEL\[\d+,\d+\]", f["text"]) for f, b in dels), "fresh writes the service form on all 40 deletes"
    assert all(b["tokens"][:2] == [38332, 9697] and b["text"].startswith("DEL[(") and b["first_diff"] == 1 for f, b in dels), "block opens every delete in the edge form: the same first token, a different second token"
    ident = [(f, b) for f, b in pr if f["tokens"] == b["tokens"]]; assert len(ident) == 43 and all(not _is_del(f) for f, b in ident)
    assert sum(1 for f, b in ident if f["kind"] == "add_edge") == 22 and sum(1 for f, b in ident if f["kind"] == "add_node") == 21
    diff_adds = [(f, b) for f, b in adds if f["tokens"] != b["tokens"]]; assert len(diff_adds) == 7 and sorted(f["kind"] for f, b in diff_adds) == ["add_edge"] * 6 + ["add_node"]
    pat = re.compile(r"ADD\[\((\d+),(\d+)\)->\((\d+),(\d+)\):(\w+)#(\d+)\]")
    for f, b in diff_adds:
        if f["kind"] == "add_edge":
            x, y = pat.fullmatch(f["text"]).groups(), pat.fullmatch(b["text"]).groups(); assert x[:2] == y[:2] and x[2:4] != y[2:4] and x[4:] == y[4:], ("same source, relation and weight; the target differs", f["text"], b["text"])
        else: assert re.fullmatch(r"ADD\[\d+,\d+:\d+\]", f["text"]) and b["text"].startswith("ADD[(") and pat.fullmatch(b["text"]), "a service added as an edge item"
    inv = 0; own = 0
    for r in _rows():
        g0, script, oracle = _scripts()[r["seed"]]
        for t, st in enumerate(script[:30]):
            if st["kind"] == "delete_dep":
                nd = CICOParser.parse_transition_delta(st["ideal"]).node_deltas[0]; m = reply_metrics(r["arms"]["block"][t]["text"], oracle[t], (nd.r, nd.c)); inv += m["invented"]; own += m["recall"] or 0
                assert all(e not in oracle[t].edges or True for e in lenient_deleted_edges(r["arms"]["block"][t]["text"]))
    assert inv == 31 and own == 0, "31 invented edges on the 31 deletes with dependants and none of the service's own edges"
    head, res = _page()
    for q in ("In all 40 delete replies of `block` the first token is `DEL`", "token 9697", "token 58", "`fresh` writes `DEL[6,0]` 40 times out of 40", "`DEL[(6,0)->(6,0):owns#8]`", "31 invented edges in all and finds none of the service's own edges",
              "**All 43 replies of `block` that equal `fresh`'s are adds** (22 of 28 add-edge, 21 of 22 add-service)", "The 6 add-edge replies that differ differ in the coordinate of the target", "I do not know why the second token flips"): assert q in res, q


def test_the_first_step_has_no_hole_in_the_positions_and_the_replies_already_differ_and_the_thirds_are_the_mix_of_requests():
    rows = _rows(); first = [r["arms"]["block"][0] for r in rows]; assert [s["gap"] for s in first] == [0, 0, 0] and [s["same_as_fresh"] for s in first] == [False] * 3 and sorted(s["kind"] for s in first) == ["add_edge", "add_edge", "delete_dep"]
    assert [round(s["v_dev"], 3) for s in first] == [0.939, 0.940, 0.940] and [r["arms"]["block"][-1]["gap"] for r in rows] == [519, 361, 397]
    thirds = {}
    for lab, rng in (("first", range(0, 10)), ("mid", range(10, 20)), ("last", range(20, 30))):
        pr = [(r["arms"]["fresh"][i], r["arms"]["block"][i]) for r in rows for i in rng]; dels = [(f, b) for f, b in pr if _is_del(f)]; adds = [(f, b) for f, b in pr if not _is_del(f)]
        thirds[lab] = (len(dels), sum(1 for f, b in adds if f["tokens"] == b["tokens"]), len(adds), sum(1 for f, b in dels if f["tokens"] == b["tokens"]), sum(1 for f, b in pr if f["tokens"] == b["tokens"]) / len(pr))
    assert thirds == {"first": (10, 17, 20, 0, 17 / 30), "mid": (16, 14, 14, 0, 14 / 30), "last": (14, 12, 16, 0, 12 / 30)}
    S = _load()["summary"]["block"]; assert abs(S["same_first"] - 0.5667) < 5e-5 and abs(S["same_late"] - 0.4) < 5e-5
    head, res = _page()
    for q in ("At step 1 of each episode the positions have no hole yet (`gap` = 0", "0.939, 0.940, 0.940", "519, 361 and 397", "Deletes were 10, 16 and 14 of the 30 steps", "`block` was identical on 17 of 20, 14 of 14 and 12 of 16, on deletes on 0 of 10, 0 of 16 and 0 of 14",
              "from 56.7 % in the first third to 40.0 % in the last"): assert q in res, q


def test_block16_four_differing_replies_block_against_drop_compact_and_the_first_build():
    rows = _rows(); S = _load()["summary"]; pr = _pairs("block16"); diff = [(f, b) for f, b in pr if f["tokens"] != b["tokens"]]
    assert len(diff) == 4 and all(f["kind"] == "delete_dep" for f, b in diff) and sum(1 for f, b in diff if b["exact_m6"]) == 1 and sum(1 for f, b in pr if b["exact_m6"]) == 87
    ends_with_service = [b for f, b in diff if re.search(r"DEL\[\d+,\d+\]$", b["text"])]; assert len(ends_with_service) == 1 and ends_with_service[0]["exact_m6"] and ends_with_service[0]["text"].count("DEL[(") == 4
    edge_only = [b for f, b in diff if not re.search(r"DEL\[\d+,\d+\]", b["text"])]; assert len(edge_only) == 3 and sum(1 for b in edge_only if b["admitted"]) == 1 and not any(b["exact_m6"] for b in edge_only)
    inv = 0
    for r in rows:
        g0, script, oracle = _scripts()[r["seed"]]
        for t, st in enumerate(script[:30]):
            if st["kind"] == "delete_dep":
                nd = CICOParser.parse_transition_delta(st["ideal"]).node_deltas[0]; inv += reply_metrics(r["arms"]["block16"][t]["text"], oracle[t], (nd.r, nd.c))["invented"]
    assert inv == 9
    bk, dc = S["block"], S["drop_compact"]; assert bk["prefilled"] == dc["prefilled"] and round(bk["prefilled_mean"], 1) == round(dc["prefilled_mean"], 1) == 164.7 and S["block16"]["prefilled"] == S["fresh"]["prefilled"]
    assert round(bk["decoded_mean"], 1) == 15.9 and round(dc["decoded_mean"], 1) == 11.0 and bk["stale_last"] is None and dc["stale_share_last"] > 0.9
    first = lambda a: _mean(r["arms"][a][0]["seconds"] for r in rows); rest = lambda a: _mean(s["seconds"] for r in rows for s in r["arms"][a][1:])
    assert [round(first(a), 1) for a in ARMS] == [9.2, 8.8, 32.3, 8.3] and [round(rest(a), 2) for a in ARMS] == [8.62, 1.30, 1.79, 8.26] and abs(rest("block") / rest("fresh") - 0.208) < 5e-4
    head, res = _page()
    for q in ("`block16`'s 4 differing replies are 4 of the 31 deletes with dependants, and three of them have `block`'s shape", "one of the three names an edge that does exist", "(96.7 %) is above its identical figure (95.6 %)", "4 of 31 deletes against 31 of 31",
              "a version that keeps the blocks and rebuilds only the block that changed was not built", "worse on identical replies, on the repaired state, on KL and on time, equal on tokens read, and better on two things", "(1.07 s against 0.32 s: one pass per new item) and longer replies (15.9 tokens a step against 11.0)",
              "Step 1 of an episode took 32.3 s for `block`", "against 9.2 s for `fresh`; steps 2 to 30 took 1.79 s against 8.62 s, 0.21 times"): assert q in res, q


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- run B2 (the rule stated)
def test_b2_the_file_is_the_run_the_page_describes_and_its_saved_summary_is_the_recomputed_one():
    d = _load2(); m = d["meta"]; assert set(d) == {"rows", "summary", "meta"}
    assert m == dict(model="Qwen/Qwen2.5-7B-Instruct", kind="kv_drop", episodes=3, steps=20, seed=21, hint_rules=True, arms=list(ARMS), k_cap=20000, max_new=120, script="large_120_200_14x14"), m
    assert (B.BLOCK_B2_EPISODES, B.BLOCK_B2_STEPS) == (3, 20)
    rows = d["rows"]; assert [r["seed"] for r in rows] == list(SEEDS) and all(r["steps"] == 20 and list(r["arms"]) == list(ARMS) and all(len(r["arms"][a]) == 20 for a in ARMS) for r in rows)
    assert all(v is None for r in rows for v in r["stopped"].values()), "no arm stopped early"
    for r in rows:
        g0, script, oracle = _scripts2()[r["seed"]]
        for a in ARMS: assert [s["kind"] for s in r["arms"][a]] == [st["kind"] for st in script[:20]] and [s["t"] for s in r["arms"][a]] == list(range(1, 21)), (r["seed"], a)
    steps = [s for a in ARMS for s in _steps(a, rows)]; assert len(steps) == 240 and all(s["ended"] == (len(s["tokens"]) < 120) and s["decoded"] == len(s["tokens"]) for s in steps)
    assert {a: sum(1 for s in _steps(a, rows) if not s["ended"]) for a in ARMS} == dict(fresh=5, drop_compact=3, block=0, block16=22)
    assert all(s["same_as_fresh"] == (s["tokens"] == f["tokens"]) for a in ARMS[1:] for f, s in _pairs(a, rows)) and all(s["same_as_fresh"] is None for s in _steps("fresh", rows))
    assert all(s["prefilled"] == s["prompt_tokens"] for s in _steps("fresh", rows)), "fresh reads the whole prompt every step"
    assert d["summary"] == B.summarize_block(rows), "the saved summary is the one the code computes from the rows"
    S = _summ(rows); sm = d["summary"]
    for a in ARMS:
        assert abs(S[a]["exact_m6"] - sm[a]["exact_m6"]) < 1e-9 and abs(S[a]["exact_blind"] - sm[a]["exact_blind"]) < 1e-9 and abs(S[a]["admitted"] - sm[a]["admitted"]) < 1e-9 and S[a]["prefilled"] == sm[a]["prefilled"]
        assert abs(S[a]["seconds"] - sm[a]["seconds"]) < 1e-9 and abs(S[a]["decoded"] - sm[a]["decoded_mean"]) < 1e-9 and S[a]["peak_mb"] == sm[a]["peak_mb"]
        if a != "fresh": assert abs(S[a]["same"] - sm[a]["same"]) < 1e-9 and abs(S[a]["kl_below_b"] - sm[a]["kl_below_b"]) < 1e-9 and abs(S[a]["kl_max"] - sm[a]["kl_max"]) < 1e-9
    assert [sm[a]["n"] for a in ARMS] == [60] * 4 and {k: v["n"] for k, v in sm["fresh"]["by_kind"].items()} == dict(add_edge=17, add_node=17, delete_dep=21, delete_leaf=5)


def test_b2_every_saved_reply_scored_again_from_its_text_gives_the_saved_scores():
    n = 0
    for r in _load2()["rows"]:
        g0, script, oracle = _scripts2()[r["seed"]]
        for a in ARMS:
            for t, x in enumerate(r["arms"][a]):
                live, expected, instr = oracle[t], oracle[t + 1], script[t]["instruction"]; d = strict_parse(x["text"])
                got = dict(parsed=d is not None, admitted=bool(d is not None and check(live, d).ok), exact_blind=same_graph(next_state("blind", live, d, instr), expected), exact_m6=same_graph(next_state("repair_m6", live, d, instr), expected))
                assert {k: x[k] for k in got} == got, (r["seed"], a, t + 1)
                kit = types.SimpleNamespace(eos=151645, dec=lambda ids, text=x["text"]: text); assert x["ended"] == (x["tokens"][-1] == kit.eos), "a reply ended when it ends with the end token"
                sc = K.score_reply(kit, [i for i in x["tokens"]], live, expected, instr); assert sc == dict(text=x["text"], **got), "`kv_drop.score_reply` gives the same record"; n += 1
    assert n == 240


def test_b2_lines_12_13_14_read_no_at_their_boundaries_and_the_page_says_the_numbers():
    d, d2 = _load(), _load2(); S2 = d2["summary"]; f, dc, bk, b16 = (S2[a] for a in ARMS)
    assert _lines_b2(_summ(), _summ(d2["rows"])) == [False, False, False] == [v for _, v in B.verdict_block(d["summary"], S2)][11:]
    assert (bk["exact_m6"], f["exact_m6"]) == (45 / 60, 50 / 60) and (bk["exact_blind"], f["exact_blind"]) == (31 / 60, 37 / 60) and bk["same"] == 28 / 60
    assert abs(100 * (f["exact_m6"] - bk["exact_m6"]) - 8.33) < 5e-3 and abs(100 * (f["exact_blind"] - bk["exact_blind"]) - 10.0) < 5e-3
    pr = _pairs("block", d2["rows"]); gm = sum(1 for a, b in pr if b["exact_m6"] and not a["exact_m6"]); lm = sum(1 for a, b in pr if a["exact_m6"] and not b["exact_m6"])
    gb = sum(1 for a, b in pr if b["exact_blind"] and not a["exact_blind"]); lb = sum(1 for a, b in pr if a["exact_blind"] and not b["exact_blind"])
    assert (gm, lm, gb, lb) == (8, 13, 2, 8) and abs(sign_test(gm, lm) - 0.3833) < 5e-5 and abs(sign_test(gb, lb) - 0.1094) < 5e-5
    # the boundaries: line 12 needs 47 of 60 repaired (fresh 50), line 13 needs 34 of 60 right as written (fresh 37), line 14 needs 42 of 60 identical
    nm = lambda b, f: not b["exact_m6"]; nbl = lambda b, f: not b["exact_blind"]; ns = lambda b, f: not b["same_as_fresh"]
    for key, flag, where, need, idx in (("exact_m6", "exact_m6", nm, 47, 11), ("exact_blind", "exact_blind", nbl, 34, 12)):
        for add, want in ((need - 45 - 1 if key == "exact_m6" else need - 31 - 1, False), (need - 45 if key == "exact_m6" else need - 31, True)):
            dd = copy.deepcopy(d2); _flip(dd["rows"], add, key, True, where); s = B.summarize_block(dd["rows"])
            assert B.verdict_block(d["summary"], s)[idx][1] is want and _lines_b2(_summ(), _summ(dd["rows"]))[idx - 11] is want, (key, add, want)
    for add, want in ((13, False), (14, True)):
        dd = copy.deepcopy(d2); _flip(dd["rows"], add, "same_as_fresh", True, ns); s = B.summarize_block(dd["rows"]); assert B.verdict_block(d["summary"], s)[13][1] is want, (add, want)
    head, res = _page()
    for q in ("| 12 | 40 % | **NO** | (updated after run B2) `block` right after `repair_m6` on 45 of 60 steps (75.0 %), `fresh` on 50 of 60 (83.3 %): **8.3 points below**, the line allowed 5; step by step 8 better and 13 worse (exact sign test p = 0.38",
              "| 13 | 55 % | **NO** | (updated after run B2) exactly right as written: `block` 31 of 60 (51.7 %), `fresh` 37 of 60 (61.7 %): **10.0 points below**, the line allowed 5 (2 better, 8 worse, p = 0.11)",
              "| 14 | 40 % | **NO** | (updated after run B2) `block` identical to `fresh` on 28 of 60 steps (**46.7 %**); the line needed 70 %"): assert q in res, q


def test_b2_the_tables_of_the_page_the_kinds_the_sign_tests_the_cost():
    d2 = _load2(); S = d2["summary"]; rows = d2["rows"]; head, res = _page(); pc = lambda x: f"{100 * x:.1f} %"; fmt = lambda x: f"{x:,.1f}"
    line = lambda a: ("| `%s` | %s | %s | %s | %s | %s | %s | %.2f | %s |" % (a, pc(S[a]["same"]) if a != "fresh" else "-", pc(S[a]["exact_m6"]), pc(S[a]["exact_blind"]), pc(S[a]["kl_below_b"]) if a != "fresh" else "-",
                                                                         ("%.3f" % S[a]["kl_max"]) if a != "fresh" else "-", fmt(S[a]["prefilled_mean"]), S[a]["seconds"], ("%.3f" % S[a]["v_dev"]) if a != "fresh" else "-"))
    for a in ARMS: assert line(a) in res, line(a)
    kinds = {"add edge": ("add_edge", 17), "add service": ("add_node", 17), "delete a service with dependants": ("delete_dep", 21), "delete a leaf service": ("delete_leaf", 5)}
    got = re.findall(r"^\| (add edge|add service|delete a service with dependants|delete a leaf service) \((\d+)\) \| ([^|]+) \| ([^|]+) \| ([^|]+) \| ([^|]+) \|$", res, re.M)
    b2rows = [g for g in got if re.match(r"^[\d.]+ %$", g[2].strip())]; assert len(b2rows) == 4, b2rows
    num = lambda c: [float(x) for x in re.findall(r"(\d+(?:\.\d+)?) %", c)]
    for name, n, c_fresh, c_dc, c_bk, c_b16 in b2rows:
        k, nn = kinds[name]; assert int(n) == nn and S["fresh"]["by_kind"][k]["n"] == nn
        assert abs(num(c_fresh)[0] - 100 * S["fresh"]["by_kind"][k]["exact_m6"]) < 0.051
        for a, c in (("drop_compact", c_dc), ("block", c_bk), ("block16", c_b16)):
            v = S[a]["by_kind"][k]; same_n = sum(1 for f, b in _pairs(a, rows) if b["kind"] == k and f["tokens"] == b["tokens"]); assert abs(same_n / nn - v["same"]) < 1e-9
            assert all(abs(x - y) < 0.051 for x, y in zip(num(c), [100 * v["same"], 100 * v["exact_m6"]])), (k, a, c)
    pairs = {}
    for a in ARMS[1:]:
        pr = _pairs(a, rows); pairs[a] = [(sum(1 for f, b in pr if b[m] and not f[m]), sum(1 for f, b in pr if f[m] and not b[m])) for m in ("exact_m6", "exact_blind")]
    assert pairs == {"drop_compact": [(3, 3), (0, 2)], "block": [(8, 13), (2, 8)], "block16": [(5, 18), (3, 11)]}
    assert abs(sign_test(5, 18) - 0.0106) < 5e-5 and abs(sign_test(3, 11) - 0.0574) < 5e-5 and sign_test(3, 3) == 1.0 and sign_test(0, 2) == 0.5
    assert "after the repair step `drop_compact` 3 / 3 (p = 1.0), `block` 8 / 13 (p = 0.38), `block16` 5 / 18 (p = 0.011); as written `drop_compact` 0 / 2 (p = 0.5), `block` 2 / 8 (p = 0.11), `block16` 3 / 11 (p = 0.057)" in res
    bk = S["block"]; assert (bk["rebuilt_checks"], bk["rebuilt_dev_v_max"], bk["rebuilt_dev_k_max"]) == (6, 0.0, 0.0) and [s["t"] for s in _steps("block", rows) if s.get("rebuilt_dev_v") is not None] == [1, 11] * 3
    assert [round(S[a]["peak_mb"] / 1000, 1) for a in ARMS] == [11.1, 10.5, 7.0, 7.0] and "`fresh` 11.1 GB, `drop_compact` 10.5 GB, `block` and `block16` 7.0 GB" in res
    total = sum(s["seconds"] for a in ARMS for s in _steps(a, rows)); assert round(total) == 2109 and "add up to 2,109 s" in res and "2,303 s" in res
    f = S["fresh"]; assert (round(bk["prefilled_mean"], 1), round(f["prefilled_mean"], 1)) == (233.3, 3963.6) and abs(100 * bk["prefilled"] / f["prefilled"] - 5.89) < 5e-3 and abs(bk["seconds"] / f["seconds"] - 0.338) < 5e-4
    first = lambda a: _mean(r["arms"][a][0]["seconds"] for r in rows); rest = lambda a: _mean(s["seconds"] for r in rows for s in r["arms"][a][1:])
    assert (round(first("block"), 1), round(first("fresh"), 1), round(rest("block"), 2), round(rest("fresh"), 2)) == (35.1, 12.8, 2.38, 11.84) and abs(rest("block") / rest("fresh") - 0.201) < 5e-4
    assert round(S["drop_compact"]["stale_share_last"], 3) == 0.922 and round(S["drop_compact"]["same"] * 60) == 46 and round(S["block16"]["same"] * 60) == 20 and round(S["block16"]["exact_m6"] * 60) == 37


def test_b2_looks_taken_afterwards():
    d2 = _load2(); rows = d2["rows"]; S = d2["summary"]; pr = _pairs("block", rows)
    # the form of the first item of a delete reply
    first_form = lambda a: collections.Counter("service" if re.match(r"DEL\[\d+,\d+\]", b["text"]) else "edge" for f, b in _pairs(a, rows) if _is_del(f))
    assert first_form("block") == collections.Counter(service=17, edge=9) and all(first_form(a) == collections.Counter(service=26) for a in ("fresh", "drop_compact", "block16"))
    # add-service: the five that differ for block are edge items with the service's coordinates at both ends; block16 adds DEL items after the ADD
    diff = [(f, b) for f, b in pr if f["kind"] == "add_node" and f["tokens"] != b["tokens"]]; assert len(diff) == 5
    for f, b in diff:
        m = re.fullmatch(r"ADD\[(\d+),(\d+):(\d+)\]", f["text"]); x = re.fullmatch(r"ADD\[\((\d+),(\d+)\)->\((\d+),(\d+)\):owns#(\d+)\]", b["text"]); assert m and x and x.group(1, 2) == m.group(1, 2) == x.group(3, 4) and x.group(5) == m.group(3), (f["text"], b["text"])
    assert sum(1 for f, b in pr if f["kind"] == "add_edge" and f["tokens"] != b["tokens"]) == 1
    adds_del = {a: sum(1 for f, b in _pairs(a, rows) if f["kind"] == "add_node" and b["text"].startswith("ADD[") and "DEL[" in b["text"]) for a in ARMS}; assert adds_del == dict(fresh=0, drop_compact=0, block=0, block16=14)
    assert collections.Counter(b["kind"] for f, b in _pairs("block16", rows) if not b["ended"]) == collections.Counter(delete_dep=11, add_node=10, delete_leaf=1)
    # the deletes with dependants: recall of the service's own edges and the invented edges
    tot = {a: collections.Counter() for a in ARMS}
    for r in rows:
        g0, script, oracle = _scripts2()[r["seed"]]
        for t, st in enumerate(script[:20]):
            if st["kind"] == "delete_dep":
                nd = CICOParser.parse_transition_delta(st["ideal"]).node_deltas[0]
                for a in ARMS:
                    m = reply_metrics(r["arms"][a][t]["text"], oracle[t], (nd.r, nd.c)); tot[a]["n"] += 1; tot[a]["recall"] += m["recall"] or 0; tot[a]["invented"] += m["invented"]
    assert all(tot[a]["n"] == 21 for a in ARMS) and tot["block"]["recall"] == 0 and tot["block"]["invented"] == 28 and tot["fresh"]["invented"] == 76 and tot["block16"]["invented"] == 96 and tot["drop_compact"]["invented"] == 66
    assert [round(tot[a]["recall"] / 21, 2) for a in ("fresh", "drop_compact", "block16")] == [0.19, 0.13, 0.14] and round(tot["fresh"]["invented"] / 21, 2) == 3.62 and round(tot["block"]["invented"] / 21, 2) == 1.33 and round(tot["block16"]["invented"] / 21, 2) == 4.57
    assert (round(S["block"]["decoded_mean"], 1), round(S["fresh"]["decoded_mean"], 1)) == (21.2, 37.6)
    bk = S["block"]["by_kind"]; fr = S["fresh"]["by_kind"]; assert round(bk["delete_dep"]["exact_m6"] * 21) == round(fr["delete_dep"]["exact_m6"] * 21) == 14 and round(bk["add_node"]["exact_m6"] * 17) == 12 and round(bk["add_edge"]["exact_m6"] * 17) == 16
    dc = S["drop_compact"]["by_kind"]; assert (round(dc["delete_dep"]["same"] * 21), round(dc["delete_leaf"]["same"] * 5)) == (8, 4) and dc["add_edge"]["same"] == dc["add_node"]["same"] == 1.0
    head, res = _page()
    for q in ("finds 0 of the service's own edges (recall 0.0; `fresh` 0.19, `drop_compact` 0.13, `block16` 0.14) and names 28 invented edges (1.33 a reply; `fresh` 3.62, `block16` 4.57)", "(21.2 tokens on average over all kinds of request, against 37.6 for `fresh`)",
              "After the repair step it is right on 14 of 21, the same as `fresh`", "`block` is right on 12 of 17, `fresh` on 17 of 17", "`ADD[(13,7)->(13,7):owns#6]`", "Add-edge: 16 of 17", "9 of the 26 delete replies of `block` open in the edge form",
              "in run B1 it was 40 of 40", "Identical on 33.3 % (20 of 60), below `block`'s 46.7 %, and right after the repair step on 61.7 % (37 of 60)", "14 of its 17 add-service replies write the service and then go on to write `DEL[...]` items",
              "22 of its 60 replies run to the 120-token limit (11 deletes with dependants, 10 add-service, 1 leaf; `fresh` 5, `drop_compact` 3, `block` 0)", "about 0.06 after a correction", "Identical on 46 of 60 steps (76.7 %): 8 of 21 deletes with dependants, 4 of 5 leaves, all 34 adds",
              "(46.7 against 76.7 %)", "(75.0 against 83.3 %)", "(45.0 against 90.0 % of the steps below 0.5 nats)", "(4.02 s against 4.41 s)", "(none, against 92.2 %)", "(7.0 GB against 10.5 GB)",
              "233.3 tokens a step against 3,963.6 for `fresh` (5.89 %; the dry run said 6.1 %)", "4.02 s a step against 11.89 s (0.338 times)", "(35.1 s on average, against 12.8 s for `fresh`)", "2.38 s against 11.84 s (0.20 times)"): assert q in res, q
    for dry, share in ((6.1, 5.89),): assert abs(dry - share) < 0.3
    assert abs(100 * B.block_dry_run(3, 20)["drop_over_fresh"] - 6.1) < 0.1, "the dry run of the page"


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- 6. the page
def test_the_page_above_the_result_is_what_was_written_before_the_run_and_the_result_quotes_the_figures():
    head, res = _page(); assert hashlib.sha256((head.rstrip("\n") + "\n").encode()).hexdigest() == PAGE_SHA256, "the predictions and reading rules must not change after the run; later notes go below them"
    assert PAGE_SHA256 in res and "test 112" in res and "docs/results/axi_block_b1_Qwen2.5-7B_seed21.json" in res and "docs/results/axi_block_b2_Qwen2.5-7B_seed21.json" in res and "`python/tests/test_kv_block_result.py`" in res
    for q in ("The headline failed on two of its four lines", "Lines 5 and 6 read NO; lines 10 and 11 read yes", "eight read yes (1, 2, 3, 4, 8, 9, 10, 11) and three read NO (5, 6, 7)", "about 29 % and my single number was 40 %",
              "Run B2 (Cell 10, the rule stated) was made a few hours later and is read in its own section below", "## Run B2: the rule stated", "**Lines 12, 13 and 14 all read NO**",
              "eight lines yes (1, 2, 3, 4, 8, 9, 10, 11) and six NO (5, 6, 7, 12, 13, 14)", "**Blocks of 16 are not better here; they are the worst arm.**", "(updated after run B2: this holds for the prompt without the rule only",
              "so no block size tested here is shown to keep the replies in both prompts", "because size 16 is right in one and wrong in the other", "Above I wrote that I would not spend the time on this run before the block-size run; it was made anyway",
              "keeps the replies (95.6 %) in this run (updated after run B2".replace("keeps","kept"), "3 episodes x 30 steps x 4 arms = 360 replies", "the prompt of `fresh` held 3,849.8 tokens on average (3,545 to 4,008)", "value rows differ from `fresh`'s by 0.936 (mean relative L2, over layers and steps) against 0.134 for `drop_compact`", "`block` read 14,827 tokens against 346,483 for `fresh`",
              "(**47.8 %**); the line needed 90 %", "**52.2 points below**", "(**40.0 %**; the line needed 90 %); the largest KL anywhere is 5.325", "95.6 %", "0.936", "0.134", "**4.28 %**", "**0.325 times**",
              "**every one of the 40 delete steps**", "None of the 40 delete steps ends in the right state", "seven times more", "had no entry for this combination", "*\"5 NO and 6 yes with 8 yes:", "NOT pre-registered",
              "reasons to test again, not as findings", "\"no pre-training required\" is false for independent single-item blocks on this model and this task", "(the related work says it is what is needed, and it was not tested)",
              "Not seen by me", "deliberate changes", "The block size", "Blocks that follow the graph", "Run B2** was made (above)", "sha256 `8f893996",
              "by `python/tests/test_kv_block_result.py` (test 112). Run B2", "## What I read from the rows afterwards (NOT pre-registered; each of these is a look at tables I had already seen"): assert q.lower() in res.lower(), q
    assert "[X]" not in res and "TODO" not in res and "MUTATIONS" not in res and "pending" not in res.lower().replace("not pending", ""), "no placeholder is left in the Result"


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn): fn(); print("PASS", name); n += 1
    print(n, "tests passed")
