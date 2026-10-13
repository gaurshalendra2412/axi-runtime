"""Implements concept-map row 3 / 9 / 27 (structure against scale), item 86: RESULT of the hosted-model test, Part A, Gemini 3.5 Flash Lite, files received 10 Oct 2026. Page: docs/API_TEST_GEMINI.md
(prediction, reading rules and two addenda written before any result file was read; the Result is appended below them; Part B has its own section when its files arrive).
The four files are docs/results/axi_gemini_{single_rule, single_norule, long_rule, long_norule}_gemini-3.5-flash-lite_seed7.json.
This file does not run a model. It recomputes from the saved rows (not from their summaries):
  1. the files are what the page says (model, regime, seed, counts, request kinds equal make_scope_tasks(140, 7), the oracle scripts of seeds 7000 to 7005 / 7000 to 7003), every saved summary equals the summary
     recomputed from the rows, and the bytes are the ones whose sha256 the page lists;
  2. the format: 78 of 651 replies do not parse (30 + 3 single, 45 + 0 long), all 78 are the same slip (`DEL[r,c:n]`) and all 78 parse once that number is dropped; the 22 old deletes scope_m7 misses
     in the rule-stated single file are exactly the 22 unparsed delete_dep and delete_leaf replies;
  3. Part 1: every cell of the page's tables, corrupted counts, helped / harmed, the 15 corrupted graphs are all delete_pair;
  4. Part 2: the held state can be rebuilt from the recorded texts for all three policies and every recorded field agrees; in the rule-stated file the first step off the oracle is the first unparsed step in
     all 18 policy-episodes; the no-rule table, the 59 corrupted steps, the onsets (all delete_pair), the edge-slot and prompt-token counts;
  5. the notebook's reading rules: 16 lines, 9 yes and 7 no, each named; the lines flip when a number is doctored;
  6. the exploratory offline re-read (drop the number after the id in `DEL[r,c:n]`): 137 / 136 / 138 / 138 and 123 / 122 / 138 / 138 of 140;
  7. the page quotes these numbers, and the Result sits once, below the locked text (the locking itself is in test_api_gemini.py).
What it does NOT show: Part B, other models or families, sampling, or that a second run gives the same text (not tested)."""
import copy, hashlib, json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
from axi.experiments.agent_loop import strict_parse
from axi.experiments.api_test import unparsed_single, unparsed_long, verdict_api_format
from axi.experiments.bigger_model import verdict_big_single, verdict_big_long
from axi.experiments.long_horizon import oracle_script, summarize_long
from axi.experiments.scope_tasks import make_scope_tasks
from axi.experiments.scope_test import summarize_scope, evaluate
from test_scope_result import replay

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "..", "..", "docs", "results")
PAGE = os.path.join(HERE, "..", "..", "docs", "API_TEST_GEMINI.md")
MODEL = "gemini-3.5-flash-lite"
NAMES = {"sr": "single_rule", "sn": "single_norule", "lr": "long_rule", "ln": "long_norule"}
SHA = {"sr": "f3357d614ad7f2cb42b1a31ece80d8a6dae64068bb4a1c14fa2efde19eefd9ca", "sn": "b4163b4f414d2ef318084581d9de7ba26831349ef208a668c0e97fc83e0964fd",
       "lr": "09ced730e937d7333d06842b0b471b1f2db73cb6a6777cfd0f5a276817618733", "ln": "a97db3d75130c813f7b73ad6849435ef149f1f86f229dcbe9766a510b691870b"}
RESULT_MARK = "\n\n---\n\n# Result"
P5 = ("blind", "gate", "repair_m6", "scope_m7", "scope_delta")
SLIP = re.compile(r"DEL\[(\d+),(\d+):\d+\]")


def load():
    paths = {k: os.path.join(RESULTS, f"axi_gemini_{n}_{MODEL}_seed7.json") for k, n in NAMES.items()}
    if not all(os.path.exists(p) for p in paths.values()): return None                            # docs/results not shipped with this copy of the repo
    return {k: json.load(open(p)) for k, p in paths.items()}


def page():
    return open(PAGE).read() if os.path.exists(PAGE) else None


TASKS = None
def tasks():
    global TASKS
    if TASKS is None: TASKS = make_scope_tasks(140, 7)
    return TASKS


def fix(text): return SLIP.sub(r"DEL[\1,\2]", text)


def long_steps(rows, policy): return [s for r in rows for s in r["policies"][policy]]


def test_the_files_are_what_the_page_says_and_the_summaries_equal_the_rows():
    R = load()
    if R is None: return
    for k, hint in (("sr", True), ("sn", False)):
        m = R[k]["meta"]; rows = R[k]["rows"]
        assert (m["model"], m["seed"], m["n"], m["hint_rules"], m["test"]) == (MODEL, 7, 140, hint, "scope_single_step"), k
        assert [r["kind"] for r in rows] == [t.kind for t in tasks()] and len(rows) == 140
        assert json.loads(json.dumps(summarize_scope(rows, tasks()))) == R[k]["summary"], k
    steps = 0
    for k, hint, eps in (("lr", True, 6), ("ln", False, 4)):
        m = R[k]["meta"]
        assert (m["model"], m["mode"], m["episodes"], m["steps"], m["seed"], m["hint_rules"], m["policies"], m["mix"]) == (MODEL, "state", eps, 30, 7, hint, ["blind", "repair_m6", "scope_m7"], "scope"), k
        assert [r["seed"] for r in R[k]["rows"]] == [7000 + e for e in range(eps)]
        for row in R[k]["rows"]:
            _, script, _ = oracle_script(row["seed"], 30, "scope"); assert len(script) == 30
            assert row["stopped"] == {"blind": None, "repair_m6": None, "scope_m7": None}
            for p, log in row["policies"].items():
                assert [x["kind"] for x in log] == [s["kind"] for s in script], (row["seed"], p); steps += len(log)
        assert json.loads(json.dumps(summarize_long(R[k]["rows"]))) == R[k]["summary"], k
    assert steps == 900                                                                            # 540 + 360 recorded policy steps
    for k, n in NAMES.items():                                                                     # the bytes are the ones the page lists
        data = open(os.path.join(RESULTS, f"axi_gemini_{n}_{MODEL}_seed7.json"), "rb").read(); assert hashlib.sha256(data).hexdigest() == SHA[k], k
        assert b"AIza" not in data and b"api_key" not in data                                      # no key in a result file
    res = page()
    if res is not None: assert all(f"`{h}`" in res for h in SHA.values())
    bad = copy.deepcopy(R["ln"]); bad["rows"][0]["policies"]["blind"][3]["distance"] += 1
    assert json.loads(json.dumps(summarize_long(bad["rows"]))) != bad["summary"]                  # a doctored row no longer matches its summary
    bad = copy.deepcopy(R["sn"]); bad["rows"][5]["text"] = "ADD[0,0:1]"
    assert json.loads(json.dumps(summarize_scope(bad["rows"], tasks()))) != bad["summary"]


def test_the_format_line_78_of_651_all_one_slip_and_what_it_explains():
    R = load()
    if R is None: return
    s_r, s_n = unparsed_single(R["sr"]["rows"]), unparsed_single(R["sn"]["rows"]); l_r, l_n = unparsed_long(R["lr"]["rows"]), unparsed_long(R["ln"]["rows"])
    assert (s_r, s_n, l_r, l_n) == ((30, 140), (3, 140), (45, 194), (0, 177))
    bad = sum(x[0] for x in (s_r, s_n, l_r, l_n)); tot = sum(x[1] for x in (s_r, s_n, l_r, l_n)); assert (bad, tot) == (78, 651) and 20 * bad > tot and round(100 * bad / tot, 1) == 12.0
    # every unparsed reply is the same slip, and parses once the number after the id is dropped
    texts = [r["text"] for k in ("sr", "sn") for r in R[k]["rows"] if strict_parse(r["text"]) is None]
    texts += [s["text"] for k in ("lr", "ln") for row in R[k]["rows"] for st in row["policies"].values() for s in st if not s.get("cached") and not s["parsed"]]
    assert len(texts) == 78 and all(SLIP.search(t) for t in texts) and all(strict_parse(fix(t)) is not None for t in texts)
    assert not any(SLIP.search(r["text"]) and strict_parse(r["text"]) is not None for k in ("sr", "sn") for r in R[k]["rows"])   # the slip never parses by itself
    from collections import Counter
    assert Counter(r["kind"] for r in R["sr"]["rows"] if strict_parse(r["text"]) is None) == {"delete_leaf": 12, "delete_dep": 10, "delete_pair": 8}
    assert Counter(r["kind"] for r in R["sn"]["rows"] if strict_parse(r["text"]) is None) == {"delete_pair": 2, "delete_leaf": 1}
    assert 30 + 45 == 75
    # the rule-stated old deletes scope_m7 misses are exactly the unparsed delete_dep and delete_leaf replies
    S = summarize_scope(R["sr"]["rows"], tasks()); assert S["old_deletes"]["scope_m7"]["success"] == 18 and 40 - 18 == 10 + 12
    ev = evaluate(R["sr"]["rows"], tasks())
    assert [e["kind"] for e in ev if e["kind"] in ("delete_dep", "delete_leaf") and not e["scope_m7"]["success"]] == [e["kind"] for e in ev if e["kind"] in ("delete_dep", "delete_leaf") and not e["parsed"]]
    # rule-stated long file: the first step off the oracle is the first unparsed step, for all 18 policy-episodes
    firsts = []
    for row in R["lr"]["rows"]:
        for p, st in row["policies"].items():
            fne = next(s["t"] for s in st if not s["exact"]); fu = next(s["t"] for s in st if not s["parsed"]); assert fne == fu, (row["seed"], p); firsts.append(fu)
    assert len(firsts) == 18 and sorted(set(firsts)) == [1, 2, 7]
    for p in ("blind", "repair_m6", "scope_m7"): assert round(100 * sum(s["exact"] for s in long_steps(R["lr"]["rows"], p)) / 180, 1) == 7.8


def test_part_1_the_tables_of_the_page():
    R = load()
    if R is None: return
    exp = {  # kind: (no rule cells, rule cells), each in the order blind, gate, repair_m6, scope_m7, scope_delta
        "delete_dep": ((20, 20, 20, 20, 20), (10, 10, 10, 10, 10)), "delete_leaf": ((19, 19, 19, 19, 19), (8, 8, 8, 8, 8)),
        "delete_edge": ((20, 20, 20, 20, 0), (20, 20, 20, 20, 0)), "delete_pair": ((3, 3, 18, 18, 0), (11, 11, 12, 12, 0)),
        "delete_weight": ((18, 17, 18, 18, 0), (18, 17, 18, 18, 0)), "add_node": ((20,) * 5, (20,) * 5), "add_edge": ((20,) * 5, (20,) * 5),
        "old_deletes": ((39, 39, 39, 39, 39), (18, 18, 18, 18, 18)), "all": ((120, 119, 135, 135, 79), (107, 106, 108, 108, 58))}
    S = {k: summarize_scope(R[k]["rows"], tasks()) for k in ("sn", "sr")}
    for g, (nr, ru) in exp.items():
        for key, want in (("sn", nr), ("sr", ru)):
            s = S[key][g]; assert tuple(s[p]["success"] for p in P5) == want, (g, key)
            assert s["n"] == (40 if g == "old_deletes" else 140 if g == "all" else 20)
    for key, blind in (("sn", 15), ("sr", 1)):
        assert S[key]["all"]["blind"]["corrupted"] == blind and all(S[key]["all"][p]["corrupted"] == 0 for p in P5[1:]), key
    assert sum(S[k]["all"][p]["corrupted"] for k in S for p in ("gate", "repair_m6", "scope_m7")) == 0 and 140 * 2 == 280      # 0 of 280 single-request graphs per policy
    assert S["sn"]["paired"] == dict(helped=0, harmed=0, both_right=135, both_wrong=5) and S["sr"]["paired"] == dict(helped=0, harmed=0, both_right=108, both_wrong=32)
    assert S["sn"]["dropped"] == dict(tasks_with_drops=0, items=0, tasks_with_drops_and_right=0) and S["sr"]["dropped"] == dict(tasks_with_drops=0, items=0, tasks_with_drops_and_right=0)
    ev = evaluate(R["sn"]["rows"], tasks())
    assert [e["kind"] for e in ev if e["blind"]["corrupted"]] == ["delete_pair"] * 15                # all the raw model's damage is delete_pair
    assert 10 * (135 - 120) >= 140 and round(100 * (135 - 120) / 140, 1) == 10.7 and round(100 * 135 / 140, 1) == 96.4
    assert 107 - 108 < 0 and 108 - 107 == 1                                                          # rule stated: scope_m7 ahead of blind apply by 1 task
    tok = lambda k: sum(r["tokens"] for r in R[k]["rows"]); assert (tok("sn"), tok("sr")) == (2536, 3025) and round(3025 / 2536, 2) == 1.19
    sec = lambda k: sum(r["seconds"] for r in R[k]["rows"]) / 140; assert (round(sec("sn"), 2), round(sec("sr"), 2)) == (0.57, 0.79)


def test_part_2_the_held_state_rebuilt_and_the_numbers_of_the_page():
    R = load()
    if R is None: return
    for key in ("lr", "ln"):
        for row in R[key]["rows"]:
            for p in row["policies"]:
                for rec, st, before, after, orc, df in replay(row, p):
                    assert (len(df), after.is_well_formed(), not df) == (rec["distance"], not rec["corrupted"], rec["exact"]), (key, row["seed"], p, rec["t"])
                    assert len(after.nodes) == rec["nodes"] and len(after.edges) == rec["edges"]
    for key in ("lr", "ln"):                                                                       # by construction, never corrupted
        assert not any(s["corrupted"] for p in ("repair_m6", "scope_m7") for s in long_steps(R[key]["rows"], p))
    assert sum(len(long_steps(R[k]["rows"], p)) for k in ("lr", "ln") for p in ("blind", "repair_m6", "scope_m7")) == 900
    ln = R["ln"]["rows"]
    for p in ("repair_m6", "scope_m7"): assert all(s["exact"] for s in long_steps(ln, p)) and len(long_steps(ln, p)) == 120
    bl = long_steps(ln, "blind"); assert sum(s["exact"] for s in bl) == 61 and round(100 * 61 / 120, 1) == 50.8                       # exact steps 50.8 %
    assert sum(s["corrupted"] for s in bl) == 59
    first = {row["seed"]: next((s["t"] for s in row["policies"]["blind"] if s["corrupted"]), None) for row in ln}; assert first == {7000: 14, 7001: None, 7002: 3, 7003: 13}
    assert sum(1 for row in ln if first[row["seed"]] is not None) == 3 and sum(1 for v in first.values() if v is not None and v <= 10) == 1
    assert [row["policies"]["blind"][-1]["dangling"] for row in ln] == [0, 0, 1, 3] and sum(row["policies"]["blind"][-1]["dangling"] for row in ln) / 4 == 1.0
    onsets = []
    for row in ln:
        st = row["policies"]["blind"]
        for i, s in enumerate(st):
            if s["corrupted"] and (i == 0 or not st[i - 1]["corrupted"] or s["dangling"] > st[i - 1]["dangling"]): onsets.append(s["kind"])
    assert onsets == ["delete_pair"] * 3
    edges = {p: sum(s["edges"] for s in long_steps(ln, p)) for p in ("blind", "repair_m6", "scope_m7")}; dang = sum(s["dangling"] for s in bl)
    assert edges == {"blind": 354, "repair_m6": 254, "scope_m7": 254} and dang == 100 and edges["blind"] - edges["repair_m6"] == dang
    ptok = {p: sum(s["prompt_tokens"] for s in long_steps(ln, p)) for p in ("blind", "repair_m6")}; assert ptok == {"blind": 38263, "repair_m6": 37038} and round(100 * (38263 / 37038 - 1), 1) == 3.3
    lr = R["lr"]["rows"]; bl = long_steps(lr, "blind")
    assert sum(1 for row in lr if any(s["corrupted"] for s in row["policies"]["blind"])) == 1 and not any(s["corrupted"] and s["t"] <= 10 for s in bl)   # 0 of 6 by step 10
    def mean_sec(rows):
        v = [s["seconds"] for r in rows for st in r["policies"].values() for s in st if not s.get("cached")]; return sum(v) / len(v)
    assert (round(mean_sec(ln), 2), round(mean_sec(lr), 2)) == (0.55, 0.64)


def test_the_reading_rules_of_the_notebook_16_lines_9_yes_7_no():
    R = load()
    if R is None: return
    s_r, s_n = summarize_scope(R["sr"]["rows"], tasks()), summarize_scope(R["sn"]["rows"], tasks())
    p1 = verdict_big_single({True: s_r, False: s_n}); p2 = verdict_big_long(R["lr"]["summary"], R["ln"]["summary"])
    fmt = verdict_api_format({True: R["sr"]["rows"], False: R["sn"]["rows"]}, {True: R["lr"]["rows"], False: R["ln"]["rows"]})
    assert [ok for _, ok in p1] == [True, True, True, True, False, True, False, False] and [ok for _, ok in p2] == [True, True, True, False, False, True, False] and [ok for _, ok in fmt] == [False]
    lines = p1 + p2 + fmt; assert len(lines) == 16 and sum(ok for _, ok in lines) == 9
    no = [t for t, ok in lines if not ok]
    assert sum(t.startswith("[need]") for t in no) == 4 and sum(t.startswith("[guard]") for t in no) == 2 and sum(t.startswith("[format]") for t in no) == 1
    assert "(78 of 651)" in fmt[0][0]
    # the two [guard] misses are the rule-stated ones; the four [need] misses include the no-rule delete_dep and the no-rule step-10 line
    assert any("rule stated: scope_m7 solves at least 90 percent of the old delete" in t for t in no) and any("rule stated: scope_m7 holds the oracle's state on at least 90 percent" in t for t in no)
    assert any("no rule: blind apply is right on at most half of the 20 delete_dep" in t for t in no) and any(t.startswith("[need] no rule: blind apply is corrupted by step 10") for t in no)
    # the lines flip when a number is doctored: more broken replies in the single file
    bad = copy.deepcopy(R["sn"]["rows"])
    for r in bad[:10]: r["text"] = "oops"
    assert not verdict_api_format({False: bad}, {})[0][1] and verdict_api_format({False: R["sn"]["rows"]}, {})[0][1]
    s_bad = copy.deepcopy(R["sn"]["rows"]); dd = [i for i, r in enumerate(s_bad) if r["kind"] == "delete_dep"]
    for i in dd[:12]: s_bad[i]["text"] = "DEL[0,0]"                                             # blind apply wrong on many delete_dep: the need line turns yes
    assert verdict_big_single({True: s_r, False: summarize_scope(s_bad, tasks())})[4][1] is True


def test_the_exploratory_offline_re_read_and_the_page():
    R = load()
    if R is None: return
    for key, want in (("sr", (137, 136, 138, 138, 1)), ("sn", (123, 122, 138, 138, 15))):
        rows = [dict(r, text=fix(r["text"])) for r in R[key]["rows"]]
        assert all(strict_parse(r["text"]) is not None for r in rows)
        s = summarize_scope(rows, tasks())["all"]; assert (s["blind"]["success"], s["gate"]["success"], s["repair_m6"]["success"], s["scope_m7"]["success"], s["blind"]["corrupted"]) == want, key
        assert s["gate"]["corrupted"] == s["repair_m6"]["corrupted"] == s["scope_m7"]["corrupted"] == 0
    res = page()
    if res is None: return
    assert res.count(RESULT_MARK) == 1 and res.index(RESULT_MARK) > res.index("## Addendum 2 (10 Oct 2026")
    body = res[res.index(RESULT_MARK):]
    for q in ("78 of 651 replies do not parse (12.0 %", "(18; the format slip", "all 78 unparsed replies parse (78 of 78)", "137, gate 136, repair_m6 138, scope_m7 138 of 140", "blind apply 123, gate 122, repair_m6 138, scope_m7 138",
              "354 edge-slots against 254", "38,263 against 37,038", "59 of its 120 steps", "**120** | **119** | **135** | **135**", "16 lines: 9 yes, 7 no", "2,536 output tokens", "3,025", "0 of 900 long-run policy steps",
              "all 18 policy-episodes", "Part B (Gemini 3.8 Flash"):
        assert q in body, q
    assert "AIza" not in res


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn): fn(); print("PASS", name); n += 1
    print(f"{n} tests passed")
