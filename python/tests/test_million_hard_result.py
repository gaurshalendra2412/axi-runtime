"""Implements concept-map item 95 (the hard million-token test on Gemini 3.8 Flash; rows 3, 9, 27): the RESULT of run 1 and of the contrast in docs/MILLION_TOKEN_TEST_HARD.md, recomputed from the three saved files. It checks that
  1. the saved files are the registered run and the registered contrast (model, stream, seed, steps, policies, two re-asks, the repair deleting up to 8 dependants, no scripted reply), that run 1's rows are, call for call, the replies
     in the saved-reply file, that the file says what the page says about it (4,020 calls that all ended with a normal stop, no reply rewritten by the adapter, two retries after a busy server, a 20-step check under the thinking
     trigger), and that every arm of the contrast replays to its saved flags, step by step, from its saved replies;
  2. the fourteen reading-rule lines recomputed by the page's own `verdict_hard` are the ones printed in the Result (line 4 NO, line 13 yes, line 14 not run), the headline is PASSED, and the sentence the page allows is built
     from numbers that follow from the files (3.7 million tokens, 100 percent, 0 items, never corrupted, blind apply corrupted on 55.8 percent of its first 400 steps);
  3. every number of the Result follows from the files: tokens, time, thinking, the ten tenths, the 66 / 20 / 46 replies that differ from the ideal reply and the kind of request they are in, the dependants left out, the one-step
     replay of each policy from the oracle's state, the four arms of the contrast (steps exactly right, steps corrupted, distances, offsets and how long they lasted, how the blind arm's corrupted stretches ended, tokens and cost),
     and the predictions the page says came true or not;
  4. the tables on the page are, character for character, what `million_hard_result` renders from the files, every sentence of the Result that carries a figure is built here from the data, and the check notices a doctored file;
  5. what a step with the right state cost (added after the contrast was read; not one of the reading rules): recomputed here from the raw rows of the four arms, with every call counted, and the page carries no energy figure and no claim about any market;
  6. the page's text above the Result is still the registered text (the pinned hash lives in tests/test_million_hard.py, repeated here) and the Result occurs once; nothing the run used imports the result module.
What it does NOT show: anything the page does not measure (the control and run 2 are not in these files; a chat that keeps its history, an agent that corrects its own state, a Pro model, another provider)."""
import copy, datetime, functools, hashlib, json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
from axi.experiments import million as M
from axi.experiments import million_hard as H
from axi.experiments import million_hard_result as R

HERE = os.path.dirname(__file__)
PAGE = os.path.join(HERE, "..", "..", "docs", "MILLION_TOKEN_TEST_HARD.md")
PAGE_SHA256 = "9f786621d81c62eb717cdf518a3e4aca1657c2aae09266030f5145f20ce00b59"
RESULT_MARK = "\n\n---\n\n# Result"
IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
RUN_LABELS = ["run 1"] * 19 + ["run 2"] * 3 + ["Run 1", "Run 2"]            # how often the Result says "run 1" and "run 2" (the labels of the main run and of the replication); a section added later changes these counts


def _h(text): return hashlib.sha256((text.rstrip("\n") + "\n").encode()).hexdigest()


@functools.lru_cache(maxsize=None)
def _run(): return R.load()


@functools.lru_cache(maxsize=None)
def _a(): return R.analyse(_run())


@functools.lru_cache(maxsize=None)
def _cache():
    with open(os.path.join(R.RESULTS, R.CACHE1_FILE)) as f: return [json.loads(x) for x in f if x.strip()]


def _text(): return open(PAGE).read()


def _result_text(): t = _text(); assert t.count(RESULT_MARK) == 1; return t[t.index(RESULT_MARK):]


@functools.lru_cache(maxsize=None)
def _con(): return R.load_contrast()


@functools.lru_cache(maxsize=None)
def _cx(): return R.analyse_contrast(_run(), _con())


@functools.lru_cache(maxsize=None)
def _cc(): return R.analyse_cost(_run(), _con())


def _ok_calls(): return [x for x in _cache()[1:] if "fail" not in x]


def test_the_saved_files_are_the_registered_run_and_agree_with_each_other():
    run = _run(); rows, meta = run["rows"], run["meta"]
    assert meta["model"] == "gemini-3.8-flash" and meta["thinking"] is None and meta["seed"] == 20261031 and meta["steps"] == 4000 and meta["policy"] == "scope_m7" and meta["stream"] == "HardStream"
    assert meta["reasks"] == 2 and meta["degree_threshold"] == 8 and meta["hint_rules"] is False and meta.get("scripted", 0) == 0 and H.registered(meta, "scope_m7", 4000) and run["stopped"] is None
    assert [r["t"] for r in rows] == list(range(1, 4001)) and not any(r["cached"] for r in rows) and all(r["parsed"] for r in rows) and all("asks" not in r and "reasks" not in r for r in rows)
    c = _cache(); assert c[0] == {"_meta": dict(provider="gemini", model="gemini-3.8-flash", thinking=None, temperature=None, max_output_tokens=8192)}
    ok = _ok_calls(); assert len(ok) == 4020 and len({x["k"] for x in ok}) == 4020 and all(x["finish"] == "FinishReason.STOP" for x in ok) and not any(x["normalized"] for x in ok)
    assert [x for x in c[1:] if "fail" in x] == [x for x in c[1:] if x.keys() == {"fail", "ts"}] and sum(1 for x in c[1:] if x.get("fail") == "retry") == 2 and len(c) == 1 + 4020 + 2
    run_calls = ok[20:]                                                                                           # the first 20 are the 20-step check, on another stream
    assert [(x["text"], x["tokens"], x["think"], x["prompt_tokens"]) for x in run_calls] == [(r["text"], r["tokens"], r["think"], r["prompt_tokens"]) for r in rows]
    assert all(abs(x["seconds"] - r["seconds"]) < 1e-3 for x, r in zip(run_calls, rows))
    f = R.cache_facts(os.path.join(R.RESULTS, R.CACHE1_FILE)); assert (f["calls"], f["retries"], f["stop"], f["normalized"]) == (4020, 2, 4020, 0)
    retries = [i for i, x in enumerate(c[1:]) if x.get("fail") == "retry"]; before = [sum(1 for y in c[1:][:i] if "fail" not in y) - 20 for i in retries]
    assert before == [1283, 3217]
    gaps = [c[1:][i + 1]["ts"] - c[1:][i - 1]["ts"] for i in retries]; assert all(115 <= g <= 130 for g in gaps)                 # about two minutes each
    first, last = (datetime.datetime.fromtimestamp(ok[20]["ts"], IST), datetime.datetime.fromtimestamp(ok[-1]["ts"], IST))
    assert (first.month, first.day, first.hour, first.minute) == (10, 13, 21, 30) and (last.month, last.day, last.hour, last.minute) == (10, 14, 0, 26) and round((ok[-1]["ts"] - ok[20]["ts"]) / 3600, 1) == 2.9
    chk = ok[:20]; pt, th, vis, sec = (sum(x[k] for x in chk) / 20 for k in ("prompt_tokens", "think", "tokens", "seconds"))
    assert round(pt, 1) == 723.8 and round(th, 1) == 92.5 and round(sec, 2) == 2.11 and round(pt + th + vis) == 840 and th < H.HARD_THINKING_LIMIT
    assert round(4000 * (pt * R.PRICE_IN + (th + vis) * R.PRICE_OUT) / 1e6, 2) == 3.91                                  # the check's projection for 4,000 steps at the typed-in prices


def test_the_contrast_file_is_the_registered_contrast_every_arm_replays_and_its_numbers_follow():
    con, run = _con(), _run(); cx = _cx(); assert list(con) == ["blind", "gate", "repair_m6"] and set(con) == set(M.CONTRAST_POLICIES)
    for p, d in con.items():
        m = d["meta"]; rows = d["rows"]
        assert m["model"] == "gemini-3.8-flash" and m["thinking"] is None and m["seed"] == 20261031 and m["steps"] == 400 == len(rows) and m["policy"] == p and m["stream"] == "HardStream" and m["reasks"] == 2 and m["degree_threshold"] == 8 and m["hint_rules"] is False and m.get("scripted", 0) == 0
        assert H.registered(m, p, 400) and d["stopped"] is None and [r["t"] for r in rows] == list(range(1, 401)) and all(r["parsed"] for r in rows) and all("asks" not in r for r in rows)
        _, _, bad = M.replay(rows, m["seed"], m["policy"], strict=False, stream=m["stream"], reasks=m["reasks"], degree_threshold=m["degree_threshold"]); assert bad == [], p        # every saved flag of every step is reproduced from the saved replies
        for r in rows:                                                                                                                                                         # a reused reply is run 1's reply to the same prompt (the same step number, or the same text)
            if r["cached"]: assert r["text"] in {x["text"] for x in run["rows"]}
    c = {p: d["rows"] for p, d in con.items()}; assert all(x["refused"] == 0 for x in cx.values())                           # no reply of any arm was refused
    r16 = {p: c[p][15] for p in c}                                                                                            # step 16: the same saved reply for all three; what each arm did with it
    assert r16["blind"]["corrupted"] and not r16["blind"]["exact"] and not r16["gate"]["corrupted"] and not r16["gate"]["exact"] and r16["gate"]["nodes"] > r16["gate"]["oracle_nodes"] and r16["repair_m6"]["exact"]
    stream = M.make_stream("HardStream", 20261031); first = None                                                             # the first of run 1's replies that left items out of the ideal reply is step 16
    for rec in run["rows"]:
        st, _ = stream.next(); d = R.strict_parse(rec["text"]); ideal = R.T(st["ideal"])
        if rec["text"].strip() != st["ideal"].strip() and set(R._items(d)) != set(R._items(ideal)): first = rec["t"]; break
    assert first == 16
    assert [sum(1 for r in c[p] if r["cached"]) for p in ("blind", "gate", "repair_m6")] == [86, 85, 400] and [sum(1 for r in c[p] if not r["cached"]) for p in ("blind", "gate", "repair_m6")] == [314, 315, 0]
    assert [r["text"] for r in c["repair_m6"]] == [r["text"] for r in run["rows"][:400]] and [r["exact"] for r in c["repair_m6"]] == [True] * 400 and all(r["cached"] for r in c["repair_m6"])
    assert [r["text"] for r in c["blind"][:16]] == [r["text"] for r in c["gate"][:16]] == [r["text"] for r in run["rows"][:16]] and all(r["cached"] for r in c["blind"][:16]) and run["rows"][15]["text"] == "DEL[0,0] DEL[(0,2)->(1,1):owns#7]"
    assert [M.tokens_sent(c[p]) for p in ("blind", "gate", "repair_m6")] == [253132, 272660, 0] and sum(1 for p in c for r in c[p] if not r["parsed"]) == 0
    assert (round(cx["blind"]["dollars"], 2), round(cx["gate"]["dollars"], 2), round(cx["blind"]["dollars"] + cx["gate"]["dollars"], 2), cx["repair_m6"]["dollars"]) == (0.32, 0.37, 0.69, 0.0)
    got = {a: (x["exact"], x["corrupted"], round(x["mean_201_400"], 2), round(x["mean_all"], 2), x["max_distance"], len(x["new_offsets"]), x["inapplicable"]) for a, x in cx.items()}
    assert got == {"scope_m7": (400, 0, 0.0, 0.0, 0, 0, 0), "blind": (85, 223, 1.83, 1.6, 5, 4, 9), "gate": (84, 0, 4.63, 3.55, 8, 3, 16), "repair_m6": (400, 0, 0.0, 0.0, 0, 0, 0)}
    assert cx["blind"]["new_offsets"] == [16, 132, 210, 361] and cx["gate"]["new_offsets"] == [16, 132, 210] and cx["blind"]["new_offset_kinds"] == cx["gate"]["new_offset_kinds"] == ["delete_pair"]
    assert cx["blind"]["off"] == [(16, 104), (132, 167), (210, 359), (361, 400)] and cx["blind"]["corrupted_stretches"] == [(16, 87), (132, 151), (210, 259), (262, 302), (361, 400)] and cx["gate"]["off"] == [(16, 104), (132, 167), (210, 400)] and cx["gate"]["corrupted_stretches"] == []
    assert (round(cx["gate"]["last100_distance"], 2), cx["gate"]["last100_exact"], cx["blind"]["last100_exact"]) == (5.5, 0, 1) and R.blind_stretch_ends(con) == [(88, "add_node", "service added back"), (152, "add_node", "service added back"), (260, "delete_hub", "leftover edges deleted"), (303, "delete_hub", "leftover edges deleted")]
    assert round(sum(r["distance"] for r in c["gate"][:100]) / 100, 2) == 3.14 and round(sum(r["distance"] for r in c["gate"][300:]) / 100, 2) == 5.5
    t = _result_text(); assert "the first 15 steps" in t and "86 steps of blind and 85 of gate" in t


def test_the_fourteen_lines_the_headline_and_the_sentence_follow_from_the_files():
    run = _run(); con = _con(); v = H.verdict_hard(run, con); text = _result_text()
    assert [a for _, _, a in v] == ["yes", "yes", "yes", "NO", "yes", "yes", "yes", "yes", "yes", "yes", "yes", "yes", "yes", "not run"] and H.headline(v) == "PASSED"
    assert [a for _, _, a in H.verdict_hard(run)] == ["yes", "yes", "yes", "not run", "yes", "yes", "yes", "yes", "yes", "yes", "yes", "yes", "not run", "not run"]       # without the contrast, 4 and 13 are 'not run'
    assert R.lines_block(v) in text and "**Headline (lines 1, 2, 3, 5, 6, 7, 8, 9, 10): PASSED.**" in text
    rows = run["rows"]; assert M.tokens_sent(rows) == 3661209 and round(M.tokens_sent(rows) / 1e6, 1) == 3.7 and all(r["exact"] for r in rows) and max(r["distance"] for r in rows) == 0 and not any(r["corrupted"] for r in rows)
    assert H.report_hard(run)["new_offsets"] == 0 and H.report_hard(run)["reasked_steps"] == 0 and H.report_hard(run)["extra_calls"] == 0 and H.report_hard(run)["dropped_items"] == 0 and M.wrong_rate(rows) == (0, 4000)
    b = _cx()["blind"]; assert (b["corrupted"], b["steps"]) == (223, 400) and round(100 * b["corrupted"] / b["steps"], 1) == 55.8 and b["corrupted"] < H.H_BLIND_MIN * b["steps"] and 0.8 * 400 == 320      # line 4: under the bar of 80 percent
    s7, g, bl = _cx()["scope_m7"], _cx()["gate"], _cx()["blind"]; assert s7["mean_201_400"] == 0 <= H.H_CONTRAST_MEAN and s7["mean_201_400"] < g["mean_201_400"] and s7["mean_201_400"] < bl["mean_201_400"]       # line 13
    assert (round(bl["mean_201_400"], 2), round(g["mean_201_400"], 2)) == (1.83, 4.63)
    for needle in ("(3.7 million tokens sent)", "was exactly the right state on 100 percent of the steps (4,000 of 4,000)", "(it was 100 percent in every tenth), was never more than 0 items from it, and was never corrupted; blind application of the same model's replies was corrupted on 55.8 percent of its first 400 steps.",
                   "Line 13 reads yes; **line 4 reads NO** (it is not a headline line); line 14 needs run 2 and is \"not run\"", "It does not carry \"and on a second seed\" (line 14, not run).",
                   "Blind apply was corrupted on 223 of its first 400 steps (55.8 percent), under the bar of 80 percent.", "it is a statement about how often the model broke the raw state on this stream, not a failure of anything"): assert needle in text, needle
    assert "Y percent" not in text                                                                                                              # the clause is filled in, not quoted as pending


def test_the_numbers_of_the_result_follow_from_the_file():
    a = _a(); t = a["tokens"]; text = _result_text()
    assert (t["prompt"], t["visible"], t["think"], t["total"], t["calls"]) == (2926781, 80984, 653444, 3661209, 4000) and t["prompt"] + t["visible"] + t["think"] == t["total"] and round(100 * t["think"] / t["total"], 1) == 17.8
    s = a["seconds"]; rows = _run()["rows"]; srt = sorted(r["seconds"] for r in rows)
    assert round(s["total"]) == 10198 and round(s["total"] / 3600, 2) == 2.83 and round(s["mean"], 2) == 2.55 and round((srt[1999] + srt[2000]) / 2, 2) == 1.82 and s["over_10"] == 60 and round(100 * s["over_10"] / 4000, 1) == 1.5 and round(s["max"], 1) == 62.5
    k = a["think"]; assert round(k["mean"], 1) == 163.4 and k["median"] == 54 and k["p95"] == 573 and k["over_1000"] == 61 and k["max"] == 7862 and k["max"] < 8192
    d = a["dollars"]; assert round(d["total"], 2) == 4.95 and round(d["thinking"], 2) == 2.45 and (R.PRICE_IN, R.PRICE_OUT) == (0.75, 3.75)
    ten = a["tenths"]; assert [x["first"] for x in ten] == list(range(1, 4000, 400)) and all(x["steps"] == 400 and x["exact"] == 1.0 and x["distance"] == 0 for x in ten)
    assert round(ten[0]["prompt"], 1) == 648.8 and round(ten[-1]["prompt"], 1) == 755.4 and round(ten[-1]["prompt"] / ten[0]["prompt"], 2) == 1.16 and round(ten[-1]["seconds"], 2) == 2.41 and round(ten[0]["seconds"], 2) == 2.57 and round(ten[-1]["seconds"] / ten[0]["seconds"], 2) == 0.94
    big = max(range(10), key=lambda i: ten[i]["prompt"]); small = min(range(10), key=lambda i: ten[i]["prompt"])
    assert (big + 1, round(ten[big]["prompt"], 1), round(ten[big]["edges"], 1), small + 1, round(ten[small]["prompt"], 1), round(ten[small]["edges"], 1)) == (6, 882.2, 37.0, 7, 618.6, 18.4)
    for needle in ("3,661,209 = 2,926,781 prompt + 80,984 visible reply + 653,444 thinking", "Thinking was 17.8 percent of the tokens", "10,198 seconds in calls (2.83 hours), 2.55 a call on average, 1.82 the median", "60 calls (1.5 percent) took more than 10 seconds and the longest took 62.5",
                   "$4.95 for the run, of which $2.45 is thinking", "The mean was 163.4 tokens a step but the median was 54 (95th percentile 573); 61 calls used more than 1,000 and the largest used 7,862 of the 8,192 allowed",
                   "(1.16 times)", "(0.94 times)", "tenth 6 (882.2 tokens; the graph had 37.0 edges on average) and the smallest in tenth 7 (618.6; 18.4 edges)", "723.8 prompt tokens, 92.5 thinking tokens and 2.11 seconds a call, 840 tokens a step in all",
                   "4,020 calls ended with a normal stop", "after calls 1,283 and 3,217 of the run", "13 Oct 21:30 to 14 Oct 00:27", "2.8 hours of calls in about 2.9 hours"): assert needle in text, needle


def test_what_the_model_did_and_what_each_policy_would_have_done_follow_from_the_replies():
    a = _a(); text = _result_text(); kd = a["kinds"]
    assert {n: (v["requests"], v["same_text"], v["other_order"], v["other_items"], v["off"]) for n, v in kd.items()} == {
        "add_edge": (2410, 2410, 0, 0, 0), "add_node": (739, 739, 0, 0, 0), "delete_hub": (346, 342, 4, 0, 0), "delete_dep": (126, 118, 8, 0, 0), "delete_leaf": (129, 129, 0, 0, 0), "delete_edge": (121, 121, 0, 0, 0), "delete_pair": (129, 75, 8, 46, 0)}
    tot = lambda key: sum(v[key] for v in kd.values()); assert (tot("requests"), tot("same_text"), tot("other_order"), tot("other_items"), tot("off")) == (4000, 3934, 20, 46, 0) and 4000 - 3934 == 66
    assert a["only_left_out"] == 46                                                                                    # every one of the 46 is the ideal reply with items missing, none added or changed
    assert {d: tuple(v) for d, v in a["pair"].items()} == {0: (32, 0), 1: (29, 21), 2: (36, 14), 3: (32, 11)} and sum(v[1] for v in a["pair"].values()) == 46 and sum(v[0] for k, v in a["pair"].items() if k) == 97 and round(100 * 46 / 97) == 47
    assert kd["delete_hub"]["same_text"] + kd["delete_hub"]["other_order"] + kd["delete_dep"]["same_text"] + kd["delete_dep"]["other_order"] == 472 and kd["delete_hub"]["other_order"] + kd["delete_dep"]["other_order"] == 12
    assert {p: tuple(v) for p, v in a["counterfactual"].items()} == {"blind": (46, 46), "gate": (46, 0), "repair_m6": (0, 0), "scope_m7": (0, 0)} and a["counterfactual_default"] == {"repair_m6": 0, "scope_m7": 0}
    assert round(100 * 46 / 4000, 2) == 1.15 and 46 / 4000 > H.H_WRONG_MAX and 3954 == 4000 - 46                       # over the page's own bar for line 10, as a replay and not as a rule line
    for needle in ("66 of the 4,000 replies differ from the ideal reply as text. **20 have the right items in another order** (8 of the deletes with one to three dependants, 4 hub deletes, 8 service-and-edge deletes)",
                   "**46 have other items, and in all 46 the reply is the ideal reply with items left out: none added, none changed.**", "the model left some out in 46 of 97 replies (47 percent); when it has none there is nothing to leave out (0 of 32)",
                   "**it named every dependant every time: 472 of 472** (12 in another order)", "The dependants left out were never more than 3.", "changed the outcome of **46 steps (1.15 percent)**", "0 steps asked twice, 0 extra calls", "removed 0 items in 0 steps",
                   "**no offset ever started.**", "**which items were left out is the service's dependants**", "it was right on 3,954 of 4,000 replies from a correct state", "46 of 4,000 (1.15 percent), which is over that bar", "made no difference to the 4,000 replies of this run",
                   "**1,825 for the service-and-edge request**", "47 to 67 for adding a service or an edge and for deleting an edge, 325 to 422 for deleting a service"): assert needle in text, needle
    # what was left out is the deleted service's dependants: the reply is the service and the named edge, and every missing item is an edge that touches the deleted service
    stream = M.make_stream(_run()["meta"]["stream"], _run()["meta"]["seed"]); left_out = 0
    for rec in _run()["rows"]:
        st, ob = stream.next()
        if rec["kind"] != "delete_pair" or rec["text"].strip() == st["ideal"].strip(): continue
        got, want = rec["text"].split(" "), st["ideal"].split(" ")
        if sorted(got) == sorted(want): continue                                                                            # same items in another order
        node = want[0]; assert got[0] == node and len(got) == 2 and set(got) < set(want); (r, c) = (int(x) for x in node[4:-1].split(","))
        missing = [x for x in want if x not in got]; assert missing and all(x.startswith("DEL[(") and (f"({r},{c})->" in x or f"->({r},{c}):" in x) for x in missing); left_out += 1
    assert left_out == 46 and sum(1 for x in _run()["rows"] if x["think"] == 1000) == 0                                    # (no call used exactly 1,000 thinking tokens, so "more than" and "at least" give the same count)
    th = {n: round(v["think"] / v["requests"], 1) for n, v in kd.items()}
    assert th["delete_pair"] == 1824.7 and 47 <= min(th["add_node"], th["add_edge"], th["delete_edge"]) and max(th["add_node"], th["add_edge"], th["delete_edge"]) < 67 and 325 <= min(th["delete_dep"], th["delete_hub"], th["delete_leaf"]) and max(th["delete_dep"], th["delete_hub"], th["delete_leaf"]) < 423


def test_the_predictions_the_page_says_came_true_or_not_follow_from_the_file():
    run = _run(); rows = run["rows"]; a = _a(); text = _result_text(); ck = M.tokens_sent(rows)
    assert 2_800_000 <= ck <= 4_000_000 and 1.0 <= a["seconds"]["mean"] <= 5.0 and 2 <= a["dollars"]["total"] <= 8 and 2 <= 3.91 <= 8
    assert sum(1 for r in rows if not r["parsed"]) == 0 and sum(r["exact"] for r in rows) / 4000 == 1.0 and not (0.60 <= 1.0 <= 0.99)               # refused 0.0 (not 0.1 to 2.0 percent); exact 100 (not 60 to 99)
    assert a["think"]["mean"] > 150 and _ok_calls()[:20] and sum(x["think"] for x in _ok_calls()[:20]) / 20 < 150                                  # the run averaged more than 150, the check did not: the page's trigger is the check's number
    assert a["kinds"]["add_edge"]["other_items"] == 0 and a["kinds"]["add_edge"]["requests"] == 2410
    for needle in ("(rated 85, 99, 93, 90, **30, 40, 45**, 65, 80, 97, 60 and 45 percent)", "**the model's rate of slips on this job was lower than I assumed**", "all nine headline lines yes, rated about 20 percent: it happened",
                   "thinking above 150 a step at the default (65 percent): no, 92.5", "Tokens in run 1 between 2.8 and 4.0 million (60 percent): yes, 3.66", "the 20-step check projected $3.91 and the arithmetic on the run is $4.95",
                   "Mean seconds of a call between 1.0 and 5.0 (65 percent): yes, 2.55", "between 0.1 and 2.0 percent (60 percent): no, none (0.0)", "no step was re-asked, so this cannot be judged", "Share exact between 60 and 99 percent (65 percent): no, 100",
                   "(0 of 2,410 add-edge requests), so the second is wrong and the first cannot be judged", "the control has not been run",
                   "**Line 4 (rated 55 percent): NO**, blind apply was corrupted on 55.8 percent of its first 400 steps, under 80. **Line 13 (45 percent): yes.** The guard and instrument lines 1 to 4 all yes (43 percent): did not happen, because of line 4. All fourteen lines yes (under 3 percent): cannot happen now, since line 4 is NO and line 14 is not run."): assert needle in text, needle
    head = _text()[:_text().index(RESULT_MARK)]                                                                         # the percentages quoted are the ones written before the run
    for needle in ("(85 %)", "(99 %)", "(93 %)", "(90 %)", "(30 %)", "(40 %)", "(45 %)", "(65 %)", "(80 %)", "(97 %)", "(60 %)", "(55 %)", "about **43 %**", "about **20 %**", "**under 3 %**"): assert needle in head, needle


def test_the_tables_on_the_page_are_what_the_code_renders_and_a_doctored_file_is_noticed():
    a = _a(); text = _result_text()
    for block in (R.table_tenths(a), R.table_kinds(a), R.table_dependants(a), R.table_counterfactual(a), R.table_contrast(_cx()), R.table_cost(_cc())): assert block in text
    assert text.count("\n|") == 12 + 10 + 7 + 6 + 6 + 6 and R.table_cost(_cc()).replace("| 85 | 333,040 |", "| 86 | 333,040 |") not in text and "| 85 | 333,040 |" in text and R.table_contrast(_cx()).replace("| 85 | 223 |", "| 86 | 223 |") not in text and "| 85 | 223 |" in text and R.table_counterfactual(a).replace("| 46 | 46 |", "| 45 | 46 |") not in text and "| 46 | 46 |" in text
    # a doctored saved file does not give the printed numbers: put one dependant back into one of the 46 replies, and the count of replies with items left out is 45
    run = copy.deepcopy(_run()); stream = M.make_stream(run["meta"]["stream"], run["meta"]["seed"]); fixed = False
    for rec in run["rows"]:
        st, ob = stream.next()
        if not fixed and rec["kind"] == "delete_pair" and rec["text"].strip() != st["ideal"].strip() and set(rec["text"].split(" ")) < set(st["ideal"].split(" ")): rec["text"] = st["ideal"]; fixed = True
    assert fixed; b = R.analyse(run); assert b["kinds"]["delete_pair"]["other_items"] == 45 and b["counterfactual"]["blind"][0] == 45 and R.table_kinds(b) not in text
    # a doctored flag is noticed by line 1 (the replay of the saved replies does not reproduce it), and a missing step is
    bad = copy.deepcopy(_run()); bad["rows"][1000]["exact"] = False; assert H.verdict_hard(bad)[0][2] == "NO"
    short = copy.deepcopy(_run()); short["rows"] = short["rows"][:3999]; assert H.verdict_hard(short)[0][2] == "NO"
    other = copy.deepcopy(_run()); other["meta"]["degree_threshold"] = 4; assert H.verdict_hard(other)[0][2] == "NO"                  # a file made with another setting is not the registered run
    # a doctored contrast: one blind step less corrupted does not give the printed table; a contrast made with another setting is 'not run'; a contrast with a missing step is 'not run'
    con = copy.deepcopy(_con()); k = next(i for i, r in enumerate(con["blind"]["rows"]) if r["corrupted"]); con["blind"]["rows"][k]["corrupted"] = False
    assert R.table_contrast(R.analyse_contrast(_run(), con)) not in text and R.analyse_contrast(_run(), con)["blind"]["corrupted"] == 222
    con = copy.deepcopy(_con()); con["gate"]["meta"]["degree_threshold"] = 4; assert H.verdict_hard(_run(), con)[3][2] == "not run" and H.verdict_hard(_run(), con)[12][2] == "not run"
    con = copy.deepcopy(_con()); con["repair_m6"]["rows"].pop(); assert H.verdict_hard(_run(), con)[3][2] == "not run"
    con = copy.deepcopy(_con()); [r.update(distance=r["distance"] + 5) for r in con["gate"]["rows"][200:]]; assert H.verdict_hard(_run(), con)[12][2] == "yes"        # a worse gate keeps line 13 yes
    con = copy.deepcopy(_con()); [r.update(distance=0) for r in con["gate"]["rows"][200:]]; assert H.verdict_hard(_run(), con)[12][2] == "NO"                       # a gate as good as scope_m7 is not 'smaller': line 13 NO


def _head_ratings():
    """The percentages written before the run: {prediction number: first percentage of its paragraph}, and the ten percentages of paragraph 15 in order."""
    t = _text(); head = t[:t.index(RESULT_MARK)]; pred = head[head.index("## Predictions"):head.index("## Reading rules")]
    items = re.split(r"\n(?=\d+\. )", pred); out = {}; p15 = ""
    for it in items:
        m = re.match(r"(\d+)\. ", it)
        if m:
            k = int(m.group(1)); out[k] = int(re.search(r"\((\d+) %\)", it).group(1)) if re.search(r"\((\d+) %\)", it) else None
            if k == 15: p15 = it
    return head, out, p15


def test_every_sentence_that_carries_a_number_is_built_from_the_data():
    """Each sentence of the Result with a figure in it is rendered here from the saved files (and from the constants and the percentages written before the run) and must be on the page character for character."""
    run, a, text = _run(), _a(), _result_text(); rows, meta = run["rows"], run["meta"]; n = len(rows); t = a["tokens"]; kd = a["kinds"]; ten = a["tenths"]; cf = a["counterfactual"]
    two = {2: "two"}[meta["reasks"]]; w = {3: "three", 4: "Four"}; deg = meta["degree_threshold"]; head, rate, p15 = _head_ratings(); ok = _ok_calls(); chk = ok[:20]; c = _cache(); fct = R.cache_facts(os.path.join(R.RESULTS, R.CACHE1_FILE))
    exact_n = sum(1 for r in rows if r["exact"]); maxd = max(r["distance"] for r in rows); lo, hi = round(min(x["nodes"] for x in ten)), round(max(x["nodes"] for x in ten)); mn = 100 * min(x["exact"] for x in ten)
    pt, th, vis, sec = (sum(x[k] for x in chk) / len(chk) for k in ("prompt_tokens", "think", "tokens", "seconds")); r1 = ten[-1]["prompt"] / ten[0]["prompt"]; r2 = ten[-1]["seconds"] / ten[0]["seconds"]
    off = a["kinds"]["delete_pair"]["other_items"]; pair = a["pair"]; hours = a["seconds"]["total"] / 3600; wall = (ok[-1]["ts"] - ok[20]["ts"]) / 3600
    retries = [i for i, x in enumerate(c[1:]) if x.get("fail") == "retry"]; before = [sum(1 for y in c[1:][:i] if "fail" not in y) - 20 for i in retries]
    import inspect
    from axi.engine.diagnostics import ground_and_complete
    assert inspect.signature(ground_and_complete).parameters["degree_threshold"].default == 4 and (H.HUB_MIN, H.HUB_MAX) == (4, 8) and deg == H.HARD_DEGREE == 8 and meta["reasks"] == H.HARD_REASKS
    assert "did not do what was asked" in H.HARD_LINES[9][1] and H.H_WRONG_MAX == 0.01 and [l for l in H.HARD_LINES][13][1].find("second seed") > 0 and "scope_m7's mean distance" in H.HARD_LINES[12][1] and "blind apply" in H.HARD_LINES[3][1] and H.HARD_CONTRAST_STEPS == 400 and H.HARD_CONTROL_STEPS == 1000 and H.HEADLINE_LINES == (1, 2, 3, 5, 6, 7, 8, 9, 10)
    nbt = open(os.path.join(HERE, "..", "colab", "axi_million_hard.ipynb")).read()
    for h in ("### Cell 8: CONTRAST", "### Cell 9 (optional): CONTROL", "### Cell 10 (optional): RUN 2"): assert h in nbt, h
    yes_lines = [i for i, (_, _, ans) in enumerate(H.verdict_hard(run, _con()), 1) if ans == "yes"]; assert yes_lines == [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    lines_txt = ", ".join(str(i) for i in yes_lines[:-1]) + " and " + str(yes_lines[-1]); bl, g, s7 = _cx()["blind"], _cx()["gate"], _cx()["scope_m7"]; cons = _cx()
    rated = ", ".join(str(rate[i]) for i in yes_lines[:-1]) + " and " + str(rate[yes_lines[-1]]); rated = rated.replace(str(rate[6]) + ", " + str(rate[7]) + ", " + str(rate[8]), "**" + str(rate[6]) + ", " + str(rate[7]) + ", " + str(rate[8]) + "**")
    m15 = re.search(r"equal to the oracle over the whole run is between (\d+) and (\d+) percent \((\d+) %\)", p15); assert m15
    overall = re.search(r"about \*\*(\d+) %\*\*", head).group(1)
    assert rate[13] == 45 and rate[4] == 55 and rate[6] == 30 and overall == "20" and rate[1] == 85 and rate[2] == 99 and rate[3] == 93 and rate[5] == 90 and rate[9] == 65 and rate[10] == 80 and rate[11] == 97 and rate[12] == 60
    fragments = [
        f"On Gemini 3.8 Flash, in one sampled run of {n:,} requests ({t['total'] / 1e6:.1f} million tokens sent) on a dense service graph ({lo} to {hi} services on average), the held state kept by the framework (scope filter, deterministic repair deleting up to {deg} dependants, up to {two} re-asks) "
        f"was exactly the right state on {100 * exact_n / n:.0f} percent of the steps ({exact_n:,} of {n:,}) and at least {round(100 * H.H_TENTH_EXACT)} percent in every tenth (it was {mn:.0f} percent in every tenth), was never more than {maxd} items from it, and was never corrupted; blind application of the same model's replies was corrupted on {100 * bl['corrupted'] / bl['steps']:.1f} percent of its first {H.HARD_CONTRAST_STEPS} steps.",
        f"Blind apply was corrupted on {bl['corrupted']} of its first {H.HARD_CONTRAST_STEPS} steps ({100 * bl['corrupted'] / bl['steps']:.1f} percent), under the bar of {round(100 * H.H_BLIND_MIN)} percent.",
        f"* The file records `{meta['model']}`, the hard stream, seed {meta['seed']}, {meta['steps']:,} steps, policy {meta['policy']}, up to {two} re-asks, the repair deleting up to {deg} dependants, no rule about dependants stated, no scripted reply. The saved-reply file records the model's default thinking (no setting) and default temperature, "
        f"at most {c[0]['_meta']['max_output_tokens']:,} output tokens, and every one of its {fct['calls']:,} calls ended with a normal stop",
        f"* **The 20-step check** (the first {len(chk)} calls in the saved-reply file) averaged {pt:.1f} prompt tokens, {th:.1f} thinking tokens and {sec:.2f} seconds a call, {pt + th + vis:.0f} tokens a step in all. That is under the page's trigger of {H.HARD_THINKING_LIMIT} thinking tokens, so **thinking was left at the model's default**, as the page required. "
        f"(Prediction: more than {H.HARD_THINKING_LIMIT}, 65 percent. It was not.)",
        f"* **Run 1** took the calls from 13 Oct 21:30 to 14 Oct 00:27 (IST, from the timestamps in the file), {hours:.1f} hours of calls in about {wall:.1f} hours. **Two retries after a busy server** (a wait of about two minutes each, after calls {before[0]:,} and {before[1]:,} of the run); the run continued from where it was and the replay of the saved replies reproduces every flag (line 1).",
        f"{t['calls']:,} calls for {n:,} steps (no step was asked twice).", f"(${R.PRICE_IN:.2f} in, ${R.PRICE_OUT:.2f} out, thinking counted as out;",
        f"Line 11: the mean prompt of the last tenth is {ten[-1]['prompt']:.1f} tokens against {ten[0]['prompt']:.1f} in the first ({r1:.2f} times); line 12: {ten[-1]['seconds']:.2f} seconds against {ten[0]['seconds']:.2f} ({r2:.2f} times).",
        f"The distance to the oracle is {max(x['distance'] for x in ten):.2f} in every tenth, so the comparison of the first and the last (line 9) is {ten[-1]['distance']:.2f} against {ten[0]['distance']:.2f}.",
        f"**Every one of the {n:,} steps ended on the oracle's exact state.** There was no new offset (a step that starts on the oracle's state and ends off it), the largest distance was {maxd} and no state was ever malformed.",
        f"All {off} are in one kind of request", f"with one to three of them ({kd['delete_dep']['requests']} requests) and with four to eight ({kd['delete_hub']['requests']}), **it named every dependant every time: {kd['delete_dep']['requests'] + kd['delete_hub']['requests']} of {kd['delete_dep']['requests'] + kd['delete_hub']['requests']}**",
        f"Applied as written, {cf['blind'][0]} replies ({100 * cf['blind'][0] / n:.2f} percent of the steps) would each have left edges pointing at a deleted service; the gate would have refused them and kept a service the oracle has deleted",
        f"With the framework's own default of 4 dependants the result is the same on these replies ({a['counterfactual_default']['repair_m6']} steps off the oracle's state for repair_m6 and for scope_m7), because every service whose dependants were left out had at most {max(d for d, v in pair.items() if v[1])}, "
        f"and the hub deletes ({H.HUB_MIN} to {H.HUB_MAX} dependants) were named in full by the model: **the setting of {deg}, fixed on the page before the run, made no difference to the {n:,} replies of this run.**",
        f"(at most {100 * H.H_WRONG_MAX:.1f} percent of the asks not done as asked): the framework's held state did what was asked on {n - M.wrong_rate(rows)[0]:,} of {n:,} asks ({M.wrong_rate(rows)[0]} not done); the raw replies applied as written from a correct state did not on {cf['blind'][0]} of {n:,} ({100 * cf['blind'][0] / n:.2f} percent), which is over that bar.",
        f"The repair changed the outcome of **{off} steps ({100 * off / n:.2f} percent)**, described next.",
        f"* **Lines {lines_txt} came out yes** (rated {rated} percent). Line 6 was the one I expected to fail first; lines 6, 7 and 8, the three about how often the model slips, all held, and the share of steps equal to the oracle ({100 * exact_n / n:.0f} percent) was above the {m15.group(1)} to {m15.group(2)} percent I gave for it ({m15.group(3)} percent)",
        f"One run is one sample, and one outcome at about {overall} percent does not say that the {overall} percent was wrong.", f"all nine headline lines yes, rated about {overall} percent: it happened.",
        f"**It shows**, for this model on this job, in one sample: that the host layer's held state stayed equal to the oracle's for {n:,} requests ({t['total'] / 1e6:.1f} million tokens sent) with a prompt that did not grow ({r1:.2f} times from the first tenth to the last, following the graph and not the step) and a speed that did not fall ({r2:.2f} times); "
        f"that the one kind of slip the model made, leaving a service's dependants out of a combined request, was completed by a deterministic repair {off} times out of {off}; and that over the first {H.HARD_CONTRAST_STEPS} steps of the same stream the same replies applied as written left the state exactly right on {bl['exact']} steps of {H.HARD_CONTRAST_STEPS} "
        f"against {s7['exact']} of {H.HARD_CONTRAST_STEPS} with the repair, and corrupted on {bl['corrupted']}, while a gate that only refuses was never corrupted and exactly right on {g['exact']} of {H.HARD_CONTRAST_STEPS}; and that, with every call counted at the owner's prices, a step with the right state cost ${_cc()['scope_m7']['dollars_per_exact']:.5f} with the repair, "
        f"{_cc()['blind']['times_framework']:.1f} times as much (${_cc()['blind']['dollars_per_exact']:.5f}) applying the replies as written and {_cc()['gate']['times_framework']:.1f} times as much (${_cc()['gate']['dollars_per_exact']:.5f}) with the gate, on about the same spend.",
        "or another task, and nothing about money beyond the tokens counted here (energy; what a wrong state costs afterwards).",
        f"**It does not show** that the model is bad: it was right on {n - cf['blind'][0]:,} of {n:,} replies from a correct state, and the damage in the contrast comes from one slip staying until the stream removes it.",
        "The control (Cell 9) asks what the re-ask bought; since no reply was refused in run 1 the control will find every one of its prompts already in the saved-reply file", "Run 2 (Cell 10) is the replication (line 14).",
        "Line 13 reads yes; **line 4 reads NO** (it is not a headline line); line 14 needs run 2 and is \"not run\".", "(the saved-reply file as it was when run 1 ended: the 20-step check and the 4,000 calls", "**Against the page's own bar for line 10** (at most", "it says that line 10 passes here because of the repair and not because the model never slipped.", f"(0 of {kd['add_edge']['requests']:,} add-edge requests)", f"Projected cost between $2 and $8 (65 percent): yes, the 20-step check projected ${4000 * (pt * R.PRICE_IN + (th + vis) * R.PRICE_OUT) / 1e6:.2f} and the arithmetic on the run is ${a['dollars']['total']:.2f}.",
        f"Tokens in run 1 between 2.8 and 4.0 million (60 percent): yes, {t['total'] / 1e6:.2f}.", f"thinking above 150 a step at the default (65 percent): no, {th:.1f}.", f"Mean seconds of a call between 1.0 and 5.0 (65 percent): yes, {a['seconds']['mean']:.2f}.",
        "The share of first replies refused between 0.1 and 2.0 percent (60 percent): no, none (0.0).", "At least 80 percent of the re-asked steps ended readable (75 percent): no step was re-asked", "More refused steps in the control than in the main run's first 1,000 (65 percent)",
        f"Share exact between {m15.group(1)} and {m15.group(2)} percent ({m15.group(3)} percent): no, {100 * exact_n / n:.0f}.",
        f"Applied as written, the replies left the state exactly right on {bl['exact']} of {bl['steps']} steps ({100 * bl['exact'] / bl['steps']:.1f} percent) and corrupted on {bl['corrupted']} ({100 * bl['corrupted'] / bl['steps']:.1f} percent); the framework (scope_m7, run 1's first {s7['steps']} steps) was exactly right on {s7['exact']} of {s7['steps']}. "
        f"**Line 13 (steps 201 to {H.HARD_CONTRAST_STEPS}, mean distance to the oracle): scope_m7 {s7['mean_201_400']:.2f}, blind {bl['mean_201_400']:.2f}, gate {g['mean_201_400']:.2f}: yes.**",
        f"The arms are the same for the first {bl['new_offsets'][0] - 1} steps, and at step {bl['new_offsets'][0]} all three are given the same saved reply", f"(blind {len(bl['new_offsets'])}, at steps {', '.join(str(x) for x in bl['new_offsets'])}; gate {len(g['new_offsets'])}, at steps {', '.join(str(x) for x in g['new_offsets'])}) comes after a request of that one kind.",
        f"Blind was off the oracle's state in the stretches {', '.join(f'{x} to {y}' for x, y in bl['off'])} ({', '.join(str(y - x + 1) for x, y in bl['off'][:-1])} and {bl['off'][-1][1] - bl['off'][-1][0] + 1} steps; the last was still going at step {H.HARD_CONTRAST_STEPS}) and exactly right in the rest, {bl['exact']} steps.",
        f"these lasted between {min(y - x + 1 for x, y in bl['off'])} and {max(y - x + 1 for x, y in bl['off'])}.",
        f"It never admitted a corrupted state ({g['corrupted']} steps) and was exactly right on {g['exact']} of {g['steps']}. Its third offset began at step {g['off'][-1][0]} and was still open at step {H.HARD_CONTRAST_STEPS}: it was off the oracle's state on every step from {g['off'][-1][0]} to {H.HARD_CONTRAST_STEPS}, "
        f"with the mean distance {g['last100_distance']:.2f} in the last hundred steps (largest {g['max_distance']}), and {g['inapplicable']} requests could no longer be applied as written to its state (blind: {bl['inapplicable']}).",
        f"Four stretches ended, at steps {', '.join(str(e[0]) for e in R.blind_stretch_ends(_con()))}: twice a later request added a service back at the coordinate the leftover edges pointed at (the state is then well formed again, and still wrong), twice a later delete of a service named the leftover edges; the last stretch was still going at step {H.HARD_CONTRAST_STEPS}.",
        f"The raw state **was** broken, first at step {bl['corrupted_stretches'][0][0]} and again after later requests; it was corrupted in {len(bl['corrupted_stretches'])} separate stretches (steps {', '.join(f'{x} to {y}' for x, y in bl['corrupted_stretches'])})",
        f"* **The contrast** ({', '.join(list(_con())[:-1])} and {list(_con())[-1]}, the registered settings, seed {_con()['blind']['meta']['seed']}, {_con()['blind']['meta']['steps']} steps each; each arm keeps its own held state and shows the model that state). "
        f"Where an arm's prompt equals one already asked, the saved reply is reused and nothing is sent: {bl['reused']} steps of blind and {g['reused']} of gate (the first {bl['new_offsets'][0] - 1} steps, which are the same for every arm, and the stretches in which an arm's state is the oracle's again), "
        f"and **all {cons['repair_m6']['reused']} of repair_m6**, whose held state equals run 1's at every step, so that arm is a replay of run 1's replies and not a separate sample.", f"Live: {bl['live_calls']} calls for blind and {g['live_calls']} for gate; no reply was refused in either.",
        f"${bl['dollars']:.2f} for the blind arm and ${g['dollars']:.2f} for the gate arm of the contrast (${bl['dollars'] + g['dollars']:.2f} together; repair_m6 sent nothing)",
        f"* **The repair alone is the framework on this stream.** repair_m6 was exactly right on {cons['repair_m6']['exact']} of {cons['repair_m6']['steps']}, as scope_m7 was: the scope filter made no difference here (it removed nothing in run 1 either). "
        f"That arm sent nothing (all {cons['repair_m6']['reused']} of its steps reuse run 1's replies), so it is a recomputation of run 1's replies and not a second sample.",
        "(the contrast: blind, gate and repair_m6, 400 steps each, received after run 1; the run 1 part of this section was first written before it arrived and the contrast was then added)", "**Line 4 NO, in the page's own terms.**",
        f"in which the model left out the service's dependants (the first of the {off} of run 1). Blind applied it and held edges pointing at a deleted service from that step; the gate refused it and kept a service the oracle had deleted; the repair added the dependants. Every new offset of the two arms",
        f"* **What limits this reading.** {w[len(bl['new_offsets'])]} new offsets for blind and {w[len(g['new_offsets'])]} for the gate, on one stream, one seed and {H.HARD_CONTRAST_STEPS} steps; the arms share replies (the first {bl['new_offsets'][0] - 1} steps, and every stretch in which an arm's state is the oracle's again), "
        f"so they are not three independent samples of the model; every offset starts at a request of that one kind; and the comparison is with a host that applies the replies as written or only checks them, **not** with a chat that keeps its history and **not** with an agent that reads its own state back and corrects it (neither is measured). Over {n:,} steps the blind arm was not run.",
        "A check that refuses a bad reply keeps the state well formed; it does not make it right.", "The comparison that is still missing, and that matters most for a claim about savings, is against a chat that keeps its history."]
    assert cons["repair_m6"]["exact"] == s7["exact"] == H.HARD_CONTRAST_STEPS and cons["repair_m6"]["tokens"] == 0 and cons["repair_m6"]["reused"] == H.HARD_CONTRAST_STEPS and H.report_hard(run)["dropped_items"] == 0         # what 'the scope filter made no difference' and 'sent nothing' rest on
    assert g["corrupted"] == 0 and g["exact"] < g["steps"] and bl["new_offset_kinds"] == g["new_offset_kinds"] == ["delete_pair"]
    for fr in fragments: assert fr in text, fr
    # the labels: which policy and which run each sentence names
    assert set(re.findall(r"\b(?:scope|repair)_m\d\b", text)) == {"scope_m7", "repair_m6"}
    from collections import Counter
    assert Counter(re.findall(r"\b[Rr]un [0-9]\b", text)) == Counter(RUN_LABELS), Counter(re.findall(r"\b[Rr]un [0-9]\b", text))
    # the thresholds and percentages of paragraph 15 quoted in the Result are those written before the run
    b = text[text.index("* **Line 15:**"):text.index("## What this run does and does not show")]
    assert re.findall(r"\((\d+) percent\)", b) == re.findall(r"\((\d+) %\)", p15) and len(re.findall(r"\((\d+) %\)", p15)) == 10
    for ph in ("between 2.8 and 4.0 million", "between $2 and $8", "between 1.0 and 5.0", "between 0.1 and 2.0 percent", "at least 80 percent", "between 60 and 99 percent", "more than 150"): assert ph in p15.lower().replace("**", "") or ph in p15, ph
    for ph in ("between 2.8 and 4.0 million", "between $2 and $8", "between 1.0 and 5.0", "between 0.1 and 2.0 percent", "at least 80 percent"): assert ph in b.lower() or ph in b, ph


def test_what_a_step_with_the_right_state_cost_follows_from_the_saved_rows_and_the_page_claims_nothing_more():
    """Recomputed from the raw rows, without the module: tokens and dollars of every call of each arm of the contrast (a reused reply is a call of the saved-reply file with its own counts), the steps that held the exact state, the dollars and tokens
    per such step and the ratios, the sentences of the section built from them, and nothing on the page about energy, markets or any company."""
    run, con, text = _run(), _con(), _result_text(); cc = _cc(); n = H.HARD_CONTRAST_STEPS; cx = _cx(); b, g = cx["blind"], cx["gate"]
    arms = {"scope_m7": run["rows"][:n], "blind": con["blind"]["rows"], "gate": con["gate"]["rows"], "repair_m6": con["repair_m6"]["rows"]}
    saved = {(x["text"], x["prompt_tokens"], x["tokens"], x["think"]) for x in _ok_calls()}; assert (R.PRICE_IN, R.PRICE_OUT) == (0.75, 3.75); got = {}
    for arm, rows in arms.items():
        assert len(rows) == n and all("asks" not in r and "reasks" not in r for r in rows)                                       # one call a step in every arm
        for r in rows:                                                                                                        # a reused reply (and every step of run 1) is a call in the saved-reply file, with its own token counts; a live step of the contrast was billed in the contrast
            if r["cached"] or arm == "scope_m7": assert (r["text"], r["prompt_tokens"], r["tokens"], r["think"]) in saved, (arm, r["t"])
        p, v, t = (sum(r[k] for r in rows) for k in ("prompt_tokens", "tokens", "think")); ex = sum(1 for r in rows if r["exact"]); usd = (0.75 * p + 3.75 * (v + t)) / 1e6
        got[arm] = (p + v + t, usd, ex, (p + v + t) / ex, usd / ex)
    assert sum(1 for r in arms["blind"] if r["cached"]) == 86 and sum(1 for r in arms["gate"] if r["cached"]) == 85 and sum(1 for r in arms["repair_m6"] if r["cached"]) == 400 and not any(r["cached"] for r in arms["scope_m7"])
    assert [got[a][0] for a in arms] == [332838, 333040, 347263, 332838] and [got[a][2] for a in arms] == [400, 85, 84, 400] and M.tokens_sent(run["rows"][:n]) == 332838
    assert [round(got[a][1], 2) for a in arms] == [0.47, 0.45, 0.48, 0.47] and [round(got[a][4], 5) for a in arms] == [0.00117, 0.00524, 0.00573, 0.00117] and [round(got[a][3]) for a in arms] == [832, 3918, 4134, 832]
    ratio = {a: got[a][4] / got["scope_m7"][4] for a in arms}; assert [round(ratio[a], 1) for a in arms] == [1.0, 4.5, 4.9, 1.0] and got["repair_m6"] == got["scope_m7"]
    assert round(400 / 85, 2) == 4.71 and round(got["blind"][1] / got["scope_m7"][1], 2) == 0.95 and abs(ratio["blind"] - (400 / 85) * (got["blind"][1] / got["scope_m7"][1])) < 1e-9              # the 4.5 is the ratio of right steps times the ratio of spend
    assert max(abs(got[a][1] / got["scope_m7"][1] - 1) for a in ("blind", "gate")) < 0.055 and got["blind"][0] - got["scope_m7"][0] == 202 and got["gate"][0] - got["scope_m7"][0] == 14425
    for a in arms:                                                                                                                # the module gives the same
        x = cc[a]; assert x["tokens"] == got[a][0] and abs(x["dollars"] - got[a][1]) < 1e-12 and x["exact"] == got[a][2] and abs(x["dollars_per_exact"] - got[a][4]) < 1e-12 and abs(x["times_framework"] - ratio[a]) < 1e-9 and x["calls"] == n
    sec = text[text.index("## What a step with the right state cost"):text.index("## The predictions, one by one")]; fw, bc, gc = cc["scope_m7"], cc["blind"], cc["gate"]
    assert R.table_cost(cc) in sec
    for fr in ("This section was added after the contrast was read, at the owner's request. It was not on the page before the run, so it is not a pre-registered reading and it cannot change any of the fourteen lines; it is arithmetic on the saved files.",
               f"(the test checks that each of the {b['reused']}, {g['reused']} and {cx['repair_m6']['reused']} reused replies is, text and token counts, a call in the saved-reply file). So each arm is priced as if all {fw['calls']} of its calls had been billed, at the prices the owner typed in "
               f"(${R.PRICE_IN:.2f} in, ${R.PRICE_OUT:.2f} out, thinking counted as out); in the contrast itself only the live calls were billed (${b['dollars']:.2f} for blind and ${g['dollars']:.2f} for the gate, as in the table above).",
               f"* **The same spend.** The framework's {fw['calls']} calls were {fw['tokens']:,} tokens and ${fw['dollars']:.2f}; blind's, {bc['tokens']:,} tokens and ${bc['dollars']:.2f}; the gate's, {gc['tokens']:,} tokens and ${gc['dollars']:.2f}. "
               f"The three arms spent within {max(abs(x['dollars'] / fw['dollars'] - 1) for x in (bc, gc)) * 100:.0f} percent of each other in dollars, and in tokens the framework used {bc['tokens'] - fw['tokens']:,} fewer than blind and {gc['tokens'] - fw['tokens']:,} fewer than the gate. "
               "**The spend is about the same; what differs is how many steps ended with the right state.**",
               f"* **What differed is how many steps ended with the right state:** {fw['exact']} of {fw['steps']} for the framework, {bc['exact']} for blind and {gc['exact']} for the gate. A step with the right state cost ${fw['dollars_per_exact']:.5f} ({fw['tokens_per_exact']:,.0f} tokens) with the framework, "
               f"${bc['dollars_per_exact']:.5f} ({bc['tokens_per_exact']:,.0f} tokens) with the replies applied as written, **{bc['times_framework']:.1f} times as much**, and ${gc['dollars_per_exact']:.5f} ({gc['tokens_per_exact']:,.0f} tokens) with the gate, **{gc['times_framework']:.1f} times as much**.",
               f"* **How the {bc['times_framework']:.1f} comes about.** It is the ratio of steps with the right state ({fw['exact']} to {bc['exact']} is {fw['exact'] / bc['exact']:.2f}) times the ratio of spend ({bc['dollars'] / fw['dollars']:.2f}). It moves with how long a wrong state lasts in the stream: "
               f"here blind was off the oracle's state for {min(y - x + 1 for x, y in b['off'])} to {max(y - x + 1 for x, y in b['off'])} steps at a time; on a stream whose slips are removed sooner the ratio would be smaller and on one where they stay longer it would be larger (not measured).",
               f"* **What this number leaves out.** Only the calls of the first {n} requests, counted in the service's tokens at the owner's prices. Not energy (the service does not report it; tokens are the stand-in, and nothing here converts them), not what a wrong state costs afterwards (the next actions that build on it), "
               "not the cost of an agent that reads its own state back and corrects it, not a chat that keeps its history (neither is measured), and not another model, task or price. The blind and gate arms were not run over 4,000 steps, so there is no figure for the whole run."): assert fr in sec, fr
    head = _text()[:_text().index(RESULT_MARK)]; assert "per step with the right state" not in head and "times the framework" not in head and not any("$" in l[1] or "dollar" in l[1] for l in H.HARD_LINES)           # not a reading rule, not on the page before the run
    low = text.lower()
    for word in ("billion", "trillion", "hyperscaler", "industry", "profit", "market", "compute budget", "inference cost", "capex", "openai", "google ", "microsoft", "amazon", "nvidia"): assert word not in low, word        # nothing about companies or markets
    assert not re.search(r"\d\s*(kwh|wh|watt|joule|mwh)\b", low) and "energy" in low and low.count("energy") == 2                                                                                   # no energy figure; energy only named as not measured
    d = copy.deepcopy(con); d["blind"]["rows"][50]["prompt_tokens"] += 1000; c2 = R.analyse_cost(run, d); assert c2["blind"]["tokens"] == bc["tokens"] + 1000 and R.table_cost(c2) not in text          # a doctored file does not give the printed table
    d = copy.deepcopy(con); [r.update(exact=False) for r in d["gate"]["rows"]]
    try: R.analyse_cost(run, d); raise SystemExit("an arm with no right step must be refused")
    except AssertionError: pass


def test_the_header_and_the_files_named_on_the_page_are_the_ones_that_exist():
    text = _result_text(); head = [l for l in text.split("\n") if l.startswith("# Result")]
    assert head == ["# Result (run 1 and the contrast of 13 and 14 Oct 2026, seed 20261031, on the owner's Colab with his paid key; the control and run 2 have not been made yet and each gets its own section below when its files arrive)"]
    for f in (R.RUN1_FILE, R.CACHE1_FILE, R.CONTRAST_FILE): assert f"`docs/results/{f}`" in text and os.path.exists(os.path.join(R.RESULTS, f))
    assert R.RUN1_FILE == "axi_million_hard_run1_seed20261031_default.json" and _run()["meta"]["seed"] == 20261031 and len(_run()["rows"]) == 4000
    for needle in ("(the 4,000 steps, every reply)", "the 20-step check and the 4,000 calls, with the two retries after a busy server", "## The fourteen reading rules, run 1 and the contrast", "`tests/test_million_hard_result.py` recomputes every number of this section from the three files",
                   "(`million.py` and `million_hard.py`, which the run used, were not touched)", "a hash in `tests/test_million_hard.py` checks it"): assert needle in text, needle
    assert [l for l in text.split("\n") if l.startswith("## ")] == ["## The fourteen reading rules, run 1 and the contrast", "## What was run", "## The numbers", "## What the reading rules do not show", "## The contrast: the same first 400 requests, four ways of keeping the state", "## What a step with the right state cost", "## The predictions, one by one", "## What this run does and does not show"]


def test_the_page_above_the_result_is_still_the_registered_text_and_nothing_the_run_used_imports_the_result_module():
    t = _text(); assert t.count(RESULT_MARK) == 1 and _h(t[:t.index(RESULT_MARK)]) == PAGE_SHA256
    root = os.path.join(HERE, "..", "axi"); users = []
    for d, _, fs in os.walk(root):
        for f in fs:
            if f.endswith(".py") and f != "million_hard_result.py" and "million_hard_result" in open(os.path.join(d, f)).read(): users.append(f)
    assert users == []
    nb = open(os.path.join(HERE, "..", "colab", "axi_million_hard.ipynb")).read(); assert "million_hard_result" not in nb
    src = open(R.__file__).read(); assert "def analyse" in src and "def cache_facts" in src and not re.search(r"\bapi_llm\b|\bgenerate\(|\bgenai\b", src)           # it calls no model


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn): fn(); print("PASS", name); n += 1
    print(n, "tests passed")
