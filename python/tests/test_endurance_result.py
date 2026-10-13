"""Implements concept-map item 92 (endurance and cost): the RESULT of docs/ENDURANCE_TEST.md, recomputed from the saved file and from the saved runs. It checks that
  1. the saved file is the registered run (sizes, seeds, the 15 kinds, the four policies, no run ended early) and that every number the page states about Part C follows from it: the ten reading-rule lines recomputed
     by the module say "yes" (and the page says the same, line by line), the headline counts (1.6 million steps, no crash, no corrupted state for the gate, repair_m6 and scope_m7, blind apply corrupted on 99.6 percent of
     its steps), the answers to prediction 11, and the shapes the text describes (scope_m7 flat, repair_m6 flattening, the gate growing);
  2. the tables on the page are, character for character, what the code renders from the saved data (Part C three tables, Part A, Part B), so a number cannot be edited on the page without the check failing;
  3. Part A's headline figures follow from the saved Qwen and Gemini runs (105 steps, 304,289 and 39,225 tokens, 87.1 percent, 207.4 tokens a turn) and Part B's replay reproduces all 22 policy rows without one mismatch;
  4. the page's text above the Result is still the registered text (the hash lives in tests/test_endurance.py).
What it does NOT show: that a model behaves like the stand-in, or anything the page does not measure."""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
from axi.experiments import endurance as E
from axi.experiments.confirm_test import wasted_tokens

HERE = os.path.dirname(__file__)
PAGE = os.path.join(HERE, "..", "..", "docs", "ENDURANCE_TEST.md")
SAVED = os.path.join(E.RESULTS, E.RESULT_FILE)
RESULT_MARK = "\n\n---\n\n# Result"


def _load(): return json.load(open(SAVED))


def _result_text(): t = open(PAGE).read(); assert t.count(RESULT_MARK) == 1; return t[t.index(RESULT_MARK):]


def test_the_saved_file_is_the_registered_run():
    r = _load()
    assert r["control"]["meta"] == dict(episodes=20, steps=2000, seed=20261011, p_fault=0.0, faults=list(E.FAULTS), mix="steady", policies=list(E.POLICIES_C))
    assert r["million"]["meta"] == dict(episodes=500, steps=2000, seed=20261012, p_fault=0.5, faults=list(E.FAULTS), mix="steady", policies=list(E.GATED))
    assert r["blind"]["meta"] == dict(episodes=50, steps=2000, seed=20261012, p_fault=0.5, faults=list(E.FAULTS), mix="steady", policies=["blind"])
    assert [x["meta"] for x in r["long"]] == [dict(episodes=1, steps=250000, seed=s, p_fault=0.5, faults=list(E.FAULTS), mix="steady", policies=list(E.GATED)) for s in (20261013, 20261014)]
    assert [r["control"]["steps_run"], r["million"]["steps_run"], r["blind"]["steps_run"], r["long"][0]["steps_run"], r["long"][1]["steps_run"]] == [40000, 1000000, 100000, 250000, 250000]    # no stream ended early
    for part in [r["control"], r["million"], r["blind"]] + r["long"]:
        for p, s in part["policies"].items(): assert s["steps"] == part["steps_run"], p
    assert set(r["million"]["fault_counts"]) == set(E.FAULTS) and max(r["million"]["fault_counts"].values()) - min(r["million"]["fault_counts"].values()) <= 40       # the deck dealt every kind equally often
    assert all(k in r for k in ("_python", "_platform", "_cpus", "_code_sha256")) and len(r["_code_sha256"]) == 64
    assert r["control"]["damaged_replies"] == 0 and r["control"]["dealt"] == 0


def test_the_ten_lines_and_the_headline_numbers_follow_from_the_file():
    r = _load(); v = E.verdict_endurance(r); assert len(v) == 10 and all(ok for _, ok in v)
    text = _result_text()
    for i, ((tag, line), (l2, ok)) in enumerate(zip(E.LINES_C, v), 1): assert l2 == line and f"\n{i}. yes [{tag}] {line}\n" in text, i
    parts = [r["control"], r["million"], r["blind"]] + r["long"]
    assert sum(x["parse_crashes"] + sum(c["count"] for c in x["crashes"].values()) for x in parts) == 0 and all(x["crashes"] == {} for x in parts)
    for part in [r["million"]] + r["long"]:
        for p in E.GATED: assert part["policies"][p]["corrupted"] == 0
    b = r["blind"]["policies"]["blind"]; assert b["corrupted"] == 99629 and b["steps"] == 100000 and round(100 * b["corrupted"] / b["steps"], 1) == 99.6 and b["max_distance"] == 9032
    t = E.tokens_equivalent(r, E.token_bill()["per_step_state_gemini"]); assert t["steps"] == 1600000 and round(t["tokens"] / 1e6) == 517
    for needle in ("99.6 percent", "9,032 items", "1,600,000 steps", "about 517 million tokens", "0 mismatches in 4,800 steps"): assert needle in text, needle
    m = r["million"]; assert round(100 * m["unparsed"] / m["steps_run"], 1) == 19.0 and 14 <= 100 * m["unparsed"] / m["steps_run"] <= 24 and round(100 * m["damaged_replies"] / m["steps_run"]) == 42
    s = m["policies"]["scope_m7"]; mean = sum(s["dist_sum"]) / sum(s["dist_n"]); assert 5 <= mean <= 20 and round(mean, 1) == 11.1
    assert [x["policies"]["repair_m6"]["max_edges"] for x in r["long"]] == [305, 284] and all(x["policies"]["repair_m6"]["max_edges"] > 200 for x in r["long"])
    assert "refused 19.0 percent of the replies" in text and "mean distance over the million steps was 11.1" in text and "was 305 and 284 edges" in text and "changed replies on 42 percent" in text
    assert round(sum(s["ns"]) / s["steps"] / 1000, 1) == 32.0 and "was 32.0 microseconds a step" in text and r["million"]["peak_rss_mb"]["after_10_percent"] == r["million"]["peak_rss_mb"]["end"]


def test_the_shapes_the_text_describes_are_in_the_data():
    r = _load()
    for x in r["long"]:
        sc, rp, gt = (E.block_distance(x["policies"][p]) for p in ("scope_m7", "repair_m6", "gate")); us = E.block_us(x["policies"]["gate"]); su = E.block_us(x["policies"]["scope_m7"])
        assert max(sc) / min(sc) < 1.1 and 11 <= sc[0] <= 12 and 11 <= sc[-1] <= 12                                       # scope_m7 flat at about 11
        assert rp[0] < rp[2] and max(rp[2:]) / min(rp[2:]) < 1.1 and 240 <= rp[-1] <= 260                                   # repair_m6 rises and flattens after the third tenth
        assert all(gt[i] < gt[i + 1] for i in range(9)) and gt[0] < 900 and gt[-1] > 2500 and us[-1] > 3 * us[0]               # the gate keeps growing, and so does its cost
        assert max(su) / min(su) < 1.3 and 35 <= min(su) and max(su) <= 45
        assert x["policies"]["scope_m7"]["max_nodes"] <= 24 and x["policies"]["scope_m7"]["max_edges"] <= 15 and x["policies"]["gate"]["max_edges"] > 2000
    text = _result_text()
    for needle in ("11.4 and 11.3 in the first tenth", "11.4 and 11.4 in the last", "never held more than 24 services or 23 edges", "877 in the first tenth, 2,510 in the last", "222 to 725 microseconds", "rose from about 198", "to about 250 in the last"):
        assert needle in text, needle


def test_the_tables_on_the_page_are_what_the_code_renders_from_the_saved_data():
    r = _load(); text = _result_text(); bill = E.token_bill(); rep = E.cascade_report()
    for block in (E.table_c_overview(r), E.table_c_replies(r), E.table_c_tenths(r), E.table_a(bill), E.table_b(rep)): assert block in text
    assert text.count("\n| ") == 70
    # a doctored number in a table is not accepted: change one cell and the block is no longer on the page
    assert E.table_c_overview(r).replace("| 99,629 |", "| 99,628 |") not in text and "| 99,629 |" in text


def test_part_a_follows_from_the_saved_runs():
    b = E.token_bill(); h = b["history"]
    assert [p["steps"] for p in b["pairs"]] == [30, 26, 19, 30] and b["measured_steps"] == 105 and b["measured_history_tokens"] == 304289 and b["measured_state_tokens"] == 39225
    assert round(100 * b["measured_saving"], 1) == 87.1 and round(h["slope"], 1) == 207.4 and round(h["intercept"], 1) == 246.8 and h["steps"] == 19
    assert round(b["per_step_state_gemini"], 1) == 323.2 and round(b["per_step_state_qwen"], 1) == 368.3 and round(b["state_steps_for"]["1,000,000 tokens"]) == 3094 and round(b["state_steps_for"]["2,000,000 tokens"]) == 6187
    assert b["windows"] == {"5,500 (the T4 GPU ran out of memory near here)": 26, "32,768": 157, "128,000": 616, "1,000,000": 4820, "2,000,000": 9641}
    assert [p["last_prompt"] for p in b["pairs"]] == [5230, 5530, 5135, 4503]
    text = _result_text()
    for needle in ("7.8 times the tokens", "304,289 against 39,225", "a saving of 87 percent", "207.4 tokens a step", "step 616", "about 3,094 steps and 2 million about 6,187", "nothing beyond 30 steps was run"): assert needle in text, needle
    # the saved history run is the one the 5.1x and out-of-memory numbers of LONG_HORIZON_TEST.md come from
    d = json.load(open(os.path.join(E.RESULTS, E.HISTORY_FILE + ".json"))); assert [r["stopped"]["repair_m6"] for r in d["rows"]].count("oom") == 2


def test_part_b_replays_all_22_rows_and_the_statements_about_them_follow():
    rep = E.cascade_report(); rows = [(d["label"], p, c) for d in rep.values() for p, c in d["policies"].items()]
    assert len(rows) == 22 and sum(c["mismatch"] for _, _, c in rows) == 0 and sum(c["steps"] for _, _, c in rows) == 4800
    for name, d in rep.items():
        for p, c in d["policies"].items():
            w, t = wasted_tokens(json.load(open(os.path.join(E.RESULTS, name + ".json")))["rows"], p); assert (c["wasted"], c["tokens"]) == (w, t)
    pw = lambda name, p: 100.0 * rep[name]["policies"][p]["wasted"] / rep[name]["policies"][p]["tokens"]
    assert round(pw("axi_long_state_norule_Qwen2.5-7B_seed3", "blind")) == 89 and round(pw("axi_14b_long_norule_Qwen2.5-14B_seed7", "blind")) == 88
    assert round(pw("axi_long_state_norule_Qwen2.5-7B_seed3", "repair_m6"), 1) == 0.3 and pw("axi_14b_long_norule_Qwen2.5-14B_seed7", "repair_m6") == 0.0
    scope = [(d["label"], 100.0 * d["policies"]["scope_m7"]["wasted"] / d["policies"]["scope_m7"]["tokens"]) for d in rep.values() if "scope_m7" in d["policies"]]
    assert len(scope) == 6 and sum(1 for _, x in scope if x <= 2.0) == 5 and [round(x, 1) for _, x in scope if x <= 2.0] == [0.8, 1.8, 0.0, 0.0, 0.0]
    fl = rep["axi_gemini_long_rule_gemini-3.5-flash-lite_seed7"]["policies"]; assert [round(pw("axi_gemini_long_rule_gemini-3.5-flash-lite_seed7", p)) for p in fl] == [88, 88, 88] and [c["unparsed"] for c in fl.values()] == [43, 42, 42]
    assert round(pw("axi_long_state_rule_Qwen2.5-7B_seed3", "repair_m6")) == 42 and round(pw("axi_scope_long_rule_Qwen2.5-7B_seed7", "repair_m6")) == 28 and round(pw("axi_14b_long_rule_Qwen2.5-14B_seed7", "repair_m6")) == 17
    ok = [(d["label"], p, c) for d in rep.values() for p, c in d["policies"].items() if c["applicable_same"] >= 20 and c["applicable_off"] >= 20]
    higher = [x for x in ok if x[2]["wrong_off"] / x[2]["applicable_off"] > x[2]["wrong_same"] / x[2]["applicable_same"]]
    assert len(ok) == 11 and len(higher) == 1 and higher[0][:2] == ("Qwen 7B, rule stated (scope mix)", "repair_m6") and (higher[0][2]["wrong_off"], higher[0][2]["applicable_off"], higher[0][2]["wrong_same"], higher[0][2]["applicable_same"]) == (6, 65, 17, 264)
    inap = [100.0 * (c["inapplicable_same"] + c["inapplicable_off"]) / c["steps"] for d in rep.values() for p, c in d["policies"].items() if p in ("blind", "gate")]; assert round(min(inap)) == 2 and round(max(inap)) == 23
    text = _result_text()
    for needle in ("In 11 policy rows", "failed more often in 1 (Qwen 7B, rule stated (scope mix), repair_m6: 9.2 percent against 6.4)", "less often or as often in the other 10", "88 percent for Flash Lite with the rule stated", "42 of 180 replies were refused", "(2 to 23 percent of the steps",
                   "repair_m6 alone spent 42 percent (Qwen 7B, base mix), 28 percent (Qwen 7B, scope mix) and 17 percent (Qwen 14B)", "five of the six settings", "(42 of 180, 23 percent, for Flash Lite with the rule stated)"): assert needle in text, needle


def test_the_table_renderers_are_the_only_code_after_the_marker_and_the_page_says_so():
    """The recorded fingerprint is of the module as it was when the run started, before the table renderers were appended; the page says so. Here the renderers are separated from the rest of the module by a marker, the
    set of renderers is fixed, and nothing the run uses is defined after the marker."""
    src = open(E.__file__).read(); marker = "# ================================================================ the Result tables"
    assert src.count(marker) == 1
    before, after = src[:src.index(marker)], src[src.index(marker):]
    assert {n for n in dir(E) if n.startswith("table_")} == {"table_c_overview", "table_c_replies", "table_c_tenths", "table_a", "table_b"} and "table_" not in before
    defined_after = set(re.findall(r"^def (\w+)", after, flags=re.M)); assert defined_after == {"_pc", "_frac", "_crashes", "table_c_overview", "table_c_replies", "table_c_tenths", "table_a", "table_b"}
    for name in ("run_all", "run_endurance", "verdict_endurance", "damage", "Stream", "Garbler", "block_us", "block_distance", "tokens_equivalent"): assert name in before, name
    assert len(_load()["_code_sha256"]) == 64 and "After the run the module gained only the table renderers" in _result_text()


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn): fn(); print("PASS", name); n += 1
    print(n, "tests passed")
