"""Implements concept-map item 100: RESULT of the concurrency test of 15 Oct 2026. Page: docs/SWARM_TEST.md (predictions and reading rules written before the run; the result is appended below them).
The files: docs/results/axi_swarm_seed41.json (the run: seed 41, Python 3.13), axi_swarm_seed41_python311.json (the same run under Python 3.11) and axi_swarm_exhaustive.json (the pass added afterwards).
This file does not re-run the experiment (that takes minutes; `python -m axi.experiments.swarm` does, and prints the digest this file pins). It recomputes from the saved files:
  1. the digest of pairs, rounds and sweep equals the one on the page, in both files, and the two files agree on pairs, rounds and sweep (only the thread counts differ);
  2. the nineteen reading rules, all yes, for the run and for the Python 3.11 run; the headline lines;
  3. every figure of the table of the page (counts and rates, per regime and together), by recomputing it from the raw tallies;
  4. the figures of "what I read from it": the share kept by the merge by agents per service, the order independence (200 of 200 against 36 of 200), the clash kinds (all 'both orders refused');
  5. the exhaustive pass: the counts, the rates the page quotes; and that the pass is what `swarm_exhaustive` computes (checked at two operations in test_swarm_exhaustive.py);
  6. the part of the page above the Result line is byte for byte what was written before the run, and the page quotes the figures above.
What it does NOT show: anything beyond these edits and these sizes, 10,000 agents, processes, the Rust crate."""
import hashlib, json, os, re, sys

sys.path.insert(0, os.path.dirname(__file__))
import axi.experiments.swarm as W

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..", "..")
PAGE = os.path.join(ROOT, "docs", "SWARM_TEST.md")
RES = os.path.join(ROOT, "docs", "results")
RESULT_MARK = "\n\n---\n\n# Result"
PAGE_SHA256 = "e18d1a644f9b3394e8845df20f40c28a2cf86a4c6eeb7a7d94771a87656b2a04"
DIGEST = "dd08fd515ff59bff"
_cache = {}


def _load(name):
    if name not in _cache: _cache[name] = json.load(open(os.path.join(RES, name)))
    return _cache[name]


def run(): return _load("axi_swarm_seed41.json")
def run311(): return _load("axi_swarm_seed41_python311.json")
def exh(): return _load("axi_swarm_exhaustive.json")


def _text(): return open(PAGE).read()


def _result():
    t = _text(); assert t.count(RESULT_MARK) == 1; return t[t.index(RESULT_MARK):]


def _digest(res): return hashlib.sha256(json.dumps([res["pairs"], res["rounds"], res["sweep"]], sort_keys=True, default=str).encode()).hexdigest()[:16]


def pct(x, k=1): return f"{x * 100:.{k}f} %"


# ---------------------------------------------------------------- 1 the files
def test_the_files_are_the_run_the_page_describes_and_the_digest_is_the_same_under_two_pythons():
    for res in (run(), run311()):
        assert res["seed"] == 41 and res["quick"] is False and _digest(res) == res["digest"] == DIGEST and set(res) == {"digest", "pairs", "quick", "rounds", "seconds", "seed", "sweep", "threads"}
    assert run()["pairs"] == run311()["pairs"] and run()["rounds"] == run311()["rounds"] and run()["sweep"] == run311()["sweep"] and run()["threads"] != run311()["threads"]
    for reg in ("uniform", "hot"):
        p = run()["pairs"][reg]; assert p["pairs"] == 120000 and p["conflict"] + p["commute"] == 120000 and sum(p["detectors"]["dpo"].values()) == 120000
        for det in ("dpo", "pager", "names"): assert sum(p["detectors"][det].values()) == 120000 and p["detectors"][det]["tp"] + p["detectors"][det]["fn"] == p["conflict"] and p["detectors"][det]["fp"] + p["detectors"][det]["tn"] == p["commute"]
        r = run()["rounds"][reg]; assert (r["agents"], r["rounds"], r["perms"]) == (16, 100, 10)
    assert set(run()["sweep"]) == {f"{s}/{n}" for s in (40, 160, 640) for n in (4, 16, 64)} and all(v["rounds"] == 400 for v in run()["sweep"].values())
    assert all(v["rounds"] == 300 and v["threads"] == 8 for v in run()["threads"].values()) and set(run()["threads"]) == {"rule", "gate", "lww"}


# ---------------------------------------------------------------- 2 the nineteen lines
def test_the_nineteen_reading_rules_all_read_yes_for_both_runs_and_the_headline_lines_too():
    for res in (run(), run311()):
        checks = W.verdict_swarm(W.summarize_swarm(res)); assert len(checks) == 19 and [v for _, v in checks] == [True] * 19 and [k for k, _ in checks] == [k for k, _ in W.SWARM_LINES]
        assert all(checks[i - 1][1] for i in W.SWARM_HEADLINE)
    s = W.summarize_swarm(run()); assert s["conflict"] == 4842 and s["commute"] == 235158 and s["pairs"] == 240000 and s["dpo_missed"] == {"hot": 0, "uniform": 0} and s["dpo_false"] == {"hot": 0.0, "uniform": 0.0}


# ---------------------------------------------------------------- 3 the table of the page
def test_every_figure_of_the_table_is_recomputed_from_the_raw_tallies_and_is_on_the_page():
    P = run()["pairs"]; res = _result(); u, h = P["uniform"], P["hot"]; D = lambda r, d, c: r["detectors"][d][c]
    assert (u["conflict"], h["conflict"]) == (2369, 2473) and (u["commute"], h["commute"]) == (117631, 117527) and "4,842 of the 240,000" in res and "2,369 uniform, 2,473 hot" in res and "235,158 commute" in res
    assert u["why"] == {"both orders refused": 2369} and h["why"] == {"both orders refused": 2473} and u["both_valid"] == h["both_valid"] == u["first_commit_ok"] == h["first_commit_ok"] == 120000
    assert D(u, "dpo", "fn") == D(h, "dpo", "fn") == D(u, "dpo", "fp") == D(h, "dpo", "fp") == 0
    fn_p = D(u, "pager", "fn") + D(h, "pager", "fn"); assert (D(u, "pager", "fn"), D(h, "pager", "fn"), fn_p) == (651, 669, 1320) and pct(fn_p / 4842) == "27.3 %" and pct(651 / 2369) == "27.5 %" and pct(669 / 2473) == "27.1 %"
    assert (D(u, "pager", "fp"), D(h, "pager", "fp")) == (1594, 2901) and pct(2901 / 117527, 2) == "2.47 %" and pct(1594 / 117631, 2) == "1.36 %" and u["misses"]["pager"]["fp +e/+e"] == 1028
    fn_n = D(u, "names", "fn") + D(h, "names", "fn"); assert (D(u, "names", "fn"), D(h, "names", "fn"), fn_n) == (1123, 814, 1937) and pct(fn_n / 4842) == "40.0 %" and pct(1123 / 2369) == "47.4 %" and pct(814 / 2473) == "32.9 %"
    assert pct(2369 / 120000, 2) == "1.97 %" and pct(2473 / 120000, 2) == "2.06 %"
    for needle in ("27.3 %", "27.5 % uniform, 27.1 % hot", "2.47 %", "1.36 %", "1,028", "40.0 %", "47.4 % uniform, 32.9 % hot", "1.97 % and 2.06 %", "1,320 of 4,842", "1,937 of 4,842", "651 + 669", "1,123 + 814", "2,901 of 117,527"): assert needle in res, needle
    R = run()["rounds"]; sch = lambda reg, k: R[reg]["schedulers"][k]
    assert all(sch(reg, "merge")["same_graph"] == 100 and sch(reg, "merge")["gate_refusals"] == 0 and sch(reg, "merge")["ill_formed"] == 0 and sch(reg, "gated")["ill_formed"] == 0 for reg in R)
    assert (sch("hot", "gated")["same_graph"], sch("uniform", "gated")["same_graph"]) == (17, 19) and (100 - 17, 100 - 19) == (83, 81) and (sch("hot", "blind")["ill_formed"], sch("uniform", "blind")["ill_formed"]) == (52, 68)
    assert [round(sch(reg, k)["mean_taken"], 3) for reg in ("uniform", "hot") for k in ("gated", "merge")] == [0.874, 0.875, 0.874, 0.873] and all(sch(reg, k)["gate_refusals"] == 0 for reg in R for k in ("blind", "gated", "merge"))
    for needle in ("83 of 100 hot rounds (81 uniform)", "52 of 100 hot rounds (68 uniform)", "**100 of 100** rounds", "`gated` took 0.874 and `merge` 0.875 of the edits (uniform), 0.874 and 0.873 (hot)"): assert needle in res, needle
    T = run()["threads"]; assert (T["rule"]["accepted"], T["rule"]["clash"], T["rule"]["attempts"], T["rule"]["gate"]) == (2083, 317, 2400, 0) and (T["gate"]["accepted"], T["gate"]["gate"], T["gate"]["clash"]) == (2035, 365, 0) and pct(317 / 2400) == "13.2 %"
    assert T["lww"]["accepted"] == 2400 and T["lww"]["replay_ok"] is False and all(T[k]["replay_ok"] and T[k]["well_formed"] and T[k]["accounted"] and T[k]["finished"] and not T[k]["errors"] and T[k]["gate_after_rule"] == 0 for k in ("rule", "gate"))
    for needle in ("2,083 accepted + 317 refused as clashes", "2,035 accepted, 365 refused by the gate", "317 of 2,400 attempts: 13.2 %"): assert needle in res, needle


# ---------------------------------------------------------------- 4 what was read from it
def test_the_curve_of_the_share_kept_against_agents_per_service_is_what_the_page_says():
    S = run()["sweep"]; pts = sorted(((v["agents"] / v["services"], v["mean_taken"]) for v in S.values()))
    assert [round(r, 5) for r, _ in pts] == [0.00625, 0.025, 0.025, 0.1, 0.1, 0.1, 0.4, 0.4, 1.6] and [round(x, 3) for _, x in pts] == [0.998, 0.990, 0.991, 0.962, 0.962, 0.976, 0.872, 0.881, 0.699]       # sorted by ratio, then share
    ys = [x for _, x in pts]; assert ys == sorted(ys, reverse=True) or all(abs(a - b) < 0.02 for a, b in zip(ys, sorted(ys, reverse=True)))          # one curve: falling with the ratio (ties within 0.01)
    assert max(abs(x - 0.967) for r, x in pts if abs(r - 0.1) < 1e-9) < 0.01                                                          # the three points at 0.1 agents per service agree within a point
    res = _result()
    for needle in ("99.8 %", "99.0 % and 99.1 %", "97.6 %, 96.2 % and 96.2 %", "88.1 % and 87.2 %", "69.9 %", "about 96 % on a graph of about 100,000 services and about 70 % on one of about 6,000", "10,000 agents"): assert needle in res, needle
    assert round(10000 / 0.1) == 100000 and round(10000 / 1.6, -3) == 6000
    R = run()["rounds"]; assert sum(R[reg]["schedulers"]["merge"]["same_graph"] for reg in R) == 200 and sum(R[reg]["schedulers"]["gated"]["same_graph"] for reg in R) == 36 and "200 of 200" in res and "36 of 200" in res
    assert all(R[reg]["schedulers"]["merge"]["same_graph"] == 100 for reg in R)
    assert abs(run()["sweep"]["40/64"]["mean_taken"] - 0.698515625) < 1e-12 and abs(run()["sweep"]["640/64"]["mean_taken"] - 0.961640625) < 1e-12


# ---------------------------------------------------------------- 5 the exhaustive pass
def test_the_exhaustive_pass_found_the_rule_exact_on_every_pair_and_the_page_quotes_it():
    e = exh(); D = e["detectors"]
    assert (e["max_ops"], e["graphs"], e["pairs"], e["conflict"], e["commute"]) == (3, 80, 1262688, 1075824, 186864) and e["why"] == {"both orders refused": 1075824} and e["cells"] == [[0, 0], [0, 1], [1, 0]]
    assert D["dpo"] == dict(tp=1075824, fn=0, fp=0, tn=186864) and D["pager"] == dict(tp=1024427, fn=51397, fp=137912, tn=48952) and D["names"] == dict(tp=1013848, fn=61976, fp=0, tn=186864)
    assert pct(51397 / 1075824) == "4.8 %" and pct(137912 / 186864) == "73.8 %" and pct(61976 / 1075824) == "5.8 %" and e["examples"]["dpo"] == [] and e["misses"]["dpo"] == {}
    res = _result()
    for needle in ("1,262,688 ordered pairs, of which 1,075,824", "| `dpo` (the rule) | 1,075,824 | **0** | **0** |", "| `pager` | 1,024,427 | 51,397 (4.8 %) | 137,912 (73.8 % of the 186,864 pairs that commute) |", "| `names` | 1,013,848 | 61,976 (5.8 %) | 0 |", "not a proof for the language", "added after the result, not pre-registered"): assert needle in res, needle
    for k in D: assert sum(D[k].values()) == e["pairs"] and D[k]["tp"] + D[k]["fn"] == e["conflict"]


# ---------------------------------------------------------------- 6 the page
def test_the_page_above_the_result_is_what_was_written_before_the_run_and_the_result_says_what_it_must():
    t = _text(); head = t[:t.index(RESULT_MARK)]; assert hashlib.sha256((head.rstrip("\n") + "\n").encode()).hexdigest() == PAGE_SHA256
    res = _result()
    for needle in ("dd08fd515ff59bff", "Python 3.13", "Python 3.11", "82 seconds", "Headline (lines 5, 6, 10, 16): all four read yes", "All 19 predictions came out as I wrote them, and I do not take that as a sign that the test is strong",
                   "Where my hand numbers were off", "What I read from it (derived afterwards; none of this was pre-registered)", "does not keep more edits than first come, first served", "wrong in both directions", "With one head, the gate alone already keeps the state correct",
                   "Not tested at 10,000", "inevitably", "atomic transaction manager", "statelessness", "Not done and worth doing", "python -m axi.experiments.swarm_exhaustive", "test 109", "test 110"):
        assert needle.lower() in res.lower(), needle
    probs = [92, 99, 97, 99, 90, 88, 90, 70, 93, 97, 80, 92, 99, 90, 75, 95, 93, 90, 95]; rows = re.findall(r"\n\| (\d+) \| (\d+) % \|", res); assert [(int(a), int(b)) for a, b in rows] == list(zip(range(1, 20), probs))
    assert len(re.findall(r"\| yes( \(by 3 rounds\))? \|", res)) == 19


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("PASS", n)
