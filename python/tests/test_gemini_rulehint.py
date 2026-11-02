import hashlib
import json
import os

DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs")


def _make(arm, n=60, sabotage_every=0):
    """Rows of one arm made by running the stream with the ORACLE's replies (no model), the replies of delete requests cut down to the service alone on every `sabotage_every`-th step."""
    from axi.experiments import million as M, million_hard as H, gemini_rulehint as R
    from axi.experiments.agent_loop import copy_graph, strict_parse
    pol, hint = R.ARMS[arm]
    s = M.make_stream("HardStream", R.RH_SEED); live = copy_graph(s.start); rows = []
    for t in range(1, n + 1):
        st, ob = s.next(); oa = s.live; text = st["ideal"]
        if sabotage_every and st["kind"] in ("delete_hub", "delete_dep") and t % sabotage_every == 0: text = text.split(" ")[0]
        after, fl = M.step_fields(pol, live, st, ob, oa, strict_parse(text), H.HARD_DEGREE)
        rows.append(dict(t=t, kind=st["kind"], text=text, tokens=10, think=5, cached=False, prompt_tokens=1000, seconds=1.0, **fl)); live = after
    meta = dict(seed=R.RH_SEED, steps=n, policy=pol, hint_rules=hint, stream="HardStream", reasks=2, degree_threshold=8)
    return dict(rows=rows, meta=meta)


def test_registered_and_cost_arithmetic():
    from axi.experiments import gemini_rulehint as R
    ok = dict(seed=R.RH_SEED, steps=400, policy="gate", hint_rules=True, stream="HardStream", reasks=2, degree_threshold=8)
    assert R.registered(ok, "hint_gate") and not R.registered(ok, "gate") and not R.registered(dict(ok, seed=20261031), "hint_gate")
    assert not R.registered(dict(ok, scripted=5), "hint_gate") and not R.registered(dict(ok, reasks=0), "hint_gate")
    # one call of 1,000 prompt tokens, 10 reply and 5 thinking tokens at $0.75 and $3.75 per million; a re-ask adds a second call
    r = dict(prompt_tokens=1000, tokens=10, think=5, cached=False)
    assert abs(R.row_cost(r, 0.75, 3.75) - (1000 * 0.75 + 15 * 3.75) / 1e6) < 1e-12
    r2 = dict(r, reasks=[dict(prompt_tokens=1100, tokens=8, think=0, cached=True)])
    assert abs(R.row_cost(r2, 0.75, 3.75) - ((1000 * 0.75 + 15 * 3.75) + (1100 * 0.75 + 8 * 3.75)) / 1e6) < 1e-12


def test_slips_counted_from_the_replies_and_verdict_with_made_up_runs():
    from axi.experiments import gemini_rulehint as R
    perfect = _make("repair", 60); cut = _make("repair", 60, 2)
    assert R.raw_slips(perfect["rows"], perfect["meta"])["slips"] == 0
    c = R.raw_slips(cut["rows"], cut["meta"])
    assert c["slips"] > 0 and c["steps_counted"] == 60 and set(c["by_kind"]) <= {"add_node", "add_edge", "delete_dep", "delete_hub", "delete_leaf", "delete_edge", "delete_pair"}
    # the repair finishes what the cut replies left: the held state stays exact; blind and the gate do not
    assert all(r["exact"] for r in cut["rows"]) and not all(r["exact"] for r in _make("blind", 60, 2)["rows"])
    s = R.arm_stats(cut, 0.75, 3.75)
    assert s["exact"] == 60 and abs(s["cost"] - 60 * (1000 * 0.75 + 15 * 3.75) / 1e6) < 1e-9 and s["cost_per_exact"] == s["cost"] / 60
    # with only some arms (and too few steps) no line can read yes where it needs an arm that is missing
    v = R.verdict_rulehint({"repair": cut})
    assert v[0][2] == "NO" and v[2][2] == "not run" and v[12][2] == "not run" and len(v) == 13
    assert R.headline(v) == "DOES NOT REPEAT"
    assert R.headline([("g", "x", "yes")] * 13) == "REPEATS" and R.headline([("g", "x", "not run")] * 13) == "NOT DECIDED"


def test_page_lock():
    t = open(os.path.join(DOCS, "GEMINI_RULEHINT_TEST.md")).read()
    head = t.split("\n## Result\n")[0]
    want = open(os.path.join(DOCS, "results", "gemini_rulehint_hash.txt")).read().strip()
    assert hashlib.sha256((head.rstrip("\n") + "\n").encode()).hexdigest() == want
    from axi.experiments import gemini_rulehint as R
    for i, (_, text) in enumerate(R.RH_LINES, 1):
        assert ("%d. %s" % (i, text)) in head, i


if __name__ == "__main__":
    for _n, _f in sorted(globals().items()):
        if _n.startswith("test_") and callable(_f): _f(); print("PASS", _n)


def test_rulehint_result_figures():
    from axi.experiments import gemini_rulehint as R
    d = json.load(open(os.path.join(DOCS, "results", "gemini_rulehint.json")))
    assert d["seed"] == R.RH_SEED and d["price_in"] == 0.75 and d["price_out"] == 3.75 and set(d["arms"]) == set(R.ARM_ORDER)
    S = {a: R.arm_stats(d["arms"][a], d["price_in"], d["price_out"]) for a in R.ARM_ORDER}
    assert [S[a]["exact"] for a in R.ARM_ORDER] == [400, 400, 400, 41, 41]
    assert [S[a]["corrupted"] for a in R.ARM_ORDER] == [0, 0, 0, 246, 0]
    assert [S[a]["raw"]["slips"] for a in R.ARM_ORDER] == [7, 0, 0, 2, 2] and S["repair"]["raw"]["by_kind"]["delete_pair"] == (7, 9)
    assert round(S["blind"]["last100_distance"], 2) == 3.95 and round(S["gate"]["last100_distance"], 2) == 7.94
    assert [round(S[a]["cost_per_exact"], 4) for a in R.ARM_ORDER] == [0.0012, 0.0011, 0.0011, 0.0115, 0.013]
    assert round(sum(S[a]["cost"] for a in R.ARM_ORDER), 2) == 2.38 and round(S["blind"]["cost_per_exact"] / S["repair"]["cost_per_exact"], 1) == 9.3
    v = R.verdict_rulehint(d["arms"], d["price_in"], d["price_out"])
    assert [x[2] for x in v] == ["yes"] * 5 + ["NO"] + ["yes"] * 7 and R.headline(v) == "REPEATS"
    t = open(os.path.join(DOCS, "GEMINI_RULEHINT_TEST.md")).read()
    for needle in ("12 read yes and 1 reads NO", "$2.38 in all", "9.3 times", "7 of 9 such requests", "THE FIRST RESULT REPEATS ON A SECOND SEED"):
        assert needle in t
