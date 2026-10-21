"""Engineering support for docs/PROVIDER_SIDE_RETRY_TEST.md (concept-map item 109): the exact-state cells of the provider-side serving test run with ONE retry when the gate refused a reply and the repair
of repair_m6 cannot fix it. Checks that need no GPU and no model:
  1. nothing the test imports is edited (sha256 of seven files), the settings are the third run's, and the new session adds methods and overrides only what it must;
  2. the rule, replayed on the 3,360 saved replies of the third run: it fires exactly where `next_state` of repair_m6 leaves a parsed, refused reply unchanged (5, 12 and 12 steps, jobs 3, 4, 19, 20 only),
     never on the 507 deletes the repair completes; the table of six first misses, the lost steps per job and the ceilings the page quotes;
  3. the second request: the first request, the first reply, the framework's feedback (the four texts the page gives);
  4. a job that never asks again is the pinned job, and the lockstep without a retry is the pinned lockstep;
  5. the harness with a test double that writes the third run's slips: the second reply decides the step, one retry at most, the seconds of the second round are in the cell's seconds;
  6. the five cells, the merge (what it refuses), the probe;
  7. the 12 lines on a good run built from the third run's own replies, each line at its edge, 'not run' for a missing cell, the rule about what may be claimed, the printout;
  8. the locked page, the figures it quotes, the notebook and its safety cell, the front pages.
What it does NOT show: anything about a real engine, a real GPU, or what a model answers when it is told why a reply was refused."""
import contextlib, copy, glob, hashlib, inspect, io, json, os, re, subprocess, sys, tempfile, time, types

sys.path.insert(0, os.path.dirname(__file__))
import axi.engine.gate as G
import axi.engine.diagnostics as D
import axi.experiments.agent_loop as A
import axi.experiments.cascade as K
import axi.experiments.provider_side as PS
import axi.experiments.provider_side_constrained as C
import axi.experiments.provider_side_bounded as B
import axi.experiments.provider_side_retry as R
import test_provider_side_constrained as T124                                       # its fake `vllm` is reused, not copied
from axi.engine.gate import check
from axi.engine.diagnostics import diagnose, ground_and_complete
from axi.experiments.agent_loop import feedback_text, same_graph, strict_parse
from axi.experiments.cascade import next_state

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..", "..")
RES = os.path.join(ROOT, "docs", "results")
PINS = dict(provider_side="bfb4734e171d9244cd47d10402be46528a557095a8275d02b38481a0e2b0b1d3", constrained="03452148e28c9f31372f12cc11ef1808ac67308adff972fa19d4aeeef7c96f7d",
            bounded="7427a686d151c4294809aebaa9664d0b5fa420854817eb79e910372c3f57b6bc", gate="b6a50408788791f76bc4d4129dad84debb6e09c9c5e804673cdf04abec224ce7",
            diagnostics="255f3c11db989b645077d10706a187e2c913576538a826e6dbdb8892b014b4f2", agent_loop="d658afaa29e3bce7e1fb7f43d88281fc40ee68ca59864421a65284e9973516b6",
            cascade="b0533316bc094e80a1d68eddb7e137c7c176af9a342341288360c18d61f74fe7")
REGEX_SHA256 = "6f4e5a993f26ccd471bff13422fa622b7820858a91e05d43c2bccfbe6b864245"
THIRD_FILE = os.path.join(RES, "axi_provider_bounded.json")
THIRD_SHA256 = "1f12a846b1325f9fed4e5df2cc755b3a7fd996266cb590311230f7cf418f2733"
PAGE = os.path.join(ROOT, "docs", "PROVIDER_SIDE_RETRY_TEST.md")
PAGE_SHA256 = "74243e1ab623ec5c20168de0978bc33224b06875630fe83e56987806a529ce33"
RESULT_MARK = "\n\n# Result"
PROBS = [98, 92, 85, 93, 90, 85, 75, 35, 55, 88, 40, 70]
SEED = PS.SEED
# job: (step, what the engine answered, what the oracle wrote, the gate's reason, the user turn of the second request)
SLIPS = {3: (16, "ADD[(4,4)->(4,4):value#6]", "ADD[4,4:6]", "well-formedness: edge ((4, 4), (4, 4), 'value') endpoint missing", "OBSTRUCTION malformed edge=((4, 4), (4, 4), 'value') Reply with only the corrected delta."),
         4: (18, "ADD[1,0:6]", "ADD[0,0:3]", "identification: node (1, 0) already exists", "OBSTRUCTION ident_node node=(1,0) Reply with only the corrected delta."),
         19: (12, "ADD[(3,0)->(3,0):4tworowsdown4col#4]", "ADD[3,0:4]", "well-formedness: edge ((3, 0), (3, 0), '4tworowsdown4col') endpoint missing",
              "OBSTRUCTION malformed edge=((3, 0), (3, 0), '4tworowsdown4col') Reply with only the corrected delta."),
         20: (30, "ADD[(1,5)->(2,1):dep#3]", "ADD[(1,1)->(2,1):dep#3]", "identification: edge ((1, 5), (2, 1), 'dep') already exists", "OBSTRUCTION ident_edge edge=((1, 5), (2, 1), 'dep') Reply with only the corrected delta.")}
ADMITTED_MISSES = {17: (20, "DEL[(2,3)->(0,4):dep#6] ADD[(0,4)->(2,3):dep#4]", "ADD[(0,4)->(2,3):dep#4]"), 62: (12, "DEL[(1,1)->(1,0):owns#5] ADD[(1,1)->(1,0):dep#9]", "ADD[(1,1)->(1,0):dep#9]")}
LOST = {3: 15, 4: 13, 17: 4, 19: 19, 20: 1, 62: 1}


def raises(fn, exc=Exception):
    try: fn()
    except exc: return True
    return False


def _sha(path): return hashlib.sha256(open(path, "rb").read()).hexdigest()


_CACHE = {}


def third():
    if "third" not in _CACHE: _CACHE["third"] = json.load(open(THIRD_FILE))
    return _CACHE["third"]


def tcells(): return {(c["arm"], c["K"]): c for c in third()["cells"]}
def tlist(): return third()["cells"]
def seed_of(job): return SEED * 1000 + job


def replay_to(job, step):
    """The job as the oracle's own replies leave it before `step` (1-based)."""
    s = R.RetrySession(seed_of(job), PS.STEPS)
    while s.t < step - 1:
        rq = s.request(); s.accept(PS.Reply(rq.stand_in_reply, 0, 0, 0))
    return s


# ---------------------------------------------------------------- 1 nothing edited, the settings
def test_nothing_the_test_imports_is_edited_the_settings_are_the_third_runs_and_the_new_session_adds_methods_and_overrides_only_two():
    mods = dict(provider_side=PS, constrained=C, bounded=B, gate=G, diagnostics=D, agent_loop=A, cascade=K)
    for name, m in mods.items(): assert _sha(m.__file__) == PINS[name], name
    assert R.pinned_hashes() == PINS
    assert (R.MODEL, R.STEPS, R.SEED, R.POLICY, R.HINT_RULES, R.MAX_NEW, R.CAPACITY_TOKENS) == (PS.MODEL, PS.STEPS, PS.SEED, PS.POLICY, PS.HINT_RULES, PS.MAX_NEW, PS.CAPACITY_TOKENS) == ("Qwen/Qwen2.5-7B-Instruct-AWQ", 30, 20261017, "repair_m6", False, 96, 80000)
    assert R.REGEX is B.REGEX and R.REGEX_SHA256 == B.REGEX_SHA256 == REGEX_SHA256 and R.NAME_CAP == 16 and R.BoundedVLLMEngine is B.BoundedVLLMEngine and R.BoundedStandIn is B.BoundedStandIn
    assert R.FEEDBACK == "structured_imperative" and R.DEGREE_THRESHOLD == 4 and R.MAX_RETRIES == 1
    assert inspect.signature(next_state).parameters["degree_threshold"].default == 4 == inspect.signature(ground_and_complete).parameters["degree_threshold"].default
    assert R.CELLS == {"R8": (8, 0, True), "R32": (32, 0, True), "R64": (64, 0, True), "H0": (64, 64, False), "H1": (64, 64, True)} and R.ORDER == ["R8", "R32", "R64", "H0", "H1"] == R.cells_planned()
    own = {k for k in vars(R.RetrySession) if not k.startswith("__")}
    assert own == {"request", "read_reply", "retry_request", "accept", "accept_retried"}, own
    assert R.RetrySession.done is PS.Session.done and issubclass(R.RetrySession, PS.Session) and issubclass(R.RRequest, PS.Request)
    assert (R.R_PARSED, R.R_MAX_SHARE, R.R_HIGH, R.R_VERY_HIGH, R.R_HELD_GAIN, R.R_TIME_TOL) == (0.995, 0.03, 0.98, 0.99, 0.01, 0.10) and R.RL_HEADLINE == (9, 11) and len(R.RL_LINES) == 12
    assert _sha(THIRD_FILE) == THIRD_SHA256


# ---------------------------------------------------------------- 2 the rule on the third run's replies
def _walk(cell):
    """Every step of a cell played again from the text of its replies, without any retry: (job, record, the rule's reading, what next_state does with the reply)."""
    for job, sess in enumerate(cell["sessions"]):
        s = R.RetrySession(seed_of(job), PS.STEPS)
        for r in sess["log"]:
            s.request(); info = s.read_reply(r["text"]); d = strict_parse(r["text"])
            unchanged = d is not None and same_graph(next_state(PS.POLICY, s.live, d), s.live)
            refused = d is not None and not check(s.live, d).ok
            before_exact = (s.log[-1]["exact"] if s.log else True)
            s.accept(PS.Reply(r["text"], r["n_prompt"], r["n_cached"], r["n_new"]))
            for f in ("parsed", "admitted", "corrupted", "exact", "distance"): assert s.log[-1][f] == r[f], (job, r["t"], f)
            yield job, r, info, refused, unchanged, before_exact


def test_the_rule_fires_exactly_where_repair_m6_leaves_a_refused_reply_unchanged_and_only_on_four_jobs_and_never_on_a_delete_the_repair_completes():
    want = {8: 5, 32: 12, 64: 12}; tc = tcells(); total = 0; dang = dict(n=0, exact=0, repairable=0, behind=0)
    for K, n_steps in ((8, 240), (32, 960), (64, 1920)):
        fires = []; steps = 0
        for job, r, info, refused, unchanged, prev_exact in _walk(tc[("S", K)]):
            steps += 1; total += 1
            assert info["fires"] == (refused and unchanged), (K, job, r["t"])
            assert info["parsed"] == (r["parsed"]) and info["admitted"] == r["admitted"] and (info["repairable"] or info["admitted"] or info["fires"] or not info["parsed"])
            if info["fires"]: fires.append((job, r["t"]))
            if K == 64 and r["kind"] == "delete_dep" and info["parsed"] and not info["admitted"] and info["reason"].startswith("dangling"):
                dang["n"] += 1; dang["exact"] += r["exact"]; dang["repairable"] += info["repairable"]; dang["behind"] += (not r["exact"] and not prev_exact)
        assert steps == n_steps and len(fires) == want[K], (K, fires)
        assert {j for j, _ in fires} <= {3, 4, 19, 20}
        for j in (3, 4, 19, 20):
            if any(f[0] == j for f in fires): assert min(t for jj, t in fires if jj == j) == SLIPS[j][0], (K, j)
    assert total == 3120 and dang == dict(n=507, exact=491, repairable=507, behind=16), dang
    shares = {K: round(100 * want[K] / n, 2) for K, n in ((8, 240), (32, 960), (64, 1920))}; assert shares == {8: 2.08, 32: 1.25, 64: 0.62}, shares
    page = open(PAGE).read(); assert "fires on 5 of 240 steps at 8 jobs (2.1 percent), 12 of 960 at 32 jobs (1.25) and 12 of 1,920 at 64 jobs (0.6)" in page


def test_the_six_first_misses_the_lost_steps_per_job_and_the_ceilings_are_the_ones_the_page_gives():
    tc = tcells(); cell = tc[("S", 64)]; lost = {}
    for job, sess in enumerate(cell["sessions"]):
        n = sum(1 for x in sess["log"] if not x["exact"])
        if n: lost[job] = n
    assert lost == LOST and sum(LOST.values()) == 53 and sum(LOST[j] for j in SLIPS) == 48 and sum(LOST[j] for j in ADMITTED_MISSES) == 5
    for job, (step, text, oracle, reason, fb) in SLIPS.items():
        first_miss = next(x for x in cell["sessions"][job]["log"] if not x["exact"]); assert (first_miss["t"], first_miss["text"]) == (step, text) and first_miss["parsed"] and not first_miss["admitted"]
        s = replay_to(job, step); rq = s.request(); assert rq.stand_in_reply == oracle
        info = s.read_reply(text); assert info == dict(parsed=True, admitted=False, repairable=False, fires=True, reason=reason)
        g = ground_and_complete(s.live, strict_parse(text), 4); assert g is not None and not g.repairable
    for job, (step, text, oracle) in ADMITTED_MISSES.items():
        first_miss = next(x for x in cell["sessions"][job]["log"] if not x["exact"]); assert (first_miss["t"], first_miss["text"]) == (step, text) and first_miss["parsed"] and first_miss["admitted"]
        s = replay_to(job, step); assert s.request().stand_in_reply == oracle and s.read_reply(text)["fires"] is False
    exact = {K: sum(x["exact"] for s in tc[("S", K)]["sessions"] for x in s["log"]) for K in (8, 32, 64)}; assert exact == {8: 212, 32: 908, 64: 1867}
    ceil = {K: exact[K] + sum(1 for s in tc[("S", K)]["sessions"][:K] for x in s["log"] if not x["exact"] and tc[("S", K)]["sessions"].index(s) in SLIPS) for K in (8, 32, 64)}
    assert ceil == {8: 240, 32: 956, 64: 1915}, ceil
    assert [round(100 * ceil[K] / n, 1) for K, n in ((8, 240), (32, 960), (64, 1920))] == [100.0, 99.6, 99.7]
    S = {K: C.summarize_cell2(tc[("S", K)]) for K in (8, 32, 64)}
    assert [round(S[K]["wall_seconds"], 1) for K in S] == [39.4, 119.3, 217.4] and [round(S[K]["seconds_per_step"], 3) for K in S] == [0.164, 0.124, 0.113] and [round(S[K]["exact_per_hour"]) for K in S] == [19379, 27390, 30914]
    # in the 64-job cell the rule fires in 10 different rounds without a retry (the page says "at most 10")
    rounds = {t for job, r, info, *_ in _walk(cell) if info["fires"] for t in [r["t"]]}; assert len(rounds) == 10


# ---------------------------------------------------------------- 3 the second request
def test_the_second_request_is_the_first_request_the_first_reply_and_the_frameworks_own_feedback_and_the_four_texts_are_the_pages():
    page = open(PAGE).read()
    for job, (step, text, oracle, reason, fb) in SLIPS.items():
        s = replay_to(job, step); rq = s.request(); rq2 = s.retry_request(rq, text)
        assert [m["role"] for m in rq2.messages] == ["system", "user", "assistant", "user"] and rq2.messages[:2] == rq.messages and rq2.messages[2] == {"role": "assistant", "content": text}
        assert rq2.messages[3]["content"] == fb == feedback_text("structured_imperative", diagnose(s.live, strict_parse(text))) == feedback_text("structured", diagnose(s.live, strict_parse(text)))
        assert rq.retry is False and rq2.retry is True and (rq.seed, rq.step) == (rq2.seed, rq2.step) == (seed_of(job), step) and rq2.stand_in_reply == rq.stand_in_reply == oracle
        assert f"  * job {job}: `{fb}`" in page, job
        assert rq.messages == s.request().messages                                                    # asking for the second request does not move the job
    assert raises(lambda: replay_to(3, 16).retry_request(replay_to(3, 16).request(), "Sure, here you go"), AssertionError)       # a reply that does not parse is not asked again
    assert replay_to(3, 16).read_reply("Sure, here you go") == dict(parsed=False, admitted=False, repairable=False, fires=False, reason="not parsed")        # what the rule says about a reply that does not parse
    s16 = replay_to(3, 16); assert s16.read_reply(s16.request().stand_in_reply) == dict(parsed=True, admitted=True, repairable=False, fires=False, reason="")             # an admitted reply is not asked again
    rq0 = R.RRequest([{"role": "user", "content": "x"}], "ADD[0,0:1]"); assert (rq0.seed, rq0.step, rq0.retry) == (0, 0, False)                                  # the defaults of a request
    assert feedback_text("structured_imperative", diagnose(replay_to(3, 16).live, strict_parse("DEL[0,0]"))) != ""                  # the dangling variant exists; the rule never sends it for a delete the repair completes


# ---------------------------------------------------------------- 4 the pinned job and the pinned lockstep
def _strip(sessions): return [[{k: v for k, v in x.items() if k != "retried"} for x in s["log"]] for s in sessions]


def test_a_job_that_never_asks_again_is_the_pinned_job_and_the_lockstep_without_a_retry_is_the_pinned_lockstep():
    a = C.run_cell(B.BoundedStandIn(), "S", 5, steps=8)
    for retry in (False, True):
        b = R.run_jobs(B.BoundedStandIn(), list(range(5)), retry, steps=8)
        assert _strip(b["sessions"]) == [s["log"] for s in a["sessions"]] and all(not x["retried"] for s in b["sessions"] for x in s["log"])
        assert [{k: v for k, v in r.items() if k != "kind"} for r in b["rounds"]] == a["rounds"] and all(r["kind"] == "first" for r in b["rounds"]) and b["wall_seconds"] == a["wall_seconds"]
    # with the third run's slips written by the double and the retry off: the pinned session fed the same replies gives the same jobs
    eng = R.FaultyStandIn(faults=R.third_run_slips()); b = R.run_jobs(eng, [3, 4, 19, 20], False, steps=30)
    for sess in b["sessions"]:
        p = PS.Session("S", sess["seed"], 30)
        for x in sess["log"]: p.request(); p.accept(PS.Reply(x["text"], x["n_prompt"], x["n_cached"], x["n_new"]))
        assert p.log == [{k: v for k, v in x.items() if k != "retried"} for x in sess["log"]]
    miss = {s["seed"] - SEED * 1000: [x["t"] for x in s["log"] if not x["exact"]] for s in b["sessions"]}
    assert all(miss[j][0] == SLIPS[j][0] for j in miss) and {j: len(m) for j, m in miss.items()} == {3: 12, 4: 13, 19: 19, 20: 1}       # without a retry the job is behind from the slip on (the double writes only the first slip)


# ---------------------------------------------------------------- 5 the harness
def test_the_second_reply_decides_the_step_one_retry_at_most_and_the_seconds_of_the_second_round_are_in_the_cell():
    eng = R.FaultyStandIn(faults=R.third_run_slips()); c = R.run_jobs(eng, [3, 4, 19, 20, 5], True, steps=30)
    logs = {s["seed"] - SEED * 1000: s["log"] for s in c["sessions"]}; asked = {j: [x for x in logs[j] if x["retried"]] for j in logs}
    assert {j: [x["t"] for x in v] for j, v in asked.items()} == {3: [16], 4: [18], 19: [12], 20: [30], 5: []}
    for j in (3, 4, 19, 20):
        x = asked[j][0]; step, text, oracle, reason, fb = SLIPS[j]
        assert x["first_text"] == text and x["text"] == oracle and x["exact"] and x["admitted"] and x["parsed"] and x["first_parsed"] and not x["first_admitted"] and not x["first_repairable"] and x["first_reason"] == reason and x["feedback"] == fb
        assert all(y["exact"] for y in logs[j]) and len(logs[j]) == 30
    assert [r["kind"] for r in c["rounds"]].count("retry") == 4 and [r["kind"] for r in c["rounds"]].count("first") == 30 and len(c["rounds"]) == 34
    assert [r["round"] for r in c["rounds"]] == list(range(1, 35)) and abs(c["wall_seconds"] - sum(r["seconds"] for r in c["rounds"])) < 1e-9
    first_rounds = [r for r in c["rounds"] if r["kind"] == "first"]; assert [r["active"] for r in first_rounds] == [5] * 30
    retry_secs = sum(r["seconds"] for r in c["rounds"] if r["kind"] == "retry"); plain = R.run_jobs(R.FaultyStandIn(faults=R.third_run_slips()), [3, 4, 19, 20, 5], False, steps=30)
    assert retry_secs > 0 and abs(c["wall_seconds"] - (plain["wall_seconds"] + retry_secs)) < 1e-6 or c["wall_seconds"] > plain["wall_seconds"]
    # the second reply is refused again: asked once, the state stays, no third request
    again = {(SEED * 1000 + 3, 16): "ADD[(4,4)->(4,4):other#1]"}; eng2 = R.FaultyStandIn(faults=R.third_run_slips(), retry_faults=again); c2 = R.run_jobs(eng2, [3], True, steps=30)
    log2 = c2["sessions"][0]["log"]; x = log2[15]; assert x["retried"] and x["text"] == again[(SEED * 1000 + 3, 16)] and not x["admitted"] and not x["exact"] and x["first_text"] == SLIPS[3][1]
    assert [y["t"] for y in log2 if y["retried"]][0] == 16 and [r["kind"] for r in c2["rounds"]].count("retry") == sum(1 for y in log2 if y["retried"]) and log2[16]["t"] == 17     # one retry per step at most; the next request is built on the state as it was
    assert sum(1 for y in log2[:16] if not y["exact"]) == 1 and not log2[15]["exact"]                                # the job is behind after a second refusal
    # the second reply is admitted but wrong: the step is decided by it, the job is not exact
    wrong = {(SEED * 1000 + 3, 16): "ADD[4,3:6]"}; eng3 = R.FaultyStandIn(faults=R.third_run_slips(), retry_faults=wrong); c3 = R.run_jobs(eng3, [3], True, steps=30)
    y = c3["sessions"][0]["log"][15]; assert y["retried"] and y["admitted"] and not y["exact"] and y["text"] == "ADD[4,3:6]" and y["first_text"] == SLIPS[3][1]
    # a reply the repair completes is not asked again: a delete of a service with edges
    for sess in tcells()[("S", 8)]["sessions"][:1]: assert any(x["kind"] == "delete_dep" and not x["admitted"] for x in sess["log"])
    # the double is a double: it records its own name
    assert R.FaultyStandIn().info()["constraint"]["kind"] == "stand_in" and "NOT a result" not in json.dumps(R.FaultyStandIn().info())


# ---------------------------------------------------------------- 6 cells, merge, probe
def test_the_planned_cells_the_merge_and_what_it_refuses_and_the_probe():
    eng = R.FaultyStandIn(faults=R.third_run_slips()); small = {lab: R.run_cell(eng, lab, steps=2) for lab in R.ORDER[:2]}
    cells = [R.run_cell(eng, lab, steps=2) for lab in reversed(R.ORDER)]
    m = R.merge_cells(cells); assert [c["label"] for c in m["cells"]] == R.ORDER and m["meta"]["name_cap"] == 16 and m["meta"]["cells"] == {k: list(R.CELLS[k]) for k in R.ORDER} and m["meta"]["retry_rule"] == dict(feedback="structured_imperative", degree_threshold=4, max_retries=1)
    assert [c["K"] for c in m["cells"]] == [8, 32, 64, 64, 64] and [c["first_episode"] for c in m["cells"]] == [0, 0, 0, 64, 64] and [c["retry"] for c in m["cells"]] == [True, True, True, False, True]
    seeds = {c["label"]: [s["seed"] for s in c["sessions"]] for c in cells}; assert seeds["H0"] == seeds["H1"] == [SEED * 1000 + e for e in range(64, 128)] and seeds["R64"] == [SEED * 1000 + e for e in range(64)]
    plain = PS.run_cell(PS.StandInEngine(), "S", 2, steps=2); assert raises(lambda: R.merge_cells([plain]), (ValueError, KeyError))
    old = B.run_cell(B.BoundedStandIn(), "S", 8, steps=2); assert raises(lambda: R.merge_cells([old]), (ValueError, KeyError))                          # the third run's cell has no label
    assert raises(lambda: R.merge_cells([cells[0], cells[0]]), ValueError) and raises(lambda: R.merge_cells([]))
    o = copy.deepcopy(cells[1]); o["label"] = "R9"; assert raises(lambda: R.merge_cells([o]), ValueError)
    o = copy.deepcopy(cells[1]); o["retry"] = not o["retry"]; assert raises(lambda: R.merge_cells([o]), ValueError)                                    # not the settings of its label
    o = copy.deepcopy(cells[1]); o["first_episode"] = 1; assert raises(lambda: R.merge_cells([o]), ValueError)
    o = copy.deepcopy(cells[1]); o["retry_rule"]["feedback"] = "structured"; assert raises(lambda: R.merge_cells([o]), ValueError)
    o = copy.deepcopy(cells[1]); o["engine"]["constraint"]["regex_sha256"] = "0" * 64; assert raises(lambda: R.merge_cells([o]), ValueError)
    o = copy.deepcopy(cells[1]); o["retry_rule"] = None; assert raises(lambda: R.merge_cells([o]), ValueError)
    for field, val in (("seed", 1), ("steps", 7)):
        o = copy.deepcopy(cells[1]); o[field] = val; assert raises(lambda: R.merge_cells([cells[0], o]), ValueError), field
    for field in ("version", "gpu"):
        o = copy.deepcopy(cells[1]); o["engine"][field] = "other"; assert raises(lambda: R.merge_cells([cells[0], o]), ValueError), field
    with tempfile.TemporaryDirectory() as d:
        paths = [R.save_cell(c, os.path.join(d, c["label"] + ".json")) for c in cells]; out = os.path.join(d, "m.json"); mm = R.merge_files(paths, out, third_run=THIRD_FILE); back = json.load(open(out))
        assert back["module_sha256"] == R.fingerprint() == _sha(R.__file__) and back["pinned_sha256"] == PINS and back["third_run_sha256"] == THIRD_SHA256 and back["third_run_file"] == "axi_provider_bounded.json" and back["cells"] == mm["cells"]
        assert os.path.samefile(R.default_third_run_path(), THIRD_FILE) and R.load_third_run()["cells"] == tlist()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf): R.print_cells(R.summarize_all(back["cells"])); R.print_verdict(R.verdict_retry(back["cells"], tlist()))
        assert "NOT a result" not in buf.getvalue() and buf.getvalue().count("[claim]") == 7 and buf.getvalue().count("[guard]") == 3 and buf.getvalue().count("[instrument]") == 2
    # the probe: one second request for job 19, step 12, with the third run's refused reply as the assistant's turn
    class Rec(B.BoundedStandIn):
        def __init__(self): super().__init__(); self.seen = []
        def generate_round(self, requests, reset_cache=False): self.seen.append(requests); return super().generate_round(requests, reset_cache)
    e = Rec(); p = R.retry_probe(e); assert len(e.seen) == 1 and len(e.seen[0]) == 1
    msgs = e.seen[0][0].messages; step, text, oracle, reason, fb = SLIPS[19]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"] and msgs[2]["content"] == text and msgs[3]["content"] == fb and msgs[:2] == replay_to(19, 12).request().messages
    assert (p["job"], p["step"], p["kind"], p["first"], p["why"], p["feedback"], p["oracle"], p["reply"]) == (19, 12, "add_node", text, reason, fb, oracle, oracle) and p["parsed"] and p["admitted"] and p["n_new"] > 0
    assert raises(lambda: R.retry_probe(e, seed=1), (StopIteration, ValueError))


# ---------------------------------------------------------------- 7 the lines
class ReplayThird(R.FaultyStandIn):
    """First requests: the third run's saved reply of that job and step, until the job's first second request, the oracle's own reply after; extra slips for other jobs; second requests: the oracle's own reply."""
    def __init__(self, saved, extra, **kw): super().__init__(**kw); self.saved, self.extra, self.fixed = saved, extra, set()
    def reset_cache(self): self.fixed.clear(); super().reset_cache()
    def generate_round(self, requests, reset_cache=False):
        rqs = []
        for r in requests:
            if r.retry: self.fixed.add(r.seed); rqs.append(r)
            elif r.seed in self.fixed: rqs.append(r)
            elif (r.seed, r.step) in self.saved: rqs.append(R.RRequest(r.messages, self.saved[(r.seed, r.step)], seed=r.seed, step=r.step, retry=False))
            elif (r.seed, r.step) in self.extra: rqs.append(R.RRequest(r.messages, self.extra[(r.seed, r.step)], seed=r.seed, step=r.step, retry=False))
            else: rqs.append(r)
        return PS.StandInEngine.generate_round(self, rqs, reset_cache) if False else super(R.FaultyStandIn, self).generate_round(rqs, reset_cache)


def good():
    """A run that satisfies the third run's own cells: the first replies are the third run's saved replies; every second reply is the oracle's own. Reads yes on all 12 lines."""
    if "good" in _CACHE: return _CACHE["good"]
    saved = {(seed_of(j), x["t"]): x["text"] for j, s in enumerate(tcells()[("S", 64)]["sessions"]) for x in s["log"]}
    extra = {(seed_of(64 + j), 12 if i % 2 == 0 else 22): "ADD[(9,9)->(9,9):x#1]" for i, j in enumerate((6, 16, 26, 36, 46, 56))}
    eng = ReplayThird(saved, extra, capacity_tokens=int(PS.CAPACITY_TOKENS * PS.PROXY_SCALE)); _CACHE["good"] = [R.run_cell(eng, lab) for lab in R.ORDER]
    return _CACHE["good"]


def read(cells, third_cells=None): return [ok for _, _, _, ok in R.verdict_retry(cells, tlist() if third_cells is None else third_cells)]
def cell_of(cells, label): return next(c for c in cells if c["label"] == label)
def records(c): return [x for s in c["sessions"] for x in s["log"]]


def test_a_good_run_built_from_the_third_runs_own_replies_reads_yes_on_all_twelve_lines_and_the_figures_are_the_ones_the_page_expects():
    g = good(); S = R.summarize_all(g); assert read(g) == [True] * 12
    assert [S[l]["retries"] for l in R.ORDER] == [2, 4, 4, 0, 6] and [S[l]["retry_exact"] for l in R.ORDER] == [2, 4, 4, 0, 6] and [S[l]["steps_done"] for l in R.ORDER] == [240, 960, 1920, 1920, 1920]
    assert [round(S[l]["exact_share"] * 100, 2) for l in R.ORDER] == [100.0, 99.58, 99.74, 97.14, 100.0] and [S[l]["exact_steps"] for l in R.ORDER] == [240, 956, 1915, 1865, 1920]
    assert [S[l]["retry_rounds"] for l in R.ORDER] == [2, 4, 4, 0, 2] and S["H1"]["retry_seconds"] > 0 and S["H1"]["wall_seconds"] > S["H0"]["wall_seconds"]
    assert R.jobs_exact(cell_of(g, "H1")) == [30] * 64 and sorted(R.jobs_exact(cell_of(g, "H0")))[:6] == [11, 16, 21, 23, 26, 28]
    # the first replies are the third run's, step by step, until the first retry of a job
    for lab in ("R8", "R32", "R64"): assert R.same_first_replies(cell_of(g, lab), tcells()[("S", R.CELLS[lab][0])])
    assert R.same_first_replies(cell_of(g, "H1"), cell_of(g, "H0"))


def changed(cells, base=None):
    base = [True] * 12 if base is None else base; r = read(cells); return [i + 1 for i, (a, b) in enumerate(zip(base, r)) if a != b], r


def flip(label, fn):
    cells = copy.deepcopy(good()); fn(cell_of(cells, label)); return cells


def test_each_of_the_twelve_lines_flips_at_its_edge_and_a_missing_cell_reads_not_run_never_no():
    # 1: a missing cached count, in a second request or a first one
    ch, _ = changed(flip("R8", lambda c: next(x for x in records(c) if x["retried"]).update(first_n_cached=None))); assert ch == [1]
    ch, _ = changed(flip("H0", lambda c: records(c)[5].update(n_cached=None))); assert ch == [1]
    # 2: one reply with all 96 new tokens; the share of replies refused by the strict parser at its limit and one over
    ch, _ = changed(flip("H0", lambda c: records(c)[5].update(n_new=96))); assert ch == [2]
    N = sum(len(R.replies_of(c)) for c in good()); k = int(0.005 * N + 1e-9)
    def unparse(n): return lambda c: [x.update(parsed=False) for x in records(c)[:n]]
    assert changed(flip("H0", unparse(k)))[0] == [] and changed(flip("H0", unparse(k + 1)))[0] == [2], (N, k)
    # 3: a first reply that is not the earlier run's; a reply after the job's first retry may differ; a job that never asked again must have the same number of steps; the pair is compared too
    ch, _ = changed(flip("R8", lambda c: c["sessions"][0]["log"][4].update(text="DEL[0,0]"))); assert ch == [3]
    ch, _ = changed(flip("R8", lambda c: c["sessions"][3]["log"][20].update(text="DEL[0,0]"))); assert ch == [], "a step after the job's first retry is not compared"
    ch, _ = changed(flip("R8", lambda c: c["sessions"][0]["log"].pop())); assert ch == [3]
    ch, _ = changed(flip("H0", lambda c: c["sessions"][10]["log"][4].update(text="DEL[0,0]"))); assert ch == [3]
    # 4: a second request for a reply that the repair could have fixed; a retried step in the cell without the retry
    ch, _ = changed(flip("R8", lambda c: next(x for x in records(c) if x["retried"]).update(first_repairable=True))); assert ch == [4]
    ch, _ = changed(flip("R8", lambda c: next(x for x in records(c) if x["retried"]).update(first_admitted=True))); assert ch == [4]
    def mark(c, n=1, k0=0):
        for x in records(c)[k0:k0 + n]: x.update(retried=True, first_text=x["text"], first_parsed=True, first_admitted=False, first_repairable=False, first_n_prompt=x["n_prompt"], first_n_cached=x["n_cached"], first_n_new=x["n_new"], first_reason="x", feedback="x")
    ch, _ = changed(flip("H0", lambda c: mark(c, 1, 3))); assert ch == [4]
    # 5: the retry is rare: at 240 steps 7 of them (2.9 percent) is allowed, 8 (3.3 percent) is not; R8 has 2 already
    def more(n): return lambda c: mark(c, n, 0)
    assert changed(flip("R8", more(5)))[0] == [] and changed(flip("R8", more(6)))[0] == [5]
    # 6: exactly the third run's share is not higher
    def lose(label_n, per): return lambda c: [x.update(exact=False) for x in [y for y in records(c) if y["exact"]][-label_n:]]
    assert changed(flip("R32", lose(47, 0)))[0] == [] and changed(flip("R32", lose(48, 0)))[0] == [6]
    assert changed(flip("R8", lose(27, 0)))[0] == [] and changed(flip("R8", lose(28, 0)))[0] == [6]                      # 240 against 212 at 8 jobs; 956 against 908 at 32; 1,915 against 1,867 at 64
    assert changed(flip("R64", lose(47, 0)))[0] == [7, 8] and changed(flip("R64", lose(48, 0)))[0] == [6, 7, 8]
    ch, _ = changed(flip("R32", lambda c: c["sessions"][5]["log"][3].update(text="DEL[0,0]"))); assert ch == [3]
    ch, _ = changed(flip("R64", lambda c: c["sessions"][5]["log"][3].update(text="DEL[0,0]"))); assert ch == [3]
    # 7 and 8: R64 holds 1,915 of 1,920; 98 percent is 1,882 steps, 99 percent is 1,901
    assert changed(flip("R64", lose(14, 0)))[0] == [] and changed(flip("R64", lose(15, 0)))[0] == [8]
    assert changed(flip("R64", lose(33, 0)))[0] == [8] and changed(flip("R64", lose(34, 0)))[0] == [7, 8]
    # 9: H0 holds 1,836 of 1,920; one point below H1's 100 percent is 1,900 steps
    def gain(n): return lambda c: [x.update(exact=True) for x in [y for y in records(c) if not y["exact"]][:n]]
    assert sum(1 for x in records(cell_of(good(), "H0")) if not x["exact"]) == 55
    assert changed(flip("H0", gain(31)))[0] == [] and changed(flip("H0", gain(32)))[0] == [11]            # H0 at 1,897 exact steps overtakes H1 in the steps per hour (line 11), not yet in the share (line 9)
    assert changed(flip("H0", gain(35)))[0] == [11] and changed(flip("H0", gain(36)))[0] == [9, 11]
    # 10 and 11: the seconds per step with the retry at most 1.10 times those without; the exact steps per hour higher
    def wall(f): return lambda c: c.update(wall_seconds=cell_of(good(), "H0")["wall_seconds"] * f)
    r = read(flip("H1", wall(1.10))); assert r[9] is True and r[10] is False
    r = read(flip("H1", wall(1.1001))); assert r[9] is False and r[10] is False
    r = read(flip("H1", wall(1.02))); assert r[9] is True and r[10] is True
    r = read(flip("H1", wall(1.029))); assert r[10] is True                                                                       # 1,920 of 1,865: the exact steps per hour are higher up to 1.0295 times the seconds
    r = read(flip("H1", wall(1.03))); assert r[9] is True and r[10] is False
    r = read(flip("H1", wall(0.5))); assert r[9] is True and r[10] is True
    # 12: one job of the pair with fewer exact steps with the retry; the same number is allowed
    ch, _ = changed(flip("H1", lambda c: c["sessions"][0]["log"][29].update(exact=False))); assert ch == [12]
    both = copy.deepcopy(good()); both_log = lambda lab: cell_of(both, lab)["sessions"][0]["log"]
    both_log("H0")[29].update(exact=False); both_log("H1")[29].update(exact=False); assert read(both) == [True] * 12       # the same number of exact steps in a job is allowed
    both_log("H1")[28].update(exact=False); assert changed(both)[0] == [12]                                                                               # one fewer is not
    # the lines are numbered 1 to 12 with the page's tags and words; fewer sessions than the third run's is a different set of jobs; equal exact steps per hour is not higher
    assert [(n, t, x) for n, t, x, _ in R.verdict_retry(good(), tlist())] == [(i, t, x) for i, (t, x) in enumerate(R.RL_LINES, 1)]
    ch, _ = changed(flip("R8", lambda c: c["sessions"].pop())); assert ch == [3]
    same = copy.deepcopy(good()); h0 = cell_of(same, "H0"); h1 = cell_of(same, "H1"); h1["sessions"] = copy.deepcopy(h0["sessions"]); h1["rounds"] = copy.deepcopy(h0["rounds"]); h1["wall_seconds"] = h0["wall_seconds"]
    assert read(same)[8:] == [False, True, False, True]                                                       # no gain (9), the same seconds per step (10), the same steps per hour is not higher (11), no job worse (12)
    # a missing cell reads not run, never NO
    g = [c for c in good() if c["label"] != "H1"]; r = read(g); assert r[8:] == [None] * 4 and r[:8] == [True] * 8
    g = [c for c in good() if c["label"] != "R64"]; r = read(g); assert r[6:8] == [None, None] and r[5] is True and r[:6] == [True] * 6 and r[8:] == [True] * 4
    g = [c for c in good() if c["label"] != "H0"]; r = read(g); assert r[8:] == [None] * 4 and r[2] is True
    assert read([]) == [None] * 12 and read([c for c in good() if c["label"] == "R8"], []) == [True, True, None, True, True, None, None, None, None, None, None, None]
    empty = copy.deepcopy(good()); [s.update(log=[]) for s in cell_of(empty, "H1")["sessions"]]; assert read(empty)[8:] == [None] * 4                      # a cell with no step is a cell that did not run
    # the third run's cells missing: the lines that compare with them read not run
    r = read(good(), []); assert r == [True, True, True, True, True, None, True, True, True, True, True, True]


def test_the_rule_about_what_may_be_claimed_follows_lines_2_3_and_4_and_the_printout_says_it():
    ok = R.claims_allowed(R.verdict_retry(good(), tlist())); assert ok == dict(base=True, constraint_held=True, first_replies_as_before=True, fired_by_the_rule=True, same_jobs={6: True, 7: True, 8: True}, new_jobs_exactness={9: True, 12: True},
                                                                              time_and_cost_in_one_session={10: True, 11: True})
    for i, name in ((2, "constraint_held"), (3, "first_replies_as_before"), (4, "fired_by_the_rule")):
        checks = [(n, t, x, (False if n == i else v)) for n, t, x, v in R.verdict_retry(good(), tlist())]; a = R.claims_allowed(checks)
        assert a["base"] is False and a[name] is False and a["same_jobs"] == {6: None, 7: None, 8: None} and a["new_jobs_exactness"] == {9: None, 12: None} and a["time_and_cost_in_one_session"] == {10: None, 11: None}, i
    checks = [(n, t, x, (False if n in (9, 11) else v)) for n, t, x, v in R.verdict_retry(good(), tlist())]; a = R.claims_allowed(checks); assert a["base"] and a["new_jobs_exactness"] == {9: False, 12: True} and a["time_and_cost_in_one_session"] == {10: True, 11: False}
    for shown in (R.verdict_retry(good(), tlist()), [(n, t, x, (False if n == 3 else v)) for n, t, x, v in R.verdict_retry(good(), tlist())]):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf): R.print_verdict(shown)
        out = buf.getvalue(); assert ("never: a multiple of capacity" in out) != ("NO claim is made from this run" in out)
        assert out.count("yes ") + out.count("NO ") >= 12 and "headline lines 9 and 11" in out
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): R.print_verdict(R.verdict_retry(good(), tlist()))
    out = buf.getvalue(); assert "rests on lines 9 and 12 (new jobs) only" in out and "only from the pair in one session (lines 10 and 11)" in out and "'100 percent'" in out and "anything about more than one retry" in out
    assert out.endswith("  rule on the page: lines 2, 3 and 4 read yes, so the figures of the cells are claimed as read. That the retry raises the exactness rests on lines 9 and 12 (new jobs) only; lines 6 to 8 are on the jobs the slips were read on.\n"
                        "  a statement about time or cost comes only from the pair in one session (lines 10 and 11), never from a cell against the third run's.\n"
                        "  never: a multiple of capacity, dollars, '100 percent', a statement about any provider, anything about more than one retry.\n")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): S = R.dry_run(steps=4)
    assert buf.getvalue().startswith("STAND-IN (a model of a block cache; the third run's four slips are written by the test double and the second request is answered with the oracle's own reply), NOT a result\n")
    assert sorted(S) == sorted(R.ORDER) and all(s["steps_done"] > 0 for s in S.values())


def test_the_command_line_merges_cell_files_prints_the_lines_and_shows_its_help_and_the_dry_run_says_it_is_not_a_result():
    env = dict(os.environ, PYTHONPATH=os.path.join(HERE, ".."))
    h = subprocess.run([sys.executable, "-m", "axi.experiments.provider_side_retry"], capture_output=True, text=True, env=env, cwd=os.path.join(HERE, "..")); assert h.returncode == 0 and "--dry-run" in h.stdout and "--merge" in h.stdout and "run every cell on the stand-in engine (no GPU); prints, does not save" in " ".join(h.stdout.split()) and "put cell files together" in " ".join(h.stdout.split())
    d = subprocess.run([sys.executable, "-m", "axi.experiments.provider_side_retry", "--dry-run"], capture_output=True, text=True, env=env, cwd=os.path.join(HERE, ".."))
    assert d.returncode == 0 and "NOT a result" in d.stdout and "R64" in d.stdout and "H1" in d.stdout
    with tempfile.TemporaryDirectory() as t:
        eng = R.FaultyStandIn(faults=R.third_run_slips()); paths = [R.save_cell(R.run_cell(eng, lab, steps=3), os.path.join(t, lab + ".json")) for lab in R.ORDER]; out = os.path.join(t, "m.json")
        m = subprocess.run([sys.executable, "-m", "axi.experiments.provider_side_retry", "--merge", *paths, "--out", out], capture_output=True, text=True, env=env, cwd=os.path.join(HERE, ".."))
        assert m.returncode == 0 and "written to" in m.stdout and "-- provider-side test, one retry" in m.stdout and m.stdout.count("[claim]") == 7 and os.path.exists(out)



# ---------------------------------------------------------------- 8 the locked page
def _h(text): return hashlib.sha256((text.rstrip("\n") + "\n").encode()).hexdigest()


def _head():
    text = open(PAGE).read(); assert text.count(RESULT_MARK) <= 1
    return text[:text.index(RESULT_MARK)] if RESULT_MARK in text else text


def test_the_page_is_locked_and_carries_the_twelve_lines_the_rule_the_four_feedback_texts_the_pins_and_the_thresholds_the_code_uses():
    head = _head(); assert _h(head) == PAGE_SHA256, "the predictions and reading rules must not change after the run; later notes go below them"
    for i, (tag, line) in enumerate(R.RL_LINES, 1): assert f"\n{i}. [{tag}] {line} (" in head, i
    probs = [int(m.group(1)) for m in re.finditer(r"\n\d+\. \[\w+\] .*\((\d+) %\)(?=\n)", head)]; assert probs == PROBS, probs
    assert [tag for tag, _ in R.RL_LINES] == ["instrument", "instrument", "guard", "guard", "guard"] + ["claim"] * 7 and R.RL_HEADLINE == (9, 11)
    for needle in ("**Headline** (lines 9 and 11)", "Lines 2, 3 and 4 all reading yes: about **75 %**", "Lines 9 and 11 both reading yes: about **30 %**", "(lines 1 to 7, 9, 10 and 12 yes; 8 and 11 NO): about **12 %**", "All 12 yes: about **6 %**"):
        assert needle in head, needle
    for name, sha in PINS.items(): assert sha in head, name
    assert f"`{REGEX_SHA256}`" in head and "`python/axi/experiments/provider_side_retry.py`" in head and "Concept-map item 109" in head and "and edits none of them" in head
    for job, (step, text, oracle, reason, fb) in SLIPS.items(): assert f"  * job {job}: `{fb}`" in head and f"| {job} | {step} | `{text}` | `{oracle}` |" in head, job
    for job, (step, text, oracle) in ADMITTED_MISSES.items(): assert f"| {job} | {step} | `{text}` | `{oracle}` | admitted | not needed | {LOST[job]} |" in head, job
    for job, (step, *_r) in SLIPS.items(): assert f"| {job} | {step} |" in head and f"| cannot fix | {LOST[job]} |" in head, job
    th = (f"line 5 each cell with the retry (retried steps over steps, at most {R.R_MAX_SHARE:g}); line 6 `R8`, `R32` and `R64` against the third run's exact share (strictly higher in each); lines 7 and 8 `R64` ({R.R_HIGH:g} and {R.R_VERY_HIGH:g}); "
          f"line 9 `H1` against `H0` (the exact share at least {R.R_HELD_GAIN:g} higher); line 10 the seconds per step of `H1` against `H0` (at most {1 + R.R_TIME_TOL:.2f} times)")
    assert th in head, th
    assert "at least 99.5 percent were accepted by the strict parser" in head and "at most 3 percent of the steps asked again" in head and "at least 98 percent" in head and "at least 99 percent" in head and "at least one point above" in head and "at most 10 percent above" in head
    for needle in ("**The rule that decides what the Result may claim.**", "Line 2, 3 or 4 reads NO: no claim is made from this run until the cause is found.", "That the retry raises the exactness is claimed only from the jobs that no earlier run had seen (lines 9 and 12)",
                   "A statement about time or cost comes only from the pair in one session (lines 10 and 11), never from a cell against the third run's (another Colab session).",
                   "Never a multiple of capacity, never a count of GPUs saved, never dollars or a percentage of cost, never \"100 percent\" for the arm, never \"proven\"", "never a statement about more than one retry",
                   "**I do not know whether this 7B model uses the feedback**", "the rule below does not fire on any of the 507", "ask once and walk away", "A second refusal is accepted", "At most one retry per step.",
                   "A reply that does not parse is not asked again", "the lockstep model charges the whole round, which is against the retry", "**Only the pair can say anything about jobs in general, and only the pair gives a time comparison that is not across two Colab sessions.**",
                   "**a wrong answer is not a STOP**", "A cell that did not run:* its lines read \"not run\", never NO."):
        assert needle in head, needle
    assert head.count("\n# ") == 0 and head.startswith("# Test 17: the serving side a fourth time") and "(written BEFORE the run, 19 Oct 2026)" in head
    assert head.rstrip("\n").endswith("\n---")


def test_the_figures_the_page_quotes_from_the_earlier_runs_are_the_ones_in_their_saved_files():
    head = _head(); tc = tcells(); S = {K: C.summarize_cell2(tc[("S", K)]) for K in (8, 32, 64)}; second = json.load(open(os.path.join(RES, "axi_provider_constrained.json"))); sc = {(c["arm"], c["K"]): c for c in second["cells"]}
    for s in ("88.3, 94.6 and 97.2 percent", "39.4, 119.3 and 217.4 wall seconds; 0.164, 0.124 and 0.113 seconds per step; 19,379, 27,390 and 30,914 exact steps per GPU hour", "212, 908 and 1,867 now",
              "240 of 240 at 8 jobs (100.0 percent; the two jobs that missed there are 3 and 4), 956 of 960 at 32 jobs (99.6), 1,915 of 1,920 at 64 jobs (99.7)", "507 deletes of a service that still had edges", "491 end exact and the other 16 are steps of jobs 3, 4 and 19",
              "six of the 64 jobs missed", "48 of the 53 lost steps at 64 jobs are theirs** (jobs 3, 4, 19 and 20: 15, 13, 19 and 1)", "they are 5 steps", "recovered all 120 such deletes on four seeds", "the rule fires in at most 10 different rounds without a retry",
              "A second Colab session was 4 to 8 percent slower in every cell than the one before it"):
        assert s in head, s
    assert [round(100 * S[K]["exact_share"], 1) for K in S] == [88.3, 94.6, 97.2] and [S[K]["exact_steps"] for K in S] == [212, 908, 1867]
    ratios = [C.summarize_cell2(tc[k])["seconds_per_step"] / C.summarize_cell2(sc[k])["seconds_per_step"] - 1 for k in (("S", 8), ("S", 32), ("S", 64), ("KC", 8))]; assert all(0.04 <= r <= 0.08 for r in ratios), ratios
    n = same = 0
    for c in tlist():
        for a, b in zip(c["sessions"], sc[(c["arm"], c["K"])]["sessions"]):
            for x, y in zip(a["log"], b["log"]): n += 1; same += x["text"] == y["text"]
    assert (n, same) == (3360, 3358) and "replies were the earlier run's in 3,358 of 3,360 steps in a new session" in head
    assert [sum(1 for s in tc[("S", 8)]["sessions"] if any(not x["exact"] for x in s["log"])), sum(1 for s in tc[("S", 32)]["sessions"] if any(not x["exact"] for x in s["log"])), sum(1 for s in tc[("S", 64)]["sessions"] if any(not x["exact"] for x in s["log"]))] == [2, 5, 6]
    assert sum(1 for s in tc[("S", 64)]["sessions"] if all(x["exact"] for x in s["log"])) == 58 and "(the third run: 58)" in head
    assert [j for j, s in enumerate(tc[("S", 8)]["sessions"]) if any(not x["exact"] for x in s["log"])] == [3, 4]


# ---------------------------------------------------------------- 9 the notebook
NOTEBOOK = os.path.join(ROOT, "python", "colab", "axi_provider_side_retry.ipynb")
def _nb(): return json.load(open(NOTEBOOK))
def _srcs(): return ["".join(c["source"]) for c in _nb()["cells"]], [c["cell_type"] for c in _nb()["cells"]]


def test_the_notebook_installs_the_pinned_engine_fixes_the_stream_loads_the_engine_checks_runs_the_five_cells_in_order_and_puts_them_together():
    srcs, kinds = _srcs(); code = [s for s, k in zip(srcs, kinds) if k == "code"]; text = "\n".join(srcs); nb = _nb()
    import ast
    for s in code: ast.parse("\n".join(l for l in s.splitlines() if not l.lstrip().startswith("!")))
    for needle in ("git clone -q --depth 1 https://github.com/gaurshalendra2412/axi-runtime", "provider_side_retry.py", "axi_provider_bounded.json", "R.BoundedVLLMEngine()", "eng.warm_up()", "R.run_jobs(eng, [0, 1, 2], True, steps=3)", "R.retry_probe(eng)", "STOP:", "R.run_cell(eng, label, progress=progress)",
                   "R.save_cell", "store.persist", "R.merge_files", "R.verdict_retry", "R.print_cells", "R.print_verdict", "R.load_third_run", "files.download", "open_store", "docs/PROVIDER_SIDE_RETRY_TEST.md", "Do not paste any keys", "update 62 commands",
                   "12 reading-rule lines", "axi_provider_r_", "third_run=THIRD", "for label in R.ORDER: go(label)"): assert needle in text, needle
    assert text.count("Do not paste any keys") >= 2 and "!pip -q install vllm==0.31.0" in [x.strip() for x in code] and "!pip -q install vllm" not in [x.strip() for x in code] and "open_store(\"/content/out\", \"axi_provider_retry\")" in text
    fix = [i for i, (s, k) in enumerate(zip(srcs, kinds)) if k == "code" and "fileno" in s]; load = [i for i, (s, k) in enumerate(zip(srcs, kinds)) if k == "code" and "R.BoundedVLLMEngine()" in s]; assert len(fix) == 1 and len(load) == 1 and fix[0] < load[0]
    ran = []; ns = dict(go=lambda label: ran.append(label), R=R)
    for s in code:
        if re.fullmatch(r"for label in R\.ORDER: go\(label\)\s*", s.strip()): exec(s, ns)
    assert ran == R.ORDER == ["R8", "R32", "R64", "H0", "H1"], ran
    assert nb["metadata"]["accelerator"] == "GPU" and nb["nbformat"] == 4 and kinds[0] == "markdown" and kinds.count("code") == 9
    head = srcs[0]; assert "T4" in head and "80,000 tokens" in head and "Qwen2.5-7B-Instruct" in head and "AWQ" in head and "vLLM 0.31.0" in head and "40 to 45 minutes" in head and "Keep this browser tab open" in head and "Choose the T4 even if your plan offers a faster GPU" in head and "About 15 minutes in all" in "\n".join(srcs)
    assert "The kept chat is not run" in head and "88.3, 94.6 and 97.2 percent" in head and "docs/PROVIDER_SIDE_RETRY_TEST.md" in head
    assert "`H0`" in head and "`H1`" in head and "A wrong answer is not a STOP" in "\n".join(srcs)


def test_the_notebook_cell_runner_skips_finished_cells_and_starts_a_cell_the_last_session_cut_off_again_from_its_beginning():
    srcs, kinds = _srcs(); runner = next(s for s, k in zip(srcs, kinds) if k == "code" and "def go(label):" in s); body = runner[runner.index("def go(label):"):]
    class Crash(Exception): pass
    class Dying(R.FaultyStandIn):
        def __init__(self, die_at): super().__init__(faults=R.third_run_slips()); self.n = 0; self.die_at = die_at
        def generate_round(self, requests, reset_cache=False):
            self.n += 1
            if self.n == self.die_at: raise Crash()
            return super().generate_round(requests, reset_cache)
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as remote:
        sys.path.insert(0, os.path.join(HERE, ".."))
        from colab.colab_persist import Store
        shim = types.SimpleNamespace(CELLS=R.CELLS, run_cell=lambda e, label, progress=None: R.run_cell(e, label, steps=4, progress=progress), save_cell=R.save_cell, summarize_cell3=R.summarize_cell3)
        def session(engine):
            store = Store(d, remote); ns = dict(R=shim, eng=engine, store=store, os=os, time=time, json=json); exec(body, ns); return ns["go"]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            go = session(Dying(die_at=3))
            try: go("R8")
            except Crash: pass
            assert os.listdir(d) == [] and os.listdir(remote) == []                                                       # nothing half-finished is kept: the cell starts again from its beginning
            go = session(R.FaultyStandIn(faults=R.third_run_slips())); go("R8"); go("H1"); go("R8")
        out = buf.getvalue()
        assert out.count("already finished, skipping") == 1 and out.count("done:") == 2 and "=== R8: jobs 0 to 7, retry on" in out and "=== H1: jobs 64 to 127, retry on" in out
        s = R.summarize_cell3(R.run_cell(R.FaultyStandIn(faults=R.third_run_slips()), "R8", steps=4)); f = lambda x, k=1: "-" if x is None else f"{x:.{k}f}"
        want = (f"done: {s['steps_done']} steps, {s['wall_seconds']:.0f} s, {f(s['seconds_per_step'], 3)} s/step, asked again {s['retries']} times in {s['retry_rounds']} rounds ({f(s['retry_seconds'])} s), put right {s['retry_exact']}, "
                f"exact steps {s['exact_share'] * 100:.1f} %, jobs exact to the end {s['jobs_exact_to_end']}, refused by the strict parser {s['unparsed_all']}, exact steps per hour {f(s['exact_per_hour'], 0)}")
        assert want in out, (want, out)
        assert sorted(f for f in os.listdir(remote)) == ["axi_provider_r_H1.json", "axi_provider_r_R8.json"] and sorted(os.listdir(d)) == sorted(os.listdir(remote))


def test_the_notebooks_last_cell_puts_the_cells_together_reads_the_third_runs_file_prints_the_lines_and_offers_the_download():
    srcs, kinds = _srcs(); cell = next(s for s, k in zip(srcs, kinds) if k == "code" and "R.merge_files" in s)
    assert cell.count("/content/axi/docs/results/axi_provider_bounded.json") == 1
    cell = cell.replace("/content/axi/docs/results/axi_provider_bounded.json", THIRD_FILE)
    sys.path.insert(0, os.path.join(HERE, ".."))
    from colab.colab_persist import Store
    got = []; gc = types.ModuleType("google.colab"); gc.files = types.SimpleNamespace(download=lambda p: got.append(p)); g = types.ModuleType("google"); g.colab = gc
    saved = {k: sys.modules.get(k) for k in ("google", "google.colab")}
    sys.modules["google"], sys.modules["google.colab"] = g, gc
    try:
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as remote:
            store = Store(d, remote)
            for lab in R.ORDER: R.save_cell(R.run_cell(R.FaultyStandIn(faults=R.third_run_slips()), lab, steps=4), store.path(f"axi_provider_r_{lab}.json"))
            buf = io.StringIO(); ns = dict(R=R, store=store, glob=glob, os=os)
            with contextlib.redirect_stdout(buf): exec(cell, ns)
            out = buf.getvalue(); assert "5 cells:" in out and "-- provider-side test, one retry" in out and out.count("[claim]") == 7 and out.count("[guard]") == 3 and out.count("[instrument]") == 2
            assert ("never: a multiple of capacity" in out) != ("NO claim is made from this run" in out)
            assert got == [store.path("axi_provider_retry.json")] and os.path.exists(os.path.join(remote, "axi_provider_retry.json"))
            m = json.load(open(store.path("axi_provider_retry.json"))); assert [c["label"] for c in m["cells"]] == R.ORDER and m["third_run_sha256"] == THIRD_SHA256
    finally:
        for k, v in saved.items():
            if v is None: sys.modules.pop(k, None)
            else: sys.modules[k] = v


def test_the_stream_fix_cell_gives_the_output_the_numbers_1_and_2_when_it_has_none_and_leaves_a_real_stream_alone():
    srcs, kinds = _srcs(); cell = next(s for s, k in zip(srcs, kinds) if k == "code" and "fileno" in s); seen = []
    class Bare:
        def write(self, s): seen.append(s)
    class Real:
        def fileno(self): return 77
        def write(self, s): seen.append(s)
    old = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = Bare(), Real(); exec(cell, {}); a, b = sys.stdout.fileno(), sys.stderr.fileno()
        sys.stdout, sys.stderr = Real(), Bare(); exec(cell, {}); c, d = sys.stdout.fileno(), sys.stderr.fileno()
    finally: sys.stdout, sys.stderr = old
    assert (a, b, c, d) == (1, 77, 77, 2) and "output numbers: 1 77" in "".join(seen) and "output numbers: 77 2" in "".join(seen)


def test_the_load_cell_prints_the_engines_numbers_and_the_cap_and_defines_the_runner():
    srcs, kinds = _srcs(); cell = next(s for s, k in zip(srcs, kinds) if k == "code" and "R.BoundedVLLMEngine()" in s)
    mods, calls = T124._fake_vllm("module")
    with T124._installed(mods), tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as remote:
        sys.path.insert(0, os.path.join(HERE, ".."))
        from colab.colab_persist import Store
        ns = dict(json=json, os=os, time=time, store=Store(d, remote)); buf = io.StringIO()
        with contextlib.redirect_stdout(buf): exec(cell, ns)
        out = buf.getvalue()
        assert f"name at most 16 characters {REGEX_SHA256[:16]} (the same for the first and the second request)" in out
        assert "cache size the engine will use: 80000 tokens ( override )" in out and callable(ns["go"]) and isinstance(ns["eng"], B.BoundedVLLMEngine) and ns["R"] is R and calls["resets"] >= 1 and '"constraint"' not in out and '"kind": "vllm"' in out


def test_the_safety_cell_says_ok_on_a_working_engine_and_stop_for_each_thing_that_can_be_wrong_and_a_wrong_answer_of_the_probe_is_not_a_stop():
    srcs, kinds = _srcs(); cell = next(s for s, k in zip(srcs, kinds) if k == "code" and "smoke = " in s)
    def run(engine):
        buf = io.StringIO(); ns = dict(R=R, eng=engine); ns["re"] = re
        with contextlib.redirect_stdout(buf): exec(cell, ns)
        return buf.getvalue()
    out = run(R.BoundedStandIn()); last = out.strip().splitlines()[-1]
    assert last.startswith("OK: cached counts are reported") and "STOP" not in out and "(job 19, step 12, add_node)" in out and "the strict parser reads it" in out and "the gate admits it" in out and SLIPS[19][4] in out and SLIPS[19][3] in out
    class NoCount(R.BoundedStandIn):
        def generate_round(self, requests, reset_cache=False): reps, dt = super().generate_round(requests, reset_cache); [setattr(r, "n_cached", None) for r in reps]; return reps, dt
    def probe_says(text, n_new):
        class P(R.BoundedStandIn):
            def generate_round(self, requests, reset_cache=False):
                reps, dt = super().generate_round(requests, reset_cache)
                if len(requests) == 1: reps[0].text = text; reps[0].n_new = n_new
                return reps, dt
        return P()
    assert "STOP: this vLLM version does not report cached tokens" in run(NoCount())
    o = run(probe_says("Sure, here is the corrected delta: ADD[3,0:4]", 20)); assert "STOP: the probe's reply is refused by the strict parser" in o and "REFUSES it" in o
    o = run(probe_says("ADD[3,0:4]", 96)); assert "STOP: the probe's reply used all 96 new tokens" in o
    o = run(probe_says("ADD[(3,0)->(3,0):4tworowsdown4col#4]", 31)); assert o.strip().splitlines()[-1].startswith("OK:") and "the gate does not admit it" in o and "STOP" not in o         # the same wrong answer again is a finding, not a STOP
    o = run(probe_says("ADD[3,0:4]", 9)); assert o.strip().splitlines()[-1].startswith("OK:") and "the gate admits it" in o
    class Garbage(R.BoundedStandIn):
        def generate_round(self, requests, reset_cache=False): reps, dt = super().generate_round(requests, reset_cache); reps[0].text = "Sure, here is the delta: " + reps[0].text; return reps, dt
    assert "STOP: a reply of the safety cell was refused by the strict parser" in run(Garbage())


# ---------------------------------------------------------------- 10 what the front pages say about the pasted texts
def test_the_counts_the_front_pages_give_about_the_stray_weight_delete_and_the_old_99_4_percent_are_the_ones_in_the_saved_files():
    pat = re.compile(r"DEL\[\d+,\d+:\d+\]"); counts = {}
    for name in ("axi_provider_side.json", "axi_provider_constrained.json", "axi_provider_bounded.json"):
        d = json.load(open(os.path.join(RES, name))); texts = [x["text"] for c in d["cells"] for s in c["sessions"] for x in s["log"]]
        counts[name] = (sum(1 for t in texts if pat.search(t)), sum(1 for t in texts if t.lstrip().startswith("DEL")), len(texts))
        if name == "axi_provider_side.json": where = {(c["arm"], c["K"]): sum(1 for s in c["sessions"] for x in s["log"] if pat.search(x["text"])) for c in d["cells"]}
    assert counts == {"axi_provider_side.json": (84, 1599, 4552), "axi_provider_constrained.json": (0, 2191, 6231), "axi_provider_bounded.json": (0, 1183, 3360)}, counts
    assert {k: v for k, v in where.items() if v} == {("S", 8): 5, ("S", 32): 14, ("KC", 32): 9, ("S", 64): 56}
    first = json.load(open(os.path.join(RES, "axi_provider_side.json"))); assert all(c.get("engine", {}).get("constraint") is None for c in first["cells"])                  # the first run: no grammar
    older = open(os.path.join(ROOT, "docs", "PROVIDER_SIDE_TEST.md")).read(); assert "held the exact state on 99.4 % of the steps in state mode with constrained decoding (`LONG_HORIZON_TEST.md`)" in older
    for name in ("axi_provider_constrained.json", "axi_provider_bounded.json"):
        d = json.load(open(os.path.join(RES, name))); assert all(c["engine"]["constraint"]["kind"] == "regex" for c in d["cells"]), name                                               # the second and third runs: the grammar on in every cell


# ---------------------------------------------------------------- 12 the summary of a cell and what is printed
def test_the_summary_of_a_cell_counts_the_second_requests_the_unparsed_and_the_truncated_replies_and_the_jobs_exact_to_the_end():
    c = R.run_cell(R.FaultyStandIn(faults=R.third_run_slips()), "R8"); S0 = R.summarize_cell3(c)
    assert (S0["retries"], S0["retry_exact"], S0["retry_admitted"], S0["requests"], S0["unparsed_all"], S0["truncated_all"], S0["jobs_exact_to_end"], S0["retry_rounds"]) == (2, 2, 2, 242, 0, 0, 8, 2) and S0["retry_share"] == 2 / 240 and S0["label"] == "R8" and S0["retry"] is True
    def tweak(fn):
        d = copy.deepcopy(c); fn(d); return R.summarize_cell3(d)
    recs = lambda d: [x for s_ in d["sessions"] for x in s_["log"]]
    retried = lambda d: [x for x in recs(d) if x["retried"]]
    assert tweak(lambda d: recs(d)[0].update(parsed=False))["unparsed_all"] == 1 and tweak(lambda d: retried(d)[0].update(first_parsed=False))["unparsed_all"] == 1 and tweak(lambda d: retried(d)[0].update(parsed=False))["unparsed_all"] == 1
    assert tweak(lambda d: recs(d)[0].update(n_new=95))["truncated_all"] == 0 and tweak(lambda d: recs(d)[0].update(n_new=96))["truncated_all"] == 1 and tweak(lambda d: retried(d)[0].update(first_n_new=96))["truncated_all"] == 1
    assert tweak(lambda d: retried(d)[0].update(admitted=False))["retry_admitted"] == 1 and tweak(lambda d: retried(d)[0].update(exact=False))["retry_exact"] == 1
    assert tweak(lambda d: d["sessions"][0]["log"][4].update(exact=False))["jobs_exact_to_end"] == 7 and tweak(lambda d: d["sessions"][0]["log"].pop())["jobs_exact_to_end"] == 7 and tweak(lambda d: d["sessions"][0].update(script_steps=31))["jobs_exact_to_end"] == 7
    assert tweak(lambda d: d["rounds"].append(dict(round=99, kind="retry", seconds=2.5, active=1)))["retry_rounds"] == 3 and abs(tweak(lambda d: d["rounds"].append(dict(round=99, kind="retry", seconds=2.5, active=1)))["retry_seconds"] - S0["retry_seconds"] - 2.5) < 1e-9
    assert tweak(lambda d: [s_.update(log=[]) for s_ in d["sessions"]])["retry_share"] is None
    assert raises(lambda: R.summarize_all([c, c]), AssertionError) and sorted(R.summarize_all([c, R.run_cell(R.FaultyStandIn(), "R32", steps=2)])) == ["R32", "R8"]


def test_the_table_and_the_lines_are_printed_in_the_format_the_notebook_shows():
    S = {"R8": dict(steps_done=240, wall_seconds=39.4, seconds_per_step=0.164, retries=2, retry_rounds=2, retry_seconds=1.55, retry_exact=2, exact_share=1.0, jobs_exact_to_end=8, unparsed_all=3, exact_per_hour=21929.0),
         "H0": dict(steps_done=1920, wall_seconds=210.9, seconds_per_step=0.1098, retries=0, retry_rounds=0, retry_seconds=0.0, retry_exact=0, exact_share=0.97135, jobs_exact_to_end=58, unparsed_all=0, exact_per_hour=None)}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): R.print_cells(S)
    assert buf.getvalue() == ("\ncell   K      jobs retry | steps  seconds  s/step | asked rounds retry s fixed |  exact jobs ok unparsed  exact/h\n"
                              "R8     8       0-7    on |   240       39   0.164 |     2      2     1.6     2 | 100.0%       8        3    21929\n"
                              "H0    64    64-127   off |  1920      211   0.110 |     0      0     0.0     0 |  97.1%      58        0        -\n")
    checks = [(1, "instrument", "aaa", True), (2, "instrument", "bbb", False), (3, "guard", "ccc", None)] + [(i, "claim", f"x{i}", True) for i in range(4, 13)]; buf = io.StringIO()
    with contextlib.redirect_stdout(buf): R.print_verdict(checks)
    want = ("\n-- provider-side test, one retry (reading rules fixed before the run) --\n  yes       1. [instrument] aaa\n  NO        2. [instrument] bbb\n  not run   3. [guard] ccc\n" + "".join(f"  yes      {i:2d}. [claim] x{i}\n" for i in range(4, 13)) +
            "  (11 of 12 lines could be read; headline lines 9 and 11: yes, yes)\n  rule on the page: line 2, 3 or 4 did not read yes, so NO claim is made from this run until the cause is found.\n")
    assert buf.getvalue() == want, buf.getvalue()
    buf = io.StringIO(); checks[1] = (2, "instrument", "bbb", True); checks[2] = (3, "guard", "ccc", True); checks[10] = (11, "claim", "x11", False); checks[8] = (9, "claim", "x9", None)
    with contextlib.redirect_stdout(buf): R.print_verdict(checks, title="t")
    o = buf.getvalue(); assert o.startswith("\n-- t (reading rules fixed before the run) --\n") and "  (11 of 12 lines could be read; headline lines 9 and 11: not run, NO)\n" in o and "  NO       11. [claim] x11\n" in o and "  not run   9. [claim] x9\n" in o
    assert o.endswith("  never: a multiple of capacity, dollars, '100 percent', a statement about any provider, anything about more than one retry.\n") or "never: a multiple of capacity, dollars, '100 percent', a statement about any provider, anything about more than one retry." in o


# ---------------------------------------------------------------- 11 the front pages
FRONT_ITEM = ("109. **The serving side a fourth time, with one retry when the repair cannot fix a refused reply (19 Oct)**", "\"without changing the core of the framework\"", "replies with that stray weight were 84 of the 1,599 delete replies of the first run (no grammar) and 0 of 2,191 and 0 of 1,183 in the second and third",
              "the 99.4 % belongs to the older long-horizon test (state mode, a different setup)", "48 belong to four jobs (3, 4, 19 and 20) whose reply the gate refused and the repair of `repair_m6` could not fix", "the 507 deletes of a service with edges that the gate refuses are all completed by the repair (491 end exact, 16 are in jobs already behind)",
              "rows 26 and 42, `agent_loop.py` mode M3", "Status: RUN (the owner's Colab run, five cells in one session; the file reached the repository work on 20 Oct 2026); the Result is on the page.**", "hash-locked; sha256 " + PAGE_SHA256 + " of the text above the Result", "and edits none of them: their sha256 are pinned in the test, so the core of the framework is unchanged",
              "= **test 128** (23 tests:", "the 12 lines at their edges on a good run built from the third run's own replies", "`structured_imperative` feedback, for example `OBSTRUCTION ident_node node=(1,0) Reply with only the corrected delta.`", "a second refusal is accepted; the second round is paid for in the cell's seconds",
              "`R8`, `R32` and `R64` (jobs 0 to 7, 0 to 31 and 0 to 63, retry on) and the pair `H0`/`H1` on jobs 64 to 127, which no earlier run has seen", "the rule fires on 5, 12 and 12 steps (2.1, 1.25 and 0.6 %), only in jobs 3, 4, 19 and 20", "240 of 240, 956 of 960 and 1,915 of 1,920 (100.0, 99.6 and 99.7 %)",
              "I do not know whether a 7B model uses a terse feedback", "lines 2, 3 and 4 all yes about 75 %; lines 9 and 11 both yes about 30 %; the pattern lines 8 and 11 NO and the rest yes about 12 %; all 12 yes about 6 %", "never a multiple of capacity, a count of GPUs saved, dollars, \"100 percent\" or \"proven\"",
              "a TPU is another engine build and would be a different test")
FRONT_ROW = ("**Run (the owner's Colab run, five cells in one session, Tesla T4, vLLM 0.31.0; the file reached the repository work on 20 Oct 2026).**", "74243e1ab623ec5c… of the locked text", "**The headline reads NO:**", "48 of the 53 lost steps belong to four jobs (3, 4, 19 and 20)", "never on the 507 deletes the repair completes",
             "84 of 1,599 delete replies in the first run (no grammar) and 0 in the second and third", "The files it imports are pinned by sha256 in the test, so the core is unchanged.", "the ceiling at 64 jobs is 1,915 of 1,920 (99.7 %) against 1,867 (97.2 %)", "(test 128, 23 tests)")
FRONT_NEXT = "Next free: row 170, item 114, test 136, question 121"


def test_the_front_pages_point_to_the_test_and_say_what_the_page_expects_and_what_it_does_not_show():
    cm = open(os.path.join(ROOT, "docs", "RAJNISH_CONCEPT_MAP.md")).read(); ve = open(os.path.join(ROOT, "docs", "VERIFIED.md")).read()
    assert "PROVIDER_SIDE_RETRY_TEST.md" in cm and "PROVIDER_SIDE_RETRY_TEST.md" in ve
    item = cm[cm.index("109. **"): cm.index("110. **")]; k = ve.index("Provider-side serving test, fourth run"); row = ve[k: ve.index("\n", k)]
    assert item.count("\n109. **") == 0 and "\n110. **" not in item
    for s in FRONT_ITEM: assert s in item, s[:100]
    for s in FRONT_ROW: assert s in row, s[:100]
    for text in (item, row):
        assert "@@" not in text and "mutation check of" in text.lower() and "483 deliberate changes" in text and "467 caught" in text and "16 " in text and "capacity" in text and "NOT RUN yet" not in text and "Result" in text
        assert "proven" in text and "dollars" in text and "energy" in text
    assert FRONT_NEXT in cm and "test 128 checks the retry test and test 129 its Result, item 109; test 130" in cm and "test 126 checks the capped re-run of the exact-state cells and test 127 its Result, item 108; test 128" in cm
    assert cm.index("108. **") < cm.index("109. **") < cm.index("Next free: row") and ve.index("Provider-side serving test, third run") < k
    for name in ("test_provider_side.py", "test_provider_side_constrained.py", "test_provider_side_bounded.py", "test_provider_side_bounded_result.py"):
        t = open(os.path.join(HERE, name)).read(); assert "row 170, item 112, test 133" not in t and "row 170, item 114, test 136" in t, name


if __name__ == "__main__":
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]; bad = 0
    for n, f in fns:
        try: f(); print("ok  ", n)
        except Exception: bad += 1; print("FAIL", n); traceback.print_exc()
    print(f"{len(fns) - bad}/{len(fns)} passed"); sys.exit(1 if bad else 0)
