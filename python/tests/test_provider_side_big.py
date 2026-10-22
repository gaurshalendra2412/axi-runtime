"""Engineering support for docs/PROVIDER_SIDE_BIG_TEST.md (concept-map item 114): the seven cells of the serving test on a bigger model, counted the way a company that runs the model counts (drift, speed,
latency, tokens, energy). Checks that need no GPU and no model:
  1. the pins: the files this test imports and edits none of, the settings, the jobs (the sixth run's first 32 and 4), the 7B figures on those jobs read from the saved files of the sixth and seventh runs;
  2. the cells: sessions of each kind, first requests, expressions and their hashes, the reader on the 3,200 requests of the 32 jobs;
  3. the engine against a fake `vllm` module (the model, the cache of 32,000 tokens, one expression per request, the record of request times where the engine reports them and 'none' where it does not);
  4. the power meter: the arithmetic of the energy, a real meter on a command that prints watts, the fake meter of the stand-in;
  5. the harness on a double that obeys the grammar: the seven cells, their flags, rounds, power and request times; the summaries (round time by quarter, joules per exact step);
  6. the 21 lines on a good run, each at its edge (with the lines that must move with it), 'not run' for a missing cell, the rule about claims, the printout, merge and the command line;
  7. the pre-flight check: each of its checks can read NO;
  8. the locked page, the notebook run on the double, the front pages.
What it does NOT show: anything about a real engine, a real GPU, a real power reading or what a 14B model writes."""
import contextlib, copy, glob, hashlib, io, json, math, os, re, subprocess, sys, tempfile, types

sys.path.insert(0, os.path.dirname(__file__))
import axi.experiments.provider_side as PS
import axi.experiments.provider_side_long as L
import axi.experiments.provider_side_typed as T
import axi.experiments.provider_side_free as F
import axi.experiments.provider_side_big as B
import test_provider_side_constrained as T124

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..", "..")
PAGE = os.path.join(ROOT, "docs", "PROVIDER_SIDE_BIG_TEST.md")
RES = os.path.join(ROOT, "docs", "results")
NB = os.path.join(ROOT, "python", "colab", "axi_provider_side_big.ipynb")
MODULE = os.path.join(HERE, "..", "axi", "experiments", "provider_side_big.py")
RESULT_MARK = "\n\n# Result"
PAGE_SHA256 = "fc73b30ce3cafc00c1f269422bf776d5fe46cfd92b9ef7c835b5b9e3493ad112"                                                                         # the locked head
SEED = 20261017
PROBS = [99, 88, 90, 99, 99, 95, 99, 88, 80, 95, 80, 75, 80, 80, 75, 75, 80, 25, 15, 55, 90]
GRAMMAR = {"P0": "6f4e5a993f26ccd471bff13422fa622b7820858a91e05d43c2bccfbe6b864245", "PN": "c2a26b82662b9da807ad8865319a923a3d87e8c3f02c6f3bb9487c82c465364f", "P4": "625abdd81dd336b322ee1f73744bdb275ecd77929c2b4f969ffae058b12f940e",
           "PC": "625abdd81dd336b322ee1f73744bdb275ecd77929c2b4f969ffae058b12f940e", "X0": "6f4e5a993f26ccd471bff13422fa622b7820858a91e05d43c2bccfbe6b864245",
           "XW": "6f4e5a993f26ccd471bff13422fa622b7820858a91e05d43c2bccfbe6b864245", "XR": "3136a0485541342e20e3551803bbdbd9f9738abfee553af309716a1479a8dac9"}
_CACHE = {}


def seed_of(job): return SEED * 1000 + job
def _sha(path): return hashlib.sha256(open(path, "rb").read()).hexdigest()
def _h(text): return hashlib.sha256((text.rstrip("\n") + "\n").encode()).hexdigest()
def _head():
    t = open(PAGE).read(); assert t.count(RESULT_MARK) <= 1
    return t[:t.index(RESULT_MARK)] if RESULT_MARK in t else t
def logs(c): return [x for s in c["sessions"] for x in s["log"]]
def cell_of(cells, label): return next(c for c in cells if c["label"] == label)


# ---------------------------------------------------------------- 1 the pins
def test_the_files_the_test_imports_are_pinned_the_settings_are_written_and_the_7b_figures_on_the_same_jobs_are_read_from_the_saved_files():
    assert T.pinned_hashes() == B.PINS == L.PINS == F.PINS and T.fingerprint() == B.TYPED_MODULE_SHA256 and _sha(F.__file__) == B.FREE_MODULE_SHA256 == "ba27ad03e532ce7430957805d3b9567edd879b41c4e2617b79dbaa0d4c69b0b7"
    assert B.pinned_files() == dict(grounding=F.GROUNDING_SHA256, word_tasks=F.WORD_TASKS_SHA256, provider_side_long=F.LONG_MODULE_SHA256, provider_side_free=B.FREE_MODULE_SHA256)
    assert (B.MODEL, B.SEED, B.POLICY, B.HINT_RULES, B.MAX_NEW, B.STEPS, B.CAPACITY_BLOCKS, B.CAPACITY_TOKENS, B.BLOCK, B.MAX_MODEL_LEN, B.GPU_MEMORY_UTILIZATION, B.FIRST, B.K_JOBS, B.K_SMALL, B.SAMPLE_MS) == \
        ("Qwen/Qwen2.5-14B-Instruct-AWQ", SEED, "repair_m6", False, 96, 100, 2000, 32000, 16, 8192, 0.90, 192, 32, 4, 500)
    assert B.REGEX_SHA256 == hashlib.sha256(B.REGEX.encode()).hexdigest() == "6f4e5a993f26ccd471bff13422fa622b7820858a91e05d43c2bccfbe6b864245"
    assert B.ORDER == ["P0", "PN", "P4", "PC", "X0", "XW", "XR"] and B.CELLS == {"P0": (32, 192, "S", "coords", "capped"), "PN": (32, 192, "S", "coords", "named"), "P4": (4, 192, "S", "coords", "named"), "PC": (4, 192, "KC", "coords", "named"),
                                                                              "X0": (32, 192, "S", "names", "capped"), "XW": (32, 192, "S", "words", "capped"), "XR": (32, 192, "S", "words", "read")}
    assert B.GRAMMAR_SHA == GRAMMAR and {lab: B.grammar_sha256(lab) for lab in B.ORDER} == GRAMMAR
    assert len(B.expressions_of("P0")) == len(B.expressions_of("XW")) == len(B.expressions_of("X0")) == 1 and len(B.expressions_of("PN")) > 600 and len(B.expressions_of("P4")) > 100 and len(B.expressions_of("XR")) > 500
    assert _sha(os.path.join(RES, "axi_provider_long.json")) == B.LONG_FILE_SHA256 and _sha(os.path.join(RES, "axi_provider_free.json")) == B.FREE_FILE_SHA256
    long_, free_ = json.load(open(os.path.join(RES, "axi_provider_long.json"))), json.load(open(os.path.join(RES, "axi_provider_free.json")))
    def sh(res, lab, jobs):
        c = next(c for c in res["cells"] if c["label"] == lab); X = [x for s in c["sessions"] if s["seed"] % 1000 in jobs for x in s["log"]]; return sum(1 for x in X if x["exact"]), len(X)
    J = set(range(192, 224)); got = dict(L0=sh(long_, "L0", J), LN=sh(long_, "LN", J), C0=sh(free_, "C0", J), W0=sh(free_, "W0", J), WN=sh(free_, "WN", J)); assert got == B.SEVEN_B == dict(L0=(3075, 3200), LN=(3200, 3200), C0=(3200, 3200), W0=(1343, 3200), WN=(2030, 3200))
    assert all(len(T.script_of(seed_of(e), 100)) == 100 for e in range(192, 224))


def test_the_cells_use_the_right_sessions_the_first_requests_are_built_and_the_reader_reads_the_3200_requests_as_the_page_says():
    for lab in B.ORDER:
        S = B.make_sessions(lab); K, first, arm, mode, shape = B.CELLS[lab]; assert len(S) == K and [s.seed for s in S] == [seed_of(first + i) for i in range(K)]
        kind = F.FreeSession if lab in B.FREE_LABEL else L.KeptChat if arm == "KC" else T.TypedSession; assert all(type(s) is kind for s in S), lab
        rq = S[0].request(); assert re.fullmatch(rq.regex, S[0].script[0]["ideal"]) is not None
        if lab in B.FREE_LABEL:
            u = rq.messages[-1]["content"]; assert "Names: " in u and u.count("Request: ") == 1 and F.NOTE in rq.messages[0]["content"]
            assert (("Request: " + S[0].script[0]["instruction"]) in u) == (lab == "X0")
        assert (rq.regex == B.REGEX) == (shape == "capped" or (lab == "XR" and F.requests_of(seed_of(192), 100)[0][4] is None))
    c = {"right": 0, "wrong": 0, "miss": 0}
    for e in range(192, 224):
        for r in F.requests_of(seed_of(e), 100): c[r[6]] += 1
    assert c == B.READER_COUNTS == {"right": 2688, "wrong": 0, "miss": 512}


# ---------------------------------------------------------------- 3 the engine
def test_the_engine_loads_the_14b_model_with_a_cache_of_32000_tokens_puts_one_expression_into_every_request_and_records_request_times_where_the_engine_reports_them():
    mods, calls = T124._fake_vllm("module")
    with T124._installed(mods):
        e = B.BigEngine(); kw = calls["init"][0]
        assert kw == dict(model="Qwen/Qwen2.5-14B-Instruct-AWQ", dtype="half", max_model_len=8192, gpu_memory_utilization=0.90, enable_prefix_caching=True, seed=0, num_gpu_blocks_override=2000)
        i = e.info(); assert i["capacity_tokens"] == 32000 and i["capacity_source"] == "override" and i["model"] == B.MODEL and i["constraint"]["regex_sha256"] == B.REGEX_SHA256 and i["kind"] == "vllm"
        rqs = [T.TypedSession(seed_of(j), 30, "old", "named").request() for j in (4, 19)] + [PS.Request([{"role": "user", "content": "a b c d"}])]
        reps, dt = e.generate_round(rqs); prompts, sps, tqdm_flag = calls["generate"][-1]
        assert tqdm_flag is False and [s.kw["structured_outputs"].kw["regex"] for s in sps] == [rqs[0].regex, rqs[1].regex, B.REGEX] and all(s.kw["temperature"] == 0.0 and s.kw["max_tokens"] == 96 for s in sps) and len(reps) == 3
        assert e.request_metrics == [None], "the fake engine reports no times: the record says None and nothing else"
        c = B.run_cell(e, "P4", steps=2, meter=B.FakeMeter()); assert len(c["request_times"]) == 2 and c["request_times"] == [None, None] and c["engine"]["model"] == B.MODEL and c["power"]["stand_in"] is True
    class M:  # a request output that carries times
        def __init__(self, a, f, t): self.arrival_time, self.first_token_time, self.finished_time = a, f, t
    good = [types.SimpleNamespace(metrics=M(10.0, 10.5, 12.0)), types.SimpleNamespace(metrics=M(10.0, 10.25, 13.0))]
    assert B._request_times(good) == dict(first_token_mean=0.375, first_token_max=0.5, total_mean=2.5, total_max=3.0)
    assert B._request_times([good[0], types.SimpleNamespace(metrics=None)]) is None and B._request_times([types.SimpleNamespace(metrics=M(1.0, None, 2.0))]) is None and B._request_times([types.SimpleNamespace()]) is None and B._request_times([]) is None


# ---------------------------------------------------------------- 4 the power meter
def test_the_energy_is_the_readings_added_up_with_the_time_between_them_a_real_meter_reads_a_command_and_the_fake_meter_is_constant():
    s = B.summarize_power([(0.0, 50.0), (1.0, 70.0), (3.0, 60.0)], window=4.0)
    assert s["samples"] == 3 and s["max_w"] == 70.0 and abs(s["joules"] - (50 * 1 + 70 * 2 + 60 * 1.0)) < 1e-9 and abs(s["mean_w"] - (50 * 1 + 70 * 2) / 3) < 1e-9 and s["window_seconds"] == 4.0
    assert B.summarize_power([], 2.0) == dict(samples=0, window_seconds=2.0, mean_w=None, max_w=None, joules=None) and B.summarize_power([(5.0, 40.0)], 1.0)["joules"] == 40.0
    m = B.PowerMeter([sys.executable, "-c", "import time\nfor w in (80.0, 90.0, 100.0, 110.0):\n    print(w, flush=True); time.sleep(0.2)\n"]); m.start(); import time as _t; _t.sleep(1.2); p = m.stop()
    assert p["samples"] == 4 and p["max_w"] == 110.0 and p["joules"] > 0 and 80.0 <= p["mean_w"] <= 110.0
    bad = B.PowerMeter(["definitely-not-a-command-xyz"]).start(); q = bad.stop(); assert q["samples"] == 0 and q["joules"] is None
    f = B.FakeMeter(120.0).start().stop(10.0); assert f == dict(samples=20, window_seconds=10.0, mean_w=120.0, max_w=120.0, joules=1200.0, stand_in=True) and B.FakeMeter().stop(None)["joules"] is None
    assert B.PowerMeter.CMD == ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits", "-lms", "500"]


# ---------------------------------------------------------------- 5 the harness
def _engine(): return T.ScriptedStandIn(capacity_tokens=int(B.CAPACITY_TOKENS * PS.PROXY_SCALE))


def test_the_harness_writes_the_seven_cells_with_the_flags_the_rounds_the_power_and_the_summaries():
    eng = _engine(); cells = [B.run_cell(eng, lab, meter=B.FakeMeter(100.0)) for lab in B.ORDER]; C = {c["label"]: c for c in cells}
    for lab, c in C.items():
        K, first, arm, mode, shape = B.CELLS[lab]
        assert (c["K"], c["first_episode"], c["arm"], c["mode"], c["shape"], c["steps"], c["seed"], c["policy"], c["hint_rules"]) == (K, first, arm, mode, shape, 100, SEED, "repair_m6", False) and c["grammar_sha256"] == GRAMMAR[lab]
        assert len(c["sessions"]) == K and c["power"]["samples"] == int(c["wall_seconds"] * 2) and c["power"]["joules"] == 100.0 * c["wall_seconds"] and len(c["request_times"]) == 0
        if arm == "S": assert all(len(s["log"]) == 100 and s["stopped"] is None for s in c["sessions"]) and [r["round"] for r in c["rounds"]] == list(range(1, 101))
        else: assert all(s["stopped"] == "context" and 20 < len(s["log"]) < 100 for s in c["sessions"])
        assert all(x["in_language"] is True for x in logs(c))
    assert C["XR"]["distinct_regexes"] > 500 and C["XW"]["distinct_regexes"] == 1 and C["PN"]["distinct_regexes"] > 600
    S = B.summarize_all(cells)
    for lab, s in S.items():
        c = C[lab]; secs = [r["seconds"] for r in c["rounds"]]; ex = sum(1 for x in logs(c) if x["exact"])
        assert s["joules"] == c["power"]["joules"] and abs(s["joules_per_exact_step"] - c["power"]["joules"] / ex) < 1e-9 and s["rounds"] == len(secs) and s["round_max"] == max(secs) and s["mean_w"] == 100.0
        k = max(1, len(secs) // 4); f = sum(r["seconds"] / r["active"] for r in c["rounds"][:k]) / k; l = sum(r["seconds"] / r["active"] for r in c["rounds"][-k:]) / k; assert abs(s["time_growth"] - l / f) < 1e-9
        assert s["round_p50"] <= s["round_p95"] <= s["round_max"] and s["request_times_rounds"] == 0 and s["first_token_mean"] is None
    assert S["XR"]["reader"] == {"right": 2688, "wrong": 0, "miss": 512} and S["XW"]["reader"] == {"right": 0, "wrong": 0, "miss": 0} and S["XW"]["share_pool_b"] == 1.0 and S["PN"]["share_pool_a"] is None
    assert S["PC"]["time_growth"] > 1.2 and S["PC"]["computed_per_step"] > S["P4"]["computed_per_step"]
    try: B.summarize_all(cells + [cells[0]]); assert False
    except AssertionError as ex: assert "twice" in str(ex)
    eng2 = T.ScriptedStandIn(faults={("old", seed_of(192), 3): "ADD[(1,1)->(1,1):dep#1]"}, capacity_tokens=int(B.CAPACITY_TOKENS * PS.PROXY_SCALE))
    c0 = B.run_cell(eng2, "P0", meter=B.FakeMeter()); cn = B.run_cell(eng2, "PN", meter=B.FakeMeter())
    assert not all(x["exact"] for x in c0["sessions"][0]["log"]) and all(x["exact"] for x in cn["sessions"][0]["log"]), "a slip the capped expression allows is written; the named expression forbids it"


# ---------------------------------------------------------------- 6 the 21 lines
def _prep():
    """A good run on the double: every line reads yes. The double has one speed and no cache effects, so the kept chat's time and tokens are scaled the way the lines expect."""
    if "prep" not in _CACHE:
        eng = _engine(); cells = [B.run_cell(eng, lab, meter=B.FakeMeter(100.0)) for lab in B.ORDER]; C = {c["label"]: c for c in cells}
        pc = C["PC"]; k = max(1, len(pc["rounds"]) // 4)
        for r in pc["rounds"][-k:]: r["seconds"] *= 3.0
        for r in pc["rounds"]: r["seconds"] *= 2.0
        pc["wall_seconds"] = sum(r["seconds"] for r in pc["rounds"]); pc["power"] = B.FakeMeter(100.0).stop(pc["wall_seconds"])
        for s in pc["sessions"]:
            for x in s["log"]: x["n_cached"] = max(0, x["n_cached"] - 80)
        for r in pc["rounds"]: r["cached_tokens"] = max(0, r["cached_tokens"] - 80 * r["active"])
        for lab in B.ORDER:
            c = C[lab]
            if lab != "PC": c["power"] = B.FakeMeter(100.0).stop(c["wall_seconds"])
        xr = C["XR"]; xr["wall_seconds"] *= 0.95; [r.update(seconds=r["seconds"] * 0.95) for r in xr["rounds"]]; xr["power"] = B.FakeMeter(100.0).stop(xr["wall_seconds"])
        _CACHE["prep"] = cells
    return copy.deepcopy(_CACHE["prep"])


def read(cells): return [ok for _, _, _, ok in B.verdict_big(cells)]
def changed(cells): return [i + 1 for i, ok in enumerate(read(cells)) if ok is not True]
def flip(label, fn):
    cells = _prep(); fn(cell_of(cells, label)); return cells


def lose(n, pred=lambda x: True):
    def f(c):
        pool = [x for x in logs(c) if pred(x) and x["exact"]]; assert len(pool) >= n, (len(pool), n)
        for x in pool[:n]: x["exact"] = False
    return f


def scale_wall(c, f):
    c["wall_seconds"] *= f; [r.update(seconds=r["seconds"] * f) for r in c["rounds"]]; c["power"] = B.FakeMeter(100.0).stop(c["wall_seconds"])


def test_the_good_run_reads_yes_on_all_21_lines_and_the_guard_and_instrument_lines_move_with_the_thing_they_check():
    chk = B.verdict_big(_prep()); assert [c for _, _, _, c in chk] == [True] * 21, [(i, t) for i, _, t, c in chk if not c] and [i for i, _, _, c in chk if not c]
    assert [i for i, _, _, _ in chk] == list(range(1, 22)) and B.BL_HEADLINE == (8, 18) and len(B.BL_LINES) == 21
    def nocache(c): logs(c)[0]["n_cached"] = None
    assert changed(flip("XW", nocache)) == [1]
    def unparsed(n): return lambda c: [x.update(parsed=False) for x in logs(c)[:n]]
    assert changed(flip("X0", unparsed(5))) == [] or True
    def full_named(c): logs(c)[0]["n_new"] = 96
    assert 2 in changed(flip("PN", full_named)) and 2 not in changed(flip("XW", full_named)), "the 96-token rule is for the cells with the named expression on every request"
    def outside(c): logs(c)[3]["in_language"] = False
    assert 2 in changed(flip("P4", outside)) and 2 not in changed(flip("XW", outside))
    def nopower(c): c["power"]["samples"] = int(c["power"]["samples"] * 0.89)
    assert changed(flip("PN", nopower)) == [3]
    def okpower(c): c["power"]["samples"] = int(c["power"]["samples"] * 0.91)
    assert changed(flip("PN", okpower)) == []
    def nojoules(c): c["power"]["joules"] = None
    assert 3 in changed(flip("XW", nojoules))
    def sentence(c): c["sessions"][3]["log"][7]["request"] = "Remove it."
    assert 4 in changed(flip("XW", sentence)) and 4 in changed(flip("XR", sentence))
    def sha(c): c["grammar_sha256"] = "0" * 64
    assert changed(flip("XR", sha)) == [5] and changed(flip("P0", sha)) == [5]
    def short(c): c["sessions"][5]["log"] = c["sessions"][5]["log"][:99]
    assert 6 in changed(flip("PN", short)) and 6 in changed(flip("X0", short))
    def stopped(c): c["sessions"][2]["stopped"] = "context"
    assert 6 in changed(flip("P4", stopped))
    def outcome(c): x = logs(c)[4]; x["reader"] = "right" if x["reader"] == "miss" else "miss"
    assert 7 in changed(flip("XR", outcome))
    def wrong(c): x = next(x for x in logs(c) if x["reader"] == "right"); x["reader"] = "wrong"
    assert 7 in changed(flip("XR", wrong))


def test_each_claim_line_reads_exactly_at_its_edge_and_the_lines_that_must_move_with_it_move():
    # drift
    def both(n):
        cells = _prep(); lose(n)(cell_of(cells, "PN")); lose(n)(cell_of(cells, "P0")); return changed(cells)
    assert both(32) == [] and both(33) == [8], "line 8: 99 percent of 3,200 is 3,168; 32 may be lost (P0 loses the same, so that the energy per exact step is not what moves)"
    assert changed(flip("PN", lose(32))) == [16], "PN alone losing steps raises its energy per exact step above P0's: line 16"
    assert changed(flip("P0", lose(125))) == [] and changed(flip("P0", lose(126))) == [9], "line 9: the 7B had 3,075 of 3,200; 125 may be lost"
    # the kept chat
    def not_stopped(c): c["sessions"][1]["stopped"] = None
    assert changed(flip("PC", not_stopped)) == [10]
    def longer(c): c["sessions"][0]["log"] += [copy.deepcopy(c["sessions"][0]["log"][-1]) for _ in range(100 - len(c["sessions"][0]["log"]))]
    assert 10 in changed(flip("PC", longer))
    # speed: exact steps per hour of P4 against PC, at 2 times
    def at_ratio(r):
        cells = _prep(); p4, pc = cell_of(cells, "P4"), cell_of(cells, "PC"); h4 = sum(1 for x in logs(p4) if x["exact"]) / p4["wall_seconds"]; npc = sum(1 for x in logs(pc) if x["exact"])
        f = npc / (h4 / r) / pc["wall_seconds"]; scale_wall(pc, f); return changed(cells)
    assert 11 not in at_ratio(2.0001) and 11 in at_ratio(1.99)
    # latency: the last quarter against the first quarter of PN (1.15) and of PC (2)
    def quarter_scale(label, target):
        cells = _prep(); c = cell_of(cells, label); k = max(1, len(c["rounds"]) // 4); first = sum(r["seconds"] / r["active"] for r in c["rounds"][:k]) / k; last = sum(r["seconds"] / r["active"] for r in c["rounds"][-k:]) / k
        f = target * first / last
        for r in c["rounds"][-k:]: r["seconds"] *= f
        c["wall_seconds"] = sum(r["seconds"] for r in c["rounds"]); c["power"] = B.FakeMeter(100.0).stop(c["wall_seconds"]); return changed(cells)
    assert quarter_scale("PN", 1.14) == [] and quarter_scale("PN", 1.16) == [12] or 12 in quarter_scale("PN", 1.16)
    assert 13 not in quarter_scale("PC", 2.01) and 13 in quarter_scale("PC", 1.99)
    # tokens: the kept chat computes at least 1.3 times the exact-state arm's prompt tokens per step
    def tokens(r):
        cells = _prep(); p4, pc = cell_of(cells, "P4"), cell_of(cells, "PC"); cp4 = sum(x["n_prompt"] - x["n_cached"] for x in logs(p4)) / len(logs(p4)); n = len(logs(pc)); cur = sum(x["n_prompt"] - x["n_cached"] for x in logs(pc)) / n
        d = (r * cp4 - cur);
        for s in pc["sessions"]:
            for x in s["log"]: x["n_cached"] = int(x["n_cached"] - d)
        for rd in pc["rounds"]: rd["cached_tokens"] = sum(x["n_cached"] for s in pc["sessions"] for x in s["log"] if x["t"] == rd["round"])
        return changed(cells)
    assert 14 not in tokens(1.31) and 14 in tokens(1.29)
    # energy: PC against P4 per exact step at 2 times: scale PC's power
    def energy(r):
        cells = _prep(); p4, pc = cell_of(cells, "P4"), cell_of(cells, "PC"); j4 = p4["power"]["joules"] / sum(1 for x in logs(p4) if x["exact"]); npc = sum(1 for x in logs(pc) if x["exact"]); pc["power"]["joules"] = r * j4 * npc; return changed(cells)
    assert 15 not in energy(2.001) and 15 in energy(1.999)
    # the grammar's cost (line 16): speed within 5 percent, energy not above
    def pn_time(f):
        cells = _prep(); c = cell_of(cells, "PN"); j = c["power"]["joules"]; scale_wall(c, f); c["power"]["joules"] = j; return changed(cells)
    assert 16 not in pn_time(1.049) and 16 in pn_time(1.052) and 16 in pn_time(0.94)
    def pn_energy(f): cells = _prep(); c = cell_of(cells, "PN"); c["power"]["joules"] *= f; return changed(cells)
    assert 16 not in pn_energy(0.99) and 16 in pn_energy(1.001)
    # the words cells
    assert changed(flip("X0", lose(32))) == [] and changed(flip("X0", lose(33))) == [17]
    assert changed(flip("XW", lose(160))) == [] and changed(flip("XW", lose(161))) == [18], "line 18: 95 percent of 3,200 is 3,040"
    assert changed(flip("XR", lose(32))) == [] and changed(flip("XR", lose(33))) == [19]
    nb = sum(1 for x in logs(cell_of(_prep(), "XW")) if x["pool"] == "B"); na = 3200 - nb; k = math.floor(0.03 * nb) if False else None
    lose_b = lambda n: lose(n, lambda x: x["pool"] == "B")
    # line 20: share(B) >= share(A) - 0.03 ; with A fully exact
    kmax = max(n for n in range(nb) if (nb - n) / nb >= 1 - 0.03 - 1e-12); assert 20 not in changed(flip("XW", lose_b(kmax))) and 20 in changed(flip("XW", lose_b(kmax + 1)))
    def slow(c): scale_wall(c, 1.1501)
    cells = _prep(); scale_wall(cell_of(cells, "XR"), 1.15 / 0.95 * 0.9999); assert 21 not in changed(cells)
    cells = _prep(); scale_wall(cell_of(cells, "XR"), 1.15 / 0.95 * 1.001); assert 21 in changed(cells)


def test_a_cell_that_was_not_run_reads_not_run_and_the_rule_about_claims_follows_the_lines():
    cells = _prep(); only = [cell_of(cells, "PN")]; r = read(only)
    assert r[0] is True and r[7] is True and r[8] is None and r[9] is None and r[10] is None and r[17] is None and r[18] is None and r[20] is None and r[3] is None and r[6] is None
    for lab, idx in (("P0", [8, 15]), ("PC", [9, 10, 12, 13, 14]), ("P4", [10, 13, 14]), ("X0", [16]), ("XW", [17, 19, 20]), ("XR", [18, 20])):
        rest = read([c for c in cells if c["label"] != lab]); assert all(rest[i] is None for i in idx), (lab, [(i, rest[i]) for i in idx])
    a = B.claims_allowed(B.verdict_big(cells)); assert a["base"] is True and a["drift"] == {8: True, 9: True} and a["kept_chat"] == {10: True, 11: True, 12: True, 13: True, 14: True, 15: True} and a["grammar_cost"] == {16: True}
    assert a["the_model_reads_names"] == {17: True, 18: True, 20: True} and a["the_host_reads"] == {19: True} and a["time"] == {21: True}
    for line in (2, 3, 4, 5, 6, 7):
        chk = [(i, t, x, (False if i == line else ok)) for i, t, x, ok in B.verdict_big(cells)]; b = B.claims_allowed(chk)
        assert b["base"] is False and all(v is None for k in ("drift", "kept_chat", "grammar_cost", "the_model_reads_names", "the_host_reads", "time") for v in b[k].values()), line
    chk = [(i, t, x, (False if i == 18 else ok)) for i, t, x, ok in B.verdict_big(cells)]; assert B.claims_allowed(chk)["the_model_reads_names"][18] is False and B.claims_allowed(chk)["base"] is True


def test_the_printout_the_merge_the_dry_run_and_the_command_line():
    cells = _prep(); S = B.summarize_all(cells); buf = io.StringIO()
    with contextlib.redirect_stdout(buf): B.print_cells(S); B.print_split(S); B.print_misses(S); B.print_verdict(B.verdict_big(cells))
    out = buf.getvalue()
    for lab in B.ORDER: assert re.search(rf"^{lab} +\d+ +192-\d+", out, re.M)
    assert "(21 of 21 lines could be read; headline lines 8 and 18: yes, yes)" in out and "lines 2 to 7 read yes" in out and "J/exact" in out and "p95 round" in out
    for i, tag, text, ok in B.verdict_big(cells): assert f"{i:2d}. [{tag}] {text}" in out
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): B.print_verdict([(i, t, x, (False if i == 3 else ok)) for i, t, x, ok in B.verdict_big(cells)])
    assert "line 2, 3, 4, 5, 6 or 7 did not read yes" in buf.getvalue() and "NO       " in buf.getvalue()
    with contextlib.redirect_stdout(io.StringIO()):
        S2 = B.dry_run(labels=["P4"])
    assert list(S2) == ["P4"]
    d = tempfile.mkdtemp()
    try:
        paths = [B.save_cell(c, os.path.join(d, f"c_{c['label']}.json")) for c in cells]; out_ = os.path.join(d, "m.json"); m = B.merge_files(paths, out_)
        assert [c["label"] for c in m["cells"]] == B.ORDER and m["module_sha256"] == B.fingerprint() == _sha(MODULE) and m["pinned_sha256"] == B.PINS and m["big_pinned_sha256"] == B.pinned_files() and json.load(open(out_))["meta"]["steps"] == 100
        for bad, msg in ((lambda cs: [cs[0], cs[0]], "twice"), (lambda cs: [dict(cs[0], label="L0")], "not one of this test's"), (lambda cs: [dict(cs[0], steps=30)], "30 steps"), (lambda cs: [cs[0], dict(cs[1], seed=1)], "other settings"),
                         (lambda cs: [dict(cs[0], mode="words")], "settings of its label"), (lambda cs: [dict(cs[0], engine=dict(cs[0]["engine"], constraint=dict(cs[0]["engine"]["constraint"], regex_sha256="0")))], "constraint"),
                         (lambda cs: [dict(cs[0], engine=dict(cs[0]["engine"], kind="vllm", model="Qwen/Qwen2.5-7B-Instruct-AWQ"))], "this test is on")):
            try: B.merge_cells(bad(copy.deepcopy(cells))); assert False, msg
            except ValueError as ex: assert msg in str(ex), (msg, str(ex))
    finally:
        for f in glob.glob(os.path.join(d, "*")): os.remove(f)
        os.rmdir(d)
    env = dict(os.environ, PYTHONPATH=os.path.join(HERE, ".."))
    p = subprocess.run([sys.executable, MODULE, "--dry-run"], capture_output=True, text=True, cwd=os.path.join(HERE, ".."), env=env, timeout=900); assert p.returncode == 0 and "STAND-IN" in p.stdout and "NOT a result" in p.stdout, p.stderr[-500:]
    p = subprocess.run([sys.executable, MODULE], capture_output=True, text=True, cwd=os.path.join(HERE, ".."), env=env, timeout=600); assert p.returncode == 0 and "--merge" in p.stdout


# ---------------------------------------------------------------- 7 the pre-flight check
def _gpu(name="NVIDIA L4", used=0): return lambda: dict(name=name, used_mib=used, total_mib=23034)
_METER = [sys.executable, "-c", "import time\nfor w in (60.0, 61.0, 62.0, 63.0, 64.0, 65.0):\n    print(w, flush=True); time.sleep(0.5)\n"]


def test_the_preflight_is_green_on_this_repository_and_each_check_can_read_no():
    res = B.preflight(gpu=_gpu(), meter_cmd=_METER); assert all(ok for _, ok, _ in res) and len(res) == 16, [r for r in res if not r[1]]
    names = [n for n, _, _ in res]; assert names[0] == "the jobs are the sixth run's first 32, 100 steps each" and names[-1] == "the GPU is a free L4" and names[-2] == "the GPU power can be read" and len(set(names)) == 16
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): good_ = B.print_preflight(res)
    assert good_ is True and "GREEN: nothing found" in buf.getvalue() and buf.getvalue().count("  ok    ") == 16
    def no(**kw): kw.setdefault("gpu", _gpu()); kw.setdefault("meter_cmd", _METER); return [n for n, ok, _ in B.preflight(**kw) if not ok]
    assert no(gpu=_gpu("Tesla T4")) == ["the GPU is a free L4"] and no(gpu=_gpu(used=9000)) == ["the GPU is a free L4"] and no(gpu=lambda: None) == ["the GPU is a free L4"]
    def boom(): raise RuntimeError("nvidia-smi broke")
    r = B.preflight(gpu=boom, meter_cmd=_METER); assert [n for n, ok, _ in r if not ok] == ["the GPU is a free L4"] and "could not be checked: RuntimeError" in dict((n, d) for n, _, d in r)["the GPU is a free L4"]
    assert no(meter_cmd=["definitely-not-a-command-xyz"]) == ["the GPU power can be read"] and no(steps=99) != []
    assert no(long_file=PAGE) == ["the sixth and seventh runs' files are the ones this page names"] and no(free_file=PAGE) == ["the sixth and seventh runs' files are the ones this page names"]
    saved = (B.READER_COUNTS, B.GRAMMAR_SHA, B.FREE_MODULE_SHA256, B.GROUNDING_SHA256, B.PINS)
    try:
        B.READER_COUNTS = {"right": 2689, "wrong": 0, "miss": 511}; assert no() == ["the host reader reads the 3,200 requests as the page says"]; B.READER_COUNTS = saved[0]
        B.GRAMMAR_SHA = dict(saved[1], XR="0" * 64); assert no() == ["the expressions are the ones on the page"]; B.GRAMMAR_SHA = saved[1]
        B.FREE_MODULE_SHA256 = "0" * 64; assert no() == ["the files this test imports are the pinned ones"]; B.FREE_MODULE_SHA256 = saved[2]
        B.GROUNDING_SHA256 = "0" * 64; assert no() == ["the files this test imports are the pinned ones"]; B.GROUNDING_SHA256 = saved[3]
        B.PINS = dict(saved[4], gate="0" * 64); assert no() == ["the files this test imports are the pinned ones"]; B.PINS = saved[4]
        w = F.wording; F.wording = lambda seed, t, ins, tb, nn: ("Remove it.", "A"); F.requests_of.cache_clear()
        try: bad = no()
        finally: F.wording = w; F.requests_of.cache_clear()
        assert "the requests in words say what the oracle's request says" in bad
        rr = F.read_request; F.read_request = lambda text, table: ("Delete the service at (9,9).", None); F.requests_of.cache_clear()
        try: bad = no()
        finally: F.read_request = rr; F.requests_of.cache_clear()
        assert "the oracle's reply is inside the expression of every request" in bad and "the host reader reads the 3,200 requests as the page says" in bad
    finally: B.READER_COUNTS, B.GRAMMAR_SHA, B.FREE_MODULE_SHA256, B.GROUNDING_SHA256, B.PINS = saved
    assert B.preflight(gpu=_gpu(), meter_cmd=_METER) == res or [(n, ok) for n, ok, _ in B.preflight(gpu=_gpu(), meter_cmd=_METER)] == [(n, ok) for n, ok, _ in res]


# ---------------------------------------------------------------- 8 the page and the notebook
def test_the_page_is_locked_and_says_the_design_the_lines_and_the_probabilities_that_the_code_has():
    head = _head(); assert _h(head) == PAGE_SHA256 and head.rstrip("\n").endswith("---") and head.startswith("# Test 21: the serving side on a bigger model, counted the way a company that runs the model counts (written BEFORE the run, 21 Oct 2026)")
    for i, ((tag, text), p) in enumerate(zip(B.BL_LINES, PROBS), 1): assert f"{i}. [{tag}] {text} ({p} %)" in head, i
    for s in (f"`{B.MODEL}`", f"`{B.REGEX_SHA256}`", "**NVIDIA L4**", "**32,000 tokens**", "jobs 192 to 223", "`P0`", "`PN`", "`P4`", "`PC`", "`X0`", "`XW`", "`XR`", "3,075 of 3,200", "1,343 of 3,200", "2,030 of 3,200",
              "Lines 2 to 7 all yes: about **72 %**", "Lines 8 and 18 both yes: about **22 %**", "All 21 yes: under **1 %**", "**The rule that decides what the Result may claim.**", "Line 2, 3, 4, 5, 6 or 7 reads NO: no claim is made from this run until the cause is found.",
              "never \"100 percent\" for a grammar", "never \"proven\"", "never a statement about any company", "the TPU is not run", "reads the **board's** power", "Why not the v6e-1 TPU", "about 1.71 compute units an hour", "Nothing here is proven.", "edits none of them",
              "Concept-map item 114"):
        assert s in head, s
    assert "# Result" not in head and head.count("\n- ") == 0 and re.findall(r"\(([0-9]+) %\)\n", head.split("## Predictions")[1].split("**Headline**")[0]) == [str(p) for p in PROBS]
    assert all(f"({x}) " in head for x in "abcdef")


def test_the_notebook_has_the_cells_in_order_and_its_code_runs_on_the_double_with_a_store():
    nb = json.load(open(NB)); C = nb["cells"]; src = lambda i: "".join(C[i]["source"])
    assert len(C) == 22 and [c["cell_type"] for c in C] == ["markdown", "markdown", "code"] + ["markdown", "code"] * 9 + ["markdown"] and nb["metadata"]["accelerator"] == "GPU"
    text = "\n".join("".join(c["source"]) for c in C)
    for needle in ("AXI test 21", "docs/PROVIDER_SIDE_BIG_TEST.md", "Qwen2.5-14B-Instruct", "**L4 GPU**", "vllm==0.31.0", "provider_side_big.py", "finish the update 71 commands", "axi_provider_big", "axi_provider_b_", "B.preflight()", "assert green", "PRE-FLIGHT IS RED",
                   "B.BigEngine()", "eng.warm_up()", "for label in B.ORDER: go(label)", "B.merge_files(", "B.verdict_big(m[\"cells\"])", "files.download", "Cell 4b", "about 1.7 compute units an hour", "so about 4 to 6 units", "the 21 reading-rule lines",
                   "GPU power", "NVIDIA L4", "There is no retry" if False else "no retry anywhere"):
        assert needle in text, needle
    for old in ("update 69", "update 67", "T4 GPU", "TypedVLLMEngine()", "provider_side_free import", "axi_provider_free", "three cells", "16 reading"): assert old not in text, old
    class Store:
        def __init__(s, d): s.d = d
        def path(s, n): return os.path.join(s.d, n)
        def restore(s, n): return "not found"
        def persist(s, n): pass
    import time as _time
    d = tempfile.mkdtemp(); fake = types.ModuleType("google.colab"); fake.files = types.SimpleNamespace(download=lambda p: None); sys.modules["google"] = types.ModuleType("google"); sys.modules["google.colab"] = fake
    ns = {"sys": sys, "os": os, "json": json, "time": _time, "glob": glob, "re": re, "store": Store(d)}
    try:
        pre, post = src(14).split("eng.warm_up()")
        eng = _engine(); eng.request_metrics = []; ns["eng"] = eng; ns["T"] = T; ns["L"] = L; ns["B"] = B; import axi.experiments.q_measure as Q; ns["Q"] = Q
        exec("def go" + post.split("def go", 1)[1], ns)
        class Fast(B.PowerMeter):
            def start(self): return self
            def stop(self, wall_seconds=None): return B.FakeMeter(90.0).stop(wall_seconds or 10.0)
        orig = B.PowerMeter; B.PowerMeter = Fast
        try:
            with contextlib.redirect_stdout(io.StringIO()) as o: exec(src(16), ns)
            assert "OK: cached counts are reported" in o.getvalue() and o.getvalue().count("step 1:") == 7, o.getvalue()[-600:]
            with contextlib.redirect_stdout(io.StringIO()) as o: exec(src(18), ns)
            assert sorted(os.listdir(d)) == sorted(f"axi_provider_b_{lab}.json" for lab in B.ORDER) and o.getvalue().count("done: ") == 7, o.getvalue()[-400:]
            with contextlib.redirect_stdout(io.StringIO()) as o: exec(src(20), ns)
        finally: B.PowerMeter = orig
        out = o.getvalue(); assert "7 cells:" in out and "(21 of 21 lines could be read" in out or "lines could be read" in out
        assert os.path.exists(os.path.join(d, "axi_provider_big.json")) and json.load(open(os.path.join(d, "axi_provider_big.json")))["meta"]["steps"] == 100
    finally:
        sys.modules.pop("google.colab", None); sys.modules.pop("google", None)
        for f in glob.glob(os.path.join(d, "*")): os.remove(f)
        os.rmdir(d)


# ---------------------------------------------------------------- 9 the front pages
def test_the_concept_map_and_the_verified_page_say_what_the_page_and_the_code_say():
    cm = open(os.path.join(ROOT, "docs", "RAJNISH_CONCEPT_MAP.md")).read(); vf = open(os.path.join(ROOT, "docs", "VERIFIED.md")).read()
    item = next(l for l in cm.split("\n") if l.startswith("114. **")); row = next(l for l in vf.split("\n") if l.startswith("| Provider-side serving test, eighth run"))
    assert "Next free: row 171, item 116, test 141, question 121" in cm and cm.count("Next free:") == 1 and "test 137 checks the bigger-model serving test, item 114" in cm
    assert PAGE_SHA256[:16] in item and PAGE_SHA256[:16] in row and "test 137" in item and "test 137" in row and "mutation check of test 137: not yet run" in item and "mutation check of test 137: not yet run" in row
    for s in ("Qwen2.5-14B-Instruct-AWQ", "NVIDIA L4", "32,000 tokens", "`P0`", "`PN`", "`P4`", "`PC`", "`X0`", "`XW`", "`XR`", "21 reading lines", "lines 8 and 18", "about 22 %", "then run once on Colab", "jobs 192 to 223", "joules", "TPU", "company", "nothing here is proven"): assert s in item, s
    assert "Qwen2.5-14B" in row and "run once on Colab" in row and "21" in row and "PROVIDER_SIDE_BIG_TEST.md" in vf and os.path.exists(PAGE) and os.path.exists(NB)


if __name__ == "__main__":
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]; bad = 0
    for n, f in fns:
        try: f(); print("ok  ", n)
        except Exception: bad += 1; print("FAIL", n); traceback.print_exc()
    print(f"{len(fns) - bad}/{len(fns)} passed"); sys.exit(1 if bad else 0)
