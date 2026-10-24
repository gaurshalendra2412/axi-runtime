"""Engineering support for docs/PROVIDER_SIDE_LONG_TEST.md (concept-map item 112): the exact-state cells and a kept chat of the provider-side serving test, with jobs of 100 steps. Checks that need no GPU and no model:
  1. nothing the test imports is edited (sha256 of nine files), the settings are the earlier runs', the kept-chat session adds a grammar and decides nothing, the exact-state session is the fifth run's;
  2. the 64 jobs (192 to 255) were used by no saved run, have 100 steps, start with their 30-step job, and the oracle's reply is inside every expression (14,400 requests); the fingerprints are the page's;
  3. the harness on a test double that obeys the grammar: the four cells, their flags, rounds and expressions, the merge (what it refuses), the windows;
  4. the 17 lines on a good run built on the double, each line at its edge (with the lines that must move with it), 'not run' for a missing cell, the rule about what may be claimed, the printout, the command line;
  5. the pre-flight check: each of its checks can read NO;
  6. the locked page, the figures it quotes from the fifth run's saved file, the notebook and its cells, the front pages.
What it does NOT show: anything about a real engine, a real GPU, or what a model writes over 100 steps."""
import contextlib, copy, glob, hashlib, inspect, io, json, math, os, re, subprocess, sys, tempfile, time, types

sys.path.insert(0, os.path.dirname(__file__))
import axi.engine.gate as G
import axi.engine.diagnostics as D
import axi.experiments.agent_loop as A
import axi.experiments.cascade as K
import axi.experiments.long_horizon as LH
import axi.experiments.provider_side as PS
import axi.experiments.provider_side_constrained as C
import axi.experiments.provider_side_bounded as B
import axi.experiments.provider_side_typed as T
import axi.experiments.provider_side_long as L
import test_provider_side_constrained as T124                                       # its fake `vllm` is reused, not copied

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..", "..")
RES = os.path.join(ROOT, "docs", "results")
PINS = dict(provider_side="bfb4734e171d9244cd47d10402be46528a557095a8275d02b38481a0e2b0b1d3", constrained="03452148e28c9f31372f12cc11ef1808ac67308adff972fa19d4aeeef7c96f7d",
            bounded="7427a686d151c4294809aebaa9664d0b5fa420854817eb79e910372c3f57b6bc", gate="b6a50408788791f76bc4d4129dad84debb6e09c9c5e804673cdf04abec224ce7",
            diagnostics="255f3c11db989b645077d10706a187e2c913576538a826e6dbdb8892b014b4f2", agent_loop="d658afaa29e3bce7e1fb7f43d88281fc40ee68ca59864421a65284e9973516b6",
            cascade="b0533316bc094e80a1d68eddb7e137c7c176af9a342341288360c18d61f74fe7", long_horizon="5718435de2178de110ddbb59dd3bf32e1bec9f1fd8625a27119d789399fa20ac")
TYPED_MODULE_SHA256 = "c5f05d5840374fb77a1c67c6fd33f166b26342e93733bd9a31aa474eacbc1420"
REGEX_SHA256 = "6f4e5a993f26ccd471bff13422fa622b7820858a91e05d43c2bccfbe6b864245"
GRAMMAR_SHA = {"L0": REGEX_SHA256, "LN": "85d17bbffb556c21acf180b989fecf3b2c92ca419fce1516a675fb24dd9929af",
               "L8": "9603a4c10ec798e0d8098b00f02fd15831a79f4fcd58d8beafbce6463e1c3ddb", "LC": "9603a4c10ec798e0d8098b00f02fd15831a79f4fcd58d8beafbce6463e1c3ddb"}
TYPED_FILE = os.path.join(RES, "axi_provider_typed.json")
TYPED_SHA256 = "6bb4ca481441182b2d055248dd75339992e9f8377b9f4b759dc1281f5b17866f"
PAGE = os.path.join(ROOT, "docs", "PROVIDER_SIDE_LONG_TEST.md")
PAGE_SHA256 = "f6cdaa3b6294008b2a27e84d6165dd9a9753558b835d2644135c948f98e8ae04"
RESULT_MARK = "\n\n# Result"
PROBS = [98, 92, 78, 97, 96, 92, 92, 68, 50, 40, 80, 90, 93, 80, 92, 88, 90]
SEED = PS.SEED


def raises(fn, exc=Exception):
    try: fn()
    except exc: return True
    return False


def _sha(path): return hashlib.sha256(open(path, "rb").read()).hexdigest()
def seed_of(job): return SEED * 1000 + job
def close(a, b): return abs(a - b) < 1e-9


_CACHE = {}


def typed_run():
    if "typed" not in _CACHE: _CACHE["typed"] = json.load(open(TYPED_FILE))
    return _CACHE["typed"]


# ---------------------------------------------------------------- 1 nothing edited, the settings, the sessions
def test_nothing_the_test_imports_is_edited_the_settings_are_the_earlier_runs_and_the_cells_are_the_pages():
    pins = T.pinned_hashes(); assert pins == PINS == L.PINS, pins
    assert L.TYPED_MODULE_SHA256 == TYPED_MODULE_SHA256 == T.fingerprint() == _sha(T.__file__), "provider_side_typed.py is pinned: the fifth run's Result was read with it"
    assert (L.MODEL, L.SEED, L.POLICY, L.HINT_RULES, L.MAX_NEW, L.CAPACITY_TOKENS) == (T.MODEL, 20261017, "repair_m6", False, 96, 80000) == (PS.MODEL, PS.SEED, PS.POLICY, PS.HINT_RULES, PS.MAX_NEW, PS.CAPACITY_TOKENS)
    assert L.REGEX_SHA256 == REGEX_SHA256 == T.REGEX_SHA256 == hashlib.sha256(L.REGEX.encode()).hexdigest() and L.NAME_CAP == 16 and PS.MAX_CONTEXT == 7600 and PS.MAX_MODEL_LEN == 8192
    assert L.STEPS == 100 and L.SHORT_STEPS == 30 == T.STEPS and L.FIRST == 192 and list(L.SEEN_JOBS) == list(range(192)) and L.ORDER == ["L0", "LN", "L8", "LC"] == L.cells_planned()
    assert L.CELLS == {"L0": (64, 192, "S", "capped"), "LN": (64, 192, "S", "named"), "L8": (8, 192, "S", "named"), "LC": (8, 192, "KC", "named")}
    assert L.GRAMMAR_SHA == GRAMMAR_SHA and (L.T_REPEAT, L.T_HIGH, L.T_GAIN, L.T_KEPT_TOL, L.T_TIME, L.T_FLAT) == (0.995, 0.99, 0.03, 0.01, 0.15, 1.3)
    assert (L.WIN_FIRST30, L.WIN_EARLY, L.WIN_LATE) == ((1, 30), (1, 33), (67, 100)) and L.GPU_FREE_MIB == 500 and L.LL_HEADLINE == (8, 9)
    src = open(L.__file__).read()
    for forbidden in ("def next_state", "def check(", "def ground_and_complete", "def strict_parse", "def oracle_script", "reset_prefix_cache", "import torch", "import vllm", "from vllm"): assert forbidden not in src, forbidden
    assert [n for n, v in vars(L).items() if inspect.isclass(v) and v.__module__ == L.__name__] == ["KeptChat"]
    mine = {n for n, v in vars(L.KeptChat).items() if callable(v)}; assert mine == {"__init__", "request", "accept"}, mine


def test_the_exact_state_session_is_the_fifth_runs_with_100_steps_and_the_kept_chat_session_adds_a_grammar_and_decides_nothing():
    for lab in ("L0", "LN", "L8"):
        K_, first, arm, shape = L.CELLS[lab]; ss = L.make_sessions(lab)
        assert len(ss) == K_ and all(type(s) is T.TypedSession and s.shape == shape and s.prompt == "old" and len(s.script) == 100 for s in ss) and [s.seed for s in ss] == [seed_of(e) for e in range(first, first + K_)]
    ss = L.make_sessions("LC"); assert len(ss) == 8 and all(type(s) is L.KeptChat and s.shape == "named" and s.arm == "KC" and len(s.script) == 100 for s in ss) and ss[0].seed == seed_of(192)
    # a 30-step typed session is the same as the first 30 steps of a 100-step one, step by step, on the double
    short = T.TypedSession(seed_of(200), 30, "old", "named"); long_ = T.TypedSession(seed_of(200), 100, "old", "named")
    for _ in range(30):
        a, b = short.request(), long_.request(); assert a.messages == b.messages and a.regex == b.regex
        short.accept(PS.Reply(a.stand_in_reply, 0, 0, 0)); long_.accept(PS.Reply(b.stand_in_reply, 0, 0, 0))
    assert short.done and not long_.done and short.log == long_.log
    # the kept chat of the earlier runs and this one write the same messages, the same log and the same stop; only the expression and the flag are added
    eng = T.TypedStandIn(capacity_tokens=int(L.CAPACITY_TOKENS * PS.PROXY_SCALE))
    mine = L.run_sessions(eng, [L.KeptChat(seed_of(e), 100, "named") for e in (192, 193)])
    theirs = L.run_sessions(eng, [PS.Session("KC", seed_of(e), 100, PS.HINT_RULES) for e in (192, 193)])
    strip = lambda log: [{k: v for k, v in x.items() if k != "in_language"} for x in log]
    for a, b in zip(mine["sessions"], theirs["sessions"]):
        assert strip(a["log"]) == strip(b["log"]) and a["stopped"] == b["stopped"] == "context" and len(a["log"]) < 100 and all(x["in_language"] for x in a["log"])
    k = L.KeptChat(seed_of(192), 100, "named"); rq = k.request(); assert isinstance(rq, T.TRequest) and rq.shape == "named" and rq.regex == T.regex_for("named", k.script[0]["instruction"]) and rq.prompt == "old" and rq.step == 1
    assert [m["role"] for m in rq.messages] == ["system", "user"] and raises(lambda: L.KeptChat(seed_of(192), 100, "wide"), AssertionError)
    real = T.regex_for
    try:                                                                                                  # a grammar that cannot say the answer stops the job
        T.regex_for = lambda shape, instruction: r"ADD\[9,9:9\]"
        assert raises(lambda: L.KeptChat(seed_of(192), 100, "named").request(), AssertionError)
    finally: T.regex_for = real


# ---------------------------------------------------------------- 2 the jobs
def test_the_64_jobs_were_used_by_no_saved_run_have_100_steps_start_with_their_30_step_job_and_the_oracle_is_inside_every_expression():
    used = set()
    for f in sorted(glob.glob(os.path.join(RES, "axi_provider_*.json"))):
        if os.path.basename(f) in ("axi_provider_free.json", "axi_provider_big.json"): continue            # the later runs repeat the sixth run's jobs on purpose (they are read in their own tests)
        if os.path.basename(f) == "axi_provider_long.json":                                      # this test's own run (added with its Result): it used exactly these jobs and no other
            assert {s["seed"] - SEED * 1000 for c in json.load(open(f))["cells"] for s in c["sessions"]} == set(range(192, 200)) | set(range(192, 256)); continue
        d = json.load(open(f))
        if not isinstance(d, dict) or "cells" not in d: continue
        for c in d["cells"]:
            for s in c["sessions"]: used.add(s["seed"] - SEED * 1000)
    assert used and max(used) == 191 and min(used) == 0 and not used & set(range(192, 256)), (min(used), max(used))
    kinds = {}; named = set(); n = 0
    for e in range(192, 256):
        sc = T.script_of(seed_of(e), 100); assert len(sc) == 100
        assert [x["instruction"] for x in sc[:30]] == [x["instruction"] for x in T.script_of(seed_of(e), 30)] and [x["ideal"] for x in sc[:30]] == [x["ideal"] for x in T.script_of(seed_of(e), 30)]
        for st in sc:
            n += 1; k = T.parse_request(st["instruction"])["kind"]; kinds[k] = kinds.get(k, 0) + 1
            for shape in T.SHAPES: assert T.in_language(shape, st["instruction"], st["ideal"]), (e, st)
            named.add(T.regex_for("named", st["instruction"]))
    assert n == 6400 and kinds == {"add_node": 2158, "add_edge": 2059, "delete": 2183} and len(named) == 835
    for lab, (K_, first, arm, shape) in L.CELLS.items(): assert T.grammar_sha256(shape, first, K_, 100, SEED) == GRAMMAR_SHA[lab], lab
    assert T.grammar_sha256("capped", 192, 64, 100, SEED) == REGEX_SHA256


# ---------------------------------------------------------------- 3 the harness on a double that obeys the grammar
FAULTS = {("old", seed_of(210), 4): "ADD[(1,1)->(1,1):dep#1]", ("old", seed_of(217), 12): "ADD[(1,1)->(1,1):dep#1]", ("old", seed_of(221), 30): "ADD[(1,1)->(1,1):dep#1]"}      # three slips the capped expression allows: they cost 97, 89 and 71 steps


def _engine(faults=None): return T.ScriptedStandIn(faults=FAULTS if faults is None else faults, capacity_tokens=int(L.CAPACITY_TOKENS * PS.PROXY_SCALE))


def good():
    if "good" not in _CACHE:
        eng = _engine(); _CACHE["good"] = [L.run_cell(eng, lab) for lab in L.ORDER]
    return copy.deepcopy(_CACHE["good"])


def test_the_harness_writes_the_four_cells_with_the_flags_the_rounds_and_the_expressions_and_a_capped_cell_loses_what_the_double_makes_it_write():
    cells = good(); by = {c["label"]: c for c in cells}; S = L.summarize_all(cells)
    for lab, c in by.items():
        K_, first, arm, shape = L.CELLS[lab]
        assert (c["K"], c["first_episode"], c["arm"], c["shape"], c["steps"], c["seed"], c["prompt"]) == (K_, first, arm, shape, 100, SEED, "old") and c["grammar_sha256"] == GRAMMAR_SHA[lab]
        assert c["engine"]["kind"] == "stand_in" and c["engine"]["constraint"]["regex_sha256"] == REGEX_SHA256 and c["policy"] == "repair_m6" and c["hint_rules"] is False
        assert [s["seed"] for s in c["sessions"]] == [seed_of(e) for e in range(first, first + K_)]
        assert all(("in_language" in x) and x["in_language"] for s in c["sessions"] for x in s["log"])
        assert sum(r["active"] for r in c["rounds"]) == sum(len(s["log"]) for s in c["sessions"]) and close(c["wall_seconds"], sum(r["seconds"] for r in c["rounds"]))
    for lab in ("L0", "LN", "L8"): assert all(len(s["log"]) == 100 and s["stopped"] is None and s["script_steps"] == 100 for s in by[lab]["sessions"]) and S[lab]["steps_done"] == (6400 if lab != "L8" else 800)
    assert S["L0"]["distinct_regexes"] == 1 and S["LN"]["distinct_regexes"] == 835 and S["L8"]["distinct_regexes"] <= 835 and S["LC"]["distinct_regexes"] <= S["L8"]["distinct_regexes"]
    assert all(s["stopped"] == "context" and 10 < len(s["log"]) < 100 for s in by["LC"]["sessions"]) and S["LC"]["stopped_jobs"] == 8 and S["LC"]["jobs_exact_to_end"] == 0
    # the capped cell holds the state wrongly after the three slips the double wrote; the named cell is forced to the oracle's reply there
    lost = lambda c: [(i, x["t"]) for i, s in enumerate(c["sessions"]) for x in s["log"] if not x["exact"]]
    l0 = lost(by["L0"]); assert {i for i, _ in l0} == {18, 25, 29} and [min(t for i, t in l0 if i == j) for j in (18, 25, 29)] == [4, 12, 30] and [sum(1 for i, _ in l0 if i == j) for j in (18, 25, 29)] == [97, 89, 71] and len(l0) == 257
    assert lost(by["LN"]) == [] and lost(by["L8"]) == [] and lost(by["LC"]) == [] and sum(x["corrupted"] for c in cells for s in c["sessions"] for x in s["log"]) == 0
    assert S["L0"]["jobs_exact_to_end"] == 61 and S["LN"]["jobs_exact_to_end"] == 64 and S["L0"]["slips_by_third"] == [3, 0, 0] and S["LN"]["slips_by_third"] == [0, 0, 0]
    assert 0.95 < S["L0"]["prompt_last_quarter"] / S["L0"]["prompt_first_quarter"] < 1.05 and S["LC"]["prompt_last_quarter"] > 3.5 * S["LC"]["prompt_first_quarter"]


def test_the_windows_count_from_step_1_include_both_ends_and_the_first_slips_fall_in_the_thirds_by_the_step_they_were_made_at():
    def cell(rows):                                              # rows: per job, the set of steps that are NOT exact
        return dict(sessions=[dict(seed=i, script_steps=100, stopped=None, log=[dict(t=t, exact=(t not in bad)) for t in range(1, 101)]) for i, bad in enumerate(rows)])
    c = cell([set(), {1}, {30}, {33}, {34}, {66}, {67}, {100}])
    assert close(L.window_share(c, 1, 30), 1 - 2 / (8 * 30)) and close(L.window_share(c, 1, 33), 1 - 3 / (8 * 33)) and close(L.window_share(c, 67, 100), 1 - 2 / (8 * 34)) and close(L.window_share(c, 1, 100), 1 - 7 / 800)
    assert L.window_share(c, 101, 120) is None and L.window_share(dict(sessions=[dict(log=[])]), 1, 30) is None
    assert L.first_slip_thirds(c) == [3, 2, 2], L.first_slip_thirds(c)                        # steps 1, 30, 33 | 34, 66 | 67, 100
    # the steps 33 and 34 are on either side of the first third; 66 and 67 either side of the second
    assert L.first_slip_thirds(cell([{33}, {34}, {66}, {67}, {100}, set()])) == [1, 2, 2]
    assert L.first_slip_thirds(cell([{50, 5}, {99, 20, 70}])) == [2, 0, 0] and L.first_slip_thirds(cell([set(), set()])) == [0, 0, 0]


def test_the_planned_cells_the_merge_and_what_it_refuses():
    cells = good(); assert [c["label"] for c in cells] == L.ORDER
    m = L.merge_cells(cells); assert [c["label"] for c in m["cells"]] == L.ORDER and m["meta"]["steps"] == 100 and m["meta"]["cells"] == {k: list(v) for k, v in L.CELLS.items()} and m["meta"]["name_cap"] == 16
    assert [c["label"] for c in L.merge_cells(list(reversed(cells)))["cells"]] == L.ORDER
    c0 = lambda: copy.deepcopy(cells[0])
    assert raises(lambda: L.merge_cells([]), AssertionError)
    assert raises(lambda: L.merge_cells(cells + [c0()]), ValueError)
    bad = c0(); bad["label"] = "F0"; assert raises(lambda: L.merge_cells([bad]), ValueError)
    for field, value in (("K", 32), ("first_episode", 128), ("arm", "KC"), ("shape", "kind")):
        b = c0(); b[field] = value; assert raises(lambda: L.merge_cells([b]), ValueError), field
    b = c0(); b["steps"] = 30; assert raises(lambda: L.merge_cells([b]), ValueError)
    b = c0(); b["engine"]["constraint"]["regex_sha256"] = "x"; assert raises(lambda: L.merge_cells([b]), ValueError)
    for field, value in (("seed", 1), ("policy", "gate_only"), ("hint_rules", True)):
        b = copy.deepcopy(cells[1]); b[field] = value; assert raises(lambda: L.merge_cells([cells[0], b]), ValueError), field
    b = copy.deepcopy(cells[1]); b["engine"]["kind"] = "vllm"; assert raises(lambda: L.merge_cells([cells[0], b]), ValueError)             # the stand-in mixed with a real engine
    b = copy.deepcopy(cells[1]); b["engine"]["gpu"] = "A100"; assert raises(lambda: L.merge_cells([cells[0], b]), ValueError)
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for c in cells: paths.append(L.save_cell(c, os.path.join(d, f"axi_provider_l_{c['label']}.json")))
        out = os.path.join(d, "axi_provider_long.json"); r = L.merge_files(paths, out); back = json.load(open(out))
        assert back["module_sha256"] == L.fingerprint() == _sha(L.__file__) and back["typed_module_sha256"] == T.fingerprint() and back["pinned_sha256"] == PINS and [c["label"] for c in back["cells"]] == L.ORDER and r["module_sha256"] == back["module_sha256"]
        assert "third_run_sha256" not in back


def test_a_cell_is_summarized_with_the_windows_the_stopped_jobs_and_the_steps_per_job():
    S = L.summarize_all(good()); s = S["L0"]
    assert s["label"] == "L0" and s["arm"] == "S" and s["steps_done"] == 6400 and s["first30"] is not None and s["late"] < s["early"] and s["steps_per_job"] == [100] * 64 and s["stopped_jobs"] == 0 and s["exact_steps"] == 6400 - sum(1 for c in good() if c["label"] == "L0" for ss in c["sessions"] for x in ss["log"] if not x["exact"])
    assert S["LN"]["first30"] == S["LN"]["early"] == S["LN"]["late"] == 1.0 and S["LC"]["steps_per_job"] == [len(x["log"]) for x in good()[3]["sessions"]]
    assert raises(lambda: L.summarize_all(good() + [good()[0]]), AssertionError)


# ---------------------------------------------------------------- 4 the seventeen lines
def read(cells): return [ok for _, _, _, ok in L.verdict_long(cells)]
def cell_of(cells, label): return next(c for c in cells if c["label"] == label)
def records(c): return [x for s in c["sessions"] for x in s["log"]]


def changed(cells, base=None):
    base = [True] * 17 if base is None else base; return [i + 1 for i, (a, b) in enumerate(zip(base, read(cells))) if a != b]


def flip(label, fn):
    cells = copy.deepcopy(good()); fn(cell_of(cells, label)); return cells


def flips(**fns):
    cells = copy.deepcopy(good())
    for label, fn in fns.items(): fn(cell_of(cells, label))
    return cells


def set_exact(lo, hi, n, value):
    """For a cell: the first `n` steps (jobs in order) with a step number from lo to hi whose flag is the other one get the flag `value`."""
    def f(c):
        pool = [x for x in records(c) if lo <= x["t"] <= hi and x["exact"] is not value]; assert len(pool) >= n, (len(pool), n)
        for x in pool[:n]: x["exact"] = value
    return f


def test_a_good_run_built_on_the_double_reads_yes_on_all_seventeen_lines_and_the_figures_are_the_ones_the_page_expects():
    g = good(); S = L.summarize_all(g); checks = L.verdict_long(g); assert read(g) == [True] * 17
    assert [(n, t, x) for n, t, x, _ in checks] == [(i, t, x) for i, (t, x) in enumerate(L.LL_LINES, 1)] and [t for t, _ in L.LL_LINES] == ["instrument"] * 2 + ["guard"] * 3 + ["claim"] * 12 and len(L.LL_LINES) == 17
    assert [S[l]["steps_done"] for l in L.ORDER] == [6400, 6400, 800, 339] and [S[l]["exact_steps"] for l in L.ORDER] == [6143, 6400, 800, 339] and [S[l]["lost_steps"] for l in L.ORDER] == [257, 0, 0, 0]
    s = S["L0"]; assert close(s["first30"], 1 - 47 / 1920) and close(s["early"], 1 - 56 / 2112) and close(s["late"], 1 - 102 / 2176) and 56 + 99 + 102 == 257 and s["slips_by_third"] == [3, 0, 0]
    assert S["LN"]["exact_share"] == 1.0 and close(S["L0"]["exact_share"], 6143 / 6400) and 1.0 - S["L0"]["exact_share"] > 0.03 and S["LC"]["late"] is None and S["LC"]["stopped_jobs"] == 8
    assert S["L0"]["exact_per_hour"] < S["LN"]["exact_per_hour"] and S["LC"]["exact_per_hour"] < S["L8"]["exact_per_hour"] and S["LN"]["seconds_per_step"] <= S["L0"]["seconds_per_step"] * 1.15
    # the replay rule holds on the good run: LN writes L0's texts until the three steps at which L0's reply is outside the named language, where the reply is the oracle's
    ln, l0 = cell_of(g, "LN"), cell_of(g, "L0"); assert T.same_until_forced(ln, l0, "named") and not T.same_until_forced(ln, l0, "capped"), "the three capped slips are inside the capped language, so under that language LN differs from L0"
    first_diff = {i: next(x["t"] for x, y in zip(a["log"], b["log"]) if x["text"] != y["text"]) for i, (a, b) in enumerate(zip(ln["sessions"], l0["sessions"])) if any(x["text"] != y["text"] for x, y in zip(a["log"], b["log"]))}
    assert first_diff == {18: 4, 25: 12, 29: 30}, first_diff


def test_lines_1_to_5_flip_at_their_edges_and_the_lines_that_must_move_with_them_move():
    base = good(); N = sum(len(T.replies_of(c)) for c in base); k = int((1 - T.T_PARSED) * N + 1e-9); assert (N, k, T.T_PARSED) == (13939, 69, 0.995)
    # 1: a missing cached count in a first record, in a last record, in a cell of 8 jobs
    assert changed(flip("L0", lambda c: records(c)[0].update(n_cached=None))) == [1] and changed(flip("LC", lambda c: records(c)[-1].update(n_cached=None))) == [1] and changed(flip("L8", lambda c: records(c)[400].update(n_cached=None))) == [1]
    # 2: all 96 new tokens; one reply outside its language; a reply without the flag; the share refused by the strict parser at its limit (69 of 13,939) and one over
    assert changed(flip("LN", lambda c: records(c)[5].update(n_new=96))) == [2] and changed(flip("LN", lambda c: records(c)[5].update(n_new=95))) == []
    assert changed(flip("LN", lambda c: records(c)[5].update(in_language=False))) == [2] and changed(flip("LC", lambda c: records(c)[5].pop("in_language"))) == [2]
    unparse = lambda n: (lambda c: [x.update(parsed=False) for x in records(c)[:n]])
    assert changed(flip("L0", unparse(k))) == [] and changed(flip("L0", unparse(k + 1))) == [2], (N, k)
    # 3: a reply before the first forced step that is not L0's; one after it; the forced reply outside the language; a job of another seed; a job fewer; a step fewer in a job that is not forced
    assert changed(flip("LN", lambda c: c["sessions"][0]["log"][4].update(text="DEL[0,0]"))) == [3]
    assert changed(flip("LN", lambda c: c["sessions"][18]["log"][3].update(text="ADD[(1,1)->(1,1):dep#1]"))) == [3], "the forced reply must be inside the named language: this is the capped slip"
    assert changed(flip("LN", lambda c: c["sessions"][18]["log"][10].update(text="DEL[0,0]"))) == [], "job 210 is forced at step 4; what comes after is free"
    assert changed(flip("LN", lambda c: c["sessions"][29]["log"][28].update(text="DEL[0,0]"))) == [3] and changed(flip("LN", lambda c: c["sessions"][29]["log"][29].update(text="DEL[0,0]"))) == [3], "job 221 is forced at step 30: step 29 is before it, step 30 is the forced one (the oracle's own)"
    assert changed(flip("LN", lambda c: c["sessions"][29]["log"][30].update(text="DEL[0,0]"))) == [], "step 31 of job 221 is after the forced step"
    assert changed(flip("LN", lambda c: c["sessions"][3].update(seed=seed_of(9)))) == [3] and changed(flip("LN", lambda c: c["sessions"].pop())) == [3]
    assert changed(flip("LN", lambda c: c["sessions"][3]["log"].pop())) == [3, 5] and changed(flip("L0", lambda c: c["sessions"][3]["log"].pop())) == [3, 5], "a job with a step fewer is not the other's job (line 3) and did not run 100 steps (line 5)"
    assert changed(flip("LN", lambda c: c["sessions"][18]["log"].pop())) == [5] and changed(flip("L0", lambda c: c["sessions"][18]["log"].pop())) == [5], "job 210 is forced at step 4, so how many steps it has after that is not read by line 3"
    assert changed(flip("LC", lambda c: c["sessions"][0]["log"][2].update(text="DEL[0,0]"))) == [] and changed(flip("L8", lambda c: c["sessions"][0]["log"][2].update(text="DEL[0,0]"))) == [], "L8 and LC are not paired with L0"
    # 4: the recorded expressions are not the ones recomputed from the jobs; they are, but not the page's; the oracle outside; a cell of other jobs
    assert changed(flip("LC", lambda c: c.update(grammar_sha256="0" * 64))) == [4] and changed(flip("LN", lambda c: c.update(grammar_sha256=T.grammar_sha256("kind", 192, 64, 100, SEED)))) == [4]
    assert changed(flip("LN", lambda c: c.update(steps=99, grammar_sha256=T.grammar_sha256("named", 192, 64, 99, SEED)))) == [4], "self-consistent but not the page's"
    assert changed(flip("L8", lambda c: c.update(first_episode=193))) == [4]
    real = T.in_language
    try:
        T.in_language = lambda shape, instruction, reply: False if instruction.startswith("Add a service") else real(shape, instruction, reply)
        assert changed(copy.deepcopy(good())) == [3, 4], "the oracle's reply outside the expression: line 4, and line 3 which also asks whether a reply is inside the language"
    finally: T.in_language = real
    # 5: a job of an exact-state cell that ended early, that has a stop, or has more than 100 entries; the kept chat is not read here
    assert changed(flip("L8", lambda c: c["sessions"][0]["log"].pop())) == [5] and changed(flip("L0", lambda c: c["sessions"][7].update(stopped="context"))) == [5]
    assert changed(flip("LN", lambda c: c["sessions"][7]["log"].append(copy.deepcopy(c["sessions"][7]["log"][-1])))) == [3, 5]
    assert changed(flip("LC", lambda c: c["sessions"][0].update(stopped=None))) == [13], "a kept chat that is not stopped is line 13's business, not line 5's"


def test_lines_6_and_7_read_the_prompt_of_the_first_and_last_quarter_of_L0_and_the_corrupted_flag_of_every_cell():
    def shaped(extra_t, v):
        def f(c):
            for s in c["sessions"]:
                for x in s["log"]: x["n_prompt"] = v if x["t"] == extra_t else 1300 if x["t"] >= 76 else 1000
        return f
    assert changed(flip("L0", shaped(0, 0))) == [], "last quarter 1.3 times the first: the edge, at most"
    assert changed(flip("L0", lambda c: [x.update(n_prompt=1301 if x["t"] >= 76 else 1000) for x in records(c)])) == [6]
    assert changed(flip("L0", shaped(25, 0))) == [6] and changed(flip("L0", shaped(26, 0))) == [], "step 25 is the last of the first quarter, step 26 is in neither"
    assert changed(flip("L0", shaped(76, 100000))) == [6] and changed(flip("L0", shaped(75, 100000))) == [], "step 76 is the first of the last quarter, step 75 is in neither"
    assert changed(flip("LN", lambda c: [x.update(n_prompt=100000) for x in records(c) if x["t"] >= 76])) == [], "only L0 is read by line 6"
    assert changed(flip("LC", lambda c: [x.update(n_prompt=1) for x in records(c)])) == [], "the kept chat's prompt grows and that is not what line 6 reads"
    for label in L.ORDER: assert changed(flip(label, lambda c: records(c)[3].update(corrupted=True))) == [7], label
    assert changed(flip("LC", lambda c: records(c)[-1].update(corrupted=1))) == [7] and changed(flip("L0", lambda c: records(c)[-1].update(corrupted=0))) == []


def test_lines_8_to_12_flip_at_the_edges_of_their_windows_and_at_the_points_of_their_thresholds():
    lose = lambda lo, hi, n: set_exact(lo, hi, n, False); restore = lambda lo, hi, n: set_exact(lo, hi, n, True)
    # 8: steps 1 to 30 of LN, 64 jobs: 1,920 steps, 99.5 percent allows 9 lost (1,911) and not 10
    assert changed(flip("LN", lose(1, 30, 9))) == [] and changed(flip("LN", lose(1, 30, 10))) == [8]
    assert changed(flip("LN", lose(30, 30, 10))) == [8] and changed(flip("LN", lose(31, 31, 10))) == [], "step 30 is the last of the window, step 31 is not in it"
    assert changed(flip("LN", lose(1, 1, 10))) == [8] and changed(flip("LN", lose(34, 66, 40))) == [], "forty steps in the middle third move nothing: 6,360 of 6,400"
    # 9: all 6,400 steps of LN, 99 percent allows 64 lost and not 65; with 66 the gap to L0 (6,143 exact, 0.03 above is 6,335) closes as well
    assert changed(flip("LN", lose(34, 66, 64))) == [] and changed(flip("LN", lose(34, 66, 65))) == [9], "6,335 of 6,400 is exactly L0's share (6,143 of 6,400) plus three points: line 11 stays yes at the tie"
    assert changed(flip("LN", lose(34, 66, 66))) == [9, 11]
    # 10: steps 67 to 100 of LN, 2,176 steps, 99 percent allows 21 lost and not 22
    assert changed(flip("LN", lose(67, 100, 21))) == [] and changed(flip("LN", lose(67, 100, 22))) == [10]
    assert changed(flip("LN", lose(66, 66, 22))) == [] and changed(flip("LN", lose(67, 67, 22))) == [10], "step 66 is the last of the middle third, step 67 the first of the window"
    # 11: LN at 1.0 against L0 plus 0.03: L0 may lose 192 of its 6,400 steps (6,208 is 0.97) and not 191; restoring L0's steps in the middle third moves no window of line 12
    assert changed(flip("L0", restore(34, 66, 65))) == [] and changed(flip("L0", restore(34, 66, 66))) == [11]
    # 12: L0 late strictly below L0 early. Lose 3 steps (40 to 42) in every job: 192 steps, all in the middle: early = late = 1.0 and line 12 reads NO; one more step late and it reads yes
    def middle(c):
        for x in records(c): x["exact"] = not (40 <= x["t"] <= 42)
    def middle_and(c, t):
        middle(c); c["sessions"][0]["log"][t - 1]["exact"] = False
    assert changed(flip("L0", middle)) == [12] and changed(flip("L0", lambda c: middle_and(c, 100))) == [] and changed(flip("L0", lambda c: middle_and(c, 67))) == []
    assert changed(flip("L0", lambda c: middle_and(c, 66))) == [12] and changed(flip("L0", lambda c: middle_and(c, 34))) == [12], "a step in the middle third is in neither window"
    assert changed(flip("L0", lambda c: middle_and(c, 33))) == [12], "a step lost early only makes the early share lower than the late one"
    assert changed(flip("L0", lambda c: [x.update(exact=True) for x in records(c)])) == [11, 12], "a control that lost nothing: no gain, no loss of ground"
    assert changed(flip("LN", lose(1, 100, 100))) == [8, 9, 10, 11], "one job that lost its first 100 steps: every window of LN and the gap to L0 go"


def test_lines_13_to_17_read_the_stops_the_kept_chat_and_the_seconds():
    # 13: every kept-chat job stopped with the reason context before its 100th step
    assert changed(flip("LC", lambda c: c["sessions"][3].update(stopped=None))) == [13] and changed(flip("LC", lambda c: c["sessions"][3].update(stopped="tokens"))) == [13]
    assert changed(flip("LC", lambda c: c["sessions"][3]["log"].extend(copy.deepcopy(c["sessions"][3]["log"][-1]) for _ in range(100 - len(c["sessions"][3]["log"]))))) == [13], "a job with 100 steps did not stop before the 100th"
    assert changed(flip("LC", lambda c: c["sessions"][3]["log"].extend(copy.deepcopy(c["sessions"][3]["log"][-1]) for _ in range(99 - len(c["sessions"][3]["log"]))))) == [], "99 steps and a stop is a stop"
    assert changed(flip("L8", lambda c: c["sessions"][3].update(stopped="context"))) == [5], "an exact-state job with a stop is line 5's business"
    # 14: LC against L8 minus 0.01: 339 steps, 336 exact is 0.9911 and 335 is 0.9882
    n = len(records(cell_of(good(), "LC"))); assert n == 339 and int(0.01 * n) == 3
    assert changed(flip("LC", set_exact(1, 100, 3, False))) == [] and changed(flip("LC", set_exact(1, 100, 4, False))) == [14]
    assert changed(flips(L8=set_exact(1, 100, 80, False), LC=set_exact(1, 100, 4, False))) == [], "L8 with 80 steps less exact (0.9) makes four lost steps of LC fit line 14 (0.9882 against 0.89)"
    # 15: LC exact steps per hour strictly below those of L8; with the two walls set to 800 s and 339 s both are 3,600 an hour
    def walls(w8, wc): return flips(L8=lambda c: c.update(wall_seconds=w8), LC=lambda c: c.update(wall_seconds=wc))
    assert changed(walls(800.0, 339.0)) == [15] and changed(walls(800.0, 339.001)) == [] and changed(walls(800.0, 338.999)) == [15]
    # 16 and 17: the walls of LN against L0. 6,400 against 6,143 exact steps: LN is higher per hour up to 1.0418 times the seconds; line 16 is the 15 percent
    wall = lambda f: flips(LN=lambda c: c.update(wall_seconds=cell_of(good(), "L0")["wall_seconds"] * f))
    assert changed(wall(1.0418)) == [] and changed(wall(1.0419)) == [17] and changed(wall(1.15)) == [17] and changed(wall(1.1501)) == [16, 17] and changed(wall(0.5)) == []
    assert 6400 / 6143 > 1.0418 and 6400 / 6143 < 1.0419
    slow0 = lambda f: flips(L0=lambda c: c.update(wall_seconds=cell_of(good(), "L0")["wall_seconds"] * f))
    assert changed(slow0(1.0)) == [] and changed(slow0(0.8)) == [16, 17], "L0 at 0.8 of its seconds: LN is then more than 15 percent slower per step and does fewer exact steps per hour"


def test_a_cell_that_was_not_run_reads_not_run_and_never_no_and_a_job_cut_short_is_not_a_late_window():
    miss = lambda *labels: [c for c in good() if c["label"] not in labels]; N = None
    assert read(miss("LN")) == [True, True, N, True, True, True, True, N, N, N, N, True, True, True, True, N, N]
    assert read(miss("L0")) == [True, True, N, True, True, N, True, True, True, True, N, N, True, True, True, N, N]
    assert read(miss("L8")) == [True] * 13 + [N, N, True, True] and read(miss("LC")) == [True] * 12 + [N, N, N, True, True]
    assert read(miss("L0", "LN")) == [True, True, N, True, True, N, True, N, N, N, N, N, True, True, True, N, N]
    assert read(miss("L8", "LC")) == [True, True, True, True, True, True, True, True, True, True, True, True, N, N, N, True, True]
    assert read([cell_of(good(), "LC")]) == [True, True, N, True, N, N, True, N, N, N, N, N, True, N, N, N, N], "only the kept chat: no exact-state job to read lines 5 and 6 on"
    assert read([]) == [N] * 17 and read([cell_of(good(), "LN")]) == [True, True, N, True, True, N, True, True, True, True, N, N, N, N, N, N, N]
    empty = copy.deepcopy(good()); [s.update(log=[]) for s in cell_of(empty, "LN")["sessions"]]
    assert read(empty) == read(miss("LN")), "a cell with no step in any job was not run"
    # LN that was cut after 30 steps: steps 1 to 30 can be read, steps 67 to 100 cannot, and the jobs did not run 100 steps
    cut = copy.deepcopy(good()); [s.update(log=s["log"][:30]) for s in cell_of(cut, "LN")["sessions"]]; r = read(cut)
    assert r[4] is False and r[7] is True and r[8] is True and r[9] is None and r[2] is False
    cut = copy.deepcopy(good()); [s.update(log=s["log"][:20]) for s in cell_of(cut, "L0")["sessions"]]; r = read(cut)
    assert r[4] is False and r[11] is None and r[10] is False, "L0 cut after 20 steps has no step in 67 to 100 (line 12 cannot be read) and 26 lost steps of its 1,280 are only two points under LN (line 11)"
    # a kept chat of which no job went past step 30
    none_late = copy.deepcopy(good()); [s.update(log=s["log"][:10]) for s in cell_of(none_late, "LC")["sessions"]]; assert cell_of(none_late, "LC")["sessions"][0]["stopped"] == "context" and read(none_late)[12:15] == [True] * 3


def test_the_rule_about_what_may_be_claimed_follows_lines_2_3_4_and_5_and_the_lines_that_were_not_run_claim_nothing():
    checks = L.verdict_long(good()); a = L.claims_allowed(checks)
    assert a == dict(base=True, constraint_held=True, grammar_changed_only_what_it_forbids=True, expressions_as_defined=True, jobs_ran_all_steps=True, repeat_of_the_fifth_run={8: True}, over_100_steps={9: True, 10: True, 11: True, 12: True},
                     prompt_and_graph={6: True, 7: True}, kept_chat={13: True, 14: True, 15: True}, time_and_cost_in_one_session={16: True, 17: True}), a
    names = {2: "constraint_held", 3: "grammar_changed_only_what_it_forbids", 4: "expressions_as_defined", 5: "jobs_ran_all_steps"}
    for i, name in names.items():
        for value in (False, None):
            shown = [(n, t, x, (value if n == i else v)) for n, t, x, v in checks]; a = L.claims_allowed(shown)
            assert a["base"] is False and a[name] is value and a["repeat_of_the_fifth_run"] == {8: None} and a["over_100_steps"] == {9: None, 10: None, 11: None, 12: None} and a["prompt_and_graph"] == {6: None, 7: None} \
                and a["kept_chat"] == {13: None, 14: None, 15: None} and a["time_and_cost_in_one_session"] == {16: None, 17: None}, (i, value)
    shown = [(n, t, x, (False if n in (9, 17) else v)) for n, t, x, v in checks]; a = L.claims_allowed(shown)
    assert a["base"] and a["over_100_steps"] == {9: False, 10: True, 11: True, 12: True} and a["time_and_cost_in_one_session"] == {16: True, 17: False} and a["repeat_of_the_fifth_run"] == {8: True}
    shown = [(n, t, x, (None if n in (13, 14, 15) else v)) for n, t, x, v in checks]; a = L.claims_allowed(shown); assert a["base"] and a["kept_chat"] == {13: None, 14: None, 15: None}
    assert L.claims_allowed(L.verdict_long([]))["base"] is False


def test_the_table_the_windows_the_misses_and_the_lines_are_printed_in_the_format_the_notebook_shows():
    S = {"LN": dict(steps_done=6400, wall_seconds=632.8, seconds_per_step=0.0989, computed_per_step=106.4, exact_share=0.9984, first30=1.0, late=0.9954, jobs_exact_to_end=63, corrupted_steps=0, stopped_jobs=0, out_of_language=0, exact_per_hour=36407.0,
                    slips_by_third=[0, 0, 1], prompt_first_quarter=388.42, prompt_last_quarter=379.87, longest_prompt=748, steps_per_job=[100] * 64, first_misses=[(247, 81, "delete", "DEL[2,3]")]),
         "LC": dict(steps_done=339, wall_seconds=102.4, seconds_per_step=0.302, computed_per_step=None, exact_share=1.0, first30=1.0, late=None, jobs_exact_to_end=0, corrupted_steps=0, stopped_jobs=8, out_of_language=1, exact_per_hour=None,
                    slips_by_third=[0, 0, 0], prompt_first_quarter=1870.1, prompt_last_quarter=6896.06, longest_prompt=7557, steps_per_job=[24, 56, 40, 41, 42, 43, 44, 45], first_misses=[])}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): L.print_cells(S); L.print_windows(S); L.print_misses(S)
    assert buf.getvalue() == ("\ncell   K      jobs arm grammar |  steps  seconds  s/step comp/step |  exact   1-30  67-100 jobs ok corrupt | stopped outside  exact/h\n"
                              "LN    64   192-255   S   named |   6400      633   0.099       106 |  99.8% 100.0%   99.5%      63       0 |       0       0    36407\n"
                              "LC     8   192-199  KC   named |    339      102   0.302         - | 100.0% 100.0%       -       0       0 |       8       1        -\n"
                              "LN: first slips of jobs in steps 1-33 / 34-66 / 67-100: 0 / 0 / 1;  prompt tokens first quarter 388, last quarter 380, longest 748;  steps per job 100 to 100\n"
                              "LC: first slips of jobs in steps 1-33 / 34-66 / 67-100: 0 / 0 / 0;  prompt tokens first quarter 1870, last quarter 6896, longest 7557;  steps per job 24 to 56\n"
                              "first misses in LN: job 247 step 81 (delete) 'DEL[2,3]'\n"), buf.getvalue()
    checks = [(1, "instrument", "aaa", True), (2, "instrument", "bbb", False), (3, "guard", "ccc", None)] + [(i, "guard" if i < 6 else "claim", f"x{i}", True) for i in range(4, 18)]; buf = io.StringIO()
    with contextlib.redirect_stdout(buf): L.print_verdict(checks)
    want = ("\n-- provider-side test, jobs of 100 steps (reading rules fixed before the run) --\n  yes       1. [instrument] aaa\n  NO        2. [instrument] bbb\n  not run   3. [guard] ccc\n" + "".join(f"  yes      {i:2d}. [{'guard' if i < 6 else 'claim'}] x{i}\n" for i in range(4, 18)) +
            "  (16 of 17 lines could be read; headline lines 8 and 9: yes, yes)\n  rule on the page: line 2, 3, 4 or 5 did not read yes, so NO claim is made from this run until the cause is found.\n")
    assert buf.getvalue() == want, buf.getvalue()
    checks = L.verdict_long(good()); buf = io.StringIO()
    with contextlib.redirect_stdout(buf): L.print_verdict(checks, title="t")
    o = buf.getvalue(); assert o.startswith("\n-- t (reading rules fixed before the run) --\n") and o.count("  yes ") == 17 + 0 and "(17 of 17 lines could be read; headline lines 8 and 9: yes, yes)" in o
    assert o.endswith("  rule on the page: lines 2, 3, 4 and 5 read yes, so the figures of the cells are claimed as read. A statement about time or cost comes only from lines 16 and 17 (cells of one session).\n"
                      "  never: a multiple of capacity, dollars, '100 percent' for a grammar, a statement about any provider, anything about requests in free words, anything about a chat with a summary or a window, anything about a model that was not run.\n"), o
    assert o.count("[claim]") == 12 and o.count("[guard]") == 3 and o.count("[instrument]") == 2 and ("never: a multiple of capacity" in o) != ("NO claim is made from this run" in o)
    flipped = [(n, t, x, (False if n == 8 else v)) for n, t, x, v in checks]; buf = io.StringIO()
    with contextlib.redirect_stdout(buf): L.print_verdict(flipped)
    o = buf.getvalue(); assert "headline lines 8 and 9: NO, yes" in o and "  NO        8. [claim]" in o and o.count("  NO  ") == 1
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): L.print_verdict(L.verdict_long([]))
    assert "(0 of 17 lines could be read; headline lines 8 and 9: not run, not run)" in buf.getvalue() and buf.getvalue().count("  not run  ") == 17 and "NO claim is made from this run" in buf.getvalue()


def test_the_dry_run_says_it_is_not_a_result_and_the_command_line_merges_cell_files_prints_the_lines_and_shows_its_help():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): S = L.dry_run(steps=4)
    assert buf.getvalue().startswith("STAND-IN (a model of a block cache; it answers with the oracle's own reply), NOT a result\n") and sorted(S) == sorted(L.ORDER) and [S[l]["steps_done"] for l in L.ORDER] == [256, 256, 32, 32]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): S = L.dry_run(steps=3, labels=["LN"])
    assert list(S) == ["LN"] and "LN " in buf.getvalue() and "L0 " not in buf.getvalue()
    env = dict(os.environ, PYTHONPATH=os.path.join(HERE, "..")); cwd = os.path.join(HERE, "..")
    run = lambda *args: subprocess.run([sys.executable, "-m", "axi.experiments.provider_side_long", *args], capture_output=True, text=True, env=env, cwd=cwd)
    h = run(); flat = " ".join(h.stdout.split())
    assert h.returncode == 0 and "--dry-run" in h.stdout and "--preflight" in h.stdout and "--merge" in h.stdout and "run every cell on the stand-in engine (no GPU); prints, does not save" in flat and "run the pre-flight check (reads nvidia-smi)" in flat and "put cell files together" in flat
    d = run("--dry-run"); assert d.returncode == 0 and "NOT a result" in d.stdout and all(f"\n{lab} " in d.stdout for lab in L.ORDER)
    p = run("--preflight"); assert p.returncode == 1 and "RED: do not run the next cells" in p.stdout and "FAIL  the GPU is a free T4" in p.stdout and "ok    the jobs are new" in p.stdout, "no GPU here: the pre-flight reads RED and the command exits with 1"
    with tempfile.TemporaryDirectory() as t:
        paths = [L.save_cell(c, os.path.join(t, f"axi_provider_l_{c['label']}.json")) for c in good()]; out = os.path.join(t, "m.json")
        m = run("--merge", *paths, "--out", out)
        assert m.returncode == 0 and "written to" in m.stdout and "-- provider-side test, jobs of 100 steps" in m.stdout and m.stdout.count("[claim]") == 12 and m.stdout.count("[guard]") == 3 and os.path.exists(out) and "first slips of jobs in steps 1-33" in m.stdout
        assert [c["label"] for c in json.load(open(out))["cells"]] == L.ORDER


# ---------------------------------------------------------------- 5 the pre-flight check
T4 = lambda used=3: (lambda: dict(name="Tesla T4", used_mib=used, total_mib=15360))


def test_the_preflight_reads_green_on_this_repository_and_prints_each_finding():
    res = L.preflight(gpu=T4()); names = [n for n, _, _ in res]
    assert names == ["the jobs are new", "every job has 100 steps and starts with its 30-step job", "the files this test imports are the pinned ones", "the expressions are the ones on the page", "the oracle's reply is inside the expression of every request",
                     "cell L0: its first request is built", "cell LN: its first request is built", "cell L8: its first request is built", "cell LC: its first request is built", "the GPU is a free T4"], names
    assert all(ok for _, ok, _ in res) and "14400 of 14400 requests" in res[4][2] and "jobs 192 to 255 (64)" in res[0][2] and "L0 6f4e5a99; LN 85d17bbf; L8 9603a4c1; LC 9603a4c1" in res[3][2] and "3 of 15360 MiB" in res[9][2]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): good_ = L.print_preflight(res)
    out = buf.getvalue(); assert good_ is True and out.startswith("-- pre-flight --\n  ok    the jobs are new: ") and out.count("  ok    ") == 10 and out.endswith("  GREEN: nothing found that would waste the run. Go on to the next cell.\n") and "FAIL" not in out and "RED" not in out


def red(res):
    """The names of the checks that read NO."""
    return [n for n, ok, _ in res if not ok]


def test_each_check_of_the_preflight_can_read_no_and_only_its_own():
    real_pins, real_seen, real_make, real_in, real_sha, real_fp = T.pinned_hashes, L.SEEN_JOBS, L.make_sessions, T.in_language, dict(L.GRAMMAR_SHA), T.fingerprint
    try:
        # the files: one pinned file edited; the typed module edited
        T.pinned_hashes = lambda: dict(real_pins(), gate="0" * 64); r = L.preflight(gpu=T4()); assert red(r) == ["the files this test imports are the pinned ones"] and "changed: ['gate']" in r[2][2]; T.pinned_hashes = real_pins
        T.pinned_hashes = lambda: {k: v for k, v in real_pins().items() if k != "cascade"}; r = L.preflight(gpu=T4()); assert red(r) == ["the files this test imports are the pinned ones"] and "changed: ['cascade']" in r[2][2]; T.pinned_hashes = real_pins
        T.fingerprint = lambda: "0" * 64; r = L.preflight(gpu=T4()); assert red(r) == ["the files this test imports are the pinned ones"] and "provider_side_typed.py as pinned: False" in r[2][2]; T.fingerprint = real_fp
        # the jobs: jobs 192 to 199 were used by an earlier run
        L.SEEN_JOBS = range(0, 200); r = L.preflight(gpu=T4()); assert red(r) == ["the jobs are new"] and "[192, 193, 194, 195, 196, 197, 198, 199]" in r[0][2]; L.SEEN_JOBS = real_seen
        L.SEEN_JOBS = range(0, 192); assert red(L.preflight(gpu=T4())) == []
        # a script of 99 steps, even if 99 steps were asked for; the 30-step prefix
        r = L.preflight(gpu=T4(), steps=99); assert "every job has 100 steps and starts with its 30-step job" in red(r) and "steps per job 99 to 99" in r[1][2]
        r = L.preflight(gpu=T4(), steps=101); assert "every job has 100 steps and starts with its 30-step job" in red(r)
        real_script = T.script_of
        try:
            T.script_of = lambda seed, steps: ([dict(x, instruction=x["instruction"] + " ") for x in real_script(seed, steps)[:3]] + list(real_script(seed, steps)[3:])) if steps == 100 else real_script(seed, steps)
            r = L.preflight(gpu=T4()); assert "every job has 100 steps and starts with its 30-step job" in red(r) and "equal the 30-step job's: False" in r[1][2]
            assert "the expressions are the ones on the page" in red(r) and r[3][2].startswith("could not be checked: ValueError: not a request this test can read") and len(r) == 10, "a check that cannot be made reads NO with the error, and the others are still made"
        finally: T.script_of = real_script
        # the expressions: one of the page's fingerprints is wrong
        L.GRAMMAR_SHA = dict(real_sha, LN="0" * 64); r = L.preflight(gpu=T4()); assert red(r) == ["the expressions are the ones on the page"] and "wrong: ['LN']" in r[3][2]; L.GRAMMAR_SHA = real_sha
        # the oracle outside an expression
        T.in_language = lambda shape, instruction, reply: not instruction.startswith("Delete"); r = L.preflight(gpu=T4()); assert red(r) == ["the oracle's reply is inside the expression of every request"] and " of 14400 requests" in r[4][2] and "14400 of 14400" not in r[4][2]; T.in_language = real_in
        # a cell whose first request cannot be built, or is not what is described
        def boom(label, steps=100, seed=PS.SEED):
            if label == "LC": raise RuntimeError("no kept chat")
            return real_make(label, steps, seed)
        L.make_sessions = boom; r = L.preflight(gpu=T4()); assert red(r) == ["cell LC: its first request is built"] and "RuntimeError: no kept chat" in r[8][2]; L.make_sessions = real_make
        def wrong(label, steps=100, seed=PS.SEED):
            s = real_make(label, steps, seed)
            if label == "L8":
                s[0].script = [dict(s[0].script[0], instruction="zzz")] + s[0].script[1:]
            return s
        L.make_sessions = wrong; r = L.preflight(gpu=T4()); assert red(r) == ["cell L8: its first request is built"]; L.make_sessions = real_make
        def three(label, steps=100, seed=PS.SEED):
            s = real_make(label, steps, seed); orig = s[0].request
            if label == "LN": s[0].request = lambda: (lambda rq: T.TRequest(rq.messages + [{"role": "user", "content": "x"}], rq.stand_in_reply, seed=rq.seed, step=rq.step, prompt=rq.prompt, shape=rq.shape, regex=rq.regex))(orig())
            return s
        L.make_sessions = three; r = L.preflight(gpu=T4()); assert red(r) == ["cell LN: its first request is built"]; L.make_sessions = real_make
    finally: T.pinned_hashes, L.SEEN_JOBS, L.make_sessions, T.in_language, L.GRAMMAR_SHA, T.fingerprint = real_pins, real_seen, real_make, real_in, real_sha, real_fp
    # the GPU: a different one, a busy one at the limit and one over, none, an unreadable answer
    gpu = lambda name, used: (lambda: dict(name=name, used_mib=used, total_mib=16000))
    assert red(L.preflight(gpu=gpu("Tesla T4", 500))) == [] and red(L.preflight(gpu=gpu("Tesla T4", 501))) == ["the GPU is a free T4"]
    assert red(L.preflight(gpu=gpu("Tesla V100-SXM2-16GB", 0))) == ["the GPU is a free T4"] and red(L.preflight(gpu=gpu("NVIDIA A100-SXM4-40GB", 0))) == ["the GPU is a free T4"]
    r = L.preflight(gpu=lambda: None); assert red(r) == ["the GPU is a free T4"] and "nvidia-smi could not be read" in r[9][2]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): ok = L.print_preflight(L.preflight(gpu=gpu("Tesla T4", 9000)))
    o = buf.getvalue(); assert ok is False and "  FAIL  the GPU is a free T4: Tesla T4, 9000 of 16000 MiB in use (at most 500 before the engine is loaded)\n" in o and o.count("  ok    ") == 9 and o.endswith("  RED: do not run the next cells. Send me this printout.\n")


def test_the_gpu_is_read_from_nvidia_smi_without_torch_and_an_unreadable_answer_is_none():
    seen = []
    def fake(out, code=0):
        def run(cmd, **kw): seen.append((cmd, kw)); return types.SimpleNamespace(returncode=code, stdout=out)
        return run
    assert L.gpu_state(fake("Tesla T4, 3, 15360\n")) == dict(name="Tesla T4", used_mib=3, total_mib=15360) and seen[0][0] == ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"]
    assert seen[0][1] == dict(capture_output=True, text=True, timeout=30)
    assert L.gpu_state(fake("Tesla T4, 3.0, 15360.0\nTesla T4, 9000, 15360\n")) == dict(name="Tesla T4", used_mib=3, total_mib=15360), "the first GPU is the one read"
    assert L.gpu_state(fake("Tesla T4, 3, 15360\n", code=9)) is None and L.gpu_state(fake("")) is None and L.gpu_state(fake("   \n")) is None
    assert L.gpu_state(fake("No devices were found\n")) is None and L.gpu_state(fake("Tesla T4, x, y\n")) is None and L.gpu_state(fake("Tesla T4, 3\n")) is None
    def broke(cmd, **kw): raise FileNotFoundError("nvidia-smi")
    assert L.gpu_state(broke) is None
    src = inspect.getsource(L.gpu_state); assert "torch" not in src.replace("`torch`", "").replace("torch is", "").split('"""')[2]
    assert L.gpu_state() is None or isinstance(L.gpu_state(), dict), "the real call on this machine: no GPU is None, a GPU is a dict"


# ---------------------------------------------------------------- 6 the locked page
def _h(text): return hashlib.sha256((text.rstrip("\n") + "\n").encode()).hexdigest()


def _head():
    text = open(PAGE).read(); assert text.count(RESULT_MARK) <= 1
    return text[:text.index(RESULT_MARK)] if RESULT_MARK in text else text


def test_the_page_is_locked_and_carries_the_seventeen_lines_the_rule_the_pins_the_cells_and_the_thresholds_the_code_uses():
    head = _head(); assert _h(head) == PAGE_SHA256, "the predictions and reading rules must not change after the run; later notes go below them"
    for i, (tag, line) in enumerate(L.LL_LINES, 1): assert f"\n{i}. [{tag}] {line} (" in head, i
    probs = [int(m.group(1)) for m in re.finditer(r"\n\d+\. \[\w+\] .*\((\d+) %\)(?=\n)", head)]; assert probs == PROBS and len(probs) == 17, probs
    for needle in ("**Headline** (lines 8 and 9)", "Lines 2, 3, 4 and 5 all reading yes: about **65 %**", "Lines 8 and 9 both reading yes: about **45 %**", "Line 10 yes as well (no slip at all in `LN` before step 67, or only a very late one): about **35 %**", "All 17 yes: under **3 %**"):
        assert needle in head, needle
    for name, sha in PINS.items(): assert sha in head, name
    assert TYPED_MODULE_SHA256 in head and f"`{REGEX_SHA256}`" in head and GRAMMAR_SHA["LN"] in head and GRAMMAR_SHA["L8"] in head and "`python/axi/experiments/provider_side_long.py`" in head and "Concept-map item 112" in head and "and it edits none of them" in head and "**there is no retry in any cell.**" in head
    for row in ("| `L0` | 192 to 255 | S (exact state each step) | capped | the control, same session |", "| `LN` | 192 to 255 | S | named | `L0` |", "| `L8` | 192 to 199 | S | named | `LC`, the same 8 jobs |", "| `LC` | 192 to 199 | KC (whole chat kept, cache on) | named | `L8` |"): assert row in head, row
    for lab, (K, first, arm, shape) in L.CELLS.items(): assert (f"| `{lab}` | {first} to {first + K - 1} | " in head), lab
    th = ("line 6 the mean prompt tokens of steps 76 to 100 at most `1.3` times those of steps 1 to 25 in `L0`; line 7 the sum of the corrupted steps (the held graph is not well formed) over all cells is 0; line 8 the exact share of `LN` over steps 1 to 30 at least `0.995`; "
          "line 9 the exact share of `LN` over all steps at least `0.99`; line 10 over steps 67 to 100 at least `0.99`; line 11 the share of `LN` at least `0.03` above that of `L0`; line 12 the share of `L0` over steps 67 to 100 strictly below its share over steps 1 to 33; "
          "line 13 every job of `LC` stopped with the reason \"context\" and has fewer than 100 steps; line 14 the share of `LC` at least that of `L8` minus `0.01`; line 15 the exact steps per GPU hour of `LC` lower than those of `L8`; "
          "line 16 the seconds per step of `LN` at most `1.15` times those of `L0`; line 17 the exact steps per GPU hour of `LN` higher than those of `L0`")
    assert th in head, th
    assert (L.T_FLAT, L.T_REPEAT, L.T_HIGH, L.T_GAIN, L.T_KEPT_TOL, 1 + L.T_TIME) == (1.3, 0.995, 0.99, 0.03, 0.01, 1.15) and (L.WIN_FIRST30, L.WIN_EARLY, L.WIN_LATE) == ((1, 30), (1, 33), (67, 100)) and L.STEPS == 100
    assert "line 5 every job of `L0`, `LN` and `L8` has 100 log entries and no stop" in head and "a \"window\" counts the steps from 1 and includes both ends" in head and "\"accepted by the strict parser\" is `strict_parse(text) is not None`" in head
    assert "at least 99.5 percent were accepted by the strict parser" in head and "(a kept chat stops before its next prompt would pass 7,600 tokens)" in head
    for needle in ("**The rule that decides what the Result may claim.**", "Line 2, 3, 4 or 5 reads NO: no claim is made from this run until the cause is found.", "That the 100 percent comes back is claimed from line 8, what happens over 100 steps from lines 9 to 12, what the kept chat does from lines 13 to 15, and a statement about time or cost only from lines 16 and 17",
                   "never from a cell against another run's (another Colab session)", "Never a multiple of capacity, never a count of GPUs saved, never dollars or a percentage of cost, never \"100 percent\" for a grammar, never \"proven\"", "never a statement about requests in free words",
                   "never a statement about a chat with a summary or a window", "**for the named grammar, in the same sentence, that the model writes the numbers and that the request is otherwise copied into the reply.**", "**Nothing on this page promises 100 percent.**",
                   "*A cell that did not run:* its lines read \"not run\", never NO.", "A request in free words is a different test and is not run here.", "A \"NO\" on a claim line is a finding about this setup and goes into the article as such.",
                   "*Line 1 NO, the pre-flight RED, or the safety cell says STOP:* nothing is run or read.", "a retry on top of a grammar (no cell has one)", "I have not read a reply to any of them (none exists)."):
        assert needle in head, needle
    assert head.count("\n# ") == 0 and head.startswith("# Test 19: the serving side a sixth time, with jobs of 100 steps (written BEFORE the run, 20 Oct 2026)") and head.rstrip("\n").endswith("\n---")


def test_the_figures_the_page_quotes_from_the_fifth_runs_file_and_the_estimates_it_makes_are_the_ones_in_the_saved_files_and_in_the_arithmetic():
    head = _head(); assert _sha(TYPED_FILE) == TYPED_SHA256; cells = {c["label"]: c for c in typed_run()["cells"]}
    exact = {k: sum(1 for s in c["sessions"] for x in s["log"] if x["exact"]) for k, c in cells.items()}; steps = {k: sum(len(s["log"]) for s in c["sessions"]) for k, c in cells.items()}
    assert exact == {"F0": 1833, "FK": 1835, "FKA": 1920, "FKT": 1839, "FN": 1920, "SK": 1906, "SN": 1920} and set(steps.values()) == {1920}
    sps = {k: cells[k]["wall_seconds"] / steps[k] for k in ("F0", "FN")}; eph = {k: 3600 * exact[k] / cells[k]["wall_seconds"] for k in ("F0", "FN")}
    assert f"{sps['F0']:.4f}" == "0.1168" and f"{sps['FN']:.4f}" == "0.1166" and f"{eph['F0']:,.0f}" == "29,437" and f"{eph['FN']:,.0f}" == "30,882" and round(100 * exact["F0"] / 1920, 1) == 95.5 and round(100 * exact["FK"] / 1920, 1) == 95.6
    for s in ("held the right state on 1,920 of 1,920 steps in two cells (`FN`, the grammar that names the coordinates, and `FKA`, the kind grammar with the target line at the top and the end), against 1,833 (95.5 %) for the capped control `F0` and 1,835 (95.6 %) for the kind grammar `FK`",
              "`FN` took 0.1166 seconds per step against 0.1168 for `F0`", "the first slips of the five jobs of `F0` were at steps 2, 9, 14, 2 and 11, and the state was off on 87 steps in all", "the fifth run's `F0` 0.1168 seconds per step and 29,437 exact steps per GPU hour, `FN` 0.1166 and 30,882",
              "`FN` and `SN` together had no slip in 3,840 steps", "below 0.08 % with 95 % confidence, which allows up to five slips in the 6,400 steps of `LN`", "the second run's `KC64` at 30 steps took 10,331 seconds", "the fifth run's `FN` had 427 in 30 steps"):
        assert s in head, s
    firsts = [next(x["t"] for x in s["log"] if not x["exact"]) for s in cells["F0"]["sessions"] if any(not x["exact"] for x in s["log"])]; assert firsts == [2, 9, 14, 2, 11] and 1920 - exact["F0"] == 87
    assert exact["FN"] + exact["SN"] == 3840 and 3 / 3840 < 0.0008 and abs(6400 * 3 / 3840 - 5.0) < 1e-9
    assert round(next(c for c in json.load(open(os.path.join(RES, "axi_provider_constrained.json")))["cells"] if c["arm"] == "KC" and c["K"] == 64)["wall_seconds"]) == 10331
    # the estimate: a slip stays; the chance per step from the five jobs of F0 (steps at risk until the first slip), 100 steps, 64 jobs
    risk = sum(firsts) + (64 - len(firsts)) * 30; h = len(firsts) / risk; keep = (1 - h) ** 100; share = 1 - sum(1 - (1 - h) ** t for t in range(1, 101)) / 100
    assert 15 <= 64 * (1 - keep) <= 16 and 0.86 < share < 0.88 and 12 <= 100 * (1 - share) <= 13.5, (64 * (1 - keep), share)
    assert "about 15 of 64 capped jobs would slip and `L0` would hold the right state on about 87 % of the steps" in head and "the gap to `LN` is between twelve and thirteen points" in head
    # the stand-in the page describes (a model of a prefix cache, never a result)
    S = L.summarize_all(good()); assert f"{S['L0']['prompt_first_quarter']:.0f}" == "389" and f"{S['L0']['prompt_last_quarter']:.0f}" == "380" and f"{S['LC']['prompt_first_quarter']:,.0f}" == "1,870" and f"{S['LC']['prompt_last_quarter']:,.0f}" == "6,896" and S["LC"]["longest_prompt"] == 7557
    assert min(S["LC"]["steps_per_job"]) == 24 and max(S["LC"]["steps_per_job"]) == 56 and f"{S['L0']['prompt_last_quarter'] / S['L0']['prompt_first_quarter']:.2f}" == "0.98"
    for s in ("the mean prompt of `L0` is 389 tokens in the first quarter of the steps and 380 in the last", "the kept chat's prompt goes from 1,870 to 6,896 tokens (the longest 7,557) and its eight jobs stop between steps 24 and 56", "the stand-in gives a ratio of 0.98",
              "The 6,400 requests are 2,158 add-service, 2,059 add-edge and 2,183 delete.", "The named expressions of `LN` are 835 different ones", "All 64 jobs reach 100 steps", "its first 30 requests are the 30-step job's (checked for all 64 jobs)"):
        assert s in head, s
    assert "It follows `PROVIDER_SIDE_TYPED_TEST.md` (item 111) and the Result at the end of that page." in head and len(L.CELLS) == 4 and "14,400 requests" in head
    assert "`L8` and `LC` `9603a4c10ec798e0d8098b00f02fd15831a79f4fcd58d8beafbce6463e1c3ddb`" in head and "`L0` is the capped expression `6f4e5a99…`" in head


# ---------------------------------------------------------------- 7 the notebook
NOTEBOOK = os.path.join(ROOT, "python", "colab", "axi_provider_side_long.ipynb")
def _nb(): return json.load(open(NOTEBOOK))
def _srcs(): return ["".join(c["source"]) for c in _nb()["cells"]], [c["cell_type"] for c in _nb()["cells"]]


def test_the_notebook_installs_the_pinned_engine_fixes_the_stream_runs_the_preflight_before_the_load_checks_runs_the_four_cells_in_order_and_puts_them_together():
    srcs, kinds = _srcs(); code = [s for s, k in zip(srcs, kinds) if k == "code"]; text = "\n".join(srcs); nb = _nb()
    import ast
    for s in code: ast.parse("\n".join(l for l in s.splitlines() if not l.lstrip().startswith("!")))
    for needle in ("git clone -q --depth 1 https://github.com/gaurshalendra2412/axi-runtime", "provider_side_long.py", "provider_side_typed.py", "q_measure.py", "L.preflight()", "L.print_preflight(res)", "assert green", "T.TypedVLLMEngine()", "eng.warm_up()", "L.run_cell(eng, label, progress=progress)",
                   "L.save_cell", "store.persist", "L.merge_files", "L.verdict_long", "L.print_cells", "L.print_windows", "L.print_misses", "L.print_verdict", "Q.render_table", "Q.q_row", "files.download", "open_store", "docs/PROVIDER_SIDE_LONG_TEST.md", "Do not paste any keys",
                   "update 67 commands", "17 reading-rule lines", "axi_provider_l_", "STOP:", "for label in L.ORDER: go(label)", "L.run_sessions(eng, [T.TypedSession(L.SEED * 1000 + e, 3, \"old\", \"named\") for e in (192, 193, 194)])", "L.run_sessions(eng, [L.KeptChat(L.SEED * 1000 + e, 3, \"named\") for e in (192, 193, 194)])"):
        assert needle in text, needle
    assert text.count("Do not paste any keys") >= 2 and "!pip -q install vllm==0.31.0" in [x.strip() for x in code] and "!pip -q install vllm" not in [x.strip() for x in code] and "open_store(\"/content/out\", \"axi_provider_long\")" in text
    fix = [i for i, (s, k) in enumerate(zip(srcs, kinds)) if k == "code" and "fileno" in s]; pre = [i for i, (s, k) in enumerate(zip(srcs, kinds)) if k == "code" and "L.preflight()" in s]; load = [i for i, (s, k) in enumerate(zip(srcs, kinds)) if k == "code" and "T.TypedVLLMEngine()" in s]
    assert len(fix) == len(pre) == len(load) == 1 and fix[0] < pre[0] < load[0], "the stream fix, then the pre-flight (the GPU must still be free), then the load"
    ran = []; ns = dict(go=lambda label: ran.append(label), L=L)
    for s in code:
        if re.fullmatch(r"for label in L\.ORDER: go\(label\)\s*", s.strip()): exec(s, ns)
    assert ran == L.ORDER == ["L0", "LN", "L8", "LC"], ran
    assert nb["metadata"]["accelerator"] == "GPU" and nb["nbformat"] == 4 and kinds[0] == "markdown" and kinds.count("code") == 10 and nb["metadata"]["kernelspec"]["display_name"] == "Python 3" and nb["metadata"]["kernelspec"]["name"] == "python3"
    head = srcs[0]
    for needle in ("T4", "80,000 tokens", "Qwen2.5-7B-Instruct", "AWQ", "vLLM 0.31.0", "55 to 70 minutes", "Keep this browser tab open", "Choose the T4 even if your plan offers a faster GPU", "jobs **100 steps** long", "jobs 192 to 255", "7,600 tokens", "no retry in any cell", "docs/PROVIDER_SIDE_LONG_TEST.md", "1,920 of 1,920 steps", "95.5 %", "the four cells about 35 minutes"):
        assert needle in head or needle in text, needle
    heads = [x.splitlines()[0] for x, k in zip(srcs, kinds) if k == "markdown" and x.startswith("###")]
    assert [h.split(":")[0] for h in heads] == ["### Cell 1", "### Cell 2", "### Cell 3", "### Cell 4", "### Cell 4b", "### Cell 5", "### Cell 6", "### Cell 7", "### Cell 8", "### Cell 9", "### Done"], heads
    for needle in ("### Cell 3: install the engine, version 0.31.0 (about 5 minutes)", "### Cell 5: PRE-FLIGHT CHECK (about 10 seconds). Run it BEFORE Cell 6, never after", "**If it says RED, do not run the next cells: send me the printout**", "### Cell 6: load the model and the engine with one expression per request",
                   "### Cell 7: SAFETY CHECK (about 2 minutes). 3 jobs of 3 steps of the exact-state arm and 3 jobs of 3 steps of the kept chat, both with the grammar that names the coordinates.", "**A wrong answer is not a STOP**", "### Cell 8: the four cells, in this order: `L0` and `LN` (jobs 192 to 255, 100 steps",
                   "then `L8` and `LC` (jobs 192 to 199", "### Cell 9: put the finished cells together, print the table, the windows, the first misses, the 17 reading-rule lines and Q = O / M of every cell, and download your file.", "finish the update 67 commands (git push) first",
                   "### Cell 2: get the code from GitHub. If it stops with 'the code is not on GitHub yet', finish the update 67 commands in Codespaces (git push) and run this cell again", "run Cells 1 to 6 again (do not forget Cell 4b)"):
        assert needle in text, needle
    assert text.count("finish the update 67 commands (git push) first") == 3, "one for each file the cell looks for"


def test_the_notebook_cell_runner_skips_finished_cells_and_starts_a_cell_the_last_session_cut_off_again_from_its_beginning():
    srcs, kinds = _srcs(); runner = next(s for s, k in zip(srcs, kinds) if k == "code" and "def go(label):" in s); body = runner[runner.index("def go(label):"):]
    class Crash(Exception): pass
    class Dying(T.ScriptedStandIn):
        def __init__(self, die_at): super().__init__(); self.n = 0; self.die_at = die_at
        def generate_round(self, requests, reset_cache=False):
            self.n += 1
            if self.n == self.die_at: raise Crash()
            return super().generate_round(requests, reset_cache)
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as remote:
        sys.path.insert(0, os.path.join(HERE, ".."))
        from colab.colab_persist import Store
        shim = types.SimpleNamespace(CELLS=L.CELLS, run_cell=lambda e, label, progress=None: L.run_cell(e, label, steps=3, progress=progress), save_cell=L.save_cell, summarize_cell=L.summarize_cell)
        def session(engine):
            store = Store(d, remote); ns = dict(L=shim, eng=engine, store=store, os=os, time=time, json=json); exec(body, ns); return ns["go"]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            go = session(Dying(die_at=2))
            try: go("L0")
            except Crash: pass
            assert os.listdir(d) == [] and os.listdir(remote) == []                                                       # nothing half-finished is kept: the cell starts again from its beginning
            go = session(T.ScriptedStandIn()); go("L0"); go("LC"); go("L0")
        out = buf.getvalue()
        assert out.count("already finished, skipping") == 1 and out.count("done:") == 2 and "=== L0: jobs 192 to 255, arm S, grammar capped" in out and "=== LC: jobs 192 to 199, arm KC, grammar named" in out
        s = L.summarize_cell(L.run_cell(T.ScriptedStandIn(), "L0", steps=3)); f = lambda x, d=1: "-" if x is None else f"{x:.{d}f}"; pc = lambda x: "-" if x is None else f"{x * 100:.1f}"
        want = (f"done: {s['steps_done']} steps, {s['wall_seconds']:.0f} s, {f(s['seconds_per_step'], 3)} s/step, computed prompt tokens per step {f(s['computed_per_step'], 0)}, exact steps {s['exact_share'] * 100:.1f} %, steps 1 to 30 {pc(s['first30'])} %, steps 67 to 100 {pc(s['late'])} %, "
                f"jobs exact to the end {s['jobs_exact_to_end']}, jobs stopped {s['stopped_jobs']}, outside the language {s['out_of_language']}, refused by the strict parser {s['unparsed']}, {s['distinct_regexes']} expressions, exact steps per hour {f(s['exact_per_hour'], 0)}")
        assert want + "\n" in out and "steps 67 to 100 - %" in out, (want, out)
        assert sorted(os.listdir(remote)) == ["axi_provider_l_L0.json", "axi_provider_l_LC.json"] and sorted(os.listdir(d)) == sorted(os.listdir(remote))


def test_the_notebooks_last_cell_puts_the_cells_together_prints_the_windows_the_misses_the_lines_and_q_and_offers_the_download():
    srcs, kinds = _srcs(); cell = next(s for s, k in zip(srcs, kinds) if k == "code" and "L.merge_files" in s)
    sys.path.insert(0, os.path.join(HERE, ".."))
    from colab.colab_persist import Store
    import axi.experiments.q_measure as Qm
    got = []; gc = types.ModuleType("google.colab"); gc.files = types.SimpleNamespace(download=lambda p: got.append(p)); g = types.ModuleType("google"); g.colab = gc
    saved = {k: sys.modules.get(k) for k in ("google", "google.colab")}
    sys.modules["google"], sys.modules["google.colab"] = g, gc
    try:
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as remote:
            store = Store(d, remote)
            for c in good(): L.save_cell(c, store.path(f"axi_provider_l_{c['label']}.json"))
            buf = io.StringIO(); ns = dict(L=L, Q=Qm, store=store, glob=glob, os=os)
            with contextlib.redirect_stdout(buf): exec(cell, ns)
            out = buf.getvalue(); assert "4 cells:" in out and "-- provider-side test, jobs of 100 steps" in out and out.count("[claim]") == 12 and out.count("[guard]") == 3 and out.count("[instrument]") == 2 and "first slips of jobs in steps 1-33 / 34-66 / 67-100: 3 / 0 / 0" in out
            assert ("never: a multiple of capacity" in out) != ("NO claim is made from this run" in out) and "**Q = O / M of the cells of this run**" in out and all(f"| {lab} |" in out for lab in L.ORDER) and "exact steps per GPU hour" in out
            assert got == [store.path("axi_provider_long.json")] and os.path.exists(os.path.join(remote, "axi_provider_long.json"))
            m = json.load(open(store.path("axi_provider_long.json"))); assert [c["label"] for c in m["cells"]] == L.ORDER and m["module_sha256"] == L.fingerprint()
            os.remove(store.path("axi_provider_l_LC.json")); os.remove(store.path("axi_provider_l_L8.json")); buf = io.StringIO(); ns = dict(L=L, Q=Qm, store=store, glob=glob, os=os)
            with contextlib.redirect_stdout(buf): exec(cell, ns)
            out = buf.getvalue(); assert "2 cells:" in out and out.count("  not run  ") == 3 and "  NO       " not in out and "(14 of 17 lines could be read" in out and "lines 2, 3, 4 and 5 read yes" in out, out
            assert [c["label"] for c in json.load(open(store.path("axi_provider_long.json")))["cells"]] == ["L0", "LN"]
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


def test_the_preflight_cell_stops_the_notebook_when_it_is_red_and_lets_it_go_on_when_it_is_green():
    srcs, kinds = _srcs(); cell = next(s for s, k in zip(srcs, kinds) if k == "code" and "L.preflight()" in s); real = L.gpu_state
    def run():
        buf = io.StringIO(); ns = {}
        with contextlib.redirect_stdout(buf): exec(cell, ns)
        return buf.getvalue(), ns
    try:
        L.gpu_state = lambda run=None: dict(name="Tesla T4", used_mib=3, total_mib=15360); out, ns = run()
        assert "GREEN: nothing found" in out and ns["green"] is True and out.count("  ok    ") == 10
        for state in (None, dict(name="Tesla T4", used_mib=9000, total_mib=15360), dict(name="Tesla V100", used_mib=0, total_mib=16000)):
            L.gpu_state = lambda run=None, s=state: s; buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf): exec(cell, {})
                assert False, "the cell must stop"
            except AssertionError as ex: assert "PRE-FLIGHT IS RED: do not run the next cells" in str(ex) and "RED: do not run the next cells. Send me this printout." in buf.getvalue() and "FAIL  the GPU is a free T4" in buf.getvalue()
    finally: L.gpu_state = real


def test_the_load_cell_prints_the_engines_numbers_and_the_base_expression_and_defines_the_runner():
    srcs, kinds = _srcs(); cell = next(s for s, k in zip(srcs, kinds) if k == "code" and "T.TypedVLLMEngine()" in s)
    mods, calls = T124._fake_vllm("module")
    with T124._installed(mods), tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as remote:
        sys.path.insert(0, os.path.join(HERE, ".."))
        from colab.colab_persist import Store
        ns = dict(json=json, os=os, time=time, store=Store(d, remote)); buf = io.StringIO()
        with contextlib.redirect_stdout(buf): exec(cell, ns)
        out = buf.getvalue()
        assert f"constraint: regex_per_request base expression with a name of at most 16 characters {REGEX_SHA256[:16]} (each request carries its own expression)" in out
        assert "cache size the engine will use: 80000 tokens ( override )" in out and callable(ns["go"]) and isinstance(ns["eng"], T.TypedVLLMEngine) and ns["L"] is L and ns["T"] is T and calls["resets"] >= 1 and '"constraint"' not in out and '"kind": "vllm"' in out


def test_the_safety_cell_says_ok_on_a_working_engine_and_stop_for_each_thing_that_can_be_wrong_and_a_wrong_answer_is_not_a_stop():
    srcs, kinds = _srcs(); cell = next(s for s, k in zip(srcs, kinds) if k == "code" and "smoke = " in s)
    def run(engine):
        buf = io.StringIO(); ns = dict(T=T, L=L, eng=engine)
        with contextlib.redirect_stdout(buf): exec(cell, ns)
        return buf.getvalue()
    out = run(T.TypedStandIn()); last = out.strip().splitlines()[-1]
    assert last.startswith("OK: cached counts are reported") and "STOP" not in out and "exact state\n" in out and "kept chat\n" in out and out.count("  step 1: prompt") == 2 and out.count("  step 3: prompt") == 2
    class NoCount(T.TypedStandIn):
        def generate_round(self, requests, reset_cache=False): reps, dt = super().generate_round(requests, reset_cache); [setattr(r, "n_cached", None) for r in reps]; return reps, dt
    assert "STOP: this vLLM version does not report cached tokens" in run(NoCount())
    def says(text, n_new):
        class P(T.TypedStandIn):
            def generate_round(self, requests, reset_cache=False):
                reps, dt = super().generate_round(requests, reset_cache)
                for r in reps: r.text = text; r.n_new = n_new
                return reps, dt
        return P()
    assert "STOP: a reply of the safety cell was refused by the strict parser" in run(says("Sure, here is the delta", 20))
    assert "STOP: a reply of the safety cell is outside the expression of its own request" in run(says("DEL[1,2]", 4))
    class Long(T.TypedStandIn):                                                                                 # inside the language and parsed, but it used all 96 new tokens
        def generate_round(self, requests, reset_cache=False): reps, dt = super().generate_round(requests, reset_cache); [setattr(r, "n_new", 96) for r in reps]; return reps, dt
    assert "STOP: a reply of the safety cell used all 96 new tokens" in run(Long())
    class Wrong(T.TypedStandIn):                                                                                # a wrong answer that is inside the language and ends: not a STOP
        def generate_round(self, requests, reset_cache=False):
            reps, dt = super().generate_round(requests, reset_cache)
            for rq, r in zip(requests, reps):
                if r.text.startswith("ADD") and r.text[-2].isdigit():                                   # a different number in the same place: inside the language, not the oracle's reply
                    for d in "0123456789":
                        alt = r.text[:-2] + d + r.text[-1]
                        if alt != r.text and re.fullmatch(rq.regex, alt): r.text = alt; break
            self.changed = getattr(self, "changed", 0) + sum(1 for rq, r in zip(requests, reps) if r.text != rq.stand_in_reply); return reps, dt
    e = Wrong(); o = run(e); assert e.changed > 0 and o.strip().splitlines()[-1].startswith("OK:") and "STOP" not in o
    class Garbage(T.TypedStandIn):
        def generate_round(self, requests, reset_cache=False): reps, dt = super().generate_round(requests, reset_cache); reps[0].text = "Sure, here is the delta: " + reps[0].text; return reps, dt
    assert "STOP: a reply of the safety cell was refused by the strict parser" in run(Garbage())


# ---------------------------------------------------------------- 8 the front pages
FRONT_ITEM = ("112. **The serving side a sixth time: jobs of 100 steps, the first long ones; the exact state with and without the grammar that names the coordinates, and a kept chat under the same grammar (20 Oct)**",
              "**Status: built and checked here, then run once on Colab (see Run, at the end of this item); the page `PROVIDER_SIDE_LONG_TEST.md` was hash-locked before any run (sha256 " + PAGE_SHA256[:16] + "... of the text above the line at its end), and its Result is below that line.**",
              "the Result was written as a later update with a test of its own (134)", "four cells, no retry anywhere, on jobs 192 to 255, which no saved run has used (read from the saved files)", "each job 100 steps long (its first 30 requests are the 30-step job's, checked for all 64)",
              "`KeptChat` adds the grammar and decides nothing", "the headline is lines 8 and 9", "both yes about 45 %; lines 2 to 5 all yes about 65 %; all 17 yes under 3 %", "the oracle inside all 14,400 requests",
              "`F0` slipped on 5 of 64 jobs within 30 steps (first slips at steps 2, 9, 14, 2 and 11, 87 steps off)", "(the rule of three: below 0.08 % a step)", "about 15 of 64 capped jobs would slip within 100 steps and `L0` would hold the right state on about 87 % of the steps",
              "its eight jobs stop between steps 24 and 56 on the 7,600-token rule", "**What it cannot do:** it promises no 100 %", "the host still reads the request from its sentence (a request in free words is another test)", "stops because the context of this setup is 8,192 tokens",
              "**Deferred, not built:** a reader for requests in free words (to be tested on its own first)", "**Credit:** the architecture is Rajnish's; the cells, the code and every measurement are the owner's, with an AI assistant as a tool", "a cell that was not run reading \"not run\"",
              "the pre-flight check with each check made to read NO", "**Not shown:** anything about another engine or GPU, or about what a model writes beyond 100 steps (this is the one run there is)", "nothing here is proven")
FRONT_ROW = ("Provider-side serving test, sixth run: jobs of 100 steps (`PROVIDER_SIDE_LONG_TEST.md`, `experiments/provider_side_long.py`, `colab/axi_provider_side_long.ipynb`) |", "**Built and checked here; run once on Colab; the Result is on the page.**", "(sha256 " + PAGE_SHA256[:16] + "... of the locked text)",
             "Every serving run so far had jobs of 30 steps.", "The headline is lines 8 and 9", "(both yes about 45 % on the page). What the Result claims is in the run text at the end of this row.", "never a multiple of capacity, a count of GPUs saved, dollars, \"100 percent\" or \"proven\"", "**No route in this design promises 100 %.**",
             "(three persistent capped slips cost 257 steps there)", "the 64 new jobs against every saved file (192 to 255 are in none)", "Not shown: anything about what a model writes beyond 100 steps (this is the one run there is)", "a retry on top of a grammar")
FRONT_NEXT = "Next free: row 172, item 129, test 155, question 121"


def test_the_front_pages_say_that_the_test_is_built_and_not_run_what_it_expects_and_what_it_does_not_show():
    cm = open(os.path.join(ROOT, "docs", "RAJNISH_CONCEPT_MAP.md")).read(); ve = open(os.path.join(ROOT, "docs", "VERIFIED.md")).read(); n_tests = sum(1 for n, f in globals().items() if n.startswith("test_") and callable(f))
    assert "PROVIDER_SIDE_LONG_TEST.md" in cm and "PROVIDER_SIDE_LONG_TEST.md" in ve
    item = cm[cm.index("112. **"): cm.index("113. **")]; k = ve.index("Provider-side serving test, sixth run"); row = ve[k: ve.index("\n", k)]
    for s in FRONT_ITEM: assert s in item, s[:100]
    for s in FRONT_ROW: assert s in row, s[:100]
    for text in (item, row):
        assert MUTATION_SENTENCE in text and "380" in text and "317" in text and "63 survived" in text
        assert "@@" not in text and f"(test 133, {n_tests} tests)" in text and "mutation check of test 133: " in text and "capacity" in text and "Result" in text and "RUN (" not in text and "Status: RUN" not in text and "no run yet" not in text and "not run yet" not in text and "nothing has been run" not in text
        assert "proven" in text and "dollars" in text and "energy" in text and "100" in text
    assert "\n129. **" not in cm and "\n114. **" in cm and "\n113. **" in cm and "\n112. **" in cm and cm.count("112. **") == 1
    assert FRONT_NEXT in cm and "test 133 checks the 100-step serving test, item 112, and test 134 its Result" in cm and cm.count("Next free:") == 1
    assert cm.index("111. **") < cm.index("112. **") < cm.index("Next free: row") and ve.index("Provider-side serving test, fifth run") < k < ve.index("\n", k)
    for name in ("test_provider_side.py", "test_provider_side_constrained.py", "test_provider_side_bounded.py", "test_provider_side_bounded_result.py"):
        t = open(os.path.join(HERE, name)).read(); assert "row 170, item 112, test 133" not in t and "row 172, item 129, test 155" in t, name
    for name in ("test_provider_side_retry.py", "test_provider_side_typed.py", "test_provider_side_typed_result.py"):                      # these three name the old counter in the assertion that moves it
        assert "row 172, item 129, test 155, question 121" in open(os.path.join(HERE, name)).read(), name
    assert 'FRONT_NEXT = "Next free: row 172, item 129, test 155, question 121"' in open(os.path.join(HERE, "test_provider_side_retry.py")).read()


MUTATION_SENTENCE = "mutation check of test 133: run twice on the final code. Run 1 (380 small changes to the code, the page, the notebook and the tests) was caught 266 times and survived 114; the real gaps among those were closed with new tests. Run 2 (380 new changes, another seed) was caught 317 times and 63 survived. The 63 were grouped by kind, not each traced: numbers one step off an edge, `and` turned to `or` where both sides are true throughout, one changed word in the locked page or the notebook (the page's sha256 is the lock for those), and display-only text. "


ITEM_NUMBERS = "112 100 20 20 30 133 134 192 255 100 30 30 64 64 8 17 8 9 100 1 30 64 99.5 99 100 45 2 5 65 17 3 100 14,400 4 5 64 30 2 9 14 2 11 87 3,840 0.08 15 64 100 87 1,870 6,896 24 56 7,600 100 100 8,192 64 14,400 17 100"
ROW_NUMBERS = "100 17 30 192 255 100 64 8 8 9 100 1 30 99 100 45 100 100 256 64 192 255 14,400 257 17 100"


def _front_numbers(text):
    a = text.index("mutation check of test 133:"); b = text.index("Not shown:"); text = text[:a] + text[b:]
    for marker in ("**Run (21 Oct 2026):**", "**Run:**"):                                                                          # the run text is read by tests/test_provider_side_long_result.py, which pins its numbers
        if marker in text: text = text[:text.index(marker)]
    return " ".join(re.findall(r"\d[\d,.]*\d|\d", re.sub(r"`[^`]*`", "", re.sub(r"sha256 [0-9a-f]{16}\.\.\.", "", re.sub(r"\(test 133, \d+ tests\)", "", text)))))


def test_every_number_of_the_two_front_pages_outside_the_code_spans_and_the_mutation_sentence_is_the_one_that_was_written():
    cm = open(os.path.join(ROOT, "docs", "RAJNISH_CONCEPT_MAP.md")).read(); ve = open(os.path.join(ROOT, "docs", "VERIFIED.md")).read()
    item = cm[cm.index("112. **"): cm.index("113. **")]; k = ve.index("Provider-side serving test, sixth run"); row = ve[k: ve.index("\n", k)]
    assert _front_numbers(item) == ITEM_NUMBERS and _front_numbers(row) == ROW_NUMBERS


# ---------------------------------------------------------------- 9 what the first mutation run let through
def test_the_preflight_reads_the_scripts_of_the_cells_own_jobs_and_counts_the_requests_it_finds_outside_their_expression():
    real = T.script_of; seeds = set(); real_in = T.in_language
    def spy(seed, steps): seeds.add(seed); return real(seed, steps)
    try: T.script_of = spy; assert all(ok for _, ok, _ in L.preflight(gpu=T4()))
    finally: T.script_of = real
    assert seeds == {seed_of(e) for e in range(192, 256)}, "the lengths, the prefixes, the expressions and the oracle checks all read the jobs 192 to 255 of this seed"
    deletes = sum(1 for K, first, _, _ in L.CELLS.values() for e in range(first, first + K) for st in real(seed_of(e), 100) if T.parse_request(st["instruction"])["kind"] == "delete")
    assert 2183 * 2 < deletes < 14400 and deletes == sum(1 for lab, (K, first, _, _) in L.CELLS.items() for e in range(first, first + K) for st in real(seed_of(e), 100) if T.parse_request(st["instruction"])["kind"] == "delete")
    try:
        T.in_language = lambda shape, instruction, reply: T.parse_request(instruction)["kind"] != "delete"; r = L.preflight(gpu=T4())
        assert r[4][2] == f"{14400 - deletes} of 14400 requests" and not r[4][1], r[4]
    finally: T.in_language = real_in


HEAD_NUMBERS = "19 100 20 2026 112 111 4 0.31.0 1 30 1,920 1,920 1,833 95.5 1,835 95.6 0.1166 0.1168 64 30 2 10 50 100 100 3 2 9 14 2 11 87 100 10 90 100 30 100 0.31.0 4 80,000 8,192 96 20261017 1000 3,600 256 100 256 100 30 30 64 192 255 0 63 64 127 128 191 192 255 192 255 192 199 8 192 199 35 64 8 64 100 30 10,331 1 30 30 64 100 30 256 14,400 4 500 1,000 64 100 6,400 2,158 2,059 2,183 835 427 30 256 30 100 15 64 87 3,840 0.08 95 6,400 36 389 380 1,870 6,896 7,557 24 56 7,600 0.1168 29,437 0.1166 30,882 0.31.0 4 80,000 100 20261017 192 255 1 30 1 33 67 100 3 3 96 1 5 6 17 100 3,840 1,920 1 98 2 99.5 96 92 3 0 0 78 4 256 97 5 0 8 100 96 6 0 76 100 1.3 1 25 92 7 92 8 100 1 30 64 99.5 68 9 100 99 50 10 67 100 99 40 11 100 0 80 12 0 67 100 1 33 90 13 100 7,600 93 14 8 8 80 15 8 8 92 16 15 0 88 17 0 90 8 9 100 100 2 3 4 5 65 8 9 45 10 67 35 17 3 10 14 1 98 2 92 835 3 78 6,400 4 97 14,400 5 96 6 92 0.98 7 92 0 13,440 0 360 8 68 3,840 0.67 1,920 30 22 9 50 6,400 0.38 35 0.15 10 40 67 11 80 87 12 90 13 93 24 56 14 80 15 92 16 88 835 427 17 90 82 95 6,400 70 1 33 67 100 70 20 60 80 5 85 30 45 70 80 8 2 3 4 5 8 9 64 99 100 100 99.5 1 30 8 100 100 8 9 30 9 10 11 98 100 12 12 100 13 100 7,600 14 15 8 8 16 17 835 6 7 3 2 4 2 1 17 1 2 96 3 4 256 5 100 6 76 100 1 25 7 0 8 1 30 9 10 67 100 11 12 67 100 1 33 13 100 14 15 16 17 3,600 1 2 3 4 5 100 8 100 9 12 13 15 16 17 2 3 4 5 4 7 100 1 30 67 100 1,000 100 64 100 8,192 64 4 0.31.0 13 17"


def test_every_number_of_the_locked_page_outside_the_code_spans_is_the_one_that_was_written():
    head = _head(); spans_out = re.sub(r"`[^`]*`", "", head); found = re.findall(r"\d[\d,.]*\d|\d", spans_out)
    assert " ".join(found) == HEAD_NUMBERS, "a number of the page changed (its hash is pinned above; this reads them one by one)"
    assert len(found) == 427


def test_the_notebooks_prose_carries_the_numbers_the_run_depends_on():
    srcs, kinds = _srcs(); text = "\n".join(srcs)
    for needle in ("the grammar that names the coordinates) on 64 jobs", "on the first 8 of the same jobs", "on 64 jobs no run has seen (jobs 192 to 255)", "installing the engine about 5 minutes, loading the model 5 to 10 minutes", "downloads about 5.5 GB once; about 5 to 10 minutes",
                   "(about 1 second). vLLM 0.31.0 asks the notebook for a file number", "gives the notebook's output the usual numbers 1 and 2", "Run it once after every restart, before Cell 6", "that every job has 100 steps", "the pre-flight check and the safety check 3 minutes",
                   "(the longest takes about 13 minutes)", "A kept chat stops when its next prompt would pass 7,600 tokens", "In all about 55 to 70 minutes", "cache fixed at **80,000 tokens**", "(about 10 seconds)", "(about 2 minutes)", "About 35 minutes in all"):
        assert needle in text, needle


if __name__ == "__main__":
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]; bad = 0
    for n, f in fns:
        try: f(); print("ok  ", n)
        except Exception: bad += 1; print("FAIL", n); traceback.print_exc()
    print(f"{len(fns) - bad}/{len(fns)} passed"); sys.exit(1 if bad else 0)
