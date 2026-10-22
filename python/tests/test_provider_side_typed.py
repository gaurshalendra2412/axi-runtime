"""Engineering support for docs/PROVIDER_SIDE_TYPED_TEST.md (concept-map item 111): the exact-state cells of the provider-side serving test with the grammar of the reply chosen by the type of the request
(and the anchor line of item 110 in the prompt). Checks that need no GPU and no model:
  1. nothing the test imports is edited (sha256 of eight files), the settings are the third run's, and the new session adds two methods and nothing else;
  2. the host reads the three request sentences of the jobs and nothing else; the target line and the three prompts, word for word;
  3. the oracle's own reply is inside every language for all 192 jobs (17,280 requests) and the languages accept and refuse what the page says; kind is inside capped;
  4. the eight first misses of the table: replayed from the saved files, with the lost steps, the languages and the parity check; the parity check on 5,040 saved steps;
  5. the replay rule on the third run's replies (1,860 and 1,848 steps compared), the ceilings, the cache facts and the stand-in estimates the page quotes;
  6. the session is the pinned session with a prompt and an expression added; the vLLM adapter puts one expression into every request;
  7. the harness on a test double that obeys the grammar: the seven cells, the merge (what it refuses), the probe;
  8. the 17 lines on a good run built from the third run's own replies, each line at its edge, 'not run' for a missing cell, the rule about what may be claimed, the printout, the command line;
  9. the locked page, the figures it quotes, the notebook and its safety cell, the front pages.
What it does NOT show: anything about a real engine, a real GPU, or what a model writes when a grammar leaves it only the numbers."""
import contextlib, copy, glob, hashlib, inspect, io, json, os, re, statistics, subprocess, sys, tempfile, time, types

sys.path.insert(0, os.path.dirname(__file__))
import axi.engine.gate as G
import axi.engine.diagnostics as D
import axi.experiments.agent_loop as A
import axi.experiments.cascade as K
import axi.experiments.long_horizon as L
import axi.experiments.provider_side as PS
import axi.experiments.provider_side_constrained as C
import axi.experiments.provider_side_bounded as B
import axi.experiments.provider_side_typed as T
import test_provider_side_constrained as T124                                       # its fake `vllm` is reused, not copied
from axi.experiments.agent_loop import strict_parse

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..", "..")
RES = os.path.join(ROOT, "docs", "results")
PINS = dict(provider_side="bfb4734e171d9244cd47d10402be46528a557095a8275d02b38481a0e2b0b1d3", constrained="03452148e28c9f31372f12cc11ef1808ac67308adff972fa19d4aeeef7c96f7d",
            bounded="7427a686d151c4294809aebaa9664d0b5fa420854817eb79e910372c3f57b6bc", gate="b6a50408788791f76bc4d4129dad84debb6e09c9c5e804673cdf04abec224ce7",
            diagnostics="255f3c11db989b645077d10706a187e2c913576538a826e6dbdb8892b014b4f2", agent_loop="d658afaa29e3bce7e1fb7f43d88281fc40ee68ca59864421a65284e9973516b6",
            cascade="b0533316bc094e80a1d68eddb7e137c7c176af9a342341288360c18d61f74fe7", long_horizon="5718435de2178de110ddbb59dd3bf32e1bec9f1fd8625a27119d789399fa20ac")
REGEX_SHA256 = "6f4e5a993f26ccd471bff13422fa622b7820858a91e05d43c2bccfbe6b864245"
GRAMMAR_SHA = {(0, "capped"): REGEX_SHA256, (0, "kind"): "69152ac964df234b99a442f0115927462040714b0ae3e1bdabba55a906f766b2", (0, "named"): "12cfe62cbbb9b8a6722fb9e4fc6288ba78cf56b16e7ad89bc8a09dfbbf1f2e46",
               (128, "capped"): REGEX_SHA256, (128, "kind"): "69152ac964df234b99a442f0115927462040714b0ae3e1bdabba55a906f766b2", (128, "named"): "c1732477bafdb53fd115fb67028ade7c9c09fc44219a521612be9c63f3eeab4e"}
THIRD_FILE = os.path.join(RES, "axi_provider_bounded.json")
THIRD_SHA256 = "1f12a846b1325f9fed4e5df2cc755b3a7fd996266cb590311230f7cf418f2733"
RETRY_FILE = os.path.join(RES, "axi_provider_retry.json")
RETRY_SHA256 = "390f7464a54352764e21203f9546bd17fc4714b09cbb0aec9390963e5d94b116"
PAGE = os.path.join(ROOT, "docs", "PROVIDER_SIDE_TYPED_TEST.md")
PAGE_SHA256 = "d4c963599e8dfe67f80f928535244ae078df5b91a8313f13b0bbc473c036d19f"
RESULT_MARK = "\n\n# Result"
PROBS = [98, 90, 80, 97, 93, 38, 85, 60, 45, 70, 50, 50, 80, 70, 15, 85, 62]
SEED = PS.SEED
# job: (step, request, what the engine answered, what the oracle wrote, inside the kind language, inside the named language, steps lost by the job, the run it was read in)
MISSES = {3: (16, "Add a service at (4,4) with value 6.", "ADD[(4,4)->(4,4):value#6]", "ADD[4,4:6]", False, False, 15, "S64"),
          4: (18, "Add a service at (0,0) with value 3.", "ADD[1,0:6]", "ADD[0,0:3]", True, False, 13, "S64"),
          17: (20, "Add a dependency edge from (0,4) to (2,3) with relation dep and weight 4.", "DEL[(2,3)->(0,4):dep#6] ADD[(0,4)->(2,3):dep#4]", "ADD[(0,4)->(2,3):dep#4]", False, False, 4, "S64"),
          19: (12, "Add a service at (3,0) with value 4.", "ADD[(3,0)->(3,0):4tworowsdown4col#4]", "ADD[3,0:4]", False, False, 19, "S64"),
          20: (30, "Add a dependency edge from (1,1) to (2,1) with relation dep and weight 3.", "ADD[(1,5)->(2,1):dep#3]", "ADD[(1,1)->(2,1):dep#3]", True, False, 1, "S64"),
          62: (12, "Add a dependency edge from (1,1) to (1,0) with relation dep and weight 9.", "DEL[(1,1)->(1,0):owns#5] ADD[(1,1)->(1,0):dep#9]", "ADD[(1,1)->(1,0):dep#9]", False, False, 1, "S64"),
          101: (4, "Add a service at (3,4) with value 2.", "ADD[(3,4)->(3,4):2alue#2]", "ADD[3,4:2]", False, False, 27, "H0"),
          122: (22, "Add a dependency edge from (0,0) to (2,2) with relation dep and weight 3.", "ADD[(0,0)->(2,0):dep#3]", "ADD[(0,0)->(2,2):dep#3]", True, False, 9, "H0")}


def raises(fn, exc=Exception):
    try: fn()
    except exc: return True
    return False


def _sha(path): return hashlib.sha256(open(path, "rb").read()).hexdigest()


_CACHE = {}


def third():
    if "third" not in _CACHE: _CACHE["third"] = json.load(open(THIRD_FILE))
    return _CACHE["third"]


def retry():
    if "retry" not in _CACHE: _CACHE["retry"] = json.load(open(RETRY_FILE))
    return _CACHE["retry"]


def tlist(): return third()["cells"]
def s64(): return next(c for c in tlist() if c["arm"] == "S" and c["K"] == 64)
def h0(): return next(c for c in retry()["cells"] if c["label"] == "H0")
def seed_of(job): return SEED * 1000 + job
def instr(job, step): return T.script_of(seed_of(job))[step - 1]["instruction"]


def replay_to(job, step, prompt="old", shape="capped"):
    """The job as the oracle's own replies leave it before `step` (1-based)."""
    s = T.TypedSession(seed_of(job), PS.STEPS, prompt, shape)
    while s.t < step - 1:
        rq = s.request(); s.accept(PS.Reply(rq.stand_in_reply, 0, 0, 0))
    return s


# ---------------------------------------------------------------- 1 nothing edited, the settings
def test_nothing_the_test_imports_is_edited_the_settings_are_the_third_runs_and_the_new_session_adds_two_methods_and_nothing_else():
    mods = dict(provider_side=PS, constrained=C, bounded=B, gate=G, diagnostics=D, agent_loop=A, cascade=K, long_horizon=L)
    for name, m in mods.items(): assert _sha(m.__file__) == PINS[name], name
    assert T.pinned_hashes() == PINS and set(PINS) == set(mods)
    assert (T.MODEL, T.STEPS, T.SEED, T.POLICY, T.HINT_RULES, T.MAX_NEW, T.CAPACITY_TOKENS) == (PS.MODEL, PS.STEPS, PS.SEED, PS.POLICY, PS.HINT_RULES, PS.MAX_NEW, PS.CAPACITY_TOKENS) == ("Qwen/Qwen2.5-7B-Instruct-AWQ", 30, 20261017, "repair_m6", False, 96, 80000)
    assert T.REGEX is B.REGEX and T.REGEX_SHA256 == B.REGEX_SHA256 == REGEX_SHA256 and T.NAME_CAP == 16 and T.SHAPES == ("capped", "kind", "named") and T.PROMPTS == ("old", "anchor", "tail")
    assert T.CELLS == {"F0": (64, 128, "old", "capped"), "FK": (64, 128, "old", "kind"), "FKA": (64, 128, "anchor", "kind"), "FKT": (64, 128, "tail", "kind"), "FN": (64, 128, "old", "named"),
                       "SK": (64, 0, "old", "kind"), "SN": (64, 0, "old", "named")} and T.ORDER == ["F0", "FK", "FKA", "FKT", "FN", "SK", "SN"] == T.cells_planned()
    own = {k for k in vars(T.TypedSession) if not k.startswith("__")}
    assert own == {"request", "accept"}, own
    assert T.TypedSession.done is PS.Session.done and issubclass(T.TypedSession, PS.Session) and issubclass(T.TRequest, PS.Request)
    assert (T.T_PARSED, T.T_GAIN, T.T_HIGH, T.T_HIGHER, T.T_FRESH_NAMED, T.T_TIME_KIND, T.T_TIME_NAMED, T.T_TIME_TOP, T.T_TIME_TAIL) == (0.995, 0.01, 0.99, 0.995, 0.995, 0.05, 0.15, 0.30, 0.20) and T.TL_HEADLINE == (6, 17) and len(T.TL_LINES) == 17
    assert T.REPLAY_PAIRS == [("FK", "F0", "kind"), ("FN", "F0", "named"), ("SK", None, "kind"), ("SN", None, "named")] and T.THIRD_S64 == 64
    assert _sha(THIRD_FILE) == THIRD_SHA256 and _sha(RETRY_FILE) == RETRY_SHA256
    assert [(j, t, x) for j, t, x in T.KNOWN_SLIPS] == [(j, MISSES[j][0], MISSES[j][2]) for j in MISSES]


# ---------------------------------------------------------------- 2 the host reads the request
def test_the_host_reads_three_request_sentences_and_nothing_else_and_the_target_line_and_the_three_prompts_are_word_for_word():
    assert T.parse_request("Add a service at (4,4) with value 6.") == dict(kind="add_node", node=("4", "4"), value="6")
    assert T.parse_request("Add a service at (10,12) with value -3.") == dict(kind="add_node", node=("10", "12"), value="-3")
    assert T.parse_request("Add a dependency edge from (0,4) to (2,3) with relation dep and weight 4.") == dict(kind="add_edge", src=("0", "4"), dst=("2", "3"), rel="dep", weight="4")
    assert T.parse_request("Add a dependency edge from (0,4) to (2,3) with relation owns_2 and weight 12.")["rel"] == "owns_2"
    assert T.parse_request("Delete the service at (2,3).") == dict(kind="delete", node=("2", "3"))
    assert T.parse_request("Add a service at (10,20) with value 10.") == dict(kind="add_node", node=("10", "20"), value="10") and T.parse_request("Add a service at (0,0) with value -30.")["value"] == "-30"
    assert T.parse_request("Add a dependency edge from (10,20) to (30,40) with relation a0 and weight 50.") == dict(kind="add_edge", src=("10", "20"), dst=("30", "40"), rel="a0", weight="50") and T.parse_request("Add a dependency edge from (0,0) to (0,0) with relation d and weight -20.")["weight"] == "-20"
    for bad in ("", "Delete the service at (2,3)", "delete the service at (2,3).", "Delete the service at (2,3). Now.", "Delete the edge from (0,4) to (2,3).", "Add a service at (4,4).", "Add a service at (4, 4) with value 6.",
                "Add a dependency edge from (0,4) to (2,3) with relation and weight 4.", "Add a dependency edge from (0,4) to (2,3) with relation d-p and weight 4.", " Add a service at (4,4) with value 6.", "Add a service at (4,4) with value 6.\n"):
        assert raises(lambda: T.parse_request(bad), ValueError), bad
    assert T.target_line("Add a service at (0,0) with value 3.") == "TARGET (0,0) ADD NODE value 3"
    assert T.target_line("Add a dependency edge from (1,1) to (2,1) with relation dep and weight 3.") == "TARGET (1,1)->(2,1) ADD EDGE dep weight 3"
    assert T.target_line("Delete the service at (4,4).") == "TARGET (4,4) DELETE NODE" and T.target_body("Delete the service at (4,4).") == "(4,4) DELETE NODE"
    content = "NODES: [0,1:2]\n\nRequest: Add a service at (0,0) with value 3."
    t = "TARGET (0,0) ADD NODE value 3"; ins = "Add a service at (0,0) with value 3."
    assert T.anchored(content, ins, "old") == content and T.anchored(content, ins, "anchor") == f"{t}\n\n{content}\n{t}" and T.anchored(content, ins, "tail") == f"{content}\n{t}"
    assert raises(lambda: T.anchored(content, ins, "middle"), AssertionError)
    kinds = {}
    for first, last, name in ((128, 191, "fresh"), (0, 63, "seen")):
        c = {}
        for e in range(first, last + 1):
            for st in T.script_of(seed_of(e)): c[T.parse_request(st["instruction"])["kind"]] = c.get(T.parse_request(st["instruction"])["kind"], 0) + 1
        kinds[name] = c
    assert kinds == {"fresh": {"delete": 681, "add_node": 649, "add_edge": 590}, "seen": {"delete": 679, "add_node": 627, "add_edge": 614}}, kinds
    toks = [len(PS._toks(T.target_line(st["instruction"]))) for e in range(128, 192) for st in T.script_of(seed_of(e))]
    assert (min(toks), max(toks), round(statistics.mean(toks), 1)) == (8, 18, 11.7)
    assert len(T.script_of(seed_of(5))) == 30 and T.script_of(seed_of(5)) is T.script_of(seed_of(5)) and len(T.script_of(seed_of(5), 7)) == 7


# ---------------------------------------------------------------- 3 the languages
def test_the_oracles_reply_is_inside_every_language_for_all_192_jobs_the_languages_accept_and_refuse_what_the_page_says_and_kind_is_inside_capped():
    n = 0; distinct = {s: {0: set(), 128: set()} for s in T.SHAPES}
    for e in range(192):
        for st in T.script_of(seed_of(e)):
            n += 1
            for shape in T.SHAPES:
                rx = T.regex_for(shape, st["instruction"]); assert re.fullmatch(rx, st["ideal"]), (e, shape, st["instruction"], st["ideal"]); assert T.in_language(shape, st["instruction"], st["ideal"])
                if e < 64: distinct[shape][0].add(rx)
                if e >= 128: distinct[shape][128].add(rx)
    assert n == 5760 and {s: (len(v[0]), len(v[128])) for s, v in distinct.items()} == {"capped": (1, 1), "kind": (3, 3), "named": (436, 427)}
    for shape in T.SHAPES:
        for first in (0, 128): assert T.grammar_sha256(shape, first, 64) == GRAMMAR_SHA[(first, shape)], (shape, first)
    assert T.regex_for("capped", "Add a service at (4,4) with value 6.") is T.REGEX and raises(lambda: T.regex_for("middle", "Add a service at (4,4) with value 6."), AssertionError)
    NODE, EDGE, DEL = "Add a service at (4,4) with value 6.", "Add a dependency edge from (0,4) to (2,3) with relation dep and weight 4.", "Delete the service at (2,3)."
    accept = {("kind", NODE): ["ADD[4,4:6]", "ADD[4,4:6] ", "ADD[9,0:-1]\n", "ADD[1,2:3]"], ("kind", EDGE): ["ADD[(0,4)->(2,3):dep#4]", "ADD[(9,9)->(1,1):owns#-2]", "ADD[(0,4)->(2,3):" + "a" * 16 + "#4]"],
              ("kind", DEL): ["DEL[2,3]", "DEL[2,3] DEL[(2,3)->(0,4):dep#6]", "DEL[(2,3)->(0,4):dep#6] DEL[2,3]", "DEL[2,3]\nDEL[(1,1)->(1,0):owns#5] DEL[(0,0)->(2,3):dep#1] ", "DEL[5,5]"],
              ("named", NODE): ["ADD[4,4:6]", "ADD[4,4:0]", "ADD[4,4:-7] "], ("named", EDGE): ["ADD[(0,4)->(2,3):dep#4]", "ADD[(0,4)->(2,3):dep#-4]\n"],
              ("named", DEL): ["DEL[2,3]", "DEL[2,3] DEL[(2,3)->(0,4):dep#6]", "DEL[2,3] DEL[(1,1)->(2,3):owns#5] DEL[(2,3)->(2,3):dep#1]"]}
    refuse = {("kind", NODE): ["", "ADD[(4,4)->(4,4):value#6]", "ADD[4,4:6] ADD[1,1:1]", "DEL[4,4]", "ADD[4,4:6] DEL[4,4]", "DEL[1,0] ADD[4,4:6]", "ADD[4,4:]", "ADD[4,4:6]  ", "Sure ADD[4,4:6]"],
              ("kind", EDGE): ["", "DEL[(2,3)->(0,4):dep#6] ADD[(0,4)->(2,3):dep#4]", "ADD[(0,4)->(2,3):dep#4] ADD[(0,4)->(2,3):dep#4]", "ADD[4,4:6]", "ADD[(0,4)->(2,3):" + "a" * 17 + "#4]", "ADD[(0,4)->(2,3):#4]", "ADD[(0,4)->(2,3):dep]"],
              ("kind", DEL): ["", "ADD[1,1:1]", "DEL[2,3] ADD[1,1:1]", "DEL[2,3]DEL[1,1]", "DEL[2,3]  DEL[1,1]", "DEL[2,3] DEL[(2,3)->(0,4):" + "a" * 17 + "#6]"],
              ("named", NODE): ["ADD[1,0:6]", "ADD[4,5:6]", "ADD[44,4:6]", "ADD[4,4:]", "ADD[(4,4)->(4,4):value#6]", "ADD[4,4:6] ADD[4,4:6]"],
              ("named", EDGE): ["ADD[(1,5)->(2,3):dep#4]", "ADD[(0,4)->(2,3):owns#4]", "ADD[(2,3)->(0,4):dep#4]", "DEL[(2,3)->(0,4):dep#6] ADD[(0,4)->(2,3):dep#4]", "ADD[(0,4)->(2,3):dep#]"],
              ("named", DEL): ["", "DEL[1,1]", "DEL[(2,3)->(0,4):dep#6] DEL[2,3]", "DEL[2,3] DEL[(1,1)->(1,0):owns#5]", "DEL[2,3] DEL[4,4]", "DEL[2,3] ADD[1,1:1]", "DEL[23]"]}
    for (shape, ins), texts in accept.items():
        for t in texts: assert T.in_language(shape, ins, t), (shape, ins, t)
    for (shape, ins), texts in refuse.items():
        for t in texts: assert not T.in_language(shape, ins, t), (shape, ins, t)
    assert T.in_language("capped", NODE, "DEL[(1,1)->(1,0):owns#5] ADD[(1,1)->(1,0):dep#9]") and T.in_language("capped", DEL, "ADD[1,1:1]")                                            # the capped language allows what the others forbid
    # kind is inside capped, and named is inside kind, on the oracle's replies, the eight slips, every saved reply of the third run's S64 and H0 and every string of the lists above
    corpus = [st["ideal"] for e in range(192) for st in T.script_of(seed_of(e))] + [m[2] for m in MISSES.values()] + [x["text"] for c in (s64(), h0()) for s in c["sessions"] for x in s["log"]] + [t for v in accept.values() for t in v] + [t for v in refuse.values() for t in v]
    kinds = ["Add a service at (4,4) with value 6.", "Add a dependency edge from (0,4) to (2,3) with relation dep and weight 4.", "Delete the service at (2,3)."]
    for t in corpus:
        for ins in kinds:
            if T.in_language("kind", ins, t): assert T.in_language("capped", ins, t), (ins, t)
            if T.in_language("named", ins, t): assert T.in_language("kind", ins, t), (ins, t)
    # a regular expression of the page's own words: the relation of the request is escaped, a node of two digits is allowed
    assert T.in_language("kind", "Add a service at (10,12) with value 3.", "ADD[10,12:3]") and T.in_language("named", "Add a service at (10,12) with value 3.", "ADD[10,12:3]") and not T.in_language("named", "Add a service at (10,12) with value 3.", "ADD[1,12:3]")
    assert T.regex_for("named", "Add a dependency edge from (0,4) to (2,3) with relation a_b and weight 4.").count("a_b") == 1


# ---------------------------------------------------------------- 4 the eight first misses
def _first_miss(cell, job):
    s = next(s for s in cell["sessions"] if s["seed"] == seed_of(job)); return s, next(x for x in s["log"] if not x["exact"])


def test_the_eight_first_misses_of_the_page_table_are_the_ones_in_the_saved_files_with_their_languages_their_lost_steps_and_the_parity_check_flags_all_eight_and_no_exact_step():
    head = _head(); runs = dict(S64=s64(), H0=h0()); lost_kind = lost_named = 0
    for job, (step, request, answered, oracle, kind_in, named_in, lost, run) in MISSES.items():
        s, x = _first_miss(runs[run], job); sc = T.script_of(s["seed"])[x["t"] - 1]
        assert (x["t"], x["text"], sc["instruction"], sc["ideal"], sc["kind"] in ("add_node", "add_edge")) == (step, answered, request, oracle, True), job
        assert (T.in_language("kind", request, answered), T.in_language("named", request, answered), T.parity_ok(request, answered)) == (kind_in, named_in, False), job
        assert sum(1 for y in s["log"] if not y["exact"]) == lost, job
        lost_kind += lost * (not kind_in); lost_named += lost * kind_in
        row = f"| {job} | {step} | {request} | `{answered}` | `{oracle}` | {'inside' if kind_in else 'outside'} | {'inside' if named_in else 'outside'} | differs | {lost} |"
        assert row in head, row
        assert T.parity_ok(request, oracle)                                                                                    # the oracle's own reply passes
    assert sorted(MISSES) == [3, 4, 17, 19, 20, 62, 101, 122] and lost_kind == 66 and lost_named == 23 and 66 + 23 == 89
    assert sum(m[6] for m in MISSES.values() if m[7] == "S64") == 53 and sum(m[6] for m in MISSES.values() if m[7] == "H0") == 36
    # no other job of the two cells missed
    for name, c in runs.items(): assert {s["seed"] - SEED * 1000 for s in c["sessions"] if any(not x["exact"] for x in s["log"])} == {j for j, m in MISSES.items() if m[7] == name}
    # the parity check on the four cells without a retry: flags 3, 6, 7 and 3 steps of 5,040, and none after which the job held the right state
    cells = [next(c for c in tlist() if c["arm"] == "S" and c["K"] == k) for k in (8, 32, 64)] + [h0()]; flagged = []; steps = 0
    for c in cells:
        f = fe = 0
        for s in c["sessions"]:
            sc = T.script_of(s["seed"])
            for i, x in enumerate(s["log"]):
                steps += 1
                if not T.parity_ok(sc[i]["instruction"], x["text"]): f += 1; fe += x["exact"]
        flagged.append((f, fe))
    assert steps == 5040 and flagged == [(3, 0), (6, 0), (7, 0), (3, 0)], flagged
    assert "it flags 3, 6, 7 and 3 steps, and at none of those steps does the job hold the right state afterwards" in head and "no false alarm on 5,040 saved steps" in head


def test_the_reply_read_in_the_words_of_the_target_and_the_parity_text():
    assert T.describe_reply("ADD[1,0:6]") == "(1,0) ADD NODE value 6" and T.describe_reply("DEL[2,3]") == "(2,3) DELETE NODE" and T.describe_reply("ADD[(0,4)->(2,3):dep#4]") == "(0,4)->(2,3) ADD EDGE dep weight 4"
    assert T.describe_reply("DEL[(2,3)->(0,4):dep#6] ADD[(0,4)->(2,3):dep#4]") == "(2,3)->(0,4) DELETE EDGE dep" and T.describe_reply("  ADD[4,4:6]\n") == "(4,4) ADD NODE value 6"
    assert T.describe_reply("ADD[10,20:30]") == "(10,20) ADD NODE value 30" and T.describe_reply("ADD[(10,20)->(30,40):dep#50]") == "(10,20)->(30,40) ADD EDGE dep weight 50" and T.describe_reply("DEL[10,20]") == "(10,20) DELETE NODE" and T.describe_reply("ADD[0,0:-30]") == "(0,0) ADD NODE value -30"
    assert T.describe_reply("Sure") is None and T.describe_reply("") is None and T.describe_reply("ADD[4,4]") == "(4,4) ADD NODE value None"
    assert T.parity_ok("Add a service at (4,4) with value 6.", "ADD[4,4:6]") and not T.parity_ok("Add a service at (4,4) with value 6.", "ADD[4,4:6] ADD[1,1:1]") and not T.parity_ok("Add a service at (4,4) with value 6.", "ADD[4,4:5]")
    assert T.parity_ok("Delete the service at (2,3).", "DEL[2,3] DEL[(2,3)->(0,4):dep#6]") and not T.parity_ok("Delete the service at (2,3).", "DEL[(2,3)->(0,4):dep#6] DEL[2,3]") and not T.parity_ok("Delete the service at (2,3).", "")
    assert T.parity_text("Add a service at (0,0) with value 3.", "ADD[1,0:6]") == "PARITY request TARGET (0,0) ADD NODE value 3; reply (1,0) ADD NODE value 6. Reply with only the corrected delta."
    assert T.parity_text("Delete the service at (2,3).", "nonsense") == "PARITY request TARGET (2,3) DELETE NODE; reply unreadable. Reply with only the corrected delta."
    for t in ("(0,0) ADD NODE value 3", "(1,1)->(2,1) ADD EDGE dep weight 3", "(4,4) DELETE NODE"): assert T.target_body({"(0,0) ADD NODE value 3": "Add a service at (0,0) with value 3.", "(1,1)->(2,1) ADD EDGE dep weight 3": "Add a dependency edge from (1,1) to (2,1) with relation dep and weight 3.", "(4,4) DELETE NODE": "Delete the service at (4,4)."}[t]) == t


# ---------------------------------------------------------------- 5 the replay rule, the ceilings, the cache facts
def _compared(cell, shape):
    n = 0; forced = []
    for s in cell["sessions"]:
        sc = T.script_of(s["seed"]); k = None
        for i, x in enumerate(s["log"]):
            if not T.in_language(shape, sc[i]["instruction"], x["text"]): k = i + 1; break
        n += k if k else len(s["log"])
        if k: forced.append((s["seed"] - SEED * 1000, k))
    return n, forced


def test_the_replay_rule_the_ceilings_and_the_cache_facts_are_the_ones_the_page_gives():
    head = _head(); c = s64()
    assert _compared(c, "kind") == (1860, [(3, 16), (17, 20), (19, 12), (62, 12)]) and _compared(c, "named") == (1848, [(3, 16), (4, 18), (17, 20), (19, 12), (20, 30), (62, 12)])
    for s in ("the 1,860 steps up to and including those four are compared", "**jobs 4 and 20 must slip again at steps 18 and 30 with the same text**", "(steps 16, 18, 20, 12, 30, 12) and 1,848 steps are compared",
              "`SK` at most 1,906 of 1,920 steps (99.3 %; jobs 4 and 20 lose 14), `SN` at most 1,920 (100.0 %), against 1,867 (97.2 %) in the third run"): assert s in head, s
    S = C.summarize_cell2(c); lost = {s["seed"] - SEED * 1000: sum(1 for x in s["log"] if not x["exact"]) for s in c["sessions"]}
    assert S["exact_steps"] == 1867 and 1867 + lost[3] + lost[17] + lost[19] + lost[62] == 1906 and round(100 * 1906 / 1920, 1) == 99.3 and lost[4] + lost[20] == 14
    steps = [x for s in c["sessions"] for x in s["log"]]
    assert (round(statistics.mean(x["n_prompt"] for x in steps)), round(statistics.mean(x["n_cached"] for x in steps)), round(S["computed_per_step"]), round(statistics.mean(x["n_new"] for x in steps), 1)) == (349, 241, 109, 10.6)
    assert round(100 * S["cached_share"]) == 69 and min(x["n_cached"] for s in c["sessions"] for x in s["log"][1:]) == 192
    assert "349 prompt tokens on average, 241 of them served from the cache (69 %), 109 computed, and 10.6 written" in head and "at least the first 192 tokens of every prompt after the first step" in head
    comp = S["computed_per_step"]; tail, top = comp + 11.7, comp + 2 * 11.7 + (241 - 192)                     # the target lines add tokens; the top also loses the cache beyond the system prompt
    assert round(tail - comp) == 12 and round(top) == 181 and round(100 * (round(top) / round(comp) - 1)) == 66 and round(100 * (round(tail - comp) / round(comp))) == 11
    assert "about 181 computed tokens a step instead of 109 (+66 %)" in head and "adds about 12 computed tokens a step (+11 %)" in head
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): D_ = T.dry_run()
    assert [round(100 * D_[l]["exact_share"], 1) for l in ("SK", "SN", "FK")] == [99.3, 100.0, 100.0] and D_["SK"]["lost_steps"] == 14 and [x[:2] for x in D_["SK"]["first_misses"]] == [(4, 18), (20, 30)], "the six earlier slips are written where the capped model wrote them"
    comp_ = {k: D_[k]["computed_per_step"] for k in ("FK", "FKA", "FKT")}; sec = {k: D_[k]["seconds_per_step"] for k in ("FK", "FKA", "FKT")}
    assert [round(100 * (comp_[k] / comp_["FK"] - 1)) for k in ("FKT", "FKA")] == [10, 58] and [round(100 * (sec[k] / sec["FK"] - 1)) for k in ("FKT", "FKA")] == [8, 42]
    assert "gives +10 % and +58 % computed tokens and +8 % and +42 % seconds per step" in head and "+42 % on the stand-in and +66 % computed tokens on the cache counts" in head
    assert "(8 to 18 by the stand-in's token count, digits one by one)" in head and "427 on jobs 128 to 191 and 436 on jobs 0 to 63" in head and "(649 add-service, 590 add-edge, 681 delete)" in head
    assert "`S` at 64 jobs 217.4 seconds, 0.113 seconds per step, 30,914 exact steps per GPU hour; `H0` 216.3 seconds, 31,359" in head
    assert [round(C.summarize_cell2(c)["wall_seconds"], 1)] == [217.4] and round(h0()["wall_seconds"], 1) == 216.3


# ---------------------------------------------------------------- 6 the session, the adapter
def _strip(log): return [{k: v for k, v in x.items() if k != "in_language"} for x in log]


def test_a_session_with_the_old_prompt_and_the_capped_grammar_is_the_pinned_session_the_other_prompts_are_word_for_word_and_a_grammar_that_excludes_the_answer_stops_the_job():
    for job in (4, 19, 140):
        a = replay_to(job, 7); b = PS.Session("S", seed_of(job), PS.STEPS, False)
        while b.t < 6: rq = b.request(); b.accept(PS.Reply(rq.stand_in_reply, 0, 0, 0))
        ra, rb = a.request(), b.request(); assert ra.messages == rb.messages and ra.stand_in_reply == rb.stand_in_reply and (ra.seed, ra.step, ra.prompt, ra.shape, ra.regex) == (seed_of(job), 7, "old", "capped", T.REGEX)
        while a.t < 30:
            rq = a.request(); a.accept(PS.Reply(rq.stand_in_reply, 10, 5, 3))
        while b.t < 30: rq = b.request(); b.accept(PS.Reply(rq.stand_in_reply, 10, 5, 3))
        assert _strip(a.log) == b.log and all(x["in_language"] for x in a.log)
    s_old = replay_to(4, 18); s_top = replay_to(4, 18, "anchor", "kind"); s_tail = replay_to(4, 18, "tail", "named")
    r0, r1, r2 = s_old.request(), s_top.request(), s_tail.request(); t = "TARGET (0,0) ADD NODE value 3"
    assert r0.messages[0] == r1.messages[0] == r2.messages[0] and [m["role"] for m in r1.messages] == ["system", "user"]
    assert r1.messages[1]["content"] == f"{t}\n\n{r0.messages[1]['content']}\n{t}" and r2.messages[1]["content"] == f"{r0.messages[1]['content']}\n{t}" and r0.messages[1]["content"].endswith("Request: Add a service at (0,0) with value 3.")
    assert (r0.regex, r1.regex, r2.regex) == (T.REGEX, T.regex_for("kind", instr(4, 18)), T.regex_for("named", instr(4, 18))) and (r1.prompt, r1.shape, r2.prompt, r2.shape) == ("anchor", "kind", "tail", "named")
    assert r0.stand_in_reply == r1.stand_in_reply == r2.stand_in_reply == "ADD[0,0:3]"
    assert "NODES" in r0.messages[1]["content"] and r0.messages[1]["content"].count("Request:") == 1 and r1.messages[1]["content"].count("TARGET") == 2
    # in_language is the reply against the expression of the step's own request
    s = replay_to(4, 18, "old", "kind"); s.request(); s.accept(PS.Reply("ADD[1,0:6]", 1, 0, 1)); assert s.log[-1]["in_language"] is True and s.log[-1]["exact"] is False
    s = replay_to(3, 16, "old", "kind"); s.request(); s.accept(PS.Reply("ADD[(4,4)->(4,4):value#6]", 1, 0, 1)); assert s.log[-1]["in_language"] is False
    assert raises(lambda: T.TypedSession(1, 3, "middle", "kind"), AssertionError) and raises(lambda: T.TypedSession(1, 3, "old", "middle"), AssertionError)
    saved = T.regex_for
    try:
        T.regex_for = lambda shape, ins: r"ADD\[0,0:9\]"
        assert raises(lambda: T.TypedSession(seed_of(5), 3, "old", "kind").request(), AssertionError)
    finally: T.regex_for = saved


def test_the_vllm_adapter_puts_one_expression_into_every_request_and_refuses_to_start_without_structured_outputs():
    for how in ("module", "top"):
        mods, calls = T124._fake_vllm(how)
        with T124._installed(mods):
            e = T.TypedVLLMEngine(); kw = calls["init"][0]
            assert kw == dict(model=PS.MODEL, dtype="half", max_model_len=8192, gpu_memory_utilization=0.90, enable_prefix_caching=True, seed=0, num_gpu_blocks_override=5000)
            sp = e.sp.kw; assert set(sp) == {"temperature", "max_tokens", "structured_outputs"} and sp["temperature"] == 0.0 and sp["max_tokens"] == 96 and sp["structured_outputs"].kw == dict(regex=T.REGEX)
            i = e.info(); assert i["kind"] == "vllm" and i["version"] == "9.9.9-fake" and i["capacity_tokens"] == 80000 and i["capacity_source"] == "override"
            assert i["constraint"] == dict(kind="regex_per_request", api="structured_outputs(regex), one set of sampling parameters per request", regex=T.REGEX, regex_sha256=REGEX_SHA256, name_cap=16, shapes=["capped", "kind", "named"], prompts=["old", "anchor", "tail"])
            e.warm_up(); assert calls["generate"][-1][1] is e.sp and calls["resets"] == 1
            rqs = [T.TypedSession(seed_of(j), 30, "old", "named").request() for j in (4, 19, 140)] + [PS.Request([{"role": "user", "content": "a b c d"}])]
            tk = []; orig_tpl = e.tok.apply_chat_template; e.tok.apply_chat_template = lambda *a, **kw: (tk.append(kw), orig_tpl(*a, **kw))[1]
            reps, dt = e.generate_round(rqs); prompts, sps, tqdm_flag = calls["generate"][-1]; e.tok.apply_chat_template = orig_tpl
            assert tqdm_flag is False and tk == [dict(tokenize=False, add_generation_prompt=True)] * 4
            assert isinstance(sps, list) and len(sps) == len(prompts) == 4 and dt >= 0 and len(reps) == 4 and reps[0].text == "DEL[1,2]"
            assert [s.kw["structured_outputs"].kw["regex"] for s in sps] == [rqs[0].regex, rqs[1].regex, rqs[2].regex, T.REGEX]                # a request without an expression carries the capped one
            assert all(s.kw["temperature"] == 0.0 and s.kw["max_tokens"] == 96 for s in sps) and len({id(s) for s in sps}) == 4
            e.generate_round(rqs[:1], reset_cache=True); assert calls["resets"] == 2
            c = T.run_cell(e, "SK", steps=2); assert c["engine"]["constraint"]["regex_sha256"] == REGEX_SHA256 and c["engine"]["kind"] == "vllm"
            last = calls["generate"][-1]; assert all(isinstance(s, list) for _, s, _ in calls["generate"][1:])
    mods, calls = T124._fake_vllm(None)
    with T124._installed(mods):
        try: T.TypedVLLMEngine()
        except RuntimeError as err: assert "no StructuredOutputsParams" in str(err) and "does not fall back" in str(err)
        else: raise AssertionError("an engine that cannot constrain must refuse to start")
        assert calls["init"] == []


# ---------------------------------------------------------------- 7 the harness
def test_the_harness_on_a_double_that_obeys_the_grammar_writes_the_seven_cells_with_the_expressions_the_flags_and_the_rounds_and_a_capped_cell_is_the_pinned_cell():
    a = C.run_cell(B.BoundedStandIn(), "S", 5, steps=8); b = T.run_jobs(T.TypedStandIn(), list(range(5)), "old", "capped", steps=8)
    assert _strip([x for s in b["sessions"] for x in s["log"]]) == [x for s in a["sessions"] for x in s["log"]] and all(x["in_language"] for s in b["sessions"] for x in s["log"])
    assert [{k: v for k, v in r.items() if k != "new_grammars"} for r in b["rounds"]] == a["rounds"] and b["wall_seconds"] == a["wall_seconds"] and b["distinct_regexes"] == 1 and [r["new_grammars"] for r in b["rounds"]] == [1] + [0] * 7
    eng = T.ScriptedStandIn(); cells = {lab: T.run_cell(eng, lab, steps=3) for lab in T.ORDER}
    for lab, c in cells.items():
        K, first, prompt, shape = T.CELLS[lab]
        assert (c["label"], c["arm"], c["K"], c["first_episode"], c["prompt"], c["shape"], c["steps"], c["seed"], c["policy"], c["hint_rules"]) == (lab, "S", 64, first, prompt, shape, 3, SEED, "repair_m6", False)
        assert [s["seed"] for s in c["sessions"]] == [seed_of(e) for e in range(first, first + 64)] and all(len(s["log"]) == 3 for s in c["sessions"])
        assert c["grammar_sha256"] == T.grammar_sha256(shape, first, 64, 3) and c["distinct_regexes"] == sum(r["new_grammars"] for r in c["rounds"]) and c["engine"]["constraint"]["kind"] == "stand_in"
        assert abs(c["wall_seconds"] - sum(r["seconds"] for r in c["rounds"])) < 1e-9 and [r["round"] for r in c["rounds"]] == [1, 2, 3]
    assert [cells[l]["distinct_regexes"] for l in T.ORDER] == [1, 3, 3, 3, len({T.regex_for("named", st["instruction"]) for e in range(128, 192) for st in T.script_of(seed_of(e), 3)}), 3, len({T.regex_for("named", st["instruction"]) for e in range(64) for st in T.script_of(seed_of(e), 3)})]
    assert cells["FKA"]["wall_seconds"] > cells["FKT"]["wall_seconds"] > cells["FK"]["wall_seconds"] and cells["F0"]["wall_seconds"] == cells["FK"]["wall_seconds"]            # the anchor on top defeats the prefix cache, the tail costs less
    # a double that writes the earlier slips where the unconstrained model would: the grammar decides which of them the engine can still write
    faults = {("old", seed_of(j), t): text for j, t, text in T.KNOWN_SLIPS[:6]}; eng = T.ScriptedStandIn(faults=faults)
    out = {lab: T.run_cell(eng, lab) for lab in ("SK", "SN")}; S = T.summarize_all(list(out.values()))
    assert [(x[0], x[1]) for x in S["SK"]["first_misses"]] == [(4, 18), (20, 30)] and S["SK"]["lost_steps"] == 14 and round(S["SK"]["exact_share"], 4) == round(1906 / 1920, 4) and S["SN"]["first_misses"] == [] and S["SN"]["exact_share"] == 1.0
    assert [x["text"] for s in out["SK"]["sessions"] for x in s["log"] if (s["seed"], x["t"]) in ((seed_of(3), 16), (seed_of(19), 12))] == ["ADD[4,4:6]", "ADD[3,0:4]"]            # forced: the outside slips are not written
    # the sizes of the rounds, the seconds in them, the numbers the printout needs
    assert all(r["prompt_tokens"] > r["cached_tokens"] >= 0 for r in out["SK"]["rounds"][1:]) and S["SK"]["distinct_regexes"] == 3 and S["SN"]["distinct_regexes"] == 436
    assert S["SK"]["out_of_language"] == 0 and S["SK"]["jobs_exact_to_end"] == 62 and S["SK"]["new_grammar_rounds"] == 1 and S["SN"]["new_grammar_rounds"] > 1


def test_a_cell_is_summarized_with_the_first_miss_of_every_job_the_replies_outside_the_language_and_the_jobs_exact_to_the_end():
    eng = T.ScriptedStandIn(faults={("old", seed_of(140), 3): "ADD[(9,9)->(9,9):dep#1]"}); c = T.run_cell(eng, "FK"); S0 = T.summarize_cell4(c)
    assert S0["first_misses"] == [(140, 3, "add_edge", "ADD[(9,9)->(9,9):dep#1]")] and S0["lost_steps"] == 23 and S0["jobs_exact_to_end"] == 63 and S0["out_of_language"] == 0 and S0["label"] == "FK" and (S0["prompt"], S0["shape"]) == ("old", "kind")
    def tweak(fn):
        d = copy.deepcopy(c); fn(d); return T.summarize_cell4(d)
    recs = lambda d: [x for s in d["sessions"] for x in s["log"]]
    assert tweak(lambda d: recs(d)[0].update(in_language=False))["out_of_language"] == 1 and tweak(lambda d: recs(d)[0].pop("in_language"))["out_of_language"] == 1
    assert tweak(lambda d: recs(d)[0].update(parsed=False))["unparsed"] == 1 and tweak(lambda d: recs(d)[0].update(n_new=96))["truncated"] == 1 and tweak(lambda d: recs(d)[0].update(n_new=95))["truncated"] == 0
    assert tweak(lambda d: d["sessions"][0]["log"].pop())["jobs_exact_to_end"] == 62 and tweak(lambda d: d["sessions"][0].update(script_steps=31))["jobs_exact_to_end"] == 62
    assert tweak(lambda d: d["sessions"][2]["log"][4].update(exact=False))["first_misses"][0][:2] == (130, 5) and tweak(lambda d: d.update(distinct_regexes=None))["distinct_regexes"] is None
    assert raises(lambda: T.summarize_all([c, c]), AssertionError) and sorted(T.summarize_all([c, T.run_cell(eng, "F0", steps=2)])) == ["F0", "FK"]


def test_the_planned_cells_the_merge_and_what_it_refuses():
    eng = T.ScriptedStandIn(); cells = [T.run_cell(eng, lab, steps=2) for lab in reversed(T.ORDER)]
    m = T.merge_cells(cells); assert [c["label"] for c in m["cells"]] == T.ORDER and m["meta"]["name_cap"] == 16 and m["meta"]["cells"] == {k: list(T.CELLS[k]) for k in T.ORDER}
    assert raises(lambda: T.merge_cells([]), AssertionError) and raises(lambda: T.merge_cells([cells[0], cells[0]]), ValueError)
    plain = PS.run_cell(PS.StandInEngine(), "S", 2, steps=2); assert raises(lambda: T.merge_cells([plain]), (ValueError, KeyError))
    old = B.run_cell(B.BoundedStandIn(), "S", 8, steps=2); assert raises(lambda: T.merge_cells([old]), (ValueError, KeyError))
    base = next(c for c in cells if c["label"] == "FK")
    o = copy.deepcopy(base); o["label"] = "F9"; assert raises(lambda: T.merge_cells([o]), ValueError)
    for field, val in (("prompt", "anchor"), ("shape", "named"), ("first_episode", 129), ("K", 8)):
        o = copy.deepcopy(base); o[field] = val; assert raises(lambda: T.merge_cells([o]), ValueError), field
    o = copy.deepcopy(base); o["engine"]["constraint"]["regex_sha256"] = "0" * 64; assert raises(lambda: T.merge_cells([o]), ValueError)
    o = copy.deepcopy(base); o["engine"]["constraint"] = None; assert raises(lambda: T.merge_cells([o]), ValueError)
    for field, val in (("seed", 1), ("steps", 7), ("policy", "gate"), ("hint_rules", True)):
        o = copy.deepcopy(base); o[field] = val; o["label"] = "F0"; o["shape"] = "capped"; assert raises(lambda: T.merge_cells([next(c for c in cells if c["label"] == "F0"), o]), ValueError), field
    for field in ("version", "gpu", "model", "capacity_tokens", "kind"):
        o = copy.deepcopy(next(c for c in cells if c["label"] == "F0")); o["engine"][field] = "other"; assert raises(lambda: T.merge_cells([base, o]), ValueError), field
    with tempfile.TemporaryDirectory() as d:
        paths = [T.save_cell(c, os.path.join(d, c["label"] + ".json")) for c in cells]; out = os.path.join(d, "m.json"); mm = T.merge_files(paths, out, third_run=THIRD_FILE); back = json.load(open(out))
        assert back["module_sha256"] == T.fingerprint() == _sha(T.__file__) and back["pinned_sha256"] == PINS and back["third_run_sha256"] == THIRD_SHA256 and back["third_run_file"] == "axi_provider_bounded.json" and back["cells"] == mm["cells"]
        assert os.path.samefile(T.default_third_run_path(), THIRD_FILE) and T.load_third_run()["cells"] == tlist()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf): T.print_cells(T.summarize_all(back["cells"])); T.print_misses(T.summarize_all(back["cells"])); T.print_verdict(T.verdict_typed(back["cells"], tlist()))
        o = buf.getvalue(); assert "NOT a result" not in o and o.count("[claim]") == 13 and o.count("[guard]") == 2 and o.count("[instrument]") == 2


def test_the_probe_asks_the_eight_known_first_misses_under_five_variants_and_once_with_the_parity_turn_and_a_wrong_answer_is_a_description():
    class Rec(T.TypedStandIn):
        def __init__(self): super().__init__(); self.seen = []
        def generate_round(self, requests, reset_cache=False): self.seen.append(requests); return super().generate_round(requests, reset_cache)
    e = Rec(); p = T.known_slip_probe(e); assert len(e.seen) == 1 and len(e.seen[0]) == 8 * 6 and [x["job"] for x in p] == [3, 4, 17, 19, 20, 62, 101, 122]
    names = [n for n, _, _ in T.PROBE_VARIANTS] + ["parity"]; assert names == ["old", "kind", "named", "kind+top", "kind+tail", "parity"]
    for i, x in enumerate(p):
        job, (step, request, answered, oracle, kind_in, named_in, lost, run) = x["job"], MISSES[x["job"]]
        group = e.seen[0][6 * i: 6 * i + 6]; base = replay_to(job, step).request()
        assert (x["step"], x["instruction"], x["oracle"], x["earlier"], x["earlier_in_kind"], x["earlier_in_named"]) == (step, request, oracle, answered, kind_in, named_in) and list(x["replies"]) == names
        assert [g.messages for g in group[:3]] == [base.messages] * 3 and group[1].regex == T.regex_for("kind", request) and group[2].regex == T.regex_for("named", request) and group[0].regex == T.REGEX
        t = T.target_line(request); body = base.messages[1]["content"]
        assert group[3].messages[1]["content"] == f"{t}\n\n{body}\n{t}" and group[4].messages[1]["content"] == f"{body}\n{t}" and group[3].regex == group[4].regex == T.regex_for("kind", request)
        par = group[5]; assert par.regex == T.REGEX and [m["role"] for m in par.messages] == ["system", "user", "assistant", "user"] and par.messages[:2] == base.messages and par.messages[2] == {"role": "assistant", "content": answered}
        assert par.messages[3]["content"] == x["parity"] == T.parity_text(request, answered) and (par.seed, par.step) == (seed_of(job), step)
        assert all(r.seed == seed_of(job) and r.step == step for r in group) and set(x["cached"]) == set(names) and x["seconds"] >= 0
    assert p[1]["parity"] == "PARITY request TARGET (0,0) ADD NODE value 3; reply (1,0) ADD NODE value 6. Reply with only the corrected delta."
    assert p[4]["parity"] == "PARITY request TARGET (1,1)->(2,1) ADD EDGE dep weight 3; reply (1,5)->(2,1) ADD EDGE dep weight 3. Reply with only the corrected delta."
    assert raises(lambda: T.known_slip_probe(e, seed=1, slips=[(3, 99, "x")]), (IndexError, ValueError))
    e2 = Rec(); q = T.known_slip_probe(e2, slips=[(4, 18, "ADD[1,0:6]")]); assert len(q) == 1 and len(e2.seen[0]) == 6


# ---------------------------------------------------------------- 8 the lines
class ReplaySaved(T.ScriptedStandIn):
    """Seen jobs (0 to 63, old prompt): the third run's saved reply of that job and step while it is inside the language of the request; when the grammar forbids it the job takes the oracle's own reply from there
    on. Other jobs: the `extra` replies (prompt, seed, step) where the grammar allows them. Tests the harness only."""
    def __init__(self, saved, extra, **kw): super().__init__(**kw); self.saved, self.extra, self.released = saved, extra, set(); self.seeds = {s for s, _ in saved}
    def reset_cache(self): self.released.clear(); super().reset_cache()
    def generate_round(self, requests, reset_cache=False):
        rqs = []
        for r in requests:
            text = None
            if r.prompt == "old" and r.seed in self.seeds and r.seed not in self.released:
                text = self.saved[(r.seed, r.step)]
                if re.fullmatch(r.regex, text) is None: self.released.add(r.seed); text = None
            else:
                text = self.extra.get((r.prompt, r.seed, r.step))
                if text is not None and re.fullmatch(r.regex, text) is None: text = None
            rqs.append(r if text is None else T.TRequest(r.messages, text, seed=r.seed, step=r.step, prompt=r.prompt, shape=r.shape, regex=r.regex))
        return T.TypedStandIn.generate_round(self, rqs, reset_cache)


EXTRA = {("old", seed_of(132), 2): "ADD[(1,1)->(1,1):value#1]", ("old", seed_of(140), 3): "ADD[(9,9)->(9,9):dep#1]"}      # a shape slip (29 steps lost) and a number slip (23 steps lost) in the fresh jobs


def good():
    """A run that satisfies the third run's own cells: the seen jobs replay the third run's saved replies; in the fresh jobs the unconstrained cell writes one shape slip and one number slip; the cost of the anchor at the top
    is set to 1.25 times that of `FK` (the stand-in model makes it 1.42 times, which reads NO on line 15 as the page expected). Reads yes on all 17 lines."""
    if "good" in _CACHE: return _CACHE["good"]
    saved = {(s["seed"], x["t"]): x["text"] for s in s64()["sessions"] for x in s["log"]}
    eng = ReplaySaved(saved, EXTRA, capacity_tokens=int(PS.CAPACITY_TOKENS * PS.PROXY_SCALE)); cells = [T.run_cell(eng, lab) for lab in T.ORDER]
    _CACHE["raw"] = copy.deepcopy(cells)
    fk = next(c for c in cells if c["label"] == "FK"); fka = next(c for c in cells if c["label"] == "FKA"); fka["wall_seconds"] = fk["wall_seconds"] * 1.25
    _CACHE["good"] = cells; return cells


def read(cells, third_cells=None): return [ok for _, _, _, ok in T.verdict_typed(cells, tlist() if third_cells is None else third_cells)]
def cell_of(cells, label): return next(c for c in cells if c["label"] == label)
def records(c): return [x for s in c["sessions"] for x in s["log"]]


def test_a_good_run_built_from_the_third_runs_own_replies_reads_yes_on_all_seventeen_lines_and_the_figures_are_the_ones_the_page_expects():
    g = good(); S = T.summarize_all(g); assert read(g) == [True] * 17
    assert [S[l]["exact_steps"] for l in T.ORDER] == [1868, 1897, 1920, 1920, 1920, 1906, 1920] and [S[l]["lost_steps"] for l in T.ORDER] == [52, 23, 0, 0, 0, 14, 0]
    assert [S[l]["jobs_exact_to_end"] for l in T.ORDER] == [62, 63, 64, 64, 64, 62, 64] and [S[l]["distinct_regexes"] for l in T.ORDER] == [1, 3, 3, 3, 427, 3, 436]
    assert [(x[0], x[1]) for x in S["F0"]["first_misses"]] == [(132, 2), (140, 3)] and [(x[0], x[1]) for x in S["FK"]["first_misses"]] == [(140, 3)] and [(x[0], x[1]) for x in S["SK"]["first_misses"]] == [(4, 18), (20, 30)]
    assert T.jobs_exact(cell_of(g, "F0"))[4] == 1 and T.jobs_exact(cell_of(g, "FK"))[4] == 30 and T.jobs_exact(cell_of(g, "FK"))[12] == T.jobs_exact(cell_of(g, "F0"))[12] == 7
    raw = _CACHE["raw"]; assert read(raw) == [True] * 14 + [False] + [True] * 2                                         # the stand-in's own anchor cost: line 15 reads NO, as the page expected
    sf = {l: S[l]["seconds_per_step"] for l in ("F0", "FK", "FKA", "FKT", "FN")}; assert abs(sf["FK"] / sf["F0"] - 1) < 0.002 and abs(sf["FN"] / sf["F0"] - 1) < 0.002 and 1.07 < sf["FKT"] / sf["FK"] < 1.09 and 1.41 < T.summarize_cell4(cell_of(raw, "FKA"))["seconds_per_step"] / sf["FK"] < 1.43
    # the replay rule holds on the good run, and a pair of other cells is not a replay pair
    assert T.same_until_forced(cell_of(g, "FK"), cell_of(g, "F0"), "kind") and T.same_until_forced(cell_of(g, "FN"), cell_of(g, "F0"), "named") and T.same_until_forced(cell_of(g, "SK"), s64(), "kind") and T.same_until_forced(cell_of(g, "SN"), s64(), "named")
    assert T.same_until_forced(cell_of(g, "F0"), cell_of(g, "F0"), "capped") and not T.same_until_forced(cell_of(g, "FK"), cell_of(g, "F0"), "capped"), "FK against F0 is a replay pair only under the kind language"
    assert not T.same_until_forced(cell_of(g, "FKT"), cell_of(g, "FK"), "kind"), "a different prompt is not a replay: the number slip of job 140 is written only under the old prompt"


def changed(cells, base=None):
    base = [True] * 17 if base is None else base; r = read(cells); return [i + 1 for i, (a, b) in enumerate(zip(base, r)) if a != b]


def flip(label, fn):
    cells = copy.deepcopy(good()); fn(cell_of(cells, label)); return cells


def lose(n, job=None):
    """Set `n` exact steps of a cell to not exact (the last ones; or the first ones of the `job`-th session)."""
    def f(c):
        pool = [x for x in (records(c) if job is None else c["sessions"][job]["log"]) if x["exact"]]
        for x in (pool[-n:] if job is None else pool[:n]): x["exact"] = False
    return f


def test_each_of_the_seventeen_lines_flips_at_its_edge_and_a_missing_cell_reads_not_run_never_no():
    # 1: a missing cached count, in a first cell or a last one
    assert changed(flip("F0", lambda c: records(c)[5].update(n_cached=None))) == [1] and changed(flip("SN", lambda c: records(c)[500].update(n_cached=None))) == [1]
    # 2: all 96 new tokens; one reply outside its language; the share refused by the strict parser at its limit and one over
    assert changed(flip("FN", lambda c: records(c)[5].update(n_new=96))) == [2] and changed(flip("FN", lambda c: records(c)[5].update(in_language=False))) == [2]
    N = sum(len(T.replies_of(c)) for c in good()); k = int(0.005 * N + 1e-9); assert N == 13440 and k == 67
    unparse = lambda n: (lambda c: [x.update(parsed=False) for x in records(c)[:n]])
    assert changed(flip("FKT", unparse(k))) == [] and changed(flip("FKT", unparse(k + 1))) == [2], (N, k)
    assert changed(flip("FN", lambda c: records(c)[5].pop("in_language"))) == [2], "a reply without the in-language flag is outside"
    # 3: a reply before the first forced step that is not the base's; one after it; a missing step; a job of another seed; the pairs are compared one by one
    assert changed(flip("FK", lambda c: c["sessions"][0]["log"][4].update(text="DEL[0,0]"))) == [3]
    assert changed(flip("FK", lambda c: c["sessions"][4]["log"][10].update(text="DEL[0,0]"))) == [], "job 132 was forced at step 2; what comes after is free"
    assert changed(flip("FK", lambda c: c["sessions"][4]["log"][1].update(text="ADD[(1,1)->(1,1):value#1]"))) == [3], "the forced reply must be inside the language"
    assert changed(flip("FK", lambda c: c["sessions"][5]["log"].pop())) == [3, 7], "a job with a step fewer is not the base's job (line 3), and it has fewer exact steps than in F0 (line 7)"
    assert changed(flip("FN", lambda c: c["sessions"][3].update(seed=seed_of(9)))) == [3]
    assert 3 in changed(flip("FK", lambda c: c["sessions"].pop())), "a cell with a job fewer is not the base's cell"
    assert changed(flip("FK", lambda c: c["sessions"][4]["log"].pop())) == [], "job 132 is forced at step 2, so the rule says nothing about how many steps it has after that"
    assert changed(flip("SK", lambda c: c["sessions"][7]["log"][2].update(text="DEL[0,0]"))) == [3] and changed(flip("SN", lambda c: c["sessions"][4]["log"][10].update(text="DEL[0,0]"))) == [3]
    assert changed(flip("SN", lambda c: c["sessions"][4]["log"][20].update(text="DEL[0,0]"))) == [], "job 4 is forced at step 18 in SN (its saved reply is outside the named language); what comes after is free"
    assert changed(flip("SK", lambda c: c["sessions"][4]["log"][20].update(text="DEL[0,0]"))) == [3], "job 4 stays inside the kind language, so its replies after step 18 must be the saved ones"
    # 4: the recorded expressions are not the ones recomputed from the jobs; the cell's jobs are not the jobs of its label
    assert changed(flip("FK", lambda c: c.update(grammar_sha256="0" * 64))) == [4] and changed(flip("SN", lambda c: c.update(grammar_sha256=T.grammar_sha256("kind", 0, 64)))) == [4]
    # 5, 6, 7: FK against F0 (1,897 and 1,868 exact steps; one point is 20 steps, since 19.2 is not reached by 19)
    fk_loses = lambda n: flip("FK", lose(n, job=4)); assert read(fk_loses(0)) == [True] * 17
    assert changed(fk_loses(9)) == [] and changed(fk_loses(10)) == [6] and changed(fk_loses(28)) == [6]
    assert changed(fk_loses(29)) == [6], "FK at 1,868 exact steps, as many as F0: line 5 reads yes (not below), line 6 NO (no gain), line 7 yes (job 132 keeps one exact step as in F0), line 17 yes (its seconds are a hair less)"
    assert changed(fk_loses(30)) == [5, 6, 7, 17], "one step less than F0: lines 5 and 17 NO, line 7 NO (job 132 has no exact step, F0's has one)"
    assert changed(flip("FK", lose(1, job=12))) == [7], "one job of FK with fewer exact steps than in F0"
    # 8, 9, 10: SK holds 1,906 of 1,920 (99 percent is 1,901); SN and FN hold 1,920 (99.5 percent is 1,911)
    assert changed(flip("SK", lose(5))) == [] and changed(flip("SK", lose(6))) == [8]
    assert changed(flip("SN", lose(9))) == [] and changed(flip("SN", lose(10))) == [9]
    assert changed(flip("FN", lose(9))) == [] and changed(flip("FN", lose(10))) == [10]
    # 11, 12: FKA and FKT against FK (1,920 against 1,897); equal is not below
    assert changed(flip("FKA", lose(23))) == [] and changed(flip("FKA", lose(24))) == [11] and changed(flip("FKT", lose(23))) == [] and changed(flip("FKT", lose(24))) == [12]
    # 13 to 17: the seconds, relative to the control or to FK (the cost of the anchor at the top is set to 1.25 in the good run)
    f0w = cell_of(good(), "F0")["wall_seconds"]; fkw = cell_of(good(), "FK")["wall_seconds"]; assert abs(f0w / fkw - 1) < 0.001            # the stand-in model makes the seconds depend a little on what is written
    wall = lambda label, f: flip(label, lambda c: c.update(wall_seconds=cell_of(good(), "FK" if label in ("FKA", "FKT") else "F0")["wall_seconds"] * f))
    r = read(wall("FK", 1.05)); assert r[12] is True and r[16] is False                                                      # line 13 at its edge; the exact steps per hour of FK fall below F0's with 5 percent more seconds
    r = read(wall("FK", 1.0501)); assert r[12] is False and r[16] is False
    r = read(wall("FK", 1.015)); assert r[12] is True and r[16] is True                                                       # 1,897 against 1,868 exact steps: higher up to 1.0155 times the seconds
    r = read(wall("FK", 1.016)); assert r[12] is True and r[16] is False
    r = read(wall("FK", 0.5)); assert r[12] is True and r[16] is True
    r = read(wall("FN", 1.15)); assert r[13] is True and read(wall("FN", 1.1501))[13] is False and changed(wall("FN", 1.15)) == [] and changed(wall("FN", 1.1501)) == [14]
    assert read(wall("FKA", 1.30))[14] is True and read(wall("FKA", 1.3001))[14] is False and changed(wall("FKA", 1.3001)) == [15]
    assert read(wall("FKT", 1.20))[15] is True and read(wall("FKT", 1.2001))[15] is False and changed(wall("FKT", 1.2001)) == [16]
    # the lines are numbered 1 to 17 with the page's tags and words
    assert [(n, t, x) for n, t, x, _ in T.verdict_typed(good(), tlist())] == [(i, t, x) for i, (t, x) in enumerate(T.TL_LINES, 1)]
    assert [t for t, _ in T.TL_LINES] == ["instrument", "instrument", "guard", "guard"] + ["claim"] * 13
    # a missing cell reads not run, never NO
    g = [c for c in good() if c["label"] != "FK"]; r = read(g); assert r[4:7] == [None] * 3 and r[10:12] == [None, None] and r[12] is None and r[14:17] == [None] * 3 and r[2] is True and r[7:10] == [True] * 3 and r[13] is True
    g = [c for c in good() if c["label"] != "F0"]; r = read(g); assert r[4:7] == [None] * 3 and r[12:14] == [None, None] and r[16] is None and r[10:12] == [True, True] and r[7:10] == [True] * 3
    g = [c for c in good() if c["label"] not in ("SK", "SN")]; r = read(g); assert r[7:9] == [None, None] and r[:7] == [True] * 7 and r[9:] == [True] * 8
    assert read([]) == [None] * 17 and read([c for c in good() if c["label"] == "SK"], []) == [True, True, None, True] + [None] * 3 + [True] + [None] * 9
    empty = copy.deepcopy(good()); [s.update(log=[]) for s in cell_of(empty, "FK")["sessions"]]; r = read(empty); assert r[4:7] == [None] * 3 and r[16] is None
    # the third run's cells missing: only the pairs against them are not read
    r = read(good(), []); assert r == [True, True, True, True] + [True] * 13, "F0 against FK and FN is a pair of this run; SK and SN against nothing are skipped"
    r = read([c for c in good() if c["label"] in ("SK", "SN")], []); assert r[2] is None


def test_the_rule_about_what_may_be_claimed_follows_lines_2_3_and_4_and_the_printout_says_it():
    ok = T.claims_allowed(T.verdict_typed(good(), tlist())); assert ok == dict(base=True, constraint_held=True, grammar_changed_only_what_it_forbids=True, expressions_as_defined=True, fresh_jobs_kind={5: True, 6: True, 7: True},
                                                                              fresh_jobs_named={10: True}, anchor_lines={11: True, 12: True}, seen_jobs={8: True, 9: True}, time_and_cost_in_one_session={13: True, 14: True, 15: True, 16: True, 17: True})
    for i, name in ((2, "constraint_held"), (3, "grammar_changed_only_what_it_forbids"), (4, "expressions_as_defined")):
        checks = [(n, t, x, (False if n == i else v)) for n, t, x, v in T.verdict_typed(good(), tlist())]; a = T.claims_allowed(checks)
        assert a["base"] is False and a[name] is False and a["fresh_jobs_kind"] == {5: None, 6: None, 7: None} and a["fresh_jobs_named"] == {10: None} and a["anchor_lines"] == {11: None, 12: None} and a["seen_jobs"] == {8: None, 9: None} \
            and a["time_and_cost_in_one_session"] == {13: None, 14: None, 15: None, 16: None, 17: None}, i
    checks = [(n, t, x, (False if n in (6, 17) else v)) for n, t, x, v in T.verdict_typed(good(), tlist())]; a = T.claims_allowed(checks); assert a["base"] and a["fresh_jobs_kind"] == {5: True, 6: False, 7: True} and a["time_and_cost_in_one_session"][17] is False
    for shown in (T.verdict_typed(good(), tlist()), [(n, t, x, (False if n == 3 else v)) for n, t, x, v in T.verdict_typed(good(), tlist())]):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf): T.print_verdict(shown)
        out = buf.getvalue(); assert ("never: a multiple of capacity" in out) != ("NO claim is made from this run" in out)
        assert out.count("yes ") + out.count("NO ") >= 17 and "headline lines 6 and 17" in out
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): T.print_verdict(T.verdict_typed(good(), tlist()))
    out = buf.getvalue()
    assert out.endswith("  rule on the page: lines 2, 3 and 4 read yes, so the figures of the cells are claimed as read. That a grammar raises the exactness rests on the fresh jobs only (lines 5 to 7, 10 to 12); lines 8 and 9 are on the jobs the slips were read on.\n"
                        "  a statement about time or cost comes only from cells of one session (lines 13 to 17), never from a cell against the third run's.\n"
                        "  never: a multiple of capacity, dollars, '100 percent', a statement about any provider, anything about requests in free words, anything about a model that was not run.\n")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): S = T.dry_run(steps=4)
    assert buf.getvalue().startswith("STAND-IN (a model of a block cache that obeys the grammar; the earlier runs' slips are written by the test double), NOT a result\n") and sorted(S) == sorted(T.ORDER) and all(s["steps_done"] > 0 for s in S.values())
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): T.dry_run(steps=3, labels=["FK"])
    assert "FK" in buf.getvalue() and "F0 " not in buf.getvalue()


def test_the_table_and_the_lines_are_printed_in_the_format_the_notebook_shows():
    S = {"FK": dict(steps_done=1920, wall_seconds=215.3, seconds_per_step=0.1121, computed_per_step=123.2, exact_share=0.98906, jobs_exact_to_end=63, lost_steps=21, unparsed=0, out_of_language=0, distinct_regexes=3, exact_per_hour=31755.0, first_misses=[(140, 3, "add_edge", "ADD[(9,9)->(9,9):dep#1]")]),
         "SN": dict(steps_done=1920, wall_seconds=190.0, seconds_per_step=0.099, computed_per_step=None, exact_share=1.0, jobs_exact_to_end=64, lost_steps=0, unparsed=2, out_of_language=1, distinct_regexes=None, exact_per_hour=None, first_misses=[])}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): T.print_cells(S); T.print_misses(S)
    assert buf.getvalue() == ("\ncell   K      jobs prompt grammar | steps  seconds  s/step comp/step |  exact jobs ok lost | unparsed outside regexes  exact/h\n"
                              "FK    64   128-191    old    kind |  1920      215   0.112       123 |  98.9%      63   21 |        0       0       3    31755\n"
                              "SN    64      0-63    old   named |  1920      190   0.099         - | 100.0%      64    0 |        2       1       -        -\n"
                              "first misses in FK: job 140 step 3 (add_edge) 'ADD[(9,9)->(9,9):dep#1]'\n"), buf.getvalue()
    checks = [(1, "instrument", "aaa", True), (2, "instrument", "bbb", False), (3, "guard", "ccc", None)] + [(i, "claim", f"x{i}", True) for i in range(4, 18)]; buf = io.StringIO()
    with contextlib.redirect_stdout(buf): T.print_verdict(checks)
    want = ("\n-- provider-side test, typed requests (reading rules fixed before the run) --\n  yes       1. [instrument] aaa\n  NO        2. [instrument] bbb\n  not run   3. [guard] ccc\n" + "".join(f"  yes      {i:2d}. [claim] x{i}\n" for i in range(4, 18)) +
            "  (16 of 17 lines could be read; headline lines 6 and 17: yes, yes)\n  rule on the page: line 2, 3 or 4 did not read yes, so NO claim is made from this run until the cause is found.\n")
    assert buf.getvalue() == want, buf.getvalue()
    checks[1] = (2, "instrument", "bbb", True); checks[2] = (3, "guard", "ccc", True); checks[16] = (17, "claim", "x17", False); checks[5] = (6, "claim", "x6", None); buf = io.StringIO()
    with contextlib.redirect_stdout(buf): T.print_verdict(checks, title="t")
    o = buf.getvalue(); assert o.startswith("\n-- t (reading rules fixed before the run) --\n") and "  (16 of 17 lines could be read; headline lines 6 and 17: not run, NO)\n" in o and "  NO       17. [claim] x17\n" in o and "  not run   6. [claim] x6\n" in o


def test_the_command_line_merges_cell_files_prints_the_lines_and_shows_its_help_and_the_dry_run_says_it_is_not_a_result():
    env = dict(os.environ, PYTHONPATH=os.path.join(HERE, ".."))
    h = subprocess.run([sys.executable, "-m", "axi.experiments.provider_side_typed"], capture_output=True, text=True, env=env, cwd=os.path.join(HERE, "..")); flat = " ".join(h.stdout.split())
    assert h.returncode == 0 and "--dry-run" in h.stdout and "--merge" in h.stdout and "run every cell on the stand-in engine (no GPU); prints, does not save" in flat and "put cell files together" in flat
    d = subprocess.run([sys.executable, "-m", "axi.experiments.provider_side_typed", "--dry-run"], capture_output=True, text=True, env=env, cwd=os.path.join(HERE, ".."))
    assert d.returncode == 0 and "NOT a result" in d.stdout and "FKA" in d.stdout and "SN" in d.stdout
    with tempfile.TemporaryDirectory() as t:
        eng = T.ScriptedStandIn(); paths = [T.save_cell(T.run_cell(eng, lab, steps=3), os.path.join(t, lab + ".json")) for lab in T.ORDER]; out = os.path.join(t, "m.json")
        m = subprocess.run([sys.executable, "-m", "axi.experiments.provider_side_typed", "--merge", *paths, "--out", out], capture_output=True, text=True, env=env, cwd=os.path.join(HERE, ".."))
        assert m.returncode == 0 and "written to" in m.stdout and "-- provider-side test, typed requests" in m.stdout and m.stdout.count("[claim]") == 13 and os.path.exists(out)


# ---------------------------------------------------------------- 9 the locked page
def _h(text): return hashlib.sha256((text.rstrip("\n") + "\n").encode()).hexdigest()


def _head():
    text = open(PAGE).read(); assert text.count(RESULT_MARK) <= 1
    return text[:text.index(RESULT_MARK)] if RESULT_MARK in text else text


def test_the_page_is_locked_and_carries_the_seventeen_lines_the_rule_the_pins_the_cells_and_the_thresholds_the_code_uses():
    head = _head(); assert _h(head) == PAGE_SHA256, "the predictions and reading rules must not change after the run; later notes go below them"
    for i, (tag, line) in enumerate(T.TL_LINES, 1): assert f"\n{i}. [{tag}] {line} (" in head, i
    probs = [int(m.group(1)) for m in re.finditer(r"\n\d+\. \[\w+\] .*\((\d+) %\)(?=\n)", head)]; assert probs == PROBS, probs
    for needle in ("**Headline** (lines 6 and 17)", "Lines 2, 3 and 4 all reading yes: about **70 %**", "Lines 6 and 17 both reading yes: about **35 %**", "(lines 1 to 5, 7, 13 and 14 yes, line 15 NO, the rest open): about **25 %**", "All 17 yes: under **1 %**"):
        assert needle in head, needle
    for name, sha in PINS.items(): assert sha in head, name
    assert f"`{REGEX_SHA256}`" in head and "`python/axi/experiments/provider_side_typed.py`" in head and "Concept-map item 111" in head and "and edits none of them" in head and "**there is no retry in any cell.**" in head
    for lab, (K, first, prompt, shape) in T.CELLS.items(): assert f"| `{lab}` | {first} to {first + K - 1} | {prompt} | {shape} |" in head, lab
    th = ("lines 5 to 7 FK against F0 (not below; at least `0.01` above; no job with fewer exact steps); line 8 SK (`0.99`); line 9 SN (`0.995`); line 10 FN (`0.995`); lines 11 and 12 FKA and FKT against FK (not below); line 13 the seconds per step of FK at most `1.05` times those of F0; "
          "line 14 FN at most `1.15` times F0; line 15 FKA at most `1.30` times FK; line 16 FKT at most `1.20` times FK; line 17 the exact steps per GPU hour of FK higher than those of F0")
    assert th in head, th
    assert (T.T_GAIN, T.T_HIGH, T.T_HIGHER, T.T_FRESH_NAMED, 1 + T.T_TIME_KIND, 1 + T.T_TIME_NAMED, 1 + T.T_TIME_TOP, 1 + T.T_TIME_TAIL) == (0.01, 0.99, 0.995, 0.995, 1.05, 1.15, 1.30, 1.20)
    assert "at least 99.5 percent were accepted by the strict parser" in head and "the most it could be is 99.3" in head
    for needle in ("**The rule that decides what the Result may claim.**", "Line 2, 3 or 4 reads NO: no claim is made from this run until the cause is found.", "That a grammar raises the exactness is claimed only from the fresh jobs (lines 5 to 7, 10 to 12)",
                   "A statement about time or cost comes only from cells of one session (lines 13 to 17), never from a cell against the third run's (another Colab session).", "Never a multiple of capacity, never a count of GPUs saved, never dollars or a percentage of cost, never \"100 percent\" for a grammar, never \"proven\"",
                   "never a statement about requests in free words", "**Nothing on this page promises 100 percent.**", "**the model's share is the number of a node, the weight of an edge, and the list of edges of a delete**", "**Only the fresh cells say anything about jobs in general.**",
                   "**say so in the same sentence.**", "**for the grammar that names the coordinates, in the same sentence, that the model writes the numbers and the edge list and the request is otherwise copied into the reply.**",
                   "**a wrong answer is not a STOP.**", "**For all 192 jobs the oracle's own reply is inside the expression of every request**", "A cell that did not run:* its lines read \"not run\", never NO.",
                   "a grammar that cannot say the answer is not this test", "(this is, in another vocabulary, a per-tool schema)", "a request in free words is a different test"):
        assert needle in head, needle
    assert head.count("\n# ") == 0 and head.startswith("# Test 18: the serving side a fifth time") and "(written BEFORE the run, 20 Oct 2026)" in head
    assert head.rstrip("\n").endswith("\n---")


def test_the_figures_the_page_quotes_from_the_earlier_runs_are_the_ones_in_their_saved_files():
    head = _head(); rh = retry(); sc = {c["label"]: c for c in rh["cells"]}
    for s in ("98.1 % of the steps were exact with it and 98.1 % without it (1,884 of 1,920 each)", "8 of the 12 lines read yes", "the next thing to try is a different feedback, which would be a different test written first",
              "eight jobs missed at some step, and the first miss of each was a reply to an **Add** request; none was a reply to a Delete request", "The five shape slips cost 66 steps, the three number slips 23 (89 in all: 53 at 64 jobs of the third run, 36 in `H0`)",
              "(a service written as an edge from itself to itself with a made-up name: jobs 3, 19 and 101; an extra delete in front of the add: jobs 17 and 62)", "(a coordinate copied from the state list or changed: jobs 4, 20 and 122)",
              "The third run capped the length of a name (16 characters) and that removed the one reply that ran on to 96 tokens", "`kind` compiles 3 expressions, `named` 427 on the fresh jobs"):
        assert s in head, s
    assert sum(x["exact"] for s in sc["H0"]["sessions"] for x in s["log"]) == 1884 and sum(x["exact"] for s in sc["H1"]["sessions"] for x in s["log"]) == 1884
    assert sum(1 for s in third()["cells"] if s["arm"] == "S") == 3 and C.summarize_cell2(s64())["exact_steps"] == 1867
    assert [round(100 * C.summarize_cell2(c)["exact_share"], 1) for c in tlist() if c["arm"] == "S"] == [88.3, 94.6, 97.2]
    assert set(sc) == {"R8", "R32", "R64", "H0", "H1"} and [c["K"] for c in tlist() if c["arm"] == "S"] == [8, 32, 64]
    # the page was written before any reply to a job of 128 to 191 existed: no saved file holds one
    for d in (third(), rh, json.load(open(os.path.join(RES, "axi_provider_constrained.json"))), json.load(open(os.path.join(RES, "axi_provider_side.json")))):
        assert max(s["seed"] - SEED * 1000 for c in d["cells"] for s in c["sessions"]) < 128


# ---------------------------------------------------------------- 10 the notebook
NOTEBOOK = os.path.join(ROOT, "python", "colab", "axi_provider_side_typed.ipynb")
def _nb(): return json.load(open(NOTEBOOK))
def _srcs(): return ["".join(c["source"]) for c in _nb()["cells"]], [c["cell_type"] for c in _nb()["cells"]]


def test_the_notebook_installs_the_pinned_engine_fixes_the_stream_loads_the_engine_checks_runs_the_seven_cells_in_order_and_puts_them_together():
    srcs, kinds = _srcs(); code = [s for s, k in zip(srcs, kinds) if k == "code"]; text = "\n".join(srcs); nb = _nb()
    import ast
    for s in code: ast.parse("\n".join(l for l in s.splitlines() if not l.lstrip().startswith("!")))
    for needle in ("git clone -q --depth 1 https://github.com/gaurshalendra2412/axi-runtime", "provider_side_typed.py", "q_measure.py", "axi_provider_bounded.json", "T.TypedVLLMEngine()", "eng.warm_up()", "T.run_jobs(eng, [0, 1, 2], \"anchor\", \"named\", steps=3)", "T.known_slip_probe(eng)", "STOP:",
                   "T.run_cell(eng, label, progress=progress)", "T.save_cell", "store.persist", "T.merge_files", "T.verdict_typed", "T.print_cells", "T.print_misses", "T.print_verdict", "T.load_third_run", "Q.render_table", "Q.q_row", "files.download", "open_store", "docs/PROVIDER_SIDE_TYPED_TEST.md",
                   "Do not paste any keys", "update 64 commands", "17 reading-rule lines", "axi_provider_t_", "third_run=THIRD", "for label in T.ORDER: go(label)"): assert needle in text, needle
    assert text.count("Do not paste any keys") >= 2 and "!pip -q install vllm==0.31.0" in [x.strip() for x in code] and "!pip -q install vllm" not in [x.strip() for x in code] and "open_store(\"/content/out\", \"axi_provider_typed\")" in text
    fix = [i for i, (s, k) in enumerate(zip(srcs, kinds)) if k == "code" and "fileno" in s]; load = [i for i, (s, k) in enumerate(zip(srcs, kinds)) if k == "code" and "T.TypedVLLMEngine()" in s]; assert len(fix) == 1 and len(load) == 1 and fix[0] < load[0]
    ran = []; ns = dict(go=lambda label: ran.append(label), T=T)
    for s in code:
        if re.fullmatch(r"for label in T\.ORDER: go\(label\)\s*", s.strip()): exec(s, ns)
    assert ran == T.ORDER == ["F0", "FK", "FKA", "FKT", "FN", "SK", "SN"], ran
    assert nb["metadata"]["accelerator"] == "GPU" and nb["nbformat"] == 4 and kinds[0] == "markdown" and kinds.count("code") == 9
    head = srcs[0]; assert "T4" in head and "80,000 tokens" in head and "Qwen2.5-7B-Instruct" in head and "AWQ" in head and "vLLM 0.31.0" in head and "45 to 55 minutes" in head and "Keep this browser tab open" in head and "Choose the T4 even if your plan offers a faster GPU" in head and "the seven cells about 25 minutes" in "\n".join(srcs)
    assert "The kept chat is not run" in head and "no retry in any cell" in head and "98.1 % with it, 98.1 % without" in head and "docs/PROVIDER_SIDE_TYPED_TEST.md" in head and "128 to 191" in head
    assert "A wrong answer is not a STOP" in "\n".join(srcs)
    heads = [x.splitlines()[0] for x, k in zip(srcs, kinds) if k == "markdown" and x.startswith("###")]
    assert [h.split(":")[0] for h in heads] == ["### Cell 1", "### Cell 2", "### Cell 3", "### Cell 4", "### Cell 4b", "### Cell 5", "### Cell 6", "### Cell 7", "### Cell 8", "### Done"], heads
    for needle in ("### Cell 3: install the engine, version 0.31.0 (about 5 minutes)", "press it, then run Cells 2, 3, 4 and 5 again (Cell 3 will then be quick", "### Cell 4: connect Google Drive", "### Cell 4b: fix the output stream (about 1 second). vLLM 0.31.0 asks the notebook for a file number",
                   "before Cell 5", "### Cell 5: load the model and the engine with one expression per request (downloads about 5.5 GB once; about 5 to 10 minutes)", "### Cell 6: SAFETY CHECK (about 3 minutes). 3 jobs of 3 steps with the heaviest variant", "### Cell 7: the seven cells, in this order: `F0`, `FK`, `FKA`, `FKT`, `FN` (jobs 128 to 191: control, kind grammar, kind grammar with the anchor",
                   "then `SK` and `SN` (jobs 0 to 63, kind and kind-and-coordinates)", "### Cell 8: put the finished cells together", "(the longest takes about 6 minutes)", "In the fourth run (test 17) one retry did not raise the exactness on", "Eight jobs in the first 128 slipped, every first slip on an **Add** r",
                   "finish the update 64 commands (git push) first", "### Cell 2: get the code from GitHub. If it stops with 'the code is not on GitHub yet', finish the update 64 commands"):
        assert needle in text, needle
    assert nb["metadata"]["kernelspec"]["display_name"] == "Python 3" and nb["metadata"]["kernelspec"]["name"] == "python3"
    assert text.count("finish the update 64 commands") == 3 and text.count("finish the update 64 commands (git push) first") == 2, "the heading of Cell 2 and the check inside it say the same"


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
        shim = types.SimpleNamespace(CELLS=T.CELLS, run_cell=lambda e, label, progress=None: T.run_cell(e, label, steps=3, progress=progress), save_cell=T.save_cell, summarize_cell4=T.summarize_cell4)
        def session(engine):
            store = Store(d, remote); ns = dict(T=shim, eng=engine, store=store, os=os, time=time, json=json); exec(body, ns); return ns["go"]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            go = session(Dying(die_at=2))
            try: go("F0")
            except Crash: pass
            assert os.listdir(d) == [] and os.listdir(remote) == []                                                       # nothing half-finished is kept: the cell starts again from its beginning
            go = session(T.ScriptedStandIn()); go("F0"); go("SN"); go("F0")
        out = buf.getvalue()
        assert out.count("already finished, skipping") == 1 and out.count("done:") == 2 and "=== F0: jobs 128 to 191, prompt old, grammar capped" in out and "=== SN: jobs 0 to 63, prompt old, grammar named" in out
        s = T.summarize_cell4(T.run_cell(T.ScriptedStandIn(), "F0", steps=3)); f = lambda x, k=1: "-" if x is None else f"{x:.{k}f}"
        want = (f"done: {s['steps_done']} steps, {s['wall_seconds']:.0f} s, {f(s['seconds_per_step'], 3)} s/step, computed prompt tokens per step {f(s['computed_per_step'], 0)}, exact steps {s['exact_share'] * 100:.1f} %, jobs exact to the end {s['jobs_exact_to_end']}, "
                f"outside the language {s['out_of_language']}, refused by the strict parser {s['unparsed']}, {s['distinct_regexes']} expressions, exact steps per hour {f(s['exact_per_hour'], 0)}")
        assert want + "\n" in out, (want, out)
        assert sorted(os.listdir(remote)) == ["axi_provider_t_F0.json", "axi_provider_t_SN.json"] and sorted(os.listdir(d)) == sorted(os.listdir(remote))


def test_the_notebooks_last_cell_puts_the_cells_together_reads_the_third_runs_file_prints_the_lines_the_misses_and_q_and_offers_the_download():
    srcs, kinds = _srcs(); cell = next(s for s, k in zip(srcs, kinds) if k == "code" and "T.merge_files" in s)
    assert cell.count("/content/axi/docs/results/axi_provider_bounded.json") == 1
    cell = cell.replace("/content/axi/docs/results/axi_provider_bounded.json", THIRD_FILE)
    sys.path.insert(0, os.path.join(HERE, ".."))
    from colab.colab_persist import Store
    import axi.experiments.q_measure as Qm
    got = []; gc = types.ModuleType("google.colab"); gc.files = types.SimpleNamespace(download=lambda p: got.append(p)); g = types.ModuleType("google"); g.colab = gc
    saved = {k: sys.modules.get(k) for k in ("google", "google.colab")}
    sys.modules["google"], sys.modules["google.colab"] = g, gc
    try:
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as remote:
            store = Store(d, remote)
            for lab in T.ORDER: T.save_cell(T.run_cell(T.ScriptedStandIn(faults={("old", seed_of(3), 2): "ADD[(4,4)->(4,4):value#6]"}), lab, steps=3), store.path(f"axi_provider_t_{lab}.json"))
            buf = io.StringIO(); ns = dict(T=T, Q=Qm, store=store, glob=glob, os=os)
            with contextlib.redirect_stdout(buf): exec(cell, ns)
            out = buf.getvalue(); assert "7 cells:" in out and "-- provider-side test, typed requests" in out and out.count("[claim]") == 13 and out.count("[guard]") == 2 and out.count("[instrument]") == 2
            assert ("never: a multiple of capacity" in out) != ("NO claim is made from this run" in out)
            assert "**Q = O / M of the cells of this run**" in out and "| F0 |" in out and "| SN |" in out and "exact steps per GPU hour" in out and "first misses" in out or "first misses" not in out
            assert got == [store.path("axi_provider_typed.json")] and os.path.exists(os.path.join(remote, "axi_provider_typed.json"))
            m = json.load(open(store.path("axi_provider_typed.json"))); assert [c["label"] for c in m["cells"]] == T.ORDER and m["third_run_sha256"] == THIRD_SHA256
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
        assert "cache size the engine will use: 80000 tokens ( override )" in out and callable(ns["go"]) and isinstance(ns["eng"], T.TypedVLLMEngine) and ns["T"] is T and calls["resets"] >= 1 and '"constraint"' not in out and '"kind": "vllm"' in out


def test_the_safety_cell_says_ok_on_a_working_engine_and_stop_for_each_thing_that_can_be_wrong_and_a_wrong_answer_of_the_probe_is_not_a_stop():
    srcs, kinds = _srcs(); cell = next(s for s, k in zip(srcs, kinds) if k == "code" and "smoke = " in s)
    def run(engine):
        buf = io.StringIO(); ns = dict(T=T, eng=engine); ns["re"] = re
        with contextlib.redirect_stdout(buf): exec(cell, ns)
        return buf.getvalue()
    out = run(T.TypedStandIn()); last = out.strip().splitlines()[-1]
    assert last.startswith("OK: cached counts are reported") and "STOP" not in out and "job 4, step 18 (add_node)" in out and "= the oracle" in out
    assert "the parity turn was: 'PARITY request TARGET (0,0) ADD NODE value 3; reply (1,0) ADD NODE value 6. Reply with only the corrected delta.'" in out and "kind+top" in out and "kind+tail" in out
    class NoCount(T.TypedStandIn):
        def generate_round(self, requests, reset_cache=False): reps, dt = super().generate_round(requests, reset_cache); [setattr(r, "n_cached", None) for r in reps]; return reps, dt
    assert "STOP: this vLLM version does not report cached tokens" in run(NoCount())
    def says(text, n_new, when):
        class P(T.TypedStandIn):
            def generate_round(self, requests, reset_cache=False):
                reps, dt = super().generate_round(requests, reset_cache)
                if when(requests):
                    for r in reps: r.text = text; r.n_new = n_new
                return reps, dt
        return P()
    smoke = lambda rs: len(rs) == 3; probe = lambda rs: len(rs) == 48
    assert "STOP: a reply of the safety cell was refused by the strict parser" in run(says("Sure, here is the delta", 20, smoke))
    assert "STOP: a reply of the safety cell is outside the expression of its own request" in run(says("DEL[1,2]", 4, smoke))
    assert "STOP: a reply of the probe is refused by the strict parser" in run(says("Sure, here is the delta", 20, probe))
    o = run(says("DEL[1,2]", 4, probe)); assert "STOP: a reply of the probe is outside the expression of its own request" in o
    o = run(says("ADD[(3,0)->(3,0):4tworowsdown4col#4]", 96, lambda rs: len(rs) == 48)); assert "STOP" in o.strip().splitlines()[-1]
    class Wrong(T.TypedStandIn):                                                                                # a wrong answer that is inside the language and ends: not a STOP
        def generate_round(self, requests, reset_cache=False):
            reps, dt = super().generate_round(requests, reset_cache)
            if len(requests) == 48:
                for rq, r in zip(requests, reps):
                    if rq.shape == "named" and rq.seed == seed_of(4): r.text = "ADD[0,0:6]"
            return reps, dt
    o = run(Wrong()); assert o.strip().splitlines()[-1].startswith("OK:") and "STOP" not in o and "'ADD[0,0:6]'" in o
    class Garbage(T.TypedStandIn):
        def generate_round(self, requests, reset_cache=False): reps, dt = super().generate_round(requests, reset_cache); reps[0].text = "Sure, here is the delta: " + reps[0].text; return reps, dt
    assert "STOP: a reply of the safety cell was refused by the strict parser" in run(Garbage())


# ---------------------------------------------------------------- 11 the front pages
FRONT_ITEM = ("111. **The serving side a fifth time: the grammar of the reply chosen by the type of the request, the anchor line of item 110 as two extra cells, and Q = O / M from the saved files (20 Oct)**", "**Status: RUN (the owner's Colab run, seven cells in one session; the file reached the repository work on 20 Oct 2026); the Result is on the page `PROVIDER_SIDE_TYPED_TEST.md`, which was hash-locked before the run (sha256 " + PAGE_SHA256[:16] + "... of the text above the Result)",
              "Seven cells of 64 jobs, arm `S`, no retry anywhere", "all on jobs 128 to 191, which no earlier run has seen; `SK` and `SN` on jobs 0 to 63", "the headline is lines 6 and 17", "all 17 yes under 1 %",
              "of the eight first misses in jobs 0 to 127 five have a shape the kind grammar removes (jobs 3, 17, 19, 62, 101; 66 steps) and three have the right shape and a wrong number (jobs 4, 20, 122; 23 steps)", "(17,280 requests)", "in `SK` 1,860 steps are compared and jobs 4 and 20 must slip again at steps 18 and 30, in `SN` 1,848 are compared",
              "the ceilings are 99.3 % and 100.0 % against 97.2 %", "no step of the 5,040 saved steps of the four cells without a retry at which the job held the right state afterwards", "about 181 computed tokens a step against 109",
              "no weight between a second and a token is invented, no floor of M is claimed", "**What it cannot do:** it promises no 100 %", "nothing is trained", "`simulate_eval_suite` in one blueprint draws random numbers and prints them as accuracies", "a \"5.4 times\" figure divides the 64-job exact-state cell by the 8-job kept-chat cell",
              "no deleted service of the 192 scripts has more than 3 connections", "**Credit:**", "(test 130, 26 tests)", "(test 131, 7 tests)", "nothing here is proven")
FRONT_ROW = ("**Run (the owner's Colab run, seven cells in one session, Tesla T4, vLLM 0.31.0; the file reached the repository work on 20 Oct 2026).**", "(sha256 " + PAGE_SHA256[:16] + "... of the locked text)", "never a multiple of capacity, a count of GPUs saved, dollars, \"100 percent\" or \"proven\"",
             "(1,860 and 1,848 steps compared)", "no false alarm on 5,040 saved steps", "compared with its page character by character (test 131, 7 tests)", "(test 130, 26 tests)", "Not shown: that a grammar raises the exactness on new jobs (the kind grammar did not)")
FRONT_NEXT = "Next free: row 171, item 116, test 141, question 121"


def test_the_front_pages_point_to_the_test_and_say_what_the_page_expects_and_what_it_does_not_show():
    cm = open(os.path.join(ROOT, "docs", "RAJNISH_CONCEPT_MAP.md")).read(); ve = open(os.path.join(ROOT, "docs", "VERIFIED.md")).read()
    assert "PROVIDER_SIDE_TYPED_TEST.md" in cm and "PROVIDER_SIDE_TYPED_TEST.md" in ve
    item = cm[cm.index("111. **"): cm.index("112. **")]; k = ve.index("Provider-side serving test, fifth run"); row = ve[k: ve.index("\n", k)]
    for s in FRONT_ITEM: assert s in item, s[:100]
    for s in FRONT_ROW: assert s in row, s[:100]
    for text in (item, row):
        assert "@@" not in text and "mutation check of test 130: " in text and "mutation check of test 131: " in text and "capacity" in text and "NOT RUN" not in text and "deliberate changes" in text
        assert "proven" in text and "dollars" in text and "energy" in text
    assert FRONT_NEXT in cm and "test 130 checks the typed-requests test, test 131 the Q = O / M module and test 132 the Result of the typed-requests test, item 111; test 133 checks the 100-step serving test, item 112, and test 134 its Result" in cm
    assert cm.index("110. **") < cm.index("111. **") < cm.index("Next free: row") and ve.index("Provider-side serving test, fourth run") < k
    for name in ("test_provider_side.py", "test_provider_side_constrained.py", "test_provider_side_bounded.py", "test_provider_side_bounded_result.py"):
        t = open(os.path.join(HERE, name)).read(); assert "row 170, item 112, test 133" not in t and "row 171, item 116, test 141" in t, name
    assert 'FRONT_NEXT = "Next free: row 171, item 116, test 141, question 121"' in open(os.path.join(HERE, "test_provider_side_retry.py")).read()


if __name__ == "__main__":
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]; bad = 0
    for n, f in fns:
        try: f(); print("ok  ", n)
        except Exception: bad += 1; print("FAIL", n); traceback.print_exc()
    print(f"{len(fns) - bad}/{len(fns)} passed"); sys.exit(1 if bad else 0)
