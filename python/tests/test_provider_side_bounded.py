"""Engineering support for docs/PROVIDER_SIDE_BOUNDED_TEST.md (concept-map item 108): the exact-state cells of the provider-side serving test run again with the length of a relation name capped at 16.
Checks that need no GPU and no model:
  1. the capped expression is the second run's with the name `{1,16}` and nothing else; it accepts what the repository's parser reads, refuses a name of 17, and is inside the second run's language;
  2. the claims on the page about the second run's files, recomputed from them: which replies are outside the new language (2, the same one), the longest name, the first run against the second run
     (a reply the grammar allows is not changed by it), the same replies at every K, what the looping reply cost in time, why a capped reply leaves job 19 where it was, the estimate table;
  3. the vLLM adapter against a fake module: the capped constraint goes into every request, a version that cannot constrain fails before the model is loaded, the info records the cap;
  4. the planned cells and the merge (it refuses the second run's cells, the first run's, a cell twice and a cell this test does not plan);
  5. the runner is the second run's runner, and a checkpoint of the second run's constraint is refused;
  6. the probe: the exact-state prompt of step 12 of job 19;
  7. the 13 lines at their edges on the second run's own cells (a good new run is built from them), 'not run' for a missing cell, the rule about what may be claimed, the printout;
  8. the locked page, the notebook and its safety cell, the front pages.
What it does NOT show: anything about a real engine, a real GPU, or what a model answers."""
import contextlib, copy, glob, hashlib, io, json, os, random, re, subprocess, sys, tempfile, time, types

sys.path.insert(0, os.path.dirname(__file__))
import axi.experiments.provider_side as PS
import axi.experiments.provider_side_constrained as C
import axi.experiments.provider_side_bounded as B
import test_provider_side_constrained as T124                                       # its fake `vllm` and its walkers are reused, not copied
from axi.engine.cico_decoder import CICOGrammarDFA
from axi.experiments.agent_loop import strict_parse

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..", "..")
RES = os.path.join(ROOT, "docs", "results")
REGEX_SHA256 = "6f4e5a993f26ccd471bff13422fa622b7820858a91e05d43c2bccfbe6b864245"
SECOND_FILE = os.path.join(RES, "axi_provider_constrained.json")
SECOND_SHA256 = "7f54c3dd37fd4dcf61e8a7991af285d6ad1a0d691362496216949a26b45b819d"
PS_SHA256 = "bfb4734e171d9244cd47d10402be46528a557095a8275d02b38481a0e2b0b1d3"
C_SHA256 = "03452148e28c9f31372f12cc11ef1808ac67308adff972fa19d4aeeef7c96f7d"
LOOP_TEXT_START = "ADD[(3,0)->(3,0):4tworowsdown4colsright4tworowsdown4colsrigh"


def raises(fn, exc=Exception):
    try: fn()
    except exc: return True
    return False


def _sha(path): return hashlib.sha256(open(path, "rb").read()).hexdigest()


_CACHE = {}


def second():
    if "second" not in _CACHE: _CACHE["second"] = json.load(open(SECOND_FILE))
    return _CACHE["second"]


def first():
    if "first" not in _CACHE: _CACHE["first"] = json.load(open(os.path.join(RES, "axi_provider_side.json")))
    return _CACHE["first"]


def oldcells(): return {(c["arm"], c["K"]): c for c in second()["cells"]}
def logs(c): return [x for s in c["sessions"] for x in s["log"]]


# ---------------------------------------------------------------- 1 the capped expression
def _names(t): return re.findall(r":([A-Za-z0-9_]*)#", t)


def test_the_capped_expression_is_the_second_runs_with_the_name_at_most_16_and_nothing_else_and_it_is_inside_the_second_runs_language():
    assert B.NAME_CAP == 16 and C.REGEX.count("[A-Za-z0-9_]+") == 2 and B.REGEX == C.REGEX.replace("[A-Za-z0-9_]+", "[A-Za-z0-9_]{1,16}") and B.REGEX.count("{1,16}") == 2
    assert hashlib.sha256(B.REGEX.encode()).hexdigest() == B.REGEX_SHA256 == REGEX_SHA256 != C.REGEX_SHA256
    rx, old, dfa, rnd = re.compile(B.REGEX), re.compile(C.REGEX), CICOGrammarDFA(), random.Random(7)
    def item(L):
        name = "".join(rnd.choice("abcXYZ019_") for _ in range(L)); a, b, c, d = (rnd.randint(0, 12) for _ in range(4)); v = rnd.randint(-9, 99)
        return rnd.choice([f"ADD[{a},{b}:{v}]", f"DEL[{a},{b}]", f"ADD[({a},{b})->({c},{d}):{name}#{v}]", f"DEL[({a},{b})->({c},{d}):{name}#{v}]"])
    n_long = n_ok = n_edge = n_mut = 0
    for _ in range(9000):
        s = "".join(item(rnd.randint(1, 22)) + rnd.choice([" ", "\n", " ", "  "]) for _ in range(rnd.randint(1, 3))).rstrip(" \n") + rnd.choice(["", "", " ", "\n"])
        if rnd.random() < 0.3:
            s = list(s)
            for _ in range(rnd.randint(1, 2)): s[rnd.randrange(len(s))] = rnd.choice("ADEL[]()0123456789,:->#_ \nxyz")
            s = "".join(s); n_mut += 1
        want = bool(old.fullmatch(s)) and all(len(n) <= 16 for n in _names(s)); got = bool(rx.fullmatch(s)); assert got == want, s
        if old.fullmatch(s): assert dfa.accepts(s.encode()) and strict_parse(s) is not None, s                        # inside the repository's language, and the parser reads it
        if got: assert old.fullmatch(s) and strict_parse(s) is not None; n_ok += 1
        if old.fullmatch(s) and not got: n_long += 1                                                                    # refused by the cap and by nothing else
        if got and any(len(n) == 16 for n in _names(s)): n_edge += 1
    assert n_ok > 1500 and n_long > 800 and n_edge > 100 and n_mut > 2000, (n_ok, n_long, n_edge, n_mut)
    for L in range(1, 17):
        for kind in ("ADD", "DEL"): assert rx.fullmatch(f"{kind}[(3,0)->(3,0):{'a' * L}#1]"), (kind, L)
    for L in (17, 18, 40):
        for kind in ("ADD", "DEL"): assert not rx.fullmatch(f"{kind}[(3,0)->(3,0):{'a' * L}#1]") and old.fullmatch(f"{kind}[(3,0)->(3,0):{'a' * L}#1]")
    assert not rx.fullmatch("ADD[(3,0)->(3,0):#1]") and not rx.fullmatch("ADD[(3,0)->(3,0):a-b#1]") and not rx.fullmatch("ADD[(3,0)->(3,0):" + "a" * 16 + "b#1]")
    assert rx.fullmatch("DEL[4,5] ADD[(3,0)->(3,0):" + "a" * 16 + "#1]") and not rx.fullmatch("DEL[4,5] ADD[(3,0)->(3,0):" + "a" * 17 + "#1]")           # the cap holds in every item of a list
    assert rx.fullmatch("ADD[4,4:3]") and rx.fullmatch("DEL[4,5]") and not rx.fullmatch("DEL[4,5:8]") and not rx.fullmatch("ADD[(4,4):6]") and not rx.fullmatch("")          # nothing else changed


# ---------------------------------------------------------------- 2 what the page says about the second run's files
def test_only_the_two_copies_of_one_reply_of_the_second_run_are_outside_the_new_language_and_the_names_are_short():
    rx = re.compile(B.REGEX); out = []; n = 0; longest_ok = 0; longest_name = 0
    for k, c in oldcells().items():
        for j, s in enumerate(c["sessions"]):
            for x in s["log"]:
                n += 1
                if x["parsed"]: longest_ok = max(longest_ok, x["n_new"]); longest_name = max([longest_name] + [len(m) for m in _names(x["text"])])
                if not rx.fullmatch(x["text"]): out.append((k, j, x["t"], x["n_new"], x["parsed"], x["text"]))
    assert n == 6231 and len(out) == 2 and [(k, j, t, nn, p) for k, j, t, nn, p, _ in out] == [(("S", 32), 19, 12, 96, False), (("S", 64), 19, 12, 96, False)] and out[0][5] == out[1][5] and out[0][5].startswith(LOOP_TEXT_START)
    assert longest_ok == 31 and longest_name == 5
    for k in (("KC", 8), ("KC", 32), ("KC", 64), ("S", 8)): assert all(rx.fullmatch(x["text"]) for x in logs(oldcells()[k]))                      # the kept chat's cells and S at 8 jobs have no reply outside it
    names = set()
    for e in range(64):
        s = PS.Session("S", PS.SEED * 1000 + e, PS.STEPS); names |= {m for st in s.script for m in _names(st["ideal"])}
    assert names == {"dep", "owns"} and max(map(len, names)) == 4                                                                                   # the oracle's own replies use two short names
    loop = out[0][5]; name = loop[len("ADD[(3,0)->(3,0):"):]; assert name.startswith("4tworowsdown4col") and len(name) > 40 and name[:16] == "4tworowsdown4col" and len(name[:16]) == 16
    for k in range(1, 17): assert rx.fullmatch("ADD[(3,0)->(3,0):" + name[:k] + "#1]"), k                                                            # every cut of the looping name up to 16 characters can be closed
    for suffix in ("#1]", "a#1]", "4#1]"): assert not rx.fullmatch("ADD[(3,0)->(3,0):" + name[:17] + suffix)                                          # and a 17th character cannot
    assert not rx.fullmatch(loop)


def test_job_19_stays_where_it_was_whatever_the_capped_reply_is_and_the_gate_refuses_every_such_edge():
    old = oldcells()[("S", 32)]["sessions"][19]["log"]; assert [x["exact"] for x in old] == [True] * 11 + [False] * 19 and old[11]["text"].startswith(LOOP_TEXT_START) and not old[11]["parsed"] and old[11]["n_new"] == 96
    s = PS.Session("S", PS.SEED * 1000 + 19, PS.STEPS)
    while s.t < 11: rq = s.request(); s.accept(PS.Reply(rq.stand_in_reply, 0, 0, 0))
    rq = s.request(); assert s.script[s.t]["kind"] == "add_node" and rq.stand_in_reply == "ADD[3,0:4]"
    for name in ("4tworowsdown4col", "dep", "owns", "value", "x"):
        for kind in ("ADD", "DEL"):
            for v in ("1", "-1", "12"):
                t = f"{kind}[(3,0)->(3,0):{name}#{v}]"; d = strict_parse(t); assert d is not None and re.fullmatch(B.REGEX, t)
                ck = PS.check(s.live, d); assert not ck.ok, (t, ck)                                                                                  # the node (3,0) is not in the graph yet
                if kind == "ADD": assert "endpoint missing" in ck.reason, (t, ck.reason)
    def replay(text12):
        s = PS.Session("S", PS.SEED * 1000 + 19, PS.STEPS); msgs = []
        while not s.done:
            rq = s.request(); msgs.append(rq.messages); s.accept(PS.Reply(text12 if s.t == 11 else old[s.t]["text"], 0, 0, 0))
        return msgs, [(x["parsed"], x["admitted"], x["corrupted"], x["exact"], x["distance"]) for x in s.log]
    mo, fo = replay(old[11]["text"]); assert fo == [(x["parsed"], x["admitted"], x["corrupted"], x["exact"], x["distance"]) for x in old], "the replay does not reproduce the saved job"
    for capped in ("ADD[(3,0)->(3,0):4tworowsdown4col#1]", "ADD[(3,0)->(3,0):dep#1]", "DEL[(3,0)->(3,0):owns#2]"):
        mc, fc = replay(capped)
        assert mc == mo and len(mc) == 30, capped                                                                                                    # the same 30 prompts: steps 13 to 30 get the same prompt as before
        assert [f[1:] for f in fc] == [f[1:] for f in fo] and fc[11][0] is True and fo[11][0] is False                                              # parsed now, and the same admitted, corrupted, exact, distance at every step


def test_a_reply_the_grammar_allows_is_not_changed_by_it_in_the_files_the_page_quotes():
    c1 = {(c["arm"], c["K"]): c for c in first()["cells"]}; c2 = oldcells(); rx = re.compile(B.REGEX); comp = same = diff_in = diff_out = 0
    for k in c2:
        if k not in c1: continue
        for a, b in zip(c1[k]["sessions"], c2[k]["sessions"]):
            for x, y in zip(a["log"], b["log"]):
                comp += 1
                if x["text"] == y["text"]: same += 1; continue
                if rx.fullmatch(x["text"]): diff_in += 1
                else: diff_out += 1
                break
    assert (comp, same, diff_in, diff_out) == (3513, 3463, 0, 50) and sorted(k for k in c2 if k in c1) == [("KC", 8), ("KC", 32), ("S", 8), ("S", 32), ("S", 64)]
    for arm in ("S", "KC"):                                                                                                                          # the first 8 jobs: 8 jobs against 32 and against 64, 240 steps each, the same text
        n = bad = 0
        for j in range(8):
            for K in (32, 64):
                for x, y in zip(c2[(arm, 8)]["sessions"][j]["log"], c2[(arm, K)]["sessions"][j]["log"]): n += 1; bad += x["text"] != y["text"]
        assert (n, bad) == (480, 0), (arm, n, bad)


def test_the_time_the_looping_reply_cost_and_the_estimate_table_are_the_ones_the_page_gives():
    c = oldcells(); S = C.summarize_all(second()["cells"]); est = {}
    for K, (r11, r12, r13, share) in {32: (3.79, 6.13, 3.37, "2.2"), 64: (6.77, 9.13, 6.32, "1.3")}.items():
        r = [x["seconds"] for x in c[("S", K)]["rounds"]]; assert [round(r[10], 2), round(r[11], 2), round(r[12], 2)] == [r11, r12, r13]
        extra = r[11] - (r[10] + r[12]) / 2; assert 2.5 < extra < 2.6, extra
        assert f"{extra / c[('S', K)]['wall_seconds'] * 100:.1f}" == share, (K, extra)
        wall = c[("S", K)]["wall_seconds"] - extra; ex = S[("S", K)]["exact_steps"]; est[K] = (round(wall, 1), round(3600 * ex / wall), round(3600 * ex / wall / S[("KC", K)]["exact_per_hour"], 1))
    assert est == {32: (111.2, 29407, 40.4), 64: (201.7, 33318, 49.9)}, est
    assert {k: (S[k]["exact_steps"], S[k]["steps_done"], round(S[k]["wall_seconds"], 1), round(S[k]["exact_per_hour"])) for k in S} == {("S", 8): (212, 240, 36.6, 20878), ("KC", 8): (240, 240, 145.9, 5923), ("S", 32): (908, 960, 113.7, 28748),
        ("S", 64): (1867, 1920, 204.3, 32896), ("KC", 32): (956, 956, 4730.9, 727), ("KC", 64): (1915, 1915, 10330.5, 667)}
    assert round(S[("S", 8)]["exact_per_hour"] / S[("KC", 8)]["exact_per_hour"], 1) == 3.5 and f"{4730.9 / 3600:.2f}" == "1.31" and f"{10330.5 / 3600:.2f}" == "2.87"
    assert 240 + 240 + 960 + 1920 == 3360 and [round(S[k]["exact_share"] * 100, 1) for k in (("S", 8), ("S", 32), ("S", 64))] == [88.3, 94.6, 97.2]


# ---------------------------------------------------------------- 3 the vLLM adapter, against a fake module
def test_the_vllm_adapter_puts_the_capped_constraint_into_every_request_and_records_the_cap():
    for how in ("module", "top"):
        mods, calls = T124._fake_vllm(how)
        with T124._installed(mods):
            e = B.BoundedVLLMEngine(); kw = calls["init"][0]
            assert kw == dict(model=PS.MODEL, dtype="half", max_model_len=8192, gpu_memory_utilization=0.90, enable_prefix_caching=True, seed=0, num_gpu_blocks_override=5000)
            sp = e.sp.kw; assert set(sp) == {"temperature", "max_tokens", "structured_outputs"} and sp["temperature"] == 0.0 and sp["max_tokens"] == 96 and sp["structured_outputs"].kw == dict(regex=B.REGEX)
            assert sp["structured_outputs"].kw["regex"].count("{1,16}") == 2 and "[A-Za-z0-9_]+" not in sp["structured_outputs"].kw["regex"]
            i = e.info(); assert i["kind"] == "vllm" and i["capacity_tokens"] == 80000 and i["capacity_source"] == "override"
            assert i["constraint"] == dict(kind="regex", api="structured_outputs(regex)", regex=B.REGEX, regex_sha256=B.REGEX_SHA256, name_cap=16)
            e.warm_up(); assert calls["generate"][-1][1] is e.sp and calls["resets"] == 1
            c = B.run_cell(e, "S", 2, steps=2); assert c["engine"]["constraint"]["regex_sha256"] == B.REGEX_SHA256 and all(g[1] is e.sp for g in calls["generate"])
    mods, calls = T124._fake_vllm(None)
    with T124._installed(mods):
        try: B.BoundedVLLMEngine()
        except RuntimeError as err: assert "no StructuredOutputsParams" in str(err) and "does not fall back" in str(err)
        else: raise AssertionError("an engine that cannot constrain must refuse to start")
        assert calls["init"] == []
    mods, calls = T124._fake_vllm("module")
    with T124._installed(mods):
        e2 = C.ConstrainedVLLMEngine(); assert e2.sp.kw["structured_outputs"].kw == dict(regex=C.REGEX) and e2.info()["constraint"]["regex_sha256"] == C.REGEX_SHA256       # the second run's engine is unchanged
    assert B.constraint_info("stand_in")["name_cap"] == 16 and "name_cap" not in C.constraint_info("stand_in")


# ---------------------------------------------------------------- 4 cells and merge
def test_the_planned_cells_are_the_four_cheap_ones_and_the_merge_refuses_what_does_not_belong():
    assert B.cells_planned() == [("S", 8), ("KC", 8), ("S", 32), ("S", 64)] and B.ARMS == ("S", "KC") and B.LEVELS == (8, 32, 64) and (B.STEPS, B.SEED, B.POLICY, B.HINT_RULES, B.MAX_NEW, B.MODEL, B.CAPACITY_TOKENS) == (PS.STEPS, PS.SEED, PS.POLICY, PS.HINT_RULES, PS.MAX_NEW, PS.MODEL, PS.CAPACITY_TOKENS)
    assert B.run_cell is C.run_cell and B.summarize_cell2 is C.summarize_cell2 and B.summarize_all is C.summarize_all and B.save_cell is C.save_cell
    eng = B.BoundedStandIn(); cells = [B.run_cell(eng, a, k, steps=2) for a, k in reversed(B.cells_planned()[:3])]
    m = B.merge_cells(cells); assert [(c["arm"], c["K"]) for c in m["cells"]] == [("S", 8), ("KC", 8), ("S", 32)] and m["meta"]["name_cap"] == 16 and m["meta"]["cells"] == [list(k) for k in B.cells_planned()]
    plain = PS.run_cell(PS.StandInEngine(), "S", 2, steps=2); assert raises(lambda: B.merge_cells([plain]), ValueError)                                 # the first run's cell (no constraint)
    uncapped = C.run_cell(C.ConstrainedStandIn(), "S", 8, steps=2); assert raises(lambda: B.merge_cells([uncapped]), ValueError)                       # the second run's constraint
    assert raises(lambda: B.merge_cells([cells[0], cells[0]]), ValueError) and raises(lambda: B.merge_cells([]))
    kc32 = B.run_cell(eng, "KC", 32, steps=2); assert raises(lambda: B.merge_cells([kc32]), ValueError)                                                  # not one of this test's cells
    for field, val in (("seed", 1), ("steps", 7)):
        o = copy.deepcopy(cells[1]); o[field] = val; assert raises(lambda: B.merge_cells([cells[0], o]), ValueError), field
    for field in ("version", "gpu"):
        o = copy.deepcopy(cells[1]); o["engine"][field] = "other"; assert raises(lambda: B.merge_cells([cells[0], o]), ValueError), field
    with tempfile.TemporaryDirectory() as d:
        paths = [B.save_cell(c, os.path.join(d, f"{c['arm']}{c['K']}.json")) for c in cells]; out = os.path.join(d, "m.json"); mm = B.merge_files(paths, out, second_run=SECOND_FILE); back = json.load(open(out))
        assert back["module_sha256"] == B.fingerprint() == _sha(B.__file__) and back["constrained_sha256"] == C.fingerprint() == C_SHA256 == _sha(C.__file__) and back["provider_side_sha256"] == PS.fingerprint() == PS_SHA256 == _sha(PS.__file__)
        assert back["second_run_sha256"] == SECOND_SHA256 and back["second_run_file"] == "axi_provider_constrained.json" and back["cells"] == mm["cells"]
        assert B.default_update58_path() and os.path.samefile(B.default_update58_path(), SECOND_FILE)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf): B.print_cells(B.summarize_all(back["cells"])); B.print_verdict(B.verdict_bounded(back["cells"], second()["cells"]))
        assert "not run" in buf.getvalue() and "NOT a result" not in buf.getvalue()
    assert _sha(SECOND_FILE) == SECOND_SHA256


# ---------------------------------------------------------------- 5 the runner and its checkpoint
def test_the_runner_is_the_second_runs_and_a_checkpoint_of_the_second_runs_constraint_is_refused():
    for arm, k in (("S", 3), ("KC", 3)):
        a = C.run_cell(C.ConstrainedStandIn(), arm, k, steps=6); b = B.run_cell(B.BoundedStandIn(), arm, k, steps=6)
        assert a["sessions"] == b["sessions"] and a["rounds"] == b["rounds"] and b["resumed_at"] == [] and b["engine"]["constraint"]["name_cap"] == 16
    with tempfile.TemporaryDirectory() as d:
        ck = os.path.join(d, "x.ckpt.json")
        class Crash(Exception): pass
        class Dying(B.BoundedStandIn):
            n = 0
            def generate_round(self, requests, reset_cache=False):
                Dying.n += 1
                if Dying.n == 3: raise Crash()
                return super().generate_round(requests, reset_cache)
        assert raises(lambda: B.run_cell(Dying(), "KC", 3, steps=5, checkpoint=ck), Crash) and len(json.load(open(ck))["rounds"]) == 2
        whole = B.run_cell(B.BoundedStandIn(), "KC", 3, steps=5); resumed = B.run_cell(B.BoundedStandIn(), "KC", 3, steps=5, checkpoint=ck)
        T124._same_jobs(resumed, whole); assert resumed["resumed_at"] == [2] and len(resumed["rounds"]) == len(whole["rounds"])                    # the same jobs and replies (the first resumed round has a cold cache, so its cached counts differ)
        assert raises(lambda: C.run_cell(C.ConstrainedStandIn(), "KC", 3, steps=5, checkpoint=ck), ValueError)                                            # the second run's constraint cannot continue this cell
        assert raises(lambda: B.run_cell(B.BoundedStandIn(), "S", 3, steps=5, checkpoint=ck), ValueError)                                                 # nor can another cell
    assert raises(lambda: B.run_cell(B.BoundedStandIn(), "KN", 2, steps=2), AssertionError)


# ---------------------------------------------------------------- 6 the probe
def test_the_probe_feeds_the_engine_the_prompt_of_step_12_of_job_19_with_the_oracles_own_earlier_replies():
    class Rec(B.BoundedStandIn):
        def __init__(self): super().__init__(); self.seen = []
        def generate_round(self, requests, reset_cache=False): self.seen.append(requests); return super().generate_round(requests, reset_cache)
    e = Rec(); p = B.loop_probe(e)
    assert (B.LOOP_JOB, B.LOOP_STEP) == (19, 12) and len(e.seen) == 1 and len(e.seen[0]) == 1 and (p["episode"], p["step"], p["kind"]) == (19, 12, "add_node") and p["parsed"] and p["reply"] == p["oracle"] == "ADD[3,0:4]"
    msgs = e.seen[0][0].messages; assert [m["role"] for m in msgs] == ["system", "user"] and "Request:" in msgs[1]["content"]
    s = PS.Session("S", PS.SEED * 1000 + 19, PS.STEPS)
    while s.t < 11: s.request(); s.accept(PS.Reply(s.script[s.t]["ideal"], 0, 0, 0))
    assert msgs == s.request().messages
    old = oldcells()[("S", 32)]["sessions"][19]["log"]; assert all(x["exact"] for x in old[:11]) and old[11]["t"] == 12 and not old[11]["parsed"] and old[11]["n_new"] == 96                  # the job was exact up to step 11: the state the probe rebuilds is the one it had
    first_ = oldcells()[("S", 64)]["sessions"][19]["log"][11]; assert first_["text"] == old[11]["text"]


# ---------------------------------------------------------------- 7 the lines
def _good():
    """A new run that satisfies the second run's own cells: S at 8, 32, 64 and KC at 8 as in the second run, with job 19, step 12 replaced by the capped reply. The lines it reads: yes for 1 to 11, NO for 12 and 13."""
    o = oldcells(); new = []
    for k in B.cells_planned():
        c = copy.deepcopy(o[k])
        if len(c["sessions"]) > B.LOOP_JOB:
            r = c["sessions"][B.LOOP_JOB]["log"][B.LOOP_STEP - 1]; r.update(text="ADD[(3,0)->(3,0):4tworowsdown4col#1]", n_new=30, parsed=True, admitted=False)
        new.append(c)
    return new


def _old(): return second()["cells"]
def _read(new, old=None): return [ok for _, _, _, ok in B.verdict_bounded(new, _old() if old is None else old)]
def _cell_of(cells, arm, K): return next(c for c in cells if (c["arm"], c["K"]) == (arm, K))
def _all_exact(c):
    for x in logs(c): x["exact"] = True
def _not_exact(c, n):
    k = 0
    for x in logs(c):
        if k >= n: break
        if x["exact"]: x["exact"] = False; k += 1
def _old_with(cell_key, **kw):
    out = []
    for c in _old():
        if (c["arm"], c["K"]) == cell_key: c = dict(c); c.update(kw)
        out.append(c)
    return out


def test_the_thirteen_lines_read_as_the_pattern_the_page_expects_on_the_second_runs_own_cells_and_the_text_is_the_pages():
    r = B.verdict_bounded(_good(), _old()); assert len(r) == 13 == len(B.BL_LINES) and [x[0] for x in r] == list(range(1, 14)) and [x[3] for x in r] == [True] * 11 + [False] * 2, [x[3] for x in r]
    tags = [t for t, _ in B.BL_LINES]; assert tags[:2] == ["instrument", "instrument"] and tags[2:5] == ["guard"] * 3 and set(tags[5:]) == {"claim"} and B.BL_HEADLINE == (9, 11)
    r2 = B.verdict_bounded([copy.deepcopy(c) for c in _old() if (c["arm"], c["K"]) in B.cells_planned()], _old())                                            # the second run's cells as they are: the loop is still there
    assert [r2[i][3] for i in (1, 2, 3, 4)] == [False, True, True, False]                                                                                 # lines 2 and 5 read NO (the test would have caught the loop); the cells are the same as themselves, lines 3 and 4 yes


def test_each_line_flips_at_its_edge_and_a_missing_cell_reads_not_run_never_no():
    def mut(fn, new=None, old=None):
        new = _good() if new is None else new; fn(new); return _read(new, old)
    # line 1
    def l1(new): logs(_cell_of(new, "KC", 8))[3]["n_cached"] = None
    assert mut(l1)[0] is False
    # line 2: 4 refused replies in 960 steps is within 0.5 percent, 5 is not; a reply that used all 96 tokens fails whatever
    def l2a(new):
        for x in logs(_cell_of(new, "S", 32))[:4]: x["parsed"] = False
    def l2b(new):
        for x in logs(_cell_of(new, "S", 32))[:5]: x["parsed"] = False
    def l2c(new): logs(_cell_of(new, "KC", 8))[0]["n_new"] = 96
    assert mut(l2a)[1] is True and mut(l2b)[1] is False and mut(l2c)[1] is False
    # line 3
    def l3a(new): _cell_of(new, "S", 64)["sessions"][5]["log"][2]["text"] = "DEL[1,1]"
    def l3b(new): _cell_of(new, "S", 64)["sessions"][19]["log"][12]["text"] = "DEL[1,1]"                                                               # job 19, step 13: allowed
    def l3c(new): _cell_of(new, "S", 64)["sessions"][19]["log"][10]["text"] = "DEL[1,1]"                                                               # job 19, step 11: not allowed
    def l3d(new): _cell_of(new, "S", 32)["sessions"][7]["log"].pop()
    def l3e(new): _cell_of(new, "S", 8)["sessions"][0]["log"][0]["text"] += " "
    assert mut(l3a)[2] is False and mut(l3b)[2] is True and mut(l3c)[2] is False and mut(l3d)[2] is False and mut(l3e)[2] is False
    def l3f(new): _cell_of(new, "S", 64)["sessions"][5]["log"][12]["text"] = "DEL[1,1]"                                                               # another job, step 13: not allowed (only job 19 may differ from step 12 on)
    def l3g(new): _cell_of(new, "S", 64)["sessions"][18]["log"][11]["text"] = "DEL[1,1]"                                                              # job 18, step 12: not allowed
    assert mut(l3f)[2] is False and mut(l3g)[2] is False
    # line 4
    def l4a(new): logs(_cell_of(new, "KC", 8))[100]["text"] += "\n"
    def l4b(new): logs(_cell_of(new, "KC", 8))[100]["exact"] = False
    assert mut(l4a)[3] is False and mut(l4b)[3] is False
    # line 5: three ways to fail, and not run without a cell that holds job 19, step 12
    def l5a(new): _cell_of(new, "S", 64)["sessions"][19]["log"][11]["n_new"] = 96
    def l5b(new): _cell_of(new, "S", 64)["sessions"][19]["log"][11]["parsed"] = False
    def l5c(new): _cell_of(new, "S", 32)["sessions"][19]["log"][11]["admitted"] = True
    assert mut(l5a)[4] is False and mut(l5b)[4] is False and mut(l5c)[4] is False
    def l5d(new): _cell_of(new, "S", 32)["sessions"][19]["log"][11]["n_new"] = 95
    assert mut(l5d)[4] is True
    def cut(n_sessions, n_steps):                                                                                                                     # a cell without job 19, or with a job 19 that stopped before step 12, holds no such reply
        def fn(new):
            for K in (32, 64):
                c = _cell_of(new, "S", K); c["sessions"] = c["sessions"][:n_sessions]
                if len(c["sessions"]) > 19: c["sessions"][19]["log"] = c["sessions"][19]["log"][:n_steps]
        return fn
    assert mut(cut(19, 30))[4] is None and mut(cut(20, 11))[4] is None and mut(cut(20, 12))[4] is True and mut(cut(20, 30))[4] is True
    # line 6: half a point of exact share
    def l6a(new): _not_exact(_cell_of(new, "S", 8), 1)
    def l6b(new): _not_exact(_cell_of(new, "S", 8), 2)
    def l6c(new): _not_exact(_cell_of(new, "S", 32), 4)
    def l6d(new): _not_exact(_cell_of(new, "S", 32), 5)
    assert mut(l6a)[5] is True and mut(l6b)[5] is False and mut(l6c)[5] is True and mut(l6d)[5] is False
    # lines 7 and 8: 30 percent of the second run's seconds per step, either way
    def scale(arm, K, f):
        def fn(new): _cell_of(new, arm, K)["wall_seconds"] *= f
        return fn
    assert mut(scale("S", 64, 1.30))[6] is True and mut(scale("S", 64, 1.31))[6] is False and mut(scale("S", 64, 0.70))[6] is True and mut(scale("S", 64, 0.69))[6] is False
    assert mut(scale("S", 8, 1.31))[6] is False and mut(scale("S", 32, 0.69))[6] is False
    assert mut(scale("KC", 8, 1.30))[7] is True and mut(scale("KC", 8, 1.31))[7] is False and mut(scale("KC", 8, 0.70))[7] is True and mut(scale("KC", 8, 0.69))[7] is False
    # lines 9 to 11: exact steps per GPU hour against the kept chat
    S = C.summarize_all(_old())
    def kc8_ratio(r):
        def fn(new): _cell_of(new, "KC", 8)["wall_seconds"] = 3600 * 240 / (S[("S", 8)]["exact_per_hour"] / r)
        return fn
    assert mut(kc8_ratio(2.001))[8] is True and mut(kc8_ratio(1.999))[8] is False
    for line, K, lim in ((10, 32, 10.0), (11, 64, 10.0)):
        for r, want in ((lim + 0.001, True), (lim - 0.001, False)):
            old = _old_with(("KC", K), wall_seconds=3600 * S[("KC", K)]["exact_steps"] / (S[("S", K)]["exact_per_hour"] / r))
            assert _read(_good(), old)[line - 1] is want, (line, r)
    # a kept chat with no exact step at all: exact-state wins by any ratio; both with none: not run
    old = _old_with(("KC", 64), sessions=[dict(s, log=[dict(x, exact=False) for x in s["log"]]) for s in oldcells()[("KC", 64)]["sessions"]]); assert _read(_good(), old)[10] is True
    new = _good(); [x.__setitem__("exact", False) for x in logs(_cell_of(new, "S", 64))]; assert _read(new, old)[10] is None
    # lines 12 and 13 need S to be exact: make every S step exact, then read the edges
    def all_exact(new):
        for c in new:
            if c["arm"] == "S": _all_exact(c)
    got = mut(all_exact); assert got[11] is True and got[12] is True and got[5] is False
    def l12a(new): all_exact(new); _not_exact(_cell_of(new, "S", 8), 12)                                                                                 # 228 of 240 = 0.95 against 1.00: exactly 5 points
    def l12b(new): all_exact(new); _not_exact(_cell_of(new, "S", 8), 13)
    assert mut(l12a)[11] is True and mut(l12b)[11] is False
    def l13a(new): all_exact(new); _not_exact(_cell_of(new, "S", 8), 4)                                                                                  # 236 of 240 = 98.3 %
    def l13b(new): all_exact(new); _not_exact(_cell_of(new, "S", 8), 5)                                                                                  # 235 of 240 = 97.9 %
    assert mut(l13a)[12] is True and mut(l13b)[12] is False
    # not run: a missing cell moves only the lines that need it
    def without(*keys): return [c for c in _good() if (c["arm"], c["K"]) not in keys]
    base = _read(_good())
    for keys, lines in ((((("KC", 8),)), (4, 8, 9)), ((("S", 8),), (9,)), ((("S", 32),), (10,)), ((("S", 64),), (11,)), ((("S", 32), ("S", 64)), (5, 10, 11)),
                        ((("S", 8), ("S", 32), ("S", 64)), (3, 5, 6, 7, 9, 10, 11, 12, 13))):
        got = _read(without(*keys))
        for i in range(13):
            if i + 1 in lines: assert got[i] is None, (keys, i + 1, got)
            elif i + 1 not in (12, 13): assert got[i] is base[i], (keys, i + 1, got)
    assert all(x is None for x in _read([])) and all(x is None for x in _read(_good()[:0], _old()))
    emptied = _good(); [s.__setitem__("log", []) for s in _cell_of(emptied, "S", 8)["sessions"]]; assert _read(emptied)[8] is None                          # a cell with no step counts as not run


def test_the_rule_about_what_may_be_claimed_follows_lines_2_3_7_8_and_12_and_the_printout_says_it():
    r = B.verdict_bounded(_good(), _old()); a = B.claims_allowed(r)
    assert a == dict(constraint_held=True, replies_as_in_second_run=True, base=True, quality_adjusted={9: True, 10: True, 11: True}, point_ratios=True, plain_time_at_equal_exactness=False)
    new = _good(); [x.__setitem__("text", x["text"] + " ") for x in logs(_cell_of(new, "S", 64))[:1]]; r = B.verdict_bounded(new, _old()); a = B.claims_allowed(r)                    # line 3 NO
    assert r[2][3] is False and a["base"] is False and a["quality_adjusted"] == {9: None, 10: None, 11: None} and a["plain_time_at_equal_exactness"] is None and a["point_ratios"] is False
    new = _good(); _cell_of(new, "S", 64)["sessions"][19]["log"][11]["n_new"] = 96; _cell_of(new, "S", 64)["sessions"][19]["log"][11]["parsed"] = False; r = B.verdict_bounded(new, _old()); a = B.claims_allowed(r)         # line 2 NO
    assert r[1][3] is False and a["base"] is False and a["constraint_held"] is False
    new = _good(); _cell_of(new, "KC", 8)["wall_seconds"] *= 1.5; r = B.verdict_bounded(new, _old()); a = B.claims_allowed(r)                                                # line 8 NO alone: the point ratios wait as well
    assert r[6][3] is True and r[7][3] is False and a["base"] is True and a["point_ratios"] is False
    new = _good(); _cell_of(new, "S", 64)["wall_seconds"] *= 1.5; r = B.verdict_bounded(new, _old()); a = B.claims_allowed(r)                                                 # line 7 NO: the point ratios wait
    assert r[6][3] is False and a["base"] is True and a["quality_adjusted"][11] is True and a["point_ratios"] is False
    new = _good()
    for c in new:
        if c["arm"] == "S": _all_exact(c)
    r = B.verdict_bounded(new, _old()); a = B.claims_allowed(r); assert a["plain_time_at_equal_exactness"] is True
    for tweak, expect in ((lambda n: None, ["NOT claimed as equal-exactness ratios", "lines 7 and 8 read yes, so the point ratios", "headline lines 9 and 11: yes, yes", "13 of 13 lines could be read"]),
                          (lambda n: _cell_of(n, "S", 64).__setitem__("wall_seconds", _cell_of(n, "S", 64)["wall_seconds"] * 1.5), ["line 7 or 8 did not read yes", "only as 'at least' their limit"]),
                          (lambda n: logs(_cell_of(n, "S", 64)).__getitem__(0).__setitem__("text", "x"), ["NO claim is made from this run until the cause is found"])):
        new = _good(); tweak(new); buf = io.StringIO()
        with contextlib.redirect_stdout(buf): B.print_verdict(B.verdict_bounded(new, _old()))
        t = buf.getvalue()
        for e in expect: assert e in t, (e, t)
        assert "never: a multiple of capacity, dollars, a statement about any provider" in t or "NO claim is made" in t
    new = _good()
    for c in new:
        if c["arm"] == "S": _all_exact(c)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): B.print_verdict(B.verdict_bounded(new, _old()))
    assert "line 12 reads yes, so the plain time ratios may be claimed at equal exactness." in buf.getvalue() and "NOT claimed as equal-exactness" not in buf.getvalue()


def test_the_command_line_merges_cell_files_prints_the_lines_and_shows_its_help():
    with tempfile.TemporaryDirectory() as d:
        paths = [B.save_cell(B.run_cell(B.BoundedStandIn(), a, k, steps=3), os.path.join(d, f"c_{a}{k}.json")) for a, k in B.cells_planned()]; out = os.path.join(d, "m.json")
        env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "python"), PYTHONDONTWRITEBYTECODE="1")
        r = subprocess.run([sys.executable, "-m", "axi.experiments.provider_side_bounded", "--merge", *paths, "--out", out], capture_output=True, text=True, env=env, cwd=d, timeout=300)
        assert r.returncode == 0 and "written to " + out in r.stdout and "-- provider-side test, name capped at 16" in r.stdout and "[claim]" in r.stdout, r.stderr
        m = json.load(open(out)); assert [(c["arm"], c["K"]) for c in m["cells"]] == B.cells_planned() and m["second_run_sha256"] == SECOND_SHA256
        r = subprocess.run([sys.executable, "-m", "axi.experiments.provider_side_bounded"], capture_output=True, text=True, env=env, cwd=d, timeout=300)
        assert r.returncode == 0 and "--dry-run" in r.stdout and "--merge" in r.stdout and "usage" in r.stdout.lower()


def test_the_dry_run_prints_the_stand_in_table_and_says_it_is_not_a_result():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): S = B.dry_run(steps=6, levels=(8,))
    t = buf.getvalue(); assert "NOT a result" in t and ("S", 8) in S and ("KC", 8) in S and ("S", 32) not in S and all(s["engine_kind"] == "stand_in" for s in S.values()) and "exact/h" in t


# ---------------------------------------------------------------- 8 the locked page
PAGE = os.path.join(ROOT, "docs", "PROVIDER_SIDE_BOUNDED_TEST.md")
RESULT_MARK = "\n\n# Result"                                                                                          # the Result starts below the line that ends the locked text
PAGE_SHA256 = "888358daf88abd0e11be3f368b67ab30c375fc116ce7445018158f77dbec2083"
PROBS = [98, 90, 85, 85, 88, 82, 72, 75, 92, 95, 95, 4, 3]


def _h(text): return hashlib.sha256((text.rstrip("\n") + "\n").encode()).hexdigest()


def _head():
    text = open(PAGE).read(); assert text.count(RESULT_MARK) <= 1
    return text[:text.index(RESULT_MARK)] if RESULT_MARK in text else text


def test_the_page_is_locked_and_carries_the_thirteen_lines_the_expression_the_settings_and_the_thresholds_the_code_uses():
    head = _head(); assert _h(head) == PAGE_SHA256, "the predictions and reading rules must not change after the run; later notes go below them"
    for i, (tag, line) in enumerate(B.BL_LINES, 1): assert f"\n{i}. [{tag}] {line} (" in head, i
    probs = [int(m.group(1)) for m in re.finditer(r"\n\d+\. \[\w+\] .*\((\d+) %\)(?=\n)", head)]; assert probs == PROBS, probs
    assert "```\n" + B.REGEX + "\n```" in head and f"`{B.REGEX_SHA256}`" in head and f"`{C.REGEX_SHA256}`" in head and "`[A-Za-z0-9_]{1,16}`" in head
    for needle in ("**Headline** (lines 9 and 11)", "Lines 2 and 3 both reading yes: about **80 %**", "Lines 9, 10 and 11 all reading yes: about **85 %**", "The pattern I expect (lines 12 and 13 NO, every other line yes): about **30 %**", "All 13 yes: under 1 %"):
        assert needle in head, needle
    th = (f"Thresholds as in the code: line 2: {B.B_PARSED * 100:g} percent parsed, no reply with 96 new tokens; line 5: fewer than 96 new tokens; line 6: {B.B_SHARE_TOL}; lines 7 and 8: {B.B_TIME_TOL:.2f}; "
          f"line 9: {B.B_QA_LIGHT:g} times; lines 10 and 11: {B.B_QA_MID:g} times each; line 12: {B.B_EXACT_GAP}; line 13: {B.B_EXACT_HIGH}.")
    assert th in head, th
    assert B.B_QA_MID == B.B_QA_HEAVY == 10.0 and B.LOOP_JOB == 19 and B.LOOP_STEP == 12 and B.NAME_CAP == 16 and B.BL_HEADLINE == (9, 11)
    for needle in ("**The rule that decides what the Result may claim.**", "Line 2 or 3 reads NO: no claim is made from this run until the cause is found.", "The point ratios of lines 10 and 11 (the \"x times\") are quoted as numbers only if lines 7 and 8 also read yes",
                   "The plain time ratios are claimed as ratios at equal exactness only if line 12 reads yes.", "*N exact steps per GPU hour against M, at these exactness levels, on this setup*",
                   "Never a multiple of capacity (\"N times the capacity\", \"N GPUs for one\", \"N times as many users\"), never a count of GPUs saved, never dollars or a percentage of cost, never \"proven\"",
                   "**The kept chat at 32 and 64 jobs is not run again**", "The cap clears line 2, and the figures that line was holding back; it does not clear anything else.", "**energy (never measured; only time)**",
                   "If the probe's reply does not parse, or uses all 96 new tokens, or any reply of the safety cell does not parse, the notebook prints STOP and the cells are not run", "the cells are not merged"):
        assert needle in head, needle
    assert head.count("\n# ") == 0 and head.startswith("# Test 16: the serving side a third time") and "Concept-map item 108" in head and "docs/results/axi_provider_constrained.json" in head
    for sha in (PS_SHA256, C_SHA256, SECOND_SHA256): assert sha in head
    assert "`provider_side.py` `" + PS_SHA256 + "`" in head and "`provider_side_constrained.py` `" + C_SHA256 + "`" in head


def test_the_figures_the_page_quotes_from_the_second_and_first_runs_are_the_ones_in_their_saved_files():
    head = _head(); S = C.summarize_all(second()["cells"])
    for s in ("Of the 6,231 replies of the second run, 2 are outside the new language", "The longest name in any reply that parsed is 5 characters (`value`; the oracle's own replies use `dep` and `owns`), and the longest reply that parsed is 31 tokens.",
              "there are 3,513 steps in the five cells that both runs have. In 3,463 the text is the same. In the other 50 it is different, and in every one of the 50 the first run's reply was outside the language",
              "in none was a reply that was inside the language changed", "(480 steps compared in each arm, 0 different)", "took 6.13 seconds at 32 jobs (the rounds before and after it: 3.79 and 3.37) and 9.13 seconds at 64 jobs (6.77 and 6.32)",
              "is 2.2 and 1.3 percent of the cell's 113.7 and 204.3 seconds", "(1 h 19 min and 2 h 52 min of GPU in the second run)", "the 3,360 steps of the four cells", "88.3, 94.6 and 97.2 percent",
              "| `S` 8 | 212 of 240 | 88.3 % | 36.6 | 20,878 | 36.6 (job 19 is not in it) | 20,878 | 3.5 (`KC` 5,923) |", "| `S` 32 | 908 of 960 | 94.6 % | 113.7 | 28,748 | 111.2 | 29,407 | 40.4 (`KC` 727) |",
              "| `S` 64 | 1,867 of 1,920 | 97.2 % | 204.3 | 32,896 | 201.7 | 33,318 | 49.9 (`KC` 667) |", "| `KC` 8 | 240 of 240 | 100.0 % | 145.9 | 5,923 | re-run | | |", "| `KC` 32 | 956 of 956 | 100.0 % | 4,730.9 | 727 | not run again | | |",
              "| `KC` 64 | 1,915 of 1,915 | 100.0 % | 10,330.5 | 667 | not run again | | |", "from 39.5 and 49.3 to at most 40.3 and 49.8", "checked in the test for 5 names, among them the first 16 characters of the looping one"):
        assert s in head, s
    assert "the oracle adds the node `(3,0)` (`ADD[3,0:4]`)" in head and "(the second run's job 19 was exact up to step 11)" in head
    for s in ("5,923", "727", "667"): assert s.replace(",", "") in {str(round(S[k]["exact_per_hour"])) for k in S}
    assert "6,231" in head and "480" in head


# ---------------------------------------------------------------- 9 the notebook
NOTEBOOK = os.path.join(ROOT, "python", "colab", "axi_provider_side_bounded.ipynb")
def _nb(): return json.load(open(NOTEBOOK))
def _srcs(): return ["".join(c["source"]) for c in _nb()["cells"]], [c["cell_type"] for c in _nb()["cells"]]


def test_the_notebook_installs_the_pinned_engine_fixes_the_stream_loads_the_capped_engine_checks_runs_the_cells_in_order_and_puts_them_together():
    srcs, kinds = _srcs(); code = [s for s, k in zip(srcs, kinds) if k == "code"]; text = "\n".join(srcs); nb = _nb()
    import ast
    for s in code: ast.parse("\n".join(l for l in s.splitlines() if not l.lstrip().startswith("!")))
    for needle in ("git clone -q --depth 1 https://github.com/gaurshalendra2412/axi-runtime", "provider_side_bounded.py", "axi_provider_constrained.json", "B.BoundedVLLMEngine()", "eng.warm_up()", "B.loop_probe(eng)", "STOP:", "cached tokens", "B.run_cell(eng, arm, 3, steps=3)",
                   "B.save_cell", "store.persist", "B.merge_files", "B.verdict_bounded", "B.print_cells", "B.print_verdict", "B.load_second_run", "files.download", "open_store", "docs/PROVIDER_SIDE_BOUNDED_TEST.md", "Do not paste any keys",
                   "checkpoint=store.path(ck)", "update 60 commands", "13 reading-rule lines", "name at most", "axi_provider_b_", "re.fullmatch(B.REGEX, probe[\"reply\"])"): assert needle in text, needle
    assert text.count("Do not paste any keys") >= 2 and "!pip -q install vllm==0.31.0" in [x.strip() for x in code] and "!pip -q install vllm" not in [x.strip() for x in code] and "open_store(\"/content/out\", \"axi_provider_bounded\")" in text
    fix = [i for i, (s, k) in enumerate(zip(srcs, kinds)) if k == "code" and "fileno" in s]; load = [i for i, (s, k) in enumerate(zip(srcs, kinds)) if k == "code" and "B.BoundedVLLMEngine()" in s]; assert len(fix) == 1 and len(load) == 1 and fix[0] < load[0]
    ran = []; ns = dict(go=lambda arm, K: ran.append((arm, K)))
    for s in code:
        if re.fullmatch(r"(for arm, K in \(.*\): go\(arm, K\)|go\(\"K[CS]\", \d+\))\s*", s.strip()): exec(s, ns)
    assert ran == B.cells_planned(), ran
    assert nb["metadata"]["accelerator"] == "GPU" and nb["nbformat"] == 4 and kinds[0] == "markdown"
    head = srcs[0]; assert "T4" in head and "80,000 tokens" in head and "Qwen2.5-7B-Instruct" in head and "AWQ" in head and "vLLM 0.31.0" in head and "25 to 30 minutes" in head and "continues from its last finished round" in head and "Keep this browser tab open" in head
    assert "16 characters" in head and "is **not** run again" in head and "not run again" not in head.replace("is **not** run again", "")
    assert [c["cell_type"] for c in nb["cells"]].count("code") == 9


def test_the_notebook_cell_runner_skips_finished_cells_and_carries_on_a_cell_the_last_session_left_unfinished():
    srcs, kinds = _srcs(); runner = next(s for s, k in zip(srcs, kinds) if k == "code" and "def go(arm, K):" in s); body = runner[runner.index("def go(arm, K):"):]
    class Crash(Exception): pass
    class Dying(B.BoundedStandIn):
        def __init__(self, die_at): super().__init__(); self.n = 0; self.die_at = die_at
        def generate_round(self, requests, reset_cache=False):
            self.n += 1
            if self.n == self.die_at: raise Crash()
            return super().generate_round(requests, reset_cache)
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as remote:
        sys.path.insert(0, os.path.join(HERE, ".."))
        from colab.colab_persist import Store
        shim = types.SimpleNamespace(run_cell=lambda e, arm, K, progress=None, checkpoint=None: B.run_cell(e, arm, K, steps=5, progress=progress, checkpoint=checkpoint), save_cell=B.save_cell, summarize_cell2=B.summarize_cell2)
        def session(engine):
            store = Store(d, remote); ns = dict(B=shim, eng=engine, store=store, os=os, time=__import__("time"), json=json); exec(body, ns); return ns["go"]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            go = session(Dying(die_at=3))
            try: go("KC", 3)
            except Crash: pass
            assert sorted(os.listdir(remote)) == ["axi_provider_b_KC_K3.ckpt.json"] and len(json.load(open(os.path.join(remote, "axi_provider_b_KC_K3.ckpt.json")))["rounds"]) == 2
            for f in os.listdir(d): os.remove(os.path.join(d, f))
            go = session(B.BoundedStandIn()); go("KC", 3); go("S", 3); go("KC", 3)
        out = buf.getvalue()
        assert "carrying on from round 2" in out and out.count("already finished, skipping") == 1 and out.count("done:") == 2 and "resumed at rounds [2]" in out
        done = re.findall(r"done: (\d+) steps, (\d+) s, (\d\.\d\d) s/step, computed prompt tokens (\d+), cached share (\d+) %, exact steps (\d+\.\d) %, refused replies (\d+), exact steps per hour (\d+)", out)
        assert len(done) == 2 and all(x[0] == "15" and x[5] == "100.0" and x[6] == "0" for x in done), done                                              # 3 jobs of 5 steps, all exact, none refused
        s = B.summarize_cell2(B.run_cell(B.BoundedStandIn(), "S", 3, steps=5))                                                                              # the S cell was not interrupted, so a fresh run gives the very same line
        want = (f"done: {s['steps_done']} steps, {s['wall_seconds']:.0f} s, {s['seconds_per_step']:.2f} s/step, computed prompt tokens {s['computed_prefill']:.0f}, cached share {s['cached_share'] * 100:.0f} %, "
                f"exact steps {s['exact_share'] * 100:.1f} %, refused replies {s['unparsed']}, exact steps per hour {s['exact_per_hour']:.0f}")
        assert want in out, (want, out)
        assert sorted(f for f in os.listdir(remote) if not f.endswith(".ckpt.json")) == ["axi_provider_b_KC_K3.json", "axi_provider_b_S_K3.json"]


def test_the_notebooks_last_cell_puts_the_cells_together_reads_the_second_runs_file_prints_the_lines_and_offers_the_download():
    srcs, kinds = _srcs(); cell = next(s for s, k in zip(srcs, kinds) if k == "code" and "B.merge_files" in s)
    assert cell.count("/content/axi/docs/results/axi_provider_constrained.json") == 1
    cell = cell.replace("/content/axi/docs/results/axi_provider_constrained.json", SECOND_FILE)
    sys.path.insert(0, os.path.join(HERE, ".."))
    from colab.colab_persist import Store
    got = []; gc = types.ModuleType("google.colab"); gc.files = types.SimpleNamespace(download=lambda p: got.append(p)); g = types.ModuleType("google"); g.colab = gc
    saved = {k: sys.modules.get(k) for k in ("google", "google.colab")}
    sys.modules["google"], sys.modules["google.colab"] = g, gc
    try:
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as remote:
            store = Store(d, remote)
            for arm, k in B.cells_planned(): B.save_cell(B.run_cell(B.BoundedStandIn(), arm, k, steps=4), store.path(f"axi_provider_b_{arm}_K{k}.json"))
            open(store.path("axi_provider_b_S_K8.ckpt.json"), "w").write("{}")                                                                       # a checkpoint file is not a cell
            buf = io.StringIO(); ns = dict(B=B, store=store, glob=glob, os=os)
            with contextlib.redirect_stdout(buf): exec(cell, ns)
            out = buf.getvalue(); assert "4 cells:" in out and "ckpt" not in out and "-- provider-side test, name capped at 16" in out and out.count("[claim]") == 8
            assert ("never: a multiple of capacity" in out) != ("NO claim is made from this run" in out)                                          # the printout says exactly one of the two things the page rule allows
            assert got == [store.path("axi_provider_bounded.json")] and os.path.exists(os.path.join(remote, "axi_provider_bounded.json"))
            m = json.load(open(store.path("axi_provider_bounded.json"))); assert [(c["arm"], c["K"]) for c in m["cells"]] == B.cells_planned() and m["second_run_sha256"] == SECOND_SHA256
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
    srcs, kinds = _srcs(); cell = next(s for s, k in zip(srcs, kinds) if k == "code" and "B.BoundedVLLMEngine()" in s)
    mods, calls = T124._fake_vllm("module")
    with T124._installed(mods), tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as remote:
        sys.path.insert(0, os.path.join(HERE, ".."))
        from colab.colab_persist import Store
        ns = dict(json=json, os=os, time=time, store=Store(d, remote)); buf = io.StringIO()
        with contextlib.redirect_stdout(buf): exec(cell, ns)
        out = buf.getvalue()
        assert f"constraint: regex name at most 16 characters {B.REGEX_SHA256[:16]} (the same for both arms)" in out and "cache size the engine will use: 80000 tokens ( override )" in out
        assert callable(ns["go"]) and isinstance(ns["eng"], B.BoundedVLLMEngine) and ns["B"] is B and calls["resets"] >= 1 and '"constraint"' not in out and '"kind": "vllm"' in out


def test_the_safety_cell_says_ok_on_a_working_engine_and_stop_for_each_thing_that_can_be_wrong():
    srcs, kinds = _srcs(); cell = next(s for s, k in zip(srcs, kinds) if k == "code" and "smoke = " in s)
    def run(engine):
        buf = io.StringIO(); ns = dict(B=B, eng=engine, re=re)
        with contextlib.redirect_stdout(buf): exec(cell, ns)
        return buf.getvalue()
    out = run(B.BoundedStandIn()); last = out.strip().splitlines()[-1]
    assert last.startswith("OK: cached counts are reported") and "STOP" not in out and "(job 19, step 12, add_node)" in out and "the strict parser reads it" in out and "inside the capped language: True" in out
    class NoCount(B.BoundedStandIn):
        def generate_round(self, requests, reset_cache=False): reps, dt = super().generate_round(requests, reset_cache); [setattr(r, "n_cached", None) for r in reps]; return reps, dt
    class NoCache(B.BoundedStandIn):
        def generate_round(self, requests, reset_cache=False): reps, dt = super().generate_round(requests, reset_cache); [setattr(r, "n_cached", 0) for r in reps]; return reps, dt
    def probe_says(text, n_new):
        class P(B.BoundedStandIn):
            def generate_round(self, requests, reset_cache=False):
                reps, dt = super().generate_round(requests, reset_cache)
                if len(requests) == 1: reps[0].text = text; reps[0].n_new = n_new
                return reps, dt
        return P()
    assert "STOP: this vLLM version does not report cached tokens" in run(NoCount())
    assert "STOP: the cache did nothing for the kept chat's second step" in run(NoCache())
    loop = LOOP_TEXT_START + "t"
    o = run(probe_says(loop, 96)); assert "STOP: the probe's reply is refused by the strict parser" in o and "REFUSES it" in o
    o = run(probe_says("ADD[(3,0)->(3,0):dep#1]", 96)); assert "STOP: the probe's reply used all 96 new tokens, so the name is not capped" in o
    o = run(probe_says("ADD[(3,0)->(3,0):" + "a" * 17 + "#1]", 30)); assert "STOP: the probe's reply is not inside the capped language" in o and "inside the capped language: False" in o
    o = run(probe_says("ADD[(3,0)->(3,0):4tworowsdown4col#1]", 31)); assert o.strip().splitlines()[-1].startswith("OK:") and "4tworowsdown4col#1]" in o
    class Garbage(B.BoundedStandIn):
        def generate_round(self, requests, reset_cache=False): reps, dt = super().generate_round(requests, reset_cache); reps[0].text = "Sure, here is the delta: " + reps[0].text; return reps, dt
    assert "STOP: a reply of the safety cell was refused by the strict parser" in run(Garbage())


# ---------------------------------------------------------------- 10 the front pages
def test_the_front_pages_point_to_the_tests_and_say_what_the_page_expects_and_what_the_run_found():
    cm = open(os.path.join(ROOT, "docs", "RAJNISH_CONCEPT_MAP.md")).read(); ve = open(os.path.join(ROOT, "docs", "VERIFIED.md")).read()
    assert "PROVIDER_SIDE_BOUNDED_TEST.md" in cm and "PROVIDER_SIDE_BOUNDED_TEST.md" in ve
    item = cm[cm.index("108. **"): cm.index("109. **")]; k = ve.index("PROVIDER_SIDE_BOUNDED_TEST.md"); row = ve[k - 300: ve.index("Provider-side serving test, fourth run")]
    for text in (item, row):
        assert PAGE_SHA256[:16] in text.replace(PAGE_SHA256, PAGE_SHA256[:16]) and "test 126" in text.replace("**test 126**", "test 126") and "16 characters" in text and "13 reading lines" in text
        assert ("RUN (19 Oct 2026)" in text or "Run (19 Oct 2026" in text) and "not run yet" not in text.lower() and "@@" not in text
        assert "exact steps per GPU hour" in text and "capacity" in text and "line 2" in text
        for n in ("20,878", "29,400", "33,300", "5,923", "727", "667", "88.3, 94.6", "97.2"): assert n in text, n
    assert "Next free: row 172, item 132, test 158, question 121" in cm and "test 126 checks the capped re-run of the exact-state cells and test 127 its Result, item 108" in cm
    assert "undeniable proof" in item and "49 GPUs" in item and "not accepted" in item and PAGE_SHA256 in item and PAGE_SHA256[:16] + "…" in row
    for n in ("(19 Oct)", "to clear line 2 of item 107", "\"4,900%\"", "(job 3 at step 16, job 4 at step 18, job 19 at step 12 and so on)", "a 7B 4-bit model, an 80,000-token cache and 30-step jobs", "(97.2 % exact against 100.0 %)",
              "**This test could clear line 2 and nothing else. Status: RUN (19 Oct 2026); the Result is on the page.**", "= **test 126** (23 tests:", "6,231 replies, of which 2 (one reply, in two cells) are outside the new language", "same 30 prompts",
              "3,513 steps", "(about ten minutes of GPU)", "(1 h 19 min and 2 h 52 min)", "(0 of 3,513 comparable steps changed where the first run's reply was inside the language)", "(lines 7 and 8 check the speed)", "13 reading lines",
              "exact steps per GPU hour at 8 and 64 jobs", "and never a multiple of capacity, a count of GPUs saved, dollars or \"proven\"", "(78 %)", "88.3, 94.6 and 97.2 percent", "(4 and 3 %)", "20,878, 29,400 and 33,300 against 5,923, 727 and 667",
              "lines 2 and 3 both yes about 80 %", "cells `S` at 8, 32 and 64 jobs and `KC` at 8 jobs", "a name of 1 to 16 characters", "lines 12 and 13 (equal exactness within 5 points, 98 percent)",
              "\"49 GPUs with 1 GPU\"", "49.3 is a ratio of exact steps per GPU hour", "constrained decoding was already on in item 107", "\"49 GPUs\", dollars and \"proven\" were never measured", "the kept chat at 32 and 64 jobs under the capped expression",
              "after the Result of item 107", "about 3.5, 40 and 50 times", "(hash-locked; sha256 ", "the setup was one free T4", "the kept chat at 32 and 64 jobs (1 h 19 min and 2 h 52 min)", "Mutation check of the test: 478 deliberate changes", "396 caught", "the 82 that survived were read by hand",
              "capped at 16 characters, to clear line 2 of item 107", "as Update 60 ... run those exact-state cells again to clear Line 2", "edits neither: both sha256 are pinned", "the 13 lines at their edges on the second run's own cells", "so job 19 stays wrong from step 12 and the only reply"):
        assert n in item, n
    for n in ("(`PROVIDER_SIDE_BOUNDED_TEST.md`, `experiments/provider_side_bounded.py`, `colab/axi_provider_side_bounded.ipynb`)", "13 reading lines", "a name of 1 to 16 characters", "`S` at 8, 32 and 64 jobs and `KC` at 8 jobs run again",
              "the kept chat at 32 and 64 jobs read from the second run's file", "2 of its 6,231 replies are outside the new language (one reply, job 19 step 12, in two cells)", "of the 3,360 steps of the four cells, only that reply (two copies of it) should differ",
              "3,513 steps, 0 changed where the earlier reply was inside the language", "line 2 yes (90 %)", "(88.3, 94.6, 97.2 percent)", "lines 12 and 13 (equal exactness, 98 percent) NO again", "about 20,878, 29,400 and 33,300 against 5,923, 727 and 667 for the kept chat",
              "the locked page (test 126, 23 tests)", "**Run (19 Oct 2026, Colab, Tesla T4, vLLM 0.31.0, one session, all four cells).**", "never a multiple of capacity, a count of GPUs saved, dollars or \"proven\"",
              "the exact-state cells again with the length of a name capped at 16 characters, to clear line 2 of the second run", "locked before the run (sha256 " + PAGE_SHA256[:16] + "… of the locked text)", "so job 19 stays where it was and, of the", "the kept chat at 32 and 64 jobs under the capped expression", "mutation check of test 126: 478 deliberate changes, 396 caught, the 82 survivors read by hand"):
        assert n in row, n


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("PASS", n)
