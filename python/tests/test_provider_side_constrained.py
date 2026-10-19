"""Engineering support for docs/PROVIDER_SIDE_CONSTRAINED_TEST.md (concept-map item 107): the provider-side serving test run again with every reply constrained by the repository's grammar.
Checks that need no GPU and no model:
  1. the regular expression accepts exactly the delta language of the repository's parser and DFA (random walks, mutated strings, hand-picked strings), and the pinned hash of its text;
  2. the claim on the page about the first run's refused replies, recomputed from its saved cell files (82 of the 89 are the same slip, and for that slip the grammar leaves one way out);
  3. the vLLM adapter against a fake module: the constraint goes into the sampling parameters, a version that cannot constrain fails before the model is loaded, nothing falls back to
     replies that are not constrained, the engine's info records the constraint;
  4. the cells, their order, and the merge (it refuses cells that carry another constraint or none);
  5. the cell runner: the same cell as the first run's runner on the stand-in, and the checkpoint: a cell stopped after any round and resumed gives the same jobs, the same replies, the same
     number of rounds; the checkpoint is refused when it belongs to another cell or does not add up;
  6. the probe that feeds the engine the prompt of a reply the first run refused;
  7. the 17 reading lines at their edges, 'not run' for a missing cell, the rule about which claims the lines allow, and the printout;
  8. the locked page, the notebook, the front pages (below).
What it does NOT show: anything about a real engine, a real GPU, or what a model answers."""
import contextlib, copy, glob, hashlib, io, json, os, random, re, sys, tempfile, types

sys.path.insert(0, os.path.dirname(__file__))
import axi.experiments.provider_side as PS
import axi.experiments.provider_side_constrained as C
from axi.engine.cico_decoder import CICOGrammarDFA
from axi.experiments.agent_loop import strict_parse

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..", "..")
REGEX_SHA256 = "21cdc0bf7824cff3866413dd74fc5939ccc01135e3b241dfcf9fbb1440bb0ec9"


def raises(fn, exc=Exception):
    try: fn()
    except exc: return True
    return False


# ---------------------------------------------------------------- 1 the regular expression is the repository's delta language
def _walk(rnd, dfa, maxlen=60):
    st, out = dfa.START, bytearray()
    for _ in range(maxlen):
        nxt = list(dfa.T[st].items())
        if not nxt: break
        if dfa.is_accepting(st) and out and rnd.random() < 0.15: break
        b, t = rnd.choice(nxt); out.append(b); st = t
    return bytes(out)


def test_the_regular_expression_accepts_exactly_the_language_of_the_repositorys_grammar_and_parser():
    rx = re.compile(C.REGEX); dfa = CICOGrammarDFA(); rnd = random.Random(1); alphabet = b"ADEL[]()0123456789,:->#_ \nxyz"; accepted = 0
    for _ in range(6000):
        w = _walk(rnd, dfa); acc = dfa.accepts(w) and len(w) > 0; assert acc == bool(rx.fullmatch(w.decode())), w; accepted += acc
    assert accepted > 1500                                                         # the walks do reach the language
    mutated = 0
    for _ in range(12000):
        w = bytearray(_walk(rnd, dfa))
        if not w: continue
        for _ in range(rnd.randint(1, 2)): w[rnd.randrange(len(w))] = rnd.choice(alphabet)
        w = bytes(w); m = bool(rx.fullmatch(w.decode())); assert (dfa.accepts(w) and len(w) > 0) == m, w; mutated += 1
        if m: assert strict_parse(w.decode()) is not None, w                      # whatever the regular expression lets through, the strict parser reads
    assert mutated > 10000
    yes = ["ADD[4,4:3]", "ADD[0,0:-3]", "DEL[4,5]", "ADD[(0,0)->(1,1):dep#2]", "DEL[(0,0)->(1,1):owns_2#-7]", "DEL[2,2] DEL[(0,0)->(2,2):dep#1]", "ADD[1,0:5]\nDEL[1,0]", "DEL[4,5] ", "DEL[4,5]\n", "ADD[10,12:345]"]
    no = ["", " ", "DEL[4,5:8]", "ADD[(4,4):6]", "ADD[4,4]", "DEL[4]", "DEL[4,5]DEL[4,6]", "DEL[4,5]  DEL[4,6]", " DEL[4,5]", "DEL[4,5]  ", "del[4,5]", "ADD[4,4:3] and", "ADD[(0,0)->(1,1):dep]", "ADD[(0,0)->(1,1):#2]",
          "ADD[(0,0)->(1,1):de-p#2]", "ADD[4,4:+3]", "```DEL[4,5]```", "DEL[(0,0)]", "ADD[a,4:3]", "DEL[4,5];"]
    for t in yes: assert rx.fullmatch(t) and dfa.accepts(t.encode()) and strict_parse(t) is not None, t
    for t in no: assert not rx.fullmatch(t), t
    assert strict_parse(" DEL[4,5]") is not None and not rx.fullmatch(" DEL[4,5]")  # the regular expression is stricter than the parser at the edges (leading whitespace), never looser
    assert hashlib.sha256(C.REGEX.encode()).hexdigest() == C.REGEX_SHA256 == REGEX_SHA256, C.REGEX_SHA256                       # the text of the expression is pinned


# ---------------------------------------------------------------- 2 what the first run's refused replies were
def _first_run_cells():
    base = os.path.join(ROOT, "docs", "results", "provider_side_cells"); out = {}
    for f in glob.glob(os.path.join(base, "*.json")):
        c = json.load(open(f)); out[(c["arm"], c["K"])] = c
    return out


def test_the_refused_replies_of_the_first_run_are_the_slip_the_page_says_and_the_grammar_leaves_that_slip_one_way_out():
    cells = _first_run_cells(); assert len(cells) == 6
    refused = [x for c in cells.values() for s in c["sessions"] for x in s["log"] if not x["parsed"]]; assert len(refused) == 89
    slip = re.compile(r"DEL\[(\d+),(\d+):-?\d+\]"); fixed = 0; other = []
    dfa = CICOGrammarDFA()
    for x in refused:
        t = x["text"].strip(); m = slip.match(t)
        if m and (len(t) == m.end() or t[m.end()] == " "):                          # DEL[r,c:v] alone, or first in a list
            fixed += 1
            prefix = f"DEL[{m[1]},{m[2]}"; st = dfa.run(prefix.encode()); assert st is not None
            allowed = {chr(b) for b in dfa.T[st]}; assert allowed == set("0123456789]"), allowed        # after DEL[r,c the grammar allows digits and the closing bracket, and ':' is not among them
            rest = t[m.end():]; rewritten = f"{prefix}]" + rest
            if rest == "": assert re.fullmatch(C.REGEX, rewritten) and strict_parse(rewritten) is not None
        else: other.append(t)
    assert (fixed, len(other)) == (84, 5), (fixed, len(other), other)               # 82 that are exactly the slip + 2 lists that start with it (DEL[1,0:6] ADD[1,0:5]); the other 5 are adds written like edges
    assert sorted(set(other)) == ["ADD[(3,0):4]", "ADD[(4,4):6]"] and all(re.fullmatch(r"ADD\[\(\d+,\d+\):\d+\]", t) for t in other)
    plain = [x for x in refused if slip.fullmatch(x["text"].strip())]; assert len(plain) == 82
    for t in other:                                                                 # for an add written like an edge the grammar forces the edge arrow: no rewriting of it is the right node add
        st = dfa.run(t.split(")")[0].encode() + b")"); assert {chr(b) for b in dfa.T[st]} == {"-"}
    s = cells[("S", 8)]["sessions"]; first = next(x for x in s[0]["log"] if not x["exact"])
    assert (first["t"], first["kind"], first["text"]) == (20, "delete_dep", "DEL[4,5:8]")        # the reply the probe's prompt is built to reproduce


# ---------------------------------------------------------------- 3 the vLLM adapter, against a fake module
def _fake_vllm(structured="module", reset_returns=True):
    """A `vllm` with the calls ConstrainedVLLMEngine makes. structured: 'module' (vllm.sampling_params.StructuredOutputsParams), 'top' (vllm.StructuredOutputsParams) or None (not there)."""
    mod = types.ModuleType("vllm"); mod.__version__ = "9.9.9-fake"; calls = dict(init=[], generate=[], resets=0)
    class SamplingParams:
        def __init__(self, **kw): self.kw = kw
    class StructuredOutputsParams:
        def __init__(self, **kw): self.kw = kw
    class Tok:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False): return "|".join(m["content"] for m in messages) + "|A:"
    class Out:
        def __init__(self, prompt):
            ids = list(range(len(prompt.split()))); self.prompt_token_ids = ids; self.num_cached_tokens = len(ids) // 2
            self.outputs = [types.SimpleNamespace(text="DEL[1,2]", token_ids=[1, 2, 3])]
    class LLM:
        def __init__(self, **kw):
            calls["init"].append(kw); self.llm_engine = types.SimpleNamespace(cache_config=types.SimpleNamespace(block_size=16, num_gpu_blocks=3000))
        def get_tokenizer(self): return Tok()
        def generate(self, prompts, sp, use_tqdm=True): calls["generate"].append((list(prompts), sp, use_tqdm)); return [Out(p) for p in prompts]
        def reset_prefix_cache(self): calls["resets"] += 1; return reset_returns
    mod.LLM, mod.SamplingParams = LLM, SamplingParams
    mods = {"vllm": mod}
    if structured == "module": sp = types.ModuleType("vllm.sampling_params"); sp.StructuredOutputsParams = StructuredOutputsParams; mod.sampling_params = sp; mods["vllm.sampling_params"] = sp
    elif structured == "top": mod.StructuredOutputsParams = StructuredOutputsParams; mods["vllm.sampling_params"] = types.ModuleType("vllm.sampling_params")
    return mods, calls


@contextlib.contextmanager
def _installed(mods):
    old = {k: sys.modules.get(k) for k in ("vllm", "vllm.sampling_params")}
    for k in old: sys.modules.pop(k, None)
    sys.modules.update(mods)
    try: yield
    finally:
        for k, v in old.items():
            if v is None: sys.modules.pop(k, None)
            else: sys.modules[k] = v


def test_the_vllm_adapter_puts_the_constraint_into_every_request_and_records_it():
    for how in ("module", "top"):
        mods, calls = _fake_vllm(how)
        with _installed(mods):
            e = C.ConstrainedVLLMEngine(); kw = calls["init"][0]
            assert kw == dict(model=PS.MODEL, dtype="half", max_model_len=8192, gpu_memory_utilization=0.90, enable_prefix_caching=True, seed=0, num_gpu_blocks_override=5000)            # the first run's engine settings
            sp = e.sp.kw; assert set(sp) == {"temperature", "max_tokens", "structured_outputs"} and sp["temperature"] == 0.0 and sp["max_tokens"] == 96 and sp["structured_outputs"].kw == dict(regex=C.REGEX)
            i = e.info(); assert i["kind"] == "vllm" and i["version"] == "9.9.9-fake" and i["capacity_tokens"] == 80000 and i["capacity_source"] == "override"
            assert i["constraint"] == dict(kind="regex", api="structured_outputs(regex)", regex=C.REGEX, regex_sha256=C.REGEX_SHA256)
            e.warm_up(); assert calls["generate"][-1][1] is e.sp and calls["resets"] == 1                                    # the warm-up goes through the constraint, so its compile is not in a timed round
            reps, dt = e.generate_round([PS.Request([{"role": "user", "content": "a b c d"}])]); assert calls["generate"][-1][1] is e.sp and reps[0].text == "DEL[1,2]" and reps[0].n_cached == reps[0].n_prompt // 2
            c = C.run_cell(e, "S", 2, steps=2); assert c["engine"]["constraint"]["regex_sha256"] == C.REGEX_SHA256 and all(g[1] is e.sp for g in calls["generate"])
    mods, calls = _fake_vllm(None)
    with _installed(mods):
        try: C.ConstrainedVLLMEngine()
        except RuntimeError as err: assert "no StructuredOutputsParams" in str(err) and "does not fall back" in str(err)
        else: raise AssertionError("an engine that cannot constrain must refuse to start")
        assert calls["init"] == []                                                                                          # and it refused before the model was loaded
    mods, calls = _fake_vllm("module")
    with _installed(mods):
        plain = PS.VLLMEngine(); assert set(plain.sp.kw) == {"temperature", "max_tokens"}                                  # the first run's engine is unchanged (no constraint)


# ---------------------------------------------------------------- 4 cells and merge
def test_the_planned_cells_run_the_cheap_ones_first_and_the_merge_refuses_what_does_not_belong():
    assert C.cells_planned() == [("S", 8), ("KC", 8), ("S", 32), ("S", 64), ("KC", 32), ("KC", 64)] and set(C.ARMS) == {"S", "KC"} and C.LEVELS == (8, 32, 64) and (C.STEPS, C.SEED, C.POLICY, C.HINT_RULES, C.MAX_NEW, C.MODEL) == (PS.STEPS, PS.SEED, PS.POLICY, PS.HINT_RULES, PS.MAX_NEW, PS.MODEL)
    eng = C.ConstrainedStandIn(); cells = [C.run_cell(eng, a, k, steps=2) for a, k in reversed(C.cells_planned()[:3])]
    m = C.merge_cells(cells); assert [(c["arm"], c["K"]) for c in m["cells"]] == [("S", 8), ("KC", 8), ("S", 32)] and m["meta"]["arms"] == ["S", "KC"]
    plain = PS.run_cell(PS.StandInEngine(), "S", 2, steps=2); assert raises(lambda: C.merge_cells([plain]), ValueError)                           # a cell of the first run (no constraint) is refused
    other = copy.deepcopy(cells[0]); other["engine"]["constraint"]["regex_sha256"] = "0" * 64; assert raises(lambda: C.merge_cells([other]), ValueError)
    assert raises(lambda: C.merge_cells([cells[0], cells[0]]), ValueError) and raises(lambda: C.merge_cells([]))
    for field, val in (("seed", 1), ("steps", 7)):
        o = copy.deepcopy(cells[1]); o[field] = val; assert raises(lambda: C.merge_cells([cells[0], o]), ValueError), field
    o = copy.deepcopy(cells[1]); o["engine"]["version"] = "other"; assert raises(lambda: C.merge_cells([cells[0], o]), ValueError)
    o = copy.deepcopy(cells[1]); o["engine"]["gpu"] = "other"; assert raises(lambda: C.merge_cells([cells[0], o]), ValueError)
    with tempfile.TemporaryDirectory() as d:
        paths = [C.save_cell(c, os.path.join(d, f"{c['arm']}{c['K']}.json")) for c in cells]; out = os.path.join(d, "m.json"); mm = C.merge_files(paths, out); back = json.load(open(out))
        assert back["module_sha256"] == C.fingerprint() == hashlib.sha256(open(C.__file__, "rb").read()).hexdigest() and back["provider_side_sha256"] == PS.fingerprint() and back["cells"] == mm["cells"]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf): C.print_cells(C.summarize_all(back["cells"])); C.print_verdict(C.verdict_constrained(C.summarize_all(back["cells"])))
        assert "not run" in buf.getvalue()


# ---------------------------------------------------------------- 5 the cell runner and its checkpoint
def _same_jobs(a, b):
    assert len(a["sessions"]) == len(b["sessions"])
    for x, y in zip(a["sessions"], b["sessions"]):
        assert (x["seed"], x["script_steps"], x["stopped"]) == (y["seed"], y["script_steps"], y["stopped"])
        key = lambda r: (r["t"], r["kind"], r["text"], r["parsed"], r["admitted"], r["corrupted"], r["exact"], r["distance"], r["n_prompt"], r["n_new"])
        assert [key(r) for r in x["log"]] == [key(r) for r in y["log"]]


class _Crash(Exception): pass


def test_the_runner_is_the_first_runs_runner_plus_a_checkpoint_and_nothing_else():
    for arm, k in (("S", 3), ("KC", 3)):
        a = PS.run_cell(PS.StandInEngine(), arm, k, steps=6); b = C.run_cell(C.ConstrainedStandIn(), arm, k, steps=6)
        assert a["sessions"] == b["sessions"] and a["rounds"] == b["rounds"] and a["wall_seconds"] == b["wall_seconds"] and b["resumed_at"] == [] and b["engine"]["constraint"]["kind"] == "stand_in"
        assert {k_ for k_ in a if k_ not in ("engine",)} | {"resumed_at"} == {k_ for k_ in b if k_ not in ("engine",)}
    assert raises(lambda: C.run_cell(C.ConstrainedStandIn(), "KN", 2, steps=2), AssertionError) and raises(lambda: C.run_cell(C.ConstrainedStandIn(), "KR", 2, steps=2), AssertionError)
    seen = []; C.run_cell(C.ConstrainedStandIn(), "S", 2, steps=3, progress=lambda i, a, k: seen.append((i, a, k))); assert seen == [(1, "S", 2), (2, "S", 2), (3, "S", 2)]


def test_a_cell_stopped_after_any_round_and_resumed_gives_the_same_jobs_and_says_where_it_was_resumed():
    for arm, K, steps in (("KC", 4, 9), ("S", 3, 7)):
        whole = C.run_cell(C.ConstrainedStandIn(), arm, K, steps=steps); n_rounds = len(whole["rounds"])
        for stop in range(1, n_rounds):
            with tempfile.TemporaryDirectory() as d:
                ck = os.path.join(d, "c.ckpt.json"); calls = []
                def progress(i, a, k):
                    calls.append(i)
                    if i == stop: raise _Crash()
                try: C.run_cell(C.ConstrainedStandIn(), arm, K, steps=steps, progress=progress, checkpoint=ck)
                except _Crash: pass
                assert calls == list(range(1, stop + 1)) and os.path.exists(ck) and not os.path.exists(ck + ".tmp") and len(json.load(open(ck))["rounds"]) == stop
                back = C.run_cell(C.ConstrainedStandIn(), arm, K, steps=steps, checkpoint=ck)                      # a new engine: its cache is cold
                _same_jobs(whole, back); assert back["resumed_at"] == [stop] and len(back["rounds"]) == n_rounds and [r["active"] for r in back["rounds"]] == [r["active"] for r in whole["rounds"]]
                assert abs(back["wall_seconds"] - sum(r["seconds"] for r in back["rounds"])) < 1e-9 and [r["seconds"] for r in back["rounds"][:stop]] == [r["seconds"] for r in whole["rounds"][:stop]]
                assert back["rounds"][stop]["cached_tokens"] <= whole["rounds"][stop]["cached_tokens"]             # the first round after the resume starts cold: never more cached than in the uninterrupted cell
                if stop >= 2 and arm == "KC": assert back["rounds"][stop]["cached_tokens"] < whole["rounds"][stop]["cached_tokens"]
    with tempfile.TemporaryDirectory() as d:                                                                          # stopped twice
        ck = os.path.join(d, "c.json"); whole = C.run_cell(C.ConstrainedStandIn(), "KC", 3, steps=8)
        for stop in (2, 5):
            try: C.run_cell(C.ConstrainedStandIn(), "KC", 3, steps=8, progress=lambda i, a, k: (_ for _ in ()).throw(_Crash()) if i == stop else None, checkpoint=ck)
            except _Crash: pass
        back = C.run_cell(C.ConstrainedStandIn(), "KC", 3, steps=8, checkpoint=ck); _same_jobs(whole, back); assert back["resumed_at"] == [2, 5]
        again = C.run_cell(C.ConstrainedStandIn(), "KC", 3, steps=8, checkpoint=ck); _same_jobs(whole, again); assert again["resumed_at"] == [2, 5, 8] and len(again["rounds"]) == len(whole["rounds"])      # a finished checkpoint adds no round
    # a chat that stopped on the context rule is rebuilt as stopped
    with tempfile.TemporaryDirectory() as d:
        ck = os.path.join(d, "c.json"); whole = C.run_cell(C.ConstrainedStandIn(capacity_tokens=PS.CAPACITY_TOKENS), "KC", 2, steps=30)
        stops = [s["stopped"] for s in whole["sessions"]]
        try: C.run_cell(C.ConstrainedStandIn(), "KC", 2, steps=30, progress=lambda i, a, k: (_ for _ in ()).throw(_Crash()) if i == len(whole["rounds"]) - 1 else None, checkpoint=ck)
        except _Crash: pass
        back = C.run_cell(C.ConstrainedStandIn(), "KC", 2, steps=30, checkpoint=ck); _same_jobs(whole, back); assert [s["stopped"] for s in back["sessions"]] == stops


def test_a_checkpoint_that_belongs_to_another_cell_or_does_not_add_up_is_refused():
    with tempfile.TemporaryDirectory() as d:
        ck = os.path.join(d, "c.json")
        try: C.run_cell(C.ConstrainedStandIn(), "KC", 3, steps=6, progress=lambda i, a, k: (_ for _ in ()).throw(_Crash()) if i == 3 else None, checkpoint=ck)
        except _Crash: pass
        good = json.load(open(ck)); text = open(ck).read()
        assert raises(lambda: C.run_cell(C.ConstrainedStandIn(), "KC", 4, steps=6, checkpoint=ck), ValueError)                  # other K
        assert raises(lambda: C.run_cell(C.ConstrainedStandIn(), "S", 3, steps=6, checkpoint=ck), ValueError)                   # other arm
        assert raises(lambda: C.run_cell(C.ConstrainedStandIn(), "KC", 3, steps=7, checkpoint=ck), ValueError)                  # other length
        assert raises(lambda: C.run_cell(C.ConstrainedStandIn(), "KC", 3, steps=6, seed=1, checkpoint=ck), ValueError)           # other seed
        class Other(C.ConstrainedStandIn):
            def info(self): d_ = super().info(); d_["gpu"] = "another GPU"; return d_
        class Version(C.ConstrainedStandIn):
            def info(self): d_ = super().info(); d_["version"] = "other"; return d_
        class NoConstraint(PS.StandInEngine): pass
        assert raises(lambda: C.run_cell(Other(), "KC", 3, steps=6, checkpoint=ck), ValueError) and raises(lambda: C.run_cell(Version(), "KC", 3, steps=6, checkpoint=ck), ValueError)
        assert raises(lambda: C.run_cell(NoConstraint(), "KC", 3, steps=6, checkpoint=ck), ValueError)
        def tamper(fn):
            c = copy.deepcopy(good); fn(c); open(ck, "w").write(json.dumps(c))
            try: return raises(lambda: C.run_cell(C.ConstrainedStandIn(), "KC", 3, steps=6, checkpoint=ck), ValueError)
            finally: open(ck, "w").write(text)
        assert tamper(lambda c: c["rounds"].pop()) and tamper(lambda c: c["replies"].pop()) and tamper(lambda c: c["replies"][0].pop()) and tamper(lambda c: c["rounds"][1].__setitem__("active", 2))
        assert tamper(lambda c: c.__setitem__("format", 2)) and tamper(lambda c: c["replies"][0].append(["DEL[1,1]", 1, 1, 1]) or c["replies"][1].append(["DEL[1,1]", 1, 1, 1]) or c["replies"][2].append(["DEL[1,1]", 1, 1, 1]))
        assert not raises(lambda: C.run_cell(C.ConstrainedStandIn(), "KC", 3, steps=6, checkpoint=ck))                          # the untouched checkpoint is accepted


# ---------------------------------------------------------------- 6 the probe
def test_the_probe_feeds_the_engine_the_prompt_of_the_step_the_first_run_got_wrong_with_the_oracles_own_earlier_replies():
    class Rec(C.ConstrainedStandIn):
        def __init__(self): super().__init__(); self.seen = []
        def generate_round(self, requests, reset_cache=False): self.seen.append(requests); return super().generate_round(requests, reset_cache)
    e = Rec(); p = C.slip_probe(e)
    assert len(e.seen) == 1 and len(e.seen[0]) == 1 and (p["episode"], p["step"], p["kind"]) == (0, 20, "delete_dep") and p["parsed"] and p["reply"] == p["oracle"] == "DEL[4,5] DEL[(4,5)->(0,1):dep#1]"
    msgs = e.seen[0][0].messages; assert [m["role"] for m in msgs] == ["system", "user"]                                                      # exact-state: no chat history
    s = PS.Session("S", PS.SEED * 1000 + 0, PS.STEPS)
    while s.t < 19: s.request(); s.accept(PS.Reply(s.script[s.t]["ideal"], 0, 0, 0))
    assert msgs == s.request().messages and "Request:" in msgs[1]["content"]
    run1 = _first_run_cells()[("S", 8)]["sessions"][0]["log"]
    assert all(x["exact"] for x in run1[:19]) and run1[19]["t"] == 20 and run1[19]["text"] == "DEL[4,5:8]" and not run1[19]["parsed"] and run1[19]["kind"] == "delete_dep"    # the job was exact up to step 19: the state the probe rebuilds is the one it had
    for ep, st in ((1, 28), (6, 16), (3, 16)):
        q = C.slip_probe(C.ConstrainedStandIn(), episode=ep, step=st); assert (q["episode"], q["step"]) == (ep, st) and q["parsed"]
    assert raises(lambda: C.slip_probe(C.ConstrainedStandIn(), step=0), ValueError) and raises(lambda: C.slip_probe(C.ConstrainedStandIn(), step=31), ValueError)


# ---------------------------------------------------------------- 7 the lines
def _cell(arm, K, **kw):
    d = dict(arm=arm, K=K, engine_kind="vllm", capacity_tokens=80000, steps_done=30 * K, unparsed=0, truncated=0, cached_tokens=1000, exact_share=1.0, longest_prompt=3000, prompt_first_quarter=400.0, prompt_last_quarter=400.0,
             cached_share=0.9, cached_share_second_half=0.9, computed_per_step=100.0, seconds_per_step=0.1, exact_per_hour=30000.0, resumed_at=[])
    d.update(kw); return d


def _table():
    """Every line reads yes, with a margin."""
    return {("S", 8): _cell("S", 8, exact_share=0.99, prompt_first_quarter=400.0, prompt_last_quarter=380.0, computed_per_step=100.0, seconds_per_step=0.1, exact_per_hour=30000.0),
            ("KC", 8): _cell("KC", 8, longest_prompt=7000, prompt_first_quarter=1000.0, prompt_last_quarter=5000.0, cached_share=0.95, computed_per_step=300.0, seconds_per_step=0.5, exact_per_hour=7000.0),
            ("S", 32): _cell("S", 32, exact_share=0.99, seconds_per_step=0.1, exact_per_hour=30000.0), ("KC", 32): _cell("KC", 32, longest_prompt=7500, seconds_per_step=3.0, exact_per_hour=1000.0, cached_share_second_half=0.1),
            ("S", 64): _cell("S", 64, exact_share=0.99, seconds_per_step=0.1, exact_per_hour=30000.0), ("KC", 64): _cell("KC", 64, longest_prompt=7500, seconds_per_step=6.0, exact_per_hour=500.0, cached_share_second_half=0.05)}


def _read(T): return [ok for _, _, _, ok in C.verdict_constrained(T)]


def _flip(line, mutate, expect):
    T = _table(); mutate(T); got = _read(T)
    assert got[line - 1] is expect, (line, got); others = [g for i, g in enumerate(got) if i != line - 1]; return got, others


def test_the_seventeen_lines_all_read_yes_on_a_table_built_to_satisfy_them_and_the_text_is_the_pages():
    r = C.verdict_constrained(_table()); assert len(r) == 17 == len(C.CL_LINES) and [x[0] for x in r] == list(range(1, 18)) and all(x[3] is True for x in r), [x[3] for x in r]
    tags = [t for t, _ in C.CL_LINES]; assert tags[:3] == ["instrument"] * 3 and tags[3] == "guard" and set(tags[4:]) == {"claim"} and C.CL_HEADLINE == (15, 17)


def test_each_line_flips_at_its_edge_and_a_missing_cell_reads_not_run_never_no():
    def set_(T, key, **kw): T[key].update(kw)
    edge = [  # (line, mutation that still reads yes at the edge, mutation just past it)
        (1, None, lambda T: set_(T, ("KC", 8), cached_tokens=None)),
        (2, lambda T: set_(T, ("S", 8), unparsed=1, steps_done=200), lambda T: set_(T, ("S", 8), unparsed=2, steps_done=200)),
        (2, lambda T: set_(T, ("S", 8), unparsed=0, truncated=0), lambda T: set_(T, ("S", 8), truncated=1)),
        (3, lambda T: set_(T, ("KC", 64), longest_prompt=2500), lambda T: set_(T, ("KC", 64), longest_prompt=2499)),
        (3, lambda T: set_(T, ("KC", 8), longest_prompt=10000), lambda T: set_(T, ("KC", 8), longest_prompt=10001)),
        (4, lambda T: set_(T, ("S", 64), exact_share=0.80), lambda T: set_(T, ("S", 64), exact_share=0.79)),
        (4, lambda T: set_(T, ("KC", 32), exact_share=0.80), lambda T: set_(T, ("KC", 32), exact_share=0.79)),
        (5, lambda T: (set_(T, ("KC", 32), exact_share=1.0), set_(T, ("S", 32), exact_share=0.95)), lambda T: (set_(T, ("KC", 32), exact_share=1.0), set_(T, ("S", 32), exact_share=0.949))),
        (6, lambda T: set_(T, ("S", 32), exact_share=0.98), lambda T: set_(T, ("S", 32), exact_share=0.979)),
        (7, lambda T: set_(T, ("S", 8), prompt_first_quarter=400.0, prompt_last_quarter=520.0), lambda T: set_(T, ("S", 8), prompt_first_quarter=400.0, prompt_last_quarter=521.0)),
        (8, lambda T: set_(T, ("KC", 8), prompt_first_quarter=1000.0, prompt_last_quarter=3500.0), lambda T: set_(T, ("KC", 8), prompt_first_quarter=1000.0, prompt_last_quarter=3499.0)),
        (9, lambda T: set_(T, ("KC", 8), cached_share=0.80), lambda T: set_(T, ("KC", 8), cached_share=0.79)),
        (10, lambda T: (set_(T, ("S", 8), computed_per_step=70.0), set_(T, ("KC", 8), computed_per_step=100.0)), lambda T: (set_(T, ("S", 8), computed_per_step=71.0), set_(T, ("KC", 8), computed_per_step=100.0))),
        (11, lambda T: (set_(T, ("S", 8), seconds_per_step=0.4), set_(T, ("KC", 8), seconds_per_step=0.5)), lambda T: (set_(T, ("S", 8), seconds_per_step=0.41), set_(T, ("KC", 8), seconds_per_step=0.5))),
        (12, lambda T: (set_(T, ("S", 32), seconds_per_step=1.0), set_(T, ("KC", 32), seconds_per_step=1.3)), lambda T: (set_(T, ("S", 32), seconds_per_step=1.0), set_(T, ("KC", 32), seconds_per_step=1.29))),
        (13, lambda T: set_(T, ("KC", 64), cached_share_second_half=0.4999), lambda T: set_(T, ("KC", 64), cached_share_second_half=0.5)),
        (14, lambda T: (set_(T, ("S", 64), seconds_per_step=1.0), set_(T, ("KC", 64), seconds_per_step=2.0)), lambda T: (set_(T, ("S", 64), seconds_per_step=1.0), set_(T, ("KC", 64), seconds_per_step=1.99))),
        (15, lambda T: (set_(T, ("S", 8), exact_per_hour=2000.0), set_(T, ("KC", 8), exact_per_hour=1000.0)), lambda T: (set_(T, ("S", 8), exact_per_hour=1999.0), set_(T, ("KC", 8), exact_per_hour=1000.0))),
        (16, lambda T: (set_(T, ("S", 32), exact_per_hour=10000.0), set_(T, ("KC", 32), exact_per_hour=1000.0)), lambda T: (set_(T, ("S", 32), exact_per_hour=9999.0), set_(T, ("KC", 32), exact_per_hour=1000.0))),
        (17, lambda T: (set_(T, ("S", 64), exact_per_hour=10000.0), set_(T, ("KC", 64), exact_per_hour=1000.0)), lambda T: (set_(T, ("S", 64), exact_per_hour=9999.0), set_(T, ("KC", 64), exact_per_hour=1000.0)))]
    for line, yes, no in edge:
        if yes is not None:
            T = _table(); yes(T); assert _read(T)[line - 1] is True, ("edge yes", line)
        T = _table(); no(T); got = _read(T); assert got[line - 1] is False, ("edge no", line, got)
    T = _table(); T[("KC", 64)]["exact_per_hour"] = 0.0; assert _read(T)[16] is True                                       # a kept chat with no exact step at all: exact-state wins by any ratio
    T = _table(); T[("KC", 64)]["exact_per_hour"] = 0.0; T[("S", 64)]["exact_per_hour"] = 0.0; assert _read(T)[16] is None
    # a cell that was not run: its lines are None, never False, and the others do not move
    full = _read(_table())
    for missing, lines in ((("KC", 64), (3, 13, 14, 17)), (("KC", 32), (12, 16)), (("S", 32), (12, 16)), (("S", 64), (14, 17)), (("KC", 8), (3, 8, 9, 10, 11, 15)), (("S", 8), (7, 10, 11, 15))):
        T = _table(); del T[missing]; got = _read(T)
        for l in lines: assert got[l - 1] is None, (missing, l, got)
        assert all(got[i] is True for i in range(17) if i + 1 not in lines), (missing, got)                                        # every other line still reads yes
    T = _table(); T[("S", 8)]["steps_done"] = 0; assert _read(T)[6] is None and _read(T)[10] is None                         # a cell with no step counts as not run
    assert all(x is None for x in _read({}))


def test_the_rule_about_what_may_be_claimed_follows_line_5_and_the_printout_says_it():
    T = _table(); a = C.claims_allowed(C.verdict_constrained(T)); assert a == dict(plain_time_at_equal_exactness=True, quality_adjusted={15: True, 16: True, 17: True})
    T[("S", 8)]["exact_share"] = 0.88; T[("KC", 8)]["exact_share"] = 1.0; r = C.verdict_constrained(T); a = C.claims_allowed(r); assert a["plain_time_at_equal_exactness"] is False and a["quality_adjusted"][15] is True and r[3][3] is True and r[5][3] is False
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): C.print_verdict(r)
    t = buf.getvalue(); assert "NOT claimed as equal-exactness ratios" in t and "lines 15 to 17" in t and "17 lines could be read" in t.replace("17 of 17", "17 lines").replace("(17 lines", "17 lines") or "17 of 17 lines could be read" in t
    assert "headline lines 15 and 17: yes, yes" in t
    del T[("KC", 8)]; del T[("KC", 32)]; del T[("KC", 64)]; r = C.verdict_constrained(T); assert C.claims_allowed(r)["plain_time_at_equal_exactness"] is None
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): C.print_verdict(r)
    assert "line 5 was not run" in buf.getvalue()


def test_the_summary_adds_what_the_new_lines_read_and_exact_steps_per_hour_is_counted_from_the_steps():
    c = C.run_cell(C.ConstrainedStandIn(), "S", 3, steps=6); s = C.summarize_cell2(c)
    n = 3 * 6; assert s["steps_done"] == n and s["unparsed"] == 0 and s["truncated"] == 0 and s["exact_steps"] == n and abs(s["exact_per_hour"] - 3600.0 * n / c["wall_seconds"]) < 1e-9 and s["resumed_at"] == [] and s["constraint"] == C.REGEX_SHA256
    for x in c["sessions"][0]["log"][:2]: x["exact"] = False; x["parsed"] = False
    c["sessions"][1]["log"][0]["n_new"] = 96; s = C.summarize_cell2(c); assert s["exact_steps"] == n - 2 and s["unparsed"] == 2 and s["truncated"] == 1 and abs(s["exact_per_hour"] - 3600.0 * (n - 2) / c["wall_seconds"]) < 1e-9
    assert raises(lambda: C.summarize_all([c, c]), AssertionError)


def test_the_dry_run_prints_the_stand_in_table_and_says_it_is_not_a_result():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): S = C.dry_run(steps=6, levels=(8,))
    t = buf.getvalue(); assert "NOT a result" in t and ("S", 8) in S and ("KC", 8) in S and ("KC", 32) not in S and all(s["engine_kind"] == "stand_in" for s in S.values()) and "exact/h" in t


# ---------------------------------------------------------------- 8 the locked page
PAGE = os.path.join(ROOT, "docs", "PROVIDER_SIDE_CONSTRAINED_TEST.md")
RESULT_MARK = "\n\n---\n\n# Result"
PAGE_SHA256 = "4559fdc4eda01013d456c1407c0bd23e3e634751394d6f65cc90d891cca7e575"
PROBS = [97, 94, 95, 85, 5, 2, 95, 94, 95, 82, 94, 96, 96, 95, 88, 93, 92]


def _h(text): return hashlib.sha256((text.rstrip("\n") + "\n").encode()).hexdigest()


def _head():
    text = open(PAGE).read(); assert text.count(RESULT_MARK) <= 1
    return text[:text.index(RESULT_MARK)] if RESULT_MARK in text else text


def test_the_page_is_locked_and_carries_the_seventeen_lines_the_regular_expression_the_settings_and_the_thresholds_the_code_uses():
    head = _head(); assert _h(head) == PAGE_SHA256, "the predictions and reading rules must not change after the run; later notes go below them"
    for i, (tag, line) in enumerate(C.CL_LINES, 1): assert f"\n{i}. [{tag}] {line} (" in head, i
    probs = [int(m.group(1)) for m in re.finditer(r"\n\d+\. \[\w+\] .*\((\d+) %\)(?=\n)", head)]; assert probs == PROBS, probs
    assert "```\n" + C.REGEX + "\n```" in head and C.REGEX_SHA256 in head
    assert "**Headline** (lines 15 and 17)" in head and "both lines reading yes: about **65 %**" in head and "The pattern I expect (lines 5 and 6 NO, every other line yes): about **25 %**" in head and "All 17 yes: under 1 %" in head and "Lines 5 and 6 reading yes: about 5 % and 2 %" in head
    assert (C.C_PARSED, C.C_MIN_EXACT, C.C_EXACT_GAP, C.C_EXACT_HIGH, C.C_FLAT, C.C_GROWTH, C.C_CACHE_FITS, C.C_LIGHT_TOKENS, C.C_LIGHT_TIME, C.C_MID_RATIO, C.C_OVERRUN, C.C_PRESSURE_RATIO, C.C_QA_LIGHT, C.C_QA_MID, C.C_QA_HEAVY) == (0.995, 0.80, 0.05, 0.98, 1.3, 3.5, 0.80, 0.30, 0.20, 1.3, 0.50, 2.0, 2.0, 10.0, 10.0)
    assert "line 2: 99.5 percent parsed, no reply with 96 new tokens; line 4: 0.80; line 5: 0.05; line 6: 0.98; line 7: 1.3; line 8: 3.5; line 9: 0.80; line 10: 0.30 fewer; line 11: 0.20 less; line 12: 1.3 times; line 13: 0.50 (strictly below); line 14: 2.0 times; lines 15, 16, 17: 2, 10 and 10 times." in head
    assert C.CL_HEADLINE == (15, 17) and C.CL_PLAIN_TIME_LINES == (11, 12, 14)
    for needle in ("Concept-map item 107", "`python/axi/experiments/provider_side_constrained.py`", "`python/tests/test_provider_side_constrained.py`", "`python/colab/axi_provider_side_constrained.ipynb`", "Qwen/Qwen2.5-7B-Instruct-AWQ", "80,000 tokens",
                   "vLLM 0.31.0", "`S` 8, `KC` 8, `S` 32, `S` 64, `KC` 32, `KC` 64", "**energy (never measured; only time)**", "This page makes no claim of \"43 times\"", "Line 5 reads NO: they are **not** claimed as equal-exactness ratios",
                   "3,600 times the number of steps after which the held state equals the oracle's", "step 20 of job 0", "**A cell can be carried on.**"): assert needle in head, needle
    assert (C.MODEL, C.STEPS, C.SEED, C.MAX_NEW, C.CAPACITY_TOKENS, C.POLICY) == ("Qwen/Qwen2.5-7B-Instruct-AWQ", 30, 20261017, 96, 80000, "repair_m6") and C.cells_planned() == [("S", 8), ("KC", 8), ("S", 32), ("S", 64), ("KC", 32), ("KC", 64)]
    assert head.count("\n# ") == 0 and head.startswith("# Test 15: the serving side again")


def test_the_figures_the_page_quotes_from_the_first_run_are_the_ones_in_its_saved_files():
    cells = _first_run_cells(); head = _head()
    # the refused replies
    refused = [(k, x) for k, c in cells.items() for s in c["sessions"] for x in s["log"] if not x["parsed"]]; assert len(refused) == 89 and sum(1 for k, _ in refused if k[0] == "S") == 80 and sum(1 for k, _ in refused if k[0] == "KC") == 9
    texts = [x["text"].strip() for _, x in refused]; pure = [t for t in texts if re.fullmatch(r"DEL\[\d+,\d+:-?\d+\]", t)]; lists = [t for t in texts if t == "DEL[1,0:6] ADD[1,0:5]"]; edge = [t for t in texts if re.fullmatch(r"ADD\[\(\d+,\d+\):\d+\]", t)]
    assert (len(pure), len(lists), len(edge)) == (82, 2, 5) and sorted(collections_counter(edge).items()) == [("ADD[(3,0):4]", 2), ("ADD[(4,4):6]", 3)]
    assert sum(len(s["log"]) for c in cells.values() for s in c["sessions"]) == 4552 and max(x["n_new"] for c in cells.values() for s in c["sessions"] for x in s["log"]) == 31
    for s in ("89 of the 4,552 replies", "(80 in the `S` cells, 9 in the `KC` cells)", "82 of them are exactly `DEL[r,c:v]`", "2 more are a list that starts with it", "`ADD[(4,4):6]` (3 times) and `ADD[(3,0):4]` (2 times)", "The longest reply of the first run was 31 tokens"): assert s in head, s
    # the exactness of the arms and the times
    def ex(k): c = cells[k]; st = [x for s in c["sessions"] for x in s["log"]]; return round(100.0 * sum(x["exact"] for x in st) / len(st), 1), len(st)
    assert [ex(("S", k))[0] for k in (8, 32, 64)] == [76.2, 78.2, 69.2] and [ex(("KC", k))[0] for k in (8, 32)] == [100.0, 97.3]
    assert "76.2, 78.2 and 69.2 percent" in head and "100.0 percent at 8 jobs and 97.3 at 32" in head
    sps = lambda k: cells[k]["wall_seconds"] / ex(k)[1]
    assert [round(sps(("S", k)), 3) for k in (8, 32, 64)] == [0.154, 0.121, 0.107] and round(sps(("KC", 8)), 3) == 0.617 and round(sps(("KC", 32)), 3) == 5.149 and round(sps(("KC", 32)) / sps(("S", 32)), 1) == 42.7
    assert [round(cells[k]["wall_seconds"]) for k in (("S", 8), ("S", 32), ("S", 64), ("KC", 8), ("KC", 32), ("KN", 8))] == [37, 116, 205, 148, 4902, 1646]
    for s in ("`S` 37, 116 and 205 s at 8, 32 and 64 jobs (0.154, 0.121 and 0.107 s a step)", "`KC` 148 s at 8 jobs (0.617 s a step) and 4,902 s at 32 jobs (5.149 s a step, 1 h 22 min)", "`KN` 1,646 s", "The ratio of the kept chat's time per step to exact-state's at 32 jobs was 42.7"): assert s in head, s
    assert max(x["n_prompt"] for s in cells[("KC", 32)]["sessions"] for x in s["log"]) == 7556 and max(x["n_prompt"] for s in cells[("KC", 8)]["sessions"] for x in s["log"]) == 7154 and max(x["n_prompt"] for k in cells if k[0] == "S" for s in cells[k]["sessions"] for x in s["log"]) == 516
    assert "7,556 tokens (32 jobs) and 7,154 (8 jobs)" in head and "516 tokens or fewer" in head


def collections_counter(xs):
    out = {}
    for x in xs: out[x] = out.get(x, 0) + 1
    return out


def test_the_estimate_on_the_page_is_recomputed_from_the_first_runs_step_records_and_the_first_eight_jobs_repeat_across_cells():
    cells = _first_run_cells(); head = _head(); slip = re.compile(r"DEL\[\d+,\d+:-?\d+\]( |$)")
    rows = {}
    for K in (8, 32, 64):
        c = cells[("S", K)]; total = sum(len(s["log"]) for s in c["sessions"]); missed = []; behind = 0; on_slip = 0
        for i, s in enumerate(c["sessions"]):
            first = next((x for x in s["log"] if not x["exact"]), None)
            if first is None: continue
            missed.append(i)
            if not first["parsed"] and slip.match(first["text"].strip()): on_slip += 1
            else: behind += len(s["log"]) - first["t"] + 1; rows.setdefault(K, []).append((i, first["t"]))
        rows[K, "n"] = (len(missed), on_slip, len(c["sessions"]), round(100.0 * (1 - behind / total), 1))
    assert rows[8, "n"] == (5, 3, 8, 88.3) and rows[32, "n"] == (14, 9, 32, 93.9) and rows[64, "n"] == (37, 31, 64, 95.9), {k: v for k, v in rows.items() if isinstance(k, tuple)}
    assert rows[8] == [(3, 16), (4, 18)] and rows[32] == [(3, 16), (4, 18), (17, 20), (19, 12), (20, 30)] and rows[64] == rows[32] + [(62, 12)]
    for s in ("| `S` 8 | 5 of 8 | 3 | 2 (job 3 at 16, job 4 at 18) | 88.3 % | 76.2 % |", "| `S` 32 | 14 of 32 | 9 | 5 (jobs 3, 4, 17, 19, 20 at 16, 18, 20, 12, 30) | 93.9 % | 78.2 % |", "| `S` 64 | 37 of 64 | 31 | 6 (the five and job 62 at 12) | 95.9 % | 69.2 % |",
              "(job 28 at 32 jobs, `DEL[1,0:6]` at step 1)", "4 of the 56 `S` jobs that missed did", "job 4 at step 18 (`ADD[1,0:6]`, refused by the gate)"): assert s in head, s
    kc = cells[("KC", 32)]["sessions"]; lost = [(i, next(x for x in s["log"] if not x["exact"])) for i, s in enumerate(kc) if any(not x["exact"] for x in s["log"])]; assert len(lost) == 1 and lost[0][0] == 28 and lost[0][1]["t"] == 1 and lost[0][1]["text"] == "DEL[1,0:6]"
    recovered = 0; missed56 = 0
    for K in (8, 32, 64):
        for s in cells[("S", K)]["sessions"]:
            ex = [x["exact"] for x in s["log"]]
            if all(ex): continue
            missed56 += 1; first = ex.index(False)
            if any(ex[first:]): recovered += 1
    assert (missed56, recovered) == (56, 4)
    for arm, Ks in (("S", (8, 32, 64)), ("KC", (8, 32))):                                                    # the first 8 jobs repeat across cells: same replies, same prompt sizes
        for i in range(8):
            ref = [(x["text"], x["n_prompt"]) for x in cells[(arm, Ks[0])]["sessions"][i]["log"]]
            for K in Ks[1:]: assert [(x["text"], x["n_prompt"]) for x in cells[(arm, K)]["sessions"][i]["log"]] == ref, (arm, i, K)
    assert "the first 8 jobs gave the same replies and the same prompt sizes at 8, 32 and 64 jobs for `S`, and at 8 and 32 for `KC`" in head


# ---------------------------------------------------------------- 9 the notebook
NOTEBOOK = os.path.join(HERE, "..", "colab", "axi_provider_side_constrained.ipynb")


def _nb(): return json.load(open(NOTEBOOK))


def _srcs(): return ["".join(c["source"]) for c in _nb()["cells"]], [c["cell_type"] for c in _nb()["cells"]]


def test_the_notebook_installs_the_pinned_engine_fixes_the_stream_loads_the_constrained_engine_checks_runs_the_cells_in_order_and_puts_them_together():
    srcs, kinds = _srcs(); code = [s for s, k in zip(srcs, kinds) if k == "code"]; text = "\n".join(srcs); nb = _nb()
    import ast
    for s in code: ast.parse("\n".join(l for l in s.splitlines() if not l.lstrip().startswith("!")))
    for needle in ("git clone -q --depth 1 https://github.com/gaurshalendra2412/axi-runtime", "provider_side_constrained.py", "C.ConstrainedVLLMEngine()", "eng.warm_up()", "C.slip_probe(eng)", "STOP:", "cached tokens", "C.run_cell(eng, arm, 3, steps=3)", "C.save_cell", "store.persist",
                   "C.merge_files", "C.verdict_constrained", "C.print_cells", "files.download", "open_store", "docs/PROVIDER_SIDE_CONSTRAINED_TEST.md", "Do not paste any keys", "checkpoint=store.path(ck)", "update 58 commands"): assert needle in text, needle
    assert text.count("Do not paste any keys") >= 2 and "!pip -q install vllm==0.31.0" in [x.strip() for x in code] and "!pip -q install vllm" not in [x.strip() for x in code] and "open_store(\"/content/out\", \"axi_provider_constrained\")" in text
    fix = [i for i, (s, k) in enumerate(zip(srcs, kinds)) if k == "code" and "fileno" in s]; load = [i for i, (s, k) in enumerate(zip(srcs, kinds)) if k == "code" and "C.ConstrainedVLLMEngine()" in s]; assert len(fix) == 1 and len(load) == 1 and fix[0] < load[0]
    ran = []; ns = dict(go=lambda arm, K: ran.append((arm, K)))
    for s in code:
        if re.fullmatch(r"(for arm, K in \(.*\): go\(arm, K\)|go\(\"K[CS]\", \d+\))\s*", s.strip()): exec(s, ns)
    assert ran == C.cells_planned(), ran
    assert nb["metadata"]["accelerator"] == "GPU" and nb["nbformat"] == 4 and kinds[0] == "markdown"
    head = srcs[0]; assert "T4" in head and "80,000 tokens" in head and "Qwen2.5-7B-Instruct" in head and "AWQ" in head and "vLLM 0.31.0" in head and "three to five hours, not measured" in head and "continues from its last finished round" in head and "Keep this browser tab open" in head
    assert "exact steps per GPU hour" in head and "every reply of both arms match the repository's grammar" in head.replace("**", "")


def test_the_notebook_cell_runner_skips_finished_cells_and_carries_on_a_cell_the_last_session_left_unfinished():
    srcs, kinds = _srcs(); runner = next(s for s, k in zip(srcs, kinds) if k == "code" and "def go(arm, K):" in s); body = runner[runner.index("def go(arm, K):"):]
    class Crash(Exception): pass
    class Dying(C.ConstrainedStandIn):
        def __init__(self, die_at): super().__init__(); self.n = 0; self.die_at = die_at
        def generate_round(self, requests, reset_cache=False):
            self.n += 1
            if self.n == self.die_at: raise Crash()
            return super().generate_round(requests, reset_cache)
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as remote:
        sys.path.insert(0, os.path.join(HERE, ".."))
        from colab.colab_persist import Store
        shim = types.SimpleNamespace(run_cell=lambda e, arm, K, progress=None, checkpoint=None: C.run_cell(e, arm, K, steps=5, progress=progress, checkpoint=checkpoint), save_cell=C.save_cell, summarize_cell2=C.summarize_cell2)
        def session(engine):
            store = Store(d, remote); ns = dict(C=shim, eng=engine, store=store, os=os, time=__import__("time"), json=json); exec(body, ns); return ns["go"]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            go = session(Dying(die_at=3))                                                                      # the first session dies in round 3 of KC at 3 jobs
            try: go("KC", 3)
            except Crash: pass
            assert sorted(os.listdir(remote)) == ["axi_provider_c_KC_K3.ckpt.json"] and len(json.load(open(os.path.join(remote, "axi_provider_c_KC_K3.ckpt.json")))["rounds"]) == 2      # the checkpoint was copied to Drive after round 2
            for f in os.listdir(d): os.remove(os.path.join(d, f))                                              # the machine is wiped
            go = session(C.ConstrainedStandIn()); go("KC", 3); go("S", 3); go("KC", 3)                          # a new session: restores, carries on, finishes; S runs; the third call finds the file
        out = buf.getvalue()
        assert "carrying on from round 2" in out and out.count("already finished, skipping") == 1 and out.count("done:") == 2 and "resumed at rounds [2]" in out
        assert sorted(f for f in os.listdir(remote) if not f.endswith(".ckpt.json")) == ["axi_provider_c_KC_K3.json", "axi_provider_c_S_K3.json"]
        paths = [os.path.join(remote, f) for f in sorted(os.listdir(remote)) if not f.endswith(".ckpt.json")]; m = C.merge_files(paths, os.path.join(d, "all.json")); assert len(m["cells"]) == 2 and m["cells"][1]["resumed_at"] == [2]


def test_the_safety_cell_says_ok_on_a_working_engine_and_stop_for_each_thing_that_can_be_wrong():
    srcs, kinds = _srcs(); cell = next(s for s, k in zip(srcs, kinds) if k == "code" and "smoke = " in s)
    def run(engine):
        buf = io.StringIO(); ns = dict(C=C, eng=engine)
        with contextlib.redirect_stdout(buf): exec(cell, ns)
        return buf.getvalue()
    out = run(C.ConstrainedStandIn()); assert out.strip().splitlines()[-1].startswith("OK: cached counts are reported") and "STOP" not in out and "the first run answered 'DEL[4,5:8]'" in out and "the strict parser reads it" in out
    class NoCount(C.ConstrainedStandIn):
        def generate_round(self, requests, reset_cache=False): reps, dt = super().generate_round(requests, reset_cache); [setattr(r, "n_cached", None) for r in reps]; return reps, dt
    class NoCache(C.ConstrainedStandIn):
        def generate_round(self, requests, reset_cache=False): reps, dt = super().generate_round(requests, reset_cache); [setattr(r, "n_cached", 0) for r in reps]; return reps, dt
    class ProbeOnly(C.ConstrainedStandIn):
        def generate_round(self, requests, reset_cache=False):
            reps, dt = super().generate_round(requests, reset_cache)
            if len(requests) == 1 and "Request:" in requests[0].messages[-1]["content"] and requests[0].stand_in_reply.startswith("DEL[4,5]"): reps[0].text = "DEL[4,5:8]"
            return reps, dt
    assert "STOP: this vLLM version does not report cached tokens" in run(NoCount())
    assert "STOP: the cache did nothing for the kept chat's second step" in run(NoCache())
    assert "STOP: the probe's reply is refused by the strict parser" in run(ProbeOnly()) and "REFUSES it" in run(ProbeOnly())
    class Garbage(C.ConstrainedStandIn):
        def generate_round(self, requests, reset_cache=False): reps, dt = super().generate_round(requests, reset_cache); reps[0].text = "Sure, here is the delta: " + reps[0].text; return reps, dt
    assert "STOP: a reply of the safety cell was refused by the strict parser" in run(Garbage())


# ---------------------------------------------------------------- 10 the front pages
def test_the_front_pages_point_to_the_tests_and_say_what_the_run_found():
    cm = open(os.path.join(ROOT, "docs", "RAJNISH_CONCEPT_MAP.md")).read(); ve = open(os.path.join(ROOT, "docs", "VERIFIED.md")).read()
    assert "PROVIDER_SIDE_CONSTRAINED_TEST.md" in cm and "PROVIDER_SIDE_CONSTRAINED_TEST.md" in ve
    item = cm[cm.index("107. **"): cm.index("108. **")]; row = ve[ve.index("PROVIDER_SIDE_CONSTRAINED_TEST.md") - 300: ve.index("PROVIDER_SIDE_CONSTRAINED_TEST.md") + 4000]
    for text in (item, row):
        assert "Run (18 and 19 Oct" in text and "NOT run" not in text and "test 124" in text.replace("Tests 124", "test 124") and "test 125" in text and PAGE_SHA256[:16] in text.replace(PAGE_SHA256, PAGE_SHA256[:16])
        assert "14 of 17 lines" in text.replace("14 of the 17 lines read yes", "14 of 17 lines yes").replace("14 of 17 lines yes", "14 of 17 lines") and ("lines 2, 5 and 6" in text or "line 2 reads NO" in text) and "neither the plain time ratios nor the exact-steps-per-hour figures are claimed" in text.replace("neither the time ratios nor", "neither the plain time ratios nor")
        assert "20,878" in text and "28,748" in text and "32,896" in text and "5,923" in text and "727" in text and "667" in text and "88.3" in text and "97.2" in text
    assert "Next free: row 169, item 109, test 127, question 120" in cm and "test 124 checks the constrained provider-side test and test 125 its Result, item 107" in cm


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("PASS", n)
