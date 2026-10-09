"""Implements concept-map row 9 / 27, item 85 (addendum of 9 Oct): the 14B test must survive Colab's usage limit WITHOUT becoming a smaller test (docs/SCALE_TEST_14B.md, "Addendum (9 Oct 2026)").
Checks the INSTRUMENT, not a result:
  1. `Store` copies a progress file to a remote folder through a temporary name and an atomic rename (a copy that dies halfway leaves the last good copy), brings it back only when the local one is
     missing or smaller, and does nothing harmful when there is no Drive;
  2. `open_store` mounts Drive and points the Store at MyDrive/<subdir>; when Drive is refused or missing it says so and still returns a working Store;
  3. SESSION DEATH: a `run_long` that dies in the middle of an episode, with the local folder wiped (what Colab does), continues from the Drive copy and returns EXACTLY what a run that was never
     interrupted returns, and the second session asks the model only for the episodes that were missing;
  4. the Part-2-only notebook runs exactly the same test as the main notebook (same model string, episodes, steps, seed, rule flag, policies, mix, file names), is well formed, and runs end to end
     (including a stop in the middle and a restart) on scripted stand-ins with a fake Drive;
  5. the pre-registration text of SCALE_TEST_14B.md above the addendum is byte for byte the file written before the run (its hash is the one of the file delivered in update 25), and the addendum says plainly that the test is unchanged.
What it does NOT show: anything about the 14B itself, or that the real Drive behaves as the fake one does (the notebook prints '(saved)' or '(NOT saved to Drive)' after every episode so you can see it)."""
import ast, contextlib, hashlib, io, json, os, shutil, sys, tempfile, types
sys.path.insert(0, os.path.dirname(__file__))
from axi.experiments.long_horizon import run_long
from colab.colab_persist import Store, open_store
from test_scope import StandIn

HERE = os.path.dirname(__file__)
PAGE = os.path.join(HERE, "..", "..", "docs", "SCALE_TEST_14B.md")
NB_MAIN = os.path.join(HERE, "..", "colab", "axi_14b_test.ipynb")
NB_P2 = os.path.join(HERE, "..", "colab", "axi_14b_part2.ipynb")
PREREG_SHA256 = "e51b2a8b44941b03b7df6e727029bb650f97a223cc639d5010e0b802c471c29b"        # sha256 of docs/SCALE_TEST_14B.md as delivered (update 25), before the addendum was added
ADDENDUM = "\n\n---\n\n# Addendum (9 Oct 2026)"


def write(path, text):
    with open(path, "w") as f: f.write(text)


def read(path):
    with open(path) as f: return f.read()


# ---------------------------------------------------------------- 1: Store
def test_store_without_a_remote_folder_keeps_working_and_keeps_nothing_across_sessions():
    tmp = tempfile.mkdtemp(); st = Store(os.path.join(tmp, "out"))
    assert not st.persistent and st.path("a.jsonl") == os.path.join(tmp, "out", "a.jsonl")
    assert st.restore("a.jsonl") == "nothing" and st.persist("a.jsonl") is False
    write(st.path("a.jsonl"), "x\n"); assert st.restore("a.jsonl") == "kept local" and st.persist("a.jsonl") is False
    seen = []; cb = st.progress("a.jsonl", lambda i, n: seen.append((i, n))); cb(2, 6); assert seen == [(2, 6)]       # the counter still runs
    st.progress("a.jsonl")(1, 1)                                                                                      # and no counter is fine too


def test_store_persist_is_a_copy_through_a_temporary_name_and_an_atomic_rename():
    tmp = tempfile.mkdtemp(); st = Store(os.path.join(tmp, "out"), os.path.join(tmp, "drive", "axi_14b")); assert st.persistent and os.path.isdir(st.remote_dir)
    assert st.persist("a.jsonl") is False                                         # nothing to save yet
    write(st.path("a.jsonl"), "v1\n"); assert st.persist("a.jsonl") is True
    remote = os.path.join(st.remote_dir, "a.jsonl"); assert read(remote) == "v1\n" and not os.path.exists(remote + ".tmp")
    write(st.path("a.jsonl"), "v1\nv2\n"); assert st.persist("a.jsonl") and read(remote) == "v1\nv2\n"
    # a session that dies while the copy is under way: the last good copy is still there and is the one restored
    import colab.colab_persist as cp
    real = cp.shutil.copyfile
    def dying(src, dst):
        with open(dst, "w") as f: f.write("v1\nv2\nv3 half")                       # the temporary file is half written, then the session dies
        raise OSError("session died")
    cp.shutil.copyfile = dying
    try:
        write(st.path("a.jsonl"), "v1\nv2\nv3\n")
        try: st.persist("a.jsonl"); raise AssertionError("must raise")
        except OSError: pass
    finally: cp.shutil.copyfile = real
    assert read(remote) == "v1\nv2\n"                                              # untouched
    shutil.rmtree(st.local_dir); os.makedirs(st.local_dir)                         # the new session
    fresh = Store(st.local_dir, st.remote_dir); assert fresh.restore("a.jsonl") == "restored" and read(fresh.path("a.jsonl")) == "v1\nv2\n"
    assert fresh.restore("zzz.jsonl") == "nothing"                                 # the leftover .tmp is never taken for a progress file
    assert os.path.exists(remote + ".tmp")
    write(fresh.path("a.jsonl"), "v1\nv2\nv3\n"); assert fresh.persist("a.jsonl") and read(remote) == "v1\nv2\nv3\n" and not os.path.exists(remote + ".tmp")      # the next good copy replaces it


def test_store_restore_takes_the_remote_copy_only_when_the_local_one_is_missing_or_smaller():
    tmp = tempfile.mkdtemp(); st = Store(os.path.join(tmp, "out"), os.path.join(tmp, "drive"))
    remote = os.path.join(st.remote_dir, "a.jsonl")
    assert st.restore("a.jsonl") == "nothing"
    write(remote, "1\n2\n3\n"); assert st.restore("a.jsonl") == "restored" and read(st.path("a.jsonl")) == "1\n2\n3\n"          # local missing
    assert st.restore("a.jsonl") == "kept local"                                                                            # same size
    write(st.path("a.jsonl"), "1\n2\n3\n4\n"); assert st.restore("a.jsonl") == "kept local" and read(st.path("a.jsonl")) == "1\n2\n3\n4\n"      # local further on: never overwritten by an older copy
    write(st.path("a.jsonl"), "1\n"); assert st.restore("a.jsonl") == "restored" and read(st.path("a.jsonl")) == "1\n2\n3\n"      # local behind
    write(st.path("b.jsonl"), "only here\n"); assert st.restore("b.jsonl") == "kept local"


def _fake_drive(mount_ok=True):
    calls = []; google = types.ModuleType("google"); colab = types.ModuleType("google.colab"); drv = types.ModuleType("google.colab.drive")
    def mount(p):
        calls.append(p)
        if not mount_ok: raise ValueError("permission refused")
        os.makedirs(os.path.join(p, "MyDrive"), exist_ok=True)
    drv.mount = mount; colab.drive = drv; google.colab = colab
    return {"google": google, "google.colab": colab, "google.colab.drive": drv}, calls


@contextlib.contextmanager
def patched_modules(mods):
    old = {k: sys.modules.get(k) for k in mods}; sys.modules.update(mods)
    try: yield
    finally:
        for k, v in old.items():
            if v is None: sys.modules.pop(k, None)
            else: sys.modules[k] = v


def test_open_store_mounts_drive_and_falls_back_with_a_plain_warning():
    tmp = tempfile.mkdtemp(); mods, calls = _fake_drive(True); out = io.StringIO()
    with patched_modules(mods), contextlib.redirect_stdout(out):
        st = open_store(os.path.join(tmp, "out"), "axi_14b", os.path.join(tmp, "drive"))
    assert calls == [os.path.join(tmp, "drive")] and st.persistent and st.remote_dir == os.path.join(tmp, "drive", "MyDrive", "axi_14b") and os.path.isdir(st.remote_dir)
    assert "Drive connected" in out.getvalue() and "MyDrive/axi_14b" in out.getvalue()
    mods, calls = _fake_drive(False); out = io.StringIO()
    with patched_modules(mods), contextlib.redirect_stdout(out):
        st = open_store(os.path.join(tmp, "out2"), "axi_14b", os.path.join(tmp, "drive2"))
    assert calls and not st.persistent and "NOT connected" in out.getvalue() and "progress is lost" in out.getvalue() and os.path.isdir(st.local_dir)
    out = io.StringIO()                                                                    # no google.colab at all (not on Colab)
    with patched_modules({"google": None, "google.colab": None, "google.colab.drive": None}), contextlib.redirect_stdout(out):
        st = open_store(os.path.join(tmp, "out3"), "axi_14b", os.path.join(tmp, "drive3"))
    assert not st.persistent and "NOT connected" in out.getvalue()


# ---------------------------------------------------------------- 3: session death and restart
class Counting:
    """Wraps a model and counts how many requests it was asked."""
    def __init__(self, inner): self.inner, self.n = inner, 0

    def generate(self, messages, constrained, max_new_tokens=400):
        self.n += 1; return self.inner.generate(messages, constrained, max_new_tokens)

    @property
    def last_prompt_tokens(self): return self.inner.last_prompt_tokens


ARGS = dict(mode="state", episodes=4, steps=8, seed=7, hint_rules=True)


def _run(llm, path, progress=None):
    return run_long(llm, ARGS["mode"], ARGS["episodes"], ARGS["steps"], ARGS["seed"], ARGS["hint_rules"], path, meta=dict(model="stand-in"), policies=("blind", "repair_m6", "scope_m7"), progress=progress, mix="scope")


def test_a_run_that_dies_in_the_middle_and_is_restarted_from_drive_gives_exactly_the_uninterrupted_result():
    tmp = tempfile.mkdtemp(); wc = Counting(StandIn("rule_like")); whole = _run(wc, os.path.join(tmp, "whole.jsonl")); total = wc.n
    assert len(whole["rows"]) == 4 and total > 8
    for stop in (1, total // 4 + 1, total // 2, total - 1):                              # dies at the first request, in episode 1 or 2, in the middle, one request before the end
        d = os.path.join(tmp, f"s{stop}"); local = os.path.join(d, "out"); drive = os.path.join(d, "drive")
        st = Store(local, drive); seen = []
        def watch(i, n):                                                                   # the progress callback of the notebook: counter, then save; after it the Drive copy holds i episodes
            st.progress("r.jsonl", lambda i, n: seen.append(i))(i, n)
            assert len(read(os.path.join(drive, "r.jsonl")).splitlines()) == i + 1, "Drive copy must be up to date after every finished episode"
        try: _run(StandIn("rule_like", stop_after=stop), st.path("r.jsonl"), watch); raise AssertionError("session must die")
        except RuntimeError as e: assert "session dropped" in str(e)
        assert len(seen) < 4
        shutil.rmtree(local)                                                               # Colab wipes /content
        st2 = Store(local, drive); assert st2.restore("r.jsonl") == ("restored" if seen else "nothing")
        model = Counting(StandIn("rule_like")); res = _run(model, st2.path("r.jsonl"), st2.progress("r.jsonl"))
        assert res["rows"] == whole["rows"] and res["summary"] == whole["summary"] and res["meta"] == whole["meta"], stop
        assert read(os.path.join(drive, "r.jsonl")).splitlines() == read(os.path.join(tmp, "whole.jsonl")).splitlines(), "the Drive file ends up identical to the file of an uninterrupted run"
        if seen: assert 0 < model.n < total, "the second session must not redo the episodes that were saved"
        else: assert model.n == total
    # a restart that finds all episodes saved asks the model nothing at all
    d = os.path.join(tmp, "full"); st = Store(os.path.join(d, "out"), os.path.join(d, "drive")); _run(StandIn("rule_like"), st.path("r.jsonl"), st.progress("r.jsonl"))
    shutil.rmtree(st.local_dir); st2 = Store(st.local_dir, st.remote_dir); assert st2.restore("r.jsonl") == "restored"
    never = Counting(StandIn("rule_like", stop_after=0)); res = _run(never, st2.path("r.jsonl")); assert never.n == 0 and res["rows"] == whole["rows"]


def test_a_progress_file_of_another_run_is_refused_not_mixed_in():
    tmp = tempfile.mkdtemp(); st = Store(os.path.join(tmp, "out"), os.path.join(tmp, "drive"))
    _run(StandIn("rule_like"), st.path("r.jsonl"), st.progress("r.jsonl"))
    shutil.rmtree(st.local_dir); st2 = Store(st.local_dir, st.remote_dir); st2.restore("r.jsonl")
    try: run_long(StandIn("rule_like"), "state", 4, 8, 7, False, st2.path("r.jsonl"), meta=dict(model="stand-in"), policies=("blind", "repair_m6", "scope_m7"), mix="scope"); raise AssertionError("must raise")
    except ValueError as e: assert "different run" in str(e)                               # the no-rule run cannot continue from the rule-stated file
    try: run_long(StandIn("rule_like"), "state", 4, 8, 7, True, st2.path("r.jsonl"), meta=dict(model="another-model"), policies=("blind", "repair_m6", "scope_m7"), mix="scope"); raise AssertionError("must raise")
    except ValueError as e: assert "different run" in str(e)


# ---------------------------------------------------------------- 4: the notebooks
def _code_cells(path):
    nb = json.load(open(path)); return nb, [c for c in nb["cells"] if c["cell_type"] == "code"]


def _src(c): return "".join(c["source"])


def _no_bang(s): return "".join(l for l in s.splitlines(keepends=True) if not l.startswith("!"))


def _calls(tree, name):
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name]


def _run_long_signature(call):
    """Everything about a run_long call except the file it writes and the progress callback."""
    kw = {k.arg: ast.dump(k.value) for k in call.keywords if k.arg not in ("progress",)}
    pos = [ast.dump(a) for i, a in enumerate(call.args) if i != 6]                         # argument 6 is the path
    return pos, kw


def test_the_part_2_notebook_runs_exactly_the_test_of_the_main_notebook():
    nb1, c1 = _code_cells(NB_MAIN); nb2, c2 = _code_cells(NB_P2)
    for nb in (nb1, nb2): assert nb["nbformat"] == 4 and nb["metadata"]["accelerator"] == "GPU"
    for nb in (nb1, nb2):
        for c in nb["cells"]: assert all(l.endswith("\n") for l in c["source"][:-1]), "every line but the last must end with a newline"
    for s in map(_src, c2): compile(_no_bang(s), "cell", "exec")
    t1 = ast.parse("\n".join(_no_bang(_src(c)) for c in c1)); t2 = ast.parse("\n".join(_no_bang(_src(c)) for c in c2))
    long1, long2 = _calls(t1, "run_long"), _calls(t2, "run_long")
    assert len(long1) == 3 and len(long2) == 2                                             # main: smoke + rule + no rule; part 2: rule + no rule
    for a, b in zip(long1[1:], long2): assert _run_long_signature(a) == _run_long_signature(b), "same mode, episodes, steps, seed, rule flag, meta, policies and mix"
    # the two Part-2 calls in the main notebook, spelled out, so a change of the test itself cannot slip through both notebooks together
    for call, (eps, rule) in zip(long2, ((6, True), (4, False))):
        a = call.args
        assert ast.literal_eval(a[1]) == "state" and ast.literal_eval(a[2]) == eps and ast.literal_eval(a[3]) == 30 and ast.literal_eval(a[4]) == 7 and ast.literal_eval(a[5]) == rule
        kw = {k.arg: k.value for k in call.keywords}
        assert ast.literal_eval(kw["policies"]) == ("blind", "repair_m6", "scope_m7") and ast.literal_eval(kw["mix"]) == "scope" and ast.dump(kw["meta"]) == ast.dump(ast.parse("dict(model=MODEL)", mode="eval").body)
    src1, src2 = "\n".join(map(_src, c1)), "\n".join(map(_src, c2))
    assert 'MODEL = "Qwen/Qwen2.5-14B-Instruct"' in src1 and 'MODEL = "Qwen/Qwen2.5-14B-Instruct"' in src2 and "load_4bit=True" in src2 and 'build_hf_llm(MODEL, "cuda", load_4bit=True)' in src2
    for f in ("axi_14b_long_rule", "axi_14b_long_norule"):
        assert f + ".jsonl" in src1 and f + ".json" in src1 and f'name = "{f}"' in src2         # the Part-2 notebook builds the same two file names from `name`
    assert src2.count('name + ".jsonl"') == 6 and src2.count('name + ".json"') == 2 + 2        # restore, run_long path, progress (x2 cells) / save, persist (x2 cells)
    assert "verdict_big_long(res_l_rule[\"summary\"], res_l_norule[\"summary\"])" in src1 and "verdict_big_long(res_l_rule[\"summary\"], res_l_norule[\"summary\"])" in src2
    for p in ("print_long(res_l_rule", "print_exact(res_l_rule", "print_long(res_l_norule", "print_exact(res_l_norule"): assert p in src1 and p in src2
    # the intro says the test is the same and the restart rule
    intro = "".join(nb2["cells"][0]["source"])
    assert "same test" in intro and "Nothing is smaller" in intro and "Cells 1 to 7" in intro and "restored" in intro and "docs/SCALE_TEST_14B.md" in intro
    assert "axi-colab-14b-v2.zip" in "".join(nb2["cells"][3]["source"]) and "open_store" in src2 and "restore(name" in src2 and "store.progress(name" in src2 and "store.persist(name" in src2
    assert 'prog' in src2 and "(saved)" in src2 and "(NOT saved to Drive)" in src2


def _run_part2_notebook(stops, drive_root, work, drive_ok=True):
    """Execute the Part-2 notebook cell by cell with the model, torch, Colab and Drive replaced. `stops` = None or a number of requests after which the stand-in model 'drops the session'.
    Returns (namespace, printed output, names of the files 'downloaded')."""
    nb = json.load(open(NB_P2)); out = io.StringIO()
    class Files:
        downloaded = []
        def upload(self): return {"axi-colab-14b-v2.zip": b""}
        def download(self, p): Files.downloaded.append(os.path.basename(p))
    mods, calls = _fake_drive(drive_ok)
    mods["google.colab"].files = Files()
    torch = types.ModuleType("torch"); torch.cuda = types.SimpleNamespace(memory_allocated=lambda: 9.1e9); mods["torch"] = torch
    cl = types.ModuleType("colab.colab_llm_experiment"); cl.build_hf_llm = lambda name, dev, load_4bit=False: (StandIn("rule_like", stop_after=stops), None); mods["colab.colab_llm_experiment"] = cl
    ns = {}
    with patched_modules(mods), contextlib.redirect_stdout(out):
        for i, c in enumerate(nb["cells"]):
            if c["cell_type"] != "code": continue
            src = _no_bang(_src(c)).replace('open_store("/content/out", "axi_14b")', f'open_store("{work}", "axi_14b", "{drive_root}")')
            src = src.replace('shutil.disk_usage("/content")', f'shutil.disk_usage("{work}")').replace('sys.path.insert(0, "/content/axi/python")', "")
            assert "/content" not in src, src                                                 # nothing may touch the real /content in this test
            exec(compile(src, f"cell{i}", "exec"), ns)
    return ns, out.getvalue(), Files.downloaded


def test_the_part_2_notebook_runs_end_to_end_stops_in_the_middle_and_restarts_from_drive():
    tmp = tempfile.mkdtemp(); drive = os.path.join(tmp, "drive"); saved_dir = os.path.join(drive, "MyDrive", "axi_14b")
    # session 1: the model drops after 150 requests, somewhere in the rule-stated run
    work1 = os.path.join(tmp, "s1"); os.makedirs(work1)
    try: _run_part2_notebook(150, drive, work1); raise AssertionError("session 1 must stop")
    except RuntimeError as e: assert "session dropped" in str(e)
    saved = os.path.join(saved_dir, "axi_14b_long_rule.jsonl"); assert os.path.exists(saved)
    k = len(read(saved).splitlines()) - 1; assert 0 < k < 6, ("some episodes saved, not all", k)
    assert not os.path.exists(os.path.join(saved_dir, "axi_14b_long_norule.jsonl"))
    # session 2: a brand new machine (new work folder); the model works to the end
    work2 = os.path.join(tmp, "s2"); os.makedirs(work2)
    ns, out, downloaded = _run_part2_notebook(None, drive, work2)
    assert "progress file: restored" in out and "Drive connected" in out and "(saved)" in out and "NOT saved" not in out
    assert downloaded == ["axi_14b_long_rule.json", "axi_14b_long_norule.json"]
    assert len(ns["res_l_rule"]["rows"]) == 6 and len(ns["res_l_norule"]["rows"]) == 4 and ns["res_l_rule"]["meta"]["episodes"] == 6 and ns["res_l_norule"]["meta"]["hint_rules"] is False
    assert "PART 2, both regimes" in out and "[guard]" in out and "[need]" in out
    # the result is exactly what one uninterrupted session returns
    work3 = os.path.join(tmp, "s3"); os.makedirs(work3); ns3, out3, _ = _run_part2_notebook(None, os.path.join(tmp, "drive3"), work3)
    assert "progress file: nothing" in out3
    for key in ("res_l_rule", "res_l_norule"): assert ns[key]["rows"] == ns3[key]["rows"] and ns[key]["summary"] == ns3[key]["summary"], key
    for f in ("axi_14b_long_rule.json", "axi_14b_long_norule.json", "axi_14b_long_rule.jsonl", "axi_14b_long_norule.jsonl"): assert os.path.exists(os.path.join(saved_dir, f)), f
    assert read(os.path.join(saved_dir, "axi_14b_long_rule.jsonl")) == read(os.path.join(tmp, "drive3", "MyDrive", "axi_14b", "axi_14b_long_rule.jsonl"))
    # session 3: everything was already finished; the notebook runs through again and the model is never asked
    work4 = os.path.join(tmp, "s4"); os.makedirs(work4); ns4, out4, _ = _run_part2_notebook(0, drive, work4)
    assert ns4["res_l_rule"]["rows"] == ns3["res_l_rule"]["rows"] and ns4["res_l_norule"]["rows"] == ns3["res_l_norule"]["rows"]
    # without Drive the notebook still runs and says, after every episode, that nothing is saved
    work5 = os.path.join(tmp, "s5"); os.makedirs(work5); ns5, out5, _ = _run_part2_notebook(None, os.path.join(tmp, "drive5"), work5, drive_ok=False)
    assert "Drive NOT connected" in out5 and "(NOT saved to Drive)" in out5 and "(saved)" not in out5 and "progress file: nothing" in out5 and not os.path.exists(os.path.join(tmp, "drive5"))
    assert ns5["res_l_rule"]["rows"] == ns3["res_l_rule"]["rows"]


# ---------------------------------------------------------------- 5: the page
def test_the_pre_registration_is_unchanged_and_the_addendum_says_the_test_is_the_same():
    text = read(PAGE); assert ADDENDUM in text and text.count(ADDENDUM) == 1
    head, add = text.split(ADDENDUM)
    assert hashlib.sha256((head + "\n").encode()).hexdigest() == PREREG_SHA256, "the text written before the run must not change"
    result = "\n\n---\n\n# Result"
    if result in add: add = add[:add.index(result)]                                                # the result, once it exists, is appended below the addendum (checked in test_bigger_model_result.py)
    for needle in ("usage limit", "Part 2", "rule stated", "axi_14b_part2.ipynb", "colab_persist.py", "Google Drive", "every finished episode", "same model", "same tasks", "same policies", "same episodes", "same reading rules",
                   "Nothing was made smaller", "restarted session", "not guaranteed", "Part 1 is not run again"):
        assert needle in add, needle
    assert "# Result" not in add
    assert add.count("at most 10 of the 20") == 0 and "[guard]" not in add and "[need]" not in add                    # the addendum adds no new reading rule and no new prediction


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
