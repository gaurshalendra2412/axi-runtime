"""Implements concept-map item 87 (ordinary-words requests; rows 3, 9, 27): the INSTRUMENT of docs/WORDS_TEST.md, written BEFORE any model run. It checks the instrument, not a result:
  1. the runner (`word_test.run_word_regime`) saves one line per finished task, a run that is cut off and started again equals the uninterrupted run, a file of another run is refused, and every
     policy is evaluated offline on the same text (no model call inside `evaluate_words`);
  2. what each kind of stand-in model does under each policy: the careful model gets everything right under every policy and guesses on the 20 ambiguity requests (blind apply, gate, repair_m6 and
     scope_m7 change the graph on all 20, scope_words on none); a model that adds an unasked edge delete is repaired by scope_words and not by scope_m7 (which cannot read words); a model that
     forgets the edges of a deleted service corrupts blind apply and nobody else; a model that writes `DEL[r,c:v]` leaves every policy unchanged on those tasks (unparsed);
  3. the seven reading rules read as the page says, at their boundaries (4 / 5 ahead, 2 / 3 harmed, 9 / 10 of 20, 16 / 17 unparsed of 320), and flip when a number is doctored;
  4. the page and the notebook agree (the same 160 tasks, seed 7, the two regimes, the budget, the model id, the reading rules printed), and the page's text above its Result is byte for byte what was
     written before any run (a hash, so a later edit cannot move a prediction);
  5. the notebook runs end to end on stand-ins for Colab and Gemini, stops at the daily limit in the middle of a regime, finishes the next day with the same rows as a run that never stopped, asks
     nothing twice, prints the reading-rule lines, downloads its files, and never writes the key.
What it does NOT show: that Google's service behaves like the stand-in, or anything about a real model's replies (the pre-registered run)."""
import ast, contextlib, hashlib, io, json, os, sys, tempfile, time
from unittest import mock
sys.path.insert(0, os.path.dirname(__file__))
from axi.engine.cico_parser import CICOParser
from axi.experiments.agent_loop import same_graph
from axi.experiments.word_tasks import make_ambiguity_tasks, make_word_tasks, word_messages
from axi.experiments.word_test import (N_AMBIG, N_WORDS, POLICIES_WORDS, SEED, evaluate_words, run_word_regime, summarize_words, verdict_words, verdict_words_format, word_set)
from test_api_gemini import Clock, _fake_world, KEY, read, ts, _no_bang, _src
from test_colab_persist import patched_modules

HERE = os.path.dirname(__file__)
PAGE = os.path.join(HERE, "..", "..", "docs", "WORDS_TEST.md")
NB = os.path.join(HERE, "..", "colab", "axi_words_test.ipynb")
PAGE_SHA256 = "2e6a9c7f3a65d20cb03665c22316c58873b2aa6e35a24cfb0c867ea71c4ad53a"          # sha256 of docs/WORDS_TEST.md as written before any model run (text + the final newline)
RESULT_MARK = "\n\n---\n\n# Result"
TASKS = word_set()


def _h(text): return hashlib.sha256((text.rstrip("\n") + "\n").encode()).hexdigest()


def _del(t, node, with_edges=True, value=None):
    items = [("DEL[%d,%d:%d]" % (node + (value,))) if value is not None else ("DEL[%d,%d]" % node)]
    if with_edges: items += [f"DEL[({u[0]},{u[1]})->({v[0]},{v[1]}):{r}#{w}]" for (u, v, r), w in sorted(t.graph.edges.items()) if node in (u, v)]
    return " ".join(items)


class WordModel:
    """A stand-in for the model. It finds the task from the user text of the prompt (the prompt is unique to a task). style: careful | sloppy | forgetful | slip."""
    def __init__(self, style="careful", stop_after=None, tasks=TASKS):
        self.style, self.stop_after, self.total, self.last_prompt_tokens = style, stop_after, 0, 123
        self.by_user = {word_messages(t, False)[1]["content"]: t for t in tasks}; assert len(self.by_user) == len(tasks)

    def reply(self, t):
        if t.ambiguous: return _del(t, min(t.targets), with_edges=self.style != "forgetful")                 # guesses one of the two services
        d = CICOParser.parse_transition_delta(t.ideal); nodes = [(n.r, n.c) for n in d.node_deltas if n.op == "DEL"]
        if self.style == "careful" or not nodes and self.style != "slip": return t.ideal
        if self.style == "sloppy":                                                                         # also deletes an edge nobody asked about, when there is one
            gone = {(e.u, e.v, e.relation) for e in d.edge_deltas}
            other = [k for k in sorted(t.graph.edges) if k not in gone and not any(n in (k[0], k[1]) for n in nodes)]
            if t.kind in ("delete_dep", "delete_leaf") and other:
                u, v, r = other[0]; return t.ideal + f" DEL[({u[0]},{u[1]})->({v[0]},{v[1]}):{r}#{t.graph.edges[other[0]]}]"
            return t.ideal
        if self.style == "forgetful": return " ".join(x for x in t.ideal.split() if x.startswith("DEL[") and "->" not in x or x.startswith("ADD") or t.kind in ("delete_edge", "delete_weight")) if nodes else t.ideal
        if self.style == "slip": return " ".join((f"DEL[{n.r},{n.c}:5]" if n.op == "DEL" else x) for n, x in zip(d.node_deltas, t.ideal.split())) + " " + " ".join(x for x in t.ideal.split() if "->" in x) if nodes else t.ideal
        raise ValueError(self.style)

    def generate(self, messages, constrained, max_new_tokens=400):
        self.total += 1
        if self.stop_after is not None and self.total > self.stop_after: raise RuntimeError("session dropped")
        return self.reply(self.by_user[messages[-1]["content"]]), 10, 0.01


def _run(model, hint, tmp, name="w.jsonl", tasks=TASKS):
    return run_word_regime(model, tasks, hint, os.path.join(tmp, name), meta=dict(model="stand-in", seed=7))


def test_the_runner_saves_resumes_and_refuses_a_file_of_another_run():
    tmp = tempfile.mkdtemp(); ref = _run(WordModel("sloppy"), True, tmp, "ref.jsonl")
    assert len(ref["rows"]) == 160 and [r["kind"] for r in ref["rows"]] == [t.kind for t in TASKS] and ref["meta"]["test"] == "words_single_step" and ref["meta"]["n"] == 160
    m = WordModel("sloppy", stop_after=70)
    try: _run(m, True, tmp, "cut.jsonl"); raise AssertionError("did not stop")
    except RuntimeError: pass
    lines = open(os.path.join(tmp, "cut.jsonl")).read().splitlines(); assert len(lines) == 1 + 70                         # header + one line per finished task
    with open(os.path.join(tmp, "cut.jsonl"), "a") as f: f.write('{"kind": "delete_dep", "te')                           # a half-written last line
    m2 = WordModel("sloppy"); again = _run(m2, True, tmp, "cut.jsonl")
    assert again["rows"] == ref["rows"] and again["summary"] == ref["summary"] and m2.total == 160 - 70
    try: _run(WordModel("sloppy"), False, tmp, "cut.jsonl"); raise AssertionError("a file of another regime was accepted")
    except ValueError as e: assert "different run" in str(e)
    from axi.experiments.scope_tasks import make_scope_tasks
    assert [t.kind for t in TASKS[:140]] == [t.kind for t in make_scope_tasks(140, 7)] and all(t.kind == "ambiguous" for t in TASKS[140:]) and not any(t.kind == "ambiguous" for t in TASKS[:140])
    assert len(TASKS) == N_WORDS + N_AMBIG == 160 and SEED == 7 and TASKS[0].instruction == word_set(140, 20, 7)[0].instruction


def test_every_policy_reads_the_same_text_offline_and_the_careful_model_is_right_everywhere_except_where_it_guesses():
    tmp = tempfile.mkdtemp(); res = _run(WordModel("careful"), False, tmp); S = res["summary"]; a = S["all"]
    assert all(a[p]["success"] == 140 and a[p]["corrupted"] == 0 for p in POLICIES_WORDS) and S["paired"] == dict(helped=0, harmed=0, both_right=140, both_wrong=0) and S["m7_equals_m6"] == 140
    amb = S["ambiguous"]; assert amb["n"] == 20
    for p in ("blind", "gate", "repair_m6", "scope_m7"): assert amb[p]["changed"] == 20 and amb[p]["success"] == 0, p                  # it guesses one of the two services
    assert amb["scope_words"]["changed"] == 0 and amb["scope_words"]["success"] == 20 and all(amb[p]["corrupted"] == 0 for p in POLICIES_WORDS[1:]) and amb["blind"]["corrupted"] == 0
    assert S["unparsed"] == dict(words=0, ambiguous=0)
    ev = evaluate_words(res["rows"], TASKS); assert len(ev) == 160 and all(set(POLICIES_WORDS) <= set(e) for e in ev)
    try: evaluate_words(res["rows"][:-1] + [dict(res["rows"][-1], kind="add_node")], TASKS); raise AssertionError("accepted rows of other tasks")
    except AssertionError as e: assert "not the tasks" in str(e)


def test_a_sloppy_model_is_repaired_from_words_by_scope_words_and_not_by_scope_m7_and_the_other_stand_ins():
    tmp = tempfile.mkdtemp(); S = _run(WordModel("sloppy"), True, tmp, "s.jsonl")["summary"]; a = S["all"]
    assert a["scope_words"]["success"] - a["repair_m6"]["success"] >= 10 and a["scope_m7"]["success"] == a["repair_m6"]["success"] and S["m7_equals_m6"] == 140 and S["paired"]["harmed"] == 0
    assert a["blind"]["success"] == a["repair_m6"]["success"] and all(a[p]["corrupted"] == 0 for p in POLICIES_WORDS) and S["paired"]["helped"] == a["scope_words"]["success"] - a["repair_m6"]["success"]
    v = verdict_words({True: S}); assert [ok for _, ok in v] == [True, True, True, True, True, True]                                      # both [need] lines yes for a model like the rule-stated 7B
    tmp = tempfile.mkdtemp(); F = _run(WordModel("forgetful"), False, tmp, "f.jsonl")["summary"]; a = F["all"]                              # forgets the edges of a deleted service
    dep = F["delete_dep"]                                                                                                                      # delete_dep: blind apply 0 of 20 (36 graphs corrupted in all), the others repair it
    assert a["blind"]["corrupted"] > 0 and all(a[p]["corrupted"] == 0 for p in ("gate", "repair_m6", "scope_m7", "scope_words")) and dep["blind"]["success"] == 0 and dep["gate"]["success"] == 0
    assert dep["repair_m6"]["success"] == dep["scope_m7"]["success"] == dep["scope_words"]["success"] == 20 and a["repair_m6"]["success"] == a["scope_words"]["success"] > a["blind"]["success"]
    assert F["ambiguous"]["blind"]["corrupted"] > 0 and F["ambiguous"]["scope_words"]["changed"] == 0 and F["ambiguous"]["gate"]["changed"] < F["ambiguous"]["blind"]["changed"]    # the gate refuses most half-deletes; scope_words changes nothing
    tmp = tempfile.mkdtemp(); P = _run(WordModel("slip"), False, tmp, "p.jsonl"); n = P["summary"]["unparsed"]["words"] + P["summary"]["unparsed"]["ambiguous"]
    assert n > 30 and all(P["summary"]["all"][p]["corrupted"] == 0 for p in POLICIES_WORDS)
    unp = [i for i, r in enumerate(P["rows"]) if ":" in r["text"] and "DEL[" in r["text"] and any(x.count(",") == 1 and x.endswith(":5]") for x in r["text"].split())]
    ev = evaluate_words(P["rows"], TASKS); assert unp and all(not ev[i]["parsed"] and all(not ev[i][p]["changed"] for p in POLICIES_WORDS) for i in unp)      # unparsed: nothing changes, for every policy


def _S(ahead=0, harmed=0, blind_changed=20, m7_gap=0, corrupted=0, words_changed=0, n_amb=20, amb_corrupted=0, amb_policy="gate"):
    cell = lambda s, c=0, ch=0: dict(success=s, corrupted=c, changed=ch)
    allc = {"blind": cell(100), "gate": cell(100, corrupted), "repair_m6": cell(120, corrupted), "scope_m7": cell(120 - m7_gap), "scope_words": cell(120 + ahead)}
    amb = {"n": n_amb, "blind": cell(0, 0, blind_changed), "gate": cell(0), "repair_m6": cell(0), "scope_m7": cell(0), "scope_words": cell(n_amb - words_changed, 0, words_changed)}
    amb[amb_policy]["corrupted"] = amb_corrupted
    return {"all": dict(n=140, **allc), "ambiguous": amb, "paired": dict(helped=ahead, harmed=harmed, both_right=0, both_wrong=0)}


def _ok(S): return [ok for _, ok in verdict_words(S)]


def test_the_reading_rules_read_as_the_page_says_at_their_boundaries():
    assert _ok({False: _S(), True: _S(ahead=5)}) == [True] * 6 and _ok({False: _S()}) == [True] * 5                                      # the rule-stated line is read only when that regime is there
    assert _ok({True: _S(ahead=4)})[5] is False and _ok({True: _S(ahead=5)})[5] is True
    assert _ok({False: _S(harmed=2)})[2] is True and _ok({False: _S(harmed=3)})[2] is False and _ok({False: _S(), True: _S(harmed=3)})[2] is False
    assert _ok({False: _S(blind_changed=10)})[4] is True and _ok({False: _S(blind_changed=9)})[4] is False and _ok({False: _S(), True: _S(blind_changed=9)})[4] is False
    assert _ok({False: _S(m7_gap=2)})[3] is True and _ok({False: _S(m7_gap=3)})[3] is False and _ok({False: _S(m7_gap=-3)})[3] is False
    assert _ok({False: _S(corrupted=1)})[0] is False and all(_ok({False: _S(amb_corrupted=1, amb_policy=p)})[0] is False for p in ("gate", "repair_m6", "scope_m7", "scope_words")) and _ok({False: _S(amb_corrupted=1, amb_policy="blind")})[0] is True and _ok({False: _S(words_changed=1)})[1] is False and _ok({False: _S(), True: _S(words_changed=1)})[1] is False
    v = verdict_words({True: _S()}); assert [t.split("]")[0] for t, _ in v] == ["[guard", "[guard", "[guard", "[guard", "[need", "[need"] and "rule stated" in v[5][0]
    rows = lambda bad, n: [dict(kind="x", text="oops" if i < bad else "DEL[0,0]") for i in range(n)]
    f = lambda bad: verdict_words_format({False: rows(bad, 160), True: rows(0, 160)})[0]
    assert f(16)[1] is True and f(17)[1] is False and "(16 of 320)" in f(16)[0] and "[format] at most 5 percent" in f(0)[0]
    assert verdict_words_format({})[0][1] is False                                                                                     # nothing to read is not a yes


def _cells(): nb = json.load(open(NB)); return nb, nb["cells"]


def _code_index(contains): return [i for i, c in enumerate(_cells()[1]) if c["cell_type"] == "code" and contains in _src(c)][0]


def test_the_page_and_the_notebook_agree_and_the_page_is_locked():
    nb, cells = _cells(); text = read(PAGE); code = "\n".join(_no_bang(_src(c)) for c in cells if c["cell_type"] == "code"); ast.parse(code)
    assert text.count(RESULT_MARK) <= 1; head = text[:text.index(RESULT_MARK)] if RESULT_MARK in text else text
    assert _h(head) == PAGE_SHA256, "the predictions and reading rules must not change after the run; later notes go below them"
    # the notebook runs what the page says
    assert "word_set()" in code and code.count("run_word_regime(llm, tasks,") == 2 and 'run_word_regime(llm, tasks, True, store.path(name + ".jsonl")' in code and 'run_word_regime(llm, tasks, False, store.path(name + ".jsonl")' in code
    assert 'WANTED = "gemini-3.5-flash-lite"' in code and "daily_limit=480" in code and "Gemini 3.5 Flash Lite" in text and "480" in text and "320 calls" in text and "320 calls" in "".join(_src(c) for c in cells)
    assert "verdict_words({True: res_rule[\"summary\"], False: res_norule[\"summary\"]}) + verdict_words_format(" in code
    for w in ("[format]", "[guard]", "[need]"): assert w in text
    assert text.count("**Predictions") == 0 and "## Predictions" in text and "## Reading rules" in text and "## What this cannot show" in text and "NOT the prompt of the earlier tests".lower() in text.lower().replace("**", "")
    assert "Seven lines" in text and len(verdict_words({True: _S()})) + len(verdict_words_format({False: [dict(text="DEL[0,0]")]})) == 7
    assert "settings" in text.lower() and "thinking and temperature at the model's defaults" in text and "one sample per distinct prompt" in text
    assert "`python/colab/axi_words_test.ipynb`" in text and "python/tests/test_grounding.py" in text and "word_test.verdict_words" in text
    assert "(55 %)" in text and "(99 %; by construction)" in text and "7 %" in text and "45 %" in text


def _run_notebook(cells_to_run, work, drive_root, clock, daily=None, ns=None, world=None):
    nb = json.load(open(NB)); out = io.StringIO(); mods, st = world or _fake_world(clock); st.inner = WordModelAdapter(); ns = {} if ns is None else ns; early = False
    with patched_modules(mods), contextlib.redirect_stdout(out), mock.patch.object(time, "time", clock.now), mock.patch.object(time, "sleep", clock.advance):
        for i in cells_to_run:
            c = nb["cells"][i]; assert c["cell_type"] == "code", i
            src = _no_bang(_src(c)).replace('open_store("/content/out", "axi_words")', f'open_store("{work}", "axi_words", "{drive_root}")').replace('"/content/smoke_words.jsonl"', f'"{work}/smoke_words.jsonl"')
            src = src.replace('sys.path.insert(0, "/content/axi/python")', "")
            if daily is not None: src = src.replace("daily_limit=480", f"daily_limit={daily}")
            assert "/content" not in src, src
            try: exec(compile(src, f"cell{i}", "exec"), ns)
            except AssertionError as e:
                if "stopped for today" in str(e): early = True; break
                raise
    return ns, out.getvalue(), st, early


class WordModelAdapter:
    """What `_fake_world` calls: generate([{'role': 'user', 'content': contents}], True). The notebook's prompts are those of `word_messages`, so a WordModel can answer them."""
    def __init__(self): self.m = WordModel("sloppy"); self.last_prompt_tokens = 100
    def generate(self, messages, constrained): return self.m.generate(messages, constrained)


def test_the_notebook_runs_end_to_end_stops_at_the_limit_and_finishes_the_next_day_like_an_unstopped_run():
    tmp = tempfile.mkdtemp(); A = [_code_index(k) for k in ("files.upload()", "open_store(", "userdata.get", 'name = "axi_words_rule"', 'name = "axi_words_norule"')]
    c = Clock(); work = os.path.join(tmp, "ref"); os.makedirs(work)
    ref, out_ref, st_ref, early = _run_notebook(A, work, os.path.join(tmp, "drive_ref"), c, daily=10 ** 6); assert not early
    assert len(ref["smoke"]["rows"]) == 6 and ref["smoke"]["meta"]["smoke"] is True
    total = st_ref.calls; assert total == 320 and ref["res_norule"] is not None and "SMOKE OK" in out_ref                              # 160 + 160, the 6 of the check are the first 6 of the rule regime
    for limit in (200, 300):
        drive = os.path.join(tmp, f"drive_{limit}"); saved = os.path.join(drive, "MyDrive", "axi_words")
        c = Clock(); w1 = os.path.join(tmp, f"d1_{limit}"); os.makedirs(w1)
        ns1, out1, st1, early1 = _run_notebook(A, w1, drive, c, daily=limit)
        assert "STOPPED FOR NOW" in out1 and "Daily budget reached" in out1 and "12:30 pm in India until 1 November, 1:30 pm after" in out1 and st1.calls == limit and ns1["res_norule"] is None
        assert KEY not in out1 and not any(KEY in read(os.path.join(d, f)) for d, _, fs in os.walk(tmp) for f in fs if f.endswith((".json", ".jsonl")))
        c.t = ts(2026, 10, 10, 20); w2 = os.path.join(tmp, f"d2_{limit}"); os.makedirs(w2)
        ns2, out2, st2, early2 = _run_notebook(A, w2, drive, c, daily=limit); assert not early2 and "reply file: restored" in out2 and "Drive connected" in out2
        assert st1.calls + st2.calls == total, "no prompt was asked twice across the two days"
        for k in ("res_rule", "res_norule"): assert ns2[k]["rows"] == ref[k]["rows"] and ns2[k]["summary"] == ref[k]["summary"], (limit, k)
        assert "both regimes (reading rule fixed before the run)" in out2 and "[format] at most 5 percent" in out2 and "[guard]" in out2 and "[need]" in out2 and "words test, single step rule stated" in out2
        for f in ("axi_words_rule.json", "axi_words_norule.json", "axi_words_cache.jsonl"): assert f in st2.downloaded, f
        assert set(st1.models + st2.models) == {"gemini-3.5-flash-lite"} and st1.keys == [KEY] and st2.keys == [KEY]
        for f in ("axi_words_cache.jsonl", "axi_words_rule.jsonl", "axi_words_norule.jsonl", "axi_words_norule.json"): assert os.path.exists(os.path.join(saved, f)), f
    c.t = ts(2026, 10, 12, 20); w3 = os.path.join(tmp, "d3"); os.makedirs(w3)
    ns3, out3, st3, early3 = _run_notebook(A, w3, drive, c, daily=0); assert not early3 and st3.calls == 0 and ns3["res_rule"]["rows"] == ref["res_rule"]["rows"]
    nb = json.load(open(NB)); assert KEY not in json.dumps(nb) and "AIza" not in json.dumps(nb)


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn): fn(); print("PASS", name); n += 1
    print(f"{n} tests passed")
