"""Engineering support for docs/PROVIDER_SIDE_FREE_TEST.md (concept-map item 113): the three cells of the provider-side serving test with requests in ordinary words. Checks that need no GPU and no model:
  1. the pins: the files this test imports and edits none of, the settings of the earlier runs, the jobs (the sixth run's 192 to 255, read from its saved file) and the sixth run's figures the page quotes;
  2. the names: the host's table follows the oracle's graph, a service keeps its name while it exists, a new service takes an unused name, aliases only where the pool has them;
  3. the wordings: all 6,400 sentences say what the oracle's request says (checked by a plain search that does not use the reader), every template is used, the second pool is a quarter, and the check itself
     is made to read NO when a sentence is changed;
  4. the host reader: sentences by hand (each outcome, each reason, the direction of an edge, case), and the 6,400 requests (5,380 right, 0 wrong, 1,020 not read, by kind and pool);
  5. the session: the prompt carries the names and the request in words or with coordinates, the expression of each cell, the log flags, the harness on a test double that obeys the grammar (a slip the
     capped expression allows is written, a slip the named expression forbids is not);
  6. the 16 lines on a good run built on the double, each line at its edge (with the lines that must move with it), 'not run' for a missing cell, the rule about what may be claimed, the printout;
  7. the pre-flight check: each of its checks can read NO;
  8. the locked page, the notebook and its cells, the front pages.
What it does NOT show: anything about a real engine, a real GPU, or what a model writes when it reads a name."""
import contextlib, copy, glob, hashlib, io, json, math, os, re, subprocess, sys, types

sys.path.insert(0, os.path.dirname(__file__))
import axi.engine.grounding as GR
import axi.experiments.provider_side as PS
import axi.experiments.provider_side_long as L
import axi.experiments.provider_side_typed as T
import axi.experiments.provider_side_free as F
import axi.experiments.word_tasks as WT

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..", "..")
PAGE = os.path.join(ROOT, "docs", "PROVIDER_SIDE_FREE_TEST.md")
RES = os.path.join(ROOT, "docs", "results")
LONG_FILE = os.path.join(RES, "axi_provider_long.json")
NB = os.path.join(ROOT, "python", "colab", "axi_provider_side_free.ipynb")
MODULE = os.path.join(HERE, "..", "axi", "experiments", "provider_side_free.py")
RESULT_MARK = "\n\n# Result"
PAGE_SHA256 = "8b53170cbd095425c5adacc18992b472e156a265479b34a8d60a78d91129fa4d"                  # the locked head
SEED = 20261017
PINS = dict(provider_side="bfb4734e171d9244cd47d10402be46528a557095a8275d02b38481a0e2b0b1d3", constrained="03452148e28c9f31372f12cc11ef1808ac67308adff972fa19d4aeeef7c96f7d",
            bounded="7427a686d151c4294809aebaa9664d0b5fa420854817eb79e910372c3f57b6bc", gate="b6a50408788791f76bc4d4129dad84debb6e09c9c5e804673cdf04abec224ce7",
            diagnostics="255f3c11db989b645077d10706a187e2c913576538a826e6dbdb8892b014b4f2", agent_loop="d658afaa29e3bce7e1fb7f43d88281fc40ee68ca59864421a65284e9973516b6",
            cascade="b0533316bc094e80a1d68eddb7e137c7c176af9a342341288360c18d61f74fe7", long_horizon="5718435de2178de110ddbb59dd3bf32e1bec9f1fd8625a27119d789399fa20ac")


def seed_of(job): return SEED * 1000 + job
def _sha(path): return hashlib.sha256(open(path, "rb").read()).hexdigest()
def _h(text): return hashlib.sha256((text.rstrip("\n") + "\n").encode()).hexdigest()
def _head():
    t = open(PAGE).read(); assert t.count(RESULT_MARK) <= 1
    return t[:t.index(RESULT_MARK)] if RESULT_MARK in t else t
JOBS = range(192, 256)
_CACHE = {}


# ---------------------------------------------------------------- 1 the pins and the jobs
def test_the_files_the_test_imports_are_pinned_the_settings_are_the_sixth_runs_and_the_jobs_and_figures_are_read_from_its_saved_file():
    assert T.pinned_hashes() == PINS == L.PINS == F.PINS and T.fingerprint() == F.TYPED_MODULE_SHA256 == L.TYPED_MODULE_SHA256 and F.pinned_files()["provider_side_long"] == F.LONG_MODULE_SHA256 == _sha(L.__file__)
    assert F.pinned_files()["grounding"] == F.GROUNDING_SHA256 == _sha(GR.__file__) and F.pinned_files()["word_tasks"] == F.WORD_TASKS_SHA256 == _sha(WT.__file__), "grounding.py and word_tasks.py are imported and edited by nothing"
    assert (F.MODEL, F.SEED, F.POLICY, F.HINT_RULES, F.MAX_NEW, F.CAPACITY_TOKENS, F.STEPS, F.FIRST, F.K_JOBS) == ("Qwen/Qwen2.5-7B-Instruct-AWQ", SEED, "repair_m6", False, 96, 80000, 100, 192, 64)
    assert F.REGEX_SHA256 == hashlib.sha256(F.REGEX.encode()).hexdigest() == "6f4e5a993f26ccd471bff13422fa622b7820858a91e05d43c2bccfbe6b864245"
    assert F.ORDER == ["C0", "W0", "WN"] and F.CELLS == {"C0": (64, 192, "S", "coords", "capped"), "W0": (64, 192, "S", "words", "capped"), "WN": (64, 192, "S", "words", "read")}
    assert _sha(LONG_FILE) == F.LONG_FILE_SHA256 == "a0ad579961d006e9d561ff65cb301de77dd25dd85ebb348827201ff0fdaeba2b"
    d = json.load(open(LONG_FILE)); c = {x["label"]: x for x in d["cells"]}; L0 = [x for s in c["L0"]["sessions"] for x in s["log"]]
    assert (sum(1 for x in L0 if x["exact"]), len(L0)) == (F.L0_EXACT, F.L0_STEPS) == (6244, 6400) and {s["seed"] - SEED * 1000 for s in c["L0"]["sessions"]} == set(JOBS) == {s["seed"] - SEED * 1000 for s in c["LN"]["sessions"]}
    assert d["pinned_sha256"] == PINS and all(len(T.script_of(seed_of(e), 100)) == 100 for e in JOBS)
    n = {"add_node": 0, "add_edge": 0, "delete": 0}
    for e in JOBS:
        for st in T.script_of(seed_of(e), 100): n[T.parse_request(st["instruction"])["kind"]] += 1
    assert n == {"add_node": 2158, "add_edge": 2059, "delete": 2183}, n
    assert {lab: F.grammar_sha256(lab) for lab in F.ORDER} == F.GRAMMAR_SHA and len(F.expressions_of("C0")) == len(F.expressions_of("W0")) == 1 and len(F.expressions_of("WN")) == 775 and F.REGEX in F.expressions_of("WN")


# ---------------------------------------------------------------- 2 the names
def test_the_hosts_table_follows_the_oracles_graph_a_service_keeps_its_name_and_a_new_one_takes_an_unused_name():
    pool = set(WT.POOL); aliased = 0; seen_new = 0
    for e in JOBS:
        seed = seed_of(e); tables, news = F.name_tables(seed, 100); g0, script, oracle = F.oracle_script(seed, 100); assert len(tables) == len(news) == 100
        assert F.name_tables(seed, 100) is F.name_tables(seed, 100)
        for t in range(100):
            tb = tables[t]; assert set(tb) == set(oracle[t].nodes), (e, t)                                 # the table is the oracle's graph's, not the held state's
            assert len(set(tb.values())) == len(tb) and set(tb.values()) <= pool and GR.NameIndex(tb, F.aliases_of(tb)) is not None
            assert all(set(a) <= set(WT.ALIASES.get(tb[c], [])) for c, a in F.aliases_of(tb).items())
            aliased += len(F.aliases_of(tb))
            if t + 1 < 100:
                nxt = tables[t + 1]; kind = script[t]["kind"]; p = T.parse_request(script[t]["instruction"])
                if kind == "add_node":
                    node = (int(p["node"][0]), int(p["node"][1])); seen_new += 1
                    assert news[t] is not None and news[t] not in tb.values() and node not in tb and nxt[node] == news[t] and {k: v for k, v in nxt.items() if k != node} == tb
                elif kind in ("delete_dep", "delete_leaf"):
                    node = (int(p["node"][0]), int(p["node"][1])); assert news[t] is None and node in tb and node not in nxt and {k: v for k, v in tb.items() if k != node} == nxt
                else: assert news[t] is None and nxt == tb
    assert aliased > 1000 and seen_new == 2158 - sum(1 for e in JOBS if T.script_of(seed_of(e), 100)[-1]["kind"] == "add_node")
    assert F.name_tables(seed_of(192), 100)[0] == F.name_tables(seed_of(192), 100)[0] and F.name_tables(seed_of(192), 100)[0] != F.name_tables(seed_of(193), 100)[0]
    assert set(F.NOTE.split("\n")) >= {'  remove the service called "demo", which sits at (2,1) -> DEL[2,1]'} and F.NOTE == WT.WORDS_NOTE


# ---------------------------------------------------------------- 3 the wordings
def _all_requests():
    if "req" not in _CACHE: _CACHE["req"] = [(e, i, r) for e in JOBS for i, r in enumerate(F.requests_of(seed_of(e), 100))]
    return _CACHE["req"]


def test_all_6400_sentences_say_what_the_oracles_request_says_every_template_is_used_and_the_second_pool_is_a_quarter():
    pools = {"A": 0, "B": 0}; used = {}; alias_mentions = alias_possible = 0
    for e, i, (ins, w, pool, tb, rd, why, oc) in _all_requests():
        news = F.name_tables(seed_of(e), 100)[1]; assert F.wording_ok(ins, w, pool, tb, news[i]), (e, i, w, ins)
        assert F.wording(seed_of(e), i, ins, tb, news[i]) == (w, pool) and pool in pools; pools[pool] += 1
        p = T.parse_request(ins); k = {"delete": (F.A_DELETE, F.B_DELETE), "add_node": (F.A_NODE, F.B_NODE), "add_edge": (F.A_EDGE, F.B_EDGE)}[p["kind"]][0 if pool == "A" else 1]
        for tpl in k:
            if re.fullmatch(re.escape(tpl).replace(r"\{n\}", ".+").replace(r"\{m\}", ".+").replace(r"\{x\}", ".+").replace(r"\{a\}", ".+").replace(r"\{b\}", ".+").replace(r"\{rel\}", ".+").replace(r"\{w\}", ".+").replace(r"\{r\}", ".+").replace(r"\{c\}", ".+").replace(r"\{v\}", ".+"), w): used[tpl] = used.get(tpl, 0) + 1
        for c in ([(int(p["node"][0]), int(p["node"][1]))] if p["kind"] == "delete" else [(int(p["src"][0]), int(p["src"][1])), (int(p["dst"][0]), int(p["dst"][1]))] if p["kind"] == "add_edge" else []):
            if c in F.aliases_of(tb): alias_possible += 1; alias_mentions += any(re.search(r"(?<![A-Za-z])" + a + r"(?![A-Za-z])", w, re.IGNORECASE) for a in F.aliases_of(tb)[c])
    assert pools == {"A": 4852, "B": 1548} and 0.23 < pools["B"] / 6400 < 0.26
    every = F.A_DELETE + F.B_DELETE + F.A_NODE + F.B_NODE + F.A_EDGE + F.B_EDGE; assert len(every) == 6 + 2 + 4 + 1 + 5 + 2 and set(used) == set(every) and all(n > 50 for n in used.values()), used
    assert alias_possible > 800 and 0.15 < alias_mentions / alias_possible < 0.25, (alias_mentions, alias_possible)


def test_the_check_of_the_sentences_is_made_to_read_no_when_a_sentence_is_changed():
    e = 192; seed = seed_of(e); tables, news = F.name_tables(seed, 100)
    for i, (ins, w, pool, tb, rd, why, oc) in enumerate(F.requests_of(seed, 100)):
        p = T.parse_request(ins)
        if p["kind"] == "add_edge" and pool == "A":
            a, b = tb[(int(p["src"][0]), int(p["src"][1]))], tb[(int(p["dst"][0]), int(p["dst"][1]))]
            if a in w and b in w and w.index(a) < w.index(b):
                good = w; assert F.wording_ok(ins, good, pool, tb, news[i])
                sw = good.replace(a, "@@").replace(b, a).replace("@@", b)
                assert not F.wording_ok(ins, sw, pool, tb, news[i]), "source and target swapped"
                assert not F.wording_ok(ins, re.sub(r"weight (-?\d+)", "weight 77", good, flags=re.I) if re.search(r"weight (-?\d+)", good, re.I) else good + " 7", pool, tb, news[i]), "another number"
                assert not F.wording_ok(ins, good.replace(a, "the xyz"), pool, tb, news[i]), "a name missing"
                other = next(n for c, n in tb.items() if n not in (a, b)); assert not F.wording_ok(ins, good + f" The {other} service is involved.", pool, tb, news[i]), "an extra service in the first pool"
                assert not F.wording_ok(ins, good, "B", tb, news[i]), "a first-pool sentence marked second pool"
                assert not F.wording_ok(ins, good.replace(f"relation {p['rel']}", "relation other"), pool, tb, news[i]), "another relation"
                break
    for i, (ins, w, pool, tb, rd, why, oc) in enumerate(F.requests_of(seed, 100)):
        if T.parse_request(ins)["kind"] == "add_node" and pool == "A":
            assert not F.wording_ok(ins, w.replace(news[i], "zzz"), pool, tb, news[i]), "the new name missing"
            assert not F.wording_ok(ins, re.sub(r"\(\d+,", "(9,", w), pool, tb, news[i]), "another coordinate"
            break
    for i, (ins, w, pool, tb, rd, why, oc) in enumerate(F.requests_of(seed, 100)):
        if T.parse_request(ins)["kind"] == "delete" and pool == "B":
            assert F.wording_ok(ins, w, pool, tb, news[i]) and not F.wording_ok(ins, w, "A", tb, news[i]), "a second-pool delete read as the first pool"
            assert not F.wording_ok(ins, "Remove it.", pool, tb, news[i]); break


# ---------------------------------------------------------------- 4 the host reader
def _table():
    t = {(0, 0): "billing", (0, 1): "auth", (1, 0): "payments", (1, 1): "user directory", (2, 2): "email"}; return t


def test_the_reader_by_hand_each_outcome_each_reason_the_direction_of_an_edge_and_case():
    tb = _table(); r = lambda s: F.read_request(s, tb)
    assert r("Remove the billing service.") == ("Delete the service at (0,0).", None)
    assert r("Please retire the user directory service.") == ("Delete the service at (1,1).", None) and r("Take the cache service out.")[0] is None, "cache is not in this table"
    assert r("Drop mailer from the graph.")[0] == "Delete the service at (2,2).", "mailer is an alias of email"
    assert r("Remove the billing service; the auth service stays.") == (None, "delete: not one service")
    assert r("Remove the cache service.") == (None, "delete: unresolved") and r("Remove every edge.")[0] is None
    assert r("Add a dependency edge from the billing service to the auth service with relation dep and weight 4.") == ("Add a dependency edge from (0,0) to (0,1) with relation dep and weight 4.", None)
    assert r("Add a dependency edge from the auth service to the billing service with relation dep and weight 4.")[0] == "Add a dependency edge from (0,1) to (0,0) with relation dep and weight 4."
    assert r("Please add an edge from billing to payments. Relation owns, weight -3.")[0] == "Add a dependency edge from (0,0) to (1,0) with relation owns and weight -3.", "case ignored, a negative weight"
    assert r("Wire payments -> user directory (relation dep, weight 2).")[0] == "Add a dependency edge from (1,0) to (1,1) with relation dep and weight 2."
    assert r("Link billing to auth using relation dep and weight 4; the email service is not involved.") == (None, "edge: not two services")
    assert r("Link billing to billing using relation dep and weight 4.") == (None, "edge: not two services"), "two mentions of one service are one service"
    assert r("Add a new service called inventory at (3,3) with value 5.") == ("Add a service at (3,3) with value 5.", None)
    assert r("Add inventory at (3,3) with value 5; it will sit next to the billing service.")[0] == "Add a service at (3,3) with value 5."
    assert r("Add inventory at (3,3) and (4,4) with value 5.") == (None, "node: not one pair of coordinates") and r("Add inventory with value 5.") == (None, "node: not one pair of coordinates")
    assert r("Add inventory at (3,3) with value 5 and weight 2.") == (None, "kind not clear") and r("Link billing to auth with weight 4.") == (None, "kind not clear")
    tb3 = {(0, 0): "billing api", (0, 1): "billing worker", (1, 0): "email"}
    assert F.read_request("Remove the billing service.", tb3) == (None, "ambiguous") and F.read_request("Remove the billing api service.", tb3) == ("Delete the service at (0,0).", None)
    assert F.read_request("Remove the worker service.", tb3)[0] == "Delete the service at (0,1).", "a leftover word that is part of exactly one name is bound (grounding.py's rule)"


def test_the_reader_over_the_6400_requests_reads_none_wrong_and_the_outcomes_are_those_on_the_page():
    by = {}
    for e, i, (ins, w, pool, tb, rd, why, oc) in _all_requests():
        k = T.parse_request(ins)["kind"]; by[(k, pool, oc)] = by.get((k, pool, oc), 0) + 1
        assert oc == F.reader_outcome(ins, rd) and (rd is None) == (oc == "miss") and (why is None) == (rd is not None)
        if rd is not None: assert T.parse_request(rd) == T.parse_request(ins), (w, rd, ins)
    assert by == {("add_edge", "A", "right"): 1585, ("add_edge", "B", "miss"): 474, ("add_node", "A", "right"): 1630, ("add_node", "B", "right"): 528, ("delete", "A", "right"): 1637, ("delete", "B", "miss"): 546}, by
    tot = {k: sum(v for (a, b, c), v in by.items() if c == k) for k in ("right", "wrong", "miss")}; assert tot == F.READER_COUNTS == {"right": 5380, "wrong": 0, "miss": 1020}
    assert F.reader_outcome("Delete the service at (1,1).", "Delete the service at (1,2).") == "wrong" and F.reader_outcome("Delete the service at (1,1).", None) == "miss" and F.reader_outcome("Delete the service at (1,1).", "Delete the service at (1,1).") == "right"


# ---------------------------------------------------------------- 5 the session and the harness
def test_the_session_shows_the_names_and_the_request_in_words_or_with_coordinates_and_carries_the_expression_of_its_cell():
    seed = seed_of(192)
    for mode, shape in (("coords", "capped"), ("words", "capped"), ("words", "read")):
        s = F.FreeSession(seed, 100, mode, shape); rq = s.request(); tb = s.tables[0]; ins = s.script[0]["instruction"]; w = F.wording(seed, 0, ins, tb, s.news[0])[0]
        sysm, user = rq.messages; assert [m["role"] for m in rq.messages] == ["system", "user"] and sysm["content"] == PS.SYSTEM + F.NOTE if hasattr(PS, "SYSTEM") else sysm["content"].endswith(F.NOTE)
        assert "Names: " + F.legend(tb) in user["content"] and user["content"].endswith("Request: " + (w if mode == "words" else ins)) and user["content"].count("Request: ") == 1
        read = F.read_request(w, tb)[0] if shape == "read" else None
        assert rq.regex == (F.REGEX if read is None else T.regex_for("named", read)) and re.fullmatch(rq.regex, rq.stand_in_reply) and rq.shape == ("capped" if rq.regex == F.REGEX else "named")
        s.accept(PS.Reply(rq.stand_in_reply, 400, 0, 20)); x = s.log[0]
        assert x["in_language"] is True and x["exact"] is True and x["request"] == (w if mode == "words" else ins) and x["pool"] == (F.wording(seed, 0, ins, tb, s.news[0])[1] if mode == "words" else None)
        assert x["reader"] == (F.reader_outcome(ins, read) if shape == "read" else None) and "why" in x
    assert F.legend({(0, 0): "billing", (1, 2): "auth"}) == "billing = (0,0); auth (also authentication) = (1,2)" and F.legend({(2, 1): "auth"}) == "auth (also authentication) = (2,1)"
    for bad in (("x", "capped"),):
        try: F.FreeSession(seed, 100, *bad); assert False
        except AssertionError as ex: assert str(ex) != ""


def _engine(faults=None, forced=None): return T.ScriptedStandIn(faults=faults or {}, forced=forced or {}, capacity_tokens=int(F.CAPACITY_TOKENS * PS.PROXY_SCALE))
SLIP = "ADD[(1,1)->(1,1):dep#1]"            # a reply the capped expression allows and every named expression forbids


def _steps_where(label_outcome, kind=None):
    """(job, step) of the first request in the 64 jobs that the reader read (right) or did not read (miss), optionally of a kind."""
    for e, i, (ins, w, pool, tb, rd, why, oc) in _all_requests():
        if oc == label_outcome and (kind is None or T.parse_request(ins)["kind"] == kind) and i < 60: return e, i + 1


def good():
    if "good" not in _CACHE:
        e1, t1 = _steps_where("right", "add_edge"); e2, t2 = _steps_where("miss", "delete")
        _CACHE["faults"] = {("old", seed_of(e1), t1): SLIP, ("old", seed_of(e2), t2): SLIP}; _CACHE["fault_steps"] = ((e1, t1), (e2, t2))
        eng = _engine(); _CACHE["good"] = [F.run_cell(eng, lab) for lab in F.ORDER]
    return copy.deepcopy(_CACHE["good"])


def test_the_harness_writes_the_three_cells_with_the_flags_the_rounds_and_the_expressions():
    cells = good(); C = {c["label"]: c for c in cells}
    for lab, c in C.items():
        K, first, arm, mode, shape = F.CELLS[lab]
        assert (c["K"], c["first_episode"], c["arm"], c["mode"], c["shape"], c["steps"], c["seed"], c["policy"], c["hint_rules"]) == (64, 192, "S", mode, shape, 100, SEED, "repair_m6", False) and c["grammar_sha256"] == F.GRAMMAR_SHA[lab]
        assert len(c["sessions"]) == 64 and all(len(s["log"]) == 100 and s["stopped"] is None for s in c["sessions"]) and [r["round"] for r in c["rounds"]] == list(range(1, 101)) and all(r["active"] == 64 for r in c["rounds"])
        assert all(x["in_language"] is True and x["request"] and "reader" in x and "pool" in x for s in c["sessions"] for x in s["log"])
    assert C["WN"]["distinct_regexes"] == 775 and C["W0"]["distinct_regexes"] == C["C0"]["distinct_regexes"] == 1
    assert {k: sum(1 for s in C["WN"]["sessions"] for x in s["log"] if x["reader"] == k) for k in ("right", "wrong", "miss")} == F.READER_COUNTS and all(x["reader"] is None for s in C["W0"]["sessions"] for x in s["log"])
    assert all(x["pool"] is None for s in C["C0"]["sessions"] for x in s["log"]) and all(x["pool"] in "AB" for s in C["W0"]["sessions"] for x in s["log"])
    (e1, t1), (e2, t2) = _CACHE["fault_steps"]
    for lab in ("C0", "W0", "WN"):                                                                 # the slip on a request the host read: written where the expression is the capped one, forbidden where it is the named one
        pass
    eng = _engine(_CACHE["faults"]); D = {lab: F.run_cell(eng, lab) for lab in F.ORDER}
    lost = {lab: sum(1 for s in D[lab]["sessions"] for x in s["log"] if not x["exact"]) for lab in D}; assert lost["C0"] > 2 and lost["W0"] > 2 and 0 < lost["WN"] < lost["W0"], lost      # WN: the first slip is forbidden, the second (unread) is written
    wn = {job_step: [x["exact"] for x in next(s for s in D["WN"]["sessions"] if s["seed"] == seed_of(job_step[0]))["log"]] for job_step in ((e1, t1), (e2, t2))}
    assert wn[(e1, t1)][t1 - 1] is True and wn[(e2, t2)][t2 - 1] is False, "read: forced to the oracle's reply; not read: the slip is written and stays"


# ---------------------------------------------------------------- 6 the 16 lines
def read(cells): return [ok for _, _, _, ok in F.verdict_free(cells)]
def cell_of(cells, label): return next(c for c in cells if c["label"] == label)
def records(c): return [x for s in c["sessions"] for x in s["log"]]


def _prep():
    if "prep" not in _CACHE:                                        # a good run: every step exact, WN 5 percent quicker than W0 (the double has one speed)
        cells = [F.run_cell(_engine(), lab) for lab in F.ORDER]
        wn = cell_of(cells, "WN"); wn["wall_seconds"] *= 0.95
        for r in wn["rounds"]: r["seconds"] *= 0.95
        _CACHE["prep"] = cells
    return copy.deepcopy(_CACHE["prep"])


def changed(cells, base=None):
    base = [True] * 16 if base is None else base; return [i + 1 for i, (a, b) in enumerate(zip(base, read(cells))) if a != b]


def flip(label, fn):
    cells = _prep(); fn(cell_of(cells, label)); return cells


def lose(n, pred=lambda x: True, value=False):
    """For a cell: the first `n` records (jobs in order) that satisfy `pred` and whose flag is the other one get the flag `value`."""
    def f(c):
        pool = [x for x in records(c) if pred(x) and x["exact"] is not value]; assert len(pool) >= n, (len(pool), n)
        for x in pool[:n]: x["exact"] = value
    return f


def test_the_good_run_reads_yes_on_all_16_lines_and_the_first_six_lines_move_with_the_thing_they_check():
    assert read(_prep()) == [True] * 16 and [i for i, _, _, _ in F.verdict_free(_prep())] == list(range(1, 17)) and F.FL_HEADLINE == (8, 11)
    def nocache(c): records(c)[0]["n_cached"] = None
    assert changed(flip("W0", nocache)) == [1]
    def unparsed(n): return lambda c: [x.update(parsed=False) for x in records(c)[:n]]
    assert changed(flip("C0", unparsed(96))) == [] and changed(flip("C0", unparsed(97))) == [2], "19,200 replies: 96 may be unparsed (99.5 percent), 97 may not"
    def full(c): records(c)[0]["n_new"] = 96
    assert changed(flip("WN", full)) == [2]
    def outside(c): records(c)[5]["in_language"] = False
    assert changed(flip("W0", outside)) == [2]
    def sentence(c): c["sessions"][3]["log"][7]["request"] = "Remove it."
    assert changed(flip("W0", sentence)) == [3], "a sentence changed in W0: line 3 only (W0 has no recorded reader outcome)"
    assert changed(flip("WN", sentence)) == [3, 6], "the same in WN: the reader gives another outcome on it, or the recorded sentence is not the one generated"
    def pool(c): s = c["sessions"][3]["log"][7]; s["pool"] = "B" if s["pool"] == "A" else "A"
    assert changed(flip("W0", pool)) == [3]
    def sha(c): c["grammar_sha256"] = "0" * 64
    assert changed(flip("WN", sha)) == [4] and changed(flip("C0", sha)) == [4]
    def short(c): c["sessions"][5]["log"] = c["sessions"][5]["log"][:99]
    assert 5 in changed(flip("C0", short)) and 5 in changed(flip("WN", short))
    def stopped(c): c["sessions"][5]["stopped"] = "context"
    assert changed(flip("W0", stopped)) == [5]
    def outcome(c): x = records(c)[4]; x["reader"] = "right" if x["reader"] == "miss" else "miss"
    assert 6 in changed(flip("WN", outcome))
    def wrong(c):
        x = next(x for x in records(c) if x["reader"] == "right"); x["reader"] = "wrong"
    assert 6 in changed(flip("WN", wrong))


def with_w0(label_pred_n):
    """WN loses n steps satisfying pred; W0 loses n steps too, so that line 10 (WN at least W0) is not what moves."""
    n, pred = label_pred_n; cells = _prep(); lose(n, pred)(cell_of(cells, "WN")); lose(n)(cell_of(cells, "W0")); return changed(cells)


def test_each_claim_line_reads_exactly_at_its_edge_and_the_lines_that_must_move_with_it_move():
    assert math.ceil((F.L0_EXACT / F.L0_STEPS - F.T_C0_TOL) * 6400 - 1e-9) == 6180
    assert changed(flip("C0", lose(220))) == [] and changed(flip("C0", lose(221))) == [7], "line 7: 6,180 of 6,400 is the fewest C0 may have"
    assert changed(flip("C0", lose(6))) == [], "C0 a little lower moves nothing else"
    assert changed(flip("W0", lose(192))) == [] and changed(flip("W0", lose(193))) == [9], "line 9: three points of 6,400 is 192 steps"
    assert changed(flip("W0", lose(320))) == [9] and changed(flip("W0", lose(321))) == [8, 9], "line 8: 95 percent of 6,400 is 6,080"
    assert with_w0((64, lambda x: True)) == [12] and with_w0((65, lambda x: True)) == [11, 12], "line 11: 99 percent of 6,400 is 6,336, 64 may be lost"
    assert with_w0((26, lambda x: x["reader"] == "right")) == [] and with_w0((27, lambda x: x["reader"] == "right")) == [12], "line 12: 0.5 percent of 5,380 is 26"
    assert with_w0((51, lambda x: x["reader"] == "miss")) == [] and with_w0((52, lambda x: x["reader"] == "miss")) == [13], "line 13: 5 percent of 1,020 is 51"
    assert changed(flip("WN", lose(1))) == [10], "line 10: WN a step below W0"
    cells = _prep(); lose(10)(cell_of(cells, "WN")); lose(10)(cell_of(cells, "W0")); assert changed(cells) == [], "equal shares: line 10 is 'at least'"
    assert changed(flip("W0", lose(5))) == [], "W0 lower than WN: line 10 holds"
    lose_b = lambda n: lose(n, lambda x: x["pool"] == "B")
    assert (sum(1 for x in records(cell_of(_prep(), "W0")) if x["pool"] == "B")) == 1548
    assert changed(flip("W0", lose_b(46))) == [] and changed(flip("W0", lose_b(47))) == [14], "line 14: 46 of 1,548 is 2.97 points"
    def slow(c, f): c["wall_seconds"] *= f; [r.update(seconds=r["seconds"] * f) for r in c["rounds"]]
    def at(ratio):
        cells = _prep(); w0 = cell_of(cells, "W0"); wn = cell_of(cells, "WN"); slow(wn, ratio * w0["wall_seconds"] / wn["wall_seconds"]); return changed(cells)
    assert at(1.15) == [16], "15 percent over is allowed (line 15), and that is not faster per hour (line 16)"
    assert at(1.1501) == [15, 16] and at(1.0) == [16] and at(0.95) == []


def test_a_cell_that_was_not_run_reads_not_run_and_the_rule_about_claims_follows_the_lines():
    cells = _prep(); only = [cell_of(cells, "C0")]; r = read(only)
    assert r[0] is True and r[1] is True and r[6] is True and r[7] is None and r[9] is None and r[10] is None and r[15] is None and r[2] is None and r[5] is None
    assert read([c for c in cells if c["label"] != "C0"])[6] is None and read([c for c in cells if c["label"] != "WN"])[9] is None and read([c for c in cells if c["label"] != "W0"])[7] is None
    assert read([c for c in cells if c["label"] != "W0"])[13] is None and read([c for c in cells if c["label"] != "WN"])[11] is None
    a = F.claims_allowed(F.verdict_free(cells)); assert a["base"] is True and a["names_do_not_hurt"] == {7: True} and a["the_model_reads_names"] == {8: True, 9: True, 14: True} and a["the_host_reads"] == {10: True, 11: True, 12: True, 13: True} and a["time_and_cost_in_one_session"] == {15: True, 16: True}
    for line in (2, 3, 4, 5, 6):
        chk = [(i, t, x, (False if i == line else ok)) for i, t, x, ok in F.verdict_free(cells)]; b = F.claims_allowed(chk)
        assert b["base"] is False and all(v is None for k in ("names_do_not_hurt", "the_model_reads_names", "the_host_reads", "time_and_cost_in_one_session") for v in b[k].values()), line
    chk = [(i, t, x, (False if i == 11 else ok)) for i, t, x, ok in F.verdict_free(cells)]; assert F.claims_allowed(chk)["the_host_reads"][11] is False and F.claims_allowed(chk)["base"] is True


def test_the_printout_the_dry_run_the_merge_and_the_command_line():
    cells = _prep(); S = F.summarize_all(cells); buf = io.StringIO()
    with contextlib.redirect_stdout(buf): F.print_cells(S); F.print_split(S); F.print_misses(S); F.print_verdict(F.verdict_free(cells))
    out = buf.getvalue()
    for lab in F.ORDER: assert re.search(rf"^{lab} +64 +192-255", out, re.M)
    assert "host read right 5380, wrong 0, not read 1020" in out and "-- provider-side test, requests in words (reading rules fixed before the run) --" in out and "(16 of 16 lines could be read; headline lines 8 and 11: yes, yes)" in out and "lines 2 to 6 read yes" in out
    for i, tag, text, ok in F.verdict_free(cells): assert f"{i:2d}. [{tag}] {text}" in out
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): F.print_verdict([(i, t, x, (False if i == 3 else ok)) for i, t, x, ok in F.verdict_free(cells)])
    assert "line 2, 3, 4, 5 or 6 did not read yes" in buf.getvalue() and "yes      " in buf.getvalue() and "NO       " in buf.getvalue()
    with contextlib.redirect_stdout(io.StringIO()) as o: S2 = F.dry_run(labels=["C0"])
    assert list(S2) == ["C0"] and S2["C0"]["exact_share"] == 1.0
    with contextlib.redirect_stdout(io.StringIO()) as o: F.print_cells(S2)
    d = os.path.join(os.path.dirname(MODULE), "..", "..", "..", "tmp_free_merge"); os.makedirs(d, exist_ok=True)
    try:
        paths = [F.save_cell(c, os.path.join(d, f"c_{c['label']}.json")) for c in cells]; out = os.path.join(d, "m.json"); m = F.merge_files(paths, out)
        assert [c["label"] for c in m["cells"]] == F.ORDER and m["module_sha256"] == F.fingerprint() == _sha(MODULE) and m["pinned_sha256"] == PINS and m["free_pinned_sha256"] == F.pinned_files() and json.load(open(out))["meta"]["steps"] == 100
        for bad, msg in ((lambda cs: [cs[0], cs[0]], "twice"), (lambda cs: [dict(cs[0], label="L0")], "not one of this test's"), (lambda cs: [dict(cs[0], steps=30)], "30 steps"), (lambda cs: [cs[0], dict(cs[1], seed=1)], "other settings"),
                         (lambda cs: [dict(cs[0], mode="words")], "settings of its label"), (lambda cs: [dict(cs[0], engine=dict(cs[0]["engine"], constraint=dict(cs[0]["engine"]["constraint"], regex_sha256="0")))], "constraint"),
                         (lambda cs: [cs[0], dict(cs[1], engine=dict(cs[1]["engine"], kind="vllm"))], "other settings")):
            try: F.merge_cells(bad(copy.deepcopy(cells))); assert False, msg
            except ValueError as ex: assert msg in str(ex), (msg, str(ex))
        try: F.merge_cells([]); assert False
        except AssertionError: pass
    finally:
        for f in glob.glob(os.path.join(d, "*")): os.remove(f)
        os.rmdir(d)
    p = subprocess.run([sys.executable, MODULE, "--dry-run"], capture_output=True, text=True, cwd=os.path.join(HERE, ".."), env=dict(os.environ, PYTHONPATH=os.path.join(HERE, "..")), timeout=600)
    assert p.returncode == 0 and "STAND-IN" in p.stdout and "NOT a result" in p.stdout, p.stderr[-500:]
    p = subprocess.run([sys.executable, MODULE], capture_output=True, text=True, cwd=os.path.join(HERE, ".."), env=dict(os.environ, PYTHONPATH=os.path.join(HERE, "..")), timeout=600); assert p.returncode == 0 and "--merge" in p.stdout


# ---------------------------------------------------------------- 7 the pre-flight check
def _gpu(name="Tesla T4", used=0): return lambda: dict(name=name, used_mib=used, total_mib=15360)


def test_the_preflight_is_green_on_this_repository_and_each_check_can_read_no():
    res = F.preflight(gpu=_gpu()); assert all(ok for _, ok, _ in res) and len(res) == 11, [r for r in res if not r[1]]
    names = [n for n, _, _ in res]; assert names[0] == "the jobs are the sixth run's, 100 steps each" and names[-1] == "the GPU is a free T4" and len(set(names)) == 11
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): good_ = F.print_preflight(res)
    assert good_ is True and "GREEN: nothing found" in buf.getvalue() and buf.getvalue().count("  ok    ") == 11
    def no(**kw):
        r = F.preflight(**kw); bad = [n for n, ok, _ in r if not ok]; return bad
    assert no(gpu=_gpu("Tesla V100")) == ["the GPU is a free T4"] and no(gpu=_gpu(used=9000)) == ["the GPU is a free T4"] and no(gpu=lambda: None) == ["the GPU is a free T4"]
    def boom(): raise RuntimeError("nvidia-smi broke")
    r = F.preflight(gpu=boom); assert [n for n, ok, _ in r if not ok] == ["the GPU is a free T4"] and "could not be checked: RuntimeError" in dict((n, d) for n, _, d in r)["the GPU is a free T4"]
    assert no(gpu=_gpu(), steps=99) != [] and no(gpu=_gpu(), long_file=PAGE) == ["the sixth run's file is the one this page names"]
    saved = (F.READER_COUNTS, F.GRAMMAR_SHA, F.GROUNDING_SHA256, F.WORD_TASKS_SHA256, F.LONG_MODULE_SHA256)
    try:
        F.READER_COUNTS = {"right": 5381, "wrong": 0, "miss": 1019}; assert no(gpu=_gpu()) == ["the host reader reads the 6,400 requests as the page says"]; F.READER_COUNTS = saved[0]
        F.GRAMMAR_SHA = dict(saved[1], WN="0" * 64); assert no(gpu=_gpu()) == ["the expressions are the ones on the page"]; F.GRAMMAR_SHA = saved[1]
        F.GROUNDING_SHA256 = "0" * 64; assert no(gpu=_gpu()) == ["the files this test imports are the pinned ones"]; F.GROUNDING_SHA256 = saved[2]
        F.WORD_TASKS_SHA256 = "0" * 64; assert no(gpu=_gpu()) == ["the files this test imports are the pinned ones"]; F.WORD_TASKS_SHA256 = saved[3]
        F.LONG_MODULE_SHA256 = "0" * 64; assert no(gpu=_gpu()) == ["the files this test imports are the pinned ones"]; F.LONG_MODULE_SHA256 = saved[4]
        pins = dict(F.PINS); F.PINS = dict(pins, gate="0" * 64); assert no(gpu=_gpu()) == ["the files this test imports are the pinned ones"]; F.PINS = pins
        w = F.wording; F.wording = lambda seed, t, ins, tb, nn: ("Remove it.", "A"); F.requests_of.cache_clear()
        try: bad = no(gpu=_gpu())
        finally: F.wording = w; F.requests_of.cache_clear()
        assert "the requests in words say what the oracle's request says" in bad and "the host reader reads the 6,400 requests as the page says" in bad
        rr = F.read_request; F.read_request = lambda text, table: ("Delete the service at (9,9).", None); F.requests_of.cache_clear()
        try: bad = no(gpu=_gpu())
        finally: F.read_request = rr; F.requests_of.cache_clear()
        assert "the oracle's reply is inside the expression of every request" in bad and "the host reader reads the 6,400 requests as the page says" in bad
    finally: F.READER_COUNTS, F.GRAMMAR_SHA, F.GROUNDING_SHA256, F.WORD_TASKS_SHA256, F.LONG_MODULE_SHA256 = saved
    assert F.preflight(gpu=_gpu()) == res


# ---------------------------------------------------------------- 8 the locked page
def test_the_page_is_locked_and_says_the_design_the_wordings_the_reader_and_the_lines_that_the_code_has():
    head = _head(); assert _h(head) == PAGE_SHA256 and head.rstrip("\n").endswith("---") and head.startswith("# Test 20: the serving side a seventh time, requests in ordinary words (written BEFORE the run, 21 Oct 2026)")
    for tpl in F.A_DELETE + F.B_DELETE + F.A_NODE + F.B_NODE + F.A_EDGE + F.B_EDGE: assert f"`{tpl}`" in head, tpl
    for i, ((tag, text), p) in enumerate(zip(F.FL_LINES, [99, 92, 99, 99, 96, 99, 65, 60, 65, 90, 55, 50, 65, 55, 88, 75]), 1): assert f"{i}. [{tag}] {text} ({p} %)" in head, i
    for s in (f"`{F.REGEX_SHA256}`", f"`{F.GRAMMAR_SHA['WN']}`", "**5,380 right, 0 wrong and 1,020 not at all**", "(546 deletes, 474 edges)", "774 named ones and the capped one", "24.2 %", "192 to 255", "6,244 of 6,400", "97.6 %", "19,200 requests",
              "the 22 names and the 7 aliases of `word_tasks.py`", "the lines of the words test about names (`NOTE` in the code)"[:0], "one mention of five", "Lines 2 to 6 all yes: about **85 %**", "Lines 8 and 11 both yes: about **40 %**", "All 16 yes: about **3 %**",
              "**The rule that decides what the Result may claim.**", "Line 2, 3, 4, 5 or 6 reads NO: no claim is made from this run until the cause is found.", "never \"100 percent\" for a grammar", "never \"proven\"", "never a statement about requests in the words of a user",
              "it looked for \"relation\" and \"weight\" in lower case", "No wording was changed.", "is not language understanding", "A \"NO\" on a claim line is a finding about this setup", "Nothing here is proven.", "in the same sentence that the second pool of wordings was refused by the host and read by the model"):
        if s: assert s in head, s
    assert "# Result" not in head and "pip install" not in head and "Concept-map item 113" in head and "edits none of them" in head
    assert re.findall(r"\(([0-9]+) %\)\n", head.split("## Predictions")[1].split("**Headline**")[0]) == [str(p) for p in (99, 92, 99, 99, 96, 99, 65, 60, 65, 90, 55, 50, 65, 55, 88, 75)]
    assert all(f"({x}" in head for x in "abcdef")
    for needle in ("`C0`", "`W0`", "`WN`"): assert head.count(needle) > 5


# ---------------------------------------------------------------- 9 the notebook
def test_the_notebook_has_the_cells_in_order_and_its_code_runs_on_the_double_with_a_store():
    nb = json.load(open(NB)); C = nb["cells"]; src = lambda i: "".join(C[i]["source"])
    assert len(C) == 22 and [c["cell_type"] for c in C] == ["markdown", "markdown", "code"] + ["markdown", "code"] * 9 + ["markdown"] and nb["metadata"]["accelerator"] == "GPU"
    text = "\n".join("".join(c["source"]) for c in C)
    for needle in ("AXI test 20", "docs/PROVIDER_SIDE_FREE_TEST.md", "jobs 192 to 255", "**T4 GPU**", "vllm==0.31.0", "provider_side_free.py", "provider_side_long.py", "provider_side_typed.py", "finish the update 69 commands", "axi_provider_free",
                   "axi_provider_f_", "F.preflight()", "assert green", "PRE-FLIGHT IS RED", "TypedVLLMEngine()", "eng.warm_up()", "for label in F.ORDER: go(label)", "F.merge_files(", "F.verdict_free(m[\"cells\"])", "files.download", "Cell 4b",
                   "the three cells about 33 minutes", "There is **no retry in any cell**"):
        assert needle in text, needle
    assert "update 67" not in text
    assert "provider_side_long import" not in text and "axi_provider_long" not in text and "LN" not in text and "17 reading-rule" not in text and "four cells" not in text
    # the code cells run on the double: Cell 6's go(), Cell 7's safety check, Cell 8, Cell 9 (with a fake store and a fake 'google.colab')
    class Store:
        def __init__(s, d): s.d = d
        def path(s, n): return os.path.join(s.d, n)
        def restore(s, n): return "not found"
        def persist(s, n): pass
    import tempfile, time as _time
    d = tempfile.mkdtemp(); fake = types.ModuleType("google.colab"); fake.files = types.SimpleNamespace(download=lambda p: None); sys.modules["google"] = types.ModuleType("google"); sys.modules["google.colab"] = fake
    ns = {"sys": sys, "os": os, "json": json, "time": _time, "glob": glob, "re": re, "store": Store(d)}
    try:
        pre, post = src(14).split("eng.warm_up()")
        ns["eng"] = _engine(); ns["T"] = T; ns["L"] = L; ns["F"] = F; import axi.experiments.q_measure as Q; ns["Q"] = Q
        go_src = "def go" + post.split("def go", 1)[1]; exec(go_src, ns)
        with contextlib.redirect_stdout(io.StringIO()) as o: exec(src(16), ns)
        assert "OK: cached counts are reported" in o.getvalue() and "STOP" not in o.getvalue().replace("A wrong answer", "") and o.getvalue().count("step 1:") == 3
        t0 = _time.time()
        with contextlib.redirect_stdout(io.StringIO()) as o: exec(src(18), ns)
        assert sorted(os.listdir(d)) == ["axi_provider_f_C0.json", "axi_provider_f_W0.json", "axi_provider_f_WN.json"] and o.getvalue().count("done: 6400 steps") == 3, o.getvalue()[-400:]
        with contextlib.redirect_stdout(io.StringIO()) as o: exec(src(20), ns)
        out = o.getvalue(); assert "3 cells:" in out and "(16 of 16 lines could be read" in out and os.path.exists(os.path.join(d, "axi_provider_free.json")) and json.load(open(os.path.join(d, "axi_provider_free.json")))["meta"]["steps"] == 100
    finally:
        sys.modules.pop("google.colab", None); sys.modules.pop("google", None)
        for f in glob.glob(os.path.join(d, "*")): os.remove(f)
        os.rmdir(d)


# ---------------------------------------------------------------- 10 the front pages
def test_the_concept_map_and_the_verified_page_say_what_the_page_and_the_code_say():
    cm = open(os.path.join(ROOT, "docs", "RAJNISH_CONCEPT_MAP.md")).read(); vf = open(os.path.join(ROOT, "docs", "VERIFIED.md")).read()
    item = next(l for l in cm.split("\n") if l.startswith("113. **")); row = next(l for l in vf.split("\n") if l.startswith("| Provider-side serving test, seventh run"))
    assert "Next free: row 186, item 140, test 165, question 122" in cm and cm.count("Next free:") == 1 and "test 135 checks the free-word serving test, item 113" in cm
    assert PAGE_SHA256[:16] in item and PAGE_SHA256[:16] in row and "test 135" in item and "test 135" in row and "mutation check of test 135: not yet run" in item and "mutation check of test 135: not yet run" in row
    for s in ("5,380", "1,020", "none wrong", "16 reading lines", "lines 8 and 11", "about 40 %", "then run once on Colab", "jobs 192 to 255", "`C0`", "`W0`", "`WN`", "11 checks", "nothing here is proven"): assert s in item, s
    assert "5,380" in row and "1,020" in row and "run once on Colab" in row and "Next free: row 170, item 113" not in cm
    assert "PROVIDER_SIDE_FREE_TEST.md" in vf and os.path.exists(PAGE) and os.path.exists(NB)


if __name__ == "__main__":
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]; bad = 0
    for n, f in fns:
        try: f(); print("ok  ", n)
        except Exception: bad += 1; print("FAIL", n); traceback.print_exc()
    print(f"{len(fns) - bad}/{len(fns)} passed"); sys.exit(1 if bad else 0)
