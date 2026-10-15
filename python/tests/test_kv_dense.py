"""Engineering support for docs/DENSE_TEST.md (concept-map item 101): does the way a dense graph is written change what a stock model can read from it? This file tests the INSTRUMENT, not the finding:
  1. the graphs and the questions: counts, density, truths, balance of the existence questions, repeatable from the seed, a fingerprint that changes with anything;
  2. the five formats: each is exactly what the page says (a three-service graph written out by hand, bit order included) and each is read back by its own decoder to exactly the graph's edges; a
     tampered state or a wrong column order is not read back as the graph;
  3. the token count (the lower bound) on hand cases, and the dry-run table of the page (its numbers are the code's);
  4. the scoring: replies parsed to services and to yes or no, precision, recall, F1, exact, the asked service not counted, the baseline of answering with everything;
  5. greedy decoding with no grammar: stops at the end token, is cut at max_new, never picks a token outside the vocabulary;
  6. the runner on the numpy stand-in: every question is a fresh prompt holding the state and the question and nothing of the truth, one line per (graph, format), resumes after a dropped session, refuses
     a file made with other settings; the summary adds up by hand;
  7. the fourteen reading rules, each flipping exactly at its threshold, and "not run" when the data is missing; the printout;
  8. the page is locked (hash of the text above the Result), carries the fourteen lines with their probabilities, the settings and thresholds the code uses, the words quoted from the pasted text, and the
     facts it states about the repository (which this test looks up)."""
import contextlib, hashlib, io, itertools, json, os, random, re, sys, tempfile

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import axi.experiments.kv_dense as D
from axi.engine.cico_parser import CICOParser
from axi.engine.gate import Graph
from kv_standin import TinyCache, TinyEngine, byte_kit, byte_table

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..", "..")
PAGE = os.path.join(ROOT, "docs", "DENSE_TEST.md")
PAGE_SHA256 = "e5a2b70b7ddaaae3b2473238a291045b9431a8f182134e9f866e6e6b3e387059"
RESULT_MARK = "\n\n---\n\n# Result"
KIT, TAB = byte_kit(), byte_table()
TINY = dict(regimes=(("dense", 1, 12), ("sparse", 1, 5)), nodes=6, grid=(3, 3))


def _h(text): return hashlib.sha256((text.rstrip("\n") + "\n").encode()).hexdigest()


def raises(exc, f, *a, **k):
    try: f(*a, **k)
    except exc: return
    raise AssertionError(f"{f} did not raise {exc}")


def hand_graph():
    """Three services (0,0) (0,1) (1,0); edges (0,0)->(0,1), (0,0)->(1,0), (1,0)->(0,0)."""
    g = Graph()
    for n, v in (((0, 0), 1), ((0, 1), 2), ((1, 0), 3)): g.add_node(n, v)
    g.add_edge((0, 0), (0, 1), "dep", 4); g.add_edge((0, 0), (1, 0), "dep", 5); g.add_edge((1, 0), (0, 0), "dep", 6)
    return g


# ---------------------------------------------------------------- 1 graphs and questions
def test_the_graphs_have_the_density_the_page_says_and_repeat_from_the_seed():
    ps = D.make_problems()
    assert [p["id"] for p in ps] == ["d0", "d1", "d2", "d3", "d4", "s0", "s1", "s2"] and [p["regime"] for p in ps] == ["dense"] * 5 + ["sparse"] * 3
    for p in ps:
        g = p["graph"]; assert len(g.nodes) == 24 and len(g.edges) == (400 if p["regime"] == "dense" else 138) and all(0 <= r < 5 and 0 <= c < 5 for r, c in g.nodes) and all(1 <= v <= 9 for v in g.nodes.values())
        assert all(u != v and u in g.nodes and v in g.nodes and rel == "dep" and 1 <= w <= 9 for (u, v, rel), w in g.edges.items()) and len(D.pair_set(g)) == len(g.edges)
        assert abs(D.density(g) - (400 if p["regime"] == "dense" else 138) / 552) < 1e-12
    assert abs(D.density(ps[0]["graph"]) - 0.7246376811594203) < 1e-12 and abs(D.density(ps[5]["graph"]) - 0.25) < 1e-12
    assert D.fingerprint(D.make_problems()) == D.fingerprint(ps) and D.fingerprint(D.make_problems(seed=52)) != D.fingerprint(ps)
    other = D.make_problems(); other[3]["questions"][0]["truth"] = other[3]["questions"][0]["truth"][:-1] + [[9, 9]]; assert D.fingerprint(other) != D.fingerprint(ps)
    other = D.make_problems(); other[0]["graph"].edges[next(iter(other[0]["graph"].edges))] += 1; assert D.fingerprint(other) != D.fingerprint(ps)
    other = D.make_problems(); n0 = next(iter(other[0]["graph"].nodes)); other[0]["graph"].nodes[n0] += 1; assert D.fingerprint(other) != D.fingerprint(ps)                 # a service's value
    other = D.make_problems(); other[2]["questions"][1]["text"] += " "; assert D.fingerprint(other) != D.fingerprint(ps)                                            # the wording of a question
    other = D.make_problems(); other[5]["regime"] = "dense"; assert D.fingerprint(other) != D.fingerprint(ps)
    rng = random.Random(1); raises(AssertionError, D.make_graph, rng, 600); raises(AssertionError, D.make_graph, rng, 5, 30, (5, 5))


def test_the_questions_are_eight_per_graph_with_the_right_truths_and_balanced_existence():
    for p in D.make_problems():
        g, qs = p["graph"], p["questions"]; n = len(g.nodes); assert [q["kind"] for q in qs] == ["out", "out", "in", "in", "exist", "exist", "exist", "exist"] and [q["qid"] for q in qs] == [f"q{i}" for i in range(8)]
        pairs = D.pair_set(g)
        for q in qs[:4]:
            x = tuple(q["node"]); want = D.out_nb(g, x) if q["kind"] == "out" else D.in_nb(g, x)
            assert q["text"].startswith("List every service that the service at (" if q["kind"] == "out" else "List every service that has an edge to the service at (")                  # the wording says the direction
            assert [tuple(v) for v in q["truth"]] == want and 1 <= len(want) <= n - 2 and x not in want and f"({x[0]},{x[1]})" in q["text"] and q["text"].count("(") == 2
        assert qs[0]["node"] != qs[1]["node"] and qs[2]["node"] != qs[3]["node"]
        ex = qs[4:]; assert sorted(q["truth"] for q in ex) == ["no", "no", "yes", "yes"]
        for q in ex:
            u, v = (tuple(x) for x in q["pair"]); assert u != v and ((u, v) in pairs) == (q["truth"] == "yes") and f"({u[0]},{u[1]})" in q["text"] and f"({v[0]},{v[1]})" in q["text"] and "yes or no" in q["text"]
    a, b = D.make_problems(seed=51), D.make_problems(seed=51); assert [q["text"] for p in a for q in p["questions"]] == [q["text"] for p in b for q in p["questions"]]
    ts = {tuple(q["text"] for q in p["questions"]) for p in D.make_problems(seed=52)} | {tuple(q["text"] for q in p["questions"]) for p in a}; assert len(ts) == 16


def test_a_list_question_never_asks_for_everything_or_for_nothing():
    g = Graph(); a, b, c, d, e = (0, 0), (0, 1), (0, 2), (1, 0), (1, 1)
    for n in (a, b, c, d, e): g.add_node(n, 1)
    for u, v in ((a, b), (a, c), (a, d), (a, e), (b, c), (c, d), (d, e)): g.add_edge(u, v, "dep", 1)                 # a points to all four others, e to none; nothing points to a
    assert D.out_nb(g, a) == [b, c, d, e] and D.out_nb(g, e) == [] and D.in_nb(g, a) == []
    for seed in range(60):
        qs = D.make_questions(random.Random(seed), g)
        for q in qs[:2]: assert tuple(q["node"]) in (b, c, d) and 1 <= len(q["truth"]) <= 3, (seed, q)
        for q in qs[2:4]: assert tuple(q["node"]) != a and 1 <= len(q["truth"]) <= 3, (seed, q)
        assert sorted(q["truth"] for q in qs[4:]) == ["no", "no", "yes", "yes"]


def test_in_and_out_neighbours_are_told_apart_on_a_hand_graph():
    g = hand_graph(); assert D.out_nb(g, (0, 0)) == [(0, 1), (1, 0)] and D.in_nb(g, (0, 0)) == [(1, 0)] and D.out_nb(g, (0, 1)) == [] and D.in_nb(g, (0, 1)) == [(0, 0)] and D.pair_set(g) == {((0, 0), (0, 1)), ((0, 0), (1, 0)), ((1, 0), (0, 0))}
    assert abs(D.density(g) - 3 / 6) < 1e-12


# ---------------------------------------------------------------- 2 the five formats
def test_each_format_is_what_the_page_says_on_a_hand_graph():
    g = hand_graph(); cols = "(0,0) (0,1) (1,0)"
    t = D.render(g, "tuples"); assert t == CICOParser.serialize_state([(0, 0, 1), (0, 1, 2), (1, 0, 3)], [(0, 0, 0, 1, "dep", 4), (0, 0, 1, 0, "dep", 5), (1, 0, 0, 0, "dep", 6)]) and "->" in t and "dep#4" in t
    a = D.render(g, "adjacency"); assert a.startswith("<GRAPH>\n") and a.endswith("\n</GRAPH>") and "(0,0) out: (0,1) (1,0) in: (1,0)\n(0,1) out:  in: (0,0)\n(1,0) out: (0,0) in: (0,0)" in a
    m = D.render(g, "matrix"); assert f"Columns, in order: {cols}\n" in m and "(0,0): 0 1 1\n(0,1): 0 0 0\n(1,0): 1 0 0\n" in m + "\n" and "otherwise 0" in m
    b = D.render(g, "binary"); assert f"Columns, in order: {cols}\n" in b and "(0,0): 011\n(0,1): 000\n(1,0): 100\n" in b + "\n" and "counting from the left, starting at 0" in b
    h = D.render(g, "bitset"); assert f"Columns, in order (column 0 first): {cols}\n" in h and "(0,0): 0x0000000000000006\n(0,1): 0x0000000000000000\n(1,0): 0x0000000000000001\n" in h + "\n" and "counting from the right, starting at 0" in h and "64-bit" in h
    for f in D.DENSE_FORMATS: assert D.decode_state(D.render(g, f), f) == D.pair_set(g), f
    raises(AssertionError, D.render, g, "json")


def test_every_format_of_every_graph_is_read_back_to_exactly_its_edges_and_a_wrong_state_is_not():
    for p in D.make_problems():
        g = p["graph"]
        for f in D.DENSE_FORMATS: assert D.decode_state(D.render(g, f), f) == D.pair_set(g), (p["id"], f)
    p = D.make_problems()[0]; g = p["graph"]; ns = sorted(g.nodes); u = next(n for n in ns if (n, ns[0]) in D.pair_set(g) and n != ns[0])
    bs = D.render(g, "bitset"); lines = bs.split("\n"); k = next(i for i, ln in enumerate(lines) if ln.startswith(f"({u[0]},{u[1]}): 0x")); w = int(lines[k].split("0x")[1], 16); lines[k] = f"({u[0]},{u[1]}): 0x{w ^ 1:016x}"
    assert D.decode_state("\n".join(lines), "bitset") == D.pair_set(g) - {(u, ns[0])}                                          # one bit off = one edge off
    mx = D.render(g, "matrix"); swapped = mx.replace(f"{ns[0][0]},{ns[0][1]}) ({ns[1][0]},{ns[1][1]})", f"{ns[1][0]},{ns[1][1]}) ({ns[0][0]},{ns[0][1]})", 1); assert swapped != mx and D.decode_state(swapped, "matrix") != D.pair_set(g)
    ad = D.render(g, "adjacency"); first_in = re.search(r" in: (\(\d+,\d+\) )", ad); assert first_in
    raises(AssertionError, D.decode_state, ad.replace(first_in[0], " in: ", 1), "adjacency")                               # an edge on one side only
    short = re.sub(r"(\(\d+,\d+\): [01]+)[01]\n", r"\1\n", D.render(g, "binary"), count=1); raises(AssertionError, D.decode_state, short, "binary")
    tup = D.render(g, "tuples"); assert len(D.decode_state(tup, "tuples")) == 400 and D.decode_state(tup.replace("dep#", "dep#1", 1), "tuples") == D.pair_set(g)


# ---------------------------------------------------------------- 3 tokens
def test_the_token_count_is_the_pre_token_count_of_qwen_and_the_page_table_is_the_codes():
    c = D.qwen_like_tokens
    assert c("(0,1)") == 5 and c(" 0 1") == 4 and c("0x00ff") == 5 and c("hello world") == 2 and c("a\n\nb") == 3 and c("") == 0 and c("0 1 0") == 5 and c("0101") == 4 and c("12345") == 5 and c("a_b") == 3 and c("->(") == 1 and c(")->(0") == 2 and c("a\n") == 2
    dry = D.dense_dry_run()
    for reg, row in (("dense", (4982, 4248, 1459, 918, 743)), ("sparse", (1838, 1628, 1459, 918, 789))):
        assert [round(dry[reg][f]) for f in D.DENSE_FORMATS] == list(row), (reg, [dry[reg][f] for f in D.DENSE_FORMATS])
    assert [round(dry["dense"][f + "_over_tuples"], 2) for f in D.DENSE_FORMATS] == [1, 0.85, 0.29, 0.18, 0.15] and [round(dry["sparse"][f + "_over_tuples"], 2) for f in D.DENSE_FORMATS] == [1, 0.89, 0.79, 0.50, 0.43]
    assert abs(dry["dense"]["rho"] - 0.7246) < 1e-3 and abs(dry["sparse"]["rho"] - 0.25) < 1e-9 and dry["dense"]["matrix_over_tuples"] <= D.N_TOK_MATRIX and dry["dense"]["bitset_over_tuples"] <= D.N_TOK_BITSET
    assert D.dense_dry_run(count=len)["dense"]["tuples_over_tuples"] == 1.0 and D.dense_dry_run(count=len)["dense"]["tuples"] == np.mean([len(D.render(p["graph"], "tuples")) for p in D.make_problems() if p["regime"] == "dense"])


def test_where_the_token_counts_cross_is_what_the_page_says():
    rng = random.Random(5); page = {20: (422, 469, 791), 60: (902, 851, 790), 138: (1838, 1628, 789), 300: (3782, 3248, 767), 400: (4982, 4248, 742)}; got = {}
    for E in (20, 40, 60, 80, 138, 200, 300, 400):                                                                      # the order and the seed of the page's computation
        gs = [D.make_graph(rng, E) for _ in range(3)]; got[E] = tuple(int(np.mean([D.qwen_like_tokens(D.render(g, f)) for g in gs])) for f in ("tuples", "adjacency", "bitset"))
    for E, want in page.items(): assert got[E] == want, (E, got[E], want)
    assert got[60][2] < got[60][0] and got[20][2] > got[20][0] and got[40][2] > got[40][0] and got[80][2] < got[80][0]                  # the packed form is shorter than the tuples from about 60 edges: a density of 0.11


# ---------------------------------------------------------------- 4 scoring
def test_replies_are_parsed_and_scored_by_hand():
    assert D.parse_nodes("(0,1) (2, 3)\n( 4 ,4 ), (1,1).") == [(0, 1), (2, 3), (4, 4), (1, 1)] and D.parse_nodes("none") == [] and D.parse_nodes("(1,2,3)") == []
    assert D.parse_yes_no("Yes.") == "yes" and D.parse_yes_no("No, there is none") == "no" and D.parse_yes_no("The answer is YES") == "yes" and D.parse_yes_no("maybe") is None and D.parse_yes_no("") is None and D.parse_yes_no("nothing") is None
    assert D.parse_yes_no("no yes") == "no" and D.parse_yes_no("yesterday") is None and D.parse_yes_no("(0,1) yes") == "yes"
    s = D.score_list("(0,1) (1,0) (2,2)", [[0, 1], [1, 0], [1, 1], [2, 0]], [3, 3]); assert s["precision"] == 2 / 3 and s["recall"] == 0.5 and abs(s["f1"] - 2 * (2 / 3) * 0.5 / (2 / 3 + 0.5)) < 1e-12 and not s["exact"] and s["listed"] == 3 and s["wrong"] == 1
    s = D.score_list("(0,1) (0,1) (1,0)", [[0, 1], [1, 0]], [3, 3]); assert s["precision"] == 1 and s["recall"] == 1 and s["f1"] == 1 and s["exact"] and s["listed"] == 2                    # a repeat is one service
    s = D.score_list("(3,3) (0,1)", [[0, 1]], [3, 3]); assert s["exact"] and s["listed"] == 1                                                                                         # the asked service is not counted
    s = D.score_list("", [[0, 1]], [3, 3]); assert (s["precision"], s["recall"], s["f1"], s["exact"], s["listed"]) == (0.0, 0.0, 0.0, False, 0)
    s = D.score_list("(2,2)", [[0, 1]], [3, 3]); assert (s["precision"], s["recall"], s["f1"]) == (0.0, 0.0, 0.0) and s["wrong"] == 1
    assert abs(D.everything_f1(1, 24) - 1 / 12) < 1e-12 and abs(D.everything_f1(23, 24) - 1.0) < 1e-12 and abs(D.everything_f1(12, 24) - 2 * (12 / 23) / (12 / 23 + 1)) < 1e-12
    ps = D.make_problems(); dense_k = [len(q["truth"]) for p in ps if p["regime"] == "dense" for q in p["questions"] if q["kind"] != "exist"]; sparse_k = [len(q["truth"]) for p in ps if p["regime"] == "sparse" for q in p["questions"] if q["kind"] != "exist"]
    assert abs(np.mean([D.everything_f1(k, 24) for k in dense_k]) - 0.833) < 1e-3 and abs(np.mean([D.everything_f1(k, 24) for k in sparse_k]) - 0.361) < 1e-3 and (len(dense_k), len(sparse_k)) == (20, 12)
    q = dict(kind="exist", truth="yes"); assert D.score_question(q, "Yes", 24) == dict(answer="yes", correct=True, parsed=True) and D.score_question(q, "no", 24)["correct"] is False and D.score_question(q, "unsure", 24) == dict(answer=None, correct=False, parsed=False)
    q = dict(kind="out", node=[0, 0], truth=[[0, 1], [1, 0]]); s = D.score_question(q, "(0,1)", 24); assert s["recall"] == 0.5 and s["precision"] == 1 and abs(s["baseline_f1"] - D.everything_f1(2, 24)) < 1e-12


# ---------------------------------------------------------------- 5 decoding
class Scripted:
    """A cache whose logits put the highest value on the next scripted token (and a higher one OUTSIDE the vocabulary, which must be ignored)."""
    def __init__(self, script, V=10): self.script, self.V, self.calls = list(script), V, []

    def _lg(self):
        lg = np.zeros(self.V + 3, np.float32); lg[self.V + 1] = 50.0
        if self.script: lg[self.script.pop(0)] = 5.0
        return lg

    def forward(self, tokens, positions): self.calls.append((list(tokens), list(positions))); return self._lg()


def test_free_decoding_is_greedy_stops_at_the_end_token_and_stays_in_the_vocabulary():
    c = Scripted([4, 5, 9]); lg = c._lg(); toks, ended = D.free_decode(c, lg, 10, 9, 7, 20); assert toks == [4, 5, 9] and ended is True and c.calls == [([4], [7]), ([5], [8])]
    c = Scripted([1, 2, 3, 4, 5, 6]); toks, ended = D.free_decode(c, c._lg(), 10, 9, 0, 3); assert toks == [1, 2, 3] and ended is False and len(c.calls) == 2
    c = Scripted([9]); toks, ended = D.free_decode(c, c._lg(), 10, 9, 0, 5); assert toks == [9] and ended is True and c.calls == []
    c = Scripted([]); toks, ended = D.free_decode(c, c._lg(), 10, 9, 0, 2); assert toks == [0, 0] and ended is False                                    # nothing above zero in the vocabulary: argmax is the first, never index 11


# ---------------------------------------------------------------- 6 the runner on the stand-in
class Spy(TinyCache):
    def forward(self, tokens, positions):
        if self.n == 0: self.e.firsts.append(list(tokens))
        return super().forward(tokens, positions)


def _engine():
    e = TinyEngine(); e.cache_class = Spy; e.firsts = []; return e


def test_a_unit_gives_every_question_a_fresh_prompt_with_the_state_and_the_question_and_nothing_of_the_truth():
    ps = D.make_problems(seed=5, **TINY); e = _engine(); p = ps[0]
    row = D.run_unit(e, KIT, TAB, p, "bitset", max_new=3); assert row["rho"] == D.density(p["graph"]) and row["id"] == "d0/bitset" and row["pid"] == "d0" and row["regime"] == "dense" and row["fmt"] == "bitset" and row["decodes"] is True and len(row["qs"]) == 8
    state = D.render(p["graph"], "bitset"); assert len(e.firsts) == 8 and row["state_tokens"] == len(KIT.enc(state)) == len(state.encode())
    for q, toks, rec in zip(p["questions"], e.firsts, row["qs"]):
        assert toks == KIT.enc(D.prompt_text(state, q["text"])) and rec["prompt_tokens"] == len(toks) and rec["qid"] == q["qid"] and rec["kind"] == q["kind"]
        assert {k: v for k, v in rec.items() if k in D.score_question(q, rec["text"], len(p["graph"].nodes))} == D.score_question(q, rec["text"], len(p["graph"].nodes))   # scored with the graph's size
        text = KIT.dec(toks); assert D.DENSE_SYSTEM in text and "\n\nQuestion: " in text and state in text and q["text"] in text and text.endswith("<|im_start|>assistant\n") and text.count("<|im_start|>") == 3
        assert set(rec) >= {"qid", "kind", "text", "ended", "prompt_tokens", "seconds", "peak_mb"} and (rec["kind"] != "exist" or {"answer", "correct", "parsed"} <= set(rec)) and (rec["kind"] == "exist" or {"precision", "recall", "f1", "exact", "wrong", "baseline_f1"} <= set(rec))
    bad = dict(p); g = D.make_graph(random.Random(9), 5, 6, (3, 3)); bad["graph"] = g; bad["questions"] = []; assert D.run_unit(e, KIT, TAB, bad, "matrix", 2)["qs"] == []
    raises(AssertionError, D.run_unit, e, KIT, TAB, p, "csv", 2)


class FakeCache:
    """Logs every forward call; the first logits point at token 3, the next ones at the end token 6."""
    def __init__(self, log): self.log, self.n = log, 0

    def forward(self, tokens, positions):
        self.log.append((list(tokens), [int(p) for p in positions])); self.n += len(tokens); lg = np.zeros(10, np.float32); lg[3 if len(self.log) % 2 == 1 else 6] = 5.0; return lg


class FakeEngine:
    def __init__(self): self.log, self.caches = [], 0
    def new_cache(self): self.caches += 1; return FakeCache(self.log)
    def reset_peak(self): pass
    def sync(self): pass
    def peak_mb(self): return 1.5


def test_a_unit_counts_the_tokenizers_tokens_decodes_from_the_end_of_the_prompt_and_keeps_the_whole_reply():
    kit = type("Kit", (), dict(enc=staticmethod(lambda s: [ord(c) % 5 for c in s][::3]), dec=staticmethod(lambda ids: "tok:" + ",".join(map(str, ids))), eos=6))(); tab = type("Tab", (), dict(V=10))()
    p = D.make_problems(seed=5, **TINY)[0]; e = FakeEngine(); row = D.run_unit(e, kit, tab, p, "matrix", max_new=5); state = D.render(p["graph"], "matrix")
    assert row["state_tokens"] == len(kit.enc(state)) != len(state) and e.caches == 8 and len(e.log) == 16
    for i, q in enumerate(p["questions"]):
        toks = kit.enc(D.prompt_text(state, q["text"])); prefill, step = e.log[2 * i], e.log[2 * i + 1]; rec = row["qs"][i]
        assert prefill == (toks, list(range(len(toks)))) and step == ([3], [len(toks)]) and rec["prompt_tokens"] == len(toks) and rec["text"] == "tok:3,6" and rec["ended"] is True and rec["peak_mb"] == 1.5
    assert len(kit.enc(D.prompt_text(state, p["questions"][0]["text"]))) != len(D.prompt_text(state, p["questions"][0]["text"]))


class FakeTok:
    """A tokenizer that builds the ChatML text for two messages (system, user) as Qwen's template does, or something else when `broken`."""
    def __init__(self, broken=False): self.broken = broken

    def apply_chat_template(self, msgs, add_generation_prompt=True, tokenize=False):
        assert [m["role"] for m in msgs] == ["system", "user"] and add_generation_prompt and not tokenize
        sep = "\n" if not self.broken else " "; return f"<|im_start|>system{sep}{msgs[0]['content']}<|im_end|>\n<|im_start|>user{sep}{msgs[1]['content']}<|im_end|>\n<|im_start|>assistant\n"


def test_the_preflight_on_the_tokenizer_passes_and_each_of_its_checks_can_fail():
    ps = D.make_problems(seed=5, **TINY); checks = D.dense_preflight(FakeTok(), KIT, ps)
    assert len(checks) == 3 and all(ok for _, ok, _ in checks) and "5 formats compared" in checks[0][2] and "10 states" in checks[1][2] and "10 states" in checks[2][2]
    bad = D.dense_preflight(FakeTok(broken=True), KIT, ps); assert [ok for _, ok, _ in bad] == [False, True, True] and "first difference at character" in bad[0][2] and bad[0][2].startswith("tuples:")
    short = type("Kit", (), dict(enc=staticmethod(lambda s: list(range(max(1, len(s.split()) // 4))))))(); under = D.dense_preflight(FakeTok(), short, ps); assert [ok for _, ok, _ in under] == [True, False, True] and "below the bound" in under[1][2]
    import axi.experiments.kv_dense as KD
    orig = KD.decode_state
    try:
        KD.decode_state = lambda text, fmt: set(); wrong = D.dense_preflight(FakeTok(), KIT, ps); assert [ok for _, ok, _ in wrong] == [True, True, False] and "wrong:" in wrong[2][2]
    finally: KD.decode_state = orig


def test_the_runner_writes_one_line_per_graph_and_format_resumes_and_refuses_another_run():
    ps = D.make_problems(seed=5, **TINY); tmp = tempfile.mkdtemp(); path = os.path.join(tmp, "d.jsonl"); e = TinyEngine()
    res = D.run_dense(e, KIT, TAB, path, problems=ps, max_new=3, meta=dict(model="stand-in")); rows = res["rows"]
    assert [r["id"] for r in rows] == [f"{p['id']}/{f}" for p in ps for f in D.DENSE_FORMATS] and len(rows) == 10 and res["meta"]["kind"] == "kv_dense" and res["meta"]["units"] == 10 and res["meta"]["model"] == "stand-in" and res["meta"]["fp"] == D.fingerprint(ps)
    lines = open(path).read().splitlines(); assert len(lines) == 11 and "_meta" in json.loads(lines[0]) and [json.loads(x)["id"] for x in lines[1:]] == [r["id"] for r in rows]
    calls = e.calls; again = D.run_dense(e, KIT, TAB, path, problems=ps, max_new=3, meta=dict(model="stand-in")); assert e.calls == calls and again["rows"] == rows and open(path).read().splitlines() == lines          # nothing is run twice
    p2 = os.path.join(tmp, "d2.jsonl"); e2 = TinyEngine(); e2.fail_after = 150                                           # a session that drops half-way
    try: D.run_dense(e2, KIT, TAB, p2, problems=ps, max_new=3, meta=dict(model="stand-in"))
    except RuntimeError as err: assert "dropped" in str(err)
    else: raise AssertionError("the dropped session did not raise")
    part = open(p2).read().splitlines(); assert 1 < len(part) < 11
    with open(p2, "a") as f: f.write('{"id": "half-writ')                                                                         # a half-written last line
    e3 = TinyEngine(); res3 = D.run_dense(e3, KIT, TAB, p2, problems=ps, max_new=3, meta=dict(model="stand-in")); strip = lambda r: dict(r, qs=[{k: v for k, v in q.items() if k != "seconds"} for q in r["qs"]])
    assert [r["id"] for r in res3["rows"]] == [r["id"] for r in rows] and [strip(r) for r in res3["rows"]] == [strip(r) for r in rows]                       # the same answers as the run that was not interrupted
    assert [x["id"] for x in map(json.loads, open(p2).read().splitlines()[1:])] == [r["id"] for r in rows]
    raises(ValueError, D.run_dense, e, KIT, TAB, path, problems=ps, max_new=4, meta=dict(model="stand-in"))                    # other settings
    raises(ValueError, D.run_dense, e, KIT, TAB, path, problems=D.make_problems(seed=6, **TINY), max_new=3, meta=dict(model="stand-in"))
    raises(ValueError, D.run_dense, e, KIT, TAB, path, problems=ps, formats=("tuples", "matrix"), max_new=3, meta=dict(model="stand-in"))
    sub = D.run_dense(e, KIT, TAB, os.path.join(tmp, "d3.jsonl"), problems=ps, formats=("tuples", "bitset"), max_new=2); assert [r["fmt"] for r in sub["rows"]] == ["tuples", "bitset"] * 2 and sub["meta"]["units"] == 4


def _q(qid, kind, **kw):
    base = dict(qid=qid, kind=kind, text="", ended=True, prompt_tokens=100, seconds=1.0, peak_mb=10.0)
    if kind == "exist": base.update(answer="yes", correct=True, parsed=True)
    else: base.update(precision=1.0, recall=1.0, f1=1.0, exact=True, listed=3, wrong=0, baseline_f1=0.5)
    base.update(kw); return base


def _row(pid, regime, fmt, qs, tokens=1000, decodes=True): return dict(id=f"{pid}/{fmt}", pid=pid, regime=regime, fmt=fmt, state_tokens=tokens, decodes=decodes, rho=0.5, qs=qs)


def test_the_summary_adds_up_by_hand():
    qs1 = [_q("q0", "out", recall=0.5, precision=1.0, f1=2 / 3, exact=False, wrong=0, baseline_f1=0.8), _q("q1", "in", recall=0.0, precision=0.0, f1=0.0, exact=False, wrong=2, baseline_f1=0.6, ended=False),
           _q("q2", "exist", correct=True), _q("q3", "exist", correct=False, answer=None, parsed=False, prompt_tokens=300, seconds=3.0, peak_mb=30.0)]
    qs2 = [_q("q0", "out", recall=1.0, precision=0.5, f1=2 / 3, exact=False, wrong=1, baseline_f1=0.4), _q("q1", "in", recall=0.5, precision=1.0, f1=2 / 3, exact=False, wrong=0, baseline_f1=0.6),
           _q("q2", "exist", correct=True), _q("q3", "exist", correct=True)]
    rows = [_row("d0", "dense", "tuples", qs1, tokens=1000), _row("d1", "dense", "tuples", qs2, tokens=3000), _row("s0", "sparse", "tuples", qs2, tokens=500, decodes=False)]
    s = D.summarize_dense(rows); t = s["dense"]["tuples"]; assert set(s) == {"dense", "sparse"} and list(s["dense"]) == ["tuples"]
    assert t["graphs"] == 2 and t["n_list"] == 4 and t["n_exist"] == 4 and t["decodes"] is True and s["sparse"]["tuples"]["decodes"] is False and t["state_tokens"] == 2000 and t["qids"] == ["d0/q0", "d0/q1", "d0/q2", "d0/q3", "d1/q0", "d1/q1", "d1/q2", "d1/q3"]
    assert abs(t["f1"] - np.mean([2 / 3, 0, 2 / 3, 2 / 3])) < 1e-12 and abs(t["f1_out"] - 2 / 3) < 1e-12 and abs(t["f1_in"] - 1 / 3) < 1e-12 and t["recall_out"] == 0.75 and t["recall_in"] == 0.25 and t["recall"] == 0.5
    assert abs(t["precision"] - np.mean([1, 0, 0.5, 1])) < 1e-12 and t["exact"] == 0.0 and abs(t["baseline_f1"] - np.mean([0.8, 0.6, 0.4, 0.6])) < 1e-12 and t["wrong"] == np.mean([0, 2, 1, 0])
    assert t["exist_acc"] == 0.75 and t["exist_unparsed"] == 0.25 and abs(t["ran_on"] - 1 / 8) < 1e-12 and t["prompt_tokens"] == np.mean([100, 100, 100, 300, 100, 100, 100, 100]) and t["seconds"] == np.mean([1, 1, 1, 3, 1, 1, 1, 1]) and t["peak_mb"] == 30.0
    assert D.summarize_dense([]) == {} and set(D.summarize_dense(rows[:1])["dense"]) == {"tuples"}
    mixed = D.summarize_dense([_row("d0", "dense", "tuples", qs1, decodes=True), _row("d1", "dense", "tuples", qs2, decodes=False)]); assert mixed["dense"]["tuples"]["decodes"] is False                # one graph that does not read back is enough


# ---------------------------------------------------------------- 7 the reading rules
def _fm(tokens, exist, f1, **kw):
    d = dict(n_list=20, n_exist=20, qids=[f"q{i}" for i in range(40)], decodes=True, state_tokens=tokens, exist_acc=exist, f1=f1, recall_in=0.4, recall_out=0.8, baseline_f1=0.83); d.update(kw); return d


def _summary(**o):
    """A summary on which all fourteen lines read yes; `o` changes single numbers: o['dense'] = {format: {field: value}}, o['sparse'] likewise."""
    dense = {"tuples": _fm(5000, 0.75, 0.70, recall_in=0.40, recall_out=0.80), "adjacency": _fm(4200, 0.80, 0.90, recall_in=0.90), "matrix": _fm(1400, 0.60, 0.50), "binary": _fm(900, 0.55, 0.45), "bitset": _fm(700, 0.50, 0.30)}
    sq = lambda **kw: dict(_fm(1800, 0.80, 0.70, n_list=12, n_exist=12, qids=[f"q{i}" for i in range(24)]), **kw)
    sparse = {"tuples": sq(), "adjacency": sq(f1=0.85), "matrix": sq(exist_acc=0.60), "binary": sq(exist_acc=0.55), "bitset": sq(exist_acc=0.50, f1=0.30)}
    for reg, tab in (("dense", dense), ("sparse", sparse)):
        for f, ch in o.get(reg, {}).items(): tab[f].update(ch)
    return {"dense": dense, "sparse": sparse}


def _flip(**o): return [v for _, v in D.verdict_dense(_summary(**o))]


def test_the_fourteen_reading_rules_flip_exactly_at_their_thresholds():
    assert _flip() == [True] * 14 and len(D.DENSE_LINES) == 14 and [t for t, _ in D.verdict_dense(_summary())] == [t for _, t in D.DENSE_LINES]
    def only(i, **o):                                                                                                      # the line i (1-based) is the only one that changes against the all-yes summary
        got = _flip(**o); want = [True] * 14; want[i - 1] = False
        extra = [k + 1 for k in range(14) if got[k] != want[k]]; assert not extra, (i, o, extra)
    t = lambda **kw: {"dense": {"tuples": kw}}
    only(1, dense={"matrix": dict(qids=[f"q{i}" for i in range(39)] + ["qX"])}); only(1, dense={"adjacency": dict(n_exist=19)}); only(1, sparse={"tuples": dict(n_list=13)})
    s = _summary(); del s["sparse"]["matrix"]; assert D.verdict_dense(s)[0][1] is False and D.verdict_dense(s)[12][1] is None   # not every format run in both regimes
    s = _summary(); del s["sparse"]; assert D.verdict_dense(s)[0][1] is False and D.verdict_dense(s)[13][1] is None                  # a whole regime missing
    s = _summary(); s["extra"] = s["sparse"]; assert D.verdict_dense(s)[0][1] is False                                                 # a regime that should not be there
    short = [f"q{i}" for i in range(39)]; only(1, dense={f: dict(qids=short) for f in D.DENSE_FORMATS})                              # the counts say 40 but the ids say 39
    only(2, dense={"bitset": dict(decodes=False)}); only(2, sparse={"adjacency": dict(decodes=False)})
    assert _flip(dense={"tuples": dict(exist_acc=0.60)})[2] is True; got = _flip(dense={"tuples": dict(exist_acc=0.5999)})
    assert got[2] is False                                                                                                 # line 3 drops just under 0.60
    assert _flip(dense={"matrix": dict(state_tokens=1750.0)})[3] is True and _flip(dense={"matrix": dict(state_tokens=1751.0)})[3] is False
    assert _flip(dense={"bitset": dict(state_tokens=1000.0)})[4] is True and _flip(dense={"bitset": dict(state_tokens=1001.0)})[4] is False
    assert _flip(dense={"adjacency": dict(recall_in=0.55)})[5] is True and _flip(dense={"adjacency": dict(recall_in=0.549)})[5] is False
    assert _flip(dense={"tuples": dict(recall_out=0.50)})[6] is True and _flip(dense={"tuples": dict(recall_out=0.499)})[6] is False
    assert _flip(dense={"matrix": dict(exist_acc=0.70)})[7] is True and _flip(dense={"matrix": dict(exist_acc=0.71)})[7] is False
    assert _flip(dense={"bitset": dict(exist_acc=0.60)})[8] is True and _flip(dense={"bitset": dict(exist_acc=0.61)})[8] is False
    assert _flip(dense={"bitset": dict(f1=0.829)})[9] is True and _flip(dense={"bitset": dict(f1=0.83)})[9] is False and _flip(dense={"bitset": dict(f1=0.9)})[9] is False
    both = lambda e, f: _flip(dense={"bitset": dict(exist_acc=e, f1=f)})[10]
    assert both(0.65, 0.60) is True and both(0.66, 0.60) is False and both(0.65, 0.61) is False and both(0.66, 0.61) is False
    assert _flip(dense={"matrix": dict(f1=0.90)})[11] is True and _flip(dense={"matrix": dict(f1=0.91)})[11] is False and _flip(dense={"tuples": dict(f1=0.95)})[11] is False and _flip(dense={"bitset": dict(f1=0.95)})[11] is False
    assert _flip(sparse={"matrix": dict(exist_acc=0.80)})[12] is True and _flip(sparse={"matrix": dict(exist_acc=0.81)})[12] is False
    assert _flip(sparse={"adjacency": dict(f1=0.70)})[13] is True and _flip(sparse={"adjacency": dict(f1=0.69)})[13] is False


def test_a_line_is_not_run_when_its_numbers_are_missing_and_the_printout_says_so():
    none = D.verdict_dense({}); assert [v for _, v in none] == [None] * 14
    s = _summary(); only_t = {"dense": {"tuples": s["dense"]["tuples"]}}; got = [v for _, v in D.verdict_dense(only_t)]
    assert got == [False, True, True, None, None, None, True, None, None, None, None, None, None, None]                    # line 7 needs tuples only; line 1 is False: four formats were not run
    s2 = _summary(); del s2["dense"]["bitset"]; got = [v for _, v in D.verdict_dense(s2)]; assert got[4] is None and got[8] is None and got[9] is None and got[10] is None and got[11] is None and got[3] is True
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): D.print_verdict_dense(D.verdict_dense(_summary(dense={"bitset": dict(exist_acc=0.9)})), "TITLE"); D.print_verdict_dense(D.verdict_dense(only_t), "PART")
    out = buf.getvalue(); assert "-- TITLE (reading rules fixed before the run) --" in out and out.count("  yes      ") >= 10 and "  NO       " in out and "  not run  " in out and " 1. Every format answered" in out and "14. sparse: adjacency's mean F1" in out


def test_the_table_prints_every_format_and_regime():
    ps = D.make_problems(seed=5, **TINY); res = D.run_dense(TinyEngine(), KIT, TAB, os.path.join(tempfile.mkdtemp(), "d.jsonl"), problems=ps, max_new=3); buf = io.StringIO()
    with contextlib.redirect_stdout(buf): D.print_dense(res["summary"], "stand-in")
    out = buf.getvalue(); assert "dense-graph test stand-in" in out and "dense graphs:" in out and "sparse graphs:" in out and all(f in out for f in D.DENSE_FORMATS) and "all-F1" in out and "exist ok" in out
    checks = D.verdict_dense(res["summary"]); assert len(checks) == 14 and checks[1][1] is True                                    # the stand-in cannot answer; only the instrument lines mean anything on it


# ---------------------------------------------------------------- 8 the page
def _head():
    text = open(PAGE).read(); assert text.count(RESULT_MARK) <= 1
    return text[:text.index(RESULT_MARK)] if RESULT_MARK in text else text


def test_the_page_is_locked_and_carries_the_fourteen_lines_the_settings_and_the_repository_facts_it_states():
    head = _head(); assert _h(head) == PAGE_SHA256, "the predictions and reading rules must not change after the run; later notes go below them"
    for i, (tag, line) in enumerate(D.DENSE_LINES, 1): assert f"\n{i}. [{tag}] {line} (" in head, i
    probs = [int(m.group(1)) for m in re.finditer(r"\n\d+\. \[\w+\] .*\((\d+) %\)(?=\n)", head)]; assert probs == [99, 99, 80, 90, 90, 85, 75, 72, 80, 80, 60, 65, 75, 80]
    prod = 100.0
    for i in D.DENSE_HEADLINE: prod *= probs[i - 1] / 100
    assert abs(prod - 39) <= 0.5 and "about 39 %" in head and "about **45 %**" in head and "**Headline** (lines 4, 5, 9 and 11)" in head
    assert (D.DENSE_SEED, D.DENSE_NODES, D.DENSE_GRID, D.DENSE_MAX_NEW) == (51, 24, (5, 5), 160) and D.DENSE_REGIMES == (("dense", 5, 400), ("sparse", 3, 138)) and D.DENSE_FORMATS == ("tuples", "adjacency", "matrix", "binary", "bitset") and D.DENSE_QUESTIONS == (("out", 2), ("in", 2), ("exist", 4))
    assert (D.N_EXIST_MIN, D.N_TOK_MATRIX, D.N_TOK_BITSET, D.N_IN_GAIN, D.N_OUT_OVER_IN, D.N_MATRIX_BELOW, D.N_BITSET_CHANCE, D.N_BITSET_BELOW) == (0.60, 0.35, 0.20, 0.15, 0.10, 0.05, 0.60, 0.10) and D.DENSE_HEADLINE == (4, 5, 9, 11)
    for needle in ("Concept-map item 101", "`python/axi/experiments/kv_dense.py`", "400 / (24 x 23) = 0.725", "Seed 51", "138 edges, rho = 0.25", "160 new tokens at most", "64 questions per format", "I do not attribute them to him", "Qwen2.5-7B-Instruct 4-bit",
                   "\"explicitly defines a relational density threshold\"", "\"streams coordinate tuples to the model\"", "\"text serialization causes catastrophic token bloat and attention confusion\"",
                   "\"dynamically cuts over to Packed Bit-Matrix Representations (64-bit integer bitsets)\"", "\"a massive, unmeasured efficiency lever\"", "`crates/axi-ir/src/router.rs`", "`DensityRouter`", "`SparseTuples`", "`PackedBitMatrix`",
                   "(services + edges) divided by the cells of the grid", "`python/axi/wire.py`", "`python/tests/test_wire.py`", "1/128", "What a model can read of either form has not been measured anywhere in the repository",
                   "Fatemi, Halcrow and Perozzi", "Wang et al.", "NLGraph", "no model was run on any question", "no accuracy or F1 was computed from anything", "`python/colab/axi_dense_test.ipynb`", "a separate one, so that the notebook of the layout and block tests is untouched", "a tie for the highest F1 counts", "`kv_dense.verdict_dense`", "axi_dense.json",
                   "| dense (rho 0.725) | 4,982 | 4,248 | 1,459 | 918 | 743 |", "| sparse (rho 0.25) | 1,838 | 1,628 | 1,459 | 918 | 789 |", "0.83 (mean list of 16.5 services)", "0.36 (mean list of 5.2)", "20: 422 / 469 / 791; 60: 902 / 851 / 790; 138: 1,838 / 1,628 / 789; 300: 3,782 / 3,248 / 767; 400: 4,982 / 4,248 / 742"):
        assert needle in head, needle


def test_the_repository_facts_the_page_states_are_true():
    router = open(os.path.join(ROOT, "crates", "axi-ir", "src", "router.rs")).read()
    assert "Self { threshold: 0.65 }" in router and "(nodes.len() + edges.len()) as f64 / max_entities" in router and "SparseTuples" in router and "PackedBitMatrix" in router and "let max_entities = (rows * cols) as f64" in router
    lib = open(os.path.join(ROOT, "crates", "axi-ir", "src", "lib.rs")).read(); assert "pub use router::{DensityRouter, RepresentationFormat};" in lib
    users = []
    for base, _, files in os.walk(ROOT):
        if "/.git" in base or "/target" in base or "__pycache__" in base: continue
        for name in files:
            if name.endswith((".rs", ".py")) and name != "router.rs":
                path = os.path.join(base, name)
                try: txt = open(path, errors="ignore").read()
                except OSError: continue
                if ("DensityRouter" in txt or "RepresentationFormat" in txt) and not path.endswith(("test_kv_dense.py", "kv_dense.py")) and not path.endswith("lib.rs"): users.append(path)
    assert not users, users                                                                                                         # "no caller"
    wire = open(os.path.join(ROOT, "python", "tests", "test_wire.py")).read(); assert "def test_density_threshold_is_1_over_128_not_0_65" in wire
    assert os.path.exists(os.path.join(ROOT, "python", "axi", "wire.py"))



# ---------------------------------------------------------------- 9 the notebook, end to end on the stand-in
NB = os.path.join(ROOT, "python", "colab", "axi_dense_test.ipynb")


def _run_dense_notebook(eng, tok, tmp, until=None):
    """Execute the code cells of the notebook with Colab and torch replaced by stubs, the model builder by the numpy stand-in, the graphs by two tiny ones and the reply length by 4 tokens."""
    import types
    import axi.experiments.kv_torch as KT, colab.colab_persist as CP
    downloads = []; saved = {k: getattr(D, k) for k in ("make_problems", "run_unit")}
    mods = {k: sys.modules.get(k) for k in ("google", "google.colab", "torch")}; orig = (KT.build_kv, KT.make_hf_kit, CP.open_store)
    gc = types.ModuleType("google.colab"); gc.files = types.SimpleNamespace(upload=lambda: {"axi-colab-tensor.zip": b""}, download=lambda path: downloads.append(path)); gc.drive = types.SimpleNamespace(mount=lambda mp: None)
    g = types.ModuleType("google"); g.colab = gc; tm = types.ModuleType("torch"); tm.cuda = types.SimpleNamespace(memory_allocated=lambda: 0, max_memory_allocated=lambda: 0, empty_cache=lambda: None)
    real_problems, real_unit = D.make_problems, D.run_unit
    try:
        sys.modules.update({"google": g, "google.colab": gc, "torch": tm})
        D.make_problems = lambda *a, **k: real_problems(seed=5, **TINY); D.run_unit = lambda engine, kit, table, problem, fmt, max_new=D.DENSE_MAX_NEW: real_unit(engine, kit, table, problem, fmt, 4)
        KT.build_kv = lambda name, load_4bit, device: (eng, tok, TAB, None); KT.make_hf_kit = lambda tk, hint_rules=False: byte_kit(hint_rules)
        CP.open_store = lambda local_dir="/content/out", subdir="x", mount_point="/content/drive": orig[2](tmp + "/out", subdir, tmp + "/drive")
        ns = {"__name__": "__main__"}; out = io.StringIO(); nb = json.load(open(NB)); n = 0
        for c in nb["cells"]:
            if c["cell_type"] != "code": continue
            src = "".join(l for l in c["source"] if not l.lstrip().startswith("!")).replace("/content/out", tmp + "/out")
            if not src.strip() or "files.upload" in src: continue
            n += 1
            if until is not None and n > until: break
            with contextlib.redirect_stdout(out): exec(compile(src, "cell %d" % n, "exec"), ns)
        ns["_printed"] = out.getvalue(); return ns, downloads
    finally:
        for k, v in saved.items(): setattr(D, k, v)
        KT.build_kv, KT.make_hf_kit, CP.open_store = orig
        for k, v in mods.items():
            if v is None: sys.modules.pop(k, None)
            else: sys.modules[k] = v


def test_the_notebook_is_clean_saves_no_outputs_holds_no_key_and_runs_the_safety_check_before_the_long_run():
    nb = json.load(open(NB)); cells = nb["cells"]; code = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]; allsrc = "\n".join("".join(c["source"]) for c in cells); joined = "\n".join(code)
    assert all(c["cell_type"] != "code" or (c.get("outputs") == [] and c.get("execution_count") is None) for c in cells) and not re.search(r"AIza[0-9A-Za-z_\-]{10,}|hf_[A-Za-z0-9]{20,}", allsrc) and "userdata" not in joined
    for needle in ("docs/DENSE_TEST.md", "axi-colab-tensor.zip", "T4", "Do not paste", "$0", "14 reading rules", "axi_dense.json", "64-bit integer", "does NOT test", "packed bit matrices"): assert needle in allsrc, needle
    assert "build_kv(" in joined and "make_hf_kit(tok, hint_rules=False)" in joined and "dense_preflight(tok, kit_plain, dense_problems)" in joined and "assert all(ok for _, ok, _ in checks)" in joined and "STOP: a check failed" in joined
    assert "run_dense(" in joined and "verdict_dense(" in joined and "print_verdict_dense(" in joined and "open_store(" in joined and "dense_dry_run(" in joined and "files.download(" in joined
    assert joined.index("dense_preflight(") < joined.index("run_unit(") < joined.index("run_dense(") and joined.count("files.download(") == 1 and "KV_MODEL" in joined and "load_4bit" not in joined
    run_cells = [c for c in code if "run_dense(" in c and "files.download" in c]; assert len(run_cells) == 1 and "kit_plain" in run_cells[0] and "store.restore(" in run_cells[0] and "store.progress(" in run_cells[0] and "store.persist(" in run_cells[0]
    smoke = [c for c in code if "run_unit(" in c]; assert len(smoke) == 1 and '"tuples", "bitset"' in smoke[0] and "dense_problems[0]" in smoke[0]
    src = open(D.__file__).read(); assert "torch" not in src and "import transformers" not in src and "import google" not in src
    assert "KV_MODEL" in open(os.path.join(HERE, "..", "axi", "experiments", "kv_drop.py")).read()
    from axi.experiments.kv_drop import KV_MODEL; assert "Qwen2.5-7B-Instruct" in KV_MODEL and "Qwen2.5-7B-Instruct" in allsrc


def test_every_code_cell_of_the_notebook_runs_on_the_stand_in_saves_after_each_run_and_stops_when_a_check_fails():
    from axi.experiments.kv_drop import KV_MODEL
    tmp = tempfile.mkdtemp(); ns, downloads = _run_dense_notebook(TinyEngine(), FakeTok(), tmp)
    assert [os.path.basename(d) for d in downloads] == ["axi_dense.json"]
    pre = json.load(open(tmp + "/out/axi_dense_preflight.json")); assert len(pre) == 3 and all(x["ok"] for x in pre) and [set(x) for x in pre] == [{"name", "ok", "detail"}] * 3
    for name in ("axi_dense_preflight.json", "axi_dense.jsonl", "axi_dense.json"): assert os.path.exists(tmp + "/drive/MyDrive/axi_dense/" + name), name
    res = json.load(open(tmp + "/out/axi_dense.json")); rows = res["rows"]; ps = D.make_problems(seed=5, **TINY)
    assert res["meta"]["kind"] == "kv_dense" and res["meta"]["model"] == KV_MODEL and res["meta"]["units"] == 10 and res["meta"]["max_new"] == D.DENSE_MAX_NEW == 160 and res["meta"]["formats"] == list(D.DENSE_FORMATS) and res["meta"]["fp"] == D.fingerprint(ps)
    assert [r["id"] for r in rows] == [f"{p['id']}/{f}" for p in ps for f in D.DENSE_FORMATS] and all(len(r["qs"]) == 8 and r["decodes"] for r in rows) and res["summary"] == D.summarize_dense(rows)
    assert len(open(tmp + "/out/axi_dense.jsonl").read().splitlines()) == 11 and open(tmp + "/out/axi_dense.jsonl").read() == open(tmp + "/drive/MyDrive/axi_dense/axi_dense.jsonl").read()          # the copy on Drive is the finished file
    fd = ns["final_dense"]; assert len(fd) == 14 and [t for t, _ in fd] == [t for _, t in D.DENSE_LINES] and fd == D.verdict_dense(ns["res_dense"]["summary"]) and ns["res_dense"]["summary"] == res["summary"]
    sm = ns["smoke"]; assert [(r["pid"], r["fmt"]) for r in sm] == [("d0", "tuples"), ("d0", "bitset")] and all(len(r["qs"]) == 8 and r["wall"] >= 0 for r in sm)
    txt = ns["_printed"]; assert "ALL CHECKS OK" in txt and "REAL tokens of the state" in txt and "== dense-graph test DENSE TEST ==" in txt and "-- DENSE TEST (reading rules fixed before the run)" in txt and "One graph in one format took about" in txt
    assert re.search(r"\n  dense  rho 0\.\d{3}: tuples \d+ \(1\.00 x tuples\)", txt) and re.search(r"\n  sparse rho 0\.\d{3}: tuples \d+ \(1\.00 x tuples\)", txt); assert all(f in txt for f in D.DENSE_FORMATS) and "progress file:" in txt and "peak GPU memory (GB)" in txt and "ok    " in txt and "FAIL" not in txt
    for i in range(1, 15): assert f"{i:2d}. " in txt, i
    part, dl = _run_dense_notebook(TinyEngine(), FakeTok(), tempfile.mkdtemp(), until=2); assert dl == [] and "smoke" not in part and "res_dense" in part and part["res_dense"] is None                    # up to the model load: nothing was run
    broken_tmp = tempfile.mkdtemp()
    try: _run_dense_notebook(TinyEngine(), FakeTok(broken=True), broken_tmp)
    except AssertionError as err: assert "STOP: a check failed" in str(err)
    else: raise AssertionError("the notebook went on after a failed check")
    saved = json.load(open(broken_tmp + "/out/axi_dense_preflight.json")); assert [x["ok"] for x in saved] == [False, True, True] and sorted(os.listdir(broken_tmp + "/out")) == ["axi_dense_preflight.json"]
    assert os.listdir(broken_tmp + "/drive/MyDrive/axi_dense") == ["axi_dense_preflight.json"]


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("PASS", n)
