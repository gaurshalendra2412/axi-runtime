"""Does the way a DENSE graph is written change what a stock model can read from it? Page with the predictions written BEFORE any run: docs/DENSE_TEST.md (concept-map item 101).

Where the idea comes from: a text the owner pasted on 15 Oct 2026 says the framework "dynamically cuts over to Packed Bit-Matrix Representations (64-bit integer bitsets)" when the relation density
rho = |E| / (|V| (|V| - 1)) passes 0.65, because "text serialization causes catastrophic token bloat and attention confusion", and calls the cut-over "a massive, unmeasured efficiency lever". Nobody has
measured it. The part a stock model allows is the reading side: write the SAME dense graph in five ways and ask the SAME questions.

One graph = 24 services on a 5 x 5 grid (one cell empty), either DENSE (400 edges, rho = 0.725, the example of the text) or SPARSE (138 edges, rho = 0.25). Five formats of the same facts:
  tuples      the repository's text: NODES [r,c:v] ... and EDGES (r,c)->(r,c):dep#w ... (carries relation and weight; the others carry presence only)
  adjacency   one line per service: the services it has an edge to (out) and the services that have an edge to it (in); every edge appears twice
  matrix      the adjacency matrix as rows of 0 and 1 separated by spaces, with the column order stated
  binary      the same rows as unbroken strings of 0 and 1
  bitset      each row packed as one 64-bit integer in hexadecimal (bit j set = an edge to column j), with the column order and the bit order stated: the format of the text
Eight questions per graph, whose answers need presence only: the services a given service has an edge to (2), the services that have an edge to a given service (2), and "is there an edge from A to B?"
(4: two that exist, two that do not). Each format answers the same questions; greedy decoding, no grammar, 160 new tokens at most; the model reads the whole prompt afresh each time (no cache is kept)."""
import hashlib
import json
import os
import random
import re
import time
from math import comb
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from axi.engine.cico_parser import CICOParser
from axi.engine.gate import Graph
from axi.experiments.kv_drop import IM_END, IM_START, Tracked, _fx, _ge, _mean, _pc
from axi.experiments.resumable import _roundtrip, load_rows

DENSE_SEED = 51
DENSE_NODES, DENSE_GRID = 24, (5, 5)
DENSE_REGIMES = (("dense", 5, 400), ("sparse", 3, 138))          # (name, graphs, edges)
DENSE_FORMATS = ("tuples", "adjacency", "matrix", "binary", "bitset")
DENSE_QUESTIONS = (("out", 2), ("in", 2), ("exist", 4))          # per graph
DENSE_MAX_NEW = 160
DENSE_SYSTEM = ("You answer questions about a directed graph of services. Use only the graph given in the message. "
                "Give only the answer asked for: services written as (r,c) separated by spaces for a list, or the single word yes or no.")

# thresholds of the reading rules (docs/DENSE_TEST.md)
N_EXIST_MIN, N_TOK_MATRIX, N_TOK_BITSET, N_IN_GAIN, N_OUT_OVER_IN, N_MATRIX_BELOW, N_BITSET_CHANCE, N_BITSET_BELOW = 0.60, 0.35, 0.20, 0.15, 0.10, 0.05, 0.60, 0.10
DENSE_HEADLINE = (4, 5, 9, 11)
DENSE_LINES = [
    ("instrument", "Every format answered the same 64 questions (40 on the dense graphs, 24 on the sparse ones), and the questions of one graph are the same in every format."),
    ("instrument", "In every prompt the state, read back by the decoder of the format, holds exactly the graph's edges (all 40 graph-and-format pairs)."),
    ("guard", "dense, tuples: the share of the 20 existence questions answered right is at least 0.60 (the model can read this text at all)."),
    ("claim", "dense: the `matrix` prompt holds at most 0.35 times the tokens of the `tuples` prompt (mean over the 5 graphs, by the pre-tokenizer count of the dry run, then by the tokenizer's own count in the run)."),
    ("claim", "dense: the `bitset` prompt holds at most 0.20 times the tokens of the `tuples` prompt (same count)."),
    ("claim", "dense: adjacency's recall on the 'who has an edge to X' questions is at least 0.15 above tuples'."),
    ("claim", "dense: tuples' recall on the 'who does X have an edge to' questions is at least 0.10 above its recall on the 'who has an edge to X' questions."),
    ("claim", "dense: matrix answers the existence questions at least 0.05 worse than tuples."),
    ("claim", "dense: bitset answers the existence questions right at most 0.60 of the time (close to chance, which is 0.50)."),
    ("claim", "dense: bitset's mean F1 on the list questions is below the F1 of answering every list question with all the other services."),
    ("claim", "dense: bitset is at least 0.10 worse than tuples on the existence questions AND at least 0.10 worse on the F1 of the list questions."),
    ("claim", "dense: of the five formats adjacency has the highest mean F1 on the list questions."),
    ("claim", "sparse: tuples answers the existence questions at least as well as matrix."),
    ("claim", "sparse: adjacency's mean F1 on the list questions is at least tuples'."),
]


# ---------------------------------------------------------------- the graphs, the questions
def make_graph(rng: random.Random, edges: int, nodes: int = DENSE_NODES, grid: Tuple[int, int] = DENSE_GRID) -> Graph:
    """`nodes` services on random cells of the grid (values 1 to 9) and `edges` distinct directed edges (relation dep, weight 1 to 9) between distinct services."""
    cells = [(r, c) for r in range(grid[0]) for c in range(grid[1])]; assert nodes <= len(cells); rng.shuffle(cells); ns = sorted(cells[:nodes]); g = Graph()
    for n in ns: g.add_node(n, rng.randint(1, 9))
    pairs = [(u, v) for u in ns for v in ns if u != v]; assert edges <= len(pairs)
    for u, v in sorted(rng.sample(pairs, edges)): g.add_edge(u, v, "dep", rng.randint(1, 9))
    return g


def out_nb(g: Graph, n) -> List[tuple]: return sorted({k[1] for k in g.edges if k[0] == n})
def in_nb(g: Graph, n) -> List[tuple]: return sorted({k[0] for k in g.edges if k[1] == n})
def pair_set(g: Graph) -> set: return {(k[0], k[1]) for k in g.edges}
def density(g: Graph) -> float: n = len(g.nodes); return len(g.edges) / (n * (n - 1))


def make_questions(rng: random.Random, g: Graph) -> List[dict]:
    """The eight questions of one graph, in a fixed order: 2 'out', 2 'in', 4 'exist' (two yes, two no, in random order). Every list has at least one member and leaves at least one other service out."""
    ns = sorted(g.nodes); n = len(ns); qs = []
    for kind, how in (("out", 2), ("in", 2)):
        nb = out_nb if kind == "out" else in_nb; cand = [x for x in ns if 1 <= len(nb(g, x)) <= n - 2]
        for x in rng.sample(cand, how):
            text = (f"List every service that the service at ({x[0]},{x[1]}) has an edge to. Reply with the services written as (r,c), separated by spaces." if kind == "out"
                    else f"List every service that has an edge to the service at ({x[0]},{x[1]}). Reply with the services written as (r,c), separated by spaces.")
            qs.append(dict(kind=kind, node=list(x), text=text, truth=[list(v) for v in nb(g, x)]))
    pairs = pair_set(g); yes = rng.sample(sorted(pairs), 2); no = rng.sample([(u, v) for u in ns for v in ns if u != v and (u, v) not in pairs], 2)
    ex = [(p, True) for p in yes] + [(p, False) for p in no]; rng.shuffle(ex)
    for (u, v), t in ex: qs.append(dict(kind="exist", pair=[list(u), list(v)], text=f"Is there an edge from ({u[0]},{u[1]}) to ({v[0]},{v[1]})? Reply yes or no.", truth="yes" if t else "no"))
    for i, q in enumerate(qs): q["qid"] = f"q{i}"
    return qs


def make_problems(seed: int = DENSE_SEED, regimes=DENSE_REGIMES, nodes: int = DENSE_NODES, grid: Tuple[int, int] = DENSE_GRID) -> List[dict]:
    rng = random.Random(seed * 1000); out = []
    for name, count, edges in regimes:
        for k in range(count):
            g = make_graph(rng, edges, nodes, grid); out.append(dict(id=f"{name[0]}{k}", regime=name, graph=g, questions=make_questions(rng, g)))
    return out


def fingerprint(problems: Sequence[dict]) -> str:
    h = hashlib.sha256()
    for p in problems: h.update(repr((p["id"], p["regime"], sorted(p["graph"].nodes.items()), sorted(p["graph"].edges.items()), [(q["kind"], q["text"], q["truth"]) for q in p["questions"]])).encode())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------- the five formats and their decoders
def lab(n) -> str: return f"({n[0]},{n[1]})"


def render(g: Graph, fmt: str) -> str:
    assert fmt in DENSE_FORMATS, fmt
    ns = sorted(g.nodes)
    if fmt == "tuples":
        return CICOParser.serialize_state([(n[0], n[1], v) for n, v in sorted(g.nodes.items())], [(k[0][0], k[0][1], k[1][0], k[1][1], k[2], w) for k, w in sorted(g.edges.items())])
    cols = " ".join(lab(n) for n in ns)
    if fmt == "adjacency":
        head = "<GRAPH>\nOne line per service: the services it has an edge to (out), then the services that have an edge to it (in).\n"
        body = "\n".join(f"{lab(n)} out: {' '.join(lab(v) for v in out_nb(g, n))} in: {' '.join(lab(v) for v in in_nb(g, n))}" for n in ns)
        return head + body + "\n</GRAPH>"
    row = {n: [1 if (n, m) in pair_set(g) else 0 for m in ns] for n in ns}
    if fmt == "matrix":
        head = f"<GRAPH>\nAdjacency matrix. Columns, in order: {cols}\nEach row is a service; the number in column j is 1 if that service has an edge to column j's service, otherwise 0.\n"
        return head + "\n".join(f"{lab(n)}: {' '.join(str(b) for b in row[n])}" for n in ns) + "\n</GRAPH>"
    if fmt == "binary":
        head = f"<GRAPH>\nAdjacency matrix. Columns, in order: {cols}\nEach row is a service; character j of its string (counting from the left, starting at 0) is 1 if that service has an edge to column j's service, otherwise 0.\n"
        return head + "\n".join(f"{lab(n)}: {''.join(str(b) for b in row[n])}" for n in ns) + "\n</GRAPH>"
    head = (f"<GRAPH>\nAdjacency matrix, packed. Columns, in order (column 0 first): {cols}\nEach row is a service followed by one 64-bit integer in hexadecimal; bit j of the integer "
            "(counting from the right, starting at 0) is 1 if that service has an edge to column j's service, otherwise 0.\n")
    return head + "\n".join(f"{lab(n)}: 0x{sum(b << j for j, b in enumerate(row[n])):016x}" for n in ns) + "\n</GRAPH>"


_C = r"\((\d+),(\d+)\)"


def decode_state(text: str, fmt: str) -> set:
    """The set of edges (u, v) that a prompt's state holds, read back the way the format says it is written (not from the graph)."""
    if fmt == "tuples": return {((int(a), int(b)), (int(c), int(d))) for a, b, c, d in re.findall(_C + r"->" + _C + r":", text)}
    if fmt == "adjacency":
        out, inn = set(), set()
        for line in text.splitlines():
            m = re.match(_C + r" out:(.*) in:(.*)$", line)
            if not m: continue
            u = (int(m[1]), int(m[2]))
            out |= {(u, (int(a), int(b))) for a, b in re.findall(_C, m[3])}; inn |= {((int(a), int(b)), u) for a, b in re.findall(_C, m[4])}
        assert out == inn, "the out lists and the in lists disagree"; return out
    cols = [(int(a), int(b)) for a, b in re.findall(_C, re.search(r"Columns, in order[^:]*: (.*)", text)[1])]; edges = set()
    for line in text.splitlines():
        m = re.match(_C + r": (.+)$", line)
        if not m: continue
        u, rest = (int(m[1]), int(m[2])), m[3].strip()
        if fmt == "matrix": bits = [int(x) for x in rest.split()]
        elif fmt == "binary": bits = [int(x) for x in rest]
        else: w = int(rest, 16); bits = [(w >> j) & 1 for j in range(len(cols))]
        assert len(bits) == len(cols), (fmt, len(bits), len(cols)); edges |= {(u, cols[j]) for j, b in enumerate(bits) if b}
    return edges


def prompt_text(state: str, question: str) -> str:
    return f"{IM_START}system\n{DENSE_SYSTEM}{IM_END}\n{IM_START}user\n{state}\n\nQuestion: {question}{IM_END}\n{IM_START}assistant\n"


def qwen_like_tokens(text: str) -> int:
    """A LOWER BOUND of Qwen's token count: the number of pre-tokens of Qwen2's pre-tokenizer (a word with at most one non-letter before it, a single digit, a run of punctuation with a space before it, white
    space), which the byte-pair merges can only split further. Digits are single tokens and a space before a digit is a token of its own, which is what matters for the matrix formats. It is NOT the tokenizer;
    the run counts the real tokens."""
    return len(re.findall(r"[^\r\n\w]?[^\W\d_]+|\d| ?[^\s\w]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+|_", text))


def dense_dry_run(problems: Optional[List[dict]] = None, count: Callable[[str], int] = qwen_like_tokens) -> dict:
    """What the design alone fixes: the tokens of the state in each format (mean over the graphs of a regime) and the ratios to `tuples`."""
    problems = problems if problems is not None else make_problems(); out = {}
    for reg in sorted({p["regime"] for p in problems}):
        ps = [p for p in problems if p["regime"] == reg]; out[reg] = {f: _mean(count(render(p["graph"], f)) for p in ps) for f in DENSE_FORMATS}
        out[reg]["rho"] = _mean(density(p["graph"]) for p in ps)
        for f in DENSE_FORMATS: out[reg][f + "_over_tuples"] = out[reg][f] / out[reg]["tuples"]
    return out


def dense_preflight(tok, kit, problems: Optional[List[dict]] = None) -> List[Tuple[str, bool, str]]:
    """Checks on the real tokenizer, before any model time is spent (they need the tokenizer, not a GPU). Each returns (name, ok, detail)."""
    problems = problems if problems is not None else make_problems(); res = []; p = problems[0]; q = p["questions"][0]; bad = ""
    for fmt in DENSE_FORMATS:
        state = render(p["graph"], fmt); got = prompt_text(state, q["text"])
        want = tok.apply_chat_template([{"role": "system", "content": DENSE_SYSTEM}, {"role": "user", "content": f"{state}\n\nQuestion: {q['text']}"}], add_generation_prompt=True, tokenize=False)
        if want != got: bad = f"{fmt}: first difference at character " + str(next((i for i, (a, b) in enumerate(zip(want, got)) if a != b), min(len(want), len(got)))); break
    res.append(("the prompt of every format is the tokenizer's own chat template for the same two messages", not bad, bad or f"{len(DENSE_FORMATS)} formats compared"))
    under = [f"{p['id']}/{f}" for p in problems for f in DENSE_FORMATS if len(kit.enc(render(p['graph'], f))) < qwen_like_tokens(render(p['graph'], f))]
    res.append(("the tokenizer's own count of every state is at least the pre-token count that the page calls a lower bound", not under, f"{len(problems) * len(DENSE_FORMATS)} states" if not under else "below the bound: " + ", ".join(under[:3])))
    wrong = [f"{p['id']}/{f}" for p in problems for f in DENSE_FORMATS if decode_state(render(p['graph'], f), f) != pair_set(p['graph'])]
    res.append(("every state is read back by the decoder of its format to exactly the graph's edges", not wrong, f"{len(problems) * len(DENSE_FORMATS)} states" if not wrong else "wrong: " + ", ".join(wrong[:3])))
    return res


# ---------------------------------------------------------------- scoring
def parse_nodes(text: str) -> List[tuple]:
    return [(int(a), int(b)) for a, b in re.findall(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", text)]


def parse_yes_no(text: str) -> Optional[str]:
    m = re.search(r"\b(yes|no)\b", text, re.I); return m[1].lower() if m else None


def score_list(text: str, truth: Sequence, asked: Sequence) -> dict:
    """Precision, recall, F1 and exact match of the services a reply lists (the asked service itself, if the reply repeats it, is not counted)."""
    P = set(parse_nodes(text)) - {tuple(asked)}; T = {tuple(x) for x in truth}; tp = len(P & T)
    prec = tp / len(P) if P else 0.0; rec = tp / len(T); f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(precision=prec, recall=rec, f1=f1, exact=P == T, listed=len(P), wrong=len(P - T))


def everything_f1(truth_size: int, n_nodes: int) -> float:
    """The F1 of answering a list question with all the other services: precision k / (n - 1), recall 1."""
    p = truth_size / (n_nodes - 1); return 2 * p / (p + 1)


def score_question(q: dict, text: str, n_nodes: int) -> dict:
    if q["kind"] == "exist":
        a = parse_yes_no(text); return dict(answer=a, correct=a == q["truth"], parsed=a is not None)
    asked = q["node"]; s = score_list(text, q["truth"], asked); s["baseline_f1"] = everything_f1(len(q["truth"]), n_nodes); return s


# ---------------------------------------------------------------- running
def free_decode(cache, logits, V: int, eos: int, start_pos: int, max_new: int) -> Tuple[List[int], bool]:
    """Greedy decoding with no grammar (the answers here are lists and a word). Stops at the end token."""
    toks, lg, pos = [], logits, start_pos
    for _ in range(max_new):
        tok = int(np.argmax(np.asarray(lg[:V], np.float32))); toks.append(tok)
        if tok == eos: return toks, True
        if len(toks) >= max_new: break
        lg = cache.forward([tok], [pos]); pos += 1
    return toks, False


def run_unit(engine, kit, table, problem: dict, fmt: str, max_new: int = DENSE_MAX_NEW) -> dict:
    g = problem["graph"]; state = render(g, fmt); n = len(g.nodes); decodes = decode_state(state, fmt) == pair_set(g); qs = []; state_tokens = len(kit.enc(state))
    for q in problem["questions"]:
        toks = kit.enc(prompt_text(state, q["text"])); cache = Tracked(engine.new_cache()); engine.reset_peak(); engine.sync(); t0 = time.perf_counter()
        logits = cache.forward(toks, range(len(toks))); rep, ended = free_decode(cache, logits, table.V, kit.eos, len(toks), max_new); engine.sync(); t1 = time.perf_counter()
        text = kit.dec(rep); rec = dict(qid=q["qid"], kind=q["kind"], text=text, ended=ended, prompt_tokens=len(toks), seconds=t1 - t0, peak_mb=engine.peak_mb()); rec.update(score_question(q, text, n)); qs.append(rec); del cache
    return dict(id=f"{problem['id']}/{fmt}", pid=problem["id"], regime=problem["regime"], fmt=fmt, state_tokens=state_tokens, decodes=decodes, rho=density(g), qs=qs)


def run_dense(engine, kit, table, path: str, problems: Optional[List[dict]] = None, formats: Sequence[str] = DENSE_FORMATS, meta=None, progress=None, max_new: int = DENSE_MAX_NEW) -> dict:
    """One JSON line per finished (graph, format) pair; run again with the same arguments to carry on after a dropped session. A file made with other settings is refused."""
    problems = problems if problems is not None else make_problems(); units = [(p, f) for p in problems for f in formats]
    meta = dict(meta or {}, kind="kv_dense", units=len(units), formats=list(formats), seed=DENSE_SEED, nodes=DENSE_NODES, max_new=max_new, ids=[f"{p['id']}/{f}" for p, f in units], fp=fingerprint(problems))
    rows = load_rows(path, meta)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w") as f: f.write(json.dumps({"_meta": _roundtrip(meta)}) + "\n")
    with open(path, "a") as f:
        for i in range(len(rows), len(units)):
            p, fmt = units[i]; row = _roundtrip(run_unit(engine, kit, table, p, fmt, max_new)); assert row["id"] == f"{p['id']}/{fmt}"
            f.write(json.dumps(row) + "\n"); f.flush(); os.fsync(f.fileno()); rows.append(row)
            if progress: progress(i + 1, len(units))
    return {"rows": rows, "summary": summarize_dense(rows), "meta": meta}


# ---------------------------------------------------------------- summaries and the reading rules
def summarize_dense(rows) -> dict:
    out = {}
    for reg in sorted({r["regime"] for r in rows}):
        out[reg] = {}
        for fmt in [f for f in DENSE_FORMATS if any(r["fmt"] == f for r in rows)]:
            rs = [r for r in rows if r["regime"] == reg and r["fmt"] == fmt]; qs = [q for r in rs for q in r["qs"]]
            lst = [q for q in qs if q["kind"] != "exist"]; ex = [q for q in qs if q["kind"] == "exist"]; o = [q for q in lst if q["kind"] == "out"]; i = [q for q in lst if q["kind"] == "in"]
            out[reg][fmt] = dict(graphs=len(rs), n_list=len(lst), n_exist=len(ex), qids=sorted(f"{r['pid']}/{q['qid']}" for r in rs for q in r['qs']), decodes=all(r["decodes"] for r in rs), state_tokens=_mean(r["state_tokens"] for r in rs), prompt_tokens=_mean(q["prompt_tokens"] for q in qs),
                                 f1=_mean(q["f1"] for q in lst), f1_out=_mean(q["f1"] for q in o), f1_in=_mean(q["f1"] for q in i), recall=_mean(q["recall"] for q in lst), recall_out=_mean(q["recall"] for q in o), recall_in=_mean(q["recall"] for q in i),
                                 precision=_mean(q["precision"] for q in lst), exact=_mean(q["exact"] for q in lst), baseline_f1=_mean(q["baseline_f1"] for q in lst), wrong=_mean(q["wrong"] for q in lst),
                                 exist_acc=_mean(q["correct"] for q in ex), exist_unparsed=_mean(not q["parsed"] for q in ex), ran_on=_mean(not q["ended"] for q in qs), seconds=_mean(q["seconds"] for q in qs),
                                 peak_mb=max([q["peak_mb"] for q in qs] or [0]))
    return out


def _same_questions(s: dict) -> bool:
    """Both regimes were run in all five formats; every format of a regime answered the same questions (same graph and question ids), 40 on the dense graphs and 24 on the sparse ones."""
    want = {"dense": 40, "sparse": 24}
    if set(s) != set(want): return False
    for reg, fm in s.items():
        if set(fm) != set(DENSE_FORMATS) or len({tuple(v["qids"]) for v in fm.values()}) != 1: return False
        if any(v["n_list"] + v["n_exist"] != want[reg] or len(v["qids"]) != want[reg] for v in fm.values()): return False
    return True


def verdict_dense(s: dict) -> List[Tuple[str, Optional[bool]]]:
    """The fourteen reading rules, in the order of DENSE_LINES."""
    d, sp = s.get("dense", {}), s.get("sparse", {}); tu, ad, ma, bi, bs = (d.get(f, {}) for f in ("tuples", "adjacency", "matrix", "binary", "bitset")); stu, sad, sma = (sp.get(f, {}) for f in ("tuples", "adjacency", "matrix"))
    ge = lambda x, y, k: None if x is None or y is None else x - y >= k - 1e-12
    all_f = [v for r in s.values() for v in r.values()]
    ok = [None if not all_f else _same_questions(s),
          None if not all_f else all(v["decodes"] for v in all_f),
          _ge(tu.get("exist_acc"), N_EXIST_MIN),
          None if tu.get("state_tokens") is None or ma.get("state_tokens") is None else ma["state_tokens"] <= N_TOK_MATRIX * tu["state_tokens"] + 1e-9,
          None if tu.get("state_tokens") is None or bs.get("state_tokens") is None else bs["state_tokens"] <= N_TOK_BITSET * tu["state_tokens"] + 1e-9,
          ge(ad.get("recall_in"), tu.get("recall_in"), N_IN_GAIN), ge(tu.get("recall_out"), tu.get("recall_in"), N_OUT_OVER_IN),
          ge(tu.get("exist_acc"), ma.get("exist_acc"), N_MATRIX_BELOW),
          None if bs.get("exist_acc") is None else bs["exist_acc"] <= N_BITSET_CHANCE + 1e-12,
          None if bs.get("f1") is None else bs["f1"] < bs["baseline_f1"],
          None if bs.get("exist_acc") is None or tu.get("exist_acc") is None else bool(ge(tu["exist_acc"], bs["exist_acc"], N_BITSET_BELOW) and ge(tu["f1"], bs["f1"], N_BITSET_BELOW)),
          None if not d or any(d.get(f, {}).get("f1") is None for f in DENSE_FORMATS) else all(ad["f1"] >= d[f]["f1"] - 1e-12 for f in DENSE_FORMATS),
          None if stu.get("exist_acc") is None or sma.get("exist_acc") is None else stu["exist_acc"] >= sma["exist_acc"] - 1e-12,
          None if sad.get("f1") is None or stu.get("f1") is None else sad["f1"] >= stu["f1"] - 1e-12]
    return [(text, v) for (_, text), v in zip(DENSE_LINES, ok)]


def print_dense(summary: dict, title: str = ""):
    print(f"\n== dense-graph test {title} ==")
    for reg, fm in summary.items():
        print(f"\n{reg} graphs:")
        print(f"{'format':10s} {'state tok':>9s} {'F1':>6s} {'out rec':>8s} {'in rec':>7s} {'precis.':>8s} {'exact':>6s} {'all-F1':>7s} {'exist ok':>9s} {'no answer':>9s} {'cut off':>8s} {'sec':>6s}")
        for f, v in fm.items():
            print(f"{f:10s} {_fx(v['state_tokens'], 0):>9s} {_fx(v['f1']):>6s} {_fx(v['recall_out']):>8s} {_fx(v['recall_in']):>7s} {_fx(v['precision']):>8s} {_pc(v['exact'], 0):>6s} {_fx(v['baseline_f1']):>7s} {_pc(v['exist_acc'], 0):>9s} {_pc(v['exist_unparsed'], 0):>9s} {_pc(v['ran_on'], 0):>8s} {_fx(v['seconds'], 1):>6s}")


def print_verdict_dense(checks, title: str):
    print(f"\n-- {title} (reading rules fixed before the run) --")
    for i, (text, ok) in enumerate(checks, 1): print(("  yes      " if ok else "  not run  " if ok is None else "  NO       ") + f"{i:2d}. {text}")
