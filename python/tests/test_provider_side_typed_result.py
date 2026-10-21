"""Implements concept-map item 111 (Result): the fifth provider-side serving run, seven exact-state cells with the reply grammar chosen by the type of the request (and the target line of item 110 in
two prompts), made on Colab (vLLM 0.31.0, Tesla T4, Qwen2.5-7B-Instruct-AWQ, a cache of 80,000 tokens). Page: docs/PROVIDER_SIDE_TYPED_TEST.md (the question, the design, the 17 reading lines and the
predictions were written before the run; the Result is appended below them).
The file: docs/results/axi_provider_typed.json (the merged file the notebook's last cell wrote and the owner sent: `F0`, `FK`, `FKA`, `FKT`, `FN` on the fresh jobs 128 to 191 and `SK`, `SN` on jobs 0 to 63),
read against docs/results/axi_provider_bounded.json (the third run: its `S` cell at 64 jobs is the base of `SK` and `SN`).
This file recomputes from the saved files, with its own arithmetic and not with the code that printed the figures:
  1. the files: their sha256, the settings, the seeds, the rounds adding up to the cells, and every one of the 13,440 steps played again from the text of its reply through the framework's own session, gate,
     repair and diagnosis (parsed, admitted, corrupted, exact, distance and inside-the-language of every step);
  2. the 17 reading lines (15 yes; lines 6 and 15 NO; none not run) by the code and by hand from the raw numbers, the headline, and what the page's rule lets the Result claim;
  3. the jobs that missed (what the request said, what the oracle writes, what the model wrote, which language the reply is in), the replay rule on the four pairs, and what the model wrote in the deletes;
  4. the parity check on jobs it was not built on, the seconds and the tokens of every cell, the Q = O / M block of this run;
  5. the whole Result is `render(result)` and the page must equal the locked head followed by it; the figures quoted in it are recounted;
  6. that the Result claims what the page's rule allows and nothing more (no multiple of capacity, no GPUs saved, no dollars, no 'proven', no '100 percent' for a grammar, nothing about requests in free words).
  7. (added on 20 Oct 2026, evening) the probe printout, docs/results/axi_provider_typed_probe.txt (Cell 6, a plain text file with its own sha256): parsed, checked step by step against the saved runs and the module
     (the request, the oracle's reply, the earlier reply, its language, the parity turn, the language of every reply), and the page's side predictions (b) to (f) read from it.
What it does NOT show: anything about a real server's mixed traffic, requests in free words, another model, GPU, engine or cache size, dollars, energy, or what the probe means for the loop (eight steps, one reply under each variant)."""
import hashlib, json, math, os, re, sys

sys.path.insert(0, os.path.dirname(__file__))
import axi.experiments.provider_side as P
import axi.experiments.provider_side_bounded as B
import axi.experiments.cascade as K
import axi.experiments.provider_side_typed as T
import axi.experiments.q_measure as Q
from axi.engine.gate import check
from axi.experiments.agent_loop import same_graph, strict_parse

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..", "..")
PAGE = os.path.join(ROOT, "docs", "PROVIDER_SIDE_TYPED_TEST.md")
RESD = os.path.join(ROOT, "docs", "results")
RES = os.path.join(RESD, "axi_provider_typed.json")
PROBE = os.path.join(RESD, "axi_provider_typed_probe.txt")                                                 # Cell 6, sent by the owner after the Result was first published
PROBE_SHA256 = "b7ff0352161c4f086dd5effb4e6eb4b38581ae7242dcb6ae41bcbc26a1e77fc7"
PROBE_VARIANTS = ("old", "kind", "named", "kind+top", "kind+tail", "parity")
PROBE_SHAPE = {"old": "capped", "parity": "capped", "kind": "kind", "kind+top": "kind", "kind+tail": "kind", "named": "named"}
THIRD = os.path.join(RESD, "axi_provider_bounded.json")
RETRY = os.path.join(RESD, "axi_provider_retry.json")
MODULE = os.path.join(HERE, "..", "axi", "experiments", "provider_side_typed.py")
RESULT_MARK = "\n\n# Result"
PAGE_SHA256 = "d4c963599e8dfe67f80f928535244ae078df5b91a8313f13b0bbc473c036d19f"                    # the locked head
FILE_SHA256 = "6bb4ca481441182b2d055248dd75339992e9f8377b9f4b759dc1281f5b17866f"
MODULE_SHA256 = "c5f05d5840374fb77a1c67c6fd33f166b26342e93733bd9a31aa474eacbc1420"
THIRD_SHA256 = "1f12a846b1325f9fed4e5df2cc755b3a7fd996266cb590311230f7cf418f2733"
RETRY_SHA256 = "390f7464a54352764e21203f9546bd17fc4714b09cbb0aec9390963e5d94b116"
PINS = dict(provider_side="bfb4734e171d9244cd47d10402be46528a557095a8275d02b38481a0e2b0b1d3", constrained="03452148e28c9f31372f12cc11ef1808ac67308adff972fa19d4aeeef7c96f7d",
            bounded="7427a686d151c4294809aebaa9664d0b5fa420854817eb79e910372c3f57b6bc", gate="b6a50408788791f76bc4d4129dad84debb6e09c9c5e804673cdf04abec224ce7",
            diagnostics="255f3c11db989b645077d10706a187e2c913576538a826e6dbdb8892b014b4f2", agent_loop="d658afaa29e3bce7e1fb7f43d88281fc40ee68ca59864421a65284e9973516b6",
            cascade="b0533316bc094e80a1d68eddb7e137c7c176af9a342341288360c18d61f74fe7", long_horizon="5718435de2178de110ddbb59dd3bf32e1bec9f1fd8625a27119d789399fa20ac")
REGEX_SHA256 = "6f4e5a993f26ccd471bff13422fa622b7820858a91e05d43c2bccfbe6b864245"
GRAMMAR_SHA = {"F0": REGEX_SHA256, "FK": "69152ac964df234b99a442f0115927462040714b0ae3e1bdabba55a906f766b2", "FKA": "69152ac964df234b99a442f0115927462040714b0ae3e1bdabba55a906f766b2",
               "FKT": "69152ac964df234b99a442f0115927462040714b0ae3e1bdabba55a906f766b2", "FN": "c1732477bafdb53fd115fb67028ade7c9c09fc44219a521612be9c63f3eeab4e",
               "SK": "69152ac964df234b99a442f0115927462040714b0ae3e1bdabba55a906f766b2", "SN": "12cfe62cbbb9b8a6722fb9e4fc6288ba78cf56b16e7ad89bc8a09dfbbf1f2e46"}
SEED, STEPS = 20261017, 30
ORDER = ["F0", "FK", "FKA", "FKT", "FN", "SK", "SN"]
FRESH = ["F0", "FK", "FKA", "FKT", "FN"]
JOBS = {lab: (T.CELLS[lab][1], T.CELLS[lab][0]) for lab in ORDER}                               # (first job, number of jobs)
PROBS = [98, 90, 80, 97, 93, 38, 85, 60, 45, 70, 50, 50, 80, 70, 15, 85, 62]
THRESH = dict(l6=0.01, l8=0.99, l9=0.995, l10=0.995, l13=1.05, l14=1.15, l15=1.30, l16=1.20)               # the numbers of the page's reading rules, in the order the page gives them
SIDE = ("(a) In `SK`, jobs 4 and 20 miss again at steps 18 and 30 with the earlier text (follows from line 3): 88 %.", "(b) In the probe the capped expression with the old prompt writes the earlier slip again at 7 or more of the 8 steps: 88 %.",
        "(c) In the probe the `kind` variant writes the oracle's reply at 4 or more of the 5 steps whose earlier reply is outside the kind language (jobs 3, 17, 19, 62, 101): 65 %.", "(d) In the probe the `named` variant writes the oracle's reply at 6 or more of the 8 steps: 70 %; at all 8: 35 %.",
        "(e) In the probe the anchor at the top writes the oracle's reply at least once for the three number slips (jobs 4, 20, 122): 40 %.", "(f) In the probe the parity turn gets the oracle's reply for job 4: 40 %; for at least one of jobs 4, 20 and 122: 55 %.",
        "(g) `F0` is between 96.5 and 99.0 percent exact: 75 %.", "(h) The computed prompt tokens per step of `FKA` are at least 50 percent above those of `FK`: 80 %; those of `FKT` between 5 and 20 percent above: 80 %.", "(i) The seven cells take 20 to 40 minutes in all, loading excluded: 70 %.")
RULE_LABELS = ("Lines 2, 3, 4 yes, lines 5 and 6 yes, line 7 yes:", "Line 5 yes, line 6 NO:", "Line 8 yes:", "Lines 9 and 10 yes:", "Line 10 NO:", "Lines 11 or 12 NO:", "Line 15 NO, line 16 yes:", "Line 13 or 14 NO:", "Line 17 NO with line 6 yes:",
               "Line 3 NO with line 2 yes:", "Line 4 NO:", "Line 2 NO:", "Line 1 NO, or the safety cell says STOP:")
READ = [True, True, True, True, True, False, True, True, True, True, True, True, True, True, False, True, True]
_cache = {}


def run():
    if "run" not in _cache: _cache["run"] = json.load(open(RES))
    return _cache["run"]


def _load(key, path):
    if key not in _cache: _cache[key] = json.load(open(path))
    return _cache[key]


def third(): return _load("third", THIRD)
def retry(): return _load("retry", RETRY)
def _text(): return open(PAGE).read()


def _head():
    t = _text(); assert t.count(RESULT_MARK) <= 1
    return t[:t.index(RESULT_MARK)] if RESULT_MARK in t else t


def _result(): t = _text(); return t[t.index(RESULT_MARK):]
def _h(text): return hashlib.sha256((text.rstrip("\n") + "\n").encode()).hexdigest()
def _sha(path): return hashlib.sha256(open(path, "rb").read()).hexdigest()
def _norm(t): return re.sub(r"\s+", " ", t)


# ---------------------------------------------------------------- the figures, from the step and round records
def pc(x, k=1): return f"{x * 100:.{k}f} %"
def num(x): return f"{x:,}"
def sg(x): return f"{x:+.1f} %".replace("+-", "-")
def cells(res): return {c["label"]: c for c in res["cells"]}
def logs(c): return [x for s in c["sessions"] for x in s["log"]]
def secs(c): return sum(r["seconds"] for r in c["rounds"])
def job_of(s): return s["seed"] - SEED * 1000
def script(job): return T.script_of(SEED * 1000 + job)
def third_cell(k=64): return next(c for c in third()["cells"] if c["arm"] == "S" and c["K"] == k)
def share(c): L = logs(c); return sum(1 for x in L if x["exact"]) / len(L)
def ex_by_job(c): return [sum(1 for x in s["log"] if x["exact"]) for s in c["sessions"]]
def jobs_hit(c): return sorted(job_of(s) for s in c["sessions"] if any(not x["exact"] for x in s["log"]))


def st(c):
    L = logs(c); n = len(L); wall = secs(c); ex = sum(1 for x in L if x["exact"]); P_ = sum(x["n_prompt"] for x in L); C_ = sum(x["n_cached"] for x in L); W_ = sum(x["n_new"] for x in L); m_tok = P_ - C_ + W_
    return dict(n=n, wall=wall, sps=wall / n, ex=ex, share=ex / n, eph=3600 * ex / wall, jobs=len(c["sessions"]), whole=sum(1 for s in c["sessions"] if all(x["exact"] for x in s["log"])), lost=n - ex,
                prompt=P_, cached=C_, new=W_, comp_step=(P_ - C_) / n, cached_share=C_ / P_, m_tok=m_tok, q1000=1000 * ex / m_tok, unparsed=sum(1 for x in L if not x["parsed"]), full96=sum(1 for x in L if x["n_new"] >= 96),
                longest=max(x["n_new"] for x in L), corrupted=sum(1 for x in L if x["corrupted"]), out_of_language=sum(1 for x in L if not x["in_language"]), regexes=c["distinct_regexes"],
                new_grammars=sum(r.get("new_grammars", 0) for r in c["rounds"]), new_rounds=sum(1 for r in c["rounds"] if r.get("new_grammars")), rounds=len(c["rounds"]), first_round=c["rounds"][0]["seconds"])


def misses(c):
    """the first step not exact of every job of the cell that has one"""
    out = []
    for s in c["sessions"]:
        x = next((y for y in s["log"] if not y["exact"]), None)
        if x is None: continue
        job = job_of(s); sc = script(job)[x["t"] - 1]
        out.append(dict(job=job, step=x["t"], kind=sc["kind"], request=sc["instruction"], oracle=sc["ideal"], reply=x["text"], admitted=x["admitted"], lost=sum(1 for y in s["log"] if not y["exact"]),
                        kind_ok=T.in_language("kind", sc["instruction"], x["text"]), named_ok=T.in_language("named", sc["instruction"], x["text"]), parity=T.parity_ok(sc["instruction"], x["text"])))
    return sorted(out, key=lambda m: m["job"])


def items_of(text): return re.findall(r"(?:ADD|DEL)\[[^\]]*\]", text)


def slip_class(m):
    its = items_of(m["reply"])
    if m["kind"].startswith("delete"):
        node = T.parse_request(m["request"])["node"]
        if its and its[0] == f"DEL[{node[0]},{node[1]}]": return "right node, wrong rest"
        if its and re.fullmatch(r"DEL\[\d+,\d+\]", its[0]): return "wrong node"
        return "edge without the node"
    if len(its) > 1 and its[0].startswith("DEL") and " ".join(its[1:]) == m["oracle"]: return "extra delete before the add"
    if re.fullmatch(r"ADD\[\((\d+),(\d+)\)->\(\1,\2\):[A-Za-z0-9_]{1,16}#-?\d+\]", m["reply"]): return "node written as an edge"
    return "wrong number"


def forced(new, base, shape):
    """job by job: (steps compared, [(job, step, base reply, new reply, oracle's reply)]) -- the steps up to and including the first step whose base reply is outside the language"""
    n = 0; out = []
    for a, b in zip(new["sessions"], base["sessions"]):
        sc = T.script_of(a["seed"]); k = None
        for i, (x, y) in enumerate(zip(a["log"], b["log"])):
            if not T.in_language(shape, sc[i]["instruction"], y["text"]): k = i; break
        n += (k + 1) if k is not None else len(b["log"])
        if k is not None: out.append((job_of(a), k + 1, b["log"][k]["text"], a["log"][k]["text"], sc[k]["ideal"]))
    return n, out


PAIRS = [("FK", "F0", "kind"), ("FN", "F0", "named"), ("SK", None, "kind"), ("SN", None, "named")]


def base_of(res, name): return third_cell(64) if name is None else cells(res)[name]


def same_text_until_forced(new, base, shape):
    for a, b in zip(new["sessions"], base["sessions"]):
        sc = T.script_of(a["seed"])
        for i, (x, y) in enumerate(zip(a["log"], b["log"])):
            if T.in_language(shape, sc[i]["instruction"], y["text"]):
                if x["text"] != y["text"]: return False
            else: break
        else:
            if len(a["log"]) != len(b["log"]): return False
    return True


def parity_facts(c):
    """steps whose reply does not say what the request says (or an add reply with more than one item): flagged steps, how many are exact, how many are the first miss of their job, the first misses not flagged"""
    flagged = []; first_missed = 0; unflagged_first = 0
    for s in c["sessions"]:
        sc = script(job_of(s)); fm = next((x["t"] for x in s["log"] if not x["exact"]), None)
        for x, q in zip(s["log"], sc):
            if not T.parity_ok(q["instruction"], x["text"]): flagged.append((job_of(s), x["t"], x["exact"], fm))
        if fm is not None:
            first_missed += 1
            if T.parity_ok(sc[fm - 1]["instruction"], s["log"][fm - 1]["text"]): unflagged_first += 1
    return dict(flagged=len(flagged), flagged_exact=sum(1 for f in flagged if f[2]), at_first_miss=sum(1 for f in flagged if f[1] == f[3]), first_misses=first_missed, first_misses_unflagged=unflagged_first)


def delete_facts(c):
    """the replies to a delete of a service with connections (`delete_dep`) and to a delete of a leaf: how many are the node alone, how many equal the oracle's text, how many carry an edge item"""
    d = dict(dep=0, dep_node_only=0, dep_node_only_exact=0, dep_edge_item=0, leaf=0, leaf_equal=0, add=0, add_equal=0)
    for s in c["sessions"]:
        for x, q in zip(s["log"], script(job_of(s))):
            its = items_of(x["text"])
            if x["kind"] == "delete_dep":
                d["dep"] += 1; only = len(its) == 1 and re.fullmatch(r"DEL\[\d+,\d+\]", its[0]) is not None
                d["dep_node_only"] += only; d["dep_node_only_exact"] += only and x["exact"]; d["dep_edge_item"] += any("->" in i for i in its)
            elif x["kind"] == "delete_leaf": d["leaf"] += 1; d["leaf_equal"] += x["text"] == q["ideal"]
            else: d["add"] += 1; d["add_equal"] += x["text"] == q["ideal"]
    return d


def brier(probs, read): return sum((p / 100 - (1 if r else 0)) ** 2 for p, r in zip(probs, read)) / len(probs)


def fisher_one_sided(k, n=64):
    """the chance that a group of n jobs holds none of k missing jobs when k of 2n are missing at random (hypergeometric), one side"""
    return math.comb(n, 0) * math.comb(n, k) / math.comb(2 * n, k)


# ---------------------------------------------------------------- replay: every job played again from the texts of its replies
def replay_job(c, s):
    """the job played again through the typed session (which is the pinned session with a prompt and an expression added): the step record of every reply"""
    sess = T.TypedSession(s["seed"], STEPS, c["prompt"], c["shape"]); out = []
    for x in s["log"]:
        sess.request(); sess.accept(P.Reply(x["text"], 0, 0, 0)); y = sess.log[-1]
        out.append((y["parsed"], y["admitted"], y["corrupted"], y["exact"], y["distance"], y["in_language"]))
    return out


# ---------------------------------------------------------------- the lines, by hand from the raw numbers
def lines_by_hand(res):
    c = cells(res); S = {k: st(v) for k, v in c.items()}; allx = [x for lab in ORDER for x in logs(c[lab])]
    l1 = all(x["n_cached"] is not None for x in allx)
    l2 = sum(1 for x in allx if not x["parsed"]) <= 0.005 * len(allx) and all(x["n_new"] < 96 for x in allx) and all(x["in_language"] for x in allx)
    l3 = all(same_text_until_forced(c[n], base_of(res, b), sh) for n, b, sh in PAIRS)
    l4 = all(c[lab]["grammar_sha256"] == T.grammar_sha256(c[lab]["shape"], c[lab]["first_episode"], c[lab]["K"], c[lab]["steps"], c[lab]["seed"]) for lab in ORDER) \
        and all(re.fullmatch(T.regex_for(c[lab]["shape"], q["instruction"]), q["ideal"]) for lab in ORDER for s in c[lab]["sessions"] for q in script(job_of(s)))
    l5 = S["FK"]["share"] >= S["F0"]["share"]
    l6 = S["FK"]["share"] >= S["F0"]["share"] + THRESH["l6"]
    l7 = all(a >= b for a, b in zip(ex_by_job(c["FK"]), ex_by_job(c["F0"])))
    l8 = S["SK"]["share"] >= THRESH["l8"]
    l9 = S["SN"]["share"] >= THRESH["l9"]
    l10 = S["FN"]["share"] >= THRESH["l10"]
    l11 = S["FKA"]["share"] >= S["FK"]["share"]
    l12 = S["FKT"]["share"] >= S["FK"]["share"]
    l13 = S["FK"]["sps"] <= THRESH["l13"] * S["F0"]["sps"]
    l14 = S["FN"]["sps"] <= THRESH["l14"] * S["F0"]["sps"]
    l15 = S["FKA"]["sps"] <= THRESH["l15"] * S["FK"]["sps"]
    l16 = S["FKT"]["sps"] <= THRESH["l16"] * S["FK"]["sps"]
    l17 = S["FK"]["eph"] > S["F0"]["eph"]
    return [l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15, l16, l17]


def better_same_worse(c, a, b):
    x, y = ex_by_job(c[a]), ex_by_job(c[b]); return sum(1 for p, q in zip(x, y) if p > q), sum(1 for p, q in zip(x, y) if p == q), sum(1 for p, q in zip(x, y) if p < q)


def jl(jobs): return ", ".join(str(j) for j in jobs)


# ---------------------------------------------------------------- the lines' "what was measured" text
def measured(res, i):
    c = cells(res); S = {k: st(v) for k, v in c.items()}; allx = [x for lab in ORDER for x in logs(c[lab])]; n = len(allx)
    fz = {name: forced(c[name], base_of(res, b), sh) for name, b, sh in PAIRS}
    if i == 1: return f"the engine reported the cached tokens of every request: {num(n)} requests in the 7 cells"
    if i == 2:
        return (f"replies accepted by the strict parser: {num(n - sum(1 for x in allx if not x['parsed']))} of {num(n)}, {pc((n - sum(1 for x in allx if not x['parsed'])) / n)} (limit 99.5 %, met); replies that used all 96 new tokens: none "
                f"(the longest has {max(x['n_new'] for x in allx)} tokens); replies inside the language of their own request: {num(sum(1 for x in allx if x['in_language']))} of {num(n)}")
    if i == 3:
        return ("`FK` and `FN` against `F0`, `SK` and `SN` against the third run's `S` cell at 64 jobs, job by job, up to each job's first step whose base reply is outside the new language: all the same text, and the reply at that step inside it ("
                + ", ".join(f"{num(fz[name][0])} steps for `{name}`" for name, _, _ in PAIRS) + "; the jobs forced: " + "; ".join(f"`{name}` {jl([j for j, *_ in fz[name][1]])}" for name, _, _ in PAIRS) + ")")
    if i == 4:
        return (f"each cell's recorded set of expressions has the sha256 recomputed from its jobs (capped `{REGEX_SHA256[:8]}…`, kind `{GRAMMAR_SHA['FK'][:8]}…`, named `{GRAMMAR_SHA['SN'][:8]}…` on jobs 0 to 63 and `{GRAMMAR_SHA['FN'][:8]}…` on jobs 128 to 191), "
                f"and the oracle's own reply is inside the expression of all {num(n)} requests")
    if i == 5: return f"`FK` {num(S['FK']['ex'])} of {num(S['FK']['n'])} steps exact ({pc(S['FK']['share'])}), `F0` {num(S['F0']['ex'])} ({pc(S['F0']['share'])}): not below"
    if i == 6: return f"`FK` against `F0`: {(S['FK']['share'] - S['F0']['share']) * 100:+.1f} points (limit +1.0, not met)"
    if i == 7:
        b, s_, w = better_same_worse(c, "FK", "F0"); return f"the 64 jobs: {s_} hold the right state after as many steps with `FK` as with `F0`, {b} after more (job {jl([j for j in range(128, 192) if ex_by_job(c['FK'])[j - 128] > ex_by_job(c['F0'])[j - 128]])}), {w} after fewer"
    if i == 8: return f"`SK` {num(S['SK']['ex'])} of {num(S['SK']['n'])} exact ({pc(S['SK']['share'])}) (limit 99 %, met; the most it could be was 99.3 %; the third run: 97.2 %)"
    if i == 9: return f"`SN` {num(S['SN']['ex'])} of {num(S['SN']['n'])} exact ({pc(S['SN']['share'])}) (limit 99.5 %, met; the ceiling)"
    if i == 10: return f"`FN` {num(S['FN']['ex'])} of {num(S['FN']['n'])} exact ({pc(S['FN']['share'])}) (limit 99.5 %, met)"
    if i == 11:
        b, s_, w = better_same_worse(c, "FKA", "FK"); return f"`FKA` {num(S['FKA']['ex'])} exact ({pc(S['FKA']['share'])}), `FK` {num(S['FK']['ex'])} ({pc(S['FK']['share'])}): not below ({b} jobs better, {w} worse)"
    if i == 12:
        b, s_, w = better_same_worse(c, "FKT", "FK"); return f"`FKT` {num(S['FKT']['ex'])} exact ({pc(S['FKT']['share'])}), `FK` {num(S['FK']['ex'])} ({pc(S['FK']['share'])}): not below ({b} jobs better, {w} worse)"
    if i == 13: return f"`FK` {S['FK']['sps']:.4f} seconds per step, `F0` {S['F0']['sps']:.4f}: {S['FK']['sps'] / S['F0']['sps']:.3f} times (limit 1.05, met)"
    if i == 14: return f"`FN` {S['FN']['sps']:.4f}, `F0` {S['F0']['sps']:.4f}: {S['FN']['sps'] / S['F0']['sps']:.3f} times (limit 1.15, met)"
    if i == 15: return f"`FKA` {S['FKA']['sps']:.4f}, `FK` {S['FK']['sps']:.4f}: {S['FKA']['sps'] / S['FK']['sps']:.3f} times (limit 1.30, not met)"
    if i == 16: return f"`FKT` {S['FKT']['sps']:.4f}, `FK` {S['FK']['sps']:.4f}: {S['FKT']['sps'] / S['FK']['sps']:.3f} times (limit 1.20, met)"
    if i == 17: return f"`FK` {num(round(S['FK']['eph']))} exact steps per GPU hour, `F0` {num(round(S['F0']['eph']))}: {sg((S['FK']['eph'] / S['F0']['eph'] - 1) * 100)} (limit: higher)"


def table_lines(res): return [f"| {i} | {PROBS[i - 1]} % | {measured(res, i)} | {'yes' if READ[i - 1] else 'NO'} |" for i in range(1, 18)]


def table_cells(res):
    c = cells(res); S = {k: st(v) for k, v in c.items()}
    rows = ["| cell | jobs | prompt | grammar | steps | seconds | s / step | exact steps | exact | jobs exact to the end | exact steps per hour | expressions |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for lab in ORDER:
        f, k = JOBS[lab]; s = S[lab]
        rows.append(f"| `{lab}` | {f} to {f + k - 1} | {c[lab]['prompt']} | {c[lab]['shape']} | {num(s['n'])} | {s['wall']:.1f} | {s['sps']:.4f} | {num(s['ex'])} | {pc(s['share'])} | {s['whole']} of {s['jobs']} | {num(round(s['eph']))} | {s['regexes']} |")
    return rows


def table_cost(res):
    c = cells(res); S = {k: st(v) for k, v in c.items()}
    rows = ["| cell | s / step | against | prompt tokens per step | cached | computed per step | written per step | tokens computed + written | exact steps per 1,000 tokens |", "|---|---|---|---|---|---|---|---|---|"]
    base = {"FK": "F0", "FKA": "FK", "FKT": "FK", "FN": "F0"}
    for lab in ORDER:
        s = S[lab]; rel = f"`{base[lab]}`: {s['sps'] / S[base[lab]]['sps']:.3f} times" if lab in base else "-"
        rows.append(f"| `{lab}` | {s['sps']:.4f} | {rel} | {s['prompt'] / s['n']:.1f} | {pc(s['cached_share'])} | {s['comp_step']:.1f} | {s['new'] / s['n']:.2f} | {num(s['m_tok'])} | {s['q1000']:.1f} |")
    return rows


def table_jobs(res):
    """every job that missed in one of the five fresh cells: what the request said, what the oracle writes, and what each cell did"""
    c = cells(res); jobs = sorted({m["job"] for lab in FRESH for m in misses(c[lab])})
    rows = ["| job | step | the request | the oracle writes | `F0` | `FK` | `FKA` | `FKT` | `FN` |", "|---|---|---|---|---|---|---|---|---|"]
    for j in jobs:
        ms = {lab: next((m for m in misses(c[lab]) if m["job"] == j), None) for lab in FRESH}; first = next(m for lab in FRESH for m in [ms[lab]] if m)
        cell = lambda m: "exact to the end" if m is None else f"step {m['step']}: `{m['reply']}` ({m['lost']} lost)"
        rows.append(f"| {j} | {first['step']} | {first['request']} | `{first['oracle']}` | " + " | ".join(cell(ms[lab]) for lab in FRESH) + " |")
    return rows


def table_forced(res):
    c = cells(res); rows = ["| pair | steps compared | jobs forced (job, step) | what the third run or `F0` wrote there | what the new cell wrote | the oracle's reply | exact to the end |", "|---|---|---|---|---|---|---|"]
    for name, b, sh in PAIRS:
        n, fl = forced(c[name], base_of(res, b), sh)
        for j, step, was, now, orc in fl:
            end = next(s for s in c[name]["sessions"] if job_of(s) == j); ok = "yes" if all(x["exact"] for x in end["log"]) else "no"
            rows.append(f"| `{name}` | {num(n)} | {j}, {step} | `{was}` | `{now}` | " + ("the same" if now == orc else f"`{orc}`") + f" | {ok} |")
    return rows


def table_parity(res):
    c = cells(res); rows = ["| cell | steps | steps flagged | of them exact | first misses (jobs) | first misses flagged |", "|---|---|---|---|---|---|"]
    for lab in ORDER:
        f = parity_facts(c[lab]); n = len(logs(c[lab]))
        rows.append(f"| `{lab}` | {num(n)} | {f['flagged']} | {f['flagged_exact']} | {f['first_misses']} | {f['first_misses'] - f['first_misses_unflagged']} of {f['first_misses']} |")
    tot = [parity_facts(c[lab]) for lab in ORDER]
    rows.append(f"| all | {num(sum(len(logs(c[lab])) for lab in ORDER))} | {sum(f['flagged'] for f in tot)} | {sum(f['flagged_exact'] for f in tot)} | {sum(f['first_misses'] for f in tot)} | {sum(f['first_misses'] - f['first_misses_unflagged'] for f in tot)} of {sum(f['first_misses'] for f in tot)} |")
    return rows


def table_deletes(res):
    c = cells(res); rows = ["| cell | add replies | equal to the oracle's | deletes of a leaf | equal | deletes of a service with connections | the node alone | of them exact | with an edge item |", "|---|---|---|---|---|---|---|---|---|"]
    for lab in ORDER:
        d = delete_facts(c[lab]); rows.append(f"| `{lab}` | {num(d['add'])} | {num(d['add_equal'])} | {d['leaf']} | {d['leaf_equal']} | {d['dep']} | {d['dep_node_only']} | {d['dep_node_only_exact']} | {d['dep_edge_item']} |")
    return rows


def q_block(res):
    """the fifth run's block of the Q table, as q_measure renders it from the saved file"""
    return Q.render_table({"fifth run, typed requests": Q.q_rows(RES)}).rstrip("\n")


def probe():
    """the Cell 6 printout parsed: (the three smoke steps, the eight records, the last lines); every line must have the shape the printing code gives it"""
    raw = open(PROBE, "rb").read(); assert raw.count(b"\r\n") == raw.count(b"\n") == 78 and raw.endswith(b"\r\n")
    lines = raw.decode().replace("\r\n", "\n").split("\n"); smoke = []
    for l in lines[:3]:
        m = re.fullmatch(r"step (\d+): prompt +(\d+)  cached (\d+)  new +(\d+)  '(.*)'", l); assert m, l; smoke.append((int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), m.group(5)))
    assert lines[3] == ""; recs = []; i = 4
    while i < len(lines) and lines[i].startswith("job "):
        m = re.fullmatch(r"job (\d+), step (\d+) \((\w+)\): (.*)", lines[i]); assert m, lines[i]
        m2 = re.fullmatch(r"   the earlier reply was '(.*)' \(inside the kind language: (True|False), inside the named language: (True|False)\); the oracle wrote '(.*)'", lines[i + 1]); assert m2, lines[i + 1]
        var = {}
        for name, l in zip(PROBE_VARIANTS[:5], lines[i + 2: i + 7]):
            m3 = re.fullmatch(r"   (\S+) +'(.*?)'( +  = the oracle)?", l); assert m3 and m3.group(1) == name, l; var[name] = (m3.group(2), bool(m3.group(3)))
        m4 = re.fullmatch(r"   parity +'(.*?)'( +  = the oracle)?", lines[i + 7]); assert m4, lines[i + 7]; var["parity"] = (m4.group(1), bool(m4.group(2)))
        m5 = re.fullmatch(r"   the parity turn was: '(.*)'", lines[i + 8]); assert m5, lines[i + 8]
        recs.append(dict(job=int(m.group(1)), step=int(m.group(2)), kind=m.group(3), instr=m.group(4), earlier=m2.group(1), in_kind=m2.group(2) == "True", in_named=m2.group(3) == "True", oracle=m2.group(4), var=var, pturn=m5.group(1))); i += 9
    return smoke, recs, lines[i:]


def probe_facts(recs):
    """counts by variant and the readings of the page's side predictions (b) to (f), from the records"""
    ora = {v: sum(1 for r in recs if r["var"][v][1]) for v in PROBE_VARIANTS}; ear = {v: sum(1 for r in recs if r["var"][v][0] == r["earlier"]) for v in PROBE_VARIANTS}
    shape = [r for r in recs if not r["in_kind"]]; number = [r for r in recs if r["in_kind"]]
    both = [r["job"] for r in recs if r["var"]["parity"][0] == r["oracle"] + " " + r["earlier"]]
    return dict(oracle=ora, earlier=ear, shape_jobs=[r["job"] for r in shape], number_jobs=[r["job"] for r in number], kind_on_shape=sum(1 for r in shape if r["var"]["kind"][1]), kind_on_number=sum(1 for r in number if r["var"]["kind"][1]),
                top_on_number=sum(1 for r in number if r["var"]["kind+top"][1]), parity_both=both, parity_job4=next(r for r in recs if r["job"] == 4)["var"]["parity"][1],
                parity_any=[r["job"] for r in recs if r["job"] in (4, 20, 122) and r["var"]["parity"][1]],
                side=dict(b=ear["old"] >= 7, c=sum(1 for r in shape if r["var"]["kind"][1]) >= 4, d6=ora["named"] >= 6, d8=ora["named"] == 8, e=sum(1 for r in number if r["var"]["kind+top"][1]) >= 1,
                          f4=next(r for r in recs if r["job"] == 4)["var"]["parity"][1], fany=any(r["var"]["parity"][1] for r in recs if r["job"] in (4, 20, 122))))


def tf(b): return "true" if b else "**false**"


def probe_cell(r, v):
    text, eq = r["var"][v]
    if eq: return "the oracle's reply"
    if text == r["earlier"]: return "the earlier reply again"
    if text == r["oracle"] + " " + r["earlier"]: return "the oracle's reply, then the earlier reply"
    return f"`{text}`"


def table_probe(recs):
    out = ["| job | step | request | the earlier reply | `old` | `kind` | `named` | `kind+top` | `kind+tail` | parity turn |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in recs: out.append(f"| {r['job']} | {r['step']} | {r['kind']} | `{r['earlier']}` | " + " | ".join(probe_cell(r, v) for v in PROBE_VARIANTS) + " |")
    return out


def render(res):
    c = cells(res); S = {k: st(v) for k, v in c.items()}; e = res["meta"]["engine"]; wall = sum(S[k]["wall"] for k in S); M = {lab: misses(c[lab]) for lab in ORDER}
    same_cost = [S[k]["wall"] for k in ("F0", "FK", "FN", "SK", "SN")]; fz = {name: forced(c[name], base_of(res, b), sh) for name, b, sh in PAIRS}
    dF0, dFN, dFKA = delete_facts(c["F0"]), delete_facts(c["FN"]), delete_facts(c["FKA"]); pf = {lab: parity_facts(c[lab]) for lab in ORDER}
    f0 = M["F0"]; f0_del = [m for m in f0 if m["kind"].startswith("delete")]; wrong_nodes = sorted({(m["job"], m["reply"], T.parse_request(m["request"])["node"]) for lab in FRESH for m in M[lab] if slip_class(m) == "wrong node"})
    bsw = {k: better_same_worse(c, *k) for k in (("FK", "F0"), ("FKA", "FK"), ("FKT", "FK"), ("FN", "FK"), ("FN", "F0"), ("FKA", "F0"))}
    lines = []; A = lines.append
    A(""); A(""); A("# Result"); A("")
    A(f"*Written after the run, below the locked text; nothing above this line has been changed (its hash is checked by `tests/test_provider_side_typed.py` and again by `tests/test_provider_side_typed_result.py`). The run was made by the owner on Colab with `python/colab/axi_provider_side_typed.ipynb`; the file reached me on 20 Oct 2026 and holds no date of its own. It records: vLLM {e['version']}, GPU {e['gpu']}, Python {e['python']}, `{e['model']}`, the cache fixed at {num(e['capacity_tokens'])} tokens with the engine's block override (capacity source \"{e['capacity_source']}\"), 30-step jobs, seed {SEED}, one regular expression per request (the base expression of the third run is sha256 `{REGEX_SHA256[:16]}…`), the policy `repair_m6`, and no retry in any cell. **All seven planned cells ran** (`F0`, `FK`, `FKA`, `FKT`, `FN`, `SK`, `SN`); each holds its 30 rounds in order. The file is `docs/results/axi_provider_typed.json` (sha256 `{FILE_SHA256}`): the merged file that the notebook's last cell wrote with `provider_side_typed.merge_files`. It records the sha256 of that module (`{MODULE_SHA256[:16]}…`, the module in the repository), those of the eight files the test imports (all as pinned on the page), and the third run's file it was compared with (`axi_provider_bounded.json`, `{THIRD_SHA256[:16]}…`). The seconds around the engine's calls of the seven cells add up to {wall:,.1f}, which is {wall / 60:.1f} minutes. **What the safety cell and the probe printed is not in the saved file, and no table or figure below rests on it; the printout of the probe reached me later and is read in its own section, \"The probe (Cell 6), read afterwards\".** Every figure below is computed from the saved file by `tests/test_provider_side_typed_result.py`.*")
    A("")
    A(f"**What the page expected, and what came out.** The page put lines 2, 3 and 4 all yes at about 70 %, and they read yes, as do lines 1 and 5: the run is the run the page describes. **{sum(READ)} of the 17 lines read yes; lines 6 and 15 read NO, and no line is \"not run\".** The pattern the page called most likely (lines 1 to 5, 7, 13 and 14 yes, line 15 NO, the rest open: about 25 %) is what came out; the headline is not what it called most likely: lines 6 and 17 read NO and yes, where \"both yes\" stood at about 35 %. On the 64 jobs that no earlier run had seen (jobs 128 to 191) the grammar by kind is {pc(S['FK']['share'])} exact against {pc(S['F0']['share'])} for the capped one ({num(S['FK']['ex'])} and {num(S['F0']['ex'])} of {num(S['F0']['n'])} steps), {(S['FK']['share'] - S['F0']['share']) * 100:+.1f} points where line 6 asked for +1.0, and its exact steps per GPU hour are {num(round(S['FK']['eph']))} against {num(round(S['F0']['eph']))} ({sg((S['FK']['eph'] / S['F0']['eph'] - 1) * 100)}), which is the size of the spread between the five cells that did the same amount of work ({min(same_cost):.1f} to {max(same_cost):.1f} seconds, {(max(same_cost) / min(same_cost) - 1) * 100:.1f} %). **Line 17 reads yes by the page's rule and this Result does not read it as a gain.** Two other cells reached all {num(S['FN']['n'])} steps exact on those jobs: the grammar that names the coordinates (`FN`, {S['FN']['sps']:.3f} seconds per step, the same as `F0`) and the target line at the top and at the end of the message (`FKA`, {sg((S['FKA']['sps'] / S['FK']['sps'] - 1) * 100)} seconds per step). The page's reading of \"line 5 yes, line 6 NO\" was \"too few slips of the removable kind among the new jobs to see one point, or the forced replies were wrong\"; it is the first: the kind grammar forced one reply in `FK` (job {fz['FK'][1][0][0]}, step {fz['FK'][1][0][1]}) and that reply is the oracle's own.")
    A("")
    A("## The 17 lines")
    A("")
    A("| # | my probability | what was measured | read |")
    A("|---|---|---|---|")
    lines.extend(table_lines(res))
    A("")
    A(f"**{sum(READ)} of 17 lines read yes, lines 6 and 15 read NO, and no line is \"not run\".** The two NO stood at {PROBS[5]} % (line 6) and {PROBS[14]} % (line 15) on the page, so both were the side I expected. The probabilities of lines 9, 11 and 12 ({PROBS[8]}, {PROBS[10]} and {PROBS[11]} %) were near a coin and they read yes. Scored against what happened, the 17 probabilities have a Brier score of {brier(PROBS, READ):.3f} (always writing 50 % scores 0.250; lower is better); that is a description of 17 lines written by one person and not a measure of anything.")
    A("")
    A("## What the page's rule lets this Result claim")
    A("The page's rule, applied line by line to what was read:")
    A("* **Lines 2, 3 and 4 read yes**, so the figures of the cells are claimed as read.")
    A(f"* **Line 5 reads yes and line 6 reads NO.** The page allows the claim that a grammar raises the exactness only from the fresh jobs (lines 5 to 7 and 10 to 12). The kind grammar did not raise it there ({pc(S['FK']['share'])} against {pc(S['F0']['share'])}; one job better, none worse). **This Result does not claim that the grammar by kind raises the exactness.**")
    A(f"* **Line 10 reads yes.** `FN` is exact at all {num(S['FN']['n'])} steps of the fresh jobs. The page's condition for saying so is in the same sentence: the host wrote the coordinates of the request into the reply, and what the model wrote is the number (below, \"What the model still writes\").")
    A(f"* **Lines 11 and 12 read yes.** `FKA` is exact at all {num(S['FKA']['n'])} steps, and no job is worse than with `FK` ({bsw[('FKA', 'FK')][0]} jobs better); `FKT` is {S['FKT']['ex'] - S['FK']['ex']:+d} steps from `FK` ({bsw[('FKT', 'FK')][0]} jobs better and {bsw[('FKT', 'FK')][2]} worse), which is not a claim. The anchor at the top and the end costs {sg((S['FKA']['sps'] / S['FK']['sps'] - 1) * 100)} seconds per step (line 15 NO).")
    A("* **Lines 8 and 9 are about the jobs the slips were read on** (jobs 0 to 63) and say what the replay rule said they would: the grammar can only change a reply that is outside it (below, \"The seen jobs\").")
    A("* **Lines 13 to 17 are from cells of one session.** The seconds of the cells of this run are compared with each other and never with the third run's.")
    A("* **Never**, as the page says: a multiple of capacity, a count of GPUs saved, dollars or a percentage of cost, \"100 percent\" for a grammar, \"proven\", a statement about any provider, a statement about requests in free words, or about a model that was not run.")
    A("")
    A(f"**The sentence the page allows, with this run's numbers.** On jobs 128 to 191, which no earlier run had seen, with the requests read by the host from their sentence, on one Tesla T4 (Colab), with vLLM {e['version']}, `{e['model']}`, a cache of {num(e['capacity_tokens'])} tokens and 30-step jobs: the reply restricted to the kind of the request left the exactness at {pc(S['FK']['share'])} where the capped expression gave {pc(S['F0']['share'])} ({num(S['FK']['ex'])} and {num(S['F0']['ex'])} of {num(S['F0']['n'])} steps; no gain that one run of 64 jobs can show), at {S['FK']['sps']:.3f} and {S['F0']['sps']:.3f} seconds per step, {num(round(S['FK']['eph']))} and {num(round(S['F0']['eph']))} exact steps per GPU hour and {S['FK']['q1000']:.1f} and {S['F0']['q1000']:.1f} per 1,000 tokens. The grammar that also names the coordinates of the request had all {num(S['FN']['n'])} steps exact at {S['FN']['sps']:.3f} seconds per step, {num(round(S['FN']['eph']))} exact steps per GPU hour and {S['FN']['q1000']:.1f} per 1,000 tokens, **and in the same sentence: in that grammar the host writes the coordinates of the request into the reply, the model writes the value of an added service or the weight of an added edge, and in every one of the {dFN['dep'] + dFN['leaf']} delete replies it wrote the node alone, the edges being completed by the framework's repair.** The target line at the top and the end of the message with the kind grammar had all {num(S['FKA']['n'])} steps exact at {S['FKA']['sps']:.3f} seconds per step ({num(round(S['FKA']['eph']))} exact steps per GPU hour, {S['FKA']['q1000']:.1f} per 1,000 tokens). One run, 64 jobs.")
    A("")
    A("## The jobs that missed")
    A(f"On the fresh jobs {len(jobs_hit(c['F0']))} jobs of 64 missed somewhere in `F0` (jobs {jl(jobs_hit(c['F0']))}), {len(jobs_hit(c['FK']))} in `FK`, {len(jobs_hit(c['FKT']))} in `FKT` and none in `FKA` and `FN`. The first miss of each, with what each cell did at the job (\"lost\" is the steps of the job after which the held state was not the oracle's; the exact-state arm keeps the state it holds, so a step that was not done stays not done):")
    A("")
    lines.extend(table_jobs(res))
    A("")
    cl = [slip_class(m) for m in f0]; kind_ok = [m for m in f0 if m["kind_ok"]]
    A(f"**{len(f0_del)} of the {len(f0)} first misses of `F0` are replies to a Delete request.** The eight first misses in the 128 jobs that the earlier runs read (the page's table) were all replies to an Add request. In `F0`: job 136 and job 175 wrote `DEL[0,1]` for the service (1,0) and for (1,1) (the wrong service), jobs 172 and 178 wrote an edge of the service and not the service (the node is missing, and in job 172 the edge is also written in the other direction), and job 150 wrote an extra delete in front of the right add, the kind of slip of jobs 17 and 62 on the page. **Four of the five replies are inside the kind language** (a delete may name any service, and the grammar by kind does not look at which), so the grammar by kind could change only job 150, and it did: the forced reply is the oracle's own and the job is exact to the end (2 steps). The wrong services in the four different wrong-node replies of the five cells ({', '.join(f'job {j} `{r}` for ({n[0]},{n[1]})' for j, r, n in wrong_nodes)}) all have a 0 where the request has a 1 as the first coordinate; four replies are too few to say why. The gate refused the replies of jobs 136, 172 and 175 and admitted those of jobs 150 (a legal extra delete) and 178 (an edge deleted and the service left); no cell here asks again. The grammar that names the service (`FN`) removes all five, as it must, since the service is written by the host; the target line (`FKA`) removed the four that the kind grammar leaves, and nothing was written by the host there.")
    A("")
    A(f"`FKT`, the target line at the end only, is the same prompt as `FK` plus {S['FKT']['comp_step'] - S['FK']['comp_step']:.1f} computed tokens a step. It put right jobs 172 and 178, left 136 and 175 and made 136 worse ({M['FKT'][0]['lost']} lost against {M['FK'][0]['lost']}) and added a new miss, job 154. A prompt is a different draw of the model's replies, and with so few misses this is what such a draw looks like; it is not a claim about the tail line.")
    A("")
    A(f"**On `FKA` against `FK`: four jobs changed from missing to exact and none changed the other way** (jobs 136, 172, 175, 178). Under chance alone (each of the jobs that missed in either cell equally likely to be in either) four to none has a one-sided chance of about {fisher_one_sided(4) * 100:.0f} in 100 (the hypergeometric count for 0 of 64 against 4 of 64). That is too few jobs to call it settled and not nothing, and it came from one run.")
    A("")
    A("## What the model still writes")
    A(f"The page said that in the grammar that names the coordinates \"the model's share is the number of a node, the weight of an edge, and the list of edges of a delete\". The saved replies say more. In `FN`, **all {num(dFN['add'])} add replies and all {dFN['leaf']} replies to a delete of a service without connections are, letter for letter, what the oracle writes ({num(dFN['add_equal'] + dFN['leaf_equal'])} of {num(dFN['add'] + dFN['leaf'])}), and all {dFN['dep']} replies to a delete of a service with connections are `DEL[x,y]` alone** ({dFN['dep_node_only']} node-only, {dFN['dep_node_only_exact']} of them exact): the gate does not admit a delete that leaves edges without their service, and the framework's repair (`repair_m6`, which completes the dangling edges; unchanged) puts them in, so the held state is the oracle's. In no cell did the model write the edge list of a delete: the only delete replies with an edge item in the seven cells are the two edge-only replies of `F0` and of `FK` that are the first misses of jobs 172 and 178 (the service itself is missing from them), and the same node-alone reply is what the other cells wrote, right or wrong, at every other delete of a service with connections. So what the model wrote that mattered in `FN` is the value of an added service ({dFN['add'] - sum(1 for x in logs(c['FN']) if x['kind'] == 'add_edge')} replies) and the weight of an added edge ({sum(1 for x in logs(c['FN']) if x['kind'] == 'add_edge')} replies), and all of them were right. **The all-exact result of `FN` is therefore the host writing the coordinates and the model writing {num(dFN['add'])} numbers right; it is not the model reading the request.** Per cell:")
    A("")
    lines.extend(table_deletes(res))
    A("")
    A("## The seen jobs")
    A(f"With greedy decoding a grammar cannot change a reply that is already inside it, so the page could say before the run what `SK` and `SN` must write. It came out as said: in `SK` the replies are the third run's up to each job's first reply outside the kind language ({num(fz['SK'][0])} steps compared; forced at jobs {jl([j for j, *_ in fz['SK'][1]])}), and in `SN` up to the first reply outside the named language ({num(fz['SN'][0])} steps; forced at jobs {jl([j for j, *_ in fz['SN'][1]])}). Every forced step gives a job that is exact to the end of its 30 steps (the table), so `SK` has {num(S['SK']['ex'])} exact steps, which is the ceiling the page computed (jobs 4 and 20 lose 14), and jobs 4 and 20 slip again at steps 18 and 30 with the text of the third run (`ADD[1,0:6]`, `ADD[(1,5)->(2,1):dep#3]`), as the replay rule says they must: their replies are inside the kind language. `SN` is exact at all {num(S['SN']['n'])} steps. The same rule holds on the fresh jobs: `FK` and `FN` write `F0`'s replies until `F0`'s first reply outside their language ({num(fz['FK'][0])} and {num(fz['FN'][0])} steps compared).")
    A("")
    lines.extend(table_forced(res))
    A("")
    A("## The parity check on jobs it was built to fit and on jobs it was not")
    A(f"The page's check (does the first item of the reply say what the request says, and does an add reply have exactly one item) was built on the eight first misses of the jobs 0 to 127. It is used nowhere in this run to decide a step. On the {num(sum(len(logs(c[lab])) for lab in ORDER))} steps of the seven cells it flags {sum(f['flagged'] for f in pf.values())} steps; **every one of them is the first miss of its job, none is a step after which the job held the right state, and every first miss of every cell is flagged.** That includes the five fresh `F0` misses, which it had not seen when it was written. It does not mean that the check would be right in another setting: it reads the reply against the sentence the host read, a reply that is wrong in a way the sentence does not name (a value, a weight) would not be flagged, and none occurred here.")
    A("")
    lines.extend(table_parity(res))
    A("")
    A("## Seconds and tokens")
    A(f"The five cells that did the same work took {min(same_cost):.1f} to {max(same_cost):.1f} seconds (`F0`, `FK`, `FN`, `SK`, `SN`): the grammar by kind and the grammar that names the coordinates cost nothing that shows in the seconds, and the {S['FN']['new_grammars']} expressions that `FN` carries on the fresh jobs (new ones in {S['FN']['new_rounds']} of its {S['FN']['rounds']} rounds) are inside that spread; its first round took {S['FN']['first_round']:.2f} seconds against {S['F0']['first_round']:.2f} for `F0`. The target line is what costs: at the top and the end it moves every prompt after the first step away from the cache beyond the system prompt ({pc(S['FKA']['cached_share'])} of the prompt tokens served from the cache against {pc(S['FK']['cached_share'])}), the engine computed {S['FKA']['comp_step']:.1f} prompt tokens a step against {S['FK']['comp_step']:.1f} ({S['FKA']['comp_step'] / S['FK']['comp_step']:.2f} times; the page estimated about 181 against 109), and the seconds per step rose by {(S['FKA']['sps'] / S['FK']['sps'] - 1) * 100:.0f} %, inside the 30 to 60 % the page expected. At the end only, it is {S['FKT']['comp_step']:.1f} tokens a step ({S['FKT']['comp_step'] / S['FK']['comp_step']:.2f} times) and {(S['FKT']['sps'] / S['FK']['sps'] - 1) * 100:.1f} % more seconds per step. For the steps it bought, `FKA` has {S['FKA']['ex'] - S['FK']['ex']} more exact steps than `FK` ({sg((S['FKA']['ex'] / S['FK']['ex'] - 1) * 100)}) for {S['FKA']['wall'] - S['FK']['wall']:.1f} more seconds, so its exact steps per GPU hour are {sg((S['FKA']['eph'] / S['FK']['eph'] - 1) * 100)} against `FK` and its exact steps per 1,000 tokens {S['FKA']['q1000']:.1f} against {S['FK']['q1000']:.1f}: more exact steps, fewer per hour, on this measure.")
    A("")
    lines.extend(table_cost(res))
    A("")
    A("## Every cell")
    A("")
    lines.extend(table_cells(res))
    A("")
    A("Q = O / M of this run, as `q_measure.py` computes it from the saved file (the same block is on `Q_FROM_SAVED_FILES.md`; the two numbers are never mixed into one):")
    A("")
    A(q_block(res))
    A("")
    A("The third run's cells were in another Colab session and the page says a cell of this run is not compared with them for time; the seconds above are for reading each cell and the pairs of this run, which ran in one session.")
    A("")
    smoke, precs, ptail = probe(); PF = probe_facts(precs); npr = len(precs); fkt_kinds = [m["kind"] for m in M["FKT"]]
    A("## The probe (Cell 6), read afterwards")
    A("")
    A(f"*Added on 20 Oct 2026, in the evening, after the Result above was published: the owner sent the printout of Cell 6. It is saved as `docs/results/axi_provider_typed_probe.txt` (sha256 `{PROBE_SHA256}`; a plain text file, not part of the merged result file) and this test reads it line by line against the saved runs. The page called the probe a description and never a step of the loop: for each of the {npr} steps where an earlier run had a first miss, the prompt of that step is rebuilt with the oracle's own replies before it and asked once under each variant, and once more under the capped expression with the earlier wrong reply as the assistant's turn and a `PARITY` user turn that says what the request names and what the reply named.*")
    A("")
    lines.extend(table_probe(precs))
    A("")
    A(f"**It agrees with the saved runs.** On all {npr} steps the request, the oracle's reply, the earlier reply, whether that reply is inside the `kind` and the `named` language and the parity turn are what the saved files and the module give, and all {len(PROBE_VARIANTS) * npr} replies are inside the expression of their own variant. The three smoke steps show prompts of {smoke[0][1]}, {smoke[1][1]} and {smoke[2][1]} tokens of which {smoke[0][2]}, {smoke[1][2]} and {smoke[2][2]} came from the cache: the first {smoke[1][2]} tokens are the system prompt, as the page expected.")
    A("")
    A(f"* **`old`, the capped expression with the old prompt:** the earlier reply again at {PF['earlier']['old']} of the {npr} steps. At job 101 it wrote the oracle's reply this time, so that step says nothing about the other variants; the printout does not say why and I did not test it.")
    A(f"* **`kind`:** the oracle's reply at {PF['kind_on_shape']} of the {len(PF['shape_jobs'])} steps whose earlier reply was outside the kind language (jobs {jl(PF['shape_jobs'])}) and at {PF['kind_on_number']} of the {len(PF['number_jobs'])} number slips (jobs {jl(PF['number_jobs'])}), where it wrote the earlier reply again. That is what the replay rule and the page said this grammar can do and cannot do.")
    A(f"* **`named`, `kind+top` and `kind+tail`:** the oracle's reply at all {npr} steps. In `kind+top` and `kind+tail` the host writes nothing of the reply (the grammar only fixes its shape); the target line is in the prompt and the model wrote the right number at the {len(PF['number_jobs'])} number slips. All {npr} steps are Adds. In the run, the line at the end only (`FKT`) gave {pc(S['FKT']['share'])} against {pc(S['FK']['share'])} for `FK` on the fresh jobs: its {len(fkt_kinds)} first misses were all Deletes ({', '.join(fkt_kinds)}), which this probe does not ask.")
    other = [r for r in precs if not r["var"]["parity"][1] and r["var"]["parity"][0] not in (r["earlier"], r["oracle"] + " " + r["earlier"])]
    A(f"* **The parity turn:** the oracle's reply alone at {PF['oracle']['parity']} of the {npr} steps (job {jl(PF['parity_any'])}); the oracle's reply followed by the earlier reply at {len(PF['parity_both'])} (jobs {jl(PF['parity_both'])}); the earlier reply again at {PF['earlier']['parity']}; at job {jl([r['job'] for r in other])} the right coordinates with a `DEL` in front of them. Told what it had written, the model rarely replaced the reply. This is {npr} steps, and no step of the loop uses it.")
    A("")
    A("## Where my expectation was off, and where it was right")
    A(f"* **Lines 1 to 4 (98, 90, 80 and 97 %): the run is the run the page describes.** {num(sum(S[k]['n'] for k in S))} requests, every one with its cached count; every reply accepted by the strict parser, none at 96 new tokens (the longest has {max(S[k]['longest'] for k in S)}) and every reply inside the language of its own request, so the engine did apply a different expression to each request of one call; the four pairs replay as the rule says; the expressions are the ones the page defines.")
    A(f"* **Line 6 (38 %) reads NO,** as the larger share of my probability said. I expected about 1.7 slips of the removable kind in 64 jobs; there was one (job 150). The surprise is not the line but the kind of slip: I read the eight first misses of the earlier jobs, all Adds, and designed the grammar for them, and four of the five misses of the new jobs are Deletes that name the wrong service or leave the service out.")
    A(f"* **Lines 9 and 10 (45 and 70 %) read yes and say little.** `SN` at its ceiling is the host writing the coordinates in the six jobs where the third run's reply was outside the language, and `FN` is the same on new jobs; what they add is that the model wrote {num(dFN['add'])} numbers right in `FN` and {num(delete_facts(c['SN'])['add'])} in `SN`.")
    A(f"* **Lines 11 and 12 (50 % each) read yes.** The anchor is the one cell on the fresh jobs where nothing was written by the host and all {num(S['FKA']['n'])} steps are exact. I do not call it settled (one run, 64 jobs, a different prompt is a different draw, and `FKT` shows that a draw can move in both directions).")
    A(f"* **Line 15 (15 %) reads NO,** as expected: {S['FKA']['sps'] / S['FK']['sps']:.3f} times, limit 1.30. **Line 16 (85 %) reads yes:** {S['FKT']['sps'] / S['FK']['sps']:.3f} times.")
    A(f"* **Line 17 (62 %) reads yes by {abs((S['FK']['eph'] / S['F0']['eph'] - 1) * 100):.1f} %,** which is {S['FK']['ex'] - S['F0']['ex']} more exact steps and {S['F0']['wall'] - S['FK']['wall']:.1f} fewer seconds, the size of the spread between the cells that did the same work.")
    sk_ = M["SK"]; side12 = [[(m["job"], m["step"]) for m in sk_] == [(4, 18), (20, 30)], PF["side"]["b"], PF["side"]["c"], PF["side"]["d6"], PF["side"]["d8"], PF["side"]["e"], PF["side"]["f4"], PF["side"]["fany"], 0.965 <= S["F0"]["share"] <= 0.99,
              S["FKA"]["comp_step"] / S["FK"]["comp_step"] >= 1.5, 1.05 <= S["FKT"]["comp_step"] / S["FK"]["comp_step"] <= 1.20, 20 <= wall / 60 <= 40]
    A(f"* **The side predictions, read afterwards as descriptions.** (a) in `SK`, jobs 4 and 20 miss again at steps 18 and 30 with the earlier text (88 %): true. (g) `F0` is between 96.5 and 99.0 percent exact (75 %): **false,** {pc(S['F0']['share'])}; the capped expression gave {pc(share(third_cell(64)))} on jobs 0 to 63 in the third run and {pc(share(cells(retry())['H0']))} on jobs 64 to 127 in the fourth, so three sets of 64 jobs give three different figures. (h) the computed prompt tokens per step of `FKA` are at least 50 percent above those of `FK` (80 %): true, {S['FKA']['comp_step'] / S['FK']['comp_step']:.2f} times; those of `FKT` between 5 and 20 percent above (80 %): true, {S['FKT']['comp_step'] / S['FK']['comp_step']:.2f} times. (i) the seven cells take 20 to 40 minutes in all, loading excluded (70 %): true by the saved seconds ({wall / 60:.1f} minutes around the engine's calls); the time with the prompts built and the files written is not in the file. (b) with the capped expression and the old prompt the probe writes the earlier slip again at 7 or more of the 8 steps (88 %): {tf(PF['side']['b'])}, {PF['earlier']['old']} of 8. (c) the `kind` variant writes the oracle's reply at 4 or more of the 5 steps whose earlier reply is outside the kind language (65 %): {tf(PF['side']['c'])}, {PF['kind_on_shape']} of 5. (d) the `named` variant writes the oracle's reply at 6 or more of the 8 steps (70 %): {tf(PF['side']['d6'])}, and at all 8 (35 %): {tf(PF['side']['d8'])}. (e) the anchor at the top writes the oracle's reply at least once for the three number slips (40 %): {tf(PF['side']['e'])}, for {PF['top_on_number']} of the 3. (f) the parity turn gets the oracle's reply for job 4 (40 %): {tf(PF['side']['f4'])} (it wrote the oracle's reply and then the earlier one); for at least one of jobs 4, 20 and 122 (55 %): {tf(PF['side']['fany'])}, job {jl(PF['parity_any'])}. **Of the twelve statements the page made as side predictions, {sum(side12)} read true and {12 - sum(side12)} false.**")
    A(f"* **A correction to the page.** It said that the edge list of a delete is still the model's in `FN`. In the replies the model never wrote one (above); the repair did.")
    A("")
    A("## What this finds, in plain words")
    A(f"No step of the {num(sum(S[k]['n'] for k in S))} left the held graph malformed ({sum(S[k]['corrupted'] for k in S)} corrupted), and every reply was inside the language of its own request. On jobs nobody had read, the capped expression gave {pc(S['F0']['share'])} exact and the grammar by kind {pc(S['FK']['share'])}: **it did not raise the exactness, because four of the five jobs that slipped slipped on a Delete (two named the wrong service, two wrote an edge and left the service out), which the kind grammar still lets the model write.** Two cells reached all {num(S['FN']['n'])} steps exact on those jobs, and they are not the same kind of thing. **The grammar that names the coordinates did it at no cost in seconds ({S['FN']['sps']:.3f} per step, {num(round(S['FN']['eph']))} exact steps per GPU hour against {num(round(S['F0']['eph']))}), but by the host writing the coordinates it had read from the request sentence; the model wrote the value or the weight and, in the deletes, the node alone.** That says what a host that reads the request can hand to the engine; it does not say that the model reads requests. **The target line at the top and the end did it with the grammar by kind only and the model choosing the node, at {sg((S['FKA']['sps'] / S['FK']['sps'] - 1) * 100)} seconds per step and {num(round(S['FKA']['eph']))} exact steps per GPU hour,** four jobs changed from missing to exact and none the other way; the same line at the end only did not (`FKT`, {pc(S['FKT']['share'])}). The probe printed afterwards points the same way on {npr} steps that earlier runs missed: the grammar by kind wrote the oracle's reply at {PF['kind_on_shape']} of the {len(PF['shape_jobs'])} shape slips and at {PF['kind_on_number']} of the {len(PF['number_jobs'])} number slips, and naming the coordinates, or the target line at the top or at the end, wrote it at all {npr}. The parity check named every first miss of the run and no other step. What is left to find out is whether a short anchor can be made cheap (it costs the cache), whether it holds on more jobs than 64, and what happens with requests in free words, where the host cannot read the coordinates.")
    A("")
    A("## What this does not show")
    A("That the model reads requests (in the grammar that names the coordinates the host writes them); that the target line raises the exactness in general (one run, 64 new jobs, one model); anything about a placement of the line other than the two run; what the probe means for the loop (eight steps, one reply under each variant, descriptions the page said would never decide a step); a retry on top of a grammar (no cell has one); requests in free words; the kept chat with a grammar (not run); another model, GPU, engine or cache size; dollars; a capacity in jobs per GPU; a count of GPUs; energy (never measured); a lower bound of the cost per exact step; jobs longer than 30 steps; a real server's mixed traffic. The third run's cells were not re-measured here. It says nothing about the tensor-level roadmap.")
    return "\n".join(lines)


# ================================================================ the tests
# ---------------------------------------------------------------- 1 the files
def test_the_files_are_the_ones_the_page_is_about_and_every_step_replays_from_its_reply_text_through_the_frameworks_own_rules():
    res = run(); assert _sha(RES) == FILE_SHA256, "the saved result file must not change"; assert _sha(THIRD) == THIRD_SHA256 == res["third_run_sha256"] and res["third_run_file"] == "axi_provider_bounded.json" and _sha(RETRY) == RETRY_SHA256
    assert _sha(MODULE) == MODULE_SHA256 == res["module_sha256"] == T.fingerprint(), "the module that merged the cells must be the one in the repository"
    assert set(res) == {"meta", "cells", "third_run_file", "third_run_sha256", "module_sha256", "pinned_sha256"} and res["pinned_sha256"] == PINS == T.pinned_hashes()
    for name, path in (("provider_side", ("experiments", "provider_side.py")), ("constrained", ("experiments", "provider_side_constrained.py")), ("bounded", ("experiments", "provider_side_bounded.py")), ("gate", ("engine", "gate.py")),
                       ("diagnostics", ("engine", "diagnostics.py")), ("agent_loop", ("experiments", "agent_loop.py")), ("cascade", ("experiments", "cascade.py")), ("long_horizon", ("experiments", "long_horizon.py"))):
        assert _sha(os.path.join(HERE, "..", "axi", *path)) == PINS[name], name
    m = res["meta"]; e = m["engine"]; con = e["constraint"]
    assert {k: v for k, v in e.items() if k != "constraint"} == dict(kind="vllm", name="Qwen/Qwen2.5-7B-Instruct-AWQ", version="0.31.0", gpu="Tesla T4", capacity_tokens=80000, capacity_source="override", block_size=16, kv_bytes_per_token=57344, model="Qwen/Qwen2.5-7B-Instruct-AWQ", python="3.13.15")
    assert con["kind"] == "regex_per_request" and con["regex"] == B.REGEX and con["regex_sha256"] == REGEX_SHA256 == hashlib.sha256(B.REGEX.encode()).hexdigest() and con["name_cap"] == 16 and con["shapes"] == ["capped", "kind", "named"] and con["prompts"] == ["old", "anchor", "tail"]
    assert (m["steps"], m["seed"], m["policy"], m["hint_rules"], m["name_cap"], m["model"]) == (30, SEED, "repair_m6", False, 16, "Qwen/Qwen2.5-7B-Instruct-AWQ") and m["cells"] == {k: list(v) for k, v in T.CELLS.items()}
    assert [c["label"] for c in res["cells"]] == ORDER == T.ORDER and T.merge_cells(res["cells"]) == {k: res[k] for k in ("meta", "cells")}
    for c in res["cells"]:
        lab = c["label"]; f, k = JOBS[lab]
        assert (c["arm"], c["K"], c["first_episode"], c["prompt"], c["shape"], c["steps"], c["seed"], c["policy"], c["hint_rules"], c["engine"]) == ("S", k, f, T.CELLS[lab][2], T.CELLS[lab][3], 30, SEED, "repair_m6", False, e)
        assert c["grammar_sha256"] == GRAMMAR_SHA[lab] == T.grammar_sha256(c["shape"], f, k), lab
        assert [s["seed"] for s in c["sessions"]] == [SEED * 1000 + f + i for i in range(k)] and all(s["script_steps"] == 30 and s["stopped"] is None and len(s["log"]) == 30 for s in c["sessions"])
        L = logs(c); assert [x["t"] for s in c["sessions"] for x in s["log"]] == list(range(1, 31)) * k
        assert abs(secs(c) - c["wall_seconds"]) < 1e-6 and [r["round"] for r in c["rounds"]] == list(range(1, 31)) and all(r["active"] == 64 for r in c["rounds"])
        assert (sum(r["prompt_tokens"] for r in c["rounds"]), sum(r["cached_tokens"] for r in c["rounds"]), sum(r["new_tokens"] for r in c["rounds"])) == tuple(sum(x[key] for x in L) for key in ("n_prompt", "n_cached", "n_new"))
        assert all(0 <= x["n_cached"] <= x["n_prompt"] and 0 < x["n_new"] < 96 for x in L) and all(bool(x["exact"]) == (x["distance"] == 0) for x in L) and all(x["parsed"] or not x["admitted"] for x in L)
        for s in c["sessions"]:                                                                          # the reply is inside the language of its own request, by the expression recomputed here
            for x, q in zip(s["log"], script(job_of(s))):
                assert re.fullmatch(T.regex_for(c["shape"], q["instruction"]), x["text"]) and x["in_language"] is True and x["kind"] == q["kind"], (lab, job_of(s), x["t"])
                assert re.fullmatch(T.regex_for(c["shape"], q["instruction"]), q["ideal"]), "the oracle's own reply is inside the expression of its request"
        for s in c["sessions"]:                                                                          # the job played again through the typed session (the pinned session with a prompt and an expression)
            for rec, x in zip(replay_job(c, s), s["log"]): assert rec == (x["parsed"], x["admitted"], x["corrupted"], x["exact"], x["distance"], x["in_language"]), (lab, job_of(s), x["t"])
    assert sum(len(logs(c)) for c in res["cells"]) == 13440 and round(sum(c["wall_seconds"] for c in res["cells"]), 1) == 1708.1
    assert {lab: c["distinct_regexes"] for lab, c in cells(res).items()} == dict(F0=1, FK=3, FKA=3, FKT=3, FN=427, SK=3, SN=436)


# ---------------------------------------------------------------- 2 the lines
def test_the_seventeen_lines_read_as_the_result_says_by_the_code_and_by_hand_and_the_rule_about_claims_is_followed():
    res = run(); v = T.verdict_typed(res["cells"], third()["cells"]); assert [x for _, _, _, x in v] == READ and len(PROBS) == 17 == len(T.TL_LINES) and T.TL_HEADLINE == (6, 17)
    assert [(i, tag, text) for i, tag, text, _ in v] == [(i + 1, T.TL_LINES[i][0], T.TL_LINES[i][1]) for i in range(17)]
    assert sum(READ) == 15 and [i + 1 for i, x in enumerate(READ) if not x] == [6, 15] and [READ[i - 1] for i in T.TL_HEADLINE] == [False, True]
    assert lines_by_hand(res) == READ, "the seventeen lines from the raw numbers, by hand"
    assert [int(m_.group(1)) for m_ in re.finditer(r"\n\d+\. \[\w+\] .*\((\d+) %\)(?=\n)", _head())] == PROBS
    ca = T.claims_allowed(v); assert ca == dict(base=True, constraint_held=True, grammar_changed_only_what_it_forbids=True, expressions_as_defined=True, fresh_jobs_kind={5: True, 6: False, 7: True}, fresh_jobs_named={10: True},
                                               anchor_lines={11: True, 12: True}, seen_jobs={8: True, 9: True}, time_and_cost_in_one_session={13: True, 14: True, 15: False, 16: True, 17: True})
    S = {k: st(v_) for k, v_ in cells(res).items()}
    assert [S[k]["ex"] for k in ORDER] == [1833, 1835, 1920, 1839, 1920, 1906, 1920] and [S[k]["whole"] for k in ORDER] == [59, 60, 64, 61, 64, 62, 64] and [S[k]["lost"] for k in ORDER] == [87, 85, 0, 81, 0, 14, 0]
    assert [round(S[k]["wall"], 1) for k in ORDER] == [224.2, 223.5, 342.3, 244.8, 223.8, 224.8, 224.8] and [round(S[k]["eph"]) for k in ORDER] == [29437, 29560, 20191, 27048, 30882, 30529, 30747]
    assert [round(S[k]["sps"], 4) for k in ORDER] == [0.1168, 0.1164, 0.1783, 0.1275, 0.1166, 0.1171, 0.1171] and [round(S[k]["q1000"], 1) for k in ORDER] == [8.0, 8.0, 5.4, 7.4, 8.4, 8.3, 8.3]
    assert [better_same_worse(cells(res), a, b) for a, b in (("FK", "F0"), ("FKA", "F0"), ("FKT", "F0"), ("FN", "F0"), ("FKA", "FK"), ("FKT", "FK"), ("FN", "FK"))] == [(1, 63, 0), (5, 59, 0), (3, 59, 2), (5, 59, 0), (4, 60, 0), (2, 60, 2), (4, 60, 0)]
    R_ = _result()
    for needle in ("**Lines 2, 3 and 4 read yes**", "**Line 5 reads yes and line 6 reads NO.**", "**This Result does not claim that the grammar by kind raises the exactness.**", "15 of the 17 lines read yes; lines 6 and 15 read NO", "**Line 17 reads yes by the page's rule and this Result does not read it as a gain.**"): assert needle in R_, needle
    assert round(brier(PROBS, READ), 3) == 0.094 and round(brier([50] * 17, READ), 3) == 0.25


# ---------------------------------------------------------------- 3 the page
def test_the_part_above_the_result_is_what_was_written_before_the_run():
    head = _head(); assert _h(head) == PAGE_SHA256 and _text().count(RESULT_MARK) == 1 and head.rstrip("\n").endswith("\n\n---")
    # the hash holds the head; these read, one by one, the parts of it that the Result is read against (so that a head with a re-pinned hash would still have to say the same)
    for i, (tag, text) in enumerate(T.TL_LINES, 1): assert f"\n{i}. [{tag}] {text} ({PROBS[i - 1]} %)\n" in head, i
    for s_ in list(PINS.values()) + [REGEX_SHA256]: assert head.count(s_) == 1, s_
    for s_ in SIDE: assert head.count(s_) == 1, s_[:30]
    for s_ in RULE_LABELS: assert head.count(f"\n* *{s_}*") == 1, s_
    rules = head[head.index("lines 5 to 7 FK against F0"): head.index("line 17 the exact steps per GPU hour of FK higher than those of F0")]
    assert [float(x) for x in re.findall(r"`([0-9.]+)`", rules)] == [0.01, 0.99, 0.995, 0.995, 1.05, 1.15, 1.30, 1.20] == [THRESH[k] for k in ("l6", "l8", "l9", "l10", "l13", "l14", "l15", "l16")]
    assert "(not below; at least `0.01` above; no job with fewer exact steps)" in rules and "lines 11 and 12 FKA and FKT against FK (not below)" in rules and "at least 99.5 percent were accepted by the strict parser, none used all 96 new tokens" in head


def test_the_page_carries_exactly_the_result_this_file_renders_and_its_tables_say_what_the_rules_say():
    res = run(); text = _text(); assert text == _head().rstrip("\n") + render(res).rstrip("\n") + "\n"
    R_ = _result(); rows = [ln for ln in R_.split("\n") if re.match(r"\| \d+ \| \d+ % \|", ln)]; assert len(rows) == 17
    for i, ln in enumerate(rows, 1):
        cs = [c.strip() for c in ln.strip("|").split("|")]; assert int(cs[0]) == i and cs[1] == f"{PROBS[i - 1]} %" and cs[2] == measured(res, i) and cs[3] == ("yes" if READ[i - 1] else "NO"), i
    for tbl in (table_cells(res), table_cost(res), table_jobs(res), table_forced(res), table_parity(res), table_deletes(res)):
        for ln in tbl: assert ln in R_.split("\n"), ln
        assert "\n".join(tbl) in R_
    assert q_block(res) in R_ and q_block(res) in open(os.path.join(ROOT, "docs", "Q_FROM_SAVED_FILES.md")).read(), "the Q block of the Result is the block of the Q page"
    assert [ln for ln in R_.split("\n") if ln.startswith("## ")] == ["## The 17 lines", "## What the page's rule lets this Result claim", "## The jobs that missed", "## What the model still writes", "## The seen jobs",
                                                                     "## The parity check on jobs it was built to fit and on jobs it was not", "## Seconds and tokens", "## Every cell", "## The probe (Cell 6), read afterwards", "## Where my expectation was off, and where it was right", "## What this finds, in plain words", "## What this does not show"]


# ---------------------------------------------------------------- 4 the figures
def test_the_figures_quoted_in_the_result_are_recounted_from_the_records():
    res = run(); R_ = _result(); c = cells(res)
    for needle in ("vLLM 0.31.0", "GPU Tesla T4", "Python 3.13.15", FILE_SHA256, MODULE_SHA256[:16], THIRD_SHA256[:16], REGEX_SHA256[:16], "20 Oct 2026", "**All seven planned cells ran**", "add up to 1,708.1, which is 28.5 minutes",
                   "15 of the 17 lines read yes", "95.6 % exact against 95.5 %", "(1,835 and 1,833 of 1,920 steps)", "+0.1 points where line 6 asked for +1.0", "29,560 against 29,437 (+0.4 %)", "(223.5 to 224.8 seconds, 0.6 %)", "+53.2 % seconds per step",
                   "(job 150, step 9)", "30,882", "20,191", "8.4 per 1,000 tokens", "5.4 per 1,000 tokens", "in every one of the 681 delete replies it wrote the node alone", "1,899 steps for `FK`, 1,808 steps for `FN`, 1,860 steps for `SK`, 1,848 steps for `SN`",
                   "a Brier score of 0.094", "**4 of the 5 first misses of `F0` are replies to a Delete request.**", "4 in `FK`, 3 in `FKT` and none in `FKA` and `FN`", "(29 lost against 19)", "about 6 in 100",
                   "all 1,239 add replies and all 180 replies", "(1,419 of 1,419)", "501 node-only, 501 of them exact", "(649 replies)", "(590 replies)", "1,860 steps compared; forced at jobs 3, 17, 19, 62", "1,848 steps; forced at jobs 3, 4, 17, 19, 20, 62",
                   "it flags 14 steps", "14 of 14", "(53.2 % of the prompt tokens served from the cache against 69.0 %)", "175.0 prompt tokens a step against 108.5", "1.61 times", "119.7 tokens a step (1.10 times) and 9.5 % more seconds per step", "85 more exact steps", "-31.7 %",
                   "new ones in 30 of its 30 rounds", "its first round took 12.47 seconds against 11.86", "(g) `F0` is between 96.5 and 99.0 percent exact (75 %): **false,** 95.5 %", "97.2 % on jobs 0 to 63", "98.1 % on jobs 64 to 127", "(i) the seven cells take 20 to 40 minutes in all"):
        assert needle in R_, needle
    # plain recounts
    for lab, (steps, ex, jobs_hit_) in dict(F0=(1920, 1833, [136, 150, 172, 175, 178]), FK=(1920, 1835, [136, 172, 175, 178]), FKA=(1920, 1920, []), FKT=(1920, 1839, [136, 154, 175]), FN=(1920, 1920, []), SK=(1920, 1906, [4, 20]), SN=(1920, 1920, [])).items():
        L = logs(c[lab]); assert (len(L), sum(1 for x in L if x["exact"])) == (steps, ex) and jobs_hit(c[lab]) == jobs_hit_, lab
    wall = {lab: sum(r["seconds"] for r in c[lab]["rounds"]) for lab in ORDER}
    assert [round(wall[l], 1) for l in ORDER] == [224.2, 223.5, 342.3, 244.8, 223.8, 224.8, 224.8] and [round(3600 * sum(1 for x in logs(c[l]) if x["exact"]) / wall[l]) for l in ORDER] == [29437, 29560, 20191, 27048, 30882, 30529, 30747]
    assert round(wall["FK"] / wall["F0"], 3) == 0.997 and round(wall["FN"] / wall["F0"], 3) == 0.998 and round(wall["FKA"] / wall["FK"], 3) == 1.532 and round(wall["FKT"] / wall["FK"], 3) == 1.095
    same = [wall[k] for k in ("F0", "FK", "FN", "SK", "SN")]; assert (round(min(same), 1), round(max(same), 1), round((max(same) / min(same) - 1) * 100, 1)) == (223.5, 224.8, 0.6)
    assert round(wall["FKA"] - wall["FK"], 1) == 118.9 and 1920 - 1835 == 85 and round(100 * (20191 / 29560 - 1), 1) == -31.7


# ---------------------------------------------------------------- 5 the jobs that missed
EXPECT_MISSES = {"F0": [(136, 2, "DEL[0,1]", 19), (150, 9, "DEL[(0,1)->(0,0):owns#1] ADD[(0,0)->(0,1):dep#4]", 2), (172, 14, "DEL[(2,4)->(1,0):dep#6]", 17), (175, 2, "DEL[0,1]", 29), (178, 11, "DEL[(2,4)->(0,0):dep#8]", 20)],
                 "FK": [(136, 2, "DEL[0,1]", 19), (172, 14, "DEL[(2,4)->(1,0):dep#6]", 17), (175, 2, "DEL[0,1]", 29), (178, 11, "DEL[(2,4)->(0,0):dep#8]", 20)], "FKA": [],
                 "FKT": [(136, 2, "DEL[0,0]", 29), (154, 5, "DEL[0,3]", 23), (175, 2, "DEL[0,1]", 29)], "FN": [], "SK": [(4, 18, "ADD[1,0:6]", 13), (20, 30, "ADD[(1,5)->(2,1):dep#3]", 1)], "SN": []}


def test_the_jobs_that_missed_are_read_from_the_records_and_each_is_of_the_kind_the_result_says():
    res = run(); c = cells(res)
    for lab in ORDER: assert [(m["job"], m["step"], m["reply"], m["lost"]) for m in misses(c[lab])] == EXPECT_MISSES[lab], lab
    f0 = misses(c["F0"]); assert [slip_class(m) for m in f0] == ["wrong node", "extra delete before the add", "edge without the node", "wrong node", "edge without the node"]
    assert [m["kind_ok"] for m in f0] == [True, False, True, True, True] and [m["named_ok"] for m in f0] == [False] * 5 and [m["parity"] for m in f0] == [False] * 5 and [m["admitted"] for m in f0] == [False, True, False, False, True]
    assert [m["request"] for m in f0] == ["Delete the service at (1,0).", "Add a dependency edge from (0,0) to (0,1) with relation dep and weight 4.", "Delete the service at (2,4).", "Delete the service at (1,1).", "Delete the service at (2,4)."]
    assert [m["oracle"] for m in f0] == ["DEL[1,0]", "ADD[(0,0)->(0,1):dep#4]", "DEL[2,4] DEL[(1,0)->(2,4):dep#6]", "DEL[1,1] DEL[(0,3)->(1,1):owns#1]", "DEL[2,4] DEL[(2,4)->(0,0):dep#8]"]
    assert [m["kind"] for m in f0] == ["delete_leaf", "add_edge", "delete_dep", "delete_dep", "delete_dep"] and sum(1 for m in f0 if m["kind"].startswith("delete")) == 4
    sk = misses(c["SK"]); assert [slip_class(m) for m in sk] == ["wrong number", "wrong number"] and [m["kind_ok"] for m in sk] == [True, True]
    # the eight first misses of the earlier runs (jobs 0 to 127) were all replies to an Add request
    early = [m for cell in (third_cell(64), cells(retry())["H0"]) for m in misses(cell)]; assert [m["job"] for m in early] == [3, 4, 17, 19, 20, 62, 101, 122] and all(m["kind"] in ("add_node", "add_edge") and m["request"].startswith("Add") for m in early)
    # the four different wrong-node replies of the fresh cells all write 0 where the request has 1 as the first coordinate
    wn = sorted({(m["job"], m["reply"], T.parse_request(m["request"])["node"]) for lab in FRESH for m in misses(c[lab]) if slip_class(m) == "wrong node"})
    assert wn == [(136, "DEL[0,0]", ("1", "0")), (136, "DEL[0,1]", ("1", "0")), (154, "DEL[0,3]", ("1", "3")), (175, "DEL[0,1]", ("1", "1"))] and all(re.match(r"DEL\[0,", r) and n[0] == "1" for _, r, n in wn)
    # FKA against FK: four jobs from missing to exact and none the other way; the chance
    assert sorted(set(jobs_hit(c["FK"])) - set(jobs_hit(c["FKA"]))) == [136, 172, 175, 178] and set(jobs_hit(c["FKA"])) - set(jobs_hit(c["FK"])) == set() and round(fisher_one_sided(4), 3) == 0.060 and math.comb(64, 4) * 1 / math.comb(128, 4) == fisher_one_sided(4)
    assert sorted(set(jobs_hit(c["FKT"])) - set(jobs_hit(c["FK"]))) == [154] and sorted(set(jobs_hit(c["FK"])) - set(jobs_hit(c["FKT"]))) == [172, 178] and misses(c["FKT"])[0]["lost"] == 29 and misses(c["FK"])[0]["lost"] == 19
    # the jobs the grammar forced in FK is job 150 alone, and the forced reply is the oracle's own text
    n, fl = forced(c["FK"], c["F0"], "kind"); assert fl == [(150, 9, "DEL[(0,1)->(0,0):owns#1] ADD[(0,0)->(0,1):dep#4]", "ADD[(0,0)->(0,1):dep#4]", "ADD[(0,0)->(0,1):dep#4]")]
    from collections import Counter
    assert Counter(q["kind"] for e_ in range(128, 192) for q in script(e_)) == Counter(add_node=649, add_edge=590, delete_dep=501, delete_leaf=180)


# ---------------------------------------------------------------- 6 what the model wrote in the deletes
def test_what_the_model_still_writes_in_the_deletes_is_counted_and_the_framework_completes_the_edges():
    res = run(); c = cells(res); d = {lab: delete_facts(c[lab]) for lab in ORDER}
    assert d["FN"] == dict(dep=501, dep_node_only=501, dep_node_only_exact=501, dep_edge_item=0, leaf=180, leaf_equal=180, add=1239, add_equal=1239)
    assert d["FKA"] == dict(dep=501, dep_node_only=501, dep_node_only_exact=501, dep_edge_item=0, leaf=180, leaf_equal=180, add=1239, add_equal=1239)
    assert d["F0"] == dict(dep=501, dep_node_only=499, dep_node_only_exact=482, dep_edge_item=2, leaf=180, leaf_equal=179, add=1239, add_equal=1238) and d["FK"] == dict(dep=501, dep_node_only=499, dep_node_only_exact=482, dep_edge_item=2, leaf=180, leaf_equal=179, add=1239, add_equal=1239)
    assert d["FKT"] == dict(dep=501, dep_node_only=501, dep_node_only_exact=484, dep_edge_item=0, leaf=180, leaf_equal=179, add=1239, add_equal=1239)
    assert d["SK"] == dict(dep=508, dep_node_only=508, dep_node_only_exact=506, dep_edge_item=0, leaf=171, leaf_equal=171, add=1241, add_equal=1239) and d["SN"] == dict(dep=508, dep_node_only=508, dep_node_only_exact=508, dep_edge_item=0, leaf=171, leaf_equal=171, add=1241, add_equal=1241)
    assert sum(1 for lab in ORDER for x in logs(c[lab]) if x["kind"] == "delete_dep" and any("->" in i for i in items_of(x["text"]))) == 4 and [(m["job"], lab) for lab in ORDER for m in misses(c[lab]) if slip_class(m) == "edge without the node"] == [(172, "F0"), (178, "F0"), (172, "FK"), (178, "FK")]
    # in FN every delete of a service with connections is the node alone, the gate does not admit it, and the held state is the oracle's
    fn = [x for x in logs(c["FN"]) if x["kind"] == "delete_dep"]; assert len(fn) == 501 and all((not x["admitted"]) and x["exact"] and x["parsed"] for x in fn)
    # the same, with the framework's own gate and repair, for one of the 501 (job 172, step 14: the node alone, the edge is the oracle's other item)
    s = P.Session("S", SEED * 1000 + 172, STEPS)
    while s.t < 13: rq = s.request(); s.accept(P.Reply(rq.stand_in_reply, 0, 0, 0))
    dlt = strict_parse("DEL[2,4]"); assert not check(s.live, dlt).ok and same_graph(K.next_state("repair_m6", s.live, dlt), s.oracle[14]) and script(172)[13]["ideal"] == "DEL[2,4] DEL[(1,0)->(2,4):dep#6]"
    assert K.POLICIES[-1] == "repair_m6" and P.POLICY == "repair_m6"
    # the model's share in FN: the value of an added service and the weight of an added edge
    assert sum(1 for x in logs(c["FN"]) if x["kind"] == "add_node") == 649 and sum(1 for x in logs(c["FN"]) if x["kind"] == "add_edge") == 590 and 649 + 590 == 1239


# ---------------------------------------------------------------- 7 the replay rule on the four pairs
def test_the_replay_rule_holds_on_the_four_pairs_and_every_forced_step_gives_a_job_that_is_exact_to_the_end():
    res = run(); c = cells(res)
    want = {"FK": (1899, [(150, 9)]), "FN": (1808, [(136, 2), (150, 9), (172, 14), (175, 2), (178, 11)]), "SK": (1860, [(3, 16), (17, 20), (19, 12), (62, 12)]), "SN": (1848, [(3, 16), (4, 18), (17, 20), (19, 12), (20, 30), (62, 12)])}
    for name, b, sh in PAIRS:
        n, fl = forced(c[name], base_of(res, b), sh); assert (n, [(j, s) for j, s, *_ in fl]) == want[name], name
        assert same_text_until_forced(c[name], base_of(res, b), sh) and T.same_until_forced(c[name], base_of(res, b), sh)
        for j, s, was, now, orc in fl:
            assert not T.in_language(sh, script(j)[s - 1]["instruction"], was) and T.in_language(sh, script(j)[s - 1]["instruction"], now), (name, j)       # the base reply was outside, the new one is inside
            assert all(x["exact"] for q in c[name]["sessions"] if job_of(q) == j for x in q["log"]), (name, j)                                                  # and the job is exact to the end
            if name in ("FK", "SK", "SN"): assert now == orc, (name, j)
    # the forced replies of FN that are not the oracle's text are the node alone, which the repair completes
    assert [(j, now) for j, s, was, now, orc in forced(c["FN"], c["F0"], "named")[1] if now != orc] == [(172, "DEL[2,4]"), (175, "DEL[1,1]"), (178, "DEL[2,4]")]
    # SK: jobs 4 and 20 slip again at steps 18 and 30 with the text of the third run (their replies are inside the kind language)
    t3 = third_cell(64)
    for j, step in ((4, 18), (20, 30)):
        a = next(q for q in c["SK"]["sessions"] if job_of(q) == j)["log"][step - 1]; b = t3["sessions"][j]["log"][step - 1]; assert a["text"] == b["text"] and not a["exact"] and T.in_language("kind", script(j)[step - 1]["instruction"], b["text"])
    assert 1920 - 14 == 1906 == st(c["SK"])["ex"] and [m["lost"] for m in misses(c["SK"])] == [13, 1] and "the most it could be is 99.3" in _head()


# ---------------------------------------------------------------- 8 the parity check
def test_the_parity_check_flags_every_first_miss_of_the_seven_cells_and_no_other_step():
    res = run(); c = cells(res); pf = {lab: parity_facts(c[lab]) for lab in ORDER}
    assert [pf[lab]["flagged"] for lab in ORDER] == [5, 4, 0, 3, 0, 2, 0] and [pf[lab]["first_misses"] for lab in ORDER] == [5, 4, 0, 3, 0, 2, 0] and all(pf[lab]["flagged_exact"] == 0 and pf[lab]["first_misses_unflagged"] == 0 and pf[lab]["at_first_miss"] == pf[lab]["flagged"] for lab in ORDER)
    assert sum(pf[lab]["flagged"] for lab in ORDER) == 14 and sum(len(logs(c[lab])) for lab in ORDER) == 13440
    # the check, applied by hand: the first item of the reply in the words of the target, and exactly one item for an add
    for lab in ORDER:
        for s in c[lab]["sessions"]:
            for x, q in zip(s["log"], script(job_of(s))):
                by_hand = T.describe_reply(x["text"]) == T.target_body(q["instruction"]) and (q["kind"].startswith("delete") or len(items_of(x["text"])) == 1)
                assert by_hand == T.parity_ok(q["instruction"], x["text"]), (lab, job_of(s), x["t"])
                assert by_hand or not x["exact"], (lab, job_of(s), x["t"])
        for m in misses(c[lab]): assert m["parity"] is False and not T.parity_ok(m["request"], m["reply"])
    # it is a probe only: no step of a session is decided by it
    assert "parity_ok" not in open(MODULE).read().split("class TypedSession")[1].split("# ---------------------------------------------------------------- the engines")[0]


# ---------------------------------------------------------------- 9 seconds, tokens, Q
def test_the_seconds_the_tokens_and_the_q_block_of_the_cells_are_recomputed():
    res = run(); c = cells(res); S = {k: st(v) for k, v in c.items()}
    assert [round(S[k]["comp_step"], 1) for k in ORDER] == [108.5, 108.5, 175.0, 119.7, 108.7, 109.3, 109.3] and [round(100 * S[k]["cached_share"], 1) for k in ORDER] == [69.0, 69.0, 53.2, 66.8, 68.9, 68.7, 68.7]
    assert [round(S[k]["prompt"] / S[k]["n"], 1) for k in ORDER] == [350.0, 350.0, 374.3, 360.7, 349.7, 349.6, 349.7] and [round(S[k]["new"] / S[k]["n"], 2) for k in ORDER] == [10.46, 10.45, 10.44, 10.44, 10.44, 10.53, 10.53]
    assert [S[k]["m_tok"] for k in ORDER] == [228379, 228374, 356091, 249814, 228793, 230036, 230050] and [S[k]["unparsed"] for k in ORDER] == [0] * 7 and [S[k]["full96"] for k in ORDER] == [0] * 7
    assert round(S["FKA"]["comp_step"] / S["FK"]["comp_step"], 2) == 1.61 and round(S["FKT"]["comp_step"] / S["FK"]["comp_step"], 2) == 1.10 and round(S["FKT"]["comp_step"] - S["FK"]["comp_step"], 1) == 11.2
    assert (S["FN"]["new_rounds"], S["FN"]["rounds"], S["SN"]["new_rounds"], S["F0"]["new_rounds"], S["FK"]["new_rounds"]) == (30, 30, 30, 1, 1) and (S["FN"]["new_grammars"], S["SN"]["new_grammars"]) == (427, 436)
    assert (round(S["FN"]["first_round"], 2), round(S["F0"]["first_round"], 2)) == (12.47, 11.86)
    # the page's own estimates
    assert "about 181 computed tokens a step instead of 109" in _head() and "I expect the top to cost 30 to 60 percent more seconds per step" in _head()
    # the Q block of this run is the one q_measure gives for the file, and the page has it
    rows = Q.all_rows(); assert list(rows)[-1] == "fifth run, typed requests" and [r["cell"] for r in rows["fifth run, typed requests"]] == ORDER
    for r in rows["fifth run, typed requests"]:
        s = S[r["cell"]]; assert (r["exact_steps"], r["m_tokens"], r["steps"]) == (s["ex"], s["m_tok"], s["n"]) and abs(r["wall_seconds"] - s["wall"]) < 1e-9 and abs(r["exact_per_hour"] - s["eph"]) < 1e-9 and abs(r["exact_per_1000_tokens"] - s["q1000"]) < 1e-9
    assert q_block(res) in Q.render_table(rows)


# ---------------------------------------------------------------- 10 claims
def test_the_result_claims_what_the_pages_rule_allows_and_nothing_more():
    R_ = _result(); low = R_.lower()
    assert "**Never**, as the page says: a multiple of capacity, a count of GPUs saved, dollars or a percentage of cost, \"100 percent\" for a grammar, \"proven\", a statement about any provider, a statement about requests in free words, or about a model that was not run." in R_
    for w in ("proven", "gpus saved", "dollars", "capacity"):
        for m_ in re.finditer(w, low):
            ctx = low[max(0, m_.start() - 140): m_.end() + 90]; assert any(k in ctx for k in ("never", "neither", "not measured", "does not show", "not claim", "was not", "were not", "not ruled", "a capacity in", "cache size", "capacity source", "capacity_source", "capacity_tokens", "block override", "capacity ")), (w, ctx)
    assert "$" not in R_ and R_.count("100 percent") == 1 and R_.count("proven") == 1 and "100 %" not in R_ and "perfect" not in low and "×" not in R_
    for q in ("this Result does not read it as a gain", "does not claim that the grammar by kind raises the exactness", "it is not the model reading the request", "it does not say that the model reads requests", "That the model reads requests (in the grammar that names the coordinates the host writes them)",
              "what the probe means for the loop (eight steps, one reply under each variant, descriptions the page said would never decide a step)", "One run, 64 jobs.", "in that grammar the host writes the coordinates of the request into the reply, the model writes the value of an added service or the weight of an added edge, and in every one of the 681 delete replies it wrote the node alone"):
        assert _norm(q) in _norm(R_), q
    # the sentence the page demands for the grammar that names the coordinates is in the same sentence as its figures
    sent = next(s for s in R_.split("\n") if s.startswith("**The sentence the page allows")); assert "30,882 exact steps per GPU hour" in sent and "the host writes the coordinates of the request into the reply" in sent
    for q in ("the next thing to try", "requests in free words"): assert q in _head() or q in R_
    for q in ("**say so in the same sentence.**", "Line 5 yes, line 6 NO:* too few slips of the removable kind among the new jobs to see one point, or the forced replies were wrong.", "*Line 15 NO, line 16 yes:* as the estimate said."): assert _norm(q) in _norm(_head()), q


# ---------------------------------------------------------------- 11 the probe printout (Cell 6, added on 20 Oct 2026 in the evening)
def test_the_probe_printout_agrees_with_the_saved_runs_and_its_side_predictions_are_read_from_it():
    assert _sha(PROBE) == PROBE_SHA256
    smoke, recs, tail = probe()
    assert smoke == [(1, 423, 0, 9, "ADD[1,4:7]"), (2, 452, 192, 16, "ADD[(2,2)->(0,1):dep#3]"), (3, 465, 192, 16, "ADD[(0,3)->(1,4):dep#5]")]
    assert tail == ["", "OK: cached counts are reported, every reply is well formed and inside the expression of its request, and the probe's replies end. You can go on.", ""]
    assert [(r["job"], r["step"], r["earlier"]) for r in recs] == [tuple(x) for x in T.KNOWN_SLIPS] and [r["job"] for r in recs] == [3, 4, 17, 19, 20, 62, 101, 122]
    early = {m["job"]: m for cell in (third_cell(64), cells(retry())["H0"]) for m in misses(cell)}
    for r in recs:                                                                                       # every line of the printout against the saved runs and the module
        sc = script(r["job"])[r["step"] - 1]; e = early[r["job"]]
        assert (r["instr"], r["oracle"], r["kind"]) == (sc["instruction"], sc["ideal"], sc["kind"]) and r["kind"] in ("add_node", "add_edge") and (e["step"], e["reply"]) == (r["step"], r["earlier"]), r["job"]
        assert r["in_kind"] == T.in_language("kind", r["instr"], r["earlier"]) and r["in_named"] == T.in_language("named", r["instr"], r["earlier"]) and r["pturn"] == T.parity_text(r["instr"], r["earlier"]), r["job"]
        assert list(r["var"]) == list(PROBE_VARIANTS)
        for v, (text, eq) in r["var"].items(): assert (text == r["oracle"]) == eq and re.fullmatch(T.regex_for(PROBE_SHAPE[v], r["instr"]), text), (r["job"], v)
    # the counts, again by hand
    ok = lambda v: [r["job"] for r in recs if r["var"][v][0] == r["oracle"]]; same = lambda v: [r["job"] for r in recs if r["var"][v][0] == r["earlier"]]; every = [3, 4, 17, 19, 20, 62, 101, 122]
    assert ok("old") == [101] and ok("kind") == [3, 17, 19, 62, 101] and ok("named") == ok("kind+top") == ok("kind+tail") == every and ok("parity") == [20]
    assert same("old") == [3, 4, 17, 19, 20, 62, 122] and same("kind") == [4, 20, 122] and same("parity") == [3, 17, 19, 62] and same("named") == same("kind+top") == same("kind+tail") == []
    assert [r["job"] for r in recs if r["var"]["parity"][0] == r["oracle"] + " " + r["earlier"]] == [4, 101] and recs[-1]["var"]["parity"][0] == "DEL[(0,0)->(2,2):dep#8] ADD[(0,0)->(2,2):dep#3]" and len(recs[-1]["var"]["parity"][0].split(" ")) == 2
    assert [r["job"] for r in recs if not r["in_kind"]] == [3, 17, 19, 62, 101] and [r["job"] for r in recs if r["in_kind"]] == [4, 20, 122] and all(r["in_kind"] != (r["job"] in (3, 17, 19, 62, 101)) for r in recs)
    assert sorted(m["kind"] for m in misses(cells(run())["FKT"])) == ["delete_dep", "delete_dep", "delete_leaf"]
    F = probe_facts(recs)
    assert F["oracle"] == {"old": 1, "kind": 5, "named": 8, "kind+top": 8, "kind+tail": 8, "parity": 1} and F["earlier"] == {"old": 7, "kind": 3, "named": 0, "kind+top": 0, "kind+tail": 0, "parity": 4}
    assert (F["kind_on_shape"], F["kind_on_number"], F["top_on_number"], F["parity_both"], F["parity_any"], F["parity_job4"]) == (5, 0, 3, [4, 101], [20], False)
    assert F["side"] == dict(b=True, c=True, d6=True, d8=True, e=True, f4=False, fany=True)
    # the side predictions as the page wrote them (b) to (f), read once more from the counts
    assert len(ok("old")) <= 1 and len(ok("kind")) >= 4 and len(ok("named")) >= 6 and len(ok("named")) == 8 and any(r["var"]["kind+top"][1] for r in recs if r["job"] in (4, 20, 122)) and not any(r["var"]["parity"][1] for r in recs if r["job"] == 4) and any(r["var"]["parity"][1] for r in recs if r["job"] in (4, 20, 122))
    R_ = _result(); k = R_.index("## The probe (Cell 6), read afterwards"); sec = R_[k: R_.index("## Where my expectation was off")]
    assert "\n".join(table_probe(recs)) in sec and sec.count("\n| ") == 9 and f"`{PROBE_SHA256}`" in sec and "docs/results/axi_provider_typed_probe.txt" in sec
    for q in ("On all 8 steps the request, the oracle's reply, the earlier reply, whether that reply is inside the `kind` and the `named` language and the parity turn are what the saved files and the module give, and all 48 replies are inside the expression of their own variant.",
              "The three smoke steps show prompts of 423, 452 and 465 tokens of which 0, 192 and 192 came from the cache: the first 192 tokens are the system prompt, as the page expected.",
              "the earlier reply again at 7 of the 8 steps. At job 101 it wrote the oracle's reply this time, so that step says nothing about the other variants; the printout does not say why and I did not test it.",
              "the oracle's reply at 5 of the 5 steps whose earlier reply was outside the kind language (jobs 3, 17, 19, 62, 101) and at 0 of the 3 number slips (jobs 4, 20, 122), where it wrote the earlier reply again.",
              "the oracle's reply at all 8 steps. In `kind+top` and `kind+tail` the host writes nothing of the reply (the grammar only fixes its shape); the target line is in the prompt and the model wrote the right number at the 3 number slips. All 8 steps are Adds.",
              "gave 95.8 % against 95.6 % for `FK` on the fresh jobs: its 3 first misses were all Deletes (delete_leaf, delete_dep, delete_dep), which this probe does not ask.",
              "the oracle's reply alone at 1 of the 8 steps (job 20); the oracle's reply followed by the earlier reply at 2 (jobs 4, 101); the earlier reply again at 4; at job 122 the right coordinates with a `DEL` in front of them.", "no step of the loop uses it."):
        assert q in sec, q[:60]
    for q in ("(b) with the capped expression and the old prompt the probe writes the earlier slip again at 7 or more of the 8 steps (88 %): true, 7 of 8.", "(c) the `kind` variant writes the oracle's reply at 4 or more of the 5 steps whose earlier reply is outside the kind language (65 %): true, 5 of 5.",
              "(d) the `named` variant writes the oracle's reply at 6 or more of the 8 steps (70 %): true, and at all 8 (35 %): true.", "(e) the anchor at the top writes the oracle's reply at least once for the three number slips (40 %): true, for 3 of the 3.",
              "(f) the parity turn gets the oracle's reply for job 4 (40 %): **false** (it wrote the oracle's reply and then the earlier one); for at least one of jobs 4, 20 and 122 (55 %): true, job 20.", "**Of the twelve statements the page made as side predictions, 10 read true and 2 false.**"):
        assert q in R_, q[:60]
    assert "are not read" not in R_ and "not in the saved file; nothing" not in R_ and "(b) to (f) are about the probe" not in R_
    assert "The probe printed afterwards points the same way on 8 steps that earlier runs missed: the grammar by kind wrote the oracle's reply at 5 of the 5 shape slips and at 0 of the 3 number slips, and naming the coordinates, or the target line at the top or at the end, wrote it at all 8." in R_
    # the page said the probe is a description: nothing in a session may call it
    src = open(MODULE).read(); assert "known_slip_probe" not in src.split("class TypedSession")[1].split("# ---------------------------------------------------------------- the engines")[0]


# ---------------------------------------------------------------- 12 the front pages
def test_the_front_pages_say_what_the_run_found_and_the_next_free_line_moved():
    res = run(); c = cells(res); S = {k: st(v) for k, v in c.items()}; dFN = delete_facts(c["FN"])
    cm = open(os.path.join(ROOT, "docs", "RAJNISH_CONCEPT_MAP.md")).read(); ve = open(os.path.join(ROOT, "docs", "VERIFIED.md")).read()
    item = cm[cm.index("111. **"): cm.index("112. **")]; k = ve.index("Provider-side serving test, fifth run"); row = ve[k: ve.index("\n", k)]
    assert "\n111. **" not in item[3:] and item.count("**Result:**") == 1 and "@@" not in item and "@@" not in row and "NOT RUN" not in item and "NOT RUN" not in row
    shares = (pc(S["F0"]["share"]), pc(S["FK"]["share"])); eph = (num(round(S["FK"]["eph"])), num(round(S["F0"]["eph"])))
    for need in ("**Status: RUN (the owner's Colab run, seven cells in one session; the file reached the repository work on 20 Oct 2026); the Result is on the page `PROVIDER_SIDE_TYPED_TEST.md`, which was hash-locked before the run (sha256 d4c963599e8dfe67... of the text above the Result).",
                 "test 130 checks the first three, test 131 the last two and test 132 recomputes the Result from the saved file.**",
                 "15 of the 17 lines read yes; lines 6 and 15 read NO; none is \"not run\"", "the headline, lines 6 and 17, read NO and yes where both yes stood at about 35 %", "13,440 of 13,440 replies accepted by the strict parser and inside the language of their own request; 0 of 13,440 steps corrupted",
                 f"On the 64 new jobs the capped expression gave {shares[0]} exact and the grammar by kind {shares[1]} (1,833 and 1,835 of 1,920 steps; 0.117 and 0.116 seconds per step; {eph[1]} and {eph[0]} exact steps per GPU hour, +0.4 %, the size of the spread between cells that did the same work)",
                 "**the Result does not claim that the grammar by kind raises the exactness.**", "Four of the 5 jobs that slipped in `F0` slipped on a Delete (two named the wrong service, two wrote an edge and left the service out)", "the eight earlier first misses were all Adds",
                 "each had all 1,920 steps exact", f"`FN` did it at 0.117 seconds per step and {num(round(S['FN']['eph']))} exact steps per GPU hour, but the host wrote the coordinates it had read from the sentence, and the model wrote the value or the weight and, in each of the {dFN['dep'] + dFN['leaf']} deletes, the node alone (the framework's repair completes the edges): it does not show that the model reads requests.**",
                 f"`FKA` did it with the model choosing the node, at +53 % seconds per step, {num(round(S['FKA']['eph']))} exact steps per GPU hour and {S['FKA']['q1000']:.1f} per 1,000 tokens, with four jobs changed from missing to exact and none the other way (one run, 64 jobs); the line at the end only (`FKT`) did not ({pc(S['FKT']['share'])})",
                 "`SK` has 1,906 exact steps (the ceiling) and `SN` 1,920, as the replay rule said", "The parity check flagged all 14 first misses of the seven cells and no other step.", "The probe printout (Cell 6, sent the same evening and saved as `axi_provider_typed_probe.txt`) agrees with the saved runs on all 8 steps, and of the page's twelve side predictions 10 read true and 2 false: the grammar by kind wrote the oracle's reply at the 5 shape slips and at none of the 3 number slips, `named` and the target line at the top or at the end at all 8 (all 8 are Adds), the parity turn at 1 of 8.",
                 "from the five saved runs", "the Result recomputed from the saved file (test 132, @N132@ tests)".replace("@N132@", N132), "mutation check of test 132: ", "mutation check of test 130: 250 deliberate changes (170 in the code, 40 digits and 15 words of the locked page with the test's page hash re-pinned, 25 digits in the notebook), 211 caught at once; the 39 that got through", "**Not shown:** that the model reads requests, that the grammar by kind or the target line raises the exactness in general, what the probe means for the loop (eight steps, one reply under each variant)"):
        assert need in item, need[:100]
    for need in ("**Run (the owner's Colab run, seven cells in one session, Tesla T4, vLLM 0.31.0; the file reached the repository work on 20 Oct 2026).**", "(sha256 d4c963599e8dfe67... of the locked text); the Result is on the page and is checked by test 132.", "**15 of 17 lines yes; lines 6 and 15 NO; none not run**",
                 "(the two NO stood at 38 and 15 % on the page", "13,440 of 13,440 replies accepted by the strict parser and inside the language of their own request",
                 f"the exact share is {shares[0]} with the capped expression and {shares[1]} with the grammar by kind (1,833 and 1,835 of 1,920) at 0.117 and 0.116 seconds per step and {eph[1]} and {eph[0]} exact steps per GPU hour (+0.4 %, the size of the spread of cells that did the same work)",
                 "**the Result does not claim that the grammar by kind raises the exactness**; four of the five jobs that slipped slipped on a Delete, which that grammar leaves free.",
                 "The grammar that names the coordinates (`FN`) and the target line at the top and the end (`FKA`) each had all 1,920 steps exact: `FN` at 0.117 seconds per step and " + num(round(S['FN']['eph'])) + " exact steps per GPU hour, ",
                 "On the seen jobs `SK` has 1,906 exact steps (the ceiling) and `SN` 1,920, as the replay rule said.", f"**but the host wrote the coordinates it read from the sentence and the model wrote the value or the weight and, in each of the {dFN['dep'] + dFN['leaf']} deletes, the node alone (the repair completes the edges): it does not show that the model reads requests;**",
                 f"`FKA` at +53 % seconds per step, {num(round(S['FKA']['eph']))} exact steps per GPU hour and {S['FKA']['q1000']:.1f} per 1,000 tokens (four jobs changed from missing to exact, none the other way; one run)", "The parity check flagged all 14 first misses of the seven cells and no other step.",
                 "The rule stands: never a multiple of capacity, a count of GPUs saved, dollars, \"100 percent\" or \"proven\".", "**No route in this design promises 100 %.**", "the Result recomputed from the saved file (test 132, @N132@ tests)".replace("@N132@", N132), "mutation check of test 132: ",
                 "(test 131, 7 tests)", "from the five saved runs and compared with its page character by character", "mutation check of test 130: 250 deliberate changes (170 in the code, 40 digits and 15 words of the locked page with the test's page hash re-pinned, 25 digits in the notebook), 211 caught at once; the 39 that got through", "The files it imports are pinned by sha256 in the test, so the core is unchanged.",
                 "the replay rule on the third run's saved 64-job cell (1,860 and 1,848 steps compared)", "Lines 2, 3 and 4 read yes (13,440 of 13,440 replies accepted by the strict parser and inside the language of their own request; every pair replays as the rule says), so the figures are claimed as read.", "On the 64 jobs that no earlier run had seen the exact share is", "The Q table (exact steps per GPU hour and per 1,000 tokens, never mixed into one number) now has the fifth run's seven cells.", "Not shown: that a grammar raises the exactness on new jobs (the kind grammar did not), that the model reads requests, what the probe means for the loop",
                 "The probe printout (Cell 6) was added afterwards (`axi_provider_typed_probe.txt`) and agrees with the saved runs on all 8 steps: the `kind` variant wrote the oracle's reply at the 5 shape slips and at none of the 3 number slips, `named` and the target line at the top or at the end at all 8 (all Adds), the parity turn at 1 of 8; of the page's twelve side predictions 10 read true and 2 false."):
        assert need in row, need[:100]
    for text in (item, row):
        assert f"mutation check of test 132: {MUT132}" in text and "mutation check of test 131: 90 deliberate changes (20 in the code, 45 digits and 25 words of the page with the test's page hash re-pinned), 77 caught; none of the 20 code changes and none of the 45 digits got through, the 13 survivors are words of the prose" in text
        assert "mutation check of test 130: " in text and "mutation check of test 131: " in text and "mutation check of test 132: " in text and "proven" in text and "dollars" in text and "energy" in text and "capacity" in text
    N_NEXT = "Next free: row 170, item 114, test 136, question 121"
    assert N_NEXT in cm and "test 130 checks the typed-requests test, test 131 the Q = O / M module and test 132 the Result of the typed-requests test, item 111; test 133 checks the 100-step serving test, item 112, and test 134 its Result" in cm and cm.count("Next free:") == 1
    assert cm.index("110. **") < cm.index("111. **") < cm.index("Next free: row")
    for name in ("test_provider_side.py", "test_provider_side_constrained.py", "test_provider_side_bounded.py", "test_provider_side_bounded_result.py"):
        t = open(os.path.join(HERE, name)).read(); assert "item 112, test 133" not in t and "item 114, test 136" in t, name
    for name in ("test_provider_side_retry.py", "test_provider_side_typed.py"):                       # these two name the old counter in the assertion that moves it
        assert "item 114, test 136" in open(os.path.join(HERE, name)).read(), name


N132 = "13"                                                                                         # the number of tests in this file
# the mutation check of this file as the two front pages state it (the check was made on this file before this constant was added)
MUT132 = ("570 deliberate changes (150 digits and 50 words of the Result, 30 digits of the locked head with the test's page hash re-pinned, 100 numbers of the saved file with its hash re-pinned in the test and on the page, 40 digits of the two front pages, 200 in the test's own code; "
          "a change of the head or of a front page that this test let pass was also run through test 130); the first run of this kind let 64 through (digits of the locked head and of the front pages that the test did not read one by one, the eight file fingerprints and the side predictions of the page, "
          "the parity check applied by hand to every reply, the count of unparsed replies) and the test was extended; the second run caught 547 and let 23 through: 5 digits in the VERIFIED row, which new assertions now read (run again alone against the final test: caught), "
          "and 18 changes in the test's own code that cannot be told apart from the original (11 `and` to `or` in assertions whose terms are all true, a `<=` to `<` and a `<` to `<=` that no figure touches, a number inside a failing message and one in the negative half of a check, "
          "a tolerance of 1e-9 made 1e-10, two roundings that print the same figure); no change to the Result's numbers or words and none to the saved file got through in either run. "
          "The part added in Update 66 (the probe printout: test 11 of 13 and its section of the Result) was checked in a run of its own: 285 changes (70 digits and 25 words of the new section of the Result, 60 digits of the printout with its hash re-pinned in the test and on the page, 30 digits of the two front pages, 100 in the new code), 268 caught and 17 let through: "
          "13 changes in the test's code that cannot be told apart from the original (5 `and` to `or` in assertions whose terms are all true, 5 changes of a threshold or a job number that the counts clear anyway, such as \"at least 1\" made \"at least 2\" where the count is 3, a slice length in a failing message, a zip over five lines made over six, a word inside a string) and 4 digits of one sentence of the VERIFIED row, which a new assertion now reads (run again alone: caught)")


if __name__ == "__main__":
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]; bad = 0
    for n, f in fns:
        try: f(); print("ok  ", n)
        except Exception: bad += 1; print("FAIL", n); traceback.print_exc()
    print(f"{len(fns) - bad}/{len(fns)} passed"); sys.exit(1 if bad else 0)
