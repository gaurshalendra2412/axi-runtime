"""Implements concept-map item 112 (Result): the sixth provider-side serving run, jobs of 100 steps, four cells (`L0` the capped expression and `LN` the grammar that names the coordinates on jobs 192 to 255,
`L8` (exact state, named) and `LC` (the whole chat kept, named) on jobs 192 to 199), made by the owner on Colab (vLLM 0.31.0, Tesla T4, Qwen2.5-7B-Instruct-AWQ, a cache of 80,000 tokens).
Page: docs/PROVIDER_SIDE_LONG_TEST.md (the question, the design, the 17 reading lines and the predictions were written and hash-locked before the run; the Result is appended below them).
The file: docs/results/axi_provider_long.json (the merged file the notebook's last cell wrote and the owner sent).
This file recomputes from the saved file, with its own arithmetic and not with the code that printed the figures:
  1. the file: its sha256, the settings, the seeds, the rounds adding up to the cells, and every one of the 14,004 steps played again from the text of its reply through the framework's own session, gate,
     repair and diagnosis (parsed, admitted, corrupted, exact, distance and inside-the-language of every step);
  2. the 17 reading lines (16 yes; line 11 NO; none not run) by the code and by hand from the raw numbers, and what the page's rule lets the Result claim;
  3. the replies of `L0` that are outside the named language (16, in 14 jobs: every first slip), how long each slip lasted, what the model wrote in the deletes of `LN`, the kept chat's stops;
  4. the predictions written beside the lines, read afterwards: the ones the run did not bear out are said so;
  5. the whole Result is `render(result)` and the page must equal the locked head followed by it; the figures quoted in it are recounted;
  6. that the Result claims what the page's rule allows and nothing more (no multiple of capacity, no GPUs saved, no dollars, no 'proven', no '100 percent' for a grammar, nothing about requests in free words);
  7. the two front pages (the concept map's item 112 and the VERIFIED row) say what was run and what was read, and the counter has moved.
What it does NOT show: anything about requests in free words, tasks of more than 100 steps, another model, GPU, engine or cache size, a kept chat with a summary or a window, dollars, energy."""
import hashlib, json, math, os, re, statistics, sys

sys.path.insert(0, os.path.dirname(__file__))
import axi.experiments.provider_side as P
import axi.experiments.provider_side_typed as T
import axi.experiments.provider_side_long as L

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..", "..")
PAGE = os.path.join(ROOT, "docs", "PROVIDER_SIDE_LONG_TEST.md")
RES = os.path.join(ROOT, "docs", "results", "axi_provider_long.json")
MODULE = os.path.join(HERE, "..", "axi", "experiments", "provider_side_long.py")
RESULT_MARK = "\n\n# Result"
PAGE_SHA256 = "f6cdaa3b6294008b2a27e84d6165dd9a9753558b835d2644135c948f98e8ae04"                   # the locked head
FILE_SHA256 = "a0ad579961d006e9d561ff65cb301de77dd25dd85ebb348827201ff0fdaeba2b"
MODULE_SHA256 = "ec38d450257bf3114d898baffede3f9bbab33013f55c1b903340113fedb4c967"
TYPED_MODULE_SHA256 = "c5f05d5840374fb77a1c67c6fd33f166b26342e93733bd9a31aa474eacbc1420"
PINS = dict(provider_side="bfb4734e171d9244cd47d10402be46528a557095a8275d02b38481a0e2b0b1d3", constrained="03452148e28c9f31372f12cc11ef1808ac67308adff972fa19d4aeeef7c96f7d",
            bounded="7427a686d151c4294809aebaa9664d0b5fa420854817eb79e910372c3f57b6bc", gate="b6a50408788791f76bc4d4129dad84debb6e09c9c5e804673cdf04abec224ce7",
            diagnostics="255f3c11db989b645077d10706a187e2c913576538a826e6dbdb8892b014b4f2", agent_loop="d658afaa29e3bce7e1fb7f43d88281fc40ee68ca59864421a65284e9973516b6",
            cascade="b0533316bc094e80a1d68eddb7e137c7c176af9a342341288360c18d61f74fe7", long_horizon="5718435de2178de110ddbb59dd3bf32e1bec9f1fd8625a27119d789399fa20ac")
REGEX_SHA256 = "6f4e5a993f26ccd471bff13422fa622b7820858a91e05d43c2bccfbe6b864245"
GRAMMAR_SHA = {"L0": REGEX_SHA256, "LN": "85d17bbffb556c21acf180b989fecf3b2c92ca419fce1516a675fb24dd9929af",
               "L8": "9603a4c10ec798e0d8098b00f02fd15831a79f4fcd58d8beafbce6463e1c3ddb", "LC": "9603a4c10ec798e0d8098b00f02fd15831a79f4fcd58d8beafbce6463e1c3ddb"}
SEED, STEPS = 20261017, 100
ORDER = ["L0", "LN", "L8", "LC"]
JOBS = {"L0": (192, 64), "LN": (192, 64), "L8": (192, 8), "LC": (192, 8)}                          # (first job, number of jobs)
PROBS = [98, 92, 78, 97, 96, 92, 92, 68, 50, 40, 80, 90, 93, 80, 92, 88, 90]                    # the page's probability of every line, in the order the page gives them
READ = [True] * 10 + [False] + [True] * 6                                                         # line 11 reads NO


def run():
    global _RUN
    try: return _RUN
    except NameError:
        with open(RES) as f: _RUN = json.load(f)
        return _RUN


def _text(): return open(PAGE).read()
def _head():
    t = _text(); assert t.count(RESULT_MARK) == 1
    return t[:t.index(RESULT_MARK)]
def _result(): t = _text(); return t[t.index(RESULT_MARK):]
def _h(text): return hashlib.sha256((text.rstrip("\n") + "\n").encode()).hexdigest()
def _sha(path): return hashlib.sha256(open(path, "rb").read()).hexdigest()


def pc(x, k=1): return f"{x * 100:.{k}f} %"
def num(x): return f"{x:,}"
def cells(res): return {c["label"]: c for c in res["cells"]}
def logs(c): return [x for s in c["sessions"] for x in s["log"]]
def secs(c): return sum(r["seconds"] for r in c["rounds"])
def job_of(s): return s["seed"] - SEED * 1000
def script(job): return T.script_of(SEED * 1000 + job, STEPS)
def items_of(text): return re.findall(r"(?:ADD|DEL)\[[^\]]*\]", text)


def st(c):
    X = logs(c); n = len(X); wall = secs(c); ex = sum(1 for x in X if x["exact"]); P_ = sum(x["n_prompt"] for x in X); C_ = sum(x["n_cached"] for x in X); W_ = sum(x["n_new"] for x in X); m_tok = P_ - C_ + W_
    return dict(n=n, wall=wall, sps=wall / n, ex=ex, share=ex / n, eph=3600 * ex / wall, jobs=len(c["sessions"]), whole=sum(1 for s in c["sessions"] if all(x["exact"] for x in s["log"])), lost=n - ex,
                prompt=P_, cached=C_, new=W_, comp_step=(P_ - C_) / n, m_tok=m_tok, q1000=1000 * ex / m_tok, unparsed=sum(1 for x in X if not x["parsed"]), full96=sum(1 for x in X if x["n_new"] >= 96),
                longest=max(x["n_new"] for x in X), corrupted=sum(1 for x in X if x["corrupted"]), out_of_language=sum(1 for x in X if not x["in_language"]), regexes=c["distinct_regexes"],
                new_grammars=sum(r.get("new_grammars", 0) for r in c["rounds"]))


def win(c, lo, hi):
    xs = [x for s in c["sessions"] for x in s["log"] if lo <= x["t"] <= hi]
    return (sum(1 for x in xs if x["exact"]), len(xs))


def wshare(c, lo, hi): a, b = win(c, lo, hi); return a / b


def quarter_prompt(c, lo, hi):
    xs = [x["n_prompt"] for s in c["sessions"] for x in s["log"] if lo <= x["t"] <= hi]
    return sum(xs) / len(xs)


def first_slips(c):
    """for every job with an inexact step: (job, first inexact step, kind there, inexact steps in all, lengths of its runs of inexact steps)"""
    out = []
    for s in c["sessions"]:
        ex = [x["exact"] for x in s["log"]]
        if all(ex): continue
        runs, cur = [], 0
        for e in ex:
            if not e: cur += 1
            elif cur: runs.append(cur); cur = 0
        if cur: runs.append(cur)
        t = ex.index(False) + 1; out.append((job_of(s), t, s["log"][t - 1]["kind"], ex.count(False), runs))
    return out


def outside_named(c):
    """the replies that are not in the named language of their own request: (job, step, exact, kind)"""
    out = []
    for s in c["sessions"]:
        for x, q in zip(s["log"], script(job_of(s))):
            if not T.in_language("named", q["instruction"], x["text"]): out.append((job_of(s), x["t"], x["exact"], x["kind"]))
    return out


def named_reply_at(res, job, t):
    s = next(s for s in cells(res)["LN"]["sessions"] if job_of(s) == job); return s["log"][t - 1]["text"]


def delete_facts(c):
    d = dict(dep=0, dep_node_only=0, dep_edge_item=0, leaf=0, leaf_node_only=0, add=0)
    for s in c["sessions"]:
        for x in s["log"]:
            its = items_of(x["text"]); only = len(its) == 1 and re.fullmatch(r"DEL\[\d+,\d+\]", its[0]) is not None
            if x["kind"] == "delete_dep": d["dep"] += 1; d["dep_node_only"] += only; d["dep_edge_item"] += any("->" in i for i in its)
            elif x["kind"] == "delete_leaf": d["leaf"] += 1; d["leaf_node_only"] += only
            else: d["add"] += 1
    return d


def stops(c): return [(job_of(s), len(s["log"]), s["stopped"], s["log"][-1]["n_prompt"]) for s in c["sessions"]]


# ---------------------------------------------------------------- the lines, by hand from the raw numbers
def lines_by_hand(res):
    c = cells(res); S = {k: st(v) for k, v in c.items()}; X = [x for v in c.values() for x in logs(v)]
    out = [all(isinstance(x["n_cached"], int) for x in X) and sum(x["n_cached"] for x in X) > 0]                                        # 1
    out.append(sum(1 for x in X if not x["parsed"]) <= 0.005 * len(X) and all(x["n_new"] < 96 for x in X) and all(x["in_language"] for x in X))   # 2
    same = True
    for a, b in zip(c["L0"]["sessions"], c["LN"]["sessions"]):
        q = script(job_of(a)); stop = False
        for x, y, r in zip(a["log"], b["log"], q):
            if stop: break
            if T.in_language("named", r["instruction"], x["text"]): same = same and x["text"] == y["text"]
            else: same = same and T.in_language("named", r["instruction"], y["text"]); stop = True
    out.append(same)                                                                                                                     # 3
    out.append(all(v["grammar_sha256"] == GRAMMAR_SHA[k] for k, v in c.items()) and all(re.fullmatch(T.regex_for(v["shape"], q["instruction"]), q["ideal"]) for v in c.values() for s in v["sessions"] for q in script(job_of(s))))  # 4
    out.append(all(len(s["log"]) == 100 and s["stopped"] is None for k in ("L0", "LN", "L8") for s in c[k]["sessions"]))                # 5
    out.append(quarter_prompt(c["L0"], 76, 100) <= 1.3 * quarter_prompt(c["L0"], 1, 25))                                                # 6
    out.append(all(S[k]["corrupted"] == 0 for k in S))                                                                                   # 7
    out.append(wshare(c["LN"], 1, 30) >= 0.995)                                                                                          # 8
    out.append(S["LN"]["share"] >= 0.99)                                                                                                 # 9
    out.append(wshare(c["LN"], 67, 100) >= 0.99)                                                                                         # 10
    out.append(S["LN"]["share"] >= S["L0"]["share"] + 0.03)                                                                              # 11
    out.append(wshare(c["L0"], 67, 100) < wshare(c["L0"], 1, 33))                                                                        # 12
    out.append(all(s["stopped"] == "context" and len(s["log"]) < 100 for s in c["LC"]["sessions"]))                                      # 13
    out.append(S["LC"]["share"] >= S["L8"]["share"] - 0.01)                                                                              # 14
    out.append(S["LC"]["eph"] < S["L8"]["eph"])                                                                                          # 15
    out.append(S["LN"]["sps"] <= S["L0"]["sps"] * 1.15)                                                                                  # 16
    out.append(S["LN"]["eph"] > S["L0"]["eph"])                                                                                          # 17
    return out


def measured(res, i):
    c = cells(res); S = {k: st(v) for k, v in c.items()}; X = [x for v in c.values() for x in logs(v)]
    a30, b30 = win(c["LN"], 1, 30); a67, b67 = win(c["LN"], 67, 100)
    return {
        1: f"cached tokens reported on all {num(len(X))} requests ({num(sum(x['n_cached'] for x in X))} of {num(sum(x['n_prompt'] for x in X))} prompt tokens served from the cache)",
        2: f"{num(sum(1 for x in X if x['parsed']))} of {num(len(X))} replies accepted by the strict parser; longest reply {max(x['n_new'] for x in X)} of 96 new tokens; {sum(1 for x in X if not x['in_language'])} outside the language of their own request",
        3: f"`LN` and `L0` write the same text at {num(sum(1 for a, b in zip(c['L0']['sessions'], c['LN']['sessions']) for x, y in zip(a['log'], b['log']) if x['text'] == y['text']))} of 6,400 steps; the other {len(outside_named(c['L0']))} are the `L0` replies outside the named language (below), and the `LN` reply there is inside it",
        4: f"the recorded sha256 of every cell equals the one recomputed from its jobs; `LN` has {num(S['LN']['regexes'])} expressions; the oracle's reply is inside the expression of all {num(len(X))} requests",
        5: f"all {S['L0']['jobs'] + S['LN']['jobs'] + S['L8']['jobs']} jobs of `L0`, `LN` and `L8` ran 100 steps",
        6: f"mean prompt {quarter_prompt(c['L0'], 1, 25):.0f} tokens in steps 1 to 25 and {quarter_prompt(c['L0'], 76, 100):.0f} in steps 76 to 100 (ratio {quarter_prompt(c['L0'], 76, 100) / quarter_prompt(c['L0'], 1, 25):.2f})",
        7: f"{sum(S[k]['corrupted'] for k in S)} corrupted steps in {num(len(X))}",
        8: f"`LN` steps 1 to 30: {num(a30)} of {num(b30)} exact ({pc(a30 / b30)})",
        9: f"`LN` all steps: {num(S['LN']['ex'])} of {num(S['LN']['n'])} exact ({pc(S['LN']['share'])}); {S['LN']['whole']} of 64 jobs exact to the end",
        10: f"`LN` steps 67 to 100: {num(a67)} of {num(b67)} exact ({pc(a67 / b67)})",
        11: f"`LN` {pc(S['LN']['share'])} against `L0` {pc(S['L0']['share'])}: {(S['LN']['share'] - S['L0']['share']) * 100:.1f} points, the line asks for 3",
        12: f"`L0` {pc(wshare(c['L0'], 67, 100))} over steps 67 to 100 against {pc(wshare(c['L0'], 1, 33))} over steps 1 to 33",
        13: f"all 8 jobs of `LC` stopped for the 7,600-token rule, at steps {', '.join(str(n) for _, n, _, _ in stops(c['LC']))}",
        14: f"`LC` {pc(S['LC']['share'])} on {S['LC']['n']} steps, `L8` {pc(S['L8']['share'])} on {S['L8']['n']}",
        15: f"`LC` {num(round(S['LC']['eph']))} exact steps per GPU hour, `L8` {num(round(S['L8']['eph']))}",
        16: f"`LN` {S['LN']['sps']:.4f} s per step, `L0` {S['L0']['sps']:.4f} ({(S['LN']['sps'] / S['L0']['sps'] - 1) * 100:+.1f} %)",
        17: f"`LN` {num(round(S['LN']['eph']))} exact steps per GPU hour, `L0` {num(round(S['L0']['eph']))}",
    }[i]


def table_lines(res): return [f"| {i} | {PROBS[i - 1]} % | {measured(res, i)} | {'yes' if READ[i - 1] else 'NO'} |" for i in range(1, 18)]


def table_cells(res):
    c = cells(res); out = ["| cell | jobs | steps | exact | steps 1 to 30 | steps 67 to 100 | jobs exact to the end | seconds | s per step | exact steps per GPU hour | per 1,000 tokens |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for lab in ORDER:
        s = st(c[lab]); w30, w100 = win(c[lab], 1, 30), win(c[lab], 67, 100); late = pc(w100[0] / w100[1]) if w100[1] else "-"
        out.append(f"| `{lab}` | {s['jobs']} | {num(s['n'])} | {pc(s['share'])} | {pc(w30[0] / w30[1])} | {late} | {s['whole']} | {s['wall']:.0f} | {s['sps']:.3f} | {num(round(s['eph']))} | {s['q1000']:.1f} |")
    return out


def table_slips(res):
    c = cells(res); f = {j: (t, k, n, r) for j, t, k, n, r in first_slips(c["L0"])}; o = {}
    for j, t, e, k in outside_named(c["L0"]): o.setdefault(j, []).append(t)
    out = ["| job | first slip at step | request | inexact steps in all | runs | `L0` replies outside the named language at steps | the `LN` reply at the first slip |", "|---|---|---|---|---|---|---|"]
    for j in sorted(f):
        t, k, n, r = f[j]; qd = script(j)[t - 1]; ln = named_reply_at(res, j, t)
        out.append(f"| {j} | {t} | {k} | {n} | {', '.join(str(x) for x in r)} | {', '.join(str(x) for x in o.get(j, []))} | {'the oracle\'s text' if ln == qd['ideal'] else 'a shorter delete, completed by the framework'} |")
    return out


def table_pred(res):
    c = cells(res); S = {k: st(v) for k, v in c.items()}; st_ = [n for _, n, _, _ in stops(c["LC"])]; fs = first_slips(c["L0"]); th = [sum(1 for f in fs if lo <= f[1] <= hi) for lo, hi in ((1, 33), (34, 66), (67, 100))]
    wall_min = sum(S[k]["wall"] for k in S) / 60
    rows = [("(a) `L0` holds the right state on between 82 and 95 percent of its 6,400 steps", 70, pc(S["L0"]["share"]), 82 <= S["L0"]["share"] * 100 <= 95),
            ("(b) the first slips of `L0` fall in steps 1 to 33 for more jobs than in steps 67 to 100", 70, f"{th[0]} / {th[1]} / {th[2]} in the three thirds", th[0] > th[2]),
            ("(c) the eight kept-chat jobs stop between steps 20 and 60", 80, f"stops at {min(st_)} to {max(st_)}; {sum(1 for n in st_ if 20 <= n <= 60)} of 8 inside", all(20 <= n <= 60 for n in st_)),
            ("(d) the computed prompt tokens per step of `LN` are within 5 percent of those of `L0`", 85, f"{S['LN']['comp_step']:.1f} and {S['L0']['comp_step']:.1f}", abs(S["LN"]["comp_step"] / S["L0"]["comp_step"] - 1) <= 0.05),
            ("(e) the four cells take 30 to 45 minutes in all, loading excluded", 70, f"{wall_min:.1f} minutes", 30 <= wall_min <= 45)]
    return ["| written first | probability | measured | came out |", "|---|---|---|---|"] + [f"| {a} | {p} % | {m} | {'yes' if ok else 'NO'} |" for a, p, m, ok in rows]


def render(res):
    c = cells(res); S = {k: st(v) for k, v in c.items()}; e = res["meta"]["engine"]; fs = first_slips(c["L0"]); out0 = outside_named(c["L0"]); dN = delete_facts(c["LN"]); X = [x for v in c.values() for x in logs(v)]
    wall = sum(S[k]["wall"] for k in S); runs = sorted(r for f in fs for r in f[4]); stp = stops(c["LC"]); w = lambda lab, lo, hi: win(c[lab], lo, hi)
    kinds = {k: sum(1 for j, t, ex, kk in out0 if kk == k) for k in ("add_edge", "add_node", "delete_dep", "delete_leaf")}
    lines = []; A = lines.append
    A(""); A(""); A("# Result"); A("")
    A(f"*Written after the run, below the locked text; nothing above this line has been changed (its hash is checked by `tests/test_provider_side_long.py` and again by `tests/test_provider_side_long_result.py`). The run was made by the owner on Colab on 21 Oct 2026 with `python/colab/axi_provider_side_long.ipynb`: {num(len(X))} steps in four cells, {wall / 60:.1f} minutes of generation, vLLM {e['version']}, `{e['model']}`, {e['gpu']}, a cache of {num(e['capacity_tokens'])} tokens. The merged file is `docs/results/axi_provider_long.json` (sha256 `{FILE_SHA256[:16]}...`). The architecture is Rajnish's; the cells, the code and every measurement are the owner's, with an AI assistant as a tool.*")
    A("")
    A(f"**What the page expected, and what came out.** Lines 2, 3, 4 and 5 all read yes (the page put them at about 65 % together), so the run is the run the page describes and the figures are claimed as read. **{sum(READ)} of the 17 lines read yes; line 11 reads NO, and no line is \"not run\".** The two headline lines read yes: the repeat on new jobs (line 8, {PROBS[7]} %) and the named grammar over 100 steps (line 9, {PROBS[8]} %), and so does line 10 ({PROBS[9]} %), which the page had put with line 14 as the pair it least expected together. Both headline lines yes stood at about 45 % on the page, all three lines at about 35 %.")
    A("")
    A("## The 17 lines"); A(""); A("| # | my probability | what was measured | read |"); A("|---|---|---|---|")
    lines.extend(table_lines(res)); A("")
    A(f"**Line 11 is the finding I did not expect, and I say so.** The page put it at {PROBS[10]} % because it counted a slip as staying until a later request removes the wrong item and took the fifth run's rate: about 15 of 64 capped jobs would slip and `L0` would hold the right state on about 87 % of the steps. The first half came out as written: {len(fs)} of 64 jobs of `L0` slipped. The second half did not: `L0` held the right state on {pc(S['L0']['share'])} of its steps, {num(S['L0']['lost'])} inexact steps of {num(S['L0']['n'])}, because most slips ended soon. The {len(runs)} runs of inexact steps lasted {runs[0]} to {runs[-1]} steps (median {statistics.median(runs):g}); only job 195 kept its slip to the end of the job ({next(f for f in fs if f[0] == 195)[3]} steps). The page said that with `L0` near 98 percent, 100 steps might not yet be long enough to see the problem in this setup, and that a NO on line 12 would say so. Line 12 reads yes ({pc(w('L0', 67, 100)[0] / w('L0', 67, 100)[1])} over steps 67 to 100 against {pc(w('L0', 1, 33)[0] / w('L0', 1, 33)[1])} over steps 1 to 33), so the control does lose ground as the job goes on, by 3 points and not by twelve; with 14 jobs that slipped, that is a measured size of the loss in this setup and not a rate for other tasks.")
    A("")
    A("## The cells"); A(""); lines.extend(table_cells(res)); A("")
    A(f"`L0` and `LN` are the same {num(S['L0']['n'])} requests on the same {S['L0']['jobs']} jobs in one session, which is what lines 16 and 17 compare; `L8` and `LC` are the same {S['L8']['jobs']} jobs. The computed prompt tokens per step are {S['L0']['comp_step']:.0f} (`L0`), {S['LN']['comp_step']:.0f} (`LN`), {S['L8']['comp_step']:.0f} (`L8`) and {S['LC']['comp_step']:.0f} (`LC`).")
    A("")
    A("## What the page's rule lets this Result claim")
    A("The page's rule, applied line by line to what was read:")
    A("* **Lines 2, 3, 4 and 5 read yes**, so the figures of the cells are claimed as read.")
    A(f"* **Line 8 reads yes.** On steps 1 to 30 of the 64 new jobs `LN` is exact at all {num(w('LN', 1, 30)[1])} steps. The page's condition for saying so is in the same sentence: the host wrote the coordinates of the request into the reply, and what the model wrote is the number (below, \"What the model writes in `LN`\").")
    A(f"* **Lines 9 and 10 read yes, line 11 reads NO and line 12 reads yes.** Over 100 steps `LN` is exact at all {num(S['LN']['n'])} steps and `L0` at {pc(S['L0']['share'])}; the gap of {(S['LN']['share'] - S['L0']['share']) * 100:.1f} points is below the three that the line asked for.")
    A(f"* **Lines 13, 14 and 15 read yes.** The kept chat stopped before step 100 in every job and held the state as exactly as the exact-state arm on the steps it ran ({S['LC']['n']} steps, {pc(S['LC']['share'])}), and did fewer exact steps per GPU hour.")
    A("* **Lines 16 and 17 are from cells of one session**, so a statement about time is allowed, between `L0` and `LN` and between `L8` and `LC` and not against any other run.")
    A("* **Never**, as the page says: a multiple of capacity, a count of GPUs saved, dollars or a percentage of cost, \"100 percent\" for a grammar, \"proven\", a statement about any provider, about requests in free words, about a chat with a summary or a window, or about a model that was not run.")
    A("")
    A(f"**The sentence the page allows, with this run's numbers.** On jobs 192 to 255, which no earlier run had seen, with the requests read by the host from their sentence, on one Tesla T4 (Colab), with vLLM {e['version']}, `{e['model']}`, a cache of {num(e['capacity_tokens'])} tokens and jobs of 100 steps: the capped expression left the exact share at {pc(S['L0']['share'])} ({num(S['L0']['ex'])} of {num(S['L0']['n'])} steps; {pc(w('L0', 67, 100)[0] / w('L0', 67, 100)[1])} over steps 67 to 100) at {S['L0']['sps']:.3f} seconds per step, {num(round(S['L0']['eph']))} exact steps per GPU hour and {S['L0']['q1000']:.1f} per 1,000 tokens. The grammar that names the coordinates of the request had all {num(S['LN']['n'])} steps exact, {S['LN']['whole']} of 64 jobs exact to the end, at {S['LN']['sps']:.3f} seconds per step, {num(round(S['LN']['eph']))} exact steps per GPU hour and {S['LN']['q1000']:.1f} per 1,000 tokens, **and in the same sentence: in that grammar the host writes the coordinates of the request into the reply, the model writes the value of an added service or the weight of an added edge, and in every one of the {num(dN['dep'] + dN['leaf'])} delete replies it wrote the node alone, the edges being completed by the framework's repair.** The kept chat stopped at its context after {', '.join(str(n) for _, n, _, _ in stp)} steps (jobs 192 to 199) and held the state on all {S['LC']['n']} steps it ran at {num(round(S['LC']['eph']))} exact steps per GPU hour where the exact-state arm had {num(round(S['L8']['eph']))} on the same 8 jobs. One run, 64 jobs.")
    A("")
    A("## The jobs that slipped")
    A(f"{len(fs)} of 64 jobs of `L0` had an inexact step ({', '.join(str(f[0]) for f in fs)}); `LN` had none. Every one of the {len(fs)} first slips is a reply outside the named language of its own request; the {len(out0)} replies of `L0` outside it are in these {len({j for j, _, _, _ in out0})} jobs ({kinds['add_edge']} of an `add_edge` request, {kinds['delete_dep']} of a `delete_dep`, {kinds['delete_leaf']} of a `delete_leaf`, {kinds['add_node']} of an `add_node`; the whole run has {num(sum(1 for x in logs(c['L0']) if x['kind'] == 'add_edge'))} `add_edge` requests). One of them (job 225, step 12) left the state exact. The first slip, what it was, how long the slip lasted and what `LN` wrote at the same step:")
    A(""); lines.extend(table_slips(res)); A("")
    first_t = [f[1] for f in fs]
    A(f"By the thirds of the job the first slips fall {sum(1 for t in first_t if t <= 33)} / {sum(1 for t in first_t if 34 <= t <= 66)} / {sum(1 for t in first_t if t >= 67)}: more in the last third than in the first. The page expected the opposite (side prediction (b) below), and with 14 jobs this is a count, not a trend.")
    A("")
    A("## What the model writes in `LN`")
    A(f"Same as in the fifth run, the host writes the coordinates of the request into the reply and the model writes the numbers. In `LN` the {num(dN['add'])} add replies carry a value or a weight; of the {num(dN['dep'])} `delete_dep` replies {num(dN['dep_node_only'])} are the node alone ({dN['dep_edge_item']} carry an edge item), and of the {dN['leaf']} `delete_leaf` replies {dN['leaf_node_only']} are the node alone; the edges of a deleted service are completed by the framework's repair, which this test does not change. The 100-step result for `LN` is therefore a result for the part of the reply that the model still writes, and it says nothing about a model that has to read the coordinates from a request in free words.")
    A("")
    A("## The kept chat")
    A(f"| job | " + " | ".join(str(j) for j, _, _, _ in stp) + " |"); A("|---|" + "---|" * len(stp))
    A("| steps run | " + " | ".join(str(n) for _, n, _, _ in stp) + " |"); A("| prompt of the last step | " + " | ".join(num(p) for _, _, _, p in stp) + " |"); A("")
    A(f"All 8 stopped for the context (the next prompt would have passed 7,600 tokens); the prompt of the kept chat went from {num(round(quarter_prompt(c['LC'], 1, 25)))} tokens on average in steps 1 to 25 to {max(p for _, _, _, p in stp):,} at the longest, where the exact-state prompt stays near {quarter_prompt(c['L8'], 76, 100):.0f}. The kept chat computes more prompt tokens per step ({S['LC']['comp_step']:.0f} against {S['L8']['comp_step']:.0f}) and takes longer per step ({S['LC']['sps']:.3f} s against {S['L8']['sps']:.3f}), so it does fewer exact steps per hour; it is exact on all the steps it runs, and it cannot run the job to the end in this setup.")
    A("")
    A("## The predictions written beside the lines, read afterwards")
    A(""); lines.extend(table_pred(res)); A("")
    A("(f) *whether the model slips more often at step 80 than at step 8 when the grammar names the coordinates* cannot be read: `LN` has no slip, so the run does not say.")
    A("")
    A("## What this does not show")
    A("Requests in free words (the host reads the sentence the job was built from, and the reply is built from what it read); tasks of more than 100 steps, let alone days; real company tasks; another model, size, GPU, engine or cache size; a kept chat with a summary or a sliding window (only the plain kept chat is run, and it stops at the context of this setup); a kept chat at 64 jobs; a retry on top of a grammar; dollars; a capacity in jobs per GPU; energy; hallucination in free text. Nothing here is proven.")
    A("")
    A("## What comes next")
    A("Two things the page left out on purpose, each as its own update with its own page locked first: a reader for requests in free words (the host's copying of coordinates is the open question this result leaves), and the two controls that were missing (the target line at the top only, a plain restatement of the request).")
    A("")
    return "\n".join(lines)


# ---------------------------------------------------------------- 1 the file and every step
def test_the_file_is_the_one_the_page_is_about_and_every_step_replays_from_its_reply_text_through_the_frameworks_own_rules():
    res = run(); assert _sha(RES) == FILE_SHA256, "the saved result file must not change"; assert _sha(MODULE) == MODULE_SHA256 == res["module_sha256"], "the module that merged the cells must be the one in the repository"
    assert set(res) == {"meta", "cells", "module_sha256", "typed_module_sha256", "pinned_sha256"} and res["pinned_sha256"] == PINS == T.pinned_hashes() and res["typed_module_sha256"] == TYPED_MODULE_SHA256 == T.fingerprint()
    m = res["meta"]; e = m["engine"]; con = e["constraint"]
    assert {k: v for k, v in e.items() if k != "constraint"} == dict(kind="vllm", name="Qwen/Qwen2.5-7B-Instruct-AWQ", version="0.31.0", gpu="Tesla T4", capacity_tokens=80000, capacity_source="override", block_size=16, kv_bytes_per_token=57344, model="Qwen/Qwen2.5-7B-Instruct-AWQ", python="3.13.15")
    assert con["kind"] == "regex_per_request" and con["regex_sha256"] == REGEX_SHA256 == hashlib.sha256(con["regex"].encode()).hexdigest() == T.REGEX_SHA256 and con["name_cap"] == 16
    assert (m["steps"], m["seed"], m["policy"], m["hint_rules"], m["name_cap"], m["model"]) == (100, SEED, "repair_m6", False, 16, "Qwen/Qwen2.5-7B-Instruct-AWQ") and m["cells"] == {k: [JOBS[k][1], JOBS[k][0], L.CELLS[k][2], L.CELLS[k][3]] for k in ORDER}
    assert [c["label"] for c in res["cells"]] == ORDER and all(L.CELLS[k][:2] == (JOBS[k][1], JOBS[k][0]) for k in ORDER)
    for c in res["cells"]:
        lab = c["label"]; f, k = JOBS[lab]
        assert (c["arm"], c["K"], c["first_episode"], c["shape"], c["steps"], c["seed"], c["policy"], c["hint_rules"], c["engine"]) == (L.CELLS[lab][2], k, f, L.CELLS[lab][3], 100, SEED, "repair_m6", False, e)
        assert c["grammar_sha256"] == GRAMMAR_SHA[lab], lab
        assert [s["seed"] for s in c["sessions"]] == [SEED * 1000 + f + i for i in range(k)] and all(s["script_steps"] == 100 for s in c["sessions"])
        if c["arm"] == "S": assert all(s["stopped"] is None and len(s["log"]) == 100 for s in c["sessions"])
        else: assert all(s["stopped"] == "context" and 20 < len(s["log"]) < 100 for s in c["sessions"])
        assert all([x["t"] for x in s["log"]] == list(range(1, len(s["log"]) + 1)) for s in c["sessions"])
        X = logs(c); assert abs(secs(c) - c["wall_seconds"]) < 1e-6 and [r["round"] for r in c["rounds"]] == list(range(1, len(c["rounds"]) + 1))
        assert (sum(r["prompt_tokens"] for r in c["rounds"]), sum(r["cached_tokens"] for r in c["rounds"]), sum(r["new_tokens"] for r in c["rounds"])) == tuple(sum(x[key] for x in X) for key in ("n_prompt", "n_cached", "n_new"))
        assert sum(r["active"] for r in c["rounds"]) == len(X) and all(0 <= x["n_cached"] <= x["n_prompt"] and 0 < x["n_new"] < 96 for x in X) and all(bool(x["exact"]) == (x["distance"] == 0) for x in X) and all(x["parsed"] or not x["admitted"] for x in X)
        for s in c["sessions"]:
            for x, q in zip(s["log"], script(job_of(s))):
                assert re.fullmatch(T.regex_for(c["shape"], q["instruction"]), x["text"]) and x["in_language"] is True and x["kind"] == q["kind"], (lab, job_of(s), x["t"])
                assert re.fullmatch(T.regex_for(c["shape"], q["instruction"]), q["ideal"])
        for s in c["sessions"]:                                                                          # the job played again through the typed session (the pinned session with a prompt and an expression)
            sess = T.TypedSession(s["seed"], 100, c["prompt"], c["shape"])
            for x in s["log"]:
                sess.request(); sess.accept(P.Reply(x["text"], 0, 0, 0)); y = sess.log[-1]
                assert (y["parsed"], y["admitted"], y["corrupted"], y["exact"], y["distance"], y["in_language"]) == (x["parsed"], x["admitted"], x["corrupted"], x["exact"], x["distance"], x["in_language"]), (lab, job_of(s), x["t"])
    assert sum(len(logs(c)) for c in res["cells"]) == 14004 and round(sum(c["wall_seconds"] for c in res["cells"]), 1) == 1633.7
    assert {lab: c["distinct_regexes"] for lab, c in cells(res).items()} == dict(L0=1, LN=835, L8=290, LC=191)


# ---------------------------------------------------------------- 2 the lines
def test_the_seventeen_lines_read_as_the_result_says_by_the_code_and_by_hand_and_the_rule_about_claims_is_followed():
    res = run(); chk = L.verdict_long(res["cells"]); by_code = [c for _, _, _, c in chk]; assert by_code == READ == lines_by_hand(res), "code, hand and the Result must agree"
    assert sum(READ) == 16 and [i + 1 for i, r in enumerate(READ) if not r] == [11] and all(c is not None for c in by_code) and [i for i, _, _, _ in chk] == list(range(1, 18))
    a = L.claims_allowed(chk); assert a["base"] is True and a["repeat_of_the_fifth_run"] == {8: True} and a["over_100_steps"] == {9: True, 10: True, 11: False, 12: True} and a["kept_chat"] == {13: True, 14: True, 15: True} and a["time_and_cost_in_one_session"] == {16: True, 17: True}
    S = {k: st(v) for k, v in cells(run()).items()}
    assert (S["LN"]["ex"], S["LN"]["n"], S["L0"]["ex"], S["L0"]["n"]) == (6400, 6400, 6244, 6400) and S["LN"]["share"] - S["L0"]["share"] < 0.03 and abs((S["LN"]["share"] - S["L0"]["share"]) - 0.024375) < 1e-9
    assert win(cells(run())["LN"], 1, 30) == (1920, 1920) and win(cells(run())["LN"], 67, 100) == (2176, 2176) and win(cells(run())["L0"], 67, 100) == (2078, 2176) and win(cells(run())["L0"], 1, 33) == (2081, 2112)
    assert len(PROBS) == 17 and PROBS[7] == 68 and PROBS[8] == 50 and PROBS[9] == 40 and PROBS[10] == 80


# ---------------------------------------------------------------- 3 the page
def test_the_part_above_the_result_is_what_was_written_before_the_run():
    head = _head(); assert _h(head) == PAGE_SHA256, "the locked head of the page must not change"
    assert "# Result" not in head and head.rstrip("\n").endswith("---") and head.startswith("# Test 19: the serving side a sixth time")
    for i, p in enumerate(PROBS, 1): assert f"({p} %)" in head.split("## Reading rules")[0], i


def test_the_page_carries_exactly_the_result_this_file_renders_and_its_tables_say_what_the_rules_say():
    res = run(); text = _text(); assert text == _head() + render(res), "the Result on the page must be render(result)"
    R = _result(); assert R.count("# Result") == 1 and "\n| 11 | 80 % |" in R and R.count("| NO |") == 5 == 1 + 4                                       # line 11, and predictions (a), (b), (c) and (e)
    rows = [r for r in R.split("\n") if re.match(r"\| \d+ \| \d+ % \|", r)]; assert len(rows) == 17 and [r.endswith("| yes |") for r in rows] == READ


def test_the_figures_quoted_in_the_result_are_recounted_from_the_records():
    R = _result(); res = run(); c = cells(res); S = {k: st(v) for k, v in c.items()}
    for s in ("16 of the 17 lines read yes", "line 11 reads NO", "6,400 of 6,400 exact (100.0 %); 64 of 64 jobs exact to the end", "1,920 of 1,920 exact (100.0 %)", "2,176 of 2,176 exact (100.0 %)",
              "`LN` 100.0 % against `L0` 97.6 %: 2.4 points, the line asks for 3", "`L0` 95.5 % over steps 67 to 100 against 98.5 % over steps 1 to 33", "14 of 64 jobs of `L0` had an inexact step",
              "the 16 replies of `L0` outside it are in these 14 jobs", "(12 of an `add_edge` request, 2 of a `delete_dep`, 1 of a `delete_leaf`, 1 of an `add_node`", "14,004", "28, 64, 53, 42, 59, 45, 54, 59", "0 corrupted steps in 14,004",
              "21 Oct 2026", "`docs/results/axi_provider_long.json`", "6,244 of 6,400 steps", "2,183 delete replies"):
        assert s in R, s
    fs = first_slips(c["L0"]); runs = sorted(r for f in fs for r in f[4])
    assert len(fs) == 14 and sum(f[3] for f in fs) == 156 == S["L0"]["lost"] and len(runs) == 15 and runs[0] == 1 and runs[-1] == 60 and statistics.median(runs) == 6 and next(f for f in fs if f[0] == 195)[1:4:2] == (41, 60)
    assert [f[1] for f in fs] == [17, 41, 92, 74, 66, 21, 74, 100, 92, 93, 31, 83, 17, 89] and [sum(1 for f in fs if lo <= f[1] <= hi) for lo, hi in ((1, 33), (34, 66), (67, 100))] == [4, 2, 8]
    assert (S["L0"]["sps"], S["LN"]["sps"]) == (secs(c["L0"]) / 6400, secs(c["LN"]) / 6400) and round(S["L0"]["eph"]) == 35113 and round(S["LN"]["eph"]) == 35764 and round(S["L8"]["eph"]) == 24001 and round(S["LC"]["eph"]) == 6342
    assert (S["LC"]["n"], S["LC"]["ex"], S["L8"]["n"], S["L8"]["ex"]) == (404, 404, 800, 800) and [n for _, n, _, _ in stops(c["LC"])] == [28, 64, 53, 42, 59, 45, 54, 59] and all(p <= 7600 for _, _, _, p in stops(c["LC"]))
    assert "never" in R.lower() and "100 percent" in R and R.count("\"100 percent\" for a grammar") == 1 and "proven" in R and "dollars" in R and "energy" in R
    for bad in ("GPUs saved", "x capacity", "times the capacity"):
        assert bad not in R.replace("a count of GPUs saved", "")


def test_the_replies_outside_the_named_language_the_deletes_and_the_predictions_are_read_from_the_records_and_each_is_of_the_kind_the_result_says():
    res = run(); c = cells(res); o = outside_named(c["L0"]); fs = first_slips(c["L0"])
    assert len(o) == 16 and len({j for j, _, _, _ in o}) == 14 == len(fs) and {j for j, _, _, _ in o} == {f[0] for f in fs} and sum(1 for _, _, ex, _ in o if ex) == 1 and [x for x in o if x[2]] == [(225, 12, True, "add_edge")]
    assert all(any(j == f[0] and t == f[1] for j, t, _, _ in o) for f in fs), "every first slip is a reply outside the named language"
    assert outside_named(c["LN"]) == [] and sum(1 for a, b in zip(c["L0"]["sessions"], c["LN"]["sessions"]) for x, y in zip(a["log"], b["log"]) if x["text"] == y["text"]) == 6384
    for j, t, k, n, r in fs:
        q = script(j)[t - 1]; assert named_reply_at(res, j, t) == q["ideal"] or k in ("delete_dep", "delete_leaf")
    d = delete_facts(c["LN"]); assert (d["dep"], d["dep_node_only"], d["dep_edge_item"], d["leaf"], d["leaf_node_only"], d["add"]) == (1488, 1488, 0, 695, 695, 4217)
    S = {k: st(v) for k, v in c.items()}; P_ = table_pred(res)
    assert [r.split("|")[-2].strip() for r in P_[2:]] == ["NO", "NO", "NO", "yes", "NO"], P_
    assert abs(S["LN"]["comp_step"] / S["L0"]["comp_step"] - 1) < 0.005 and abs(sum(S[k]["wall"] for k in S) / 60 - 27.2) < 0.05


# ---------------------------------------------------------------- 4 the front pages
RUN_ITEM = "**Run (21 Oct 2026):**"
RUN_ROW = "**Run:**"


def test_the_two_front_pages_say_what_was_run_what_was_read_and_what_it_does_not_show():
    cm = open(os.path.join(ROOT, "docs", "RAJNISH_CONCEPT_MAP.md")).read(); ve = open(os.path.join(ROOT, "docs", "VERIFIED.md")).read(); n_tests = sum(1 for n, f in globals().items() if n.startswith("test_") and callable(f))
    item = cm[cm.index("112. **"): cm.index("113. **")]; k = ve.index("Provider-side serving test, sixth run"); row = ve[k: ve.index("\n", k)]
    for text, mark in ((item, RUN_ITEM), (row, RUN_ROW)):
        assert text.count(mark) == 1; run_ = text[text.index(mark):]
        for s in ("16 of 17 lines read yes", "line 11 reads NO", f"(test 134, {n_tests} tests)", "mutation check of test 134: ", "`LN` was exact at all 6,400 steps", "97.6 %", "6,244 of 6,400", "ost wrote the coordinates of the request into the reply", "kept chat stopped before step 100 in all 8 jobs",
                  "free words"):
            if "mutation" in s: assert s in text, s
            else: assert s in run_, s
        assert "@@" not in text and "\"100 percent\"" not in run_ and "dollars" in text and ("nothing here is proven" in text or "proven" in text)
    assert "Next free: row 179, item 134, test 160, question 122" in cm and cm.count("Next free:") == 1
    assert "test 133 checks the 100-step serving test, item 112, and test 134 its Result" in cm
    assert "(sha256 " + PAGE_SHA256[:16] + "... of the text above the line at its end)" in item and "(sha256 " + PAGE_SHA256[:16] + "... of the locked text)" in row


if __name__ == "__main__":
    ok = 0; fails = 0
    for name, f in list(globals().items()):
        if name.startswith("test_") and callable(f):
            try: f(); ok += 1; print("ok  ", name)
            except Exception as ex: fails += 1; import traceback; print("FAIL", name, repr(ex)[:300]); traceback.print_exc()
    print(f"{ok}/{ok + fails} passed"); sys.exit(1 if fails else 0)
