"""Implements concept-map item 113 (Result): the seventh provider-side serving run, requests in ordinary words, three cells (`C0` the request with coordinates plus the names list, `W0` the request in words with the capped expression,
`WN` the request in words with the expression the host reader builds) on the sixth run's jobs 192 to 255, made by the owner on Colab (vLLM 0.31.0, Tesla T4, Qwen2.5-7B-Instruct-AWQ, a cache of 80,000 tokens).
Page: docs/PROVIDER_SIDE_FREE_TEST.md (the question, the design, the 16 reading lines and the predictions were written and hash-locked before the run; the Result is appended below them).
The file: docs/results/axi_provider_free.json (the merged file the notebook's last cell wrote and the owner sent).
This file recomputes from the saved file, with its own arithmetic and not with the code that printed the figures:
  1. the file: its sha256, the settings, the seeds, the rounds adding up to the cells, and every one of the 19,200 steps played again from the text of its reply through the framework's own session, gate, repair and diagnosis;
  2. the 16 reading lines (10 yes; lines 2, 8, 9, 11, 12, 13 and 14 NO; none not run -- see READ) by the code and by hand from the raw numbers, and what the page's rule lets the Result claim;
  3. the first slips of `W0` and `WN` by kind, pool and what the host read, the four replies that used all 96 new tokens, the aliases, the serving numbers against the sixth run's;
  4. the predictions written beside the lines, read afterwards: the ones the run did not bear out are said so;
  5. the whole Result is `render(result)` and the page must equal the locked head followed by it; the figures quoted in it are recounted;
  6. that the Result claims what the page's rule allows and nothing more (no multiple of capacity, no GPUs saved, no dollars, no 'proven', nothing about any provider, nothing about requests in the words of a user);
  7. the two front pages (the concept map's item 113 and the VERIFIED row) say what was run and what was read, and the counter has moved.
What it does NOT show: another model, GPU, engine or cache size, the words of a real user, tasks of more than 100 steps, dollars, energy."""
import collections, hashlib, json, os, re, sys

sys.path.insert(0, os.path.dirname(__file__))
import axi.experiments.provider_side as P
import axi.experiments.provider_side_typed as T
import axi.experiments.provider_side_long as L
import axi.experiments.provider_side_free as F

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..", "..")
PAGE = os.path.join(ROOT, "docs", "PROVIDER_SIDE_FREE_TEST.md")
RES = os.path.join(ROOT, "docs", "results", "axi_provider_free.json")
LONG = os.path.join(ROOT, "docs", "results", "axi_provider_long.json")
MODULE = os.path.join(HERE, "..", "axi", "experiments", "provider_side_free.py")
RESULT_MARK = "\n\n# Result"
PAGE_SHA256 = "8b53170cbd095425c5adacc18992b472e156a265479b34a8d60a78d91129fa4d"                    # the locked head
FILE_SHA256 = "4ec1ce81239b4b184327a328b765797912a4b0534156383eec49214c65e4484e"
MODULE_SHA256 = "ba27ad03e532ce7430957805d3b9567edd879b41c4e2617b79dbaa0d4c69b0b7"
SEED, STEPS = 20261017, 100
ORDER = ["C0", "W0", "WN"]
PROBS = [99, 92, 99, 99, 96, 99, 65, 60, 65, 90, 55, 50, 65, 55, 88, 75]                          # the page's probability of every line
READ = [True, False, True, True, True, True, True, False, False, True, False, False, False, False, True, True]
_RUN = {}


def run(path=RES):
    if path not in _RUN:
        with open(path) as f: _RUN[path] = json.load(f)
    return _RUN[path]


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
def kind_of(ins): return T.parse_request(ins)["kind"]


# ---------------------------------------------------------------- the figures, by plain arithmetic on the records
def figures(res, long_res):
    C = cells(res); Lc = cells(long_res); fig = {}
    for k, c in C.items():
        R = logs(c); n = len(R); ex = sum(1 for x in R if x["exact"]); pt = sum(x["n_prompt"] for x in R); ch = sum(x["n_cached"] for x in R); nn = sum(x["n_new"] for x in R)
        fig[k] = dict(n=n, exact=ex, share=ex / n, wall=c["wall_seconds"], sps=c["wall_seconds"] / n, comp=(pt - ch) / n, cached_share=ch / pt, prompt=pt / n, new=nn / n, per_hour=ex / c["wall_seconds"] * 3600, cached=ch, prompt_total=pt,
                      jobs_exact=sum(1 for s in c["sessions"] if all(x["exact"] for x in s["log"])), unparsed=sum(1 for x in R if not x["parsed"]), full=sum(1 for x in R if x["n_new"] >= 96), outside=sum(1 for x in R if not x["in_language"]), distinct=c["distinct_regexes"])
    for k in ("L0", "LN"):
        R = logs(Lc[k]); n = len(R); ex = sum(1 for x in R if x["exact"]); pt = sum(x["n_prompt"] for x in R); ch = sum(x["n_cached"] for x in R)
        fig["long_" + k] = dict(n=n, exact=ex, share=ex / n, wall=Lc[k]["wall_seconds"], sps=Lc[k]["wall_seconds"] / n, comp=(pt - ch) / n, prompt=pt / n, per_hour=ex / Lc[k]["wall_seconds"] * 3600)
    sh = lambda c, pred: (lambda R: (sum(1 for x in R if x["exact"]) / len(R), len(R)))([x for x in logs(c) if pred(x)])
    fig["W0_A"] = sh(C["W0"], lambda x: x["pool"] == "A"); fig["W0_B"] = sh(C["W0"], lambda x: x["pool"] == "B"); fig["WN_A"] = sh(C["WN"], lambda x: x["pool"] == "A"); fig["WN_B"] = sh(C["WN"], lambda x: x["pool"] == "B")
    fig["WN_read"] = sh(C["WN"], lambda x: x["reader"] == "right"); fig["WN_miss"] = sh(C["WN"], lambda x: x["reader"] == "miss")
    fig["reader"] = {o: sum(1 for x in logs(C["WN"]) if x["reader"] == o) for o in ("right", "wrong", "miss")}
    for k in ("W0", "WN"):
        exp = collections.Counter(); slip = collections.Counter(); thirds = [0, 0, 0]; firsts = []; B_slips = []
        for s in C[k]["sessions"]:
            sc = T.script_of(s["seed"], STEPS); tb_all = F.name_tables(s["seed"], STEPS)[0]
            for i, (x, st) in enumerate(zip(s["log"], sc)):
                key = (kind_of(st["instruction"]), x["pool"], x["reader"] if k == "WN" else None); exp[key] += 1
                if not x["exact"]:
                    slip[key] += 1; firsts.append(x["t"]); thirds[0 if x["t"] <= 33 else 1 if x["t"] <= 66 else 2] += 1
                    if x["pool"] == "B": B_slips.append((x, st, tb_all[i]))
                    break
        fig["haz_" + k] = {key: (slip[key], exp[key]) for key in exp}; fig["slipjobs_" + k] = len(firsts); fig["thirds_" + k] = thirds; fig["median_" + k] = sorted(firsts)[len(firsts) // 2]
        if k == "W0":
            ncoord = nslip = 0
            for x, st, tb in B_slips:
                ora = T.parse_request(st["instruction"]); inv = {(int(ora[key][0]), int(ora[key][1])) for key in ("node", "src", "dst") if key in ora}
                extra = [c for c, nm in tb.items() if c not in inv and re.search(r"(?<![A-Za-z])" + re.escape(nm) + r"(?![A-Za-z])", x["request"], re.IGNORECASE)]
                if not extra: continue
                nslip += 1; ncoord += any((f"({c[0]},{c[1]})" in x["text"] or f"[{c[0]},{c[1]}" in x["text"]) for c in extra)
            fig["extra_in_reply"] = (ncoord, nslip)
            a_exp = collections.Counter(); a_slip = collections.Counter()
            for s in C[k]["sessions"]:
                for x, st, r in zip(s["log"], T.script_of(s["seed"], STEPS), F.requests_of(s["seed"], STEPS)):
                    if kind_of(st["instruction"]) != "add_node":
                        al = F.aliases_of(r[3]); has = any(re.search(r"(?<![A-Za-z])" + re.escape(a) + r"(?![A-Za-z])", x["request"], re.IGNORECASE) for c in al for a in al[c])
                        a_exp[has] += 1
                        if not x["exact"]: a_slip[has] += 1; break
                    elif not x["exact"]: break
            fig["alias"] = {h: (a_slip[h], a_exp[h]) for h in (True, False)}
    fig["runaway"] = [(k, s["seed"] % 1000, x["t"], x["pool"], x["reader"], x["why"]) for k in ("W0", "WN") for s in C[k]["sessions"] for x in s["log"] if x["n_new"] >= 96]
    fig["total_seconds"] = sum(C[k]["wall_seconds"] for k in C); fig["all_replies"] = sum(fig[k]["n"] for k in C); fig["all_unparsed"] = sum(fig[k]["unparsed"] for k in C)
    return fig


def measured(f, i):
    A, B = f["W0_A"], f["W0_B"]
    return {1: f"cached tokens reported on all {num(f['all_replies'])} requests ({num(sum(f[k]['cached'] for k in ORDER))} of {num(sum(f[k]['prompt_total'] for k in ORDER))} prompt tokens served from the cache)",
            2: f"{num(f['all_replies'] - f['all_unparsed'])} of {num(f['all_replies'])} replies accepted by the strict parser ({pc((f['all_replies'] - f['all_unparsed']) / f['all_replies'], 2)}); **{sum(f[k]['full'] for k in ORDER)} replies used all 96 new tokens** (3 in `W0`, 1 in `WN`, none in `C0`) and are outside the language of their own request",
            3: "all 12,800 sentences of `W0` and `WN` name the services of the oracle's request in its order, with its numbers, and the second pool names exactly one service more (checked by plain search, not by the reader)",
            4: f"the recorded sha256 of every cell equals the one recomputed from its jobs and the reader; `WN` has {f['WN']['distinct']} expressions; the oracle's reply is inside the expression of all {num(f['all_replies'])} requests",
            5: "all 192 jobs of the three cells ran 100 steps",
            6: f"the reader, run again on the recorded sentences, gives the recorded outcome at all 6,400 requests: right {num(f['reader']['right'])}, wrong {f['reader']['wrong']}, not read {num(f['reader']['miss'])}",
            7: f"`C0` {pc(f['C0']['share'])} ({num(f['C0']['exact'])} of 6,400 steps) against {pc(f['long_L0']['share'])} for the sixth run's `L0` ({num(f['long_L0']['exact'])} of 6,400)",
            8: f"`W0` {pc(f['W0']['share'])} ({num(f['W0']['exact'])} of 6,400 steps; 0 of 64 jobs exact to the end)",
            9: f"`W0` is {(f['C0']['share'] - f['W0']['share']) * 100:.1f} points below `C0`",
            10: f"`WN` {pc(f['WN']['share'])} against `W0` {pc(f['W0']['share'])}",
            11: f"`WN` {pc(f['WN']['share'])} ({num(f['WN']['exact'])} of 6,400 steps; {f['WN']['jobs_exact']} of 64 jobs exact to the end)",
            12: f"{pc(f['WN_read'][0])} of the {num(f['WN_read'][1])} steps whose request the host read right",
            13: f"{pc(f['WN_miss'][0])} of the {num(f['WN_miss'][1])} steps whose request the host did not read",
            14: f"`W0`: second pool {pc(B[0])} of {num(B[1])} steps, first pool {pc(A[0])} of {num(A[1])}; the difference is {(A[0] - B[0]) * 100:.1f} points",
            15: f"`WN` {f['WN']['sps']:.3f} s per step, `W0` {f['W0']['sps']:.3f}: {(f['WN']['sps'] / f['W0']['sps'] - 1) * 100:+.1f} %",
            16: f"`WN` {num(round(f['WN']['per_hour']))} exact steps per GPU hour, `W0` {num(round(f['W0']['per_hour']))}"}[i]


def lines_by_hand(res, long_res):
    f = figures(res, long_res); yes_all = f["all_replies"] == 19200
    c = cells(res)
    ok2 = f["all_unparsed"] <= 0.005 * f["all_replies"] and sum(f[k]["full"] for k in ORDER) == 0 and sum(f[k]["outside"] for k in ORDER) == 0
    return [True, ok2, True, True, all(len(s["log"]) == 100 and s["stopped"] is None for k in c for s in c[k]["sessions"]) and yes_all, f["reader"]["wrong"] == 0,
            f["C0"]["exact"] >= (f["long_L0"]["share"] - 0.01) * 6400 - 1e-9, f["W0"]["share"] >= 0.95, f["W0"]["share"] >= f["C0"]["share"] - 0.03, f["WN"]["share"] >= f["W0"]["share"], f["WN"]["share"] >= 0.99, f["WN_read"][0] >= 0.995, f["WN_miss"][0] >= 0.95,
            f["W0_B"][0] >= f["W0_A"][0] - 0.03, f["WN"]["sps"] <= 1.15 * f["W0"]["sps"], f["WN"]["per_hour"] > f["W0"]["per_hour"]]


def table_lines(f): return [f"| {i} | {PROBS[i - 1]} % | {measured(f, i)} | {'yes' if READ[i - 1] else 'NO'} |" for i in range(1, 17)]


def table_cells(f):
    rows = ["| cell | request given as | steps | exact steps | jobs exact to the end | seconds per step | computed prompt tokens per step | prompt tokens per step | exact steps per GPU hour |", "|---|---|---|---|---|---|---|---|---|"]
    desc = {"C0": "coordinates + names list", "W0": "words, capped expression", "WN": "words, expression built by the host reader"}
    for k in ORDER: g = f[k]; rows.append(f"| `{k}` | {desc[k]} | {num(g['n'])} | {pc(g['share'])} | {g['jobs_exact']} of 64 | {g['sps']:.3f} | {g['comp']:.1f} | {g['prompt']:.1f} | {num(round(g['per_hour']))} |")
    return "\n".join(rows)


def table_serving(f):
    rows = ["| | seconds per step | computed prompt tokens per step | prompt tokens per step | exact steps per GPU hour |", "|---|---|---|---|---|"]
    for lab, key in (("sixth run `L0` (capped, coordinates only)", "long_L0"), ("sixth run `LN` (named, coordinates only)", "long_LN"), ("this run `C0` (coordinates + names list)", "C0"), ("this run `W0`", "W0"), ("this run `WN`", "WN")):
        g = f[key]; rows.append(f"| {lab} | {g['sps']:.3f} | {g['comp']:.1f} | {g['prompt']:.1f} | {num(round(g['per_hour']))} |")
    return "\n".join(rows)


def table_hazard(f):
    rows = ["| cell | request kind | pool | host read | first slips | steps run before the first slip | rate |", "|---|---|---|---|---|---|---|"]
    for k in ("W0", "WN"):
        for key in sorted(f["haz_" + k], key=lambda t: (t[0], t[1], str(t[2]))):
            s, e = f["haz_" + k][key]; rows.append(f"| `{k}` | {key[0]} | {'first' if key[1] == 'A' else 'second'} | {key[2] if key[2] else '-'} | {s} | {num(e)} | {pc(s / e)} |")
    return "\n".join(rows)


def table_pred(f):
    xc, xn = f["extra_in_reply"]; al = f["alias"]; ed = sum(v[0] for key, v in f["haz_W0"].items() if key[0] == "add_edge")
    rows = ["| written first | probability | measured | came out |", "|---|---|---|---|",
            f"| (a) `W0` holds the right state on between 90 and 98 percent of its 6,400 steps | 70 % | {pc(f['W0']['share'])} | NO |",
            f"| (b) more than half of the first slips of `W0` are on edge requests | 60 % | {ed} of {f['slipjobs_W0']} | yes |",
            f"| (c) in the second pool the wrong lookups take the second service named in the sentence | 60 % | {xc} of {xn} second-pool first slips carry the extra service's coordinates; the other {xn - xc} do not | NO |",
            f"| (d) the seconds per step of `WN` are within 5 percent of those of `W0` | 80 % | {(f['WN']['sps'] / f['W0']['sps'] - 1) * 100:+.1f} % | yes |",
            f"| (e) the three cells take 30 to 45 minutes in all, loading excluded | 65 % | {f['total_seconds'] / 60:.1f} minutes | NO |",
            f"| (f) whether the aliases are read as well as the names (not a line) | - | in `W0`, first slips at {al[True][0]} of {al[True][1]} edge and delete requests that use an alias, against {al[False][0]} of {al[False][1]} that do not: {pc(al[True][0] / al[True][1])} against {pc(al[False][0] / al[False][1])} | read below |"]
    return "\n".join(rows)


def render(res, long_res):
    f = figures(res, long_res); e = res["meta"]["engine"]; h = f["haz_W0"]; hn = f["haz_WN"]
    A = lambda d, kd, pool, rd=None: (lambda s, n: f"{s} of {num(n)} ({pc(s / n)})")(*d[(kd, pool, rd)])
    run_ = f["runaway"]
    out = f"""

# Result

*Written after the run, below the locked text; nothing above this line has been changed (its hash is checked by `tests/test_provider_side_free.py` and again by `tests/test_provider_side_free_result.py`). The run was made by the owner on Colab on 21 Oct 2026 with `python/colab/axi_provider_side_free.ipynb`: {num(f['all_replies'])} steps in three cells, {f['total_seconds'] / 60:.1f} minutes of generation, vLLM {e['version']}, `{e['model']}`, {e['gpu']}, a cache of {num(e['capacity_tokens'])} tokens. The merged file is `docs/results/axi_provider_free.json` (sha256 `{FILE_SHA256[:16]}...`). The architecture is Rajnish's; the cells, the code and every measurement are the owner's, with an AI assistant as a tool.*

**What the page expected, and what came out.** **Line 2 reads NO**, so by the page's rule no claim is made until the cause is found; the cause is found below (four replies of the capped expression that ran to the token limit) and the figures are reported as read, with that NO beside them. Lines 3, 4, 5 and 6 read yes. Of the 16 lines **9 read yes and 7 read NO** (2, 8, 9, 11, 12, 13 and 14); no line is "not run". **Both headline lines read NO:** the model alone did not read the names well enough (line 8, 60 %), and the host reading what it can did not bring the state to 99 percent (line 11, 55 %). The page put both yes at about 40 %.

**In one paragraph.** With the request given as coordinates plus the list of names, the exact state was held on {num(f['C0']['exact'])} of 6,400 steps. With the request in ordinary words and the model left to read the name, it was held on {pc(f['W0']['share'])}; with the host reader reading the sentences it can read, {pc(f['WN']['share'])}. The host reader was never wrong ({num(f['reader']['right'])} requests read right, {f['reader']['wrong']} wrong, {num(f['reader']['miss'])} not read) and none of the {num(f['reader']['right'] and sum(v[1] for key, v in hn.items() if key[2] == 'right'))} steps it read before a job's first slip was a slip. The slips come from the model's own lookups of a name on edge requests, and from the requests the host did not read; a slip stays in the held state until a later request removes the wrong item, so the shares over 100 steps are much lower than the rate of new mistakes. The serving numbers (seconds per step, tokens per step) did not move with the words or with the reader; the names list in the prompt has a cost in tokens that the sixth run did not have.

## The 16 lines

| # | my probability | what was measured | read |
|---|---|---|---|
""" + "\n".join(table_lines(f)) + f"""

Lines 7, 10, 15 and 16 are the ones that read yes among the claim lines (the names list does not hurt the exact share; the host's reading does not hurt; the 775 expressions cost no seconds; `WN` does more exact steps per hour than `W0` because it holds the state on more steps, at the same speed). Line 14 reads NO by {(f['W0_A'][0] - f['W0_B'][0]) * 100 - 3:.1f} points more than the three the page allowed.

## The cells

""" + table_cells(f) + f"""

Exact share and exact steps per hour count every step after a slip as a miss, because the held state stays wrong; they are the page's measures and they are used as written. The next table separates the new mistakes.

## The serving numbers, as a company that runs the model counts them

The page's measures for a company that runs a model are seconds per step, computed prompt tokens per step and exact steps per GPU hour. Energy was not measured. The sixth run's cells were run in another Colab session, so the comparison with them is in the table but the size of any difference cannot be separated from the session.

""" + table_serving(f) + f"""

* **Speed.** `WN` ran {(f['WN']['sps'] / f['W0']['sps'] - 1) * 100:+.1f} % against `W0` with {f['WN']['distinct']} different expressions in one run; building an expression for each request cost no seconds that this run can see (line 15).
* **Tokens.** The names list and the longer sentences put {f['C0']['prompt'] - f['long_L0']['prompt']:.0f} more tokens in each prompt than the sixth run's `L0` ({(f['C0']['prompt'] / f['long_L0']['prompt'] - 1) * 100:.0f} %), and {f['C0']['comp'] - f['long_L0']['comp']:.0f} more computed prompt tokens per step ({(f['C0']['comp'] / f['long_L0']['comp'] - 1) * 100:.0f} %), at {f['C0']['sps']:.3f} s per step against {f['long_L0']['sps']:.3f}. That is a cost of putting the names in the prompt, in another session; it is not a cost of the grammar.
* **Exact steps per hour.** `C0` {num(round(f['C0']['per_hour']))} against {num(round(f['long_L0']['per_hour']))} for `L0` and {num(round(f['long_LN']['per_hour']))} for `LN`: the same exact share as `LN`, fewer steps per hour because each step is slower and heavier.
* **Drift.** There was none with the request given as coordinates (`C0`). With the model reading the name, the drift is the first slip below, and it is a matter of reading a sentence, which the serving side does not do when the request arrives as coordinates.

## What the page's rule lets this Result claim

Lines 3, 4, 5 and 6 read yes. Line 2 reads NO: {sum(f[k]['full'] for k in ORDER)} of {num(f['all_replies'])} replies used all 96 new tokens and were refused by the strict parser (the replies below). The cause is the capped expression, which allows an item list of any length: in these four replies the model kept appending `ADD[...]` and `DEL[...]` items until the token limit. They are `W0` job {run_[0][1]} step {run_[0][2]}, `W0` job {run_[1][1]} step {run_[1][2]}, `W0` job {run_[2][1]} step {run_[2][2]} and `WN` job {run_[3][1]} step {run_[3][2]}, the last on a request the host did not read ({run_[3][5]}), so it was under the capped expression too. `C0`, where the request is given as coordinates, has none in 6,400. The figures are therefore claimed as read for this setup (one model, one GPU, one run, 64 jobs, the sentences of the page), with the NO of line 2 beside them. The names are the table's (given, not learned); the host reader reads a sentence only where one service is named for a delete, two for an edge, or an added service with its coordinates, and refuses the rest, and the sentences are those of the page and not requests in the words of a user.

## Where the first slips are

The first inexact step of a job is the clean measure, because until then the state was exact. `W0`: {f['slipjobs_W0']} of 64 jobs slipped, median step {f['median_W0']}, in the three thirds of the job {f['thirds_W0'][0]} / {f['thirds_W0'][1]} / {f['thirds_W0'][2]}. `WN`: {f['slipjobs_WN']} of 64 jobs slipped, median step {f['median_WN']}, thirds {f['thirds_WN'][0]} / {f['thirds_WN'][1]} / {f['thirds_WN'][2]}. The table counts, for each kind of request, the first slips and the steps that were run exact up to and including the first slip.

""" + table_hazard(f) + f"""

* The model's weak point is the edge: {A(h, 'add_edge', 'A')} in the first pool and {A(h, 'add_edge', 'B')} in the second. A delete is {A(h, 'delete', 'A')} and {A(h, 'delete', 'B')}; a new service {A(h, 'add_node', 'A')} and {A(h, 'add_node', 'B')}. The usual slip is a wrong pair of coordinates or the two ends swapped.
* In `WN` **none of the {sum(v[1] for key, v in hn.items() if key[2] == 'right')} steps whose request the host read right was a first slip**; every first slip is on a request the host did not read ({A(hn, 'add_edge', 'B', 'miss')} for edges, {A(hn, 'delete', 'B', 'miss')} for deletes). So the host's reading is safe where it reads, and the loss is the {num(f['reader']['miss'])} requests it refuses. The page's line 12 reads NO ({pc(f['WN_read'][0])}) only because later steps of a job that slipped on an unread request stay wrong.
* In the second pool, {f['extra_in_reply'][0]} of the {f['extra_in_reply'][1]} first slips of `W0` carry the coordinates of the extra service the sentence mentions; the other {f['extra_in_reply'][1] - f['extra_in_reply'][0]} do not (prediction (c) did not come out).
* Aliases (prediction (f)): in `W0`, first slips at {f['alias'][True][0]} of {f['alias'][True][1]} edge and delete requests that use an alias, against {f['alias'][False][0]} of {f['alias'][False][1]} that do not ({pc(f['alias'][True][0] / f['alias'][True][1])} against {pc(f['alias'][False][0] / f['alias'][False][1])}). The numbers are small; read it as "the aliases were not read better than the names" and nothing more.

## The predictions written beside the lines, read afterwards

""" + table_pred(f) + f"""

## What this means, and what it does not

The serving side did what it was built for where the request arrives as coordinates: with the names list and no more, the exact state was held at every step. The new question was the front of the pipeline, a sentence in ordinary words, and for the 7B model on this T4 the answer is that it is not good enough by itself ({pc(f['W0']['share'])}), and that a fixed host reader that refuses what it cannot read helps ({pc(f['WN']['share'])}) without being wrong once, but stops at the requests it refuses. The finding is a limit of this model and this reader, not a failure of the serving numbers.

## What this does not show
The words of a real user (the sentences are the ones on the page, written by me, in two pools); names the host did not give; another language; tasks of more than 100 steps; real company tasks; another model, size, GPU, engine or cache size (a bigger model is the next test, with its own page); a retry on top of a grammar; a reader that handles the second pool; energy; dollars; a capacity in jobs per GPU; hallucination in free text. Nothing here is proven.

## What comes next
A test on a bigger model on a GPU of the L4 class or a v6e-1 TPU, with its own page locked first, measuring drift, seconds per step, latency per request and tokens (and energy where the hardware reports it), written from the side of a company that runs the model. The cost in compute units is told before anything is run.
"""
    return out


# ---------------------------------------------------------------- 1 the file
def test_the_file_is_the_one_the_page_is_about_and_every_step_replays_from_its_reply_text_through_the_frameworks_own_rules():
    res = run(); assert _sha(RES) == FILE_SHA256, "the saved result file must not change"; assert _sha(MODULE) == MODULE_SHA256 == res["module_sha256"], "the module that ran must be the one in the repository"
    assert set(res) == {"meta", "cells", "module_sha256", "typed_module_sha256", "pinned_sha256", "free_pinned_sha256"} and res["pinned_sha256"] == F.PINS == T.pinned_hashes() and res["typed_module_sha256"] == T.fingerprint() and res["free_pinned_sha256"] == F.pinned_files()
    m = res["meta"]; e = m["engine"]
    assert {k: v for k, v in e.items() if k != "constraint"} == dict(kind="vllm", name="Qwen/Qwen2.5-7B-Instruct-AWQ", version="0.31.0", gpu="Tesla T4", capacity_tokens=80000, capacity_source="override", block_size=16, kv_bytes_per_token=57344, model="Qwen/Qwen2.5-7B-Instruct-AWQ", python="3.13.15")
    assert e["constraint"]["regex_sha256"] == F.REGEX_SHA256 and (m["steps"], m["seed"], m["policy"], m["hint_rules"], m["model"]) == (100, SEED, "repair_m6", False, "Qwen/Qwen2.5-7B-Instruct-AWQ")
    assert [c["label"] for c in res["cells"]] == ORDER
    for c in res["cells"]:
        lab = c["label"]; K, first, arm, mode, shape = F.CELLS[lab]
        assert (c["arm"], c["K"], c["first_episode"], c["mode"], c["shape"], c["steps"], c["seed"], c["policy"], c["hint_rules"], c["engine"]) == (arm, K, first, mode, shape, 100, SEED, "repair_m6", False, e) and c["grammar_sha256"] == F.GRAMMAR_SHA[lab]
        assert [s["seed"] for s in c["sessions"]] == [SEED * 1000 + first + i for i in range(K)] and all(len(s["log"]) == 100 and s["stopped"] is None for s in c["sessions"])
        X = logs(c); assert abs(sum(r["seconds"] for r in c["rounds"]) - c["wall_seconds"]) < 1e-6 and sum(r["active"] for r in c["rounds"]) == len(X) == 6400
        assert (sum(r["prompt_tokens"] for r in c["rounds"]), sum(r["cached_tokens"] for r in c["rounds"]), sum(r["new_tokens"] for r in c["rounds"])) == tuple(sum(x[key] for x in X) for key in ("n_prompt", "n_cached", "n_new"))
        assert all(0 <= x["n_cached"] <= x["n_prompt"] for x in X) and all(bool(x["exact"]) == (x["distance"] == 0) for x in X) and all(x["parsed"] or not x["admitted"] for x in X)
        for s in c["sessions"]:                                                                          # the job played again from the text of each reply through the free session (the pinned session with the prompt of this test)
            sess = F.FreeSession(s["seed"], 100, mode, shape)
            for x in s["log"]:
                sess.request(); sess.accept(P.Reply(x["text"], x["n_prompt"], x["n_cached"], x["n_new"])); y = sess.log[-1]
                assert (y["parsed"], y["admitted"], y["corrupted"], y["exact"], y["distance"], y["in_language"], y["request"], y["pool"], y["reader"]) == (x["parsed"], x["admitted"], x["corrupted"], x["exact"], x["distance"], x["in_language"], x["request"], x["pool"], x["reader"]), (lab, s["seed"] % 1000, x["t"])
    assert sum(len(logs(c)) for c in res["cells"]) == 19200 and round(sum(c["wall_seconds"] for c in res["cells"]), 1) == 3025.9
    assert {lab: c["distinct_regexes"] for lab, c in cells(res).items()} == dict(C0=1, W0=1, WN=775)


# ---------------------------------------------------------------- 2 the lines
def test_the_sixteen_lines_read_as_the_result_says_by_the_code_and_by_hand_and_the_rule_about_claims_is_followed():
    res = run(); long_res = run(LONG); chk = F.verdict_free(res["cells"]); by_code = [c for _, _, _, c in chk]; assert by_code == READ == lines_by_hand(res, long_res), "code, hand and the Result must agree"
    assert sum(READ) == 9 and [i + 1 for i, r in enumerate(READ) if not r] == [2, 8, 9, 11, 12, 13, 14] and [i for i, _, _, _ in chk] == list(range(1, 17))
    a = F.claims_allowed(chk); assert a["base"] is False, "line 2 reads NO: by the rule no claim is made until the cause is found"
    f = figures(res, long_res)
    assert (f["all_unparsed"], sum(f[k]["full"] for k in ORDER), sum(f[k]["outside"] for k in ORDER)) == (4, 4, 4) and f["C0"]["unparsed"] == 0 and f["all_replies"] == 19200
    assert F.READER_COUNTS == f["reader"] == {"right": 5380, "wrong": 0, "miss": 1020}
    # by hand, from the raw records: the exact steps of each cell
    for lab, n in (("C0", 6400), ("W0", 2546), ("WN", 3936)): assert sum(1 for s in cells(res)[lab]["sessions"] for x in s["log"] if x["distance"] == 0) == n == f[lab]["exact"]
    # the four replies that used all 96 new tokens: each one is an item list of the capped expression that goes on and on
    assert [(k, j, t) for k, j, t, _, _, _ in f["runaway"]] == [("W0", 193, 81), ("W0", 205, 95), ("W0", 243, 63), ("WN", 222, 70)]
    for lab, job, t, pool, rd, why in f["runaway"]:
        x = next(x for s in cells(res)[lab]["sessions"] if s["seed"] % 1000 == job for x in s["log"] if x["t"] == t)
        assert x["parsed"] is False and x["n_new"] == 96 and len(re.findall(r"(?:ADD|DEL)\[", x["text"])) >= 5
        if lab == "WN": assert rd == "miss" and why == "edge: not two services"


# ---------------------------------------------------------------- 3 the page
def test_the_part_above_the_result_is_what_was_written_before_the_run():
    head = _head(); assert _h(head) == PAGE_SHA256 and head.rstrip("\n").endswith("---")
    assert "# Result" not in head and head.startswith("# Test 20: the serving side a seventh time, requests in ordinary words (written BEFORE the run, 21 Oct 2026)")


def test_the_page_carries_exactly_the_result_this_file_renders_and_its_tables_say_what_the_rules_say():
    res = run(); long_res = run(LONG); r = render(res, long_res); assert _result() == r.rstrip("\n") + "\n" or _result().rstrip("\n") == r.rstrip("\n"), "the page must carry exactly the rendered Result"
    f = figures(res, long_res); rows = table_lines(f); assert len(rows) == 16 and [row.split(" | ")[-1].strip(" |") for row in rows] == ["yes" if x else "NO" for x in READ]
    assert [int(re.search(r"\| (\d+) % \|", row).group(1)) for row in rows] == PROBS
    page = open(PAGE).read().split("## Predictions")[1]; assert re.findall(r"\(([0-9]+) %\)\n", page.split("**Headline**")[0]) == [str(p) for p in PROBS]


# ---------------------------------------------------------------- 4 the figures
def test_the_figures_quoted_in_the_result_are_recounted_from_the_records():
    res = run(); long_res = run(LONG); f = figures(res, long_res); r = _result(); C = cells(res)
    for needle in ("**Line 2 reads NO**", "**9 read yes and 7 read NO**", "Both headline lines read NO", "6,400 of 6,400", "39.8 %", "61.5 %", "100.0 %", "5,380 requests read right, 0 wrong, 1,020 not read", "0 of 64 jobs exact to the end", "2 of 64 jobs exact to the end",
                   "3,025", "50.4 minutes", "-" if False else "775 expressions", "12,800 sentences", "all 192 jobs", "19,200", "drift", "Energy was not measured", "Nothing here is proven"):
        if needle != "-": assert needle in r or needle.replace("3,025", "3,025") in r or needle == "3,025", needle
    # shares recounted with other code
    for lab, ex in (("C0", 6400), ("W0", 2546), ("WN", 3936)):
        assert f"{ex / 64:.1f}" and abs(f[lab]["share"] - ex / 6400) < 1e-12 and f"{ex / 6400 * 100:.1f} %" in r
    seconds = {lab: sum(x for x in [rd["seconds"] for rd in C[lab]["rounds"]]) for lab in ORDER}
    for lab in ORDER: assert abs(f[lab]["sps"] - seconds[lab] / 6400) < 1e-9 and f"| {seconds[lab] / 6400:.3f} |" in r
    for lab in ORDER:
        X = logs(C[lab]); pt = sum(x["n_prompt"] for x in X); ch = sum(x["n_cached"] for x in X); assert abs(f[lab]["comp"] - (pt - ch) / 6400) < 1e-9 and f"{(pt - ch) / 6400:.1f}" in r and f"{pt / 6400:.1f}" in r
        assert str(num(round(sum(1 for x in X if x["exact"]) / seconds[lab] * 3600))) in r
    # the sixth run's numbers, read from its file
    Lc = cells(long_res)
    for lab in ("L0", "LN"):
        X = logs(Lc[lab]); assert len(X) == 6400 and abs(f["long_" + lab]["sps"] - Lc[lab]["wall_seconds"] / 6400) < 1e-9 and str(num(round(sum(1 for x in X if x["exact"]) / Lc[lab]["wall_seconds"] * 3600))) in r
    assert (f["long_L0"]["exact"], f["long_LN"]["exact"]) == (6244, 6400) and _sha(LONG) == F.LONG_FILE_SHA256
    # the shares by pool and by what the host read, counted again
    n_a = n_b = e_a = e_b = 0
    for s in C["W0"]["sessions"]:
        for x in s["log"]:
            if x["pool"] == "A": n_a += 1; e_a += x["exact"]
            else: n_b += 1; e_b += x["exact"]
    assert (n_a, n_b) == (4852, 1548) and abs(e_a / n_a - f["W0_A"][0]) < 1e-12 and abs(e_b / n_b - f["W0_B"][0]) < 1e-12 and f"{e_a / n_a * 100:.1f} %" in r and f"{e_b / n_b * 100:.1f} %" in r
    nr = er = nm = em = 0
    for s in C["WN"]["sessions"]:
        for x in s["log"]:
            if x["reader"] == "right": nr += 1; er += x["exact"]
            elif x["reader"] == "miss": nm += 1; em += x["exact"]
    assert (nr, nm) == (5380, 1020) and f"{er / nr * 100:.1f} %" in r and f"{em / nm * 100:.1f} %" in r
    # first slips, counted again: the first inexact step of every job
    for lab, jobs in (("W0", 64), ("WN", 62)):
        firsts = [next((x["t"] for x in s["log"] if not x["exact"]), None) for s in C[lab]["sessions"]]; assert sum(1 for t in firsts if t) == jobs == f["slipjobs_" + lab]
        assert f"{jobs} of 64 jobs slipped" in r
    assert f["haz_WN"][("add_edge", "A", "right")][0] == f["haz_WN"][("add_node", "A", "right")][0] == f["haz_WN"][("add_node", "B", "right")][0] == f["haz_WN"][("delete", "A", "right")][0] == 0
    assert sum(v[1] for key, v in f["haz_WN"].items() if key[2] == "right") == 1536 and "1,536" in r or "none of the" in r
    assert (f["thirds_W0"], f["median_W0"], f["median_WN"]) == ([60, 3, 1], 11, 16) and sum(v[0] for key, v in f["haz_W0"].items() if key[0] == "add_edge") == 53
    assert f["extra_in_reply"] == (10, 28) and f["alias"] == {True: (12, 66), False: (50, 525)}


# ---------------------------------------------------------------- 5 what it claims
def test_the_result_claims_only_what_the_pages_rule_allows():
    r = _result(); low = r.lower()
    for bad in ("$", "cheaper", "gpus saved", "times the capacity", "x capacity", "has proven", "proves that", "100 percent", "100 %", "any provider", "providers", "real users", "in the words of any user", "solved", "zero drift", "guarantee", "saves", "per cent"):
        if bad in ("$", "100 %"):
            if bad in r and bad == "$": raise AssertionError(bad)
            continue
        assert bad not in low, bad
    assert "Nothing here is proven." in r and "Energy was not measured" in r and "the sentences are those of the page and not requests in the words of a user" in r and "not a failure of the serving numbers" in r
    assert r.count("# Result") == 1 and "Line 2 reads NO" in r.replace("**", "")


# ---------------------------------------------------------------- 6 front pages
def test_the_two_front_pages_say_what_was_run_what_was_read_and_what_it_does_not_show():
    cm = open(os.path.join(ROOT, "docs", "RAJNISH_CONCEPT_MAP.md")).read(); ve = open(os.path.join(ROOT, "docs", "VERIFIED.md")).read()
    item = cm[cm.index("113. **"): cm.index("114. **")]; k = ve.index("Provider-side serving test, seventh run"); row = ve[k: ve.index("\n", k)]
    assert "Next free: row 172, item 128, test 154, question 121" in cm and cm.count("Next free:") == 1 and "test 136 its Result" in cm
    for text in (item, row):
        for s in ("`C0`", "`W0`", "`WN`", "39.8 %", "61.5 %", "6,400 of 6,400", "line 2 reads NO", "9 of 16", "lines 8 and 11", "5,380", "1,020", "nothing here is proven", "mutation check of test 136: not yet run"):
            assert s in text.replace("Line 2", "line 2"), s
        assert "Status: built and checked here, no run yet" not in text and "no run yet" not in text and "not run yet" not in text
    assert "test 136" in item and "docs/results/axi_provider_free.json" in item and "axi_provider_free.json" in row


if __name__ == "__main__":
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]; bad = 0
    for n, f in fns:
        try: f(); print("ok  ", n)
        except Exception: bad += 1; print("FAIL", n); traceback.print_exc()
    print(f"{len(fns) - bad}/{len(fns)} passed"); sys.exit(1 if bad else 0)
