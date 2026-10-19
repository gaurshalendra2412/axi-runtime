"""Implements concept-map item 107: RESULT of the constrained provider-side serving test of 18 and 19 Oct 2026. Page: docs/PROVIDER_SIDE_CONSTRAINED_TEST.md (predictions and reading rules written before the run; the result is appended below them).
The files: docs/results/provider_side_constrained_cells/ (the six cell files the owner's Colab runs saved: S and KC at K = 8, 32 and 64; Tesla T4, vLLM 0.31.0, Qwen2.5-7B-Instruct-AWQ, a cache of 80,000 tokens, every reply of both arms constrained by one regular expression)
and docs/results/axi_provider_constrained.json (the six put together by provider_side_constrained.merge_files, with the sha256 of the module that did it).
This file recomputes from the saved files, with its own arithmetic and not with the code that printed the figures:
  1. the files: their sha256, the merge (the merged file is what merge_cells makes of the six, plus the module's sha256, and the module on disk is that module), the settings, the seeds, the same job script in every cell, the rounds adding up to the cell, every cached count at most its prompt;
  2. the seventeen reading rules (14 yes, lines 2, 5 and 6 NO, none not run), the headline (lines 15 and 17 yes), by the code and by hand from the raw numbers;
  3. the 'what was measured' text of each line of the page, the tables and the figures quoted in the prose of the Result (the whole Result is `render(result)`, and the page must equal the locked head followed by it);
  4. that the part of the page above the Result is byte for byte what was written before the run;
  5. the counts of the part written afterwards (which jobs missed and why, the one reply that ran to 96 tokens, the rounds of the 64-job kept chat, the resumed rounds, the side predictions) from the step records and the round records;
  6. that the Result says what the page said it would say when line 2 reads NO and lines 5 and 6 read NO, and claims nothing more.
What it does NOT show: anything about another GPU, model, engine or cache size, dollars, energy, retries, a random order of requests, or a grammar that bounds the length of a name."""
import collections, hashlib, json, os, re, sys

sys.path.insert(0, os.path.dirname(__file__))
import axi.experiments.provider_side as P
import axi.experiments.provider_side_constrained as C

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..", "..")
PAGE = os.path.join(ROOT, "docs", "PROVIDER_SIDE_CONSTRAINED_TEST.md")
RES = os.path.join(ROOT, "docs", "results", "axi_provider_constrained.json")
CELLDIR = os.path.join(ROOT, "docs", "results", "provider_side_constrained_cells")
RUN1 = os.path.join(ROOT, "docs", "results", "axi_provider_side.json")
MODULE = os.path.join(HERE, "..", "axi", "experiments", "provider_side_constrained.py")
PS_MODULE = os.path.join(HERE, "..", "axi", "experiments", "provider_side.py")
RESULT_MARK = "\n\n---\n\n# Result"
PAGE_SHA256 = "4559fdc4eda01013d456c1407c0bd23e3e634751394d6f65cc90d891cca7e575"
FILE_SHA256 = "7f54c3dd37fd4dcf61e8a7991af285d6ad1a0d691362496216949a26b45b819d"
MODULE_SHA256 = "03452148e28c9f31372f12cc11ef1808ac67308adff972fa19d4aeeef7c96f7d"
PS_SHA256 = "bfb4734e171d9244cd47d10402be46528a557095a8275d02b38481a0e2b0b1d3"
REGEX_SHA256 = "21cdc0bf7824cff3866413dd74fc5939ccc01135e3b241dfcf9fbb1440bb0ec9"
CELL_SHA256 = {"KC_K32": "b925183823939ca65589d88de49589d11ad61741d6ac725a8cfc9c9b765f00e7", "KC_K64": "b64031ee2b9b7b5e41273bef8ae82799447fcff67112fca9d380c3af2081533e", "KC_K8": "b5321da20675a98375f628873256fcdbc133cfcb2cdded4331e199e30e023099",
               "S_K32": "90d2179cb457198c84d168a2ac4a9859a017503e24346cf3def9f8530101aaa4", "S_K64": "e2fe7cc72383759130b854f0cc3c03a40a0fb4383beef88a899445189c7df759", "S_K8": "df2b5866e5507b3d29e1e27936546391daf1d536c6dd742623ebb2991267628b"}
PROBS = [97, 94, 95, 85, 5, 2, 95, 94, 95, 82, 94, 96, 96, 95, 88, 93, 92]
READ = [True, False, True, True, False, False, True, True, True, True, True, True, True, True, True, True, True]
ORDER = [("S", 8), ("KC", 8), ("S", 32), ("KC", 32), ("S", 64), ("KC", 64)]
SEED, STEPS = 20261017, 30
EST = {8: 88.3, 32: 93.9, 64: 95.9}                      # the page's optimistic estimate of the exact share of S, from the first run's files
FIRST_MISS_JOBS = {8: [3, 4], 32: [3, 4, 17, 19, 20], 64: [3, 4, 17, 19, 20, 62]}
_cache = {}


def run():
    if "run" not in _cache: _cache["run"] = json.load(open(RES))
    return _cache["run"]


def run1():
    if "run1" not in _cache: _cache["run1"] = json.load(open(RUN1))
    return _cache["run1"]


def _text(): return open(PAGE).read()


def _head():
    t = _text(); assert t.count(RESULT_MARK) <= 1
    return t[:t.index(RESULT_MARK)] if RESULT_MARK in t else t


def _h(text): return hashlib.sha256((text.rstrip("\n") + "\n").encode()).hexdigest()


def _sha(path): return hashlib.sha256(open(path, "rb").read()).hexdigest()


# ---------------------------------------------------------------- the figures, from the step records
def pc(x, k=1): return f"{x * 100:.{k}f} %"
def num(x): return f"{x:,}"
def cells(res): return {(c["arm"], c["K"]): c for c in res["cells"]}
def logs(c): return [x for s in c["sessions"] for x in s["log"]]
def mean(v): v = list(v); return sum(v) / len(v)


def st(c):
    L = logs(c); n = len(L); wall = sum(r["seconds"] for r in c["rounds"]); prompt = sum(x["n_prompt"] for x in L); cached = sum(x["n_cached"] for x in L)
    late = [x for s in c["sessions"] for x in s["log"][15:]]; ex = sum(1 for x in L if x["exact"])
    return dict(n=n, wall=wall, sps=wall / n, prompt=prompt, cached=cached, computed=prompt - cached, cps=(prompt - cached) / n, share=cached / prompt,
                late=sum(x["n_cached"] for x in late) / sum(x["n_prompt"] for x in late),
                q1=mean(x["n_prompt"] for s in c["sessions"] for x in s["log"][0:7]), q4=mean(x["n_prompt"] for s in c["sessions"] for x in s["log"][23:30]),
                longest=max(x["n_prompt"] for x in L), ex=ex, exact=ex / n, eph=3600 * ex / wall, stopped=sum(1 for s in c["sessions"] if len(s["log"]) < 30),
                whole=sum(all(x["exact"] for x in s["log"]) for s in c["sessions"]), jobs=len(c["sessions"]), corrupted=sum(x["corrupted"] for x in L),
                unparsed=sum(1 for x in L if not x["parsed"]), full96=sum(1 for x in L if x["n_new"] >= 96), longest_ok=max(x["n_new"] for x in L if x["parsed"]))


def S_(res): return {k: st(c) for k, c in cells(res).items()}


def missed_jobs(c): return [i for i, s in enumerate(c["sessions"]) if not all(x["exact"] for x in s["log"])]


def came_back(c):
    out = []
    for i, s in enumerate(c["sessions"]):
        ex = [x["exact"] for x in s["log"]]
        if False in ex and any(ex[ex.index(False):]): out.append(i)
    return out


def oracle_replies(job):
    """what the oracle writes at every step of job `job` (the exact path of the job), with the kind of the step"""
    s = P.Session("S", SEED * 1000 + job, STEPS); out = []
    while s.t < len(s.script):
        kind = s.script[s.t]["kind"]; rq = s.request(); out.append((kind, rq.stand_in_reply)); s.accept(P.Reply(rq.stand_in_reply, 0, 0, 0))
    return out


def first_miss(c, job):
    return next((x for x in c["sessions"][job]["log"] if not x["exact"]), None)


def measured(res, i):
    s = S_(res); n = sum(v["n"] for v in s.values()); g = lambda a, k: s[(a, k)]
    parsed = lambda a, k: 1 - g(a, k)["unparsed"] / g(a, k)["n"]
    gap = lambda k: (g("KC", k)["exact"] - g("S", k)["exact"]) * 100
    ratio = lambda k: g("S", k)["eph"] / g("KC", k)["eph"]
    return {
        1: f"the engine reported the cached tokens of every request: {num(n)} steps in the {len(s)} cells that ran",
        2: f"parsed: `S` {pc(parsed('S', 8))} (K = 8), {pc(parsed('S', 32))} (32), {pc(parsed('S', 64))} (64), `KC` {pc(parsed('KC', 8))}, {pc(parsed('KC', 32))}, {pc(parsed('KC', 64))} (limit 99.5 %, met); replies that used all 96 new tokens: {g('S', 32)['full96']} in `S` at K = 32 and {g('S', 64)['full96']} in `S` at K = 64 (job 19, step 12, the same reply), none in the other four cells (limit: none, not met)",
        3: f"the longest kept-chat prompt at K = 64 was {num(g('KC', 64)['longest'])} tokens, 64 times that is {num(64 * g('KC', 64)['longest'])}, {64 * g('KC', 64)['longest'] / 80000:.1f} times the cache of 80,000 (limit 2); at K = 8 it was {num(g('KC', 8)['longest'])}, 8 times that is {num(8 * g('KC', 8)['longest'])}, which fits",
        4: f"the lowest exact share of the six cells is `S` at K = 8, {pc(g('S', 8)['exact'])} (limit 80 %); the others are {pc(g('S', 32)['exact'])}, {pc(g('S', 64)['exact'])} and 100.0 % in the three `KC` cells",
        5: f"`S` against `KC`: K = 8 {pc(g('S', 8)['exact'])} against {pc(g('KC', 8)['exact'])}, {gap(8):.1f} points short; K = 32 {pc(g('S', 32)['exact'])} against {pc(g('KC', 32)['exact'])}, {gap(32):.1f} points short; K = 64 {pc(g('S', 64)['exact'])} against {pc(g('KC', 64)['exact'])}, {gap(64):.1f} points short (limit 5; met at K = 64 only)",
        6: f"`S` exact steps: {pc(g('S', 8)['exact'])} (K = 8), {pc(g('S', 32)['exact'])} (32), {pc(g('S', 64)['exact'])} (64): none reaches 98 %",
        7: f"`S` at K = 8: mean prompt {g('S', 8)['q1']:.0f} tokens in the first 7 steps, {g('S', 8)['q4']:.0f} in the last 7: {g('S', 8)['q4'] / g('S', 8)['q1']:.2f} times (limit 1.3)",
        8: f"`KC` at K = 8: mean prompt {num(round(g('KC', 8)['q1']))} tokens in the first 7 steps, {num(round(g('KC', 8)['q4']))} in the last 7: {g('KC', 8)['q4'] / g('KC', 8)['q1']:.2f} times (limit 3.5)",
        9: f"{pc(g('KC', 8)['share'])} of the prompt tokens of `KC` at K = 8 came from the cache (limit 80 %)",
        10: f"`S` computes {g('S', 8)['cps']:.0f} prompt tokens a step, `KC` {g('KC', 8)['cps']:.0f}: {(1 - g('S', 8)['cps'] / g('KC', 8)['cps']) * 100:.1f} % fewer (limit 30 %)",
        11: f"`S` {g('S', 8)['sps']:.3f} seconds a step, `KC` {g('KC', 8)['sps']:.3f}: {(1 - g('S', 8)['sps'] / g('KC', 8)['sps']) * 100:.1f} % less (limit 20 %); `KC` took {g('KC', 8)['sps'] / g('S', 8)['sps']:.2f} times as long (exact steps: `S` {pc(g('S', 8)['exact'])}, `KC` {pc(g('KC', 8)['exact'])})",
        12: f"`KC` {g('KC', 32)['sps']:.3f} seconds a step, `S` {g('S', 32)['sps']:.3f}: {g('KC', 32)['sps'] / g('S', 32)['sps']:.1f} times (limit 1.3; exact steps: `S` {pc(g('S', 32)['exact'])}, `KC` {pc(g('KC', 32)['exact'])})",
        13: f"`KC` at K = 64: {pc(g('KC', 64)['late'])} of the prompt tokens of the second half came from the cache ({pc(g('KC', 64)['share'])} over all steps; limit: below 50 % in the second half)",
        14: f"`KC` {g('KC', 64)['sps']:.3f} seconds a step, `S` {g('S', 64)['sps']:.3f}: {g('KC', 64)['sps'] / g('S', 64)['sps']:.1f} times (limit 2; exact steps: `S` {pc(g('S', 64)['exact'])}, `KC` {pc(g('KC', 64)['exact'])})",
        15: f"`S` {num(round(g('S', 8)['eph']))} exact steps per GPU hour ({pc(g('S', 8)['exact'])} exact), `KC` {num(round(g('KC', 8)['eph']))} ({pc(g('KC', 8)['exact'])}): {ratio(8):.2f} times (limit 2)",
        16: f"`S` {num(round(g('S', 32)['eph']))} exact steps per GPU hour ({pc(g('S', 32)['exact'])} exact), `KC` {num(round(g('KC', 32)['eph']))} ({pc(g('KC', 32)['exact'])}): {ratio(32):.1f} times (limit 10)",
        17: f"`S` {num(round(g('S', 64)['eph']))} exact steps per GPU hour ({pc(g('S', 64)['exact'])} exact), `KC` {num(round(g('KC', 64)['eph']))} ({pc(g('KC', 64)['exact'])}): {ratio(64):.1f} times (limit 10)",
    }[i]


def table_lines(res):
    out = []
    for i, (tag, text) in enumerate(C.CL_LINES, 1):
        out.append(f"| {i} | {PROBS[i - 1]} % | {measured(res, i)} | {'yes' if READ[i - 1] else 'NO'} |")
    return out


def kk(a, k): return f"{k} (resumed)" if (a, k) == ("KC", 64) else str(k)


def table_cells(res):
    s = S_(res); out = []
    for (a, k) in ORDER:
        v = s[(a, k)]
        out.append(f"| `{a}` | {kk(a, k)} | {num(v['n'])} | {num(round(v['wall']))} | {v['sps']:.3f} | {num(v['prompt'])} | {num(v['computed'])} | {pc(v['share'], 0)} | {pc(v['late'], 0)} | {num(round(v['q1']))} | {num(round(v['q4']))} | {num(v['longest'])} | {pc(v['exact'])} |")
    return out


def table_anatomy(res):
    s = S_(res); c = cells(res); r1 = S_(run1()); r1c = cells(run1()); out = []
    for (a, k) in ORDER:
        v = s[(a, k)]; missed = missed_jobs(c[(a, k)]); back = came_back(c[(a, k)])
        first = f"{v['whole']} of {v['jobs']}" + (f" (first run {r1[(a, k)]['whole']})" if (a, k) in r1 else "")
        refused = f"{v['unparsed']}" + (f" (first run {r1[(a, k)]['unparsed']})" if (a, k) in r1 else "")
        out.append(f"| `{a}` | {kk(a, k)} | {first} | {refused} | {v['full96']} | {', '.join(map(str, missed)) if missed else 'none'} | {', '.join(map(str, back)) if back else 'none'} | {v['corrupted']} |")
    return out


def table_misses(res):
    c = cells(res)[("S", 64)]; out = []
    for job in FIRST_MISS_JOBS[64]:
        f = first_miss(c, job); o = oracle_replies(job)[f["t"] - 1]; answered = f["text"] if f["n_new"] < 96 else f["text"][:34] + "… (cut at 96 tokens)"
        where = "all three cells" if job < 8 else "K = 32 and 64" if job < 32 else "K = 64 only"
        out.append(f"| {job} | {f['t']} | `{o[0]}` | `{answered}` | `{o[1]}` | {where} |")
    return out


def table_hour(res):
    s = S_(res); out = []
    for (a, k) in ORDER:
        v = s[(a, k)]; out.append(f"| `{a}` | {kk(a, k)} | {num(v['ex'])} | {num(round(v['wall']))} | {num(round(v['eph']))} | {pc(v['exact'])} | {v['sps']:.4f} |")
    return out


def table_rounds(res):
    r = cells(res)[("KC", 64)]["rounds"]; out = []
    for i in (1, 3, 6, 7, 8, 10, 11, 12, 13, 14, 21, 22, 23, 30):
        x = r[i - 1]; out.append(f"| {i} | {x['active']} | {num(x['prompt_tokens'])} | {num(x['cached_tokens'])} | {pc(x['cached_tokens'] / x['prompt_tokens'])} | {x['seconds']:.1f} | {x['seconds'] / x['active']:.2f} |")
    return out


def back_steps(c):
    """job -> the step at which a job that had missed held the oracle's state again"""
    out = {}
    for i, s in enumerate(c["sessions"]):
        ex = [x["exact"] for x in s["log"]]
        if False in ex:
            j = ex.index(False)
            if any(ex[j:]): out[i] = j + ex[j:].index(True) + 1
    return out


def slip_pat(): return re.compile(r"^DEL\[\d+,\d+:-?\d+\]$")


def side(res):
    """the five side predictions of the page, read from the files"""
    c = cells(res); c1 = cells(run1()); r1 = S_(run1()); s = S_(res); pat = slip_pat(); nosv = re.compile(r"^DEL\[\d+,\d+\]$"); n = hit = 0
    for i, sn in enumerate(c1[("S", 64)]["sessions"]):
        f = next((x for x in sn["log"] if not x["exact"]), None)
        if f is not None and pat.match(f["text"].strip()):
            n += 1; g = c[("S", 64)]["sessions"][i]["log"][f["t"] - 1]; hit += bool(nosv.match(g["text"].strip()))
    dev = [s[("S", k)]["sps"] / r1[("S", k)]["sps"] - 1 for k in (8, 32, 64)]
    return dict(slip_n=n, slip_hit=hit, job4=c[("S", 64)]["sessions"][4]["log"][17]["text"], job3_exact_steps=sum(x["exact"] for x in c[("S", 64)]["sessions"][3]["log"]),
                job3_misses_at_16=not c[("S", 64)]["sessions"][3]["log"][15]["exact"], dev=dev, kc_lost=sum(1 for k in (8, 32, 64) if s[("KC", k)]["exact"] < 1))


def kc_stopped(res):
    out = {}
    for k in (32, 64):
        out[k] = [(i, len(sn["log"]), sn["stopped"]) for i, sn in enumerate(cells(res)[("KC", k)]["sessions"]) if len(sn["log"]) < 30]
    return out


def figs(res):
    s = S_(res); c = cells(res); e = res["meta"]["engine"]; n = sum(v["n"] for v in s.values()); wall = sum(v["wall"] for v in s.values()); g = lambda a, k: s[(a, k)]
    r1 = S_(run1()); kc64 = c[("KC", 64)]["rounds"]; kc32 = c[("KC", 32)]["rounds"]
    j19 = c[("S", 32)]["sessions"][19]["log"]; inexact19 = sum(1 for x in j19 if not x["exact"])
    sens_one = {k: 3600 * (g("S", k)["ex"] + 1) / g("S", k)["wall"] for k in (32, 64)}; sens_all = {k: 3600 * (g("S", k)["ex"] + inexact19) / g("S", k)["wall"] for k in (32, 64)}
    resumed = c[("KC", 64)]["resumed_at"]; between = lambda i: (kc64[i - 2]["seconds"] + kc64[i]["seconds"]) / 2          # round i against the mean of its two neighbours
    return dict(n=n, wall=wall, e=e, g=g, r1=r1, inexact19=inexact19, sens_one=sens_one, sens_all=sens_all, resumed=resumed, kc64=kc64, kc32=kc32, between=between, j19=j19,
                ratio=lambda k: g("S", k)["eph"] / g("KC", k)["eph"], tratio=lambda k: g("KC", k)["sps"] / g("S", k)["sps"],
                mspt=lambda k: g("KC", k)["wall"] / g("KC", k)["computed"] * 1000, tps=lambda k: g("KC", k)["computed"] / g("KC", k)["wall"])


def render(res):
    F = figs(res); g = F["g"]; e = F["e"]; r1 = F["r1"]; kc64 = F["kc64"]; kc32 = F["kc32"]; between = F["between"]; sd = side(res); stopped = kc_stopped(res)
    ratio, tratio, mspt, tps = F["ratio"], F["tratio"], F["mspt"], F["tps"]
    c = cells(res); jobs_missed = {k: missed_jobs(c[("S", k)]) for k in (8, 32, 64)}; back = {k: back_steps(c[("S", k)]) for k in (8, 32, 64)}
    longest_ok = max(v["longest_ok"] for v in S_(res).values()); share = lambda r: r["cached_tokens"] / r["prompt_tokens"]
    t = []
    a = t.append
    a(f"""

---

# Result

*Written after the run, below the locked text; nothing above this line has been changed (its hash is checked by `tests/test_provider_side_constrained.py` and again by `tests/test_provider_side_constrained_result.py`). The run was made by the owner on Colab on 18 and 19 Oct 2026 with `python/colab/axi_provider_side_constrained.ipynb`: vLLM {e['version']}, GPU {e['gpu']}, Python {e['python']}, `{e['model']}`, the cache fixed at {num(e['capacity_tokens'])} tokens with the engine's block override (capacity source "{e['capacity_source']}"), 30-step jobs, seed {SEED}, every reply of both arms constrained by the regular expression of this page (sha256 `{REGEX_SHA256[:16]}…`). **All six planned cells ran:** `S` and `KC` at K = 8, 32 and 64. The owner reported that the safety cell (Cell 6) printed OK, which by the page means that the probe and every reply of the safety cell parsed; the probe (step 20 of job 0) was answered `DEL[4,5]` where the first run had answered `DEL[4,5:8]`. `KC` at K = 64 was carried over three Colab sessions, as the page said it might be (resumed after rounds {F['resumed'][0]} and {F['resumed'][1]}). The six cell files are in `docs/results/provider_side_constrained_cells/`; I put them together here, not in Colab, with `provider_side_constrained.merge_files`: `docs/results/axi_provider_constrained.json` (sha256 `{FILE_SHA256}`; it records the sha256 of the module that merged it, `{MODULE_SHA256}`, and of `provider_side.py`, `{PS_SHA256}`). The owner's Cell 10 was run in a session whose folder held only the 64-job `KC` cell, so the file it merged had that one cell; the 64-job cell file here was taken out of that file, and the other five are the files from the Drive folder. The module that ran the cells is the one in the repository (the owner's file records the same sha256). The GPU seconds of the six cells (engine {e['version']}, GPU {e['gpu']}) add up to {num(round(F['wall']))}, which is {round(F['wall'] / 60)} minutes, {num(round(g('KC', 64)['wall']))} of them in `KC` at K = 64. Every figure below is computed from the step records and the round records of those files by `tests/test_provider_side_constrained_result.py`.*

**Three things about the run that the page did not foresee or guessed differently.**
* **One reply of `S` ran to the 96-token limit inside a name** (job 19, step 12; the same reply in the 32-job and the 64-job cell). That is why line 2 reads NO. It has its own section below, because the page said what a NO on line 2 would mean.
* **The slow cells took less time than I guessed.** `KC` at K = 64 took {g('KC', 64)['wall'] / 3600:.2f} hours of GPU (the page's guess: 3 to 5) and `KC` at K = 32 took {g('KC', 32)['wall'] / 60:.0f} minutes (80 to 90).
* **{len(stopped[32])} chats of the 32-job `KC` cell and {len(stopped[64])} of the 64-job cell ended before step 30 by the test's own rule** (the page's jobs row: a kept chat stops before its next prompt would pass 7,600 tokens): {', '.join(f'job {i} after {n} steps' for i, n, _ in stopped[64])} at 64 jobs, the first two also at 32 jobs. So `KC` did {num(g('KC', 32)['n'])} steps in the 32-job cell instead of 960 and {num(g('KC', 64)['n'])} in the 64-job cell instead of 1,920. The steps it did were all exact.

## The 17 lines

| # | my probability | what was measured | read |
|---|---|---|---|""")
    a("\n" + "\n".join(table_lines(res)) + "\n")
    a(f"""
**14 of the 17 lines read yes, lines 2, 5 and 6 read NO, and no line is "not run".** The headline (lines 15 and 17) reads yes on both lines, and so does line 16 between them; I had put both headline lines together at about 65 %. Lines 5 and 6 are the two NO I expected (5 and 2 %). **Line 2 is a NO I did not expect (94 %).** The pattern I called most likely (lines 5 and 6 NO and every other line yes, about 25 %) is therefore not the one that came out: line 2 is a third NO. The three lines I trusted least among those I expected to read yes (line 10 at 82 %, line 4 at 85 %, line 15 at 88 %) all read yes.

**What the page said about these NOs.**
* *Lines 5 and 6 NO, lines 15 to 17 yes:* "the constraint removed the format slips and left the content errors. Exact-state holds less of the right state than the kept chat at small K, by the amount measured, and still buys many more correct steps per GPU hour than the kept chat at every K. The article gets the exact-steps-per-hour figures with the exactness gap next to them, and not a plain time ratio." The first sentence is what happened to the format slips and the content errors (the section "What I read from it" has the counts). The plain time ratios (lines 11, 12, 14: {tratio(8):.2f}, {tratio(32):.1f} and {tratio(64):.1f} times, `S` exact on {pc(g('S', 8)['exact'])}, {pc(g('S', 32)['exact'])} and {pc(g('S', 64)['exact'])} of its steps against 100.0 % for `KC`) are in the tables because nothing is to be hidden, and are **not** claimed as ratios at equal exactness: line 5 reads NO.
* *Line 2 NO:* "the constraint did not hold. Exactness and times are not read until the cause is found", and in the rule that decides what the Result may claim: "Line 2 reads NO: neither is claimed." The next section is about that.

## Line 2: the cause, and what I do with the rule
**What happened.** At step 12 of job 19 (an `add_node` step; the oracle writes `ADD[3,0:4]`) the first run's engine had written `ADD[(3,0):4]` (in both cells in which it ran job 19), which the page counted among the 5 adds written like an edge (job 19 twice, job 3 three times). With the constraint, after `ADD[(3,0)` the grammar allows only `-`, so the reply was forced to continue as an edge, `ADD[(3,0)->(3,0):`, and the name after the colon (the grammar allows any letters, digits and underscores, with **no limit on how long**) became `4tworowsdown4colsright4tworowsdown4colsright…`, repeated until the 96 new tokens were used up. Every prefix of that reply is one the grammar allows, so the constraint held as designed; the reply was cut before its closing bracket and the strict parser refused it. Job 19 held the right state on {sum(x['exact'] for x in F['j19'])} of its 30 steps (steps 1 to 11) and on none after. The same reply appears in the 32-job and the 64-job cell, as it must with greedy decoding. Every other reply of the six cells that parsed is at most {longest_ok} tokens long. The other three of the five (job 3, `ADD[(4,4):6]`, in the 8-, 32- and 64-job cells) became `ADD[(4,4)->(4,4):value#6]`: well formed, parsed, and refused by the gate.

**What the rule says, and what I do.** The page withholds both kinds of claim when line 2 reads NO ("neither is claimed"). The cause is found, above, but the rule does not say that finding it lifts the NO, and I do not take it as lifting it. So: (1) every line is printed as it read, and line 2 is NO in the table and in the summary; (2) **this Result claims neither the plain time ratios nor the exact-steps-per-hour figures.** Both are reported as what was measured, each with the exactness of both arms next to it, and the sentence at the end says what may be reported and what has to wait; (3) I checked afterwards, and this was not pre-registered, how much that one reply could matter. Counting the cut-off step as exact would change `S`'s exact steps per GPU hour at 32 jobs from {num(round(g('S', 32)['eph']))} to {num(round(F['sens_one'][32]))} and at 64 jobs from {num(round(g('S', 64)['eph']))} to {num(round(F['sens_one'][64]))}; counting all {F['inexact19']} steps that job 19 got wrong as exact (an upper bound) would give {num(round(F['sens_all'][32]))} and {num(round(F['sens_all'][64]))}, and the ratios against `KC` would be {F['sens_all'][32] / g('KC', 32)['eph']:.1f} and {F['sens_all'][64] / g('KC', 64)['eph']:.1f} instead of {ratio(32):.1f} and {ratio(64):.1f}: lines 16 and 17 (limit 10) would still read yes, and the exact share of `S` would be {pc((g('S', 32)['ex'] + F['inexact19']) / g('S', 32)['n'])} and {pc((g('S', 64)['ex'] + F['inexact19']) / g('S', 64)['n'])}, still short of `KC`. (4) The clean way to clear line 2 is to run the three `S` cells again with a grammar that bounds the length of a name (a few minutes of GPU for the three; the `KC` cells would not change, since no reply of theirs is longer than 16 tokens and a reply the grammar allows is not changed by it). That is a new regular expression and so a new page with new predictions; I have not made it.

## Where my expectation was off, and where it was right
* **Line 2 (94 %): the grammar leaves the length of a name open, and a greedy loop found it.** I wrote that one risk was that "a reply is cut by the 96-token limit (the longest in the first run was 31)". The longest reply that parsed is still {longest_ok}; one reply of the {num(g('S', 32)['n'])} of `S` at 32 jobs and one of the {num(g('S', 64)['n'])} at 64 jobs (the same reply) did not stop. I had also written that an add written like an edge "cannot be made into the node add the model meant" and that I expected those to "stay wrong". They did, in two different ways (above).
* **The estimate of the exact share was right about which jobs miss.** The page's table gave 88.3, 93.9 and 95.9 % for `S` at 8, 32 and 64 jobs "if the rest stays exact" and named the jobs that would miss: 3 and 4; 3, 4, 17, 19 and 20; those and 62. **The jobs that missed are exactly those** ({', '.join(map(str, jobs_missed[8]))}; {', '.join(map(str, jobs_missed[32]))}; {', '.join(map(str, jobs_missed[64]))}) and each missed at the step the page gave, the 8-job figure is hit to the digit ({pc(g('S', 8)['exact'])}), and the other two came out {(g('S', 32)['exact'] - EST[32] / 100) * 100:.1f} and {(g('S', 64)['exact'] - EST[64] / 100) * 100:.1f} points higher ({pc(g('S', 32)['exact'])} and {pc(g('S', 64)['exact'])}) because jobs that were behind came back to the right state, which the page had allowed for ("once a job is behind it can come back"): {', '.join(f'job {j} at step {st_}' for j, st_ in back[64].items())}.
* **Lines 5 and 6 came out as I put them (5 % and 2 %).** The gap to the kept chat is {(g('KC', 8)['exact'] - g('S', 8)['exact']) * 100:.1f}, {(g('KC', 32)['exact'] - g('S', 32)['exact']) * 100:.1f} and {(g('KC', 64)['exact'] - g('S', 64)['exact']) * 100:.1f} points at 8, 32 and 64 jobs: over the limit of 5 at 8 and 32 jobs, under it at 64. Line 5 needs every K, so it reads NO. (The page expected the arms "equally exact only at 64 jobs, if at all"; that is where the gap came under the limit.) The 98 % of line 6 is reached by none of the three `S` cells.
* **The headline lines came out where the page put them.** Line 15: {ratio(8):.2f} times at 8 jobs (the page: "about 3.5 against a limit of 2"). Line 16: {ratio(32):.1f} at 32 jobs ("about 40 against a limit of 10"). Line 17: {ratio(64):.1f} at 64 jobs ("more than at 32"). The page said the exactness correction costs `S` "between 4 and 12 percent of its steps"; it cost {(1 - g('S', 8)['exact']) * 100:.1f}, {(1 - g('S', 32)['exact']) * 100:.1f} and {(1 - g('S', 64)['exact']) * 100:.1f} % at 8, 32 and 64 jobs (the 64-job figure is below the 4). It did not come near the size of the ratio, as the page said.
* **The side predictions, read afterwards as descriptions.** (a) Of the {sd['slip_n']} `S` jobs at 64 jobs whose first miss in the first run was the `DEL[r,c:v]` slip, {sd['slip_hit']} answered at that step with the delete without a value (the page: at least 90 %, my probability 88 %): true. (b) Job 4 answered `{sd['job4']}` at step 18 again and job 3 still missed at step 16 (85 %): true. (c) `S`'s seconds per step were {g('S', 8)['sps']:.3f}, {g('S', 32)['sps']:.3f} and {g('S', 64)['sps']:.3f} against 0.154, 0.121 and 0.107 in the first run, between {min(sd['dev']) * 100:+.1f} and {max(sd['dev']) * 100:+.1f} % (limit 30 %; 70 %): true, the constraint did not slow the engine in this setup. (d) No chat of the `KC` cells was lost: all three cells are 100.0 % exact (at most one; 90 %): true. (e) The cells took: `S` {round(g('S', 8)['wall'])}, {round(g('S', 32)['wall'])} and {round(g('S', 64)['wall'])} seconds (the page said 1 to 4 minutes each; the 8-job cell was under a minute), `KC` at 8 jobs {g('KC', 8)['wall'] / 60:.1f} minutes (3), `KC` at 32 jobs {g('KC', 32)['wall'] / 60:.0f} minutes (80 to 90), `KC` at 64 jobs {g('KC', 64)['wall'] / 3600:.2f} hours (3 to 5, "not measured, a guess"): the engine's seconds add up to {F['wall'] / 3600:.2f} hours, below the "roughly 5 to 7 hours of GPU time" of the page, which also had installs and model loads in it; I did not time those.
* **The kept chat's time barely moved from the first run.** `KC` took {g('KC', 8)['sps']:.3f} seconds a step at 8 jobs (first run {r1[('KC', 8)]['sps']:.3f}, {(g('KC', 8)['sps'] / r1[('KC', 8)]['sps'] - 1) * 100:+.1f} %) and {g('KC', 32)['sps']:.3f} at 32 jobs (first run {r1[('KC', 32)]['sps']:.3f}, {(g('KC', 32)['sps'] / r1[('KC', 32)]['sps'] - 1) * 100:+.1f} %). What changed for `KC` is the replies the parser refused: {r1[('KC', 32)]['unparsed']} at 32 jobs in the first run, none now, and the one chat the first run lost at 32 jobs is exact now. {g('KC', 64)['sps']:.3f} seconds a step at 64 jobs is the new number.

## The tables
**Every cell** (the seconds are the wall seconds around the engine's call, summed over the rounds of the cell; "computed" is the prompt tokens the engine did not take from its cache; "cached, 2nd half" is the cached share over the steps from the 16th on; the two prompt columns are the mean prompt of the first 7 and of the last 7 steps of the jobs; "exact" is the share of steps after which the held state equalled the oracle's; engine vLLM {e['version']} on a {e['gpu']}; `KC` at K = 64 was resumed after rounds {F['resumed'][0]} and {F['resumed'][1]}):

| arm | K | steps | seconds | s / step | prompt tokens | computed | cached | cached, 2nd half | prompt, first 7 | prompt, last 7 | longest | exact |
|---|---|---|---|---|---|---|---|---|---|---|---|---|""")
    a("\n" + "\n".join(table_cells(res)) + "\n")
    a("""
**Where the exactness went (counted afterwards, from the step records; none of it was pre-registered).** "Jobs exact to the end" are jobs whose held state equalled the oracle's after every step they did (the first run's number in brackets); "refused" are replies the strict parser refused (the first run's number in brackets); "96 tokens" are replies that used the whole limit; "came back" are jobs whose state differed at some step and equalled the oracle's again at a later one; "corrupted" counts states that were not well formed. Job numbers start at 0.

| arm | K | jobs exact to the end | refused | 96 tokens | jobs that missed | jobs that came back | corrupted |
|---|---|---|---|---|---|---|---|""")
    a("\n" + "\n".join(table_anatomy(res)) + "\n")
    a("""
**The six jobs of `S` that missed, and what they answered at their first miss** (steps count from 1; the oracle's reply is what `oracle_script` writes at that step for a job that is exact up to it; the replies are the same at every K in which the job is run):

| job | step | kind of step | the engine answered | the oracle wrote | in |
|---|---|---|---|---|---|""")
    a("\n" + "\n".join(table_misses(res)) + "\n")
    a("""
**Exact steps per GPU hour** (3,600 times the exact steps over the wall seconds of the cell; "s / step" is the wall seconds over the steps done):

| arm | K | exact steps | seconds | exact steps per hour | exact | s / step |
|---|---|---|---|---|---|---|""")
    a("\n" + "\n".join(table_hour(res)) + "\n")
    a(f"""
Reported, not claimed (line 2 NO, and lines 5 and 6 NO for the time ratio); the exactness of both arms is next to each figure:

| K | exact steps per hour, `S` over `KC` | exact, `S` | exact, `KC` | time per step, `KC` over `S` (not an equal-exactness ratio) |
|---|---|---|---|---|
| 8 | {ratio(8):.2f} | {pc(g('S', 8)['exact'])} | {pc(g('KC', 8)['exact'])} | {tratio(8):.2f} |
| 32 | {ratio(32):.1f} | {pc(g('S', 32)['exact'])} | {pc(g('KC', 32)['exact'])} | {tratio(32):.1f} |
| 64 (`KC` resumed) | {ratio(64):.1f} | {pc(g('S', 64)['exact'])} | {pc(g('KC', 64)['exact'])} | {tratio(64):.1f} |

**`KC` at 64 jobs, round by round** (`active` is the chats still running; seconds per step is the round's seconds over `active`; rounds {F['resumed'][0] + 1} and {F['resumed'][1] + 1} are the first after a resume, with an empty cache):

| round | active | prompt tokens | cached | cached share | round seconds | s / step |
|---|---|---|---|---|---|---|""")
    a("\n" + "\n".join(table_rounds(res)) + "\n")
    a(f"""
## What I read from it (derived afterwards; none of this was pre-registered)
1. **The kept chat's cache gave way at 64 jobs the way it did at 32, only sooner.** At 64 jobs it served {pc(share(kc64[0]), 0)} of the prompt in round 1 and {pc(share(kc64[5]), 0)} in round 6 (round 6 took {kc64[5]['seconds']:.0f} seconds), then {pc(share(kc64[6]), 0)} in round 7 ({kc64[6]['seconds']:.0f} seconds), {pc(share(kc64[7]), 0)} in round 8 and {pc(share(kc64[10]), 0)} in round 11; from round 11 on it served about {num(int(round(sum(r['cached_tokens'] for r in kc64[10:]) / len(kc64[10:]), -2)))} tokens a round, about {sum(r['cached_tokens'] for r in kc64[10:]) / sum(r['active'] for r in kc64[10:]):.0f} per chat, which looks like the shared instruction text and nothing of a chat's own history (an inference: I did not check which tokens they were). At 32 jobs it held up to round 12 ({pc(share(kc32[11]), 0)}) and gave way in round 13 ({pc(share(kc32[12]), 0)}). Over the second half of the steps the cache served {pc(g('KC', 64)['late'])} of the prompt at 64 jobs, over all steps {pc(g('KC', 64)['share'])}. A round of 64 chats took {kc64[5]['seconds']:.0f} seconds when everything fitted and a round of {kc64[29]['active']} chats took {kc64[29]['seconds']:.0f} at round 30. The cache was {num(e['capacity_tokens'])} tokens; 64 chats of the longest prompt seen in that cell ({num(g('KC', 64)['longest'])} tokens) are {num(64 * g('KC', 64)['longest'])} tokens.
2. **At this size the time of a kept chat is the time of the prompt tokens it has to compute again.** `KC` computed {num(g('KC', 32)['computed'])} prompt tokens in {num(round(g('KC', 32)['wall']))} seconds at 32 jobs ({tps(32):.0f} tokens a second, {mspt(32):.2f} milliseconds each) and {num(g('KC', 64)['computed'])} in {num(round(g('KC', 64)['wall']))} at 64 jobs ({tps(64):.0f} a second, {mspt(64):.2f} milliseconds); at 8 jobs, with the cache working, {tps(8):.0f} a second. The replies are short, so with the cache gone the time is almost all prompt work.
3. **Exact-state stayed flat and small, and its time per step fell as the jobs grew in number.** Its mean prompt was {g('S', 8)['q1']:.0f} tokens in the first 7 steps and {g('S', 8)['q4']:.0f} in the last 7 at K = 8, the longest {g('S', 64)['longest']} tokens in any of its cells; 64 jobs of that size need at most {num(64 * g('S', 64)['longest'])} tokens, {64 * g('S', 64)['longest'] / e['capacity_tokens'] * 100:.0f} % of the cache. It served {pc(g('S', 64)['share'], 0)} of its prompt from the cache at 64 jobs. Seconds per step: {g('S', 8)['sps']:.3f}, {g('S', 32)['sps']:.3f}, {g('S', 64)['sps']:.3f}.
4. **Where the exactness went.** The constraint did what the page said it would do to the format: the refused replies of `S` are {g('S', 8)['unparsed']}, {g('S', 32)['unparsed']} and {g('S', 64)['unparsed']} at 8, 32 and 64 jobs (first run {r1[('S', 8)]['unparsed']}, {r1[('S', 32)]['unparsed']}, {r1[('S', 64)]['unparsed']}) and the jobs exact to the end are {g('S', 8)['whole']} of 8, {g('S', 32)['whole']} of 32 and {g('S', 64)['whole']} of 64 (first run {r1[('S', 8)]['whole']}, {r1[('S', 32)]['whole']}, {r1[('S', 64)]['whole']}). No state was ever corrupted: 0 of the {num(F['n'])} steps. **What is left is the content.** Of the six first misses, five are replies that parsed and were wrong (an add written like an edge with a made-up name, a wrong coordinate and value, a wrong coordinate, and a delete written before the add that should have stood alone, twice) and one is the cut-off loop. A grammar makes a reply well formed, not right. The jobs that miss are the same jobs at every K and answer the same way, so the exact share of `S` rising from {pc(g('S', 8)['exact'])} to {pc(g('S', 64)['exact'])} with K is the share of failing jobs among the first 8, 32 and 64 jobs (2 of 8, 5 of 32, 6 of 64) and not an effect of the load.
5. **The kept chat was exact at every size** ({pc(g('KC', 8)['exact'])} at 8, 32 and 64 jobs, no refused reply), at the price in the table: {g('KC', 8)['sps']:.2f}, {g('KC', 32)['sps']:.2f} and {g('KC', 64)['sps']:.2f} seconds a step.
6. **The figures that go with the headline,** each with the exactness of both arms next to it: at 8 jobs `S` does {num(round(g('S', 8)['eph']))} exact steps per GPU hour ({pc(g('S', 8)['exact'])} exact) and `KC` {num(round(g('KC', 8)['eph']))} (100.0 %), {ratio(8):.1f} times; at 32 jobs {num(round(g('S', 32)['eph']))} ({pc(g('S', 32)['exact'])}) against {num(round(g('KC', 32)['eph']))} (100.0 %), {ratio(32):.1f} times; at 64 jobs {num(round(g('S', 64)['eph']))} ({pc(g('S', 64)['exact'])}) against {num(round(g('KC', 64)['eph']))} (100.0 %), {ratio(64):.1f} times. The count is conservative against `S`: a job that is wrong from step 12 on counts as nothing for the rest of its 30 steps.
7. **The resumed rounds.** Round {F['resumed'][0] + 1} took {kc64[F['resumed'][0]]['seconds']:.1f} seconds against {between(F['resumed'][0] + 1):.1f} for the mean of its neighbours ({(kc64[F['resumed'][0]]['seconds'] / between(F['resumed'][0] + 1) - 1) * 100:+.1f} %); round {F['resumed'][1] + 1} took {kc64[F['resumed'][1]]['seconds']:.1f} against {between(F['resumed'][1] + 1):.1f} ({(kc64[F['resumed'][1]]['seconds'] / between(F['resumed'][1] + 1) - 1) * 100:+.1f} %). The cache had already given way by then ({num(kc64[F['resumed'][0] - 1]['cached_tokens'])} cached tokens in the round before the first resume, {num(kc64[F['resumed'][0]]['cached_tokens'])} in the resumed round), so an empty cache at the start of those rounds cost little or nothing that shows.

## What this does and does not show
* **Shows, on one {e['gpu']}, one 7B model with 4-bit AWQ weights, vLLM {e['version']}, a cache of {num(e['capacity_tokens'])} tokens, 30-step jobs and every reply constrained by one regular expression** (a description of this run, with line 2 NO beside it, and not a claim under the page's rule): the kept chat's time per step went from {g('KC', 8)['sps']:.2f} (8 jobs) to {g('KC', 32)['sps']:.2f} (32) and {g('KC', 64)['sps']:.2f} seconds (64), exact-state's from {g('S', 8)['sps']:.2f} to {g('S', 64)['sps']:.2f}; exact-state did {ratio(8):.1f}, {ratio(32):.1f} and {ratio(64):.1f} times as many exact steps per GPU hour as the kept chat at 8, 32 and 64 jobs, while holding the right state on {pc(g('S', 8)['exact'])}, {pc(g('S', 32)['exact'])} and {pc(g('S', 64)['exact'])} of its steps against 100.0 % for the kept chat; and the sensitivity above says the one cut-off reply does not move lines 16 and 17.
* **Does not show that exact-state is as exact as the kept chat** (lines 5 and 6 NO), and so no ratio of times is claimed as a ratio at equal exactness. It does not show another GPU (this one computed about {tps(64):.0f} prompt tokens a second in the kept chat's slow cells; a data-centre GPU is much faster, and the ratios, not the seconds, are what might carry over, which this test does not show), another model, engine or cache size, dollars, **energy (never measured; only time)**, retries (this test sends one request a step and never asks again; a refused reply costs a retry in a real system, for either arm, and the one reply of `S` that was cut off would have cost one), traffic that is not a round of K jobs, jobs longer than 30 steps, or a number of agents per GPU. It does not show that a grammar that bounds the length of a name clears line 2: I have not run it.
* **What may be reported now, and what has to wait.** The figures above may be reported as the measurements of this run with line 2 NO next to them; the rule of the page does not let this Result turn them into a claim. **What waits** is a sentence of the form *with every reply constrained, exact-state did N exact steps per GPU hour against M for a kept chat, at a stated exactness for both*, which the page reserves for a run in which line 2 reads yes. It is not a claim of a speed multiple and not a claim of "as good": the quality of exact-state is lower and is written next to every number.
* **Not done and worth doing, in this order:** (1) the three `S` cells again with the length of a name bounded (clears or confirms line 2; a few minutes of GPU; a new page and a new hash); (2) retries counted, so that a refused reply costs what it costs; (3) the same cells on another GPU; (4) the GPU's power read next to the seconds.

## Reproduce
In Colab, `python/colab/axi_provider_side_constrained.ipynb` with a T4; about {F['wall'] / 3600:.1f} hours of engine time for the six cells, plus the installs and model loads of each session (the 64-job kept chat took three). Nothing needs the GPU after that: `provider_side_constrained.merge_files` puts cell files together on any machine. `tests/test_provider_side_constrained_result.py` recomputes every figure of this Result from the cell files, checks that the merged file is what `merge_cells` makes of them, and that the page is the locked text followed by this Result.
""")
    return "".join(t)


# ================================================================ the tests
RES1_SHA256 = "19ce96633931980292f2ab6b25c42799b50696db2ea25225f194e02c8ca93d48"
TOTALS = {("S", 8): (89589, 60112, 2526, 212, 240, 516), ("KC", 8): (831529, 784384, 2510, 240, 240, 7154), ("S", 32): (345237, 235504, 10255, 908, 960, 516),          # prompt, cached, new tokens, exact steps, steps, longest prompt
          ("KC", 32): (3051176, 804768, 10105, 956, 956, 7556), ("S", 64): (670785, 462128, 20354, 1867, 1920, 516), ("KC", 64): (5840418, 686256, 20172, 1915, 1915, 7556)}
QUOTES = ["the constraint removed the format slips and left the content errors. Exact-state holds less of the right state than the kept chat at small K, by the amount measured, and still buys many more correct steps per GPU hour than the kept chat at every K. The article gets the exact-steps-per-hour figures with the exactness gap next to them, and not a plain time ratio.",
          "the constraint did not hold. Exactness and times are not read until the cause is found", "Line 2 reads NO: neither is claimed.", "a reply is cut by the 96-token limit (the longest in the first run was 31)", "cannot be made into the node add the model meant", "stay wrong",
          "if the rest stays exact", "once a job is behind it can come back", "about 3.5 against a limit of 2", "about 40 against a limit of 10", "more than at 32", "between 4 and 12 percent of its steps", "equally exact only at 64 jobs, if at all",
          "not measured, a guess", "roughly 5 to 7 hours of GPU time"]


def _norm(t): return re.sub(r"\s+", " ", t)


# ---------------------------------------------------------------- 1 the files
def test_the_files_are_the_ones_the_page_is_about_the_merge_is_what_the_code_makes_of_them_and_the_numbers_add_up():
    res = run(); assert _sha(RES) == FILE_SHA256, "the saved result file must not change"; assert _sha(RUN1) == RES1_SHA256, "the first run's file must not change"
    assert _sha(MODULE) == MODULE_SHA256 == res["module_sha256"], "the module that merged the cells must be the one in the repository"; assert _sha(PS_MODULE) == PS_SHA256 == res["provider_side_sha256"]
    assert set(res) == {"meta", "cells", "module_sha256", "provider_side_sha256"} and C.REGEX_SHA256 == REGEX_SHA256 == hashlib.sha256(C.REGEX.encode()).hexdigest()
    names = sorted(os.listdir(CELLDIR)); assert names == sorted(f"axi_provider_c_{k}.json" for k in CELL_SHA256), names
    src = []
    for k, sha in CELL_SHA256.items():
        p = os.path.join(CELLDIR, f"axi_provider_c_{k}.json"); assert _sha(p) == sha, k; src.append(json.load(open(p)))
    merged = C.merge_cells(src); assert merged == {k: v for k, v in res.items() if k not in ("module_sha256", "provider_side_sha256")}
    e = res["meta"]["engine"]; con = e.pop("constraint")
    try:
        assert e == dict(kind="vllm", name="Qwen/Qwen2.5-7B-Instruct-AWQ", version="0.31.0", gpu="Tesla T4", capacity_tokens=80000, capacity_source="override", block_size=16, kv_bytes_per_token=57344, model="Qwen/Qwen2.5-7B-Instruct-AWQ", python="3.13.15")
        assert con["kind"] == "regex" and con["regex"] == C.REGEX and con["regex_sha256"] == REGEX_SHA256
    finally: e["constraint"] = con
    assert (res["meta"]["steps"], res["meta"]["seed"], res["meta"]["policy"], res["meta"]["hint_rules"], res["meta"]["levels"], res["meta"]["arms"]) == (30, SEED, "repair_m6", False, [8, 32, 64], ["S", "KC"])
    assert [(c["arm"], c["K"]) for c in res["cells"]] == ORDER_MERGED and all(c["engine"] == e and c["steps"] == 30 and c["seed"] == SEED for c in res["cells"])
    assert [c["resumed_at"] for c in res["cells"] if (c["arm"], c["K"]) != ("KC", 64)] == [[]] * 5 and cells(res)[("KC", 64)]["resumed_at"] == [12, 21]
    scripts = {}
    for c in res["cells"]:
        assert [s["seed"] for s in c["sessions"]] == [SEED * 1000 + i for i in range(c["K"])] and all(s["script_steps"] == 30 for s in c["sessions"])
        for s in c["sessions"]:
            kinds = [x["kind"] for x in s["log"]]; assert [x["t"] for x in s["log"]] == list(range(1, len(kinds) + 1)); full = scripts.setdefault(s["seed"], kinds)
            assert kinds == full[:len(kinds)], (c["arm"], c["K"], s["seed"])                                                    # the same job script in every cell, at every K
            assert (s["stopped"] is not None) == (len(kinds) < 30) and s["stopped"] in (None, "context")
        L = logs(c); assert abs(sum(r["seconds"] for r in c["rounds"]) - c["wall_seconds"]) < 1e-6 and len(c["rounds"]) == 30
        assert sum(r["prompt_tokens"] for r in c["rounds"]) == sum(x["n_prompt"] for x in L) and sum(r["cached_tokens"] for r in c["rounds"]) == sum(x["n_cached"] for x in L) and sum(r["new_tokens"] for r in c["rounds"]) == sum(x["n_new"] for x in L)
        for i, r in enumerate(c["rounds"], 1): assert r["round"] == i and r["active"] == sum(1 for s in c["sessions"] if len(s["log"]) >= i)
        assert all(0 <= x["n_cached"] <= x["n_prompt"] and isinstance(x["n_cached"], int) and x["n_new"] > 0 for x in L)
        assert all(bool(x["exact"]) == (x["distance"] == 0) for x in L) and all(x["parsed"] or not x["admitted"] for x in L)
    assert len(scripts) == 64 and sorted(scripts) == [SEED * 1000 + i for i in range(64)]


ORDER_MERGED = [("S", 8), ("KC", 8), ("S", 32), ("S", 64), ("KC", 32), ("KC", 64)]          # the order merge_cells puts the cells in (cells_planned)


# ---------------------------------------------------------------- 2 the rules
def test_the_seventeen_rules_read_as_the_page_says_the_headline_is_read_and_the_rule_about_claims_is_followed():
    res = run(); S = C.summarize_all(res["cells"]); v = C.verdict_constrained(S); assert [x for _, _, _, x in v] == READ and len(PROBS) == 17 and C.CL_HEADLINE == (15, 17) and len(C.CL_LINES) == 17
    assert [(i, tag, text) for i, tag, text, _ in v] == [(i + 1, C.CL_LINES[i][0], C.CL_LINES[i][1]) for i in range(17)]
    assert sum(1 for x in READ if x is True) == 14 and [i + 1 for i, x in enumerate(READ) if x is False] == [2, 5, 6] and [READ[i - 1] for i in C.CL_HEADLINE] == [True, True]
    # the same rules from the raw numbers, by hand
    s = S_(res); g = lambda a, k: s[(a, k)]; allc = list(s.values())
    hand = [all(x["n_cached"] is not None for c in res["cells"] for x in logs(c)),
            all(v_["unparsed"] / v_["n"] <= 0.005 for v_ in allc) and all(v_["full96"] == 0 for v_ in allc),
            64 * g("KC", 64)["longest"] >= 2 * 80000 and 8 * g("KC", 8)["longest"] <= 80000,
            all(v_["exact"] >= 0.80 for v_ in allc),
            all(g("S", k)["exact"] >= g("KC", k)["exact"] - 0.05 for k in (8, 32, 64)),
            all(g("S", k)["exact"] >= 0.98 for k in (8, 32, 64)),
            g("S", 8)["q4"] / g("S", 8)["q1"] <= 1.3, g("KC", 8)["q4"] / g("KC", 8)["q1"] >= 3.5, g("KC", 8)["share"] >= 0.80,
            g("S", 8)["cps"] <= 0.7 * g("KC", 8)["cps"], g("S", 8)["sps"] <= 0.8 * g("KC", 8)["sps"],
            g("KC", 32)["sps"] >= 1.3 * g("S", 32)["sps"], g("KC", 64)["late"] < 0.5, g("KC", 64)["sps"] >= 2 * g("S", 64)["sps"],
            g("S", 8)["eph"] >= 2 * g("KC", 8)["eph"], g("S", 32)["eph"] >= 10 * g("KC", 32)["eph"], g("S", 64)["eph"] >= 10 * g("KC", 64)["eph"]]
    for i in range(17): assert hand[i] == READ[i], (i + 1, hand[i], READ[i])
    assert len(s) == 6
    # the rule that decides what the Result may claim: line 5 NO -> no time ratio at equal exactness; line 2 NO -> neither claimed
    ca = C.claims_allowed(v); assert ca == dict(plain_time_at_equal_exactness=False, quality_adjusted={15: True, 16: True, 17: True}) and READ[1] is False
    R = _text()[_text().index(RESULT_MARK):]; assert "**this Result claims neither the plain time ratios nor the exact-steps-per-hour figures.**" in R
    # the probabilities on the page are the ones this file uses
    assert [int(m.group(1)) for m in re.finditer(r"\n\d+\. \[\w+\] .*\((\d+) %\)(?=\n)", _head())] == PROBS


# ---------------------------------------------------------------- 3 the page
def test_the_part_above_the_result_is_what_was_written_before_the_run():
    head = _head(); assert _h(head) == PAGE_SHA256 and _text().count(RESULT_MARK) == 1 and len(head) in (23746, 23747)


def test_the_page_carries_exactly_the_result_this_file_renders_and_its_table_says_what_the_rules_say():
    res = run(); text = _text(); assert text == _head().rstrip("\n") + render(res).rstrip("\n") + "\n"
    rows = [ln for ln in text[text.index(RESULT_MARK):].split("\n") if re.match(r"\| \d+ \| \d+ % \|", ln)]; assert len(rows) == 17
    for i, ln in enumerate(rows, 1):
        cells_ = [c.strip() for c in ln.strip("|").split("|")]; assert int(cells_[0]) == i and cells_[1] == f"{PROBS[i - 1]} %" and cells_[2] == measured(res, i) and cells_[3] == ("yes" if READ[i - 1] else "NO"), i
    R = text[text.index(RESULT_MARK):]
    for tbl in (table_cells(res), table_anatomy(res), table_misses(res), table_hour(res), table_rounds(res)):
        for ln in tbl: assert ln in R.split("\n"), ln
        assert "\n".join(tbl) in R


# ---------------------------------------------------------------- 4 the figures
def test_the_figures_quoted_in_the_result_are_the_ones_in_the_files():
    res = run(); R = _text()[_text().index(RESULT_MARK):]
    for needle in ("vLLM 0.31.0", "GPU Tesla T4", "Python 3.13.15", FILE_SHA256, MODULE_SHA256, PS_SHA256, REGEX_SHA256[:16], "resumed after rounds 12 and 21", "add up to 15,562, which is 259 minutes, 10,331 of them in `KC` at K = 64",
                   "6,231 steps in the 6 cells that ran", "7,556 tokens, 64 times that is 483,584, 6.0 times the cache of 80,000", "7,154, 8 times that is 57,232", "88.3 %", "94.6 %", "97.2 %", "20,878", "28,748", "32,896", "5,923", "727", "667", "3.52 times", "39.5 times", "49.3 times",
                   "3.99", "41.8", "50.7", "0.152", "0.608", "0.118", "4.949", "0.106", "5.395", "74.9 % less", "37.5 % fewer", "5.36 times", "94.3 %", "4.6 % of the prompt tokens of the second half", "11.8 % over all steps", "2.87 hours", "79 minutes",
                   "956 steps in the 32-job cell instead of 960 and 1,915 in the 64-job cell instead of 1,920", "job 27 after 28 steps, job 31 after 28 steps, job 46 after 29 steps", "28,780", "32,914", "29,350", "33,231", "40.3", "49.8", "96.6 %", "98.2 %", "19 steps that job 19 got wrong",
                   "(first run 6, 16, 58)", "(first run 3, 18, 27)", "0 of the 6,231 steps", "475 tokens a second", "499 a second", "323 a second", "job 17 at step 24, job 62 at step 13", "+3.9 %", "-2.6 %", "4.32 hours", "between -1.9 and -0.4 %", "-1.6 %", "-3.9 %", "(first run 0.617", "(first run 5.149"):
        assert needle in R, needle
    # recomputed here from the raw records with plain arithmetic
    c = cells(res)
    for k, (prompt, cached, new, ex, steps, longest) in TOTALS.items():
        L = logs(c[k]); assert (sum(x["n_prompt"] for x in L), sum(x["n_cached"] for x in L), sum(x["n_new"] for x in L), sum(1 for x in L if x["exact"]), len(L), max(x["n_prompt"] for x in L)) == (prompt, cached, new, ex, steps, longest), k
    wall = {k: sum(r["seconds"] for r in c[k]["rounds"]) for k in TOTALS}; assert [round(wall[k], 1) for k in ORDER] == [36.6, 145.9, 113.7, 4730.9, 204.3, 10330.5] and round(sum(wall.values())) == 15562
    assert [round(3600 * TOTALS[k][3] / wall[k]) for k in ORDER] == [20878, 5923, 28748, 727, 32896, 667]
    assert [f"{TOTALS[k][3] / TOTALS[k][4] * 100:.1f}" for k in ORDER] == ["88.3", "100.0", "94.6", "100.0", "97.2", "100.0"]
    rt = lambda k: (3600 * TOTALS[("S", k)][3] / wall[("S", k)]) / (3600 * TOTALS[("KC", k)][3] / wall[("KC", k)]); assert [f"{rt(k):.2f}" for k in (8, 32, 64)] == ["3.52", "39.52", "49.29"]
    assert sum(len(logs(c[k])) for k in ORDER) == 6231 and sum(x["corrupted"] for k in ORDER for x in logs(c[k])) == 0
    # the sensitivity: one more exact step, and all the 19 steps job 19 got wrong
    for k in (32, 64):
        j19 = [x["exact"] for x in c[("S", k)]["sessions"][19]["log"]]; assert j19 == [True] * 11 + [False] * 19
        e_ = TOTALS[("S", k)][3]; assert round(3600 * (e_ + 1) / wall[("S", k)]) == {32: 28780, 64: 32914}[k] and round(3600 * (e_ + 19) / wall[("S", k)]) == {32: 29350, 64: 33231}[k]
    assert round((3600 * (908 + 19) / wall[("S", 32)]) / (3600 * 956 / wall[("KC", 32)]), 1) == 40.3
    assert round((3600 * (1867 + 19) / wall[("S", 64)]) / (3600 * 1915 / wall[("KC", 64)]), 1) == 49.8 and f"{(908 + 19) / 960 * 100:.1f}" == "96.6" and f"{(1867 + 19) / 1920 * 100:.1f}" == "98.2"


# ---------------------------------------------------------------- 5 the part written afterwards
def test_where_the_exactness_went_is_counted_from_the_step_records():
    res = run(); c = cells(res); r1 = cells(run1()); s = S_(res); s1 = S_(run1())
    assert [s[k]["whole"] for k in ORDER] == [6, 8, 27, 32, 58, 64] and [s[k]["unparsed"] for k in ORDER] == [0, 0, 1, 0, 1, 0] and [s[k]["full96"] for k in ORDER] == [0, 0, 1, 0, 1, 0]
    assert [s1[k]["whole"] for k in (("S", 8), ("KC", 8), ("S", 32), ("KC", 32), ("S", 64))] == [3, 8, 18, 31, 27] and [s1[k]["unparsed"] for k in (("S", 8), ("S", 32), ("S", 64), ("KC", 32))] == [6, 16, 58, 9]
    miss = {k: missed_jobs(c[("S", k)]) for k in (8, 32, 64)}; assert miss == FIRST_MISS_JOBS and all(not missed_jobs(c[("KC", k)]) for k in (8, 32, 64))
    assert set(miss[8]) < set(miss[32]) < set(miss[64])                                                                      # nested: the jobs that miss are the same jobs at every K
    # the jobs of the first 8 give the same replies and the same prompt sizes at every K, in both arms
    for arm in ("S", "KC"):
        for i in range(8):
            for k in (32, 64):
                a_, b_ = c[(arm, 8)]["sessions"][i]["log"], c[(arm, k)]["sessions"][i]["log"]; assert [(x["text"], x["n_prompt"]) for x in a_] == [(x["text"], x["n_prompt"]) for x in b_], (arm, i, k)
        for i in range(8, 32): assert [(x["text"], x["n_prompt"]) for x in c[(arm, 32)]["sessions"][i]["log"]] == [(x["text"], x["n_prompt"]) for x in c[(arm, 64)]["sessions"][i]["log"]], (arm, i)
    # the six first misses: step, what the engine answered, what the oracle writes, the page's step for them
    expect = {3: (16, "ADD[(4,4)->(4,4):value#6]", "ADD[4,4:6]"), 4: (18, "ADD[1,0:6]", "ADD[0,0:3]"), 17: (20, "DEL[(2,3)->(0,4):dep#6] ADD[(0,4)->(2,3):dep#4]", "ADD[(0,4)->(2,3):dep#4]"),
              20: (30, "ADD[(1,5)->(2,1):dep#3]", "ADD[(1,1)->(2,1):dep#3]"), 62: (12, "DEL[(1,1)->(1,0):owns#5] ADD[(1,1)->(1,0):dep#9]", "ADD[(1,1)->(1,0):dep#9]")}
    for job, (step, got, want) in expect.items():
        f = first_miss(c[("S", 64)], job); assert (f["t"], f["text"], f["parsed"]) == (step, got, True) and oracle_replies(job)[step - 1][1] == want and f["admitted"] == (job in (17, 62)), job
        assert all(x["exact"] for x in c[("S", 64)]["sessions"][job]["log"][:step - 1])
    f19 = first_miss(c[("S", 64)], 19); assert (f19["t"], f19["parsed"], f19["n_new"], f19["kind"]) == (12, False, 96, "add_node") and oracle_replies(19)[11][1] == "ADD[3,0:4]"
    assert [first_miss(c[("S", k)], 19)["text"] for k in (32, 64)][0] == f19["text"]
    body = f19["text"][len("ADD[(3,0)->(3,0):"):]; assert f19["text"].startswith("ADD[(3,0)->(3,0):") and body.startswith("4tworowsdown4colsright" * 3) and re.fullmatch(r"(?:4tworowsdown4colsright)*4?(?:tworowsdown4colsright)?[a-z0-9]*", body) and len(body) > 200
    assert re.fullmatch(C.REGEX, f19["text"] + "#0]") is not None and re.fullmatch(C.REGEX, f19["text"]) is None                  # every prefix of the loop is allowed; it just never closes
    for tail in range(len("ADD[(3,0)->(3,0):") + 1, len(f19["text"]), 7): assert re.fullmatch(C.REGEX, f19["text"][:tail] + "#0]") is not None
    # the first run's version of the two replies the page names
    o3 = first_miss(cells(run1())[("S", 64)], 3); o19 = first_miss(cells(run1())[("S", 64)], 19); assert (o3["text"], o19["text"]) == ("ADD[(4,4):6]", "ADD[(3,0):4]")
    # no reply that parsed is longer than 31 tokens; the kept chat's longest is 16
    assert [s[k]["longest_ok"] for k in ORDER] == [16, 16, 31, 16, 31, 16]
    # jobs that came back, and when
    assert {k: back_steps(c[("S", k)]) for k in (8, 32, 64)} == {8: {}, 32: {17: 24}, 64: {17: 24, 62: 13}}
    # the kept chats that stopped
    assert kc_stopped(res) == {32: [(27, 28, "context"), (31, 28, "context")], 64: [(27, 28, "context"), (31, 28, "context"), (46, 29, "context")]}
    # side predictions
    sd = side(res); assert (sd["slip_n"], sd["slip_hit"], sd["job4"], sd["job3_misses_at_16"], sd["job3_exact_steps"], sd["kc_lost"]) == (31, 31, "ADD[1,0:6]", True, 15, 0) and max(abs(d) for d in sd["dev"]) < 0.30 and min(sd["dev"]) > -0.025


def test_the_kept_chats_rounds_say_where_its_cache_gave_way_and_what_the_resumed_rounds_cost():
    res = run(); c = cells(res); r64 = c[("KC", 64)]["rounds"]; r32 = c[("KC", 32)]["rounds"]; sh = lambda r: r["cached_tokens"] / r["prompt_tokens"]
    assert [r["active"] for r in r64] == [64] * 28 + [62, 61] and [r["active"] for r in r32] == [32] * 28 + [30, 30]
    # at 64 jobs the cache holds through round 6 and falls from round 7; at 32 jobs it holds through round 12 and falls in round 13
    assert all(sh(r64[i]) > 0.85 for i in (5,)) and sh(r64[6]) < 0.62 and sh(r64[7]) < 0.4 and sh(r64[10]) < 0.1 and all(sh(r64[i]) > sh(r64[i + 1]) for i in range(5, 29))
    assert sh(r32[11]) > 0.92 and sh(r32[12]) < 0.77 and sh(r32[13]) < 0.5 and all(sh(r32[i]) < sh(r32[i - 1]) for i in range(12, 30))
    assert [round(r64[i]["seconds"], 1) for i in (5, 6, 7, 11, 12, 20, 21, 22, 29)] == [20.3, 55.1, 95.9, 224.6, 273.6, 519.3, 523.5, 555.9, 732.2]
    assert all(11000 < r["cached_tokens"] < 13000 for r in r64[10:]) and 190 < sum(r["cached_tokens"] for r in r64[10:]) / sum(r["active"] for r in r64[10:]) < 205
    assert abs(r64[12]["seconds"] / ((r64[11]["seconds"] + r64[13]["seconds"]) / 2) - 1) < 0.04 and abs(r64[21]["seconds"] / ((r64[20]["seconds"] + r64[22]["seconds"]) / 2) - 1) < 0.03
    # the time of the kept chat tracks the prompt tokens it computes: tokens a second 475 and 499 in the two big cells
    comp = lambda k: sum(x["n_prompt"] - x["n_cached"] for x in logs(c[("KC", k)])); w = lambda k: sum(r["seconds"] for r in c[("KC", k)]["rounds"])
    assert (round(comp(32) / w(32)), round(comp(64) / w(64)), round(comp(8) / w(8))) == (475, 499, 323) and (comp(32), comp(64)) == (2246408, 5154162)
    # exact-state stays flat and small: 64 jobs of its longest prompt are 41 % of the cache
    assert max(x["n_prompt"] for k in (8, 32, 64) for x in logs(c[("S", k)])) == 516 and round(64 * 516 / 80000 * 100) == 41


# ---------------------------------------------------------------- 6 what the page said it would say
def test_the_result_says_what_the_page_said_it_would_say_when_lines_2_5_and_6_read_no_and_claims_nothing_more():
    head, R = _head(), _text()[_text().index(RESULT_MARK):]; hn = _norm(head)
    for q in QUOTES: assert _norm(q) in hn, q                                                                                # every sentence the Result quotes from the page is on the page
    for q in QUOTES[:4] + QUOTES[5:6]: assert _norm(q) in _norm(R), q
    assert "*Line 2 NO:* the constraint did not hold. Exactness and times are not read until the cause is found." in head and "Line 2 reads NO: neither is claimed." in head
    for needle in ("**this Result claims neither the plain time ratios nor the exact-steps-per-hour figures.**", "**Line 2 is a NO I did not expect (94 %).**", "**14 of the 17 lines read yes, lines 2, 5 and 6 read NO, and no line is \"not run\".**",
                   "and are **not** claimed as ratios at equal exactness: line 5 reads NO.", "What waits", "not a claim of a speed multiple", "**Does not show that exact-state is as exact as the kept chat**", "energy (never measured; only time)",
                   "retries (this test sends one request a step and never asks again", "It does not show that a grammar that bounds the length of a name clears line 2: I have not run it.", "That is a new regular expression and so a new page with new predictions; I have not made it.",
                   "Reported, not claimed", "not an equal-exactness ratio", "(a description of this run, with line 2 NO beside it, and not a claim under the page's rule)", "none of it was pre-registered", "(an inference: I did not check which tokens they were)"):
        assert needle in R, needle
    # the sentences that carry a reading, a rule or a limit, each with the words that follow it (a changed sentence or an added word breaks the match)
    for sent, tail in (("line 15 at 88 %) all read yes", ".\n\n**What the page said"), ("line 2 is a third NO", ". The three lines"), ("The next section is about that.", "\n\n## Line 2"), ("with **no limit on how long**", ") became"), ("so the constraint held as designed", "; the reply was cut"),
                       ("Every prefix of that reply is one the grammar allows", ", so the constraint held"), ("Job 19 held the right state on 11 of its 30 steps (steps 1 to 11) and on none after", ". The same reply appears in the 32-job and the 64-job cell, as it must with greedy decoding. Every other"),
                       ("and I do not take it as lifting it", ". So: (1)"), ("**this Result claims neither the plain time ratios nor the exact-steps-per-hour figures.**", " Both are reported"), ("(an upper bound)", " would give 29,350"),
                       ("lines 16 and 17 (limit 10) would still read yes", ", and the exact share"), ("still short of `KC`", ". (4) The clean way"), ("the `KC` cells would not change", ", since no reply of theirs"), ("the 8-job figure is hit to the digit", " (88.3 %), and the other two came out"),
                       ("which the page had allowed for", " (\"once a job is behind it can come back\")"), ("under it at 64. Line 5 needs every K, so it reads NO", ". (The page expected"), ("(the 64-job figure is below the 4)", ". It did not come near"),
                       ("true, the constraint did not slow the engine in this setup", ". (d) No chat of the `KC` cells was lost: all three"), ("which also had installs and model loads in it; I did not time those", ".\n* **The kept chat's time barely moved"),
                       ("What changed for `KC` is the replies the parser refused", ": 9 at 32 jobs"), ("not an equal-exactness ratio", ") |"), ("none of this was pre-registered", ")\n1. **The kept chat's cache"), ("What waits", "** is a sentence"), ("not a claim of a speed multiple", " and not a claim"),
                       ("energy (never measured; only time)", "**, retries"), ("A grammar makes a reply well formed, not right", ". The jobs that miss"), ("What is left is the content", ".** Of the six first misses"), ("for the rest of its 30 steps", ".\n7. **The resumed rounds.**"),
                       ("so an empty cache at the start of those rounds cost little or nothing that shows", ".\n\n## What this does"), ("the sensitivity above says the one cut-off reply does not move lines 16 and 17", ".\n* **Does not show"),
                       ("ended before step 30 by the test's own rule", "** (the page's jobs row"), ("The steps it did were all exact", ".\n\n## The 17 lines"), ("which by the page means that the probe and every reply of the safety cell parsed", "; the probe (step 20 of job 0)"),
                       ("the other five are the files from the Drive folder", ". The module that ran the cells")):
        assert sent + tail in R, sent
    for banned in ("proves", "proven", "wins", "massive", "hyperscaler", "dollars saved", "43 times", "$", "x faster", "times faster", "cheaper"): assert banned not in R.lower(), banned
    assert R.count("dollars") == 1 and "other GPU" not in R.replace("another GPU", "")
    # every figure of the time-per-step ratio that the Result prints has the exactness of both arms next to it
    for ln in R.split("\n"):
        if re.match(r"\| (11|12|14) \| \d+ % \|", ln) or "times as long" in ln: assert "exact steps:" in ln, ln
    # the page's own pre-registered text stays as it was: the Result adds, it does not edit
    assert "Built and pre-registered" not in R


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("PASS", n)
