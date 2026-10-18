"""Implements concept-map item 106: RESULT of the provider-side serving test of 17 and 18 Oct 2026. Page: docs/PROVIDER_SIDE_TEST.md (predictions and reading rules written before the run; the result is appended below them).
The files: docs/results/provider_side_cells/ (the six cell files the owner's Colab run saved: S at K = 8, 32 and 64, KC at K = 8 and 32, KN at K = 8; Tesla T4, vLLM 0.31.0, Qwen2.5-7B-Instruct-AWQ, a cache of 80,000 tokens)
and docs/results/axi_provider_side.json (the six put together by provider_side.merge_cells, with the sha256 of the module that did it).
This file recomputes from the saved files, with its own arithmetic and not with the code that printed the figures:
  1. the files: their sha256, the merge (the merged file is what merge_cells makes of the six, plus the module's sha256, and the module on disk is that module), the settings, the seeds, the same job script in every cell, the rounds adding up to the cell, every cached count at most its prompt;
  2. the thirteen reading rules (8 yes, lines 3 and 4 NO, lines 2, 11 and 12 not run), the headline (line 10 yes, line 12 not run), by the code and by hand from the raw numbers;
  3. the 'what was measured' text of each line of the page, the tables and the figures quoted in the prose of the Result (the whole Result is `render(result)`, and the page must equal the locked head followed by it);
  4. that the part of the page above the Result is byte for byte what was written before the run;
  5. the counts of the part written afterwards (where the exactness went: every first miss of every job, unparsed replies by kind) from the step records;
  6. that the Result says what the page said it would say when lines 3 and 4 read NO.
What it does NOT show: anything about the kept chat at 64 jobs, a random order, another GPU, model, engine or cache size, energy, dollars, or replies that are constrained by a grammar."""
import collections, hashlib, json, os, re, sys

sys.path.insert(0, os.path.dirname(__file__))
import axi.experiments.provider_side as P

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..", "..")
PAGE = os.path.join(ROOT, "docs", "PROVIDER_SIDE_TEST.md")
RES = os.path.join(ROOT, "docs", "results", "axi_provider_side.json")
CELLDIR = os.path.join(ROOT, "docs", "results", "provider_side_cells")
MODULE = os.path.join(HERE, "..", "axi", "experiments", "provider_side.py")
LONGPAGE = os.path.join(ROOT, "docs", "LONG_HORIZON_TEST.md")
RESULT_MARK = "\n\n---\n\n# Result"
PAGE_SHA256 = "d37805796784b1a2c1925e7d764c017e313fa9fdfefeb4027217c70510e9a614"
FILE_SHA256 = "19ce96633931980292f2ab6b25c42799b50696db2ea25225f194e02c8ca93d48"
MODULE_SHA256 = "bfb4734e171d9244cd47d10402be46528a557095a8275d02b38481a0e2b0b1d3"
CELL_SHA256 = {"KC_K32": "69a88a7727e800aefb3d08ff0b39ea8d18ba9a7aac55e42811ef897a2a3e5885", "KC_K8": "afb8d3921cf789e256f76522bce6f0b6e68075482a658559583e07f37f46b529", "KN_K8": "a1ed5fa856b4228b1bc602404156bbb72b908db7b8a531d4ea020d6788b5bbe0",
               "S_K32": "08e8005668a4be2d8eeff66d743eb09c41718853f8e2dd51c01d8ee477cc31c8", "S_K64": "bb8c9733d4f9843bd67554f40e6b829634bd991ca937e786453c125f678bca82", "S_K8": "dcbcbe2a01d81f770d8f3bb54c4fd8f87f03ad7783b01641827cb12b78465679"}
PROBS = [85, 90, 85, 85, 95, 90, 88, 95, 50, 40, 88, 92, 90]
READ = [True, None, False, False, True, True, True, True, True, True, None, None, True]            # None = not run
ORDER = [("S", 8), ("KC", 8), ("KN", 8), ("S", 32), ("KC", 32), ("S", 64)]
_cache = {}


def run():
    if "run" not in _cache: _cache["run"] = json.load(open(RES))
    return _cache["run"]


def _text(): return open(PAGE).read()


def _head():
    t = _text(); assert t.count(RESULT_MARK) == 1; return t[:t.index(RESULT_MARK)]


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
    late = [x for s in c["sessions"] for x in s["log"][15:]]
    return dict(n=n, wall=wall, sps=wall / n, prompt=prompt, cached=cached, computed=prompt - cached, cps=(prompt - cached) / n, share=cached / prompt,
                late=sum(x["n_cached"] for x in late) / sum(x["n_prompt"] for x in late),
                q1=mean(x["n_prompt"] for s in c["sessions"] for x in s["log"][0:7]), q4=mean(x["n_prompt"] for s in c["sessions"] for x in s["log"][23:30]),
                longest=max(x["n_prompt"] for x in L), exact=sum(x["exact"] for x in L) / n, sph=3600 * n / wall, stopped=sum(1 for s in c["sessions"] if s["stopped"]),
                whole=sum(all(x["exact"] for x in s["log"]) for s in c["sessions"]), jobs=len(c["sessions"]), corrupted=sum(x["corrupted"] for x in L), new=sum(x["n_new"] for x in L))


def S_(res): return {k: st(c) for k, c in cells(res).items()}


DELETES = ("delete_dep", "delete_leaf")


def anatomy(c):
    L = logs(c); un = [x for x in L if not x["parsed"]]
    firsts = [f for f in (next((x for x in s["log"] if not x["exact"]), None) for s in c["sessions"]) if f]
    a = sum(1 for f in firsts if not f["parsed"]); b = sum(1 for f in firsts if f["parsed"] and not f["admitted"]); o = sum(1 for f in firsts if f["parsed"] and f["admitted"])
    byk = collections.Counter(x["kind"] for x in L); unk = collections.Counter(x["kind"] for x in un)
    return dict(unparsed=len(un), unparsed_del=sum(1 for x in un if x["kind"] in DELETES), a=a, b=b, o=o, firsts=len(firsts), byk=byk, unk=unk)


def recovered(c):
    """jobs whose state was inexact at some step and exact again at a later one"""
    n = 0
    for sess in c["sessions"]:
        ex = [x["exact"] for x in sess["log"]]
        if False in ex and any(ex[ex.index(False):]): n += 1
    return n


def measured(res, i):
    s = S_(res); n = sum(v["n"] for v in s.values())
    return {
        1: f"the engine reported the cached tokens of every request: {num(n)} steps in the {len(s)} cells that ran",
        2: f"not run: it needs the 64-job kept chat. What the cells show, and not the line: the longest kept-chat prompt at K = 32 was {num(s[('KC', 32)]['longest'])} tokens, 32 times that is {num(32 * s[('KC', 32)]['longest'])}, {32 * s[('KC', 32)]['longest'] / 80000:.1f} times the cache of 80,000; at K = 8 it was {num(s[('KC', 8)]['longest'])}, 8 times that is {num(8 * s[('KC', 8)]['longest'])}, which fits",
        3: f"exact steps, `S` {pc(s[('S', 8)]['exact'])} (K = 8), {pc(s[('S', 32)]['exact'])} (K = 32), {pc(s[('S', 64)]['exact'])} (K = 64); `KC` {pc(s[('KC', 8)]['exact'])} (8), {pc(s[('KC', 32)]['exact'])} (32); `KN` {pc(s[('KN', 8)]['exact'])} (8): three of the six cells are below 90 %",
        4: f"`S` against `KC`: K = 8 {pc(s[('S', 8)]['exact'])} against {pc(s[('KC', 8)]['exact'])}, {(s[('KC', 8)]['exact'] - s[('S', 8)]['exact']) * 100:.1f} points short; K = 32 {pc(s[('S', 32)]['exact'])} against {pc(s[('KC', 32)]['exact'])}, {(s[('KC', 32)]['exact'] - s[('S', 32)]['exact']) * 100:.1f} points short (limit 5; K = 64 has no `KC`)",
        5: f"`S` at K = 8: mean prompt {s[('S', 8)]['q1']:.0f} tokens in the first 7 steps, {s[('S', 8)]['q4']:.0f} in the last 7: {s[('S', 8)]['q4'] / s[('S', 8)]['q1']:.2f} times (limit 1.3)",
        6: f"`KC` at K = 8: mean prompt {num(round(s[('KC', 8)]['q1']))} tokens in the first 7 steps, {num(round(s[('KC', 8)]['q4']))} in the last 7: {s[('KC', 8)]['q4'] / s[('KC', 8)]['q1']:.2f} times (limit 3.5)",
        7: f"{pc(s[('KC', 8)]['share'])} of the prompt tokens of `KC` at K = 8 came from the cache (limit 80 %)",
        8: f"`S` computes {s[('S', 8)]['cps']:.0f} prompt tokens a step, `KN` {num(round(s[('KN', 8)]['cps']))}: {(1 - s[('S', 8)]['cps'] / s[('KN', 8)]['cps']) * 100:.1f} % fewer (limit 80 %)",
        9: f"`S` computes {s[('S', 8)]['cps']:.0f} tokens a step, `KC` {s[('KC', 8)]['cps']:.0f}: {(1 - s[('S', 8)]['cps'] / s[('KC', 8)]['cps']) * 100:.1f} % fewer (limit 30 %)",
        10: f"`S` {s[('S', 8)]['sps']:.3f} seconds a step, `KC` {s[('KC', 8)]['sps']:.3f}: {(1 - s[('S', 8)]['sps'] / s[('KC', 8)]['sps']) * 100:.1f} % less (limit 20 %); `KC` took {s[('KC', 8)]['sps'] / s[('S', 8)]['sps']:.2f} times as long",
        11: f"not run: it needs the 64-job kept chat (at K = 32, for what it is worth, {pc(s[('KC', 32)]['late'])} of the prompt came from the cache in the second half, {pc(s[('KC', 32)]['share'])} over all steps)",
        12: "not run: it needs the 64-job kept chat",
        13: f"`KC` {s[('KC', 32)]['sps']:.3f} seconds a step, `S` {s[('S', 32)]['sps']:.3f}: {s[('KC', 32)]['sps'] / s[('S', 32)]['sps']:.1f} times (limit 1.3)",
    }[i]


def table_cells(res):
    s = S_(res); out = []
    for (a, k) in ORDER:
        v = s[(a, k)]
        out.append(f"| `{a}` | {k} | {num(v['n'])} | {num(round(v['wall']))} | {v['sps']:.2f} | {num(v['prompt'])} | {num(v['computed'])} | {pc(v['share'], 0)} | {pc(v['late'], 0)} | {num(round(v['q1']))} | {num(round(v['q4']))} | {num(v['longest'])} | {pc(v['exact'])} |")
    return out


def table_anatomy(res):
    c = cells(res); s = S_(res); out = []
    for key in ORDER:
        v, an = s[key], anatomy(c[key]); a, k = key
        first = "no job lost its state" if an["firsts"] == 0 else f"{an['a']} did not parse, {an['b']} parsed and were refused, {an['o']} were admitted and still differ"
        out.append(f"| `{a}` | {k} | {v['whole']} of {v['jobs']} | {an['unparsed']} of {num(v['n'])} | {an['unparsed_del']} | {first} | {v['corrupted']} |")
    return out


def table_hour(res):
    s = S_(res); return [f"| `{a}` | {k} | {num(round(s[(a, k)]['sph']))} | {pc(s[(a, k)]['exact'])} | {num(round(s[(a, k)]['sph'] * s[(a, k)]['exact']))} |" for (a, k) in ORDER]


def figs(res):
    s = S_(res); c = cells(res); e = res["meta"]["engine"]; n = sum(v["n"] for v in s.values()); S8, K8, N8, S32, K32, S64 = (s[k] for k in ORDER)
    an = {k: anatomy(c[k]) for k in ORDER}; sa = [an[("S", 8)], an[("S", 32)], an[("S", 64)]]
    s_first = sum(x["firsts"] for x in sa); s_nopar = sum(x["a"] for x in sa); s_un = sum(x["unparsed"] for x in sa); s_undel = sum(x["unparsed_del"] for x in sa); s_n = S8["n"] + S32["n"] + S64["n"]
    u64, t64 = an[("S", 64)]["unk"], an[("S", 64)]["byk"]
    return dict(
        version=e["version"], gpu=e["gpu"], python=e["python"], sha=_sha(RES), msha=res["module_sha256"], total_secs=f"{sum(v['wall'] for v in s.values()):,.0f}", total_min=f"{sum(v['wall'] for v in s.values()) / 60:.0f}",
        kn_min=f"{N8['wall'] / 60:.0f}", kc32_min=f"{K32['wall'] / 60:.0f}", nsteps=num(n), s8=f"{S8['sps']:.3f}", k8=f"{K8['sps']:.3f}", s32=f"{S32['sps']:.3f}", k32=f"{K32['sps']:.3f}", s64=f"{S64['sps']:.3f}", n8=f"{N8['sps']:.3f}",
        s8_2=f"{S8['sps']:.2f}", k8_2=f"{K8['sps']:.2f}", s32_2=f"{S32['sps']:.2f}", k32_2=f"{K32['sps']:.2f}", s64_2=f"{S64['sps']:.2f}", n8_2=f"{N8['sps']:.2f}",
        k8_over_s8=f"{K8['sps'] / S8['sps']:.2f}", less10=f"{(1 - S8['sps'] / K8['sps']) * 100:.1f}", k32_over_s32=f"{K32['sps'] / S32['sps']:.1f}", k32_over_k8=f"{K32['sps'] / K8['sps']:.1f}", k32_of_n8=f"{K32['sps'] / N8['sps'] * 100:.0f}",
        n8_over_s8=f"{N8['sps'] / S8['sps']:.1f}", slow_round=f"{max(r['seconds'] for r in c[('KC', 32)]['rounds']):.0f}",
        s8_cps=f"{S8['cps']:.0f}", k8_cps=f"{K8['cps']:.0f}", n8_cps=num(round(N8["cps"])), fewer_kc=f"{(1 - S8['cps'] / K8['cps']) * 100:.1f}", fewer_kn=f"{(1 - S8['cps'] / N8['cps']) * 100:.1f}", k32_cps=num(round(K32["cps"])), s32_cps=f"{S32['cps']:.0f}",
        k8_share=pc(K8["share"]), k32_share=pc(K32["share"]), k32_late=pc(K32["late"]), s64_share=pc(S64["share"], 0),
        kq1=num(round(K8["q1"])), kq4=num(round(K8["q4"])), kgrow=f"{K8['q4'] / K8['q1']:.2f}", sq1=f"{S8['q1']:.0f}", sq4=f"{S8['q4']:.0f}", sflat=f"{S8['q4'] / S8['q1']:.2f}",
        s_long=num(S64["longest"]), s_long64=num(64 * S64["longest"]), s_long64_share=f"{64 * S64['longest'] / 80000 * 100:.0f}", k32_long=num(K32["longest"]), k32_long32=num(32 * K32["longest"]), k32_long32_x=f"{32 * K32['longest'] / 80000:.1f}",
        k8_long=num(K8["longest"]), k8_long8=num(8 * K8["longest"]), k32_steps=num(K32["n"]), k32_stopped=str(K32["stopped"]),
        ex_s8=pc(S8["exact"]), ex_s32=pc(S32["exact"]), ex_s64=pc(S64["exact"]), ex_k32=pc(K32["exact"]), gap8=f"{(K8['exact'] - S8['exact']) * 100:.1f}", gap32=f"{(K32['exact'] - S32['exact']) * 100:.1f}",
        corrupted=str(sum(v["corrupted"] for v in s.values())), s_first=str(s_first), s_nopar=str(s_nopar), s_un=str(s_un), s_n=num(s_n), s_undel=str(s_undel),
        u64_dd=f"{u64['delete_dep']} of {t64['delete_dep']}", u64_dl=f"{u64['delete_leaf']} of {t64['delete_leaf']}", u64_an=f"{u64['add_node']} of {t64['add_node']}", u64_ae=f"{u64['add_edge']} of {t64['add_edge']}",
        k32_un=str(an[("KC", 32)]["unparsed"]), s_recov=str(sum(recovered(c[k]) for k in (("S", 8), ("S", 32), ("S", 64)))), s_stuck=str(s_first - sum(recovered(c[k]) for k in (("S", 8), ("S", 32), ("S", 64)))), k32_undel=str(an[("KC", 32)]["unparsed_del"]), k32_whole=str(K32["whole"]), new_step=f"{sum(v['new'] for v in s.values()) / n:.1f}",
        sph_s32=num(round(S32["sph"])), sph_k32=num(round(K32["sph"])), right_s32=num(round(S32["sph"] * S32["exact"])), right_k32=num(round(K32["sph"] * K32["exact"])),
        right_ratio32=f"{S32['sph'] * S32['exact'] / (K32['sph'] * K32['exact']):.0f}", right_ratio8=f"{S8['sph'] * S8['exact'] / (K8['sph'] * K8['exact']):.1f}",
        kn_rate=num(round((N8["computed"] - K8["computed"]) / (N8["wall"] - K8["wall"]) / 10) * 10), ex_lo=pc(S64["exact"]), ex_hi=pc(S32["exact"]))


def render(res):
    f = figs(res); rows = "\n".join(f"| {i} | {PROBS[i - 1]} % | {measured(res, i)} | {'yes' if READ[i - 1] else 'not run' if READ[i - 1] is None else 'NO'} |" for i in range(1, 14))
    R = r"""

---

# Result

*Written after the run, below the locked text; nothing above this line has been changed (its hash is checked by `tests/test_provider_side.py` and again by `tests/test_provider_side_result.py`). The run was made by the owner on a free Colab GPU on 17 and 18 Oct 2026 with `python/colab/axi_provider_side.ipynb`: vLLM __version__, GPU __gpu__, Python __python__, `Qwen/Qwen2.5-7B-Instruct-AWQ`, the cache fixed at 80,000 tokens with the engine's block override (the engine took it: capacity source "override"), 30-step jobs, seed 20261017, replies not constrained. **Six of the eight planned cells ran:** `S` at K = 8, 32 and 64, `KC` at K = 8 and 32, `KN` at K = 8. **Not run: `KC` at K = 64** (the owner's free GPU time ran out inside that cell, and a cell that is cut off is not saved) **and `KR` at K = 32** (optional). The six cell files are in `docs/results/provider_side_cells/`; I put them together here, not in Colab, with `provider_side.merge_cells`: `docs/results/axi_provider_side.json` (sha256 `__sha__`; it records the sha256 of the module that merged it, `__msha__`). The GPU time of the six cells adds up to __total_secs__ seconds (__total_min__ minutes). Every figure below is computed from the step records of those files by `tests/test_provider_side_result.py`.*

**Two things about the run that the page did not foresee.**
* Cell 5 of the notebook first stopped with `io.UnsupportedOperation: fileno`: this vLLM version asks the notebook's output stream for a file number, which a Colab cell does not have. The owner ran a short cell that gives `sys.stdout` and `sys.stderr` the numbers 1 and 2 and ran Cell 5 again. `provider_side.py` was not changed (its sha256 is the one above). I tested the short cell against a copy of the situation here, not on Colab; the run is the evidence that it worked. Update 57 puts that cell into the notebook and pins vLLM to __version__, so that the missing cell can be run on the same version (a cell from another version is refused by the merge).
* My time guesses were too low. `KN` at K = 8 took __kn_min__ minutes, not about ten, and `KC` at K = 32 took __kc32_min__, not about thirty. `KN` computed prompt tokens at about __kn_rate__ a second (the tokens `KN` computed beyond `KC`, over the seconds it took beyond `KC`; a rough rate). For `KC` at K = 64 I would now expect about three hours or more: twice the jobs of K = 32 and a cache that overflows even earlier. That is an estimate, not a measurement.

## The 13 lines

| # | my probability | what was measured | read |
|---|---|---|---|
__ROWS__

**8 of the 13 lines read yes, lines 3 and 4 read NO, and lines 2, 11 and 12 are "not run".** Of the headline pair (lines 10 and 12) line 10 reads yes and line 12 could not be read, so the headline is half read and I do not call it either way. Lines 1 and 5 to 10 and 13 read yes; of those, lines 9 and 10 were the two I trusted least (50 and 40 %) and both came out yes, and lines 3 and 4, which I put at 85 % each, are the two NO.

**What the page said about lines 3 and 4.** "Line 3 or 4 NO: the arms are not doing the same work, and their times are not compared until the cause is found." So, whatever lines 9, 10 and 13 read, **this run makes no claim that exact-state is faster than a kept chat.** Their times are in the tables because nothing is to be hidden; what is claimed is below, under "What this does and does not show". **Line 2** is an instrument line and it is not read; the page said that a NO on it would stop the reading of everything else, and "not run" is not a NO. The cells that ran show the pressure was there at K = 32 (the figures are in the table row), but line 2 is written for K = 64.

## Where my expectation was off
* **Lines 9 and 10 (50 and 40 %): at light load the kept chat was not about as cheap as exact-state.** I wrote that at light load a working cache would make the two about equal (my model of a cache gave 35 % fewer tokens and 16 % less time for exact-state, the second under the 20 % line). The cache worked: __k8_share__ of the kept chat's prompt tokens came from it. Exact-state still computed __fewer_kc__ % fewer prompt tokens and took __less10__ % less time per step, so the kept chat took __k8_over_s8__ times as long (__k8__ against __s8__ seconds a step). I did not look for the reason. Candidates I have not tested: a prompt that grows from __kq1__ to __kq4__ tokens makes every step read more rows even when they are not computed again; the replies are short (__new_step__ new tokens a step in every cell), so the time is almost all prompt work.
* **Lines 3 and 4 (85 % each): exact-state did not hold the right state on 9 steps in 10.** __ex_s8__ at K = 8, __ex_s32__ at 32, __ex_s64__ at 64, against __ex_k32__ for the kept chat at 32 and 100 % at 8. My note under line 3 on the page had named the risk (a reply with words after the delta fails the strict parser and leaves the state unchanged, and the AWQ weights and the replies that are not constrained were new); I gave it a 15 % chance and it was the main event. See below for where the exactness went.
* **The time guesses** (above) and **line 6's threshold**: the kept chat's prompt grew __kgrow__ times, more than the 4.4 of the earlier run, so the lowered threshold of 3.5 was not needed; the lowering was disclosed on the page.
* **Lines 5, 7, 8 and 13 came out as expected**, line 13 by a wide margin (the limit is 1.3 times; the figure is __k32_over_s32__).

## The tables
**Every cell** (the seconds are the wall seconds around the engine's call, summed over the 30 rounds; "computed" is the prompt tokens the engine did not take from its cache; "cached, second half" is the steps from the 16th on; the prompt columns are the mean prompt of the first 7 and the last 7 steps of the jobs; "exact" is the share of steps after which the held state equalled the oracle's):

| arm | K | steps | seconds | s / step | prompt tokens | computed | cached | cached, 2nd half | prompt, first 7 | prompt, last 7 | longest | exact |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
__T1__

**Where the exactness went (counted afterwards, from the step records; none of it was pre-registered).** "Jobs exact to the end" are jobs whose held state equalled the oracle's after every step. "Unparsed" are replies the strict parser refused; "on delete steps" is how many of those were on a `delete_dep` or `delete_leaf` step. "First misses" classifies the first step of each job after which its state differed. "Corrupted" counts states that were not well formed.

| arm | K | jobs exact to the end | unparsed replies | on delete steps | first misses | corrupted |
|---|---|---|---|---|---|---|
__T2__

**Steps per hour (derived afterwards; read the caution).** "Steps per hour" is 3,600 times the steps done over the seconds of the cell; the last column multiplies it by the share of exact steps. It is crude: once a job's state falls behind the oracle's it usually stays behind (in 52 of the 56 `S` jobs that missed, it did), so every later step of the job counts as inexact, and by the page's own rule for lines 3 and 4 these times are not to be compared as a claim yet.

| arm | K | steps per hour | exact | exact steps per hour |
|---|---|---|---|---|
__T3__

## What I read from it (derived afterwards; none of this was pre-registered)
1. **The kept chat's cache worked at 8 jobs and gave way at 32.** At K = 8, __k8_share__ of its prompt came from the cache and it computed __k8_cps__ prompt tokens a step. At K = 32 the cache served __k32_share__ over all steps and __k32_late__ in the second half, it computed __k32_cps__ tokens a step, and the time per step went from __k8__ to __k32__ seconds, __k32_over_k8__ times as much, while each job did the same work in the same 30 steps and the cache was the same size. That is __k32_of_n8__ % of the time per step of `KN` at K = 8, the kept chat whose cache was emptied on purpose (__n8__ seconds a step). The slowest round of `KC` at K = 32 took __slow_round__ seconds. The cache was 80,000 tokens; 32 chats of the sizes measured here (the longest __k32_long__ tokens, 32 of them __k32_long32__, __k32_long32_x__ times the cache) do not fit in it. In __k32_stopped__ of the 32 chats the test's own stop rule ended the job (the prompt would have passed 7,600 tokens): __k32_steps__ steps were done instead of 960. A kept chat also runs out of room in the model's context.
2. **Exact-state stayed flat and small.** Its prompt was __sq1__ tokens over the first 7 steps and __sq4__ over the last 7 at K = 8 (__sflat__ times), the longest __s_long__ tokens in any cell; 64 jobs of that size need at most __s_long64__ tokens, __s_long64_share__ % of the cache. It served __s64_share__ of its prompt from the cache at 64 jobs, the shared instructions. Its time per step fell as the jobs grew in number: __s8__, __s32__ and __s64__ seconds at 8, 32 and 64 jobs (a GPU is used better with more jobs in a round).
3. **Where the exactness went.** No state was ever corrupted: __corrupted__ of the __nsteps__ steps. What failed was a reply that the strict parser refused (or the gate refused): the job then keeps its last valid state (policy `repair_m6`) and its state is behind the oracle's from then on: in __s_stuck__ of the __s_first__ `S` jobs that missed, no later step was exact again, and in __s_recov__ the state became exact again later. In the `S` cells __s_nopar__ of the __s_first__ first misses were replies that did not parse, for example `DEL[4,5:8]` (a value after a delete) and `ADD[(4,4):6]` (brackets around the coordinates); __s_un__ of the __s_n__ replies of the `S` cells did not parse, __s_undel__ of them on delete steps. At 64 jobs the refusals were __u64_dd__ of the `delete_dep` steps and __u64_dl__ of the `delete_leaf` steps, __u64_an__ of the `add_node` steps and __u64_ae__ of the `add_edge` steps. The kept chat made the same slip rarely: none of its 240 replies at 8 jobs, __k32_un__ of its __k32_steps__ at 32 jobs (all __k32_undel__ on delete steps), which cost one job its state (__k32_whole__ of 32 jobs exact to the end). **The cause is not found, but the most likely one is the replies that are not constrained.** The earlier long-horizon run held the exact state on 99.4 % of the steps in state mode with constrained decoding (`LONG_HORIZON_TEST.md`); this page chose replies that are not constrained because the constrained and the unconstrained call gave the same text in 60 of 60 single tasks, and 30-step jobs show that this did not carry over. A kept chat shows the model its own earlier replies in the right form, which fits its near-zero slip rate; I have not tested that either. Until a run with the constrained replies (or with equal exactness in another way) exists, the exactness of the arms is not equal and their times are not compared.
4. **Rough, and not a claim:** exact steps per hour at K = 32 are __right_s32__ for `S` and __right_k32__ for `KC`, a ratio of __right_ratio32__; at K = 8 the ratio is __right_ratio8__. The ratio at K = 8 is the one to keep in mind: it is the one measured where the cache worked.

## What this does and does not show
* **Shows, on one Tesla T4, one 7B model with 4-bit AWQ weights, vLLM __version__, a cache of 80,000 tokens and 30-step jobs:** a kept chat's prefix cache works when the chats fit in it and stops working when they do not (K = 8 against K = 32); the time of a kept chat per step then rose __k32_over_k8__ times; the exact-state prompt does not grow and stays a small share of the cache at 64 jobs. That is the measured part of the capacity idea: how many jobs a cache holds depends on the size of a job's prompt, and the size of the exact-state prompt does not grow with the length of the job.
* **Does not show that exact-state is cheaper at equal quality** (lines 3 and 4 NO), and so no ratio of the times is a claim. It does not show the kept chat at 64 jobs (line 12, the second half of the headline), a random order (`KR`), another GPU (this one computed about __kn_rate__ prompt tokens a second; a data-centre GPU is much faster, and the ratios, not the seconds, are what might carry over, which is not shown either), another model, engine or cache size, dollars, **energy (never measured; only time)**, retries (this test sends one request a step and never asks again; a job whose reply is refused costs a retry in a real system, for either arm), traffic that is not a round of jobs, or a number of agents per GPU: "four to ten times as many" is arithmetic on prompt sizes, not a measurement.
* **The sentence the article may carry from this run, with its limits:** *on one T4 with a 7B model and a cache of 80,000 tokens, a kept chat of 32 jobs overflowed the cache and took __k32__ seconds a step, against __k8__ for 8 jobs; exact-state prompts stayed at __s_long__ tokens or fewer and 64 jobs of them needed at most __s_long64_share__ % of the cache; exact-state held the right state on __ex_lo__ to __ex_hi__ of its steps against __ex_k32__ for the kept chat at 32 jobs, with replies that were not constrained, so the arms are not compared.*
* **Not done and worth doing, in this order:** (1) the same cells with replies constrained by a grammar for all arms, so that exactness is equal first and the times are compared after (a new page and new predictions; `S` is cheap, a few minutes a cell, `KC` at K = 32 is about an hour and a half); (2) `KC` at K = 64 on the same vLLM version; (3) retries counted, so that a refused reply costs what it costs; (4) the GPU's power read next to the seconds.

## Reproduce
In Colab, `python/colab/axi_provider_side.ipynb` with a T4 (the cell that fixes the output stream first, then Cell 5 onward); about __total_min__ minutes of GPU for the six cells here, and about three hours more for `KC` at K = 64. Nothing needs the GPU after that: `provider_side.merge_cells` puts cell files together on any machine. `tests/test_provider_side_result.py` recomputes every figure of this Result from the cell files, checks that the merged file is what `merge_cells` makes of them, and that the page is the locked text followed by this Result.
"""
    for k, v in f.items(): R = R.replace(f"__{k}__", v)
    R = R.replace("__ROWS__", rows).replace("__T1__", "\n".join(table_cells(res))).replace("__T2__", "\n".join(table_anatomy(res))).replace("__T3__", "\n".join(table_hour(res)))
    assert "__" not in R.replace("__init__", ""), [w for w in R.split() if "__" in w][:5]
    return R


# ---------------------------------------------------------------- 1 the files
def test_the_files_are_the_ones_the_page_is_about_the_merge_is_what_the_code_makes_of_them_and_the_numbers_add_up():
    res = run(); assert _sha(RES) == FILE_SHA256, "the saved result file must not change"; assert _sha(MODULE) == MODULE_SHA256 == res["module_sha256"], "the module that merged the cells must be the one in the repository"
    names = sorted(os.listdir(CELLDIR)); assert names == sorted(f"axi_provider_{k}.json" for k in CELL_SHA256), names
    src = []
    for k, sha in CELL_SHA256.items():
        p = os.path.join(CELLDIR, f"axi_provider_{k}.json"); assert _sha(p) == sha, k; src.append(json.load(open(p)))
    merged = P.merge_cells(src); assert merged == {k: v for k, v in res.items() if k != "module_sha256"} and set(res) == {"meta", "cells", "module_sha256"}
    e = res["meta"]["engine"]; assert e == dict(kind="vllm", name="Qwen/Qwen2.5-7B-Instruct-AWQ", version="0.31.0", gpu="Tesla T4", capacity_tokens=80000, capacity_source="override", block_size=16, kv_bytes_per_token=57344, model="Qwen/Qwen2.5-7B-Instruct-AWQ", python="3.13.15")
    assert (res["meta"]["steps"], res["meta"]["seed"], res["meta"]["policy"], res["meta"]["hint_rules"]) == (30, 20261017, "repair_m6", False)
    assert [(c["arm"], c["K"]) for c in res["cells"]] == ORDER and all(c["engine"] == e and c["steps"] == 30 and c["seed"] == 20261017 for c in res["cells"])
    scripts = {}
    for c in res["cells"]:
        assert [s["seed"] for s in c["sessions"]] == [20261017 * 1000 + i for i in range(c["K"])] and all(s["script_steps"] == 30 for s in c["sessions"])
        for s in c["sessions"]:
            kinds = [x["kind"] for x in s["log"]]; assert [x["t"] for x in s["log"]] == list(range(1, len(kinds) + 1)); full = scripts.setdefault(s["seed"], kinds)
            assert kinds == full[:len(kinds)], (c["arm"], c["K"], s["seed"])                                                    # the same job script in every cell, at every K
            assert (s["stopped"] is not None) == (len(kinds) < 30) and s["stopped"] in (None, "context")
        L = logs(c); assert abs(sum(r["seconds"] for r in c["rounds"]) - c["wall_seconds"]) < 1e-6 and len(c["rounds"]) == 30
        assert sum(r["prompt_tokens"] for r in c["rounds"]) == sum(x["n_prompt"] for x in L) and sum(r["cached_tokens"] for r in c["rounds"]) == sum(x["n_cached"] for x in L) and sum(r["new_tokens"] for r in c["rounds"]) == sum(x["n_new"] for x in L)
        for i, r in enumerate(c["rounds"], 1): assert r["round"] == i and r["active"] == sum(1 for s in c["sessions"] if len(s["log"]) >= i)
        assert all(0 <= x["n_cached"] <= x["n_prompt"] and isinstance(x["n_cached"], int) and x["n_new"] > 0 for x in L)
        assert all(bool(x["exact"]) == (x["distance"] == 0) for x in L)
    assert len(scripts) == 64 and sorted(scripts) == [20261017000 + i for i in range(64)]
    kn, kc = cells(res)[("KN", 8)], cells(res)[("KC", 8)]
    assert [x["text"] for x in logs(kn)] == [x["text"] for x in logs(kc)] and [x["n_prompt"] for x in logs(kn)] == [x["n_prompt"] for x in logs(kc)]      # KN is KC with the cache emptied: the same chats and the same replies


# ---------------------------------------------------------------- 2 the rules
def test_the_thirteen_rules_read_as_the_page_says_and_the_headline_is_half_read():
    res = run(); v = [x for _, _, _, x in P.verdict_provider(P.summarize_all(res["cells"]))]; assert v == READ and len(PROBS) == 13 and P.PL_HEADLINE == (10, 12) and [v[i - 1] for i in P.PL_HEADLINE] == [True, None]
    assert sum(1 for x in v if x is True) == 8 and sum(1 for x in v if x is False) == 2 and sum(1 for x in v if x is None) == 3
    # the same rules from the raw numbers, by hand
    s = S_(res); g = lambda a, k: s[(a, k)]
    hand = [all(x["n_cached"] is not None for c in res["cells"] for x in logs(c)), None, all(x["exact"] >= 0.90 for x in s.values()),
            all(g("S", k)["exact"] >= g("KC", k)["exact"] - 0.05 for k in (8, 32)), g("S", 8)["q4"] / g("S", 8)["q1"] <= 1.3, g("KC", 8)["q4"] / g("KC", 8)["q1"] >= 3.5, g("KC", 8)["share"] >= 0.80,
            g("S", 8)["cps"] <= 0.2 * g("KN", 8)["cps"], g("S", 8)["cps"] <= 0.7 * g("KC", 8)["cps"], g("S", 8)["sps"] <= 0.8 * g("KC", 8)["sps"], None, None, g("KC", 32)["sps"] >= 1.3 * g("S", 32)["sps"]]
    for i in range(13): assert hand[i] == v[i], (i + 1, hand[i], v[i])
    assert ("KC", 64) not in s and ("KR", 32) not in s and len(s) == 6


# ---------------------------------------------------------------- 3 the page
def test_the_part_above_the_result_is_what_was_written_before_the_run():
    head = _head(); assert _h(head) == PAGE_SHA256 and _text().count(RESULT_MARK) == 1


def test_the_page_carries_exactly_the_result_this_file_renders_and_its_table_says_what_the_rules_say():
    res = run(); text = _text(); assert text == _head().rstrip("\n") + render(res).rstrip("\n") + "\n"
    rows = [ln for ln in text[text.index(RESULT_MARK):].split("\n") if re.match(r"\| \d+ \| \d+ % \|", ln)]; assert len(rows) == 13
    for i, ln in enumerate(rows, 1):
        cells_ = [c.strip() for c in ln.strip("|").split("|")]; assert int(cells_[0]) == i and cells_[1] == f"{PROBS[i - 1]} %" and cells_[2] == measured(res, i) and cells_[3] == ("yes" if READ[i - 1] else "not run" if READ[i - 1] is None else "NO"), i
    probs_on_page = [int(m.group(1)) for m in re.finditer(r"\n\d+\. \[\w+\] .*\((\d+) %\)(?=\n)", _head())]; assert probs_on_page == PROBS


def test_the_figures_quoted_in_the_result_are_the_ones_in_the_files():
    res = run(); R = _text()[_text().index(RESULT_MARK):]
    for needle in ("vLLM 0.31.0", "GPU Tesla T4", "Python 3.13.15", "19ce96633931980292f2ab6b25c42799b50696db2ea25225f194e02c8ca93d48", "bfb4734e171d9244cd47d10402be46528a557095a8275d02b38481a0e2b0b1d3", "7,055 seconds (118 minutes)", "took 27 minutes", "took 82, not about thirty",
                   "4,552 steps in the 6 cells that ran", "241,792", "3.0 times the cache of 80,000", "57,232", "8 of the 13 lines read yes", "4.00 times as long", "0.617 against 0.154", "37.3 % fewer prompt tokens", "75.0 % less time per step",
                   "5.36 times", "96.3 % fewer", "42.7", "5.149 seconds a step", "0.121", "0.107", "76.2 %", "78.2 %", "69.2 %", "97.3 %", "94.3 %", "26.0 %", "6.3 %", "8.3 times", "392 seconds", "952 steps were done instead of 960",
                   "0 of the 4,552 steps", "52 of the 56 `S` jobs that missed", "in 4 the state became exact again later", "48 of the 56 first misses", "80 of the 3,120 replies", "37 of 508", "18 of 171", "3 of 627", "0 of 614", "9 of its 952", "31 of 32 jobs exact to the end", "`DEL[4,5:8]`", "`ADD[(4,4):6]`", "99.4 %", "60 of 60"):
        assert needle in R, needle
    # recomputed here from the raw records with plain arithmetic
    c = cells(res); S8, K8, N8, S32, K32, S64 = (c[k] for k in ORDER)
    for cell, steps, wall in ((S8, 240, 37.0), (K8, 240, 148.2), (N8, 240, 1646.3), (S32, 960, 115.9), (K32, 952, 4901.9), (S64, 1920, 205.2)):
        assert len(logs(cell)) == steps and abs(sum(r["seconds"] for r in cell["rounds"]) - wall) < 0.06
    tot = lambda cell, key: sum(x[key] for x in logs(cell))
    assert (tot(S8, "n_prompt"), tot(S8, "n_cached")) == (90679, 61136) and (tot(K8, "n_prompt"), tot(K8, "n_cached")) == (831529, 784384) and (tot(N8, "n_prompt"), tot(N8, "n_cached")) == (831529, 40320)
    assert (tot(S32, "n_prompt"), tot(S32, "n_cached")) == (349176, 239488) and (tot(K32, "n_prompt"), tot(K32, "n_cached")) == (3065348, 796944) and (tot(S64, "n_prompt"), tot(S64, "n_cached")) == (682313, 473824)
    assert [max(x["n_prompt"] for x in logs(k)) for k in (S8, K8, S32, K32, S64)] == [516, 7154, 516, 7556, 516] and [sum(x["exact"] for x in logs(k)) for k in (S8, K8, N8, S32, K32, S64)] == [183, 240, 240, 751, 926, 1329]
    assert sum(len(logs(k)) for k in (S8, K8, N8, S32, K32, S64)) == 4552 and sum(x["corrupted"] for k in (S8, K8, N8, S32, K32, S64) for x in logs(k)) == 0
    assert sum(1 for s in K32["sessions"] if s["stopped"] == "context") == 3 and sum(len(s["log"]) for s in K32["sessions"]) == 952 and all(len(s["log"]) == 30 for k in (S8, K8, N8, S32, S64) for s in k["sessions"])
    ex = lambda cell: f"{sum(x['exact'] for x in logs(cell)) / len(logs(cell)) * 100:.1f} %"; assert [ex(k) for k in (S8, S32, S64, K32)] == ["76.2 %", "78.2 %", "69.2 %", "97.3 %"]


# ---------------------------------------------------------------- 5 the part written afterwards
def test_where_the_exactness_went_is_counted_from_the_step_records():
    res = run(); c = cells(res); an = {k: anatomy(v) for k, v in c.items()}
    assert [an[k]["firsts"] for k in ORDER] == [5, 0, 0, 14, 1, 37] and [(an[k]["a"], an[k]["b"], an[k]["o"]) for k in ORDER] == [(4, 1, 0), (0, 0, 0), (0, 0, 0), (11, 2, 1), (1, 0, 0), (33, 2, 2)]
    assert [(an[k]["unparsed"], an[k]["unparsed_del"]) for k in ORDER] == [(6, 5), (0, 0), (0, 0), (16, 13), (9, 9), (58, 55)] and sum(an[k]["unparsed"] for k in (("S", 8), ("S", 32), ("S", 64))) == 80
    assert sum(an[k]["unparsed_del"] for k in (("S", 8), ("S", 32), ("S", 64))) == 73 and sum(an[k]["firsts"] for k in (("S", 8), ("S", 32), ("S", 64))) == 56 and sum(an[k]["a"] for k in (("S", 8), ("S", 32), ("S", 64))) == 48
    s = S_(res); assert [s[k]["whole"] for k in ORDER] == [3, 8, 8, 18, 31, 27]
    # a job's state that fell behind usually stays behind: 52 of the 56 jobs that missed never came back, 4 did (S at K = 32: 1; at K = 64: 3); no other cell has a job that came back
    assert [recovered(c[k]) for k in ORDER] == [0, 0, 0, 1, 0, 3]
    for k in ORDER:
        lost = sum(len(sess["log"]) - [x["exact"] for x in sess["log"]].index(False) for sess in c[k]["sessions"] if False in [x["exact"] for x in sess["log"]])      # steps from the first miss to the end of the job
        assert lost >= len(logs(c[k])) - sum(x["exact"] for x in logs(c[k])), k
    # the two examples named in the prose are the first misses they are said to be
    texts = [x["text"] for x in logs(c[("S", 8)]) if not x["parsed"]]; assert "DEL[4,5:8]" in texts and "ADD[(4,4):6]" in texts
    # the earlier run that held the state at 99.4 % used constrained decoding, as the Result says
    assert "Qwen2.5-7B-Instruct, 4-bit, Colab T4, greedy, 400 new tokens, constrained decoding." in open(LONGPAGE).read()
    assert "the constrained and the unconstrained call gave the same text in 60 of 60 tasks" in _head()


# ---------------------------------------------------------------- 6 what the page said it would say
def test_the_result_says_what_the_page_said_it_would_say_when_lines_3_and_4_read_no_and_claims_nothing_more():
    head, R = _head(), _text()[_text().index(RESULT_MARK):]
    assert "*Line 3 or 4 NO:* the arms are not doing the same work, and their times are not compared until the cause is found." in head
    for needle in ("Line 3 or 4 NO: the arms are not doing the same work, and their times are not compared until the cause is found.", "this run makes no claim that exact-state is faster than a kept chat", "**Does not show that exact-state is cheaper at equal quality**",
                   "energy (never measured; only time)", "retries (this test sends one request a step and never asks again", "\"four to ten times as many\" is arithmetic on prompt sizes, not a measurement", "so the arms are not compared",
                   "**The cause is not found, but the most likely one is the replies that are not constrained.**", "line 12, the second half of the headline", "Not run: `KC` at K = 64"):
        assert needle in R, needle
    assert R.count("faster than") == 1 and "no claim that exact-state is faster than a kept chat" in R and "Does not show that exact-state is cheaper" in R
    for banned in ("proves", "proven", "wins", "massive", "hyperscaler", "dollars saved"): assert banned not in R.lower().replace("never in dollars", ""), banned


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("PASS", n)
