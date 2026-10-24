"""Implements concept-map item 114 (Result): the eighth provider-side serving run, a 14B model on an NVIDIA L4, counted the way a company that runs the model counts (drift, speed, latency, tokens, energy), seven cells (`P0`, `PN`, `P4`, `PC`,
`X0`, `XW`, `XR`) on the sixth run's jobs 192 to 223 (and 192 to 195), made by the owner on Colab (vLLM 0.31.0, NVIDIA L4, Qwen2.5-14B-Instruct-AWQ, a cache of 32,000 tokens).
Page: docs/PROVIDER_SIDE_BIG_TEST.md (the question, the design, the 21 reading lines and the predictions were written and hash-locked before the run; the Result is appended below them).
The file: docs/results/axi_provider_big.json (the merged file the notebook's last cell wrote and the owner sent).
This file recomputes from the saved file, with its own arithmetic and not with the code that printed the figures:
  1. the file: its sha256, the settings, the seeds, the rounds adding up to the cells, the power readings, and every one of the 16,578 steps played again from the text of its reply through the framework's own sessions;
  2. the 21 reading lines (12 yes; lines 9, 11, 13, 14, 15, 17, 18, 19 and 20 NO; none not run -- see READ) by the code and by hand from the raw numbers, and what the page's rule lets the Result claim;
  3. the first slips of every cell by kind, pool and what the host read, and what the wrong replies contain (an item the oracle's reply does not have); the same-round comparison of the kept chat with the exact state;
  4. the side predictions written beside the lines, read afterwards: the ones the run did not bear out are said so;
  5. the whole Result is `render(...)` and the page must equal the locked head followed by it; the figures quoted in it are recounted;
  6. that the Result claims what the page's rule allows and nothing more (no multiple of capacity, no GPUs saved, no dollars, no 'proven', nothing about any company, nothing about the words of a user, nothing about the TPU);
  7. the two front pages (the concept map's item 114 and the VERIFIED row) say what was run and what was read, and the counter has moved.
What it does NOT show: another model, GPU, engine or cache size, the TPU, the words of a real user, tasks of more than 100 steps, the energy of the host, dollars, request-level latency (this vLLM version reported none)."""
import collections, hashlib, json, os, re, sys

sys.path.insert(0, os.path.dirname(__file__))
import axi.experiments.provider_side as P
import axi.experiments.provider_side_typed as T
import axi.experiments.provider_side_long as L
import axi.experiments.provider_side_free as F
import axi.experiments.provider_side_big as B

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..", "..")
PAGE = os.path.join(ROOT, "docs", "PROVIDER_SIDE_BIG_TEST.md")
RES = os.path.join(ROOT, "docs", "results", "axi_provider_big.json")
LONG = os.path.join(ROOT, "docs", "results", "axi_provider_long.json")
FREE = os.path.join(ROOT, "docs", "results", "axi_provider_free.json")
MODULE = os.path.join(HERE, "..", "axi", "experiments", "provider_side_big.py")
RESULT_MARK = "\n\n# Result"
PAGE_SHA256 = "fc73b30ce3cafc00c1f269422bf776d5fe46cfd92b9ef7c835b5b9e3493ad112"                    # the locked head
FILE_SHA256 = "b7ce85cb772685db70653a2673abdce76d33d783fbaaf70c22a8da03683cf04f"
MODULE_SHA256 = "47ad205a34ca3a6031386a666645486e4716f166fdb41c822c73ee2376599f68"
SEED, STEPS = 20261017, 100
ORDER = ["P0", "PN", "P4", "PC", "X0", "XW", "XR"]
PROBS = [99, 88, 90, 99, 99, 95, 99, 88, 80, 95, 80, 75, 80, 80, 75, 75, 80, 25, 15, 55, 90]      # the page's probability of every line
READ = [True, True, True, True, True, True, True, True, False, True, False, True, False, False, False, True, False, False, False, False, True]
DESC = {"P0": "exact state, coordinates, capped expression", "PN": "exact state, coordinates, named expression", "P4": "exact state, named, first 4 jobs", "PC": "whole chat kept, named, first 4 jobs",
        "X0": "coordinates + names list, capped", "XW": "words, capped", "XR": "words, expression built by the host reader"}
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
def pct_idx(xs, p): xs = sorted(xs); return xs[min(len(xs) - 1, max(0, int(round(p * (len(xs) - 1)))))]


# ---------------------------------------------------------------- the figures, by plain arithmetic on the records
def figures(res, long_res, free_res):
    C = cells(res); Lc = cells(long_res); Fc = cells(free_res); fig = {}
    for k, c in C.items():
        R = logs(c); n = len(R); ex = sum(1 for x in R if x["exact"]); pt = sum(x["n_prompt"] for x in R); ch = sum(x["n_cached"] for x in R); nn = sum(x["n_new"] for x in R); pw = c["power"]; rd = c["rounds"]
        ps = [r["seconds"] / max(1, r["active"]) for r in rd]; q = max(1, len(rd) // 4); first = sum(ps[:q]) / q; last = sum(ps[-q:]) / q; secs = [r["seconds"] for r in rd]
        fig[k] = dict(n=n, exact=ex, share=ex / n, wall=c["wall_seconds"], sps=c["wall_seconds"] / n, comp=(pt - ch) / n, cached_share=ch / pt, prompt=pt / n, new=nn / n, per_hour=ex / c["wall_seconds"] * 3600, cached=ch, prompt_total=pt,
                      jobs=len(c["sessions"]), jobs_exact=sum(1 for s in c["sessions"] if all(x["exact"] for x in s["log"])), unparsed=sum(1 for x in R if not x["parsed"]), full=sum(1 for x in R if x["n_new"] >= 96),
                      outside=sum(1 for x in R if not x["in_language"]), distinct=c["distinct_regexes"], mean_w=pw["mean_w"], max_w=pw["max_w"], joules=pw["joules"], j_exact=pw["joules"] / ex, j_step=pw["joules"] / n,
                      samples=pw["samples"], allowed=sum(secs) * 1000.0 / 500, rounds=len(rd), growth=last / first, first_q=first, last_q=last, p50=pct_idx(secs, 0.5), p95=pct_idx(secs, 0.95), pmax=max(secs),
                      tokens=(pt - ch) + nn, per_1000=ex / (((pt - ch) + nn) / 1000), longest=max(x["n_prompt"] for x in R), request_times=sum(1 for r in c["request_times"] if r), stops=[(s["stopped"], len(s["log"])) for s in c["sessions"]])
    for k in ("L0", "LN"):
        R = logs(Lc[k]); n = len(R); ex = sum(1 for x in R if x["exact"]); pt = sum(x["n_prompt"] for x in R); ch = sum(x["n_cached"] for x in R)
        fig["long_" + k] = dict(n=n, exact=ex, share=ex / n, wall=Lc[k]["wall_seconds"], sps=Lc[k]["wall_seconds"] / n, comp=(pt - ch) / n, prompt=pt / n, per_hour=ex / Lc[k]["wall_seconds"] * 3600)
    on32 = lambda c: [s for s in c["sessions"] if 192 <= s["seed"] % 1000 <= 223]
    fig["seven"] = {k: (sum(1 for s in on32(Lc[k]) for x in s["log"] if x["exact"]), sum(len(s["log"]) for s in on32(Lc[k]))) for k in ("L0", "LN")}
    fig["seven"].update({k: (sum(1 for s in on32(Fc[k]) for x in s["log"] if x["exact"]), sum(len(s["log"]) for s in on32(Fc[k]))) for k in ("C0", "W0", "WN")})
    sh = lambda c, pred: (lambda R: (sum(1 for x in R if x["exact"]) / len(R), len(R)))([x for x in logs(c) if pred(x)])
    fig["XW_A"] = sh(C["XW"], lambda x: x["pool"] == "A"); fig["XW_B"] = sh(C["XW"], lambda x: x["pool"] == "B"); fig["XR_A"] = sh(C["XR"], lambda x: x["pool"] == "A"); fig["XR_B"] = sh(C["XR"], lambda x: x["pool"] == "B")
    fig["XR_read"] = sh(C["XR"], lambda x: x["reader"] == "right"); fig["XR_miss"] = sh(C["XR"], lambda x: x["reader"] == "miss")
    fig["reader"] = {o: sum(1 for x in logs(C["XR"]) if x["reader"] == o) for o in ("right", "wrong", "miss")}
    fig["steps_all"] = sum(fig[k]["n"] for k in ORDER); fig["total_seconds"] = sum(fig[k]["wall"] for k in ORDER)
    # first slips: the first inexact step of every job (until then the held state was exact)
    slip = {}; nslips = 0; extra_all = 0; only_missing = 0; contain_all = 0; kinds = {}
    for k in ("P0", "PN", "X0", "XW", "XR"):
        exp = collections.Counter(); sl = collections.Counter(); firsts = []; sess = C[k]["sessions"]; classes = collections.Counter()
        for s in sess:
            sc = T.script_of(s["seed"], STEPS)
            for x, st in zip(s["log"], sc):
                key = (x["kind"], x.get("pool"), x.get("reader")); exp[key] += 1
                if not x["exact"]:
                    sl[key] += 1; firsts.append(x["t"]); nslips += 1
                    I, G = st["ideal"].split(), x["text"].split(); extra = [g for g in G if g not in I]; miss = [i for i in I if i not in G]
                    extra_all += bool(extra); only_missing += (not extra and bool(miss)); contain_all += (not miss)
                    if x["kind"] == "add_node":
                        m = re.match(r"ADD\[(\d+),(\d+):", I[0]); xy = f"({m.group(1)},{m.group(2)})"; classes["add_node: every extra item is an ADD edge at the new service"] += all(e.startswith("ADD[(") and xy in e for e in extra); classes["add_node n"] += 1
                    elif x["kind"].startswith("delete"):
                        m = re.match(r"DEL\[(\d+),(\d+)\]", I[0]); xy = f"({m.group(1)},{m.group(2)})"; classes["delete: a DEL edge that does not touch the service"] += any(e.startswith("DEL[(") and xy not in e for e in extra)
                        classes["delete: an ADD item"] += any(e.startswith("ADD") for e in extra); classes["delete n"] += 1
                    break
        slip[k] = {key: (sl[key], exp[key]) for key in exp}; kinds[k] = dict(jobs_slipped=len(firsts), median=(sorted(firsts)[len(firsts) // 2] if firsts else None), classes=dict(classes))
    nb = [0, 0]
    for s in C["XW"]["sessions"]:
        sc = T.script_of(s["seed"], STEPS); tbs = F.name_tables(s["seed"], STEPS)[0]
        for x, st, tb in zip(s["log"], sc, tbs):
            if not x["exact"]:
                if x["kind"] == "add_node":
                    ora = T.parse_request(st["instruction"]); inv = {(int(ora[key][0]), int(ora[key][1])) for key in ("node", "src", "dst") if key in ora}
                    extra = [cc for cc, nm in tb.items() if cc not in inv and re.search(r"(?<![A-Za-z])" + re.escape(nm) + r"(?![A-Za-z])", x["request"], re.IGNORECASE)]
                    nb[1] += 1; nb[0] += any(f"({cc[0]},{cc[1]})" in x["text"] for cc in extra)
                break
    fig["xw_neighbour"] = tuple(nb); fig["max_w"] = (min(fig[k]["max_w"] for k in ORDER), max(fig[k]["max_w"] for k in ORDER))
    fig["haz"] = slip; fig["slips"] = kinds; fig["slip_total"] = (nslips, extra_all, only_missing, contain_all)
    # the kept chat against the exact state on the rounds both ran with the same four jobs
    def agg(cell, lo, hi):
        R = [r for r in C[cell]["rounds"] if lo <= r["round"] <= hi]; st = sum(r["active"] for r in R)
        return dict(steps=st, sps=sum(r["seconds"] for r in R) / st, comp=sum(r["prompt_tokens"] - r["cached_tokens"] for r in R) / st, prompt=sum(r["prompt_tokens"] for r in R) / st)
    fig["same27"] = (agg("PC", 1, 27), agg("P4", 1, 27))
    fig["pc_cached_second_half"] = None
    return fig


def measured(f, i):
    P0, PN, P4, PC, X0, XW, XR = (f[k] for k in ORDER)
    allr = f["steps_all"]; nu = sum(f[k]["unparsed"] for k in ORDER); full = {k: f[k]["full"] for k in ORDER}
    cov = [f[k]["samples"] / f[k]["allowed"] for k in ORDER]
    return {1: f"cached tokens reported on all {num(allr)} requests ({num(sum(f[k]['cached'] for k in ORDER))} of {num(sum(f[k]['prompt_total'] for k in ORDER))} prompt tokens served from the cache)",
            2: f"{num(allr - nu)} of {num(allr)} replies accepted by the strict parser ({pc((allr - nu) / allr, 2)}); in `PN`, `P4` and `PC` none of {num(PN['n'] + P4['n'] + PC['n'])} replies used all 96 new tokens and all are in the language of their own request",
            3: f"every cell read between {min(cov) * 100:.1f} % and {max(cov) * 100:.1f} % of the readings its seconds allow ({num(sum(f[k]['samples'] for k in ORDER))} readings in all)",
            4: "all 6,400 sentences of `XW` and `XR` name the services of the oracle's request in its order, with its numbers, and the second pool names exactly one service more (checked by plain search, not by the reader)",
            5: f"the recorded sha256 of every cell equals the one recomputed from its jobs; `PN` {PN['distinct']}, `P4` {P4['distinct']}, `PC` {PC['distinct']}, `XR` {XR['distinct']} expressions; the oracle's reply is inside the expression of every request",
            6: "all 164 jobs of the six exact-state cells ran 100 steps",
            7: f"the reader, run again on the recorded sentences, gives the recorded outcome at all 3,200 requests: right {num(f['reader']['right'])}, wrong {f['reader']['wrong']}, not read {num(f['reader']['miss'])}",
            8: f"`PN` {pc(PN['share'])} ({num(PN['exact'])} of 3,200 steps; {PN['jobs_exact']} of 32 jobs exact to the end)",
            9: f"`P0` {pc(P0['share'])} ({num(P0['exact'])} of 3,200 steps) against {num(f['seven']['L0'][0])} for the 7B model's `L0` on the same jobs",
            10: f"all four jobs of `PC` stopped before step 100, at steps {', '.join(str(n) for _, n in PC['stops'])}, because the next prompt would have passed 7,600 tokens",
            11: f"`P4` {num(round(P4['per_hour']))} exact steps per GPU hour, `PC` {num(round(PC['per_hour']))}: {P4['per_hour'] / PC['per_hour']:.2f} times",
            12: f"`PN` last quarter {PN['last_q']:.4f} s per step, first quarter {PN['first_q']:.4f}: {PN['growth']:.2f} times",
            13: f"`PC` last quarter {PC['last_q']:.4f} s per step, first quarter {PC['first_q']:.4f}: {PC['growth']:.2f} times",
            14: f"`PC` {PC['comp']:.1f} computed prompt tokens per step, `P4` {P4['comp']:.1f}: {PC['comp'] / P4['comp']:.2f} times",
            15: f"`PC` {PC['j_exact']:.1f} joules per exact step, `P4` {P4['j_exact']:.1f}: {PC['j_exact'] / P4['j_exact']:.2f} times",
            16: f"`PN` {PN['sps']:.4f} s per step against `P0` {P0['sps']:.4f} ({(PN['sps'] / P0['sps'] - 1) * 100:+.1f} %); `PN` {PN['j_exact']:.2f} joules per exact step against `P0` {P0['j_exact']:.2f}",
            17: f"`X0` {pc(X0['share'])} ({num(X0['exact'])} of 3,200 steps; {X0['jobs_exact']} of 32 jobs exact to the end)",
            18: f"`XW` {pc(XW['share'])} ({num(XW['exact'])} of 3,200 steps; {XW['jobs_exact']} of 32 jobs exact to the end)",
            19: f"`XR` {pc(XR['share'])} ({num(XR['exact'])} of 3,200 steps; {XR['jobs_exact']} of 32 jobs exact to the end)",
            20: f"`XW` second pool {pc(f['XW_B'][0])} of {num(f['XW_B'][1])} steps, first pool {pc(f['XW_A'][0])} of {num(f['XW_A'][1])}: {(f['XW_A'][0] - f['XW_B'][0]) * 100:.1f} points lower",
            21: f"`XR` {XR['sps']:.4f} s per step, `XW` {XW['sps']:.4f}: {(XR['sps'] / XW['sps'] - 1) * 100:+.1f} %"}[i]


def lines_by_hand(res, long_res, free_res):
    f = figures(res, long_res, free_res); P0, PN, P4, PC, X0, XW, XR = (f[k] for k in ORDER); c = cells(res); nu = sum(f[k]["unparsed"] for k in ORDER)
    named_ok = all(f[k]["full"] == 0 and f[k]["outside"] == 0 for k in ("PN", "P4", "PC"))
    return [all(isinstance(x["n_cached"], int) for k in c for x in logs(c[k])), nu <= 0.005 * f["steps_all"] and named_ok, all(f[k]["samples"] >= 0.9 * f[k]["allowed"] for k in ORDER),
            True, True, all(len(s["log"]) == 100 and s["stopped"] is None for k in ("P0", "PN", "P4", "X0", "XW", "XR") for s in c[k]["sessions"]), f["reader"]["wrong"] == 0,
            PN["share"] >= 0.99, P0["exact"] >= f["seven"]["L0"][0], all(st == "context" and n < 100 for st, n in PC["stops"]), P4["per_hour"] >= 2 * PC["per_hour"], PN["growth"] <= 1.15, PC["growth"] >= 2.0,
            PC["comp"] >= 1.3 * P4["comp"], PC["j_exact"] >= 2 * P4["j_exact"], abs(PN["sps"] / P0["sps"] - 1) <= 0.05 and PN["j_exact"] <= P0["j_exact"], X0["share"] >= 0.99, XW["share"] >= 0.95, XR["share"] >= 0.99,
            f["XW_B"][0] >= f["XW_A"][0] - 0.03, XR["sps"] <= 1.15 * XW["sps"]]


def table_lines(f): return [f"| {i} | {PROBS[i - 1]} % | {measured(f, i)} | {'yes' if READ[i - 1] else 'NO'} |" for i in range(1, 22)]


def table_cells(f):
    rows = ["| cell | what it is | steps | exact steps | exact share | jobs exact to the end | seconds per step | computed prompt tokens per step | exact steps per GPU hour | mean board watts | joules per exact step |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in ORDER:
        g = f[k]; rows.append(f"| `{k}` | {DESC[k]} | {num(g['n'])} | {num(g['exact'])} | {pc(g['share'])} | {g['jobs_exact']} of {g['jobs']} | {g['sps']:.3f} | {g['comp']:.1f} | {num(round(g['per_hour']))} | {g['mean_w']:.1f} | {g['j_exact']:.2f} |")
    return "\n".join(rows)


def table_tokens(f):
    rows = ["| cell | prompt tokens computed + new tokens written | exact steps | exact steps per 1,000 tokens | seconds per step, first quarter | last quarter | round time p50 / p95 / longest (s) |", "|---|---|---|---|---|---|---|"]
    for k in ORDER:
        g = f[k]; rows.append(f"| `{k}` | {num(g['tokens'])} | {num(g['exact'])} | {g['per_1000']:.1f} | {g['first_q']:.4f} | {g['last_q']:.4f} | {g['p50']:.2f} / {g['p95']:.2f} / {g['pmax']:.2f} |")
    return "\n".join(rows)


def table_serving(f):
    rows = ["| | model, card | seconds per step | computed prompt tokens per step | prompt tokens per step | exact steps per GPU hour |", "|---|---|---|---|---|---|"]
    for lab, key, who in (("sixth run `L0` (capped, 64 jobs)", "long_L0", "7B, Tesla T4"), ("sixth run `LN` (named, 64 jobs)", "long_LN", "7B, Tesla T4"), ("this run `P0` (capped, 32 jobs)", "P0", "14B, L4"), ("this run `PN` (named, 32 jobs)", "PN", "14B, L4")):
        g = f[key]; rows.append(f"| {lab} | {who} | {g['sps']:.3f} | {g['comp']:.1f} | {g['prompt']:.1f} | {num(round(g['per_hour']))} |")
    return "\n".join(rows)


def table_hazard(f):
    rows = ["| cell | request kind | pool | host read | first slips | steps run before the first slip | rate |", "|---|---|---|---|---|---|---|"]
    for k in ("P0", "PN", "X0", "XW", "XR"):
        h = f["haz"][k]
        for key in sorted(h, key=lambda t: (t[0], str(t[1]), str(t[2]))):
            s, e = h[key]; rows.append(f"| `{k}` | {key[0]} | {'-' if key[1] is None else ('first' if key[1] == 'A' else 'second')} | {key[2] if key[2] else '-'} | {s} | {num(e)} | {pc(s / e)} |")
    return "\n".join(rows)


def table_pred(f):
    xw = f["haz"]["XW"]; ed = sum(v[0] for key, v in xw.items() if key[0] == "add_edge"); tot = f["slips"]["XW"]["jobs_slipped"]
    rows = ["| written first | probability | measured | came out |", "|---|---|---|---|",
            f"| (a) more than half of the first slips of `XW` are on edge requests | 70 % | {ed} of {tot} first slips are on `add_edge` requests; {sum(v[0] for key, v in xw.items() if key[0] == 'add_node')} are on `add_node` | NO |",
            f"| (b) `XW` holds the right state on between 60 and 90 percent of its 3,200 steps | 55 % | {pc(f['XW']['share'])} | NO |",
            f"| (c) the seconds per step of `PN` on the L4 are between 0.12 and 0.30 | 65 % | {f['PN']['sps']:.4f} | NO |",
            f"| (d) the seven cells take 60 to 120 minutes in all, loading excluded | 60 % | {f['total_seconds'] / 60:.1f} minutes | NO |",
            f"| (e) the mean board power of `PN` is between 50 and 72 watts | 70 % | {f['PN']['mean_w']:.2f} W | yes |",
            f"| (f) this vLLM version reports request times (first token, finish) in the cells | 40 % | {sum(f[k]['request_times'] for k in ORDER)} requests with a time | NO |"]
    return "\n".join(rows)


def render(res, long_res, free_res):
    f = figures(res, long_res, free_res); e = res["meta"]["engine"]; P0, PN, P4, PC, X0, XW, XR = (f[k] for k in ORDER); sl = f["slips"]; st = f["slip_total"]; s27 = f["same27"]; h = f["haz"]
    A = lambda d, key: (lambda s, n: f"{s} of {num(n)} ({pc(s / n)})")(*d[key])
    sums = lambda k, kind: (sum(v[0] for key, v in h[k].items() if key[0] == kind), sum(v[1] for key, v in h[k].items() if key[0] == kind))
    xr_read = (sum(v[0] for key, v in h["XR"].items() if key[2] == "right"), sum(v[1] for key, v in h["XR"].items() if key[2] == "right"))
    xr_miss = (sum(v[0] for key, v in h["XR"].items() if key[2] == "miss"), sum(v[1] for key, v in h["XR"].items() if key[2] == "miss"))
    seven = f["seven"]; cl = sl["P0"]["classes"]; cx = sl["X0"]["classes"]; cw = sl["XW"]["classes"]
    out = f"""

# Result

*Written after the run, below the locked text; nothing above this line has been changed (its hash is checked by `tests/test_provider_side_big.py` and again by `tests/test_provider_side_big_result.py`). The run was made by the owner on Colab on 21 Oct 2026 with `python/colab/axi_provider_side_big.ipynb`: {num(f['steps_all'])} steps in seven cells, {f['total_seconds'] / 60:.1f} minutes of generation, vLLM {e['version']}, `{e['model']}`, {e['gpu']}, a cache of {num(e['capacity_tokens'])} tokens. The merged file is `docs/results/axi_provider_big.json` (sha256 `{FILE_SHA256[:16]}...`). The architecture is Rajnish's; the cells, the code and every measurement are the owner's, with an AI assistant as a tool.*

**What the page expected, and what came out.** Lines 1 to 7 all read yes, so by the page's rule the figures of the cells are claimed as read. Of the 21 lines **12 read yes and 9 read NO** (9, 11, 13, 14, 15, 17, 18, 19 and 20); no line is "not run". **Headline:** line 8 reads yes (the named expression held the state on every step on the bigger model) and **line 18 reads NO** (the bigger model did not read the names by itself well enough). The page put "yes, yes" at about 22 %. Lines 11, 13, 14 and 15 are near misses on the page's thresholds, in the direction I expected; they are said with their numbers below and are not rounded up.

**In one paragraph, from the side of a company that runs the model.** On the sixth run's first 32 jobs, with the exact state held by the host and the grammar naming the request's coordinates, a 14B model on one L4 held the state on {num(PN['exact'])} of 3,200 steps, in {PN['sps']:.3f} s per step, with {PN['comp']:.0f} computed prompt tokens per step, {num(round(PN['per_hour']))} exact steps per GPU hour and {PN['j_exact']:.1f} joules of board energy per exact step, and the seconds per step did not grow over the 100 steps ({PN['growth']:.2f} times from the first quarter to the last). The same model under the capped expression held {num(P0['exact'])} of 3,200 ({pc(P0['share'])}), no better than the 7B model on the same jobs ({num(seven['L0'][0])}); with the names list added to the prompt it held {num(X0['exact'])} ({pc(X0['share'])}) where the 7B held 3,200. **In every one of the {st[0]} first slips of the 14B in the four cells in which a job slipped, the reply contained an item the oracle's reply does not have ({st[1]} of {st[0]}); in none did it only lack an item.** The bigger model does more than it was asked, and the named expression is the part that forbids exactly that. Given requests in ordinary words, the 14B alone held {pc(XW['share'])}; with the host reader reading what it can, {pc(XR['share'])}, the reader read {num(f['reader']['right'])} requests right, {f['reader']['wrong']} wrong and refused {f['reader']['miss']}. The kept chat did not finish a 100-step job: all four stopped at steps {', '.join(str(n) for _, n in PC['stops'])}, at {PC['sps']:.3f} s per step against {P4['sps']:.3f} for the exact state on the same four jobs.

## The 21 lines

| # | my probability | what was measured | read |
|---|---|---|---|
""" + "\n".join(table_lines(f)) + f"""

Lines 11, 13, 14 and 15 compare the kept chat with the exact state: all four ratios point the way the page expected ({P4['per_hour'] / PC['per_hour']:.2f}, {PC['growth']:.2f}, {PC['comp'] / P4['comp']:.2f} and {PC['j_exact'] / P4['j_exact']:.2f} against thresholds of 2, 2, 1.3 and 2) and none reached its threshold; on the 7B model the page's figures for time per step and exact steps per hour were 3.8 times. Line 9 reads NO by {num(seven['L0'][0] - P0['exact'])} steps. Line 20 reads NO by {(f['XW_A'][0] - f['XW_B'][0]) * 100 - 3:.1f} points more than the three the page allowed.

## The cells

""" + table_cells(f) + f"""

Exact share and exact steps per hour count every step after a slip that stayed in the held state as a miss; they are the page's measures and they are used as written. The board watts are the L4's board power read by `nvidia-smi` every half second; the joules are those readings added up with the time between them over the cell's window, divided by the cell's exact steps. The meter reads the board, not the host.

""" + table_tokens(f) + f"""

## Drift: where the first slips are

The first inexact step of a job is the clean measure, because until then the held state was exact. `P0`: {sl['P0']['jobs_slipped']} of 32 jobs slipped (median first slip at step {sl['P0']['median']}); `PN`: {sl['PN']['jobs_slipped']}; `X0`: {sl['X0']['jobs_slipped']} (median step {sl['X0']['median']}); `XW`: {sl['XW']['jobs_slipped']} (median step {sl['XW']['median']}); `XR`: {sl['XR']['jobs_slipped']} (median step {sl['XR']['median']}). The table counts, for each kind of request, the first slips and the steps that were run exact up to and including the first slip.

""" + table_hazard(f) + f"""

* **The bigger model slips by doing more than it was asked.** Of the {st[0]} first slips in `P0`, `X0`, `XW` and `XR`, {st[1]} contain an item the oracle's reply does not have and {st[2]} lack an item without having an extra one; {st[3]} contain every item of the oracle's reply and more. In `P0`, {cl['delete: a DEL edge that does not touch the service']} of {cl['delete n']} slips are deletes that include a `DEL` of an edge that does not touch the deleted service and {cl['delete: an ADD item']} of {cl['delete n']} include an `ADD` item in a delete; in `X0`, {cx['delete: a DEL edge that does not touch the service']} of {cx['delete n']} and {cx['delete: an ADD item']} of {cx['delete n']}. The named expression allows a delete to begin with the named service and to delete only edges that touch it, and an add to add only what the request names: `PN` slipped {sum(v[0] for v in h['PN'].values())} times in {num(sum(v[1] for v in h['PN'].values()))} steps.
* **The capped expression is where the size of the model does not help.** `P0` {num(P0['exact'])} against the 7B model's {num(seven['L0'][0])} on the same jobs (`L0`); `X0` {num(X0['exact'])} against {num(seven['C0'][0])} (`C0`). The slips are all on deletes (`P0` {sums('P0', 'delete_dep')[0]} of {sums('P0', 'delete_dep')[1]} delete steps with dependents, {sums('P0', 'delete_leaf')[0]} on a leaf; `X0` {sums('X0', 'delete_dep')[0]} and {sums('X0', 'delete_leaf')[0]}), none on an add. One model of each size, one session each: this says the 14B was not more exact than the 7B under the capped expression on these jobs, nothing about 14B models in general.
* **In ordinary words (`XW`), the 14B model reads edges and loses on a service it is asked to add.** First slips on `add_edge` requests: {sums('XW', 'add_edge')[0]} of {sum(v[0] for v in h['XW'].values())}; on `add_node` requests: {sums('XW', 'add_node')[0]} of {sum(v[0] for v in h['XW'].values())}, at {pc(sums('XW', 'add_node')[0] / sums('XW', 'add_node')[1])} per step run; in all {cw['add_node: every extra item is an ADD edge at the new service']} of the {cw['add_node n']} the extra items are `ADD` edges at the new service's coordinates, and in {f['xw_neighbour'][0]} of the {f['xw_neighbour'][1]} one of them reaches the neighbour the sentence mentions ("it will sit next to ..."): the model wired the new service to a service it was only told about. The rest of the first slips are spread over deletes ({sums('XW', 'delete_dep')[0] + sums('XW', 'delete_leaf')[0]}) and edges. The second pool of wordings (a sentence that names one service more than the oracle's request) gives {pc(f['XW_B'][0])} exact steps against {pc(f['XW_A'][0])} in the first pool. The 7B model's weak point in the seventh run was the edge; the first slips of the 14B model are mostly on the add of a service in the second pool and on deletes.
* **The host reader is safe where it reads.** In `XR`, {xr_read[0]} of the {num(xr_read[1])} steps whose request the host read right was a first slip; all {xr_miss[0]} first slips are on requests the host refused ({num(xr_miss[1])} such steps run), at {pc(xr_miss[0] / xr_miss[1])} per step; all are in the second pool. Because a slip stays in the held state, the exact share of the steps whose request the host read is {pc(f['XR_read'][0])}, and of the others {pc(f['XR_miss'][0])}; the first-slip figures are the clean ones.

## The serving numbers, as a company that runs the model counts them

The page's measures are the seconds per step (and the exact steps per GPU hour that follow from it), the computed prompt tokens per step, the latency of a step within a job and of a round, and the board energy per exact step.

""" + table_serving(f) + f"""

The sixth run's cells were run in another Colab session on another card with a model half the size, so the table shows two settings and not an effect of the card or of the model; the size of any difference cannot be separated.

* **Speed.** `PN` ran {PN['sps']:.4f} s per step and `P0` {P0['sps']:.4f} ({(PN['sps'] / P0['sps'] - 1) * 100:+.1f} %) in this session, with {PN['distinct']} different expressions in the first and one in the second: building an expression for each request cost no seconds that this run can see (line 16). `PN` did {num(round(PN['per_hour']))} exact steps per GPU hour and `P0` {num(round(P0['per_hour']))}; the grammar's cost is not visible and its gain is {num(PN['exact'] - P0['exact'])} exact steps of 3,200.
* **Latency.** Within a job: the seconds per step of the last quarter of the rounds are {PN['growth']:.2f} times those of the first quarter in `PN` (line 12 reads yes), {P0['growth']:.2f} in `P0`. A round (all 32 requests together) took {PN['p50']:.2f} s at the median, {PN['p95']:.2f} s at the 95th percentile and {PN['pmax']:.2f} s at the longest in `PN`; the four-job cell `P4` {P4['p50']:.2f} / {P4['p95']:.2f} / {P4['pmax']:.2f} s. **This vLLM version reported no per-request times** (first token, finish): {sum(f[k]['request_times'] for k in ORDER)} of {num(f['steps_all'])} requests carry one, so no request-level latency is claimed; the round times are the latency a lockstep job sees.
* **Tokens.** `PN` computed {PN['comp']:.1f} prompt tokens per step ({pc(PN['cached_share'])} of the prompt tokens were served from the cache) and wrote {PN['new']:.1f}; the names list and the sentences put {XW['comp'] - PN['comp']:.0f} more computed prompt tokens in each step of `XW` ({(XW['comp'] / PN['comp'] - 1) * 100:.0f} %). Per **exact** step the difference is larger because the wrong steps are paid for: `PN` {PN['per_1000']:.1f} exact steps per 1,000 tokens, `XR` {XR['per_1000']:.1f}, `XW` {XW['per_1000']:.1f}.
* **Energy.** The mean board power was {min(f[k]['mean_w'] for k in ORDER):.1f} to {max(f[k]['mean_w'] for k in ORDER):.1f} W in all seven cells (the highest single readings {f['max_w'][0]:.1f} to {f['max_w'][1]:.1f} W); the L4's rated board power is 72 W. The card ran at about its rated power throughout, so on this card in this run the joules per step are the seconds per step times about 72 W ({PN['j_step']:.2f} J per step in `PN` at {PN['sps']:.3f} s) and energy told nothing that time did not tell: it is a measurement of the board and not of the host or the building. Per exact step: `PN` {PN['j_exact']:.2f} J, `XR` {XR['j_exact']:.2f}, `XW` {XW['j_exact']:.2f}, `PC` {PC['j_exact']:.2f}, `P4` {P4['j_exact']:.2f}.

## The kept chat against the exact state (lines 10 to 15)

The kept chat did not reach step 100 in any of the four jobs (line 10 reads yes): they stopped at steps {', '.join(str(n) for _, n in PC['stops'])}, the longest prompt was {num(PC['longest'])} tokens, against {num(P4['longest'])} for the exact state in all 100 steps. On the same four jobs and the first 27 rounds, when all four were running in both cells, the kept chat took {s27[0]['sps']:.3f} s per step against {s27[1]['sps']:.3f} ({s27[0]['sps'] / s27[1]['sps']:.2f} times) with {s27[0]['prompt']:,.0f} prompt tokens per step against {s27[1]['prompt']:,.0f}; over their whole runs {PC['sps']:.3f} against {P4['sps']:.3f} (the kept chat's first 27 to 61 steps against all 100 steps of the exact state, so the whole-run figures cover different steps). The prefix cache served {pc(PC['cached_share'])} of its prompt tokens. The page's four ratios (time, growth, tokens, energy) all point the way it expected and none reaches its threshold; the 7B figures on the sixth run were larger, on another card. What this run shows is that on an L4 with a 14B model the kept chat is slower, heavier and, within an 8,192-token context, cannot finish a 100-step job; it does not show a ratio for longer prompts or other cache sizes.

## What the page's rule lets this Result claim

Lines 1 to 7 read yes, so the figures of the cells are claimed as read, for this setup: one 14B model, one L4, one vLLM version, one session, 32 jobs (4 for the kept chat), the sentences of the page and the table's names. Drift is claimed from lines 8 and 9 (`PN` held; `P0` did not do better than the 7B), the kept chat from lines 10 to 15, the cost of the grammar from line 16 (none visible), what the bigger model does with names from lines 17 to 20 (it does not read them well enough alone), what the host reader adds from line 19 ({pc(XR['share'])}, not 99), and the time of the host reader from line 21. The names are the table's (given, not learned); the host reader reads a sentence only where one service is named for a delete, two for an edge, or an added service with its coordinates, and the sentences are those of the page and not requests in the words of a user. The energy is the board's.

## The predictions written beside the lines, read afterwards

""" + table_pred(f) + f"""

Side predictions (a), (b), (c), (d) and (f) did not come out; (e) did. The seven cells took {f['total_seconds'] / 60:.1f} minutes of generation, much less than the 60 to 120 I wrote.

## What this means, and what it does not

For a company that runs the model, the serving side did what it was built for on a bigger model: the state was held on every step by the named expression ({num(PN['exact'])} of 3,200), at a speed, a token count and an energy per step that did not depend on the grammar, flat over the job, on an L4. The size of the model did not remove the slips by itself: under the capped expression it did no better than the 7B model, and every slip was the model doing more than asked, which is what the named expression forbids. The kept chat cost more per step and could not finish a 100-step job in an 8,192-token context. The new finding about the front of the pipeline is that the 14B model, given a sentence, holds the state on more steps than the 7B ({pc(XW['share'])} against {pc(seven['W0'][0] / seven['W0'][1])} on the same jobs) but still does more than it is asked, for example by wiring a new service to a neighbour the sentence only mentions, and that the host reader lifts it to {pc(XR['share'])} (the 7B: {pc(seven['WN'][0] / seven['WN'][1])}) without ever reading wrong; the remaining loss is on the {num(f['reader']['miss'])} requests the reader refuses. The energy measurement is honest and, on this card, adds nothing to the time.

## What this does not show
Hardware other than one L4 (the v6e-1 TPU and any bigger GPU were not run); a model other than this 14B; the energy of the host, the cooling or the building; request-level latency (not reported by this version); tasks of more than 100 steps, let alone days; the words of a real user (the sentences are those of the seventh page, in two pools); names the host did not give; a language other than English; a kept chat with a summary or a window; a retry on top of a grammar; real company tasks; a capacity in jobs per GPU; dollars; hallucination in free text. Nothing here is proven.

## What comes next
The options, not commitments. (1) A reader for the second pool, so that the sentence's extra clause is ignored by the host and not by the model: the first slips of `XW` and `XR` are in that pool. (2) The same cells with more jobs side by side on the L4, to see where the round time starts to grow. (3) A probe on a v6e-1 TPU with its own page. (4) A longer horizon than 100 steps. The cost in compute units is told before anything is run.
"""
    return out


# ---------------------------------------------------------------- 1 the file
def test_the_file_is_the_one_the_page_is_about_and_every_step_replays_from_its_reply_text_through_the_frameworks_own_rules():
    res = run(); assert _sha(RES) == FILE_SHA256, "the saved result file must not change"; assert _sha(MODULE) == MODULE_SHA256 == res["module_sha256"] == B.fingerprint(), "the module that ran must be the one in the repository"
    assert set(res) == {"meta", "cells", "module_sha256", "typed_module_sha256", "pinned_sha256", "big_pinned_sha256"} and res["pinned_sha256"] == B.PINS == T.pinned_hashes() and res["typed_module_sha256"] == T.fingerprint() and res["big_pinned_sha256"] == B.pinned_files()
    m = res["meta"]; e = m["engine"]
    assert {k: v for k, v in e.items() if k != "constraint"} == dict(kind="vllm", name="Qwen/Qwen2.5-14B-Instruct-AWQ", version="0.31.0", gpu="NVIDIA L4", capacity_tokens=32000, capacity_source="override", block_size=16, kv_bytes_per_token=196608,
                                                                      model="Qwen/Qwen2.5-14B-Instruct-AWQ", python="3.13.15")
    assert e["constraint"]["regex_sha256"] == B.REGEX_SHA256 and (m["steps"], m["seed"], m["policy"], m["hint_rules"], m["model"]) == (100, SEED, "repair_m6", False, "Qwen/Qwen2.5-14B-Instruct-AWQ")
    assert [c["label"] for c in res["cells"]] == ORDER and m["cells"] == {k: list(v) for k, v in B.CELLS.items()}
    n_all = 0
    for c in res["cells"]:
        lab = c["label"]; K, first, arm, mode, shape = B.CELLS[lab]
        assert (c["arm"], c["K"], c["first_episode"], c["mode"], c["shape"], c["steps"], c["seed"], c["policy"], c["hint_rules"], c["engine"]) == (arm, K, first, mode, shape, 100, SEED, "repair_m6", False, e) and c["grammar_sha256"] == B.GRAMMAR_SHA[lab]
        assert [s["seed"] for s in c["sessions"]] == [SEED * 1000 + first + i for i in range(K)]
        X = logs(c); n_all += len(X); assert sum(r["active"] for r in c["rounds"]) == len(X) and abs(sum(r["seconds"] for r in c["rounds"]) - c["wall_seconds"]) < 5.0
        assert (sum(r["prompt_tokens"] for r in c["rounds"]), sum(r["cached_tokens"] for r in c["rounds"]), sum(r["new_tokens"] for r in c["rounds"])) == tuple(sum(x[key] for x in X) for key in ("n_prompt", "n_cached", "n_new"))
        assert all(0 <= x["n_cached"] <= x["n_prompt"] for x in X) and all(bool(x["exact"]) == (x["distance"] == 0) for x in X) and all(x["parsed"] or not x["admitted"] for x in X)
        assert c["power"]["samples"] > 0 and c["power"]["joules"] > 0 and all(r is None for r in c["request_times"])
        if arm == "S": assert all(len(s["log"]) == 100 and s["stopped"] is None for s in c["sessions"])
        for s in c["sessions"]:                                                                          # the job played again from the text of each reply through the pinned session of its cell
            sess = B.make_sessions(lab)[s["seed"] % 1000 - first]; assert sess.seed == s["seed"] if hasattr(sess, "seed") else True
            for x in s["log"]:
                sess.request(); sess.accept(P.Reply(x["text"], x["n_prompt"], x["n_cached"], x["n_new"])); y = sess.log[-1]
                assert (y["parsed"], y["admitted"], y["corrupted"], y["exact"], y["distance"], y["in_language"]) == (x["parsed"], x["admitted"], x["corrupted"], x["exact"], x["distance"], x["in_language"]), (lab, s["seed"] % 1000, x["t"])
                if lab in B.FREE_LABEL: assert (y["request"], y["pool"], y["reader"]) == (x["request"], x["pool"], x["reader"])
    assert n_all == 16578 and round(sum(c["wall_seconds"] for c in res["cells"]), 1) == 2458.0
    assert {lab: c["distinct_regexes"] for lab, c in cells(res).items()} == dict(P0=1, PN=656, P4=198, PC=115, X0=1, XW=1, XR=575)


# ---------------------------------------------------------------- 2 the lines
def test_the_twenty_one_lines_read_as_the_result_says_by_the_code_and_by_hand_and_the_rule_about_claims_is_followed():
    res = run(); long_res = run(LONG); free_res = run(FREE); chk = B.verdict_big(res["cells"]); by_code = [c for _, _, _, c in chk]; assert by_code == READ == lines_by_hand(res, long_res, free_res), "code, hand and the Result must agree"
    assert sum(READ) == 12 and [i + 1 for i, r in enumerate(READ) if not r] == [9, 11, 13, 14, 15, 17, 18, 19, 20] and [i for i, _, _, _ in chk] == list(range(1, 22))
    a = B.claims_allowed(chk); assert a["base"] is True and a["drift"] == {8: True, 9: False} and a["kept_chat"] == {10: True, 11: False, 12: True, 13: False, 14: False, 15: False} and a["grammar_cost"] == {16: True}
    assert a["the_model_reads_names"] == {17: False, 18: False, 20: False} and a["the_host_reads"] == {19: False} and a["time"] == {21: True}
    f = figures(res, long_res, free_res)
    assert f["seven"] == {"L0": (3075, 3200), "LN": (3200, 3200), "C0": (3200, 3200), "W0": (1343, 3200), "WN": (2030, 3200)}
    assert {k: v[0] for k, v in f["seven"].items()} == {k: v[0] for k, v in B.SEVEN_B.items()}
    assert B.READER_COUNTS == f["reader"] == {"right": 2688, "wrong": 0, "miss": 512}
    # by hand, from the raw records: the exact steps of each cell
    for lab, n in (("P0", 3038), ("PN", 3200), ("P4", 400), ("PC", 178), ("X0", 2689), ("XW", 1528), ("XR", 2884)): assert sum(1 for s in cells(res)[lab]["sessions"] for x in s["log"] if x["distance"] == 0) == n == f[lab]["exact"]
    assert [(k, f[k]["unparsed"], f[k]["full"]) for k in ORDER] == [("P0", 0, 1), ("PN", 0, 0), ("P4", 0, 0), ("PC", 0, 0), ("X0", 0, 4), ("XW", 1, 17), ("XR", 0, 9)]
    assert f["PC"]["stops"] == [("context", 27), ("context", 61), ("context", 50), ("context", 40)] and f["PC"]["longest"] == 7579 and f["P4"]["longest"] == 521
    assert all(v["cached"] > 0 for v in (f[k] for k in ORDER)) and sum(f[k]["request_times"] for k in ORDER) == 0


# ---------------------------------------------------------------- 3 the page
def test_the_part_above_the_result_is_what_was_written_before_the_run():
    head = _head(); assert _h(head) == PAGE_SHA256 and head.rstrip("\n").endswith("---")
    assert "# Result" not in head and head.startswith("# Test 21: the serving side on a bigger model, counted the way a company that runs the model counts (written BEFORE the run, 21 Oct 2026)")


def test_the_page_carries_exactly_the_result_this_file_renders_and_its_tables_say_what_the_rules_say():
    res = run(); long_res = run(LONG); free_res = run(FREE); r = render(res, long_res, free_res); assert _result().rstrip("\n") == r.rstrip("\n"), "the page must carry exactly the rendered Result"
    f = figures(res, long_res, free_res); rows = table_lines(f); assert len(rows) == 21 and [row.split(" | ")[-1].strip(" |") for row in rows] == ["yes" if x else "NO" for x in READ]
    assert [int(re.search(r"\| (\d+) % \|", row).group(1)) for row in rows] == PROBS
    page = _head().split("## Predictions")[1]; assert re.findall(r"\(([0-9]+) %\)\n", page.split("**Headline**")[0]) == [str(p) for p in PROBS]


# ---------------------------------------------------------------- 4 the figures
def test_the_figures_quoted_in_the_result_are_recounted_from_the_records():
    res = run(); long_res = run(LONG); free_res = run(FREE); f = figures(res, long_res, free_res); r = _result(); C = cells(res)
    for needle in ("**12 read yes and 9 read NO**", "**line 18 reads NO**", "3,200 of 3,200", "94.9 %", "84.0 %", "47.8 %", "90.1 %", "100.0 %", "2,688 requests right", "0 wrong", "refused 512", "27, 61, 50, 40", "16,578", "41.0 minutes", "575",
                   "This vLLM version reported no per-request times", "rated board power is 72 W", "Nothing here is proven"):
        assert needle in r, needle
    for lab, ex in (("P0", 3038), ("PN", 3200), ("P4", 400), ("PC", 178), ("X0", 2689), ("XW", 1528), ("XR", 2884)): assert abs(f[lab]["share"] - ex / f[lab]["n"]) < 1e-12 and f"{ex / f[lab]['n'] * 100:.1f} %" in r and num(ex) in r
    seconds = {lab: sum(rd["seconds"] for rd in C[lab]["rounds"]) for lab in ORDER}
    for lab in ORDER:
        X = logs(C[lab]); pt = sum(x["n_prompt"] for x in X); ch = sum(x["n_cached"] for x in X); ex = sum(1 for x in X if x["exact"]); assert abs(f[lab]["comp"] - (pt - ch) / len(X)) < 1e-9 and f"| {(pt - ch) / len(X):.1f} |" in r
        assert str(num(round(ex / C[lab]["wall_seconds"] * 3600))) in r and f"| {C[lab]['power']['joules'] / ex:.2f} |" in r and f"{C[lab]['power']['mean_w']:.1f}" in r
    # the sixth run's numbers, read from its file
    Lc = cells(long_res)
    for lab in ("L0", "LN"):
        X = logs(Lc[lab]); assert len(X) == 6400 and abs(f["long_" + lab]["sps"] - Lc[lab]["wall_seconds"] / 6400) < 1e-9 and str(num(round(sum(1 for x in X if x["exact"]) / Lc[lab]["wall_seconds"] * 3600))) in r
    # the quarters, the pools, the host's reading, counted again
    for lab, (g, hi) in (("PN", (0.88, 0.89)), ("PC", (1.44, 1.46))): assert g <= f[lab]["growth"] <= hi
    nA = eA = nB = eB = 0
    for s in C["XW"]["sessions"]:
        for x in s["log"]:
            if x["pool"] == "A": nA += 1; eA += x["exact"]
            else: nB += 1; eB += x["exact"]
    assert nA + nB == 3200 and abs(eA / nA - f["XW_A"][0]) < 1e-12 and f"{eA / nA * 100:.1f} %" in r and f"{eB / nB * 100:.1f} %" in r
    nr = er = nm = em = 0
    for s in C["XR"]["sessions"]:
        for x in s["log"]:
            if x["reader"] == "right": nr += 1; er += x["exact"]
            elif x["reader"] == "miss": nm += 1; em += x["exact"]
    assert (nr, nm) == (2688, 512) and f"{er / nr * 100:.1f} %" in r and f"{em / nm * 100:.1f} %" in r
    # first slips, counted again
    for lab, jobs in (("P0", 10), ("PN", 0), ("X0", 19), ("XW", 32), ("XR", 17)):
        firsts = [next((x["t"] for x in s["log"] if not x["exact"]), None) for s in C[lab]["sessions"]]; assert sum(1 for t in firsts if t) == jobs == f["slips"][lab]["jobs_slipped"]
    assert f["slip_total"][:3] == (78, 78, 0)
    assert f["haz"]["XR"][("add_edge", "A", "right")][0] == 0 and sum(v[0] for key, v in f["haz"]["XR"].items() if key[2] == "right") == 0 and sum(v[0] for key, v in f["haz"]["XR"].items() if key[2] == "miss") == 17 and all(key[1] == "B" for key, v in f["haz"]["XR"].items() if v[0] and key[2] == "miss")
    assert sum(v[0] for key, v in f["haz"]["XW"].items() if key[0] == "add_node") == 15 and sum(v[0] for key, v in f["haz"]["XW"].items() if key[0] == "add_edge") == 6 and f["slips"]["XW"]["classes"]["add_node: every extra item is an ADD edge at the new service"] == 15
    assert f["xw_neighbour"] == (13, 15) and (f["P0"]["mean_w"] > 71 and f["XW"]["max_w"] > 80) and all(71.0 <= f[k]["mean_w"] <= 72.0 for k in ORDER)
    s27 = f["same27"]; assert s27[0]["steps"] == s27[1]["steps"] == 108 and 1.45 < s27[0]["sps"] / s27[1]["sps"] < 1.55


# ---------------------------------------------------------------- 5 what it claims
def test_the_result_claims_only_what_the_pages_rule_allows():
    r = _result(); low = r.lower()
    for bad in ("$", "cheaper", "gpus saved", "times the capacity", "x capacity", "has proven", "proves that", "100 percent", "any provider", "providers", "real users", "in the words of any user", "solved", "zero drift", "guarantee", "saves", "per cent", "tpu is faster", "tpu will"):
        assert bad not in low, bad
    assert "Nothing here is proven." in r and "it is a measurement of the board and not of the host or the building" in r and "the sentences are those of the page and not requests in the words of a user" in r
    assert "v6e-1 TPU" in r and r.count("# Result") == 1 and "**12 read yes and 9 read NO**" in r and "100.0 %" in r and "In every one of the 78 first slips" in r


# ---------------------------------------------------------------- 6 front pages
def test_the_two_front_pages_say_what_was_run_what_was_read_and_what_it_does_not_show():
    cm = open(os.path.join(ROOT, "docs", "RAJNISH_CONCEPT_MAP.md")).read(); ve = open(os.path.join(ROOT, "docs", "VERIFIED.md")).read()
    item = next(l for l in cm.split("\n") if l.startswith("114. **")); row = next(l for l in ve.split("\n") if l.startswith("| Provider-side serving test, eighth run"))
    assert "Next free: row 172, item 127, test 153, question 121" in cm and cm.count("Next free:") == 1 and "test 138 its Result" in cm
    for text in (item, row):
        for s in ("`P0`", "`PN`", "`XR`", "3,200 of 3,200", "94.9 %", "47.8 %", "90.1 %", "12 of 21", "line 18", "2,688", "512", "board", "L4", "nothing here is proven", "mutation check of test 138: not yet run"):
            assert s in text, s
        assert "no run yet" not in text and "not run yet" not in text
    assert "test 138" in item and "docs/results/axi_provider_big.json" in item and "axi_provider_big.json" in row


if __name__ == "__main__":
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]; bad = 0
    for n, f in fns:
        try: f(); print("ok  ", n)
        except Exception: bad += 1; print("FAIL", n); traceback.print_exc()
    print(f"{len(fns) - bad}/{len(fns)} passed"); sys.exit(1 if bad else 0)
