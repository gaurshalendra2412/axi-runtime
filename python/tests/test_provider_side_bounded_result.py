"""Implements concept-map item 108: RESULT of the third provider-side serving run (the exact-state cells again, the length of a name capped at 16), made on Colab on 19 Oct 2026. Page: docs/PROVIDER_SIDE_BOUNDED_TEST.md
(the question, the 13 reading lines and the predictions were written before the run; the Result is appended below them).
The file: docs/results/axi_provider_bounded.json (the merged file the notebook's last cell wrote and the owner sent: `S` at K = 8, 32 and 64 and `KC` at K = 8; Tesla T4, vLLM 0.31.0, Qwen2.5-7B-Instruct-AWQ, a cache of 80,000 tokens, every
reply of both arms constrained by the capped expression), read against docs/results/axi_provider_constrained.json (the second run: its `KC` cells at 32 and 64 jobs are the kept chat of this comparison).
This file recomputes from the saved files, with its own arithmetic and not with the code that printed the figures:
  1. the files: their sha256, the merge (the merged file is what merge_cells makes of its cells, the module that merged it is the one on disk, the second run's file is the one it was compared with), the settings, the seeds,
     the same job script as the second run's, the rounds adding up to the cells, every cached count at most its prompt, and every step record replayed from its reply text (parsed, admitted, corrupted, exact, distance);
  2. the 13 reading lines (11 yes, lines 12 and 13 NO, none not run), the headline (lines 9 and 11 yes), by the code and by hand from the raw numbers, and what the page's rule lets the Result claim;
  3. the 'what was measured' text of each line, the tables and the figures quoted in the prose (the whole Result is `render(result)`, and the page must equal the locked head followed by it);
  4. the comparison with the second run, step by step: which replies differ (2, job 19 step 12), which fields differ (one more: a token count), the cached counts (identical), the speed (every cell slower by 4 to 8 percent), round 12;
  5. job 19 replayed with the capped reply: the gate refuses it, the job stays where it was;
  6. that the Result claims what the page's rule allows and nothing more (no multiple of capacity, no GPUs saved, no dollars, no 'proven', no plain time ratio at equal exactness).
What it does NOT show: anything about another GPU, model, engine or cache size, dollars, energy, retries, a random order of requests, or the kept chat at 32 and 64 jobs under the capped expression (it was not run)."""
import hashlib, json, os, re, sys

sys.path.insert(0, os.path.dirname(__file__))
import axi.experiments.provider_side as P
import axi.experiments.provider_side_constrained as C
import axi.experiments.provider_side_bounded as B
from axi.experiments.agent_loop import strict_parse

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..", "..")
PAGE = os.path.join(ROOT, "docs", "PROVIDER_SIDE_BOUNDED_TEST.md")
RES = os.path.join(ROOT, "docs", "results", "axi_provider_bounded.json")
SECOND = os.path.join(ROOT, "docs", "results", "axi_provider_constrained.json")
MODULE = os.path.join(HERE, "..", "axi", "experiments", "provider_side_bounded.py")
C_MODULE = os.path.join(HERE, "..", "axi", "experiments", "provider_side_constrained.py")
PS_MODULE = os.path.join(HERE, "..", "axi", "experiments", "provider_side.py")
RESULT_MARK = "\n\n# Result"
PAGE_SHA256 = "888358daf88abd0e11be3f368b67ab30c375fc116ce7445018158f77dbec2083"
FILE_SHA256 = "1f12a846b1325f9fed4e5df2cc755b3a7fd996266cb590311230f7cf418f2733"
SECOND_SHA256 = "7f54c3dd37fd4dcf61e8a7991af285d6ad1a0d691362496216949a26b45b819d"
MODULE_SHA256 = "7427a686d151c4294809aebaa9664d0b5fa420854817eb79e910372c3f57b6bc"
C_SHA256 = "03452148e28c9f31372f12cc11ef1808ac67308adff972fa19d4aeeef7c96f7d"
PS_SHA256 = "bfb4734e171d9244cd47d10402be46528a557095a8275d02b38481a0e2b0b1d3"
REGEX_SHA256 = "6f4e5a993f26ccd471bff13422fa622b7820858a91e05d43c2bccfbe6b864245"
PROBS = [98, 90, 85, 85, 88, 82, 72, 75, 92, 95, 95, 4, 3]
READ = [True, True, True, True, True, True, True, True, True, True, True, False, False]
ORDER = [("S", 8), ("KC", 8), ("S", 32), ("S", 64)]
SEED, STEPS = 20261017, 30
EST = {8: 3.5, 32: 40.4, 64: 49.9}                               # the page's estimate "against the kept chat", same speed as the second run
EST_EPH = {8: 20878, 32: 29407, 64: 33318}                      # the page's estimate of exact steps per GPU hour, same speed
LOOP_START = "ADD[(3,0)->(3,0):4tworowsdown4colsright4tworowsdown4colsrigh"
_cache = {}


def run():
    if "run" not in _cache: _cache["run"] = json.load(open(RES))
    return _cache["run"]


def second():
    if "second" not in _cache: _cache["second"] = json.load(open(SECOND))
    return _cache["second"]


def _text(): return open(PAGE).read()


def _head():
    t = _text(); assert t.count(RESULT_MARK) <= 1
    return t[:t.index(RESULT_MARK)] if RESULT_MARK in t else t


def _h(text): return hashlib.sha256((text.rstrip("\n") + "\n").encode()).hexdigest()


def _sha(path): return hashlib.sha256(open(path, "rb").read()).hexdigest()


# ---------------------------------------------------------------- the figures, from the step records
def pc(x, k=1): return f"{x * 100:.{k}f} %"
def num(x): return f"{x:,}"
def sg(x): return f"{x:+.1f} %".replace("+-", "-")
def cells(res): return {(c["arm"], c["K"]): c for c in res["cells"]}
def logs(c): return [x for s in c["sessions"] for x in s["log"]]
def mean(v): v = list(v); return sum(v) / len(v)
def secs(c): return sum(r["seconds"] for r in c["rounds"])


def st(c):
    L = logs(c); n = len(L); wall = secs(c); prompt = sum(x["n_prompt"] for x in L); cached = sum(x["n_cached"] for x in L)
    late = [x for s in c["sessions"] for x in s["log"][15:]]; ex = sum(1 for x in L if x["exact"])
    return dict(n=n, wall=wall, sps=wall / n, prompt=prompt, cached=cached, computed=prompt - cached, cps=(prompt - cached) / n, share=cached / prompt,
                late=sum(x["n_cached"] for x in late) / sum(x["n_prompt"] for x in late),
                q1=mean(x["n_prompt"] for s in c["sessions"] for x in s["log"][0:7]), q4=mean(x["n_prompt"] for s in c["sessions"] for x in s["log"][23:30]),
                longest=max(x["n_prompt"] for x in L), ex=ex, exact=ex / n, eph=3600 * ex / wall, jobs=len(c["sessions"]), whole=sum(all(x["exact"] for x in s["log"]) for s in c["sessions"]),
                corrupted=sum(x["corrupted"] for x in L), unparsed=sum(1 for x in L if not x["parsed"]), full96=sum(1 for x in L if x["n_new"] >= 96), longest_reply=max(x["n_new"] for x in L),
                new_tokens=sum(x["n_new"] for x in L), stopped=sum(1 for s in c["sessions"] if len(s["log"]) < 30))


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


def first_miss(c, job): return next((x for x in c["sessions"][job]["log"] if not x["exact"]), None)


def replay(arm, job, texts):
    """a job played again from the texts of its replies: (parsed, admitted, corrupted, exact, distance) of every step, and the gate's verdict on each parsed reply"""
    s = P.Session(arm, SEED * 1000 + job, STEPS); out = []
    for t in texts:
        s.request(); s.accept(P.Reply(t, 0, 0, 0))
    return [(x["parsed"], x["admitted"], x["corrupted"], x["exact"], x["distance"]) for x in s.log]


def capped_reply_gate(res):
    """the capped reply of job 19, step 12 in the 32-job cell, played against the state the job held: (the oracle's reply, gate ok, the gate's reason)"""
    log = cells(res)[("S", 32)]["sessions"][19]["log"]; s = P.Session("S", SEED * 1000 + 19, STEPS)
    while s.t < 11: rq = s.request(); s.accept(P.Reply(rq.stand_in_reply, 0, 0, 0))
    rq = s.request(); d = strict_parse(log[11]["text"]); ck = P.check(s.live, d); return rq.stand_in_reply, ck.ok, ck.reason


def vs_second(res):
    """step by step against the second run's cell of the same arm and K: the fields that differ, by field, and the positions of the differences"""
    new, old = cells(res), cells(second()); out = {}
    for k in ORDER:
        steps = same_text = 0; fields = {}; where = {}; jobs_len = 0
        for j, (a, b) in enumerate(zip(new[k]["sessions"], old[k]["sessions"])):
            jobs_len += len(a["log"]) != len(b["log"])
            for x, y in zip(a["log"], b["log"]):
                steps += 1; same_text += x["text"] == y["text"]
                for f in ("t", "kind", "text", "n_prompt", "n_cached", "n_new", "parsed", "admitted", "corrupted", "exact", "distance"):
                    if x[f] != y[f]: fields[f] = fields.get(f, 0) + 1; where.setdefault(f, []).append((j, x["t"]))
        out[k] = dict(steps=steps, same_text=same_text, fields=fields, where=where, jobs_len=jobs_len)
    return out


def speed(res):
    """seconds per step against the second run's, rounds that took longer, the sums"""
    new, old = cells(res), cells(second()); out = {}
    for k in ORDER:
        a, b = new[k], old[k]; n1, n2 = len(logs(a)), len(logs(b)); ra = [r["seconds"] for r in a["rounds"]]; rb = [r["seconds"] for r in b["rounds"]][:30]
        out[k] = dict(sps=secs(a) / n1, sps_old=secs(b) / n2, change=(secs(a) / n1) / (secs(b) / n2) - 1, slower=sum(1 for x, y in zip(ra, rb) if x > y), wall=secs(a), wall_old=secs(b), ra=ra, rb=rb)
    return out


def measured(res, i):
    s = S_(res); so = S_(second()); n = sum(v["n"] for v in s.values()); g = lambda a, k: s[(a, k)]; d = vs_second(res); sp = speed(res)
    kc = {8: g("KC", 8), 32: so[("KC", 32)], 64: so[("KC", 64)]}
    gap = lambda k: (kc[k]["exact"] - g("S", k)["exact"]) * 100
    ratio = lambda k: g("S", k)["eph"] / kc[k]["eph"]
    sc = [("S", 8), ("S", 32), ("S", 64)]
    cmp_steps = sum(d[k]["steps"] for k in sc); cmp_same = sum(d[k]["same_text"] for k in sc)
    gate = capped_reply_gate(run())
    return {
        1: f"the engine reported the cached tokens of every request: {num(n)} steps in the {len(s)} cells that ran",
        2: f"parsed: `S` {pc(1 - g('S', 8)['unparsed'] / g('S', 8)['n'])} (K = 8), {pc(1 - g('S', 32)['unparsed'] / g('S', 32)['n'])} (32), {pc(1 - g('S', 64)['unparsed'] / g('S', 64)['n'])} (64), `KC` {pc(1 - g('KC', 8)['unparsed'] / g('KC', 8)['n'])} (K = 8) (limit 99.5 %, met); replies that used all 96 new tokens: none in the four cells, the longest reply has {max(v['longest_reply'] for v in s.values())} tokens (limit: none, met)",
        3: f"`S` cells against the second run's, step by step: {num(cmp_steps)} replies compared, {num(cmp_same)} the same text; the {cmp_steps - cmp_same} that differ are job 19, step 12, in the 32-job and in the 64-job cell (allowed); every job took the same number of steps",
        4: f"`KC` at K = 8: {d[('KC', 8)]['same_text']} of {d[('KC', 8)]['steps']} replies the same text as the second run's, and {g('KC', 8)['ex']} of {g('KC', 8)['n']} steps exact ({pc(g('KC', 8)['exact'])})",
        5: f"job 19, step 12 (the 32-job and the 64-job cell, the same reply): `{cells(res)[('S', 32)]['sessions'][19]['log'][11]['text']}`, accepted by the strict parser, refused by the gate, {cells(res)[('S', 32)]['sessions'][19]['log'][11]['n_new']} new tokens (limit: fewer than 96; the second run's reply used 96)",
        6: f"exact steps of `S`: {g('S', 8)['ex']} of {g('S', 8)['n']} ({pc(g('S', 8)['exact'])}), {num(g('S', 32)['ex'])} of {g('S', 32)['n']} ({pc(g('S', 32)['exact'])}), {num(g('S', 64)['ex'])} of {num(g('S', 64)['n'])} ({pc(g('S', 64)['exact'])}); the second run's: {so[('S', 8)]['ex']}, {so[('S', 32)]['ex']}, {num(so[('S', 64)]['ex'])}; largest difference in the share {max(abs(g(a, k)['exact'] - so[(a, k)]['exact']) for (a, k) in sc) * 100:.1f} points (limit 0.5)",
        7: f"`S` seconds per step {sp[('S', 8)]['sps']:.3f}, {sp[('S', 32)]['sps']:.3f}, {sp[('S', 64)]['sps']:.3f} against the second run's {sp[('S', 8)]['sps_old']:.3f}, {sp[('S', 32)]['sps_old']:.3f}, {sp[('S', 64)]['sps_old']:.3f}: {sg(sp[('S', 8)]['change'] * 100)}, {sg(sp[('S', 32)]['change'] * 100)}, {sg(sp[('S', 64)]['change'] * 100)} (limit 30 % either way)",
        8: f"`KC` at K = 8: {sp[('KC', 8)]['sps']:.3f} seconds per step against the second run's {sp[('KC', 8)]['sps_old']:.3f}: {sg(sp[('KC', 8)]['change'] * 100)} (limit 30 % either way)",
        9: f"`S` {num(round(g('S', 8)['eph']))} exact steps per GPU hour ({pc(g('S', 8)['exact'])} exact), `KC` {num(round(kc[8]['eph']))} ({pc(kc[8]['exact'])}), both in this run: {ratio(8):.2f} times (limit 2)",
        10: f"`S` {num(round(g('S', 32)['eph']))} exact steps per GPU hour ({pc(g('S', 32)['exact'])} exact), `KC` {num(round(kc[32]['eph']))} ({pc(kc[32]['exact'])}), the second run's cell: {ratio(32):.1f} times (limit 10)",
        11: f"`S` {num(round(g('S', 64)['eph']))} exact steps per GPU hour ({pc(g('S', 64)['exact'])} exact), `KC` {num(round(kc[64]['eph']))} ({pc(kc[64]['exact'])}), the second run's cell: {ratio(64):.1f} times (limit 10)",
        12: f"`S` against `KC`: K = 8 {pc(g('S', 8)['exact'])} against {pc(kc[8]['exact'])}, {gap(8):.1f} points short; K = 32 {pc(g('S', 32)['exact'])} against {pc(kc[32]['exact'])}, {gap(32):.1f} points short; K = 64 {pc(g('S', 64)['exact'])} against {pc(kc[64]['exact'])}, {gap(64):.1f} points short (limit 5; met at K = 64 only)",
        13: f"`S` exact steps: {pc(g('S', 8)['exact'])} (K = 8), {pc(g('S', 32)['exact'])} (32), {pc(g('S', 64)['exact'])} (64): none reaches 98 %",
    }[i]


def table_lines(res):
    return [f"| {i} | {PROBS[i - 1]} % | {measured(res, i)} | {'yes' if READ[i - 1] else 'NO'} |" for i in range(1, 14)]


def table_cells(res):
    s = S_(res); out = []
    for (a, k) in ORDER:
        v = s[(a, k)]
        out.append(f"| `{a}` | {k} | {num(v['n'])} | {v['wall']:.1f} | {v['sps']:.3f} | {num(v['prompt'])} | {num(v['computed'])} | {pc(v['share'], 0)} | {pc(v['late'], 0)} | {num(round(v['q1']))} | {num(round(v['q4']))} | {num(v['longest'])} | {v['longest_reply']} | {pc(v['exact'])} |")
    return out


def table_hour(res):
    s = S_(res); so = S_(second()); out = []
    for (a, k) in ORDER:
        v = s[(a, k)]; w = so[(a, k)]
        out.append(f"| `{a}` | {k} | {num(v['ex'])} | {v['wall']:.1f} | {num(round(v['eph']))} | {pc(v['exact'])} | {num(round(w['eph']))} | {sg((v['eph'] / w['eph'] - 1) * 100)} |")
    return out


def table_ratio(res):
    s = S_(res); so = S_(second()); out = []
    for k in (8, 32, 64):
        kc = s[("KC", 8)] if k == 8 else so[("KC", k)]; a = s[("S", k)]; src = "this run (same session)" if k == 8 else "the second run (another Colab session)"
        out.append(f"| {k} | {a['eph'] / kc['eph']:.2f} | {EST[k]} | {pc(a['exact'])} | {pc(kc['exact'])} | {src} | {kc['sps'] / a['sps']:.2f} |")
    return out


def table_second(res):
    d = vs_second(res); sp = speed(res); s = S_(res); so = S_(second()); out = []
    for (a, k) in ORDER:
        x = d[(a, k)]; y = sp[(a, k)]
        out.append(f"| `{a}` | {k} | {num(x['steps'])} | {num(x['same_text'])} | {x['steps'] - x['same_text']} | {y['sps']:.4f} | {y['sps_old']:.4f} | {sg(y['change'] * 100)} | {y['slower']} of 30 | {y['wall']:.1f} | {y['wall_old']:.1f} | {num(s[(a, k)]['prompt'])} and {num(s[(a, k)]['cached'])} | {'the same' if (s[(a, k)]['prompt'], s[(a, k)]['cached']) == (so[(a, k)]['prompt'], so[(a, k)]['cached']) else 'different'} |")
    return out


def table_round12(res):
    sp = speed(res); new, old = cells(res), cells(second()); out = []
    for k in (32, 64):
        a, b = sp[("S", k)]["ra"], sp[("S", k)]["rb"]
        out.append(f"| {k} | {a[10]:.2f} / {a[11]:.2f} / {a[12]:.2f} | {b[10]:.2f} / {b[11]:.2f} / {b[12]:.2f} | {a[11] - b[11]:+.2f} | {sg((sum(a) - a[11]) / (sum(b) - b[11]) * 100 - 100)} | {new[('S', k)]['rounds'][11]['new_tokens']} against {old[('S', k)]['rounds'][11]['new_tokens']} |".replace("+-", "-"))
    return out


def table_misses(res):
    c = cells(res)[("S", 64)]; old = cells(second())[("S", 64)]; out = []
    for job in (3, 4, 17, 19, 20, 62):
        f = first_miss(c, job); g_ = first_miss(old, job); o = oracle_replies(job)[f["t"] - 1]
        out.append(f"| {job} | {f['t']} | `{o[0]}` | `{f['text']}` | `{o[1]}` | {'the same text' if f['text'] == g_['text'] else 'the second run wrote the loop (cut at 96 tokens); now a closed reply'} |")
    return out


def table_anatomy(res):
    s = S_(res); c = cells(res); so = S_(second()); out = []
    for (a, k) in ORDER:
        v = s[(a, k)]; missed = missed_jobs(c[(a, k)]); back = came_back(c[(a, k)])
        out.append(f"| `{a}` | {k} | {v['whole']} of {v['jobs']} (second run {so[(a, k)]['whole']}) | {v['unparsed']} (second run {so[(a, k)]['unparsed']}) | {v['full96']} (second run {so[(a, k)]['full96']}) | {', '.join(map(str, missed)) if missed else 'none'} | {', '.join(map(str, back)) if back else 'none'} | {v['corrupted']} |")
    return out


def figs(res):
    s = S_(res); so = S_(second()); sp = speed(res); d = vs_second(res); g = lambda a, k: s[(a, k)]
    kc = {8: s[("KC", 8)], 32: so[("KC", 32)], 64: so[("KC", 64)]}
    return dict(s=s, so=so, sp=sp, d=d, g=g, kc=kc, ratio=lambda k: g("S", k)["eph"] / kc[k]["eph"], wall=sum(v["wall"] for v in s.values()), n=sum(v["n"] for v in s.values()))


def render(res):
    F = figs(res); s, so, sp, d, g, kc, ratio = F["s"], F["so"], F["sp"], F["d"], F["g"], F["kc"], F["ratio"]; e = res["meta"]["engine"]
    oracle19, gate_ok, gate_reason = capped_reply_gate(res); log19 = cells(res)[("S", 32)]["sessions"][19]["log"]; log19_old = cells(second())[("S", 32)]["sessions"][19]["log"]
    cut = log19[11]["text"]; name = re.findall(r":([A-Za-z0-9_]*)#", cut)[0]
    est_dev = {k: g("S", k)["eph"] / EST_EPH[k] - 1 for k in (8, 32, 64)}
    rd = {k: (sp[("S", k)]["ra"], sp[("S", k)]["rb"]) for k in (32, 64)}
    other = {k: (sum(rd[k][0]) - rd[k][0][11]) / (sum(rd[k][1]) - rd[k][1][11]) - 1 for k in (32, 64)}
    gen = [x for x in (cells(second())[("S", 8)]["sessions"][3]["log"][15], cells(second())[("S", 32)]["sessions"][3]["log"][15], cells(second())[("S", 64)]["sessions"][3]["log"][15])]
    lines = []
    A = lines.append
    A("")
    A("")
    A("# Result")
    A("")
    A(f"*Written after the run, below the locked text; nothing above this line has been changed (its hash is checked by `tests/test_provider_side_bounded.py` and again by `tests/test_provider_side_bounded_result.py`). The run was made by the owner on Colab on 19 Oct 2026 with `python/colab/axi_provider_side_bounded.ipynb`: vLLM {e['version']}, GPU {e['gpu']}, Python {e['python']}, `{e['model']}`, the cache fixed at {num(e['capacity_tokens'])} tokens with the engine's block override (capacity source \"{e['capacity_source']}\"), 30-step jobs, seed {res['meta']['seed']}, every reply of both arms constrained by the capped expression of this page (sha256 `{REGEX_SHA256[:16]}…`). **All four planned cells ran, in one session and without a resume:** `S` at K = 8, 32 and 64 and `KC` at K = 8. The owner ran the safety cell before them and the four cells after it; what the safety cell printed is not in the saved file, and nothing below rests on it (the probe asks about the same step that line 5 reads in the two cells that hold job 19, step 12 of job 19). The file is `docs/results/axi_provider_bounded.json` (sha256 `{FILE_SHA256}`): the merged file that the notebook's last cell wrote with `provider_side_bounded.merge_files` and the owner sent. It holds the four cells; the four separate cell files were not sent. It records the sha256 of the module that merged it, `{MODULE_SHA256}`, which is the module in the repository, those of `provider_side.py` and `provider_side_constrained.py` (both pinned), and the second run's file it was compared with (`axi_provider_constrained.json`, `{SECOND_SHA256[:16]}…`). The GPU seconds of the four cells add up to {F['wall']:.1f}, which is {F['wall'] / 60:.1f} minutes, {g('S', 64)['wall']:.1f} of them in `S` at K = 64. Every figure below is computed from the step records and the round records of the two files by `tests/test_provider_side_bounded_result.py`.*")
    A("")
    A(f"**What the page expected, and what came out.** The page named the shape \"Lines 2 and 3 yes, lines 9 to 11 yes, lines 12 and 13 NO (the shape I expect)\" as \"the clean run for the claim\", and put exactly that pattern (lines 12 and 13 NO, every other line yes) at about 30 %. **That is what came out: 11 of the 13 lines read yes, lines 12 and 13 read NO, and no line is \"not run\".** The two NO are the two I put at 4 and 3 %; the cap could not move them, since it does not make job 19 right. The headline (lines 9 and 11) reads yes on both, and so does line 10 between them. Lines 2 and 3 both read yes (the page: about 80 %), and so do the three lines I trusted least among the yes (line 7 at 72 %, line 8 at 75 %, line 6 at 82 %).")
    A("")
    A("**Three things about the run that the page did not foresee or guessed differently.**")
    A(f"* **The new session was slower than the second run's in every cell.** Seconds per step were {sg(sp[('S', 8)]['change'] * 100)}, {sg(sp[('KC', 8)]['change'] * 100)}, {sg(sp[('S', 32)]['change'] * 100)} and {sg(sp[('S', 64)]['change'] * 100)} against the second run's in `S` at 8 jobs, `KC` at 8, `S` at 32 and `S` at 64, and the round took longer than the same round of the second run in {sp[('S', 8)]['slower']}, {sp[('KC', 8)]['slower']}, {sp[('S', 32)]['slower']} and {sp[('S', 64)]['slower']} of the 30 rounds. The page said a session would move things and put the line at 30 %; this one moved them by 4 to 8 %, in all four cells in the same direction. That is why the exact steps per GPU hour here are {sg(est_dev[8] * 100)[1:]}, {sg(est_dev[32] * 100)[1:]} and {sg(est_dev[64] * 100)[1:]} below the page's estimate for \"the same speed\" ({num(EST_EPH[8])}, {num(EST_EPH[32])}, {num(EST_EPH[64])}).")
    A(f"* **The wall seconds of the 32- and 64-job cells are not below the second run's (side prediction (c), 45 %); they are above, by {sp[('S', 32)]['wall'] - sp[('S', 32)]['wall_old']:.1f} and {sp[('S', 64)]['wall'] - sp[('S', 64)]['wall_old']:.1f} seconds.** The round that held job 19's step 12 took {rd[32][0][11]:.2f} and {rd[64][0][11]:.2f} seconds, {rd[32][1][11] - rd[32][0][11]:.2f} and {rd[64][1][11] - rd[64][0][11]:.2f} less than in the second run ({rd[32][1][11]:.2f} and {rd[64][1][11]:.2f}), while the other 29 rounds took {other[32] * 100:.1f} and {other[64] * 100:.1f} % more: the cap shortened that round, and the session's speed hid it in the total. How much of the {rd[32][1][11] - rd[32][0][11]:.2f} and {rd[64][1][11] - rd[64][0][11]:.2f} seconds is the cap and how much is noise in one round this run cannot separate.")
    A(f"* **One reply differs from the second run's in a field the page did not compare, and not in its text:** `S` at 64 jobs, job 3, step 16, `{gen[2]['text']}` is the same text with {cells(res)[('S', 64)]['sessions'][3]['log'][15]['n_new']} new tokens where the second run's cell at 64 jobs recorded {gen[2]['n_new']} (its cells at 8 and 32 jobs recorded {gen[0]['n_new']} and {gen[1]['n_new']}; now all three record {cells(res)[('S', 64)]['sessions'][3]['log'][15]['n_new']}). Nothing else of that step differs (prompt, cached count, parsed, admitted, exact, distance), no reading line reads that field, and I do not know why the count was {gen[2]['n_new']} once.")
    A("")
    A("## The 13 lines")
    A("")
    A("| # | my probability | what was measured | read |")
    A("|---|---|---|---|")
    lines.extend(table_lines(res))
    A("")
    A(f"**{sum(1 for x in READ if x)} of 13 lines read yes, lines 12 and 13 read NO, and no line is \"not run\".** The pattern \"lines 12 and 13 NO, every other line yes\" is the one the page called the most likely (about 30 %), and it is the one that came out. Lines 12 and 13 are the two NO that the page put at 4 and 3 %, so they are not surprises: `S` holds {pc(g('S', 8)['exact'])}, {pc(g('S', 32)['exact'])} and {pc(g('S', 64)['exact'])} of the right state at 8, 32 and 64 jobs, the kept chat {pc(kc[8]['exact'])}, and the cap never touched that.")
    A("")
    A("## What the page's rule lets this Result claim")
    A("The page's rule, applied line by line to what was read:")
    A("* **Lines 2 and 3 read yes**, so the exact-steps-per-hour lines (9 to 11) are claimed as read, with the exactness of both arms written next to every figure.")
    A(f"* **Lines 7 and 8 read yes**, so the point ratios of lines 10 and 11 may be quoted as numbers: {ratio(32):.1f} at 32 jobs and {ratio(64):.1f} at 64 jobs, each with the two sessions named (the exact-state cells are from the session of 19 Oct; the kept chat's cells at 32 and 64 jobs are the second run's, from another Colab session) and the exactness of both arms beside it. At 8 jobs both arms were measured in this session: {ratio(8):.2f}.")
    A("* **Line 12 reads NO**, so the plain time ratios (seconds per step of the kept chat over seconds per step of exact-state) are **not** claimed as ratios at equal exactness. They are in the table below, with the exactness of both arms beside them, because nothing is to be hidden.")
    A("* **Never**, as the page says: a multiple of capacity, a count of GPUs saved, dollars or a percentage of cost, \"proven\", a statement about any provider, or a multiple of speed without the exactness beside it.")
    A("")
    A(f"**The sentence the page allows, with this run's numbers.** On one Tesla T4 (Colab), with vLLM {e['version']}, `{e['model']}`, a cache of {num(e['capacity_tokens'])} tokens, 30-step jobs and every reply constrained by one regular expression, exact-state did {num(round(g('S', 8)['eph']))}, {num(round(g('S', 32)['eph']))} and {num(round(g('S', 64)['eph']))} exact steps per GPU hour at 8, 32 and 64 jobs, holding the right state after {pc(g('S', 8)['exact'])}, {pc(g('S', 32)['exact'])} and {pc(g('S', 64)['exact'])} of its steps, against {num(round(kc[8]['eph']))} (8 jobs, the same session), {num(round(kc[32]['eph']))} and {num(round(kc[64]['eph']))} (32 and 64 jobs, measured in the second run, another Colab session) for a kept chat that held it after {pc(kc[8]['exact'])} of its steps. The ratios of those figures are {ratio(8):.2f}, {ratio(32):.1f} and {ratio(64):.1f}.")
    A("")
    A(f"**On the number 49.3.** In the Result of the second run, 49.3 was the ratio at 64 jobs, and that Result withheld it because line 2 read NO. The same ratio under the capped expression, in a new session, is **{ratio(64):.1f}** ({num(round(g('S', 64)['eph']))} against {num(round(kc[64]['eph']))}); it is {abs(ratio(64) / (so[('S', 64)]['eph'] / kc[64]['eph']) - 1) * 100:.1f} % lower than 49.3 because the new session was slower for exact-state and the kept chat's figure is the second run's. It is a ratio of exact steps per GPU hour, with exact-state exact on {pc(g('S', 64)['exact'])} of its steps and the kept chat on {pc(kc[64]['exact'])}. **It is not a capacity.** The test sends K jobs in lockstep and counts exact steps per hour; it never looks for the number of jobs a GPU can hold, it measured no GPU other than one free T4, and it says nothing about how many GPUs anything would take. The same goes for {ratio(32):.1f} at 32 jobs.")
    A("")
    A("## Where my expectation was off, and where it was right")
    A(f"* **Line 2 (90 %): the cap held.** Every one of the {num(F['n'])} replies was accepted by the strict parser, none used the 96 new tokens, and the longest has {max(v['longest_reply'] for v in s.values())} tokens (in `S` at 32 and 64 jobs; the longest in `S` at 8 jobs and in `KC` is {g('S', 8)['longest_reply']}). The one reply that used all 96 in the second run is now {log19[11]['n_new']} tokens.")
    A(f"* **Line 3 (85 %): the replies are the second run's.** Of the {num(sum(d[k]['steps'] for k in ORDER if k[0] == 'S'))} replies of the three `S` cells, {num(sum(d[k]['same_text'] for k in ORDER if k[0] == 'S'))} are the same text as the second run's at the same job and step, and the {sum(d[k]['steps'] - d[k]['same_text'] for k in ORDER if k[0] == 'S')} that are not are job 19, step 12, in the 32- and the 64-job cell, the one step the page named. Apart from that step, nothing in these cells differs in text, prompt size, cached count, whether the reply parsed or was admitted, or whether it left the held state exact; at that step the text, the token count and \"parsed\" differ (admitted, exact and distance do not), and the one other difference is the token count of one reply (first section). **The engine's cache counts were identical in a second session: {sum(d[k]['fields'].get('n_cached', 0) for k in ORDER)} of the {num(F['n'])} steps have another cached-token count than the second run's.**")
    A(f"* **Line 4 (85 %) and the argument for not running the kept chat at 32 and 64 jobs again.** `KC` at 8 jobs gave the same {d[('KC', 8)]['steps']} replies as the second run's and was 100.0 % exact again. The page's argument was that a reply the narrower grammar still allows is not changed by it. In this run it held on every step where it applies: of the {num(F['n'])} replies the second run wrote at these jobs and steps, {num(F['n'] - 2)} are inside the capped language, and all {num(F['n'] - 2)} were written again as the same text, in a new session. That is one more case in which the argument held; it does not show what the kept chat would do at 32 and 64 jobs, which was not run.")
    A(f"* **Line 5 (88 %) and side prediction (a) (88 %): the capped reply is as the page said.** Job 19, step 12 is now `{cut}`: the first 16 characters of the looping name, `{name}`, and then `#4`, {log19[11]['n_new']} tokens (the page: fewer than 40). It begins `ADD[(3,0)->(3,0):` and ends with `]`. The oracle writes `{oracle19}` there. The gate refuses it (replayed here against the state the job held: \"{gate_reason}\"), so the state does not move. **Job 19 stayed where it was, as the page said it would:** it holds the right state after {sum(x['exact'] for x in log19)} of its 30 steps (steps 1 to 11) and after none of the others, the exact flags of every step are the second run's, the replies of steps 13 to 30 are the second run's, and so are their prompts.")
    A(f"* **Line 6 (82 %): the exact steps are the second run's, to the digit** ({g('S', 8)['ex']}, {g('S', 32)['ex']} and {num(g('S', 64)['ex'])}), and so are the six jobs that missed and the steps at which they first missed (table below). The cap cleared line 2 and nothing else, as the page said.")
    A(f"* **Lines 7 and 8 (72 and 75 %) read yes, by a wide margin:** {sg(sp[('S', 8)]['change'] * 100)}, {sg(sp[('S', 32)]['change'] * 100)}, {sg(sp[('S', 64)]['change'] * 100)} and {sg(sp[('KC', 8)]['change'] * 100)} against a limit of 30. What I did not know before the run, the spread of this Colab's T4 from one session to the next, is one observation now: 4.0 to 7.7 %, every cell slower.")
    A(f"* **Lines 9 to 11 (92, 95, 95 %) read yes.** The ratios came out {ratio(8):.2f}, {ratio(32):.1f} and {ratio(64):.1f} where the page's estimate for the second run's speed was {EST[8]}, {EST[32]} and {EST[64]}: {sg((ratio(8) / EST[8] - 1) * 100)}, {sg((ratio(32) / EST[32] - 1) * 100)} and {sg((ratio(64) / EST[64] - 1) * 100)}. The page said it was \"not a prediction of the figures\" and that the ratios I expect are \"the second run's, plus or minus the session\"; they are.")
    A(f"* **Lines 12 and 13 (4 and 3 %) read NO, as expected:** the gap to the kept chat is {(kc[8]['exact'] - g('S', 8)['exact']) * 100:.1f}, {(kc[32]['exact'] - g('S', 32)['exact']) * 100:.1f} and {(kc[64]['exact'] - g('S', 64)['exact']) * 100:.1f} points at 8, 32 and 64 jobs (limit 5, met at 64 jobs only), and none of the three `S` cells reaches 98 %.")
    A(f"* **The side predictions, read afterwards as descriptions.** (a) true, above (88 %). (b) The only replies that differ from the second run's in text, among the {num(F['n'])} steps of the four cells, are the two copies of job 19, step 12 (78 %): true; by token count there is one more, the 64-job cell's job 3, step 16 (first section). (c) The wall seconds of the 32- and 64-job cells below the second run's (45 %): **NO**, above by {sp[('S', 32)]['wall'] - sp[('S', 32)]['wall_old']:.1f} and {sp[('S', 64)]['wall'] - sp[('S', 64)]['wall_old']:.1f} seconds, because the session was slower; the round that held the loop was shorter by {rd[32][1][11] - rd[32][0][11]:.2f} and {rd[64][1][11] - rd[64][0][11]:.2f} seconds. (d) The whole run takes 25 to 30 minutes, 10 of them GPU time in the cells (70 %): the cells took {F['wall'] / 60:.1f} minutes of engine time; the length of the whole run (installs, the model load, the safety cell) is not in the files and was not timed, so the 25 to 30 cannot be read.")
    A("")
    A("## The tables")
    A("**Every cell** (the seconds are the wall seconds around the engine's call, summed over the rounds of the cell; \"computed\" is the prompt tokens the engine did not take from its cache; \"cached, 2nd half\" is the cached share over the steps from the 16th on; the two prompt columns are the mean prompt of the first 7 and of the last 7 steps of the jobs; \"longest reply\" is the most new tokens any reply used; \"exact\" is the share of steps after which the held state equalled the oracle's; engine vLLM 0.31.0 on a Tesla T4, no cell resumed):")
    A("")
    A("| arm | K | steps | seconds | s / step | prompt tokens | computed | cached | cached, 2nd half | prompt, first 7 | prompt, last 7 | longest prompt | longest reply | exact |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    lines.extend(table_cells(res))
    A("")
    A("**Exact steps per GPU hour** (3,600 times the exact steps over the wall seconds of the cell; the last two columns are the second run's figure for the same cell and the change):")
    A("")
    A("| arm | K | exact steps | seconds | exact steps per hour | exact | second run: per hour | change |")
    A("|---|---|---|---|---|---|---|---|")
    lines.extend(table_hour(res))
    A("")
    A("The kept chat at 32 and 64 jobs is not in this run's table because it was not run: its cells are the second run's (727 and 667 exact steps per hour, 100.0 % exact, 4,730.9 and 10,330.5 seconds; the 64-job cell was carried over three Colab sessions).")
    A("")
    A("**The ratios, with the exactness of both arms beside them** (exact steps per hour of `S` over those of `KC`; \"page's estimate\" is the estimate for the second run's speed that the page gave; the last column is the time per step of `KC` over that of `S`, **not** an equal-exactness ratio, because line 12 reads NO):")
    A("")
    A("| K | exact steps per hour, `S` over `KC` | page's estimate | exact, `S` | exact, `KC` | where the `KC` figure is from | time per step, `KC` over `S` (not an equal-exactness ratio) |")
    A("|---|---|---|---|---|---|---|")
    lines.extend(table_ratio(res))
    A("")
    A("**This run against the second run, step by step** (\"same text\" counts the replies that are the same text at the same job and step; the cached counts are the sum over the cell and whether every step has the same count as the second run's; \"rounds slower\" is the number of the 30 rounds that took longer than the same round of the second run):")
    A("")
    A("| arm | K | steps compared | same text | different | s / step | second run | change | rounds slower | seconds | second run | prompt and cached tokens | against the second run |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    lines.extend(table_second(res))
    A("")
    A("**The round that held job 19's step 12** (round 12; seconds of rounds 11 / 12 / 13 in this run and in the second run; \"other rounds\" is the change in the seconds of the other 29 rounds):")
    A("")
    A("| K | this run | second run | round 12, change in seconds | other rounds | new tokens in round 12 |")
    A("|---|---|---|---|---|---|")
    lines.extend(table_round12(res))
    A("")
    A("**Where the exactness went** (counted from the step records; \"jobs exact to the end\" are jobs whose held state equalled the oracle's after every step; \"refused\" are replies the strict parser refused; \"96 tokens\" are replies that used the whole limit; \"came back\" are jobs whose state differed at some step and equalled the oracle's again later; \"corrupted\" counts states that were not well formed. Job numbers start at 0):")
    A("")
    A("| arm | K | jobs exact to the end | refused | 96 tokens | jobs that missed | jobs that came back | corrupted |")
    A("|---|---|---|---|---|---|---|---|")
    lines.extend(table_anatomy(res))
    A("")
    A("**The six jobs of `S` that missed, and what they answered at their first miss** (steps count from 1; the oracle's reply is what `oracle_script` writes at that step for a job that is exact up to it; the replies are the same at every K in which the job is run):")
    A("")
    A("| job | step | kind of step | the engine answered | the oracle wrote | against the second run |")
    A("|---|---|---|---|---|---|")
    lines.extend(table_misses(res))
    A("")
    A("## What I read from it (derived afterwards; none of this was pre-registered)")
    A(f"1. **A second session moved the speed of this setup by 4 to 8 percent, in the same direction in every cell, and the figures follow it.** Every cell was slower (`S` at 8 jobs {sg(sp[('S', 8)]['change'] * 100)}, `KC` {sg(sp[('KC', 8)]['change'] * 100)}), and nearly every round of every cell: {sp[('S', 8)]['slower'] + sp[('KC', 8)]['slower'] + sp[('S', 32)]['slower'] + sp[('S', 64)]['slower']} of the 120 rounds of the four cells took longer than the same round of the second run. The exact steps and the cached counts are the same as the second run's, so the whole difference between its exact steps per hour and these ({sg((g('S', 8)['eph'] / so[('S', 8)]['eph'] - 1) * 100)}, {sg((g('S', 32)['eph'] / so[('S', 32)]['eph'] - 1) * 100)}, {sg((g('S', 64)['eph'] / so[('S', 64)]['eph'] - 1) * 100)} for `S`, {sg((kc[8]['eph'] / so[('KC', 8)]['eph'] - 1) * 100)} for `KC` at 8 jobs) is in the seconds.")
    A(f"2. **The ratio at 8 jobs, with both arms in one session, is {ratio(8):.2f}; in the second run it was {so[('S', 8)]['eph'] / so[('KC', 8)]['eph']:.2f}** ({sg((ratio(8) / (so[('S', 8)]['eph'] / so[('KC', 8)]['eph']) - 1) * 100)}). That is the one place where this run can say how much the ratio moves between two sessions: by a few percent, not by a factor. At 32 and 64 jobs the ratio mixes two sessions (exact-state from this one, the kept chat from the second run's). **An inference, not tested:** the new session was slower for both arms at 8 jobs, so a kept chat run in this session at 32 or 64 jobs would probably have been somewhat slower than the second run's, which would make these two ratios a little higher, not lower; this run does not show it.")
    A(f"3. **Exactness is where it was.** The capped expression changed the format of one reply and nothing of what the model got wrong: the same six jobs missed at the same steps, with the same replies (table above), and the exact shares are {pc(g('S', 8)['exact'])}, {pc(g('S', 32)['exact'])} and {pc(g('S', 64)['exact'])}. Refused replies of `S` (refused by the strict parser) are {s[('S', 8)]['unparsed']}, {s[('S', 32)]['unparsed']} and {s[('S', 64)]['unparsed']} at 8, 32 and 64 jobs (the second run: {so[('S', 8)]['unparsed']}, {so[('S', 32)]['unparsed']} and {so[('S', 64)]['unparsed']}); the reply that the parser refused is now a reply the parser reads and the gate refuses. The count stays conservative against `S`: a job that is wrong from step 12 on counts as nothing for the rest of its 30 steps.")
    sc = {k: rd[k][1][11] * (1 + other[k]) for k in (32, 64)}
    A(f"4. **What the cap saved in time is between about 1 and 2 seconds in a cell, and no closer than that can be read.** The round of the loop took {rd[32][0][11]:.2f} and {rd[64][0][11]:.2f} seconds in this run, {rd[32][1][11] - rd[32][0][11]:.2f} and {rd[64][1][11] - rd[64][0][11]:.2f} less than in the second run. The other 29 rounds took {other[32] * 100:.1f} and {other[64] * 100:.1f} % more in this session; if the second run's loop round is slowed the same way it would have taken {sc[32]:.2f} and {sc[64]:.2f} seconds, and the cap saved {sc[32] - rd[32][0][11]:.2f} and {sc[64] - rd[64][0][11]:.2f} (the page expected about 2.5, from the second run alone). That scaling assumes the loop round was slowed by the session like the others, which this run cannot check, and the noise of one round's seconds was not measured.")
    A("")
    A("## What this does and does not show")
    A(f"* **Shows, on one Tesla T4, one 7B model with 4-bit AWQ weights, vLLM {e['version']}, a cache of {num(e['capacity_tokens'])} tokens, 30-step jobs and every reply constrained by one regular expression with a name of at most 16 characters:** line 2 now reads yes, so the exact-steps-per-hour figures of this run are claimed as read; exact-state did {num(round(g('S', 8)['eph']))}, {num(round(g('S', 32)['eph']))} and {num(round(g('S', 64)['eph']))} exact steps per GPU hour at 8, 32 and 64 jobs (exact on {pc(g('S', 8)['exact'])}, {pc(g('S', 32)['exact'])} and {pc(g('S', 64)['exact'])} of its steps) against {num(round(kc[8]['eph']))}, {num(round(kc[32]['eph']))} and {num(round(kc[64]['eph']))} for the kept chat (exact on {pc(kc[8]['exact'])}); the kept chat's figures at 32 and 64 jobs are the second run's.")
    A("* **Does not show that exact-state is as exact as the kept chat** (lines 12 and 13 NO), and so no ratio of times is claimed as a ratio at equal exactness. It does not show the kept chat at 32 and 64 jobs under the capped expression (not run), a capacity in jobs per GPU, another GPU (the seconds are a free T4's, and a data-centre GPU is much faster; the ratios, not the seconds, are what might carry over, which this test does not show), another model, engine or cache size, dollars, **energy (never measured; only time)**, retries (this test sends one request a step and never asks again; a refused reply costs a retry in a real system, for either arm), traffic that is not a round of K jobs, a cache that moves rows to CPU memory or disk, jobs longer than 30 steps, or any statement about a provider. It says nothing about the tensor-level roadmap.")
    A("* **What may be reported now.** The sentence above, with its numbers, its setup and the exactness of both arms in it; the ratios as ratios of exact steps per GPU hour, naming the two sessions where the kept chat is the second run's; lines 12 and 13 as NO, in the same breath, as findings about this setup. **What stays out:** a multiple of capacity, a number of GPUs, dollars, \"proven\", and a ratio of times as if both arms were equally exact.")
    A("* **Not done and worth doing, in this order:** (1) the kept chat at 32 and 64 jobs run again in one session with the exact-state cells, so that every ratio compares one session (1 h 19 min and about 3 hours of GPU; it would also test the argument that the cap does not change the kept chat); (2) retries counted, so that a refused reply costs what it costs; (3) the same cells on another GPU; (4) the GPU's power read next to the seconds.")
    A("")
    A("## Reproduce")
    A(f"In Colab, `python/colab/axi_provider_side_bounded.ipynb` with a T4; about {F['wall'] / 60:.0f} minutes of engine time for the four cells, plus the installs and the model load. Nothing needs the GPU after that: `provider_side_bounded.merge_files` puts cell files together on any machine. `tests/test_provider_side_bounded_result.py` recomputes every figure of this Result from the saved file and the second run's, replays every job from the text of its replies, checks that the merged file is what `merge_cells` makes of its cells, and that the page is the locked text followed by this Result.")
    return "\n".join(lines)


# ================================================================ the tests
TOTALS = {("S", 8): (89589, 60112, 2526, 212, 240, 516), ("KC", 8): (831529, 784384, 2510, 240, 240, 7154), ("S", 32): (345237, 235504, 10180, 908, 960, 516),          # prompt, cached, new tokens, exact steps, steps, longest prompt
          ("S", 64): (670785, 462128, 20278, 1867, 1920, 516)}
QUOTES = ["Lines 2 and 3 yes, lines 9 to 11 yes, lines 12 and 13 NO (the shape I expect)", "the clean run for the claim", "not a prediction of the figures", "the second run's, plus or minus the session",
          "the argument that", "a reply whose every token the narrower grammar allows is the same reply", "The pattern I expect (lines 12 and 13 NO, every other line yes): about **30 %**", "Lines 2 and 3 both reading yes: about **80 %**"]


def _norm(t): return re.sub(r"\s+", " ", t)


def _result(): t = _text(); return t[t.index(RESULT_MARK):]


# ---------------------------------------------------------------- 1 the files
def test_the_files_are_the_ones_the_page_is_about_the_merge_is_what_the_code_makes_of_them_and_every_step_replays_from_its_reply():
    res = run(); assert _sha(RES) == FILE_SHA256, "the saved result file must not change"; assert _sha(SECOND) == SECOND_SHA256, "the second run's file must not change"
    assert _sha(MODULE) == MODULE_SHA256 == res["module_sha256"], "the module that merged the cells must be the one in the repository"
    assert _sha(C_MODULE) == C_SHA256 == res["constrained_sha256"] and _sha(PS_MODULE) == PS_SHA256 == res["provider_side_sha256"]
    assert set(res) == {"meta", "cells", "second_run_file", "second_run_sha256", "module_sha256", "constrained_sha256", "provider_side_sha256"}
    assert res["second_run_file"] == "axi_provider_constrained.json" and res["second_run_sha256"] == SECOND_SHA256
    assert B.REGEX_SHA256 == REGEX_SHA256 == hashlib.sha256(B.REGEX.encode()).hexdigest() and B.NAME_CAP == 16 and B.REGEX.count("{1,16}") == 2
    assert B.merge_cells(res["cells"]) == {k: res[k] for k in ("meta", "cells")}, "the merged file is what merge_cells makes of its own cells"
    e = res["meta"]["engine"]; con = e.pop("constraint")
    try:
        assert e == dict(kind="vllm", name="Qwen/Qwen2.5-7B-Instruct-AWQ", version="0.31.0", gpu="Tesla T4", capacity_tokens=80000, capacity_source="override", block_size=16, kv_bytes_per_token=57344, model="Qwen/Qwen2.5-7B-Instruct-AWQ", python="3.13.15")
        assert con["kind"] == "regex" and con["api"] == "structured_outputs(regex)" and con["regex"] == B.REGEX and con["regex_sha256"] == REGEX_SHA256 and con["name_cap"] == 16
    finally: e["constraint"] = con
    m = res["meta"]; assert (m["steps"], m["seed"], m["policy"], m["hint_rules"], m["name_cap"], m["model"]) == (30, SEED, "repair_m6", False, 16, "Qwen/Qwen2.5-7B-Instruct-AWQ") and m["cells"] == [list(k) for k in ORDER]
    old = cells(second()); assert [(c["arm"], c["K"]) for c in res["cells"]] == ORDER == B.cells_planned()
    assert all(c["engine"] == m["engine"] and c["steps"] == 30 and c["seed"] == SEED and c["policy"] == "repair_m6" and c["hint_rules"] is False and c["resumed_at"] == [] for c in res["cells"])
    for c in res["cells"]:
        o = old[(c["arm"], c["K"])]
        assert [s["seed"] for s in c["sessions"]] == [SEED * 1000 + i for i in range(c["K"])] and all(s["script_steps"] == 30 and s["stopped"] is None and len(s["log"]) == 30 for s in c["sessions"])
        assert [[x["kind"] for x in s["log"]] for s in c["sessions"]] == [[x["kind"] for x in s["log"]] for s in o["sessions"]], "the same job script as the second run's"
        L = logs(c); assert [x["t"] for s in c["sessions"] for x in s["log"]] == list(range(1, 31)) * c["K"]
        assert abs(secs(c) - c["wall_seconds"]) < 1e-6 and len(c["rounds"]) == 30 and all(r["round"] == i and r["active"] == c["K"] for i, r in enumerate(c["rounds"], 1))
        assert sum(r["prompt_tokens"] for r in c["rounds"]) == sum(x["n_prompt"] for x in L) and sum(r["cached_tokens"] for r in c["rounds"]) == sum(x["n_cached"] for x in L) and sum(r["new_tokens"] for r in c["rounds"]) == sum(x["n_new"] for x in L)
        assert all(0 <= x["n_cached"] <= x["n_prompt"] and isinstance(x["n_cached"], int) and 0 < x["n_new"] < 96 for x in L) and all(bool(x["exact"]) == (x["distance"] == 0) for x in L) and all(x["parsed"] or not x["admitted"] for x in L)
        assert all(re.fullmatch(B.REGEX, x["text"]) for x in L), "every reply is inside the capped language (the constraint held)"
        for j, s in enumerate(c["sessions"]):                                                                                                   # every step record is what the gate and the oracle make of the reply's text
            got = replay(c["arm"], j, [x["text"] for x in s["log"]])
            assert got == [(x["parsed"], x["admitted"], x["corrupted"], x["exact"], x["distance"]) for x in s["log"]], (c["arm"], c["K"], j)
    assert sum(len(logs(c)) for c in res["cells"]) == 3360 and sum(c["K"] for c in res["cells"]) * 30 == 3360 and round(sum(c["wall_seconds"] for c in res["cells"]), 1) == 527.9
    assert not os.path.exists(os.path.join(ROOT, "docs", "results", "provider_side_bounded_cells")), "the separate cell files were not sent; the Result says so"


# ---------------------------------------------------------------- 2 the rules
def test_the_thirteen_lines_read_as_the_page_says_the_headline_is_read_and_the_rule_about_claims_is_followed():
    res = run(); v = B.verdict_bounded(res["cells"], second()["cells"]); assert [x for _, _, _, x in v] == READ and len(PROBS) == 13 and B.BL_HEADLINE == (9, 11) and len(B.BL_LINES) == 13
    assert [(i, tag, text) for i, tag, text, _ in v] == [(i + 1, B.BL_LINES[i][0], B.BL_LINES[i][1]) for i in range(13)]
    assert sum(1 for x in READ if x is True) == 11 and [i + 1 for i, x in enumerate(READ) if x is False] == [12, 13] and [READ[i - 1] for i in B.BL_HEADLINE] == [True, True]
    # the same rules from the raw numbers, by hand
    s = S_(res); so = S_(second()); g = lambda a, k: s[(a, k)]; d = vs_second(res); sp = speed(res); kc = {8: g("KC", 8), 32: so[("KC", 32)], 64: so[("KC", 64)]}; allc = list(s.values())
    j19 = {k: cells(res)[k]["sessions"][19]["log"][11] for k in (("S", 32), ("S", 64))}
    hand = [all(x["n_cached"] is not None for c in res["cells"] for x in logs(c)),
            all(v_["unparsed"] / v_["n"] <= 0.005 for v_ in allc) and all(v_["full96"] == 0 for v_ in allc),
            all(all(j == 19 and t_ >= 12 for j, t_ in d[k]["where"].get("text", [])) and d[k]["jobs_len"] == 0 for k in ORDER if k[0] == "S"),
            d[("KC", 8)]["same_text"] == d[("KC", 8)]["steps"] and g("KC", 8)["exact"] == 1.0,
            all(r["parsed"] and not r["admitted"] and r["n_new"] < 96 for r in j19.values()),
            all(abs(g("S", k)["exact"] - so[("S", k)]["exact"]) <= 0.005 for k in (8, 32, 64)),
            all(abs(sp[("S", k)]["sps"] / sp[("S", k)]["sps_old"] - 1) <= 0.30 for k in (8, 32, 64)), abs(sp[("KC", 8)]["sps"] / sp[("KC", 8)]["sps_old"] - 1) <= 0.30,
            g("S", 8)["eph"] >= 2 * kc[8]["eph"], g("S", 32)["eph"] >= 10 * kc[32]["eph"], g("S", 64)["eph"] >= 10 * kc[64]["eph"],
            all(g("S", k)["exact"] >= kc[k]["exact"] - 0.05 for k in (8, 32, 64)), all(g("S", k)["exact"] >= 0.98 for k in (8, 32, 64))]
    for i in range(13): assert hand[i] == READ[i], (i + 1, hand[i], READ[i])
    assert len(s) == 4
    # the rule that decides what the Result may claim: lines 2 and 3 yes -> 9 to 11 claimed; lines 7 and 8 yes -> point ratios; line 12 NO -> no plain time ratio at equal exactness
    ca = B.claims_allowed(v); assert ca == dict(constraint_held=True, replies_as_in_second_run=True, base=True, quality_adjusted={9: True, 10: True, 11: True}, point_ratios=True, plain_time_at_equal_exactness=False)
    R = _result(); assert "**Lines 2 and 3 read yes**" in R and "**Lines 7 and 8 read yes**" in R and "**Line 12 reads NO**" in R and "are **not** claimed as ratios at equal exactness" in R
    # the probabilities on the page are the ones this file uses
    assert [int(m.group(1)) for m in re.finditer(r"\n\d+\. \[\w+\] .*\((\d+) %\)(?=\n)", _head())] == PROBS


# ---------------------------------------------------------------- 3 the page
def test_the_part_above_the_result_is_what_was_written_before_the_run():
    head = _head(); assert _h(head) == PAGE_SHA256 and _text().count(RESULT_MARK) == 1 and len(head) in (22528, 22529) and head.rstrip("\n").endswith("\n\n---")


def test_the_page_carries_exactly_the_result_this_file_renders_and_its_table_says_what_the_rules_say():
    res = run(); text = _text(); assert text == _head().rstrip("\n") + render(res).rstrip("\n") + "\n"
    R = _result(); rows = [ln for ln in R.split("\n") if re.match(r"\| \d+ \| \d+ % \|", ln)]; assert len(rows) == 13
    for i, ln in enumerate(rows, 1):
        cells_ = [c.strip() for c in ln.strip("|").split("|")]; assert int(cells_[0]) == i and cells_[1] == f"{PROBS[i - 1]} %" and cells_[2] == measured(res, i) and cells_[3] == ("yes" if READ[i - 1] else "NO"), i
    for tbl in (table_cells(res), table_hour(res), table_ratio(res), table_second(res), table_round12(res), table_anatomy(res), table_misses(res)):
        for ln in tbl: assert ln in R.split("\n"), ln
        assert "\n".join(tbl) in R
    for q in QUOTES[:4] + QUOTES[5:]: assert _norm(q) in _norm(_head()), q                                                                       # what the Result quotes of the page is on the page, word for word
    for q in QUOTES[:4]: assert _norm(q) in _norm(R), q


# ---------------------------------------------------------------- 4 the figures
def test_the_figures_quoted_in_the_result_are_the_ones_in_the_files():
    res = run(); R = _result(); c = cells(res)
    for needle in ("vLLM 0.31.0", "GPU Tesla T4", "Python 3.13.15", FILE_SHA256, MODULE_SHA256, SECOND_SHA256[:16], REGEX_SHA256[:16], "19 Oct 2026", "no cell resumed", "3,360 steps in the 4 cells that ran", "add up to 527.9, which is 8.8 minutes, 217.4 of them in `S` at K = 64",
                   "88.3 %", "94.6 %", "97.2 %", "19,379", "27,390", "30,914", "5,694", "727", "667", "3.40", "37.7", "46.3", "46.32", "37.65", "0.164", "0.124", "0.113", "0.632", "0.152", "0.118", "0.106", "0.608", "+7.7 %", "+5.0 %", "+6.4 %", "+4.0 %",
                   "30, 30, 29 and 29 of the 30 rounds", "118 of the 120 rounds", "7.2 %, 6.9 % and 7.2 % below", "20,878, 29,407, 33,318", "by 5.6 and 13.1 seconds", "4.56 and 8.11 seconds, 1.56 and 1.01 less", "(6.13 and 9.13)", "+6.7 %", "+7.2 %", "364 against 439", "728 against 803",
                   "3,118 are the same text", "3,120 replies", "0 of the 3,360 steps have another cached-token count", "all 3,358 were written again as the same text", "-2.8 %, -6.8 % and -7.2 %", "3.52", "(-3.4 %)", "(the second run: 0, 1 and 1)",
                   "ADD[(3,0)->(3,0):4tworowsdown4col#4]", "`4tworowsdown4col`", "21 tokens", "`ADD[3,0:4]`", "endpoint missing", "11 of its 30 steps", "16 new tokens where the second run's cell at 64 jobs recorded 17", "recorded 16 and 16",
                   "6.54 and 9.79 seconds", "1.97 and 1.67", "11 of 13 lines read yes", "11.7, 5.4 and 2.8 points", "8.8 minutes of engine time", "(1 h 19 min and about 3 hours of GPU", "6.0 % lower than 49.3", "It is not a capacity."):
        assert needle in R, needle
    # recomputed here from the raw records with plain arithmetic
    for k, (prompt, cached, new, ex, steps, longest) in TOTALS.items():
        L = logs(c[k]); assert (sum(x["n_prompt"] for x in L), sum(x["n_cached"] for x in L), sum(x["n_new"] for x in L), sum(1 for x in L if x["exact"]), len(L), max(x["n_prompt"] for x in L)) == (prompt, cached, new, ex, steps, longest), k
    wall = {k: sum(r["seconds"] for r in c[k]["rounds"]) for k in TOTALS}; assert [round(wall[k], 1) for k in ORDER] == [39.4, 151.7, 119.3, 217.4]
    eph = {k: 3600 * TOTALS[k][3] / wall[k] for k in TOTALS}; assert [round(eph[k]) for k in ORDER] == [19379, 5694, 27390, 30914]
    old = cells(second()); oe = {k: 3600 * sum(x["exact"] for x in logs(old[k])) / sum(r["seconds"] for r in old[k]["rounds"]) for k in old}; assert [round(oe[k]) for k in (("S", 8), ("KC", 8), ("S", 32), ("KC", 32), ("S", 64), ("KC", 64))] == [20878, 5923, 28748, 727, 32896, 667]
    assert [f"{eph[('S', 8)] / eph[('KC', 8)]:.2f}", f"{eph[('S', 32)] / oe[('KC', 32)]:.2f}", f"{eph[('S', 64)] / oe[('KC', 64)]:.2f}"] == ["3.40", "37.65", "46.32"]
    assert [f"{eph[('S', 8)] / eph[('KC', 8)]:.1f}", f"{eph[('S', 32)] / oe[('KC', 32)]:.1f}", f"{eph[('S', 64)] / oe[('KC', 64)]:.1f}"] == ["3.4", "37.7", "46.3"] and f"{oe[('S', 8)] / oe[('KC', 8)]:.2f}" == "3.52" and f"{oe[('S', 64)] / oe[('KC', 64)]:.1f}" == "49.3"
    assert f"{(eph[('S', 64)] / oe[('KC', 64)]) / (oe[('S', 64)] / oe[('KC', 64)]) * 100 - 100:.1f}" == "-6.0"
    assert [f"{eph[('S', k)] / EST_EPH[k] * 100 - 100:.1f}" for k in (8, 32, 64)] == ["-7.2", "-6.9", "-7.2"] and [f"{(eph[('S', 8)] / eph[('KC', 8)]) / EST[8] * 100 - 100:.1f}", f"{(eph[('S', 32)] / oe[('KC', 32)]) / EST[32] * 100 - 100:.1f}", f"{(eph[('S', 64)] / oe[('KC', 64)]) / EST[64] * 100 - 100:.1f}"] == ["-2.8", "-6.8", "-7.2"]
    assert round(sum(wall.values()) / 60, 1) == 8.8 and round(sum(wall.values()), 1) == 527.9
    h = _head()                                                                                                                                  # the page's own estimate table is the one this file uses
    for needle in ("| 36.6 (job 19 is not in it) | 20,878 | 3.5 (`KC` 5,923) |", "| 111.2 | 29,407 | 40.4 (`KC` 727) |", "| 201.7 | 33,318 | 49.9 (`KC` 667) |"): assert needle in h, needle


# ---------------------------------------------------------------- 5 against the second run, step by step
def test_the_replies_the_cached_counts_and_the_speed_against_the_second_run_are_counted_from_the_step_and_round_records():
    res = run(); d = vs_second(res); sp = speed(res)
    assert {k: (d[k]["steps"], d[k]["same_text"]) for k in ORDER} == {("S", 8): (240, 240), ("KC", 8): (240, 240), ("S", 32): (960, 959), ("S", 64): (1920, 1919)}
    assert all(d[k]["jobs_len"] == 0 for k in ORDER) and sum(d[k]["steps"] for k in ORDER) == 3360
    assert d[("S", 8)]["fields"] == {} and d[("KC", 8)]["fields"] == {}                                                                         # nothing differs in the 8-job cells, in any field, cached counts included
    assert d[("S", 32)]["fields"] == {"text": 1, "n_new": 1, "parsed": 1} and d[("S", 32)]["where"] == {"text": [(19, 12)], "n_new": [(19, 12)], "parsed": [(19, 12)]}
    assert d[("S", 64)]["fields"] == {"text": 1, "n_new": 2, "parsed": 1} and d[("S", 64)]["where"] == {"text": [(19, 12)], "n_new": [(3, 16), (19, 12)], "parsed": [(19, 12)]}
    assert not any("n_cached" in d[k]["fields"] or "n_prompt" in d[k]["fields"] or "admitted" in d[k]["fields"] or "exact" in d[k]["fields"] or "distance" in d[k]["fields"] for k in ORDER)
    new, old = cells(res), cells(second())                                                                                                       # the three records of job 3, step 16, in both runs
    assert [old[("S", k)]["sessions"][3]["log"][15]["n_new"] for k in (8, 32, 64)] == [16, 16, 17] and [new[("S", k)]["sessions"][3]["log"][15]["n_new"] for k in (8, 32, 64)] == [16, 16, 16]
    assert len({new[("S", k)]["sessions"][3]["log"][15]["text"] for k in (8, 32, 64)} | {old[("S", k)]["sessions"][3]["log"][15]["text"] for k in (8, 32, 64)}) == 1
    rx = re.compile(B.REGEX); inside = outside = 0                                                                                               # the argument: a reply inside the narrower language is written again unchanged
    for k in ORDER:
        for a, b in zip(new[k]["sessions"], old[k]["sessions"]):
            for x, y in zip(a["log"], b["log"]):
                if rx.fullmatch(y["text"]): inside += 1; assert x["text"] == y["text"]
                else: outside += 1
    assert (inside, outside) == (3358, 2)
    # speed
    assert [round(sp[k]["change"] * 100, 1) for k in ORDER] == [7.7, 4.0, 5.0, 6.4] and [sp[k]["slower"] for k in ORDER] == [30, 30, 29, 29]
    assert [(round(sp[k]["sps"], 4), round(sp[k]["sps_old"], 4)) for k in ORDER] == [(0.1641, 0.1523), (0.6323, 0.6078), (0.1243, 0.1184), (0.1132, 0.1064)]
    assert all(len(sp[k]["rb"]) == 30 for k in ORDER) and len(old[("S", 8)]["rounds"]) == 30
    ra, rb = sp[("S", 32)]["ra"], sp[("S", 32)]["rb"]; assert [round(x, 2) for x in (ra[10], ra[11], ra[12], rb[10], rb[11], rb[12])] == [4.06, 4.56, 3.60, 3.79, 6.13, 3.37]
    ra, rb = sp[("S", 64)]["ra"], sp[("S", 64)]["rb"]; assert [round(x, 2) for x in (ra[10], ra[11], ra[12], rb[10], rb[11], rb[12])] == [7.27, 8.11, 6.81, 6.77, 9.13, 6.32]
    for K, (o_, n_) in {32: (439, 364), 64: (803, 728)}.items(): assert (old[("S", K)]["rounds"][11]["new_tokens"], new[("S", K)]["rounds"][11]["new_tokens"]) == (o_, n_)
    assert [round((sum(sp[("S", K)]["ra"]) - sp[("S", K)]["ra"][11]) / (sum(sp[("S", K)]["rb"]) - sp[("S", K)]["rb"][11]) * 100 - 100, 1) for K in (32, 64)] == [6.7, 7.2]
    assert [round(sp[("S", K)]["wall"] - sp[("S", K)]["wall_old"], 1) for K in (32, 64)] == [5.6, 13.1] and sum(sp[k]["slower"] for k in ORDER) == 118
    # the page's side prediction (c): "below the second run's" is false in both cells; (d): the cells took 8.8 minutes
    assert all(sp[("S", K)]["wall"] > sp[("S", K)]["wall_old"] for K in (32, 64))


def test_job_19_stays_where_it_was_the_gate_refuses_the_capped_reply_and_every_name_of_16_characters_is_refused_the_same_way():
    res = run(); c = cells(res); old = cells(second())
    oracle, ok, reason = capped_reply_gate(res); assert oracle == "ADD[3,0:4]" and ok is False and reason == "well-formedness: edge ((3, 0), (3, 0), '4tworowsdown4col') endpoint missing", reason
    for K in (32, 64):
        n, o = c[("S", K)]["sessions"][19]["log"], old[("S", K)]["sessions"][19]["log"]
        assert n[11]["text"] == "ADD[(3,0)->(3,0):4tworowsdown4col#4]" and o[11]["text"].startswith(LOOP_START) and o[11]["text"][len("ADD[(3,0)->(3,0):"):][:16] == "4tworowsdown4col" and len(n[11]["text"]) < 40
        assert (n[11]["parsed"], n[11]["admitted"], n[11]["n_new"], o[11]["parsed"], o[11]["admitted"], o[11]["n_new"]) == (True, False, 21, False, False, 96)
        assert [x["exact"] for x in n] == [x["exact"] for x in o] == [True] * 11 + [False] * 19 and [x["n_prompt"] for x in n] == [x["n_prompt"] for x in o] and [x["n_cached"] for x in n] == [x["n_cached"] for x in o]
        assert [x["text"] for x in n[12:]] == [x["text"] for x in o[12:]] and [x["text"] for x in n[:11]] == [x["text"] for x in o[:11]] and sum(x["admitted"] for x in n[12:]) == sum(x["admitted"] for x in o[12:]) == 6
        assert [x["distance"] for x in n] == [x["distance"] for x in o] and [x["admitted"] for x in n] == [x["admitted"] for x in o]
    s = P.Session("S", SEED * 1000 + 19, STEPS)
    while s.t < 11: rq = s.request(); s.accept(P.Reply(rq.stand_in_reply, 0, 0, 0))
    s.request()
    for kind in ("ADD", "DEL"):
        for name in ("4tworowsdown4col", "dep", "owns", "x" * 16):
            t = f"{kind}[(3,0)->(3,0):{name}#4]"; assert re.fullmatch(B.REGEX, t) and not P.check(s.live, strict_parse(t)).ok
    # the six jobs that missed are the second run's six, at the steps the second run gave, in the three cells in which they are run
    miss = {k: missed_jobs(c[("S", k)]) for k in (8, 32, 64)}; assert miss == {8: [3, 4], 32: [3, 4, 17, 19, 20], 64: [3, 4, 17, 19, 20, 62]} == {k: missed_jobs(old[("S", k)]) for k in (8, 32, 64)} and not missed_jobs(c[("KC", 8)])
    assert [(j, first_miss(c[("S", 64)], j)["t"]) for j in miss[64]] == [(3, 16), (4, 18), (17, 20), (19, 12), (20, 30), (62, 12)] and [(j, first_miss(old[("S", 64)], j)["t"]) for j in miss[64]] == [(3, 16), (4, 18), (17, 20), (19, 12), (20, 30), (62, 12)]
    assert [came_back(c[("S", k)]) for k in (8, 32, 64)] == [[], [17], [17, 62]] == [came_back(old[("S", k)]) for k in (8, 32, 64)]
    assert [sum(1 for x in logs(c[k]) if not x["parsed"]) for k in ORDER] == [0, 0, 0, 0] and [sum(1 for x in logs(old[k]) if not x["parsed"]) for k in ORDER] == [0, 0, 1, 1] and [sum(x["corrupted"] for x in logs(c[k])) for k in ORDER] == [0, 0, 0, 0]


# ---------------------------------------------------------------- 5b the prose, word for word
GOLDEN = [   # every line of the Result that is not a row of a data table, as it stands on the page (numbers included); the rows of the data tables are recomputed from the files, and a change in any of these lines has to be made here too
    '# Result',
    '*Written after the run, below the locked text; nothing above this line has been changed (its hash is checked by `tests/test_provider_side_bounded.py` and again by `tests/test_provider_side_bounded_result.py`). The run was made by the owner on Colab on 19 Oct 2026 with `python/colab/axi_provider_side_bounded.ipynb`: vLLM 0.31.0, GPU Tesla T4, Python 3.13.15, `Qwen/Qwen2.5-7B-Instruct-AWQ`, the cache fixed at 80,000 tokens with the engine\'s block override (capacity source "override"), 30-step jobs, seed 20261017, every reply of both arms constrained by the capped expression of this page (sha256 `6f4e5a993f26ccd4…`). **All four planned cells ran, in one session and without a resume:** `S` at K = 8, 32 and 64 and `KC` at K = 8. The owner ran the safety cell before them and the four cells after it; what the safety cell printed is not in the saved file, and nothing below rests on it (the probe asks about the same step that line 5 reads in the two cells that hold job 19, step 12 of job 19). The file is `docs/results/axi_provider_bounded.json` (sha256 `1f12a846b1325f9fed4e5df2cc755b3a7fd996266cb590311230f7cf418f2733`): the merged file that the notebook\'s last cell wrote with `provider_side_bounded.merge_files` and the owner sent. It holds the four cells; the four separate cell files were not sent. It records the sha256 of the module that merged it, `7427a686d151c4294809aebaa9664d0b5fa420854817eb79e910372c3f57b6bc`, which is the module in the repository, those of `provider_side.py` and `provider_side_constrained.py` (both pinned), and the second run\'s file it was compared with (`axi_provider_constrained.json`, `7f54c3dd37fd4dcf…`). The GPU seconds of the four cells add up to 527.9, which is 8.8 minutes, 217.4 of them in `S` at K = 64. Every figure below is computed from the step records and the round records of the two files by `tests/test_provider_side_bounded_result.py`.*',
    '**What the page expected, and what came out.** The page named the shape "Lines 2 and 3 yes, lines 9 to 11 yes, lines 12 and 13 NO (the shape I expect)" as "the clean run for the claim", and put exactly that pattern (lines 12 and 13 NO, every other line yes) at about 30 %. **That is what came out: 11 of the 13 lines read yes, lines 12 and 13 read NO, and no line is "not run".** The two NO are the two I put at 4 and 3 %; the cap could not move them, since it does not make job 19 right. The headline (lines 9 and 11) reads yes on both, and so does line 10 between them. Lines 2 and 3 both read yes (the page: about 80 %), and so do the three lines I trusted least among the yes (line 7 at 72 %, line 8 at 75 %, line 6 at 82 %).',
    '**Three things about the run that the page did not foresee or guessed differently.**',
    '* **The new session was slower than the second run\'s in every cell.** Seconds per step were +7.7 %, +4.0 %, +5.0 % and +6.4 % against the second run\'s in `S` at 8 jobs, `KC` at 8, `S` at 32 and `S` at 64, and the round took longer than the same round of the second run in 30, 30, 29 and 29 of the 30 rounds. The page said a session would move things and put the line at 30 %; this one moved them by 4 to 8 %, in all four cells in the same direction. That is why the exact steps per GPU hour here are 7.2 %, 6.9 % and 7.2 % below the page\'s estimate for "the same speed" (20,878, 29,407, 33,318).',
    "* **The wall seconds of the 32- and 64-job cells are not below the second run's (side prediction (c), 45 %); they are above, by 5.6 and 13.1 seconds.** The round that held job 19's step 12 took 4.56 and 8.11 seconds, 1.56 and 1.01 less than in the second run (6.13 and 9.13), while the other 29 rounds took 6.7 and 7.2 % more: the cap shortened that round, and the session's speed hid it in the total. How much of the 1.56 and 1.01 seconds is the cap and how much is noise in one round this run cannot separate.",
    "* **One reply differs from the second run's in a field the page did not compare, and not in its text:** `S` at 64 jobs, job 3, step 16, `ADD[(4,4)->(4,4):value#6]` is the same text with 16 new tokens where the second run's cell at 64 jobs recorded 17 (its cells at 8 and 32 jobs recorded 16 and 16; now all three record 16). Nothing else of that step differs (prompt, cached count, parsed, admitted, exact, distance), no reading line reads that field, and I do not know why the count was 17 once.",
    '## The 13 lines',
    '| # | my probability | what was measured | read |',
    '|---|---|---|---|',
    '**11 of 13 lines read yes, lines 12 and 13 read NO, and no line is "not run".** The pattern "lines 12 and 13 NO, every other line yes" is the one the page called the most likely (about 30 %), and it is the one that came out. Lines 12 and 13 are the two NO that the page put at 4 and 3 %, so they are not surprises: `S` holds 88.3 %, 94.6 % and 97.2 % of the right state at 8, 32 and 64 jobs, the kept chat 100.0 %, and the cap never touched that.',
    "## What the page's rule lets this Result claim",
    "The page's rule, applied line by line to what was read:",
    '* **Lines 2 and 3 read yes**, so the exact-steps-per-hour lines (9 to 11) are claimed as read, with the exactness of both arms written next to every figure.',
    "* **Lines 7 and 8 read yes**, so the point ratios of lines 10 and 11 may be quoted as numbers: 37.7 at 32 jobs and 46.3 at 64 jobs, each with the two sessions named (the exact-state cells are from the session of 19 Oct; the kept chat's cells at 32 and 64 jobs are the second run's, from another Colab session) and the exactness of both arms beside it. At 8 jobs both arms were measured in this session: 3.40.",
    '* **Line 12 reads NO**, so the plain time ratios (seconds per step of the kept chat over seconds per step of exact-state) are **not** claimed as ratios at equal exactness. They are in the table below, with the exactness of both arms beside them, because nothing is to be hidden.',
    '* **Never**, as the page says: a multiple of capacity, a count of GPUs saved, dollars or a percentage of cost, "proven", a statement about any provider, or a multiple of speed without the exactness beside it.',
    "**The sentence the page allows, with this run's numbers.** On one Tesla T4 (Colab), with vLLM 0.31.0, `Qwen/Qwen2.5-7B-Instruct-AWQ`, a cache of 80,000 tokens, 30-step jobs and every reply constrained by one regular expression, exact-state did 19,379, 27,390 and 30,914 exact steps per GPU hour at 8, 32 and 64 jobs, holding the right state after 88.3 %, 94.6 % and 97.2 % of its steps, against 5,694 (8 jobs, the same session), 727 and 667 (32 and 64 jobs, measured in the second run, another Colab session) for a kept chat that held it after 100.0 % of its steps. The ratios of those figures are 3.40, 37.7 and 46.3.",
    "**On the number 49.3.** In the Result of the second run, 49.3 was the ratio at 64 jobs, and that Result withheld it because line 2 read NO. The same ratio under the capped expression, in a new session, is **46.3** (30,914 against 667); it is 6.0 % lower than 49.3 because the new session was slower for exact-state and the kept chat's figure is the second run's. It is a ratio of exact steps per GPU hour, with exact-state exact on 97.2 % of its steps and the kept chat on 100.0 %. **It is not a capacity.** The test sends K jobs in lockstep and counts exact steps per hour; it never looks for the number of jobs a GPU can hold, it measured no GPU other than one free T4, and it says nothing about how many GPUs anything would take. The same goes for 37.7 at 32 jobs.",
    '## Where my expectation was off, and where it was right',
    '* **Line 2 (90 %): the cap held.** Every one of the 3,360 replies was accepted by the strict parser, none used the 96 new tokens, and the longest has 31 tokens (in `S` at 32 and 64 jobs; the longest in `S` at 8 jobs and in `KC` is 16). The one reply that used all 96 in the second run is now 21 tokens.',
    '* **Line 3 (85 %): the replies are the second run\'s.** Of the 3,120 replies of the three `S` cells, 3,118 are the same text as the second run\'s at the same job and step, and the 2 that are not are job 19, step 12, in the 32- and the 64-job cell, the one step the page named. Apart from that step, nothing in these cells differs in text, prompt size, cached count, whether the reply parsed or was admitted, or whether it left the held state exact; at that step the text, the token count and "parsed" differ (admitted, exact and distance do not), and the one other difference is the token count of one reply (first section). **The engine\'s cache counts were identical in a second session: 0 of the 3,360 steps have another cached-token count than the second run\'s.**',
    "* **Line 4 (85 %) and the argument for not running the kept chat at 32 and 64 jobs again.** `KC` at 8 jobs gave the same 240 replies as the second run's and was 100.0 % exact again. The page's argument was that a reply the narrower grammar still allows is not changed by it. In this run it held on every step where it applies: of the 3,360 replies the second run wrote at these jobs and steps, 3,358 are inside the capped language, and all 3,358 were written again as the same text, in a new session. That is one more case in which the argument held; it does not show what the kept chat would do at 32 and 64 jobs, which was not run.",
    '* **Line 5 (88 %) and side prediction (a) (88 %): the capped reply is as the page said.** Job 19, step 12 is now `ADD[(3,0)->(3,0):4tworowsdown4col#4]`: the first 16 characters of the looping name, `4tworowsdown4col`, and then `#4`, 21 tokens (the page: fewer than 40). It begins `ADD[(3,0)->(3,0):` and ends with `]`. The oracle writes `ADD[3,0:4]` there. The gate refuses it (replayed here against the state the job held: "well-formedness: edge ((3, 0), (3, 0), \'4tworowsdown4col\') endpoint missing"), so the state does not move. **Job 19 stayed where it was, as the page said it would:** it holds the right state after 11 of its 30 steps (steps 1 to 11) and after none of the others, the exact flags of every step are the second run\'s, the replies of steps 13 to 30 are the second run\'s, and so are their prompts.',
    "* **Line 6 (82 %): the exact steps are the second run's, to the digit** (212, 908 and 1,867), and so are the six jobs that missed and the steps at which they first missed (table below). The cap cleared line 2 and nothing else, as the page said.",
    "* **Lines 7 and 8 (72 and 75 %) read yes, by a wide margin:** +7.7 %, +5.0 %, +6.4 % and +4.0 % against a limit of 30. What I did not know before the run, the spread of this Colab's T4 from one session to the next, is one observation now: 4.0 to 7.7 %, every cell slower.",
    '* **Lines 9 to 11 (92, 95, 95 %) read yes.** The ratios came out 3.40, 37.7 and 46.3 where the page\'s estimate for the second run\'s speed was 3.5, 40.4 and 49.9: -2.8 %, -6.8 % and -7.2 %. The page said it was "not a prediction of the figures" and that the ratios I expect are "the second run\'s, plus or minus the session"; they are.',
    '* **Lines 12 and 13 (4 and 3 %) read NO, as expected:** the gap to the kept chat is 11.7, 5.4 and 2.8 points at 8, 32 and 64 jobs (limit 5, met at 64 jobs only), and none of the three `S` cells reaches 98 %.',
    "* **The side predictions, read afterwards as descriptions.** (a) true, above (88 %). (b) The only replies that differ from the second run's in text, among the 3,360 steps of the four cells, are the two copies of job 19, step 12 (78 %): true; by token count there is one more, the 64-job cell's job 3, step 16 (first section). (c) The wall seconds of the 32- and 64-job cells below the second run's (45 %): **NO**, above by 5.6 and 13.1 seconds, because the session was slower; the round that held the loop was shorter by 1.56 and 1.01 seconds. (d) The whole run takes 25 to 30 minutes, 10 of them GPU time in the cells (70 %): the cells took 8.8 minutes of engine time; the length of the whole run (installs, the model load, the safety cell) is not in the files and was not timed, so the 25 to 30 cannot be read.",
    '## The tables',
    '**Every cell** (the seconds are the wall seconds around the engine\'s call, summed over the rounds of the cell; "computed" is the prompt tokens the engine did not take from its cache; "cached, 2nd half" is the cached share over the steps from the 16th on; the two prompt columns are the mean prompt of the first 7 and of the last 7 steps of the jobs; "longest reply" is the most new tokens any reply used; "exact" is the share of steps after which the held state equalled the oracle\'s; engine vLLM 0.31.0 on a Tesla T4, no cell resumed):',
    '| arm | K | steps | seconds | s / step | prompt tokens | computed | cached | cached, 2nd half | prompt, first 7 | prompt, last 7 | longest prompt | longest reply | exact |',
    '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|',
    "**Exact steps per GPU hour** (3,600 times the exact steps over the wall seconds of the cell; the last two columns are the second run's figure for the same cell and the change):",
    '| arm | K | exact steps | seconds | exact steps per hour | exact | second run: per hour | change |',
    '|---|---|---|---|---|---|---|---|',
    "The kept chat at 32 and 64 jobs is not in this run's table because it was not run: its cells are the second run's (727 and 667 exact steps per hour, 100.0 % exact, 4,730.9 and 10,330.5 seconds; the 64-job cell was carried over three Colab sessions).",
    '**The ratios, with the exactness of both arms beside them** (exact steps per hour of `S` over those of `KC`; "page\'s estimate" is the estimate for the second run\'s speed that the page gave; the last column is the time per step of `KC` over that of `S`, **not** an equal-exactness ratio, because line 12 reads NO):',
    "| K | exact steps per hour, `S` over `KC` | page's estimate | exact, `S` | exact, `KC` | where the `KC` figure is from | time per step, `KC` over `S` (not an equal-exactness ratio) |",
    '|---|---|---|---|---|---|---|',
    '**This run against the second run, step by step** ("same text" counts the replies that are the same text at the same job and step; the cached counts are the sum over the cell and whether every step has the same count as the second run\'s; "rounds slower" is the number of the 30 rounds that took longer than the same round of the second run):',
    '| arm | K | steps compared | same text | different | s / step | second run | change | rounds slower | seconds | second run | prompt and cached tokens | against the second run |',
    '|---|---|---|---|---|---|---|---|---|---|---|---|---|',
    '**The round that held job 19\'s step 12** (round 12; seconds of rounds 11 / 12 / 13 in this run and in the second run; "other rounds" is the change in the seconds of the other 29 rounds):',
    '| K | this run | second run | round 12, change in seconds | other rounds | new tokens in round 12 |',
    '|---|---|---|---|---|---|',
    '**Where the exactness went** (counted from the step records; "jobs exact to the end" are jobs whose held state equalled the oracle\'s after every step; "refused" are replies the strict parser refused; "96 tokens" are replies that used the whole limit; "came back" are jobs whose state differed at some step and equalled the oracle\'s again later; "corrupted" counts states that were not well formed. Job numbers start at 0):',
    '| arm | K | jobs exact to the end | refused | 96 tokens | jobs that missed | jobs that came back | corrupted |',
    '|---|---|---|---|---|---|---|---|',
    "**The six jobs of `S` that missed, and what they answered at their first miss** (steps count from 1; the oracle's reply is what `oracle_script` writes at that step for a job that is exact up to it; the replies are the same at every K in which the job is run):",
    '| job | step | kind of step | the engine answered | the oracle wrote | against the second run |',
    '|---|---|---|---|---|---|',
    '## What I read from it (derived afterwards; none of this was pre-registered)',
    "1. **A second session moved the speed of this setup by 4 to 8 percent, in the same direction in every cell, and the figures follow it.** Every cell was slower (`S` at 8 jobs +7.7 %, `KC` +4.0 %), and nearly every round of every cell: 118 of the 120 rounds of the four cells took longer than the same round of the second run. The exact steps and the cached counts are the same as the second run's, so the whole difference between its exact steps per hour and these (-7.2 %, -4.7 %, -6.0 % for `S`, -3.9 % for `KC` at 8 jobs) is in the seconds.",
    "2. **The ratio at 8 jobs, with both arms in one session, is 3.40; in the second run it was 3.52** (-3.4 %). That is the one place where this run can say how much the ratio moves between two sessions: by a few percent, not by a factor. At 32 and 64 jobs the ratio mixes two sessions (exact-state from this one, the kept chat from the second run's). **An inference, not tested:** the new session was slower for both arms at 8 jobs, so a kept chat run in this session at 32 or 64 jobs would probably have been somewhat slower than the second run's, which would make these two ratios a little higher, not lower; this run does not show it.",
    '3. **Exactness is where it was.** The capped expression changed the format of one reply and nothing of what the model got wrong: the same six jobs missed at the same steps, with the same replies (table above), and the exact shares are 88.3 %, 94.6 % and 97.2 %. Refused replies of `S` (refused by the strict parser) are 0, 0 and 0 at 8, 32 and 64 jobs (the second run: 0, 1 and 1); the reply that the parser refused is now a reply the parser reads and the gate refuses. The count stays conservative against `S`: a job that is wrong from step 12 on counts as nothing for the rest of its 30 steps.',
    "4. **What the cap saved in time is between about 1 and 2 seconds in a cell, and no closer than that can be read.** The round of the loop took 4.56 and 8.11 seconds in this run, 1.56 and 1.01 less than in the second run. The other 29 rounds took 6.7 and 7.2 % more in this session; if the second run's loop round is slowed the same way it would have taken 6.54 and 9.79 seconds, and the cap saved 1.97 and 1.67 (the page expected about 2.5, from the second run alone). That scaling assumes the loop round was slowed by the session like the others, which this run cannot check, and the noise of one round's seconds was not measured.",
    '## What this does and does not show',
    "* **Shows, on one Tesla T4, one 7B model with 4-bit AWQ weights, vLLM 0.31.0, a cache of 80,000 tokens, 30-step jobs and every reply constrained by one regular expression with a name of at most 16 characters:** line 2 now reads yes, so the exact-steps-per-hour figures of this run are claimed as read; exact-state did 19,379, 27,390 and 30,914 exact steps per GPU hour at 8, 32 and 64 jobs (exact on 88.3 %, 94.6 % and 97.2 % of its steps) against 5,694, 727 and 667 for the kept chat (exact on 100.0 %); the kept chat's figures at 32 and 64 jobs are the second run's.",
    "* **Does not show that exact-state is as exact as the kept chat** (lines 12 and 13 NO), and so no ratio of times is claimed as a ratio at equal exactness. It does not show the kept chat at 32 and 64 jobs under the capped expression (not run), a capacity in jobs per GPU, another GPU (the seconds are a free T4's, and a data-centre GPU is much faster; the ratios, not the seconds, are what might carry over, which this test does not show), another model, engine or cache size, dollars, **energy (never measured; only time)**, retries (this test sends one request a step and never asks again; a refused reply costs a retry in a real system, for either arm), traffic that is not a round of K jobs, a cache that moves rows to CPU memory or disk, jobs longer than 30 steps, or any statement about a provider. It says nothing about the tensor-level roadmap.",
    '* **What may be reported now.** The sentence above, with its numbers, its setup and the exactness of both arms in it; the ratios as ratios of exact steps per GPU hour, naming the two sessions where the kept chat is the second run\'s; lines 12 and 13 as NO, in the same breath, as findings about this setup. **What stays out:** a multiple of capacity, a number of GPUs, dollars, "proven", and a ratio of times as if both arms were equally exact.',
    "* **Not done and worth doing, in this order:** (1) the kept chat at 32 and 64 jobs run again in one session with the exact-state cells, so that every ratio compares one session (1 h 19 min and about 3 hours of GPU; it would also test the argument that the cap does not change the kept chat); (2) retries counted, so that a refused reply costs what it costs; (3) the same cells on another GPU; (4) the GPU's power read next to the seconds.",
    '## Reproduce',
    "In Colab, `python/colab/axi_provider_side_bounded.ipynb` with a T4; about 9 minutes of engine time for the four cells, plus the installs and the model load. Nothing needs the GPU after that: `provider_side_bounded.merge_files` puts cell files together on any machine. `tests/test_provider_side_bounded_result.py` recomputes every figure of this Result from the saved file and the second run's, replays every job from the text of its replies, checks that the merged file is what `merge_cells` makes of its cells, and that the page is the locked text followed by this Result.",
]


def test_the_prose_of_the_result_is_on_the_page_word_for_word_and_every_other_line_is_a_row_of_a_table_recomputed_from_the_files():
    res = run(); R = _result(); data = set(table_lines(res)) | set(table_cells(res)) | set(table_hour(res)) | set(table_ratio(res)) | set(table_second(res)) | set(table_round12(res)) | set(table_anatomy(res)) | set(table_misses(res))
    assert [ln for ln in R.split("\n") if ln.strip() and ln not in data] == GOLDEN


# ---------------------------------------------------------------- 6 what the Result claims
def test_the_result_says_what_the_page_said_it_would_say_and_claims_nothing_more():
    R = _result(); sentences = re.split(r"(?<=[.!?])\s+|\n", R)
    forbidden = [r"times the capacity", r"times as many users", r"GPUs? (?:saved|for one|with 1)", r"\bproven\b", r"dollars?", r"\$\d", r"percentage of cost", r"multiple of capacity", r"\bcapacity\b"]
    safe = re.compile(r"\b(?:not|never|Never|no|without|nothing|neither)\b|\bnone\b|\bis not\b|cannot|stays out", re.I)
    hits = [(p, s_) for s_ in sentences for p in forbidden if re.search(p, s_)]
    assert hits, "the Result names the forbidden claims in order to refuse them"
    for p, s_ in hits:
        assert safe.search(s_) or "capacity source" in s_ or "capacity_tokens" in s_, (p, s_[:200])                                              # a forbidden phrase appears only inside a sentence that refuses it (or as the engine's own setting)
    assert "**It is not a capacity.**" in R and "never looks for the number of jobs a GPU can hold" in R and "says nothing about how many GPUs anything would take" in R
    assert "**The sentence the page allows, with this run's numbers.**" in R and "exact steps per GPU hour at 8, 32 and 64 jobs, holding the right state after 88.3 %, 94.6 % and 97.2 % of its steps" in R and "measured in the second run, another Colab session" in R
    assert "**What stays out:** a multiple of capacity, a number of GPUs, dollars, \"proven\", and a ratio of times as if both arms were equally exact." in R and "**energy (never measured; only time)**" in R
    assert "not claimed as ratios at equal exactness" in R.replace("**not** claimed", "not claimed") or "are **not** claimed as ratios at equal exactness" in R
    # the plain time ratios appear only in the table that says they are not equal-exactness ratios
    assert "(not an equal-exactness ratio)" in R and R.count("time per step, `KC` over `S`") == 1
    # no sentence of the Result turns the kept chat's cells of the second run into this run's
    assert "the kept chat's figures at 32 and 64 jobs are the second run's" in R and "(32 and 64 jobs, measured in the second run, another Colab session)" in R
    # honest outcomes are written: the NO lines, the NO side prediction, the unexpected field
    assert "lines 12 and 13 read NO" in R and "(c) The wall seconds of the 32- and 64-job cells below the second run's (45 %): **NO**" in R and "One reply differs from the second run's in a field the page did not compare" in R
    assert [ln for ln in R.split("\n") if ln.startswith("## ")] == ["## The 13 lines", "## What the page's rule lets this Result claim", "## Where my expectation was off, and where it was right", "## The tables",
                                                                    "## What I read from it (derived afterwards; none of this was pre-registered)", "## What this does and does not show", "## Reproduce"]


# ---------------------------------------------------------------- 7 the front pages
FRONT_ITEM = [   # item 108 of the concept map: what came out, and the test that checks it
    '**What came out (the Result is on the page; the owner\'s Colab run of 19 Oct 2026, one session, Tesla T4, vLLM 0.31.0, all four cells, none resumed):** 11 of the 13 lines read yes, lines 12 and 13 read NO (4 and 3 %, as expected) and none is "not run"; the pattern the page called most likely (about 30 %) is the one that came out. **Line 2 now reads yes:** 3,360 of 3,360 replies were accepted by the strict parser, none used 96 tokens (the longest has 31), and the reply that looped (job 19, step 12) is now `ADD[(3,0)->(3,0):4tworowsdown4col#4]`, 21 tokens, read by the parser and refused by the gate, so job 19 stayed where it was, as the page said. Line 3 reads yes: of the 3,120 replies of the three `S` cells, 3,118 are the same text as in the second run and the 2 that differ are that one step; the engine\'s cached-token counts were identical in a second session (0 of 3,360 steps differ). The new session was 4.0 to 7.7 % slower than the second run\'s in every cell (lines 7 and 8 yes, limit 30 %), so the figures are about 7 % below the page\'s estimate: exact-state did 19,379, 27,390 and 30,914 exact steps per GPU hour at 8, 32 and 64 jobs (88.3, 94.6 and 97.2 % of its steps exact) against 5,694 (the same session), 727 and 667 (the second run\'s cells, another Colab session) for the kept chat (100.0 % exact), which is 3.40, 37.7 and 46.3 times. By the page\'s rule these ratios may be quoted as numbers, naming the two sessions (lines 2, 3, 7 and 8 yes), and not as ratios of time at equal exactness (line 12 NO). **49.3 is not the number: it was the second run\'s, withheld there because line 2 read NO; the figure this run supports is 46.3, a ratio of exact steps per GPU hour with exact-state at 97.2 % exact and the kept chat at 100.0 %, and it is not a capacity.** Two things the page did not foresee: the wall seconds of the 32- and 64-job cells are above the second run\'s, not below (side prediction (c), NO), and one reply has the same text with 16 new tokens where the second run\'s 64-job cell recorded 17. **Not shown:** that exact-state is as exact as the kept chat (it is not: lines 12 and 13 NO), the kept chat at 32 and 64 jobs under the capped expression (not run; its cells are the second run\'s), a capacity in jobs per GPU, any number of GPUs, dollars, energy, other GPUs, models or engines. **Next, in this order, if wanted:** the kept chat at 32 and 64 jobs run again in one session with the exact-state cells (about 4 hours of GPU), retries counted, the same cells on another GPU, the GPU\'s power read.',
    "The Result is checked by `tests/test_provider_side_bounded_result.py` = **test 127** (10 tests: the files and their sha256, the merge, every one of the 3,360 step records replayed from the text of its reply, the 13 lines by the code and by hand, the page as the locked text followed by the rendered Result, every figure and table recomputed from the step and round records of `docs/results/axi_provider_bounded.json`, sha256 1f12a846b1325f9fed4e5df2cc755b3a7fd996266cb590311230f7cf418f2733, against the second run's file, the replies and cached counts and speeds against the second run step by step, job 19 replayed through the gate, that the Result claims nothing the page forbids, every line of its prose word for word, and the sentences of the two front pages that say what the run found). Mutation check of this test, run on its first nine tests (the tenth was added after it and pins the front pages): 1,132 deliberate changes (300 digits of the Result in the page, 60 digits of the locked text, 260 numbers of the result file changed with everything derived made consistent, and 260 more with the word-for-word prose test switched off, 40 numbers of the second run's file, 76 claim sentences and 127 pieces of prose in the Result's generator, 9 settings of the test): all caught. A first version of the test let 49 of those 127 pieces of prose change unnoticed, which is why the prose is pinned line by line.",
]
FRONT_ROW = [   # the row of VERIFIED.md for the third run
    "**Run (19 Oct 2026, Colab, Tesla T4, vLLM 0.31.0, one session, all four cells).** The page and its 13 reading lines were locked before the run (sha256 888358daf88abd0e… of the locked text); the Result is on the page and is checked by test 127. **11 of 13 lines yes, lines 12 and 13 NO as expected (4 and 3 %): line 2 now reads yes** (3,360 of 3,360 replies accepted by the strict parser, none at 96 tokens; the looping reply is now a 21-token reply the gate refuses, and job 19 stayed where it was), and line 3 reads yes (3,118 of 3,120 replies of the `S` cells are the second run's text; the 2 that differ are that one step; cached counts identical). The new session was 4.0 to 7.7 % slower than the second run's in every cell (lines 7 and 8 yes). Exact-state did 19,379, 27,390 and 30,914 exact steps per GPU hour at 8, 32 and 64 jobs (88.3, 94.6, 97.2 % exact) against 5,694 (same session), 727 and 667 (the second run's cells, another session) for the kept chat (100.0 %): 3.40, 37.7, 46.3 times, ratios of exact steps per GPU hour, not capacities; **the second run's 49.3 is replaced by 46.3.** Side prediction (c) read NO (the wall seconds are above the second run's).",
    "The Result: every figure recomputed from the saved file and the second run's, every one of the 3,360 step records replayed from its reply text, the 13 lines by the code and by hand, job 19 replayed through the gate, the prose pinned word for word, these two front pages pinned (test 127, 10 tests); mutation check of test 127, run on its first nine tests: 1,132 deliberate changes, all caught.",
    "Not shown: that the cap makes exact-state as exact as the kept chat (lines 12 and 13 NO), the kept chat at 32 and 64 jobs under the capped expression (not run; the second run's cells), a capacity in jobs per GPU, any number of GPUs, dollars, energy, other GPUs, models or engines.",
]
FRONT_NEXT = 'Next free: row 191, item 144, test 169, question 122'


def test_the_front_pages_say_what_the_run_found_and_point_to_this_test():
    cm = open(os.path.join(ROOT, "docs", "RAJNISH_CONCEPT_MAP.md")).read(); ve = open(os.path.join(ROOT, "docs", "VERIFIED.md")).read()
    item = cm[cm.index("108. **"): cm.index("109. **")]; k = ve.index("Provider-side serving test, third run"); row = ve[k: ve.index("\n", k)]
    for s in FRONT_ITEM: assert s in item, s[:100]
    for s in FRONT_ROW: assert s in row, s[:100]
    assert FRONT_NEXT in cm and "test 126 checks the capped re-run of the exact-state cells and test 127 its Result, item 108; test 128 checks the retry test and test 129 its Result, item 109; test 130" in cm
    R = _result(); assert "11 of 13 lines read yes" in R and "46.3" in item and "46.3" in row and "49.3 is not the number" in item and "the second run's 49.3 is replaced by 46.3" in row
    for text in (item, row): assert "@@" not in text and "not run yet" not in text.lower() and "capacity" in text


if __name__ == "__main__":
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]; bad = 0
    for n, f in fns:
        try: f(); print("ok  ", n)
        except Exception: bad += 1; print("FAIL", n); traceback.print_exc()
    print(f"{len(fns) - bad}/{len(fns)} passed"); sys.exit(1 if bad else 0)
