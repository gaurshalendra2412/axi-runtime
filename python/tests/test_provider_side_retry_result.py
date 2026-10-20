"""Implements concept-map item 109 (Result): the fourth provider-side serving run, the exact-state cells with ONE retry when the gate refuses a reply that the repair cannot fix, made on Colab
(vLLM 0.31.0, Tesla T4, Qwen2.5-7B-Instruct-AWQ, a cache of 80,000 tokens, every reply constrained by the capped expression). Page: docs/PROVIDER_SIDE_RETRY_TEST.md (the question, the design, the 12 reading
lines and the predictions were written before the run; the Result is appended below them).
The file: docs/results/axi_provider_retry.json (the merged file the notebook's last cell wrote and the owner sent: `R8`, `R32`, `R64` with the retry, and the pair `H0`/`H1` on jobs 64 to 127 without and with it),
read against docs/results/axi_provider_bounded.json (the third run: its `S` cells are the baseline of the three small cells).
This file recomputes from the saved files, with its own arithmetic and not with the code that printed the figures:
  1. the files: their sha256, the settings, the seeds, the rounds adding up to the cells, and every job replayed from the texts of its replies through the framework's own gate, repair and diagnosis (parsed,
     admitted, corrupted, exact, distance of every step; the rule that fires the retry and the feedback text of every second request);
  2. the 12 reading lines (8 yes; lines 6, 8, 9 and 11 NO; none not run) by the code and by hand from the raw numbers, the headline, and what the page's rule lets the Result claim;
  3. the 29 second requests: what each was asked, what came back, which were asked after the job had slipped, which first replies were the oracle's own, what the rule could not know;
  4. the eight first slips (the request, the oracle's reply, the model's reply, the kind, what the retry did), the lost steps per job against the third run, the cost of the second rounds, the side predictions;
  5. the whole Result is `render(result)` and the page must equal the locked head followed by it; the figures quoted in it are recounted; what the Result says about a text pasted after the run is recomputed from the
     first, second and third runs' files;
  6. that the Result claims what the page's rule allows and nothing more (no multiple of capacity, no GPUs saved, no dollars, no 'proven', nothing about more than one retry).
What it does NOT show: anything about another feedback wording, more than one retry, another model, GPU, engine or cache size, dollars, energy, or a real server's mixed traffic."""
import hashlib, json, os, re, sys

sys.path.insert(0, os.path.dirname(__file__))
import axi.experiments.provider_side as P
import axi.experiments.provider_side_bounded as B
import axi.experiments.provider_side_retry as R
from axi.experiments.agent_loop import feedback_text, strict_parse
from axi.engine.diagnostics import diagnose
from axi.engine.gate import check
from axi.engine.scope import contain_scope, named_services

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..", "..")
PAGE = os.path.join(ROOT, "docs", "PROVIDER_SIDE_RETRY_TEST.md")
RESD = os.path.join(ROOT, "docs", "results")
RES = os.path.join(RESD, "axi_provider_retry.json")
THIRD = os.path.join(RESD, "axi_provider_bounded.json")
SECOND = os.path.join(RESD, "axi_provider_constrained.json")
FIRST = os.path.join(RESD, "axi_provider_side.json")
MODULE = os.path.join(HERE, "..", "axi", "experiments", "provider_side_retry.py")
RESULT_MARK = "\n\n# Result"
PAGE_SHA256 = "74243e1ab623ec5c20168de0978bc33224b06875630fe83e56987806a529ce33"
FILE_SHA256 = "390f7464a54352764e21203f9546bd17fc4714b09cbb0aec9390963e5d94b116"
THIRD_SHA256 = "1f12a846b1325f9fed4e5df2cc755b3a7fd996266cb590311230f7cf418f2733"
SECOND_SHA256 = "7f54c3dd37fd4dcf61e8a7991af285d6ad1a0d691362496216949a26b45b819d"
FIRST_SHA256 = "19ce96633931980292f2ab6b25c42799b50696db2ea25225f194e02c8ca93d48"
MODULE_SHA256 = "9cd1398b912c187ddd8e62819e7396057477596a463de850d7ed612be1a8be1f"
PINS = dict(provider_side="bfb4734e171d9244cd47d10402be46528a557095a8275d02b38481a0e2b0b1d3", constrained="03452148e28c9f31372f12cc11ef1808ac67308adff972fa19d4aeeef7c96f7d",
            bounded="7427a686d151c4294809aebaa9664d0b5fa420854817eb79e910372c3f57b6bc", gate="b6a50408788791f76bc4d4129dad84debb6e09c9c5e804673cdf04abec224ce7",
            diagnostics="255f3c11db989b645077d10706a187e2c913576538a826e6dbdb8892b014b4f2", agent_loop="d658afaa29e3bce7e1fb7f43d88281fc40ee68ca59864421a65284e9973516b6",
            cascade="b0533316bc094e80a1d68eddb7e137c7c176af9a342341288360c18d61f74fe7")
REGEX_SHA256 = "6f4e5a993f26ccd471bff13422fa622b7820858a91e05d43c2bccfbe6b864245"
SEED, STEPS = 20261017, 30
ORDER = ["R8", "R32", "R64", "H0", "H1"]
WITH_RETRY = ["R8", "R32", "R64", "H1"]
JOBS = {"R8": (0, 8), "R32": (0, 32), "R64": (0, 64), "H0": (64, 64), "H1": (64, 64)}              # (first job, number of jobs)
PROBS = [98, 92, 85, 93, 90, 85, 75, 35, 55, 88, 40, 70]
READ = [True, True, True, True, True, False, True, False, False, True, False, True]
KINDS = {3: "shape", 19: "shape", 101: "shape", 4: "number", 20: "number", 122: "number", 17: "extra delete", 62: "extra delete"}
_cache = {}


def run():
    if "run" not in _cache: _cache["run"] = json.load(open(RES))
    return _cache["run"]


def third():
    if "third" not in _cache: _cache["third"] = json.load(open(THIRD))
    return _cache["third"]


def _load(key, path):
    if key not in _cache: _cache[key] = json.load(open(path))
    return _cache[key]


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
def first_text(x): return x["first_text"] if x.get("retried") else x["text"]


def oracle(job):
    """what the oracle writes at every step of the job: the exact path (the first slip of a job is read on this path, which the job is on until it slips)"""
    if ("oracle", job) not in _cache:
        s = P.Session("S", SEED * 1000 + job, STEPS); out = []
        while s.t < len(s.script):
            rq = s.request(); out.append(rq.stand_in_reply); s.accept(P.Reply(rq.stand_in_reply, 0, 0, 0))
        _cache[("oracle", job)] = out
    return _cache[("oracle", job)]


def request_sentence(job, step):
    """the request of the job at `step` (from 1), on the exact path"""
    s = P.Session("S", SEED * 1000 + job, STEPS)
    while s.t < step - 1: rq = s.request(); s.accept(P.Reply(rq.stand_in_reply, 0, 0, 0))
    last = [m for m in s.request().messages if m["role"] == "user"][-1]["content"]
    return re.search(r"Request: (.*)$", last, re.S).group(1).strip()


def st(c):
    L = logs(c); n = len(L); wall = secs(c); ex = sum(1 for x in L if x["exact"]); asked = [x for x in L if x.get("retried")]
    rr = [r for r in c["rounds"] if r.get("kind") == "retry"]; fr = [r for r in c["rounds"] if r.get("kind", "first") == "first"]
    allrep = [(x["parsed"], x["n_new"]) for x in L] + [(x["first_parsed"], x["first_n_new"]) for x in asked]
    return dict(n=n, wall=wall, sps=wall / n, ex=ex, share=ex / n, eph=3600 * ex / wall, asked=len(asked), retry_rounds=len(rr), retry_s=sum(r["seconds"] for r in rr), first_rounds=len(fr),
                per_round=(sum(r["seconds"] for r in rr) / len(rr)) if rr else None, retry_share=(sum(r["seconds"] for r in rr) / wall), jobs=len(c["sessions"]),
                whole=sum(1 for s in c["sessions"] if all(x["exact"] for x in s["log"])), unparsed=sum(1 for p, _ in allrep if not p), replies=len(allrep), full96=sum(1 for _, k in allrep if k >= 96),
                longest=max(k for _, k in allrep), corrupted=sum(1 for x in L if x["corrupted"]), exact_after=sum(1 for x in asked if x["exact"]))


def lost(c): return {job_of(s): sum(1 for x in s["log"] if not x["exact"]) for s in c["sessions"] if any(not x["exact"] for x in s["log"])}


def third_cell(k): return next(c for c in third()["cells"] if c["arm"] == "S" and c["K"] == k)


def retried(res):
    """every second request: the cell, the job, the step, both replies, the oracle's reply, whether the job was already behind (an earlier step of the job in that cell was not exact)"""
    out = []
    for lab in WITH_RETRY:
        for s in cells(res)[lab]["sessions"]:
            job = job_of(s); behind = False
            for x in s["log"]:
                if x.get("retried"):
                    o = oracle(job)[x["t"] - 1]
                    out.append(dict(cell=lab, job=job, step=x["t"], kind=x["kind"], first=x["first_text"], second=x["text"], reason=x["first_reason"], oracle=o, behind=behind, first_ok=x["first_text"] == o,
                                    second_ok=x["text"] == o, admitted=x["admitted"], exact=x["exact"], feedback=x["feedback"]))
                if not x["exact"]: behind = True
    return out


def outcome(r):
    if r["second"] == r["first"]: return "same"
    if r["second_ok"] and not r["first_ok"] and r["exact"]: return "put right"
    if r["admitted"]: return "admitted, wrong"
    return "refused again"


OUTCOMES = ["same", "refused again", "admitted, wrong", "put right"]


def first_slips(res):
    """the first step of every job at which it is not exact or was asked again, in `R64` (jobs 0 to 63) and `H1` (jobs 64 to 127): (job, cell, step, the first reply, the record)"""
    out = []
    for lab in ("R64", "H1"):
        for s in cells(res)[lab]["sessions"]:
            for x in s["log"]:
                if not x["exact"] or x.get("retried"): out.append((job_of(s), lab, x["t"], first_text(x), x)); break
    return sorted(out)


def slip_kind(first, o):
    if re.fullmatch(r"ADD\[\((\d+),(\d+)\)->\(\1,\2\):[A-Za-z0-9_]{1,16}#-?\d+\]", first) and re.fullmatch(r"ADD\[\d+,\d+:-?\d+\]", o): return "shape"
    if " " in first and first.startswith("DEL") and first.split(" ", 1)[1] == o: return "extra delete"
    if re.fullmatch(r"ADD\[\d+,\d+:-?\d+\]", first) and re.fullmatch(r"ADD\[\d+,\d+:-?\d+\]", o): return "number"
    if re.fullmatch(r"ADD\[\(\d+,\d+\)->\(\d+,\d+\):\w+#-?\d+\]", first) and re.fullmatch(r"ADD\[\(\d+,\d+\)->\(\d+,\d+\):\w+#-?\d+\]", o): return "number"
    return "other"


def short_reason(r): return r.split(":")[0]


# ---------------------------------------------------------------- replay: every job played again from the texts of its replies
def replay_job(c, s):
    """the job played again through the framework's own session: for every step the verdict on the first reply (the rule), the feedback of the second request, and the step record of the reply that decided"""
    rs = R.RetrySession(s["seed"], STEPS); out = []
    for x in s["log"]:
        rq = rs.request(); first = first_text(x); info = rs.read_reply(first); fb = None
        if info["fires"]:
            fb = rq2 = rs.retry_request(rq, first); fb = rq2.messages[-1]["content"]
        rs.accept(P.Reply(x["text"], 0, 0, 0)); y = rs.log[-1]
        out.append((info, fb, (y["parsed"], y["admitted"], y["corrupted"], y["exact"], y["distance"])))
    return out


# ---------------------------------------------------------------- the lines, by hand from the raw numbers
def same_first(new, base):
    """job by job: the first reply of every step up to and including the first step at which the job asked again is the base's reply of that job and step; a job that never asked again is the same throughout"""
    if len(new["sessions"]) != len(base["sessions"]): return False
    for a, b in zip(new["sessions"], base["sessions"]):
        asked = any(x.get("retried") for x in a["log"])
        if not asked and len(a["log"]) != len(b["log"]): return False
        for x, y in zip(a["log"], b["log"]):
            if first_text(x) != y["text"]: return False
            if x.get("retried"): break
    return True


def lines_by_hand(res):
    c = cells(res); S = {k: st(v) for k, v in c.items()}; t3 = {k: st(third_cell(k)) for k in (8, 32, 64)}; K = {"R8": 8, "R32": 32, "R64": 64}
    allreq = [x for lab in ORDER for x in logs(c[lab])]
    l1 = all(x["n_cached"] is not None and (x["first_n_cached"] is not None if x.get("retried") else True) for x in allreq)
    rep = sum(S[k]["replies"] for k in S); unp = sum(S[k]["unparsed"] for k in S)
    l2 = unp <= 0.005 * rep and all(S[k]["full96"] == 0 for k in S)
    l3 = all(same_first(c[lab], third_cell(K[lab])) for lab in ("R8", "R32", "R64")) and same_first(c["H1"], c["H0"])
    asked = [x for lab in ORDER for x in logs(c[lab]) if x.get("retried")]
    l4 = all(x["first_parsed"] and not x["first_admitted"] and not x["first_repairable"] for x in asked) and S["H0"]["asked"] == 0
    l5 = all(S[lab]["asked"] / S[lab]["n"] <= 0.03 for lab in WITH_RETRY)
    l6 = all(S[lab]["share"] > t3[K[lab]]["share"] for lab in K)
    l7 = S["R64"]["share"] >= 0.98
    l8 = S["R64"]["share"] >= 0.99
    l9 = S["H1"]["share"] >= S["H0"]["share"] + 0.01
    l10 = S["H1"]["sps"] <= 1.10 * S["H0"]["sps"]
    l11 = S["H1"]["eph"] > S["H0"]["eph"]
    ex = lambda cc: [sum(1 for x in s["log"] if x["exact"]) for s in cc["sessions"]]
    l12 = all(a >= b for a, b in zip(ex(c["H1"]), ex(c["H0"])))
    return [l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12]


def third_fires(k):
    """where the rule fires on the replies of the third run's `S` cell at k jobs (the page's replay)"""
    out = []
    for j, s in enumerate(third_cell(k)["sessions"]):
        rs_ = R.RetrySession(SEED * 1000 + j, STEPS)
        for x in s["log"]:
            rs_.request(); info = rs_.read_reply(x["text"])
            if info["fires"]: out.append((j, x["t"]))
            rs_.accept(P.Reply(x["text"], 0, 0, 0))
    return out


def fires_h0(res):
    """where the rule WOULD fire on the replies of H0 (no retry was made there)"""
    out = []
    for s in cells(res)["H0"]["sessions"]:
        for (info, fb, rec), x in zip(replay_job(cells(res)["H0"], s), s["log"]):
            if info["fires"]: out.append((job_of(s), x["t"]))
    return out


# ---------------------------------------------------------------- the lines' "what was measured" text
def measured(res, i):
    c = cells(res); S = {k: st(v) for k, v in c.items()}; t3 = {k: st(third_cell(k)) for k in (8, 32, 64)}; K = {"R8": 8, "R32": 32, "R64": 64}
    n_req = sum(S[k]["n"] + S[k]["asked"] for k in S); rep = sum(S[k]["replies"] for k in S); unp = sum(S[k]["unparsed"] for k in S)
    ex = lambda cc: [sum(1 for x in s["log"] if x["exact"]) for s in cc["sessions"]]
    if i == 1: return f"the engine reported the cached tokens of every request: {num(n_req)} requests in the 5 cells, {num(sum(S[k]['asked'] for k in S))} of them second requests"
    if i == 2:
        return (f"replies accepted by the strict parser: {num(rep - unp)} of {num(rep)} (first and second), {pc((rep - unp) / rep)} (limit 99.5 %, met); replies that used all 96 new tokens: none "
                f"(the longest has {max(S[k]['longest'] for k in S)} tokens)")
    if i == 3:
        return (f"first replies against the third run's `S` cell (`R8`, `R32`, `R64`) and `H1` against `H0`, job by job, up to each job's first retried step: all the same text "
                f"({num(sum(S[k]['n'] for k in ('R8', 'R32', 'R64')))} steps in the three cells and {num(S['H1']['n'])} in the pair compared; the jobs that asked again are 3, 4, 19 and 20 in the three cells and 101 and 122 in `H1`)")
    if i == 4: return f"{num(sum(S[k]['asked'] for k in WITH_RETRY))} second requests, every one for a first reply that parsed, was refused by the gate and could not be repaired; `H0` made none"
    if i == 5: return "retried steps over steps: " + ", ".join(f"`{lab}` {S[lab]['asked']} of {num(S[lab]['n'])} ({pc(S[lab]['asked'] / S[lab]['n'])})" for lab in WITH_RETRY) + " (limit 3 %, met)"
    if i == 6:
        return "exact steps with the retry against the third run's `S`: " + ", ".join(f"K = {K[lab]} {num(S[lab]['ex'])} of {num(S[lab]['n'])} ({pc(S[lab]['share'])}) against {num(t3[K[lab]]['ex'])} ({pc(t3[K[lab]]['share'])})" for lab in K) + " (limit: higher in each; equal at K = 8)"
    if i == 7: return f"`R64`: {num(S['R64']['ex'])} of {num(S['R64']['n'])} steps exact, {pc(S['R64']['share'])} (limit 98 %, met)"
    if i == 8: return f"`R64`: {pc(S['R64']['share'])} (limit 99 %, not met; the most it could be was 99.7 %)"
    if i == 9: return f"jobs 64 to 127: `H1` {num(S['H1']['ex'])} of {num(S['H1']['n'])} exact ({pc(S['H1']['share'])}), `H0` {num(S['H0']['ex'])} ({pc(S['H0']['share'])}): {(S['H1']['share'] - S['H0']['share']) * 100:+.1f} points (limit +1.0)"
    if i == 10: return f"`H1` {S['H1']['sps']:.4f} seconds per step, `H0` {S['H0']['sps']:.4f}: {S['H1']['sps'] / S['H0']['sps']:.3f} times (limit 1.10, met)"
    if i == 11: return f"`H1` {num(round(S['H1']['eph']))} exact steps per GPU hour, `H0` {num(round(S['H0']['eph']))}: {sg((S['H1']['eph'] / S['H0']['eph'] - 1) * 100)} (limit: higher)"
    if i == 12:
        a, b = ex(c["H1"]), ex(c["H0"]); same = sum(1 for x, y in zip(a, b) if x == y)
        return f"the 64 jobs of the pair: {same} hold the right state after as many steps with the retry as without, {sum(1 for x, y in zip(a, b) if x > y)} after more, {sum(1 for x, y in zip(a, b) if x < y)} after fewer"


def table_lines(res):
    return [f"| {i} | {PROBS[i - 1]} % | {measured(res, i)} | {'yes' if READ[i - 1] else 'NO'} |" for i in range(1, 13)]


def table_cells(res):
    c = cells(res); S = {k: st(v) for k, v in c.items()}; rows = ["| cell | jobs | retry | steps | seconds | s / step | exact steps | exact | jobs exact to the end | exact steps per hour |", "|---|---|---|---|---|---|---|---|---|---|"]
    for lab in ORDER:
        f, k = JOBS[lab]; s = S[lab]
        rows.append(f"| `{lab}` | {f} to {f + k - 1} | {'on' if lab in WITH_RETRY else 'off'} | {num(s['n'])} | {s['wall']:.1f} | {s['sps']:.3f} | {num(s['ex'])} | {pc(s['share'])} | {s['whole']} of {s['jobs']} | {num(round(s['eph']))} |")
    return rows


def table_third(res):
    c = cells(res); S = {k: st(c[k]) for k in ("R8", "R32", "R64")}; rows = ["| K | exact steps, third run | with the retry | change | jobs that missed, third run | with the retry |", "|---|---|---|---|---|---|"]
    for lab, k in (("R8", 8), ("R32", 32), ("R64", 64)):
        t = st(third_cell(k)); a, b = lost(third_cell(k)), lost(c[lab])
        rows.append(f"| {k} | {t['ex']} ({pc(t['share'])}) | {S[lab]['ex']} ({pc(S[lab]['share'])}) | {S[lab]['ex'] - t['ex']:+d} | " + ", ".join(f"{j} ({n})" for j, n in sorted(a.items())) + " | " + ", ".join(f"{j} ({n})" for j, n in sorted(b.items())) + " |")
    return rows


def outcome_counts(rs):
    d = {o: 0 for o in OUTCOMES}
    for r in rs: d[outcome(r)] += 1
    return d


def table_asked(res):
    c = cells(res); rs = retried(res); rows = ["| cell | steps asked again | second rounds | their seconds | per round | share of the cell's seconds | same reply | other reply, refused again | other reply, admitted and wrong | put right | exact after the step |",
                                                 "|---|---|---|---|---|---|---|---|---|---|---|"]
    for lab in WITH_RETRY:
        s = st(c[lab]); o = outcome_counts([r for r in rs if r["cell"] == lab])
        rows.append(f"| `{lab}` | {s['asked']} | {s['retry_rounds']} | {s['retry_s']:.2f} | {s['per_round']:.2f} | {pc(s['retry_share'])} | {o['same']} | {o['refused again']} | {o['admitted, wrong']} | {o['put right']} | {s['exact_after']} |")
    o = outcome_counts(rs); S = {k: st(c[k]) for k in WITH_RETRY}
    rows.append(f"| all | {sum(S[k]['asked'] for k in S)} | {sum(S[k]['retry_rounds'] for k in S)} | {sum(S[k]['retry_s'] for k in S):.2f} | {sum(S[k]['retry_s'] for k in S) / sum(S[k]['retry_rounds'] for k in S):.2f} | | {o['same']} | {o['refused again']} | {o['admitted, wrong']} | {o['put right']} | {sum(S[k]['exact_after'] for k in S)} |")
    return rows


def table_retried(res):
    rows = ["| cell | job | step | first reply | why the gate refused it | second reply | what came back | exact after |", "|---|---|---|---|---|---|---|---|"]
    for r in retried(res):
        note = outcome(r) + (" (first reply was the oracle's own)" if r["first_ok"] else "")
        rows.append(f"| `{r['cell']}` | {r['job']} | {r['step']} | `{r['first']}` | {short_reason(r['reason'])} | `{r['second']}` | {note} | {'yes' if r['exact'] else 'no'} |")
    return rows


def table_slips(res):
    c = cells(res); rows = ["| job | step | the request | the oracle writes | the model's first reply | kind | the gate | the retry |", "|---|---|---|---|---|---|---|---|"]
    for job, lab, step, first, x in first_slips(res):
        o = oracle(job)[step - 1]; k = slip_kind(first, o)
        if x.get("retried"):
            r = next(r for r in retried(res) if r["cell"] == lab and r["job"] == job and r["step"] == step); gate = short_reason(r["reason"]) + ", refused"; what = {"put right": "put right", "same": "the same reply again", "admitted, wrong": "answered with `" + r["second"] + "`, admitted and wrong", "refused again": "answered with `" + r["second"] + "`, refused again"}[outcome(r)]
        else: gate, what = "admitted", "not asked: the gate admitted the reply"
        rows.append(f"| {job} | {step} | {request_sentence(job, step)} | `{o}` | `{first}` | {k} | {gate} | {what} |")
    return rows


def scope_flags(res):
    """the jobs whose first-slip reply the framework's scope filter changes (drops something), with the request's own coordinates as the names"""
    out = []
    for j, lab, s, f, x in first_slips(res):
        sc, dropped = contain_scope(strict_parse(f), named_services(request_sentence(j, s)))
        if dropped: out.append(j)
    return out


def table_missed(res):
    c = cells(res); rows = ["| cell | job | steps not exact | first step not exact |", "|---|---|---|---|"]
    for lab in ("R64", "H1"):
        for s in c[lab]["sessions"]:
            bad = [x["t"] for x in s["log"] if not x["exact"]]
            if bad: rows.append(f"| `{lab}` | {job_of(s)} | {len(bad)} | {bad[0]} |")
    return rows


def pairs(res):
    """the 64 jobs of the pair: steps exact in H1 and in H0, per job"""
    c = cells(res); return [(job_of(a), sum(1 for x in a["log"] if x["exact"]), sum(1 for x in b["log"] if x["exact"])) for a, b in zip(c["H1"]["sessions"], c["H0"]["sessions"])]


def job3_gain(res):
    """the steps of job 3 that are exact in R64 and are not in the third run's S at 64 jobs"""
    a = next(s for s in cells(res)["R64"]["sessions"] if job_of(s) == 3)["log"]; b = third_cell(64)["sessions"][3]["log"]
    return [x["t"] for x, y in zip(a, b) if x["exact"] and not y["exact"]]


def render(res):
    c = cells(res); S = {k: st(v) for k, v in c.items()}; t3 = {k: st(third_cell(k)) for k in (8, 32, 64)}; e = res["meta"]["engine"]; rs = retried(res); o = outcome_counts(rs); K = {"R8": 8, "R32": 32, "R64": 64}
    wall = sum(S[k]["wall"] for k in S); behind = [r for r in rs if r["behind"]]; ok1 = [r for r in rs if r["first_ok"]]; fresh = [r for r in rs if not r["first_ok"]]
    ok1o = outcome_counts(ok1); frs = outcome_counts(fresh); slips = first_slips(res); kinds = {j: slip_kind(f, oracle(j)[s - 1]) for j, _, s, f, _ in slips}
    gain = {k: S[lab]["ex"] - t3[k]["ex"] for lab, k in K.items()}; g3 = job3_gain(res); pr = pairs(res)
    fb = rs[0]["feedback"]; l19 = next(r for r in rs if r["job"] == 19 and r["cell"] == "R64")
    n_retry_rounds = sum(S[k]["retry_rounds"] for k in S); retry_s = sum(S[k]["retry_s"] for k in S)
    put_four = len({r["job"] for r in rs if outcome(r) == "put right" and r["job"] in (3, 4, 19, 20)}); fires8 = third_fires(8)
    lines = []
    A = lines.append
    A("")
    A("")
    A("# Result")
    A("")
    A(f"*Written after the run, below the locked text; nothing above this line has been changed (its hash is checked by `tests/test_provider_side_retry.py` and again by `tests/test_provider_side_retry_result.py`). The run was made by the owner on Colab with `python/colab/axi_provider_side_retry.ipynb`; the file reached me on 20 Oct 2026 and holds no date of its own. It records: vLLM {e['version']}, GPU {e['gpu']}, Python {e['python']}, `{e['model']}`, the cache fixed at {num(e['capacity_tokens'])} tokens with the engine's block override (capacity source \"{e['capacity_source']}\"), 30-step jobs, seed {SEED}, every reply constrained by the capped expression of the third run (sha256 `{REGEX_SHA256[:16]}…`), the policy `repair_m6`, and one retry with the `structured_imperative` feedback. **All five planned cells ran** (`R8`, `R32`, `R64`, `H0`, `H1`); each holds its 30 first rounds in order. The file is `docs/results/axi_provider_retry.json` (sha256 `{FILE_SHA256}`): the merged file that the notebook's last cell wrote with `provider_side_retry.merge_files`. It records the sha256 of that module (`{MODULE_SHA256[:16]}…`, the module in the repository), those of the seven files the test imports (all as pinned on the page), and the third run's file it was compared with (`axi_provider_bounded.json`, `{THIRD_SHA256[:16]}…`). The seconds around the engine's calls of the five cells add up to {wall:.1f}, which is {wall / 60:.1f} minutes, {S['H0']['wall'] + S['H1']['wall']:.1f} of them in the pair. What the safety cell and the probe printed is not in the saved file, and nothing below rests on it. Every figure below is computed from the saved file by `tests/test_provider_side_retry_result.py`.*")
    A("")
    A(f"**What the page expected, and what came out.** The page put lines 2, 3 and 4 all yes at about 75 %, and they read yes, as do lines 1 and 5: the run is the run the page describes. About the retry itself it said that it did not know whether a 7B model uses a terse feedback, and it wrote down in advance what each outcome would mean. The nearest of those is \"*Line 6 NO:* the model does not use the feedback (or uses it to repeat itself)\", and what the page says to do then stands: \"the next thing to try is a different feedback, which would be a different test written first\". **{sum(READ)} of the 12 lines read yes; lines 6, 8, 9 and 11 read NO, and no line is \"not run\".** The headline (lines 9 and 11) reads NO on both: on the 64 jobs that no earlier run had seen, {pc(S['H1']['share'])} of the steps are exact with the retry and {pc(S['H0']['share'])} without it, and the exact steps per GPU hour are {num(round(S['H1']['eph']))} with it against {num(round(S['H0']['eph']))} without it. Line 8 was expected NO (35 %). Line 6 reads NO only because of `R8` ({t3[8]['ex']} of {S['R8']['n']} steps exact in the third run and in `R8`); at 32 and 64 jobs it is higher ({S['R32']['ex']} against {t3[32]['ex']} steps and {num(S['R64']['ex'])} against {num(t3[64]['ex'])}), and that gain is job 19 ({lost(third_cell(32))[19]} steps) at both, plus {len(g3)} steps of job 3 at 64 jobs (steps {g3[0]} to {g3[-1]}), which are exact for another reason (below).")
    A("")
    A("## The 12 lines")
    A("")
    A("| # | my probability | what was measured | read |")
    A("|---|---|---|---|")
    lines.extend(table_lines(res))
    A("")
    A(f"**{sum(READ)} of 12 lines read yes, lines 6, 8, 9 and 11 read NO, and no line is \"not run\".** The page did not name this exact pattern; its nearest entry is the one for line 6 (above). The four NO stood at 85 % (line 6), 55 % (line 9), 40 % (line 11) and 35 % (line 8) on the page.")
    A("")
    A("## What the page's rule lets this Result claim")
    A("The page's rule, applied line by line to what was read:")
    A("* **Lines 2, 3 and 4 read yes**, so the figures of the cells are claimed as read.")
    A(f"* **Line 9 reads NO and line 12 reads yes.** The page allows the claim that the retry raises the exactness only from the jobs that no earlier run had seen (lines 9 and 12). It did not raise it there ({pc(S['H1']['share'])} with the retry, {pc(S['H0']['share'])} without), and no job was made worse by it. **This Result does not claim that the retry raises the exactness.**")
    A(f"* **Lines 6 to 8 are about the jobs the slips were read on** (jobs 0 to 63), and say what the retry does to them: it put right one of the four slips that were read (job 19, step 12) and none of the other three, and the exact share at 64 jobs is {pc(S['R64']['share'])} against {pc(t3[64]['share'])} in the third run (another Colab session, same replies up to each job's first retried step).")
    A("* **Never**, as the page says: a multiple of capacity, a count of GPUs saved, dollars or a percentage of cost, \"100 percent\" for the arm, \"proven\", a statement about any provider, or a statement about more than one retry.")
    A("")
    A(f"**The sentence the page allows, with this run's numbers.** On the 64 jobs that no earlier run had seen (jobs 64 to 127), on one Tesla T4 (Colab), with vLLM {e['version']}, `{e['model']}`, a cache of {num(e['capacity_tokens'])} tokens, 30-step jobs and every reply constrained by one regular expression, one retry with the framework's one-line feedback left the exactness at {pc(S['H1']['share'])} where it was {pc(S['H0']['share'])} without it ({num(S['H1']['ex'])} and {num(S['H0']['ex'])} of {num(S['H1']['n'])} steps), the seconds per step at {S['H1']['sps']:.3f} against {S['H0']['sps']:.3f}, and the exact steps per GPU hour at {num(round(S['H1']['eph']))} against {num(round(S['H0']['eph']))} (both in the same session). The retry asked {S['H1']['asked']} times and put none of them right. On jobs 0 to 63, where the four slips had been read, it put right {put_four} of the four (job 19) and the exact share at 64 jobs is {pc(S['R64']['share'])} against {pc(t3[64]['share'])} in the third run.")
    A("")
    A(f"## What the {len(rs)} second requests did")
    A(f"A second request is the first request, the first reply as the assistant's turn, and the framework's feedback as the user's turn (for example, in `R8`, job 3, step 16: `{rs[0]['feedback']}`). {len(rs)} second requests were made in the four cells with the retry, {n_retry_rounds} second rounds (jobs asking in the same step share a round), costing {retry_s:.2f} seconds in all. Every second reply was accepted by the strict parser.")
    A("")
    lines.extend(table_asked(res))
    A("")
    A(f"* **{len(behind)} of the {len(rs)} were asked after the job had already slipped, and in {sum(1 for r in behind if r['first_ok'])} of those the first reply was, letter for letter, the oracle's own reply.** Each of these 14 replies is legal in the oracle's state at its step and was refused in the state the job held, which an earlier slip had left different from the oracle's: they are job 3 at step 28 and job 4 at steps 19, 24 and 28 (in `R8`, `R32` and `R64`) and job 101 at steps 7 and 27 (in `H1`). {ok1o['same']} of the {len(ok1)} came back as the same reply; {len(ok1) - ok1o['same']} came back changed to another reply that was refused as well (`R32`, job 4, step 19: `dep#7` to `dep#0`; `H1`, job 101, step 7: `ADD` to `DEL`). **The rule cannot tell these from a real slip:** it knows that the gate refused a reply that parsed, and nothing about whether the reply is what the request asked.")
    A(f"* **The other {len(fresh)} first replies were not the oracle's:** {frs['put right']} were put right (job 19, step 12, in `R32` and `R64`: `{l19['first']}` became `{l19['second']}`, the oracle's reply); {frs['same']} came back as the same reply; {frs['admitted, wrong']} came back as `DEL[1,0]`, which the gate admits and which is wrong (job 4, step 18, in `R8`, `R32` and `R64`: the feedback said that the node (1,0) already exists, and the model deleted it); {frs['refused again']} came back changed and refused again (`H1`, job 122, step 22: `ADD` to `DEL`).")
    A(f"* **The \"exact after the step\" column counts steps, not repairs.** In `R64` it is {S['R64']['exact_after']} and the replies put right are {o['put right'] - 1}: the other step is job 3, step 28, where the same refused reply `DEL[4,4]` came back and the job is exact anyway, because the node (4,4) that the oracle deletes there was never added in that job (its add at step 16 was refused), so the refusal leaves the same graph as the oracle's.")
    A(f"* **`R8` asked {S['R8']['asked']} times where the page's replay of the third run gave {len(fires8)}.** The extra two are job 4, steps 19 and 24: after step 18 was answered with the admitted `DEL[1,0]`, job 4 stood in another wrong state than in the third run, and its later replies were refused at other steps (the rule fired at steps 18, 19, 24 and 28 instead of 18 and 28).")
    A(f"* **Time.** A second round took {S['R8']['per_round']:.2f}, {S['R32']['per_round']:.2f}, {S['R64']['per_round']:.2f} and {S['H1']['per_round']:.2f} seconds on average in `R8`, `R32`, `R64` and `H1` ({retry_s / n_retry_rounds:.2f} over all), and the rounds of the retry were {pc(S['R8']['retry_share'])}, {pc(S['R32']['retry_share'])}, {pc(S['R64']['retry_share'])} and {pc(S['H1']['retry_share'])} of the seconds of their cells. In the pair, `H1` took {S['H1']['wall']:.1f} seconds against {S['H0']['wall']:.1f} for `H0` ({sg((S['H1']['wall'] / S['H0']['wall'] - 1) * 100)}), and the retry bought no exact step there, so the exact steps per hour fell by {abs((S['H1']['eph'] / S['H0']['eph'] - 1) * 100):.1f} %.")
    A("")
    A(f"The {len(rs)} second requests, one row each (\"why the gate refused it\" is the first word of the gate's reason; \"first reply was the oracle's own\" means that it is, letter for letter, what the oracle writes at that step):")
    A("")
    lines.extend(table_retried(res))
    A("")
    A("## The eight first slips")
    A(f"Over the 128 jobs (jobs 0 to 63 in `R64`, jobs 64 to 127 in `H1`), **{len(slips)} jobs ever slip**, and the misses of such a job after its first slip happen in a state that is no longer the oracle's (the exact-state arm keeps the state it holds, so a step that was not done stays not done); they are read here as the consequence of the first slip and not as new slips. Job 101 lost {lost(c['H1'])[101]} of its 30 steps from one slip at step 4. The first slip of each job, with the request, what the oracle writes and what the model wrote:")
    A("")
    lines.extend(table_slips(res))
    A("")
    kc = {k: sorted(j for j, v in kinds.items() if v == k) for k in ("shape", "number", "extra delete")}; scope_flagged = scope_flags(res)
    A(f"They are three kinds. **A service added as an edge from the service to itself** (jobs {', '.join(map(str, kc['shape']))}): the request asks for a service, and the model writes the edge form from that service to itself with a made-up name ({', '.join('`' + re.search(r':([A-Za-z0-9_]+)#', f).group(1) + '`' for j, _, s, f, _ in slips if j in kc['shape'])}) and the value after the `#`; the oracle writes `ADD[x,y:v]`. **A wrong number** (jobs {', '.join(map(str, kc['number']))}): the model copies a coordinate or a value wrongly (in job 4 it wrote the node that the state already holds). **A legal extra** (jobs {', '.join(map(str, kc['extra delete']))}): the model adds the right edge and also deletes another edge between the same two services; the reply is legal, so the gate admits it and the rule never fires. Of the six first slips that the gate refused (jobs 3, 4, 19, 20, 101, 122), the retry put right one (job 19). It could not reach the two that the gate admitted. **What the framework's other checks see of these eight:** its scope filter (`scope.py`: a change is in scope if the coordinates it touches are written in the request) flags the wrong coordinate in jobs {', '.join(map(str, scope_flagged))} and nothing in the other five (in the shape slips both ends of the edge are the named service, and in the two legal extras the deleted edge is between two named services). The serving loop did not use the scope filter, and the feedback of the retry did not carry its reason; this is what it would have seen, not a claim about what it would have done.")
    A("")
    A("Where the steps were lost (steps after which the held state was not the oracle's):")
    A("")
    lines.extend(table_missed(res))
    A("")
    lines.extend(table_third(res))
    A("")
    A(f"At 64 jobs the retry cut the lost steps from {S['R64']['n'] - t3[64]['ex']} to {S['R64']['n'] - S['R64']['ex']}: job 19 from {lost(third_cell(64))[19]} to 0, job 3 from {lost(third_cell(64))[3]} to {lost(c['R64'])[3]}, the others unchanged. The pair lost {S['H0']['n'] - S['H0']['ex']} steps in both cells, in the same two jobs (101 and 122), and the {sum(1 for _, a, b in pr if a == b)} of its {len(pr)} jobs hold the right state after the same number of steps with and without the retry.")
    A("")
    A("## Every cell")
    A("")
    lines.extend(table_cells(res))
    A("")
    A("The third run's cells were in another Colab session and the page says a cell of this run is not compared with them for time; the seconds above are for reading each cell and the pair `H0`/`H1`, which ran in one session.")
    A("")
    A("## Where my expectation was off, and where it was right")
    A(f"* **Lines 1 to 5 (98, 92, 85, 93 and 90 %): the run is the run the page describes.** {num(sum(S[k]['n'] + S[k]['asked'] for k in S))} requests, every one with its cached count; {num(sum(S[k]['replies'] for k in S))} replies, every one accepted by the strict parser, none at 96 new tokens (the longest has {max(S[k]['longest'] for k in S)}); the first replies are the third run's up to each job's first retried step, and `H1`'s are `H0`'s; the {len(rs)} second requests were all made where the rule says and `H0` made none; the retry asked on at most {pc(max(S[k]['asked'] / S[k]['n'] for k in WITH_RETRY))} of the steps of a cell (`R8`, just under the limit of 3 %). On the replies of `H0` the rule would fire at exactly the {len(fires_h0(res))} steps where `H1` asked.")
    A(f"* **Line 6 (85 %) reads NO,** at 8 jobs only (equal), and the model did not use the feedback in the way the page hoped: of the {len(rs)} second requests {o['same']} came back as the same reply, and {o['put right']} were put right, both the same step of the same job. The page put line 6 at 85 %, so this is the surprise of the run; it also said that it did not know whether this 7B model would use the feedback.")
    A(f"* **Line 7 (75 %) reads yes and line 8 (35 %) NO:** {pc(S['R64']['share'])} at 64 jobs against a ceiling of 99.7 %; the {S['R64']['n'] - S['R64']['ex']} steps still lost are jobs 3, 4, 17, 20 and 62.")
    A(f"* **Line 9 (55 %) and line 11 (40 %) read NO.** The page expected about six slips in 64 new jobs, four of them of the refused kind, and a retry that fixes two thirds of them. There were two slips (jobs 101 and 122), both of the refused kind, the retry was asked {S['H1']['asked']} times for them and put none right. With nothing gained, the cost of the second rounds is all there is: {sg((S['H1']['eph'] / S['H0']['eph'] - 1) * 100)} in exact steps per hour.")
    A(f"* **Line 10 (88 %) reads yes,** {S['H1']['sps'] / S['H0']['sps']:.3f} times; **line 12 (70 %) reads yes,** with no job worse (and none better).")
    A(f"* **The side predictions, read afterwards as descriptions.** (a) the number of second requests in `R64` is between 4 and 12 (85 %): true, {S['R64']['asked']}. (b) the second reply of job 19, step 12 is `ADD[3,0:4]` (55 %): true. (c) every second reply parses (97 %): true, {len(rs)} of {len(rs)}. (d) at least 60 of the 64 jobs of `R64` hold the right state at every step (60 %): **false,** {S['R64']['whole']} (the third run: 58). (e) a second round takes 0.4 to 2.0 seconds on average (80 %): true, {retry_s / n_retry_rounds:.2f} over the {n_retry_rounds} second rounds ({min(S[k]['per_round'] for k in WITH_RETRY):.2f} to {max(S[k]['per_round'] for k in WITH_RETRY):.2f} by cell). (f) `H0` is between 90 and 99 percent exact (80 %): true, {pc(S['H0']['share'])}. (g) the five cells take 12 to 25 minutes in all, loading excluded (70 %): true by the saved seconds ({wall / 60:.1f} minutes around the engine's calls); the time the cells took in Colab, with the prompts built and the files written, is not in the file.")
    A("")
    A("## On the text pasted after the run")
    f1 = {(c_["arm"], c_["K"]): c_ for c_ in _load("first", FIRST)["cells"]}; s2 = {(c_["arm"], c_["K"]): c_ for c_ in _load("second", SECOND)["cells"]}; s3 = {(c_["arm"], c_["K"]): c_ for c_ in third()["cells"]}
    sh = lambda d, a, k: st(d[(a, k)])
    A("On 20 Oct the owner pasted a text (author unknown) that reads this run's file. Its sentences, quoted, checked against the files:")
    A(f"* \"pushing you up to 97.2%\", and the run before it \"hovering around 69%–78%\": **the retry did not do that.** 97.2 % is the exact share of `S` at 64 jobs in the second and in the third run, which had no retry. The first run (no grammar) had {pc(sh(f1, 'S', 8)['share'])}, {pc(sh(f1, 'S', 32)['share'])} and {pc(sh(f1, 'S', 64)['share'])} at 8, 32 and 64 jobs; the grammar of the second run took it to {pc(sh(s2, 'S', 8)['share'])}, {pc(sh(s2, 'S', 32)['share'])} and {pc(sh(s2, 'S', 64)['share'])}. With the retry it is {pc(S['R8']['share'])}, {pc(S['R32']['share'])} and {pc(S['R64']['share'])}, and on the 64 new jobs {pc(S['H1']['share'])} with it and {pc(S['H0']['share'])} without.")
    A(f"* \"5.4 times more correct, usable state transitions per hour\": **that divides two different sizes:** {num(round(sh(s3, 'S', 64)['eph']))} (`S` at 64 jobs) over {num(round(sh(s3, 'KC', 8)['eph']))} (`KC` at 8 jobs) is {sh(s3, 'S', 64)['eph'] / sh(s3, 'KC', 8)['eph']:.2f}. The ratios at the same number of jobs are the third run's: {sh(s3, 'S', 8)['eph'] / sh(s3, 'KC', 8)['eph']:.2f} at 8 jobs (one session) and {sh(s3, 'S', 32)['eph'] / sh(s2, 'KC', 32)['eph']:.1f} and {sh(s3, 'S', 64)['eph'] / sh(s2, 'KC', 64)['eph']:.1f} at 32 and 64 jobs (the kept chat from the second run, another session), each with the exactness of both arms beside it. Neither a \"540% increase in hardware capacity\", nor \"pack 8× more agents onto a single server\", nor dollars were measured.")
    A(f"* Its section on \"Lines 12 and 13\": those are lines of the third page (13 lines). This page has 12 lines, and the NO are 6, 8, 9 and 11.")
    A(f"* \"Seed 20261017000, Step 16\": seed 20261017000 is job 0, which is exact at every step in every cell. The step described is job 3 (seed 20261017003), step 16.")
    A(f"* \"the ultimate, empirical proof that the AI industry must adopt\" a tensor-level architecture: **nothing in this run can show that.** A 7B model that repeats a reply after a one-line feedback says that this feedback did not help this model on these slips; a different wording, a stronger model or a check of the reply against the request are not ruled out by it, and the tensor-level roadmap is neither tested nor touched.")
    A("")
    A("## What this finds, in plain words")
    A(f"No step of the {num(sum(S[k]['n'] for k in S))} left the held graph malformed ({sum(S[k]['corrupted'] for k in S)} corrupted), and the exactness is {pc(S['R64']['share'])} at 64 jobs and {pc(S['H0']['share'])} on jobs nobody had read. What is left is a small number of first slips in a few jobs ({len(slips)} of 128), of three kinds. **One retry with the framework's one-line feedback put right one of the six that the gate refused (job 19) and none of the other cases,** and it cost {pc(S['H1']['retry_share'])} of the seconds of its cell in the pair. Most of the second requests were not about a slip at all: {len(behind)} of the {len(rs)} came after the job had slipped, and {sum(1 for r in behind if r['first_ok'])} of those were asked about a reply that was the oracle's own, which the rule cannot tell from a slip. On this model, on these slips, that feedback is not enough to move the headline. It does not say what a different feedback, a stronger model or more than one retry would do, and it says nothing against the gate, the repair or the loop as designed.")
    A("")
    A("## What this does not show")
    A("More than one retry (a budget sweep is not run); another feedback wording (this was one, `structured_imperative`; the framework has three others); another model, GPU, engine or cache size; dollars; a capacity in jobs per GPU; energy; the kept chat with a retry (not run); retries in a real server, where they would be mixed with other traffic; the scope filter's reason carried in the feedback (the filter flags 3 of the 8 first slips, above; it was not used here); that the retry fixes slips of kinds not seen here; replies in ordinary words; jobs longer than 30 steps. The third run's ratios (37.7 and 46.3) were not re-measured here. It says nothing about the tensor-level roadmap.")
    return "\n".join(lines)


# ================================================================ the tests
# ---------------------------------------------------------------- 1 the files
def test_the_files_are_the_ones_the_page_is_about_and_every_job_replays_from_its_reply_texts_through_the_frameworks_own_rules():
    res = run(); assert _sha(RES) == FILE_SHA256, "the saved result file must not change"; assert _sha(THIRD) == THIRD_SHA256 == res["third_run_sha256"] and res["third_run_file"] == "axi_provider_bounded.json"
    assert _sha(SECOND) == SECOND_SHA256 and _sha(FIRST) == FIRST_SHA256 and _sha(MODULE) == MODULE_SHA256 == res["module_sha256"], "the module that merged the cells must be the one in the repository"
    assert set(res) == {"meta", "cells", "third_run_file", "third_run_sha256", "module_sha256", "pinned_sha256"} and res["pinned_sha256"] == PINS
    for name, path in (("provider_side", "provider_side.py"), ("constrained", "provider_side_constrained.py"), ("bounded", "provider_side_bounded.py")): assert _sha(os.path.join(HERE, "..", "axi", "experiments", path)) == PINS[name]
    for name, path in (("gate", ("engine", "gate.py")), ("diagnostics", ("engine", "diagnostics.py")), ("agent_loop", ("experiments", "agent_loop.py")), ("cascade", ("experiments", "cascade.py"))): assert _sha(os.path.join(HERE, "..", "axi", *path)) == PINS[name], name
    m = res["meta"]; e = m["engine"]; con = e["constraint"]
    assert {k: v for k, v in e.items() if k != "constraint"} == dict(kind="vllm", name="Qwen/Qwen2.5-7B-Instruct-AWQ", version="0.31.0", gpu="Tesla T4", capacity_tokens=80000, capacity_source="override", block_size=16, kv_bytes_per_token=57344, model="Qwen/Qwen2.5-7B-Instruct-AWQ", python="3.13.15")
    assert con["kind"] == "regex" and con["regex"] == B.REGEX and con["regex_sha256"] == REGEX_SHA256 == hashlib.sha256(B.REGEX.encode()).hexdigest() and con["name_cap"] == 16
    assert (m["steps"], m["seed"], m["policy"], m["hint_rules"], m["name_cap"], m["model"]) == (30, SEED, "repair_m6", False, 16, "Qwen/Qwen2.5-7B-Instruct-AWQ") and m["retry_rule"] == dict(feedback="structured_imperative", degree_threshold=4, max_retries=1)
    assert m["cells"] == {"R8": [8, 0, True], "R32": [32, 0, True], "R64": [64, 0, True], "H0": [64, 64, False], "H1": [64, 64, True]}
    assert [c["label"] for c in res["cells"]] == ORDER == R.ORDER and R.merge_cells(res["cells"]) == {k: res[k] for k in ("meta", "cells")}
    for c in res["cells"]:
        f, k = JOBS[c["label"]]; rt = c["label"] in WITH_RETRY
        assert (c["arm"], c["K"], c["first_episode"], c["retry"], c["steps"], c["seed"], c["policy"], c["hint_rules"], c["engine"]) == ("S", k, f, rt, 30, SEED, "repair_m6", False, e)
        assert [s["seed"] for s in c["sessions"]] == [SEED * 1000 + f + i for i in range(k)] and all(s["script_steps"] == 30 and s["stopped"] is None and len(s["log"]) == 30 for s in c["sessions"])
        L = logs(c); asked = [x for x in L if x.get("retried")]; assert [x["t"] for s in c["sessions"] for x in s["log"]] == list(range(1, 31)) * k
        assert abs(secs(c) - c["wall_seconds"]) < 1e-6 and [r["round"] for r in c["rounds"]] == list(range(1, len(c["rounds"]) + 1)) and len([r for r in c["rounds"] if r["kind"] == "first"]) == 30
        assert sum(1 for r in c["rounds"] if r["kind"] == "retry") <= len(asked) and (len(asked) == 0) == (not any(r["kind"] == "retry" for r in c["rounds"]))
        fp = sum(x["first_n_prompt"] if x.get("retried") else x["n_prompt"] for x in L); fc = sum(x["first_n_cached"] if x.get("retried") else x["n_cached"] for x in L); fn = sum(x["first_n_new"] if x.get("retried") else x["n_new"] for x in L)
        assert (sum(r["prompt_tokens"] for r in c["rounds"] if r["kind"] == "first"), sum(r["cached_tokens"] for r in c["rounds"] if r["kind"] == "first"), sum(r["new_tokens"] for r in c["rounds"] if r["kind"] == "first")) == (fp, fc, fn)
        assert (sum(r["prompt_tokens"] for r in c["rounds"] if r["kind"] == "retry"), sum(r["cached_tokens"] for r in c["rounds"] if r["kind"] == "retry"), sum(r["new_tokens"] for r in c["rounds"] if r["kind"] == "retry")) == tuple(sum(x[key] for x in asked) for key in ("n_prompt", "n_cached", "n_new"))
        assert all(0 <= x["n_cached"] <= x["n_prompt"] and 0 < x["n_new"] < 96 for x in L) and all(0 <= x["first_n_cached"] <= x["first_n_prompt"] and 0 < x["first_n_new"] < 96 for x in asked)
        assert all(re.fullmatch(B.REGEX, x["text"]) for x in L) and all(re.fullmatch(B.REGEX, x["first_text"]) for x in asked), "every reply, first and second, is inside the capped language"
        assert all(bool(x["exact"]) == (x["distance"] == 0) for x in L) and all(x["parsed"] or not x["admitted"] for x in L)
        for s in c["sessions"]:                                                                         # the job played again through the framework's session, rule and feedback
            for (info, fb, rec), x in zip(replay_job(c, s), s["log"]):
                assert rec == (x["parsed"], x["admitted"], x["corrupted"], x["exact"], x["distance"]), (c["label"], job_of(s), x["t"])
                if rt:
                    assert info["fires"] == bool(x.get("retried")), (c["label"], job_of(s), x["t"])
                    if x.get("retried"):
                        assert (info["parsed"], info["admitted"], info["repairable"], info["reason"]) == (x["first_parsed"], x["first_admitted"], x["first_repairable"], x["first_reason"]) and fb == x["feedback"]
                else: assert not x.get("retried")
    assert sum(len(logs(c)) for c in res["cells"]) == 6960 and round(sum(c["wall_seconds"] for c in res["cells"]), 1) == 818.3
    assert fires_h0(res) == [(job_of(s), x["t"]) for s in cells(res)["H1"]["sessions"] for x in s["log"] if x.get("retried")], "on H0's replies the rule fires exactly where H1 asked"


# ---------------------------------------------------------------- 2 the lines
def test_the_twelve_lines_read_as_the_result_says_by_the_code_and_by_hand_and_the_rule_about_claims_is_followed():
    res = run(); v = R.verdict_retry(res["cells"], third()["cells"]); assert [x for _, _, _, x in v] == READ and len(PROBS) == 12 and R.RL_HEADLINE == (9, 11) and len(R.RL_LINES) == 12
    assert [(i, tag, text) for i, tag, text, _ in v] == [(i + 1, R.RL_LINES[i][0], R.RL_LINES[i][1]) for i in range(12)]
    assert sum(READ) == 8 and [i + 1 for i, x in enumerate(READ) if not x] == [6, 8, 9, 11] and [READ[i - 1] for i in R.RL_HEADLINE] == [False, False]
    assert lines_by_hand(res) == READ, "the twelve lines from the raw numbers, by hand"
    assert [int(m_.group(1)) for m_ in re.finditer(r"\n\d+\. \[\w+\] .*\((\d+) %\)(?=\n)", _head())] == PROBS
    ca = R.claims_allowed(v); assert ca == dict(base=True, constraint_held=True, first_replies_as_before=True, fired_by_the_rule=True, same_jobs={6: False, 7: True, 8: False}, new_jobs_exactness={9: False, 12: True}, time_and_cost_in_one_session={10: True, 11: False})
    S = {k: st(v_) for k, v_ in cells(res).items()}
    assert (S["R8"]["ex"], S["R32"]["ex"], S["R64"]["ex"], S["H0"]["ex"], S["H1"]["ex"]) == (212, 927, 1889, 1884, 1884) and (S["R8"]["asked"], S["R32"]["asked"], S["R64"]["asked"], S["H0"]["asked"], S["H1"]["asked"]) == (7, 9, 8, 0, 5)
    assert [st(third_cell(k))["ex"] for k in (8, 32, 64)] == [212, 908, 1867]
    R_ = _result()
    for needle in ("**Lines 2, 3 and 4 read yes**", "**Line 9 reads NO and line 12 reads yes.**", "**This Result does not claim that the retry raises the exactness.**", "8 of the 12 lines read yes; lines 6, 8, 9 and 11 read NO"): assert needle in R_, needle


# ---------------------------------------------------------------- 3 the page
def test_the_part_above_the_result_is_what_was_written_before_the_run():
    head = _head(); assert _h(head) == PAGE_SHA256 and _text().count(RESULT_MARK) == 1 and head.rstrip("\n").endswith("\n\n---")


def test_the_page_carries_exactly_the_result_this_file_renders_and_its_tables_say_what_the_rules_say():
    res = run(); text = _text(); assert text == _head().rstrip("\n") + render(res).rstrip("\n") + "\n"
    R_ = _result(); rows = [ln for ln in R_.split("\n") if re.match(r"\| \d+ \| \d+ % \|", ln)]; assert len(rows) == 12
    for i, ln in enumerate(rows, 1):
        cs = [c.strip() for c in ln.strip("|").split("|")]; assert int(cs[0]) == i and cs[1] == f"{PROBS[i - 1]} %" and cs[2] == measured(res, i) and cs[3] == ("yes" if READ[i - 1] else "NO"), i
    for tbl in (table_cells(res), table_asked(res), table_retried(res), table_slips(res), table_missed(res), table_third(res)):
        for ln in tbl: assert ln in R_.split("\n"), ln
        assert "\n".join(tbl) in R_


# ---------------------------------------------------------------- 4 the figures
def test_the_figures_quoted_in_the_result_are_recounted_from_the_records():
    res = run(); R_ = _result(); c = cells(res)
    for needle in ("vLLM 0.31.0", "GPU Tesla T4", "Python 3.13.15", FILE_SHA256, MODULE_SHA256[:16], THIRD_SHA256[:16], REGEX_SHA256[:16], "20 Oct 2026", "**All five planned cells ran**", "add up to 818.3, which is 13.6 minutes",
                   "88.3 %", "96.6 %", "98.4 %", "98.1 %", "30,951", "31,359", "30,459", "19,846", "27,537", "212 of 240", "927 against 908", "1,889 against 1,867", "29 second requests", "26 second rounds", "12.03 seconds",
                   "17 of the 29 were asked after the job had already slipped", "in 14 of those the first reply was, letter for letter, the oracle's own reply", "12 of the 14 came back as the same reply", "The other 15 first replies were not the oracle's",
                   "2 were put right", "9 came back as the same reply", "3 came back as `DEL[1,0]`", "1 came back changed and refused again", "`ADD[(3,0)->(3,0):4tworowsdown4col#4]` became `ADD[3,0:4]`", "`R8` asked 7 times where the page's replay of the third run gave 5",
                   "Job 101 lost 27 of its 30 steps from one slip at step 4", "**8 jobs ever slip**", "job 19 from 19 to 0, job 3 from 15 to 12", "0.40, 0.45, 0.44 and 0.59 seconds", "6.2 %", "2.9 %", "1.4 %", "(d) at least 60 of the 64 jobs of `R64` hold the right state at every step (60 %): **false,** 59",
                   "0 corrupted", "on jobs nobody had read", "5.43", "76.2 %", "78.2 %", "69.2 %", "3.40", "37.7", "46.3", "seed 20261017003", "seed 20261017000 is job 0"):
        assert needle in R_, needle
    # plain recounts
    for lab, (steps, ex, asked, rounds) in dict(R8=(240, 212, 7, 6), R32=(960, 927, 9, 8), R64=(1920, 1889, 8, 7), H0=(1920, 1884, 0, 0), H1=(1920, 1884, 5, 5)).items():
        L = logs(c[lab]); assert (len(L), sum(1 for x in L if x["exact"]), sum(1 for x in L if x.get("retried")), sum(1 for r in c[lab]["rounds"] if r["kind"] == "retry")) == (steps, ex, asked, rounds), lab
    wall = {lab: sum(r["seconds"] for r in c[lab]["rounds"]) for lab in ORDER}
    assert [round(wall[l], 1) for l in ORDER] == [38.5, 121.2, 223.3, 216.3, 219.1] and [round(3600 * sum(1 for x in logs(c[l]) if x["exact"]) / wall[l]) for l in ORDER] == [19846, 27537, 30459, 31359, 30951]
    assert round(sum(r["seconds"] for l in WITH_RETRY for r in c[l]["rounds"] if r["kind"] == "retry"), 2) == 12.03 and sum(1 for l in WITH_RETRY for r in c[l]["rounds"] if r["kind"] == "retry") == 26
    assert [round(sum(r["seconds"] for r in c[l]["rounds"] if r["kind"] == "retry") / sum(1 for r in c[l]["rounds"] if r["kind"] == "retry"), 2) for l in WITH_RETRY] == [0.4, 0.45, 0.44, 0.59]
    assert [round(100 * sum(r["seconds"] for r in c[l]["rounds"] if r["kind"] == "retry") / wall[l], 1) for l in WITH_RETRY] == [6.2, 2.9, 1.4, 1.4]
    assert round(wall["H1"] / wall["H0"], 3) == 1.013 and round(wall["H1"] / 1920, 3) == 0.114 and round(wall["H0"] / 1920, 3) == 0.113 and 1884 / 1920 == 0.98125


# ---------------------------------------------------------------- 5 the second requests
def test_the_second_requests_are_counted_by_what_came_back_and_by_whether_the_first_reply_was_the_oracles():
    res = run(); rs = retried(res); assert len(rs) == 29 and sorted((r["cell"], r["job"], r["step"]) for r in rs) == sorted(
        [("R8", 3, 16), ("R8", 3, 17), ("R8", 3, 28), ("R8", 4, 18), ("R8", 4, 19), ("R8", 4, 24), ("R8", 4, 28), ("R32", 3, 16), ("R32", 3, 17), ("R32", 3, 28), ("R32", 4, 18), ("R32", 4, 19), ("R32", 4, 24), ("R32", 4, 28),
         ("R32", 19, 12), ("R32", 20, 30), ("R64", 3, 16), ("R64", 3, 28), ("R64", 4, 18), ("R64", 4, 19), ("R64", 4, 24), ("R64", 4, 28), ("R64", 19, 12), ("R64", 20, 30), ("H1", 101, 4), ("H1", 101, 7), ("H1", 101, 13), ("H1", 101, 27), ("H1", 122, 22)])
    assert all(strict_parse(r["second"]) is not None for r in rs) and len({(r["job"], r["step"]) for r in rs}) == 14
    assert outcome_counts(rs) == {"same": 21, "refused again": 3, "admitted, wrong": 3, "put right": 2}
    assert sum(1 for r in rs if r["behind"]) == 17 and sum(1 for r in rs if r["first_ok"]) == 14 and all(r["behind"] for r in rs if r["first_ok"])
    ok1 = [r for r in rs if r["first_ok"]]; assert outcome_counts(ok1) == {"same": 12, "refused again": 2, "admitted, wrong": 0, "put right": 0}
    fresh = [r for r in rs if not r["first_ok"]]; assert len(fresh) == 15 and outcome_counts(fresh) == {"same": 9, "refused again": 1, "admitted, wrong": 3, "put right": 2}
    assert sorted((r["cell"], r["job"], r["step"]) for r in rs if outcome(r) == "put right") == [("R32", 19, 12), ("R64", 19, 12)] and all(r["second"] == "ADD[3,0:4]" for r in rs if outcome(r) == "put right")
    assert sorted((r["cell"], r["job"], r["step"]) for r in rs if outcome(r) == "admitted, wrong") == [("R32", 4, 18), ("R64", 4, 18), ("R8", 4, 18)] and all(r["second"] == "DEL[1,0]" for r in rs if outcome(r) == "admitted, wrong")
    assert sorted((r["cell"], r["job"], r["step"]) for r in rs if outcome(r) == "refused again") == [("H1", 101, 7), ("H1", 122, 22), ("R32", 4, 19)]
    r64 = next(r for r in rs if (r["cell"], r["job"], r["step"]) == ("R64", 3, 28)); assert r64["second"] == r64["first"] == "DEL[4,4]" and r64["exact"] and r64["first_ok"] and "node (4, 4) not in host" in r64["reason"]
    assert [r["step"] for r in rs if r["cell"] == "R8" and r["job"] == 4] == [18, 19, 24, 28]
    assert third_fires(8) == [(3, 16), (3, 17), (3, 28), (4, 18), (4, 28)]                              # the rule on the third run's replies at 8 jobs: the page's five
    assert job3_gain(res) == [28, 29, 30]


# ---------------------------------------------------------------- 6 the first slips
def test_the_eight_first_slips_are_read_from_the_records_and_each_is_of_the_kind_the_result_says():
    res = run(); sl = first_slips(res); assert [j for j, *_ in sl] == [3, 4, 17, 19, 20, 62, 101, 122] and [s for _, _, s, _, _ in sl] == [16, 18, 20, 12, 30, 12, 4, 22]
    assert {j: slip_kind(f, oracle(j)[s - 1]) for j, _, s, f, _ in sl} == KINDS
    want = {3: ("Add a service at (4,4) with value 6.", "ADD[4,4:6]", "ADD[(4,4)->(4,4):value#6]"), 4: ("Add a service at (0,0) with value 3.", "ADD[0,0:3]", "ADD[1,0:6]"),
            17: ("Add a dependency edge from (0,4) to (2,3) with relation dep and weight 4.", "ADD[(0,4)->(2,3):dep#4]", "DEL[(2,3)->(0,4):dep#6] ADD[(0,4)->(2,3):dep#4]"),
            19: ("Add a service at (3,0) with value 4.", "ADD[3,0:4]", "ADD[(3,0)->(3,0):4tworowsdown4col#4]"), 20: ("Add a dependency edge from (1,1) to (2,1) with relation dep and weight 3.", "ADD[(1,1)->(2,1):dep#3]", "ADD[(1,5)->(2,1):dep#3]"),
            62: ("Add a dependency edge from (1,1) to (1,0) with relation dep and weight 9.", "ADD[(1,1)->(1,0):dep#9]", "DEL[(1,1)->(1,0):owns#5] ADD[(1,1)->(1,0):dep#9]"), 101: ("Add a service at (3,4) with value 2.", "ADD[3,4:2]", "ADD[(3,4)->(3,4):2alue#2]"),
            122: ("Add a dependency edge from (0,0) to (2,2) with relation dep and weight 3.", "ADD[(0,0)->(2,2):dep#3]", "ADD[(0,0)->(2,0):dep#3]")}
    for j, lab, s, f, x in sl: assert (request_sentence(j, s), oracle(j)[s - 1], f) == want[j], j
    # every earlier step of the job is exact, and the two admitted ones are admitted and not exact
    for j, lab, s, f, x in sl:
        job = next(q for q in cells(res)[lab]["sessions"] if job_of(q) == j)["log"]; assert all(y["exact"] for y in job[:s - 1])
        assert x.get("retried") == (j not in (17, 62)) and (x["admitted"] if j in (17, 62) else True)
    c = cells(res); assert lost(c["R64"]) == {3: 12, 4: 13, 17: 4, 20: 1, 62: 1} and lost(c["H1"]) == {101: 27, 122: 9} and lost(c["H0"]) == lost(c["H1"])
    assert lost(third_cell(64)) == {3: 15, 4: 13, 17: 4, 19: 19, 20: 1, 62: 1} and sum(lost(third_cell(64)).values()) == 53 and sum(lost(c["R64"]).values()) == 31 and lost(third_cell(32))[19] == 19
    pr = pairs(res); assert len(pr) == 64 and sum(1 for _, a, b in pr if a == b) == 64 and sum(a for _, a, _ in pr) == 1884
    assert all(strict_parse(oracle(j)[s - 1]) is not None for j, _, s, _, _ in sl)
    # the 14 first replies that are the oracle's own are legal in the oracle's state at their step (so the state the job held is what made the gate refuse them)
    for r in [r for r in retried(res) if r["first_ok"]]:
        s_ = P.Session("S", SEED * 1000 + r["job"], STEPS)
        while s_.t < r["step"] - 1: rq = s_.request(); s_.accept(P.Reply(rq.stand_in_reply, 0, 0, 0))
        assert check(s_.live, strict_parse(r["first"])).ok and r["behind"], r
    # the framework's scope filter: it drops something in the wrong-number slips only
    assert scope_flags(res) == [4, 20, 122] and "flags the wrong coordinate in jobs 4, 20, 122 and nothing in the other five" in _result()
    for j, lab, s, f, x in sl:
        sc, dropped = contain_scope(strict_parse(f), named_services(request_sentence(j, s))); assert bool(dropped) == (j in (4, 20, 122)), j
    assert [re.search(r":([A-Za-z0-9_]+)#", f).group(1) for j, _, s, f, _ in sl if j in (3, 19, 101)] == ["value", "4tworowsdown4col", "2alue"] and "`value`, `4tworowsdown4col`, `2alue`" in _result()
    # no job other than the eight is ever asked again or inexact in R64 and H1
    assert sorted({job_of(s) for lab in ("R64", "H1") for s in c[lab]["sessions"] if any(not x["exact"] or x.get("retried") for x in s["log"])}) == [3, 4, 17, 19, 20, 62, 101, 122]


# ---------------------------------------------------------------- 7 the text pasted after the run
def test_what_the_result_says_about_the_pasted_text_is_recomputed_from_the_first_second_and_third_runs():
    f1 = {(c["arm"], c["K"]): st(c) for c in _load("first", FIRST)["cells"]}; s2 = {(c["arm"], c["K"]): st(c) for c in _load("second", SECOND)["cells"]}; s3 = {(c["arm"], c["K"]): st(c) for c in third()["cells"]}
    assert [round(f1[("S", k)]["share"] * 100, 1) for k in (8, 32, 64)] == [76.2, 78.2, 69.2] and [round(s2[("S", k)]["share"] * 100, 1) for k in (8, 32, 64)] == [88.3, 94.6, 97.2]
    assert [round(s3[("S", k)]["share"] * 100, 1) for k in (8, 32, 64)] == [88.3, 94.6, 97.2]
    assert f"{s3[('S', 64)]['eph'] / s3[('KC', 8)]['eph']:.2f}" == "5.43" and [f"{s3[('S', 8)]['eph'] / s3[('KC', 8)]['eph']:.2f}", f"{s3[('S', 32)]['eph'] / s2[('KC', 32)]['eph']:.1f}", f"{s3[('S', 64)]['eph'] / s2[('KC', 64)]['eph']:.1f}"] == ["3.40", "37.7", "46.3"]
    c = cells(run()); assert all(x["exact"] and not x.get("retried") for lab in ORDER for s in c[lab]["sessions"] if job_of(s) == 0 for x in s["log"]) and any(job_of(s) == 3 for s in c["R8"]["sessions"])
    assert len(R.RL_LINES) == 12 and len(B.BL_LINES) == 13


# ---------------------------------------------------------------- 8 claims
def test_the_result_claims_what_the_pages_rule_allows_and_nothing_more():
    R_ = _result(); low = R_.lower()
    assert "**Never**, as the page says: a multiple of capacity, a count of GPUs saved, dollars or a percentage of cost, \"100 percent\" for the arm, \"proven\", a statement about any provider, or a statement about more than one retry." in R_
    for w in ("proven", "gpus saved", "dollars", "capacity"):
        for m_ in re.finditer(w, low):
            ctx = low[max(0, m_.start() - 140): m_.end() + 90]; assert any(k in ctx for k in ("never", "neither", "not measured", "does not show", "not claim", "was not", "were not", "not ruled", "a capacity in", "cache size", "capacity source", "capacity_source", "capacity_tokens", "block override", "capacity ")), (w, ctx)
    assert "$" not in R_ and R_.count("100 percent") == 1 and R_.count("proven") == 1
    for q in ("does not claim that the retry raises the exactness", "does not say what a different feedback, a stronger model or more than one retry would do", "More than one retry (a budget sweep is not run)"): assert q in R_ or q in _norm(R_), q
    for q in ("I do not know whether this 7B model uses the feedback", "the next thing to try is a different feedback, which would be a different test written first", "Line 6 NO:* the model does not use the feedback (or uses it to repeat itself)"): assert _norm(q) in _norm(_head()), q


# ---------------------------------------------------------------- 10 the front pages
def test_the_front_pages_say_what_the_run_found_and_item_110_says_that_it_is_an_idea_that_was_not_run():
    res = run(); c = cells(res); S = {k: st(v) for k, v in c.items()}; t3 = {k: st(third_cell(k)) for k in (8, 32, 64)}
    cm = open(os.path.join(ROOT, "docs", "RAJNISH_CONCEPT_MAP.md")).read(); ve = open(os.path.join(ROOT, "docs", "VERIFIED.md")).read()
    item = cm[cm.index("109. **"): cm.index("110. **")]; item110 = cm[cm.index("110. **"): cm.index("Next free: row")]
    k = ve.index("Provider-side serving test, fourth run"); row = ve[k: ve.index("\n", k)]
    assert cm.index("109. **") < cm.index("110. **") < cm.index("Next free: row") and "\n110. **" not in item110 and "PROVIDER_SIDE_RETRY_TEST.md" in row and "test 129" in row and "test 129** (10 tests" in item
    reqs = retried(res); outs = [outcome(r) for r in reqs]; cnt = {o: outs.count(o) for o in OUTCOMES}
    after = [r for r in reqs if r["behind"]]; own = [r for r in after if r["first_ok"]]; flags = scope_flags(res); slips = first_slips(res)
    rep = sum(S[x]["replies"] for x in S); unp = sum(S[x]["unparsed"] for x in S); corrupted = sum(S[x]["corrupted"] for x in S); total = sum(S[x]["n"] for x in S)
    assert sum(READ) == 8 and rep == 6989 and unp == 0 and total == 6960 and corrupted == 0 and len(reqs) == 29 and cnt == {"same": 21, "refused again": 3, "admitted, wrong": 3, "put right": 2}
    assert all(r["second"] == "DEL[1,0]" and r["job"] == 4 and r["step"] == 18 for r in reqs if outcome(r) == "admitted, wrong")
    assert (len(after), len(own)) == (17, 14) and flags == [4, 20, 122] and len(slips) == 8 and S["H1"]["ex"] == S["H0"]["ex"] == 1884 and S["H1"]["n"] == S["H0"]["n"] == 1920
    eph1, eph0 = round(S["H1"]["eph"]), round(S["H0"]["eph"]); chg = sg((S["H1"]["eph"] / S["H0"]["eph"] - 1) * 100)
    s64, kc8 = t3[64]["eph"], st(next(x for x in third()["cells"] if x["arm"] == "KC" and x["K"] == 8))["eph"]
    assert (num(round(s64)), num(round(kc8)), f"{s64 / kc8:.1f}") == ("30,914", "5,694", "5.4")
    shares = ", ".join(f"{S[x]['share'] * 100:.1f}" for x in ("R8", "R32")) + f" and {S['R64']['share'] * 100:.1f}"; shares3 = f"{t3[8]['share'] * 100:.1f}, {t3[32]['share'] * 100:.1f} and {t3[64]['share'] * 100:.1f}"
    rounds = sum(S[x]["retry_rounds"] for x in WITH_RETRY); rsec = sum(S[x]["retry_s"] for x in WITH_RETRY)
    for need in (f"{sum(READ)} of the 12 lines read yes; lines 6, 8, 9 and 11 read NO; none is \"not run\"", f"{num(rep - unp)} of {num(rep)} replies accepted by the strict parser",
                 f"the exact share was {pc(S['H1']['share'])} with the retry and {pc(S['H0']['share'])} without it ({num(S['H1']['ex'])} of {num(S['H1']['n'])} each)",
                 f"the exact steps per GPU hour {num(eph1)} against {num(eph0)} ({chg})", f"({S['H1']['wall']:.1f} s against {S['H0']['wall']:.1f} s)", f"the retry's second rounds were {pc(S['H1']['retry_share'])} of the cell's seconds",
                 "**This Result does not claim that the retry raises the exactness.**", f"{shares} % exact at 8, 32 and 64 jobs against {shares3} % in the third run",
                 f"Of the {len(reqs)} second requests ({rounds} second rounds, {rsec:.1f} seconds in all), {cnt['same']} came back with the same reply, {cnt['refused again']} with another reply that the gate refused again, {cnt['admitted, wrong']} with `DEL[1,0]` (admitted by the gate and wrong; job 4, step 18) and {cnt['put right']} were put right",
                 f"{len(after)} were asked after the job had already slipped, and in {len(own)} of those the first reply was, letter for letter, the oracle's own reply", f"flags {len(flags)} of the {len(slips)} (jobs 4, 20 and 122)",
                 f"{corrupted} of {num(total)} steps were corrupted", f"the 5.4 is a 64-job cell over an 8-job cell, {num(round(s64))} over {num(round(kc8))}", "no route in this design promises 100 %",
                 "(the four NO stood at 85, 35, 55 and 40 % on the page, which did not name this exact pattern)", "Status: RUN (the owner's Colab run, five cells in one session; the file reached the repository work on 20 Oct 2026); the Result is on the page.**"):
        assert need in item, need[:90]
    for need in (f"**{sum(READ)} of 12 lines yes; lines 6, 8, 9 and 11 NO; none not run**", f"{num(rep - unp)} of {num(rep)} replies accepted by the strict parser",
                 f"the exact share is {pc(S['H1']['share'])} with the retry and {pc(S['H0']['share'])} without it ({num(S['H1']['ex'])} of {num(S['H1']['n'])} each)", f"the exact steps per GPU hour {num(eph1)} against {num(eph0)} ({chg})",
                 "**the Result does not claim that the retry raises the exactness.**", f"{S['R64']['share'] * 100:.1f} % exact at 64 jobs against {t3[64]['share'] * 100:.1f} % in the third run",
                 f"Of {len(reqs)} second requests {cnt['same']} came back with the same reply, {cnt['refused again']} with another reply refused again, {cnt['admitted, wrong']} with `DEL[1,0]` (admitted and wrong) and {cnt['put right']} were put right; {len(own)} of the {len(after)} asked after a slip were on the oracle's own reply",
                 f"the scope check flags {len(flags)} of the {len(slips)}; {corrupted} of {num(total)} steps corrupted", "**No route in this design promises 100 %.**"):
        assert need in row, need[:90]
    # item 110: an idea, not run; what it says about the saved files is read from them
    assert "**Status: IDEA, NOT RUN; no page, no code and no test exist for it.**" in item110 and "NOT RUN" in item110 and "nothing is trained" in item110 and "so it cannot give 100 %" in item110
    s = P.Session("S", SEED * 1000 + 4, STEPS)
    while s.t < 17: rq = s.request(); s.accept(P.Reply(rq.stand_in_reply, 0, 0, 0))
    prompt = [m for m in s.request().messages if m["role"] == "user"][-1]["content"]; j4 = [x for x in slips if x[0] == 4]
    assert "NODES: [1,0:6] [1,2:2]" in prompt and request_sentence(4, 18) == "Add a service at (0,0) with value 3." and j4 and j4[0][2:4] == (18, "ADD[1,0:6]")
    for need in ("at job 4, step 18 the prompt listed `NODES: [1,0:6] [1,2:2]` and asked \"Add a service at (0,0) with value 3\", and the model wrote `ADD[1,0:6]`, a line copied from the state list instead of the request",
                 f"(jobs 4, 20 and 122); it does not obviously help the three shape slips (jobs 3, 19 and 101) and does nothing for the two extra deletes that the gate admits (jobs 17 and 62)",
                 f"that the model repeated the wrong coordinate 21 times ({len(own)} of the {len(after)} second requests asked after a slip were on the oracle's own reply)"):
        assert need in item110, need[:90]
    s24 = [r for r in reqs if r["job"] == 4 and r["step"] == 24]; assert s24 and all(r["first"] == "ADD[(1,0)->(4,5):dep#7]" and r["first_ok"] for r in s24)
    assert "job 4, step 24 asked for exactly that edge and the model was right; the real slip is step 18" in item110 and sorted(KINDS) == [3, 4, 17, 19, 20, 62, 101, 122]
    assert "sheaf attention and Lie-group spinors are in none of the 11 texts" in item110 and "4 cells of 64 jobs" not in item110 and "four cells of 64 jobs with the retry off" in item110


if __name__ == "__main__":
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]; bad = 0
    for n, f in fns:
        try: f(); print("ok  ", n)
        except Exception: bad += 1; print("FAIL", n); traceback.print_exc()
    print(f"{len(fns) - bad}/{len(fns)} passed"); sys.exit(1 if bad else 0)
