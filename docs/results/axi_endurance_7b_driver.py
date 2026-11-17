"""Test 43 driver (docs/ENDURANCE_7B_TEST.md): four jobs of 20,000 steps, the exact state held by the host, Qwen2.5-7B-Instruct-AWQ on one L4.
Run on the Colab machine as:  nohup python -u t43_driver.py > /content/out/t43.log 2>&1 &
Everything else is the harness of Tests 35, 38 and 42 (provider_side_big.py, repo commit 55631ec), unchanged."""
import sys, json, time, os, hashlib

SEED = 20261017
MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
STEPS = 20000
SHAPE = "named"
JOBS = [192, 232, 241, 290]
W = 1000


def f(x, d=2):
    return None if x is None else round(x, d)


def slips(log):
    """A slip is a step on which the held state stops being equal to the oracle's after being equal. Returns (count, [(t, steps_it_lasted)], open_at_end)."""
    out, prev, start = [], True, None
    for x in log:
        if prev and not x["exact"]: start = x["t"]
        if (not prev) and x["exact"] and start is not None: out.append((start, x["t"] - start)); start = None
        prev = x["exact"]
    return out, (not prev)


def window_stats(sess, per_round, w=W):
    """Per window of w steps: the exact share per job, mean prompt and computed prompt tokens per step, cached share, mean seconds per round divided by the jobs in it."""
    rows = []
    n = max((len(s.log) for s in sess), default=0)
    for lo in range(1, n + 1, w):
        hi = lo + w
        per_job = []; xs = []
        for s in sess:
            ys = [x for x in s.log if lo <= x["t"] < hi]
            xs += ys
            per_job.append(f(sum(1 for y in ys if y["exact"]) / len(ys), 4) if ys else None)
        if not xs: continue
        p = sum(x["n_prompt"] for x in xs); c = sum((x["n_cached"] or 0) for x in xs)
        rr = per_round[lo - 1:hi - 1]
        rows.append(dict(lo=lo, exact=per_job, prompt=f(p / len(xs), 1), computed=f((p - c) / len(xs), 1), cached=f(c / p, 4) if p else None, sec=f(sum(rr) / len(rr), 4) if rr else None))
    return rows


def summary(sess, per_round, wall):
    logs = {s.seed % 1000: s.log for s in sess}
    out = dict(wall_seconds=f(wall, 1), steps={k: len(v) for k, v in logs.items()})
    allx = [x for v in logs.values() for x in v]
    out["exact_total"] = sum(1 for x in allx if x["exact"]); out["steps_total"] = len(allx)
    out["parsed"] = sum(1 for x in allx if x["parsed"]); out["corrupted"] = sum(1 for x in allx if x["corrupted"])
    out["per_job"] = {}
    for k, v in logs.items():
        sl, still = slips(v)
        a = [x for x in v if x["t"] <= 5000]; b = [x for x in v if x["t"] > len(v) - 5000]
        out["per_job"][k] = dict(steps=len(v), exact=sum(1 for x in v if x["exact"]), slips=len(sl), longest_slip=max((d for _, d in sl), default=0), open_at_end=still,
                                  first_slips=sl[:12], exact_first5000=f(sum(1 for x in a if x["exact"]) / len(a), 4) if a else None, exact_last5000=f(sum(1 for x in b if x["exact"]) / len(b), 4) if b else None,
                                  stopped=next((s.stopped for s in sess if s.seed % 1000 == k), None))
    out["windows"] = window_stats(sess, per_round)
    return out


def main():
    sys.path.insert(0, "/content/axi/python")
    import axi.experiments.provider_side_typed as T
    import axi.experiments.provider_side_long as L
    import axi.experiments.provider_side_big as B
    os.makedirs("/content/out", exist_ok=True)
    t0 = time.time()
    eng = B.BigEngine(model=MODEL)
    eng.warm_up()
    print("LOADED", MODEL, round(time.time() - t0), "s cache", eng.info().get("capacity_tokens"), flush=True)
    sess = [T.TypedSession(SEED * 1000 + e, STEPS, "old", SHAPE) for e in JOBS]
    print("SCRIPTS", [len(s.script) for s in sess], flush=True)
    meter = B.PowerMeter(); meter.start()
    last = [time.time()]; t_start = time.time()

    def progress(r):
        if r % 1000 == 0:
            done = sum(1 for s in sess for x in s.log if x["exact"]); tot = sum(len(s.log) for s in sess)
            print("PROG round", r, "elapsed", round(time.time() - t_start), "s last1000", round(time.time() - last[0]), "s exact", done, "of", tot, flush=True); last[0] = time.time()
        if r % 4000 == 0:
            try:
                json.dump(dict(round=r, partial=True), open("/content/out/t43_partial.json", "w"))
            except Exception: pass

    r = L.run_sessions(eng, sess, progress)
    power = meter.stop(r["wall_seconds"])
    c = dict(label="E4", arm="S", jobs=JOBS, steps=STEPS, seed=SEED, shape=SHAPE, engine=eng.info(), power=power, **r)
    path = "/content/out/t43_E4.json"
    json.dump(c, open(path, "w"), default=str)
    raw = open(path, "rb").read()
    per_round = [x["seconds"] / max(1, x["active"]) for x in r["rounds"]]
    s = summary(sess, per_round, r["wall_seconds"]); s["file"] = os.path.basename(path); s["bytes"] = len(raw); s["sha256"] = hashlib.sha256(raw).hexdigest(); s["power"] = power
    s["distinct_regexes"] = r.get("distinct_regexes")
    json.dump(s, open("/content/out/t43_summary.json", "w"), default=str)
    print("SUMMARY", json.dumps({k: v for k, v in s.items() if k != "windows"}, default=str), flush=True)
    for w in s["windows"]: print("WIN", json.dumps(w), flush=True)
    print("DONE t43", flush=True)


if __name__ == "__main__":
    main()
