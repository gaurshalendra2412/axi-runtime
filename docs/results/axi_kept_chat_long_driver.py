"""Test 42 driver (docs/KEPT_CHAT_LONG_TEST.md): the kept chat with prefix caching on against the exact-state arm, jobs of 1,500 steps.
Run on the Colab machine as:  python -u kept_chat_long_driver.py 1   (stage 1, 32,768-token window)   or   ... 2   (stage 2, YaRN 131,072).
Sets, at run time and without editing any repository file: max_model_len, the kept-chat stop limit (provider_side.MAX_CONTEXT), the cache size (and, for stage 2, a static YaRN override).
Everything else is the harness of Tests 35 and 38 (provider_side_big.py, commit 55631ec)."""
import sys, json, time, os, hashlib

SEED = 20261017
MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
STEPS = 1500
SHAPE = "named"
CELLS = {  # label: (arm, jobs)
    "1": [("A1", "S", [192]), ("B1", "KC", [192]), ("A4", "S", [194, 196, 197, 198]), ("B2", "KC", [194]), ("B3", "KC", [196]), ("B4", "KC", [197])],
    "2": [("C1", "S", [192]), ("D1", "KC", [192])],
}


def f(x, d=2):
    return None if x is None else round(x, d)


def summarize(c):
    logs = [x for s in c["sessions"] for x in s["log"]]
    n = len(logs)
    out = dict(label=c["label"], arm=c["arm"], jobs=c["jobs"], steps=n, wall_seconds=f(c["wall_seconds"], 1))
    out["exact"] = sum(1 for x in logs if x["exact"]); out["parsed"] = sum(1 for x in logs if x["parsed"]); out["corrupted"] = sum(1 for x in logs if x["corrupted"])
    out["stops"] = [(s["seed"] % 1000, len(s["log"]), s["stopped"], s["script_steps"]) for s in c["sessions"]]
    tot_p = sum(x["n_prompt"] for x in logs); tot_c = sum((x["n_cached"] or 0) for x in logs)
    out["computed_per_step"] = f((tot_p - tot_c) / n, 1) if n else None
    out["cached_share"] = f(tot_c / tot_p, 4) if tot_p else None
    out["new_tokens_per_step"] = f(sum(x["n_new"] for x in logs) / n, 1) if n else None
    rs = c["rounds"]; per = [r["seconds"] / max(1, r["active"]) for r in rs]
    k = min(30, len(per))
    out["sec_per_step_first30"] = f(sum(per[:k]) / k, 4) if k else None
    out["sec_per_step_last30"] = f(sum(per[-k:]) / k, 4) if k else None
    out["sec_per_step_all"] = f(c["wall_seconds"] / n, 4) if n else None
    out["steps_per_hour"] = f(3600.0 * out["exact"] / c["wall_seconds"], 0) if c["wall_seconds"] else None
    out["max_prompt"] = max((x["n_prompt"] for x in logs), default=None)
    out["power"] = c.get("power")
    win = []
    for lo in range(1, STEPS + 1, 100):
        xs = [x for x in logs if lo <= x["t"] < lo + 100]
        if not xs: continue
        rr = [per[i] for i in range(lo - 1, min(lo + 99, len(per)))] if len(c["jobs"]) == 1 else []
        win.append(dict(lo=lo, n=len(xs), exact=f(sum(1 for x in xs if x["exact"]) / len(xs), 4), prompt=f(sum(x["n_prompt"] for x in xs) / len(xs), 1),
                        computed=f(sum(x["n_prompt"] - (x["n_cached"] or 0) for x in xs) / len(xs), 1), sec=f(sum(rr) / len(rr), 4) if rr else None))
    out["windows"] = win
    return out


def show(s):
    print("CELL", s["label"], s["arm"], "jobs", s["jobs"], "steps", s["steps"], "exact", s["exact"], "parsed", s["parsed"], "corrupt", s["corrupted"], "wall", s["wall_seconds"],
          "s/step all", s["sec_per_step_all"], "first30", s["sec_per_step_first30"], "last30", s["sec_per_step_last30"], "computed/step", s["computed_per_step"],
          "cached", s["cached_share"], "maxprompt", s["max_prompt"], "steps/h", s["steps_per_hour"], "stops", s["stops"], "power", s["power"], flush=True)
    for w in s["windows"]:
        print("  WIN", s["label"], w["lo"], "n", w["n"], "exact", w["exact"], "prompt", w["prompt"], "computed", w["computed"], "sec", w["sec"], flush=True)


def run_all(eng, stage, steps=STEPS, out="/content/out", cells=None, power=True):
    import axi.experiments.provider_side_typed as T
    import axi.experiments.provider_side_long as L
    import axi.experiments.provider_side_big as B
    os.makedirs(out, exist_ok=True)
    summ = {}
    for label, arm, jobs in (cells or CELLS[stage]):
        if arm == "S": sess = [T.TypedSession(SEED * 1000 + e, steps, "old", SHAPE) for e in jobs]
        else: sess = [L.KeptChat(SEED * 1000 + e, steps, SHAPE) for e in jobs]
        meter = B.PowerMeter() if power else None
        if hasattr(eng, "request_metrics"): eng.request_metrics = []
        if meter: meter.start()
        r = L.run_sessions(eng, sess, None)
        pw = meter.stop(r["wall_seconds"]) if meter else None
        c = dict(label=label, arm=arm, jobs=jobs, steps=steps, seed=SEED, shape=SHAPE, engine=eng.info(), power=pw, request_times=list(getattr(eng, "request_metrics", [])), **r)
        path = os.path.join(out, "t42_%s_%s.json" % (stage, label))
        json.dump(c, open(path, "w"), default=str)
        raw = open(path, "rb").read()
        s = summarize(c); s["file"] = os.path.basename(path); s["bytes"] = len(raw); s["sha256"] = hashlib.sha256(raw).hexdigest()
        summ[label] = s; show(s)
    json.dump(summ, open(os.path.join(out, "t42_summary_%s.json" % stage), "w"), default=str)
    print("DONE t42 stage", stage, flush=True)
    return summ


def main():
    stage = sys.argv[1]
    sys.path.insert(0, "/content/axi/python")
    import axi.experiments.provider_side as PS
    import axi.experiments.provider_side_big as B
    if stage == "2":
        import vllm
        _o = vllm.LLM
        def _L(*a, **k):
            k["hf_overrides"] = {"rope_scaling": {"rope_type": "yarn", "type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}, "max_position_embeddings": 131072}
            return _o(*a, **k)
        vllm.LLM = _L
        PS.MAX_CONTEXT, cap, mml = 130000, 9000, 131072
    else:
        PS.MAX_CONTEXT, cap, mml = 32000, 3000, 32768
    t0 = time.time()
    eng = B.BigEngine(model=MODEL, capacity_blocks=cap, max_model_len=mml)
    eng.warm_up()
    print("LOADED", MODEL, "stage", stage, round(time.time() - t0), "s cache", eng.info().get("capacity_tokens"), "max_model_len", mml, "MAX_CONTEXT", PS.MAX_CONTEXT, flush=True)
    run_all(eng, stage)


if __name__ == "__main__":
    main()
