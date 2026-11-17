"""Test 45 driver (docs/STATE_KINDS_TEST.md): four worlds (T S C J) x four arms (B P G D), 150 steps, jobs 1 2 3, on one model.
Run on the Colab machine as:  nohup python -u t45_driver.py Qwen/Qwen2.5-7B-Instruct-AWQ q7 > /content/out/t45_q7.log 2>&1 &
Needs /content/axi/python/axi/experiments/state_kinds.py (a new file written beside the repository's modules; no repository file is edited)."""
import sys, json, time, os, hashlib

sys.path.insert(0, "/content/axi/python")
OUT = "/content/out"
STEPS = 150


def main():
    import axi.experiments.state_kinds as K
    import axi.experiments.provider_side_big as B
    model, tag = sys.argv[1], sys.argv[2]
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    eng = B.BigEngine(model=model)
    eng.warm_up()
    print("LOADED", model, round(time.time() - t0), "s cache", eng.info().get("capacity_tokens"), "regex_sha256", K.regex_sha256(), flush=True)
    allsum = {}
    for world in K.WORLDS:
        for arm in K.ARMS:
            c0 = time.time()
            cell = K.run_world(eng, world, arm, STEPS)
            path = "%s/t45_%s_%s_%s.json" % (OUT, tag, world, arm)
            json.dump(cell, open(path, "w"), default=str)
            raw = open(path, "rb").read()
            s = K.summarize(cell); s.update(file=os.path.basename(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            allsum["%s%s" % (world, arm)] = s
            print("CELL", world, arm, "steps", s["steps"], "parsed", s["parsed"], "exact", s["exact"], "step_ok", s["step_ok"], "ok_valid", s["step_ok_valid_requests"], "ok_conf", s["step_ok_conflicts"], "ok_gen", s["step_ok_generic_conflicts"],
                  "ok_dom", s["step_ok_domain_conflicts"], "held_valid", s["held_valid"], "refused", s["refused"], "literal_on_conf", s["literal_on_conflicts"], "n_conf", s["n_conflicts"], "wall", s["wall_seconds"],
                  "prompt/step", s["prompt_per_step"], "computed/step", s["computed_per_step"], "cached", s["cached_share"], "first_off", s["first_off"], "sha", s["sha256"][:16], flush=True)
            print("  KINDS", world, arm, json.dumps(s["step_ok_by_kind"]), flush=True)
            json.dump(allsum, open("%s/t45_summary_%s.json" % (OUT, tag), "w"), default=str)
    print("DONE t45", tag, round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
