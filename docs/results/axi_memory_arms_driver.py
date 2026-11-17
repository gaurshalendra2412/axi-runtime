"""Test 44 driver (docs/MEMORY_ARMS_TEST.md): the arms A K S V W R on the same three jobs (192, 194, 196), 600 steps, Qwen2.5-14B-Instruct-AWQ on one L4.
Run on the Colab machine as:  nohup python -u t44_driver.py > /content/out/t44.log 2>&1 &      (optionally with arm letters: ... t44_driver.py A K)
Needs /content/axi/python/axi/experiments/provider_side_memory.py (a new file, written beside the repository's modules; no repository file is edited)."""
import sys, json, time, os

sys.path.insert(0, "/content/axi/python")
ORDER = ["A", "K", "S", "V", "W", "R"]
OUT = "/content/out"


def main():
    import axi.experiments.provider_side_memory as M
    import axi.experiments.provider_side_big as B
    arms = [a for a in (sys.argv[1:] or ORDER)]
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    eng = B.BigEngine(model=M.MODEL, capacity_blocks=2000, max_model_len=20000)
    eng.warm_up()
    print("LOADED", M.MODEL, round(time.time() - t0), "s cache", eng.info().get("capacity_tokens"), "max_model_len 20000", flush=True)
    for arm in arms:
        a0 = time.time()

        def progress(r, arm=arm, a0=a0):
            if r % 100 == 0: print("PROG", arm, "round", r, "elapsed", round(time.time() - a0), "s", flush=True)

        cell = M.run_arm(eng, arm, progress=progress)
        info = M.save_arm(cell, "%s/t44_%s.json" % (OUT, arm))
        s = M.summarize_arm(cell)
        s.update(info)
        s["summaries"] = [dict(job=x["seed"] % 1000, t=m["t"], n_new=m["n_new"], truncated=m["truncated"], state_fidelity=m["state_fidelity"], text=m["text"]) for x in cell["sessions"] for m in x.get("summaries", [])]
        json.dump(s, open("%s/t44_summary_%s.json" % (OUT, arm), "w"), default=str)
        print("CELL", arm, "steps", s["steps"], "step_ok", s["step_ok"], "exact_blind", s["exact_blind"], "exact_policy", s["exact_policy"], "parsed", s["parsed"], "corrupt_blind", s["corrupted_blind"],
              "ok_delete", s["step_ok_delete"], "ok_other", s["step_ok_add"], "first_miss", s["first_miss"], "first_blind_off", s["first_blind_off"], "stops", s["stops"], flush=True)
        print("COST", arm, "prompt/step", s["prompt_per_step"], "computed/step", s["computed_per_step"], "computed_steps_only", s["computed_per_step_steps_only"], "cached", s["cached_share"],
              "new/step", s["new_tokens_per_step"], "sec/step", s["sec_per_step"], "wall", s["wall_seconds"], "compaction_sec", s["compaction_seconds"], "max_prompt", s["max_prompt"], "power", s["power"], flush=True)
        print("SUMS", arm, "n", s["compactions"], "mean_tokens", s["summary_tokens_mean"], "truncated", s["summaries_truncated"], "fidelity", s["summary_state_fidelity"], "file", s["file"], s["bytes"], s["sha256"], flush=True)
        for w in s["windows"]: print("  WIN", arm, w["lo"], "n", w["n"], "ok", w["step_ok"], "blind", w["exact_blind"], "policy", w["exact_policy"], "prompt", w["prompt"], flush=True)
    print("DONE t44", flush=True)


if __name__ == "__main__":
    main()
