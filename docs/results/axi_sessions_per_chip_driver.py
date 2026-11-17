import sys, json, time, os
sys.path.insert(0, "/content/axi/python")
os.makedirs("/content/out", exist_ok=True)
from axi.experiments import provider_side_big as B
def f(x, d=2, k=1):
    return None if x is None else round(x * k, d)
t0 = time.time()
eng = B.BigEngine(model="Qwen/Qwen2.5-7B-Instruct-AWQ", capacity_blocks=10000)
eng.warm_up()
print("LOADED", round(time.time() - t0), "s cache", eng.info().get("capacity_tokens"), flush=True)
summ = {}
for K in (8, 32, 128, 256):
    lab = "S%d" % K
    B.CELLS[lab] = (K, 192, "S", "coords", "named")
    c = B.run_cell(eng, lab)
    s = B.summarize_cell(c)
    B.save_cell(c, "/content/out/sessions_" + lab + ".json")
    summ[lab] = s
    print("CELL", lab, "steps", s.get("steps_done"), "exact%", f(s.get("exact_share"), 2, 100), "corrupt", s.get("corrupted_steps"), "parsed%", f(s.get("parsed_share"), 2, 100),
          "wall_s", f(s.get("wall_seconds"), 0), "steps/h", f(s.get("steps_per_hour"), 0), "round p50/p95/max", f(s.get("round_p50"), 2), f(s.get("round_p95"), 2), f(s.get("round_max"), 2),
          "computed/step", f(s.get("computed_per_step"), 1), "cached%", f(s.get("cached_share"), 2, 100), "W", f(s.get("mean_w"), 0), "J/exact", f(s.get("joules_per_exact_step"), 3), flush=True)
json.dump(summ, open("/content/out/sessions_summary.json", "w"), default=str)
print("DONE sessions", flush=True)
