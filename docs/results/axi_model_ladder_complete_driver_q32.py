import sys, json, time, os
sys.path.insert(0, "/content/axi/python")
OFFLOAD_GB = float(os.environ.get("CPU_OFFLOAD_GB", "10"))
import vllm
_orig_init = vllm.LLM.__init__
def _init(self, *a, **k):
    k.setdefault("cpu_offload_gb", OFFLOAD_GB)
    return _orig_init(self, *a, **k)
vllm.LLM.__init__ = _init
model = sys.argv[1]
tag = model.replace("/", "_")
os.makedirs("/content/out", exist_ok=True)
from axi.experiments import provider_side_big as B
def f(x, d=2, k=1):
    return None if x is None else round(x * k, d)
print("SETTING cpu_offload_gb", OFFLOAD_GB, "(added by this driver; harness file unchanged)", flush=True)
t0 = time.time()
eng = B.BigEngine(model=model)
eng.warm_up()
print("LOADED", model, round(time.time() - t0), "s cache", eng.info().get("capacity_tokens"), "cpu_offload_gb", OFFLOAD_GB, flush=True)
summ = {}
CELLS = [c for c in os.environ.get("CELLS", "P4,PC,PN,P0,X0,XW,XR").split(",") if c]
print("CELLS", CELLS, flush=True)
for lab in CELLS:
    c = B.run_cell(eng, lab)
    s = B.summarize_cell(c)
    B.save_cell(c, "/content/out/" + tag + "_" + lab + ".json")
    summ[lab] = s
    print("CELL", lab, "steps", s.get("steps_done"), "exact%", f(s.get("exact_share"), 2, 100), "corrupt", s.get("corrupted_steps"), "parsed%", f(s.get("parsed_share"), 2, 100),
          "s/step", f(s.get("seconds_per_step"), 3), "steps/h", f(s.get("steps_per_hour"), 0), "computed/step", f(s.get("computed_per_step"), 1), "cached%", f(s.get("cached_share"), 2, 100),
          "growth", f(s.get("time_growth"), 2), "stopped", s.get("stopped_jobs", s.get("stopped")), flush=True)
    json.dump(summ, open("/content/out/" + tag + "_summary_" + "_".join(CELLS) + ".json", "w"), default=str)
print("DONE", model, flush=True)
