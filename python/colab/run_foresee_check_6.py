"""The sixth forward check from a terminal (Codespaces): the same replay as colab/axi_foresee_check_6.ipynb. No model, no key, no cost.
Run from anywhere inside the repository, after `pip install arc-agi arcengine`:   python python/colab/run_foresee_check_6.py
Writes docs/results/foresee_check_6_all.json and docs/results/foresee_check_6_printout.txt."""
import json, os, subprocess, sys
root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
sys.path.insert(0, os.path.join(root, "python"))
import arc_agi
from axi.experiments import gemini_arc as G
from axi.experiments import ledger_replay as LR
from axi.experiments import foresee6 as F
ARC = arc_agi.Arcade()
def make_env(game): return ARC.make(game, seed=G.GAME_SEED)
RES = os.path.join(root, "docs", "results") + os.sep
FILES = [("axi_gemini_arc_default.json", "K"), ("axi_gemini_arc_default.json", "A"), ("axi_gemini_arc_b_default.json", "B"), ("axi_gemini_arc_c_default.json", "C")]
per_arm, per_game, ok = {}, {v: {} for v in F.VARIANTS}, {}
for f, arm in FILES:
    print("=" * 60, "ARM", arm, flush=True)
    try:
        s = LR.run_file(RES + f, arm, make_env); vs = F.from_session(s); per_arm[arm] = vs; ok[arm] = True
        for v, pg in vs.items():
            for g, c in pg.items(): per_game[v][g] = F.combine([per_game[v].get(g, {}), c])
    except Exception as e:
        print(type(e).__name__, ":", e); ok[arm] = False; per_arm[arm] = {"error": f"{type(e).__name__}: {e}"}
allok = bool(ok) and all(ok.values())
checks = F.verdict(per_game if allok else None, allok if ok else None)
out = F.text(per_game) + "\n\n" + "\n".join(f"{i} {t} | {({True: 'yes', False: 'NO', None: 'not read'})[v]} | {x}" for i, t, x, v in checks) + "\nheadline: " + F.headline(checks)
print(out)
json.dump(dict(per_arm=per_arm, per_game=per_game, replays=ok, checks=checks), open(os.path.join(root, "docs", "results", "foresee_check_6_all.json"), "w"), indent=1, default=str)
open(os.path.join(root, "docs", "results", "foresee_check_6_printout.txt"), "w").write(out + "\n")
print("saved docs/results/foresee_check_6_all.json and foresee_check_6_printout.txt")
