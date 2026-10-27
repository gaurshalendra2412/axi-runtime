"""The eighth forward check from a terminal (Codespaces): the same replay as colab/axi_foresee_check_8.ipynb. No model, no key, no cost.
Run from anywhere inside the repository, after `pip install arc-agi arcengine`:   python python/colab/run_foresee_check_8.py
Writes docs/results/foresee_check_8_all.json and docs/results/foresee_check_8_printout.txt."""
import json, os, subprocess, sys
root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
sys.path.insert(0, os.path.join(root, "python"))
import arc_agi
from axi.experiments import gemini_arc as G
from axi.experiments import ledger_replay as LR
from axi.experiments import foresee8 as F
ARC = arc_agi.Arcade()
def make_env(game): return ARC.make(game, seed=G.GAME_SEED)
RES = os.path.join(root, "docs", "results") + os.sep
FILES = [("axi_gemini_arc_default.json", "K"), ("axi_gemini_arc_default.json", "A"), ("axi_gemini_arc_b_default.json", "B"), ("axi_gemini_arc_c_default.json", "C")]
sessions, ok = {}, {}
for f, arm in FILES:
    print("=" * 60, "ARM", arm, flush=True)
    try:
        rows, games, moves = LR.load(RES + f, arm); sessions[arm] = F.feed8(rows, make_env, arm, games, moves); ok[arm] = True
    except Exception as e:
        print(type(e).__name__, ":", e); ok[arm] = False
allok = bool(ok) and all(ok.values())
pooled = per_arm_prior = clears = None
if allok:
    pooled_arms = F.from_sessions(sessions, True); prior_arms = F.from_sessions(sessions, False)
    pooled, per_arm_prior, clears = F.pool(pooled_arms), F.pool(prior_arms), F.clear_study(sessions)
checks = F.verdict(pooled, per_arm_prior, clears, allok if ok else None)
out = F.text(pooled, per_arm_prior, clears) + "\n\n" + "\n".join(f"{i} {t} | {({True: 'yes', False: 'NO', None: 'not read'})[v]} | {x}" for i, t, x, v in checks) + "\nheadline: " + F.headline(checks)
print(out)
json.dump(dict(pooled=pooled, per_arm_prior=per_arm_prior, clears=clears, replays=ok, checks=checks), open(os.path.join(root, "docs", "results", "foresee_check_8_all.json"), "w"), indent=1, default=str)
open(os.path.join(root, "docs", "results", "foresee_check_8_printout.txt"), "w").write(out + "\n")
print("saved docs/results/foresee_check_8_all.json and foresee_check_8_printout.txt")
