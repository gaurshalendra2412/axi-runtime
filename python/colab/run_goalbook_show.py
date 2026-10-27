"""Print what the three readers of axi/experiments/goalbook.py say at the end of every level of the saved ARC-AGI-3 runs (arms K, A, B and C). No model, no key, no cost; it plays nothing new and checks nothing: a printout, not a verdict.
Run from anywhere inside the repository, after `pip install arc-agi arcengine`:   python python/colab/run_goalbook_show.py
Writes docs/results/goalbook_show_printout.txt."""
import os, subprocess, sys
root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
sys.path.insert(0, os.path.join(root, "python"))
import arc_agi
from axi.experiments import gemini_arc as G
from axi.experiments import ledger_replay as LR
from axi.experiments import foresee8 as F
from axi.experiments import goalbook as GB
ARC = arc_agi.Arcade()
def make_env(game): return ARC.make(game, seed=G.GAME_SEED)
RES = os.path.join(root, "docs", "results") + os.sep
FILES = [("axi_gemini_arc_default.json", "K"), ("axi_gemini_arc_default.json", "A"), ("axi_gemini_arc_b_default.json", "B"), ("axi_gemini_arc_c_default.json", "C")]
sessions = {}
for f, arm in FILES:
    print("=" * 60, "ARM", arm, flush=True)
    rows, games, moves = LR.load(RES + f, arm); sessions[arm] = F.feed8(rows, make_env, arm, games, moves)
out = GB.show(sessions)
print(out)
open(os.path.join(root, "docs", "results", "goalbook_show_printout.txt"), "w").write(out + "\n")
print("saved docs/results/goalbook_show_printout.txt")
