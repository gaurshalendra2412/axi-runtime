"""Run every test_*.py in this folder as a script. Usage (from repo root):  PYTHONPATH=python python3 python/tests/run_all.py"""
import glob, os, subprocess, sys
here = os.path.dirname(os.path.abspath(__file__)); env = dict(os.environ, PYTHONPATH=os.path.join(here, ".."))
skip = {"test_block_sparse_attention.py", "test_block_sparse_attention_backward.py", "test_cico_attention.py", "test_end_to_end_pipeline.py"}  # need torch/triton/GPU
bad = []
for f in sorted(glob.glob(os.path.join(here, "test_*.py"))):
    name = os.path.basename(f)
    if name in skip: print("SKIP (needs torch/GPU):", name); continue
    r = subprocess.run([sys.executable, f], env=env, capture_output=True, text=True)
    print(("OK   " if r.returncode == 0 else "FAIL ") + name, f"({r.stdout.count('PASS')} tests)")
    if r.returncode: bad.append(name); print(r.stdout[-800:], r.stderr[-800:])
sys.exit(1 if bad else 0)
