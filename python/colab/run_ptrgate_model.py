"""Test 28 (docs/PTRGATE_MODEL_TEST.md): Gemini 3.8 Flash plain against Gemini 3.8 Flash given the pointer gate's findings, on 300 Juliet functions (600 calls, a dollar or two).
From the `python` folder, after `pip install google-genai`:
  1. python colab/run_ptrgate_model.py run --price-in P --price-out P --price-cached P --budget 3     (resumable: run the same command again after a stop)
  2. python colab/run_ptrgate_model.py read                                                          (no key, no cost; writes docs/results/ptrgate_model.json)
The key is the environment variable GOOGLE_API_KEY (or you are asked once). Prices are dollars per million tokens, typed by you (check them in Google AI Studio); there is no built-in price."""
import argparse, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))
from axi.experiments import ptrgate_model as M
from axi.experiments.api_llm import QuotaStop, ServiceBusy, BudgetStop

OUT = os.path.expanduser("~/axi_ptr_out")
JULIET = os.path.expanduser("~/juliet-test-suite-c")
REPO = "https://github.com/arichardson/juliet-test-suite-c.git"


def clean_key(raw):
    return "".join(c for c in (raw or "") if 33 <= ord(c) <= 126 and c not in "\"'`")


def get_key():
    f = os.path.join(OUT, ".gemini_key"); key = clean_key(open(f).read()) if os.path.exists(f) else ""
    key = key or clean_key(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
    if not key:
        key = clean_key(input("Paste your Gemini key and press Enter: "))
        fd = os.open(f, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as h: h.write(key + "\n")
    return key


def items():
    if not os.path.isdir(os.path.join(JULIET, "testcases")):
        subprocess.check_call(["git", "clone", "--depth", "1", REPO, JULIET])
    return M.build_items(os.path.join(JULIET, "testcases"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["run", "read"])
    ap.add_argument("--price-in", type=float); ap.add_argument("--price-out", type=float); ap.add_argument("--price-cached", type=float, default=0.0)
    ap.add_argument("--budget", type=float)
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True); path = os.path.join(OUT, "rows.json"); its = items()
    if a.cmd == "run":
        assert a.price_in and a.price_out and a.budget, "Type --price-in, --price-out and --budget."
        from axi.experiments import gemini_long as GL
        llm, model = GL.build_chat_llm(get_key(), os.path.join(OUT, "cache.jsonl"), budget_usd=a.budget, price_in=a.price_in, price_out=a.price_out, price_cached=a.price_cached)
        try: M.run(llm, its, path)
        except (QuotaStop, ServiceBusy, BudgetStop) as e: print("STOPPED:", e, "\nRun the same command again later.")
        return
    rows = json.load(open(path))["rows"]; s = M.summarize(its, rows)
    res = dict(summary=s, rows=rows, items=[{k: v for k, v in i.items() if k != "code"} for i in its])
    dst = os.path.join(ROOT, "docs", "results", "ptrgate_model.json"); json.dump(res, open(dst, "w"))
    print(json.dumps(s, indent=1)); print("written", dst)


if __name__ == "__main__":
    main()
