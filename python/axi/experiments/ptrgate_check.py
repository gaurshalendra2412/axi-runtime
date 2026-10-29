"""Free check of the pointer gate on the Juliet C/C++ test cases (CWE-416 and CWE-415). No model, no cost.

    python -m axi.experiments.ptrgate_check TESTCASES_DIR [dev|held|all] [--out FILE]

Unit: one function. `*_bad` is vulnerable; `goodG2B`, `goodB2G`, ... are clean; the `*_good`
wrapper is left out. Comments are removed before the gate sees the code, so the words
"FLAW" and "FIX" and the function names carry no hint. Only the control-flow variants 01 to 18 (`_NN.c`,
`_NN.cpp`; one file each) are used. `dev` is the families the gate was written on; `held` is every other family.
"""
import glob
import json
import os
import re
import sys

from axi.engine import ptrgate

DEV = ("malloc_free_char_", "malloc_free_int_")


def label(name):
    if name.endswith("_bad") or name == "bad":
        return 1
    if name.endswith("_good") or not re.search(r"good", name):
        return None
    return 0


def files(root, part):
    out = []
    for cwe in ("CWE416_Use_After_Free", "CWE415_Double_Free"):
        for f in sorted(glob.glob(os.path.join(root, cwe, "**", "*"), recursive=True)):
            if re.search(r"_(0\d|1[0-8])\.(c|cpp)$", f):
                dev = any(d in os.path.basename(f) for d in DEV)
                if part == "all" or (part == "dev") == dev:
                    out.append(f)
    return out


def run(root, part):
    rows, tally = [], {"TP": 0, "FN": 0, "FP": 0, "TN": 0, "unsupported": 0}
    by_variant = {}
    for f in files(root, part):
        src = open(f, errors="replace").read()
        var = re.search(r"_(\d\d)\.", f).group(1)
        for name, (verdict, info) in ptrgate.analyze_source(src).items():
            y = label(name)
            if y is None:
                continue
            if verdict == "unsupported":
                tally["unsupported"] += 1
                by_variant.setdefault(var, [0, 0])[1] += 1
                rows.append([os.path.basename(f), name, y, "unsupported", info])
                continue
            pred = 1 if verdict == "vulnerable" else 0
            k = ("TP" if y else "FP") if pred else ("FN" if y else "TN")
            tally[k] += 1
            by_variant.setdefault(var, [0, 0])[0] += 1
            if (pred != y):
                rows.append([os.path.basename(f), name, y, verdict, str(info)[:200]])
    t = tally
    n = t["TP"] + t["FN"] + t["FP"] + t["TN"]
    t["analysed"] = n
    t["accuracy"] = round((t["TP"] + t["TN"]) / n, 4) if n else None
    t["recall"] = round(t["TP"] / (t["TP"] + t["FN"]), 4) if t["TP"] + t["FN"] else None
    t["false_alarm_rate"] = round(t["FP"] / (t["FP"] + t["TN"]), 4) if t["FP"] + t["TN"] else None
    return {"part": part, "files": len(files(root, part)), "tally": t, "by_variant_[analysed,unsupported]": by_variant,
            "errors_and_unsupported": rows[:60]}


if __name__ == "__main__":
    root = sys.argv[1]
    part = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "all"
    r = run(root, part)
    if "--out" in sys.argv:
        json.dump(r, open(sys.argv[sys.argv.index("--out") + 1], "w"), indent=1)
    print(json.dumps({k: r[k] for k in ("part", "files", "tally", "by_variant_[analysed,unsupported]")}, indent=1))
    for row in r["errors_and_unsupported"][:25]:
        print(row)
