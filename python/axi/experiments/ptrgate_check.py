"""Free check of the pointer gate on the Juliet C/C++ test cases (CWE-416 and CWE-415). No model, no cost.

    python -m axi.experiments.ptrgate_check TESTCASES_DIR [dev|held|flow|multi|rest|all] [--out FILE]

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
    if name.endswith("_good") or not re.search(r"good", name) or re.search(r"(Sink|Source)$", name):
        return None
    return 0


FLOW = r"_(21|3[1-4]|4[1-5])\.(c|cpp)$"
MULTI = r"^(.*_(5[1-4]|6[1-8]))([a-e])\.(c|cpp)$"
REST = r"^(.*_(7[2-4]|8[1-4]))(a|b|_bad|_goodB2G|_goodG2B)\.(c|cpp)$"


def groups(root, pattern=None):
    """{base: [files]} for the multi-file variants 51 to 54 and 61 to 68 (or the pattern given)."""
    out = {}
    for cwe in ("CWE416_Use_After_Free", "CWE415_Double_Free"):
        for f in sorted(glob.glob(os.path.join(root, cwe, "**", "*"), recursive=True)):
            m = re.match(pattern or MULTI, f)
            if m:
                out.setdefault(m.group(1), []).append(f)
    return out


def files(root, part):
    out = []
    if part == "flow":
        for cwe in ("CWE416_Use_After_Free", "CWE415_Double_Free"):
            out += [f for f in sorted(glob.glob(os.path.join(root, cwe, "**", "*"), recursive=True)) if re.search(FLOW, f)]
        return out
    for cwe in ("CWE416_Use_After_Free", "CWE415_Double_Free"):
        for f in sorted(glob.glob(os.path.join(root, cwe, "**", "*"), recursive=True)):
            if re.search(r"_(0\d|1[0-8])\.(c|cpp)$", f):
                dev = any(d in os.path.basename(f) for d in DEV)
                if part == "all" or (part == "dev") == dev:
                    out.append(f)
    return out


def units(root, part):
    """(variant, cwe, source text) for every unit of the part."""
    if part in ("multi", "rest"):
        for base, fs in sorted(groups(root, REST if part == "rest" else None).items()):
            yield (re.search(r"_(\d\d)$", base).group(1), os.path.basename(base)[:6],
                   "\n".join(open(f, errors="replace").read() for f in fs))
        return
    for f in files(root, part):
        yield re.search(r"_(\d\d)\.", f).group(1), os.path.basename(f)[:6], open(f, errors="replace").read()


def run(root, part):
    import collections
    new = lambda: dict(TP=0, FN=0, FP=0, TN=0, unsup=0)
    tallies = collections.defaultdict(new)
    misses = []
    for var, cwe, src in units(root, part):
        for name, (verdict, info) in ptrgate.analyze_source(src).items():
            y = label(name)
            if y is None:
                continue
            if verdict == "unsupported":
                k = "unsup"
            else:
                pred = 1 if verdict == "vulnerable" else 0
                k = ("TP" if y else "FP") if pred else ("FN" if y else "TN")
            for key in ("all", "v" + var, cwe):
                tallies[key][k] += 1
            if k in ("FN", "FP", "unsup"):
                misses.append([var, cwe, name, k])
    t = tallies["all"]
    n = t["TP"] + t["FN"] + t["FP"] + t["TN"]
    t["analysed"] = n
    t["accuracy"] = round((t["TP"] + t["TN"]) / n, 4) if n else None
    t["recall"] = round(t["TP"] / (t["TP"] + t["FN"]), 4) if t["TP"] + t["FN"] else None
    t["false_alarm_rate"] = round(t["FP"] / (t["FP"] + t["TN"]), 4) if t["FP"] + t["TN"] else None
    return {"part": part, "tally": t, "by_variant_and_cwe": {k: v for k, v in sorted(tallies.items()) if k != "all"},
            "errors_and_unsupported": misses[:200]}


if __name__ == "__main__":
    root = sys.argv[1]
    part = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "all"
    r = run(root, part)
    if "--out" in sys.argv:
        json.dump(r, open(sys.argv[sys.argv.index("--out") + 1], "w"), indent=1)
    print(json.dumps({k: r[k] for k in ("part", "tally", "by_variant_and_cwe")}, indent=0))
